"""molecule v3 抗取消持久化：把 checkpoint + 时钟 + 语料 + 监测日志 推到滚动发布 v3-checkpoint。

设计复盘（鲁棒性）：
  CI 平台会在训练中取消 run（cancel_actor=null，非人为），若只靠 workflow 末尾的
  `if: always()` artifact 上传，被取消时不会执行 → checkpoint / 20 天时钟全部丢失。
  解法：训练在【每次周期存档】与【优雅退出(SIGTERM)】都调用本脚本，把最新状态推到
  仓库级滚动发布 v3-checkpoint（覆盖同名资产），下一轮 Resume 直接拉取，链自愈。

2026-11-16 关键修复（此前每轮都从头、20 天时钟永不累加的根因链）：
  1. _curl 返回值是 (stdout, returncode)。旧代码用 `if stdout is None` 判断上传失败，
     但 curl 失败时 stdout 是错误 JSON（非空）→ 永远误判"已推成功"，掩盖了真实失败。
     现统一改为 **检查 returncode**，并让 curl 带 -f（HTTP 4xx/5xx 视为失败）。
  2. 下载 release asset 的原始字节必须用 Accept: application/octet-stream；
     用 json accept 会拿到资产元数据 JSON（{ 开头）→ torch.load UnpicklingError。
  3. 上传失败现在会写 data/v3_persist.log（随 artifact 上传），便于事后诊断。

模式：
  full → checkpoint + 时钟 +（语料版本变化时）语料 bin/BPE + 监测日志；
  logs → 仅监测日志 + 时钟，轻量、可高频调用。

仅在 CI（GITHUB_ACTIONS=true 且有 token）时生效；本地冒烟不推送。
"""

import glob
import json
import os
import subprocess
import sys
import time
import urllib.parse

RELEASE = "v3-checkpoint"
LOCK = "checkpoints/.persist.lock"
PERSIST_LOG = "data/v3_persist.log"

CKPT = "checkpoints/v3.pt"
CLOCK = "data/v3_clock.json"
BPE = "data/self_bpe_8192.json"
STAMP = "data/v3_corpus_stamp.txt"
LOG_FILES = ("data/v3_train_log.jsonl", "data/v3_eval_probe.jsonl",
             "data/v3_dashboard.md", "data/v3_checks.log")


def plog(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg)
    try:
        os.makedirs("data", exist_ok=True)
        with open(PERSIST_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print("[persist] " + msg, flush=True)


def _curl(method, url, token, extra_headers=None, data=None, timeout=900, binary_file=None):
    """用 curl + GITHUB_TOKEN 打 GitHub API，绕过失效的 gh CLI。

    返回 (stdout_text, returncode_int)。
    curl 带 -f：HTTP 4xx/5xx → returncode != 0（不会再被误判为成功）。
    """
    cmd = ["curl", "-sS", "-f", "-X", method, url,
           "-H", "Authorization: Bearer %s" % token,
           "-H", "Accept: application/vnd.github+json"]
    if extra_headers:
        for h in extra_headers:
            cmd += ["-H", h]
    if binary_file is not None:
        # 上传 release asset 原始字节：Content-Type 必须 octet-stream
        cmd += ["-H", "Content-Type: application/octet-stream",
                "--data-binary", "@%s" % binary_file]
    elif data is not None:
        cmd += ["--data-binary", data]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return r.stdout.decode(errors="replace"), r.returncode
    except subprocess.TimeoutExpired:
        return None, 124
    except Exception as e:  # noqa
        return None, -1


def corpus_needs_upload(token, repo):
    """仅当滚动发布缺失语料或【本地配方戳与远端不一致】时才上传语料 bin/BPE。"""
    if not os.path.exists(STAMP):
        return True
    local_stamp = open(STAMP, encoding="utf-8").read().strip()
    out, code = _curl("GET",
                      "https://api.github.com/repos/%s/releases/tags/%s" % (repo, RELEASE),
                      token, timeout=30)
    if code != 0 or not out:
        return True
    try:
        rel = json.loads(out)
    except Exception:
        return True
    for a in rel.get("assets", []):
        if a["name"] == STAMP:
            # 下载配方戳原始字节（必须 octet-stream，否则拿到的是元数据 JSON）
            out2, code2 = _curl("GET", a["url"], token, timeout=30,
                                extra_headers=["Accept: application/octet-stream"])
            if code2 == 0 and out2 and local_stamp in out2:
                return False
            return True
    return True


def upload_assets(files, token, repo):
    """把文件推到滚动发布（覆盖同名资产）。返回 (ok, detail)。"""
    # 取 release（不存在则建）
    out, code = _curl("GET",
                      "https://api.github.com/repos/%s/releases/tags/%s" % (repo, RELEASE),
                      token, timeout=30)
    rel = None
    if code == 0 and out:
        try:
            cand = json.loads(out)
            if isinstance(cand, dict) and cand.get("id"):
                rel = cand
        except Exception:
            pass
    if rel is None:
        # 建 release
        body = {"tag_name": RELEASE, "name": "v3 滚动检查点（自动，勿手动改）",
                "body": "自动滚动检查点：checkpoint + 时钟 + 语料 + 监测日志，供续训 resume。勿手动修改或删除。",
                "prerelease": True}
        out2, code2 = _curl("POST",
                            "https://api.github.com/repos/%s/releases" % repo,
                            token, data=json.dumps(body), timeout=60)
        if code2 != 0 or not out2:
            return False, "create release failed(code=%s): %s" % (code2, (out2 or "")[:200])
        try:
            rel = json.loads(out2)
        except Exception:
            return False, "create release parse fail: %s" % (out2 or "")[:200]
    rel_id = rel.get("id")
    upload_url = rel.get("upload_url", "").split("{")[0]
    if not rel_id or not upload_url:
        return False, "release 缺少 id/upload_url"
    existing = {a["name"]: a["id"] for a in rel.get("assets", [])}

    failed = []
    for f in files:
        name = os.path.basename(f)
        if name in existing:
            _curl("DELETE",
                  "https://api.github.com/repos/%s/releases/assets/%s" % (repo, existing[name]),
                  token, timeout=60)
        out3, code3 = _curl("POST",
                            "%s?name=%s" % (upload_url, urllib.parse.quote(name)),
                            token, timeout=900, binary_file=f)
        if code3 != 0:
            failed.append("%s(code=%s,%s)" % (name, code3, (out3 or "")[:80]))
    if failed:
        return False, "上传失败: " + "; ".join(failed)
    return True, "ok"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else (os.environ.get("V3_PERSIST_MODE") or "full")
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        return
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        return

    # 等锁释放（最多 ~120s）：避免异步 logs 持锁时同步 full 被跳过。
    for _ in range(120):
        if not os.path.exists(LOCK):
            break
        time.sleep(1)
    else:
        plog("等待锁超时(120s)，放弃本轮（下一轮存档会补）")
        return
    try:
        os.makedirs("checkpoints", exist_ok=True)
        # 原子建锁：仅当锁不存在时才创建，避免并发进程互相覆盖锁。
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        # 别的进程刚建了锁，再等一轮
        for _ in range(120):
            if not os.path.exists(LOCK):
                break
            time.sleep(1)
    except Exception:
        return

    try:
        files = []
        if mode == "logs":
            for f in LOG_FILES + (CLOCK,):
                if os.path.exists(f):
                    files.append(f)
        else:  # full
            for f in (CKPT, CLOCK):
                if os.path.exists(f):
                    files.append(f)
            if corpus_needs_upload(token, repo):
                for f in glob.glob("data/v3_*.bin"):
                    files.append(f)
                for f in (BPE, STAMP):
                    if os.path.exists(f):
                        files.append(f)
            for f in LOG_FILES:
                if os.path.exists(f):
                    files.append(f)

        if not files:
            plog("无文件可推（mode=%s）" % mode)
            return

        ok, detail = upload_assets(files, token, repo)
        if ok:
            plog("已推 %d 个文件到滚动发布 %s（mode=%s）" % (len(files), RELEASE, mode))
        else:
            plog("上传失败（mode=%s）：%s" % (mode, detail))
    except Exception as e:  # noqa
        plog("异常跳过：%s" % e)
    finally:
        try:
            os.remove(LOCK)
        except Exception:
            pass


if __name__ == "__main__":
    main()
