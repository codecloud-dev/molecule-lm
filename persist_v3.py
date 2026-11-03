"""molecule v3 抗取消持久化：把 checkpoint + 时钟 + 语料 + 监测日志 推到滚动发布 v3-checkpoint。

为什么需要它（设计复盘 §鲁棒性）：
  CI 平台会在训练中取消 run（cancel_actor=null，非人为），而 workflow 的
  `if: always()` artifact 上传在被取消时不会执行 → checkpoint / 20 天时钟全部丢失、
  下一轮只能从零开始、20 天永远攒不够。

  解法：训练步骤在【每次周期存档】与【优雅退出(SIGTERM)】时都调用本脚本，
  把最新 checkpoint + 时钟 + 语料推到仓库级「滚动发布 v3-checkpoint」
  （覆盖同名资产，不会无限堆积）。滚动发布是仓库级持久存储，不受单次 run 被取消影响；
  下一轮 Resume 直接从它拉取，链自愈。

模式（train_v3.persist 传入）：
  full  → 上传 checkpoint + 时钟 +（语料版本变化时才传）语料 bin/BPE + 监测日志；
  logs  → 仅上传监测日志（v3_train_log.jsonl / v3_eval_probe.jsonl /
          v3_dashboard.md / v3_checks.log / 时钟），轻量、可高频调用。

实现要点（2026-11-04 修正）：
  全部走 `curl + GITHUB_TOKEN`（Actions 真实注入、永远在环境里的 token），
  不再依赖 `gh` CLI——此前 `gh` 在 CI 中对 release 的读写整体失效，导致
  滚动发布从不更新、resume 也拉不到检查点、每轮从头随机初始化。
  任何失败都写 data/v3_persist.log（随 artifact 上传），便于事后诊断。

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
    try:
        os.makedirs("data", exist_ok=True)
        with open(PERSIST_LOG, "a", encoding="utf-8") as f:
            f.write("[%s] %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg))
    except Exception:
        pass
    print("[persist] " + msg, flush=True)


def _curl(method, url, token, extra_headers=None, data=None, timeout=900, binary_file=None):
    """用 curl + GITHUB_TOKEN 打 GitHub API，绕过失效的 gh CLI。"""
    cmd = ["curl", "-sS", "-X", method, url,
           "-H", "Authorization: Bearer %s" % token,
           "-H", "Accept: application/vnd.github+json"]
    if extra_headers:
        for h in extra_headers:
            cmd += ["-H", h]
    if binary_file is not None:
        cmd += ["-H", "Content-Type: application/octet-stream",
                "--data-binary", "@%s" % binary_file]
    elif data is not None:
        cmd += ["--data-binary", data]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "curl timeout"
    except Exception as e:  # noqa
        return None, "curl error: %s" % e
    return r.stdout.decode(errors="replace"), r.returncode


def corpus_needs_upload(token, repo):
    """仅当滚动发布缺失语料或【本地配方戳与远端不一致】时才上传语料 bin/BPE。"""
    if not os.path.exists(STAMP):
        return True
    local_stamp = open(STAMP, encoding="utf-8").read().strip()
    rc, out = _curl("GET",
                    "https://api.github.com/repos/%s/releases/tags/%s" % (repo, RELEASE),
                    token, timeout=30)
    if rc is None or rc.strip() == "" or "404" in (out or "")[:4]:
        return True
    try:
        rel = json.loads(out)
    except Exception:
        return True
    for a in rel.get("assets", []):
        if a["name"] == STAMP:
            # 已存在配方戳资产，下载比对
            rc2, out2 = _curl("GET", a["url"], token, timeout=30)
            if rc2 and local_stamp in (rc2 or ""):
                return False
            return True
    return True


def upload_assets(files, token, repo):
    """把文件推到滚动发布（覆盖同名资产）。返回 (ok, detail)。"""
    # 取 release（不存在则建）
    rc, out = _curl("GET",
                    "https://api.github.com/repos/%s/releases/tags/%s" % (repo, RELEASE),
                    token, timeout=30)
    if rc is None:
        return False, "get release: %s" % out
    try:
        rel = json.loads(rc)
        if isinstance(rel, dict) and rel.get("message") == "Not Found":
            raise ValueError("release not found")
    except Exception:
        # 建 release
        body = {"tag_name": RELEASE, "name": "v3 滚动检查点（自动，勿手动改）",
                "body": "自动滚动检查点：checkpoint + 时钟 + 语料 + 监测日志，供续训 resume。勿手动修改或删除。",
                "prerelease": True}
        rc2, out2 = _curl("POST",
                          "https://api.github.com/repos/%s/releases" % repo,
                          token, data=json.dumps(body), timeout=60)
        if rc2 is None:
            return False, "create release: %s" % out2
        try:
            rel = json.loads(rc2)
        except Exception:
            return False, "create release parse fail: %s" % (rc2 or "")[:200]
    rel_id = rel.get("id")
    upload_url = rel.get("upload_url", "").split("{")[0]
    if not rel_id or not upload_url:
        return False, "release 缺少 id/upload_url"
    existing = {a["name"]: a["id"] for a in rel.get("assets", [])}

    for f in files:
        name = os.path.basename(f)
        if name in existing:
            _curl("DELETE",
                  "https://api.github.com/repos/%s/releases/assets/%s" % (repo, existing[name]),
                  token, timeout=30)
        rc3, out3 = _curl("POST",
                          "%s?name=%s" % (upload_url, urllib.parse.quote(name)),
                          token, timeout=900, binary_file=f)
        if rc3 is None:
            return False, "upload %s: %s" % (name, out3)
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

    # 等待锁释放（最多 ~120s）：避免「异步 logs 持久化」持锁时，「同步 full 持久化（checkpoint）」
    # 被直接 return 跳过、导致检查点没推上去。
    for _ in range(120):
        if not os.path.exists(LOCK):
            break
        time.sleep(1)
    else:
        plog("等待锁超时(120s)，放弃本轮（下一轮存档会补）")
        return
    try:
        os.makedirs("checkpoints", exist_ok=True)
        open(LOCK, "w").close()
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
