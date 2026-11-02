"""molecule v3 抗取消持久化：把 checkpoint + 时钟 + 语料 + 监测日志 推到滚动发布 v3-checkpoint。

为什么需要它（设计复盘 §鲁棒性）：
  CI 平台会在训练中取消 run（cancel_actor=null，非人为），而 workflow 的
  `if: always()` artifact 上传在被取消时不会执行 → checkpoint / 20 天时钟全部丢失、
  下一轮只能从零开始、20 天永远攒不够。

  解法：训练步骤在【每次周期存档】与【优雅退出(SIGTERM)】时都调用本脚本，
  把最新 checkpoint + 时钟 + 语料推到仓库级「滚动发布 v3-checkpoint」
  （prerelease，--clobber 覆盖同名资产，不会无限堆积）。滚动发布是仓库级持久存储，
  不受单次 run 被取消影响；下一轮 Resume 直接从它拉取，链自愈。

  双保险：本地周期存档 + 远端滚动发布，最坏情况只丢「上一次上传之后」的训练量
  （由 --save_every 控制，默认 30 分钟），20 天时钟按实际训练分钟准确累加。

模式（train_v3.persist 传入）：
  full  → 上传 checkpoint + 时钟 +（语料版本变化时才传）语料 bin/BPE + 监测日志；
  logs  → 仅上传监测日志（v3_train_log.jsonl / v3_eval_probe.jsonl /
          v3_dashboard.md / v3_checks.log / 时钟），轻量、可高频调用。

仅在 CI（GITHUB_ACTIONS=true 且有 token）时生效；本地冒烟不推送。
"""

import glob
import os
import subprocess
import sys

RELEASE = "v3-checkpoint"
LOCK = "checkpoints/.persist.lock"

CKPT = "checkpoints/v3.pt"
CLOCK = "data/v3_clock.json"
BPE = "data/self_bpe_8192.json"
STAMP = "data/v3_corpus_stamp.txt"
LOG_FILES = ("data/v3_train_log.jsonl", "data/v3_eval_probe.jsonl",
             "data/v3_dashboard.md", "data/v3_checks.log")


def _gh(*args, timeout=60, capture=True, env=None):
    return subprocess.run(
        ["gh", *args], capture_output=capture, timeout=timeout, check=False,
        env=env or GH_ENV,
    )


GH_ENV = dict(os.environ)


def corpus_needs_upload():
    """仅当滚动发布缺失语料或【本地配方戳与远端不一致】时才上传语料 bin/BPE。
    语料在 20 天训练期内基本不变，避免每轮重复上传 80MB+。"""
    if not os.path.exists(STAMP):
        return True
    local_stamp = open(STAMP, encoding="utf-8").read().strip()
    tmp = "/tmp/v3_rel_stamp.txt"
    try:
        r = _gh("release", "download", RELEASE, "-p", STAMP, "-D", "/tmp",
                timeout=30, capture=True)
        if r.returncode != 0:
            return True
        if os.path.exists(tmp):
            rel = open(tmp, encoding="utf-8").read().strip()
            return rel != local_stamp
        return True
    except Exception:
        return True


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else (os.environ.get("V3_PERSIST_MODE") or "full")
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        return
    gh_env = dict(os.environ)
    gh_env["GH_TOKEN"] = token
    GH_ENV.clear()
    GH_ENV.update(gh_env)

    if os.path.exists(LOCK):
        return
    try:
        os.makedirs("checkpoints", exist_ok=True)
        open(LOCK, "w").close()
    except Exception:
        return

    try:
        try:
            repo = subprocess.check_output(
                ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
                timeout=30, env=gh_env,
            ).decode().strip()
        except Exception:
            return
        if not repo:
            return

        if _gh("release", "view", RELEASE, timeout=30).returncode != 0:
            _gh("release", "create", RELEASE,
                "--prerelease",
                "-t", "v3 滚动检查点（自动，勿手动改）",
                "-n", "自动滚动检查点：checkpoint + 时钟 + 语料 + 监测日志，供续训 resume。勿手动修改或删除。",
                timeout=60)

        files = []
        if mode == "logs":
            for f in LOG_FILES + (CLOCK,):
                if os.path.exists(f):
                    files.append(f)
        else:  # full
            for f in (CKPT, CLOCK):
                if os.path.exists(f):
                    files.append(f)
            if corpus_needs_upload():
                for f in glob.glob("data/v3_*.bin"):
                    files.append(f)
                for f in (BPE, STAMP):
                    if os.path.exists(f):
                        files.append(f)
            # 监测日志始终随 full 一起上传（便宜且利于事后回看）
            for f in LOG_FILES:
                if os.path.exists(f):
                    files.append(f)

        if not files:
            return

        _gh("release", "upload", RELEASE, "--clobber", *files, timeout=900)
        print("[persist] 已推 %d 个文件到滚动发布 %s（mode=%s）" % (len(files), RELEASE, mode), flush=True)
    except Exception as e:
        print("[persist] 异常跳过：%s" % e, flush=True)
    finally:
        try:
            os.remove(LOCK)
        except Exception:
            pass


if __name__ == "__main__":
    main()
