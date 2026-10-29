"""molecule v3 抗取消持久化：把 checkpoint + 时钟 + 语料推到滚动发布 v3-checkpoint。

为什么需要它（设计复盘 §鲁棒性）：
  CI 平台会在训练中取消 run（cancel_actor=null，非人为），而 workflow 的
  `if: always()` artifact 上传在被取消时不会执行 → checkpoint / 10 天时钟全部丢失、
  下一轮只能从零开始、10 天永远攒不够。

  解法：训练步骤在【每次周期存档】与【优雅退出(SIGTERM)】时都调用本脚本，
  把最新 checkpoint + 时钟 + 语料推到仓库级「滚动发布 v3-checkpoint」
  （prerelease，--clobber 覆盖同名资产，不会无限堆积）。滚动发布是仓库级持久存储，
  不受单次 run 被取消影响；下一轮 Resume 直接从它拉取，链自愈。

  双保险：本地周期存档 + 远端滚动发布，最坏情况只丢「上一次上传之后」的训练量
  （由 --save_every 控制，默认 10 分钟），10 天时钟按实际训练分钟准确累加。

仅在 CI（GITHUB_ACTIONS=true 且有 GITHUB_TOKEN）时生效；本地冒烟不推送。
"""

import glob
import os
import subprocess
import sys

RELEASE = "v3-checkpoint"
LOCK = "checkpoints/.persist.lock"


def _gh(*args, timeout=60, capture=True):
    return subprocess.run(
        ["gh", *args],
        capture_output=capture,
        timeout=timeout,
        check=False,
    )


def main():
    # 仅在 CI 且有 token 时推送；本地冒烟跳过，避免误推。
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    if not os.environ.get("GITHUB_TOKEN"):
        return

    # 避免并发上传互相 clobber：简单的文件锁，上一个还在传就跳过本轮。
    if os.path.exists(LOCK):
        return
    try:
        os.makedirs("checkpoints", exist_ok=True)
        open(LOCK, "w").close()
    except Exception:
        return

    try:
        # 解析仓库（带 fallback）。
        try:
            repo = subprocess.check_output(
                ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
                timeout=30,
            ).decode().strip()
        except Exception:
            return
        if not repo:
            return

        # 确保滚动发布存在（prerelease，不出现在正式发布列表显著位置）。
        if _gh("release", "view", RELEASE, timeout=30).returncode != 0:
            _gh("release", "create", RELEASE,
                "--prerelease",
                "-t", "v3 滚动检查点（自动，勿手动改）",
                "-n", "自动滚动检查点：checkpoint + 时钟 + 语料，供续训 resume。勿手动修改或删除。",
                timeout=60)

        # 收集要上传的文件（--clobber 覆盖同名，不堆积）。
        files = []
        for f in ("checkpoints/v3.pt", "data/v3_clock.json", "data/self_bpe_8192.json"):
            if os.path.exists(f):
                files.append(f)
        for f in glob.glob("data/v3_*.bin"):
            files.append(f)
        if not files:
            return

        # 推送到滚动发布（大文件，给足超时；失败不阻塞训练，下一轮再补）。
        _gh("release", "upload", RELEASE, "--clobber", *files, timeout=900)
        print("[persist] 已推 %d 个文件到滚动发布 %s" % (len(files), RELEASE), flush=True)
    except Exception as e:
        print("[persist] 异常跳过：%s" % e, flush=True)
    finally:
        try:
            os.remove(LOCK)
        except Exception:
            pass


if __name__ == "__main__":
    main()
