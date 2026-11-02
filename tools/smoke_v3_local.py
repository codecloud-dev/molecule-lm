"""molecule v3.0 本地端到端冒烟（不改动生产代码语义）。

做三件事：
  1) 走 train_v3.main() 的完整 CLI 路径（课程采样器 / 原子存档 / 时钟增量记账），
     但把 warmup 缩短到 20 步，让有限步数内能观察到 loss 真实下降（生产 warmup=2000 是给 10 天窗口用的）。
  2) 从训练出的 checkpoint 生成文本。
  3) 体检生成结果：无 U+FFFD（诡异文字）、无控制/特殊 token、重复度统计。

用法： python3.11 tools/smoke_v3_local.py [--steps 60] [--seq_len 128]
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    import torch

    import train_v3
    from tokenizer_bpe import ByteBPETokenizer

    # 仅缩短 warmup，让短程冒烟能看到收敛趋势（不影响生产默认）
    # 注意：lr_at 的 peak/warmup 是【定义时绑定的默认参数】，改模块属性无效，
    # 必须改写 __defaults__（(peak, warmup) 顺序）。
    train_v3.PEAK_LR = args.lr
    train_v3.lr_at.__defaults__ = (args.lr, 20)

    # 清掉上一轮冒烟产物，保证干净随机初始化
    if os.path.exists(train_v3.CKPT):
        os.remove(train_v3.CKPT)
    if os.path.exists(train_v3.CLOCK):
        os.remove(train_v3.CLOCK)

    sys.argv = [
        "train_v3.py",
        "--seq_len", str(args.seq_len),
        "--batch", str(args.batch),
        "--max_steps", str(args.steps),
        "--lr", str(args.lr),
        "--no_download",
        "--save_every", "1000",
    ]
    print("=== [1/3] 训练（真实 CLI 路径，warmup 缩短至 20）===", flush=True)
    train_v3.main()

    # ---- 生成 + 体检 ----
    print("\n=== [2/3] 生成 + 体检 ===", flush=True)
    import generate_v3 as G
    from model_v3 import MoleculeV3

    tk = ByteBPETokenizer.load()
    st = torch.load(train_v3.CKPT, map_location="cpu", weights_only=False)
    model = MoleculeV3(vocab_size=8192)
    model.load_state_dict(st["model"], strict=False)
    model.eval()
    print("checkpoint step=%s loss=%.3f" % (st.get("step"), st.get("loss", float("nan"))), flush=True)

    prompts = [
        "请介绍一下你自己。",
        "中国的首都是哪里？",
        "写一段代码计算斐波那契数列。",
    ]
    report = []
    for p in prompts:
        out = G.generate(model, tk, p, max_new=120, temperature=0.8, top_p=0.92)
        ids = tk.encode(out)
        n_fffd = out.count("�")
        n_ctl = sum(1 for i in ids if 256 <= i < 286)
        # 重复度：最长的连续重复片段（字符级）
        sents = [x for x in re.split(r"[。！？，、\n]", out) if len(x) >= 4]
        max_rep = max((len(s) for s in sents), default=0)
        chars = [c for c in out if c.strip()]
        uniq_ratio = len(set(chars)) / max(len(chars), 1)
        print("\nPROMPT: %s" % p)
        print("OUTPUT: %s" % out.replace("\n", "\\n"))
        print("  U+FFFD=%d  控制/特殊token=%d  最长重复片段=%d字  字符唯一率=%.3f"
              % (n_fffd, n_ctl, max_rep, uniq_ratio))
        report.append({"prompt": p, "output": out, "ufffd": n_fffd,
                       "ctl": n_ctl, "max_rep": max_rep, "uniq_ratio": uniq_ratio})

    print("\n=== [3/3] 汇总 ===", flush=True)
    ok = all(r["ufffd"] == 0 and r["ctl"] == 0 for r in report)
    print("无 U+FFFD 且无控制 token: %s" % ("PASS ✓" if ok else "FAIL ✗"))
    with open("/tmp/smoke_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("详细报告 -> /tmp/smoke_report.json")


if __name__ == "__main__":
    main()
