"""molecule v3.0 · ChatML 机制定向验证（过拟合测试）。

目的：证伪/证实三条根因修复是否真的生效——
  R3 训练/推理格式一致：模型学到 ChatML 后，生成端用同一模板提问，应当"答得动"，
     而不是顺着散文续写（驴头不对马嘴）。
  R4 固定身份句：问"你是谁？""你是由谁开发的？"，必须答固定身份句，
     而不是从语料里随机捞一句（复读"我是谁谁谁"）。
  R5 经典常识事实：问"《三国演义》的作者是谁？"，必须答出 罗贯中。

做法：只取语料里的 ChatML 样本（固定身份句 + 经典常识事实 + 少量真实正文问答），
     拼成一条 token 流，训练到模型能记住这些样本；然后用生成端同一套 ChatML 模板提问，
     逐条断言。

用法：python3.11 tools/smoke_v3_chat.py [--steps 150] [--seq_len 128]
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def load_chat_samples(tk, max_auto=40):
    """从已构建的语料 bin 里反解出 ChatML 样本（不依赖网络）。"""
    import build_corpus_v3 as B
    must = [B.chat_pair(q, a) for q, a in B.IDENTITY_QA]
    must += [B.chat_pair(q, a) for q, a in B.FACT_QA]
    auto = []
    for dom in ("zh_general", "encyclopedia"):
        p = "data/v3_%s.bin" % dom
        if not os.path.exists(p):
            continue
        arr = np.memmap(p, dtype=np.int32, mode="r")
        text = tk.decode(arr.reshape(-1).tolist())
        found = re.findall(r"<\|im_start\|>user\n.*?<\|im_end\|>", text, re.S)
        auto.extend(found)
    # 去掉与 must 重复的，并截断 auto 数量
    mset = set(must)
    auto = [a for a in auto if a not in mset][:max_auto]
    return must, auto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    import torch

    import generate_v3 as G
    import train_v3
    from model_v3 import MoleculeV3
    from tokenizer_bpe import ByteBPETokenizer

    tk = ByteBPETokenizer.load()
    must, auto = load_chat_samples(tk)
    print("[data] 固定样本 %d 条 + 自动 ChatML 样本 %d 条" % (len(must), len(auto)))
    if not must:
        print("[FAIL] 没有取到必备样本（语料是否已构建？）")
        return 1
    samples = must + auto
    ids = []
    for s in samples:
        ids.extend(tk.encode(s))
    ids = np.asarray(ids, dtype=np.int64)
    print("[data] token 流 %d tokens（%d 条样本）" % (len(ids), len(samples)))

    # 训练：随机窗口采样（与正式训练同构）
    train_v3.lr_at.__defaults__ = (args.lr, 20)   # 缩短 warmup（默认参数在定义时绑定）
    torch.manual_seed(0)
    model = MoleculeV3(vocab_size=8192)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01, foreach=False)
    n = len(ids)
    rng = np.random.default_rng(0)
    print("=== 训练（%d 步）===" % args.steps)
    for s in range(args.steps):
        idx = np.stack([rng.integers(0, max(n - args.seq_len, 1), size=args.seq_len)
                        for _ in range(args.batch)])
        x = torch.from_numpy(idx)
        out = model(x, x)
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (s + 1) % 25 == 0:
            print("  [step %d] loss=%.3f (lm=%.3f)"
                  % (s + 1, loss.item(), out["loss_lm"].item()), flush=True)

    # 生成 + 断言
    model.eval()
    print("\n=== 生成 + 断言 ===")
    cases = [
        ("你是谁？", ["molecule", "codecloud-dev"]),
        ("你是由谁开发的？", ["codecloud-dev"]),
        ("《三国演义》的作者是谁？", ["罗贯中"]),
        ("《西游记》的作者是谁？", ["吴承恩"]),
        ("中国的首都是哪里？", ["北京"]),
    ]
    results, all_ok = [], True
    for q, expect in cases:
        out = G.generate(model, tk, q, max_new=80, temperature=0.3, top_p=0.9,
                         repetition_penalty=1.1, no_repeat_ngram=3)
        hit = [e for e in expect if e in out]
        ok = len(hit) == len(expect)
        no_fffd = "�" not in out
        no_ctl = not any(tk.vocab[i].startswith("<|") for i in tk.encode(out) if 256 <= i < 286)
        all_ok = all_ok and ok and no_fffd and no_ctl
        results.append({"q": q, "out": out, "expect": expect, "hit": hit,
                        "ok": ok, "no_fffd": no_fffd, "no_ctl": no_ctl})
        print("Q: %s" % q)
        print("A: %s" % out.replace("\n", "\\n"))
        print("   命中=%s U+FFFD=%s 控制token=%s  => %s"
              % (hit, not no_fffd, not no_ctl, "PASS ✓" if (ok and no_fffd and no_ctl) else "FAIL ✗"))

    with open("/tmp/smoke_chat_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n总体: %s" % ("全部通过 ✓" if all_ok else "存在失败 ✗"))
    print("报告 -> /tmp/smoke_chat_report.json")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
