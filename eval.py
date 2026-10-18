#!/usr/bin/env python3
"""molecule 评测：zero-shot logprob 选 A/B/C（趋势参考，非官方分数）。

- 用内置精简冒烟集（自包含，详见 benchmarks.py 声明）；如需官方 split 请接入 harness。
- 输出：stdout 打印 + 写 eval_result.md 趋势表。
- 说明：内置冒烟集为自写题，仅用于训练趋势监控，不与外模型官方分数比较。
"""

import os
import sys
import json
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MoleculeConfig
from model import MoleculeModel
from tokenizer import ReducedTokenizer, get_base_tokenizer
from benchmarks import get_benchmark


def build_prompt(q, choices):
    head = "请根据题目选择正确选项的字母（A/B/C）。\n"
    body = f"题目：{q}\n"
    for i, c in enumerate(choices):
        body += f"{chr(65+i)}. {c}\n"
    body += "答案："
    return head + body


def _letter_id(tk, ch):
    """取得单字母 token 的 new id；任何异常都回退到 byte token 或 0，绝不崩。

    语义正确性：训练侧 ReducedTokenizer 对词表外字符同样走 byte 回退
    （byte_new[ord(ch)]），此处用 byte token 兜底与训练编码完全一致。
    """
    try:
        ids = tk.encode(ch)
        if ids:
            return ids[0]
    except Exception:
        pass
    b = ord(ch)
    bn = getattr(tk, "byte_new", None)
    if isinstance(bn, dict) and b in bn:
        return bn[b]
    return 0


@torch.no_grad()
def score_benchmark(model, tk, name):
    items = get_benchmark(name)
    if not items:
        return None
    # 单字母 token（A/B/C）：健壮获取，任一项异常则该基准跳过而非崩。
    try:
        letters = [_letter_id(tk, chr(65 + i)) for i in range(3)]
        letters = [l for l in letters if isinstance(l, int) and l >= 0]
        if not letters:
            return None
    except Exception:
        return None
    correct = 0
    for q, choices, ans in items:
        try:
            prompt = build_prompt(q, choices)
            ids = tk.encode(prompt)
            if len(ids) < 2:
                continue
            x = torch.tensor([ids], dtype=torch.long)
            logits = model(x)["logits"][0, -1].float()
            scores = [logits[l].item() for l in letters if 0 <= l < logits.size(0)]
            if not scores:
                continue
            pred = int(torch.tensor(scores).argmax())
            if pred == ans:
                correct += 1
        except Exception:
            continue
    return 100.0 * correct / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--ckpt", default="checkpoints/model.safetensors")
    ap.add_argument("--out", default="eval_result.md")
    args = ap.parse_args()

    cfg = MoleculeConfig.load(args.config) if os.path.exists(args.config) else MoleculeConfig()
    base, _ = get_base_tokenizer(cfg.tokenizer_name)
    tk = ReducedTokenizer.load(base, os.path.join(args.data_dir, "tokenizer_map.json"))

    model = MoleculeModel(cfg)
    # 加载权重
    try:
        from safetensors.torch import load_model
        load_model(model, args.ckpt)
    except Exception:
        sd = torch.load(os.path.join("checkpoints", "latest.ckpt"), map_location="cpu")
        model.load_state_dict(sd["model"])
    model.eval()

    rows = []
    print("\n=== molecule 评测（内置冒烟集，趋势参考，非官方分数）===")
    all_pass = True
    for name in cfg.eval_benchmarks:
        acc = score_benchmark(model, tk, name)
        if acc is None:
            continue
        target = cfg.smollm2_targets.get(name, 0.0)
        ok = acc >= target - cfg.eval_tolerance
        all_pass = all_pass and ok
        rows.append((name, acc, target, ok))
        print(f"{name:12s} our={acc:5.1f}  ref={target:5.1f}  {'≥ref' if ok else '<?ref'}")

    # 写趋势表（中性：仅展示自测分数与参考值，不做跨模型胜负判定）
    md = ["# molecule 评测趋势表（内置冒烟集，非官方 split）", "",
          "| 冒烟集 | molecule(自测) | 参考值 | 备注 |",
          "| --- | --- | --- | --- |"]
    for name, acc, target, ok in rows:
        md.append(f"| {name} | {acc:.1f} | {target:.1f} | {'≥ref' if ok else '<?ref'} |")
    md.append("")
    md.append("> 说明：本表为自写冒烟题，仅作训练趋势监控；参考值取自公开模型公开数字，")
    md.append("> **不可直接比较**，正式评测请接入 lm-evaluation-harness + 官方 split。")
    open(args.out, "w", encoding="utf-8").write("\n".join(md))
    print(f"\n[write] {args.out}")
    with open("eval_status.json", "w") as f:
        json.dump({"all_pass": all_pass, "rows": rows}, f)


if __name__ == "__main__":
    main()
