#!/usr/bin/env python3
"""molecule V0.1 评测：zero-shot logprob 选 A/B/C + 与 SmolLM2-135M 硬靶对比。

- 用内置精简基准（自包含）；如需官方 dev 全集，替换 benchmarks 即可。
- 输出：stdout 打印 + 写 eval_result.md 对比表。
- 达标：六项基准逐项超 SmolLM2(±0.5) 即视为“惊艳达标”。
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


@torch.no_grad()
def score_benchmark(model, tk, name):
    items = get_benchmark(name)
    if not items:
        return None
    letters = [tk.encode(chr(65 + i))[0] for i in range(3)]
    correct = 0
    for q, choices, ans in items:
        prompt = build_prompt(q, choices)
        ids = tk.encode(prompt)
        if len(ids) < 2:
            continue
        x = torch.tensor([ids], dtype=torch.long)
        logits = model(x)["logits"][0, -1].float()
        scores = [logits[l].item() for l in letters]
        pred = int(torch.tensor(scores).argmax())
        if pred == ans:
            correct += 1
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
    print("\n=== molecule V0.1 评测 vs SmolLM2-135M ===")
    all_pass = True
    for name in cfg.eval_benchmarks:
        acc = score_benchmark(model, tk, name)
        if acc is None:
            continue
        target = cfg.smollm2_targets.get(name, 0.0)
        ok = acc >= target - cfg.eval_tolerance
        all_pass = all_pass and ok
        rows.append((name, acc, target, ok))
        print(f"{name:12s} our={acc:5.1f}  smol={target:5.1f}  {'PASS' if ok else 'FAIL'}")

    # 写对比表
    md = ["# molecule V0.1 评测对比表", "",
          "| 基准 | molecule(我们) | SmolLM2-135M | 达标(±0.5) |",
          "| --- | --- | --- | --- |"]
    for name, acc, target, ok in rows:
        md.append(f"| {name} | {acc:.1f} | {target:.1f} | {'✅' if ok else '❌'} |")
    md.append("")
    md.append(f"**结论**：{'全部逐项超越 SmolLM2（惊艳达标）🎉' if all_pass else '尚未全部超越，继续续训/扩充数据'}")
    open(args.out, "w", encoding="utf-8").write("\n".join(md))
    print(f"\n[write] {args.out}")
    with open("eval_status.json", "w") as f:
        json.dump({"all_pass": all_pass, "rows": rows}, f)


if __name__ == "__main__":
    main()
