#!/usr/bin/env python3
"""molecule V0.1 多老师分领域蒸馏数据生成。

- 5 位老师分领域 1对1：通用/推理/数学/代码/EQ，按 prompt 标签路由。
- 每位老师温度按 DESIGN.md 设置（数学/代码偏低保准确、通用/EQ 偏高保多样）。
- 生成结果存 data/teacher_<domain>.jsonl；训练时由 data.build_all_bins 并入对应 domain 的 .bin。
- 支持断点续跑（已存在且达标则跳过）、批量生成、设备选择、离线 dry-run 校验。

用法：
  python distill.py --dry_run --all                  # 离线校验教师映射/路由/提示，不下载模型
  python distill.py --domain math --n 2000           # 单个域生成
  python distill.py --all --n 4000 --device cuda     # 全部域（GPU）
  python distill.py --all --n 4000 --overwrite       # 强制重生成
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MoleculeConfig

# 各 domain 的引导提示（决定生成什么风格的语料）
SEEDS = {
    "general": "请生成一段通顺、知识正确的中文科普或日常文本，200字以内。",
    "math": "请出一道中文数学题，并给出逐步解答。",
    "code": "请写一段带中文注释的简洁代码，并说明用途。",
    "reason": "请给出一个需要推理的小问题，并展示推理过程与结论。",
    "eq": "请写一段温暖、有共情力的中文对话回复，体现倾听与陪伴。",
}


def build_prompt(domain, idx):
    return (
        f"<|im_start|>system\n{SEEDS[domain]}<|im_end|>\n"
        f"<|im_start|>user\n第{idx + 1}条<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def count_existing(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def gen(domain, n, cfg, out_path, device="cpu", max_new_tokens=256,
        batch_size=4, overwrite=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tcfg = cfg.teachers[domain]
    model_name = tcfg["model"]
    temp = tcfg["temp"]

    # 断点续跑：已达标则跳过
    have = count_existing(out_path)
    if not overwrite and have >= n:
        print(f"[distill] {domain}: 已有 {have}≥{n} 条，跳过（--overwrite 可强制重生成）")
        return
    start = 0 if overwrite else have

    print(f"[distill] loading teacher {model_name} (temp={temp} device={device})")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
    model.eval()

    mode = "w" if overwrite else "a"
    written = 0
    with open(out_path, mode, encoding="utf-8") as f:
        i = start
        while i < n:
            bsz = min(batch_size, n - i)
            prompts = [build_prompt(domain, j) for j in range(i, i + bsz)]
            enc = tok(prompts, return_tensors="pt", padding=True).to(device)
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temp, top_p=0.9,
            )
            for k in range(bsz):
                prompt_len = enc["input_ids"][k].shape[0]
                text = tok.decode(out[k][prompt_len:], skip_special_tokens=True)
                f.write(json.dumps({"domain": domain, "text": text}, ensure_ascii=False) + "\n")
                written += 1
            i += bsz
            if i % 50 < bsz or i == n:
                print(f"[distill] {domain} {i}/{n} (temp={temp})", flush=True)
    print(f"[distill] wrote {written} -> {out_path}")


def dry_run(cfg, domains):
    print("=== distill dry-run：教师映射 / 路由 / 提示校验（不下载模型）===")
    for d in domains:
        tcfg = cfg.teachers[d]
        print(f"  domain={d:8s} -> teacher={tcfg['model']:42s} temp={tcfg['temp']}")
        print(f"    seed_prompt: {SEEDS[d]}")
    print("=== 校验完成：以上即 5 位老师分领域 1对1 的蒸馏路由 ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--device", default="auto",
                    help="auto=有 cuda 用 cuda 否则 cpu；也可显式 cuda / cpu")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true",
                    help="离线校验教师映射/路由/提示，不下载模型、不生成")
    args = ap.parse_args()
    cfg = MoleculeConfig.load("config.json") if os.path.exists("config.json") else MoleculeConfig()
    os.makedirs(args.data_dir, exist_ok=True)
    domains = list(cfg.teachers.keys()) if args.all else [args.domain or "general"]

    if args.dry_run:
        dry_run(cfg, domains)
        return

    device = args.device
    if device == "auto":
        device = "cuda" if (torch.cuda.is_available() if "torch" in sys.modules else False) else "cpu"
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    for d in domains:
        gen(d, args.n, cfg, os.path.join(args.data_dir, f"teacher_{d}.jsonl"),
            device=device, max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
