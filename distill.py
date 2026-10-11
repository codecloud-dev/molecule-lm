#!/usr/bin/env python3
"""molecule V0.1 多老师分领域蒸馏数据生成。

- 5 位老师分领域 1对1：通用/推理/数学/代码/EQ，按 prompt 标签路由。
- 每位老师温度按 DESIGN.md 设置（数学/代码偏低保准确、通用/EQ 偏高保多样）。
- 生成结果存 data/teacher_<domain>.jsonl；训练时并入对应 domain 的 .bin。
- 用法：python distill.py --domain math --n 2000
        python distill.py --all   （全部 domain）
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


def gen(domain, n, cfg, out_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tcfg = cfg.teachers[domain]
    print(f"[distill] loading teacher {tcfg['model']} (temp={tcfg['temp']})")
    tok = AutoTokenizer.from_pretrained(tcfg["model"])
    model = AutoModelForCausalLM.from_pretrained(tcfg["model"], torch_dtype="auto")
    model.eval()
    samples = []
    for i in range(n):
        prompt = f"<|im_start|>system\n{SEEDS[domain]}<|im_end|>\n<|im_start|>user\n第{i+1}条<|im_end|>\n<|im_start|>assistant\n"
        ids = tok(prompt, return_tensors="pt").input_ids
        out = model.generate(ids, max_new_tokens=256, do_sample=True, temperature=tcfg["temp"], top_p=0.9)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        samples.append({"domain": domain, "text": text})
        if (i + 1) % 50 == 0:
            print(f"[distill] {domain} {i+1}/{n}")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[distill] wrote {len(samples)} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()
    cfg = MoleculeConfig.load("config.json") if os.path.exists("config.json") else MoleculeConfig()
    os.makedirs(args.data_dir, exist_ok=True)
    domains = list(cfg.teachers.keys()) if args.all else [args.domain or "general"]
    for d in domains:
        gen(d, args.n, cfg, os.path.join(args.data_dir, f"teacher_{d}.jsonl"))


if __name__ == "__main__":
    main()
