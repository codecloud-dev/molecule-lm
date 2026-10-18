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
import re

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


# ---- 生成式评测：要求模型中英各答一句，查讲话有无问题 ----
_GEN_SYS = "你是 molecule，一个开源语言模型，由 codecloud-dev 独立开发，擅长中英双语。"


def _has_chinese(text):
    return any("\u4e00" <= c <= "\u9fff" for c in text)


def _has_latin_alpha(text):
    return any(c.isascii() and c.isalpha() for c in text)


@torch.no_grad()
def eval_generative(model, tk, max_new=140, temperature=0.9, top_p=0.9):
    user = "请用一句话用中文回答，再用一句话用英文回答：什么是人工智能？"
    prompt = ("<|im_start|>system\n" + _GEN_SYS + "<|im_end|>\n"
              "<|im_start|>user\n" + user + "<|im_end|>\n"
              "<|im_start|>assistant\n")
    ids = tk.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new=max_new, temperature=temperature, top_p=top_p)
    text = tk.decode(out[0].tolist())
    words = text.split()
    n = max(len(words), 1)
    rep = max((text.count(w) for w in set(words)), default=0)
    degenerate = (len(text) < 12) or (rep / n > 0.6)
    problems = []
    if not _has_chinese(text):
        problems.append("缺中文")
    if not _has_latin_alpha(text):
        problems.append("缺英文")
    if degenerate:
        problems.append("疑似重复/过短")
    return {"text": text[:400], "ok": not problems, "problems": problems}


# ---- 代码能力评测：及格线（add 函数基础用例），不搞全能 ----
_CODE_SYS = "你是 molecule，一个开源语言模型，由 codecloud-dev 独立开发。只输出 Python 代码，不要解释。"


def _extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    i = text.find("def ")
    if i >= 0:
        return text[i:].strip()
    return text.strip()


def _safe_run_code(code, fn_name, tests):
    safe_builtins = {
        "len": len, "range": range, "int": int, "float": float, "str": str,
        "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool,
        "abs": abs, "min": min, "max": max, "sum": sum, "enumerate": enumerate,
        "zip": zip, "sorted": sorted, "round": round, "pow": pow, "map": map,
        "filter": filter, "any": any, "all": all, "True": True, "False": False,
        "None": None,
    }
    ns = {}
    try:
        exec(compile(code, "<gen>", "exec"), {"__builtins__": safe_builtins}, ns)
    except Exception:
        return False, "编译/执行失败"
    fn = ns.get(fn_name)
    if not callable(fn):
        return False, f"未定义函数 {fn_name}"
    for args, expected in tests:
        try:
            if fn(*args) != expected:
                return False, f"{fn_name}{args} 期望 {expected}"
        except Exception:
            return False, f"{fn_name}{args} 运行报错"
    return True, "通过"


@torch.no_grad()
def eval_code(model, tk, max_new=220, temperature=0.8, top_p=0.9):
    user = "写一个 Python 函数 add(a, b)，返回 a 与 b 的和。只输出代码。"
    prompt = ("<|im_start|>system\n" + _CODE_SYS + "<|im_end|>\n"
              "<|im_start|>user\n" + user + "<|im_end|>\n"
              "<|im_start|>assistant\n")
    ids = tk.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new=max_new, temperature=temperature, top_p=top_p)
    text = tk.decode(out[0].tolist())
    code = _extract_code(text)
    ok, detail = _safe_run_code(code, "add", [((2, 3), 5), ((-1, 1), 0), ((0, 0), 0)])
    return {"text": text[:400], "code": code[:300], "ok": ok, "detail": detail}


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

    # 生成式评测（中英各一句，查讲话有无问题）+ 代码能力评测（及格线）
    try:
        gen = eval_generative(model, tk)
    except Exception as e:
        gen = {"text": "", "ok": False, "problems": [f"error: {e}"]}
    try:
        code_eval = eval_code(model, tk)
    except Exception as e:
        code_eval = {"ok": False, "detail": f"error: {e}"}
    print(f"[gen] 中英各一句 ok={gen['ok']} problems={gen.get('problems')}")
    print(f"[code] add 及格 ok={code_eval['ok']} detail={code_eval.get('detail')}")

    # 写趋势表（中性：仅展示自测分数与参考值，不做跨模型胜负判定）
    md = ["# molecule 评测趋势表（内置冒烟集，非官方 split）", "",
          "| 冒烟集 | molecule(自测) | 参考值 | 备注 |",
          "| --- | --- | --- | --- |"]
    for name, acc, target, ok in rows:
        md.append(f"| {name} | {acc:.1f} | {target:.1f} | {'≥ref' if ok else '<?ref'} |")
    md.append("")
    md.append("> 说明：本表为自写冒烟题，仅作训练趋势监控；参考值取自公开模型公开数字，")
    md.append("> **不可直接比较**，正式评测请接入 lm-evaluation-harness + 官方 split。")
    md.append("")
    md.append("## 生成式评测（中英各一句 · 查讲话有无问题）")
    md.append(f"- 结果：**{'通过' if gen['ok'] else '未通过'}**  "
              f"{'；'.join(gen.get('problems', [])) or '中英文齐备、无异常重复'}")
    md.append(f"- 模型输出（节选）：{gen.get('text', '').replace(chr(10), ' ')}")
    md.append("")
    md.append("## 代码能力评测（及格线 · add 函数基础用例）")
    md.append(f"- 结果：**{'通过' if code_eval['ok'] else '未通过'}**  {code_eval.get('detail', '')}")
    md.append(f"- 模型输出（节选）：{code_eval.get('text', '').replace(chr(10), ' ')}")
    open(args.out, "w", encoding="utf-8").write("\n".join(md))
    print(f"\n[write] {args.out}")
    with open("eval_status.json", "w") as f:
        json.dump({
            "all_pass": all_pass,
            "rows": rows,
            "gen_ok": bool(gen["ok"]),
            "code_ok": bool(code_eval["ok"]),
        }, f)


if __name__ == "__main__":
    main()
