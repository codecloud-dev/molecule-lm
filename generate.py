#!/usr/bin/env python3
"""molecule 推理 + 轻量工具调用。

- 系统提示如实标注开发者：开源语言模型 · molecule · codecloud-dev 独立开发。
- 支持 function calling：模型输出 <tool_call>{json}</tool_call>，脚本解析并执行注册的工具。
- 用法：python generate.py --prompt "你好"  或  python generate.py --chat
"""

import os
import sys
import json
import ast
import operator as _op
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MoleculeConfig
from model import MoleculeModel
from tokenizer import ReducedTokenizer

SYS_IDENTITY = (
    "你是 molecule，一个开源语言模型，由 codecloud-dev 独立开发。"
    "你擅长中英双语，乐于助人、有温度。当被问到你是谁时，如实说明你的身份。"
)

# 示例工具（可扩展）
TOOLS = [
    {"name": "get_time", "description": "获取当前时间", "params": {}},
    {"name": "calc", "description": "计算简单算式", "params": {"expr": "string"}},
]


def _safe_calc(expr):
    """仅允许四则运算的安全求值（无 eval、无 import，杜绝注入）。"""
    ops = {
        ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
        ast.Div: _op.truediv, ast.Mod: _op.mod, ast.Pow: _op.pow,
        ast.USub: _op.neg, ast.UAdd: _op.pos,
    }

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("仅支持数字与 + - * / % ** 运算")

    return ev(ast.parse(str(expr), mode="eval"))


TOOL_IMPL = {
    "get_time": lambda args: "现在时间是本地时间。",
    "calc": lambda args: str(_safe_calc(args.get("expr", "0"))),
}


def build_messages(system, history, user):
    # 用 ChatML 组织对话
    prompt = "<|im_start|>system\n" + system + "<|im_end|>\n"
    for r, c in history:
        prompt += f"<|im_start|>{r}\n{c}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def load_model_tokenizer(cfg, data_dir, ckpt):
    model = MoleculeModel(cfg)
    try:
        from safetensors.torch import load_model
        load_model(model, ckpt)
    except Exception:
        sd = torch.load(os.path.join("checkpoints", "latest.ckpt"), map_location="cpu")
        model.load_state_dict(sd["model"])
    model.eval()
    from tokenizer import get_base_tokenizer
    base, _ = get_base_tokenizer(cfg.tokenizer_name)
    tk = ReducedTokenizer.load(base, os.path.join(data_dir, "tokenizer_map.json"))
    return model, tk


@torch.no_grad()
def generate_text(model, tk, prompt, max_new=128, temperature=0.9, top_p=0.9):
    ids = tk.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new=max_new, temperature=temperature, top_p=top_p)
    return tk.decode(out[0].tolist())


def maybe_tool(text):
    if "<tool_call>" in text:
        try:
            j = text.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            return json.loads(j)
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--ckpt", default="checkpoints/model.safetensors")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()
    cfg = MoleculeConfig.load(args.config) if os.path.exists(args.config) else MoleculeConfig()
    model, tk = load_model_tokenizer(cfg, args.data_dir, args.ckpt)

    if args.prompt:
        prompt = build_messages(SYS_IDENTITY, [], args.prompt)
        print(generate_text(model, tk, prompt))
        return

    if args.chat:
        history = []
        print("molecule 对话（输入 exit 退出）")
        while True:
            u = input("你: ")
            if u.strip() == "exit":
                break
            prompt = build_messages(SYS_IDENTITY, history, u)
            text = generate_text(model, tk, prompt)
            call = maybe_tool(text)
            if call and call.get("name") in TOOL_IMPL:
                res = TOOL_IMPL[call["name"]](call.get("arguments", {}))
                text += f"\n[工具结果] {res}"
            print("molecule:", text)
            history.append(("user", u))
            history.append(("assistant", text))


if __name__ == "__main__":
    main()
