#!/usr/bin/env python3
"""molecule V0.1 本地 Gradio 对话 demo。

启动：python demo.py  ->  浏览器打开提示的地址即可对话（体验中文 + 高情商）。
依赖：pip install gradio torch transformers
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MoleculeConfig
from model import MoleculeModel
from tokenizer import ReducedTokenizer
from generate import SYS_IDENTITY, load_model_tokenizer, build_messages, generate_text, maybe_tool, TOOL_IMPL


def make_predictor(ckpt, data_dir):
    cfg = MoleculeConfig.load("config.json") if os.path.exists("config.json") else MoleculeConfig()
    model, tk = load_model_tokenizer(cfg, data_dir, ckpt)

    def predict(user, history):
        hist = [(r, c) for r, c in history]
        prompt = build_messages(SYS_IDENTITY, hist, user)
        text = generate_text(model, tk, prompt, max_new=160)
        call = maybe_tool(text)
        if call and call.get("name") in TOOL_IMPL:
            res = TOOL_IMPL[call["name"]](call.get("arguments", {}))
            text += f"\n[工具结果] {res}"
        return text

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/model.safetensors")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    try:
        import gradio as gr
    except ImportError:
        print("请先安装 gradio: pip install gradio")
        return
    pred = make_predictor(args.ckpt, args.data_dir)
    demo = gr.ChatInterface(pred, title="molecule V0.1 对话", description=SYS_IDENTITY)
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
