"""molecule v3.0 生成器（基于 .pt 检查点，CPU 推理，ChatML 模板）。

设计要点：
  - 控制 token / 思维 token 仅作词表保留位；生成时默认不主动产出它们（除非 --mode 指定），
    保证 v3.0（PPL<10 地基阶段）输出对人类可见、不退化。
  - 用 PyTorch 直接前向（v3 架构 MInfer 私有引擎暂未发布；评测/调试走 Python 推理）。

用法：
  python generate_v3.py --ckpt checkpoints/v3.pt --prompt "请介绍一下你自己。" --max_new 120
"""

import argparse
import math
import os

import torch
from model_v3 import MoleculeV3
from tokenizer_bpe import ByteBPETokenizer


CHAT_TEMPLATE = ("<|im_start|>user\n{prompt}<|im_end|>\n"
                 "<|im_start|>assistant\n")


def load(ckpt):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = MoleculeV3(vocab_size=8192)
    model.load_state_dict(st["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def generate(model, tk, prompt, max_new=120, temperature=0.85, top_p=0.9,
             no_control=True):
    ids = tk.encode(CHAT_TEMPLATE.format(prompt=prompt))
    ctx = torch.tensor(ids, dtype=torch.long)
    st = model.init_state(1)
    logits = None
    for t in ctx:                      # 先走过 prompt 建立状态（流式 step）
        logits = model.step(t.view(1, 1), st)[0, 0]
    out_ids = []
    for _ in range(max_new):
        if no_control:
            for tid in range(262, 286):
                logits[tid] = float("-inf")
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            lp = (logits / temperature).softmax(-1)
            sp, si = torch.sort(lp, descending=True)
            cut = sp[sp.cumsum(0) <= top_p].shape[0]
            m = torch.ones_like(lp, dtype=torch.bool)
            m[si[:cut]] = False
            lp = lp.masked_fill(m, 0.0) / lp.masked_fill(m, 0.0).sum()
            nxt = int(torch.multinomial(lp, 1))
        if nxt == tk._special_ids["eos"]:
            break
        out_ids.append(nxt)
        logits = model.step(torch.tensor([[nxt]]), st)[0, 0]
    return tk.decode(out_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v3.pt")
    ap.add_argument("--prompt", default="请介绍一下你自己。")
    ap.add_argument("--max_new", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--allow_control", action="store_true",
                    help="允许生成控制/思维 token（v3.5+ 才用）")
    args = ap.parse_args()

    tk = ByteBPETokenizer.load()
    model = load(args.ckpt)
    txt = generate(model, tk, args.prompt, args.max_new, args.temperature,
                   args.top_p, no_control=not args.allow_control)
    print(txt)


if __name__ == "__main__":
    main()
