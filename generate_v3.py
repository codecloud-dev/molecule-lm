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

# 控制/思维 token 区间（生成时默认屏蔽，v3.0 基础模型不产出这些内部 token）
CTRL_LO, CTRL_HI = 262, 286
# 特殊 token 区间（pad/unk/bos/im_start/im_end），生成时屏蔽 pad/unk/bos/im_start；
# 保留 eos 与 im_end 作停止位。
SPECIAL_LO, SPECIAL_HI = 256, 262
# ChatML 结束标记的字面形式：训练与推理都是按【文本】编码的（不是特殊 token id），
# 所以停止条件要基于解码后的字节，而不是某个 token id。
# ChatML 结束标记：分词器把 <|im_end|> 编成【原子特殊 token id=261】，
# 因此停止判定直接看 token id（不再靠字面文本匹配，更可靠且省 token）。
IM_END_ID = 261


def _utf8_valid_prefix(bs):
    """bs 是否为合法 UTF-8 串或「合法的不完整前缀」（用于阻止生成破坏 UTF-8 → 杜绝 U+FFFD）。"""
    try:
        bs.decode("utf-8", errors="strict")
        return True
    except UnicodeDecodeError as e:
        return e.reason == "unexpected end of data"   # 不完整前缀是合法的


def load(ckpt):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = MoleculeV3(vocab_size=8192)
    model.load_state_dict(st["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def generate(model, tk, prompt, max_new=120, temperature=0.85, top_p=0.9,
             no_control=True, repetition_penalty=1.25, no_repeat_ngram=3,
             min_p=0.05):
    """v3 生成。根因修复在分词器（汉字整字）+ 训练侧；此处为纵深防御兜底：
      - 重复惩罚 + no_repeat_ngram：抑制「翻来覆去的重复文字」；
      - min_p + top_p：剔除低概率随机 token，避免乱跳；
      - UTF-8 合法约束：禁止选会破坏 UTF-8 的 token → 不可能吐出 U+FFFD 诡异字符；
      - NaN/Inf 兜底：logits 异常时退回均匀分布，避免数值崩坏导致的乱码；
      - 屏蔽控制/思维 token（262–285），默认不产出内部 token。
    """
    ids = tk.encode(CHAT_TEMPLATE.format(prompt=prompt))
    ctx = torch.tensor(ids, dtype=torch.long)
    st = model.init_state(1)
    logits = None
    for t in ctx:                      # 先走过 prompt 建立状态（流式 step）
        logits = model.step(t.view(1, 1), st)[0, 0]
    out_ids = []
    byte_buf = b""                     # 已生成 token 的真实字节（UTF-8 约束用）
    for _ in range(max_new):
        l = logits.clone()
        # 1) 数值兜底：NaN/Inf 直接退回均匀
        if not torch.isfinite(l).all():
            l = torch.zeros_like(l)
        # 2) 屏蔽控制/思维 token（262–285）+ 特殊 token（pad/unk/bos/im_start；保留 eos/im_end）
        if no_control:
            l[CTRL_LO:CTRL_HI] = float("-inf")
            for sid in range(SPECIAL_LO, SPECIAL_HI):
                if sid not in (tk._special_ids["eos"], tk._special_ids["im_end"]):
                    l[sid] = float("-inf")
        # 3) 重复惩罚：对已经出现过的 token 降权（抑制复述）
        if repetition_penalty != 1.0 and out_ids:
            for tid in set(out_ids):
                l[tid] = l[tid] / repetition_penalty if l[tid] > 0 else l[tid] * repetition_penalty
        # 4) no_repeat_ngram：禁止重复最近 n-gram（强抗循环）
        if no_repeat_ngram and len(out_ids) >= no_repeat_ngram:
            ctx_ng = tuple(out_ids[-(no_repeat_ngram - 1):])
            bad = set()
            for i in range(len(out_ids) - no_repeat_ngram + 1):
                if tuple(out_ids[i:i + no_repeat_ngram - 1]) == ctx_ng:
                    bad.add(out_ids[i + no_repeat_ngram - 1])
            for tid in bad:
                l[tid] = float("-inf")
        # 5) UTF-8 合法约束：禁止会破坏 UTF-8 的候选 token
        for tid in range(l.shape[0]):
            if not _utf8_valid_prefix(byte_buf + tk._tbytes.get(tid, b"")):
                l[tid] = float("-inf")
        # 6) 采样
        finite = torch.isfinite(l)
        if not finite.any():            # 全被屏蔽（极端情况）：只允许 EOS/合法字节
            l = torch.zeros_like(l)
            l[CTRL_LO:CTRL_HI] = float("-inf")
            for sid in range(SPECIAL_LO, SPECIAL_HI):
                if sid not in (tk._special_ids["eos"], tk._special_ids["im_end"]):
                    l[sid] = float("-inf")
        if temperature <= 0:
            nxt = int(l.argmax())
        else:
            lp = (l / temperature).softmax(-1)
            # min_p：剔除低于 max*min_p 的长尾
            if min_p > 0:
                top = lp.max().item()
                lp = lp.masked_fill(lp < top * min_p, 0.0)
            if lp.sum() <= 0:
                lp = torch.ones_like(lp) * finite.float()
            sp, si = torch.sort(lp, descending=True)
            cut = max(1, sp[sp.cumsum(0) <= top_p].shape[0])
            m = torch.ones_like(lp, dtype=torch.bool)
            m[si[:cut]] = False
            lp = lp.masked_fill(m, 0.0) / lp.masked_fill(m, 0.0).sum()
            nxt = int(torch.multinomial(lp, 1))
        if nxt == tk._special_ids["eos"] or nxt == IM_END_ID:
            break
        out_ids.append(nxt)
        byte_buf = byte_buf + tk._tbytes.get(nxt, b"")
        byte_buf = byte_buf + tk._tbytes.get(nxt, b"")
        logits = model.step(torch.tensor([[nxt]]), st)[0, 0]
    return byte_buf.decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v3.pt")
    ap.add_argument("--prompt", default="请介绍一下你自己。")
    ap.add_argument("--max_new", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.25)
    ap.add_argument("--no_repeat_ngram", type=int, default=3)
    ap.add_argument("--min_p", type=float, default=0.05)
    ap.add_argument("--allow_control", action="store_true",
                    help="允许生成控制/思维 token（v3.5+ 才用）")
    args = ap.parse_args()

    tk = ByteBPETokenizer.load()
    model = load(args.ckpt)
    txt = generate(model, tk, args.prompt, args.max_new, args.temperature,
                   args.top_p, no_control=not args.allow_control,
                   repetition_penalty=args.repetition_penalty,
                   no_repeat_ngram=args.no_repeat_ngram, min_p=args.min_p)
    print(txt)


if __name__ == "__main__":
    main()
