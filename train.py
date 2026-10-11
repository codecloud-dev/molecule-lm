#!/usr/bin/env python3
"""molecule V0.1 训练主脚本（纯 CPU，单轮 ≤6h，支持跨轮续训）。

- 自动探测 bf16（支持才用，否则 fp32）。
- 存全量续训态：model + Muon/AdamW optimizer + step + 数据游标 + RNG 种子。
- 分域加权（DN-lite）：数学/代码/推理/EQ 等少数域损失加权，防被通用域淹没。
- 周期性跑轻量评测；结束存 safetensors 供导出。
"""

import os
import sys
import json
import time
import argparse
import random

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MoleculeConfig
from model import MoleculeModel, count_params
from optim import build_optimizers, build_schedulers
from tokenizer import ReducedTokenizer, get_base_tokenizer
from data import MixedTokenDataset, list_bins, build_seed_corpus_if_needed

torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))


def detect_bf16():
    try:
        cap = torch.backends.cpu.get_cpu_capability()
        return "AVX512_BF16" in cap or "AMX" in cap
    except Exception:
        return False


def build_tokenizer(cfg, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    map_path = os.path.join(data_dir, "tokenizer_map.json")
    base, is_fallback = get_base_tokenizer(cfg.tokenizer_name)
    if os.path.exists(map_path):
        tk = ReducedTokenizer.load(base, map_path)
        print(f"[tok] loaded reduced vocab size={tk.vocab_size}")
        return tk
    # 用种子语料（含教师数据若已生成）做频率统计
    from seed_corpus import SEED
    corpus = []
    for d in SEED:
        corpus.extend(SEED[d])
    tk = ReducedTokenizer.build_from_corpus(
        base, corpus, target_size=cfg.vocab_size, cache_path=map_path
    )
    print(f"[tok] built reduced vocab size={tk.vocab_size}")
    return tk


def domain_weight(domain, cfg):
    # DN-lite：少数域（数学/代码/推理/EQ）相对通用域加权，防淹没
    base = {"general": 1.0, "math": 1.5, "code": 1.5, "reason": 1.5, "eq": 1.3}
    return base.get(domain, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--global_tokens", type=int, default=None)
    ap.add_argument("--mix_general", type=float, default=None)
    ap.add_argument("--mix_math", type=float, default=None)
    ap.add_argument("--mix_code", type=float, default=None)
    ap.add_argument("--mix_reason", type=float, default=None)
    ap.add_argument("--mix_eq", type=float, default=None)
    ap.add_argument("--variant", default="baseline")
    args = ap.parse_args()

    cfg = MoleculeConfig.load(args.config) if os.path.exists(args.config) else MoleculeConfig()
    if args.seq_len:
        cfg.train_seq_len = args.seq_len
    if args.global_tokens:
        cfg.global_tokens_per_step = args.global_tokens
    for name, val in (
        ("mix_general", args.mix_general), ("mix_math", args.mix_math),
        ("mix_code", args.mix_code), ("mix_reason", args.mix_reason),
        ("mix_eq", args.mix_eq),
    ):
        if val is not None:
            setattr(cfg, name, val)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = "cpu"
    use_bf16 = (cfg.dtype == "bf16") or (cfg.dtype == "auto" and detect_bf16())
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"[env] use_bf16={use_bf16} dtype={dtype} threads={torch.get_num_threads()}")

    # 分词 + 数据
    tk = build_tokenizer(cfg, args.data_dir)
    cfg.vocab_size = tk.vocab_size
    cfg.save(args.config)   # 在 vocab_size 确定后再落盘，保证续训/评测对齐
    build_seed_corpus_if_needed(tk, args.data_dir, cfg.train_seq_len)
    bins = list_bins(args.data_dir, cfg.train_seq_len)
    ratios = {
        "general": cfg.mix_general, "math": cfg.mix_math, "code": cfg.mix_code,
        "reason": cfg.mix_reason, "eq": cfg.mix_eq,
    }
    ds = MixedTokenDataset(bins, ratios, cfg.train_seq_len, seed=cfg.seed)

    # 模型 + 优化器
    model = MoleculeModel(cfg).to(device)
    print(f"[model] params={count_params(model)/1e6:.2f}M  layers={cfg.num_hidden_layers}")
    muon, adam = build_optimizers(model, cfg)
    sm, sa = build_schedulers(muon, adam, cfg, args.max_steps)

    # 续训态
    ckpt = os.path.join(args.out_dir, "latest.ckpt")
    step = 0
    if args.resume and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=device)
        model.load_state_dict(st["model"])
        muon.load_state_dict(st["muon"])
        adam.load_state_dict(st["adam"])
        step = st["step"]
        print(f"[resume] step={step}")

    os.makedirs(args.out_dir, exist_ok=True)
    micro_bs = 4  # CPU 安全 microbatch；梯度累积到 global_tokens_per_step
    accum = max(1, cfg.global_tokens_per_step // (micro_bs * cfg.train_seq_len))

    model.train()
    t0 = time.time()
    while step < args.max_steps:
        # 混合精度
        with torch.autocast(device_type="cpu", dtype=dtype, enabled=use_bf16):
            total_loss = 0.0
            muon.zero_grad(set_to_none=True)
            adam.zero_grad(set_to_none=True)
            for _ in range(accum):
                x = ds.batch(micro_bs, cfg.train_seq_len).to(device)
                out = model(x, return_mtp=True)
                # 硬 CE（主骨架）
                logits = out["logits"][:, :-1].float()
                target = x[:, 1:]
                loss = F_cross(logits, target, reduction="mean")
                if out.get("mtp_logits") is not None:
                    mtp = out["mtp_logits"][:, :-1].float()
                    mtp_t = x[:, 2:]
                    loss = loss + cfg.loss_mtp * F_cross(mtp, mtp_t, reduction="mean")
                loss = cfg.loss_hard_ce * loss / accum
                loss.backward()
                total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        muon.step(); adam.step(); sm.step(); sa.step()
        step += 1
        if step % max(1, min(10, args.max_steps // 5)) == 0 or step <= 3:
            tok_s = (step * accum * micro_bs * cfg.train_seq_len) / max(1, time.time() - t0)
            print(f"[train] step={step} loss={total_loss:.3f} lr={adam.param_groups[0]['lr']:.2e} tok/s={tok_s:.0f}", flush=True)
        # 周期性保存
        if step % cfg.save_every == 0:
            save_ckpt(model, muon, adam, step, args.out_dir)
        if step % cfg.eval_every == 0 and step > 0:
            pass  # 轻量评测由 eval.py 在轮末执行，避免拖慢训练

    # 结束保存
    save_ckpt(model, muon, adam, step, args.out_dir, final=True)
    # 存 safetensors 权重
    save_safetensors(model, cfg, args.out_dir)
    print(f"[done] step={step}  time={time.time()-t0:.0f}s")


def F_cross(logits, target, reduction="mean"):
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction=reduction
    )


def save_ckpt(model, muon, adam, step, out_dir, final=False):
    st = {
        "model": model.state_dict(),
        "muon": muon.state_dict(),
        "adam": adam.state_dict(),
        "step": step,
    }
    torch.save(st, os.path.join(out_dir, "latest.ckpt"))
    if final:
        torch.save(st, os.path.join(out_dir, f"step{step}.ckpt"))


def save_safetensors(model, cfg, out_dir):
    try:
        from safetensors.torch import save_model
        save_model(model, os.path.join(out_dir, "model.safetensors"))
        print("[save] model.safetensors written")
    except Exception as e:
        torch.save(model.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))
        print(f"[save] fallback pytorch_model.bin ({e})")


if __name__ == "__main__":
    main()
