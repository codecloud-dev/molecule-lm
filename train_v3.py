"""molecule v3.0 基础模型训练入口（Mamba+BinaryAttention 混合骨干，byte-BPE 8192）。

用法（本地冒烟 / CI 同一入口）：
  python train_v3.py --seq_len 256 --batch 4 --max_minutes 280 --resume

纪律（严格对齐设计文档 §7 / §10 / §12.1）：
  - 这是 v3.0 基础模型：只夯骨干，什么高级功能都不接（System1/元认知/调度器/工具/记忆
    全是 v3.5+ 的事）；目标 PPL<10，能生成通顺中文即可。
  - 数据：Phase1 配比 中文60% / 英文10% / 代码20% / 书籍10%（build_corpus_v3.py 预编码）。
  - 损失：L_lm + 0.3*L_mtp(位移+2) + 1e-4*Z-loss。
  - 优化器：AdamW（矩阵参数走 Muon 为后续优化项；CPU 续训先用 AdamW 保稳定），
    β2=0.95，weight_decay=0.01，全局梯度裁剪 1.0。
  - 学习率：warmup 2000 步 + WSD 余弦衰减到 10%（基于【全局步数】，跨轮续训不重置）。
  - 断点：checkpoints/v3.pt（model+optimizer+step）；--resume 续训。
  - 训练时钟：data/v3_clock.json 累计 {分钟, token, 步数, 轮次}，供 10 天发版闸门使用。
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import build_corpus_v3
from model_v3 import MoleculeV3

DOMAINS = build_corpus_v3.DOMAINS
MIX = build_corpus_v3.MIX
CKPT = "checkpoints/v3.pt"
CLOCK = "data/v3_clock.json"
VOCAB = 8192

WARMUP = 2000
DECAY_STEPS = 20000
PEAK_LR = 3e-4


def lr_at(step, peak=PEAK_LR, warmup=WARMUP, decay_steps=DECAY_STEPS):
    if step < warmup:
        return peak * (step / warmup)
    t = (step - warmup) / decay_steps
    if t >= 1.0:
        return peak * 0.1
    return peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t)))


class MixedSampler:
    """按 Phase1 域混比抽整段序列。"""

    def __init__(self, seq_len, seed=42):
        self.seq_len = seq_len
        self.bins = {}
        for dom in DOMAINS:
            p = "data/v3_%s.bin" % dom
            if os.path.exists(p):
                self.bins[dom] = np.memmap(p, dtype=np.int32, mode="r").reshape(-1, seq_len)
        self.rng = np.random.default_rng(seed)
        self.domains = list(self.bins) or ["zh_general"]
        self.totmix = sum(MIX.get(d, 0.25) for d in self.domains) or 1.0

    def batch(self, bs):
        xs = []
        for _ in range(bs):
            r = self.rng.random() * self.totmix
            acc, pick = 0.0, self.domains[0]
            for d in self.domains:
                acc += MIX.get(d, 0.25)
                if r <= acc:
                    pick = d
                    break
            arr = self.bins[pick]
            i = self.rng.integers(0, max(arr.shape[0], 1))
            seg = np.asarray(arr[i], dtype=np.int64)
            xs.append(seg)
        return torch.from_numpy(np.stack(xs))


def load_clock():
    if os.path.exists(CLOCK):
        try:
            with open(CLOCK, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"total_minutes": 0.0, "total_tokens": 0, "total_steps": 0, "runs": 0}


def save_clock(clk):
    os.makedirs("data", exist_ok=True)
    with open(CLOCK, "w", encoding="utf-8") as f:
        json.dump(clk, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_steps", type=int, default=20000, help="全局步数上限（软时限优先）")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=PEAK_LR)
    ap.add_argument("--max_minutes", type=float, default=280)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(42)
    # 构建/复用语料（内置 seed + 可选下载增强）
    build_corpus_v3.build(args.seq_len)
    sampler = MixedSampler(args.seq_len)

    model = MoleculeV3(vocab_size=VOCAB)
    n_params = model.num_params() / 1e6
    print("[model] v3 基础模型参数: %.1fM（15 组权重共享，纯骨干）" % n_params)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)

    step0 = 0
    if args.resume and os.path.exists(CKPT):
        st = torch.load(CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step0 = st.get("step", 0)
        print("[resume] 从 step %d 续训" % step0)
    model.train()

    t_start = time.time()
    run_lm = run_mtp = None
    last_step = step0
    tokens_this_run = 0
    for step in range(step0, args.max_steps):
        if (time.time() - t_start) / 60 > args.max_minutes:
            print("[stop] 软时限 %.0f 分钟到，优雅停止 @ step %d"
                  % (args.max_minutes, step))
            break
        gstep = step0 + (step - step0)  # 全局步数（warmup 基于它）
        lr = lr_at(gstep, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = sampler.batch(args.batch)
        out = model(idx, idx)
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_step = step + 1
        tokens_this_run += args.batch * args.seq_len
        run_lm = out["loss_lm"].item() if run_lm is None else 0.95 * run_lm + 0.05 * out["loss_lm"].item()
        run_mtp = out["loss_mtp"].item() if run_mtp is None else 0.95 * run_mtp + 0.05 * out["loss_mtp"].item()
        if (step + 1) % 10 == 0 or step == step0:
            el = (time.time() - t_start) / 60
            ppl = math.exp(min(run_lm, 20))
            print("[step %d] loss=%.3f (lm=%.3f mtp=%.3f) ppl≈%.2f lr=%.2e %.1f min"
                  % (step + 1, loss.item(), run_lm, run_mtp, ppl, lr, el))

    # 累计训练时钟（供 10 天发版闸门）
    clk = load_clock()
    clk["total_minutes"] += (time.time() - t_start) / 60.0
    clk["total_tokens"] += tokens_this_run
    clk["total_steps"] = last_step
    clk["runs"] += 1
    clk["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_clock(clk)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": last_step, "loss": float(loss.item()),
                "ppl": float(math.exp(min(run_lm, 20))),
                "args": vars(args)}, CKPT)
    print("[done] checkpoint -> %s | 累计训练 %.1f 分钟 / %.1f 小时（%d 轮）"
          % (CKPT, clk["total_minutes"], clk["total_minutes"] / 60.0, clk["runs"]))


if __name__ == "__main__":
    main()
