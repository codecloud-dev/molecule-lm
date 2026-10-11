"""Molecule 优化器：Muon（2D 权重）+ AdamW（嵌入/bias/1D）混合，WSD 调度。

Muon 用 Newton-Schulz 正交化，比 AdamW 更省 token、小模型更稳（Kimi K2 / GLM-4.5 量产验证）。
其余参数（嵌入、bias、1D 向量）用 AdamW。
"""

import torch
import math


class Muon(torch.optim.Optimizer):
    """精简 Muon：对每个 2D 参数做正交化更新（Newton-Schulz）。"""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.ndim != 2:
                    continue
                m = g["momentum"]
                ns = g["ns_steps"]
                nesterov = g["nesterov"]
                lr = g["lr"]
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)
                buf = state["momentum_buffer"]
                buf.mul_(m).add_(grad)
                if nesterov:
                    grad = grad.add(buf, alpha=m)
                else:
                    grad = buf
                # Newton-Schulz 正交化：update = (G Gᵀ)^(-1/2) · G = U·Vᵀ
                # 若 out > in 先转置，使 Gram 取较小维度；最后转回 (out,in)。
                U = grad.float()
                transposed = U.shape[0] > U.shape[1]
                if transposed:
                    U = U.T
                X = U @ U.mT
                X /= X.norm() + 1e-8
                for _ in range(ns):
                    A = X @ X.mT
                    B = A @ X * 3 - X * 2
                    X = X + (B - X @ A) / 4
                update = X @ U            # 正交化 Gram 乘回 G，回到 (out,in)/(in,out)
                if transposed:
                    update = update.T
                p.data.add_(update.to(p.dtype) * -lr)


def build_optimizers(model, cfg):
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "embed" not in name and "lm_head" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)
    muon = Muon(muon_params, lr=cfg.mu_lr)
    adam = torch.optim.AdamW(adam_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    return muon, adam


def build_schedulers(muon, adam, cfg, total_steps):
    warmup = int(total_steps * cfg.warmup_ratio)
    decay = int(total_steps * cfg.wsd_decay_ratio)

    def lam(current_step):
        if current_step < warmup:
            return (current_step + 1) / max(1, warmup)
        if current_step < total_steps - decay:
            return 1.0
        # WSD 末段余弦衰减到 0.1
        t = (current_step - (total_steps - decay)) / max(1, decay)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    sm = torch.optim.lr_scheduler.LambdaLR(muon, lam)
    sa = torch.optim.lr_scheduler.LambdaLR(adam, lam)
    return sm, sa
