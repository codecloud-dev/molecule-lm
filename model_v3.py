"""molecule v3.0 统一熔铸骨干：24 Mamba 线性递推层 + 6 BinaryAttention 层（4:1）。

几何（v3.0 起点）：
  - 30 层，模式 [M,M,A,M,M] x 6；块级权重共享（同型每 2 层共享，12+3=15 组）
  - d_model=640，Mamba d_inner=2560 / 8 头 / 状态 16；注意力 GQA 10Q:3KV、head_dim 64
  - FFN 1704 SwiGLU；Pre-LN RMSNorm + 每头 QK-Norm；DeepNorm 残差 alpha=2.78
  - 词表 8192（与 tokenizer_bpe 布局一致），嵌入与输出头绑定，*sqrt(d) 缩放
  - MTP 辅助头（复用绑定输出头，位移 +2，loss 权重 0.3）；Z-loss 1e-4
  - BinaryAttention：Q/K 1-bit（sign + STE），±1 浮点点积与 XNOR+popcount 数学等价
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA_DEEPNORM = 2.78
MTP_WEIGHT = 0.3
ZLOSS_WEIGHT = 1e-4


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


# ---------------------------------------------------------------------------
# Mamba 块：选择性 SSM（Mamba-2 式 SSD 分块并行扫描，头级标量衰减）
# ---------------------------------------------------------------------------
class MambaSSM(nn.Module):
    def __init__(self, d_model=640, d_inner=2560, n_heads=8, d_state=16, dt_rank=48,
                 chunk=16, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model, self.d_inner, self.n_heads = d_model, d_inner, n_heads
        self.d_state, self.dt_rank, self.chunk = d_state, dt_rank, chunk
        self.dt_min, self.dt_max = dt_min, dt_max
        self.head_dim = d_inner // n_heads

        self.norm_in = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=4, groups=d_inner,
                                padding=3, bias=True)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, n_heads, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.rand(n_heads) * 15.0 + 1.0))  # A∈(1,16)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.02 / ALPHA_DEEPNORM)

    def forward(self, x):
        B, L, _ = x.shape
        x = self.norm_in(x)
        xz = self.in_proj(x)
        xb, z = xz.chunk(2, dim=-1)
        # 因果 depthwise conv（k=4）
        xb = self.conv1d(xb.transpose(1,2))[..., :L].transpose(1, 2)
        xb = F.silu(xb)

        xdb = self.x_proj(xb)
        dt_raw, Bp, Cp = torch.split(
            xdb, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_min + (self.dt_max - self.dt_min) * torch.sigmoid(self.dt_proj(dt_raw))
        log_a = -dt * torch.exp(self.A_log)                    # [B,L,H] 头级对数衰减

        H, E, N, T = self.n_heads, self.head_dim, self.d_state, self.chunk
        xh = xb.view(B, L, H, E)

        nchunk = (L + T - 1) // T
        h = x.new_zeros(B, H, N, E)                            # 跨块状态
        ys = []
        for c in range(nchunk):
            s, e = c * T, min((c + 1) * T, L)
            tlen = e - s
            cl = torch.cumsum(log_a[:, s:e], dim=1)            # [B,t,H]
            a_cum = torch.exp(cl)                              # [B,t,H] 全<=1
            Bc = Bp[:, s:e]                                    # [B,t,N]
            xc = xh[:, s:e]                                    # [B,t,H,E]
            Cc = Cp[:, s:e]                                    # [B,t,N]
            # 因果衰减矩阵（含对角）：M[t,m]=exp(cl_t-cl_m), m<=t，恒<=1 数值安全
            # 显式因果掩码（M 形状 [B,t,m,H]，掩码作用在 (t,m) 平面，避免 tril 维度错位）
            M = torch.exp(cl.unsqueeze(2) - cl.unsqueeze(1))   # [B,t,m,H]
            tc = torch.arange(tlen, device=M.device).view(1, -1, 1, 1)
            mc = torch.arange(tlen, device=M.device).view(1, 1, -1, 1)
            M = M.masked_fill(mc > tc, 0.0)
            # G[b,m,h,n,e] = Bc[b,m,n] * xc[b,m,h,e]
            G = Bc.unsqueeze(2).unsqueeze(-1) * xc.unsqueeze(3)   # [B,t,H,N,E]
            # 块内衰减求和：S[b,t,h,n,e] = sum_{m<=t} M[b,t,m,h] * G[b,m,h,n,e]
            S = torch.einsum("btmh,bmhne->bthne", M, G)           # [B,t,H,N,E]
            # 输出 = C 加权（块内）+ 起点 carry（a_cum_t * h_start 经 C 加权）
            y_in = torch.einsum("btn,bthne->bthe", Cc, S)         # [B,t,H,E]
            # 起点 carry 逐 t 用各自 C 加权：a_cum[t] * sum_n C[t,n] h[n,e]
            hC = torch.einsum("btn,bhne->bthe", Cc, h)            # [B,t,H,E]
            y = y_in + a_cum.unsqueeze(-1) * hC                   # [B,t,H,E]
            ys.append(y.reshape(B, tlen, self.d_inner))
            # 状态推进到块尾
            h = S[:, -1] + a_cum[:, -1].unsqueeze(-1).unsqueeze(-1) * h
        y = torch.cat(ys, dim=1)                                  # [B,L,d_inner]
        y = y + xh.reshape(B, L, self.d_inner) * self.D           # D 跳连
        y = y * F.silu(z)                                         # 门控
        return self.out_proj(y)

    # ---- 增量 step（O(1)/步，用于流式生成与 MInfer 接口）----
    def init_state(self, B, device, dtype):
        k = self.conv1d.kernel_size[0]
        return {
            "h": torch.zeros(B, self.n_heads, self.d_state, self.head_dim,
                             device=device, dtype=dtype),
            "conv": torch.zeros(B, self.d_inner, k - 1, device=device, dtype=dtype),
        }

    def step(self, x_t, st):
        B = x_t.shape[0]
        x = self.norm_in(x_t)
        xz = self.in_proj(x)
        xb, z = xz.chunk(2, dim=-1)
        window = torch.cat([st["conv"], xb.transpose(1, 2)], dim=2)   # [B,di,k]
        # 因果卷积对齐：padding=3 时局部窗口卷积的输出索引 3 == 全局位置 t 的输出
        xb = F.silu(self.conv1d(window)[..., 3:4]).transpose(1, 2)    # [B,1,di]
        st["conv"] = window[..., 1:]
        xdb = self.x_proj(xb)
        dt_raw, Bp, Cp = torch.split(xdb, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_min + (self.dt_max - self.dt_min) * torch.sigmoid(self.dt_proj(dt_raw))
        a = torch.exp(-dt * torch.exp(self.A_log)).squeeze(1)         # [B,H], 恒<=1
        H, E, N = self.n_heads, self.head_dim, self.d_state
        xh = xb.view(B, H, E)                                        # [B,H,E]
        Bp = Bp.squeeze(1); Cp = Cp.squeeze(1)                        # [B,N]
        G = Bp.unsqueeze(1).unsqueeze(-1) * xh.unsqueeze(2)           # [B,H,N,E]
        # 单步递推：h = G + a*h_prev；输出 = C·h（与分块扫描逐 step 等价）
        st["h"] = G + a.unsqueeze(-1).unsqueeze(-1) * st["h"]         # [B,H,N,E]
        y = torch.einsum("bn,bhne->bhe", Cp, st["h"]).reshape(B, 1, self.d_inner)  # [B,1,di]
        y = y + xb * self.D                                          # D 跳连（di 维）
        y = y * F.silu(z)                                            # 门控
        return self.out_proj(y), st


# ---------------------------------------------------------------------------
# BinaryAttention 块（1-bit Q/K + GQA 10:3 + SwiGLU FFN，内部含 DeepNorm 残差）
# ---------------------------------------------------------------------------
class BinaryAttention(nn.Module):
    def __init__(self, d_model=640, n_q_heads=10, n_kv_heads=3, head_dim=64, ffn_dim=1704):
        super().__init__()
        self.n_q, self.n_kv, self.hd = n_q_heads, n_kv_heads, head_dim
        self.norm_attn = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, n_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.qk_norm = RMSNorm(head_dim)
        self.out_proj = nn.Linear(n_q_heads * head_dim, d_model, bias=False)
        self.norm_ffn = RMSNorm(d_model)
        self.w_gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_up = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, d_model, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.02 / ALPHA_DEEPNORM)
        nn.init.normal_(self.w_down.weight, std=0.02 / ALPHA_DEEPNORM)

    @staticmethod
    def _binarize(x):
        """sign + 直通估计器（STE）：前向 ±1，反向传原值梯度。"""
        s = torch.where(x >= 0, 1.0, -1.0)
        return x + (s - x).detach()

    def _attn(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_q, self.hd)
        k = self.k_proj(x).view(B, L, self.n_kv, self.hd)
        v = self.v_proj(x).view(B, L, self.n_kv, self.hd)
        q, k = self.qk_norm(q), self.qk_norm(k)
        qb, kb = self._binarize(q), self._binarize(k)          # ±1
        q, k, v = (t.transpose(1, 2) for t in (qb, kb, v))     # [B,h,L,d]
        # GQA 10Q:3KV（10 不被 3 整除，按就近映射 4/3/3 分组）
        qmap = torch.div(torch.arange(self.n_q, device=q.device) * self.n_kv,
                         self.n_q, rounding_mode="floor")
        k, v = k.index_select(1, qmap), v.index_select(1, qmap)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        return self.out_proj(y.transpose(1, 2).reshape(B, L, -1))

    def init_state(self, B, device, dtype):
        return {"K": torch.zeros(B, 0, self.n_kv, self.hd, device=device, dtype=dtype),
                "V": torch.zeros(B, 0, self.n_kv, self.hd, device=device, dtype=dtype)}

    def step(self, x_t, st):
        B = x_t.shape[0]
        # 与 forward._attn 一致：先对 block 输入做注意力归一化，再投影 q/k/v
        xa = self.norm_attn(x_t)
        q = self.q_proj(xa).view(B, 1, self.n_q, self.hd)
        k = self.k_proj(xa).view(B, 1, self.n_kv, self.hd)
        v = self.v_proj(xa).view(B, 1, self.n_kv, self.hd)
        q, k = self.qk_norm(q), self.qk_norm(k)
        qb, kb = self._binarize(q), self._binarize(k)
        st["K"] = torch.cat([st["K"], kb], dim=1)              # [B,L,n_kv,hd]
        st["V"] = torch.cat([st["V"], v], dim=1)
        qmap = torch.div(torch.arange(self.n_q, device=q.device) * self.n_kv,
                         self.n_q, rounding_mode="floor")
        # 当前位置 query：[B,n_q,1,hd]；KV 累积态：[B,n_q,L,hd]（已按 GQA 映射展开）
        q = qb.transpose(1, 2)                                  # [B,n_q,1,hd]
        K = st["K"].index_select(2, qmap).transpose(1, 2)       # [B,n_q,L,hd]
        V = st["V"].index_select(2, qmap).transpose(1, 2)       # [B,n_q,L,hd]
        # 与 forward._attn 严格对齐：最后一行 query 对所有"过去"位置做注意力（无需 mask）
        att = (q @ K.transpose(-1, -2)) / math.sqrt(self.hd)    # [B,n_q,1,L]
        att = F.softmax(att, dim=-1) @ V                        # [B,n_q,1,hd]
        att_out = self.out_proj(att.transpose(1, 2).reshape(B, 1, -1))  # [B,1,d_model]
        # 复刻 forward 的 DeepNorm 残差 + SwiGLU FFN（与 MoleculeV3.step 调用约定一致：返回完整 block 输出）
        x = x_t + ALPHA_DEEPNORM * att_out
        xn = self.norm_ffn(x)
        ffn = self.w_down(F.silu(self.w_gate(xn)) * self.w_up(xn))
        x = x + ALPHA_DEEPNORM * ffn
        return x, st

    def forward(self, x):
        x = x + ALPHA_DEEPNORM * self._attn(self.norm_attn(x))
        x = x + ALPHA_DEEPNORM * self.w_down(
            F.silu(self.w_gate(self.norm_ffn(x))) * self.w_up(self.norm_ffn(x)))
        return x


# ---------------------------------------------------------------------------
# v3.0 模型：[M,M,A,M,M] x 6，同型每 2 层共享权重（15 组）
# ---------------------------------------------------------------------------
class MoleculeV3(nn.Module):
    def __init__(self, vocab_size=8192, d_model=640, d_inner=2560, d_state=16,
                 dt_rank=48, ffn_dim=1704, n_layers=30):
        super().__init__()
        self.vocab_size, self.d_model, self.n_layers = vocab_size, d_model, n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.norm_out = RMSNorm(d_model)

        m_pos = [i for i in range(n_layers) if i % 5 != 2]
        a_pos = [i for i in range(n_layers) if i % 5 == 2]
        self.plan = {}
        for p in m_pos:
            self.plan[p] = ("m", m_pos.index(p) // 2)
        for p in a_pos:
            self.plan[p] = ("a", a_pos.index(p) // 2)
        m_uniq = max(g for k, g in self.plan.values() if k == "m") + 1
        a_uniq = max(g for k, g in self.plan.values() if k == "a") + 1

        self.mamba_blocks = nn.ModuleList(
            [MambaSSM(d_model, d_inner, 8, d_state, dt_rank) for _ in range(m_uniq)])
        self.attn_blocks = nn.ModuleList(
            [BinaryAttention(d_model, 10, 3, 64, ffn_dim) for _ in range(a_uniq)])

        nn.init.normal_(self.embed.weight, std=0.02)
        with torch.no_grad():
            self.embed.weight.mul_(1.0 / ALPHA_DEEPNORM)
        self.lm_head = self.embed  # 绑定

    def forward(self, idx, targets=None):
        x = self.embed(idx) * math.sqrt(self.d_model)
        for p in range(self.n_layers):
            kind, gid = self.plan[p]
            if kind == "m":
                x = x + ALPHA_DEEPNORM * self.mamba_blocks[gid](x)
            else:
                x = self.attn_blocks[gid](x)
        x = self.norm_out(x)
        logits = F.linear(x, self.lm_head.weight)
        out = {"logits": logits}
        if targets is not None:
            lm = F.cross_entropy(logits[:, :-1].reshape(-1, self.vocab_size),
                                 targets[:, 1:].reshape(-1))
            mtp = F.cross_entropy(logits[:, :-2].reshape(-1, self.vocab_size),
                                  targets[:, 2:].reshape(-1))
            z = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
            out["loss"] = lm + MTP_WEIGHT * mtp + ZLOSS_WEIGHT * z
            out["loss_lm"], out["loss_mtp"] = lm.detach(), mtp.detach()
        return out

    # ---- 增量生成（O(L)/整段，流式 + MInfer 接口共用）----
    # 注意：权重按组共享，但递推状态（SSM 隐状态 / 注意力 KV 缓存）必须【逐层独立】——
    # 同组相邻层处理的输入序列不同（layer p+1 的输入是 layer p 的输出），共享状态会污染递推轨迹。
    def init_state(self, B, device="cpu", dtype=torch.float32):
        st = {"mamba": {}, "attn": {}}
        for p in range(self.n_layers):
            kind, gid = self.plan[p]
            if kind == "m":
                st["mamba"][p] = self.mamba_blocks[gid].init_state(B, device, dtype)
            else:
                st["attn"][p] = self.attn_blocks[gid].init_state(B, device, dtype)
        return st

    def step(self, idx_t, st):
        x = self.embed(idx_t) * math.sqrt(self.d_model)   # [B,1,d_model]
        for p in range(self.n_layers):
            kind, gid = self.plan[p]
            if kind == "m":
                x = x + ALPHA_DEEPNORM * self.mamba_blocks[gid].step(x, st["mamba"][p])[0]
            else:
                x = self.attn_blocks[gid].step(x, st["attn"][p])[0]
        x = self.norm_out(x)
        return F.linear(x, self.lm_head.weight)           # [B,1,vocab]

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # 导出支持：权威张量命名 + 架构元数据（导出 GGUF / 续训共用，不改动 forward）
    # ------------------------------------------------------------------
    def block_modules(self):
        """按真实层序 yield (layer_idx, kind, group_id, module)。"""
        for p in range(self.n_layers):
            kind, gid = self.plan[p]
            mod = self.mamba_blocks[gid] if kind == "m" else self.attn_blocks[gid]
            yield p, kind, gid, mod

    def export_state(self):
        """返回 {权威张量名: Tensor}，同型共享权重天然只存一次。"""
        sd = {}
        sd["token_embd.weight"] = self.embed.weight
        sd["output_norm.weight"] = self.norm_out.weight
        for _, kind, gid, mod in self.block_modules():
            pfx = ("mamba.g%d." if kind == "m" else "attn.g%d.") % gid
            for nm, ten in mod.named_parameters():
                sd[pfx + nm] = ten
        return sd

    def architecture(self):
        m_uniq = max(g for k, g in self.plan.values() if k == "m") + 1
        a_uniq = max(g for k, g in self.plan.values() if k == "a") + 1
        return {
            "fqn": "molecule-v3",
            "n_layers": self.n_layers,
            "n_mamba": sum(1 for v in self.plan.values() if v[0] == "m"),
            "n_attn": sum(1 for v in self.plan.values() if v[0] == "a"),
            "d_model": self.d_model, "d_inner": 2560, "d_state": 16, "dt_rank": 48,
            "n_q": 10, "n_kv": 3, "head_dim": 64, "ffn_dim": 1704,
            "vocab_size": self.vocab_size, "alpha": ALPHA_DEEPNORM,
            "mamba_group": m_uniq, "attn_group": a_uniq,
            "block_share": 2, "mode": "[M,M,A,M,M] x 6 (24 Mamba + 6 BinaryAttention)",
        }

    def layer_plan(self):
        """真实层序：list[[kind, group_id]]，kind 0=Mamba 1=Attn。"""
        return [[0 if k == "m" else 1, g] for p in range(self.n_layers)
                for k, g in (self.plan[p],)]


if __name__ == "__main__":
    torch.manual_seed(0)
    m = MoleculeV3()
    print("unique params: %.1fM" % (m.num_params() / 1e6))
    print("layers: 30 (24 Mamba + 6 BinaryAttention), unique groups: %d" % len(set(m.plan.values())))
    idx = torch.randint(0, 8192, (2, 64))
    tgt = torch.randint(0, 8192, (2, 64))
    out = m(idx, tgt)
    print("logits:", tuple(out["logits"].shape), "loss: %.4f (随机基线≈%.2f)"
          % (out["loss"].item(), math.log(8192)))
    out["loss"].backward()
    g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    print("grad sum: %.1f (非零即梯度通)" % g)
