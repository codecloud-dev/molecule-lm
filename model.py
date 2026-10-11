"""Molecule V0.1 模型定义。

融合架构（来源见 DESIGN.md）：
  RoPE + NTK-aware 动态外推 / GQA(9头,3KV) / QK-Norm(RoPE前) /
  SwiGLU(2.67x) / RMSNorm+Pre-Norm / logit软封顶(50/30) /
  DeepNorm增益缩放 / MTP多令牌预测(共享嵌入+输出头) / 绑定嵌入。
维度对齐 MobileLLM-125M：hidden 576 / FFN 1536 / 30层 / 9头 / 3KV。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MoleculeConfig


def soft_cap(x, cap):
    """logit 软封顶（Gemma）：cap * tanh(x / cap)，平滑、梯度不裂。"""
    if cap is None or cap <= 0:
        return x
    return cap * torch.tanh(x / cap)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 数值稳：在 float32 算 norm，再回原 dtype
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x.to(dtype))


class RotaryEmbedding(nn.Module):
    """NTK-aware 动态 RoPE。训练时 pos<=train_seq_len 不缩放；推理更长才缩放。"""

    def __init__(self, cfg: MoleculeConfig):
        super().__init__()
        self.dim = cfg.head_dim
        self.base = cfg.rope_theta
        self.ntk = cfg.rope_ntk_scaling
        self.train_len = cfg.train_seq_len
        inv = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x, position_ids):
        # x: (B, n_heads, T, dim)
        B, H, T, D = x.shape
        max_pos = float(position_ids.max().item()) + 1
        if self.ntk and max_pos > self.train_len:
            scale = (max_pos / self.train_len) ** (self.dim / (self.dim - 2))
            eff_base = self.base * scale
            inv_freq = 1.0 / (eff_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
            inv_freq = inv_freq.to(x.device)
        else:
            inv_freq = self.inv_freq.to(x.device)
        pos = position_ids.float()  # (B, T)
        # freqs: (B, T, D/2)
        freqs = pos.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)  # (B, T, D)
        cos = emb.cos().unsqueeze(1)  # (B,1,T,D)
        sin = emb.sin().unsqueeze(1)
        return cos, sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: (B, H, T, D)
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    q = (q * cos + rotate_half(q) * sin)
    k = (k * cos + rotate_half(k) * sin)
    return q, k


class Attention(nn.Module):
    def __init__(self, cfg: MoleculeConfig):
        super().__init__()
        d = cfg.hidden_size
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.qk_norm = cfg.qk_norm
        self.attn_cap = cfg.attn_logit_cap
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)

    def forward(self, x, cos, sin, attention_mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)
        # GQA：KV 头 repeat 到 Q 头数
        k = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        # 注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T,T)
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = soft_cap(scores, self.attn_cap)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B,H,T,head_dim)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: MoleculeConfig):
        super().__init__()
        d = cfg.hidden_size
        f = cfg.intermediate_size
        self.gate = nn.Linear(d, f, bias=False)
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: MoleculeConfig, is_mtp=False):
        super().__init__()
        self.is_mtp = is_mtp
        self.attn = Attention(cfg)
        self.ffn = SwiGLU(cfg)
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)

    def forward(self, x, cos, sin, attention_mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MTPModule(nn.Module):
    """DeepSeek-V3 式 MTP：共享嵌入 + 共享输出头，仅多一个 Transformer 块 + 投影。"""

    def __init__(self, cfg: MoleculeConfig, shared_block: TransformerBlock):
        super().__init__()
        d = cfg.hidden_size
        self.norm = RMSNorm(d, cfg.norm_eps)
        self.norm_emb = RMSNorm(d, cfg.norm_eps)
        self.proj = nn.Linear(2 * d, d, bias=False)  # 合并 main hidden 与 next-token emb
        self.block = TransformerBlock(cfg)  # 独立 MTP 块
        # 不注册嵌入/输出头：由 MoleculeModel 传入共享引用

    def forward(self, x_main, emb_next, shared_lm_head, out_norm, cos, sin, attn_mask):
        h = self.norm(x_main)
        e = self.norm_emb(emb_next)
        y = self.proj(torch.cat([h, e], dim=-1))
        y = self.block(y, cos, sin, attn_mask)
        y = out_norm(y)
        return shared_lm_head(y)


class MoleculeModel(nn.Module):
    def __init__(self, cfg: MoleculeConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        if cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.embed.weight  # 绑定嵌入
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # MTP
        self.mtp = None
        if cfg.mtp_depth >= 1:
            self.mtp = MTPModule(cfg, self.blocks[0])
        # DeepNorm 残差增益
        self.alpha = (2 * cfg.num_hidden_layers) ** 0.25 if cfg.deepnorm else 1.0
        self._init_weights()

    def _init_weights(self):
        # 轻量 DeepNorm 风格 init：残差投影小缩放，稳 30 层
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if self.cfg.deepnorm:
                        m.weight.mul_(1.0 / (2 * self.cfg.num_hidden_layers) ** 0.25)

    def _causal_mask(self, T, device):
        # 返回 (1,1,T,T) 加性 mask（-inf 上三角）
        m = torch.full((T, T), float("-inf"), device=device)
        m = torch.triu(m, diagonal=1)
        return m.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids, position_ids=None, return_mtp=False):
        B, T = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        cos_sin = self._rope_cos_sin(input_ids, position_ids)
        cos, sin = cos_sin
        attn_mask = self._causal_mask(T, input_ids.device)
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x, cos, sin, attn_mask)
        x = self.final_norm(x)
        logits = soft_cap(self.lm_head(x), self.cfg.final_logit_cap)
        out = {"logits": logits}
        if return_mtp and self.mtp is not None:
            # MTP depth-1：用位置 i 的 hidden + 位置 i+1 的 token emb，预测 i+1 之后的 token
            emb_next = self.embed(input_ids[:, 1:])  # (B, T-1, d) 对应位置 1..T-1
            x_main = x[:, :-1]                       # (B, T-1, d) 对应位置 0..T-2
            cos2, sin2 = cos[:, :, :-1, :], sin[:, :, :-1, :]
            mtp_mask = self._causal_mask(T - 1, input_ids.device)
            mtp_logits = self.mtp(x_main, emb_next, self.lm_head, self.final_norm, cos2, sin2, mtp_mask)
            mtp_logits = soft_cap(mtp_logits, self.cfg.final_logit_cap)
            out["mtp_logits"] = mtp_logits
        return out

    def _rope_cos_sin(self, input_ids, position_ids):
        # 借用 Attention 里的 RoPE 逻辑：简单实现（避免循环依赖）
        B, T = input_ids.shape
        D = self.cfg.head_dim
        max_pos = float(position_ids.max().item()) + 1
        if self.cfg.rope_ntk_scaling and max_pos > self.cfg.train_seq_len:
            scale = (max_pos / self.cfg.train_seq_len) ** (D / (D - 2))
            base = self.cfg.rope_theta * scale
        else:
            base = self.cfg.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        inv_freq = inv_freq.to(input_ids.device)
        pos = position_ids.float().unsqueeze(-1)  # (B,T,1)
        freqs = pos * inv_freq.unsqueeze(0).unsqueeze(0)  # (B,T,D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (B,T,D)
        cos = emb.cos().unsqueeze(1)  # (B,1,T,D)
        sin = emb.sin().unsqueeze(1)
        return cos, sin

    @torch.no_grad()
    def generate(self, input_ids, max_new=64, temperature=0.9, top_p=0.9):
        self.eval()
        device = input_ids.device
        generated = input_ids.clone()
        T0 = input_ids.shape[1]
        for _ in range(max_new):
            out = self.forward(generated)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-5)
            # top-p 采样
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            logits[0, sorted_idx[0, mask[0]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat([generated, nxt], dim=1)
            if int(nxt) == self.cfg.vocab_size - 1:  # 假设末尾为 EOS
                break
        return generated[:, T0:]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    cfg = MoleculeConfig()
    m = MoleculeModel(cfg)
    print("params:", count_params(m) / 1e6, "M")
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    out = m(x, return_mtp=True)
    print("logits", out["logits"].shape, "mtp", out["mtp_logits"].shape)
