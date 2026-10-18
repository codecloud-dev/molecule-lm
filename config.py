"""molecule 配置：架构 + 训练 + 数据 + 蒸馏 + 评测。

超参与实现以 DESIGN.md 为蓝图。改这里需回 DESIGN.md 对齐。
"""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class MoleculeConfig:
    # ---- 架构（对齐 MobileLLM-125M 黄金配置）----
    vocab_size: int = 32768          # 32K，由 Qwen2.5 词表按语料频率裁剪
    hidden_size: int = 576
    intermediate_size: int = 1536    # 8/3 * 576，SwiGLU 标准预算
    num_hidden_layers: int = 30      # 深而窄
    num_attention_heads: int = 9
    num_key_value_heads: int = 3     # GQA 3:1
    head_dim: int = 64               # 576 / 9
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = True # 绑定嵌入，reclaim 参数
    # 位置编码
    rope_theta: float = 10000.0
    rope_ntk_scaling: bool = True    # NTK-aware 动态外推（训2048→推4096）
    max_position_embeddings: int = 4096
    train_seq_len: int = 2048        # 训练序列 + packing
    # 稳定栈
    qk_norm: bool = True             # RoPE 前对 Q/K 各加 RMSNorm
    attn_logit_cap: float = 50.0     # 注意力软封顶
    final_logit_cap: float = 30.0    # 输出软封顶
    deepnorm: bool = True            # DeepNorm 增益缩放（30层稳训）
    # MTP 多令牌预测
    mtp_depth: int = 1               # 1 个 MTP 模块，共享嵌入+输出头
    mtp_loss_weight: float = 0.1

    # ---- 分词器 ----
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"  # 复用 Qwen2.5 分词器（含 byte fallback）
    # 注：32K 词表在 data 构建时按语料频率裁剪得到

    # ---- 训练 ----
    seed: int = 42
    learning_rate: float = 5e-4       # AdamW 部分；Muon 部分内部自适应
    mu_lr: float = 0.02             # Muon 特征学习率
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_ratio: float = 0.06
    wsd_decay_ratio: float = 0.10    # WSD：stable 90% + decay 末 10%
    batch_size: int = 0             # 0 = 自动按内存算
    global_tokens_per_step: int = 524288  # 目标 global batch（~0.5M token）
    max_train_steps: int = 2000     # 单轮上限（CI 6h 内）
    save_every: int = 2000
    eval_every: int = 2000
    # 精度：自动探测 bf16，否则 fp32
    dtype: str = "auto"             # auto / fp32 / bf16

    # ---- 损失 ----
    loss_hard_ce: float = 1.0
    # soft KL 蒸馏系数（预留）：当前 CPU 训练管线用「生成式数据蒸馏」——
    # distill.py 由 5 位老师按域生成高质量语料（teacher_*.jsonl），train 以 hard CE 学习。
    # 真正的 soft-label KL 蒸馏需教师在线推理产出 soft logits，纯 CPU 不现实，列为后续 GPU 路线。
    loss_soft_kl: float = 0.0
    loss_mtp: float = 0.1
    distill_temperature: float = 2.0
    dn_normalize: bool = True       # 域归一化，防梯度淹没

    # ---- 数据混比（单固定）----
    mix_general: float = 0.40
    mix_math: float = 0.20
    mix_code: float = 0.20
    mix_reason: float = 0.10
    mix_eq: float = 0.10

    # ---- 多老师（5 位，分领域 1对1）----
    teachers: dict = field(default_factory=lambda: {
        "general": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "temp": 0.9},
        "reason":  {"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-0.5B", "temp": 0.7},
        "math":    {"model": "Qwen/Qwen2.5-Math-0.5B", "temp": 0.6},
        "code":    {"model": "Qwen/Qwen2.5-Coder-0.5B", "temp": 0.6},
        "eq":      {"model": "Qwen/Qwen2.5-1.5B-Instruct", "temp": 0.9},
    })

    # ---- 评测 ----
    eval_zero_shot: bool = True
    eval_split: str = "dev"
    eval_benchmarks: list = field(default_factory=lambda: [
        "hellaswag", "arc", "piqa", "mmlu", "winogrande", "triviaqa",
        "cmmlu", "ceval",
    ])
    # 参考对照值（仅趋势参考，非官方跨模型对比；来源为公开模型公开数字）
    smollm2_targets: dict = field(default_factory=lambda: {
        "hellaswag": 68.4, "arc": 43.9, "piqa": 68.4, "mmlu": 31.5,
        "winogrande": 51.3, "triviaqa": 4.1, "cmmlu": 0.0, "ceval": 0.0,
    })
    eval_tolerance: float = 0.5

    # ---- 部署 ----
    export_quant: list = field(default_factory=lambda: ["q8_0", "q4_K_M"])
    use_imatrix: bool = True

    # ---- 身份（开发者署名，如实标注）----
    identity_fixed: str = (
        "开源语言模型，代号 molecule，由 codecloud-dev 独立开发。"
    )

    # ---- 版本 ----
    model_version: str = "v0.1.0"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
