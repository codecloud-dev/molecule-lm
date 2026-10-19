"""离线聊天冒烟测试：无网、无检查点也能跑通完整推理路径。

直接构造 ByteLevelTok 兜底分词器 + 极小随机模型，验证
build_messages -> generate_text 全链路（用户侧对话路径）不崩、能产出文本。
不下载 Qwen 分词器、不加载训练检查点，秒级、纯离线。
"""

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def test_generate_offline():
    from config import MoleculeConfig
    from model import MoleculeModel
    from tokenizer import ByteLevelTok, ReducedTokenizer
    import generate as G

    # 1) 极小配置（不写大检查点）
    cfg = MoleculeConfig(
        vocab_size=262,       # 256 byte tokens + 6 特殊符
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )

    # 2) 离线 byte 兜底分词器（不联网），1:1 映射即可冒烟
    base = ByteLevelTok()
    n = cfg.vocab_size
    old2new = {i: i for i in range(n)}
    byte_new = {b: b for b in range(256)}
    tk = ReducedTokenizer(base, old2new, dict(old2new), byte_new)
    assert tk.vocab_size == n, "分词器词表与模型词表不一致"

    # 3) 随机小模型（eval 模式，不训练）
    model = MoleculeModel(cfg).eval()

    # 4) 跑完整对话路径：中文 + 英文 prompt 各一次
    for prompt in ["你好，你是谁？", "Hello, who are you?"]:
        full = G.build_messages(G.SYS_IDENTITY, [], prompt)
        enc = tk.encode(full)
        assert len(enc) > 0, "编码后 prompt 不应为空"

        out = G.generate_text(model, tk, full, max_new=32, temperature=0.8, top_p=0.9)
        assert isinstance(out, str) and len(out) > 0, "generate_text 应返回非空文本"
        # 解码往返不抛异常（乱码可接受，关键是能跑）
        _ = tk.decode(tk.encode(out))
        print(f"[smoke] prompt={prompt!r} -> 生成长度 {len(out)} 字符 (OK)")

    print("[PASS] test_generate_offline: 离线聊天路径可跑（无网/无检查点）")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_generate_offline()
