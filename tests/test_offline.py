#!/usr/bin/env python3
"""molecule 离线自测（不联网、不依赖 checkpoint）。

覆盖四个关键正确性点：
  1. 模型前向：logits 形状正确且数值有限。
  2. 字节级兜底分词器：中英文往返一致（离线，无需下载 Qwen）。
  3. 空 bin 兜底：小语料也能产出非空 .bin（修复首轮 mmap 崩溃）。
  4. KV 缓存生成：forward_cached 首步结果与全量前向一致（校验 790c20f 重构）。

运行：
  python tests/test_offline.py          # 直接跑，打印 PASS/FAIL
  pytest tests/test_offline.py -q       # 或走 pytest
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from config import MoleculeConfig
from model import MoleculeModel
from tokenizer import ByteLevelTok
from data import build_domain_bin, load_domain_bin


def test_config():
    cfg = MoleculeConfig()
    assert cfg.vocab_size == 2500
    assert cfg.hidden_size == 576
    assert cfg.num_hidden_layers == 30
    assert cfg.num_attention_heads == 9
    assert cfg.num_key_value_heads == 3  # GQA 3:1
    assert isinstance(cfg.identity_fixed, str) and "codecloud-dev" in cfg.identity_fixed
    assert cfg.mtp_depth >= 1
    print("[PASS] test_config: 架构/身份常量符合 DESIGN")


def test_forward_finite():
    cfg = MoleculeConfig()
    model = MoleculeModel(cfg).eval()
    B, T = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        out = model(ids, return_mtp=True)
    logits = out["logits"]
    assert logits.shape == (B, T, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    if cfg.mtp_depth >= 1:
        assert "mtp_logits" in out
        assert torch.isfinite(out["mtp_logits"]).all()
    print("[PASS] test_forward_finite: 前向 logits 形状/有限性 OK（含 MTP）")


def test_byte_level_roundtrip():
    tk = ByteLevelTok()
    samples = ["你好，世界！", "hello molecule", "molecule 由 codecloud-dev 独立开发。", "🙂"]
    for s in samples:
        ids = tk.encode(s)
        back = tk.decode(ids)
        assert back == s, f"往返不一致: {s!r} -> {back!r}"
        assert all(0 <= i < len(tk.get_vocab()) for i in ids)
    print("[PASS] test_byte_level_roundtrip: 字节级兜底分词器中英文/emoji 往返一致")


def test_build_domain_bin_nonempty():
    # 极小语料：若没有兜底会切成 0 段导致后续 mmap 崩溃
    tk = ByteLevelTok()
    texts = ["你好世界", "hello"]
    seq_len, min_seqs = 8, 4
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tiny.bin")
        n = build_domain_bin(tk, texts, seq_len, p, min_seqs=min_seqs)
        assert n >= min_seqs, f"兜底应至少产出 {min_seqs} 段，实际 {n}"
        assert os.path.getsize(p) > 0
        arr = load_domain_bin(p)  # 必须能 mmap 不崩（1D 扁平，按 stride 索引）
        assert arr.ndim == 1
        assert arr.shape[0] == n * seq_len
    print(f"[PASS] test_build_domain_bin_nonempty: 小语料兜底产出 {n} 段且可 mmap")


def test_kv_cache_matches_full():
    cfg = MoleculeConfig()
    model = MoleculeModel(cfg).eval()
    T0 = 8
    ids = torch.randint(0, cfg.vocab_size, (1, T0))
    pos = torch.arange(T0).unsqueeze(0)
    past = [None] * cfg.num_hidden_layers
    with torch.no_grad():
        full = model(ids)["logits"]  # 全量前向
        cached, past = model.forward_cached(ids, pos, past)  # KV 首步
    # KV 缓存首步应与全量前向的最后一位 logits 完全一致
    assert torch.allclose(full[:, -1, :], cached[:, -1, :], atol=1e-5), "KV 首步与全量前向不一致"
    # 续生成若干 token，验证 KV 路径不崩且输出合法
    gen = model.generate(ids, max_new=6, temperature=1.0)
    assert gen.shape[0] == 1 and 0 < gen.shape[1] <= 7
    assert int(gen.max()) < cfg.vocab_size and int(gen.min()) >= 0
    print(f"[PASS] test_kv_cache_matches_full: KV 缓存与全量一致，生成 {gen.shape[1]} token 合法")


def test_v3_step_matches_full():
    """v3.0 增量 step（O(L) 流式）必须与全量 forward 数值一致。

    这条测试专门守护 v3 增量路径曾踩过的四类坑：
      - Mamba 分块扫描的因果掩码维度、增量卷积取指位置
      - BinaryAttention 增量缺 norm_attn / 维度 / 残差+FFN
      - 权重共享 ≠ 状态共享：递归状态必须逐层独立（同组相邻层输入不同）
    """
    from model_v3 import MoleculeV3
    torch.manual_seed(0)
    model = MoleculeV3().eval()
    T0 = 20
    ids = torch.randint(0, model.vocab_size, (1, T0))   # 不依赖分词器，纯随机序列
    with torch.no_grad():
        full = model(ids)["logits"][0, -1]               # 全量最后一位
        st = model.init_state(1)
        lg = None
        for t in ids[0]:
            lg = model.step(t.view(1, 1), st)[0, 0]       # 增量逐 token
    diff = (full - lg).abs().max().item()
    assert diff < 1e-4, "v3 增量 step 与全量 forward 不一致: %.3e" % diff
    print("[PASS] test_v3_step_matches_full: 增量 step 与全量 forward 一致 (max|Δ|=%.2e)" % diff)


def _run_all():
    test_config()
    test_forward_finite()
    test_byte_level_roundtrip()
    test_build_domain_bin_nonempty()
    test_kv_cache_matches_full()
    test_v3_step_matches_full()
    print("\nALL OFFLINE TESTS PASSED ✓")


if __name__ == "__main__":
    _run_all()
