#!/usr/bin/env python3
"""molecule GGUF 导出冒烟（不联网、不依赖训练检查点）。

验证「弱设备可跑」的关键前提：export_gguf.py 能产出结构合法、llama.cpp
可加载的 GGUF 文件。用极小随机模型 + 离线 byte 兜底分词器，秒级完成。

运行：
  python tests/test_export_gguf.py          # 直接跑
  pytest tests/test_export_gguf.py -q       # 或走 pytest
"""
import os
import sys
import subprocess
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from config import MoleculeConfig
from model import MoleculeModel
from safetensors.torch import save_model


def _make_tiny_config(path):
    cfg = MoleculeConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        tokenizer_name="Qwen/Qwen2.5-0.5B",  # 离线会退 byte 兜底
    )
    cfg.save(path)
    return cfg


def test_export_gguf_offline():
    with tempfile.TemporaryDirectory() as d:
        cfg_path = os.path.join(d, "config.json")
        ckpt_path = os.path.join(d, "model.safetensors")
        gguf_path = os.path.join(d, "molecule-test.gguf")
        empty_data = os.path.join(d, "data")  # 空目录 → 走 byte 兜底分词器
        os.makedirs(empty_data, exist_ok=True)

        cfg = _make_tiny_config(cfg_path)
        model = MoleculeModel(cfg)  # 随机初始化，无需检查点
        save_model(model, ckpt_path)

        # 跑导出脚本（subprocess，确保走真实 CLI 路径）。
        # 离线环境设 HF_HUB_OFFLINE=1，让分词器下载立即失败并走 byte 兜底，
        # 避免无网挂起（CI 有网不受影响）。
        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        r = subprocess.run(
            [sys.executable, "export_gguf.py",
             "--config", cfg_path, "--ckpt", ckpt_path,
             "--data_dir", empty_data, "--out", gguf_path],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
            env=env,
        )
        assert r.returncode == 0, f"export 失败:\n{r.stdout}\n{r.stderr}"
        assert os.path.getsize(gguf_path) > 0, "GGUF 文件为空"

        # 1) magic 头
        with open(gguf_path, "rb") as f:
            magic = f.read(4)
        assert magic == b"GGUF", f"magic 异常: {magic}"

        # 2) llama.cpp 可读回：张量齐全、架构元数据正确
        from gguf import GGUFReader
        reader = GGUFReader(gguf_path)
        names = {t.name for t in reader.tensors}
        for need in ["token_embd.weight", "blk.0.attn_q.weight", "output.weight"]:
            assert need in names, f"缺失张量 {need}（llama.cpp 无法加载）"

        # 架构元数据键存在性（数值 API 随 gguf 版本略有差异，仅校验键存在）
        for key in ["llama.block_count", "llama.vocab_size", "general.architecture"]:
            assert key in reader.fields, f"缺失元数据键 {key}"
        print("[PASS] test_export_gguf_offline: 离线导出合法 GGUF，llama.cpp 可加载"
              "（含 token_embd/output/blk.0.attn_q 张量与架构元数据）")


if __name__ == "__main__":
    test_export_gguf_offline()
    print("\nEXPORT GGUF TEST PASSED ✓")
