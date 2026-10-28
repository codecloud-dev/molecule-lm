"""molecule v3.0 GGUF 导出器（官方 GGUFWriter，非手写二进制——规避 v1.0 损坏）。

设计：
  - 权重用官方 add_tensor，dtype 由 numpy 数组决定（默认 F16 半精度，可选 F32 精确源），
    张量字节由 GGUFWriter 计算，不存在 v1.0「声明字节 != 真实字节」的损坏路径。
  - q4/q8/1.58bit 的逐层混合精度压缩不在导出期完成，改由 MInfer 私有引擎在发版后处理（设计文档 §6）。
  - 架构信息写进 molecule.* 自定义 KV + general.block_count 等通用字段，MInfer 私有引擎读头自适配。
  - 词表 8192：256 字节 + 6 特殊 + 16 控制 token + 8 思维 token + BPE，保留给 v3.5/v4 的开关。

用法：
  python export_gguf_v3.py --ckpt checkpoints/v3.pt --out molecule-v3.gguf --quant f16
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from gguf import GGUFWriter, GGMLQuantizationType

import model_v3
from model_v3 import MoleculeV3
from tokenizer_bpe import ByteBPETokenizer


def build_token_records(tk: ByteBPETokenizer):
    """返回 (tokens:list[str], scores:list[float], types:list[int])。"""
    tokens, scores, types = [], [], []
    for i, tok in enumerate(tk.vocab):
        tokens.append(tok)
        # 特殊/控制/思维 token 标为控制类（type=2），其余普通（0）
        if tok.startswith("<|") or tok.startswith("<0x"):
            types.append(2)
        else:
            types.append(0)
        scores.append(0.0)
    return tokens, scores, types


def export(ckpt, out_path, quant="f16"):
    if not os.path.exists(ckpt):
        raise SystemExit("找不到检查点: %s（先跑 train_v3.py 训练）" % ckpt)
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = MoleculeV3(vocab_size=8192)
    model.load_state_dict(st["model"], strict=False)
    model.eval()

    tk = ByteBPETokenizer.load()
    arch = model.architecture()
    plan = model.layer_plan()

    # 量化说明：官方 add_tensor 的 raw_dtype 对 q8 不可靠（且不兼容 1D 张量块约束），
    # 最终 q4/q8/1.58bit 由 MInfer 私有引擎在发版后逐层混合精度完成（设计文档 §6）。
    # 这里只产出「通用可加载的半精度源」：f16 为默认发布格式（体积减半、无块约束），f32 为精确源。
    qtype = {"f16": GGMLQuantizationType.F16,
             "f32": None}[quant]
    writer = GGUFWriter(out_path, "molecule", use_temp_file=False)

    # ---- 通用 KV（llama.cpp / 通用工具可读基础信息） ----
    writer.add_string("general.name", "molecule-v3")
    writer.add_string("general.description", "molecule v3.0 统一熔铸混合骨干 (Mamba+BinaryAttention)")
    writer.add_string("general.authors", "codecloud-dev")
    writer.add_string("general.organization", "codecloud-dev")
    writer.add_string("general.basename", "molecule-v3")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_uint32("general.file_type", {"f16": 1, "f32": 0}[quant])
    writer.add_uint32("general.quantization_version", 1)
    writer.add_uint32("general.block_count", arch["n_layers"])
    writer.add_uint32("general.context_length", 2048)
    writer.add_uint32("general.vocab_size", arch["vocab_size"])
    writer.add_uint32("general.embedding_length", arch["d_model"])

    # ---- molecule.* 自定义架构（MInfer 私有引擎自适配，零配置加载） ----
    writer.add_string("molecule.arch", json.dumps(arch, ensure_ascii=False))
    writer.add_string("molecule.layer_plan", json.dumps(plan))
    writer.add_string("molecule.version", "3.0")
    writer.add_float32("molecule.deepnorm_alpha", float(arch["alpha"]))
    writer.add_uint32("molecule.n_mamba", arch["n_mamba"])
    writer.add_uint32("molecule.n_attn", arch["n_attn"])
    writer.add_uint32("molecule.d_inner", arch["d_inner"])
    writer.add_uint32("molecule.d_state", arch["d_state"])
    writer.add_uint32("molecule.dt_rank", arch["dt_rank"])
    writer.add_uint32("molecule.n_q", arch["n_q"])
    writer.add_uint32("molecule.n_kv", arch["n_kv"])
    writer.add_uint32("molecule.head_dim", arch["head_dim"])
    writer.add_uint32("molecule.ffn_dim", arch["ffn_dim"])
    writer.add_uint32("molecule.mamba_group", arch["mamba_group"])
    writer.add_uint32("molecule.attn_group", arch["attn_group"])
    writer.add_uint32("molecule.block_share", arch["block_share"])

    # ---- 分词器：8192 全量 + 256 字节回退 + 控制/思维 token ----
    tokens, scores, types = build_token_records(tk)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_tokenizer_pre("default")
    writer.add_bos_token_id(tk._special_ids["bos"])
    writer.add_eos_token_id(tk._special_ids["eos"])
    writer.add_uint32("tokenizer.ggml.unk_token_id", tk._special_ids["unk"])
    writer.add_uint32("tokenizer.ggml.pad_token_id", tk._special_ids["pad"])

    # ---- 张量（共享权重天然只存一次） ----
    sd = model.export_state()
    n_t = 0
    np_dtype = {"f16": np.float16, "f32": np.float32}[quant]
    for name, ten in sd.items():
        t = ten.detach().to(torch.float32).contiguous().numpy().astype(np_dtype)
        writer.add_tensor(name, t)        # dtype 由 numpy 数组决定（f16 半精度 / f32）
        n_t += 1
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print("[export] %s" % out_path)
    print("[export] tensors=%d  vocab=%d  bytes=%d  quant=%s"
          % (n_t, len(tokens), os.path.getsize(out_path), quant))
    print("[export] arch: %s" % json.dumps(arch, ensure_ascii=False))
    print("[export] layer_plan(%d): %s" % (len(plan), plan))
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v3.pt")
    ap.add_argument("--out", default="molecule-v3.gguf")
    ap.add_argument("--quant", default="f16", choices=["f16", "f32"])
    args = ap.parse_args()
    export(args.ckpt, args.out, args.quant)
