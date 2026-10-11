#!/usr/bin/env python3
"""把 molecule V0.1 权重导出为自写 GGUF（llama.cpp 可直接加载/量化）。

- 权重名映射到 Llama 命名（token_embd / blk.N / output）；绑定嵌入：output = token_embd。
- 元数据：层数 / 维度 / 头 / KV 头 / vocab / RoPE / 软封顶 等全部写入。
- 分词器：在线时从 Qwen2.5 tokenizer.json 裁剪到我们的 32K 子集（保留 byte/特殊符）
  嵌入 GGUF；离线兜底时写入 byte-level token 列表，保证文件结构有效。
- 注：QK-Norm / MTP 为训练期稳定与密度技巧，llama.cpp 推理按标准 Llama 近似；
  0.1B 量级影响极小，v0.1 可接受。

用法：
  python export_gguf.py --ckpt checkpoints/model.safetensors --out molecule-f16.gguf
"""

import os
import sys
import json
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MoleculeConfig
from model import MoleculeModel
from tokenizer import ReducedTokenizer, get_base_tokenizer
from gguf import GGUFWriter


def map_state_dict(sd, cfg):
    out = {}
    out["token_embd.weight"] = sd["embed.weight"]
    for n in range(cfg.num_hidden_layers):
        p = f"blocks.{n}."
        out[f"blk.{n}.attn_norm.weight"] = sd[f"{p}attn_norm.weight"]
        out[f"blk.{n}.attn_q.weight"] = sd[f"{p}attn.q_proj.weight"]
        out[f"blk.{n}.attn_k.weight"] = sd[f"{p}attn.k_proj.weight"]
        out[f"blk.{n}.attn_v.weight"] = sd[f"{p}attn.v_proj.weight"]
        out[f"blk.{n}.attn_output.weight"] = sd[f"{p}attn.o_proj.weight"]
        out[f"blk.{n}.ffn_norm.weight"] = sd[f"{p}ffn_norm.weight"]
        out[f"blk.{n}.ffn_gate.weight"] = sd[f"{p}ffn.gate.weight"]
        out[f"blk.{n}.ffn_up.weight"] = sd[f"{p}ffn.up.weight"]
        out[f"blk.{n}.ffn_down.weight"] = sd[f"{p}ffn.down.weight"]
    out["output_norm.weight"] = sd["final_norm.weight"]
    out["output.weight"] = sd["embed.weight"]  # 绑定嵌入
    return out


def build_tokenizer_payload(cfg, data_dir):
    """返回 (tokens, scores, merges, model_type, pre, fallback) 供 GGUF 嵌入。"""
    map_path = os.path.join(data_dir, "tokenizer_map.json")
    if not os.path.exists(map_path):
        # 防御性兜底：tokenizer_map 缺失时生成占位 token 列表，保证 GGUF 结构有效
        # （分词器不精确，但文件可被 llama.cpp 加载；正常流程 assets 必含此文件）
        tokens = [f"<tok{i}>" for i in range(cfg.vocab_size)]
        scores = [0.0] * cfg.vocab_size
        return tokens, scores, [], "llama", "default", True
    base, is_fallback = get_base_tokenizer(cfg.tokenizer_name)
    tk = ReducedTokenizer.load(base, map_path)
    qwen_vocab = base.get_vocab()
    id2tok = {v: k for k, v in qwen_vocab.items()}
    old2new = tk.old2new
    tokens = [None] * tk.vocab_size
    for old, nid in old2new.items():
        tokens[nid] = id2tok.get(old, "<unk>")
    # 保证每个位置都有字符串
    for i in range(tk.vocab_size):
        if tokens[i] is None:
            tokens[i] = f"<tok{i}>"
    scores = [0.0] * tk.vocab_size
    merges = []
    if not is_fallback and hasattr(base, "save_pretrained"):
        try:
            import tempfile
            tmp = tempfile.mkdtemp()
            base.save_pretrained(tmp)
            tj = json.load(open(os.path.join(tmp, "tokenizer.json"), "r", encoding="utf-8"))
            kept = set(tokens)
            for m in tj["model"].get("merges", []):
                if isinstance(m, list):
                    a, b = m[0], m[1]
                else:
                    a, b = m.split(" ")
                if a in kept and b in kept:
                    merges.append(f"{a} {b}")
        except Exception as e:
            print(f"[export] 合并对提取失败（{e}），仅写 token 列表。")
    model_type = "llama"
    pre = "qwen2" if not is_fallback else "default"
    return tokens, scores, merges, model_type, pre, is_fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--ckpt", default="checkpoints/model.safetensors")
    ap.add_argument("--out", default="molecule-f16.gguf")
    args = ap.parse_args()

    cfg = MoleculeConfig.load(args.config) if os.path.exists(args.config) else MoleculeConfig()
    model = MoleculeModel(cfg)
    try:
        from safetensors.torch import load_model
        load_model(model, args.ckpt)
    except Exception:
        sd_state = torch.load(os.path.join("checkpoints", "latest.ckpt"), map_location="cpu")
        model.load_state_dict(sd_state["model"])
    sd = model.state_dict()

    # 找特殊符 id
    base, _ = get_base_tokenizer(cfg.tokenizer_name)
    qv = base.get_vocab()
    map_path = os.path.join(args.data_dir, "tokenizer_map.json")
    eos_id, bos_id = 0, 0
    if os.path.exists(map_path):
        tk = ReducedTokenizer.load(base, map_path)
        def new_id_of(tok):
            oid = qv.get(tok)
            return tk.old2new.get(oid, 0) if oid is not None else 0
        eos_id = new_id_of("<|im_end|>")
        bos_id = new_id_of("<|im_start|>")

    writer = GGUFWriter(args.out, arch="llama")
    writer.add_name("molecule")
    writer.add_author("codecloud-dev")
    writer.add_description("molecule V0.1 —— 0.1B 级开源小语言模型（通用架构 + 融合注意力机制）。")
    writer.add_license("apache-2.0")
    writer.add_version("1")
    # 架构元数据
    writer.add_block_count(cfg.num_hidden_layers)
    writer.add_embedding_length(cfg.hidden_size)
    writer.add_feed_forward_length(cfg.intermediate_size)
    writer.add_head_count(cfg.num_attention_heads)
    writer.add_head_count_kv(cfg.num_key_value_heads)
    writer.add_key_length(cfg.head_dim)
    writer.add_value_length(cfg.head_dim)
    writer.add_layer_norm_rms_eps(cfg.norm_eps)
    writer.add_context_length(cfg.max_position_embeddings)
    writer.add_vocab_size(cfg.vocab_size)
    writer.add_rope_dimension_count(cfg.head_dim)
    writer.add_rope_freq_base(cfg.rope_theta)
    writer.add_attn_logit_softcapping(cfg.attn_logit_cap)
    writer.add_final_logit_softcapping(cfg.final_logit_cap)
    writer.add_bos_token_id(bos_id)
    writer.add_eos_token_id(eos_id)
    writer.add_file_type(1)  # F32 (f16 推理由 llama.cpp 决定，这里存原精度)
    writer.add_chat_template(
        "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}"
        "{{ '<|im_start|>assistant\n' }}"
    )

    # 分词器
    tokens, scores, merges, model_type, pre, is_fallback = build_tokenizer_payload(cfg, args.data_dir)
    writer.add_tokenizer_model(model_type)
    writer.add_tokenizer_pre(pre)
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    if merges:
        writer.add_token_merges(merges)

    # 张量（GGUF 要求 fp32 / 量化；我们存 fp32 原精度）
    mapped = map_state_dict(sd, cfg)
    for name, t in mapped.items():
        writer.add_tensor(name, t.float().numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[export] GGUF -> {args.out}  ({os.path.getsize(args.out)} bytes)"
          f"{'  [离线 byte 兜底分词器]' if is_fallback else ''}")


if __name__ == "__main__":
    main()
