"""molecule 分词器：复用 Qwen2.5 分词器，按语料频率裁剪到 32K，绑定到模型。

- 始终保留：特殊符 + 全部 256 个 byte 回退 token（永不乱码）。
- 其余内容 token 按训练语料词频取最高频，凑到 vocab_size。
- byte 回退：词表外字符用 Qwen 自带 byte token 逐字节编码。
"""

import json
import os
import re
import torch
from collections import Counter


class ByteLevelTok:
    """离线 byte-level 兜底分词器（GPT-2 风格）。

    当 Qwen2.5 分词器因网络不可达而下不下来时启用，保证训练/评测/导出
    全链路仍可跑（CI 韧性 + 纯离线自测）。它不是正式分词器——正式训练
    始终优先用 Qwen2.5，仅在联网失败时才退到这层。
    """

    SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>", "<s>", "</s>"]

    def __init__(self):
        self._b2u = self._bytes_to_unicode()
        self._vocab = {}
        for b in range(256):
            self._vocab[self._b2u[b]] = b
        for i, s in enumerate(self.SPECIALS):
            self._vocab[s] = 256 + i
        self._id2tok = {v: k for k, v in self._vocab.items()}

    @staticmethod
    def _bytes_to_unicode():
        bs = (list(range(ord("!"), ord("~") + 1)) +
              list(range(ord("¡"), ord("¬") + 1)) +
              list(range(ord("®"), ord("ÿ") + 1)))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=False):
        return [b for b in text.encode("utf-8", errors="replace")]

    def decode(self, ids):
        bs = bytes([int(i) for i in ids if 0 <= int(i) <= 255])
        return bs.decode("utf-8", errors="replace")

    def id_to_token(self, oid):
        return self._id2tok.get(oid, "")


def get_base_tokenizer(name):
    """优先加载 Qwen2.5 分词器；联网失败则退到离线 byte-level 兜底。

    返回 (tokenizer, is_fallback)。正式训练/导出应使用 is_fallback=False 的路径。
    """
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name), False
    except Exception as e:
        print(f"[tok] 无法下载 {name}（{e}），退到离线 byte-level 兜底分词器。"
              "正式训练请保证能联网加载 Qwen2.5 分词器。")
        return ByteLevelTok(), True


class ReducedTokenizer:
    """把 Qwen2.5 大词表缩减到 target_size，并提供 encode/decode（new id 空间）。"""

    SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>", "<s>", "</s>"]

    def __init__(self, base_tokenizer, old2new, new2old, byte_new_ids):
        self.base = base_tokenizer
        self.old2new = old2new      # qwen id -> new id
        self.new2old = new2old      # new id -> qwen id
        self.byte_new = byte_new_ids  # byte value -> new id
        self.vocab_size = len(new2old)

    @classmethod
    def build_from_corpus(cls, base_tokenizer, corpus, target_size=32768, cache_path=None):
        vocab = base_tokenizer.get_vocab()
        # 特殊符 + byte token 始终保留
        keep = []
        keep_set = set()
        for tok in cls.SPECIAL_TOKENS:
            if tok in vocab and vocab[tok] not in keep_set:
                keep.append(vocab[tok]); keep_set.add(vocab[tok])
        # byte token：<0x00>..<0xFF>
        byte_old = {}
        for i in range(256):
            key = f"<0x{i:02X}>"
            if key in vocab:
                byte_old[i] = vocab[key]
                if vocab[key] not in keep_set:
                    keep.append(vocab[key]); keep_set.add(vocab[key])
        # 词频统计（排除已保留的）
        cnt = Counter()
        for text in corpus:
            ids = base_tokenizer.encode(text, add_special_tokens=False)
            for i in ids:
                if i not in keep_set:
                    cnt[i] += 1
        # 取最高频 content token 补齐
        need = target_size - len(keep)
        top = [i for i, _ in cnt.most_common(need)]
        selected = keep + top
        selected = selected[:target_size]
        old2new = {old: nid for nid, old in enumerate(selected)}
        new2old = {nid: old for old, nid in old2new.items()}
        byte_new = {b: old2new[o] for b, o in byte_old.items()}
        tk = cls(base_tokenizer, old2new, new2old, byte_new)
        if cache_path:
            tk.save(cache_path)
        return tk

    @classmethod
    def load(cls, base_tokenizer, cache_path):
        d = json.load(open(cache_path, "r", encoding="utf-8"))
        byte_new = {int(k): v for k, v in d["byte_new"].items()}
        tk = cls(base_tokenizer, d["old2new"], {int(k): v for k, v in d["new2old"].items()}, byte_new)
        return tk

    def save(self, cache_path):
        json.dump({
            "old2new": self.old2new,
            "new2old": self.new2old,
            "byte_new": self.byte_new,
        }, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)

    def encode(self, text):
        old_ids = self.base.encode(text, add_special_tokens=False)
        out = []
        for oid in old_ids:
            if oid in self.old2new:
                out.append(self.old2new[oid])
            else:
                # byte 回退：把该 piece 的字节逐字节编码
                piece = self.base.id_to_token(oid) or ""
                for b in piece.encode("utf-8", errors="replace"):
                    out.append(self.byte_new.get(b, 0))
        return out

    def decode(self, new_ids):
        old_ids = [self.new2old.get(int(i), 0) for i in new_ids]
        return self.base.decode(old_ids)

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]
