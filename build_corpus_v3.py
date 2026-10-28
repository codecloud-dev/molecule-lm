"""molecule v3.0 语料构建：内置高质量 seed + CI 时下载公开语料增强 → 预编码为 mmap .bin。

用法：
  python build_corpus_v3.py [--seq_len 256] [--no_download]

- 配比严格对齐设计文档 §7.2 Phase1：
    zh_general 60% / en_general 10% / code 20% / book 10%
- 分词器：data/self_bpe_8192.json（不存在则现场训练并保存）
- 下载：仅使用稳定、许可干净的公有领域来源（Project Gutenberg 等）；
  每个来源带超时与异常兜底，任一失败都不影响整体，自动回退到 seed 循环。
- 产出：data/v3_<domain>.bin（int32 mmap，训练零分词开销）
"""

import argparse
import os
import sys
import time
import urllib.request

import numpy as np

import seed_corpus_v3
from tokenizer_bpe import ByteBPETokenizer

DOMAINS = ["zh_general", "en_general", "code", "book"]
MIX = {"zh_general": 0.60, "en_general": 0.10, "code": 0.20, "book": 0.10}

# 稳定、公有领域的英文文本来源（Project Gutenberg 等），按 domain 归属。
# 失败自动跳过，绝不阻塞训练。
DOWNLOAD_SOURCES = {
    "en_general": [
        ("https://www.gutenberg.org/files/1342/1342-0.txt", 220_000),   # Pride and Prejudice
        ("https://www.gutenberg.org/files/11/11-0.txt", 200_000),       # Alice in Wonderland
        ("https://www.gutenberg.org/files/2701/2701-0.txt", 200_000),   # Moby-Dick (excerpt cap)
    ],
    "book": [
        ("https://www.gutenberg.org/files/84/84-0.txt", 200_000),       # Frankenstein
        ("https://www.gutenberg.org/files/1661/1661-0.txt", 200_000),   # The Adventures of Sherlock Holmes
    ],
    "zh_general": [
        # 中文公有领域/开放文本（若不可达则回退 seed，不阻塞）
        ("https://www.gutenberg.org/files/24028/24028-0.txt", 120_000),
    ],
}


def get_tokenizer():
    if os.path.exists("data/self_bpe_8192.json"):
        return ByteBPETokenizer.load()
    texts = []
    for d in DOMAINS:
        texts.extend(seed_corpus_v3.SEED.get(d, []))
    merges = ByteBPETokenizer.train(texts)
    tk = ByteBPETokenizer(merges=merges)
    os.makedirs("data", exist_ok=True)
    tk.save()
    print("[tok] trained BPE: %d merges (saved)" % len(tk.merges))
    return tk


def fetch(url, cap):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "molecule-lm/3.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            raw = r.read(cap).decode("utf-8", errors="ignore")
        return raw
    except Exception as e:  # 任一来源失败都不影响整体
        print("[dl] 跳过 %s -> %s" % (url, e))
        return ""


def collect_texts(domain, allow_download):
    texts = list(seed_corpus_v3.SEED.get(domain, []))
    if allow_download:
        for url, cap in DOWNLOAD_SOURCES.get(domain, []):
            t = fetch(url, cap)
            if t:
                texts.append(t)
                print("[dl] +%d chars from %s" % (len(t), url))
    return texts


def build_bins(tk, seq_len, allow_download, min_seqs_per_domain=400):
    os.makedirs("data", exist_ok=True)
    total_tokens = 0
    for dom in DOMAINS:
        texts = collect_texts(dom, allow_download)
        ids = []
        for t in texts:
            ids.extend(tk.encode(t))
        if not ids:
            print("[data] %-10s 空，跳过" % dom)
            continue
        # 过训不饱和（设计文档 §7.2）：循环补足到最小规模
        target = seq_len * min_seqs_per_domain
        if len(ids) < target:
            rep = (target // len(ids)) + 1
            ids = ids * rep
        n = (len(ids) // seq_len) * seq_len
        arr = np.asarray(ids[:n], dtype=np.int32).reshape(-1, seq_len)
        path = "data/v3_%s.bin" % dom
        arr.tofile(path)
        total_tokens += n
        print("[data] %-10s %8d tokens -> %s (%d seqs)" % (dom, n, path, arr.shape[0]))
    return total_tokens


def build(seq_len, allow_download=True):
    t0 = time.time()
    tk = get_tokenizer()
    print("[tok] BPE merges: %d" % len(tk.merges))
    total = build_bins(tk, seq_len, allow_download)
    print("[done] 总 token 数 ≈ %d，耗时 %.1fs" % (total, time.time() - t0))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--no_download", action="store_true", help="仅用内置 seed，不下载")
    args = ap.parse_args()
    build(args.seq_len, allow_download=not args.no_download)


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
