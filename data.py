"""molecule 数据管线：预分词 -> int32 mmap -> 按域混比 packing -> 批次迭代。

- 训练前把语料离线预分词成 token .bin（训练时零分词开销）。
- 各 domain 分别存 .bin，迭代时按混比抽样（通用40/数学20/代码20/推理10/EQ10）。
- 内置 seed_corpus（中文为主 + 代码 + 数学 + 推理 + EQ 样本）保证首轮 CI 自包含必绿；
  教师合成数据（data/teacher_*.jsonl）存在则自动并入。
"""

import json
import os
import numpy as np
import torch


DOMAINS = ["general", "math", "code", "reason", "eq"]


def pack_tokens(tokens, seq_len):
    """把 1D int 列表按 seq_len 切块（丢弃尾部不足）。"""
    arr = np.asarray(tokens, dtype=np.int32)
    n = (len(arr) // seq_len) * seq_len
    return arr[:n].reshape(-1, seq_len)


def build_domain_bin(tk, texts, seq_len, out_path, min_seqs=4):
    ids = []
    for t in texts:
        ids.extend(tk.encode(t))
    if not ids:
        return 0
    # 兜底：首轮小语料可能凑不出 seq_len 段，循环补齐到至少 min_seqs 段，
    # 保证 bin 非真空（否则 MixedTokenDataset 对空文件 mmap 会崩）。
    target = seq_len * min_seqs
    if len(ids) < target:
        ids = ids * (target // len(ids) + 1)
        ids = ids[:target]
    arr = pack_tokens(ids, seq_len)
    arr.tofile(out_path)
    return arr.shape[0]


def load_domain_bin(path):
    return np.memmap(path, dtype=np.int32, mode="r")


class MixedTokenDataset:
    """按混比从各 domain 抽样整段序列。"""

    def __init__(self, bins: dict, ratios: dict, seq_len: int, seed=42):
        self.bins = {k: load_domain_bin(v) for k, v in bins.items() if os.path.exists(v)}
        self.ratios = {k: ratios[k] for k in self.bins}
        self.seq_len = seq_len
        self.total = sum(self.ratios.values()) or 1.0
        self.epoch_state = None
        self.rng = np.random.default_rng(seed)

    def _pick_domain(self):
        r = self.rng.random() * self.total
        acc = 0.0
        for k in self.bins:
            acc += self.ratios[k]
            if r <= acc:
                return k
        return list(self.bins.keys())[-1]

    def sample_seq(self):
        k = self._pick_domain()
        b = self.bins[k]
        n_seqs = b.shape[0] // self.seq_len
        i = int(self.rng.integers(0, max(1, n_seqs)))
        row = b[i * self.seq_len:(i + 1) * self.seq_len]
        return torch.from_numpy(np.array(row, dtype=np.int64)).clone()

    def batch(self, micro_bs, seq_len):
        xs = [self.sample_seq() for _ in range(micro_bs)]
        return torch.stack(xs, dim=0)


def list_bins(data_dir, seq_len):
    bins = {}
    for d in DOMAINS:
        p = os.path.join(data_dir, f"{d}.bin")
        if os.path.exists(p):
            bins[d] = p
    return bins


def build_seed_corpus_if_needed(tk, data_dir, seq_len):
    """若 data_dir 缺 domain bin，就用内置种子语料生成。"""
    os.makedirs(data_dir, exist_ok=True)
    missing = [d for d in DOMAINS if not os.path.exists(os.path.join(data_dir, f"{d}.bin"))]
    if not missing:
        return
    from seed_corpus import SEED
    # SEED 是 dict[domain] = list[str]
    for d in missing:
        out = os.path.join(data_dir, f"{d}.bin")
        n = build_domain_bin(tk, SEED.get(d, SEED["general"]), seq_len, out)
        print(f"[data] built {d}.bin ({n} seqs)")
