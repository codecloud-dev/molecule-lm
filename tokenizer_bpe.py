"""molecule v3.0 自研 byte-level BPE 分词器（词表 8192，零外部依赖）。

词表布局（严格固定，导出/续训/引擎共用同一布局）：
  [0, 256)     256 个 UTF-8 基础字节（byte-level 回退，永不 OOV）
  [256, 262)   6 个特殊 token：pad/unk/bos/eos/im_start/im_end
  [262, 278)   16 个单字节控制 token（模式开关，见内部设计文档）
  [278, 286)   8 个抽象思维 token（潜空间推理，不输出给用户）
  [286, 8192)  BPE 合并 token（按训练语料频率从高到低）

编码：UTF-8 -> 字节 id -> 按 merge 优先级合并（贪心，rank 最小优先）。
解码：id -> 字节串（特殊/控制/思维 token 用其字面文本表示），errors=replace 永不抛异常。
"""

import json
import os
from collections import Counter

VOCAB_SIZE = 8192
NUM_BYTES = 256
NUM_SPECIAL = 6
NUM_CONTROL = 16
NUM_THINK = 8
NUM_BASE = NUM_BYTES + NUM_SPECIAL + NUM_CONTROL + NUM_THINK   # 286

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<|im_start|>", "<|im_end|>"]
CONTROL_TOKENS = [
    "<|ctl_chat|>",      # 标准对话
    "<|ctl_flash|>",     # 极速
    "<|ctl_deep|>",      # 深思
    "<|ctl_meta|>",      # 元认知自检
    "<|ctl_tool|>",      # 工具调用
    "<|ctl_mem|>",       # 长时记忆
    "<|ctl_foresight|>", # 前瞻预演
    "<|ctl_pulse|>",     # 脉冲稀疏
    "<|ctl_rsv8|>", "<|ctl_rsv9|>", "<|ctl_rsv10|>", "<|ctl_rsv11|>",
    "<|ctl_rsv12|>", "<|ctl_rsv13|>", "<|ctl_rsv14|>", "<|ctl_rsv15|>",
]
THINK_TOKENS = [f"<|think{i}|>" for i in range(8)]

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "self_bpe_8192.json")


def _base_vocab():
    vocab = []
    for i in range(NUM_BYTES):
        vocab.append("<0x%02X>" % i)
    vocab.extend(SPECIAL_TOKENS)
    vocab.extend(CONTROL_TOKENS)
    vocab.extend(THINK_TOKENS)
    return vocab


class ByteBPETokenizer:
    def __init__(self, merges=None, path=DEFAULT_PATH):
        self.merges = list(merges or [])
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = _base_vocab()
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.vocab += ["<unused_%d>" % i for i in range(len(self.vocab), VOCAB_SIZE)]
        # 解码用：id -> bytes（合并 token 可能是连续多个 <0xHH> 的拼接）
        import re
        self._byteof = {}
        byte_re = re.compile(r"^(?:<0x[0-9A-F]{2}>)+$")
        for i, tok in enumerate(self.vocab):
            if byte_re.match(tok):
                self._byteof[i] = bytes(
                    int(m, 16) for m in re.findall(r"<0x([0-9A-F]{2})>", tok))
            elif i < NUM_BYTES:
                self._byteof[i] = bytes([i])
        self._special_ids = {
            "pad": 256, "unk": 257, "bos": 258, "eos": 259,
            "im_start": 260, "im_end": 261,
        }
        self.id_to_token = list(self.vocab)

    # ---------- 训练 ----------
    @staticmethod
    def train(texts, target_merges=VOCAB_SIZE - NUM_BASE, verbose=False):
        """标准 BPE 训练：词 = 字节序列，迭代合并最高频相邻对。"""
        words = []  # 每个词是 list[int]（字节 id）
        for t in texts:
            for piece in t.split(" "):
                b = piece.encode("utf-8")
                if b:
                    words.append([x + 0 for x in b])  # 字节值即 id（0..255）
        if not words:
            return []
        merges = []
        for step in range(target_merges):
            pairs = Counter()
            for w in words:
                for i in range(len(w) - 1):
                    pairs[(w[i], w[i + 1])] += 1
            if not pairs:
                break
            (a, b), cnt = pairs.most_common(1)[0]
            if cnt < 2:
                break
            new_id = NUM_BASE + len(merges)
            merges.append((a, b))
            repl = [a, b]
            nw = []
            for w in words:
                i, out = 0, []
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                nw.append(out)
            words = nw
            if verbose and (step + 1) % 500 == 0:
                print("bpe merge %d (cnt=%d)" % (step + 1, cnt))
        return merges

    # ---------- 编码 ----------
    def _merge_word(self, ids):
        """对单个词（字节 id 列表）按 rank 贪心合并。"""
        while len(ids) >= 2:
            best, bi = None, None
            for i in range(len(ids) - 1):
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            ids = ids[:bi] + [NUM_BASE + best] + ids[bi + 2:]
        return ids

    def encode(self, text):
        """整文本编码：空格作为普通字节参与合并，保证 roundtrip 无损。"""
        return self._merge_word(list(text.encode("utf-8")))

    encode_raw = encode  # 兼容别名

    # ---------- 解码 ----------
    def decode(self, ids, stop_at_im_end=False):
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i in self._byteof:
                buf += self._byteof[i]
            else:
                tok = self.vocab[i] if i < len(self.vocab) else ""
                if tok.startswith("<0x") and tok.endswith(">"):
                    buf += bytes([int(tok[3:-1], 16)])
                elif tok in ("<|im_start|>", "<|im_end|>"):
                    buf += tok.encode("utf-8")
                    if stop_at_im_end and tok == "<|im_end|>":
                        break
                elif tok.startswith("<|"):
                    buf += tok.encode("utf-8")
                # unused -> 跳过
        return buf.decode("utf-8", errors="replace")

    # ---------- 持久化 ----------
    def save(self, path=DEFAULT_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab_size": VOCAB_SIZE, "merges": [list(m) for m in self.merges]}, f)

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(merges=[tuple(m) for m in d["merges"]])

    @property
    def vocab_size(self):
        return VOCAB_SIZE


if __name__ == "__main__":
    import seed_corpus
    texts = []
    for dom, arr in seed_corpus.SEED.items():
        texts.extend(arr)
    tk = ByteBPETokenizer.train(texts, verbose=True)
    print("learned merges:", len(tk))
    tok = ByteBPETokenizer(merges=tk)
    s = "北京是中国的首都。The capital of France is Paris."
    ids = tok.encode_raw(s)
    print("ids:", ids[:20])
    print("roundtrip ok:", tok.decode(ids) == s)
    tok.save()
    print("saved ->", DEFAULT_PATH)
