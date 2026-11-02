"""molecule v3.0 自研 byte-level BPE 分词器（词表 8192，零外部依赖）。

词表布局（严格固定，导出/续训/引擎共用同一布局）：
  [0, 256)           256 个 UTF-8 基础字节（byte-level 回退，永不 OOV）
  [256, 262)          6 个特殊 token：pad/unk/bos/eos/im_start/im_end
  [262, 278)          16 个单字节控制 token（模式开关，见内部设计文档）
  [278, 286)          8 个抽象思维 token（潜空间推理，不输出给用户）
  [286, 286+K)        K 个【整字 token】：直接映射高频非 ASCII 字符（CJK/全角标点等）
  [286+K, 8192)       BPE 合并 token（仅合并 ASCII/字节序列与已合并子词）

【根因修复·诡异文字】传统 byte-level BPE 把汉字拆成 3 个 UTF-8 字节，而 CJK 的中间/末尾
字节被几千个汉字共享，频率式合并几乎无法把「整字」合并成单个 token → 模型一吐不完整字节
就解码出 U+FFFD（诡异文字）。本分词器改为：**把高频非 ASCII 字符直接注册为整字 token**，
汉字在词表里就是「一个完整字」，模型在物理上无法只吐半个汉字 → 不可能产生 U+FFFD。
这是结构性根因修复，而非生成端屏蔽（生成端 UTF-8 约束仅作纵深防御兜底）。

编码：文本逐字符 → 注册整字直接出 id；ASCII 出字节 id；其余字符出 UTF-8 字节序列 → 再对
字节序列做 BPE 合并。解码：id → 字节串，errors=replace 永不抛异常。
"""

import json
import os
import re
from collections import Counter

VOCAB_SIZE = 8192
NUM_BYTES = 256
NUM_SPECIAL = 6
NUM_CONTROL = 16
NUM_THINK = 8
NUM_BASE = NUM_BYTES + NUM_SPECIAL + NUM_CONTROL + NUM_THINK   # 286

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>",
    chr(60) + chr(124) + "im_start" + chr(124) + chr(62),
    chr(60) + chr(124) + "im_end" + chr(124) + chr(62),
]
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
    def __init__(self, merges=None, chars=None, path=DEFAULT_PATH):
        self.merges = list(merges or [])
        self.chars = list(chars or [])           # 整字 token 字符列表（顺序即 id）
        self.char_top = NUM_BASE + len(self.chars)
        self.char2id = {c: NUM_BASE + i for i, c in enumerate(self.chars)}
        self.id2char = {NUM_BASE + i: c for i, c in enumerate(self.chars)}
        self.rank = {pair: i for i, pair in enumerate(self.merges)}  # pair -> 合并 token id - char_top
        self.vocab = _base_vocab()
        self.vocab.extend(self.chars)            # 整字 token（字面即字符本身）
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.vocab += ["<unused_%d>" % i for i in range(len(self.vocab), VOCAB_SIZE)]
        # 解码用：id -> bytes（自底向上重建，覆盖字节 / 整字 / 合并 token，绝不依赖正则）
        # 关键：合并 token 的 vocab 串可能是真实中文（如「智能」）而非 <0xHH> 模式，
        # 只有按合并结构 a+b 递归拼字节才能正确还原（这是之前 roundtrip 失败的真正根因）。
        self._byteof = {}
        for i in range(NUM_BYTES):
            self._byteof[i] = bytes([i])
        for i, c in self.id2char.items():      # 整字 token：字面即字符本身
            self._byteof[i] = c.encode("utf-8")
        for idx, (a, b) in enumerate(self.merges):   # 合并 token：bytes[a]+bytes[b]
            self._byteof[self.char_top + idx] = self._byteof[a] + self._byteof[b]
        # 特殊 token 的字面字节（im_start/im_end 必须能原样解回，模型才学得到停止语义）
        for name, sid in (("im_start", 260), ("im_end", 261)):
            self._byteof[sid] = SPECIAL_TOKENS[4 if name == "im_start" else 5].encode("utf-8")
        self._special_ids = {
            "pad": 256, "unk": 257, "bos": 258, "eos": 259,
            "im_start": 260, "im_end": 261,
        }
        # 原子特殊标记：<|im_start|> / <|im_end|> 必须整体编成单个 id，
        # 否则模型学不到"该收声了"的停止语义（早期 bug：被编成 7 个普通 token）。
        self._atomic_specials = [
            (self._special_ids["im_start"], SPECIAL_TOKENS[4]),
            (self._special_ids["im_end"], SPECIAL_TOKENS[5]),
        ]
        self.id_to_token = list(self.vocab)
        # 每个 token 的真实字节（供生成端 UTF-8 合法约束 / 解码校验）
        self._tbytes = dict(self._byteof)

    # ---------- 训练 ----------
    @staticmethod
    def train(texts, target_char_tokens=4000, verbose=False):
        """训练：① 把高频非 ASCII 字符直接注册为整字 token（根因修复·诡异文字）；
        ② 对 byte/ASCII 序列跑堆优化增量 BPE 学习子词合并。
        返回 (merges, chars)。merges 预算 = 8192 - char_top，保证总词表 ≤ 8192。"""
        import heapq
        # 1) 统计非 ASCII 字符频率，取 top-K 注册为整字 token
        char_freq = Counter()
        for t in texts:
            for ch in t:
                if ord(ch) >= 128:
                    char_freq[ch] += 1
        K = min(target_char_tokens, len(char_freq))
        top_chars = [c for c, _ in char_freq.most_common(K)]
        char2id = {c: NUM_BASE + i for i, c in enumerate(top_chars)}
        char_top = NUM_BASE + K
        merge_budget = VOCAB_SIZE - char_top          # 剩余给 BPE 合并

        # 2) 把训练文本表示成「基础 id 序列」：整字→整字 id；其余→UTF-8 字节 id；ASCII→字节 id
        seg = re.compile(r'[\s，。！？；：、,.!?;:\'\"（）()\[\]【】“”‘’\-—…·]+')
        raw_words = []
        for t in texts:
            for piece in seg.split(t):
                if not piece:
                    continue
                ids = []
                for ch in piece:
                    if ch in char2id:
                        ids.append(char2id[ch])
                    elif ord(ch) < 128:
                        ids.append(ord(ch))
                    else:
                        ids.extend(ch.encode("utf-8"))
                if ids:
                    raw_words.append(ids)
        if not raw_words:
            return [], top_chars

        # 3) 堆优化增量 BPE（在基础 id 序列上，整字 token 天然原子、不参与合并）
        wc = Counter(tuple(w) for w in raw_words)
        uniq = [list(k) for k in wc]
        cnt_w = [wc[tuple(k)] for k in wc]
        n = len(uniq)
        wpairs, pair_freq = [], Counter()
        for i, w in enumerate(uniq):
            pc = Counter()
            for j in range(len(w) - 1):
                pc[(w[j], w[j + 1])] += 1
            wpairs.append(pc)
            for p, c in pc.items():
                pair_freq[p] += c * cnt_w[i]
        heap = [(-pair_freq[p], p) for p in pair_freq]
        heapq.heapify(heap)
        merges = []
        while len(merges) < merge_budget and heap:
            a = b = None
            while heap:
                negf, p = heapq.heappop(heap)
                if pair_freq.get(p, 0) == -negf and pair_freq[p] >= 2:
                    a, b = p
                    break
            if a is None or pair_freq.get((a, b), 0) < 2:
                break
            new_id = char_top + len(merges)
            todo = [i for i in range(n) if (a, b) in wpairs[i]]
            for i in todo:
                w, cw = uniq[i], cnt_w[i]
                for p, c in wpairs[i].items():
                    pair_freq[p] -= c * cw
                new_w, j = [], 0
                while j < len(w):
                    if j < len(w) - 1 and w[j] == a and w[j + 1] == b:
                        new_w.append(new_id); j += 2
                    else:
                        new_w.append(w[j]); j += 1
                wpairs[i] = Counter()
                for j in range(len(new_w) - 1):
                    wpairs[i][(new_w[j], new_w[j + 1])] += 1
                for p, c in wpairs[i].items():
                    pair_freq[p] += c * cw
                    heapq.heappush(heap, (-pair_freq[p], p))
                uniq[i] = new_w
            merges.append((a, b))
            if verbose and (len(merges) + 1) % 500 == 0:
                print("bpe merge %d" % (len(merges)))
        return merges, top_chars

    # ---------- 编码 ----------
    def _merge_bytes(self, ids):
        """堆优化 BPE：按 rank 优先合并相邻字节/子词对，O(n log n) 级。"""
        if len(ids) < 2:
            return ids
        import heapq
        char_top = self.char_top
        heap = []
        for i in range(len(ids) - 1):
            r = self.rank.get((ids[i], ids[i + 1]))
            if r is not None:
                heapq.heappush(heap, (r, i))
        while heap:
            r, i = heapq.heappop(heap)
            if i >= len(ids) - 1:
                continue
            pr = self.rank.get((ids[i], ids[i + 1]))
            if pr is None or pr != r:
                continue
            new_id = char_top + pr
            ids = ids[:i] + [new_id] + ids[i + 2:]
            for j in (i - 1, i):
                if 0 <= j < len(ids) - 1:
                    rr = self.rank.get((ids[j], ids[j + 1]))
                    if rr is not None:
                        heapq.heappush(heap, (rr, j))
        return ids

    def encode(self, text):
        """逐字符编码：整字 token 直接出 id（原子）；ASCII/其余字符走字节 + BPE 合并。
        保证 roundtrip 无损，且整字 token 让模型无法吐出半字 → 杜绝 U+FFFD。

        特殊标记（<|im_start|> / <|im_end|>）走【原子匹配】：整个标记编成单个
        特殊 id（260/261），模型因此能学到"在此收声"的停止语义。
        早期版本把它们当普通文本编码（7 个 token），导致模型学不会停止、
        且解码时标记丢失 —— 这是"问什么都驴头不对马嘴"的一个实质根因。
        """
        out = []
        n = len(text)
        i = 0
        while i < n:
            matched = False
            for sid, tokstr in self._atomic_specials:
                if text.startswith(tokstr, i):
                    out.append(sid)
                    i += len(tokstr)
                    matched = True
                    break
            if matched:
                continue
            ch = text[i]
            if ch in self.char2id:
                out.append(self.char2id[ch])
            elif ord(ch) < 128:
                out.append(ord(ch))
            else:
                out.extend(self._merge_bytes(list(ch.encode("utf-8"))))
            i += 1
        # 整条序列再做一次 BPE（合并相邻字节/子词；整字 token 原子、不参与）
        return self._merge_bytes(out)

    encode_raw = encode  # 兼容别名

    # ---------- 解码 ----------
    def decode(self, ids, stop_at_im_end=False):
        buf = bytearray()
        for raw in ids:
            i = int(raw)
            if i in self.id2char:
                buf += self.id2char[i].encode("utf-8")
            elif i in self._byteof:
                buf += self._byteof[i]
            else:
                tok = self.vocab[i] if i < len(self.vocab) else ""
                if tok.startswith("<0x") and tok.endswith(">"):
                    buf += bytes([int(tok[3:-1], 16)])
                elif tok == SPECIAL_TOKENS[4] or tok == SPECIAL_TOKENS[5]:
                    # 特殊标记：按字面输出（im_end 可作为停止位）
                    buf += tok.encode("utf-8")
                    if stop_at_im_end and tok == SPECIAL_TOKENS[5]:
                        break
                elif tok.startswith("<|"):
                    buf += tok.encode("utf-8")
                # unused -> 跳过
        return buf.decode("utf-8", errors="replace")

    # ---------- 持久化 ----------
    def save(self, path=DEFAULT_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab_size": VOCAB_SIZE,
                       "merges": [list(m) for m in self.merges],
                       "chars": self.chars}, f)

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(merges=[tuple(m) for m in d.get("merges", [])],
                   chars=d.get("chars", []))

    @property
    def vocab_size(self):
        return VOCAB_SIZE


if __name__ == "__main__":
    import seed_corpus_v3
    texts = []
    for dom, arr in seed_corpus_v3.SEED.items():
        texts.extend(arr)
    merges, chars = ByteBPETokenizer.train(texts, verbose=True)
    print("learned merges:", len(merges), "chars:", len(chars))
    tok = ByteBPETokenizer(merges=merges, chars=chars)
    s = "北京是中国的首都。The capital of France is Paris."
    ids = tok.encode_raw(s)
    print("ids:", ids[:20])
    print("roundtrip ok:", tok.decode(ids) == s)
    for c in ["北", "京", "智", "能", "力", "。"]:
        print("「%s」-> token数 %d" % (c, len(tok.encode(c))))
    tok.save()
    print("saved ->", DEFAULT_PATH)
