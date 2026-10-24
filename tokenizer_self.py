"""molecule 自研字表分词器（v2.0 起替代 Qwen 裁剪方案）。

设计目标（对齐用户「自研字表 + 表情包 + 完全适配深度融合」）：
- 完全离线、零外部依赖：不再下载 Qwen，词表字符串直接落盘。
- 字节回退完备：内置 <0x00>..<0xFF> 256 个 byte token，词表外字符逐字节编码，永不乱码。
- 高频优先：基于训练语料（种子语料 + 教师数据若已生成）统计中文字/词、英文子词频率，
  取最高频凑到 target_size，并强制保留标点/数字/常用英文词/表情包。
- 与 llama.cpp 对齐：导出时使用 tokenizer.ggml.pre="llama3"（无 byte-fallback 之外的特殊预分词），
  并把 <0xXX> 作为 byte token 写入词表；编码走纯查表 + 字节兜底，导出零 Qwen2 依赖。

词表布局（new id 空间）：
  0      .. 5     : 6 个特殊符
  6      .. 261   : <0x00>..<0xFF> 字节回退（256 个）
  262    .. 261+N : 标点/数字/英文常用词/表情包（固定优先集）
  262+N  .. V-1   : 语料高频中文字 + 词（频率排序）
"""

import json
import re
from collections import Counter

# ---- 固定优先保留集（始终进词表，保证基础覆盖） ----
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>", "<s>", "</s>"]

# 常用标点 + 空白 + 数字（覆盖中英文基础符号）
PUNCT = list(" \n\t") + list("，。！？；：、（）《》「」『』""''…—·0123456789") + \
        list(".,!?;:'\"()[]{}@#$%^&*-+=/\\|~`<>") + list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

# 常用英文词 / 子词（小写，推理时大小写分别映射；这里保留小写基形 + 常见大写首字母形态）
ENGLISH = [
    "the","a","an","is","are","was","were","be","to","of","and","or","in","on","for",
    "with","that","this","it","you","we","they","he","she","i","me","my","your","our",
    "not","no","yes","do","does","did","have","has","had","will","would","can","could",
    "should","what","who","how","why","when","where","which","all","any","some","one",
    "two","three","first","new","good","bad","time","day","year","world","people","data",
    "model","language","learning","computer","system","code","function","python","the ",
    " is"," to"," of"," and"," in"," a "," on"," for"," with"," that"," we"," you"," our",
    "ing","ed","ly","er","tion","ment","ness","un","re","pre","pro","sub","over","under",
]

# 表情包（常用 emoji，作为独立 token；用户要求「多添加表情包」）
EMOJIS = [
    "😀","😁","😂","🤣","😊","😍","😘","😎","🤔","😅","😭","😡","🥺","😴","🤗","🙄",
    "👍","👎","👏","🙏","💪","🤝","👌","✌️","🤞","❤️","💔","💕","💖","🔥","⭐","✨",
    "🌟","💡","🎉","🎊","🚀","💡","⚡","🌈","☀️","🌙","🌍","🌏","🌎","🍎","🍊","🍉",
    "🍞","🍜","🍚","🥟","🍵","☕","🍺","🍷","🐱","🐶","🐼","🦄","🌸","🌹","🌻","🍀",
    "💰","💎","📱","💻","📚","📝","📌","✅","❌","⚠️","🚨","💯","🆗","🉐","🈶","🈚",
    "😄","😆","😉","😋","😜","🤩","😏","😢","😤","😱","🥰","😇","🤓","🧐","😬","🙃",
    "👋","💃","🕺","🤰","👀","🧠","💭","🗣️","👂","👃","👅","💅","🦷","👁️","🫀","🫁",
    "🌹","🌺","🌼","🌷","🍁","🍂","🍃","❄️","🌊","💧","🪨","🔥","💥","🌋","⛰️","🏔️",
    "🏙️","🏠","🏢","🏫","🏥","🏪","🚗","🚌","🚄","✈️","🚀","🛸","⛵","🚦","🗺️","🧭",
]

# 用户要求「多添加点文字」：额外补充一批高频中文常用词（2 字为主，少量 3 字），
# 与语料统计结果合并去重后按频率排序补足词表。
EXTRA_CN_WORDS = [
    "我们","你们","他们","自己","什么","这个","那个","因为","所以","但是","如果","可以",
    "已经","现在","时候","知道","觉得","应该","需要","通过","进行","成为","问题","方面",
    "社会","国家","世界","中国","人类","科学","学习","生活","发展","研究","工作","文化",
    "经济","历史","自然","系统","技术","信息","网络","知识","能力","过程","情况","环境",
    "重要","基础","共同","不同","主要","一定","可能","目前","由于","对于","以及","其中",
    "表示","包括","具有","根据","提供","利用","实现","达到","形成","表现","影响","说明",
    "人工智能","机器学习","深度学习","神经网络","语言模型","计算机","互联网","大数据","区块链",
    "量子","基因","细胞","行星","恒星","宇宙","重力","光速","能量","分子","原子","电子",
]


def _chinese_chars(text):
    return [c for c in text if '\u4e00' <= c <= '\u9fff']


def build_self_designed_vocab(seed_texts, target_size=2500, cache_path=None):
    """基于语料构建自研词表：特殊符 + 字节 + 固定集 + 语料高频中文字/词 + 表情包。

    返回 (vocab_list, meta) 其中 vocab_list 是 new_id->token 字符串，meta 含 new2old/byte_new 兼容字段。
    """
    # 1) 固定集（特殊 + 字节 + 标点数字 + 英文 + 表情包）
    fixed = list(SPECIALS)
    fixed += [f"<0x{i:02X}>" for i in range(256)]
    fixed += PUNCT + ENGLISH + EMOJIS
    fixed = list(dict.fromkeys(fixed))  # 去重保序
    vocab = list(fixed)

    # 2) 语料统计：中文字 unigram + 中文 2/3 字 bigram/trigram + 英文词
    from collections import Counter
    char_cnt = Counter()
    word_cnt = Counter()
    en_word_re = re.compile(r"[a-zA-Z]{2,}")
    for t in seed_texts:
        chars = _chinese_chars(t)
        char_cnt.update(chars)
        # 2/3 字中文词（滑窗），低频过滤
        for n in (2, 3):
            for i in range(len(chars) - n + 1):
                word_cnt.update([ "".join(chars[i:i+n]) ])
        for w in en_word_re.findall(t):
            word_cnt.update([w.lower()])

    # 3) 选高频中文单字（未进词表的）
    chosen_chars = [c for c in char_cnt.most_common() if c[0] not in set(vocab)]
    # 4) 选高频中文词 / 英文词（未进词表的，且频率足够）
    chosen_words = [w for w, _ in word_cnt.most_common() if w[0] not in set(vocab) and len(w) >= 2]

    # 5) 凑到 target_size：先单字后词，并强制把 EXTRA_CN_WORDS 优先塞入
    extra_set = set(EXTRA_CN_WORDS)
    extra_words = [w for w in EXTRA_CN_WORDS if w not in set(vocab)]
    # 把 extra 词按出现频率重新排序（出现过则靠前）
    extra_words_sorted = sorted(extra_words, key=lambda w: -word_cnt.get(w, 0))

    def remaining():
        return target_size - len(vocab)

    # 先塞 extra 中文词（用户要求多文字）
    for w in extra_words_sorted:
        if remaining() <= 0:
            break
        if w not in vocab:
            vocab.append(w)

    # 再塞高频中文单字
    ci = 0
    while remaining() > 0 and ci < len(chosen_chars):
        c = chosen_chars[ci][0]
        ci += 1
        if c not in vocab:
            vocab.append(c)

    # 再塞高频中文词/英文词
    wi = 0
    while remaining() > 0 and wi < len(chosen_words):
        w = chosen_words[wi]
        wi += 1
        if w not in vocab and w not in extra_set:
            vocab.append(w)

    # 若仍不足（语料太小），用频率更高的单字兜底已由上面覆盖；若超出则截断
    vocab = vocab[:target_size]

    # 6) 兼容字段：new2old 全部映射到自身 new id（自研词表无「旧 Qwen id」概念，
    #    导出时直接用 token 字符串；byte_new 记录 byte->new_id 供运行时兜底）。
    new2old = {i: i for i in range(len(vocab))}
    byte_new = {b: 6 + b for b in range(256)}   # 字节 token 固定在 6..261
    meta = {
        "vocab": vocab,
        "new2old": new2old,
        "byte_new": byte_new,
        "specials": SPECIALS,
        "target_size": target_size,
    }
    if cache_path:
        json.dump(meta, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return vocab, meta


class SelfDesignedTokenizer:
    """自研字表分词器：贪心最长匹配 + 字节兜底。"""

    def __init__(self, vocab, byte_new, specials):
        self.vocab = vocab
        self.id2tok = {i: t for i, t in enumerate(vocab)}
        self.tok2id = {t: i for i, t in enumerate(vocab)}
        self.byte_new = byte_new
        self.specials = specials
        # 按长度降序排布可匹配 token（用于最长匹配），排除单字节 byte token
        self.match_tokens = sorted(
            [t for t in vocab if t not in byte_new.values() and t not in specials and len(t) >= 1],
            key=lambda s: -len(s),
        )
        self.vocab_size = len(vocab)

    @classmethod
    def from_meta(cls, meta):
        return cls(meta["vocab"], meta["byte_new"], meta.get("specials", SPECIALS))

    @classmethod
    def load(cls, cache_path):
        m = json.load(open(cache_path, "r", encoding="utf-8"))
        return cls.from_meta(m), m

    def encode(self, text):
        out = []
        i = 0
        L = len(text)
        while i < L:
            # 特殊符优先（最长优先）
            matched = None
            for sp in sorted(self.specials, key=lambda s: -len(s)):
                if text.startswith(sp, i):
                    matched = sp
                    break
            if matched is None:
                for tok in self.match_tokens:
                    if text.startswith(tok, i):
                        matched = tok
                        break
            if matched is not None:
                out.append(self.tok2id[matched])
                i += len(matched)
                continue
            # 字节兜底：单字符按 UTF-8 拆字节，逐字节映射（含 ascii 直接映射）
            b = text[i].encode("utf-8")
            for bb in b:
                out.append(self.byte_new.get(bb, 6 + (bb % 256)))
            i += 1
        return out

    def decode(self, ids):
        # 遇到 byte token 累积字节，其余直接拼
        out = []
        buf = bytearray()
        for i in ids:
            tok = self.id2tok.get(int(i), "")
            if tok.startswith("<0x") and tok.endswith(">") and len(tok) == 6:
                buf.append(int(tok[3:5], 16))
            else:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                out.append(tok)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]


# ---- 便捷：从种子语料构建并落盘（供 train / export 调用） ----
def build_from_seed(cache_path, target_size=2500):
    import seed_corpus as sc
    texts = []
    for d in sc.SEED.values():
        texts.extend(d)
    # 兼容：若已有教师数据则并入
    import os, glob
    for fn in glob.glob(os.path.join("data", "teacher_*.jsonl")):
        try:
            for line in open(fn, "r", encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                txt = obj.get("text") or obj.get("content") or ""
                if txt:
                    texts.append(txt)
        except Exception:
            pass
    vocab, meta = build_self_designed_vocab(texts, target_size=target_size, cache_path=cache_path)
    print(f"[self_tok] 自研词表构建完成 size={len(vocab)}（特殊6 + 字节256 + 固定集 {len(SPECIALS)+256} 起）")
    return vocab, meta


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    build_from_seed("data/self_vocab.json", target_size=2500)
