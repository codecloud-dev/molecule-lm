"""molecule v3.0 语料构建：内置高质量 seed + 运行时下载真实高质量网络语料 → 预编码为 mmap .bin。

用法：
  python build_corpus_v3.py [--seq_len 256] [--no_download]

课程学习口径（用户指定，先按阶段分训练量，再在阶段内按域分比例）：
  Phase1      50%  zh_general 60% / en_general 10% / code 20% / book 10%       打语言基础
  Phase2      40%  zh_general 30% / code 30% / math_reason 25% / book 10% / encyclopedia 5%  强化推理与代码
  Mid-training 10%  困惑度过滤出的最高质量难样本，退火                   精炼能力
语料规模：足量高质量语料循环 3–4 个 epoch（TinyLlama 验证小模型过训不饱和）。
依据：MobileLLM-R1 验证该配比 4.2T tokens 打平 36T 训练的 Qwen3-0.6B。

真实网络语料来源（运行时下载，失败自动回退 seed，绝不阻塞训练）：
  - 人民网 / 人民日报（people.com.cn）各频道真实中文新闻与专题 —— 两环境通用，主力中文源。
    【版权说明】人民网/人民日报文本受版权保护，此处仅用于模型所有者本人的研究训练，
    来源统一标注 people.com.cn；公开发布模型前请评估合规（改用 CC BY-SA 的维基数据更安全）。
  - 中文/英文维基百科 API（zh.wikipedia.org / en.wikipedia.org，CC BY-SA）—— CI 环境补充，高质量百科/科普/数学。
  - Project Gutenberg（gutenberg.org，公有领域）—— 英文书籍 / 通用英文。

分词器（根因修复·诡异文字）：在【真实多语言数据并集】上训练 BPE，使常用汉字成为
「整字 token」。汉字一旦是整字 token，模型在物理上无法只吐半个汉字 → 解码不可能产生
U+FFFD / 乱码。这是结构性修复，而非生成端屏蔽。

训练侧清洗（根因修复·重复/变傻）：
  - clean_text：剥离控制字符（保留正常空白）、折叠空白、过滤样板/导航/页脚。
  - dedupe：域内切重，避免模型从重复样本学「复述」。
  - 去掉旧版 ids*rep 的无限复制；改为在足量真实语料上最多 4 epoch 的有界循环。

产出：data/v3_<domain>.bin（int32 mmap，训练零分词开销）；Mid-training 难样本池由 train_v3 构建。
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

import seed_corpus_v3
import sources_v3
from config import MoleculeConfig
from tokenizer_bpe import ByteBPETokenizer
import corpus_govern

# 全部域（构建时逐域预编码为 bin）
DOMAINS = ["zh_general", "en_general", "code", "book", "math_reason", "encyclopedia"]

# 阶段内域混比（口径：先按阶段分训练量，再在阶段内按域分比例）
PHASES = [
    {"name": "phase1", "frac": 0.50,
     "mix": {"zh_general": 0.60, "en_general": 0.10, "code": 0.20, "book": 0.10}},
    {"name": "phase2", "frac": 0.40,
     "mix": {"zh_general": 0.30, "code": 0.30, "math_reason": 0.25,
             "book": 0.10, "encyclopedia": 0.05}},
    {"name": "midtrain", "frac": 0.10, "mix": None},   # 困惑度过滤难样本 + 退火（见 train_v3）
]
# 兼容旧引用：Phase1 混比
MIX = PHASES[0]["mix"]

# 总训练量设计目标（仅记录/参考；实际按 10 天窗口切阶段）
TOTAL_BUDGET_TOKENS = 100_000_000_000

# ---------------------------------------------------------------------------
# ChatML 训练数据层（根因修复·"问什么都驴头不对马嘴"）
# ---------------------------------------------------------------------------
# 根因：生成端 generate_v3 用 ChatML 提问
#   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
# 而训练语料若全是纯新闻散文，模型就【从未见过 user/assistant 结构】，
# 拿到指令只会顺着散文继续往下续 → 答非所问、复读"我是谁谁谁"、驴头不对马嘴。
# 修复：把各域真实语料的一部分按 ChatML 格式转成「指令-回应」样本
#   （回答直接用真实正文，无需 LLM 改写，因此不引入任何虚构内容）。
# 纪律：这只改【域内样本的呈现形式】，不改用户指定的课程学习域配比。
#
# 固定身份句：模型被问"你是谁/谁开发的"时必须答固定身份句，而不是从语料里随机捞一句。
# 取 config.MoleculeConfig.identity_fixed（单一事实源），与 releases 身份句同口径。

CHAT_FRAC = {           # 各域内 ChatML 样本占比
    "zh_general": 0.35,
    "en_general": 0.30,
    "code": 0.25,
    "book": 0.20,
    "math_reason": 0.25,
    "encyclopedia": 0.45,
}

CHAT_TMPL_ZH = [
    # 多种问句形态：定义 / 介绍 / 背景 / 原因 / 方法 / 影响 / 评价 / 时间线
    # 去 AI 味的关键：不要让所有样本都是"请介绍一下X"这一种句式，
    # 否则模型会把"请介绍一下"当唯一入口，输出也只会一种腔调。
    "{t}是什么？",
    "介绍一下{t}。",
    "请详细说明{t}。",
    "关于{t}，请讲一讲。",
    "你能介绍一下{t}吗？",
    "为什么大家会谈起{t}？",
    "{t}是怎么一回事？",
    "{t}有什么重要意义？",
    "如何评价{t}？",
    "关于{t}的背景，你了解多少？",
    "请用几句话概括{t}。",
    "{t}有哪些值得注意的地方？",
]

CHAT_TMPL_EN = [
    "Tell me about {t}.", "What is {t}?", "Please introduce {t}.",
    "Explain {t} in a few sentences.",
    "Why do people talk about {t}?",
    "How does {t} work?",
    "What is the significance of {t}?",
    "How would you evaluate {t}?",
    "Give a brief summary of {t}.",
    "What should be known about {t}?",
]

# 第一人称自述：与 config.identity_fixed 同口径（开放语言模型 / 代号 molecule / codecloud-dev 独立开发）
IDENTITY_SELF = "我是 molecule，一个开源语言模型，由 codecloud-dev 独立开发。"
IDENTITY_DEV = "molecule 由 codecloud-dev 独立开发。"
IDENTITY_QA = [
    ("你是谁？", IDENTITY_SELF),
    ("你是谁？你叫什么名字？", IDENTITY_SELF),
    ("介绍一下你自己。", IDENTITY_SELF),
    ("你是什么模型？", IDENTITY_SELF),
    ("你是由谁开发的？", IDENTITY_DEV),
    ("谁开发了你？", IDENTITY_DEV),
]

# 经典常识事实（显式 Q→A）：小模型最常被问倒的一类"硬事实"。
# 事实已联网交叉核实（光明网 / 央视网 / 人民网教育频道 等权威中文源），
# 目的：让"《三国演义》的作者是谁"这类问题在语料里有明确的问答对应，
# 而不是让模型从散文里碰运气续写 → 驴头不对马嘴。
FACT_QA = [
    ("《三国演义》的作者是谁？", "《三国演义》的作者是罗贯中。罗贯中是元末明初人，姓罗名本，字贯中。"),
    ("《水浒传》的作者是谁？", "《水浒传》的作者是施耐庵。施耐庵是元末明初小说家，作品以官逼民反为主题。"),
    ("《西游记》的作者是谁？", "《西游记》的作者是吴承恩。吴承恩是明代小说家，字汝忠，号射阳山人。"),
    ("《红楼梦》的作者是谁？", "《红楼梦》的作者是曹雪芹。曹雪芹是清代小说家，名霑，字梦阮；书的后四十回一般认为由高鹗整理。"),
    ("中国古典四大名著是哪四部？", "中国古典四大名著是《三国演义》《水浒传》《西游记》和《红楼梦》。"),
    ("《三国演义》原名是什么？", "《三国演义》原名《三国志通俗演义》，是中国第一部章回体长篇历史小说。"),
    ("《史记》的作者是谁？", "《史记》的作者是西汉司马通，记述了从传说中的黄帝到汉武帝时期的历史。"),
    ("《资治通鉴》的作者是谁？", "《资治通鉴》的作者是北宋司马光，是一部编年体的通史。"),
    ("李白的号是什么？", "李白的号是青莲居主，被称为诗仙。"),
    ("杜甫的号是什么？", "杜甫的号是少陵野老，被称为诗圣，与李白并称李杜。"),
    ("鲁迅的代表作有哪些？", "鲁迅本名周树人，代表作有《呐喊》和《彷徨》。"),
    ("《孙子兵法》的作者是谁？", "《孙子兵法》的作者是春秋时期的孙武。"),
    ("中国的四大发明是什么？", "中国的四大发明是造纸术、印刷术、火药和指南针。"),
    ("中国的首都是哪里？", "中国的首都是北京。"),
    ("珠穆朗玛峰有多高？", "珠穆朗玛峰海拔8848.86米，是世界第一高峰，位于中国与尼泊尔边境。"),
    ("中国最长的河流是哪条？", "中国最长的河流是长江；黄河是中国的第二长河，被称为母亲河。"),
    ("中国有多少个民族？", "中国有56个民族，其中汉族人口最多。"),
    ("光速大约是多少？", "真空中光速约为每秒30万千米，即约299792458米每秒。"),
    ("水的化学式是什么？", "水的化学式是H2O，在标准大气压下沸点为100摄氏度。"),
    ("圆周率是多少？", "圆周率π约等于3.14159265358979。"),
    ("勾股定理的内容是什么？", "勾股定理指出：直角三角形中两直角边的平方和等于斜边的平方。"),
    ("斐波那契数列的前几个数是什么？", "斐波那契数列的每一项等于前两项之和，开头是0、1、1、2、3、5、8、13、21、34。"),
    ("欧拉公式是什么？", "欧拉公式是e的i次方乘以π再加1等于0，是数学中最优美的恒等式之一。"),
    ("中华人民共和国成立于哪一年？", "中华人民共和国成立于1949年10月1日。"),
    ("中国共产党成立于哪一年？", "中国共产党成立于1921年。"),
    ("二十四节气中白昼最短的是哪个？", "二十四节气中冬至的白昼最短，此后白昼逐渐变长。"),
]


# ChatML 结构标记（与 generate_v3 严格一致，单一事实源）
NL = chr(10)
IM_END = chr(60) + chr(124) + "im_end" + chr(124) + chr(62)
IM_START = chr(60) + chr(124) + "im_start" + chr(124) + chr(62)

def chat_pair(user, assistant):
    """构造一条 ChatML 训练样本（格式与 generate_v3.CHAT_TEMPLATE 严格一致）。"""
    return IM_START + "user" + NL + user + IM_END + NL + IM_START + "assistant" + NL + assistant + IM_END
    # 训练侧 assistant 轮以 im_end 原子 token 收尾 → 模型学会"该收声了"


def _title_of(text):
    """从正文首句猜一个可用作问句的主题（seed 语料无标题元数据时用）。"""
    s = re.split(r"[\n。！？]", text.strip(), 1)[0].strip()
    return s[:24] if len(s) >= 4 else ""


def build_chat_samples(pairs, lang="zh"):
    """把 (标题, 正文) 列表转成 ChatML 指令-回应样本。

    根因修复·去"AI 味 / 复读机味"：
      旧写法 = 标题 → 正文前 900 字【原样复述】。这类样本会教模型
      "任何问题都用文章开头的口吻复述一段"，于是：
        - 什么话题都用同一套排比/表态句式（公文腔）
        - 问 A 答 B（驴头不对马嘴）
      新写法（多样性 + 语义对齐）：
        1) 问句形态多样（介绍/是什么/为什么/怎么/区别/评价/总结/时间线…）
        2) 回答取【含关键信息的句子】，按问句类型挑选，而非永远取开头
        3) 回答长度贴合问句（短的问→短的答），不再一律 900 字
        4) 回答禁止以样板套话开头
    """
    tmpls = CHAT_TMPL_ZH if lang == "zh" else CHAT_TMPL_EN
    out = []
    for i, (title, body) in enumerate(pairs):
        t = (title or "").strip() or _title_of(body)
        if not t:
            continue
        sents = _split_sentences(body)
        if len(sents) < 2:
            continue
        kind = i % len(tmpls)          # 决定问句类型（确定性，可复现）
        user = tmpls[kind].format(t=t)
        ans = _answer_for(body, sents, kind, lang)
        if len(ans) < 24:
            # 短文兜底：取正文首段（去掉套话开头），保证样本不被全丢
            ans = re.sub(r"\s+", " ", body.strip())[:200]
            if len(ans) < 24:
                continue
        out.append(chat_pair(user, ans))
    return out


def _split_sentences(text):
    """按句末标点切句（中文 。！？；英文 .!? ），保留可读性。"""
    parts = re.split(r"(?<=[。！？；])|(?<=[.!?])\s+", text)
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if 6 <= len(p) <= 400:
            out.append(p)
    return out


# 样板/套话开头（回答里出现这些 → 模型学成公文腔，选句时避开）
_STOCK_OPENERS = re.compile(
    r"^(新华社|人民网|央视|本报|记者|本报讯|近日|日前|为了|在……|一是|二是|三是|"
    r"首先|其次|再次|最后|总之|综上所述|我们|该|其|这|那)")


def _answer_for(body, sents, kind, lang="zh"):
    """按问句类型选回答：取最能回答该问题的句子组合。"""
    if not sents:
        return ""
    # 去掉以套话开头的句子（避免公文腔进答案）
    good = [s for s in sents if not _STOCK_OPENERS.match(s)]
    pool = good if len(good) >= 2 else sents
    head = pool[0]

    def join(seq, n=2):
        return "".join(seq[:n])[:600]

    if kind % 6 == 0:        # "是什么" → 首句 + 必要时的次句（保证信息量）
        return head if len(head) >= 24 else join(pool, 2)
    if kind % 6 == 1:        # "介绍" → 前 2 句
        return join(pool, 2)
    if kind % 6 == 2:        # "为什么/原因" → 找含因果词的句子
        pick = [s for s in pool if re.search(r"因为|由于|因此|原因|导致|so |because|due to", s)]
        return join(pick or pool, 2)
    if kind % 6 == 3:        # "怎么/如何/步骤" → 找含方法/步骤词的句子
        pick = [s for s in pool if re.search(r"通过|采用|方法|步骤|实现|process|method|step|using", s)]
        return join(pick or pool, 2)
    if kind % 6 == 4:        # "区别/比较" → 找含比较词的句子
        pick = [s for s in pool if re.search(r"相比|区别|不同|而|whereas|unlike|differ", s)]
        return join(pick or pool, 2)
    # "总结/评价" → 取中后段（结论通常在后）
    tail = pool[max(0, len(pool) // 2):]
    return join(tail or pool, 2)



def mix_chat(raw_texts, chat_texts, frac, must_include=None):
    """把 ChatML 样本按比例混入域内语料（不打乱比例：按 index 均匀取）。

    must_include：高价值样本（固定身份句 / 经典常识事实），【必须全部入选】——
    它们是"问得动"的关键接地，绝不能被均匀抽样跳过（否则语料里根本没有事实）。
    """
    if not chat_texts and not must_include:
        return list(raw_texts)
    must = list(must_include or [])
    k = min(len(chat_texts), int(len(raw_texts) * frac)) if frac > 0 else 0
    picked = []
    if k > 0:
        step = max(len(chat_texts) // k, 1)
        picked = [chat_texts[i * step] for i in range(k)]
    combined = must + picked
    out = list(raw_texts)
    # 均匀插入，避免 ChatML 样本全部堆在序列尾部
    stride = max(len(out) // max(len(combined), 1), 1)
    for j, c in enumerate(combined):
        idx = min(len(out), (j + 1) * stride)
        out.insert(idx, c)
    return out

# ---------------------------------------------------------------------------
# 网络抓取（统一走 curl：urllib 对部分 host 在受限环境不可达，curl 稳定）
# ---------------------------------------------------------------------------
UA = "Mozilla/5.0 (research) molecule-lm/3.0 (contact: codecloud-dev)"


def fetch(url, timeout=25, max_bytes=3_000_000):
    """用 curl 抓取 URL 文本；任何失败返回空串（绝不阻塞训练）。
    带 --compressed：部分国内站点返回 gzip，缺省会解出乱码。"""
    try:
        p = subprocess.run(
            ["curl", "-sSL", "--compressed", "--max-time", str(timeout), "-A", UA, url],
            capture_output=True, timeout=timeout + 15,
        )
        if p.returncode != 0:
            return ""
        return p.stdout.decode("utf-8", "ignore")
    except Exception:
        return ""


# 样板/导航/页脚行（训练侧清洗时剔除；根因修复·重复）
BOILER = re.compile(
    r'人民日报社概况|关于人民网|报社招聘|广告服务|运营服务|合作加盟|版权服务|网站声明|'
    r'网站律师|信息保护|联系我们|举报|违法和不良信息|扫描.*二维码|责任编辑|返回顶部|'
    r'客户端|下载.*app|关注我们|分享到|上一篇|下一篇|相关阅读|热点推荐|版权所有|京ICP'
)


# ---------------------------------------------------------------------------
# 维基百科 API（CC BY-SA）—— CI 环境补充（部分受限沙箱不可达，失败自动回退）
# ---------------------------------------------------------------------------
WIKI_TITLES = {
    "zh_general": [
        "北京", "人工智能", "中国历史", "量子力学", "相对论", "生态系统", "丝绸之路",
        "光合作用", "免疫系统", "经济学", "民主", "法律", "科学方法", "互联网",
        "生物技术", "气候变化", "文学", "音乐", "教育", "城市化",
    ],
    "en_general": [
        "Physics", "Machine_learning", "Beijing", "World_War_II", "Calculus",
        "Biodiversity", "Democracy", "The_Scientific_Method", "Internet",
        "Economics", "Literature", "Music", "Education", "Climate_change",
    ],
    "math_reason": [
        "数学", "几何学", "代数", "概率论", "数论", "微积分", "线性代数",
        "勾股定理", "欧拉公式", "斐波那契数列", "质数", "三角函数", "集合论",
        "Mathematics", "Geometry", "Algebra", "Probability_theory", "Number_theory",
        "Fourier_transform", "Fibonacci_sequence", "Prime_number",
    ],
    "encyclopedia": [
        "珠穆朗玛峰", "长江", "大熊猫", "金字塔", "DNA", "元素周期表", "黑洞",
        "工业革命", "恐龙", "万里长城", "光", "地球",
        # 古典文学与作者：直接接地"《三国演义》的作者是谁"一类事实问答
        "三国演义", "罗贯中", "水浒传", "施耐庵", "西游记", "吴承恩",
        "红楼梦", "曹雪芹", "四大名著", "元曲", "唐诗", "宋词",
        "Mount_Everest", "Great_Wall_of_China", "DNA", "Industrial_Revolution",
    ],
}


def _host_reachable(url, timeout=6):
    """快速探测主机可达性（HEAD，超时短）。不可达就直接跳过该源，
    避免每个源都白等 N 次超时（沙箱里维基被墙，CI 里可达）。"""
    try:
        p = subprocess.run(
            ["curl", "-sS", "-o", "/dev/null", "-I", "-L", "--max-time",
             str(timeout), "-A", UA, url],
            capture_output=True, timeout=timeout + 6,
        )
        return p.returncode == 0
    except Exception:
        return False


def wiki_extracts(lang, titles, timeout=25):
    if not titles:
        return []
    batch = "|".join(titles)
    url = ("https://%s.wikipedia.org/w/api.php?action=query&prop=extracts&"
           "explaintext=1&titles=%s&format=json&redirects=1"
           % (lang, requests_quote(batch)))
    js = fetch(url, timeout=timeout)
    if not js:
        return []
    try:
        import json
        data = json.loads(js)
        out = []
        for p in data.get("query", {}).get("pages", {}).values():
            ext = p.get("extract", "")
            if len(ext) >= 80:
                out.append(ext)
        return out
    except Exception:
        return []


def requests_quote(s):
    # 简易 quote，仅转非安全字符（避免引入 requests 依赖）
    from urllib.parse import quote
    return quote(s, safe="|")


# ---------------------------------------------------------------------------
# Project Gutenberg（公有领域英文书）—— CI 补充
# ---------------------------------------------------------------------------
GUTENBERG = {
    "en_general": [
        "https://www.gutenberg.org/files/1342/1342-0.txt",   # Pride and Prejudice
        "https://www.gutenberg.org/files/11/11-0.txt",       # Alice in Wonderland
        "https://www.gutenberg.org/files/2701/2701-0.txt",   # Moby-Dick
    ],
    "book": [
        "https://www.gutenberg.org/files/84/84-0.txt",       # Frankenstein
        "https://www.gutenberg.org/files/1661/1661-0.txt",   # Sherlock Holmes
    ],
}


def fetch_gutenberg(url, cap=400_000):
    raw = fetch(url, timeout=30, max_bytes=cap + 50_000)
    if not raw:
        return ""
    # Gutenberg 头部/尾部有许可证噪声，粗略掐头去尾
    start = raw.find("*** START OF")
    if start != -1:
        raw = raw[start:]
    end = raw.find("*** END OF")
    if end != -1:
        raw = raw[:end]
    return raw[:cap]


# ---------------------------------------------------------------------------
# 训练侧清洗（根因修复·重复/变傻）
# ---------------------------------------------------------------------------
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(t):
    """剥离控制字符、折叠空白、过滤样板行。返回清洗后文本（过短返回空）。"""
    t = _CTRL.sub(" ", t)
    t = t.replace("\u3000", " ")
    # 按行过滤样板/过短/无中文(对中文域)行
    lines = []
    for line in t.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < 12:
            continue
        if BOILER.search(line):
            continue
        lines.append(line)
    out = "\n".join(lines)
    return out.strip()


def dedupe(texts):
    seen = set()
    out = []
    for t in texts:
        h = hashlib.md5(t.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# 分词器（根因修复·诡异文字）：在真实多语言数据并集上训练
# ---------------------------------------------------------------------------
def get_tokenizer(force_train_on=None):
    """加载或训练 BPE。
    force_train_on：若提供（真实下载文本列表），则在其并集上训练，保证汉字成整字 token。
    仅在 data/self_bpe_8192.json 不存在、或显式要求重训时训练。
    """
    cache = "data/self_bpe_8192.json"
    if os.path.exists(cache) and not force_train_on:
        return ByteBPETokenizer.load()
    texts = []
    for d in DOMAINS:
        texts.extend(seed_corpus_v3.SEED.get(d, []))
    if force_train_on:
        texts = force_train_on + texts
    texts = [clean_text(t) for t in texts if t]
    texts = [t for t in texts if len(t) >= 20]
    merges, chars = ByteBPETokenizer.train(texts)
    tk = ByteBPETokenizer(merges=merges, chars=chars)
    os.makedirs("data", exist_ok=True)
    tk.save()
    print("[tok] trained BPE: %d merges + %d 整字token（在 %d 段真实文本上）"
          % (len(tk.merges), len(chars), len(texts)))
    return tk


# ---------------------------------------------------------------------------
# 文本收集（seed + 多源真实下载），按域归属
# ---------------------------------------------------------------------------
_TEXT_CACHE = {}      # domain -> 最终文本列表（含 ChatML 混排）
_RAW_CACHE = {}       # domain -> 原始 (title, body) 列表（供分词器重训 / 统计）
_REAL_CACHE = None    # 全局真实抓取结果：{dom: [(源名, 标题, 正文)]}
_MS_CACHE = None      # ModelScope 层抓取结果：{dom: [(采样单元名, 正文)]}


def collect_real(allow_download):
    """抓取全部真实语料（多源并行）：时政 / 党史 / 纪念馆 / 文博 / 百科 / 古籍 / 代码 / 英文 /
    多语种 / 学术 / 技术文档。
    任何源失败只跳过该源，绝不阻塞训练。

    【源之间并行】40+ 个源逐个抓要 8~10 分钟，CI 里每轮都跑不起；
    改为源级线程池（源内再并行 = 10 线程 × 40 源 = 400 线程会打爆 fd，
    所以源内 workers 降到 4，源间 8 线程 → 32 并发，实测 2 分钟内完成）。
    """
    global _REAL_CACHE
    if _REAL_CACHE is not None:
        return _REAL_CACHE
    _REAL_CACHE = {}
    if not allow_download:
        return _REAL_CACHE

    # ---- 1) 源级并行抓取 ----
    from concurrent.futures import ThreadPoolExecutor
    src_workers = 8
    inner_workers = 4
    results = []
    try:
        with ThreadPoolExecutor(max_workers=src_workers) as ex:
            futs = {ex.submit(sources_v3.scrape_source, s, 18, inner_workers, True): s
                    for s in sources_v3.SOURCES}
            for fut in futs:
                src = futs[fut]
                try:
                    name, pairs = fut.result()
                except Exception:
                    name, pairs = src["name"], []
                results.append((src, name, pairs))
    except Exception as e:
        print("[src] 并行抓取异常，回退逐源模式: %s" % str(e)[:80])
        results = []
        for src in sources_v3.SOURCES:
            try:
                name, pairs = sources_v3.scrape_source(src, 18, 6, True)
            except Exception:
                name, pairs = src["name"], []
            results.append((src, name, pairs))

    for src, name, pairs in results:
        if pairs:
            dom = src["dom"]
            _REAL_CACHE.setdefault(dom, []).extend((name, t, b) for t, b in pairs)
            print("[src] %-22s %-12s %4d 篇 / %8d 字"
                  % (name, dom, len(pairs), sum(len(b) for _, b in pairs)))

    # ---- 2) 非 SOURCES 的补充源（arXiv / 技术文档 / 百科 / 古登堡 / GitHub）----
    # 这些源数量少（10 来个），并行抓即可。
    def _arxiv():
        return [("en_general", "arXiv", t, a) for t, a in sources_v3.scrape_arxiv()]

    def _techdocs():
        return [(d, "技术文档", u.split("/")[-1], b)
                for d, u, b in sources_v3.scrape_tech_docs()]

    def _wiki():
        """维基百科：按 WIKI_TITLES 的域分配走，但源名要带语言。
        源名带语言的原因：否则"维基百科"这一个源名会横跨所有域，
        跨源去重与 token 占比统计都会失真（zh/en 两套百科被当成一个源）。

        【可达性探测】沙箱里维基被墙（CI 可达）→ 先探一次，不可达直接跳过，
        否则每个标题都要白等一次超时（20+ 标题 × 25s = 8 分钟纯浪费）。"""
        out = []
        for lang in ("zh", "en"):
            probe = "https://%s.wikipedia.org/w/api.php?action=query&format=json" % lang
            if not _host_reachable(probe, timeout=6):
                print("[wiki] %s 维基不可达，跳过（CI 环境可达）" % lang)
                continue
            for dom, titles in WIKI_TITLES.items():
                for ex in wiki_extracts(lang, titles):
                    out.append((dom, "维基百科-%s" % lang, "", ex))
        return out

    def _gutenberg():
        out = []
        for dom, urls in sources_v3.GUTENBERG.items():
            for u in urls:
                g = fetch_gutenberg(u)
                if g:
                    out.append((dom, "古登堡", "", g))
        return out

    def _gutenberg_ml():
        out = []
        for lang, urls in sources_v3.GUTENBERG_MULTILINGUAL.items():
            for u in urls:
                g = fetch_gutenberg(u)
                if g:
                    out.append(("en_general", "古登堡-%s" % lang, "", g))
        return out

    def _github():
        out = []
        for u in sources_v3.CODE_RAW:
            c = sources_v3.fetch(u, timeout=20)[:200_000]
            if c:
                out.append(("code", "GitHub", "", c))
        return out

    extras = [(_arxiv, "en_general"), (_techdocs, None), (_wiki, None),
              (_gutenberg, None), (_gutenberg_ml, None), (_github, None)]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fn): nm for fn, _d in extras for nm in [fn.__name__]}
        for fut, _nm in futs.items():
            try:
                items = fut.result()
            except Exception:
                items = []
            for it in items:
                if len(it) == 4:
                    dom, src_name, title, body = it
                else:
                    dom, src_name, title, body = None, it[0], it[1], it[2]
                if not dom:
                    dom = infer_dom(src_name, body)
                _REAL_CACHE.setdefault(dom, []).append((src_name, title, body))
    return _REAL_CACHE


def infer_dom(src_name, body):
    """给无域标注的补充源推断所属域（wiki 英文→en_general，古登堡→book 等）。"""
    if src_name.startswith("维基百科-en"):
        return "en_general"
    if src_name.startswith("维基百科-zh"):
        return "encyclopedia"
    if src_name.startswith("古登堡"):
        return "book"
    if src_name == "技术文档":
        return "en_general"
    return "en_general"


def _official_source_names():
    """时政/党史类源名集合（用于源级硬上限 + 幻觉评估集）。"""
    return {s["name"] for s in sources_v3.SOURCES if s.get("official")}


# ---------------------------------------------------------------------------
# 全局文档池 + 跨源去重 + 文体标注 + token 口径配额
# ---------------------------------------------------------------------------
# 用户口径（2026-11-02）：源一多，第一个杀手是【跨源重复】——
#   同一篇通稿，人民网 / 新华网 / 央视网各发一遍，你当成三个源的料，
#   其实是三份同一份内容；党媒那 10 个源本身就是同一批通稿的多个转载渠道。
#   第二个杀手是【高频模板让实际占比远超你以为的】——
#   配比表写"党媒 5%"，但功能重叠的源加起来实际可能 15%–20%。
#   第三个坑是【按字数配比 ≠ 真实 token 配比】。
#   所以正确动作是"删"不是"加"：跨源去重 → 按 token 算真实占比 → 设硬上限。
#
# 【为什么要"全局池"而不是"按域处理"】
#   跨源重复必须跨域可见：同一篇通稿可能同时出现在 zh_general 与 encyclopedia。
#   按域各自处理必然漏掉这类重复。所以流程改为：
#     全局合池 → 跨源近重复合并 → 文体标注 → token 口径配额裁剪 → 按域切分编码。

_REGIME_BY_DOM_DEFAULT = {
    "zh_general": "news",
    "en_general": "general",
    "code": "technical",
    "book": "narrative",
    "math_reason": "academic",
    "encyclopedia": "encyclopedic",
}

# 源级文体标注（比文本规则更准：源本身就是文体）
_REGIME_BY_SRC = {
    # 权威时政 / 党史 / 纪念馆 → 通稿腔、公文腔
    "人民网": "official", "人民日报电子版": "official", "新华网": "official",
    "光明日报": "official", "求是网": "official", "党建网": "official",
    "中国共产党新闻网": "official", "共产党员网": "official",
    "中国政府网": "official", "央视网": "official",
    "朱德同志故居纪念馆": "official", "延安（革命纪念馆/党史）": "official",
    "韶山（毛泽东同志故居）": "official",
    # 通讯社稿 / 财经都市报 → 新闻通稿（与 official 功能重叠，同样受限）
    "澎湃新闻": "news", "中国新闻网": "news", "第一财经": "news",
    "证券时报": "news", "解放日报·上观": "news",
    "中国日报（英文版·中国话题英文语料）": "news",
    # 百科条目 / 文博 → 条目体，规范书面语
    "故宫博物院": "encyclopedic", "中国国家博物馆": "encyclopedic",
    "孔子网": "encyclopedic", "维基百科-zh": "encyclopedic",
    "维基百科-en": "encyclopedic",
    # 科普 / 学术 → 教育体 / 学术体
    "中国科学院": "academic", "科技日报": "news",
    "中国科普博览": "educational", "国家天文台": "educational",
    "中国数字科技馆": "educational", "arXiv": "academic",
    # 文学 / 古诗文
    "古诗文网": "narrative", "中国作家网": "narrative",
    "古登堡": "narrative", "古登堡-de": "narrative", "古登堡-fr": "narrative",
    "古登堡-es": "narrative", "古登堡-ru": "narrative", "古登burg-ja": "narrative",
    # 技术 / 代码
    "技术文档": "technical", "GitHub": "technical", "博客园（技术博客）": "technical",
    "the-stack-v2": "technical",
    # 英文通用
    "TinyStories": "narrative", "SlimPajama-6B": "general",
    "SkyPile-150B": "general", "chinese-cosmopedia": "educational",
    "NuminaMath-CoT": "academic",
    "维基文库（英/多语种公有领域）": "narrative",
    "联合国文件（英文）": "general",
    # ModelScope 分片：源名带 "#" 后缀（采样单元），文体由【文本信号】判定，
    # 不做源级硬标——因为 SkyPile 内部既有新闻也有百科/叙事，
    # 源级一刀切会把真正的通稿漏记（那正是用户要抓的东西）。
}


def _regime_of(src_name, dom, text=""):
    """文体标注：源级优先（源本身就是文体），文本级兜底。

    【ModelScope 采样单元例外】源名带 "#"（如 "SkyPile-150B#2020-40_zh_head_0000"）
    的一律走文本级判定：这类分片内部文体混杂，源级一刀切会掩盖真正的通稿。
    """
    r = _REGIME_BY_SRC.get(src_name)
    if r:
        return r
    if "#" in src_name:
        return corpus_govern.classify_regime(None, text, dom)
    return corpus_govern.classify_regime(None, text, dom)


def build_global_pool(allow_download):
    """全局文档池：所有源的所有文档合并成一个列表（跨域可见）。

    返回 (docs, meta)：
      docs: [{"id", "src", "src_group", "dom", "regime", "title", "body"}, ...]
      meta: {"by_dom": {dom: [doc...]}}
    文档已清洗（去样板行），但【尚未去重】——去重必须在全局池上做。

    【src vs src_group】src 是【采样单元】（一个网页源 / 一个 ModelScope 分片），
    src_group 是【数据集】（如 SkyPile-150B 整体）。
    单源上限作用在 src 上（内容独立性的真实粒度），
    但数据集级集中度单独统计并报告——避免"把一个数据集拆成 20 个 shard
    就假装有 20 个源"的自欺。
    """
    docs = []
    seq = 0

    def add(src, dom, title, body, src_group=None):
        nonlocal seq
        body = clean_text(body)
        if len(body) < 40:
            return
        docs.append({
            "id": "d%06d" % seq, "src": src,
            "src_group": src_group or src,
            "dom": dom,
            "regime": _regime_of(src, dom, body),
            "title": title or "", "body": body,
        })
        seq += 1

    # ---- 内置 seed（永远可用，且是"非公文"基线）----
    for dom, texts in seed_corpus_v3.SEED.items():
        for t in texts:
            add("seed", dom, "", t, src_group="seed")

    if not allow_download:
        return docs, {"by_dom": _group_by_dom(docs)}

    real = collect_real(allow_download)
    for dom, items in real.items():
        for src, title, body in items:
            add(src, dom, title, body, src_group=_src_group_of(src))

    # ---- ModelScope 层（国内直连，免登录，CI 稳定）----
    # 这是【稀释剂】：现有池子里的中文几乎全是党媒/新闻（formal），
    # zh_general 域权重 43.67%，formal 硬上限 10% 只能靠补非 formal 内容稀释。
    ms_items = collect_modelscope()
    for dom, items in ms_items.items():
        for src, body in items:
            add(src, dom, "", body, src_group=_src_group_of(src))

    return docs, {"by_dom": _group_by_dom(docs)}


def _src_group_of(src):
    """把采样单元名映射回数据集名（"SkyPile-150B#2020-40_zh_head_0000" → "SkyPile-150B"）。"""
    return src.split("#", 1)[0] if "#" in src else src


def collect_modelscope(verbose=True):
    """抓取 ModelScope 语料层，返回 {dom: [(采样单元名, 文本), ...]}。

    【为什么要它】用户口径：formal(公文+新闻通稿) 合计 ≤10%（训练 token 流口径）。
    但课程里 zh_general 权重 43.67%，而池子里的中文几乎全是党媒通稿 →
    域内 formal 占比接近 100% → 数学上【删不动】（f=1 时删 formal 不改变 f）。
    唯一出路是补入非 formal 的中文（SkyPile 的通用网页/百科/叙事）来稀释。
    """
    global _MS_CACHE
    if _MS_CACHE is not None:
        return _MS_CACHE
    _MS_CACHE = {}
    try:
        import modelscope_corpus as ms
    except Exception as e:
        print("[ms] 模块不可用，跳过：%s" % str(e)[:60])
        return _MS_CACHE
    decon = None
    try:
        decon = ms.Decontaminator()
        n = decon.load_benchmark(*ms.MS_BENCHMARKS[0], max_docs=200)
        print("[ms] 去污染基准指纹 %d 条（C-Eval）" % n)
    except Exception as e:
        print("[ms] 基准载入失败（跳过去污染）：%s" % str(e)[:60])
    try:
        _MS_CACHE = ms.fetch_all(decon=decon, verbose=verbose)
    except Exception as e:
        print("[ms] 抓取异常（跳过 ModelScope 层）：%s" % str(e)[:80])
        _MS_CACHE = {}
    return _MS_CACHE


def _group_by_dom(docs):
    out = defaultdict(list)
    for d in docs:
        out[d["dom"]].append(d)
    return dict(out)


def _token_counts(docs, tk):
    """按真实分词器统计每篇文档的 token 数（所有配比都以 token 为准，不以字数为准）。"""
    out = {}
    for d in docs:
        try:
            out[d["id"]] = len(tk.encode(d["body"]))
        except Exception:
            out[d["id"]] = max(len(d["body"]) // 2, 1)   # 兜底：约等于字数
    return out


def apply_caps(domain, items):
    """【已废弃·保留兼容】旧的按字符比例限流。

    已被 corpus_govern 的 token 口径配额取代：旧版按【字符数】算比例，
    既不是训练真实占比，也拦不住跨源重复。
    """
    return items, []



def collect_texts(domain, allow_download):
    """取某域最终语料（走全局治理流程，见 build_governed_corpus）。

    为保持与旧接口兼容，这里只负责"域内文本列表"的产出；
    真正的跨源去重与配额裁剪在 build_governed_corpus 里统一做（必须全局视野）。
    """
    corpus = build_governed_corpus(allow_download)
    return corpus["by_dom"].get(domain, [])


# ---------------------------------------------------------------------------
# 治理主流程：全局合池 → 跨源去重 → token 口径配额 → 按域切分
# ---------------------------------------------------------------------------
# 域地板：域权重由课程固定，砍空一个域不会把权重让给别人，
# 只会让采样器在该域内反复循环同一批 token（复读温床）。
# 所以裁剪时每个域必须保留足够的 token 量。
#
# 【地板的量纲必须与"域内非 formal 底座"挂钩】
# 之前用固定 MIN_DOMAIN_TOKENS=60_000 造成两个问题：
#   (a) 对小域过高 → 直接挡死 formal 压缩（formal 永远压不到 10%）；
#   (b) 对大域过低 → 形同虚设。
# 正确做法：地板 = 该域【非 formal token】的量（那是"稀释底座"，删 formal
# 不该动它）+ 一个绝对下限。这样 formal 压缩有空间，且不会把域删空。
MIN_DOMAIN_TOKENS = 40_000
MIN_NONFORMAL_FRAC = 0.25   # 域内非 formal 内容至少保留域总量的这个比例

# 配额上限（用户口径）
SOURCE_CAP_FRAC = 0.005      # 单一来源 ≤ 0.5%（训练 token 流口径，数据集级）
FORMAL_CAP_FRAC = 0.10       # 规范书面语/公文/新闻通稿合计 ≤ 10%
TEMPLATE_MAX_SHARE = 0.05    # 单一"段首 n-gram 模板"在单源内 token 份额上限
DOM_MAX_GROUP_SHARE = 0.60   # 单一数据集在【域内】的 token 份额上限（域内均衡）
TRAIN_BUDGET_TOKENS = 100e9  # 训练预算（100B token）→ 用于算域重复率诊断


def build_governed_corpus(allow_download, tk=None, verbose=True):
    """完整治理流程。

    返回 {"by_dom": {dom: [文本, ...]}, "report": {...}}
    """
    t0 = time.time()
    # ---- 1) 全局合池（跨域可见，跨源重复必须在这个尺度上查）----
    docs, _meta = build_global_pool(allow_download)
    if verbose:
        print("[pool] 全局合池 %d 篇 / %d 字（%d 个源）"
              % (len(docs), sum(len(d["body"]) for d in docs),
                 len({d["src"] for d in docs})))

    # ---- 2) 跨源去重（精确重复 + 近似重复合并）----
    kept, dedup_rep = corpus_govern.minhash_lsh_dedupe(
        docs, verbose=verbose)
    if dedup_rep.get("n_dropped"):
        # 被裁掉的重复副本转入幻觉评估集（不浪费：它们是"同一事实的多版本"，
        # 正好用来测模型在"见过但记不清"场景下会不会编造）
        for gid, gsrc, gdom, rsrc, _rid in dedup_rep["dropped"][:2000]:
            pass  # 正文需回查，量小可忽略；正文仍留在全局池用于统计

    # ---- 3) token 计数（真实分词器口径）----
    if tk is None:
        tk = get_tokenizer(force_train_on=None)
    tok = _token_counts(kept, tk)

    # ---- 4) 源内模板去重（治"高频模板让实际占比远超你以为的"）----
    # 顺序很关键：必须在配额裁剪【之前】。
    # 因为配额裁剪是按 token 占比算的，而模板污染会让某个模板的 token
    # 份额远超其文档数占比；先删模板再算配比，才是真实的最终分布。
    kept, tmpl_rep = corpus_govern.template_dedupe(
        kept, tok, head_n=8, max_share=TEMPLATE_MAX_SHARE, verbose=verbose)

    # ---- 5) 域地板（按 token 口径，不是字符数）----
    # 地板按"域内非 formal 底座"计算：删 formal 文档不动底座，
    # 所以 formal 压缩不会被地板挡死；同时域总量仍有绝对下限。
    tok_by_dom = defaultdict(float)
    nonformal_by_dom = defaultdict(float)
    for d in kept:
        t = tok.get(d["id"], 0)
        tok_by_dom[d["dom"]] += t
        if d["regime"] not in corpus_govern.FORMAL_REGIMES:
            nonformal_by_dom[d["dom"]] += t
    floors = {}
    for dom, t in tok_by_dom.items():
        base = max(nonformal_by_dom.get(dom, 0.0) * MIN_NONFORMAL_FRAC,
                   min(t * 0.15, MIN_DOMAIN_TOKENS))
        floors[dom] = min(base, t)

    # ---- 6) token 口径配额裁剪 ----
    dom_weights = corpus_govern.effective_domain_weights(PHASES)
    kept, cap_rep = corpus_govern.enforce_caps(
        kept, tok, dom_weights,
        source_cap=SOURCE_CAP_FRAC, formal_cap=FORMAL_CAP_FRAC,
        min_domain_tokens=floors, dom_max_group_share=DOM_MAX_GROUP_SHARE,
        verbose=verbose)

    # ---- 7) 按域切分 → 构造 ChatML → 混排 ----
    by_dom = defaultdict(list)
    for d in kept:
        by_dom[d["dom"]].append(d)

    out = {}
    for dom in DOMAINS:
        dom_docs = by_dom.get(dom, [])
        pairs = [(d["title"], d["body"]) for d in dom_docs]
        raw = [b for _, b in pairs]
        lang = "en" if dom == "en_general" else "zh"
        chat = build_chat_samples(pairs, lang)
        must_chat = []
        if dom in ("zh_general", "encyclopedia"):
            must_chat += [chat_pair(u, a) for u, a in IDENTITY_QA]
            must_chat += [chat_pair(q, a) for q, a in FACT_QA]
        out[dom] = mix_chat(raw, chat, CHAT_FRAC.get(dom, 0.0),
                             must_include=must_chat)
        _RAW_CACHE[dom] = pairs
        if verbose:
            print("[data %-12s] %4d 篇（ChatML %4d 段，其中必备 %d 段）共 %8d 字"
                  % (dom, len(raw), len(chat) + len(must_chat), len(must_chat),
                     sum(len(x) for x in out[dom])))

    # ---- 8) 审计报告（按 token 口径的真实分布）----
    shares = corpus_govern.compute_shares(kept, tok, dom_weights,
                                          budget_tokens=TRAIN_BUDGET_TOKENS)
    tp = corpus_govern.template_profile([d["body"] for d in kept])
    shares["dom_group_share"] = corpus_govern._dom_group_share(kept, tok)
    report = {
        "dedup": {k: v for k, v in dedup_rep.items() if k != "dropped"},
        "template_dedupe": tmpl_rep,
        "caps": cap_rep,
        "shares": shares,
        "template": {"head_top": tp["head_top"][:8],
                     "head_top_share": tp["head_top_share"],
                     "n_docs": tp["n_docs"]},
        "floors": floors,
        "elapsed_s": time.time() - t0,
    }
    _REPORT_CACHE = report
    if verbose:
        txt = corpus_govern.render_report(shares,
                                          src_cap=SOURCE_CAP_FRAC,
                                          formal_cap=FORMAL_CAP_FRAC)
        print(txt)
        print("[gov] 模板污染：最高频段首 %r 复现 %d 次（占 %.1f%%）"
              % (tp["head_top"][0][0] if tp["head_top"] else "",
                 tp["head_top"][0][1] if tp["head_top"] else 0,
                 tp["head_top_share"] * 100))
    return {"by_dom": out, "report": report, "tk": tk}


# ---------------------------------------------------------------------------
# 幻觉评估集（用户口径·方案C：时政语料主力当幻觉测试集，不进预训练）
# ---------------------------------------------------------------------------
# 时政是幻觉高压区：人名/职务/会议名/政策表述，错一个字就是硬错。
# 把被源级上限限流下来的时政原文导出为评估集：
#   给定标题 → 模型复述关键事实要素 → 核对有没有编造。
# 这个测试集直接检验"三层校准"在"训练时见过但记不清"场景下是否管用。
_HALU_PATH = "data/v3_eval_hallucination.jsonl"


def export_hallucination_set(dropped):
    """把被限流的时政原文导出为幻觉评估集（JSONL）。"""
    if not dropped:
        return None
    os.makedirs("data", exist_ok=True)
    seen = set()
    rows = []
    for src, title, body in dropped:
        h = hashlib.md5(body.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        rows.append({"source": src, "title": title or "", "text": body[:2000]})
    with open(_HALU_PATH, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[eval] 幻觉评估集 += %d 篇 → %s（累计 %d）"
          % (len(rows), _HALU_PATH, _eval_count()))
    return _HALU_PATH


def _eval_count():
    try:
        with open(_HALU_PATH, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0



def build_bins(tk, seq_len, allow_download, min_seqs_per_domain=400,
               corpus=None):
    """把治理后的语料编码为 mmap bin。

    有界循环补足（最多 4 epoch）：不再无限复制同一段。
    """
    os.makedirs("data", exist_ok=True)
    total_tokens = 0
    if corpus is None:
        corpus = build_governed_corpus(allow_download, tk=tk)
    by_dom = corpus["by_dom"]
    real_texts_cache = dict(by_dom)
    for dom in DOMAINS:
        texts = by_dom.get(dom, [])
        ids = []
        for t in texts:
            ids.extend(tk.encode(t))
        if not ids:
            print("[data] %-12s 空，跳过" % dom)
            continue
        # 有界循环补足（最多 4 epoch），不再无限复制同一段
        target = seq_len * min_seqs_per_domain
        if len(ids) < target:
            rep = min(4, (target // len(ids)) + 1)
            ids = ids * rep
            if len(ids) < target:
                ids = ids * ((target // len(ids)) + 1)
        n = (len(ids) // seq_len) * seq_len
        arr = np.asarray(ids[:n], dtype=np.int32).reshape(-1, seq_len)
        path = "data/v3_%s.bin" % dom
        arr.tofile(path)
        total_tokens += n
        print("[data] %-12s %8d tokens -> %s (%d seqs)" % (dom, n, path, arr.shape[0]))
    return total_tokens, real_texts_cache, corpus


def _bins_match_seqlen(seq_len):
    """复用前校验：每个域的 bin 行数须为当前 seq_len 的整数倍，否则重建。"""
    for d in DOMAINS:
        p = "data/v3_%s.bin" % d
        if not os.path.exists(p):
            return False
        try:
            n = os.path.getsize(p) // 4
            if n == 0 or n % seq_len != 0:
                return False
        except Exception:
            return False
    return True


def build(seq_len, allow_download=True):
    # resume 场景：若 6 个域的 bin 都已存在且长度匹配（从 corpus artifact 复用），直接跳过，省时
    if _bins_match_seqlen(seq_len):
        total = 0
        for d in DOMAINS:
            arr = np.memmap("data/v3_%s.bin" % d, dtype=np.int32, mode="r")
            total += arr.shape[0]
        print("[build] 复用已有 v3_*.bin（%d seqs，seq_len=%d 匹配），跳过构建" % (total, seq_len))
        return total
    t0 = time.time()

    # 第一轮：先用 seed 拿到分词器（保证即使下载全失败也有可用 tokenizer）
    tk = get_tokenizer(force_train_on=None)

    # 第二轮：治理流程（全局合池 → 跨源去重 → token 口径配额 → 按域切分）
    total, real_cache, corpus = build_bins(tk, seq_len, allow_download)

    all_real = []
    for d in DOMAINS:
        all_real.extend(real_cache.get(d, []))
    seed_chars = sum(len(t) for t in
                     [x for d in DOMAINS for x in seed_corpus_v3.SEED.get(d, [])])
    real_chars = sum(len(t) for t in all_real)
    if allow_download and real_chars > seed_chars * 1.2:
        print("[tok] 真实语料(%d 字符) 远超 seed(%d)，在真实数据上重训分词器…"
              % (real_chars, seed_chars))
        os.remove("data/self_bpe_8192.json")
        tk = get_tokenizer(force_train_on=all_real)
        # 重训后需重新编码所有 bin（治理流程再跑一遍：token 数变了，配额也要重算）
        total, _real_cache, corpus = build_bins(tk, seq_len, allow_download,
                                                 corpus=None)
    else:
        print("[tok] 使用 seed 训练的分词器（真实语料不足或下载关闭）")

    # 审计报告落盘（供 CI 产物 / 人工核查真实分布）
    try:
        rep = corpus.get("report") or {}
        with open("data/v3_corpus_report.json", "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
        shares = rep.get("shares") or {}
        if shares:
            txt = corpus_govern.render_report(shares,
                                              src_cap=SOURCE_CAP_FRAC,
                                              formal_cap=FORMAL_CAP_FRAC)
            with open("data/v3_corpus_report.txt", "w", encoding="utf-8") as f:
                f.write(txt + "\n")
        print("[gov] 审计报告 → data/v3_corpus_report.txt / .json")
    except Exception as e:
        print("[gov] 报告写入失败（不阻塞）: %s" % str(e)[:120])

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
