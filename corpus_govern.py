"""molecule v3.0 语料治理层：跨源去重 / 文体归类 / token 口径统计与配额硬上限。

用户口径（2026-11-02，四条动作，优先级高于"再加任何一个源"）：
  1) 【跨源重复是头号杀手】同一篇通稿，人民网 / 新华网 / 央视网各发一遍 →
     你当成三个源的料，其实是三份同一份内容。网页语料去重后常只剩原来三到五成，
     这是正常水位，不是异常。
     → 合池后先跑一遍【跨源近重复检测】（MinHash + LSH），跨域也查。
     → 统计一律按【去重后的 token 数】，不信"文件数"。
  2) 【高频模板让实际占比远超你以为的】配比表上写"党媒 5%"，但和它功能重叠的
     源（政府网 / 百科 / 规范书面语）加起来实际可能 15%–20%。
     → 按 token 算"规范书面语 / 公文 / 新闻通稿"合计占比，超过 10% 就压。
  3) 【按字数感觉的配比 ≠ 真实 token 配比】中英代码数学数字标点的 token 密度差好几倍。
     → 所有配比统一用 token 数算；训练前统计"实际采样出来的 token 分布"，不信标注表。
  4) 【现在该做的是"删"不是"加"】单一来源硬上限 0.5%，规范书面语合计 ≤10%。

本模块提供三件事：
  A. minhash_lsh_dedupe()  —— 跨源近重复检测（精确重复 + 近似重复两档）
  B. classify_regime()     —— 文体归类（official / news / encyclopedic / educational / ...）
  C. enforce_caps()        —— token 口径的配额求解器（单源上限 + 文体上限）

设计要点：
  - 去重先于统计：所有占比都是"去重后"的数字。
  - 所有配额以【token 数】为单位，不以字符数、不以篇数。
  - 上限作用在【训练时真实采样分布】上（域权重 × 域内占比），不是作用在池子上——
    池子里占比 5% 的东西，如果它所在域在课程里占 60%，那么它在 token 流里就是 30%。
    这是用户第 3 条真正的含义，也是"标注表骗人"的根源。
"""

import hashlib
import re
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# 语料状态分级（regime）：用于"文体维度"的配额，而不是"域维度"
# ---------------------------------------------------------------------------
# official      党媒 / 政府 / 党建 / 纪念馆 / 央视 → 通稿腔、公文腔
# news          通讯社稿 / 财经报道 / 都市报纸     → 新闻通稿（与 official 功能重叠）
# encyclopedic  百科条目 / 展馆说明               → 条目体，规范书面语
# educational   科普 / 教科书 / 教育级网页        → 讲解体，最适合小模型
# narrative     文学 / 故事 / 古诗文               → 叙事体
# technical     技术文档 / 代码                    → 精确、非宣传
# academic      论文摘要 / 数理推导                → 引用体
# general       其他通用西文 / 多语种
# qa            内置身份句 / 常识事实问答（必须保留，不参与配额）
REGIMES = ["official", "news", "encyclopedic", "educational",
           "narrative", "technical", "academic", "general", "qa"]

# 【硬上限作用类】规范书面语 / 公文 / 新闻通稿 —— 用户要求合计 ≤10%（训练 token 流口径）
FORMAL_REGIMES = ("official", "news")

# 软告警类：百科条目体同样高度同质，但有独立域预算，只做提示不做硬压
WATCH_REGIMES = ("encyclopedic",)

# 不参与单源上限的来源（人工精修 / 必须保留的接地语料）
EXEMPT_SOURCES = ("seed", "builtin-qa")

# 域内稀释底座：删掉后域内非formal token 至少保留"域总量的这个比例"。
# 【为什么必须动态】若按"初始非formal量"做静态基准，域内均衡会被地板挡死：
# 削到地板前就停 → 集中度永远降不下来。
DILUTION_FRAC = 0.25

# 域内均衡的最低收益：主数据集份额的相对改善幅度低于这个值就不动手
# （削掉的 token 不会凭空变成别的源，只是让域变小 → 白白损失语料）
MIN_BALANCE_GAIN = 0.15

# ---------------------------------------------------------------------------
# 文本归一化 + shingle
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_CJK = re.compile(r"[一-鿿]")
_WORD = re.compile(r"[A-Za-z0-9]+")


def normalize_text(t):
    """归一化：折叠空白、去零宽字符。用于精确重复与 shingle 抽取。"""
    if not t:
        return ""
    t = t.replace("", "").replace("﻿", "")
    return _WS.sub(" ", t).strip()


def _h64(s):
    """64-bit 稳定哈希（blake2b，跨进程可复现；不能用内置 hash()，它带随机盐）。"""
    if isinstance(s, str):
        s = s.encode("utf-8")
    return int.from_bytes(hashlib.blake2b(s, digest_size=8).digest(), "big")


def shingles(text, n=5, cap=20000):
    """抽取 shingle 集合（返回 np.uint64 数组，已排序去重）。

    中日韩文本用【字符 n-gram】（词边界不可靠）；西文用【词 n-gram】。
    同一篇通稿在不同站点转载时，正文几乎逐字相同 → shingle 集合 Jaccard 很高，
    这正是我们要抓的重复。
    """
    t = normalize_text(text)
    if not t:
        return np.zeros(0, dtype=np.uint64)
    cjk_ratio = len(_CJK.findall(t[:2000])) / max(len(t[:2000]), 1)
    if cjk_ratio >= 0.25:
        s = re.sub(r"\s+", "", t)
        if len(s) < n:
            grams = [s] if s else []
        else:
            grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    else:
        w = _WORD.findall(t.lower())
        if len(w) < n:
            grams = [" ".join(w)] if w else []
        else:
            grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    if len(grams) > cap:                      # 超长文档只取前 cap 个 shingle
        grams = grams[:cap]
    if not grams:
        return np.zeros(0, dtype=np.uint64)
    arr = np.fromiter((_h64(g) for g in grams), dtype=np.uint64, count=len(grams))
    return np.unique(arr)


# ---------------------------------------------------------------------------
# MinHash + LSH
# ---------------------------------------------------------------------------
_N_PERM = 64
_SEED = np.uint64(20261102)
# 每条 permutation 一个独立 salt；最终指纹 = 各 permutation 下的最小哈希
_PERM_SALT = np.array([_h64("molecule-perm-%d" % i) for i in range(_N_PERM)],
                      dtype=np.uint64)
# 二次混淆（splitmix64 变体），避免 x^salt 结构相关导致最小值分布退化
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def _mix64(arr):
    """splitmix64 混淆（numpy uint64 数组，溢出按环回处理）。"""
    with np.errstate(over="ignore"):
        x = arr.copy()
        x ^= x >> np.uint64(30)
        x *= _M1
        x ^= x >> np.uint64(27)
        x *= _M2
        x ^= x >> np.uint64(31)
    return x


def minhash(shingles_arr):
    """MinHash 指纹：_N_PERM 维 uint64 数组。

    Jaccard 估计 = 各维最小值相同的比例 → 无偏。
    """
    if shingles_arr is None or len(shingles_arr) == 0:
        return np.zeros(_N_PERM, dtype=np.uint64)
    out = np.empty(_N_PERM, dtype=np.uint64)
    for i in range(_N_PERM):
        salted = shingles_arr ^ _PERM_SALT[i]
        out[i] = _mix64(salted).min()
    return out


def _bands(sig, n_bands=4, rows=16):
    """把指纹切成 n_bands × rows 的 band，产出 LSH 桶键。"""
    keys = []
    for b in range(n_bands):
        chunk = sig[b * rows:(b + 1) * rows]
        if len(chunk) == 0:
            continue
        keys.append((b, _h64(chunk.tobytes())))
    return keys


def jaccard(a, b):
    """两个 shingle 集合的精确 Jaccard（集合已排序，用 searchsorted 求交）。"""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = len(a) + len(b) - inter
    return inter / float(union)


def _pairs_from_keys(keymap):
    """从 LSH 桶键里取候选对（同一 band 内两两配对，桶过大则跳过以防爆量）。"""
    pairs = set()
    for _key, ids in keymap.items():
        if len(ids) > 400:            # 超级大桶（超常见内容）跳过，避免 O(n^2) 爆量
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                pairs.add((a, b) if a < b else (b, a))
    return pairs


def minhash_lsh_dedupe(docs, near_thr=0.85, watch_thr=0.55, verbose=True):
    """跨源（也跨域）近重复检测与裁剪。

    docs: [{"id": str, "src": str, "dom": str, "body": str}, ...]
    返回 (kept_docs, report)
      kept_docs：裁剪后的文档列表（代表文档优先保留"更完整"的那份）
      report：{
        "n_in", "n_exact_dup", "n_near_dup", "n_watch_pairs",
        "exact_pairs_by_src": {("A","B"): n}, "near_pairs_by_src": {...},
        "kept_tokens_ratio": ...
      }

    两档阈值：
      near_thr(0.85)：判定为同一篇（转载/镜像）→ 直接裁掉重复副本
      watch_thr(0.55)：模板级孪生（同一批通稿的换标题版）→ 不裁，只报告，
                       因为这部分是"高频模板"污染的真实度量。
    裁剪策略：同一簇里保留【token 更长、字符更多】的那份（通常是更完整/更干净的原文），
    其余副本丢弃；被丢弃的副本不浪费，转入幻觉评估集（若来自 official 源）。
    """
    if not docs:
        return [], {"n_in": 0, "n_exact_dup": 0, "n_near_dup": 0,
                    "exact_pairs_by_src": {}, "near_pairs_by_src": {}}

    # ---- 1) 精确重复（归一化后 md5）----
    exact_by_hash = defaultdict(list)
    for d in docs:
        key = hashlib.md5(normalize_text(d["body"]).encode("utf-8")).hexdigest()
        exact_by_hash[key].append(d)

    # ---- 2) MinHash 指纹 + LSH 候选对 ----
    sigs = {}
    shing = {}
    for d in docs:
        sh = shingles(d["body"])
        shing[d["id"]] = sh
        sigs[d["id"]] = minhash(sh)

    keymap = defaultdict(list)
    for d in docs:
        for key in _bands(sigs[d["id"]]):
            keymap[key].append(d["id"])
    candidates = _pairs_from_keys(keymap)

    id_by_src = {}
    for d in docs:
        id_by_src[d["id"]] = d["src"]

    # ---- 3) 精确 Jaccard 判定 ----
    near_pairs = []
    watch_pairs = []
    for a, b in candidates:
        j = jaccard(shing[a], shing[b])
        if j >= near_thr:
            near_pairs.append((a, b, j))
        elif j >= watch_thr:
            watch_pairs.append((a, b, j))

    def _pair_counter(pairs):
        c = defaultdict(int)
        for a, b, _j in pairs:
            sa, sb = id_by_src.get(a, "?"), id_by_src.get(b, "?")
            c[(sa, sb) if sa <= sb else (sb, sa)] += 1
        return {("%s|%s" % k): v for k, v in sorted(c.items(), key=lambda kv: -kv[1])}

    # ---- 4) 并查集分组 → 每组只留代表 ----
    parent = {d["id"]: d["id"] for d in docs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for a, b, _j in near_pairs:
        union(a, b)
    for key, group in exact_by_hash.items():
        for i in range(1, len(group)):
            union(group[0]["id"], group[i]["id"])

    clusters = defaultdict(list)
    for d in docs:
        clusters[find(d["id"])].append(d)

    kept, dropped = [], []
    for _root, group in clusters.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        # 代表：正文最长者（更完整）；同长取 id 最小者，保证确定性
        rep = min(group, key=lambda d: (-len(normalize_text(d["body"])), d["id"]))
        # 【内容组元信息】：这一簇里的所有源其实是同一份内容的多个转载渠道。
        # 真正的重复风险不是"人民网"这个名字，而是"这份内容"本身出现了几次。
        # src 仍保留渠道名（单源上限按渠道算），但记账时把 token 平摊给簇内各渠道，
        # 这样"人民网自己有多少 token"才不会被随机地甩给新华网或央视网。
        members = sorted({g["src"] for g in group})
        rep = dict(rep)
        rep["src_members"] = members
        rep["n_merged"] = len(group)
        kept.append(rep)
        for g in group:
            if g["id"] != rep["id"]:
                dropped.append((g, rep))

    # 精确重复但没有被 near_pairs 覆盖的（短文本 shingle 不足）也算重复
    n_exact = 0
    for key, group in exact_by_hash.items():
        if len(group) > 1:
            n_exact += len(group) - 1

    report = {
        "n_in": len(docs),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "n_exact_dup": n_exact,
        "n_near_dup": len(dropped) - n_exact,
        "n_watch_pairs": len(watch_pairs),
        "exact_pairs_by_src": _pair_counter([(a, b, 1.0) for key, group in exact_by_hash.items()
                                             for a in [g["id"] for g in group]
                                             for b in [g["id"] for g in group] if a < b]),
        "near_pairs_by_src": _pair_counter(near_pairs),
        "watch_pairs_by_src": _pair_counter(watch_pairs),
        "dropped": [(g["id"], g["src"], g.get("dom", ""), rep["src"], rep["id"])
                    for g, rep in dropped],
    }
    if verbose:
        print("[dedup] 入池 %d 篇 → 保留 %d 篇（精确重复 %d + 近似重复 %d，模板孪生对 %d）"
              % (report["n_in"], report["n_kept"], report["n_exact_dup"],
                 report["n_near_dup"], report["n_watch_pairs"]))
        for k, v in list(report["near_pairs_by_src"].items())[:8]:
            print("[dedup]   近似重复来源对 %-40s %d 对" % (k, v))
    return kept, report


# ---------------------------------------------------------------------------
# 文体归类
# ---------------------------------------------------------------------------
# 文本级公文/通稿信号（源级 regime 优先；文本级用于兜底与二次确认）
_FORMAL_SIGNALS = re.compile(
    r"(会议指出|会议强调|强调要|指出，要|要求各|必须坚持|坚持以人民为中心|"
    r"深入开展|扎实推进|广泛开展|认真贯彻|全面落实|不断增强|切实加强|"
    r"本报讯|新华社记者|本报记者|通讯员|新闻发言人|"
    r"第一，|第二，|第三，|一是要|二是要|三是要|"
    r"以习近平|党的二十大|二十届三中|全会精神| Leitung)")
_ENCY_SIGNALS = re.compile(r"(，简称|又名| refers to |，是指|属于.{0,6}科|位于.{0,20}省|字[，。]|号[，。])")


def classify_regime(src_regime, text, dom=""):
    """文体归类：源级标注优先；无标注时用文本信号兜底。

    返回 REGIMES 之一。
    """
    if src_regime:
        return src_regime
    t = text or ""
    if _FORMAL_SIGNALS.search(t):
        return "official"
    if _ENCY_SIGNALS.search(t):
        return "encyclopedic"
    if dom == "code":
        return "technical"
    if dom in ("math_reason",):
        return "academic"
    if dom == "book":
        return "narrative"
    if dom == "encyclopedia":
        return "encyclopedic"
    return "general"


# ---------------------------------------------------------------------------
# 模板污染度量：高频开头/结尾 n-gram（"高频模板让实际占比远超你以为的"的直接证据）
# ---------------------------------------------------------------------------
def template_profile(texts, n=8, top=15):
    """统计最高频的段首 n-gram 与段尾 n-gram（中文按字，西文按词）。

    高频开头占比高 → 语料被少数模板主导（通稿/公文腔的典型特征）。
    """
    head = defaultdict(int)
    tail = defaultdict(int)
    total = 0
    for t in texts:
        s = normalize_text(t)
        if len(s) < n * 2:
            continue
        total += 1
        head[s[:n]] += 1
        tail[s[-n:]] += 1
    head_top = sorted(head.items(), key=lambda kv: -kv[1])[:top]
    tail_top = sorted(tail.items(), key=lambda kv: -kv[1])[:top]
    return {
        "n_docs": total,
        "head_top": [(k, v) for k, v in head_top],
        "tail_top": [(k, v) for k, v in tail_top],
        "head_top_share": (head_top[0][1] / total) if (total and head_top) else 0.0,
    }


# ---------------------------------------------------------------------------
# token 口径配额求解器
# ---------------------------------------------------------------------------
def effective_domain_weights(phase_mixes):
    """把课程阶段混比折叠成【每个域在训练 token 流里的权重】。

    phase_mixes: [{"frac": 0.5, "mix": {dom: 0.6, ...}}, ...]
    midtrain 阶段（mix=None）按"难样本池"的域分布近似：等权取各域。

    返回 {dom: weight}，权重和为 1。
    """
    doms = set()
    for ph in phase_mixes:
        doms |= set(ph.get("mix") or {})
    w = {d: 0.0 for d in doms}
    for ph in phase_mixes:
        frac = ph.get("frac", 0.0)
        mix = ph.get("mix")
        if not mix:
            if doms:
                for d in doms:
                    w[d] += frac / len(doms)
            continue
        for d, p in mix.items():
            w[d] += frac * p
    tot = sum(w.values()) or 1.0
    return {d: v / tot for d, v in w.items()}


def compute_shares(docs, tok, dom_weights, budget_tokens=None):
    """计算训练 token 流里的真实占比（不是池子占比）。

    docs: [{"id","src","dom","regime","src_members"?,"src_group"?}...]
    tok:  {doc_id: token 数}
    dom_weights: {dom: 训练流权重}

    budget_tokens : 训练预算（token 数）。用于算各域的重复率诊断。

    训练时：先按域权重抽域，再在域内均匀抽序列 → 某源在训练流中的 token 占比
      = Σ_dom  weight[dom] × (该源在该域内的 token 占比)
    这就是"标注表骗人"的数学形式，也是所有硬上限的作用对象。

    【跨源簇的 token 平摊】：一篇通稿被 3 个渠道转载时，它在训练流里的 token
    应该记在【3 个渠道】头上（各记 1/3），而不是全记在代表源头上——否则
    "单一来源占比"这个指标会被记账方式扭曲。

    【两级来源口径】src 是【采样单元】（如 SkyPile 的不同文件/档位，各为独立
    抓取批次）；src_group 是【数据集】（如 SkyPile-150B 整体）。
    单源上限作用在 src 上（因为它才是内容独立性的真实粒度），
    但数据集级集中度也必须被看见——否则"50 个 shard 都合规"会掩盖
    "一个数据集占了 30%" 的事实。两级都算，都报。
    """
    tok_by_src_dom = defaultdict(float)
    tok_by_reg_dom = defaultdict(float)
    tok_by_dom = defaultdict(float)
    tok_by_src = defaultdict(float)
    tok_by_reg = defaultdict(float)
    tok_by_grp_dom = defaultdict(float)
    tok_by_grp = defaultdict(float)
    for d in docs:
        t = float(tok.get(d["id"], 0))
        if t <= 0:
            continue
        dom, reg = d["dom"], d["regime"]
        members = d.get("src_members") or [d["src"]]
        grp = d.get("src_group") or d["src"]
        share = t / len(members)
        tok_by_dom[dom] += t
        tok_by_reg_dom[(reg, dom)] += t
        tok_by_reg[reg] += t
        tok_by_grp_dom[(grp, dom)] += t
        tok_by_grp[grp] += t
        for m in members:
            tok_by_src_dom[(m, dom)] += share
            tok_by_src[m] += share

    total_pool = sum(tok_by_dom.values()) or 1.0

    def _eff(counter, key_fn):
        """把 (key, dom) 计数折算成训练流占比。"""
        out = defaultdict(float)
        for (k, dom), t in counter.items():
            dom_t = tok_by_dom.get(dom, 0.0)
            if dom_t <= 0:
                continue
            out[key_fn(k)] += dom_weights.get(dom, 0.0) * (t / dom_t)
        return dict(out)

    src_eff = _eff(tok_by_src_dom, lambda k: k)
    reg_eff = _eff(tok_by_reg_dom, lambda k: k)
    grp_eff = _eff(tok_by_grp_dom, lambda k: k)

    # ---- 逐域的 formal 稀释诊断 ----
    # 【关键数学】一个域内 formal 占比 f = F_dom / T_dom。
    #   只删 formal 文档 → F 与 T 同减 → f 变化 (F-x)/(T-x)：
    #     - 若 F < T（含非 formal 内容），f 随 x 单调下降 → 删得动；
    #     - 若 F == T（域内 100% formal），f 恒等于 1 → 【删多少次都没用】。
    #   所以"formal 压不下去"不是算法 bug，而是该域缺少【非 formal 稀释剂】。
    #   正确动作：给该域补非 formal 内容（稀释），而不是继续删。
    dom_formal = {}
    for dom, t in tok_by_dom.items():
        f = sum(v for (r, dd), v in tok_by_reg_dom.items() if dd == dom and r in FORMAL_REGIMES)
        n = sum(v for (r, dd), v in tok_by_reg_dom.items()
                if dd == dom and r not in FORMAL_REGIMES)
        dom_formal[dom] = {
            "total": t, "formal": f, "nonformal": n,
            "formal_frac": (f / t) if t > 0 else 0.0,
            "dilutable": (f / t) < 0.999 if t > 0 else False,
        }

    # ---- 域重复率诊断：训练流分给该域的 token 数 vs 池子里实际有的 ----
    # 【为什么这个比集中度更关键】
    # 训练预算 100B token。域权重 w_dom 固定 → 该域被分配的 token = 100B × w_dom。
    # 池子里该域只有 T_dom token → 重复率 = 100B × w_dom / T_dom。
    # 重复率 > 1 意味着同一批 token 要被反复喂给模型 → 复读的温床。
    # 这个数字决定了"削语料"到底是变好还是变坏：
    # 削掉 69% 的 code 域 token → 重复率从 3700 倍涨到 9800 倍。
    budget = float(budget_tokens or 0)
    dom_repeat = {}
    for dom, t in tok_by_dom.items():
        w = dom_weights.get(dom, 0.0)
        if budget > 0 and t > 0:
            dom_repeat[dom] = {
                "pool_tokens": t,
                "budget_tokens": budget * w,
                "repeat_factor": (budget * w) / t,
            }

    return {
        "dom_repeat": dom_repeat,
        "budget_tokens": budget,
        "tok_by_dom": dict(tok_by_dom),
        "tok_by_src": dict(tok_by_src),
        "tok_by_reg": dict(tok_by_reg),
        "tok_by_grp": dict(tok_by_grp),
        "pool_tokens": total_pool,
        "src_pool_share": {s: t / total_pool for s, t in tok_by_src.items()},
        "reg_pool_share": {s: t / total_pool for s, t in tok_by_reg.items()},
        "grp_pool_share": {s: t / total_pool for s, t in tok_by_grp.items()},
        "src_eff_share": src_eff,
        "grp_eff_share": grp_eff,
        "reg_eff_share": reg_eff,
        "formal_eff_share": sum(v for k, v in reg_eff.items() if k in FORMAL_REGIMES),
        "formal_wide_eff_share": sum(
            v for k, v in reg_eff.items() if k in FORMAL_REGIMES + WATCH_REGIMES),
        "dom_formal": dom_formal,
        "dom_eff_weight": dict(dom_weights),
        "dom_pool_share": {d: t / total_pool for d, t in tok_by_dom.items()},
    }


def template_dedupe(docs, tok, head_n=8, max_share=0.05, verbose=True):
    """【源内模板去重】砍掉"同一开头 n-gram 刷屏"的文档。

    用户第 2 条动作的正面解法："高频模板让实际占比远超你以为的"。
    典型症状：TinyStories 里 17302 段有 10299 段开头都是 "Once upon a time"，
    段首模板复现率 31.9% —— 模型会学成"凡开头必是 Once upon a time"，
    这是复读的种子，而且它在池子里的真实密度远高于"看起来"的占比。

    做法：对每个源，统计【段首 head_n 字符】的 token 累计量；
    任何单个模板的 token 份额超过 max_share 时，按 drop_priority
    （模板味最重 + 篇幅短的先砍）把它压回阈值内。

    【与跨源去重的分工】跨源去重抓的是"同一篇内容被多个渠道转载"；
    这里抓的是"同一模板被同一源反复生成"——后者跨源去重看不见，
    因为每篇的 shingle 集合并不相同。

    返回 (kept_docs, report)
    """
    if not docs:
        return list(docs), {"n_in": 0, "n_dropped": 0, "by_src": {}}

    # 每个源：head_n → 该模板下的文档列表 + token 累计
    by_src = defaultdict(lambda: defaultdict(list))
    for d in docs:
        s = normalize_text(d.get("body", ""))[:head_n]
        if s:
            by_src[d["src"]][s].append(d)

    dropped = []
    for src, tmpl in by_src.items():
        src_docs = [d for d in docs if d["src"] == src]
        src_tok = sum(float(tok.get(d["id"], 0)) for d in src_docs)
        if src_tok <= 0:
            continue
        limit = src_tok * max_share
        for s, ds in tmpl.items():
            t_total = sum(float(tok.get(d["id"], 0)) for d in ds)
            if t_total <= limit:
                continue
            # 超配额：模板味最重（该模板下最短的，即信息量最低）+ 篇幅短的先砍
            ds_sorted = sorted(ds, key=lambda d: (
                len(d.get("body", "")), d["id"]))
            acc = 0.0
            for d in ds_sorted:
                t = float(tok.get(d["id"], 0))
                if acc + t <= limit:
                    acc += t
                    continue
                dropped.append(d)
    drop_ids = {d["id"] for d in dropped}
    kept = [d for d in docs if d["id"] not in drop_ids]

    report = {"n_in": len(docs), "n_dropped": len(dropped),
              "by_src": {s: sum(1 for d in dropped if d["src"] == s)
                         for s in sorted({d["src"] for d in dropped})},
              "head_n": head_n, "max_share": max_share}
    if verbose:
        print("[tmpl] 源内模板去重：%d → %d 篇（段首 %d 字符模板份额上限 %.1f%%）"
              % (len(docs), len(kept), head_n, max_share * 100))
        for s, n in sorted(report["by_src"].items(), key=lambda kv: -kv[1])[:8]:
            print("[tmpl]   %-34s 砍 %d 篇" % (s[:34], n))
    return kept, report
def enforce_caps(docs, tok, dom_weights, source_cap=0.005, formal_cap=0.10,
                 exempt=EXEMPT_SOURCES, min_domain_tokens=None, max_iter=40,
                 dom_max_group_share=None, verbose=True):
    """配额求解：把超限的文档裁掉，直到训练 token 流满足上限。

      source_cap  : 单一来源在训练 token 流中的占比上限（默认 0.5%）
      formal_cap  : 规范书面语/公文/新闻通稿合计在训练 token 流中的占比上限（默认 10%）
      min_domain_tokens : {dom: 最少保留 token}（域地板）

    【单一来源的口径 —— 作用在数据集级，不是采样单元级】
    用户说"对单一来源设硬上限"，"来源"指的是内容供给方（SkyPile-150B 整体、
    人民网、the-stack-dedup），不是"我们从它抽的第几个分片"。
    如果按采样单元算，把一个数据集拆成 20 个 shard 就能让每个都"合规"——
    这是自欺欺人，真实集中度一点没降。
    所以：裁剪按【数据集】为单位做（src_group），报告里同时给出
    采样单元级与数据集级两个口径的数字，让两边都看得见。

    【可达性判定用"域地板下的严格下界"，不是拍脑袋的估计】
    某数据集 g 在域 d 里，至少要留下
        max(0, floor[d] - 域内其他数据集的 token)
    个 token（否则域地板被打破）。因此 g 在训练流中的占比有严格下界
        L(g) = Σ_d  w_d × keep_g,d / T_d
    若 L(g) > cap，则这个上限在当前语料结构下【数学不可达】——
    继续裁剪只会把域砍空（域权重固定，砍空不改变权重分配，
    只会让采样器在该域内反复循环同一批 token → 复读）。
    此时停止裁剪并如实报告"可达上限 + 缺口 + 补多少个独立源才可能达标"。
    这本身就是最有价值的信息：它告诉你"要满足这个上限还缺多少独立内容"。

    裁剪优先级：先砍【模板味最重】的文档（段首 n-gram 复现最多的，即通稿套话），
    保住说法多样的样本。
    返回 (kept_docs, report)
    """
    docs = list(docs)
    alive = {d["id"]: d for d in docs}
    dropped = []
    floors = dict(min_domain_tokens or {})
    conc_blocked = None

    def group_of(d):
        """数据集级口径：优先 src_group（数据集），回退 src（采样单元）。"""
        return d.get("src_group") or d["src"]

    # 全局段首频次（模板味度量）
    head_freq = defaultdict(int)
    for d in docs:
        s = normalize_text(d.get("body", ""))[:8]
        if s:
            head_freq[s] += 1

    def drop_priority(d):
        """数值越小越该先被砍：模板味最重 + 篇幅短（信息量低）。"""
        return (-head_freq.get(normalize_text(d.get("body", ""))[:8], 0),
                len(d.get("body", "")))

    def dom_tokens_now():
        t = defaultdict(float)
        for d in alive.values():
            t[d["dom"]] += float(tok.get(d["id"], 0))
        return t

    def dom_doc_count():
        c = defaultdict(int)
        for d in alive.values():
            c[d["dom"]] += 1
        return c

    def dom_nonformal_now():
        n = defaultdict(float)
        for d in alive.values():
            if d["regime"] not in FORMAL_REGIMES:
                n[d["dom"]] += float(tok.get(d["id"], 0))
        return n

    def can_delete(d, cur_dom, cur_cnt, cur_nonformal=None):
        """删除前的保护检查：
        (a) 域地板（绝对下限）：删掉后域内 token 不能低于 floor；
        (b) 域非空：域权重固定，空域 → 采样器在该域内反复循环同一批 token → 复读；
        (c) 稀释底座（动态）：删掉后域内【非formal】token 不能低于
            min(当前非formal量, 删后域总量 × DILUTION_FRAC)。
            这一项必须是【动态】的：用初始值做静态基准会与"域内均衡"
            形成死锁（削到地板前就被挡住，均衡永远推不动）。
        """
        dom = d["dom"]
        t_doc = float(tok.get(d["id"], 0))
        fl = floors.get(dom)
        if fl is not None and cur_dom[dom] - t_doc < fl:
            return False
        if cur_cnt.get(dom, 0) <= 1:
            return False
        if cur_nonformal is not None and d["regime"] not in FORMAL_REGIMES:
            nf = cur_nonformal.get(dom, 0.0)
            need = min(nf, max(0.0, (cur_dom[dom] - t_doc)) * DILUTION_FRAC)
            if nf - t_doc < need:
                return False
        return True

    def groups_by_dom(docs_):
        out = defaultdict(set)
        for d in docs_:
            out[d["dom"]].add(group_of(d))
        return out

    def dom_balance_plan(dom, cap_share, F, T, s):
        """域 d 的域内均衡方案：给出【可达到的最大份额】与【允许的削减量】。

        【数学】设域 d 内 formal token 为 F、总量 T，单一数据集占 s·T
        （它是非formal 主体）。削掉 x 个非formal token 后：
            新份额  s' = (sT - x)/(T - x) ≤ c
                 ⟺  x ≥ T(s-c)/(1-c)                          ……（均衡目标）
            formal 约束（域权重 w_d 固定，削非formal 会抬高域内 formal 占比）：
                 w_d · F/(T-x) ≤ f_cap
                 ⟺  T-x ≥ F·w_d/f_cap
                 ⟺  x ≤ T - F·w_d/f_cap                       ……（formal 上限）
        两者联立：
            x_min = T(s-c)/(1-c)      （达到目标份额所需的最少削减）
            x_max = T - F·w_d/f_cap   （formal 上限允许的最大削减）
        若 x_min ≤ x_max → 无冲突，c_eff = cap_share，x = x_min
        否则冲突 → 只能削到 x_max，c_eff = (sT-x_max)/(T-x_max)
        返回 (c_eff, x_target)
        """
        w = dom_weights.get(dom, 0.0)
        if w <= 0 or T <= 0:
            return (cap_share, 0.0)
        x_max = T - F * w / formal_cap if formal_cap > 0 else T
        x_max = max(0.0, min(x_max, T))
        c = cap_share
        if c >= 1.0:
            return (c, 0.0)
        x_min = T * (s - c) / (1.0 - c)
        if x_min <= 0:
            return (c, 0.0)
        if x_min <= x_max:
            return (c, x_min)
        T2 = T - x_max
        c_eff = ((s * T - x_max) / T2) if T2 > 0 else s
        return (max(c_eff, s), x_max)

    def dom_feasible(dom, docs_, cap):
        """域 d 内的【来源上限】在当前结构下是否可能达标。

        域 d 的训练流权重 w_d 是固定的（课程配比），域内有 n_d 个独立数据集，
        任一数据集在域内的 token 份额 ≥ 1/n_d（ pigeonhole 原理），
        因此它在训练流中的占比 ≥ w_d / n_d。
        若 w_d / n_d > cap，则该域内【没有任何数据集】能达标 →
        对该域做来源裁剪不仅没用，还会破坏稀释底座和多样性。
        返回 (feasible: bool, bound: w_d/n_d)
        """
        n = len({group_of(d) for d in docs_ if d["dom"] == dom})
        w = dom_weights.get(dom, 0.0)
        if n <= 0:
            return (False, w)
        bound = w / n
        return (bound <= cap * 1.001, bound)

    def exclusive_dom_share(grp, docs_):
        """grp 被【域独占】强制的下界份额。

        若 grp 是域 d 内唯一的数据集，则域 d 的全部 token 都归它，
        而域权重 w_d 是固定的（课程配比）→ 删文档不改变 w_d，
        → grp 在训练流中的占比至少是 w_d。这一项是硬的。
        """
        tot = defaultdict(float)
        mine = defaultdict(float)
        other = defaultdict(float)
        for d in docs_:
            t = float(tok.get(d["id"], 0))
            if t <= 0:
                continue
            tot[d["dom"]] += t
            if group_of(d) == grp:
                mine[d["dom"]] += t
            else:
                other[d["dom"]] += t
        lb = 0.0
        for dom, T in tot.items():
            if T <= 0:
                continue
            if other.get(dom, 0.0) <= 0:
                lb += dom_weights.get(dom, 0.0)
        return lb

    def min_achievable_share():
        """忽略域地板时的理想下界：max over doms of (w_dom / n_dom)。
        n_dom = 该域内【独立数据集】数（内容组已合并转载）。"""
        n_by_dom = groups_by_dom(list(alive.values()))
        vals = []
        for dom, w in dom_weights.items():
            n = len(n_by_dom.get(dom, ()))
            if n:
                vals.append(w / n)
        return max(vals) if vals else 0.0

    def try_trim_group(grp, cap):
        """把数据集 grp 削到全局占比 ≤ cap。返回实际削掉的文档数。"""
        killed = 0
        for _round in range(max_iter):
            alive_docs = list(alive.values())
            sh = compute_shares(alive_docs, tok, dom_weights)
            share = sh["grp_eff_share"].get(grp, 0.0)
            if share <= cap * 1.001:
                break
            if exclusive_dom_share(grp, alive_docs) > cap * 1.001:
                break        # 被域独占强制 → 数学不可达，停止裁剪，保住语料
            # 【域级可行性预检】该 group 所在的所有域都必须"有可能达标"，
            # 否则这个 group 无论怎么砍都超限 → 砍了只是白砍，
            # 还会毁掉该域的稀释底座与非formal多样性。
            if not all(dom_feasible(dom, alive_docs, cap)[0]
                       for dom in {d["dom"] for d in alive_docs
                                   if group_of(d) == grp}):
                break
            cand = [d for d in alive_docs if group_of(d) == grp]
            # 保底：每个 (group, dom) 至少留 1 篇 —— 否则删光 A 会让 B 变成域独占，
            # 域内 group 数下降 → 下限上升 → 越删越不可达（自我拆台），
            # 而且会毁掉该域的稀释底座。
            cnt_by_dom = defaultdict(int)
            for d in cand:
                cnt_by_dom[d["dom"]] += 1
            keep_needed = set()
            for dom, c in cnt_by_dom.items():
                for d in cand:
                    if d["dom"] == dom and d["id"] not in keep_needed:
                        keep_needed.add(d["id"])
                        cnt_by_dom[dom] -= 1
                        if cnt_by_dom[dom] <= 0:
                            break
            cand = [d for d in cand if d["id"] not in keep_needed]
            if not cand:
                break
            cand.sort(key=drop_priority)
            # 域地板 / 域非空检查
            cur_dom = dom_tokens_now()
            cur_cnt = dom_doc_count()
            cur_nf = dom_nonformal_now()
            step = max(1, len(cand) // 8)
            progressed = False
            for d in cand[:step]:
                if not can_delete(d, cur_dom, cur_cnt, cur_nf):
                    continue
                dom = d["dom"]
                t_doc = float(tok.get(d["id"], 0))
                alive.pop(d["id"], None)
                dropped.append(d)
                killed += 1
                cur_dom[dom] -= t_doc
                cur_cnt[dom] -= 1
                if d["regime"] not in FORMAL_REGIMES:
                    cur_nf[dom] -= t_doc
                progressed = True
            if not progressed:
                break
        return killed

    def trim_formal(cap):
        """压"规范书面语/公文/新闻通稿"合计占比到 ≤cap。

        【关键 1：每轮都重算真实占比】而不是用解析式估算边际效应——
        因为砍掉一篇 formal 文档同时会改变该域的总量 T_dom，
        占比是非线性的，估算式会系统性低估残余超标量，导致提前停止。

        【关键 2：域地板只保护"非 formal 的稀释底座"】
        域地板的原始目的是防止把某个域砍空（域权重固定，砍空不改变权重分配，
        只会让采样器在该域内反复循环同一批 token → 复读）。
        但如果该域内容 100% 是 formal，那么砍 formal 会让域变空 → 地板挡死 →
        formal 永远压不下去。
        【数学事实】域内 formal 占比 f = F/T，只删 formal 文档 → F 与 T 同减：
          - f < 1 时：f 单调下降（删得动）
          - f = 1 时：f 恒等于 1（删不动，必须靠【补非 formal 内容稀释】）
        所以正确策略是：
          - 对 f < 1 的域：正常删 formal（每轮重算占比，步长 1/8，源轮转保多样性）
          - 对 f = 1 的域：不删（删了只会把域清空），标记为"需稀释"
        返回 (删除数, 需稀释的域列表)
        """
        killed_total = 0
        need_dilute = set()
        for _round in range(max_iter):
            alive_docs = list(alive.values())
            sh = compute_shares(alive_docs, tok, dom_weights)
            if sh["formal_eff_share"] <= cap * 1.001:
                break
            formal_docs = [d for d in alive_docs
                           if d["regime"] in FORMAL_REGIMES
                           and d["src"] not in exempt and d["id"] in alive]
            if not formal_docs:
                break
            # 【只删"可稀释"的域里的 formal】——纯 formal 域删不动，硬删会清空域
            by_dom = defaultdict(list)
            for d in formal_docs:
                by_dom[d["dom"]].append(d)
            dom_info = sh["dom_formal"]
            deletable = []
            for dom, ds in by_dom.items():
                info = dom_info.get(dom)
                if info and info["total"] > 0 and info["formal_frac"] >= 0.999:
                    need_dilute.add(dom)     # 纯 formal 域 → 只能靠补内容稀释
                    continue
                deletable.extend(ds)
            if not deletable:
                break
            # 源轮转排序：同源内按模板味排，源之间轮流取
            by_src = defaultdict(list)
            for d in deletable:
                by_src[d["src"]].append(d)
            for v in by_src.values():
                v.sort(key=drop_priority)
            order, srcs = [], sorted(by_src)
            i = 0
            while any(by_src[s] for s in srcs):
                s = srcs[i % len(srcs)]
                if by_src[s]:
                    order.append(by_src[s].pop(0))
                i += 1
            # 砍掉前 1/8（保守步长），重算后再决定下一轮
            n_kill = max(1, len(order) // 8)
            cur_dom = dom_tokens_now()
            cur_cnt = dom_doc_count()
            cur_nf = dom_nonformal_now()
            killed = 0
            for d in order[:n_kill]:
                if not can_delete(d, cur_dom, cur_cnt, cur_nf):
                    continue
                dom = d["dom"]
                t_doc = float(tok.get(d["id"], 0))
                alive.pop(d["id"], None)
                dropped.append(d)
                cur_dom[dom] -= t_doc
                cur_cnt[dom] -= 1
                if d["regime"] not in FORMAL_REGIMES:
                    cur_nf[dom] -= t_doc
                killed += 1
            if killed == 0:
                break        # 全被域地板挡住 → 无法再压，停止
            killed_total += killed
        return killed_total, sorted(need_dilute)

    def trim_dom_concentration(cap_share, cap):
        """【域内集中度削减】——用 formal 余量换来源多样性。

        【为什么这是最值钱的一刀】
        用户要"单一来源 ≤0.5%"，但域权重固定（zh_general 43.67%）→
        数学上需要 87 个独立源才可能达标 → 硬上限不可达。
        但"不可达"不等于"无事可做"。看真实结构：
          zh_general 域内 SkyPile 占 ~92% 的 token，
          而 20+ 个真正多样的中文源（人民网/新华网/第一财经/中国新闻网…）
          加起来只占 ~8% → 它们在训练流里几乎学不到。
        换句话说：不是"源不够多"，而是"权重全被一个源吃掉了"。

        【怎么做才是"减法"】
        削掉超限源【多出来的】token，而不是新增源。削掉的域内份额会被
        域内其他源自然接管 → 训练流里其他源的占比上升（多样性↑），
        同时 formal 的相对占比上升（所以必须盯着 formal 上限）。

        【与 formal 约束的耦合】
        削掉的通常是非 formal 内容（超限源往往就是最好的通用语料），
        → 域内 formal 占比 f 上升 → f 逼近 1 时 formal 上限就压不住了。
        所以这一步必须受 formal 上限约束：
        每削一轮就重算 formal_eff_share，一旦超标就停手。
        返回 (删除数, 受阻原因)
        """
        killed_total = 0
        blocked = None
        for _round in range(max_iter):
            alive_docs = list(alive.values())
            sh = compute_shares(alive_docs, tok, dom_weights)
            if sh["formal_eff_share"] > cap * 1.001:
                blocked = "formal 已达上限，停止域内均衡"
                break
            # 找超限的 (域, 数据集)
            tok_dom = defaultdict(float)
            tok_dom_grp = defaultdict(float)
            for d in alive_docs:
                t = float(tok.get(d["id"], 0))
                if t <= 0:
                    continue
                tok_dom[d["dom"]] += t
                tok_dom_grp[(d["dom"], group_of(d))] += t
            # 各域的 formal 量（用于联立求解）
            F_dom = defaultdict(float)
            for d in alive_docs:
                t = float(tok.get(d["id"], 0))
                if t > 0 and d["regime"] in FORMAL_REGIMES:
                    F_dom[d["dom"]] += t
            over = []
            for (dom, grp), t in tok_dom_grp.items():
                T = tok_dom.get(dom, 0.0)
                if T <= 0:
                    continue
                sh_dom = t / T
                c_eff, x_target = dom_balance_plan(dom, cap_share,
                                                   F_dom.get(dom, 0.0),
                                                   T, sh_dom)
                if sh_dom <= c_eff * 1.001 or x_target <= 0:
                    continue
                # 【均衡收益判据】只有当"能削到的份额"相对现状有显著改善时
                # 才值得动手。否则这一刀是纯浪费：
                #   削掉的 token 不会变成别的源（token 不能凭空创造），
                #   只会让整个域变小 → 训练流里该域权重不变，
                #   其他源的域内份额虽然上升，但绝对量没变。
                # 典型例子：zh_general 域内 SkyPile 97.2%，
                #   而 formal 上限（域权重 43.67%）把均衡目标锁死在 94.2%
                #   → 削 4.3M token 只换 3 个百分点的份额变化 → 不做。
                gain = (sh_dom - c_eff) / sh_dom if sh_dom > 0 else 0.0
                if gain < MIN_BALANCE_GAIN:
                    continue
                over.append((sh_dom, dom, grp, c_eff, x_target))
            if not over:
                break
            over.sort(reverse=True)
            cur_dom = dom_tokens_now()
            cur_cnt = dom_doc_count()
            cur_nf = dom_nonformal_now()
            killed = 0
            for _sh, dom, grp, c_eff, x_target in over:
                # 该数据集要削到域内份额 = c_eff
                T = tok_dom.get(dom, 0.0)
                cur_grp = tok_dom_grp.get((dom, grp), 0.0)
                x_left = x_target
                if x_left <= 0:
                    continue
                cand = [d for d in alive.values()
                        if d["dom"] == dom and group_of(d) == grp]
                cand.sort(key=drop_priority)
                for d in cand:
                    if x_left <= 0:
                        break
                    if not can_delete(d, cur_dom, cur_cnt, cur_nf):
                        continue
                    t_doc = float(tok.get(d["id"], 0))
                    alive.pop(d["id"], None)
                    dropped.append(d)
                    cur_dom[dom] -= t_doc
                    cur_cnt[dom] -= 1
                    if d["regime"] not in FORMAL_REGIMES:
                        cur_nf[dom] -= t_doc
                    cur_grp -= t_doc
                    tok_dom[dom] -= t_doc
                    x_left -= t_doc
                    killed += 1
                # 削完立刻检查 formal 是否被顶破
                sh2 = compute_shares(list(alive.values()), tok, dom_weights)
                if sh2["formal_eff_share"] > cap * 1.001:
                    blocked = "削 %s 时 formal 顶到上限（%.2f%%）" % (
                        grp[:28], sh2["formal_eff_share"] * 100)
                    break
            killed_total += killed
            if blocked:
                break
            if killed == 0:
                break
        return killed_total, blocked

    for _it in range(max_iter):
        alive_docs = list(alive.values())
        sh = compute_shares(alive_docs, tok, dom_weights)

        # --- 1) 治规范书面语合计超标（最硬的约束，先做）---
        if sh["formal_eff_share"] > formal_cap * 1.001:
            killed, need_dilute = trim_formal(formal_cap)
            if killed:
                continue
            # 删不动 → 如实报告"需稀释"，不砍空语料
            break

        # --- 2) 域内集中度均衡（用 formal 余量换来源多样性）---
        if dom_max_group_share is not None:
            killed, blocked = trim_dom_concentration(dom_max_group_share,
                                                     formal_cap)
            if killed:
                conc_blocked = blocked
                continue
            if blocked:
                conc_blocked = blocked
                break

        # --- 2) 治超限【数据集】（按超标倍数排序，先治最严重的）---
        bad = [(g, v) for g, v in sh["grp_eff_share"].items()
               if g not in exempt and v > source_cap * 1.001]
        bad.sort(key=lambda kv: -kv[1])
        killed = 0
        for grp, _v in bad:
            killed += try_trim_group(grp, source_cap)
        if killed:
            continue
        break

    final = list(alive.values())
    final_sh = compute_shares(final, tok, dom_weights)

    # ---- 可行性分析：单源（数据集级）全局占比的数学下限 ----
    n_grp_by_dom = groups_by_dom(final)
    floor_share = {}
    for dom, w in dom_weights.items():
        n = len(n_grp_by_dom.get(dom, ()))
        if n:
            floor_share[dom] = w / n
    global_floor = max(floor_share.values()) if floor_share else 0.0

    over_src = [(s, v) for s, v in final_sh["src_eff_share"].items()
                if s not in exempt and v > source_cap * 1.001]
    over_grp = [(g, v) for g, v in final_sh["grp_eff_share"].items()
                if g not in exempt and v > source_cap * 1.001]

    # ---- 逐数据集的严格下界：区分"能压但没压够" vs "数学不可达" ----
    grp_bounds = {}
    for g, v in sorted(final_sh["grp_eff_share"].items(), key=lambda kv: -kv[1]):
        lb = exclusive_dom_share(g, final)
        grp_bounds[g] = {
            "share": v,
            "lower_bound": lb,
            "reachable": lb <= source_cap * 1.001,
            "n_units": sum(1 for d in final if group_of(d) == g),
            "doms": sorted({d["dom"] for d in final if group_of(d) == g}),
        }

    # ---- 稀释诊断：哪些域是"纯 formal 域"（删不动）----
    pure = []
    for dom, info in final_sh["dom_formal"].items():
        if info["total"] > 0 and info["formal_frac"] >= 0.999:
            pure.append({"dom": dom, "tokens": int(info["total"]),
                         "formal_frac": info["formal_frac"]})

    report = {
        "n_before": len(docs),
        "n_after": len(final),
        "n_dropped": len(dropped),
        "dropped_by_src": {s: sum(1 for d in dropped if d["src"] == s)
                           for s in sorted({d["src"] for d in dropped})},
        "dropped_by_group": {g: sum(1 for d in dropped if group_of(d) == g)
                             for g in sorted({group_of(d) for d in dropped})},
        "dropped_by_regime": {r: sum(1 for d in dropped if d["regime"] == r)
                              for r in sorted({d["regime"] for d in dropped})},
        "source_cap": source_cap,
        "formal_cap": formal_cap,
        "final_formal_eff_share": final_sh["formal_eff_share"],
        "final_formal_wide_eff_share": final_sh["formal_wide_eff_share"],
        "final_src_eff_share": final_sh["src_eff_share"],
        "final_grp_eff_share": final_sh["grp_eff_share"],
        "final_reg_eff_share": final_sh["reg_eff_share"],
        "final_tok_by_dom": final_sh["tok_by_dom"],
        "final_reg_pool_share": final_sh["reg_pool_share"],
        "dom_formal": final_sh["dom_formal"],
        "pure_formal_doms": pure,
        # ---- 可行性 ----
        "source_cap_feasible": not over_grp,
        "min_achievable_src_share": global_floor,
        "group_bounds": grp_bounds,
        "groups_per_dom": {d: len(s) for d, s in n_grp_by_dom.items()},
        "groups_needed_for_cap": {d: (w / source_cap if source_cap else 0)
                                  for d, w in dom_weights.items()},
        "src_per_dom": {d: len(s) for d, s in
                        defaultdict(set, {d: {m for x in final if x["dom"] == d
                                              for m in (x.get("src_members") or [x["src"]])}
                                          for d in {x["dom"] for x in final}}).items()},
        "over_cap_sources": [{"src": s, "share": v} for s, v in over_src],
        "over_cap_groups": [{"group": g, "share": v} for g, v in over_grp],
        "dom_max_group_share": dom_max_group_share,
        "dom_concentration_blocked": conc_blocked,
        "dom_group_share": _dom_group_share(final, tok),
    }
    if verbose:
        print("[cap] 配额裁剪：%d → %d 篇（数据集上限 %.2f%% / 规范书面语上限 %.1f%%）"
              % (len(docs), len(final), source_cap * 100, formal_cap * 100))
        print("[cap] 规范书面语(公文+新闻通稿) 训练流占比：%.2f%%（上限 %.1f%%）%s"
              % (final_sh["formal_eff_share"] * 100, formal_cap * 100,
                 "达标" if final_sh["formal_eff_share"] <= formal_cap * 1.001 else "未达标"))
        if pure:
            print("[cap] ⚠ 纯 formal 域（域内 100%% 通稿，删不动）：%s"
                  % "，".join("%s(%d tok)" % (p["dom"], p["tokens"]) for p in pure))
            print("[cap]   → 只能靠【补该域的非 formal 内容】稀释，不能靠删")
        worst = sorted(((v, s) for s, v in final_sh["grp_eff_share"].items()
                        if s not in exempt), reverse=True)[:3]
        if worst:
            print("[cap] 数据集级最高占比：" + "，".join("%s %.3f%%" % (s, v * 100)
                                                         for v, s in worst))
        # 可达性：区分"没压够"和"压不动"
        unreachable = [(g, b) for g, b in grp_bounds.items()
                       if b["share"] > source_cap * 1.001 and not b["reachable"]]
        if unreachable:
            print("[cap] ⚠ 数据集上限 %.2f%% 在当前源结构下【数学不可达】："
                  % (source_cap * 100))
            for g, b in sorted(unreachable, key=lambda kv: -kv[1]["share"])[:5]:
                print("[cap]   %-28s 现状 %.3f%%，严格下界 %.3f%%（受域地板约束）"
                      % (g[:28], b["share"] * 100, b["lower_bound"] * 100))
            need = max(report["groups_needed_for_cap"].values()) if report["groups_needed_for_cap"] else 0
            have = max(report["groups_per_dom"].values()) if report["groups_per_dom"] else 0
            print("[cap]   理想下界（忽略地板）%.3f%%：权重最大的域需 %.0f 个独立数据集才可能达标，当前 %d 个"
                  % (global_floor * 100, need, have))
            print("[cap]   → 要达标只能【补真正独立的内容源】，不能靠切 shard 假装多样")
    return final, report



def _dom_group_share(docs, tok=None):
    """每个域内各数据集的 token 份额（域内集中度诊断）。"""
    tot = defaultdict(float)
    grp = defaultdict(float)
    for d in docs:
        t = float(tok.get(d["id"], 0)) if tok else 1.0
        tot[d["dom"]] += t
        grp[(d["dom"], d.get("src_group") or d["src"])] += t
    out = defaultdict(dict)
    for (dom, g), t in grp.items():
        T = tot.get(dom, 0.0)
        if T > 0:
            out[dom][g] = t / T
    return {k: dict(sorted(v.items(), key=lambda kv: -kv[1]))
            for k, v in out.items()}


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
def render_report(shares, src_cap=0.005, formal_cap=0.10, top=20):
    """把统计结果渲染成人能读的文本（写进 data/v3_corpus_report.txt）。"""
    L = []
    A = L.append
    A("=" * 78)
    A("molecule v3.0 语料真实分布审计（按 token 数口径）")
    A("=" * 78)
    A("")
    A("【一】池子 token 结构")
    A("  池内总 token: %d" % int(shares["pool_tokens"]))
    A("")
    A("  按域（池内占比 / 训练流权重）：")
    for d in sorted(shares["tok_by_dom"], key=lambda k: -shares["tok_by_dom"][k]):
        A("    %-16s %10d tok   池内 %6.2f%%   训练流权重 %6.2f%%"
          % (d, int(shares["tok_by_dom"][d]),
             shares["dom_pool_share"].get(d, 0) * 100,
             shares["dom_eff_weight"].get(d, 0) * 100))
    A("")
    A("【二】文体（regime）在【训练 token 流】中的真实占比")
    A("  ← 这才是真正喂给模型的比例；配比表上的数字不是它")
    for r in sorted(shares["reg_eff_share"], key=lambda k: -shares["reg_eff_share"][k]):
        flag = ""
        if r in FORMAL_REGIMES:
            flag = "  ← 受限类（合计上限 %.0f%%）" % (formal_cap * 100)
        elif r in WATCH_REGIMES:
            flag = "  ← 软告警"
        A("    %-16s 训练流 %6.2f%%   池内 %6.2f%%%s"
          % (r, shares["reg_eff_share"][r] * 100,
             shares["reg_pool_share"].get(r, 0) * 100, flag))
    A("    %-16s 训练流 %6.2f%%  （上限 %.0f%%，%s）"
      % ("[规范书面语合计]", shares["formal_eff_share"] * 100, formal_cap * 100,
         "达标" if shares["formal_eff_share"] <= formal_cap + 1e-9 else "超标"))
    A("")
    A("【三】单一来源 = 【数据集级】（训练流 token 占比，上限 %.1f%%）" % (src_cap * 100))
    A("  ← 口径说明：同一数据集的多个采样单元（shard）合并计算")
    A("     把一个数据集拆成 N 个 shard 就能让每个都『合规』，那是自欺欺人")
    grows = sorted(shares["grp_eff_share"].items(), key=lambda kv: -kv[1])[:top]
    for g, v in grows:
        flag = "  ← 超限" if v > src_cap * 1.001 and g not in EXEMPT_SOURCES else ""
        A("    %-34s %6.3f%%%s" % (g[:34], v * 100, flag))
    over = [(g, v) for g, v in shares["grp_eff_share"].items()
            if v > src_cap * 1.001 and g not in EXEMPT_SOURCES]
    if over:
        A("    !! 超限数据集 %d 个：%s" % (len(over), ", ".join(g for g, _ in over[:10])))
    else:
        A("    所有数据集均在上限内")
    A("")
    A("【四】采样单元级（抓取批次）占比 —— 仅作诊断，不作为合规口径")
    rows = sorted(shares["src_eff_share"].items(), key=lambda kv: -kv[1])[:top]
    for s, v in rows:
        A("    %-34s %6.3f%%" % (s[:34], v * 100))
    A("")
    A("【五】逐域 formal 稀释诊断")
    A("  域内 formal 占比 f = F/T；只删 formal 文档时 f=(F-x)/(T-x)：")
    A("    f<1 → f 单调下降（删得动）；f=1 → f 恒为 1（删不动，只能靠补非 formal 内容稀释）")
    for dom in sorted(shares["dom_formal"],
                      key=lambda d: -shares["dom_formal"][d]["total"]):
        info = shares["dom_formal"][dom]
        flag = "  ← 纯 formal 域，需稀释" if info["formal_frac"] >= 0.999 else ""
        A("    %-16s 总 %8d tok  formal %8d (%5.1f%%)  非formal %8d%s"
          % (dom, int(info["total"]), int(info["formal"]),
             info["formal_frac"] * 100, int(info["nonformal"]), flag))
    A("")
    A("【六】域内集中度（域内各数据集的 token 份额）")
    A("  ← 训练流占比是『域权重 × 域内份额』，所以域内份额才是可操作的那一半")
    dom_gs = shares.get("dom_group_share") or shares.get("_dom_group_share") or {}
    if dom_gs:
        for dom in sorted(dom_gs):
            rows = list(dom_gs[dom].items())[:6]
            if not rows:
                continue
            A("    %-14s %s" % (dom, "，".join("%s %.1f%%" % (g[:22], v * 100)
                                                for g, v in rows)))
    A("")
    A("【七】域重复率（训练流分给该域的 token ÷ 池子里实际有的）")
    A("  ← 决定『削语料』到底是变好还是变坏：削掉 token → 域变小 → 重复率升高")
    A("  ← 注意：这是【当前抽样规模】下的数字（CI 每源只抓几 MB），")
    A("     它证明的是『抽样规模离 100B 还差几个数量级』，")
    A("     不是『语料本身不足』。它同时说明：在真实全量语料到位前，")
    A("     继续【削】语料是错的方向——只会把重复率推得更高。")
    rep = shares.get("dom_repeat") or {}
    if rep:
        for dom in sorted(rep, key=lambda k: -rep[k]["repeat_factor"]):
            r = rep[dom]
            flag = "  ← 严重复读风险" if r["repeat_factor"] > 50 else ""
            A("    %-14s 池内 %9d tok   需喂 %12.0f tok   重复 %8.1f 倍%s"
              % (dom, int(r["pool_tokens"]), r["budget_tokens"],
                 r["repeat_factor"], flag))
    else:
        A("    （未提供训练预算，跳过）")
    A("")
    return "\n".join(L)
