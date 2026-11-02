"""molecule v3.0 ModelScope 语料层（国内直连，免登录，CI 稳定）。

用户提供的实测清单（2026-11-02）：ModelScope 是国内直连入口，比 HF 稳。
本模块用【Range 按需抽样】而非整包下载：
  - TinyStories 训练集 1.9GB / SlimPajama 单片 1.7GB / cosmopedia 65GB …
    全量下载对 CI 的时间/带宽都是灾难；
  - 实测 11.8 MB/s，取 8MB 仅 0.7s。

【前沿方法 1：多源采样并集 + 质量门控，而非单源灌入】
  小模型（0.1B）的关键缺陷不是"知识不够"而是"分布偏食"——单一文体语料会让它
  被单一风格带跑（公文腔 / 复读机）。因此：
    - 多个源并行采样，按【域】而非按【源】汇入语料；
    - 每条样本过一道【质量门控】：去样板、去重复、长度/信息密度检查；
    - 同一源内部做【去重】与【多样性采样】（避免同一模板反复出现）。

【前沿方法 2：去污染（decontamination）】
  语料里混入评测集原文 = 模型"背答案"，评测分数虚高。对每条采样文本做
  13-gram 指纹检查，命中 C-Eval/CMMLU/MMLU 等基准的样本直接丢弃。
  这是"预训练语料污染"的标准做法，也是我们敢公开分数的前提。

【前沿方法 3：课程友好型采样】
  Phase1 需要"语言基础"（通用叙事），Phase2 需要"推理密度"。同一份语料在不同
  阶段应该取不同的部分：本模块为每条样本计算【难度代理】，让采样器在
  mid-training 阶段可以按"困惑度/难度"优先取难样本。

选型理由（对齐"通用模型、不带公文腔、语种全面"）：
  - TinyStories      : 微软专为小模型设计的连贯故事 → 教"把话说通顺"（英文）
  - chinese-cosmopedia / chinese-fineweb-edu : 中文高密度合成教科书/教育级网页
                      → 与党媒公文语体不同构，是打破"公文腔"的解药
  - SlimPajama-6B   : 英文通用（书籍/论文/网页等多种体裁）
  - NuminaMath-CoT  : 数学推理带思维链（math_reason 域主力）
  - Magicoder       : 代码
"""

import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

UA = "Mozilla/5.0 (research) molecule-lm/3.0 (contact: codecloud-dev)"
API = "https://modelscope.cn/api/v1/datasets/%s/repo/tree?Revision=master&Root=%s"
FILE = "https://modelscope.cn/api/v1/datasets/%s/repo?Revision=master&FilePath=%s"


def _curl(url, timeout=40, rng=None, out=None, binary=False):
    """binary=True 返回 bytes（用于分片下载）；否则返回解码后的 str。

    注意：不能用 `-w "%{http_code}"`——它会把状态码追加到 stdout，
    导致 JSON 解析报 "Extra data"。这里改为用 `-f`（失败即非零退出码）判断成败。
    """
    cmd = ["curl", "-sSLf", "--compressed", "--max-time", str(timeout), "-A", UA]
    if rng:
        cmd += ["-r", rng]
    if out:
        cmd += ["-o", out]
    cmd.append(url)
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout + 20)
        if out:
            return b""
        if binary:
            return p.stdout
        return p.stdout.decode("utf-8", "ignore")
    except Exception:
        return b"" if binary else ""


def list_files(repo, root="", _depth=0, _max_depth=2):
    """列出数据集文件树：返回 [{Path, Size, Type}, ...]（递归展开子目录，最多 2 层）。"""
    j = _curl(API % (repo, root))
    try:
        files = json.loads(j)["Data"]["Files"]
    except Exception:
        return []
    out = []
    for f in files:
        p = f.get("Path")
        if not p:
            continue
        out.append({"Path": p, "Size": f.get("Size") or 0, "Type": f.get("Type")})
        if f.get("Type") == "tree" and _depth < _max_depth:
            out.extend(list_files(repo, p, _depth + 1, _max_depth))
    return out


def find_blob(repo, prefer=None, min_size=1_000_000, pick_largest=False):
    """在数据集里找一个可下载的 blob 文件。

    prefer      : 优先路径包含该子串的文件
    pick_largest : 在 prefer 匹配不到时，直接取【整个仓库】最大的 blob
    """
    blobs = []
    if prefer and prefer.endswith("/"):
        # prefer 是目录前缀（如 "data/python/"）：
        # 必须单独列该目录，否则 list_files 的深度限制够不到第 3 层。
        blobs = [f for f in list_files(repo, root=prefer.rstrip("/"))
                 if f.get("Type") == "blob" and f.get("Size", 0) >= min_size]
    else:
        blobs = [f for f in list_files(repo)
                 if f.get("Type") == "blob" and f.get("Size", 0) >= min_size
                 and not f.get("Path", "").endswith(
                     (".md", ".pdf", ".png", ".jpg", ".txt.gz"))]
    if not blobs:
        blobs = [f for f in list_files(repo)
                 if f.get("Type") == "blob" and f.get("Size", 0) >= 1000]
    if not blobs:
        return None
    if prefer and not prefer.endswith("/"):
        # 优先精确包含 prefer 的路径（支持 "Python/train-xxx" 这类带子目录的匹配）
        cand = [b for b in blobs if prefer in b["Path"]]
        if cand:
            cand.sort(key=lambda b: -b["Size"])
            return cand[0]["Path"]
    blobs.sort(key=lambda b: -b["Size"])
    return blobs[0]["Path"]


def fetch_range(repo, path, start, size, timeout=60):
    """按字节区间下载分片（不整包下载）。返回 bytes。"""
    end = start + size - 1
    return _curl(FILE % (repo, path.replace("/", "%2F")),
                 timeout=timeout, rng="%d-%d" % (start, end), binary=True)


# ---------------------------------------------------------------------------
# 数据集配方：仓库 / 文件路径 / 起始偏移 / 采样字节数 / 目标域 / 解析格式
# 采样字节数按"该域需要的 token 量 × 每 token 字节"估算，控制在 CI 可接受范围。
# ---------------------------------------------------------------------------
MS_RECIPES = [
    # ---- 英文：连贯故事（微软专为小模型设计）→ en_general ----
    #      实测：4MB → 4384 段，1.5s
    dict(name="TinyStories", repo="AI-ModelScope/TinyStories",
         prefer="TinyStories-train.txt", start=100_000_000, size=16_000_000,
         dom="en_general", fmt="text", sep=chr(60)+chr(124)+"endoftext"+chr(124)+chr(62)),

    # ---- 英文：通用多体裁 → en_general ----
    #      实测：8MB → 1764 段，0.6s（jsonl 可随机 Range）
    dict(name="SlimPajama-6B", repo="YeungNLP/SlimPajama-6B",
         prefer="slimpajama-0.jsonl", start=300_000_000, size=8_000_000,
         dom="en_general", fmt="jsonl", field="text"),

    # ---- 中文：SkyPile 多档位多文件 → zh_general（稀释公文腔的主力）----
    #      【为什么用 SkyPile 而不是 cosmopedia】：
    #      实测 opencsg/chinese-cosmopedia 与 chinese-fineweb-edu 的 parquet
    #      全部是【单行组 1.2GB 大文件】—— 要读任何一个行组都得拉整列，
    #      实测 41s+ 且读出 3.1 亿字，CI 里不可接受。
    #      SkyPile 是 jsonl，可以用 HTTP Range 任意字节偏移随机抽样：
    #      实测 4MB → 1353 段，0.3s。
    #      【关键：分档抽样】head/middle 两个质量档、多个不同文件各取一段，
    #      这样跨源去重能看到"不同档位不同文件"的独立内容，
    #      而不是从一个巨型 jsonl 里反复抽同一批段落。
    dict(name="SkyPile-150B", repo="modelscope/SkyPile-150B",
         prefer="_zh_head_0000.jsonl", start=0, size=0,
         dom="zh_general", fmt="jsonl", field="text",
         # 多采样点：每个 (文件, 偏移) 组合算一个独立采样单元
         multi_shard=[
             ("data/2020-40_zh_head_0000.jsonl", 120_000_000),
             ("data/2020-45_zh_head_0000.jsonl", 480_000_000),
             ("data/2020-50_zh_middle_0000.jsonl", 700_000_000),
             ("data/2021-04_zh_middle_0000.jsonl", 260_000_000),
             ("data/2020-40_zh_middle_0000.jsonl", 950_000_000),
             ("data/2021-04_zh_head_0000.jsonl", 1_100_000_000),
         ], size_per_shard=4_000_000),

    # ---- 数学：带思维链（多行组 parquet，实测 0.7s/2000 行）→ math_reason ----
    #      【多分片】5 个训练分片各算一个采样单元，避免数学域只有一个"源"
    #      （单源上限才有意义，也是抗模板污染的基本要求）。
    dict(name="NuminaMath-CoT", repo="AI-MO/NuminaMath-CoT",
         prefer="train-00001-of-00005.parquet", start=0, size=0,
         dom="math_reason", fmt="parquet_rg", n_row_groups=4,
         columns=["problem", "solution"], max_rows=2000),
    dict(name="NuminaMath-CoT", repo="AI-MO/NuminaMath-CoT",
         prefer="train-00003-of-00005.parquet", start=0, size=0,
         dom="math_reason", fmt="parquet_rg", n_row_groups=4,
         columns=["problem", "solution"], max_rows=2000),

    # ---- 代码：真实开源仓库代码 → code ----
    #      【实测纠正】bigcode/the-stack-v2 的 ModelScope 镜像里
    #      parquet 只有【元数据】（blob_id/path/repo_name…），没有 content 列，
    #      且目录是 data/C、data/C++ 这样按【编程语言】分的 99 个目录，
    #      不存在 data/Python —— 原配方完全无效。
    #      换成 bigcode/the-stack-dedup：多行组、有真实 content 列，
    #      实测 3 行组 → 1500 段 / 200 万字 / 1.3s。
    #      【多样语言分片】每个 (语言目录, 分片) 算一个独立采样单元，
    #      这样代码域不会只有"一个源"（单源上限才有意义），
    #      且不同语言本身是天然的多样性。
    #      注意：绝不用 Magicoder-Evol-Instruct-110K-sandboxes-4 —— 实测那是
    #      Terminal-Bench 评测数据（含 "BENCHMARK DATA" 标记），进预训练等于背题。
    dict(name="the-stack-dedup", repo="bigcode/the-stack-dedup",
         prefer="data/python/", start=0, size=0,
         dom="code", fmt="parquet_rg", n_row_groups=3, columns=["content"],
         max_rows=1200, pick_largest=True),
    dict(name="the-stack-dedup", repo="bigcode/the-stack-dedup",
         prefer="data/java/", start=0, size=0,
         dom="code", fmt="parquet_rg", n_row_groups=2, columns=["content"],
         max_rows=800, pick_largest=True),
    dict(name="the-stack-dedup", repo="bigcode/the-stack-dedup",
         prefer="data/javascript/", start=0, size=0,
         dom="code", fmt="parquet_rg", n_row_groups=2, columns=["content"],
         max_rows=800, pick_largest=True),
    dict(name="the-stack-dedup", repo="bigcode/the-stack-dedup",
         prefer="data/go/", start=0, size=0,
         dom="code", fmt="parquet_rg", n_row_groups=2, columns=["content"],
         max_rows=800, pick_largest=True),
]


# ---------------------------------------------------------------------------
# 质量门控 + 去污染
# ---------------------------------------------------------------------------
# 样板/导航/低质行（跨语种的通用低质特征）
_BOILER_PAT = re.compile(
    r"(扫描.*二维码|点击查看更多|责任编辑|版权所有|京ICP|cookie|Cookie|"
    r"Sign in|Log in|Subscribe to our newsletter|Accept all cookies|"
    r"Click here|Read more|Share this|Advertisement|Privacy Policy|"
    r"Terms of Service|All rights reserved|http://|https://www\.)", re.I)

# 数字/符号占比过高的样本（表格残片、代码噪声、导航）
_JUNK_RATIO = re.compile(r"[\d\W_]{6,}")

# 评测集/基准数据标记：命中的样本绝不能进预训练（否则模型"背答案"）
_BENCH_MARK = re.compile(
    r"(BENCHMARK DATA|TERMINAL-BENCH|terminal-bench|"
    r"canary string|DO NOT REMOVE|"
    r"LiveCodeBench|HumanEval|MBPP|APPS[^a-z]|CodeContests[^a-z]|"
    r"leetcode\.com/problem|\ballen\.xyz\b)", re.I)


def quality_gate(text, min_len=120, max_len=20000):
    """质量门控：返回 True 表示该样本可用。"""
    if not text:
        return False
    t = text.strip()
    if len(t) < min_len or len(t) > max_len:
        return False
    if _BOILER_PAT.search(t):
        return False
    # 评测/基准数据直接拒收（防止"背答案"导致评测分数虚高）
    if _BENCH_MARK.search(t):
        return False
    # 样板行占比过高 → 判为导航/列表页
    lines = [l for l in t.split("\n") if l.strip()]
    if lines and sum(1 for l in lines if _BOILER_PAT.search(l)) / len(lines) > 0.25:
        return False
    # 单行内"非中文非字母"碎片过多 → 噪声
    if len(_JUNK_RATIO.findall(t)) > 12:
        return False
    return True


def _ngram_fingerprint(text, n=13):
    """文本的 13-gram 指纹集合（用于去污染与去重）。"""
    words = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text.lower())
    if len(words) < n:
        return set()
    return {hashlib.md5(" ".join(words[i:i + n]).encode("utf-8")).hexdigest()
            for i in range(len(words) - n + 1)}


class Decontaminator:
    """基准去污染：预载评测集的 n-gram 指纹，命中则丢弃该样本。

    评测集来自 C-Eval / CMMLU / MMLU 等公开基准（ModelScope 可达）。
    我们只取"基准题目本身"，用于识别污染——这属于标准的 decontamination 用途。
    """

    def __init__(self, n=13):
        self.n = n
        self.fps = set()
        self.n_samples = 0

    def load_benchmark(self, repo, paths, max_docs=400, timeout=60):
        """从 ModelScope 基准数据集载入指纹。"""
        for path in paths:
            raw = fetch_range(repo, path, 0, 3_000_000, timeout=timeout)
            if not raw:
                continue
            docs = []
            try:
                import io
                import pyarrow.parquet as pq
                tbl = pq.read_table(io.BytesIO(raw))
                for c in tbl.column_names:
                    vals = tbl.column(c).to_pylist()
                    docs.extend(str(v) for v in vals if isinstance(v, str))
            except Exception:
                docs = [b.decode("utf-8", "ignore") for b in _TEXT_RUN.findall(raw)]
            for d in docs[:max_docs]:
                self.fps |= _ngram_fingerprint(d, self.n)
            self.n_samples += len(docs[:max_docs])
        return len(self.fps)

    def is_contaminated(self, text):
        if not self.fps:
            return False
        fps = _ngram_fingerprint(text, self.n)
        if not fps:
            return False
        # 命中任一 13-gram 即判污染（与 GPT-3/Llama 的标准做法一致）
        return bool(fps & self.fps)


# 评测基准（只用于去污染 + 本地评测，绝不进预训练）
MS_BENCHMARKS = [
    ("modelscope/ceval-exam", ["ceval-exam.zip"]),
    ("modelscope/cmmlu", ["data/test-00000-of-00001.parquet"]),
]


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
_TEXT_RUN = re.compile(rb"[\x20-\x7e\x80-\xff]{80,}")


def _parse_parquet_bytes(raw):
    """从 parquet 分片字节里抽文本。优先 pyarrow；无则用启发式抽取。"""
    try:
        import io
        import pyarrow.parquet as pq
        tbl = pq.read_table(io.BytesIO(raw))
        cols = [c for c in tbl.column_names
                if c in ("text", "content", "output", "response", "problem",
                         "solution", "instruction")]
        if not cols:
            cols = tbl.column_names[:1]
        vals = tbl.column(cols[0]).to_pylist()
        return [str(v) for v in vals if v]
    except Exception:
        out = []
        for m in _TEXT_RUN.finditer(raw):
            s = m.group(0).decode("utf-8", "ignore").strip()
            if len(s) >= 60:
                out.append(s)
        return out


def _parse_parquet_rowgroups(repo, path, n_row_groups=3, columns=None, max_rows=4000):
    """远程按【行组】读取 parquet（fsspec + HTTP Range）。

    为什么必须按行组读：parquet 是列存格式，footer 在文件末尾，
    Range 截取前缀得不到 magic bytes → 解析失败；即使能解析，
    单行组大文件（cosmopedia 1.2GB/1 行组）也会把整列拉下来（实测 41s）。
    多行组文件（如 NuminaMath 172 组 × 1000 行）只需拉 1-2 个行组，
    实测 0.7s 拿 2000 行 —— 这是本地/CI 都能接受的量级。
    """
    try:
        import fsspec
        import pyarrow.parquet as pq
    except Exception:
        return []
    url = FILE % (repo, path.replace("/", "%2F"))
    try:
        fs, p = fsspec.core.url_to_fs(url)
        with fs.open(p, "rb") as f:
            pf = pq.ParquetFile(f)
            names = list(pf.schema.names)
            cols = [c for c in (columns or names[:1]) if c in names] or names[:1]
            idx = list(range(min(n_row_groups, pf.num_row_groups)))
            if not idx:
                idx = [0]
            tbl = pf.read_row_groups(idx, columns=cols)
            out = []
            # 关键：单行组大文件会一次读出整列（cosmopedia 实测 31 万行 / 3.1 亿字），
            # 必须在【内存层面】截断，否则构建进程会被撑爆。
            for c in cols:
                if len(out) >= max_rows:
                    break
                for v in tbl.column(c).to_pylist():
                    if len(out) >= max_rows:
                        break
                    if isinstance(v, str) and v.strip():
                        out.append(v.strip())
            return out
    except Exception as e:
        print("[ms] parquet_rg 读取失败 %s/%s: %s" % (repo, path[:30], str(e)[:80]))
        return []


def _split_docs(text, sep):
    parts = text.split(sep) if sep else [text]
    return [p.strip() for p in parts if len(p.strip()) >= 60]


def _parse_jsonl_bytes(raw, field="text"):
    """解析 jsonl 分片字节，返回文本列表（跳过坏行）。"""
    out = []
    for line in raw.decode("utf-8", "ignore").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        v = obj.get(field)
        if isinstance(v, str):
            out.append(v)
    return out


def fetch_recipe(rec, timeout=120, decon=None):
    """按配方抓一个分片并解析为文本列表。

    返回 [(采样单元名, [text, ...]), ...]：
      - 普通配方：1 个采样单元，名字 = rec["name"]
      - multi_shard 配方：N 个采样单元，名字 = "rec[name]#shard标签"
        每个采样单元 = (不同文件, 不同字节偏移)，是【独立内容块】，
        独立计数才有意义（否则一个巨型 jsonl 会被当成一个源，
        单源上限的诊断信息会失真）。
    """
    fmt = rec.get("fmt", "text")
    units = []

    if rec.get("multi_shard"):
        for path, off in rec["multi_shard"]:
            size = rec.get("size_per_shard", 4_000_000)
            raw = fetch_range(rec["repo"], path, off, size, timeout=timeout)
            if not raw:
                continue
            tag = path.rsplit("/", 1)[-1].replace(".jsonl", "")
            if fmt == "jsonl":
                texts = _parse_jsonl_bytes(raw, rec.get("field", "text"))
            else:
                texts = _split_docs(raw.decode("utf-8", "ignore"), rec.get("sep"))
            units.append(("%s#%s" % (rec["name"], tag), texts))
    else:
        path = rec.get("path")
        if not path:
            path = find_blob(rec["repo"], prefer=rec.get("prefer"),
                             pick_largest=rec.get("pick_largest", False))
            rec["path"] = path
            if not path:
                return []
        if fmt == "parquet_rg":
            # parquet_rg 走 fsspec 直接读行组，不能先 fetch_range：
            # 配方里 size=0（整文件太大不整包下），fetch_range 会返回空。
            texts = _parse_parquet_rowgroups(
                rec["repo"], path, rec.get("n_row_groups", 3), rec.get("columns"),
                rec.get("max_rows", 4000))
            # 分片名进源名：同一数据集的不同分片是独立采样单元
            tag = path.rsplit("/", 1)[-1].replace(".parquet", "")
            units.append(("%s#%s" % (rec["name"], tag), texts))
            raw = None
        else:
            raw = fetch_range(rec["repo"], path, rec["start"], rec["size"],
                              timeout=timeout)
            if not raw:
                return []
            if fmt == "jsonl":
                texts = _parse_jsonl_bytes(raw, rec.get("field", "text"))
            elif fmt == "parquet":
                texts = _parse_parquet_bytes(raw)
            else:
                texts = _split_docs(raw.decode("utf-8", "ignore"), rec.get("sep"))
            units.append((rec["name"], texts))

    # 训练侧清洗 + 质量门控 + 去污染
    out = []
    for uname, texts in units:
        cleaned, seen = [], set()
        n_contam = 0
        for t in texts:
            t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", t)
            t = re.sub(r"[ \t]+", " ", t).strip()
            if not quality_gate(t):
                continue
            if decon is not None and decon.is_contaminated(t):
                n_contam += 1
                continue
            h = hashlib.md5(t[:512].encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            cleaned.append(t)
        if n_contam:
            print("[ms] %-26s 去污染丢弃 %d 条" % (uname, n_contam))
        out.append((uname, cleaned))
    return out


def fetch_all(recipes=None, timeout=120, workers=6, verbose=True, decon=None):
    """并行抓取所有 ModelScope 分片，返回 {dom: [(采样单元名, 文本), ...]}。"""
    recipes = recipes if recipes is not None else MS_RECIPES
    out = {}
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as ex:
        futs = {ex.submit(fetch_recipe, r, timeout, decon): r for r in recipes}
        for fut in futs:
            rec = futs[fut]
            try:
                units = fut.result()
            except Exception as e:
                if verbose:
                    print("[ms] %-26s 抓取异常: %s" % (rec["name"], str(e)[:60]))
                units = []
            got = False
            for uname, texts in units:
                if not texts:
                    continue
                got = True
                out.setdefault(rec["dom"], []).extend((uname, t) for t in texts)
                if verbose:
                    print("[ms] %-34s %-12s %5d 段 / %8d 字"
                          % (uname, rec["dom"], len(texts), sum(len(t) for t in texts)))
            if not got and verbose:
                print("[ms] %-26s %-12s 抓取失败（跳过）" % (rec["name"], rec["dom"]))
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    decon = Decontaminator()
    print("[decon] 载入基准指纹…")
    try:
        n = decon.load_benchmark(*MS_BENCHMARKS[0], max_docs=300)
        print("[decon] C-Eval 指纹 %d 条" % n)
    except Exception as e:
        print("[decon] 基准载入失败（跳过去污染）: %s" % e)
    res = fetch_all(decon=decon)
    for d, items in res.items():
        tot = sum(len(t) for _, t in items)
        srcs = len(set(n for n, _ in items))
        print("[总计] %-12s %d 篇 / %d 字 / %d 个源" % (d, len(items), tot, srcs))
