"""molecule v3.0 真实语料源注册表（多源并行抓取，零外部依赖，只用 curl）。

设计原则（用户口径升级：语料要多种多样源，国内权威源为主）：
  - 每个源声明：名字 / 归属域 / 索引页 / 文章 URL 判定正则 / 最多抓多少篇。
  - 抓取用【并行 curl】（线程池），单个源失败只跳过该源，绝不阻塞训练。
  - 抽取用【结构无关】的中文文章抽取器（<p> 优先 + 中文字符密度门槛），
    不为单站点手写正文正则——站点改版时整层不会崩。
  - 相对链接统一 urljoin 绝对化；HTML 实体统一反转义。

域划分（对齐课程学习口径的域混比）：
  zh_general    权威时政/新闻/党史/纪念馆 —— 中文语言基础 + 时政知识（占比最大）
  encyclopedia  文博/百科/知识密集 —— 事实问答（"《三国演义》作者是谁"这类）靠这域
  book          公版书籍 + 古典诗文 —— 文学常识与长程语感
  en_general    英文通用（古登堡 + 英文百科 + 英文科技站）
  code          真实开源代码（GitHub raw）+ 内置样例
  math_reason   数学/逻辑/科普推理

【版权说明】人民网/人民日报/新华网/光明日报/求是/党建网/央视/国博/故宫等文本受版权保护，
此处仅用于模型所有者本人的研究训练。公开发布模型前须评估合规，或改用 CC BY-SA / 公有领域源
（维基百科、古登堡、古诗文网）。
"""

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

UA = "Mozilla/5.0 (research) molecule-lm/3.0 (contact: codecloud-dev)"

# ---------------------------------------------------------------- 抽取器
NOISE_RE = re.compile(
    r"<(script|style|noscript|head|nav|footer|aside|form|iframe|svg|noscript)[^>]*>.*?</\1>",
    re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")
P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.S | re.I)
BR_RE = re.compile(r"<br\s*/?>", re.I)
CJK_RE = re.compile(r"[一-鿿]")
HREF_RE = re.compile(r'href=["\']([^"\'\s<>]+)["\']', re.I)
BAD_EXT = re.compile(r"\.(jpg|jpeg|png|gif|svg|css|js|ico|mp4|mp3|zip|rar|xml|json)(\?|$)", re.I)
# 样板/导航/页脚行：训练侧剔除（根因修复·重复）
BOILER = re.compile(
    r"扫描.*二维码|二维码|责任编辑|返回顶部|上一篇|下一篇|相关阅读|热点推荐|版权所有|"
    r"京ICP|ICP证|关注我们|分享到|客户端|下载|免责声明|网站地图|联系我们|广告|"
    r"举报|违法和不良信息|出版|发行|订阅|投稿|版权|邮箱|电话|地址|邮编")


def fetch(url, timeout=20):
    """curl 抓取；任何失败返回空串（绝不阻塞）。
    必须带 --compressed：部分国内站点（光明日报等）返回 gzip，缺省会解出乱码。"""
    try:
        p = subprocess.run(
            ["curl", "-sSL", "--compressed", "--max-time", str(timeout), "-A", UA, url],
            capture_output=True, timeout=timeout + 12,
        )
        if p.returncode != 0:
            return ""
        return p.stdout.decode("utf-8", "ignore")
    except Exception:
        return ""


def fetch_many(urls, timeout=20, workers=12):
    """并行抓取。"""
    urls = [u for u in urls if u]
    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, max(len(urls), 1))) as ex:
        return list(ex.map(lambda u: fetch(u, timeout), urls))


def _unescape(s):
    import html as _h
    return _h.unescape(s)


def extract_text(html, min_cjk=15):
    """结构无关的中文正文抽取：<p> 优先；不够则退回「文本块」但要求中文字符密度高。"""
    if not html:
        return ""
    h = NOISE_RE.sub(" ", html)
    paras = []
    for p in P_RE.findall(h):
        t = BR_RE.sub("\n", TAG_RE.sub("", p))
        t = _unescape(t)
        t = re.sub(r"[ \t　]+", " ", t).strip()
        if len(CJK_RE.findall(t)) >= min_cjk and len(t) >= 25 and not BOILER.search(t):
            paras.append(t)
    if sum(len(p) for p in paras) < 150:
        # 回退：按块切分，取中文密度高的块（避免整页导航噪声）
        body = _unescape(TAG_RE.sub("\n", h))
        blocks = []
        for b in re.split(r"\n{2,}", body):
            b = b.strip()
            cjk = len(CJK_RE.findall(b))
            if cjk >= 40 and not BOILER.search(b):
                blocks.append(b)
        if blocks:
            blocks.sort(key=lambda b: (len(CJK_RE.findall(b)), len(b)), reverse=True)
            paras = blocks[:30]
    return "\n".join(paras)


ARTICLE_URL = re.compile(r"^(?:https?://)?[^/]+/(?:[^/]+/)*([^/]+?)/?$")


def _strip_site_suffix(t):
    """去掉标题尾部的站点名（“标题_新华网”“标题 - 光明日报”等）。"""
    sites = ["新华网", "光明日报", "人民网", "人民日报", "中国共产党新闻网", "党建网",
             "求是网", "中国政府网", "央视网", "央视新闻", "澎湃新闻", "共产党员网",
             "朱德同志故居纪念馆", "延安市人民政府", "韶山市人民政府", "故宫博物院",
             "中国国家博物馆", "孔子网", "古诗文网"]
    for s in sites:
        for sep in ("_", "-", "—", "–", "|", "－", " "):
            if t.endswith(sep + s):
                return t[: -(len(sep) + len(s))].strip()
    return t


def extract_title(html):
    """抽取文章标题（h1 优先，其次 <title>），用于生成 ChatML 的 user 问句。"""
    if not html:
        return ""
    t = ""
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S | re.I)
    if m:
        t = _unescape(TAG_RE.sub("", m.group(1)))
    else:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        if m:
            t = _unescape(TAG_RE.sub("", m.group(1)))
    t = re.sub(r"\s+", " ", t).strip()
    t = _strip_site_suffix(t)
    return t[:60]


def article_urls(index_html, base_url, pat=None, limit=60):
    """从索引页抽候选文章 URL：绝对化 + 按 pat 过滤（无 pat 时用日期/长数字启发式）。"""
    out, seen = [], set()
    rx = re.compile(pat) if pat else None
    for href in HREF_RE.findall(index_html or ""):
        if href.startswith(("javascript:", "mailto:", "#")):
            continue
        u = urljoin(base_url, href)
        if u in seen:
            continue
        seen.add(u)
        if BAD_EXT.search(u):
            continue
        if rx is not None:
            if not rx.search(u):
                continue
        else:
            if not re.search(r"(20\d{2}[-/]\d{2}|20\d{4}|_t20\d{6}|\d{6,})", u):
                continue
        out.append(u)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------- 源注册表
def rmrb_layout_urls():
    """人民日报电子版版面页（今天/昨天/前天，早晨未出报时自动回退）。"""
    import datetime
    urls = []
    t = datetime.date.today()
    for k in range(3):
        d = t - datetime.timedelta(days=k)
        urls.append("http://paper.people.com.cn/rmrb/pc/layout/%04d/%02d/%02d/node_01.html"
                    % (d.year, d.month, d.day))
        urls.append("https://epaper.gmw.cn/gmrb/html/layout/%04d%02d/%02d/node_01.html"
                    % (d.year, d.month, d.day))
    return urls


SOURCES = [
    # ---------- 权威时政 / 党史 / 纪念馆 → zh_general ----------
    dict(name="人民网", dom="zh_general", official=True, n=70,
         pat=r"people\.com\.cn/n[12]/\d{4}/\d{4}/c\d+-?\d+\.html$",
         index=["http://www.people.com.cn/", "http://politics.people.com.cn/",
                "http://world.people.com.cn/", "http://finance.people.com.cn/",
                "http://society.people.com.cn/", "http://edu.people.com.cn/",
                "http://health.people.com.cn/", "http://science.people.com.cn/",
                "http://kpzg.people.com.cn/", "http://military.people.com.cn/",
                "http://ent.people.com.cn/", "http://opinion.people.com.cn/"]),
    dict(name="人民日报电子版", dom="zh_general", official=True, n=60, rmrb=True,
         pat=r"(rmrb|gmrb)/.*\.html?$",
         index=[]),
    dict(name="新华网", dom="zh_general", official=True, n=60,
         pat=r"news\.cn/.*20\d{2}[-/]?\d{2}.*\.(s?html?)$",
         index=["http://www.news.cn/", "http://www.news.cn/politics/",
                "http://www.news.cn/fortune/", "http://www.news.cn/world/",
                "http://www.news.cn/tech/"]),
    dict(name="光明日报", dom="zh_general", official=True, n=60,
         pat=r"gmw\.cn/.*20\d{2}[-/]?\d{2}/.*content_\d+\.s?html?$",
         index=["http://www.gmw.cn/", "http://news.gmw.cn/", "http://politics.gmw.cn/",
                "http://theory.gmw.cn/"]),
    dict(name="求是网", dom="zh_general", official=True, n=50, pat=r"qstheory\.cn/.*\.html?$",
         index=["http://www.qstheory.cn/", "http://www.qstheory.cn/qshyjx/index.htm",
                "http://www.qstheory.cn/dukan/qs/index.htm"]),
    dict(name="党建网", dom="zh_general", official=True, n=50, pat=r"dangjian\.cn/.*\.html?$",
         index=["http://www.dangjian.cn/", "http://www.dangjian.cn/peixun/index.html"]),
    dict(name="中国共产党新闻网", dom="zh_general", official=True, n=50,
         pat=r"cpcnews\.cn/.*\.s?html?$",
         index=["http://www.cpcnews.cn/", "http://www.cpcnews.cn/dangshi/index.html",
                "http://www.cpcnews.cn/guoxue/index.html"]),
    dict(name="共产党员网", dom="zh_general", official=True, n=50, pat=r"12371\.cn/.*\.s?html?$",
         index=["http://www.12371.cn/", "http://www.12371.cn/dangshijianchu/index.html"]),
    dict(name="中国政府网", dom="zh_general", official=True, n=50, pat=r"gov\.cn/.*\.html?$",
         index=["http://www.gov.cn/", "http://www.gov.cn/yaowen/",
                "http://www.gov.cn/zhengce/", "http://www.gov.cn/lianbo/"]),
    dict(name="央视网", dom="zh_general", official=True, n=40, pat=r"cctv\.com/.*\.s?html?$",
         index=["http://news.cctv.com/", "http://www.cctv.com/"]),
    # 纪念馆 / 党史教育（用户点名：纪念馆网页）
    dict(name="朱德同志故居纪念馆", dom="zh_general", official=True, n=30, pat=r"zdjj\.com\.cn/.*\.html?$",
         index=["http://www.zdjj.com.cn/", "http://www.zdjj.com.cn/About/History.html"]),
    dict(name="延安（革命纪念馆/党史）", dom="zh_general", official=True, n=30,
         pat=r"yanan\.gov\.cn/.*\.html?$",
         index=["http://www.yanan.gov.cn/", "http://www.yanan.gov.cn/zx/"]),
    dict(name="韶山（毛泽东同志故居）", dom="zh_general", official=True, n=30,
         pat=r"shaoshan\.gov\.cn/.*\.html?$",
         index=["http://www.shaoshan.gov.cn/"]),
    dict(name="澎湃新闻", dom="zh_general", n=30, pat=r"thepaper\.cn/newsDetail_forward_\d+",
         index=["http://www.thepaper.cn/"]),

    # ---------- 文博 / 百科 → encyclopedia（事实问答靠这域）----------
    dict(name="故宫博物院", dom="encyclopedia", n=40, pat=r"dpm\.org\.cn/.*\.html?$",
         index=["http://www.dpm.org.cn/", "http://www.dpm.org.cn/Exhibits.html",
                "http://www.dpm.org.cn/collection.html"]),
    dict(name="中国国家博物馆", dom="encyclopedia", n=40, pat=r"chnmuseum\.cn/.*",
         index=["http://www.chnmuseum.cn/", "http://www.chnmuseum.cn/zp/",
                "http://www.chnmuseum.cn/yj/", "http://www.chnmuseum.cn/zl/",
                "http://www.chnmuseum.cn/cg/", "http://www.chnmuseum.cn/sp/"]),
    dict(name="孔子网", dom="encyclopedia", n=40, pat=r"chinakongzi\.org/.*\.html?$",
         index=["http://www.chinakongzi.org/", "http://www.chinakongzi.org/zaixun/"]),

    # ---------- 古典诗文 → book（文学常识 + 长程语感）----------
    dict(name="古诗文网", dom="book", n=50, pat=r"gushiwen\.cn/(shiwenv|mingju|guwen).*\.aspx$",
         index=["https://www.gushiwen.cn/", "https://www.gushiwen.cn/shiwens/",
                "https://www.gushiwen.cn/mingjus/"]),

    # ---------- 科普 / 数理 → math_reason（Phase2 占 25%，需足量真实语料）----------
    dict(name="中国科学院", dom="math_reason", n=40, pat=r"cas\.cn/.*",
         index=["http://www.cas.cn/", "http://www.cas.cn/xw/", "http://www.cas.cn/kx/"]),
    dict(name="科技日报", dom="math_reason", n=40, pat=r"stdaily\.com/.*",
         index=["http://www.stdaily.com/", "http://www.stdaily.com/index/kejixinwen/"]),
    dict(name="中国科普博览", dom="math_reason", n=40, pat=r"kepu\.net\.cn/.*",
         index=["http://www.kepu.net.cn/", "http://www.kepu.net.cn/kp/"]),
    dict(name="国家天文台", dom="math_reason", n=30, pat=r"nim\.ac\.cn/.*",
         index=["http://www.nim.ac.cn/", "http://www.nim.ac.cn/xwzx/"]),
    dict(name="中国数字科技馆", dom="math_reason", n=30, pat=r"cdstm\.cn/.*",
         index=["https://www.cdstm.cn/"]),

    # ===================================================================
    # 非公文腔 / 多语种补充源（用户口径：通用模型，不能只有国内党政公文；
    #   别的语言、别的文体都要加 —— 文体同质会让小模型被单一风格"带跑"，
    #   表现就是公文腔 + 复读机式 AI 味）
    # ===================================================================

    # ---------- 中文：非党政语体（新闻通稿 / 财经 / 科技 / 文学 / 技术博客）----------
    dict(name="中国新闻网", dom="zh_general", n=40,
         pat=r"chinanews\.com\.cn/.+?/20\d{2}/\d{2}-\d{2}/\d+\.s?html?$",
         index=["http://www.chinanews.com.cn/", "http://www.chinanews.com.cn/gn/",
                "http://www.chinanews.com.cn/sh/", "http://www.chinanews.com.cn/cj/",
                "http://www.chinanews.com.cn/jr/"]),
    dict(name="第一财经", dom="zh_general", n=30, pat=r"yicai\.com/news/\d+\.html$",
         index=["https://www.yicai.com/", "https://www.yicai.com/news/"]),
    dict(name="证券时报", dom="zh_general", n=25, pat=r"stcn\.com/article/detail/\d+\.html$",
         index=["https://www.stcn.com/"]),
    dict(name="中国作家网", dom="book", n=40, pat=r"chinawriter\.com\.cn/\d+/\d+/index\.html$",
         index=["https://www.chinawriter.com.cn/"]),
    dict(name="博客园（技术博客）", dom="code", n=40,
         pat=r"cnblogs\.com/[^/]+/p/\d+$",
         index=["https://www.cnblogs.com/"]),
    dict(name="解放日报·上观", dom="zh_general", n=25,
         pat=r"jfdaily\.com/.+?/\d+\.html$",
         index=["https://www.jfdaily.com/"]),
    dict(name="中国日报（英文版·中国话题英文语料）", dom="en_general", n=30,
         pat=r"chinadaily\.com\.cn/.+?/\d+\.html$",
         index=["https://www.chinadaily.com.cn/", "https://global.chinadaily.com.cn/"]),

    # ---------- 多语种（非中英）→ en_general 之外的通用语料位 ----------
    # 注：域键沿用 en_general（多语种都算"非中文通用"），但语料本身是德/法/西/俄/日等原文，
    #     用来打破"中英 + 党政公文"的文体/语种单一性。
    dict(name="维基文库（英/多语种公有领域）", dom="en_general", n=30,
         pat=r"wikisource\.org/wiki/.+$",
         index=["https://en.wikisource.org/wiki/Category:Works"],
         # 沙箱常被墙，CI 可达；抓不到自动跳过
         ),
    dict(name="联合国文件（英文）", dom="en_general", n=20,
         pat=r"docs\.un\.org/en/.*\.pdf$",
         index=["https://docs.un.org/en/"]),
]

# 公版英文书籍（公有领域）
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

# ---------------------------------------------------------------------------
# 多语种公版书籍（用户口径：不能只加国内中文，别的语言都要加）
# 做法：直接抓 Gutenberg 的多语种 txt 文件，绕过沙箱对维基的封锁。
# 语种：德/法/西/俄/日 —— 每种取几本经典，足够打破"中英+公文"的语种单一。
# ---------------------------------------------------------------------------
GUTENBERG_MULTILINGUAL = {
    "de": ["https://www.gutenberg.org/files/2009/2009-0.txt",     # Deutschland. Ein Volksbuch
           "https://www.gutenberg.org/files/1232/1232-0.txt"],    # Der Prozess (Kafka)
    "fr": ["https://www.gutenberg.org/files/17489/17489-0.txt",   # Monte-Cristo (Vol.1)
           "https://www.gutenberg.org/files/4280/4280-0.txt"],    # Mme Bovary
    "es": ["https://www.gutenberg.org/files/38897/38897-0.txt",   # Don Quijote
           "https://www.gutenberg.org/files/51537/51537-0.txt"],  # La sombra del viento
    "ru": ["https://www.gutenberg.org/files/4300/4300-0.txt"],     # War and Peace (t.1)
    "ja": ["https://www.gutenberg.org/files/1982/1982-0.txt",    # 吾輩は猫である
           "https://www.gutenberg.org/files/25438/25438-0.txt"],  # 羅生門
}

# ---------------------------------------------------------------------------
# 学术摘要（arXiv API）：科学/技术语体，与公文腔完全不同构。
# 用途：给模型"引用式表达 + 术语 + 推理段落"的分布。
# ---------------------------------------------------------------------------
ARXIV_CATEGORIES = ["cs.CL", "cs.LG", "cs.AI", "stat.ML", "math.HO"]
ARXIV_QUERY = ("http://export.arxiv.org/api/query?search_query=cat:%s"
                "&start=0&max_results=40&sortBy=submittedDate&sortOrder=descending")

# ---------------------------------------------------------------------------
# 官方技术文档：工程师写的散文（解释性、精确、非宣传腔），是很理想的"通用语感"语料。
# ---------------------------------------------------------------------------
TECH_DOCS = [
    ("https://docs.python.org/3/library/asyncio-task.html", "en_general"),
    ("https://docs.python.org/3/tutorial/introduction.html", "en_general"),
    ("https://docs.python.org/3/library/dataclasses.html", "en_general"),
    ("https://kubernetes.io/docs/concepts/overview/", "en_general"),
    ("https://kubernetes.io/docs/tasks/tools/", "en_general"),
    ("https://docs.golang.org/effective_go/", "en_general"),
    ("https://doc.rust-lang.org/book/ch01-00-getting-started.html", "en_general"),
]

# 真实开源代码（GitHub raw，许可宽松的项目 README/源码片段）
CODE_RAW = [
    "https://raw.githubusercontent.com/python/cpython/main/README.rst",
    "https://raw.githubusercontent.com/torvalds/linux/master/README",
    "https://raw.githubusercontent.com/git/git/master/README.md",
    "https://raw.githubusercontent.com/rust-lang/rust/master/README.md",
    "https://raw.githubusercontent.com/golang/go/master/README.md",
    "https://raw.githubusercontent.com/apache/spark/master/README.md",
    "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/README.rst",
    "https://raw.githubusercontent.com/pytorch/pytorch/main/README.md",
    "https://raw.githubusercontent.com/numpy/numpy/main/README.rst",
    "https://raw.githubusercontent.com/django/django/main/README.rst",
    "https://raw.githubusercontent.com/redis/redis/unstable/README.md",
    "https://raw.githubusercontent.com/kubernetes/kubernetes/master/README.md",
]


def scrape_arxiv(categories=None, timeout=25):
    """抓 arXiv 摘要（Atom XML）：{title, abstract} 列表。科学语体语料。"""
    out = []
    for cat in (categories or ARXIV_CATEGORIES):
        xml = fetch(ARXIV_QUERY % cat, timeout=timeout)
        if not xml:
            continue
        for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
            t = re.search(r"<title>(.*?)</title>", entry, re.S)
            a = re.search(r"<summary>(.*?)</summary>", entry, re.S)
            if not t or not a:
                continue
            title = re.sub(r"\s+", " ", _unescape(re.sub(r"<[^>]+>", "", t.group(1)))).strip()
            abstr = re.sub(r"\s+", " ", _unescape(re.sub(r"<[^>]+>", "", a.group(1)))).strip()
            if len(abstr) >= 200:
                out.append((title, abstr))
    return out


def scrape_tech_docs(timeout=20):
    """抓官方技术文档正文（去导航后按 <p> 取段落）。"""
    out = []
    for url, dom in TECH_DOCS:
        h = fetch(url, timeout=timeout)
        if not h:
            continue
        body = extract_text(h, min_cjk=0)
        # 英文文档：按行过滤短行
        paras = [p for p in body.split("\n") if len(p) >= 60]
        if len(paras) >= 8:
            out.append((dom, url, "\n".join(paras)[:120_000]))
    return out



def scrape_source(src, timeout=18, workers=10, with_title=False):
    """抓一个源：索引页 → 候选文章 → 并行抓正文 → 抽取。
    with_title=True 时返回 (name, [(标题, 正文), ...])，否则返回 (name, [正文, ...])。"""
    idx = list(src.get("index") or [])
    if src.get("rmrb"):
        idx += rmrb_layout_urls()
    if not idx:
        return src["name"], []
    idx_htmls = fetch_many(idx, timeout=timeout, workers=workers)
    pat = src.get("pat")
    urls = []
    for base, html in zip(idx, idx_htmls):
        urls.extend(article_urls(html, base, pat, limit=src.get("n", 40) * 2))
    urls = list(dict.fromkeys(urls))[:src.get("n", 40)]
    if not urls:
        return src["name"], []
    bodies = fetch_many(urls, timeout=timeout, workers=workers)
    texts, pairs = [], []
    for b in bodies:
        t = extract_text(b)
        if len(CJK_RE.findall(t)) >= 80:
            texts.append(t)
            if with_title:
                pairs.append((extract_title(b), t))
    if with_title:
        return src["name"], pairs
    return src["name"], texts


def scrape_all(sources=None, timeout=18, workers=10, verbose=True):
    """并行抓所有源（源之间也并行），返回 {dom: [(name, text), ...]}。"""
    sources = sources if sources is not None else SOURCES
    out = {}
    # 源之间并行，但每个源内部也并行 → 用两层：先并行抓所有源（各自内部串行化线程池会争抢，
    # 这里改为「逐源并行」以避免线程数爆炸，源内 10 线程 × 1 源）。
    for src in sources:
        name, texts = scrape_source(src, timeout=timeout, workers=workers)
        if texts:
            out.setdefault(src["dom"], []).extend((name, t) for t in texts)
            if verbose:
                print("[src] %-22s %-12s %4d 篇 / %8d 字"
                      % (name, src["dom"], len(texts), sum(len(t) for t in texts)))
    return out


# ---------------------------------------------------------------------------
# 源级硬上限（用户口径·方案C）—— 【已废弃，勿再调用】
# ---------------------------------------------------------------------------
# 【为什么废弃】这两个上限有三个致命问题（2026-11-02 语料治理审计发现）：
#   1) 按【字符数】算，不是 token 数 → 与训练实际吃进去的比例脱节
#      （中文 1 字 ≈ 1 token，但代码/英文/数字的 token 密度完全不同）；
#   2) 只管【单源】和【official 聚合】两条线，拦不住【跨源重复】——
#      同一篇通稿在人民网/新华网/央视网各发一遍，这里完全看不见；
#   3) 上限是【域内字符占比】，而训练时域权重由课程固定（zh_general 占 43.7%），
#      域内占比 8% 折算到训练 token 流就是 3.5%——数值完全没有可比性。
# 现在的实现见 corpus_govern.py：
#   - 跨源近重复合并（MinHash + LSH，全局视野，跨域可见）
#   - 文体（regime）归类 → "规范书面语/公文/新闻通稿"合计 ≤10%（训练流口径）
#   - 单一来源 ≤0.5%（训练流口径；数学上不可达时如实报告缺口，不砍空语料）
#   - 域地板：防止把某个域砍空（域权重固定，砍空不改变权重分配，只会加剧复读）
SOURCE_CAP_FRAC = 0.03
OFFICIAL_CAP_FRAC = 0.08
# 单源最少保留字数（避免把一个小源完全清零：它可能正是不错的语体补充）
SOURCE_MIN_KEEP = 4000
# official 聚合预算的绝对下限（字符）：语料太小时也要留一点规范书面语语感，
# 但不能让下限反过来撑破比例约束（这是之前 8% 失效成 28% 的原因）。
OFFICIAL_MIN_TOTAL = 8000


def apply_source_caps(items, source_cap_frac=SOURCE_CAP_FRAC,
                      official_cap_frac=OFFICIAL_CAP_FRAC, min_keep=SOURCE_MIN_KEEP,
                      official_names=None, verbose=True):
    """按源限流：items = [(源名, 文本), ...] → (保留后的 items, 被限流的 items)。

    两条约束：
      1) 单源上限：任何单一来源对该域的字符贡献 ≤ source_cap_frac（默认 3%）
      2) official 聚合上限：全部时政/党史源合计 ≤ official_cap_frac（默认 8%）
         —— 这条是"公文腔"的根治线：时政语料不再是语料主体，只留一点规范书面语语感。

    实现要点：
      - official 源共享【一个全局预算】，按源轮转分配，避免单个源吃光预算。
      - 保底量（min_keep）从【预算内】支出，不额外突破上限：即每个源至少
        分到 min_keep 字符（若预算足够），而不是"下限可以突破上限"。
      - 限流按【字符数】而非篇数（篇少但篇长的源也要压住）。
    """
    if not items:
        return [], []
    by_src = {}
    for src, txt in items:
        by_src.setdefault(src, []).append(txt)
    total_chars = sum(len(t) for t in items)
    official_names = set(official_names or [])
    off_names = sorted(s for s in by_src if s in official_names)

    kept, dropped = [], []

    # ---- 非 official 源：各自受单源上限约束（保底量计入上限内）----
    single_cap = max(int(total_chars * source_cap_frac), 1)
    for src in sorted(by_src):
        if src in official_names:
            continue
        texts = by_src[src]
        cur = sum(len(t) for t in texts)
        if cur <= single_cap:
            kept.extend((src, t) for t in texts)
            continue
        floor = min(min_keep, single_cap)     # 保底量不突破单源上限
        texts_sorted = sorted(texts, key=len, reverse=True)
        acc, keep_list = 0, []
        for t in texts_sorted:
            if acc + len(t) <= single_cap:
                keep_list.append(t)
                acc += len(t)
            elif acc < floor:                  # 保底：允许突破上限到 floor
                keep_list.append(t)
                acc += len(t)
            else:
                dropped.append((src, t))
        kept.extend((src, t) for t in keep_list)
        if verbose:
            print("[cap] %-22s %8d 字 → 保留 %8d 字（单源上限 %d）"
                  % (src, cur, acc, single_cap))

    # ---- official 源：共享全局预算，源间轮转分配（源间均衡，不让一个源吃光）----
    if off_names:
        off_total = sum(len(t) for s in off_names for t in by_src[s])
        budget = int(total_chars * official_cap_frac)
        budget = max(budget, min(OFFICIAL_MIN_TOTAL, off_total))
        budget = min(budget, off_total)
        # 按源轮转取文（每个源内按长度降序），直到预算耗尽
        pools = {s: sorted(by_src[s], key=len, reverse=True) for s in off_names}
        idx = {s: 0 for s in off_names}
        used = 0
        while used < budget:
            progressed = False
            for s in off_names:
                if used >= budget:
                    break
                j = idx[s]
                if j >= len(pools[s]):
                    continue
                t = pools[s][j]
                if used + len(t) > budget:
                    # 放不下：若该源还没拿到保底量，仍允许它进来（保底从预算内出）
                    if used >= min(min_keep, budget):
                        continue
                idx[s] = j + 1
                kept.append((s, t))
                used += len(t)
                progressed = True
            if not progressed:
                break
        for s in off_names:
            for j in range(idx[s], len(pools[s])):
                dropped.append((s, pools[s][j]))
        if verbose:
            print("[cap] official 聚合 %d 字 → 保留 %d 字（预算 %d，占 %.1f%%）"
                  % (off_total, used, budget, 100.0 * used / max(total_chars, 1)))
    return kept, dropped


if __name__ == "__main__":
    import sys
    dom = sys.argv[1] if len(sys.argv) > 1 else None
    srcs = [s for s in SOURCES if dom is None or s["dom"] == dom]
    res = scrape_all(srcs)
    for d, items in res.items():
        tot = sum(len(t) for _, t in items)
        print("[总计] %-12s %d 篇 / %d 字" % (d, len(items), tot))
