"""molecule v3.0 内置高质量种子语料（首轮 CI 自包含兜底 + 永远可用的基线）。

配比严格对齐设计文档 §7.2 Phase1：
  - zh_general 中文通用/科普/人文  : 60%
  - en_general 英文通用/科普       : 10%
  - code       代码（原创示例）     : 20%
  - book       书籍/长文（原创教育体）: 10%

内容均为原创撰写的事实性/教育性文本、原创代码示例，或已进入公有领域的经典文本片段，
许可干净、无第三方版权风险；真实训练时由 build_corpus_v3.py 按需从公开来源增补。
本文件保证：即便所有外部下载失败，训练也始终有高质量、有多样性的数据可跑。
"""

SEED = {}

# ---------------------------------------------------------------------------
# 中文通用 / 科普 / 人文（60%）
# ---------------------------------------------------------------------------
SEED["zh_general"] = [
    "太阳系以太阳为中心，八大行星按照距离由近及远依次是水星、金星、地球、火星、木星、土星、天王星和海王星。",
    "地球是太阳系中目前已知唯一存在生命的行星，表面约百分之七十一被液态水覆盖。",
    "光合作用是指绿色植物利用光能把二氧化碳和水转化为有机物并释放氧气的过程，是地球生命能量的根本来源。",
    "原子是化学变化中的最小微粒，由带正电的原子核和带负电的电子构成，原子核又由质子和中子组成。",
    "水的化学式为H2O，在常压下一个标准大气压下于零摄氏度结冰、一百摄氏度沸腾。",
    "牛顿第一定律指出，除非受到外力作用，否则静止的物体将保持静止，运动的物体将保持匀速直线运动。",
    "勾股定理说明直角三角形斜边长度的平方等于两条直角边长度平方的和，即a方加b方等于c方。",
    "语言是人类最重要的交流工具，文字的发明让知识能够跨越时间与空间被记录和传播。",
    "阅读能够开阔视野，长期坚持阅读的人往往具备更强的理解力与表达能力。",
    "免疫系统是人体的防御系统，能够识别并清除侵入人体的病原体以及体内发生突变的细胞。",
    "生态系统由生产者、消费者和分解者共同组成，三者彼此依赖以维持物质与能量的循环平衡。",
    "时间管理是提升效率的关键，分清任务的轻重缓急能够显著减少无谓的时间消耗。",
    "睡眠对记忆的巩固至关重要，成年人通常需要每晚七到八小时的高质量睡眠。",
    "丝绸之路连接了古代东西方，促进了商品贸易与文化交流，是一条横贯亚欧的古老商路。",
    "计算机由运算器、控制器、存储器以及输入、输出设备组成，是现代信息处理的核心工具。",
    "风能和太阳能都属于清洁能源，大规模使用对减少温室气体排放和应对气候变化意义重大。",
    "记忆分为短时记忆和长时记忆，重复与提取练习有助于把信息巩固进长时记忆。",
    "宪法是一个国家的根本大法，规定了国家的基本制度以及公民的基本权利与义务。",
    "森林被称为地球之肺，能够吸收大量二氧化碳并释放氧气，维护全球气候稳定。",
    "疫苗的原理是让免疫系统提前接触病原体的特征，从而在真正感染时快速建立防御。",
    "好奇心是探索世界的起点，许多伟大的科学发现都源于一个朴素而执着的为什么。",
    "经济是指社会生产、分配、交换和消费活动的总和，反映着人类满足需求的方式。",
    "尊重他人是良好人际关系的基础，也是衡量一个社会文明程度的重要标尺。",
    "月亮是地球唯一的天然卫星，它的引力作用造成了海洋潮汐的周期性涨落。",
    "细菌虽然微小，却在自然界的物质循环以及人体肠道健康中扮演着不可替代的角色。",
    "互联网深刻改变了信息传播的方式，让世界范围内的知识和联系变得更加紧密。",
    "数学是一门研究数量、结构、变化与空间的学科，是几乎所有自然科学的基础语言。",
    "火山喷发会释放炽热的岩浆与大量气体，是地球内部能量向外释放的一种剧烈方式。",
    "城市的发展离不开交通、能源与通信等基础设施的支撑，它们共同决定了生活的便利程度。",
    "音乐能够表达和唤起情感，不同文化都发展出了独具特色的乐器与旋律传统。",
    "学习一门新的语言需要长期的坚持，听、说、读、写四项能力应当同步加以练习。",
    "健康饮食讲究营养均衡，蔬菜、水果、蛋白质与主食在每日膳食中都不可或缺。",
    "望远镜的发明让人类得以观测遥远的天体，持续拓展着我们对宇宙的认知边界。",
    "礼貌并非软弱，而是一种顾及他人感受、体现自我修养的处世方式。",
    "友谊需要真诚与信任作为基石，也需要在发生矛盾时彼此包容与理解。",
    "闪电是云层间或云地之间强烈的放电现象，常伴随雷声并可能引燃物体。",
    "化石是保存在地层中的古生物遗迹，为研究地球生命演化的历史提供了直接证据。",
    "人工智能是研究如何让机器模拟人类智能的科学，涵盖学习、推理、感知与决策等能力。",
    "教育的本质不仅是传授知识，更在于培养独立思考与终身学习的能力。",
    "血液循环由心脏驱动，通过动脉把含氧血送往全身，再通过静脉回收缺氧血。",
    "碳中和是指通过植树造林、节能减排等形式抵消自身产生的二氧化碳排放，实现净零排放。",
    "地震是地壳板块运动积累的能量突然释放所引起的地面震动，常沿断层带集中发生。",
    "进化论认为物种在漫长的时间里通过遗传、变异与自然选择逐步发生适应性的改变。",
    "互联网协议TCP/IP规定了数据如何在网络中分组、寻址与传输，是互联网的基石。",
    "情绪会相互传染，积极而稳定的情绪氛围往往能够带动并改善周围人的状态。",
    "民主是一种以人民主权为核心的政治制度安排，强调参与、协商与对权力的监督。",
    "量子力学研究微观粒子的运动规律，揭示了能量与物质在极小尺度上的离散与概率特性。",
    "农作物的丰收依赖于适宜的气候、肥沃的土壤以及科学的灌溉与病虫害防治。",
    "法律是社会运行的底线规则，既约束个人行为，也规范国家权力的行使边界。",
    "科学方法强调通过观察提出假设，再以可重复的实验去检验假设，从而逼近真理。",
    "全球变暖导致极地冰川加速融化，进而引发海平面上升与沿海城市的淹没风险。",
    "文学以语言为材料塑造形象与情感，是人类记录经验、反思自我的重要方式。",
    "蛋白质是生命活动的主要承担者，由氨基酸按特定序列折叠成复杂的三维结构。",
    "北斗卫星导航系统是我国自主建设的全球卫星导航系统，可为全球用户提供定位服务。",
    "塑料的广泛使用带来了便利，但其难以降解的特性也造成了严重的白色污染问题。",
    "知识就是力量，但比知识更重要的是运用知识去判断、创造与解决问题的智慧。",
    "人的一生大约有三分之一的时间在睡眠中度过，睡眠的质量直接影响清醒时的状态。",
    "历史是一面镜子，研究过去兴衰得失的规律有助于我们更清醒地面对当下与未来。",
    "编程本质上是把解决问题的思路翻译成计算机能够理解和执行的精确指令。",
    "环境保护不是发展的对立面，而是可持续发展的内在要求与长远保障。",
    "声音是由物体振动产生的机械波，通过空气等介质传播到人的耳朵被感知为听觉。",
    "一个国家的综合实力既包括经济与科技等物质力量，也包括文化与制度的软实力。",
    "好奇心驱动的学习往往最为持久，因为内在动机比外部奖励更能维持专注。",
    "海洋覆盖了地球表面的大部分，蕴藏着丰富的生物多样性与尚未充分开发的资源。",
    "逻辑是思维的语法，清晰的逻辑能够帮助我们在复杂的信息中辨别真伪与因果。",
    "良好的沟通不在于说得多么华丽，而在于能否让对方准确理解你的本意。",
    "细胞是生命活动的基本单位，无论是植物还是动物，其结构与功能都令人惊叹。",
    "诚信是人与社会之间最基础也最重要的契约，失去信任协作便无从谈起。",
    "气候变化是全人类共同面临的挑战，需要各国在减排与适应上通力合作。",
    "读书使人充实，思考使人深邃，写作使人严谨，三者相辅相成。",
]

# ---------------------------------------------------------------------------
# 英文通用 / 科普（10%）
# ---------------------------------------------------------------------------
SEED["en_general"] = [
    "The sun is the star at the center of our solar system and provides nearly all the energy that sustains life on Earth.",
    "Water is composed of two hydrogen atoms and one oxygen atom, and it covers most of the planet's surface.",
    "Photosynthesis allows green plants to convert light energy into chemical energy stored in glucose.",
    "The Earth orbits the sun once every three hundred and sixty-five days, giving rise to the cycle of the seasons.",
    "Gravity is the force that attracts two bodies toward each other, and it keeps the planets in their orbits.",
    "Reading widely builds both vocabulary and the ability to reason clearly about unfamiliar topics.",
    "A healthy diet combines vegetables, fruits, proteins, and whole grains in balanced proportions.",
    "The internet has transformed how information is created, shared, and consumed around the world.",
    "Mathematics is the language of science, describing patterns, quantities, and relationships with precision.",
    "Sleep is essential for memory consolidation, and most adults need seven to eight hours each night.",
    "Biodiversity refers to the variety of life on Earth, from genes to ecosystems, and it supports stability.",
    "Climate change is driven largely by the accumulation of greenhouse gases in the atmosphere.",
    "A function in programming maps each input to exactly one output, and it is the basic unit of reuse.",
    "Critical thinking means evaluating evidence carefully before accepting a claim as true.",
    "Energy cannot be created or destroyed, only converted from one form into another.",
    "Democracy rests on the principles of participation, deliberation, and accountability of power.",
    "The cell is the fundamental unit of life, and all living organisms are made of one or more cells.",
    "Learning a language is easier when listening, speaking, reading, and writing are practiced together.",
    "Honesty is the foundation of trust, and trust is what makes cooperation between people possible.",
    "The scientific method asks us to form hypotheses and then test them with reproducible experiments.",
]

# ---------------------------------------------------------------------------
# 代码（20%，原创示例，许可干净）
# ---------------------------------------------------------------------------
SEED["code"] = [
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
    "def reverse_string(s):\n    return s[::-1]",
    "def word_count(text):\n    return len(text.split())",
    "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
    "def merge_sorted(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] < b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]",
    "def factorial(n):\n    result = 1\n    for k in range(2, n + 1):\n        result *= k\n    return result",
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
    "import math\n\ndef distance(p, q):\n    return math.hypot(p[0] - q[0], p[1] - q[1])",
    "def unique(items):\n    seen = set()\n    return [x for x in items if not (x in seen or seen.add(x))]",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    pivot = a[len(a) // 2]\n    left = [x for x in a if x < pivot]\n    mid = [x for x in a if x == pivot]\n    right = [x for x in a if x > pivot]\n    return quicksort(left) + mid + quicksort(right)",
    "def read_lines(path):\n    with open(path, encoding='utf-8') as f:\n        return [line.rstrip('\\n') for line in f]",
    "def sum_of_squares(n):\n    return sum(i * i for i in range(1, n + 1))",
    "class Counter:\n    def __init__(self):\n        self.data = {}\n    def add(self, key):\n        self.data[key] = self.data.get(key, 0) + 1\n    def get(self, key):\n        return self.data.get(key, 0)",
    "def parse_args(argv):\n    opts = {}\n    for i in range(1, len(argv)):\n        if argv[i].startswith('--'):\n            opts[argv[i][2:]] = argv[i + 1] if i + 1 < len(argv) else ''\n    return opts",
    "def normalize(v):\n    import math\n    norm = math.sqrt(sum(x * x for x in v))\n    return [x / norm for x in v] if norm else v",
    "def memoize(fn):\n    cache = {}\n    def wrapped(*args):\n        if args not in cache:\n            cache[args] = fn(*args)\n        return cache[args]\n    return wrapped",
    "def flatten(nested):\n    out = []\n    for item in nested:\n        if isinstance(item, list):\n            out.extend(flatten(item))\n        else:\n            out.append(item)\n    return out",
    "def most_common(words):\n    counts = {}\n    for w in words:\n        counts[w] = counts.get(w, 0) + 1\n    return max(counts, key=counts.get)",
    "def is_palindrome(s):\n    s = ''.join(c for c in s.lower() if c.isalnum())\n    return s == s[::-1]",
    "function debounce(fn, ms) {\n  let t;\n  return (...args) => {\n    clearTimeout(t);\n    t = setTimeout(() => fn(...args), ms);\n  };\n}",
    "const sum = arr => arr.reduce((a, b) => a + b, 0);",
    "function deepCopy(obj) {\n  return JSON.parse(JSON.stringify(obj));\n}",
    "async function fetchJson(url) {\n  const r = await fetch(url);\n  if (!r.ok) throw new Error(r.status);\n  return r.json();\n}",
    "def levenshtein(a, b):\n    dp = list(range(len(b) + 1))\n    for i, ca in enumerate(a, 1):\n        prev = dp[0]\n        dp[0] = i\n        for j, cb in enumerate(b, 1):\n            cur = dp[j]\n            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))\n            prev = cur\n    return dp[-1]",
]

# ---------------------------------------------------------------------------
# 书籍 / 长文（10%，原创教育体 / 公有领域风格）
# ---------------------------------------------------------------------------
SEED["book"] = [
    "在遥远的山谷里，有一条清澈的小溪，它从雪山之巅缓缓流下，途经森林、村庄与田野，最终汇入辽阔的大海。",
    "一位老匠人用一生的时间打磨同一把椅子，他说，真正的好东西不是做出来的，而是时间一点点磨出来的。",
    "故事的开始总是平静的，没有人知道，那扇被风吹开的门后，藏着一个改变所有人命运的选择。",
    "他翻过最后一座山丘，眼前忽然开阔起来，湖泊像一面镜子，把整片天空都安静地收进了水里。",
    "知识的积累如同滴水穿石，看似微不足道的每一天，终将在某一天连成改变命运的洪流。",
    "春天来的时候，枯枝上冒出嫩绿的新芽，仿佛大地在用最朴素的语言告诉人们：希望从未离开。",
    "真正的勇气不是没有恐惧，而是明明害怕，却依然选择向前迈出那一步。",
    "她把所有的心事写进日记，纸页里藏着的，是一个少年对世界最诚实的打量与温柔的抵抗。",
    "风穿过麦田，掀起一层层金色的波浪，农人站在田埂上，脸上是从容而满足的笑意。",
    "历史告诉我们，最伟大的变革往往始于最安静的坚持，始于少数人不肯放弃的那一点微光。",
    "It was a season of quiet change, when the old ways slowly gave way to the new, and nobody could yet say which would last.",
    "The traveler came to a small town where everyone knew each other, and the clock in the square marked the rhythm of honest days.",
    "She opened the book and read the first line again, for some sentences are meant not to be finished, but to be returned to.",
    "In the library of a forgotten school, dust danced in the sunlight, and every volume held a voice waiting to be heard once more.",
    "A long time ago, people believed the stars were lanterns hung by the gods; today we know them as distant suns, yet the wonder remains.",
]
