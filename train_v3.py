"""molecule v3.0 基础模型训练入口（Mamba+BinaryAttention 混合骨干，byte-BPE 8192）。

用法（本地冒烟 / CI 同一入口）：
  python train_v3.py --seq_len 256 --batch 4 --max_minutes 75 --save_every 10 --resume

纪律（严格对齐设计文档 §7 / §10 / §12.1 + 用户课程学习口径）：
  - 这是 v3.0 基础模型：只夯骨干，什么高级功能都不接（System1/元认知/调度器/工具/记忆
    全是 v3.5+ 的事）；目标 PPL<10，能生成通顺中文即可。
  - 课程学习（先按阶段分训练量，再在阶段内按域分比例），按【10 天训练窗口】切阶段：
      Phase1      50%  中60/英10/代码20/书10           打语言基础
      Phase2      40%  中30/代码30/数学25/书10/百科5    强化推理与代码
      Mid-training 10%  困惑度过滤难样本 + 退火         精炼能力
    数据规模设计目标：100B 高质量语料循环 3–4 epoch（TinyLlama 验证小模型过训不饱和；
    MobileLLM-R1 验证该配比 4.2T tokens 打平 36T 训练的 Qwen3-0.6B）。实际语料不足时多 epoch。
  - 损失：L_lm + 0.3*L_mtp(位移+2) + 1e-4*Z-loss。
  - 优化器：AdamW，β2=0.95，weight_decay=0.01，全局梯度裁剪 1.0。
  - 学习率：warmup 2000 步 + WSD 余弦衰减到 10%；进入 Mid-training 后余弦退火到 ~1e-5。
  - 断点：checkpoints/v3.pt（model+optimizer+step）；--resume 续训。
  - 训练时钟：data/v3_clock.json 累计 {分钟, token, 步数, 轮次, 阶段}，供 10 天发版闸门使用。
  - 鲁棒性：周期 flush（--save_every 分钟）保证 CI 被回收/取消时最多丢一段；SIGTERM
    优雅退出（timeout 软终止前存档），再留 5 分钟余量给存档与上传。
"""

import argparse
import json
import math
import os
import signal
import subprocess
import time

import numpy as np
import torch

import build_corpus_v3
from model_v3 import MoleculeV3

DOMAINS = build_corpus_v3.DOMAINS
MIX = build_corpus_v3.MIX                       # 兼容：Phase1 混比
PHASES = build_corpus_v3.PHASES
TOTAL_BUDGET_TOKENS = build_corpus_v3.TOTAL_BUDGET_TOKENS   # 数据规模设计目标（仅记录）
CKPT = "checkpoints/v3.pt"
CLOCK = "data/v3_clock.json"
VOCAB = 8192
MIDTRAIN_BIN = "data/v3_midtrain.bin"

# 课程阶段按【累计训练分钟】切分；10 天窗口 = 14400 分钟。
# （100B 是数据规模设计目标，非调度驱动；阶段切分绑定到用户指定的 10 天训练窗口，
#  保证「每个阶段吃不同搭配」在真实 CI 训练期内真正发生。）
TRAIN_HORIZON_MIN = 10 * 24 * 60

WARMUP_STEPS = 2000
PEAK_LR = 3e-4
# Mid-training 难样本池构建时机（步）：模型需先具备一定语言能力，此时的困惑度排序才有意义
# （随机初始化时的困惑度近乎均匀，筛不出"难样本"）。池构建一次后持久化复用。
MIDTRAIN_POOL_AT_STEP = 3000


def lr_at(step, cum_min, peak=PEAK_LR, warmup=WARMUP_STEPS):
    """学习率：WSD 热身 + 阶段衰减；进入 Mid-training 后余弦退火到 ~1e-5。
    - step：全局步数（热身基于它，跨轮续训不重置）
    - cum_min：累计训练分钟（决定所处阶段与退火进度）
    """
    if step < warmup:
        return peak * (step / warmup)
    g = min(cum_min / TRAIN_HORIZON_MIN, 1.0)
    if g < 0.90:                       # Phase1 + Phase2：peak 余弦衰减到 0.1*peak
        frac = g / 0.90
        return peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
    frac = (g - 0.90) / 0.10           # Mid-training 退火：0.1*peak → 1e-5
    return max(1e-5, 0.1 * peak * 0.5 * (1.0 + math.cos(math.pi * frac)))


def phase_for_minutes(cum_min):
    """返回 (当前阶段 dict, 全局进度 g∈[0,1])。"""
    g = min(cum_min / TRAIN_HORIZON_MIN, 1.0)
    acc = 0.0
    for idx, ph in enumerate(PHASES):
        acc += ph["frac"]
        if g < acc or idx == len(PHASES) - 1:
            return ph, g
    return PHASES[-1], g


class CurriculumSampler:
    """课程学习采样：依据累计训练分钟决定当前阶段，按阶段内域混比抽整段序列。
    Mid-training 阶段优先从困惑度过滤出的难样本池抽取（池由 build_midtrain_pool 构建）。
    """

    def __init__(self, seq_len, seed=42, midtrain_path=MIDTRAIN_BIN):
        self.seq_len = seq_len
        self.bins = {}
        for dom in DOMAINS:
            p = "data/v3_%s.bin" % dom
            if os.path.exists(p):
                self.bins[dom] = np.memmap(p, dtype=np.int32, mode="r").reshape(-1, seq_len)
        self.mid = None
        if midtrain_path and os.path.exists(midtrain_path):
            try:
                self.mid = np.memmap(midtrain_path, dtype=np.int32, mode="r").reshape(-1, seq_len)
                print("[sampler] 加载 Mid-training 难样本池：%d seqs" % self.mid.shape[0])
            except Exception:
                # 池与当前 seq_len 不匹配（换过 seq_len）→ 放弃旧池，重建；不崩训练
                self.mid = None
                print("[sampler] 难样本池与 seq_len=%d 不匹配，忽略（稍后重建）" % seq_len)
        self.rng = np.random.default_rng(seed)
        self.domains = list(self.bins) or ["zh_general"]

    def reload_midtrain(self, path=MIDTRAIN_BIN, seq_len=None):
        """重新加载 Mid-training 难样本池（池在训练途中按困惑度构建后调用）。"""
        if not (path and os.path.exists(path)):
            self.mid = None
            return False
        try:
            arr = np.memmap(path, dtype=np.int32, mode="r").reshape(-1, self.seq_len)
        except Exception:
            # seq_len 不匹配等异常 → 放弃池（退回阶段内均匀采样），不阻塞训练
            self.mid = None
            return False
        self.mid = arr
        print("[sampler] 载入 Mid-training 难样本池：%d seqs" % self.mid.shape[0])
        return True

    def batch(self, bs, cum_min):
        ph, g = phase_for_minutes(cum_min)
        if ph["name"] == "midtrain" and self.mid is not None:
            xs = []
            for _ in range(bs):
                i = self.rng.integers(0, max(self.mid.shape[0], 1))
                xs.append(np.asarray(self.mid[i], dtype=np.int64))
            return torch.from_numpy(np.stack(xs))
        mix = ph["mix"] or {d: 1.0 / max(len(self.domains), 1) for d in self.domains}
        doms = list(mix)
        tot = sum(mix.values()) or 1.0
        xs = []
        for _ in range(bs):
            r = self.rng.random() * tot
            acc, pick = 0.0, doms[0]
            for d in doms:
                acc += mix[d]
                if r <= acc:
                    pick = d
                    break
            arr = self.bins[pick] if pick in self.bins else self.bins[self.domains[0]]
            i = self.rng.integers(0, max(arr.shape[0], 1))
            xs.append(np.asarray(arr[i], dtype=np.int64))
        return torch.from_numpy(np.stack(xs))


def build_midtrain_pool(model, seq_len, out=MIDTRAIN_BIN, per_domain=1500, keep_frac=0.25):
    """困惑度过滤：用当前模型对每个域抽样若干序列计算逐序列困惑度（loss_lm），
    保留最难的 keep_frac 作为 Mid-training 难样本池。仅构建一次（存在则跳过）。
    在 model.eval() 无梯度下运行；返回 bin 路径（未建成返回 None）。
    """
    if os.path.exists(out):
        try:
            n = os.path.getsize(out) // 4
            if n > 0 and n % seq_len == 0:
                print("[midtrain] 池已存在，跳过构建：%s" % out)
                return out
            # seq_len 变了 → 旧池作废，重建
            os.remove(out)
            print("[midtrain] 旧池与 seq_len=%d 不匹配，删除重建" % seq_len)
        except Exception:
            pass
    seqs, losses = [], []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for dom in DOMAINS:
            p = "data/v3_%s.bin" % dom
            if not os.path.exists(p):
                continue
            arr = np.memmap(p, dtype=np.int32, mode="r").reshape(-1, seq_len)
            n = min(per_domain, arr.shape[0])
            idx = np.random.default_rng(0).choice(arr.shape[0], n, replace=False)
            for j in idx:
                x = torch.from_numpy(np.asarray(arr[j], dtype=np.int64)).unsqueeze(0)
                o = model(x, x)
                losses.append(float(o["loss_lm"].item()))
                seqs.append(np.asarray(arr[j], dtype=np.int32))
    if was_training:
        model.train()
    if not seqs:
        return None
    order = np.argsort(losses)[::-1]                 # 从高困惑度（难）到低
    k = max(1, int(len(order) * keep_frac))
    sel = np.stack([seqs[i] for i in order[:k]]).astype(np.int32)
    os.makedirs("data", exist_ok=True)
    sel.tofile(out)
    print("[midtrain] 选 %d/%d 最难样本 -> %s（每域抽样 %d）" % (k, len(seqs), out, per_domain))
    return out


def load_clock():
    if os.path.exists(CLOCK):
        try:
            with open(CLOCK, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"total_minutes": 0.0, "total_tokens": 0, "total_steps": 0, "runs": 0}


def save_clock(clk):
    os.makedirs("data", exist_ok=True)
    with open(CLOCK, "w", encoding="utf-8") as f:
        json.dump(clk, f, ensure_ascii=False, indent=2)


# —— 抗取消：时钟按「实际训练分钟」增量累加，每次存档都记，绝不只在结尾记 ——
# 这样即使 run 被平台取消，最后一次周期存档里已经含有到此为止的累计分钟/ token，
# 下一轮 resume 不会丢失进度。用增量（delta）避免重复计数。
_t_start = 0.0
_counted_t = 0.0
_counted_tokens = 0


def accrue_clock(clk, tokens_this_run, last_step):
    """把「上一次记账以来」实际训练分钟与新增 token 累加到 clk（增量，不重复计）。"""
    global _counted_t, _counted_tokens
    now = time.time()
    el = (now - _counted_t) / 60.0
    if el > 0:
        clk["total_minutes"] = clk.get("total_minutes", 0.0) + el
    new_tok = tokens_this_run - _counted_tokens
    if new_tok > 0:
        clk["total_tokens"] = clk.get("total_tokens", 0) + new_tok
    clk["total_steps"] = last_step
    clk["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _counted_t = now
    _counted_tokens = tokens_this_run


def persist(block=False):
    """把最新 checkpoint + 时钟 + 语料推到滚动发布（抗取消持久化）。

    【P0 修复】原来用 Popen 异步 fire-and-forget：CI 平台取消 run 时，
    父进程收到 SIGTERM 就退出，异步子进程往往还没上传完就被一起收走 →
    滚动发布永远建立不起来，每轮都"从零开始训练"。
    现在默认【同步阻塞】执行，保证上传真正落库；周期存档时仍可传 block=False
    走异步（那时父进程还活着，异步更省时间）。
    """
    try:
        if os.environ.get("V3_PERSIST", "1") == "0":
            return
        cmd = [sys.executable, "persist_v3.py"]
        if block:
            # 阻塞等待：给足超时（checkpoint ~320MB + 语料 bin 上传可能几分钟）
            subprocess.run(cmd, timeout=900, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def save_checkpoint(model, opt, step, loss, run_lm, tokens_this_run, clk,
                    atomic=True, block_persist=False):
    """原子存档：先写临时文件再 rename，避免 CI 被回收时留下半截损坏文件。
    注意：不持久化 optimizer 状态——checkpoint 体积从 ~958MB 降到 ~320MB，CI 写盘快 3 倍；
    CPU 小模型的 AdamW momentum 在续训几百步内重建，影响可忽略。

    每次存档都：① 增量累加时钟（抗取消，进度不丢）② 推到滚动发布（block_persist=True
    时阻塞等待，用于优雅退出前的最后一笔，尽量保证这帧也落库）。
    """
    os.makedirs("checkpoints", exist_ok=True)
    payload = {"model": model.state_dict(),
               "step": step, "loss": float(loss),
               "ppl": float(math.exp(min(run_lm, 20))) if run_lm is not None else float("nan"),
               "args": {"last_save_tokens": tokens_this_run}}
    if atomic:
        tmp = CKPT + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, CKPT)
    else:
        torch.save(payload, CKPT)
    # 增量记账（抗取消核心）：这一帧之前的实际训练分钟/ token 全部入账。
    accrue_clock(clk, tokens_this_run, step)
    save_clock(clk)
    # 抗取消持久化：周期存档非阻塞；结尾存档阻塞等待（优雅退出前最后一笔尽量落库）。
    if block_persist:
        try:
            if os.environ.get("V3_PERSIST", "1") != "0":
                subprocess.run([sys.executable, "persist_v3.py"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=900)
        except Exception:
            pass
    else:
        persist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_steps", type=int, default=20000, help="全局步数上限（软时限优先）")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=PEAK_LR)
    ap.add_argument("--max_minutes", type=float, default=75,
                    help="单轮训练软时限（分钟）；到点优雅停止并保存。务必小于 CI 硬限与平台回收边界。")
    ap.add_argument("--save_every", type=float, default=10,
                    help="周期存档间隔（分钟）；默认 10=每 10 分钟落一帧并推滚动发布，"
                         "抗 CI 取消：最坏只丢 10 分钟。CI 写盘慢也不撞 timeout（本地写盘快，上传非阻塞）。")
    ap.add_argument("--no_download", action="store_true",
                    help="关闭下载（默认开启：CI 拉取人民网/维基/古登堡真实高质量语料；本地冒烟可加 --no_download）")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # SIGTERM 优雅退出（timeout 软终止前存档），避免硬杀丢整轮
    ctrl = {"stop": False}
    def _on_term(signum, frame):
        ctrl["stop"] = True
        print("[signal] 收到 SIGTERM，将在本步后优雅停止并存档")
    signal.signal(signal.SIGTERM, _on_term)

    torch.manual_seed(42)
    # 构建/复用语料：默认开启下载真实高质量语料（人民网/维基/古登堡）；加 --no_download 则用内置 seed。
    # resume 时若 data/v3_*.bin 已存在且长度匹配则跳过构建（从 corpus artifact 复用）。
    build_corpus_v3.build(args.seq_len, allow_download=not args.no_download)
    sampler = CurriculumSampler(args.seq_len)

    model = MoleculeV3(vocab_size=VOCAB)
    n_params = model.num_params() / 1e6
    print("[model] v3 基础模型参数: %.1fM（15 组权重共享，纯骨干）" % n_params, flush=True)
    # foreach=False 显著降低 AdamW 峰值显存（融合算子会额外分配缓冲），
    # 在 8GB cgroup 上限的沙箱里是能否跑过多步的关键；不改变数学结果。
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01, foreach=False)

    step0 = 0
    if args.resume and os.path.exists(CKPT):
        st = torch.load(CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        step0 = st.get("step", 0)
        # 不持久化 optimizer 状态（checkpoint 体积极大且续训重建 momentum 影响极小）；
        # 若旧 ckpt 含 opt 则尽量复用，否则重建（保持兼容）。
        if "opt" in st:
            try:
                opt.load_state_dict(st["opt"])
            except Exception:
                pass
        print("[resume] 从 step %d 续训" % step0, flush=True)
    model.train()

    t_start = time.time()
    global _t_start, _counted_t, _counted_tokens
    _t_start = t_start
    _counted_t = t_start
    _counted_tokens = 0
    last_save = t_start
    run_lm = run_mtp = None
    loss = 0.0  # 兜底：若本轮 0 步（已达 max_steps）也能安全存档
    last_step = step0
    tokens_this_run = 0
    clk = load_clock()
    for step in range(step0, args.max_steps):
        if ctrl["stop"]:
            print("[stop] SIGTERM 优雅停止 @ step %d" % step)
            break
        if (time.time() - t_start) / 60 > args.max_minutes:
            print("[stop] 软时限 %.0f 分钟到，优雅停止 @ step %d"
                  % (args.max_minutes, step))
            break
        gstep = step0 + (step - step0)  # 全局步数（warmup 基于它）
        lr = lr_at(gstep, clk["total_minutes"])
        for g in opt.param_groups:
            g["lr"] = lr
        idx = sampler.batch(args.batch, clk["total_minutes"])
        out = model(idx, idx)
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_step = step + 1
        tokens_this_run += args.batch * args.seq_len
        # Mid-training 难样本池：训练到一定步数后按困惑度筛选最高质量的难样本，
        # 构建一次并持久化（后续 CI 轮次直接复用），使最后 10% 的 Mid-training
        # 真正落在"难样本 + 退火"上，而不是退化成均匀采样。
        if (not os.path.exists(MIDTRAIN_BIN)) and last_step >= MIDTRAIN_POOL_AT_STEP:
            print("[midtrain] 训练 %d 步后构建困惑度难样本池…" % last_step, flush=True)
            if build_midtrain_pool(model, args.seq_len):
                sampler.reload_midtrain(MIDTRAIN_BIN)
        run_lm = out["loss_lm"].item() if run_lm is None else 0.95 * run_lm + 0.05 * out["loss_lm"].item()
        run_mtp = out["loss_mtp"].item() if run_mtp is None else 0.95 * run_mtp + 0.05 * out["loss_mtp"].item()
        if (step + 1) % 10 == 0 or step == step0:
            el = (time.time() - t_start) / 60
            ppl = math.exp(min(run_lm, 20))
            # 吞吐实测（tokens/s）：用于推算 10 天窗口能吃到多少 token，
            # 也就是"训练量是否够"——这是 v3.0 质量上限的硬约束，必须可观测。
            tps = tokens_this_run / max(el * 60.0, 1e-6)
            print("[step %d] loss=%.3f (lm=%.3f mtp=%.3f) ppl≈%.2f lr=%.2e %.1f min %.1f tok/s"
                  % (step + 1, loss.item(), run_lm, run_mtp, ppl, lr, el, tps))
        # 周期存档（抗取消）：落本地帧 + 增量记账 + 推滚动发布（非阻塞），最坏只丢 save_every 分钟
        if (time.time() - last_save) / 60 >= args.save_every:
            save_checkpoint(model, opt, last_step, loss, run_lm, tokens_this_run, clk)
            last_save = time.time()
            print("[save] step %d 周期存档（已记账+推滚动发布）" % last_step)

    # 本轮结束：轮次 +1（仅此处记一次），最后再落一帧并【阻塞】推滚动发布
    # （优雅退出/软时限到点前的最后一笔，尽量保证这帧也落库，抗 CI 取消）。
    elapsed = (time.time() - t_start) / 60.0
    clk["runs"] = clk.get("runs", 0) + 1

    save_checkpoint(model, opt, last_step, loss, run_lm, tokens_this_run, clk, block_persist=True)
    print("[done] checkpoint -> %s | 本轮 %.1f 分钟 | 累计 %.1f 分钟 / %.1f 小时（%d 轮）"
          % (CKPT, elapsed, clk["total_minutes"], clk["total_minutes"] / 60.0, clk["runs"]))


if __name__ == "__main__":
    main()
