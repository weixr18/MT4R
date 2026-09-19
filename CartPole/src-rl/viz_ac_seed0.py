# -*- coding: utf-8 -*-
"""可视化 AC(无 GAE) seed0 最优 ckpt 的训练曲线与 1000 步评估轨迹（供人工判断）。

生成（写入 src-rl/res/s_ac_formal_500/seed0/viz/）：
- ac_train_curve.png：训练回报/回合长曲线（标注 iter≈310 突破点）；
- ac_seed0_L{1,2,3}.png：各 level 第 1 回合 1000 步的 x(t)/θ(t)/F(t)，标注
  ±2° θ 稳杆带与 ±0.5 m 位置目标带、末 500 步区间。

用法（cwd=CartPole）：
    py311-gym\\...\\python.exe src-rl/viz_ac_seed0.py
"""
import json
import os
import sys

import numpy as np

_THIS = os.path.abspath(os.path.dirname(__file__))
for _p in (_THIS, os.path.join(_THIS, "..", "src"), os.path.join(_THIS, "..", "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench  # noqa: E402
from rl_common import ReinforcePolicyController, load_policy_ckpt  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 0
_RUN = os.path.join(_THIS, "res", "s_ac_formal_500", "seed0")
CKPT = os.path.join(_RUN, "ckpt", "ckpt_cartpole_ac_iter0499.pth")
HIST = os.path.join(_RUN, "reward_history_ac_td_cartpole.json")
OUTDIR = os.path.join(_RUN, "viz")
THETA_DEG = 2.0
X_BAND = 0.5


def make_train_curve():
    os.makedirs(OUTDIR, exist_ok=True)
    hist = json.load(open(HIST, encoding="utf-8"))
    iters = [h["iter"] for h in hist]
    ret = [h["ret_mean"] for h in hist]
    rets = [h["ret_std"] for h in hist]
    leng = [h["ep_len_mean"] for h in hist]
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(iters, ret, color="C0")
    ax[0].fill_between(iters, np.array(ret) - np.array(rets), np.array(ret) + np.array(rets),
                       color="C0", alpha=0.25)
    ax[0].axvline(310, color="r", ls="--", alpha=0.6)
    ax[0].annotate("breakthrough ~iter310", xy=(310, ax[0].get_ylim()[0] + 0.05 * (ax[0].get_ylim()[1] - ax[0].get_ylim()[0])),
                   color="r", fontsize=8)
    ax[0].set_ylabel("episode return (mean±std)")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(iters, leng, color="C1")
    ax[1].axhline(500, color="k", ls=":", lw=0.8)
    ax[1].set_ylabel("episode length")
    ax[1].set_xlabel("iteration")
    ax[1].grid(True, alpha=0.3)
    fig.suptitle(f"AC (no GAE) on CartPole (seed={SEED}, max_steps=500, actor_lr=3e-3)")
    fig.tight_layout()
    p = os.path.join(OUTDIR, "ac_train_curve.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[fig]", p)


def make_level_fig(lv):
    policy, obs_rms, meta = load_policy_ckpt(
        CKPT, obs_dim=4, act_dim=1, hidden_sizes=(64, 64), device="cpu", init_log_std=0.0)
    ctrl = ReinforcePolicyController(policy, obs_rms, device="cpu", mean=True)
    env = bench.CartPoleCustomEnv(bench.CONFIG)
    np.random.seed(bench.SEED)
    init_by_level = {l: bench.gen_init_states(l, bench.K) for l in bench.LEVELS}
    x0 = init_by_level[lv][0]
    x_seq, u_seq, surv, reason = bench.rollout(ctrl, env, x0)
    T = len(x_seq) - 1
    t = np.arange(T) * bench.CONFIG["dt"]
    x = x_seq[1:, 0]
    th = x_seq[1:, 2] * 180.0 / np.pi
    H0 = max(0, T - bench.H)  # 末 500 步起点索引
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(t, x, color="C0", lw=1.0)
    ax[0].axhline(X_BAND, color="r", ls="--", lw=0.9)
    ax[0].axhline(-X_BAND, color="r", ls="--", lw=0.9)
    ax[0].axvspan(t[H0], t[-1], color="gray", alpha=0.12)
    ax[0].set_ylabel("x (m)")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(["x", "±0.5m target"], loc="upper right", fontsize=8)
    ax[1].plot(t, th, color="C2", lw=1.0)
    ax[1].axhline(THETA_DEG, color="r", ls="--", lw=0.9)
    ax[1].axhline(-THETA_DEG, color="r", ls="--", lw=0.9)
    ax[1].axvspan(t[H0], t[-1], color="gray", alpha=0.12)
    ax[1].set_ylabel("theta (deg)")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(["theta", "±2° band"], loc="upper right", fontsize=8)
    ax[2].plot(t, u_seq, color="C4", lw=1.0)
    ax[2].set_ylabel("F (N)")
    ax[2].set_xlabel("time [s]")
    ax[2].grid(True, alpha=0.3)
    fig.suptitle(
        f"AC(no GAE) seed={SEED} - Level {lv} - Episode 1 "
        f"(len={T}, reason={({'满步': 'full-horizon', 'x越界': 'x-out', 'θ越界': 'theta-out'}[reason])}, surv={surv})")
    fn = os.path.join(OUTDIR, f"ac_seed0_L{lv[-1]}.png")
    fig.savefig(fn, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {fn}  (len={T}, reason={reason}, surv={surv})")


if __name__ == "__main__":
    make_train_curve()
    for lv in bench.LEVELS:
        make_level_fig(lv)
