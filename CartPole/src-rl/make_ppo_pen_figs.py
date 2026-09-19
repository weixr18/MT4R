# -*- coding: utf-8 -*-
"""生成 PPO-Penalty (--algo pen, klb0.02_banddef, ne2) 书§28.2 的图（与 REINFORCE/AC/AC-GAE/og 同风格）。

- imgs/cartpole/ppo_pen_train_curve.png：训练回报/回合长曲线（最优种子 seed=2 的 history JSON；
  seed=2 的 best ckpt 是 iter0499，且全程单调改善、无「冲高回落」）。
- imgs/cartpole/ppo_pen_L{1,2,3}.png：seed=2 最优 ckpt (iter0499) 各 level 第 1 回合时间序列。

用法（cwd=CartPole）：
    py311-gym\\...\\python.exe src-rl/make_ppo_pen_figs.py
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

SEED = 2
RES = os.path.join(_THIS, "res", "s_ppo_pen_formal500")
CKPT = os.path.join(RES, f"seed{SEED}", "ckpt", "ckpt_cartpole_ppo_iter0499.pth")
# 出图写书仓库 imgs/cartpole/（本文件位于 4-MT4R-github/CartPole/src-rl/），可用 FIG_DIR 覆盖
FIG_DIR = os.environ.get("FIG_DIR", os.path.abspath(os.path.join(
    _THIS, "..", "..", "..", "1-MN4R", "imgs", "cartpole")))


def make_train_curve():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    hist = json.load(open(os.path.join(RES, f"seed{SEED}", "reward_history_ppo_pen_cartpole.json"),
                          encoding="utf-8"))
    iters = [h["iter"] for h in hist]
    ret = [h["ret_mean"] for h in hist]
    ret_std = [h["ret_std"] for h in hist]
    leng = [h["ep_len_mean"] for h in hist]
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(iters, ret, color="C0")
    ax[0].fill_between(iters, np.array(ret) - np.array(ret_std),
                       np.array(ret) + np.array(ret_std), color="C0", alpha=0.25)
    ax[0].set_ylabel("episode return (mean±std)")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(iters, leng, color="C1")
    ax[1].set_ylabel("episode length")
    ax[1].set_xlabel("iteration")
    ax[1].grid(True, alpha=0.3)
    fig.suptitle(f"PPO-Penalty (adaptive beta) on CartPole (seed={SEED})")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ppo_pen_train_curve.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[fig] train_curve ->", out)


def make_rollout_figs():
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(FIG_DIR, exist_ok=True)
    policy, obs_rms, meta = load_policy_ckpt(
        CKPT, obs_dim=4, act_dim=1, hidden_sizes=(64, 64), device="cpu", init_log_std=0.0)
    ctrl = ReinforcePolicyController(policy, obs_rms, device="cpu", mean=True)
    env = bench.CartPoleCustomEnv(bench.CONFIG)
    np.random.seed(bench.SEED)
    init_by_level = {lv: bench.gen_init_states(lv, bench.K) for lv in bench.LEVELS}
    for lv in bench.LEVELS:
        x0 = init_by_level[lv][0]
        x_seq, u_seq, _surv, reason = bench.rollout(ctrl, env, x0)
        path = os.path.join(FIG_DIR, f"ppo_pen_L{lv[-1]}.png")
        bench.visualize_episode(x_seq, u_seq, path,
                                f"PPO-Penalty - Level {lv} - Episode 1 (seed={SEED}, iter499)")
        print(f"  [fig] {lv} -> {os.path.basename(path)} ({reason}, len={len(x_seq)-1})")


if __name__ == "__main__":
    make_train_curve()
    make_rollout_figs()
