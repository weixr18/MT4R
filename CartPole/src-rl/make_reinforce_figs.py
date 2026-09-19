# -*- coding: utf-8 -*-
"""按「当前定稿种子（seed2）」重新生成书 §28.1 的 REINFORCE 图。

生成：
- imgs/cartpole/reinforce_train_curve.png：训练回报/回合长曲线（来自 seed2 的 history JSON）；
- imgs/cartpole/reinforce_L{1,2,3}.png：seed2 最优 ckpt 在各 level 的**第 1 回合**时间序列（5 面板）。

用法（cwd=CartPole）：
    py311-gym\\...\\python.exe src-rl/make_reinforce_figs.py
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
RES = os.path.join(_THIS, "res", "s_rf_retrain_l3_noc_term")
# 出图写书仓库 imgs/cartpole/（本文件位于 4-MT4R-github/CartPole/src-rl/），可用 FIG_DIR 覆盖
FIG_DIR = os.environ.get("FIG_DIR", os.path.abspath(os.path.join(
    _THIS, "..", "..", "..", "1-MN4R", "imgs", "cartpole")))


def make_train_curve():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hist = json.load(open(os.path.join(RES, f"seed{SEED}", "reward_history_reinforce_cartpole.json"),
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
    fig.suptitle(f"REINFORCE on CartPole (seed={SEED})")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "reinforce_train_curve.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[fig] train_curve ->", out)


def make_rollout_figs():
    ckpt = os.path.join(RES, f"seed{SEED}", "ckpt", "ckpt_cartpole_iter0499.pth")
    policy, obs_rms, meta = load_policy_ckpt(
        ckpt, obs_dim=4, act_dim=1, hidden_sizes=(64, 64), device="cpu", init_log_std=0.0)
    ctrl = ReinforcePolicyController(policy, obs_rms, device="cpu", mean=True)
    env = bench.CartPoleCustomEnv(bench.CONFIG)
    np.random.seed(bench.SEED)
    init_by_level = {lv: bench.gen_init_states(lv, bench.K) for lv in bench.LEVELS}
    for lv in bench.LEVELS:
        x0 = init_by_level[lv][0]
        x_seq, u_seq, _surv, reason = bench.rollout(ctrl, env, x0)
        path = os.path.join(FIG_DIR, f"reinforce_L{lv[-1]}.png")
        bench.visualize_episode(x_seq, u_seq, path,
                                f"REINFORCE - Level {lv} - Episode 1 (seed={SEED})")
        print(f"  [fig] {lv} -> {os.path.basename(path)} ({reason}, len={len(x_seq)-1})")


if __name__ == "__main__":
    make_train_curve()
    make_rollout_figs()
