# -*- coding: utf-8 -*-
"""评估一个训练好的 REINFORCE 策略的「位置回中」能力（Session 5 原型/正式评估共用）。

对给定 `.pth`，复用 `eval/bench.py` 的统一评估口径（同一批 seed=42 初值、同一 `rollout`/
`cost_quadratic`/`J_ach` 判据、`max_steps=1000`），额外统计每个 level 各回合的：
- 回合长、存活率、存活回合的 `J_ach`；
- 末 `H=500` 步 `|x|` 的均值/最大、`|θ|` 的最大（是否 `|x|≤0.5m` 且 `|θ|≤2°`）；
- 一个「标称」初值（x0=0、θ=3°）下的完整轨迹 `|x|`，用于隔离位置控制本身。

用法（cwd=CartPole）：
    E:/Anaconda3/envs/py311-gym/python.exe src-rl/eval_centering.py <ckpt.pth>
"""
import json
import os
import sys

import numpy as np

# 路径：src-rl 当前目录 + 同级的 eval/ 与 src/
_THIS = os.path.abspath(os.path.dirname(__file__))
for _p in (_THIS, os.path.join(_THIS, "..", "src"), os.path.join(_THIS, "..", "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench  # noqa: E402  （复用 LEVELS/gen_init_states/rollout/cost_quadratic/口径）
from rl_common import ReinforcePolicyController, load_policy_ckpt  # noqa: E402


def fmt(v):
    return "—" if v is None else f"{v:.2f}"


def eval_ckpt(ckpt_path, device="cpu"):
    """评估一个 ckpt；返回聚合 dict（每个 level 一组统计）。"""
    policy, obs_rms, meta = load_policy_ckpt(
        ckpt_path, obs_dim=4, act_dim=1, hidden_sizes=(64, 64),
        device=device, init_log_std=0.0)
    ctrl = ReinforcePolicyController(policy, obs_rms, device=device, mean=True)
    env = bench.CartPoleCustomEnv(bench.CONFIG)

    # 同一批 seed=42 初值（与 bench.py 完全一致）
    np.random.seed(bench.SEED)
    init_by_level = {lv: bench.gen_init_states(lv, bench.K) for lv in bench.LEVELS}

    out = {"meta": meta, "levels": {}}
    for lv in bench.LEVELS:
        x0_list = init_by_level[lv][:bench.K]
        lens, jac, lx_mean, lx_max, ltheta_max, surv = [], [], [], [], [], 0
        for x0 in x0_list:
            x_seq, u_seq, survived, reason = bench.rollout(ctrl, env, x0)
            lens.append(len(x_seq) - 1)
            if survived:
                surv += 1
                jac.append(bench.cost_quadratic(x_seq, u_seq, bench.Q_EVAL, bench.R_EVAL))
            if len(x_seq) >= bench.H:
                last = x_seq[-bench.H:]
                lx = np.abs(last[:, 0])
                ltheta = np.abs(last[:, 2])
                lx_mean.append(lx.mean())
                lx_max.append(lx.max())
                ltheta_max.append(ltheta.max())
        out["levels"][lv] = dict(
            lens=lens, surv=surv, K=bench.K,
            J_mean=float(np.mean(jac)) if jac else None,
            J_std=float(np.std(jac)) if jac else None,
            last_x_mean=float(np.mean(lx_mean)) if lx_mean else None,
            last_x_max=float(np.max(lx_max)) if lx_max else None,
            last_theta_max=float(np.max(ltheta_max)) if ltheta_max else None,
        )

    # 标称初值（隔离位置控制；θ0=3°（近 L1 上界）、其余 0）
    x0_nom = np.array([0.0, 0.0, 3.0 * np.pi / 180.0, 0.0], dtype=np.float64)
    x_seq, u_seq, survived, reason = bench.rollout(ctrl, env, x0_nom)
    if len(x_seq) >= bench.H:
        last = x_seq[-bench.H:]
        out["nominal"] = dict(
            len=len(x_seq) - 1, survived=survived, reason=reason,
            last_x_mean=float(np.abs(last[:, 0]).mean()),
            last_x_max=float(np.abs(last[:, 0]).max()),
            last_theta_max=float(np.abs(last[:, 2]).max()),
        )
    else:
        out["nominal"] = dict(len=len(x_seq) - 1, survived=survived, reason=reason)
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法: eval_centering.py <ckpt.pth>")
    ckpt = sys.argv[1]
    out = eval_ckpt(ckpt)
    print("=" * 78)
    print("ckpt:", ckpt)
    print("meta:", json.dumps(out["meta"], ensure_ascii=False))
    print("=" * 78)
    for lv, r in out["levels"].items():
        print(f"[{lv}] 回合长 min/mean/max = {min(r['lens']):.0f}/"
              f"{np.mean(r['lens']):.0f}/{max(r['lens']):.0f} | 存活率 {r['surv']}/{r['K']} | "
              f"J_ach={fmt(r['J_mean'])}±{fmt(r['J_std'])} | "
              f"末500步 |x| 均值={fmt(r['last_x_mean'])} 最大={fmt(r['last_x_max'])} | "
              f"|θ|最大={fmt(r['last_theta_max'])}")
    n = out["nominal"]
    print(f"[标称] len={n['len']:.0f} survived={n['survived']} reason={n['reason']} | "
          f"末500步 |x| 均值={fmt(n.get('last_x_mean'))} 最大={fmt(n.get('last_x_max'))} | "
          f"|θ|最大={fmt(n.get('last_theta_max'))}")
    return out


if __name__ == "__main__":
    main()
