# -*- coding: utf-8 -*-
"""按书 §28.1「稳杆为主」口径评估一个训练好的 REINFORCE 策略。

对给定 `.pth`，复用 `eval/bench.py` 的同一批 seed=42 初值、`rollout`/`cost_quadratic`/`J_ach`、
`max_steps=1000`，但存活判据改为**稳杆为主**：全程不越界（满 1000 步）且 末 H=500 步 `|θ|≤2°`；
`|x|≤0.5m` 位置回中**降为次级参考**（仅记录，不作存活门槛）。这与书 §28.3（及 §28.1 的 REINFORCE
叙事）所用的稳杆为主口径一致。

用法（cwd=CartPole）：
    python src-rl/eval_noc_term.py <ckpt.pth>

输出每个 level 的：回合长、稳杆存活率、`J_ach`（存活回合）、末 500 步 `|x|` 均值/最大、`|θ|` 最大。
"""
import json
import os
import sys

import numpy as np

_THIS = os.path.abspath(os.path.dirname(__file__))
for _p in (_THIS, os.path.join(_THIS, "..", "src"), os.path.join(_THIS, "..", "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench  # noqa: E402  （复用 LEVELS/gen_init_states/rollout/cost_quadratic/口径）
from rl_common import ReinforcePolicyController, load_policy_ckpt  # noqa: E402

THETA_DEG = 2.0  # 稳杆为主：末 500 步 |θ|≤2°


def fmt(v):
    return "—" if v is None else f"{v:.2f}"


def eval_ckpt(ckpt_path, device="cpu"):
    """评估一个 ckpt（稳杆为主口径）；返回聚合 dict（每个 level 一组统计）。"""
    policy, obs_rms, meta = load_policy_ckpt(
        ckpt_path, obs_dim=4, act_dim=1, hidden_sizes=(64, 64),
        device=device, init_log_std=0.0)
    ctrl = ReinforcePolicyController(policy, obs_rms, device=device, mean=True)
    env = bench.CartPoleCustomEnv(bench.CONFIG)

    np.random.seed(bench.SEED)
    init_by_level = {lv: bench.gen_init_states(lv, bench.K) for lv in bench.LEVELS}

    out = {"meta": meta, "levels": {}}
    for lv in bench.LEVELS:
        x0_list = init_by_level[lv][:bench.K]
        lens, jac, lx_mean, lx_max, ltheta_max, surv = [], [], [], [], [], 0
        for x0 in x0_list:
            x_seq, u_seq, _surv, reason = bench.rollout(ctrl, env, x0)
            T = len(x_seq) - 1
            lens.append(T)
            full_horizon = reason == "满步"           # 未提前越界
            if full_horizon and len(x_seq) >= bench.H:
                last = x_seq[-bench.H:]
                settled = bool(
                    np.all(np.abs(last[:, 2]) <= THETA_DEG * np.pi / 180.0))
            else:
                settled = False
            survived = full_horizon and settled
            if survived:
                surv += 1
                jac.append(bench.cost_quadratic(x_seq, u_seq, bench.Q_EVAL, bench.R_EVAL))
            if len(x_seq) >= bench.H:
                last = x_seq[-bench.H:]
                lx_mean.append(np.abs(last[:, 0]).mean())
                lx_max.append(np.abs(last[:, 0]).max())
                ltheta_max.append(np.abs(last[:, 2]).max())
        out["levels"][lv] = dict(
            lens=lens, surv=surv, K=bench.K,
            J_mean=float(np.mean(jac)) if jac else None,
            J_std=float(np.std(jac)) if jac else None,
            last_x_mean=float(np.mean(lx_mean)) if lx_mean else None,
            last_x_max=float(np.max(lx_max)) if lx_max else None,
            last_theta_max=float(np.max(ltheta_max)) if ltheta_max else None,
        )
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法: eval_noc_term.py <ckpt.pth>")
    ckpt = sys.argv[1]
    out = eval_ckpt(ckpt)
    print("=" * 78)
    print("ckpt:", ckpt)
    print("meta:", json.dumps(out["meta"], ensure_ascii=False))
    print("=" * 78)
    for lv, r in out["levels"].items():
        print(f"[{lv}] 回合长 {min(r['lens']):.0f}/{np.mean(r['lens']):.0f}/{max(r['lens']):.0f} | "
              f"稳杆存活 {r['surv']}/{r['K']} | J_ach={fmt(r['J_mean'])}±{fmt(r['J_std'])} | "
              f"末500步|x|均值={fmt(r['last_x_mean'])} 最大={fmt(r['last_x_max'])} | |θ|最大={fmt(r['last_theta_max'])}")
    return out


if __name__ == "__main__":
    main()
