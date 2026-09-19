# -*- coding: utf-8 -*-
"""Session 3 诊断工具：可视化已训练策略的轨迹 + 终止原因分布。

对应 `1-MN4R/docs/cartpole-training/REINFORCE-training-log.md`（原 `REINFORCE-verify.md` 已并入）「先诊断，后小步改」：
- 从 `1_reinforce.py` 保存的 .pth 加载策略；
- 对给定初值跑一条轨迹，画 5 面板（x、ẋ、θ、θ̇、F）时间序列，看清它是怎么早期终止的
  （力不够饱和 / 力过大震荡 / x 越界 vs θ 越界）；
- 跑 N 条随机轨迹，统计终止原因分布与 return/ep_len 统计量。

用法（cwd 任意，脚本自解析 ../src 路径）：
    python src-rl/diagnose.py --ckpt src-rl/res/checkpoints/<...>/ckpt_cartpole_latest.pth
                             [--level L1|L2|L3] [--n-eval 50] [--sample] [--out-dir res/diag]

初值：--init 不设 seed（不动全局 RNG）；--level 用 bench 同款 L1–L3 初值生成逻辑
（仅诊断复现，与 `eval/bench.py` 的 `gen_init_states` 一致）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# 让本脚本从任意 cwd 都能 import ../src/env_cartpole.py（单一环境来源）
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rl_common import (  # noqa: E402
    ReinforcePolicyController,
    load_policy_ckpt,
)

# 与 eval/bench.py 相同的级别定义与初值生成（仅用于诊断复现；正式评估以 bench.py 为准）
LEVELS = {
    "L1": dict(theta_amp=(0.0, 3.0), theta_dot=1.0, x=0.3, x_dot=0.3),
    "L2": dict(theta_amp=(3.0, 10.0), theta_dot=3.0, x=0.8, x_dot=0.5),
    "L3": dict(theta_amp=(20.0, 35.0), theta_dot=10.0, x=1.5, x_dot=1.0),
}
SEED = 42


def gen_init_states(level, k, seed=None):
    """与 bench.py 的 gen_init_states 同口径，仅用于诊断。"""
    if seed is not None:
        np.random.seed(seed)
    spec = LEVELS[level]
    states = np.zeros((k, 4))
    for i in range(k):
        theta_amp = np.random.uniform(*spec["theta_amp"])
        sign = np.random.choice([-1.0, 1.0])
        theta = sign * theta_amp * np.pi / 180.0
        theta_dot = np.random.uniform(-spec["theta_dot"], spec["theta_dot"]) * np.pi / 180.0
        x = np.random.uniform(-spec["x"], spec["x"])
        x_dot = np.random.uniform(-spec["x_dot"], spec["x_dot"])
        states[i] = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
    return states


def rollout(controller, env, x0):
    """闭环跑一条轨迹，返回 (x_seq, u_seq, rewards, survived, reason)。

    与 eval/bench.py 的 rollout() 同口径（只写 env.state，u_seq/rewards 记录每步实际量）。
    """
    env.state = np.array(x0, dtype=np.float32)
    env.steps = 0
    controller.reset()
    x_seq = [np.array(x0, dtype=np.float64)]
    u_seq = []
    rewards = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = controller.compute_action(x_seq[-1])
        next_state, reward, terminated, truncated, info = env.step(action)
        u_seq.append(float(info["F_applied"]))
        rewards.append(float(reward))
        x_seq.append(np.array(next_state, dtype=np.float64))
    x_seq = np.array(x_seq)
    u_seq = np.array(u_seq)
    rewards = np.array(rewards, dtype=np.float64)
    x, _, theta, _ = x_seq[-1]
    if abs(x) > env.config["x_threshold"]:
        reason = "x越界"
    elif abs(theta) > env.config["theta_threshold"]:
        reason = "θ越界"
    else:
        reason = "满步"
    max_steps_reached = env.steps >= env.config["max_steps"]
    out_of_bounds = not max_steps_reached
    if len(x_seq) >= 500:
        last = x_seq[-500:]
        settled = bool(np.all(np.abs(last[:, 2]) <= 2 * np.pi / 180.0) and
                       np.all(np.abs(last[:, 0]) <= 0.5))
    else:
        settled = False
    survived = (not out_of_bounds) and settled
    return x_seq, u_seq, rewards, survived, reason


def visualize_trajectory(x_seq, u_seq, dt, path, title):
    """5 面板时间序列（x、ẋ、θ、θ̇、F），风格同 eval/bench.py。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states = x_seq[1:]
    actions = u_seq
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title, fontsize=16)
    t = np.arange(len(states)) * dt
    theta_deg = states[:, 2] * 180.0 / np.pi
    theta_dot_deg = states[:, 3] * 180.0 / np.pi
    specs = [("x (Position)", "C0", states[:, 0]),
             (r"$\dot{x}$ (Velocity)", "C1", states[:, 1]),
             (r"$\theta$ (°)", "C2", theta_deg),
             (r"$\dot{\theta}$ (°/s)", "C3", theta_dot_deg)]
    for i, (label, color, y) in enumerate(specs):
        axs[i].plot(t, y, label=label, color=color)
        axs[i].set_ylabel(label)
        axs[i].legend()
    axs[4].plot(t, actions, label="F (Force)", color="C4")
    axs[4].axhline(10, color="gray", ls="--", lw=0.8)
    axs[4].axhline(-10, color="gray", ls="--", lw=0.8, label=r"$\pm F_{max}$")
    axs[4].set_ylabel("F [N]")
    axs[4].set_xlabel("Time [s]")
    axs[4].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")


def main():
    p = argparse.ArgumentParser(description="诊断：可视化已训练策略轨迹 + 终止原因分布")
    p.add_argument("--ckpt", required=True, help="1_reinforce.py 保存的 .pth")
    p.add_argument("--hidden", default="64,64", help="网络结构（需与训练一致）")
    p.add_argument("--init-log-std", type=float, default=0.0)
    p.add_argument("--level", choices=list(LEVELS.keys()), help="用该级初值跑一条轨迹")
    p.add_argument("--sample", action="store_true", help="评估时采样动作（默认取均值 μ）")
    p.add_argument("--max-steps", type=int, default=None,
                   help="覆盖环境单回合步数上限（默认用 env CONFIG 的 1000）")
    p.add_argument("--n-eval", type=int, default=50,
                   help="统计终止原因/return 分布的随机轨迹条数（env 自然分布）")
    p.add_argument("--out-dir", default=None, help="产物目录（默认 <脚本目录>/res/diag）")
    args = p.parse_args()

    from env_cartpole import CONFIG, CartPoleCustomEnv
    env_config = None
    if args.max_steps is not None:
        env_config = {"max_steps": args.max_steps}
    env = CartPoleCustomEnv({**CONFIG, **(env_config or {})})
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    hidden = tuple(int(x) for x in args.hidden.split(","))
    dt = CONFIG["dt"]

    policy, obs_rms, meta = load_policy_ckpt(
        args.ckpt, obs_dim, act_dim, hidden_sizes=hidden, init_log_std=args.init_log_std)
    device = next(policy.parameters()).device
    controller = ReinforcePolicyController(policy, obs_rms, device=device, mean=not args.sample)

    print(f"[load] meta={meta} | mean_action={not args.sample}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "res", "diag"))
    os.makedirs(out_dir, exist_ok=True)

    # ---- 定点轨迹：--level 或 env 自然分布首条 ----
    if args.level:
        states = gen_init_states(args.level, 1, seed=SEED)
        x0 = states[0]
        src = f"level {args.level}"
    else:
        x0 = env.reset()[0]
        src = "env-natural"
    x_seq, u_seq, rewards, survived, reason = rollout(controller, env, x0)
    path = os.path.join(out_dir, f"traj_{src.replace(' ', '_')}_"
                                 f"{'survived' if survived else 'failed'}.png")
    title = f"REINFORCE ckpt {os.path.basename(args.ckpt)} | {src} | {reason}"
    if not survived:
        title += " (not settled)"
    visualize_trajectory(x_seq, u_seq, dt, path, title)
    print(f"[ep] {src}: len={len(x_seq)-1}, reason={reason}, return={rewards.sum():.1f}, "
          f"survived={survived}")

    # ---- 随机轨迹统计（env 自然分布）----
    reasons = []
    lens = []
    rets = []
    for _ in range(args.n_eval):
        x0 = env.reset()[0]
        x_seq, u_seq, rewards, survived, reason = rollout(controller, env, x0)
        reasons.append(reason)
        lens.append(len(x_seq) - 1)
        rets.append(float(rewards.sum()))

    n = len(reasons)
    counts = {r: reasons.count(r) for r in sorted(set(reasons))}
    print(f"\n[stats] over {n} episodes (natural init):")
    print(f"  reason: {counts}")
    print(f"  ep_len mean={float(np.mean(lens)):.1f} std={float(np.std(lens)):.1f} "
          f"min={int(np.min(lens))} max={int(np.max(lens))}")
    print(f"  return mean={float(np.mean(rets)):.1f} std={float(np.std(rets)):.1f} "
          f"min={float(np.min(rets)):.1f} max={float(np.max(rets)):.1f}")

    summary = dict(
        ckpt=args.ckpt, meta=meta, mean_action=not args.sample, n_eval=n,
        reason_counts=counts,
        ep_len=dict(mean=float(np.mean(lens)), std=float(np.std(lens)),
                    min=int(np.min(lens)), max=int(np.max(lens))),
        return_stats=dict(mean=float(np.mean(rets)), std=float(np.std(rets)),
                          min=float(np.min(rets)), max=float(np.max(rets))),
    )
    with open(os.path.join(out_dir, "diag_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[save] {os.path.join(out_dir, 'diag_summary.json')}")


if __name__ == "__main__":
    main()
