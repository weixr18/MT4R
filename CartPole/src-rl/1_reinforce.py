# -*- coding: utf-8 -*-
"""纯 REINFORCE 主循环（连续动作、高斯策略）。

对应书：
- `chap6_3_pgmeth_2.tex` §21.5 REINFORCE 算法（algo:pg_reinforce）的连续动作版；
- `chap6_3_pgmeth_1.tex` §21.3 eq:plcgrad-thrm-gauss（高斯策略梯度）。

每轮采样一批轨迹 → 算后缀回报 G_t（γ=1）→ 标量 baseline b=mean(G_t)（可选 return
归一化）→ 一步梯度更新，loss = -mean( (G_t - b) · ln π_θ(a_t|s_t) )。b 以常数参与
（不沿 θ 求导，即 stop-gradient），与书 §21.5 一致。

用法（cwd 任意，脚本自解析 ../src 路径）：
    python 1_reinforce.py --env cartpole   # S2.4 冒烟：自定义倒立摆
    python 1_reinforce.py --env pendulum   # S2.3 外部基准：gym Pendulum-v1

环境说明：gym 0.26.2 与 numpy>=2 存在 np.bool8 兼容问题（numpy 2 移除该属性，而
gym 被动环境检查器仍引用）。仅在使用 gym.make（Pendulum-v1 等标准环境）触发 step
检查时需要，本文件顶部打一次补丁即可；自定义 CartPoleCustomEnv 不经 gym.make，不受影响。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

# gym 0.26.2 + numpy>=2 兼容补丁：必须在首次触发 env.step()（被动检查器）之前完成。
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import torch

# 可选进度条（tqdm 缺失时自动退化为简易文本进度）。
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

# 让本脚本从任意 cwd 都能 import ../src/env_cartpole.py（单一环境来源）
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rl_common import (  # noqa: E402
    PolicyNetwork,
    RunningMeanStd,
    compute_advantages,
    compute_returns,
    sample_trajectory,
)


def make_env(name, env_config=None):
    """构造环境。cartpole 复用 ../src/env_cartpole.py（单一环境、单一奖励）。

    `env_config` 为可选覆盖 dict（如 {"reward_coefs": {...}}），深拷贝 CONFIG 后更新，
    便于 Session 3 微调奖励系数而不改动 env 源文件。
    """
    if name == "cartpole":
        from env_cartpole import CONFIG, CartPoleCustomEnv
        cfg = json.loads(json.dumps(CONFIG))  # 浅深拷贝，避免污染全局 CONFIG
        if env_config:
            cfg.update(env_config)
        return CartPoleCustomEnv(cfg)
    if name == "pendulum":
        import gym
        return gym.make("Pendulum-v1")
    raise ValueError(f"unknown env: {name}")


def save_checkpoint(policy, optimizer, obs_rms, it, meta, path, numbered_path=None):
    """把当前策略 / 优化器 / 状态归一化统计量写成 .pth，供回顾、debug、存档与下游复现。

    - `path` 写最新一份（同名覆盖，不累积）；
    - `numbered_path`（可选）额外写一份带 iter 编号的文件（如 ckpt_cartpole_iter0400.pth），
      用于正式训练「每隔 N 轮落一份、训后取表现最好的一份」——避免好的中间权重被
      `latest` 覆盖（Session 5 踩过此坑）。`meta` 存超参与 seed，便于核对/恢复。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "obs_rms_mean": obs_rms.mean,
        "obs_rms_var": obs_rms.var,
        "obs_rms_count": obs_rms.count,
        "iter": it,
        "meta": meta,
    }
    torch.save(data, path)
    if numbered_path:
        os.makedirs(os.path.dirname(numbered_path), exist_ok=True)
        torch.save(data, numbered_path)


def run_reinforce(env, policy, optimizer, *, n_iters, episodes_per_iter, gamma,
                  normalize_returns, obs_rms, reset_seed, seed, env_name,
                  log_every=10, save_dir=None, ckpt_dir=None, ckpt_interval=0,
                  use_progress=True):
    """REINFORCE 训练主循环。返回 history（每轮一条统计）。

    - `ckpt_interval`>0 时每该轮数保存一次中间 .pth（覆盖式，不累积）到 `ckpt_dir`；
    - `use_progress` 时显示 tqdm 进度条（当前轮/总轮 + ETA + 关键统计量）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = next(policy.parameters()).device
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    meta = dict(env=env_name, seed=seed, n_iters=n_iters,
                episodes_per_iter=episodes_per_iter, gamma=gamma,
                normalize_returns=normalize_returns)

    history = []
    ep_counter = 0
    t_start = time.time()
    ckpt_path = (os.path.join(ckpt_dir, f"ckpt_{env_name}_latest.pth")
                 if ckpt_dir is not None else None)
    bar = (tqdm(total=n_iters, desc=f"REINFORCE[{env_name}]", unit="iter")
           if (use_progress and tqdm is not None) else None)
    for it in range(n_iters):
        # ---- 1. on-policy 采样一批轨迹 ----
        trajs = []
        for _ in range(episodes_per_iter):
            rs = (reset_seed + ep_counter) if reset_seed is not None else None
            traj = sample_trajectory(env, policy, obs_rms, reset_seed=rs)
            trajs.append(traj)
            if obs_rms is not None:
                obs_rms.update(traj["raw_states"])
            ep_counter += 1

        # ---- 2. 汇总 batch 的 G_t / A / ln π ----
        all_returns, all_logp_inputs = [], []
        ep_returns = []
        for traj in trajs:
            G = compute_returns(traj["rewards"], gamma)
            all_returns.append(G)
            ep_returns.append(float(G[0]))
            all_logp_inputs.append((traj["states"], traj["actions"]))

        rets = np.concatenate(all_returns)
        adv, baseline = compute_advantages(rets, normalize=normalize_returns)

        states = np.concatenate([s for s, _ in all_logp_inputs])
        actions = np.concatenate([a for _, a in all_logp_inputs])
        states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)

        # ---- 3. 梯度更新（baseline/advantage 为常数，stop-gradient）----
        log_prob = policy.log_prob(states_t, actions_t)
        loss = -(adv_t * log_prob).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- 4. 记录关键量（return / baseline / σ / 长度）----
        with torch.no_grad():
            _, log_std = policy(states_t)
            mean_log_std = float(log_std.mean().item())

        entry = dict(
            iter=it,
            ret_mean=float(np.mean(ep_returns)),
            ret_std=float(np.std(ep_returns)),
            ret_min=float(np.min(ep_returns)),
            ret_max=float(np.max(ep_returns)),
            baseline=float(baseline),
            mean_log_std=mean_log_std,
            ep_len_mean=float(np.mean([t["length"] for t in trajs])),
            loss=float(loss.item()),
        )
        # 末位置 x（仅 cartpole 有意义：state[0]=x）——用于中途看出「车是否在回中」，
        # 而不必等训练结束后再单独评估。带符号均值指示漂移方向，|x| 均值/最大给出量级。
        if env_name == "cartpole":
            fx = np.array([float(t["final_state"][0]) for t in trajs])
            entry["final_x_mean"] = float(np.mean(fx))
            entry["final_x_abs_mean"] = float(np.mean(np.abs(fx)))
            entry["final_x_abs_max"] = float(np.max(np.abs(fx)))
        history.append(entry)

        # ---- 5. 保存中间 checkpoint（latest 覆盖式 + iter 编号各一份，每 ckpt_interval 轮）+ 进度条 ----
        if ckpt_path is not None and ckpt_interval > 0 and (it % ckpt_interval == 0 or it == n_iters - 1):
            num_path = os.path.join(ckpt_dir, f"ckpt_{env_name}_iter{it:04d}.pth")
            save_checkpoint(policy, optimizer, obs_rms, it, meta, ckpt_path, num_path)
            print(f"[ckpt] iter {it:5d} -> {ckpt_path} (+{num_path})", flush=True)

        fx_str = (f" | fx|x| {entry.get('final_x_abs_mean', float('nan')):6.2f}"
                  + f" max {entry.get('final_x_abs_max', float('nan')):6.2f}") if env_name == "cartpole" else ""
        if bar is not None:
            bar.set_postfix(
                ret=f"{entry['ret_mean']:.1f}",
                ep_len=f"{entry['ep_len_mean']:.0f}",
                sigma=f"{np.exp(mean_log_std):.2f}",
                b=f"{entry['baseline']:.1f}",
                loss=f"{entry['loss']:.3f}",
                **({} if env_name != "cartpole" else
                   {"fx": f"{entry.get('final_x_abs_mean', float('nan')):.2f}"}),
            )
            bar.update(1)
        elif it % log_every == 0 or it == n_iters - 1:
            print(
                f"iter {it:5d} | ret {entry['ret_mean']:8.2f}±{entry['ret_std']:7.2f} "
                f"[{entry['ret_min']:8.2f}, {entry['ret_max']:8.2f}] | "
                f"b {entry['baseline']:8.2f} | σ {np.exp(mean_log_std):5.3f} "
                f"| len {entry['ep_len_mean']:7.1f} | loss {entry['loss']:9.4f}"
                f"{fx_str}",
                flush=True,
            )

    if bar is not None:
        bar.close()

    elapsed = time.time() - t_start
    print(f"[elapsed] {elapsed:.1f}s over {n_iters} iters "
          f"({elapsed / n_iters:.3f}s/iter) on {device}", flush=True)

    if save_dir is not None:
        save_history(history, save_dir, env_name)
    return history


def save_history(history, save_dir, env_name):
    """把训练历史写成 JSON + 回报曲线 PNG（供核对 / 后续成文引用）。"""
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"reward_history_reinforce_{env_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rets = [h["ret_mean"] for h in history]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rets, label="episode return (mean)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("return")
        ax.set_title(f"REINFORCE on {env_name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        png_path = os.path.join(save_dir, f"reward_history_reinforce_{env_name}.png")
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {json_path}")
        print(f"[save] {png_path}")
    except Exception as exc:  # 出图失败不阻断训练
        print(f"[warn] 出图失败（忽略）: {exc}")


def parse_hidden(s):
    return tuple(int(x) for x in s.split(","))


def _fix_neg_range_argv(argv):
    """argparse 不把 `-40,40` 当作 `--init-*-range` 的值（首个 `-` 被当作选项前缀）。

    把「空格分隔 + 负值」改写为 `--opt=value` 形式，使文档里的
    `--init-theta-range -40,40`（以及 `--init-x-range -1.5,1.5` 等）可直接使用。
    """
    names = {"--init-theta-range", "--init-x-range",
             "--init-xdot-range", "--init-thetadot-range"}
    out = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if (tok in names and i + 1 < len(argv)
                and argv[i + 1].startswith("-") and not argv[i + 1].startswith("--")):
            out.append(f"{tok}={argv[i + 1]}")
            i += 2
        else:
            out.append(tok)
            i += 1
    return out


def main():
    sys.argv[:] = _fix_neg_range_argv(sys.argv)
    p = argparse.ArgumentParser(description="纯 REINFORCE（连续动作、高斯策略）")
    p.add_argument("--env", choices=["cartpole", "pendulum"], default="cartpole")
    p.add_argument("--iters", type=int, default=None,
                   help="训练轮数（默认 cartpole=200, pendulum=600）")
    p.add_argument("--eps-per-iter", type=int, default=None,
                   help="每轮采样轨迹数（默认 cartpole=16, pendulum=32）")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=parse_hidden, default=(64, 64))
    p.add_argument("--init-log-std", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--no-normalize", action="store_true",
                   help="关闭 return 归一化（只保留标量 baseline 减法）")
    p.add_argument("--reward-coefs", type=str, default="0.2,0.05,0.1,0.01",
                   help="存活型奖励系数 c_theta,c_theta_dot,c_x,c_u（cartpole 专用；"
                        "可选第 5 个数为 c_term 终端位置惩罚系数，缺省不启用。"
                        "改动后需同步回写 env_cartpole.py 与书 eq:cartpole-1-reward）")
    p.add_argument("--init-theta-range", type=str, default=None,
                   help="覆盖初始 θ 采样范围（°），如 -40,40（cartpole 专用；默认用 "
                        "CONFIG['init_theta_range']=±10°。用于 Session 6 拓宽覆盖 L1–L3）")
    p.add_argument("--init-x-range", type=str, default=None,
                   help="覆盖初始 x 采样范围（m），如 -1.5,1.5（cartpole 专用；默认用 "
                        "CONFIG['init_x_range']=±0.5 m。拓宽覆盖 L3）")
    p.add_argument("--init-xdot-range", type=str, default=None,
                   help="覆盖初始 ẋ 采样范围（m/s），如 -1.0,1.0（cartpole 专用；默认用 "
                        "CONFIG['init_dot_x_range']=±0.3 m/s。拓宽覆盖 L3）")
    p.add_argument("--init-thetadot-range", type=str, default=None,
                   help="覆盖初始 θ̇ 采样范围（°/s），如 -10,10（cartpole 专用；默认用 "
                        "CONFIG['init_dot_theta_range']=±1°/s。拓宽覆盖 L3）")
    p.add_argument("--max-steps", type=int, default=None,
                   help="覆盖环境单回合步数上限（cartpole 专用；小规模对比测试可缩短，"
                        "如 200，便于快速看出趋势；正式训练不传用默认 1000）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--no-save", action="store_true", help="不写 JSON/PNG 产物")
    p.add_argument("--no-progress", action="store_true",
                   help="关闭 tqdm 进度条（回到逐行文本日志）")
    p.add_argument("--ckpt-interval", type=int, default=0,
                   help="每该轮数保存一次中间 .pth 到 --ckpt-dir（0=关闭；"
                        "如 200 表示每 200 轮落一份：latest 覆盖式 + iter 编号各一份，"
                        "便于训后取表现最好的一份）")
    p.add_argument("--ckpt-dir", default=None,
                   help="checkpoint 目录（默认 <脚本目录>/res/checkpoints/<env>）")
    # 本工作负载（tiny MLP + 每环境步 batch=1 采样）在 MX450 上是传输瓶颈，GPU 反而
    # 慢 ~14×（见 2026-09 实测基准），故默认 CPU；--device cuda 可显式强制 GPU。
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu",
                   help="训练设备（默认 cpu；auto=有 CUDA 则 GPU 否则 CPU；"
                        "cuda=强制 GPU。本负载在 MX450 上 GPU 更慢）")
    p.add_argument("--save-dir", default=None,
                   help="产物目录（默认 <本脚本目录>/res）")
    args = p.parse_args()

    # 每环境默认超参（S2 冒烟/基准用；S3 调参可覆盖）
    if args.iters is None:
        args.iters = 200 if args.env == "cartpole" else 600
    if args.eps_per_iter is None:
        args.eps_per_iter = 16 if args.env == "cartpole" else 32

    # 固定随机种子：必须在构造 policy 之前 seed，否则 PolicyNetwork 的随机初始化不可复现，
    # 同 seed 下每次运行策略初值都不同（这会在 REINFORCE 高方差之上再叠加一层初始化方差，
    # 使「固定 seed 以复现」失效）。run_reinforce 内部会再次 seed（冗余但无碍）。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 构造 env（cartpole：支持用 --reward-coefs / --max-steps 临时覆盖，便于 S3 调参对照）
    env_config = None
    if args.env == "cartpole":
        env_config = {}
        coefs = [float(x) for x in args.reward_coefs.split(",")]
        assert len(coefs) in (4, 5), "--reward-coefs 需给 4 个数（或 5 个数含 c_term）"
        rc = dict(c_theta=coefs[0], c_theta_dot=coefs[1], c_x=coefs[2], c_u=coefs[3])
        if len(coefs) == 5:
            rc["c_term"] = coefs[4]           # 可选终端位置惩罚系数（>0 时启用）
        env_config["reward_coefs"] = rc
        if args.max_steps is not None:
            env_config["max_steps"] = args.max_steps
        if args.init_theta_range is not None:
            lo, hi = [float(v) for v in args.init_theta_range.split(",")]
            assert hi > lo, "--init-theta-range 需 hi > lo"
            env_config["init_theta_range"] = [lo * np.pi / 180.0, hi * np.pi / 180.0]
        if args.init_x_range is not None:
            lo, hi = [float(v) for v in args.init_x_range.split(",")]
            assert hi > lo, "--init-x-range 需 hi > lo"
            env_config["init_x_range"] = [lo, hi]
        if args.init_xdot_range is not None:
            lo, hi = [float(v) for v in args.init_xdot_range.split(",")]
            assert hi > lo, "--init-xdot-range 需 hi > lo"
            env_config["init_dot_x_range"] = [lo, hi]
        if args.init_thetadot_range is not None:
            lo, hi = [float(v) for v in args.init_thetadot_range.split(",")]
            assert hi > lo, "--init-thetadot-range 需 hi > lo"
            env_config["init_dot_theta_range"] = [lo * np.pi / 180.0, hi * np.pi / 180.0]
    env = make_env(args.env, env_config)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    policy = PolicyNetwork(obs_dim, act_dim, hidden_sizes=args.hidden,
                           init_log_std=args.init_log_std).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    obs_rms = RunningMeanStd(shape=env.observation_space.shape)

    # 自定义 CartPoleCustomEnv.reset(seed=...) 会覆写全局 np.random，故不传 seed；
    # 标准 gym 环境（Pendulum-v1）传 per-episode seed 以复现。
    reset_seed = args.seed if args.env == "pendulum" else None

    save_dir = args.save_dir
    if save_dir is None and not args.no_save:
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "res"))
    if args.no_save:
        save_dir = None

    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None and args.ckpt_interval > 0:
        ckpt_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "res", "checkpoints", args.env))

    dev_name = (f"cuda ({torch.cuda.get_device_name(0)})"
                if device.type == "cuda" else "cpu")
    print(f"[env] {args.env} | obs_dim={obs_dim} act_dim={act_dim} | "
          f"iters={args.iters} eps/iter={args.eps_per_iter} lr={args.lr} "
          f"init_log_std={args.init_log_std} normalize={not args.no_normalize} "
          f"seed={args.seed} | device={dev_name}")
    if args.env == "cartpole":
        ir = env.config
        def _pr(lo, hi):
            return f"[{lo:+.2f},{hi:+.2f}]"
        print(f"[init-dist] x {_pr(*ir['init_x_range'])} m | "
              f"ẋ {_pr(*ir['init_dot_x_range'])} m/s | "
              f"θ {_pr(ir['init_theta_range'][0]*180/np.pi, ir['init_theta_range'][1]*180/np.pi)}° | "
              f"θ̇ {_pr(ir['init_dot_theta_range'][0]*180/np.pi, ir['init_dot_theta_range'][1]*180/np.pi)}°/s")

    run_reinforce(
        env, policy, optimizer,
        n_iters=args.iters,
        episodes_per_iter=args.eps_per_iter,
        gamma=args.gamma,
        normalize_returns=not args.no_normalize,
        obs_rms=obs_rms,
        reset_seed=reset_seed,
        seed=args.seed,
        env_name=args.env,
        log_every=args.log_every,
        save_dir=save_dir,
        ckpt_dir=ckpt_dir,
        ckpt_interval=args.ckpt_interval,
        use_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
