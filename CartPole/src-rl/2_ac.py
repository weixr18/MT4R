# -*- coding: utf-8 -*-
"""Actor-Critic（AC）主循环（连续动作、高斯策略 + 学习值函数 V(s)）。

对应书：
- `chap6_4_ppo_1.tex` §22.2 原始 Actor-Critic 算法（algo:pg_ac）；
- `chap6_4_ppo_1.tex` §22.3 Actor-Critic GAE（algo:pg_ac_gae）。

与 `1_reinforce.py`（纯 REINFORCE）的关系：AC 与 REINFORCE 用**同一套**高斯策略
（`PolicyNetwork`，μ/logσ），**唯一差别**是 AC 额外引入一个**学习值函数 V(s)**
（`ValueNetwork`，critic），并把优势从「标量 baseline b=mean(G_t)」换成：
- 无 GAE（`--adv td`）：单步 TD 优势 `δ_t = r_t + γ·V(s_{t+1}) − V(s_t)`；
- 有 GAE（`--adv gae`）：λ 加权优势 `A_t = Σ_j (γλ)^j δ_{t+j}`（`rl_common.compute_gae`）。

**关键口径（见 AC-training-plan.md「注意点 1」）**：critic 目标与非 GAE 优势共用同一个自洽
δ_t（`rewards[t]` = 从 s_t 到 s_{t+1} 的回报），避免书里 `algo:pg_ac` 对同一 t 用两个不同
回报索引的问题。末态 `V(s_T)` 置 0（本环境终止 = 越界或满步，truncated 恒 False，均视为真终态、
不 bootstrap）。

`log_prob` 与 REINFORCE 相同：一律在**未裁剪动作**上计算（防 σ 崩塌）。

用法（cwd 任意，脚本自解析 ../src 路径）：
    python 2_ac.py --env cartpole --adv td --max-steps 200 --iters 120 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
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
    ValueNetwork,
    compute_gae,
    compute_returns,
    sample_trajectory,
    save_ac_checkpoint,
    set_seed,
)


def make_env(name, env_config=None):
    """构造环境。cartpole 复用 ../src/env_cartpole.py（单一环境、单一奖励）。"""
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


def _per_traj_adv_target(traj, value_net, obs_rms, gamma, lam, adv_mode, device):
    """单条轨迹：计算优势（detached）与 critic 目标（detached）。

    返回 (adv, target, returns_go)：
    - adv：长度 T 的 numpy 数组，TD 优势 δ_t 或 GAE 优势 A_t（供 actor 用，stop-gradient）；
    - target：长度 T 的 numpy 数组，`r_t + γ·V(s_{t+1})`（critic 的 TD 目标，V(s_T)=0）；
    - returns_go：长度 T 的 numpy 数组，前缀回报 G_t（仅用于记录「V(s) 对齐度」诊断）。
    """
    s = np.asarray(traj["states"], dtype=np.float32)          # 归一化状态 (T,dim)
    r = np.asarray(traj["rewards"], dtype=np.float64)          # (T,)
    # 相邻状态 next_states = [s_1, ..., s_{T-1}, s_T]，末态 s_T 由 final_state 归一化得到，
    # 长度恰为 T；末态 V(s_T) 置 0（真终态、不 bootstrap）。
    ns = np.vstack([s[1:], obs_rms.normalize(traj["final_state"])[None, :]])
    states_t = torch.as_tensor(s, dtype=torch.float32, device=device)
    next_states_t = torch.as_tensor(ns, dtype=torch.float32, device=device)
    with torch.no_grad():
        v = value_net(states_t)                                # (T,)
        nv = value_net(next_states_t)                          # (T,)
    nv = nv.cpu().numpy().copy()
    nv[-1] = 0.0                                               # V(s_T)=0（不 bootstrap）
    v = v.cpu().numpy()

    if adv_mode == "gae":
        adv = compute_gae(r, v, nv, gamma=gamma, lam=lam)
    else:
        adv = r + gamma * nv - v                              # 单步 TD 优势 δ_t

    target = r + gamma * nv                                    # critic TD 目标
    returns_go = compute_returns(r, gamma)
    return adv, target, returns_go


def run_ac(env, policy, value_net, actor_optimizer, critic_optimizer, *,
           n_iters, episodes_per_iter, gamma, adv_mode, lam, value_coef,
           normalize_adv, obs_rms, reset_seed, seed, env_name,
           log_every=10, save_dir=None, ckpt_dir=None, ckpt_interval=0,
           use_progress=True):
    """AC 训练主循环。返回 history（每轮一条统计）。

    - 每轮：采样一批轨迹 → 前向 value_net 得 V(s_t)/V(s_{t+1}) → 算 TD/GAE 优势 →
      critic（TD 目标，`V(s_{t+1})` stop-gradient）→ actor（优势 stop-gradient）两步更新。
    - `--adv td` 时 `lam` 不生效；`--adv gae` 时 `lam` 为 GAE λ。
    - `normalize_adv` 为 True 时对 batch 优势做中心化 + 除标准差（稳定训练，同 REINFORCE
      的 return 归一化惯例），即 `--no-normalize` 关闭它。
    """
    set_seed(seed)                                            # 双网络已构造，冗余但无碍
    device = next(policy.parameters()).device

    meta = dict(env=env_name, seed=seed, n_iters=n_iters,
                episodes_per_iter=episodes_per_iter, gamma=gamma,
                adv=adv_mode, lam=lam, value_coef=value_coef,
                normalize_adv=normalize_adv,
                actor_lr=float(actor_optimizer.param_groups[0]["lr"]),
                critic_lr=float(critic_optimizer.param_groups[0]["lr"]))

    history = []
    ep_counter = 0
    t_start = time.time()
    ckpt_path = (os.path.join(ckpt_dir, f"ckpt_{env_name}_ac_latest.pth")
                 if ckpt_dir is not None else None)
    bar = (tqdm(total=n_iters, desc=f"AC[{env_name}:{adv_mode}]", unit="iter")
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

        # ---- 2. 逐轨迹算优势 / critic 目标，并汇聚成 batch ----
        all_states, all_actions, all_adv, all_target, all_returns_go = [], [], [], [], []
        ep_returns = []
        for traj in trajs:
            adv, target, returns_go = _per_traj_adv_target(
                traj, value_net, obs_rms, gamma, lam, adv_mode, device)
            all_states.append(traj["states"])
            all_actions.append(traj["actions"])
            all_adv.append(adv)
            all_target.append(target)
            all_returns_go.append(returns_go)
            ep_returns.append(float(returns_go[0]))            # 每轨迹末段回报 G_0

        states = np.concatenate(all_states)
        actions = np.concatenate(all_actions)
        adv = np.concatenate(all_adv)
        target = np.concatenate(all_target)
        returns_go = np.concatenate(all_returns_go)

        # 优势归一化（可选：中心化 + 除以标准差，stop-gradient）
        if normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)      # detached
        target_t = torch.as_tensor(target, dtype=torch.float32, device=device)  # detached

        # ---- 3. 更新 actor：loss_a = -mean(adv · ln π)，adv stop-gradient ----
        log_prob = policy.log_prob(states_t, actions_t)        # 在未裁剪动作上算（防 σ 崩塌）
        actor_loss = -(adv_t * log_prob).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # ---- 4. 更新 critic：TD 目标，V(s_{t+1}) stop-gradient ----
        # 重新前向 value_net 得 V(s_t)（带梯度），与 detached 目标比较。
        values = value_net(states_t)
        critic_loss = value_coef * ((values - target_t) ** 2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # ---- 5. 记录关键量 ----
        with torch.no_grad():
            _, log_std = policy(states_t)
            mean_log_std = float(log_std.mean().item())
            v_pred_all = value_net(states_t).cpu().numpy()
        v_corr = float(np.corrcoef(v_pred_all, returns_go)[0, 1]) \
            if (len(v_pred_all) > 1 and np.std(v_pred_all) > 1e-8 and np.std(returns_go) > 1e-8) \
            else float("nan")

        entry = dict(
            iter=it,
            ret_mean=float(np.mean(ep_returns)),
            ret_std=float(np.std(ep_returns)),
            ep_len_mean=float(np.mean([t["length"] for t in trajs])),
            mean_log_std=mean_log_std,
            loss_a=float(actor_loss.item()),
            loss_v=float(critic_loss.item()),
            v_pred_mean=float(np.mean(v_pred_all)),
            v_corr=v_corr,
        )
        # 末位置 x（仅 cartpole 有意义：state[0]=x）——用于中途看出「车是否在回中」。
        if env_name == "cartpole":
            fx = np.array([float(t["final_state"][0]) for t in trajs])
            entry["final_x_abs_mean"] = float(np.mean(np.abs(fx)))
            entry["final_x_abs_max"] = float(np.max(np.abs(fx)))
        history.append(entry)

        # ---- 6. 保存中间 checkpoint（latest 覆盖式 + iter 编号各一份，每 ckpt_interval 轮）----
        if ckpt_path is not None and ckpt_interval > 0 and (it % ckpt_interval == 0 or it == n_iters - 1):
            num_path = os.path.join(ckpt_dir, f"ckpt_{env_name}_ac_iter{it:04d}.pth")
            save_ac_checkpoint(policy, actor_optimizer, value_net, critic_optimizer,
                               obs_rms, it, meta, ckpt_path, num_path)
            print(f"[ckpt] iter {it:5d} -> {ckpt_path} (+{num_path})", flush=True)

        fx_str = (f" | fx|x| {entry.get('final_x_abs_mean', float('nan')):6.2f}"
                  + f" max {entry.get('final_x_abs_max', float('nan')):6.2f}") \
            if env_name == "cartpole" else ""
        if bar is not None:
            bar.set_postfix(
                ret=f"{entry['ret_mean']:.1f}",
                ep_len=f"{entry['ep_len_mean']:.0f}",
                sigma=f"{np.exp(mean_log_std):.2f}",
                la=f"{entry['loss_a']:.3f}",
                lv=f"{entry['loss_v']:.2f}",
                vc=f"{entry['v_corr']:.2f}",
                **({} if env_name != "cartpole" else
                   {"fx": f"{entry.get('final_x_abs_mean', float('nan')):.2f}"}),
            )
            bar.update(1)
        elif it % log_every == 0 or it == n_iters - 1:
            print(
                f"iter {it:5d} | ret {entry['ret_mean']:8.2f}±{entry['ret_std']:7.2f} "
                f"| σ {np.exp(mean_log_std):5.3f} | loss_a {entry['loss_a']:9.4f} "
                f"| loss_v {entry['loss_v']:8.3f} | v_corr {entry['v_corr']:7.4f} "
                f"| v_pred {entry['v_pred_mean']:8.2f} | len {entry['ep_len_mean']:7.1f}"
                f"{fx_str}", flush=True,
            )

    if bar is not None:
        bar.close()

    elapsed = time.time() - t_start
    print(f"[elapsed] {elapsed:.1f}s over {n_iters} iters "
          f"({elapsed / n_iters:.3f}s/iter) on {device}", flush=True)

    if save_dir is not None:
        save_history(history, save_dir, env_name, adv_mode)
    return history


def save_history(history, save_dir, env_name, adv_mode):
    """把训练历史写成 JSON（供核对 / 后续成文引用）。"""
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"reward_history_ac_{adv_mode}_{env_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[save] {json_path}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rets = [h["ret_mean"] for h in history]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rets, label="episode return (mean)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("return")
        ax.set_title(f"AC ({adv_mode}) on {env_name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        png_path = os.path.join(save_dir, f"reward_history_ac_{adv_mode}_{env_name}.png")
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {png_path}")
    except Exception as exc:  # 出图失败不阻断训练
        print(f"[warn] 出图失败（忽略）: {exc}")


def parse_hidden(s):
    return tuple(int(x) for x in s.split(","))


def _fix_neg_range_argv(argv):
    """argparse 不把 `-40,40` 当作 `--init-*-range` 的值（首个 `-` 被当作选项前缀）。

    把「空格分隔 + 负值」改写为 `--opt=value` 形式，使文档里的
    `--init-theta-range -40,40`（以及 `--init-x-range -1.5,1.5` 等）可直接使用；
    `--opt=value` 形式原样保留。
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
    ap = argparse.ArgumentParser(description="Actor-Critic（连续动作、高斯策略 + 学习值函数 V(s)）")
    ap.add_argument("--env", choices=["cartpole", "pendulum"], default="cartpole")
    ap.add_argument("--iters", type=int, default=None,
                    help="训练轮数（默认 cartpole=200, pendulum=600）")
    ap.add_argument("--eps-per-iter", type=int, default=None,
                    help="每轮采样轨迹数（默认 cartpole=16, pendulum=32）")
    ap.add_argument("--adv", choices=["td", "gae"], default="td",
                    help="优势估计方式：td=单步 TD 优势 δ_t；gae=GAE λ 加权优势")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.95,
                    help="GAE λ（仅 --adv gae 生效；λ=0 退化为 TD 优势）")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="actor（策略）学习率（与 1_reinforce.py 同义）")
    ap.add_argument("--actor-lr", type=float, default=None,
                    help="actor（策略）学习率；默认取 --lr")
    ap.add_argument("--critic-lr", type=float, default=1e-3,
                    help="critic（值函数）学习率")
    ap.add_argument("--value-coef", type=float, default=1.0,
                    help="critic 损失系数（loss_v = value_coef * MSE(V(s_t), TD target)）")
    ap.add_argument("--hidden", type=parse_hidden, default=(64, 64))
    ap.add_argument("--init-log-std", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--no-normalize", action="store_true",
                    help="关闭优势归一化（只保留 TD/GAE 优势本身）")
    ap.add_argument("--reward-coefs", type=str, default="0.2,0.05,1.0,0.01",
                    help="存活型奖励系数 c_theta,c_theta_dot,c_x,c_u[,c_term]（cartpole 专用；"
                         "默认 4 个数、无 c_term，终止步 r=0，与书 eq:cartpole-1-reward 一致；"
                         "若给 5 个数即含 c_term 终端位置惩罚，与书 §28.1 口径不符，务必只在探索时使用）")
    ap.add_argument("--init-theta-range", type=str, default=None,
                    help="覆盖初始 θ 采样范围（°），如 -40,40（cartpole 专用；默认用 "
                         "CONFIG['init_theta_range']=±10°）")
    ap.add_argument("--init-x-range", type=str, default=None,
                    help="覆盖初始 x 采样范围（m），如 -1.5,1.5（cartpole 专用；默认用 "
                         "CONFIG['init_x_range']=±0.5 m）")
    ap.add_argument("--init-xdot-range", type=str, default=None,
                    help="覆盖初始 ẋ 采样范围（m/s），如 -1.0,1.0（cartpole 专用；默认用 "
                         "CONFIG['init_dot_x_range']=±0.3 m/s）")
    ap.add_argument("--init-thetadot-range", type=str, default=None,
                    help="覆盖初始 θ̇ 采样范围（°/s），如 -10,10（cartpole 专用；默认用 "
                         "CONFIG['init_dot_theta_range']=±1°/s）")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="覆盖环境单回合步数上限（cartpole 专用；小规模对比测试可缩短，"
                         "如 200，便于快速看出趋势；正式训练不带用环境默认）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--no-save", action="store_true", help="不写 JSON 产物")
    ap.add_argument("--no-progress", action="store_true",
                    help="关闭 tqdm 进度条（回到逐行文本日志）")
    ap.add_argument("--ckpt-interval", type=int, default=0,
                    help="每该轮数保存一次中间 .pth 到 --ckpt-dir（0=关闭）")
    ap.add_argument("--ckpt-dir", default=None,
                    help="checkpoint 目录（默认 <脚本目录>/res/checkpoints/<env>）")
    # 本负载（tiny MLP + batch=1 采样）在 MX450 上是传输瓶颈，GPU 反而慢，故默认 CPU。
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu",
                    help="训练设备（默认 cpu；auto=有 CUDA 则 GPU 否则 CPU）")
    ap.add_argument("--save-dir", default=None,
                    help="产物目录（默认 <本脚本目录>/res）")
    args = ap.parse_args()

    # 每环境默认超参（与 1_reinforce.py 对齐）
    if args.iters is None:
        args.iters = 200 if args.env == "cartpole" else 600
    if args.eps_per_iter is None:
        args.eps_per_iter = 16 if args.env == "cartpole" else 32
    if args.actor_lr is None:
        args.actor_lr = args.lr

    # 固定随机种子：必须在构造 policy 与 value_net 两个网络之前 seed（否则随机初始化不可复现）。
    set_seed(args.seed)

    # 构造 env（cartpole：支持用 --reward-coefs / --max-steps 临时覆盖，便于调参对照）
    env_config = None
    if args.env == "cartpole":
        env_config = {}
        coefs = [float(x) for x in args.reward_coefs.split(",")]
        assert len(coefs) in (4, 5), "--reward-coefs 需给 4 个数（或 5 个数含 c_term）"
        rc = dict(c_theta=coefs[0], c_theta_dot=coefs[1], c_x=coefs[2], c_u=coefs[3])
        if len(coefs) == 5:
            rc["c_term"] = coefs[4]
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

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else "cpu" if args.device in ("auto", "cpu") else "cuda")

    policy = PolicyNetwork(obs_dim, act_dim, hidden_sizes=args.hidden,
                           init_log_std=args.init_log_std).to(device)
    value_net = ValueNetwork(obs_dim, hidden_sizes=args.hidden).to(device)
    actor_optimizer = torch.optim.Adam(policy.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.critic_lr)
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
          f"iters={args.iters} eps/iter={args.eps_per_iter} "
          f"adv={args.adv} lam={args.lam} "
          f"actor_lr={args.actor_lr} critic_lr={args.critic_lr} "
          f"value_coef={args.value_coef} normalize={not args.no_normalize} "
          f"init_log_std={args.init_log_std} seed={args.seed} | device={dev_name}")
    if args.env == "cartpole":
        ir = env.config
        def _pr(name, lo, hi):
            return f"{name}=[{lo:+.2f},{hi:+.2f}]"
        print(f"[init-dist] x {_pr('m', *ir['init_x_range'])} | "
              f"ẋ {_pr('m/s', *ir['init_dot_x_range'])} | "
              f"θ {_pr('°', ir['init_theta_range'][0]*180/np.pi, ir['init_theta_range'][1]*180/np.pi)} | "
              f"θ̇ {_pr('°/s', ir['init_dot_theta_range'][0]*180/np.pi, ir['init_dot_theta_range'][1]*180/np.pi)}")

    run_ac(
        env, policy, value_net, actor_optimizer, critic_optimizer,
        n_iters=args.iters,
        episodes_per_iter=args.eps_per_iter,
        gamma=args.gamma,
        adv_mode=args.adv,
        lam=args.lam,
        value_coef=args.value_coef,
        normalize_adv=not args.no_normalize,
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
