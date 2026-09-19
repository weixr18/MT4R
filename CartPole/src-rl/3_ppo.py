# -*- coding: utf-8 -*-
"""PPO 主循环（连续动作、高斯策略 + 学习值函数 V(s)），支持三变体一键切换。

对应书（`chap6_4_ppo_2.tex` §22.4，三种 PPO 变体；理论沿用 `chap6_4_ppo_1.tex` §22.2/§22.3）：
- `algo:pg_ppo_og`    原始 PPO（固定 KL 惩罚系数 β）    -> `--algo og`
- `algo:pg_ppo_pen`   PPO-Penalty（自适应 β 滞回）      -> `--algo pen`
- `algo:pg_ppo_clip`  PPO-Clip（重要性裁剪 ε）          -> `--algo clip`

与 `2_ac.py`（AC 主循环）的关系：三者共用**同一套**采样（`sample_trajectory`）+ GAE 优势
（`rl_common.compute_gae` 的自洽单 δ 形式）+ critic（`ValueNetwork`）+ 高斯策略
（`PolicyNetwork`，μ/状态无关 logσ）。PPO 在此基础上加：
  ① 批内多轮更新（`--n-epochs-policy` 轮 actor / `--n-epochs-value` 轮 critic，B_θ / B_ω）；
  ② 重要性采样 `r_IS = π_θ(a|s) / π_θ'(a|s)`（`log_prob_old` 在采样时用 θ' 算好并冻结）；
  ③ 更新限幅（og: β·KL 惩罚 / pen: 自适应 β / clip: `min(r_IS·A, clip(r_IS)·A)`）。

**关键口径（见 PPO-training-plan.md「注意点 1」）**：critic 目标与非 GAE 优势共用同一个自洽
δ_t（`rewards[t]` = 从 s_t 到 s_{t+1} 的回报），避免书里 `algo:pg_ppo_*` 对同一 t 用两个不同
回报索引的问题。末态 `V(s_T)` 置 0（本环境终止 = 越界或满步，truncated 恒 False，均视为真终态、
不 bootstrap）。

**minibatch 采样方式（注意点 2）**：书里写「Randomly sample N_1 Trajectories from K」即按轨迹抽，
但标准 PPO 实现常把全部 `(s,a,adv,target,log_prob_old)` 元组打乱后**按元组抽子批**。本实现采用
**元组级**（flat minibatch），更为简便高效、且与 AC 的 tuple 级处理一致；写书文本时注明用的是这一种。

**r_IS 与 KL（注意点 4/5）**：
- `r_IS = exp(log_prob_new - log_prob_old)`（用 log_prob 差值而非除法，避免数值不稳）；
- `log_prob_old` 在**采样时**用 θ' 算好并冻结；`log_prob_new` 为当前 θ 每内层轮算；
- `log_prob` 一律在**未裁剪动作**上算（防 σ 崩塌，同 REINFORCE/AC）；
- KL 用 k3 估计器 `KL̂ = mean(r_IS − 1 − ln r_IS)`（书 `chap6_4_ppo_1.tex`「KL散度近似计算」）；
- 数值安全：`log_ratio` 夹到 [−max_ratio_log, +max_ratio_log]（默认 ±20）防止 exp 溢出为 inf/NaN，
  并监控 `mean_r_IS`/`mean_KL`。

**书 `algo:pg_ppo_clip` 的残留项（注意点 3）**：其框首写了 `β ← β_0` 但后续从未使用 β（Clip 只用 ε）。
本实现忽略该残留，不影响 Clip 正确性。

用法（cwd 任意，脚本自解析 ../src 路径）：
    python 3_ppo.py --env cartpole --algo og --max-steps 200 --iters 120 --seed 0
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


def _per_traj_adv_target(traj, policy, value_net, obs_rms, gamma, lam, adv_mode, device):
    """单条轨迹：计算优势 / critic 目标 / 前缀回报 / 采样时 log_prob_old（全部 detached）。

    返回 (adv, target, returns_go, log_prob_old)：
    - adv：长度 T 的 numpy 数组，TD 优势 δ_t 或 GAE 优势 A_t（供 actor 用，stop-gradient）；
    - target：长度 T 的 numpy 数组，`r_t + γ·V(s_{t+1})`（critic 的 TD 目标，V(s_T)=0）；
    - returns_go：长度 T 的 numpy 数组，前缀回报 G_t（仅用于记录「V(s) 对齐度」诊断）；
    - log_prob_old：长度 T 的 numpy 数组，`ln π_θ'(a_t|s_t)`（θ' = 采样时的策略，冻结，供 r_IS）。
    """
    s = np.asarray(traj["states"], dtype=np.float32)          # 归一化状态 (T,dim)
    a = np.asarray(traj["actions"], dtype=np.float32)          # 未裁剪动作 (T,act)
    r = np.asarray(traj["rewards"], dtype=np.float64)          # (T,)
    # 相邻状态 next_states = [s_1, ..., s_{T-1}, s_T]，末态 s_T 由 final_state 归一化得到，
    # 长度恰为 T；末态 V(s_T) 置 0（真终态、不 bootstrap）。
    ns = np.vstack([s[1:], obs_rms.normalize(traj["final_state"])[None, :]])
    states_t = torch.as_tensor(s, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(a, dtype=torch.float32, device=device)
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

    # PPO 需要：采样时用 θ' 在 (s,a) 上算 log_prob 并冻结（供 r_IS = exp(lp_new − lp_old)）。
    with torch.no_grad():
        log_prob_old = policy.log_prob(states_t, actions_t)   # (T,)
    log_prob_old = log_prob_old.cpu().numpy().copy()
    return adv, target, returns_go, log_prob_old


def _is_ratio_safe(log_ratio, max_ratio_log):
    """log_ratio 落到数值安全区间，防止 exp 溢出为 inf / 下溢为 0。"""
    return torch.clamp(log_ratio, min=-max_ratio_log, max=max_ratio_log)


def run_ppo(env, policy, value_net, actor_optimizer, critic_optimizer, *,
            n_iters, episodes_per_iter, gamma, lam, adv_mode, value_coef,
            normalize_adv, obs_rms, reset_seed, seed, env_name,
            algo, n_epochs_policy, n_epochs_value, minibatch,
            kl_beta, kl_beta0, kl_min, kl_max, kl_delta, clip_eps,
            grad_clip, max_ratio_log,
            log_every=10, save_dir=None, ckpt_dir=None, ckpt_interval=0,
            use_progress=True):
    """PPO 训练主循环。返回 history（每轮一条统计）。

    每轮（外层迭代 m）：
    1. on-policy 采样 K 条轨迹（用当前策略 θ'=θ），更新 obs_rms；
    2. 逐轨迹：算 GAE/TD 优势、critic 目标、前缀回报、log_prob_old（冻结）；
    3. 汇聚成 batch，可选优势归一化；
    4. 内层 actor 多轮更新（B_θ = n_epochs_policy），每轮抽小子批（元组级）：
       r_IS = exp(lp_new − lp_old)，按 --algo 用 KL 惩罚 / 自适应 β / 裁剪 得到 loss_a = −J_PPO；
    5. 内层 critic 多轮更新（B_ω = n_epochs_value），loss_v = value_coef·MSE(V(s_t), target)；
    6. 记录收益/回合长/σ/loss/v_corr/mean_r_IS/mean_KL/末位置。

    - β（og）或 β_0（pen）用 `kl_beta` / `kl_beta0`；pen 的 β 自适应发生在内层 actor 循环内。
    - `minibatch<=0` 表示用全 batch（单子批、每 epoch 一次全量更新）。
    """
    set_seed(seed)                                            # 双网络已构造，冗余但无碍
    device = next(policy.parameters()).device
    n_minibatch = int(minibatch) if minibatch > 0 else 0

    # 记录 β 需要可变状态：pen 时用 clipper（闭包），og 用固定 kl_beta。
    # 统一以 beta 状态跟踪，便于记录与复盘。
    beta = float(kl_beta0 if algo == "pen" else kl_beta)

    meta = dict(env=env_name, seed=seed, n_iters=n_iters,
                episodes_per_iter=episodes_per_iter, gamma=gamma, lam=lam,
                adv=adv_mode, algo=algo, value_coef=value_coef,
                normalize_adv=normalize_adv,
                n_epochs_policy=n_epochs_policy, n_epochs_value=n_epochs_value,
                minibatch=minibatch,
                kl_beta=kl_beta, kl_beta0=kl_beta0, kl_min=kl_min, kl_max=kl_max,
                kl_delta=kl_delta, clip_eps=clip_eps, grad_clip=grad_clip,
                actor_lr=float(actor_optimizer.param_groups[0]["lr"]),
                critic_lr=float(critic_optimizer.param_groups[0]["lr"]))

    history = []
    ep_counter = 0
    t_start = time.time()
    ckpt_path = (os.path.join(ckpt_dir, f"ckpt_{env_name}_ppo_latest.pth")
                 if ckpt_dir is not None else None)
    bar = (tqdm(total=n_iters, desc=f"PPO[{env_name}:{algo}]", unit="iter")
           if (use_progress and tqdm is not None) else None)

    # 记录「最后一轮平均 r_IS / KL」供诊断。
    last_mean_r_is = float("nan")
    last_mean_kl = float("nan")
    last_loss_a = float("nan")

    for it in range(n_iters):
        # ---- 1. on-policy 采样一批轨迹（θ' = 当前 θ）----
        trajs = []
        for _ in range(episodes_per_iter):
            rs = (reset_seed + ep_counter) if reset_seed is not None else None
            traj = sample_trajectory(env, policy, obs_rms, reset_seed=rs)
            trajs.append(traj)
            if obs_rms is not None:
                obs_rms.update(traj["raw_states"])
            ep_counter += 1

        # ---- 2. 逐轨迹算优势 / critic 目标 / log_prob_old，并汇聚成 batch ----
        all_states, all_actions = [], []
        all_adv, all_target, all_returns_go, all_lp_old = [], [], [], []
        ep_returns = []
        for traj in trajs:
            adv, target, returns_go, lp_old = _per_traj_adv_target(
                traj, policy, value_net, obs_rms, gamma, lam, adv_mode, device)
            all_states.append(traj["states"])
            all_actions.append(traj["actions"])
            all_adv.append(adv)
            all_target.append(target)
            all_returns_go.append(returns_go)
            all_lp_old.append(lp_old)
            ep_returns.append(float(returns_go[0]))            # 每轨迹末段回报 G_0

        states = np.concatenate(all_states)
        actions = np.concatenate(all_actions)
        adv = np.concatenate(all_adv)
        target = np.concatenate(all_target)
        returns_go = np.concatenate(all_returns_go)
        log_prob_old = np.concatenate(all_lp_old)

        # 优势归一化（可选：中心化 + 除以标准差，stop-gradient）
        if normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = states.shape[0]
        states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)          # detached
        target_t = torch.as_tensor(target, dtype=torch.float32, device=device)    # detached
        lp_old_t = torch.as_tensor(log_prob_old, dtype=torch.float32, device=device)  # detached

        # ---- 3. 内层 actor 多轮更新（B_θ = n_epochs_policy）----
        for _e in range(n_epochs_policy):
            # 元组级 minibatch：打乱所有 (s,a,adv,target,lp_old) 元组后按批抽子批。
            epoch_sum_r_is, epoch_sum_kl, epoch_n = 0.0, 0.0, 0
            idx = np.random.permutation(N)
            for mb_start in range(0, N, n_minibatch or N):
                mb = idx[mb_start:mb_start + (n_minibatch or N)]
                mb_states = states_t[mb]
                mb_actions = actions_t[mb]
                mb_adv = adv_t[mb]
                mb_lp_old = lp_old_t[mb]

                # 当前 θ 的 log_prob；r_IS = exp(lp_new − lp_old)；log_prob 在未裁剪动作上算。
                log_prob_new = policy.log_prob(mb_states, mb_actions)
                log_ratio = _is_ratio_safe(log_prob_new - mb_lp_old, max_ratio_log)
                r_is = torch.exp(log_ratio)

                if algo == "clip":
                    r_is_clip = torch.clamp(r_is, 1.0 - clip_eps, 1.0 + clip_eps)
                    # J_PPO2 = mean(min(r_IS·A, r̄_IS·A))（书 eq:ppo-clip-obj）
                    j_ppo = torch.min(r_is * mb_adv, r_is_clip * mb_adv).mean()
                    kl = (r_is - 1.0 - log_ratio).mean()       # 仅记录，不参与 loss
                else:
                    # 原始 PPO：J_PPO = J̄ − β·KL̂；J̄ = mean(r_IS·A)
                    j_bar = (r_is * mb_adv).mean()
                    kl = (r_is - 1.0 - log_ratio).mean()       # k3 估计器
                    j_ppo = j_bar - beta * kl

                # 梯度上升/下降等价：loss_a = −J_PPO（PyTorch 默认梯度下降）
                loss_a = -j_ppo
                actor_optimizer.zero_grad()
                loss_a.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                actor_optimizer.step()

                # 记录（用该子批的 r_is / kl）
                epoch_sum_r_is += float(r_is.detach().mean().item())
                epoch_sum_kl += float(kl.detach().mean().item())
                epoch_n += 1
                last_loss_a = float(loss_a.detach().item())
            last_mean_r_is = epoch_sum_r_is / max(epoch_n, 1)
            last_mean_kl = epoch_sum_kl / max(epoch_n, 1)

            # PPO-Penalty：β 滞回自适应（在 actor 内层循环内进行，书 algo:pg_ppo_pen）
            if algo == "pen":
                if last_mean_kl > kl_max:
                    beta = (1.0 + kl_delta) * beta
                elif last_mean_kl < kl_min:
                    beta = (1.0 - kl_delta) * beta

        # ---- 4. 内层 critic 多轮更新（B_ω = n_epochs_value）----
        last_loss_v = float("nan")
        for _e in range(n_epochs_value):
            idx = np.random.permutation(N)
            for mb_start in range(0, N, n_minibatch or N):
                mb = idx[mb_start:mb_start + (n_minibatch or N)]
                mb_states = states_t[mb]
                mb_target = target_t[mb]
                values = value_net(mb_states)                  # 带梯度
                critic_loss = value_coef * ((values - mb_target) ** 2).mean()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                last_loss_v = float(critic_loss.detach().item())

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
            sigma=float(np.exp(mean_log_std)),
            loss_a=float(last_loss_a),
            loss_v=float(last_loss_v),
            v_pred_mean=float(np.mean(v_pred_all)),
            v_corr=v_corr,
            mean_r_is=float(last_mean_r_is),
            mean_kl=float(last_mean_kl),
            beta=float(beta),
        )
        # 末位置 x（仅 cartpole 有意义：state[0]=x）——用于中途看出「车是否在回中」。
        if env_name == "cartpole":
            fx = np.array([float(t["final_state"][0]) for t in trajs])
            entry["final_x_abs_mean"] = float(np.mean(np.abs(fx)))
            entry["final_x_abs_max"] = float(np.max(np.abs(fx)))
        history.append(entry)

        # ---- 6. 保存中间 checkpoint（latest 覆盖式 + iter 编号各一份，每 ckpt_interval 轮）----
        if ckpt_path is not None and ckpt_interval > 0 and (it % ckpt_interval == 0 or it == n_iters - 1):
            num_path = os.path.join(ckpt_dir, f"ckpt_{env_name}_ppo_iter{it:04d}.pth")
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
                sigma=f"{entry['sigma']:.2f}",
                la=f"{entry['loss_a']:.3f}",
                lv=f"{entry['loss_v']:.2f}",
                ris=f"{entry['mean_r_is']:.2f}",
                kl=f"{entry['mean_kl']:.4f}",
                vc=f"{entry['v_corr']:.2f}",
                **({} if env_name != "cartpole" else
                   {"fx": f"{entry.get('final_x_abs_mean', float('nan')):.2f}"}),
            )
            bar.update(1)
        elif it % log_every == 0 or it == n_iters - 1:
            print(
                f"iter {it:5d} | ret {entry['ret_mean']:8.2f}±{entry['ret_std']:7.2f} "
                f"| σ {entry['sigma']:5.3f} | loss_a {entry['loss_a']:9.4f} "
                f"| loss_v {entry['loss_v']:8.3f} | r_IS {entry['mean_r_is']:7.3f} "
                f"| KL {entry['mean_kl']:8.4f} | v_corr {entry['v_corr']:7.4f} "
                f"| v_pred {entry['v_pred_mean']:8.2f} | len {entry['ep_len_mean']:7.1f}"
                f"{fx_str}", flush=True,
            )

    if bar is not None:
        bar.close()

    elapsed = time.time() - t_start
    print(f"[elapsed] {elapsed:.1f}s over {n_iters} iters "
          f"({elapsed / n_iters:.3f}s/iter) on {device}", flush=True)

    if save_dir is not None:
        save_history(history, save_dir, env_name, algo)
    return history


def save_history(history, save_dir, env_name, algo):
    """把训练历史写成 JSON（供核对 / 后续成文引用）。"""
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"reward_history_ppo_{algo}_{env_name}.json")
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
        ax.set_title(f"PPO ({algo}) on {env_name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        png_path = os.path.join(save_dir, f"reward_history_ppo_{algo}_{env_name}.png")
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
    ap = argparse.ArgumentParser(
        description="PPO（连续动作、高斯策略 + 学习值函数 V(s)），三变体：og=原始/pen=Penalty/clip=Clip")
    ap.add_argument("--env", choices=["cartpole", "pendulum"], default="cartpole")
    ap.add_argument("--algo", choices=["og", "pen", "clip"], default="og",
                    help="PPO 变体：og=原始PPO(KL惩罚β)、pen=PPO-Penalty(自适应β)、clip=PPO-Clip(ε裁剪)")
    ap.add_argument("--iters", type=int, default=None,
                    help="训练轮数（默认 cartpole=200, pendulum=600）")
    ap.add_argument("--eps-per-iter", type=int, default=None,
                    help="每轮采样轨迹数（默认 cartpole=16, pendulum=32）")
    ap.add_argument("--adv", choices=["td", "gae"], default="gae",
                    help="优势估计方式：td=单步 TD 优势 δ_t；gae=GAE λ 加权优势（默认 gae）")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.95,
                    help="GAE λ（默认 0.95；λ=0 退化为 TD 优势）")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="actor（策略）学习率（与 1_reinforce.py 同义）")
    ap.add_argument("--actor-lr", type=float, default=None,
                    help="actor（策略）学习率；默认取 --lr")
    ap.add_argument("--critic-lr", type=float, default=1e-3,
                    help="critic（值函数）学习率")
    ap.add_argument("--value-coef", type=float, default=0.5,
                    help="critic 损失系数（loss_v = value_coef * MSE(V(s_t), TD target)）")
    ap.add_argument("--hidden", type=parse_hidden, default=(64, 64))
    ap.add_argument("--init-log-std", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--no-normalize", action="store_true",
                    help="关闭优势归一化（只保留 TD/GAE 优势本身）")

    # ---- PPO 专项超参 ----
    ap.add_argument("--n-epochs-policy", type=int, default=4,
                    help="内层 actor 多轮更新次数 B_θ（批内复用数据的轮数）")
    ap.add_argument("--n-epochs-value", type=int, default=4,
                    help="内层 critic 多轮更新次数 B_ω")
    ap.add_argument("--minibatch", type=int, default=0,
                    help="元组级子批大小（0=全 batch 一次更新；>0 则打乱后按该大小抽子批）")
    ap.add_argument("--kl-beta", type=float, default=0.02,
                    help="原始 PPO 的固定 KL 惩罚系数 β（--algo og）")
    ap.add_argument("--kl-beta0", type=float, default=0.02,
                    help="PPO-Penalty 的初始 β（--algo pen）")
    ap.add_argument("--kl-min", type=float, default=0.005,
                    help="PPO-Penalty 的 KL̂ 下限（低于则减小 β）")
    ap.add_argument("--kl-max", type=float, default=0.05,
                    help="PPO-Penalty 的 KL̂ 上限（高于则增大 β）")
    ap.add_argument("--kl-delta", type=float, default=0.2,
                    help="PPO-Penalty 的自适应步长 δ（β ← (1±δ)β）")
    ap.add_argument("--clip-eps", type=float, default=0.2,
                    help="PPO-Clip 的裁剪宽度 ε（--algo clip）")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="梯度范数裁剪（0=关闭；如 0.5/1.0）")
    ap.add_argument("--max-ratio-log", type=float, default=20.0,
                    help="log_ratio 数值安全上限（exp 前夹到 ±该值，防溢出 inf/NaN）")

    # ---- 复用 AC/REINFORCE 的既有超参 ----
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

    # 每环境默认超参（与 1_reinforce.py / 2_ac.py 对齐）
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
          f"algo={args.algo} adv={args.adv} lam={args.lam} "
          f"actor_lr={args.actor_lr} critic_lr={args.critic_lr} "
          f"value_coef={args.value_coef} normalize={not args.no_normalize} "
          f"n_epochs_p={args.n_epochs_policy} n_epochs_v={args.n_epochs_value} "
          f"minibatch={args.minibatch} grad_clip={args.grad_clip} "
          f"init_log_std={args.init_log_std} seed={args.seed} | device={dev_name}")
    if args.algo == "og":
        print(f"[algo] 原始 PPO: kl-beta={args.kl_beta}")
    elif args.algo == "pen":
        print(f"[algo] PPO-Penalty: kl-beta0={args.kl_beta0} "
              f"kl-min={args.kl_min} kl-max={args.kl_max} kl-delta={args.kl_delta}")
    else:
        print(f"[algo] PPO-Clip: clip-eps={args.clip_eps}")
    if args.env == "cartpole":
        ir = env.config
        def _pr(name, lo, hi):
            return f"{name}=[{lo:+.2f},{hi:+.2f}]"
        print(f"[init-dist] x {_pr('m', *ir['init_x_range'])} | "
              f"ẋ {_pr('m/s', *ir['init_dot_x_range'])} | "
              f"θ {_pr('°', ir['init_theta_range'][0]*180/np.pi, ir['init_theta_range'][1]*180/np.pi)} | "
              f"θ̇ {_pr('°/s', ir['init_dot_theta_range'][0]*180/np.pi, ir['init_dot_theta_range'][1]*180/np.pi)}")

    run_ppo(
        env, policy, value_net, actor_optimizer, critic_optimizer,
        n_iters=args.iters,
        episodes_per_iter=args.eps_per_iter,
        gamma=args.gamma,
        lam=args.lam,
        adv_mode=args.adv,
        value_coef=args.value_coef,
        normalize_adv=not args.no_normalize,
        obs_rms=obs_rms,
        reset_seed=reset_seed,
        seed=args.seed,
        env_name=args.env,
        algo=args.algo,
        n_epochs_policy=args.n_epochs_policy,
        n_epochs_value=args.n_epochs_value,
        minibatch=args.minibatch,
        kl_beta=args.kl_beta,
        kl_beta0=args.kl_beta0,
        kl_min=args.kl_min,
        kl_max=args.kl_max,
        kl_delta=args.kl_delta,
        clip_eps=args.clip_eps,
        grad_clip=args.grad_clip,
        max_ratio_log=args.max_ratio_log,
        log_every=args.log_every,
        save_dir=save_dir,
        ckpt_dir=ckpt_dir,
        ckpt_interval=args.ckpt_interval,
        use_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
