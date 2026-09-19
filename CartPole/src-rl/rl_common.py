# -*- coding: utf-8 -*-
"""RL 公共组件（REINFORCE 起步，兼容 PPO）。

对应书：
- `chap6_3_pgmeth_2.tex` §21.5 REINFORCE 算法（algo:pg_reinforce）的连续动作版；
- `chap6_3_pgmeth_1.tex` §21.3 eq:plcgrad-thrm-gauss（高斯策略梯度定理）。

组件：
- set_seed             固定 Python/numpy/torch 随机种子（含 CUDA），AC/PPO 在构造双网络前调用
- RunningMeanStd      状态归一化（增量式均值/方差）
- PolicyNetwork       MLP + 高斯头 (μ, log σ)，π_θ(a|s)=N(a; μ_{s,θ}, σ²_{s,θ})
- ValueNetwork        V(s) 标量值网络（纯 REINFORCE 不用；AC/PPO 的 critic，已可训练使用）
- sample_trajectory   on-policy 采一条轨迹
- compute_returns     γ=1 后缀回报 G_t（书 eq:plcgrad-mc）
- compute_advantages  标量 baseline（b=mean(G_t)）+ 可选 return 归一化
- compute_gae         GAE(γ, λ) 优势估计（书 algo:pg_ac_gae，AC-GAE / PPO 共用基础）
- load/save_ac_checkpoint  AC 专用存档（在 REINFORCE checkpoint 键上多存 value_net/value_optimizer）

符号约定（重点核对，见 eq:plcgrad-thrm-gauss）：
    ∇_θ J(θ) = E[ A(s,a) · ∇_θ ln π_θ(a|s) ]
PyTorch 用梯度下降，故取 loss = -mean(A · ln π)。`torch.distributions.Normal.log_prob`
恰好实现 ln π = -1/2 Σ_i ln(2πσ²_i) - 1/2 Σ_i (a_i-μ_i)²/σ²_i，其梯度与
eq:plcgrad-thrm-gauss 的 -1/2 ∇_θ(Σ ln σ² + Σ (a-μ)²/σ²) 一致（常数项梯度为 0）。
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

__all__ = [
    "set_seed",
    "RunningMeanStd",
    "PolicyNetwork",
    "ValueNetwork",
    "sample_trajectory",
    "compute_returns",
    "compute_advantages",
    "compute_gae",
    "load_policy_ckpt",
    "load_ac_checkpoint",
    "save_ac_checkpoint",
    "ReinforcePolicyController",
]


def set_seed(seed):
    """固定 Python / NumPy / PyTorch 随机种子（含 CUDA），用于可复现训练。

    AC / PPO 等同时有策略网络与价值网络的算法，必须在**构造两个网络之前**调用本函数，
    否则网络的随机初始化不可复现（`1_reinforce.py` 在构造 policy 前固种的做法即此约定）。

    注意：CartPole 环境 `reset(seed=...)` 会覆写全局 `np.random`，故倒立摆训练**不传**
    `reset_seed`（同 REINFORCE），靠本函数在每轮采样前维持稳定的随机流。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RunningMeanStd:
    """增量式均值/方差估计，用于状态归一化（OpenAI baselines 风格）。

    - `update(x)`：x 形状 (N, dim) 或 (dim,)，用 Welford 式的增量合并更新统计量；
    - `normalize(x)`：按当前统计量做 z-score 归一化（不动统计量）。
    """

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0.0

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class PolicyNetwork(nn.Module):
    """高斯策略网络：输入状态 s，输出动作分布参数 (μ, log σ)。

    对应 eq:plcgrad-thrm-gauss 的 π_θ(a|s) = N(a; μ_{s,θ}, σ²_{s,θ})。
    - μ 由 MLP 输出（状态相关）；
    - log σ 取为**状态无关的可学习标量参数**（每动作维一个），是 σ_{s,θ} 的退化
      情形 σ 常数。这样训练更稳，且其梯度与定理一致（见类 docstring 符号说明）。
    log σ 裁剪到 [log_std_min, log_std_max] 防止 σ 崩塌 / 数值爆炸。

    采样返回「未裁剪」的动作，由环境内部裁剪到动作上下界（CartPoleCustomEnv 与
    Pendulum-v1 均如此）；log_prob 在未裁剪动作上计算，使 E[(a-μ)²/σ²]=1 对当前
    策略采样严格成立，避免因裁剪导致 σ 被系统性地推小（σ 崩塌）。
    """

    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64),
                 activation=nn.Tanh, log_std_min=-2.3, log_std_max=2.0,
                 init_log_std=0.0):
        super().__init__()
        trunk = []
        in_dim = state_dim
        for h in hidden_sizes:
            trunk.append(nn.Linear(in_dim, h))
            trunk.append(activation())
            in_dim = h
        self.trunk = nn.Sequential(*trunk)
        self.mu_head = nn.Linear(in_dim, action_dim)
        # 状态无关 log σ（可学习标量参数）
        self.log_std = nn.Parameter(
            torch.full((action_dim,), float(init_log_std), dtype=torch.float32))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        mu = self.mu_head(self.trunk(state))
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        log_std = log_std.expand_as(mu)
        return mu, log_std

    def get_dist(self, state):
        mu, log_std = self.forward(state)
        return Normal(mu, log_std.exp())

    def log_prob(self, state, action):
        """ln π_θ(a|s)（对动作各维求和），供训练 loss 使用（保留计算图）。"""
        return self.get_dist(state).log_prob(action).sum(dim=-1)

    def sample(self, state):
        """从 π_θ(·|s) 采样一个（未裁剪的）动作。环境内部负责裁剪到上下界。"""
        return self.get_dist(state).sample()


class ValueNetwork(nn.Module):
    """V(s) 标量值网络：输入状态，输出单值 V(s)。纯 REINFORCE 不用，预留给 AC/PPO。"""

    def __init__(self, state_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        trunk = []
        in_dim = state_dim
        for h in hidden_sizes:
            trunk.append(nn.Linear(in_dim, h))
            trunk.append(activation())
            in_dim = h
        self.trunk = nn.Sequential(*trunk)
        self.value_head = nn.Linear(in_dim, 1)

    def forward(self, state):
        return self.value_head(self.trunk(state)).squeeze(-1)


def sample_trajectory(env, policy, obs_rms=None, max_steps=None, reset_seed=None):
    """在当前策略下与环境交互一条 on-policy 轨迹。

    - 状态若给 obs_rms 则先归一化再喂给策略，同时记录原始状态（供 obs_rms.update）；
    - 动作为策略采样的「未裁剪」值，由环境内部裁剪到动作上下界；
    - 使用 gym 0.26 的新 step API：env.step -> (obs, reward, terminated, truncated, info)。

    返回 dict：
        states      归一化后（喂给策略的）状态，(T, dim)
        raw_states  原始状态，(T, dim)
        actions     采样（未裁剪）动作，(T, act_dim)
        rewards     奖励，(T,)
        length      轨迹长度 T
        final_state 末步状态（最后一步 step 之后的 obs），用于观测「末位置」等
    """
    if reset_seed is not None:
        raw_obs, _info = env.reset(seed=int(reset_seed))
    else:
        raw_obs, _info = env.reset()

    states = []
    raw_states = []
    actions = []
    rewards = []

    # 策略所在设备（CPU 或 CUDA）：喂给策略的张量放到同一设备，动作取回 CPU 再 step 环境。
    device = next(policy.parameters()).device

    terminated = truncated = False
    t = 0
    while not (terminated or truncated):
        raw_states.append(np.asarray(raw_obs, dtype=np.float32))
        obs = obs_rms.normalize(raw_obs) if obs_rms is not None else raw_obs
        states.append(np.asarray(obs, dtype=np.float32))

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t = policy.sample(obs_t)
        action = action_t.cpu().numpy()[0]

        raw_obs, reward, terminated, truncated, _info = env.step(action)

        actions.append(action)
        rewards.append(float(reward))
        t += 1
        if max_steps is not None and t >= max_steps:
            break

    return {
        "states": np.array(states, dtype=np.float32),
        "raw_states": np.array(raw_states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float64),
        "length": t,
        "final_state": np.asarray(raw_obs, dtype=np.float32),
    }


def compute_returns(rewards, gamma=1.0):
    """后缀回报 G_t = Σ_{τ=t}^{T-1} γ^{τ-t} r_τ（γ=1 时即累计回报）。

    与书 eq:plcgrad-mc / algo:pg_reinforce 的 G_t 一致（从轨迹末端向前累加）。
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    returns = np.zeros_like(rewards)
    g = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns


def compute_advantages(returns, baseline=None, normalize=True):
    """优势 A(s,a) = G_t - b；可选再按批归一化（除以标准差）。

    - baseline 为 None 时取批内均值（书 b = mean(G_t)，标量 baseline）；
    - normalize=True 时进一步除以批内 G_t 的标准差（return 归一化，稳定训练）。
    返回 (advantage, baseline)。advantage 为 numpy 数组，训练时转 tensor 即天然 stop-gradient。
    """
    returns = np.asarray(returns, dtype=np.float64)
    if baseline is None:
        baseline = float(returns.mean())
    adv = returns - baseline
    if normalize:
        std = returns.std()
        if std > 1e-8:
            adv = adv / std
    return adv, baseline


def compute_gae(rewards, values, next_values, gamma=1.0, lam=0.95):
    """GAE(γ, λ) 优势估计 A_t，对应 `algo:pg_ac_gae`（书 chap6_4_ppo_1.tex）。

    **口径与书差异（见 AC-training-plan.md「注意点 1」）**：书里 critic 用
    `δ_ω(t)=r_t+γV(s_{t+1})−V(s_t)`、优势用 `δ_a(t)=r_{t+1}+γV(s_{t+1})−V(s_t)`，对同一个
    `t` 取了不同回报索引，数学上不自洽。代码采用**自洽的单 δ 形式**：

        δ_t = rewards[t] + γ·V(s_{t+1}) − V(s_t)
        A_t = Σ_{j≥0} (γλ)^j δ_{t+j}    （递归：A_t = δ_t + γλ·A_{t+1}，A_T = 0）

    其中 `rewards[t]` = 从 `s_t` 转移到 `s_{t+1}` 的回报。critic 的 TD 目标与非 GAE 优势
    共用同一个 `δ_t`；`λ=0` 时 `A_t = δ_t`（退化为单步 TD 优势）。

    参数：
    - rewards:     长度为 T 的数组（r_0..r_{T-1}），float64。
    - values:      长度为 T 的数组，V(s_0..s_{T-1})。
    - next_values: 长度为 T 的数组，V(s_1..s_T)；**末态 V(s_T) 由调用方置 0**（本环境终止
                    = 越界或满步，truncated 恒 False，均视为真终态、不 bootstrap）。
    - gamma:       折扣因子（本项目 γ=1）。
    - lam:         GAE λ（缺省 0.95）。

    返回长度为 T 的 numpy 数组 A_0..A_{T-1}，训练时转 tensor 即天然 stop-gradient。
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    T = rewards.shape[0]
    deltas = rewards + gamma * next_values - values
    adv = np.zeros_like(deltas)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        gae = deltas[t] + gamma * lam * gae
        adv[t] = gae
    return adv


def load_policy_ckpt(path, obs_dim, act_dim, hidden_sizes=(64, 64),
                     device="cpu", init_log_std=0.0):
    """从 `1_reinforce.py` 保存的 .pth 加载策略 + 状态归一化统计量。

    返回 (policy, obs_rms, meta)：
    - policy: `PolicyNetwork`，`eval()` 已置于推理模式（如需采样请再 `.train()`）；
    - obs_rms: 恢复到存档均值/方差/计数的 `RunningMeanStd`；
    - meta: 训练时写入的超参 dict（env/seed/n_iters/…）。
    """
    ckpt = torch.load(path, map_location=device)
    policy = PolicyNetwork(obs_dim, act_dim, hidden_sizes=hidden_sizes,
                           init_log_std=init_log_std).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    obs_rms.mean = np.asarray(ckpt["obs_rms_mean"], dtype=np.float64)
    obs_rms.var = np.asarray(ckpt["obs_rms_var"], dtype=np.float64)
    obs_rms.count = float(ckpt["obs_rms_count"])
    return policy, obs_rms, ckpt.get("meta", {})


def save_ac_checkpoint(policy, optimizer, value_net, value_optimizer,
                       obs_rms, it, meta, path, numbered_path=None):
    """AC 专用 checkpoint：在 REINFORCE 的 `policy`/`meta`/`obs_rms_*` 键之上额外存价值网络与优化器。

    - 保留 `policy`+`meta`+`obs_rms_*` 键 → `load_policy_ckpt`（`eval/bench.py`、`eval_centering.py`
      复用的评估入口）**可直接读**该 `.pth`，不必改动 bench 代码；
    - 新增键 `value_net`/`value_optimizer`，以及沿用 REINFORCE 命名的 `optimizer`（actor/策略优化器）；
    - `path` 写最新一份（同名覆盖）；`numbered_path`（可选）额外写带 iter 编号的一份，供正式训练
      「每隔 N 轮落一份、训后取表现最好的一份」。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "value_net": value_net.state_dict(),
        "value_optimizer": value_optimizer.state_dict(),
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


def load_ac_checkpoint(path, obs_dim, act_dim, hidden_sizes=(64, 64),
                       device="cpu", init_log_std=0.0):
    """从 `save_ac_checkpoint` 保存的 `.pth` 恢复 AC 网络与状态归一化统计量。

    返回 dict：
    - policy / value_net: 已 `.eval()` 的 `PolicyNetwork` / `ValueNetwork`；
    - obs_rms: 恢复到存档均值/方差/计数的 `RunningMeanStd`；
    - meta: 训练超参 dict；iter: 存档时轮数；
    - optimizer / value_optimizer: optimizer `state_dict`（ckpt 缺失则 None），
      供 `2_ac.py` 断点续训时加载到新建的优化器上。
    """
    ckpt = torch.load(path, map_location=device)
    policy = PolicyNetwork(obs_dim, act_dim, hidden_sizes=hidden_sizes,
                           init_log_std=init_log_std).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    value_net = ValueNetwork(obs_dim, hidden_sizes=hidden_sizes).to(device)
    value_net.load_state_dict(ckpt["value_net"])
    value_net.eval()
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    obs_rms.mean = np.asarray(ckpt["obs_rms_mean"], dtype=np.float64)
    obs_rms.var = np.asarray(ckpt["obs_rms_var"], dtype=np.float64)
    obs_rms.count = float(ckpt["obs_rms_count"])
    return dict(
        policy=policy,
        value_net=value_net,
        obs_rms=obs_rms,
        meta=ckpt.get("meta", {}),
        iter=ckpt.get("iter", None),
        optimizer=ckpt.get("optimizer"),
        value_optimizer=ckpt.get("value_optimizer"),
    )


class ReinforcePolicyController:
    """把训练好的高斯策略封装成 `eval/bench.py` 期望的 controller 接口。

    bench.py 的 `rollout()` 对控制器只调用 `.compute_action(state)` 与可选的 `.reset()`，
    本类据此实现。评估默认取均值动作 μ（确定性策略，标准做法）；`mean=False` 时采样，
    可用于对比随机性对结果的影响。env 内部会把 F 裁剪到 ±F_max，因此返回的未裁剪
    动作与 bench.py 用 `info["F_applied"]`（裁剪后）计算 J_ach 的既有口径一致。
    """

    def __init__(self, policy, obs_rms, device="cpu", mean=True):
        self.policy = policy
        self.obs_rms = obs_rms
        self.device = device
        self.mean = mean

    def reset(self):
        """bench.py 的 rollout() 每回合调用；无内部状态，空实现。"""

    def compute_action(self, state):
        state = np.asarray(state, dtype=np.float64)
        s = self.obs_rms.normalize(state)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.mean:
                mu, _ = self.policy(s_t)
                action = mu.cpu().numpy()[0]
            else:
                action = self.policy.sample(s_t).cpu().numpy()[0]
        return action
