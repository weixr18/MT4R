# -*- coding: utf-8 -*-
"""
chap6_1_RLbasic/code_2_mc_td_predict.py

验证书 chap6_1_basics_2.tex 的采样与策略评估（B3/B4/B7 修复 + 2026-08-26 出发约定统一后原样）：
  - sample_trail_policy 索引对齐：rs[t] = R̄(s_{t-1}, a_t)（进入 s_t 的回报）、
    rs[0] 占位不被下游使用；
  - MDP_MC_policy_est / MDP_td0_policy_est / MDP_tdn_policy_est 统一为「出发约定」：
    状态 s_t 的价值折扣回报从 r_{t+1} 起算（离开 s_t 后的第一笔回报），收敛到
    均匀策略的贝尔曼不动点（标准出发约定，见 corridor.bellman_fixed_point）。

约定说明（2026-08-26 统一，方案 A）：
  - 书理论（eq:mdp-bellman-v / eq:mdp-vtoq）与 DP 为出发约定：R̄(s,a) 是在状态 s
    取动作 a 的回报。Q 估计（chap6_2/6_3）配对 (s_{t-1},a_t) 与 r_t，本就不受影响。
  - 采样 sample_trail_policy 记 rs[t]=R̄(s_{t-1},a_t)=进入 s_t 的回报（即标准记号 R_t）；
    故状态 s_t 的出发回报 = r_{t+1} + γr_{t+2} + ... 。
  - MC ：Gs[t]=Σ_{k≥0} γ^k rs[t+k+1]；V(s_t) ← Gs[t]（t=0..L_e-1，末位 s_{L_e} 为
    截断位不更新）。
  - TD0：T_{t-1} = r_t + γV(s_t)。
  - TD(n)：T_{t-n-1} = Σ_{i=0}^{n} γ^i r_{t-n+i} + γ^{n+1}V(s_t)（窗 r_{t-n}..r_t，
    条件 t>n，覆盖状态 0）。

已知特性（非 bug）：
  - L_e 有限截断：MC/TD 收敛到「截断回报」的期望，与无限时域不动点差一个
    γ^(剩余步数) 量级的偏差（L_e=100、γ=0.9 时约 0.1；拉长 L_e 可降到 <0.05）。
  - TD0/TDn 用常数步长 α（书中为超参数），存在噪声底（verify-RL-codes.md 问题 3）：
    α 大则噪声大、不随 N_I 无限下降。验证以「系统偏差已消除（数值向标准不动点
    靠拢）」为准，不声称严格收敛。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from corridor import MDP, make_env, sample_trail_policy, bellman_fixed_point, uniform_policy

np.random.seed(42)


def MDP_MC_policy_est(env: MDP, pi_func, gamma=0.9, L_e=100, vmin=0, vmax=99, N_I=100):
    """书 chap6_1_basics_2.tex 的 MDP-MC策略评估（出发约定：Gs[t] 从 r_{t+1} 起算）。"""
    V_pi = np.random.uniform(low=vmin, high=vmax, size=(env.Ns,))
    num_s = np.zeros([env.Ns])
    for _ in range(N_I):
        rs, states, a_s = sample_trail_policy(env, pi_func, L_e)
        Gs = np.zeros([L_e + 2])
        for t in range(L_e - 1, -1, -1):
            Gs[t] = rs[t + 1] + gamma * Gs[t + 1]
        for t in range(0, L_e):
            num_s[states[t]] += 1
            V_pi[states[t]] += (Gs[t] - V_pi[states[t]]) / num_s[states[t]]
    return V_pi


def MDP_td0_policy_est(env: MDP, pi_func, gamma=0.9, alpha=0.5, L_e=100, vmin=0, vmax=99, N_I=100):
    """书 chap6_1_basics_2.tex 的 MDP-TD(0)策略评估（出发约定：T_{t-1}=r_t+γV(s_t)）。"""
    V_pi = np.random.uniform(low=vmin, high=vmax, size=(env.Ns,))
    num_s = np.zeros([env.Ns])
    for _ in range(N_I):
        states = [env.init_state()]
        rs = [0]
        for t in range(1, L_e + 1):
            a_t = np.random.choice(env.Na, p=pi_func(states[t - 1]))
            states.append(env.state_transfer(states[t - 1], a_t))
            rs.append(env.reward(states[t - 1], a_t))
            target_last = rs[t] + gamma * V_pi[states[t]]
            V_pi[states[t - 1]] += alpha * (target_last - V_pi[states[t - 1]])
    return V_pi


def MDP_tdn_policy_est(env: MDP, pi_func, gamma=0.9, n_td=4, alpha=0.5, L_e=100, vmin=0, vmax=99, N_I=100):
    """书 chap6_1_basics_2.tex 的 MDP-TD(n)策略评估（出发约定：窗 r_{t-n}..r_t，条件 t>n）。"""
    V_pi = np.random.uniform(low=vmin, high=vmax, size=(env.Ns,))
    num_s = np.zeros([env.Ns])
    for _ in range(N_I):
        states = [env.init_state()]
        rs = [0]
        for t in range(1, L_e + 1):
            a_t = np.random.choice(env.Na, p=pi_func(states[t - 1]))
            states.append(env.state_transfer(states[t - 1], a_t))
            rs.append(env.reward(states[t - 1], a_t))
            if t > n_td:
                target_old = gamma ** (n_td + 1) * V_pi[states[t]]
                for i in range(n_td + 1):
                    target_old += (gamma ** i) * rs[t - n_td + i]
                V_pi[states[t - n_td - 1]] += alpha * (target_old - V_pi[states[t - n_td - 1]])
    return V_pi


def err_report(name, V_pi, V_ref, states_range=range(5), note=""):
    err = np.max(np.abs(V_pi[list(states_range)] - V_ref[list(states_range)]))
    flag = "PASS (<0.05)" if err < 0.05 else "低于0.05"
    print(f"{name}: V = {V_pi}")
    print(f"        不动点 = {V_ref}")
    print(f"        最大绝对误差(状态{list(states_range)}) = {err:.5f}  ({flag})  {note}")
    return err


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    env = make_env()
    V_ref = bellman_fixed_point(uniform_policy)
    print("均匀策略贝尔曼不动点 V_ref =", V_ref)
    print()

    # --- 采样索引对齐检查 ---
    rs, states, a_s = sample_trail_policy(env, uniform_policy, L_e=5)
    print("sample_trail_policy 索引检查（L_e=5）：")
    for t in range(1, 6):
        ok = rs[t] == (1.0 if (states[t - 1] == 3 and a_s[t] == 1) else 0.0)
        print(f"  t={t}: s_{t-1}={states[t - 1]} a_t={a_s[t]} -> s_t={states[t]} "
              f"rs[t]={rs[t]} {'OK' if ok else 'MISMATCH'}")
    print(f"  rs[0] 占位 = {rs[0]}（不参与下游计算）")
    print()

    # --- MC 策略评估 ---
    for L_e, N_I in [(100, 2000), (800, 1500)]:
        V_mc = MDP_MC_policy_est(env, uniform_policy, L_e=L_e, N_I=N_I)
        err_report(f"MDP_MC_policy_est(L_e={L_e},N_I={N_I})", V_mc, V_ref,
                   note="截断偏差随 L_e 下降（L_e=100 时 ~0.1 为截断底，非差一）")
        print()

    # --- TD(0) 策略评估 ---
    for alpha in [0.1, 0.05]:
        V_td0 = MDP_td0_policy_est(env, uniform_policy, alpha=alpha, N_I=2000)
        err_report(f"MDP_td0_policy_est(N_I=2000,alpha={alpha})", V_td0, V_ref,
                   note="常数步长噪声底，α 小则更贴近不动点（问题 3）")
        print()

    # --- TD(n) 策略评估 ---
    for alpha in [0.1, 0.05]:
        V_tdn = MDP_tdn_policy_est(env, uniform_policy, n_td=4, alpha=alpha, N_I=2000)
        err_report(f"MDP_tdn_policy_est(N_I=2000,n_td=4,alpha={alpha})", V_tdn, V_ref,
                   note="n 步目标方差大，常数步长噪声底更明显（问题 3）；状态 0 已随 t>n 覆盖")
        print()
