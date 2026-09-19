# -*- coding: utf-8 -*-
"""
chap6_2_RLtable/code_1_tabular_control.py

验证书 chap6_2_tables.tex 的四个表格型控制方法（B12 修复后原样）：
  MDP_MC_policy_iter / MDP_SARSA / MDP_SARSA_n / MDP_Q_learning
  - 可运行、无异常；
  - 贪心策略收敛到最优 [1,1,1,1,0]（状态 4 向左回退以重新收集回报 1）。

注意：验证计划 §4.3 原写「贪心动作 ≈ 全右，状态 4 两个动作等价」——错误；
真实最优在状态 4 应取左（action 0），两动作不等价。见 corridor.py 头部说明。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chap6_1_RLbasic"))
from corridor import MDP, make_env, sample_trail_policy, optimal_value_and_greedy

np.random.seed(42)


def MDP_MC_policy_iter(env: MDP, N_K, gamma=0.9, L_e=100, vmin=0, vmax=99):
    """书 chap6_2_tables.tex 的 MDP-MC策略迭代（B12 修复后原样）。"""
    Q_mat = np.random.uniform(low=vmin, high=vmax, size=[env.Ns, env.Na])

    def pi_func_greedy_eps(s: int, epsilon: float):
        assert 0 <= s and s < Q_mat.shape[0]
        pi = np.zeros_like(Q_mat[s])
        pi[np.argmax(Q_mat[s])] = 1 - epsilon
        pi += epsilon / Q_mat.shape[1]
        return pi

    num_s = np.zeros([env.Ns, env.Na])
    for k in range(N_K):
        eps_k = 1.0 / (k + 1)                       # 对应框内 ε ← ε(k)，ε(k)=1/k
        rs, states, a_s = sample_trail_policy(env, lambda s: pi_func_greedy_eps(s, eps_k))
        Gs = np.zeros([L_e + 2])
        for t in range(L_e, 0, -1):
            Gs[t] = rs[t] + gamma * Gs[t + 1]
        for t in range(1, L_e + 1):
            num_s[states[t - 1], a_s[t]] += 1
            Q_mat[states[t - 1], a_s[t]] += (Gs[t] - Q_mat[states[t - 1], a_s[t]]) / num_s[states[t - 1], a_s[t]]
    return pi_func_greedy_eps


def MDP_SARSA(env: MDP, N_K, alpha, gamma=0.9, L_e=100, vmin=0, vmax=99):
    """书 chap6_2_tables.tex 的 MDP-SARSA策略迭代（B12 修复后原样）。"""
    Q_mat = np.random.uniform(low=vmin, high=vmax, size=[env.Ns, env.Na])

    def pi_func_greedy_eps(s: int, epsilon: float):
        assert 0 <= s and s < Q_mat.shape[0]
        pi = np.zeros_like(Q_mat[s])
        pi[np.argmax(Q_mat[s])] = 1 - epsilon
        pi += epsilon / Q_mat.shape[1]
        return pi

    num_s = np.zeros([env.Ns, env.Na])
    for k in range(N_K):
        eps_k = 1.0 / (k + 1)                       # 对应框内 ε ← ε(k)，ε(k)=1/k
        rs, states, a_s = sample_trail_policy(env, lambda s: pi_func_greedy_eps(s, eps_k))
        for t in range(1, L_e):
            target_last = rs[t] + gamma * Q_mat[states[t], a_s[t + 1]]
            Q_mat[states[t - 1], a_s[t]] += alpha * (target_last - Q_mat[states[t - 1], a_s[t]])
    return pi_func_greedy_eps


def MDP_SARSA_n(env: MDP, N_K, n_sarsa, alpha, gamma=0.9, L_e=100, vmin=0, vmax=99):
    """书 chap6_2_tables.tex 的 MDP-n步SARSA策略迭代（B12 修复后原样）。"""
    Q_mat = np.random.uniform(low=vmin, high=vmax, size=[env.Ns, env.Na])

    def pi_func_greedy_eps(s: int, epsilon: float):
        assert 0 <= s and s < Q_mat.shape[0]
        pi = np.zeros_like(Q_mat[s])
        pi[np.argmax(Q_mat[s])] = 1 - epsilon
        pi += epsilon / Q_mat.shape[1]
        return pi

    num_s = np.zeros([env.Ns, env.Na])
    for k in range(N_K):
        eps_k = 1.0 / (k + 1)                       # 对应框内 ε ← ε(k)，ε(k)=1/k
        rs, states, a_s = sample_trail_policy(env, lambda s: pi_func_greedy_eps(s, eps_k))
        for t in range(n_sarsa, L_e):
            target_old = sum(gamma ** j * rs[t - n_sarsa + 1 + j] for j in range(n_sarsa)) \
                         + gamma ** n_sarsa * Q_mat[states[t], a_s[t + 1]]
            Q_mat[states[t - n_sarsa], a_s[t - n_sarsa + 1]] += \
                alpha * (target_old - Q_mat[states[t - n_sarsa], a_s[t - n_sarsa + 1]])
    return pi_func_greedy_eps


def MDP_Q_learning(env: MDP, N_K, alpha, gamma=0.9, L_e=100, vmin=0, vmax=99):
    """书 chap6_2_tables.tex 的 MDP-Q学习方法（B12 修复后原样）。"""
    Q_mat = np.random.uniform(low=vmin, high=vmax, size=[env.Ns, env.Na])

    def pi_func_greedy_eps(s: int, epsilon: float):
        assert 0 <= s and s < Q_mat.shape[0]
        pi = np.zeros_like(Q_mat[s])
        pi[np.argmax(Q_mat[s])] = 1 - epsilon
        pi += epsilon / Q_mat.shape[1]
        return pi

    num_s = np.zeros([env.Ns, env.Na])
    for k in range(N_K):
        eps_k = 1.0 / (k + 1)                       # 对应框内 ε ← ε(k)，ε(k)=1/k
        rs, states, a_s = sample_trail_policy(env, lambda s: pi_func_greedy_eps(s, eps_k))
        for t in range(1, L_e):
            target_last = rs[t] + gamma * np.max(Q_mat[states[t]])
            Q_mat[states[t - 1], a_s[t]] += alpha * (target_last - Q_mat[states[t - 1], a_s[t]])
    return pi_func_greedy_eps


def greedy_actions(pi_func):
    """以 epsilon=0 提取贪心动作（对应框内最终 π* ← π_ε(s,a;Q_K) 的贪心解释）。"""
    return np.array([int(np.argmax(pi_func(s, 0.0))) for s in range(5)])


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    env = make_env()
    _, greedy_star = optimal_value_and_greedy()
    print("参考最优贪心 =", greedy_star)
    print()

    N_K = 1500
    alpha = 0.5
    cases = [
        ("MDP_MC_policy_iter(N_K=1500)",      lambda: MDP_MC_policy_iter(env, N_K)),
        ("MDP_SARSA(N_K=1500,alpha=0.5)",     lambda: MDP_SARSA(env, N_K, alpha)),
        ("MDP_SARSA_n(N_K=1500,n=4,alpha=0.5)", lambda: MDP_SARSA_n(env, N_K, 4, alpha)),
        ("MDP_Q_learning(N_K=1500,alpha=0.5)", lambda: MDP_Q_learning(env, N_K, alpha)),
    ]
    for name, fn in cases:
        pi = fn()
        g = greedy_actions(pi)
        ok = (g == greedy_star).all()
        print(f"{name}: 贪心 = {g}  {'PASS' if ok else 'FAIL'}")
