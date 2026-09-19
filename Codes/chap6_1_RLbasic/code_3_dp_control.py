# -*- coding: utf-8 -*-
"""
chap6_1_RLbasic/code_3_dp_control.py

验证书 chap6_1_basics_3.tex 的 MDP_dp_iter（B9）与 MDP_value_iter（B10）：
  - 可运行、无异常；
  - 策略迭代收敛到最优贪心动作 [1,1,1,1,0]；
  - 价值迭代的 V 收敛到 V* = [3.8368, 4.2632, 4.7368, 5.2632, 4.7368]，贪心动作最优。

注意：验证计划 §4.2 原写「贪心动作全为右」——基于错误的 V* 表述；
真实最优在状态 4 应取左（action 0）。见 corridor.py 头部说明。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from corridor import MDPModel, make_model, optimal_value_and_greedy
from code_1_dp_predict import MDP_DP_policy_est

np.random.seed(42)


def MDP_dp_iter(mdp: MDPModel, N_k, gamma=0.9):
    """书 chap6_1_basics_3.tex 的 MDP-DP策略迭代（B9 修复后原样）。"""
    Q_mat = np.random.rand(mdp.Ns, mdp.Na)

    def pi_func_greedy(s: int):
        assert 0 <= s and s < Q_mat.shape[0]
        pi = np.zeros_like(Q_mat[s])
        pi[np.argmax(Q_mat[s])] = 1
        return pi

    for k in range(N_k):
        V_k = MDP_DP_policy_est(mdp, pi_func_greedy, gamma=gamma)
        for s in range(mdp.Ns):
            for a in range(mdp.Na):
                r_sa = mdp.reward(s, a)
                psa = mdp.state_transfer_distb(s, a)
                v_next = (gamma * psa[None, :] @ V_k[:, None])[0, 0]
                Q_mat[s, a] = r_sa + v_next
        pass
    return pi_func_greedy


def MDP_value_iter(mdp: MDPModel, N_k, gamma=0.9, vmin=0, vmax=99):
    """书 chap6_1_basics_3.tex 的 MDP-确定性价值迭代（B10 修复后原样）。"""
    Q_mat = np.random.rand(mdp.Ns, mdp.Na)

    def pi_func_greedy(s: int):
        assert 0 <= s and s < Q_mat.shape[0]
        pi = np.zeros_like(Q_mat[s])
        pi[np.argmax(Q_mat[s])] = 1
        return pi

    V = np.random.uniform(low=vmin, high=vmax, size=(mdp.Ns,))
    for k in range(N_k):
        for s in range(mdp.Ns):
            for a in range(mdp.Na):
                r_sa = mdp.reward(s, a)
                psa = mdp.state_transfer_distb(s, a)
                v_next = (gamma * psa[None, :] @ V[:, None])[0, 0]
                Q_mat[s, a] = r_sa + v_next
        V = np.max(Q_mat, axis=1)
    return pi_func_greedy


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    mdp = make_model()
    V_star, greedy_star = optimal_value_and_greedy()
    print("参考 V* =", V_star, "  最优贪心 =", greedy_star)
    print()

    # --- 策略迭代 ---
    pi_g = MDP_dp_iter(mdp, N_k=10)
    g_dp = np.array([int(np.argmax(pi_g(s))) for s in range(mdp.Ns)])
    print("MDP_dp_iter(N_k=10) 贪心动作 =", g_dp,
          "  PASS" if (g_dp == greedy_star).all() else "  FAIL")
    print()

    # --- 价值迭代（书函数原样，仅验贪心）---
    pi_vi = MDP_value_iter(mdp, N_k=100)
    g_vi = np.array([int(np.argmax(pi_vi(s))) for s in range(mdp.Ns)])
    print("MDP_value_iter(N_k=100) 贪心动作 =", g_vi,
          "  PASS" if (g_vi == greedy_star).all() else "  FAIL")

    # --- 价值迭代 V 收敛（镜像书内更新并捕捉 V，验证 V→V*）---
    V = np.random.uniform(low=0, high=99, size=(mdp.Ns,))
    Q_mat = np.random.rand(mdp.Ns, mdp.Na)
    for k in range(100):
        for s in range(mdp.Ns):
            for a in range(mdp.Na):
                r_sa = mdp.reward(s, a)
                psa = mdp.state_transfer_distb(s, a)
                v_next = (0.9 * psa[None, :] @ V[:, None])[0, 0]
                Q_mat[s, a] = r_sa + v_next
        V = np.max(Q_mat, axis=1)
    err_v = np.max(np.abs(V - V_star))
    print("镜像价值迭代 V 与 V* 最大误差 =", err_v,
          "  PASS(<0.01)" if err_v < 0.01 else "  FAIL")
    print("  V =", V)
