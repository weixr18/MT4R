# -*- coding: utf-8 -*-
"""
chap6_1_RLbasic/code_1_dp_predict.py

验证书 chap6_1_basics_2.tex 的 MDP_DP_policy_est（B2 修复后原样）：
  - 可运行、无异常；
  - 对走廊 MDP 的固定均匀策略，收敛到贝尔曼期望方程不动点（误差 < 0.01）。
对照：corridor.bellman_fixed_point()（模型已知的精确线性解）。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from corridor import MDPModel, make_model, bellman_fixed_point, uniform_policy

np.random.seed(42)


def MDP_DP_policy_est(mdp: MDPModel, pi_func, gamma=0.9, vmin=0, vmax=99, N_I=100):
    """书 chap6_1_basics_2.tex 的 MDP-DP策略评估（B2 修复后原样）。"""
    V_pi = np.random.uniform(low=vmin, high=vmax, size=(mdp.Ns,))
    for i in range(N_I):
        for s in range(mdp.Ns):
            v_s = 0
            pi_s = pi_func(s)
            for a in range(mdp.Na):
                r_sa = mdp.reward(s, a)
                psa = mdp.state_transfer_distb(s, a)
                v_next = (gamma * psa[None, :] @ V_pi[:, None])[0, 0]
                v_s += pi_s[a] * (r_sa + v_next)
            V_pi[s] = v_s
    return V_pi


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    mdp = make_model()
    V_ref = bellman_fixed_point(uniform_policy)
    print("均匀策略贝尔曼不动点 V_ref =", V_ref)
    print()
    for N_I in [10, 100]:
        V_pi = MDP_DP_policy_est(mdp, uniform_policy, N_I=N_I)
        err = np.max(np.abs(V_pi - V_ref))
        print(f"N_I={N_I}: V_π = {V_pi}")
        print(f"            不动点 = {V_ref}")
        print(f"            最大绝对误差 = {err:.6f}  "
              f"{'PASS (<0.01)' if err < 0.01 else 'FAIL'}")
