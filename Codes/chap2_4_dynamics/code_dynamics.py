# -*- coding: utf-8 -*-
"""《Math Toolbox for Robotics》Part II 机器人动力学解析解（chap2_4_dynamics.tex）。

核心实现，与书 `algo:robot_dynamics` 算法框及正文代码块一致。
数组索引对齐约定（见 2index.tex「数组索引」）：长度为 N+1、下标 1..N 与数学记号一致，[0] 不用。

- robot_B(q, d, a, alpha, m, p_cents, I_inn, N=6)：惯量矩阵 B(q)，复用质心雅可比 robot_j_centroid
- robot_dyn(q, qdot, d, a, alpha, m, p_cents, I_inn, g0, N=6, h=1e-6)：
  动力学项 B(q), C(q, qdot), g(q)，满足 B(q)q̈ + C(q,q̇)q̇ + g(q) = τ。
  C 矩阵所需的 ∂B/∂q 在本实现中用中心差分近似（书中注明实际工程可用自动微分替代）。

完整验证脚本见 test_dynamics.py。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "chap2_2_kinematcs"))
from code_fk import calc_T_n_to_last
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "chap2_3_diffk"))
from code_jacobian import robot_j_centroid


def robot_B(q, d, a, alpha, m, p_cents, I_inn, N=6):
    """惯量矩阵 B(q) = Σ_n m_n J_p,n^T J_p,n + J_θ,n^T R_n I_n^n (R_n)^T J_θ,n。

    J_p,n / J_θ,n 由质心雅可比 robot_j_centroid 给出（第 n 个连杆对应列 1..n，
    n 之后为 0，求和不受影响）；R_n^b = (T_n^0)_{1:3,1:3} 由前向递推得到。
    """
    assert q.shape == (N + 1,) and d.shape == (N + 1,)
    assert a.shape == (N + 1,) and alpha.shape == (N + 1,)
    assert m.shape == (N + 1,) and p_cents.shape == (N + 1, 3)
    assert I_inn.shape == (N + 1, 3, 3)
    J_w_ns = robot_j_centroid(q, d, a, alpha, p_cents, N)
    T_n_to_bases = np.zeros([N + 1, 4, 4])
    T_n_to_bases[0] = np.eye(4)
    for n in range(1, N + 1):
        T_n_to_bases[n] = T_n_to_bases[n - 1] @ calc_T_n_to_last(
            q[n], d[n], a[n], alpha[n]
        )
    B = np.zeros((N, N))
    for n in range(1, N + 1):
        Jp = J_w_ns[n, :3, :]
        Jt = J_w_ns[n, 3:, :]
        Rn = T_n_to_bases[n, :3, :3]
        B += m[n] * (Jp.T @ Jp) + Jt.T @ (Rn @ I_inn[n] @ Rn.T) @ Jt
    return B


def robot_dyn(q, qdot, d, a, alpha, m, p_cents, I_inn, g0, N=6, h=1e-6):
    """动力学项 B(q), C(q, qdot), g(q)，满足 B(q)q̈ + C(q,q̇)q̇ + g(q) = τ。

    - g(q) = -Σ_n m_n J_p,n^T g0 为 N 维列向量，即 g(q) = ∂U/∂q（U = -Σ m_n g0^T p_n）。
      标准重力项：重力补偿控制 u = ... + g(q) 直接抵消重力（见第4章）。
    - C(q,q̇) 元素 c_njk = ½(∂b_nj/∂q_k + ∂b_nk/∂q_j - ∂b_jk/∂q_n)，∂B/∂q 用中心差分近似。
    """
    assert q.shape == (N + 1,) and qdot.shape == (N + 1,)
    assert g0.shape == (3,)
    B = robot_B(q, d, a, alpha, m, p_cents, I_inn, N)
    J_w_ns = robot_j_centroid(q, d, a, alpha, p_cents, N)
    g = np.zeros(N)
    for n in range(1, N + 1):
        g -= m[n] * (J_w_ns[n, :3, :].T @ g0)
    d_Bs = np.zeros((N, N, N))              # d_Bs[k,i,j] = ∂b_ij/∂q_{k+1}
    for k in range(1, N + 1):
        qp = q.copy(); qp[k] += h
        qm = q.copy(); qm[k] -= h
        d_Bs[k - 1] = (robot_B(qp, d, a, alpha, m, p_cents, I_inn, N)
                       - robot_B(qm, d, a, alpha, m, p_cents, I_inn, N)) / (2 * h)
    C = np.zeros((N, N))
    for n in range(1, N + 1):
        for j in range(1, N + 1):
            for k in range(1, N + 1):
                c_njk = 0.5 * (d_Bs[k - 1, n - 1, j - 1]
                               + d_Bs[j - 1, n - 1, k - 1]
                               - d_Bs[n - 1, j - 1, k - 1])
                C[n - 1, j - 1] += c_njk * qdot[k]
    return B, C, g
