# -*- coding: utf-8 -*-
"""验证《Math Toolbox for Robotics》Part II 分析/质心雅可比解析解（chap2_3_diffk_1.tex）。

验证内容（独立参照 = 中心差分数值导数，h=1e-6）：
1. eular_diff_to_w：ω = T_ϑ ϑ̇ 与旋转矩阵数值导数 vee(ṘRᵀ) 一致；且 T_ϑ⁻¹ 还原 ϑ̇
2. robot_jacobian_w（几何雅可比）：平移块 = ∂t_e/∂q，旋转块 = vee(ṘRᵀ)
3. robot_jacobian_a（分析雅可比）：J_a = ∂(robot_fk 输出)/∂q（位置+欧拉角），端到端一致
4. robot_j_centroid（质心雅可比）：第 n 个连杆，位置块 1..n 列 = ∂COM_n/∂q、
   旋转块 1..n 列 = vee(Ṙ_n R_nᵀ)，n 之后列为 0
5. 2 连杆平面臂闭式雅可比（J_w/J_a 平移块 = 平面臂导数）

运行：E:\\Anaconda3\\envs\\py311-gym\\python.exe test_jacobian.py
"""
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "chap2_2_kinematcs"))
from code_fk import calc_T_n_to_last, robot_fk, rot_to_eular
from code_jacobian import (eular_diff_to_w, robot_jacobian_w,
                           robot_jacobian_a, robot_j_centroid)

np.set_printoptions(precision=6, suppress=True)
TOL = 1e-6          # 数值导数（中心差分 h=1e-6）对比容差


def vee(R_hat):
    return np.array([R_hat[2, 1], R_hat[0, 2], R_hat[1, 0]])


def T_N0(q, d, a, alpha, N):
    T = np.eye(4)
    for n in range(1, N + 1):
        T = T @ calc_T_n_to_last(q[n], d[n], a[n], alpha[n])
    return T


def wrap_delta(angle):
    """把欧拉角差分折回 (-π, π]，避免 atan2 分支穿越导致数值导数突变。"""
    return angle - 2.0 * np.pi * np.round(angle / (2.0 * np.pi))


def num_jac_fk(q, d, a, alpha, N, h=1e-6):
    """robot_fk 输出（6 维）对 q 的数值雅可比。"""
    J = np.zeros((6, N))
    for i in range(1, N + 1):
        qp = q.copy(); qp[i] += h
        qm = q.copy(); qm[i] -= h
        fp = robot_fk(qp, d, a, alpha, N)
        fm = robot_fk(qm, d, a, alpha, N)
        J[:3, i - 1] = (fp[:3] - fm[:3]) / (2 * h)
        J[3:, i - 1] = wrap_delta(fp[3:] - fm[3:]) / (2 * h)
    return J


def main():
    print("=" * 60)
    print("验证 分析/质心雅可比解析解（code_jacobian.py）")
    print("=" * 60)

    rng = np.random.default_rng(42)
    N = 6
    d = np.zeros(N + 1); a = np.zeros(N + 1); alpha = np.zeros(N + 1)
    d[1:] = rng.uniform(-0.3, 0.3, N)
    a[1:] = rng.uniform(0.2, 1.0, N)
    alpha[1:] = rng.uniform(-np.pi / 2, np.pi / 2, N)
    q = np.zeros(N + 1)
    q[1:] = np.array([0.6, -0.4, 0.3, -0.5, 0.2, -0.3])   # 避开万向锁

    # ---- 1) eular_diff_to_w：ω = T_ϑ ϑ̇ ----
    print("\n[验证1] eular_diff_to_w：ω = T_ϑ·ϑ̇ vs 数值 vee(ṘRᵀ)")
    max_err = 0.0
    for _ in range(30):
        t = rng.uniform(-0.5, 0.5, 3)
        ϑdot = rng.uniform(-0.8, 0.8, 3)
        # 直接构造 R 与 Ṙ：R(t) = RyRxRz, 数值微分
        def R_of(th):
            ϑx, ϑy, ϑz = th
            Ry = lambda t: np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
            Rx = lambda t: np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
            Rz = lambda t: np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
            return Ry(th[1]) @ Rx(th[0]) @ Rz(th[2])
        R0 = R_of(t)
        tau = 1e-6
        Rdot = (R_of(t + tau * ϑdot) - R_of(t - tau * ϑdot)) / (2 * tau)
        omega_num = vee(Rdot @ R0.T)
        omega_an = eular_diff_to_w(t) @ ϑdot
        max_err = max(max_err, np.abs(omega_num - omega_an).max())
    print(f"  ω 数值 vs 解析 最大差 = {max_err:.3e}")
    assert max_err < TOL, "T_ϑ ϑ̇ 与旋转矩阵数值导数不一致!"
    # T_ϑ 可逆性
    ϑ = np.array([0.4, -0.3, 0.2])
    inv_err = np.abs(eular_diff_to_w(ϑ) @ np.linalg.inv(eular_diff_to_w(ϑ)) - np.eye(3)).max()
    assert inv_err < TOL, "T_ϑ 不可逆!"

    # ---- 2) 几何雅可比 J_w ----
    print("\n[验证2] 几何雅可比 J_w vs 数值")
    J_w = robot_jacobian_w(q, d, a, alpha, N)
    T0 = T_N0(q, d, a, alpha, N)
    t_e, R_e = T0[:3, 3], T0[:3, :3]
    h = 1e-6
    err_p = err_r = 0.0
    for i in range(1, N + 1):
        qp = q.copy(); qp[i] += h
        qm = q.copy(); qm[i] -= h
        Tp, Tm = T_N0(qp, d, a, alpha, N), T_N0(qm, d, a, alpha, N)
        dt = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
        dR = (Tp[:3, :3] - Tm[:3, :3]) / (2 * h)
        omega_i = vee(dR @ R_e.T)
        err_p = max(err_p, np.abs(J_w[:3, i - 1] - dt).max())
        err_r = max(err_r, np.abs(J_w[3:, i - 1] - omega_i).max())
    print(f"  平移块 ∂t/∂q 最大差 = {err_p:.3e}")
    print(f"  旋转块 vee(ṘRᵀ) 最大差 = {err_r:.3e}")
    assert err_p < TOL and err_r < TOL, "几何雅可比与数值不一致!"

    # ---- 3) 分析雅可比 J_a（端到端）----
    print("\n[验证3] 分析雅可比 J_a = ∂(robot_fk 输出)/∂q（数值）")
    x_e = robot_fk(q, d, a, alpha, N)
    J_a = robot_jacobian_a(q, d, a, alpha, x_e[3:], N)
    J_num = num_jac_fk(q, d, a, alpha, N)
    err_pos = np.abs(J_a[:3] - J_num[:3]).max()
    err_eul = np.abs(J_a[3:] - J_num[3:]).max()
    print(f"  位置行块 最大差 = {err_pos:.3e}")
    print(f"  欧拉角行块 最大差 = {err_eul:.3e}")
    assert err_pos < TOL and err_eul < TOL, "分析雅可比与 FK 数值导数不一致!"

    # ---- 4) 质心雅可比 J_{w,n} ----
    print("\n[验证4] 质心雅可比 J_{w,n} vs 数值（N=6，p_cents 随机）")
    p_cents = np.zeros((N + 1, 3))
    p_cents[1:] = rng.uniform(-0.2, 0.2, (N, 3))
    J_w_ns = robot_j_centroid(q, d, a, alpha, p_cents, N)
    max_err_p = max_err_r = 0.0
    for n in range(1, N + 1):
        Tn = T_N0(q, d, a, alpha, n)                      # 前 n 个连杆
        Rn = Tn[:3, :3]
        com_n = Tn[:3, :3] @ p_cents[n] + Tn[:3, 3]       # COM_n 在基座系
        for i in range(1, n + 1):
            qp = q.copy(); qp[i] += h
            qm = q.copy(); qm[i] -= h
            Tnp = T_N0(qp, d, a, alpha, n)
            Tnm = T_N0(qm, d, a, alpha, n)
            d_com = (Tnp[:3, :3] @ p_cents[n] + Tnp[:3, 3]
                     - (Tnm[:3, :3] @ p_cents[n] + Tnm[:3, 3])) / (2 * h)
            dR = (Tnp[:3, :3] - Tnm[:3, :3]) / (2 * h)
            omega_n_i = vee(dR @ Rn.T)
            max_err_p = max(max_err_p, np.abs(J_w_ns[n, :3, i - 1] - d_com).max())
            max_err_r = max(max_err_r, np.abs(J_w_ns[n, 3:, i - 1] - omega_n_i).max())
        # n 之后列应为 0（仅当存在后续列时检查）
        if n < N:
            assert np.abs(J_w_ns[n, :, n:]).max() == 0.0, "质心雅可比 n 之后列非零!"
    print(f"  位置块 ∂COM_n/∂q 最大差 = {max_err_p:.3e}")
    print(f"  旋转块 vee(Ṙ_n R_nᵀ) 最大差 = {max_err_r:.3e}")
    print("  n 之后列全零 ✔")
    assert max_err_p < TOL and max_err_r < TOL, "质心雅可比与数值不一致!"

    # ---- 5) 2 连杆平面臂闭式雅可比 ----
    print("\n[验证5] 2连杆平面臂闭式雅可比")
    L1, L2, θ1, θ2 = 1.0, 0.8, 0.6, -0.4
    q2 = np.array([0.0, θ1, θ2])
    d2 = np.zeros(3); a2 = np.array([0.0, L1, L2]); alpha2 = np.zeros(3)
    J_w2 = robot_jacobian_w(q2, d2, a2, alpha2, N=2)
    J_exp = np.array([
        [-L1 * np.sin(θ1) - L2 * np.sin(θ1 + θ2), -L2 * np.sin(θ1 + θ2)],
        [L1 * np.cos(θ1) + L2 * np.cos(θ1 + θ2), L2 * np.cos(θ1 + θ2)],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
    ])
    err = np.abs(J_w2 - J_exp).max()
    print(f"  J_w 闭式误差 = {err:.3e}")
    assert err < TOL, "平面臂闭式雅可比不匹配!"
    # 平面臂 ϑ=[0,0,θ1+θ2] 时 T_ϑ = I，故 J_a = J_w
    x2 = robot_fk(q2, d2, a2, alpha2, N=2)
    J_a2 = robot_jacobian_a(q2, d2, a2, alpha2, x2[3:], N=2)
    err_a = np.abs(J_a2 - J_exp).max()
    print(f"  J_a (T_ϑ=I 情形) vs J_w 最大差 = {err_a:.3e}")
    assert err_a < TOL, "平面臂 J_a 与 J_w 不一致!"

    print("\n全部验证通过 ✔")


if __name__ == "__main__":
    main()
