# -*- coding: utf-8 -*-
"""《Math Toolbox for Robotics》Part II 微分运动学雅可比解析解（chap2_3_diffk_1.tex）。

核心实现，与书 `algo:robot_ja` / `algo:robot_j_centroid` 算法框及正文代码块一致。
数组索引对齐约定（见 2index.tex「数组索引」）：长度为 N+1、下标 1..N 与数学记号一致，[0] 不用。

- eular_diff_to_w(eular)：ZXY 欧拉角微分 -> 角速度变换矩阵 T_ϑ（ω = T_ϑ·ϑ̇）
  约定与 chap2_3_diffk_1.tex 的 T_ϑ 公式一致：R = R_y(ϑy)R_x(ϑx)R_z(ϑz)
- robot_jacobian_w(q, d, a, alpha, N=6)：几何雅可比 J_w（J_w 的解析构造，供验证与复用）
- robot_jacobian_a(q, d, a, alpha, eular_e, N=6)：分析雅可比 J_a = diag{I_3, T_ϑ^{-1}} J_w
- robot_j_centroid(q, d, a, alpha, p_cents, N=6)：质心雅可比 J_{w,n}, n=1..N

完整验证脚本见 test_jacobian.py。
"""
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "chap2_2_kinematcs"))
from code_fk import calc_T_n_to_last, robot_fk, rot_to_eular


def eular_diff_to_w(eular):
    # ZXY: R = R_y(ϑy) R_x(ϑx) R_z(ϑz)，ω = T_ϑ·[ϑ̇x, ϑ̇y, ϑ̇z]^T
    # T_ϑ 列：[R_y[1,0,0], [0,1,0], R_y R_x[0,0,1]]
    ϑx, ϑy, ϑz = eular
    sx, cx = np.sin(ϑx), np.cos(ϑx)
    sy, cy = np.sin(ϑy), np.cos(ϑy)
    return np.array([
        [cy, 0, sy * cx],
        [0, 1, -sx],
        [-sy, 0, cy * cx],
    ])


def robot_jacobian_w(q, d, a, alpha, N=6):
    """几何雅可比 J_w：v_{ew} = [ṗ_e; ω_e] = J_w q̇，列为 [z_{n-1}^b × p_{n-1,N}^b; z_{n-1}^b]。"""
    assert q.shape == (N + 1,) and d.shape == (N + 1,)
    assert a.shape == (N + 1,) and alpha.shape == (N + 1,)
    T_n_to_bases = np.zeros([N + 1, 4, 4])
    T_n_to_bases[0] = np.eye(4)
    T_n_to_lasts = np.zeros([N + 1, 4, 4])
    z_n_bs = np.zeros([N, 3])
    for n in range(1, N + 1):
        z_n_bs[n - 1] = T_n_to_bases[n - 1][:3, 2]
        T_n_to_lasts[n] = calc_T_n_to_last(q[n], d[n], a[n], alpha[n])
        T_n_to_bases[n] = T_n_to_bases[n - 1] @ T_n_to_lasts[n]
    T_N_to_ns = np.zeros([N + 1, 4, 4])
    T_N_to_ns[N] = np.eye(4)
    p_narm_in_Bs = np.zeros([N, 3])
    for n in range(N, 0, -1):
        T_N_to_ns[n - 1] = T_n_to_lasts[n] @ T_N_to_ns[n]
        t_N_in_nlast = T_N_to_ns[n - 1, :3, 3]
        R_nlast_to_b = T_n_to_bases[n - 1, :3, :3]
        p_narm_in_Bs[n - 1] = R_nlast_to_b @ t_N_in_nlast
    J_w = np.zeros((6, N))
    for n in range(N):
        J_w[:3, n] = np.cross(z_n_bs[n], p_narm_in_Bs[n])
        J_w[3:, n] = z_n_bs[n]
    return J_w


def robot_jacobian_a(q, d, a, alpha, eular_e, N=6):
    assert q.shape == (N + 1,) and d.shape == (N + 1,)
    assert a.shape == (N + 1,) and alpha.shape == (N + 1,)
    # 先构造几何雅可比 J_w（与 algo:robot_ja 算法框一致）
    z_n_bs = np.zeros([N, 3])
    T_n_to_lasts = np.zeros([N + 1, 4, 4])
    T_n_to_bases = np.zeros([N + 1, 4, 4])
    T_n_to_bases[0] = np.eye(4)
    for n in range(1, N + 1):
        z_n_bs[n - 1] = T_n_to_bases[n - 1][:3, 2]
        T_n_to_last = calc_T_n_to_last(q[n], d[n], a[n], alpha[n])
        T_n_to_lasts[n, :, :] = T_n_to_last
        T_n_to_bases[n] = T_n_to_bases[n - 1] @ T_n_to_last
    T_N_to_ns = np.zeros([N + 1, 4, 4])
    T_N_to_ns[N] = np.eye(4)
    p_narm_in_Bs = np.zeros([N, 3])
    for n in range(N, 0, -1):
        T_N_to_ns[n - 1] = T_n_to_lasts[n] @ T_N_to_ns[n]
        t_N_in_nlast = T_N_to_ns[n - 1, :3, 3]
        R_nlast_to_b = T_n_to_bases[n - 1, :3, :3]
        p_narm_in_Bs[n - 1] = R_nlast_to_b @ t_N_in_nlast
    J_w = np.zeros((6, N))
    for n in range(N):
        J_w[:3, n] = np.cross(z_n_bs[n], p_narm_in_Bs[n])
        J_w[3:, n] = z_n_bs[n]
    # 分析雅可比 J_a = diag{I_3, T_ϑ^{-1}} J_w
    T_theta = eular_diff_to_w(eular_e)
    J_a = np.zeros((6, 6))
    J_a[:3, :3] = np.eye(3)
    J_a[3:, 3:] = np.linalg.inv(T_theta)
    return J_a @ J_w


def robot_j_centroid(q, d, a, alpha, p_cents, N=6):
    assert q.shape == (N + 1,) and d.shape == (N + 1,)
    assert a.shape == (N + 1,) and alpha.shape == (N + 1,)
    assert p_cents.shape == (N + 1, 3)
    T_n_to_bases = np.zeros([N + 1, 4, 4])
    T_n_to_bases[0] = np.eye(4)
    z_n_bs = np.zeros([N + 1, 3])
    z_n_bs[0] = T_n_to_bases[0, :3, 2]          # 基座系 z 轴 = [0,0,1]
    T_n_to_lasts = np.zeros([N + 1, 4, 4])
    J_w_ns = np.zeros([N + 1, 6, N])
    for n in range(1, N + 1):
        T_n_to_lasts[n] = calc_T_n_to_last(q[n], d[n], a[n], alpha[n])
        T_n_to_bases[n] = T_n_to_bases[n - 1] @ T_n_to_lasts[n]
        z_n_bs[n] = T_n_to_bases[n, :3, 2]
        T_n_to_ms = np.zeros([n + 1, 4, 4])
        T_n_to_ms[n] = np.eye(4)
        for m in range(n - 1, -1, -1):
            T_n_to_ms[m] = T_n_to_lasts[m + 1] @ T_n_to_ms[m + 1]
        for m in range(n):
            p_narm_in_m = T_n_to_ms[m, :3, :3] @ p_cents[n] + T_n_to_ms[m, :3, 3]
            p_narm_in_b = T_n_to_bases[m, :3, :3] @ p_narm_in_m
            J_w_ns[n, :3, m] = np.cross(z_n_bs[m], p_narm_in_b)
            J_w_ns[n, 3:, m] = z_n_bs[m]
    return J_w_ns
