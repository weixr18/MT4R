# -*- coding: utf-8 -*-
"""《Math Toolbox for Robotics》Part II 逆运动学数值求解（chap2_3_diffk_2.tex）。

核心实现，与书 `algo:robot-ik-gd` / `algo:robot-ik-gauss-newton` / `algo:robot-ik-lm`
算法框及正文代码块一致（修复 bug 后）。数组索引对齐约定（见 2index.tex「数组索引」）：
长度为 N+1、下标 1..N 与数学记号一致，[0] 不用。

- robot_ik_gd：梯度下降法 IK（雅可比转置法），一阶收敛、慢，步长 alpha_p 需手工调
- robot_ik_gauss_newton：高斯-牛顿法 IK（雅可比伪逆法），二次收敛
- robot_ik_lm：阻尼最小二乘法 IK（L-M 法），标准 Marquardt 增益比 + 自适应阻尼，最鲁棒

三个算法均以 x_e = [t_{0N}^0; ϑ_e]（6 维，robot_fk 输出）为给定末端位姿，
复用 robot_fk / robot_jacobian_a 迭代求解关节向量 q（下标 1..N，q[0] 不用）。

完整验证脚本见 test_ik.py。
"""
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "chap2_2_kinematcs"))
from code_fk import robot_fk
from code_jacobian import robot_jacobian_a


def robot_ik_gd(x_e, q_0, d, a, alpha, N=6, H=None, epsilon=1e-4, alpha_p=1e-2,
                max_iter=100000):
    """梯度下降法 IK（雅可比转置法）。

    每次迭代 ∇J = -J_a^T H (x_e - x_{e,k})，q_{k+1} = q_k - α_p·∇J。
    一阶方法，收敛慢；alpha_p 过大可能发散，需按问题手工调节；max_iter 防止不收敛挂起。
    """
    assert q_0.shape == (N + 1,) and d.shape == (N + 1,)
    assert a.shape == (N + 1,) and alpha.shape == (N + 1,)
    if H is None:
        H = np.eye(6)                # H ∈ R^{6×6}（误差 e = x_e - x_{e,k} ∈ R^6）
    else:
        assert H.shape == (6, 6)
    q_k = q_0.copy()
    for _ in range(max_iter):
        x_ek = robot_fk(q_k, d, a, alpha, N)
        J_a = robot_jacobian_a(q_k, d, a, alpha, x_ek[3:], N)
        grad = -J_a.T @ H @ (x_e - x_ek)
        if np.linalg.norm(x_ek - x_e) < epsilon:
            break
        q_k[1:] = q_k[1:] - alpha_p * grad   # q_k[0] 为冗余前导元素，不更新
    return q_k


def robot_ik_gauss_newton(x_e, q_0, d, a, alpha, N=6, H=None, epsilon=1e-4,
                          max_iter=10000):
    """高斯-牛顿法 IK（雅可比伪逆法）。

    e = x_e - x_{e,k}，δq = (J_a^T H J_a)^{-1} J_a^T H e，q_{k+1} = q_k + δq。
    二次收敛；H_n 接近奇异（运动学奇异位形）时需改用阻尼版（L-M）；max_iter 防挂起。
    """
    assert q_0.shape == (N + 1,) and d.shape == (N + 1,)
    assert a.shape == (N + 1,) and alpha.shape == (N + 1,)
    if H is None:
        H = np.eye(6)
    else:
        assert H.shape == (6, 6)
    q_k = q_0.copy()
    for _ in range(max_iter):
        x_ek = robot_fk(q_k, d, a, alpha, N)
        J_a = robot_jacobian_a(q_k, d, a, alpha, x_ek[3:], N)
        if np.linalg.norm(x_ek - x_e) < epsilon:
            break
        g_n = J_a.T @ H @ (x_e - x_ek)
        H_n = J_a.T @ H @ J_a
        q_k[1:] = q_k[1:] + np.linalg.inv(H_n) @ g_n
    return q_k


def robot_ik_lm(x_e, q_0, d, a, alpha, N=6, H=None, epsilon=1e-4,
                rmin=0.25, rmax=0.75, lambda_0=1, max_iter=10000):
    """阻尼最小二乘法 IK（L-M 法，标准 Marquardt 增益比）。

    e = x_{e,k} - x_e，δq = -(J_a^T H J_a + λI)^{-1} J_a^T H e，q_{k+1} = q_k + δq。
    增益比 ρ = (f_k - f_{k+1}) / (f_k - m_k)，m_k = f_k + 2g_k^T δq + δq^T (J_a^T H J_a) δq
    为二次模型代价；ρ > rmax 说明模型好 → λ 减半，ρ < rmin 说明模型差 → λ 加倍。
    仅接受使代价下降的步（f_{k+1} < f_k）。λ_0 较大时表现接近梯度下降，较小时接近高斯-牛顿。
    max_iter 防止不收敛挂起。
    """
    assert q_0.shape == (N + 1,) and d.shape == (N + 1,)
    assert a.shape == (N + 1,) and alpha.shape == (N + 1,)
    if H is None:
        H = np.eye(6)
    else:
        assert H.shape == (6, 6)
    q_k, lambda_k = q_0.copy(), lambda_0
    for _ in range(max_iter):
        x_ek = robot_fk(q_k, d, a, alpha, N)
        J_a = robot_jacobian_a(q_k, d, a, alpha, x_ek[3:], N)
        delta_xe = x_ek - x_e
        if np.linalg.norm(delta_xe) < epsilon:
            break
        g_l = J_a.T @ H @ delta_xe
        H_n = J_a.T @ H @ J_a
        delta_q = -np.linalg.inv(H_n + lambda_k * np.eye(N)) @ g_l
        q_new = q_k.copy()
        q_new[1:] = q_new[1:] + delta_q
        f_k = delta_xe @ H @ delta_xe
        x_ek_new = robot_fk(q_new, d, a, alpha, N)
        delta_xe_new = x_ek_new - x_e
        f_new = delta_xe_new @ H @ delta_xe_new
        m_k = f_k + 2 * g_l @ delta_q + delta_q @ H_n @ delta_q   # 二次模型代价
        rho = (f_k - f_new) / (f_k - m_k)
        if rho > rmax:
            lambda_k = lambda_k / 2
        elif rho < rmin:
            lambda_k = 2 * lambda_k
        if f_new < f_k:                 # 仅接受使代价下降的步
            q_k = q_new
    return q_k
