# -*- coding: utf-8 -*-
"""验证《Math Toolbox for Robotics》Part II 机器人动力学解析解（chap2_4_dynamics.tex）。

独立参照 = 直接运动学中心差分数值导数 / 2 连杆平面臂教科书闭式解。

1. B：2 连杆平面臂惯量矩阵与教科书闭式解逐元素一致；平面臂 g(q)=0
2. g：g(q) = ∂U/∂q，U(q) = -Σ m_n g0^T p_n(q)（p_n 由正运动学直接计算，独立于雅可比）
3. B：动能恒等式 ½q̇ᵀBq̇ = T(q,q̇)，T 由各连杆质心速度/角速度数值微分直接求和（独立于 B）
4. C：恒等式 C(q,q̇)q̇ = Ḃ(q)q̇ - (∂T/∂q)ᵀ（∂T/∂q = ½q̇ᵀ∂B/∂qᵀ 解析形式，验证 Christoffel 构造）
5. 性质：N = Ḃ - 2C 为反对称阵（书中性质，xᵀNx = 0）
6. 端到端：τ = B q̈ + C q̇ + g 与数值欧拉-拉格朗日 τ = d/dt(∂T/∂q̇) - ∂T/∂q + ∂U/∂q 一致
   （沿匀速轨迹 q(t)=q+q̇t 对动量做时间差分；嵌套二阶差分，容差放宽为 1e-3）

运行：E:\\Anaconda3\\envs\\py311-gym\\python.exe test_dynamics.py
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
from code_dynamics import robot_B, robot_dyn

np.set_printoptions(precision=6, suppress=True)
TOL = 1e-6          # 一阶/闭式对比容差
TOL_EL = 1e-3       # 端到端欧拉-拉格朗日（嵌套差分）容差
H = 1e-6            # q 方向中心差分步长


def vee(R_hat):
    return np.array([R_hat[2, 1], R_hat[0, 2], R_hat[1, 0]])


def com_link(q, d, a, alpha, p_cents, n, N):
    """连杆 n 质心在基坐标系的位置（由正运动学直接计算，独立于雅可比）。"""
    T = np.eye(4)
    for i in range(1, n + 1):
        T = T @ calc_T_n_to_last(q[i], d[i], a[i], alpha[i])
    return T[:3, :3] @ p_cents[n] + T[:3, 3]


def R_link(q, d, a, alpha, n, N):
    """连杆 n 的旋转矩阵 R_n^b。"""
    T = np.eye(4)
    for i in range(1, n + 1):
        T = T @ calc_T_n_to_last(q[i], d[i], a[i], alpha[i])
    return T[:3, :3]


def T_direct(q, qdot, d, a, alpha, m, p_cents, I_inn, N, eps=1e-6):
    """独立动能：T = Σ_n ½m v_nᵀv_n + ½ω_nᵀ(R I Rᵀ)ω_n。

    v_n、ω_n 由 p_n(q)、R_n(q) 沿速度方向 eps*qdot 的数值导数得到，不依赖任何雅可比/B。
    """
    qp = q + eps * qdot
    qm = q - eps * qdot
    T = 0.0
    for n in range(1, N + 1):
        Rn = R_link(q, d, a, alpha, n, N)
        v = (com_link(qp, d, a, alpha, p_cents, n, N)
             - com_link(qm, d, a, alpha, p_cents, n, N)) / (2 * eps)
        Rdot = (R_link(qp, d, a, alpha, n, N)
                - R_link(qm, d, a, alpha, n, N)) / (2 * eps)
        omega = vee(Rdot @ Rn.T)
        T += 0.5 * m[n] * (v @ v) + 0.5 * omega @ (Rn @ I_inn[n] @ Rn.T) @ omega
    return T


def U_pot(q, d, a, alpha, m, p_cents, g0, N):
    """势能 U(q) = -Σ_n m_n g0^T p_n(q)。"""
    return -sum(m[n] * g0 @ com_link(q, d, a, alpha, p_cents, n, N)
                for n in range(1, N + 1))


def make_robot(rng, N=6):
    """随机 N 自由度转动关节臂（DH 取典型范围随机值）+ 随机连杆参数。"""
    d = np.zeros(N + 1); a = np.zeros(N + 1); alpha = np.zeros(N + 1)
    d[1:] = rng.uniform(-0.3, 0.3, N)
    a[1:] = rng.uniform(0.2, 1.0, N)
    alpha[1:] = rng.uniform(-np.pi / 2, np.pi / 2, N)
    m = np.zeros(N + 1)
    m[1:] = rng.uniform(0.5, 3.0, N)
    p_cents = np.zeros((N + 1, 3))
    p_cents[1:] = rng.uniform(-0.2, 0.2, (N, 3))
    I_inn = np.zeros((N + 1, 3, 3))
    for n in range(1, N + 1):
        # 随机对称正定惯量矩阵（在连杆系下为常值）
        A = rng.uniform(0.05, 0.4, (3, 3))
        I_inn[n] = A @ A.T + np.eye(3) * 1e-2
    return d, a, alpha, m, p_cents, I_inn


def main():
    print("=" * 64)
    print("验证 机器人动力学解析解（code_dynamics.py）")
    print("=" * 64)
    rng = np.random.default_rng(42)
    N = 6
    d, a, alpha, m, p_cents, I_inn = make_robot(rng, N)
    g0 = np.array([0.0, 0.0, -9.81])
    q = np.zeros(N + 1)
    q[1:] = np.array([0.6, -0.4, 0.3, -0.5, 0.2, -0.3])   # 避开万向锁
    qdot = np.zeros(N + 1)
    qdot[1:] = rng.uniform(-0.8, 0.8, N)
    qddot = np.zeros(N + 1)
    qddot[1:] = rng.uniform(-0.6, 0.6, N)

    # ---- 1) 2 连杆平面臂 B 闭式解 + g=0 ----
    print("\n[验证1] 2连杆平面臂惯量矩阵 B 闭式解（含角向惯量 I1/I2）")
    L1, L2 = 1.0, 0.8
    m1, m2, I1, I2 = 2.0, 1.5, 0.1, 0.08
    θ1, θ2 = 0.6, -0.4
    q2 = np.array([0.0, θ1, θ2])
    d2 = np.zeros(3); a2 = np.array([0.0, L1, L2]); alpha2 = np.zeros(3)
    m2v = np.array([0.0, m1, m2])
    # D-H 第 n 号连杆系原点位于关节 n+1 处，质心（连杆中点）在连杆系下坐标为 -L_n/2
    pc2 = np.array([[0, 0, 0], [-L1 / 2, 0, 0], [-L2 / 2, 0, 0]])
    I2v = np.zeros((3, 3, 3))
    I2v[1] = np.diag([0.0, 0.0, I1])
    I2v[2] = np.diag([0.0, 0.0, I2])
    B2 = robot_B(q2, d2, a2, alpha2, m2v, pc2, I2v, N=2)
    c2 = np.cos(θ2)
    B_exp = np.array([
        [m1 * L1 ** 2 / 4 + I1 + m2 * (L1 ** 2 + L2 ** 2 / 4 + L1 * L2 * c2) + I2,
         m2 * (L2 ** 2 / 4 + L1 * L2 / 2 * c2) + I2],
        [m2 * (L2 ** 2 / 4 + L1 * L2 / 2 * c2) + I2,
         m2 * L2 ** 2 / 4 + I2],
    ])
    err_B2 = np.abs(B2 - B_exp).max()
    print(f"  B vs 闭式解 最大差 = {err_B2:.3e}")
    assert err_B2 < TOL, "平面臂 B 闭式解不匹配!"
    _, _, g2 = robot_dyn(q2, q2 * 0.0, d2, a2, alpha2, m2v, pc2, I2v, g0, N=2)
    print(f"  平面臂重力项 max|g| = {np.abs(g2).max():.3e}（应≈0）")
    assert np.abs(g2).max() < TOL, "平面臂 g 应为 0!"

    # ---- 2) g(q) = ∂U/∂q ----
    print("\n[验证2] 重力项 g(q) = ∂U/∂q（U 由正运动学直接计算）")
    B, C, g = robot_dyn(q, qdot, d, a, alpha, m, p_cents, I_inn, g0, N)
    g_num = np.zeros(N)
    for i in range(1, N + 1):
        ei = np.zeros(N + 1); ei[i] = H
        dU = (U_pot(q + ei, d, a, alpha, m, p_cents, g0, N)
              - U_pot(q - ei, d, a, alpha, m, p_cents, g0, N)) / (2 * H)
        g_num[i - 1] = dU
    err_g = np.abs(g - g_num).max()
    print(f"  g vs ∂U/∂q 最大差 = {err_g:.3e}")
    assert err_g < TOL, "g(q) 与 ∂U/∂q 不一致!"

    # ---- 3) 动能恒等式 ½q̇ᵀBq̇ = T_direct ----
    print("\n[验证3] 动能恒等式 ½q̇ᵀBq̇ = T(q,q̇)（T 独立数值求值）")
    max_err_T = 0.0
    for _ in range(5):
        qr = np.zeros(N + 1); qr[1:] = rng.uniform(-1.2, 1.2, N)
        qr_dot = np.zeros(N + 1); qr_dot[1:] = rng.uniform(-0.8, 0.8, N)
        Bq = robot_B(qr, d, a, alpha, m, p_cents, I_inn, N)
        T_via_B = 0.5 * qr_dot[1:] @ Bq @ qr_dot[1:]
        T_dir = T_direct(qr, qr_dot, d, a, alpha, m, p_cents, I_inn, N)
        max_err_T = max(max_err_T, abs(T_via_B - T_dir))
    print(f"  ½q̇ᵀBq̇ vs T_direct 最大差 = {max_err_T:.3e}")
    assert max_err_T < TOL, "动能恒等式不成立!"

    # ---- 4) C 恒等式 C q̇ = Ḃ q̇ - (∂T/∂q)ᵀ（∂T/∂q = ½q̇ᵀ∂B/∂qᵀ 解析形式）----
    print("\n[验证4] C 恒等式 C(q,q̇)q̇ = Ḃ(q)q̇ - (∂T/∂q)ᵀ")
    dBdq = np.zeros((N, N, N))              # dBdq[k,i,j] = ∂b_ij/∂q_{k+1}
    for k in range(1, N + 1):
        ek = np.zeros(N + 1); ek[k] = H
        dBdq[k - 1] = (robot_B(q + ek, d, a, alpha, m, p_cents, I_inn, N)
                       - robot_B(q - ek, d, a, alpha, m, p_cents, I_inn, N)) / (2 * H)
    Bdot_q = np.zeros(N)                    # Ḃ q̇ = Σ_k ∂B/∂q_k·q̇_k·q̇
    for k in range(1, N + 1):
        Bdot_q += qdot[k] * (dBdq[k - 1] @ qdot[1:])
    dTdq = np.array([0.5 * qdot[1:] @ (dBdq[i - 1] @ qdot[1:]) for i in range(1, N + 1)])
    lhs = C @ qdot[1:]                       # C q̇（算法输出）
    rhs = Bdot_q - dTdq                      # Ḃ q̇ - (∂T/∂q)ᵀ
    err_C = np.abs(lhs - rhs).max()
    print(f"  C q̇ vs Ḃ q̇ - (∂T/∂q)ᵀ 最大差 = {err_C:.3e}")
    assert err_C < TOL, "C 矩阵恒等式不成立!"

    # ---- 5) 反对称性质 N = Ḃ - 2C ----
    print("\n[验证5] 反对称性质 N = Ḃ - 2C")
    Bdot = np.zeros((N, N))
    for k in range(1, N + 1):
        Bdot += qdot[k] * dBdq[k - 1]
    N_mat = Bdot - 2.0 * C
    err_skew = np.abs(N_mat + N_mat.T).max()
    x = rng.uniform(-1, 1, N)
    qform = abs(x @ N_mat @ x)
    print(f"  |N + Nᵀ| 最大差 = {err_skew:.3e}")
    print(f"  |xᵀ N x| = {qform:.3e}（应≈0）")
    assert err_skew < TOL, "N = Ḃ - 2C 不是反对称阵!"

    # ---- 6) 端到端欧拉-拉格朗日（沿匀速轨迹 q(t)=q+q̇t，q̈=0）----
    print("\n[验证6] 端到端 τ = B q̈ + C q̇ + g vs 数值欧拉-拉格朗日（容差 1e-3）")
    # 嵌套差分的二阶量精度受限，容差放宽为 1e-3；τ/h2 步长经调参取折中
    qddot[:] = 0.0
    tau_alg = B @ qddot[1:] + C @ qdot[1:] + g
    tau_el = np.zeros(N)
    tau_traj = 1e-2                        # 轨迹时间差分步长
    H2 = 1e-4                              # 嵌套 T_direct 的 q 方向差分步长
    def momentum_i(qq, qqd, i, hh=H2):
        ei = np.zeros(N + 1); ei[i] = hh
        return (T_direct(qq, qqd + ei, d, a, alpha, m, p_cents, I_inn, N)
                - T_direct(qq, qqd - ei, d, a, alpha, m, p_cents, I_inn, N)) / (2 * hh)
    for i in range(1, N + 1):
        def p_i(t):                        # 轨迹 q(t) = q + q̇t（匀速）
            qt = q + qdot * t
            return momentum_i(qt, qdot, i)
        dp_i = (p_i(tau_traj) - p_i(-tau_traj)) / (2 * tau_traj)
        ei = np.zeros(N + 1); ei[i] = H2
        dT = (T_direct(q + ei, qdot, d, a, alpha, m, p_cents, I_inn, N)
              - T_direct(q - ei, qdot, d, a, alpha, m, p_cents, I_inn, N)) / (2 * H2)
        ei = np.zeros(N + 1); ei[i] = H
        dU = (U_pot(q + ei, d, a, alpha, m, p_cents, g0, N)
              - U_pot(q - ei, d, a, alpha, m, p_cents, g0, N)) / (2 * H)
        tau_el[i - 1] = dp_i - dT + dU
    err_el = np.abs(tau_alg - tau_el).max()
    print(f"  τ_alg vs τ_EL 最大差 = {err_el:.3e}")
    assert err_el < TOL_EL, "动力学方程与数值欧拉-拉格朗日不一致!"

    print("\n全部验证通过 ✔")


if __name__ == "__main__":
    main()
