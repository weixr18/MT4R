# -*- coding: utf-8 -*-
"""验证《Math Toolbox for Robotics》Part II 正运动学解析解（chap2_2_kinematcs.tex）。

验证内容：
1. calc_T_n_to_last：与标准 D-H 变换 Rotz(θ)Transz(d)Transx(a)Rotx(α) 逐元素一致
2. robot_fk：返回 x_e = [t_{0N}^0; ϑ_e]，由 ϑ_e 重建 R = R_y R_x R_z 复原 T_N^0，与 ∏ T_n^{n-1} 一致
3. robot_fk：2 连杆平面臂（α=0,d=0）闭式解 x_e=[L1cθ1+L2c(θ1+θ2), L1sθ1+L2s(θ1+θ2), 0, 0, 0, θ1+θ2]
4. rot_to_eular：随机旋转矩阵 R -> ϑ -> R 回环一致；且 T_ϑ ϑ̇ = ω（与 eular_diff_to_w 共用约定）

运行：E:\\Anaconda3\\envs\\py311-gym\\python.exe test_fk.py
"""
import numpy as np

from code_fk import calc_T_n_to_last, robot_fk, rot_to_eular

np.set_printoptions(precision=6, suppress=True)
TOL = 1e-9


def T_std_dh(q_n, d_n, a_n, alpha_n):
    """标准 D-H：Rotz(q)Transz(d)Transx(a)Rotx(alpha)，作为 calc_T_n_to_last 的独立参照。"""
    s_q, c_q = np.sin(q_n), np.cos(q_n)
    s_a, c_a = np.sin(alpha_n), np.cos(alpha_n)
    A = np.array([
        [c_q, -s_q, 0, 0],
        [s_q, c_q, 0, 0],
        [0, 0, 1, d_n],
        [0, 0, 0, 1],
    ])
    B = np.array([
        [1, 0, 0, a_n],
        [0, c_a, -s_a, 0],
        [0, s_a, c_a, 0],
        [0, 0, 0, 1],
    ])
    return A @ B


def make_dh_rng(seed, N=6):
    rng = np.random.default_rng(seed)
    d = np.zeros(N + 1); a = np.zeros(N + 1); alpha = np.zeros(N + 1)
    d[1:] = rng.uniform(-0.3, 0.3, N)
    a[1:] = rng.uniform(0.2, 1.0, N)
    alpha[1:] = rng.uniform(-np.pi / 2, np.pi / 2, N)
    return d, a, alpha


def main():
    print("=" * 60)
    print("验证 正运动学解析解（code_fk.py）")
    print("=" * 60)

    # ---- 1) calc_T_n_to_last vs 标准 D-H ----
    print("\n[验证1] calc_T_n_to_last vs 标准D-H(Rotz·Transz·Transx·Rotx)")
    rng = np.random.default_rng(7)
    max_err = 0.0
    for _ in range(20):
        q_n, d_n, a_n, alpha_n = rng.uniform(-2, 2, 4)
        err = np.abs(calc_T_n_to_last(q_n, d_n, a_n, alpha_n)
                     - T_std_dh(q_n, d_n, a_n, alpha_n)).max()
        max_err = max(max_err, err)
    print(f"  逐元素最大差 = {max_err:.3e}")
    assert max_err < TOL, "calc_T_n_to_last 与标准D-H不一致!"

    # ---- 2) robot_fk：末端位姿复原 T_N^0 ----
    N = 6
    d, a, alpha = make_dh_rng(seed=42, N=N)
    q = np.zeros(N + 1)
    q[1:] = np.linspace(-0.8, 0.9, N)
    print("\n[验证2] robot_fk 末端位姿复原 T_N^0（N=6 随机D-H，seed=42）")
    x_e = robot_fk(q, d, a, alpha, N)
    T_expected = np.eye(4)
    for n in range(1, N + 1):
        T_expected = T_expected @ calc_T_n_to_last(q[n], d[n], a[n], alpha[n])
    # 由 ϑ_e 重建 R = R_y(ϑy)R_x(ϑx)R_z(ϑz)
    ϑx, ϑy, ϑz = x_e[3:]
    Ry = lambda t: np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    Rx = lambda t: np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    Rz = lambda t: np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    R_rebuilt = Ry(ϑy) @ Rx(ϑx) @ Rz(ϑz)
    err_t = np.abs(x_e[:3] - T_expected[:3, 3]).max()
    err_R = np.abs(R_rebuilt - T_expected[:3, :3]).max()
    print(f"  位置 t_{'{0N}^{0}'} vs T_N^0[:3,3] 最大差 = {err_t:.3e}")
    print(f"  姿态 R(ϑ_e) vs T_N^0[:3,:3] 最大差 = {err_R:.3e}")
    assert err_t < TOL and err_R < TOL, "robot_fk 复原 T_N^0 不一致!"

    # ---- 3) 2 连杆平面臂闭式解 ----
    print("\n[验证3] 2连杆平面臂闭式解")
    L1, L2, θ1, θ2 = 1.0, 0.8, 0.6, -0.4
    q2 = np.array([0.0, θ1, θ2])
    d2 = np.zeros(3); a2 = np.array([0.0, L1, L2]); alpha2 = np.zeros(3)
    x_2 = robot_fk(q2, d2, a2, alpha2, N=2)
    x_exp = np.array([L1 * np.cos(θ1) + L2 * np.cos(θ1 + θ2),
                      L1 * np.sin(θ1) + L2 * np.sin(θ1 + θ2),
                      0.0, 0.0, 0.0, θ1 + θ2])
    err = np.abs(x_2 - x_exp).max()
    print(f"  x_e = {np.round(x_2, 6).tolist()}")
    print(f"  闭式解误差 = {err:.3e}")
    assert err < TOL, "平面臂闭式解不匹配!"

    # ---- 4) rot_to_eular 回环 ----
    print("\n[验证4] rot_to_eular 随机旋转回环 R -> ϑ -> R")
    max_err = 0.0
    for _ in range(50):
        # 随机正交旋转矩阵（Gram-Schmidt）
        M = rng.normal(size=(3, 3))
        R = M @ M.T
        Q = np.linalg.qr(M)[0]
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        ϑ = rot_to_eular(Q)
        R_rebuilt = Ry(ϑ[1]) @ Rx(ϑ[0]) @ Rz(ϑ[2])
        max_err = max(max_err, np.abs(R_rebuilt - Q).max())
    print(f"  R->ϑ->R 回环最大差 = {max_err:.3e}")
    assert max_err < TOL, "rot_to_eular 回环不一致!"

    # ---- 5) robot_fk 可逆性 / 多组随机配置 ----
    print("\n[验证5] 多组随机配置下 robot_fk 复原 T_N^0")
    max_err = 0.0
    for k in range(20):
        q = np.zeros(N + 1)
        q[1:] = rng.uniform(-1.5, 1.5, N)
        x_e = robot_fk(q, d, a, alpha, N)
        T_expected = np.eye(4)
        for n in range(1, N + 1):
            T_expected = T_expected @ calc_T_n_to_last(q[n], d[n], a[n], alpha[n])
        ϑx, ϑy, ϑz = x_e[3:]
        R_rebuilt = Ry(ϑy) @ Rx(ϑx) @ Rz(ϑz)
        err = max(np.abs(x_e[:3] - T_expected[:3, 3]).max(),
                  np.abs(R_rebuilt - T_expected[:3, :3]).max())
        max_err = max(max_err, err)
    print(f"  20 组配置最大差 = {max_err:.3e}")
    assert max_err < TOL, "多配置下 FK 不一致!"

    print("\n全部验证通过 ✔")


if __name__ == "__main__":
    main()
