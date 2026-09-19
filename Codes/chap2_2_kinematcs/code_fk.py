# -*- coding: utf-8 -*-
"""《Math Toolbox for Robotics》Part II 正运动学解析解（chap2_2_kinematcs.tex）。

核心实现，与书 `algo:robot_fk` 算法框及正文代码块一致。
数组索引对齐约定（见 2index.tex「数组索引」）：长度为 N+1、下标 1..N 与数学记号一致，[0] 不用。

- calc_T_n_to_last(q_n, d_n, a_n, alpha_n)：D-H 连杆变换 T_n^{n-1}（4x4 齐次矩阵）
- rot_to_eular(R)：旋转矩阵 -> ZXY 欧拉角 [ϑx, ϑy, ϑz]
  约定（与 chap2_3_diffk_1.tex 的 T_ϑ 公式一致）：R = R_y(ϑy) R_x(ϑx) R_z(ϑz)
- robot_fk(q, d, a, alpha, N=6)：正运动学解析解，返回末端位姿 x_e = [t_{0N}^0; ϑ_e]（6 维）

完整验证脚本见 test_fk.py。
"""
import numpy as np


def calc_T_n_to_last(q_n, d_n, a_n, alpha_n):
    s_q, c_q = np.sin(q_n), np.cos(q_n)
    s_a, c_a = np.sin(alpha_n), np.cos(alpha_n)
    T_n_to_last = np.array([
        [c_q, -s_q * c_a, s_q * s_a, a_n * c_q],
        [s_q, c_q * c_a, -c_q * s_a, a_n * s_q],
        [0, s_a, c_a, d_n],
        [0, 0, 0, 1],
    ])
    return T_n_to_last


def rot_to_eular(R):
    # ZXY: R = R_y(ϑy) R_x(ϑx) R_z(ϑz)，返回 [ϑx, ϑy, ϑz]
    # 由 R[1,2]=-sϑx, R[1,0]=cϑx sϑz, R[1,1]=cϑx cϑz, R[0,2]=sϑy cϑx, R[2,2]=cϑy cϑx
    c_ϑx = np.sqrt(R[1, 0] ** 2 + R[1, 1] ** 2)
    ϑx = np.arctan2(-R[1, 2], c_ϑx)
    ϑy = np.arctan2(R[0, 2], R[2, 2])
    ϑz = np.arctan2(R[1, 0], R[1, 1])
    return np.array([ϑx, ϑy, ϑz])


def robot_fk(q, d, a, alpha, N=6):
    assert q.shape == (N + 1,) and d.shape == (N + 1,)
    assert a.shape == (N + 1,) and alpha.shape == (N + 1,)
    T_n_to_base = np.eye(4)
    for n in range(1, N + 1):
        T_n_to_base = T_n_to_base @ calc_T_n_to_last(
            q[n], d[n], a[n], alpha[n]
        )
    R_N_to_0, t_0N = T_n_to_base[:3, :3], T_n_to_base[:3, 3]
    eular_e = rot_to_eular(R_N_to_0)
    return np.concatenate([t_0N, eular_e])
