# -*- coding: utf-8 -*-
"""验证《Math Toolbox for Robotics》Part III 最优控制的 LQR 跟踪两类算法
（书 chap3_3_optmctrol_2.tex / chap3_3_optmctrol_4.tex，对应 code_lqr.py）。

验证方式：三类最优控制问题均可写成无约束 QP，批量闭式解 U* = -H^{-1} g
（对应书 chap3_4_mpc_1.tex 的无约束线性 MPC 公式），
与 code_lqr.py 的 DP/Riccati 递推逐点对比，要求逐点一致。

运行：E:\\Anaconda3\\envs\\py311-gym\\python.exe test_lqr.py
"""
import numpy as np
import scipy.linalg

from code_lqr import oc_LQR_disc, oc_LQR_track_disc, oc_LQR_track_smooth_disc

np.set_printoptions(precision=6, suppress=True)
TOL = 1e-8


def build_batch_matrices(As, Bs, N):
    """批量形式 X = [x_1; ...; x_N] = Phi x_0 + Gamma U（U = [u_0; ...; u_{N-1}]）。"""
    n = As[0].shape[0]
    m = Bs[0].shape[1]
    Phi_rows = []
    for k in range(1, N + 1):                       # x_k = A_{k-1}...A_0 x_0
        P = np.eye(n)
        for i in range(k):
            P = As[i] @ P
        Phi_rows.append(P)
    Phi = np.vstack(Phi_rows)                       # (Nn, n)
    Gamma = np.zeros((N * n, N * m))
    for k in range(1, N + 1):                       # x_k 中 u_j (j<k) 的系数
        for j in range(k):
            block = np.eye(n)
            for i in range(k - 1, j, -1):           # A_{k-1} ... A_{j+1}
                block = As[i] @ block
            block = block @ Bs[j]
            Gamma[(k - 1) * n:k * n, j * m:(j + 1) * m] = block
    return Phi, Gamma


def max_diff(list_of_arrays, batched):
    """DP 返回的逐点控制序列 vs 批量闭式解拼接向量，比较逐点最大差。"""
    us = np.concatenate([np.atleast_1d(u).ravel() for u in list_of_arrays])
    ub = np.atleast_1d(batched).ravel()
    return float(np.max(np.abs(us - ub)))


def simulate(x_0, A, B, us):
    xs = [np.array(x_0)]
    x_k = x_0
    for u in us:
        x_k = A @ x_k + B @ u
        xs.append(x_k)
    return xs


def diff_seq(us, u_prev):
    return [us[0] - u_prev] + [us[k] - us[k - 1] for k in range(1, len(us))]


def smooth_cost(xs, xds, dus, Q, R, S):
    """输入增量控制的代价：J = ||x_N-x_{d,N}||²_S + Σ(||x_k-x_{d,k}||²_Q + ||Δu_k||²_R)。"""
    J = (xs[-1] - xds[-1]) @ S @ (xs[-1] - xds[-1])
    for k, du in enumerate(dus):
        e = xs[k] - xds[k]
        J += e @ Q @ e + du @ R @ du
    return float(J)


def input_rate(us, u_prev):
    du = np.concatenate([np.atleast_1d(x).ravel() for x in diff_seq(us, u_prev)])
    return float(np.max(np.abs(du))), float(np.sum(du ** 2))


def main():
    print("=" * 60)
    print("验证 LQR跟踪控制 / LQR输入增量控制（code_lqr.py）")
    print("=" * 60)

    # 测试系统：2 维稳定离散系统（可控）
    A = np.array([[0.9, 0.1], [-0.1, 0.85]])
    B = np.array([[0.05], [0.10]])
    Q = np.diag([1.0, 1.0])
    R = np.array([[0.5]])
    S = 10.0 * Q
    N = 30
    x_0 = np.array([0.0, 0.0])
    u_prev = np.array([0.0])

    Cc = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(A.shape[0])])
    assert np.linalg.matrix_rank(Cc) == A.shape[0], "测试系统不可控!"

    As = [A] * N; Bs = [B] * N; Qs = [Q] * N; Rs = [R] * N
    Q_t = scipy.linalg.block_diag(*Qs[1:], S)       # Q_1..Q_{N-1}, S（X 不含 x_0）
    R_t = scipy.linalg.block_diag(*Rs)              # R_0..R_{N-1}
    Phi, Gamma = build_batch_matrices(As, Bs, N)

    # ---- 1) 离散 LQR 调节 ----
    print("\n[验证1] 离散LQR调节：DP递推 vs 批量闭式解")
    u_dp = oc_LQR_disc(x_0, As, Bs, Rs, Qs, S, N)
    H = R_t + Gamma.T @ Q_t @ Gamma
    U_batch = -np.linalg.solve(H, Gamma.T @ Q_t @ Phi @ x_0)
    err = max_diff(u_dp, U_batch)
    print(f"  DP vs batch 最大逐点差 = {err:.3e}")
    assert err < TOL, "调节问题 DP 与批量解不一致!"

    # ---- 2) LQR 跟踪控制 ----
    xds = [np.array([1.0, 0.0])] * (N + 1)          # 恒值参考
    print("\n[验证2] LQR跟踪控制：DP递推 vs 批量闭式解")
    u_dp = oc_LQR_track_disc(x_0, xds, A, B, Rs, Qs, S, N)
    X_d = np.concatenate([np.atleast_1d(xd).ravel() for xd in xds[1:]])
    H = R_t + Gamma.T @ Q_t @ Gamma
    U_batch = -np.linalg.solve(H, Gamma.T @ Q_t @ (Phi @ x_0 - X_d))
    err = max_diff(u_dp, U_batch)
    print(f"  DP vs batch 最大逐点差 = {err:.3e}")
    assert err < TOL, "跟踪问题 DP 与批量解不一致!"
    # A_d 一致性：A_d x_{d,k} = x_{d,k+1}
    n = x_0.shape[0]
    for k in range(N):
        Ad_k = 0.5 * np.eye(n)
        Ad_k += ((xds[k + 1] - 0.5 * xds[k])[:, None]
                 @ xds[k][None, :]) / (xds[k] @ xds[k])
        assert np.allclose(Ad_k @ xds[k], xds[k + 1]), "A_d 不满足参考传播!"
    print("  A_d x_{d,k} = x_{d,k+1} 对所有 k 成立")
    u_plain = u_dp

    # ---- 3) LQR 输入增量控制 ----
    print("\n[验证3] LQR输入增量控制：DP递推 vs 批量velocity-form闭式解")
    u_smooth = oc_LQR_track_smooth_disc(x_0, u_prev, xds, A, B, Rs, Qs, S, N)
    dus = diff_seq(u_smooth, u_prev)
    n, m = B.shape
    # U = C ΔU + (1 ⊗ u_{-1})，C 为下三角块全一矩阵
    C = np.zeros((N * m, N * m))
    for i in range(N):
        for j in range(i + 1):
            C[i * m:(i + 1) * m, j * m:(j + 1) * m] = np.eye(m)
    U_prev = np.kron(np.ones(N), u_prev)
    rhs = C.T @ Gamma.T @ Q_t @ (Phi @ x_0 + Gamma @ U_prev - X_d)
    H = R_t + C.T @ Gamma.T @ Q_t @ Gamma @ C
    dU_batch = -np.linalg.solve(H, rhs)
    err = max_diff(dus, dU_batch)
    print(f"  DP vs batch 最大逐点差 = {err:.3e}")
    assert err < TOL, "输入增量问题 DP 与批量解不一致!"
    # 扩张模型还原：x_{k+1} = A x_k + B u_{k-1} + B Δu_k
    xs = simulate(x_0, A, B, u_smooth)
    uu = u_prev.copy()
    for k in range(N):
        uu = uu + dus[k]
        assert np.allclose(xs[k + 1], A @ xs[k] + B @ uu), "扩张模型不还原原系统!"
    print("  扩张模型 x_{k+1} = A x_k + B u_{k-1} + B Δu_k 还原原系统")
    # 平滑性：在 Δu 代价下，平滑控制器（最优）必然优于普通跟踪
    J_smooth = smooth_cost(xs, xds, dus, Q, R, S)
    J_plain = smooth_cost(simulate(x_0, A, B, u_plain),
                          xds, diff_seq(u_plain, u_prev), Q, R, S)
    m_s, s_s = input_rate(u_smooth, u_prev)
    m_p, s_p = input_rate(u_plain, u_prev)
    print(f"  Δu 代价：平滑跟踪 {J_smooth:.4f} < 普通跟踪 {J_plain:.4f}")
    print(f"  输入变化率：普通 max|Δu|={m_p:.4f} ΣΔu²={s_p:.4f}，"
          f"平滑 max|Δu|={m_s:.4f} ΣΔu²={s_s:.4f}")
    assert J_smooth < J_plain, "平滑控制器应使 Δu 代价更小!"

    # ---- 3b) 平滑机制：R_du 越大 → 输入变化率越小（单调） ----
    print("\n[验证3b] 平滑机制：Δu 权重 R 越大 → 输入变化率越小")
    prev_rate = np.inf
    for r in (0.05, 0.2, 0.5, 2.0, 8.0):
        Rr = np.array([[r]])
        u = oc_LQR_track_smooth_disc(x_0, u_prev, xds, A, B,
                                     [Rr] * N, [Q] * N, S, N)
        mdu, _ = input_rate(u, u_prev)
        print(f"  R_du={r:>5}: max|Δu|={mdu:.4f}")
        assert mdu < prev_rate, "输入变化率应随 R_du 单调下降!"
        prev_rate = mdu

    print("\n全部验证通过 ✔")


if __name__ == "__main__":
    main()
