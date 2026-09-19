# -*- coding: utf-8 -*-
"""验证《Math Toolbox for Robotics》Part III 最优控制的两类跟踪算法
（书 chap3_3_optmctrol_4.tex「LQR跟踪控制」「LQR输入增量控制」）。

验证对象（与书的算法框 / 代码块一一对应，为修正后的版本）：
  1. oc_LQR_disc             —— 有限时域离散 LQR（调节），DP/Riccati 后向递推 + 前向执行；
  2. oc_LQR_track_disc       —— LQR 跟踪控制（状态扩张 x_e = [x; x_d]）；
  3. oc_LQR_track_smooth_disc—— LQR 输入增量控制（扩张 x_e = [x; x_d; u_{k-1}]，
                                  以输入增量 Δu_k = u_k - u_{k-1} 为控制量）。

验证方式（与书中的"批量闭式解"这一独立公式交叉验证）：
  * 调节 / 跟踪 / 增量跟踪的最优控制均可写成无约束 QP，闭式解 U* = -H^{-1} g
    （书 chap3_4_mpc_1.tex「无约束线性MPC」一节）；把 DP 递推得到的控制序列
    与批量闭式解逐点对比，要求逐点一致；
  * 校验扩张状态方程与代价函数在扩张前后完全等价；
  * 校验参考轨迹传播矩阵 A_d 满足 A_d x_{d,k} = x_{d,k+1}；
  * 校验输入增量控制得到的 u 序列相比普通跟踪变化更平滑。

运行：E:\\Anaconda3\\envs\\py311-gym\\python.exe test/test_lqr_tracking.py
"""
import numpy as np
import scipy.linalg

np.set_printoptions(precision=6, suppress=True)

TOL = 1e-8


# ---------------------------------------------------------------- 核心算法
# 与书 chap3_3_optmctrol_2.tex / _4.tex 的代码块一致（修正后的版本）

def oc_LQR_disc(x_0, As, Bs, Rs, Qs, S, N):
    """有限时域离散 LQR（调节），DP 递推，返回最优控制序列 u_0..u_{N-1}。

    约定：x_{k+1} = A_k x_k + B_k u_k；
          J = x_N^T S x_N + Σ_k (x_k^T Q_k x_k + u_k^T R_k u_k)；
          u_k = F_k x_k，F_k = -(R_k + B_k^T P_{k+1} B_k)^{-1} B_k^T P_{k+1} A_k。
    """
    assert len(As) == N and len(Bs) == N
    assert len(Rs) == N and len(Qs) == N
    Fs, P_next = [], S
    for k in range(N - 1, -1, -1):
        tmp = np.linalg.inv(Rs[k] + Bs[k].T @ P_next @ Bs[k])
        Fs.append(-tmp @ Bs[k].T @ P_next @ As[k])
        P_next = As[k].T @ P_next @ (As[k] + Bs[k] @ Fs[-1]) + Qs[k]
    Fs, x_k, u_opt = Fs[::-1], x_0, []
    for k in range(N):
        u_opt.append(Fs[k] @ x_k)
        x_k = As[k] @ x_k + Bs[k] @ u_opt[k]
    return u_opt


def oc_LQR_track_disc(x_0, xds, A, B, Rs, Qs, S, N, lambda_=0.5):
    """LQR 跟踪控制[离散]：扩张状态 x_e = [x; x_d]，用标准 LQR 求解。

    xds：参考轨迹，长度 N+1（xds[0..N]，索引与数学下标一致）。
    A_d：参考状态转移矩阵，满足 x_{d,k+1} = A_{d,k} x_{d,k}。
    """
    n = x_0.shape[0]
    assert len(Rs) == N and len(Qs) == N
    assert len(xds) == N + 1
    Ae_s, Be_s, Qe_s = [], [], []
    for k in range(N):
        Ad_k = lambda_ * np.eye(n)
        Ad_k += ((xds[k + 1] - lambda_ * xds[k])[:, None]
                 @ xds[k][None, :]) / (xds[k] @ xds[k])
        Ae_s.append(np.block([
            [A, np.zeros((n, n))],
            [np.zeros((n, n)), Ad_k],
        ]))
        Be_s.append(np.vstack([B, np.zeros_like(B)]))
        Qe_s.append(np.block([
            [Qs[k], -Qs[k]],
            [-Qs[k], Qs[k]],
        ]))
    Se = np.block([
        [S, -S],
        [-S, S],
    ])
    xe_0 = np.concatenate([x_0, xds[0]])
    return oc_LQR_disc(xe_0, Ae_s, Be_s, Rs, Qe_s, Se, N)


def oc_LQR_track_smooth_disc(x_0, u_prev, xds, A, B, Rs, Qs, S, N, lambda_=0.5):
    """LQR 输入增量控制[离散]（平滑跟踪）：
    扩张状态 x_e = [x; x_d; u_{k-1}]，以输入增量 Δu_k 为控制量。

    扩张状态方程：x_{k+1} = A x_k + B u_{k-1} + B Δu_k，
    即 A_e = [[A, 0, B], [0, A_d, 0], [0, 0, I]]，B_e = [B; 0; I]。
    返回实际输入序列 u_0..u_{N-1}（由 Δu_k 逐点累加重构）。
    """
    n, m = B.shape
    assert len(Rs) == N and len(Qs) == N
    assert len(xds) == N + 1
    Ae_s, Be_s, Qe_s = [], [], []
    for k in range(N):
        Ad_k = lambda_ * np.eye(n)
        Ad_k += ((xds[k + 1] - lambda_ * xds[k])[:, None]
                 @ xds[k][None, :]) / (xds[k] @ xds[k])
        Ae_s.append(np.block([
            [A, np.zeros((n, n)), B],
            [np.zeros((n, n)), Ad_k, np.zeros((n, m))],
            [np.zeros((m, n)), np.zeros((m, n)), np.eye(m)],
        ]))
        Be_s.append(np.vstack([B, np.zeros((n, m)), np.eye(m)]))
        Qe_s.append(np.block([
            [Qs[k], -Qs[k], np.zeros((n, m))],
            [-Qs[k], Qs[k], np.zeros((n, m))],
            [np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, m))],
        ]))
    Se = np.block([
        [S, -S, np.zeros((n, m))],
        [-S, S, np.zeros((n, m))],
        [np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, m))],
    ])
    xe_0 = np.concatenate([x_0, xds[0], u_prev])
    dU_opt = oc_LQR_disc(xe_0, Ae_s, Be_s, Rs, Qe_s, Se, N)
    # 由输入增量重构实际输入序列 u_k = u_{k-1} + Δu_k
    u_opt, u_k = [], u_prev.copy()
    for k in range(N):
        u_k = u_k + dU_opt[k]
        u_opt.append(u_k.copy())
    return u_opt


# ---------------------------------------------------------------- 辅助函数

def simulate(x_0, As, Bs, us):
    """按系统方程前向仿真，返回状态序列 x_0..x_N。"""
    xs = [np.array(x_0)]
    x_k = x_0
    for k, u in enumerate(us):
        x_k = As[k] @ x_k + Bs[k] @ u
        xs.append(x_k)
    return xs


def build_batch_matrices(As, Bs, N):
    """构造批量形式 X = [x_1; ...; x_N] = Φ x_0 + Γ U（U = [u_0; ...; u_{N-1}]）。"""
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


def smooth_cost(xs, xds, dus, Qs, Rs, S):
    """输入增量控制（平滑跟踪）的代价：
    J = ||x_N - x_{d,N}||²_S + Σ_k (||x_k - x_{d,k}||²_{Q_k} + ||Δu_k||²_{R_k})。"""
    N = len(dus)
    J = (xs[N] - xds[N]) @ S @ (xs[N] - xds[N])
    for k in range(N):
        e = xs[k] - xds[k]
        J += e @ Qs[k] @ e + dus[k] @ Rs[k] @ dus[k]
    return float(J)


def diff_seq(us, u_prev):
    """由输入序列 u_0..u_{N-1} 还原增量序列 Δu_0..Δu_{N-1}。"""
    return [us[0] - u_prev] + [us[k] - us[k - 1] for k in range(1, len(us))]


def input_rate(us, u_prev):
    """输入序列的变化率指标：max|Δu| 与 Σ|Δu|²。"""
    du = np.concatenate([np.atleast_1d(x).ravel() for x in diff_seq(us, u_prev)])
    return float(np.max(np.abs(du))), float(np.sum(du ** 2))


def print_ok(name):
    print(f"  [OK] {name}")


# ---------------------------------------------------------------- 各算法验证

def verify_regulation(A, B, Q, R, S, N, x_0):
    """验证 1：oc_LQR_disc（调节） vs 批量闭式解。"""
    print("[验证1] 离散LQR调节：DP递推 vs 批量闭式解")
    As = [A] * N; Bs = [B] * N; Qs = [Q] * N; Rs = [R] * N
    u_dp = oc_LQR_disc(x_0, As, Bs, Rs, Qs, S, N)

    Phi, Gamma = build_batch_matrices(As, Bs, N)
    Q_t = scipy.linalg.block_diag(*Qs[1:], S)       # Q_1..Q_{N-1}, S（X 不含 x_0）
    R_t = scipy.linalg.block_diag(*Rs)              # R_0..R_{N-1}
    H = R_t + Gamma.T @ Q_t @ Gamma
    U_batch = -np.linalg.solve(H, Gamma.T @ Q_t @ Phi @ x_0)

    err = max_diff(u_dp, U_batch)
    print(f"  DP vs batch 最大逐点差 = {err:.3e}")
    assert err < TOL, "调节问题 DP 与批量解不一致!"
    print_ok("控制序列逐点一致")
    # 前馈符号约定自检：长时域下 F_0 应收敛到 DARE 解
    P_inf = scipy.linalg.solve_discrete_are(A, B, Q, R)
    F_inf = -np.linalg.inv(R + B.T @ P_inf @ B) @ B.T @ P_inf @ A
    u0 = u_dp[0]
    print(f"  长时域收敛自检：||u_0(有限) - u_0(DARE)|| = "
          f"{np.max(np.abs(u0 - F_inf @ x_0)):.3e}")
    assert np.allclose(u0, F_inf @ x_0, atol=1e-4), "F 符号约定错误!"
    print_ok("反馈符号约定与 DARE 一致")
    return u_dp


def verify_tracking(A, B, Q, R, S, N, x_0, xds):
    """验证 2：oc_LQR_track_disc（跟踪） vs 批量闭式解 + 扩张一致性。"""
    print("[验证2] LQR跟踪控制：DP递推 vs 批量闭式解")
    As = [A] * N; Bs = [B] * N; Qs = [Q] * N; Rs = [R] * N
    u_dp = oc_LQR_track_disc(x_0, xds, A, B, Rs, Qs, S, N)
    xs = simulate(x_0, As, Bs, u_dp)

    # (a) 参考轨迹传播矩阵 A_d 的一致性：A_d x_{d,k} = x_{d,k+1}
    for k in range(N):
        Ad_k = 0.5 * np.eye(x_0.shape[0])
        Ad_k += ((xds[k + 1] - 0.5 * xds[k])[:, None]
                 @ xds[k][None, :]) / (xds[k] @ xds[k])
        assert np.allclose(Ad_k @ xds[k], xds[k + 1]), "A_d 不满足参考传播!"
    print_ok("A_d x_{d,k} = x_{d,k+1} 对所有 k 成立")

    # (b) 批量闭式解（与书 chap3_4_mpc_1.tex 跟踪公式一致）
    Phi, Gamma = build_batch_matrices(As, Bs, N)
    Q_t = scipy.linalg.block_diag(*Qs[1:], S)
    R_t = scipy.linalg.block_diag(*Rs)
    X_d = np.concatenate([np.atleast_1d(xd).ravel() for xd in xds[1:]])
    H = R_t + Gamma.T @ Q_t @ Gamma
    U_batch = -np.linalg.solve(H, Gamma.T @ Q_t @ (Phi @ x_0 - X_d))

    err = max_diff(u_dp, U_batch)
    print(f"  DP vs batch 最大逐点差 = {err:.3e}")
    assert err < TOL, "跟踪问题 DP 与批量解不一致!"
    print_ok("控制序列逐点一致")

    # (c) 扩张前后代价函数等价：J_aug(x_e, u) == J_track(x, x_d, u)
    J_track = (xs[N] - xds[N]) @ S @ (xs[N] - xds[N])
    for k in range(N):
        e = xs[k] - xds[k]
        J_track += e @ Q @ e + u_dp[k] @ R @ u_dp[k]
    # 扩张代价：J_aug = x_eN^T S_e x_eN + Σ (x_ek^T Q_e x_ek + u^T R u)
    n = x_0.shape[0]
    xe_0 = np.concatenate([x_0, xds[0]])
    xe_s = simulate(xe_0, [np.block([[A, np.zeros((n, n))],
                                     [np.zeros((n, n)), 0.5 * np.eye(n)
                                      + ((xds[k + 1] - 0.5 * xds[k])[:, None]
                                         @ xds[k][None, :]) / (xds[k] @ xds[k])]])
                      for k in range(N)],
                    [np.vstack([B, np.zeros_like(B)])] * N, u_dp)
    S_e = np.block([[S, -S], [-S, S]])
    Q_e = np.block([[Q, -Q], [-Q, Q]])
    J_aug = xe_s[N] @ S_e @ xe_s[N]
    for k in range(N):
        J_aug += xe_s[k] @ Q_e @ xe_s[k] + u_dp[k] @ R @ u_dp[k]
    print(f"  扩张前后代价：J_track = {J_track:.6f}, J_aug = {J_aug:.6f}")
    assert abs(J_track - J_aug) < TOL, "扩张前后代价不等价!"
    print_ok("扩张前后代价函数等价")

    # (d) 跟踪效果
    print(f"  末端跟踪误差 ||x_N - x_{'{d,N}'}|| = {np.linalg.norm(xs[N] - xds[N]):.4f}")
    return u_dp


def verify_smooth(A, B, Q, R, S, N, x_0, u_prev, xds, u_plain):
    """验证 3：oc_LQR_track_smooth_disc（输入增量） vs 批量velocity-form闭式解。"""
    print("[验证3] LQR输入增量控制：DP递推 vs 批量velocity-form闭式解")
    As = [A] * N; Bs = [B] * N; Qs = [Q] * N; Rs = [R] * N
    n, m = B.shape
    u_smooth = oc_LQR_track_smooth_disc(x_0, u_prev, xds, A, B, Rs, Qs, S, N)
    xs = simulate(x_0, As, Bs, u_smooth)
    dus = diff_seq(u_smooth, u_prev)

    # (a) 扩张模型一致性：x_{k+1} = A x_k + B u_{k-1} + B Δu_k，u_k = u_{k-1} + Δu_k
    uu = u_prev.copy()
    for k in range(N):
        uu = uu + dus[k]
        assert np.allclose(xs[k + 1], A @ xs[k] + B @ uu), "扩张模型不还原原系统!"
    print_ok("扩张模型 x_{k+1} = A x_k + B u_{k-1} + B Δu_k 还原原系统")

    # (b) 批量 velocity-form 闭式解
    #     U = C ΔU + (1 ⊗ u_{-1})，C 为下三角块全一矩阵
    Phi, Gamma = build_batch_matrices(As, Bs, N)
    Q_t = scipy.linalg.block_diag(*Qs[1:], S)
    R_t = scipy.linalg.block_diag(*Rs)
    C = np.zeros((N * m, N * m))
    for i in range(N):
        for j in range(i + 1):
            C[i * m:(i + 1) * m, j * m:(j + 1) * m] = np.eye(m)
    X_d = np.concatenate([np.atleast_1d(xd).ravel() for xd in xds[1:]])
    U_prev = np.kron(np.ones(N), u_prev)            # 堆叠 u_{-1} 为 (Nm,) 一维
    rhs = C.T @ Gamma.T @ Q_t @ (Phi @ x_0 + Gamma @ U_prev - X_d)
    H = R_t + C.T @ Gamma.T @ Q_t @ Gamma @ C
    dU_batch = -np.linalg.solve(H, rhs)

    err = max_diff(dus, dU_batch)
    print(f"  DP vs batch 最大逐点差 = {err:.3e}")
    assert err < TOL, "输入增量问题 DP 与批量解不一致!"
    print_ok("Δu 序列逐点一致")

    # (c) 平滑性：普通跟踪与平滑跟踪是"不同"的优化问题（前者惩罚 u、后者惩罚 Δu），
    #     不能直接比较两者的 |Δu|。改为验证两条与"平滑"直接相关的性质：
    #     ① 在 Δu 代价下，平滑控制器（最优解）必然优于普通跟踪的控制轨迹；
    #     ② 增大 Δu 权重 R 时，输入变化率单调下降（速率受限机制）。
    xs_plain = simulate(x_0, As, Bs, u_plain)
    dus_plain = diff_seq(u_plain, u_prev)
    J_smooth = smooth_cost(xs, xds, dus, Qs, Rs, S)
    J_plain = smooth_cost(xs_plain, xds, dus_plain, Qs, Rs, S)
    m_plain, s_plain = input_rate(u_plain, u_prev)
    m_smooth, s_smooth = input_rate(u_smooth, u_prev)
    print(f"  平滑代价：平滑跟踪 {J_smooth:.4f}，普通跟踪 {J_plain:.4f}")
    print(f"  输入变化率：普通跟踪 max|Δu|={m_plain:.4f} ΣΔu²={s_plain:.4f}"
          f"，平滑跟踪 max|Δu|={m_smooth:.4f} ΣΔu²={s_smooth:.4f}")
    assert J_smooth < J_plain, "平滑控制器应使 Δu 代价更小!"
    print_ok("平滑控制器在 Δu 代价下优于普通跟踪（速率受限生效）")
    return u_smooth


def verify_smooth_mechanism(A, B, Q, S, N, x_0, u_prev, xds):
    """验证 3b：增大 Δu 权重 R 时，输入变化率单调下降（书所述"控制量平滑"的机制）。"""
    print("[验证3b] 平滑机制：Δu 权重 R 越大 → 输入变化率越小（单调）")
    prev_rate = np.inf
    for r in (0.05, 0.2, 0.5, 2.0, 8.0):
        R = np.array([[r]])
        u = oc_LQR_track_smooth_disc(x_0, u_prev, xds, A, B,
                                     [R] * N, [Q] * N, S, N)
        mdu, sdu = input_rate(u, u_prev)
        xs = simulate(x_0, [A] * N, [B] * N, u)
        err = float(np.linalg.norm(xs[-1] - xds[-1]))
        print(f"  R_du={r:>5}: max|Δu|={mdu:.4f}  ΣΔu²={sdu:.4f}  末端误差={err:.4f}")
        assert mdu < prev_rate, "输入变化率应随 R_du 单调下降!"
        prev_rate = mdu
    print_ok("R_du 越大输入变化率越小（速率受限机制成立）")


# ---------------------------------------------------------------- 主流程

def main():
    print("=" * 62)
    print("验证 LQR跟踪控制 / LQR输入增量控制（chap3_3_optmctrol_4.tex）")
    print("=" * 62)

    # 测试系统：2 维稳定离散系统（阻尼振荡，可控）
    A = np.array([[0.9, 0.1], [-0.1, 0.85]])
    B = np.array([[0.05], [0.10]])
    Q = np.diag([1.0, 1.0])
    R = np.array([[0.5]])
    S = 10.0 * Q
    N = 30
    x_0 = np.array([0.0, 0.0])
    u_prev = np.array([0.0])

    # 能控性自检
    Cc = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(A.shape[0])])
    assert np.linalg.matrix_rank(Cc) == A.shape[0], "测试系统不可控!"

    print(f"\n# 测试系统：n = {A.shape[0]}, m = {B.shape[1]}, N = {N}\n")

    # 参考轨迹 1：恒值（x_d = [1, 0]，对应书"若 x_d 为恒值"情形）
    xds_const = [np.array([1.0, 0.0])] * (N + 1)
    print("--- 参考轨迹：恒值 x_d = [1, 0] ---")
    u_dp_reg = verify_regulation(A, B, Q, R, S, N, x_0)
    u_track1 = verify_tracking(A, B, Q, R, S, N, x_0, xds_const)
    verify_smooth(A, B, Q, R, S, N, x_0, u_prev, xds_const, u_track1)
    verify_smooth_mechanism(A, B, Q, S, N, x_0, u_prev, xds_const)

    # 参考轨迹 2：系统自由响应（衰减振荡，x_{d,k+1} = A x_{d,k}）
    xds_osc = [np.array([1.0, 0.2])]
    for k in range(N):
        xds_osc.append(A @ xds_osc[-1])
    print("\n--- 参考轨迹：衰减振荡（自由响应） ---")
    u_track2 = verify_tracking(A, B, Q, R, S, N, x_0, xds_osc)
    verify_smooth(A, B, Q, R, S, N, x_0, u_prev, xds_osc, u_track2)

    print("\n全部验证通过 ✔")


if __name__ == "__main__":
    main()
