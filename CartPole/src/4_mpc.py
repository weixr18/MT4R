# -*- coding: utf-8 -*-
"""模型预测控制（MPC）—— 倒立摆验证（书 chap3_4_mpc_1.tex / chap3_4_mpc_2.tex）

三个算法（对应书算法框，算法代码 / 验证脚本一体）：
  算法1 无约束线性 MPC（调节）      algo:MPC_LQR_nocons          —— U* = F x_k
  算法2 无约束线性 MPC 跟踪控制      algo:MPC_LQR_track_nocons    —— U* = F x_k + K X_d,k
  算法3 不等式约束线性 MPC（调节）   algo:MPC_LQR_necons          —— 在线 QP（复用 algo:opt_qp_barrier）

演示对象为 Part VIII 倒立摆平衡点线性化离散模型（与 src/2_lqr.py、3_lqr_discrete.py 同源）。

运行（cwd 任意，推荐仓库根）：
  E:\\Anaconda3\\envs\\py311-gym\\python.exe src/4_mpc.py
出图默认写入书仓库 1-MN4R/imgs/mpc/，可用环境变量 FIG_DIR 覆盖。
"""
import os
import sys

import numpy as np
import gym
from scipy.linalg import expm

from env_cartpole import CONFIG, register_custom_cartpole

# 复用 Part I 的 QP 障碍函数法求解器（书 chap1_3_optimz_2.tex 的 algo:opt_qp_barrier，
# 位于本仓库 Codes/chap1_3_optmz/code_4_qp.py，保持单一来源）
QP_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "Codes", "chap1_3_optmz"))
if QP_DIR not in sys.path:
    sys.path.append(QP_DIR)
from code_4_qp import opt_qp_barrier

# 出图目录：书仓库 imgs/mpc/（脚本位于 4-MT4R-github/CartPole/src/）
FIG_DIR = os.environ.get("FIG_DIR", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "1-MN4R", "imgs", "mpc")))


# ======================================================================
# 演示对象：倒立摆平衡点线性化 + ZOH 精确离散化
# ======================================================================
def cartpole_linear(dt=None):
    """返回连续与 ZOH 离散的 (A, B)。状态 x = [x, x_dot, theta, theta_dot]^T，输入 u = F。

    系数 k1..k6 见书 chap8_1_1storder_1.tex，与 src/2_lqr.py 的 calc_sys_mat() 一致。
    离散化采用分块矩阵指数（连续 A 奇异，不能直接 A^{-1}(e^{AT}-I)B）：
        expm([[A, B], [0, 0]] * dt) = [[A_d, B_d], [0, I]]
    """
    if dt is None:
        dt = CONFIG['dt']
    M, m1, l1, mu, g = CONFIG['M'], CONFIG['m1'], CONFIG['l1'], CONFIG['mu'], CONFIG['g']
    D = 4 * M + m1
    k1 = 4.0 / D
    k2 = -3.0 * m1 * g / D
    k3 = 3.0 * (M + m1) * g / (l1 * D)
    k4 = -3.0 / (l1 * D)
    k6 = 3.0 * mu / (l1 * D)

    Ac = np.array([[0.0, 1.0, 0.0, 0.0],
                   [0.0, -mu * k1, k2, 0.0],
                   [0.0, 0.0, 0.0, 1.0],
                   [0.0, k6, k3, 0.0]])
    Bc = np.array([[0.0], [k1], [0.0], [k4]])

    n = Ac.shape[0]
    M_exp = expm(np.block([[Ac * dt, Bc * dt],
                           [np.zeros((1, n)), np.zeros((1, 1))]]))
    Ad = M_exp[:n, :n]
    Bd = M_exp[:n, n:]
    return (Ac, Bc), (Ad, Bd)


# ======================================================================
# 算法1 / 算法2 公共离线部分：预测区间矩阵构建（书式 Φ, Γ, Q̄, R̄）
# ======================================================================
def mpc_build(A, B, Q, R, S, Np):
    """离线构建并返回 (F, K, G, H, Phi, Gamma, Qbar, Rbar)。

    预测窗口含 Np 个状态槽位（x_{k|k+1}..x_{k|k+Np}）与 Np 个控制槽位（u_{k|k}..u_{k|k+Np-1}）：
      Phi   = [A; ...; A^{Np}]（Np 块，不含 I）
      Gamma = 下三角块 Toeplitz：Gamma[i,j] = A^{i-j} B（i>=j），无零行/零列
      Qbar  = diag(Q, ..., Q, S)（末块 S 为终端权重）；Rbar = diag(R, ..., R)（全正定）
      G = Phi^T Qbar Gamma；H = Rbar + Gamma^T Qbar Gamma
      F = -H^{-1} G^T（状态反馈）；K = H^{-1} Gamma^T Qbar（参考前馈）

    Rbar 正定且 Gamma 满列秩 ⇒ H 正定可逆，直接求逆即可（无需伪逆）。
    """
    n, m = B.shape
    slots = Np
    Phi = np.zeros((slots * n, n))
    for i in range(slots):
        Phi[i * n:(i + 1) * n, :] = np.linalg.matrix_power(A, i + 1)   # A^1 .. A^Np
    Gamma = np.zeros((slots * n, slots * m))
    for i in range(slots):                                              # 状态 x_{k|k+i+1}
        for j in range(i + 1):                                          # 控制 u_{k|k+j}
            A_pow_B = np.linalg.matrix_power(A, i - j) @ B
            Gamma[i * n:(i + 1) * n, j * m:(j + 1) * m] = A_pow_B
    Qbar = np.zeros((slots * n, slots * n))
    Rbar = np.zeros((slots * m, slots * m))
    for i in range(slots - 1):                                          # x_{k+1}..x_{k+Np-1}
        Qbar[i * n:(i + 1) * n, i * n:(i + 1) * n] = Q
        Rbar[i * m:(i + 1) * m, i * m:(i + 1) * m] = R
    Rbar[(slots - 1) * m:, (slots - 1) * m:] = R                        # u_{k+Np-1}
    Qbar[(slots - 1) * n:, (slots - 1) * n:] = S                        # 终端 x_{k+Np} 配 S
    G = Phi.T @ Qbar @ Gamma
    H = Rbar + Gamma.T @ Qbar @ Gamma
    F = -np.linalg.inv(H) @ G.T
    K = np.linalg.inv(H) @ (Gamma.T @ Qbar)
    return F, K, G, H, Phi, Gamma, Qbar, Rbar


def mpc_ref_window(xds, k, Np):
    """跟踪参考在预测窗口 (k, k+Np] 的取值 X_d,k（Np 块：x_{d,k+1}..x_{d,k+Np}）。

    超出参考末端时取末值 xds[-1]（保持期望状态不变，保证优化问题良定）。
    """
    blocks = []
    for j in range(k + 1, k + Np + 1):
        if j >= len(xds):
            j = len(xds) - 1
        blocks.append(np.atleast_1d(xds[j]).ravel())
    return np.concatenate(blocks)


# ======================================================================
# 算法1：无约束线性 MPC（调节）在线步
# ======================================================================
def mpc_LQR_nocons(F, x_k, m=1):
    """U* = F x_k，取前 m 个分量作为当前控制 u_k（滚动优化只执行第一项）。"""
    U = F @ x_k
    return U[:m].copy()


# ======================================================================
# 算法2：无约束线性 MPC 跟踪控制在线步
# ======================================================================
def mpc_LQR_track_nocons(F, K, x_k, Xd_k, m=1):
    """U* = F x_k + K X_d,k（状态反馈 + 参考前馈），取前 m 个分量。"""
    U = F @ x_k + K @ Xd_k
    return U[:m].copy()


# ======================================================================
# 算法3：不等式约束线性 MPC（调节）—— 在线 QP 求解
# ======================================================================
def mpc_cons_matrices(Phi, Gamma, Np, Mx, Mu, beta):
    """由约束矩阵列（每个槽位相同的 Mx, Mu, beta）构造预测区间块对角矩阵。

    预测区间约束 M_k U <= b_k，其中 M_k = M_x Γ + M_u，
    b_k = β_k - M_x Φ x_k（x_k 项从 Φ 块中拆出；β_k 为每槽位 beta 平铺 slots 次）。
    返回 (M_k, Mx_, beta_rep)。
    """
    slots = Np
    n, m = Phi.shape[1], Gamma.shape[1] // slots
    rows = Mx.shape[0]
    Mx_ = np.zeros((rows * slots, slots * n))
    Mu_ = np.zeros((rows * slots, slots * m))
    beta_rep = np.zeros(rows * slots)
    for i in range(slots):
        Mx_[i * rows:(i + 1) * rows, i * n:(i + 1) * n] = Mx
        Mu_[i * rows:(i + 1) * rows, i * m:(i + 1) * m] = Mu
        beta_rep[i * rows:(i + 1) * rows] = beta
    M_k = Mx_ @ Gamma + Mu_
    return M_k, Mx_, beta_rep


def mpc_LQR_necons(H, G, Phi, M_k, Mx_, beta_rep, x_k, U0, m=1):
    """构造 b_k = beta_rep - Mx Phi x_k 并求解含不等式约束的 QP，返回当前控制 u_k。

    min_U 1/2 U^T H U + x_k^T G U   s.t.  M_k U <= b_k
    求解器 opt_qp_barrier 见书 chap1_3_optimz_2.tex（QP 障碍函数法，algo:opt_qp_barrier）。
    """
    g = G.T @ x_k
    b_k = beta_rep - Mx_ @ Phi @ x_k
    U = opt_qp_barrier(H, g, np.zeros((0, len(g))), np.zeros(0), M_k, b_k, U0)
    return U[:m].copy()


def mpc_necons_constraints(Np, u_max, theta_max=None, n=4, m=1):
    """构造输入约束 |u|<=u_max 与（可选的）摆角约束 |theta|<=theta_max。

    返回 (Mx, Mu, beta)：预测区间内约束 Mx X + Mu U <= beta 的矩阵列。
    """
    rows = 2 if theta_max is None else 4
    Mx = np.zeros((rows, n))
    Mu = np.zeros((rows, m))
    beta = np.zeros(rows)
    Mx[0] = 0.0; Mu[0] = 1.0; beta[0] = u_max        # u <= u_max
    Mx[1] = 0.0; Mu[1] = -1.0; beta[1] = u_max       # -u <= u_max
    if theta_max is not None:
        Mx[2, 2] = 1.0; beta[2] = theta_max          # theta <= theta_max
        Mx[3, 2] = -1.0; beta[3] = theta_max         # -theta <= theta_max
    return Mx, Mu, beta


# ======================================================================
# 闭环仿真辅助
# ======================================================================
def simulate(x_0, A, B, us):
    """给定控制序列 us，用离散模型 x_{k+1} = A x_k + B u_k 推出状态序列。"""
    xs = [np.array(x_0, dtype=float)]
    x_k = xs[0]
    for u in us:
        x_k = A @ x_k + B @ np.atleast_1d(u)
        xs.append(x_k)
    return np.array(xs)


def mpc_ref_traj(N, dt, x_target=1.0):
    """小车位置参考：正弦加速/减速的平滑轨迹（含速度参考，动力学可行）。

    返回 xds[0..N]，每个元素为 [x_d, x_dot, theta_d, theta_dot] = [.., 0, 0]。
    """
    T = N * dt
    xds = []
    for k in range(N + 1):
        t = k * dt
        if t < T / 2:
            s = 0.5 * (1 - np.cos(np.pi * t / (T / 2)))   # 0 -> 1
            ds = np.pi / T * np.sin(np.pi * t / (T / 2))
        else:
            s, ds = 1.0, 0.0
        xds.append(np.array([x_target * s, x_target * ds, 0.0, 0.0]))
    return xds


# ======================================================================
# 验证（三个算法）+ 出图
# ======================================================================
def verify_mpc():
    """运行三个 MPC 算法的验证：KKT 残差 / 闭环收敛 / 约束满足 / 与 SLSQP 对拍。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    print("=" * 64)
    print("验证 模型预测控制（MPC）三个算法  @ 倒立摆线性化模型")
    print("=" * 64)

    DT = CONFIG['dt']
    Np = 50                       # 预测时域（Np = 50 步，堆叠 50 状态 + 50 控制），区间长 1.0 s
    Q = np.diag([5.0, 0.1, 10.0, 1.0])   # x, x_dot, theta, theta_dot
    R = np.array([[0.01]])
    S = 20.0 * Q
    N1, N3 = 600, 600             # 调节/约束仿真步数（12 s，位置慢模需较长收敛）
    N2 = 400                      # 跟踪仿真步数（8 s）

    (Ac, Bc), (A, B) = cartpole_linear(dt=DT)
    n, m = B.shape
    print(f"离散模型 A,B 形状: {A.shape}, {B.shape}; "
          f"开环特征值: {np.round(np.linalg.eigvals(A), 3)}")

    F, K, G, H, Phi, Gamma, Qbar, Rbar = mpc_build(A, B, Q, R, S, Np)

    # ------------------------------------------------------------------
    # 算法1：无约束线性 MPC（调节）
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("[算法1] 无约束线性 MPC（调节）algo:MPC_LQR_nocons")
    print("=" * 64)
    x0 = np.array([0.15, 0.0, 0.05, 0.0])
    xs, us = [x0.copy()], []
    max_kkt = 0.0
    for k in range(N1):
        x_k = xs[-1]
        u_k = mpc_LQR_nocons(F, x_k, m)
        U = F @ x_k
        max_kkt = max(max_kkt, float(np.linalg.norm(H @ U + G.T @ x_k)))
        us.append(u_k)
        xs.append(A @ x_k + B @ u_k)
    xs = np.array(xs); us = np.array(us)
    t = DT * np.arange(N1 + 1)
    print(f"  逐点 KKT 残差最大值 = {max_kkt:.3e}（应≈0）")
    assert max_kkt < 1e-8, "算法1 不满足 KKT 条件!"
    print(f"  终态 x({N1*DT:.0f}s) = {xs[-1]}，||x||∞ = {np.max(np.abs(xs[-1])):.3e}")
    print(f"  |x| 峰值 {np.max(np.abs(xs[:, 0])):.3f} m，|θ| 峰值 {np.max(np.abs(xs[:, 2])):.4f} rad")
    print(f"  |u| 峰值 {np.max(np.abs(us)):.3f} N，稳态 |u| {float(np.abs(us[-1][0])):.3e} N")
    assert np.max(np.abs(xs[-3:])) < 1e-3, "算法1 未收敛到原点!"
    # 开环对照：验证系统确实不稳定（无控制发散）
    xs_open = simulate(x0, A, B, [np.zeros(m)] * N1)
    assert np.max(np.abs(xs_open)) > 1e3, "开环应发散，用于衬托闭环收敛"
    print(f"  开环对照：无控制 {N1*DT:.0f}s 后 ||x||∞ = {np.max(np.abs(xs_open)):.1f}（发散）")

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.0), sharex=True)
    axes[0].plot(t, xs[:, 0], label="x [m]")
    axes[0].plot(t, xs[:, 2] * 180.0 / np.pi, label=r"$\theta$ [°]")
    axes[0].set_ylabel("states"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(t[:-1], us[:, 0], color="C2", label="u = F [N]")
    axes[1].set_ylabel("input"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(t, xs[:, 1], color="C3", label=r"$\dot x$ [m/s]")
    axes[2].plot(t, xs[:, 3] * 180.0 / np.pi, color="C4", label=r"$\dot\theta$ [°/s]")
    axes[2].set_ylabel("velocities"); axes[2].set_xlabel("t [s]")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.suptitle("MPC-1 unconstrained regulator (cartpole linearized)")
    _save_fig(fig, "mpc_nocons_regulate.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 算法2：无约束线性 MPC 跟踪控制
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("[算法2] 无约束线性 MPC 跟踪控制 algo:MPC_LQR_track_nocons")
    print("=" * 64)
    xds = mpc_ref_traj(N2, DT, x_target=1.0)
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    xs, us = [x0.copy()], []
    max_kkt = 0.0
    for k in range(N2):
        x_k = xs[-1]
        Xd_k = mpc_ref_window(xds, k, Np)
        u_k = mpc_LQR_track_nocons(F, K, x_k, Xd_k, m)
        U = F @ x_k + K @ Xd_k
        max_kkt = max(max_kkt, float(np.linalg.norm(
            H @ U + G.T @ x_k - Gamma.T @ Qbar @ Xd_k)))
        us.append(u_k)
        xs.append(A @ x_k + B @ u_k)
    xs = np.array(xs); us = np.array(us)
    t = DT * np.arange(N2 + 1)
    print(f"  逐点 KKT 残差最大值 = {max_kkt:.3e}（应≈0）")
    assert max_kkt < 1e-8, "算法2 不满足 KKT 条件!"
    xd_arr = np.array([xd[0] for xd in xds])
    track_err = np.max(np.abs(xs[:, 0] - xd_arr[: N2 + 1]))
    print(f"  位置跟踪最大误差 max|x - x_d| = {track_err:.4f} m")
    print(f"  |θ| 峰值 {np.max(np.abs(xs[:, 2])):.4f} rad（摆保持竖直）")
    print(f"  终态 x = {xs[-1]}")
    assert track_err < 0.05, "算法2 跟踪误差过大!"
    assert np.max(np.abs(xs[:, 2])) < 0.1, "跟踪时摆角偏离过大!"

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.0), sharex=True)
    axes[0].plot(t, xd_arr, "k--", lw=1.2, label=r"$x_d$ [m]")
    axes[0].plot(t, xs[:, 0], label="x [m]")
    axes[0].plot(t, xs[:, 2] * 180.0 / np.pi, label=r"$\theta$ [°]")
    axes[0].set_ylabel("states"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(t[:-1], us[:, 0], color="C2", label="u = F [N]")
    axes[1].set_ylabel("input"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(t, xs[:, 1], color="C3", label=r"$\dot x$ [m/s]")
    axes[2].plot(t, xs[:, 3] * 180.0 / np.pi, color="C4", label=r"$\dot\theta$ [°/s]")
    axes[2].set_ylabel("velocities"); axes[2].set_xlabel("t [s]")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.suptitle("MPC-2 unconstrained tracking (cartpole linearized)")
    _save_fig(fig, "mpc_nocons_track.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 算法3：不等式约束线性 MPC（调节）—— 暂缓验证（求解器未调通；见 1-MN4R/docs/todo/cartpole_unfinished.md「遗留事项 A」）
    # 实现与验证脚本保留于 verify_mpc_alg3()，本会话默认跳过；
    # 待求解器大规模收敛性问题调通后再启用（RUN_MPC_ALG3=1 可强制运行）。
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("[算法3] 不等式约束线性 MPC（调节）algo:MPC_LQR_necons —— 本会话暂缓验证")
    print("=" * 64)
    print("  算法实现已就绪（mpc_LQR_necons / mpc_cons_matrices / mpc_necons_constraints），")
    print("  验证脚本保留于 verify_mpc_alg3()；QP 障碍求解器（algo:opt_qp_barrier）在")
    print("  大规模问题上（50 变量 / 100 约束）的收敛性未调通，待下个 session 排查。")
    if os.environ.get("RUN_MPC_ALG3") == "1":
        verify_mpc_alg3(A, B, H, G, Phi, Gamma, Np, N3, m)

    print("\n算法1、算法2 验证通过 ✔（算法3 暂缓）")


def verify_mpc_alg3(A, B, H, G, Phi, Gamma, Np, N3, m):
    """算法3：不等式约束线性 MPC（调节）—— 在线 QP 求解 + 约束/对拍验证。

    当前暂缓：复用 algo:opt_qp_barrier，其大规模收敛性未调通（梯度累加器 bug 已修，
    但 50 变量 / 100 约束问题上仍可能不收敛/卡死）。待排查后启用（RUN_MPC_ALG3=1）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    print("\n" + "=" * 64)
    print("[算法3] 不等式约束线性 MPC（调节）algo:MPC_LQR_necons")
    print("=" * 64)
    DT = CONFIG['dt']
    u_max = 1.0                     # 输入约束 |u| <= u_max（刻意取小，使约束活跃）
    Mx, Mu, beta = mpc_necons_constraints(Np, u_max, theta_max=None)
    M_k, Mx_, beta_rep = mpc_cons_matrices(Phi, Gamma, Np, Mx, Mu, beta)
    x0 = np.array([0.5, 0.0, 0.25, 0.0])   # 大扰动，使最优控制饱和
    xs, us, U0 = [x0.copy()], [], np.zeros(Np * m)
    for k in range(N3):
        x_k = xs[-1]
        u_k = mpc_LQR_necons(H, G, Phi, M_k, Mx_, beta_rep, x_k, U0, m)
        us.append(u_k)
        xs.append(A @ x_k + B @ u_k)
    xs = np.array(xs); us = np.array(us)
    t = DT * np.arange(N3 + 1)
    # 约束满足检查：u 均满足约束
    cons_viol = max(float(np.max(us - u_max)), float(np.max(-us - u_max)))
    print(f"  输入约束违规量（应≤0）= {cons_viol:.3e}")
    assert cons_viol <= 1e-9, "算法3 违反输入约束!"
    n_sat = int(np.sum(np.abs(us) >= u_max - 1e-6))
    print(f"  饱和步数 = {n_sat}/{N3}（约束活跃段占比 {n_sat/N3*100:.1f}%）")
    print(f"  终态 x({N3*DT:.0f}s) = {xs[-1]}，||x||∞ = {np.max(np.abs(xs[-1])):.3e}")
    print(f"  |x| 峰值 {np.max(np.abs(xs[:, 0])):.3f} m，|θ| 峰值 {np.max(np.abs(xs[:, 2])):.4f} rad")
    assert n_sat > 0, "输入约束应活跃（存在饱和段）!"
    assert np.max(np.abs(xs[-3:])) < 1e-3, "算法3 未收敛!"

    # 与 scipy SLSQP 对拍（首步）
    x_k = x0
    g_k = G.T @ x_k
    b_k = beta_rep - Mx_ @ Phi @ x_k
    res = minimize(lambda u: 0.5 * u @ H @ u + g_k @ u, np.zeros(Np * m),
                   method="SLSQP",
                   constraints={"type": "ineq", "fun": lambda u: b_k - M_k @ u})
    U_slsqp = res.x
    U_b = mpc_LQR_necons(H, G, Phi, M_k, Mx_, beta_rep, x_k,
                         np.zeros(Np * m), m)
    diff = float(np.max(np.abs(U_b[:10] - U_slsqp[:10])))
    print(f"  barrier vs SLSQP 首步前10维最大差 = {diff:.3e}")
    assert diff < 0.05, "算法3 与 SLSQP 不一致!"

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.0), sharex=True)
    axes[0].plot(t, xs[:, 0], label="x [m]")
    axes[0].plot(t, xs[:, 2] * 180.0 / np.pi, label=r"$\theta$ [°]")
    axes[0].set_ylabel("states"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(t[:-1], us[:, 0], color="C2", label="u = F [N]")
    axes[1].axhline(u_max, color="k", ls="--", lw=1, label=r"$\pm u_{max}$")
    axes[1].axhline(-u_max, color="k", ls="--", lw=1)
    axes[1].set_ylabel("input"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(t, xs[:, 1], color="C3", label=r"$\dot x$ [m/s]")
    axes[2].plot(t, xs[:, 3] * 180.0 / np.pi, color="C4", label=r"$\dot\theta$ [°/s]")
    axes[2].set_ylabel("velocities"); axes[2].set_xlabel("t [s]")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.suptitle("MPC-3 constrained regulator |u|<=1N (cartpole linearized)")
    _save_fig(fig, "mpc_necons.png")
    plt.close(fig)

    print("\n算法3 验证通过 ✔")


def _save_fig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  图已保存: {path}")


# ======================================================================
# gym 环境回合 J_ach 评估（无约束调节MPC）—— 与 src/1_pid.py、2_lqr.py 同框架，供"倒立摆不同算法对比验证"
# ======================================================================
class MPCController:
    """无约束线性MPC控制器（调节）—— 在 gym 非线性环境中的闭环包装。

    控制器在平衡点线性化离散模型上设计（cartpole_linear + mpc_build），
    在线步即 mpc_LQR_nocons（U* = F x_k，滚动优化只取首分量），
    施加前限幅到 ±f_max，与 src/2_lqr.py 的 LQRContinuousController 用法一致。
    """
    def __init__(self, Q, R, S, Np, f_max, dt):
        (Ac, Bc), (A, B) = cartpole_linear(dt=dt)
        F, K, G, H, Phi, Gamma, Qbar, Rbar = mpc_build(A, B, Q, R, S, Np)
        self.F = F
        self.f_max = f_max

    def compute_action(self, state):
        """根据状态计算控制力 u = clip(F x, -F_max, F_max)"""
        u = mpc_LQR_nocons(self.F, state, m=1)
        return np.clip(u, -self.f_max, self.f_max).astype(np.float32)


def _visualize_ep1(states, actions, path):
    """第1回合 5 面板时间序列图（x, x_dot, theta, theta_dot, F），样式同 src/1_pid.py、2_lqr.py。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    actions = np.array(actions).flatten()
    states = np.array(states)
    t = np.arange(len(states)) * CONFIG["dt"]   # 秒（dt = 0.02 s）
    theta_deg = states[:, 2] * 180.0 / np.pi
    theta_dot_deg = states[:, 3] * 180.0 / np.pi
    specs = [("x (Position)", "C0", states[:, 0]),
             (r"$\dot{x}$ (Velocity)", "C1", states[:, 1]),
             (r"$\theta$ (°)", "C2", theta_deg),
             (r"$\dot{\theta}$ (°/s)", "C3", theta_dot_deg)]
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Episode 1 Visualization (MPC)", fontsize=16)
    for i, (label, color, y) in enumerate(specs):
        axs[i].plot(t, y, label=label, color=color)
        axs[i].set_ylabel(label); axs[i].legend()
    axs[4].plot(t, actions, label="F (Force)", color="C4")
    axs[4].set_ylabel("F (Force)"); axs[4].set_xlabel("Time [s]"); axs[4].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存: {path}")


def test_mpc_policy():
    """在 gym 非线性环境上按 5 回合评估无约束调节MPC（与 PID/LQR 同框架，seed=42）。

    控制器设计于平衡点线性化离散模型，而环境为完整非线性动力学——
    这正是"模型驱动"设计（LQR 小节同款）在非线性系统上的有效性检验。
    """
    register_custom_cartpole()
    env = gym.make('CartPoleCustom-v1', disable_env_checker=True)
    np.random.seed(42)  # 固定种子以复现，与 src/1_pid.py、2_lqr.py 一致

    DT = CONFIG['dt']
    Np = 50
    Q = np.diag([5.0, 0.1, 10.0, 1.0])   # x, x_dot, theta, theta_dot
    R = np.array([[0.01]])
    S = 20.0 * Q
    controller = MPCController(Q, R, S, Np, CONFIG['F_max'], DT)

    MAX_EPISODES = 5
    # 统一评估口径（与 eval/bench.py 一致）：经典控制用 J_ach，不依赖环境奖励
    Q_eval = np.diag([1.0, 1.0, 100.0, 1.0])   # x, x_dot, theta, theta_dot
    R_eval = 1.0                                # u 为标量力
    episode_jachs, episode_lengths = [], []
    states_ep1 = actions_ep1 = None
    print("=" * 64)
    print("gym 环境回合评估：无约束线性MPC（调节）@倒立摆（seed=42，5 回合，max_steps=1000）")
    print("=" * 64)
    for episode in range(MAX_EPISODES):
        state, info = env.reset()
        j_ach, step_count = 0.0, 0
        terminated = truncated = False
        states, actions = [], []
        while not (terminated or truncated):
            xk = state  # 第 k 步施加动作前的状态，计入 J_ach
            action = controller.compute_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            u = float(info['F_applied'])  # 实际施加（裁剪后）的力
            j_ach += float(xk @ Q_eval @ xk) + R_eval * u * u
            state = next_state
            step_count += 1
            states.append(next_state)
            actions.append(action)
        episode_jachs.append(j_ach)
        episode_lengths.append(step_count)
        x, x_dot, theta, theta_dot = state
        reason = "达到最大步数"
        if abs(x) > CONFIG['x_threshold']:
            reason = f"x越界 (x={x:.3f} m)"
        elif abs(theta) > CONFIG['theta_threshold']:
            reason = f"θ越界 (θ={theta*180/np.pi:.1f}°)"
        print(f"  Episode {episode+1}: J_ach {j_ach:.2f}, 步数 {step_count} "
              f"({step_count*DT:.1f}s), 终止原因: {reason}")
        if episode == 0:
            states_ep1, actions_ep1 = states, actions

    mean_j, std_j = np.mean(episode_jachs), np.std(episode_jachs)
    mean_l, std_l = np.mean(episode_lengths), np.std(episode_lengths)
    print(f"  回合 J_ach: {[f'{v:.2f}' for v in episode_jachs]}")
    print(f"  平均 J_ach: {mean_j:.2f} ± {std_j:.2f}，平均步数: {mean_l:.1f} ± {std_l:.1f}")
    n_full = sum(1 for l in episode_lengths if l >= CONFIG['max_steps'])
    print(f"  满步回合: {n_full}/{MAX_EPISODES}")
    env.close()

    os.makedirs(FIG_DIR, exist_ok=True)
    _visualize_ep1(states_ep1, actions_ep1, os.path.join(FIG_DIR, "mpc_visualize_ep1.png"))


if __name__ == "__main__":
    verify_mpc()
    if os.environ.get("RUN_MPC_GYM") == "1":
        test_mpc_policy()
