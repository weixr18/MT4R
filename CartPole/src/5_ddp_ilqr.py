# -*- coding: utf-8 -*-
"""iLQR / DDP 非线性最优控制 —— 倒立摆验证（书 chap3_3_optmctrol_3.tex）

两个算法（对应书算法框，算法代码 / 验证脚本一体）：
  非线性最优控制-DDP    algo:OC_DDP_disc   —— 二阶：Q_xx/Q_xu/Q_uu 含 f 的二阶导项
  非线性最优控制-iLQR   algo:OC_iLQR_disc  —— 一阶：Q_xx/Q_xu/Q_uu 只含 f_x^T J_xx f_x（去 f 二阶导）

演示对象为 Part VIII 倒立摆的**完整非线性动力学**（与 src/env_cartpole.py 的 step() 逐字节一致：
解 2×2 线性方程得 x_ddot, θ_ddot，再用梯形/半隐式积分更新，θ wrap 到 [-π,π)）——即"模型 = 控制对象"。

运行（cwd 任意，推荐仓库根）：
  E:\\Anaconda3\\envs\\py311-gym\\python.exe src/5_ddp_ilqr.py
出图默认写入书仓库 1-MN4R/imgs/cartpole/，可用环境变量 FIG_DIR 覆盖。

结构：
  cartpole_f / cartpole_derivs       —— 非线性动力学 + 导数（f_x,f_u 解析，f_xx/f_xu/f_uu 由解析一阶差分）
  DDPILQRSolver                      —— 后向(贝尔曼) + 前向(轨迹滚动) + 代价线搜索 + LM 正则；mode=ilqr/ddp
  DDPILQRController                  —— 滚动时域控制闭环包装（warm-start），供 eval/bench.py 复用
  test_lq_anchor()                   —— LQ 锚点验证（线性 ZOH 模型 + S=P：1 次收敛到 x0^T P x0，K0≈F_LQR）
  test_nonlinear_episode()           —— 非线性倒立摆单回合演示（滚运时域），保存 5 面板图
"""
import os
import sys
import numpy as np

# 保证从任意 cwd 都能找到同目录模块（env_cartpole）
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from env_cartpole import CONFIG, register_custom_cartpole

# 出图目录：书仓库 imgs/cartpole/（脚本位于 4-MT4R-github/CartPole/src/）
FIG_DIR = os.environ.get("FIG_DIR", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "1-MN4R", "imgs", "cartpole")))

N_X = 4   # 状态维数 [x, x_dot, theta, theta_dot]
N_U = 1   # 输入维数（力 F）


# ======================================================================
# 1. 完整非线性动力学（模型 = 控制对象，复刻 env.step()）
# ======================================================================
def cartpole_f(x, u):
    """单步非线性动力学 f(x, u) -> x_{k+1}。

    完全复刻 CartPoleCustomEnv.step()：解 `_compute_accelerations` 的 2×2 线性方程
    得 x_ddot, θ_ddot，再梯形（半隐式）积分，θ wrap 到 [-π, π)。
    """
    x = np.asarray(x, dtype=float).ravel()
    u = float(np.atleast_1d(u)[0])
    M, m1, l1 = CONFIG['M'], CONFIG['m1'], CONFIG['l1']
    g, mu, dt = CONFIG['g'], CONFIG['mu'], CONFIG['dt']
    X, Xd, Th, Thd = x[0], x[1], x[2], x[3]
    c, s = np.cos(Th), np.sin(Th)

    A = np.array([[M + m1, m1 * l1 * c],
                  [c, 4 / 3 * l1]])
    b = np.array([u + m1 * l1 * s * Thd ** 2 - mu * Xd,
                  g * s])
    try:
        a = np.linalg.solve(A, b)
        Xdd, Thdd = a[0], a[1]
    except np.linalg.LinAlgError:
        Xdd, Thdd = 0.0, 0.0

    Xd_new = Xd + Xdd * dt
    Thd_new = Thd + Thdd * dt
    X_new = X + (Xd + Xd_new) * 0.5 * dt
    Th_new = Th + (Thd + Thd_new) * 0.5 * dt
    Th_new = ((Th_new + np.pi) % (2 * np.pi)) - np.pi
    return np.array([X_new, Xd_new, Th_new, Thd_new])


def cartpole_derivs_first(x, u):
    """解析一阶导数 f_x (4×4), f_u (4×1)。

    - 对加速度系统 A·[x_ddot;θ_ddot]=b 手推偏导（系数见书 chap8_1_1storder_1.tex），
      再按梯形积分公式组合。θ 的 wrap 在 |θ|<π 内为恒等映射，故导数在可行域内有效。
    - 与数值中心差分交叉校验（见 test 用）。
    """
    x = np.asarray(x, dtype=float).ravel()
    u = float(np.atleast_1d(u)[0])
    M, m1, l1 = CONFIG['M'], CONFIG['m1'], CONFIG['l1']
    g, mu, dt = CONFIG['g'], CONFIG['mu'], CONFIG['dt']
    X, Xd, Th, Thd = x[0], x[1], x[2], x[3]
    c, s = np.cos(Th), np.sin(Th)

    A11 = M + m1            # 常数
    A22 = 4 / 3 * l1        # 常数
    A12 = m1 * l1 * c       # 随 θ 变
    D = A11 * A22 - A12 * c

    b1 = u + m1 * l1 * s * Thd ** 2 - mu * Xd
    b2 = g * s
    num1 = b1 * A22 - A12 * b2            # x_ddot 分子
    num2 = A11 * b2 - c * b1              # θ_ddot 分子

    # 中间量对 (xdot, theta, thdot, F) 的偏导
    dD_dth = 2 * m1 * l1 * c * s
    db1_dxd = -mu
    db1_dth = m1 * l1 * c * Thd ** 2
    db1_dthd = 2 * m1 * l1 * s * Thd
    db1_dF = 1.0
    db2_dth = g * c

    dnum1_dxd = db1_dxd * A22
    dnum1_dth = db1_dth * A22 - (-m1 * l1 * s) * b2 - A12 * db2_dth
    dnum1_dthd = db1_dthd * A22
    dnum1_dF = A22

    dnum2_dxd = mu * c
    dnum2_dth = A11 * db2_dth - (-s) * b1 - c * db1_dth
    dnum2_dthd = -c * db1_dthd
    dnum2_dF = -c

    # 对每个参数求 ∂x_ddot/∂p, ∂θ_ddot/∂p
    params = ['x', 'xd', 'th', 'thd', 'F']
    d_x = {'x': 0.0, 'xd': dnum1_dxd, 'th': dnum1_dth, 'thd': dnum1_dthd, 'F': dnum1_dF}
    d_th = {'x': 0.0, 'xd': dnum2_dxd, 'th': dnum2_dth, 'thd': dnum2_dthd, 'F': dnum2_dF}
    dXdd = {}
    dThdd = {}
    for p in params:
        # ∂(num/D)/∂p = (∂num/∂p * D - num * ∂D/∂p) / D^2；仅 θ 时 ∂D/∂p 非零
        if p == 'th':
            dXdd[p] = (d_x[p] * D - num1 * dD_dth) / (D * D)
            dThdd[p] = (d_th[p] * D - num2 * dD_dth) / (D * D)
        else:
            dXdd[p] = d_x[p] / D
            dThdd[p] = d_th[p] / D

    # f_1 = X + Xd*dt + 0.5*dt^2 * Xdd
    # f_2 = Xd + dt * Xdd
    # f_3 = Th + Thd*dt + 0.5*dt^2 * Thdd
    # f_4 = Thd + dt * Thdd
    f_x = np.zeros((N_X, N_X))
    f_u = np.zeros((N_X, N_U))
    # 列顺序（对 x_j 求导）：x, xd, th, thd
    f_x[0, 0] = 1.0
    f_x[0, 1] = dt + 0.5 * dt ** 2 * dXdd['xd']
    f_x[0, 2] = 0.5 * dt ** 2 * dXdd['th']
    f_x[0, 3] = 0.5 * dt ** 2 * dXdd['thd']
    f_x[1, 1] = 1.0 + dt * dXdd['xd']
    f_x[1, 2] = dt * dXdd['th']
    f_x[1, 3] = dt * dXdd['thd']
    f_x[2, 0] = 0.0
    f_x[2, 1] = 0.5 * dt ** 2 * dThdd['xd']
    f_x[2, 2] = 1.0 + 0.5 * dt ** 2 * dThdd['th']
    f_x[2, 3] = dt + 0.5 * dt ** 2 * dThdd['thd']
    f_x[3, 1] = dt * dThdd['xd']
    f_x[3, 2] = dt * dThdd['th']
    f_x[3, 3] = 1.0 + dt * dThdd['thd']

    f_u[0, 0] = 0.5 * dt ** 2 * dXdd['F']
    f_u[1, 0] = dt * dXdd['F']
    f_u[2, 0] = 0.5 * dt ** 2 * dThdd['F']
    f_u[3, 0] = dt * dThdd['F']
    return f_x, f_u


def cartpole_derivs(x, u, h=1e-6):
    """f 的一阶与二阶导数，供 iLQR/DDP 使用。

    - 一阶：解析（cartpole_derivs_first）。
    - 二阶：对解析一阶导数做中心差分（f_xx/f_xu/f_uu），比直接对 f 套四点差分级联噪声更小、更快。
    返回 (f_x(n,n), f_u(n,m), f_xx(n,n,n), f_xu(n,m,n), f_uu(m,m,n))，
    约定 f_xx[:, :, i] 为分量 f_i 的 Hessian，f_xu[:, :, i] 为 ∂²f_i/∂x∂u，f_uu[:, :, i] 为 ∂²f_i/∂u²。
    """
    fx = np.zeros((N_X, N_X))
    fu = np.zeros((N_X, N_U))
    x = np.asarray(x, dtype=float).ravel()
    u = float(np.atleast_1d(u)[0])

    fx, fu = cartpole_derivs_first(x, u)
    fxx = np.zeros((N_X, N_X, N_X))   # [in_x, in_x, out]
    fxu = np.zeros((N_X, N_U, N_X))   # [in_x, in_u, out]
    fuu = np.zeros((N_U, N_U, N_X))   # [in_u, in_u, out]

    # 一阶导对 x 的差分 -> f_xx[in_x, in_x', out] = ∂²f_out/∂x_inx∂x_inx'
    for j in range(N_X):
        dx = np.zeros(N_X); dx[j] = h
        fxp, _ = cartpole_derivs_first(x + dx, u)
        fxm, _ = cartpole_derivs_first(x - dx, u)
        d = (fxp - fxm) / (2 * h)                      # (out, in_x')
        fxx[j, :, :] = d.T                             # 对称：∂(∂f_out/∂x_inx')/∂x_inx

    # 一阶导对 u 的差分 -> f_xx 的 x-u 混合块 与 f_uu
    for j in range(N_U):
        du = np.zeros(N_U); du[j] = h
        fxp, fup = cartpole_derivs_first(x, u + du[j])
        fxm, fum = cartpole_derivs_first(x, u - du[j])
        d = (fxp - fxm) / (2 * h)                      # (out, in_x) = ∂f_x/∂u
        fxu[:, j, :] = d.T                             # [in_x, in_u, out]
        duu = (fup - fum) / (2 * h)                    # (out, in_u) = ∂f_u/∂u
        fuu[j, :, :] = duu.T                           # [in_u, in_u, out]

    # 使二阶张量的 x-x 块对称（数值微分会引入微小不对称）
    fxx = 0.5 * (fxx + np.transpose(fxx, (1, 0, 2)))
    return fx, fu, fxx, fxu, fuu


def cartpole_derivs_ilqr(x, u, h=1e-6):
    """iLQR 用的一阶导数（跳过二阶有限差分以提速）。

    与 cartpole_derivs 返回结构一致：(f_x, f_u, f_xx, f_xu, f_uu)，
    其中后三者以零张量占位——iLQR 后向递推不消费 f 的二阶导（见 DDPILQRSolver._backward），
    故跳过数值差分不影响 iLQR 结果，仅大幅降低单步求解开销。
    """
    fx, fu = cartpole_derivs_first(x, u)
    return (fx, fu,
            np.zeros((N_X, N_X, N_X)),
            np.zeros((N_X, N_U, N_X)),
            np.zeros((N_U, N_U, N_X)))


# ======================================================================
# 2. 通用 iLQR / DDP 求解器（后向 + 前向 + 线搜索 + LM 正则）
# ======================================================================
def _contract_fxx(J_x, fxx):
    """Σ_i J_x[i] · fxx[:, :, i]  -> (n,n)"""
    return np.einsum('o,abo->ab', J_x, fxx)


def _contract_fxu(J_x, fxu):
    """Σ_i J_x[i] · fxu[:, :, i]  -> (n,m)"""
    return np.einsum('o,abo->ab', J_x, fxu)


def _contract_fuu(J_x, fuu):
    """Σ_i J_x[i] · fuu[:, :, i]  -> (m,m)"""
    return np.einsum('o,abo->ab', J_x, fuu)


class DDPILQRSolver:
    """通用离散 iLQR / DDP 求解器。

    求解 min_u h(x_N) + Σ_{k=0}^{N-1} g(x_k, u_k)，g=x^TQx+u^TRu，h=x^T S x。
    - `mode='ilqr'`：Q_xx = g_xx + f_x^T J_xx f_x，Q_xu/Q_uu 同理只含 f_x^T J_xx f_·（去 f 二阶导）。
    - `mode='ddp'`：Q_xx/Q_xu/Q_uu 再补 Σ J_x·f_· 二阶导项（书上 DDP）。
    标准式（含 f_x^T J_xx f_x）见 chap3_3_optmctrol_3.tex，是本类的对照基准。
    """
    def __init__(self, f, f_derivs, Q, R, S, N, dt, fmax,
                 mode='ilqr',
                 reg_init=1e-3, reg_inc=4.0, reg_dec=0.4, reg_max=1e6,
                 alpha_init=1.0, line_search_iters=12, max_iter=30):
        self.f = f
        self.f_derivs = f_derivs
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.S = np.asarray(S, dtype=float)
        self.N = int(N)
        self.dt = dt
        self.fmax = fmax
        self.mode = mode
        self.reg_init = reg_init
        self.reg_inc = reg_inc
        self.reg_dec = reg_dec
        self.reg_max = reg_max
        self.alpha_init = alpha_init
        self.line_search_iters = line_search_iters
        self.max_iter = max_iter

    # --- 代价及其导数 ---------------------------------------------------
    def _cost(self, x_tr, u_tr):
        J = 0.0
        for k in range(self.N):
            J += float(x_tr[k] @ self.Q @ x_tr[k])
            J += float(u_tr[k] @ self.R @ u_tr[k])
        J += float(x_tr[self.N] @ self.S @ x_tr[self.N])
        return J

    def _roll(self, x0, u_tr):
        xs = [np.asarray(x0, dtype=float).ravel().copy()]
        for k in range(self.N):
            xs.append(self.f(xs[-1], u_tr[k]))
        return np.array(xs)

    def _init_u(self, x0, u_init):
        if u_init is not None:
            return np.asarray(u_init, dtype=float).reshape(self.N, N_U).copy()
        return np.zeros((self.N, N_U))

    # --- 后向（贝尔曼）递推 ----------------------------------------------
    def _backward(self, x_bar, u_bar, reg):
        n, m = N_X, N_U
        d = np.zeros((self.N, m))
        K = np.zeros((self.N, m, n))
        # 终端价值函数导数 h(x_N)=x^T S x
        J_x = (self.S + self.S.T) @ x_bar[self.N]
        J_xx = self.S + self.S.T
        for k in range(self.N - 1, -1, -1):
            xk, uk = x_bar[k], u_bar[k]
            f_x, f_u, f_xx, f_xu, f_uu = self.f_derivs(xk, uk)
            g_x = (self.Q + self.Q.T) @ xk
            g_u = (self.R + self.R.T) @ uk
            g_xx = self.Q + self.Q.T
            g_xu = np.zeros((n, m))
            g_uu = self.R + self.R.T

            Qx = g_x + f_x.T @ J_x
            Qu = g_u + f_u.T @ J_x
            Qxx = g_xx + f_x.T @ J_xx @ f_x
            Qxu = g_xu + f_x.T @ J_xx @ f_u
            Quu = g_uu + f_u.T @ J_xx @ f_u
            if self.mode == 'ddp':
                Qxx = Qxx + _contract_fxx(J_x, f_xx)
                Qxu = Qxu + _contract_fxu(J_x, f_xu)
                Quu = Quu + _contract_fuu(J_x, f_uu)

            Quu_reg = Quu + reg * np.eye(m)
            Qux = Qxu.T                       # (m,n)
            # 求解 (Q_uu+λI)
            K[k] = -np.linalg.solve(Quu_reg, Qux)   # m×n = -Q_uu^{-1} Q_ux
            d[k] = -np.linalg.solve(Quu_reg, Qu)    # m   = -Q_uu^{-1} Q_u
            # 更新价值函数导数
            J_x = Qx - Qxu @ np.linalg.solve(Quu_reg, Qu)
            J_xx = Qxx - Qxu @ np.linalg.solve(Quu_reg, Qux)
        return K, d

    # --- 前向（轨迹滚动 + 线搜索） -----------------------------------------
    def _forward(self, x0, x_bar, u_bar, K, d, alpha):
        n, m = N_X, N_U
        xs = np.zeros((self.N + 1, n))
        us = np.zeros((self.N, m))
        x = np.asarray(x0, dtype=float).ravel().copy()
        xs[0] = x
        for k in range(self.N):
            du = alpha * d[k] + K[k] @ (x - x_bar[k])
            uk = u_bar[k] + du
            uk = np.clip(uk, -self.fmax, self.fmax)
            us[k] = uk
            x = self.f(x, uk)
            xs[k + 1] = x
        return xs, us

    # --- 外层轨迹迭代 -----------------------------------------------------
    def solve(self, x0, u_init=None, tol=1e-8, max_iter=None):
        if max_iter is None:
            max_iter = self.max_iter
        x0 = np.asarray(x0, dtype=float).ravel()
        u_bar = self._init_u(x0, u_init)
        x_bar = self._roll(x0, u_bar)
        J = self._cost(x_bar, u_bar)
        history = [J]
        reg = self.reg_init

        for itr in range(max_iter):
            K, d = self._backward(x_bar, u_bar, reg)
            alpha = self.alpha_init
            improved = False
            for _ in range(self.line_search_iters):
                xs_new, us_new = self._forward(x0, x_bar, u_bar, K, d, alpha)
                J_new = self._cost(xs_new, us_new)
                if J_new <= J + 1e-9:                # 下降（含数值容差）即接受
                    x_bar, u_bar = xs_new, us_new
                    J = J_new
                    improved = True
                    break
                alpha *= 0.5
            if improved:
                reg = max(self.reg_init, reg * self.reg_dec)     # 减少正则
            else:
                reg *= self.reg_inc                                # 增大正则（LM）
                if reg > self.reg_max:
                    break
            history.append(J)
            # 收敛判据：代价不再显著变化
            if len(history) >= 2 and abs(J - history[-2]) <= tol * max(1.0, abs(history[-2])):
                break
        return x_bar, u_bar, K, d, history


# ======================================================================
# 3. 线性模型包装（LQ 锚点验证用）
# ======================================================================
def _make_linear_model(A_d, B_d):
    """线性模型 f(x,u)=A_d x+B_d u 的 f 与导数（f 二阶导为 0）。"""
    A_d = np.asarray(A_d, dtype=float)
    B_d = np.asarray(B_d, dtype=float)

    def f(x, u):
        return A_d @ np.asarray(x, dtype=float).ravel() + B_d @ np.atleast_1d(u)

    def derivs(x, u):
        n, m = N_X, N_U
        return (A_d.copy(), B_d.copy(),
                np.zeros((n, n, n)), np.zeros((n, m, n)), np.zeros((m, m, n)))
    return f, derivs


# ======================================================================
# 4. 滚动时域控制闭环包装（供 bench 复用）
# ======================================================================
def _lqr_gain(Q, R, dt):
    """由平衡点线性化 + ZOH 离散模型求离散 LQR 增益 F（用于首解种子轨迹）。"""
    from scipy.linalg import solve_discrete_are
    A, B = _cartpole_linear_cont()
    A_d, B_d = _zoh_discretize(A, B, dt)
    P = solve_discrete_are(A_d, B_d, Q, R)
    F = -np.linalg.solve(R + B_d.T @ P @ B_d, B_d.T @ P @ A_d)
    return F


def _lqr_seed_trajectory(x0, N, f, F, fmax):
    """用 LQR 反馈在**非线性**模型上滚出一条初始控制轨迹（用作 iLQR/DDP 首解的种子）。

    倒立摆这类非线性 OCP 存多个局部最优：若从零控制种子开始，iLQR/DDP 可能收敛到
    "甩一整圈"的摆动局部最优（θ 到 ~175°）而非竖直镇定。先滚一条稳定的小角度轨迹
    （LQR 反馈，θ 幅值严格保持在低位），把求解器留在"竖直镇定"的势能盆地，再行优化。
    F 只用于确定初始盆地，最终的 u* 仍由完整非线性模型求解。
    """
    u = np.zeros((N, N_U))
    x = np.asarray(x0, dtype=float).ravel().copy()
    for k in range(N):
        uk = np.clip(F @ x, -fmax, fmax)
        u[k, 0] = uk
        x = f(x, uk)
    return u


class DDPILQRController:
    """滚动时域（receding-horizon）iLQR / DDP 控制器。

    每次 compute_action(state)：从当前状态做一次有限时域 N 的 iLQR/DDP 求解，
    取首分量 u_0（限幅到 ±f_max）施加。采用 warm-start（把上一次解的 (x^ref, u^ref)
    平移一步作本次初值），通常 1–3 次外循环即可收敛；首个解用 LQR 反馈在非线性模型上
    滚出的种子轨迹初始化（见 _lqr_seed_trajectory）。

    注意：iLQR/DDP 原生是"整段轨迹优化"；这里将其用作**反馈控制器**（每步重解、
    只执行首控制分量），与 27.4 线性 MPC 的"每步重规划"平行。
    """
    def __init__(self, Q, R, S, N, fmax, dt, mode='ilqr',
                 solver_iter=25, warm_start=True):
        self.f = cartpole_f
        # DDP 用完整二阶导数；iLQR 只消费一阶导数，跳过二阶有限差分以提速
        f_derivs = cartpole_derivs if mode == 'ddp' else cartpole_derivs_ilqr
        self.f_derivs = f_derivs
        self.solver = DDPILQRSolver(cartpole_f, f_derivs, Q, R, S, N, dt,
                                    fmax, mode=mode, max_iter=solver_iter)
        self.N = N
        self.mode = mode
        self.fmax = fmax
        self.warm_start = warm_start
        self._F = _lqr_gain(Q, R, dt)   # 仅用于首解种子轨迹
        self._x_bar = None
        self._u_bar = None

    def reset(self):
        self._x_bar = None
        self._u_bar = None

    def compute_action(self, state):
        state = np.asarray(state, dtype=float).ravel()
        u_init = None
        if self.warm_start and self._u_bar is not None:
            # 平移一步作为初值：u 序列左移，末位补 0；x 轨迹由当前状态重滚
            u_init = np.vstack([self._u_bar[1:], np.zeros((1, N_U))])
        else:
            # 首个解：用 LQR 反馈在非线性模型上滚出的种子轨迹，锁定"竖直镇定"盆地
            u_init = _lqr_seed_trajectory(state, self.N, cartpole_f, self._F, self.fmax)
        x_bar, u_bar, K, d, hist = self.solver.solve(state, u_init=u_init)
        u0 = float(np.atleast_1d(u_bar[0])[0])
        self._x_bar = x_bar
        self._u_bar = u_bar
        return np.array([np.clip(u0, -self.fmax, self.fmax)], dtype=np.float32)


# ======================================================================
# 5. LQ 锚点验证（先确证算法本身，独立于倒立摆）
# ======================================================================
def test_lq_anchor():
    """在线性化 ZOH 模型 + Q/R + S=P 上跑 iLQR/DDP，验收：
      ① 代价 1 次收敛到解析最优 x0^T P x0；② K_0（首步反馈增益）≈ F_LQR；
      ③ 开环轨迹与线性 LQ 解析轨迹一致。
    """
    from scipy.linalg import solve_discrete_are

    DT = CONFIG['dt']
    Q = np.diag([1.0, 1.0, 100.0, 1.0])
    R = np.array([[1.0]])

    # 复刻 3_lqr_discrete.py 的平衡点线性化 + 精确 ZOH 离散化
    A, B = _cartpole_linear_cont()
    A_d, B_d = _zoh_discretize(A, B, DT)
    P = solve_discrete_are(A_d, B_d, Q, R)
    F_lqr = -np.linalg.solve(R + B_d.T @ P @ B_d, B_d.T @ P @ A_d)
    x0 = np.array([0.3, 0.2, 0.15, -0.1])
    J_star = float(x0 @ P @ x0)

    print("=" * 70)
    print("LQ 锚点验证：iLQR / DDP @ 线性化 ZOH 模型（S = P_DARE）")
    print("=" * 70)
    print(f"  dt={DT}, N=50, x0={x0}")
    print(f"  J* = x0^T P x0 = {J_star:.8f}")
    print(f"  F_LQR = {F_lqr.ravel()}")

    N_anchor = 50
    f_lin, der_lin = _make_linear_model(A_d, B_d)

    for mode in ('ilqr', 'ddp'):
        # 线性二次问题上 iLQR/DDP 是精确 Newton 步，无需 LM 正则；reg_init=0 使首步即达解析最优
        solver = DDPILQRSolver(f_lin, der_lin, Q, R, P, N_anchor, DT,
                               CONFIG['F_max'], mode=mode, max_iter=8, reg_init=0.0)
        x_bar, u_bar, K, d, hist = solver.solve(x0, u_init=np.zeros((N_anchor, N_U)))
        J1 = solver._cost(x_bar, u_bar)
        print(f"\n[{mode}]")
        print(f"  迭代代价: {[f'{v:.6f}' for v in hist]}")
        print(f"  1 次更新后 J = {hist[1]:.8f}，最终 J = {J1:.8f}（J* = {J_star:.8f}）")
        # ① 首次外循环更新即收敛到解析最优 x0^T P x0（后续仅确认、无实质变化）
        assert abs(hist[1] - J_star) < 1e-4, f"{mode}: 未在 1 次迭代收敛到 J*（hist={hist}）"
        assert abs(J1 - J_star) < 1e-6, f"{mode}: J 未收敛到解析最优 {J_star}"
        # ② 首步反馈增益 K_0 应与 F_LQR 一致
        K0 = K[0].reshape(1, N_X)
        print(f"  K_0 = {K0}  (与 F_LQR 应一致)")
        assert np.allclose(K0, F_lqr, atol=1e-6), f"{mode}: K_0 与 F_LQR 不一致"
        # ③ 开环轨迹与线性 LQ 解析轨迹一致
        xs_open = [x0]
        for k in range(N_anchor):
            xs_open.append(A_d @ xs_open[-1] + B_d @ u_bar[k])
        xs_open = np.array(xs_open)
        assert np.allclose(xs_open, x_bar, atol=1e-6), f"{mode}: 开环轨迹不一致"
        print(f"  ✓ 开环轨迹 = 线性 LQ 解析轨迹（{mode}）")

    print("\nLQ 锚点验证通过 ✔（iLQR 与 DDP 均 1 次收敛到 J*，K_0=F_LQR）")


# ======================================================================
# 6. 非线性非线性倒立摆单回合演示（滚动时域）+ 出图
# ======================================================================
def _visualize_episode(x_seq, u_seq, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    states = np.asarray(x_seq)[1:]
    actions = np.asarray(u_seq).flatten()
    t = np.arange(len(states)) * CONFIG["dt"]   # 秒（dt = 0.02 s）
    theta_deg = states[:, 2] * 180.0 / np.pi
    theta_dot_deg = states[:, 3] * 180.0 / np.pi
    specs = [("x (Position)", "C0", states[:, 0]),
             (r"$\dot{x}$ (Velocity)", "C1", states[:, 1]),
             (r"$\theta$ (°)", "C2", theta_deg),
             (r"$\dot{\theta}$ (°/s)", "C3", theta_dot_deg)]
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title, fontsize=16)
    for i, (label, color, y) in enumerate(specs):
        axs[i].plot(t, y, label=label, color=color)
        axs[i].set_ylabel(label); axs[i].legend()
    axs[4].plot(t, actions, label="F (Force)", color="C4")
    axs[4].set_ylabel("F (Force)"); axs[4].set_xlabel("Time [s]"); axs[4].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存: {path}")


def test_nonlinear_episode(mode='ddp', N=100):
    """在 gym 非线性倒立摆上做单回合滚动时域演示（固定初值），保存 5 面板图。

    选取一个偏大初始摆角（θ≈30°）以体现 iLQR/DDP 在偏离平衡点的非线性工况下的表现。
    """
    register_custom_cartpole()
    import gym
    env = gym.make('CartPoleCustom-v1', disable_env_checker=True)

    from scipy.linalg import solve_discrete_are
    A, B = _cartpole_linear_cont()
    A_d, B_d = _zoh_discretize(A, B, CONFIG['dt'])
    Q = np.diag([1.0, 1.0, 100.0, 1.0])
    R = np.array([[1.0]])
    S = solve_discrete_are(A_d, B_d, Q, R)

    x0 = np.array([0.5, 0.0, 30 * np.pi / 180, 0.0])
    env.reset()   # gym 0.26 要求先 reset 再 step；随后显式设置本次初值
    env.state = np.array(x0, dtype=np.float32)
    env.steps = 0
    ctrl = DDPILQRController(Q, R, S, N, CONFIG['F_max'], CONFIG['dt'],
                             mode=mode, solver_iter=20, warm_start=True)

    xs, us = [x0.copy()], []
    term = False
    step = 0
    while not term and step < CONFIG['max_steps']:
        a = ctrl.compute_action(xs[-1])
        ns, _, term, _, info = env.step(a)
        us.append(float(info['F_applied']))
        xs.append(np.array(ns, dtype=float))
        step += 1
    xs = np.array(xs); us = np.array(us)
    t = CONFIG['dt'] * np.arange(len(xs))
    print("=" * 70)
    print(f"非线性倒立摆单回合演示（{mode.upper()}，N={N}，滚动时域，x0 θ=30°）")
    print("=" * 70)
    print(f"  回合步数: {step} ({step * CONFIG['dt']:.1f}s)")
    print(f"  终态: x={xs[-1,0]:.3f}, ẋ={xs[-1,1]:.3f}, θ={xs[-1,2]*180/np.pi:.3f}°, θ̇={xs[-1,3]*180/np.pi:.3f}°/s")
    reason = "满步"
    if abs(xs[-1, 0]) > CONFIG['x_threshold']:
        reason = "x越界"
    elif abs(xs[-1, 2]) > CONFIG['theta_threshold']:
        reason = "θ越界"
    print(f"  终止原因: {reason}")
    name = f"{mode}_nonlinear_ep1"
    _visualize_episode(xs, us, f"Nonlinear cartpole - {mode.upper()} (NH) - Ep1",
                       os.path.join(FIG_DIR, name + ".png"))
    env.close()


# ======================================================================
# 7. 辅助：平衡点线性化 + 精确 ZOH 离散（单一来源；供 LQ 锚点 / 终端 S 用）
# ======================================================================
def _cartpole_linear_cont():
    """连续平衡点线性化 (A,B)，系数 k1..k6 见书 chap8_1_1storder_1.tex。"""
    M, m1, l1, mu, g = CONFIG['M'], CONFIG['m1'], CONFIG['l1'], CONFIG['mu'], CONFIG['g']
    D = 4 * M + m1
    k1 = 4.0 / D
    k2 = -3.0 * m1 * g / D
    k3 = 3.0 * (M + m1) * g / (l1 * D)
    k4 = -3.0 / (l1 * D)
    k6 = 3.0 * mu / (l1 * D)
    A = np.array([[0., 1., 0., 0.],
                  [0., -mu * k1, k2, 0.],
                  [0., 0., 0., 1.],
                  [0., k6, k3, 0.]])
    B = np.array([[0., k1, 0., k4]]).T
    return A, B


def _zoh_discretize(A, B, dt):
    from scipy.linalg import expm
    n, m = A.shape[0], B.shape[1]
    MA = np.block([[A, B], [np.zeros((m, n)), np.zeros((m, m))]])
    exp_M = expm(MA * dt)
    return exp_M[:n, :n], exp_M[:n, n:]


# ======================================================================
# 8. 入口
# ======================================================================
if __name__ == "__main__":
    # ① LQ 锚点验证（算法本身正确性，独立于倒立摆）
    test_lq_anchor()
    # ② 非线性倒立摆单回合演示（滚动时域，DDP 与 iLQR 各一份）
    print()
    test_nonlinear_episode(mode='ddp', N=100)
    print()
    test_nonlinear_episode(mode='ilqr', N=100)
