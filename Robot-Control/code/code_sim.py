# -*- coding: utf-8 -*-
"""仿真层（计划 §5.2.1）：被控对象状态方程 + 定步长 RK4 + 控制零阶保持 + 全部记录钩子。

被控对象直接由书中的动力学方程得到（`F_f` 为摩擦矩阵、`τ_e` 为外力矩钩子）：

    q̈ = B(q)⁻¹ ( u + τ_e − C(q, q̇) q̇ − F_f q̇ − g(q) )

| 设计项 | 取值 / 做法 |
|---|---|
| 积分器 | RK4 **定步长**，默认 `h_sim = 1 ms`（计划 §5.2.1；高增益下若数值不稳，先排查物理而非直接缩步长）|
| 控制周期 | `h_c` 默认 `= h_sim`，**零阶保持**（`h_c = k·h_sim` 时每 `k` 步更新一次控制） |
| 摩擦 `F_f` | **基线 `None`（即 0）**：`model/code_dynamics.py` 未含该项、书中方程含它，故必须在此显式加入 |
| 预置钩子 | 外力矩 `tau_e(t, q, q̇)`、**测量噪声 + 低通滤波（阶段 4 启用，见 `noise`）**、参数失配（`code_models.make_case(..., plant=…)`）|
| 记录 | `t, q, q̇, u, x_e, x̃, q_d, q̃, σ_min(J_a), σ_max(J_a), cond(B), g(q), u−g（"反馈部分"）, 控制器实际看到的量 (q_meas, q̇_meas), 是否饱和/NaN/异常` |

**符号约定**（与书中一致）：`q̃ = q_d − q`、`x̃ = x_d − x_e`（都是「期望 − 实际」）。

**红线（`Robot-Control/AGENTS.md`「关键约定」2）**：本层是**被控对象**；控制器模型由 `case` 单独给出。
阶段 2/3 两者同源是**有意为之的标称用例**（并已在此显式标注）——它只能证明「实现与公式自洽」，
排除「书的模型本身错了」由阶段 1（Pinocchio 对拍）承担，量化对模型误差的敏感度由阶段 4 承担。

运行环境：`E:\\Anaconda3\\envs\\py311-gym\\python.exe`（加 `PYTHONUTF8=1`）。
"""
import time

import numpy as np

DEFAULT_DT = 1e-3        # h_sim：RK4 步长 1 ms（计划 §5.2.1）
DEFAULT_HC = 1e-3        # h_c：控制周期 1 ms（零阶保持）


def plant_qddot(q, qdot, tau, case, F_f=None, tau_e=None):
    """被控对象加速度：`q̈ = B⁻¹(τ + τ_e − Cq̇ − F_fq̇ − g)`（长度 `N+1`，`[0]` 不用）。

    `tau_e` 为 `None` 或长度 `N+1` 的外力矩向量（阶段 4 / 力控批次的钩子）。
    """
    B, C, g = case.dyn(q, qdot)
    rhs = tau[1:] - C @ qdot[1:] - g
    if F_f is not None:
        rhs = rhs - F_f @ qdot[1:]
    if tau_e is not None:
        rhs = rhs + tau_e[1:]
    qddot = np.zeros(case.N + 1)
    qddot[1:] = np.linalg.solve(B, rhs)
    return qddot


def rk4_step(f, q, qdot, dt):
    """RK4 一步（状态 `y = [q; q̇]`，`f(q, q̇) -> q̈`，均为长度 `N+1`）。

    ⚠️ **2026-09-20 修掉一个真 bug（阶段 4 期间发现，见书仓库日志「尝试 12」）**：
    `q_new` 的权重原本写成 `(a1 + 2a2 + a3)/6`，**多算了 `a2`**；由增广一阶系统
    `ẏ = [q̇; a(q,q̇)]` 上做标准 RK4 可推出正确组合是 `(a1 + a2 + a3)/6`
    （`q̇_new` 的 `(a1 + 2a2 + 2a3 + a4)/6` 本来就是对的）。
    错版的**全局精度只有一阶**（实测经验阶 1.000，正确版 4.0，同一算例误差小 1.35e6 倍），
    其稳定域也远小于 RK4 的 `hω* = 2.785`——这正是阶段 4 里
    「失稳阈值 `hω* ≈ 0.85` 与 RK4 理论差 3.3×」的真正原因，也放大了阶段 3 判据 1 的
    `O(dt)` 偏差。**这不是「离散化本身的残差」，而是积分器的实现错误。**
    守卫见 `test/model/test_convention.py` 的 `check_integrator_order`（收敛阶必须 ≈4）。
    """
    a1 = f(q, qdot)
    a2 = f(q + 0.5 * dt * qdot, qdot + 0.5 * dt * a1)
    a3 = f(q + 0.5 * dt * qdot + 0.25 * dt * dt * a1, qdot + 0.5 * dt * a2)
    a4 = f(q + dt * qdot + 0.5 * dt * dt * a2, qdot + dt * a3)
    q_new = q + dt * qdot + dt * dt / 6.0 * (a1 + a2 + a3)
    qdot_new = qdot + dt / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    return q_new, qdot_new


def make_friction_tau_e(viscous=None, coulomb=None, eps=1e-3, N=None):
    """构造**摩擦外力矩**钩子（阶段 4：「plant 有摩擦、控制器不补偿」的用例）。

    `τ_f = −(F_v q̇ + F_c tanh(q̇/ε))`，即以 `tanh` 光滑化的**粘性 + 库仑**摩擦，
    作为 `tau_e` 加进被控对象方程（`code_sim.simulate(..., tau_e=…)`）。
    ⚠️ 走 `tau_e` 而不是 `F_f` 参数：`F_f` 在方程里是**线性**项 `F_f q̇`，表达不了库仑摩擦的
    「静摩擦死区」；而 `tau_e` 是任意函数，两者物理上都在同一处相加。

    `viscous` / `coulomb`：长度 `N` 的对角元（关节 1..N，`[0]` 不用）。返回 `callable(t,q,q̇)`。
    """
    fv = np.zeros(N) if viscous is None else np.asarray(viscous, dtype=float)
    fc = np.zeros(N) if coulomb is None else np.asarray(coulomb, dtype=float)
    if N is None:
        N = max(fv.size, fc.size)
    fv = np.broadcast_to(fv, (N,)).astype(float)
    fc = np.broadcast_to(fc, (N,)).astype(float)
    eps = float(eps)

    def _tau_e(_t, _q, qdot):
        out = np.zeros(N + 1)
        out[1:] = -(fv * qdot[1:] + fc * np.tanh(qdot[1:] / eps))
        return out

    return _tau_e


def simulate(case, ctrl, ref, q0, qdot0, T, dt=DEFAULT_DT, h_c=None,
             F_f=None, tau_e=None, noise=None, progress=False, label="", log_frac=0.1):
    """定步长 RK4 + 控制零阶保持的闭环仿真，返回**全部记录**（见模块文档的表）。

    参数
    ----
    case : `code_models.RobotCase`
    ctrl : callable `(state, ref) -> u`，`state = (q, q̇)`；**已绑定**增益与模型（见 `code_ctrl_common`）
    ref : 参考量字典，或 `t -> 字典`（调节/跟踪两种形式，见 `code_traj`）
    q0, qdot0 : 初值（长度 `N+1`）
    T : 仿真时长 (s)；`dt`/`h_c` : 积分步长 / 控制周期 (s)
    F_f : `N×N` 摩擦矩阵或 `None`（基线）；`tau_e` : `None` 或 `callable(t, q, q̇) -> 长度 N+1`
    noise : `None` 或**测量噪声规格**（阶段 4 新增）：

        | 键 | 含义 |
        |---|---|
        | `q_sigma` | 关节位置测量的高斯噪声标准差 (rad) |
        | `qd_sigma` | 关节速度测量的高斯噪声标准差 (rad/s)（`vel_from_pos` 为真时忽略）|
        | `vel_from_pos` | 为真时**速度由位置测量做后向差分得到**（`(y_q,k − y_q,k−1)/h_c`），模拟「无速度传感器」的常见做法 |
        | `lpf_tau` | 一阶低通滤波时间常数 (s)；`None` = 不滤波 |
        | `seed` | 噪声随机种子（默认 0）|

    progress : 是否按 `log_frac` 打实时进度日志（2R 单步约 4.3 ms，长积分可达分钟级）

    返回 `dict`；键 `nan` 为 `True` 表示出现 NaN/Inf（记录被截断到出事前一步），
    `error` 非 `None` 表示控制/积分抛异常（如奇异位形下 `J_a` 不可解）——**都不吞掉，如实上报**。
    ⚠️ **噪声只加在控制器的输入上**（被控对象 `rk4_step` 始终用真值），这正是「状态来自传感器」的建模方式；
    记录里的 `q_meas`/`qdot_meas` 是**控制器实际看到的量**（零阶保持），供噪声影响分析用。
    """
    if h_c is None:
        h_c = dt
    n_steps = max(1, int(round(T / dt)))
    n_ctrl = max(1, int(round(h_c / dt)))
    N, m = case.N, case.m

    nz = dict(noise or {})
    sig_q = float(nz.get("q_sigma", 0.0) or 0.0)
    sig_qd = float(nz.get("qd_sigma", 0.0) or 0.0)
    vel_from_pos = bool(nz.get("vel_from_pos", False))
    lpf_tau = nz.get("lpf_tau", None)
    noise_on = bool(sig_q or sig_qd or vel_from_pos or lpf_tau)
    rng = np.random.default_rng(int(nz.get("seed", 0) or 0))
    prev_yq = None
    lpf_q = lpf_qd = None
    q_meas = np.zeros(N + 1)
    qd_meas = np.zeros(N + 1)

    t_arr = np.arange(n_steps + 1) * dt
    q_arr = np.zeros((n_steps + 1, N + 1))
    qd_arr = np.zeros((n_steps + 1, N + 1))
    u_arr = np.zeros((n_steps + 1, N + 1))
    qm_arr = np.zeros((n_steps + 1, N + 1))
    qdm_arr = np.zeros((n_steps + 1, N + 1))
    g_arr = np.zeros((n_steps + 1, N + 1))
    ueff_arr = np.zeros((n_steps + 1, N + 1))
    xe_arr = np.zeros((n_steps + 1, m))
    xt_arr = np.zeros((n_steps + 1, m))
    qdr_arr = np.zeros((n_steps + 1, N + 1))
    qt_arr = np.zeros((n_steps + 1, N + 1))
    smin_arr = np.full(n_steps + 1, np.nan)
    smax_arr = np.full(n_steps + 1, np.nan)
    cond_arr = np.full(n_steps + 1, np.nan)

    q, qdot = np.array(q0, dtype=float), np.array(qdot0, dtype=float)
    q_meas, qd_meas = q.copy(), qdot.copy()
    u = np.zeros(N + 1)
    u_abs_max = 0.0
    nan = False
    error = None
    n_rec = 0                       # 已完整记录的样本数（出现 NaN/异常时据此截断）
    log_every = max(1, int(round(n_steps * log_frac)))
    t_wall = time.perf_counter()

    def _record(i, r, u_now):
        q_arr[i], qd_arr[i], u_arr[i] = q, qdot, u_now
        qm_arr[i], qdm_arr[i] = q_meas, qd_meas
        x_e = case.fk(q)
        xe_arr[i] = x_e
        if r is not None and "x_d" in r:
            xt_arr[i] = np.asarray(r["x_d"], dtype=float) - x_e
        if r is not None and "q_d" in r:
            q_d = np.asarray(r["q_d"], dtype=float)
            qdr_arr[i] = q_d
            qt_arr[i] = q_d - q
        J_a = case.jac(q)
        sv = np.linalg.svd(J_a, compute_uv=False)     # 一次 SVD 同时给 σ_min 与 σ_max
        smin_arr[i] = float(sv.min())
        smax_arr[i] = float(sv.max())
        B, _C, g = case.dyn(q, qdot)   # 同一次 `robot_dyn` 调用同时给 cond(B)、g(q) 与 u−g
        cond_arr[i] = float(np.linalg.cond(B))
        g_arr[i, 1:] = g
        ueff_arr[i, 1:] = u_now[1:] - g

    try:
        for i in range(n_steps):
            ti = i * dt
            r = ref(ti) if callable(ref) else ref
            if i % n_ctrl == 0:
                y_q, y_qd = q, qdot                 # ⚠️ 每拍都要取**当前**状态（噪声关闭时即真值）
                if noise_on:
                    y_q, y_qd = q.copy(), qdot.copy()
                    if sig_q:
                        y_q[1:] = q[1:] + sig_q * rng.standard_normal(N)
                    if vel_from_pos:
                        # 速度由**位置测量**后向差分（首个控制步没有上一拍，只能用真值起步）
                        y_qd = qdot.copy()
                        y_qd[1:] = ((y_q[1:] - prev_yq[1:]) / h_c if prev_yq is not None
                                    else qdot[1:])
                    elif sig_qd:
                        y_qd[1:] = qdot[1:] + sig_qd * rng.standard_normal(N)
                    prev_yq = y_q
                    if lpf_tau:
                        alpha = h_c / (float(lpf_tau) + h_c)
                        if lpf_q is None:
                            lpf_q, lpf_qd = y_q.copy(), y_qd.copy()
                        lpf_q = lpf_q + alpha * (y_q - lpf_q)
                        lpf_qd = lpf_qd + alpha * (y_qd - lpf_qd)
                        y_q, y_qd = lpf_q, lpf_qd
                q_meas, qd_meas = y_q, y_qd
                u = ctrl((q_meas, qd_meas), r)
                u_abs_max = max(u_abs_max, float(np.max(np.abs(u[1:]))))
            _record(i, r, u)
            n_rec = i + 1

            def _f(qq, qdd, _ti=ti):
                te = tau_e(_ti, qq, qdd) if tau_e is not None else None
                return plant_qddot(qq, qdd, u, case, F_f=F_f, tau_e=te)

            q, qdot = rk4_step(_f, q, qdot, dt)
            if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qdot))):
                nan = True
                break
            if progress and ((i + 1) % log_every == 0 or i + 1 == n_steps):
                print("         [%s] %3d%%  t=%6.3f s  已用 %5.1f s  max‖u‖=%.3e  σ_min=%.3e"
                      % (label, int(round(100.0 * (i + 1) / n_steps)), (i + 1) * dt,
                         time.perf_counter() - t_wall, u_abs_max, smin_arr[i]), flush=True)
        else:
            r = ref(n_steps * dt) if callable(ref) else ref
            _record(n_steps, r, u)
            n_rec = n_steps + 1
    except Exception as exc:                                            # noqa: BLE001
        error = "%s: %s" % (type(exc).__name__, exc)
        nan = bool(n_rec > 0 and not np.all(np.isfinite(q_arr[:n_rec])))

    sl = slice(0, n_rec)
    return {
        "t": t_arr[sl], "q": q_arr[sl], "qdot": qd_arr[sl], "u": u_arr[sl],
        "q_meas": qm_arr[sl], "qdot_meas": qdm_arr[sl],
        # ⚠️ `u_eff = u − g(q)` 是**去掉重力补偿之后的"反馈"力矩**（阶段 4 新增）：
        # 奇异边界等实验里，总力矩 `u` 被 `g`（本用例 ‖g‖∞ ≈ 45 N·m）主导，
        # 会把 `J_a⁻¹` 的病态放大完全掩盖——必须看 `u_eff`（与 `g` 分开报告）。
        "g": g_arr[sl], "u_eff": ueff_arr[sl],
        "x_e": xe_arr[sl], "x_tilde": xt_arr[sl],
        "q_d": qdr_arr[sl], "q_tilde": qt_arr[sl],
        "sigma_min": smin_arr[sl], "sigma_max": smax_arr[sl], "cond_B": cond_arr[sl],
        "dt": dt, "h_c": h_c, "T": T, "n_steps": int(max(0, n_rec - 1)),
        "nan": bool(nan), "error": error, "noise": nz, "noise_on": noise_on,
        "u_abs_max": u_abs_max, "u_sat": bool(u_abs_max > case.u_max),
        "case": case.name, "label": label,
        "wall_time_s": time.perf_counter() - t_wall,
    }
