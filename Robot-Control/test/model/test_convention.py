# -*- coding: utf-8 -*-
"""阶段 0.2 + 0.3：`C(q,q̇)` 的精度与一致性 + 约定锁定（计划 §5.0.2、§5.0.3）。

**0.2 背景**：`model/code_dynamics.py` 的 `robot_dyn` 用**中心差分**近似 `∂B/∂q` 来构造
Christoffel 符号，于是 `C` 带 `O(h²)` 误差。后果与两个逆动力学控制算法直接冲突：
精确线性化不再精确，残差 `(C_true − C_数)q̇` 作为**有界扰动**进入闭环，稳态误差不真正为零；
闭环 Lyapunov 导数 `V̇ = −q̇ᵀ(F_f + J_aᵀK_DJ_a)q̇` 的等式也不严格成立。

本脚本为此提供一个**机器精度参照**：

1. 本文件内**独立实现** `B_ref(q)`（自己按 D-H 递推 + 质心雅可比列 `z_{i-1}×(p_c−p_{i-1})` 组装，
   与 `robot_B` 走的是不同代码路径），并验证 `B_ref ≡ robot_B`（相对误差应为机器精度）；
2. 用**复步微分**（`h=1e-20`，`Im(B)/h`）对 `B_ref` 求 `∂B/∂q` —— 这是被测实现
   （`robot_dyn` 的 `h=1e-6` 中心差分）之外的另一条路线；
   ⚠️ `model/` 的 `robot_B`/`robot_j_centroid` 预分配了实数组、**不能**直接吃复数，
   故参照必须在脚本内独立实现（这也正好保证了两条路线不同源）；
3. 用与 `robot_dyn` **同一个** Christoffel 版式从 `∂B_ref/∂q` 构造 `C_ref`，然后
   ① 测 `C_数(h)q̇` 的收敛阶，② 测 `N = Ḃ − 2C` 的反对称性偏差（`C` 不唯一，但 `N` 必反对称；
   有限差分 `C` 只能近似满足，该偏差本身就是它误差的一种度量），
   ③ 判断有无**结构性错误**（若 Christoffel 版式错，`N_ref` 也不会反对称）。

**0.3 约定锁定**：
- 重力符号：书 `g(q) = ∂U/∂q`，`U = −Σ_n m_n g₀ᵀp_n`。用复步微分对 `U(q)`（由正运动学直接算
  质心位置，**不经过雅可比**）求导，与 `robot_dyn` 的 `g` 对拍；
- **静平衡测试**：取 `τ = ∂U/∂q`（独立路线算出）、`F_f = 0`，仿 10 s，要求 `‖q̇‖_∞ < 1e-10`、
  `q` 无漂移（`< 1e-9` rad）。并做**扰动对照**（`τ = 0`、`τ = −g`、`τ = 0.99g`）证明它确实能
  检出重力符号/量级错误（对应计划 §5.1 的「故意失配」思路）；
- 分析雅可比 `J_a = diag{I₃, T_ϑ⁻¹}J_w`（6R）、`ẋ_e = J_a q̇`（`ẋ_e` 由正运动学数值微分给出）；
- 记录 `‖J_w q̇ − J_a q̇‖`：书中偶尔写 `ẋ_e = J_a q̇`，严格说 `[v; ω] = J_w q̇` 才成立，
  `J_a q̇` 的含义是「欧拉角导数」。本计划按书中一阶近似处理，偏差在阶段 3 量化——这里记录基线。

运行（`py311-gym`，任意 cwd）：

    python Robot-Control/test/model/test_convention.py
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RC = os.path.abspath(os.path.join(_HERE, "..", ".."))          # Robot-Control/
sys.path.insert(0, _RC)
sys.path.insert(0, os.path.join(_RC, "code"))

import code_models as cm                                          # noqa: E402
from model import (calc_T_n_to_last, eular_diff_to_w, robot_B,    # noqa: E402
                   robot_dyn, robot_fk, robot_jacobian_a,
                   robot_jacobian_w)

SEED = 0
# 判据阈值
TOL_G_REL = 1e-10        # 判据 4：重力项与 ∂U/∂q 相对误差
TOL_DRIFT_QDOT = 1e-10   # 判据 5：静平衡全程关节速度上限
TOL_DRIFT_Q = 1e-9       # 判据 5：静平衡全程位形漂移上限 (rad)
TOL_C_REL = 1e-8         # 0.2：C 的有限差分相对误差（h 最小处）
H_C = 1e-6               # 0.2：robot_dyn 默认的差分步长
T_SIM = 10.0             # 判据 5b：稳定平衡位形上的仿真时长 (s)
# ⚠️ 判据 5b 的积分步长：稳定平衡处 q̈ ≡ 0，`dt` 只影响开销、不影响结论（位形不变），
# 故取 5e-2（阶段 2 的仿真层才用 1e-3）。原因见实测：`robot_dyn` 单次 6R 需 ~17 ms
# （内部 7 次 robot_B），dt=1e-3 的 10 s 仿真要 ~7×10⁴ 次调用 ≈ 11 min。
DT_SIM = 5e-2
T_UNST = 2.0             # 判据 5d：不稳定平衡位形的诊断时长 (s，够拟合指数段即可)
DT_UNST = 1e-2
T_CTRL = 0.1             # 判据 5c：扰动对照的仿真时长 (s)
DT_CTRL = 2e-3
CS_H = 1e-20             # 复步微分的虚部步长
PROGRESS = True          # 实时进度日志（长积分时便于观察，输出到 stdout，flush）


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def rel_err(A, B):
    """相对误差 `‖A−B‖_F / ‖B‖_F`。"""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))


def fit_slope(hs, errs, factors=(10.0, 3.0, 1.0)):
    """对**截断主导段**做 log-log 最小二乘拟合，返回 `(slope, 用到的 h 列表, 用的倍数)`。

    中心差分的误差是 `O(h²) + 舍入平台`：`h` 小到一定程度后误差不再下降（本机实测在
    `1e-5` 附近见底，相对误差 ~1e-10）。若把舍入平台也算进拟合，斜率会被明显拉低
    （实测 2.000 → 1.0），故这里只取 `err > factor × min(err)` 的点；factor 从 10 逐级放松，
    以保证至少有两个点可拟合。
    """
    hs = np.asarray(hs, dtype=float)
    errs = np.asarray(errs, dtype=float)
    for f in factors:
        mask = errs > f * errs.min()
        if mask.sum() >= 2:
            slope = float(np.polyfit(np.log(hs[mask]), np.log(errs[mask]), 1)[0])
            return slope, hs[mask].tolist(), f
    slope = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
    return slope, hs.tolist(), 0.0


def unpack(prm):
    d, a, alpha, m, p_cents, I_inn, g0 = prm
    return d, a, alpha, m, p_cents, I_inn, g0, len(d) - 1


def _cross(u, v):
    """手写叉乘（避免 `np.cross` 的 dtype 细节，保证复数可用）。"""
    return np.array([u[1] * v[2] - u[2] * v[1],
                     u[2] * v[0] - u[0] * v[2],
                     u[0] * v[1] - u[1] * v[0]])


# ---------------------------------------------------------------------------
# 机器精度参照（本文件内独立实现；复数可用，以支持复步微分）
# ---------------------------------------------------------------------------
def com_link(q, d, a, alpha, p_cents, n):
    """连杆 n 质心在基座系的位置（D-H 递推直接算，**不经过雅可比**；复数可用）。"""
    T = np.eye(4, dtype=np.result_type(q.dtype, float))
    for i in range(1, n + 1):
        T = T @ calc_T_n_to_last(q[i], d[i], a[i], alpha[i])
    return T[:3, :3] @ p_cents[n] + T[:3, 3]


def U_pot(q, d, a, alpha, m, p_cents, g0, N):
    """势能 `U(q) = −Σ_n m_n g₀ᵀp_n(q)`（书中的定义）。"""
    return -sum(m[n] * (g0 @ com_link(q, d, a, alpha, p_cents, n)) for n in range(1, N + 1))


def B_ref(q, d, a, alpha, m, p_cents, I_inn, N):
    """`B(q)` 的**独立参照实现**（复数可用）。

    `B = Σ_n m_n J_{p,n}ᵀJ_{p,n} + J_{θ,n}ᵀR_nI_nR_nᵀJ_{θ,n}`，其中质心雅可比第 i 列
    （关节 i）取 `z_{i-1} × (p_{c,n} − p_{i-1})` 与 `z_{i-1}`，`i > n` 列为 0。
    与 `model/code_dynamics.py` 的 `robot_B`（走 `robot_j_centroid`）实现路径不同，用于交叉校验。
    """
    dt_ = np.result_type(q.dtype, float)
    T_ns = np.zeros((N + 1, 4, 4), dtype=dt_)
    T_ns[0] = np.eye(4, dtype=dt_)
    for n in range(1, N + 1):
        T_ns[n] = T_ns[n - 1] @ calc_T_n_to_last(q[n], d[n], a[n], alpha[n])
    B = np.zeros((N, N), dtype=dt_)
    for n in range(1, N + 1):
        R_n = T_ns[n][:3, :3]
        p_com = R_n @ p_cents[n] + T_ns[n][:3, 3]
        Jp = np.zeros((3, N), dtype=dt_)
        Jt = np.zeros((3, N), dtype=dt_)
        for i in range(1, n + 1):
            z_im1 = T_ns[i - 1][:3, 2]
            Jp[:, i - 1] = _cross(z_im1, p_com - T_ns[i - 1][:3, 3])
            Jt[:, i - 1] = z_im1
        B = B + m[n] * (Jp.T @ Jp) + Jt.T @ (R_n @ I_inn[n] @ R_n.T) @ Jt
    return B


def dU_dq_complex_step(q, d, a, alpha, m, p_cents, g0, N, h=CS_H):
    """`∂U/∂q` 的机器精度参照（复步微分）。"""
    g = np.zeros(N)
    for k in range(1, N + 1):
        qc = q.astype(complex)
        qc[k] += 1j * h
        g[k - 1] = np.imag(U_pot(qc, d, a, alpha, m, p_cents, g0, N)) / h
    return g


def dB_dq_complex_step(q, d, a, alpha, m, p_cents, I_inn, N, h=CS_H):
    """`∂B/∂q` 的机器精度参照：`dB[k-1] = ∂B/∂q_{k+1}`（复步微分 `B_ref`）。"""
    dB = np.zeros((N, N, N))
    for k in range(1, N + 1):
        qc = q.astype(complex)
        qc[k] += 1j * h
        dB[k - 1] = np.imag(B_ref(qc, d, a, alpha, m, p_cents, I_inn, N)) / h
    return dB


def dB_dq_central(q, d, a, alpha, m, p_cents, I_inn, N, h=1e-5):
    """`∂B/∂q` 的中心差分（仅用于**自检**复步参照）。"""
    dB = np.zeros((N, N, N))
    for k in range(1, N + 1):
        qp, qm = q.copy(), q.copy()
        qp[k] += h
        qm[k] -= h
        dB[k - 1] = (B_ref(qp, d, a, alpha, m, p_cents, I_inn, N)
                     - B_ref(qm, d, a, alpha, m, p_cents, I_inn, N)) / (2 * h)
    return dB


def christoffel_C(dBdq, qdot, N):
    """用 `∂B/∂q` 按 Christoffel 符号构造 `C`（**与 robot_dyn 完全相同**的版式）。"""
    C = np.zeros((N, N))
    for n in range(1, N + 1):
        for j in range(1, N + 1):
            for k in range(1, N + 1):
                c_njk = 0.5 * (dBdq[k - 1, n - 1, j - 1]
                               + dBdq[j - 1, n - 1, k - 1]
                               - dBdq[n - 1, j - 1, k - 1])
                C[n - 1, j - 1] += c_njk * qdot[k]
    return C


def B_dot(dBdq, qdot, N):
    """`Ḃ(q) = Σ_k ∂B/∂q_k · q̇_k`。"""
    Bdot = np.zeros((N, N))
    for k in range(1, N + 1):
        Bdot += qdot[k] * dBdq[k - 1]
    return Bdot


def skewness(M):
    """反对称性偏差 `‖M + Mᵀ‖_F / ‖M‖_F`。"""
    den = np.linalg.norm(M)
    if den == 0.0:
        return 0.0
    return float(np.linalg.norm(M + M.T) / den)


# ---------------------------------------------------------------------------
# 0.2 C 的精度与反对称性（判据 6）
# ---------------------------------------------------------------------------
def check_C(prm, tag, hs):
    d, a, alpha, m, p_cents, I_inn, g0, N = unpack(prm)
    q = cm.Q_NOM_2R if tag == "2R" else cm.Q_NOM_6R
    qdot = cm.QDOT_NOM_2R if tag == "2R" else cm.QDOT_NOM_6R

    # 参照自检 ①：B_ref 与 robot_B 是否同一个 B
    B_impl = B_ref(q, d, a, alpha, m, p_cents, I_inn, N)
    B_model = robot_B(q, d, a, alpha, m, p_cents, I_inn, N)
    ref_B_self = rel_err(B_impl, B_model)
    # 参照自检 ②：复步 ∂B/∂q vs 中心差分(h=1e-5)
    dB_ref = dB_dq_complex_step(q, d, a, alpha, m, p_cents, I_inn, N)
    dB_cen = dB_dq_central(q, d, a, alpha, m, p_cents, I_inn, N, h=1e-5)
    ref_self = rel_err(dB_ref, dB_cen)

    C_ref = christoffel_C(dB_ref, qdot, N)
    Bdot_ref = B_dot(dB_ref, qdot, N)
    M_ref = Bdot_ref - 2.0 * C_ref

    print("  [%s] 参照自检 ①：B_ref（独立实现）vs robot_B 相对误差 = %.3e" % (tag, ref_B_self))
    print("       参照自检 ②：复步 ∂B/∂q vs 中心差分(h=1e-5) 相对误差 = %.3e" % ref_self)
    print("       N_ref = Ḃ−2C 的反对称性偏差 ‖N+Nᵀ‖/‖N‖ = %.3e（应为机器精度量级）"
          % skewness(M_ref))

    rows = []
    print("       C_数(h)q̇ vs 机器精度参照 C_ref q̇：")
    for h in hs:
        _, C_h, _ = robot_dyn(q, qdot, d, a, alpha, m, p_cents, I_inn, g0, N, h=h)
        dc = C_h @ qdot[1:] - C_ref @ qdot[1:]
        e = rel_err(C_h @ qdot[1:], C_ref @ qdot[1:])
        rows.append((h, e, float(np.abs(dc).max())))
        print("         h = %.0e  相对误差 = %.3e   max|Δ(Cq̇)| = %.3e" % (h, e, np.abs(dc).max()))
    slope, used_h, fac = fit_slope(list(hs), [e for _, e, _ in rows])
    e_min = min(e for _, e, _ in rows)
    h_min = hs[int(np.argmin([e for _, e, _ in rows]))]
    print("       拟合收敛阶 = %.3f（理论 2.000；用 err > %.0f×min 的 %d 个点 %s）"
          % (slope, fac, len(used_h), ["%.0e" % v for v in used_h]))
    print("       舍入平台：最小相对误差 %.3e（h = %.0e），故 h 再小已无意义" % (e_min, h_min))

    print("       N = Ḃ_ref − 2C_数 的反对称性偏差随 h（偏差 ≈ 2(C_ref−C_数)，即差分误差的度量）：")
    skew_rows = []
    for h in hs:
        _, C_h, _ = robot_dyn(q, qdot, d, a, alpha, m, p_cents, I_inn, g0, N, h=h)
        sk = skewness(Bdot_ref - 2.0 * C_h)
        skew_rows.append((h, sk))
        print("         h = %.0e  ‖N+Nᵀ‖/‖N‖ = %.3e" % (h, sk))
    skew_slope, used_h_skew, fac_skew = fit_slope(list(hs), [s for _, s in skew_rows])
    print("       反对称性偏差的收敛阶 = %.3f（理论 2.000；用 %d 个点）"
          % (skew_slope, len(used_h_skew)))

    _, C_def, _ = robot_dyn(q, qdot, d, a, alpha, m, p_cents, I_inn, g0, N, h=H_C)
    residual = C_def @ qdot[1:] - C_ref @ qdot[1:]
    measured = {
        "reference": "本文件独立实现的 B_ref + 复步微分 ∂B/∂q（h=1e-20）+ 同版式 Christoffel C",
        "ref_self_check_B_rel_err": ref_B_self,
        "ref_self_check_dB_rel_err": ref_self,
        "N_ref_skewness": skewness(M_ref),
        "C_rel_err_by_h": {("%.0e" % h): float(e) for h, e, _ in rows},
        "C_abs_err_by_h": {("%.0e" % h): float(a_) for h, _, a_ in rows},
        "C_min_rel_err": float(min(e for _, e, _ in rows)),
        "C_h_at_min": float(h_min),
        "C_slope": slope,
        "C_slope_points": [float(v) for v in used_h],
        "N_model_skewness_by_h": {("%.0e" % h): float(s) for h, s in skew_rows},
        "N_model_skewness_min": float(min(s for _, s in skew_rows)),
        "N_model_skewness_slope": skew_slope,
        "C_model_h_default": H_C,
        "residual_Cqdot_abs_at_default_h": float(np.abs(residual).max()),
        "residual_Cqdot_rel_at_default_h": float(np.linalg.norm(residual) / np.linalg.norm(C_ref @ qdot[1:])),
    }
    # 判据 6：参照反对称到机器精度（无结构性错误）、偏差随 h 按 O(h²) 下降、C 误差足够小
    passed = (ref_B_self < 1e-12 and ref_self < 1e-8 and skewness(M_ref) < 1e-12
              and measured["C_min_rel_err"] < TOL_C_REL and 1.8 <= slope <= 2.2
              and 1.8 <= skew_slope <= 2.2)
    return passed, measured


# ---------------------------------------------------------------------------
# 0.3 重力项与 ∂U/∂q（判据 4）
# ---------------------------------------------------------------------------
def check_gravity(prm, tag):
    d, a, alpha, m, p_cents, I_inn, g0, N = unpack(prm)
    q = cm.Q_NOM_2R if tag == "2R" else cm.Q_NOM_6R
    qdot = np.zeros(N + 1)

    _, _, g_model = robot_dyn(q, qdot, d, a, alpha, m, p_cents, I_inn, g0, N)
    g_ref = dU_dq_complex_step(q, d, a, alpha, m, p_cents, g0, N)
    h = 1e-6
    g_cen = np.zeros(N)
    for k in range(1, N + 1):
        qp, qm = q.copy(), q.copy()
        qp[k] += h
        qm[k] -= h
        g_cen[k - 1] = (U_pot(qp, d, a, alpha, m, p_cents, g0, N)
                        - U_pot(qm, d, a, alpha, m, p_cents, g0, N)) / (2 * h)

    e_rel = rel_err(g_model, g_ref)
    print("  [%s] g_模型 vs ∂U/∂q（复步参照）相对误差 = %.3e，max|Δ| = %.3e"
          % (tag, e_rel, np.abs(g_model - g_ref).max()))
    print("       参照自检：复步 vs 中心差分(h=1e-6) 相对误差 = %.3e" % rel_err(g_ref, g_cen))
    print("       g_模型 = %s" % np.array2string(g_model, precision=5, suppress_small=True))
    print("       ‖g‖_∞ = %.4f（符号约定 g = +∂U/∂q，见 code_dynamics.py 的 docstring）"
          % np.abs(g_model).max())

    measured = {
        "rel_err": e_rel,
        "max_abs_diff": float(np.abs(g_model - g_ref).max()),
        "g_inf_norm": float(np.abs(g_model).max()),
        "g_model": [float(v) for v in g_model],
        "g_ref_complex_step": [float(v) for v in g_ref],
        "reference_self_check_rel_err": rel_err(g_ref, g_cen),
    }
    return e_rel < TOL_G_REL, measured


# ---------------------------------------------------------------------------
# 0.3 静平衡测试（判据 5，**含判据本身的一次修订**）
# ---------------------------------------------------------------------------
# ⚠️ 计划 §5.0.5 判据 5 的原文是「取 τ = g(q)、F_f = 0，仿 10 s，‖q̇‖_∞ < 1e-10，q 无漂移」。
# 实测发现该判据**在任意位形上并不成立**：只有**稳定**平衡位形（∂²U/∂q² ⪰ 0）才谈得上「不漂移」；
# 在标称位形（臂抬起的不稳定平衡）上，1e-14 的机器 eps 会被指数放大，10 s 后 max‖q̇‖ ~ 1e40。
# 且实测增长率与线性化预测 sqrt(|λ_min(B⁻¹H)|) 一致，说明是**物理不稳定**、不是模型/实现错误。
# 故判据 5 修订为四项（本脚本逐项实测）：
#   5a 瞬时平衡残差：任意位形上 τ = ∂U/∂q 时 ‖q̈‖_∞ < 1e-12（这才真正检验 g 的符号/量级）
#   5b 10 s 静平衡：在**稳定**平衡位形上 τ = g(q)，要求全程无漂移（判据 5 的原文诉求）
#   5c 灵敏度对照：τ = 0 / −g / 0.99g 时 ‖q̈₀‖_∞ 必须 ≫ 1e-12（证明测试有鉴别力）
#   5d 稳定性诊断：不稳定平衡位形上的指数增长率须与线性化预测一致（解释 5b 为何需要稳定位形）
def gravity_hessian(q, d, a, alpha, m, p_cents, g0, N, h=1e-5):
    """`∂²U/∂q²`（对复步 ∂U/∂q 再做中心差分；只用于稳定性判定，不追求机器精度）。"""
    H = np.zeros((N, N))
    for k in range(1, N + 1):
        qp, qm = q.copy(), q.copy()
        qp[k] += h
        qm[k] -= h
        H[:, k - 1] = (dU_dq_complex_step(qp, d, a, alpha, m, p_cents, g0, N)
                       - dU_dq_complex_step(qm, d, a, alpha, m, p_cents, g0, N)) / (2 * h)
    return H


def linearized_modes(prm, q):
    """线性化模态：返回 `(λ 排序, 最不稳定模态增长率, 对应 q 方向)`。

    `δq̈ = −B⁻¹(∂²U/∂q²)δq`，故 `λ < 0` 的方向按 `e^{sqrt(|λ|)t}` 指数增长。
    """
    d, a, alpha, m, p_cents, I_inn, g0, N = unpack(prm)
    B = robot_B(q, d, a, alpha, m, p_cents, I_inn, N)
    H = gravity_hessian(q, d, a, alpha, m, p_cents, g0, N)
    lam, vec = np.linalg.eig(np.linalg.solve(B, H))
    lam = lam.real
    order = np.argsort(lam)
    i_min = int(order[0])
    rate = float(np.sqrt(-lam[i_min])) if lam[i_min] < 0 else 0.0
    return lam[order], rate, np.asarray(vec[:, i_min].real)


def polish_equilibrium(q_start, prm, iters=5):
    """把 `q_start` 用阻尼 Newton 抛光到 `∂U/∂q = 0` 的机器精度。

    为什么需要：`code/code_models.py` 里的 `Q_EQ_*` 只有 10 位有效数字，直接用它当平衡点会
    留下 `‖∂U/∂q‖ ~ 4e-10` 的残差（`H ~ 45` → `Δq ~ 1e-11`），10 s 后就是可见漂移。
    抛光后 `‖∂U/∂q‖ ~ 1e-15`，静平衡测试才名副其实。用 `lstsq(rcond=1e-6)` 排除
    中性方向（绕基座 z 轴，`λ = 0`）以免过冲。
    """
    d, a, alpha, m, p_cents, I_inn, g0, N = unpack(prm)
    q = np.asarray(q_start, dtype=float).copy()
    for _ in range(iters):
        g = dU_dq_complex_step(q, d, a, alpha, m, p_cents, g0, N)
        H = gravity_hessian(q, d, a, alpha, m, p_cents, g0, N)
        step = np.linalg.lstsq(H, g, rcond=1e-6)[0]
        q[1:] -= np.clip(step, -0.2, 0.2)
    return q


def check_static_equilibrium(prm, tag):
    d, a, alpha, m, p_cents, I_inn, g0, N = unpack(prm)
    q_nom = cm.Q_NOM_2R if tag == "2R" else cm.Q_NOM_6R
    q_eq0 = cm.Q_EQ_2R if tag == "2R" else cm.Q_EQ_6R
    q_eq = polish_equilibrium(q_eq0, prm)
    qdot0 = np.zeros(N + 1)
    out = {}

    def g_indep_at(qq):
        g = np.zeros(N + 1)
        g[1:] = dU_dq_complex_step(qq, d, a, alpha, m, p_cents, g0, N)
        return g

    # ---- 5a 瞬时平衡残差（任意位形，g ≠ 0）----
    g_nom = g_indep_at(q_nom)
    qddot0 = plant_qddot(q_nom, qdot0, prm, g_nom)
    print("  [%s] 5a 瞬时平衡残差（标称位形，τ = ∂U/∂q，F_f = 0）：‖q̈‖_∞ = %.3e rad/s²"
          % (tag, np.abs(qddot0).max()))
    print("       该位形 ‖g‖_∞ = %.4f（非零，故此项能检出 g 的符号/量级错误）" % np.abs(g_nom[1:]).max())
    ok_a = float(np.abs(qddot0).max()) < 1e-12

    # ---- 5c 灵敏度对照（同一标称位形）----
    print("       5c 灵敏度对照（同一标称位形，取不同 τ）：")
    controls = {}
    for name, tau_of in (
        ("τ = 0（不补偿重力）", lambda qq: np.zeros(N + 1)),
        ("τ = −g（重力符号错）", lambda qq: -g_nom),
        ("τ = 0.99·g（量级差 1%）", lambda qq: 0.99 * g_nom),
    ):
        tau_c = tau_of(q_nom)
        qddot_c = plant_qddot(q_nom, qdot0, prm, tau_c)
        _, _, mqd, mdq, _ = simulate(prm, lambda qq, qd, tt: tau_c, q_nom, qdot0,
                                     T_CTRL, DT_CTRL, label="%s 对照" % tag, progress=False)
        controls[name] = {"qddot0_inf": float(np.abs(qddot_c).max()),
                          "max_qdot_inf": mqd, "max_dq_inf": mdq, "T_ctrl": T_CTRL}
        print("         %-24s ‖q̈₀‖_∞ = %.3e rad/s²；%.2f s 后 max‖q−q₀‖_∞ = %.3e rad"
              % (name, np.abs(qddot_c).max(), T_CTRL, mdq))
    ok_c = all(c["qddot0_inf"] > 1e-9 for c in controls.values())

    # ---- 5d 稳定性诊断（标称位形是不稳定平衡）----
    lam_nom, rate_pred, v_unst = linearized_modes(prm, q_nom)
    eps_seed = 1e-9
    q_seed = q_nom.copy()
    q_seed[1:] += eps_seed * v_unst
    samples = []
    n_steps, dt = int(round(T_UNST / DT_UNST)), DT_UNST
    q_s, qd_s = q_seed.copy(), qdot0.copy()
    t_wall = time.perf_counter()
    for i in range(n_steps):
        q_s, qd_s = rk4_step(lambda qq, qd: plant_qddot(qq, qd, prm, g_nom), q_s, qd_s, dt)
        samples.append(((i + 1) * dt, float(np.max(np.abs(qd_s)))))
    # 取仍处线性指数段的样本（1e-8 ~ 1e-2）拟合斜率
    fit = [(t, v) for t, v in samples if 1e-8 <= v <= 1e-2]
    rate_meas = float("nan")
    if len(fit) >= 3:
        ts = np.array([t for t, _ in fit])
        vs = np.log(np.array([v for _, v in fit]))
        rate_meas = float(np.polyfit(ts, vs, 1)[0])
    rate_err = abs(rate_meas - rate_pred) / rate_pred if rate_pred > 0 else float("inf")
    print("       5d 不稳定平衡诊断：λ_min(B⁻¹∂²U/∂q²) = %.3f → 预测增长率 %.4f /s；"
          % (lam_nom[0], rate_pred))
    print("          实测（种子 %.0e·v_unst，%.1f s，%d 步，耗时 %.1f s）增长率 = %.4f /s，"
          "相对偏差 %.2f%%" % (eps_seed, T_UNST, n_steps, time.perf_counter() - t_wall,
                              rate_meas, 100 * rate_err))
    ok_d = rate_err < 0.05

    # ---- 5b 稳定平衡位形上的 10 s 静平衡 ----
    g_eq = g_indep_at(q_eq)
    lam_eq, rate_eq, _ = linearized_modes(prm, q_eq)
    qddot_eq = plant_qddot(q_eq, qdot0, prm, g_eq)
    premise = (float(np.abs(g_eq[1:]).max()) < 1e-12 and float(lam_eq[0]) > -1e-9)
    print("       5b 稳定平衡位形（Q_EQ_%s 抛光后）：‖∂U/∂q‖_∞ = %.3e，"
          "λ(B⁻¹∂²U/∂q²) 最小 = %.3e（≥0 即稳定）→ %s"
          % (tag, np.abs(g_eq[1:]).max(), lam_eq[0], "成立" if premise else "**不成立**"))
    print("          q_eq = %s" % np.array2string(q_eq, precision=12, separator=", "))
    _, _, max_qd, max_dq, nan = simulate(prm, lambda qq, qd, tt: g_eq, q_eq, qdot0,
                                         T_SIM, DT_SIM, label="%s Q_EQ" % tag)
    print("          τ = ∂U/∂q(q_eq)、F_f = 0，仿 %.0f s（dt = %.0e，%d 步）："
          % (T_SIM, DT_SIM, int(round(T_SIM / DT_SIM))))
    print("          平衡点瞬时 ‖q̈‖_∞ = %.3e；全程 max‖q̇‖_∞ = %.3e，max‖q−q_eq‖_∞ = %.3e rad"
          % (np.abs(qddot_eq).max(), max_qd, max_dq))
    ok_b = premise and (max_qd < TOL_DRIFT_QDOT) and (max_dq < TOL_DRIFT_Q) and not nan

    out = {
        "T_sim": T_SIM, "dt_eq": DT_SIM, "T_unst": T_UNST, "dt_unst": DT_UNST,
        "T_ctrl": T_CTRL, "dt_ctrl": DT_CTRL, "F_f": "0",
        "q_nom": [float(v) for v in q_nom],
        "q_eq_start": [float(v) for v in q_eq0],
        "q_eq_polished": [float(v) for v in q_eq],
        "a_instantaneous_qddot_inf_at_q_nom": float(np.abs(qddot0).max()),
        "a_passed": bool(ok_a),
        "b_premise_g_inf": float(np.abs(g_eq[1:]).max()),
        "b_premise_lambda_min": float(lam_eq[0]),
        "b_qddot_inf_at_q_eq": float(np.abs(qddot_eq).max()),
        "b_max_qdot_inf": max_qd, "b_max_dq_inf": max_dq, "b_nan": bool(nan),
        "b_passed": bool(ok_b),
        "c_controls": controls, "c_passed": bool(ok_c),
        "d_lambda_min_at_q_nom": float(lam_nom[0]),
        "d_rate_predicted": rate_pred, "d_rate_measured": rate_meas,
        "d_rate_rel_err": float(rate_err), "d_passed": bool(ok_d),
    }
    passed = ok_a and ok_b and ok_c and ok_d
    return passed, out


def plant_qddot(q, qdot, prm, tau):
    """被控对象状态方程：`q̈ = B⁻¹(τ − Cq̇ − F_f q̇ − g)`（基线 `F_f = 0`）。

    `q, qdot, tau` 均为长度 `N+1`（`[0]` 不用），返回长度 `N+1` 的 `q̈`。
    """
    d, a, alpha, m, p_cents, I_inn, g0, N = unpack(prm)
    B, C, g = robot_dyn(q, qdot, d, a, alpha, m, p_cents, I_inn, g0, N)
    qddot = np.zeros(N + 1)
    qddot[1:] = np.linalg.solve(B, tau[1:] - C @ qdot[1:] - g)   # F_f 基线 = 0（书上方程含该项）
    return qddot


def rk4_step(f, q, qdot, dt):
    """RK4 一步（状态 `y = [q; q̇]`，`f(q, q̇) -> q̈`，均为长度 N+1）。

    ⚠️ 与 `code/code_sim.py` 的同名函数**必须逐字一致**——两处都曾在 2026-09-20 之前
    把 `q_new` 的权重写成 `(a1 + 2a2 + a3)/6`（多算 `a2`），使积分器**退化为一阶**。
    正确组合由增广系统 `ẏ = [q̇; a]` 上的标准 RK4 推出：`(a1 + a2 + a3)/6`。
    守卫见下方 `check_integrator_order`。
    """
    a1 = f(q, qdot)
    a2 = f(q + 0.5 * dt * qdot, qdot + 0.5 * dt * a1)
    a3 = f(q + 0.5 * dt * qdot + 0.25 * dt * dt * a1, qdot + 0.5 * dt * a2)
    a4 = f(q + dt * qdot + 0.5 * dt * dt * a2, qdot + dt * a3)
    q_new = q + dt * qdot + dt * dt / 6.0 * (a1 + a2 + a3)
    qdot_new = qdot + dt / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    return q_new, qdot_new


def simulate(prm, tau_fun, q0, qdot0, T, dt=DT_SIM, label="", progress=PROGRESS):
    """定步长 RK4 仿真，返回 `(q, q̇, max‖q̇‖_∞, max‖q−q₀‖_∞, 是否 NaN)`。

    `progress=True` 时按 10% 打实时日志（含已用时间与当前 `max‖q̇‖`），
    因为 `robot_dyn` 单次 6R 约 17 ms、长积分可达分钟级，不打印会像「卡住」。
    """
    q, qdot = q0.copy(), qdot0.copy()
    n_steps = max(1, int(round(T / dt)))
    max_qd, max_dq = 0.0, 0.0
    log_every = max(1, n_steps // 10)
    t_wall = time.perf_counter()
    for i in range(n_steps):
        tau = tau_fun(q, qdot, i * dt)
        q, qdot = rk4_step(lambda qq, qd: plant_qddot(qq, qd, prm, tau), q, qdot, dt)
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qdot))):
            return q, qdot, float("inf"), float("inf"), True
        max_qd = max(max_qd, float(np.max(np.abs(qdot))))
        max_dq = max(max_dq, float(np.max(np.abs(q - q0))))
        if progress and ((i + 1) % log_every == 0 or i + 1 == n_steps):
            print("         [%s] %3d%%  t=%6.2f s  已用 %5.1f s  max‖q̇‖=%.2e  max‖Δq‖=%.2e"
                  % (label, int(round(100.0 * (i + 1) / n_steps)), (i + 1) * dt,
                     time.perf_counter() - t_wall, max_qd, max_dq), flush=True)
    return q, qdot, max_qd, max_dq, False


# ---------------------------------------------------------------------------
# 约定锁定（0.3；记录基线，不单独设判据）
# ---------------------------------------------------------------------------
def check_conventions(prm, tag):
    d, a, alpha, m, p_cents, I_inn, g0, N = unpack(prm)
    q = cm.Q_NOM_2R if tag == "2R" else cm.Q_NOM_6R
    qdot = cm.QDOT_NOM_2R if tag == "2R" else cm.QDOT_NOM_6R

    B = robot_B(q, d, a, alpha, m, p_cents, I_inn, N)
    sym = float(np.abs(B - B.T).max() / np.abs(B).max())
    eig = np.linalg.eigvalsh(B)
    out = {
        "B_symmetry_rel": sym,
        "B_pd": bool(np.all(eig > 0)),
        "B_cond": float(np.linalg.cond(B)),
        "B_eig_min": float(eig.min()), "B_eig_max": float(eig.max()),
    }
    print("  [%s] B 对称性 max|B−Bᵀ|/max|B| = %.3e；正定：%s；cond(B) = %.3e；eig(B) ∈ [%.3e, %.3e]"
          % (tag, sym, out["B_pd"], out["B_cond"], out["B_eig_min"], out["B_eig_max"]))

    tau = 1e-6
    if tag == "6R":
        x_e = robot_fk(q, d, a, alpha, N)
        J_w = robot_jacobian_w(q, d, a, alpha, N)
        J_a = robot_jacobian_a(q, d, a, alpha, x_e[3:], N)
        M = np.zeros((6, 6))
        M[:3, :3] = np.eye(3)
        M[3:, 3:] = np.linalg.inv(eular_diff_to_w(x_e[3:]))
        e_ja = rel_err(J_a, M @ J_w)
        print("       J_a = diag{I₃, T_ϑ⁻¹}J_w 相对误差 = %.3e" % e_ja)
        xdot_fd = (robot_fk(q + tau * qdot, d, a, alpha, N)
                   - robot_fk(q - tau * qdot, d, a, alpha, N)) / (2 * tau)
        e_xdot = rel_err(xdot_fd, J_a @ qdot[1:])
        print("       ẋ_e（robot_fk 中心差分, τ=1e-6）vs J_a q̇ 相对误差 = %.3e" % e_xdot)
        gap = (np.linalg.norm(J_w @ qdot[1:] - J_a @ qdot[1:])
               / np.linalg.norm(J_w @ qdot[1:]))
        print("       ⚠️ 约定缺口 ‖J_w q̇ − J_a q̇‖/‖J_w q̇‖ = %.3e（书中偶写 ẋ_e = J_a q̇；"
              "严格说 [v; ω] = J_w q̇，J_a q̇ 是欧拉角导数；偏差留待阶段 3 量化）" % gap)
        out.update({"J_a_decomposition_rel_err": e_ja, "xdot_vs_Ja_qdot_rel_err": e_xdot,
                    "Jw_vs_Ja_gap_rel": float(gap),
                    "Jw_qdot": [float(v) for v in J_w @ qdot[1:]],
                    "Ja_qdot": [float(v) for v in J_a @ qdot[1:]]})
    else:
        J_a = cm.robot_jacobian_a_2r(q)
        xdot_fd = (cm.robot_fk_2r(q + tau * qdot) - cm.robot_fk_2r(q - tau * qdot)) / (2 * tau)
        e_xdot = rel_err(xdot_fd, J_a @ qdot[1:])
        print("       ẋ_e（2R 闭式正运动学中心差分, τ=1e-6）vs J_a q̇ 相对误差 = %.3e" % e_xdot)
        out.update({"xdot_vs_Ja_qdot_rel_err": e_xdot})
    return out


# ---------------------------------------------------------------------------
# 积分器自检（0.3 补充；2026-09-20 新增——为一个**真 bug** 立的守卫）
# ---------------------------------------------------------------------------
def check_integrator_order(hs=(1e-2, 5e-3, 2.5e-3, 1.25e-3), wn=2.0, T=1.0):
    """积分器收敛阶自检：`rk4_step` 对 `q̈ = −ω²q` 必须给出**四阶**收敛。

    ⚠️ **为什么需要它**（2026-09-20，见书仓库日志「尝试 12」）：`rk4_step` 的 `q_new`
    权重原先写成 `(a1 + 2a2 + a3)/6`（多算 `a2`），使积分器**全局只有一阶**——
    这直接污染了阶段 2/3 的全部数值（阶段 3 判据 1 那条被归因为「离散化 `O(dt)`」的
    `q̃_6` 偏差其实主要来自这个实现错误），也让阶段 4 的失稳阈值
    `hω* ≈ 0.85` 与 RK4 理论 `2.785` 差了 3.3×。
    本检查把「积分器真的是四阶」变成一条**可判定的守卫**：经验阶必须 ∈ [3.5, 4.5]。
    （`code/code_sim.py` 与 `test_convention.py` 各有一份同名实现，本检查只覆盖后者；
    前者由阶段 2/3/4 的判据覆盖。）
    """
    def f(q, _qd, _w=wn):
        return -_w * _w * q

    errs, used = [], []
    for h in hs:
        n = max(1, int(round(T / h)))
        q, qd = 1.0, 0.0
        for _ in range(n):
            q, qd = rk4_step(f, q, qd, h)
        errs.append(abs(q - np.cos(wn * T)) + abs(qd + wn * np.sin(wn * T)))
        used.append(h)
    order = fit_slope(used, errs, factors=(2.0,))[0]
    ok = bool(np.isfinite(order) and 3.5 <= order <= 4.5)
    print("  积分器收敛阶（q̈=−ω²q，ω=%.1f，T=%.1f s）：" % (wn, T))
    for h, e in zip(used, errs):
        print("     h=%-9.3g  err=%.4e" % (h, e))
    print("     经验阶 = %.3f（理论 4）→ %s" % (order, "OK" if ok else "!! 积分器不是四阶！"))
    return ok, {"order": float(order), "hs": list(used), "errs": [float(v) for v in errs],
                "wn": wn, "T": T}


# ---------------------------------------------------------------------------
# 性能基线（不是判据；阶段 2 的仿真步长预算与阶段 4 的周期敏感性要用）
# ---------------------------------------------------------------------------
def measure_timing(cases, reps_B=60, reps_dyn=20):
    """实测 `robot_B` / `robot_dyn` 单次求值耗时（ms）。"""
    out = {}
    print("  单次求值耗时（ms/call，`robot_dyn` 内部要做 1+6 次 `robot_B` 来差分 ∂B/∂q）：")
    for tag, prm in cases.items():
        d, a, alpha, m, p_cents, I_inn, g0, N = unpack(prm)
        q = cm.Q_NOM_2R if tag == "2R" else cm.Q_NOM_6R
        qdot = np.zeros(N + 1)
        robot_B(q, d, a, alpha, m, p_cents, I_inn, N)
        t0 = time.perf_counter()
        for _ in range(reps_B):
            robot_B(q, d, a, alpha, m, p_cents, I_inn, N)
        t_B = (time.perf_counter() - t0) / reps_B * 1e3
        robot_dyn(q, qdot, d, a, alpha, m, p_cents, I_inn, g0, N)
        t0 = time.perf_counter()
        for _ in range(reps_dyn):
            robot_dyn(q, qdot, d, a, alpha, m, p_cents, I_inn, g0, N)
        t_dyn = (time.perf_counter() - t0) / reps_dyn * 1e3
        out[tag] = {"robot_B_ms": t_B, "robot_dyn_ms": t_dyn}
        print("     [%s] robot_B = %.3f ms   robot_dyn = %.3f ms" % (tag, t_B, t_dyn))
    return out


# ---------------------------------------------------------------------------
def run(hs=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)):
    print("=" * 72)
    print("阶段 0.2 + 0.3  C 的精度与一致性、约定锁定")
    print("  参照：复步微分（h=1e-20）∂B/∂q（独立 B 实现）、∂U/∂q")
    print("  静平衡：稳定平衡位形上 RK4 dt=%.0e、仿 %.0f s（不动点，dt 只影响开销）" % (DT_SIM, T_SIM))
    print("  不稳定平衡诊断：dt=%.0e、%.1f s（拟合机器 eps 被放大的指数增长率）" % (DT_UNST, T_UNST))
    print("=" * 72)

    cases = {"2R": cm.two_link_2r(), "6R": cm.puma_6r()}

    print("\n-- 0.2 C 的精度与反对称性（判据 6）--")
    c_res = {}
    for tag, prm in cases.items():
        c_res[tag] = check_C(prm, tag, hs)
        print()
    ok_c = all(v[0] for v in c_res.values())

    print("-- 0.3 重力项 g(q) = ∂U/∂q（判据 4）--")
    g_res = {}
    for tag, prm in cases.items():
        g_res[tag] = check_gravity(prm, tag)
        print()
    ok_g = all(v[0] for v in g_res.values())

    print("-- 0.3 静平衡测试（判据 5）--")
    s_res = {}
    for tag, prm in cases.items():
        s_res[tag] = check_static_equilibrium(prm, tag)
        print()
    ok_s = all(v[0] for v in s_res.values())

    print("-- 约定锁定（0.3；记录基线，不单独设判据）--")
    conv = {}
    for tag, prm in cases.items():
        conv[tag] = check_conventions(prm, tag)
        print()

    print("-- 积分器收敛阶自检（0.3 补充；判据 7，2026-09-20 为一个真 bug 立）--")
    ok_int, int_res = check_integrator_order()
    print()

    print("-- 性能基线（不是判据；供阶段 2 步长预算 / 阶段 4 周期敏感性）--")
    timing = measure_timing(cases)
    print()

    checks = {
        "integrator_order": {
            "desc": "积分器自检：`rk4_step` 对 q̈=−ω²q 的**全局收敛阶**必须是四阶"
                    "（2026-09-20 新增——此前 q_new 的 RK4 权重多算 a2，实际只有一阶，"
                    "污染了阶段 2/3/4 的全部数值）",
            "threshold": "经验收敛阶 ∈ [3.5, 4.5]（理论 4）",
            "measured": int_res,
            "passed": bool(ok_int),
        },
        "gravity_du": {
            "desc": "重力项与 ∂U/∂q 一致（U 由正运动学直接算质心，复步微分参照）",
            "threshold": "rel_err < 1e-10",
            "measured": {tag: m for tag, (_, m) in g_res.items()},
            "passed": bool(ok_g),
        },
        "static_equilibrium": {
            "desc": "静平衡（**判据 5 已按实测修订为 5a–5d**）：5a 任意位形上 τ=∂U/∂q 的瞬时 ‖q̈‖_∞；"
                    "5b 稳定平衡位形上仿 10 s 无漂移；5c 扰动对照可检出重力错误；"
                    "5d 不稳定平衡位形的指数增长率与线性化预测一致",
            "threshold": "5a ‖q̈‖_∞<1e-12；5b premise ‖∂U/∂q‖_∞<1e-12 且 λ_min≥-1e-9，"
                         "且 ‖q̇‖_∞<1e-10、‖q−q_eq‖_∞<1e-9 rad；5c 三个对照 ‖q̈₀‖_∞>1e-9；"
                         "5d 实测增长率与 sqrt(|λ_min|) 相对偏差 <5%",
            "measured": {tag: m for tag, (_, m) in s_res.items()},
            "passed": bool(ok_s),
        },
        "c_antisymmetry": {
            "desc": "C 的反对称性：N = Ḃ − 2C 的偏差与差分阶一致（O(h²)）且无结构性错误；"
                    "同时实测 C 的有限差分误差（供阶段 3 判断残差量级）",
            "threshold": "参照 N_ref 反对称 <1e-12；C rel_err(h_min) < 1e-8；两条斜率∈[1.8, 2.2]（理论 2）",
            "measured": {tag: m for tag, (_, m) in c_res.items()},
            "passed": bool(ok_c),
        },
    }
    passed = ok_c and ok_g and ok_s and ok_int
    print("-" * 72)
    print("阶段 0.2/0.3 结论：%s" % ("PASS" if passed else "FAIL"))
    for k, v in checks.items():
        print("   [%s] %s" % ("OK" if v["passed"] else "!!", k))
    return {"name": "阶段 0.2 + 0.3 C 精度与约定锁定", "passed": bool(passed),
            "checks": checks, "conventions": conv, "timing_ms": timing, "hs": list(hs)}


def main():
    res = run()
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
