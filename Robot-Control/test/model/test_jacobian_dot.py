# -*- coding: utf-8 -*-
"""阶段 0.1：`J̇_a` 的实现与验证（对应计划 §5.0.1 的两层判据）。

验证的是书 Part IV 两个 OS 算法框里的 `J̇_a ← auto_diff(J_a, t)`：

    J̇_a(q, q̇) = Σ_{k=1}^{N} ∂J_a(q)/∂q_k · q̇_k

**两层判据**（计划 §5.0.1，也是 §5.0.5 判据表的第 2、3 条）：

1. **矩阵一致性**：沿已知轨迹比较 `J̇_a^diff`（差分：`model/code_jacobian_dot.py`）与
   `J̇_a^exact`（**2R 闭式**：`code/code_models.py`，机器精度真值）——
   相对误差应随 `h` 按 `O(h²)` 下降，最小值 `< 1e-6`；6R 用 **Richardson 外推的时间
   中心差分**（第二条件独立路线，误差 `O(τ⁴)`）作参照。
2. **乘积法则一致性**（更关键、易漏）：

       d/dt[ J_a(q(t)) q̇(t) ] = J̇_a(q, q̇) q̇ + J_a(q) q̈

   左边由 `robot_fk` 的二阶中心差分给出（即 `ẍ_e`），右边由解析 `J_a` 与差分 `J̇_a` 给出。
   **常见 bug 是算对了 `J̇_a` 矩阵却漏掉 `J̇_a q̇` 这一项**——本脚本同时算出「漏项版」的
   误差作为**灵敏度自检**，证明该测试确实能捕获它。

独立参照的意义：2R 闭式、Richardson 时间差分与 FK 二阶差分是三条**互不依赖**的路线，
与被测的「对 q 逐列中心差分」实现均不同源，故不是自证。

运行（`py311-gym`，任意 cwd）：

    python Robot-Control/test/model/test_jacobian_dot.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RC = os.path.abspath(os.path.join(_HERE, "..", ".."))          # Robot-Control/
sys.path.insert(0, _RC)
sys.path.insert(0, os.path.join(_RC, "code"))

import code_models as cm                                          # noqa: E402
from model import (robot_fk, robot_jacobian_a,                   # noqa: E402
                   robot_jacobian_dot_a, jacobian_dot_fd)

# 阶段 0 的随机种子（结果归档里记录，保证可复现）
SEED = 0
# 判据阈值（计划 §5.0.5 判据 2、3）
TOL_REL = 1e-6


def rel_err(A, B):
    """相对误差：`‖A-B‖_F / ‖B‖_F`（以参照量的 Frobenius 范数为分母）。"""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))


def fit_slope(hs, errs, factors=(10.0, 3.0, 1.0)):
    """对**截断主导段**做 log-log 最小二乘拟合，返回 `(slope, 用到的 h 列表, 用的倍数)`。

    中心差分的误差是 `O(h²) + 舍入平台`：`h` 小到一定程度后误差不再下降。若把舍入平台
    也算进拟合，斜率会被明显拉低，故只取 `err > factor × min(err)` 的点。
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


def wrap_delta(d):
    """把角度差折回 (-π, π]，避免 atan2 分支穿越污染差分。"""
    return d - 2.0 * np.pi * np.round(d / (2.0 * np.pi))


def f_jac_6r(d, a, alpha, N=6):
    """6R 的 `q -> J_a`（内部随 q 重算欧拉角，与 `robot_jacobian_dot_a` 的被测路径一致）。"""
    def f_jac(qq):
        eular_e = robot_fk(qq, d, a, alpha, N)[3:]
        return robot_jacobian_a(qq, d, a, alpha, eular_e, N)
    return f_jac


def jacobian_dot_time_fd(q, qdot, f_jac, tau=1e-3, richardson=True):
    """沿轨迹 `q(t) = q + q̇ t` 对 `J_a` 做**时间**中心差分，得 `J̇_a`（第二条路线）。

    Richardson 外推 `(4D(τ) - D(2τ))/3` 把误差从 `O(τ²)` 提到 `O(τ⁴)`，
    在 `τ = 1e-3` 时约 `1e-12` 量级，足以充当 6R 的参照。
    """
    def D(tau_):
        return (np.asarray(f_jac(q + qdot * tau_))
                - np.asarray(f_jac(q - qdot * tau_))) / (2.0 * tau_)
    if richardson:
        return (4.0 * D(tau) - D(2.0 * tau)) / 3.0
    return D(tau)


def fk_second_derivative(q, qdot, qddot, f_fk, tau=1e-4, richardson=True, wrap_rows=None):
    """`ẍ_e(0)`：沿 `q(t) = q + q̇t + ½q̈t²` 对 `robot_fk` 输出做二阶中心差分。

    `wrap_rows` 给定时，对这些行（欧拉角）先折回角度差再组合，避免 atan2 分支穿越；
    返回 `(ẍ, 最大角度增量)`，后者用于确认没有发生分支穿越。
    """
    def x_at(t):
        return np.asarray(f_fk(q + qdot * t + 0.5 * qddot * t * t))

    def D2(tau_):
        xp, x0, xm = x_at(tau_), x_at(0.0), x_at(-tau_)
        if wrap_rows is None:
            return (xp - 2.0 * x0 + xm) / (tau_ ** 2)
        d2 = wrap_delta(xp - x0) - wrap_delta(x0 - xm)
        out = (xp - 2.0 * x0 + xm) / (tau_ ** 2)
        out[wrap_rows] = d2[wrap_rows] / (tau_ ** 2)
        return out

    if wrap_rows is None:
        max_step = 0.0
    else:
        step = np.abs(wrap_delta(x_at(tau) - x_at(0.0)))
        max_step = float(np.max(step[wrap_rows]))
    if richardson:
        return (4.0 * D2(tau) - D2(2.0 * tau)) / 3.0, max_step
    return D2(tau), max_step


def check_matrix_2r(hs):
    """判据 1（2R）：差分 `J̇_a` vs 闭式 `J̇_a` 的收敛阶与相对误差。"""
    q, qdot = cm.Q_NOM_2R, cm.QDOT_NOM_2R
    exact = cm.robot_jacobian_dot_a_2r(q, qdot)
    f_jac2 = lambda qq: cm.robot_jacobian_a_2r(qq)               # noqa: E731

    errs = []
    print("  [2R] 差分 J̇_a（对 q 逐列中心差分）vs 2R 闭式真值：")
    for h in hs:
        approx = jacobian_dot_fd(f_jac2, q, qdot, h)
        e = rel_err(approx, exact)
        errs.append(e)
        print("       h = %.0e  相对误差 = %.3e  (max|Δ| = %.3e)" % (h, e, np.abs(approx - exact).max()))
    slope, used_h_2r, fac2r = fit_slope(list(hs), errs)
    e_min_2r = float(min(errs))
    h_min_2r = float(hs[int(np.argmin(errs))])
    print("       拟合收敛阶 = %.3f（理论 2.000；用 err > %.0f×min 的 %d 个点 %s）"
          % (slope, fac2r, len(used_h_2r), ["%.0e" % v for v in used_h_2r]))
    print("       舍入平台：最小相对误差 %.3e（h = %.0e）" % (e_min_2r, h_min_2r))

    # 第二条独立路线：时间中心差分 + Richardson，对闭式真值本身做交叉校验
    ref_time = jacobian_dot_time_fd(q, qdot, f_jac2, tau=1e-3, richardson=True)
    e_time = rel_err(ref_time, exact)
    print("       交叉校验：时间 Richardson 差分 vs 闭式 = %.3e" % e_time)

    measured = {
        "min_rel_err": float(min(errs)),
        "h_at_min": float(hs[int(np.argmin(errs))]),
        "slope": slope,
        "slope_points": [float(v) for v in used_h_2r],
        "rel_err_by_h": {("%.0e" % h): float(e) for h, e in zip(hs, errs)},
        "time_fd_vs_closed_form_rel_err": e_time,
    }
    passed = (min(errs) < TOL_REL) and (1.8 <= slope <= 2.2) and (e_time < TOL_REL)
    return passed, measured


def check_matrix_6r(hs):
    """判据 1（6R）：`model/code_jacobian_dot.py` vs Richardson 时间差分参照。"""
    d, a, alpha, _, _, _, _ = cm.puma_6r()
    q, qdot = cm.Q_NOM_6R, cm.QDOT_NOM_6R
    f_jac = f_jac_6r(d, a, alpha)

    ref = jacobian_dot_time_fd(q, qdot, f_jac, tau=1e-3, richardson=True)
    ref_plain = jacobian_dot_time_fd(q, qdot, f_jac, tau=1e-3, richardson=False)
    ref_plain_fine = jacobian_dot_time_fd(q, qdot, f_jac, tau=1e-4, richardson=False)
    ref_self = rel_err(ref, ref_plain_fine)
    print("  [6R] 参照路线自身的精度：τ=1e-3 的 O(τ²) 中心差分 vs Richardson 外推 = %.3e；"
          % rel_err(ref_plain, ref))
    print("       而 Richardson(τ=1e-3) vs O(τ²) 中心差分(τ=1e-4) = %.3e —— 取前者为参照。"
          % ref_self)

    errs = []
    print("  [6R] 差分 J̇_a（model/code_jacobian_dot.py）vs Richardson 时间差分参照：")
    for h in hs:
        approx = robot_jacobian_dot_a(q, qdot, d, a, alpha, N=6, h=h)
        e = rel_err(approx, ref)
        errs.append(e)
        print("       h = %.0e  相对误差 = %.3e  (max|Δ| = %.3e)" % (h, e, np.abs(approx - ref).max()))
    slope, used_h_6r, fac6r = fit_slope(list(hs), errs)
    e_min_6r = float(min(errs))
    h_min_6r = float(hs[int(np.argmin(errs))])
    print("       拟合收敛阶 = %.3f（理论 2.000；用 err > %.0f×min 的 %d 个点 %s）"
          % (slope, fac6r, len(used_h_6r), ["%.0e" % v for v in used_h_6r]))
    print("       舍入平台：最小相对误差 %.3e（h = %.0e）" % (e_min_6r, h_min_6r))

    measured = {
        "min_rel_err": float(min(errs)),
        "h_at_min": float(hs[int(np.argmin(errs))]),
        "slope": slope,
        "slope_points": [float(v) for v in used_h_6r],
        "rel_err_by_h": {("%.0e" % h): float(e) for h, e in zip(hs, errs)},
        "reference": "Richardson 时间中心差分, tau=1e-3",
        "reference_self_consistency_rel_err": ref_self,
    }
    passed = (min(errs) < TOL_REL) and (1.8 <= slope <= 2.2)
    return passed, measured


def check_product_rule(case, taus):
    """判据 2：乘积法则 `d/dt[J_a q̇] = J̇_a q̇ + J_a q̈`，并做「漏项版」灵敏度自检。"""
    rng = np.random.default_rng(SEED)
    if case == "2R":
        q, qdot = cm.Q_NOM_2R, cm.QDOT_NOM_2R
        qddot = np.zeros(3)
        qddot[1:] = rng.uniform(-0.8, 0.8, 2)
        f_fk = lambda qq: cm.robot_fk_2r(qq)                      # noqa: E731
        J_a = cm.robot_jacobian_a_2r(q)
        Jdot = cm.robot_jacobian_dot_a_2r(q, qdot)
        wrap_rows = None
    else:
        d, a, alpha, _, _, _, _ = cm.puma_6r()
        q, qdot = cm.Q_NOM_6R, cm.QDOT_NOM_6R
        qddot = np.zeros(7)
        qddot[1:] = rng.uniform(-0.8, 0.8, 6)
        f_fk = lambda qq: robot_fk(qq, d, a, alpha, 6)             # noqa: E731
        J_a = robot_jacobian_a(q, d, a, alpha, robot_fk(q, d, a, alpha, 6)[3:], 6)
        Jdot = robot_jacobian_dot_a(q, qdot, d, a, alpha, N=6)
        wrap_rows = [3, 4, 5]

    rhs_full = Jdot @ qdot[1:] + J_a @ qddot[1:]
    rhs_missing = J_a @ qddot[1:]                                 # 漏掉 J̇_a q̇ 的「常见 bug」版

    print("  [%s] 乘积法则（左边 = robot_fk 的二阶中心差分 = ẍ_e）：" % case)
    errs, errs_missing = [], []
    for tau in taus:
        lhs, max_step = fk_second_derivative(q, qdot, qddot, f_fk, tau=tau,
                                             richardson=True, wrap_rows=wrap_rows)
        e, e_m = rel_err(rhs_full, lhs), rel_err(rhs_missing, lhs)
        errs.append(e)
        errs_missing.append(e_m)
        print("       τ = %.0e  完整式 = %.3e   漏掉 J̇_a q̇ 项 = %.3e   （欧拉角单步最大变化 %.2e rad）"
              % (tau, e, e_m, max_step))

    measured = {
        "min_rel_err": float(min(errs)),
        "tau_at_min": float(taus[int(np.argmin(errs))]),
        "rel_err_by_tau": {("%.0e" % t): float(e) for t, e in zip(taus, errs)},
        "missing_term_min_rel_err": float(min(errs_missing)),
        "sensitivity_ratio": float(min(errs_missing) / max(min(errs), 1e-300)),
        "qddot_seed": SEED,
    }
    passed = min(errs) < TOL_REL and min(errs_missing) > 100.0 * min(errs)
    return passed, measured


def run(hs=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6), taus=(1e-3, 1e-4, 1e-5)):
    """跑全部 0.1 项测试，返回结论字典（供 run_stage0.py 归档）。"""
    print("=" * 72)
    print("阶段 0.1  J̇_a 的实现与验证（矩阵一致性 + 乘积法则）")
    print("  被测实现：model/code_jacobian_dot.py（对 q 逐列中心差分再加权）")
    print("  相对误差口径：‖差值‖_F / ‖参照‖_F；判据阈值 %.0e" % TOL_REL)
    print("=" * 72)

    print("\n-- 判据 2：J̇_a 矩阵一致性 --")
    ok2r, m2r = check_matrix_2r(hs)
    print()
    ok6r, m6r = check_matrix_6r(hs)
    ok_matrix = ok2r and ok6r

    print("\n-- 判据 3：乘积法则一致性 --")
    ok_pr_2r, m_pr_2r = check_product_rule("2R", taus)
    print()
    ok_pr_6r, m_pr_6r = check_product_rule("6R", taus)
    ok_product = ok_pr_2r and ok_pr_6r

    checks = {
        "jacobian_dot_matrix": {
            "desc": "J̇_a 矩阵一致性：最小相对误差 < 1e-6，且随 h 按 O(h²) 下降（2R 闭式 + 6R Richardson 参照）",
            "threshold": "min rel_err < 1e-6; 截断主导段拟合斜率∈[1.8, 2.2]（理论 2）",
            "measured": {"2R": m2r, "6R": m6r},
            "passed": bool(ok_matrix),
        },
        "jacobian_dot_product": {
            "desc": "J̇_a 乘积法则 d/dt[J_a q̇] = J̇_a q̇ + J_a q̈（左边由 robot_fk 二阶中心差分给出）",
            "threshold": "rel_err < 1e-6; 且漏项版误差 > 100× 完整版",
            "measured": {"2R": m_pr_2r, "6R": m_pr_6r},
            "passed": bool(ok_product),
        },
    }
    passed = ok_matrix and ok_product
    print("\n" + "-" * 72)
    print("阶段 0.1 结论：%s" % ("PASS" if passed else "FAIL"))
    for k, v in checks.items():
        print("   [%s] %s" % ("OK" if v["passed"] else "!!", k))
    return {"name": "阶段 0.1 J̇a 与乘积法则", "passed": bool(passed),
            "checks": checks, "hs": list(hs), "taus": list(taus)}


def main():
    res = run()
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
