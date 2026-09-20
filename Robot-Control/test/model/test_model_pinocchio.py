# -*- coding: utf-8 -*-
"""阶段 1：书 Part II 模型与 **Pinocchio** 权威实现的交叉验证。

验证对象（书 Part II，Robot-Control/model/ 的副本）：
  `robot_fk`、`robot_jacobian_w` / `robot_jacobian_a`、`robot_B`、`robot_dyn[B,C,g]`。

判据（书仓库 `docs/robot-control-verify/part4-motion-control-plan.md` §5.1）：
  | 对拍项 | 判据 |
  |---|---|
  | 正运动学 x_e | 位置 < 1e-10 m（本脚本用位姿矩阵 max|Δ| < 1e-10）|
  | 质量矩阵 B(q) | 相对误差 < 1e-8，且严格对称 |
  | 重力项 g(q)   | 相对误差 < 1e-8 |
  | C q̇           | 相对误差 < 1e-8（C 不唯一，只比乘积）|
  | 雅可比 J_w    | 相对误差 < 1e-8（两侧同为世界系 6×N）|
  另加**故意失配灵敏度自检**：把书中模型的质量 +5%，对拍必须能检出。

---

## 为什么需要「装配」这一步（本阶段的最大障碍，详见书仓库日志尝试 6/7）

书用的是**标准 D-H**：`A_n(q_n) = Tz(d_n)Tx(a_n)Rz(q_n)Rx(alpha_n)`，
`C_n(q) = A_1(q_1)···A_n(q_n)` 是第 n 个连杆系（D-H 第 n 号系，原点在远端关节处）。

Pinocchio 的关节链语义（本脚本用单关节实验钉死）是

    oM_i = Σ_1 Rz(q_1) Σ_2 Rz(q_2) ··· Σ_i Rz(q_i)          （Σ_i 为常量 SE3）

这里 **Rz 被施加在 Σ 之后**（等价于「Σ 的前两列被 Rz 旋转」）。直接取
`Σ_i = A_i(0)` **不等价**于书链（两者旋转相同、平移差一个 `Rz(q_i)`），
上一轮 session 正是卡在这里，并把「错位一格」误判成书的问题。

本脚本的做法：**不去猜下标**，而是把 `Σ_i` 与末端固定变换 `Mf` 作为未知常量，
在随机位形上用最小二乘**解**出来，并以「残差 ≤ 1e-12（机器精度）」作为
「书链确实能被 pin 的串联链精确表示」的证据（2R / 6R 均可达 ~2e-16）。
解出的常量只用于构造对拍模型，**不参与**任何结论——结论来自两侧独立模型的逐项比对。

运行环境：`D:\\Projects\\2024_MN4R\\.envs\\py311-pin\\python.exe`（pinocchio 4.1.0），
且**必须先**把该 env 的 `Library\\bin` 加进 `PATH`（否则 numpy 的 MKL 延迟加载失败）。
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

# --- 必须在 import numpy 之前把 pin 环境自带的 DLL 目录加进 PATH ---
_PIN_ENV = r"D:\Projects\2024_MN4R\.envs\py311-pin"
if os.path.isdir(os.path.join(_PIN_ENV, "Library", "bin")):
    os.environ["PATH"] = os.path.join(_PIN_ENV, "Library", "bin") + os.pathsep + os.environ.get("PATH", "")

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))            # .../Robot-Control
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_CODE = os.path.join(_ROOT, "code")
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from model import robot_B, robot_dyn, robot_fk, robot_jacobian_w  # noqa: E402

try:
    import pinocchio as pin
except Exception as exc:                                   # pragma: no cover
    pin = None
    _PIN_IMPORT_ERROR = exc
else:
    _PIN_IMPORT_ERROR = None


# ===========================================================================
# 用例参数：**唯一定义处是 code/code_models.py**，这里通过它取，不另写一份
# ===========================================================================
def _cases():
    import code_models as cm
    out = {}
    # 2R
    d2, a2, al2, m2, pc2, I2, g2 = cm.two_link_2r()
    out["2r"] = dict(N=2, d=d2, a=a2, alpha=al2, m=m2, p_cents=pc2, I_inn=I2, g0=g2)
    # 6R
    D, A, AL, M, PC, II, G = cm.puma_6r()
    out["6r"] = dict(N=6, d=D, a=A, alpha=AL, m=M, p_cents=PC, I_inn=II, g0=G)
    return out


_TOL_FIT = 1e-12


def _A_dh(q, d, a, alpha):
    """标准 D-H 变换（与 model/code_fk.py 的 calc_T_n_to_last 同式）。"""
    sq, cq = np.sin(q), np.cos(q)
    sa, ca = np.sin(alpha), np.cos(alpha)
    return np.array([
        [cq, -sq * ca, sq * sa, a * cq],
        [sq, cq * ca, -cq * sa, a * sq],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _Rz4(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])


def _book_fk_T(q_full, cs):
    """书中链的末端位姿（A_1(q_1)···A_N(q_N)），q_full 长度 N+1（[0] 不用）。"""
    d, a, al = cs["d"], cs["a"], cs["alpha"]
    T = np.eye(4)
    for i in range(1, cs["N"] + 1):
        T = T @ _A_dh(float(q_full[i]), float(d[i]), float(a[i]), float(al[i]))
    return T


def solve_assembly(cs, seed=0, n_samples=24, restarts=24, verbose=False):
    """解出 pin 串联链的常量 Σ_i 与末端固定变换 Mf，使 fk 链 == 书链。

    目标函数：随机位形上 ‖(Σ_1Rz(q_1)···Σ_NRz(q_N))·Mf − C_N(q)‖_F。
    返回 (max_residual, Sig(list of 4x4), Mf(4x4), info)。
    """
    from scipy.optimize import least_squares

    N = cs["N"]
    rng = np.random.default_rng(seed)
    Q = [rng.uniform(-1.3, 1.3, N) for _ in range(n_samples)]
    Tgt = [_book_fk_T(np.concatenate([[0.0], q]), cs) for q in Q]

    def build(x):
        Sig = [np.block([[pin.exp3(x[6 * i:6 * i + 3]), x[6 * i + 3:6 * i + 6].reshape(3, 1)],
                         [np.zeros((1, 3)), np.ones((1, 1))]]) for i in range(N)]
        Mf = np.block([[pin.exp3(x[6 * N:6 * N + 3]), x[6 * N + 3:6 * N + 6].reshape(3, 1)],
                       [np.zeros((1, 3)), np.ones((1, 1))]])
        return Sig, Mf

    def resid(x):
        Sig, Mf = build(x)
        out = []
        for q, T in zip(Q, Tgt):
            X = np.eye(4)
            for i in range(N):
                X = X @ Sig[i] @ _Rz4(q[i])
            out.append(((X @ Mf) - T).ravel())
        return np.concatenate(out)

    best = None
    for r in range(restarts):
        x0 = np.zeros(6 * N + 6)
        for i in range(N):
            x0[6 * i + 3:6 * i + 6] = _A_dh(0.0, float(cs["d"][i + 1]), float(cs["a"][i + 1]),
                                            float(cs["alpha"][i + 1]))[:3, 3] \
                + rng.normal(0, 0.25, 3)
        sol = least_squares(resid, x0, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=80000)
        e = float(np.abs(sol.fun).max())
        if best is None or e < best[0]:
            best = (e, sol.x)
        if e < 1e-14:
            break
    Sig, Mf = build(best[1])
    if verbose:
        print("    装配求解：残差 = %.3e（%d 次重启）" % (best[0], restarts))
    return best[0], Sig, Mf, {"restarts": restarts, "n_samples": n_samples, "seed": seed}


def build_pin_model(cs, Sig, Mf):
    """按解出的 Σ_i 建 pin 模型，并把书中每根连杆的惯量挂到正确的连杆系上。

    关键：**连杆 i 的 D-H 系 `C_i` 与 pin 第 i 个关节帧是同一根连杆上的固定关系**
    （本脚本以 `g` 作闸门核验：映射 `i → 关节 i` 时 `g` 相对误差 ~1e-15，
    其它映射（逆序、全部挂末端）差 0.4~7.6，见书仓库日志尝试 7）。
        Mrel_i = inv(oM_i)|_{q=0} · C_i(0)          （常量）
        c_pin  = Mrel_i^t + Mrel_i^R p_cents[i]
        I_pin  = Mrel_i^R I_inn[i] Mrel_i^{R,T}
    ⚠️ `pin.Inertia(m, c, I)` 的 `I` 是**绕质心**的惯量（内部自己做平行轴定理），
    因此这里**不能**再加 `m(‖c‖²E − c cᵀ)`——加了会重复计入（实测 B 相对误差从 1e-16 变 2e-1）。
    """
    N = cs["N"]
    d, a, al = cs["d"], cs["a"], cs["alpha"]
    model = pin.Model()
    parent = 0
    for i in range(1, N + 1):
        S = Sig[i - 1]
        parent = model.addJoint(parent, pin.JointModelRZ(),
                                pin.SE3(S[:3, :3].copy(), S[:3, 3].copy()), "joint%d" % i)
    model.addFrame(pin.Frame("ee", N, 0, pin.SE3(Mf[:3, :3].copy(), Mf[:3, 3].copy()),
                             pin.FrameType.OP_FRAME))
    model.gravity.linear = np.asarray(cs["g0"], float).copy()
    # q=0 的 pin 关节帧姿态，用来定出常量 Mrel_i
    data0 = model.createData()
    pin.forwardKinematics(model, data0, np.zeros(N))
    Cu = np.eye(4)
    for i in range(1, N + 1):
        Cu = Cu @ _A_dh(0.0, float(d[i]), float(a[i]), float(al[i]))
        Mrel = np.linalg.inv(data0.oMi[i].homogeneous) @ Cu
        c = Mrel[:3, 3] + Mrel[:3, :3] @ np.asarray(cs["p_cents"][i], float)
        Ii = Mrel[:3, :3] @ np.asarray(cs["I_inn"][i], float) @ Mrel[:3, :3].T
        model.appendBodyToJoint(
            model.getJointId("joint%d" % i),
            pin.Inertia(float(cs["m"][i]), c.copy(), np.asarray(Ii, float).copy()),
            pin.SE3.Identity())
    return model, model.createData(), model.getFrameId("ee")


# ===========================================================================
# 对拍项
# ===========================================================================
def _rel(a, b):
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / (nb if nb > 0 else 1.0))


def check_case(tag, cs, seed, verbose=True):
    N = cs["N"]
    res = {"case": tag, "N": N, "items": {}, "ok": True}
    if pin is None:                                        # pragma: no cover
        res["ok"] = False
        res["skip"] = "pinocchio 不可用：%s" % _PIN_IMPORT_ERROR
        return res

    t0 = time.perf_counter()
    fit_res, Sig, Mf, fit_info = solve_assembly(cs, seed=seed)
    res["assembly"] = {"fit_residual": fit_res, **fit_info}

    model, data, fid = build_pin_model(cs, Sig, Mf)
    d, a, al, m = cs["d"], cs["a"], cs["alpha"], cs["m"]
    g0 = cs["g0"]

    rng = np.random.default_rng(seed + 1000)
    Q = [rng.uniform(-1.2, 1.2, N) for _ in range(12)]
    Q += [np.zeros(N)]

    # --- 1) 正运动学 ---
    e_fk = 0.0
    for q in Q:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        e_fk = max(e_fk, float(np.abs(data.oMf[fid].homogeneous - _book_fk_T(np.concatenate([[0.0], q]), cs)).max()))
    res["items"]["fk"] = {"metric": "位姿矩阵 max|Δ|", "value": e_fk, "tol": 1e-10,
                          "pass": bool(e_fk < 1e-10)}

    # --- 2) 质量矩阵 B ---
    e_B, sym = 0.0, 0.0
    for q in Q:
        qq = np.concatenate([[0.0], q])
        B_book = robot_B(qq, d, a, al, m, cs["p_cents"], cs["I_inn"], N)
        pin.crba(model, data, q)
        e_B = max(e_B, _rel(data.M, B_book))
        sym = max(sym, float(np.abs(data.M - data.M.T).max()))
    res["items"]["B"] = {"metric": "相对误差", "value": e_B, "tol": 1e-8,
                         "pass": bool(e_B < 1e-8), "symmetry_dev": sym,
                         "symmetry_pass": bool(sym < 1e-12)}

    # --- 3) 重力项 g ---
    e_g = 0.0
    for q in Q:
        qq = np.concatenate([[0.0], q])
        _, _, g_book = robot_dyn(qq, np.zeros(N + 1), d, a, al, m, cs["p_cents"], cs["I_inn"], g0, N)
        gp = pin.rnea(model, data, q, np.zeros(N), np.zeros(N))
        e_g = max(e_g, _rel(gp, g_book))
    res["items"]["g"] = {"metric": "相对误差", "value": e_g, "tol": 1e-8, "pass": bool(e_g < 1e-8)}

    # --- 4) C q̇（C 不唯一，只比乘积）---
    # ⚠️ 本项存在**方法本身的下限**：书 robot_dyn 的 C 是对 ∂B/∂q 的中心差分近似
    # （阶段 0 已量化：h=1e-5 最优、相对误差 ~7.5e-11），而 pin.rnea 的 C 是解析的。
    # 因此残差 = 书侧差分残差 (C_数 − C_true)q̇，量级 ~1e-9（q̇ 放大后），
    # 与 1e-8 的判据同量级但**不是模型错误**——判据通过，且该残差本身就是阶段 0.2
    # 「是否要升级 C」的定量依据（见书仓库日志尝试 2）。
    e_cq = 0.0
    for q in Q:
        qd = rng.uniform(-0.8, 0.8, N)
        qq = np.concatenate([[0.0], q])
        qqd = np.concatenate([[0.0], qd])
        _, C_book, _ = robot_dyn(qq, qqd, d, a, al, m, cs["p_cents"], cs["I_inn"], g0, N)
        tau = pin.rnea(model, data, q, qd, np.zeros(N))
        gp = pin.rnea(model, data, q, np.zeros(N), np.zeros(N))
        e_cq = max(e_cq, _rel(tau - gp, C_book @ qd))
    res["items"]["Cqdot"] = {"metric": "相对误差（Cq̇）", "value": e_cq, "tol": 1e-8,
                             "pass": bool(e_cq < 1e-8)}

    # --- 5) 几何雅可比 J_w（世界系）---
    e_j = 0.0
    data_j = model.createData()
    for q in Q:
        Jw_book = robot_jacobian_w(np.concatenate([[0.0], q]), d, a, al, N)
        J = pin.computeFrameJacobian(model, data_j, q, fid, pin.LOCAL_WORLD_ALIGNED)
        e_j = max(e_j, _rel(J, Jw_book))
    res["items"]["Jw"] = {"metric": "相对误差", "value": e_j, "tol": 1e-8, "pass": bool(e_j < 1e-8)}

    res["elapsed_s"] = time.perf_counter() - t0
    res["ok"] = all(v.get("pass", True) for v in res["items"].values())
    if verbose:
        print("  [%s] 装配残差 %.2e | FK %.2e | B %.2e | g %.2e | Cq̇ %.2e | J_w %.2e"
              % (tag, fit_res, e_fk, e_B, e_g, e_cq, e_j))
    return res


def check_mismatch(tag, cs, seed):
    """故意失配灵敏度自检：把书中模型质量整体 +5%，对拍必须能检出。"""
    N = cs["N"]
    if pin is None:                                        # pragma: no cover
        return {"case": tag, "skip": "pinocchio 不可用", "pass": False}
    cs_nom = dict(cs)
    cs_bad = dict(cs)
    cs_bad["m"] = np.asarray(cs["m"], float).copy()
    cs_bad["m"][1:] *= 1.05

    # 用同一套 Σ/Mf（只由运动学决定），只把惯量改成 +5% 版
    fit_res, Sig, Mf, _ = solve_assembly(cs_nom, seed=seed)
    model_ref, data_ref, _ = build_pin_model(cs_nom, Sig, Mf)
    model_bad, data_bad, _ = build_pin_model(cs_bad, Sig, Mf)

    rng = np.random.default_rng(seed + 7)
    q = rng.uniform(-1.2, 1.2, N)
    qq = np.concatenate([[0.0], q])
    B_book = robot_B(qq, cs_nom["d"], cs_nom["a"], cs_nom["alpha"], cs_nom["m"],
                     cs_nom["p_cents"], cs_nom["I_inn"], N)
    pin.crba(model_ref, data_ref, q)
    err_ref = _rel(data_ref.M, B_book)
    pin.crba(model_bad, data_bad, q)
    err_bad = _rel(data_bad.M, B_book)
    detected = err_bad > 1e-3 and err_bad > 100 * max(err_ref, 1e-16)
    return {"case": tag, "err_nominal": err_ref, "err_perturbed_5pct": err_bad,
            "detected": bool(detected), "pass": bool(detected)}


def run(verbose=True):
    out = {"stage": 1, "name": "与 Pinocchio 权威实现对拍", "pinocchio": None,
           "cases": {}, "mismatch": {}, "ok": None}
    if pin is None:                                        # pragma: no cover
        out["ok"] = False
        out["skip"] = "pinocchio 不可用：%s" % _PIN_IMPORT_ERROR
        if verbose:
            print("!! pinocchio 不可用，阶段 1 跳过：%s" % _PIN_IMPORT_ERROR)
        return out
    out["pinocchio"] = {"version": pin.__version__,
                        "numpy": np.__version__, "python": sys.version.split()[0],
                        "interpreter": sys.executable}
    cases = _cases()
    seeds = {"2r": 11, "6r": 5}
    for tag, cs in cases.items():
        if verbose:
            print("== 用例 %s ==" % tag)
        out["cases"][tag] = check_case(tag, cs, seeds[tag], verbose=verbose)
    for tag, cs in cases.items():
        out["mismatch"][tag] = check_mismatch(tag, cs, seeds[tag])
        if verbose:
            mm = out["mismatch"][tag]
            print("  [%s] 故意失配自检：标称 %.2e → +5%% 质量后 %.2e  ⇒ %s"
                  % (tag, mm.get("err_nominal", float("nan")),
                     mm.get("err_perturbed_5pct", float("nan")),
                     "已检出" if mm.get("pass") else "未检出（测试太松）"))
    out["ok"] = all(c["ok"] for c in out["cases"].values()) \
        and all(m["pass"] for m in out["mismatch"].values())
    return out


def main():
    print("=" * 72)
    print("阶段 1：与 Pinocchio 权威实现交叉验证")
    print("=" * 72)
    result = run(verbose=True)
    res_dir = os.path.join(_ROOT, "res", "model_stage1")
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "result.json"), "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print("-" * 72)
    print("阶段 1 结论：%s（归档 res/model_stage1/result.json）"
          % ("PASS" if result["ok"] else "FAIL"))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                      # pragma: no cover
        traceback.print_exc()
        sys.exit(2)
