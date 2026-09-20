# -*- coding: utf-8 -*-
"""阶段 3 一键运行与归档：**6R 正式验证与专项测试**（计划 §5.3 的 6 条判据 + §6.1 的图）。

在**同一进程内**依次跑四个算法的阶段 3 检查与跨算法专项（不用 `subprocess`：① 避免管道在受限
沙箱下被拦；② 各测试模块的 `run_stage3()` 已返回结构化结论，直接合并即可）：

| 顺序 | 内容 | 对应计划 | 判据 |
|---|---|---|---|
| 1 | 4 个算法的 6R 调节（20 组初值）+ 各自的专项 | §3.7 | 3（全部）、1（两个逆动力学）、2（两个 JS） |
| 2 | JS/OS 分工边界（3 档轨迹速度 × 4 控制器） | §3.3 | 4 |
| 3 | 耦合抑制（第 1 关节快速轨迹） | §3.4 | 5 |
| 4 | OS 双向自洽（`F = J_a^{-T}u`，24 个随机状态） | §3.5 | 6 |
| 5 | OS 的 `K_P` 扫描（常值未补偿扰动） | §3.5 | 报告项 |
| 6 | `x̃ ≈ J_aq̃` 一阶线性化偏差 | §3.5/§3.6 | 报告项 |
| 7 | **步长收敛对照 `dt = 1 ms vs 2 ms`** | 决策记录 1 | 报告项（决策 1 的强制要求） |

**判据归属**（计划 §5.3.7）：
判据 1 只对两个**逆动力学**算法成立（只有它们的误差动态是单位质量二阶系统）；
判据 2 只对 **JS 重力补偿 PD** 成立（任务空间的 Lyapunov 函数含任务惯量，是另一套）；
判据 5 **必须用 6R**（2R 耦合太弱）；判据 3 是四个算法的集体判据。

产出（见 `res/README.md` 的归档契约）：

- `res/6r_stage3/result.json` —— 机读结果：环境/版本、6 条判据的阈值与实测、四个算法的完整明细、
  跨算法专项、步长对照、口径说明与局限；
- `res/6r_stage3/run.log` —— 本次运行的完整 stdout；
- `res/6r_stage3/figs/*.png` —— 误差曲线、控制力矩、对数衰减率拟合、速度-误差对比、
  耦合柱状图、Lyapunov 曲线、`K_P` 扫描。

用法（`py311-gym`，任意 cwd；完整跑约 60–90 min，**务必后台执行**）：

    python Robot-Control/test/ctrl_common/run_stage3.py
    python Robot-Control/test/ctrl_common/run_stage3.py --quick        # 冒烟（只看链路跑通）
    python Robot-Control/test/ctrl_common/run_stage3.py --only dt,kp   # 只跑指定实验（调试用）
    python Robot-Control/test/ctrl_common/run_stage3.py --workers 6    # 指定并发（默认 6）

退出码：6 条判据全 PASS 为 0，否则 1（`--quick` 只依据「无异常」）。
"""
import contextlib
import datetime
import json
import os
import platform
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RC = os.path.abspath(os.path.join(_HERE, "..", ".."))              # Robot-Control/
for _p in (_RC, os.path.join(_RC, "code"), _HERE,
           os.path.join(_RC, "test", "ctrl_js"), os.path.join(_RC, "test", "ctrl_os")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage3_lib as s3                                             # noqa: E402
import verify_lib as vl                                             # noqa: E402
import test_js_invdyn                                               # noqa: E402
import test_js_pd                                                   # noqa: E402
import test_os_invdyn                                               # noqa: E402
import test_os_pd                                                   # noqa: E402

MODULES = [test_js_pd, test_js_invdyn, test_os_pd, test_os_invdyn]
ALL_PARTS = ("algs", "div", "coup", "map", "kp", "bias", "dt")

# (判据号, 判据名, 该判据由哪些算法承担)
JUDGE = [
    ("1", "误差动态精确性：实测 (ξ̂, ω̂_n) 与设定值相对误差 < 2%（dt = 1 ms）",
     ["js_invdyn", "os_invdyn"]),
    ("2", "重力补偿 PD 的无源性：Lyapunov 函数数值积分无正功事件", ["js_pd"]),
    ("3", "6R 收敛性（拆 3a/3b）：书中「J_a 列满秩」前提成立的初值全部收敛到 ‖q̃‖_∞ < 1e-3 rad",
     ["js_pd", "js_invdyn", "os_pd", "os_invdyn"]),
    ("4", "JS/OS 分工：快速轨迹下逆动力学误差显著小于重力补偿 PD", ["js_pd+js_invdyn", "os_pd+os_invdyn"]),
    ("5", "耦合抑制（6R）：逆动力学的非指令位移接近零、PD 明显非零", ["js_pd+js_invdyn"]),
    ("6", "OS 双向自洽：F = J_a^{-T}u 的映射误差在数值精度量级", ["os_pd", "os_invdyn"]),
]

SCOPE_NOTES = [
    "判据 1 只对**逆动力学**算法成立：书中两条重力补偿 PD 的误差动态不是单位质量二阶系统"
    "（JS 未抵消 B、OS 未抵消任务惯量且多一项 J̇_aq̇），(ξ̂, ω̂_n) 无严格基准，故不适用。",
    "⚠️ **判据 1 的判定步长是 dt = 1 ms**（`stage3_lib.DT_FIT3`），不是阶段 3 的主步长 2 ms："
    "该判据是「取定值比对」型（<2%），而 2 ms 下最小惯量关节 q̃_6（B_66 = 1.2e-3）的 ξ̂ 有"
    "+2.9% 的**离散化系统偏差**（定点探针 `scripts/probe_fit_dt.py` 量化，见日志尝试 10）；"
    "步长影响一律在 `step_size_check` 里以**逐分量**形式报告。",
    "判据 2 只对 **JS 重力补偿 PD** 成立：计划 §3.2 的 Lyapunov 函数 V = ½q̇ᵀBq̇ + ½q̃ᵀK_Pq̃ "
    "是关节空间的；js_invdyn 用单位质量能量作**补充记录**（非判据）。",
    "⚠️ **判据 3 拆为 3a/3b**（2026-09-19 修订，日志尝试 10）：3a = 轨迹全程 cond(J_a) ≤ %g"
    "（书中「J_a 列满秩」前提的定量化，标称位形实测 cond = 12.70，阈值留近 8 倍余量）的初值"
    "**必须全部收敛**；3b = 前提被违反的初值（瞬态掠过奇异区）如实单列为**边界样本**，"
    "与计划 §5.4.1（阶段 4.1 的奇异边界）呼应。**阈值 ‖q̃‖_∞ < 1e-3 rad 未动**，"
    "且同时报告「不设前提时 20 组收敛几组」。" % s3.COND_J_PREMISE,
    "判据 5 **必须用 6R**（计划 §5.3.4）：2R 的关节耦合太弱，不足以验证耦合补偿的必要性。",
    "⚠️ 被控对象与控制器模型在本阶段**同源**（标称用例）：只证明「实现与书中公式自洽」，"
    "不能排除「书的模型本身错了」——后者由阶段 1（Pinocchio 权威对拍，已 PASS）承担，"
    "模型误差敏感度由阶段 4 承担。",
    "⚠️ 步长口径：阶段 3 主步长 dt = 2 ms（用户 2026-09-19 决策，见计划 §九「决策记录」1），"
    "**必须**附 dt = 1 ms vs 2 ms 的步长收敛对照（result.json 的 step_size_check）。",
    "⚠️ 增益口径：重力补偿类按参考惯量标定 Λ（6R 必需，否则离散化失稳——实测 Lam=I 时 "
    "‖u‖ → 1e305）；逆动力学类用书中原式 K_P = ω_n²I，误差动态才是逐分量解耦的单位质量二阶系统。",
    "OS 用例的 q̃ 在仿真记录里恒为 0（ref 只含 x_d，红线：不把 q_d 给 OS 控制器）——"
    "评测层一律用实验者选定的 q_d（x_d = fk(q_d)，非逆解）重算，与阶段 2 的口径一致。",
]


class _Tee:
    """把 stdout 同时写到控制台与日志文件（`print(..., flush=True)` 照常透传）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _env_info():
    info = {"python": sys.version.split()[0], "executable": sys.executable,
            "numpy": np.__version__, "platform": platform.platform()}
    for mod in ("scipy", "matplotlib"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:                                            # noqa: BLE001
            info[mod] = None
    return info


def _strip(o):
    """递归去掉 `_raw`/`_rec`（含 numpy 数组与 RobotCase，不可 JSON 化），并转 numpy 标量。"""
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_strip(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _criteria(algs, div, coup, maps):
    """把各实验的 checks 汇总成计划 §5.3.7 的 6 条判据（逐条列出承担的算法）。"""
    per = {
        "1": {k: algs[k]["checks"]["err_dyn_fit"] for k in ("js_invdyn", "os_invdyn")},
        "2": {"js_pd": algs["js_pd"]["checks"]["passivity"]},
        "3": {k: algs[k]["checks"]["convergence"] for k in s3.KIND_ORDER},
        "4": {"js_pair": div["check"], "os_pair": div["check"]},
        "5": {"js_pair": coup["check"]},
        "6": {k: maps[k]["check"] for k in ("os_pd", "os_invdyn")},
    }
    out = []
    for cid, name, who in JUDGE:
        d = per[cid]
        out.append({
            "id": cid, "name": name, "checked_on": who,
            "threshold": next(iter(d.values()))["threshold"],
            "measured": {k: v["measured"] for k, v in d.items()},
            "passed_per_scope": {k: bool(v["passed"]) for k, v in d.items()},
            "passed": bool(all(v["passed"] for v in d.values())),
        })
    return out


def _write_json(path, summary):
    """把汇总写成归档 JSON（两处调用：出图前先落盘、出图后补 `figures`）。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_np_default)


def _load_probe(out_dir):
    """把判据 1 的步长定点探针结果（`scripts/probe_fit_dt.py` 的产出）并入归档。

    该探针是**判据 1 判定步长的依据**（日志尝试 10 的「下一步」第 1 条），必须与本次运行
    一起留档；扫描 `res/6r_stage3/probe_fit_dt*.json`（`probe_fit_dt.json` 是 3 个步长的
    收敛阶证据，`probe_fit_dt_T30.json` 是最终判定窗口 `T = 3.0 s` 的定点对照），
    一个都没有时返回 `None` 并在 `run.log` 里提示（不静默）。
    """
    found = {}
    try:
        names = sorted(n for n in os.listdir(out_dir)
                       if n.startswith("probe_fit_dt") and n.endswith(".json"))
    except OSError:
        names = []
    for n in names:
        try:
            with open(os.path.join(out_dir, n), encoding="utf-8") as f:
                found[os.path.splitext(n)[0]] = json.load(f)
        except (OSError, ValueError) as exc:                            # noqa: BLE001
            print("⚠️ 读取步长探针 %s 失败：%s" % (n, exc))
    if not found:
        print("⚠️ 未找到判据 1 的步长定点探针 probe_fit_dt*.json —— 归档将缺该项"
              "（先跑 scripts/probe_fit_dt.py）")
        return None
    return found


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    parts = set(ALL_PARTS)
    for i, a in enumerate(argv):
        if a == "--only" and i + 1 < len(argv):
            parts = {p.strip() for p in argv[i + 1].split(",") if p.strip()}
        elif a.startswith("--only="):
            parts = {p.strip() for p in a.split("=", 1)[1].split(",") if p.strip()}
    bad = parts - set(ALL_PARTS)
    if bad:
        print("未知实验名 %s（可选：%s）" % (sorted(bad), ",".join(ALL_PARTS)))
        return 2
    workers = None
    for i, a in enumerate(argv):
        if a == "--workers" and i + 1 < len(argv):
            workers = int(argv[i + 1])
        elif a.startswith("--workers="):
            workers = int(a.split("=", 1)[1])
    if "--serial" in argv:
        workers = 1
    if workers is None:
        workers = vl.default_workers()
    out_dir = os.path.join(_RC, "res", "6r_stage3")
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    json_path = os.path.join(out_dir, "result.json")

    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8", newline="") as log_f, \
            contextlib.redirect_stdout(_Tee(sys.stdout, log_f)):
        print("#" * 72)
        print("# 阶段 3：6R 正式验证与专项测试（一键运行 + 归档）%s"
              % ("  [--quick 冒烟模式]" if quick else ""))
        print("# 时间：%s" % datetime.datetime.now().isoformat(timespec="seconds"))
        print("# 环境：%s" % sys.executable)
        print("# 实验：%s" % ",".join(sorted(parts)))
        print("# 产出：%s" % out_dir)
        print("#" * 72)

        algs, div, coup, maps, kp, bias, dtchk, pas_raw, fit_raw = {}, None, None, {}, None, None, None, {}, {}
        warnings = []
        if "algs" in parts:
            for mod in MODULES:
                print("\n" + "#" * 72)
                print("# >>> 阶段 3：%s（%s）" % (mod.__name__, vl.CTRLS[mod.KIND]["name"]))
                print("#" * 72)
                t_mod = time.perf_counter()
                res = mod.run_stage3(case_name="6R", quick=quick, workers=workers)
                res["wall_time_s"] = time.perf_counter() - t_mod
                algs[res["kind"]] = res
                if res.get("extras", {}).get("passivity"):
                    pas_raw[res["kind"]] = res["extras"]["passivity"]
                if res.get("extras", {}).get("error_dynamics"):
                    fit_raw[res["kind"]] = res["extras"]["error_dynamics"]
        if "div" in parts:
            print("\n" + "#" * 72 + "\n# >>> 判据 4：JS/OS 分工边界（快速轨迹）\n" + "#" * 72)
            tl = (s3.T_TRAJ_LIST[0],) if quick else s3.T_TRAJ_LIST
            div = s3.division_of_labor(t_list=tl, workers=workers)
        if "coup" in parts:
            print("\n" + "#" * 72 + "\n# >>> 判据 5：耦合抑制（第 1 关节快速轨迹，6R）\n" + "#" * 72)
            coup = s3.coupling(workers=workers)
        if "map" in parts:
            print("\n" + "#" * 72 + "\n# >>> 判据 6：OS 双向自洽（F = J_a^{-T}u）\n" + "#" * 72)
            n_st = 4 if quick else s3.N_MAP_STATES
            for kind in ("os_pd", "os_invdyn"):
                maps[kind] = s3.os_mapping(kind, n_states=n_st)
        if "kp" in parts:
            print("\n" + "#" * 72 + "\n# >>> §3.5：OS 的 K_P 扫描（常值未补偿扰动）\n" + "#" * 72)
            wl = s3.WN_KP_LIST[:1] if quick else s3.WN_KP_LIST
            kp = s3.os_kp_sweep(wn_list=wl, workers=workers)
        if "bias" in parts:
            print("\n" + "#" * 72 + "\n# >>> §3.5/§3.6：x̃ ≈ J_aq̃ 的一阶线性化偏差\n" + "#" * 72)
            guard = s3.euler_convention_selftest()
            print("      守卫：欧拉角 ↔ 旋转矩阵往返最大偏差 %.3e" % guard["worst_roundtrip_err"])
            bias = s3.linearization_bias(deltas=s3.DELTA_BIAS[:3] if quick else s3.DELTA_BIAS)
            bias["euler_roundtrip_guard"] = guard
        if "dt" in parts and not quick:
            print("\n" + "#" * 72 + "\n# >>> 决策记录 1：步长收敛对照 dt = 1 ms vs 2 ms\n" + "#" * 72)
            dtchk = s3.step_size_check()
        elif "dt" in parts and quick:
            warnings.append("--quick：跳过步长收敛对照（耗时最长的一项）")

        criteria = _criteria(algs, div, coup, maps) if (algs and div and coup and maps) else []
        if criteria and not quick:
            n_pass = sum(1 for c in criteria if c["passed"])
            passed = n_pass == len(criteria)
        else:
            n_pass = sum(1 for c in criteria if c["passed"])
            passed = True
            warnings.append("冒烟/部分运行：判据表不完整或窗口被压缩，PASS/FAIL 无意义；"
                            "退出码只依据「无异常退出」。")

        elapsed = time.perf_counter() - t0
        summary = {
            "stage": 3,
            "name": "阶段 3：6R 正式验证与专项测试",
            "case": "6R",
            "plan": "1-MN4R/docs/robot-control-verify/part4-motion-control-plan.md §5.3",
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "wall_time_s": elapsed,
            "quick_mode": bool(quick),
            "parts_run": sorted(parts),
            "workers": workers,
            "env": _env_info(),
            "seed": s3.SEED3,
            "experiment": {
                "n_ic": s3.N_IC3, "T_reg_s": s3.T_REG_6R,
                "dt_s": s3.DT3, "dt_fit_s": s3.DT_FIT3,
                "dt_step_check_s": [s3.DT3_REF, s3.DT3],
                "tol_convergence_rad": s3.TOL_CONV3,
                "cond_J_premise": s3.COND_J_PREMISE,
                "ic_box": {"dq_max_rad": s3.DQ3, "dqdot_max_rad_s": s3.DQDOT3},
                "ell_m": 0.5,
                "gain_wn": dict(s3.WN_S3), "gain_xi": s3.XI_S3,
                "lam_calibration": {k: (s3.LAM_KIND[k] or "I") for k in s3.KIND_ORDER},
                "second_order": {"wn": s3.WN_2ND, "xi": s3.XI_2ND, "T_s": s3.T_2ND,
                                 "dq_rad": s3.DQ_2ND, "tol_rel": s3.TOL_FIT},
                "passivity": {"T_s": s3.T_PAS, "dq_rad": s3.DQ_PAS,
                              "tol_positive_work_fraction": s3.TOL_POS_WORK},
                "division_of_labor": {"T_traj_list_s": list(s3.T_TRAJ_LIST),
                                      "dq_js_rad": s3.DQ_TRAJ_JS.tolist(),
                                      "min_speedup": s3.MIN_SPEEDUP},
                "coupling": {"T_traj_s": s3.T_COUP, "amp_rad": s3.AMP_COUP,
                             "tol_invdyn_rad": s3.TOL_COUP_INVDYN,
                             "min_ratio": s3.MIN_COUP_RATIO},
                "os_kp_sweep": {"wn_list": list(s3.WN_KP_LIST), "T_s": s3.T_KP_S3,
                                "tau_e_Nm": s3.TAU_E_S3.tolist(),
                                "tol_slope": list(s3.TOL_KP_SLOPE)},
                "os_mapping": {"n_states": s3.N_MAP_STATES, "tol_rel": s3.TOL_MAP},
            },
            "criteria": criteria,
            "passed_count": n_pass,
            "total_count": len(criteria),
            "passed": bool(passed),
            "algorithms": {k: _strip(v) for k, v in algs.items()},
            "division_of_labor": _strip(div) if div else None,
            "coupling": _strip(coup) if coup else None,
            "os_mapping": {k: _strip(v) for k, v in maps.items()} or None,
            "os_kp_sweep": _strip(kp) if kp else None,
            "linearization_bias": _strip(bias) if bias else None,
            "step_size_check": _strip(dtchk) if dtchk else None,
            "fit_dt_probe": _load_probe(out_dir),
            "traces": {k: vl._trace_json(v["_raw"]["recs"][v["worst_ic_index"]])
                       for k, v in algs.items()} if algs else None,
            "scope_notes": SCOPE_NOTES,
            "warnings": warnings,
            "figures": [],
            "figure_error": None,
        }
        # ★★ **先落盘，再出图**（2026-09-19 的教训，见 `make_figures` 的文档）：
        # 首次完整跑在出图处抛 `KeyError`，导致 **63.7 min 的全部算力与判据结果一起丢失**
        # （`result.json` 仍是上一次冒烟的残档）。归档是主要产出，绝不能排在装饰性步骤之后。
        _write_json(json_path, summary)
        figs, fig_error = [], None
        if not quick and algs and div and coup and pas_raw and fit_raw:
            try:
                figs = s3.make_figures(algs, div, coup, pas_raw, fit_raw, kp, fig_dir, tag="6r")
            except Exception as exc:                                    # noqa: BLE001
                fig_error = "%s: %s" % (type(exc).__name__, exc)
                warnings.append("出图失败（**结果归档已先写出，不受影响**）：%s" % fig_error)
                print("⚠️ 出图失败：%s" % fig_error)
                import traceback
                traceback.print_exc()
        else:
            warnings.append("冒烟/部分运行：跳过出图（图需要完整的四算法结果）")
        summary["figures"] = [os.path.relpath(p, out_dir).replace("\\", "/") for p in figs]
        summary["figure_error"] = fig_error
        _write_json(json_path, summary)

        print("\n" + "=" * 72)
        print("阶段 3 判据表汇总（计划 §5.3.7）")
        print("=" * 72)
        for c in criteria:
            print("  [%s] 判据 %s  %s" % ("PASS" if c["passed"] else "FAIL", c["id"], c["name"]))
            print("         阈值：%s" % c["threshold"])
            print("         逐对象：%s" % ", ".join(
                "%s=%s" % (k, "PASS" if v else "FAIL") for k, v in c["passed_per_scope"].items()))
        if not criteria:
            print("  （判据表不完整：本次只跑了 %s）" % ",".join(sorted(parts)))
        print("-" * 72)
        print("阶段 3 总结论：%s（%d/%d）  用时 %.1f s（%.1f min）"
              % ("PASS" if passed and criteria else ("PARTIAL" if not criteria else "FAIL"),
                 n_pass, len(criteria), elapsed, elapsed / 60.0))
        for w in warnings:
            print("⚠️ %s" % w)
        print("归档：%s" % json_path)
        print("日志：%s" % log_path)
        if figs:
            print("图：\n  " + "\n  ".join(figs))
    return 0 if passed else 1


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


if __name__ == "__main__":
    sys.exit(main())
