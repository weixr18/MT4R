# -*- coding: utf-8 -*-
"""阶段 4 一键运行与归档：**鲁棒性与边界**（计划 §5.4 的三条判据 + §6.1 的图）。

| 顺序 | 内容 | 对应计划 | 判据 |
|---|---|---|---|
| 1 | **奇异位形边界**（腕部奇异族，σ_min 扫到 ≈0；含 JS 对照组与 DLS 对照） | §5.4.1 | **2** |
| 2 | **模型失配**（质量/惯量 ±10%/±30%，plant ≠ 控制器） | §5.4.2 | **1** |
| 3 | **摩擦未补偿**（粘性+库仑，控制器 `F_f = 0`） | §5.4.2 | **1** |
| 4 | **测量噪声**（含「速度由位置差分」与低通滤波） | §5.4.2 | **1** |
| 5 | **控制周期**（`h_c ∈ {0.5, 1, 5} ms`） | §5.4.2 | **1** |
| 6 | **离散化失稳阈值** `ω*(dt)`（两档 dt，检验 `hω* ≈ const`） | §5.4.2/§5.4.3 | **3** |
| 7 | 每周期计算耗时实测 | §5.4.2 | 报告项 |

**判据 1** 是四项鲁棒性实验的集体判据（每项都要给出「误差随扰动量的量级关系」）；
判据 2 只由奇异边界承担；判据 3 只由离散化失稳阈值承担。逐条定义见 `stage4_lib`。

产出（见 `res/README.md` 的归档契约）：

- `res/6r_stage4/result.json` —— 机读结果：环境/版本、3 条判据的阈值与实测、全部实验明细、
  口径说明与局限；
- `res/6r_stage4/run.log` —— 本次运行的完整 stdout；
- `res/6r_stage4/figs/*.png` —— 奇异度-控制量/误差、失配-误差、摩擦死区、噪声抖动、
  周期敏感性、失稳阈值 vs dt。

用法（`py311-gym`，任意 cwd；完整跑约 55–75 min，**务必后台执行**）：

    python Robot-Control/test/ctrl_common/run_stage4.py
    python Robot-Control/test/ctrl_common/run_stage4.py --quick         # 冒烟（只看链路跑通）
    python Robot-Control/test/ctrl_common/run_stage4.py --only sing,noise
    python Robot-Control/test/ctrl_common/run_stage4.py --workers 6
    python Robot-Control/test/ctrl_common/run_stage4.py --figs-only     # 只由归档重建图集（秒级，不跑仿真）

⚠️ **改动过任何出图代码后，先跑 `--figs-only`**（秒级）而不是重跑完整实验（64.6 min）——
它由已落盘的 `result.json` 重建全部图并回写归档，前提见 `_figs_only` 的文档。

退出码：3 条判据全 PASS 为 0，否则 1（`--quick` 只依据「无异常」）。

⚠️ **沿用阶段 3 的硬规矩**（日志尝试 10 的教训）：**先写 `result.json`、再出图**，
出图整体包在 `try` 里——归档是主要产出，绝不能排在装饰性步骤之后。
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
           os.path.join(_RC, "test", "ctrl_os")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage4_lib as s4                                             # noqa: E402
import test_os_singular as tsing                                    # noqa: E402
import test_robust as trb                                           # noqa: E402
import verify_lib as vl                                             # noqa: E402

ALL_PARTS = ("sing", "mismatch", "friction", "noise", "period", "instab", "cost")

JUDGE = [
    ("1", "每项给出「误差随扰动量的量级关系」并作图，趋势与书中定性论断一致",
     ["模型失配（质量/惯量）", "摩擦未补偿", "测量噪声", "控制周期"]),
    ("2", "奇异边界：给出明确的失效点（σ_min 阈值）与失稳表现",
     ["os_pd", "os_invdyn"]),
    ("3", "离散化失稳与连续律正确性明确区分并分别报告",
     ["四个算法 × 两档 dt"]),
]


class _Tee:
    """把 stdout 同时写到控制台与日志文件。"""

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
    """递归去掉下划线开头的键（`_raw`/`_recs`，含 numpy 数组与 RobotCase），并转 numpy 标量。"""
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


def _write_json(path, summary):
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_np_default)


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


def _criteria(sing, mm, fr, nz, pd, inst):
    """把各实验的 checks 汇总成计划 §5.4.3 的 3 条判据。"""
    per = {
        "1": {},
        "2": {"singularity": sing["check"]} if sing else {},
        "3": {"discretization": inst["check"]} if inst else {},
    }
    if mm:
        per["1"]["model_mismatch"] = mm["check"]
    if fr:
        per["1"]["friction"] = fr["check"]
    if nz:
        per["1"]["noise"] = nz["check"]
    if pd:
        per["1"]["control_period"] = pd["check"]
    out = []
    for cid, name, who in JUDGE:
        d = per[cid]
        if not d:
            out.append({"id": cid, "name": name, "checked_on": who, "threshold": None,
                        "measured": None, "passed_per_scope": {}, "passed": False,
                        "note": "本次未跑承担该判据的实验"})
            continue
        out.append({
            "id": cid, "name": name, "checked_on": who,
            "threshold": next(iter(d.values()))["threshold"],
            "measured": {k: v.get("measured") for k, v in d.items()},
            "passed_per_scope": {k: bool(v["passed"]) for k, v in d.items()},
            "passed": bool(all(v["passed"] for v in d.values())),
        })
    return out


def _figs_only(json_path, fig_dir, out_dir):
    """`--figs-only`：**只由已落盘的 `result.json` 重建图集**，不跑任何仿真。

    **为什么需要它**（2026-09-20 完整跑的实际教训）：完整跑用了 64.6 min，3 条判据全 PASS，
    但出图在第 4 张（`noise`）处抛 `ParseSyntaxException`（mathtext 的 `\\sqrt2` 少了花括号），
    于是 8 张图只画出前 6 张。若没有这条路，修一个字符的锅就要**再付 64.6 min**。
    ⚠️ **能这么做的前提是 `make_figures` 只读「已剥离下划线键」的数据**
    （`rows` / `by_kind` / `mass` / `inertia` / …，全部在归档里），
    故归档即为出图的完整数据源——这与阶段 3 不同：阶段 3 的 `make_figures` 依赖 `extras` 里的
    `_raw` 原始曲线，那样就必须重跑（见 `res/6r_stage3/run_attempt2_crash.log` 的教训）。

    会回写 `result.json` 的 `figures` 与 `figure_error`（并追加一条 `post_run_refreshes` 记录），
    使归档与磁盘上的图一致。
    """
    if not os.path.exists(json_path):
        print("⚠️ 找不到归档 %s——请先完整跑一次 `run_stage4.py`" % json_path)
        return 2
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    need = ("singularity", "model_mismatch", "friction", "measurement_noise",
            "control_period", "discretization_instability")
    missing = [k for k in need if not summary.get(k)]
    if missing:
        print("⚠️ 归档缺少实验块 %s——无法只由归档重建全图（该归档可能来自 `--quick`/`--only`）"
              % missing)
        return 2
    before = list(summary.get("figures") or [])
    figs = s4.make_figures(summary["singularity"], summary["model_mismatch"],
                           summary["friction"], summary["measurement_noise"],
                           summary["control_period"], summary["discretization_instability"],
                           summary.get("compute_cost"), fig_dir, tag="6r")
    summary["figures"] = [os.path.relpath(p, out_dir).replace("\\", "/") for p in figs]
    summary["figure_error"] = None
    summary.setdefault("post_run_refreshes", []).append({
        "what": "figures",
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "reason": "--figs-only：修掉 mathtext `\\sqrt2` 后由归档重建图集（未跑仿真）",
        "figures_before": before, "figures_after": summary["figures"],
        "wall_time_note": "本次为纯出图（秒级），仿真结果仍出自 %s 的那次运行"
                          % summary.get("timestamp"),
    })
    _write_json(json_path, summary)
    print("\n" + "=" * 72)
    print("--figs-only：由归档重建 %d 张图（**未跑仿真**）" % len(figs))
    for p in figs:
        print("  %s" % p)
    print("已回写：%s（figures / figure_error / post_run_refreshes）" % json_path)
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    figs_only = "--figs-only" in argv
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

    out_dir = os.path.join(_RC, "res", "6r_stage4")
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    json_path = os.path.join(out_dir, "result.json")

    # ⚠️ `--figs-only` **绝不能覆盖 `run.log`**（2026-09-20 实际踩过：主日志是那次 64.6 min
    # 完整跑的唯一 stdout 留档，被辅助运行以 `"w"` 打开后整份清空）。
    # 故它写**独立**日志，主日志只由「真正跑了仿真的那次」写。
    if figs_only:
        aux_path = os.path.join(out_dir, "figs_only.log")
        with open(aux_path, "w", encoding="utf-8", newline="") as log_f, \
                contextlib.redirect_stdout(_Tee(sys.stdout, log_f)):
            print("# 阶段 4：只由归档重建图集（--figs-only，未跑仿真）")
            print("# 时间：%s" % datetime.datetime.now().isoformat(timespec="seconds"))
            print("# 主日志 run.log **保持不动**（它属于上一次完整跑）")
            return _figs_only(json_path, fig_dir, out_dir)

    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8", newline="") as log_f, \
            contextlib.redirect_stdout(_Tee(sys.stdout, log_f)):
        print("#" * 72)
        print("# 阶段 4：鲁棒性与边界（一键运行 + 归档）%s"
              % ("  [--quick 冒烟模式]" if quick else ""))
        print("# 时间：%s" % datetime.datetime.now().isoformat(timespec="seconds"))
        print("# 环境：%s" % sys.executable)
        print("# 实验：%s" % ",".join(sorted(parts)))
        print("# 产出：%s" % out_dir)
        print("#" * 72)

        sing = mm = fr = nz = pd_ = inst = cost = None
        warnings = []

        if "sing" in parts:
            print("\n" + "#" * 72 + "\n# >>> 判据 2：奇异位形边界（计划 §5.4.1）\n" + "#" * 72)
            sing = tsing.run(quick=quick, workers=workers)
        if "mismatch" in parts:
            print("\n" + "#" * 72 + "\n# >>> 判据 1：模型失配（plant ≠ 控制器模型）\n" + "#" * 72)
            mm = trb.run("mismatch", quick=quick, workers=workers)
        if "friction" in parts:
            print("\n" + "#" * 72 + "\n# >>> 判据 1：摩擦未补偿（粘性 + 库仑）\n" + "#" * 72)
            fr = trb.run("friction", quick=quick, workers=workers)
        if "noise" in parts:
            print("\n" + "#" * 72 + "\n# >>> 判据 1：测量噪声 + 低通滤波\n" + "#" * 72)
            nz = trb.run("noise", quick=quick, workers=workers)
        if "period" in parts:
            print("\n" + "#" * 72 + "\n# >>> 判据 1：控制周期 h_c 敏感性\n" + "#" * 72)
            pd_ = trb.run("period", quick=quick, workers=workers)
        if "instab" in parts:
            print("\n" + "#" * 72 + "\n# >>> 判据 3：离散化失稳阈值 ω*(dt)\n" + "#" * 72)
            inst = trb.run("instab", quick=quick, workers=workers)
        if "cost" in parts:
            print("\n" + "#" * 72 + "\n# >>> 报告项：每周期计算耗时实测\n" + "#" * 72)
            cost = trb.run("cost", quick=quick)
            print("      控制器单周期耗时 (ms)：%s"
                  % ", ".join("%s=%.3f" % (k, v["ctrl_ms_per_cycle"])
                              for k, v in cost["items"].items()))
            print("      被控对象一步 (RK4 ×4) ≈ %.3f ms；fk+jac ≈ %.3f ms"
                  % (cost["plant_rk4_step_ms_est"], cost["fk_plus_jac_ms"]))

        criteria = _criteria(sing, mm, fr, nz, pd_, inst)
        complete = all(c["threshold"] is not None for c in criteria)
        if complete and not quick:
            n_pass = sum(1 for c in criteria if c["passed"])
            passed = n_pass == len(criteria)
        else:
            n_pass = sum(1 for c in criteria if c["passed"])
            passed = True
            warnings.append("冒烟/部分运行：判据表不完整或窗口被压缩，PASS/FAIL 无意义；"
                            "退出码只依据「无异常退出」。")

        elapsed = time.perf_counter() - t0
        summary = {
            "stage": 4,
            "name": "阶段 4：鲁棒性与边界",
            "case": "6R",
            "plan": "1-MN4R/docs/robot-control-verify/part4-motion-control-plan.md §5.4",
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "wall_time_s": elapsed,
            "quick_mode": bool(quick),
            "parts_run": sorted(parts),
            "workers": workers,
            "env": _env_info(),
            "seed": s4.SEED4,
            "experiment": {
                "case": s4.CASE4, "dt_s": s4.DT4, "dt_ref_s": s4.DT4_REF,
                "singular": {"targets": list(s4.SING_TARGETS), "amp": s4.SING_AMP,
                             "T_s": s4.T_SING,
                             # ⚠️ 主扫描用 `DT4_REF`（1 ms），不是 `DT4`（2 ms）——旧版此处写错，已更正
                             "dt_s": s4.DT4_REF,
                             "in_range_subset": list(s4.SING_IN_RANGE_SUBSET),
                             "js_subset": list(s4.SING_JS_SUBSET),
                             "tol_u_Nm": s4.SING_TOL_U,
                             "tol_qdot_rad_s": s4.SING_QDOT_FAIL,
                             "decay_tol_final_over_peak": s4.SING_DECAY_TOL,
                             "cross_q5_rad": s4.SING_CROSS_Q5,
                             "decay_norm": "误差末值 / **全程峰值**（旧口径「后半窗峰值」已废弃）",
                             "scenarios": {
                                 "out_of_range": "x_d = fk(q_t) + a·u_min，q0 = q_t（退化方向）",
                                 "in_range": "x_d = fk(q_t) + a·u_max，q0 = q_t（最良态方向；"
                                             "与 out_of_range 初值/幅度相同，只换方向）",
                                 "js_hold": "q_d = q_t，q0 = q_t + DQ4·v（JS 对照组；"
                                            "其初值扰动在深奇异处会跨过奇异面，但 JS 律不含 J_a，"
                                            "故穿越不计为 JS 的失效）",
                             },
                             "tol_err": s4.TOL_ERR_FAIL,
                             "dls_lambda": s4.DLS_LAMBDA},
                "mismatch": {"mass_levels": list(s4.MISMATCH_LEVELS),
                             "inertia_levels": list(s4.MISMATCH_INERTIA_LEVELS),
                             "families": dict(s4.MISMATCH_FAMILIES),
                             "T_mass_s": s4.T_MISMATCH,
                             "T_inertia_s": s4.T_MISMATCH_INERTIA,
                             "T_long_s": s4.T_MISMATCH_LONG,
                             "dq_rad": s4.DQ4,
                             "mass_slope_tol": list(s4.MISMATCH_MASS_SLOPE_TOL),
                             "pred_ratio_tol": list(s4.MISMATCH_PRED_RATIO_TOL),
                             "inertia_rate_ratio_tol": list(s4.INERTIA_RATE_RATIO_TOL)},
                "friction": {"coulomb_levels": list(s4.FRIC_COULOMB_LEVELS),
                             "viscous_floor": s4.FRIC_VISCOUS_SWEEP,
                             "cases": s4.FRIC_CASES, "eps_rad_s": s4.FRIC_EPS,
                             "dt_s": s4.FRIC_DT,
                             "resolution": s4.friction_resolution(
                                 s4.cm.make_case(s4.CASE4), max(s4.FRIC_COULOMB_LEVELS)),
                             "guard_decay_ratio": s4.GUARD_DECAY_RATIO,
                             "T_sweep_s": s4.T_FRICTION, "T_shape_s": s4.T_FRICTION_SHAPE},
                "noise": {"q_sigma_rad": s4.NOISE_Q_SIGMA, "qd_sigma_rad_s": s4.NOISE_QD_SIGMA,
                          "lpf_tau_s": s4.NOISE_LPF_TAU,
                          "cases": {k: v for k, v in s4.NOISE_CASES.items()},
                          # ⚠️ 噪声实验用 `DT4_REF`（1 ms，`stage4_lib.measurement_noise` 的默认值），
                          # 不是 `DT4`（2 ms）——旧版此处与 singular 同类写错，已更正（2026-09-20，阶段 5 期间）
                          "T_s": s4.T_NOISE, "dt_s": s4.DT4_REF,
                          "ic": "q0 = q_d（平衡点）"},
                "control_period": {"hc_list_s": list(s4.PERIOD_HC_LIST), "dt_s": s4.PERIOD_DT,
                                   "T_s": s4.T_PERIOD, "T_traj_s": s4.T_PERIOD_TRAJ,
                                   "note": "跟踪用例（调节用例的稳态 ZOH 残差恒为 0，指标会退化）"},
                "discretization": {"dt_list_s": list(s4.INSTAB_DT_LIST),
                                   "wn_max": s4.WN_INSTAB_MAX, "n_bisect": s4.N_BISECT_INSTAB,
                                   "T_s": s4.T_INSTAB, "growth_fail": s4.RATE_FAIL,
                                   "qdot_blowup": s4.QDOT_BLOWUP,
                                   "rk4_theory_hw_star": s4.rk4_stability_limit(1.0),
                                   "zoh_theory_hw_star": s4.zoh_stability_limit(1.0)},
            },
            "criteria": criteria,
            "passed_count": n_pass,
            "total_count": len(criteria),
            "passed": bool(passed),
            "singularity": _strip(sing) if sing else None,
            "model_mismatch": _strip(mm) if mm else None,
            "friction": _strip(fr) if fr else None,
            "measurement_noise": _strip(nz) if nz else None,
            "control_period": _strip(pd_) if pd_ else None,
            "discretization_instability": _strip(inst) if inst else None,
            "compute_cost": _strip(cost) if cost else None,
            "scope_notes": s4.SCOPE_NOTES,
            "warnings": warnings,
            "figures": [],
            "figure_error": None,
        }
        # ★★ **先落盘，再出图**（阶段 3 的教训，见 `res/6r_stage3/run_attempt2_crash.log`）：
        # 归档是主要产出，绝不能排在装饰性步骤之后。
        _write_json(json_path, summary)
        figs, fig_error = [], None
        if not quick and sing and mm and fr and nz and pd_ and inst and cost:
            try:
                figs = s4.make_figures(sing, mm, fr, nz, pd_, inst, cost, fig_dir, tag="6r")
            except Exception as exc:                                # noqa: BLE001
                fig_error = "%s: %s" % (type(exc).__name__, exc)
                warnings.append("出图失败（**结果归档已先写出，不受影响**）：%s" % fig_error)
                print("⚠️ 出图失败：%s" % fig_error)
                import traceback
                traceback.print_exc()
        else:
            warnings.append("冒烟/部分运行：跳过出图（图需要完整的全部实验）")
        summary["figures"] = [os.path.relpath(p, out_dir).replace("\\", "/") for p in figs]
        summary["figure_error"] = fig_error
        _write_json(json_path, summary)

        print("\n" + "=" * 72)
        print("阶段 4 判据表汇总（计划 §5.4.3）")
        print("=" * 72)
        for c in criteria:
            print("  [%s] 判据 %s  %s" % ("PASS" if c["passed"] else "FAIL", c["id"], c["name"]))
            print("         阈值：%s" % c["threshold"])
            print("         逐对象：%s" % ", ".join(
                "%s=%s" % (k, "PASS" if v else "FAIL") for k, v in c["passed_per_scope"].items()))
        print("-" * 72)
        print("阶段 4 总结论：%s（%d/%d）  用时 %.1f s（%.1f min）"
              % ("PASS" if passed and complete else ("PARTIAL" if not complete else "FAIL"),
                 n_pass, len(criteria), elapsed, elapsed / 60.0))
        for w in warnings:
            print("⚠️ %s" % w)
        print("归档：%s" % json_path)
        print("日志：%s" % log_path)
        if figs:
            print("图：\n  " + "\n  ".join(figs))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
