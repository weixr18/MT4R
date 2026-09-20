# -*- coding: utf-8 -*-
"""阶段 2 一键运行与归档（计划 §5.2.5 判据表 + `res/2r_stage2/` 产出的唯一入口）。

在**同一进程内**依次跑四个算法的阶段 2 检查（不用 `subprocess`：① 避免管道在受限沙箱下被拦；
② 各测试模块的 `run()` 已返回结构化结论，直接合并即可）：

| 顺序 | 模块 | 算法 |
|---|---|---|
| 1 | `ctrl_js/test_js_pd.py` | JS 重力补偿 PD（`algo:robctrl_js_pd`） |
| 2 | `ctrl_js/test_js_invdyn.py` | JS 逆动力学 PD（`algo:robctrl_js_invdyn`） |
| 3 | `ctrl_os/test_os_pd.py` | OS 重力补偿 PD（`algo:robctrl_os_pd`） |
| 4 | `ctrl_os/test_os_invdyn.py` | OS 逆动力学 PD（`algo:robctrl_os_invdyn`） |

**汇总口径**：计划 §5.2.5 的四条判据都是「4 个算法在 2R 上」的集体判据，
故每条判据的 `passed = 四个算法全部通过`，`measured` 逐算法列出。

产出（见 `res/README.md` 的归档契约）：

- `res/2r_stage2/result.json` —— 机读结果：环境/版本、四条判据的阈值与实测、四个算法的完整明细
  （增益、参数、指标表、跟踪冒烟、二阶诊断、降采样轨迹）、两处**判据修订记录**；
- `res/2r_stage2/run.log` —— 本次运行的完整 stdout；
- `res/2r_stage2/figs/*.png` —— 首批图（误差曲线、控制力矩、2R 相图、K_P 扫描）。

> ★ **硬规矩（2026-09-20 与阶段 3 对齐）：先写 `result.json`、再出图，出图整体包在 `try` 里。**
> 归档是主要产出，绝不能排在装饰性步骤之后（阶段 3 曾因出图处抛异常白跑 63.7 min，
> 见日志尝试 10）；出图失败只记 `figure_error`，**绝不影响归档**。

用法（`py311-gym`，任意 cwd；完整跑约 25–30 min，建议后台执行）：

    python Robot-Control/test/ctrl_common/run_stage2.py
    python Robot-Control/test/ctrl_common/run_stage2.py --quick     # 冒烟（约 1 min）

退出码：四条判据全 PASS 为 0，否则 1。
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

import test_js_invdyn                                               # noqa: E402
import test_js_pd                                                   # noqa: E402
import test_os_invdyn                                               # noqa: E402
import test_os_pd                                                   # noqa: E402
import verify_lib as vl                                             # noqa: E402

MODULES = [test_js_pd, test_js_invdyn, test_os_pd, test_os_invdyn]

# (判据号, 判据名, 各算法 checks 里的键)
JUDGE = [
    ("1", "4 个算法在 2R 上跑通（无 NaN/Inf、无异常退出）", "no_nan"),
    ("2", "收敛性：固定种子 20 组初值全部收敛到 ‖q̃‖_∞ < 1e-3 rad", "convergence"),
    ("3", "力矩量级合理：‖u‖_∞ 在用例力矩上限内（不饱和）", "torque_limit"),
    ("4", "端到端自洽：稳态误差随 K_P 增大而下降（**已按实测修订为 4a/4b**）", "kp_monotone"),
]

CRITERION_REVISIONS = [
    {
        "criterion": "判据 4 端到端自洽",
        "original": "调节用例的稳态误差随 K_P 增大而单调下降",
        "problem": "原文隐含「稳态误差有限且 ∝ K_P⁻¹」。但在本阶段的**标称用例**下（plant 与控制器"
                   "模型同源、F_f = 0、无外力），重力补偿 PD 与逆动力学控制的平衡条件都是 K_Pq̃ = 0，"
                   "即稳态误差**理论上恒为 0**、与 K_P 无关——原文口径下该判据退化为「无趋势可比」，"
                   "单调性既不能证伪也不能证实。",
        "revised": "拆成两条，阈值一个字未动：4a 标称无扰动下实测稳态误差须 <1e-9（即**证实**"
                   "「恒为 0」这一理论预测，而不是假装测到了趋势）；4b 施加常值未补偿扰动力矩 τ_e 后"
                   "扫描 K_P，要求稳态误差**单调下降**且 log-log 斜率 ∈[-1.30,-0.70]（理论 −1）。"
                   "4b 才真正检验「端到端自洽 + 书中 K_P⁻¹ 规律」。",
        "note": "4a 的实测值在 1e-12~1e-16 量级，是**正向证据**（说明重力补偿与逆动力学抵消是精确的），"
                "不是判据失效；报告须同时给出 4a 与 4b。",
    },
    {
        "criterion": "2R 用例的「控制用任务空间」维数",
        "original": "平面 2R 的任务空间是 3 维 [x, y, φ]（计划 §5.0.1 的说明）",
        "problem": "`robot_fk_2r` 确实输出 3 维，其分析雅可比 J_a 是 3×2 的**浸入**（满列秩但非方阵）。"
                   "书里 OS 重力补偿 PD 只用 J_aᵀ（3×2 也可用），但 **OS 逆动力学算法框写的是 J_a⁻¹**"
                   "——非方阵没有逆；而且 2 自由度手臂本来也无法独立指令 3 个任务坐标。",
        "revised": "OS 用例的任务空间取**位置子任务 [x, y]（2×2 方阵）**，JS 用例仍在关节空间；"
                   "`(x, y, φ) ↔ q` 是双射、φ 行被丢弃不损失信息，故位置子任务与全任务本质同一。"
                   "这样 4 个算法都能**严格按书中公式**落地（`J_a⁻¹` 是真逆）。",
        "note": "替代方案（保留 3 维任务 + 用 `J_a⁺` 最小二乘左逆）会让 OS 逆动力学的误差动态不再严格"
                "线性，反而损害验证；故不采用。此项属「用例设计修订」，**不是书的错误**。",
    },
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


def _clean(res):
    """去掉 `_raw`（含 numpy 数组与 `RobotCase` 对象，不可 JSON 化）。"""
    return {k: v for k, v in res.items() if k != "_raw"}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
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
    out_dir = os.path.join(_RC, "res", "2r_stage2")
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    json_path = os.path.join(out_dir, "result.json")

    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8", newline="") as log_f, \
            contextlib.redirect_stdout(_Tee(sys.stdout, log_f)):
        print("#" * 72)
        print("# 阶段 2：仿真层 + 2R 用例跑通（一键运行 + 归档）%s"
              % ("  [--quick 冒烟模式]" if quick else ""))
        print("# 时间：%s" % datetime.datetime.now().isoformat(timespec="seconds"))
        print("# 环境：%s" % sys.executable)
        print("# 产出：%s" % out_dir)
        print("#" * 72)

        results = []
        for mod in MODULES:
            print("\n" + "#" * 72)
            print("# >>> %s（%s）" % (mod.__name__, vl.CTRLS[mod.KIND]["name"]))
            print("#" * 72)
            t_mod = time.perf_counter()
            res = mod.run(quick=quick, workers=workers)
            res["wall_time_s"] = time.perf_counter() - t_mod
            results.append(res)

        # ---- 汇总成计划 §5.2.5 的判据表（四条都是「4 个算法」的集体判据）----
        criteria = []
        for cid, name, key in JUDGE:
            per = {r["kind"]: r["checks"][key] for r in results}
            criteria.append({
                "id": cid, "name": name, "source_test": key,
                "threshold": next(iter(per.values()))["threshold"],
                "measured": {k: v["measured"] for k, v in per.items()},
                "passed_per_algorithm": {k: bool(v["passed"]) for k, v in per.items()},
                "passed": bool(all(v["passed"] for v in per.values())),
            })
        n_pass = sum(1 for c in criteria if c["passed"])
        if quick:
            passed = all(r["checks"]["no_nan"]["passed"] for r in results)
            print("\n⚠️ 冒烟模式（--quick）：判据 2/3/4 的时长与增益点被压缩，其 PASS/FAIL 无意义；"
                  "退出码只依据「判据 1 跑通」。")
        else:
            passed = n_pass == len(criteria)

        elapsed = time.perf_counter() - t0

        summary = {
            "stage": 2,
            "name": "阶段 2 仿真层 + 2R 用例跑通",
            "case": "2R",
            "plan": "1-MN4R/docs/robot-control-verify/part4-motion-control-plan.md §5.2.5",
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "wall_time_s": elapsed,
            "quick_mode": bool(quick),
            "workers": workers,
            "env": _env_info(),
            "seed": vl.SEED,
            "experiment": {
                "n_ic": vl.N_IC, "T_reg_s": vl.T_REG, "dt_s": vl.DT, "h_c_s": vl.HC,
                "tol_convergence_rad": vl.TOL_CONV,
                "ic_box": {"dq_max_rad": vl.DQ_MAX, "dqdot_max_rad_s": vl.DQDOT_MAX},
                "tol_steady_zero": vl.TOL_STEADY_ZERO,
                "T_kp_s": vl.T_KP, "tau_e_sweep_Nm": [float(v) for v in vl.TAU_E_SWEEP],
                "T_track_s": vl.T_TRACK, "ell_m": 0.5,
                "reg_q_d_2R": results[0]["params"]["q_d"],
                "reg_x_d_2R": results[0]["params"]["x_d"],
                "controllers": {k: {"name": v["name"], "wn": v["wn"], "xi": v["xi"],
                                    "gain_space": v["gain_dim"], "kp_wn_sweep": list(v["kp_wn"])}
                                for k, v in vl.CTRLS.items()},
            },
            "criteria": criteria,
            "passed_count": n_pass,
            "total_count": len(criteria),
            "passed": bool(passed),
            "criterion_revisions": CRITERION_REVISIONS,
            "algorithms": [_clean(r) for r in results],
            "figures": [],
            "figure_error": None,
            "notes": [
                "被控对象与控制器模型在本阶段**同源**（标称用例）：这只能证明「实现与书中公式自洽」，"
                "**不能**排除「书的模型本身错了」——后者由阶段 1（Pinocchio 权威对拍，已 PASS）承担；"
                "模型误差敏感性由阶段 4 承担。",
                "2R 的 OS 任务空间取 2 维位置子任务 [x, y]（J_a 为 2×2）：见 criterion_revisions 第 2 条。",
                "初值盒为 q0 = q_d + U(-0.15,0.15)、q̇0 = U(-0.15,0.15)，固定种子 %d，%d 组。"
                % (vl.SEED, vl.N_IC),
                "各控制器的 K_P/K_D 由二阶指标 (ω_n, ξ) 参数化（书中的增益选取式），"
                "**逐算法记录在 algorithms[*].gains**；书中未给定具体取值（计划 §八「K_P, K_D 具体取值书中未给定」）。",
                "K_P 扫描的 ω_n 取值逐算法不同（见 experiment.controllers[*].kp_wn_sweep）："
                "JS/OS 的 K_P 只是不同空间下的刚度，等效闭环带宽不同（见 code_ctrl_common.pd_gains 说明）。",
                "跟踪冒烟用例只验证「前馈链路跑通（q̈_d/ẍ_d 与 J̇_a 不炸）」；"
                "JS/OS 分工边界与误差动态精确性是**阶段 3** 的判据。",
                "二阶诊断（ξ̂, ω̂_n）本阶段只作记录，2% 判据在阶段 3.1。",
                "阶段 3 前置发现（2R 冒烟时顺带实测，与阶段 2 结论无关）：6R 的 cond(B) ≈ 1.7e3、"
                "λ_min(B) ≈ 1.15e-3，PD 增益若直接取 K_P = ω_n²I、K_D = 2ξω_nI，最小惯量方向上的阻尼"
                "速率达 ~1.4e4 /s，**远超 dt = 1 ms 的 RK4 稳定域**（实测 50 ms 内 ‖u‖ 冲到 1e92）——"
                "属离散化失稳。解法：按参考惯量标定增益（`code_ctrl_common.pd_gains(..., Lam=B(q_d))`，"
                "OS 用 Λ = (J_aB⁻¹J_aᵀ)⁻¹），实测同样条件下 ‖u‖ 回到 ~47 N·m。阶段 3 采用该标定并记录。",
            ],
        }
        # ★ **硬规矩（2026-09-20 补上，沿用阶段 3 的教训）：先写 result.json、再出图，出图整体包 `try`。**
        #   归档是主要产出，绝不能排在装饰性步骤之后——阶段 3 曾因出图处抛异常而白跑 63.7 min
        #   （`run_attempt2_crash.log`），且会留下「新图 + 旧 result.json」这种自相矛盾的目录。
        with open(json_path, "w", encoding="utf-8", newline="") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=_np_default)
        try:
            figs = vl.make_figures(results, fig_dir, tag="2r")
            summary["figures"] = [os.path.relpath(p, out_dir).replace("\\", "/") for p in figs]
        except Exception as exc:                                        # noqa: BLE001
            figs = []
            summary["figure_error"] = "%s: %s" % (type(exc).__name__, exc)
            print("\n⚠️ 出图失败（**不影响已落盘的归档**）：%s" % summary["figure_error"])
            print("   出图链路可用 scripts/smoke_figures.py（阶段 3）单独排查。")
        # 出图结果回写（失败时只是多一个 `figure_error` 字段，归档本身始终是完整的）
        with open(json_path, "w", encoding="utf-8", newline="") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=_np_default)

        print("\n" + "=" * 72)
        print("阶段 2 判据表汇总（计划 §5.2.5，四条均为「4 个算法」的集体判据）")
        print("=" * 72)
        for c in criteria:
            print("  [%s] 判据 %s  %s" % ("PASS" if c["passed"] else "FAIL", c["id"], c["name"]))
            print("         阈值：%s" % c["threshold"])
            print("         逐算法：%s" % ", ".join(
                "%s=%s" % (k, "PASS" if v else "FAIL") for k, v in c["passed_per_algorithm"].items()))
        print("-" * 72)
        print("阶段 2 总结论：%s（%d/%d）  用时 %.1f s（%.1f min）"
              % ("PASS" if passed else "FAIL", n_pass, len(criteria), elapsed, elapsed / 60.0))
        print("归档：%s" % json_path)
        print("日志：%s" % log_path)
        print("图：\n  " + "\n  ".join(figs))
    return 0 if passed else 1


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


if __name__ == "__main__":
    sys.exit(main())
