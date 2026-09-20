# -*- coding: utf-8 -*-
"""由 `res/6r_stage4/result.json` **重建 `run.log`**（一次性修复脚本，只读归档 + 写日志）。

**为什么需要它**：2026-09-20 那次阶段 4 完整跑（64.6 min、判据 3/3 PASS）之后，
为修一处 mathtext（`$\\sqrt2$` 少了花括号）而跑的 `--figs-only` 辅助流程
**以 `"w"` 打开了 `run.log`**，把那次运行的唯一 stdout 留档清空了。
`run_stage4.py` 已修（`--figs-only` 改写独立的 `figs_only.log`），但主日志需要恢复。

**为什么能重建**：`result.json` 里存着每个 `print` 用到的全部字段
（逐 level 的 `q5/sigma_min/cond_J`、逐档残留、逐算法斜率、失稳阈值区间…），
故日志是归档的**确定性函数**——重建出来的数字与那次运行**逐位相同**。

⚠️ **两处如实声明**（文件头也写进了日志本身）：
1. **并行进度行的耗时数字无法恢复**（`run_specs` 只在那次运行时打印），重建版省略这些行；
2. 原日志噪声表的「（×nan vs clean）」是格式化格式串里的常量（不该出现），
   重建版改为打印**真实**的 `u_ac_amplification_vs_clean`。

用法（`py311-gym`，任意 cwd）：

    python Robot-Control/scripts/restore_stage4_log.py --dry-run   # 只打印，不写
    python Robot-Control/scripts/restore_stage4_log.py             # 写 res/6r_stage4/run.log
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RC = os.path.abspath(os.path.join(_HERE, ".."))
_DEFAULT_JSON = os.path.join(_RC, "res", "6r_stage4", "result.json")

KIND_ORDER = ("js_pd", "js_invdyn", "os_pd", "os_invdyn")
OS_KINDS = ("os_pd", "os_invdyn")


def _t(t):
    return "标称" if t is None else "%.1e" % t


def _num(v, fmt="%.3e", dash="n/a"):
    return dash if v is None else fmt % v


def build(summary):
    exp = summary["experiment"]
    L = []
    w = L.append
    sep = "#" * 72
    line = "=" * 72
    hyp = "-" * 72
    out_dir = os.path.join(_RC, "res", "6r_stage4")

    w(sep)
    w("# 阶段 4：鲁棒性与边界（一键运行 + 归档）")
    w("# 时间：%s" % summary["timestamp"])
    w("# 环境：%s" % summary["env"]["executable"])
    w("# 实验：%s" % ",".join(summary["parts_run"]))
    w("# 产出：%s" % out_dir)
    w(sep)
    w("")
    w("【本文件为重建版】由 `Robot-Control/scripts/restore_stage4_log.py` 从 `result.json` 生成：")
    w("      那次完整跑（%s，%.1f min）的 stdout 被随后一次 `--figs-only` 以 \"w\" 清空。"
      % (summary["timestamp"], summary["wall_time_s"] / 60.0))
    w("      ① 全部**判据与实测数字**逐位取自归档；")
    w("      ② 并行进度行的**耗时数字**无法恢复，重建版省略了这些行（不影响任何结论）；")
    w("      ③ 原日志噪声表的「（×nan vs clean）」是格式串里的常量，重建版改为打印真实的")
    w("         `u_ac_amplification_vs_clean`。")
    w("")

    # ---------------- 判据 2：奇异位形边界 ----------------
    sing = summary.get("singularity")
    if sing:
        w(sep)
        w("# >>> 判据 2：奇异位形边界（计划 §5.4.1）")
        w(sep)
        w("      判据2 [奇异扫描] 腕部奇异族（只动 q5）：")
        for lv in sing["levels"]:
            w("        目标 σ_min=%s → q5=%+.8f，实测 σ_min=%.6e  cond(J_a)=%.3e"
              "；所需关节运动 in_range=%.2e / out_of_range=%.2e rad"
              % (_t(lv["target_sigma_min"]), lv["q5"], lv["sigma_min"], lv["cond_J"],
                 lv["pred_joint_motion_in_range_rad"], lv["pred_joint_motion_out_of_range_rad"]))
        for kind in OS_KINDS:
            fp = sing["failure_point"][kind]
            ir = sing["in_range_failure_point"][kind]
            dc = sing["direction_contrast"][kind]
            iv = fp["sigma_min_threshold_interval"]
            w("        失效点 [%s / out_of_range 退化方向]：首个失效 σ_min=%s（%s）；"
              "最后一个通过 σ_min=%s；阈值区间=%s；共 %d/%d 个 level 失效"
              "（指令爆掉 %d、误差不衰减 %d、穿越奇异面 %d）"
              % (kind, _num(fp["first_failure_sigma_min"]), fp["first_failure_reason"] or "-",
                 _num(fp["last_passing_sigma_min"]),
                 "n/a" if iv is None else "[%.3e, %.3e]" % tuple(iv),
                 fp["n_failed"], fp["n_levels"],
                 fp["n_failed_effort"], fp["n_failed_decay"], fp["n_failed_cross"]))
            w("            [%s / in_range 最良态方向]：首个失效 σ_min=%s（%s）；"
              "共 %d/%d 个 level 失效 ⇒ 方向对照：%s"
              % (kind, _num(ir["first_failure_sigma_min"]), ir["first_failure_reason"] or "-",
                 ir["n_failed"], ir["n_levels"],
                 {True: "退化方向**更早**失效 ✓", False: "退化方向未更早失效",
                  None: "两者都有失效点缺失，无法对照"}[dc["out_of_range_fails_first"]]))
            sc = sing["scaling"][kind]
            w("            放大律（用**反馈力矩** ‖u−g‖∞，不是总力矩 u）：斜率=%.3f（仅未失效点 %s）"
              "、关节加速度 %.3f、关节速度 %.3f（理论 1）"
              % (sc["loglog_slope_u_eff_vs_inv_sigma"],
                 _num(sc["loglog_slope_u_eff_only_passing"], "%.3f"),
                 sc["loglog_slope_qddot_vs_inv_sigma"], sc["loglog_slope_qdot_vs_inv_sigma"]))
        nr = sing["n_runs_per_scenario"]
        w("        场景对照：in_range 失效 %d/%d、out_of_range 失效 %d/%d、JS 对照组失效 %d/%d"
          "（同一批近奇异位形）"
          % (sing["in_range_failures"], nr["in_range"], sing["out_of_range_failures"],
             nr["out_of_range"],
             sum(v["n_failed"] for v in sing["js_control_group"].values()), nr["js_hold"]))
        dls = sing["dls_compare"]
        w("        DLS 对照（λ=%.3f，**超出书本范围的建议**）：%s"
          % (dls["lam"], "；".join("σ=%.1e→‖u−g‖∞=%.2e%s"
                                   % (r["sigma_min"], r["u_eff_inf"], "✗" if r["failure"] else "")
                                   for r in dls["rows"])))
        dr = sing["dt_refine"]
        rows = sorted(dr["rows"], key=lambda r: -r["dt_s"])
        if len(rows) >= 2:
            w("        步长归因（σ_min=%.3e，看 **u−g**）：dt=%.0e → %.3e；dt=%.0e → %.3e；比值 %.3f"
              % (dr["sigma_min"], rows[0]["dt_s"], rows[0]["u_eff_inf"],
                 rows[1]["dt_s"], rows[1]["u_eff_inf"],
                 dr["u_eff_inf_ratio_dt_main_over_ref"]))
        w("")

    # ---------------- 判据 1：模型失配 ----------------
    mm = summary.get("model_mismatch")
    if mm:
        w(sep)
        w("# >>> 判据 1：模型失配（plant ≠ 控制器模型）")
        w(sep)
        w("      判据1 [模型失配/mass] 稳态误差（理论 ∝|s−1|，闭式预测见 mismatch_prediction）：")
        for kind in KIND_ORDER:
            d = mm["mass"][kind]
            w("        %-10s 斜率=%+.3f（窗口 %s）  各档 err=%s"
              % (kind, d["loglog_slope_err_vs_abs_delta"],
                 tuple(exp["mismatch"]["mass_slope_tol"]), d["err_final_at_levels"]))
            w("                   实测/预测=%s"
              % {k: "%.2f" % v for k, v in
                 d["ratio_measured_over_predicted_at_levels"].items()
                 if v is not None})
        w("      判据1 [模型失配/inertia] 惯量失配**不改 g** ⇒ 无稳态偏移，只改暂态：")
        for key in sorted(mm["inertia_mode_prediction"]):
            p = mm["inertia_mode_prediction"][key]
            w("        ι=%s：λ_min(M)=%.4f ⇒ 最慢模态速率比预测 %.3f（λ_max=%.3f）"
              % (key, p["lambda_min_M"], p["predicted_rate_ratio_vs_nominal"], p["lambda_max_M"]))
        for kind in KIND_ORDER:
            pl = mm["inertia"][kind]["per_level"]
            w("        %-10s 实测/预测的衰减率比：%s"
              % (kind, {k: _num(v["ratio_of_ratios"], "%.2f") for k, v in sorted(pl.items())}))
        for key in sorted(mm["inertia_long"]):
            r = mm["inertia_long"][key]
            w("        长时段对照 [%s] T=%.1f s：末值=%.3e rad，衰减率=%.2f /s（>0 ⇒ 继续收敛）"
              % (key, r["T_s"], r["err_final"], r["decay"]["rate_per_s"]))
        w("")

    # ---------------- 判据 1：摩擦 ----------------
    fr = summary.get("friction")
    if fr:
        w(sep)
        w("# >>> 判据 1：摩擦未补偿（粘性 + 库仑）")
        w(sep)
        res = fr["resolution"]
        w("      摩擦数值分辨率：硬性数 dt·(F_c,max/ε)/λ_min(B) = %.2f（须 ≤ %.3f）→ %s"
          % (res["stiffness_number_h_over_tau"], res["rk4_limit"],
             "分辨" if res["resolved"] else "未分辨"))
        for kind in KIND_ORDER:
            d = fr["coulomb"][kind]
            w("      判据1 [摩擦 %s] 残留=%s（log-log 斜率 %s ← **不作判据**；单调=%s；"
              "F_c>0 档 / 零摩擦基线 ≥ %.1f×）"
              % (kind, d["err_at_levels"], _num(d["loglog_slope_err_vs_coulomb"], "%.3f"),
                 d["monotone_in_coulomb"], d["min_ratio_measured_over_zero_friction"]))
            ub = d["upper_bound_at_levels"]
            w("            静平衡上界（K_P_eff⁻¹ 最坏盒）：%s ⇒ 实测/上界 %s"
              % ({k: _num(v) for k, v in ub.items()},
                 "~".join("%.4f" % x for x in sorted(d["ratio_measured_over_upper_bound_range"]))))
        base = fr["zero_friction_baseline"]
        w("        守卫（真零摩擦基线的误差必须衰减到 ≤ %.0f%% **全程**峰值）：%s → %s"
          % (exp["friction"]["guard_decay_ratio"] * 100.0,
             {k: "%.4f" % base[k]["decay_ratio"] for k in KIND_ORDER},
             "ok" if fr["check"]["measured"]["guard_ok"] else "异常！控制器可能没读当前状态"))
        w("            （对照：末段窗口口径会给出 %s —— 峰值落在已衰减段上，比值被放大）"
          % {k: "%.4f" % base[k]["decay_ratio_tail_window"] for k in KIND_ORDER})
        for cn in sorted(fr["shapes"]):
            for kind in KIND_ORDER:
                d = fr["shapes"][cn][kind]
                w("        形态[%s] %s：残留误差=%.3e  末段 |q̇|max=%.3e rad/s  末段误差峰峰值=%.3e"
                  % (cn, kind, d["err_steady"], d["qdot_tail_max_rad_s"], d["err_tail_ptp_rad"]))
        w("")

    # ---------------- 判据 1：测量噪声 ----------------
    nz = summary.get("measurement_noise")
    if nz:
        w(sep)
        w("# >>> 判据 1：测量噪声 + 低通滤波")
        w(sep)
        for kind in KIND_ORDER:
            w("      判据1 [测量噪声/%s] 力矩 AC-RMS（末段，初值取平衡点 ⇒ 纯噪声激发）：" % kind)
            for cn in ("clean", "q_only", "qd_only", "q_and_qd", "q_diff", "q_diff_lpf"):
                r = next((x for x in nz["rows"] if x["noise_case"] == cn and x["kind"] == kind), None)
                if r is None:
                    continue
                amp = r["u_ac_amplification_vs_clean"]
                w("        %-12s 实测=%.3e（×%s vs clean） 预测=%s       比值=%s    误差RMS=%.3e"
                  % (cn, r["u_ac_rms_max"],
                     "?" if amp is None else "%.0f" % amp,
                     _num(r["pred_u_ac_rms_max"]),
                     _num(r["ratio_measured_over_predicted"], "%.2f"), r["err_rms"]))
        w("        敏感度（在 q_d、q̇=0）：%s"
          % "；".join("%s |S_q|=%.1f |S_qd|=%.1f"
                      % (k, nz["sensitivity"][k]["S_q_abs_max"], nz["sensitivity"][k]["S_qd_abs_max"])
                      for k in KIND_ORDER))
        w("")

    # ---------------- 判据 1：控制周期 ----------------
    pd_ = summary.get("control_period")
    if pd_:
        w(sep)
        w("# >>> 判据 1：控制周期 h_c 敏感性")
        w(sep)
        for kind in KIND_ORDER:
            pk = pd_["per_kind"][kind]
            rs = sorted(pk["rows"], key=lambda r: r["h_c_s"])
            w("      判据1 [控制周期/%s] 运动过程峰值误差=%s（斜率 %.3f，5ms/0.5ms=%.1f×）"
              % (kind, ["%.2e" % r["err_peak"] for r in rs],
                 pk["loglog_slope_err_peak_vs_hc"], pk["err_peak_ratio_5ms_over_0.5ms"]))
        w("")

    # ---------------- 判据 3：离散化失稳 ----------------
    inst = summary.get("discretization_instability")
    if inst:
        w(sep)
        w("# >>> 判据 3：离散化失稳阈值 ω*(dt)")
        w(sep)
        for kind in KIND_ORDER:
            for r in sorted(inst["by_kind"][kind]["rows"], key=lambda r: -r["dt_s"]):
                if r.get("wn_star") is None:
                    w("      判据3 [失稳/%s dt=%.0e] 到 ω_n=%.0f 仍稳定"
                      % (kind, r["dt_s"], inst["wn_max"]))
                else:
                    lo, hi = r["threshold_bracket"]
                    w("      判据3 [失稳/%s dt=%.0e] ω*∈[%.1f, %.1f] rad/s（几何均值 %.1f）"
                      " ⇒ hω*≈%.3f" % (kind, r["dt_s"], lo, hi, r["wn_star"], r["hw_star_est"]))
        w("      ── 机理对照（判据 3 的关键区分）：同一批增益、plant 换成**线性冻结模型**")
        for kind in KIND_ORDER:
            v = inst["frozen_linear_mechanism_check"][kind]
            w("        机理对照 [%s] 线性冻结模型 dt=%.0e：最大稳定 ω=%s、最小失稳 ω=%s ⇒ hω* ∈ %s"
              % (kind, v["dt_s"],
                 "nan" if v["largest_stable_wn"] is None else "%.0f" % v["largest_stable_wn"],
                 "nan" if v["smallest_unstable_wn"] is None else "%.0f" % v["smallest_unstable_wn"],
                 "n/a" if v["hw_star_bracket"] is None
                 else "[%.2f, %.2f]" % tuple(v["hw_star_bracket"])))
        for kind in KIND_ORDER:
            c = inst["by_kind"][kind]
            if not c.get("hw_const_rel_diff"):
                continue
            w("        1/dt 标定 [%s]：ω*(%.0e)=%.1f、ω*(%.0e)=%.1f → 比值 %.2f（理论 %.2f）；"
              "hω* = %.3f vs %.3f（相对差 %.1f%%；RK4 理论 hω* = %.3f）"
              % (kind, c["dt_max_s"], c["wn_star_dt_max"], c["dt_min_s"], c["wn_star_dt_min"],
                 c["ratio_wn_star"], c["ratio_expected_if_hw_const"],
                 c["hw_star_dt_max"], c["hw_star_dt_min"], c["hw_const_rel_diff"] * 100.0,
                 c["rk4_theory_hw_star"]))
        w("        两种理论参照：连续反馈 hω*=%.3f、**零阶保持采样闭环 hω*=%.3f**"
          "（后者与本仿真层同口径，是判据 3 的正确参照）"
          % (inst["rk4_theory_hw_star"], inst["zoh_theory_hw_star"]))
        w("")

    # ---------------- 报告项：每周期耗时 ----------------
    cost = summary.get("compute_cost")
    if cost:
        w(sep)
        w("# >>> 报告项：每周期计算耗时实测")
        w(sep)
        w("      控制器单周期耗时 (ms)：%s"
          % ", ".join("%s=%.3f" % (k, cost["items"][k]["ctrl_ms_per_cycle"])
                      for k in KIND_ORDER))
        w("      被控对象一步 (RK4 ×4) ≈ %.3f ms；fk+jac ≈ %.3f ms"
          % (cost["plant_rk4_step_ms_est"], cost["fk_plus_jac_ms"]))
        w("")

    # ---------------- 判据表汇总 ----------------
    w(line)
    w("阶段 4 判据表汇总（计划 §5.4.3）")
    w(line)
    for c in summary["criteria"]:
        w("  [%s] 判据 %s  %s" % ("PASS" if c["passed"] else "FAIL", c["id"], c["name"]))
        w("         阈值：%s" % c["threshold"])
        w("         逐对象：%s" % ", ".join("%s=%s" % (k, "PASS" if v else "FAIL")
                                           for k, v in c["passed_per_scope"].items()))
    w(hyp)
    complete = all(c["threshold"] is not None for c in summary["criteria"])
    w("阶段 4 总结论：%s（%d/%d）  用时 %.1f s（%.1f min）"
      % ("PASS" if (summary["passed"] and complete) else ("PARTIAL" if not complete else "FAIL"),
         summary["passed_count"], summary["total_count"],
         summary["wall_time_s"], summary["wall_time_s"] / 60.0))
    for x in summary.get("warnings", []):
        w("⚠️ %s" % x)
    if not summary.get("figure_error") and (summary.get("figures") or []):
        w("【运行后处置】上面那条出图异常**已修复并重建**：`stage4_lib` 的 mathtext `$\\sqrt2$` 补上花括号、")
        w("            并把图上缺字形的「⚠️ / ε」换成 ASCII 后，用 `run_stage4.py --figs-only`")
        w("            （由本归档重建，**未重跑仿真**）生成全部 %d 张图；`figure_error` 已置空。"
          % len(summary["figures"]))
    w("归档：%s" % os.path.join(out_dir, "result.json"))
    w("日志：%s" % os.path.join(out_dir, "run.log"))
    figs = summary.get("figures") or []
    if figs:
        w("图：")
        for f in figs:
            w("  %s" % os.path.join(out_dir, f.replace("/", os.sep)))
    w("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="由 result.json 重建阶段 4 的 run.log")
    ap.add_argument("--json", default=_DEFAULT_JSON)
    ap.add_argument("--out", default=None, help="默认写同目录的 run.log")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    a = ap.parse_args(argv)
    with open(a.json, "r", encoding="utf-8") as f:
        summary = json.load(f)
    text = build(summary)
    if a.dry_run:
        sys.stdout.write(text)
        return 0
    out = a.out or os.path.join(os.path.dirname(a.json), "run.log")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print("已重建日志：%s（%d 字节，%d 行）" % (out, len(text.encode("utf-8")),
                                              text.count("\n") + 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
