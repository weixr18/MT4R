# -*- coding: utf-8 -*-
"""把 `res/6r_stage3/result.json` 压成一份**人读摘要**（写日志与验证报告时直接引用，避免抄错数）。

用法（`py311-gym`，任意 cwd）：

    python Robot-Control/scripts/summarize_stage3.py                  # 打印到 stdout
    python Robot-Control/scripts/summarize_stage3.py --md out.md      # 另存 Markdown

只读归档、不跑仿真。数字一律取自 `result.json`，**不在这里做任何计算或加工**——
报告里的每个数字都必须能这样原样取出来。
"""
import argparse
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RC = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_JSON = os.path.join(_RC, "res", "6r_stage3", "result.json")


def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return ("%%.%de" % nd) % v if (v and (abs(v) < 1e-3 or abs(v) >= 1e4)) else ("%%.%df" % nd) % v
    return str(v)


def summarize(d):
    L = []
    add = L.append
    add("# 阶段 3 归档摘要（`res/6r_stage3/result.json`）\n")
    add("- 运行时间：%s；用时 %.1f s（%.1f min）；主步长 dt = %s s；判据 1 步长 dt = %s s"
        % (d.get("timestamp"), d.get("wall_time_s", float("nan")),
           d.get("wall_time_s", 0) / 60.0, d["experiment"]["dt_s"], d["experiment"]["dt_fit_s"]))
    add("- 结论：**%s（%d/%d）**；实验项 %s；figure_error = %s"
        % ("PASS" if d.get("passed") else "FAIL", d.get("passed_count"), d.get("total_count"),
           ",".join(d.get("parts_run", [])), d.get("figure_error")))
    add("- 环境：python %s / numpy %s / scipy %s / matplotlib %s"
        % (d["env"]["python"], d["env"]["numpy"], d["env"].get("scipy"), d["env"].get("matplotlib")))
    add("- 口径：种子 %s；%d 组初值；T_reg = %s s；cond(J_a) ≤ %s；ℓ = %s m"
        % (d.get("seed"), d["experiment"]["n_ic"], d["experiment"]["T_reg_s"],
           d["experiment"]["cond_J_premise"], d["experiment"]["ell_m"]))
    add("")
    add("## 判据表（计划 §5.3.7）\n")
    add("| # | 判据 | 承担 | 结果 |")
    add("|---|---|---|---|")
    for c in d.get("criteria", []):
        add("| %s | %s | %s | **%s** |"
            % (c["id"], c["name"], ", ".join(c["checked_on"]),
               "PASS" if c["passed"] else "FAIL"))
    add("")
    add("## 四个算法（判据 3a/3b + 指标表）\n")
    add("| 算法 | 3a 组数 | 3a 末态 max‖q̃‖∞ | 3b 边界样本 | 全 20 组收敛 | ‖u‖∞ | max cond(J_a)（均值） | 调节时间均值 |")
    add("|---|---|---|---|---|---|---|---|")
    for k, a in d.get("algorithms", {}).items():
        m = a["checks"]["convergence"]["measured"]
        t = a.get("table") or {}
        add("| %s | %d/%d | %s | %d | %d/%d | %s | %s | %s |"
            % (k, m["3a"]["n_ic"], m["n_ic"], _fmt(m["3a"]["worst_final_q_tilde_inf"]),
               m["n_premise_violated"], m["all_ics_unconditioned"]["n_converged"], m["n_ic"],
               _fmt(m["worst_u_inf"], 3),
               _fmt((t.get("max cond(J_a)") or {}).get("mean"), 2),
               _fmt((t.get("调节时间_2%") or {}).get("mean"), 3)))
    add("")
    for k, a in d.get("algorithms", {}).items():
        m = a["checks"]["convergence"]["measured"]
        b = m["3b_boundary"]
        if b["n_ic"]:
            add("- **3b 边界样本**（%s）：%s" % (k, json.dumps(b["rows"], ensure_ascii=False)))
    add("")
    add("## 判据 1：误差动态 (ξ̂, ω̂_n) 逐分量\n")
    for k, a in d.get("algorithms", {}).items():
        e = (a.get("extras") or {}).get("error_dynamics")
        if not e:
            continue
        cm_ = e["check"]["measured"]
        add("- **%s**（dt = %.0e s，T = %.1f s，设定 ξ = %.2f、ω_n = %.1f）：最差相对偏差 **%.4f%%** → %s"
            % (k, cm_["dt_s"], cm_["T_s"], e["xi_set"], e["wn_set"],
               100 * cm_["worst_rel_err"], "PASS" if e["check"]["passed"] else "FAIL"))
        for c in e["components"]:
            add("    - %s：ξ̂ = %.4f（%+.3f%%）、ω̂_n = %.4f（%+.3f%%），正峰 %d"
                % (c["component"], c["xi_hat"], 100 * (c["xi_hat"] - e["xi_set"]) / e["xi_set"],
                   c["wn_hat"], 100 * (c["wn_hat"] - e["wn_set"]) / e["wn_set"], c["n_peaks"]))
    add("")
    add("## 判据 2：无源性（Lyapunov 数值积分）\n")
    for k, a in d.get("algorithms", {}).items():
        p = (a.get("extras") or {}).get("passivity")
        if not p:
            continue
        mm = p["measured"]
        add("- %s：V(0) = %s → V(T) = %s；Σ正增量 = %s、最大正增量 = %s、逐点 V ≤ V(0) = %s"
            % (k, _fmt(mm["V0"]), _fmt(mm["VT"]), _fmt(mm["sum_positive_dV"]),
               _fmt(mm["max_positive_dV"]), mm["V_never_exceeds_V0"]))
    add("")
    add("## 判据 4：JS/OS 分工（快速轨迹）\n")
    for pair, v in (d.get("division_of_labor") or {}).get("pairs", {}).items():
        add("- **%s**：最快 T = %.2f s 时 PD 峰值误差 %s、逆动力学 %s → **%.1f×**"
            "（阈值 ≥ 10×）；各速度下峰值误差离散度 PD %.2f×、逆动力学 %.2f×"
            % (pair, v["fastest_T_s"], _fmt(v["err_peak_pd"], 3), _fmt(v["err_peak_invdyn"], 3),
               v["ratio_pd_over_invdyn"], v["pd_peak_spread_over_speeds"],
               v["invdyn_peak_spread_over_speeds"]))
        for r in v["rows"]:
            add("    - %s T = %.1f s（峰加速度 %.2f rad/s²）：峰值误差 %s，RMS %s"
                % (r["kind"], r["T_traj_s"], r["peak_accel_rad_s2"],
                   _fmt(r["err"]["peak"], 3), _fmt(r["err"]["rms"], 3)))
    add("")
    cp = d.get("coupling") or {}
    if cp:
        add("## 判据 5：耦合抑制\n")
        add("- 第 1 关节 %.2f rad / %.2f s 快速轨迹：PD 非指令位移 %s rad（指令幅度的 %.3f%%）、"
            "逆动力学 %s rad（%.4f%%）→ **%.1f×**"
            % (cp["amp_rad"], cp["T_traj_s"], _fmt(cp["rows"]["js_pd"]["noncommanded_max_rad"], 3),
               100 * cp["fraction_of_commanded_pd"],
               _fmt(cp["rows"]["js_invdyn"]["noncommanded_max_rad"], 3),
               100 * cp["fraction_of_commanded_invdyn"], cp["ratio_pd_over_invdyn"]))
        rf = cp.get("dt_refinement")
        if rf:
            add("- 步长归因：dt = %.0e s 重跑 → %s rad（降 %.2f×，零阶保持 O(dt) 预期 %.2f×）"
                % (rf["dt_s"], _fmt(rf["noncommanded_max_rad"], 3),
                   rf["drop_ratio_vs_main"] or float("nan"), rf["expected_ratio_if_O_dt"]))
        add("")
    if d.get("os_mapping"):
        add("## 判据 6：OS 双向自洽\n")
        for k, v in d["os_mapping"].items():
            add("- %s（%d 个随机状态）：‖F − 意图‖/‖意图‖ 最大 %s，功率一致性偏差最大 %s"
                % (k, v["n_states"], _fmt(v["worst_rel_err_F"], 3), _fmt(v["worst_rel_err_power"], 3)))
        add("")
    if d.get("os_kp_sweep"):
        add("## §3.5：OS 的 K_P 扫描（常值未补偿扰动）\n")
        for k, v in d["os_kp_sweep"]["kinds"].items():
            add("- %s：K_P = %s → 稳态误差 %s（斜率 %.4f，单调 %s，首末比 %.1f×；理论 −1）"
                % (k, ["%.3g" % x for x in v["K_P_mean"]], ["%.3e" % x for x in v["err"]],
                   v["slope"], v["monotone"], v["decade_ratio"]))
        add("")
    if d.get("linearization_bias"):
        b = d["linearization_bias"]
        add("## §3.5/§3.6：x̃ ≈ J_aq̃ 一阶线性化偏差\n")
        add("- 相对偏差 log-log 斜率 %.3f（理论 1）、绝对偏差斜率 %.3f（理论 2）；"
            "‖x̃_rot‖/真实姿态角 %.3f~%.3f；欧拉角往返守卫 %.3e"
            % (b["rel_bias_slope"], b["abs_bias_slope"], b["rows"][0]["rot_ratio"],
               b["rows"][-1]["rot_ratio"], b["euler_roundtrip_guard"]["worst_roundtrip_err"]))
        add("")
    sc = d.get("step_size_check")
    if sc:
        add("## 决策记录 1：步长收敛对照（逐分量）\n")
        for r in sc["2nd_order"]:
            add("- 二阶辨识 [%s] dt %.0e → %.0e：最差 |Δξ̂| = %.4f（占设定 ξ 的 %.2f%%）、"
                "|Δω̂_n| = %.4f（%.2f%%）"
                % (r["kind"], r["diff"]["dt_main_s"], r["diff"]["dt_ref_s"],
                   r["diff"]["worst_xi_hat_abs_diff"],
                   100 * r["diff"]["worst_xi_hat_abs_diff_rel_to_set"],
                   r["diff"]["worst_wn_hat_abs_diff"],
                   100 * r["diff"]["worst_wn_hat_abs_diff_rel_to_set"]))
            for c in r["diff"]["components"]:
                add("    - %s：ξ̂ %s → %s，ω̂_n %s → %s"
                    % (c["component"], _fmt(c["xi_hat_dt_main"], 4), _fmt(c["xi_hat_dt_ref"], 4),
                       _fmt(c["wn_hat_dt_main"], 4), _fmt(c["wn_hat_dt_ref"], 4)))
        for r in sc["regulation"]:
            add("- 调节 [%s] dt %.0e → %.0e：末态 ‖q̃‖∞ %s → %s（相对差 %.2f%%），‖u‖∞ %s → %s"
                % (r["kind"], r["rows"][0]["dt_s"], r["rows"][1]["dt_s"],
                   _fmt(r["rows"][0]["q_tilde_inf_final"], 3), _fmt(r["rows"][1]["q_tilde_inf_final"], 3),
                   100 * r["diff"]["q_tilde_inf_final_rel"], _fmt(r["rows"][0]["u_inf"], 2),
                   _fmt(r["rows"][1]["u_inf"], 2)))
        add("")
    for name, pr in (d.get("fit_dt_probe") or {}).items():
        add("## 判据 1 步长探针 `%s`（T = %.1f s）\n" % (name, pr["T_s"]))
        for row in pr["rows"]:
            worst = max(c["xi_rel_err"] for c in row["components"] if c["xi_rel_err"] is not None)
            add("- %s @ dt = %.0e s（%d 步）：最差 ξ̂ 相对偏差 **%.2f%%**"
                % (row["kind"], row["dt_s"], row["n_steps"], 100 * worst))
        add("")
    add("## 图\n")
    for f in d.get("figures", []):
        add("- `%s`" % f)
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段 3 归档摘要（只读 result.json）")
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--md", default=None, help="另存 Markdown 文件路径")
    args = ap.parse_args(argv)
    if not os.path.exists(args.json):
        print("找不到归档：%s" % args.json)
        return 2
    with io.open(args.json, encoding="utf-8") as f:
        d = json.load(f)
    txt = summarize(d)
    sys.stdout.write(txt)
    if args.md:
        with io.open(args.md, "w", encoding="utf-8") as f:
            f.write(txt)
        print("\n已写出：%s" % args.md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
