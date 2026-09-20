# -*- coding: utf-8 -*-
"""**一次性定点探针**：判据 1（误差动态精确性）的步长口径判定（日志尝试 9「下一步」第 1 条）。

背景：阶段 3 首轮完整跑（`res/6r_stage3/run_attempt1.log`）在 `dt = 2 ms` 上测得最小惯量关节
`q̃_6`（`B_66 = 1.200e-03`）的 `ξ̂` 有 **+2.93% 系统偏差**（其余 5 关节 ≤1.07%、`ω̂_n` 全部 ≤0.30%），
且已排除「观测窗太短」（窗 2.4→2.9→3.4 s 反而 2.93%→3.08%→3.08%）。
唯一未排除的来源是**离散化（零阶保持 + RK4）**，故本脚本把**同一条判据 1 实验**在多个步长上重跑，
看 `ξ̂_6` 是否随 `dt → 0` 收敛回 2% 阈值内。

用法（`py311-gym`，任意 cwd；6 路并行，约 4–5 min）：

    python Robot-Control/scripts/probe_fit_dt.py
    python Robot-Control/scripts/probe_fit_dt.py --dts 2e-3,1e-3,5e-4 --kinds js_invdyn,os_invdyn

产出（**证据留档**）：`res/6r_stage3/probe_fit_dt.json`（机读）+ 同目录 `probe_fit_dt.log` 由 shell 重定向。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RC = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_RC, os.path.join(_RC, "code"), os.path.join(_RC, "test", "ctrl_common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import code_metrics as met                                          # noqa: E402
import code_models as cm                                            # noqa: E402
import stage3_lib as s3                                             # noqa: E402
import verify_lib as vl                                             # noqa: E402


def probe(kinds, dts, T=s3.T_2ND, wn=s3.WN_2ND, xi=s3.XI_2ND, dq=s3.DQ_2ND,
          seed=s3.SEED3, workers=None):
    """把判据 1 的同一条实验在 `dts` 上各跑一遍，逐分量辨识 `(ξ̂, ω̂_n)`。"""
    case = cm.make_case("6R")
    q_d = s3.q_reg("6R")
    v = s3._perturb_dir(case.N, seed=seed)
    q0 = q_d.copy()
    q0[1:] += dq * v
    specs, meta = [], []
    for kind in kinds:
        gains = s3.s3_gains(kind, case, q_d, wn=wn, xi=xi)
        for dt in dts:
            specs.append(vl.make_spec(kind, "6R", gains, vl.make_ref_spec(kind, case, q_d),
                                      q0, np.zeros(case.N + 1), T, dt, dt,
                                      label="%s-2nd-dt%.0e" % (kind, dt)))
            meta.append((kind, float(dt)))
    t0 = time.perf_counter()
    recs = vl.run_specs(specs, workers=workers, tag="probe-fitdt", verbose=True)
    wall = time.perf_counter() - t0
    out = {"T_s": T, "wn_set": wn, "xi_set": xi, "dq_rad": dq, "seed": seed,
           "q_d": [float(x) for x in q_d], "perturb_dir": [float(x) for x in v],
           "wall_time_s": wall, "rows": []}
    for (kind, dt), res in zip(meta, recs):
        if kind.startswith("js"):
            sigs, names = res["q_tilde"][:, 1:], ["q̃_%d" % (j + 1) for j in range(case.N)]
        else:
            sigs, names = res["x_tilde"], ["x̃_%d" % (i + 1) for i in range(case.m)]
        comps = []
        for i in range(sigs.shape[1]):
            ident = met.identify_second_order(res["t"], sigs[:, i])
            comps.append({"component": names[i], "n_peaks": ident["n_peaks"],
                          "xi_hat": ident["xi_hat"], "wn_hat": ident["wn_hat"],
                          "xi_rel_err": (abs(ident["xi_hat"] - xi) / xi
                                         if ident["xi_hat"] is not None else None),
                          "wn_rel_err": (abs(ident["wn_hat"] - wn) / wn
                                         if ident["wn_hat"] is not None else None),
                          "peak_t": ident["peak_t"], "peak_v": ident["peak_v"],
                          "period_s": ident["period_s"]})
        out["rows"].append({"kind": kind, "dt_s": dt, "components": comps,
                            "n_steps": int(res["n_steps"]),
                            "wall_time_s": float(res["wall_time_s"]),
                            "nan": bool(res["nan"]), "error": res["error"]})
        if not out.get("x_manifold_note"):
            out["x_manifold_note"] = ("OS 用例：x̃ 由 ref 的 x_d 与 fk(q) 直接给出；"
                                      "控制器始终只看 x_d（红线）")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="判据 1 的步长定点探针")
    ap.add_argument("--dts", default="2e-3,1e-3,5e-4", help="逗号分隔的 dt（s）")
    ap.add_argument("--kinds", default="js_invdyn,os_invdyn", help="逗号分隔的算法")
    ap.add_argument("--T", type=float, default=s3.T_2ND, help="观测窗（s）")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(_RC, "res", "6r_stage3", "probe_fit_dt.json"))
    args = ap.parse_args(argv)
    dts = [float(s) for s in args.dts.split(",") if s.strip()]
    kinds = [s.strip() for s in args.kinds.split(",") if s.strip()]
    print("#" * 72)
    print("# 判据 1 步长定点探针：dt = %s；算法 = %s" % (dts, kinds))
    print("# 设定 ξ = %.2f、ω_n = %.1f rad/s，T = %.1f s，初值扰动 %.3f rad（全正随机方向）"
          % (s3.XI_2ND, s3.WN_2ND, args.T, s3.DQ_2ND))
    print("#" * 72)
    res = probe(kinds, dts, T=args.T, workers=args.workers)
    print("-" * 72)
    for row in res["rows"]:
        print("[%s @ dt=%.0e s]  %d 步，墙钟 %.0f s" % (row["kind"], row["dt_s"],
                                                        row["n_steps"], row["wall_time_s"]))
        for c in row["components"]:
            print("    %s：ξ̂=%s  ω̂_n=%s  相对偏差 %s / %s  正峰 %d"
                  % (c["component"],
                     "%.4f" % c["xi_hat"] if c["xi_hat"] is not None else "n/a",
                     "%.4f" % c["wn_hat"] if c["wn_hat"] is not None else "n/a",
                     "%.2f%%" % (100 * c["xi_rel_err"]) if c["xi_rel_err"] is not None else "n/a",
                     "%.2f%%" % (100 * c["wn_rel_err"]) if c["wn_rel_err"] is not None else "n/a",
                     c["n_peaks"]))
    # 焦点量：每个算法在最小惯量/最后一个分量上的 ξ̂ 偏差，随 dt 的变化
    print("-" * 72)
    print("焦点：ξ̂ 相对偏差随 dt 的变化（%s）"
          % "，".join("分量 %s" % r["components"][-1]["component"] for r in res["rows"][:1]))
    for kind in kinds:
        rows = [r for r in res["rows"] if r["kind"] == kind]
        rows.sort(key=lambda r: -r["dt_s"])
        last = rows[0]["components"][-1]["component"]
        seq = [(r["dt_s"], [c for c in r["components"] if c["component"] == last][0]["xi_rel_err"])
               for r in rows]
        print("  %-9s %s：%s" % (kind, last,
                                 "  ".join("dt=%.0e→%.2f%%" % (dt, 100 * e)
                                           for dt, e in seq if e is not None)))
        if len(seq) >= 2 and all(e is not None for _, e in seq):
            # 经验收敛阶：相邻两点 log(e1/e2)/log(dt1/dt2)
            for (dt1, e1), (dt2, e2) in zip(seq[:-1], seq[1:]):
                if e1 > 0 and e2 > 0:
                    print("            经验阶 dt %.0e→%.0e：log(e1/e2)/log(dt1/dt2) = %.2f"
                          % (dt1, dt2, np.log(e1 / e2) / np.log(dt1 / dt2)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("归档：%s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
