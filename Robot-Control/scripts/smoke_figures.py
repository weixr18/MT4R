# -*- coding: utf-8 -*-
"""**出图烟雾测试**：用合成数据把 `stage3_lib.make_figures` 的每条路径都跑一遍（几秒，不出仿真）。

## 为什么需要它

2026-09-19 阶段 3 的**首次完整跑**（63.7 min）在最后一步 `make_figures` 抛
`KeyError: '_raw'` 而**全部作废**——`algorithm_stage3` 当时把 `extras["passivity"] /
["error_dynamics"]` 的 `_raw` 剥离了，而 `make_figures` 要 `_raw` 里的原始曲线。
两处修好之后：
1. `algorithm_stage3` 的 `extras` **整份保留**（写归档时 `run_stage3._strip` 递归丢掉 `_raw`）；
2. `run_stage3.main` **先写 result.json、再出图**，且出图整体包在 `try` 里。

本脚本是第 3 道保险：**不开仿真**就能验证「出图这条链路 + 字段契约」是否完好，
几秒钟即可跑完，适合在任何改动 `stage3_lib`/`code_metrics` 出图函数之后先跑一次。

用法（`py311-gym`，任意 cwd）：

    python Robot-Control/scripts/smoke_figures.py            # 出图到临时目录并校验，然后删除
    python Robot-Control/scripts/smoke_figures.py --keep     # 保留图以便肉眼检查
"""
import argparse
import os
import shutil
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RC = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_RC, os.path.join(_RC, "code"), os.path.join(_RC, "test", "ctrl_common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import code_models as cm                                            # noqa: E402
import stage3_lib as s3                                             # noqa: E402


def _fake_rec(kind, n=400, T=2.0, seed=0):
    """一条**结构合法**的假仿真记录（`simulate` 的键子集，够出图用）。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, n)
    decay = np.exp(-2.0 * t)
    n_j = 6
    q = np.zeros((n, n_j + 1))
    q_tilde = np.zeros((n, n_j + 1))
    u = np.zeros((n, n_j + 1))
    for j in range(1, n_j + 1):
        q[:, j] = 0.3 + 0.4 * decay * np.cos(6.0 * t + j)
        q_tilde[:, j] = 0.4 * decay * np.cos(6.0 * t + j)
        u[:, j] = 50.0 * decay * np.sin(6.0 * t + j)
    m = 6
    x_tilde = np.zeros((n, m))
    for i in range(m):
        x_tilde[:, i] = 0.2 * decay * np.cos(6.0 * t + i * 0.3)
    return {"t": t, "q": q, "qdot": np.zeros((n, n_j + 1)), "u": u,
            "x_tilde": x_tilde, "q_tilde": q_tilde, "dt": 2e-3, "h_c": 2e-3,
            "n_samples": n, "u_abs_max": float(np.abs(u).max()),
            "nan": False, "error": None, "wall_time_s": 1.0,
            "case": kind, "label": kind, "T": T, "sigma_min": None,
            "sigma_max": None, "cond_B": None}


def _fake_algs():
    import verify_lib as vl
    algs = {}
    for k in s3.KIND_ORDER:
        recs = [_fake_rec(k, seed=i) for i in range(3)]
        # ⚠️ 名字必须与真实用例同构（`make_figures` 会按空格切掉空间前缀，如
        #    "JS 重力补偿 PD".split(" ", 1)[1] == "重力补偿 PD"）
        algs[k] = {"kind": k, "name": vl.CTRLS[k]["name"], "params": {"n_ic": 3},
                   "worst_ic_index": 1, "_raw": {"recs": recs}}
    return algs


def _fake_fit():
    n, T = 600, 3.0
    t = np.linspace(0.0, T, n)
    sig = 0.05 * np.exp(-0.3 * 6.0 * t) * np.cos(5.72 * t)
    ident = {"xi_hat": 0.2999, "wn_hat": 6.001, "peak_t": [0.55, 1.65, 2.75],
             "peak_v": [0.018, 0.0025, 0.00035], "n_peaks": 3}
    out = {}
    for k in ("js_invdyn", "os_invdyn"):
        out[k] = {"xi_set": s3.XI_2ND, "wn_set": s3.WN_2ND, "_raw": {"t": t, "signal": sig,
                                                                    "ident": ident}}
    return out


def _fake_pas():
    n, T = 800, 1.5
    t = np.linspace(0.0, T, n)
    return {k: {"_raw": {"t": t, "V": 2.0 * np.exp(-3.0 * t) + 1e-12}}
            for k in ("js_pd", "js_invdyn")}


def _fake_div():
    rows = []
    for pair in ("js", "os"):
        for suffix in ("_pd", "_invdyn"):
            for T in s3.T_TRAJ_LIST:
                rows.append({"kind": pair + suffix, "T_traj_s": T,
                             "err": {"peak": 0.2 if suffix == "_pd" else 2e-3}})
    return {"_raw": {"rows": rows}}


def _fake_coup():
    return {"amp_rad": s3.AMP_COUP, "T_traj_s": s3.T_COUP,
            "rows": {"js_pd": {"noncommanded_max_rad": 2e-2},
                     "js_invdyn": {"noncommanded_max_rad": 3e-4}}}


def _fake_kp():
    return {"kinds": {k: {"K_P_mean": [64.0, 640.0, 6400.0],
                          "err": [5.0, 0.5, 0.05]} for k in ("os_pd", "os_invdyn")}}


def main(argv=None):
    ap = argparse.ArgumentParser(description="stage3 出图烟雾测试（合成数据，不出仿真）")
    ap.add_argument("--keep", action="store_true", help="保留出图目录以便肉眼检查")
    ap.add_argument("--dir", default=os.path.join(_RC, "res", "_figs_smoke"))
    args = ap.parse_args(argv)
    # σ_min/σ_max/cond_B 只在指标汇总里用，出图不需要；这里显式补成数组以免误用
    algs = _fake_algs()
    for v in algs.values():
        for rec in v["_raw"]["recs"]:
            rec["sigma_min"] = np.full(rec["t"].size, 0.15)
            rec["sigma_max"] = np.full(rec["t"].size, 1.9)
            rec["cond_B"] = np.full(rec["t"].size, 1.7e3)
    if os.path.isdir(args.dir):
        shutil.rmtree(args.dir, ignore_errors=True)
    print("出图烟雾测试：写入 %s（用例 %s，ℓ = %.1f m）"
          % (args.dir, cm.make_case("6R").name, cm.ELL_DEFAULT))
    figs = s3.make_figures(algs, _fake_div(), _fake_coup(), _fake_pas(), _fake_fit(),
                           _fake_kp(), args.dir, tag="smoke")
    missing = [p for p in figs if not os.path.exists(p)]
    print("生成 %d 张图：" % len(figs))
    for p in figs:
        print("  %-46s %8d B" % (os.path.basename(p), os.path.getsize(p) if os.path.exists(p) else -1))
    ok = len(figs) == 7 and not missing
    print("结论：%s（期望 7 张：误差曲线 / 控制力矩 / 对数衰减率 / 速度对比 / 耦合柱状 / "
          "无源性 / K_P 扫描）" % ("PASS" if ok else "FAIL"))
    if not args.keep and ok:
        shutil.rmtree(args.dir, ignore_errors=True)
        print("已删除临时出图目录（--keep 可保留）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
