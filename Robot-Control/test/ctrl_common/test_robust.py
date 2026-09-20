# -*- coding: utf-8 -*-
"""阶段 4.2：**鲁棒性与边界**的四类实验（计划 §5.4.2）——模型失配 / 摩擦未补偿 /
测量噪声 / 控制周期 / 离散化失稳 + 每周期计算耗时实测。

| 实验名 | 内容 | 对应书中论断 | 判据 |
|---|---|---|---|
| `mismatch` | 质量/惯量 ±10%/±30% 施加于 plant，控制器用标称模型 | 强调 model-based 依赖参数 | 1 |
| `friction` | plant 启用粘性+库仑摩擦，控制器 `F_f = 0` | `F_f` 半正定、需补偿 | 1 |
| `noise` | `q, q̇` 加高斯噪声（含速度由位置差分）＋一阶低通 | 状态来自传感器/滤波器 | 1 |
| `period` | `h_c ∈ {0.5, 1, 5} ms`（零阶保持） | 「实时计算动力学曾是工程难点」 | 1 |
| `instab` | 增益扫到超出 `h_c` 可承受范围，记录失稳阈值 | ——（用于区分离散化失稳与连续律错误）| 3 |
| `cost` | 每周期计算耗时实测（`time.perf_counter`） | 同上（实时性）| 报告项 |

⚠️ 全部实验的**被控对象与控制器模型都不同源**（失配/摩擦/噪声只作用于 plant）——
这是计划 §二「绝不能用同一个模型同时当 plant 和控制器模型」的落地。

驱动与口径见 `test/ctrl_common/stage4_lib.py`；一键入口与归档见 `run_stage4.py`。

用法（`py311-gym`，任意 cwd）：

    python Robot-Control/test/ctrl_common/test_robust.py                 # 全部（约 15–25 min）
    python Robot-Control/test/ctrl_common/test_robust.py --quick         # 冒烟
    python Robot-Control/test/ctrl_common/test_robust.py --only noise    # 只跑一项
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, "..", "..", "code"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                  # noqa: E402
import stage4_lib as s4                                             # noqa: E402

EXPERIMENTS = ("mismatch", "friction", "noise", "period", "instab", "cost")


def run(name, quick=False, workers=None):
    """跑单项实验，返回结构化结论（键名与 `run_stage4.py` 的归档字段一致）。"""
    if name == "mismatch":
        if quick:
            return s4.model_mismatch(levels=(0.9, 1.0, 1.1), inertia_levels=(0.7, 1.0),
                                     T=0.3, T_inertia=0.4, T_long=0.8, workers=workers)
        return s4.model_mismatch(workers=workers)
    if name == "friction":
        if quick:
            return s4.friction_uncompensated(coulomb_levels=(0.0, 1.0),
                                             fric_cases={"both": s4.FRIC_CASES["both"]},
                                             T=0.3, T_shape=0.25, workers=workers)
        return s4.friction_uncompensated(workers=workers)
    if name == "noise":
        if quick:
            cases = {"clean": None, "qd_only": s4.NOISE_CASES["qd_only"],
                     "q_diff": s4.NOISE_CASES["q_diff"],
                     "q_diff_lpf": s4.NOISE_CASES["q_diff_lpf"]}
            return s4.measurement_noise(noise_cases=cases, T=0.2, workers=workers)
        return s4.measurement_noise(workers=workers)
    if name == "period":
        if quick:
            return s4.control_period(hc_list=(5e-4, 5e-3), T=0.15, T_traj=0.12,
                                     dq=np.array([0.0, 0.2, -0.2, 0.2, 0.1, -0.1, 0.2]),
                                     workers=workers)
        return s4.control_period(workers=workers)
    if name == "instab":
        if quick:
            return s4.discretization_instability(dt_list=(s4.DT4,), wn_max=300.0, n_bisect=3)
        return s4.discretization_instability()
    if name == "cost":
        return s4.compute_cost(n_rep=50 if quick else 200)
    raise ValueError("未知实验 %r（可选：%s）" % (name, ",".join(EXPERIMENTS)))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    parts = list(EXPERIMENTS)
    for i, a in enumerate(argv):
        if a == "--only" and i + 1 < len(argv):
            parts = [p.strip() for p in argv[i + 1].split(",") if p.strip()]
        elif a.startswith("--only="):
            parts = [p.strip() for p in a.split("=", 1)[1].split(",") if p.strip()]
    bad = [p for p in parts if p not in EXPERIMENTS]
    if bad:
        print("未知实验名 %s（可选：%s）" % (bad, ",".join(EXPERIMENTS)))
        return 2
    workers = None
    for i, a in enumerate(argv):
        if a == "--workers" and i + 1 < len(argv):
            workers = int(argv[i + 1])
        elif a.startswith("--workers="):
            workers = int(a.split("=", 1)[1])
    print("=" * 72)
    print("阶段 4.2：鲁棒性与边界实验（计划 §5.4.2）")
    print("=" * 72)
    ok_all = True
    for name in parts:
        print("\n>>> %s" % name)
        res = run(name, quick=quick, workers=workers)
        if isinstance(res, dict) and "check" in res:
            c = res["check"]
            print("   [%s] %s" % ("PASS" if c["passed"] else "FAIL", c["threshold"]))
            ok_all = ok_all and bool(c["passed"])
    if quick:
        print("\n⚠️ 冒烟模式（--quick）：时长/扫描点被压缩，判据结论无意义。")
        ok_all = True
    print("结论：%s" % ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
