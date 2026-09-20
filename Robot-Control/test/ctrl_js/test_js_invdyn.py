# -*- coding: utf-8 -*-
"""阶段 2（2R 首轮贯通）：**JS 逆动力学 PD 控制**——书 `algo:robctrl_js_invdyn`。

`chap4_3_jsctrl.tex`：`y = K_Pq̃ + K_Dq̇̃ + q̈_d`，`u = By + Cq̇ + F_fq̇ + g`（关节空间跟踪控制）。
本脚本在**平面 2R** 用例上跑计划 §5.2.5 的四条判据；驱动与判据口径见
`test/ctrl_common/verify_lib.py`。

> 阶段 3.1 的**最强判据**（误差动态 `q̈̃ + K_Dq̇̃ + K_Pq̃ = 0` 的 `(ξ̂, ω̂_n)` 拟合偏差 <2%）
> 在阶段 3 的 `run_stage3()` 里正式判定（6R，设定 ξ=0.3、ω_n=6；本阶段只作**诊断量**打印）。

用法（`py311-gym`，任意 cwd）：

    python Robot-Control/test/ctrl_js/test_js_invdyn.py            # 阶段 2 完整
    python Robot-Control/test/ctrl_js/test_js_invdyn.py --quick    # 阶段 2 冒烟
    python Robot-Control/test/ctrl_js/test_js_invdyn.py --stage3   # 阶段 3（6R，判据 1+3）

退出码：判据全 PASS 为 0，否则 1。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "ctrl_common")))

import stage3_lib as s3                                            # noqa: E402
import verify_lib as vl                                            # noqa: E402

KIND = "js_invdyn"


def run(case_name="2R", **kwargs):
    """跑该算法在 `case_name` 上的阶段 2 检查，返回结构化结果（见 `verify_lib.stage2_checks`）。"""
    return vl.stage2_checks(KIND, case_name=case_name, **kwargs)


def run_stage3(case_name="6R", **kwargs):
    """阶段 3：该算法在 6R 上的正式验证（判据 1 误差动态精确性 + 判据 3 收敛性）。"""
    return s3.algorithm_stage3(KIND, case_name=case_name, **kwargs)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    if "--stage3" in argv:
        print("=" * 72)
        print("阶段 3：%s @ 6R（书 algo:robctrl_js_invdyn）" % vl.CTRLS[KIND]["name"])
        print("=" * 72)
        res = run_stage3(quick=quick)
        print("-" * 72)
        for k, c in res["checks"].items():
            print("   [%s] %s（阈值：%s）" % ("PASS" if c["passed"] else "FAIL", k, c["threshold"]))
        ok = all(c["passed"] for c in res["checks"].values())
        if quick:
            print("⚠️ 冒烟模式（--quick）：判据结论无意义，退出码只看「无异常」。")
            ok = True
        print("结论：%s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    print("=" * 72)
    print("阶段 2：%s @ 2R（书 algo:robctrl_js_invdyn）" % vl.CTRLS[KIND]["name"])
    print("=" * 72)
    res = run(quick=quick)
    print("-" * 72)
    vl.print_checks(res)
    ok = vl.all_passed(res, quick=quick)
    print("结论：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
