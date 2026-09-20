# -*- coding: utf-8 -*-
"""阶段 2（2R 首轮贯通）：**JS 重力补偿 PD 控制**——书 `algo:robctrl_js_pd`。

`chap4_3_jsctrl.tex`：`u = K_P(q_d − q) − K_D q̇ + g(q)`（关节空间调节控制）。
本脚本在**平面 2R** 用例上跑计划 §5.2.5 的四条判据（跑通 / 收敛 / 力矩量级 / 端到端自洽），
实现与判据口径全部在 `test/ctrl_common/verify_lib.py`（四个控制器共用同一套驱动）。

- 通过判据的**正式结论**见 `res/2r_stage2/result.json`（由 `run_stage2.py` 汇总归档）；
- **阶段 3**（6R 正式验证）的算法内检查见 `run_stage3()`：判据 3（20 组初值收敛）+
  判据 2（无源性，`V = ½q̇ᵀBq̇ + ½q̃ᵀK_Pq̃` 无正功事件）；驱动口径在
  `test/ctrl_common/stage3_lib.py`，跨算法判据（4/5/6）由 `run_stage3.py` 统一驱动。
- 精度类结论（误差动态、JS/OS 分工、耦合抑制）见阶段 3 的对应脚本。

用法（`py311-gym`，任意 cwd）：

    python Robot-Control/test/ctrl_js/test_js_pd.py            # 阶段 2 完整（约 5 min）
    python Robot-Control/test/ctrl_js/test_js_pd.py --quick    # 阶段 2 冒烟（约 10 s）
    python Robot-Control/test/ctrl_js/test_js_pd.py --stage3   # 阶段 3（6R，约 15 min）

退出码：判据全 PASS 为 0，否则 1。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "ctrl_common")))

import stage3_lib as s3                                            # noqa: E402
import verify_lib as vl                                            # noqa: E402

KIND = "js_pd"


def run(case_name="2R", **kwargs):
    """跑该算法在 `case_name` 上的阶段 2 检查，返回结构化结果（见 `verify_lib.stage2_checks`）。"""
    return vl.stage2_checks(KIND, case_name=case_name, **kwargs)


def run_stage3(case_name="6R", **kwargs):
    """阶段 3：该算法在 6R 上的正式验证（判据 3 + 该算法适用的专项判据）。

    见 `test/ctrl_common/stage3_lib.algorithm_stage3`；跨算法判据（4/5/6）由 `run_stage3.py` 驱动。
    """
    return s3.algorithm_stage3(KIND, case_name=case_name, **kwargs)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    if "--stage3" in argv:
        print("=" * 72)
        print("阶段 3：%s @ 6R（书 algo:robctrl_js_pd）" % vl.CTRLS[KIND]["name"])
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
    print("阶段 2：%s @ 2R（书 algo:robctrl_js_pd）" % vl.CTRLS[KIND]["name"])
    print("=" * 72)
    res = run(quick=quick)
    print("-" * 72)
    vl.print_checks(res)
    ok = vl.all_passed(res, quick=quick)
    print("结论：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
