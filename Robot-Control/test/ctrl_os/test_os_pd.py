# -*- coding: utf-8 -*-
"""阶段 2（2R 首轮贯通）：**OS 重力补偿 PD 控制**——书 `algo:robctrl_os_pd`。

`chap4_4_osctrl.tex`：`u = J_aᵀ(K_P(x_d − x_e) − K_D J_aq̇) + g`（操作空间调节控制）。

⚠️ **2R 的任务空间取 2 维位置子任务 `[x, y]`**（`J_a` 为 2×2 方阵）：`robot_fk_2r` 输出 3 维
`[x, y, φ]`、其 `J_a` 是 3×2（满列秩的浸入），书中 OS 逆动力学算法的 `J_a⁻¹` 不存在。
修订理由与替代方案见 `code/code_models.py` 文件头与书仓库日志尝试 8。

驱动与判据口径见 `test/ctrl_common/verify_lib.py`（阶段 2）与 `test/ctrl_common/stage3_lib.py`
（阶段 3）；跨算法判据（4/5/6）由 `test/ctrl_common/run_stage3.py` 统一驱动。

用法（`py311-gym`，任意 cwd）：

    python Robot-Control/test/ctrl_os/test_os_pd.py            # 阶段 2 完整
    python Robot-Control/test/ctrl_os/test_os_pd.py --quick    # 阶段 2 冒烟
    python Robot-Control/test/ctrl_os/test_os_pd.py --stage3   # 阶段 3（6R，判据 3；另参与 4/6）
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "ctrl_common")))

import stage3_lib as s3                                            # noqa: E402
import verify_lib as vl                                            # noqa: E402

KIND = "os_pd"


def run(case_name="2R", **kwargs):
    """跑该算法在 `case_name` 上的阶段 2 检查，返回结构化结果（见 `verify_lib.stage2_checks`）。"""
    return vl.stage2_checks(KIND, case_name=case_name, **kwargs)


def run_stage3(case_name="6R", **kwargs):
    """阶段 3：该算法在 6R 上的正式验证（判据 3 收敛性 + 增益按任务惯量标定）。"""
    return s3.algorithm_stage3(KIND, case_name=case_name, **kwargs)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    if "--stage3" in argv:
        print("=" * 72)
        print("阶段 3：%s @ 6R（书 algo:robctrl_os_pd）" % vl.CTRLS[KIND]["name"])
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
    print("阶段 2：%s @ 2R（书 algo:robctrl_os_pd）" % vl.CTRLS[KIND]["name"])
    print("=" * 72)
    res = run(quick=quick)
    print("-" * 72)
    vl.print_checks(res)
    ok = vl.all_passed(res, quick=quick)
    print("结论：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
