# -*- coding: utf-8 -*-
"""阶段 4.1（**必须做**）：**OS 控制器的奇异位形边界**——计划 §5.4.1。

书中 OS 控制的稳定性证明依赖「`J_a` **列满秩**」；本脚本把该前提定量化：
用 `σ_min(J_a)` 作为奇异度度量，从标称值（实测 `1.53e-1`）扫到 ≈0，记录末端误差与控制量的变化，
并给出**明确的失效点**与失稳表现。

**怎么把 σ_min 变成可控自变量**：只动腕部关节 `q5`（`q5 → 0` 时关节 4/6 轴共线、`σ_min → 0`），
其余关节固定在调节目标上，用二分反解出「给定 σ_min 对应的 q5」。两种场景 + 一个 JS 对照组：

| 场景 | 指令 | 初值 | 含义 |
|---|---|---|---|
| `out_of_range` | `x_d = fk(q_t) + a·u_min` | `q_t` | 指令落在**退化方向**（`u_min` 是最小左奇异向量）⇒ 所需关节运动 `a/σ_min`，预期直至失效 |
| `in_range` | `x_d = fk(q_t) + a·u_max` | `q_t` | 同一幅度沿**最良态**方向（`u_max` 是第一左奇异向量）⇒ 所需关节运动 `a/σ_max ≈ 0.01 rad`，**不应失效** |
| `js_hold`（对照组） | `q_d = q_t` | `q_t + DQ4·v` | 同一位形上的 JS 调节：`J_a` 不进 JS 控制律 ⇒ 应完全正常 |

> ⚠️ **两个 OS 场景的初值与幅度完全相同，唯一差别是方向**（2026-09-20 探针定下的口径，见日志尝试 13）。
> 旧口径的 `in_range` 用「指令 = `fk(q_t)`、初值 = `q_t + DQ4` 的**关节**扰动」，而深奇异处
> `|q5_t|` 只有 `2.25e-3`（σ = 1e-3）、`DQ4 = 0.05 rad`，扰动**必然把臂推过奇异面**——
> 于是「本不该失效」的对照组测到的是「穿过奇异面」而不是「位形病态」。改成方向对照后，
> `in_range` 所需的关节运动与 `σ_min` 无关，两个场景才可比。
>
> 失效判据：NaN/异常；**反馈力矩** `‖u−g‖∞` 超上限或 `‖q̇‖∞` 超限；误差末值/**全程峰值** > 0.1
> （不衰减）；**穿越奇异面**（仅 OS 场景）。

⚠️ **这不是书的错误**：书中已声明 `J_a` 列满秩的前提。阻尼最小二乘（DLS）只是
**超出书本范围的建议**（`stage4_lib.ctrl_os_invdyn_dls`），在报告里必须如此标注。

驱动与口径见 `test/ctrl_common/stage4_lib.py`；一键入口与归档见 `run_stage4.py`。

用法（`py311-gym`，任意 cwd）：

    python Robot-Control/test/ctrl_os/test_os_singular.py            # 完整（约 20–30 min）
    python Robot-Control/test/ctrl_os/test_os_singular.py --quick    # 冒烟（只看链路跑通）
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "ctrl_common")))

import stage4_lib as s4                                             # noqa: E402

KIND = "os_singular"
QUICK_TARGETS = (None, 1e-2, 1e-4)


def run(quick=False, workers=None, **kw):
    """跑奇异边界扫描，返回结构化结论（见 `stage4_lib.singular_sweep`）。"""
    if quick:
        kw.setdefault("targets", QUICK_TARGETS)
        kw.setdefault("T", 0.2)
    return s4.singular_sweep(workers=workers, **kw)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    print("=" * 72)
    print("阶段 4.1：OS 控制的奇异位形边界（书 algo:robctrl_os_pd / os_invdyn）")
    print("=" * 72)
    res = run(quick=quick)
    print("-" * 72)
    print("   [%s] %s" % ("PASS" if res["check"]["passed"] else "FAIL",
                          res["check"]["threshold"]))
    ok = bool(res["check"]["passed"])
    if quick:
        print("⚠️ 冒烟模式（--quick）：level 与时长被压缩，判据结论无意义，退出码只看「无异常」。")
        ok = all(not r["error"] for r in res["rows"])
    print("结论：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
