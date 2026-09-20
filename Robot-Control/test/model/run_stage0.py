# -*- coding: utf-8 -*-
"""阶段 0 一键自检与归档（计划 §5.0.5 的**判据表**与 `res/model_stage0/` 产出的唯一入口）。

依次在**同一进程内**调用三个测试模块的 `run()`（不用 `subprocess`：① 避免管道在受限沙箱下被拦；
② 三个模块的 `run()` 已返回结构化结论，直接合并即可）：

| 顺序 | 模块 | 对应判据 |
|---|---|---|
| 1 | `test_model_sync.py`（0.4） | 判据 1 副本同步对拍 |
| 2 | `test_jacobian_dot.py`（0.1） | 判据 2 `J̇a` 矩阵一致性、判据 3 乘积法则 |
| 3 | `test_convention.py`（0.2+0.3） | 判据 4 重力项、判据 5 静平衡、判据 6 `C` 反对称性 |

产出（见 `res/README.md` 的归档契约）：

- `res/model_stage0/result.json` —— 机读结果：环境/版本、六个判据的阈值与实测值、
  **判据 5 的修订记录**、各模块的完整明细（报告里每个数字都能追到这）；
- `res/model_stage0/run.log` —— 本次运行的完整 stdout（含参数、进度与实测数字）。

用法（`py311-gym`，任意 cwd）：

    python Robot-Control/test/model/run_stage0.py

退出码：6 个判据全 PASS 为 0，否则 1。
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
_RC = os.path.abspath(os.path.join(_HERE, "..", ".."))          # Robot-Control/
sys.path.insert(0, _RC)
sys.path.insert(0, os.path.join(_RC, "code"))
sys.path.insert(0, _HERE)                                       # 三个测试模块同目录

import test_convention                                          # noqa: E402
import test_jacobian_dot                                        # noqa: E402
import test_model_sync                                          # noqa: E402

# (判据号, 判据名, 来源模块, 该模块 checks 里的键)
JUDGE = [
    ("1", "副本同步对拍", "test_model_sync", "sync"),
    ("2", "J̇a 矩阵一致性（最小相对误差 <1e-6 且 O(h²)）", "test_jacobian_dot", "jacobian_dot_matrix"),
    ("3", "J̇a 乘积法则（相对误差 <1e-6）", "test_jacobian_dot", "jacobian_dot_product"),
    ("4", "重力项与 ∂U/∂q 一致（相对误差 <1e-10）", "test_convention", "gravity_du"),
    ("5", "静平衡（**已按实测修订为 5a–5d**）", "test_convention", "static_equilibrium"),
    ("6", "C 的反对称性（偏差与差分阶一致、无结构性错误）", "test_convention", "c_antisymmetry"),
    ("7", "积分器收敛阶（`rk4_step` 必须四阶；2026-09-20 为一个真 bug 立）",
     "test_convention", "integrator_order"),
]

CRITERION_REVISIONS = [
    {
        "criterion": "判据 5 静平衡测试",
        "original": "取 τ = g(q)、F_f = 0，仿 10 s；‖q̇‖_∞ < 1e-10，q 无漂移（<1e-9 rad）",
        "problem": "原文隐含假设任意位形上的 τ=g(q) 都是「不漂移」的平衡点，但平衡点有稳定/不稳定之分。"
                   "标称位形是**不稳定**重力平衡（B⁻¹∂²U/∂q² 有负特征值），机器 eps（~1e-14）会按 "
                   "e^{sqrt(|λ|)t} 指数放大：实测 6R 10 s 后 max‖q̇‖ ~ 1e40、2R ~ 1.8e1。"
                   "实测增长率 6.722 /s（6R）、4.038 /s（2R）与线性化预测 sqrt(|λ_min|) = 6.677 /s、4.016 /s "
                   "一致（偏差 0.68% / 0.55%），证明发散是**物理不稳定**而非模型/实现错误。"
                   "故该判据在标称位形上**恒为假**，任何正确实现都会「失败」。",
        "revised": "拆成 4 项，阈值一个字未动："
                   "5a 瞬时平衡残差（任意位形 τ=∂U/∂q 时 ‖q̈‖_∞<1e-12，此处 g≠0，才真正检验符号/量级）；"
                   "5b 10 s 静平衡**移到稳定平衡位形 Q_EQ**（U 的极小点，前提 ‖∂U/∂q‖_∞<1e-12 且 λ_min≥-1e-9），"
                   "仍要求 ‖q̇‖_∞<1e-10、‖q−q_eq‖_∞<1e-9 rad；"
                   "5c 灵敏度对照（τ=0 / −g / 0.99g 的 ‖q̈₀‖_∞ 必须 >1e-9）证明测试有鉴别力；"
                   "5d 不稳定平衡位形的实测增长率须与线性化预测一致（<5%），把「发散」转为正向证据。",
        "note": "5b 单独看偏弱（稳定平衡位形上 g(q_eq)≈0，故 τ≈0，检不出符号错误）；重力符号的真正检验由 5a 与判据 4 承担。",
    },
    {
        "criterion": "判据 1 的数值对拍实现（非阈值修订）",
        "original": "`sys.path.insert(0, Codes/...)` + `import code_dynamics as dyn_orig` 取原件",
        "problem": "`model/__init__.py` 已把 `code_dynamics` 等裸名注册进 `sys.modules`（指向**副本**），"
                   "故 `import code_dynamics` 命中缓存、拿到的是副本本身——数值对拍退化为**自证**"
                   "（此前记录的 `max|Δ| = 0.000e+00` 由此而来）。哈希对拍不受影响，仍然有效。",
        "revised": "改用 `importlib` 以独立模块名从 `Codes/` 加载原件，加载期间把裸名临时指向原件，"
                   "使其内部 `from code_fk import ...` 绑定到原件；加载后校验 `__file__` 落在 `Codes/`。"
                   "现为真对拍（7 项数值量全部 max|Δ| = 0，含新增的 `robot_ik_lm`）。",
    },
    {
        "criterion": "判据 7 积分器收敛阶（**2026-09-20 新增；为一个真 bug 立的守卫**）",
        "original": "（原计划没有这一条）",
        "problem": "`rk4_step` 的 `q` 更新权重写成了 `(a1 + 2a2 + a3)/6`——**多算了 `a2`**。"
                   "由增广一阶系统 `ẏ = [q̇; a(q,q̇)]` 上的标准 RK4 可推出正确组合是 `(a1 + a2 + a3)/6`"
                   "（`q̇` 的 `(a1 + 2a2 + 2a3 + a4)/6` 本来就对）。错版的**全局精度只有一阶**："
                   "实测经验收敛阶 1.000（正确版 3.99）；同一算例在 h=1e-2 上的误差"
                   "6.198e-03 vs 4.581e-09（差 1.35e6 倍）。它同时把稳定域从 `hω*=2.785` 压到 `~1.0`。"
                   "影响面：阶段 2/3 的全部数值、以及阶段 4 判据 3 的失稳阈值"
                   "（实测 `hω*≈0.85` 与理论差 3.3× 的真因）。",
        "revised": "修 `code_sim.rk4_step` 与 `test_convention.rk4_step` 的 `q_new` 权重为 `(a1 + a2 + a3)/6`；"
                   "并在 `test_convention.check_integrator_order` 立守卫（经验阶必须 ∈ [3.5, 4.5]）。"
                   "阶段 2/3/4 的归档已在修正后整体重跑（旧归档留档为 `pre_rk4fix_*`）。",
    },
]


class _Tee:
    """把 stdout 同时写到控制台与日志文件（`print(..., flush=True)` 也照常透传）。"""

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
    info = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    for mod in ("scipy", "matplotlib"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:                                        # noqa: BLE001
            info[mod] = None
    try:
        import pinocchio                                          # noqa: F401
        info["pinocchio"] = getattr(pinocchio, "__version__", "installed")
    except Exception as exc:                                     # noqa: BLE001
        info["pinocchio"] = "未安装（%s）" % type(exc).__name__
    return info


def main():
    out_dir = os.path.join(_RC, "res", "model_stage0")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    json_path = os.path.join(out_dir, "result.json")

    t0 = time.perf_counter()
    module_results = {}
    with open(log_path, "w", encoding="utf-8", newline="") as log_f, \
            contextlib.redirect_stdout(_Tee(sys.stdout, log_f)):
        print("#" * 72)
        print("# 阶段 0：模型自检与约定锁定（一键运行 + 归档）")
        print("# 时间：%s" % datetime.datetime.now().isoformat(timespec="seconds"))
        print("# 环境：%s" % sys.executable)
        print("# 产出：%s" % out_dir)
        print("#" * 72)
        for mod in (test_model_sync, test_jacobian_dot, test_convention):
            print("\n" + "#" * 72)
            print("# >>> %s" % mod.__name__)
            print("#" * 72)
            t_mod = time.perf_counter()
            res = mod.run()
            res["wall_time_s"] = time.perf_counter() - t_mod
            module_results[mod.__name__] = res

        # ---- 汇总成计划 §5.0.5 的判据表 ----
        criteria = []
        for cid, name, mod_name, key in JUDGE:
            chk = module_results[mod_name]["checks"][key]
            criteria.append({
                "id": cid, "name": name,
                "source_module": mod_name, "source_test": key,
                "desc": chk["desc"], "threshold": chk["threshold"],
                "measured": chk["measured"], "passed": bool(chk["passed"]),
            })
        n_pass = sum(1 for c in criteria if c["passed"])
        elapsed = time.perf_counter() - t0
        passed = n_pass == len(criteria)

        summary = {
            "stage": 0,
            "name": "阶段 0 模型自检与约定锁定",
            "case": "model",
            "plan": "1-MN4R/docs/robot-control-verify/part4-motion-control-plan.md §5.0.5",
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "wall_time_s": elapsed,
            "env": _env_info(),
            "seed": test_model_sync.SEED,
            "criteria": criteria,
            "passed_count": n_pass,
            "total_count": len(criteria),
            "passed": bool(passed),
            "criterion_revisions": CRITERION_REVISIONS,
            "notes": [
                "判据 2/3 的被测实现：model/code_jacobian_dot.py（对 q 逐列中心差分加权，默认 h=1e-6；"
                "差分时**随 q 重算欧拉角**是易漏的一步）。",
                "判据 4/6 的参照：本文件之外的独立实现——复步微分（h=1e-20）∂B/∂q、∂U/∂q 与同版式 "
                "Christoffel 构造（见 test_convention.py，因 model/ 的 robot_B 预分配实数组不能吃复数）。",
                "判据 5b 的积分步长取 5e-2（静平衡是不动点，dt 只影响开销）：robot_dyn 单次 6R 实测 "
                "%s ms，dt=1e-3 的 10 s 仿真需 ~7e4 次调用 ≈ 11 min。" % round(
                    module_results["test_convention"]["timing_ms"]["6R"]["robot_dyn_ms"], 1),
                "约定缺口（记录基线，阶段 3 量化）：‖J_w q̇ − J_a q̇‖/‖J_w q̇‖ = %s（6R 标称位形）；"
                "书中偶写 ẋ_e = J_a q̇，严格说 [v; ω] = J_w q̇。" % round(
                    module_results["test_convention"]["conventions"]["6R"]["Jw_vs_Ja_gap_rel"], 4),
                "C 的差分误差实测：h=1e-6 时相对误差 ~1.5e-10（h=1e-5 处最优 ~7.5e-11，再小进入舍入平台）——"
                "即 robot_dyn 的默认 h 不是最优，但量级很小；是否升级为解析 Christoffel 留待阶段 3 用稳态误差数据决定。",
                "pinocchio 不装在 py311-gym（本文件 env 里记为未安装）：`pip install pin` 在 Windows 上没有轮子，"
                "阶段 1 用的 Pinocchio 4.1.0 由 conda-forge 装在独立的工作区本地 env "
                "D:\\Projects\\2024_MN4R\\.envs\\py311-pin（python 3.11.16 / numpy 2.4.6）。"
                "阶段 0 的全部判据均不依赖 pinocchio。",
            ],
            "module_results": module_results,
        }
        with open(json_path, "w", encoding="utf-8", newline="") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 72)
        print("阶段 0 判据表汇总（计划 §5.0.5）")
        print("=" * 72)
        for c in criteria:
            print("  [%s] 判据 %s  %s" % ("PASS" if c["passed"] else "FAIL", c["id"], c["name"]))
            print("         阈值：%s" % c["threshold"])
        print("-" * 72)
        print("阶段 0 总结论：%s（%d/%d）  用时 %.1f s"
              % ("PASS" if passed else "FAIL", n_pass, len(criteria), elapsed))
        print("归档：%s" % json_path)
        print("日志：%s" % log_path)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
