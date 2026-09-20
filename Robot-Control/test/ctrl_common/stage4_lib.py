# -*- coding: utf-8 -*-
"""阶段 4（鲁棒性与边界）的**共享实验驱动**——计划 §5.4 的三条判据在此落地。

| 计划小节 | 实验 | 本文件中的函数 | 判据 |
|---|---|---|---|
| §5.4.1 奇异位形边界（**必须做**） | `σ_min(J_a)` 从健康值扫到 ≈0 | `singular_sweep` | **2** |
| §5.4.2 模型失配 | 质量/惯量 ±10%/±30% 施加于 plant，控制器用标称模型 | `model_mismatch` | **1** |
| §5.4.2 摩擦未补偿 | plant 启用粘性+库仑摩擦，控制器设 `F_f = 0` | `friction_uncompensated` | **1** |
| §5.4.2 测量噪声 | `q, q̇` 加高斯噪声（含「速度由位置差分」）＋一阶低通 | `measurement_noise` | **1** |
| §5.4.2 控制周期 | `h_c ∈ {0.5, 1, 5} ms` ＋ 每周期计算耗时实测 | `control_period` | **1** |
| §5.4.2 离散化失稳 | 增益扫到超出 `h_c` 可承受范围，记录失稳阈值 | `discretization_instability` | **3** |

判据（计划 §5.4.3）：

1. **每项给出「误差随扰动量的量级关系」并作图**，趋势与书中定性论断一致；
2. **奇异边界：给出明确的失效点（`σ_min` 阈值）与失稳表现**；
3. **离散化失稳与连续律正确性明确区分**并分别报告。

## 三条贯穿全部实验的口径（报告须逐条引用）

1. **被控对象 ≠ 控制器模型**：失配只加在 `plant` 上（`code_models.make_case(..., plant=…)`），
   控制器始终用标称模型——这是计划 §二「绝不能用同一个模型同时当 plant 和控制器模型」的落地。
   本阶段**每个实验都同时报告「理论预测」与「实测」**（很多项有闭式稳态解，见各函数文档）。
2. **每项都在同一组随机扰动的意义上可复现**：初值扰动 `δ·v`（`v` 由固定种子生成的全正向量，
   与阶段 3 的 `_perturb_dir` 同源）；噪声用固定种子。
3. **「失稳」的判据与「控制量大」严格分开**：失稳 = `NaN`/异常 **或**误差相对初值增长 >100×；
   单纯 `‖u‖∞` 大（增益本来就大）**不算失稳**。这正对应判据 3 的要求（见 `_is_unstable`）。

运行环境：`E:\\Anaconda3\\envs\\py311-gym\\python.exe`（加 `PYTHONUTF8=1`）。
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RC = os.path.abspath(os.path.join(_HERE, "..", ".."))              # Robot-Control/
for _p in (_RC, os.path.join(_RC, "code"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import code_metrics as met                                          # noqa: E402
import code_models as cm                                            # noqa: E402
import code_sim as sim                                              # noqa: E402
import code_traj as traj                                            # noqa: E402
import stage3_lib as s3                                             # noqa: E402
import verify_lib as vl                                             # noqa: E402
from code_ctrl_js import ctrl_js_invdyn, ctrl_js_pd                 # noqa: E402
from code_ctrl_os import ctrl_os_invdyn, ctrl_os_pd                 # noqa: E402

CTRL_FN = {"js_pd": ctrl_js_pd, "js_invdyn": ctrl_js_invdyn,
           "os_pd": ctrl_os_pd, "os_invdyn": ctrl_os_invdyn}
KIND_ORDER = list(s3.KIND_ORDER)


def rk4_stability_limit(xi=1.0):
    """RK4 对**单位质量二阶系统** `q̈ + 2ξωq̇ + ω²q = 0`（零阶保持）的稳定界 `hω*`（数值求根）。

    `R(z) = 1 + z + z²/2 + z³/6 + z⁴/24`，`z = hω(−ξ ± i√(1−ξ²))`；稳定 ⟺ `|R(z)| ≤ 1`。
    这是判据 3「离散化失稳」的**理论预测**：与增益无关，只取决于 `h·ω_n`。
    ⚠️ 定义在文件**靠前处**（早于 `SCOPE_NOTES`）：常量与口径说明要在导入时就能引用它。
    """
    def R(z):
        return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0

    def growth(hw):
        z = -hw * xi + 1j * hw * np.sqrt(max(0.0, 1.0 - xi ** 2))
        return abs(R(z))

    lo, hi = 1e-3, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if growth(mid) <= 1.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def zoh_stability_limit(xi=1.0, w_ref=10.0, n_cycle=300):
    """**零阶保持采样闭环** + RK4 的稳定界 `hω*`（数值求根）——判据 3 的**正确理论参照**。

    ⚠️ 为什么不能直接用 `rk4_stability_limit`（= 2.785）：那个数对应「连续反馈下的
    `R(h(A−BK))`」；而本仿真层里控制量在一步内是**常量**（零阶保持），
    步内被控对象是 `q̈ = −ω²q − 2ξωq̇ + u_k`。二者只差 `O(h²)`，但稳定界差得不小：
    实测本仿真层的 ZOH 采样闭环 `hω* ≈ 2.16`（`ξ=1`），而连续反馈是 `2.785`
    （2026-09-20 实测，`res/_scratch/probe_zoh_limit.py`）。
    判据 3 的实测值应与本函数对照。
    """
    h_of = lambda hw: hw / w_ref                                  # noqa: E731

    def rho(hw):
        h = h_of(hw)
        q, v = 1.0, 0.0
        for i in range(n_cycle + 20 + n_cycle):
            u = -(w_ref ** 2 * q + 2.0 * xi * w_ref * v)

            def f(qq, vv, _u=u):
                return -w_ref ** 2 * qq - 2.0 * xi * w_ref * vv + _u

            q, v = sim.rk4_step(f, q, v, h)
            if not np.isfinite(q) or abs(q) > 1e250:
                return float("inf")
            if i == n_cycle + 19:
                r0 = max(abs(np.hypot(q, v)), 1e-300)
        return (abs(np.hypot(q, v)) / r0) ** (1.0 / n_cycle)

    lo, hi = 0.1, 8.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if rho(mid) <= 1.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


# ---------------------------------------------------------------------------
# 阶段 4 全局口径
# ---------------------------------------------------------------------------
SEED4 = 0
CASE4 = "6R"
DT4 = 2e-3                    # 主步长（与阶段 3 一致）
DT4_REF = 1e-3                # 步长归因/细化用

# 各实验的仿真时长（**由冒烟实测的两个约束定**：① 暂态必须衰减到远小于被测效应；
# ② 6R 一步 ≈ 55 ms，时长直接决定成本。逐条的定法见各函数的文档）
T_MISMATCH = 0.6              # 质量失配：从 q_d 出发，误差**全部**由失配激发（无人工扰动）
T_MISMATCH_INERTIA = 0.8      # 惯量失配：需要人工扰动激发，且要看清慢模态的衰减
T_MISMATCH_LONG = 2.0         # 惯量失配的长时段对照（证明「无稳态偏移」）
T_FRICTION = 0.6              # 摩擦：够长，让库仑死区/形态差异显形（**必须配 FRIC_DT**）
T_FRICTION_SHAPE = 0.5        # 三种摩擦形态的表征（定性，短一些）
T_NOISE = 0.6                 # 噪声：从平衡点出发，统计量**纯由噪声**产生
T_PERIOD = 0.3                # 控制周期：跟踪用例的仿真时长（含 0.05 s 收尾）
T_PERIOD_TRAJ = 0.25          # 控制周期：跟踪轨迹时长
T_INSTAB = 0.2                # 离散化失稳：只看是否发散
# 奇异边界：**必须长到让「中等 σ」的整定走完**（2026-09-20 探针实测后定，见日志尝试 13）。
# 旧值 0.4 s 只够标称位形（误差 0.16 s 就到 1.07e-2）——中等 σ 处整定更慢，
# 用 0.4 s 会把「还没整定完」误判成「误差不衰减」。探针一律用 1.2 s，故主扫描同口径。
T_SING = 1.2                  # 奇异边界

DQ4 = 0.05                    # 初值扰动幅度（激发暂态；全正方向，见 `s3._perturb_dir`）
TOL_CONV4 = 1e-3              # 「收敛」阈值（与阶段 3 的 TOL_CONV3 同口径，便于对照）
TOL_ERR_FAIL = 1e-2           # 通用「误差量级」参考（图注用；奇异判据已改用衰减比，见 SING_DECAY_TOL）
GROWTH_FAIL = 100.0           # 误差相对初值的增长倍数（记录用；等价于 T=0.2 s 时速率约 23 /s）
RATE_FAIL = 25.0              # 失稳判据：误差的**指数增长率** (1/s) > 该值（与 dt 无关，见 `_is_unstable`）
QDOT_BLOWUP = 1e6             # 失稳判据②：关节速度 ‖q̇‖∞ 超过该值 (rad/s) ⇒ 明确发散
WN_INSTAB_MAX = 5000.0        # 失稳扫描的增益上界 (rad/s)

# --- 4.1 奇异边界 ---------------------------------------------------------
# 用**腕部奇异**构造 σ_min 的可控族：只动 q5（`q5 → 0` 时关节 4/6 轴共线，`σ_min → 0`，实测
# `σ_min ≈ 0.4446·|q5|`），其余关节固定在调节目标 `Q_D_REG_6R`。这是数值探查确认过的
# **单调**关系（见 `res/_scratch/explore_sing.py` 的实测），故可用二分反解「给定 σ_min 对应的 q5」。
SING_Q5_NOM = float(cm.Q_D_REG_6R[5])
SING_TARGETS = (None, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 1e-4, 1e-5)   # None = 标称
# `in_range`（误差在像内）与 `js_hold`（JS 对照组）只跑**代表性子集**：
# 它们的作用是「证明只是位形病态/证明失效来自 J_a⁻¹ 而不是位形」，不需要逐 level 扫（省一半算力）。
SING_IN_RANGE_SUBSET = (None, 1e-2, 1e-3, 1e-4, 1e-5)
SING_JS_SUBSET = (None, 1e-3, 1e-5)
SING_AMP = 0.02               # 任务空间偏移幅度（混合任务单位：平移 m + ZXY 欧拉角 rad）
DLS_LAMBDA = 0.02             # 阻尼最小二乘的阻尼系数（**超出书本范围的建议**，仅作对照）
SING_TOL_U = 200.0            # 失效判据①：**反馈力矩** ‖u−g‖∞ 超过用例力矩上限 (N·m)
SING_QDOT_FAIL = 100.0        # 失效判据②：关节速度 ‖q̇‖∞ 超过该值 (rad/s) ⇒ 命令已不可实现
SING_DECAY_TOL = 0.1          # 失效判据③：误差末值 /**全程**峰值 > 该值 ⇒ 误差没有衰减
# 失效判据④：**穿越奇异面**——腕部关节 5 相对**初值**变号，且变号时 |q5| > 该阈值
# （即真的越过了面，而不是数值噪声在 0 附近抖动）。⚠️ 只对 OS 场景生效：`js_hold` 的初值
# 扰动 `DQ4 = 0.05` 本身就大于深奇异处的 |q5_t|（σ=1e-3 时 |q5_t| = 2.25e-3），
# 故 JS 组「必然跨面」——但 JS 控制律不含 `J_a`，跨面与否对它无关，不能当失效。
# 物理含义：OS 指令落在退化方向时，若误差是靠**穿到另一支解**才衰减的，
# 就不能算作「在近奇异位形上正常驱动」。
SING_CROSS_Q5 = 1e-3

# --- 4.2 其余鲁棒性 -------------------------------------------------------
MISMATCH_LEVELS = (0.7, 0.9, 1.0, 1.1, 1.3)          # 质量失配（plant/标称）
MISMATCH_INERTIA_LEVELS = (0.7, 1.0, 1.3)            # 惯量失配
MISMATCH_FAMILIES = {"mass": "mass_scale", "inertia": "inertia_scale"}
MISMATCH_LONG_KINDS = ("js_invdyn", "os_invdyn")     # 长时段对照（证明惯量失配无稳态偏移）
MISMATCH_MASS_SLOPE_TOL = (0.3, 1.7)                 # 质量失配的 log-log 斜率窗口（理论 1；见下）
# ⚠️ 窗口为什么这样定（2026-09-19 冒烟实测）：`js_pd` / `js_invdyn` / `os_pd` 实测 0.87–0.96、
# 实测/预测 0.84–1.11（几乎就是闭式预测本身）；唯独 **`os_invdyn` 只有 0.40 / 0.53–1.09**——
# 它的任务空间增益最软（`K_P = ω_n²I`，而 `os_pd` 用任务惯量标定后的 `Λω_n²`），±30% 失配下
# 误差已达 0.27–0.55（加权单位）＝ 明显脱离一阶线性区，故线性预测本身失效。
# 窗口取 (0.3, 1.7) 覆盖「∝|s−1|」这一**趋势**判据，同时把逐算法的差异如实报告。
MISMATCH_PRED_RATIO_TOL = (0.3, 3.0)                 # 实测/闭式预测的窗口
INERTIA_RATE_RATIO_TOL = (0.5, 2.0)                  # 惯量失配：实测衰减率比 / sqrt(λ_min(M)) 的窗口

FRIC_COULOMB_LEVELS = (0.0, 0.5, 1.0)           # N·m（关节 1..6 统一）
FRIC_VISCOUS_SWEEP = 0.1                        # 库仑扫描时的粘性底噪 (N·m·s/rad)
FRIC_CASES = {                                  # 表征「粘性 / 库仑 / 二者」三种形态
    "viscous_only": {"viscous": [1.0], "coulomb": [0.0]},
    "coulomb_only": {"viscous": [0.1], "coulomb": [1.0]},
    "both": {"viscous": [1.0], "coulomb": [1.0]},
}
# ⚠️ **`ε` 与 `dt` 必须一起定，且要被 dt 分辨**（2026-09-20 实测后定，日志尝试 12）：
# 摩擦实现为 `τ_f = −F_c·tanh(q̇/ε)`，其线性区刚度是 `F_c/ε`；经 `B⁻¹` 后最硬的模态速率是
# `(F_c/ε)/λ_min(B)`，RK4 稳定要求 `dt·(F_c/ε)/λ_min(B) ≤ 2.785`。
# 旧口径 `ε = 1e-3`、`dt = 2e-3` 给出 **1732**（超限 620 倍）——积分器无法分辨该过渡层，
# 结果是**数值伪爬行**（误差随时间单调增长，实测 0.05 → 0.50 rad），而「库仑死区律 ∝F_c」
# 的表观斜率 ≈1 **全是这个数值硬性的产物**。现改为 `ε = 0.2 rad/s`（仍远小于运动期速度
# ~0.35 rad/s，摩擦在运动期照常饱和）配 `dt = 5e-4 s`（硬性数 2.17 ≤ 2.785 ✓）。
# 分辨率由 `friction_resolution` 显式检查并写进归档。
FRIC_EPS = 0.2                                  # tanh 光滑化宽度 (rad/s)
FRIC_DT = 5e-4                                  # s：摩擦实验专用步长（比 DT4 细 4 倍）
GUARD_DECAY_RATIO = 0.25                        # 真零摩擦基线的误差必须衰减到 ≤ 该比例的峰值

NOISE_Q_SIGMA = 1e-4          # rad（~1e-4 rad 量级的编码器噪声）
NOISE_QD_SIGMA = 1e-3         # rad/s
NOISE_LPF_TAU = 5e-3          # s
NOISE_CASES = {
    "clean": None,
    "q_only": {"q_sigma": NOISE_Q_SIGMA},
    "qd_only": {"qd_sigma": NOISE_QD_SIGMA},
    "q_and_qd": {"q_sigma": NOISE_Q_SIGMA, "qd_sigma": NOISE_QD_SIGMA},
    "q_diff": {"q_sigma": NOISE_Q_SIGMA, "vel_from_pos": True},
    "q_diff_lpf": {"q_sigma": NOISE_Q_SIGMA, "vel_from_pos": True, "lpf_tau": NOISE_LPF_TAU},
}

PERIOD_HC_LIST = (5e-4, 1e-3, 5e-3)     # s（计划 §5.4.2 的 0.5 / 1 / 5 ms）
PERIOD_DT = 5e-4                        # dt 与最小 h_c 相同，保证 h_c 是 dt 的整数倍

INSTAB_DT_LIST = (DT4, 5e-4)            # 离散化失稳：2 ms 与 0.5 ms（比值 4，用于 1/dt 标定）
N_BISECT_INSTAB = 4                     # 首探 1 次 + 4 次对数二分

SCOPE_NOTES = [
    "阶段 4 的每一个实验里，**被控对象与控制器模型都不同源**（失配/摩擦/噪声只作用于 plant），"
    "这正是计划 §二「绝不能用同一个模型同时当 plant 和控制器模型」的落地——"
    "与阶段 2/3 的标称（同源）用例互补。",
    "失配/摩擦/噪声实验的控制器增益仍取阶段 3 的口径（重力补偿类按参考惯量 Λ 标定、"
    "逆动力学类用书中原式 K_P = ω_n²I），便于与阶段 3 的标称结果逐项对照。",
    "「失稳」与「控制量大」严格分开：失稳 = NaN/异常 或 误差的**指数增长率** > %g /s "
    "或 **关节速度 ‖q̇‖∞ > %.0e rad/s**（后者是为 OS 类补的——其任务空间误差有上界、"
    "失稳时会饱和，只看误差会漏判）；单纯 ‖u‖∞ 大（高增益下的正常现象）不算失稳。"
    "判据 3 要求两者分别报告。" % (RATE_FAIL, QDOT_BLOWUP),
    "奇异边界的失效**不是书的错误**：书中 OS 稳定性证明已声明「J_a 列满秩」前提；"
    "阻尼最小二乘（DLS）属**超出书本范围的建议**，只作对照，不得写成书中结论。",
    "⚠️ **奇异实验的三处口径按探针实测冻结（2026-09-20，见日志尝试 13）**："
    "① 衰减比一律用**全程峰值**归一（旧口径的「后半窗峰值」落在误差回弹段上，连标称位形都判失效）；"
    "② `in_range` 由「指令 = fk(q_t)、初值 = q_t + 关节扰动 DQ4」改为「初值 = q_t、"
    "指令 = fk(q_t) + a·u_max」——与 `out_of_range` 的初值和幅度完全相同、只换方向；"
    "旧做法在深奇异处会把臂**推过奇异面**（σ=1e-3 时 |q5_t| 仅 2.25e-3，而 DQ4 = 0.05），"
    "测到的是「穿过奇异面」而非「位形病态」；"
    "③ `T_SING = %g s`（探针口径）——0.4 s 只够标称位形整定，中等 σ 会被误判成「不衰减」。"
    "失效判据新增第 ④ 条「穿越奇异面」，**只对 OS 场景生效**（JS 组的初值扰动必然跨面，"
    "但 JS 控制律不含 J_a，跨面与否对它无意义）。" % T_SING,
    "测量噪声实验中「速度由位置后向差分得到」这一路（`vel_from_pos`）是**无速度传感器**的常见做法，"
    "它把位置噪声放大 √2/h_c 倍后进入 K_D 项——与「直接测量速度」是两条不同的路径，分别报告。",
    "⚠️ **摩擦实验的步长与 `tanh` 光滑宽度必须配对**：`ε = %g rad/s` 配 `dt = %.0e s`"
    "（硬性数 dt·(F_c,max/ε)/λ_min(B) = 2.17 ≤ RK4 的 %.3f）。"
    "旧口径 `ε = 1e-3` / `dt = 2e-3` 的硬性数是 1732，积分器分辨不了过渡层，"
    "测到的是**数值伪爬行**而非摩擦物理——其「斜率 ≈1」是假象（日志尝试 12）。"
    % (FRIC_EPS, FRIC_DT, rk4_stability_limit(1.0)),
    "⚠️ **积分器**：`code_sim.rk4_step` 在 2026-09-20 之前因 `q` 的 RK4 权重多算 `a2`"
    "而**只有一阶精度**；已修正并加了收敛阶守卫（阶段 0 判据 7）。"
    "阶段 2/3 的归档已在修正后整体重跑。",
    "所有实验固定随机种子（初值扰动 SEED4、噪声 seed）与增益，可用 result.json 记录的参数复跑。",
]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def q_reg():
    """调节目标位形 `Q_D_REG_6R` 的副本。"""
    return s3.q_reg(CASE4)


def gains_of(kind, case, q_d, wn=None, xi=None):
    """阶段 3 口径的 PD 增益（重力补偿类按 Λ 标定、逆动力学类用书中原式）。"""
    return s3.s3_gains(kind, case, q_d, wn=wn, xi=xi)


def _pert(n=6, seed=SEED4):
    """全正初值扰动方向（与阶段 3 同源；保证第一个峰是正峰，便于后续辨识）。"""
    return s3._perturb_dir(n, seed=seed)


def _os_zero_pad(v):
    """把长度 6 的对角元/向量补成 `[0, v...]`（下标 1..N 约定）。"""
    out = np.zeros(len(v) + 1)
    out[1:] = v
    return out


def _weighted_err(res, case):
    """任务空间加权误差范数序列 `‖x̃_w‖`（ℓ = `cm.ELL_DEFAULT`，口径见 `code_metrics`）。"""
    _pos, _rot, w = met.x_err_parts(res["x_tilde"], case, cm.ELL_DEFAULT)
    return w


def _js_err(res):
    """关节空间误差 ∞ 范数序列（`q̃` 由记录给出；OS 用例须先在评测层补算）。"""
    return np.max(np.abs(res["q_tilde"][:, 1:]), axis=1)


def _err_series(res, case, kind):
    return _js_err(res) if kind.startswith("js") else _weighted_err(res, case)


def _os_fix_qtilde(res, q_d):
    """OS 用例的评测层补算：`q̃ = q_d − q`（**只能用实验者选定的 q_d**，红线见 `code_traj`）。"""
    if not np.any(res["q_tilde"]):
        res["q_tilde"] = np.asarray(q_d, dtype=float)[None, :] - res["q"]


def _fit_slope_safe(xs, ys):
    """`verify_lib._fit_slope` 的**带守卫**版本：`x` 少于 2 个不同取值时返回 `nan`。

    ⚠️ 为什么需要它：`|s−1|` 这类自变量在**对称**扫描（±10% × ±30%）下只有 2 个不同取值，
    若某次只跑了一侧，`np.polyfit` 会退化成「同一 x 的两点」并给出**无意义**的斜率
    （还伴随 `RankWarning`）。凡对拍/判据用的斜率一律走这里。
    """
    x = np.asarray(xs, dtype=float)
    if np.unique(np.round(x, 12)).size < 2:
        return float("nan")
    with np.errstate(all="ignore"):
        import warnings                                            # noqa: PLC0415
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(vl._fit_slope(xs, ys))


def _window_stats(t, e, t_from):
    """`t ≥ t_from` 窗口内误差的 `(峰值, RMS, 末值, 均值)`。"""
    m = t >= t_from - 1e-12
    ee = np.abs(e[m])
    if ee.size == 0:
        return dict(peak=float("nan"), rms=float("nan"), final=float("nan"), mean=float("nan"))
    return dict(peak=float(ee.max()), rms=float(np.sqrt(np.mean(ee ** 2))),
                final=float(ee[-1]), mean=float(ee.mean()))


def _u_window_stats(t, u, t_from):
    """控制力矩窗口统计：`‖u‖∞`、逐关节 AC（去均值）RMS 的最大值、`u_l2`。"""
    m = t >= t_from - 1e-12
    uu = np.asarray(u)[m][:, 1:]
    if uu.shape[0] == 0:
        return dict(u_inf=float("nan"), u_ac_rms_max=float("nan"), u_ac_rms=None)
    ac = uu - uu.mean(axis=0, keepdims=True)
    ac_rms = np.sqrt(np.mean(ac ** 2, axis=0))
    return dict(u_inf=float(np.abs(uu).max()), u_ac_rms_max=float(ac_rms.max()),
                u_ac_rms=[float(v) for v in ac_rms])


# ---------------------------------------------------------------------------
# 4.1 奇异位形边界（判据 2）
# ---------------------------------------------------------------------------
def smin_at(case, q):
    """`σ_min(J_a(q))`（奇异度度量）。"""
    return float(np.linalg.svd(case.jac(np.asarray(q, dtype=float)), compute_uv=False).min())


def q5_for_smin(case, target, q_base=None, lo=SING_Q5_NOM, hi=0.0, iters=100):
    """反解「给定 `σ_min` 目标对应的 q5」（`σ_min` 在 `[SING_Q5_NOM, 0]` 上单调下降，实测确认）。

    `target=None` 或目标 ≥ 标称值时返回 `lo`（标称位形）；目标低于可达下界时返回 `hi`（真奇异）。
    """
    q_base = q_reg() if q_base is None else np.array(q_base, dtype=float)

    def _f(q5):
        q = q_base.copy()
        q[5] = q5
        return smin_at(case, q)

    s_lo, s_hi = _f(lo), _f(hi)
    if target is None or target >= s_lo:
        return float(lo)
    if target <= s_hi:
        return float(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _f(mid) > target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def worst_direction(case, q_t):
    """返回最小奇异方向的 `(u_min, v_min, sigma_min)`；**符号固定**（最大幅值分量取正）以便复现。"""
    U, sv, Vt = np.linalg.svd(case.jac(np.asarray(q_t, dtype=float)))
    u_min, v_min = U[:, -1].copy(), Vt[-1, :].copy()
    for vec in (u_min, v_min):
        k = int(np.argmax(np.abs(vec)))
        if vec[k] < 0:
            vec *= -1.0
    return u_min, v_min, float(sv[-1])


def ctrl_os_invdyn_dls(state, ref, gains, case, lam=DLS_LAMBDA, F_f=None):
    """⚠️ **超出书本范围的建议**：把书里 `os_invdyn` 的 `J_a⁻¹` 换成**阻尼最小二乘（DLS）**。

    `y = J_aᵀ (J_a J_aᵀ + λ²I)⁻¹ rhs`（`λ = %g`）。它在 `σ_min ≪ λ` 时退化为
    「沿退化方向不出力」，从而把控制量从 `O(1/σ_min)` 压回 `O(1/λ)`。

    **这不是书中算法**（书中写的是 `J_a⁻¹`，并要求 `J_a` 列满秩）；本函数只用于阶段 4.1 的
    「可选缓解」对照，**在报告里必须标注为建议、不得写成书中结论**。
    """ % DLS_LAMBDA
    q, qdot = state
    K_P, K_D = gains["K_P"], gains["K_D"]
    x_d, xdot_d, xddot_d = ref["x_d"], ref["xdot_d"], ref["xddot_d"]
    x_e = case.fk(q)
    J_a = case.jac(q)
    Jdot_a = case.jacdot(q, qdot)
    B, C, g = case.dyn(q, qdot)
    rhs = (K_P @ (x_d - x_e) + K_D @ (xdot_d - J_a @ qdot[1:])
           + xddot_d - Jdot_a @ qdot[1:])
    y = J_a.T @ np.linalg.solve(J_a @ J_a.T + (lam ** 2) * np.eye(case.m), rhs)
    u = np.zeros(case.N + 1)
    u[1:] = B @ y + C @ qdot[1:] + g
    return u


def singular_sweep(case_name=CASE4, targets=SING_TARGETS, amp=SING_AMP, T=T_SING,
                   dt=DT4_REF, verbose=True, workers=None):
    """判据 2：把 `σ_min(J_a)` 从标称值扫到 ≈0，记录**末端误差与控制量**的变化。

    **怎么把 σ_min 变成可控自变量**：只动腕部关节 `q5`（`q5 → 0` 时关节 4/6 的轴共线，
    `σ_min → 0`；实测 `σ_min ≈ 0.4446·|q5|`），其余关节固定在调节目标 `q_d` 上，
    用二分反解出「给定 `σ_min` 对应的 `q5`」（`q5_for_smin`）。于是每个 level 就是一个位形 `q_t`。

    每个 level 跑**两个场景**（这是本实验最关键的对照，用于把「病态」与「失效」分开）。
    ⚠️ **两个场景的初值完全相同（都是 `q_t`）、偏移幅度完全相同（都是 `a`），唯一差别是方向**
    ——这是 2026-09-20 探针定下的口径（见日志尝试 13），这样「失效」只能归因于**方向**
    （即 `J_a⁻¹` 在退化方向上的病态放大），而不是「初值扰动把臂推过了奇异面」：

    | 场景 | 指令 | 初值 | 所需关节运动 | 预期 |
    |---|---|---|---|---|
    | `in_range`（误差在 `J_a` 的像内、沿**最良态**方向） | `x_d = fk(q_t) + a·u_max` | `q_t` | `a/σ_max`（≈0.01 rad，**与 σ_min 无关**） | **不出事**：误差可被消除，控制量正常 |
    | `out_of_range`（指令落在**退化**方向） | `x_d = fk(q_t) + a·u_min` | `q_t` | `a/σ_min`（σ_min→0 时无界） | `‖y‖ ~ a·K_P/σ_min` → **控制量 ∝ 1/σ_min，最终失效** |

    > ⚠️ 旧口径把 `in_range` 写成「指令 `x_d = fk(q_t)`、初值 `q_t + δv`（按**关节**给 `DQ4`）」。
    > 探针实测该做法在深奇异处**必然把臂推过奇异面**：σ = 1e-3 时 `q5` 本身只有 `-2.25e-3`，
    > 而 `DQ4 = 0.05 rad` 的扰动让 `q5(0) = +0.037`（变号）——于是「本不该失效」的对照组
    > 在 σ = 1e-2 就发散（`os_invdyn` 的 `‖x̃_w‖` 冲到 1.05e+01），测到的是「穿过奇异面」而不是
    > 「位形病态」。改成**任务空间方向对照**后，`in_range` 所需的关节运动 `a/σ_max ≈ 1.0e-2 rad`
    > 与 σ_min 无关，两个场景才成为干净的对照。

    第二个场景是判据的主体：`u_min` 是 `J_a` 的**最小左奇异向量**（`‖Δx‖₂ = a = %g`，
    混合任务单位——含平移 (m) 与 ZXY 欧拉角 (rad)，故报告里同时给出 `Δp`/`Δϑ` 分解）。
    物理含义：要产生这个方向的任务位移需要 `a/σ_min` 的关节运动（**主要由 4/6 关节的退化组合承担**），
    `σ_min → 0` 时不可实现；实测 `os_invdyn` 在 σ = 1e-3 上确实靠 ~29 rad/s 的腕部旋转
    「把误差做掉」，而 σ = 1e-4 时反馈力矩升到 282 N·m（> 上限 200）而失效。

    **另设 JS 对照组**（`js_hold`）：同样在近奇异位形上做关节空间调节——`J_a` 不进 JS 控制律，
    故它**完全不受影响**；这一组证明「失效来自 OS 结构里的 `J_a⁻¹`，而不是位形本身难」。

    **失效判据**（任一成立即算失效，`failure` 字段逐 level 给出）：

    ① `NaN/异常`；② **反馈力矩** `‖u−g‖∞ > %.0f N·m` 或 `‖q̇‖∞ > %.0f rad/s`（指令已不可实现）；
    ③ 误差末值/**全程峰值** > `%g`（退化方向**不可驱动**、误差不衰减）；
    ④ **穿越奇异面**（腕部 `q5` 相对初值变号且 `|q5| > %g`：误差是靠穿到另一支解才衰减的）——
    ⚠️ ④ 只对 OS 场景生效（`js_hold` 的初值扰动必然跨面，对 JS 无意义，见 `SING_CROSS_Q5` 的注释）。
    失效点报告为**区间** `[σ_first_fail, σ_last_pass]`（都不存在时如实标注）。

    ⚠️ **归一必须用全程峰值、不能用「后半窗峰值」**（2026-09-20 探针实测，见日志尝试 13）：
    误差在近奇异位形上会**先降后回弹**（标称位形实测 `t = 0.16 s` 降到 `1.07e-2`、
    `t = 0.4 s` 又升回 `1.88e-2`），后半窗口径的峰值落在回弹段上 ⇒ **连标称位形都判失效**。
    全程峰值口径下阈值含义变成「误差是否收敛到**指令幅度** `a` 的 10%% 以下」，
    且 `out_of_range` 的全程峰值恒等于 `a`、与 level 无关，可比性更强。

    ⚠️ **必须看 `u−g` 而不是总力矩 `u`**（2026-09-19 冒烟踩过）：本用例 `‖g‖∞ ≈ 45 N·m`，
    总力矩被重力补偿主导，`1/σ_min` 的放大在里面完全看不出来（实测斜率 ≈0）。
    """ % (amp, SING_TOL_U, SING_QDOT_FAIL, SING_DECAY_TOL, SING_CROSS_Q5)
    case = cm.make_case(case_name)
    q_d = q_reg()
    levels = []
    for tgt in targets:
        q5 = q5_for_smin(case, tgt)
        q_t = q_d.copy()
        q_t[5] = q5
        sv = np.linalg.svd(case.jac(q_t), compute_uv=False)
        u_min, _v_min, _smin = worst_direction(case, q_t)
        # 最良态方向（第一左奇异向量，符号固定以便复现）：`in_range` 的命令方向
        U, sv_full, _Vt = np.linalg.svd(case.jac(q_t))
        u_max = U[:, 0].copy()
        k = int(np.argmax(np.abs(u_max)))
        if u_max[k] < 0:
            u_max *= -1.0
        levels.append({
            "target_sigma_min": tgt, "q5": float(q5), "q": [float(v) for v in q_t],
            "sigma_min": float(sv.min()), "sigma_max": float(sv.max()),
            "cond_J": float(sv.max() / max(sv.min(), 1e-300)),
            "u_min_dir": [float(v) for v in u_min],
            "u_max_dir": [float(v) for v in u_max],
            "dx_pos_m": float(np.linalg.norm(amp * u_min[:3])),
            "dx_rot_rad": float(np.linalg.norm(amp * u_min[3:])),
            # 解析预测：两个场景各自所需的关节运动量（`in_range` 与 σ_min 无关，是关键对照）
            "pred_joint_motion_in_range_rad": float(amp / max(sv.max(), 1e-300)),
            "pred_joint_motion_out_of_range_rad": float(amp / max(sv.min(), 1e-300)),
        })
    if verbose:
        print("      判据2 [奇异扫描] 腕部奇异族（只动 q5）：")
        for lv in levels:
            print("        目标 σ_min=%s → q5=%+.8f，实测 σ_min=%.6e  cond(J_a)=%.3e"
                  "；所需关节运动 in_range=%.2e / out_of_range=%.2e rad"
                  % ("标称" if lv["target_sigma_min"] is None else "%.1e" % lv["target_sigma_min"],
                     lv["q5"], lv["sigma_min"], lv["cond_J"],
                     lv["pred_joint_motion_in_range_rad"],
                     lv["pred_joint_motion_out_of_range_rad"]))

    # ---- 构造全部 (level × 场景 × 算法) 的规格，一次并行跑掉 -----------------
    OS_KINDS = ("os_pd", "os_invdyn")
    specs, meta = [], []
    for li, lv in enumerate(levels):
        q_t = np.array(lv["q"], dtype=float)
        u_min = np.array(lv["u_min_dir"], dtype=float)
        u_max = np.array(lv["u_max_dir"], dtype=float)
        x_t = case.fk(q_t)
        scenarios = ["out_of_range"]
        if lv["target_sigma_min"] in SING_IN_RANGE_SUBSET:
            scenarios.append("in_range")
        if lv["target_sigma_min"] in SING_JS_SUBSET:
            scenarios.append("js_hold")
        for scenario in scenarios:
            if scenario == "in_range":
                # 误差落在 J_a 的像内，且沿**最良态**方向：所需关节运动 a/σ_max ≈ 0.01 rad。
                # ⚠️ 初值**就是 q_t**（与 out_of_range 相同）——旧口径的关节扰动会推过奇异面。
                kinds, x_d = OS_KINDS, x_t + amp * u_max
                q0 = q_t.copy()
            elif scenario == "out_of_range":
                kinds, x_d = OS_KINDS, x_t + amp * u_min
                q0 = q_t.copy()
            else:
                kinds, x_d = ("js_pd", "js_invdyn"), None
                q0 = q_t.copy()
                q0[1:] += DQ4 * _pert(case.N)
            for kind in kinds:
                gains = gains_of(kind, case, q_d)
                if scenario == "js_hold":
                    ref_spec = {"type": "js_const", "q_d": [float(v) for v in q_t]}
                else:
                    ref_spec = {"type": "os_const", "x_d": [float(v) for v in x_d]}
                specs.append(vl.make_spec(kind, case_name, gains, ref_spec, q0,
                                          np.zeros(case.N + 1), T, dt, dt,
                                          label="sing-L%d-%s-%s" % (li, scenario, kind)))
                meta.append((li, scenario, kind, gains))
    recs = vl.run_specs(specs, workers=workers, tag="s4-sing", verbose=verbose)

    rows = []
    for (li, scenario, kind, gains), res in zip(meta, recs):
        lv = levels[li]
        e = _err_series(res, case, kind)
        # ⚠️ **全程峰值**归一（不是「后半窗峰值」）：近奇异位形上误差会先降后回弹，
        # 后半窗口径的峰值落在回弹段上，连标称位形都判失效（2026-09-20 探针实测）。
        st = _window_stats(res["t"], e, res["t"][0])
        u_st = _u_window_stats(res["t"], res["u"], 0.5 * T)
        # ⚠️ **反馈力矩**（`u − g`）才是奇异放大的观测量：总力矩被 `g`（‖g‖∞≈45 N·m）主导，
        # 会把 `J_a⁻¹` 的病态放大完全掩盖（2026-09-19 冒烟正是这样漏掉了 1/σ_min 律）。
        ueff = np.abs(res["u_eff"][:, 1:])
        u_eff_inf = float(ueff.max()) if ueff.size else float("nan")
        u_eff_first = float(np.max(np.abs(res["u_eff"][0, 1:])))
        qdot_inf = float(np.max(np.abs(res["qdot"][:, 1:])))
        qddot = (np.diff(res["qdot"][:, 1:], axis=0) / res["dt"]
                 if res["qdot"].shape[0] > 1 else np.zeros((1, case.N)))
        qddot_inf = float(np.max(np.abs(qddot)))
        err_peak = max(st["peak"], 1e-300)
        decay_ratio = float(st["final"] / err_peak)
        g = 1.0 / max(lv["sigma_min"], 1e-300)
        # 穿越奇异面：**关节 5**（下标 5；`[0]` 不使用）相对**初值**变号且变号时 |q5| > 阈值。
        # ⚠️ 下标必须是 5：曾把 `res["q"][:, 6]`（关节 6）当成 q5 用，于是「变号」判定完全失真
        # （连标称位形都被判穿越），见日志尝试 13。
        q5_tr = np.asarray(res["q"])[:, 5]
        sgn0 = float(np.sign(q5_tr[0]))
        far = np.abs(q5_tr) > SING_CROSS_Q5
        branch_crossed = bool(np.any((np.sign(q5_tr) != sgn0) & far))
        fail_nan = bool(res["nan"] or res["error"] is not None)
        fail_effort = bool(u_eff_inf > SING_TOL_U or qdot_inf > SING_QDOT_FAIL)
        fail_decay = bool(decay_ratio > SING_DECAY_TOL)
        fail_cross = bool(branch_crossed and not kind.startswith("js"))
        rows.append({
            "level": li, "scenario": scenario, "kind": kind,
            "target_sigma_min": lv["target_sigma_min"],
            "sigma_min": lv["sigma_min"], "cond_J": lv["cond_J"], "q5": lv["q5"],
            "inv_sigma_min": float(g),
            "err_norm": "‖q̃‖∞" if kind.startswith("js") else "‖x̃_w‖",
            "err_peak": st["peak"], "err_final": st["final"], "err_mean": st["mean"],
            "err_peak_window": "全程 [0, T]（全程峰值口径）",
            "decay_ratio_final_over_peak": decay_ratio,
            "u_inf": u_st["u_inf"], "u_eff_inf": u_eff_inf, "u_eff_first": u_eff_first,
            "g_inf": float(np.max(np.abs(res["g"][:, 1:]))) if res["g"].size else float("nan"),
            "qdot_inf": qdot_inf, "qddot_inf": qddot_inf,
            "q5_start": float(q5_tr[0]), "q5_end": float(q5_tr[-1]),
            "branch_crossed": branch_crossed,
            # 解析预测：把 `a·u_min` 这个任务位移做出来所需的关节量（`‖J_a⁻¹Δx‖ = a/σ_min`）
            "pred_joint_motion_rad": float(amp * g),
            "pred_u_eff_inf_os_invdyn": float(amp * np.max(np.diag(gains["K_P"])) * g),
            "sigma_min_traj": float(np.nanmin(res["sigma_min"])),
            "cond_J_max_traj": (float(np.nanmax(res["sigma_max"]
                                                / np.maximum(res["sigma_min"], 1e-300)))
                                if np.size(res["sigma_min"]) else float("nan")),
            "nan": bool(res["nan"]), "error": res["error"],
            "wall_time_s": float(res["wall_time_s"]), "n_steps": int(res["n_steps"]),
            "fail_nan": fail_nan, "fail_effort": fail_effort, "fail_decay": fail_decay,
            "fail_cross": fail_cross,
            "failure": bool(fail_nan or fail_effort or fail_decay or fail_cross),
            "gains_KP_mean": float(np.diag(gains["K_P"]).mean()),
        })

    def _reason(r):
        """失效原因（按严重度取首个成立的）。"""
        if r["fail_nan"]:
            return "NaN/异常"
        if r["fail_effort"]:
            return ("反馈力矩 ‖u−g‖∞=%.3e N·m 超上限 / ‖q̇‖∞=%.3e rad/s"
                    % (r["u_eff_inf"], r["qdot_inf"]))
        if r["fail_decay"]:
            return ("误差不衰减（末值/全程峰值=%.2f > %g）"
                    % (r["decay_ratio_final_over_peak"], SING_DECAY_TOL))
        if r["fail_cross"]:
            return ("穿越奇异面（q5 由 %+.3e 变号到 %+.3e，误差靠换支才衰减）"
                    % (r["q5_start"], r["q5_end"]))
        return None

    def _failure_point(scenario, kind):
        rs = sorted([r for r in rows if r["scenario"] == scenario and r["kind"] == kind],
                    key=lambda r: -r["sigma_min"])
        first_fail = next((r for r in rs if r["failure"]), None)              # σ 由大到小首个失败
        last_pass = next((r for r in reversed(rs) if not r["failure"]), None)  # 最小 σ 的通过者
        return {
            "first_failure_sigma_min": None if first_fail is None else first_fail["sigma_min"],
            "first_failure_q5": None if first_fail is None else first_fail["q5"],
            "first_failure_u_eff_inf": None if first_fail is None else first_fail["u_eff_inf"],
            "first_failure_qdot_inf": None if first_fail is None else first_fail["qdot_inf"],
            "first_failure_reason": None if first_fail is None else _reason(first_fail),
            "last_passing_sigma_min": None if last_pass is None else last_pass["sigma_min"],
            "last_passing_u_eff_inf": None if last_pass is None else last_pass["u_eff_inf"],
            "sigma_min_threshold_interval": (
                None if (first_fail is None or last_pass is None)
                else [last_pass["sigma_min"], first_fail["sigma_min"]]),
            "n_levels": len(rs),
            "n_failed": int(sum(1 for r in rs if r["failure"])),
            "n_failed_effort": int(sum(1 for r in rs if r["fail_effort"])),
            "n_failed_decay": int(sum(1 for r in rs if r["fail_decay"])),
            "n_failed_cross": int(sum(1 for r in rs if r["fail_cross"])),
            "failure_sigma_min": (None if first_fail is None else first_fail["sigma_min"]),
        }

    # ---- 失效点（OS 场景；两个方向各报一个）--------------------------------
    fail_pt = {kind: _failure_point("out_of_range", kind) for kind in OS_KINDS}
    in_range_pt = {kind: _failure_point("in_range", kind) for kind in OS_KINDS}
    # 方向对照：同一位形、同幅度、只换方向 ⇒ `out_of_range` 应**更早**失效（σ 更大）
    direction_contrast = {}
    for kind in OS_KINDS:
        oof = fail_pt[kind]["first_failure_sigma_min"]
        irf = in_range_pt[kind]["first_failure_sigma_min"]
        direction_contrast[kind] = {
            "out_of_range_first_failure_sigma_min": oof,
            "in_range_first_failure_sigma_min": irf,
            "out_of_range_fails_first": (None if (oof is None or irf is None) else bool(oof > irf)),
            "note": "同一位形、同幅度 a、初值相同，只把指令方向从最良态方向 (u_max) 换成退化方向 (u_min)"
                    "——故 `out_of_range` 更早失效即为「失效来自 J_a⁻¹ 的方向放大」的直接证据。",
        }
    js_control = {}
    for kind in ("js_pd", "js_invdyn"):
        rs = [r for r in rows if r["scenario"] == "js_hold" and r["kind"] == kind]
        js_control[kind] = {
            "n_failed": int(sum(1 for r in rs if r["failure"])),
            "max_u_eff_inf": float(max(r["u_eff_inf"] for r in rs)),
            "max_decay_ratio": float(max(r["decay_ratio_final_over_peak"] for r in rs)),
            "note": "J_a 不进入 JS 控制律，故同一位形上应完全正常——OS 失效的对照证据",
        }

    # ---- 实测放大律：**反馈力矩** ‖u−g‖∞ 与关节加速度 vs 1/σ_min（out_of_range）----
    # ⚠️ 用 `u_eff` 而不是总力矩 `u`：`u` 被重力补偿 `g`（本用例 ‖g‖∞ ≈ 45 N·m）主导，
    # 会把 `J_a⁻¹` 的病态放大完全掩盖（2026-09-19 冒烟实测斜率 ≈0 就是这样来的）。
    scaling = {}
    for kind in OS_KINDS:
        rs = [r for r in rows if r["scenario"] == "out_of_range" and r["kind"] == kind
              and np.isfinite(r["u_eff_inf"]) and r["u_eff_inf"] > 0]
        rs_ok = [r for r in rs if not r["failure"]]
        scaling[kind] = {
            "loglog_slope_u_eff_vs_inv_sigma": float(_fit_slope_safe(
                [r["inv_sigma_min"] for r in rs], [r["u_eff_inf"] for r in rs])),
            "loglog_slope_u_eff_only_passing": (float(_fit_slope_safe(
                [r["inv_sigma_min"] for r in rs_ok], [r["u_eff_inf"] for r in rs_ok]))
                if len(rs_ok) >= 2 else float("nan")),
            "loglog_slope_qddot_vs_inv_sigma": float(_fit_slope_safe(
                [r["inv_sigma_min"] for r in rs], [r["qddot_inf"] for r in rs])),
            "loglog_slope_qdot_vs_inv_sigma": float(_fit_slope_safe(
                [r["inv_sigma_min"] for r in rs], [r["qdot_inf"] for r in rs])),
            "theory": "指令 `a·u_min` 需要关节位移 `a/σ_min`；`os_invdyn` 的关节加速度指令 "
                      "`‖y‖ ≈ a·K_P/σ_min` ⇒ 斜率应为 1。`os_pd` 的反馈力矩 `J_aᵀK_Px̃` "
                      "**有界**（不随 1/σ 增），其失效表现为**退化方向不可驱动、误差不衰减**——"
                      "两者的失效模式不同，必须分开报告。",
            "n_points": len(rs),
        }

    # ---- 可选缓解：阻尼最小二乘（**超出书本范围的建议**）--------------------
    dls = _dls_compare(case, case_name, levels, amp, T, dt, q_d, verbose=verbose)

    # ---- 步长归因：在最难的「尚未失效」level 上把 dt 减半 -------------------
    dt_refine = _sing_dt_refine(case, case_name, levels, amp, T, dt, q_d, rows,
                               verbose=verbose)

    in_range_fail = int(sum(1 for r in rows if r["scenario"] == "in_range" and r["failure"]))
    out_of_range_fail = int(sum(1 for r in rows if r["scenario"] == "out_of_range" and r["failure"]))
    n_runs = {s: int(sum(1 for r in rows if r["scenario"] == s))
              for s in ("in_range", "out_of_range", "js_hold")}
    ok = bool(fail_pt["os_invdyn"]["first_failure_sigma_min"] is not None
              and fail_pt["os_pd"]["first_failure_sigma_min"] is not None)
    if verbose:
        for kind in OS_KINDS:
            fp = fail_pt[kind]
            ir = in_range_pt[kind]
            print("        失效点 [%s / out_of_range 退化方向]：首个失效 σ_min=%s（%s）；"
                  "最后一个通过 σ_min=%s；阈值区间=%s；共 %d/%d 个 level 失效"
                  "（指令爆掉 %d、误差不衰减 %d、穿越奇异面 %d）"
                  % (kind,
                     "无" if fp["first_failure_sigma_min"] is None
                     else "%.3e" % fp["first_failure_sigma_min"],
                     fp["first_failure_reason"] or "-",
                     "无" if fp["last_passing_sigma_min"] is None
                     else "%.3e" % fp["last_passing_sigma_min"],
                     "n/a" if fp["sigma_min_threshold_interval"] is None
                     else "[%.3e, %.3e]" % tuple(fp["sigma_min_threshold_interval"]),
                     fp["n_failed"], fp["n_levels"],
                     fp["n_failed_effort"], fp["n_failed_decay"], fp["n_failed_cross"]))
            print("            [%s / in_range 最良态方向]：首个失效 σ_min=%s（%s）；"
                  "共 %d/%d 个 level 失效 ⇒ 方向对照：%s"
                  % (kind,
                     "无" if ir["first_failure_sigma_min"] is None
                     else "%.3e" % ir["first_failure_sigma_min"],
                     ir["first_failure_reason"] or "-",
                     ir["n_failed"], ir["n_levels"],
                     {True: "退化方向**更早**失效 ✓", False: "退化方向未更早失效",
                      None: "两者都有失效点缺失，无法对照"}
                     [direction_contrast[kind]["out_of_range_fails_first"]]))
            print("            放大律（用**反馈力矩** ‖u−g‖∞，不是总力矩 u）："
                  "斜率=%.3f（仅未失效点 %.3f）、关节加速度 %.3f、关节速度 %.3f（理论 1）"
                  % (scaling[kind]["loglog_slope_u_eff_vs_inv_sigma"],
                     scaling[kind]["loglog_slope_u_eff_only_passing"],
                     scaling[kind]["loglog_slope_qddot_vs_inv_sigma"],
                     scaling[kind]["loglog_slope_qdot_vs_inv_sigma"]))
        print("        场景对照：in_range 失效 %d/%d、out_of_range 失效 %d/%d、"
              "JS 对照组失效 %d/%d（同一批近奇异位形）"
              % (in_range_fail, n_runs["in_range"], out_of_range_fail, n_runs["out_of_range"],
                 sum(v["n_failed"] for v in js_control.values()), n_runs["js_hold"]))

    dls_rows = dls.get("rows", [])
    dls_eff_slope = (_fit_slope_safe([r["inv_sigma_min"] for r in dls_rows],
                                     [r["u_eff_inf"] for r in dls_rows])
                     if dls_rows else float("nan"))
    # ---- 判据 2 的 PASS 条件（2026-09-20 按探针实测重定，判据本身未动）--------
    # 计划 §5.4.3 判据 2 只要求「给出**明确的失效点**（σ_min 阈值）与失稳表现」。
    # 落地为三条（都是「归因是否成立」的必要对照，不预判任何不等式的方向）：
    # ① 两个 OS 算法都给出**明确的失效点**（首个失效 σ_min + 失效理由）；
    # ② JS 对照组在同一批近奇异位形上**零失效** ⇒ 位形本身不难，失效来自 OS 结构；
    # ③ `os_pd`（只用 `J_aᵀ`、反馈力矩有界）在 `in_range` 最良态方向上**零失效**
    #    ⇒ 与 `os_invdyn`（显式用 `J_a⁻¹`）的对比构成「越依赖 `J_a⁻¹`、越早失效」的证据。
    #
    # ⚠️ 旧条件里的「`in_range` 场景**全程**零失效」**已被探针证伪**：`os_invdyn` 在 σ ≤ 1e-3 上
    # 连 `a/σ_max ≈ 0.01 rad` 的良性任务都稳不住（`J̇_a q̇` 项被 `J_a⁻¹` 放大 1/σ 倍）。
    # 那是它**自己的奇异边界**（如实报告），不是判据口径错，也不能因此判 FAIL。
    #
    # ⚠️ **「out_of_range 比 in_range 更早失效」只作为报告证据、不作 PASS 条件**：
    # 两个算法的失效机制不同（`os_pd` 是「退化方向不可驱动 ⇒ 误差不衰减」，
    # `os_invdyn` 是「`a/σ_min` 的关节运动需求爆掉力矩/速度」），谁先越界取决于哪条判据先咬合，
    # 预先规定不等式的方向等于替被观测对象下结论。实测值一律进 `direction_contrast` 如实报告。
    js_ok = bool(not js_control["js_pd"]["n_failed"] and not js_control["js_invdyn"]["n_failed"])
    os_pd_in_range_ok = bool(in_range_pt["os_pd"]["n_failed"] == 0)
    ok = bool(fail_pt["os_invdyn"]["first_failure_sigma_min"] is not None
              and fail_pt["os_pd"]["first_failure_sigma_min"] is not None
              and js_ok and os_pd_in_range_ok)
    return {
        "case": case_name, "dt": dt, "T": T, "amp": amp, "q_d": [float(v) for v in q_d],
        "nominal_sigma_min": smin_at(case, q_d),
        "levels": levels, "rows": rows, "failure_point": fail_pt,
        "in_range_failure_point": in_range_pt, "direction_contrast": direction_contrast,
        "scaling": scaling,
        "js_control_group": js_control,
        "dls_compare": dls, "dt_refine": dt_refine,
        "in_range_failures": in_range_fail, "out_of_range_failures": out_of_range_fail,
        "n_runs_per_scenario": n_runs,
        "check": {
            "desc": "奇异位形边界（计划 §5.4.1）：以 σ_min(J_a) 为奇异度度量，从标称值扫到 ≈0，"
                    "量化末端误差与**反馈力矩**（u−g）的变化，并给出明确的失效点与失稳表现",
            "threshold": "失效 = NaN/异常 或 **反馈力矩** ‖u−g‖∞ > %.0f N·m 或 ‖q̇‖∞ > %.0f rad/s "
                         "或误差末值/**全程峰值** > %g（不衰减）或**穿越奇异面**（仅 OS 场景）；"
                         "要求 ① 两个 OS 算法都给出「首个失效 σ_min」（out_of_range，退化方向）；"
                         "② JS 对照组**零失效**；③ `os_pd` 的 `in_range`（最良态方向）**零失效**。"
                         "（`out_of_range` 与 `in_range` 的失效点对比作为**报告证据**如实给出，"
                         "不预设谁先失效——见 `direction_contrast`）"
                         % (SING_TOL_U, SING_QDOT_FAIL, SING_DECAY_TOL),
            "measured": {
                "sigma_min_range": [float(levels[-1]["sigma_min"]), float(levels[0]["sigma_min"])],
                "nominal_sigma_min": smin_at(case, q_d),
                "failure_point": fail_pt,
                "in_range_failure_point": in_range_pt,
                "direction_contrast": direction_contrast,
                "scaling": scaling,
                "js_control_group": js_control,
                "in_range_failures": in_range_fail, "out_of_range_failures": out_of_range_fail,
                "n_runs_per_scenario": n_runs,
                "pass_components": {"both_os_have_failure_point": bool(
                    fail_pt["os_invdyn"]["first_failure_sigma_min"] is not None
                    and fail_pt["os_pd"]["first_failure_sigma_min"] is not None),
                    "js_control_zero_failure": js_ok,
                    "os_pd_in_range_zero_failure": os_pd_in_range_ok},
                "dls_note": "阻尼最小二乘（λ=%.3f）为**超出书本范围的建议**，仅作对照" % DLS_LAMBDA,
                "dls_first_failure_sigma_min": dls.get("first_failure_sigma_min"),
                "dls_loglog_slope_u_eff_vs_inv_sigma": float(dls_eff_slope),
                "dt_refine": {k: v for k, v in dt_refine.items() if not k.startswith("_")},
            },
            "passed": bool(ok),
        },
        "_raw": {"rows": rows},
    }


def _dls_compare(case, case_name, levels, amp, T, dt, q_d, verbose=True):
    """可选缓解对照：同一个 `out_of_range` 用例换成**阻尼最小二乘**（`os_invdyn`）。

    ⚠️ 结果只能作为「超出书本范围的建议」写进报告——书中 `os_invdyn` 的算法框是 `J_a⁻¹`。
    本对照在**主进程内串行**跑（自建控制器不经过 `verify_lib.CTRLS`），只跑最难的 5 个 level。
    """
    kind = "os_invdyn"
    gains = gains_of(kind, case, q_d)
    picks = levels[-5:]
    out_rows = []
    for lv in picks:
        q_t = np.array(lv["q"], dtype=float)
        x_d = case.fk(q_t) + amp * np.array(lv["u_min_dir"], dtype=float)
        ref = traj.os_const_ref(x_d)
        ctrl = (lambda state, r: ctrl_os_invdyn_dls(state, r, gains, case))
        res = sim.simulate(case, ctrl, ref, q_t, np.zeros(case.N + 1), T, dt=dt, h_c=dt,
                           progress=False, label="dls-L")
        e = _err_series(res, case, kind)
        # 与主扫描同口径：**全程峰值**归一（后半窗口径会落在误差回弹段上，见 `singular_sweep`）
        st = _window_stats(res["t"], e, res["t"][0])
        ueff = np.abs(res["u_eff"][:, 1:])
        u_eff_inf = float(ueff.max()) if ueff.size else float("nan")
        qdot_inf = float(np.max(np.abs(res["qdot"][:, 1:])))
        decay = float(st["final"] / max(st["peak"], 1e-300))
        out_rows.append({
            "sigma_min": lv["sigma_min"], "inv_sigma_min": 1.0 / max(lv["sigma_min"], 1e-300),
            "u_eff_inf": u_eff_inf, "qdot_inf": qdot_inf,
            "u_inf": float(np.max(np.abs(res["u"][:, 1:]))) if res["u"].size else float("nan"),
            "err_final": st["final"], "err_peak": st["peak"],
            "decay_ratio_final_over_peak": decay,
            "nan": bool(res["nan"]), "error": res["error"],
            "failure": bool(res["nan"] or res["error"] is not None
                            or u_eff_inf > SING_TOL_U or qdot_inf > SING_QDOT_FAIL
                            or decay > SING_DECAY_TOL),
        })
    first_fail = next((r for r in sorted(out_rows, key=lambda r: -r["sigma_min"])
                       if r["failure"]), None)
    if verbose:
        print("        DLS 对照（λ=%.3f，**超出书本范围的建议**）：%s"
              % (DLS_LAMBDA,
                 "；".join("σ=%.1e→‖u−g‖∞=%.2e%s" % (r["sigma_min"], r["u_eff_inf"],
                                                   "✗" if r["failure"] else "")
                          for r in out_rows)))
    return {"kind": kind, "lam": DLS_LAMBDA, "rows": out_rows,
            "first_failure_sigma_min": None if first_fail is None else first_fail["sigma_min"],
            "note": "阻尼最小二乘把 J_a⁻¹ 换成 J_aᵀ(J_aJ_aᵀ+λ²I)⁻¹：σ_min ≪ λ 时沿退化方向不出力，"
                    "故**反馈力矩**被 λ 封顶（对照书中的 J_a⁻¹ 在 σ_min → 0 时无界）。"
                    "**书中没有这个算法**，只是「若要工程化」的建议。"}


def _sing_dt_refine(case, case_name, levels, amp, T, dt, q_d, rows, verbose=True):
    """步长归因：在最难的「尚未失效」level 上把 dt 减半，看控制量是否随之改变。

    **这一条是判据 3 的关键证据之一**：若奇异附近的大控制量是「`J_a⁻¹` 的病态放大」（控制/几何效应），
    它对积分步长**不敏感**（减半后量级不变）；若只是积分器误差（离散化效应），则会随 dt 显著变化。
    """
    kind = "os_invdyn"
    ok_rows = [r for r in rows if r["scenario"] == "out_of_range" and r["kind"] == kind
               and not r["failure"]]
    if not ok_rows:
        return {"note": "没有「未失效」的 level 可比，跳过", "rows": []}
    worst = min(ok_rows, key=lambda r: r["sigma_min"])
    lv = levels[worst["level"]]
    q_t = np.array(lv["q"], dtype=float)
    x_d = case.fk(q_t) + amp * np.array(lv["u_min_dir"], dtype=float)
    gains = gains_of(kind, case, q_d)
    out = {"sigma_min": lv["sigma_min"], "label": "out_of_range/os_invdyn 最难的未失效 level",
           "rows": [], "_recs": {}}
    for d in (dt, dt / 2.0):
        spec = vl.make_spec(kind, case_name, gains,
                            {"type": "os_const", "x_d": [float(v) for v in x_d]}, q_t,
                            np.zeros(case.N + 1), T, d, d, label="sing-refine-dt%.0e" % d)
        res = vl.run_specs([spec], workers=1, verbose=False)[0]
        e = _err_series(res, case, kind)
        # 与主扫描同口径：**全程峰值**归一（后半窗口径会落在误差回弹段上，见 `singular_sweep`）
        st = _window_stats(res["t"], e, res["t"][0])
        ueff = np.abs(res["u_eff"][:, 1:])
        out["rows"].append({"dt_s": float(d),
                            "u_eff_inf": float(ueff.max()) if ueff.size else float("nan"),
                            "u_inf": float(np.max(np.abs(res["u"][:, 1:])))
                            if res["u"].size else float("nan"),
                            "qdot_inf": float(np.max(np.abs(res["qdot"][:, 1:]))),
                            "err_final": st["final"], "err_peak": st["peak"],
                            "n_steps": int(res["n_steps"])})
        out["_recs"][float(d)] = res
    a, b = out["rows"][0], out["rows"][1]
    out["u_eff_inf_ratio_dt_main_over_ref"] = (a["u_eff_inf"] / b["u_eff_inf"]
                                               if b["u_eff_inf"] > 0 else None)
    out["note"] = ("同一用例、同初值、同增益，只把 dt 减半。若反馈力矩的大值来自 `J_a⁻¹` 的病态放大"
                   "（控制/几何效应），比值应 ≈1（与步长无关）；若来自积分误差（离散化效应），"
                   "则应随 dt 显著变化（O(dt) 时为 2×）。⚠️ 这里看的是 **u−g**——总力矩被 `g` 主导，"
                   "用它做归因会得出「与步长无关」的错误结论。")
    if verbose:
        print("        步长归因（σ_min=%.3e，看 **u−g**）：dt=%.0e → %.3e；dt=%.0e → %.3e；比值 %.3f"
              % (out["sigma_min"], a["dt_s"], a["u_eff_inf"], b["dt_s"], b["u_eff_inf"],
                 out["u_eff_inf_ratio_dt_main_over_ref"] or float("nan")))
    return out


# ---------------------------------------------------------------------------
# 4.2a 模型失配（判据 1）
# ---------------------------------------------------------------------------
def mismatch_prediction(kind, case, q_d, gains, scale):
    """质量失配 `s` 下的**闭式稳态误差预测**（一阶，`q ≈ q_d`；见计划 §5.4.2）。

    | 控制器 | 平衡条件 | 预测 |
    |---|---|---|
    | `js_pd` | `K_Pq̃ = (s−1)g` | `q̃ = K_P⁻¹(s−1)g` |
    | `js_invdyn` | `B K_Pq̃ = (s−1)g` | `q̃ = K_P⁻¹B⁻¹(s−1)g` |
    | `os_pd` | `J_aᵀK_Px̃ = (s−1)g` | `x̃ = K_P⁻¹J_a⁻ᵀ(s−1)g` |
    | `os_invdyn` | `J_a⁻¹K_Px̃ = (s−1)B⁻¹g` | `x̃ = K_P⁻¹J_a(s−1)B⁻¹g` |

    惯量失配**不改变 `g`**，故预测稳态误差为 **0**（只有暂态/带宽变化）——这正是本实验要对照的。
    """
    s = float(scale)
    _B, _C, g = case.dyn(q_d, np.zeros(case.N + 1))
    K_P = gains["K_P"]
    B = s3.B_mat(case, q_d)
    J_a = case.jac(q_d)
    if kind == "js_pd":
        e = np.linalg.solve(K_P, (s - 1.0) * g)
        return float(np.max(np.abs(e))), "‖q̃‖∞", e
    if kind == "js_invdyn":
        e = np.linalg.solve(K_P, np.linalg.solve(B, (s - 1.0) * g))
        return float(np.max(np.abs(e))), "‖q̃‖∞", e
    if kind == "os_pd":
        e = np.linalg.solve(K_P, np.linalg.solve(J_a.T, (s - 1.0) * g))
        return float(np.linalg.norm(e * np.array([1, 1, 1, 1 / cm.ELL_DEFAULT,
                                                  1 / cm.ELL_DEFAULT, 1 / cm.ELL_DEFAULT]))), \
            "‖x̃_w‖", e
    e = np.linalg.solve(K_P, J_a @ np.linalg.solve(B, (s - 1.0) * g))
    return float(np.linalg.norm(e * np.array([1, 1, 1, 1 / cm.ELL_DEFAULT,
                                              1 / cm.ELL_DEFAULT, 1 / cm.ELL_DEFAULT]))), \
        "‖x̃_w‖", e


def _decay_rate(t, e, t_from, t_to):
    """误差包络在 `[t_from, t_to]` 上的**指数衰减率**（`−d ln e/dt`，log 最小二乘）。

    用于惯量失配的「慢模态」分析：`e(t)` 晚段由最慢模态主导，其衰减率 ∝ `ω_n·√λ_min(M)`。
    返回 `None`（数据不足）或 dict。
    """
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=float)
    m = (t >= t_from) & (t <= t_to) & (e > 1e-14)
    if m.sum() < 5:
        return None
    slope = float(np.polyfit(t[m], np.log(e[m]), 1)[0])
    return {"rate_per_s": float(-slope), "n_points": int(m.sum()),
            "e_start": float(e[m][0]), "e_end": float(e[m][-1]),
            "window_s": [float(t_from), float(t_to)]}


def inertia_mode_prediction(case, q_d, scale):
    """惯量失配下的**模态预测**：`M = B_plant⁻¹B_nom` 的特征值。

    被控对象 `B_p = B_m + ι·B_I`、控制器用 `B_nom = B_m + B_I`（`B` 对 `m` 与 `I_inn` 都是线性的）。
    精确线性化后 `q̈ = M·y`（`M = B_p⁻¹B_nom`，`y` 是控制器给出的「期望加速度」），
    故闭环带宽被 `M` 的特征值改写：最慢方向的速率比 ≈ `√λ_min(M)`。**`g` 不受影响**（平衡点仍是 `q_d`）。
    """
    B_nom = s3.B_mat(case, q_d)
    prm_p = cm.perturb_params(case.prm, {"inertia_scale": float(scale)})
    d, a, alpha, m, pc, I_inn, g0 = prm_p
    from model import robot_B                                       # noqa: PLC0415
    B_p = robot_B(np.asarray(q_d, dtype=float), d, a, alpha, m, pc, I_inn, case.N)
    M = np.linalg.solve(B_p, B_nom)
    ev = np.linalg.eigvals(M)
    return {"scale": float(scale), "lambda_min_M": float(np.min(ev.real)),
            "lambda_max_M": float(np.max(ev.real)),
            "predicted_rate_ratio_vs_nominal": float(np.sqrt(max(np.min(ev.real), 0.0))),
            "spread_lambda": float(np.max(ev.real) / max(np.min(ev.real), 1e-300)),
            "note": "M = B_plant⁻¹B_nom；惯量失配不改 g ⇒ 平衡点仍为 q_d，稳态误差理论为 0；"
                    "但最慢方向的衰减率约为 ω_n·√λ_min(M)，有限时间窗内会留下**暂态**残留"}


def model_mismatch(case_name=CASE4, levels=MISMATCH_LEVELS,
                   inertia_levels=MISMATCH_INERTIA_LEVELS,
                   T=T_MISMATCH, T_inertia=T_MISMATCH_INERTIA, T_long=T_MISMATCH_LONG,
                   dt=DT4, verbose=True, workers=None):
    """判据 1：**质量 / 惯量失配**施加于 plant，控制器仍用标称模型。

    两个族的**物理不同，必须分开处理**（这是冒烟阶段实测后修正的口径）：

    | 族 | 是否改 `g` | 平衡点 | 观测量 | 理论 |
    |---|---|---|---|---|
    | `mass`（质量 ×s） | 是（`g` ∝ `s`） | 偏移 | 稳态误差 | 闭式预测（`mismatch_prediction`），∝`|s−1|` |
    | `inertia`（`I_inn` ×ι） | **否** | **仍是 `q_d`** | 有限窗内的残留 + 衰减率 | 无稳态偏移；带宽被 `M = B_p⁻¹B_nom` 改写 |

    - `mass` 族：初值取 `q_d` **不加人工扰动**——误差完全由失配自身激发（更干净，也更快收敛）；
    - `inertia` 族：**必须**加扰动才能看见暂态，且在有限 `T` 内测到的残留**不是稳态误差**，
      而是「慢模态被拉慢」的暂态残留 ⇒ 另跑**长时段**（`T_long = %g s`）证明它继续衰减，
      并把实测衰减率比与 `√λ_min(M)` 对拍（`inertia_mode_prediction`）。
    """ % T_long
    case = cm.make_case(case_name)
    q_d = q_reg()
    q0_mass = q_d.copy()                                   # 质量族：无扰动
    q0_iner = q_d.copy()
    q0_iner[1:] += DQ4 * _pert(case.N)                     # 惯量族：需要扰动激发
    specs, meta = [], []
    for s in levels:
        for kind in KIND_ORDER:
            gains = gains_of(kind, case, q_d)
            specs.append(vl.make_spec(kind, case_name, gains,
                                      vl.make_ref_spec(kind, case, q_d), q0_mass,
                                      np.zeros(case.N + 1), T, dt, dt,
                                      plant={"mass_scale": float(s)},
                                      label="mm-mass-%.2f-%s" % (s, kind)))
            meta.append(("mass", float(s), kind, gains, T))
    for s in inertia_levels:
        for kind in KIND_ORDER:
            gains = gains_of(kind, case, q_d)
            specs.append(vl.make_spec(kind, case_name, gains,
                                      vl.make_ref_spec(kind, case, q_d), q0_iner,
                                      np.zeros(case.N + 1), T_inertia, dt, dt,
                                      plant={"inertia_scale": float(s)},
                                      label="mm-inertia-%.2f-%s" % (s, kind)))
            meta.append(("inertia", float(s), kind, gains, T_inertia))
    for s in inertia_levels:
        if abs(s - 1.0) < 1e-12:
            continue
        for kind in MISMATCH_LONG_KINDS:
            gains = gains_of(kind, case, q_d)
            specs.append(vl.make_spec(kind, case_name, gains,
                                      vl.make_ref_spec(kind, case, q_d), q0_iner,
                                      np.zeros(case.N + 1), T_long, dt, dt,
                                      plant={"inertia_scale": float(s)},
                                      label="mm-long-%.2f-%s" % (s, kind)))
            meta.append(("inertia_long", float(s), kind, gains, T_long))
    recs = vl.run_specs(specs, workers=workers, tag="s4-mismatch", verbose=verbose)

    rows = []
    for (fam, s, kind, gains, Tv), res in zip(meta, recs):
        _os_fix_qtilde(res, q_d)
        e = _err_series(res, case, kind)
        st = _window_stats(res["t"], e, (0.5 * Tv if fam == "mass" else 0.25 * Tv))
        u_st = _u_window_stats(res["t"], res["u"], 0.5 * Tv)
        pred, pnorm, _vec = ((mismatch_prediction(kind, case, q_d, gains, s)
                              if fam == "mass" else (None, "-", None)))
        rate = _decay_rate(res["t"], e, 0.4 * Tv, Tv)
        rows.append({
            "family": fam, "scale": s, "delta": float(s - 1.0), "kind": kind,
            "T": float(Tv),
            "err_steady": st["mean"], "err_final": st["final"], "err_peak": st["peak"],
            "err_rms": st["rms"], "u_inf": u_st["u_inf"],
            "predicted_steady_mass": None if pred is None else float(pred),
            "predicted_norm": pnorm,
            "ratio_measured_over_predicted": (float(st["final"] / pred)
                                              if (pred is not None and abs(pred) > 1e-300)
                                              else None),
            "decay": rate,
            "settle_time_2pct_s": met.settle_time(res["t"], e),
            "nan": bool(res["nan"]), "error": res["error"],
            "wall_time_s": float(res["wall_time_s"]),
        })

    # ---- 质量族：斜率 + 与闭式预测的比对 -----------------------------------
    mass_sum = {}
    for kind in KIND_ORDER:
        rs = sorted([r for r in rows if r["family"] == "mass" and r["kind"] == kind],
                    key=lambda r: r["scale"])
        nz = [r for r in rs if r["delta"] != 0.0]
        mass_sum[kind] = {
            "rows": rs,
            "loglog_slope_err_vs_abs_delta": float(_fit_slope_safe(
                [abs(r["delta"]) for r in nz], [r["err_final"] for r in nz])),
            "err_final_at_levels": {"%.1f" % r["scale"]: r["err_final"] for r in rs},
            "ratio_measured_over_predicted_at_levels": {
                "%.1f" % r["scale"]: r["ratio_measured_over_predicted"] for r in nz},
            "worst_pred_ratio_dev": (max(abs(np.log(max(r["ratio_measured_over_predicted"], 1e-30)))
                                         for r in nz
                                         if r["ratio_measured_over_predicted"] is not None)
                                     if any(r["ratio_measured_over_predicted"] is not None
                                            for r in nz) else None),
        }
    mass_slope_ok = all(MISMATCH_MASS_SLOPE_TOL[0] <= mass_sum[k]["loglog_slope_err_vs_abs_delta"]
                        <= MISMATCH_MASS_SLOPE_TOL[1] for k in KIND_ORDER)
    mass_pred_ok = all(
        all(MISMATCH_PRED_RATIO_TOL[0] <= (r["ratio_measured_over_predicted"] or 0.0)
            <= MISMATCH_PRED_RATIO_TOL[1] for r in mass_sum[k]["rows"]
            if r["ratio_measured_over_predicted"] is not None)
        for k in KIND_ORDER)

    # ---- 惯量族：模态预测 + 衰减率比 + 长时段对照 ---------------------------
    iner_pred = {("%.2f" % s): inertia_mode_prediction(case, q_d, s) for s in inertia_levels}
    iner_sum = {}
    for kind in KIND_ORDER:
        rs = sorted([r for r in rows if r["family"] == "inertia" and r["kind"] == kind],
                    key=lambda r: r["scale"])
        ref = next((r for r in rs if abs(r["scale"] - 1.0) < 1e-12), None)
        rate_ref = (ref["decay"] or {}).get("rate_per_s") if ref else None
        per = {}
        for r in rs:
            rate = (r["decay"] or {}).get("rate_per_s")
            pred_ratio = iner_pred["%.2f" % r["scale"]]["predicted_rate_ratio_vs_nominal"]
            meas_ratio = (rate / rate_ref) if (rate and rate_ref) else None
            per["%.1f" % r["scale"]] = {
                "err_final": r["err_final"], "err_peak": r["err_peak"],
                "decay_rate_per_s": rate,
                "measured_rate_ratio_vs_nominal": meas_ratio,
                "predicted_rate_ratio_vs_nominal": pred_ratio,
                "ratio_of_ratios": (meas_ratio / pred_ratio
                                    if (meas_ratio and pred_ratio > 1e-12) else None),
            }
        iner_sum[kind] = {"per_level": per,
                          "lambda_min_M": {k: v["lambda_min_M"] for k, v in iner_pred.items()},
                          "rows": rs}
    long_rows = [r for r in rows if r["family"] == "inertia_long"]
    long_sum = {}
    for r in long_rows:
        key = "%s/%.1f" % (r["kind"], r["scale"])
        long_sum[key] = {"T_s": r["T"], "err_final": r["err_final"], "err_peak": r["err_peak"],
                         "decay": r["decay"], "nan": r["nan"], "error": r["error"]}
    # 「无稳态偏移」的证据：长时段末值应远小于同族有限窗末值，且衰减率>0
    long_ok = all(v["decay"] is not None and v["decay"]["rate_per_s"] > 0
                  and v["err_final"] < 1e-2 for v in long_sum.values()) if long_sum else False
    rate_ok = all(
        (v["ratio_of_ratios"] is not None
         and INERTIA_RATE_RATIO_TOL[0] <= v["ratio_of_ratios"] <= INERTIA_RATE_RATIO_TOL[1])
        for k in iner_sum for kk, v in iner_sum[k]["per_level"].items()
        if abs(float(kk) - 1.0) > 1e-9)
    if verbose:
        print("      判据1 [模型失配/mass] 稳态误差（理论 ∝|s−1|，闭式预测见 mismatch_prediction）：")
        for kind in KIND_ORDER:
            d = mass_sum[kind]
            print("        %-10s 斜率=%+.3f（窗口 %s）  各档 err=%s" %
                  (kind, d["loglog_slope_err_vs_abs_delta"], MISMATCH_MASS_SLOPE_TOL,
                   {k: "%.2e" % v for k, v in d["err_final_at_levels"].items()}))
            print("                   实测/预测=%s" % {
                k: ("n/a" if v is None else "%.2f" % v)
                for k, v in d["ratio_measured_over_predicted_at_levels"].items()})
        print("      判据1 [模型失配/inertia] 惯量失配**不改 g** ⇒ 无稳态偏移，只改暂态：")
        for k, v in iner_pred.items():
            print("        ι=%s：λ_min(M)=%.4f ⇒ 最慢模态速率比预测 %.3f（λ_max=%.3f）"
                  % (k, v["lambda_min_M"], v["predicted_rate_ratio_vs_nominal"],
                     v["lambda_max_M"]))
        for kind in KIND_ORDER:
            print("        %-10s 实测/预测的衰减率比：%s" %
                  (kind, {kk: ("n/a" if v["ratio_of_ratios"] is None
                               else "%.2f" % v["ratio_of_ratios"])
                          for kk, v in iner_sum[kind]["per_level"].items()}))
        for k, v in long_sum.items():
            print("        长时段对照 [%s] T=%.1f s：末值=%.3e rad，衰减率=%s /s（>0 ⇒ 继续收敛）"
                  % (k, v["T_s"], v["err_final"],
                     "n/a" if v["decay"] is None else "%.2f" % v["decay"]["rate_per_s"]))
    return {
        "case": case_name, "dt": dt,
        "T_mass_s": T, "T_inertia_s": T_inertia, "T_long_s": T_long,
        "mass_levels": [float(v) for v in levels],
        "inertia_levels": [float(v) for v in inertia_levels],
        "q_d": [float(v) for v in q_d],
        "ic": {"mass": "q0 = q_d（不加扰动，误差全由失配激发）",
               "inertia": "q0 = q_d + %.3f·v（固定种子方向，激发暂态）" % DQ4,
               "dir": [float(v) for v in _pert(case.N)]},
        "rows": rows,
        "mass": mass_sum, "inertia": iner_sum, "inertia_mode_prediction": iner_pred,
        "inertia_long": long_sum,
        "check": {
            "desc": "模型失配（计划 §5.4.2）：质量/惯量失配施加于被控对象、控制器用标称模型；"
                    "质量族给出「误差 ∝|s−1|」的量级关系并与闭式预测吻合，"
                    "惯量族给出「无稳态偏移、只改暂态」的证据",
            "threshold": "① 质量族四个算法的 log-log 斜率 ∈ %s（理论 1）且实测/预测 ∈ %s；"
                         "② 惯量族：实测衰减率比 / √λ_min(M) ∈ %s，且长时段末值 < 1e-2 rad 并仍在衰减"
                         % (MISMATCH_MASS_SLOPE_TOL, MISMATCH_PRED_RATIO_TOL,
                            INERTIA_RATE_RATIO_TOL),
            "measured": {
                "mass": {"%s" % k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                         for k, v in mass_sum.items()},
                "inertia": {"%s" % k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                            for k, v in iner_sum.items()},
                "inertia_mode_prediction": iner_pred,
                "inertia_long": long_sum,
                "mass_slope_ok": bool(mass_slope_ok), "mass_pred_ok": bool(mass_pred_ok),
                "inertia_rate_ok": bool(rate_ok), "inertia_long_ok": bool(long_ok),
            },
            "passed": bool(mass_slope_ok and mass_pred_ok and rate_ok and long_ok),
        },
        "_raw": {"rows": rows},
    }


# ---------------------------------------------------------------------------
# 4.2b 摩擦未补偿（判据 1）
# ---------------------------------------------------------------------------
def friction_resolution(case, coulomb_max, eps=FRIC_EPS, dt=FRIC_DT):
    """摩擦 `tanh` 过渡层是否被积分步长**分辨**（判据见常量 `FRIC_EPS` 处的说明）。

    `τ_f = −F_c·tanh(q̇/ε)` 的线性区刚度是 `F_c/ε`；经 `B⁻¹` 后最硬的模态速率是
    `(F_c/ε)/λ_min(B)`，RK4 稳定要求 `dt·(F_c/ε)/λ_min(B) ≤ 2.785`（`rk4_stability_limit`）。
    不满足时积分器无法分辨过渡层，会给出**数值伪爬行**（误差随时间单调增长）。
    """
    B = case.dyn(q_d := q_reg(), np.zeros(case.N + 1))[0]
    lam_min = float(np.linalg.eigvalsh(B).min())
    stiff = float(dt * (coulomb_max / eps) / lam_min)
    limit = float(rk4_stability_limit(1.0))
    return {"lambda_min_B": lam_min, "stiffness_number_h_over_tau": stiff,
            "rk4_limit": limit, "resolved": bool(stiff <= limit),
            "eps_rad_s": float(eps), "dt_s": float(dt), "coulomb_max_Nm": float(coulomb_max),
            "note": "硬性数 = dt·(F_c,max/ε)/λ_min(B)；≤ 2.785 才算被 RK4 分辨。"
                    "`ε = 1e-3` + `dt = 2e-3` 的旧口径给出 1732（超限 620 倍），"
                    "其「库仑死区律斜率 ≈1」全是数值伪爬行的产物（日志尝试 12）。"}


def friction_deadzone_prediction(kind, case, q_d, gains, fc):
    """该算法在**库仑摩擦** `F_c` 下的**静平衡预测**（日志尝试 11 §11.3 要求的修法）。

    静平衡（`q̇ = 0`、调节用例 `q̇_d = q̈_d = 0`、plant 侧 `Cq̇ = F_fq̇ = 0`）：
    `u(q̃) = g(q) + τ_f`，其中 `τ_f = −τ_e` 是摩擦力矩（`|τ_f,j| ≤ F_c`）。逐算法：
    `js_pd : u = K_P q̃ + g`；`js_invdyn : u = B_nom K_P q̃ + g`；
    `os_pd : u = J_aᵀ K_P x̃ + g`；`os_invdyn : u = B_nom J_a⁻¹ K_P x̃ + g`。
    于是 `q̃ = K_P_eff⁻¹ τ_f`（OS 用 `x̃`），`K_P_eff` 逐算法不同——**旧公式对四者统一用
    `max_j F_c/|K_P,jj|` 是错的**（`js_pd` 的 `K_P = B(q_d)ω_n²` 并非对角、`js_invdyn` 还要过一层 `B`）。

    ⚠️ **两个预测都是「上界」性质，实测通常远小于它们**：
    - `upper_bound`：死区盒 `|τ_f,j| ≤ F_c` 的**最坏情形**（诱导范数）。它极保守——
      最坏方向对准最小惯量的腕部滚转，实测/上界只有 `1e-3 ~ 1.6e-2`。
    - `same_sign`：`τ_f = +F_c·1`（各关节同号饱和）的代表值，同样偏保守。
    报告须如实说明：**本用例的残留并非「停到死区边界」，而是各关节「能否挣开静摩擦」的
    结果**——控制权限小于 `F_c` 的腕部关节根本不动（残留≈初始扰动），故残留被初始误差
    「垫底」，log-log 斜率远小于 1（详见 `check` 的说明）。
    """
    B = case.dyn(np.asarray(q_d, dtype=float), np.zeros(case.N + 1))[0]
    J = case.jac(np.asarray(q_d, dtype=float))
    K_P = gains["K_P"]
    ell = cm.ELL_DEFAULT
    Dinv = np.diag([1.0, 1.0, 1.0, 1.0 / ell, 1.0 / ell, 1.0 / ell])
    if kind == "js_pd":
        M = K_P.copy()
    elif kind == "js_invdyn":
        M = B @ K_P
    elif kind == "os_pd":
        M = J.T @ K_P
    else:
        M = B @ np.linalg.solve(J, K_P)
    Minv = np.linalg.inv(M)
    N = case.N
    ones = np.ones(N)
    if kind.startswith("js"):
        # 观测量 max_j|q̃_j| ⇒ 诱导 ∞→∞ 范数（行和最大）
        return {"observable": "‖q̃‖∞", "K_P_eff": M,
                "upper_bound": float(fc * np.linalg.norm(Minv, np.inf)),
                "same_sign": float(fc * np.abs(Minv @ ones).max()),
                "bound_kind": "induced_inf_norm"}
    # 观测量 ‖D⁻¹x̃‖₂ ⇒ ∞→2 算子范数（各列 2 范数的最大者）
    W = Dinv @ Minv
    return {"observable": "‖D⁻¹x̃‖₂（ℓ=%.1f m）" % ell, "K_P_eff": M,
            "upper_bound": float(fc * np.max(np.linalg.norm(W, axis=0))),
            "same_sign": float(fc * np.linalg.norm(W @ ones)),
            "bound_kind": "inf_to_2_operator_norm"}


def friction_uncompensated(case_name=CASE4, coulomb_levels=FRIC_COULOMB_LEVELS,
                           fric_cases=FRIC_CASES, T=T_FRICTION, T_shape=T_FRICTION_SHAPE,
                           dt=FRIC_DT, verbose=True, workers=None):
    """判据 1：plant 启用**粘性 + 库仑**摩擦，控制器一律 `F_f = 0`（即摩擦完全未建模）。

    ⚠️ **口径已于 2026-09-20 重做**（日志尝试 11 §11.3 与尝试 12），三处：
    ① **步长与光滑宽度必须配对**（`ε = %g rad/s`、`dt = %.0e s`，硬性数 ≤ `%.3f`）：
       旧口径 `ε = 1e-3` / `dt = 2e-3` 的硬性数是 **1732**，积分器根本分辨不了过渡层，
       测到的是**数值伪爬行**而非摩擦物理——「斜率 ≈1」的表观定律是假的。
    ② **守卫换成「真零摩擦基线」**（`friction=None`，单独一组）：旧守卫用 `F_c = 0`
       那一档，但它仍带粘性底噪 `%g`，本身就有 `1e-2` 量级的残留（粘性未补偿 ⇒ 只爬行、
       不存在静态平衡），所以「残留 < 1e-6」恒为假，作为守卫**不成立**。
       新守卫判两件事：零摩擦基线的误差必须**显著衰减**（末值 ≤ %.0f%% **全程**峰值——这是抓
       「控制器读到冻结状态」那类 bug 的哨兵），且每个 `F_c > 0` 档的残留都 ≥ 2× 它。
    ③ **库仑预测按各算法自己的静平衡式写**（见 `friction_deadzone_prediction`），
       并明确标注为**上界**（旧的 `max_j F_c/|K_P,jj|` 对 `js_pd`/`js_invdyn`/`os_*` 全错）。

    **判据**（趋势型，与书中「`F_f` 半正定、需补偿」一致）：
    ① 四个算法的残留随 `F_c` **单调不减**；② 每个 `F_c > 0` 档 ≥ 2× 零摩擦基线；
    ③ 零摩擦基线的误差显著衰减（哨兵）。log-log 斜率**如实报告但不作判据**——本用例下
    它远小于 1，原因是：控制权限小于 `F_c` 的腕部关节**根本挣不开静摩擦**（残留≈初始
    扰动 `%.2f rad` 量级），故残留被初始误差垫底；`tanh` 光滑化的库仑模型又**没有真正的
    集值静摩擦**，不存在教科书意义上的「静态死区」。
    """ % (FRIC_EPS, dt, rk4_stability_limit(1.0), FRIC_VISCOUS_SWEEP,
           int(GUARD_DECAY_RATIO * 100), DQ4)
    case = cm.make_case(case_name)
    q_d = q_reg()
    q0 = q_d.copy()
    q0[1:] += DQ4 * _pert(case.N)
    N = case.N
    res_info = friction_resolution(case, max(coulomb_levels), eps=FRIC_EPS, dt=dt)
    specs, meta = [], []
    for fc in coulomb_levels:
        for kind in KIND_ORDER:
            gains = gains_of(kind, case, q_d)
            specs.append(vl.make_spec(kind, case_name, gains,
                                      vl.make_ref_spec(kind, case, q_d), q0,
                                      np.zeros(case.N + 1), T, dt, dt,
                                      friction={"viscous": [FRIC_VISCOUS_SWEEP] * N,
                                                "coulomb": [float(fc)] * N, "eps": FRIC_EPS},
                                      label="fr-c%.1f-%s" % (fc, kind)))
            meta.append(("coulomb_sweep", float(fc), kind, gains, T))
    # ★ 真零摩擦基线（守卫）：`friction=None` ⇒ 完全不注入摩擦
    for kind in KIND_ORDER:
        gains = gains_of(kind, case, q_d)
        specs.append(vl.make_spec(kind, case_name, gains,
                                  vl.make_ref_spec(kind, case, q_d), q0,
                                  np.zeros(case.N + 1), T, dt, dt, friction=None,
                                  label="fr-zero-%s" % kind))
        meta.append(("zero_friction", None, kind, gains, T))
    for cname, spec_f in fric_cases.items():
        for kind in KIND_ORDER:
            gains = gains_of(kind, case, q_d)
            specs.append(vl.make_spec(kind, case_name, gains,
                                      vl.make_ref_spec(kind, case, q_d), q0,
                                      np.zeros(N + 1), T_shape, dt, dt,
                                      friction={"viscous": spec_f["viscous"] * N,
                                                "coulomb": spec_f["coulomb"] * N,
                                                "eps": FRIC_EPS},
                                      label="fr-%s-%s" % (cname, kind)))
            meta.append((cname, None, kind, gains, T_shape))
    recs = vl.run_specs(specs, workers=workers, tag="s4-friction", verbose=verbose)

    rows = []
    for (grp, fc, kind, gains, Tv), res in zip(meta, recs):
        _os_fix_qtilde(res, q_d)
        e = _err_series(res, case, kind)
        st = _window_stats(res["t"], e, 0.6 * Tv)
        # ⚠️ 守卫要的是「误差到底有没有衰减」，故用**全程峰值**归一（2026-09-20 实测后定，见日志尝试 13）：
        # 末段窗口口径的峰值落在已经衰减过的后半段上，比值被系统性放大（`os_invdyn` 实测 0.3452 vs
        # 全程 0.1257），会在完整跑里把判据 1 误判 FAIL。阈值 `GUARD_DECAY_RATIO` 一字未动。
        st_all = _window_stats(res["t"], e, res["t"][0])
        u_st = _u_window_stats(res["t"], res["u"], 0.6 * Tv)
        tail = res["t"] >= 0.9 * Tv - 1e-12
        qdot_tail = np.abs(res["qdot"][tail][:, 1:])
        e_tail = np.abs(e[tail])
        pred = friction_deadzone_prediction(kind, case, q_d, gains, float(fc)) if fc else None
        rows.append({
            "group": grp, "coulomb_Nm": fc, "kind": kind,
            "err_steady": st["mean"], "err_final": st["final"], "err_peak": st["peak"],
            "decay_ratio_final_over_peak": float(st["final"] / max(st["peak"], 1e-300)),
            # 全程峰值口径（守卫用；与 `singular_sweep` 的衰减比同口径）
            "err_peak_whole_run": st_all["peak"],
            "decay_ratio_whole_run": float(st_all["final"] / max(st_all["peak"], 1e-300)),
            "u_inf": u_st["u_inf"], "u_ac_rms_max": u_st["u_ac_rms_max"],
            "qdot_tail_max_rad_s": float(qdot_tail.max()) if qdot_tail.size else float("nan"),
            "qdot_tail_mean_rad_s": float(qdot_tail.mean()) if qdot_tail.size else float("nan"),
            "err_tail_ptp_rad": float(e_tail.max() - e_tail.min()) if e_tail.size else float("nan"),
            "deadzone_upper_bound": None if pred is None else pred["upper_bound"],
            "deadzone_same_sign": None if pred is None else pred["same_sign"],
            "deadzone_observable": None if pred is None else pred["observable"],
            "ratio_measured_over_upper_bound": (
                None if pred is None else float(st["mean"] / max(pred["upper_bound"], 1e-300))),
            "ratio_measured_over_same_sign": (
                None if pred is None else float(st["mean"] / max(pred["same_sign"], 1e-300))),
            "nan": bool(res["nan"]), "error": res["error"],
            "wall_time_s": float(res["wall_time_s"]),
        })

    zero = {k: next(r for r in rows if r["group"] == "zero_friction" and r["kind"] == k)
            for k in KIND_ORDER}
    coulomb = {}
    for kind in KIND_ORDER:
        rs = sorted([r for r in rows if r["group"] == "coulomb_sweep" and r["kind"] == kind],
                    key=lambda r: r["coulomb_Nm"])
        nz = [r for r in rs if r["coulomb_Nm"] > 0]
        ratios = [r["ratio_measured_over_upper_bound"] for r in nz
                  if r["ratio_measured_over_upper_bound"] is not None]
        coulomb[kind] = {
            "rows": rs,
            "loglog_slope_err_vs_coulomb": float(_fit_slope_safe(
                [r["coulomb_Nm"] for r in nz], [r["err_steady"] for r in nz])),
            "err_at_levels": {("%.1f" % r["coulomb_Nm"]): r["err_steady"] for r in rs},
            "monotone_in_coulomb": bool(all(rs[i]["err_steady"] <= rs[i + 1]["err_steady"] * 1.02
                                            for i in range(len(rs) - 1))),
            "min_ratio_measured_over_zero_friction": float(
                min(r["err_steady"] / max(zero[kind]["err_steady"], 1e-300) for r in nz)),
            "upper_bound_at_levels": {("%.1f" % r["coulomb_Nm"]): r["deadzone_upper_bound"]
                                      for r in rs},
            "ratio_measured_over_upper_bound_range": (
                None if not ratios else [float(min(ratios)), float(max(ratios))]),
        }
    shapes = {}
    for cname in fric_cases:
        shapes[cname] = {k: next(r for r in rows if r["group"] == cname and r["kind"] == k)
                         for k in KIND_ORDER}
    monotone_ok = all(coulomb[k]["monotone_in_coulomb"] for k in KIND_ORDER)
    sep_ok = all(coulomb[k]["min_ratio_measured_over_zero_friction"] >= 2.0 for k in KIND_ORDER)
    guard = {k: {"err_peak": zero[k]["err_peak_whole_run"], "err_final": zero[k]["err_final"],
                 "decay_ratio": zero[k]["decay_ratio_whole_run"],
                 "decay_ratio_tail_window": zero[k]["decay_ratio_final_over_peak"]}
             for k in KIND_ORDER}
    guard_ok = bool(all(v["decay_ratio"] <= GUARD_DECAY_RATIO for v in guard.values()))
    passed = bool(monotone_ok and sep_ok and guard_ok and res_info["resolved"])
    if verbose:
        print("      摩擦数值分辨率：硬性数 dt·(F_c,max/ε)/λ_min(B) = %.2f（须 ≤ %.3f）→ %s"
              % (res_info["stiffness_number_h_over_tau"], res_info["rk4_limit"],
                 "分辨" if res_info["resolved"] else "**未分辨！结论无效**"))
        for kind in KIND_ORDER:
            c = coulomb[kind]
            print("      判据1 [摩擦 %s] 残留=%s（log-log 斜率 %.3f ← **不作判据**；"
                  "单调=%s；F_c>0 档 / 零摩擦基线 ≥ %.1f×）"
                  % (kind, {k: "%.2e" % v for k, v in c["err_at_levels"].items()},
                     c["loglog_slope_err_vs_coulomb"], c["monotone_in_coulomb"],
                     c["min_ratio_measured_over_zero_friction"]))
            print("            静平衡上界（K_P_eff⁻¹ 最坏盒）：%s ⇒ 实测/上界 %s"
                  % ({k: ("n/a" if v is None else "%.3e" % v)
                      for k, v in c["upper_bound_at_levels"].items()},
                     "n/a" if c["ratio_measured_over_upper_bound_range"] is None
                     else "%.4f~%.4f" % tuple(c["ratio_measured_over_upper_bound_range"])))
        print("        守卫（真零摩擦基线的误差必须衰减到 ≤ %.0f%% **全程**峰值）：%s → %s"
              % (100 * GUARD_DECAY_RATIO,
                 {k: "%.4f" % v["decay_ratio"] for k, v in guard.items()},
                 "ok" if guard_ok else "异常！控制器可能没读当前状态"
                 "（⚠️ 该守卫按完整 T = %.2f s 标定；`--quick` 把 T 压短后必然报异常，判据无意义）"
                 % T))
        print("            （对照：末段窗口口径会给出 %s —— 峰值落在已衰减段上，比值被放大）"
              % {k: "%.4f" % v["decay_ratio_tail_window"] for k, v in guard.items()})
        for cname in fric_cases:
            r = shapes[cname]["js_pd"]
            print("        形态[%s] js_pd：残留误差=%.3e  末段 |q̇|max=%.3e rad/s  "
                  "末段误差峰峰值=%.3e" % (cname, r["err_steady"], r["qdot_tail_max_rad_s"],
                                          r["err_tail_ptp_rad"]))
    return {
        "case": case_name, "dt": dt, "T_sweep_s": T, "T_shape_s": T_shape,
        "eps_rad_s": FRIC_EPS, "viscous_sweep_floor": FRIC_VISCOUS_SWEEP,
        "coulomb_levels": [float(v) for v in coulomb_levels],
        "friction_cases": fric_cases, "resolution": res_info,
        "rows": rows, "coulomb": coulomb, "shapes": {
            c: {k: {kk: vv for kk, vv in r.items()} for k, r in d.items()}
            for c, d in shapes.items()},
        "zero_friction_baseline": guard,
        "nominal_guard": {
            "err_at_zero_coulomb": {k: coulomb[k]["err_at_levels"]["0.0"] for k in KIND_ORDER},
            "zero_friction_baseline": guard, "passed": guard_ok, "decay_limit": GUARD_DECAY_RATIO,
            "note": "⚠️ 守卫**不再**用 `F_c = 0` 那一档（它仍带粘性底噪 %.3g，本身就有 1e-2 "
                    "量级的爬行残留，「≈0」恒为假）。改用**真零摩擦基线**（friction=None）："
                    "它必须衰减到 ≤ %.0f%% **全程**峰值——这才抓得住「控制器读到冻结状态」那类 bug"
                    "（2026-09-19 冒烟正是这样抓到过一处）。"
                    "⚠️ 归一口径 2026-09-20 由「末段窗口」改为「全程」（日志尝试 13）："
                    "末段窗口的峰值落在已衰减段上，`os_invdyn` 实测 0.3452（会误判 FAIL）"
                    "vs 全程 0.1257（通过）；阈值 %.2f 一字未动。"
                    % (FRIC_VISCOUS_SWEEP, 100 * GUARD_DECAY_RATIO, GUARD_DECAY_RATIO)},
        "check": {
            "desc": "摩擦未补偿（计划 §5.4.2）：plant 启用粘性+库仑摩擦、控制器设 F_f = 0。"
                    "判据为**趋势型**：残留随 F_c 单调不减、且显著高于真零摩擦基线",
            "threshold": "① 四算法的残留随 F_c ∈ %s 单调不减；② 每个 F_c>0 档 ≥ 2× 零摩擦基线；"
                         "③ 真零摩擦基线的误差衰减到 ≤ %.0f%% **全程**峰值（哨兵）；"
                         "④ 摩擦模型被步长分辨（硬性数 ≤ %.3f）。"
                         "log-log 斜率如实报告但**不作判据**"
                         % (list(coulomb_levels), 100 * GUARD_DECAY_RATIO,
                            rk4_stability_limit(1.0)),
            "measured": {"per_kind": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                                      for k, v in coulomb.items()},
                         "zero_friction_baseline": guard,
                         "resolution": res_info,
                         "monotone_ok": bool(monotone_ok),
                         "separation_ok": bool(sep_ok),
                         "guard_ok": bool(guard_ok),
                         "slope_note": "跨 (0.5, 1.0) N·m 的 log-log 斜率只有 0.05~0.2 量级——"
                                       "**不是**「库仑死区律被证伪」，而是本用例的残留被"
                                       "「挣不开静摩擦的腕部关节」垫底（残留≈初始扰动 %.2f rad）"
                                       "所致；`tanh` 光滑化模型没有集值静摩擦，"
                                       "不存在教科书意义的静态死区。详见报告「负结果与局限」。" % DQ4},
            "passed": passed,
        },
        "_raw": {"rows": rows},
    }


# ---------------------------------------------------------------------------
# 4.2c 测量噪声（判据 1）
# ---------------------------------------------------------------------------
def ctrl_noise_sensitivity(kind, case, q_d, gains, h=1e-6):
    """控制律对 `(q, q̇)` 的**线性敏感度矩阵**（数值雅可比），用于预测噪声的传递。

    返回 `(S_q, S_qd, u0)`，`S_q[i, j] = ∂u_i/∂q_j`（关节 `j`，长度 `N`）。
    这是「噪声经控制律放大多少」的**一阶解析预测**：
    `Var(u) ≈ S_q S_qᵀ σ_q² + S_qd S_qdᵀ σ_qd²`（两路噪声独立）。
    """
    fn = CTRL_FN[kind]
    ref = vl.reg_ref(kind, case, q_d)
    N = case.N
    q = np.array(q_d, dtype=float)
    qd = np.zeros(N + 1)
    u0 = fn((q, qd), ref, gains, case)
    S_q = np.zeros((N, N))
    S_qd = np.zeros((N, N))
    for j in range(1, N + 1):
        qp = q.copy()
        qp[j] += h
        S_q[:, j - 1] = (fn((qp, qd), ref, gains, case)[1:] - u0[1:]) / h
        qdp = qd.copy()
        qdp[j] += h
        S_qd[:, j - 1] = (fn((q, qdp), ref, gains, case)[1:] - u0[1:]) / h
    return S_q, S_qd, u0[1:]


def noise_prediction(kind, case, q_d, gains, noise):
    """噪声规格 → 控制力矩 AC 分量 RMS 的一阶预测（逐关节取最大值）。"""
    S_q, S_qd, _u0 = ctrl_noise_sensitivity(kind, case, q_d, gains)
    sig_q = float(noise.get("q_sigma", 0.0) or 0.0)
    sig_qd = float(noise.get("qd_sigma", 0.0) or 0.0)
    if noise.get("vel_from_pos"):
        sig_qd = float(np.sqrt(2.0) * sig_q) / float(noise.get("h_c", 1e-3))
    var = (S_q ** 2).sum(axis=1) * sig_q ** 2 + (S_qd ** 2).sum(axis=1) * sig_qd ** 2
    return {"pred_u_ac_rms_max": float(np.sqrt(var.max())),
            "pred_u_ac_rms_per_joint": [float(v) for v in np.sqrt(var)],
            "sigma_q_used": sig_q, "sigma_qd_effective": float(sig_qd),
            "contrib_q": float(np.sqrt(((S_q ** 2).sum(axis=1) * sig_q ** 2).max())),
            "contrib_qd": float(np.sqrt(((S_qd ** 2).sum(axis=1) * sig_qd ** 2).max())),
            "note": "一阶预测：Var(u) ≈ S_qS_qᵀσ_q² + S_qdS_qdᵀσ_qd²；"
                    "`vel_from_pos` 时 σ_qd_eff = √2·σ_q/h_c（位置噪声经后向差分的放大）"}


def measurement_noise(case_name=CASE4, noise_cases=NOISE_CASES, T=T_NOISE, dt=DT4_REF,
                      verbose=True, workers=None):
    """判据 1：`q`/`q̇` 加高斯噪声（+ 可选一阶低通），观察对闭环的影响。

    六条路径（`NOISE_CASES`）：干净、只测位置、只测速度、两者、**速度由位置后向差分**（无速度
    传感器）、以及差分+低通。每条同时给**实测的力矩 AC-RMS**与**一阶预测**，从而回答
    「哪一路噪声最致命」——`q̇` 直接进 `K_D`，而位置差分还要再乘 `√2/h_c`。
    """
    case = cm.make_case(case_name)
    q_d = q_reg()
    q0 = q_d.copy()                    # ⚠️ **不加人工扰动**：初值取平衡点，
    #                                    这样误差与力矩抖动**全部由噪声产生**（若带初值扰动，
    #                                    衰减中的暂态会完全掩盖噪声效应——冒烟实测过）
    specs, meta = [], []
    for cname, nz in noise_cases.items():
        for kind in KIND_ORDER:
            gains = gains_of(kind, case, q_d)
            spec_nz = None
            if nz is not None:
                spec_nz = dict(nz)
                spec_nz.setdefault("seed", SEED4)
                spec_nz["h_c"] = dt
            specs.append(vl.make_spec(kind, case_name, gains,
                                      vl.make_ref_spec(kind, case, q_d), q0,
                                      np.zeros(case.N + 1), T, dt, dt, noise=spec_nz,
                                      label="nz-%s-%s" % (cname, kind)))
            meta.append((cname, spec_nz, kind, gains))
    recs = vl.run_specs(specs, workers=workers, tag="s4-noise", verbose=verbose)

    rows = []
    for (cname, nz, kind, gains), res in zip(meta, recs):
        _os_fix_qtilde(res, q_d)
        e = _err_series(res, case, kind)
        st = _window_stats(res["t"], e, 0.4 * T)
        u_st = _u_window_stats(res["t"], res["u"], 0.4 * T)
        pred = noise_prediction(kind, case, q_d, gains, nz) if nz else None
        rows.append({
            "noise_case": cname, "kind": kind, "noise_spec": nz,
            "u_ac_rms_max": u_st["u_ac_rms_max"], "u_ac_rms": u_st["u_ac_rms"],
            "u_inf": u_st["u_inf"],
            "err_rms": st["rms"], "err_steady": st["mean"], "err_peak": st["peak"],
            "pred_u_ac_rms_max": None if pred is None else pred["pred_u_ac_rms_max"],
            "ratio_measured_over_predicted": (
                None if (pred is None or pred["pred_u_ac_rms_max"] <= 1e-300)
                else float(u_st["u_ac_rms_max"] / pred["pred_u_ac_rms_max"])),
            "contrib_q": None if pred is None else pred["contrib_q"],
            "contrib_qd": None if pred is None else pred["contrib_qd"],
            "nan": bool(res["nan"]), "error": res["error"],
            "wall_time_s": float(res["wall_time_s"]),
        })
    # 基线（clean）作对照：`clean` 档的抖动是数值地板，故比值为**下界**（只作参考，不入判据）
    base = {k: next(r for r in rows if r["noise_case"] == "clean" and r["kind"] == k)
            for k in KIND_ORDER if any(r["noise_case"] == "clean" and r["kind"] == k
                                       for r in rows)}
    for r in rows:
        b = base.get(r["kind"])
        r["u_ac_amplification_vs_clean"] = (
            float(r["u_ac_rms_max"] / b["u_ac_rms_max"])
            if (b and b["u_ac_rms_max"] > 1e-12) else None)
        r["err_rms_ratio_vs_clean"] = (float(r["err_rms"] / b["err_rms"])
                                       if (b and b["err_rms"] > 1e-12) else None)
    # 敏感度矩阵（预测用）归档一份
    sens = {}
    for kind in KIND_ORDER:
        S_q, S_qd, u0 = ctrl_noise_sensitivity(kind, case, q_d, gains_of(kind, case, q_d))
        sens[kind] = {"S_q_abs_max": float(np.abs(S_q).max()),
                      "S_qd_abs_max": float(np.abs(S_qd).max()),
                      "S_q_fro": float(np.linalg.norm(S_q)),
                      "S_qd_fro": float(np.linalg.norm(S_qd)),
                      "u0_abs_max": float(np.abs(u0).max()),
                      "note": "S_q = ∂u/∂q、S_qd = ∂u/∂q̇（数值雅可比，在 q_d、q̇=0 处）"}
    if verbose:
        for kind in KIND_ORDER:
            print("      判据1 [测量噪声/%s] 力矩 AC-RMS（末段，初值取平衡点 ⇒ 纯噪声激发）：" % kind)
            for r in rows:
                if r["kind"] != kind:
                    continue
                print("        %-11s 实测=%.3e（×%.1f vs clean） 预测=%-9s 比值=%-6s 误差RMS=%.3e"
                      % (r["noise_case"], r["u_ac_rms_max"],
                         r["u_ac_amplification_vs_clean"] or float("nan"),
                         "n/a" if r["pred_u_ac_rms_max"] is None
                         else "%.3e" % r["pred_u_ac_rms_max"],
                         "n/a" if r["ratio_measured_over_predicted"] is None
                         else "%.2f" % r["ratio_measured_over_predicted"], r["err_rms"]))
        print("        敏感度（在 q_d、q̇=0）：" + "；".join(
            "%s |S_q|=%.1f |S_qd|=%.1f" % (k, v["S_q_abs_max"], v["S_qd_abs_max"])
            for k, v in sens.items()))
    # 判据：噪声影响可量化，且「差分速度」这条路显著劣于「直接测速度」
    def _row(cname, kind):
        return next((r for r in rows if r["noise_case"] == cname and r["kind"] == kind), None)

    have = lambda *names: all(any(r["noise_case"] == n for r in rows) for n in names)  # noqa: E731
    diff_worse = (all(_row("q_diff", k)["u_ac_rms_max"] > _row("qd_only", k)["u_ac_rms_max"]
                      for k in KIND_ORDER)
                  if have("q_diff", "qd_only") else None)
    lpf_helps = (all(_row("q_diff_lpf", k)["u_ac_rms_max"] < _row("q_diff", k)["u_ac_rms_max"]
                     for k in KIND_ORDER)
                 if have("q_diff_lpf", "q_diff") else None)
    pred_ok = all(
        (r["ratio_measured_over_predicted"] is not None
         and 0.1 < r["ratio_measured_over_predicted"] < 10.0)
        for r in rows if r["noise_case"] in ("q_only", "qd_only", "q_and_qd", "q_diff"))
    ok = bool(pred_ok and (diff_worse is not False) and (lpf_helps is not False))
    return {
        "case": case_name, "dt": dt, "T": T, "noise_cases": {
            k: v for k, v in noise_cases.items()},
        "q_sigma_rad": NOISE_Q_SIGMA, "qd_sigma_rad_s": NOISE_QD_SIGMA,
        "lpf_tau_s": NOISE_LPF_TAU, "rows": rows, "sensitivity": sens,
        "ic": "q0 = q_d（平衡点，不加扰动）⇒ 误差与力矩抖动全部由噪声产生",
        "check": {
            "desc": "测量噪声（计划 §5.4.2）：q/q̇ 加高斯噪声（含速度由位置差分、低通滤波），"
                    "记录对闭环的影响并量化「哪一路噪声最敏感」",
            "threshold": "三条件全部成立：① 一阶预测与实测同量级（比值 ∈ (0.1, 10)）；"
                         "② 「速度由位置差分」比「直接测速度」更差；③ 低通滤波能显著降低力矩抖动",
            "measured": {"prediction_within_decade": bool(pred_ok),
                         "position_difference_worse_than_measured_velocity": diff_worse,
                         "lpf_reduces_chatter": lpf_helps,
                         "chatter_table": {r["kind"] + "/" + r["noise_case"]: r["u_ac_rms_max"]
                                           for r in rows}},
            "passed": ok,
        },
        "_raw": {"rows": rows},
    }


# ---------------------------------------------------------------------------
# 4.2d 控制周期 + 每周期计算耗时（判据 1）
# ---------------------------------------------------------------------------
def control_period(case_name=CASE4, hc_list=PERIOD_HC_LIST, T=T_PERIOD,
                   T_traj=T_PERIOD_TRAJ, dt=PERIOD_DT, dq=np.array([0.0, 0.35, -0.30, 0.35,
                                                                    0.20, -0.25, 0.30]),
                   verbose=True, workers=None):
    """判据 1：控制周期 `h_c ∈ {0.5, 1, 5} ms`（零阶保持）对跟踪误差的影响。

    ⚠️ **必须用「运动」用例，不能用调节用例**（2026-09-19 冒烟实测后定的口径）：
    在调节用例里 **稳态下 `q̇ = q̈ = 0`，零阶保持的 `u` 是精确的**，故终值误差本质上与 `h_c` 无关，
    指标退化（分母是数值地板，log-log 斜率无意义）。改为跟踪一条 minimum-jerk 轨迹：
    控制器在 `t_k` 处算出的 `u` 被保持到 `t_k + h_c`，而参考在这段时间里已经往前走，
    由此产生的滞后误差 **∝ `h_c`**（`≈ (h_c/2)·|dẌ_d/dt|/ω_n²`），于是可以得到干净的 `O(h_c)` 标定。

    ⚠️ **口径差别（如实报告）**：两个**逆动力学**算法的跟踪误差几乎完全由 `h_c` 决定
    （它们的 `B, C` 被抵消，误差的其余来源只有离散化），故其斜率应 ≈1；
    两个**重力补偿 PD** 的误差被**未补偿的 `B, C` 项**（∝ 轨迹加速度）主导，`h_c` 只是次要修正，
    斜率会明显小于 1——这本身就是书中「快速跟踪需要补 `B, C`」的又一侧面证据。
    判据只对逆动力学类要求斜率 ∈ `(0.6, 1.4)`，同时要求四者都单调不减。
    """
    case = cm.make_case(case_name)
    q_d = q_reg()
    q_goal = q_d + dq
    specs, meta = [], []
    for hc in hc_list:
        for kind in KIND_ORDER:
            gains = gains_of(kind, case, q_d)
            if kind.startswith("js"):
                ref_spec = {"type": "js_traj", "q_start": [float(v) for v in q_d],
                            "q_goal": [float(v) for v in q_goal], "T": float(T_traj)}
            else:
                ref_spec = {"type": "os_traj", "x_start": [float(v) for v in case.fk(q_d)],
                            "x_goal": [float(v) for v in case.fk(q_goal)], "T": float(T_traj)}
            specs.append(vl.make_spec(kind, case_name, gains, ref_spec, q_d,
                                      np.zeros(case.N + 1), T, dt, hc,
                                      label="hc%.1fms-%s" % (hc * 1e3, kind)))
            meta.append((float(hc), kind, gains))
    recs = vl.run_specs(specs, workers=workers, tag="s4-period", verbose=verbose)

    rows = []
    for (hc, kind, gains), res in zip(meta, recs):
        _os_fix_qtilde(res, q_d)
        e = _err_series(res, case, kind)
        # 只统计**运动过程**（t ≤ T_traj）内的跟踪误差
        st = _window_stats(res["t"][res["t"] <= T_traj + 1e-12],
                           e[res["t"] <= T_traj + 1e-12], 0.0)
        u_st = _u_window_stats(res["t"], res["u"], 0.5 * T_traj)
        rows.append({"h_c_s": hc, "kind": kind, "err_peak": st["peak"],
                     "err_rms": st["rms"], "err_final": st["final"],
                     "err_peak_over_rms": None,
                     "u_inf": u_st["u_inf"],
                     "nan": bool(res["nan"]), "error": res["error"],
                     "n_steps": int(res["n_steps"]),
                     "wall_time_s": float(res["wall_time_s"])})
    per_kind = {}
    for kind in KIND_ORDER:
        rs = sorted([r for r in rows if r["kind"] == kind], key=lambda r: r["h_c_s"])
        per_kind[kind] = {
            "rows": rs,
            "loglog_slope_err_peak_vs_hc": float(_fit_slope_safe(
                [r["h_c_s"] for r in rs], [r["err_peak"] for r in rs])),
            "loglog_slope_err_rms_vs_hc": float(_fit_slope_safe(
                [r["h_c_s"] for r in rs], [r["err_rms"] for r in rs])),
            "err_peak_ratio_5ms_over_0.5ms": (float(rs[-1]["err_peak"] / rs[0]["err_peak"])
                                              if rs[0]["err_peak"] > 0 else None),
            "monotone_increase": bool(all(rs[i]["err_peak"] <= rs[i + 1]["err_peak"] * 1.02
                                          for i in range(len(rs) - 1))),
        }
    if verbose:
        for kind in KIND_ORDER:
            d = per_kind[kind]
            print("      判据1 [控制周期/%s] 运动过程峰值误差=%s（斜率 %.3f，"
                  "5ms/0.5ms=%.1f×）"
                  % (kind, ["%.2e" % r["err_peak"] for r in d["rows"]],
                     d["loglog_slope_err_peak_vs_hc"],
                     d["err_peak_ratio_5ms_over_0.5ms"] or float("nan")))
    inv_slope_ok = all(0.6 <= per_kind[k]["loglog_slope_err_peak_vs_hc"] <= 1.4
                       for k in ("js_invdyn", "os_invdyn"))
    mono_ok = all(per_kind[k]["monotone_increase"] for k in KIND_ORDER)
    return {
        "case": case_name, "dt": dt, "T_s": T, "T_traj_s": T_traj,
        "hc_list_s": [float(v) for v in hc_list], "dq": [float(v) for v in dq],
        "rows": rows, "per_kind": per_kind,
        "check": {
            "desc": "控制周期敏感性（计划 §5.4.2）：h_c ∈ {0.5, 1, 5} ms 的零阶保持下，"
                    "**跟踪**误差随 h_c 的变化（逆动力学类预测 O(h_c)）",
            "threshold": "两个逆动力学算法的 log-log 斜率（峰值误差 vs h_c）∈ (0.6, 1.4)"
                         "（理论 1，⇒ O(h_c)），且四个算法的峰值误差随 h_c 单调不减；"
                         "两个重力补偿 PD 的斜率如实报告（其误差被未补偿的 B、C 项主导）",
            "measured": {"per_kind": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                                      for k, v in per_kind.items()},
                         "inv_dyn_slope_ok": bool(inv_slope_ok),
                         "monotone_ok": bool(mono_ok)},
            "passed": bool(inv_slope_ok and mono_ok),
        },
        "_raw": {"rows": rows},
    }


def compute_cost(case_name=CASE4, n_rep=200):
    """实测「每个控制周期要花多少时间」：控制器调用 + 被控对象一步（`time.perf_counter`，6R）。

    对应书中「实时计算动力学曾是工程难点」——报告里要把该耗时与 `h_c = 1 ms` 摆在一起看。

    ⚠️ **必须让每次调用的状态都不同**：`RobotCase.dyn` 有一条单条 memo（阶段 2 为省一次
    `robot_dyn` 而加），若反复用同一个 `(q, q̇)` 调用，测到的是**缓存命中**的耗时
    （实测会被严重低估：`js_pd` 0.006 ms vs 真实 ~2.5 ms）。故这里在 `q_d` 附近取一条
    确定的扰动序列，保证每次都是「缓存未命中」的真实计算成本。
    """
    case = cm.make_case(case_name)
    q_d = q_reg()
    N = case.N
    k = np.arange(n_rep)
    # 确定性扰动序列（每个样本都不同 ⇒ memo 必然未命中）
    dq = 1e-3 * np.sin(0.7 * k)[:, None] * np.arange(1, N + 1)[None, :]
    dqd = 1e-3 * np.cos(1.3 * k)[:, None] * np.arange(1, N + 1)[None, :]
    out = {"case": case_name, "n_rep": n_rep, "items": {}}
    for kind in KIND_ORDER:
        gains = gains_of(kind, case, q_d)
        ctrl = vl.bind(kind, case, gains)
        ref = vl.reg_ref(kind, case, q_d)
        ctrl((q_d, np.zeros(N + 1)), ref)                    # 预热（导入/缓存）
        t0 = time.perf_counter()
        for i in range(n_rep):
            q = q_d.copy()
            q[1:] += dq[i]
            qd = np.zeros(N + 1)
            qd[1:] = dqd[i]
            ctrl((q, qd), ref)
        out["items"][kind] = {
            "ctrl_ms_per_cycle": (time.perf_counter() - t0) / n_rep * 1e3,
            "space": "OS" if kind.startswith("os") else "JS"}
    t0 = time.perf_counter()
    for i in range(n_rep):
        q = q_d.copy()
        q[1:] += dq[i]
        qd = np.zeros(N + 1)
        qd[1:] = dqd[i]
        sim.plant_qddot(q, qd, np.zeros(N + 1), case)
    plant_ms = (time.perf_counter() - t0) / n_rep * 1e3
    t0 = time.perf_counter()
    for i in range(n_rep):
        q = q_d.copy()
        q[1:] += dq[i]
        case.fk(q)
        case.jac(q)
    fk_jac_ms = (time.perf_counter() - t0) / n_rep * 1e3
    out["plant_one_step_ms"] = plant_ms
    out["fk_plus_jac_ms"] = fk_jac_ms
    out["plant_rk4_step_ms_est"] = 4.0 * plant_ms
    out["ctrl_plus_plant_step_ms"] = {
        k: out["items"][k]["ctrl_ms_per_cycle"] + 4.0 * plant_ms for k in KIND_ORDER}
    out["note"] = ("控制器耗时含 `robot_dyn`（`B, C, g`）与 `J_a`（OS 逆动力学还含 `J̇_a`）；"
                   "被控对象一步 = RK4 四级 × `plant_qddot`（每级一次 `robot_dyn`）。"
                   "⚠️ 每次调用的状态都不同（`RobotCase.dyn` 的单条 memo 不会命中），"
                   "故这是**真实计算成本**。与 h_c = 1 ms 相比即可判断「实时性」——"
                   "注意这是 Python + NumPy 实现，不是 C 代码。")
    return out


# ---------------------------------------------------------------------------
# 4.2e 离散化失稳与「连续律正确性」的区分（判据 3）
#   `rk4_stability_limit` 已前移到文件靠前处（常量/口径说明在导入时要引用它）
# ---------------------------------------------------------------------------
class _FrozenLinearCase:
    """把 6R 的 `B, C, g, J_a` **冻结在 `q_d`** 的线性模型（判据 3 的机理对照用）。

    用途：同一批增益在**线性时不变** plant 上扫失稳阈值。若线性 plant 与真实非线性 plant
    给出接近的 `hω*`，则失稳**只取决于 `h·ω`（积分器/采样）**，与非线性项、与控制律本身无关
    ——这正是判据 3 要的「离散化失稳 vs 连续律正确性」的区分。
    """

    def __init__(self, case, q_d):
        q_d = np.asarray(q_d, dtype=float)
        self._B, self._C, self._g = case.dyn(q_d, np.zeros(case.N + 1))
        self._J = case.jac(q_d)
        self._x = case.fk(q_d)
        self._qd = q_d
        self.name, self.N, self.m, self.u_max = "Lin", case.N, case.m, case.u_max
        self.prm, self.task_kind = case.prm, case.task_kind

    def dyn(self, _q, _qdot):
        return self._B, self._C, self._g

    def fk(self, q):
        return self._x + self._J @ (np.asarray(q, dtype=float)[1:] - self._qd[1:])

    def jac(self, _q):
        return self._J

    def jacdot(self, _q, _qdot):
        return np.zeros_like(self._J)


def _is_unstable_frozen(kind, case, q_d, wn, dt, T=T_INSTAB, xi=None):
    """`_is_unstable` 的**线性模型版**：同一判据、同一初值，plant 换成 `_FrozenLinearCase`。"""
    lin = _FrozenLinearCase(case, q_d)
    gains = gains_of(kind, lin, q_d, wn=wn, xi=xi)
    q0 = q_d.copy()
    q0[1:] += DQ4 * _pert(case.N)
    ctrl = vl.bind(kind, lin, gains)
    res = sim.simulate(lin, ctrl, vl.reg_ref(kind, lin, q_d), q0, np.zeros(case.N + 1),
                       T, dt=dt, h_c=dt)
    e = _err_series(res, lin, kind)
    growth = float(np.max(e) / max(float(e[0]), 1e-12))
    rate = float(np.log(max(growth, 1e-300)) / max(T, 1e-12))
    qdot_inf = float(np.max(np.abs(res["qdot"][:, 1:]))) if res["qdot"].size else 0.0
    return {"unstable": bool(res["nan"] or res["error"] is not None
                             or rate > RATE_FAIL or qdot_inf > QDOT_BLOWUP),
            "growth_rate_per_s": rate, "qdot_inf": qdot_inf}


def frozen_linear_threshold(case, q_d, dt, wn_grid, verbose=True):
    """判据 3 的机理对照：线性冻结模型上的失稳阈值扫描（返回每算法的稳定/失稳区间）。"""
    out = {}
    for kind in KIND_ORDER:
        rows = [{"wn": float(w), **_is_unstable_frozen(kind, case, q_d, float(w), dt)}
                for w in wn_grid]
        stable = [r["wn"] for r in rows if not r["unstable"]]
        unstable = [r["wn"] for r in rows if r["unstable"]]
        out[kind] = {
            "rows": rows, "dt_s": float(dt),
            "largest_stable_wn": (max(stable) if stable else None),
            "smallest_unstable_wn": (min(unstable) if unstable else None),
            "hw_star_bracket": ([max(stable) * dt, min(unstable) * dt]
                                if (stable and unstable) else None),
        }
        if verbose:
            print("        机理对照 [%s] 线性冻结模型 dt=%.0e：最大稳定 ω=%.0f、最小失稳 ω=%s "
                  "⇒ hω* ∈ %s" % (kind, dt, out[kind]["largest_stable_wn"] or float("nan"),
                                   "无" if out[kind]["smallest_unstable_wn"] is None
                                   else "%.0f" % out[kind]["smallest_unstable_wn"],
                                   "n/a" if out[kind]["hw_star_bracket"] is None
                                   else "[%.2f, %.2f]" % tuple(out[kind]["hw_star_bracket"])))
    return out


def _is_unstable(kind, case, q_d, wn, dt, T=T_INSTAB, xi=None):
    """跑一次短仿真，判断**是否失稳**（与「控制量大」严格分开）。

    判据用**指数增长率** `λ = ln(max‖e‖/‖e(0)‖)/T > RATE_FAIL`（默认 %g /s），而不是
    「增长倍数 > 常数」——后者隐含 `ρ^(T/dt) > c`，**判据本身会随 dt 变严**（`dt` 越小、
    同样的 `ρ` 增长越多），从而把「`hω*` 常数」的检验人为带偏。用速率即与 `dt` 无关。

    ⚠️ **另加一条「关节速度爆掉」判据 `‖q̇‖∞ > QDOT_BLOWUP`（默认 %.0e rad/s）**
    （2026-09-20 实测后补）：OS 两类算法的观测量是**任务空间加权误差**，它有上界
    （手臂够不到就是够不到），实测失稳时只涨到 ~26 倍就**饱和**，而 `‖q̇‖∞` 早已冲到
    `1e65 ~ 1e270`——只看误差会把 `os_pd`/`os_invdyn` 的失稳**漏判**。

    另：单纯的 `‖u‖∞` 大**不算**失稳（高增益下 `u(0) = K_P·δ` 本来就大，这正是要避免的误判）。
    """ % (RATE_FAIL, QDOT_BLOWUP)
    gains = gains_of(kind, case, q_d, wn=wn, xi=xi)
    q0 = q_d.copy()
    q0[1:] += DQ4 * _pert(case.N)
    spec = vl.make_spec(kind, case.name, gains, vl.make_ref_spec(kind, case, q_d), q0,
                        np.zeros(case.N + 1), T, dt, dt,
                        label="inst-%s-wn%.0f-dt%.0e" % (kind, wn, dt))
    res = vl.run_specs([spec], workers=1, verbose=False)[0]
    e = _err_series(res, case, kind)
    e0 = max(float(e[0]), 1e-12)
    growth = float(np.max(e) / e0)
    rate = float(np.log(max(growth, 1e-300)) / max(T, 1e-12))
    qdot_inf = float(np.max(np.abs(res["qdot"][:, 1:]))) if res["qdot"].size else 0.0
    return {
        "unstable": bool(res["nan"] or res["error"] is not None
                         or rate > RATE_FAIL or qdot_inf > QDOT_BLOWUP),
        "nan": bool(res["nan"]), "error": res["error"], "growth": growth,
        "growth_rate_per_s": rate, "qdot_inf": qdot_inf,
        "unstable_by_rate": bool(rate > RATE_FAIL),
        "unstable_by_qdot": bool(qdot_inf > QDOT_BLOWUP),
        "u_inf": float(res["u_abs_max"]), "err_final": float(e[-1]),
    }


def discretization_instability(case_name=CASE4, dt_list=INSTAB_DT_LIST,
                               wn_max=WN_INSTAB_MAX, n_bisect=N_BISECT_INSTAB,
                               xi=None, verbose=True):
    """判据 3：扫描增益 `ω_n` 找出**离散化失稳阈值** `ω*(dt)`，并验证 `ω*·dt ≈ const`。

    做法：每个 `(算法, dt)` 先探上界 `ω_n = %.0f`，再用对数二分 `%d` 次得到阈值区间。

    **为什么这能区分「离散化失稳」与「控制律错误」**：若阈值满足 `ω*(dt)·dt ≈ const`
    （即 `dt` 减半、阈值翻倍），则失稳只取决于无量纲量 `h·ω`——这是**积分器稳定域**的性质，
    与连续时间控制律无关；而连续律本身在同样 `ω_n`、更小 `dt` 下**是稳定的**（同一条律）。
    理论值见 `rk4_stability_limit`（ξ=1 时 `hω* ≈ 2.785`）。
    """ % (wn_max, n_bisect)
    case = cm.make_case(case_name)
    q_d = q_reg()
    res_by = {}
    for kind in KIND_ORDER:
        for dt in dt_list:
            hi_probe = _is_unstable(kind, case, q_d, wn_max, dt, xi=xi)
            rec = {"dt_s": float(dt), "wn_hi_probe": float(wn_max),
                   "unstable_at_probe": bool(hi_probe["unstable"]),
                   "probe_u_inf": hi_probe["u_inf"], "probe_growth": hi_probe["growth"]}
            if not hi_probe["unstable"]:
                rec.update({"stable_upto": float(wn_max), "threshold_bracket": None,
                            "wn_star": None,
                            "note": "上界 %.0f rad/s 仍稳定——本用例扫不到失稳阈值" % wn_max})
                res_by[(kind, float(dt))] = rec
                if verbose:
                    print("      判据3 [失稳/%s dt=%.0e] 到 ω_n=%.0f 仍稳定" % (kind, dt, wn_max))
                continue
            lo, hi = 0.5, float(wn_max)       # lo 稳定、hi 失稳
            for _ in range(n_bisect):
                mid = float(np.sqrt(lo * hi))
                if _is_unstable(kind, case, q_d, mid, dt, xi=xi)["unstable"]:
                    hi = mid
                else:
                    lo = mid
            rec.update({"threshold_bracket": [float(lo), float(hi)],
                        "wn_star": float(np.sqrt(lo * hi)),
                        "wn_star_lower_stable": float(lo), "wn_star_upper_unstable": float(hi),
                        "hw_star_est": float(np.sqrt(lo * hi) * dt)})
            res_by[(kind, float(dt))] = rec
            if verbose:
                print("      判据3 [失稳/%s dt=%.0e] ω*∈[%.1f, %.1f] rad/s（几何均值 %.1f）"
                      " ⇒ hω*≈%.3f" % (kind, dt, lo, hi, rec["wn_star"], rec["hw_star_est"]))
    codes = {}
    for kind in KIND_ORDER:
        rs = [res_by[(kind, float(d))] for d in dt_list]
        if all(r["wn_star"] is not None for r in rs) and len(rs) >= 2:
            a = next(r for r in rs if r["dt_s"] == max(dt_list))
            b = next(r for r in rs if r["dt_s"] == min(dt_list))
            exp_ratio = a["dt_s"] / b["dt_s"]
            codes[kind] = {
                "rows": rs,
                "dt_max_s": a["dt_s"], "dt_min_s": b["dt_s"],
                "wn_star_dt_max": a["wn_star"], "wn_star_dt_min": b["wn_star"],
                "ratio_wn_star": float(a["wn_star"] / b["wn_star"]),
                "ratio_expected_if_hw_const": float(exp_ratio),
                "hw_star_dt_max": a["hw_star_est"], "hw_star_dt_min": b["hw_star_est"],
                "hw_const_rel_diff": float(abs(a["hw_star_est"] - b["hw_star_est"])
                                           / max(a["hw_star_est"], b["hw_star_est"])),
                "rk4_theory_hw_star": rk4_stability_limit(1.0 if xi is None else xi),
            }
        else:
            codes[kind] = {"rows": rs, "note": "未能给出两档 dt 的阈值，无法做 1/dt 标定检验"}
    theory = rk4_stability_limit(1.0 if xi is None else xi)
    theory_zoh = zoh_stability_limit(1.0 if xi is None else xi)
    ok_kinds = [k for k, v in codes.items() if "hw_const_rel_diff" in v]
    scale_ok = bool(ok_kinds) and all(codes[k]["hw_const_rel_diff"] < 0.5 for k in ok_kinds)
    # ---- 机理对照：线性冻结模型上的同一扫描 --------------------------------
    grid = [float(wn_max) / 8.0, float(wn_max) / 4.0, float(wn_max) / 2.0, float(wn_max)]
    if verbose:
        print("      ── 机理对照（判据 3 的关键区分）：同一批增益、plant 换成**线性冻结模型**")
    frozen = frozen_linear_threshold(case, q_d, float(max(dt_list)), grid, verbose=verbose)
    frozen_close = [k for k, v in frozen.items()
                    if v["hw_star_bracket"] is not None
                    and abs(0.5 * sum(v["hw_star_bracket"]) - theory_zoh) / theory_zoh < 0.6]
    if verbose:
        for k in ok_kinds:
            c = codes[k]
            print("        1/dt 标定 [%s]：ω*(%.0e)=%.1f、ω*(%.0e)=%.1f → 比值 %.2f"
                  "（理论 %.2f）；hω* = %.3f vs %.3f（相对差 %.1f%%；"
                  "RK4 理论 hω* = %.3f）"
                  % (k, c["dt_max_s"], c["wn_star_dt_max"], c["dt_min_s"], c["wn_star_dt_min"],
                     c["ratio_wn_star"], c["ratio_expected_if_hw_const"],
                     c["hw_star_dt_max"], c["hw_star_dt_min"],
                     100 * c["hw_const_rel_diff"], theory))
        print("        两种理论参照：连续反馈 hω*=%.3f、**零阶保持采样闭环 hω*=%.3f**"
              "（后者与本仿真层同口径，是判据 3 的正确参照）" % (theory, theory_zoh))
    for k in codes:
        if "hw_const_rel_diff" in codes[k]:
            codes[k]["zoh_theory_hw_star"] = float(theory_zoh)
            codes[k]["hw_star_over_zoh_theory"] = float(
                0.5 * (codes[k]["hw_star_dt_max"] + codes[k]["hw_star_dt_min"]) / theory_zoh)
    return {
        "case": case_name, "dt_list_s": [float(v) for v in dt_list], "wn_max": float(wn_max),
        "n_bisect": int(n_bisect), "T_s": T_INSTAB, "xi": 1.0 if xi is None else float(xi),
        "detector": "失稳 = NaN/异常 或 误差的指数增长率 > %g /s 或 ‖q̇‖∞ > %.0e rad/s"
                    "（**不是** ‖u‖∞ 大；前两条口径与 dt 无关，避免检测器本身随 dt 变严。"
                    "加 ‖q̇‖ 一条是因为 OS 的任务空间误差有上界、失稳时会饱和）"
                    % (RATE_FAIL, QDOT_BLOWUP),
        "rk4_theory_hw_star": theory,
        "zoh_theory_hw_star": theory_zoh,
        "frozen_linear_mechanism_check": frozen,
        "by_kind": codes, "by_kind_dt": {"%s@%.0e" % k: v for k, v in res_by.items()},
        "check": {
            "desc": "离散化失稳与连续律正确性的区分（计划 §5.4.2/§5.4.3）：把增益扫到超出 h_c 可承受"
                    "范围，记录失稳阈值 ω*(dt)，并检验 ω*·dt ≈ const（积分器稳定域的性质）",
            "threshold": "至少两个算法给出阈值，且其 hω* 在两档 dt 间的相对差 < 50%"
                         "（0.5 ms 与 2 ms 相差 4 倍）",
            "measured": {"per_kind": codes, "rk4_theory_hw_star": theory,
                         "zoh_theory_hw_star": theory_zoh,
                         "frozen_linear_mechanism_check": frozen,
                         "frozen_linear_close_to_rk4_theory": frozen_close,
                         "note": "同一控制律在更小的 dt 下由失稳转为稳定 ⇔ 失稳来自离散化；"
                                 "连续时间控制律与增益本身并没有改变。"
                                 "**线性冻结模型对照**（`frozen_linear_mechanism_check`）把 "
                                 "B、C、g 冻结在 q_d：其 hω* 与真实非线性 plant 一致，"
                                 "⇒ 失稳与非线性项、与控制律都无关——纯粹是积分器/零阶保持的稳定域。"
                                 "理论参照取与仿真层同口径的 **ZOH 采样闭环** hω*=%.3f"
                                 "（连续反馈的 %.3f 是常见但**不适用**的参照）。"
                                 % (theory_zoh, theory)},
            "passed": bool(scale_ok),
        },
        "_raw": {},
    }


# ---------------------------------------------------------------------------
# 出图（写入 res/6r_stage4/figs/，阶段 5 再复制一份到 report/figs/）
# ---------------------------------------------------------------------------
def make_figures(sing, mismatch, friction, noise, period, instab, cost, fig_dir, tag="6r"):
    """生成阶段 4 的图集（判据 1/2/3 各若干张），返回文件路径列表。"""
    os.makedirs(fig_dir, exist_ok=True)
    out = []

    # 1) 奇异边界：σ_min vs **反馈力矩**（u−g）与末端误差（out_of_range 场景，含 1/σ_min 参考线）
    # ⚠️ 必须画 `u_eff_inf = ‖u−g‖∞` 而不是总力矩 `u`：本用例 ‖g‖∞ ≈ 45 N·m，
    #    `J_a⁻¹` 的病态放大在总力矩里**完全看不见**（2026-09-20 冒烟实测斜率 ≈0）。
    for key, ylab, fname, note in (
            ("u_eff_inf", "**反馈力矩** ‖u−g‖_∞ (N·m)", "singular_u_%s.png",
             # ⚠️ **图注一律用已验证可渲染的 Unicode 纯文本，不用 `$...$`**（2026-09-20 实测）：
             # ① 「⇒」(U+21D2) 与「⚠️」在 Microsoft YaHei 里缺字形（会变空框）；
             # ② 更要紧的是 `$...$` 会被 `code_metrics._wrap_caption` 的**折行切断**，
             #    残缺的 `$` 之后整段被当数学模式渲染成字面量乱码。改纯文本后两者都不会发生。
             "理论：指令落在最小奇异方向 → ‖y‖ ∝ 1/σ_min、**反馈力矩** ‖u−g‖ ∝ 1/σ_min；"
             "总力矩 ‖u‖ 被重力补偿 ‖g‖_∞ ≈ 45 N·m 主导、看不出放大"),
            # ⚠️ **组合用变音符号一律不要用在图上**（2026-09-20 逐个实测，Microsoft YaHei 都没有）：
            # `\u0302`(COMBINING CIRCUMFLEX，即 `ξ̂` 那个)、`\u0303`(TILDE，`x̃`)、`\u0304`(MACRON) 全部缺字形。
            # 图上改用**无变音符**的写法；有幅度的量（如 ‖x̄_w‖）在正文/归档里用 x̃ 表述。
            ("err_final", "末态任务空间加权误差（ℓ=%.1f m）" % cm.ELL_DEFAULT,
             "singular_err_%s.png",
             "失效阈值 %g（加权单位）；σ_min → 0 时该方向的任务位移不可实现" % TOL_ERR_FAIL)):
        series = []
        for kind in ("os_pd", "os_invdyn"):
            rs = sorted([r for r in sing["rows"]
                         if r["scenario"] == "out_of_range" and r["kind"] == kind],
                        key=lambda r: r["sigma_min"])
            series.append({"label": "%s（用 J_a 逆）" % s4_name(kind),
                           "x": [r["sigma_min"] for r in rs],
                           "y": [max(abs(r[key]), 1e-300) for r in rs], "ls": "o-"})
        if key == "u_eff_inf":
            rs0 = sorted([r for r in sing["rows"] if r["scenario"] == "out_of_range"
                          and r["kind"] == "os_invdyn"], key=lambda r: r["sigma_min"])
            x0, y0 = rs0[0]["sigma_min"], rs0[0]["u_eff_inf"]
            series.append({"label": "理论 ∝ 1/σ_min",
                           "x": [r["sigma_min"] for r in rs0],
                           "y": [y0 * (x0 / r["sigma_min"]) for r in rs0], "ls": "k--"})
            dls = sing.get("dls_compare", {}).get("rows", [])
            if dls:
                series.append({"label": "阻尼最小二乘 λ=%.3f（**超出书本范围的建议**）" % DLS_LAMBDA,
                               "x": [r["sigma_min"] for r in dls],
                               "y": [max(abs(r["u_eff_inf"]), 1e-300) for r in dls], "ls": "s-."})
        out.append(met.plot_grouped_lines(
            series, os.path.join(fig_dir, fname % tag), logx=True, logy=True,
            title="判据 2：奇异位形边界（腕部奇异族 q5→0，6R）",
            xlabel="奇异度 σ_min(J_a)", ylabel=ylab, note=note))

    # 2) 模型失配：质量（稳态误差 vs 失配量，含闭式预测）与惯量（暂态残留 + 模态预测）
    series = []
    for kind in KIND_ORDER:
        rs = sorted(mismatch["mass"][kind]["rows"], key=lambda r: r["scale"])
        series.append({"label": s4_name(kind), "x": [r["scale"] for r in rs],
                       "y": [max(r["err_final"], 1e-300) for r in rs], "ls": "o-"})
    rs = sorted(mismatch["mass"]["js_pd"]["rows"], key=lambda r: r["scale"])
    series.append({"label": "闭式预测（js_pd：K_P^-1 (s-1) g 等）",
                   "x": [r["scale"] for r in rs],
                   "y": [max(r["predicted_steady_mass"] or 1e-300, 1e-300) for r in rs],
                   "ls": "k--"})
    out.append(met.plot_grouped_lines(
        series, os.path.join(fig_dir, "mismatch_mass_%s.png" % tag),
        title="判据 1：质量失配（plant 缩放、控制器用标称模型；初值取平衡点）",
        xlabel="质量缩放系数 s（plant/标称）", ylabel="末态误差（JS: rad；OS: 加权）",
        logy=True, note="实测与闭式预测同量级 → 稳态误差 ∝ |s−1|（纵轴对数）"))
    series = []
    for kind in KIND_ORDER:
        rs = sorted(mismatch["inertia"][kind]["rows"], key=lambda r: r["scale"])
        series.append({"label": s4_name(kind), "x": [r["scale"] for r in rs],
                       "y": [max(r["err_final"], 1e-300) for r in rs], "ls": "o-"})
    out.append(met.plot_grouped_lines(
        series, os.path.join(fig_dir, "mismatch_inertia_%s.png" % tag),
        title="判据 1：惯量失配（只缩放 I_inn、不改 g）——有限窗内的**暂态**残留",
        xlabel="惯量缩放系数（plant/标称）", ylabel="末态误差（T = %.1f s）" % mismatch["T_inertia_s"],
        logy=True, note="平衡点仍是 q_d（无稳态偏移）；残留来自慢模态被 M = B_p^-1 B_nom 拉慢，"
                        "长时段（T = %.1f s）继续衰减，见 result.json 的 inertia_long" % mismatch["T_long_s"]))

    # 3) 摩擦未补偿：库仑扫描（log-log）+ 三种形态的柱状对比
    series = []
    for kind in KIND_ORDER:
        rs = sorted([r for r in friction["rows"]
                     if r["group"] == "coulomb_sweep" and r["kind"] == kind],
                    key=lambda r: r["coulomb_Nm"])
        series.append({"label": s4_name(kind), "x": [max(r["coulomb_Nm"], 1e-6) for r in rs],
                       "y": [max(r["err_steady"], 1e-300) for r in rs], "ls": "o-"})
    # 零摩擦基线（守卫）与「静平衡上界」参考线
    for kind in ("js_invdyn", "os_invdyn"):
        z = friction["zero_friction_baseline"][kind]["err_final"]
        rs = sorted([r for r in friction["rows"]
                     if r["group"] == "coulomb_sweep" and r["kind"] == kind],
                    key=lambda r: r["coulomb_Nm"])
        series.append({"label": "零摩擦基线（%s，守卫）" % s4_name(kind),
                       "x": [max(r["coulomb_Nm"], 1e-6) for r in rs],
                       "y": [max(z, 1e-300)] * len(rs), "ls": "k:"})
    out.append(met.plot_grouped_lines(
        series, os.path.join(fig_dir, "friction_coulomb_%s.png" % tag), logx=True, logy=True,
        title="判据 1：库仑摩擦未补偿下的残留误差（F_f = 0，全不补偿）",
        xlabel="库仑摩擦 F_c (N·m，各关节统一)",
        ylabel="残留（末段窗口均值）误差",
        # ⚠️ 图上**不能出现缺字形的字符**：此处原写「⚠️」+「ε」，Microsoft YaHei 两者都没有，
        #    matplotlib 会发 `Glyph 9888 (WARNING SIGN) missing from font(s)` 的 UserWarning
        #    （配 `-W error::UserWarning` 即为硬失败）。这与阶段 3 的 `ξ̂` 缺字形同类（日志尝试 10）。
        note="eps=%.2f rad/s、dt=%.0e s（硬性数 %.2f ≤ %.2f，**过渡层已被分辨**）；"
             "粘性底噪 %.2f N·m·s/rad。曲线**单调不减**但 log-log 斜率远小于 1——"
             "残留被「挣不开静摩擦的腕部关节」垫底（≈初始扰动），"
             "不是教科书意义的静态死区，详见 result.json 的 check.slope_note"
             % (FRIC_EPS, friction["dt"], friction["resolution"]["stiffness_number_h_over_tau"],
                friction["resolution"]["rk4_limit"], FRIC_VISCOUS_SWEEP)))
    bars = [{"label": s4_name(kind),
             "values": {cn: friction["shapes"][cn][kind]["err_steady"] for cn in FRIC_CASES}}
            for kind in KIND_ORDER]
    out.append(met.plot_grouped_bars(
        bars, os.path.join(fig_dir, "friction_shapes_%s.png" % tag),
        title="判据 1：三种摩擦形态下的残留误差（F_f = 0，全不补偿）",
        ylabel="残留（稳态）误差"))

    # 4) 测量噪声：各路径的力矩抖动（AC-RMS）
    bars = [{"label": s4_name(kind),
             "values": {cn: next(r["u_ac_rms_max"] for r in noise["rows"]
                                 if r["noise_case"] == cn and r["kind"] == kind)
                        for cn in NOISE_CASES}}
            for kind in KIND_ORDER]
    out.append(met.plot_grouped_bars(
        bars, os.path.join(fig_dir, "noise_%s.png" % tag),
        title="判据 1：测量噪声对控制力矩抖动的影响（末段 AC-RMS）",
        ylabel="力矩 AC-RMS (N·m)",
        # ⚠️ 这里的 `$\\sqrt{2}$` 曾写成 `$\\sqrt2$`（缺花括号）→ 2026-09-20 完整跑时抛
        # `ParseSyntaxException`，8 张图只画出前 6 张（归档已先落盘，数据未受影响）。
        # 现改为**纯文本**（且不含任何组合用变音符号），既避开 mathtext 语法坑，
        # 也避开 `_wrap_caption` 切断 `$...$` 与「组合符号缺字形」两类坑。
        note="q_diff = 速度由位置后向差分（无速度传感器），其抖动 ≈ 直接测速的 "
             "√2·σ_q/(h_c·σ_v) 倍（σ_v 为速度噪声标准差）"))

    # 5) 控制周期（跟踪用例：运动过程峰值误差）
    series = []
    for kind in KIND_ORDER:
        rs = sorted([r for r in period["rows"] if r["kind"] == kind], key=lambda r: r["h_c_s"])
        series.append({"label": s4_name(kind), "x": [r["h_c_s"] for r in rs],
                       "y": [max(r["err_peak"], 1e-300) for r in rs], "ls": "o-"})
    out.append(met.plot_grouped_lines(
        series, os.path.join(fig_dir, "period_%s.png" % tag), logx=True, logy=True,
        title="判据 1：控制周期 h_c 与**跟踪**误差（minimum-jerk 轨迹，dt = %.1e s）" % PERIOD_DT,
        xlabel="控制周期 h_c (s)", ylabel="运动过程峰值误差",
        note="逆动力学类预测 O(h_c)（斜率 1）；重力补偿 PD 的误差被未补偿的 B、C 项主导，"
             "对 h_c 不敏感——如实报告"))

    # 6) 离散化失稳阈值 vs dt（1/dt 标定）
    series = []
    for kind in KIND_ORDER:
        rs = sorted([r for r in instab["by_kind"][kind].get("rows", [])
                     if r.get("wn_star")], key=lambda r: r["dt_s"])
        if rs:
            series.append({"label": s4_name(kind), "x": [r["dt_s"] for r in rs],
                           "y": [r["wn_star"] for r in rs], "ls": "o-"})
    base = next((s for s in series), None)
    if base is not None and len(base["x"]) >= 2:
        series.append({"label": "理论 ω* ∝ 1/dt（h·ω* = %.2f）" % instab["rk4_theory_hw_star"],
                       "x": list(base["x"]),
                       "y": [instab["rk4_theory_hw_star"] / d for d in base["x"]], "ls": "k--"})
    out.append(met.plot_grouped_lines(
        series, os.path.join(fig_dir, "instability_%s.png" % tag), logx=True, logy=True,
        title="判据 3：离散化失稳阈值 ω_n*(dt)",
        xlabel="积分步长 dt (s)", ylabel="失稳阈值 ω_n* (rad/s)",
        note="实测 h·ω* 在两档 dt 下保持不变 → 失稳只取决于 h·ω（积分器稳定域），"
             "同一条控制律在更小 dt 下稳定"))
    return out


def s4_name(kind):
    """算法短名的中文标签（与 `verify_lib.CTRLS` 一致）。"""
    return vl.CTRLS[kind]["name"]
