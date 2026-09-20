# -*- coding: utf-8 -*-
"""Part IV 机器人控制验证的**用例模型**：DH 与连杆参数的**唯一定义处**。

红线（见书仓库 `docs/robot-control-verify/part4-motion-control-plan.md` §5.2.2）：
**DH 与连杆参数只能有一处定义**，全部阶段、两个用例都从这里取。

| 用例 | 任务空间 | 用途 |
|---|---|---|
| `TWO_LINK_2R` | 位置子任务 2 维 `[x, y]`（见下方⚠️） | 阶段 2 首轮贯通；阶段 0/3 的**机器精度真值基准** |
| `PUMA_6R` | 6 维 `[x, y, z, ϑx, ϑy, ϑz]`（ZXY 欧拉角） | 阶段 3 正式验证；阶段 4 奇异与耦合测试 |

⚠️ **为什么 2R 的模型自带闭式**：`model/code_jacobian.py` 的 `robot_jacobian_a` 硬编码 6×6
输出（位置 3 + ZXY 欧拉角 3），**不能用**于平面 2R（其任务空间只有 3 维）。按计划 §5.0.1 的
做法，本文件自带精简版 `robot_fk_2r` / `robot_jacobian_a_2r` / `robot_jacobian_dot_a_2r`
（闭式、机器精度），**不去改** `model/` 的副本——改了会破坏它与 `Codes/` 原件的同步对拍。

⚠️ **2R 的「控制用任务空间」取 2 维位置子任务 `[x, y]`（2026-09-18 阶段 2 修订，见日志尝试 8）**：
`robot_fk_2r` 输出 3 维 `[x, y, φ]`，其分析雅可比 `J_a` 是 **3×2**（满列秩的浸入），
于是书里 OS 逆动力学算法框的 `J_a^{-1}` **不存在**——2 自由度手臂无法独立指令 3 个任务坐标。
为使 4 个算法都能严格按书中公式落地，OS 用例的任务空间取 **位置子任务 `[x, y]`（`J_a` 为 2×2 方阵）**；
`robot_jacobian_a_2r_pos` / `robot_jacobian_dot_a_2r_pos` / `robot_ik_2r_pos` 即该子任务的闭式实现。
（`(x, y, φ) ↔ q` 是双射，位置子任务与全任务在本质上是同一件事；`φ` 行被丢弃不损失信息。）

约定（与 Part II 一致）：长度为 `N+1`、下标 `1..N` 与数学记号对齐，`[0]` 不使用。

运行环境：`E:\\Anaconda3\\envs\\py311-gym\\python.exe`（加 `PYTHONUTF8=1`）。
"""
import os as _os
import sys as _sys

import numpy as np

# 本文件在 `code/` 下，需 `Robot-Control/` 进 sys.path 才能 `from model import ...`（可在任意 cwd 运行）
_RC = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _RC not in _sys.path:
    _sys.path.insert(0, _RC)

from model import (  # noqa: E402
    robot_dyn,
    robot_fk,
    robot_ik_lm,
    robot_jacobian_a,
    robot_jacobian_dot_a,
)

# ===========================================================================
# 用例 1：平面 2R（任务空间 [x, y, φ]）
# ===========================================================================
# 参数口径与 `Codes/chap2_4_dynamics/test_dynamics.py`、`Codes/chap2_3_diffk/test_jacobian.py`
# 的平面臂用例一致（L1=1.0, L2=0.8, m1=2.0, m2=1.5, I1=0.1, I2=0.08），便于与 Part II 结论对照。
L1_2R, L2_2R = 1.0, 0.8
M1_2R, M2_2R = 2.0, 1.5
I1_2R, I2_2R = 0.1, 0.08

# 臂在 xy 平面内运动，重力取面内 -y 方向，故 g(q) ≠ 0（Part II 的 test_dynamics 用
# g0=[0,0,-9.81] 只检验了 g ≡ 0 的退化情形；Part II 的 test_model_sync 用 [0,-9.81,0]）。
G0_2R = np.array([0.0, -9.81, 0.0])

# 标称位形（避开 φ 的 atan2 分支与奇异；供各阶段复用同一组基线输入）
Q_NOM_2R = np.array([0.0, 0.7, -0.5])
QDOT_NOM_2R = np.array([0.0, 0.9, -0.6])

# **稳定**重力平衡位形（`∂U/∂q = 0` 且 `B⁻¹∂²U/∂q²` 的特征值 ≥ 0，即 U 的极小点）。
# 阶段 0.5 的静平衡测试必须在稳定平衡位上做：在不稳定平衡位上，机器 eps 级的初值偏差
# 会被指数放大（实测增长率与线性化预测 `sqrt(|λ_min|)` 一致），10 s 后必然「漂移」——
# 那是物理不稳定，不是模型错误。详见书仓库日志 `part4-motion-control-verify/…-log.md` 尝试 2。
# 2R：两杆自然下垂（解析解，λ = [8.175, 56.60] > 0）；
# 6R：由 `∂U/∂q = 0` 的阻尼 Newton 抛光得到（|g| ≤ 3e-15，λ = [0, 21.0, 28.0, 43.6, 75.7, 83.4]，
#     其中一个零特征值对应绕基座 z 轴的中性方向）。q6 已折回 (-π, π]。
Q_EQ_2R = np.array([0.0, -np.pi / 2, 0.0])
Q_EQ_6R = np.array([0.0, 0.0, -np.pi / 2, -2.3510482238, 0.0, -0.7905444299, 0.0])


def two_link_2r():
    """平面 2R 的 DH 与连杆参数（与 Part II 平面臂用例同口径）。

    返回 `(d, a, alpha, m, p_cents, I_inn, g0)`，长度均为 `N+1=3`（下标 1..2，`[0]` 不用）。
    D-H 第 n 号连杆系原点位于关节 n+1 处，故质心（连杆中点）在连杆系下为 `[-L_n/2, 0, 0]`。
    """
    d = np.zeros(3)
    a = np.array([0.0, L1_2R, L2_2R])
    alpha = np.zeros(3)
    m = np.array([0.0, M1_2R, M2_2R])
    p_cents = np.array([[0.0, 0.0, 0.0],
                        [-L1_2R / 2, 0.0, 0.0],
                        [-L2_2R / 2, 0.0, 0.0]])
    I_inn = np.zeros((3, 3, 3))
    I_inn[1] = np.diag([0.0, 0.0, I1_2R])      # 连杆系下绕自身 z 轴（关节轴）的惯量
    I_inn[2] = np.diag([0.0, 0.0, I2_2R])
    return d, a, alpha, m, p_cents, I_inn, G0_2R


def robot_fk_2r(q):
    """2R 正运动学**闭式**：返回任务空间 `x_e = [x, y, φ]`（3 维）。

    与 D-H 参数（`a=[0,L1,L2]`、`alpha=0`、`d=0`、标准 D-H）一致：
    `x = L1 c1 + L2 c12`、`y = L1 s1 + L2 s12`、`φ = q1 + q2`（ZXY 欧拉角退化情形）。
    """
    q1, q2 = q[1], q[2]
    c1, s1 = np.cos(q1), np.sin(q1)
    c12, s12 = np.cos(q1 + q2), np.sin(q1 + q2)
    return np.array([L1_2R * c1 + L2_2R * c12,
                     L1_2R * s1 + L2_2R * s12,
                     q1 + q2])


def robot_jacobian_a_2r(q):
    """2R 分析雅可比**闭式**：`ẋ_e = J_a q̇`，`J_a` 为 3×2。

    平面情形下 `φ̇ = ω_z`，故这里的 `J_a` 同时等于几何雅可比 `J_w` 的 3×2 形式
    （位置块相同，角速度行 `[0, 0, 1]` 退化为 `∂φ/∂q = [1, 1]`）。
    """
    q1, q2 = q[1], q[2]
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([
        [-L1_2R * s1 - L2_2R * s12, -L2_2R * s12],
        [L1_2R * c1 + L2_2R * c12, L2_2R * c12],
        [1.0, 1.0],
    ])


def robot_jacobian_dot_a_2r(q, qdot):
    """2R `J̇_a(q, q̇)` **闭式**（机器精度真值基准，用于检验 6R 的差分实现）。

    `J̇_a = Σ_k ∂J_a/∂q_k q̇_k`，逐元素手推：
    `d/dt(-L1 s1 - L2 s12) = -L1 c1 q̇1 - L2 c12 (q̇1+q̇2)`，其余同理；第 3 行为 0。
    """
    q1, q2 = q[1], q[2]
    qd1, qd2 = qdot[1], qdot[2]
    qd12 = qd1 + qd2
    c1, s1 = np.cos(q1), np.sin(q1)
    c12, s12 = np.cos(q1 + q2), np.sin(q1 + q2)
    return np.array([
        [-L1_2R * c1 * qd1 - L2_2R * c12 * qd12, -L2_2R * c12 * qd12],
        [-L1_2R * s1 * qd1 - L2_2R * s12 * qd12, -L2_2R * s12 * qd12],
        [0.0, 0.0],
    ])


# --- 2R 的「控制用任务空间」= 位置子任务 [x, y]（2 维方阵，见文件头 ⚠️）------
def robot_fk_2r_pos(q):
    """2R 位置子任务正运动学：`x_e = [x, y]`（2 维）。"""
    return robot_fk_2r(q)[:2]


def robot_jacobian_a_2r_pos(q):
    """2R 位置子任务雅可比 `J_a`（2×2 方阵）：`J = [[-L1s1-L2s12, -L2s12], [L1c1+L2c12, L2c12]]`。

    `det J = L1·L2·sin(q2)`（闭式），故 `q2 = 0, ±π` 为奇异位形（σ_min → 0）。
    """
    return robot_jacobian_a_2r(q)[:2, :]


def robot_jacobian_dot_a_2r_pos(q, qdot):
    """2R 位置子任务 `J̇_a`（2×2）：取 3 维闭式 `J̇_a` 的位置两行（第 3 行恒为 0）。"""
    return robot_jacobian_dot_a_2r(q, qdot)[:2, :]


def robot_ik_2r_pos(x_e, q_seed):
    """2R 位置子任务**闭式逆解**：由 `[x, y]` 解 `q`，取离 `q_seed` 最近的一支。

    ⚠️ **只用于「评测参照」**（评估 OS 算法达到的位形对应的关节角），
    **绝不可把解出的 `q_d` 喂给 OS 控制器**——那等价于把 OS 控制偷偷变成 JS 控制（计划 §5.2.4 红线）。

    返回长度 `N+1=3` 的 `q`（`[0]` 不用），不可达时返回 `None`。
    """
    x, y = float(x_e[0]), float(x_e[1])
    r2 = x * x + y * y
    c2 = (r2 - L1_2R ** 2 - L2_2R ** 2) / (2.0 * L1_2R * L2_2R)
    if abs(c2) > 1.0 + 1e-12:
        return None
    c2 = float(np.clip(c2, -1.0, 1.0))
    s2_abs = float(np.sqrt(max(0.0, 1.0 - c2 * c2)))
    cands = []
    for s2 in (s2_abs, -s2_abs):
        q2 = float(np.arctan2(s2, c2))
        q1 = float(np.arctan2(y, x) - np.arctan2(L2_2R * s2, L1_2R + L2_2R * c2))
        cands.append(np.array([0.0, q1, q2]))
    # 关节角按 2π 周期比较（差取到 (-π, π]）
    def _wrap(dq):
        return (dq + np.pi) % (2.0 * np.pi) - np.pi

    best = min(cands, key=lambda qc: float(np.max(np.abs(_wrap(qc - q_seed)))))
    return best


# ===========================================================================
# 用例 2：6R（Puma 560 型，标准 D-H，重力 -z）
# ===========================================================================
# D-H 取经典 Puma 560 的标准 D-H（与 Corke 机器人工具箱 p560 的关节表一致，
# T = Rz(θ)·Tz(d)·Tx(a)·Rx(α)，与 `model/code_fk.py` 的 `calc_T_n_to_last` 同构）。
#
# ⚠️ **连杆惯性是简化值**：按各连杆的长度尺度当成均匀细杆估算（质量取常用量级），
# **不是 Puma 560 的官方惯性参数**。本批次只要求物理合理、`B(q)` 正定、量级可信；
# 与 Pinocchio/RTB 对拍时两侧传入**同一组数组**，故简化值不影响对拍的有效性。
PUMA_D = np.array([0.0, 0.0, 0.0, 0.15005, 0.4318, 0.0, 0.0])
PUMA_A = np.array([0.0, 0.0, 0.4318, 0.0203, 0.0, 0.0, 0.0])
PUMA_ALPHA = np.array([0.0, np.pi / 2, 0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, 0.0])
PUMA_M = np.array([0.0, 10.0, 17.4, 4.8, 0.82, 0.34, 0.09])
G0_6R = np.array([0.0, 0.0, -9.81])

# 标称位形：远离肘部/腕部奇异，且 ZXY 欧拉角不落在 ϑx = ±π/2 的万向锁上
# ⚠️ 该位形在重力下是**不稳定**平衡（`B⁻¹∂²U/∂q²` 有负特征值，实测增长率 ≈ 6.67 /s，
#    见 Q_EQ_6R 上方说明）——用于控制验证没问题，但不能用作静平衡测试的位形。
Q_NOM_6R = np.array([0.0, 0.30, -0.60, 0.80, 0.40, -0.50, 0.60])
QDOT_NOM_6R = np.array([0.0, 0.70, -0.50, 0.60, -0.40, 0.80, 0.50])


def puma_6r():
    """Puma 560 型 6R 的 D-H 与连杆参数（简化惯性，见上方说明）。

    连杆长度尺度 `L_n = max(|a_n|, |d_n|, 0.2)`；质心取连杆系下 `[-L_n/2, 0, 0]`
    （D-H 第 n 号连杆系原点在关节 n+1 处，质心在连杆中段故为负 x）；
    惯量按均匀细杆 `diag(0.05, 1, 1)·m L²/12`（绕连杆轴向的滚转惯量取小值）。
    """
    N = 6
    p_cents = np.zeros((N + 1, 3))
    I_inn = np.zeros((N + 1, 3, 3))
    for n in range(1, N + 1):
        L = max(abs(PUMA_A[n]), abs(PUMA_D[n]), 0.2)
        p_cents[n] = [-L / 2, 0.0, 0.0]
        I_inn[n] = np.diag([0.05, 1.0, 1.0]) * PUMA_M[n] * L ** 2 / 12.0
    return PUMA_D, PUMA_A, PUMA_ALPHA, PUMA_M, p_cents, I_inn, G0_6R


# ===========================================================================
# 控制用例的统一接口（阶段 2 起）：仿真层与 4 个控制器只依赖 `RobotCase`
# ===========================================================================
# 任务空间维数：2R 取位置子任务 2 维（见文件头 ⚠️），6R 取 6 维位姿。
TASK_DIM_2R, TASK_DIM_6R = 2, 6

# 关节力矩上限（N·m）：判据「‖u‖_∞ 在用例的力矩上限内（不饱和）」用。
# Puma 560 级关节力矩量级为 ~100–200 N·m，本批次统一取 200 N·m 作为**宽松上限**；
# 它不是电机真实饱和值，只用于「控制量是否量级失控」这一判定。
U_MAX_2R, U_MAX_6R = 200.0, 200.0

# 调节用例的目标位形（阶段 2 的主力用例）。
# 2R：q2 = 1.3 rad 远离奇异（det J = L1L2 sin q2 = 0.77），且位形在可达域内。
# 6R：复用标称位形（远离肘/腕奇异与万向锁）。
Q_D_REG_2R = np.array([0.0, 0.3, 1.3])
Q_D_REG_6R = Q_NOM_6R.copy()

# 初值采样范围（判据「≥20 组初值全部收敛」用）：q0 = q_d + U(-d,d)，q̇0 = U(-d,d)
IC_DQ_MAX, IC_DQDOT_MAX = 0.4, 0.3

# 任务空间误差的等效臂长 ℓ（计划 §5.3.6 / §7「关键约定」6）：标量范数用
# ‖diag(1,1,1,ℓ,ℓ,ℓ)⁻¹x̃‖₂ = √(‖x̃_pos‖² + ‖x̃_rot‖²/ℓ²)，默认 ℓ = 0.5 m。
ELL_DEFAULT = 0.5


class RobotCase:
    """一个控制用例（2R / 6R）的统一接口：仿真层与控制器只依赖它。

    提供与**书中算法框同一个层次**的原语（数学记号同名）：

    | 方法 | 数学量 | 书中算法框的对应 |
    |---|---|---|
    | `dyn(q, q̇)` | `B, C, g` | `B, C, g ← robot_dyn(...)` |
    | `fk(q)` | `x_e`（`m` 维） | `x_e ← robot_fk(...)` |
    | `jac(q)` | `J_a`（`m×N`） | `J_a ← robot_jacobian_a(...)` |
    | `jacdot(q, q̇)` | `J̇_a`（`m×N`） | `J̇_a ← auto_diff(J_a, t)` |
    | `ik(x_e, q_seed)` | `q` | **仅评测参照**（红线：绝不喂给 OS 控制器） |

    `dyn` 内置**单条 memo**（纯函数，按 `(q, q̇)` 的字节精确匹配）：RK4 的首级与控制器
    恰好在同一个 `(q, q̇)` 上求值，可省掉一次完整 `robot_dyn`（2R 实测 ~1.07 ms/次）。
    返回的 `B, C, g` 是**共享引用，调用方不得就地修改**。
    """

    def __init__(self, name, N, m, task_kind, u_max, prm, fk, jac, jacdot, ik,
                 plant=None):
        self.name = name
        self.N = N
        self.m = m
        self.task_kind = task_kind          # "position"（纯平移）或 "pose"（平移+旋转）
        self.u_max = float(u_max)
        self.prm = prm
        # 失配规格（阶段 4）：空的 `{}` 表示标称模型；非空时本 case 只能当**被控对象**，
        # 控制器必须另建一个标称 case（见 `test/ctrl_common/verify_lib.run_task`）。
        self.plant = dict(plant or {})
        self.fk = fk
        self.jac = jac
        self.jacdot = jacdot
        self.ik = ik
        self._memo_key = None
        self._memo_val = None

    def dyn(self, q, qdot):
        """返回 `(B, C, g)`，满足 `B q̈ + C q̇ + g = τ`（含单条 memo，见类文档）。"""
        key = (q.tobytes(), qdot.tobytes())
        if key == self._memo_key:
            return self._memo_val
        d, a, alpha, m, p_cents, I_inn, g0 = self.prm
        val = robot_dyn(q, qdot, d, a, alpha, m, p_cents, I_inn, g0, self.N)
        self._memo_key, self._memo_val = key, val
        return val

    def __repr__(self):
        return "RobotCase(%s, N=%d, m=%d)" % (self.name, self.N, self.m)


def perturb_params(prm, plant=None):
    """按**失配规格**缩放连杆参数，返回**新的** `prm` 元组（阶段 4：被控对象 ≠ 控制器模型）。

    | 键 | 含义 | 对模型的影响 |
    |---|---|---|
    | `mass_scale` | 所有连杆质量 ×此系数 | `B`、`g` 同时线性缩放（`g` 只依赖质量与质心位置） |
    | `inertia_scale` | 绕质心惯量 `I_inn` ×此系数 | **只影响 `B`**（`g` 与惯量无关）——故预期「无稳态误差、只改暂态」 |
    | `com_scale` | 质心位置 `p_cents` ×此系数 | `B` 与 `g` 都变（力臂变了） |
    | `gravity_scale` | 重力向量 ×此系数 | **只影响 `g`** |

    `plant=None` 或 `{}` 时原样返回（标称）。**不改原件**：本函数返回新数组，
    调用方（`make_case`）拿到的永远是新构造的 `prm`。
    """
    if not plant:
        return prm
    d, a, alpha, m, p_cents, I_inn, g0 = prm
    ms = float(plant.get("mass_scale", 1.0) or 1.0)
    i_s = float(plant.get("inertia_scale", 1.0) or 1.0)
    c_s = float(plant.get("com_scale", 1.0) or 1.0)
    g_s = float(plant.get("gravity_scale", 1.0) or 1.0)
    return (np.array(d, dtype=float), np.array(a, dtype=float),
            np.array(alpha, dtype=float), np.array(m, dtype=float) * ms,
            np.array(p_cents, dtype=float) * c_s,
            np.array(I_inn, dtype=float) * i_s,
            np.array(g0, dtype=float) * g_s)


def make_case(name, plant=None):
    """按名字构造用例：`"2R"`（平面两杆，位置子任务）或 `"6R"`（Puma 560 型位姿任务）。

    `plant`（阶段 4 新增）：**参数失配规格**（见 `perturb_params`）。给定时返回的 case
    是**被控对象**，其 `prm` 已按规格缩放；控制器必须另建一个 `plant=None` 的标称 case。
    """
    if name == "2R":
        return RobotCase(
            "2R", N=2, m=TASK_DIM_2R, task_kind="position", u_max=U_MAX_2R,
            prm=perturb_params(two_link_2r(), plant),
            fk=robot_fk_2r_pos,
            jac=robot_jacobian_a_2r_pos,
            jacdot=robot_jacobian_dot_a_2r_pos,
            ik=robot_ik_2r_pos,
            plant=plant,
        )
    if name == "6R":
        d, a, alpha = PUMA_D, PUMA_A, PUMA_ALPHA
        N = 6

        def _fk(q):
            return robot_fk(q, d, a, alpha, N)

        def _jac(q):
            return robot_jacobian_a(q, d, a, alpha, _fk(q)[3:], N)

        def _jacdot(q, qdot):
            return robot_jacobian_dot_a(q, qdot, d, a, alpha, N)

        def _ik(x_e, q_seed, eps=1e-10):
            # 数值逆解（阶段 3 的评测参照用；比默认 epsilon 更严）
            return robot_ik_lm(x_e, q_seed, d, a, alpha, N, epsilon=eps)

        return RobotCase(
            "6R", N=N, m=TASK_DIM_6R, task_kind="pose", u_max=U_MAX_6R,
            prm=perturb_params(puma_6r(), plant), fk=_fk, jac=_jac, jacdot=_jacdot, ik=_ik,
            plant=plant,
        )
    raise ValueError("未知用例 %r（可选 '2R' / '6R'）" % (name,))
