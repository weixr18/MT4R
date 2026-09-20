# -*- coding: utf-8 -*-
"""参考轨迹生成（计划 §5.2.4）——**避免循环论证**的关键一层。

红线（`Robot-Control/AGENTS.md`「关键约定」5，最容易被无意识违反的一条）：

- **JS 用例**在**关节空间**生成 `q_d(t), q̇_d(t), q̈_d(t)`；
- **OS 用例**必须在**操作空间**生成 `x_d(t), ẋ_d(t), Ẍ_d(t)`，**绝不能**把逆解出的 `q_d` 喂给 OS 控制器
  ——那等价于把 OS 控制偷偷变成 JS 控制，期望曲线与实测曲线会因共享逆解而重合，验证失效。
  逆解只允许用于**评测参照**（`code_models.robot_ik_2r_pos` / `RobotCase.ik`）。

提供的参考量形式：**时间恒定的字典**（调节用例）或**关于 `t` 的函数**（跟踪用例），
仿真层 `code_sim.simulate` 两者都接受。键名与书中记号同名：
JS 用 `q_d/qdot_d/qddot_d`，OS 用 `x_d/xdot_d/xddot_d`。

跟踪轨迹用**五次多项式（minimum jerk）**：`s(τ) = 10τ³ − 15τ⁴ + 6τ⁵`（`τ = t/T`），
首末两端位置、速度、加速度**六条件全满足**（`s'(0)=s'(T)=0`、`s''(0)=s''(T)=0`），
各阶导数均为**解析式**——逆动力学控制对 `q̈_d`/`ẍ_d` 的精度直接敏感，
用数值微分得到的参考加速度会成为额外的有界扰动，故这里不用插值表的差分。

> 阶段 3 的 JS 跟踪用例将按计划 §5.2.4 复用 `Codes/chap1_4_interp` 的插值（同样是解析系数、
> 可给出各阶导数），本文件先提供解析五次多项式；两者都是「在关节空间生成轨迹」，不违反红线。

运行环境：`E:\\Anaconda3\\envs\\py311-gym\\python.exe`（加 `PYTHONUTF8=1`）。
"""
import numpy as np


def _quintic(t, T):
    """minimum-jerk 五次多项式的 `(s, s', s'')`（`τ = t/T`；`t ≥ T` 后保持 `s=1, s'=s''=0`）。"""
    tau = min(max(t / T, 0.0), 1.0)
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    ds = 30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4
    dds = 60 * tau - 180 * tau ** 2 + 120 * tau ** 3
    return s, ds, dds


# ---------------------------------------------------------------------------
# JS：关节空间
# ---------------------------------------------------------------------------
def js_const_ref(q_d):
    """关节空间**调节**参考：`q_d` 常值，`q̇_d = q̈_d = 0`（长度 `N+1`，`[0]` 不用）。"""
    q_d = np.asarray(q_d, dtype=float)
    z = np.zeros_like(q_d)
    return {"q_d": q_d, "qdot_d": z, "qddot_d": z.copy()}


def js_traj_ref(q_start, q_goal, T):
    """关节空间**跟踪**参考：`[q_start, q_goal]` 上的 minimum-jerk 五次多项式。

    返回 `t -> {"q_d", "qdot_d", "qddot_d"}`（长度 `N+1`，`[0]` 恒为 0）。
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    N = len(q_start) - 1
    delta = q_goal[1:] - q_start[1:]

    def _ref(t):
        s, ds, dds = _quintic(t, T)
        q_d = np.zeros(N + 1)
        q_d[1:] = q_start[1:] + delta * s
        qdot_d = np.zeros(N + 1)
        qdot_d[1:] = delta * ds / T
        qddot_d = np.zeros(N + 1)
        qddot_d[1:] = delta * dds / (T * T)
        return {"q_d": q_d, "qdot_d": qdot_d, "qddot_d": qddot_d}

    return _ref


# ---------------------------------------------------------------------------
# OS：操作空间
# ---------------------------------------------------------------------------
def os_const_ref(x_d):
    """操作空间**调节**参考：`x_d` 常值，`ẋ_d = Ẍ_d = 0`（长度 `m`）。"""
    x_d = np.asarray(x_d, dtype=float)
    z = np.zeros_like(x_d)
    return {"x_d": x_d, "xdot_d": z, "xddot_d": z.copy()}


def os_traj_ref(x_start, x_goal, T):
    """操作空间**跟踪**参考：`[x_start, x_goal]` 上的 minimum-jerk 五次多项式。

    ⚠️ 端点由调用方给出**任务空间坐标**（通常取 `case.fk(q)` 的结果），
    中间点在任务空间插值——这正是「在操作空间生成轨迹」；逆解不参与参考生成。
    返回 `t -> {"x_d", "xdot_d", "xddot_d"}`（长度 `m`）。
    """
    x_start = np.asarray(x_start, dtype=float)
    x_goal = np.asarray(x_goal, dtype=float)
    delta = x_goal - x_start

    def _ref(t):
        s, ds, dds = _quintic(t, T)
        return {"x_d": x_start + delta * s,
                "xdot_d": delta * ds / T,
                "xddot_d": delta * dds / (T * T)}

    return _ref
