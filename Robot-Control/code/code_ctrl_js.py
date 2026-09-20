# -*- coding: utf-8 -*-
"""关节空间（JS）两个控制器的落地实现——**严格按书 Part IV 算法框**逐行对应。

| 本文件函数 | 书 `\\label` | 书中控制律 |
|---|---|---|
| `ctrl_js_pd` | `algo:robctrl_js_pd` | `u = K_P(q_d − q) − K_D q̇ + g(q)` |
| `ctrl_js_invdyn` | `algo:robctrl_js_invdyn` | `y = K_Pq̃ + K_Dq̇̃ + q̈_d`，`u = B y + Cq̇ + F_fq̇ + g` |

变量名与书中公式**严格同名**（`K_P`/`K_D`/`B`/`C`/`g`/`y`/`q_d`/`F_f`/`u`），便于「算法 ↔ 代码」逐行对照
（`Robot-Control/AGENTS.md`「关键约定」3）。数组索引约定：长度 `N+1`、下标 `1..N`、`[0]` 不用。

`state = (q, q̇)`；`ref` 是参考量字典（键 `q_d` / `qdot_d` / `qddot_d`，均为长度 `N+1`）；
`gains = {"K_P":…, "K_D":…}`（见 `code_ctrl_common.pd_gains`）；`case` 见 `code_models.RobotCase`。

运行环境：`E:\\Anaconda3\\envs\\py311-gym\\python.exe`（加 `PYTHONUTF8=1`）。
"""
import numpy as np


def ctrl_js_pd(state, ref, gains, case, F_f=None):
    """书 `algo:robctrl_js_pd`：关节空间**重力补偿 PD 调节**控制律。

    ```
    g ← robot_dyn(q, q̇, …)
    u ← K_P (q_d − q) − K_D q̇ + g
    ```

    `F_f` 未在书中该算法框出现（书里 `F_f` 仅用于稳定性证明的半正定假设），故本函数忽略它。
    """
    q, qdot = state
    K_P, K_D = gains["K_P"], gains["K_D"]
    q_d = ref["q_d"]
    _B, _C, g = case.dyn(q, qdot)          # 书中：g ← robot_dyn(q, q̇, ...)
    u = np.zeros(case.N + 1)
    u[1:] = K_P @ (q_d[1:] - q[1:]) - K_D @ qdot[1:] + g
    return u


def ctrl_js_invdyn(state, ref, gains, case, F_f=None):
    """书 `algo:robctrl_js_invdyn`：关节空间**逆动力学 PD 跟踪**控制律。

    ```
    B, C, g ← robot_dyn(q, q̇, …)
    y ← K_P (q_d − q) + K_D (q̇_d − q̇) + q̈_d
    u ← B y + C q̇ + F_f q̇ + g
    ```

    闭环误差动态（书中推导）：`q̈̃ + K_D q̇̃ + K_P q̃ = 0`——`B`、`C`、`g` 被精确抵消，
    故「精确线性化」的精度只取决于 `robot_dyn` 的 `C`（中心差分，阶段 0 已量化为 ~1e-10）。
    """
    q, qdot = state
    K_P, K_D = gains["K_P"], gains["K_D"]
    q_d, qdot_d, qddot_d = ref["q_d"], ref["qdot_d"], ref["qddot_d"]
    B, C, g = case.dyn(q, qdot)
    y = K_P @ (q_d[1:] - q[1:]) + K_D @ (qdot_d[1:] - qdot[1:]) + qddot_d[1:]
    u = np.zeros(case.N + 1)
    u[1:] = B @ y + C @ qdot[1:] + g
    if F_f is not None:
        u[1:] += F_f @ qdot[1:]
    return u
