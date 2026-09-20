# -*- coding: utf-8 -*-
"""操作空间（OS）两个控制器的落地实现——**严格按书 Part IV 算法框**逐行对应。

| 本文件函数 | 书 `\\label` | 书中控制律 |
|---|---|---|
| `ctrl_os_pd` | `algo:robctrl_os_pd` | `u = J_aᵀ(K_P(x_d − x_e) − K_D J_aq̇) + g` |
| `ctrl_os_invdyn` | `algo:robctrl_os_invdyn` | `y = J_a⁻¹(K_Px̃ + K_D(ẋ_d − J_aq̇) + ẍ_d − J̇_aq̇)`，`u = B y + Cq̇ + F_fq̇ + g` |

变量名与书中公式**严格同名**（`x_e`/`J_a`/`J̇_a`/`x_d`/`x_tilde`/`y`/`B`/`C`/`g`/`F_f`/`u`）。

⚠️ **`J_a⁻¹` 的落地方式**：书中写 `J_a^{-1}`（要求 `J_a` 方阵且列满秩，即 `m = N`）。
本实现用 `np.linalg.solve`（数学上等价、数值上比显式求逆稳定）；两个用例都保证 `m = N`
（2R 取位置子任务 `m=2`，6R 取位姿 `m=6`，见 `code_models` 文件头 ⚠️）。
`J_a` 接近奇异时 `solve` 结果会放大——这正是书里「`J_a` 列满秩」前提的边界，
阶段 4 用 `σ_min(J_a)` 扫描量化，本批次**不做**阻尼最小二乘（那超出书本范围）。

⚠️ **书中 `ẋ̃` 用一阶近似 `ẋ_e ≈ J_aq̇`**（严格说 `ẋ_e = J_wq̇`，而 `J_aq̇` 是「欧拉角导数」）。
本实现照书中写法落地；该近似的偏差阶段 3.5 量化（阶段 0 已记录基线：6R 标称位形 `‖J_wq̇ − J_aq̇‖/‖J_wq̇‖ = 0.276`）。

运行环境：`E:\\Anaconda3\\envs\\py311-gym\\python.exe`（加 `PYTHONUTF8=1`）。
"""
import numpy as np


def ctrl_os_pd(state, ref, gains, case, F_f=None):
    """书 `algo:robctrl_os_pd`：操作空间**重力补偿 PD 调节**控制律。

    ```
    x_e ← robot_fk(q)
    J_a ← robot_jacobian_a(q, x_e)
    g   ← robot_dyn(q, …)
    u   ← J_aᵀ ( K_P (x_d − x_e) − K_D J_a q̇ ) + g
    ```

    `F_f` 未在书中该算法框出现，故忽略（与 `ctrl_js_pd` 同理）。
    """
    q, qdot = state
    K_P, K_D = gains["K_P"], gains["K_D"]
    x_d = ref["x_d"]
    x_e = case.fk(q)
    J_a = case.jac(q)
    _B, _C, g = case.dyn(q, qdot)
    u = np.zeros(case.N + 1)
    u[1:] = J_a.T @ (K_P @ (x_d - x_e) - K_D @ (J_a @ qdot[1:])) + g
    return u


def ctrl_os_invdyn(state, ref, gains, case, F_f=None):
    """书 `algo:robctrl_os_invdyn`：操作空间**逆动力学 PD 跟踪**控制律。

    ```
    x_e ← robot_fk(q)
    J_a ← robot_jacobian_a(q, x_e)
    J̇_a ← auto_diff(J_a, t)
    B, C, g ← robot_dyn(q, q̇, …)
    y ← J_a⁻¹ ( K_P (x_d − x_e) + K_D (ẋ_d − J_a q̇) + ẍ_d − J̇_a q̇ )
    u ← B y + C q̇ + F_f q̇ + g
    ```

    `J̇_a` 由 `model/code_jacobian_dot.py` 提供（Part II 未实现，阶段 0.1 补齐，误差 ~1e-10）。
    """
    q, qdot = state
    K_P, K_D = gains["K_P"], gains["K_D"]
    x_d, xdot_d, xddot_d = ref["x_d"], ref["xdot_d"], ref["xddot_d"]
    x_e = case.fk(q)
    J_a = case.jac(q)
    Jdot_a = case.jacdot(q, qdot)
    B, C, g = case.dyn(q, qdot)
    rhs = (K_P @ (x_d - x_e) + K_D @ (xdot_d - J_a @ qdot[1:])
           + xddot_d - Jdot_a @ qdot[1:])
    y = np.linalg.solve(J_a, rhs)                 # 书中 J_a^{-1}
    u = np.zeros(case.N + 1)
    u[1:] = B @ y + C @ qdot[1:] + g
    if F_f is not None:
        u[1:] += F_f @ qdot[1:]
    return u
