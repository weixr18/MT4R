# -*- coding: utf-8 -*-
"""JS/OS 四个控制器**共用**的东西：PD 增益参数化（书 Part IV 式 `eq:ctrl-js-pd-gains`）。

四个控制器的签名与调用约定统一为（计划 §5.2.3）：

    u = ctrl(state, ref, gains, case, F_f=None)

| 参数 | 含义 |
|---|---|
| `state` | `(q, q̇)`，均为长度 `N+1`、下标 `1..N` 有效（`[0]` 不用） |
| `ref` | 参考量字典，键与书中记号同名：JS 用 `q_d/q̇_d/q̈_d`，OS 用 `x_d/ẋ_d/Ẍ_d` |
| `gains` | `{"K_P": K_P, "K_D": K_D}`，`N×N`（JS）或 `m×m`（OS）矩阵 |
| `case` | `code_models.RobotCase`（提供 `dyn/fk/jac/jacdot` 与 `N/m/u_max`） |
| `F_f` | 摩擦矩阵（`N×N`，半正定），`None` 等价于 `F_f = 0`（阶段 2/3 的基线） |
| 返回 `u` | 长度 `N+1` 的关节力矩向量（`u[0] = 0`） |

参考轨迹的**符号约定**（与书中一致）：`q̃ = q_d − q`、`x̃ = x_d − x_e`（都是「期望 − 实际」）。

⚠️ **`ω_n` 在四个算法里的实际含义不同**（阶段 2 实测，见日志尝试 8）：
- JS 逆动力学：误差动态 `q̈̃ + K_Dq̇̃ + K_Pq̃ = 0`（`B` 被精确抵消）→ `ω_n` 即闭环带宽；
- JS 重力补偿 PD：未抵消 `B`，实际带宽 `≈ √(K_P/λ(B))`，随位形在 `√cond(B)` 倍内变化；
- OS 逆动力学：误差动态 `ẍ̃ + K_Dẋ̃ + K_Px̃ = 0`（任务空间「单位质量」）→ `ω_n` 即闭环带宽；
- OS 重力补偿 PD：未抵消任务惯量 `Λ = (J_aB⁻¹J_aᵀ)⁻¹`，实际带宽 `≈ σ_min(J_a)·√(K_P/λ_max(B))`，
  比 `ω_n` **小得多**（2R 上实测约 0.22·ω_n），故 OS PD 需要明显更大的任务空间增益。
"""
import numpy as np


def pd_gains(wn, xi, n, Lam=None):
    """把二阶系统指标 `(ω_n, ξ)` 换成 PD 增益矩阵（书 Part IV §"参数选取"的公式）。

    `K_P = diag{ω_1², …, ω_N²}`、`K_D = diag{2ξ_1ω_1, …, 2ξ_Nω_N}`；
    标量输入时对所有自由度取同一组值（对角阵）。

    `Lam`（可选）：**参考惯量矩阵**，给定时改为 `K_P = Lam·diag{ω_n²}`、`K_D = Lam·diag{2ξω_n}`。
    此时闭环在标称位形处近似「单位质量」，`ω_n` 才真正是闭环带宽。取法：
    JS 用 `B(q_d)`（关节惯量）、OS 用任务惯量 `Λ = (J_aB⁻¹J_aᵀ)⁻¹`。

    ⚠️ **为什么需要 `Lam`（阶段 2 实测，6R 冒烟时发现，留给阶段 3）**：
    `Lam = I` 只在「惯量各向同性且量级为 1」时才不产生副作用。
    6R（Puma 型，含小惯量的腕部滚转关节）实测 `cond(B) ≈ 1.7e3`，最小的 `λ_min(B)` 极小，
    此时 `K_D = 16I` 在最小惯量方向上给出 ~1e4 /s 量级的阻尼速率，**远超 `dt = 1 ms` 的 RK4 稳定域**
    （`dt·ω ≫ 2.8`），仿真会以「数值失稳」的形式炸掉（实测 20 ms 内 `‖u‖` 冲到 1e58）——
    这属于**离散化失稳**，不是控制律错误（阶段 4.3 要求把两者明确区分）。
    2R 的 `cond(B) ≈ 34`、`λ_min ≈ 0.105`，`Lam = I` 安全（`dt·(K_D/λ_min) ≈ 0.15`）。
    """
    wn = np.atleast_1d(np.asarray(wn, dtype=float))
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    if wn.size == 1:
        wn = np.full(n, wn.item())
    if xi.size == 1:
        xi = np.full(n, xi.item())
    assert wn.shape == (n,) and xi.shape == (n,)
    K_P, K_D = np.diag(wn ** 2), np.diag(2.0 * xi * wn)
    if Lam is not None:
        Lam = np.asarray(Lam, dtype=float)
        K_P, K_D = Lam @ K_P, Lam @ K_D
    return {"K_P": K_P, "K_D": K_D}
