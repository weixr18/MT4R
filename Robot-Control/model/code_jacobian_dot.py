# -*- coding: utf-8 -*-
"""`J̇_a(q, q̇)` —— 分析雅可比的时间导数（Part IV 阶段 0.1 的**硬阻塞项**补实现）。

> ⚠️ **本文件不是 `Codes/` 的副本**，而是 `Robot-Control/` 自有的实现：
> Part II 的 `Codes/chap2_3_diffk/code_jacobian.py` **没有** `J̇_a`，
> 而书 Part IV 的 OS 逆动力学算法框（`algo:robctrl_os_invdyn`）写了 `J̇_a ← auto_diff(J_a, t)`。
> 故这里新增实现，**不改** `model/code_jacobian.py` 等 4 个副本
> （改了会破坏 `test/model/test_model_sync.py` 与 `Codes/` 原件的哈希/数值对拍）。
> 未纳入 `test_model_sync.py` 的副本对拍范围。

实现方式（见书仓库 `docs/robot-control-verify/part4-motion-control-plan.md` §5.0.1 的取舍表）：

    J̇_a(q, q̇) = Σ_{k=1}^{N} ∂J_a(q)/∂q_k · q̇_k

对 `q` 逐列**中心差分**再加权，`O(h²)` 精度——与 `model/code_dynamics.py` 里
`robot_dyn` 构造 `C` 的做法一致（同一精度档次，便于阶段 0.2/3 一并量化残差）。

⚠️ **易漏的一步**：`J_a(q) = diag{I₃, T_ϑ(ϑ)⁻¹} J_w(q)`，其中欧拉角 `ϑ = ϑ(q)` **也随 q 变化**。
差分 `∂J_a/∂q_k` 时必须在扰动后的 `q` 上**重算欧拉角**，否则漏掉 `T_ϑ⁻¹` 随 `ϑ(q)` 的那一项。
`test/model/test_jacobian_dot.py` 的乘积法则测试专门捕获这类错误。

⚠️ **已知限制**：ZXY 欧拉角在 `|cos ϑx| → 0`（万向锁）处 `T_ϑ⁻¹` 奇异，`J_a` 本身发散；
本实现不回避该位形（与 `robot_jacobian_a` 同性质），用时需保证位形远离万向锁。

运行环境：`E:\\Anaconda3\\envs\\py311-gym\\python.exe`（加 `PYTHONUTF8=1`）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # model/ 内扁平导入
from code_fk import robot_fk                                      # noqa: E402
from code_jacobian import robot_jacobian_a                        # noqa: E402


def jacobian_dot_fd(f_jac, q, qdot, h=1e-6):
    """通用 `J̇ = Σ_k ∂J/∂q_k · q̇_k`（对 q 逐列中心差分再加权），`O(h²)`。

    参数
    ----
    f_jac : callable
        `q -> J`，返回形状 `(m, N)` 的数组；**调用方负责 J 对 q 的全部依赖**
        （如 `J_a` 依赖的欧拉角必须在 `f_jac` 内部随 q 重算）。
    q, qdot : ndarray, shape (N+1,)
        下标对齐约定：`[0]` 不用，`1..N` 有效。
    h : float
        中心差分步长（默认 `1e-6`）。

    返回
    ----
    ndarray, shape (m, N)
    """
    J0 = np.asarray(f_jac(q))
    m, N = J0.shape
    Jdot = np.zeros((m, N))
    for k in range(1, N + 1):
        qp = q.copy()
        qm = q.copy()
        qp[k] += h
        qm[k] -= h
        Jdot += qdot[k] * (np.asarray(f_jac(qp)) - np.asarray(f_jac(qm))) / (2.0 * h)
    return Jdot


def robot_jacobian_dot_a(q, qdot, d, a, alpha, N=6, h=1e-6):
    """6R（通用 N 关节）分析雅可比的时间导数 `J̇_a(q, q̇)`，形状 `(6, N)`。

    与 `robot_jacobian_a` 同一套约定（位置 3 行 + ZXY 欧拉角 3 行，`J_a = diag{I₃, T_ϑ⁻¹}J_w`）；
    对应书 Part IV 算法框里的 `J̇_a ← auto_diff(J_a, t)`。

    参数
    ----
    q, qdot : ndarray, shape (N+1,)
    d, a, alpha : ndarray, shape (N+1,)
        D-H 参数（见 `code/code_models.py`，全项目唯一定义处）。
    N : int
        自由度数（默认 6）。
    h : float
        中心差分步长（默认 `1e-6`）。误差量级见 `test/model/test_jacobian_dot.py` 的收敛阶实测。

    返回
    ----
    ndarray, shape (6, N)
    """

    def f_jac(qq):
        # 关键：欧拉角随 q 重算，否则 ∂J_a/∂q_k 少掉 T_ϑ(ϑ(q))⁻¹ 的那一项
        eular_e = robot_fk(qq, d, a, alpha, N)[3:]
        return robot_jacobian_a(qq, d, a, alpha, eular_e, N)

    return jacobian_dot_fd(f_jac, q, qdot, h)
