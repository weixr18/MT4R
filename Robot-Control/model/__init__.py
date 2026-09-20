# -*- coding: utf-8 -*-
"""`model/` —— 机器人模型库（Part II 已数值验证代码的副本）。

本包是 `4-MT4R-github/Codes/` 下 Part II 已 `$\\dag$` 代码的**副本**，供 `Robot-Control/`
独立引用（书中的 Part II 代码保持不动、仍是单一来源）。各文件顶部有 `PROVENANCE` 行记录
来源路径与同步哈希；`test/model/test_model_sync.py` 会对副本与原件做数值对拍防漂移。

| 模块 | 内容 | 对应书 |
|---|---|---|
| `code_fk` | `calc_T_n_to_last`、`rot_to_eular`、`robot_fk` | `algo:robot_fk`（Part II 正运动学） |
| `code_jacobian` | `eular_diff_to_w`、`robot_jacobian_w`、`robot_jacobian_a`、`robot_j_centroid` | `algo:robot_ja`、`algo:robot_j_centroid` |
| `code_jacobian_dot` | `jacobian_dot_fd`、`robot_jacobian_dot_a`（**Robot-Control 自有，非副本**） | Part IV `algo:robctrl_os_invdyn` 的 `J̇_a`（Part II 未实现） |
| `code_ik` | `robot_ik_gd`、`robot_ik_gauss_newton`、`robot_ik_lm` | `algo:robot-ik-*`（数值逆解） |
| `code_dynamics` | `robot_B`、`robot_dyn` | `algo:robot_dynamics` |

数组索引对齐约定（全书统一）：长度为 `N+1`、下标 `1..N` 与数学记号一致，`[0]` 不使用。

⚠️ **已知限制**：`robot_jacobian_a` 硬编码 6×6 输出（位置 3 + ZXY 欧拉角 3），
因此**不能直接用于 2R 平面臂用例**（任务空间应为 3 维 `[x, y, φ]`）。见
`docs/robot-control-verify/part4-motion-control-plan.md` 阶段 0 的说明。

⚠️ **`code_jacobian_dot.py` 不是副本**：Part II 的 `Codes/` 里没有 `J̇_a`，而 Part IV 的两个
OS 算法框都写了 `J̇_a ← auto_diff(J_a, t)`，故本包新增该模块（Robot-Control 自有），
**不参与** `test/model/test_model_sync.py` 的副本对拍。
"""
import os as _os
import sys as _sys

# 允许 `from model import robot_fk` 这类扁平导入（按需导入，避免仅用一部分时也拉入全部模块）
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from code_dynamics import robot_B, robot_dyn  # noqa: E402
from code_fk import calc_T_n_to_last, robot_fk, rot_to_eular  # noqa: E402
from code_ik import robot_ik_gauss_newton, robot_ik_gd, robot_ik_lm  # noqa: E402
from code_jacobian import (  # noqa: E402
    eular_diff_to_w,
    robot_j_centroid,
    robot_jacobian_a,
    robot_jacobian_w,
)
from code_jacobian_dot import jacobian_dot_fd, robot_jacobian_dot_a  # noqa: E402

__all__ = [
    "calc_T_n_to_last",
    "robot_fk",
    "rot_to_eular",
    "eular_diff_to_w",
    "robot_jacobian_w",
    "robot_jacobian_a",
    "robot_j_centroid",
    "jacobian_dot_fd",
    "robot_jacobian_dot_a",
    "robot_ik_gd",
    "robot_ik_gauss_newton",
    "robot_ik_lm",
    "robot_B",
    "robot_dyn",
]
