# -*- coding: utf-8 -*-
"""Part II 模型代码副本的**同步自检**（防止 `Robot-Control/model/` 与 `Codes/` 原件漂移）。

背景：`Robot-Control/model/` 是 `Codes/chap2_2_kinematcs/`、`chap2_3_diffk/`、`chap2_4_dynamics/`
下 Part II 代码的**副本**，目的是让 Part IV 机器人控制验证代码独立、可自包含运行。
仓库根 `AGENTS.md` 的约定是「复用优先、单一来源，不复制副本」——本目录属**有意豁免**，
代价就是两份可能漂移，故用本脚本兜底（阶段 0.4 / 判据 1）。

本脚本做两件事：

1. **哈希对拍**：对每个副本文件，按 `PROVENANCE` 行里记录的 `sha256[:16]`（**原文件**全文哈希）
   重新计算原件哈希，不一致即报「副本已过期」并给出重新同步方式；
2. **数值对拍**：在同一组随机 `(q, q̇)` 上，分别用**原件**与**副本**的
   `robot_fk` / `robot_jacobian_a` / `robot_B` / `robot_dyn` / `robot_ik_lm` 求值，
   要求逐元素一致（容差 `1e-12`）。

⚠️ **2026-09-18 修正的一处假通过**（阶段 0 执行时发现）：原实现用
`sys.path.insert(0, Codes/...)` + `import code_dynamics as dyn_orig` 取原件，但
`model/__init__.py` 早已把 `code_dynamics` 等裸名注册进 `sys.modules`（指向副本），
于是 `import code_dynamics` 直接命中缓存——**「原件」拿到的其实是副本本身，数值对拍退化为自证**
（这也解释了此前记录的 `max|Δ| = 0.000e+00`）。现改为 `load_originals()`：用
`importlib` 以独立模块名从 `Codes/` 路径加载原件，并在加载期间把裸名临时指向原件，
使其内部的 `from code_fk import ...` 绑定到**原件**；加载后校验 `__file__` 确实落在 `Codes/`，
否则直接报错。

> 数据来源：副本的 import 段是**为扁平导入改写过**的（原件用 `../chap2_2_kinematcs` 相对路径），
> 故副本文件的哈希与 `PROVENANCE` 记录的（原件的）哈希**不相等**是正常的；
> 判据看的是「原件当前哈希 == 记录哈希」。

运行（`py311-gym`，任意 cwd）：

    python Robot-Control/test/model/test_model_sync.py
"""
import hashlib
import importlib.util
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RC = os.path.abspath(os.path.join(_HERE, "..", ".."))          # Robot-Control/
_REPO = os.path.abspath(os.path.join(_RC, ".."))                # 4-MT4R-github/
sys.path.insert(0, _RC)

# 副本文件 -> 原件路径（相对仓库根 4-MT4R-github/）
COPY_TO_ORIG = {
    "model/code_fk.py": "Codes/chap2_2_kinematcs/code_fk.py",
    "model/code_jacobian.py": "Codes/chap2_3_diffk/code_jacobian.py",
    "model/code_ik.py": "Codes/chap2_3_diffk/code_ik.py",
    "model/code_dynamics.py": "Codes/chap2_4_dynamics/code_dynamics.py",
}

# 原件加载顺序（= 依赖顺序）：(裸名, 独立模块名, 原件相对路径)
ORIG_SPECS = [
    ("code_fk", "_orig_code_fk", COPY_TO_ORIG["model/code_fk.py"]),
    ("code_jacobian", "_orig_code_jacobian", COPY_TO_ORIG["model/code_jacobian.py"]),
    ("code_ik", "_orig_code_ik", COPY_TO_ORIG["model/code_ik.py"]),
    ("code_dynamics", "_orig_code_dynamics", COPY_TO_ORIG["model/code_dynamics.py"]),
]

TOL = 1e-12
SEED = 0


def _sha16(path):
    with open(path, encoding="utf-8", newline="") as f:
        raw = f.read().replace("\r\n", "\n")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _recorded_hash(copy_path):
    """从副本的 PROVENANCE 注释里取出记录的 sha256[:16]。

    PROVENANCE 是跨行注释，格式为 `... sha256(前16位) = <16位十六进制>；...`，
    故用正则精确截取十六进制串，避免把同一行后半句的说明文字也当成哈希。
    """
    with open(copy_path, encoding="utf-8") as f:
        for line in f:
            if "sha256(前16位)" in line:
                m = re.search(r"sha256\(前16位\)\s*=\s*([0-9a-f]{16})", line)
                return m.group(1) if m else None
    return None


def load_originals():
    """把 `Codes/` 下的原件加载到**独立模块命名空间**，返回 `{裸名: 模块}`。

    关键：加载期间把裸名（`code_fk` 等）临时指向已加载的原件，使其内部的
    `from code_fk import ...` 绑定到原件而不是 `model/` 的副本；加载完再还原 `sys.modules`。
    """
    loaded = {}
    for bare, alias, rel in ORIG_SPECS:
        path = os.path.join(_REPO, rel.replace("/", os.sep))
        if not os.path.exists(path):
            raise FileNotFoundError("原件不存在：%s" % path)
        saved = {k: sys.modules.get(k) for k in loaded}
        sys.modules.update(loaded)                     # 裸名 -> 原件
        try:
            spec = importlib.util.spec_from_file_location(alias, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[alias] = mod
            spec.loader.exec_module(mod)
        finally:
            for k, v in saved.items():                 # 还原裸名（仍指向副本）
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        # 守卫：原件必须真的来自 Codes/（防「再次拿到副本」这类静默退化）
        got = os.path.abspath(mod.__file__)
        want = os.path.abspath(path)
        if got != want:
            raise AssertionError("原件加载错误：期望 %s，实际 %s" % (want, got))
        loaded[bare] = mod
    return loaded


def check_hashes():
    """副本是否落后于原件（哈希对拍）。"""
    print("== 1. 哈希对拍（副本是否落后于原件）==")
    all_ok = True
    details = {}
    for rel, orig_rel in COPY_TO_ORIG.items():
        copy_path = os.path.join(_RC, rel.replace("/", os.sep))
        orig_path = os.path.join(_REPO, orig_rel.replace("/", os.sep))
        if not os.path.exists(orig_path):
            print("  [SKIP] 原件不存在：%s" % orig_rel)
            continue
        recorded = _recorded_hash(copy_path)
        actual = _sha16(orig_path)
        ok = (recorded == actual)
        all_ok &= ok
        details[rel] = {"recorded": recorded, "original_actual": actual, "passed": bool(ok)}
        print("  [%s] %-24s 记录=%s 原件=%s" % ("OK" if ok else "!!", os.path.basename(rel), recorded, actual))
        if not ok:
            print("         -> 原件已变更，请重新同步该副本（见 Robot-Control/AGENTS.md「模型库」）")

    print("  结论：%s" % ("全部一致" if all_ok else "存在过期副本，需重新同步"))
    return all_ok, details


def _two_link_params():
    """数值对拍用的**测试本地**任意输入（不是 2R 用例参数，用例参数见 `code/code_models.py`）。"""
    d = np.array([0.0, 0.0, 0.0])
    a = np.array([0.0, 0.5, 0.4])
    alpha = np.array([0.0, 0.0, 0.0])
    return d, a, alpha


def check_values():
    """原件 vs 副本的数值对拍（同参数、同输入，逐元素一致）。"""
    print("== 2. 数值对拍（原件 vs 副本，同参数同输入）==")
    from model import code_dynamics as dyn_copy
    from model import code_fk as fk_copy
    from model import code_ik as ik_copy
    from model import code_jacobian as jac_copy

    orig = load_originals()
    fk_orig, jac_orig = orig["code_fk"], orig["code_jacobian"]
    dyn_orig, ik_orig = orig["code_dynamics"], orig["code_ik"]
    print("  原件实际路径（守卫已校验）：")
    for k, m in orig.items():
        print("    %-14s %s" % (k, os.path.relpath(m.__file__, _REPO)))

    d, a, alpha = _two_link_params()
    N = 2
    rng = np.random.default_rng(SEED)
    q = np.zeros(N + 1)
    q[1:] = rng.uniform(-1.0, 1.0, N)
    qdot = np.zeros(N + 1)
    qdot[1:] = rng.uniform(-1.0, 1.0, N)
    q_0 = np.zeros(N + 1)
    q_0[1:] = rng.uniform(-0.3, 0.3, N)

    # 动力学参数（简单连杆：质心在半长处，绕自身轴的惯量）
    m = np.array([0.0, 1.0, 1.0])
    p_cents = np.zeros((N + 1, 3))
    p_cents[1] = [0.25, 0.0, 0.0]
    p_cents[2] = [0.2, 0.0, 0.0]
    I_inn = np.zeros((N + 1, 3, 3))
    I_inn[1] = np.diag([1e-3, 1e-2, 1e-2])
    I_inn[2] = np.diag([1e-3, 1e-2, 1e-2])
    g0 = np.array([0.0, -9.81, 0.0])

    checks = []
    x_c, x_o = fk_copy.robot_fk(q, d, a, alpha, N), fk_orig.robot_fk(q, d, a, alpha, N)
    checks.append(("robot_fk", x_c, x_o))

    Ja_c = jac_copy.robot_jacobian_a(q, d, a, alpha, x_c[3:], N)
    Ja_o = jac_orig.robot_jacobian_a(q, d, a, alpha, x_o[3:], N)
    checks.append(("robot_jacobian_a", Ja_c, Ja_o))

    B_c = dyn_copy.robot_B(q, d, a, alpha, m, p_cents, I_inn, N)
    B_o = dyn_orig.robot_B(q, d, a, alpha, m, p_cents, I_inn, N)
    checks.append(("robot_B", B_c, B_o))

    res_c = dyn_copy.robot_dyn(q, qdot, d, a, alpha, m, p_cents, I_inn, g0, N)
    res_o = dyn_orig.robot_dyn(q, qdot, d, a, alpha, m, p_cents, I_inn, g0, N)
    for name, vc, vo in zip(("robot_dyn[B]", "robot_dyn[C]", "robot_dyn[g]"), res_c, res_o):
        checks.append((name, vc, vo))

    # 逆解（L-M，确定性）：用原件正运动学造一个目标位姿
    x_target = fk_orig.robot_fk(q, d, a, alpha, N)
    lm_c = ik_copy.robot_ik_lm(x_target, q_0, d, a, alpha, N)
    lm_o = ik_orig.robot_ik_lm(x_target, q_0, d, a, alpha, N)
    checks.append(("robot_ik_lm", lm_c, lm_o))

    all_ok = True
    details = {}
    for name, vc, vo in checks:
        err = float(np.max(np.abs(np.asarray(vc) - np.asarray(vo))))
        ok = err <= TOL
        all_ok &= ok
        details[name] = err
        print("  [%s] %-18s max|Δ| = %.3e" % ("OK" if ok else "!!", name, err))

    print("  结论：%s" % ("数值完全一致" if all_ok else "存在数值差异，副本可能已过期或被改动"))
    return all_ok, details


def run():
    """跑 0.4 项全部对拍，返回结论字典（供 run_stage0.py 归档）。"""
    print("=" * 72)
    print("阶段 0.4  副本同步自检（哈希 + 数值，判据 1）")
    print("  副本目录：%s" % _RC)
    print("  原件仓库：%s" % _REPO)
    print("=" * 72)
    ok_h, det_h = check_hashes()
    print()
    ok_v, det_v = check_values()
    print()
    passed = bool(ok_h and ok_v)
    print("-" * 72)
    print("阶段 0.4 结论：%s" % ("PASS" if passed else "FAIL"))
    checks = {
        "sync": {
            "desc": "副本同步对拍：PROVENANCE 记录哈希 == 原件当前哈希，且原件与副本数值逐元素一致",
            "threshold": "哈希全一致；数值 max|Δ| ≤ 1e-12",
            "measured": {"hashes": det_h, "value_max_abs_diff": det_v, "seed": SEED},
            "passed": passed,
        }
    }
    return {"name": "阶段 0.4 副本同步自检", "passed": passed, "checks": checks}


def main():
    res = run()
    if res["passed"]:
        print("PASS：副本与原件一致（哈希 + 数值）")
        return 0
    print("FAIL：副本已过期或数值不一致——请按 Robot-Control/AGENTS.md 重新同步副本")
    return 1


if __name__ == "__main__":
    sys.exit(main())
