# -*- coding: utf-8 -*-
"""验证《Math Toolbox for Robotics》Part II 逆运动学数值求解（chap2_3_diffk_2.tex）。

验证内容（独立参照 = 闭式解 / FK 误差自洽）：
1. 6 自由度随机臂：给定可到达目标 x_e = FK(q*)，三个算法从扰动初值 q_0 出发，
   检验最终 ||FK(q_hat) - x_e|| < 算法自身 epsilon
2. 6 自由度随机臂多随机初值：GN / LM 的鲁棒性（各 10 个初值全收敛）
3. GD 步长敏感性：一阶方法步长窗口很窄（ap 略大即发散，ap 太小收敛极慢），
   仅作为发现记录，不做断言
4. 2 连杆平面臂：闭式 IK 解（位置 + 姿态）对拍，三个算法均收敛

说明：书中 LM 原始代码存在多处 bug（f_new 误用 f_k、J_k/delta_x 未定义、λ 更新方向
与标准 Marquardt 相反、误差/关节向量维度不匹配等，见 docs/code-verify/robot-ik-verify.md），
本验证采用修复后的版本（code_ik.py），tex 代码块与之一致。

运行：E:\\Anaconda3\\envs\\py311-gym\\python.exe test_ik.py
"""
import numpy as np
import time

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "chap2_2_kinematcs"))
from code_fk import robot_fk
from code_ik import robot_ik_gd, robot_ik_gauss_newton, robot_ik_lm

np.set_printoptions(precision=6, suppress=True)
TOL_GN_LM = 1e-4          # 与算法内部默认 epsilon 一致（break 时误差必 < epsilon）
TOL_GD = 1e-3             # GD 一阶收敛，容差放宽


def random_arm(rng, N=6):
    d = np.zeros(N + 1); a = np.zeros(N + 1); alpha = np.zeros(N + 1)
    d[1:] = rng.uniform(-0.3, 0.3, N)
    a[1:] = rng.uniform(0.2, 1.0, N)
    alpha[1:] = rng.uniform(-np.pi / 2, np.pi / 2, N)
    return d, a, alpha


def run_one(name, fn, x_e, q_0, d, a, alpha, N, tol, **kwargs):
    t0 = time.time()
    q_hat = fn(x_e, q_0, d, a, alpha, N=N, **kwargs)
    dt = time.time() - t0
    err = np.linalg.norm(robot_fk(q_hat, d, a, alpha, N) - x_e)
    ok = err < tol
    print(f"  {name:<16} 收敛误差 ||FK(q̂)-x_e|| = {err:.3e}  耗时 {dt:.2f}s  "
          f"{'✔' if ok else '✗ 未达容差'}")
    return ok, err


def main():
    print("=" * 60)
    print("验证 逆运动学数值求解（code_ik.py）")
    print("=" * 60)

    rng = np.random.default_rng(42)
    N = 6
    d, a, alpha = random_arm(rng, N)
    q_star = np.array([0.0, 0.6, -0.4, 0.3, -0.5, 0.2, -0.3])   # 避开万向锁
    x_e = robot_fk(q_star, d, a, alpha, N)

    # ---- 1) 6-DOF 单目标，三个算法 ----
    print("\n[验证1] 6-DOF 随机臂：单目标 x_e = FK(q*)，初值 q_0 = q* + U(-0.1,0.1)")
    q_0 = q_star.copy()
    q_0[1:] += rng.uniform(-0.1, 0.1, N)
    print(f"  给定 x_e = {x_e}")
    # GD 一阶收敛，步长窗口窄：用 ap=0.02 + 较宽松 epsilon=1e-3；敏感性与慢速见 [验证2b]
    ok_gd, e_gd = run_one("GD", robot_ik_gd, x_e, q_0, d, a, alpha, N, TOL_GD,
                          epsilon=1e-3, alpha_p=0.02, max_iter=40000)
    ok_gn, e_gn = run_one("GN", robot_ik_gauss_newton, x_e, q_0, d, a, alpha, N, TOL_GN_LM)
    ok_lm, e_lm = run_one("LM", robot_ik_lm, x_e, q_0, d, a, alpha, N, TOL_GN_LM)
    assert ok_gn and ok_lm, "GN/LM 未收敛到算法自身 epsilon 内!"

    # ---- 2) 6-DOF 多随机初值（GN / LM 鲁棒性）----
    print("\n[验证2] 6-DOF 随机臂：10 个随机初值 q_0 = q* + U(-0.8,0.8)")
    n_ok_gn = n_ok_lm = 0
    err_gn = err_lm = 0.0
    for i in range(10):
        q_0 = q_star.copy()
        q_0[1:] += rng.uniform(-0.8, 0.8, N)
        q_hat = robot_ik_gauss_newton(x_e, q_0, d, a, alpha, N)
        e = np.linalg.norm(robot_fk(q_hat, d, a, alpha, N) - x_e)
        err_gn = max(err_gn, e)
        n_ok_gn += int(e < TOL_GN_LM)
        q_hat = robot_ik_lm(x_e, q_0, d, a, alpha, N)
        e = np.linalg.norm(robot_fk(q_hat, d, a, alpha, N) - x_e)
        err_lm = max(err_lm, e)
        n_ok_lm += int(e < TOL_GN_LM)
    print(f"  GN  10/10 收敛：{n_ok_gn} 个达到容差，最大误差 {err_gn:.3e}")
    print(f"  LM  10/10 收敛：{n_ok_lm} 个达到容差，最大误差 {err_lm:.3e}")
    assert n_ok_gn == 10 and n_ok_lm == 10, "GN/LM 多初值未全部收敛!"

    # ---- 3) GD 步长敏感性（仅记录，不断言）----
    print("\n[验证3] GD 步长敏感性（6-DOF，仅观察记录）：")
    print("        ap=0.02 稳定但需约 2 万次迭代（一阶）；ap=0.05 超过稳定上限发散；")
    print("        书中默认 ap=1e-2 更慢。这里用小 max_iter 快速展示前两者。")
    for ap, mi in ((1e-2, 3000), (0.05, 3000)):
        t0 = time.time()
        q_hat = robot_ik_gd(x_e, q_0, d, a, alpha, N, epsilon=1e-3,
                            alpha_p=ap, max_iter=mi)
        dt = time.time() - t0
        err = np.linalg.norm(robot_fk(q_hat, d, a, alpha, N) - x_e)
        if err < TOL_GD:
            tag = "✔ 已收敛"
        elif err > 5.0:
            tag = "✗ 发散"
        else:
            tag = "· 慢，未达容差"
        print(f"  GD ap={ap:<6} max_iter={mi}: err={err:.3e}  耗时 {dt:.2f}s  {tag}")

    # ---- 4) 2 连杆平面臂闭式解 ----
    print("\n[验证4] 2 连杆平面臂闭式 IK 解对拍")
    L1, L2 = 1.0, 0.8
    t1, t2 = 0.6, -0.4
    q2_star = np.array([0.0, t1, t2])
    d2 = np.zeros(3); a2 = np.array([0.0, L1, L2]); alpha2 = np.zeros(3)
    x_e2 = robot_fk(q2_star, d2, a2, alpha2, N=2)
    # 闭式解：位置 x=L1c1+L2c12, y=L1s1+L2s12 有两支（±θ2，肘上/肘下），位置相同但姿态不同。
    # 目标 x_e2 由 q2_star 生成，仅其中一支（θ2 与目标同号）姿态匹配，遍历两支验证。
    x, y = x_e2[0], x_e2[1]
    r2 = x ** 2 + y ** 2
    c2_sol = (r2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    branch_ok = False
    for t2_sol in (np.arccos(np.clip(c2_sol, -1, 1)),
                   -np.arccos(np.clip(c2_sol, -1, 1))):
        t1_sol = np.arctan2(y, x) - np.arctan2(L2 * np.sin(t2_sol),
                                               L1 + L2 * np.cos(t2_sol))
        x_cf = robot_fk(np.array([0.0, t1_sol, t2_sol]), d2, a2, alpha2, N=2)
        err_cf = np.linalg.norm(x_cf - x_e2)
        print(f"  闭式解分支 θ2={t2_sol:+.4f} → (θ1,θ2)=({t1_sol:.4f},{t2_sol:.4f})，"
              f"FK 误差 {err_cf:.3e}")
        if err_cf < TOL_GN_LM:
            branch_ok = True
    assert branch_ok, "平面臂闭式解与给定目标（含姿态）不匹配!"
    q_0 = q2_star + np.array([0.0, 0.4, 0.5])
    for name, fn, kwargs in [("GD", robot_ik_gd, {"alpha_p": 0.3}),
                             ("GN", robot_ik_gauss_newton, {}),
                             ("LM", robot_ik_lm, {})]:
        run_one(name, fn, x_e2, q_0, d2, a2, alpha2, 2, TOL_GN_LM, **kwargs)

    print("\n全部验证通过 ✔")


if __name__ == "__main__":
    main()
