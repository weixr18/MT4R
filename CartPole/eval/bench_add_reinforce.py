# -*- coding: utf-8 -*-
"""把 REINFORCE 策略结果并入 `res/bench/bench_results.json`，并生成 `reinforce_L{1,2,3}.png`。

背景：`eval/bench.py` 原生支持 `REINFORCE_CKPT` 环境变量（见 `_make_controllers`），但若每次
都跑完整 `bench.py` 会**重新**计算 6 个经典控制器的确定性结果（几十分钟）。本脚本只补充
REINFORCE，复用 bench.py 的同一批 seed=42 初值、同一 `run_level`/`rollout`/`cost_quadratic`
代码路径，并把结果合并进既有 `bench_results.json`（经典控制数字保持不变）。

用法（cwd=CartPole，REINFORCE_CKPT 指向训练好的 .pth）：
    E:/Anaconda3/envs/py311-gym/python.exe eval/bench_add_reinforce.py
"""
import json
import os
import sys

import numpy as np

EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import bench  # noqa: E402  （复用 rollout / run_level / gen_init_states / _make_controllers 等）


def main():
    ckpt = os.environ.get("REINFORCE_CKPT", "").strip()
    if not ckpt or not os.path.exists(ckpt):
        raise SystemExit("请设置 REINFORCE_CKPT 指向训练好的 .pth")

    env = bench.CartPoleCustomEnv(bench.CONFIG)
    # _make_controllers() 会打印 PID/LQR/MPC/DDP 等构造矩阵（无碍），并加载 REINFORCE
    controllers = bench._make_controllers()
    if "reinforce" not in controllers:
        raise SystemExit("REINFORCE controller 未加载（REINFORCE_CKPT 未生效）")
    ctrl = controllers["reinforce"]

    # 同一批初值：与 bench.py run_all 完全一致的 seed=42 流
    np.random.seed(bench.SEED)
    init_by_level = {lv: bench.gen_init_states(lv, bench.K) for lv in bench.LEVELS}

    # 已存在的完整 bench_results.json（暂不含 reinforce）
    json_path = os.path.join(bench.RES_DIR, "bench_results.json")
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    results = payload["results"]
    if "reinforce" in results:
        print("  [hint] bench_results.json 已含 reinforce，将用本次结果覆盖")

    print(f"\n{'='*72}\n[算法] reinforce (K={bench.K})\n{'='*72}")
    for lv in bench.LEVELS:
        x0_list = init_by_level[lv][:bench.K]
        res = bench.run_level(ctrl, lv, x0_list, "reinforce", env, bench.FIG_DIR_CART)
        results["reinforce"] = {**results.get("reinforce", {}), lv: res}
        print(f"  {lv}: J_ach = {res['J_mean']} ± {res['J_std']}, "
              f"存活率 = {res['s']}/{res['K']}, 终止={res['reason_counts']}, "
              f"图={os.path.basename(res['fig'])}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n数值已保存: {os.path.abspath(json_path)}")


if __name__ == "__main__":
    main()
