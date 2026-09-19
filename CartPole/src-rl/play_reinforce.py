# -*- coding: utf-8 -*-
"""可视化已训练 REINFORCE 策略：多组初值，按指定倍速在倒立摆上实时播放。

加载 `src-rl/res/s3_formal_1000/seed1/best_policy.pth`（纯 REINFORCE、连续动作高斯策略），
用任一组的确定性策略（均值动作 mu，与 `eval/bench.py` 同口径；--stochastic 可采样）播放。

用法（cwd 任意，脚本自解析 ../src 路径；需有图形界面）：
    E:/Anaconda3/envs/py311-gym/python.exe src-rl/play_reinforce.py --speed 2

说明：
  - 默认播放 5 组不同的初值（内置 demo 套件），每组播完打印该组小结后自动进入下一组；
  - --speed S：播放倍速，S=1 表示实时（1000 步 = 20s），S=2 表示 2 倍速（1000 步 = 10s）；
  - 倍速按「墙钟推进物理步数」实现，不受显示器 60Hz 上限影响；
  - Esc 或关闭窗口随时退出。

覆盖初值（theta/thetadot 用度）：--state "x,xdot,theta,thetadot" 可重复，给一组则播放给定组。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src"))
for _p in (SRC_DIR, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pygame
import torch

from env_cartpole import CONFIG, CartPoleCustomEnv
from rl_common import load_policy_ckpt


# 内置 demo 套件：5 组不同的初值（均落于环境初始范围，以体现不同起始姿态下的表现）
DEFAULT_STATES = [
    ("1", 0.30, 0.00, 2.0, 0.0),
    ("2", -0.40, 0.20, 5.0, -0.5),
    ("3", 0.00, -0.30, -8.0, 1.0),
    ("4", -0.20, 0.30, 10.0, -1.0),
    ("5", 0.40, -0.20, 3.0, 0.8),
]


def parse_args():
    ap = argparse.ArgumentParser(description="播放已训练 REINFORCE 策略（多组初值 + 倍速）")
    ap.add_argument(
        "--ckpt",
        default=os.path.join(SCRIPT_DIR, "res", "s3_formal_1000", "seed1", "best_policy.pth"),
        help="策略权重（默认 = src-rl/res/s3_formal_1000/seed1/best_policy.pth）",
    )
    ap.add_argument("--speed", type=float, default=1.0,
                    help="播放倍速（1=实时 20s，2=2倍速 10s，…）")
    ap.add_argument(
        "--state", action="append", metavar="X,XDOT,THETA,TDOT",
        help="初值 'x,xdot,theta(deg),thetadot(deg/s)'，可重复；不给则用内置 5 组 demo")
    ap.add_argument("--max-steps", type=int, default=1000,
                    help="单回合最大步数（默认 1000 = 20s 实时）")
    ap.add_argument("--pause", type=float, default=2.0,
                    help="每组之间暂停秒数（默认 2.0；Esc 可跳过）")
    ap.add_argument("--stochastic", action="store_true",
                    help="策略采样（带 sigma 噪声）；默认用均值动作 mu（确定性）")
    ap.add_argument("--device", default="cpu")
    return ap.parse_args()


def _parse_state(s: str):
    """解析 'x,xdot,theta(deg),thetadot(deg/s)' 为一维 float 初值数组（theta 转弧度）。"""
    vals = [float(v.strip()) for v in s.split(",")]
    if len(vals) != 4:
        raise ValueError(f"--state 需给 4 个数 x,xdot,theta(deg),thetadot(deg/s)，收到: {s!r}")
    x, xd, th, thd = vals
    return np.array([x, xd, math.radians(th), math.radians(thd)], dtype=np.float32)


def _action(policy, obs_rms, state, stochastic, device):
    s = obs_rms.normalize(state)
    s_t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if stochastic:
            return policy.sample(s_t).cpu().numpy()[0]
        mu, _ = policy(s_t)
        return mu.cpu().numpy()[0]


def _wait_key_or_timeout(seconds: float):
    """暂停 seconds 秒，期间 Esc / 关窗可提前退出；返回是否退出。"""
    deadline = time.time() + seconds
    while time.time() < deadline:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        time.sleep(0.02)
    return False


def run_episode(env, policy, obs_rms, x0, args, label, device):
    """播放一组初值；返回 (步数, 是否越界终止, 终态 x, 终态 theta_deg)。"""
    env.state = x0.copy()
    env.steps = 0
    env.render()

    steps_per_sec = args.speed / CONFIG["dt"]   # 目标：每秒推进的物理步数
    t0 = time.time()
    steps_done = 0
    terminated_by_bound = False

    while steps_done < args.max_steps and not terminated_by_bound:
        desired = int((time.time() - t0) * steps_per_sec)
        desired = min(desired, args.max_steps)
        if desired <= steps_done:
            # 推进过快帧：保持当前帧并处理事件（env.render 自带 60fps 节拍，通常不会走到这）
            if env.render() is not None:
                pass
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return steps_done, terminated_by_bound, float(env.state[0]), math.degrees(env.state[2]), True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return steps_done, terminated_by_bound, float(env.state[0]), math.degrees(env.state[2]), True
            continue

        while steps_done < desired:
            if steps_done >= args.max_steps:
                break
            a = _action(policy, obs_rms, env.state, args.stochastic, device)
            _obs, _r, _term, _trunc, _info = env.step(a)
            steps_done = env.steps
            if steps_done < args.max_steps and (
                    abs(env.state[0]) > env.config["x_threshold"]
                    or abs(env.state[2]) > env.config["theta_threshold"]):
                terminated_by_bound = True
                break

        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return steps_done, terminated_by_bound, float(env.state[0]), math.degrees(env.state[2]), True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return steps_done, terminated_by_bound, float(env.state[0]), math.degrees(env.state[2]), True

    return steps_done, terminated_by_bound, float(env.state[0]), math.degrees(env.state[2]), False


def main():
    args = parse_args()

    env = CartPoleCustomEnv(CONFIG)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    policy, obs_rms, meta = load_policy_ckpt(args.ckpt, obs_dim, act_dim, device=args.device)
    policy.eval()
    device = next(policy.parameters()).device

    # 解析初值列表
    if args.state:
        states = [(f"#{i+1}", *_parse_state(s)) for i, s in enumerate(args.state)]
    else:
        states = DEFAULT_STATES
        # 转成 x0 数组并保留标签
        states = [(lab, np.array([x, xd, math.radians(th), math.radians(thd)], dtype=np.float32))
                  for lab, x, xd, th, thd in DEFAULT_STATES]

    print("=" * 70)
    print(f"REINFORCE 策略回放（{'采样' if args.stochastic else '确定性均值 mu'}，倍速 x{args.speed:g}）")
    print(f"  ckpt : {args.ckpt}")
    print(f"  meta : seed={meta.get('seed')} env={meta.get('env')} iters={meta.get('n_iters')} "
          f"eps/iter={meta.get('episodes_per_iter')}")
    dur_desc = "实时" if args.speed == 1 else f"{args.max_steps * CONFIG['dt'] / args.speed:.0f}s"
    print(f"  共 {len(states)} 组，每组 max_steps={args.max_steps}（{dur_desc}）；Esc / 关窗可退出")
    print("=" * 70)

    total_runs = len(states)
    for i, (label, x0) in enumerate(states):
        xs, xd, th, thd = x0
        print(f"\n>>> 第 {i+1}/{total_runs} 组（{label}）初值: x={xs:+.2f} m, xdot={xd:+.2f} m/s, "
              f"theta={math.degrees(th):+.2f} deg, thetadot={math.degrees(thd):+.2f} deg/s")
        steps, term, fx, ftheta, quit_early = run_episode(env, policy, obs_rms, x0, args, label, device)
        if quit_early:
            print("已退出。")
            return
        if steps >= args.max_steps:
            print(f"  ▸ 跑满 {args.max_steps} 步（{steps * CONFIG['dt'] / args.speed:.1f}s）："
                  f"终态 x={fx:+.3f} m, theta={ftheta:+.2f} deg —— 稳杆、不控位置")
        else:
            why = "theta 越界(>80°)" if abs(env.state[2]) > env.config["theta_threshold"] else "x 越界(>5m)"
            print(f"  ▸ 提前终止于 step={steps}（{why}）：x={fx:+.3f} m, theta={ftheta:+.2f} deg")
        if i < total_runs - 1:
            if _wait_key_or_timeout(args.pause):   # 自动进入下一组；Esc/关窗退出
                print("已退出。")
                return
            print("  （下组即将开始 ...）")


if __name__ == "__main__":
    main()
