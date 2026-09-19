import time
import numpy as np
import gym
from env_cartpole import register_custom_cartpole, CONFIG

VERBOSE = False

# 随机基线口径说明（两类指标不可混用）：
# - 经典控制口径：用统一评估 J_ach（Q=diag(1,1,100,1)、R=[1.0]，见 eval/bench.py）。
#   随机策略无法存活，按评估约定 J_ach 记 +∞（仅作 PID/LQR 等经典控制器参照下界）。
# - RL 口径：用 return（γ=1 存活型奖励，见书 eq:cartpole-1-reward）。
#   随机策略 return ≈ 存活步数量级，作为 REINFORCE 训练回报的参照下界。
# 本脚本分别打印两者，以明确区分。

if __name__ == "__main__":
    register_custom_cartpole()  # 注册环境
    env = gym.make('CartPoleCustom-v1', disable_env_checker=True)  # 创建环境，禁用环境检查器
    print("一阶倒立摆-随机策略测试")
    print(f"动作空间: {env.action_space}")
    low, high = env.observation_space.low, env.observation_space.high
    print(f"状态空间: ")
    print(f"    x: ({low[0].item():.4f}, {high[0].item():.4f})")
    print(f"    x_dot: {(low[1].item(), high[1].item())}")
    print(f"    theta: ({low[2].item():.4f}, {high[2].item():.4f})")
    print(f"    theta_dot: {(low[3].item(), high[3].item())}")

    # 统一评估口径（与 eval/bench.py 一致）
    Q = np.diag([1.0, 1.0, 100.0, 1.0])   # x, x_dot, theta, theta_dot
    R = 1.0                                # u 为标量力

    # 随机策略测试
    state, info = env.reset()
    total_return = 0.0   # RL 口径：γ=1 存活型回报
    j_ach = 0.0          # 经典口径：统一评估成本
    step = 0
    while step < CONFIG['max_steps']:
        xk = state  # 第 step 步施加动作前的状态，计入 J_ach
        action = env.action_space.sample()  # 随机动作
        state, reward, terminated, truncated, info = env.step(action)  # 执行一步
        total_return += reward
        u = float(info['F_applied'])  # 实际施加（裁剪后）的力
        j_ach += float(xk @ Q @ xk) + R * u * u
        step += 1
        if VERBOSE:
            print(f"Step {step}: Reward={reward:.4f}, TotalReturn={total_return:.2f}")
            print(f"State: {state}")
            print(f"Action: {action}, Applied F: {info['F_applied']:.2f}")
        env.render()
        time.sleep(CONFIG['dt'] * 2)  # 添加sleep，控制帧率
        if terminated or truncated:
            print(f"Episode结束于第{step}步 ({step*CONFIG['dt']}秒)")
            break
    print(f"RL return（γ=1 存活型奖励）: {total_return:.2f}")
    print(f"J_ach（经典口径，随机策略未存活 => 按评估约定记 +∞，原始累计 {j_ach:.2f}）")
    env.close()
