import time
import gym
import numpy as np
from env_cartpole import register_custom_cartpole, CONFIG

np.random.seed(42)

VERBOSE = False
RENDER = True
RENDER_STEP = 50
MAX_EPISODES = 5
MAX_INTEGRAL = 10

class PIDController:
    """PID控制器类"""
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-CONFIG['F_max'], CONFIG['F_max'])):
        self.Kp = Kp  # 比例增益
        self.Ki = Ki  # 积分增益
        self.Kd = Kd  # 微分增益
        self.setpoint = setpoint  # 目标值
        self.prev_error = 0
        self.integral = 0
        self.output_limits = output_limits
        self.integral_min = -MAX_INTEGRAL
        self.integral_max = MAX_INTEGRAL
    
    def compute(self, measurement, dt):
        """计算PID输出"""
        error = self.setpoint - measurement
        P = self.Kp * error
        self.integral += error * dt
        self.integral = np.clip(self.integral, self.integral_min, self.integral_max)
        I = self.Ki * self.integral
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        D = self.Kd * derivative
        output = P + I + D
        self.prev_error = error
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        return output
    
    def reset(self):
        """重置PID状态"""
        self.prev_error = 0
        self.integral = 0


ANGLE_WEIGHT = 0.9
POSITION_WEIGHT = 1.0 - ANGLE_WEIGHT

class PIDBalanceController:
    """倒立摆平衡PID控制器"""
    def __init__(self, config):
        # 角度PID控制器（主要控制器）
        self.angle_pid = PIDController(
            Kp=30.0,  
            Kd=1.0,   
            Ki=0.0,   
            setpoint=0,  # 目标角度：垂直向上
            output_limits=(-config['F_max'], config['F_max'] )
        )
        # 位置PID控制器（次级控制器，防止小车跑出边界）
        self.position_pid = PIDController(
            Kp=1.0,
            Kd=4.0,      
            Ki=0.0,   
            setpoint=0, 
            output_limits=(-config['F_max'], config['F_max'])
        )
        self.config = config
        self.dt = config['dt']
        self.angle_weight = ANGLE_WEIGHT
        self.position_weight = POSITION_WEIGHT
        

    def compute_action(self, state):
        """根据状态计算控制力"""
        x, x_dot, theta, theta_dot = state
        angle_control = self.angle_pid.compute(theta, self.dt)
        position_control = self.position_pid.compute(x, self.dt)
        control = - (angle_control * self.angle_weight + position_control * self.position_weight)
        control = np.clip(control, -self.config['F_max'], self.config['F_max'])
        return np.array([control], dtype=np.float32)
    

    def reset(self):
        """重置所有PID控制器"""
        self.angle_pid.reset()
        self.position_pid.reset()



def visualize(actions, states, episode):
    import matplotlib.pyplot as plt
    actions = np.array(actions).flatten()
    states = np.array(states)
    t = np.arange(len(states)) * CONFIG["dt"]   # 秒（dt = 0.02 s）
    theta_deg = states[:, 2] * 180.0 / np.pi    # °
    theta_dot_deg = states[:, 3] * 180.0 / np.pi  # °/s
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Episode {episode + 1} Visualization", fontsize=16)
    axs[0].plot(t, states[:, 0], label="x (Position)")
    axs[0].set_ylabel("x (Position)")
    axs[0].legend()
    axs[1].plot(t, states[:, 1], label="x_dot (Velocity)", color="orange")
    axs[1].set_ylabel("x_dot (Velocity)")
    axs[1].legend()
    axs[2].plot(t, theta_deg, label="theta (°)", color="green")
    axs[2].set_ylabel("theta (°)")
    axs[2].legend()
    axs[3].plot(t, theta_dot_deg, label="theta_dot (°/s)", color="red")
    axs[3].set_ylabel("theta_dot (°/s)")
    axs[3].legend()
    axs[4].plot(t, actions, label="Action (Force)", color="purple")
    axs[4].set_ylabel("Action (Force)")
    axs[4].set_xlabel("Time [s]")
    axs[4].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def test_pid_policy():
    """测试PID策略"""
    register_custom_cartpole()
    env = gym.make('CartPoleCustom-v1', disable_env_checker=True)
    print("一阶倒立摆-PID策略测试")
    print(f"动作空间: {env.action_space}")
    low, high = env.observation_space.low, env.observation_space.high
    print(f"状态空间: ")
    print(f"    x: ({low[0].item():.4f}, {high[0].item():.4f})")
    print(f"    x_dot: {(low[1].item(), high[1].item())}")
    print(f"    theta: ({low[2].item():.4f}, {high[2].item():.4f})")
    print(f"    theta_dot: {(low[3].item(), high[3].item())}")
    
    # 统一评估口径（与 eval/bench.py 一致）：经典控制用 J_ach，不依赖环境奖励
    Q = np.diag([1.0, 1.0, 100.0, 1.0])   # x, x_dot, theta, theta_dot
    R = 1.0                                # u 为标量力
    episode_jachs = []
    episode_lengths = []
    pid_controller = PIDBalanceController(CONFIG)
    for episode in range(MAX_EPISODES):
        # 重置环境和控制器
        state, info = env.reset()
        pid_controller.reset()
        j_ach = 0.0
        terminated = truncated = False
        step_count = 0
        print(f"\n=== Episode {episode + 1} ===")
        
        actions, states = [], []
        while not (terminated or truncated):
            xk = state  # 第 k 步施加动作前的状态，计入 J_ach
            action = pid_controller.compute_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            u = float(info['F_applied'])  # 实际施加（裁剪后）的力
            j_ach += float(xk @ Q @ xk) + R * u * u
            state = next_state
            step_count += 1
            actions.append(action)
            states.append(next_state)
            if RENDER and step_count % RENDER_STEP == 0:
                env.render()
                time.sleep(CONFIG['dt']) 
            if VERBOSE and step_count % 50 == 0:
                x, x_dot, theta, theta_dot = state
                print(f"Step {step_count}:")
                print(f"  State: x={x:.3f}, x_dot={x_dot:.3f}, theta={theta:.3f}, theta_dot={theta_dot:.3f}")
                print(f"  Action: {action[0]:.3f}, J_ach(累计): {j_ach:.2f}")
                print(f"  PID状态: error={pid_controller.angle_pid.prev_error:.3f}, "
                      f"integral={pid_controller.angle_pid.integral:.3f}")
            pass
        visualize(actions, states, episode)
        episode_jachs.append(j_ach)
        episode_lengths.append(step_count)
        
        # 显示本回合结果
        print(f"Episode {episode + 1} 结束:")
        print(f"  总步数: {step_count} ({step_count * CONFIG['dt']:.1f}s)")
        print(f"  J_ach: {j_ach:.2f}")
        
        # 检查终止原因
        x, _, theta, _ = state
        if abs(x) > CONFIG['x_threshold']:
            print(f"  终止原因: x超出范围 ({x:.3f} > {CONFIG['x_threshold']})")
        elif abs(theta) > CONFIG['theta_threshold']:
            print(f"  终止原因: theta超出范围 ({theta:.3f} > {CONFIG['theta_threshold']})")
        else:
            print(f"  终止原因: 达到最大步数")
    # 统计信息
    print(f"\n=== 统计信息 ===")
    print(f"平均 J_ach: {np.mean(episode_jachs):.2f} ± {np.std(episode_jachs):.2f}")
    print(f"平均步数: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"最佳回合 J_ach: {np.min(episode_jachs):.2f}")
    print(f"最长回合: {np.max(episode_lengths)} 步")
    env.close()



if __name__ == "__main__":
    test_pid_policy()
    