import time
import gym
import numpy as np
from env_cartpole import register_custom_cartpole, CONFIG
from scipy.linalg import solve_continuous_are

np.random.seed(42)

VERBOSE = False
RENDER = True
RENDER_STEP = 50
MAX_EPISODES = 5

N = 4 # dimention of system
M = 1 # dimention of input
Q_param = np.diag([1, 1, 100, 1]) # x, x_dot, theta, theta_dot
R_param = np.diag([1.0])

def is_controllable(A, B):
    from numpy.linalg import matrix_rank
    n = A.shape[0]
    controllability_matrix = np.hstack([
        np.linalg.matrix_power(A, i) @ B for i in range(n)
    ])
    rank = matrix_rank(controllability_matrix)
    return rank == n

def calc_sys_mat():
    M, m1, l1, mu, g = CONFIG['M'], CONFIG['m1'], CONFIG['l1'], CONFIG['mu'], CONFIG['g']
    k1 = 4 / (4*M + m1)
    k2 = - 3*m1*g / (4*M + m1)
    k3 = 3*(M+m1)*g / (l1 * (4*M + m1))
    k4 = - 3 / (l1 * (4*M + m1))
    k6 = 3*mu / (l1 * (4*M + m1))
    A = np.array([
        [0., 1., 0., 0.],
        [0., -mu*k1, k2, 0.],
        [0., 0., 0., 1.],
        [0., k6, k3, 0.],
    ])
    B = np.array([[0, k1, 0, k4]]).T
    assert is_controllable(A, B)
    return A, B


class LQRContinuousController:
    """LQR控制器类"""
    def __init__(self, Q, R, f_max):
        # 连续LQR控制器
        assert isinstance(Q, np.ndarray) and Q.shape == (N, N)
        assert isinstance(R, np.ndarray) and R.shape == (M, M)
        A, B = calc_sys_mat()
        P = solve_continuous_are(A, B, Q, R)
        F = -np.linalg.inv(R) @ B.T @ P
        print("F:", F)
        self.F = F
        self.f_max = f_max

    def compute_action(self, state):
        """根据状态计算控制力"""
        assert isinstance(state, np.ndarray) and state.shape == (N,)
        control = self.F @ state
        control = np.clip(control, -self.f_max, self.f_max)
        return np.array([control], dtype=np.float32)

    def reset(self):
        pass



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


def test_lqr_policy():
    """测试LQR策略"""
    register_custom_cartpole()
    env = gym.make('CartPoleCustom-v1', disable_env_checker=True)
    print("一阶倒立摆-LQR策略测试")
    print(f"动作空间: {env.action_space}")
    low, high = env.observation_space.low, env.observation_space.high
    print(f"状态空间: ")
    print(f"    x: ({low[0].item():.4f}, {high[0].item():.4f})")
    print(f"    x_dot: {(low[1].item(), high[1].item())}")
    print(f"    theta: ({low[2].item():.4f}, {high[2].item():.4f})")
    print(f"    theta_dot: {(low[3].item(), high[3].item())}")
    # 统一评估口径（与 eval/bench.py 一致）：经典控制用 J_ach，不依赖环境奖励
    R_scalar = float(np.squeeze(R_param))
    episode_jachs = []
    episode_lengths = []
    pid_controller = LQRContinuousController(Q_param, R_param, CONFIG["F_max"])
    for episode in range(MAX_EPISODES):
        # 重置环境和控制器
        state, info = env.reset()
        pid_controller.reset()
        j_ach = 0.0
        terminated = truncated = False
        step_count = 0
        print(f"\n=== Episode {episode + 1} ===")
        print(f"  初始状态：x = {state[0]:.3f} m, dot x = {state[1]:.3f} m/s,", end=' ')
        print(f"  theta = {state[2]*180/np.pi:.3f} °, v = {state[3]*180/np.pi:.3f} °/s")
        
        actions, states = [], []
        while not (terminated or truncated):
            xk = state  # 第 k 步施加动作前的状态，计入 J_ach
            action = pid_controller.compute_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            u = float(info['F_applied'])  # 实际施加（裁剪后）的力
            j_ach += float(xk @ Q_param @ xk) + R_scalar * u * u
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
    test_lqr_policy()
    
