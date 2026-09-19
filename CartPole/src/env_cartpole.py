import sys
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any
import gym
from gym import spaces
import pygame


CONFIG = {
    # 质量参数
    'M': 1.0,        # 小车质量 (kg)
    'm1': 0.1,       # 摆杆质量 (kg)
    'l1': 0.5,       # 半摆长 (m)
    # 物理参数
    'g': 9.8,        # 重力加速度 (m/s^2)
    'mu': 0.1,       # 摩擦系数
    # 控制参数
    'F_max': 10.0,   # 最大控制力 (N)
    # 仿真参数
    'dt': 0.02,      # 时间步长 (s)
    'max_steps': 1000, # 最大步数
    # 状态约束
    'x_threshold': 5.0,         # 小车位置阈值 (m)
    'theta_threshold': 80 * np.pi / 180,  # 角度阈值 (rad)
    # 初始状态
    'init_x_range': [-0.5, 0.5],
    'init_dot_x_range': [-0.3, 0.3],
    'init_theta_range': [-10 * np.pi/180, 10 * np.pi/180],
    'init_dot_theta_range': [-1 * np.pi/180, 1 * np.pi/180],
    # 存活型奖励系数（无终端项，与书 eq:cartpole-1-reward 一致）
    # 非终止步: r = 1 - cθ·(θ/θ_th)² - cθ̇·(θ̇/θ̇_ref)² - cx·(x/x_th)² - cu·(F/F_max)²
    # 终止步:   r = 0   （不启用终端位置惩罚 c_term；早期版本曾用 c_term=30，见书 eq:cartpole-1-reward-term）
    'reward_coefs': {
        'c_theta': 0.2,
        'c_theta_dot': 0.05,
        'c_x': 1.0,
        'c_u': 0.01,
    },
}

class CartPoleCustomEnv(gym.Env):
    """
    自定义倒立摆环境，基于提供的物理公式
    状态: [x, x_dot, theta, theta_dot]
    动作: 施加的力 F ∈ [-F_max, F_max]
    """
    
    def __init__(self, config: dict = None):
        super(CartPoleCustomEnv, self).__init__()
        self.default_config = CONFIG
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        # 动作空间: 连续力 F
        self.action_space = spaces.Box(
            low=-self.config['F_max'],
            high=self.config['F_max'],
            shape=(1,),
            dtype=np.float32
        )
        # 状态空间: [x, x_dot, theta, theta_dot]
        self.observation_space = spaces.Box(
            low=np.array([
                -self.config['x_threshold'] * 2,
                -np.inf,
                -self.config['theta_threshold'] * 2,
                -np.inf
            ]),
            high=np.array([
                self.config['x_threshold'] * 2,
                np.inf,
                self.config['theta_threshold'] * 2,
                np.inf
            ]),
            dtype=np.float32
        )
        self.state = None
        self.steps = 0
        
    def _compute_accelerations(self, state: np.ndarray, F: float) -> Tuple[float, float]:
        """
        根据动力学公式，解二元线性方程组，得到[x_ddot, theta_ddot]
        (M+m1) * x_ddot + m1*l1*cos(theta) * theta_ddot = F + m1*l1*sin(theta)*theta_dot^2 - mu*x_dot
        cos(theta) * x_ddot + (4/3)*l1 * theta_ddot = g*sin(theta)
        """
        x, x_dot, theta, theta_dot = state
        M = self.config['M']
        m1 = self.config['m1']
        l1 = self.config['l1']
        g = self.config['g']
        mu = self.config['mu']
        A = np.array([
            [M + m1, m1 * l1 * math.cos(theta)],
            [math.cos(theta), 4/3 * l1]
        ])
        B = np.array([
            F + m1 * l1 * math.sin(theta) * theta_dot**2 - mu * x_dot,
            g * math.sin(theta)
        ])
        try:
            accelerations = np.linalg.solve(A, B)
            x_ddot, theta_ddot = accelerations
        except np.linalg.LinAlgError:
            x_ddot = 0.0
            theta_ddot = 0.0
        return x_ddot, theta_ddot
    
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境到初始状态
        """
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        x0 = np.random.uniform(*self.config['init_x_range'])
        theta0 = np.random.uniform(*self.config['init_theta_range'])
        x_dot0 = np.random.uniform(*self.config['init_dot_x_range'])
        theta_dot0 = np.random.uniform(*self.config['init_dot_theta_range'])
        self.state = np.array([x0, x_dot0, theta0, theta_dot0], dtype=np.float32)
        self.steps = 0
        info = {"steps": self.steps}
        return self.state.copy(), info

   
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行一步动作
        """
        # 确保动作在合法范围内（处理数组到标量的转换）
        if isinstance(action, np.ndarray):
            action = float(action[0]) if action.shape == (1,) else float(action)
        F = np.clip(action, -self.config['F_max'], self.config['F_max'])
        F = float(F)  # 转换为标量
        
        x, x_dot, theta, theta_dot = self.state
        x_ddot, theta_ddot = self._compute_accelerations(self.state, F)
        dt = self.config['dt']
        x_dot_new = x_dot + x_ddot * dt
        theta_dot_new = theta_dot + theta_ddot * dt
        x_new = x + (x_dot + x_dot_new) * 0.5 * dt
        theta_new = theta + (theta_dot + theta_dot_new) * 0.5 * dt
        theta_new = ((theta_new + math.pi) % (2 * math.pi)) - math.pi
        self.state = np.array([
            x_new, x_dot_new, theta_new, theta_dot_new
        ], dtype=np.float32)
        self.steps += 1
        terminated = self._is_terminated()
        reward = self._compute_reward(terminated, F)
        info = {
            'F_applied': F,
            'x_ddot': x_ddot,
            'theta_ddot': theta_ddot,
            'steps': self.steps
        }
        return self.state.copy(), reward, terminated, False, info
    

    def _is_terminated(self) -> bool:
        """
        检查是否终止
        """
        x, _, theta, _ = self.state
        x_out = abs(x) > self.config['x_threshold']
        theta_out = abs(theta) > self.config['theta_threshold']
        steps_out = self.steps >= self.config['max_steps']
        return x_out or theta_out or steps_out



    def _compute_reward(self, terminated: bool, F: float) -> float:
        """存活型奖励（γ=1 无折扣回报，对应书 eq:cartpole-1-reward）。

        非终止步：
            r = 1 - c_θ·(θ/θ_th)² - c_θ̇·(θ̇/θ̇_ref)²
                  - c_x·(x/x_th)² - c_u·(F/F_max)²
        终止步（默认）：r = 0（不加额外惩罚，避免高方差悬崖）。
        若 reward_coefs 给出可选 `c_term`（>0），终止步改为：
            r = -c_term·(x_T/x_th)²     （终端位置惩罚，激励把车停在中心 x→0）

        归一化尺度来源：θ_th=80°、x_th=5 m、F_max=10 N（均取自 CONFIG），
        θ̇_ref 自定 = θ_th（rad/s，即角速度每秒扫过一个阈值角的量级）。
        系数 c_θ/c_θ̇/c_x/c_u 由 Session 6 定稿（c_x=1.0）；`c_term` 为终端回中项，
        默认不启用（缺省 0），早期版本曾用 c_term=30（与书 eq:cartpole-1-reward-term 一致）。
        """
        coefs = self.config.get("reward_coefs", {})
        theta_th = self.config['theta_threshold']   # 80° (rad)
        theta_dot_ref = theta_th                     # rad/s，见 docstring
        x_th = self.config['x_threshold']            # 5.0 m
        F_max = self.config['F_max']                 # 10 N
        x, x_dot, theta, theta_dot = self.state
        c_term = coefs.get("c_term", 0.0)
        if terminated:
            return 0.0 if c_term <= 0 else -c_term * (x / x_th) ** 2
        C_THETA = coefs.get("c_theta", 0.2)
        C_THETA_DOT = coefs.get("c_theta_dot", 0.05)
        C_X = coefs.get("c_x", 0.1)
        C_U = coefs.get("c_u", 0.01)
        reward = 1.0
        reward -= C_THETA * (theta / theta_th) ** 2
        reward -= C_THETA_DOT * (theta_dot / theta_dot_ref) ** 2
        reward -= C_X * (x / x_th) ** 2
        reward -= C_U * (F / F_max) ** 2
        return reward


    def render(self, mode='human'):
        """
        使用Pygame进行可视化
        """
        x, x_dot, theta, theta_dot = self.state
        l1 = self.config['l1']
        # 初始化Pygame
        if not hasattr(self, 'pygame_initialized'):
            pygame.init()
            self.screen_width = 800
            self.screen_height = 400
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('倒立摆仿真系统')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.pygame_initialized = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.screen.fill((255, 255, 255)) # 清屏
        scale = 100  # 像素/米
        offset_x = self.screen_width // 2
        offset_y = self.screen_height // 2 + 100
        track_y = offset_y
        pygame.draw.line(
            self.screen, (0, 0, 0),
            (0, track_y), (self.screen_width, track_y), 3
        )
        cart_width = int(0.3 * scale)
        cart_height = int(0.2 * scale)
        cart_x = int(x * scale + offset_x)
        cart_rect = pygame.Rect(
            cart_x - cart_width//2,
            track_y - cart_height,
            cart_width, cart_height
        )
        pygame.draw.rect(self.screen, (135, 206, 235), cart_rect)  # 天蓝色
        pygame.draw.rect(self.screen, (0, 0, 128), cart_rect, 2)   # 海军蓝边框
        pole_length = l1 * 2 * scale
        pole_end_x = cart_x + pole_length * math.sin(theta)
        pole_end_y = track_y - cart_height//2 - pole_length * math.cos(theta)
        pygame.draw.line(
            self.screen, (255, 0, 0),
            (cart_x, track_y - cart_height//2),
            (pole_end_x, pole_end_y), 5
        )
        pygame.draw.circle(
            self.screen, (0, 0, 0),
            (int(pole_end_x), int(pole_end_y)), 8
        )
        info_lines = [
            f"step: {self.steps}",
            f"position x: {x:.3f} m",
            f"velocity x_dot: {x_dot:.3f} m/s",
            f"angle theta: {theta*180/math.pi:.2f}°",
            f"angular velocity: {theta_dot*180/math.pi:.2f}°/s"
        ]
        if hasattr(self, '_last_info'):
            F = self._last_info.get('F_applied', 0)
            info_lines.append(f"adding force: {F:.2f} N")
        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(text, (10, 10 + i * 25))
        pygame.display.flip()
        self.clock.tick(60)  # 限制帧率
        if mode == 'rgb_array':
            return pygame.surfarray.array3d(self.screen).swapaxes(0, 1)


    def close(self):
        """
        关闭Pygame
        """
        if hasattr(self, 'pygame_initialized'):
            pygame.quit()



def register_custom_cartpole():
    """
    注册自定义倒立摆环境
    """
    try:
        gym.register(
            id='CartPoleCustom-v1',
            entry_point=__name__ + ':CartPoleCustomEnv',
            max_episode_steps=CONFIG['max_steps']
        )
    except gym.error.NameAlreadyInUse:
        # 如果已经注册过，跳过
        pass

