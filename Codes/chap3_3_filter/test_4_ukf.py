import numpy as np
import matplotlib.pyplot as plt

from code_kfs import filter_UKF
np.random.seed(42)
dt = 0.01
def f_func(x, u):
    return np.array([
        x[0] + x[1] * dt,
        x[1] - x[0] * 0.1
    ])
def h_func(x):
    return np.array([
        x[0] - x[1],
        np.arctan(x[0])
    ])

def generate_data(N_K):
    x_0 = np.array([0.5, 1.0])  # 初始状态 [位置, 速度]
    P_0 = np.diag([5.0, 5.0])  # 初始协方差
    Q = np.diag([1e-1, 1e-1])  # 过程噪声协方差
    R = np.diag([1e-2, 1e-2])  # 观测噪声协方差
    us = [np.array([[0.0]]) for _ in range(N_K)]
    x_trues, zs = [x_0], []
    for k in range(N_K):
        x_next = f_func(x_trues[-1], us[-1])
        x_trues.append(x_next)
        z = h_func(x_next + np.random.multivariate_normal([0,0], Q))
        z += np.random.multivariate_normal([0, 0], R)
        zs.append(z)
    Qs = [Q*100 for _ in range(N_K)]
    Rs = [R*100 for _ in range(N_K)]
    x_0 = x_0 + np.array([-0.1, 0.5])
    return x_0, P_0, us, Qs, Rs, x_trues, zs


def visualize(x_trues, xs_est, zs):
    # 提取真实和估计的位置、速度
    true_pos = [x[0] for x in x_trues]
    true_vel = [x[1] for x in x_trues]
    est_pos = [x[0] for x in xs_est]
    est_vel = [x[1] for x in xs_est]
    plt.figure(figsize=(12, 6))
    # 位置图
    plt.subplot(1, 2, 1)
    plt.plot(true_pos, label='True Position')
    plt.plot(est_pos, '--', label='Estimated Position')
    plt.xlabel('Time step')
    plt.ylabel('Position')
    plt.title('Position Estimation')
    plt.legend()
    # 速度图
    plt.subplot(1, 2, 2)
    plt.plot(true_vel, label='True Velocity')
    plt.plot(est_vel, '--', label='Estimated Velocity')
    plt.xlabel('Time step')
    plt.ylabel('Velocity')
    plt.title('Velocity Estimation')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    N_K = 500
    # UKF参数
    alpha = 1.2 # similar to sigma of gaussian
    beta = 2.0 # best for gaussian, most commonly used
    kappa = 0.0 # suggested value and most commonly used
    x_0, P_0, us, Qs, Rs, x_trues, zs = generate_data(N_K)
    xs_est = filter_UKF(
        x_0, us, zs, f_func, h_func, Qs, Rs, P_0, alpha, kappa, beta, N_K
    )
    visualize(x_trues, xs_est, zs)