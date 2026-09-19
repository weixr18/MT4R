import numpy as np
import matplotlib.pyplot as plt
from code_kfs import filter_EKF

np.random.seed(0)

############################# Test system #############################

def f_func(x, u):
    return np.array([
        x[0] + u[0] * np.cos(x[1]),
        x[1] + u[1]
    ])
def df_func(x, u):
    return np.array([
        [1, -u[0] * np.sin(x[1])],
        [0, 1]
    ])
def h_func(x):
    return np.array([
        x[0]**2,
        np.sin(x[1])
    ])

def dh_func(x):
    return np.array([
        [2 * x[0], 0],
        [0, np.cos(x[1])]
    ])


############################# Components #############################

def generate_data(N_K):
    x_true = np.array([1.0, 0.5])
    x_0 = np.array([0.0, 0.0])  # 初始估计
    P_0 = np.eye(2) * 0.1
    # 控制输入（固定）
    us = [np.array([1.0, 0.1]) for _ in range(N_K)]
    # 噪声协方差
    Q = np.diag([0.01, 0.01])
    R = np.diag([0.1, 0.1])
    Qs = [Q for _ in range(N_K)]
    Rs = [R for _ in range(N_K)]   # 长度 N_K，Rs[k] 与第 k 步量测一一对应（0 起始、无冗余前导）
    # 生成真实轨迹和观测值
    x_trues = [x_true]
    zs = []   # zs[k]：第 k 步施加 us[k] 一步预测后取得的量测
    for k in range(N_K):
        x_next = f_func(x_trues[-1], us[k]) + np.random.multivariate_normal([0, 0], Q)
        z_k = h_func(x_next) + np.random.multivariate_normal([0, 0], R)
        x_trues.append(x_next)
        zs.append(z_k)
    return x_0, P_0, us, Qs, Rs, x_trues, zs

def visualize(x_trues, xs_est, zs):
    xs_est = np.array(xs_est)
    x_trues = np.array(x_trues)
    plt.figure(figsize=(10, 5))
    plt.plot(x_trues[:, 0], x_trues[:, 1], label='True Trajectory', color='blue')
    plt.plot(xs_est[:, 0], xs_est[:, 1], label='EKF Estimated Trajectory', color='red', linestyle='--')
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.legend()
    plt.title('EKF with Nonlinear System')
    plt.grid(True)
    plt.show()


############################# Test #############################


if __name__ == "__main__":
    N_K = 500
    x_0, P_0, us, Qs, Rs, x_trues, zs = generate_data(N_K)
    xs_est = filter_EKF(
        x_0, us, zs, f_func, df_func, 
        h_func, dh_func, Qs, Rs, P_0, N_K
    )
    visualize(x_trues, xs_est, zs)

