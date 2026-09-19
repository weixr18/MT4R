import numpy as np
import matplotlib.pyplot as plt
from code_kfs import filter_ESKF
np.random.seed(42)

############################# Test system #############################

def f_func(x, u):
    return (x + u) % (2 * np.pi)
def df_func(x, u):
    return np.array([[1.0]])
def h_func(x):
    return x
def dh_func(x):
    return np.array([[1.0]])
def add_func(x, delta_x):
    return (x + delta_x) % (2 * np.pi)

def generate_data(N_K):
    dt = 0.1
    x_true_0 = np.pi / 4  # 初始角度 45°
    x_est_0 = x_true_0 + 0.1  # 初始估计有误差
    P_0 = np.array([[5]])
    omega = 0.2  # rad/s
    us = [omega * dt for _ in range(N_K)]
    x_trues = [x_true_0]
    for u in us:
        x_trues.append(f_func(x_trues[-1], u))
    R_std = 0.05
    zs = [x + np.random.randn() * R_std for x in x_trues[1:]]   # 0 起始、无冗余前导
    Qs = [np.array([[1e-4]]) for _ in range(N_K)]
    Rs = [5 * np.array([[R_std**2]]) for _ in range(N_K)]   # 长度 N_K，与 zs 一一对应
    return x_est_0, P_0, us, Qs, Rs, x_trues, zs


def visualize(x_trues, xs_est, zs):
    x_trues = np.array(x_trues[1:])
    x_ests = np.array([x[0] for x in xs_est[1:]])
    plt.figure(figsize=(10, 5))
    plt.plot(x_trues, label='True Angle')
    plt.plot([z for z in zs], label='Measurements', linestyle='dotted')
    plt.plot(x_ests, label='ESKF Estimate')
    plt.title("ESKF on SO(2) Angle Tracking")
    plt.xlabel("Time Step")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid(True)
    plt.show()

############################# Test #############################


if __name__ == "__main__":
    N_K = 500
    x_0, P_0, us, Qs, Rs, x_trues, zs = generate_data(N_K)
    xs_est = filter_ESKF(
        x_0=np.array([x_0]),
        us=[np.array([u]) for u in us],
        zs=[np.array([z]) for z in zs],
        f_func=lambda x, u: np.array([f_func(x[0], u[0])]),
        df_func=lambda x, u: df_func(x[0], u[0]),
        h_func=lambda x: np.array([h_func(x[0])]),
        dh_func=lambda x: dh_func(x[0]),
        add_func=lambda x, dx: np.array([add_func(x[0], dx[0])]),
        Qs=Qs, Rs=Rs, P_0=P_0, N_K=N_K
    )
    visualize(x_trues, xs_est, zs)


