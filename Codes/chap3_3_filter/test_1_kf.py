import numpy as np
import matplotlib.pyplot as plt

from code_kfs import filter_KF_disc

def generate_test_data(N, A, B, C, x_0, true_acc, n_x, n_z):
    x = x_0
    xs_true = [x.copy()]
    zs = []   # 0 起始、不加冗余前导：zs[k] 即第 k 步施加 us[k] 一步预测后取得的量测
    us = []
    for _ in range(N):
        u = np.array([[true_acc]]) + np.random.randn(1, 1) * 0.1  # add small noise to control
        process_noise = np.random.randn(2, 1) * n_x
        x = A @ x + B @ u + process_noise
        z = C @ x + np.random.randn(1, 1) * n_z
        xs_true.append(x.copy())
        zs.append(z)
        us.append(u)
    return np.array(us), np.array(zs), np.array(xs_true)


def plot_res(xs_true, xs_est, zs):
    # Plot results
    xs_est = np.array(xs_est).squeeze()
    xs_true = xs_true.squeeze()
    zs = zs.squeeze()
    plt.figure(figsize=(12, 6))
    plt.plot(xs_true[:, 0], label='True Position')
    plt.plot(xs_est[:, 0], label='Estimated Position')
    plt.plot(zs, label='Measurements', linestyle='dotted')
    plt.xlabel('Time step')
    plt.ylabel('Position')
    plt.title('Kalman Filter Test (2D State: Position & Velocity)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    N_K = 50
    dt = 1.0
    true_acc = 0.2
    n_x = 0.1
    n_z = 1.0
    A = np.array([[1, dt],[0, 1]])
    B = np.array([[0.5 * dt ** 2],[dt]])
    C = np.array([[1, 0]])
    x_0 = np.array([[0], [0]])
    P_0 = np.eye(2) * 1.0
    Qs = [np.eye(2) * n_x**2 for _ in range(N_K)]
    Rs = [np.eye(1) * n_z**2 for _ in range(N_K)]   # 长度 N_K，Rs[k] 与第 k 步量测一一对应

    us, zs, xs_true = generate_test_data(
        N_K, A, B, C, x_0, true_acc, n_x, n_z
    )
    xs_est = filter_KF_disc(x_0, us, zs, A, B, C, Qs, Rs, P_0, N_K)
    plot_res(xs_true, xs_est, zs)

