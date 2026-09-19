import numpy as np


def oc_LQR_disc(x_0, As, Bs, Rs, Qs, S, N):
    assert len(As) == N and len(Bs) == N
    assert len(Rs) == N and len(Qs) == N
    Fs, P_next = [], S
    for k in range(N-1, -1, -1):
        tmp = np.linalg.inv(Rs[k] + Bs[k].T @ P_next @ Bs[k])
        Fs.append(-tmp @ Bs[k].T @ P_next @ As[k])
        P_next = As[k].T @ P_next @ (As[k] + Bs[k] @ Fs[-1]) + Qs[k]
    Fs, x_k, u_opt = Fs[::-1], x_0, []
    for k in range(N):
        u_opt.append(Fs[k] @ x_k)
        x_k = As[k] @ x_k + Bs[k] @ u_opt[k]
    return u_opt


def oc_LQR_track_disc(x_0, xds, A, B, Rs, Qs, S, N, lambda_=0.5):
    # 参考轨迹 xds 长度 N+1：xds[0..N]，与数学下标一致
    n = x_0.shape[0]
    assert len(Rs) == N and len(Qs) == N
    assert len(xds) == N + 1
    Ae_s, Be_s, Qe_s = [], [], []
    for k in range(N):
        Ad_k = lambda_ * np.eye(n)
        Ad_k += ((xds[k+1] - lambda_ * xds[k])[:, None]
                 @ xds[k][None, :]) / (xds[k] @ xds[k])
        Ae_s.append(np.block([
            [A, np.zeros((n, n))],
            [np.zeros((n, n)), Ad_k],
        ]))
        Be_s.append(np.vstack([B, np.zeros_like(B)]))
        Qe_s.append(np.block([
            [Qs[k], -Qs[k]],
            [-Qs[k], Qs[k]],
        ]))
    Se = np.block([
        [S, -S],
        [-S, S],
    ])
    xe_0 = np.concatenate([x_0, xds[0]])
    return oc_LQR_disc(xe_0, Ae_s, Be_s, Rs, Qe_s, Se, N)


def oc_LQR_track_smooth_disc(x_0, u_prev, xds, A, B, Rs, Qs, S, N,
                             lambda_=0.5):
    # 参考轨迹 xds 长度 N+1：xds[0..N]；u_prev 为前一步输入 u_{-1}
    n, m = B.shape
    assert len(Rs) == N and len(Qs) == N
    assert len(xds) == N + 1
    Ae_s, Be_s, Qe_s = [], [], []
    for k in range(N):
        Ad_k = lambda_ * np.eye(n)
        Ad_k += ((xds[k+1] - lambda_ * xds[k])[:, None]
                 @ xds[k][None, :]) / (xds[k] @ xds[k])
        Ae_s.append(np.block([
            [A, np.zeros((n, n)), B],
            [np.zeros((n, n)), Ad_k, np.zeros((n, m))],
            [np.zeros((m, n)), np.zeros((m, n)), np.eye(m)],
        ]))
        Be_s.append(np.vstack([B, np.zeros((n, m)), np.eye(m)]))
        Qe_s.append(np.block([
            [Qs[k], -Qs[k], np.zeros((n, m))],
            [-Qs[k], Qs[k], np.zeros((n, m))],
            [np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, m))],
        ]))
    Se = np.block([
        [S, -S, np.zeros((n, m))],
        [-S, S, np.zeros((n, m))],
        [np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, m))],
    ])
    xe_0 = np.concatenate([x_0, xds[0], u_prev])
    dU_opt = oc_LQR_disc(xe_0, Ae_s, Be_s, Rs, Qe_s, Se, N)
    # 由输入增量重构实际输入序列 u_k = u_{k-1} + Δu_k
    u_opt, u_k = [], u_prev.copy()
    for k in range(N):
        u_k = u_k + dU_opt[k]
        u_opt.append(u_k.copy())
    return u_opt
