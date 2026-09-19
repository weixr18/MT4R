"""Scratch test for RANSAC code (to be embedded in chap7_3_pairprb_5.tex)."""
import numpy as np
import scipy.linalg

rng = np.random.default_rng(42)


# ---------------- 通用: 迭代次数估计 ----------------
def ransac_iters(s, w, eta=0.01):
    """保证以 1-eta 概率至少采到一组全内点的迭代次数"""
    if w >= 1.0:
        return 0
    return int(np.ceil(np.log(eta) / np.log(1 - w ** s)))


# ---------------- 1. 直线拟合 RANSAC ----------------
def ransac_line(pts, eps=0.5, eta=0.01):
    """RANSAC 直线拟合 y=kx+b, pts: (N,2)"""
    N = pts.shape[0]
    kb_best, n_best, N_iter = np.zeros(2), 0, 1000
    while N_iter > 0:
        idx = rng.choice(N, 2, replace=False)
        (x1, y1), (x2, y2) = pts[idx[0]], pts[idx[1]]
        if abs(x2 - x1) < 1e-12:
            N_iter -= 1
            continue
        k = (y2 - y1) / (x2 - x1)
        kb = np.array([k, y1 - k * x1])
        n_in = np.sum(np.abs(pts[:, 1] - (kb[0] * pts[:, 0] + kb[1])) < eps)
        if n_in > n_best:
            kb_best, n_best = kb, n_in
            N_iter = min(N_iter, ransac_iters(2, n_in / N, eta))
        N_iter -= 1
    mask = np.abs(pts[:, 1] - (kb_best[0] * pts[:, 0] + kb_best[1])) < eps
    if mask.sum() >= 2:
        A = np.column_stack([pts[mask, 0], np.ones(mask.sum())])
        kb_best = np.linalg.lstsq(A, pts[mask, 1], rcond=None)[0]
    return kb_best


# ---------------- 2. 4点法H阵 RANSAC ----------------
def solve_Hmat_4p(nu_1s, nu_2s, K):
    """由 N>=4 对点(最小二乘)求解归一化坐标下的单应矩阵, nu_1s,nu_2s: (N,3)"""
    invK = np.linalg.inv(K)
    N = nu_1s.shape[0]
    u1 = (invK @ nu_1s.T).T
    u2 = (invK @ nu_2s.T).T
    u1 = u1 / u1[:, 2:3]
    u2 = u2 / u2[:, 2:3]
    A, b = np.zeros([2 * N, 8]), np.zeros(2 * N)
    for i in range(N):
        A[2 * i, :] = [u2[i, 0], u2[i, 1], 1, 0, 0, 0,
                       -u1[i, 0] * u2[i, 0], -u1[i, 0] * u2[i, 1]]
        A[2 * i + 1, :] = [0, 0, 0, u2[i, 0], u2[i, 1], 1,
                           -u1[i, 1] * u2[i, 0], -u1[i, 1] * u2[i, 1]]
        b[2 * i], b[2 * i + 1] = u1[i, 0], u1[i, 1]
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])


def res_Hmat(nu_1s, nu_2s, H, K):
    """单应转移误差(像素空间), nu_1s,nu_2s: (N,3)"""
    H_pix = K @ H @ np.linalg.inv(K)
    pred = (H_pix @ nu_2s.T).T
    pred = pred / pred[:, 2:3]
    return np.linalg.norm(nu_1s[:, :2] - pred[:, :2], axis=1)


def vo_Hmat_ransac(nu_1s, nu_2s, K, eps=2.0, eta=0.01):
    N = nu_1s.shape[0]
    H_best, n_best, N_iter = None, 0, 1000
    while N_iter > 0:
        idx = rng.choice(N, 4, replace=False)
        H = solve_Hmat_4p(nu_1s[idx], nu_2s[idx], K)
        n_in = np.sum(res_Hmat(nu_1s, nu_2s, H, K) < eps)
        if n_in > n_best:
            H_best, n_best = H, n_in
            N_iter = min(N_iter, ransac_iters(4, n_in / N, eta))
        N_iter -= 1
    mask = res_Hmat(nu_1s, nu_2s, H_best, K) < eps
    H_best = solve_Hmat_4p(nu_1s[mask], nu_2s[mask], K)
    return H_best, mask


# ---------------- 3. 8点法F阵 RANSAC ----------------
def solve_Fmat_8p(nu_1s, nu_2s):
    """8点法求解基础矩阵, nu_1s,nu_2s: (N,3), N>=8"""
    A = np.zeros([nu_1s.shape[0], 9])
    for i in range(nu_1s.shape[0]):
        u2, v2 = nu_2s[i, 0], nu_2s[i, 1]
        A[i, :] = np.concatenate([u2 * nu_1s[i], v2 * nu_1s[i], nu_1s[i]])
    f = np.linalg.svd(A)[2][-1]
    F = np.array([[f[0], f[3], f[6]], [f[1], f[4], f[7]], [f[2], f[5], f[8]]])
    U, s, VT = np.linalg.svd(F)
    return U @ np.diag([s[0], s[1], 0.0]) @ VT


def res_Fmat(nu_1s, nu_2s, F):
    """极线距离: 点 nu_1 到对应极线的距离, nu_1s,nu_2s: (N,3)"""
    l1 = (F @ nu_2s.T).T          # 位姿1下的极线
    d = np.abs(np.sum(nu_1s * l1, axis=1))
    return d / np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2)


def vo_Fmat_ransac(nu_1s, nu_2s, eps=2.0, eta=0.01):
    N = nu_1s.shape[0]
    F_best, n_best, N_iter = None, 0, 1000
    while N_iter > 0:
        idx = rng.choice(N, 8, replace=False)
        F = solve_Fmat_8p(nu_1s[idx], nu_2s[idx])
        n_in = np.sum(res_Fmat(nu_1s, nu_2s, F) < eps)
        if n_in > n_best:
            F_best, n_best = F, n_in
            N_iter = min(N_iter, ransac_iters(8, n_in / N, eta))
        N_iter -= 1
    mask = res_Fmat(nu_1s, nu_2s, F_best) < eps
    F_best = solve_Fmat_8p(nu_1s[mask], nu_2s[mask])
    return F_best, mask


# ---------------- 4. 6点法DLT RANSAC ----------------
def solve_DLT_6p(p1s, nu_2s, K):
    """6点法DLT求解位姿 T_2^1, p1s:(N,3), nu_2s:(N,3), N>=6"""
    M = np.zeros([2 * p1s.shape[0], 12])
    p1_ext = np.column_stack([p1s, np.ones(p1s.shape[0])])
    for i in range(p1s.shape[0]):
        A = np.array([[1, 0, -nu_2s[i, 0]], [0, 1, -nu_2s[i, 1]]]) @ K
        M[2 * i:2 * i + 2, :4] = A[:, 0:1] @ p1_ext[i:i + 1, :]
        M[2 * i:2 * i + 2, 4:8] = A[:, 1:2] @ p1_ext[i:i + 1, :]
        M[2 * i:2 * i + 2, 8:] = A[:, 2:3] @ p1_ext[i:i + 1, :]
    t = np.linalg.svd(M)[2][-1]
    Rt = np.column_stack([t[:4], t[4:8], t[8:]]).T
    R12, t21 = Rt[:, :3], Rt[:, 3]
    if np.linalg.det(R12) < 0:
        R12, t21 = -R12, -t21
    a = np.power(np.abs(np.linalg.det(R12)), -1.0 / 3)
    R12, t21 = R12 * a, t21 * a
    T21 = np.vstack([np.column_stack([R12.T, -R12.T @ t21]), [0, 0, 0, 1]])
    return T21


def res_DLT(p1s, nu_2s, T21, K):
    """重投影误差, p1s:(N,3), nu_2s:(N,3)"""
    R12, t21 = T21[:3, :3].T, -T21[:3, :3].T @ T21[:3, 3]
    pred = (K @ (R12 @ p1s.T + t21[:, None])).T
    pred = pred / pred[:, 2:3]
    return np.linalg.norm(nu_2s[:, :2] - pred[:, :2], axis=1)


def vo_DLT_ransac(p1s, nu_2s, K, eps=2.0, eta=0.01):
    N = p1s.shape[0]
    T_best, n_best, N_iter = None, 0, 1000
    while N_iter > 0:
        idx = rng.choice(N, 6, replace=False)
        T = solve_DLT_6p(p1s[idx], nu_2s[idx], K)
        n_in = np.sum(res_DLT(p1s, nu_2s, T, K) < eps)
        if n_in > n_best:
            T_best, n_best = T, n_in
            N_iter = min(N_iter, ransac_iters(6, n_in / N, eta))
        N_iter -= 1
    mask = res_DLT(p1s, nu_2s, T_best, K) < eps
    T_best = solve_DLT_6p(p1s[mask], nu_2s[mask], K)
    return T_best, mask


# ================= 测试 =================
def rand_pose():
    """随机小角度旋转 + 适度平移, 保证数值稳定"""
    axis = rng.standard_normal(3)
    axis = axis / np.linalg.norm(axis)
    theta = rng.uniform(0.1, 0.4)
    R = scipy.linalg.expm(np.array([[0, -axis[2], axis[1]],
                                    [axis[2], 0, -axis[0]],
                                    [-axis[1], axis[0], 0]]) * theta)
    t = rng.uniform(-0.6, 0.6, 3)
    return R, t


def test_line():
    k0, b0 = 2.0, 1.0
    x = rng.uniform(-5, 5, 100)
    y = k0 * x + b0 + rng.normal(0, 0.1, 100)
    pts = np.column_stack([x, y])
    out_x = rng.uniform(-5, 5, 40)
    out_y = rng.uniform(-20, 20, 40)
    pts = np.vstack([pts, np.column_stack([out_x, out_y])])
    kb = ransac_line(pts, eps=0.5)
    assert abs(kb[0] - k0) < 0.05 and abs(kb[1] - b0) < 0.1, f"line fail: {kb}"
    print("line OK:", kb)


def test_H():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float)
    R21, t12 = rand_pose()
    R12, t21 = R21.T, -R21.T @ t12
    # 共面点: 位姿1系下坐标 z 满足 z=5
    xy = rng.uniform(-2, 2, [200, 2])
    p1 = np.column_stack([xy, np.full(200, 5.0)])
    p2 = (R12 @ p1.T).T + t21
    nu_1 = (K @ p1.T).T; nu_1 = nu_1 / nu_1[:, 2:3]
    nu_2 = (K @ p2.T).T; nu_2 = nu_2 / nu_2[:, 2:3]
    # 加噪声 + 外点
    nu_1 += np.column_stack([rng.normal(0, 0.3, 200), rng.normal(0, 0.3, 200), np.zeros(200)])
    nu_2 += np.column_stack([rng.normal(0, 0.3, 200), rng.normal(0, 0.3, 200), np.zeros(200)])
    idx_out = rng.choice(200, 60, replace=False)
    for i in idx_out:
        nu_1[i] = rng.uniform(0, 640, 3); nu_1[i, 2] = 1
        nu_2[i] = rng.uniform(0, 640, 3); nu_2[i, 2] = 1
    H, mask = vo_Hmat_ransac(nu_1, nu_2, K, eps=3.0)
    assert mask.sum() > 120, f"H mask {mask.sum()}"
    assert mask[idx_out].sum() <= 3, f"H outliers kept: {mask[idx_out].sum()}"
    pred = (K @ H @ np.linalg.inv(K) @ nu_2[mask].T).T
    pred = pred / pred[:, 2:3]
    err = np.linalg.norm(nu_1[mask][:, :2] - pred[:, :2], axis=1).mean()
    assert err < 0.8, f"H reproj err {err}"
    print(f"H OK: inliers={mask.sum()}/200, mean err={err:.3f}")


def test_F():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float)
    R21, t12 = rand_pose()
    R12, t21 = R21.T, -R21.T @ t12
    p1 = rng.uniform(2, 10, [200, 3])
    p2 = (R12 @ p1.T).T + t21
    nu_1 = (K @ p1.T).T; nu_1 = nu_1 / nu_1[:, 2:3]
    nu_2 = (K @ p2.T).T; nu_2 = nu_2 / nu_2[:, 2:3]
    nu_1 += np.column_stack([rng.normal(0, 0.3, 200), rng.normal(0, 0.3, 200), np.zeros(200)])
    nu_2 += np.column_stack([rng.normal(0, 0.3, 200), rng.normal(0, 0.3, 200), np.zeros(200)])
    idx_out = rng.choice(200, 60, replace=False)
    for i in idx_out:
        nu_1[i] = rng.uniform(0, 640, 3); nu_1[i, 2] = 1
        nu_2[i] = rng.uniform(0, 640, 3); nu_2[i, 2] = 1
    F, mask = vo_Fmat_ransac(nu_1, nu_2, eps=3.0)
    assert mask.sum() > 120, f"F mask {mask.sum()}"
    assert mask[idx_out].sum() <= 3, f"F outliers kept: {mask[idx_out].sum()}"
    l1 = (F @ nu_2[mask].T).T
    d = np.abs(np.sum(nu_1[mask] * l1, axis=1)) / np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2)
    assert d.mean() < 1.5, f"F epipolar err {d.mean()}"
    print(f"F OK: inliers={mask.sum()}/200, epipolar mean err={d.mean():.3f}")


def test_DLT():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float)
    R21, t12 = rand_pose()
    R12, t21 = R21.T, -R21.T @ t12
    p1 = rng.uniform(2, 10, [200, 3])
    p2 = (R12 @ p1.T).T + t21
    nu_2 = (K @ p2.T).T; nu_2 = nu_2 / nu_2[:, 2:3]
    nu_2 += np.column_stack([rng.normal(0, 0.3, 200), rng.normal(0, 0.3, 200), np.zeros(200)])
    idx_out = rng.choice(200, 60, replace=False)
    for i in idx_out:
        nu_2[i] = rng.uniform(0, 640, 3); nu_2[i, 2] = 1
    T_hat, mask = vo_DLT_ransac(p1, nu_2, K, eps=3.0)
    assert mask.sum() > 120, f"DLT mask {mask.sum()}"
    assert mask[idx_out].sum() <= 3, f"DLT outliers kept: {mask[idx_out].sum()}"
    R_err = np.linalg.norm(T_hat[:3, :3] - R21, ord='fro')
    t_err = np.linalg.norm(T_hat[:3, 3] - t12)
    assert R_err < 0.1 and t_err < 0.5, f"DLT pose err {R_err} {t_err}"
    print(f"DLT OK: inliers={mask.sum()}/200, R err={R_err:.3f}, t err={t_err:.3f}")


if __name__ == "__main__":
    test_line()
    test_H()
    test_F()
    test_DLT()
    print("ALL TESTS PASSED")
