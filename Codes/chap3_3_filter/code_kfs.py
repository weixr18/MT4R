# -*- coding: utf-8 -*-
"""《Math Toolbox for Robotics》Part III 观测器与滤波器 KF/EKF/ESKF/UKF 实现
（chap3_2_obsfilter_2.tex / chap3_2_obsfilter_3.tex）。

核心实现，与书 `algo:` 算法框及正文代码块一致。
数组索引：数学下标从 0 开始，代码直接 0 起始、不加冗余前导（见 2index.tex「符号」一节）。
量测按步号编号：第 k 步（k=0..N_K-1）先施加 us[k] 完成一步预测得先验，
再用该步的量测 zs[k]（协方差 Rs[k]）修正，故观测方程写作 z_k = C x_{k+1} + v_k。
"""

import numpy as np
import scipy

def filter_KF_disc(x_0, us, zs, A, B, C, Qs, Rs, P_0, N_K:int):
    assert len(us) == N_K, len(zs) == N_K
    assert len(Rs) == N_K, len(Qs) == N_K
    Ks, xs, Pk = [], [x_0], P_0
    for k in range(N_K):
        xk_ = A @ xs[k] + B @ us[k]
        Pk_ = A @ Pk @ A.T + Qs[k]
        tmp = np.linalg.inv(Rs[k] + C @ Pk_ @ C.T)
        Ks.append(Pk_ @ C.T @ tmp)
        xs.append(xk_ + Ks[k] @ (zs[k] - C @ xk_))
        tmp2 = np.eye(x_0.shape[0]) - Ks[k] @ C
        Pk = Ks[k] @ Rs[k] @ Ks[k].T + tmp2 @ Pk_ @ tmp2.T
    return xs


def filter_EKF(x_0, us, zs, f_func, df_func, h_func, dh_func, Qs, Rs, P_0, N_K:int):
    assert len(us) == N_K, len(zs) == N_K
    assert len(Rs) == N_K, len(Qs) == N_K
    Ks, xs, Pk = [], [x_0], P_0
    for k in range(N_K):
        xk_ = f_func(xs[k], us[k])
        Fk = df_func(xs[k], us[k])
        Pk_ = Fk @ Pk @ Fk.T + Qs[k]
        Hk = dh_func(xk_)
        tmp = np.linalg.inv(Rs[k] + Hk @ Pk_ @ Hk.T)
        Ks.append(Pk_ @ Hk.T @ tmp)
        xs.append(xk_ + Ks[k] @ (zs[k] - h_func(xk_)))
        tmp2 = np.eye(x_0.shape[0]) - Ks[k] @ Hk
        Pk = Ks[k] @ Rs[k] @ Ks[k].T + tmp2 @ Pk_ @ tmp2.T
    return xs


def filter_ESKF(x_0, us, zs, f_func, df_func, h_func, dh_func, add_func, Qs, Rs, P_0, N_K:int):
    assert len(us) == N_K, len(zs) == N_K
    assert len(Rs) == N_K, len(Qs) == N_K
    Ks, xs, Pk = [], [x_0], P_0
    for k in range(N_K):
        xk_ = f_func(xs[k], us[k])
        Fk = df_func(xs[k], us[k])
        Pk_ = Fk @ Pk @ Fk.T + Qs[k]
        Hk = dh_func(xk_)
        tmp = np.linalg.inv(Rs[k] + Hk @ Pk_ @ Hk.T)
        Ks.append(Pk_ @ Hk.T @ tmp)
        delta_x = Ks[k] @ (zs[k] - h_func(xk_))
        xs.append(add_func(xk_, delta_x))
        tmp2 = np.eye(delta_x.shape[0]) - Ks[k] @ Hk
        Pk = Ks[k] @ Rs[k] @ Ks[k].T + tmp2 @ Pk_ @ tmp2.T
    return xs


def filter_UKF(x_0, us, zs, f_func, h_func, Qs, Rs, P_0, alpha, kappa, beta, N_K:int):
    assert len(us) == N_K, len(zs) == N_K
    assert len(Rs) == N_K, len(Qs) == N_K
    n, m = x_0.shape[0], zs[0].shape[0]
    lambda_ = alpha**2 *(n + kappa) - n
    wms, wcs = np.zeros([2*n+1]), np.zeros([2*n+1])
    wms[0] = lambda_ / (n + lambda_)
    wcs[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    wms[1:], wcs[1:] = 0.5 / (n + lambda_), 0.5 / (n + lambda_)
    Ks, xs, Pk = [], [x_0], P_0
    for k in range(N_K):
        points, fs, hs = np.zeros([2*n+1, n]), np.zeros([2*n+1, n]), np.zeros([2*n+1, m])
        ps = scipy.linalg.sqrtm((n + lambda_) * Pk)
        points[0] = xs[k]
        points[1:1+n], points[1+n:] = ps + xs[k], - ps + xs[k]
        xk_, zk_est = np.zeros_like(x_0), np.zeros_like(zs[0])
        for i in range(2*n+1):
            fs[i] = f_func(points[i], us[k])
            hs[i] = h_func(points[i])
            xk_ += wms[i] * fs[i]
            zk_est += wms[i] * hs[i]
        Pk_, Hk = Qs[k].copy(), np.zeros([m, n])   # copy：Pk_ 会就地累加，不能改写调用方的 Qs
        for i in range(2*n+1):
            Pk_ += wcs[i] * (fs[i] - xk_)[:, None] @ (fs[i] - xk_)[None, :]
            Hk += wcs[i] * (hs[i] - zk_est)[:, None] @ (fs[i] - xk_)[None, :]
        tmp = np.linalg.inv(Rs[k] + Hk @ Pk_ @ Hk.T)
        Ks.append(Pk_ @ Hk.T @ tmp)
        xs.append(xk_ + Ks[k] @ (zs[k] - zk_est))
        tmp2 = np.eye(x_0.shape[0]) - Ks[k] @ Hk
        Pk = Ks[k] @ Rs[k] @ Ks[k].T + tmp2 @ Pk_ @ tmp2.T
    return xs

