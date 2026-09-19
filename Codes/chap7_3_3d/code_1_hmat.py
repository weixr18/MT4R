"""chap7_3_pairprb_2.tex: algo:vo_hmat_4p —— 4点法求解单应矩阵.

书中算法 (algot:vo_hmat_4p / algo:vo_hmat_4p):
    输入: 位姿1各点像素齐次坐标 nu_1s (N,3), 位姿2各点像素齐次坐标 nu_2s (N,3), 内参 K
    输出: 单应矩阵 H (归一化坐标空间, h_33 = 1)
    步骤:
        1. 像素坐标 -> 归一化坐标: u = K^{-1} nu
        2. 组装 A(2N x 8), b(2N): 由 u_1 = \\hat\\lambda H u_2 消去 \\hat\\lambda 得 2N 个线性方程
        3. 最小二乘解 h = lstsq(A, b)
        4. H = [[h1,h2,h3],[h4,h5,h6],[h7,h8,1]]

注意:
    - 求解得到的是归一化坐标空间的单应矩阵 (右下角置 1), 与真正单应矩阵相差
      一个比例因子 = 真值右下角元素。若要在像素空间使用, 需作共轭变换
      H_pix = K H K^{-1} (见 calc_vo_Hmat_pix)。
    - N = 4 时 A 为 8x8 方阵, 恰定满秩可唯一求解; N > 4 时最小二乘。
"""
import numpy as np


def calc_vo_Hmat_4p(N, nu_1s, nu_2s, K):
    """4点法求解归一化坐标空间的单应矩阵 (书中 algo:vo_hmat_4p)。

    输入:
        nu_1s: (N,3) 位姿1下各点像素齐次坐标
        nu_2s: (N,3) 位姿2下各点像素齐次坐标
        K    : (3,3) 相机内参
    输出:
        H: (3,3) 归一化坐标空间的单应矩阵, h_33 = 1
    """
    assert nu_1s.shape == (N, 3) and nu_2s.shape == (N, 3) and K.shape == (3, 3)
    invK = np.linalg.inv(K)
    A, b = np.zeros((2 * N, 8)), np.zeros(2 * N)
    for i in range(N):
        u1 = invK @ nu_1s[i]
        u1 /= u1[2]
        u2 = invK @ nu_2s[i]
        u2 /= u2[2]
        A[2 * i, :] = [u2[0], u2[1], 1, 0, 0, 0,
                       -u1[0] * u2[0], -u1[0] * u2[1]]
        A[2 * i + 1, :] = [0, 0, 0, u2[0], u2[1], 1,
                           -u1[1] * u2[0], -u1[1] * u2[1]]
        b[2 * i:2 * i + 2] = u1[:2]
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([[h[0], h[1], h[2]],
                     [h[3], h[4], h[5]],
                     [h[6], h[7], 1.0]])


def calc_vo_Hmat_pix(H, K):
    """归一化空间单应 -> 像素空间单应: H_pix = K H K^{-1}。"""
    return K @ H @ np.linalg.inv(K)


def calc_vo_Hmat_4p_dlt(nu_1s, nu_2s):
    """独立交叉验证: 经典 2N x 9 DLT (v1 x (H v2) = 0), 直接在像素空间求 H。

    与 calc_vo_Hmat_4p 完全不同的组装与求解方式 (SVD 零空间, 9 个未知数),
    用于对拍验证。N >= 4 时秩亏为 1, 解唯一 (到整体比例)。
    """
    N = nu_1s.shape[0]
    A = np.zeros((2 * N, 9))
    for i in range(N):
        x1, y1 = nu_1s[i, :2]
        x2, y2, w2 = nu_2s[i]
        # v1 x (H v2) = 0 的前两行 (第三行与前两行线性相关)
        A[2 * i, :] = [0, 0, 0, x2, y2, w2, -y1 * x2, -y1 * y2, -y1 * w2]
        A[2 * i + 1, :] = [x2, y2, w2, 0, 0, 0, -x1 * x2, -x1 * y2, -x1 * w2]
    h = np.linalg.svd(A)[2][-1]      # 最小奇异值对应右奇异向量
    H = h.reshape(3, 3)
    return H / H[2, 2]
