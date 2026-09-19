"""验证 chap7_3_pairprb_2.tex 的 algo:vo_hmat_4p —— 4点法求解单应矩阵.

验证项:
    1. 最小情形 N=4 (无噪声): 归一化空间重投影误差 ~0; 与地面真值 A_gt/A_gt[2,2] 一致
    2. 超定 N=30 (无噪声): 同上, 验证最小二乘
    3. 像素空间: H_pix = K H K^{-1} 重投影误差 ~0; 与真值 H_pix_gt/H_pix_gt[2,2] 一致
       (验证书中注释: 解出的 H 与真正单应矩阵相差比例因子 = 右下角元素)
    4. 独立对拍: DLT (9 未知数, SVD 零空间) 与 K H K^{-1} 一致
    5. 噪声鲁棒性: 像素噪声 sigma 增大 -> 重投影误差增大; 点数增多 -> 误差减小
    6. N=3 欠定演示: 6x8 矩阵秩亏, 不能唯一求解 (说明"4点"是最低要求)

运行: E:\\Anaconda3\\envs\\py311-gym\\python.exe test_1_4_hmat_4p.py
"""
import numpy as np

from code_1_hmat import calc_vo_Hmat_4p, calc_vo_Hmat_pix, calc_vo_Hmat_4p_dlt

# ---------------- 数据生成 ----------------

def rot_y(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def gen_planar_data(N, K, R_c2, t_c2, n=(0, 0, 1), d=-4.0, seed=0, noise=0.0):
    """世界平面 n^T x + d = 0 (默认 z=4 平面) 上的 N 个共面点, 两个相机视图。

    相机 c1: 位姿 (I, 0)  相机 c2: 位姿 (R_c2, t_c2)  (x^{c2} = R_c2 (x - t_c2))
    返回:
        nu_1s, nu_2s : (N,3) 像素齐次坐标
        A_gt         : (3,3) 归一化空间单应矩阵真值 R - t n2^T/d2
        H_pix_gt     : (3,3) 像素空间单应矩阵真值 K A_gt K^{-1}
    """
    rng = np.random.default_rng(seed)
    n, d = np.array(n, float), float(d)
    # 平面上取点: n^T x + d = 0
    pts_w = np.column_stack([rng.uniform(-1.5, 1.5, N),
                             rng.uniform(-1.5, 1.5, N),
                             np.full(N, -d / n[2])])
    x_c1 = pts_w                                # c1 位姿 (I, 0)
    x_c2 = (R_c2 @ (pts_w - t_c2).T).T          # c2 系坐标
    keep = (x_c1[:, 2] > 0.1) & (x_c2[:, 2] > 0.1)
    x_c1, x_c2 = x_c1[keep], x_c2[keep]
    nu_1s = K @ x_c1.T
    nu_1s = (nu_1s / nu_1s[2]).T
    nu_2s = K @ x_c2.T
    nu_2s = (nu_2s / nu_2s[2]).T
    if noise > 0:
        nu_1s[:, :2] += rng.normal(0, noise, nu_1s[:, :2].shape)
        nu_2s[:, :2] += rng.normal(0, noise, nu_2s[:, :2].shape)
    # 地面真值: R = R_c1 R_c2^T = R_c2^T, t = R_c1(t_c2 - t_c1) = t_c2
    R, t = R_c2.T, t_c2
    n2, d2 = R_c2 @ n, n @ t_c2 + d
    A_gt = R - np.outer(t, n2) / d2
    H_pix_gt = K @ A_gt @ np.linalg.inv(K)
    return nu_1s, nu_2s, A_gt, H_pix_gt


def reproj_err(v1s, v2s, H):
    """按 v1 = \\lambda H v2 重投影 v2, 返回与 v1 的像素坐标欧氏误差 (N,)。"""
    pred = (H @ v2s.T).T
    pred = pred / pred[:, 2:3]
    return np.linalg.norm(v1s[:, :2] - pred[:, :2], axis=1)


def rel_fro(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(B)


def K_default():
    return np.array([[600, 0, 320], [0, 500, 240], [0, 0, 1]], float)


# ---------------- 验证项 ----------------

def test_minimal_and_overdetermined(K, R_c2, t_c2):
    """1&2. 无噪声: N=4 (恰定) 与 N=30 (超定最小二乘) 均应与真值一致。"""
    print("== 1&2. 无噪声: 恰定 N=4 与超定 N=30 ==")
    for N in (4, 30):
        nu_1s, nu_2s, A_gt, H_pix_gt = gen_planar_data(N, K, R_c2, t_c2, seed=0)
        H_est = calc_vo_Hmat_4p(N, nu_1s, nu_2s, K)

        # 归一化空间重投影
        invK = np.linalg.inv(K)
        u1 = (invK @ nu_1s.T).T; u1 = u1 / u1[:, 2:3]
        u2 = (invK @ nu_2s.T).T; u2 = u2 / u2[:, 2:3]
        err_norm = reproj_err(u1, u2, H_est).max()

        # 与真值对比 (归一化空间, 真值也归一化到 h_33=1)
        H_gt_norm = A_gt / A_gt[2, 2]
        rel_err = rel_fro(H_est, H_gt_norm)

        # 像素空间: K H K^{-1}
        # 注: 共轭变换 K(.)K^{-1} 会破坏 h_33=1 的归一化 (像素空间真值右下角
        #     != 归一化空间真值右下角), 因此像素空间对比必须按自身 [2,2] 归一化
        #     (尺度不变), 或直接验证重投影一致性。
        H_pix = calc_vo_Hmat_pix(H_est, K)
        err_pix = reproj_err(nu_1s, nu_2s, H_pix).max()
        rel_pix = rel_fro(H_pix / H_pix[2, 2], H_pix_gt / H_pix_gt[2, 2])

        # 地面真值自洽性 (验证测试架本身)
        gt_self = reproj_err(nu_1s, nu_2s, H_pix_gt).max()

        print(f"N={N:2d}: 归一化重投影 err={err_norm:.2e}, "
              f"H vs A_gt/A_gt[2,2] rel={rel_err:.2e}")
        print(f"        像素重投影 err={err_pix:.2e} px, "
              f"H_pix vs 真值 rel={rel_pix:.2e}, 真值自洽={gt_self:.2e} px")
        assert err_norm < 1e-9 and err_pix < 1e-7, f"N={N} 重投影失败"
        assert rel_err < 1e-9 and rel_pix < 1e-7, f"N={N} 真值对比失败"
        assert gt_self < 1e-7, f"N={N} 测试架真值不自洽"


def test_scale_note(K, R_c2, t_c2):
    """3. 书中注释: 解出的 H 与真正单应矩阵相差比例因子 = 真值右下角元素。"""
    print("== 3. 比例因子注释验证 ==")
    nu_1s, nu_2s, A_gt, H_pix_gt = gen_planar_data(30, K, R_c2, t_c2, seed=0)
    H_est = calc_vo_Hmat_4p(30, nu_1s, nu_2s, K)
    print(f"H_est[2,2] = {H_est[2, 2]:.1f} (固定置 1)")
    print(f"A_gt[2,2]  = {A_gt[2, 2]:.6f}  (归一化空间真值右下角)")
    print(f"H_est = A_gt / A_gt[2,2] ?  rel={rel_fro(H_est, A_gt / A_gt[2, 2]):.2e}")
    assert rel_fro(H_est, A_gt / A_gt[2, 2]) < 1e-9


def test_dlt_crosscheck(K, R_c2, t_c2):
    """4. 独立对拍: DLT (9未知数 SVD 零空间) 与 4点法 结果一致。"""
    print("== 4. DLT 独立对拍 ==")
    for N in (4, 30):
        nu_1s, nu_2s, _, H_pix_gt = gen_planar_data(N, K, R_c2, t_c2, seed=1)
        H_est = calc_vo_Hmat_4p(N, nu_1s, nu_2s, K)
        H_dlt = calc_vo_Hmat_4p_dlt(nu_1s, nu_2s)   # 已归一到 h_33=1
        # 4点法结果经 K(.)K^{-1} 到像素空间后按自身 [2,2] 归一化 (尺度不变), 再与 DLT 比
        rel = rel_fro(calc_vo_Hmat_pix(H_est, K) / calc_vo_Hmat_pix(H_est, K)[2, 2],
                      H_dlt)
        # DLT 与像素空间真值比对
        rel_gt = rel_fro(H_dlt / H_dlt[2, 2], H_pix_gt / H_pix_gt[2, 2])
        print(f"N={N:2d}: 4点法(KHK^-1) vs DLT rel={rel:.2e}, "
              f"DLT vs 真值 rel={rel_gt:.2e}")
        assert rel < 1e-7 and rel_gt < 1e-7, f"N={N} DLT 对拍失败"


def test_noise(K, R_c2, t_c2):
    """5. 噪声鲁棒性: 误差随噪声增大; 点数增多误差减小。"""
    print("== 5. 噪声鲁棒性 ==")
    print("  (a) 固定 N=30, 扫描像素噪声 sigma:")
    errs_sigma = []
    for sigma in (0.0, 0.5, 1.0, 2.0):
        nu_1s, nu_2s, A_gt, _ = gen_planar_data(30, K, R_c2, t_c2, seed=2, noise=sigma)
        H_est = calc_vo_Hmat_4p(30, nu_1s, nu_2s, K)
        reproj = reproj_err(nu_1s, nu_2s, calc_vo_Hmat_pix(H_est, K)).mean()
        h_err = rel_fro(H_est, A_gt / A_gt[2, 2])
        errs_sigma.append(reproj)
        print(f"    sigma={sigma:4.1f} px -> 重投影误差 {reproj:.4f} px, "
              f"H 估计 rel 误差 {h_err:.4f}")
    assert errs_sigma[-1] > errs_sigma[0], "噪声应增大误差"

    print("  (b) 固定 sigma=1.0 px, 扫描点数 N (看 H 估计误差, 越多点越准):")
    errs_N = []
    for N in (4, 10, 30, 100, 300):
        nu_1s, nu_2s, A_gt, _ = gen_planar_data(N, K, R_c2, t_c2, seed=3, noise=1.0)
        H_est = calc_vo_Hmat_4p(N, nu_1s, nu_2s, K)
        h_err = rel_fro(H_est, A_gt / A_gt[2, 2])
        reproj = reproj_err(nu_1s, nu_2s, calc_vo_Hmat_pix(H_est, K)).mean()
        errs_N.append(h_err)
        # 注: N=4 是 8x8 方阵, 最小二乘解精确插值 4 个带噪点 (重投影误差=0),
        #     但 H 对噪声敏感; N 增多后最小二乘平均噪声, H 估计误差下降。
        print(f"    N={N:3d} -> H 估计 rel 误差 {h_err:.4f}, "
              f"拟合点重投影 {reproj:.4f} px")
    assert errs_N[-1] < errs_N[0], "点数增多 H 估计误差应减小"
    return errs_sigma, errs_N


def test_underdetermined(K, R_c2, t_c2):
    """6. N=3 欠定演示: 6x8 矩阵秩亏, 最小二乘解不唯一, 无法正确重投影。"""
    print("== 6. N=3 欠定 (说明 4 点是最低要求) ==")
    nu_1s, nu_2s, _, _ = gen_planar_data(3, K, R_c2, t_c2, seed=0)
    A = np.zeros((6, 8))
    invK = np.linalg.inv(K)
    for i in range(3):
        u1 = invK @ nu_1s[i]; u1 /= u1[2]
        u2 = invK @ nu_2s[i]; u2 /= u2[2]
        A[2 * i] = [u2[0], u2[1], 1, 0, 0, 0, -u1[0] * u2[0], -u1[0] * u2[1]]
        A[2 * i + 1] = [0, 0, 0, u2[0], u2[1], 1, -u1[1] * u2[0], -u1[1] * u2[1]]
    print(f"A 形状 {A.shape}, 秩 = {np.linalg.matrix_rank(A)} (< 8, 欠定)")


if __name__ == "__main__":
    np.random.seed(42)
    K = K_default()
    R_c2 = rot_y(20.0) @ np.array([[1, 0, 0],
                                   [0, np.cos(np.deg2rad(10)), -np.sin(np.deg2rad(10))],
                                   [0, np.sin(np.deg2rad(10)), np.cos(np.deg2rad(10))]])
    t_c2 = np.array([0.3, -0.1, 0.5])

    test_minimal_and_overdetermined(K, R_c2, t_c2)
    test_scale_note(K, R_c2, t_c2)
    test_dlt_crosscheck(K, R_c2, t_c2)
    errs_sigma, errs_N = test_noise(K, R_c2, t_c2)
    test_underdetermined(K, R_c2, t_c2)
    print("\nALL TESTS PASSED")
