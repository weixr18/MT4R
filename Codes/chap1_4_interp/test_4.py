# 四元数二次样条插值验证（Part I, chap1_4_interp_2）
#
# 运行：cd 4-MT4R-github/Codes/chap1_4_interp
#       E:\Anaconda3\envs\py311-gym\python.exe test_4.py
# 七项验证 + 三张图（存 1-MN4R/imgs/interp/）。
# 全程使用四元数代数 + SO(3) 右雅可比（code_4_lie_spline.py），不引入旋转矩阵。
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体（Windows：微软雅黑），避免图上中文显示为方框
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from code_4_lie_spline import (
    q_exp, q_log, q_mul, q_inv,
    quad_segment_coeffs, quad_segment, quad_segment_body_vel,
    solve_node_velocities, LieQuadSplineSample, q_slerp,
)
from code_1_quad import quadratic_spline_interp   # 欧氏二次样条（小角度退化参照）

np.random.seed(42)
IMGS = r"d:/Projects/2024_MN4R/1-MN4R/imgs/interp"
os.makedirs(IMGS, exist_ok=True)

N = 8          # 采样四元数个数
M = 50         # 每段内采样点数
MAX_ANGLE = 1.4  # 采样旋转矢量模长上限（rad），相邻相对转角 <= 2.8 < pi


# ---------------- 测试数据 ----------------
def random_quats(n, max_angle, min_angle=0.8):
    v = np.random.uniform(-1, 1, (n, 3))
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * np.random.uniform(min_angle, max_angle, (n, 1))
    return np.array([q_exp(vv) for vv in v])


qs = random_quats(N, MAX_ANGLE)
for i in range(N - 1):
    rel = np.linalg.norm(q_log(q_mul(q_inv(qs[i]), qs[i + 1])))
    assert rel < np.pi - 1e-9, f"相邻相对转角 {rel} >= pi，对数主值失效"
w = [q_log(q_mul(q_inv(qs[i]), qs[i + 1])) for i in range(N - 1)]
om = solve_node_velocities(w)   # 节点体坐标角速度（C1 前向递推）


# ---------------- 验证 1：单位范数保持 ----------------
curve = LieQuadSplineSample(qs, M)
norm_err = float(np.max(np.abs(np.linalg.norm(curve, axis=1) - 1)))
assert norm_err < 1e-12, norm_err

# ---------------- 验证 2：过采样点（C0） ----------------
c0_max = 0.0
for i in range(N - 1):
    a1, a2 = quad_segment_coeffs(w[i], om[i])
    qend = q_mul(qs[i], q_exp(a1 + a2))          # 段末 u=1
    c0_max = max(c0_max, min(np.linalg.norm(qend - qs[i + 1]),
                             np.linalg.norm(qend + qs[i + 1])))
assert c0_max < 1e-12, c0_max

# ---------------- 验证 3：节点连续性 C1（体坐标角速度） ----------------
c1_max = 0.0
for j in range(1, N - 1):                              # 节点 j 位于段 j-1 与段 j 之间
    wL = quad_segment_body_vel(w[j - 1], om[j - 1], 1.0)   # 段 j-1 末端 = J_r(w)(2w-ω) = ω_j
    wR = quad_segment_body_vel(w[j], om[j], 0.0)           # 段 j 首端 = ω_j
    c1_max = max(c1_max, np.linalg.norm(wL - wR))
assert c1_max < 1e-9, c1_max

# ---------------- 验证 4：体坐标角速度闭式 vs 数值差分 ----------------
h3 = 1e-6
vel_max = 0.0
for i in range(N - 1):
    for u in np.linspace(0.05, 0.95, M):
        qc = quad_segment(qs[i], w[i], om[i], u)
        qn = quad_segment(qs[i], w[i], om[i], u + h3)
        w_fd = q_log(q_mul(q_inv(qc), qn)) / h3                # ω^b ≈ log(Δq)/h
        w_cl = quad_segment_body_vel(w[i], om[i], u)
        vel_max = max(vel_max, np.linalg.norm(w_fd - w_cl))
assert vel_max < 5e-5, vel_max

# ---------------- 验证 5：节点角加速度跳变（仅 C1，非 C2 的佐证） ----------------
# 二次样条只保证角速度连续；节点两侧体坐标角加速度 α^b = dJ_r[φ̇]φ̇ + J_r φ̈ 一般不同。
# 用段内闭式速度的中心差分估计节点两侧角加速度，验证跳变显著非零。
def body_acc_fd(seg_w, wi, u):
    h = 1e-4
    return (quad_segment_body_vel(seg_w, wi, u + h)
            - quad_segment_body_vel(seg_w, wi, u - h)) / (2 * h)


acc_jump = 0.0
acc_scale = 0.0
for j in range(1, N - 1):
    aL = body_acc_fd(w[j - 1], om[j - 1], 1.0)   # 段 j-1 末端
    aR = body_acc_fd(w[j], om[j], 0.0)           # 段 j 首端
    acc_jump = max(acc_jump, np.linalg.norm(aL - aR))
    acc_scale = max(acc_scale, 0.5 * (np.linalg.norm(aL) + np.linalg.norm(aR)))
assert acc_scale > 1e-9 and acc_jump > 0.05 * acc_scale, (acc_jump, acc_scale)

# ---------------- 验证 6：小角度欧氏退化一致性 ----------------
# 参照：旋转矢量上的欧氏夹持二次样条（第 1.4.1 节 quadratic_spline_interp，独立实现）。
# 小角下李群二次样条（J_r→I）应趋近欧氏二次样条，偏差随角度缩小。
base = np.random.uniform(-1, 1, (N, 3))
base = base / np.linalg.norm(base, axis=1, keepdims=True) * 0.05
base = base - base[0]                                       # 首采样为 identity（q_0=I）


def euc_quad(sv):
    xs = np.arange(sv.shape[0], dtype=float)   # sv.shape[0] 个采样点
    Ns = sv.shape[0] - 1                       # Ns 段
    fs = [quadratic_spline_interp(xs, sv[:, d], Ns) for d in range(3)]
    return lambda t: np.array([f(t) for f in fs])


def small_angle_dev(scale):
    """同一几何按 scale 缩放后，李群 vs 欧氏二次样条的旋转矢量（相对 q0=I）最大差。"""
    sv = scale * base
    qqs = np.array([q_exp(s) for s in sv])
    wws = [q_log(q_mul(q_inv(qqs[i]), qqs[i + 1])) for i in range(N - 1)]
    omm = solve_node_velocities(wws)
    Feuc = euc_quad(sv)
    dev = 0.0
    for i in range(N - 1):
        a1, a2 = quad_segment_coeffs(wws[i], omm[i])
        for u in np.linspace(0.01, 0.99, M):
            qcur = q_mul(qqs[i], q_exp(a1 * u + a2 * u ** 2))
            dev = max(dev, np.linalg.norm(q_log(qcur) - Feuc(i + u)))   # 相对 identity，小角无缠绕
    return dev


deg_max_1 = small_angle_dev(1.0)    # 模长 <= 0.05 rad
deg_max_2 = small_angle_dev(0.5)    # 模长减半
deg_max_4 = small_angle_dev(0.25)   # 模长再减半
order_12 = np.log2(deg_max_1 / deg_max_2) if deg_max_2 > 0 else float("inf")
order_23 = np.log2(deg_max_2 / deg_max_4) if deg_max_4 > 0 else float("inf")
assert deg_max_2 < 1e-4, deg_max_2
assert 2.5 < order_12 < 3.5 and 2.5 < order_23 < 3.5, (order_12, order_23)  # 三阶收敛

# ---------------- 验证 7：Slerp 过点 / 测地线（基线） ----------------
q0, q1 = qs[0], qs[-1]
slerp_pts = np.array([q_slerp(q0, q1, u) for u in np.linspace(0.0, 1.0, M + 1)])
slerp_err = max(
    np.linalg.norm(slerp_pts[0] - q0),
    np.linalg.norm(slerp_pts[-1] - q1),
    np.linalg.norm(slerp_pts[M // 2] - q_mul(q0, q_exp(0.5 * q_log(q_mul(q_inv(q0), q1))))),
    np.max(np.abs(np.linalg.norm(slerp_pts, axis=1) - 1)),
)
assert slerp_err < 1e-12, slerp_err

# ---------------- 结果汇总 ----------------
print("========== 四元数二次样条插值 验证结果 (seed=42) ==========")
print(f"[V1] 单位范数最大偏差           : {norm_err:.3e}")
print(f"[V2] 过采样点 C0 最大差         : {c0_max:.3e}")
print(f"[V3] 节点连续性 C1 最大差       : {c1_max:.3e}")
print(f"[V4] 闭式 vs 差分角速度最大差   : {vel_max:.3e}")
print(f"[V5] 节点角加速度跳变 / 特征尺度 : {acc_jump:.3e} / {acc_scale:.3e}")
print(f"[V6] 欧氏退化一致性最大差(θ≤0.05): {deg_max_1:.3e}")
print(f"[V6] 欧氏退化一致性最大差(θ≤0.025): {deg_max_2:.3e}")
print(f"[V6] 欧氏退化一致性最大差(θ≤0.0125): {deg_max_4:.3e}")
print(f"[V6] 收敛阶(1→0.5→0.25 逐级)   : {order_12:.2f} / {order_23:.2f}")
print(f"[V7] Slerp 过点/测地线最大差    : {slerp_err:.3e}")

# ---------------- 出图 ----------------
# 图数据集：平滑小转角旋转矢量路径（全程 < pi，无缠绕），便于直观展示"过采样点"
def smooth_rotvec_path(n, amp=0.55):
    t = np.linspace(0.0, 1.0, n)
    v = np.zeros((n, 3))
    for d in range(3):
        rng = np.random.default_rng(1000 + d)
        for k in range(1, 4):
            a = rng.uniform(-1, 1)
            p = rng.uniform(0.0, 2.0 * np.pi)
            v[:, d] += amp * a / k * np.sin(k * np.pi * t + p)
    return v - v[0]                                    # 首采样为 identity


sv_fig = smooth_rotvec_path(N)
qs_fig = np.array([q_exp(s) for s in sv_fig])
w_fig = [q_log(q_mul(q_inv(qs_fig[i]), qs_fig[i + 1])) for i in range(N - 1)]
om_fig = solve_node_velocities(w_fig)


def global_rotvec(qq, ww, oo):
    """拼成全局参数 t∈[0, N-1]，返回 (t, 相对 q0=I 的旋转矢量 (K,3))。
    图数据全程 |log q| < pi，主值即连续真值；节点处恰为 log(q_i)=sv_fig[i]（过点）。"""
    ts, rs = [], []
    for i in range(N - 1):
        a1, a2 = quad_segment_coeffs(ww[i], oo[i])
        for u in np.linspace(0.0, 1.0, M, endpoint=False):
            qcur = q_mul(qq[i], q_exp(a1 * u + a2 * u ** 2))
            ts.append(i + u)
            rs.append(q_log(qcur))
    ts.append(N - 1)
    rs.append(q_log(qq[-1]))
    return np.array(ts), np.array(rs)


def slerp_global(qq):
    ts, rs = [], []
    for j in range(N - 1):
        for u in np.linspace(0.0, 1.0, M, endpoint=False):
            ts.append(j + u)
            rs.append(q_log(q_slerp(qq[j], qq[j + 1], u)))
    ts.append(N - 1)
    rs.append(q_log(qq[-1]))
    return np.array(ts), np.array(rs)


t_spl, v_spl = global_rotvec(qs_fig, w_fig, om_fig)
t_sl, v_sl = slerp_global(qs_fig)
assert np.max(np.linalg.norm(v_spl, axis=1)) < np.pi - 0.2      # 全程 < pi，无缠绕

# 体坐标角速度（插值样条：C1 连续；slerp：每段恒定、节点跳变）
om_spl = [quad_segment_body_vel(w_fig[i], om_fig[i], u)
          for i in range(N - 1)
          for u in np.linspace(0.0, 1.0, M, endpoint=False)]
om_spl.append(quad_segment_body_vel(w_fig[N - 2], om_fig[N - 2], 1.0))
om_spl = np.array(om_spl)
om_sl = np.concatenate([[np.linalg.norm(w_fig[j])] * M for j in range(N - 1)]
                       + [[np.linalg.norm(w_fig[N - 2])]])

# --- 图 1：旋转矢量分量（相对 q0=I）：插值样条过点且光滑 vs Slerp 节点折角 ---
fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
for d, ax in enumerate(axs):
    ax.plot(t_spl, v_spl[:, d], '-', color='C0', label='四元数二次样条插值')
    ax.plot(t_sl, v_sl[:, d], '--', color='C3', label='分段Slerp')
    ax.scatter(np.arange(N), sv_fig[:, d], color='k', zorder=5, s=18,
               label='采样点' if d == 0 else None)
    ax.set_ylabel(rf'$v_{{{d+1}}}$ (rad)')
axs[0].legend(loc='best', fontsize=8)
axs[-1].set_xlabel('t（段索引+段内参数 u）')
axs[0].set_title('旋转矢量分量：二次样条插值（过采样点、光滑）vs 分段Slerp（节点折角）')
fig.tight_layout()
fig.savefig(os.path.join(IMGS, 'lie_quad_interp_rotvec.png'), dpi=150)
plt.close(fig)

# --- 图 2：体坐标角速度幅值：插值样条 C1 连续 vs Slerp 节点跳变 ---
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(t_spl, np.linalg.norm(om_spl, axis=1), '-', color='C0', label='四元数二次样条插值')
ax.plot(t_sl, om_sl, '--', color='C3', drawstyle='steps-post', label='分段Slerp')
ax.set_xlabel('t（段索引+段内参数 u）')
ax.set_ylabel(r'$\|\omega^b\|$ (rad)')
ax.set_title('体坐标角速度幅值：二次样条 C1 连续 vs Slerp 节点跳变')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(IMGS, 'lie_quad_interp_angular_velocity.png'), dpi=150)
plt.close(fig)

# --- 图 3：小角度欧氏退化（尺度 1 的同一几何 base，相对 q0=I） ---
t3 = np.linspace(0.0, N - 1.0, (N - 1) * M + 1)
euc_v = np.array([euc_quad(base)(t) for t in t3])
qqs3 = np.array([q_exp(s) for s in base])
wws3 = [q_log(q_mul(q_inv(qqs3[i]), qqs3[i + 1])) for i in range(N - 1)]
omm3 = solve_node_velocities(wws3)
lie_v = np.array([q_log(quad_segment(qqs3[i], wws3[i], omm3[i], u))
                  for i in range(N - 1)
                  for u in np.linspace(0.0, 1.0, M, endpoint=False)]
                 + [q_log(qqs3[-1])])
fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
for d, ax in enumerate(axs):
    ax.plot(t3, lie_v[:, d], '-', color='C0', label='李群二次样条(旋转矢量)')
    ax.plot(t3, euc_v[:, d], '--', color='C1', label='欧氏二次样条(旋转矢量)')
    ax.scatter(np.arange(N), base[:, d], color='k', s=18, zorder=5,
               label='采样点' if d == 0 else None)
    ax.set_ylabel(rf'$v_{{{d+1}}}$ (rad)')
axs[0].legend(fontsize=8)
axs[-1].set_xlabel('t')
axs[0].set_title('小角度退化：欧氏二次样条与李群二次样条旋转矢量重合')
fig.tight_layout()
fig.savefig(os.path.join(IMGS, 'lie_quad_interp_smallangle.png'), dpi=150)
plt.close(fig)

print("图已保存到", IMGS)
