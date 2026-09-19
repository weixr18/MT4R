# 四元数二次样条插值（Part I, chap1_4_interp_2）—— 核心算法
#
# 以 Hamilton 单位四元数 [w,x,y,z] 在 S^3（单位四元数流形）上实现"过采样点"的
# 二次样条插值（C1，体坐标角速度连续）：逐段用切空间（旋转矢量）二次多项式
#     q(u) = q_i ⊗ exp(a1 u + a2 u^2),   u ∈ [0,1]
# 段内系数 a1,a2 由段首节点体坐标角速度与段末过点条件显式确定；节点角速度由
# "节点两侧体坐标角速度连续"（C1 连续）的前向显式递推得到，无需解方程组/迭代。
# 体坐标角速度用第 1.2 节的 SO(3) 右雅可比：ω^b = J_r(φ) φ̇
# （见 docs/code-verify/lie-spline-interp-verify.md；tex 只贴核心函数，完整文件在此）。
# 数组索引为 0 基：采样序列 qs = [q_0, ..., q_N] 共 N+1 个、N 段，节点角速度 om 长 N+1（含终端节点 N）。
import numpy as np

EPS = 1e-8


# ---------------- 四元数基础 ----------------

def q_exp(v):
    """指数映射：旋转矢量 v (3,) -> 单位四元数 [w,x,y,z]（Hamilton）。"""
    theta = np.linalg.norm(v)
    if theta < EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate(([np.cos(theta / 2)], np.sin(theta / 2) * v / theta))


def q_log(q):
    """对数映射：单位四元数 [w,x,y,z]（Hamilton）-> 旋转矢量 (3,)（主值）。"""
    q = q / np.linalg.norm(q)
    qv = q[1:]
    n = np.linalg.norm(qv)
    if n < EPS:
        return np.zeros(3)
    theta = 2 * np.arctan2(n, q[0])
    return theta * qv / n


def q_mul(p, q):
    """Hamilton 四元数乘法 p⊗q，输入输出均为 [w,x,y,z]。"""
    w = p[0] * q[0] - np.dot(p[1:], q[1:])
    v = p[0] * q[1:] + q[0] * p[1:] + np.cross(p[1:], q[1:])
    return np.concatenate(([w], v))


def q_inv(q):
    """单位四元数逆元 = 共轭。"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _skew(v):
    """反对称阵 [v]_x。"""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


# ---------------- SO(3) 右雅可比（第 1.2 节 subsubsec:rot-disturb-jacobian） ----------------

def so3_Jr(phi):
    """SO(3) 右雅可比 J_r(φ)：ω^b = J_r(φ) φ̇。
    J_r(φ) = I - a(θ)[φ]_× + b(θ)[φ]_×²,  a=(1-cosθ)/θ², b=(θ-sinθ)/θ³。"""
    theta = np.linalg.norm(phi)
    K = _skew(phi)
    if theta < EPS:                       # 小角度 Taylor：J_r ≈ I - ½[φ]_× + ⅙[φ]_×²
        return np.eye(3) - 0.5 * K + (1.0 / 6.0) * (K @ K)
    a = (1.0 - np.cos(theta)) / theta ** 2
    b = (theta - np.sin(theta)) / theta ** 3
    return np.eye(3) - a * K + b * (K @ K)


# ---------------- 段内二次多项式系数与运动量 ----------------

def quad_segment_coeffs(w, wi):
    """第 i 段切空间二次多项式系数 a1,a2：
        q(u) = q_i ⊗ exp(a1 u + a2 u²),   u ∈ [0,1]
    其中 w = log(q_i⁻¹⊗q_{i+1})（段相对旋转），wi=ω_i 为段首体坐标角速度。
    约束：a1=ω_i（段首物理角速度，J_r(0)=I）、a1+a2=w（过 q_{i+1}）。"""
    a1 = wi
    a2 = w - wi
    return a1, a2


def quad_segment(qi, w, wi, u):
    """第 i 段在 u ∈ [0,1] 处的插值四元数：q(u) = q_i ⊗ exp(a1 u + a2 u²)。"""
    a1, a2 = quad_segment_coeffs(w, wi)
    return q_mul(qi, q_exp(a1 * u + a2 * u ** 2))


def quad_segment_body_vel(w, wi, u):
    """第 i 段在 u 处的体坐标角速度（闭式）：ω^b(u) = J_r(φ(u)) φ̇(u)。"""
    a1, a2 = quad_segment_coeffs(w, wi)
    phi = a1 * u + a2 * u ** 2
    phid = a1 + 2 * a2 * u
    return so3_Jr(phi) @ phid


# ---------------- 节点角速度求解（C1 连续，前向显式递推） ----------------

def solve_node_velocities(w):
    """求使曲线 C1 连续的节点体坐标角速度 ω_{0:N}（N 段、N+1 个节点，夹持起点 ω_0=0）。
    C1 条件：内部节点 j 两侧体坐标角速度相等，即
        ω^b_{j-1}(1) = J_r(w_{j-1})(2w_{j-1} - ω_{j-1}) = ω_j，j=1..N-1
    这是前向显式递推，无需解方程组/迭代；末位 om[N] 为终端节点 N 的角速度（末段终点处），
    由同一递推多推一步得到，无法预先指定。"""
    Nn = len(w)                            # N 段 -> 节点 0..N 共 N+1 个
    om = np.zeros((Nn + 1, 3))             # 夹持起点 ω_0 = 0（静止起动）
    for i in range(Nn):
        om[i + 1] = so3_Jr(w[i]) @ (2 * w[i] - om[i])
    return om


# ---------------- 整体采样与 Slerp ----------------

def LieQuadSplineSample(qs, M):
    """给定采样四元数 qs ((N+1),4)，返回过全部采样点的二次样条插值曲线（四元数序列）。
    N+1 个采样点 -> N 段。每段内取 M 个点（不含段末 u=1，由下一段 u=0 承接；末段补终点）。"""
    N = len(qs) - 1                        # N+1 个采样四元数 -> N 段（节点 0..N）
    w = [q_log(q_mul(q_inv(qs[i]), qs[i + 1])) for i in range(N)]
    om = solve_node_velocities(w)
    out = []
    for i in range(N):
        a1, a2 = quad_segment_coeffs(w[i], om[i])
        for u in np.linspace(0.0, 1.0, M, endpoint=(i == N - 1)):
            out.append(q_mul(qs[i], q_exp(a1 * u + a2 * u ** 2)))
    return np.array(out)


def q_slerp(g0, g1, u):
    """球面线性插值 Slerp（S^3 上测地线/匀速旋转）：
    q(u) = g0 ⊗ exp(u log(g0⁻¹⊗g1))。"""
    w = q_log(q_mul(q_inv(g0), g1))
    return q_mul(g0, q_exp(u * w))
