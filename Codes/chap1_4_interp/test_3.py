import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from code_3_rot import Quadratic_spline_sample_q

def generate_test_quaternions():
    angles = np.linspace(0, np.pi, 5)  # 5 个点，从 0 到 180 度
    qs = R.from_euler('z', angles).as_quat()  # [x, y, z, w]
    qs = np.roll(qs, shift=1, axis=1)  # 转换为 [w, x, y, z]
    return angles, qs

def visualize_quaternions(qs, title="Quaternion Rotation Visualization"):
    rs = R.from_quat(np.roll(qs, -1, axis=1))  # 转换为 [x, y, z, w]
    directions = rs.apply(np.array([1, 0, 0]))  # 应用到 x 轴方向向量
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.quiver(
        np.zeros(len(directions)), np.zeros(len(directions)),
        directions[:, 0], directions[:, 1],
        angles='xy', scale_units='xy', scale=1, color='blue'
    )
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    xs, qs = generate_test_quaternions()
    N = len(xs) - 1  # N 段、N+1 个采样点
    M = 100  # 插值点数量
    interp_qs = Quadratic_spline_sample_q(xs, qs, N, M)
    print("原始旋转点（粗线）:")
    visualize_quaternions(qs, title="Original Quaternion Rotations")
    print("插值旋转路径（平滑线）:")
    visualize_quaternions(interp_qs, title="Interpolated Quaternion Rotations")