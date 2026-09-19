import numpy as np
import matplotlib.pyplot as plt
from code_1_quad import Quadratic_spline_sample

# 测试函数
def test_quadratic_spline():
    func = np.cos
    # 原始采样点：N 段、N+1 个点（首尾即区间边缘）
    N = 29
    xs = np.linspace(0, 2 * np.pi, N + 1, endpoint=True)
    print("xs:", xs.shape)
    ys = func(xs).reshape(-1, 1)
    # 插值采样点
    M = 200
    new_xs = np.linspace(0, 2 * np.pi, M, endpoint=True)
    new_ys = Quadratic_spline_sample(xs, ys, N, M)
    true_ys = func(new_xs)
    # visualize
    plt.figure(figsize=(10, 6))
    plt.plot(new_xs, true_ys, label='True Function', color='green', linestyle='--')
    plt.plot(xs, ys, 'o', label='Original Points', color='red')
    plt.plot(new_xs, new_ys, label='Quadratic Spline Interpolation', color='blue')
    plt.title('Quadratic Spline Interpolation vs True Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# 运行测试
test_quadratic_spline()
