import numpy as np
import matplotlib.pyplot as plt
from code_2_cubic import Cubic_spline_sample

# 测试函数
def test_quadratic_spline():
    # func = lambda x: x**2 - 0.5 * x**2 + 2
    # func = lambda x: -0.1 * x**3 + 0.5 * x**2 + 2
    func = np.sin
    N = 49  # N 段、N+1 个点（首尾即区间边缘）
    xs = np.linspace(0, 2 * np.pi, N + 1)
    # print("xs:", xs)
    ys = func(xs).reshape(-1, 1)
    M = 200
    new_xs = np.linspace(0, 2 * np.pi, M)
    new_ys = Cubic_spline_sample(xs, ys, N, M)
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
