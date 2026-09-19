import numpy as np
from scipy.linalg import solve_discrete_are

# 定义矩阵
A = np.array([[1, 1], [0, 1]])  # 状态矩阵
B = np.array([[0.5], [1]])      # 输入矩阵
Q = np.array([[1, 0], [0, 1]])  # 状态权重矩阵
R = np.array([[1]])              # 输入权重矩阵

# 求解DARE
P = solve_discrete_are(A, B, Q, R)
print("P = \n", P)

# 验证结果
P_check = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
print("\n验证结果：\n", P_check)
print("\n误差：\n", np.linalg.norm(P - P_check))