import numpy as np
from code_1_nocons import GR_line_search


def opt_qp_barrier(H, g, M_eq, b_eq, M, b, x_0, lambda_b0=10, 
                   gamma=0.95, beta=0.9, epsilon_1=1e-1, epsilon_2=1e-4):
    lambda_0, lambda_b = np.ones_like(b_eq), lambda_b0
    x_ext_k = np.concatenate([x_0, lambda_0])
    n_x, n_lamb, x_last = x_0.shape[0], lambda_0.shape[0], x_0
    def barrier(x_ext):
        return np.sum(-np.log(b - M @ x_ext[:n_x]))
    while True:
        lambda_b, epsilon_1 = lambda_b * gamma, epsilon_1 * np.sqrt(gamma)
        momentum = np.zeros_like(x_ext_k)
        while True:
            def dL_b(x_ext, lbd_b):
                x, lambda_ = x_ext[:n_x], x_ext[n_x:]
                t = np.zeros(n_x)   # 梯度累加器维度 = 变量数 n_x（M 行数 ≠ n_x 也成立）
                for i in range(b.shape[0]):
                    t += M[i, :] / (b[i] - M[i:i+1, :] @ x)
                L_x = H @ x + g + M_eq.T @ lambda_ + lbd_b * t
                L_lambda = M_eq @ x - b_eq
                return L_x, L_lambda
            def J_func(x_ext):
                A, B = dL_b(x_ext, lambda_b)
                return 0.5 * (np.sum(A**2) + np.sum(B**2))
            L_x, L_lambda = dL_b(x_ext_k, lambda_b)
            K_k, x_k = np.zeros_like(H), x_ext_k[:n_x]
            for i in range(b.shape[0]):
                m_i = M[i:i+1, :]
                K_k += m_i.T @ m_i / (m_i @ x_k - b[i])**2
            tmp1 = np.block([
                [H + lambda_b * K_k, M_eq.T], [M_eq, np.zeros([n_lamb, n_lamb])]
            ]) 
            gradJ = tmp1 @ np.hstack([L_x, L_lambda])
            momentum = beta * momentum + (1-beta) * gradJ
            h = GR_line_search(J_func, x_ext_k, - momentum, h_0=1)
            # print(x_ext_k, J_func(x_ext_k), h, momentum)
            x_ext_k = x_ext_k - h * momentum
            if J_func(x_ext_k) < epsilon_1:
                break
        neq_gap = lambda_b * barrier(x_ext_k)
        if np.isnan(neq_gap):
            return x_last
        elif neq_gap < epsilon_2:
            return x_ext_k[:n_x]
        x_last = x_ext_k[:n_x]
    
