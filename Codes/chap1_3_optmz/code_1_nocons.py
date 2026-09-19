import numpy as np

####################### Algorithms #######################


def opt_nc_gd(df_func, x_0, epsilon=1e-4, alpha=1e-2):
    x_k = x_0
    while True:
        grad = df_func(x_k) # grad = (df/dx)^T
        if np.linalg.norm(grad) < epsilon:
            break
        x_k = x_k - alpha * grad
    return x_k


def opt_nc_newton(df_func, ddf_func, x_0, epsilon=1e-4):
    x_k = x_0
    while True:
        grad = df_func(x_k)
        if np.linalg.norm(grad) < epsilon:
            break
        H_inv = np.linalg.inv(ddf_func(x_k))
        x_k = x_k - H_inv @ grad
    return x_k


def GR_line_search(f_func, x_0, t, h_0=1, r=1e-5):
    a, b, h, phi = 0, h_0, h_0, 0.618
    while np.abs(h) > r:
        h = b - a
        p, q = a + (1-phi) * h, a + phi * h
        f_p, f_q = f_func(x_0 + p*t), f_func(x_0 + q*t)
        f_0, f_1 = f_func(x_0), f_func(x_0 + t)
        if f_p > phi*f_0 + (1-phi)*f_1:
            a, b = a, p
        elif f_q > (1-phi)*f_0 + phi*f_1:
            a, b = a, q
        elif f_p < f_q:
            a, b = a, q
        else:
            a, b = p, b
    return (a + b) / 2 


def opt_nc_LSGD(f_func, df_func, x_0, epsilon=1e-4):
    x_k = x_0
    while True:
        grad = df_func(x_k)
        if np.linalg.norm(grad) < epsilon:
            break
        t = - grad
        h = GR_line_search(f_func, x_k, t)
        x_k = x_k + h * t
    return x_k


def opt_nc_damp_newton(f_func, df_func, ddf_func, x_0, epsilon=1e-4):
    x_k, = x_0
    while True:
        grad = df_func(x_k)
        if np.linalg.norm(grad) < epsilon:
            break
        H_inv = np.linalg.inv(ddf_func(x_k))
        t = - H_inv @ grad
        h = GR_line_search(f_func, x_k, t)
        x_k = x_k + h * t
    return x_k


def opt_nc_conj_grad(f_func, df_func, x_0, epsilon=1e-4, n:int=10):
    k, x_k = 0, x_0
    while True:
        grad = df_func(x_k)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < epsilon:
            break
        t = - grad
        if k % n != 0:
            t += (grad_norm / last_grad_norm)**2 * last_t
        h = GR_line_search(f_func, x_k, t)
        x_k = x_k + h * t
        last_t, last_grad_norm = t, grad_norm
        k += 1
    return x_k

