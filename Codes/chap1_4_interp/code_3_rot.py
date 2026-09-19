import numpy as np
from numpy.linalg import norm
from code_1_quad import quadratic_spline_interp
EPSILON = 1e-8


def q_to_vec(q):
    theta = 2 * np.arctan2(norm(q[1:]), q[0])
    if np.abs(np.sin(theta/2)) < EPSILON * 1e-4:
        return np.zeros(3)
    if np.abs(np.sin(theta/2)) < EPSILON:
        return q[1:] 
    return q[1:] / np.sin(theta/2) * theta


def vec_to_q(vec):
    q, theta = np.zeros(4), norm(vec)
    if theta < EPSILON:
        q[0] = 1
        q[1:] = vec
    else:
        q[0] = np.cos(theta/2)
        q[1:] = vec / theta * np.sin(theta/2)
    return q


def Quadratic_spline_sample_q(xs, qs, N, M):
    assert xs.shape == (N+1,)                # N+1 个采样点 -> N 段
    assert qs.shape == (N+1, 4)
    a, b, vecs = np.min(xs), np.max(xs), np.zeros([N+1, 3])
    for i in range(N+1):
        vecs[i] = q_to_vec(qs[i])
    new_xs = np.linspace(a, b, num=M, endpoint=True)
    new_vecs, new_qs = np.zeros([M, 3]), np.zeros([M, 4])
    for dd in range(3):
        f_func = quadratic_spline_interp(xs, vecs[:, dd], N)
        new_vecs[:, dd] = np.array(list(map(f_func, new_xs)))
    for i in range(M):
        new_qs[i] = vec_to_q(new_vecs[i])
    new_qs = new_qs / np.linalg.norm(new_qs, axis=1, keepdims=True)
    return new_qs

