# -*- coding: utf-8 -*-
"""
chap6_3_pgmeth/code_1_reinforce.py

验证书 chap6_3_pgmeth_2.tex 的 sample_trail_policy_NN + REINFORCE_og
（B22/B23/B26 整块重写后原样）：
  - 可运行、无异常（softmax / 索引 / 形状）；
  - 采样平均回报随 N_M 上升；
  - 最终策略向最优方向收敛（[1,1,1,1,0]）或至少非退化、可解释。

复用（单一来源）：
  - MLP：chap5_1_DL/code_1_MLP.py（书第 V 部分 MLP）
  - 环境：chap6_1_RLbasic/corridor.py 的走廊 MDP

已知注意点：
  - 书中 DL 章把 GD_update 定义为各优化器内的嵌套函数，模块级并不存在；
    REINFORCE_og 直接调用模块级 GD_update，故本脚本按书中一行定义补一个
    模块级 GD_update（属「书中依赖的辅助函数需自行定义」，非 bug）。
  - MLP 末层为 ReLU（书 MLP.forward 全层 ReLU），softmax 的 logits 即 ReLU
    输出（>=0）。z - z.max() 只保证数值稳定，无法消除「某样本输出层
    pre-activation 全 <= 0 时梯度被 ReLU 导数清零」的死神经元风险，收敛
    慢时需记录（见验证计划 §4.4 第 4 点）。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chap5_1_DL"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chap6_1_RLbasic"))
from code_1_MLP import MLP
from corridor import make_env

np.random.seed(42)


def GD_update(param, grad, alpha):
    """书中定义的一行梯度下降更新（DL 章内嵌于优化器，此处按 REINFORCE 需要补为模块级）。"""
    return param - alpha * grad


def sample_trail_policy_NN(env, piMLP: MLP, L_e=100):
    """书 chap6_3_pgmeth_2.tex 的 sample_trail_policy_NN（整块重写后原样）。"""
    states, rs, a_s = [env.init_state()], [0], [None]
    for t in range(1, L_e + 1):
        s_in = np.eye(env.Ns)[states[t - 1]]              # one-hot
        z = piMLP.forward(s_in)
        z = z - z.max()                                   # 数值稳定 softmax
        pi_a = np.exp(z) / np.exp(z).sum()
        a_s.append(np.random.choice(env.Na, p=pi_a))
        states.append(env.state_transfer(states[t - 1], a_s[t]))
        rs.append(env.reward(states[t - 1], a_s[t]))
    return rs, states, a_s


def REINFORCE_og(env, batch_size: int, N_S: int = 10, N_M: int = 100,
                 alpha: float = 1e-4, L_e: int = 100, W_NN: int = 100):
    """书 chap6_3_pgmeth_2.tex 的 REINFORCE_og（整块重写后原样）。

    验证用附加：返回 (pi_func_mlp, ret_log)，ret_log 为每个外循环后
    采样 N_S 条轨迹的平均回报（γ=1 时 = 每条轨迹的总回报），供「回报上升」判定。
    """
    piMLP = MLP(3, [env.Ns, W_NN, W_NN, env.Na])          # L=3：输入/两隐层/输出
    ret_log = []                                          # 验证用：平均回报记录

    def pi_func_mlp(s: int):
        z = piMLP.forward(np.eye(env.Ns)[s])
        z = z - z.max()
        return np.exp(z) / np.exp(z).sum()

    for k in range(N_M):
        xs, allGs, alla_s = [], [], []
        rets = []
        for _ in range(N_S):
            rs, states, a_s = sample_trail_policy_NN(env, piMLP, L_e)
            Gs = np.zeros(L_e + 2)
            for t in range(L_e, 0, -1):                   # 后缀回报 G_t = Σ_{τ=t}^{L_e} r_τ（γ=1）
                Gs[t] = rs[t] + Gs[t + 1]
            rets.append(Gs[1])                            # 本轨迹总回报
            for t in range(1, L_e + 1):
                xs.append(states[t - 1]), allGs.append(Gs[t]), alla_s.append(a_s[t])
        ret_log.append(float(np.mean(rets)))
        N, N_b = len(xs), len(xs) // batch_size
        xs, allGs, alla_s = np.array(xs), np.array(allGs), np.array(alla_s)
        ids = np.arange(N); np.random.shuffle(ids)
        for b in range(N_b):
            piMLP.zero_grad()
            b_ids = ids[b * batch_size:(b + 1) * batch_size]
            b_J, b_xs, b_Gs, b_as = 0, xs[b_ids], allGs[b_ids], alla_s[b_ids]
            for i in range(batch_size):
                z = piMLP.forward(np.eye(env.Ns)[b_xs[i]])
                z = z - z.max()
                pi_est = np.exp(z) / np.exp(z).sum()
                b_J += b_Gs[i] * np.log(pi_est[b_as[i]])
                one_hot = np.zeros(env.Na); one_hot[b_as[i]] = 1
                dJdy = - b_Gs[i] * (one_hot - pi_est)     # softmax 交叉熵精确梯度（"-" 使 GD 等价于梯度上升）
                piMLP.backward(dJdy)
            for p in piMLP.params:
                for l in range(1, piMLP.L + 1):
                    piMLP.params[p][l] = GD_update(piMLP.params[p][l], piMLP.grads[p][l], alpha)
    return pi_func_mlp, ret_log


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    env = make_env()
    N_M = 60
    pi_func_mlp, ret_log = REINFORCE_og(env, batch_size=250, N_S=10, N_M=N_M,
                                        alpha=1e-4, L_e=100, W_NN=100)
    print()
    print(f"采样平均回报：初始 {ret_log[0]:.3f}，最终 {ret_log[-1]:.3f}"
          f"（最优策略下理论均值约 49）")
    print("最终策略 π(a|s)：")
    for s in range(env.Ns):
        print(f"  s={s}: P(左)={pi_func_mlp(s)[0]:.3f}  P(右)={pi_func_mlp(s)[1]:.3f}")
