# -*- coding: utf-8 -*-
"""
chap6_1_RLbasic/corridor.py

Part VI 强化学习配套代码验证 —— 走廊 MDP 共享模块（单一来源）。

环境规格（见 docs/find-bugs/verify-RL-codes.md §3）：
  - 状态 0..Ns-1（Ns=5），动作 0（左）、1（右），确定性转移：
      action 0 -> max(0, s-1)；action 1 -> min(Ns-1, s+1)
  - 回报 R(3, 1) = 1，其余 0；γ=0.9；初始状态均匀分布。

本模块同时提供：
  - MDPModel（模型已知，供 DP 类算法）：state_transfer_distb / init_state_distb / reward
  - MDP（模型未知，供采样类算法）：state_transfer / init_state / reward
  - sample_trail_policy：书中采样函数（单一来源，chap6_1 与 chap6_2 共用）
  - 参考求解器：均匀策略的贝尔曼不动点、最优价值 V* 与最优贪心动作

注意：书中最优解与验证计划的原始描述不同。计划 §3 声称「最优策略 = 全取
action 1、V* = [0.729, 0.81, 0.9, 1.0, 0]、状态 4 两个动作等价」——这是错误的，
那其实是「全右策略」的价值（隐含把状态 4 当吸收态）。真实最优为：
  最优贪心动作 [1, 1, 1, 1, 0]（状态 4 应取 action 0 回退到状态 3 以重复收集回报 1）
  最优价值 V* = [3.8368, 4.2632, 4.7368, 5.2632, 4.7368]
  状态 4：Q(4,0)=γ·V*(3)=4.7368 > Q(4,1)=γ·V*(4)=4.2632，两动作不等价。
"""

import numpy as np

NS = 5          # 状态数（走廊长度）
NA = 2          # 动作数（0=左, 1=右）


class MDPModel:
    """模型已知的 MDP（供 DP 类算法）——书 chap6_1_basics_2.tex 的 MDPModel。"""
    def __init__(self, Ns: int, Na: int):
        self.Ns, self.Na = Ns, Na
        self.p_sas = np.zeros([self.Ns, self.Na, self.Ns])
        self.p_s0 = np.zeros([self.Ns])
        for s in range(self.Ns):
            self.p_s0[s] = 1.0 / self.Ns                     # 均匀初始分布
            for a in range(self.Na):
                s2 = max(0, s - 1) if a == 0 else min(self.Ns - 1, s + 1)
                self.p_sas[s, a, s2] = 1.0                   # 确定性转移
        pass

    def state_transfer_distb(self, s: int, a: int):
        assert 0 <= s and s < self.Ns and 0 <= a and a < self.Na
        return self.p_sas[s, a]

    def init_state_distb(self):
        return self.p_s0

    def reward(self, s: int, a: int):
        # 走廊回报：仅 (s=3, a=1) 取 1，其余 0
        r = 1.0 if (s == 3 and a == 1) else 0.0
        return r


class MDP:
    """模型未知的 MDP 环境（供采样类算法）——书 chap6_1_basics_2.tex 的 MDP。"""
    def __init__(self, Ns: int, Na: int):
        self.Ns, self.Na = Ns, Na
        self.p_sas = np.zeros([self.Ns, self.Na, self.Ns])
        self.p_s0 = np.zeros([self.Ns])
        for s in range(self.Ns):
            self.p_s0[s] = 1.0 / self.Ns
            for a in range(self.Na):
                s2 = max(0, s - 1) if a == 0 else min(self.Ns - 1, s + 1)
                self.p_sas[s, a, s2] = 1.0
        pass

    def state_transfer(self, s: int, a: int):
        assert 0 <= s and s < self.Ns and 0 <= a and a < self.Na
        s_out = np.random.choice(self.Ns, p=self.p_sas[s, a])
        return s_out

    def init_state(self):
        return np.random.choice(self.Ns, p=self.p_s0)

    def reward(self, s: int, a: int):
        r = 1.0 if (s == 3 and a == 1) else 0.0
        return r


def make_model():
    return MDPModel(NS, NA)


def make_env():
    return MDP(NS, NA)


def sample_trail_policy(env: MDP, pi_func=None, L_e=100):
    """书 chap6_1_basics_2.tex 的采样函数（单一来源）。

    rs[t] = R̄(s_{t-1}, a_t)，即进入 s_t 的回报（标准记号 R_t），rs[0] 为占位。
    按出发约定（2026-08-26 统一，见 code_2_mc_td_predict.py 头注）：
      - Q 侧（chap6_2/6_3）配对 (s_{t-1}, a_t) 与 rs[t]，直接用即可；
      - V 侧（MC/TD0/TDn）状态 s_t 的回报从 rs[t+1] 起算。
    """
    states, rs, a_s = [env.init_state()], [0], [None]
    for t in range(1, L_e + 1):
        a_s.append(np.random.choice(env.Na, p=pi_func(states[t - 1])))
        states.append(env.state_transfer(states[t - 1], a_s[t]))
        rs.append(env.reward(states[t - 1], a_s[t]))
    return rs, states, a_s


# ----------------------------------------------------------------------
# 参考求解器（模型已知，用于对照）
# ----------------------------------------------------------------------

def uniform_policy(s: int):
    """均匀随机策略 π(a|s) = 1/Na。"""
    return np.ones(NA) / NA


def bellman_fixed_point(pi_func=uniform_policy, gamma=0.9):
    """给定策略下的贝尔曼期望方程精确不动点 V = (I - γP(π))^{-1} R̄(π)。

    供 DP/MC/TD 策略评估对照（预测问题）。"""
    mdp = MDPModel(NS, NA)
    P = np.zeros([NS, NS])
    R = np.zeros([NS])
    for s in range(NS):
        pi_s = pi_func(s)
        R[s] = sum(pi_s[a] * mdp.reward(s, a) for a in range(NA))
        P[s, :] = sum(pi_s[a] * mdp.p_sas[s, a] for a in range(NA))
    I = np.eye(NS)
    return np.linalg.solve(I - gamma * P, R)


def optimal_value_and_greedy(gamma=0.9, iters=2000):
    """模型已知的价值迭代 → V* 与最优贪心动作（控制问题真值）。

    返回 (V_star, greedy)：
      V_star = [3.8368, 4.2632, 4.7368, 5.2632, 4.7368]
      greedy = [1, 1, 1, 1, 0]"""
    mdp = MDPModel(NS, NA)
    V = np.zeros(NS)
    for _ in range(iters):
        V_new = np.zeros(NS)
        for s in range(NS):
            Qs = [mdp.reward(s, a) + gamma * mdp.p_sas[s, a] @ V for a in range(NA)]
            V_new[s] = max(Qs)
        V = V_new
    greedy = np.zeros(NS, dtype=int)
    for s in range(NS):
        Qs = [mdp.reward(s, a) + gamma * mdp.p_sas[s, a] @ V for a in range(NA)]
        greedy[s] = int(np.argmax(Qs))
    return V, greedy


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    V_fp = bellman_fixed_point()
    V_star, greedy = optimal_value_and_greedy()
    print("均匀策略贝尔曼不动点 V_π =", V_fp)
    print("最优价值 V*            =", V_star)
    print("最优贪心动作           =", greedy)
    print("状态4: Q(4,0) =", 0.9 * V_star[3], " Q(4,1) =", 0.9 * V_star[4])
