# -*- coding: utf-8 -*-
"""AC 机制验证工具：critic 判别力 + 优势方差（对照 REINFORCE 标量 baseline）。

对应 AC-training-plan.md 阶段 2「关键的机制验证」：
- ① critic 是否真正学到**有判别力**的 V(s)（而非只拟合均值）：
    用「V(s) 与 G_t 的相关系数」+「按状态分组（危险程度）的 V 是否单调」检查；
- ② 优势估计方差是否比 REINFORCE 标量 baseline 不更大：
    比较 AC 优势（TD 优势 δ_t 或 GAE 优势 A_t）的 std 与 REINFORCE 优势（G_t - mean(G_t)）的 std。

用法（cwd 任意，脚本自解析 ../src 与 ../src-rl 路径）：
    # 用训练好的 AC checkpoint 评估（需先由 2_ac.py --ckpt-interval 落一份 .pth）
    python diagnose_ac.py --ckpt res/phase2_sweep/base/ckpt_cartpole_ac_latest.pth

    # 一键训练 + 评估（train 后再拿同一批数据做机制验证）
    python diagnose_ac.py --train-iters 250 --max-steps 200 --seed 0 --adv td

    # 只看某个配置下的优势方差（不依赖 checkpoint，加载/训练后即时采样评估）
    python diagnose_ac.py --ckpt <path> --adv td
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import torch

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (SRC_DIR, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rl_common import (  # noqa: E402
    PolicyNetwork,
    RunningMeanStd,
    ValueNetwork,
    compute_gae,
    compute_returns,
    load_ac_checkpoint,
    sample_trajectory,
    set_seed,
)
from env_cartpole import CONFIG, CartPoleCustomEnv  # noqa: E402


def make_env(env_config=None):
    cfg = json.loads(json.dumps(CONFIG))
    if env_config:
        cfg.update(env_config)
    return CartPoleCustomEnv(cfg)


def sample_batch(env, policy, value_net, obs_rms, n_eps, gamma, adv_mode, lam, device):
    """采样 n_eps 条轨迹，汇聚一个 batch，返回机制验证所需的数据。

    返回 dict：
    - states: 归一化状态 (B,dim)；raw_states: 原始状态 (B,dim)
    - G: 各时刻回报 G_t (B,)；V: V(s_t) (B,)（detached）
    - adv_ac: AC 优势 (B,)（TD 或 GAE）；adv_rf: REINFORCE 优势 G_t - mean(G_t) (B,)
    - theta_abs / x_abs: 各状态（原始）的 |theta|, |x| (B,)（用于按危险程度分组）
    """
    states, raw_states, Gs, Vs, adv_ac, adv_rf, th, xs = [], [], [], [], [], [], [], []
    for _ in range(n_eps):
        traj = sample_trajectory(env, policy, obs_rms)
        obs_rms.update(traj["raw_states"]) if obs_rms is not None else None
        s = np.asarray(traj["states"], dtype=np.float32)
        raw = np.asarray(traj["raw_states"], dtype=np.float32)
        r = np.asarray(traj["rewards"], dtype=np.float64)
        ns = np.vstack([s[1:], obs_rms.normalize(traj["final_state"])[None, :]])
        G = compute_returns(r, gamma)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
        ns_t = torch.as_tensor(ns, dtype=torch.float32, device=device)
        with torch.no_grad():
            v = value_net(s_t).cpu().numpy()
            nv = value_net(ns_t).cpu().numpy()
        nv[-1] = 0.0
        adv = r + gamma * nv - v if adv_mode == "td" else compute_gae(r, v, nv, gamma, lam)
        states.append(s); raw_states.append(raw); Gs.append(G); Vs.append(v)
        adv_ac.append(adv); adv_rf.append(G - G.mean())
        th.append(np.abs(raw[:, 2])); xs.append(np.abs(raw[:, 0]))
    return dict(
        states=np.concatenate(states), raw_states=np.concatenate(raw_states),
        G=np.concatenate(Gs), V=np.concatenate(Vs),
        adv_ac=np.concatenate(adv_ac), adv_rf=np.concatenate(adv_rf),
        theta_abs=np.concatenate(th), x_abs=np.concatenate(xs),
    )


def evaluate_mechanism(env, policy, value_net, obs_rms, gamma, adv_mode, lam, device,
                       n_eps=40, n_bins=5):
    """做阶段 2 的机制验证 ①（critic 判别力）与 ②（优势方差）。"""

    def _corr(a, b):
        if len(a) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    results = {}
    # 用多批采样汇总（降低单批噪声）
    all_V, all_G, all_adv_ac, all_adv_rf, all_th, all_x = [], [], [], [], [], []
    for _ in range(3):
        b = sample_batch(env, policy, value_net, obs_rms, n_eps, gamma, adv_mode, lam, device)
        all_V.append(b["V"]); all_G.append(b["G"]); all_adv_ac.append(b["adv_ac"])
        all_adv_rf.append(b["adv_rf"]); all_th.append(b["theta_abs"]); all_x.append(b["x_abs"])
    V = np.concatenate(all_V); G = np.concatenate(all_G)
    adv_ac = np.concatenate(all_adv_ac); adv_rf = np.concatenate(all_adv_rf)
    th = np.concatenate(all_th); x = np.concatenate(all_x)

    # ① critic 判别力：V(s) 与 G_t 相关
    results["v_corr_G"] = _corr(V, G)

    # ① 危险程度分组：按 |theta| 分桶，看 V 是否随 |theta| 增大而单调下降
    if np.max(th) > 1e-8 and np.std(th) > 1e-8:
        order = np.argsort(th)
        bins = np.array_split(order, n_bins)
        bin_means = [float(np.mean(V[b])) for b in bins]
        bin_theta = [float(np.mean(th[b])) for b in bins]
        results["V_vs_theta_bins"] = bin_means
        results["theta_bins"] = bin_theta
        dec = sum(1 for i in range(1, n_bins) if bin_means[i] <= bin_means[i - 1])
        results["theta_monotone_dec"] = dec  # 完全单调下降 = n_bins-1
    if np.max(x) > 1e-8 and np.std(x) > 1e-8:
        order = np.argsort(x)
        bins = np.array_split(order, n_bins)
        results["V_vs_x_bins"] = [float(np.mean(V[b])) for b in bins]
        results["x_bins"] = [float(np.mean(x[b])) for b in bins]

    # ② 优势方差：AC 优势 vs REINFORCE 优势（标量 baseline）
    results["adv_ac_std"] = float(np.std(adv_ac))
    results["adv_rf_std"] = float(np.std(adv_rf))
    results["adv_ratio_ac_over_rf"] = float(np.std(adv_ac) / (np.std(adv_rf) + 1e-12))

    # 附：V 的尺度（略偏小的 V 代表 critic 尚在收敛途中）
    results["V_mean"] = float(np.mean(V))
    results["V_std"] = float(np.std(V))
    results["G_mean"] = float(np.mean(G))
    results["G_std"] = float(np.std(G))
    return results


def main():
    ap = argparse.ArgumentParser(description="AC 机制验证（critic 判别力 + 优势方差）")
    ap.add_argument("--ckpt", default=None, help="训练好的 AC .pth（由 2_ac.py 落盘）")
    ap.add_argument("--train-iters", type=int, default=0,
                    help=">0 时先训练这么多轮再从最终模型评估（不依赖 --ckpt）")
    ap.add_argument("--adv", choices=["td", "gae"], default="td")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.95)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--reward-coefs", type=str, default="0.2,0.05,1.0,0.01,30")
    ap.add_argument("--init-theta-range", type=str, default="-40,40")
    ap.add_argument("--eps-per-iter", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--n-eval-eps", type=int, default=40)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--no-train-print", action="store_true")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    set_seed(args.seed)
    env_config = {}
    coefs = [float(x) for x in args.reward_coefs.split(",")]
    assert len(coefs) in (4, 5)
    rc = dict(c_theta=coefs[0], c_theta_dot=coefs[1], c_x=coefs[2], c_u=coefs[3])
    if len(coefs) == 5:
        rc["c_term"] = coefs[4]
    env_config["reward_coefs"] = rc
    if args.max_steps is not None:
        env_config["max_steps"] = args.max_steps
    if args.init_theta_range is not None:
        lo, hi = [float(v) for v in args.init_theta_range.split(",")]
        env_config["init_theta_range"] = [lo * np.pi / 180.0, hi * np.pi / 180.0]
    env = make_env(env_config)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    # 训练或加载
    if args.ckpt is not None:
        loaded = load_ac_checkpoint(args.ckpt, obs_dim, act_dim, device=device)
        policy, value_net, obs_rms = loaded["policy"], loaded["value_net"], loaded["obs_rms"]
    else:
        from rl_common import RunningMeanStd
        policy = PolicyNetwork(obs_dim, act_dim, hidden_sizes=(64, 64),
                               init_log_std=0.0).to(device)
        value_net = ValueNetwork(obs_dim, hidden_sizes=(64, 64)).to(device)
        actor_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        critic_opt = torch.optim.Adam(value_net.parameters(), lr=1e-3)
        obs_rms = RunningMeanStd(shape=env.observation_space.shape)
        if args.train_iters > 0:
            import time
            from rl_common import save_ac_checkpoint
            gamma = 1.0
            t0 = time.time()
            for it in range(args.train_iters):
                trajs = []
                for _ in range(args.eps_per_iter):
                    traj = sample_trajectory(env, policy, obs_rms)
                    trajs.append(traj)
                    obs_rms.update(traj["raw_states"])
                S, A, ADV, TGT = [], [], [], []
                for traj in trajs:
                    s = np.asarray(traj["states"], dtype=np.float32)
                    r = np.asarray(traj["rewards"], dtype=np.float64)
                    ns = np.vstack([s[1:], obs_rms.normalize(traj["final_state"])[None, :]])
                    st = torch.as_tensor(s, dtype=torch.float32, device=device)
                    nst = torch.as_tensor(ns, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        v = value_net(st).cpu().numpy()
                        nv = value_net(nst).cpu().numpy()
                    nv[-1] = 0.0
                    adv = (r + gamma * nv - v) if args.adv == "td" else compute_gae(r, v, nv, gamma, args.lam)
                    S.append(s); A.append(traj["actions"]); ADV.append(adv); TGT.append(r + gamma * nv)
                states = np.concatenate(S); actions = np.concatenate(A)
                adv = np.concatenate(ADV); tgt = np.concatenate(TGT)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                st = torch.as_tensor(states, dtype=torch.float32, device=device)
                at = torch.as_tensor(actions, dtype=torch.float32, device=device)
                advt = torch.as_tensor(adv, dtype=torch.float32, device=device)
                tgtt = torch.as_tensor(tgt, dtype=torch.float32, device=device)
                lp = policy.log_prob(st, at)
                actor_opt.zero_grad(); (-(advt * lp).mean()).backward(); actor_opt.step()
                val = value_net(st)
                critic_opt.zero_grad(); ((val - tgtt) ** 2).mean().backward(); critic_opt.step()
            print(f"[train] {args.train_iters} iters in {time.time()-t0:.1f}s", flush=True)

    policy.eval(); value_net.eval()
    res = evaluate_mechanism(env, policy, value_net, obs_rms, 1.0, args.adv, args.lam,
                             device, n_eps=args.n_eval_eps, n_bins=args.n_bins)
    print("=== AC 机制验证 ===")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("\n判读：")
    print(f"  ① V(s)~G_t 相关 = {res.get('v_corr_G', float('nan')):.4f}（越高越有判别力；"
          f"理论下界 0≈只拟合均值）")
    thb = res.get("V_vs_theta_bins")
    if thb:
        print(f"  ① 按 |theta| 分桶的 V（应随危险增大单调下降）：{np.round(thb,2)}")
        print(f"     单调下降步数 = {res.get('theta_monotone_dec')}/{len(thb)-1}")
    print(f"  ② 优势 std：AC={res.get('adv_ac_std'):.4f} vs REINFORCE={res.get('adv_rf_std'):.4f} "
          f"→ 比值 {res.get('adv_ratio_ac_over_rf'):.3f}（<1 说明 AC 优势方差更小/更稳）")
    print(f"  附：V 均/std = {res.get('V_mean'):.2f}/{res.get('V_std'):.2f}；"
          f"G_t 均/std = {res.get('G_mean'):.2f}/{res.get('G_std'):.2f}")


if __name__ == "__main__":
    main()
