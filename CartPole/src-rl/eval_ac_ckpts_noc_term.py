# -*- coding: utf-8 -*-
"""按书 §28.1「稳杆为主」口径，对一组 AC checkpoint 做统一评估并逐种子择优。

对每个 AC 编号 ckpt（`ckpt_<env>_ac_iterNNNN.pth`）跑 `eval_noc_term.eval_ckpt`
（稳杆为主判据：全程不越界 且 末 H=500 步 `|θ|≤2°`，`|x|` 仅作次级参考、不作存活门槛），
按「L1+L2+L3 稳杆存活总数 → 5/5 档数 → L1 J_ach(越小越好) → L2 → L3 → 平均回合长」择优，
取该种子的最优 ckpt，供阶段 3 门禁判定与书文本引用。

这也正是 AC 与 REINFORCE §28.1 基准「同一评估口径」的入口：`eval_noc_term.eval_ckpt` 走
`load_policy_ckpt`（只读 `.pth` 的 `policy` + `obs_rms` + `meta` 键），而 AC checkpoint
（`save_ac_checkpoint`）保留这些键，故可直接复用，无需改 `eval/bench.py` 主体。

用法（cwd=CartPole，幂等可复现）：
    E:/Anaconda3/envs/py311-gym/python.exe -X utf8 src-rl/eval_ac_ckpts_noc_term.py \
        --root res/s_ac_formal_500
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_THIS = os.path.abspath(os.path.dirname(__file__))
for _p in (_THIS, os.path.join(_THIS, "..", "src"), os.path.join(_THIS, "..", "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_noc_term  # noqa: E402


def _num(ckpt):
    m = re.search(r"iter(\d+)", os.path.basename(ckpt))
    return int(m.group(1)) if m else -1


def eval_dir(ckpt_dir, pattern="ckpt_*_ac_iter*.pth"):
    """评估 `ckpt_dir` 下所有编号 ckpt（默认 AC 命名，可 `--pattern` 覆盖为 PPO 命名），返回 {iter: res}。"""
    results = {}
    for ckpt in sorted(glob.glob(os.path.join(ckpt_dir, pattern)), key=_num):
        results[_num(ckpt)] = eval_noc_term.eval_ckpt(ckpt)
        print(f"  [eval] {os.path.basename(ckpt)}", flush=True)
    return results


def score(res):
    """稳杆为主择优分（越大越好）：存活总数 → 5/5 档数 → 负 L1 J_ach → 负 L2 → 负 L3 → 平均回合长。"""
    lv = res["levels"]
    surv = sum(lv[l]["surv"] for l in ["L1", "L2", "L3"])
    fives = sum(1 for l in ["L1", "L2", "L3"] if lv[l]["surv"] == lv[l]["K"])
    js = []
    for l in ["L1", "L2", "L3"]:
        j = lv[l]["J_mean"]
        js.append(-j if j is not None else -1e6)
    mean_len = float(np.mean([np.mean(lv[l]["lens"]) for l in ["L1", "L2", "L3"]]))
    return (surv, fives, js[0], js[1], js[2], mean_len)


def eval_root(root, iter_filter=None, pattern="ckpt_*_ac_iter*.pth"):
    """遍历 `root/seed*/ckpt`，逐种子择优。返回 {seed: dict}。"""
    summary = {}
    for sd in sorted(glob.glob(os.path.join(root, "seed*"))):
        seed = os.path.basename(sd)
        ckpt_dir = os.path.join(sd, "ckpt")
        res_iter = eval_dir(ckpt_dir, pattern=pattern) if os.path.isdir(ckpt_dir) else {}
        if not res_iter:
            print(f"==== {seed}: no iter ckpts ====", flush=True)
            continue
        if iter_filter is not None:
            it = iter_filter if iter_filter in res_iter else max(res_iter)
        else:
            it = max(res_iter, key=lambda k: score(res_iter[k]))
        res = res_iter[it]
        print(f"\n==== {seed} ====", flush=True)
        for k, r in sorted(res_iter.items()):
            surv = sum(x["surv"] for x in r["levels"].values())
            j = r["levels"]["L1"]["J_mean"]
            print(f"  iter{k:04d}: surv_total={surv}/15  L1_J={round(j, 2) if j is not None else '—'}",
                  flush=True)
        print(f"  ** 最优 = iter{it:04d} **", flush=True)
        for l, q in res["levels"].items():
            jm = q["J_mean"]
            print(f"    [{l}] 回合长 {min(q['lens']):.0f}/{np.mean(q['lens']):.0f}/{max(q['lens']):.0f} | "
                  f"稳杆 {q['surv']}/{q['K']} | J_ach={jm if jm is None else round(jm, 2)}±"
                  f"{q['J_std'] if q['J_std'] is None else round(q['J_std'], 2)} | "
                  f"末500|x|均/最={None if q['last_x_mean'] is None else round(q['last_x_mean'], 2)}/"
                  f"{None if q['last_x_max'] is None else round(q['last_x_max'], 2)} | "
                  f"|θ|max={None if q['last_theta_max'] is None else round(q['last_theta_max'], 2)}",
                  flush=True)
        summary[seed] = {"best_iter": it, "res": res}
        with open(os.path.join(sd, "eval_ac_noc_term_best.json"), "w", encoding="utf-8") as f:
            json.dump({"best_iter": it}, f, ensure_ascii=False, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=None, help="单个 seed 的 ckpt 目录")
    ap.add_argument("--root", default=None, help="含 seed*/ckpt 的根目录（递归处理每个 seed）")
    ap.add_argument("--pattern", default="ckpt_*_ac_iter*.pth",
                    help="ckpt 文件名 glob（默认按 AC 命名 `ckpt_*_ac_iter*.pth`；"
                         "PPO ckpt 用 `ckpt_*_ppo_iter*.pth`）")
    ap.add_argument("--iter", type=int, default=None, help="只评估指定 iter 的 ckpt（不择优）")
    ap.add_argument("--json", default=None, help="可选：把汇总写到该 JSON")
    args = ap.parse_args()

    if args.root:
        summary = eval_root(args.root, iter_filter=args.iter, pattern=args.pattern)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump({s: {"best_iter": v["best_iter"]} for s, v in summary.items()},
                          f, ensure_ascii=False, indent=2)
            print(f"\n[save] {args.json}")
    elif args.ckpt_dir:
        res_iter = eval_dir(args.ckpt_dir, pattern=args.pattern)
        if not res_iter:
            raise SystemExit("(无 AC 编号 ckpt)")
        it = args.iter if args.iter in res_iter else max(res_iter, key=lambda k: score(res_iter[k]))
        res = res_iter[it]
        print(f"** 最优 = iter{it:04d} **")
        for l, q in res["levels"].items():
            jm = q["J_mean"]
            print(f"  [{l}] 回合长 {min(q['lens']):.0f}/{np.mean(q['lens']):.0f}/{max(q['lens']):.0f} | "
                  f"稳杆 {q['surv']}/{q['K']} | J_ach={jm if jm is None else round(jm, 2)}±"
                  f"{q['J_std'] if q['J_std'] is None else round(q['J_std'], 2)}")
    else:
        raise SystemExit("--ckpt-dir 或 --root 二选一")


if __name__ == "__main__":
    main()
