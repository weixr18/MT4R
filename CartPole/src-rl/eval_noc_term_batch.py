# -*- coding: utf-8 -*-
"""按书 §28.1「稳杆为主」口径，对一组 REINFORCE checkpoint 做批量评估并逐种子择优。

对每个 `seed*/ckpt/ckpt_cartpole_iterNNNN.pth` 跑 `eval_noc_term.eval_ckpt`（稳杆为主口径），
按「L1+L2+L3 稳杆存活总数 → 5/5 档数 → L1 J_ach(越小越好) → 平均回合长」择优，取该种子的最优 ckpt。

用法（cwd=CartPole）：
    python src-rl/eval_noc_term_batch.py --root src-rl/res/s_rf_retrain_l3_noc_term
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_THIS = os.path.abspath(os.path.dirname(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

import eval_noc_term  # noqa: E402


def _num(ckpt):
    m = re.search(r"iter(\d+)", os.path.basename(ckpt))
    return int(m.group(1)) if m else -1


def eval_dir(ckpt_dir):
    results = {}
    for ckpt in sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_cartpole_iter*.pth")),
                       key=_num):
        results[_num(ckpt)] = eval_noc_term.eval_ckpt(ckpt)
    return results


def score(res):
    surv = sum(r["surv"] for r in res["levels"].values())
    fives = sum(1 for r in res["levels"].values() if r["surv"] == r["K"])
    l1_j = res["levels"]["L1"]["J_mean"]
    l1_j = l1_j if l1_j is not None else float("inf")
    mean_len = float(np.mean([np.mean(r["lens"]) for r in res["levels"].values()]))
    # 越大越好（存活优先）
    return (surv, fives, -l1_j, mean_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="含 seed*/ckpt 的根目录")
    ap.add_argument("--iter", type=int, default=None,
                    help="只评估指定 iter 的 ckpt（不择优）")
    args = ap.parse_args()

    summary = {}
    for sd in sorted(glob.glob(os.path.join(args.root, "seed*"))):
        seed = os.path.basename(sd)
        ckpt_dir = os.path.join(sd, "ckpt")
        res_iter = eval_dir(ckpt_dir) if os.path.isdir(ckpt_dir) else {}
        if not res_iter:
            print(f"==== {seed}: no iter ckpts ====")
            continue
        if args.iter is not None:
            it = args.iter if args.iter in res_iter else max(res_iter)
            best_iter = it
            res = res_iter[it]
        else:
            best_iter = max(res_iter, key=lambda k: score(res_iter[k]))
            res = res_iter[best_iter]
        # 打印全部 iter 的概览 + 最优
        print(f"\n==== {seed} ====")
        for it, r in sorted(res_iter.items()):
            surv = sum(x["surv"] for x in r["levels"].values())
            j = r["levels"]["L1"]["J_mean"]
            print(f"  iter{it:04d}: surv_total={surv}/15  L1_J={j if j is None else round(j,2)}")
        summary[seed] = {"best_iter": best_iter, "res": res}
        r = res
        print(f"  ** 最优 = iter{best_iter:04d} **")
        for lv, q in r["levels"].items():
            jm = q["J_mean"]
            print(f"    [{lv}] 回合长 {min(q['lens']):.0f}/{np.mean(q['lens']):.0f}/{max(q['lens']):.0f} | "
                  f"稳杆 {q['surv']}/{q['K']} | J_ach={jm if jm is None else round(jm,2)}±"
                  f"{q['J_std'] if q['J_std'] is None else round(q['J_std'],2)} | "
                  f"末500|x|均/最={None if q['last_x_mean'] is None else round(q['last_x_mean'],2)}/"
                  f"{None if q['last_x_max'] is None else round(q['last_x_max'],2)} | "
                  f"|θ|max={None if q['last_theta_max'] is None else round(q['last_theta_max'],2)}")

    out_path = os.path.join(args.root, "eval_noc_term_best.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({s: {"best_iter": v["best_iter"]} for s, v in summary.items()},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
