# -*- coding: utf-8 -*-
"""对一组 AC checkpoint 做统一评估（eval_centering 口径），并按「存活」选每种子最好一份。

用途：AC 正式训练（`--ckpt-interval 100`）落下编号 ckpt 后，逐 seed 跑全部编号 ckpt，
按 L1/L2 存活优先、L3 次之、L1 J_ach 更小者取胜，挑出该 seed 的「最好 checkpoint」，
供门禁判定与书文本引用。

统一口径与 `eval/bench.py`/`eval_centering.py` 完全一致（同一批 seed=42 初值、`max_steps=1000`、
`J_ach = Σ(xᵀQx+uᵀRu)`、存活 = 全程不越界 且 末 500 步 `|θ|≤2°` 且 `|x|≤0.5m`）。

用法（cwd=CartPole）：
    E:/Anaconda3/envs/py311-gym/python.exe -X utf8 src-rl/eval_ac_ckpts.py \
        --ckpt-dir res/s_ac_formal_500/seed0/ckpt
    或（递归处理某目录下所有 seedN/ckpt）：
        ... eval_ac_ckpts.py --root res/s_ac_formal_500
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_THIS = os.path.abspath(os.path.dirname(__file__))
for _p in (_THIS, os.path.join(_THIS, "..", "src"), os.path.join(_THIS, "..", "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_centering  # noqa: E402


def rank_key(res):
    """按生存优先、J_ach 次之给一次评估打分（越小越好）。

    主判据看 L1/L2 全 5/5，故用 lexicographic：
    (-L1_surv, -L2_surv, -L3_surv, L1_J, L2_J, L3_J)。
    """
    lv = res["levels"]
    surv = [lv[l]["surv"] for l in ["L1", "L2", "L3"]]
    js = [lv[l]["J_mean"] if lv[l]["J_mean"] is not None else 1e6 for l in ["L1", "L2", "L3"]]
    return tuple([-s for s in surv] + js)


def eval_dir(ckpt_dir):
    """评估 `ckpt_dir` 下所有 `ckpt_*ac_iter*.pth`，返回 (best_path, {path: res})。"""
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*_ac_iter*.pth")))
    if not paths:
        # 兼容：可能没有 iter 编号文件（默认 latest）
        paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")))
    results = {}
    for p in paths:
        print(f"  [eval] {os.path.basename(p)}", flush=True)
        results[p] = eval_centering.eval_ckpt(p)
    best = min(results, key=lambda p: rank_key(results[p])) if results else None
    return best, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=None, help="单个 seed 的 ckpt 目录")
    ap.add_argument("--root", default=None, help="含 seed*/ckpt 的根目录（递归处理每个 seed）")
    ap.add_argument("--json", default=None, help="可选：把汇总写到该 JSON")
    args = ap.parse_args()

    targets = []
    if args.ckpt_dir:
        targets.append(args.ckpt_dir)
    elif args.root:
        for sd in sorted(glob.glob(os.path.join(args.root, "seed*"))):
            cd = os.path.join(sd, "ckpt")
            if os.path.isdir(cd):
                targets.append(cd)
    else:
        raise SystemExit("--ckpt-dir 或 --root 二选一")

    summary = {}
    for cd in targets:
        seed = os.path.basename(os.path.dirname(cd))
        print(f"\n==== {seed} ({cd}) ====", flush=True)
        best, results = eval_dir(cd)
        summary[seed] = {"ckpt_dir": cd, "best": best, "results": {
            os.path.basename(p): r for p, r in results.items()}}
        if best is None:
            print("  (无 ckpt)")
            continue
        print(f"  >>> BEST = {os.path.basename(best)}", flush=True)
        print(eval_centering_fmt(results[best]))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
        print(f"\n[save] {args.json}")


def eval_centering_fmt(res):
    lines = ["  level   surv    len(mean)   J_ach          |x|(mean/max)  |θ|max"]
    for lv, r in res["levels"].items():
        Jm = "  — " if r["J_mean"] is None else f"{r['J_mean']:6.2f}"
        Js = "—" if r["J_std"] is None else f"{r['J_std']:.2f}"
        lx_m = "  — " if r["last_x_mean"] is None else f"{r['last_x_mean']:6.2f}"
        lx_x = "  — " if r["last_x_max"] is None else f"{r['last_x_max']:6.2f}"
        lt_m = "  — " if r["last_theta_max"] is None else f"{r['last_theta_max']:6.2f}"
        lm = float(np.mean(r["lens"])) if r["lens"] else float("nan")
        lines.append(
            f"  {lv}  {r['surv']}/{r['K']}    {lm:8.1f}     "
            f"{Jm}±{Js}     {lx_m}/{lx_x}   {lt_m}")
    n = res.get("nominal", {})
    if n.get("len") is not None:
        def fnum(v):
            return "  — " if v is None else f"{v:6.2f}"
        lines.append(
            f"  NOM   {n['survived']}  {n['len']:8.1f}     "
            f"reason={n.get('reason')}  |x|(m/mx)={fnum(n.get('last_x_mean'))}/{fnum(n.get('last_x_max'))} "
            f"|θ|max={fnum(n.get('last_theta_max'))}")
    return "\n".join(lines)


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return str(o)


if __name__ == "__main__":
    main()
