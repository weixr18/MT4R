# -*- coding: utf-8 -*-
"""把 `probe_fit_dt*.json` **重新嵌回**阶段 3 归档的 `fit_dt_probe` 键（不动其它任何字段）。

**为什么需要它**（2026-09-20 发现，见书仓库日志尝试 13）：
`run_stage3.main` 的 `fit_dt_probe` 字段并不是本次运行算出来的，而是由 `run_stage3._load_probe(out_dir)`
**从磁盘上已存在的 `res/6r_stage3/probe_fit_dt*.json` 读进来的**。于是：

- 只重跑 `scripts/probe_fit_dt.py`（刷新那两个文件）**不会**更新 `result.json` 里的内嵌副本；
- 而 `scripts/summarize_stage3.py` 的探针表恰恰是从**内嵌副本**（`d["fit_dt_probe"]`）打印的
  ⇒ 结果会是「`summary.md` 印旧数、独立探针文件是新数」的自相矛盾归档。

本脚本用**与 `run_stage3._load_probe` 完全相同的逻辑**重读探针文件并替换该键，
在归档里追加一条 `post_run_refreshes` 记录（时间戳 + 旧/新 `wall_time_s`），并默认先备份
`result.json.probe-refresh.bak`，使这次「事后回填」本身可追溯。

用法（`py311-gym`，任意 cwd；只读写 JSON，秒级）：

    python Robot-Control/scripts/refresh_archive_probe.py                 # 就地刷新 res/6r_stage3/result.json
    python Robot-Control/scripts/refresh_archive_probe.py --dry-run       # 只比对、不写
"""
import argparse
import datetime
import io
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RC = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_RC, "test", "ctrl_common"))

import run_stage3 as rs3                                             # noqa: E402


def _stamp(probe):
    """探针 dict 的可比对摘要：每个文件的 `T_s` / `wall_time_s` / 逐行最差 ξ̂ 相对偏差。"""
    out = {}
    for name, pr in (probe or {}).items():
        rows = {}
        for row in pr.get("rows", []):
            errs = [c["xi_rel_err"] for c in row.get("components", [])
                    if c.get("xi_rel_err") is not None]
            rows["%s@dt=%.0e" % (row["kind"], row["dt_s"])] = (
                None if not errs else 100.0 * max(errs))
        out[name] = {"T_s": pr.get("T_s"), "wall_time_s": pr.get("wall_time_s"), "worst_xi_pct": rows}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="把探针结果重新嵌回阶段 3 归档的 fit_dt_probe 键")
    ap.add_argument("--json", default=os.path.join(_RC, "res", "6r_stage3", "result.json"))
    ap.add_argument("--dir", default=None, help="探针文件所在目录（默认 = result.json 所在目录）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args(argv)

    probe_dir = args.dir or os.path.dirname(args.json)
    if not os.path.exists(args.json):
        print("找不到归档：%s" % args.json)
        return 2
    with io.open(args.json, encoding="utf-8") as f:
        d = json.load(f)

    old = d.get("fit_dt_probe")
    new = rs3._load_probe(probe_dir)          # ← 与 run_stage3 完全同一逻辑
    if not new:
        print("✗ %s 下没有可用的 probe_fit_dt*.json，拒绝写入（归档保持原样）" % probe_dir)
        return 2

    old_s, new_s = _stamp(old), _stamp(new)
    print("归档：%s" % args.json)
    print("探针目录：%s" % probe_dir)
    print("文件：%s" % ", ".join(sorted(new)))
    changed = json.dumps(old_s, sort_keys=True, ensure_ascii=False) != \
        json.dumps(new_s, sort_keys=True, ensure_ascii=False)
    for name in sorted(new):
        o, n = old_s.get(name), new_s[name]
        print("  [%s] T_s: %s → %s；wall: %s → %s" % (
            name,
            "无" if o is None else o["T_s"], n["T_s"],
            "无" if o is None else ("%.1f s" % (o["wall_time_s"] or 0)),
            "%.1f s" % (n["wall_time_s"] or 0)))
        if o is None:
            continue
        for key in sorted(n["worst_xi_pct"]):
            ov, nv = o["worst_xi_pct"].get(key), n["worst_xi_pct"][key]
            print("      %-22s 最差 ξ̂ 偏差 %s → %s" % (
                key,
                "n/a" if ov is None else "%.4f%%" % ov,
                "n/a" if nv is None else "%.4f%%" % nv))
    print("内容是否变化：%s" % ("是" if changed else "否（幂等）"))

    if args.dry_run:
        print("--dry-run：未写入。")
        return 0

    d["fit_dt_probe"] = new
    eff = d.get("effect_size") if isinstance(d.get("effect_size"), dict) else None
    d.setdefault("post_run_refreshes", []).append({
        "at": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/refresh_archive_probe.py",
        "keys": ["fit_dt_probe"],
        "reason": "run_stage3._load_probe 在写归档时把 probe_fit_dt*.json 读入 result.json；"
                  "事后重跑探针不会更新内嵌副本，而 summarize_stage3 是从内嵌副本打印的。",
        "probe_dir": os.path.relpath(probe_dir, _RC).replace("\\", "/"),
        "before": old_s, "after": new_s,
        "effect_size_note": None if eff is None else "未改动 effect_size",
    })
    if not args.no_backup:
        bak = args.json + ".probe-refresh.bak"
        shutil.copyfile(args.json, bak)
        print("已备份：%s" % bak)
    with io.open(args.json, "w", encoding="utf-8", newline="") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print("✓ 已刷新 %s 的 fit_dt_probe（并追加 post_run_refreshes 记录）" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
