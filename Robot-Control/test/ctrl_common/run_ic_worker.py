# -*- coding: utf-8 -*-
"""阶段 2–4 并行执行的**单任务子进程**：读一个任务 JSON → 跑一次仿真 → 把结果 pickle 到文件。

    python run_ic_worker.py <spec.json> <out.pkl>

⚠️ **为什么不用 `multiprocessing`**（2026-09-19 实测，见书仓库日志尝试 8）：
本机沙箱**禁止创建命名管道**——`mp.Pool` 在 `_setup_queues` → `Connection.Pipe` → `CreateFile`
处直接报 `PermissionError: [WinError 5] 拒绝访问`。故改用「子进程 + **文件重定向**」：
`stdio` 指向日志文件、任务与结果走 JSON/pickle 文件，**全程不涉及管道**，
在默认 `workspace-write` 沙箱下即可运行（实测 3 个并发子进程各自 1.75 s 的工作在 1.97 s 内完成）。
`multiprocessing` 的另一个坑是 Windows 只有 spawn、需可 pickle 的闭包；文件交接天然绕开。

任务规格（spec）由 `verify_lib.make_spec` 生成，字段见该函数；本文件只负责「读 → 跑 → 写」。
"""
import json
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RC = os.path.abspath(os.path.join(_HERE, "..", ".."))              # Robot-Control/
for _p in (_RC, os.path.join(_RC, "code"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import verify_lib as vl                                             # noqa: E402


def main(argv):
    if len(argv) < 3:
        print("用法：run_ic_worker.py <spec.json> <out.pkl>")
        return 2
    spec_path, out_path = argv[1], argv[2]
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    res = vl.run_task(spec)          # 异常直接抛出并写进日志（父进程据退出码判定失败）
    with open(out_path, "wb") as f:
        pickle.dump(res, f, protocol=4)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
