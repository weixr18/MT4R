# -*- coding: utf-8 -*-
"""阶段 2–4 共用的**实验驱动**：初值采样、批量仿真、指标汇总、判据判定、出图。

放在 `test/ctrl_common/` 而不是 `code/`：`code/` 是「模型 + 仿真层 + 控制器 + 度量」这套**可复用库**，
本文件是**验证编排**（跑多少组初值、扫哪些增益、判据阈值怎么定），属 `test/` 的职责。

阶段 2 的四条判据（计划 §5.2.5）在这里落地，逐条对应关系：

| 判据 | 本文件中的键 | 实测口径 |
|---|---|---|
| 1 四个算法在 2R 上跑通 | `no_nan` | 无 NaN/Inf、无异常退出（奇异/奇异求解失败都不吞） |
| 2 收敛性 | `convergence` | 固定种子 20 组初值，末态 `‖q̃‖_∞ < 1e-3 rad` |
| 3 力矩量级合理 | `torque_limit` | 全程 `‖u‖_∞` 不超过用例力矩上限（不饱和） |
| 4 端到端自洽 | `kp_monotone` | 4a 标称无扰动：稳态误差恒为 0（理论预测）；4b 常值未补偿扰动：∝K_P⁻¹（斜率 −1） |

⚠️ **判据 4 的修订（阶段 2 实测，见书仓库日志尝试 8）**：原文只写「稳态误差随 K_P 增大而单调下降」，
但在**标称用例**（plant 与控制器模型同源、无摩擦、无外力）下，重力补偿 PD 的稳态误差
**理论上恒为 0**（平衡条件 `K_Pq̃ = 0`），与 K_P 无关——该判据在原文口径下**退化**。
修订为 4a（证实「恒为 0」这一理论预测本身）+ 4b（加常值未补偿扰动 `τ_e` 后，`K_P⁻¹` 规律可被实测）。
阈值一个字未动：单调下降与量级关系仍在 4b 上检验。

## 并行执行（2026-09-19 加，见书仓库日志尝试 8）

仿真层是**单线程 CPU 密集**的：一次 2R 闭环每毫秒仿真要算 4 次 `robot_dyn`（RK4 四级；控制器那次
被 `RobotCase.dyn` 的单条 memo 吸收），实测 **~4.0 ms 墙钟 / ms 仿真**——8 核机器上只用了 1 核。
故初值批与 `K_P` 扫描点改为**多进程并行**，`workers` 默认 `min(6, CPU-2)`（可用环境变量
`RC_WORKERS` 或 `--workers N` 覆盖，1 = 串行）。

⚠️ **并行不走 `multiprocessing`**：本机沙箱禁止创建命名管道，`mp.Pool` 实测直接报
`PermissionError: [WinError 5]`。这里用 `subprocess.Popen(..., stdout=文件)` + 任务/结果走文件
（`run_ic_worker.py`）——**全程不涉及管道**，默认沙箱即可用（见 `run_task`/`run_specs`）。
串行与并行**共用同一条 `run_task` 路径**，故两种模式的数值结果完全一致（可用来交叉验证）。

运行环境：`E:\\Anaconda3\\envs\\py311-gym\\python.exe`（加 `PYTHONUTF8=1`）。
"""
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RC = os.path.abspath(os.path.join(_HERE, "..", ".."))          # Robot-Control/
for _p in (_RC, os.path.join(_RC, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import code_metrics as met                                        # noqa: E402
import code_models as cm                                          # noqa: E402
import code_sim as sim                                            # noqa: E402
import code_traj as traj                                          # noqa: E402
from code_ctrl_common import pd_gains                             # noqa: E402
from code_ctrl_js import ctrl_js_invdyn, ctrl_js_pd               # noqa: E402
from code_ctrl_os import ctrl_os_invdyn, ctrl_os_pd               # noqa: E402

# ---- 全局实验口径（固定，报告里逐条记录）---------------------------------
SEED = 0
N_IC = 20                 # 判据 2 要求的初值组数（≥20）
T_REG = 3.5               # 调节用例仿真时长 (s)
DT = 1e-3                 # RK4 步长 (s)（计划 §5.2.1 默认 1 ms）
HC = 1e-3                 # 控制周期 (s)，零阶保持
TOL_CONV = 1e-3           # 判据 2：末态 ‖q̃‖_∞ 阈值 (rad)
DQ_MAX, DQDOT_MAX = 0.15, 0.15   # 初值盒：q0 = q_d + U(-d,d)，q̇0 = U(-d,d)
TOL_STEADY_ZERO = 1e-9    # 判据 4a：标称无扰动时的稳态误差上限（≈ 机器精度量级）
T_KP = 4.0                # 判据 4：K_P 扫描的单次仿真时长 (s)
TAU_E_SWEEP = np.array([0.0, 0.5, 0.5])   # 判据 4b 的常值未补偿扰动力矩 (N·m)
T_TRACK = 1.5             # 跟踪冒烟用例时长 (s)
Q_TRACK_GOAL_2R = np.array([0.0, 0.9, 1.0])   # 2R 跟踪终点（任务空间直线段全程可达，见日志尝试 8）

# 四个控制器：`wn/xi` 为**本阶段固定**的二阶指标（书中未给定取值，按 §"参数选取"参数化并逐条记录）。
# ⚠️ JS 与 OS 的 `K_P` 单位/含义不同（见 `code_ctrl_common.pd_gains` 的说明）：
#    OS 重力补偿 PD 未抵消任务惯量，实际带宽约为 0.22·ω_n，故取明显更大的任务空间增益。
CTRLS = {
    "js_pd": dict(fn=ctrl_js_pd, name="JS 重力补偿 PD", wn=8.0, xi=1.0, gain_dim="N",
                  kp_wn=(4.0, 8.0, 16.0)),
    "js_invdyn": dict(fn=ctrl_js_invdyn, name="JS 逆动力学 PD", wn=8.0, xi=1.0, gain_dim="N",
                      kp_wn=(2.0, 4.0, 8.0)),
    "os_pd": dict(fn=ctrl_os_pd, name="OS 重力补偿 PD", wn=12.0, xi=1.0, gain_dim="m",
                  kp_wn=(8.0, 16.0, 32.0)),
    "os_invdyn": dict(fn=ctrl_os_invdyn, name="OS 逆动力学 PD", wn=6.0, xi=1.0, gain_dim="m",
                      kp_wn=(2.0, 4.0, 8.0)),
}
CTRL_ORDER = ["js_pd", "js_invdyn", "os_pd", "os_invdyn"]

# ---- 并行执行的默认并发数 -------------------------------------------------
# 本机 8 核（较老），`Robot-Control/AGENTS.md` 要求**同时启动的后台任务不超过 8 个**；
# 本驱动一次运行占 `WORKERS` 个子进程，留 2 核给系统/其它任务。
DEFAULT_WORKERS = 6
_WORKER = os.path.join(_HERE, "run_ic_worker.py")


def default_workers():
    """默认并发数：`min(6, CPU-2)`，可用环境变量 `RC_WORKERS` 覆盖（1 = 串行）。"""
    env = os.environ.get("RC_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cpu = os.cpu_count() or 2
    return max(1, min(DEFAULT_WORKERS, cpu - 2))


# ---------------------------------------------------------------------------
# 任务规格与「一条任务」的执行（**与并行/串行无关**，两条路径共用同一实现）
# ---------------------------------------------------------------------------
def make_ref_spec(kind, case, q_d):
    """调节参考的规格：JS 给 `q_d`；OS 给 `x_d = fk(q_d)`（**不经过逆解**，红线见 `code_traj`）。"""
    if kind.startswith("js"):
        return {"type": "js_const", "q_d": [float(v) for v in q_d]}
    return {"type": "os_const", "x_d": [float(v) for v in case.fk(q_d)]}


def build_ref(spec):
    """规格 → `sim.simulate` 能用的参考（常值字典或 `t -> 字典`）。"""
    t = spec["type"]
    if t == "js_const":
        return traj.js_const_ref(np.array(spec["q_d"], dtype=float))
    if t == "os_const":
        return traj.os_const_ref(np.array(spec["x_d"], dtype=float))
    if t == "js_traj":
        return traj.js_traj_ref(np.array(spec["q_start"], dtype=float),
                                np.array(spec["q_goal"], dtype=float), spec["T"])
    if t == "os_traj":
        return traj.os_traj_ref(np.array(spec["x_start"], dtype=float),
                                np.array(spec["x_goal"], dtype=float), spec["T"])
    raise ValueError("未知参考规格 %r" % (t,))


def make_spec(kind, case_name, gains, ref_spec, q0, qdot0, T, dt, h_c,
              tau_e=None, label="", plant=None, friction=None, noise=None):
    """构造一条任务规格（**只含 JSON 可序列化的量**，故可以交给子进程）。

    `plant` / `friction` / `noise`（阶段 4 新增，默认 `None` ＝ 阶段 2/3 的标称情形）：

    - `plant`：**被控对象**的参数失配规格（`code_models.perturb_params` 的键）；
      给定时控制器仍用**标称模型**（`run_task` 里另建一个 `plant=None` 的 case）；
    - `friction`：`{"viscous": […], "coulomb": […], "eps": …}` → 由
      `code_sim.make_friction_tau_e` 变成 `tau_e` 钩子（**控制器不补偿**，`F_f=None`）；
    - `noise`：测量噪声规格（`code_sim.simulate` 的 `noise`，见其文档）。
    """
    return {
        "kind": kind, "case": case_name,
        "gains": {"K_P": np.asarray(gains["K_P"]).tolist(),
                  "K_D": np.asarray(gains["K_D"]).tolist()},
        "ref": ref_spec,
        "q0": [float(v) for v in q0], "qdot0": [float(v) for v in qdot0],
        "T": float(T), "dt": float(dt), "h_c": float(h_c),
        "tau_e": None if tau_e is None else [float(v) for v in tau_e],
        "label": label,
        "plant": None if not plant else {k: float(v) for k, v in plant.items()},
        "friction": None if not friction else {
            "viscous": [float(v) for v in friction.get("viscous", [])],
            "coulomb": [float(v) for v in friction.get("coulomb", [])],
            "eps": float(friction.get("eps", 1e-3))},
        "noise": None if not noise else {
            "q_sigma": float(noise.get("q_sigma", 0.0) or 0.0),
            "qd_sigma": float(noise.get("qd_sigma", 0.0) or 0.0),
            "vel_from_pos": bool(noise.get("vel_from_pos", False)),
            "lpf_tau": (None if noise.get("lpf_tau", None) is None
                        else float(noise["lpf_tau"])),
            "seed": int(noise.get("seed", 0) or 0)},
    }


def run_task(spec):
    """跑一条任务规格，返回 `code_sim.simulate` 的记录（串行与子进程共用这一条路径）。

    ⚠️ **被控对象与控制器模型可以不同源**（阶段 4 的核心手段）：`spec["plant"]` 只作用于
    传给 `simulate` 的 case；控制器的 `case` 永远是标称模型。`plant` 为 `None` 时两者是
    **同一个对象**（保留 `RobotCase.dyn` 的单条 memo，阶段 2/3 的耗时口径不变）。
    """
    plant_spec = spec.get("plant")
    if plant_spec:
        case_plant = cm.make_case(spec["case"], plant_spec)     # 被控对象（失配）
        case_ctrl = cm.make_case(spec["case"])                  # 控制器模型（标称）
    else:
        case_plant = case_ctrl = cm.make_case(spec["case"])
    gains = {k: np.array(v, dtype=float) for k, v in spec["gains"].items()}
    ctrl = bind(spec["kind"], case_ctrl, gains)
    ref = build_ref(spec["ref"])
    tau_e = None
    if spec.get("friction"):
        fr = spec["friction"]
        tau_e = sim.make_friction_tau_e(fr.get("viscous"), fr.get("coulomb"),
                                        eps=fr.get("eps", 1e-3), N=case_plant.N)
    elif spec.get("tau_e") is not None:
        te = np.array(spec["tau_e"], dtype=float)
        tau_e = lambda t, q, qd: te                                 # noqa: E731
    return sim.simulate(case_plant, ctrl, ref, np.array(spec["q0"], dtype=float),
                        np.array(spec["qdot0"], dtype=float), spec["T"],
                        dt=spec["dt"], h_c=spec["h_c"], tau_e=tau_e,
                        noise=spec.get("noise"),
                        progress=False, label=spec.get("label", ""))


def _sweep_stale_tmp(root, max_age_s=3600):
    """清掉 `_workers_tmp/` 下**超过 1 小时**的残留目录（进程被强杀时 `finally` 来不及跑）。"""
    try:
        now = time.time()
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isdir(p) and now - os.path.getmtime(p) > max_age_s:
                shutil.rmtree(p, ignore_errors=True)
    except OSError:
        pass


def run_specs(specs, workers=None, tag="", verbose=True):
    """按顺序返回每条规格的仿真记录；`workers > 1` 时用**子进程池**并行。

    ⚠️ **为什么用子进程 + 文件而不是 `multiprocessing`**：本机沙箱禁止创建命名管道，
    `mp.Pool` 实测直接报 `PermissionError: [WinError 5]`（见 `run_ic_worker.py` 与日志尝试 8）。
    这里用 `subprocess.Popen(..., stdout=文件)`：**没有任何管道**，默认沙箱下即可并行；
    任务走 spec JSON、结果走 pickle 文件，父进程按完成顺序回收。

    子进程非零退出或缺结果文件时**直接抛错并附日志尾部**——不静默降级为「跳过」，
    否则会重演「假通过」（见日志尝试 4/8 的教训）。
    """
    workers = default_workers() if workers is None else max(1, int(workers))
    if workers <= 1 or len(specs) <= 1:
        return [run_task(s) for s in specs]

    # ⚠️ **必须把每个子进程的 BLAS 线程数钉成 1**（2026-09-19 实测）：
    # 本用例的线代规模都很小（2×2/6×6），但 MKL/OpenMP 默认仍会按核数开线程；
    # `workers` 个子进程各自再开 N 个线程 → 8 核上严重超订，
    # 实测 6 路并行时**单条任务从 14.3 s 涨到 ~50 s**，净加速只有 1.7×（而不是 ~5×）。
    child_env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        child_env[var] = "1"

    tmp = os.path.join(_RC, "res", "_workers_tmp", "%s-%d" % (tag or "task", os.getpid()))
    os.makedirs(tmp, exist_ok=True)
    _sweep_stale_tmp(os.path.dirname(tmp))
    results = [None] * len(specs)
    running = []
    nxt = 0
    t0 = time.perf_counter()
    try:
        while nxt < len(specs) or running:
            while nxt < len(specs) and len(running) < workers:
                i = nxt
                nxt += 1
                sp = os.path.join(tmp, "spec_%04d.json" % i)
                op = os.path.join(tmp, "out_%04d.pkl" % i)
                lp = os.path.join(tmp, "log_%04d.txt" % i)
                with open(sp, "w", encoding="utf-8") as f:
                    json.dump(specs[i], f)
                lf = open(lp, "w", encoding="utf-8")
                pr = subprocess.Popen([sys.executable, _WORKER, sp, op],
                                      stdout=lf, stderr=subprocess.STDOUT, cwd=tmp,
                                      env=child_env)
                running.append([pr, i, op, lp, lf])
            time.sleep(0.15)
            for item in list(running):
                pr, i, op, lp, lf = item
                if pr.poll() is None:
                    continue
                lf.close()
                running.remove(item)
                if pr.returncode != 0 or not os.path.exists(op):
                    tail = ""
                    try:
                        with open(lp, encoding="utf-8") as f:
                            tail = "".join(f.readlines()[-15:])
                    except OSError:
                        pass
                    raise RuntimeError("并行子进程失败（task %d，%s，退出码 %s）：\n%s"
                                       % (i, specs[i].get("label", ""), pr.returncode, tail))
                with open(op, "rb") as f:
                    results[i] = pickle.load(f)
                if verbose:
                    print("      [并行 %s] 完成 %d/%d  已用 %.0f s"
                          % (tag or "task", sum(r is not None for r in results),
                             len(specs), time.perf_counter() - t0), flush=True)
    finally:
        for pr, _i, _op, _lp, lf in running:
            try:
                pr.kill()
                lf.close()
            except OSError:
                pass
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            os.rmdir(os.path.dirname(tmp))          # 父目录空了就一并删掉，别留空壳
        except OSError:
            pass
    return results


def run_specs_timed(specs, workers=None, tag="", verbose=True):
    """`run_specs` 的包装：额外返回**每条任务各自的墙钟时间**（并行时是「占用一个 worker 的时长」）。"""
    t0 = time.perf_counter()
    recs = run_specs(specs, workers=workers, tag=tag, verbose=verbose)
    total = time.perf_counter() - t0
    return recs, total


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _downsample(arr, n_max=250):
    """按等间隔抽样把长序列压到 `n_max` 点（存档用，图仍由完整序列生成）。"""
    a = np.asarray(arr)
    if a.shape[0] <= n_max:
        return a
    idx = np.unique(np.linspace(0, a.shape[0] - 1, n_max).round().astype(int))
    return a[idx]


def _trace_json(res):
    """把一次仿真的记录压成**可 JSON 序列化**的降采样轨迹（供归档复现与报告作图）。

    只存 `t/q/q̇/u/q̃` 五个量（`x_e`、`x̃` 可由 `q` 与 `params.x_d` 复算），
    完整精度记录不落盘——需要重算就按 `res/README.md` 的记录口径重跑该用例。
    """
    return {
        "t": _downsample(res["t"]).tolist(),
        "q": _downsample(res["q"]).tolist(),
        "qdot": _downsample(res["qdot"]).tolist(),
        "u": _downsample(res["u"]).tolist(),
        "q_tilde": _downsample(res["q_tilde"]).tolist(),
        "dt": res["dt"], "h_c": res["h_c"], "n_samples": int(res["t"].size),
        "u_abs_max": res["u_abs_max"], "nan": bool(res["nan"]), "error": res["error"],
    }


def _gains_for(cfg, case, wn=None, xi=None):
    n_g = case.m if cfg["gain_dim"] == "m" else case.N
    return pd_gains(cfg["wn"] if wn is None else wn,
                    cfg["xi"] if xi is None else xi, n_g)


def bind(kind, case, gains, F_f=None):
    """把一个控制器**绑定**成 `ctrl(state, ref) -> u`（仿真层只需要这个形式）。"""
    fn = CTRLS[kind]["fn"]
    return lambda state, ref: fn(state, ref, gains, case, F_f)


def sample_ics(case, q_d, n_ic=N_IC, seed=SEED, dq_max=DQ_MAX, dqdot_max=DQDOT_MAX):
    """固定种子采样初值：`q0 = q_d + U(-d,d)`、`q̇0 = U(-d,d)`（各关节独立）。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_ic):
        q0 = np.array(q_d, dtype=float)
        q0[1:] += rng.uniform(-dq_max, dq_max, size=case.N)
        qd0 = np.zeros(case.N + 1)
        qd0[1:] = rng.uniform(-dqdot_max, dqdot_max, size=case.N)
        out.append((q0, qd0))
    return out


def reg_ref(kind, case, q_d):
    """调节参考（**已构建**，供单次运行使用）：等价于 `build_ref(make_ref_spec(...))`。"""
    return build_ref(make_ref_spec(kind, case, q_d))


def steady_err(m, kind):
    """判据 4 的「稳态误差」口径：JS 取 `‖q̃‖_∞`，OS 取任务空间加权范数 `‖x̃_w‖`。"""
    return m["q_tilde_inf"]["steady"] if kind.startswith("js") else m["x_weighted"]["steady"]


def _fit_slope(xs, ys):
    """log-log 最小二乘斜率。"""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[m]), np.log(ys[m]), 1)[0])


# ---------------------------------------------------------------------------
# 判据 1–3：20 组初值上的调节
# ---------------------------------------------------------------------------
def run_ic_batch(kind, case, q_d, ref_spec, gains, ics, T=T_REG, dt=DT, h_c=HC,
                 tau_e=None, verbose=True, tag="", workers=None):
    """在给定的初值列表上跑一遍闭环，返回 `(指标列表, 记录列表)`；多初值**并行**。

    ⚠️ **OS 用例的关节误差由本函数在评测层补上**：OS 的 `ref` 只含 `x_d`（红线：不把 `q_d` 交给
    OS 控制器），仿真记录里的 `q̃` 全为 0——若直接拿来判「收敛」会**假通过**。
    故这里用**实验者选定的目标位形** `q_d`（`x_d = fk(q_d)`，不是逆解结果）重算 `q̃ = q_d − q`，
    并覆盖记录里的 `q_tilde`/`q_d`。`q_d` 只用于**评测**，控制器始终只看 `x_d`。
    """
    specs = [make_spec(kind, case.name, gains, ref_spec, q0, qd0, T, dt, h_c, tau_e,
                       label="%s-ic%02d" % (kind, k))
             for k, (q0, qd0) in enumerate(ics)]
    t0 = time.perf_counter()
    recs = run_specs(specs, workers=workers, tag="%s-ic" % kind, verbose=verbose)
    wall = time.perf_counter() - t0
    rows = []
    for k, res in enumerate(recs):
        if q_d is not None:
            qd = np.asarray(q_d, dtype=float)
            res["q_tilde"] = qd[None, :] - res["q"]
            res["q_d"] = np.tile(qd, (res["q"].shape[0], 1))
        m = met.summarize(res, case, ell=cm.ELL_DEFAULT,
                          has_task=not kind.startswith("js"))
        rows.append(m)
        if verbose:
            print("      [%s] ic%02d/%02d  ‖q̃‖_∞ 末值=%.3e  峰值=%.3e  ‖u‖_∞=%.3e  σ_min=%.3e  %s"
                  % (tag or kind, k + 1, len(ics), m["q_tilde_inf"]["final"],
                     m["q_tilde_inf"]["peak"], m["control"]["u_inf"],
                     m["numerics"]["sigma_min_min"],
                     "NaN!" if (m["nan"] or m["error"]) else "ok  "), flush=True)
    if verbose:
        print("      [%s] %d 组初值合计墙钟 %.1f s（%d 路并行，单组平均 %.2f s）"
              % (tag or kind, len(ics), wall, max(1, workers or default_workers()),
                 wall / max(1, len(ics))), flush=True)
    return rows, recs


# ---------------------------------------------------------------------------
# 判据 4：K_P 扫描（4a 标称 / 4b 常值未补偿扰动）
# ---------------------------------------------------------------------------
def kp_sweep(kind, case, q_d, ref_spec, wn_list, T=T_KP, dt=DT, h_c=HC, tau_e=None,
             workers=None):
    """扫描 K_P（= ω_n²，ξ 固定 1），返回每一步的稳态误差与拟合斜率；各 K_P 点**并行**。

    初值取目标位形本身（`q0 = q_d`、`q̇0 = 0`），误差**只由扰动 `τ_e` 激发**——
    这样「稳态误差 vs K_P」的关系才干净。
    """
    specs, gains_list = [], []
    q_d = np.asarray(q_d, dtype=float) if q_d is not None else None
    for wn in wn_list:
        gains = _gains_for(CTRLS[kind], case, wn=wn)
        gains_list.append(gains)
        q0 = q_d if q_d is not None else np.zeros(case.N + 1)
        specs.append(make_spec(kind, case.name, gains, ref_spec, q0,
                               np.zeros(case.N + 1), T, dt, h_c, tau_e,
                               label="%s-kp%.0f" % (kind, wn)))
    recs = run_specs(specs, workers=workers, tag="%s-kp" % kind, verbose=False)
    errs, cost, kps = [], [], []
    for wn, gains, res in zip(wn_list, gains_list, recs):
        if q_d is not None:
            res["q_tilde"] = q_d[None, :] - res["q"]
            res["q_d"] = np.tile(q_d, (res["q"].shape[0], 1))
        m = met.summarize(res, case, ell=cm.ELL_DEFAULT,
                          has_task=not kind.startswith("js"))
        kps.append(float(np.diag(gains["K_P"]).mean()))
        errs.append(steady_err(m, kind))
        cost.append(m["control"]["u_inf"])
    return {"K_P": kps, "err": errs, "u_inf": cost,
            "slope": _fit_slope(kps, errs),
            "monotone": bool(all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)))}


# ---------------------------------------------------------------------------
# 跟踪冒烟用例（跑通 `q̈_d`/`ẍ_d` 前馈链路；正式判据在阶段 3）
# ---------------------------------------------------------------------------
def tracking_smoke(kind, case, q_start, q_goal, T=T_TRACK, dt=DT, h_c=HC):
    """在**参考所在的空间**生成一条 minimum-jerk 轨迹并跟踪（JS 用关节空间、OS 用操作空间）。

    返回 `(res, 结论字典)`。OS 的关节误差用**末时刻逆解**（`case.ik`，**仅评测参照**）
    给出的「到解流形的距离」，并附逆解残差作为守卫（残差大说明逆解本身没收敛，该指标不可信）。
    """
    gains = _gains_for(CTRLS[kind], case)
    ctrl = bind(kind, case, gains)
    js = kind.startswith("js")
    if js:
        ref = traj.js_traj_ref(q_start, q_goal, T)
    else:
        ref = traj.os_traj_ref(case.fk(q_start), case.fk(q_goal), T)
    res = sim.simulate(case, ctrl, ref, np.array(q_start, float), np.zeros(case.N + 1),
                       T, dt=dt, h_c=h_c, progress=False, label="%s-track" % kind)
    m = met.summarize(res, case, ell=cm.ELL_DEFAULT,
                          has_task=not kind.startswith("js"))
    out = {"nan": m["nan"], "error": m["error"], "u_inf": m["control"]["u_inf"],
           "q_tilde_inf_final": None, "q_manifold_dist_final": None, "ik_residual": None}
    if js:
        res["q_tilde"] = res["q_d"] - res["q"]
        m = met.summarize(res, case, ell=cm.ELL_DEFAULT,
                          has_task=not kind.startswith("js"))
        out["q_tilde_inf_final"] = m["q_tilde_inf"]["final"]
    else:
        x_d_end = np.asarray(ref(T)["x_d"], dtype=float)
        q_eval = case.ik(x_d_end, res["q"][-1])
        out["x_pos_final"] = m["x_pos"]["final"]
        out["x_pos_peak"] = m["x_pos"]["peak"]
        if q_eval is not None:
            out["q_manifold_dist_final"] = float(np.max(np.abs(q_eval - res["q"][-1])))
            out["ik_residual"] = float(np.max(np.abs(case.fk(q_eval) - x_d_end)))
            res["q_tilde"] = q_eval[None, :] - res["q"]
    return res, out


# ---------------------------------------------------------------------------
# 诊断（阶段 3.1 的主判据在此只作**诊断量**报告：欠阻尼自由响应的 (ξ̂, ω̂_n)）
# ---------------------------------------------------------------------------
def underdamped_diag(kind, case, q_d, wn=6.0, xi=0.3, dq=0.05, T=3.0, dt=DT, h_c=HC):
    """给一个小初值偏差，辨识 `(ξ̂, ω̂_n)`；**仅逆动力学控制**有严格二阶基准，其余返回不适用。

    JS 逆动力学：关节误差 `q̃_j(t)` 满足 `q̈̃ + K_Dq̇̃ + K_Pq̃ = 0`（逐关节解耦）→ 拟合关节 1；
    OS 逆动力学：任务空间误差满足 `ẍ̃ + K_Dẋ̃ + K_Px̃ = 0` → 拟合 `x̃` 分量 1。
    """
    if not kind.endswith("invdyn"):
        return {"applicable": False,
                "note": "重力补偿 PD 未抵消 B/任务惯量，误差动态非线性，对数衰减率法无严格基准"}
    gains = _gains_for(CTRLS[kind], case, wn=wn, xi=xi)
    ctrl = bind(kind, case, gains)
    q0 = np.array(q_d, float)
    q0[1] += dq
    ref = reg_ref(kind, case, q_d)
    res = sim.simulate(case, ctrl, ref, q0, np.zeros(case.N + 1), T, dt=dt, h_c=h_c,
                       progress=False, label="%s-2nd" % kind)
    sig = res["q_tilde"][:, 1] if kind.startswith("js") else res["x_tilde"][:, 0]
    ident = met.identify_second_order(res["t"], sig)
    ident.update({"applicable": True, "wn_set": float(wn), "xi_set": float(xi),
                  "signal": "q̃_1(t)" if kind.startswith("js") else "x̃_1(t)"})
    if ident["xi_hat"] is not None:
        ident["wn_rel_err"] = abs(ident["wn_hat"] - wn) / wn
        ident["xi_rel_err"] = abs(ident["xi_hat"] - xi) / xi
    return ident


# ---------------------------------------------------------------------------
# 单个算法的阶段 2 全流程
# ---------------------------------------------------------------------------
def stage2_checks(kind, case_name="2R", n_ic=N_IC, T=T_REG, dt=DT, h_c=HC, seed=SEED,
                  quick=False, verbose=True, workers=None):
    """跑一个算法在阶段 2 的全部检查，返回结构化结论（含降采样轨迹，可直接 JSON 化）。

    `workers`：初值与 `K_P` 扫描点的**并行路数**（`None` → `default_workers()`，1 → 串行）。
    """
    cfg = CTRLS[kind]
    case = cm.make_case(case_name)
    q_d = cm.Q_D_REG_2R if case_name == "2R" else cm.Q_D_REG_6R
    gains = _gains_for(cfg, case)
    ref_spec = make_ref_spec(kind, case, q_d)
    if quick:
        n_ic, T = min(n_ic, 3), 1.0
        wn_list, T_nom, T_dis, T_tr, T_2nd = cfg["kp_wn"][:2], 0.8, 1.0, 0.3, 0.5
        print("      ⚠️ 冒烟模式（--quick）：初值 3 组、T 缩短、K_P 只取 2 点——"
              "**判据结论无意义**，只验证「链路能跑通」。")
    else:
        wn_list, T_nom, T_dis, T_tr, T_2nd = cfg["kp_wn"], 2.0, T_KP, T_TRACK, 3.0
    ics = sample_ics(case, q_d, n_ic=n_ic, seed=seed)

    print("    ── %s @ %s：调节用例，%d 组初值，T=%.1f s，dt=%.0e，并行 %d 路 ──"
          % (cfg["name"], case_name, n_ic, T, dt, workers or default_workers()))
    rows, recs = run_ic_batch(kind, case, q_d, ref_spec, gains, ics, T=T, dt=dt, h_c=h_c,
                              workers=workers)
    keys = [(("q_tilde_inf", "final"), "q̃∞_末值"), (("q_tilde_inf", "peak"), "q̃∞_峰值"),
            (("q_tilde_inf", "rms"), "q̃∞_RMS"), (("settle_time_2pct_s",), "调节时间_2%"),
            (("control", "u_inf"), "‖u‖∞"), (("numerics", "sigma_min_min"), "min σ_min"),
            (("numerics", "cond_B_max"), "max cond(B)"), (("wall_time_s",), "单次墙钟")]
    if not kind.startswith("js"):
        keys += [(("x_weighted", "final"), "‖x̃_w‖_末值"), (("x_pos", "final"), "‖x̃_pos‖_末值")]
    table = met.aggregate(rows, keys)

    ok_nan = all((not r["nan"]) and r["error"] is None for r in rows)
    worst_conv = max(r["q_tilde_inf"]["final"] for r in rows)
    ok_conv = ok_nan and worst_conv < TOL_CONV
    worst_u = max(r["control"]["u_inf"] for r in rows)
    ok_torque = worst_u <= case.u_max
    print("      判据1 无 NaN/异常：%s    判据2 末态 max‖q̃‖∞=%.3e (<1e-3)：%s    "
          "判据3 max‖u‖∞=%.3e (≤%.0f)：%s"
          % (ok_nan, worst_conv, ok_conv, worst_u, case.u_max, ok_torque))

    # 判据 4（4a 标称 / 4b 常值未补偿扰动；两组的 K_P 点合起来并行跑）
    s_nom = kp_sweep(kind, case, q_d, ref_spec, wn_list, T=T_nom, dt=dt, h_c=h_c,
                     tau_e=None, workers=workers)
    s_dis = kp_sweep(kind, case, q_d, ref_spec, wn_list, T=T_dis, dt=dt, h_c=h_c,
                     tau_e=TAU_E_SWEEP, workers=workers)
    ok_4a = max(s_nom["err"]) < TOL_STEADY_ZERO
    ok_4b = s_dis["monotone"] and (-1.30 <= s_dis["slope"] <= -0.70)
    print("      判据4a 标称无扰动稳态误差 max=%.3e (<1e-9)：%s" % (max(s_nom["err"]), ok_4a))
    print("      判据4b 扰动下 K_P 扫描 err=%s  斜率=%.3f  单调=%s：%s"
          % (["%.2e" % e for e in s_dis["err"]], s_dis["slope"], s_dis["monotone"], ok_4b))

    # 跟踪冒烟（非判据）
    q_goal = Q_TRACK_GOAL_2R if case_name == "2R" else cm.Q_NOM_6R.copy()
    q_goal = np.array(q_goal, float)
    if case_name == "6R":
        q_goal[1] += 0.25
    tr_res, tr = tracking_smoke(kind, case, q_d, q_goal, T=T_tr)
    print("      跟踪冒烟（T=%.1f s）：末态 ‖q̃‖∞=%s  ‖x̃_pos‖=%s  ‖u‖∞=%.3e  NaN=%s"
          % (T_tr,
             "%.3e" % tr["q_tilde_inf_final"] if tr["q_tilde_inf_final"] is not None else "n/a",
             "%.3e" % tr.get("x_pos_final", float("nan")) if "x_pos_final" in tr else "n/a",
             tr["u_inf"], tr["nan"]))

    # 诊断：欠阻尼自由响应的 (ξ̂, ω̂_n)
    diag = underdamped_diag(kind, case, q_d, wn=(6.0 if kind.startswith("js") else 4.0), T=T_2nd)
    if diag.get("applicable") and diag.get("xi_hat") is not None:
        print("      诊断（仅记录，判据在阶段 3.1）：设定 ξ=%.2f, ω_n=%.1f → 辨识 ξ̂=%.4f, ω̂_n=%.4f"
              "（相对偏差 %.3f%% / %.3f%%）"
              % (diag["xi_set"], diag["wn_set"], diag["xi_hat"], diag["wn_hat"],
                 100 * diag["xi_rel_err"], 100 * diag["wn_rel_err"]))
    else:
        print("      诊断：%s" % diag.get("note", "不适用"))

    # 代表性轨迹：末态误差最大的一组（作图与归档用）
    i_worst = int(np.argmax([r["q_tilde_inf"]["final"] for r in rows]))
    # OS 附加：到「精确解流形」的距离（末态逆解，**仅评测参照**）+ 逆解残差守卫
    extra_conv = {}
    if not kind.startswith("js"):
        x_d = case.fk(q_d)
        q_eval = case.ik(x_d, recs[i_worst]["q"][-1])
        if q_eval is not None:
            extra_conv = {
                "manifold_dist_final_rad": float(np.max(np.abs(q_eval - recs[i_worst]["q"][-1]))),
                "ik_residual": float(np.max(np.abs(case.fk(q_eval) - x_d))),
                "manifold_note": "到解流形的距离 = 末态逆解 q* 与实测 q 的 ∞ 范数差（逆解只作评测参照）",
            }
    checks = {
        "no_nan": {"desc": "在 %s 上跑通：无 NaN/Inf、无异常退出" % case_name,
                   "threshold": "全部 %d 组运行 nan=False 且 error=None" % n_ic,
                   "measured": {"n_runs": len(rows),
                                "n_nan": int(sum(1 for r in rows if r["nan"])),
                                "n_error": int(sum(1 for r in rows if r["error"] is not None)),
                                "errors": [r["error"] for r in rows if r["error"] is not None]},
                   "passed": bool(ok_nan)},
        "convergence": {"desc": "固定种子 %d 组初值全部收敛（末态 ‖q̃‖_∞ < %g rad）" % (n_ic, TOL_CONV),
                        "threshold": "worst ‖q̃‖_∞(T) < %g rad" % TOL_CONV,
                        "measured": {"worst_final_q_tilde_inf": worst_conv,
                                     "mean_final_q_tilde_inf": float(np.mean(
                                         [r["q_tilde_inf"]["final"] for r in rows])),
                                     "seed": seed, "n_ic": n_ic, "T": T,
                                     "q0_box": "q_d + U(-%.2f,%.2f)，q̇0 = U(-%.2f,%.2f)"
                                               % (DQ_MAX, DQ_MAX, DQDOT_MAX, DQDOT_MAX),
                                     **extra_conv},
                        "passed": bool(ok_conv)},
        "torque_limit": {"desc": "控制力矩在用例上限内（不饱和）",
                         "threshold": "max‖u‖_∞ ≤ %.0f N·m" % case.u_max,
                         "measured": {"worst_u_inf": worst_u, "u_max": case.u_max},
                         "passed": bool(ok_torque)},
        "kp_monotone": {"desc": "端到端自洽：稳态误差随 K_P 增大而下降"
                                "（4a 标称无扰动：理论预测恒为 0；4b 常值未补偿扰动：∝K_P⁻¹）",
                        "threshold": "4a 全部稳态误差 < %g；4b 单调下降且 log-log 斜率 ∈[-1.30,-0.70]"
                                     % TOL_STEADY_ZERO,
                        "measured": {"nominal": s_nom, "disturbance": s_dis,
                                     "tau_e_Nm": [float(v) for v in TAU_E_SWEEP[1:]],
                                     "steady_err_def": "‖q̃‖_∞" if kind.startswith("js")
                                                       else "‖x̃_w‖（ℓ=%.1f m）" % cm.ELL_DEFAULT,
                                     "kp_wn": list(wn_list)},
                        "passed": bool(ok_4a and ok_4b)},
    }
    gains_json = {"wn": cfg["wn"], "xi": cfg["xi"],
                  "K_P": [float(v) for v in np.diag(gains["K_P"])],
                  "K_D": [float(v) for v in np.diag(gains["K_D"])],
                  "space": "任务空间 (N/m, N·s/m)" if cfg["gain_dim"] == "m" else "关节空间 (N·m/rad, N·m·s/rad)"}
    return {
        "kind": kind, "name": cfg["name"], "case": case_name,
        "gains": gains_json,
        "params": {"n_ic": n_ic, "T": T, "dt": dt, "h_c": h_c, "seed": seed,
                   "q_d": [float(v) for v in q_d], "x_d": [float(v) for v in case.fk(q_d)],
                   "u_max": case.u_max, "ell": cm.ELL_DEFAULT},
        "checks": checks,
        "table": table,
        "tracking_smoke": tr,
        "second_order_diag": diag,
        "worst_ic_index": i_worst,
        "traces": {
            "worst": _trace_json(recs[i_worst]),
            "ics": [{"q0": [float(v) for v in r["q"][0]], "qdot0": [float(v) for v in r["qdot"][0]]}
                    for r in recs],
        },
        "_raw": {"recs": recs, "rows": rows, "q_d": q_d, "s_nom": s_nom, "s_dis": s_dis,
                 "tr_res": tr_res, "case": case},
    }


# ---------------------------------------------------------------------------
# 出图（阶段 2 首批图：误差曲线、控制力矩、2R 相图、K_P 扫描）
# ---------------------------------------------------------------------------
def make_figures(results, fig_dir, tag="2r"):
    """由四个算法的 `stage2_checks` 结果生成计划 §2.5 的首批图，返回文件路径列表。"""
    os.makedirs(fig_dir, exist_ok=True)
    out = []
    curves_e, curves_u, labels = [], [], []
    for r in results:
        rec = r["_raw"]["recs"][r["worst_ic_index"]]
        e = np.max(np.abs(rec["q_tilde"][:, 1:]), axis=1)
        curves_e.append({"label": r["name"], "t": rec["t"], "e": e})
        curves_u.append({"label": r["name"], "t": rec["t"], "u": rec["u"]})
        labels.append(r["name"])
    out.append(met.plot_error_curves(
        curves_e, os.path.join(fig_dir, "err_curves_%s.png" % tag),
        title="2R 调节用例：最差初值的关节误差 $\\|\\tilde q\\|_\\infty$"
              "（%d 组初值中最差的一组）" % results[0]["params"]["n_ic"]))
    out.append(met.plot_torque_curves(
        curves_u, os.path.join(fig_dir, "torque_curves_%s.png" % tag),
        title="2R 调节用例：控制力矩（同一最差初值）"))
    js = next(r for r in results if r["kind"] == "js_pd")
    traces = [{"label": "初值轨迹", "q": rec["q"], "qdot": rec["qdot"]}
              for rec in js["_raw"]["recs"]]
    out.append(met.plot_phase_portrait(
        traces, os.path.join(fig_dir, "phase_portrait_%s.png" % tag), js["_raw"]["q_d"],
        title="2R 相图（吸引域）：JS 重力补偿 PD，%d 组初值" % len(traces)))
    rows = [{"label": r["name"], "K_P": r["_raw"]["s_dis"]["K_P"], "err": r["_raw"]["s_dis"]["err"]}
            for r in results]
    out.append(met.plot_kp_sweep(
        rows, os.path.join(fig_dir, "kp_sweep_%s.png" % tag),
        title="判据 4b：常值未补偿扰动下稳态误差 vs K_P"))
    return out


# ---------------------------------------------------------------------------
# 统一的结论打印 / 判定（四个 `test_*.py` 与 `run_stage2.py` 共用）
# ---------------------------------------------------------------------------
def all_passed(res, quick=False):
    """四条判据是否全中；`quick=True`（冒烟模式）只看判据 1——其余判据的时长/增益点被压缩，结论无意义。"""
    if quick:
        return bool(res["checks"]["no_nan"]["passed"])
    return all(bool(c["passed"]) for c in res["checks"].values())


def print_checks(res):
    for k, c in res["checks"].items():
        print("   [%s] %s（阈值：%s）" % ("PASS" if c["passed"] else "FAIL", k, c["threshold"]))

