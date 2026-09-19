# -*- coding: utf-8 -*-
"""3-level 统一评估：PID / 连续LQR / 离散LQR / 无约束调节MPC / DDP / iLQR 在三个初始状态 level 下的仿真结果。
（REINFORCE 为可选：设置环境变量 `REINFORCE_CKPT` 指向训练好的 `.pth` 时一并纳入评估。）

对应书文档 `1-MN4R/docs/todo/cartpole_unfinished.md`（书↔代码对应与遗留事项）与 `1-MN4R/docs/cartpole-training/`（训练计划/日志）。
- 复用 `CartPole/src/` 下现有控制器类（importlib 导入，保证单一代码来源），不改任何现有文件。
- 不改 `env_cartpole.py`；通过直接写 `env.state` 来显式设定每个初值。
- 统一评估口径：
    - 成本 `J_ach = Σ_{k=0}^{T-1}(x_k^T Q x_k + u_k^T R u_k)`，Q=diag(1,1,100,1)、R=[1.0]，
      `x_0` 为初始状态也计入，终态 `x_T` 不计入；`u_k` 为实际施加（裁剪后）的力。
    - 存活 = 全程不越界（|x|≤5m 且 |θ|≤80° 至满 1000 步） 且 成功镇定
      （最后 H=500 个状态连续满足 |θ|≤2° 且 |x|≤0.5m）。
- DDP/iLQR（`src/5_ddp_ilqr.py` 的 `DDPILQRController`）为滚动时域非线性最优控制：
  设计权重 `Q=Q_eval`、`R=R_eval`、终端 `S=P_DARE`，时域 `N=20`。
- 运行时（cwd = `CartPole/eval/`）：
    E:\\Anaconda3\\envs\\py311-gym\\python.exe bench.py
- 输出：
    - 18 张图：`1-MN4R/imgs/cartpole/`（pid/lqr/lqr_disc/ddp/ilqr）与 `1-MN4R/imgs/mpc/`（mpc_reg），
      命名 `<algo>_L<level>.png`。
    - 数值：`CartPole/res/bench/bench_results.json`（各算法各 level 的 J_mean/J_std/s/K）。
- 目录可用环境变量 `FIG_DIR_CART` / `FIG_DIR_MPC` / `RES_DIR` 覆盖。
"""
import os
import sys
import json
import importlib

import numpy as np
from scipy.linalg import solve_discrete_are

# ======================================================================
# 0. 路径与环境
# ======================================================================
# 把 ../src 加进 sys.path，复用现有控制器类（单一来源）
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 把 ../src-rl 加进 sys.path，复用 RL 公共组件（load_policy_ckpt / ReinforcePolicyController）——仅当
# 设置了环境变量 REINFORCE_CKPT（指向训练好的 .pth）时惰性导入，避免普通评估引入 torch 依赖。
SRL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src-rl"))
if SRL_DIR not in sys.path:
    sys.path.insert(0, SRL_DIR)

from env_cartpole import CartPoleCustomEnv, CONFIG

# 现有四个控制器模块（文件名以数字开头，无法用普通 import，用 importlib）
pid_mod = importlib.import_module("1_pid")
lqr_mod = importlib.import_module("2_lqr")
lqr_disc_mod = importlib.import_module("3_lqr_discrete")
mpc_mod = importlib.import_module("4_mpc")
ddp_ilqr_mod = importlib.import_module("5_ddp_ilqr")

# 注意：上述模块顶层各自执行了 np.random.seed(42)，因此导入完成后，
# 务必在抽取初值前重新 np.random.seed(42)，以固定本脚本自己的随机流。

# 工作区根（4-MT4R-github/CartPole/eval/ -> ../../../ = 2024_MN4R；书仓库是其下的 1-MN4R/）
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FIG_DIR_CART = os.environ.get(
    "FIG_DIR_CART",
    os.path.join(WORKSPACE, "1-MN4R", "imgs", "cartpole"))
FIG_DIR_MPC = os.environ.get(
    "FIG_DIR_MPC",
    os.path.join(WORKSPACE, "1-MN4R", "imgs", "mpc"))
RES_DIR = os.environ.get(
    "RES_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "res", "bench")))

# ======================================================================
# 1. 统一评估口径（plan §1）
# ======================================================================
Q_EVAL = np.diag([1.0, 1.0, 100.0, 1.0])   # 对应 (x, x_dot, theta, theta_dot)
R_EVAL = np.array([[1.0]])

# 成功镇定判据（plan §1.2）
THETA_BAND = 2.0 * np.pi / 180.0   # |θ| ≤ 2°
X_BAND = 0.5                        # |x| ≤ 0.5 m
H = 500                             # 末尾 500 步（10 s）连续满足稳态带

K = 5                               # 每个 level 的回合数
SEED = 42

# 无约束 MPC 调节的设计权重（沿用 4_mpc.py 的设定，见 plan §4.1）
MPC_Q = np.diag([5.0, 0.1, 10.0, 1.0])
MPC_R = np.array([[0.01]])
MPC_S = 20.0 * MPC_Q
MPC_NP = 50

# DDP / iLQR 的设计权重（沿用统一评估口径 Q_eval/R_eval，终端 S = 无穷时域 DARE 解 P；
# 见 plan-ddp-ilqr.md §2.2/§5——与 27.3 离散 LQR 同权重、S=P 使有限时域代价逼近 LQ 最优）
ILQR_Q = np.diag([1.0, 1.0, 100.0, 1.0])   # = Q_EVAL
ILQR_R = np.array([[1.0]])                  # = R_EVAL
ILQR_N = 20                                 # 有限时域（0.4 s）
ILQR_SOLVER_ITER = 20                       # 每次滚动求解的外层轨迹迭代上限
_A_d, _B_d = lqr_disc_mod.discretize(*lqr_disc_mod.calc_sys_mat(), CONFIG["dt"])
ILQR_S = solve_discrete_are(_A_d, _B_d, ILQR_Q, ILQR_R)   # S = P_DARE

# ======================================================================
# 2. 三个 level 的定义（plan §2）：只改初始状态范围
# ======================================================================
# theta_amp: |θ| 的幅度区间（°）；theta_dot/x/x_dot: ± 区间（°/s、m、m/s）
LEVELS = {
    "L1": dict(theta_amp=(0.0, 3.0), theta_dot=1.0, x=0.3, x_dot=0.3),
    "L2": dict(theta_amp=(3.0, 10.0), theta_dot=3.0, x=0.8, x_dot=0.5),
    "L3": dict(theta_amp=(20.0, 35.0), theta_dot=10.0, x=1.5, x_dot=1.0),
}


# ======================================================================
# 3. 初值生成（plan §3.1）：一次生成，四算法共用
# ======================================================================
def gen_init_states(level, k, seed=None):
    """生成某个 level 的 k 个初始状态，形状 (k, 4)，顺序为 [x, x_dot, theta, theta_dot]。

    - θ 幅度：先在本 level 的幅度区间均匀取 |θ|（°），再随机取 ± 号。
    - θ̇（°/s）、x（m）、ẋ（m/s）：各自在其 ± 区间内均匀随机。
    - 内部全部换算为 SI（rad、rad/s）。
    - `seed` 非空时先固定随机流；批次内对不同 level 连续调用时不再重设 seed，
      以得到彼此独立、且可复现的一组初值（plan：这批初值只生成一次）。
    """
    if seed is not None:
        np.random.seed(seed)
    spec = LEVELS[level]
    states = np.zeros((k, 4))
    for i in range(k):
        theta_amp = np.random.uniform(*spec["theta_amp"])      # 幅度（°）
        sign = np.random.choice([-1.0, 1.0])
        theta = sign * theta_amp * np.pi / 180.0               # rad
        theta_dot = np.random.uniform(-spec["theta_dot"], spec["theta_dot"]) * np.pi / 180.0
        x = np.random.uniform(-spec["x"], spec["x"])
        x_dot = np.random.uniform(-spec["x_dot"], spec["x_dot"])
        states[i] = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
    return states


# ======================================================================
# 4. 闭环 rollout（plan §3.2、§1.1）
# ======================================================================
def rollout(controller, env, x0):
    """在 gym 非线性环境上闭环跑一回合。

    返回 (x_seq, u_seq, survived, reason)：
    - x_seq: shape (T+1, 4)，x_seq[0] = x0（初始状态），x_seq[k] = 执行 k 步之后的状态；
    - u_seq: shape (T,)，第 k 步实际施加（裁剪到 ±F_max）的力；
    - survived: 存活 = 全程不越界 且 成功镇定；
    - reason: 终止原因描述。
    """
    env.state = np.array(x0, dtype=np.float32)
    env.steps = 0
    if hasattr(controller, "reset"):
        controller.reset()

    x_seq = [np.array(x0, dtype=np.float64)]
    u_seq = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = controller.compute_action(x_seq[-1])
        next_state, _, terminated, truncated, info = env.step(action)
        # 使用 env 实际施加（裁剪后）的力，与 plan §1.1 的 "已裁剪" 一致
        u_seq.append(float(info["F_applied"]))
        x_seq.append(np.array(next_state, dtype=np.float64))

    x_seq = np.array(x_seq)
    u_seq = np.array(u_seq)

    # 终止原因
    x, _, theta, _ = x_seq[-1]
    if abs(x) > CONFIG["x_threshold"]:
        reason = "x越界"
    elif abs(theta) > CONFIG["theta_threshold"]:
        reason = "θ越界"
    else:
        reason = "满步"

    # 存活判定
    max_steps_reached = env.steps >= CONFIG["max_steps"]
    out_of_bounds = not max_steps_reached          # 未满步即终止 => 越界
    if len(x_seq) >= H:
        last = x_seq[-H:]
        settled = bool(
            np.all(np.abs(last[:, 2]) <= THETA_BAND) and
            np.all(np.abs(last[:, 0]) <= X_BAND))
    else:
        settled = False
    survived = (not out_of_bounds) and settled
    return x_seq, u_seq, survived, reason


# ======================================================================
# 5. 成本 J_ach（plan §1.1）
# ======================================================================
def cost_quadratic(x_seq, u_seq, Q, R):
    """J_ach = Σ_{k=0}^{T-1}(x_k^T Q x_k + u_k^T R u_k)。

    x_seq[0]（初始状态）也计入；终态 x_seq[T] 不计入；u 为标量力。
    """
    T = len(x_seq) - 1
    cost = 0.0
    for k in range(T):
        xk = x_seq[k]
        cost += float(xk @ Q @ xk)
        uk = float(np.atleast_1d(u_seq[k])[0])
        cost += uk * uk * float(np.squeeze(R))
    return cost


# ======================================================================
# 6. 可视化（plan §3.2、沿用现有 5 面板 + Agg 保存）
# ======================================================================
def visualize_episode(x_seq, u_seq, path, title):
    """第 1 回合 5 面板时间序列（x、ẋ、θ、θ̇、F），风格同 src/1_pid.py、4_mpc.py。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states = x_seq[1:]   # T×4：每步之后的状态，与原脚本 states.append(next_state) 一致
    actions = u_seq      # T
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title, fontsize=16)
    t = np.arange(len(states)) * CONFIG["dt"]   # 秒（dt = 0.02 s）
    theta_deg = states[:, 2] * 180.0 / np.pi    # °
    theta_dot_deg = states[:, 3] * 180.0 / np.pi  # °/s
    specs = [("x (Position)", "C0", states[:, 0]),
             (r"$\dot{x}$ (Velocity)", "C1", states[:, 1]),
             (r"$\theta$ (°)", "C2", theta_deg),
             (r"$\dot{\theta}$ (°/s)", "C3", theta_dot_deg)]
    for i, (label, color, y) in enumerate(specs):
        axs[i].plot(t, y, label=label, color=color)
        axs[i].set_ylabel(label)
        axs[i].legend()
    axs[4].plot(t, actions, label="F (Force)", color="C4")
    axs[4].set_ylabel("F (Force)")
    axs[4].set_xlabel("Time [s]")
    axs[4].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 分辨率减半（dpi 150 -> 75，像素长宽各变为原来一半）；再量化为 256 色调色板 PNG，
    # 在肉眼几乎无差的前提下大幅减小文件体积（对比无损 RGB PNG 约省 2/3）。
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=75, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB").quantize(
        colors=256, method=Image.Quantize.MEDIANCUT)
    img.save(path, format="PNG", optimize=True)
    buf.close()
    plt.close(fig)


# ======================================================================
# 7. 逐 level 评估（plan §4.1 run_level）
# ======================================================================
def run_level(controller, level, x0_list, name, env, fig_dir):
    """跑 K 回合，返回该 level 的结果 dict 与第 1 回合的轨迹。

    - 失败回合 J_ach 记 +∞，不计入 mean/std，但计入存活率分母；
    - 某 level 全失败 => J_mean/J_std 记为 None（打印/JSON 用 "—"）。
    """
    Js = []
    s = 0
    first_x_seq = first_u_seq = None
    first_survived = None
    reasons = []
    for i, x0 in enumerate(x0_list):
        x_seq, u_seq, survived, reason = rollout(controller, env, x0)
        reasons.append(reason)
        if survived:
            s += 1
            Js.append(cost_quadratic(x_seq, u_seq, Q_EVAL, R_EVAL))
        if i == 0:
            first_survived = bool(survived)
            first_x_seq, first_u_seq = x_seq, u_seq

    # 第 1 回合可视化（无论是否存活都画；失败时在图标题中注出）
    vis_path = os.path.join(fig_dir, f"{name}_L{level[-1]}.png")
    title = f"{name} - Level {level} - Episode 1 (first init state)"
    if first_survived is False:
        title += " (regulation failed)"
    visualize_episode(first_x_seq, first_u_seq, vis_path, title)

    J_mean = float(np.mean(Js)) if Js else None
    J_std = float(np.std(Js)) if Js else None
    return dict(
        level=level, name=name,
        J_mean=J_mean, J_std=J_std, s=s, K=len(x0_list),
        fig=os.path.abspath(vis_path),
        reason_counts={r: reasons.count(r) for r in sorted(set(reasons))},
        first_survived=first_survived,
        first_x0=[float(v) for v in x0_list[0]],
    )


# ======================================================================
# 8. 控制器工厂（plan §4.1）
# ======================================================================
def _make_controllers():
    """返回 {算法名: 控制器实例}。构建一次、跨回合复用（避免 LQR/MPC 构造函数重复打印矩阵）；
    DDP/iLQR 带 warm-start 内部状态，但 `rollout()` 每回合调用 `controller.reset()` 已清除。

    REINFORCE 策略为**可选**：设置环境变量 `REINFORCE_CKPT`（指向 `src-rl/1_reinforce.py` 保存的
    `.pth`）时一并纳入统一评估；未设置或路径不存在则跳过。评估取均值动作 μ（确定性策略，
    标准做法），与 PID/LQR 同口径跑同一批 seed=42 初值。
    """
    controllers = {
        "pid": pid_mod.PIDBalanceController(CONFIG),
        "lqr": lqr_mod.LQRContinuousController(Q_EVAL, R_EVAL, CONFIG["F_max"]),
        "lqr_disc": lqr_disc_mod.LQRDiscreteController(Q_EVAL, R_EVAL, CONFIG["F_max"], CONFIG["dt"]),
        "mpc_reg": mpc_mod.MPCController(MPC_Q, MPC_R, MPC_S, MPC_NP, CONFIG["F_max"], CONFIG["dt"]),
        "ddp": ddp_ilqr_mod.DDPILQRController(
            ILQR_Q, ILQR_R, ILQR_S, ILQR_N, CONFIG["F_max"], CONFIG["dt"],
            mode="ddp", solver_iter=ILQR_SOLVER_ITER, warm_start=True),
        "ilqr": ddp_ilqr_mod.DDPILQRController(
            ILQR_Q, ILQR_R, ILQR_S, ILQR_N, CONFIG["F_max"], CONFIG["dt"],
            mode="ilqr", solver_iter=ILQR_SOLVER_ITER, warm_start=True),
    }
    ckpt = os.environ.get("REINFORCE_CKPT", "").strip()
    if ckpt and os.path.exists(ckpt):
        from rl_common import load_policy_ckpt, ReinforcePolicyController
        policy, obs_rms, meta = load_policy_ckpt(
            ckpt, obs_dim=4, act_dim=1, hidden_sizes=(64, 64),
            device="cpu", init_log_std=0.0)
        controllers["reinforce"] = ReinforcePolicyController(
            policy, obs_rms, device="cpu", mean=True)
        print(f"[REINFORCE] 已加载 checkpoint: {ckpt}")
        print(f"            meta={meta}")
    elif ckpt:
        print(f"[REINFORCE] 警告：checkpoint 不存在 {ckpt}，跳过 RL 评估。")
    return controllers


def _fig_dir_for(name):
    return FIG_DIR_MPC if name == "mpc_reg" else FIG_DIR_CART


# ======================================================================
# 9. 主流程
# ======================================================================
def fmt_js(v):
    return "—" if v is None else f"{v:.2f}"


def run_all():
    env = CartPoleCustomEnv(CONFIG)
    controllers = _make_controllers()

    # 初值只生成一次，四算法共用（seed 固定一次，视为一批）
    np.random.seed(SEED)
    init_by_level = {lv: gen_init_states(lv, K) for lv in LEVELS}
    for lv, xs in init_by_level.items():
        print(f"  Level {lv} 初值（首列）: θ={xs[0,2]*180/np.pi:+.2f}°, "
              f"θ̇={xs[0,3]*180/np.pi:+.2f}°/s, x={xs[0,0]:+.3f}m, ẋ={xs[0,1]:+.3f}m/s")

    results = {}
    for name, controller in controllers.items():
        results[name] = {}
        print(f"\n{'='*72}\n[算法] {name} (K={K})\n{'='*72}")
        for lv in LEVELS:
            x0_list = init_by_level[lv][:K]
            res = run_level(controller, lv, x0_list, name, env, _fig_dir_for(name))
            results[name][lv] = res
            jm = fmt_js(res["J_mean"])
            js = fmt_js(res["J_std"])
            print(f"  {lv}: J_ach = {jm} ± {js}, 存活率 = {res['s']}/{res['K']}, "
                  f"终止={res['reason_counts']}, 图={os.path.basename(res['fig'])}")

    # 打印每算法 2×3 表
    print_table(results)

    # 保存 JSON
    os.makedirs(RES_DIR, exist_ok=True)
    json_path = os.path.join(RES_DIR, "bench_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "K": K,
                   "Q_eval": Q_EVAL.tolist(), "R_eval": R_EVAL.tolist(),
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n数值已保存: {os.path.abspath(json_path)}")
    return results


def print_table(results):
    levels = list(LEVELS.keys())
    for name in results:
        print(f"\n{name} 的 2×3 表（行 = J_ach / 存活率，列 = L1–L3）：")
        row_j = []
        row_s = []
        for lv in levels:
            r = results[name][lv]
            row_j.append("—" if r["J_mean"] is None else f"{r['J_mean']:.2f}±{r['J_std']:.2f}")
            row_s.append(f"{r['s']}/{r['K']}")
        print(f"  J_ach   : " + "  ".join(f"{v:>12}" for v in row_j))
        print(f"  存活率  : " + "  ".join(f"{v:>12}" for v in row_s))


if __name__ == "__main__":
    run_all()
