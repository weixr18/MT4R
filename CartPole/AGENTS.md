# AGENTS.md

本文件为 AI 编码助手（Claude Code、Codex 等）在 `4-MT4R-github/CartPole/` 目录下工作时提供指引。

## 这是什么

本目录 `CartPole/` 是《Math Toolbox for Robotics》（机器人数学工具书，简称 **MN4R**）Part VIII（倒立摆）的**配套仿真与控制实验代码**：一阶倒立摆（CartPole）问题。

覆盖方法谱系：经典控制（随机策略 / PID / 连续与离散 LQR / 线性 MPC / DDP / iLQR）与强化学习（REINFORCE、原始 Actor-Critic、Actor-Critic GAE、原始 PPO、PPO-Penalty、PPO-Clip）。每种方法都有可运行的仿真与训练产物，结果统一归档在 `res/`（每个算法一个子目录，多数带一份 `README.md`，见「其他」）。书侧 Part VIII 章节是这些代码的「文字化」，Part VI（强化学习）的算法也在此落地。

**当前完成度**：Part VIII 的**经典控制**（问题建模、PID、连续/离散 LQR、无约束线性 MPC、DDP / iLQR）与**六类 RL 方法**（§28.1 REINFORCE、§28.2 五个 PPO/AC 小节、§28.3 汇总对比表 `tb:rl-summary`）**均已定稿成文**，六个汇总行均有实证值。遗留事项只剩「有约束线性 MPC」（求解器未调通）与规划中的「ADRC 控制倒立摆」，详见「已知问题」。

**在仓库中的位置**：本书配套代码仓库是 `4-MT4R-github`（远端 `main` → `git@github.com:weixr18/MT4R.git`）。`Codes/` 按书章节组织（Part I–VII），本目录 `CartPole/` 承载 Part VIII（倒立摆）的仿真与控制实验代码；仓库整体的约定（环境、`Codes/` 写法、提交与 PDF 发布物等）见仓库根 `../AGENTS.md`。书稿仓库在同工作区 `../../1-MN4R`（`D:\Projects\2024_MN4R\1-MN4R`），两者配套开发、相互独立。写/改 Part VIII 前先读书稿仓库 `docs/todo/cartpole_unfinished.md`（书 ↔ 代码的对应关系与完成状态以此为最终依据）。

## 环境与依赖

- **Python 环境：conda `py311-gym`**（Python 3.11.9，解释器 `E:\Anaconda3\envs\py311-gym\python.exe`）。本目录所有脚本一律使用该环境运行（下文「运行」中的 `python` 均指此环境）。Git Bash 下可用 `PY="/e/Anaconda3/envs/py311-gym/python.exe"` 后以 `"$PY" ...` 调用。
- `gym`、`pygame` —— 环境与可视化渲染
- `numpy`、`scipy` —— LQR / Riccati 方程求解（`solve_continuous_are` / `solve_discrete_are`）
- `torch` —— 强化学习网络与训练
- `matplotlib` —— 训练曲线与仿真图

## 目录结构

### `src/` —— 经典控制（使用当前版环境 `CartPoleCustom-v1`）

| 文件 | 内容 | 对应书章节 |
|---|---|---|
| `env_cartpole.py` | gym 环境 `CartPoleCustom-v1`：物理、存活型奖励、渲染 | chap8_1_1storder_1 建模 / 代码实现 |
| `0_random.py` | 随机策略 baseline | — |
| `1_pid.py` | 双 PID（角度主环 + 位置副环） | chap8_1_1storder_2 PID |
| `2_lqr.py` | 连续 LQR（`solve_continuous_are`），含能控性检查 | chap8_1_1storder_2 LQR（连续） |
| `3_lqr_discrete.py` | 离散 LQR（DARE / `solve_discrete_are`），精确 ZOH 离散化 | chap8_1_1storder_2 LQR（离散） |
| `4_mpc.py` | 线性 MPC：alg1 无约束调节 / alg2 无约束跟踪 / alg3 不等式约束调节（`mpc_LQR_necons`，**待调通**，默认跳过，`RUN_MPC_ALG3=1` 才运行）；复用本仓库 `Codes/chap1_3_optmz/code_4_qp.py` 的 `opt_qp_barrier` | chap3_4_mpc_1/2（chap8_1_1storder_3） |
| `5_ddp_ilqr.py` | iLQR / DDP 非线性最优控制（完整非线性动力学，模型 = 控制对象；`DDPILQRController` 供 benchmark 复用） | chap3_3_optmctrol_3（chap8_1_1storder_4 §27.5/§27.6） |

### `eval/` —— 统一评估

| 文件 | 内容 |
|---|---|
| `bench.py` | 在三个初始状态 level（L1/L2/L3，每 level 5 回合、固定 `seed=42`）下对 PID / 连续 LQR / 离散 LQR / 无约束调节 MPC / DDP / iLQR 做统一评估（`J_ach` + 存活率，`max_steps=1000`）；**可选**纳入 REINFORCE：设环境变量 `REINFORCE_CKPT` 指向训练好的 `.pth` 即一并评估。结果写 `res/bench/bench_results.json`（`RES_DIR` 可覆盖） |
| `bench_add_reinforce.py` | 只补充 REINFORCE 结果到既有 `res/bench/bench_results.json`（避免重跑经典控制器） |

> **`bench.py` 口径**（书侧经典控制与 `imgs/cartpole/` 时域图共用）：`J_ach = Σ(xᵀQx + uᵀRu)`，`Q=diag(1,1,100,1)`、`R=[1.0]`（`u` 取实际施加、裁剪后的力），`x_0` 计入、终态不计入；**存活 = 全程不越界**（`|x|≤5m` 且 `|θ|≤80°`，撑满 1000 步）**且成功镇定**（末 `H=500` 步连续满足 `|θ|≤2°` 且 `|x|≤0.5m`）。出图目录可用 `FIG_DIR_CART`（CartPole 类）与 `FIG_DIR_MPC`（`mpc_reg`）覆盖，默认写书稿仓库 `1-MN4R/imgs/`。

### `src-rl/` —— 强化学习（复用 `../src/env_cartpole.py`，单一环境单一奖励）

| 文件 | 内容 | 对应书章节 |
|---|---|---|
| `rl_common.py` | 公共组件：`set_seed`、`RunningMeanStd`（状态归一化）、`PolicyNetwork`（高斯头 μ/log σ）、`ValueNetwork`（AC/PPO critic）、`sample_trajectory`、`compute_returns`（γ=1）、`compute_advantages`（标量 baseline + return 归一化）、`compute_gae`（GAE(γ,λ)）、`load_policy_ckpt` + `ReinforcePolicyController`、`load/save_ac_checkpoint` | chap6_3_pgmeth_2 §21.6 / chap6_4_ppo_1 §22.2–22.3 |
| `1_reinforce.py` | 纯 REINFORCE 主循环（连续动作、高斯策略），`--env cartpole|pendulum` 一键切换；构造 policy 前固定种子（可复现），日志输出末位置 `fx|x|`；支持 `--reward-coefs` 等（⚠️ 默认系数与书不一致，见「关键约定」6） | chap8_2_RL_1 REINFORCE（书 §28.1 定稿） |
| `2_ac.py` | Actor-Critic 主循环，`--adv td|gae` 切换单步 TD 优势 / GAE 优势；critic 目标与非 GAE 优势共用同一自洽 δ_t | chap6_4_ppo_1 §22.2/§22.3（书 §28.2 原始AC、AC-GAE 定稿） |
| `3_ppo.py` | PPO 主循环，`--algo og|pen|clip` 一键切换三变体（固定 β 的 KL 惩罚 / 自适应 β 滞回 / ε 重要性比率裁剪）；与 `2_ac.py` 共用同一套采样 + GAE + critic + 高斯策略，额外做批内多轮更新与 `r_IS` 限幅（`log_prob_old` 采样时冻结、KL 用 k3 估计器 `mean(r_IS−1−ln r_IS)`、`log_ratio` 夹 ±20 防溢出），并监控 `mean_r_IS`/`mean_KL` | chap6_4_ppo_2 §22.4（书 §28.2 PPO 三小节定稿） |
| `eval_centering.py` | 评估训练策略的「位置回中」：复用 bench 同口径报告末 500 步 `|x|`、存活率、`J_ach`（**较严口径**，存活另含 `|x|≤0.5m`） | chap8_2_RL_1 |
| `eval_noc_term.py` | 按书 §28.1「稳杆为主」口径评估单个 ckpt（全程不越界 且 末 `H=500` 步 `|θ|≤2°`；`|x|≤0.5m` 仅作次级参考） | chap8_2_RL_1（书 §28.3） |
| `eval_noc_term_batch.py` | 对一组 REINFORCE ckpt 做稳杆为主批量评估，逐种子择优 | chap8_2_RL_1 |
| `eval_ac_ckpts.py` | 对一组 AC ckpt 做统一评估（`eval_centering` 较严口径），按「存活优先」逐 seed 择优 | — |
| `eval_ac_ckpts_noc_term.py` | 稳杆为主口径的批量评估 + 逐种子择优，AC / PPO 通用（`--pattern` 指定 ckpt 命名）——**书 §28.1/§28.2 的门禁判定与 `tb:rl-summary` 数字都出自它** | 书 §28.1/§28.2 统一评估入口 |
| `make_reinforce_figs.py`、`make_ac_figs.py`、`make_ac_gae_figs.py`、`make_ppo_og_figs.py`、`make_ppo_pen_figs.py`、`make_ppo_clip_figs.py` | 六种方法各自的书用图：训练曲线 `*_train_curve.png` + 各 level 第 1 回合时间序列 `*_L{1,2,3}.png`，默认写 `1-MN4R/imgs/cartpole/` | chap8_2_RL_1 / chap8_2_RL_2 |
| `viz_ac_seed0.py` | AC seed0 的人工检查可视化（训练曲线 + 三档 1000 步轨迹，标注 ±2° 稳杆带与末 500 步区间），写 `src-rl/res/s_ac_formal_500/seed0/viz/` | — |
| `diagnose.py` | 可视化已训练策略的轨迹 + 终止原因分布 | — |
| `diagnose_ac.py` | AC 机制验证：critic 判别力 + 优势方差（对照 REINFORCE 标量 baseline） | — |
| `play_reinforce.py` | 可视化播放已训练策略（`--ckpt` 指定 `.pth`，按 `--speed` 倍速实时渲染） | — |
| `res/` | 各次运行的**中间产物**（不纳入版本控制）：正式训练 `s_ac_formal_500/`、`s_ppo_{og,pen,clip}_formal*/`、`s_ppo_og_formal500_ne2/`、`s_rf_retrain_l3_noc_term/`（书 §28.1 REINFORCE）；筛参 `s_ppo_{og,pen,clip}_grid{A120,B300}/`、`s_ac_lr_scan/`。**书里引用的数字以 `../res/<算法>/` 的归档为准**，`src-rl/res/` 只是过程记录 | — |

> `run_noc_term_seed.ps1`：单种子重训启动器，产物写入 `src-rl/res/s_rf_retrain_l3_noc_term/`（书 §28.1 的 REINFORCE 结果即该目录产物）。训练历史见书 `docs/cartpole-training/REINFORCE-training-log.md`。
>
> **一次性驱动脚本已清理**：`run_stageA/B/C*`（原始 PPO）、`run_pen_*`、`run_clip_*`（PPO-Penalty / PPO-Clip）与 `run_ppo_og_archive.py`（把正式结果归档进 `res/PPO-og/`）——即「阶段 A 宽筛 → 阶段 B 复验 → 阶段 C 正式训练 → 评估 / 归档」的编排工具——已于任务完成后删除（同时清掉冒烟/诊断残留 `s_ppo_og_smoke*/`、`s_ac_diag_500/`、`s_ppo_og_scan/` 与空归档残留 `res/s_ppo_pen_smoke/`）。当时的命令与产物路径仍记录在书 `docs/cartpole-training/PPO-training-{plan,log}.md`，需要重跑筛参时按日志重建即可。

### 其他

- `test/` —— `test_riccati.py`（离散 DARE 验证）、`test_lqr_tracking.py`（LQR 跟踪 / 输入增量控制验证，对应书 Part III 最优控制的两个跟踪算法）
- `res/` —— **各算法的正式结果归档**（不纳入版本控制，但**是书里数字的依据，勿删**），每个算法一个子目录：`REINFORCE/`（**无** `README.md`，说明见书 `docs/cartpole-training/REINFORCE-training-log.md`）、`AC/`、`AC-GAE/`、`PPO-og/`、`PPO-pen/`、`PPO-clip/`（后五个各带一份 `README.md`）。注意 `AC/` 下的 `s_ac_gae_formal_500/`、`s_ac_gae_lambda/` 与 `AC-GAE/` 下的同名目录**逐字节重复**（书侧两处引用并存，删其一需同步改引用）。`eval/bench.py` 的产物 `res/bench/bench_results.json` 属**可再生成**的结果（跑一次 `bench.py` 即重建），当前本地未保留。
- `.claude/settings.json` —— 已配置的允许命令（当前仅两条 `1_reinforce.py` 的 Bash 白名单）；`.vscode/launch.json` —— 调试配置，默认运行 `src-rl/1_reinforce.py --env cartpole`（cwd 为 `src-rl/`）
- `CartPole.code-workspace` —— 本目录的 VS Code 工作区文件

## 关键约定

1. **环境注册**：`env_cartpole.py` 中 `register_custom_cartpole()` 将环境注册为 gym 名 `'CartPoleCustom-v1'`。
2. **物理模型与书一致**：状态 `[x, x_dot, theta, theta_dot]`，动作是连续力 `F ∈ [-F_max, F_max]`；微分方程见 `chap8_1_1storder_1.tex` 的 `eq:cartpole-1-ODE`。`src/2_lqr.py` 的 `calc_sys_mat()` 返回的 A/B 矩阵即书中平衡点线性化结果（系数 `k1..k4`），`4_mpc.py`、`5_ddp_ilqr.py` 的离散化 / 非线性模型与它同源，新增公式需与之一致。
3. **单一环境**：`src/env_cartpole.py` 是唯一环境（存活型奖励：非终止步对应书 `eq:cartpole-1-reward`，系数 `c_θ=0.2`、`c_θ̇=0.05`、`c_x=1.0`、`c_u=0.01`，终止步 `r=0`）。`src-rl/` 与 `eval/` 脚本经相对路径复用 `../src/env_cartpole.py`，**无副本**。
4. **工作目录**：经典控制脚本（`0_random.py` / `1_pid.py` / `2_lqr.py` / `3_lqr_discrete.py`）用相对 import，须在 `src/` 下运行；`4_mpc.py`、`5_ddp_ilqr.py` 与 `src-rl/*`、`eval/*` 用 `__file__`/`os.path` 解析路径，可任意 cwd 运行。
5. **评估口径（三套，切勿混用）**：
   - **经典控制**（PID / LQR / MPC / DDP / iLQR）：`J_ach` + 存活率（与奖励无关），见 `eval/bench.py`——存活另要求 `|x|≤0.5m`（位置回中）。
   - **RL 与书对比**：**稳杆为主**——全程不越界（撑满 1000 步）且末 `H=500` 步 `|θ|≤2°`，`|x|` 只作次级参考。入口是 `eval_noc_term.py`（单个 ckpt）与 `eval_ac_ckpts_noc_term.py`（批量 + 逐种子择优）；**书 §28.1/§28.2/§28.3 的表与结论全部用这一口径**。
   - **RL 训练本身**：用奖励的 return（γ=1）。
   `eval_centering.py` / `eval_ac_ckpts.py` 走的是「较严口径」（额外要求 `|x|≤0.5m`），**不用于书章对比**，别拿它的存活率去对书里的数字。
6. **奖励系数默认值有坑**：环境默认与书 `eq:cartpole-1-reward` 一致（`c_θ=0.2`、`c_θ̇=0.05`、`c_x=1.0`、`c_u=0.01`；可选终端回中项 `c_term` 默认不启用），但 **`1_reinforce.py` 的 `--reward-coefs` 默认是 `0.2,0.05,0.1,0.01`（即 `c_x=0.1`）**，与书不一致——要复现书 §28.1 的 REINFORCE 必须显式传 `--reward-coefs 0.2,0.05,1.0,0.01`（`run_noc_term_seed.ps1` 正是这么写的）。`2_ac.py` / `3_ppo.py` 的默认值与书一致。
7. **产物与归档**：`res/`、`src-rl/res/` 及 `*.pth` / `*.log` 都不纳入版本控制（见仓库根 `.gitignore`），但 **`res/<算法>/` 是书里数字与插图的依据，不要删**；一次性驱动脚本（`src-rl/run_*_stage*.py` 等）与冒烟/诊断残留**已清理**，`src-rl/res/` 里剩下的筛参过程产物需按「已知问题」中列出的三处耦合条件再处理。若改动了奖励、判据或超参导致数字变化，必须同步书稿侧（`docs/todo/cartpole_unfinished.md`、`docs/cartpole-training/*.md` 与正文对应小节）。

## 运行

所有命令中的 `python` 均指 `py311-gym` 环境的解释器 `E:\Anaconda3\envs\py311-gym\python.exe`。

```
# 经典控制（cwd: src/）
python 0_random.py
python 1_pid.py
python 2_lqr.py
python 3_lqr_discrete.py
python 4_mpc.py          # alg1/2 默认运行并出图；alg3 需 RUN_MPC_ALG3=1（暂未调通）
python 5_ddp_ilqr.py     # DDP/iLQR 验证，出图写 1-MN4R/imgs/cartpole/（FIG_DIR 可覆盖）

# 统一评估（cwd: eval/）
python bench.py                              # 经典控制器 3-level 评估
$env:REINFORCE_CKPT="...pth"; python bench.py   # 一并纳入 REINFORCE
python bench_add_reinforce.py                # 只补 REINFORCE，并入 bench_results.json

# 强化学习（任意 cwd；产物默认写 src-rl/res/，用 --save-dir/--ckpt-dir 可改）
python src-rl/1_reinforce.py --env cartpole --iters 500 --seed 0    # 纯 REINFORCE
python src-rl/1_reinforce.py --env pendulum                        # 外部基准：gym Pendulum-v1
python src-rl/2_ac.py --env cartpole --adv td  --iters 120 --seed 0 # 原始 Actor-Critic
python src-rl/2_ac.py --env cartpole --adv gae --lambda 0.99 --iters 120 --seed 0  # AC-GAE
python src-rl/3_ppo.py --env cartpole --algo og   --max-steps 200 --iters 120 --seed 0  # 原始 PPO
python src-rl/3_ppo.py --env cartpole --algo pen  --max-steps 200 --iters 120 --seed 0  # PPO-Penalty
python src-rl/3_ppo.py --env cartpole --algo clip --max-steps 200 --iters 120 --seed 0  # PPO-Clip

# RL 评估（稳杆为主口径，即书里用的那套）
python src-rl/eval_noc_term.py <ckpt.pth>                        # 单个 ckpt
python src-rl/eval_ac_ckpts_noc_term.py --root <含 seed*/ckpt 的目录>   # 批量 + 逐种子择优
python src-rl/eval_noc_term_batch.py --root src-rl/res/s_rf_retrain_l3_noc_term  # REINFORCE 批量
# 较严口径（不用于书章对比）：位置回中 / AC ckpt 复评
python src-rl/eval_centering.py <ckpt.pth>
python src-rl/eval_ac_ckpts.py --ckpt-dir <dir>

# 出书用图（默认写 1-MN4R/imgs/cartpole/，可用 FIG_DIR 覆盖）
python src-rl/make_reinforce_figs.py   # 另有 make_ac_figs.py / make_ac_gae_figs.py /
                                       # make_ppo_og_figs.py / make_ppo_pen_figs.py / make_ppo_clip_figs.py
```

> **复现书里数字的两条硬要求**：① 用 `py311-gym`；② 显式给定与书同口径的超参与初始分布。例如书 §28.1 的 REINFORCE 定稿训练为 `--lr 3e-3 --iters 500 --eps-per-iter 32 --max-steps 500 --gamma 1 --init-log-std 0 --reward-coefs 0.2,0.05,1.0,0.01 --init-theta-range -40,40 --init-x-range -1.5,1.5 --init-xdot-range -1.0,1.0 --init-thetadot-range -10,10`（见 `run_noc_term_seed.ps1`）；§28.2 各方法的正式超参见 `res/<算法>/README.md`。中文日志在 Windows 控制台可能乱码，建议用 `-X utf8` 运行或设 `PYTHONUTF8=1`。

> ⚠️ **后台任务并发上限（重要）**：本机 CPU 为 **8 核**（较老），**同时启动的后台训练/运行任务一次不要超过 8 个**——开多了反而因互相抢占而整体变慢（实测 10 并发时单任务明显劣化，8 核正是瓶颈）。启动前先检查已占用名额，分批进行；RL 多种子/多 lr 扫描务必按此分批。

## 已知问题 / 未完成

- **MPC 算法 3**（不等式约束调节，`mpc_LQR_necons`）：QP 障碍求解器（`opt_qp_barrier`）在较大规模问题（50 变量 / 100 约束）上存在收敛性问题，**默认跳过**（仅打印保留说明），`RUN_MPC_ALG3=1` 可强制运行。书侧有约束 MPC（`subsec:mpc-cons`）因此待补。
- **「ADRC 控制倒立摆」尚未创建**：规划中的新小节，需先补 ADRC 的 Python 实现与二阶系统数值验证（代码拟放 `Codes/chap3_2_adrc/`）。
- **RL 部分已全部收尾，但六种方法的种子依赖均未消除**。判据：**达标 = L1–L3 三档全 `5/5` 稳杆满步**（稳杆为主口径）。各方法最优种子结果与达标种子数：

  | 方法 | 达标种子 | 最优种子 | L1–L3 `J_ach` | 末 500 步 `|x|` |
  |---|---|---|---|---|
  | REINFORCE（§28.1） | 1/4 | seed=2 | 2623.22 / 2787.54 / 5758.2 | ≈1.58 m |
  | 原始 Actor-Critic | 0/4 | seed=0 | 623.56 / 752.01 / —（稳杆 4/5、2/5、0/5） | ≈0.62 m |
  | Actor-Critic GAE（λ=0.99） | 3/4 | seed=0 | 112.85 / 413.86 / 3785.79 | ≈0.28 m |
  | 原始 PPO（ne2） | 2/4 | seed=2 | 32.24 / 318.35 / 3741.02 | ≈0.05 m |
  | PPO-Penalty | **4/4** | seed=2 | 128.13 / 455.16 / 4392.40 | ≈0.29 m |
  | PPO-Clip（ε=0.2） | 3/4 | seed=0 | 308.06 / 521.75 / 3801.59 | ≈0.55 m |

  结论：**降低种子依赖**上 PPO-Penalty 最好（唯一 4/4），**最优种子性能**上原始 PPO 最强（最接近 LQR），AC-GAE 综合最优；原始 AC 未达「降低种子依赖」主判据（但抗翻倒 / 位置回中优于 REINFORCE）。**书已如实保留这些局限与权衡**，不要改写成「某方法全面更优」。每个算法的完整表格、λ/ε/β 扫描与失败记录见 `res/<算法>/README.md` 与书仓库 `docs/cartpole-training/{REINFORCE,AC,PPO}-training-{plan,log}.md`。
- **书 ↔ 代码的一处口径差异（已按代码为准）**：书 `chap6_4_ppo_1.tex` 的 `algo:pg_ac` / `algo:pg_ac_gae` 对同一个 `t` 用了两个不同的回报索引（critic 用 `δ_ω`、优势用 `δ_a`），数学上不自洽；**代码采用自洽的单 δ 形式** `δ_t = r_t + γ·V(s_{t+1}) − V(s_t)`，同时用于 critic 目标与非 GAE 优势，GAE 由其递归 `A_t = δ_t + γλ·A_{t+1}`（`rl_common.compute_gae`）。书 §28.2 已用「口径说明」段写明该差异。
- **可清理项（L1/L2 已执行，L3 待定）**：一次性编排脚本 `src-rl/run_*_stage*.py`、`run_ppo_og_archive.py`，冒烟/诊断残留（`s_ppo_og_smoke*/`、`s_ac_diag_500/`、`s_ppo_og_scan/`），以及混进归档目录的 `res/s_ppo_pen_smoke/` —— **均已清理**。剩余可清理的是 `src-rl/res/` 里的网格筛参产物（`s_ppo_{og,pen,clip}_grid{A120,B300}/`，约 63 MB）与和 `res/` 逐字节重复的正式副本（`s_ac_formal_500/`、`s_ac_lr_scan/`、`s_ppo_pen_formal500/`、`s_rf_retrain_l3_noc_term/`，约 16 MB）。**删这两类前必须先处理三处耦合**：① 书稿 Web 文档会统计 `res`/`src-rl/res` 下**全部** `result.json`（`1-MN4R/docs/web/webdoc/cartpole.py`），而 `docs/web/verify.js` 写死了条数（当前 86，其中 8 条是 `res/` 与 `src-rl/res/` 对同一批运行的重复计入、66 条来自网格筛参）；② `make_ac_figs.py` / `make_ppo_pen_figs.py` / `make_reinforce_figs.py` 直接读 `src-rl/res/` 的正式训练目录（`make_ppo_og_figs.py` / `make_ac_gae_figs.py` 已改读 `res/` 归档），删前需把 `RES` 指向 `res/` 归档（两份逐字节相同，改一行即可；`make_ppo_clip_figs.py` 例外，其过程目录与归档差 1 个文件）；③ `res/<算法>/README.md` 里以「探索/初筛产物在 `src-rl/res/...`」的形式指路，删了会留下死引用。**`res/<算法>/` 归档本身必须保留**（书里数字与图的依据）。另注：`res/`、`src-rl/res/` 与 `*.pth`/`*.log` 均被 `.gitignore` 排除，**删除不可恢复**，动手前先自行备份。
- 章节对应关系与遗留事项的完整清单见书仓库 `docs/todo/cartpole_unfinished.md`（书 ↔ 代码对应表、门禁数字、后续工作）。

## 其他

- **本文件与仓库根 `../AGENTS.md`**：根文件管整仓库（环境、`Codes/` 写法、提交与 PDF 发布物等），本文件管 `CartPole/` 细节；两者冲突以本文件为准。改动本目录结构或关键约定后，顺手核对根文件的「`CartPole/`（简略）」一节是否仍然成立。
- 本目录属于 `4-MT4R-github` 仓库，改动随该仓库提交（远端名是 `main`）。`res/`、`src-rl/res/` 下的训练产物（`.pth` / `.log` / history JSON / PNG 等）**保留在本地但不纳入版本控制**，见仓库根 `.gitignore`；提交/推送只在用户明确要求时进行。
- 出图脚本（`src/3_lqr_discrete.py`、`src/4_mpc.py`、`src/5_ddp_ilqr.py`、`eval/bench.py`、`src-rl/make_*_figs.py`）默认把图写到同工作区的书稿仓库 `1-MN4R/imgs/`，可用 `FIG_DIR` / `FIG_DIR_CART` / `FIG_DIR_MPC` 环境变量覆盖（注意 `4_mpc.py` 默认写 `imgs/mpc/`，其余写 `imgs/cartpole/`）。
- **书侧还会读本目录**：书仓库的交互式 Web 文档（`1-MN4R/docs/web/build.py`）会扫描 `CartPole/{src,src-rl,eval,test}/**` 生成代码清单、并读 `res/`、`src-rl/res/` 下的 `result.json` 展示归档结果（见 `1-MN4R/docs/web/README.md`）。因此**改文件名 / 挪目录 / 删 `result.json` 前先确认书侧 Web 文档与正文的引用**。
- 训练与调参的全过程记录都在书仓库 `docs/cartpole-training/`：`REINFORCE-training-log.md`（纯 REINFORCE 的尝试 1–18）、`AC-training-{plan,log}.md`（AC / AC-GAE 阶段 0–6）、`PPO-training-{plan,log}.md`（PPO 三变体的阶段 A/B/C 筛参与正式训练）。本目录只保留**可运行代码 + 结果归档**，历史成败不在此重复记录。
- 本目录的 `.claude/settings.json` 只放了两条 `1_reinforce.py` 的命令白名单，属历史遗留；它不影响脚本本身的运行方式。
