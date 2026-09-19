# AGENTS.md

本文件为 AI 编码助手（Claude Code、Codex 等）在 `4-MT4R-github/` 仓库下工作时提供指引。请先通读再动手。

> 进入 `CartPole/` 时，该目录自带一份更具体的 `CartPole/AGENTS.md`（环境、目录明细、关键约定、运行命令、已知问题）。两者冲突时以 `CartPole/AGENTS.md` 为准。

## 回复语言（强制）

无论什么时候，永远用中文回复。

## 这是什么

本仓库是《Math Toolbox for Robotics》（机器人数学工具书，简称 **MN4R**，作者 Xinran Wei）的**配套代码仓库**，对外公开；代码以 GPLv3 开源（见 `LICENSE`）。

- 远端：`main` → `git@github.com:weixr18/MT4R.git`。**注意远端名就叫 `main`**（不是 `origin`），默认分支也是 `main`。
- 两个顶层代码目录：

| 目录 | 内容 |
|---|---|
| `Codes/` | 按书章节组织的算法实现，覆盖 Part I–VII（优化、插值、运动学、动力学、滤波器、LQR、深度学习、强化学习、视觉导航） |
| `CartPole/` | Part VIII（一阶倒立摆）配套仿真与控制实验代码：经典控制 + 强化学习，含统一评估与训练产物归档 |

- 书稿正文在同工作区的**另一个仓库** `../1-MN4R`（`D:\Projects\2024_MN4R\1-MN4R`）。两者**配套开发、相互独立**（各自独立 git 提交）：本仓库是「代码」，`1-MN4R` 是「书」。书稿侧的工作指引见 `../1-MN4R/AGENTS.md`。
- `README.md` 是本仓库对外的门面（中英双语，描述 `Codes/` 与 `CartPole/` 的定位），改动目录结构时记得同步。

**书 ↔ 代码的对应关系、完成状态与遗留事项，以书稿仓库的 `1-MN4R/docs/todo/cartpole_unfinished.md`（Part VIII）与 `1-MN4R/docs/todo/unfinished_contents.md`（全书）为准。** 写/改 `CartPole/` 前必读前者。

## 目录结构

```
4-MT4R-github/
├── Codes/                  # Part I–VII 各章算法实现（chap<部分>_<小节>_<主题>/）
├── CartPole/               # Part VIII 倒立摆配套代码（详见 CartPole/AGENTS.md）
├── readme-pngs/            # README 用的插图
├── README.md               # 仓库门面（中英双语）
├── LICENSE                 # GPLv3
├── MT4R-chn-v0.1.4.pdf     # 中文版预览 PDF（发布物：勿编辑、勿修改）
├── MT4R-eng-v0.1.4.pdf     # 英文版预览 PDF（同上）
└── .gitignore
```

## `Codes/` —— 按书章节组织的算法实现

### 与书章节的对应

目录命名规则 `chap<部分号>_<该部分内章节号>_<主题>/`，与书稿 `1-MN4R/Part_N_*/chap*.tex` 的文件名同源（例：`Codes/chap3_4_lqr/` ↔ `1-MN4R/Part_3_control/chap3_4_*.tex`）。

| 目录 | 主题 | 书侧 |
|---|---|---|
| `chap1_3_optmz/` | 优化（无约束 / 约束 / 极小极大 / QP 障碍法） | Part I 优化 |
| `chap1_4_interp/` | 插值（二次 / 三次 / 旋转 / 李群样条） | Part I 插值 |
| `chap2_2_kinematcs/` | 正运动学 | Part II |
| `chap2_3_diffk/` | 微分运动学（雅可比、逆运动学） | Part II |
| `chap2_4_dynamics/` | 动力学 | Part II |
| `chap3_1_cartpole/` | 经典控制入门（`test_gym.py`：在 gym 自带 CartPole 上试 PID） | Part III 经典控制 |
| `chap3_3_filter/` | 滤波器（KF / EKF / ESKF / UKF） | Part III 观测器与滤波器 |
| `chap3_4_lqr/` | LQR | Part III 最优控制 |
| `chap5_1_DL/` | 深度学习基础（MLP / Adam / BN，含 `mnist/` 数据） | Part V |
| `chap6_1_RLbasic/` | RL 基础（DP 预测 / MC-TD 预测 / DP 控制 / `corridor.py`） | Part VI |
| `chap6_2_RLtable/` | 表格型控制（SARSA / Q-learning 一类） | Part VI |
| `chap6_3_pgmeth/` | 策略梯度（REINFORCE） | Part VI §21 |
| `chap7_3_3d/` | 多视图几何与位姿估计（F/H 矩阵、三角化、ICP、DLT、EPnP、BA） | Part VII |

### 文件命名与写法约定

- `code_*.py` = 书中算法/公式的落地实现（与书里的算法框、代码块一一对应）；`test_*.py` = 该实现的验证脚本（可运行、给出验证结论）。
- 一个目录内可有多个 `code_*` / `test_*`，编号顺序与书中小节顺序一致；书里改了算法框，代码与验证脚本要跟着改。
- **复用优先、单一来源**：跨目录复用一律用相对 import，**不复制副本**。典型例子：`chap1_3_optmz/code_4_qp.py` 的 `opt_qp_barrier` 被 `CartPole/src/4_mpc.py` 经仓库内相对路径导入（对应书 `algo:opt_qp_barrier`）；`chap6_3_pgmeth/code_1_reinforce.py` 复用 `chap5_1_DL/code_1_MLP.py` 的 MLP。
- 文件顶部一般有中文 docstring，写明**验证的是书哪个文件 / 哪个算法框、验证方式与结论**——新增文件请沿用这一体例。
- 各目录都有 `__pycache__/`（已 gitignore）：不要提交，也不要据此判断某脚本是否还在用。
- `Code-Test.code-workspace`、`.vscode/launch.json` 是本目录的 VS Code 调试配置（`launch.json` 当前指向 `chap3_3_filter/test_4_ukf.py`），切换调试目标直接改它。

### 与 `CartPole/` 的分工（易混淆点）

- `Codes/` 只放**按章节的算法验证**；Part VIII 倒立摆的验证代码统一放 `CartPole/`（书稿侧约定见 `1-MN4R/AGENTS.md`「算法验证代码的放置规则」）。
- `Codes/chap3_1_cartpole/test_gym.py` **与 `CartPole/` 无关**：它是 gym 自带 CartPole 上的 PID 试手。**Part VIII 的唯一环境是 `CartPole/src/env_cartpole.py`**（`CartPoleCustom-v1`），两者不可混用。
- `Codes/chap6_3_pgmeth/code_1_reinforce.py` 验证的是书 Part VI 的 REINFORCE 理论算法（离散/表格场景），与 `CartPole/src-rl/1_reinforce.py`（连续动作、倒立摆落地）是两套东西。

## `CartPole/` —— Part VIII（一阶倒立摆）配套代码（简略）

> **该目录有自己的一份 `CartPole/AGENTS.md`**：环境与依赖、目录明细（`src/` `src-rl/` `eval/` `test/` `res/`）、关键约定、运行命令、已知问题全在那里，**动手前先读它**。下面只给最小必要信息。

- 内容：经典控制（随机策略 / PID / 连续与离散 LQR / 无约束 MPC / DDP / iLQR）＋ 强化学习（REINFORCE、原始 Actor-Critic、Actor-Critic GAE、原始 PPO、PPO-Penalty、PPO-Clip），含统一评估脚本与训练产物归档。
- 目录速览：`src/`（经典控制）、`src-rl/`（RL 训练 / 评估 / 出图）、`eval/`（统一评估）、`test/`、`res/`（各算法归档，多数子目录带一份 `README.md`，`REINFORCE/` 无）、`src-rl/res/`（各次运行的中间产物，不纳入版本控制）。
- **运行环境：conda `py311-gym`**（解释器 `E:\Anaconda3\envs\py311-gym\python.exe`）。不要用系统 `python`（无 numpy/scipy/torch/gym）。
- 书 ↔ 代码对应与完成状态：`1-MN4R/docs/todo/cartpole_unfinished.md`；训练全过程记录：`1-MN4R/docs/cartpole-training/`（`REINFORCE-training-log.md`、`AC-training-{plan,log}.md`、`PPO-training-{plan,log}.md`）。

## 环境与运行

- **Python：conda `py311-gym`**（Python 3.11.9，解释器 `E:\Anaconda3\envs\py311-gym\python.exe`）。本仓库所有需要 numpy / scipy / torch / gym 的脚本都用它。Git Bash 下可 `PY="/e/Anaconda3/envs/py311-gym/python.exe"` 后以 `"$PY" ...` 调用。
- **默认 `python` 不可用**：`PATH` 里的 `python` 是 ESP-IDF 自带的，未装科学计算库；`E:\Anaconda3\python.exe`（Anaconda 根）在本机可能启动即失败（`0xC0000022`），同样不要用。
- **中文输出**：Windows 控制台默认 GBK，Python 打印中文会乱码或抛 `UnicodeEncodeError`——用 `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` 环境变量，或给解释器加 `-X utf8`。
- `Codes/` 下若干 `__pycache__` 是 `.cpython-38`，说明部分早期脚本在 Python 3.8 下跑过；**当前标准是 `py311-gym`**，重跑验证以它为准。
- **后台任务并发上限（重要）**：本机 CPU 为 **8 核**（较老），同时启动的后台训练/运行任务一次不要超过 8 个——开多了互相抢占反而整体变慢（实测 10 并发时单任务明显劣化）。多种子 / 多超参扫描务必分批（Book 侧记录的各阶段驱动脚本其 `max_workers` 即按此设定，见 `1-MN4R/docs/cartpole-training/PPO-training-*.md`）。
- **PowerShell 不展开 glob**（传给外部程序时，如 `Codes/chap*/*.py`）：改用 `Get-ChildItem` 管道，或用工具自身的过滤参数（如 `rg -g '*.py'`）。

## 版本控制与提交约定

- 远端名 `main`，默认分支 `main`；提交信息简短说明改动（如 `add CartPole (Part VIII) code`）。
- `.gitignore` 只忽略少量内容：`*.vscode/*`、`*.code-workspace`、`__pycache__/`、`*.pyc`、`*.gz`，以及 `CartPole/res/`、`CartPole/src-rl/res/`、`*.pth`、`*.log`——**训练产物保留在本地、不纳入版本控制**。
- **两个 PDF（`MT4R-chn-v0.1.4.pdf` / `MT4R-eng-v0.1.4.pdf`）是发布物**：不要编辑、不要重新生成。书的构建与发布流水线（XeLaTeX 编译、`versions/`、`release/`、`scripts/make_release_chn.py`）都在 `1-MN4R` 仓库，本仓库只存放成品。
- **提交 / 推送只在用户明确要求时进行。**
- 若在代码侧新增了书里没有的算法、或改变了算法口径（奖励、判据、超参），记得回头同步书稿侧对应章节与索引：`1-MN4R/Part_0_prepare/2index.tex` 的 `$\dag$` / `$\circ$` 标注，以及 `1-MN4R/docs/todo/*.md`。

## 常见任务与检查清单

**在 `Codes/<chapter>/` 中新增或修改算法验证代码**：

1. 先在 `1-MN4R` 找到对应章节与算法框（`algo:` 标签），确认代码与公式的变量名一致（书刻意保持「算法 ↔ 代码」同名，见书稿 AGENTS.md）。
2. `code_*.py` 写实现、`test_*.py` 写验证；需要复用已有实现时用相对 import，不复制副本。
3. 用 `py311-gym` **实跑验证**：书侧要求「书中实测数字必须来自实际运行配套代码」，所以引用的数字必须真的从这份代码跑出来。
4. 验证通过、算法达到「已验证」状态后，同步书稿 `2index.tex` 的 `$\dag$` 标注与 `docs/todo/unfinished_contents.md`。

**在 `CartPole/` 中做改动**：先读 `CartPole/AGENTS.md`，再读 `1-MN4R/docs/todo/cartpole_unfinished.md`。

## 已知未完成 / 注意事项（速查）

1. **有约束线性 MPC**：`CartPole/src/4_mpc.py` 的 `mpc_LQR_necons`（在线 QP 障碍法）在较大规模问题上存在收敛性问题，默认跳过（`RUN_MPC_ALG3=1` 才强制运行）；书侧 `subsec:mpc-cons` 待补（见 `cartpole_unfinished.md` 遗留事项 A）。
2. **「ADRC 控制倒立摆」**：规划中、尚未创建（代码拟放 `Codes/chap3_2_adrc/`，见 `cartpole_unfinished.md` 遗留事项 B）。
3. **Part VIII 的 RL 部分已全部收尾**：REINFORCE 与五种 PPO/AC 变体均已定稿成文（书 `chap8_2_RL_1.tex` / `chap8_2_RL_2.tex`），八个 RL 算法框均已升级 `$\dag$`；但**没有任何一种方法完全消除种子依赖**（各方法的达标种子数与最优种子性能对比见 `cartpole_unfinished.md`）。
4. **`CartPole/` 的一次性驱动脚本已清理**：`src-rl/run_*_stage*.py`（筛参 / 正式训练编排）与 `run_ppo_og_archive.py`，连同冒烟 / 诊断残留（`s_ppo_og_smoke*/`、`s_ac_diag_500/`、`s_ppo_og_scan/`）和空归档残留 `CartPole/res/s_ppo_pen_smoke/` 均已删除；历史命令见 `1-MN4R/docs/cartpole-training/PPO-training-{plan,log}.md`。`CartPole/src-rl/res/` 里剩下的网格筛参产物若要清理，先处理 `CartPole/AGENTS.md`「已知问题」列出的三处耦合（书稿 Web 文档的 `result.json` 计数、`make_*_figs.py` 的数据源路径、`res/<算法>/README.md` 的指路引用）。
5. `Codes/chap7_3_3d/` 下的 `scratch_ransac_test.py`、`test_3_2_1_tmp.py` 从命名看属草稿/临时脚本，改动前先确认是否仍在使用。
