# AGENTS.md

本文件为 AI 编码助手（Claude Code、Codex 等）在 `4-MT4R-github/Robot-Control/` 目录下工作时提供指引。

## 这是什么

本目录 `Robot-Control/` 是《Math Toolbox for Robotics》（机器人数学工具书，简称 **MN4R**）**Part IV（机器人控制基础）**的**配套仿真与控制验证代码**，相对独立、自包含。

覆盖算法（第一批，**运动控制**）：

| 算法 | 书 `\label` | 章节 |
|---|---|---|
| JS 重力补偿 PD 控制 | `algo:robctrl_js_pd` | `Part_4_robot_control/chap4_3_jsctrl.tex` |
| JS 逆动力学 PD 控制 | `algo:robctrl_js_invdyn` | 同上 |
| OS 重力补偿 PD 控制 | `algo:robctrl_os_pd` | `Part_4_robot_control/chap4_4_osctrl.tex` |
| OS 逆动力学 PD 控制 | `algo:robctrl_os_invdyn` | 同上 |

后续批次（**本目录的既定归宿**，见书仓库 `docs/robot-control-verify/`）：JS/OS 阻抗控制、零力控制、导纳控制、
力/位姿混合控制（`algo:robctrl_*_impedence`、`zeroforce`、`admittance`、`hybrid`），
以及 `chap4_2` 的独立关节 PID/ADRC。

**在仓库中的位置**：本目录属于配套代码仓库 `4-MT4R-github`（远端 `main` → `git@github.com:weixr18/MT4R.git`）。
仓库整体约定见根 `../AGENTS.md`；`Codes/` 按书章节组织 Part I–VII，`CartPole/` 承载 Part VIII，
本目录承载 **Part IV**。书稿仓库在同工作区 `../../1-MN4R`（`D:\Projects\2024_MN4R\1-MN4R`），
两者配套开发、相互独立。

**书 ↔ 代码的对应关系、验证阶段与完成状态，以书仓库 `../../1-MN4R/docs/robot-control-verify/part4-motion-control-plan.md`
与 `part4-motion-control-log.md` 为准。** 本目录只保留**可运行代码 + 结果归档 + 验证报告**，验证过程与成败记录不在此重复。

> ⚠️ **与 CartPole（Part VIII）的关键区别**：CartPole 的验证结果**进书稿正文**（配图、回填实测数字）；
> **本批次的结果不进 `chap4_*.tex`、不往 `1-MN4R/imgs/` 放图、不在正文回填数字**。
> **交付物是 `report/` 下的独立 LaTeX 验证报告**（`report.tex` + `report.pdf`，结构见计划 §6.2）。
> 书侧只更新两处机读性标注：`2index.tex` 的 `$\dag$` 与 `unfinished_contents.md` 的清单。

## 环境与依赖

- **Python 环境：conda `py311-gym`**（Python 3.11.9，解释器 `E:\Anaconda3\envs\py311-gym\python.exe`）。
  本目录所有脚本一律用它，**不要**用系统 `python`（无 numpy/scipy）或 Anaconda 根解释器（本机启动即失败）。
- 依赖：`numpy`、`scipy`、`matplotlib`。**第一批验证不需要第三方仿真引擎**（见「关键约定」1）。
- **阶段 1 的 Pinocchio 单独装在 `D:\Projects\2024_MN4R\.envs\py311-pin`**（工作区本地 conda env，见下方「运行」）。
  ⚠️ 不要 `pip install pin`：PyPI 上的 `pin` **没有 Windows 轮子**（官方 README 写明 pip 仅 Linux），
  本机会退化成源码编译并必然失败；`pinocchio` 的 Windows 构建只在 **conda-forge**。
  **也不要装进 `py311-gym`**——那会牵动已对齐的 numpy/scipy/boost 等，影响 CartPole 与其他验证的可复现性。
- ⚠️ **跑 `py311-pin` 前必须先把它的 `Library\bin` 加进 `PATH`，且必须早于 `import numpy`**
  （2026-09-18 实测，见书仓库日志尝试 6）：按路径直调 `python.exe` **等于没激活 conda 环境**，
  PATH 上缺该目录会导致 numpy 的 BLAS（`Library\bin\mkl_rt.3.dll`）**延迟加载失败**，
  表现为连 `A @ A` 都崩、进程退出码 `0xc06d007f`（`-1066598273`），而 `np.eye`/`np.sin` 却正常。
  ```python
  os.environ["PATH"] = r"D:\Projects\2024_MN4R\.envs\py311-pin\Library\bin" + os.pathsep + os.environ["PATH"]
  ```
  `py311-gym`（numpy 2.1.0）不受影响。
- ⚠️ **pinocchio 的关节复合语义（2026-09-18 实测钉死，务必别记反）**：
  单关节 `Σ`、关节变量 `q` 时，**`oM = Σ · Rz(q)`**（`Rz` 施加在 `Σ` 之后，
  等价于「`Σ` 的前两列被 `Rz` 旋转」）。对照错的写法 `Rz(q)·Σ` 实测偏差 6.4e-01。
  串联链即 `oM_i = Σ_1Rz(q_1) Σ_2Rz(q_2) … Σ_iRz(q_i)`。
  ⚠️ 这与书链 `T_i = T_{i-1}·A_i(q_i)` 的复合方向**相反**，是阶段 1 装配对不上的真正原因。
- ⚠️ **`pin.Inertia(m, c, I)` 里的 `I` 是「绕质心」的惯量**（库内部自己加平行轴项）。
  若再手动加 `m(‖c‖²E − c cᵀ)` 就会重复计入：实测 `B` 相对误差从 1e-15 恶化到 2e-1，
  而 `g`（只依赖质心位置）**仍保持 ~1e-16**——所以**不能用 `g` 单独判定惯量挂对了**。
- ⚠️ **pinocchio 的 `model.addFrame()` 只改 model、不更新已有 `data`**：复用旧 `data` 会让
  `data.oMf[fid]` 越界并触发访问违例（`0xc0000374`，GC 时二次崩溃）。
  **`addFrame` 之后必须重建 `data = model.createData()`。** 这不是库缺陷。
- 中文输出：Windows 控制台默认 GBK，加 `PYTHONUTF8=1` 或给解释器 `-X utf8`，否则中文日志会乱码。
- **后台任务并发上限**：本机 8 核（较老），同时启动的后台任务一次不超过 8 个（与 `CartPole/` 同一台机器）。

## 目录结构

```
Robot-Control/
├── AGENTS.md           # 本文件（本目录的工作指引）
├── model/              # 机器人模型库：Part II 已 $\\dag$ 代码的副本（见「模型库」）
├── code/               # 本目录自有的仿真层、四类控制器、度量与出图
├── test/               # 验证脚本（按阶段分子目录）
│   ├── model/          #   阶段 0/1 模型自检与权威对拍（含副本同步对拍）
│   ├── ctrl_js/        #   关节空间控制器验证
│   ├── ctrl_os/        #   操作空间控制器验证
│   └── ctrl_common/    #   共享实验驱动（阶段 2 起）与跨用例鲁棒性测试（阶段 4）
├── scripts/            # 一次性编排脚本（阶段 3 起：步长探针 / 出图烟雾测试 / 归档摘要）
├── res/                # 实验产物与结果归档（整体不纳入版本控制）
└── report/             # ★ 验证报告（主要交付物，**例外纳入版本控制**）
    ├── report.tex      #   LaTeX 主报告（v2.1，xelatex 连续编译两次）
    ├── report.pdf      #   编译产物（65 页）
    ├── README.md       #   报告目录索引 + 编译/排版约束
    └── figs/           #   报告引用的 20 张图
```

> 📌 **2026-09-20 目录调整**：报告由 `res/report/` 迁到 `report/`；原 `docs/`（只有一份
> `README.md` 转述书仓库的验证约定）已删除——**验证计划与运行日志以书仓库
> `../../1-MN4R/docs/robot-control-verify/` 为唯一来源**，本目录不再留转发层。
> 报告同时由 Markdown 改为 LaTeX（`report/report.tex`），原 `.md` 已删除。

### `model/` —— 模型库（**副本，有意豁免单一来源约定**）

| 文件 | 内容 | 原件（**单一来源，勿在副本上改**） |
|---|---|---|
| `code_fk.py` | `calc_T_n_to_last`、`rot_to_eular`、`robot_fk` | `Codes/chap2_2_kinematcs/code_fk.py` |
| `code_jacobian.py` | `eular_diff_to_w`、`robot_jacobian_w`、`robot_jacobian_a`、`robot_j_centroid` | `Codes/chap2_3_diffk/code_jacobian.py` |
| `code_ik.py` | `robot_ik_gd`、`robot_ik_gauss_newton`、`robot_ik_lm` | `Codes/chap2_3_diffk/code_ik.py` |
| `code_dynamics.py` | `robot_B`、`robot_dyn` | `Codes/chap2_4_dynamics/code_dynamics.py` |
| `code_jacobian_dot.py` | `jacobian_dot_fd`、`robot_jacobian_dot_a`（`J̇_a`） | **—（Robot-Control 自有，`Codes/` 里没有对应实现）** |

- **为什么是副本**：为了让 Part IV 验证代码自包含、可独立运行与归档（用户明确要求），
  这与仓库根 `AGENTS.md`「复用优先、单一来源，不复制副本」的约定**有意冲突**——本目录是该约定的**唯一豁免点**。
- **代价与兜底**：两份可能漂移。每个副本文件顶部都有 `PROVENANCE` 行，记录源路径与**原文件的 sha256[:16]**；
  `test/model/test_model_sync.py` 做**哈希 + 数值**双重对拍。**任何时候都可运行它确认副本是否过期**：

  ```
  PYTHONUTF8=1 E:\Anaconda3\envs\py311-gym\python.exe Robot-Control/test/model/test_model_sync.py
  ```

- **改模型的正确姿势**：改 `Codes/` 下的**原件** → 重新同步副本 → 更新 `PROVENANCE` 的哈希。
  **不要在副本上直接改**（对拍会失败，且原件仍是书 Part II 的依据）。
- **导入方式**：`model/__init__.py` 已把 `model/` 目录加入 `sys.path`，故可扁平导入：

  ```python
  sys.path.insert(0, <Robot-Control>)
  from model import robot_fk, robot_jacobian_a, robot_B, robot_dyn
  ```

- ⚠️ **已知限制**：`robot_jacobian_a` 硬编码 **6×6** 输出（位置 3 + ZXY 欧拉角 3），
  **不能直接用于 2R 平面臂**（其任务空间应为 3 维 `[x, y, φ]`）。2R 用例需自带精简版雅可比，见
  书仓库 `docs/robot-control-verify/part4-motion-control-plan.md` 阶段 0。

### `code/` —— 仿真层与控制器

| 文件 | 内容 |
|---|---|
| `code_models.py` | ✅ **已落地（阶段 0 起）**：用例参数（平面 2R、6R Puma 型）的**唯一定义处**（DH + 连杆参数）、2R 闭式 `robot_fk_2r`/`robot_jacobian_a_2r`/`robot_jacobian_dot_a_2r`（含**位置子任务** 2×2 版与闭式 `robot_ik_2r_pos`）、标称位形 `Q_NOM_*`、**稳定平衡位形** `Q_EQ_*`、调节目标 `Q_D_REG_*`、力矩上限 `U_MAX_*`；**控制用例统一接口 `RobotCase` / `make_case("2R"|"6R")`**（阶段 2 新增） |
| `code_sim.py` | ✅ **已落地（阶段 2；阶段 3 增记 `σ_max`）**：被控对象 `q̈ = B⁻¹(u + τ_e − Cq̇ − F_fq̇ − g)`、定步长 RK4、控制零阶保持（`h_c = k·h_sim`）、外力/摩擦/噪声注入钩子、全量记录（含 `σ_min(J_a)`、**`σ_max(J_a)`**、`cond(B)`、NaN/异常标记）。`σ_max` 与 `σ_min` 共用同一次 SVD（零额外成本），用于判据 3a 的 `cond(J_a)` |
| `code_ctrl_common.py` | ✅ **已落地（阶段 2）**：`pd_gains(ω_n, ξ, n, Lam=None)`——书中 PD 增益选取式（含**参考惯量标定** `Lam`，6R 必需，见「已知问题」） |
| `code_ctrl_js.py` | ✅ **已落地（阶段 2）**：JS 重力补偿 PD、JS 逆动力学 PD（`algo:robctrl_js_pd` / `js_invdyn`），与书中算法框逐行对应 |
| `code_ctrl_os.py` | ✅ **已落地（阶段 2）**：OS 重力补偿 PD、OS 逆动力学 PD（`algo:robctrl_os_pd` / `os_invdyn`），含 `J̇_a` 与 `J_a⁻¹` 的落地方式说明 |
| `code_traj.py` | ✅ **已落地（阶段 2）**：参考轨迹生成——JS 关节空间 / OS 操作空间的常值参考与 minimum-jerk 五次多项式（**解析各阶导数**）；红线（OS 必须在操作空间生成轨迹）在此集中声明 |
| `code_metrics.py` | ✅ **已落地（阶段 2，阶段 3/4 增补）**：误差/时间/控制代价/数值健康度指标（含阶段 3 增补的 **`cond_J_max = max σ_max/σ_min`**）、任务空间误差分项范数（含等效臂长 ℓ）、`(ξ̂, ω̂_n)` 对数衰减率辨识（含**峰位置/幅值**，供判据 1 的出图标注）、出图（Agg 后端；阶段 3 增补 `plot_logdec_fit` / `plot_grouped_lines` / `plot_grouped_bars`，阶段 4 增补 **`_wrap_caption` 图注自动折行**）。⚠️ **图上的文字有硬约束（2026-09-20 逐个实测）**：① 不用「希腊字母 + 组合变音符」（`ξ̂`/`ω̂`/`x̃` 的 U+0302/U+0303/U+0304 在 Microsoft YaHei 里**都缺字形**，会渲染成方框）；② 不用 `⇒`(U+21D2)、`⚠️`、`ε`、上标 `⁻¹`；③ **图注里不写 `$...$`**——`_wrap_caption` 的折行会切断数学块，残缺的 `$` 之后整段会被当数学模式渲染成字面量乱码。**统一改用纯文本 + 已验证有字形的 Unicode（σ/ω/·/≈/≤/ℓ/→）**；配 `-W error::UserWarning` 可把这些缺字形告警变成硬失败 |

控制器对外接口统一为 `u = ctrl(state, ref, gains, case, F_f=None)`（`state = (q, q̇)`，
`ref` 为参考量字典或 `t -> 字典`）；见 `code_ctrl_common.py` 的模块文档与 `code_models.RobotCase`。

> ⚠️ `code_models.py` 里 2R 的**参数口径与 Part II 既有测试脚本一致**（`L1=1.0, L2=0.8, m1=2.0, m2=1.5, I1=0.1, I2=0.08`）；
> 6R 的 DH 取经典 Puma 560 标准 D-H，**连杆惯性是按尺寸估的简化值**（非官方参数，已在文件内声明）。
> 「DH 与连杆参数只能有一处定义」是硬约定：阶段 2 的仿真层/控制器一律从这里取，勿另写一份。
>
> ⚠️ **2R 的「控制用任务空间」是 2 维位置子任务 `[x, y]`**（阶段 2 修订）：`robot_fk_2r` 的 3 维输出
> `[x, y, φ]` 对应的 `J_a` 是 **3×2 浸入**，书里 OS 逆动力学算法框的 `J_a⁻¹` 不存在；
> 取位置子任务后 `J_a` 为 2×2 方阵，4 个算法都能严格按书中公式落地。详见 `code_models.py` 文件头。

### `test/` —— 验证脚本（对应计划的阶段）

| 文件 | 对应阶段 | 内容 |
|---|---|---|
| `model/run_stage0.py` | 阶段 0 | ✅ **一键入口**：跑下面三个脚本并把结论汇总成计划 §5.0.5 的判据表，写 `res/model_stage0/{result.json,run.log}` |
| `model/test_model_sync.py` | 阶段 0 | ✅ 副本与原件同步对拍（哈希 + 数值，判据 1）|
| `model/test_jacobian_dot.py` | 阶段 0 | ✅ `J̇a` 与乘积法则验证（2R 闭式基准 + 差分收敛阶；判据 2、3）|
| `model/test_convention.py` | 阶段 0 | ✅ 约定锁定：`C` 精度与反对称性（判据 6）、重力项与 ∂U/∂q（判据 4）、静平衡 5a–5d（判据 5）|
| `model/test_model_pinocchio.py` | 阶段 1 | ✅ 与 Pinocchio 权威对拍（`FK`/`B`/`g`/`Cq̇`/`J_w` 5 项 + 故意失配自检）；**须用 `py311-pin` 解释器跑**（不是 `py311-gym`）|
| `ctrl_common/verify_lib.py` | 阶段 2–4 | ✅ **共享实验驱动（阶段 2）**：初值采样、批量仿真、指标汇总、判据判定、出图；阶段 2 的四条判据口径都在这里 |
| `ctrl_common/stage3_lib.py` | 阶段 3 | ✅ **已落地并实测通过**：6R 增益标定（`Λ`）、判据 1–6 的实现、步长对照、出图；阶段 3 的全部实验口径都在这里（含 `T_REG_6R = 2.0 s` 的余量推导、判据 3a 的 `COND_J_PREMISE = 100`、判据 1 的判定步长 `DT_FIT3 = 1 ms` 与观测窗 `T_2ND = 3.0 s` 的依据）|
| `ctrl_common/run_stage2.py` | 阶段 2 | ✅ **一键入口（2026-09-20 用修正积分器重跑，实测 4/4 PASS，10.1 min）**：跑四个算法 → 汇总计划 §5.2.5 判据表 → 写 `res/2r_stage2/{result.json,run.log,figs/}`。⚠️ **先写 `result.json`、再出图**（2026-09-20 补齐，与阶段 3 的硬规矩对齐；出图失败只记 `figure_error`，绝不影响归档） |
| `ctrl_common/run_stage3.py` | 阶段 3 | ✅ **一键入口（2026-09-20 用修正积分器重跑，实测 6/6 PASS，41.7 min）**：6R 四算法 + 跨算法专项 → 汇总计划 §5.3.7 的 6 条判据 → 写 `res/6r_stage3/{result.json,run.log,figs/}`；支持 `--quick` / `--only` / `--workers`。⚠️ **先写 `result.json`、再出图**（出图失败不影响归档）|
| `ctrl_js/test_js_pd.py` | 阶段 2、3 | ✅ 2R（阶段 2 四条判据）+ 阶段 3 `run_stage3()`（6R 判据 3 + 判据 2，**均 PASS**）|
| `ctrl_js/test_js_invdyn.py` | 阶段 2、3 | ✅ 2R + 阶段 3 `run_stage3()`（6R 判据 1 + 3，**均 PASS**，判据 1 最差 1.2764%）|
| `ctrl_os/test_os_pd.py` | 阶段 2、3 | ✅ 2R + 阶段 3 `run_stage3()`（6R 判据 3，**PASS**：3a 18/20、3b 2 组边界样本）|
| `ctrl_os/test_os_invdyn.py` | 阶段 2、3 | ✅ 2R + 阶段 3 `run_stage3()`（6R 判据 1 + 3，**均 PASS**，判据 1 最差 0.5563%）|
| `ctrl_os/test_os_singular.py` | 阶段 4 | ✅ **已完成（2026-09-20，日志尝试 14）**：奇异位形边界——`q5 → 0` 的腕部奇异族、二分反解目标 `σ_min`；三场景（`out_of_range` 退化方向 / `in_range` **最良态方向**（初值同为 `q_t`、只换方向）/ `js_hold` 对照）+ DLS 对照（**超出书本范围的建议**）。**判据 2 PASS**：`os_pd` 首个失效 `5.0e-2`、`os_invdyn` `1.0e-4`；JS 对照 0/6、`os_pd` 的 `in_range` 0/5 |
| `ctrl_common/test_robust.py` | 阶段 4 | ✅ **已完成（2026-09-20）**：模型失配、摩擦未补偿、测量噪声、控制周期敏感性、离散化失稳阈值、每周期耗时；支持 `--only <实验名>`。**判据 1（四项全 PASS）+ 判据 3 PASS**。⚠️ 2026-09-20 修掉一处真 bug（第 61 行用 `np` 却**未 import numpy**，`period` 实验直接崩）——**改完必须跑全链路 `--quick` 冒烟** |
| `ctrl_common/stage4_lib.py` | 阶段 4 | ✅ **已完成（2026-09-20）**：阶段 4 的共享驱动（6 个实验 + 出图），全部口径与依据都在这里。判据 2 的三处口径在尝试 13 冻结（全程峰值归一、`in_range` 方向对照、`T_SING = 1.2 s`、新增「穿越奇异面」判据）；摩擦守卫改全程峰值归一。⚠️ **出图口径：图上文字一律纯文本 + 仅用已验证有字形的 Unicode**（不用 `$...$`，不用 `⇒/⚠️/ε/⁻¹/组合变音符`——Microsoft YaHei 缺字形，且 `_wrap_caption` 折行会切断 `$...$`） |
| `ctrl_common/run_stage4.py` | 阶段 4 | ✅ **已完成（2026-09-20，实测 3/3 PASS，64.6 min）**：一键入口 + 归档（`--quick` / `--only` / `--workers` / **`--figs-only`**），汇总计划 §5.4.3 的 3 条判据；沿用「**先写 `result.json` 再出图**」。⚠️ `--figs-only` 由归档重建图集（秒级、不跑仿真）并**写独立的 `figs_only.log`**，**绝不覆盖 `run.log`**。⚠️ 归档元数据的 `singular.dt_s` 曾误写 `DT4`(2 ms)，实为 `DT4_REF`(1 ms)，已修 |

> 带 ✅ 者为**已落地并实测通过**（阶段 0/1：2026-09-18；阶段 2：2026-09-20 用修正积分器重跑，4/4 PASS，
> 见书仓库日志「尝试 13」；**阶段 3：见日志「尝试 12」，6 条判据 6/6 PASS**）；其余即**待建/待归档清单**。
> 新增脚本请同步更新本表。
> 每个测试脚本都暴露 `run(...) -> dict`（结论结构化）与 `main()`（打印 + 退出码），
> 既可单独跑，也可由 `run_stage0.py` / `run_stage2.py` / `run_stage3.py` 这类一键入口汇总。
> 四个 `ctrl_*` 测试脚本都支持 `--quick` 冒烟模式（压缩时长与增益点，**判据结论无意义**、只看链路跑通）。

### `scripts/` —— 一次性编排脚本

| 文件 | 用途 |
|---|---|
| `probe_fit_dt.py` | ✅ 阶段 3：**判据 1 的步长定点探针**——把同一条误差动态实验在多个 `dt` 上重跑并逐分量辨识 `(ξ̂, ω̂_n)`，用来判断该判据的偏差是「控制律问题」还是「离散化残差」（结论：`O(dt)` 离散化）。⚠️ **会跑仿真**，一次约 9 min；窗口与档位用 `--T` / `--dts` **显式**指定（`T_2ND` 现值已是 3.0，用默认值会把历史上 T=2.4 的那个文件语义覆盖掉）。产出进 `res/6r_stage3/probe_fit_dt*.json` |
| `smoke_figures.py` | ✅ 阶段 3 起：用**合成数据**几秒钟把 `stage3_lib.make_figures` 的 7 条出图路径跑一遍（`--keep` 保留图肉眼检查；配 `-W error::UserWarning` 可把字体缺字形告警变成硬失败）。改过任何出图函数后先跑它 |
| `summarize_stage3.py` | ✅ 阶段 3 起：把 `res/6r_stage3/result.json` 压成**人读摘要**（**必须带 `--md <路径>` 才会写文件**，否则只打印到 stdout）。⚠️ 它的探针表取自 `result.json` 的**内嵌** `fit_dt_probe`（不是直接读探针文件）——重跑探针后必须先跑 `refresh_archive_probe.py`，否则 summary 印旧数 |
| `refresh_archive_probe.py` | ✅ **阶段 4 期间新增（2026-09-20，见日志尝试 13）**：把 `res/6r_stage3/probe_fit_dt*.json` **重新嵌回** `result.json` 的 `fit_dt_probe` 键。**为什么需要**：`run_stage3._load_probe(out_dir)` 是**从磁盘读**探针文件写进归档的，所以只重跑探针**不会**更新内嵌副本，而 `summarize_stage3.py` 恰恰从内嵌副本打印 ⇒ 不刷新就会得到「`summary.md` 旧数 / 独立探针文件新数」的**自相矛盾归档**。会追加一条 `post_run_refreshes` 记录（时间戳 + 旧/新数值对照）并默认备份 `result.json.probe-refresh.bak`；`--dry-run` 只比对不写。固定顺序：**重跑探针 → 回填 → 跑 summarize** |
| `restore_stage4_log.py` | ✅ **阶段 4 新增（2026-09-20，见日志尝试 14）**：**由 `res/6r_stage4/result.json` 重建 `run.log`**（秒级，只读归档 + 写日志）。**什么时候需要它**：`run_stage4.py` 曾把 `--figs-only` 也写进 `run.log`（`"w"` 打开），把那次 64.6 min 完整跑的 stdout 唯一留档清空过；脚本已修（`--figs-only` 改写 `figs_only.log`），本脚本负责恢复主日志。**之所以能重建**：归档里存着每个 `print` 用到的全部字段，日志是归档的**确定性函数**（数字逐位相同）；**并行进度行的耗时数字无法恢复**，重建版省略并在文件头声明。 |
| `smoke_stage4_figures.py` | ✅ **阶段 4 新增（2026-09-20）**：**扫描「图上真正被渲染的文字」是否含缺字形字符**——monkeypatch `matplotlib.text.Text.set_text`，把交给 matplotlib 的每个字符串里命中「危险字符集」（`⇒ ⚠️ ε ⁻¹`、组合变音符 U+0302/0303/0304/0307、上标数字…）的全部打出来。**为什么不能只 grep 源码**：危险字符可能来自归档数据或格式化串，且**注释里的字符不进图**——按源码扫描会大量误报。配 `-W error::UserWarning` 可把 matplotlib 自己的缺字形告警变成硬失败。 |

> 上表中只有 `summarize_stage3.py`（只读归档）与 `refresh_archive_probe.py`（只读写 JSON）是**秒级**；
> `probe_fit_dt.py` **会跑仿真**（一次约 9 min），`smoke_figures.py` 用合成数据（秒级）。

### 其他

- `res/` —— 实验产物：`res/<用例>_stage<N>/`，含 `result.json`（机读）+ 图 + `run.log`；
  **这是验证报告里每个数字的依据，勿删**；整体不纳入版本控制，目录契约与复现要求见 `res/README.md`。
- `report/` —— ★ **验证报告（本批次的主要交付物，✅ 2026-09-20 完成 v2.1 / LaTeX）**：
  `report.tex`（**自包含**，XeLaTeX 连续编译两次）+ `report.pdf`（65 页）+ `README.md` + `figs/`（**20 张**）。
  **该目录例外纳入版本控制**（见仓库根 `.gitignore`；只忽略 `*.aux/*.log/*.out/*.toc` 等中间产物）。
  报告结构与必填小节见计划 §6.2；**每个数字都要能追溯到 `res/` 归档**，
  且「负结果、未做项与局限」是**独立硬章节**（12 条负结果 + 8 项未做 + 8 条局限）。
  编译方式与排版约束见 `report/README.md`。
- `scripts/` —— 四个一次性工具（见上表）：`summarize_stage3.py` / `refresh_archive_probe.py` 秒级且只读写归档，
  `smoke_figures.py` 用合成数据（秒级），`probe_fit_dt.py` **会跑仿真**（约 9 min/次）。
  `test/ctrl_js|ctrl_os|ctrl_common` 的 `.gitkeep` 已随阶段 2 的真实脚本落地删除。
- **报告用图**写 `report/figs/`；**不往书仓库 `1-MN4R/imgs/` 放图**（本批次结果不进正文）。

## 关键约定

1. **第一批不引入第三方仿真引擎**：主验证回路是**自建 NumPy 仿真**（RK4 + `model/` 的模型库）。
   `Pinocchio` 仅用于**阶段 1 的模型权威对拍**（可选、按需安装）；**本批次不引入 MuJoCo/PyBullet**——
   其接触模型与书中力控的物理不同，且运动控制不需要接触。
   理由与取舍见书仓库 `docs/robot-control-verify/part4-motion-control-plan.md`。
2. **绝不用同一个模型同时当被控对象与控制器模型**（除非是有意为之的「自洽」用例，且必须标注）。
   两者同源时，即使模型有错，闭环误差动态也会「看起来完美」，验证退化为自证。
   阶段 1（外部对拍）与阶段 4（故意失配）就是为此设置的对照。
3. **控制律与书中的公式、变量名严格同名**（与 `Codes/` 的既有风格一致）：
   `K_P`/`K_D`/`F_f`/`B`/`C`/`g`/`J_a`/`x_e`/`q_d` 等一律与正文一致，便于「算法 ↔ 代码」对照。
4. **数组索引约定**：与全书一致——长度为 `N+1`、下标 `1..N` 与数学记号对齐，`[0]` 不使用。
   新增代码沿用，勿改成 0-based（会与 `model/` 的既有实现冲突）。
5. **参考轨迹的生成红线**（易无意识违反）：
   - **JS 用例**：在关节空间生成 `q_d(t)` 及其各阶导数。
   - **OS 用例**：**必须在操作空间生成轨迹**（$x_d(t)$ 及各阶导数）。可以用 `robot_ik_*` 逆解出 `q_d(t)`
     作为**真值参照**，但**绝不能把逆解出的 `q_d` 喂给 OS 控制器**——那等价于把 OS 控制偷偷变成 JS 控制，
     期望曲线与实测曲线会因共享逆解而重合，验证失效。
6. **任务空间误差范数口径**：$x$ 混合平移（m）与旋转（rad），**不可直接取欧氏范数**。
   分开报告 `‖x̃_pos‖`、`‖x̃_rot‖`；需要标量时用带等效臂长 `ℓ`（默认 0.5 m）的加权范数，
   **并在文档与图注中显式声明 `ℓ` 的取值**。
7. **每组实验固定随机种子、≥20 组初值，报告「均值 ± 标准差」**，不拿单条曲线下结论。
8. **产物、归档与报告**：`res/` 整体不纳入版本控制（见仓库根 `.gitignore`），
   但 **`report/` 例外**（验证报告是长期交付物，只忽略 LaTeX 中间产物）。
   `res/<用例>_stage<N>/` 是报告里每个数字的依据，不要删。
   改动了口径或参数导致数字变化，必须同步书仓库 `docs/robot-control-verify/part4-motion-control-log.md`
   与**验证报告**（`report/report.tex`）。
9. **结果不进书稿正文**（与 CartPole 的区别）：不改 `Part_4_robot_control/chap4_*.tex`、
   不往 `1-MN4R/imgs/` 放图、不回填正文数字、不重编 `main.pdf`。
   书侧只更新 `2index.tex` 的 `$\dag$` 与 `unfinished_contents.md` 的清单。
10. **提交 / 推送只在用户明确要求时进行**（与全仓库一致）。

## 运行

所有命令中的 `python` 均指 `E:\Anaconda3\envs\py311-gym\python.exe`。脚本一律用 `__file__` 解析路径，
**可在任意 cwd 运行**。

```powershell
# 阶段 0：模型自检（一键跑 + 归档，2026-09-18 实测 6/6 PASS，约 30 s）
$env:PYTHONUTF8='1'
python Robot-Control/test/model/run_stage0.py      # → res/model_stage0/{result.json,run.log}

# 也可单独跑（排查时用；三个脚本都可用任意 cwd 运行）
python Robot-Control/test/model/test_model_sync.py
python Robot-Control/test/model/test_jacobian_dot.py
python Robot-Control/test/model/test_convention.py

# 阶段 1：与 Pinocchio 对拍。Pinocchio 不在 py311-gym 里，而在工作区本地 env：
#   D:\Projects\2024_MN4R\.envs\py311-pin\python.exe   （实测 pinocchio 4.1.0 / python 3.11.16 / numpy 2.4.6）
# 创建方式（conda-forge，包缓存也放工作区内，不动 py311-gym；实测下载 257.8 MB）：
#   $env:CONDA_PKGS_DIRS='D:\Projects\2024_MN4R\.conda-pkgs'
#   E:\Anaconda3\Scripts\conda.exe create -p D:\Projects\2024_MN4R\.envs\py311-pin `
#       --override-channels -c conda-forge python=3.11 pinocchio
#   ⚠️ conda 4.11 会先用 current_repodata 解析失败一次再回退（耗时数分钟）；结束时返回 exit code 1
#      仅因沙箱不让写全局注册表 %USERPROFILE%\.conda\environments.txt，环境本身可用、按路径调用即可。
& D:\Projects\2024_MN4R\.envs\py311-pin\python.exe Robot-Control/test/model/test_model_pinocchio.py

# 阶段 2：仿真层 + 2R 用例跑通（一键跑 + 汇总计划 §5.2.5 判据表 + 归档）
python Robot-Control/test/ctrl_common/run_stage2.py            # → res/2r_stage2/{result.json,run.log,figs/}
python Robot-Control/test/ctrl_common/run_stage2.py --quick    # 冒烟（约 1.5 min，只看链路跑通）

# 也可单独跑一个算法（四条判据的口径完全一致，见 verify_lib.py）
python Robot-Control/test/ctrl_js/test_js_pd.py        --quick    # JS 重力补偿 PD
python Robot-Control/test/ctrl_js/test_js_invdyn.py    --quick    # JS 逆动力学 PD
python Robot-Control/test/ctrl_os/test_os_pd.py        --quick    # OS 重力补偿 PD
python Robot-Control/test/ctrl_os/test_os_invdyn.py    --quick    # OS 逆动力学 PD

# 阶段 3：6R 正式验证与专项测试（一键跑 + 汇总计划 §5.3.7 的 6 条判据 + 归档）
#   2026-09-20（**修正积分器后**）实测：**PASS（6/6），41.7 min**（--workers 6）→ res/6r_stage3/
python Robot-Control/test/ctrl_common/run_stage3.py            # → res/6r_stage3/{result.json,run.log,figs/}
python Robot-Control/test/ctrl_common/run_stage3.py --quick    # 冒烟（只看链路跑通；判据结论无意义）
python Robot-Control/test/ctrl_common/run_stage3.py --only kp,dt    # 只跑指定实验（调试用）
python Robot-Control/test/ctrl_common/run_stage3.py --workers 6     # 指定并发（默认 min(6, CPU-2)）

# 单算法跑阶段 3（算法内判据：3 收敛 + 1 误差动态 / 2 无源性）
python Robot-Control/test/ctrl_js/test_js_invdyn.py    --stage3   # 判据 1 + 3
python Robot-Control/test/ctrl_js/test_js_pd.py        --stage3   # 判据 2 + 3
python Robot-Control/test/ctrl_os/test_os_invdyn.py    --stage3   # 判据 1 + 3
python Robot-Control/test/ctrl_os/test_os_pd.py        --stage3   # 判据 3

# 阶段 3 的辅助工具（summarize / refresh 秒级；probe_fit_dt **会跑仿真**，约 9 min/次）
# ⚠️ probe 的窗口与档位要**显式**给：T_2ND 现值是 3.0，用默认值会覆盖掉 T=2.4 那个文件的语义
python Robot-Control/scripts/probe_fit_dt.py --T 2.4 --dts 2e-3,1e-3,5e-4 --out res/6r_stage3/probe_fit_dt.json
python Robot-Control/scripts/probe_fit_dt.py --T 3.0 --dts 2e-3,1e-3 --out res/6r_stage3/probe_fit_dt_T30.json
python Robot-Control/scripts/refresh_archive_probe.py           # 重跑探针后**必须**回填 result.json 的内嵌副本
python Robot-Control/scripts/summarize_stage3.py --md res/6r_stage3/summary.md
python Robot-Control/scripts/smoke_figures.py                  # 出图链路烟雾测试（7 张图）

# 阶段 4：鲁棒性与边界（✅ **2026-09-20 实测 3/3 PASS，64.6 min**，见书仓库日志尝试 14）
python Robot-Control/test/ctrl_common/run_stage4.py            # 一键跑 + 汇总计划 §5.4.3 的 3 条判据 + 归档
python Robot-Control/test/ctrl_common/run_stage4.py --quick    # 冒烟（只看链路跑通；判据结论无意义）
python Robot-Control/test/ctrl_common/run_stage4.py --only sing,noise   # 只跑指定实验
python Robot-Control/test/ctrl_common/run_stage4.py --figs-only         # 只由归档重建图集（秒级，不跑仿真）
python Robot-Control/scripts/smoke_stage4_figures.py                     # 扫「图上文字」的缺字形字符（秒级）
python Robot-Control/scripts/restore_stage4_log.py                       # 由归档重建 run.log（秒级）
python Robot-Control/test/ctrl_common/test_robust.py --only mismatch    # 单项入口（mismatch/friction/noise/period/instab/cost）
python Robot-Control/test/ctrl_os/test_os_singular.py --quick           # 奇异边界（单项）
```

> ⚠️ **阶段 4 的实测成本**：完整一键跑 `--workers 6` 一次 **3877.1 s（64.6 min）**。分解：
> 奇异 903 s（36 条规格）、模型失配 350 s、摩擦 623 s（28 条）、噪声 313 s、控制周期 134 s、
> 离散化失稳（`instab`，约 8 min，串行 in-process 的对数二分）、耗时项秒级。
> ⚠️ **改过任何出图代码后**依次跑：`smoke_stage4_figures.py`（审字符，秒级）→ `run_stage4.py --figs-only`
> （重建图，秒级，**不跑仿真**）；**不要**为此重跑 64.6 min。
> ⚠️ `--figs-only` 与 `restore_stage4_log.py` 都**只读写归档**，不改任何实验数字。

> ⚠️ **阶段 3 的实测成本（供阶段 4 预算）**：
> 2026-09-19 旧（一阶）积分器：完整跑 `--workers 6` 一次 **3486.6 s（58.1 min）**；
> **2026-09-20 修正积分器后重跑：2504.6 s（41.7 min）**——四阶 RK4 步长更准，同样的 `T`/`dt` 反而更快。
> 旧的分解（供参考）：四个算法的 20 组初值批次合计约 1680 s
> （`js_pd` 480.6 s、`js_invdyn` 607.3 s、`os_pd` 587.6 s，单组有效 24–30 s），
> 判据 1（`dt = 1 ms`、`T = 3.0 s`）每个逆动力学算法约 200 s，`step_size_check` 约 600 s。
> 步长探针：旧（一阶）积分器 `T = 2.4` 一次 **521 s**；**修正积分器后为 367 s（T=2.4，三档）/ 207 s（T=3.0，两档）**，已在尝试 13 刷新。

> ⚠️ **运行成本与并行（2026-09-19 实测，`py311-gym`；动手跑长任务前必读）**
>
> - **单步成本**：一次 2R 闭环每毫秒仿真要算 **4 次 `robot_dyn`**（RK4 四级；控制器那一次被
>   `RobotCase.dyn` 的单条 memo 吸收）→ **~4.0 ms 墙钟 / ms 仿真**；单组 3.5 s 调节 = **14.3 s**。
> - **实测总时长**：串行跑完阶段 2 全部检查 **1400.5 s（23.34 min）**；
>   `--workers 6` 并行后 **781.7 s（13.03 min）= 1.79×**，且**逐元素与串行完全一致**；
>   **2026-09-20 修正积分器后重跑为 607.8 s（10.1 min），4/4 PASS**。
> - ⚠️ **本机实际可用算力约 2–2.5 核（尽管报告 8 核）**：两个与本项目无关的基准都如此——
>   ① 纯 Python 忙等 1/2/4/6/8 路并发 → 总吞吐 0.69/1.09/1.65/1.82/1.76 任务/s（8 路时单进程慢 3×）；
>   ② 进程内 BLAS 线程 1/2/4/8 → 2000³ 矩阵乘 0.323/0.313/0.332/0.330 s（无加速）。
>   ⇒ **加 worker 数无益**，6 路已饱和；想再快只能减少工作量（缩 `T`、放宽 `dt`）或优化热点代码。
> - ⚠️ **`multiprocessing` 在本沙箱下不可用**：禁止创建命名管道，`mp.Pool` 直接报
>   `PermissionError: [WinError 5]`。并行驱动改用「**子进程 + 文件重定向**」
>   （`test/ctrl_common/run_ic_worker.py`：任务走 spec JSON、结果走 pickle、`stdio` 指向文件，**无任何管道**），
>   子进程启动 ~0.16 s，且被**钉成 BLAS 单线程**（否则子进程各自开线程、8 核上严重超订，实测净加速掉到 1.7×）。
> - **热点**：profile 显示 **`np.cross` 占 51%**（`model/code_jacobian.py` 的 `robot_j_centroid` 里，
>   3 元素叉乘走 numpy 通用通道 ~50 µs/次）。手写三元叉乘可省 ~1.4–1.5×，但**要改 Part II 已 `$\dag$` 的原件**
>   （`Codes/chap2_3_diffk/code_jacobian.py`）→ 按「开放问题 4」的同类处理**须先取得授权**，当前**未动**。
> - **务必后台跑并看进度日志**（并行时每完成一条打一行；`--workers N` / `--serial` 可控并发）。
>   归档见 `res/2r_stage2/`；`--quick` 模式压缩时长与增益点，**判据结论无意义**。
> - ⚠️ **6R 的预算**（阶段 3 用）：`robot_dyn` 单次 **10.9 ms** → 单步 ~44 ms → **一组 3.5 s 调节约 150 s**，
>   20 组 × 4 算法串行 ≈ **3.4 h**。阶段 3 开工前先把「缩 `T` / 放宽 `dt` / 是否授权 `np.cross` 优化」定下来。

> ⚠️ 阶段 0 的脚本**不依赖 pinocchio**；**阶段 1 已确定用 `py311-pin` 解释器跑**
> （`py311-gym` 里 `import pinocchio` 会失败 → 脚本自动跳过并给出提示）。
> 2026-09-18 实测：阶段 1 一键跑约 **2–4 分钟**（含装配拟合的多起点最小二乘），
> 归档 `res/model_stage1/{result.json,run.log}`，5 项对拍 + 失配自检全部 PASS。
>
> ⚠️ 阶段 0 的实测耗时：`robot_dyn` 单次 6R 约 **10.9 ms**（内部 1+6 次 `robot_B` 差分 `∂B/∂q`）。
> 阶段 2 的仿真层若用 `h_sim = 1 ms`，单线程 10 s 仿真就要 ~7×10⁴ 次动力学求值——**先做步长/耗时预算**，
> 长积分务必打**实时进度日志**（`test_convention.py` 的 `simulate(..., progress=True)` 是现成写法）。

> **复现报告里数字的硬要求**：① 用 `py311-gym`；② 显式给定随机种子与控制器增益（`ω_n`、`ξ` 等）；
> ③ 记下用例名（2R / 6R）。书仓库的验证日志与**验证报告**里逐条记录这些参数。

**阶段 5（报告）**：✅ **已完成（2026-09-20，日志尝试 15；v2.1 排版迁移见下）**——报告
`report/report.tex`（**LaTeX，自包含**）＋ `report.pdf`（65 页）＋ `README.md` ＋ `figs/×20`。
内容按计划 §6.2 的模板，**自包含**（读者不看计划与日志也能读懂），
「负结果、未做项与局限」小节列了 12 条负结果 + 8 项未做 + 8 条局限声明，
并显式区分「书中已声明的前提」与「验证发现的缺陷」。
⚠️ 报告里的数字**一律由 `res/` 归档现取**（不抄日志——本轮正因此抓到 1 处归档元数据笔误与 1 处日志抄写错误）。
📌 **v2.1 排版迁移**：报告由 Markdown 重写为 XeLaTeX（`ctexart`，`xelatex` 连续编译两次）；
图题统一 ≤15 个汉字、图的解释全部写进正文并以「如图 $X$ 所示……」关联；原 `.md` 已删除。
**技术内容、全部实测数字与四项算法判定与 v2.0 完全一致。**

**阶段 6（书侧索引）**：✅ **已完成（2026-09-20，日志尝试 16）**——`2index.tex` 的 4 个算法升 `$\dag$`、
`unfinished_contents.md` 第 2 节 12 项 → 8 项；**未改任何正文 `.tex`、未插图、未重编 PDF**
（`git diff --stat -- Part_4_robot_control/` 为空）。

## 已知问题 / 未完成

- 🔥 **2026-09-20 发现仿真层真 bug：`code_sim.rk4_step` 的 `q` 更新只有一阶精度**（见书仓库日志
  **尝试 12**）。`q` 的权重误写成 `dt²/6·(a1 + 2a2 + a3)`（多算一个 `a2`；正确是 `a1 + a2 + a3`，
  `a2` 系数为 **1**），`test/model/test_convention.py` 里的同名副本同样写错，**两处均已修**。
  实测全局收敛阶 **1.000 → 3.99**、同一算例误差（`h = 1e-2`）`6.198e-03 → 4.581e-09`（差 1.35e6 倍）、
  稳定界恢复 RK4 理论值 `hω* = 2.785`。**尝试 11 那个 `hω* ≈ 0.845` 的 3.3× 之谜就是这个 bug**，
  不是「检测器保守」也不是「带宽被 `cond(B)` 拉开」。
  - **影响面：阶段 2/3/4 的全部数字作废**。处置：修好后**整体重跑**，旧归档留档为
    `res/*/pre_rk4fix_result.json` + `pre_rk4fix_run.log`（**勿删**）。
    **重跑已全部完成（2026-09-20）**：阶段 0 **PASS 7/7**、阶段 2 **PASS 4/4**（607.8 s）、
    阶段 3 **PASS 6/6**（2504.6 s）＋ `res/6r_stage3/` 的 `probe_fit_dt*.json` / `summary.md`
    与 `result.json` 的内嵌 `fit_dt_probe` 均已刷新（见日志尝试 13）。**当前无待重跑项。**
  - **新增防线**：阶段 0 **判据 7「积分器收敛阶」**（`test_convention.check_integrator_order`，
    经验阶必须 ∈[3.5, 4.5]），已接进 `run_stage0.py` 的 `JUDGE` 与计划 §5.0.5。
  - **教训**：**理论值与实测差 3 倍以上时必须当 bug 查，不许留作「可能的口径问题」。**
    此前 10 轮没抓到，是因为阶段 2/3 的判据都是「稳态误差/收敛/定性趋势/数值恒等式」，
    **一个欠精度积分器照样全过**；判据 2 当时测到的 `Σ正增量 = 0` 更是**一阶过度耗散的伪影**
    （容差已由 `1e-10` 放宽到 `1e-6`，见下方阶段 3 条目与计划 §5.3.7 修订块 C）。
- ✅ **阶段 4 已完成并通过（2026-09-20，见书仓库日志「尝试 14」）**：代码全部落地
  （`stage4_lib.py` / `run_stage4.py` / `test_robust.py` / `test_os_singular.py`，
  以及 `code_sim`（`noise=`、`make_friction_tau_e`、**记录 `u−g`**）、
  `code_models`（`make_case(..., plant=…)`）、`verify_lib`（`plant/friction/noise` 规格）的扩展）。
  **完整一键跑 `run_stage4.py --workers 6`：计划 §5.4.3 的 3 条判据 3/3 PASS，用时 3877.1 s（64.6 min）**，
  退出码 0；归档 `res/6r_stage4/{result.json, run.log, figs/×9}`。
  | 判据 | 承担的逐项 | 实测关键数字 |
  |---|---|---|
  | **1** 每项给出「误差随扰动量的量级关系」 | `model_mismatch` / `friction` / `noise` / `control_period` **四项全 PASS** | 质量斜率 `0.934–1.066`（理论 1）、实测/闭式预测 `0.51–1.28`；惯量族长时段末值 `1.4e-6 ~ 1.1e-4 rad` 仍在衰减（**无稳态偏移**）；噪声一阶预测/实测 `0.95–1.02`、`q_diff` 比直接测速劣 `142×`、低通降 `8.2×`；控制周期逆动力学类斜率 `0.991 / 1.001`，两个 PD 类 `0.001`（如实报告） |
  | **2** 奇异边界给出明确失效点 | `singularity` **PASS** | `os_pd` 首个失效 `σ_min = 5.0e-2`（**推不动**：9/10 档误差不衰减、指令 0 次爆掉）；`os_invdyn` `σ_min = 1.0e-4`（**力矩爆**：`282.6 N·m > 200`、`‖q̇‖∞ = 36.1 rad/s`）；JS 对照 **0/6**、`os_pd` 的 `in_range` **0/5**；`in_range` 失效 0/10 vs `out_of_range` 11/20 |
  | **3** 离散化失稳 vs 连续律正确性 | `discretization` **PASS** | `ω*(2 ms) = 666.8`、`ω*(0.5 ms) = 2108.5`（比值 0.32 vs 理论 4.00）；`hω* = 1.334 vs 1.054`，相对差 **20.9% < 50%**；四算法阈值**完全相同**（同一 bracket `[500, 889.1]` / `[1581.1, 2811.7]`）；ZOH 参照 `2.156`、连续反馈参照 `2.785` |
  - ⚠️ **两处必须进报告的负结果/偏差**（本轮实测新增）：
    ① **`os_pd` 的放大律斜率只测到 `−0.011`**（理论 1）——**不是判据失效**，而是它的失效模式本就是
    「退化方向**推不动**、误差不衰减」，反馈力矩始终有界（`3.5e-01 N·m` 量级），故该斜率无意义；
    ② **步长归因那一个定点（`os_pd`、`σ_min = 5e-4`）的 `‖u−g‖∞` 随 `dt` 减半反而 `1.029e+02 → 1.359e+02`**
    （比值 `0.758`，与「`dt` 减半应下降」的直觉相反）——该点落在**失效边界附近**，机理对照
    （判据 3 的线性冻结模型）表明这属积分器/采样效应而非控制律问题，报告需单独说明。
    ③ **判据 3 的机理对照只取到 `dt = 2 ms` 一档**（`frozen_linear_mechanism_check`：四算法
    `largest_stable_wn = null`、`smallest_unstable_wn = 625`，只能给出单侧界）——**如实记录**，
    不影响判据 3 的 PASS（它的条件是「≥2 个算法给出阈值 + 两档 `hω*` 相对差 < 50%」）。
  - ⚠️ **`--quick` 的结论不能当结论**：`T` 被压到 0.3 s 时摩擦守卫必然报异常、分离比只剩 1.4×。
  - ⚠️ **完整跑踩到的两处出图坑（都已修，且不再需要重跑仿真）**：
    ① `stage4_lib` 的 mathtext `$\\sqrt2$` 缺花括号 → 完整跑在出图第 4 张处抛
    `ParseSyntaxException`，9 张图只画出前 6 张（**归档已先落盘，数据未受影响**）；
    ② 修 `\sqrt2` 后又暴露两类：图上「⚠️ / ε / ⇒ / ⁻¹」在 Microsoft YaHei **缺字形**、
    以及新加的 `code_metrics._wrap_caption` 折行会**切断 `$...$`** 使残段被当数学模式渲染成乱码。
    最终口径：**图注一律纯文本 + 仅用已验证有字形的 Unicode，不写 `$...$`**。
    ⇒ 为此新增 **`run_stage4.py --figs-only`**：**只由已落盘的 `result.json` 重建全部图**（秒级，
    **不跑仿真**），并回写 `figures` / `figure_error` / `post_run_refreshes`。
    ⚠️ 它与主日志的关系有硬规矩：`--figs-only` 写**独立的 `figs_only.log`**，
    **绝不覆盖 `run.log`**（2026-09-20 实际踩过：主日志是 64.6 min 那次跑的 stdout 唯一留档，
    被辅佐流程以 `"w"` 清空过；`scripts/restore_stage4_log.py` 负责由归档重建它）。
  - ✅ **阶段 5、6 亦已完成（2026-09-20，日志尝试 15 / 16；报告于 2026-09-20 迁至 `report/` 并改为 LaTeX）**：
    验证报告已成文（`report/report.tex` + `report.pdf` + `README.md` + `figs/×20`）；
    书侧 `2index.tex` 4 个算法升 `$\dag$`、`unfinished_contents.md` 第 2 节 12 项 → 8 项，
    正文 `Part_4_robot_control/` **零改动**（`git diff --stat` 已核验）。
    **本批次（Part IV 运动控制 4 个算法）至此全部收口**，无待办项。
  - ⚠️ **阶段 5 顺带修掉的一处元数据笔误**：`res/6r_stage4/result.json` 的
    `experiment.noise.dt_s` 曾写成 `DT4`(2 ms)，实际噪声实验用 `DT4_REF`(1 ms)
    （与 `singular.dt_s` 同类；奇异那处在阶段 4 收口时已修、噪声这处漏了）。
    已改代码 + 归档（含 `post_run_refreshes` 记录与 `result.json.noise-dt.bak` 备份），
    **只改元数据、未重跑仿真**，全部实测数字不变。
- ✅ **阶段 3 已完成并通过（2026-09-20 用修正积分器重跑，见书仓库日志「尝试 12」）**：
  6R 上四个算法的正式验证，**计划 §5.3.7 的 6 条判据 6/6 PASS**，完整跑 **2504.6 s（41.7 min）**，
  归档 `res/6r_stage3/`（`result.json` + `run.log` + `figs/×7`）。
  ✅ `summary.md`、`probe_fit_dt*.json` 与 `result.json` 的内嵌 `fit_dt_probe` 均已用修正积分器刷新
  （2026-09-20，日志尝试 13）；**T = 3.0 / dt = 1 ms 的探针值与归档判据 1 逐位一致**
  （`js_invdyn` 的 `q̃_6` `1.2764%` = `1.2764%`；`os_invdyn` 的 `x̃_1` `0.5563%` = `0.5563%`）。
  关键实测：判据 1 `ξ̂/ω̂_n` 最差 **1.2764%**（`js_invdyn` 的 `q̃_6`，其余 ≤0.70%）/
  `os_invdyn` 最差 **0.5563%**；判据 2 `Σ正增量/V(0) = 8.15e-10`（`js_pd`）/ `1.02e-09`（`js_invdyn`）；
  判据 3a 四算法 **20/20** 全部收敛（最差 `1.068e-06` rad）、3b `os_pd` 2 组边界样本（最差 `6.728e+46`）；
  判据 4 分工 **130.0× / 136.3×**；判据 5 耦合抑制 **63.1×**（步长归因降 2.00×）；
  判据 6 映射 `3.586e-16` / `3.318e-15`；§3.5 `K_P` 斜率 `-1.0244 / -0.9735`。
  ⚠️ 旧归档的数字（判据 1 `1.4433%`、判据 2 `Σ正增量 = 0`、判据 4 `131.2×/148.6×`、判据 5 `63.2×`）
  **出自有 bug 的积分器，已作废**，留档在 `pre_rk4fix_result.json`。
  - ⚠️ **两处判据口径修订（阈值一字未动，已同步进计划 §5.3.7）**：
    1. **判据 1 在 `dt = 1 ms` 上判定**（计划原定默认步长），不是阶段 3 主步长 2 ms：
       2 ms 下最小惯量关节 `q̃_6` 的 `ξ̂` 有 `+2.9%` 系统偏差；步长探针
       （`scripts/probe_fit_dt.py`，`res/6r_stage3/probe_fit_dt.json`）实测该偏差为 **`O(dt)`**
       （刷新后为 `2.60% → 1.24% → 0.61%`，`dt` = 2 / 1 / 0.5 ms，经验阶 **`js_invdyn` 1.03–1.07 /
       `os_invdyn` 1.00**——⚠️ 日志尝试 13 曾把 js 的阶误抄到 os 行，以探针文件现算值为准；
       判定窗口 `T = 3.0` 两档为 `2.75% → 1.28%`，经验阶 `js_invdyn` 1.11 / `os_invdyn` 1.00）
       ⇒ 是**零阶保持 + RK4 的离散化残差，不是控制律或书的公式错误**。观测窗同步由
       `T = 2.4 s` 改 **`3.0 s`**（按**正峰个数**取窗，保证 3 个正峰）。
    2. **判据 3 拆 3a/3b**：`code_sim` 增记 `σ_max(J_a)`（与 `σ_min` 同一次 SVD）→ `cond_J_max`；
       3a = 轨迹全程 `cond(J_a) ≤ 100`（书中「`J_a` 列满秩」前提的定量化，标称位形实测 `cond = 12.70`）
       的初值**必须全部收敛**；3b = 前提被违反的初值（`os_pd` 的 2 组在瞬态掠过奇异区：
       `cond_J_max = 6.5e4` / `1.29e2`）如实单列为边界样本。归档含逐组明细 `ic_rows`。
  - **开放问题 4 结案**：`C` **不必升级**为解析 Christoffel——控制器与被控对象同源时 `C`、`g`
    在采样点上精确抵消（`q̈ ≡ y`），其近似精度不进入标称用例的闭环。
  - ⚠️ **两次失败留档（勿删）**：`run_attempt1.log`（首次完整跑，两处判据口径问题的原始证据）、
    `run_attempt2_crash.log`（第二次完整跑，**63.7 min 算完后在出图处抛 `KeyError: '_raw'`**，
    归档没写出来）。
  - ⚠️ **由此确立的硬规矩**：**`run_stage3.main` 先写 `result.json`、再出图，且出图整体包在 `try` 里**
    （归档是主要产出，绝不能排在装饰性步骤之后）；`algorithm_stage3` 的 `extras` **整份保留 `_raw`**
    （写归档时 `_strip` 会递归丢掉下划线键）；改过任何出图代码后先跑
    `scripts/smoke_figures.py`（配 `-W error::UserWarning`，能抓出 `ξ̂`/`ω̂` 这类字体缺字形的坑）。
  - ⚠️ **一处必须写进报告的局限**：判据 1 的判定值 `1.2764%` 离 2% 阈值只有 **1.6 倍余量**，
    且它是**系统性**的 `O(dt)` 偏差（不是随机噪声）——报告须同时给出 `dt = 2 ms` 与 `dt = 0.5 ms`
    两点。✅ 这两个步长点已刷新（`dt = 2 ms` 为 `2.75%`、`dt = 0.5 ms` 为 `0.61%`，
    均出自修正积分器后的探针）。
- **阶段 0 已完成（2026-09-18）**：`run_stage0.py` 一键跑通，6 条判据全中（判据 5 按实测修订为 5a–5d）。
- **阶段 1 已完成并通过（2026-09-18，见书仓库日志「尝试 7」）**：
  产出 `test/model/test_model_pinocchio.py` + 归档 `res/model_stage1/{result.json,run.log}`。
  判据 5 项全部机器精度（6R：`FK 5.6e-16`、`B 1.0e-15`、`g 8.0e-16`、`Cq̇ 1.7e-9`、`J_w 3.5e-16`），
  「故意失配 +5% 质量」自检**已检出**（4.4e-2）⇒ **「书模型自身出错」这一假阳性来源已排除**。
  - ⚠️ **更正尝试 6 的两条错误结论**：① 关节复合语义**记反了**——pinocchio 实际是
    **`oM_i = Σ_1Rz(q_1)…Σ_iRz(q_i)`（`Rz` 施加在 `Σ` 之后）**，与书链 `T_i = T_{i-1}A_i(q_i)`
    的复合方向**相反**；② **不存在「书 D-H 错位一格」**——书关节 i 恰好对应 pin 关节 i，
    当时是把「D-H 连杆系原点（远端）」与「关节原点（近端）」混用了。
  - **做法**：不猜下标——把 `Σ_i` 与末端固定变换 `Mf` 当未知常量，在 24 个随机位形上
    最小二乘解出（残差 2.2e-16 / 5.6e-16 = 机器精度）。
  - ⚠️ **`Σ_i = A_i(0)`（最"显然"的取法）不等价于书链**：旋转相同、平移差一次 `Rz(q_i)` 的旋转。
  - ⚠️ **`pin.Inertia(m, c, I)` 的 `I` 是「绕质心」的惯量**（库内部自己做平行轴定理）；
    再额外加 `m(‖c‖²E − c cᵀ)` 会让 `B` 从 1e-15 恶化到 2e-1，**而 `g` 仍然 ~1e-16**——
    即 `g` 通过对拍**不足以**证明惯量挂对了，**必须同时看 `g` 与 `B`**。
- **阶段 2 已完成（2026-09-19，见书仓库日志「尝试 8」）**：`code/` 的仿真层、4 个控制器、度量与出图
  全部落地，四个算法在 2R 上跑通并通过计划 §5.2.5 的四条判据；归档 `res/2r_stage2/`。
  执行顺序见书仓库 `docs/robot-control-verify/part4-motion-control-plan.md` 的「阶段总览」。
  **阶段 3 也已完成**（见上方「阶段 3 已完成并通过」条目）。
- ⚠️ **阶段 2 的关键修订与坑（动手前必读）**：
  1. **2R 的「控制用任务空间」是 2 维位置子任务 `[x, y]`**（不是 `robot_fk_2r` 的 3 维输出）：
     `[x, y, φ]` 对应的 `J_a` 是 3×2 浸入，书里 OS 逆动力学的 `J_a⁻¹` 不存在。见 `code_models.py` 文件头。
  2. **判据 4 原文在本阶段口径下退化**：plant 与控制器模型同源、无摩擦无外力时，稳态误差**理论上恒为 0**
     （平衡条件 `K_Pq̃ = 0`），与 `K_P` 无关 ⇒ 拆为 4a（实测证实「恒为 0」）+ 4b（加常值未补偿扰动
     `τ_e` 后检验 `∝K_P⁻¹` 与单调性）。阈值未动；口径与理由见 `verify_lib.py` 模块文档。
  3. **OS 用例的 `q̃` 在仿真记录里恒为 0**（`ref` 只含 `x_d`，按红线不给 `q_d`）——
     若直接用它判「收敛」会**假通过**。`verify_lib.run_ic_batch` 已在**评测层**用实验者选定的
     `q_d`（`x_d = fk(q_d)`，非逆解）重算 `q̃`，并另附「到精确解流形的距离」（末态逆解，仅评测参照）。
     同类陷阱再次出现——**凡「参照/指标来源」都要检查它是不是真的承载了信息**。
  4. **6R 上 PD 增益必须按参考惯量标定**：`cond(B) ≈ 1.7e3`、`λ_min(B) ≈ 1.15e-3`，
     `K_P = ω_n²I`、`K_D = 2ξω_nI` 会让最小惯量方向的阻尼速率达 ~1.4e4 /s，
     **远超 `dt = 1 ms` 的 RK4 稳定域**（实测 50 ms 内 `‖u‖` 冲到 1e92）——属**离散化失稳**，
     不是控制律错误。解法：`pd_gains(..., Lam=B(q_d))`（OS 用 `Λ = (J_aB⁻¹J_aᵀ)⁻¹`），实测恢复正常量级。
     2R 的 `cond(B) ≈ 12`，`Lam = I` 安全，故阶段 2 的结论不受影响。
  5. **`ω_n` 在四个算法里含义不同**（见 `code_ctrl_common.pd_gains` 的说明）：只有两个逆动力学算法
     的误差动态才是「单位质量」的二阶系统；两个重力补偿 PD 的实际带宽被 `B`（JS）与任务惯量 `Λ`（OS）
     缩放，OS PD 约只有 `0.22·ω_n`，故其任务空间增益要明显更大。阶段 3 记录增益时必须写清空间。
  6. **二阶参数辨识曾把「半周期」当「整周期」（已修）**：`code_metrics.identify_second_order` 原本对
     `|e|` 取峰（振荡信号取绝对值后每半周期出一个峰）→ `T_d` 减半、`ω̂_n` 翻倍、`ξ̂` 偏小；
     实测设定 `ξ=0.3, ω_n=6` 被辨识成 `ξ̂=0.1549, ω̂_n=11.60`（**+93%**）。
     现改为对**带符号误差**取正的极大值：合成二阶系统偏差 ≤0.012%，实际运行 0.27%/0.44%。
     ⚠️ **阶段 3.1 的主判据（2%）直接依赖它**——传参时务必传带符号的误差，不要传 `|e|`。
- **`J̇a` 已补齐**（原缺）：在 `model/code_jacobian_dot.py`（**非副本**）。
  默认 `h=1e-6` 的中心差分，实测相对误差 ~1e-10；⚠️ 差分时**必须随 q 重算欧拉角**
  （`J_a = diag{I₃,T_ϑ(ϑ(q))⁻¹}J_w`），漏了这一步误差会大到量级级别。
- **`robot_jacobian_a` 不支持 2R**（硬编码 6×6，见「模型库」）——2R 用例用 `code_models.py` 里的
  闭式 `robot_fk_2r`/`robot_jacobian_a_2r`/`robot_jacobian_dot_a_2r`。
- **`robot_dyn` 的 `C` 是中心差分近似**（原代码注释已注明「实际工程可用自动微分替代」）：
  这会让逆动力学控制的稳态误差**不真正为零**（残差 `(C_true − C_数)q̇` 作为有界扰动），
  也会让书中的 Lyapunov 等式不严格成立。**阶段 0 已量化**：相对误差随 `h` 按 `O(h²)` 下降，
  `h=1e-5` 处最优（~7.5e-11），**`h=1e-6`（默认值）反而略差**（~1.5e-10，已进入舍入平台）；
  Christoffel 版式本身无结构性错误（参照 `N=Ḃ−2C` 反对称到 ~1e-16）。
  ✅ **是否升级为解析 Christoffel：已结案（阶段 3，2026-09-19）——不必升级**：
  标称用例下控制器与被控对象同源，`C`、`g` 在采样点上精确抵消（`q̈ ≡ y`），
  其近似精度**不进入闭环**；实测的 `ξ̂` 偏差是**离散化的 `O(dt)` 残差**（见日志尝试 10）。
  （升级要改 Part II 已 `$\dag$` 的原件，**没必要付这个代价**。）
- **静平衡测试必须在「稳定」平衡位形上做**（阶段 0 的新发现）：标称位形 `Q_NOM_*` 是**不稳定**重力平衡
  （`λ_min(B⁻¹∂²U/∂q²) = −16.1`（2R）/ `−44.6`（6R）），机器 eps 会按 `e^{sqrt(|λ|)t}` 指数放大
  （实测增长率与预测吻合到 0.7%）。稳定平衡位形见 `code_models.py` 的 `Q_EQ_2R`/`Q_EQ_6R`，
  且在测试里做了阻尼 Newton 抛光——**直接硬编码 10 位有效数字会留下 ~4e-10 的梯度残差**。
- **`F_f`（摩擦矩阵）不在 `model/code_dynamics.py` 里**，书中方程却含该项：
  仿真层 `code_sim.py` 必须显式加入。基线取 `F_f = 0`，阶段 4 再启用。
- **书里 OS 算法的稳定性证明假设 `J_a` 列满秩**：奇异位形附近 `J_a⁻¹` 条件数爆炸。
  书中已声明该前提，**不是错误**；阶段 4 如实报告失效点，阻尼最小二乘只作为「超出书本范围的建议」标注。
- **跨目录「取原件做参照」有静默退化的风险**：同名模块 + `import` 缓存会让对拍变成自证
  （阶段 0 就在 `test_model_sync.py` 里发现过一处，已修）。凡对拍都要有「参照真的来自参照物」的守卫。
- **`robot_dyn` 的耗时是阶段 2 的预算约束**：单次 6R **10.9 ms**、2R 0.97 ms（`robot_B` 0.85/0.16 ms）。
  动态仿真前先算步数×4 的求值次数，长积分要打实时进度日志。

## 其他

- **本文件与仓库根 `../AGENTS.md`**：根文件管整仓库，本文件管 `Robot-Control/` 细节；
  两者冲突以本文件为准。改动本目录结构或关键约定后，顺手核对根文件是否仍然成立。
- **书侧可能读本目录**：书仓库的交互式 Web 文档（`1-MN4R/docs/web/build.py`）目前**不**扫描本目录
  （只扫 `CartPole/`）；若后续要纳入，需同步改该书侧脚本与其 `README.md`。
- 本目录的验证计划、运行日志与判据**都在书仓库**：`../../1-MN4R/docs/robot-control-verify/`。
  **本目录不再保留 `docs/`**（原 `docs/README.md` 只是转发层，2026-09-20 已删），避免两处各写一份而漂移。
