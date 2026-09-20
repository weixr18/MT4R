# -*- coding: utf-8 -*-
"""度量、误差动态参数辨识与出图（计划 §6.1 的度量口径）。

**统一的记录口径**（每个用例、每个算法都一样，便于跨算法对比）：

| 类别 | 指标 |
|---|---|
| 误差 | 关节 `‖q̃‖_∞` / `‖q̃‖₂` 的峰值、RMS、末值；任务空间 `‖x̃_pos‖`、`‖x̃_rot‖`、加权范数（含等效臂长 ℓ） |
| 时间 | `t_90` / `t_10`（误差衰减到初值峰值的 90%/10%）、调节时间（2% 带）、稳态误差 |
| 控制代价 | `‖u‖_∞`、`∫‖u‖²dt`、是否超过用例力矩上限 |
| 数值健康度 | `min σ_min(J_a)`、`max cond(B)`、`max cond(J_a)`、是否 NaN / 抛异常 |
| 统计口径 | 固定随机种子、≥20 组初值，报告**均值 ± 标准差**而非单条曲线 |

**误差符号约定**：`q̃ = q_d − q`、`x̃ = x_d − x_e`（书中一致）。
**任务空间范数口径**（计划 §5.3.6 与 `Robot-Control/AGENTS.md`「关键约定」6）：
`x` 混合平移（m）与旋转（rad），**不可直接取欧氏范数**；分开报告 `‖x̃_pos‖`、`‖x̃_rot‖`，
需要标量时用 `‖diag(1,1,1,ℓ,ℓ,ℓ)⁻¹x̃‖₂ = √(‖x̃_pos‖² + ‖x̃_rot‖²/ℓ²)`，`ℓ = 0.5 m`（`code_models.ELL_DEFAULT`）。

运行环境：`E:\\Anaconda3\\envs\\py311-gym\\python.exe`（加 `PYTHONUTF8=1`）。
"""
import numpy as np

# ⚠️ **matplotlib 惰性导入**（2026-09-19 加）：本模块的指标计算与辨识**不需要**绘图库，
# 但并行执行时每个子进程都会 `import verify_lib → code_metrics`，
# 模块级 `import matplotlib.pyplot` 会让每个子进程多付 ~1 s 的启动成本。
# 故只在真正出图时导入并配置一次（`_pyplot()`），子进程只做仿真时完全不碰 matplotlib。
_PLT = None


def _pyplot():
    global _PLT
    if _PLT is None:
        import matplotlib
        matplotlib.use("Agg")                        # 无显示器环境（验证脚本一律不弹窗）
        import matplotlib.pyplot as plt              # noqa: PLC0415
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        _PLT = plt
    return _PLT

STEADY_FRAC = 0.05      # 稳态窗口：末 5% 时长
SETTLE_BAND = 0.02      # 调节时间：2% 带
T90, T10 = 0.9, 0.1     # t_90 / t_10 的阈值（相对初值误差峰值）


# ---------------------------------------------------------------------------
# 基础指标
# ---------------------------------------------------------------------------
def _vec_norm(traj, axis=-1, ord_=2):
    """按最后一维求范数（`traj` 形状 `(n, d)`；`ord_=np.inf` 为 ∞ 范数）。"""
    return np.linalg.norm(np.asarray(traj, dtype=float), ord=ord_, axis=axis)


def signal_stats(e):
    """标量误差序列 `e(t)` 的 `(峰值, RMS, 稳态均值, 稳态峰值)`；稳态取末 `STEADY_FRAC` 窗口。"""
    e = np.abs(np.asarray(e, dtype=float))
    if e.size == 0:
        return dict(peak=float("nan"), rms=float("nan"), final=float("nan"), steady=float("nan"))
    k = max(1, int(round(e.size * STEADY_FRAC)))
    tail = e[-k:]
    return dict(peak=float(e.max()), rms=float(np.sqrt(np.mean(e ** 2))),
                final=float(e[-1]), steady=float(tail.mean()))


def settle_time(t, e, band=SETTLE_BAND, amp=None):
    """调节时间：误差最后一次超出 `band × amp` 之后的时间（`amp` 默认取初值峰值）。

    本用例的目标误差是 **0**，故 2% 带按**初值误差峰值**取（步响应的常见口径按终值取，
    此处终值为 0，用终值作基准没有意义）——口径在此显式声明。
    """
    t = np.asarray(t, dtype=float)
    e = np.abs(np.asarray(e, dtype=float))
    if e.size == 0:
        return float("nan")
    if amp is None:
        amp = float(e[0])
    if amp <= 0.0:
        return 0.0
    idx = np.nonzero(e > band * amp)[0]
    if idx.size == 0:
        return 0.0
    i = int(idx[-1])
    return float(t[min(i + 1, t.size - 1)])


def decay_times(t, e, fracs=(T90, T10), amp=None):
    """误差衰减到初值峰值 `fracs` 倍所需时间（调节问题的「上升时间」类比量）。"""
    t = np.asarray(t, dtype=float)
    e = np.abs(np.asarray(e, dtype=float))
    if amp is None:
        amp = float(e[0]) if e.size else 0.0
    out = []
    for f in fracs:
        idx = np.nonzero(e <= f * amp)[0] if amp > 0 else np.array([0])
        out.append(float(t[int(idx[0])]) if idx.size else float("nan"))
    return out


def control_cost(t, u, u_max=None):
    """控制代价：`‖u‖_∞`、`∫‖u‖²dt`、是否超过用例力矩上限。"""
    t = np.asarray(t, dtype=float)
    uu = np.asarray(u, dtype=float)
    if uu.ndim == 2 and uu.shape[1] > 1:
        uu = uu[:, 1:]                       # 去掉 [0] 冗余前导元素
    u_inf = float(np.abs(uu).max()) if uu.size else float("nan")
    u_l2 = float(np.trapz(np.sum(uu ** 2, axis=1), t)) if uu.ndim == 2 else float(np.trapz(uu ** 2, t))
    return dict(u_inf=u_inf, u_l2=u_l2,
                u_sat=bool(u_max is not None and u_inf > u_max), u_max=u_max)


def numerics(sigma_min, cond_B, sigma_max=None):
    """数值健康度：`min σ_min(J_a)`、`max cond(B)`、`max cond(J_a) = max σ_max/σ_min`。

    `cond(J_a)` 是阶段 3 判据 3a 的口径（书中 OS 稳定性证明要求 `J_a` **列满秩**，
    3a 把该前提定量化为「轨迹全程 `cond(J_a) ≤ 100`」——见 `stage3_lib.COND_J_PREMISE`）。
    """
    smin = np.asarray(sigma_min, dtype=float)
    cb = np.asarray(cond_B, dtype=float)
    out = dict(sigma_min_min=float(np.nanmin(smin)) if smin.size else float("nan"),
               cond_B_max=float(np.nanmax(cb)) if cb.size else float("nan"))
    if sigma_max is not None:
        smax = np.asarray(sigma_max, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(smin > 0, smax / np.maximum(smin, 1e-300), np.inf)
        out["cond_J_max"] = float(np.nanmax(ratio)) if ratio.size else float("nan")
        out["sigma_max_max"] = float(np.nanmax(smax)) if smax.size else float("nan")
    return out


def x_err_parts(x_tilde, case, ell=0.5):
    """任务空间误差的**分项范数** `(pos, rot, weighted)`（口径见模块文档）。

    - `task_kind = "position"`（2R 位置子任务）：`pos = ‖x̃‖₂`，`rot = 0`，加权范数即 `pos`；
    - `task_kind = "pose"`（6R）：`pos = ‖x̃_{1:3}‖`、`rot = ‖x̃_{4:6}‖`、
      加权 `√(pos² + rot²/ℓ²)`。
    """
    x_tilde = np.asarray(x_tilde, dtype=float)
    if x_tilde.ndim == 1:
        x_tilde = x_tilde[None, :]
    if case.task_kind == "pose":
        pos = _vec_norm(x_tilde[:, :3])
        rot = _vec_norm(x_tilde[:, 3:6])
        w = np.sqrt(pos ** 2 + (rot / ell) ** 2)
    else:
        pos = _vec_norm(x_tilde)
        rot = np.zeros_like(pos)
        w = pos.copy()
    return pos, rot, w


# ---------------------------------------------------------------------------
# 单次仿真的指标表
# ---------------------------------------------------------------------------
def summarize(res, case, ell=0.5, has_task=None):
    """把一次 `code_sim.simulate` 的记录压成指标表（键名固定，供归档与报告引用）。"""
    t = res["t"]
    q_tilde = res["q_tilde"]
    if has_task is None:
        has_task = bool(np.any(res["x_tilde"] != 0.0))
    eq_inf, eq_2 = _vec_norm(q_tilde, ord_=np.inf), _vec_norm(q_tilde)
    out = {
        "case": case.name, "label": res.get("label", ""),
        "nan": bool(res["nan"]), "error": res["error"],
        "T": float(res["T"]), "dt": float(res["dt"]), "h_c": float(res["h_c"]),
        "n_samples": int(t.size), "wall_time_s": float(res["wall_time_s"]),
        "q_tilde_inf": signal_stats(eq_inf),
        "q_tilde_2": signal_stats(eq_2),
        "t_90_s": decay_times(t, eq_inf)[0], "t_10_s": decay_times(t, eq_inf)[1],
        "settle_time_2pct_s": settle_time(t, eq_inf),
        "control": control_cost(t, res["u"], case.u_max),
        "numerics": numerics(res["sigma_min"], res["cond_B"], res.get("sigma_max")),
    }
    if has_task:
        pos, rot, w = x_err_parts(res["x_tilde"], case, ell)
        out["x_pos"] = signal_stats(pos)
        out["x_rot"] = signal_stats(rot)
        out["x_weighted"] = signal_stats(w)
        out["ell_m"] = float(ell)
    return out


def aggregate(rows, keys):
    """把多次运行的指标**聚合为「均值 ± 标准差」**（计划 §6.1 的统计口径）。

    `rows` 为 `summarize()` 的结果列表；`keys` 为 `(路径元组, 展示名)` 列表，
    路径按字典逐层下钻（如 `("q_tilde_inf", "final")`）。
    """
    agg = {}
    for path, name in keys:
        vals = []
        for r in rows:
            v = r
            for k in path:
                v = v.get(k, None) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(float(v))
        if vals:
            a = np.asarray(vals, dtype=float)
            agg[name] = {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
                         "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}
        else:
            agg[name] = None
    return agg


# ---------------------------------------------------------------------------
# 误差动态参数辨识（阶段 3.1 的主判据；阶段 2 作为**诊断量**报告）
# ---------------------------------------------------------------------------
def identify_second_order(t, e, min_peaks=2):
    """从欠阻尼自由响应曲线辨识 `(ξ̂, ω̂_n)`——对数衰减率 + 振荡周期。

    对 `ẍ̃ + 2ξω_nẋ̃ + ω_n²x̃ = 0` 的自由响应，相邻**同号峰**（这里取正的极大值）满足
    `δ = ln(A_k/A_{k+1}) = 2πξ/√(1−ξ²)`，`ω_d = 2π/T_d`，`ω_n = ω_d/√(1−ξ²)`。

    ⚠️ **`e` 必须是带符号的误差，不能传 `|e|`**（2026-09-19 阶段 2 实测踩过）：
    取绝对值后每个**半**周期就会出现一个峰，`T_d` 被估成一半、`ω̂_n` 翻倍
    （实测设定 `ξ=0.3, ω_n=6` 被辨识成 `ξ̂=0.155, ω̂_n=11.6`，恰好是「半周期」的解析结果）。
    同样地，**峰高也必须取带符号值**（极大值），否则衰减率也会错。

    返回 `dict(xi_hat, wn_hat, n_peaks, log_dec, period_s, note)`；正峰数不足时 `xi_hat/wn_hat = None`
    （临界/过阻尼没有振荡峰，本方法**不适用**——如实返回 `None`，不要硬凑）。
    """
    from scipy.signal import find_peaks
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=float)
    pk, _ = find_peaks(e)
    if e.size >= 2 and e[0] > e[1]:          # 初值往往就是第一个极大值
        pk = np.r_[0, pk]
    out = {"xi_hat": None, "wn_hat": None, "n_peaks": int(pk.size),
           "log_dec": None, "period_s": None, "note": "",
           "peak_idx": [], "peak_t": [], "peak_v": []}
    pos = pk[e[pk] > 0] if pk.size else pk   # 只保留正的极大值（负峰不能取对数）
    # 峰的位置/时刻/幅值一并返回：出图（`plot_logdec_fit`）与报告要标出「参与拟合的点」
    out["peak_idx"] = [int(i) for i in pos]
    out["peak_t"] = [float(t[i]) for i in pos]
    out["peak_v"] = [float(e[i]) for i in pos]
    if pos.size < min_peaks:
        out["n_peaks"] = int(pos.size)
        out["note"] = "正极大值不足 %d 个（临界/过阻尼无振荡峰，或观测窗太短），对数衰减率法不适用" \
                      % min_peaks
        return out
    amps = e[pos]
    if np.any(amps[:-1] <= 0) or np.any(amps[1:] <= 0):
        out["note"] = "峰幅含 0，无法取对数"
        return out
    decs = np.log(amps[:-1] / amps[1:])
    periods = np.diff(t[pos])
    delta = float(np.mean(decs))
    T_d = float(np.mean(periods))
    xi = delta / np.sqrt(4.0 * np.pi ** 2 + delta ** 2)
    wn = (2.0 * np.pi / T_d) / np.sqrt(max(1e-12, 1.0 - xi ** 2))
    out.update({"xi_hat": float(xi), "wn_hat": float(wn), "log_dec": delta, "period_s": T_d,
                "n_peaks": int(pos.size),
                "note": "由 %d 个正极大值的对数衰减率辨识（δ=%.4f, T_d=%.4f s）"
                        % (pos.size, delta, T_d)})
    return out


# ---------------------------------------------------------------------------
# 出图（本批次结果不进书稿正文，图只进 res/ 归档与报告）
# ---------------------------------------------------------------------------
def plot_error_curves(curves, path, title="", ylabel="误差（∞ 范数）", logy=True):
    """多算法误差时间曲线叠加。`curves = [{"label":…, "t":…, "e":…}, …]`。"""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for c in curves:
        ax.plot(c["t"], np.maximum(c["e"], 1e-18), label=c["label"], lw=1.6)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_torque_curves(curves, path, title="", names=None):
    """控制力矩时间曲线（每个关节一个子图）。`curves = [{"label":…, "t":…, "u":…}, …]`。"""
    plt = _pyplot()
    n_j = curves[0]["u"].shape[1] - 1
    names = names or ["关节 %d" % (j + 1) for j in range(n_j)]
    fig, axes = plt.subplots(n_j, 1, figsize=(7.2, 1.9 * n_j), sharex=True, squeeze=False)
    for j in range(n_j):
        ax = axes[j, 0]
        for c in curves:
            ax.plot(c["t"], c["u"][:, j + 1], label=c["label"], lw=1.4)
        ax.set_ylabel("%s\n力矩 (N·m)" % names[j])
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("时间 t (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_phase_portrait(traces, path, q_d, title="", names=None):
    """2R 相图（吸引域）：每个关节一幅 `q_j–q̇_j` 平面图，多条初值轨迹叠加。

    `traces = [{"label":…, "q":(n,N+1), "qdot":(n,N+1)}, …]`；`q_d` 为目标位形（画星号）。
    """
    plt = _pyplot()
    n_j = traces[0]["q"].shape[1] - 1
    names = names or ["关节 %d" % (j + 1) for j in range(n_j)]
    fig, axes = plt.subplots(1, n_j, figsize=(4.6 * n_j, 4.2), squeeze=False)
    for j in range(n_j):
        ax = axes[0, j]
        for k, tr in enumerate(traces):
            ax.plot(tr["q"][:, j + 1], tr["qdot"][:, j + 1],
                    lw=0.9, alpha=0.8, label=tr["label"] if k == 0 else None)
        ax.plot([q_d[j + 1]], [0.0], "k*", ms=14, label="目标 $q_d$")
        ax.set_xlabel("$q_%d$ (rad)" % (j + 1))
        ax.set_ylabel("$\\dot q_%d$ (rad/s)" % (j + 1))
        ax.set_title(names[j])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_kp_sweep(rows, path, title="", xlabel="K_P 对角元 (N/rad 或 N/m)"):
    """`K_P` 扫描：稳态误差 vs `K_P`（双对数），并附理论斜率 −1 参考线。"""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r in rows:
        ax.loglog(r["K_P"], np.maximum(r["err"], 1e-18), "o-", label=r["label"], lw=1.4)
    kp = np.asarray(rows[0]["K_P"], dtype=float)
    ref = rows[0]["err"][0] * (kp / kp[0]) ** -1.0
    ax.loglog(kp, ref, "k--", lw=1.0, alpha=0.7, label="理论斜率 -1（正比于 K_P^-1）")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("稳态误差")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 阶段 3 增补的出图（判据 1 的对数衰减率拟合、通用多曲线 / 分组柱状图）
# ---------------------------------------------------------------------------
def plot_logdec_fit(t, signal, ident, path, title="", ylabel="误差（带符号）"):
    """**判据 1 的证据图**：误差曲线（对数纵轴）+ 参与拟合的正极大值 + 辨识包络。

    `ident` 为 `identify_second_order()` 的返回值（须含 `peak_t/peak_v/xi_hat/wn_hat`）。
    包络按 `|e(t)| = |e(0)|·e^{−ξ̂ω̂_n t}` 画出（对数纵轴下是直线），
    与峰值的相对位置直接体现辨识是否抓住了真实衰减率。
    """
    plt = _pyplot()
    t = np.asarray(t, dtype=float)
    signal = np.asarray(signal, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogy(t, np.maximum(np.abs(signal), 1e-18), lw=1.5, label="误差（绝对值）")
    pk_t, pk_v = ident.get("peak_t", []), ident.get("peak_v", [])
    if pk_t:
        ax.semilogy(pk_t, np.maximum(pk_v, 1e-18), "ro", ms=7,
                    label="参与拟合的正极大值（%d 个）" % len(pk_t))
    if ident.get("xi_hat") and ident.get("wn_hat") and pk_v:
        env = abs(pk_v[0]) * np.exp(-ident["xi_hat"] * ident["wn_hat"] * t)
        ax.semilogy(t, np.maximum(env, 1e-18), "k--", lw=1.0,
                    label=r"辨识包络 $e^{-\hat\xi\hat\omega_n t}$"
                          r"（$\hat\xi=%.4f$, $\hat\omega_n=%.4f$）"
                          % (ident["xi_hat"], ident["wn_hat"]))
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _wrap_caption(text, width=68):
    """把图注按**显示宽度**折行（CJK 记 2 列、其余记 1 列），返回已含 `\\n` 的字符串。

    为什么需要：`ax.set_title(title + "\\n" + note)` 里的**单行**若长于画布宽度，
    `tight_layout()` 只按**行数**留白、不会横向扩展，多余部分会被**直接裁掉**
    （2026-09-20 阶段 4 实测：`singular_u_6r.png` / `instability_6r.png` 的图注右侧被切）。

    `width=68` 是按实测标定的：`figsize=(7.2, …)`、`tight_layout` 后轴宽约 6.6 in，
    字号 10 pt 时 CJK 每字约 12.5 px（150 dpi 下 ≈ 1.2 列）——取 68 列留约 8% 余量。
    ⚠️ **图注里不要写 `$...$`**：折行会把数学块切断，残缺的 `$` 之后整段会被当数学模式
    渲染成字面量乱码（2026-09-20 实测踩过）。用纯文本 + Unicode 符号。
    """
    out, cur, wcur = [], [], 0
    for ch in str(text):
        if ch == "\n":
            out.append("".join(cur))
            cur, wcur = [], 0
            continue
        cw = 2 if ord(ch) > 0x2E7F else 1
        if wcur + cw > width and cur:
            out.append("".join(cur))
            cur, wcur = [], 0
        cur.append(ch)
        wcur += cw
    if cur:
        out.append("".join(cur))
    return "\n".join(out)


def _caption(title, note):
    """标题 + 折行后的图注（两者都折，标题一般很短不受影响）。"""
    if not note:
        return title
    return title + "\n" + _wrap_caption(note)


def plot_grouped_lines(series, path, title="", xlabel="", ylabel="",
                       logx=False, logy=False, note="", hlines=None):
    """多条曲线（可选双对数）：`series = [{"label":…, "x":…, "y":…, "ls":…}, …]`。

    阶段 3/4 的通用出图（跟踪误差 vs 轨迹速度、Lyapunov 函数曲线等）；`note` 作为图注
    写在标题下方（用于声明误差范数口径等**必须随图出现**的信息），并**自动折行**防裁切
    （见 `_wrap_caption`）。
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for s in series:
        ax.plot(s["x"], np.maximum(s["y"], 1e-300) if logy else s["y"],
                s.get("ls", "-"), label=s["label"], lw=1.6, ms=5)
    for h in (hlines or []):
        ax.axhline(h["y"], color="k", ls=":", lw=1.0, alpha=0.7, label=h.get("label"))
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(_caption(title, note), fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_grouped_bars(groups, path, title="", ylabel="", logy=True, note=""):
    """分组柱状图：`groups = [{"label": 组名, "values": {柱名: 值}}, …]`（对数纵轴可选）。

    ⚠️ 柱顶的数值标签是**旋转 90°** 写的，必须给纵轴留上边距，否则文字会被坐标轴裁掉
    （2026-09-19 出图时踩过：耦合抑制那张图的 `2.02e-02` 只露出下半截）。
    """
    plt = _pyplot()
    names = list(groups[0]["values"].keys())
    x = np.arange(len(names))
    width = 0.8 / max(1, len(groups))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    vmax = 1e-300
    for gi, g in enumerate(groups):
        vals = [max(float(g["values"][n]), 1e-300) for n in names]
        vmax = max(vmax, max(vals))
        ax.bar(x + gi * width, vals, width, label=g["label"])
        for xi, v in zip(x + gi * width, vals):
            ax.text(xi, v, "%.2e" % v, ha="center", va="bottom", fontsize=7, rotation=90)
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(top=vmax * 6.0)
    else:
        ax.set_ylim(top=vmax * 1.18)
    ax.set_xticks(x + width * (len(groups) - 1) / 2.0)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(_caption(title, note), fontsize=10)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
