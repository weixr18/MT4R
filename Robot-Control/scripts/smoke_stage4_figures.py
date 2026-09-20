# -*- coding: utf-8 -*-
"""**扫描「图上真正被渲染的文字」里有没有缺字形的字符**（阶段 4 新增，2026-09-20）。

## 为什么需要它

出图时踩过一串「数据全对、图上是方框或乱码」的坑（见书仓库日志尝试 14）：

| 症状 | 根因 |
|---|---|
| 图上出现空框/缺字告警 | `⚠️`(U+26A0)、`ε`、`⇒`(U+21D2)、上标 `⁻¹`(U+207B)、组合变音符 `U+0302/0303/0304/0307` 在 **Microsoft YaHei 里都没有字形** |
| 图注里出现 `$\|y\|\propto 1/\sigma...$` 这种字面量 | 新加的 `code_metrics._wrap_caption` 把 `$...$` **折断了**，残缺的 `$` 之后整段被当数学模式渲染 |
| 直接抛 `ParseSyntaxException` | mathtext 语法错（`$\sqrt2$` 少花括号） |

## 为什么不能只 grep 源码

危险字符可能来自**归档数据或格式化串**（本轮就有一个 `K_P⁻¹` 藏在 f-string 里），
而**注释/文档字符串里的字符根本不进图**——按源码扫描会大量误报。
所以本脚本直接 **monkeypatch `matplotlib.text.Text.set_text`**，把交给 matplotlib 的
**每一个字符串**抓下来检查：只有真正进图的文字才会被审。

## 用法（`py311-gym`，任意 cwd；秒级，不跑仿真）

    python Robot-Control/scripts/smoke_stage4_figures.py            # 用 res/6r_stage4/result.json 审真实图
    python Robot-Control/scripts/smoke_stage4_figures.py --keep     # 保留图到 res/_scratch/figscan/ 肉眼检查
    ... -W error::UserWarning                                        # 把 matplotlib 自己的缺字形告警变成硬失败

退出码：发现**危险字符**或（在 `-W error` 下）缺字形告警则为 1，否则 0。
"""
import argparse
import json
import os
import sys
import unicodedata

_HERE = os.path.dirname(os.path.abspath(__file__))
_RC = os.path.abspath(os.path.join(_HERE, ".."))
_DEFAULT_JSON = os.path.join(_RC, "res", "6r_stage4", "result.json")

# 「危险字符集」＝ 已知在 Microsoft YaHei / matplotlib 默认字体里缺字形，**或**会引发渲染歧义的字符。
# 每一条都对应一次实际踩坑（见模块文档的表）。
DANGER = {
    0x26A0: "WARNING SIGN（⚠️ 的星号部分）",
    0xFE0F: "VARIATION SELECTOR-16（⚠️ 的变体选择符）",
    0x21D2: "RIGHTWARDS DOUBLE ARROW（⇒）",
    0x21D0: "LEFTWARDS DOUBLE ARROW（⇐）",
    0x03B5: "GREEK SMALL LETTER EPSILON（ε；图上请写 eps）",
    0x03B9: "GREEK SMALL LETTER IOTA（ι）",
    0x207B: "SUPERSCRIPT MINUS（⁻¹ 的负号）",
    0x2070: "SUPERSCRIPT ZERO", 0x00B9: "SUPERSCRIPT ONE", 0x00B2: "SUPERSCRIPT TWO",
    0x00B3: "SUPERSCRIPT THREE", 0x2074: "SUPERSCRIPT FOUR", 0x2075: "SUPERSCRIPT FIVE",
    0x2076: "SUPERSCRIPT SIX", 0x2077: "SUPERSCRIPT SEVEN", 0x2078: "SUPERSCRIPT EIGHT",
    0x2079: "SUPERSCRIPT NINE",
    0x0302: "COMBINING CIRCUMFLEX ACCENT（ξ̂ / ω̂ 用）",
    0x0303: "COMBINING TILDE（x̃ 用）",
    0x0304: "COMBINING MACRON（x̄ 用）",
    0x0307: "COMBINING DOT ABOVE（q̇ 用）",
    0x1D62: "LATIN SUBSCRIPT SMALL LETTER I",
}
# 这些字符**已验证有字形**、允许使用（写在这里是为了让后人不必再试一遍）。
ALLOWED = set("σ ω ξ λ Λ α β γ θ φ ε Δ Π Σ Ω π" "·≈≤≥≠±×÷→←↔⇒" "ℓ√∝∞°" "²³¹" "—–“”「」（）₀₁₂₃₄₅₆")


def main(argv=None):
    ap = argparse.ArgumentParser(description="扫描图上文字里的缺字形字符")
    ap.add_argument("--json", default=_DEFAULT_JSON, help="阶段 4 归档（出图的数据源）")
    ap.add_argument("--keep", action="store_true", help="保留图到 res/_scratch/figscan/")
    a = ap.parse_args(argv)

    if not os.path.exists(a.json):
        print("⚠️ 找不到归档 %s——先跑一次 `run_stage4.py`（或用 --json 指定）" % a.json)
        return 2

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.text as mtext

    seen = {}
    orig = mtext.Text.set_text

    def patched(self, s):
        try:
            text = str(s)
        except Exception:                                            # noqa: BLE001
            return orig(self, s)
        for ch in text:
            if ord(ch) in DANGER:
                seen.setdefault(ord(ch), {"name": DANGER[ord(ch)], "samples": set()})
                if len(seen[ord(ch)]["samples"]) < 3:
                    seen[ord(ch)]["samples"].add(text[:160])
        return orig(self, s)

    mtext.Text.set_text = patched
    try:
        sys.path.insert(0, _RC)
        sys.path.insert(0, os.path.join(_RC, "code"))
        sys.path.insert(0, os.path.join(_RC, "test", "ctrl_common"))
        import stage4_lib as s4                                       # noqa: E402

        with open(a.json, "r", encoding="utf-8") as f:
            d = json.load(f)
        need = ("singularity", "model_mismatch", "friction", "measurement_noise",
                "control_period", "discretization_instability")
        missing = [k for k in need if not d.get(k)]
        if missing:
            print("⚠️ 归档缺少 %s ——无法重放全部出图路径" % missing)
            return 2
        fig_dir = (os.path.join(_RC, "res", "_scratch", "figscan") if a.keep
                   else os.path.join(_RC, "res", "_scratch", "_figscan_tmp"))
        figs = s4.make_figures(d["singularity"], d["model_mismatch"], d["friction"],
                               d["measurement_noise"], d["control_period"],
                               d["discretization_instability"], d.get("compute_cost"),
                               fig_dir, tag="scan")
    finally:
        mtext.Text.set_text = orig

    print("==" * 36)
    print("扫描了 %d 张图；检查了交给 matplotlib 的全部字符串。" % len(figs))
    if not seen:
        print("✅ 未发现缺字形/危险字符。")
        print("   （若同时用了 `-W error::UserWarning`，matplotlib 自身的缺字形告警也会变成硬失败，"
              "两者互补。）")
        return 0
    print("❌ 发现 %d 类危险字符——它们在图上会渲染成方框、被替换，或被当数学模式解析：" % len(seen))
    for code, info in sorted(seen.items()):
        print("  U+%04X  %s" % (code, info["name"]))
        for s in sorted(info["samples"]):
            print("       出现在：%s" % s.replace("\n", " ⏎ "))
    print("⇒ 改法：**图注一律纯文本 + 只用已验证有字形的 Unicode**（见 `code_metrics._wrap_caption`"
          " 的文档；不要写 `$...$`）。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
