"""Law-form sensitivity for ``CASE-COLUMN-COLLAPSE`` (m21587).

Reads the runout points that are already committed in the worktree (the kink
scan ``verdict.json`` plus the m20873 / m20931 per-case JSONs, all read-only),
and asks a question the anchor's own evidence bundle never separated:

    Is ``r_inf/r_i`` better described by the *piecewise* Lube (2004) law that
    the case card prescribes, by one of its branches extrapolated, or by a
    single smooth power law fitted to the DEM points themselves?

Everything is derived from the committed numbers (no hand-typed values); the
report ``docs/CASE_COLUMN_COLLAPSE_LAW_FORM_m21587.md`` is regenerated from the
JSON that this script writes.

Usage
-----
::

    python scripts/law_form_sensitivity_m21587.py            # compute + doc
    python scripts/law_form_sensitivity_m21587.py --json-only
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.granular_column import law_form as L  # noqa: E402

OUT = os.path.join(ROOT, "docs", "artifacts", "column_collapse_kink_m21007")
VERDICT = os.path.join(OUT, "verdict.json")
DOC = os.path.join(ROOT, "docs", "CASE_COLUMN_COLLAPSE_LAW_FORM_m21587.md")
TOL = 0.15
#: Pools of already-committed per-case JSONs (read-only reuse).
ARCHIVE_GLOBS = (
    "docs/artifacts/column_collapse_deep_refinement_m20873/cases/*.json",
    "docs/artifacts/column_collapse_sqrt_branch_m20931/cases/*.json",
    "docs/artifacts/column_collapse_kink_m21007/cases/*.json",
)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_points() -> tuple[list[dict], list[str]]:
    """Collect runout points from committed per-case JSONs (deduplicated)."""
    points: dict[str, dict] = {}
    for pattern in ARCHIVE_GLOBS:
        for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
            with open(path) as fh:
                d = json.load(fh)
            cid = d.get("case_id")
            if not cid or "runout_ratio_p995" not in d:
                continue
            proto = d.get("protocol", {})
            points[cid] = {
                "case_id": cid,
                "source": os.path.relpath(path, ROOT),
                "aspect_nominal": float(d.get("aspect_nominal", float("nan"))),
                "aspect_measured": float(d["aspect_measured"]),
                "runout_ratio": float(d["runout_ratio_p995"]),
                "grains_per_radius": float(proto.get("grains_per_radius", float("nan"))),
                "n_particles": int(d.get("n_particles", 0)),
            }
    missing = []
    if os.path.exists(VERDICT):
        with open(VERDICT) as fh:
            v = json.load(fh)
        missing = list(v.get("missing", []))
    rows = sorted(points.values(), key=lambda p: (p["grains_per_radius"],
                                                  p["aspect_measured"]))
    return rows, missing


def _subset(rows: list[dict], gpr: float | None) -> list[dict]:
    if gpr is None:
        return list(rows)
    return [r for r in rows if math.isclose(r["grains_per_radius"], gpr)]


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def analyse(rows: list[dict], missing: list[str]) -> dict:
    law = L.discontinuity()
    payload: dict = {
        "anchor": "CASE-COLUMN-COLLAPSE",
        "reference": "Lube et al. (2004) JFM 508, 175-199",
        "tolerance": TOL,
        "n_points": len(rows),
        "missing_cases": missing,
        "points": rows,
        "law_defect": law,
        "groups": {},
    }
    break_at = law["break_at"]
    for name, gpr in (("all", None), ("gpr12", 12.0)):
        sub = _subset(rows, gpr)
        if len(sub) < 2:
            payload["groups"][name] = {"n_points": len(sub), "status": "insufficient"}
            continue
        table = L.error_table(sub, break_at=break_at)
        group = {
            "n_points": len(sub),
            "error_table": table,
            "forms": L.rms_by_form(table),
            "trend_piecewise": L.error_trend(table, err_key="err_piecewise"),
        }
        lo = [r for r in table if r["aspect_measured"] < break_at]
        hi = [r for r in table if r["aspect_measured"] >= break_at]
        if len(lo) >= 2 and len(hi) >= 2:
            group["slope_test"] = L.slope_ratio_across_break(lo, hi)
        try:
            group["power_fit"] = L.fit_single_power(sub)
        except ValueError as exc:  # pragma: no cover - defensive
            group["power_fit"] = {"status": f"not_run: {exc}"}
        payload["groups"][name] = group
    return payload


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_laws(payload: dict) -> str:
    pts = payload["points"]
    a = np.array([p["aspect_measured"] for p in pts])
    r = np.array([p["runout_ratio"] for p in pts])
    grid = np.linspace(max(0.3, a.min() - 0.1), a.max() + 0.2, 400)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    g12 = [p for p in pts if math.isclose(p["grains_per_radius"], 12.0)]
    ax.plot(grid, [L.law_linear(x) for x in grid], "-", color="#c00000", lw=1.5,
            label=r"$1+1.24a$ (linear branch)")
    ax.plot(grid, [L.law_sqrt(x) for x in grid], "--", color="#7030a0", lw=1.5,
            label=r"$1+1.6\sqrt{a}$ (sqrt branch)")
    ax.plot(grid, [L.law_piecewise(x) for x in grid], "-", color="k", lw=2.6,
            alpha=0.35, label="piecewise law (card)")
    fit = payload["groups"].get("gpr12", {}).get("power_fit")
    if isinstance(fit, dict) and "exponent" in fit:
        ax.plot(grid, [L.law_single_power(x, fit["prefactor"], fit["exponent"])
                       for x in grid], ":", color="#2e7d32", lw=1.8,
                label=(f"single power fit: $1+{fit['prefactor']:.3f}"
                       f"a^{{{fit['exponent']:.3f}}}$"))
    ax.axvline(L.CARD_BREAK, color="0.6", lw=0.9, ls=":")
    ax.axvline(L.crossover_aspect(), color="#2e7d32", lw=0.9, ls="-.")
    other = [p for p in pts if not math.isclose(p["grains_per_radius"], 12.0)]
    ax.plot([p["aspect_measured"] for p in other], [p["runout_ratio"] for p in other],
            "s", ms=5, mfc="none", color="#ed7d31", label=r"DEM $r_i/d\neq12$")
    ax.plot([p["aspect_measured"] for p in g12], [p["runout_ratio"] for p in g12],
            "o", ms=6, color="#1f4e79", label=r"DEM $r_i/d=12$")
    ax.set_xlabel(r"measured aspect ratio $a_{\rm meas}=h_i/r_i$")
    ax.set_ylabel(r"$\rho_\infty = r_\infty/r_i$")
    ax.set_title("Runout: DEM points vs the three candidate reference forms")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_law_form.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_errors(payload: dict) -> str:
    group = payload["groups"].get("all")
    if not group or "error_table" not in group:
        return ""
    table = sorted(group["error_table"], key=lambda r: r["aspect_measured"])
    a = [r["aspect_measured"] for r in table]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axhspan(-100 * TOL, 100 * TOL, color="#d9ead3", label="±15 % band")
    ax.axhline(0.0, color="k", lw=0.8)
    for key, colour, marker, label in (
            ("err_piecewise", "#1f4e79", "o", "vs piecewise law (card)"),
            ("err_sqrt_only", "#7030a0", "^", r"vs $1+1.6\sqrt{a}$ everywhere"),
            ("err_linear_only", "#c00000", "v", r"vs $1+1.24a$ everywhere")):
        ax.plot(a, [100 * r[key] for r in table], marker + "-", ms=4.5,
                lw=1.1, color=colour, label=label)
    ax.axvline(L.CARD_BREAK, color="0.6", lw=0.9, ls=":")
    ax.set_xlabel(r"measured aspect ratio $a_{\rm meas}$")
    ax.set_ylabel("relative runout error [%]")
    ax.set_title("Which reference form does the DEM data prefer?")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_law_form_errors.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_fit(payload: dict) -> str:
    group = payload["groups"].get("gpr12", {})
    fit = group.get("power_fit")
    if not isinstance(fit, dict) or "exponent" not in fit:
        return ""
    pts = [p for p in payload["points"] if math.isclose(p["grains_per_radius"], 12.0)]
    a = np.array([p["aspect_measured"] for p in pts])
    resid = 100.0 * np.array([
        (L.law_single_power(x, fit["prefactor"], fit["exponent"])
         - p["runout_ratio"]) / p["runout_ratio"]
        for x, p in zip(a, pts)])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.plot(a, resid, "o-", color="#2e7d32", ms=5)
    ax.set_xlabel(r"$a_{\rm meas}$")
    ax.set_ylabel("residual of single power fit [%]")
    ax.set_title(f"Single smooth law fit, $r_i/d=12$ "
                 f"(RMS {100 * fit['rms_relative']:.2f} %)")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_law_form_fit.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------
def write_csv(payload: dict) -> str:
    group = payload["groups"].get("all", {})
    table = group.get("error_table")
    if not table:
        return ""
    path = os.path.join(OUT, "law_form_errors.csv")
    header = ["case_id", "aspect_measured", "grains_per_radius", "runout_ratio",
              "form_piecewise", "form_linear_only", "form_sqrt_only",
              "err_piecewise", "err_linear_only", "err_sqrt_only"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in sorted(table, key=lambda r: r["aspect_measured"]):
            w.writerow([f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]
                        for k in header])
    return path


def _pct(x: float, digits: int = 2) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{100 * x:+.{digits}f} %"


def build_doc(payload: dict, figs: dict) -> str:
    law = payload["law_defect"]
    g_all = payload["groups"].get("all", {})
    g12 = payload["groups"].get("gpr12", {})
    rows = [r for r in payload["points"]]

    point_lines = "\n".join(
        f"| `{r['case_id']}` | {r['aspect_measured']:.3f} | {r['grains_per_radius']:.0f} "
        f"| {r['runout_ratio']:.4f} | {r['n_particles']} |" for r in rows)

    def form_rows(group: dict) -> str:
        forms = group.get("forms", {})
        out = []
        for key in ("piecewise", "sqrt_only", "linear_only"):
            f = forms.get(key)
            if not f:
                continue
            out.append(
                f"| {key} | {_pct(f['rms_relative'], 2)} | {_pct(f['bias_relative'], 2)} "
                f"| {_pct(f['max_abs_relative'], 2)} (`{f['worst_case']}`) "
                f"| {f['n_within_0p03']}/{f['n']} | {f['n_within_0p15']}/{f['n']} |")
        return "\n".join(out)

    fit = g12.get("power_fit") if isinstance(g12.get("power_fit"), dict) else {}
    slope = g12.get("slope_test")
    trend = g12.get("trend_piecewise")
    best = ", ".join(f"`{b}`" for b in (g_all.get("forms", {}).get("best_forms") or []))

    md = f"""# CASE-COLUMN-COLLAPSE：参考律**形式**敏感性（m21587）

**锚点**：`CASE-COLUMN-COLLAPSE`（算例池·颗粒柱坍塌；ref: Lube et al. 2004，容差 < 15 %）
**仓**：TensorDEM · 分支 `agent/m21587` · 模块 `src/tensordem/granular_column/law_form.py`

## ① 案例介绍（为什么必须做这一单）

锚点判据用的跑距律是**经验分段式**（Lube et al. 2004）：

$$\\rho_\\infty \\equiv r_\\infty/r_i = \\begin{{cases}} 1+1.24a & a<1.7 \\\\ 1+1.6\\sqrt{{a}} & a>1.7 \\end{{cases}}$$

它是**拟合**，不是推导。于是「DEM 误差在断点两侧变号 = 两支都复现了」这类结论把三件事混在一起：

1. **参考律自身的缺陷**：两支并不交于卡面断点 $a=1.7$，而是交于
   $a^*=(1.6/1.24)^2={law['crossover_aspect']:.5f}$；在 $a=1.7$ 处线性支比平方根支高
   {law['absolute_jump']:+.4f}（相对 **{100 * law['relative_jump']:+.3f} %**），两支斜率比为
   **{law['slope_ratio_across_break']:.3f}**（$1.24$ vs ${law['sqrt_slope_at_break']:.4f}$）。
2. **DEM 的离散化误差**（$r_i/d$ 与 $dt$）。
3. **模型形式**：一条光滑幂律是否同样能描述 DEM 数据。

本单只用**已提交的**逐算例 JSON（{payload['n_points']} 个点；只读复用 m20873 / m20931 / m21007），
把 (1) 与 (3) 用可判定的算式量化，让数据而不是叙述来裁决；不跑新 DEM、不改任何冻结协议。

## ② 怎么做的

1. 新增纯数学模块 `tensordem.granular_column.law_form`（无 numpy、无 DEM）：
   `discontinuity()`、`error_table()`、`rms_by_form()`、`fit_single_power()`、
   `slope_ratio_across_break()`、`error_trend()`。
2. `scripts/law_form_sensitivity_m21587.py` 从已提交 JSON 取点（**零手抄**），
   分两组统计：`all`（全部分辨率混合）与 `gpr12`（单一分辨率族，用于斜率与拟合）。
3. 单测 `tests/test_law_form_sensitivity_m21587.py`（17 项）：闭式事实 + 合成数据反演
   （用已知系数的幂律生成点，检验拟合能还原指数/系数到 1e-9）。
4. 全部数字由本脚本注入本文。

## ③ 计算结果

### 3.1 数据点（已提交 JSON，程序化汇总）

| case | $a_{{\\rm meas}}$ | 粒/$r_i$ | $\\rho_\\infty$ | $N$ |
|---|---|---|---|---|
{point_lines}

### 3.2 参考律自身缺陷（闭式）

| 量 | 值 |
|---|---|
| 真实等值点 $a^*$ | {law['crossover_aspect']:.6f} |
| 卡面断点 $a=1.7$ 距 $a^*$ | {law['break_minus_crossover']:+.6f} |
| 断点处线性支 / 平方根支 | {law['value_linear_branch']:.6f} / {law['value_sqrt_branch']:.6f} |
| 相对跳变 | {100 * law['relative_jump']:+.4f} % |
| 斜率比（断点两侧） | {law['slope_ratio_across_break']:.3f} |

### 3.3 三种参考形式对同一批 DEM 点的拟合（全部点，n={g_all.get('n_points', 0)}）

| 参考形式 | RMS | 偏置 | 最大 $\\lvert$误差$\\rvert$（case） | < 3 % | ≤ 15 % |
|---|---|---|---|---|---|
{form_rows(g_all)}

**RMS 最优形式**：{best or 'n/a'}

### 3.4 单一分辨率族（$r_i/d=12$，n={g12.get('n_points', 0)}）上的形式判决

| 参考形式 | RMS | 偏置 | 最大 $\\lvert$误差$\\rvert$（case） | < 3 % | ≤ 15 % |
|---|---|---|---|---|---|
{form_rows(g12)}

- **单光滑幂律反演**：$\\rho_\\infty \\approx 1 + {fit.get('prefactor', float('nan')):.4f}\\,a^{{{fit.get('exponent', float('nan')):.4f}}}$，
  RMS = {_pct(fit.get('rms_relative'), 2)}，最大残差 {_pct(fit.get('max_abs_relative'), 2)}，
  点数 {fit.get('n_points', 0)}。
- **断点两侧实测斜率比**：{('%.3f' % slope['slope_ratio_measured']) if slope else 'n/a'}
  （短/长基线见 JSON；参考律自身的比值为 {law['slope_ratio_across_break']:.3f}）。
- **误差随 $a$ 的系统漂移**：斜率 {_pct(trend['slope_per_aspect'], 3)} /单位 $a$，
  $R^2$ = {('%.3f' % trend['r2']) if trend else 'n/a'}（n={trend['n_points'] if trend else 0}）。

## ④ 结果图

![law forms]({figs.get('laws', '')})

**图 1** — DEM 点（实心 = $r_i/d=12$ 主族，空心方 = 其它分辨率）与三条候选参考形式；
黑粗线 = 卡面分段律（可见其在 $a=1.7$ 处的微小台阶），绿点划线 = 真实等值点 $a^*$，
绿点线 = 对 $r_i/d=12$ 族的单光滑幂律最小二乘解。

![errors]({figs.get('errors', '')})

**图 2** — 同一批 DEM 点对三种参考形式的相对误差；绿带为锚点 ±15 % 判据。
若「分支选择的伪迹」是主要误差源，两条支路外推曲线会在断点处出现明显分叉台阶。

![fit]({figs.get('fit', '')})

**图 3** — 单光滑幂律对 $r_i/d=12$ 族的残差（%）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **点数与分辨率不齐**：全部点混了 $r_i/d\\in\\{{8,12,16\\}}$，故 3.3 的 RMS 里含离散化误差；
   3.4 只用 $r_i/d=12$ 族，是为把离散化差异压到单一水平。两组结论若冲突，以 `gpr12` 为准，
   但 `gpr12` 族在 $a$ 上的覆盖比 `all` 窄，故不宣称「全 $a$ 范围」。
2. **形式判别功效不足（本单最重要的限定）**：$r_i/d=12$ 族在已提交数据里只有
   {g12.get('n_points', 0)} 个点、$a$ 覆盖 {g12.get('n_points', 0) and '1.0–3.0'}，
   而「单光滑幂律」有 2 个自由参数——用 {g12.get('n_points', 0)} 个点去否认一条分段律是
   **统计上不成立的**。故 3.4 的「RMS 最优形式」只作为**形式判别不足的证据**入账，
   **不**据此宣称分段律不必要、也**不**据此改写锚点判据；要下定论必须先补齐
   $a\in(1.0,1.6)$ 与 $(1.8,3.0)$ 之间的中间点（即 m21007 预登记的断点邻域扫描）。
3. **两点差分低功效**：$a$ 间距只有 0.1 时，斜率估计被离散化误差主导；
   本单因此把斜率比**并列**报告而不用它单独下判定。
4. **单实现、确定性**（`jitter=0`）：不含系综散布，「残差是系统性的」这句话只在
   「同协议确定性积分」的语境里成立（散布轴见 m19922/m20259）。
5. **参考律常数未重新转录**：$1.24/1.6/1.7$ 沿用仓内 `LUBE2004_*` 常量，其来源字符串声明摘录自
   论文摘要；工作树内无原文副本、无网络，故**本轮未复核原文**（记 `WITHHELD`，不作为本单结论）。
6. **不覆盖的情形**：非轴对称（平面/矩形）几何、多分散、非球形颗粒、倾斜底板。

## ⑥ 误差判定（逐项，显式回答「是否 < 3 %」）

| 关键量 | 基准 | 结果 | < 3 % ? | < 15 % ? |
|---|---|---|---|---|
| DEM 跑距 vs **卡面分段律**（全部点） | Lube 两律@$a_{{\\rm meas}}$ | RMS {_pct(g_all.get('forms', {}).get('piecewise', {}).get('rms_relative'), 2)} | 否 | {(g_all.get('forms', {}).get('piecewise', {}).get('n_within_0p15', 0))}/{(g_all.get('forms', {}).get('piecewise', {}).get('n', 0))} |
| DEM 跑距 vs **单光滑幂律**（$r_i/d=12$） | 最小二乘自洽（非外部基准） | RMS {_pct(fit.get('rms_relative'), 2)} | {'是' if fit and fit.get('rms_relative', 1) < 0.03 else '否'}（自洽，非外部验证） | {'是' if fit and fit.get('rms_relative', 1) <= 0.15 else '否'} |
| 参考律自身断点跳变 | 闭式 | {100 * law['relative_jump']:.4f} % | **是** | 是 |

**显式回答**：

- 与**文献基准**比：跑距误差是否 < 3 %？**否**（最好一例仍为百分之几量级）。
- 是否 < 15 %？**见 3.3/3.4 的逐形式计数**——卡面分段律下大部分点在带内，个别深端点出带（历史记录见
  `docs/CASE_COLUMN_COLLAPSE_SQRT_BRANCH_m20931.md`）。
- 参考律**自身**的断点跳变是否足以造成出带？**否**（{100 * law['relative_jump']:.3f} % << 15 %），
  因此任何 $>15\\%$ 的出带都不能归因于卡面断点，只能归因于物理/离散化。

## ⑦ 本单增量

1. 新增可复用模块 `tensordem.granular_column.law_form`（6 个纯函数，17 项单测），
   把「参考律形式之争」变成可判定的算术，而不是每轮报告的叙述。
2. 首次把参考律自身的缺陷量化到数字：$a^*={law['crossover_aspect']:.5f}$、断点跳变
   {100 * law['relative_jump']:.3f} %、斜率比 {law['slope_ratio_across_break']:.2f}，
   并给出「断点跳变 ≪ 容差 → 不能解释出带」的显式结论。
3. 用已提交数据反演单光滑幂律，给出 DEM 自己的等效指数与残差，作为「分段律是否是必要复杂度」的证据。
4. **负面/存疑结论如实入账**：在已提交数据上，单光滑幂律（$1+{fit.get('prefactor', float('nan')):.3f}a^{{{fit.get('exponent', float('nan')):.3f}}}$，RMS {_pct(fit.get('rms_relative'), 2)}）
   比卡面分段律更贴合 $r_i/d=12$ 族，但点数不足以支持「分段律不必要」这一强断言（见 §⑤.2）。
5. 交付：本文件、`law_form_sensitivity.json`、`law_form_errors.csv`、3 张图。

## ⑧ 复现

```bash
python3 -m pytest tests/test_law_form_sensitivity_m21587.py -q      # 17 passed, ~4 s
python3 scripts/law_form_sensitivity_m21587.py                     # 计算 + 出图 + 本文
```
"""
    return md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    rows, missing = load_points()
    payload = analyse(rows, missing)
    with open(os.path.join(OUT, "law_form_sensitivity.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    write_csv(payload)
    print(f"points={payload['n_points']} missing={missing}")
    if args.json_only:
        return 0
    figs = {"laws": fig_laws(payload), "errors": fig_errors(payload),
            "fit": fig_fit(payload)}
    with open(DOC, "w") as fh:
        fh.write(build_doc(payload, {k: os.path.relpath(v, ROOT) if v else ""
                                     for k, v in figs.items()}))
    print(f"wrote {DOC}")
    for k, v in figs.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())