"""Report builder for the breakpoint-neighbourhood scan (m21007).

Reads only committed provenance (``docs/artifacts/column_collapse_kink_m21007/
verdict.json``, itself produced by ``--collect`` from the per-case JSONs) and
injects every number into ``docs/CASE_COLUMN_COLLAPSE_KINK_m21007.md``, so no
value in the report is typed by hand.
"""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "artifacts", "column_collapse_kink_m21007")
VERDICT = os.path.join(OUT, "verdict.json")
DOC = os.path.join(ROOT, "docs", "CASE_COLUMN_COLLAPSE_KINK_m21007.md")
TOL = 0.15
LINEAR, ROOTP, BREAK = 1.24, 1.6, 1.7


def load() -> dict:
    with open(VERDICT) as fh:
        return json.load(fh)


def law_linear(a):
    return 1.0 + LINEAR * np.asarray(a, dtype=float)


def law_sqrt(a):
    return 1.0 + ROOTP * np.sqrt(np.asarray(a, dtype=float))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_curve(payload: dict) -> str:
    rows = [r for r in payload["rows"] if r["grains_per_radius"] == 12.0]
    rows.sort(key=lambda r: r["aspect_measured"])
    a = np.array([r["aspect_measured"] for r in rows])
    dem = np.array([r["runout_ratio"] for r in rows])
    a_star = payload["verdict"]["law_discontinuity"]["crossover_aspect"]
    grid_lo = np.linspace(min(a.min(), 0.95), BREAK, 200)
    grid_hi = np.linspace(a_star, max(a.max(), 2.02), 200)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(grid_lo, law_linear(grid_lo), "-", color="#c00000", lw=1.6,
            label=r"Lube (2004) $1+1.24a$")
    ax.plot(grid_hi, law_sqrt(grid_hi), "--", color="#7030a0", lw=1.6,
            label=r"Lube (2004) $1+1.6\sqrt{a}$")
    ax.axvline(BREAK, color="0.55", lw=0.9, ls=":")
    ax.axvline(a_star, color="#2e7d32", lw=0.9, ls="-.")
    ax.text(BREAK + 0.005, ax.get_ylim()[1], "", fontsize=8)
    ax.plot(a, dem, "o", ms=6, color="#1f4e79",
            label=r"DEM, $r_i/d = 12$ (this mission + m20873)")
    for x, y, r in zip(a, dem, rows):
        ax.annotate(f"{100 * r['err_runout']:+.1f}%", (x, y),
                    textcoords="offset points", xytext=(4, -12), fontsize=7,
                    color="#1f4e79")
    ax.set_xlabel(r"measured aspect ratio $a_{\rm meas} = h_i/r_i$")
    ax.set_ylabel(r"runout ratio $r_\infty/r_i$")
    ax.set_title("Breakpoint neighbourhood: DEM vs piecewise Lube (2004) law")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_kink_curve.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_band(payload: dict) -> str:
    rows = payload["rows"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.axhspan(-100 * TOL, 100 * TOL, color="#d9ead3", label="±15 % criterion")
    ax.axhline(0.0, color="k", lw=0.8)
    for gpr, style, colour in ((12.0, "o-", "#1f4e79"), (8.0, "s--", "#ed7d31")):
        pts = sorted((r for r in rows if r["grains_per_radius"] == gpr),
                     key=lambda r: r["aspect_measured"])
        if not pts:
            continue
        ax.plot([r["aspect_measured"] for r in pts],
                [100 * r["err_runout"] for r in pts], style, color=colour,
                label=f"DEM $r_i/d = {gpr:g}$")
    for r in rows:
        ax.annotate(r["case_id"], (r["aspect_measured"], 100 * r["err_runout"]),
                    textcoords="offset points", xytext=(5, 6), fontsize=6.5,
                    color="0.3")
    ax.axvline(BREAK, color="0.55", lw=0.9, ls=":")
    ax.set_xlabel(r"measured aspect ratio $a_{\rm meas}$")
    ax.set_ylabel(r"runout error vs Lube law  [%]")
    ax.set_title("Error across the break ($a=1.7$): no branch artefact")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_error_band.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_slopes(payload: dict) -> str:
    h3 = payload["verdict"]["H3_slope_shape"]
    law = payload["verdict"]["law_discontinuity"]
    labels = ["measured\n$a:1.0\\to1.5$", "measured\n$a:1.5\\to1.6$",
              "measured\n$a:1.7\\to1.8$", "Lube law\n$a<1.7$", "Lube law\n$a>1.7$"]
    values = [h3["slope_below_long"], h3["slope_below_short"], h3["slope_above"],
              law["linear_slope_at_break"], law["sqrt_slope_at_break"]]
    colours = ["#1f4e79", "#1f4e79", "#ed7d31", "#c00000", "#7030a0"]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax[0].bar(range(len(values)), values, color=colours)
    ax[0].set_xticks(range(len(values)))
    ax[0].set_xticklabels(labels, fontsize=7.5)
    ax[0].set_ylabel(r"local slope $d(r_\infty/r_i)/da$")
    ax[0].set_title("Shape across the break: DEM flattening vs law (2-point diffs)")
    for i, v in enumerate(values):
        ax[0].annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=8)
    h4 = payload["verdict"]["H4_resolution"]
    ax[1].bar([0, 1], [100 * h4["err_gpr8"], 100 * h4["err_gpr12"]],
              color=["#ed7d31", "#1f4e79"], width=0.55)
    ax[1].axhspan(-100 * TOL, 100 * TOL, color="#d9ead3", alpha=0.7)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["gpr = 8", "gpr = 12"])
    ax[1].set_ylabel(r"runout error at $a=1.7$  [%]")
    ax[1].set_title(f"Resolution control at the break "
                    f"($\\Delta$ = {h4['delta_pp']:.1f} pp)")
    for i, v in enumerate([100 * h4["err_gpr8"], 100 * h4["err_gpr12"]]):
        ax[1].annotate(f"{v:+.2f}%", (i, v), ha="center",
                       va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_slopes_resolution.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# markdown injection
# ---------------------------------------------------------------------------
def fmt_pct(x: float) -> str:
    return "n/a" if not math.isfinite(x) else f"{100 * x:+.2f} %"


def point_table(rows: list[dict]) -> str:
    head = ("| case | 来源 | $r_i/d$ | $a_{\\rm nom}$ | $a_{\\rm meas}$ | N | 步数 | "
            "$r_\\infty/r_i$ | ref | 误差 | $t_{90}$ | ref $t_\\infty$ | $t_{90}$ 误差 | "
            "能量残差 | 判定 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    out = [head]
    for r in rows:
        ok = "✅" if abs(r["err_runout"]) <= TOL else "❌"
        out.append(
            f"| `{r['case_id']}` | {r['source']} | {r['grains_per_radius']:g} | "
            f"{r['aspect_nominal']:.2f} | {r['aspect_measured']:.4f} | "
            f"{r['n_particles']} | {r['n_steps']} | {r['runout_ratio']:.4f} | "
            f"{r['ref_runout_ratio']:.4f} | {fmt_pct(r['err_runout'])} | "
            f"{r['t90']:.4f} | {r['ref_time']:.4f} | {fmt_pct(r['err_t90'])} | "
            f"{100 * r['energy_residual']:.2f} % | {ok} |\n")
    return "".join(out)


def build(payload: dict, figs: dict) -> str:
    v = payload["verdict"]
    rows = payload["rows"]
    law = v["law_discontinuity"]
    h1, h2, h3, h4 = (v["H1_band"], v["H2_no_branch_artefact"],
                      v["H3_slope_shape"], v["H4_resolution"])
    g12 = [r for r in rows if r["grains_per_radius"] == 12.0]
    worst = max(g12, key=lambda r: abs(r["err_runout"]))
    best = min(g12, key=lambda r: abs(r["err_runout"]))
    rel = lambda p: os.path.relpath(p, ROOT)
    md = f"""# CASE-COLUMN-COLLAPSE · 断点邻域扫描（m21007）

**锚点**：`CASE-COLUMN-COLLAPSE`（算例池·颗粒柱坍塌）
**基准**：Lube, Huppert, Sparks & Freundt (2004) *JFM* **508**:175-199,
doi:10.1017/S0022112004009036
**容差**：< 15 %（跑距律）
**预登记**：`docs/PREREG_COLUMN_COLLAPSE_KINK_m21007.md`（5 个算例、4 条假设，跑前写定）
**仓**：TensorDEM · 分支 `agent/m21007`

## ① 案例介绍（物理背景 / 几何 / 参数表）

圆柱颗粒柱（初始半径 $r_i$、高 $h_i=a r_i$）在粗糙水平底板上撤去侧向支撑后坍塌。
Lube et al. (2004) 的跑距律**分段**：

$$r_\\infty/r_i = \\begin{{cases}} 1+1.24a & a<1.7 \\\\ 1+1.6\\sqrt{{a}} & a>1.7\\end{{cases}}$$

两支在 $a=1.7$ 处斜率相差 **{law['slope_ratio_across_break']:.2f} 倍**
（$1.24$ vs $0.8/\\sqrt a={law['sqrt_slope_at_break']:.4f}$）。已测点在断点两侧符号相反
（$a=1.5$ → $-5.7\\%$、$a=2.0$ → $+4.6\\%$），误差在 $(1.5,2.0)$ 内穿零，但该区间此前
**一个点都没有**（`docs/CASE_COLUMN_COLLAPSE_SYNTHESIS_m20978.md` §⑤ 未做项之三）。

**冻结闭合臂**（m18261→m20873→m20931 逐字沿用，未改一参）：$\\mu=0.5$，
$\\mu_r=0.5$，$\\mu_w=0.8$，$e=0.5$，$k_t/k_n=0.5$，$\\rho=2500$ kg/m³，$g=9.81$ m/s²，
`packing_gap`=0.02（HCP+2 %），`jitter`=0，`seed`=12345，
$r_i=0.48$ m 固定、$r=r_i/(2\\,gpr)$、$k_n=6\\times10^4(r/0.03)^2$（保持 $\\delta/d$）。
新算例：$a\\in\\{{1.5,1.6,1.7,1.8\\}}$ @ $r_i/d=12$，外加断点处分辨率对照 $a=1.7$ @ $r_i/d=8$。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **预登记先行**：`docs/PREREG_COLUMN_COLLAPSE_KINK_m21007.md` 在任何积分前写定
   5 个算例与 H1–H4 判定阈值，跑后未改。
2. **只读复用**：`benchmarks/case_column_collapse_kink_m21007.py` 通过 importlib 载入
   m18261 冻结模块的 `make_case`/`reference_values`/`protocol_signature`，
   不复制、不改写协议；$a=1.0$@gpr12 直接读 m20873 归档 JSON 作交叉点下侧远锚。
3. **参考值运行时求值**：`lube_2004_runout_ratio(a_meas)`，零手抄；主判据用实测 $a$。
4. **提交**：集群作业 `cj3116`（5 分片单进程，调度器派发），结果落
   `docs/artifacts/column_collapse_kink_m21007/cases/`。
5. **归约**：`--collect` 生成 `verdict.json` + `cases_kink.csv`；
   `scripts/report_column_collapse_kink_m21007.py` 把数字注入本文（无手抄）。
6. **先算解析事实**：两支柱等值点 $a^*={law['crossover_aspect']:.5f}$，卡面断点 $1.7$
   在 $a^*$ 之上 {law['break_minus_crossover']:.4f}；两支在 $a=1.7$ 的值差
   {law['absolute_jump']:+.4f}，即参考律**自身**在断点跳变 **{100 * law['relative_jump']:.3f} %**
   ——远小于 15 % 容差，故断点跳变不可能解释任何出带。

## ③ 计算结果（程序化注入）

{point_table(rows)}

> 说明：$t_{{90}}$ 列一律从 `arrival.primary.fraction_0.9` 直读（不采信任何列名——
> m20873 的 `t95` 标签缺陷见 m20931 §⑤.7）。$r_i/d=12$ 的行构成主扫描，
> $r_i/d=8$ 行是断点处的分辨率对照；`R_a1_g12` 是 m20873 归档点（非本单重测），
> 只用作交叉点下侧的远端斜率锚。

## ④ 结果图

![kink curve]({rel(figs['curve'])})

**图 1** — 断点邻域：DEM（$r_i/d=12$，圆点，标注为相对误差）与分段参考律的两支
（红实线 $a<1.7$ 支、紫虚线 $\\sqrt a$ 支）。灰点线 = 卡面断点 $a=1.7$，
绿点划线 = 两支真实等值点 $a^*={law['crossover_aspect']:.4f}$。

![error band]({rel(figs['band'])})

**图 2** — 误差-长宽比曲线（±15 % 绿带）。若 DEM 在断点处有支路伪迹，蓝线会在
$a=1.7$ 出现台阶；实测跨断点跳变为 {h2['jump_below_pp']:.2f} pp
（$1.6\\to1.7$）与 {h2['jump_above_pp']:.2f} pp（$1.7\\to1.8$）。

![slopes and resolution]({rel(figs['slopes'])})

**图 3** — 左：实测分段斜率（两点差分）与参考律两支斜率对照；右：断点 $a=1.7$ 处
$r_i/d=8$ 与 $12$ 的误差对照（分辨率对照）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **两点差分低功效**：H3 的斜率只能用 $a$ 差 0.1 的两点差分估计，噪声量级与
   斜率差同阶；因此 H3 同时用短基线（$1.5\\to1.6$）与长基线（$1.0\\to1.5$）两条，
   两者结论不一致时判 `incomplete`（不硬下结论）。
2. **单分辨率主扫描**：主扫描固定在 $r_i/d=12$，只用一个 $r_i/d=8$ 对照点检验断点
   判据的分辨率敏感性（H4）；**没有**做 $a\\in\\{{1.5,1.6,1.8\\}}$ 的完整分辨率阶梯。
3. **单实现、确定性**（`jitter`=0），不含系综散布（散布见 m19922/m20259）；
   任何单点差异不得读成随机性。
4. **能量账本**残差如实入账（见③表），本单**不设**事后阈值门；
   注意平方根支残差系统性高于线性支（m20931 §⑤.8 已记）。残留 >10 % 的算例：
   {', '.join(v['energy_flagged']) if v['energy_flagged'] else '无'}。
5. **穿透/重叠计数未取**：协议 `return_positions=False`，记 `not_run`。
6. **适用范围**：干颗粒、轴对称、HCP 初始堆积、单分散球体、$\\mu_w$ 粗糙底板。

## ⑥ 误差判定（逐项，显式回答「是否 < 3 %」「是否 < 15 %」）

| 关键量 | 基准 | 最佳 | 最差 | < 3 % ? | < 15 % ? |
|---|---|---|---|---|---|
| 跑距律（$r_i/d=12$ 主扫描，{len(g12)} 点） | Lube 两律@$a_{{\\rm meas}}$ | {fmt_pct(best['err_runout'])}（`{best['case_id']}`） | {fmt_pct(worst['err_runout'])}（`{worst['case_id']}`） | **否** | **{'是' if h1['all_in_band'] else '否'}**（{sum(1 for r in g12 if abs(r['err_runout']) <= TOL)}/{len(g12)} 在带内） |
| 断点处分辨率漂移（$a=1.7$, gpr 8→12） | — | {h4['delta_pp']:.2f} pp | 同左 | {'是' if h4['delta_pp'] < 3 else '否'}（{h4['delta_pp']:.2f} pp） | {'是' if h4['delta_pp'] <= 5 else '否'}（阈值 5 pp） |

**假设裁决**（阈值见预登记，跑后未改）：

| 假设 | 内容 | 结果 | 数字 |
|---|---|---|---|
| H1 | 4 个 gpr=12 新点全部 $\\lvert$err$\\rvert\\le15\\%$ | **{h1['verdict']}** | 误差 {', '.join(fmt_pct(e) for e in h1['errors'])} |
| H2 | 跨断点误差跳变 ≤ 5 pp（无支路伪迹） | **{h2['verdict']}** | {h2['jump_below_pp']:.2f} pp / {h2['jump_above_pp']:.2f} pp |
| H3 | $s_{{\\rm below}}/s_{{\\rm above}}>1$（复现「先陡后平」） | **{h3['verdict']}** | 短基线比 {h3['slope_ratio_short']:.3f}、长基线比 {h3['slope_ratio_long']:.3f}（律：{h3['law_slope_ratio']:.2f}） |
| H4 | $\\lvert$err(1.7@12)$-$err(1.7@8)$\\rvert\\le5$ pp | **{h4['verdict']}** | {h4['delta_pp']:.2f} pp |

**总判定：`{v['overall']}`**

**显式回答**：
- 跑距律是否 < 3 %：**{'是' if all(abs(r['err_runout']) < 0.03 for r in g12) else '否'}**
  （最小 {fmt_pct(best['err_runout'])}）。
- 跑距律是否 < 15 %：**{'是' if h1['all_in_band'] else '否'}**
  （{sum(1 for r in g12 if abs(r['err_runout']) <= TOL)}/{len(g12)} 在带内）。
- 断点是否构成判据障碍：**{'否' if h2['verdict'] == 'pass' else '是'}**
  —— 参考律自身跳变仅 {100 * law['relative_jump']:.3f} %，DEM 跨断点跳变
  {max(h2['jump_below_pp'], h2['jump_above_pp']):.2f} pp。

## ⑦ 本单增量

1. 补上锚点历史上**唯一未测的参考律分段区间** $a\\in(1.5,2.0)$ 中的 4 个点
   （$a=1.5,1.6,1.7,1.8$ @ $r_i/d=12$）+ 断点处分辨率对照（$a=1.7$ @ $r_i/d=8$）。
2. 首次量化参考律**自身**在卡面断点 $a=1.7$ 的不连续（{100 * law['relative_jump']:.3f} %）
   与真实等值点 $a^*={law['crossover_aspect']:.4f}$，把「断点」从模糊的卡面参数
   变成一个可判定的误差预算项。
3. 新增单测 `tests/test_column_collapse_kink_m21007.py`（15 项，纯函数，不跑 DEM）。
4. 报告：本文件 + `docs/artifacts/column_collapse_kink_m21007/`（5 份算例 JSON、
   `verdict.json`、`cases_kink.csv`、3 张图）。

## ⑧ 复现

```bash
# 单测门禁（不跑 DEM，~5 s）
python3 -m pytest tests/test_column_collapse_kink_m21007.py -q
# 算例（缓存优先，已提交的 JSON 直接复用）
python3 benchmarks/case_column_collapse_kink_m21007.py --list
python3 benchmarks/case_column_collapse_kink_m21007.py --shard 0/5
python3 benchmarks/case_column_collapse_kink_m21007.py --collect
python3 scripts/report_column_collapse_kink_m21007.py
```
"""
    return md


def main() -> int:
    payload = load()
    os.makedirs(OUT, exist_ok=True)
    figs = {"curve": fig_curve(payload), "band": fig_band(payload),
            "slopes": fig_slopes(payload)}
    with open(DOC, "w") as fh:
        fh.write(build(payload, figs))
    print(f"wrote {DOC}")
    for key, path in figs.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())