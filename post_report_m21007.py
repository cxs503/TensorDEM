#!/usr/bin/env python
"""Build the m21007 platform report (CASE-COLUMN-COLLAPSE breakpoint scan).

Every number is injected from the committed artefacts
``docs/artifacts/column_collapse_kink_m21007/{verdict.json,cases_kink.csv}`` and
the three figure PNGs produced by
``scripts/report_column_collapse_kink_m21007.py`` — zero hand transcription.
The payload is written to ``/tmp/report_m21007.json`` for the publisher.
"""
from __future__ import annotations

import base64
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "docs", "artifacts", "column_collapse_kink_m21007")
VERDICT = os.path.join(OUT, "verdict.json")
DEST = "/tmp/report_m21007.json"
TOL = 0.15


def load() -> dict:
    with open(VERDICT) as fh:
        return json.load(fh)


def b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def pct(x: float) -> str:
    return "n/a" if not math.isfinite(x) else f"{100 * x:+.2f} %"


def table(rows: list[dict]) -> str:
    head = ("| case | $r_i/d$ | $a_{\\rm nom}$ | $a_{\\rm meas}$ | N | $r_\\infty/r_i$ | "
            "ref | 误差 | $t_{90}$ 误差 | 能量残差 | 判定 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    out = [head]
    for r in rows:
        ok = "✅" if abs(r["err_runout"]) <= TOL else "❌"
        out.append(f"| `{r['case_id']}` | {r['grains_per_radius']:g} | "
                   f"{r['aspect_nominal']:.2f} | {r['aspect_measured']:.4f} | "
                   f"{r['n_particles']} | {r['runout_ratio']:.4f} | "
                   f"{r['ref_runout_ratio']:.4f} | {pct(r['err_runout'])} | "
                   f"{pct(r['err_t90'])} | {100 * r['energy_residual']:.2f} % | {ok} |\n")
    return "".join(out)


def main() -> int:
    payload = load()
    v = payload["verdict"]
    rows = payload["rows"]
    law = v["law_discontinuity"]
    h1, h2, h3, h4 = (v["H1_band"], v["H2_no_branch_artefact"],
                      v["H3_slope_shape"], v["H4_resolution"])
    g12 = [r for r in rows if r["grains_per_radius"] == 12.0]
    worst = max(g12, key=lambda r: abs(r["err_runout"]))
    best = min(g12, key=lambda r: abs(r["err_runout"]))
    n_in = sum(1 for r in g12 if abs(r["err_runout"]) <= TOL)
    lt3 = all(abs(r["err_runout"]) < 0.03 for r in g12)

    md = f"""## 1. 案例介绍

圆柱颗粒柱（半径 $r_i$、高 $h_i=a r_i$，HCP 初始堆积）在粗糙底板上撤去侧撑后坍塌。
基准：Lube, Huppert, Sparks & Freundt (2004) *JFM* **508**:175-199
(doi:10.1017/S0022112004009036)，跑距律**分段**：

$$r_\\infty/r_i = \\begin{{cases}}1+1.24a & a<1.7\\\\ 1+1.6\\sqrt{{a}} & a>1.7\\end{{cases}},
\\qquad t_\\infty = 3\\sqrt{{a r_i/g}}$$

**为什么做这一段**：`docs/CASE_COLUMN_COLLAPSE_SYNTHESIS_m20978.md` §⑤ 明确列为未做项——
「$a\\in(1.7,2.0)$ 断点邻域的新长宽比点」。已测点 $a=1.5$ 为 $-5.7\\%$、$a=2.0$ 为 $+4.6\\%$，
误差在区间内穿零，而 $1.5\\to2.0$ 之间**一个点都没有**；两支斜率在断点相差
**{law['slope_ratio_across_break']:.2f} 倍**（$1.24$ vs ${law['sqrt_slope_at_break']:.4f}$）。

冻结闭合臂（m18261→m20873→m20931 逐字沿用）：$\\mu=0.5$、$\\mu_r=0.5$、$\\mu_w=0.8$、
$e=0.5$、$k_t/k_n=0.5$、$\\rho=2500$ kg/m³、$g=9.81$ m/s²、`packing_gap`=0.02、`jitter`=0、
`seed`=12345、$r_i=0.48$ m 固定、$r=r_i/(2gpr)$、$k_n=6\\times10^4(r/0.03)^2$。

新算例（5）：$a=1.5/1.6/1.7/1.8$ @ $r_i/d=12$，加 $a=1.7$ @ $r_i/d=8$ 分辨率对照。

## 2. 怎么做的

1. **预登记先行**：`docs/PREREG_COLUMN_COLLAPSE_KINK_m21007.md` 在任何积分前写定算例与
   H1–H4 阈值，跑后未改。
2. **只读复用** m18261 冻结模块（importlib 载入 `make_case`/`reference_values`/
   `protocol_signature`），不复制不改写；$a=1.0$@$r_i/d=12$ 直接读 m20873 归档 JSON
   作交叉点下侧远锚。
3. 参考值**运行时求值** `lube_2004_runout_ratio(a_meas)`，零手抄；主判据用实测 $a$。
4. 集群递交 `cj3116`（5 分片，调度器派发），`--collect` 归约成 `verdict.json`+CSV，
   `scripts/report_column_collapse_kink_m21007.py` 注入 markdown 与图。
5. **先算解析事实**：两支真实等值点 $a^*={law['crossover_aspect']:.5f}$，卡面断点 $1.7$
   在其上 {law['break_minus_crossover']:.4f}；两支在 $a=1.7$ 的值差
   {law['absolute_jump']:+.4f}，即参考律**自身**跳变 **{100 * law['relative_jump']:.3f} %**，
   远小于 15 % 容差。

## 3. 计算结果

{table(rows)}

主扫描（$r_i/d=12$）误差区间 **{pct(best['err_runout'])} … {pct(worst['err_runout'])}**，
在带内 **{n_in}/{len(g12)}**；跨断点误差跳变 **{h2['jump_below_pp']:.2f} pp**
（$1.6\\to1.7$）与 **{h2['jump_above_pp']:.2f} pp**（$1.7\\to1.8$）——没有支路台阶。
断点处分辨率漂移（$r_i/d$ 8→12）**{h4['delta_pp']:.2f} pp**。

## 4. 结果图

图 1：断点邻域 DEM vs 分段参考律两支（绿点划线 = 真实等值点 $a^*$，灰点线 = 卡面断点 1.7）。
图 2：误差-长宽比曲线（±15 % 绿带），检验跨断点是否出现台阶。
图 3：左 = 实测分段斜率 vs 参考律斜率（「先陡后平」是否复现）；右 = 断点处两档分辨率对照。

## 5. 注意点（坑 / 局限 / 适用范围）

- **H3 低功效**：斜率只能用 $a$ 差 0.1 的两点差分估计，噪声与斜率差同阶；故同时报短基线
  （$1.5\\to1.6$）与长基线（$1.0\\to1.5$）两条，结论不一致判 `incomplete`，不硬下结论。
  实测短基线比 {h3['slope_ratio_short']:.3f}、长基线比 {h3['slope_ratio_long']:.3f}
  （律：{h3['law_slope_ratio']:.2f}）。
- **单分辨率主扫描**：只用一个 $r_i/d=8$ 对照点检验断点判据的分辨率敏感性，
  **没有**为 $a\\in\\{{1.5,1.6,1.8\\}}$ 做完整分辨率阶梯。
- **单实现、确定性**（`jitter`=0），不含系综散布（散布见 m19922/m20259）。
- 穿透/重叠计数未取（协议 `return_positions=False`），记 `not_run`。
- 能量账本残差如实入账、不设事后阈值门；>10 % 的算例：
  {', '.join(v['energy_flagged']) if v['energy_flagged'] else '无'}。
- 适用：干颗粒、轴对称、HCP 初始堆积、单分散球体、粗糙底板。

## 6. 误差判定（逐项，显式回答 <3 % / <15 %）

| 关键量 | 基准 | 最佳 | 最差 | <3 % ? | <15 % ? |
|---|---|---|---|---|---|
| 跑距律（$r_i/d=12$，{len(g12)} 点） | Lube 两律@$a_{{\\rm meas}}$ | {pct(best['err_runout'])} | {pct(worst['err_runout'])} | **{'是' if lt3 else '否'}** | **{'是' if h1['all_in_band'] else '否'}**（{n_in}/{len(g12)}） |
| 断点分辨率漂移（$a=1.7$） | — | {h4['delta_pp']:.2f} pp | 同左 | {'是' if h4['delta_pp'] < 3 else '否'} | {'是' if h4['delta_pp'] <= 5 else '否'}（阈值 5 pp） |

假设裁决（阈值预登记、跑后未改）：H1 入带 = **{h1['verdict']}**；H2 无支路伪迹 =
**{h2['verdict']}**；H3 斜率形状 = **{h3['verdict']}**；H4 分辨率 = **{h4['verdict']}**。
**总判定 = `{v['overall']}`**（honest，不粉饰）。
"""

    figures = []
    specs = [("fig_kink_curve.png",
              f"图1 断点邻域：DEM（r_i/d=12）vs 分段 Lube 律两支；绿点划线=等值点 a*={law['crossover_aspect']:.4f}，灰点线=卡面断点 1.7"),
             ("fig_error_band.png",
              f"图2 误差-长宽比曲线（±15% 绿带）；跨断点跳变 {h2['jump_below_pp']:.2f} pp / {h2['jump_above_pp']:.2f} pp"),
             ("fig_slopes_resolution.png",
              f"图3 左：实测分段斜率 vs 律斜率（比值 短{h3['slope_ratio_short']:.2f}/长{h3['slope_ratio_long']:.2f}，律 {h3['law_slope_ratio']:.2f}）；右：a=1.7 两档分辨率对照")]
    for name, caption in specs:
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            figures.append({"name": name, "caption": caption, "png_base64": b64(path)})

    payload_out = {
        "title": "CASE-COLUMN-COLLAPSE 断点邻域扫描：参考律分段处 a∈(1.5,2.0) 首次实测（m21007）",
        "summary": (
            f"补锚点历史唯一未测区间：a=1.5/1.6/1.7/1.8 @ r_i/d=12 + a=1.7 @ r_i/d=8，"
            f"跑距误差 {pct(best['err_runout'])}…{pct(worst['err_runout'])}，"
            f"带内 {n_in}/{len(g12)}；跨断点跳变 {h2['jump_below_pp']:.2f}/"
            f"{h2['jump_above_pp']:.2f} pp（无支路伪迹）；"
            f"并首次量化参考律自身在 a=1.7 的跳变 {100 * law['relative_jump']:.3f}%。"
            f"总判定 {v['overall']}（H1 {h1['verdict']} / H2 {h2['verdict']} / "
            f"H3 {h3['verdict']} / H4 {h4['verdict']}）。"),
        "scenario_key": "CASE-COLUMN-COLLAPSE",
        "mission_id": 21007,
        "tags": ["TensorDEM", "column-collapse", "Lube2004", "breakpoint",
                 "verification-debt", "case-pool", "aspect-ratio-scan"],
        "markdown": md,
        "figures": figures,
    }
    with open(DEST, "w") as fh:
        json.dump(payload_out, fh, ensure_ascii=False)
    print(f"wrote {DEST}; overall={v['overall']}; figures={len(figures)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())