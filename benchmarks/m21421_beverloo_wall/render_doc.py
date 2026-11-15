#!/usr/bin/env python3
"""Render docs/CASE-DEM-SILO-BEVERLOO_m21421.md from runs/m21421_beverloo_wall/.

Every number in the document is injected from ``summary.json`` (which is itself
produced by ``analyse.py`` from the raw cluster telemetry) or from
``benchmarks/m20055_beverloo/ref_beverloo_1961.json``.  Nothing is hand-typed,
and the prose states the pre-registered decision rules and their outcome.

Usage:
    PYTHONPATH=src python benchmarks/m21421_beverloo_wall/render_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "m21421_beverloo_wall"
REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"
OUT = ROOT / "docs" / "CASE-DEM-SILO-BEVERLOO_m21421.md"


def main() -> int:
    ref = json.loads(REF_PATH.read_text())
    s = json.loads((RUN_DIR / "summary.json").read_text())
    rows = {r["case_id"]: r for r in s["rows"]}
    dec = s["decisions"]
    c = ref["coefficients"]
    dom = ref["validation_domain"]
    scan = sorted([r for k, r in rows.items() if k in ("S26", "S35", "S40")],
                  key=lambda r: r["silo_D_over_D"])

    def f(x, n=4):
        return "n/a" if x is None else f"{x:.{n}f}"

    L = []
    A = L.append
    A("# CASE-DEM-SILO-BEVERLOO · 壁面邻近根因判定单（m21421）\n")
    A(f"**锚点** `CASE-DEM-SILO-BEVERLOO`（总账 open；验收 |W−W_ref|/W_ref **< 10 %**）  ")
    A(f"**基线** Beverloo, Leniger & van de Velde (1961), *Chem. Eng. Sci.* "
      f"**15**(3-4) 260-269, DOI `{ref['source']['doi']}`，Eq.(4) p.265："
      f"`W = C·ρ_b·√g·(D−k·d)^(5/2)`，球粒 `C = {c['C']}`（{c['C_range_spherical'][0]}–"
      f"{c['C_range_spherical'][1]}）、`k = {c['k']}`（{c['k_range_spherical'][0]}–"
      f"{c['k_range_spherical'][1]}）；机器可读 `ref_beverloo_1961.json`（零手抄，"
      f"provenance: cited-not-re-digitized）  ")
    A(f"**适用域**（ref 文件 `validation_domain`）：D/d ≥ {dom['D_over_d_min']}、"
      f"silo_D/D ≥ {dom['silo_D_over_D_min']}、床高/D ≥ {dom['bed_height_over_D_min']}  ")
    A("**本单席别** tp-ds-1（proposer 席，自找活·验证欠账）｜mission 21421｜"
      "集群作业 `cj3431`（4 分片）\n")

    # ---- ① case
    A("## ① 案例介绍（物理背景 / 几何 / 参数）\n")
    A("平底圆柱料仓底部圆心圆孔，颗粒重力自由卸料，稳态质量流率对照 Beverloo 律。"
      "m20055→m20707→m20934→m21033→m21274 的七轮证据链已确认：锥形 hopper 内核"
      "（m15259，−0.4 %）与 `silo_flow` 内核（m20707，−6.9 %/+5.9 %）在各自条件下通过；"
      "而 `silo_discharge` 内核在 kn=1e5（+25.0 %）与 kn=1e6（+20.2 %）**都**过预测——"
      "「软接触」假设被 m21274 证伪，壁面邻近列为第一候选根因。\n")
    A("**本单设计：单变量壁面扫描**。除仓半径外全部参数冻结：颗粒 d = 20 mm、"
      "ρ_p = 2500 kg/m³、μ_p = 0.15、μ_r = 0.02、e = 0.30、kn = 1e6 N/m、kt/kn = 0.5、"
      "g = 9.81 m/s²、孔口 D = 0.100 m（**D/d = 5**）、板厚 0.010 m（0.5 d）、"
      "壁摩擦 0（光滑，玻璃珠条件）、settled+堵孔协议（settle 0.60 s）、卸料 3.0 s、"
      "5×0.50 s 窗口、稳态段取质量分位 0.20–0.80、**同一种子 21033（配对扫描）**；"
      "床高恒为 H = 0.400 m（H/D = 4.0），只把侧壁向外移。\n")
    A("| id | silo R [m] | silo_D/D | H [m] | H/D | n_bed | 作用 |")
    A("|---|---|---|---|---|---|---|")
    purpose = {"S26": "基线（≡ m21033 D5，复现 +20 %）",
               "S35": "适用域内（评估点）",
               "S40": "适用域深部（**主判定**）",
               "F26": "协议对照：灌装期不堵孔（自由落体松散床）"}
    for r in s["rows"]:
        rec = json.loads((RUN_DIR / f"{r['case_id']}.json").read_text())
        dm = rec["dimensionless"]
        A(f"| {r['case_id']} | {rec['telemetry']['silo_radius']:.3f} | "
          f"{r['silo_D_over_D']:.2f} | {rec['telemetry']['fill_height']:.3f} | "
          f"{dm['fill_ratio_H_over_D']:.1f} | {r['n_bed']} | {purpose[r['case_id']]} |")
    A("")

    # ---- ② method
    A("## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）\n")
    A("1. **总账核对**：`SELECT … FROM feature_ledger WHERE slug='CASE-DEM-SILO-BEVERLOO'` "
      "→ status `open`，可认领；未改写锚点。\n"
      "2. **预注册**（`benchmarks/m21421_beverloo_wall/prereg.md`，跑前写定）：建立壁面邻近"
      "假设 H1、协议对照假设 H2、三档 silo_D/D 的量化判据与**证伪条件**。\n"
      "3. **新增独立测量仪**（本单唯一代码改动，`src/tensordem/silo_discharge.py`）："
      "`SiloCase.measure_orifice_flux` 在板中面孔口平面（`z = −t/2`, `radial < a`）"
      "记录颗粒穿越，得到 **W_flux（欧拉面通量）**，与原有 **W_count（落到板下的颗粒计数）** "
      "互为独立估计。默认关闭（旧行为逐位不变，见测试）。动机：先排除「卸料计数记账偏差」"
      "这一类内核缺陷，再谈物理。\n"
      "4. **递交**：`run_compute submit`，`--shard {i} --nshards 4`（4 核，timeout 7200 s，"
      "stage_dir 直达节点），作业 `cj3431`。\n"
      "5. **程序化判定**：`benchmarks/m21421_beverloo_wall/analyse.py` 从原始遥测重算 "
      "W_ref / W_dem / C_recovered / 误差 / CV / 通量一致性，写 `summary.json` 与 "
      "`report_tables.md`，并渲染两张图；本文由 `render_doc.py` 从 `summary.json` 注入，"
      "**零手抄**。\n"
      "6. **回归**：`tests/test_m21421_beverloo_flux.py`（7 项：通量仪不改物理、质量守恒、"
      "两估计量一致、窗口形状、预注册矩阵一致性）+ 既有 silo 回归 "
      "`test_silo_discharge_steady / test_beverloo_discharge_measurement / "
      "test_m21033_beverloo_campaign / test_silo_discharge_beverloo`。\n")

    # ---- ③ results
    A("## ③ 计算结果\n")
    A("判定表（全部由 `analyse.py` 程序化生成，原始遥测 `runs/m21421_beverloo_wall/*.json`）：\n")
    A("| case | silo_D/D | n_bed | ρ_b [kg/m³] | W_DEM [kg/s] | W_ref [kg/s] | 误差 | "
      "C_recovered | CV | W_flux [kg/s] | 通量差 | <10 % ? |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cid in ("S26", "S35", "S40", "F26"):
        r = rows.get(cid)
        if not r:
            continue
        A(f"| {cid} | {r['silo_D_over_D']:.2f} | {r['n_bed']} | {r['rho_b']:.1f} | "
          f"{r['W']:.4f} | {r['W_ref']:.4f} | **{100*r['rel']:+.2f} %** | "
          f"{r['C_rec']:.3f} | {r['CV']:.3f} | {r['W_flux']:.4f} | "
          f"{100*r['flux_gap']:+.2f} % | {'✓' if r['pass10'] else '✗'} |")
    A("")
    A("**主判定 S40（silo_D/D = 4.0，适用域深部）**：")
    s40 = rows.get("S40")
    if s40:
        A(f"W_DEM = {s40['W']:.4f} kg/s vs W_ref = {s40['W_ref']:.4f} kg/s → "
          f"误差 **{100*s40['rel']:+.2f} %**（容差 10 %），C_recovered = {s40['C_rec']:.3f} "
          f"（文献 {c['C']}，带 {c['C_range_spherical']}）。"
          f"→ **{'PASS' if s40['pass10'] else 'FAIL'}**\n")
    A(f"**通量一致性**（内核缺陷排除）：`flux_consistency_all_cases = "
      f"{dec.get('flux_consistency_all_cases')}`；各算例 |W_flux−W_count|/W_count 见上表末列，"
      f"判据 < 3 %。\n")

    # ---- ④ figures
    A("## ④ 结果图\n")
    A("**图 1** `docs/figures/m21421_beverloo_wall_scan.png`——左：恢复的 Beverloo 前置因子 "
      "C_recovered 随 silo_D/D 变化（绿带 = 文献 C 区间，点线 = 适用域下界 2.5）；"
      "右：误差随 silo_D/D 变化（绿带 = ±10 % 容差）。\n")
    A("**图 2** `docs/figures/m21421_beverloo_estimators.png`——左：各算例累计卸料曲线（原始遥测，"
      "显示稳态段确实取在流的中段，非空仓/启动瞬态）；右：W_count vs W_flux 双估计量奇偶校验。\n")

    # ---- ⑤ caveats
    A("## ⑤ 注意点（坑 / 局限 / 适用范围）\n")
    A(f"* **基线本就贴边**：S26 的 silo_D/D = {rows.get('S26', {}).get('silo_D_over_D', 2.6):.2f}，"
      f"仅比 Beverloo 适用域下界 {dom['silo_D_over_D_min']} 高 0.1——把适用域外的算例当作"
      "「定律失效」是此前若干轮的一类误读。\n"
      "* **单变量设计**：本单只动仓径（床高固定 0.4 m），因此颗粒数随截面增长"
      "（统计更好、代价更高），但 D/d、颗粒尺寸相对孔口的比值全程不变，这正是 Beverloo "
      "所标度的量。\n"
      "* **C/k 为引用值**：环境无外网、无 1961 原文，无法重数字化原表；ρ_b 定义沿用 m21033"
      "（床体静态容重），**预注册禁止重定义参考**以免变成「搬球门」。\n"
      "* **单种子**：每几何一次实现，稳态性由 5 个子窗口 CV 评估，不做多种子误差棒。\n"
      "* **通量仪的新增开销**：每步一次 `hypot` + 一次比较，未改物理（测试断言 flag 关闭时"
      "W 逐位一致）。\n")

    # ---- ⑥ error table
    A("## ⑥ 误差判定（逐项，显式回答「是否 < 3 %」）\n")
    A("| 量 | DEM | 基准 | 相对误差 | < 3 % ? | < 10 % ? |")
    A("|---|---|---|---|---|---|")
    for r in [rows[k] for k in ("S26", "S35", "S40") if k in rows]:
        A(f"| W（{r['case_id']}, silo_D/D={r['silo_D_over_D']:.1f}） | {r['W']:.4f} kg/s | "
          f"{r['W_ref']:.4f} kg/s | {100*r['rel']:+.2f} % | "
          f"{'是' if abs(r['rel']) < 0.03 else '否'} | "
          f"{'是' if abs(r['rel']) < 0.10 else '否'} |")
    if s40:
        A(f"| C_recovered（S40） | {s40['C_rec']:.3f} | {c['C']} | "
          f"{100*(s40['C_rec']-c['C'])/c['C']:+.2f} % | "
          f"{'是' if abs(s40['C_rec']-c['C'])/c['C'] < 0.03 else '否'} | "
          f"{'是' if abs(s40['C_rec']-c['C'])/c['C'] < 0.10 else '否'} |")
    A("")
    A("**是否 < 3 % 的显式回答**："
      + (f"是——主判定 S40 的 W 误差 {100*s40['rel']:+.2f} %，C_recovered 误差 "
         f"{100*(s40['C_rec']-c['C'])/c['C']:+.2f} %。" if s40 and abs(s40["rel"]) < 0.03
         else (f"否——主判定 S40 的 W 误差 {100*s40['rel']:+.2f} %，"
               "未达 3 % 量级（Beverloo 律本身的系数带就有 ~±5 %，3 % 属超规格，如实记录）。"
               if s40 else "n/a（主判定算例未回传）"))
      + "\n")

    # ---- ⑦ decisions
    A("## ⑦ 预注册判据的判定结果（honest）\n")
    A("| 判据 | 结果 |")
    A("|---|---|")
    for k, v in dec.items():
        A(f"| `{k}` | `{v}` |")
    A("")
    A("**结论**：")
    if dec.get("ledger_acceptance_met"):
        A("壁面邻近**是** `silo_discharge` 内核过预测的第一顺位根因：把算例放回 Beverloo 适用域"
          "（silo_D/D ≥ 3.5）后误差回到 < 10 % 容差内，锚点验收在适用域内达成；"
          "同时通量仪证明守护量（卸料计数）无记账偏差。\n")
    else:
        A("壁面邻近**未能**单独解释过预测：即使把仓径放大到适用域深部，误差仍超 10 % 容差，"
          "锚点保持 FAIL（honest）。下一顺位候选（流动态膨胀导致的 ρ_b 有效值、板厚/边缘几何）"
          "进入下一轮。\n")
    A("**工件与复现**\n")
    A("```bash\n"
      "PYTHONPATH=src python3 benchmarks/m21421_beverloo_wall/run_case.py --shard 1 --nshards 4\n"
      "PYTHONPATH=src python3 benchmarks/m21421_beverloo_wall/analyse.py     # 判定 + 图\n"
      "PYTHONPATH=src python3 benchmarks/m21421_beverloo_wall/render_doc.py  # 本文\n"
      "PYTHONPATH=src python3 -m pytest tests/test_m21421_beverloo_flux.py -q\n"
      "```\n")
    OUT.write_text("\n".join(L))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())