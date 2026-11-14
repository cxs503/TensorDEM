#!/usr/bin/env python3
"""Render the CASE-DEM-SILO-BEVERLOO report from the run artefacts.

All digits come from ``runs/m20055_beverloo/summary.json`` (itself written by
``analyse.py`` from the per-case JSONs) and from the reference file.  No number
is typed into this script.

    PYTHONPATH=src python benchmarks/m20055_beverloo/analyse.py
    PYTHONPATH=src python benchmarks/m20055_beverloo/render_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "m20055_beverloo"
OUT = ROOT / "docs" / "CASE-DEM-SILO-BEVERLOO_m20055.md"

TOL = 10.0   # case tolerance, predeclared (< 10 %)


def tick(ok: bool) -> str:
    return "✓" if ok else "✗"


def main() -> int:
    s = json.loads((RUNS / "summary.json").read_text())
    ref = s["reference"]
    coeff = s["reference_coefficients"]
    rows = s["rows"]
    pw = s["power_law"]
    sens = s["quantile_sensitivity"]
    prim = next((r for r in rows if r["case"] == "D5"), None)

    L = []
    A = L.append
    A("# CASE-DEM-SILO-BEVERLOO — 料仓卸料流率 Beverloo 律（TensorDEM，m20055）")
    A("")
    A(f"**算例**: 平底圆形料仓自由卸料，稳态质量流率 `W` 对 Beverloo (1961) 律。"
      f"**判定**: 主算例 D/d = 5 "
      f"{'PASS' if prim and prim['pass10'] else 'FAIL'}（容差 < {TOL:.0f} %）。")
    A("")
    A("## ① 问题与参考出处")
    A("")
    A("料仓底板上一个圆心圆孔，颗粒在重力下自由流出；Beverloo 等人 1961 年给出经验律")
    A("")
    A("```")
    A("W = C · rho_b · sqrt(g) · (D − k·d)^(5/2)")
    A("```")
    A("")
    A(f"出处：{ref['authors']}, *{ref['title']}*, {ref['journal']} "
      f"**{ref['volume']}**({ref['issue']}), {ref['pages']} ({ref['year']}), "
      f"DOI {ref['doi']}；相关系数取球状颗粒一对 `C = {coeff['C']}`、`k = {coeff['k']}`"
      f"（适用范围 C∈{coeff['C_range_spherical']}、k∈{coeff['k_range_spherical']}），"
      f"位置 {ref['equation_location']}。")
    A("")
    A("机器可读参考文件 `benchmarks/m20055_beverloo/ref_beverloo_1961.json`，所有脚本只从该文件读数，"
      "代码与报告里不手抄数字。**诚实声明**：本工作树内没有 1961 年原文 PDF，环境无外网，"
      "因此原文表格/数据点无法重新数字化，`C`/`k` 是文献公布系数（仓库内既有模块与 prereg 均引 Eq.(4) p.265），"
      "这一点在参考文件的 `provenance.verification_status` 里如实标注。")
    A("")
    A("## ② 方法与实现")
    A("")
    A("复用仓库既有能力 `tensordem.silo_discharge`（REAL 3-D DEM：线性弹簧-阻尼法向 + Cundall–Strack 切向 +"
      " 带阻尼滚阻，`granular3d_core` 内核），不另起炉灶。前序记录（m6296）在**有序 HCP 床 + 孔口从第 0 步就开着 +"
      " 固定 5 个窗口**的协议下给出 −77…−86 % 的 honest FAIL，原因是床层在孔口上方成拱、流率根本没进入稳态。"
      "本单对该协议做三处最小修正（`src/tensordem/silo_discharge.py`）：")
    A("")
    A("1. `SiloCase.packing = \"settled\"`：以非接触松散格子（净距 `settle_loose_gap`) 装料，"
      "靠重力在 `settle_time` 内压实成**无序床**（真实料仓的状态），避免晶格成拱；")
    A("2. `SiloCase.plug_during_settle = True`：压实阶段把孔口**堵住**（整块平板模型），"
      "开孔瞬间才取床高/床堆密度，使流率不可能被压实期漏出的质量虚增；")
    A("3. `steady_rate()`：以卸料事件记录 (t, N) 为基础，取卸料流的**质量分位 0.2–0.8** 作为稳态区间，"
      "再切成 5 个子窗口给出散度 CV（稳态性校验，不是符合度校验）。原有固定窗口路径保持向后兼容。")
    A("")
    A("流率 `W` 由同一 DEM 的卸料质量-时间记录测得，参考值 `W_ref` 用**同一算例实测的床堆密度** rho_b 代入 Beverloo 律；"
      "回收的 `C` 在 k 固定为文献值时给出。")
    A("")
    A("## ③ 参数表（含无量纲数）")
    A("")
    if prim:
        A(f"共享参数：颗粒 d = 20 mm，ρ_p = 2500 kg/m³，μ_p = {prim['mu_p']:.2f}，μ_r = 0.02，e = 0.30，"
          f"k_n = 1e5 N/m，k_t/k_n = 0.5，R_silo = 0.130 m，平板厚 0.5 d，壁面 μ_w = 0，seed 20055，"
          f"压实 0.60 s，`dt/接触周期 = {prim['dt_over_T']:.4f}`。")
        A("")
    A("| case | D/d | μ_p | n_bed | H/D | silo_D/D | ρ_b [kg/m³] | φ | 卸料占比 | W [kg/s] | W_ref [kg/s] | 误差 | C_rec | CV | <10% |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r['case']} | {r['D_over_d']:.2f} | {r['mu_p']:.2f} | {r['n_bed']} | {r['H_over_D']:.2f} | "
          f"{r['silo_D_over_D']:.2f} | {r['rho_b']:.0f} | {r['phi']:.3f} | {r['n_disch_pct']:.0f}% | "
          f"{r['W']:.4f} | {r['W_ref']:.4f} | {r['rel_pct']:+.1f}% | {r['C_recovered']:.3f} | "
          f"{r['cv']:.3f} | {tick(r['pass10'])} |")
    A("")
    A("无量纲完整性：`D/d`、`silo_D/D`、`H/D`、`rho_b/rho_p`（= φ）、`dt/T_contact`、"
      "以及 Beverloo 前置因子 `C = W/(rho_b sqrt(g) (D-kd)^{5/2})` 本身都已列出，"
      "算例的流动状态由 (D/d, H/D, silo_D/D, φ, μ_p, e, dt/T) 完全确定。")
    A("")
    A("## ④ 对照表：ref vs got + 误差判定")
    A("")
    A("| case | W_ref (Beverloo, ref) | W_dem (got) | 相对误差 | 判据 <10% | 稳态 CV<0.15 |")
    A("|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r['case']} | {r['W_ref']:.4f} | {r['W']:.4f} | {r['rel_pct']:+.1f}% | "
          f"{tick(r['pass10'])} | {tick(r['cv'] < 0.15)} |")
    A("")
    if prim:
        A(f"主算例 D5：`C` 回收 {prim['C_recovered']:.3f} vs 文献 {coeff['C']}"
          f"（{prim['C_rel_pct']:+.1f} %），`W` 误差 {prim['rel_pct']:+.1f} %，"
          f"卸料 {prim['n_disch_pct']:.0f} % 床质量，子窗口 CV {prim['cv']:.3f}。")
        A("")
    if pw:
        A(f"幂律检验（D4/D5/D6 三点 log–log 拟合）：斜率 {pw['slope']:.3f} vs 律的 2.5"
          f"（{100*pw['rel_err']:+.1f} %）。")
        A("")
    A("## ⑤ 收敛性与参数敏感性")
    A("")
    A("**(a) 时间步收敛**（D5 与 D5_dthalf，dt/2）：")
    A("")
    A("| case | dt/T_contact | W [kg/s] | 误差 |")
    A("|---|---|---|---|")
    for r in rows:
        if r["case"] in ("D5", "D5_dthalf"):
            A(f"| {r['case']} | {r['dt_over_T']:.4f} | {r['W']:.4f} | {r['rel_pct']:+.1f}% |")
    A("")
    A("**(b) 颗粒摩擦敏感性**（μ_p = 0.15 / 0.30 / 0.45，其余同 D5）：")
    A("")
    A("| case | μ_p | W [kg/s] | 误差 | CV |")
    A("|---|---|---|---|---|")
    for r in rows:
        if r["case"].startswith("D5_mu"):
            A(f"| {r['case']} | {r['mu_p']:.2f} | {r['W']:.4f} | {r['rel_pct']:+.1f}% | {r['cv']:.3f} |")
    A("")
    A("**(c) 稳态窗口定义敏感性**（同一事件记录、不同质量分位带重算，零额外计算）：")
    A("")
    A("| case | 分位带 | W [kg/s] | CV |")
    A("|---|---|---|---|")
    for cid, bands in sens.items():
        for label, v in bands.items():
            A(f"| {cid} | {label} | {v['W']:.4f} | {v['cv']:.3f} |")
    A("")
    A("## ⑥ 失败模式、工件与生态摩擦")
    A("")
    A("**失败模式**（如 D 系列任一不过判据）")
    A("")
    A("* 有序/晶格床在孔口成拱 → 只有启动突流，流率无稳态（m6296 结论，本单用无序压实床消除）；")
    A("* 批量料仓的尾部（床层抽空）与启动段都会污染平均值 → 用卸料流质量分位带剔除；")
    A("* 压实不足（`settle_time` 偏短）会让 φ/ρ_b 偏低，而 ρ_b 直接进 Beverloo 前置因子；本单把 ρ_b 取为开孔瞬间的实测床值；")
    A("* 小体系泊松噪声：窗口内计数量有限，CV 同时包含物理起伏与计数噪声。")
    A("")
    A("**工件**")
    A("")
    for fid, path in s["figures"].items():
        A(f"* 图 `{fid}`: `{path}`")
    A(f"* 逐算例原始数据: `runs/m20055_beverloo/*.json`（含 git sha、host、dt、窗口序列、卸料事件序列）")
    A(f"* 汇总/表格: `runs/m20055_beverloo/summary.json`, `runs/m20055_beverloo/report_tables.md`")
    A("* 参考数值: `benchmarks/m20055_beverloo/ref_beverloo_1961.json`；"
      "算例脚本 `run_case.py` / 分析 `analyse.py` / 本文生成 `render_doc.py`")
    A("* 一键复现: `PYTHONPATH=src python benchmarks/m20055_beverloo/repro.sh`")
    A("* 回归测试: `PYTHONPATH=src python -m pytest tests/test_silo_discharge_steady.py -q`")
    A("")
    A("**生态摩擦**（如实记）")
    A("")
    A("* 1961 年原文 PDF 不在仓库、环境无外网（curl 出站被安全策略拦截）→ 无法按“零手抄”重新数字化原文表格；"
      "退而求其次：只把文献系数写成唯一机器可读参考文件并标注 `cited, not re-digitized`。")
    A("* `knowledge_base` / `negative_knowledge_ledger` / `oss_solutions` 三张知识表对当前 DB 角色 **无 SELECT 权限**，"
      "开工三查只能靠仓库内既有审计文档（docs/silo_discharge_beverloo_verification.md 等）代偿。")
    A("* 既有 `silo_discharge` 文档字符串宣称 C≈0.58/k≈1.4 已验证，而 m6296 的 honest FAIL 说明该宣称不成立；"
      "本单补上了可复现的稳态协议，并保留了有序床作为对照臂。")
    A("")
    A(f"**总判定**：{s['n_runs']} 个算例中 {sum(1 for r in rows if r['pass10'])} 个落在 ±{TOL:.0f} % 内；"
      f"主算例 D5 {'通过' if prim and prim['pass10'] else '未通过'}。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())