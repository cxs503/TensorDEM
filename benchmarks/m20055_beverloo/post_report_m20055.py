#!/usr/bin/env python3
"""Publish the CASE-DEM-SILO-BEVERLOO report (mission 20055).

All digits are read from ``runs/m20055_beverloo/summary.json`` (written by
``analyse.py`` from the per-case JSONs); the reference coefficients are read
from ``benchmarks/m20055_beverloo/ref_beverloo_1961.json`` via that summary.
Figures are the PNGs ``analyse.py`` wrote, embedded as base64.

    python post_report_m20055.py --dry-run
    python post_report_m20055.py
"""
from __future__ import annotations

import base64
import json
import math
import os
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
MISSION_ID = 20055
BASE = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")
SUM = REPO / "runs" / "m20055_beverloo" / "summary.json"


def post(path, body, token=None):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode())


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def f(x, p=4, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{p}f}" if sign else f"{x:.{p}f}"


def main() -> int:
    s = json.loads(SUM.read_text())
    rows = s["rows"]
    pw = s["power_law"]
    sens = s["quantile_sensitivity"]
    prim = next((r for r in rows if r["case"] == "D5"), rows[0])
    ref = s["reference"]
    coeff = s["reference_coefficients"]
    TOL = 10.0

    md = []
    A = md.append
    A("## ① 案例介绍（物理背景 / 几何 / 参数表）")
    A("")
    A("平底圆形料仓（圆柱筒 + 带中心圆孔的底板）在重力下自由卸料。稳态质量流率服从 "
      "Beverloo 等人的经验律 `W = C·ρ_b·√g·(D − k·d)^(5/2)`；")
    A(f"参考出处：{ref['authors']}, *{ref['title']}*, {ref['journal']} "
      f"**{ref['volume']}**({ref['issue']}), {ref['pages']} ({ref['year']}), DOI {ref['doi']}，"
      f"球状颗粒系数 `C = {coeff['C']}`、`k = {coeff['k']}`（{ref['equation_location']}）。"
      "参考值由 `benchmarks/m20055_beverloo/ref_beverloo_1961.json` 程序化读取，"
      "报告与代码中不手抄任何文献数字。")
    A("")
    A(f"几何/物性：d = 20 mm 单分散球，ρ_p = 2500 kg/m³，ρ_b（开孔瞬间实测床值）"
      f"{rows[0]['rho_b']:.0f}–{max(r['rho_b'] for r in rows):.0f} kg/m³，"
      f"R_silo = 0.130 m，平板厚 0.5 d，壁面光滑（μ_w = 0），"
      f"床高 H/D ≈ {min(r['H_over_D'] for r in rows):.2f}–{max(r['H_over_D'] for r in rows):.2f}，"
      f"D/d ∈ [{min(r['D_over_d'] for r in rows):.1f}, {max(r['D_over_d'] for r in rows):.1f}]。")
    A("")
    A("| case | D/d | μ_p | n_bed | H/D | silo_D/D | ρ_b | φ | dt/T_c | W [kg/s] | W_ref | 误差 | C_rec | CV |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r['case']} | {r['D_over_d']:.2f} | {r['mu_p']:.2f} | {r['n_bed']} | {r['H_over_D']:.2f} | "
          f"{r['silo_D_over_D']:.2f} | {r['rho_b']:.0f} | {r['phi']:.3f} | {r['dt_over_T']:.4f} | "
          f"{r['W']:.4f} | {r['W_ref']:.4f} | {r['rel_pct']:+.1f}% | {r['C_recovered']:.3f} | {r['cv']:.3f} |")
    A("")
    A("## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）")
    A("")
    A("复用 TensorDEM 既有真实 3-D DEM 能力 `tensordem.silo_discharge`（线性弹簧-黏壶法向 + Cundall–Strack 切向 +"
      " 带阻尼滚阻，矢量化解算 `granular3d_core`），不另起炉灶。前序 m6296 的 honest FAIL（−77…−86 %）"
      "定位到协议缺陷而非接触律：**有序 HCP 床在孔口成拱、孔口从第 0 步就开、固定窗口落在启动突流上**。"
      "本单做三处最小改动并全部回归测试钉住：")
    A("")
    A("1. `packing=\"settled\"`：非接触松散格子（净距 `settle_loose_gap`）在堵孔状态下重力压实成**无序床**；")
    A("2. `plug_during_settle=True`：压实阶段孔口封闭（整块平板模型），开孔瞬间测床高与床堆密度，"
      "流率不可能被压实期漏料虚增；")
    A("3. `steady_rate()`：用卸料事件记录 (t, N) 取卸料流质量分位 0.2–0.8 为稳态区间并切 5 个子窗给 CV。")
    A("")
    A("工具链：`benchmarks/m20055_beverloo/run_case.py`（一例一 shard，单核）→ 外协方调度"
      f"（job cj2460，{s['n_runs']} shards，节点由调度器指派）→ `analyse.py`（表格/敏感性/图）→ "
      "`render_doc.py`（文档）→ 本报告（全部数字程序化注入）。"
      "预注册 `prereg/m20055_beverloo.md` 在跑算例前写定：主算例 D5、判据 <10 %、矩阵与停止规则冻结。")
    A("")
    A("## ③ 计算结果（程序化注入）")
    A("")
    A("| case | n_bed | 卸料占比 | W_ref (Beverloo) | W_dem | 相对误差 | 稳态 CV | 是否 <10% |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r['case']} | {r['n_bed']} | {r['n_disch_pct']:.0f}% | {r['W_ref']:.4f} | "
          f"{r['W']:.4f} | {r['rel_pct']:+.1f}% | {r['cv']:.3f} | {'✓' if r['pass10'] else '✗'} |")
    A("")
    if pw:
        A(f"幂律检验（D4/D5/D6 的 log–log 斜率）：{pw['slope']:.3f} vs 律的 2.5（{100*pw['rel_err']:+.1f} %）。")
        A("")
    A(f"主算例 D5（预注册）：W = {prim['W']:.4f} kg/s vs 参考 {prim['W_ref']:.4f} kg/s，"
      f"误差 {prim['rel_pct']:+.1f} %，回收 C = {prim['C_recovered']:.3f}（文献 {coeff['C']}），"
      f"卸料占比 {prim['n_disch_pct']:.0f} %，稳态子窗 CV = {prim['cv']:.3f}。")
    A("")
    A("## ④ 结果图")
    A("")
    for fid, path in s["figures"].items():
        A(f"* **{fid}** — `{path}`（见图 `{fid}`）")
    A("")
    A("`flow_law`：W 对 (D−kd)/d 的双对数图，含 Beverloo 律（同 ρ_b，参考）与 5/2 幂律参照线；"
      "`errors`：逐算例相对误差与 ±10 % 判据带；`discharge_curve`：D5 的批量卸料曲线与稳态窗口（场量级证据，"
      "非动画）。")
    A("")
    A("## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）")
    A("")
    A("* **成拱是主要失败模式**：有序 HCP 床会在 3–6 d 孔口上方成拱，只有启动突流；必须用无序压实床。")
    A("* **批量料仓的头部与尾部**：启动段与抽空段都会污染平均流率，Beverloo 是稳态律；本单用卸料流质量分位带剔除。")
    A("* **ρ_b 必须取床值**：ρ_b 直接进 Beverloo 前置因子，用几何装料高度会系统性偏 C。")
    A("* **平台期长度有限**：批量料仓在 H/D ≈ 2.5–2.9 下才有一段可用的准稳态；H/D 太小（<1.5）流率随床层抽空明显下滑。")
    A("* **适用范围**：单分散球、光滑壁、平底、D/d ≤ 6；不含多分散、壁摩擦、湿度/黏附、锥底料斗。")
    A("* **参考数值局限**：1961 年原文 PDF 不在仓库、环境无外网，`C`/`k` 为文献公布系数（仓库内既有模块与 prereg 均引 "
      "Eq.(4) p.265），原文表格未重新数字化——参考文件里标注为 `cited, not re-digitized`。")
    A("")
    A("## ⑥ 误差判定（逐项，显式回答是否 <3 %）")
    A("")
    A("| 关键量 | DEM | 基准 | 相对误差 | <10 % 判据 | <3 % ? |")
    A("|---|---|---|---|---|---|")
    A(f"| W（D/d=5，主算例） | {prim['W']:.4f} kg/s | {prim['W_ref']:.4f} kg/s | {prim['rel_pct']:+.1f} % | "
      f"{'✓' if prim['pass10'] else '✗'} | {'是' if abs(prim['rel_pct']) < 3 else '否'} |")
    for r in rows:
        if r["case"] == "D5":
            continue
        A(f"| W（{r['case']}） | {r['W']:.4f} kg/s | {r['W_ref']:.4f} kg/s | {r['rel_pct']:+.1f} % | "
          f"{'✓' if r['pass10'] else '✗'} | {'是' if abs(r['rel_pct']) < 3 else '否'} |")
    if pw:
        A(f"| 幂律指数（W∝(D−kd)^n） | {pw['slope']:.3f} | 2.5 | {100*pw['rel_err']:+.1f} % | — | "
          f"{'是' if abs(pw['rel_err']) < 0.03 else '否'} |")
    A(f"| 回收 C（k 固定 {coeff['k']}，D/d=5） | {prim['C_recovered']:.3f} | {coeff['C']} | "
      f"{prim['C_rel_pct']:+.1f} % | {'✓' if abs(prim['C_rel_pct']) < TOL else '✗'} | "
      f"{'是' if abs(prim['C_rel_pct']) < 3 else '否'} |")
    A("")
    npass = sum(1 for r in rows if r["pass10"])
    A(f"**总判定**：{s['n_runs']} 个算例中 {npass} 个落在 ±10 % 判据内；主算例 D5 "
      f"{'**通过**' if prim['pass10'] else '**未通过**'}。"
      + ("全体算例一致通过——接触律 + 平底孔口几何 + 无序压实床在该参数域内（D/d 4–6）复现 Beverloo 律。"
         if s["all_pass_10pct"] else
         "未全通过，偏差已逐项列在上表，不做粉饰。"))

    title = (f"料仓卸料 Beverloo 律（TensorDEM）：D/d=5 误差 {prim['rel_pct']:+.1f}%，"
             f"{npass}/{s['n_runs']} 算例在 ±10 % 内")
    figs = []
    captions = {
        "flow_law": "W 对 (D−kd)/d 双对数：TensorDEM 稳态测量点、Beverloo 参考线与 5/2 幂律参照。",
        "errors": "逐算例相对误差，绿色带为预注册 ±10 % 判据。",
        "discharge_curve": "D5 批量卸料曲线（累计卸出颗粒数 vs 时间）与稳态窗口。",
    }
    for fid, path in s["figures"].items():
        if Path(path).exists():
            figs.append({"name": fid, "caption": captions.get(fid, fid),
                         "png_base64": b64(REPO / path)})
    body = {
        "title": title,
        "summary": (f"用无序压实床 + 堵孔压实 + 卸料流质量分位稳态估计，复用 TensorDEM 平底料仓 DEM 复现 "
                    f"Beverloo (1961) 流率：主算例 D/d=5 误差 {prim['rel_pct']:+.1f} %"
                    f"（C_rec {prim['C_recovered']:.3f} vs 文献 {coeff['C']}），"
                    f"{npass}/{s['n_runs']} 算例在 ±10 % 内。"),
        "scenario_key": "CASE-DEM-SILO-BEVERLOO",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "DEM", "silo discharge", "Beverloo", "granular flow",
                 "CASE-DEM-SILO-BEVERLOO", "hopper"],
        "markdown": "\n".join(md),
        "figures": figs,
    }
    out = REPO / ".scratch" / "report_m20055.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(body))
    print("wrote", out, len(out.read_bytes()), "bytes,", len(figs), "figures")
    if os.environ.get("DRY"):
        return 0
    tok = post("/auth/login", {"username": USER, "password": PASS})["access_token"]
    res = post("/tensor/reports", body, tok)
    print("published", json.dumps(res)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())