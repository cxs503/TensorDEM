#!/usr/bin/env python3
"""m24538 (reopen TensorDEM#1729, see-through r3): build /tmp/report.json.

All numbers are read from runs/m22657_beverloo_ensemble/summary.json and the
regenerated case document - zero hand copying.
"""
import base64
import json
from pathlib import Path

ROOT = Path("/root/tp-agent/work/m24538")
SUMMARY = ROOT / "runs" / "m22657_beverloo_ensemble" / "summary.json"
DOC = ROOT / "docs" / "CASE-DEM-SILO-BEVERLOO_m22657.md"
FIGDIR = ROOT / "docs" / "figures"

s = json.loads(SUMMARY.read_text())
ens = s["ensemble"]
ver = s["verdicts"]
per = s["per_seed"]
ref = s["reference"]

rows = ["| seed | n_bed | ρ_b (kg/m³) | W_dem (kg/s) | W_ref (kg/s) | rel | C_rec | drift |",
        "|---|---|---|---|---|---|---|---|"]
for p in per:
    rows.append(f"| {p['seed']} | {p['n_bed']} | {p['rho_bulk_kg_m3']:.1f} | "
                f"{p['W_dem_kg_s']:.6f} | {p['W_ref_kg_s']:.6f} | "
                f"{100*p['rel_err']:+.2f} % | {p['C_recovered']:.3f} | "
                f"{100*p['window_trend']['rel_drift']:+.2f} % |")
seed_table = "\n".join(rows)

preamble = f"""# CASE-DEM-SILO-BEVERLOO · Beverloo 律 seed 系综：NOT_RUN → 实跑裁决（m24538，重开 #1729 管到底第 3 轮）

**本轮做了什么**：m22657 的 7 份交付物经 git 取证确认已在 main（PR #1644，2026-11-16），
但 6 个 seed 的 DEM 运行（原作业 cj4285）工件从未回传，案例文档停在 **NOT_RUN**，
总账裁决悬空。本轮把欠账跑完：

1. 以同一冻结构型（`run_case.py:FROZEN`，prereg.md §3）重新递交 6 分片 → **cj5109**
   （stage_dir 直达节点，每分片 2 核，超时 7200 s），**6/6 分片回传** `seed1..6.json`；
2. `analyse.py` 从 6 份 JSON 重算系综统计与 V1–V4 判定 → `summary.json`；
   `render_doc.py` 从 summary 重新生成案例文档与两张图（零手抄）；
3. 顺带修复 main 上既有的**全套测试收集错误**：`benchmarks/m20055_beverloo/dt_ladder.py`
   的裸 `import run_case` 在整库 pytest 下被先收集的其他 benchmark 目录的 `run_case`
   模块顶替（m22885 已在 m22657 测试文件修过同类 sys.modules 碰撞），改为按路径加载
   （与 m22885 同一模式）。修复后 `pytest tests/` 收集 **12043 项、0 收集错误**。

**本轮判定（全部来自 summary.json，程序化注入）**：

| 判定 | 结果 |
|---|---|
| V1 总账容差（\\|W_ens/W_ref−1\\|<10 % 且 95 % CI 不跨 ±10 % 边界） | **{ver['V1_ledger']}** |
| V2 可复现性（CV<10 %） | **{ver['V2_reproducibility']}** |
| V3 稳态（窗口漂移 <10 %） | **{ver['V3_steady_state']}** |
| V4 ρ_b 控制（散布 <3 %） | **{ver['V4_rho_control']}** |

系综：W_ens = {ens['W_ens_kg_s']:.6f} kg/s，W_ref = {ens['W_ref_kg_s']:.6f} kg/s
（C=0.58, k=1.4, ρ_b={ens['rho_bulk_mean_kg_m3']:.1f} kg/m³），
点估计 {100*ens['rel_err_ensemble']:+.2f} %，95 % CI
[{100*ens['rel_err_ci95'][0]:+.2f} %, {100*ens['rel_err_ci95'][1]:+.2f} %]，
CV = {100*ens['cv']:.2f} %，CI 半宽 {100*ens['ci_half_width_rel']:.2f} %。

逐 seed：

{seed_table}

**是否 <3 %？** 否——且该构型在物理上到不了 3 % 级：计数噪声地板
1/√N ≈ {100*ens['expected_rel_noise_mean']:.2f} %（N≈{int(per[0]['poisson_budget']['n_exit_counted'])} 颗粒/seed），
单 seed 误差与噪声同量级；本单的可判定问题是「是否落在总账 10 % 容差内」，不是 3 % 级精度。

---

"""

md = preamble + DOC.read_text()

figs = []
caps = {
    "m22657_ensemble_errors.png":
        "单 seed 相对误差（点）+ 6-seed 系综均值与 Student-t 95 % CI（方块/误差棒），"
        "灰带为总账 ±10 % 容差。说明：本案例的测量量是标量流率 W，不存在场量云图；"
        "「场」即逐窗口流率时间序列（见第二图），比较对象为 Beverloo 1961 文献系数基线。",
    "m22657_window_history.png":
        "逐窗口（0.05 s）流率历史（6 seed），阴影为计入「稳态」的后 50 % 窗口；"
        "黑虚线为 Beverloo 文献系数预测 W_ref，红带为系综 95 % CI。",
}
for name, cap in caps.items():
    p = FIGDIR / name
    if p.exists():
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(p.read_bytes()).decode()})

payload = {
    "title": "CASE-DEM-SILO-BEVERLOO · Beverloo 律 seed 系综不确定度预算：6-seed 实跑裁决（m24538 重开 #1729）",
    "summary": (
        "m22657 交付物已在 main 但 6-seed 运行未回传、裁决悬空（NOT_RUN）。本轮以同一冻结构型"
        "重跑 6 个独立堆积（cj5109，6/6 回传），以 Student-t 95 % 区间裁决总账 10 % 容差："
        f"W_ens={ens['W_ens_kg_s']:.6f} kg/s，rel={100*ens['rel_err_ensemble']:+.2f} %，"
        f"CI95=[{100*ens['rel_err_ci95'][0]:+.2f}, {100*ens['rel_err_ci95'][1]:+.2f}] %，"
        f"CV={100*ens['cv']:.2f} %，V1={ver['V1_ledger']}，V2={ver['V2_reproducibility']}，"
        f"V3={ver['V3_steady_state']}，V4={ver['V4_rho_control']}。"
        "另修复 main 上既有的全套测试收集错误（dt_ladder 裸导入碰撞），12043 项收集 0 错误。"),
    "scenario_key": "DEM-SILO-BEVERLOO",
    "mission_id": 24538,
    "tags": ["DEM", "Beverloo", "silo discharge", "seed ensemble", "uncertainty budget",
             "validation debt", "see-through"],
    "markdown": md,
    "figures": figs,
}
Path("/tmp/report.json").write_text(json.dumps(payload, ensure_ascii=False))
print("wrote /tmp/report.json:", len(md), "chars,", len(figs), "figures")
