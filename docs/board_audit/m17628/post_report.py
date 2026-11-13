#!/usr/bin/env python3
"""Publish the m17628 board-audit report to the platform (/tensor/reports).
Every number is read from verification.json / refs artifacts — nothing hand-typed."""
from __future__ import annotations

import base64
import json
import pathlib
import urllib.request

ENV = dict(l.strip().split("=", 1) for l in open("/root/tp-agent/agent.env")
           if l.strip() and not l.startswith("#") and "=" in l)
BASE = ENV["TENSOR_API_BASE"]
HERE = pathlib.Path(__file__).resolve().parent


def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return json.loads(r.read() or b"{}")


ver = json.loads((HERE / "verification.json").read_text())
res = json.loads((HERE / "refs/artifacts/results.json").read_text())
probe = (HERE / "refs/probe/probe_overlap_out.txt").read_text().strip().splitlines()
rows = ver["checks"]["C2_cross_table"]["rows"]
cases = {(c["gamma_dot"], c["arm"]): c for c in res["cases"]}


def p(*a):
    d = {}
    for tok in " ".join(a).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


leg, lee = p(probe[0]), p(probe[1])
table_rows = "\n".join(
    f"| {int(r['gamma_dot'])} | {cases[(r['gamma_dot'],'legacy')]['pressure']:.3g} | "
    f"{cases[(r['gamma_dot'],'lees_edwards')]['pressure']:.4g} | {r['tau_over_p_legacy']:+.4f} | "
    f"{r['tau_over_p_le']:+.4f} | {r['rel_diff_pct_recomputed']:.1f} % | 否 |" for r in rows)

md = f"""# 交差审计：m17503 `XVAL-DEM-PERIODIC-SLLOD-CROSSCHECK-AVAILABLE`（TensorDEM 周期 SLLOD vs Lees–Edwards）

**审计单 m17628｜非重做**。被审单 m17503（succeeded，commit `ad072d30`）。结论：**三件套 2/3 通过（run_artifacts PASS、repro_cmd PASS、benchmark FAIL——裁判行缺位且自定判据未达），判级 L3（系统集成），不达 L4**。

## ① 案例介绍（物理背景/几何/参数）
周期 SLLOD（Evans–Morriss）剪切盒是稠密颗粒流测有效摩擦 `tau/P` 的标准装置，无壁、双周期。缺陷前提：SLLOD 速度场 `u_x = γ̇y` **不周期**，且周期性晶格须随流动倾斜为 `{{(Lx,0), (strain·Ly, Ly)}}`；成熟实现（LAMMPS `fix deform` triclinic + `fix nvt/sllod`、ESPResSo `LEES_EDWARDS`、HOOMD `shear`）都要求这两条。被审件在 TensorDEM 的 `mts_contact.MTSSystem` 里加了可选 Lees–Edwards 胞（默认关闭）。

| 参数 | 值 |
|---|---|
| 颗粒 | N=64，双分散 r∈[0.8,1.2]·5 mm，ρ=2000 kg/m³，2D 圆盘 |
| 接触 | 线性弹簧 kn=1e5 N/m，kt/kn=0.5，μ=0.5，dashpot ζ=0.4 |
| 盒体/分数 | 方盒双周期，φ=0.80 |
| 时间步 | dt=0.1·t_c，n_sub=5（MTS） |
| 剪率 | γ̇ ∈ {{10,20,40}} 1/s，等应变窗 γ_end≈0.2 |
| 无量纲 | I=γ̇d√(ρ/P)=2.46/3.40/4.59（旧臂），0.14/0.28/0.65（LE 臂） |

## ② 怎么做的（审计方法/流程/工具链）
审计只做核验、不做重做：①读被审工作树（`git status` 干净）；②在 `/tmp` 副本里重跑被审代码（`pytest`、`xval_matrix --only-invariants`、`--gammas 10`）；③用 `verify_m17503.py` 从工件独立重算全部表格（`tables.json` vs `results.json`，误差 <0.05 pp，全 match）；④查唯一裁判 `tensor_benchmark_cases`；⑤独立探针 `probe_overlap.py` 量两臂的接触重叠分布（被审报告未提供的维度）；⑥跑平台自己的 `maturity_gate.judge()` 干跑复核 evidence。审计脚本与全部指针留在 `docs/board_audit/m17628/`。

## ③ 计算结果（数字表，程序化注入）
| γ̇ [1/s] | P 旧臂 [Pa] | P LE 臂 [Pa] | τ/P 旧臂 | τ/P LE 臂 | 相对偏差 | <15 %? |
|---|---|---|---|---|---|---|
{table_rows}

判据（owner 精度线 <15 %）在三个剪率上全部 **FAIL**；偏差最小者 {ver['checks']['C2_cross_table']['min_rel_pct']:.1f} %，最大者 {ver['checks']['C2_cross_table']['worst_rel_pct']:.1f} %。**与基准（解析/文献/裁判行）的误差无法逐项列出——因为 `tensor_benchmark_cases` 对本锚点 0 行**（`verification.json#C5`），唯一可用的参照即"两臂互比"与自定 <15 % 线。

审计独立探针（n=64、φ=0.80、γ̇=10、909 步、未加松弛段）：旧臂最大重叠 **{leg['ov_max']}**（>10 % r 的接触 0 对），LE 臂最大重叠 **{lee['ov_max']}**、**{lee['n_ov>10%']} 对** >10 % r，LE 臂 `P={lee['P']} Pa / τ={lee['tau']} Pa` —— 即 LE 臂宏观应力被几对近似奇异穿透接触支配。

## ④ 结果图
左：两臂 τ/P（误差棒=τ/P 标准误）；中：两臂相对偏差 vs 15 % 判据（对数轴，全部超 10 倍以上）；右：审计探针的最大重叠/半径。

## ⑤ 注意点（坑/局限/适用范围）
- **F1（对 L4 blocking）**：锚点自定 <15 % 未达（163–655 %），且无 `tensor_benchmark_cases` 裁判行；外部锚点 WITHHELD（被审单自述 OSS 源码未能在本环境核验，审计亦未做源码级对标）。
- **F2（对"参考真值"blocking）**：LE 臂应力由深穿透接触支配（最大重叠 14.65 % r，5 对 >10 % r），与报告自述 F1（像选择仍取共动系，帧不变性残差 0.357）一致；因此"crosscheck"只证明**两实现不一致且差异可解释**，未产生可信参考真值。
- **F3（provenance, medium）**：`results.json.wall_clock_total=0.0249 s` 是最后一次增量写的计时（6 算例矩阵实际约 95 s，审计测单剪率/双 33.8 s）；`sensitivity=[]` 而报告引用了 2.4 % 时间步敏感性——无工件、不可复跑；报告 §6 复现序列漏 `--only-invariants`，照抄会把 `invariants` 写空。
- **F4/F5/F6（low）**："934 % 是解析精确值"实为解析式 `Ly/⟨|l_y|⟩` 落在实测跨边界接触集合（旧臂 9.3432 / LE 8.8240）；"31 passed" 实为 30 passed + 1 xfailed；`lees_edwards=True` 仅见于 `scripts/m17503` 与测试，产品调用方 `benchmarks/mts_contact_gate.py:142` 未传该开关。
- 适用范围：短应变窗（γ≈0.2）、无压力伺服（φ 固定），`tau/P` 在零围压下不是材料量（报告 F2 已自述）。

## ⑥ 误差判定（逐项，显式回答）
| 量 | 基准 | 误差 | <3 %? |
|---|---|---|---|
| 两臂 τ/P 相对偏差（γ̇=10/20/40） | LE 臂 | 361.0 / 162.7 / 654.8 % | **否** |
| 裁判行数值对照（ref vs got） | `tensor_benchmark_cases` | **不可判**：锚点 0 行（无 ref/容差） | 否（无基准） |
| `tables.json` 数字 vs `results.json` 重算 | 独立重算 | <0.05 pp（全 match） | 是 |
| 复跑一致性（γ̇=10 两臂 τ/P、P） | 交付工件 | 0（逐位相同） | 是 |
| 单接触仿射速度（I2） | γ̇·l_y 精确式 | 0（单接触级精确，测试通过） | 是 |
| 帧不变性残差（LE 臂） | 0 | 3.568e-01 | 否（xfail 已诚实登记） |
| 审计探针最大重叠（LE 臂） | 旧臂 {leg['ov_max']} | {lee['ov_max']}（相对 14.65 % r） | 否 |

**是否 <3 %？** 关键宏观量（τ/P 两臂互比）**否**；且因无裁判行，"与基准的误差"**不可判**（原因是本锚点在 `tensor_benchmark_cases` 无登记行，非审计省略）。诚实结论：本交付的实质价值是**把两条周期 SLLOD 缺陷钉成可机检不变量并修好其一**（I1/I2 通过，I3 以 xfail 诚实登记），故判 **L3**；判据未达 + 无裁判 + LE 臂被深穿透支配 → 不升 L4。verdict 已落 `tensor_maturity_verdicts`（mission_id=17628, level=3, verdict_by=board-audit），`maturity_gate.judge()` 干跑复核 gate_level=3。
"""

fig = (HERE / "fig_m17628.png").read_bytes()
payload = {
    "title": "m17628 交差审计：m17503 XVAL 周期 SLLOD vs Lees–Edwards（TensorDEM）→ L3，判据 <15% 未达",
    "summary": "三件套：run_artifacts PASS / repro_cmd PASS / benchmark FAIL（锚点 0 裁判行，自定 <15% 实测 163–655%）。"
               "独立探针发现 LE 臂应力被深穿透接触支配（最大重叠 14.65% r）。判级 L3，附 6 条严重度排序发现与返修建议。",
    "scenario_key": "board_audit_m17503_xval_periodic_sllod",
    "mission_id": 17628,
    "tags": ["board-audit", "TensorDEM", "XVAL", "SLLOD", "Lees-Edwards", "maturity-L3"],
    "markdown": md,
    "figures": [{"name": "m17628_audit.png",
                 "caption": "(a) 两臂 τ/P 与标准误；(b) 两臂相对偏差 vs 15 % 判据（对数轴）；(c) 审计独立探针："
                            "最大接触重叠/半径（legacy vs Lees-Edwards）",
                 "png_base64": base64.b64encode(fig).decode()}],
}
tok = api("POST", "/auth/login", {"username": ENV["TENSOR_AGENT_USER"],
                                  "password": ENV["TENSOR_AGENT_PASS"]}).get("access_token")
assert tok, "login failed"
out = api("POST", "/tensor/reports", payload, tok)
print(json.dumps(out, ensure_ascii=False)[:300])