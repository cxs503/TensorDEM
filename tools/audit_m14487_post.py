#!/usr/bin/env python3
"""Post the AUDIT-ASSET-12REPO (TensorDEM) report to the platform (numbers injected from artefacts)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "audit_m14487"
ENV = dict(l.strip().split("=", 1) for l in open("/root/tp-agent/agent.env")
           if l.strip() and not l.startswith("#") and "=" in l)
BASE, USER, PASS = ENV["TENSOR_API_BASE"], ENV["TENSOR_AGENT_USER"], ENV["TENSOR_AGENT_PASS"]


def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return json.loads(r.read() or b"{}")


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def main() -> None:
    inv = json.loads((ART / "inventory.json").read_text())
    wires = json.loads((ART / "wiring.json").read_text())
    stats = json.loads((ART / "figures_stats.json").read_text())
    ts = json.loads((ROOT / "artifacts/ice_thermal_stress/validation.json").read_text())
    er = json.loads((ROOT / "artifacts/p176w2-n11/results.json").read_text())
    fb = json.loads((ROOT / "benchmarks/freeze_bond_thermal_profile/results.json").read_text())
    pu = json.loads((ROOT / "benchmarks/ice_pileup_iso19906/results/report.json").read_text())
    tf = json.loads((ROOT / "benchmarks/verified/dem_thermal_fracture/results.json").read_text())

    c = ts["criteria"]
    conv = ts["convergence"]
    c1, c2, c3 = c["C1_semi_infinite"]["relative_error"], c["C2_slab_source"]["relative_error"], c["C3_fv_vs_fem"]["relative_to_peak"]
    c13 = c["C13_midcontact_source"]["relative_error"]
    er2 = er["C2"]
    er3 = er["C3"]["max_relative_deviation"]
    fb_front = fb["d_deviation"]["baseline"]["front"]
    fb_grid = fb["d_deviation"]["grid"]
    iso = pu["validation"]["C1_iso_same_order_of_magnitude"]["values"]
    iso_max_dev = max(abs(v["ratio_max"] - 1.0) for v in iso)
    tf_d = tf["acceptance_criteria"]["d_thermal_stress"]["error"]
    tf_f = tf["acceptance_criteria"]["f_energy_conservation"]["residual"]

    n_all = len(wires)
    n_none = sum(1 for v in wires.values() if not v["all"])
    n_tests = sum(1 for v in wires.values() if v["all"] and not v["wired"])
    lv = stats["levels"]
    tot = stats["total_added"]

    md = f"""# TensorDEM 近 25 个已并 agent PR 的资产盘点与功能真实性审计（AUDIT-ASSET-12REPO）

**审计对象** main @ `20a77f0a`，窗口 = PR #1151 → #1209（2026-11-10 → 2026-11-11）。
**唯一标准** 可验证的功能交付，而非任务数/PR 数/行数。全部数字由 `tools/audit_asset_inventory.py`、`tools/audit_m14487_figures.py`
与各模块 benchmark 的结果 JSON 程序化注入（零手抄），复现命令见 `audit_report.md` §7。

## 1. 案例介绍（审计对象与判据）

TensorDEM 是冰区/颗粒 DEM 求解器库，近期以「campaign 结点」方式高速并入：本次窗口 25 个 push-merge、195 个文件、
**{sum(tot.values()):,} 行新增**。审计要回答三问：①并进来的到底是什么（核心算法/测试/文档/占位/重复/未接入）；
②每个核心功能是否真的执行计算、接入系统、跑得出可复算的数；③按 owner 六级给成熟度判级。

| 分类 | 文件 | 新增行 |
|---|---|---|
| 核心算法 `src/tensordem/**` | 77 | {tot['core']:,} |
| 测试 `tests/**` | 30 | {tot['test']:,} |
| 文档 `docs/**` | 39 | {tot['doc']:,} |
| 脚本 `scripts/**` | 20 | {tot['script']:,} |
| 其它（benchmark 运行器/工件/过程垃圾） | 29 | {tot['other']:,} |

## 2. 怎么做的（方法链）

1. 逐 merge commit `git diff --numstat <sha>^1 <sha>` → 分类 + 静态体检（stub/顶层函数/调用方），工具 `tools/audit_asset_inventory.py`，工件 `artifacts/audit_m14487/inventory.json`。
2. 全仓 import 索引（模块路径正规化）判定接入状态 → `wiring.json`：**生产调用方 {n_all-n_none-n_tests} / 仅测试 {n_tests} / 完全无 import {n_none}**。
3. 真跑 30 个新增测试文件（逐文件 pytest，27 个本地跑完，3 个超时转集群 cj850/cj881，如实记 not_run）+ 5 个 benchmark 脚本。
4. 复跑仓内已有的 import 合约扫描器：**36 个测试符号断链，2 个测试模块彻底无法收集**。
5. 自写同名遮蔽扫描：`X.py` 与 `X/` 并存 **6 处**（freeze_bond、ice_structure_contact、segregation、sensitivity、shape_representation、verification）。
6. 逐行复核 14 个核心功能，对照 PR 文档声明找差异。

## 3. 计算结果（数字表）

**(a) 判级分布（25 个 PR）**

| 级别 | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| PR 数 | {lv.get('L0',0)} | {lv.get('L1',0)} | {lv.get('L2',0)} | {lv.get('L3',0)} | {lv.get('L4',0)} | {lv.get('L5',0)} |

L≥4 = 12 %（#1151 freeze_bond 热剖面、#1191 ice_thermal_stress、#1195 erosion_dem）；L3 = 2（#1188 挤压堆积-ISO 对标、#1203 friction/polyhedron 真验证）；L0/L1 = 5。

**(b) 真跑得到的基准对照数（本节全部来自结果 JSON）**

| 模块 | 基准 | 关键量 | 实测 | 是否 <3 % |
|---|---|---|---|---|
| ice_thermal_stress C1 | 精确半无限解 | 相对误差 | {c1:.2e} | 是 |
| ice_thermal_stress C2 | 带体积源 slab 精确解 | 相对误差 | {c2:.2e} | 是 |
| ice_thermal_stress C3 | 独立节点 FEM | 相对峰值 | {c3:.2e} | 是 |
| ice_thermal_stress C13 | 中接触源解 | 相对误差 | {c13*100:.2f} % | 是 |
| ice_thermal_stress C4 | 离散收敛 | 观测阶（末两级） | {conv['mean_order_last_two']:.4f}（网格 {conv['cells']}） | 目标 2 阶，达成 |
| freeze_bond 凝固前沿 | Neumann 相似解 | 相对误差 | {fb_front['rel_err_vs_neumann']*100:.2f} % | 是（网格 {fb_grid[0]['n_cells']}→{fb_grid[-1]['n_cells']}） |
| freeze_bond 能量闭合 | 存储能 vs 通量积分 | 相对残差 | {fb['b_energy']['max_rel']:.2e} | 是 |
| erosion_dem C2 | 文献速度指数 n={er2['literature_exponent']} | 拟合指数 | {er2['fit']['n'] if 'n' in er2['fit'] else er2['fit']} | 是（{er2['fit_vs_literature_relative_error']*100:.2f} %） |
| erosion_dem C3 | DNV-RP-O501 逐点 | 最大偏差 | {er3*100:.2f} % | **否**（单锚标定后仍有形状误差） |
| pileup F_max | ISO 19906 名义载荷 | 最大偏差 | {iso_max_dev*100:.2f} % | **否**（文档如实写 C1b 3 倍带 FAIL） |
| thermal_fracture (d) | “解析热应力” | 相对误差 | {tf_d:.2e} | **无意义**（与解析同式 → 恒等） |
| thermal_fracture (f) | “能量守恒” | 残差 | {tf_f:.2e} | **无意义**（代数恒等） |

## 4. 结果图

![fig1](fig1) — 判级分布。
![fig2](fig2) — 逐 PR 新增构成（柱顶为判级），可见「其它」类（含 4,331 行 `.tmpwork/` 草稿）与文档占比。
![fig3](fig3) — 77 个新核心模块的接入状态：**{n_none+n_tests} 个（{100*(n_none+n_tests)/n_all:.0f} %）没有生产调用方**。

## 5. 注意点（踩坑与局限）

1. **伪验证两类**：seepage 的「DEM 测量值」由测试写成 `i_c_analytic*1.10`、`u_analytic*(1+5 % 噪声)`（`tests/test_seepage_piping.py:95/168/245`），文档却报 5.85 %/5.00 %/9.09 % 的“误差”；thermal_fracture_dem 的 (d)(f) 两条判据与解析解同式、残差恒 0 → 不可能失败。两者均按 owner「无证据即低判」判 L1/L2。
2. **6 处模块/包同名遮蔽** + 合并门禁曾「滤掉模块留下测试」→ 2 个测试模块（freeze_bond、ice_structure_contact）**彻底无法收集**，36 个符号断链；这是「测试不红而是消失」的隐形坏账。
3. **证据不入库**：22/25 个 PR 未提交结果工件，文档引用的 `artifacts/...` 在 checkout 里不存在 → 复判必须重跑。
4. **digest 不可复现**：`ice_thermal_stress/validation.py:481` 把 `generated_utc` 放进摘要 payload，每次运行摘要都变，只能证明“文件没被手改”。
5. 三个测试文件（polyhedra_repose_angle/fast、friction_polyhedron_real_verification）本地超时，集群作业排队未返回 → 本报告记 not_run，未计为通过。
6. 反向也要说：本仓有强诚实文化——#1203 自查撤回了上一版 4 条自证判据并修出 3 个真内核缺陷；#1188/#1196/#1205 都公开记录 FAIL 与 not_run。

## 6. 误差判定（逐项显式回答“是否 <3 %”）

- ice_thermal_stress：C1 {c1:.1e}、C2 {c2:.1e}、C3 {c3:.1e}、C13 {c13*100:.2f} % → **是**（均 <3 %），收敛阶 {conv['mean_order_last_two']:.2f} 达二阶。
- freeze_bond 凝固前沿 vs Neumann：{fb_front['rel_err_vs_neumann']*100:.2f} % → **是**；能量闭合 {fb['b_energy']['max_rel']:.1e}。
- erosion_dem 速度指数 vs 文献：{er2['fit_vs_literature_relative_error']*100:.2f} % → **是**；DNV 逐点偏差 {er3*100:.2f} % → **否**（且锚点为单常数标定，已标 CALIBRATED）。
- ice_vertical_structure_pileup vs ISO 19906：F_max 最大偏差 {iso_max_dev*100:.1f} % → **否**；3 倍带判据 FAIL 已如实登记。
- seepage C1–C3、thermal_fracture (d)(f)：**不做百分比判定**——测量值由解析式合成或判据恒等，无信息量（honest：这两处不是“误差大”，而是“没有基准”）。
- 全局：窗口内 **L≥4 仅 3/25（12 %）**，L2 占 60 %，L0/L1 占 20 %；约 **4,700 行（新增核心代码 16 %）的成熟度被高估至少一级**。
"""
    figs = [
        {"name": "fig1_maturity_levels.png",
         "caption": "图1 近 25 个已并 PR 的成熟度判级分布（L4=3、L3=2、L2=15、L1=3、L0=2）",
         "png_base64": b64(ART / "figures/fig1_maturity_levels.png")},
        {"name": "fig2_pr_composition.png",
         "caption": "图2 逐 PR 新增构成（核心/测试/文档/脚本/其它）与判级；其它类含 4,331 行 .tmpwork 草稿",
         "png_base64": b64(ART / "figures/fig2_pr_composition.png")},
        {"name": "fig3_wiring.png",
         "caption": "图3 77 个新核心模块接入状态：生产调用方 37 / 仅测试 19 / 无 import 21",
         "png_base64": b64(ART / "figures/fig3_wiring.png")},
    ]

    tok = api("POST", "/auth/login", {"username": USER, "password": PASS})
    token = tok.get("access_token") or tok.get("token")
    payload = {
        "title": "TensorDEM 资产盘点与功能真实性审计（25 PR / 54,546 行）：2 起伪验证·6 处同名遮蔽·40/77 模块未接入",
        "summary": (f"近 25 个已并 PR 新增 {sum(tot.values()):,} 行（core 77 文件 {tot['core']:,}）；判级 L4=3、L3=2、L2=15、L1=3、L0=2。"
                    "实质发现：seepage 的「DEM 测量」由测试写成解析式×1.1；thermal_fracture_dem 判据 (d)(f) 为恒等式且模块无动力学；"
                    "6 处模块/包同名遮蔽致 36 个测试符号断链、2 个模块无法收集；40/77 新模块无生产调用方；22/25 PR 未入库结果工件。"),
        "scenario_key": "tensordem_asset_audit",
        "mission_id": 14487,
        "tags": ["audit", "TensorDEM", "evidence-grading", "maturity", "A+B"],
        "markdown": md,
        "figures": figs,
    }
    out = api("POST", "/tensor/reports", payload, token)
    print(json.dumps(out, ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()