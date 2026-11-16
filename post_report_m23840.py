#!/usr/bin/env python3
"""Publish the m23840 report (six elements, figures embedded, numbers read from artifacts)."""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("/srv/tensor/agent/work/m23840")
RUNS = ROOT / "benchmarks" / "case_rcp_deposition" / "runs"
FIGS = [
    ("/tmp/m23840_figs/fig1_ladders_anchors.png",
     "Stiffness ladders (N=10000, mu=0/0.1) with hard-sphere intercepts, the "
     "nuisance-order bracket, ledger band [0.62,0.66], literature RCP 0.6366 "
     "(Scott 1969), the derived table entry 0.625 (mu=0.1) and the "
     "disordered-packing bracket [0.56,0.648]."),
    ("/tmp/m23840_figs/fig2_verdict_map.png",
     "Ledger reference gates after the m21145 anchor policy: green = certified "
     "vs literature anchor (mu=0); orange = WITHHELD, bracket-only (mu=0.1, "
     "derived anchor). Bar = |phi/phi_ref-1| kept for ranking."),
]


def base() -> str:
    b = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    return b if b.endswith("/api/v1") else b + "/api/v1"


def login() -> str:
    data = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base() + "/auth/login", data=data,
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read())
    return body.get("access_token") or body.get("token") or body["data"]["access_token"]


def main() -> int:
    summ = json.loads((RUNS / "summary.json").read_text())
    aud = json.loads((RUNS / "anchor_provenance_audit.json").read_text())
    g = {x["id"]: x for x in summ["gates"]}
    G12, G13 = g["G12_stiffness_limit_vs_ref"], g["G13_stiffness_limit_vs_ref_frictional"]
    G18, G18b = g["G18_production_vs_friction_ref"], g["G18b_production_vs_friction_ref"]
    A = {x["id"]: x for x in aud["gates"]}
    t01 = aud["tiers"]["0.1"]
    fl = aud["tiers"]["0"]

    def row(gid, g, what):
        return (f"| {gid} | {what} | {g['got']:.5f} | {g['phi_ref']:.4f} "
                f"({g['anchor_provenance']}) | {100*g['value']:+.2f}% | "
                f"{g['claim_status']} | {g['verdict']} |")

    md = f"""# m23840 · TensorDEM#1699 管到底第1轮 — m21145 锚点溯源策略落到合并账本路径（G13/G18b → WITHHELD）

## ① 案例介绍（物理背景 / 几何 / 参数）

**物理背景**：单分散球体随机密堆（RCP）。无摩擦经典值 φ_RCP ≈ 0.6366（Scott & Kilgour 1969；
Bernal & Mason 1960）；摩擦接触稳定更松的力链构型，φ 随 μ 增大而下降（Pournin 2001；Silbert 2010；
Onoda & Liniger 1990 的 RLP 下限 ≈0.56）。

**算例**：`CASE-DEM-RCP-PACKING-FRACTION`（DEM 倾倒堆积基准）。

| 参数 | 值 |
|---|---|
| 颗粒数 N | 10000（生产档，两摩擦档同 N） |
| 摩擦档 μ | 0.0 / 0.1（恢复系数 e=0.1） |
| 刚度阶梯 k_n | 4000 / 16000 / 64000 / 256000（生产 k_n=4000） |
| 时间步 | dt = t_c/20（另有 dt/2 配对探针） |
| 种子 | 生产档 ≥3 seeds/档；k_n=256000 顶档每档 1 seed |
| 账本带 | φ ∈ [0.62, 0.66]；精度判据 <3% |

**任务背景**：原 PR#1699（m21145 matured deliverables，5 个文件）被 owner 于 2026-11-05 裁定关闭；
其功能要点已经由 #1579 并入 main（本单开工时逐文件核对：`rcp_references.py`、
`anchor_provenance_audit.py`、`test_rcp_reference_provenance.py` 与 m21145 提交逐字节一致，
JSON/MD 为后续 m22859 重跑更新）。**剩余缺口**：m21145 的锚点策略（精度声称只允许对
`literature` 锚；`derived`/`unverified` 只能排序或做必要条件括号检验）只在审计路径生效，
**合并账本路径**（`aggregate.py` → `runs/summary.json`）的 G13/G18b 仍以自声明插值的
0.6250（derived）为基准给出 PASS —— 正是 m21145 指出的「两条分析路径判同一个截距」问题的
账本侧残留。

## ② 怎么做的（方法 / 流程 / 工具链）

1. **策略函数化**：`aggregate.py` 新增 `ref_gate()`——锚点为 `literature` 时门控为
   <3% 精度声称（`claim_status=CERTIFIED`, verdict PASS/FAIL）；否则降级为无序堆积括号
   检验（`claim_status=WITHHELD`, verdict PASS_BRACKET/FAIL_BRACKET，`pass`=括号内），
   偏差值保留供排序。G12/G13（硬球截距）与 G18/G18b（生产值）同策。
2. **跨路径强制**：`anchor_provenance_audit.py` 新增 `ledger_claims()`，A5 门从「只查本审计
   判定」扩展为「本审计 + 合并账本」双路径扫描；无 provenance 标记的参考门控本身即违规
   （防止回退 aggregate.py 后 A5 静默放行）。
3. **产物再生**：`runs/summary.json`、`docs/CASE_RCP_DEPOSITION.md`、
   `runs/anchor_provenance_audit.json`、`docs/AUDIT_m21145_RCP_ANCHOR_PROVENANCE.md` 全部由脚本重生成，零手抄。
4. **测试钉死**：`tests/test_rcp_ledger_anchor_policy.py`（11 项）钉住：G13/G18b 必须
   WITHHELD+PASS_BRACKET；G12/G18 保持 CERTIFIED；账本内不得存在对非文献锚的认证声称；
   旧格式（无 provenance 标记）账本必须被 `ledger_claims()` 检出。
5. 工具链：Python 3.11 / numpy / matplotlib；`PYTHONPATH=src python aggregate.py`、
   `anchor_provenance_audit.py`、`repro_gate.py`；pytest。改动 7 文件（≤10），单一功能点。
   热文件声明：`aggregate.py`/`anchor_provenance_audit.py` 为本算例基准脚本（非 TensorDEM
   核心模块）；沙箱内无法查询 GitHub 开 PR 列表，重叠风险在本文声明。

## ③ 计算结果（数字全部程序化注入自 runs/*.json）

**账本参考门控（本单改动对象）**

| 门 | 量 | 计算值 | 锚点 | 偏差 | 声称状态 | 判定 |
|---|---|---|---|---|---|---|
{row("G12", G12, "硬球截距 mu=0")}
{row("G13", G13, "硬球截距 mu=0.1")}
{row("G18", G18, "生产值 mu=0")}
{row("G18b", G18b, "生产值 mu=0.1")}

**审计门控（A1–A6，含扩展后的 A5）**：{ "、".join(f"{k} {'PASS' if v['pass'] else 'FAIL'}" for k, v in A.items()) }

**mu=0.1 刚度阶梯**（N=10000）：{ "；".join(f"k_n={p['kn']:g} → φ={p['phi']:.5f} ({p['n_seeds']} seed)" for p in t01['points']) }；
四点拟合 φ∞={t01['four_level']['limit']:.5f}（p={t01['four_level'].get('order')}），
留一预测误差 {t01['selfcheck']['prediction_error']:.2e}；阶次括号 p∈[0.5,1.5] →
φ∞∈[{t01['order_bracket']['lo']:.5f}, {t01['order_bracket']['hi']:.5f}]。
mu=0 四点拟合 φ∞={fl['four_level']['limit']:.5f}。

**测试**：RCP 全族本地通过（test_rcp_* 共 219 项：100+15+6+98，含新增 11 项）；
全套 pytest 已递交集群 cj4810（stage_dir=本工作树，16 核，3600 s 上限）。

## ④ 结果图

（见内嵌图 1/2。本单为策略/审计类任务，无新物理场量，故无云图；图 1 为阶梯-锚点比较图，
图 2 为判定映射图，均自已提交产物生成。）

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. mu=0.1 表值 0.6250 是**自声明插值**（Pournin 2001 的 DEM 研究内插），仓内无原文表值
   （页码+表号）转录，故该档精度声称保持 WITHHELD；一旦补入文献转录，把该条 provenance
   改为 `literature` 即可自动升级 PASS/FAIL，代码无需改动。
2. k_n=256000 顶档每档仅 1 seed（原 docstring 误写 two seeds，已更正）；截距由最软两级
   主导，顶档只进入留一检验，单种子影响有限但未重估。
3. G5/G6/G9 仍 FAIL（生产刚度收敛 0.73%/0.84% > 0.5% 门、残余重叠 1.15% > 1% 门）——
   既有欠账，与本单无关，未动。
4. 合并前 repro_gate 对 HEAD 显示 STALE（新增 verdict/claim_status 叶子，属预期）；
   合并后重跑即 REPRODUCED（本单工作树内新码+新产物自洽，drift=0）。
5. 历史文档 `docs/CASE_RCP_PACKING_FRACTION_M20918.md` 仍含旧 −1.22% PASS 表述，作为
   历史记录保留，由 m21145 审计报告取代。
6. 适用范围：本策略只约束「精度声称」的锚点资格，不改变任何求解器方程、判据数值或
   既有门控布尔（G13/G18b 的 `pass` 布尔前后均为 True，`all_gates_pass` 不变）。

## ⑥ 误差判定（关键量 vs 基准，逐项显式回答 <3%?）

| 关键量 | 基准 | 计算值 | 误差 | <3%? |
|---|---|---|---|---|
| 生产值 mu=0 | 0.6366（literature, Scott 1969） | {G18['got']:.5f} | {100*G18['value']:+.2f}% | **是**（认证 PASS） |
| 硬球截距 mu=0 | 0.6366（literature） | {G12['got']:.5f} | {100*G12['value']:+.2f}% | **是**（认证 PASS） |
| 生产值 mu=0.1 | 0.6250（derived，非文献转录） | {G18b['got']:.5f} | {100*G18b['value']:+.2f}% | **无认证基准**（锚点非文献）→ WITHHELD；最强可声明：括号内 → PASS_BRACKET |
| 硬球截距 mu=0.1 | 0.6250（derived） | {G13['got']:.5f} | {100*G13['value']:+.2f}% | **无认证基准**（同上）→ WITHHELD；PASS_BRACKET |

无基准原因声明：mu=0.1 档唯一可用锚点是本仓构造的插值点，以其为基准的「<3%」是
自指（参考值由同族 DEM 数字构造），按 m21145 策略不构成验证，故不回答「是/否」而
显式记 WITHHELD——这是本单的核心结论，不是含糊。
"""
    figures = []
    for path, cap in FIGS:
        figures.append({"name": Path(path).name, "caption": cap,
                        "png_base64": base64.b64encode(Path(path).read_bytes()).decode()})
    payload = {
        "title": "m23840 · TensorDEM#1699 管到底第1轮：m21145 锚点溯源策略落到合并账本路径（G13/G18b WITHHELD）",
        "summary": "m21145 锚点策略（精度声称仅对 literature 锚）此前只在审计路径生效；本单把策略落到合并账本路径：G13/G18b 对 derived 0.6250 锚的 PASS 降级为 WITHHELD+PASS_BRACKET，A5 门扩展为双路径扫描，4 个产物重生成，11 项新测试钉死策略。RCP 全族 219 项测试本地通过，全套递交集群 cj4810。",
        "scenario_key": "CASE-DEM-RCP-PACKING-FRACTION",
        "mission_id": 23840,
        "tags": ["TensorDEM", "RCP", "anchor-provenance", "ledger", "m21145", "see-through"],
        "markdown": md,
        "figures": figures,
    }
    tok = login()
    req = urllib.request.Request(
        base() + "/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {tok}",
                 "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            print("HTTP", r.status, r.read()[:400].decode())
    except urllib.error.HTTPError as e:
        print("HTTP ERR", e.code, e.read()[:800].decode())
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
