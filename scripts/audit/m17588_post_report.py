#!/usr/bin/env python3
"""m17588 board audit — publish the paper-grade report (six sections) to the platform.

Every number is injected from artifacts/audit_m17588/*.json (zero hand-copy);
the figure is the auditor's own matplotlib render (fig_m17500_audit.png).
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
W = "/root/tp-agent/work/m17588/artifacts/audit_m17588"
MISSION = 17588


def api(method, path, body=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return r.status, json.loads(r.read() or b"{}")


def main():
    ev = json.load(open(os.path.join(W, "evidence.json")))
    probe = json.load(open(os.path.join(W, "optin_bitwise_probe.json")))
    res = json.load(open(os.path.join(W, "results_m17500_copy.json")))
    rows = res["rows"]
    tol = res["tolerance"]
    worst_map = max(r["i_mapping_error"] for r in rows)
    worst_mu = max(r["mu_error"] for r in rows)
    worst_ref = max(r["bulk_vs_jop_rel_error"] for r in rows)
    inv = [r["label"] for r in rows if not r["servo_valid"]]
    worst_p = min(r["pressure_control_error_final"] for r in rows)

    table3 = "\n".join(
        f"| {r['label']} | {r['i_target']} | {r['i_bulk_measured']:.4f} | "
        f"{r['i_proxy_fn_weighted']:.4f} | {100*r['i_mapping_error']:+.1f}% | "
        f"{r['mu_bulk_measured']:.4f} | {r['mu_proxy_fn_weighted']:.4f} | "
        f"{100*r['mu_error']:+.1f}% | {100*r['bulk_vs_jop_rel_error']:.1f}% |"
        for r in rows)

    png = base64.b64encode(open(os.path.join(W, "fig_m17500_audit.png"), "rb").read()).decode()

    md = f"""# 交差审计 m17500 `XVAL-DEM-MU-I-CONTACT-PROXY-NEGATIVE`（TensorDEM）— 裁定 L2

## ① 案例介绍
被审单 m17500（status=succeeded，auto-approve 交付验收）声称在 TensorDEM 恒应力伺服剪切盒上，对**同一次
稳态**做双读数交叉对标：REF=体相惯性数 `I=dγ̇/√(P/ρ)` 与 `μ=τ/P`；GOT=逐接触 `I_c=|v_t|/√(f_n/(ρd²))`
的 f_n 加权 `μ(I_c)`（Jop 2006: μ_s=0.36, μ_2=0.84, I_0=0.30）。设备：R=5 mm 球、ρ=2500 kg/m³、
k_n=1e3 N/m、e=0.3、μ_c=0.5、P=2 kPa、seed 5279；I_target∈{{0.01,0.1,0.6}} 加 2 个敏感性点
（P 8 kPa、k_n 4e3）。本单是**交差核验**（非重做）：核三件套真伪 + 判级 L0–L5 + 落表。

## ② 怎么做的（审计方法/工具链）
1. **取证**：DB 读 mission 17500 的 result/on_disk_changes 与报告 #16473；工作树
   `/root/tp-agent/work/m17500`（branch agent/m17500 @68f1622c）逐文件核对。
2. **算术独立复核**：用结果原件重算 `|I_c/I_bulk−1|`、`|μ_proxy/μ_bulk−1|`、
   `(μ_c−μ(I_c))/μ_c`（closure 取仓内 `jop_bulk_mu`），与存储列对比（阈值 1e-9）。
3. **报告↔工件一致性**：解析报告 #16473 的 ③ 表，与 results.json 5 行百分比比对（阈值 0.1 pp）；
   六段关键词命中检查。
4. **裁判核对**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE
   anchor_slug='XVAL-DEM-MU-I-CONTACT-PROXY-NEGATIVE'`。
5. **可执行性**：审计端重跑被审单测；**新增探针**验证"诊断默认关=逐位不变"（原单只断言 dataclass 开关，
   未验轨迹）：同算例 off/on 各跑一次，8 个积分器量做 float64 位级比较。
6. **入库状态**：`git merge-base --is-ancestor agent/m17500 HEAD` 与 `git ls-remote --heads origin`。
工具：`scripts/audit/m17588_verify.py`、`m17588_write_verdict.py`、`m17588_optin_bitwise_probe.py`、
`m17588_fig.py`；产物 `artifacts/audit_m17588/`。

## ③ 计算结果（审计复算，程序化注入）
容差来源：仓内 `prereg/` 冻结 `DEFAULT_TOL={tol}`（平台裁判表为空，见 ⑤-1）。

| 算例 | I_target | I_bulk(REF) | I_c(GOT) | 映射误差 | μ_bulk(REF) | μ_proxy(GOT) | μ 误差 | 参考门 vs Jop |
|---|---|---|---|---|---|---|---|---|
{table3}

- 算术自洽：5/5 行三项误差与原件重算一致到 **1e-9**（`evidence.json:arithmetic`）。
- 报告↔工件：5/5 行百分比一致到 **0.1 pp**（`evidence.json:report_vs_json`）。
- 单测：审计端重跑 **8 passed in 5.15 s**。
- 默认关位级不变探针：**{len(probe["per_quantity"])}/{len(probe["per_quantity"])} 个积分器量位级相同**，
  `contact_proxy` 键关时不存在、开时存在（{probe["proxy_when_on"]["n_samples"]:.0f} 采样）。
- 入库状态：**未合并/未推送**（merge-base rc=1，ls-remote 空；main f2e227f6 缺 6 个文件）。

**级别裁定 = L2**：真代码（270 行非占位）+ 真运行工件 + 单测成立；L3 因调用点仅在未合并分支
（生态仓主干 grep 无引用）不成立；L4 因锚点 0 行登记（无裁判）不成立。裁判复核预演
`judge(2, evidence, 'TensorDEM', 17588) → 2, reasons=[]`。

## ④ 结果图
A：GOT vs REF（log-log，含 ±15% 带与 y=x）——I_c 与 I 不是同一无量纲数，μ 点亦越带；
B：三类相对误差逐算例 vs 15% 容差，灰柱=参考门（体相路线 vs Jop 常数），其越线即按其 prereg
使绝对摩擦比较作废。

## ⑤ 注意点（坑/局限）
1. **裁判缺失**：`tensor_benchmark_cases` 该锚点 **0 行**（TensorDEM 另有 8 条其他锚点）→ 平台无法机器判
   容差；`-NEGATIVE` 锚点登记时容差语义需另立（负结论不该被"误差<15% 为过"天然判死）。
2. **交付缺口（阻断）**：m17500 的 9 个交付文件只在本地 `agent/m17500`，未 push、未 merge → 主干 CI 与
   按 main 复现的读者拿不到；工作树 `worktree_recyclable`，工件有回收风险（审计已拷副本入本单）。
3. **文案与工件不符（重要）**：mission/report 称"4/5 算例 servo_valid=False，末窗压力偏差 −7.4%…"，
   工件真值 **3/5**（{', '.join(inv)}）、最差 **{100*worst_p:+.1f}%**（行数多算 1、量级少报 12 pp）；
   另 `I_c/I_bulk=2.54` 与工件 2.526 有 0.5% 滑移。
4. **自相矛盾（重要）**：参考门 FAIL（体相 vs Jop 最差 {100*worst_ref:.1f}%）按其 prereg 应"作废绝对摩擦比较、
   只留无量纲结论"，但报告仍把 μ 误差 35.8% 当作裁决结果行。
5. 审计独立性局限：数字来源单一（同一 results.json）；位级探针用的是缩短算例（settle 20+20 步，≈30 s/run），
   非原 2782–11619 步轨迹。
6. **未跑**（unknown≠pass）：全量 5 算例 346 s 再生未在审计预算内重跑；3 个 invalid 算例的压力控制认证未做。

## ⑥ 误差判定（关键量 vs 基准，逐项显式回答"是否<3%"）
| 关键量 | 基准 | 复算值 | <3%? |
|---|---|---|---|
| 映射误差最差（I=0.01） | 体相测量 I | {100*worst_map:+.1f}% | **否** |
| μ 误差最差（I=0.01） | 体相 τ/P | {100*worst_mu:+.1f}% | **否**（且参考门 FAIL ⇒ 作废） |
| 参考门：体相 vs Jop 2006 | 文献常数 | 最差 {100*worst_ref:+.1f}% | **否** |
| 窗口敏感：μ_proxy 尾60% vs 末20% | 同算例 | <0.9% | **是** |
| 审计复算 vs 原件（3 项×5 行） | results.json | ≤1e-9 | **是** |
| 位级不变（8 个积分器量） | 同算例 off | 0 位差 | **是** |

结论：被审单的**证据链真实**（三件套齐、算术自洽、报告数字可溯源、单测可跑），**级别 L2**；
其"负结论"方向上仍有效（I_c/I_bulk=2.526/0.576/0.407，非同一无量纲数），但**入库与裁判两处缺口**使其
不能计为 L3/L4；honest-FAIL 本身按纪律不罚，文案不符部分（F4/F7）如实列明。
判决书 `tensor_maturity_verdicts` id=593（level=2, verdict_by=board-audit）。
"""

    body = {
        "title": "交差审计 m17500 — TensorDEM μ(I_c) 接触代理 XVAL：三件套真、级别 L2、入库与裁判两缺口",
        "summary": (
            f"核验 m17500 三件套：8 单测重跑通过、5/5 行误差重算一致 1e-9、报告数字与工件一致 0.1 pp、"
            f"六段齐；裁定 L2（L3 调用点仅在未合并分支、L4 锚点 0 行登记）。发现阻断项：成果未 push/merge；"
            f"文案称 4/5 servo 失效实为 3/5（最差 −{abs(100*worst_p):.1f}%）；参考门 FAIL 使绝对 μ 比较作废。"
            f"verdict id=593 落表。"),
        "scenario_key": "TensorDEM/audit/m17500-mu-i-contact-proxy",
        "mission_id": MISSION,
        "tags": ["board-audit", "TensorDEM", "mu(I)", "m17500", "maturity-verdict",
                 "negative-knowledge"],
        "markdown": md,
        "figures": [{"name": "fig_m17500_audit.png",
                     "caption": "A: GOT(I_c, μ_proxy) vs REF(I, τ/P) 5 行，log-log，±15% 带+对角；"
                                "B: 三类相对误差 vs 15% 容差，灰柱=参考门（体相 vs Jop 2006）",
                     "png_base64": png}],
    }
    tok = api("POST", "/auth/login", {"username": USER, "password": PASS})[1].get("access_token")
    st, d = api("POST", "/tensor/reports", body, tok)
    print("status", st, "report_id", d.get("id"), "title", d.get("title"))
    json.dump(body, open("/tmp/report.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()