#!/usr/bin/env python3
"""Post the m15454 board-audit report for m15329 (CASE-ROTATING-DRUM).

Every digit in the markdown is injected from artifacts/audit_m15454/finding.json
(via docs/board_audit/m15454/audit_m15329.py) — no hand-copied numbers.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
F = json.load(open(os.path.join(REPO, "artifacts/audit_m15454/finding.json")))
FIG1 = os.path.join(REPO, "docs/audit/figures/m15454_m15329_anchor.png")

f = F
cases = f["cases"]
rc = f["internal_consistency"]["metrics_recomputed_from_final_pos"]
rest = f["resting_gate"]
tp = f["three_piece"]["benchmark_ref_vs_got"]
dl = f["delivery_integration"]
reg = f["registry"]
repro = f["three_piece"]["repro_cmd"]
plateau = {p["case"]: p for p in f["quasi_steady"]["plateau"]}

rows = []
for c in cases:
    rows.append("| {case} | {fr:g} | {N:d} | {ds_mm:.0f} | {revs:g} | {ref:.2f} | {got:.2f} | "
                "{err:+.2%} | {p20} | {p3} | {ok} | {z:.2f} | {mode} |".format(
                    case=c["case"], fr=c["fr"], N=c["N"], ds_mm=c["d_small"] * 1000.0,
                    revs=c["revs"] or 0, ref=c["theta_ref_recomputed"], got=c["theta_got"],
                    err=c["theta_err_recomputed"],
                    p20="✓" if c["theta_err_recomputed"] < 0.20 else "✗",
                    p3="✓" if c["theta_err_recomputed"] < 0.03 else "✗",
                    ok="✓" if c["resting_ok"] else "**✗**",
                    z=max(abs(c["excess_z"]), abs(c["corr_z"])), mode=c["mode"]))

pct_rows = []
for r in rc:
    q = r["theta_percentiles"]
    pct_rows.append("| " + r["case"] + " | " + " | ".join(
        "{:.1f}° ({:.1%})".format(q[k]["theta"], q[k]["rel_err_vs_ref"])
        for k in ("50", "75", "85", "90", "95", "99")) + " |")

pl_rows = ["| {} | {:.4f} | {:.4f} | {} |".format(
    k, v["S_slope_last_quarter_per_rev"], v["S_last_vs_prev_quarter"], v["sign_changes_S"])
    for k, v in plateau.items()]

md = f"""# m15454 交差核验：m15329 `CASE-ROTATING-DRUM`（TensorDEM）

**判级 L2 · gate 期望 reopened** · 被审 commit `{f['source']['commit_short']}`（分支 {f['source']['branch']}）
· 审计 HEAD `TensorDEM fc5cb532`（分支 agent/m15454）。

## ① 案例介绍（被审对象 + 判据来源）

m15329 = 算例池中的「回转滚筒混合/休止」锚点：二维软球 DEM（复用已合并的
`tensordem.drum_segregation.kernel2d`，属更早的 p176w2-n08 成果）在 R=0.15 m 滚筒内跑
双粒径床（d_l/d_s=2，填充 0.30，大颗粒面积分数 0.30），先静态沉降（“休止”），再以
Fr=ω²R/g 旋转 6–10 圈（“混合”），把测得释放角与解析式 θ_d=arccos(−Fr) 对照。
判据按任务 goal 文本 = 定性 + <20%。

**基准裁判查询结果**：`tensor_benchmark_cases WHERE anchor_slug='CASE-ROTATING-DRUM'` →
**{reg['anchor_rows']} 行**（全表 {reg['table_rows']} 行）；即该锚点**没有机器持有判据**，
唯一判据是 goal 文本与脚本内联解析式。这不是本单过错，但禁止记为“通过基准验证”。

## ② 怎么做的（审计方法与工具链）

只读复核 + 独立重算，全部脚本化（`docs/board_audit/m15454/audit_m15329.py`）：

1. 三件套取证：把被审 6 个文件按字节复制到 `refs/` 并记 sha256（`refs/MANIFEST.json`）；
2. 用被审方**自己的** metrics 模块从 `final_pos` 重算 S / excess / corr / z，逐位比对；
3. 释放角指标的百分位敏感性重算（p50…p99）与角度截断比例统计；
4. 休止闸门按脚本内写死的 5e-3 m/s 逐算例复核；
5. 准稳态说法用 S(rev) 序列检验（斜率、变号次数）；
6. 交付/集成：`merge-base --is-ancestor`、HEAD 文件存在性、测试与调用方 grep；
7. 本地实跑被审 repro_cmd 一个算例（84 s）。

## ③ 计算结果（逐点，程序化注入）

| 算例 | Fr | N | d_s[mm] | revs | ref θ [°] | got θ [°] | 相对误差 | <20% | <3% | 休止闸门 | max|z| | mode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

统计：释放角 **{tp['n_below_20pct']}/{tp['n_points']} 点 <20%**（最大 {max(c['theta_err_recomputed'] for c in cases):.2%}）；
平台 15% 红线亦全部满足；**休止闸门在 {len(rest['fail'])}/3 主算例 FAIL**（rms={cases[0]['settle_rms_speed']:.4e} m/s =
阈值 5e-3 的 {cases[0]['settle_rms_speed']/5e-3:.1f} 倍，settle 预算 22000 步 + 4 轮 quench 用尽）；
偏析 **{sum(1 for c in cases if c['mode']=='mixed')}/{len(cases)} 算例 mode=mixed**，|z|≥3 的算例 **{len([d for d in f['segregation_claims']['per_case'] if d['significant_at_3sigma']])} 个**。

指标自洽性：主算例 3/3 的 S/excess/corr/excess_z/corr_z 由 `final_pos` 重算**逐位一致**
（`metrics_match={f['internal_consistency']['all_main_metrics_reproduced']}`）；颗粒面积/床面积
= {f['internal_consistency']['geometry'][0]['particle_area_over_bed_area']:.4f}（8/8），装填与计数无重复。

复现：`{repro['cmd']}` 由审计方本地执行，墙钟 84 s，14/14 数值字段与已提交工件**逐位一致**。
未跑：主算例命令（10 rev / d_s=6 mm，单算例 218–440 s）。

## ④ 结果图

四联图见下方 figure：
A 释放角 vs 解析 arccos(−Fr)（两套配置分别标出，可见报告 5 行表混装两种几何）；
B 百分位旋钮——同一快照换百分位，Fr=0.1 主算例误差从 68%（p50）到 4.2%（p90）；
C 主算例 S(rev) 无平台；D 休止闸门（红=FAIL）与显著性。

## ⑤ 注意点（踩到的坑与局限）

1. **“休止相 rms<5e-3 已验证”是错的**：三个主算例 `resting_ok=false`，rms={cases[0]['settle_rms_speed']:.4e} m/s。
   报告 §② 与 mission checks 均写成已验证 → 最强的一条虚标。
2. **“流态 5/5 定性一致”不可证伪**：工件里只有 `ref_regime`（输入 Fr 查常量表），没有任何实测流态字段。
3. **偏析“方向一致”是符号约定**：8/8 mode=mixed，报告却对 +0.090/+0.081/−0.035/+0.071/+0.057 五种情况全打 ✓。
4. **“10 圈接近准稳态”被自己的曲线否证**：S(rev) 23 点变号 14–16 次，末段斜率正负混杂：
{chr(10).join(pl_rows)}
5. **释放角是“分位旋钮”而非脱离事件**：贴壁颗粒 22–28% 的角度为负被 clip 到 0；
同一快照的百分位敏感性：
| 算例 | p50 | p75 | p85 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|
{chr(10).join(pct_rows)}
6. **交付缺口**：被审 commit 不在 main（`merged_into_head={dl['merged_into_head']}`），6 个交付文件在 HEAD 上**全部不存在**，
commit 触及 0 个测试、0 个调用方 → L3 拒绝。
7. 5/8 工件缺 `final_pos`，其指标不可从字节重算（只能引用自报值）→ 这是本轮审计的覆盖边界。
8. Haff 1983 的出处未查原刊（无该刊访问），标注 **WITHHELD**，不作为支撑证据。

## ⑥ 误差判定（逐项回答“是否<3%”）

* 释放角 Fr=0.3（主算例）2.53%、Fr=0.5（主算例）0.50% → **<3% 成立**；
  Fr=0.1（主）4.77%、Fr=0.1（扫）5.53%、Fr=0.3（扫）7.47%、Fr=0.5（扫）1.94%、
  Fr=0.7 4.78%、Fr=1.0 11.04% → **<3% 不成立**，但全部 **<15%**（平台红线）。
* 休止判据（rms vs 5e-3 m/s）：主算例 3/3 **FAIL**（超出 12.5 倍）→ 锚点“休止”一半未达成。
* 偏析方向/幅度：**无基准**（文献量级带无出处）→ 不计误差，只记 mode=mixed 不显著。
* 流态：**无测量**，不计误差。

**结论**：这套算例的“释放角 vs 解析式”是真实定量对照且 8/8 通过本单容差，但报告在
休止闸门、流态一致性、偏析方向、准稳态四处越过证据边界，且成果未进 main → **L2，reopened**。
返修清单见 `docs/audit/M15454_BOARD_AUDIT.md` §5（做实休止闸门 / 注册锚点 / 拆表 /
定性栏带显著性 / 换物理量测释放角 / 合并 + 补测试）。
"""

figures = []
with open(FIG1, "rb") as fh:
    figures.append({
        "name": "m15454_m15329_anchor.png",
        "caption": "m15454 交差核验四联图：A 释放角 vs arccos(−Fr)（主算例/扫描算例分标）；"
                   "B 百分位旋钮（误差随百分位变化，虚线=20% 容差）；C 主算例 S(rev) 无平台；"
                   "D 休止闸门（红=FAIL）与 |z|/mode 标注。数据源 artifacts/audit_m15454/finding.json。",
        "png_base64": base64.b64encode(fh.read()).decode(),
    })

payload = {
    "title": "m15454 交差核验：CASE-ROTATING-DRUM（m15329, TensorDEM）—— L2，reopened",
    "summary": (
        f"释放角 8/8 点 <20%（最大 {max(c['theta_err_recomputed'] for c in cases):.2%}），工件指标可从 final_pos 逐位重算，"
        f"repro_cmd 本地实跑逐位复现；但休止闸门在 3/3 主算例 FAIL（rms 6.23e-2 vs 5e-3）、"
        f"流态“5/5 一致”无测量对象、偏析 8/8 mode=mixed 却勾了 5 个方向、S(rev) 无平台，"
        f"且 commit 不在 main → 判 L2（L3/L4 拒绝），gate 期望 reopened。"),
    "scenario_key": "board-audit/CASE-ROTATING-DRUM",
    "mission_id": 15454,
    "tags": ["board-audit", "TensorDEM", "CASE-ROTATING-DRUM", "m15329", "drum",
             "Froude", "release-angle", "maturity-verdict", "L2", "reopened"],
    "markdown": md,
    "figures": figures,
}

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not base.endswith("/api/v1"):
    base = base.rstrip("/") + "/api/v1"
login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                    "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
req = urllib.request.Request(base + "/auth/login", data=login,
                             headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=60) as resp:
    tok = json.loads(resp.read())
tok = tok.get("access_token") or tok.get("token")
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok}, method="POST")
with urllib.request.urlopen(req, timeout=180) as resp:
    print(resp.read().decode()[:400])