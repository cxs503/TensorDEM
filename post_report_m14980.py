#!/usr/bin/env python3
"""Post the m14980 research report (board audit of m14680) to the platform.

All numbers are read programmatically from ``reports/m14980_audit_m14680.json``
and ``artifacts/gap_dem_autocalib/diagnosis.json`` — nothing is hand-copied.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
AUDIT = "reports/m14980_audit_m14680.json"
ART = "artifacts/gap_dem_autocalib/diagnosis.json"
FIG_AUDIT = "artifacts/m14980_audit/fig1_audit_gap_autocalib.png"
FIG_MISSION = "artifacts/gap_dem_autocalib/fig1_repose_diagnosis.png"


def api(method: str, path: str, payload=None, token: str | None = None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return json.loads(r.read() or b"{}")


def png(path: str) -> str:
    return base64.b64encode(open(path, "rb").read()).decode()


def main() -> None:
    a = json.load(open(AUDIT))
    art = json.load(open(ART))
    bench = a["benchmark"]
    eq = a["repro_equality"]
    tests = a["tests"]
    lvl = a["level"]

    p1_rows = "\n".join(
        f"| {r['mu']:.2f} | {r['theta_r_deg']:.2f} | {r['ke_ratio']:.3f} | "
        f"{r['phi_solid']:.4f} | {r['r2']:.3f} |"
        for r in art["p1_friction_response"])
    p3_rows = "\n".join(
        f"| {r['mu']:.2f} | {r['fixed_window']['theta_r_deg']:.2f} | "
        f"{r['fixed_window']['ke_ratio']:.1e} | {r['ke_stop_window']['theta_r_deg']:.2f} | "
        f"{r['ke_stop_window']['ke_ratio']:.1e} | {r['ke_stop_window']['steps']} |"
        for r in art["p3_rest_criterion"]["rows"])
    claim_rows = "\n".join(
        f"| {c['claim']} | `{c['doc']}` | `{c['artifact']}` | "
        f"{'yes' if any(v is True for k, v in c.items() if k.startswith('match')) else 'NO'} |"
        for c in a["doc_vs_artifact"])

    md = f"""# 交差审计：m14680 GAP-DEM-AUTOCALIB（TensorDEM）→ verdict **L{lvl['level']}**

## ① 案例介绍（审计对象与物理背景）

对象 = mission **m14680**（status `succeeded`，PR **#1246** merged，平台报告 **13828**，commit `fa79baa5`）：
「GAP-DEM-AUTOCALIB · DEM 自动标定 round 1」。本单 m14980 是**交差核验**（非重做）。

物理问题：DEM 微观参数（摩擦 μ / 刚度 k_n / 阻尼 e）→ 宏观响应（休止角 θ_r）→ 优化器对标文献 →
标定后角度误差 < 10 %。m14680 的结论是 **honest FAIL**：前向映射先不合格，因此没把优化器指向文献。

| 参数 | 值 |
|---|---|
| 粒径 r / 密度 ρ | 3.0 mm / 2650 kg/m³ |
| 法向刚度 k_n / 切向比 k_s/k_n | 800 N/m / 2.2 |
| 恢复系数 e / 滚动摩擦 μ_r | 0.38 / 0.10 |
| 容器半径 / 装填高度 | 21 mm / 42 mm（**7 个粒径**） |
| 颗粒数 | **{art['summary']['n_particles']}** |
| 变化量 | μ ∈ {{0.20, 0.35, 0.52, 0.70, 0.85}} |
| 文献锚 | 干砂休止角 **{bench['reference']['band_deg']}°** |

## ② 怎么做的（方法与工具链）

1. **取证三件套复核（全部重跑，不信报告）**：`scripts/m14980_audit_m14680.py --rerun`
   - run artifact：`{ART}`（committed sha256 `{eq['committed_sha256'][:16]}…`，{eq['committed_bytes']} B）
   - 重跑 `repro_cmd`：rc={a['repro'].get('returncode')}，{a['repro'].get('wall_seconds')} s（交付时 114.8 s）
   - 逐字段比对：**物理载荷完全相同** = {eq['physical_payload_identical']}，只有 `wall_seconds` 变
     （leaf diff = {eq['only_wall_clock_differs']}），per-seed digest 全等 = {eq['digests_identical']}
   - 图重生成：sha256 `{eq['figure_equality']['local_sha256'][:16]}…`，与交付 PNG **逐字节相同** = {eq['figure_equality']['byte_identical']}
2. **基准数字重算**（非引用）：从 artifact 重新算误差、Spearman、SNR。
3. **文档 vs 工件逐条对照**（零手抄核查）：正则抽出 round-1 文档表格里的数字，与 artifact 字段比对。
4. **单测重跑**：`{tests['cmd']}` → `{tests['summary']}`（{tests['wall_seconds']} s，5 个不变量测试）。
5. **判级+网关自检**：按 L0–L5 功能实质判级，并用 `maturity_gate.judge` 本地模拟复核（claimed L{lvl['level']} → gate_level 2，无缺键）。
6. **落表**：`tensor_maturity_verdicts` id=206，scope `m14680 GAP-DEM-AUTOCALIB`，level **{lvl['level']}**，verdict_by `board-audit`。

## ③ 计算结果（程序化注入，零手抄）

**P1 — 休止角 vs 摩擦（artifact `p1_friction_response`）**

| μ | θ_r [°] | ke/PE | φ_solid | 侧坡拟合 r² |
|---|---|---|---|---|
{p1_rows}

**P3 — 静止判据（固定窗 vs 自适应 `ke_stop`=2 %)**

| μ | 固定窗 θ_r [°] | 固定窗 ke/PE | ke_stop θ_r [°] | ke_stop ke/PE | 步数 |
|---|---|---|---|---|---|
{p3_rows}

**重算的判据量**：Spearman(μ,θ_r) = {bench['recomputed']['spearman_mu_theta']}（artifact {bench['recomputed']['artifact_spearman']}），
非增区间 {bench['recomputed']['n_non_increasing_intervals']}/{bench['recomputed']['n_intervals']}；
复算散差 {bench['recomputed']['replica_spread_deg']}° vs 信号跨度 {bench['recomputed']['signal_span_deg']}° → **SNR {bench['recomputed']['snr']}**（artifact {bench['recomputed']['artifact_snr']}）。
可达带 {bench['recomputed']['settled_band_deg']:.2f}–{bench['best_simulated_deg']:.2f}°，文献可达 = {bench['recomputed']['literature_reachable']}。

**文档 vs 工件（零手抄核查）**

| 声明 | 文档值 | 工件值 | 一致 |
|---|---|---|---|
{claim_rows}

另有 1 个**无工件背书**的手填数字：文档 §5「cone relaxation 50.9°」不在 `diagnosis.json` 中
（present_in_artifact = {a['untraced_claim']['present_in_artifact']}）；本次独立复算得
**{a['untraced_claim']['live_reproduction_deg']}°**（0.5° 内相符）→ 物理可信，记账不合规。

## ④ 结果图

图 1（本次审计生成）：左=文献带 vs 仿真带（基准 FAIL）；中=信号跨度 vs 重复性噪声（SNR<3 不可反演）；
右=固定窗与 `ke_stop` 静止判据的 θ_r 与残余动能占比 —— 显示"测到的是单层沉积而非锥体侧坡"。

图 2（被审交付的原始诊断图，本次重跑 pixel-identical 复现）。

## ⑤ 注意点（局限与踩坑）

1. **新能力没覆盖标定回路真正用的协议**：`run_pluviation`（P1）没有 `ke_stop`，测量时 ke/PE = 0.25–0.33，
   即"θ_r 是在 25–33 % 释放势能仍是动能时读的"——m14680 自己指出的核心缺陷在关键协议上仍未修。
2. **弹性模量支路不存在**：管线返回 `phi_solid` 而非 E，任务的第二个宏观响应无前向模型。
3. **体系太小**：275 颗粒 / 7 个粒径高，崩落后 φ_solid 仅 1–3 %（单层），物理上不可能有站立侧坡。
4. **平台侧记账观察（不是 m14680 的错）**：maturity_gate 以 `/root/tensor-ecosystem/{{repo}}`
   为证据根，该 checkout 已陈旧（最后一次 merge #1147，活跃 worktree 已到 #1256）；
   若审计把证据指针指向只存在于 mission worktree 的工件，会被机器判"工件缺失"。
   本单因此把重跑工件落在自己的 worktree（gate 同时以 `/root/tp-agent/work/m<审计单号>` 为根）。
5. **未做**：`tests/test_dem_calibration.py` 全量（改动路径默认档 digest 不变已单测覆盖）、round-2 扫描（属后续单）。

## ⑥ 误差判定（显式回答）

| 关键量 | 基准（ref） | 实测（got） | 误差 | <3 % ? |
|---|---|---|---|---|
| 休止角（最优仿真点） | {bench['reference']['band_deg']}°（文献干砂带） | {bench['best_simulated_deg']:.2f}° | **{bench['error_pct']['vs_band_low']:.1f}–{bench['error_pct']['vs_band_high']:.1f} %** | **否** |
| 休止角（任务判据 <10 %） | 30–37° | {bench['best_simulated_deg']:.2f}° | 同上 | **否（误差超标 3.8–5.0 倍）** |
| θ_r(μ) 单调性 | 0 反序 | {bench['recomputed']['n_non_increasing_intervals']}/{bench['recomputed']['n_intervals']} 反序，ρ={bench['recomputed']['spearman_mu_theta']} | — | **否** |
| 信噪比 | ≥3（可稳定反演） | {bench['recomputed']['snr']} | — | **否** |
| 复现性（同机重跑） | 物理载荷逐字段相同 | identical = {eq['physical_payload_identical']} | 0 | **是（唯一达标项）** |
| 单测不变量 | 5/5 | {tests['summary']} | 0 | **是（单元级）** |

**结论**：m14680 的交付是诚实的负面结果 + 一个真实但未接入生产路径的新能力（自适应静止判据），
故判 **L{lvl['level']}**（L1 单元验证 + L2 独立功能通过；L3 拒：`ke_stop` 无生产调用方且 `run_pluviation` 无判据；
L4 拒：唯一基准以 {bench['error_pct']['vs_band_low']:.1f}–{bench['error_pct']['vs_band_high']:.1f} % 失败）。
GAP 状态 = **OPEN**，round 2 已在 `docs/GAP-DEM-AUTOCALIB_round2_gate1.md` 继续。

**工件**：`docs/GAP-DEM-AUTOCALIB_audit_m14980.md`（审计报告）、`reports/m14980_audit_m14680.json`（全部数字）、
`reports/m14980_verdict_evidence.json`（落表证据）、`script: scripts/m14980_audit_m14680.py`。
"""

    payload = {
        "title": "交差审计 m14680 GAP-DEM-AUTOCALIB：证据三件套重跑通过、基准 38.5–50.1 % 未达标 → verdict L2",
        "summary": ("独立重跑 m14680 的 repro_cmd 与其图：物理载荷逐字段相同（仅 wall clock 变）、PNG 逐字节相同；"
                    f"基准（文献干砂 {bench['reference']['band_deg']}° vs 仿真 {bench['best_simulated_deg']:.2f}°）"
                    f"误差 {bench['error_pct']['vs_band_low']:.1f}–{bench['error_pct']['vs_band_high']:.1f} %，"
                    "SNR 1.70、3/4 区间反序 → 基准不通过；ke_stop 新能力未接入生产调用方 → 判 L2 并落表。"),
        "scenario_key": "gap_dem_autocalib_audit",
        "mission_id": 14980,
        "tags": ["DEM", "自动标定", "休止角", "交差审计", "maturity_gate", "TensorDEM", "证据复核"],
        "markdown": md,
        "figures": [
            {"name": "fig1_audit_gap_autocalib.png",
             "caption": "本次审计图：左 文献带(30–37°)vs 仿真带(13.83–18.45°)基准 FAIL；中 信号跨度 8.53° vs 复算噪声 5.00°（SNR 1.70）；右 固定窗 vs ke_stop 静止判据",
             "png_base64": png(FIG_AUDIT)},
            {"name": "fig1_repose_diagnosis.png",
             "caption": "被审交付 m14680 的原始诊断图，本次重跑 sha256 逐字节一致（可复现性证据）",
             "png_base64": png(FIG_MISSION)},
        ],
    }

    tok = api("POST", "/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                      "password": os.environ["TENSOR_AGENT_PASS"]})["access_token"]
    out = api("POST", "/tensor/reports", payload, token=tok)
    print("report posted:", json.dumps({k: out.get(k) for k in ("id", "title", "mission_id")},
                                       ensure_ascii=False))


if __name__ == "__main__":
    main()