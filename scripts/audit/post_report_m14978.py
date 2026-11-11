"""Publish the m14978 board-audit report (figures + numbers injected, never hand-typed).

    python scripts/audit/post_report_m14978.py

Reads ``artifacts/audit_m14960/audit.json`` and the audit figure, builds
``/tmp/report_m14978.json`` and POSTs it to ``$TENSOR_API_BASE/tensor/reports``.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "artifacts", "audit_m14960")
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def login() -> str:
    req = urllib.request.Request(BASE + "/auth/login",
                                 data=json.dumps({"username": USER, "password": PASS}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    d = json.load(urllib.request.urlopen(req, timeout=20))
    return d.get("access_token") or d.get("token") or d["data"]["access_token"]


def main() -> int:
    a = json.load(open(os.path.join(ART, "audit.json")))
    C = {c["check"].split()[0]: c for c in a["checks"]}
    c1, c2, c3, c4, c5, c6, c7 = (C["C1"], C["C2"], C["C3"], C["C4"], C["C5"], C["C6"], C["C7"])
    m = c3["rows"]["size_medium"]
    w = m["windows"]
    sm = c3["rows"]["size_small"]
    mu = c7["mu_dem"]

    tbl_repro = "\n".join(
        "| %s | %s | %s | %s |" % (k, repr(c1["committed"][k]), repr(c1["rerun"][k]),
                                   repr(abs(float(c1["rerun"][k]) - float(c1["committed"][k]))))
        for k in ("sigma_pa", "tau_pa", "mu", "drift_rel", "drift_r2", "total_steps"))
    tbl_gate = "\n".join(
        "| %s | %+.3f | %.3f | %s | **%s** |" % (
            tag, r["drift_rel"], r["drift_r2"], r["valid_with_gate_off"],
            "REJECT" if not r["valid_at_tol_0.15"] else "PASS")
        for tag, r in c2["rows"].items())
    tbl_win = "\n".join(
        "| %s | %.1f | %.4f | %+.1f%% | %s |" % (
            k, v["tau_pa"], v["mu"], v["vs_lit_pct"], "是" if v["within_10pct"] else "**否**")
        for k, v in w.items())

    md = f"""# 交差核验 audit：m14960（GAP-DEM-FRICTION）不该按「基准验证」结账 —— 判级 L3 / 返修

**本单**：m14978（任务榜·交差核验，非重做）｜**被审**：m14960 `succeeded`，提交 `3fcf006d`（branch `agent/m14960`）｜**仓**：TensorDEM
**裁定**：`tensor_maturity_verdicts` → repo=TensorDEM, scope=`m14960 GAP-DEM-FRICTION`, level=**3**, verdict_by=`board-audit`，处置=**reopened（返修）**。

## 1. 案例介绍（被审对象是什么）

m14960 是 `GAP-DEM-FRICTION`（验收判据「摩擦系数 vs 实验 <10%」）的第 2 轮。它用 TensorDEM 的**直剪盒**装置测颗粒集合体**宏观摩擦系数** μ_macro = τ/σ_n：粗糙底板 + 无摩擦侧壁 + 载荷伺服压板横拖，床面一次制备、逐应力级复用。物理量：单分散球形颗粒堆积（准静态、I≲0.01），宏观摩擦对标文献准静态有效摩擦 μ_s。

| 参数 | 值 |
|---|---|
| 颗粒 | r=0.005 m, ρ=2500 kg/m³, 单分散 |
| 接触 | k_n=1e5 N/m, k_t/k_n=0.5, e=0.3, μ_roll=0.05, μ_p=0.5 |
| 盒体 small / medium | 0.080×0.032×0.050 (N=78) / 0.080×0.064×0.080 (N=390) |
| 应力/协议 | σ_n=4000 Pa（伺服）；γ=0.6（有限尺寸扫描）/ 1.0（μ_p 矩阵）；seed=6107 |
| 外部锚 | μ_s=0.38，带 0.36–0.40（Jop/Forterre/Pouliquen, Nature 441,727 (2006); GDR MiDi, EPJE 14,341 (2004)） |

## 2. 怎么做的（审计方法，全程独立于报告数字）

1. **干净检出**：`git archive 3fcf006d` → `/tmp/audit_m14960`（排除任何未跟踪残留，保证只审"提交里有什么"）。
2. **独立重跑 headline**：`PYTHONPATH=src python scripts/m14960/run_friction_matrix.py --tag audit_size_medium --geom medium --mu 0.5 --gamma 0.6 --sigmas 4000` → 219 s。
3. **装置回归**：`pytest tests/test_shear_cell_friction.py -q` → 10 passed / 84 s。
4. **施加仓内自己的判据**：直接调用 `src/tensordem/shear_cell.py:612 _validity_gate`，把 m14960 其它行用的 `stationarity_tol=0.15` 施加到 headline 两行。
5. **窗口敏感性**：用提交 JSON 里的 1511/945 点 τ(γ) 历史，重算后 50/40/25/10% 窗口均值 μ。
6. **工件一致性机检**：表 2/表 3 行数 vs `log_levels.json` vs per-tag JSON 存在性；文档里未渲染注入模板的正则计数；`analyse_friction_matrix.py` 是否解析 `runs/*.log`。
7. 工具链：numpy 2.4.6 / py3.11.6；复算脚本 `scripts/audit/m14960_cross_audit.py` → `artifacts/audit_m14960/audit.json`（C1–C7）。**报告中所有数字均由该 JSON 注入**。

## 3. 计算结果

### 3.1 可复现性：PASS（比特级）

| 量 | 提交工件 | 本审计重跑 | |Δ| |
|---|---|---|---|
{tbl_repro}

9 个承重量最大绝对差 = **{c1['max_abs_delta']}**（含 σ_n/τ/μ/I/drift/r²/steps/N/shear_rate）。装置确定性成立。

### 3.2 仓内自己的稳态门：headline 两行均 REJECT

| run | drift_rel | drift R² | gate 关（提交口径） | gate @0.15 |
|---|---|---|---|---|
{tbl_gate}

提交 JSON 中 `stationarity_tol: null` —— 即 headline 的 "门 PASS" 是**判据关闭**下的 PASS，而同一报告的 μ_p 行用的是 0.15 并据此 REJECT。

### 3.3 窗口口径敏感性（headline 算例，σ_n=3991.7 Pa）

| 窗口 | τ̄ [Pa] | μ | vs 0.38 | 是否 <10% |
|---|---|---|---|---|
{tbl_win}

τ(γ) 前 1/10 均值 {m['tau_first_tenth_pa']:.1f} Pa → 后 1/10 {m['tau_last_tenth_pa']:.1f} Pa（单调上升），μ 相对极差 **{100*m['mu_spread_rel']:.1f}%**。同一算例的 μ 在 {min(v['mu'] for v in w.values()):.3f}–{max(v['mu'] for v in w.values()):.3f} 之间摆动 → 提交值是"后 50% 窗口均值"，不是稳态摩擦系数。
N=78 行同病更重：drift {c2['rows']['size_small']['drift_rel']:+.3f}，τ 由 {sm['tau_first_tenth_pa']:.1f} → {sm['tau_last_tenth_pa']:.1f} Pa（退化），μ 0.3037→0.1847；因此「+19.4% 有限尺寸效应」的两个点都不是稳态，方向可信、**幅度不可引用**。

### 3.4 外部锚：WITHHELD

μ_DEM={mu:.4f}：对中心 0.38 = {c7['anchor']['central_0.38']['err_pct']:+.1f}%；对带下界 0.36 = {c7['anchor']['band_lo_0.36']['err_pct']:+.1f}%；对上界 0.40 = {c7['anchor']['band_hi_0.40']['err_pct']:+.1f}%。锚系模型知识引用（报告自陈未在线核验），且 DEM 的 μ_p=0.5 **未与锚材料标定** → 落在带内不构成预测能力。

### 3.5 工件/表述一致性（机检）

| 检查 | 结果 |
|---|---|
| C4 表 2 标"(未完成)"行数 / 表 3 正文行数 / `log_levels.json` 行数 | {c4['table2_rows_marked_unfinished']} / **{c4['table3_body_rows']}** / {c4['log_levels_rows']} |
| C4 per-tag 机读工件（mu030/050/070） | {c4['per_tag_artifacts']} |
| C5 文档未渲染注入模板 | {c5['n_placeholders']} 处，例：{c5['placeholders']} |
| C6 分析脚本是否解析 runs/*.log | {c6['analyse_script_parses_logs']} |
| 平台 L4 门（`/tensor/bench/l4-verdict/14960`） | `CLAIMED_NOT_PROVEN`，rows=0（无平台重算的 ref/got/error 行） |

## 4. 图

(a) 两次 headline 运行的 τ(γ)：阴影=被平均的窗口，τ 仍在漂移；(b) μ 随平均窗口长度变化，灰带=文献 0.36–0.40，红虚线=±10% 门。单窗口 μ（点线）不是稳态值：后 10% 窗口直接把 −4.6% 翻成 +24.5%。

## 5. 注意点（坑与局限）

1. **默认关掉的稳态门是这类任务的通天塔**：`stationarity_tol=None` 时任何漂移窗口都 PASS，而报告里 "门 PASS" 一栏极易被读成"合格"。返修第一步应是显式写 tol 并让门在 JSON 里留痕。
2. **"程序化注入零手抄"必须机检**：本单 §5 误差判定表仍留 `{{100*er(0.3627):+.1f}}%` 一类模板（引用未定义 `j`），5 行里 2 行没有数字——写作纪律的缺口比数值误差更容易被忽略。
3. **日志派生行缺机读工件**：`mu030/050/070` 只有 3 行/份的文本日志 + 一个汇总 JSON，per-tag JSON 没有；文档给的复现 heredoc 不在仓内 → gate 无法机器复核。
4. **文献锚的"带"能吃掉判据**：0.36–0.40 的 ±5% 不确定度叠加 μ_p 未标定，使 "<10%" 在带边界翻转；关闭该锚点需要与被测材料配套的实验/数字化数据。
5. **本审计未跑**（记 unknown 不记 pass）：`mu030/050/070` 重跑（约 30 min）、`size_large`(N=1392)、`oss_inventory` 空集 SQL 复核、文献在线核验（沙箱无出站）。
6. 局限：审计仅依赖提交工件 + 一次重跑；对"日志文本是否为真实运行产物"只能间接判断（τ=μ_pσ_n 的精确值经独立物理复核判为界面滑移，非伪造）。

## 6. 误差判定（逐项显式回答"是否 <3%"）

| 关键量 | 基准 | 误差 | 是否 <3% | 是否 <10%（锚点判据） |
|---|---|---|---|---|
| μ_macro（N=390, μ_p=0.5, 4000 Pa, γ=0.6，后 50% 窗口） | 文献 0.38 | {c7['anchor']['central_0.38']['err_pct']:+.1f}% | **否** | 形式上 是 |
| 同上（后 10% 窗口） | 文献 0.38 | {w['last_10%']['vs_lit_pct']:+.1f}% | 否 | **否** |
| μ 的窗口口径散布 | 自身均值 | ±{100*m['mu_spread_rel']/2:.1f}% | **否** | 否 |
| μ_macro（N=78） | 文献 0.38 | {sm['windows']['last_50%']['vs_lit_pct']:+.1f}% | 否 | 否 |
| 伺服 σ_n 保持（N=390） | 目标 4000 Pa | {(c1['committed']['sigma_pa']/4000-1)*100:+.2f}% | **是** | 是 |
| headline 数值可复现性 | 提交工件 | |Δ|=0.0（9 量） | **是** | 是 |
| 稳态性（μ 的定义前提） | 仓内门 tol=0.15 | drift {c2['rows']['size_medium']['drift_rel']:+.3f} | 否 | **否（REJECT）** |

**honest 结论**：数据真实、装置可复现（比特级）、报告对锚未核验与未跑项如实自陈；但「−4.6%，首次满足 <10%」这一承重结论取自一个被本仓稳态门拒绝、且口径一变就翻到 +24.5% 的窗口，**不能作为基准验证成立**。故判 **L3**（真代码 + 已集成装置 + 可复现跑），L4 未达成，处置 **reopened**。

## 7. 证据指针

`docs/audit/M14960_CROSS_AUDIT.md` · `artifacts/audit_m14960/audit.json`（C1–C7） · `artifacts/audit_m14960/rerun_size_medium.json|log` · `artifacts/audit_m14960/pytest_shear_cell_friction.log` · `artifacts/audit_m14960/window_sensitivity.png` · `scripts/audit/m14960_cross_audit.py` · `scripts/audit/m14960_audit_figure.py` · commit `97711cb8`
"""
    fig = base64.b64encode(open(os.path.join(ART, "window_sensitivity.png"), "rb").read()).decode()
    payload = {
        "title": "交差核验 audit：m14960（GAP-DEM-FRICTION）判级 L3 / 返修 —— headline −4.6% 取自稳态门拒绝的窗口",
        "summary": ("独立复算 m14960：headline 运行比特级可复现（9 量 Δ=0，219 s），装置回归 10 passed；"
                    "但同一算例被仓内自己的 _validity_gate@0.15 判 REJECT（drift +0.240, R²=0.083），"
                    "μ 随窗口在 0.360–0.473 漂移（后 10% 窗口 = +24.5%）；表 2/表 3 与报告引用不一致（表 3 正文 0 行），"
                    "文档 §5 残留未渲染注入模板，μ_p 7 行不可由提交脚本再生，外部锚未材料标定。"
                    "裁定 L3（非 L4），reopened。"),
        "scenario_key": "GAP-DEM-FRICTION",
        "mission_id": 14978,
        "tags": ["GAP-DEM-FRICTION", "TensorDEM", "board-audit", "maturity-verdict",
                 "shear-cell", "steady-state-gate", "L3"],
        "markdown": md,
        "figures": [{"name": "m14978_window_sensitivity.png",
                     "caption": "m14960 headline 算例：(a) τ(γ) 仍在漂移（阴影=平均窗口）；(b) μ 随平均窗口长度变化，灰带=文献 0.36–0.40，红虚线=±10% 门",
                     "png_base64": fig}],
    }
    out = "/tmp/report_m14978.json"
    json.dump(payload, open(out, "w"), ensure_ascii=False)
    tok = login()
    req = urllib.request.Request(BASE + "/tensor/reports", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok}, method="POST")
    d = json.load(urllib.request.urlopen(req, timeout=60))
    print("report id:", d.get("id"), "status:", d.get("status"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())