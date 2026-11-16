#!/usr/bin/env python3
"""Publish the m22628 status/methods report (honest, no fabricated DEM numbers).

The cluster scan (cj4266) was still queued when this mission's budget ran out,
so **no DEM discharge rate exists for this scan yet**.  This script therefore
publishes what was actually established:

* the pre-registration (S1–S5, P1–P5) frozen before any data;
* the estimator, validated against a **synthetic ground truth** (k = 1.2,
  C = 0.56, p = 5/2), including the failure modes it must detect;
* the case matrix proven inside the law's stated domain;
* an explicit ``not_run`` for every DEM number.

The figures are generated from the synthetic ground truth and are captioned as
such — they are *estimator* evidence, not validation evidence.
"""
from __future__ import annotations

import base64
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "benchmarks" / "m22628_beverloo_scan"))

import analyse as A          # noqa: E402
import render_figures as R   # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"
PAYLOAD = Path("/tmp/report_m22628.json")
MISSION = 22628
JOB = "cj4266"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
if not BASE.endswith("/api/v1"):
    BASE += "/api/v1"

D = 0.003
TRUTH = dict(C=0.56, k=1.2, p=2.5)     # the planted ground truth


def synthetic_runs(truth: dict) -> dict:
    """A structurally complete run set generated from a *known* law."""
    ratios = [4.0, 4.5, 5.0, 5.5, 6.0]
    ids = ["D40", "D45", "D50", "D55", "D60"]
    runs = {}
    for cid, r in zip(ids, ratios):
        D_o = r * D
        rho = 1400.0
        w = truth["C"] * rho * math.sqrt(9.81) * (D_o - truth["k"] * D) ** truth["p"]
        runs[cid] = {
            "case_id": cid,
            "case": {"orifice_diameter": D_o, "particle_diameter": D, "silo_radius": 0.030,
                     "fill_height": 3.0 * D_o, "kn": 5.0e3, "mu": 0.50,
                     "mu_rolling": 0.05, "wall_mu": 0.0, "seed": 1},
            "domain": {"ratios": {"D_over_d": r, "silo_D_over_D": 4.0,
                                  "bed_height_over_D": 3.0}, "in_domain": True},
            "result": {"rho_bulk": rho, "flow_rate_kg_s": w, "flow_rate_std_kg_s": 0.01 * w,
                       "n_bed": 4882, "flow_rate_windows_kg_s": [w] * 5,
                       "mass_fraction_discharged": 0.10},
        }
    return runs


def make_validation_figures(s: dict) -> list[dict]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt = R._plt()
    R.FIG_DIR = FIG_DIR
    files = [
        (R.fig_loglog(s, plt), "m22628_validation_loglog.png"),
        (R.fig_residuals(s, plt), "m22628_validation_residuals.png"),
    ]
    out = []
    for produced, target in files:
        final = FIG_DIR / target
        if produced != final:
            produced.replace(final)
        out.append(final)
    return out


def build_markdown(s: dict) -> str:
    st = s["statements"]
    fits = s["fits"]
    md: list[str] = []
    A_ = md.append
    A_("## ① 案例介绍")
    A_("")
    A_("**算例**：`CASE-DEM-SILO-BEVERLOO`（算例池·料仓卸料流率 Beverloo 律）。"
       "学科归属：船舶总体性能 · 绿色航行 / 散货装卸。")
    A_("")
    A_("**物理背景**：平底圆筒料仓底部中心圆孔，颗粒重力自由流出，稳态质量流率服从 "
       "Beverloo–Leniger–van de Velde (1961, *Chem. Eng. Sci.* 15, 260–269, Eq. (4))：")
    A_("")
    A_("```")
    A_("W = C · ρ_b · √g · (D − k·d)^(5/2)      C = 0.58 (0.55–0.60), k = 1.4 (1.2–1.6)")
    A_("```")
    A_("")
    A_("**本单要解决的问题（审计结论）**。此前六轮（m20055/m20707/m21033/m21274/m22124）"
       "都把 `k = 1.4` 当作已知去算参考值，只比**单个孔口的幅值**。这对律的三条可证伪命题是盲的："
       "①形状（指数是否 5/2）、②有限尺寸偏置 `k·d`（`k` 是否落在 1.2–1.6）、③与床高无关。"
       "把三者压成一个幅值，会让「`k` 错了」伪装成「`D` 相关的幅值误差」——"
       "这正是本战役观测到的指纹（`D/d = 4` 处 +20 %，`D/d ≥ 5` 处 −7…+6 %）。")
    A_("")
    A_("**扫描配置**（冻结于 `benchmarks/m22628_beverloo_scan/prereg.md`）：")
    A_("")
    A_("| 参数 | 值 |")
    A_("|---|---|")
    A_("| 颗粒直径 d | 3.0 mm |")
    A_("| 仓半径 R_silo | 30.0 mm |")
    A_("| 孔口 D | 4.0d / 4.5d / 5.0d / 5.5d / 6.0d |")
    A_("| 床高 H | 45.0 mm（D 扫描）/ 30.0、37.5 mm（H 独立性）|")
    A_("| μ_p / μ_r / e | 0.50 / 0.05 / 0.30 |")
    A_("| kn / kt / kt_kn | 5e3 N/m / 2.5e3 N/m / 0.5 |")
    A_("| 壁面摩擦 μ_wall | 0（光滑壁，Beverloo 玻璃珠配置）|")
    A_("| 协议 | 静置 0.06 s + 弃置 0.12 s + 5 窗 × 0.15 s |")
    A_("")
    A_("8 个算例**全部**落在律的适用域内（`D/d ≥ 4`、`silo_D/D ≥ 2.5`、`H/D ≥ 1.5`），"
       "由 `scan.py --list` 与一项 pytest 断言逐案核对。")
    A_("")
    A_("## ② 怎么做的")
    A_("")
    A_("1. **总账核对**：`CASE-DEM-SILO-BEVERLOO` status = `open`，可认领；未改写锚点。")
    A_("2. **预注册先于数据**：S1–S5 五条命题 + P1–P5 五条预测 + 「不事后加点/不重拟合充数」"
       "的 honest-FAIL 条款，写入 `prereg.md` 后才递交算例。")
    A_("3. **估计器先验（本单真正完成的部分）**：`analyse.py` 实现四层嵌套拟合——"
       "A 无自由参数（文献值）、B 只拟合 C、C 拟合 (C,k)（p=5/2）、D 拟合 (C,k,p)。"
       "全部在对数空间做最小二乘，`C` 解析消去，标准误由 profile 目标的数值 Hessian 给出。"
       "该估计器**在合成真值上先验证**，见 ③。")
    A_("4. **DEM 内核**：`tensordem.silo_flow`（Cundall–Strack 壁面切向史、无序装填、"
       "逐步能量账本）。不用 `silo_discharge`——其自身审计记录了 `silo_D/D` 偏小处"
       "系统性 +20 % 过预测。")
    A_("5. **递交**：8 分片并行 `run_compute`（`stage_dir` 直达节点，8 核，timeout 7200 s），"
       f"作业号 **{JOB}**。")
    A_("")
    A_("## ③ 计算结果")
    A_("")
    A_("### 3.1 DEM 卸料流率 —— **not_run**")
    A_("")
    A_(f"作业 **{JOB}** 在本单机时预算内**始终处于排队状态**（递交后 12 min 仍为 `queued`，"
       "队列内约 41 个作业在前，调度器未派发节点）。因此**本单没有任何一个 DEM 卸料流率**，"
       "S1–S5 五条命题**全部未判定**。表 1 如实留空，不填占位数字、不做「本机小试」替代"
       "（本机单算例 ~400 s×单核，超出本机 <10 min 单次限制，属重计算，必须走调度器）。")
    A_("")
    A_("| case | D/d | W_dem | W_lit | 相对误差 | 判定 |")
    A_("|---|---|---|---|---|---|")
    for cid, r in zip(["D40", "D45", "D50", "D55", "D60"], [4.0, 4.5, 5.0, 5.5, 6.0]):
        A_(f"| {cid} | {r:.2f} | not_run | — | — | **NOT_RUN** |")
    A_("| D50_H25 | 5.00 | not_run | — | — | **NOT_RUN** |")
    A_("| D50_H20 | 5.00 | not_run | — | — | **NOT_RUN** |")
    A_("| D50_s2 | 5.00 | not_run | — | — | **NOT_RUN** |")
    A_("")
    A_("### 3.2 估计器验证（**合成真值**，非验证证据）")
    A_("")
    A_(f"用已知真值 C = {TRUTH['C']}, k = {TRUTH['k']}, p = {TRUTH['p']} 生成 5 个孔口点，"
       "交给与 DEM 数据完全相同的拟合路径，检查它能否反演回来：")
    A_("")
    A_("| 模型 | 自由参数 | C | k | p | max abs rel err |")
    A_("|---|---|---|---|---|---|")
    names = {"B_free_C": "B（C 仅）", "C_free_Ck": "C（C, k）", "D_free_Ckp": "D（C, k, p）"}
    for tag, f in fits.items():
        A_(f"| {names[tag]} | — | {f['C']:.6f} | {f['k']:.4f} | {f['p']:.4f} | "
           f"{100 * f['max_abs_rel_err']:.2e} % |")
    A_("")
    A_(f"反演精度：`k` 误差 {abs(fits['C_free_Ck']['k'] - TRUTH['k']):.2e}`，"
       f"`C` 误差 {abs(fits['C_free_Ck']['C'] - TRUTH['C']):.2e}`，"
       f"`p` 误差 {abs(fits['D_free_Ckp']['p'] - TRUTH['p']):.2e}"
       "（机器精度量级，符合无噪声最小二乘的预期）。**这只是估计器自检，"
       "不构成对 Beverloo 律的任何验证**——真值是本文自己植入的。")
    A_("")
    A_("### 3.3 估计器必须能失败（这才是可比性所在）")
    A_("")
    A_("| 植入的「真相」 | 预期行为 | 实测 |")
    A_("|---|---|---|")
    A_("| p = 2.0（指数错） | 三参数拟合须报 p≈2.0 且 S1 判 FAIL | 通过（`test_fit_detects_a_wrong_exponent`）|")
    A_("| k = 0.0（偏置错），仍钉死 k=1.4 | 残差须呈 D 相关，单个 C 修不好 | 通过（`test_fixed_k_contaminates_the_amplitude`）|")
    A_("| 同上，放开 k | 残差须塌缩到 <1e-6，且 C 被高估 | 通过 |")
    A_("| 流率随 H 单调上升 20 % | 床高独立性须判 FAIL | 通过（`test_fill_height_test_detects_*`）|")
    A_("| 流率整体 +20 % | 总账 10 % 判据须判 FAIL 并如实上报 | 通过（`test_primary_verdict_reports_a_failure_honestly`）|")
    A_("")
    A_("第 2/3 行就是本战役缺陷的**机制复现**：`k` 钉在真值之上会让 `(D − k d)` 变小，"
       "补偿出的 `C` 被系统性抬高——所以「只比幅值」的旧做法既看不见 `k` 错、又会给出"
       "一个看似合理的 `C`。")
    A_("")
    A_("## ④ 结果图")
    A_("")
    A_("图 1、图 2 由 **3.2 的合成真值** 生成（真值 k=1.2, C=0.56, p=2.5），"
       "用于展示拟合路径与残差对比的行为，**不是** DEM 结果。DEM 场量/流率图待 "
       f"{JOB} 回传后由 `render_figures.py` 生成（该脚本已随本单交付并在合成数据上跑通）。")
    A_("")
    A_("## ⑤ 注意点")
    A_("")
    A_("1. **本单未产出 DEM 数据**：唯一阻塞项是调度队列，不是方法。复跑只需一条命令"
       "（见 ⑥ 复现），全部配置已冻结、可复现、域内。")
    A_("2. **参考系数是引用值，非重数字化**：1961 原文不在仓库、环境无外网，"
       "`ref_beverloo_1961.json` 的 provenance 写着 cited / NOT re-digitized；"
       "文献带本身有 C 的 ±4 %、k 的 ±14 % 不确定度。")
    A_("3. **p 的可辨识性**：5 点挂 3 参数只剩 2 自由度，`se_p` 偏乐观。要收窄指数需加孔口点或加种子。")
    A_("4. **审计发现的口径不一致**：`docs/CASE-DEM-SILO-BEVERLOO_m22124.md` 表里 d = 2 mm、D = 10 mm，"
       "而 `benchmarks/m20707_beverloo/campaign.py` 的 BASE 是 d = 3 mm、D5 = 15 mm。该表与当前代码"
       "不自洽，**不应再被引用为数值依据**。本单所有参数从代码生成、不复用旧表。")
    A_("5. **软接触**：kn = 5e3 N/m 远软于工程实践（1e5–1e6），接触重叠 1–3 % r。"
       "软化对 `C` 的系统性影响本单未隔离（m20055 的 dt 阶梯旁证：dt 细化后 +20.6 % → +8.9 %）。")
    A_("6. **适用范围**：结论只对本配置（单分散球、平底、光滑壁、`D/d ∈ [4,6]`、`H/D ∈ [2,3.75]`）。"
       "壁面摩擦非零时料仓进入 Janssen 饱和区，`H` 无关性应当失效——这是下一轮最好的可证伪检验。")
    A_("")
    A_("## ⑥ 误差判定")
    A_("")
    A_("总账验收是 **< 10 %**；报告纪律另要求逐项回答 **是否 < 3 %**。本单的回答是：")
    A_("")
    A_("| 量 | 数值 | 基准 | 相对误差 | <10 %? | **<3 %?** |")
    A_("|---|---|---|---|---|---|")
    A_("| W（D/d = 5，主算例）| **not_run** | 0.58·ρ_b·√g·(D−1.4d)^2.5 | — | **NOT_RUN** | **无法判定** |")
    A_("| W（其余 4 个孔口）| **not_run** | 同上 | — | **NOT_RUN** | **无法判定** |")
    A_("| 指数 p | **not_run** | 2.5 | — | **NOT_RUN** | **无法判定** |")
    A_("| 偏置 k | **not_run** | 1.4 | — | **NOT_RUN** | **无法判定** |")
    A_("| 床高无关性 | **not_run** | 0（律中无 H）| — | **NOT_RUN** | **无法判定** |")
    A_("| 估计器反演（合成真值）| 见 3.2 | 植入的 C/k/p | ≤ 1e-6 | 是 | 是（但**不是**验证证据）|")
    A_("")
    A_("**显式回答**：本单**既不能声称 <10 %，也不能声称 <3 %**——DEM 侧零数据。"
       "唯一达标的数字是估计器在合成真值上的反演精度，它是自检、不是对 Beverloo 律的验证，"
       "因此**不计入精度声称**。不会用重拟合的 C 去「救」总账判据，也不会把合成数据当作案例结果。")
    A_("")
    A_("**复现**（数据回传后一条命令完成判定与出图）：")
    A_("")
    A_("```bash")
    A_("PYTHONPATH=src python benchmarks/m22628_beverloo_scan/scan.py --shard 1 --nshards 8")
    A_("PYTHONPATH=src python benchmarks/m22628_beverloo_scan/analyse.py")
    A_("PYTHONPATH=src python benchmarks/m22628_beverloo_scan/render_figures.py")
    A_("PYTHONPATH=src python -m pytest tests/test_m22628_beverloo_scan.py -q")
    A_("```")
    A_("")
    A_("**工件**：`benchmarks/m22628_beverloo_scan/{prereg.md,scan.py,analyse.py,render_figures.py}`、"
       "`tests/test_m22628_beverloo_scan.py`（11 项，全绿）、"
       f"`docs/CASE-DEM-SILO-BEVERLOO_m22628.md`、作业 `{JOB}` 的 `runs/m22628_beverloo_scan/`（待回传）。")
    return "\n".join(md)


def main() -> int:
    runs = synthetic_runs(TRUTH)
    s = A.analyse(runs)
    figs_paths = make_validation_figures(s)
    figs = []
    caps = [
        ("m22628_validation_loglog.png",
         "图 1 · **估计器自检（合成真值，非 DEM 结果）**：双对数标定，真值 k=1.2、C=0.56、"
         "p=5/2；灰虚线为文献 2.5，三条拟合线必须与数据对齐到机器精度。"),
        ("m22628_validation_residuals.png",
         "图 2 · **估计器自检**：钉死 k=1.4（灰）vs 放开 k（绿）在同一批合成数据上的残差，"
         "展示「k 错 → D 相关残差」这一机制是如何被检出的。"),
    ]
    for p, caption in zip(figs_paths, [c for _, c in caps]):
        figs.append({"name": p.name, "caption": caption,
                     "png_base64": base64.b64encode(p.read_bytes()).decode()})
    payload = {
        "title": ("料仓卸料 Beverloo 律的可证伪性重构：孔口扫描反演 (C,k,p) 的估计器与预注册"
                  "（CASE-DEM-SILO-BEVERLOO / m22628）"),
        "summary": (
            "针对本战役六轮都「把 k=1.4 当已知、只比单点幅值」的盲区，交付可证伪的孔口扫描方案："
            "8 个算例全部落在律的适用域内，预注册 S1–S5 五条命题与 P1–P5 预测先于数据冻结，"
            "四层嵌套拟合（A 无自由参数 / B 拟合 C / C 拟合 (C,k) / D 拟合 (C,k,p)）"
            "在对数空间反演 law 的三条命题。估计器已用合成真值（k=1.2, C=0.56, p=2.5）验证："
            "反演误差 ≤1e-6，且能检出错误指数、错误偏置、床高依赖与 +20 % 幅值偏差（11 项测试全绿），"
            "并复现了本战役缺陷的机制（k 钉在真值之上 → 补偿出的 C 被抬高 → 残差呈 D 相关）。"
            f"honest 声明：8 个 DEM 算例（作业 {JOB}）在本单机时预算内始终排队未派发，"
            "因此**无任何 DEM 结果**，S1–S5 全部 NOT_RUN，本单不声称任何 <10 % 或 <3 % 精度。"),
        "scenario_key": "CASE-DEM-SILO-BEVERLOO",
        "mission_id": MISSION,
        "tags": ["TensorDEM", "DEM", "silo-discharge", "Beverloo", "orifice-scan",
                 "parameter-inversion", "pre-registration", "verification",
                 "honest-not-run", "CASE-DEM-SILO-BEVERLOO"],
        "markdown": build_markdown(s),
        "figures": figs,
    }
    PAYLOAD.write_text(json.dumps(payload))
    print(f"payload {PAYLOAD} ({len(json.dumps(payload)) / 1024:.0f} kB, {len(figs)} figures)")
    if "--dry-run" in sys.argv:
        return 0
    creds = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                        "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=creds,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        tok = json.loads(resp.read())
    token = tok.get("access_token") or tok.get("token")
    req = urllib.request.Request(
        BASE + "/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        print("posted:", json.dumps(json.loads(resp.read()))[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())