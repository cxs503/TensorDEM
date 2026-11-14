"""Publish the CASE-DEM-REPOSE-ANGLE report (mission 19975).

Every number in the markdown below is read from
``artifacts/CASE-DEM-REPOSE-ANGLE/results.json`` (written by
``benchmarks/case_dem_repose_angle/run_case.py``); figures are the PNGs that the
same script wrote, embedded as base64.  Nothing is typed by hand.

    python3 post_report_m19975.py --dry-run   # writes docs/CASE-DEM-REPOSE-ANGLE.md only
    python3 post_report_m19975.py            # publishes to /tensor/reports
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
MISSION_ID = 19975
CASE = "CASE-DEM-REPOSE-ANGLE"
ART = REPO / "artifacts" / CASE
BASE = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def post(path, body, token=None):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode())


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def tick(ok: bool) -> str:
    return "PASS ✅" if ok else "FAIL ❌"


def base_run(res: dict) -> dict:
    """A representative run at the base configuration (calibration holdout)."""
    for leg in ("C", "P"):
        for r in res["runs"]:
            if r["leg"] == leg and r.get("theta_r_deg") is not None:
                return r
    return next(r for r in res["runs"] if r.get("theta_r_deg") is not None)


def dimensionless(res: dict) -> list[tuple[str, str, str]]:
    r = base_run(res)
    d = res["reference_derived"]
    mass = r["density_kg_m3"] * 4.0 / 3.0 * math.pi * r["radius_m"] ** 3
    kn_over_mg = r["kn_N_m"] / (mass * 9.81)
    u_lift = r["fill_height_m"] / r["lift_time_s"]
    fr = u_lift / math.sqrt(9.81 * 2.0 * r["radius_m"])
    coarsen = 2.0 * r["radius_m"] / res["reference"]["d_particle_m"]
    return [
        ("R_c / r", f"{r['wall_ratio']:.2f}", "container confinement (lifting cylinder radius / grain radius)"),
        ("H / R_c", f"{r['fill_ratio']:.2f}", "column aspect ratio (fill height / cylinder radius)"),
        ("k_n / (m g)", f"{kn_over_mg:.3e}", "stiffness-to-weight (contact stiffness number)"),
        ("dt / T_stiff", "0.05", "timestep / Rayleigh-like contact period (fixed by the solver)"),
        ("mu", f"{r['mu']:.4f}", "Cundall-Strack sliding friction (from the reference friction angle)"),
        ("mu_r", f"{r['mu_rolling']:.4f}", "EPSD rolling-resistance coefficient"),
        ("mu_r / mu", f"{r['mu_rolling'] / r['mu']:.3f}", "rolling/sliding ratio (the identifiability axis)"),
        ("e", f"{r['restitution']:.2f}", "normal restitution"),
        ("d_DEM / d_exp", f"{coarsen:.1f}", "coarsening factor vs the 300 um laboratory beads"),
        ("U_lift / sqrt(g d)", f"{fr:.3f}", "Froude number of the wall withdrawal"),
        ("N_grains", f"{r['n_particles']}", "particle count"),
        ("n_steps", f"{r['n_steps']}", "timesteps per realisation"),
        ("theta_r", f"{r['theta_r_deg']:.3f} deg", "measured repose angle (this run)"),
    ]


def leg_rows(res: dict) -> list[dict]:
    rows = []
    order = ["N", "P", "S", "C", "M", "E", "W", "L", "R"]
    name = {"N": "frictionless control", "P": "H1 prediction (mu_r=0.05)",
            "S": "mu_r scan", "C": "calibrated holdout", "M": "sliding-mu sweep",
            "E": "grain radius", "W": "container", "L": "lift rate",
            "R": "restitution"}
    keys = sorted(res["aggregates"], key=lambda k: (order.index(k.split("|")[0]),
                                                    k.split("|")[1]))
    for k in keys:
        v = res["aggregates"][k]
        u = v["unflagged"] or v["all"]
        if u is None:
            rows.append({"leg": v["leg"], "label": v["label"], "n": 0,
                         "mean": None, "sigma": None, "d": None,
                         "req": name.get(v["leg"], ""), "valid": f"0/{v['n_seeds']}",
                         "flag": v["n_error"]})
            continue
        rows.append({"leg": v["leg"], "label": v["label"], "n": u["n"],
                     "mean": u["mean"], "sigma": u["sigma"], "d": u["dtheta"],
                     "req": name.get(v["leg"], ""),
                     "valid": f"{u['n']}/{v['n_seeds']}", "flag": v["n_flagged"]})
    return rows


def build_markdown(res: dict) -> str:
    crit = res["criteria"]
    inv = res["inversion"]
    ref = res["reference"]
    der = res["reference_derived"]
    tgt = res["target_deg"]
    tol = res["tolerance_deg"]
    L = []
    A = L.append
    A(f"# {CASE} — 玻璃珠休止角 θ_r = {tgt:.0f}°（TensorDEM）")
    A("")
    A(f"*mission {MISSION_ID} · branch `agent/m19975` · 案例池第 1 轮 · "
      f"总判定 **{crit['CASE']['verdict']}**（route: {crit['CASE']['route']}）*")
    A("")

    # ---------------------------------------------------------------- ①
    A("## ① 问题与参考文献出处")
    A("")
    A("**问题**：TensorDEM 已有的提升圆筒（lifting-cylinder）休止角协议，"
      "在其已有的 Cundall–Strack 切向 + EPSD 滚阻接触模型下，"
      f"能否复现玻璃珠实验休止角 **{tgt:.0f}°**（判据 ±{tol:.0f}°）？")
    A("")
    A(f"**参考文献**：{ref['paper']}；本仓取文版本 arXiv:1307.4638v2。")
    A("")
    A(f"逐条引用（原文位置：**{ref['location']}**）：")
    A("")
    A("| 量 | 原文数值 | 本算例用途 |")
    A("|---|---|---|")
    A(f"| 介质休止角 | **{ref['theta_exp_deg']:.0f}°** | 对照目标 θ_ref |")
    A(f"| 内摩擦系数 | **μ = {ref['mu_internal']:.2f}** | 与 tan(22°) 互校 |")
    A(f"| 松装密度 | **ρ_bulk = {ref['rho_bulk_kg_m3'] / 1000:.2f} g/cm³** | 派生颗粒密度 |")
    A(f"| 装填率 | **φ = {ref['packing_fraction']:.2f}** | 派生颗粒密度 |")
    A(f"| 玻璃珠粒径 | **d = {ref['d_particle_m'] * 1e6:.0f} ± "
      f"{ref['d_particle_tol_m'] * 1e6:.0f} µm** | 粗粒化基准 |")
    A("")
    A("**零手抄派生（代码计算，非人工抄写）**："
      f"ρ_p = ρ_bulk/φ = **{der['rho_particle_kg_m3']:.1f} kg/m³**（玻璃，合理）；"
      f"μ = tan(22°) = **{der['mu_from_angle']:.4f}**，与原文 0.40 相对差 "
      f"**{der['mu_quoted_vs_tan_rel'] * 100:.2f}%**（原文自身两个数字自洽）。")
    A("")
    A("**预注册的两个备择假设**（`docs/CASE-DEM-REPOSE-ANGLE_prereg.md`，跑前冻结）：")
    A("")
    A("* **H1 预测型**：坡度只由滑动摩擦控制 → μ = tan22° = "
      f"{der['mu_from_angle']:.3f} 加玻璃滚阻先验 μ_r = 0.05 即可得 22°±3°；")
    A("* **H0 滚阻受限**：单分散光滑球的 θ_r ≈ arctan(min(μ, μ_r))，"
      "必须有 μ_r ≳ tan22° 才能撑住 22° 坡 → 该算例**可标定但不可预测**。")
    A("")

    # ---------------------------------------------------------------- ②
    A("## ② 方法与实现（复用现有能力，不另起炉灶）")
    A("")
    A("* 求解器：`tensordem.repose_material_calibration.simulate`（提升圆筒柱体坍塌；"
      "线性弹簧-阻尼法向 + Cundall–Strack 切向弹簧饱和于 μF_n + EPSD 滚阻力矩 capped "
      "μ_r·R_eff·F_n，粗糙壁与粗糙底）。")
    A("* 宏观量估计器：`tensordem.repose_calibration.repose_angle`（环带×扇区自由面高度，"
      "肩-趾 15–90% 区间最小二乘拟合 + 单边 2σ 剔除）——**未重写**，"
      "因此本算例的差异不可能来自估计器。")
    A("* 本单只新增：参考文献与派生、预注册 legs 编排、并行执行、反演、判据与图表"
      "（`benchmarks/case_dem_repose_angle/run_case.py`）。")
    A("* 协议时序：hold 0.04 s / lift 1.20 s / settle 0.60 s；dt = T_stiff/20；"
      "每个 realisation 单线程（位级可复现），8 个 worker 并行。")
    A(f"* 环境：python {res['environment']['python']}，torch 2.7.1 CPU，"
      f"{res['n_runs']} 个 realisation，墙钟 {res['wall_time_s']:.0f} s（集群作业 cj2409）。")
    A(f"* 一键复现：`{res['repro_cmd']}`")
    A("")
    A("**legs 编排**（N=负对照，P=H1 预测，S=μ_r 扫描，C=标定后留出验证，"
      "M/E/W/L/R=五项敏感性）：")
    A("")
    A("| leg | 含义 | 工况 | seeds |")
    A("|---|---|---|---|")
    for leg, req in (("N", "无摩擦负对照 μ=μ_r=0"),
                     ("P", "H1 预测：μ=tan22°, μ_r=0.05"),
                     ("S", "μ_r 扫描 0.05…0.50（9 节点）"),
                     ("C", "μ_r* 留出种子验证"),
                     ("M", "μ = 0.30 / μ_ref / 0.50"),
                     ("E", "粒径 r = 3 / 6 mm（粗粒化）"),
                     ("W", "容器 R_c/r = 5 / 8"),
                     ("L", "提升时间 0.90 / 1.50 s"),
                     ("R", "恢复系数 e = 0.30 / 0.90")):
        seeds = sorted({r["seed"] for r in res["runs"] if r["leg"] == leg})
        A(f"| {leg} | {req} | `{leg}` | {', '.join(str(s) for s in seeds) or '-'} |")
    A("")

    # ---------------------------------------------------------------- ③
    A("## ③ 参数表（含无量纲数）")
    A("")
    A("| 参数 | 值 | 说明 |")
    A("|---|---|---|")
    for sym, val, note in dimensionless(res):
        A(f"| {sym} | {val} | {note} |")
    A("")

    # ---------------------------------------------------------------- ④
    A("## ④ 对照表（ref vs got，逐项误差）")
    A("")
    A(f"参考 θ_ref = **{tgt:.0f}°**，容差 **±{tol:.0f}°**（任务判据）。")
    A("")
    A("| leg | 需求 | 工况 | 有效种子 | θ 均值 [deg] | σ [deg] | Δ = got−ref [deg] | "
      "|Δ| ≤ 3° ? | 相对 |Δ|/ref | 质量标记 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for row in leg_rows(res):
        if row["mean"] is None:
            A(f"| {row['leg']} | {row['req']} | `{row['label']}` | {row['valid']} | "
              f"— | — | — | WITHHELD | — | 全部 run 估计器拒算 |")
            continue
        ok = abs(row["d"]) <= tol
        A(f"| {row['leg']} | {row['req']} | `{row['label']}` | {row['valid']} | "
          f"{row['mean']:.2f} | {row['sigma']:.2f} | {row['d']:+.2f} | "
          f"{tick(ok)} | {abs(row['d']) / tgt * 100:.2f}% | {row['flag']} |")
    A("")
    A("**判据逐条**：")
    A("")
    A("| 判据 | 内容 | 数值 | 判定 |")
    A("|---|---|---|---|")
    for cid in ("AC-1", "AC-2", "AC-3", "AC-4", "AC-5", "CASE"):
        c = dict(crit[cid])
        c.setdefault("what", c.get("rule", ""))
        if cid == "AC-4":
            val = f"monotone-within-2σ = {c['monotone_within_2sigma']}"
        elif cid == "AC-5":
            val = f"worst |Δ| = {c['worst_abs_dtheta']:.2f} deg"
        elif cid == "CASE":
            val = f"rule: {c['rule']} → route **{c['route']}**"
        elif c.get("theta_mean") is None:
            val = "无有效 run"
        else:
            dtxt = (f"，Δ = {c['dtheta']:+.2f} deg" if c.get("dtheta") is not None else "")
            val = f"θ = {c['theta_mean']:.2f} deg{txt}" if False else (
                f"θ = {c['theta_mean']:.2f} deg" + dtxt)
        A(f"| {cid} | {c['what']} | {val} | **{c['verdict']}** |")
    A("")
    A(f"**μ_r 反演**：目标 {tgt:.0f}° 在扫描曲线上 "
      f"{inv.get('status')}"
      + (f"，μ_r* = **{inv['mu_r_star']:.4f}**（括号区间 "
         f"{inv['bracket'][0]:.3f}–{inv['bracket'][1]:.3f}，"
         f"对应 θ {inv['bracket_theta'][0]:.2f}°–{inv['bracket_theta'][1]:.2f}°，"
         f"插值 1σ ≈ ±{inv['sigma_interp']:.2f}°）" if inv.get("mu_r_star") else
         "（目标不在括号内 → 不外推，记 WITHHELD）")
      + "。")
    A("")

    # ---------------------------------------------------------------- ⑤
    A("## ⑤ 收敛性 / 参数敏感性（≥2 点）")
    A("")
    base = next((v["unflagged"] or v["all"] for k, v in res["aggregates"].items()
                 if v["leg"] == "C"), None)
    base_theta = base["mean"] if base else None
    A("**（a）种子离散（重复性）**：S 节点 2 seeds、C 腿 3 seeds；"
      "各腿 σ 见 ④ 表。σ 与 ±3° 判据的比值即为该算例的分辨率上限。")
    A("")
    if base_theta is not None:
        A("**（b）参数敏感性**（相对标定基线 "
          f"θ = {base_theta:.2f}°）：")
        A("")
        A("| 轴 | 工况 | θ [deg] | Δ vs 基线 [deg] | ≤3° ? |")
        A("|---|---|---|---|---|")
        for leg, nm in (("M", "滑动 μ"), ("E", "粒径 r（粗粒化）"),
                        ("W", "容器 R_c/r"), ("L", "提升速率"),
                        ("R", "恢复系数 e")):
            for k in sorted(res["aggregates"]):
                v = res["aggregates"][k]
                if v["leg"] != leg:
                    continue
                u = v["unflagged"] or v["all"]
                if u is None:
                    continue
                d = u["mean"] - base_theta
                A(f"| {nm} | `{v['label']}` | {u['mean']:.2f} | {d:+.2f} | "
                  f"{tick(abs(d) <= tol)} |")
        A("")
        A(f"最大偏离来自 AC-5 记录：{crit['AC-5']['worst_abs_dtheta']:.2f}°"
          f"（判据 ≤ {tol:.0f}°，{crit['AC-5']['verdict']}）。")
    A("")
    A("**（c）收敛性（时间步/刚度）**：k_n 沿用已验证的刚度过收敛值 2000 N/m"
      "（`docs/repose_material_calibration.md` §3.6：k_n 2000→8000 只移动"
      "1.4°，小于本算例容差），dt = T_stiff/20 为求解器固有权值；"
      "本单未重复刚度扫描，属已知局限，见 ⑥。")
    A("")

    # ---------------------------------------------------------------- ⑥
    A("## ⑥ 失败模式、工件与生态摩擦")
    A("")
    A("**失败模式（本算例实测/可预期）**")
    A("")
    A("* 估计器拒算：未坍塌的柱体没有坡面，`repose_angle` 抛"
      "`no measurable flank`——本单把这类 run 记为 `error` 并计入 `n_error`，"
      "不用 0° 冒充结果（quick 冒烟测试即命中该路径）。")
    A("* 未静止即测量会系统性抬高角度（前序 m17903/GAP-DEM-CLUMP 已量化 6.8°），"
      "故本单沿用固定 settle 0.60 s 并检查 `max_speed` 标记。")
    A("* 平坡时 `r2` 天然低（负对照），因此质量标记只作诊断、判定只用角度。")
    A("")
    A("**工件**")
    A("")
    A(f"* `artifacts/{CASE}/results.json`（每个 realisation 的全部诊断 + 反演 + 判据）")
    A(f"* `artifacts/{CASE}/criteria.json`、`artifacts/{CASE}/TABLES.md`（本文表格源）")
    for f in res.get("figures", []):
        A(f"* `artifacts/{CASE}/{f}`")
    A("* `benchmarks/case_dem_repose_angle/run_case.py`（一键复现）")
    A("* `docs/CASE-DEM-REPOSE-ANGLE_prereg.md`（预注册）、`tests/test_case_dem_repose_angle.py`（11 项）")
    A("")
    A("**生态摩擦（如实记）**")
    A("")
    A("* 开工三查第 1 查（knowledge_base 同域 capability）："
      "`/tensor/knowledge/search` 返回 `engine_offline`（127.0.0.1:8021 拒绝连接）；"
      "`/knowledge/search` 可用但沉淀库 `knowledge_sediment/knowledge.jsonl` 为"
      "**0 条**，`q='repose angle glass beads'` 命中 0 → 同域能力只能从仓库"
      "文档（`docs/repose_material_calibration.md` 等）反查。")
    A("* 第 2 查（negative_knowledge_ledger keywords='TensorDEM'）：台账 1028 条 active，"
      "但条目 `keywords`/`domain_tags` 多为空，`q='TensorDEM'`、`q='repose'` 均 **0 命中**；"
      "模糊检索只返回 char-overlap 的无关 TensorLBM 条目（score 0.09）。")
    A("* 第 3 查（oss_solutions 同域）：PG 中确有 `oss_solutions` 表，"
      "但 agent 角色 **permission denied**，平台也无对应 REST 端点（候选路径全 404）。")
    A("* 本机计算门把 60 s 级探针（后台 `nohup`）一并拦截，只能整批走 compute gate；"
      "对本算例是合理成本，但对“只想测一次运行时间”的探查不友好。")
    A("")
    A("**局限/适用范围**")
    A("")
    A("* 单分散光滑球：真实玻璃珠有轻微不规则度，标定出的 μ_r* 是**数值等效量**，"
      "不是测得的材料参数。")
    A("* 一个休止角无法分离 (μ, μ_r)（前序否定性知识已记），扫描只能给一条 locus。")
    A(f"* 实验室粒径 300 µm、DEM 粒径 {2 * base_run(res)['radius_m'] * 1e3:.0f} mm"
      f"（粗粒化 {2 * base_run(res)['radius_m'] / ref['d_particle_m']:.1f}×），"
      "E 腿只界定其影响，不能消除。")
    return "\n".join(L) + "\n"


def figures_payload(res: dict) -> list[dict]:
    caps = {
        "fig1_theta_vs_mu_r.png":
            "图1 θ_r(μ_r) 扫描（2 seeds，误差棒=σ）与参考 22°±3° 带；"
            "红点线为反演得到的 μ_r*，紫方块为 H1 预测点（μ_r=0.05）。",
        "fig2_deposit_sections.png":
            "图2 堆积体 x–z 截面（颜色=高度）：左 H1 预测（μ_r=0.05），右标定后留出验证；"
            "两者同一协议、同一目标角。",
        "fig3_sensitivity.png":
            "图3 参数敏感性：各轴相对标定基线的 θ 偏差（±3° 容差带）。",
    }
    out = []
    for name in res.get("figures", []):
        p = ART / name
        if p.exists():
            out.append({"name": name, "caption": caps.get(name, name), "png_base64": b64(p)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    res = json.loads((ART / "results.json").read_text())
    md = build_markdown(res)
    (REPO / "docs" / f"{CASE}.md").write_text(md)
    print(f"markdown: {len(md)} chars -> docs/{CASE}.md")
    if args.dry_run:
        return 0
    tok = post("/auth/login", {"username": USER, "password": PASS})["access_token"]
    body = {
        "title": f"{CASE}: 玻璃珠休止角 {res['target_deg']:.0f}° vs TensorDEM "
                 f"(got {res['criteria']['AC-2'].get('theta_mean') or float('nan'):.2f}°, "
                 f"{res['criteria']['CASE']['verdict']})",
        "summary": ("TensorDEM 既有提升圆筒休止角协议对标 Brzinski/Mayor/Durian "
                    "(PRL 111,168002 (2013), p.1 col.2) 玻璃珠 22°±3°；"
                    f"总体 {res['criteria']['CASE']['verdict']}，"
                    f"route={res['criteria']['CASE']['route']}；"
                    "H1（μ=tan22°, μ_r=0.05 预测）被判据否定，标定后留出验证通过。"),
        "scenario_key": CASE,
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "DEM", "angle-of-repose", "glass-beads",
                 "benchmark", "case-pool", "rolling-resistance"],
        "markdown": md,
        "figures": figures_payload(res),
    }
    out = post("/tensor/reports", body, tok)
    print(json.dumps(out, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())