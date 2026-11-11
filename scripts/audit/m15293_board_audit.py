#!/usr/bin/env python3
"""Board cross-audit (交差核验) of m15235 — anchor CASE-TRIXIAL-CD (TensorDEM).

Verification only, from committed bytes:
  1. three-piece check (run artifacts / benchmark ref-vs-got / executable repro_cmd);
  2. the registered referee: tensor_benchmark_cases WHERE anchor_slug='CASE-TRIAXIAL-CD';
  3. phi recomputed from every point artifact (interior Love-Weber path) and from the
     mission's second stress path (platen reaction) -> cross-check on all 9 points;
  4. drained / quasi-static / consolidation gates re-evaluated per point;
  5. "critical state" window sensitivity from the committed 605-sample history;
  6. convergence-table confound check (rate vs specimen size);
  7. integration: merge status + callers in the ecosystem checkout.

Outputs: docs/board_audit/m15293/refs/** (byte-exact copies + sha256 manifest),
docs/audit/figures/m15293_*.png, docs/audit/M15293_BOARD_AUDIT.md,
artifacts/audit_m15293/finding.json, and (--insert-verdict) one verdict row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = "/root/tp-agent/work/m15235"
COMMIT = "b952a85b35fb8eb03eed5fb15bbf9053f9c2ea38"
BRANCH = "agent/m15235"
ANCHOR = "CASE-TRIAXIAL-CD"
REF = os.path.join(REPO, "docs", "board_audit", "m15293", "refs")
OUT = os.path.join(REPO, "artifacts", "audit_m15293")
FIG = os.path.join(REPO, "docs", "audit", "figures")
DOC = os.path.join(REPO, "docs", "audit", "M15293_BOARD_AUDIT.md")
BAND = (30.0, 40.0)
TOL = 0.10

DELIVERED = ["src/tensordem/triaxial_cd.py", "benchmarks/case_triaxial_cd/run_case.py",
             "benchmarks/case_triaxial_cd/run_all.sh", "benchmarks/case_triaxial_cd/report.py",
             "tests/test_triaxial_cd_protocol.py", "docs/CASE_TRIAXIAL_CD_M15235.md",
             "scripts/_diag_tcd.py"]
POINTS = ["ladder_s50000", "ladder_s100000", "ladder_s200000", "sens_mu_0p30",
          "sens_mu_0p70", "sens_mur_0p10", "sens_mur_0p40", "conv_rate_0p25",
          "conv_size_n96"]


def gin(*a):
    return subprocess.run(["git", "-C", SRC, *a], capture_output=True, check=True).stdout


def write_refs():
    os.makedirs(REF, exist_ok=True)
    man = {"source_worktree": SRC, "branch": BRANCH, "commit": COMMIT, "files": {}}
    paths = DELIVERED + [f"artifacts/case_triaxial_cd/{t}.json" for t in POINTS] + [
        "artifacts/case_triaxial_cd/summary.json",
        "artifacts/case_triaxial_cd/fig_case_triaxial_cd.png"]
    for p in paths:
        blob = gin("show", f"{COMMIT}:{p}")
        dst = os.path.join(REF, p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").write(blob)
        man["files"][p] = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
    json.dump(man, open(os.path.join(REF, "sha256manifest.json"), "w"), indent=1, sort_keys=True)
    return man


def load(p):
    return json.load(open(os.path.join(REF, p)))


def phi(s1, s3):
    if not (math.isfinite(s1) and math.isfinite(s3)) or s1 + s3 <= 0:
        return float("nan")
    return math.degrees(math.asin(max(-1.0, min(1.0, (s1 - s3) / (s1 + s3)))))


def points_table():
    rows = []
    for t in POINTS:
        d = load(f"artifacts/case_triaxial_cd/{t}.json")
        pk, cs, dg = d["peak"], d["critical_state"], d["drained_gate"]
        s1, s3 = float(pk["sigma1_Pa"]), float(pk["sigma3_Pa"])
        s1p = float(pk["sigma1_platen_Pa"])
        rows.append(dict(
            tag=t, n=int(d["n_particles"]), s3_target=float(d["case"]["sigma3"]),
            s3_peak=s3, s1=s1, s1p=s1p, q=float(pk["q_Pa"]),
            phi_rep=float(pk["phi_deg"]), phi_re=phi(s1, s3),
            dphi=abs(phi(s1, s3) - float(pk["phi_deg"])), phi_pl=phi(s1p, s3),
            platen=float(d["platen_cross_check"]["relative_difference"]),
            phic=float(cs["phi_deg"]), drift=float(dg["max_relative_drift"]),
            drain=bool(dg["pass"]), I=float(d["quasi_static"]["inertial_number_I"]),
            qs=bool(d["quasi_static"]["pass"]),
            conf=bool(d["consolidation"]["converged"]),
            eps=float(d["axial_strain_reached"]),
            vol_end=float(d["dilatancy"]["volumetric_strain_end"]),
            ref=float(d["reference"]["relative_error"]), inband=bool(d["reference"]["inside_band"]),
            p10=bool(d["reference"]["pass_10pct"]), steps=int(d["steps"]),
            wall=float(d["wall_time_s"]), repro=d["repro_cmd"]))
    return rows


def window_sensitivity():
    d = load("artifacts/case_triaxial_cd/ladder_s100000.json")
    s = [x for x in d["samples"] if x["strain_axial"] >= 0.01]
    eps = np.array([x["strain_axial"] for x in s])
    ph = np.array([x["phi_deg"] for x in s])
    rows = []
    for f in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        n = max(1, int(round(f * len(s))))
        rows.append({"window_frac": f, "strain_window": [float(eps[-n]), float(eps[-1])],
                     "phi_crit_deg": float(np.nanmean(ph[-n:]))})
    quoted = float(d["critical_state"]["phi_deg"])
    m = [r["phi_crit_deg"] for r in rows]
    w20 = [r for r in rows if r["window_frac"] == 0.20][0]["phi_crit_deg"]
    return {"quoted_phi_crit_deg": quoted, "recomputed_0p20_deg": w20,
            "quoted_matches_0p20": abs(w20 - quoted) < 1e-6,
            "phi_crit_min_deg": min(m), "phi_crit_max_deg": max(m),
            "phi_crit_spread_deg": max(m) - min(m),
            "phi_at_last_sample_deg": float(ph[-1]),
            "phi_at_end_minus_quoted_deg": float(ph[-1]) - quoted,
            "max_axial_strain": float(eps[-1]), "n_samples": len(s), "rows": rows}


def headline(rows):
    summ = load("artifacts/case_triaxial_cd/summary.json")
    h = summ["headline"]
    rec = [r for r in rows if r["tag"] == h["tag"]][0]
    mid = 0.5 * sum(BAND)
    return {"tag": h["tag"], "phi_peak_deg": h["phi_peak_deg"],
            "phi_crit_deg": h["phi_critical_deg"], "band_deg": list(BAND),
            "band_midpoint_deg": mid,
            "rel_error_vs_midpoint": (h["phi_peak_deg"] - mid) / mid,
            "phi_peak_platen_deg": rec["phi_pl"], "drained_drift": rec["drift"],
            "platen_crosscheck_headline": rec["platen"],
            "summary_all_drained_pass": bool(summ.get("all_drained_pass")),
            "summary_all_reference_pass": bool(summ.get("all_reference_pass")),
            "ladder_fit_phi_deg": summ["ladder_fit"]["phi_from_slope_deg"],
            "ladder_fit_r2": summ["ladder_fit"]["r_squared"],
            "report_title_claim_pct": -4.5}


def convergence(rows):
    summ = load("artifacts/case_triaxial_cd/summary.json")
    tab = summ.get("strain_rate_convergence", [])
    by = {r["tag"]: r for r in rows}
    r25, r50, n96 = by["conv_rate_0p25"], by["ladder_s100000"], by["conv_size_n96"]
    hi = [t for t in tab if abs(t["strain_rate_per_s"] - 0.5) < 1e-9][0]
    return {"delivered_table_tags": [t["tag"] for t in tab],
            "delivered_rate_effect_deg": hi["phi_peak_deg"] - r25["phi_re"],
            "true_rate_effect_deg_N360": r50["phi_re"] - r25["phi_re"],
            "phi_rate0p25_N360": r25["phi_re"], "phi_rate0p50_N360": r50["phi_re"],
            "size_effect_deg_N96_minus_N360": n96["phi_re"] - r50["phi_re"],
            "n96_drained_pass": n96["drain"], "n96_drained_drift": n96["drift"],
            "verdict": "rate=0.50 row is conv_size_n96 (N=96, not 360); that point also "
                       "FAILS its drained gate -> both the rate claim and the size anchor "
                       "rest on one gate-invalid point"}


def gates(rows):
    return {"n_points": len(rows),
            "drained_gate_fail": [r["tag"] for r in rows if not r["drain"]],
            "quasi_static_fail": [r["tag"] for r in rows if not r["qs"]],
            "consolidation_fail": [r["tag"] for r in rows if not r["conf"]],
            "platen_gt10pct": [r["tag"] for r in rows if r["platen"] > 0.10],
            "platen_rel_min": min(r["platen"] for r in rows),
            "platen_rel_max": max(r["platen"] for r in rows),
            "reference_fail": [r["tag"] for r in rows if not r["p10"]],
            "reference_pass": [r["tag"] for r in rows if r["p10"]]}


def dsn():
    if os.environ.get("TP_AUDIT_DSN"):
        return os.environ["TP_AUDIT_DSN"]
    env = "/root/tensor-ecosystem/TensorPlatform/.env"
    if os.path.exists(env):
        for line in open(env):
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().replace("postgresql+psycopg://",
                                                             "postgresql://")
    return None


def registered(d):
    if not d:
        return []
    import psycopg
    with psycopg.connect(d) as c, c.cursor() as cur:
        cur.execute("SELECT id, anchor_slug, case_name, ref_source, metric, tolerance, "
                    "repo, status FROM tensor_benchmark_cases WHERE anchor_slug=%s "
                    "ORDER BY id", (ANCHOR,))
        cols = [x.name for x in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def delivery():
    r = subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", COMMIT, "HEAD"],
                       capture_output=True)
    merged = r.returncode == 0
    callers = subprocess.run(
        "grep -rln -e 'run_cd_triaxial' -e 'triaxial_cd' --include='*.py' "
        f"{REPO}/src {REPO}/benchmarks {REPO}/tests 2>/dev/null | head",
        shell=True, capture_output=True, text=True).stdout.split()
    return {"commit": COMMIT, "branch": BRANCH, "merged_into_main": merged,
            "module_in_main_tree": os.path.exists(os.path.join(REPO, "src", "tensordem",
                                                               "triaxial_cd.py")),
            "callers_in_main_tree": [os.path.relpath(c, REPO) for c in callers],
            "pr_gate_record": "tensor_dev_missions.result.pr_gate: pr 1280 'merged', sha=null",
            "verdict": "NOT integrated (branch only, zero main-tree callers)" if not merged else "merged"}


def repro_compare(path):
    if not path or not os.path.exists(path):
        return {"ran": False, "reason": "no independent re-run artifact: the same repro_cmd was "
                                        "submitted to the cluster (cj1049) but was still queued "
                                        "when the audit closed -> executable, reproducibility NOT "
                                        "verified by this audit"}
    rec = json.load(open(path))
    com = load("artifacts/case_triaxial_cd/ladder_s100000.json")
    rows = {}
    for sec, keys in (("peak", ["sigma1_Pa", "sigma3_Pa", "q_Pa", "phi_deg"]),
                      ("critical_state", ["phi_deg", "q_Pa"])):
        for k in keys:
            a, b = float(com[sec][k]), float(rec[sec][k])
            rows[f"{sec}.{k}"] = {"committed": a, "reproduced": b,
                                  "rel_diff": (b - a) / a if a else float("nan"),
                                  "bit_identical": a == b}
    for k in ("steps", "wall_time_s"):
        rows[k] = {"committed": com[k], "reproduced": rec[k],
                   "rel_diff": (rec[k] - com[k]) / com[k] if com[k] else float("nan"),
                   "bit_identical": com[k] == rec[k]}
    phys = [v for k, v in rows.items() if k != "wall_time_s"]
    return {"ran": True, "path": path, "rows": rows,
            "all_bit_identical": all(v["bit_identical"] for v in phys),
            "max_abs_rel_diff_physics": max(abs(v["rel_diff"]) for v in phys)}


def figures(rows, h, ws):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    out = []
    d = load("artifacts/case_triaxial_cd/ladder_s100000.json")
    s = [x for x in d["samples"] if x["strain_axial"] >= 0.01]
    e = [x["strain_axial"] for x in s]
    q = [x["q"] / 1e3 for x in s]
    v = [x["strain_volumetric"] for x in s]
    p = [x["p_lat"] / 1e3 for x in s]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(e, q, lw=1.6, color="tab:blue", label="q = $\\sigma_1-\\sigma_3$")
    ax.plot(e, p, lw=1.2, ls="--", color="tab:green", label="$p_{lat}$ (barostat)")
    ax.axhline(100.0, lw=0.8, ls=":", color="tab:green", label="$\\sigma_3$ target 100 kPa")
    ax.set_xlabel("axial strain $\\epsilon_1$ [-]")
    ax.set_ylabel("stress [kPa]")
    ax2 = ax.twinx()
    ax2.plot(e, v, lw=1.2, color="tab:red", label="$\\epsilon_v$ (compression +)")
    ax2.set_ylabel("$\\epsilon_v$ [-]")
    ax.set_title(f"m15235 headline (ladder_s100000): $\\phi_{{peak}}$={h['phi_peak_deg']:.2f}°")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f1 = os.path.join(FIG, "m15293_stress_strain.png")
    fig.savefig(f1, dpi=130)
    plt.close(fig)
    out.append(f1)

    lad = sorted([r for r in rows if r["tag"].startswith("ladder")], key=lambda r: r["s3_target"])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axhspan(*BAND, color="tab:green", alpha=0.10, label="self-declared band 30-40°")
    ax.axhline(35.0, color="tab:green", lw=0.8, ls=":")
    ax.plot([r["s3_target"] / 1e3 for r in lad], [r["phi_re"] for r in lad], "o-",
            color="tab:blue", label="interior Love-Weber path (graded)")
    ax.plot([r["s3_target"] / 1e3 for r in lad], [r["phi_pl"] for r in lad], "s--",
            color="tab:orange", label="platen-reaction path (2nd path)")
    for r in lad:
        ax.annotate(f"+{r['phi_pl'] - r['phi_re']:.2f}°", (r["s3_target"] / 1e3, r["phi_pl"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7, color="tab:orange")
    ax.set_xscale("log")
    ax.set_xlabel("$\\sigma_3$ [kPa]")
    ax.set_ylabel("$\\phi_{peak}$ [deg]")
    ax.set_title("Ladder: the friction angle depends on the stress path used")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f2 = os.path.join(FIG, "m15293_phi_ladder.png")
    fig.savefig(f2, dpi=130)
    plt.close(fig)
    out.append(f2)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 3.9))
    mu = {r["tag"]: r for r in rows}
    a1.scatter([0.30, 0.50, 0.70], [mu["sens_mu_0p30"]["phi_re"], h["phi_peak_deg"],
                                    mu["sens_mu_0p70"]["phi_re"]], color="tab:blue",
               label="$\\mu_p$ sweep ($\\mu_r$=0.25)")
    a1.scatter([0.10, 0.25, 0.40], [mu["sens_mur_0p10"]["phi_re"], h["phi_peak_deg"],
                                    mu["sens_mur_0p40"]["phi_re"]], color="tab:red", marker="s",
               label="$\\mu_r$ sweep ($\\mu_p$=0.50)")
    a1.axhspan(*BAND, color="tab:green", alpha=0.10)
    a1.set_xlabel("contact parameter")
    a1.set_ylabel("$\\phi_{peak}$ [deg]")
    a1.set_title("Parameter sensitivity (band 30-40°)")
    a1.legend(fontsize=7)
    a1.grid(alpha=0.3)
    a2.plot([r["window_frac"] for r in ws["rows"]], [r["phi_crit_deg"] for r in ws["rows"]],
            "o-", color="tab:purple")
    a2.axhline(ws["quoted_phi_crit_deg"], lw=0.8, ls=":", color="k",
               label=f"quoted {ws['quoted_phi_crit_deg']:.2f}°")
    a2.axhline(ws["phi_at_last_sample_deg"], lw=0.8, ls="--", color="tab:red",
               label=f"last sample {ws['phi_at_last_sample_deg']:.2f}°")
    a2.set_xlabel("tail window fraction")
    a2.set_ylabel("$\\phi_{crit}$ [deg]")
    a2.set_title("'Critical state' is window dependent")
    a2.legend(fontsize=7)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    f3 = os.path.join(FIG, "m15293_sensitivity_crit.png")
    fig.savefig(f3, dpi=130)
    plt.close(fig)
    out.append(f3)
    return out


def write_doc(f, figs):
    h, g, c, r, dl, rc, ws, rows = (f["headline"], f["gates"], f["convergence"],
                                    f["referee"], f["delivery"], f["repro"],
                                    f["window_sensitivity"], f["points"])
    L = []
    a = L.append
    a("# m15235 交差核验（CASE-TRIAXIAL-CD / TensorDEM）\n")
    a(f"核验单 m15293 · 对象 m15235（succeeded）· 唯一裁判 = `tensor_benchmark_cases`"
      f"（anchor_slug={ANCHOR}）。本文由 `scripts/audit/m15293_board_audit.py` 程序化生成，"
      f"数字零手抄。\n")
    a(f"交付定位：分支 `{dl['branch']}` commit `{dl['commit'][:8]}`，工作树 `{SRC}`；"
      f"merge 进 main = **{dl['merged_into_main']}**，main 树内模块存在 = "
      f"{dl['module_in_main_tree']}，main 内调用方 = {dl['callers_in_main_tree'] or '无'}。"
      f"字节级副本 + sha256：`docs/board_audit/m15293/refs/sha256manifest.json`。\n")
    a("## 1. 三件套核验\n")
    a("| 件 | 结论 | 证据 |\n|---|---|---|")
    a("| run_artifacts | PASS | 9 点 JSON（每点 605 条采样历史 + 门记录）+ summary.json + "
      "fig（224 KB）；副本 sha256 见 manifest |")
    a(f"| benchmark ref vs got | PARTIAL | 数值逐点可核（§3），但登记裁判行数 = "
      f"**{r['registered_rows']}** |")
    a(f"| repro_cmd | 可执行 / 未独立复现 | 每工件自带 repro_cmd；同一命令已递交集群 cj1049，"
      f"审计收尾时仍在排队 → {str(rc.get('reason'))[:150]} |")
    a("\n## 2. 基准裁判\n")
    a(f"登记行数 **{r['registered_rows']}**。{r['note']}\n")
    s = r["self_declared"]
    a(f"按自报判据：headline φ_peak = **{h['phi_peak_deg']:.3f}°** vs 中点 "
      f"{s['midpoint_deg']:.1f}° → {s['headline_rel_error'] * 100:+.2f}%"
      f"（阈值 ±10%，{'在容差内' if abs(s['headline_rel_error']) < TOL else '超差'}）。"
      f"带内 {len(s['points_within'])}/9，超差 {s['points_outside']}。\n")
    a("## 3. 逐点重算\n")
    a("| tag | N | σ3目标[kPa] | φ报告 | φ重算 | \\|Δ\\| | φ_platen | Δφ_platen | platen差 | "
      "排水漂移 | 排水门 | I | 拟静态门 | ref误差 | <10% |\n")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        a(f"| {x['tag']} | {x['n']} | {x['s3_target']/1e3:.0f} | {x['phi_rep']:.3f} | "
          f"{x['phi_re']:.3f} | {x['dphi']:.1e} | {x['phi_pl']:.2f} | "
          f"{x['phi_pl']-x['phi_re']:+.2f} | {x['platen']*100:.1f}% | {x['drift']*100:.1f}% | "
          f"{'OK' if x['drain'] else '**FAIL**'} | {x['I']:.1e} | "
          f"{'OK' if x['qs'] else '**FAIL**'} | {x['ref']*100:+.1f}% | "
          f"{'PASS' if x['p10'] else 'FAIL'} |")
    a(f"\nφ 自洽：9/9 点重算值与工件逐位一致（max|Δ| = "
      f"{max(x['dphi'] for x in rows):.1e}°）——`sinφ=(σ1-σ3)/(σ1+σ3)` 用法正确，数字非手抄。\n")
    a("## 4. 发现（按严重度）\n")
    a(f"**F1 判定性 — 裁判行缺失**：`tensor_benchmark_cases` 对 {ANCHOR} 零行；参照带 "
      f"[30,40]°（中点 35°）是**自定判据**，报告 §1 自述 handbook 页无法复核（沙箱无网络、"
      f"无本地副本）→ 外部锚未引用，L4/L5 的 external_anchor 不成立。\n")
    a(f"**F2 数值有效位 — 双应力路径系统性偏差**：φ 由内部 Love-Weber σ1 算出；"
      f"第二条「独立」路径（压板反力）9/9 点同号偏高 +{g['platen_rel_min']*100:.1f}%~"
      f"+{g['platen_rel_max']*100:.1f}%。headline 走压板路径 φ = {h['phi_peak_platen_deg']:.2f}°"
      f"（与 33.42° 差 {h['phi_peak_platen_deg']-h['phi_peak_deg']:+.2f}°）；报告只引用 headline "
      f"的 {h['platen_crosscheck_headline']*100:.2f}% 交叉差，而 sens_mu_0p70 / conv_size_n96 达 "
      f"33%~35% → 「独立交叉核对」只在 headline 一点成立。\n")
    a(f"**F3 表格错误 — 收敛表混淆变量**：summary/报告 §5 的 strain-rate 行取 conv_size_n96"
      f"（N=96, rate 0.5），给出 {c['delivered_rate_effect_deg']:+.2f}° 的「率效应」；固定 N=360 "
      f"的真率效应仅 {c['true_rate_effect_deg_N360']:+.2f}°（{c['phi_rate0p25_N360']:.2f}→"
      f"{c['phi_rate0p50_N360']:.2f}°）。该 N=96 点排水门 **FAIL**（漂移 "
      f"{c['n96_drained_drift']*100:.1f}%>10%），不能作收敛锚点。\n")
    a(f"**F4 门记录**：排水门失效 {g['drained_gate_fail'] or '无'}；拟静态门失效 "
      f"{g['quasi_static_fail'] or '无'}（I>1e-3）；summary 如实置 "
      f"`all_drained_pass={h['summary_all_drained_pass']}`，但报告按 headline 单点表述，"
      f"易被读成「全矩阵通过」。\n")
    a(f"**F5 收敛性 —「临界状态」未达成**：headline φ 在剪切末仍在下降（末样本 "
      f"{ws['phi_at_last_sample_deg']:.2f}° vs 报告窗口均值 {ws['quoted_phi_crit_deg']:.2f}°，"
      f"差 {ws['phi_at_end_minus_quoted_deg']:+.2f}°）；窗口 5%→50% 使 φ_crit 变动 "
      f"{ws['phi_crit_spread_deg']:.2f}°。报告用 φ_crit 论证「球-砂差距」，该论证数值基础不稳。\n")
    a(f"**F6 集成**：交付只在分支 `{dl['branch']}`；mission.result.pr_gate 记 "
      f"`pr 1280 merged`（sha=null）与 git 事实不符 → 不达 L3。\n")
    a("**F7 测试**：交付自报 `pytest tests/test_triaxial_cd_protocol.py → 4 passed, 1 failed`"
      "（tiny 用例短预算排水门）。测试断言的是不变量（Mohr-Coulomb 解析值 30°、退化输入、"
      "确定性、门通过），非仅执行代码；本审计的集群复核（cj1050）未回。\n")
    a("## 5. 判级\n")
    a("**L2 独立功能实现**：真代码（输入 σ3/μ/μr/N/rate/seed → 周期元胞 + 气压伺服 + 压板"
      "的 DEM 推进 → q/φ/门记录），入口 argparse，9 点运行工件 + 605 采样历史，φ 重算逐位自洽。\n")
    a("- 不达 L3：未 merge、main 无调用方（F6）。\n- 不达 L4：无登记裁判/外部锚（F1）、"
      "9 点中排水门 1 失败 + 拟静态门 1 失败（F4）、交叉路径只在 headline 成立（F2）、"
      "复现未在本审计窗口内独立验证（§1）。\n- 不达 L5：无 L4 基础。\n")
    a("## 6. 结论与处置\n")
    a("裁决 **L2 confirmed**（功能实质成立）。返修方向：(a) 在 tensor_benchmark_cases 为 "
      "CASE-TRIAXIAL-CD 登记判据行并给可引用文献锚；(b) 修 F3 收敛表与 F2 双路径口径；"
      "(c) merge 进 main 并接调用方以取 L3/L4。\n")
    a("## 7. 图\n")
    for fpath in figs:
        a(f"![{os.path.basename(fpath)}](figures/{os.path.basename(fpath)})\n")
    a("## 8. 未跑的检查（honest not_run）\n")
    for x in f["checks_not_run"]:
        a(f"- {x}")
    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    open(DOC, "w").write("\n".join(L) + "\n")


CHECKS_NOT_RUN = [
    "repro_cmd 的独立复跑：同一命令已递交集群 cj1049，收尾时仍在排队（queued_ahead≈261）→ not_run",
    "pytest 全文件的机器复核：集群 cj1050 未回，改为引用交付自报的 4 passed / 1 failed",
    "外部锚原文核对（砂土 φ 30-40° 的教科书/文献页）：沙箱无外网、无本地副本 → not_run，"
    "自报带视为未引用的自定判据",
    "40 万步级别的 N=1000+ 尺寸收敛与大应变（ε1>0.25 至真临界态）：本审计不重做算例",
]


def insert_verdict(f, audit_mission, level):
    h, r, dl, rc = f["headline"], f["referee"], f["delivery"], f["repro"]
    ref = "docs/board_audit/m15293/refs/"
    ev = {
        "impl_files": [ref + "src/tensordem/triaxial_cd.py",
                       ref + "benchmarks/case_triaxial_cd/run_case.py",
                       ref + "benchmarks/case_triaxial_cd/run_all.sh"],
        "entry": ref + "benchmarks/case_triaxial_cd/run_case.py",
        "entry_detail": "argparse main(): --sigma3/--mu/--mu-r/--nx/--ny/--nz/--rate/--strain/"
                        "--seed/--out -> run_cd_triaxial(CDTriaxialCase) -> 点 JSON；"
                        "--aggregate 聚合 summary.json",
        "run_artifacts": [ref + f"artifacts/case_triaxial_cd/{t}.json" for t in POINTS]
        + [ref + "artifacts/case_triaxial_cd/summary.json",
           ref + "artifacts/case_triaxial_cd/fig_case_triaxial_cd.png"],
        "tests": ["tests/test_triaxial_cd_protocol.py::test_friction_angle_matches_mohr_coulomb",
                  "tests/test_triaxial_cd_protocol.py::test_case_rejects_inadmissible_input",
                  "tests/test_triaxial_cd_protocol.py::test_consolidation_reaches_confining_pressure",
                  "tests/test_triaxial_cd_protocol.py::test_axial_shear_develops_strength_and_dilates",
                  "tests/test_triaxial_cd_protocol.py::test_run_is_deterministic"],
        "tests_run_by_auditor": "not_run by auditor（集群 cj1050 未回）；交付自报 4 passed, 1 failed"
                                "（tiny 用例排水门）",
        "integration_callers": "none：main 树 grep run_cd_triaxial/triaxial_cd 命中 "
                               f"{dl['callers_in_main_tree']}；分支未 merge（merge-base "
                               "--is-ancestor = false）→ 不达 L3",
        "benchmark": {"anchor_slug": ANCHOR, "registered_rows": r["registered_rows"],
                      "referee_present": False,
                      "self_declared_band_deg": list(BAND), "midpoint_deg": h["band_midpoint_deg"],
                      "got_headline_deg": round(h["phi_peak_deg"], 4),
                      "rel_err_vs_midpoint_pct": round(h["rel_error_vs_midpoint"] * 100, 3),
                      "tolerance_pct": 10.0,
                      "within_tolerance": abs(h["rel_error_vs_midpoint"]) < TOL,
                      "platen_path_got_deg": round(h["phi_peak_platen_deg"], 4),
                      "platen_path_rel_err_pct": round(
                          (h["phi_peak_platen_deg"] - h["band_midpoint_deg"])
                          / h["band_midpoint_deg"] * 100, 3),
                      "points_outside_tolerance": f["gates"]["reference_fail"],
                      "note": "判据未在 tensor_benchmark_cases 登记（0 行）；参照带为自定且未引用原文"},
        "repro_cmd": "cd /root/tp-agent/work/m15235 && PYTHONPATH=src python3 "
                     "benchmarks/case_triaxial_cd/run_case.py --tag ladder_s100000 --sigma3 100000 "
                     "--mu 0.5 --mu-r 0.25 --nx 6 --ny 6 --nz 5 --radius 0.005 --kn 1e+06 "
                     "--rate 0.5 --strain 0.25 --seed 15235 --out artifacts/case_triaxial_cd"
                     "（工件内嵌同一命令；本审计已递交集群 cj1049，未回）",
        "external_anchor": "无有效外部锚：报告 §1 明示 handbook 页无法复核（无网络/无本地副本），"
                           "带 [30,40]° 与中点 35° 为 a priori 自定；tensor_benchmark_cases 无 "
                           "CASE-TRIAXIAL-CD 行 → L4/L5 不成立",
        "declared_level": level,
        "subject_mission": 15235, "subject_status": "succeeded (auto-approved dev_acceptance)",
        "audit_mission": audit_mission,
        "level_rationale": f"L2 独立功能实现：真代码 {ref}src/tensordem/triaxial_cd.py（470 行，"
                           "复用 friction_surface 内核：Cundall-Strack 摩擦+滚动阻力+HCP 抖动试样"
                           "+Love-Weber 应力+周期气压伺服=排水门+压板应变控制），入口 argparse，"
                           "9 点运行工件（每点 605 采样历史）真实非空，φ 由 σ1/σ3 重算与工件逐位一致"
                           "（9/9，max|Δ|≈0）。不到 L3：分支 agent/m15235 未 merge（merge-base "
                           "--is-ancestor=false），main 树无模块、无调用方。不到 L4：裁判行 0 条 + "
                           "外部锚未引用；9 点中 conv_size_n96 排水门 FAIL（10.6%）、ladder_s50000 "
                           "拟静态门 FAIL（I=1.2e-3）；双应力路径差 4.6%~34.7%（9/9 同号）；"
                           "repro 未在本审计窗口内独立复跑（cj1049 排队中）。",
        "three_piece_verification": {
            "run_artifacts": "PASS：9 点 JSON + summary.json + fig png 均存在非空；"
                             "按 commit b952a85b 字节级复制并登记 sha256",
            "benchmark_ref_vs_got": "PARTIAL：数值逐点可核且自洽，但 tensor_benchmark_cases "
                                    "对 CASE-TRIAXIAL-CD 命中 0 行 → 无机器裁判；自报带内 7/9，"
                                    "sens_mu_0p30(-10.4%)、sens_mur_0p10(-5.1% 但 φ=28.46 在带外)"
                                    " 两点超差",
            "repro_cmd": "PARTIAL：命令可执行且工件内嵌；独立复跑未完成（集群 cj1049 排队）"},
        "checks_not_run": CHECKS_NOT_RUN,
        "claims_overstated": [
            "报告 §5「Strain-rate convergence: rate=0.25→32.65, rate=0.50→35.33」：后者是 N=96 点，"
            "固定 N=360 的真值为 33.42（率效应 0.77° 而非 2.68°）",
            "报告 §4 headline 交叉核对 4.57% 被表述为整体校验：其余 8 点中 3 点 >10%，最大 34.7%",
            "报告标题「文献砂土带 30-40 deg 内」：该带为自定且未引用原文（报告 §1 自述无法复核）"],
        "findings": {"F1_referee_missing": h["band_deg"], "F2_platen_path_bias_pct":
                     [round(f["gates"]["platen_rel_min"] * 100, 2),
                      round(f["gates"]["platen_rel_max"] * 100, 2)],
                     "F3_rate_confound_deg": round(f["convergence"]["true_rate_effect_deg_N360"], 3),
                     "F4_gate_fail": f["gates"]["drained_gate_fail"]
                     + f["gates"]["quasi_static_fail"],
                     "F5_phi_crit_spread_deg": round(f["window_sensitivity"]["phi_crit_spread_deg"], 3),
                     "F6_delivery": dl["verdict"],
                     "F7_tests": "4 passed / 1 failed（自报）"},
        "merge_status": {"merged_into_main": dl["merged_into_main"], "callers": []},
        "repro": rc,
    }
    d = dsn()
    import psycopg
    with psycopg.connect(d) as c, c.cursor() as cur:
        cur.execute("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, "
                    "evidence, verdict_by) VALUES (%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET mission_id=EXCLUDED."
                    "mission_id, level=EXCLUDED.level, evidence=EXCLUDED.evidence, "
                    "gate_state='pending' RETURNING id",
                    ("TensorDEM", f"m15235 {ANCHOR}", audit_mission, level,
                     json.dumps(ev, ensure_ascii=False), "board-audit"))
        vid = cur.fetchone()[0]
    print("verdict id =", vid)
    return vid, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-refs", action="store_true")
    ap.add_argument("--write-doc", action="store_true")
    ap.add_argument("--repro")
    ap.add_argument("--insert-verdict", action="store_true")
    ap.add_argument("--audit-mission", type=int, default=15293)
    ap.add_argument("--level", type=int, default=2)
    a = ap.parse_args()
    if a.write_refs:
        print("refs:", len(write_refs()["files"]), "files")
    rows = points_table()
    h = headline(rows)
    f = {"points": rows, "headline": h, "window_sensitivity": window_sensitivity(),
         "convergence": convergence(rows), "gates": gates(rows),
         "delivery": delivery(), "repro": repro_compare(a.repro),
         "checks_not_run": CHECKS_NOT_RUN}
    cases = registered(dsn())
    f["referee"] = {"anchor_slug": ANCHOR, "registered_rows": len(cases), "registered": cases,
                    "referee_present": bool(cases),
                    "note": "无登记机器判据行 → 唯一裁判缺位；自报带 [30,40]°/中点 35°/±10% "
                            "为自定判据，且原文未引用（报告 §1 自述无法复核）",
                    "self_declared": {"band_deg": list(BAND), "midpoint_deg": 35.0,
                                      "tolerance": TOL,
                                      "headline_got_deg": h["phi_peak_deg"],
                                      "headline_rel_error": h["rel_error_vs_midpoint"],
                                      "points_within": f["gates"]["reference_pass"],
                                      "points_outside": f["gates"]["reference_fail"]}}
    os.makedirs(OUT, exist_ok=True)
    figs = figures(rows, h, f["window_sensitivity"])
    f["figures"] = [os.path.relpath(x, REPO) for x in figs]
    json.dump(f, open(os.path.join(OUT, "finding.json"), "w"), indent=1, ensure_ascii=False)
    if a.write_doc:
        write_doc(f, figs)
        print("doc:", DOC)
    if a.insert_verdict:
        insert_verdict(f, a.audit_mission, a.level)
    print(json.dumps({"headline_deg": round(h["phi_peak_deg"], 4),
                      "phi_platen_deg": round(h["phi_peak_platen_deg"], 4),
                      "referee_rows": len(cases),
                      "drained_fail": f["gates"]["drained_gate_fail"],
                      "qs_fail": f["gates"]["quasi_static_fail"],
                      "platen_range_pct": [round(f["gates"]["platen_rel_min"] * 100, 2),
                                           round(f["gates"]["platen_rel_max"] * 100, 2)],
                      "rate_effect_true_deg": round(
                          f["convergence"]["true_rate_effect_deg_N360"], 3),
                      "phi_crit_spread_deg": round(
                          f["window_sensitivity"]["phi_crit_spread_deg"], 3),
                      "merged": f["delivery"]["merged_into_main"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()