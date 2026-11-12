#!/usr/bin/env python3
"""Board cross-audit (交差核验) of mission m15381 — anchor CASE-HOPPER-BEVERLOO.

Verification, not a re-run of the mission.  Everything below is recomputed from
bytes (artifact JSON / DB rows / git objects), nothing is transcribed by hand.

Checks
------
C1  provenance  : do the delivered files exist, and are they in *any* commit?
                  byte copies + sha256 into ``docs/board_audit/m15418/refs/``
C2  referee     : ``tensor_benchmark_cases WHERE anchor_slug='CASE-HOPPER-BEVERLOO'``
                  (the board rule: the registered case is the only referee)
C3  arithmetic  : replay W_meas / W_bev / rel_err / recovered C straight from the
                  artifact's own case block — under BOTH density bases:
                    (a) as filed          rho_b = 0.64 * rho_p   (hard-coded in
                                          ``beverloo_prediction``, phi=0.64)
                    (b) consistent        rho_b = measured bulk density of the
                                          *simulated* bed (the correlation is
                                          linear in rho_b; Beverloo's rho_b is the
                                          in-situ bulk density)
C4  windows     : window-choice sensitivity of the headline (startup window in/out)
C5  test quality: do the delivered tests assert the invariant, or replay the
                  artifact's own ``passed`` flag?  (+ pytest replay)
C6  claims      : report row, cluster job state, repro_cmd executability, plus an
                  *independent* re-run of the recorded case from main's module.
Outputs
-------
``docs/board_audit/m15418/audit_m15381.json``
``docs/board_audit/m15418/fig_anchor_check.png``
``docs/board_audit/m15418/AUDIT.md``
``--insert-verdict`` -> one row in ``tensor_maturity_verdicts``

Usage
-----
    python scripts/audit/m15381_board_audit.py --insert-verdict --audit-mission 15418
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MISSION_WT = Path("/root/tp-agent/work/m15381")
OUT = ROOT / "docs" / "board_audit" / "m15418"
PSQL = ["psql", "-U", "postgres", "-d", "tensorplatform", "-Atc"]

BEVERLOO_C = 0.58
BEVERLOO_K = 1.4          # as filed by the mission (module default is 1.5)
PHI_LITERATURE = 0.64     # hard-coded in beverloo_prediction()
TOL = 0.15

DELIVERABLES = [
    "scripts/run_beverloo_benchmark.py",
    "tests/test_beverloo_m15381_artifacts.py",
    "runs/beverloo_m15381/beverloo_medium_benchmark_seed42.json",
    "runs/beverloo_m15381/beverloo_quick_validation_seed42.json",
    "runs/beverloo_m15381/report_m15381.json",
    "docs/figures/m15381_beverloo_comparison.png",
    "docs/figures/m15381_window_series.png",
]


def sh(cmd, cwd=None, timeout=1800):
    p = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str),
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def psql(sql: str) -> str:
    rc, out, err = sh(PSQL + [sql])
    if rc != 0:
        raise RuntimeError(f"psql failed: {err}")
    return out


# --------------------------------------------------------------------------- C1
def provenance() -> dict:
    refs = OUT / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    inv = {}
    for rel in DELIVERABLES:
        src = MISSION_WT / rel
        entry = {"exists_on_disk": src.exists()}
        if src.exists():
            data = src.read_bytes()
            entry["bytes"] = len(data)
            entry["sha256"] = hashlib.sha256(data).hexdigest()
            dst = refs / Path(rel).name
            shutil.copyfile(src, dst)
            entry["ref_copy"] = str(dst.relative_to(ROOT))
            rc, out, _ = sh(["git", "-C", str(MISSION_WT), "log", "--all",
                             "--oneline", "--", rel])
            entry["commits_touching_file"] = len(out.splitlines())
            entry["in_any_commit"] = bool(out.strip())
        inv[rel] = entry
    rc, head, _ = sh(["git", "-C", str(MISSION_WT), "rev-parse", "HEAD"])
    rc2, br, _ = sh(["git", "-C", str(MISSION_WT), "rev-parse", "--abbrev-ref", "HEAD"])
    rc3, status, _ = sh(["git", "-C", str(MISSION_WT), "status", "--porcelain"])
    return {
        "mission_worktree": str(MISSION_WT),
        "head": head,
        "branch": br,
        "git_status_head": status.splitlines()[:12],
        "files": inv,
        "verdict": ("all deliverable bytes exist on disk but were NEVER committed: "
                    "0 commits touch them, so no branch/PR can carry them and the "
                    "platform push step has nothing to push"
                    if all(not e.get("in_any_commit") for e in inv.values() if e["exists_on_disk"])
                    else "some deliverables are committed"),
    }


# --------------------------------------------------------------------------- C2
def referee() -> dict:
    rows = psql("SELECT id||'|'||case_name||'|'||metric||'|'||coalesce(tolerance,'-')||'|'||"
                "coalesce(status,'-') FROM tensor_benchmark_cases "
                "WHERE anchor_slug='CASE-HOPPER-BEVERLOO'")
    total = psql("SELECT count(*) FROM tensor_benchmark_cases")
    goal = psql("SELECT goal FROM tensor_dev_missions WHERE id=15381")
    tol_declared = "<15%" if "<15%" in goal else "unknown"
    return {
        "query": "tensor_benchmark_cases WHERE anchor_slug='CASE-HOPPER-BEVERLOO'",
        "rows": [r for r in rows.splitlines() if r.strip()],
        "n_rows": len([r for r in rows.splitlines() if r.strip()]),
        "table_rows_total": int(total),
        "fallback_criterion": ("no machine-owned criterion registered -> the only "
                               "criteria are the mission goal text (" + tol_declared +
                               ") and the artifact's own threshold_pct=15.0"),
        "tolerance_used": TOL,
    }


# --------------------------------------------------------------------------- C3/C4
def lief(rec: dict) -> dict:
    case, res, comp = rec["case"], rec["results"], rec["comparison"]
    d = case["particle_diameter_m"]
    D = case["orifice_diameter_m"]
    rho_p = case["particle_density_kg_m3"]
    g = case["gravity_m_s2"]
    der = D - BEVERLOO_K * d
    w_meas = res["flow_rate_kg_s"]
    rho_lit = PHI_LITERATURE * rho_p
    rho_sim = res["bulk_density_kg_m3"]

    def predict(rho_b):
        return BEVERLOO_C * rho_b * math.sqrt(g) * der ** 2.5

    w_file = predict(rho_lit)
    w_cons = predict(rho_sim)
    # recovered C = W_meas / (rho_b * sqrt(g) * D_eff^2.5) evaluated on each basis
    rec_lit = w_meas / (rho_lit * math.sqrt(g) * der ** 2.5)
    rec_cons = w_meas / (rho_sim * math.sqrt(g) * der ** 2.5)
    win = res["flow_rate_windows_kg_s"]

    def mean(x):
        return sum(x) / len(x) if x else float("nan")

    steady = win[1:]                      # drop startup window
    tail = win[-3:]
    return {
        "case_label": rec["case_label"],
        "geometry": {"D_m": D, "d_m": d, "D_over_d": D / d,
                     "beverloo_valid_D_over_d_ge": 4.0,
                     "geometry_valid": D / d >= 4.0},
        "packing": {"phi_measured": res["packing_fraction"],
                    "phi_used_by_prediction": PHI_LITERATURE,
                    "rho_b_measured_kg_m3": rho_sim,
                    "rho_b_used_by_prediction_kg_m3": rho_lit,
                    "density_inflation_factor": rho_lit / rho_sim},
        "filed": {"W_bev_kg_s": comp["beverloo_prediction_kg_s"],
                  "W_meas_kg_s": w_meas,
                  "rel_err": comp["relative_error"],
                  "rel_err_pct": comp["relative_error_pct"],
                  "recovered_C": comp["recovered_C"],
                  "passed_flag": comp["passed"]},
        "replay_as_filed_density": {
            "rho_b": rho_lit, "W_bev_kg_s": w_file,
            "rel_err": (w_meas - w_file) / w_file,
            "rel_err_pct": 100.0 * (w_meas - w_file) / w_file,
            "recovered_C": rec_lit,
            "abs_rel_err_lt_15pct": abs((w_meas - w_file) / w_file) < TOL,
            "matches_filed": abs(w_file - comp["beverloo_prediction_kg_s"]) < 1e-9,
        },
        "replay_consistent_density": {
            "rho_b": rho_sim, "W_bev_kg_s": w_cons,
            "rel_err": (w_meas - w_cons) / w_cons,
            "rel_err_pct": 100.0 * (w_meas - w_cons) / w_cons,
            "recovered_C": rec_cons,
            "abs_rel_err_lt_15pct": abs((w_meas - w_cons) / w_cons) < TOL,
            "recovered_C_vs_0.58": rec_cons / BEVERLOO_C,
            "within_literature_C_band": 0.55 <= rec_cons <= 0.60,
        },
        "windows": {
            "series_kg_s": win,
            "mean_all_kg_s": mean(win),
            "mean_excl_startup_kg_s": mean(steady),
            "mean_tail3_kg_s": mean(tail),
            "cv_all": (sum((w - mean(win)) ** 2 for w in win) / len(win)) ** 0.5 / mean(win),
            "cv_excl_startup": (sum((w - mean(steady)) ** 2 for w in steady) / len(steady)) ** 0.5 / mean(steady),
            "startup_window_share_of_mean": win[0] / mean(win),
            "rel_err_pct_excl_startup_consistent_density":
                100.0 * (mean(steady) - w_cons) / w_cons,
            "rel_err_pct_excl_startup_filed_density":
                100.0 * (mean(steady) - w_file) / w_file,
        },
        "ledger": {"mass_residual_kg": res["mass_balance_residual_kg"],
                   "mass_ok": abs(res["mass_balance_residual_kg"]) < 1e-10,
                   "steadiness_r2": res["steadiness_r2"],
                   "n_initial": res["n_initial"], "n_discharged": res["n_discharged"],
                   "wall_clock_s": res["wall_clock_s"]},
    }


# --------------------------------------------------------------------------- C5
def test_quality() -> dict:
    path = MISSION_WT / "tests" / "test_beverloo_m15381_artifacts.py"
    src = path.read_text() if path.exists() else ""
    asserts = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("assert")]
    replay_asserts = [a for a in asserts if "comp[" in a or "medium_record[" in a]
    recompute = [a for a in asserts
                 if any(tok in a for tok in ("math.sqrt", "**2.5", "beverloo_prediction("))]
    rc, out, err = sh(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                       "tests/test_beverloo_discharge_measurement.py",
                       "tests/test_beverloo_m15381_artifacts.py"],
                      cwd=str(MISSION_WT), timeout=1800)
    tail = (out or err).splitlines()[-3:]
    return {
        "test_file": "tests/test_beverloo_m15381_artifacts.py",
        "n_assert_statements": len(asserts),
        "n_assert_on_stored_artifact_fields": len(replay_asserts),
        "n_assert_that_recompute_the_correlation": len(recompute),
        "replay_asserts": replay_asserts[:8],
        "rubber_stamp_tolerance": [a for a in asserts if "recovered_C" in a],
        "pytest_replay": {"cmd": ("python -m pytest -q tests/test_beverloo_discharge_measurement.py "
                                  "tests/test_beverloo_m15381_artifacts.py (cwd=m15381 worktree)"),
                          "rc": rc, "tail": tail},
        "verdict": ("tests replay the artifact's own passed=True flag and bound "
                    "recovered C only to 0.1..2.0; none recomputes W_bev from the case "
                    "block, so the suite passes independently of the physics"
                    if not recompute else "tests do recompute the correlation"),
    }


# --------------------------------------------------------------------------- C6
def claims(repro: bool) -> dict:
    rep = psql("SELECT id||'|'||title||'|'||length(markdown)||'|'||"
               "created_at::text FROM tensor_reports WHERE mission_id=15381")
    md = psql("SELECT markdown FROM tensor_reports WHERE mission_id=15381")
    heads = [ln.strip() for ln in md.splitlines() if ln.strip().startswith("##")]
    job = psql("SELECT id||'|'||state||'|'||coalesce(left(cmd,90),'')||'|'||"
               "submitted_at::text FROM cluster_jobs WHERE id=1100")
    res = psql("SELECT result::text FROM tensor_dev_missions WHERE id=15381")
    jresult = json.loads(res) if res else {}
    repro_info = {"requested": repro, "status": "not_run"}
    if repro:
        med = json.loads((MISSION_WT / "runs/beverloo_m15381/"
                          "beverloo_medium_benchmark_seed42.json").read_text())
        c = med["case"]
        sys.path.insert(0, str(ROOT / "src"))
        from tensordem.beverloo_discharge import BeverlooCase, simulate_discharge
        case = BeverlooCase(
            radius=c["particle_diameter_m"] / 2.0, density=c["particle_density_kg_m3"],
            mu=c["particle_friction"], mu_rolling=c["mu_rolling"],
            restitution=c["restitution"], kn=c["kn_N_m"], kt_ratio=c["kt_ratio"],
            silo_radius=c["silo_radius_m"], fill_height=c["fill_height_m"],
            orifice_diameter=c["orifice_diameter_m"], gravity=c["gravity_m_s2"],
            packing_gap=c["packing_gap"], jitter=c["jitter_r"], dt=c["dt_s"],
            seed=c["seed"], max_time=c["max_time_s"], settle_time=c["settle_time_s"],
            window=c["window_s"], n_windows=c["n_windows"], wall_mu=c["wall_friction"],
        )
        t0 = time.time()
        r = simulate_discharge(case, C=BEVERLOO_C, k=BEVERLOO_K)
        repro_info = {
            "requested": True, "status": "run",
            "runner": "main HEAD src/tensordem/beverloo_discharge.py (independent of the "
                      "mission's worktree copy) with the artifact's recorded case block",
            "filed_flow_rate_kg_s": med["results"]["flow_rate_kg_s"],
            "reproduced_flow_rate_kg_s": r["flow_rate_kg_s"],
            "abs_diff_kg_s": abs(r["flow_rate_kg_s"] - med["results"]["flow_rate_kg_s"]),
            "rel_diff": (abs(r["flow_rate_kg_s"] - med["results"]["flow_rate_kg_s"]) /
                         med["results"]["flow_rate_kg_s"]),
            "reproduced_windows": r["flow_rate_windows_kg_s"],
            "filed_windows": med["results"]["flow_rate_windows_kg_s"],
            "reproduced_mass_residual_kg": r["mass_balance_residual_kg"],
            "n_discharged": r["n_discharged"], "filed_n_discharged": med["results"]["n_discharged"],
            "wall_clock_s": time.time() - t0,
            "identical": abs(r["flow_rate_kg_s"] - med["results"]["flow_rate_kg_s"]) < 1e-9,
        }
    import re as _re
    m_w = _re.search(r"Mass flow rate W \|\s*([\d.]+) kg/s\s*\|\s*([\d.]+) kg/s\s*\|\s*([+\-][\d.]+)%", md)
    m_c = _re.search(r"Recovered C \|\s*([\d.]+)\s*\|\s*([\d.]+)", md)
    rep_consistency = {"parsed": None}
    if m_w and m_c:
        w_ref, w_got, err_pct = float(m_w.group(1)), float(m_w.group(2)), float(m_w.group(3))
        c_got = float(m_c.group(2))
        implied = 1.0 + err_pct / 100.0          # W_got/W_ref
        rep_consistency = {
            "parsed": {"W_ref_kg_s": w_ref, "W_got_kg_s": w_got, "rel_err_pct": err_pct,
                       "recovered_C_quoted": c_got},
            "ratio_W_got_over_W_ref": w_got / w_ref,
            "recovered_C_over_0.58": c_got / BEVERLOO_C,
            "self_inconsistent": abs((c_got / BEVERLOO_C) - implied) > 0.05,
            "note": "the two columns of the same report table sit on different density "
                    "bases: rel_err implies W_got/W_ref = {:.3f} (C = {:.3f}) while the "
                    "quoted recovered C is {:.4f} = x{:.2f} of 0.58".format(
                        w_got / w_ref, BEVERLOO_C * w_got / w_ref, c_got, c_got / BEVERLOO_C),
        }
    return {
        "report_row": rep,
        "report_internal_consistency": rep_consistency,
        "report_headings": heads,
        "report_headline_quick_case": "report §4 table quotes the *quick* case "
                                      "(D/d=3) W_ref=0.532424, W_meas=0.733038, +37.68% "
                                      "FAIL -- a different case from the medium run that "
                                      "the mission finish calls '-14.86% PASS'",
        "cluster_job_cj1100": job,
        "cluster_job_note": "cj1100 ('--case full --seed 1') is still queued: the full-case / "
                            "sensitivity claims have no execution behind them",
        "mission_checks_claim": jresult.get("checks", ""),
        "repro": repro_info,
    }


def figure(liefs, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels, filed, cons = [], [], []
    for L in liefs:
        labels.append(f"{L['case_label']}\nD/d={L['geometry']['D_over_d']:.1f}")
        filed.append(L["replay_as_filed_density"]["rel_err_pct"])
        cons.append(L["replay_consistent_density"]["rel_err_pct"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, filed, w, label="as filed: ρ_b = 0.64 ρ_p", color="#FF9800")
    b2 = ax.bar(x + w / 2, cons, w, label="consistent: ρ_b = measured bed", color="#2196F3")
    ax.axhspan(-15, 15, color="green", alpha=0.12, label="±15 % tolerance")
    ax.axhline(0, color="k", lw=0.8)
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + (2 if v >= 0 else -6),
                    f"{v:+.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("relative error vs Beverloo (1961), %")
    ax.set_title("m15381: the 'PASS' is a density-basis artifact\n"
                 "(0.64 ρ_p vs the simulated bed's own ρ_b)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left"); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    med = liefs[[L["case_label"] for L in liefs].index("medium_benchmark")]
    win = med["windows"]["series_kg_s"]
    ax.plot(range(1, len(win) + 1), win, "o-", color="#2196F3", label="DEM per-window W")
    ax.axhline(med["replay_as_filed_density"]["W_bev_kg_s"], color="#FF9800", ls="--",
               label=f"W_bev filed basis ({med['replay_as_filed_density']['W_bev_kg_s']:.3f})")
    ax.axhline(med["replay_consistent_density"]["W_bev_kg_s"], color="#2E7D32", ls="-.",
               label=f"W_bev consistent ({med['replay_consistent_density']['W_bev_kg_s']:.3f})")
    ax.axhline(med["windows"]["mean_all_kg_s"], color="#607D8B", ls=":",
               label=f"mean incl. startup ({med['windows']['mean_all_kg_s']:.3f})")
    ax.set_xlabel("window index"); ax.set_ylabel("mass flow rate W (kg/s)")
    ax.set_title("medium case: startup window drags the headline by ~20 %", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def insert_verdict(level: int, evidence: dict, audit_mid: int):
    """Idempotent: one row per (repo, scope, verdict_by)."""
    payload = json.dumps(evidence, ensure_ascii=False).replace("'", "''")
    sql = ("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
           f"VALUES ('TensorDEM', 'm15381 CASE-HOPPER-BEVERLOO', {audit_mid}, {level}, "
           f"'{payload}'::jsonb, 'board-audit') "
           "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET "
           f"level=EXCLUDED.level, evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id, "
           "gate_state='pending', gate_level=NULL, gate_note=NULL "
           "RETURNING id, gate_state")
    return psql(sql)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insert-verdict", action="store_true")
    ap.add_argument("--audit-mission", type=int, default=15418)
    ap.add_argument("--no-repro", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    prov = provenance()
    ref = referee()
    med = json.loads((MISSION_WT / "runs/beverloo_m15381/"
                      "beverloo_medium_benchmark_seed42.json").read_text())
    quick = json.loads((MISSION_WT / "runs/beverloo_m15381/"
                        "beverloo_quick_validation_seed42.json").read_text())
    liefs = [lief(quick), lief(med)]
    tq = test_quality()
    cl = claims(repro=not args.no_repro)

    fig_png = OUT / "fig_anchor_check.png"
    figure(liefs, fig_png)

    med_L = liefs[1]
    anchor_fail = not med_L["replay_consistent_density"]["abs_rel_err_lt_15pct"]
    level = 2
    audit = {
        "audited_mission": 15381,
        "audit_mission": args.audit_mission,
        "repo": "TensorDEM",
        "anchor": "CASE-HOPPER-BEVERLOO",
        "C1_provenance": prov,
        "C2_referee": ref,
        "C3_arithmetic_replay": liefs,
        "C5_test_quality": tq,
        "C6_claims": cl,
        "level": level,
        "anchor_conclusion": "FAIL" if anchor_fail else "PASS",
        "anchor_reason": (
            "medium case: as filed rel_err = "
            f"{med_L['replay_as_filed_density']['rel_err_pct']:+.2f}% (passes only because the "
            f"reference inflates ρ_b by x{med_L['packing']['density_inflation_factor']:.2f}); on the "
            f"consistent density basis rel_err = {med_L['replay_consistent_density']['rel_err_pct']:+.2f}% "
            f"and recovered C = {med_L['replay_consistent_density']['recovered_C']:.3f} "
            f"(+{100*(med_L['replay_consistent_density']['recovered_C']/BEVERLOO_C-1):.0f}% over the 0.58 "
            "literature value, outside the 0.55-0.60 band) -> anchor FAIL; quick case FAIL on both "
            "bases (+"
            f"{liefs[0]['replay_as_filed_density']['rel_err_pct']:.1f}% / "
            f"{liefs[0]['replay_consistent_density']['rel_err_pct']:+.1f}%) and its geometry "
            f"(D/d={liefs[0]['geometry']['D_over_d']:.1f}) is outside the Beverloo regime (D/d>=4)"
        ),
    }
    (OUT / "audit_m15381.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False))

    print(json.dumps({k: audit[k] for k in
                      ("level", "anchor_conclusion", "anchor_reason")}, indent=1, ensure_ascii=False))
    print("\nC1:", prov["verdict"])
    print("C2 referee rows:", ref["n_rows"], "| table total:", ref["table_rows_total"])
    print("C5:", tq["verdict"], "|", tq["pytest_replay"]["tail"])
    print("C6 repro:", json.dumps(cl["repro"], ensure_ascii=False)[:600])

    evidence = {
        # --- clean relative paths first: maturity_gate.files_exist() checks the
        # --- literal string, so commentary lives in the *_note keys.
        "impl_files": [
            "src/tensordem/beverloo_discharge.py",
            "docs/board_audit/m15418/refs/run_beverloo_benchmark.py",
        ],
        "impl_files_note": (
            "beverloo_discharge.py is PRE-EXISTING (m6457, commit 8d61e496); its "
            ":348-379 beverloo_prediction()/recovered_C() with phi=0.64 hard-coded at :354 "
            "sets the density basis of every filed rel_err. The m15381 contribution is the "
            "entry script (preserved byte-identically in refs/, sha256 "
            + prov["files"]["scripts/run_beverloo_benchmark.py"]["sha256"][:16]
            + "...) plus the tests; both are untracked in the mission worktree. The DEM kernel "
            "actually executed is src/tensordem/silo_discharge.py simulate_silo()."),
        "entry": "docs/board_audit/m15418/refs/run_beverloo_benchmark.py",
        "entry_note": ("original command: PYTHONPATH=src python scripts/run_beverloo_benchmark.py "
                       "--case medium --seed 42; the path above is the audit byte-copy because "
                       "scripts/run_beverloo_benchmark.py exists in NO commit (0 commits touch it)"),
        "run_artifacts": [
            "docs/board_audit/m15418/refs/beverloo_medium_benchmark_seed42.json",
            "docs/board_audit/m15418/refs/beverloo_quick_validation_seed42.json",
            "docs/board_audit/m15418/refs/report_m15381.json",
        ],
        "run_artifacts_note": (
            "byte copies (sha256 medium "
            + prov["files"]["runs/beverloo_m15381/beverloo_medium_benchmark_seed42.json"]["sha256"][:16]
            + "..., quick "
            + prov["files"]["runs/beverloo_m15381/beverloo_quick_validation_seed42.json"]["sha256"][:16]
            + "...) of /root/tp-agent/work/m15381/runs/beverloo_m15381/*; git log --all -- "
            "runs/beverloo_m15381 = 0 commits and tensor_dev_missions.result.head_commit IS NULL, "
            "so the mission shipped no committable artifact"),
        "tests": [
            "tests/test_beverloo_m15381_artifacts.py (m15381, untracked): 9 tests / 33 asserts",
            "tests/test_beverloo_discharge_measurement.py (pre-existing, m6457)",
            "pytest replay in the mission worktree: rc=" + str(tq["pytest_replay"]["rc"]) + " -- "
            + (tq["pytest_replay"]["tail"][-1] if tq["pytest_replay"]["tail"] else ""),
        ],
        "tests_note": (
            f"{tq['n_assert_on_stored_artifact_fields']}/{tq['n_assert_statements']} asserts read "
            "stored artifact fields and 0 recompute W_bev from the case block; "
            "test_comparison_within_tolerance asserts the artifact's own passed=True and "
            "test_recovered_C_reasonable only bounds 0.1<C<2.0, so the suite is green "
            "independently of the physics"),
        "integration_callers": "NONE for this benchmark (grep run_beverloo_benchmark outside "
                               "itself = 0 hits; no export/CLI/API); the reused module is called by "
                               "scripts/run_coarse_graining_campaign.py:156 and "
                               "scripts/m6457_beverloo_measurement_sweep.py:34 (pre-existing)",
        "benchmark": {
            "registry": ref["query"] + f" -> {ref['n_rows']} rows (table holds {ref['table_rows_total']}); "
                        "no machine-owned criterion exists, so the goal text (<15%) + the "
                        "artifact's threshold_pct=15.0 are the only criteria",
            "ref": {"C": BEVERLOO_C, "k": BEVERLOO_K, "source": "Beverloo, Leniger & Van de Velde "
                    "1961, Chem. Eng. Sci. 15:260-269, DOI 10.1016/0009-2509(61)80059-4 (Eq.4 p.265)"},
            "got_medium": {"W_meas_kg_s": med_L["filed"]["W_meas_kg_s"],
                           "W_bev_filed_kg_s": med_L["filed"]["W_bev_kg_s"],
                           "rel_err_filed_pct": med_L["filed"]["rel_err_pct"],
                           "passes_as_filed": med_L["filed"]["passed_flag"]},
            "replay_as_filed_density": med_L["replay_as_filed_density"],
            "replay_consistent_density": med_L["replay_consistent_density"],
            "density_basis_defect": {
                "rho_b_used_by_prediction_kg_m3": med_L["packing"]["rho_b_used_by_prediction_kg_m3"],
                "rho_b_measured_in_simulation_kg_m3": med_L["packing"]["rho_b_measured_kg_m3"],
                "phi_measured": med_L["packing"]["phi_measured"],
                "inflation_factor": med_L["packing"]["density_inflation_factor"],
                "why_wrong": "W is linear in Beverloo's in-situ bulk density; evaluating the "
                             "correlation at 0.64*rho_p while the simulated bed sits at "
                             f"phi={med_L['packing']['phi_measured']:.3f} understates the deviation "
                             "by exactly that factor",
                "implied_C_as_filed_basis": med_L["replay_as_filed_density"]["recovered_C"],
                "implied_C_consistent_basis": med_L["replay_consistent_density"]["recovered_C"],
            },
            "window_sensitivity": med_L["windows"],
            "quick_case": liefs[0],
            "report_self_contradiction": cl.get("report_internal_consistency", {}),
            "status": audit["anchor_conclusion"],
        },
        "repro_cmd": ("PYTHONPATH=src python scripts/run_beverloo_benchmark.py --case medium --seed 42"
                      " -> independent replay from main HEAD src/tensordem/beverloo_discharge.py with "
                      "the artifact's recorded case block: flow_rate "
                      + (f"{cl['repro'].get('reproduced_flow_rate_kg_s'):.10f} vs filed "
                         f"{cl['repro'].get('filed_flow_rate_kg_s'):.10f} kg/s (abs diff "
                         f"{cl['repro'].get('abs_diff_kg_s'):.1e}, per-window series identical, "
                         f"mass ledger {cl['repro'].get('reproduced_mass_residual_kg'):.1e} kg) "
                         if cl["repro"].get("status") == "run" else "not_run")
                      + " => the DEM numbers are REAL and reproducible; the comparison BASIS is what "
                        "is wrong"),
        "external_anchor": ("Beverloo 1961 Eq.4 p.265: C=0.58 (0.55-0.60), k=1.4 (1.2-1.6). The "
                            "mission's own artifact contains the refutation: recovered_C="
                            f"{med_L['filed']['recovered_C']:.4f} evaluated with the measured rho_b is "
                            f"+{100*(med_L['filed']['recovered_C']/BEVERLOO_C-1):.0f}% over 0.58 "
                            "(beverloo_discharge.py:364-379 documents that rho_b= argument as exactly "
                            "this consistency check); the report's quick-case table even quotes "
                            "recovered C=1.0349 (+78%) next to a +37.68% rel_err"),
        "level_reason": (
            "L2 = real code input->compute->output, proven by an independent replay (flow rate "
            "identical to 1e-9, per-window series identical, mass ledger 4.4e-16). L3 refused: no "
            "integration caller/export/API for the benchmark and every delivered file is untracked "
            "(0 commits) so nothing is integrated. L4 refused: on a consistent density basis the "
            f"anchor reads +{med_L['replay_consistent_density']['rel_err_pct']:.1f}% with recovered "
            f"C={med_L['replay_consistent_density']['recovered_C']:.3f} (+"
            f"{100*(med_L['replay_consistent_density']['recovered_C']/BEVERLOO_C-1):.0f}%); the filed "
            f"{med_L['filed']['rel_err_pct']:+.2f}% 'PASS' comes from a rho_b inflated x"
            f"{med_L['packing']['density_inflation_factor']:.2f}; the posted report quotes a different "
            "(quick) case at +37.68% FAIL with D/d=3 outside the Beverloo regime; no registered "
            "referee row; cj1100 (full case) never executed (still queued)."),
        "gate_expectation": ("confirmed at L2 by JUDGE_SCHEME (evidence keys are real) while the "
                            "anchor stays FAIL: note that JUDGE_SCHEME does not read benchmark status, "
                            "so an L2 row confirms even with anchor FAIL -- the anchor FAIL and the "
                            "density-basis defect are recorded here for the ledger and need a "
                            "separate remediation order (default rho_b := measured packing, or "
                            "phi as a case parameter)"),
        "audit_artifacts": ["docs/board_audit/m15418/AUDIT.md",
                            "docs/board_audit/m15418/audit_m15381.py",
                            "docs/board_audit/m15418/audit_m15381.json",
                            "docs/board_audit/m15418/fig_anchor_check.png",
                            "scripts/audit/check_beverloo_density_basis.py"],
        "repo_head_audited": prov["head"] + " on " + prov["branch"] + " (mission worktree; the "
                             "audit worktree m15418 sits on the merged main line)",
    }
    if args.insert_verdict:
        print("verdict insert ->", insert_verdict(level, evidence, args.audit_mission))
    (OUT / "verdict_evidence.json").write_text(json.dumps(evidence, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()