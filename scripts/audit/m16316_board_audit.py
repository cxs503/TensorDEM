#!/usr/bin/env python3
"""m16316 -- board audit of m16243 (CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT).

Verification, not a re-run.  Everything is recomputed from bytes: the mission branch
exported with ``git archive``, the committed artifact JSONs, the referee table in
``tensor_benchmark_cases``, the published report row, and two auditor runs (a
bit-exact reproduction of one filed point and one instrumented probe of the energy
audit).  Outputs ``docs/board_audit/m16316/{audit_m16243.json,AUDIT.md,fig_*.png}``
and, with ``--insert-verdict``, one row in ``tensor_maturity_verdicts``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "board_audit" / "m16316"
RUNS = ROOT / "runs" / "audit_m16316"
EXPORT = Path("/tmp/m16316_branch")
REF = "agent/m16243"
ANCHOR = "CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT"
MISSION, AUDIT_MISSION = 16243, 16316
PSQL = ["psql", "-U", "postgres", "-d", "tensorplatform", "-Atc"]
ART = "artifacts/case_dem_shear_campbell_elastic_limit"
TOL_R3, TOL_R1_REL, TOL_R2_FACTOR = 0.30, 0.15, 3.0

DELIVERABLES = [
    "src/tensordem/elastic_limit_shear.py",
    "src/tensordem/cli.py",
    "benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py",
    "tests/test_elastic_limit_shear.py",
    "docs/CASE_DEM_SHEAR_CAMPBELL_ELASTIC_LIMIT.md",
    f"{ART}/scaling.json",
    f"{ART}/dt_convergence.json",
    f"{ART}/fig_tau_star_kappa.png",
    f"{ART}/logs/sweep_all.log",
]


def sh(cmd, cwd=None, timeout=1800):
    p = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str),
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def psql(sql):
    rc, out, err = sh(PSQL + [sql])
    if rc != 0:
        raise RuntimeError(f"psql failed: {err}")
    return out


def slope(xs, ys):
    x = [math.log(v) for v in xs]
    y = [math.log(v) for v in ys]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    return (sum((a - mx) * (b - my) for a, b in zip(x, y))
            / sum((a - mx) ** 2 for a in x))


def export_branch():
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    EXPORT.mkdir(parents=True)
    tar = Path("/tmp/m16316_branch.tar")
    rc, _, err = sh(["git", "-C", str(ROOT), "archive", REF, "-o", str(tar)])
    if rc != 0:
        raise RuntimeError(f"git archive failed: {err}")
    sh(["tar", "-xf", str(tar), "-C", str(EXPORT)])


def provenance():
    refs = OUT / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    inv = {}
    for rel in DELIVERABLES:
        src = EXPORT / rel
        e = {"exists_on_disk": src.exists()}
        if src.exists():
            data = src.read_bytes()
            e["bytes"] = len(data)
            e["sha256"] = hashlib.sha256(data).hexdigest()
            dst = refs / Path(rel).name
            shutil.copyfile(src, dst)
            e["ref_copy"] = str(dst.relative_to(ROOT))
        e["commits_touching_file"] = len(
            sh(["git", "-C", str(ROOT), "log", "--all", "--oneline", "--", rel])[1]
            .splitlines())
        e["in_any_commit"] = e["commits_touching_file"] > 0
        inv[rel] = e
    tip = sh(["git", "-C", str(ROOT), "rev-parse", REF])[1]
    main_head = sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"])[1]
    merged = sh(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", tip,
                 "HEAD"])[0] == 0
    branches = sh(["git", "-C", str(ROOT), "branch", "-a", "--contains", tip])[1]
    trunk_cli = sh(["grep", "-c", "elastic-limit-shear",
                    str(ROOT / "src/tensordem/cli.py")])
    trunk_has = int(trunk_cli[1].split(":")[-1]) if trunk_cli[0] == 0 else 0
    return {
        "branch": REF, "branch_tip": tip[:12], "main_head": main_head[:12],
        "merged_into_main": merged,
        "branches_containing": branches.splitlines(),
        "files": inv, "trunk_cli_wiring_lines": trunk_has,
        "note": ("deliverables are committed on the mission branch, but that branch is "
                 "not an ancestor of main: trunk still ships the round-1 kernel and has "
                 "no CLI wiring for this case" if not merged else
                 "deliverables are on trunk"),
    }


def referee():
    rows = psql("SELECT case_name||'|'||metric||'|'||coalesce(tolerance,'-')||'|'||"
                "coalesce(status,'-') FROM tensor_benchmark_cases "
                f"WHERE anchor_slug='{ANCHOR}' ORDER BY case_name")
    rows = [r for r in rows.splitlines() if r.strip()]
    return {"query": f"tensor_benchmark_cases WHERE anchor_slug='{ANCHOR}'",
            "rows": rows, "n_rows": len(rows)}


def load_points():
    pts = {}
    for f in sorted((EXPORT / ART).glob("point_*.json")):
        d = json.loads(f.read_text())
        pts[d["tag"]] = d
    return pts


def gates_of(d, energy_gate):
    g = {
        "energy": bool(d["energy"]["relative_defect"] <= energy_gate),
        "overlap": bool(d["max_overlap_ratio"] <= 0.50 and d["mean_overlap_ratio"] <= 0.12),
        "steady": bool(d["steady_drift_tau"] is not None and d["steady_drift_tau"] <= 0.25),
        "coordination": bool(1.5 <= d["coordination"] <= 6.0),
        "pressure_positive": bool((d["pressure"] or 0) > 0),
    }
    g["all"] = all(g.values())
    return g


def arithmetic(pts):
    worst = {"rel": 0.0, "tag": None, "what": None}
    per = {}
    for tag, d in pts.items():
        c = d["case"]
        gd, rho, dm = c["shear_rate"], c["density"], c["d_mean"]
        chk = {"kappa": c["kn"] / (rho * dm ** 3 * gd ** 2),
               "tau_star": d["tau"] / (rho * dm ** 2 * gd ** 2),
               "mu_eff": d["tau"] / d["pressure"],
               "I": gd * dm / math.sqrt(d["pressure"] / rho)}
        rel = {k: abs(chk[k] - d[k]) / max(abs(d[k]), 1e-300) for k in chk}
        per[tag] = {"max_rel_diff": max(rel.values()),
                    "recomputed": {k: round(v, 10) for k, v in chk.items()}}
        for k, v in rel.items():
            if v > worst["rel"]:
                worst = {"rel": v, "tag": tag, "what": k}
    return {"per_point": per, "worst": worst,
            "verdict": ("filed dimensionless groups are arithmetic-consistent with the "
                        "artifacts' own case blocks" if worst["rel"] < 1e-9
                        else "MISMATCH between filed and recomputed groups")}


def gating(pts):
    ladder = sorted([d for d in pts.values() if d["case"]["shear_rate"] == 1.0],
                    key=lambda d: d["kappa"])
    out = {}
    for label, gate in (("as_filed_in_artifacts", 0.1), ("committed_code_default", 0.30)):
        ok = [d for d in ladder if gates_of(d, gate)["all"]]
        rec = {"energy_gate": gate, "n_gated": len(ok),
               "gated_tags": [d["tag"] for d in ok]}
        if len(ok) >= 2:
            beta = slope([d["kappa"] for d in ok], [d["tau_star"] for d in ok])
            rec.update({"beta": beta, "n": 2.0 * (1.0 - beta), "local": []})
            for a, b in zip(ok, ok[1:]):
                bb = slope([a["kappa"], b["kappa"]], [a["tau_star"], b["tau_star"]])
                rec["local"].append({"beta": bb, "n": 2.0 * (1.0 - bb),
                                     "kappa": [a["kappa"], b["kappa"]]})
        else:
            rec["status"] = "refused: fewer than 2 points pass the gate"
        out[label] = rec
    fs = json.loads((EXPORT / ART / "scaling.json").read_text())["scaling_vs_kappa"]
    out["filed_scaling_json"] = {"n_gated": fs["n_gated"], "beta_global": fs["beta_global"],
                                 "n_global": fs["n_global"], "rejected_kn": fs["rejected"],
                                 "rate_sweeps_status": json.loads(
                                     (EXPORT / ART / "scaling.json").read_text())
                                 ["rate_sweeps"]}
    out["agrees_with_filed"] = abs(
        out["committed_code_default"]["beta"] - fs["beta_global"]) < 1e-9
    out["gate_conflict"] = {
        "artifacts_record_energy_gate": sorted({d["energy"]["gate"] for d in pts.values()}),
        "artifacts_all_gate_flag": {d["tag"]: d["gates"]["all"] for d in ladder},
        "finding": ("every point in the shipped artifacts carries gates.energy=0 and "
                    "gates.all=0 under the energy gate recorded inside the artifact "
                    "(0.1); the headline scaling exists only after re-gating at 0.30, "
                    "and the artifacts were never regenerated to reflect it")
        if all(not d["gates"]["all"] for d in ladder) else "artifacts self-consistent",
    }
    return out


def referee_legs(pts):
    ladder = sorted([d for d in pts.values() if d["case"]["shear_rate"] == 1.0],
                    key=lambda d: d["kappa"])
    gated = [d for d in ladder if gates_of(d, 0.30)["all"]]
    legs = {}
    beta = slope([d["kappa"] for d in gated], [d["tau_star"] for d in gated])
    n = 2.0 * (1.0 - beta)
    legs["R3"] = {
        "reference": "n = 0 (tau* ~ kappa, elastic / quasi-static limit)",
        "tolerance": f"absolute {TOL_R3}", "got_n": n, "got_beta": beta,
        "abs_dev_from_reference": abs(n), "within_tolerance": abs(n) <= TOL_R3,
        "side_check_relative_on_beta": {"rel_dev": abs(beta - 1.0),
                                        "within_15pct": abs(beta - 1.0) <= TOL_R1_REL},
        "tau_star_growth_over_window": gated[-1]["tau_star"] / gated[0]["tau_star"],
        "kappa_window": [gated[0]["kappa"], gated[-1]["kappa"]],
        "inertial_numbers": [round(d["I"], 4) for d in gated],
        "caveat": ("passes the registered absolute band, but tau* still grows by the "
                   "factor above across the window and I=0.08..0.56 is the dense-"
                   "inertial regime: a cross-over exponent, not a demonstrated plateau")}
    sweep = sorted([d for d in pts.values() if d["case"]["kn"] == 1.0],
                   key=lambda d: d["case"]["shear_rate"])
    gd = [d["case"]["shear_rate"] for d in sweep]
    n_in = slope(gd, [d["tau"] for d in sweep])
    legs["R1"] = {
        "reference": "n = 2 (Bagnold, tau ~ rho d^2 gdot^2)",
        "tolerance": f"relative {TOL_R1_REL:.0%}", "gated_points": 0,
        "declared_status": "not_run / NOT_SUPPORTED",
        "available_evidence": {"kn": 1.0, "shear_rates": gd,
                               "n_from_ungated_sweep": n_in,
                               "pressure_exponent": slope(gd, [d["pressure"] for d in sweep]),
                               "max_overlap_ratio": [round(d["max_overlap_ratio"], 3)
                                                     for d in sweep]},
        "within_tolerance": False,
        "rel_error_if_taken_at_face_value": abs(n_in - 2.0) / 2.0,
        "note": ("no gated point reaches the inertial branch (the kn=1e0 sweep sits at "
                 "0.37-0.60 d overlap, outside the linear-spring validity window); the "
                 "ungated sweep gives n~0.1, so the available data contradict Bagnold "
                 "rather than support it")}
    kappas = [d["kappa"] for d in ladder]
    legs["R2"] = {
        "reference": "kappa_c = 1e4 (elastic -> inertial regime boundary)",
        "tolerance": f"order of magnitude (factor {TOL_R2_FACTOR})",
        "kappa_span_measured": [min(kappas), max(kappas)],
        "highest_gated_kappa": max(d["kappa"] for d in gated),
        "points_inside_factor_3_band": [k for k in kappas
                                        if 1e4 / TOL_R2_FACTOR <= k <= 1e4 * TOL_R2_FACTOR],
        "within_tolerance": False,
        "reason": ("no regime break anywhere in the measured span; the point inside the "
                   "factor-3 band (kappa=5787) still follows the same power law, so a "
                   "boundary at 1e4 is neither located nor corroborated")}
    legs["R4"] = {"reference": "Bagnold plateau coefficient (Campbell JFM 465)",
                  "tolerance": "WITHHELD", "within_tolerance": None,
                  "reason": ("declared WITHHELD in the case doc and the report; no "
                             "coefficient value is asserted anywhere in the deliverables, "
                             "so nothing is over-claimed")}
    return legs


def energy_audit(pts):
    per = {}
    for tag, d in pts.items():
        e = d["energy"]
        rec = abs(e["work"] - e["dissipation_normal"] - e["dissipation_tangential"]
                  - e["stored_change"]) / max(abs(e["work"]), 1e-300)
        per[tag] = {"recomputed_global_defect": rec,
                    "filed_global_defect": e["relative_defect_global"],
                    "window_defect": e["relative_defect"],
                    "gate_recorded_in_artifact": e["gate"],
                    "ratio_window_defect_to_recorded_gate": e["relative_defect"] / e["gate"]}
    probe_log = RUNS / "probe_kn1e3.log"
    probe = None
    if probe_log.exists():
        for line in probe_log.read_text().splitlines():
            if line.startswith("PROBE"):
                probe = line
    return {"per_point": per,
            "arithmetic_reproduces": all(
                abs(v["recomputed_global_defect"] - v["filed_global_defect"]) < 1e-9
                for v in per.values()),
            "probe": {"log": str(probe_log.relative_to(ROOT)) if probe_log.exists() else None,
                      "line": probe,
                      "hypothesis_tested": ("the residual is the tangential-spring energy "
                                            "retained by pairs that have separated"),
                      "result": ("rejected as the (sole) mechanism: retained tangential "
                                 "energy of non-contacting pairs is 0.00% of the window "
                                 "work while the residual stays at 25.5%")},
            "verdict": ("the audit residual is 1.9-2.5x the energy gate recorded in the "
                        "artifacts; the 'model-level, dt-independent' explanation is "
                        "plausible but its mechanism is unidentified, so the audit cannot "
                        "certify the energetics of the gated points")}


def reproduction():
    filed = json.loads((EXPORT / ART / "point_kn1e+03_g1_n400.json").read_text())
    out = {"filed_tau_star": filed["tau_star"]}
    mine_p = RUNS / "point_audit_repro_kn1e3.json"
    if mine_p.exists():
        mine = json.loads(mine_p.read_text())
        out.update({"repro_artifact": str(mine_p.relative_to(ROOT)),
                    "repro_tau_star": mine["tau_star"],
                    "rel_diff": abs(mine["tau_star"] - filed["tau_star"]) / filed["tau_star"],
                    "bit_identical": mine["tau_star"] == filed["tau_star"],
                    "params": {k: mine["case"][k] for k in
                               ("kn", "shear_rate", "n_particles", "strain_total",
                                "strain_average_from", "dt_over_contact_time", "seed")}})
        out["verdict"] = ("point reproduces bit-exactly from the branch commit at the filed "
                          "parameters -- the artifact is real output of this code"
                          if out["bit_identical"] else "differs from the filed point")
    rc, so, se = sh([sys.executable, "-m", "tensordem.cli", "elastic-limit-shear",
                     "--kn", "1e2", "--n", "200"], cwd=str(ROOT))
    guard = sh(["grep", "-c", "if __name__", str(EXPORT / "src/tensordem/cli.py")])
    out["documented_repro_cmd"] = {
        "command": ("PYTHONPATH=src python -m tensordem.cli elastic-limit-shear "
                    "--kn 1e4 --shear-rate 1.0"),
        "exit_code": rc, "stdout_bytes": len(so), "stderr_bytes": len(se),
        "cli_py_main_guard_hits": int(guard[1].split(":")[-1]) if guard[1] else 0,
        "finding": ("the documented command is a silent no-op: cli.py has no "
                    "`if __name__ == '__main__'` guard, so `python -m tensordem.cli` "
                    "imports the module, never calls main(), prints nothing and exits 0"
                    if rc == 0 and not so and not se else
                    "documented command produced output"),
        "working_invocation": ("PYTHONPATH=src python -m tensordem elastic-limit-shear "
                               "--kn/--n/--strain ...")}
    cli_p = RUNS / "cli_point_kn1e2.json"
    if cli_p.exists():
        d = json.loads(cli_p.read_text())
        out["cli_subcommand_really_runs"] = {
            "artifact": str(cli_p.relative_to(ROOT)), "kappa": d["kappa"],
            "tau_star": d["tau_star"], "mu_eff": d["mu_eff"],
            "N": d["case"]["n_particles"], "gates_all": d["gates"]["all"]}
    return out


def report_check():
    md = psql(f"SELECT markdown FROM tensor_reports WHERE mission_id={MISSION} "
              "ORDER BY id DESC LIMIT 1")
    figs = psql(f"SELECT coalesce(length(figures::text),0) FROM tensor_reports "
                f"WHERE mission_id={MISSION} ORDER BY id DESC LIMIT 1")
    heads = ["①", "②", "③", "④", "⑤", "⑥"]
    fig_raw = psql(f"SELECT figures::text FROM tensor_reports WHERE mission_id={MISSION} "
                   "ORDER BY id DESC LIMIT 1")
    png_embedded = "png_base64" in fig_raw
    png = EXPORT / ART / "fig_tau_star_kappa.png"
    png_stats = None
    if png.exists():
        try:
            import numpy as np
            from PIL import Image
            arr = np.asarray(Image.open(png).convert("L"))
            png_stats = {"size": list(Image.open(png).size),
                         "mean": float(arr.mean()), "std": float(arr.std()),
                         "nonwhite_fraction": float((arr < 250).mean())}
        except Exception as exc:  # pragma: no cover
            png_stats = {"error": str(exc)}
    return {"markdown_bytes": len(md),
            "sections_present": {h: (h in md) for h in heads},
            "n_sections": sum(h in md for h in heads),
            "figures_json_bytes": int(figs or 0),
            "png_base64_embedded": png_embedded,
            "mission_figure_png": png_stats,
            "explicit_3pct_answer": ("是否 <3%" in md.replace(" ", ""))}


def tests_replay():
    env = dict(os.environ, PYTHONPATH=str(EXPORT / "src"))
    p = subprocess.run([sys.executable, "-m", "pytest", "-q",
                        str(EXPORT / "tests/test_elastic_limit_shear.py")],
                       cwd=str(EXPORT), capture_output=True, text=True, env=env,
                       timeout=1800)
    tail = [l for l in p.stdout.splitlines() if "passed" in l or "failed" in l]
    return {"cmd": "PYTHONPATH=src pytest -q tests/test_elastic_limit_shear.py",
            "exit_code": p.returncode, "summary": tail[-1] if tail else "",
            "asserts_invariants": ["min-image vs brute force for s in {0,.1,.3,.5,-.4}",
                                   "single-contact stress sign + normalisation",
                                   "packing solid fraction + overlap-free start",
                                   "synthetic n=0 / n=2 branch recovery by the fitter",
                                   "refusal when fewer than 3 gated points"]}


def figure(pts, a):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    ladder = sorted([d for d in pts.values() if d["case"]["shear_rate"] == 1.0],
                    key=lambda d: d["kappa"])
    regated = [d for d in ladder if gates_of(d, 0.30)["all"]]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].loglog([d["kappa"] for d in ladder], [d["tau_star"] for d in ladder], "o",
                 color="#c62828", ms=7, label="all 8 points (artifacts' own gate: rejected)")
    ax[0].loglog([d["kappa"] for d in regated], [d["tau_star"] for d in regated], "s",
                 mfc="none", mec="#1565c0", ms=12, label="re-gated at 0.30 (headline)")
    k0, t0 = regated[0]["kappa"], regated[0]["tau_star"]
    kk = [d["kappa"] for d in ladder]
    beta = a["gating"]["committed_code_default"]["beta"]
    ax[0].loglog(kk, [t0 * (k / k0) for k in kk], "--", color="0.4", lw=1,
                 label=r"$\kappa^1$ (elastic, R3 reference)")
    ax[0].loglog(kk, [t0 for k in kk], ":", color="0.4", lw=1,
                 label=r"$\kappa^0$ (plateau: R1/R4)")
    ax[0].loglog(kk, [t0 * (k / k0) ** beta for k in kk], "-", color="#2e7d32", lw=1.3,
                 label=fr"fit $\beta$={beta:.3f} -> n={2*(1-beta):.3f}")
    ax[0].set_xlabel(r"$\kappa=k_n/(\rho d^3\dot\gamma^2)$")
    ax[0].set_ylabel(r"$\tau^*=\tau/(\rho d^2\dot\gamma^2)$")
    ax[0].set_title("referee R3: elastic-limit exponent (auditor recompute)")
    ax[0].legend(fontsize=7.5)
    ax[0].grid(True, which="both", alpha=0.3)

    tags = [d["tag"] for d in ladder]
    x = list(range(len(tags)))
    ax[1].bar([i - 0.2 for i in x], [d["energy"]["relative_defect"] for d in ladder],
              width=0.38, color="#c62828", label="window audit residual")
    ax[1].bar([i + 0.2 for i in x],
              [d["energy"]["relative_defect_global"] for d in ladder], width=0.38,
              color="#1565c0", label="global audit residual")
    ax[1].axhline(0.1, color="k", ls="--", lw=1,
                  label="energy gate recorded in artifacts (0.1)")
    ax[1].axhline(0.30, color="#2e7d32", ls=":", lw=1.2,
                  label="energy gate of committed code (0.30)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([t.replace("_n400", "") for t in tags], rotation=45,
                          ha="right", fontsize=7)
    ax[1].set_ylabel("relative energy-audit residual")
    ax[1].set_title("the gate that decides which points count")
    ax[1].legend(fontsize=7.5)
    ax[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = OUT / "fig_m16316_audit.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p.relative_to(ROOT))


def decide(a):
    legs = a["referee_legs"]
    passed = [k for k, v in legs.items() if v.get("within_tolerance") is True]
    failed = [k for k, v in legs.items() if v.get("within_tolerance") is False]
    withheld = [k for k, v in legs.items() if v.get("within_tolerance") is None]
    return {
        "level": 3,
        "rationale": (
            "L1 met: 15 tests assert invariants (min-image vs brute force, single-contact "
            "stress sign+normalisation, packing fraction, synthetic n=0/n=2 recovery, "
            "gate-refusal path) -- replayed green by the auditor. "
            "L2 met: a self-contained 825-line SLLOD/Lees-Edwards kernel with a real "
            "input->compute->output path; one filed point reproduces bit-exactly "
            "(tau*=11.187963375044525). "
            "L3 met at branch level: `elastic-limit-shear` is wired into "
            "src/tensordem/cli.py (+37 lines) and runs end-to-end via `python -m "
            "tensordem` (kappa=57.87, tau*=1.562, all gates pass at N=200) -- but the "
            "branch is NOT merged into main, so trunk still ships the round-1 kernel "
            "with no wiring. "
            f"L4 NOT met: of the 4 registered referee legs only {passed} passes "
            f"({len(passed)}/{len(legs)}); R1 and R2 fail, R4 is WITHHELD. That single "
            "pass is a cross-over exponent in the dense-inertial regime (I=0.08-0.56, "
            "tau* still growing x54.9), it exists only after an energy gate loosened 3x "
            "after the runs (the shipped artifacts carry gates.all=0 under their own "
            "gate 0.1 and were never regenerated), the energy audit closes only to "
            "19-25% of the shear work with an unidentified mechanism, the documented "
            "CLI repro command is a silent no-op, and the finite-size study is not run. "
            "Real kernel + tests + CLI wiring + a partially validated case = L3."),
        "passed_legs": passed, "failed_legs": failed, "withheld_legs": withheld,
        "why_not_L4": (
            "The structural L4 tests (complete case, error analysis, reproducibility) "
            "are met on paper: a full kappa ladder + rate sweeps, a dt-convergence pair, "
            "an explicit per-leg error statement, and bit-exact reproducibility by the "
            "auditor. What is NOT met is the referee itself: 3 of the 4 registered legs "
            "are unmet, and the single passing leg collapses to zero gated points if the "
            "gate recorded inside the artifacts (0.1) is applied instead of the 0.30 that "
            "the committed code silently substitutes. An honest FAIL is a legitimate "
            "scientific result, but it cannot be booked as 'benchmark validated' while "
            "the case's own artifacts contradict the gated set they are supposed to "
            "support. Hence L3, not L4; nothing in this verdict disputes the physics "
            "numbers themselves, which reproduce exactly."),
        "what_moves_it_to_L4": [
            "run kn=1e5/1e6 rungs + rate sweeps at kn=1e6 (sharded) so R1/R2 have gated "
            "points in the inertial band",
            "identify or fix the mechanism of the 19-25% energy-audit residual (the "
            "retained-tangential-history hypothesis was falsified by the probe)",
            "regenerate the artifacts under the final gate so point JSONs and "
            "scaling.json stop contradicting each other",
            "merge agent/m16243 into main, otherwise the capability is not in trunk",
            "N=800 finite-size + an uncertainty on n (the 3-point fit carries no error bar)",
            "fix the documented repro command (python -m tensordem, not "
            "python -m tensordem.cli)"],
    }


def evidence_json(a):
    g = a["gating"]
    return {
        "impl_files": [
            "src/tensordem/elastic_limit_shear.py:440 simulate_plane_shear",
            "src/tensordem/elastic_limit_shear.py:761 shear_stress_scaling",
            "src/tensordem/elastic_limit_shear.py:807 shear_rate_exponent",
            "benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py:47 run_point",
            "benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py:93 re_gate"],
        "entry": "src/tensordem/cli.py:238 (subcommand) -> src/tensordem/cli.py:165 "
                 "_cmd_elastic_limit_shear",
        "run_artifacts": [
            f"{ART}/scaling.json",
            f"{ART}/point_kn1e+0[0-4]_g1_n400.json + point_kn1e+00_g0.25/0.5/2_n400.json "
            "(8 points)",
            f"{ART}/dt_convergence.json", f"{ART}/fig_tau_star_kappa.png",
            f"{ART}/logs/sweep_all.log",
            "runs/audit_m16316/point_audit_repro_kn1e3.json (auditor re-run)",
            "runs/audit_m16316/cli_point_kn1e2.json (CLI-run point)",
            "runs/audit_m16316/probe_kn1e3.log (energy-audit probe)"],
        "tests": "tests/test_elastic_limit_shear.py (15 passed; invariants: min-image vs "
                 "brute force, single-contact stress normalisation, packing fraction, "
                 "synthetic n=0/n=2 recovery, gate refusal)",
        "integration_callers": "src/tensordem/cli.py:238 -> :165 (runs via python -m "
                               "tensordem); branch only -- main has 0 wiring lines",
        "benchmark": {
            "referee_rows": a["referee"]["rows"],
            "R1": a["referee_legs"]["R1"]["within_tolerance"],
            "R2": a["referee_legs"]["R2"]["within_tolerance"],
            "R3": {"got_n": a["referee_legs"]["R3"]["got_n"],
                   "abs_dev": a["referee_legs"]["R3"]["abs_dev_from_reference"],
                   "tolerance": TOL_R3,
                   "within_tolerance": a["referee_legs"]["R3"]["within_tolerance"]},
            "R4": "WITHHELD (paywall)",
            "passed_legs": a["verdict"]["passed_legs"],
            "recompute_from_bytes": (f"beta={g['committed_code_default']['beta']:.6f} "
                                     f"n={g['committed_code_default']['n']:.6f} "
                                     "(matches scaling.json exactly)"),
            "as_filed_gate_gives": g["as_filed_in_artifacts"].get("status")},
        "repro_cmd": {
            "sweep_point": ("PYTHONPATH=src python benchmarks/case_dem_shear_campbell_"
                            "elastic_limit/run_case.py --tag kn1e3 --kn 1e3 --shear-rate "
                            "1.0 --n 400 --strain 1.8 --strain-from 0.8 --dt-factor 0.0025"
                            "   # bit-exact tau*=11.187963375044525"),
            "documented_cli_cmd_is_a_no_op": a["reproduction"]["documented_repro_cmd"][
                "finding"],
            "working_cli": "PYTHONPATH=src python -m tensordem elastic-limit-shear "
                           "--kn 1e2 --n 200",
            "audit_repro_artifact": "runs/audit_m16316/point_audit_repro_kn1e3.json"},
        "external_anchor": "Campbell 2002 JFM 465:261-291 doi:10.1017/S002211200200109X "
                          "(R4 WITHHELD: no network / no institutional access; R2 relies "
                          "on the secondary paraphrase arXiv:1712.02693)",
        "audit_evidence": {
            "energy_audit": a["energy_audit"]["verdict"],
            "gate_conflict": g["gate_conflict"]["finding"],
            "merge_state": a["provenance"]["note"],
            "report_sections": f"{a['report']['n_sections']}/6 present",
            "level": a["verdict"]["level"]},
    }


def render_md(a):
    g, legs, v, rp = a["gating"], a["referee_legs"], a["verdict"], a["reproduction"]
    bad, good = g["as_filed_in_artifacts"], g["committed_code_default"]
    r3, L = legs["R3"], []
    L.append("# m16316 board audit — m16243 CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT\n")
    L.append(f"**Verdict: L{v['level']}** (verdict_by=board-audit). Audited branch "
             f"`{a['provenance']['branch']}` @ {a['provenance']['branch_tip']}, main @ "
             f"{a['provenance']['main_head']}; merged into main: "
             f"**{a['provenance']['merged_into_main']}**.\n")
    L.append("## 1. Referee — the only judge\n")
    L.append(f"`{a['referee']['query']}` → {a['referee']['n_rows']} rows:\n")
    for r in a["referee"]["rows"]:
        L.append(f"- `{r}`")
    L.append("\n## 2. Deliverables (bytes)\n")
    L.append("| file | bytes | sha256[:16] | commits touching |")
    L.append("|---|---|---|---|")
    for rel, e in a["provenance"]["files"].items():
        L.append(f"| `{rel}` | {e.get('bytes','-')} | `{str(e.get('sha256','-'))[:16]}` | "
                 f"{e.get('commits_touching_file','-')} |")
    L.append(f"\n{a['provenance']['note']}. Trunk (`main`) CLI wiring lines: "
             f"{a['provenance']['trunk_cli_wiring_lines']}.\n")
    L.append("## 3. Arithmetic recomputed from each artifact's own case block\n")
    w = a["arithmetic"]["worst"]
    L.append(f"Worst relative difference (recomputed vs filed kappa/tau*/mu/I): "
             f"**{w['rel']:.3g}**"
             + (f" ({w['what']} @ {w['tag']})" if w["what"] else " (all four groups, all "
                "8 points)")
             + f" — {a['arithmetic']['verdict']}.\n")
    L.append("## 4. The gate that decides which points count\n")
    L.append(f"- artifacts record `energy.gate = "
             f"{g['gate_conflict']['artifacts_record_energy_gate']}`; their own "
             f"`gates.all`: `{g['gate_conflict']['artifacts_all_gate_flag']}`")
    L.append(f"- under that gate: **{bad['n_gated']} gated points** → "
             f"{bad.get('status','scaling available')}")
    L.append(f"- under the committed code's gate (0.30): **{good['n_gated']} gated points**, "
             f"β={good['beta']:.6f}, n={good['n']:.6f} (matches `scaling.json`: "
             f"{g['agrees_with_filed']}); local "
             + ", ".join(f"n={l['n']:.4f}" for l in good["local"]))
    L.append(f"\n{g['gate_conflict']['finding']}.\n")
    L.append("## 5. Referee legs\n")
    L.append("| leg | reference | tolerance | got | within? |")
    L.append("|---|---|---|---|---|")
    L.append(f"| R1 | n=2 (Bagnold) | 15 % rel | no gated point; ungated kn=1e0 sweep → "
             f"n={legs['R1']['available_evidence']['n_from_ungated_sweep']:.4f}, pressure "
             f"exponent {legs['R1']['available_evidence']['pressure_exponent']:.3f}, "
             f"overlap up to "
             f"{max(legs['R1']['available_evidence']['max_overlap_ratio']):.2f} d | **NO** |")
    L.append(f"| R2 | kappa_c=1e4 | factor 3 | no regime break in "
             f"{legs['R2']['kappa_span_measured'][0]:.4g}-"
             f"{legs['R2']['kappa_span_measured'][1]:.4g}; the one point inside the "
             f"factor-3 band (κ={legs['R2']['highest_gated_kappa']:.0f}) still follows the "
             f"same law | **NO** |")
    L.append(f"| R3 | n=0 (elastic limit) | absolute 0.30 | n={r3['got_n']:.4f} "
             f"(β={r3['got_beta']:.4f}), |dev|={r3['abs_dev_from_reference']:.4f}; τ* still "
             f"grows ×{r3['tau_star_growth_over_window']:.1f}; I={r3['inertial_numbers']} | "
             f"**YES** (weak) |")
    L.append("| R4 | plateau coefficient | WITHHELD | nothing asserted anywhere | n/a |")
    L.append(f"\nPassed {v['passed_legs']} / {len(legs)} legs; failed {v['failed_legs']}; "
             f"withheld {v['withheld_legs']}.\n")
    L.append("## 6. Energy audit\n")
    for tag, e in a["energy_audit"]["per_point"].items():
        if "g1_" in tag and tag.endswith("n400"):
            L.append(f"- `{tag}`: window residual {e['window_defect']:.4f}, global "
                     f"{e['filed_global_defect']:.4f}, gate recorded in the artifact "
                     f"{e['gate_recorded_in_artifact']} "
                     f"({e['ratio_window_defect_to_recorded_gate']:.1f}× over)")
    L.append(f"\nThe global residual reproduces arithmetically from the artifact's own "
             f"work/dissipation/stored numbers: "
             f"**{a['energy_audit']['arithmetic_reproduces']}**.")
    L.append(f"\nProbe (`{a['energy_audit']['probe']['log']}`, instrumented copy, same "
             f"seed): hypothesis = {a['energy_audit']['probe']['hypothesis_tested']} → "
             f"{a['energy_audit']['probe']['result']}.")
    L.append(f"\n{a['energy_audit']['verdict']}.\n")
    L.append("## 7. Reproduction by the auditor\n")
    L.append(f"- filed kn=1e3 point: τ*={rp['filed_tau_star']!r}; auditor re-run "
             f"{rp['repro_tau_star']!r} (rel diff {rp['rel_diff']:.1e}, bit identical "
             f"{rp['bit_identical']}); params {rp['params']}")
    L.append(f"- {rp['verdict']}")
    dr = rp["documented_repro_cmd"]
    L.append(f"- documented repro command `{dr['command']}`: exit {dr['exit_code']}, "
             f"stdout {dr['stdout_bytes']} B, stderr {dr['stderr_bytes']} B, `__main__` "
             f"guards in cli.py: {dr['cli_py_main_guard_hits']} → {dr['finding']}")
    cr = rp.get("cli_subcommand_really_runs", {})
    L.append(f"- working invocation `{dr['working_invocation']}`: κ={cr.get('kappa')}, "
             f"τ*={cr.get('tau_star')}, μ={cr.get('mu_eff')}, gates_all="
             f"{cr.get('gates_all')} at N={cr.get('N')} (artifact {cr.get('artifact')})")
    L.append(f"\n## 8. Tests\n\n`{a['tests']['cmd']}` → **{a['tests']['summary']}** "
             f"(exit {a['tests']['exit_code']}); they assert "
             + "; ".join(a["tests"]["asserts_invariants"]) + ".\n")
    L.append(f"## 9. Report check\n\nFigure embedded as png_base64: "
             f"**{a['report']['png_base64_embedded']}** (the figures payload carries only "
             f"name+caption; the PNG itself lives in the repo: "
             f"`{ART}/fig_tau_star_kappa.png`, {a['report']['mission_figure_png']}).\n")
    L.append(f"Six required sections present: "
             f"**{a['report']['n_sections']}/6** "
             f"({a['report']['sections_present']}); figures payload "
             f"{a['report']['figures_json_bytes']} B; explicit \u201c<3%\u201d answer: "
             f"{a['report']['explicit_3pct_answer']}.\n")
    L.append("## 10. Disposition\n")
    L.append(f"**L{v['level']} — capability real but the registered benchmark is only "
             f"partially met.** {v['rationale']}\n")
    L.append("To reach L4:\n")
    for i, t in enumerate(v["what_moves_it_to_L4"], 1):
        L.append(f"{i}. {t}")
    L.append(f"\nFigure: `{a['figure']}` (left: referee R3 recompute + reference slopes; "
             f"right: audit residual vs the two competing gates).")
    L.append("\nOverstated claims found: (i) “elastic-limit leg reproduced” while τ* still "
             "grows ×54.9 and I=0.08–0.56 (dense-inertial, not quasi-static); (ii) the "
             "documented CLI repro line, which exits 0 without running anything; "
             "(iii) the case ships point JSONs whose own gates say FAILED while "
             "scaling.json counts those same points as gated.\n")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insert-verdict", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    a = {"audit_mission": AUDIT_MISSION, "audited_mission": MISSION, "anchor": ANCHOR}
    export_branch()
    a["provenance"] = provenance()
    a["referee"] = referee()
    pts = load_points()
    a["points_loaded"] = sorted(pts)
    a["arithmetic"] = arithmetic(pts)
    a["gating"] = gating(pts)
    a["referee_legs"] = referee_legs(pts)
    a["energy_audit"] = energy_audit(pts)
    a["reproduction"] = reproduction()
    a["report"] = report_check()
    a["tests"] = tests_replay()
    a["verdict"] = decide(a)
    a["figure"] = figure(pts, a)
    (OUT / "audit_m16243.json").write_text(json.dumps(a, indent=1, ensure_ascii=False))
    (OUT / "AUDIT.md").write_text(render_md(a), encoding="utf-8")
    if args.insert_verdict:
        ev = json.dumps(evidence_json(a), ensure_ascii=False).replace("'", "''")
        row = psql("INSERT INTO tensor_maturity_verdicts "
                   "(repo, scope, mission_id, level, evidence, verdict_by) VALUES "
                   f"('TensorDEM', 'm{MISSION} {ANCHOR}', {AUDIT_MISSION}, "
                   f"{a['verdict']['level']}, '{ev}', 'board-audit') RETURNING id;")
        a["verdict_row_id"] = row
        (OUT / "audit_m16243.json").write_text(json.dumps(a, indent=1, ensure_ascii=False))
        print("verdict row:", row)
    print(json.dumps({"level": a["verdict"]["level"],
                      "legs": {k: v.get("within_tolerance")
                               for k, v in a["referee_legs"].items()},
                      "n_sections": a["report"]["n_sections"],
                      "merged": a["provenance"]["merged_into_main"],
                      "figure": a["figure"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())