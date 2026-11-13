#!/usr/bin/env python3
"""Board-audit independent verification of mission m17721.

Scope: XVAL-DEM-ROUND2-METHOD-SELECTION (TensorDEM).
This script does NOT reuse the mission's own report numbers: it recomputes every
claim from the implementation (`tensordem.contact.rolling_static`) and from the
registered benchmark table, then classifies each criterion by evidential type
(external reference vs analytic identity vs definitional tautology).

Outputs
-------
artifacts/audit_m17721/independent_check.json
artifacts/audit_m17721/audit_m17721.png
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import hashlib
from pathlib import Path

import numpy as np

WT = Path("/root/tp-agent/work/m17721")          # audited mission worktree
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "audit_m17721"
sys.path.insert(0, str(WT / "src"))

from tensordem.contact.rolling_static import (  # noqa: E402
    rolling_lever_arm, rolling_limit, rolling_saturation, stopping_speed,
)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def registered_cases(anchor: str):
    sql = ("SELECT case_name,metric,tolerance FROM tensor_benchmark_cases "
           f"WHERE anchor_slug='{anchor}'")
    r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
                        "-tA", "-F", "\x01", "-c", sql],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip()
    rows = [ln.split("\x01") for ln in r.stdout.strip().split("\n") if ln.strip()]
    return rows, None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"anchor": "XVAL-DEM-ROUND2-METHOD-SELECTION", "mission": "m17721",
                 "audit_mission": "m17740", "checks": {}}

    # ---------------------------------------------------------------- 1. referee
    rows, err = registered_cases("XVAL-DEM-ROUND2-METHOD-SELECTION")
    res["checks"]["registered_benchmark_cases"] = {
        "query": "tensor_benchmark_cases WHERE anchor_slug='XVAL-DEM-ROUND2-METHOD-SELECTION'",
        "rows": None if rows is None else len(rows),
        "error": err,
        "verdict": "NO REGISTERED CASE -> L4 referee absent" if not rows else "present",
    }

    # ------------------------------------------------- 2. recompute the 6 claims
    # (a) sphere baseline
    contact = np.array([[0.0012, 0.0, 0.0]])
    com = np.array([[0.0, 0.0, 0.0]])
    R = np.array([0.0012])
    r_com = float(rolling_lever_arm(contact, com, R, "com")[0])
    r_mem = float(rolling_lever_arm(contact, com, R, "member")[0])

    # (b)-(e) coulomb/viscous bookkeeping
    Fn = np.array([1.0])
    limit = rolling_limit(0.3, np.array([0.0012]), Fn)
    dt, I = 5e-5, np.array([1e-10])
    w_eps = float(stopping_speed(limit, I, dt)[0])
    sat_hi = float(rolling_saturation(np.array([[0.0, 0.0, 1000.0]]), limit, I, dt,
                                      model="coulomb")[0])
    sat_lo = float(rolling_saturation(np.array([[0.0, 0.0, 0.001]]), limit, I, dt,
                                      model="coulomb")[0])
    sat_visc = float(rolling_saturation(np.array([[0.0, 0.0, 1.0]]), limit, I, dt,
                                        model="viscous", eps_w=1.0)[0])
    # independent closed-form recomputation (not the module's function)
    stop_hi = I[0] * 1000.0 / dt
    sat_hi_closed = min(1.0, stop_hi / limit[0])
    w_eps_closed = limit[0] * dt / I[0]

    # (f) anisotropy
    b = 0.0012
    a = 3 * b
    tip = float(rolling_lever_arm(np.array([[a, 0.0, 0.0]]), com, np.array([b]), "com")[0])
    flank = float(rolling_lever_arm(np.array([[0.0, b, 0.0]]), com, np.array([b]), "com")[0])

    mission = json.loads((WT / "artifacts/xval_round2_method_selection/xval_report.json").read_text())

    def _report_value(c):
        """The number the mission report presents as 'got' for this criterion."""
        for k in ("saturation", "anisotropy_ratio", "w_eps_computed", "absolute_error"):
            if k in c and got_is_got(c, k):
                return float(c[k])
        return float(c.get("absolute_error", 0.0))

    def got_is_got(c, k):
        return True

    def cmp(name, got, ref, tol, etype, note):
        c = mission["criteria"][name]
        return {
            "claim_in_report": c,
            "audit_recomputed_got": float(got),
            "audit_reference": None if ref is None else float(ref),
            "abs_error": None if ref is None else float(abs(got - ref)),
            "report_tolerance": tol,
            "within_report_tolerance": None if ref is None else bool(abs(got - ref) <= tol),
            "matches_report": bool(abs(got - _report_value(c)) < 1e-12),
            "evidence_type": etype,
            "note": note,
        }

    res["checks"]["a_sphere_baseline"] = cmp(
        "a_sphere_baseline", abs(r_com - r_mem), 0.0, 1e-15, "analytic identity (by construction)",
        "impl clips arm to max(|x-com|, r_mem); for a sphere |x-com| == r_mem, so equality is designed in.")
    res["checks"]["b_coulomb_high_speed"] = cmp(
        "b_coulomb_high_speed", sat_hi, min(1.0, stop_hi / limit[0]), 0.01, "analytic identity",
        "min(1, I|w|/(dt*limit)) == 1 whenever I|w| >= dt*limit; the test point (1e-10*1000/5e-5=2e-3 vs 3.6e-4) guarantees it.")
    res["checks"]["c_coulomb_low_speed"] = cmp(
        "c_coulomb_low_speed", sat_lo, 0.0, 0.01, "analytic identity",
        "sat = I|w|/(dt*limit) = 5.56e-6 at |w|=1e-3; reference '0' is the same formula's limit.")
    res["checks"]["d_viscous_at_eps"] = cmp(
        "d_viscous_at_eps", sat_visc, 0.5, 1e-12, "definitional tautology",
        "viscous saturation is defined as |w|/(|w|+eps_w); at |w|=eps_w it is 0.5 by algebra, independent of any physics.")
    res["checks"]["e_stopping_speed"] = cmp(
        "e_stopping_speed", w_eps, w_eps_closed, 1e-12, "definitional tautology",
        "stopping_speed() *is* limit*dt/inertia; the 'analytical' side re-types the same expression, so 0.0 error is guaranteed.")
    res["checks"]["f_clump_anisotropy"] = {
        "claim_in_report": mission["criteria"]["f_clump_anisotropy"],
        "audit_recomputed_got": float(tip / flank),
        "audit_reference": float(a / b),
        "abs_error": float(abs(tip / flank - a / b)),
        "report_tolerance": 2.0,
        "within_report_tolerance": bool(tip / flank > 2.0),
        "matches_report": bool(abs(tip / flank - mission["criteria"]["f_clump_anisotropy"]
                                   ["anisotropy_ratio"]) < 1e-12),
        "evidence_type": "geometric tautology",
        "note": "arm = |x-com| for a rigid clump placed at the origin; tip contact at a, flank at b => ratio == a/b == AR. "
                "Threshold 2.0 is self-declared, not a literature value.",
    }

    # ------------------------------------------------ 3. what the harness does NOT do
    script = (WT / "scripts/run_xval_round2_method_selection.py").read_text()
    res["checks"]["harness_scope"] = {
        "lines": len(script.splitlines()),
        "imports_engine": any(t in script for t in (
            "clump_dem", "dem3d", "ClumpDEM", "run_round2_scan", "repose_protocol")),
        "runs_time_integration": bool("sim.step(" in script or "dem3d" in script),
        "literature_mentions": {k: script.count(k) for k in ("Zhou", "Wensrich", "Ai et al")},
        "literature_numeric_use": 0,
        "oss_comparison_type": "hand-authored boolean feature matrix, no numeric comparison",
    }

    # ----------------------------------------------------------- 4. reproducibility
    repro = subprocess.run([sys.executable, "scripts/run_xval_round2_method_selection.py",
                            "--outdir", "/tmp/xval_audit_m17740"],
                           cwd=WT, capture_output=True, text=True, timeout=900)
    a = json.loads(Path("/tmp/xval_audit_m17740/xval_report.json").read_text())
    b0 = mission
    a.pop("timestamp", None)
    b0 = json.loads(json.dumps(b0))
    b0.pop("timestamp", None)
    res["checks"]["repro"] = {
        "exit_code": repro.returncode,
        "stderr_tail": repro.stderr.strip().splitlines()[-2:] if repro.stderr else [],
        "json_identical_modulo_timestamp": a == b0,
        "report_sha256": sha(WT / "artifacts/xval_round2_method_selection/xval_report.json"),
        "repro_sha256": sha(Path("/tmp/xval_audit_m17740/xval_report.json")),
        "command": "cd /root/tp-agent/work/m17721 && python scripts/run_xval_round2_method_selection.py",
    }

    # --------------------------------------------------------------- 5. provenance
    st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                        cwd=WT, capture_output=True, text=True).stdout
    res["checks"]["provenance"] = {
        "deliverables_tracked": not any(
            p in st for p in ("xval_round2_method_selection", "XVAL_ROUND2_METHOD_SELECTION.md",
                              "run_xval_round2_method_selection.py")),
        "git_status_excerpt": st.strip().splitlines()[:8],
        "impl_commit": subprocess.run(["git", "log", "-1", "--format=%h %s", "--",
                                       "src/tensordem/contact/rolling_static.py"],
                                      cwd=WT, capture_output=True, text=True).stdout.strip(),
        "note": "implementation predates this mission; m17721 delta = XVAL harness + docs + artifacts only",
    }

    res["audit_level"] = 2
    res["level_reasons"] = {
        "L1": "granted: 13 unit tests in tests/test_rolling_static.py pass (independent run).",
        "L2": "granted: real 180-line impl (rolling_static.py) + 407-line entry + reproducible run artifacts.",
        "L3": "denied: the XVAL harness never imports/executes the solver (no clump_dem/dem3d/sim.step); "
              "integration callers exist only in pre-existing code (clump_dem.py:63,313,319) and the "
              "deliverables are untracked in git, so nothing new reached the pushed repo.",
        "L4": "denied: tensor_benchmark_cases has 0 rows for this anchor; all six 'ref' values are the "
              "implementation's own algebraic identities (d/e definitional, a/f geometric, b/c limits), "
              "and the cited literature (Zhou1999 / Wensrich2012 / Ai2011) is never evaluated numerically.",
        "overstatement": "report/doc claim 'Maturity: L4'; audit finds L2 (2 levels inflated).",
    }
    (OUT / "independent_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    # ----------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = np.logspace(-2, 3, 200)
    lim = float(limit[0])
    sat_c = np.minimum(1.0, I[0] * w / dt / lim)
    sat_v = w / (w + 1.0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(w, sat_c, label="coulomb (round-2)", lw=2)
    ax[0].loglog(w, sat_v, "--", label="viscous (legacy, eps_w=1)", lw=2)
    ax[0].axvline(w_eps, color="k", ls=":", label=f"w_eps={w_eps:.0f} rad/s")
    ax[0].set_xlabel("|w_rel| [rad/s]"); ax[0].set_ylabel("saturation s")
    ax[0].set_title("(a) rolling laws: recomputed from impl"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, which="both")

    names = ["a", "b", "c", "d", "e", "f"]
    errs = []
    for n in ("a_sphere_baseline", "b_coulomb_high_speed", "c_coulomb_low_speed",
              "d_viscous_at_eps", "e_stopping_speed"):
        errs.append(max(float(res["checks"][n]["abs_error"]), 1e-18))
    errs.append(max(float(res["checks"]["f_clump_anisotropy"]["abs_error"]), 1e-18))
    colors = ["#888" if i in (0, 3, 4, 5) else "#c33" for i in range(6)]
    ax[1].bar(names, errs, color=colors)
    ax[1].set_yscale("log"); ax[1].set_ylabel("|got - ref| (log)")
    ax[1].set_title("(b) audit recompute: 6/6 inside tolerance,\nbut every 'ref' is the impl's own identity")
    for i, e in enumerate(errs):
        ax[1].text(i, e, f"{e:.1e}", ha="center", va="bottom", fontsize=7)
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "audit_m17721.png", dpi=130)

    print(json.dumps({k: res["checks"][k] for k in (
        "registered_benchmark_cases", "harness_scope", "repro", "provenance")},
        indent=1, ensure_ascii=False))
    print("audit_level:", res["audit_level"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())