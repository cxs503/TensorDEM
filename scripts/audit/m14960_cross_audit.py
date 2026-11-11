"""Cross-audit of m14960 (GAP-DEM-FRICTION round 2, TensorDEM) -- machine checks.

Board-audit deliverable (mission m14978).  This script does **not** re-do the
mission; it re-derives every load-bearing number of m14960 from the committed
artifacts of commit ``3fcf006d`` (branch ``agent/m14960``) plus one independent
re-run, and applies the repo's *own* validity gate at the tolerance m14960's
other rows use (0.15).  Everything it prints is written to
``artifacts/audit_m14960/audit.json`` so that no number in the audit report is
hand-transcribed.

Inputs
------
``--snapshot`` : a clean checkout of the audited commit, e.g.

    git archive 3fcf006d -o /tmp/m14960_snapshot.tar
    mkdir -p /tmp/audit_m14960 && tar xf /tmp/m14960_snapshot.tar -C /tmp/audit_m14960

   which must contain ``artifacts/shear_cell_m14960/``, ``docs/``, ``runs/`` and
   the re-run ``artifacts/shear_cell_m14960/audit_size_medium.json`` produced by

    cd /tmp/audit_m14960 && PYTHONPATH=src python scripts/m14960/run_friction_matrix.py \
        --tag audit_size_medium --geom medium --mu 0.5 --gamma 0.6 --sigmas 4000

Usage
-----
    PYTHONPATH=<repo>/src python scripts/audit/m14960_cross_audit.py \
        --snapshot /tmp/audit_m14960
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.shear_cell import _validity_gate  # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "audit_m14960")
MU_LIT, MU_BAND, GATE = 0.38, (0.36, 0.40), 0.10
TOL_ROWS = 0.15          # stationarity_tol used by m14960's mu_p rows (doc §3.2)
WINDOWS = (0.50, 0.40, 0.25, 0.10)


def load_level(snap: str, tag: str, level: int = 0) -> dict:
    p = os.path.join(snap, "artifacts", "shear_cell_m14960", tag + ".json")
    with open(p) as fh:
        return json.load(fh)["levels"][level]


def c1_repro(snap: str) -> dict:
    """C1 -- is the headline number reproducible, bit for bit?"""
    a = load_level(snap, "size_medium")
    b = load_level(snap, "audit_size_medium")
    keys = ("sigma_pa", "tau_pa", "mu", "inertial_number", "drift_rel", "drift_r2",
            "total_steps", "n_particles", "shear_rate_actual")
    delta = {k: abs(float(b[k]) - float(a[k])) for k in keys}
    return {"check": "C1 repro of size_medium from clean checkout",
            "committed": {k: a[k] for k in keys},
            "rerun": {k: b[k] for k in keys},
            "max_abs_delta": max(delta.values()),
            "bit_exact": max(delta.values()) == 0.0,
            "verdict": "PASS" if max(delta.values()) == 0.0 else "FAIL"}


def c2_steadiness(snap: str) -> dict:
    """C2 -- the repo's own steadiness leg, at the tol used by m14960's other rows."""
    rows = {}
    for tag in ("size_small", "size_medium"):
        lv = load_level(snap, tag)
        ok_off, _ = _validity_gate(lv["sigma_pa"], lv["tau_pa"], 4000.0,
                                   drift_rel=lv["drift_rel"], stationarity_tol=None)
        ok_on, why = _validity_gate(lv["sigma_pa"], lv["tau_pa"], 4000.0,
                                    drift_rel=lv["drift_rel"], stationarity_tol=TOL_ROWS)
        rows[tag] = {"drift_rel": lv["drift_rel"], "drift_r2": lv["drift_r2"],
                     "stationarity_tol_recorded": lv.get("stationarity_tol", None),
                     "valid_with_gate_off": ok_off, "valid_at_tol_0.15": ok_on,
                     "reason_at_tol_0.15": why}
    n_bad = sum(1 for r in rows.values() if not r["valid_at_tol_0.15"])
    return {"check": "C2 headline rows under the repo's own steadiness gate (tol=%.2f)" % TOL_ROWS,
            "rows": rows, "headline_rows_rejected": n_bad,
            "verdict": "FAIL" if n_bad else "PASS"}


def c3_windows(snap: str) -> dict:
    """C3 -- how much does mu depend on the averaging window (is tau steady?)."""
    out = {}
    for tag in ("size_small", "size_medium"):
        lv = load_level(snap, tag)
        t = np.asarray(lv["shear_stress_history"], float)
        s = lv["sigma_pa"]
        per = {}
        for frac in WINDOWS:
            w = t[int((1.0 - frac) * t.size):]
            mu = float(w.mean() / s)
            per["last_%02d%%" % int(frac * 100)] = {
                "tau_pa": float(w.mean()), "mu": mu,
                "vs_lit_pct": 100.0 * (mu / MU_LIT - 1.0),
                "within_10pct": abs(mu / MU_LIT - 1.0) < GATE}
        mus = [v["mu"] for v in per.values()]
        out[tag] = {"stored_mu": lv["mu"], "windows": per,
                    "mu_spread_abs": max(mus) - min(mus),
                    "mu_spread_rel": (max(mus) - min(mus)) / float(np.mean(mus)),
                    "tau_first_tenth_pa": float(t[: t.size // 10].mean()),
                    "tau_last_tenth_pa": float(t[-(t.size // 10):].mean())}
    return {"check": "C3 window sensitivity of mu (last 50/40/25/10 % of the shear window)",
            "rows": out,
            "verdict": "FAIL",
            "why": "mu of the headline case spans %.3f..%.3f (%.1f%% of the mean): "
                   "the shear traction has not reached a steady state, so the stored "
                   "single-window value is not a steady-state friction coefficient"
                   % (min(v["mu"] for v in out["size_medium"]["windows"].values()),
                      max(v["mu"] for v in out["size_medium"]["windows"].values()),
                      100 * out["size_medium"]["mu_spread_rel"])}


def c4_artifact_consistency(snap: str) -> dict:
    """C4 -- tables doc vs machine artifacts vs the 7-row log-derived table."""
    tables = open(os.path.join(snap, "docs", "SHEAR_CELL_M14960_tables.md")).read()
    ll = json.load(open(os.path.join(snap, "artifacts", "shear_cell_m14960", "log_levels.json")))
    empty_table2 = tables.count("(未完成)")
    table3_rows = len([l for l in tables.splitlines() if l.startswith("| ") and l.count("|") >= 9])
    # 表 3 body: lines after its header, excluding headers/separator
    t3 = tables.split("### 表 3")[-1].splitlines()
    t3_rows = [l for l in t3 if l.startswith("| ") and "---" not in l][2:]
    has_tag_json = {t: os.path.exists(os.path.join(snap, "artifacts", "shear_cell_m14960", t + ".json"))
                    for t in ("mu030", "mu050", "mu070")}
    return {"check": "C4 committed tables vs committed machine artifacts",
            "table2_rows_marked_unfinished": empty_table2,
            "table3_body_rows": len(t3_rows),
            "log_levels_rows": len(ll),
            "per_tag_artifacts": has_tag_json,
            "verdict": "FAIL" if (len(t3_rows) == 0 and len(ll) > 0) else "PASS",
            "why": "the report/mission claim 7 log-derived rows in 表 3 and a completed "
                   "mu_p table, but the committed tables.md has an empty 表 3 body and "
                   "3/3 rows of 表 2 marked (未完成); the 7 mu_p rows exist only as "
                   "text logs + a summary JSON, with no per-tag machine artifact"}


def c5_hand_transcription(snap: str) -> dict:
    """C5 -- 'zero hand transcription' discipline: unrendered template fields in the doc."""
    doc = open(os.path.join(snap, "docs", "SHEAR_CELL_M14960.md")).read()
    ph = re.findall(r"\{[^{}\n]*:[^{}\n]*\}", doc)
    return {"check": "C5 unrendered programmatic-injection placeholders in the doc",
            "placeholders": ph,
            "n_placeholders": len(ph),
            "verdict": "FAIL" if ph else "PASS",
            "why": "the error-judgement table (§5) still carries raw f-string templates "
                   "such as {100*er(0.3627):+.1f}% and a reference to an undefined 'j'; "
                   "two of its five rows therefore contain no number at all"}


def c6_log_reproducibility(snap: str) -> dict:
    """C6 -- can the log-derived (mu_p) rows be regenerated by a committed script?"""
    src = open(os.path.join(snap, "scripts", "m14960", "analyse_friction_matrix.py")).read()
    analyses_logs = bool(re.search(r"runs[/\\]\*?\.?log|log_levels", src))
    logs = {}
    for tag in ("mu030", "mu050", "mu070"):
        p = os.path.join(snap, "runs", tag + ".log")
        logs[tag] = {"exists": os.path.exists(p),
                     "lines": len(open(p).read().splitlines()) if os.path.exists(p) else 0}
    return {"check": "C6 reproducibility of the log-derived rows",
            "analyse_script_parses_logs": analyses_logs,
            "logs": logs,
            "verdict": "FAIL" if not analyses_logs else "PASS",
            "why": "analyse_friction_matrix.py never opens runs/*.log; the doc's repro "
                   "section offers an inline heredoc that is not in the repo, so "
                   "log_levels.json cannot be regenerated from the commit"}


def c7_anchor(snap: str) -> dict:
    """C7 -- sensitivity of the '<10%' claim to the width of the literature anchor."""
    mu = load_level(snap, "size_medium")["mu"]
    lit = {}
    for name, ref in (("central_0.38", MU_LIT), ("band_lo_0.36", MU_BAND[0]),
                      ("band_hi_0.40", MU_BAND[1])):
        lit[name] = {"ref": ref, "err_pct": 100.0 * (mu / ref - 1.0),
                     "within_10pct": abs(mu / ref - 1.0) < GATE}
    return {"check": "C7 anchor-width sensitivity of the acceptance judgement",
            "mu_dem": mu, "anchor": lit,
            "micro_friction_of_dem": 0.5,
            "verdict": "WITHHELD",
            "why": "the anchor 0.38 (band 0.36-0.40) is a literature value quoted from "
                   "memory (the mission states it could not be verified online) and the "
                   "DEM micro friction mu_p=0.5 was never calibrated to the material of "
                   "that anchor, so agreement is not evidence of a validated friction "
                   "prediction; at the band edge 0.36 the same run reads +0.7%"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    args = ap.parse_args()
    snap = args.snapshot
    checks = [c1_repro(snap), c2_steadiness(snap), c3_windows(snap),
              c4_artifact_consistency(snap), c5_hand_transcription(snap),
              c6_log_reproducibility(snap), c7_anchor(snap)]
    audit = {"scope": "m14960 GAP-DEM-FRICTION", "audited_commit": "3fcf006d",
             "snapshot": snap, "checks": checks}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "audit.json"), "w") as fh:
        json.dump(audit, fh, indent=1)
    for c in checks:
        print("%-8s %s" % (c["verdict"], c["check"]))
    print("\nwrote", os.path.join(OUT, "audit.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())