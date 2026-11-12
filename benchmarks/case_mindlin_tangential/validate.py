#!/usr/bin/env python3
"""Acceptance gate for CASE-DEM-MINDLIN-TANGENTIAL.

Re-reads the run artifacts (never the solver) and re-applies the acceptance
thresholds of the case:

    * micro-slip segment  : max |F_t - F_MD| / (mu F_n)   < 5 %
    * transition (delta~delta_s)                          < 5 %
    * gross-slip plateau  : |F_t - mu F_n| / (mu F_n)     < 1 %
    * initial slope       : k_t vs 8 G* a                 < 2 %
    * fitted delta_s      : vs 3 mu F_n/(2 k_t)           < 5 %
    * discrimination      : legacy linear law must deviate > 5 %

Usage:
    PYTHONPATH=src python benchmarks/case_mindlin_tangential/validate.py \
        [--out artifacts/mindlin_tangential]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

GATE_MICRO = 0.05
GATE_TRANSITION = 0.05
GATE_GROSS = 0.01
GATE_SLOPE = 0.02
GATE_DELTA_S = 0.05
GATE_DISCRIMINATION = 0.05


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/mindlin_tangential")
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    path = os.path.join(out, "results.json")
    if not os.path.isfile(path):
        print(f"FAIL: missing {path}")
        return 2
    res = json.load(open(path))
    manifest_path = os.path.join(out, "checksums.json")
    manifest = (json.load(open(manifest_path)) if os.path.isfile(manifest_path)
                else {})
    got = res["got_mindlin_partial_slip"]
    ref = res["reference"]["chain"]
    cs = res["comparison_legacy_cundall_strack"]

    checks = []

    def add(name, value, gate, ok, note=""):
        checks.append({"name": name, "value": value, "gate": gate, "pass": bool(ok),
                       "note": note})

    add("micro_slip_max_err", got["max_micro_slip_err"], GATE_MICRO,
        got["max_micro_slip_err"] < GATE_MICRO, "fraction of mu*Fn")
    add("transition_max_err", got["max_transition_err"], GATE_TRANSITION,
        got["max_transition_err"] < GATE_TRANSITION,
        "|delta-delta_s| <= 0.1 delta_s")
    add("gross_slip_max_err", got["max_gross_slip_err"], GATE_GROSS,
        got["max_gross_slip_err"] < GATE_GROSS, "plateau vs mu*Fn")
    add("initial_slope_rel_err", got["k_t_rel_err"], GATE_SLOPE,
        got["k_t_rel_err"] < GATE_SLOPE, "secant k_t vs 8 G* a")
    d_s_rel = abs(got["delta_s_fit_m"] - ref["delta_s_m"]) / ref["delta_s_m"]
    add("fitted_delta_s_rel_err", d_s_rel, GATE_DELTA_S, d_s_rel < GATE_DELTA_S,
        "least-squares fit vs 3 mu Fn/(2 k_t)")
    add("linear_law_deviation", cs["max_dev_vs_analytic_muN"], GATE_DISCRIMINATION,
        cs["max_dev_vs_analytic_muN"] > GATE_DISCRIMINATION,
        "legacy Cundall-Strack must differ by more than the gate (case discriminates)")

    # artifacts + checksums
    missing = []
    for rel in ["curves.csv", "run.log"] + res.get("figures", []):
        if not os.path.isfile(os.path.join(out, rel)):
            missing.append(rel)
    add("checksum_manifest_present", 0 if manifest else 1, 0, bool(manifest),
        "checksums.json")
    add("artifacts_present", len(missing), 0, not missing,
        "missing: " + ", ".join(missing) if missing else "all present")
    bad_sha = []
    for rel, digest in (manifest.get("sha256") or {}).items():
        p = os.path.join(out, rel)
        if os.path.isfile(p) and sha256(p) != digest:
            bad_sha.append(rel)
    add("sha256_match", len(bad_sha), 0, not bad_sha,
        "mismatched: " + ", ".join(bad_sha) if bad_sha else "all match")

    ok = all(c["pass"] for c in checks)
    for c in checks:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['name']:<26} "
              f"value={c['value']:.6g} gate={c['gate']:.6g}  {c['note']}")
    print(f"\n{sum(c['pass'] for c in checks)}/{len(checks)} checks pass -> "
          f"{'ACCEPT' if ok else 'REJECT'}")
    print(f"reference: {res['reference']['citation']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())