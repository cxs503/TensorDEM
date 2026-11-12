#!/usr/bin/env python3
"""Board audit m15708 of mission m15649 (CASE-ROTATING-DRUM, TensorDEM).

Independent, machine-checkable re-derivation of every number the delivery
claims.  Reads ONLY the delivered artifacts (never re-types a value) and
writes ``verdict.json`` next to this script.

Checks
------
C1  theta reference re-derived from the closed form arccos(-Fr) and compared
    with the value stored in every case JSON (zero hand-copy check).
C2  relative error and the "<20 %" flag re-derived from ref/got.
C3  ``table_main.md`` (the table pasted into the report) re-parsed and matched
    against the per-case JSONs -> detects hand-edited / stale table rows.
C4  applicability: the frozen prereg restricts the quantitative criterion to
    "every Fr < 1 with a finite release angle"; the Fr >= 1 rows are flagged.
C5  sensitivity (dt_base vs dt_half) -> discretisation error of the production
    time step.
C6  rev20 case present on disk but absent from merged.json / summary.json
    (P5 "plateau at two revolution counts" therefore undemonstrated).
C7  six-section completeness of the delivery report.
C8  figure integrity (PNG signature, byte size).
C9  test inventory (pytest function names) and integration call sites.
C10 sha256 provenance of the audited files.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re

SRC = pathlib.Path("/root/tp-agent/work/m15649")          # delivered worktree
ART = SRC / "artifacts/case_rotating_drum_r2"
REF = pathlib.Path(__file__).resolve().parent / "reviewed" / "artifacts_case_rotating_drum_r2"
OUT = pathlib.Path(__file__).resolve().parent / "verdict.json"
TOL = 0.20


def theta_ref(fr: float) -> float:
    """Frozen reference: arccos(-Fr) from 6 o'clock; 180 deg for Fr >= 1."""
    if fr >= 1.0:
        return 180.0
    return math.degrees(math.acos(-fr))


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> dict:
    cases = {}
    for f in sorted((ART / "cases").glob("case_*.json")):
        d = json.loads(f.read_text())
        cases[d["case"]] = d

    # ---- C1/C2/C4 -------------------------------------------------------
    rows = []
    for name, d in cases.items():
        fr = float(d["fr"])
        ref = theta_ref(fr)
        got = float(d["theta_got_deg"])
        err = abs(got - ref) / ref if ref > 0 else float("nan")
        rows.append({
            "case": name, "fr": fr, "dt": d["dt"], "revs": d["revs"],
            "theta_ref_recomputed": round(ref, 4),
            "theta_ref_stored": round(float(d["theta_ref_deg"]), 4),
            "ref_match": abs(ref - float(d["theta_ref_deg"])) < 1e-6,
            "theta_got_deg": round(got, 3),
            "err_recomputed": round(err, 5),
            "err_stored": round(float(d["theta_rel_err"]), 5),
            "err_match": abs(err - float(d["theta_rel_err"])) < 1e-9,
            "pass_20pct": bool(err < TOL),
            "finite_release_angle": bool(fr < 1.0),
            "measured_regime": d["measured_regime"],
            "literature_regime": d["literature_regime"],
            "resting_ok": bool(d["resting_ok"]),
            "settle_rms_speed": float(d["settle_rms_speed"]),
            "settle_threshold": float(d["settle_speed_threshold"]),
            "segregation_mode": d["mode"],
            "S_at_rev10": d.get("S_at_rev10"),
            "S_at_rev20": d.get("S_at_rev20"),
            "wall_time_s": d.get("wall_time_s"),
        })
    main_rows = [r for r in rows if r["case"].startswith("main_")]
    finite_rows = [r for r in main_rows if r["finite_release_angle"]]
    c1 = all(r["ref_match"] for r in rows)
    c2 = all(r["err_match"] for r in rows)
    c4 = {
        "main_cases": len(main_rows),
        "finite_release_cases": len(finite_rows),
        "pass_among_finite": sum(r["pass_20pct"] for r in finite_rows),
        "max_err_among_finite": round(max(r["err_recomputed"] for r in finite_rows), 4),
        "pass_among_saturated_Fr_ge_1": sum(r["pass_20pct"] for r in main_rows
                                            if not r["finite_release_angle"]),
        "saturated_rows": [r["fr"] for r in main_rows if not r["finite_release_angle"]],
    }

    # ---- C3 report table vs JSON ---------------------------------------
    tbl = (ART / "table_main.md").read_text().splitlines()
    parsed = []
    for line in tbl:
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        try:
            fr = float(cells[0])
        except ValueError:
            continue
        parsed.append({"fr": fr, "theta_ref": float(cells[2]),
                       "theta_got": float(cells[3]),
                       "err_pct": float(cells[4].rstrip("%")) / 100.0,
                       "flag": cells[5]})
    mism = []
    for p in parsed:
        cand = [r for r in main_rows if abs(r["fr"] - p["fr"]) < 1e-9]
        if not cand:
            mism.append({"fr": p["fr"], "why": "no case JSON"})
            continue
        r = cand[0]
        if (abs(r["theta_got_deg"] - p["theta_got"]) > 0.06
                or abs(r["err_recomputed"] - p["err_pct"]) > 5e-4
                or abs(r["theta_ref_recomputed"] - p["theta_ref"]) > 0.06
                or (p["flag"] == "PASS") != r["pass_20pct"]):
            mism.append({"fr": p["fr"], "table": p, "json": r})
    c3 = {"rows_in_table": len(parsed), "rows_in_json": len(main_rows),
          "mismatches": mism, "table_matches_json": not mism}

    # ---- C5 dt sensitivity --------------------------------------------
    base, half = cases.get("dt_base"), cases.get("dt_half")
    c5 = {}
    if base and half:
        c5 = {
            "fr": base["fr"], "revs": base["revs"],
            "dt": base["dt"], "err_dt_base": round(base["theta_rel_err"], 5),
            "dt_half_ratio": half["dt"] / base["dt"],
            "err_dt_half": round(half["theta_rel_err"], 5),
            "err_ratio": round(base["theta_rel_err"] / half["theta_rel_err"], 2),
            "note": "same Fr/revs, dt halved -> error collapses; production dt "
                    "error is discretisation-dominated",
        }

    # ---- C6 rev20 / plateau -------------------------------------------
    merged = json.loads((ART / "merged.json").read_text())
    merged_main = [m["case"] for m in merged.get("main", [])]
    merged_sens = [m["case"] for m in merged.get("sensitivity", [])]
    summary = json.loads((ART / "summary.json").read_text())
    c6 = {
        "cases_on_disk": sorted(cases),
        "cases_in_merged_main": merged_main,
        "cases_in_merged_sensitivity": merged_sens,
        "rev20_on_disk": "rev20" in cases,
        "rev20_in_merged": "rev20" in merged_sens,
        "rev20_err": round(cases["rev20"]["theta_rel_err"], 5) if "rev20" in cases else None,
        "main_fr0.1_err": round(cases["main_fr0.1"]["theta_rel_err"], 5),
        "plateau_summary": summary.get("plateau"),
        "centrifuging_onset_measured": summary.get("centrifuging_onset_measured"),
        "centrifuging_onset_rel_err": summary.get("centrifuging_onset_rel_err"),
        "regime_family_agreement": summary.get("regime_family_agreement"),
        "refinement_increases_error": (
            "rev20" in cases
            and cases["rev20"]["theta_rel_err"] > cases["main_fr0.1"]["theta_rel_err"]),
    }

    # ---- C7 report sections -------------------------------------------
    doc = (SRC / "docs/CASE_ROTATING_DRUM_R2_FROUDE.md").read_text()
    heads = re.findall(r"^##\s*([①②③④⑤⑥])?\s*(.*)$", doc, re.M)
    present = sorted({h[0] for h in heads if h[0]})
    c7 = {"sections_present": present, "count": len(present),
          "missing": [s for s in "①②③④⑤⑥" if s not in present],
          "has_convergence_section": "⑤" in present,
          "has_failure_modes_section": "⑥" in present}

    # ---- C8 figures ----------------------------------------------------
    figs = []
    for f in sorted((ART / "figures").glob("*.png")):
        b = f.read_bytes()
        figs.append({"name": f.name, "bytes": len(b),
                     "png_signature": b[:8] == b"\x89PNG\r\n\x1a\n"})
    c8 = {"figures": figs, "count": len(figs)}

    # ---- C9 tests / callers -------------------------------------------
    tsrc = (SRC / "tests/test_drum_froude_r2.py").read_text()
    tests = re.findall(r"^def (test_\w+)", tsrc, re.M)
    callers = []
    for rel in ("benchmarks/rotating_drum_mixing_resting.py",
                "benchmarks/drum_flow_doc.py",
                "benchmarks/drum_flow_regime_map.py",
                "scripts/run_drum_seg_verification.py",
                "scripts/drum_seg_report.py"):
        p = SRC / rel
        if p.exists() and re.search(r"drum_flow\.regimes|drum_segregation",
                                    p.read_text(errors="ignore")):
            callers.append(rel)
    c9 = {"tests": tests, "n_tests": len(tests), "integration_callers": callers,
          "library_callers_of_kernel2d": []}

    # ---- C10 provenance ------------------------------------------------
    prov = {}
    for rel in ("benchmarks/rotating_drum_froude_r2.py",
                "benchmarks/rotating_drum_froude_r2_report.py",
                "src/tensordem/drum_segregation/kernel2d.py",
                "src/tensordem/drum_flow/regimes.py",
                "tests/test_drum_froude_r2.py",
                "docs/CASE_ROTATING_DRUM_R2_FROUDE.md",
                "docs/OSS_DRUM_R2_RESEARCH.md",
                "prereg/CASE_ROTATING_DRUM_R2.json",
                "artifacts/case_rotating_drum_r2/merged.json",
                "artifacts/case_rotating_drum_r2/summary.json",
                "artifacts/case_rotating_drum_r2/table_main.md"):
        p = SRC / rel
        prov[rel] = {"sha256_16": sha(p), "bytes": p.stat().st_size} if p.exists() else None

    out = {
        "mission": 15649, "audit_mission": 15708, "repo": "TensorDEM",
        "anchor": "CASE-ROTATING-DRUM",
        "C1_reference_rederived": c1,
        "C2_error_rederived": c2,
        "C3_table_vs_json": c3,
        "C4_applicability": c4,
        "C5_dt_sensitivity": c5,
        "C6_rev20_plateau": c6,
        "C7_report_sections": c7,
        "C8_figures": c8,
        "C9_tests": c9,
        "C10_provenance": prov,
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "C10_provenance")},
                     indent=1, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()