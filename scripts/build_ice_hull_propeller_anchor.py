#!/usr/bin/env python3
"""Build the ISO 19906 external-anchor ledger for ice-hull-propeller-threefield.

Writes ``benchmarks/anchors/ice-hull-propeller-threefield.json``.  Every number
in that file is injected here from code or from a machine check; nothing is
typed in by hand.  Re-running this script reproduces the artifact byte for byte
(no wall-clock fields), which is what makes the "recompute" criterion auditable.

Usage
-----
    python3 scripts/build_ice_hull_propeller_anchor.py            # write artifact
    python3 scripts/build_ice_hull_propeller_anchor.py --stdout    # print only

Evidence grading of the output
------------------------------
* conversion chain, dimensional checks, consistency checks, claim gate: REAL
  (closed-form arithmetic, residuals reported).
* the ISO anchors themselves: WITHHELD -- the ISO 19906:2019 primary text is not
  in this worktree, so the registered triples are unverified candidates and the
  artifact's verdict on the transcription criterion is an honest FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.threefield.hull_ice import HullIceCase  # noqa: E402
from tensordem.threefield.iso_anchor import (  # noqa: E402
    CaliberLevel,
    ChainPoint,
    convert_pressure_area,
    dimensional_homogeneity_checks,
    global_force_locator_candidate,
    internal_consistency_checks,
    pressure_area_candidate,
    recompute_residual,
    require_anchor,
    UnverifiedAnchorError,
)
from tensordem.threefield.resistance import HullGeometry, IceProperties  # noqa: E402

ARTIFACT = REPO / "benchmarks/anchors/ice-hull-propeller-threefield.json"
TASK_KEY = "paper-exp-ice-hull-propeller-threefield-1"
CASE_DOC = REPO / "docs/THREEFIELD_ICE_HULL_PROPELLER.md"

# The case is run as a unit-width section (see tensordem.threefield.hull_ice and
# docs/THREEFIELD_ICE_HULL_PROPELLER.md §2.1): the bow reaction is reported in
# N/m, so one metre of width is the normalisation the DEM actually uses.
UNIT_WIDTH_M = 1.0

# Registered (unverified) candidate triple for the ISO pressure-area law.
# Registered so the harness can be exercised and the gap is explicit; NOT usable
# for any claim until transcribed verbatim (see the claim gate).
CANDIDATE_EXPONENT = -0.3
CANDIDATE_C_REF_PA = 1.2e6
CANDIDATE_A_REF_M2 = 2.5


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # pragma: no cover - detached/no-git environments
        return "unknown"


def search_primary_text() -> dict:
    """Machine evidence for why the anchors are WITHHELD.

    Scans the worktree for any file that could hold the ISO 19906 text (name or
    content) and records what was found.  A remembered formula is not a source;
    this scan is what the artifact points at instead of an assertion.
    """
    skip_dirs = {".git", "artifacts", "results-dem3d-scale", "__pycache__", "runs"}
    doc_suffixes = {".pdf", ".txt", ".docx", ".epub", ".djvu", ".html"}
    scanned = 0
    name_hits: list[str] = []
    content_hits: list[dict] = []
    for path in sorted(REPO.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO)
        if any(part in skip_dirs for part in rel.parts):
            continue
        if path.resolve() == ARTIFACT.resolve():
            continue  # never scan our own output: keeps regeneration byte-stable
        scanned += 1
        if "19906" in path.name.lower() and path.suffix.lower() in doc_suffixes:
            name_hits.append(str(rel))
        if path.stat().st_size > 2_000_000:
            continue
        if path.suffix.lower() not in {".py", ".md", ".json", ".txt", ".toml", ".cfg"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:  # pragma: no cover
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if re.search(r"\bISO\s*19906|19906:20\d\d", line):
                content_hits.append({"path": str(rel), "line": lineno,
                                     "excerpt": line.strip()[:160]})
    return {
        "is_primary_text_present": bool(name_hits),
        "files_scanned": scanned,
        "file_name_hits": name_hits,
        "content_mention_hits": len(content_hits),
        "content_mentions_sample": content_hits[:40],
        "conclusion": (
            "no ISO 19906 document (pdf/txt/docx/...) exists in the worktree; the "
            "only occurrences are code/doc *mentions* of the standard, which are "
            "secondary and therefore cannot serve as a transcription source"
        ),
    }


def doc_declared_hull(path: Path) -> dict:
    """Parse the hull particulars the case document declares (cross-check only)."""
    text = path.read_text()
    match = re.search(
        r"L = ([\d.]+) m, B = ([\d.]+) m, T = ([\d.]+) m, waterline entry half-angle "
        r"([\d.]+)°, stem angle ([\d.]+)°, buttock angle ([\d.]+)°", text)
    if not match:
        return {"parsed": False}
    keys = ("length_m", "beam_m", "draught_m", "waterline_entry_angle_deg",
            "stem_angle_deg", "buttock_angle_deg")
    values = {key: float(group) for key, group in zip(keys, match.groups())}
    return {"parsed": True, "source": f"{path.relative_to(REPO)}:"
                                       f"{text[:match.start()].count(chr(10)) + 1}",
            "values": values}


def case_caliber() -> dict:
    """Case parameters, read from the code that actually runs them."""
    hull_ice = HullIceCase()
    ice = IceProperties()
    hull = HullGeometry()
    doc = doc_declared_hull(CASE_DOC)
    code_hull = hull.as_dict()
    doc_vs_code = None
    if doc.get("parsed"):
        diffs = {key: {"doc": value, "code_default": code_hull[key]}
                 for key, value in doc["values"].items()
                 if abs(value - code_hull[key]) > 1e-12}
        doc_vs_code = {
            "doc_source": doc["source"],
            "doc_values": doc["values"],
            "code_defaults": {key: code_hull[key] for key in doc["values"]},
            "mismatched_fields": diffs,
            "used_in_this_artifact": "code defaults (what the solver executes)",
            "consistency_ok": not diffs,
            "note": ("documentation and code defaults disagree; the calibre below "
                     "uses the executed values and the mismatch is flagged rather "
                     "than silently reconciled"),
        }
    thicknesses = sorted({round(float(hull_ice.thickness_m), 12),
                          round(float(ice.thickness_m), 12)})
    return {
        "ice_thickness_levels_m": thicknesses,
        "unit_width_m": UNIT_WIDTH_M,
        "unit_width_basis": (
            "unit-width normalisation: the hull-ice DEM is run as a unit-width "
            "section and reports the bow reaction in N/m "
            "(tensordem.threefield.hull_ice, docs/THREEFIELD_ICE_HULL_PROPELLER.md §2.1)"
        ),
        "dem_model_width_m": hull_ice.model_width_m,
        "dem_particle_radius_m": hull_ice.particle_radius_m,
        "dem_wedge_angle_deg": hull_ice.wedge_angle_deg,
        "dem_bow_speed_m_s": hull_ice.bow_speed_m_s,
        "reference_hull_code_defaults": code_hull,
        "doc_vs_code_hull_check": doc_vs_code,
        "source": {
            "HullIceCase": "src/tensordem/threefield/hull_ice.py",
            "IceProperties": "src/tensordem/threefield/resistance.py",
            "HullGeometry": "src/tensordem/threefield/resistance.py",
        },
    }


def caliber_levels(thicknesses: list[float], beam_m: float) -> list[CaliberLevel]:
    levels: list[CaliberLevel] = []
    for thickness in thicknesses:
        levels.append(CaliberLevel(
            label=f"unit_width_h{thickness:.2f}",
            width_m=UNIT_WIDTH_M,
            thickness_m=thickness,
            note="width of the contact patch equals the ice thickness (1:1 aspect, "
                 "single-floe bow contact), reported as a line load",
        ))
    for thickness in thicknesses:
        levels.append(CaliberLevel(
            label=f"full_beam_upper_bound_h{thickness:.2f}",
            width_m=beam_m,
            thickness_m=thickness,
            note="geometric upper bound: contact width taken as the full beam, so "
                 "the force is not a predicted bow load",
        ))
    return levels


def criteria(artifact: dict) -> list[dict]:
    anchors = artifact["anchors"]
    verified = [anchor for anchor in anchors if anchor["usable_for_claim"]]
    recompute = artifact["checks"]["recompute"]
    dim = artifact["checks"]["dimensional_homogeneity"]
    consistency = artifact["checks"]["internal_consistency"]
    guard = artifact["claim_guard"]
    return [
        {
            "criterion_id": "C1_verbatim_transcription",
            "requirement": (
                "quote + table/clause/page + value transcribed verbatim into the "
                "artifact (source value, no rewriting, no lossy rounding)"
            ),
            "status": "FAIL",
            "measured": {
                "anchors_registered": len(anchors),
                "anchors_verified": len(verified),
                "verified_fraction": len(verified) / len(anchors) if anchors else 0.0,
                "primary_text_present": artifact["source_availability"]
                ["is_primary_text_present"],
            },
            "why": (
                "the ISO 19906:2019 primary text is absent from the worktree (see "
                "source_availability); transcribing from recall would be fabrication, "
                "so the two numeric candidates carry quote=null and the global-force "
                "anchor carries no value at all"
            ),
        },
        {
            "criterion_id": "C2_conversion_recomputable",
            "requirement": "conversion chain landed as a script and recomputable, "
                           "recompute residual < 1e-9",
            "status": "PASS" if recompute["ok"] else "FAIL",
            "measured": {
                "worst_relative_residual": recompute["worst_relative_residual"],
                "tolerance": recompute["tolerance"],
                "script": "scripts/build_ice_hull_propeller_anchor.py",
                "chain_module": "src/tensordem/threefield/iso_anchor.py",
            },
            "why": ("the chain (ex, C_R, A_R) -> C_A -> p(A) -> q = p h -> F = p A is "
                    "re-evaluated from the artifact's own stored numbers and matches "
                    "within machine precision; conditional on the candidate triple, "
                    "which is why this criterion does not rescue C1"),
        },
        {
            "criterion_id": "C3_dimensional_and_consistency",
            "requirement": "dimensional homogeneity and internal consistency "
                           "machine checks pass",
            "status": "PASS" if (dim["all_ok"] and consistency["all_ok"]) else "FAIL",
            "measured": {
                "dimensional_equations_checked": len(dim["equations"]),
                "dimensional_all_ok": dim["all_ok"],
                "consistency_checks": len(consistency["checks"]),
                "consistency_all_ok": consistency["all_ok"],
            },
            "why": "exact rational exponent arithmetic plus numeric residuals; see "
                   "checks.dimensional_homogeneity and checks.internal_consistency",
        },
        {
            "criterion_id": "C4_unverified_not_cited",
            "requirement": "unverified candidate values may not be cited directly, "
                           "only used as cross-check references",
            "status": "PASS" if guard["all_ok"] else "FAIL",
            "measured": {
                "claims_permitted": guard["claims_permitted"],
                "unverified_anchors_blocked": guard["unverified_anchors_blocked"],
                "gate_raises_on_candidate": guard["gate_raises_on_candidate"],
            },
            "why": "require_anchor() refuses any anchor without a verbatim quote and "
                   "locator; the gate is exercised on every registered candidate and "
                   "the claims list is empty by construction",
        },
    ]


def build() -> dict:
    availability = search_primary_text()
    caliber = case_caliber()
    thicknesses = caliber["ice_thickness_levels_m"]
    beam = caliber["reference_hull_code_defaults"]["beam_m"]
    levels = caliber_levels(thicknesses, beam)

    anchors = [
        pressure_area_candidate(
            exponent=CANDIDATE_EXPONENT,
            c_ref_pa=CANDIDATE_C_REF_PA,
            a_ref_m2=CANDIDATE_A_REF_M2,
            candidate_id="A1_pressure_area_exponent_and_reference",
            locator_hint=("ISO 19906:2019 pressure-area annex (Annex A in the "
                          "2010 edition) -- clause/table/page NOT confirmed"),
            provenance_note="recall-prior (not a transcription; text unavailable)",
        ),
        global_force_locator_candidate(
            candidate_id="A2_design_global_ice_force",
            locator_hint=("ISO 19906:2019 global ice action clauses (Clause 14/15 in "
                          "the 2010 edition) -- whether a single tabulated design "
                          "global force exists at all is unconfirmed"),
        ),
    ]

    chain = convert_pressure_area(
        exponent=CANDIDATE_EXPONENT,
        c_ref_pa=CANDIDATE_C_REF_PA,
        a_ref_m2=CANDIDATE_A_REF_M2,
        levels=levels,
    )
    dim = {
        "equations": dimensional_homogeneity_checks(CANDIDATE_EXPONENT),
        "all_ok": None,  # filled below
    }
    dim["all_ok"] = all(item["homogeneous"] for item in dim["equations"])
    consistency = internal_consistency_checks(
        exponent=CANDIDATE_EXPONENT,
        c_ref_pa=CANDIDATE_C_REF_PA,
        a_ref_m2=CANDIDATE_A_REF_M2,
        levels=levels,
        chain=chain,
    )

    blocked: list[str] = []
    gate_raises = True
    for anchor in anchors:
        try:
            require_anchor(anchor)
        except UnverifiedAnchorError:
            blocked.append(anchor.anchor_id)
        else:
            gate_raises = False
            break

    artifact: dict = {
        "artifact": str(ARTIFACT.relative_to(REPO)),
        "task_key": TASK_KEY,
        "repository": "TensorDEM",
        "study": "ice-hull-propeller-threefield",
        "experiment": "exp-1 external anchor verification and calibre conversion",
        "generated_by": {
            "script": "scripts/build_ice_hull_propeller_anchor.py",
            "script_sha256": sha256_of(Path(__file__).resolve()),
            "chain_module": "src/tensordem/threefield/iso_anchor.py",
            "chain_module_sha256": sha256_of(
                REPO / "src/tensordem/threefield/iso_anchor.py"),
            "git_rev": git_revision(),
            "regenerate": "python3 scripts/build_ice_hull_propeller_anchor.py",
            "no_wall_clock_fields": True,
            "volatile_fields": [
                "source_availability.content_mentions_sample",
                "source_availability.content_mention_hits",
            ],
            "volatile_reason": (
                "these count/list every worktree file that merely *mentions* the "
                "standard, so they change when unrelated docs are added; everything "
                "else in this artifact is byte-reproducible"
            ),
        },
        "source_availability": availability,
        "case_caliber": caliber,
        "anchors": [anchor.as_dict() for anchor in anchors],
        "conversion_chain": {
            **chain,
            "steps": [
                {"step": 1, "formula": "C_A = C_R * A_R^(-ex)",
                 "dimension": chain["coefficient_dimension"],
                 "provenance": "closed form, arithmetic over registered inputs"},
                {"step": 2, "formula": "p(A) = C_A * A^ex", "dimension": "(M^1 L^-1 T^-2)",
                 "provenance": "pressure-area law as registered"},
                {"step": 3, "formula": "q = p * h", "dimension": "(M^1 T^-2)",
                 "provenance": "line load; the unit the unit-width DEM reports"},
                {"step": 4, "formula": "F = q * w = p * A", "dimension": "(M^1 L^1 T^-2)",
                 "provenance": "global force at the calibre area"},
            ],
            "caliber_levels": [{"label": level.label, "width_m": level.width_m,
                                "thickness_m": level.thickness_m,
                                "area_m2": level.area_m2, "note": level.note}
                               for level in levels],
        },
        "checks": {
            "dimensional_homogeneity": dim,
            "internal_consistency": consistency,
            "recompute": recompute_residual({"conversion_chain": chain}),
        },
        "claim_guard": {
            "policy": ("an anchor may enter a claim only with verification_status="
                       "VERIFIED plus a verbatim quote and a table/clause/page "
                       "locator; require_anchor() enforces it, so no unverified "
                       "candidate can leak into a comparison"),
            "unverified_anchors_blocked": blocked,
            "gate_raises_on_candidate": gate_raises,
            "claims_permitted": [],
            "all_ok": gate_raises and bool(blocked) and not any(
                anchor["usable_for_claim"] for anchor in
                [a.as_dict() for a in anchors]),
        },
        "downstream_gate": {
            "domain_comparison_allowed": False,
            "blocked_until": "C1 passes: anchors transcribed verbatim with locators",
            "why": ("comparing the DEM/LBM/thermal results with a recalled pressure-area "
                    "triple would measure agreement with the agent's memory, not with "
                    "the standard"),
        },
    }
    # recompute the chain from the artifact's own stored numbers (same content
    # the file will carry), then derive the per-criterion measured values.
    artifact["checks"]["recompute"] = recompute_residual(artifact)
    artifact["criteria"] = criteria(artifact)

    met = [item["criterion_id"] for item in artifact["criteria"]
           if item["status"] == "PASS"]
    unmet = [item["criterion_id"] for item in artifact["criteria"]
             if item["status"] != "PASS"]
    verdict = {
        "overall": "HONEST_FAIL" if unmet else "PASS",
        "criteria_met": len(met),
        "criteria_total": len(artifact["criteria"]),
        "met": met,
        "unmet": unmet,
        "headline": ("external anchors NOT verified: ISO 19906:2019 primary text is "
                     "absent from the worktree, so no verbatim quote/table/page can be "
                     "entered; the conversion + check machinery is in place and "
                     "self-consistent on the registered candidates"
                     if unmet else "all criteria met"),
        "gap": [
            "obtain the ISO 19906:2019 primary text (Annex A pressure-area; Clause "
            "14/15 global ice actions) and transcribe the exponent, reference "
            "pressure/area and the design global force with table/clause/page",
            "replace the recall-prior candidate triple with the transcribed values, "
            "then regenerate this artifact (one command) -- the checks and the "
            "calibre conversion need no change",
            "confirm whether a single tabulated design global force exists; if the "
            "standard only supplies an action format, register the format and the "
            "required ice-strength/action-factor tables instead of a number",
        ],
    }
    artifact["verdict"] = verdict
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", action="store_true", help="print, do not write")
    parser.add_argument("--out", default=str(ARTIFACT))
    args = parser.parse_args()

    artifact = build()
    payload = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.stdout:
        sys.stdout.write(payload)
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload)
    verdict = artifact["verdict"]
    print(f"wrote {out} ({len(payload)} bytes)")
    print(f"verdict={verdict['overall']} criteria={verdict['criteria_met']}/"
          f"{verdict['criteria_total']} unmet={verdict['unmet']}")
    print(f"recompute residual={artifact['checks']['recompute']['worst_relative_residual']:.3e}")
    print("anchors verified:",
          sum(1 for anchor in artifact["anchors"] if anchor["usable_for_claim"]),
          "/", len(artifact["anchors"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())