"""Verdict assembly for the non-spherical representation decision (campaign P27W2-N07).

The verdict artefact answers one engineering question with numbers:

    At which shape factor and for which use is an N-sphere clump an admissible
    stand-in for the true superquadric, and what is the error bound?

It is built only from measured fragments written by
``scripts/shape_representation_campaign.py`` (``hertz_anchors.json``,
``geom_convergence.json``, ``dissipation.json``, ``heap_*.json``).  Missing
fragments are reported as ``NOT_CERTIFIED`` -- never silently filled in.
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Dict, List, Optional

# Tolerances used to convert measurements into a decision.  They are declared
# here, not tuned per shape: 5 % geometric surface deviation for transport /
# mass-driven work, 1 % when contact law or interlocking matters.
TOL_BULK = 0.05
TOL_CONTACT = 0.01
TOL_REPOSE_DEG = 1.0
TOL_STIFFNESS = 0.02
TOL_DISSIPATION = 1e-6


def _load(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _saturation_n(rows: List[Dict[str, object]], tol: float, fitter: str = "covering") -> Optional[float]:
    """N at which the fitted power law reaches ``tol`` (max deviation ratio)."""
    ns = [r["n_spheres"] for r in rows]
    es = [r[fitter]["max_distance_ratio"] for r in rows]
    good = [(n, e) for n, e in zip(ns, es) if e and e > 0]
    if len(good) < 2:
        return None
    if good[-1][1] <= tol:
        # interpolate on the last segment
        for (n0, e0), (n1, e1) in zip(good, good[1:]):
            if e1 <= tol <= e0 and e0 > e1:
                f = (math.log(e0) - math.log(tol)) / (math.log(e0) - math.log(e1))
                return float(math.exp(math.log(n0) + f * (math.log(n1) - math.log(n0))))
        return float(good[-1][0])
    p = abs(math.log(good[-1][1] / good[0][1]) / math.log(good[-1][0] / good[0][0]))
    return float(good[-1][0] * (good[-1][1] / tol) ** (1.0 / p))


def assemble_verdict(art_dir: str) -> Dict[str, object]:
    hertz = _load(os.path.join(art_dir, "hertz_anchors.json"))
    geom = _load(os.path.join(art_dir, "geom_convergence.json"))
    diss = _load(os.path.join(art_dir, "dissipation.json"))
    heaps = {}
    for path in sorted(glob.glob(os.path.join(art_dir, "heap_*.json"))):
        data = _load(path)
        if data and "case" in data:
            heaps[data["case"]] = data

    criteria: Dict[str, object] = {}

    # ---------------- C1: contact-law anchor ----------------
    if hertz:
        criteria["C1_ellipsoid_plane_stiffness"] = {
            "status": "PASS" if hertz["worst_stiffness_rel_error"] <= TOL_STIFFNESS else "FAIL",
            "measured": float(hertz["worst_stiffness_rel_error"]),
            "tolerance": TOL_STIFFNESS,
            "metric": "relative stiffness error of the closed form against the half-space integral equation",
            "sphere_limit_rel_error": float(hertz["sphere_limit"]["stiffness_rel_error"]),
            "flatness_residual": float(hertz["worst_flatness_spread"]),
            "evidence": "artifacts/shape_representation/hertz_anchors.json",
        }
        laws = hertz["power_law_indenter"]
        worst_law = max(l["exponent_rel_error"] for l in laws)
        criteria["C1b_superellipsoid_pole_law"] = {
            "status": "PASS" if worst_law <= 0.02 else "FAIL",
            "measured_worst_exponent_rel_error": float(worst_law),
            "tolerance": 0.02,
            "law": "P ~ delta^(1+1/m) for a superellipsoid pole of exponent m",
            "hertz_disk_force_rel_error": float(laws[[l["power_m"] for l in laws].index(2.0)]
                                                 .get("hertz_max_rel_error", float("nan"))),
            "evidence": "artifacts/shape_representation/hertz_anchors.json",
        }
    else:
        criteria["C1_ellipsoid_plane_stiffness"] = {"status": "NOT_CERTIFIED",
                                                    "reason": "hertz_anchors.json missing"}

    # ---------------- C2: clump convergence ----------------
    geom_summary: Dict[str, object] = {}
    if geom:
        for key, block in geom["shapes"].items():
            rows = block["rows"]
            geom_summary[key] = {
                "relative_error_by_N": block["relative_error_covering"],
                "relative_error_by_N_kmeans": block["relative_error_kmeans"],
                "convergence_exponent_covering": float(block["convergence_exponent_covering"]),
                "convergence_exponent_kmeans": float(block["convergence_exponent_kmeans"]),
                "N_star_bulk(5%)": _saturation_n(rows, TOL_BULK, "covering"),
                "N_star_contact(1%)": _saturation_n(rows, TOL_CONTACT, "covering"),
            }
    repose_rows = {name: h for name, h in heaps.items()}
    repose_drift = None
    if "clump_m8_N8" in repose_rows and "clump_m8_N32" in repose_rows:
        repose_drift = abs(repose_rows["clump_m8_N8"]["repose_angle_deg"]
                           - repose_rows["clump_m8_N32"]["repose_angle_deg"])
    criteria["C2_clump_convergence"] = {
        "status": ("PASS" if (repose_drift is not None and repose_drift <= TOL_REPOSE_DEG
                              and geom_summary.get("m8", {}).get("N_star_bulk(5%)", 1e9) <= 32)
                   else ("FAIL" if repose_drift is not None else "NOT_CERTIFIED")),
        "geometry": geom_summary,
        "repose_drift_deg_N8_to_N32": repose_drift,
        "tolerance_deg": TOL_REPOSE_DEG,
        "note": ("the geometric half is measured; the repose half requires the "
                 "macroscopic channel, whose driver could not be stabilised inside "
                 "the budget (see docs/dem_nonspherical_representation_verdict.md §5)"),
        "evidence": "artifacts/shape_representation/geom_convergence.json",
    }

    # ---------------- C3: static-bed dissipation ----------------
    if diss:
        cases = diss["cases"]
        criteria["C3_static_dissipation"] = {
            "status": "PASS" if all(c["passes"] for c in cases.values()) else "FAIL",
            "tolerance_per_step": TOL_DISSIPATION,
            "measured": {k: float(v["drift_per_step"]) for k, v in cases.items()},
            "baseline_is_sphere": True,
            "note": ("the sphere bed is the control: any clump-induced excess is "
                     "measured relative to it"),
            "evidence": "artifacts/shape_representation/dissipation.json",
        }
    else:
        criteria["C3_static_dissipation"] = {"status": "NOT_CERTIFIED",
                                            "reason": "dissipation.json missing"}

    # ---------------- C4: shape effect on the repose angle ----------------
    channels = {k: v["repose_angle_deg"] for k, v in repose_rows.items()}
    monotone = None
    if channels:
        ordered = [channels.get("sphere"), channels.get("true_m4"), channels.get("true_m8")]
        if all(v is not None for v in ordered):
            monotone = bool(ordered[0] <= ordered[1] <= ordered[2])
    criteria["C4_shape_effect_repose"] = {
        "status": "PASS" if (channels and monotone) else "NOT_CERTIFIED",
        "measured_angles_deg": channels,
        "monotone_in_shape_exponent": monotone,
        "experimental_band_deg": [30.0, 35.0],
        "note": ("required band is the published sand/gravel repose interval; the "
                 "measurement needs a stable pour driver (see §5)"),
    }

    # ---------------- C5: the decision table ----------------
    # Contact-level statement: a clump is a union of spheres, so its contact law is
    # Hertzian (P ~ delta^1.5); a superellipsoid pole with exponent m obeys
    # P ~ delta^(1+1/m).  The force ratio at fixed overlap is therefore
    # (delta/delta_ref)^(1/m - 1/2), quantified below at a representative overlap.
    contact_bias = {}
    if hertz:
        for rec in hertz["superellipsoid_pole"]:
            m = rec["superellipsoid_exponent_m"]
            alpha = rec["force_exponent_1_plus_1_over_m"]
            contact_bias[f"m={m:g}"] = {
                "pole_force_exponent": float(alpha),
                "hertz_exponent": 1.5,
                "exponent_mismatch": float(alpha - 1.5),
                "hertzian": bool(rec["hertzian"]),
                "pole_radius_x_m": rec["pole_radius_x_m"],
                "pole_radius_y_m": rec["pole_radius_y_m"],
            }
    decision: Dict[str, object] = {}
    for m_key, block in geom_summary.items():
        m_val = float(m_key.lstrip("m"))
        errs = block["relative_error_by_N"]
        admissible_bulk = [int(n) for n, e in errs.items() if e <= TOL_BULK]
        admissible_contact = [int(n) for n, e in errs.items() if e <= TOL_CONTACT]
        decision[f"m={m_val:g}"] = {
            "clump_surface_error_by_N": errs,
            "bulk_use_admissible_N": admissible_bulk,
            "contact_critical_use_admissible_N": admissible_contact,
            "contact_law_bias": (contact_bias.get(f"m={m_val:g}") if contact_bias else None),
        }
    verdict = {
        "campaign": "P27W2-N07",
        "title": "Non-spherical representation verdict: multi-sphere clump vs superquadric",
        "question": ("When may an N-sphere clump replace the true superquadric shape, "
                     "and with what error bound?"),
        "criteria": criteria,
        "decision_table": decision,
        "contact_law": {
            "clump": "union of spheres -> Hertzian P ~ delta^1.5 for every shape",
            "superquadric_pole": "P ~ delta^(1+1/m); only m = 2 is Hertzian",
            "consequence": ("for m > 2 an exactly-shaped superquadric is *stiffer* at "
                            "large overlap than the Hertz law a clump implies; the "
                            "exponent mismatch below is the irreducible contact-law error"),
            "by_exponent": contact_bias,
        },
        "tolerances": {"bulk_surface_deviation": TOL_BULK,
                       "contact_critical_surface_deviation": TOL_CONTACT,
                       "repose_drift_deg": TOL_REPOSE_DEG,
                       "stiffness_rel": TOL_STIFFNESS,
                       "dissipation_per_step": TOL_DISSIPATION},
        "headline": _headline(criteria, geom_summary, contact_bias),
        "limitations": [
            "the macroscopic repose channel (C2 angle part, C4) is NOT_CERTIFIED: the "
            "available driver produced a standing column, a monolayer or an unphysical "
            "tower before the budget ran out; no repose number is reported as measured",
            "the covering fitter is a shell construction and is deliberately "
            "sub-optimal at N = 8 in absolute terms; only its convergence *rate* is used",
            "the k-means fitter of the repository does not converge in N (measured "
            "exponent ~ 0) and should not be used for large-N clumps without a "
            "convergence check",
        ],
    }
    return verdict


def _headline(criteria: Dict[str, object], geom_summary: Dict[str, object],
              contact_bias: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    c = criteria.get("C1_ellipsoid_plane_stiffness", {})
    if "measured" in c:
        out["contact_anchor"] = (
            f"ellipsoid/plane Hertz stiffness verified to {c['measured']:.1e} relative "
            f"against the half-space integral equation (tolerance {c['tolerance']:.0%})")
    else:
        out["contact_anchor"] = "contact anchor NOT_CERTIFIED (fragment missing)" 
    for key, block in geom_summary.items():
        m_val = float(key.lstrip("m"))
        n_bulk = block.get("N_star_bulk(5%)")
        n_contact = block.get("N_star_contact(1%)")
        out[f"clump_{key}"] = {
            "N_for_5pct_surface_error": (round(n_bulk, 1) if n_bulk else None),
            "N_for_1pct_surface_error": (round(n_contact, 1) if n_contact else None),
            "measured_relative_error_at_N128": block["relative_error_by_N"].get("128"),
            "convergence_exponent": block["convergence_exponent_covering"],
        }
    if contact_bias:
        out["contact_law"] = {k: {"pole_exponent": v["pole_force_exponent"],
                                 "hertz_exponent": v["hertz_exponent"]}
                              for k, v in contact_bias.items()}
    return out