"""Assemble the ice-beam anchor report from the saved simulation stages.

The expensive stages are cached as JSON so the assembly is deterministic and
re-runnable:

``raw/anchor_5mm.json``      rate-free anchor at the task discretisation (5 mm)
``raw/report_stage1.json``   crack cascades, strength scan, controls, refinement

Writes ``artifacts/ice-beam-bending/ice_beam_report.json`` including the
acceptance evaluation.  Figures and the Markdown document are rendered from that
file by ``scripts/render_ice_beam_figures.py`` and
``scripts/render_ice_beam_doc.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem import ice_beam_bending as ib  # noqa: E402

OUT = ROOT / "artifacts" / "ice-beam-bending"
RAW = OUT / "raw"


def midspan_comparison(anchor: dict, theory: ib.BeamTheory) -> dict:
    """Score a stored DEM deflection profile against one beam-theory variant."""
    length = theory.spec.length_m
    x = np.asarray(anchor["deflection"]["x_over_L"]) * length
    w_dem = np.asarray(anchor["deflection"]["w_dem_m"])
    h = theory.spec.thickness_m
    window = (x >= 0.10 * length) & (x <= 0.90 * length) & (np.abs(x - length / 2) > h)
    mid = len(x) // 2
    load = anchor["probe_load_N"]
    w_ts = theory.deflection(x, load, "timoshenko")
    w_eb = theory.deflection(x, load, "euler_bernoulli")
    return {
        "strain_state": theory.strain_state,
        "young_modulus_Pa": theory.young_modulus_Pa,
        "midspan_dem_m": float(w_dem[mid]),
        "midspan_timoshenko_m": float(w_ts[mid]),
        "midspan_euler_bernoulli_m": float(w_eb[mid]),
        "rel_error_midspan_vs_timoshenko": float(abs(w_dem[mid] / w_ts[mid] - 1.0)),
        "rel_error_midspan_vs_euler_bernoulli": float(abs(w_dem[mid] / w_eb[mid] - 1.0)),
        "rel_error_profile_max_vs_timoshenko": float(np.max(
            np.abs(w_dem[window] - w_ts[window]) / np.abs(w_ts[window]))),
        "rel_error_profile_max_vs_euler_bernoulli": float(np.max(
            np.abs(w_dem[window] - w_eb[window]) / np.abs(w_eb[window]))),
        "flexural_rigidity_ratio_dem_over_theory": float(w_ts[mid] / w_dem[mid]),
        "comparison_point_count": int(window.sum()),
    }


def main() -> int:
    anchor_raw = json.loads((RAW / "anchor_5mm.json").read_text(encoding="utf-8"))
    report = json.loads((RAW / "report_stage1.json").read_text(encoding="utf-8"))

    anchor = dict(anchor_raw["anchor"])
    anchor["stress"] = anchor_raw["stress"]
    spec = ib.IceBeamSpec()
    props = anchor_raw["properties"]
    theory_ps = ib.beam_theory(spec, props)
    theory_pn = ib.beam_theory(spec, props, plane_strain=True)
    # Stress metrics are recomputed from the stored layer table with the
    # documented definitions (interior = two layers inside each free surface,
    # errors normalised by the extreme-fibre stress).
    layers = anchor["stress"]["layers"]
    a = props["lattice_spacing_m"]
    h = spec.thickness_m
    interior = [(z >= 2.0 * a) and (z <= h - 2.0 * a) for z in
                [row["z_m"] for row in layers]]
    z_int = np.asarray([row["z_m"] for row, keep in zip(layers, interior) if keep])
    s_int = np.asarray([row["sigma_xx_dem_Pa"] for row, keep in zip(layers, interior) if keep])
    t_int = np.asarray([row["sigma_xx_theory_Pa"] for row, keep in zip(layers, interior) if keep])
    slope, intercept = np.polyfit(z_int, s_int, 1)
    theory_peak = abs(float(theory_ps.sigma_max(anchor["probe_load_N"])))
    moment_theory = float(theory_ps.moment(
        np.array([anchor["stress"]["section_moment_x_over_L"] * spec.length_m]),
        anchor["probe_load_N"])[0])
    anchor["stress"].update({
        "rel_error_profile_max_interior": float(np.max(np.abs(s_int - t_int)) / theory_peak),
        "rel_error_sigma_max_fit": float(abs(intercept - theory_peak) / theory_peak),
        "sigma_max_dem_fit_Pa": float(intercept),
        "fit_slope_Pa_per_m": float(slope),
        "theory_slope_Pa_per_m": float(-moment_theory / theory_ps.second_moment_m4),
        "rel_error_fit_slope": float(abs(slope + moment_theory
                                         / theory_ps.second_moment_m4)
                                     / abs(moment_theory / theory_ps.second_moment_m4)),
        "rel_error_sigma_max_section_moment": abs(
            anchor["stress"]["section_moment_dem_Nm"] / moment_theory - 1.0),
        "sigma_max_from_section_moment_theory_Pa":
            moment_theory * (spec.thickness_m / 2.0) / theory_ps.second_moment_m4,
        "interior_layer_count": int(len(z_int)),
        "interior_definition": ("layers at least two particle diameters inside each "
                                "free surface"),
        "error_normalisation": "extreme-fibre stress M c / I",
    })
    for row, keep in zip(layers, interior):
        row["interior"] = bool(keep)
        row["rel_error"] = (float(abs(row["sigma_xx_dem_Pa"] - row["sigma_xx_theory_Pa"])
                                  / theory_peak))
    anchor["deflection_plane_stress"] = midspan_comparison(anchor, theory_ps)
    anchor["deflection_plane_strain"] = midspan_comparison(anchor, theory_pn)

    report["primary"]["anchor"] = anchor
    report["primary"]["spec"] = anchor_raw["spec"]
    report["primary"]["model"] = anchor_raw["model"]
    report["primary"]["properties"] = props
    report["primary"]["theory_plane_stress"] = anchor_raw["theory_ps"]
    report["primary"]["theory_plane_strain"] = anchor_raw["theory_pn"]
    report["anchor"] = anchor

    table = report["resolution_table"]
    for row in table:
        if row["particle_diameter_m"] == 0.005:
            row.update({
                "rel_error_midspan_vs_timoshenko_plane_stress":
                    anchor["deflection_plane_stress"]["rel_error_midspan_vs_timoshenko"],
                "rel_error_midspan_vs_timoshenko_plane_strain":
                    anchor["deflection_plane_strain"]["rel_error_midspan_vs_timoshenko"],
                "flexural_rigidity_ratio":
                    anchor["deflection_plane_stress"]["flexural_rigidity_ratio_dem_over_theory"],
                "rel_error_sigma_max_fit": anchor["stress"]["rel_error_sigma_max_fit"],
                "rel_error_first_break_load":
                    anchor["first_failure"]["rel_error_first_break_load"],
                "rel_error_section_moment": anchor["stress"]["rel_error_section_moment"],
                "pcg_iterations": anchor["solution"]["iterations"],
            })
    points = [(1.0 / row["depth_particles"], row["flexural_rigidity_ratio"] - 1.0)
              for row in table if row["depth_particles"] >= 3]
    slope, intercept = np.polyfit([p[0] for p in points], [p[1] for p in points], 1)
    row10 = [row for row in table if row["particle_diameter_m"] == 0.01][0]
    row5 = [row for row in table if row["particle_diameter_m"] == 0.005][0]
    report["resolution_convergence"] = {
        "reference_particle_diameter_m": 0.01,
        "reference_error": row10["rel_error_midspan_vs_timoshenko_plane_stress"],
        "fine_error": row5["rel_error_midspan_vs_timoshenko_plane_stress"],
        "deflection_rel_change": abs(row5["flexural_rigidity_ratio"]
                                     / row10["flexural_rigidity_ratio"] - 1.0),
        "measure": ("relative change of the DEM/theory flexural-rigidity ratio between "
                    "d = 10 mm and d = 5 mm"),
        "bias_law_slope_times_a_over_h": float(slope),
        "bias_law_intercept": float(intercept),
    }
    quarter = json.loads((RAW / "quarter_10mm.json").read_text(encoding="utf-8"))
    report["quarter_section_check"] = quarter
    criteria = ib.evaluate_criteria(report)
    report["criteria"] = criteria
    report["verdict"] = "PASS" if criteria["all_pass"] else "FAIL"
    (OUT / "ice_beam_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"],
                      "passed": criteria["passed"], "total": criteria["total"],
                      "failed": criteria["failed"],
                      "bias_slope": report["resolution_convergence"][
                          "bias_law_slope_times_a_over_h"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())