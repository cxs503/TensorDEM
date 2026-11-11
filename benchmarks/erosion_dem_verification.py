"""Verification driver for the DEM-side erosion face (node p176w2-n11).

Machine-checkable criteria (all numbers below are *computed*, none typed in):

  C1  Finnie's angle function peaks inside [15, 25] deg (analytic root vs a
      90001-point numeric scan) and the **stream** angle curve -- the aggregate
      of trajectory-measured impacts -- peaks inside the same window; the two
      published angle curves (Finnie cutting, Oka plastic) are mutually
      adjudicated on a self-similar band around their peaks (max relative
      difference < 15 %), with a band-sensitivity table and the honest
      model-class divergence outside it.
  C2  velocity exponent of the stream, fitted from >= 5 speed samples, lies in
      [2.1, 2.5] and within 10 % of the literature value of the material.
  C3  stream erosion rate vs the DNV-RP-O501 ductile design rate per operating
      point (>= 5 points), after one global anchor fitted in log space:
      relative deviation < 15 % at every point, each point listed.
  C4  mass-loss ledger closure: depth-field integral vs deposited volume,
      removed mass vs crater geometry, strictly non-negative masses, and the
      removal efficiency bound (no more material than the impact energy can
      pay for) -- residual < 5 %.
  C5  double monotonicity: E increasing in impact speed (>= 5 points) and
      decaying monotonically from the angle peak to 90 deg (>= 5 points).
  C6  negative controls: zero impacts or zero speed give E exactly 0
      (< 1e-12), and no crater volume at zero normal velocity.
  C7  decoupling from the breakage module: identity fragmentation leaves every
      erosion number bitwise identical (residual difference < 1e-6) and the
      erosion core imports no comminution module (AST check).

Run:  python benchmarks/erosion_dem_verification.py [--outdir DIR] [--quick]
Writes results.json + figures into artifacts/p176w2-n11/.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensordem.erosion import (  # noqa: E402
    LITERATURE_VELOCITY_EXPONENTS,
    finnie_angle_function,
    finnie_peak_angle_deg,
    finnie_peak_angle_deg_numeric,
    oka_angle_function,
    oka_peak_angle_deg,
    oka_peak_angle_deg_numeric,
)
from tensordem.erosion.materials import MATERIALS  # noqa: E402
from tensordem.erosion.models import (  # noqa: E402
    dnv_ductile_erosion_ratio,
    dnv_ductile_wall_loss_rate,
)
from tensordem.erosion_dem import (  # noqa: E402
    CASE_LAW,
    CASE_MATERIAL,
    StreamConfig,
    aggregate_exponent,
    angle_sweep,
    erosion_core_imports_comminution,
    fragmentation_hook,
    indentation_depth_m,
    monotonicity,
    peak_angle_from_samples,
    run_impact_stream,
    single_impact_crater,
    sphere_cap_depth_from_volume,
    sphere_cap_radius,
    sphere_cap_volume,
    velocity_sweep,
)
from tensordem.erosion_dem.crater import OkaAnchoredLaw  # noqa: E402
from tensordem.erosion_dem.contact import (  # noqa: E402
    JointImpactKinematics,
    oblique_restitution,
    plastic_normal_restitution,
    yield_velocity_m_s,
)

DNV_PAIR = {
    "aluminium_soft_liner": {"k": 1.3e-9, "n": 2.4,
                             "provenance": "DNV-RP-O501 ductile pair for aluminium "
                                           "alloys as tabulated (ASSUMED) in "
                                           "tensordem.erosion.materials; only n "
                                           "enters the shape comparison, K is "
                                           "absorbed by the fitted anchor"},
    "carbon_steel": {"k": 2.0e-9, "n": 2.6,
                     "provenance": "DNV-RP-O501 ductile steel/sand pair (documented)"},
}

ANGLE_GRID_DEG = [2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0]
SPEED_GRID_M_S = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0]
DNV_SPEED_GRID_M_S = [20.0, 25.0, 30.0, 35.0, 40.0]
N_PARTICLES = 1500


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def rel_dev(a, b) -> float:
    return abs(a - b) / abs(b) if b != 0.0 else math.inf


def self_similar_shape_deviation(hv_gpa: float, band: float) -> float:
    """Max relative difference of the two normalised angle curves, peak-scaled.

    Both curves are normalised to their own maximum and evaluated on
    ``alpha / alpha_peak in [1 - band, 1 + band]``: the neighbourhood of the
    peak where both published models describe the same (cutting-dominated)
    physics.  ``band`` is swept in the report to show the sensitivity.
    """
    a = np.linspace(0.0, 90.0, 9001)
    f = np.asarray(finnie_angle_function(a), float)
    g = np.asarray(oka_angle_function(a, hv_gpa), float)
    f /= f.max()
    g /= g.max()
    fp = finnie_peak_angle_deg()
    op = float(oka_peak_angle_deg(hv_gpa))
    t = np.linspace(1.0 - band, 1.0 + band, 401)
    fa = np.interp(t * fp, a, f)
    ga = np.interp(t * op, a, g)
    d = np.abs(fa - ga) / np.maximum(np.maximum(fa, ga), 1e-12)
    return float(d.max())


def anchor_fit(dnv_rates, stream_unit_rates) -> float:
    """Log-space least-squares anchor of one global constant."""
    ln_d = np.log(np.asarray(dnv_rates, float))
    ln_s = np.log(np.asarray(stream_unit_rates, float))
    return float(math.exp(float(np.mean(ln_d - ln_s))))


# ---------------------------------------------------------------------------
# C1: angle functions
# ---------------------------------------------------------------------------
def section_c1() -> dict:
    a = np.linspace(0.0, 90.0, 90001)
    f = np.asarray(finnie_angle_function(a), float)
    peak_analytic = float(finnie_peak_angle_deg())
    peak_numeric = float(a[int(np.argmax(f))])
    oka_peaks = {
        name: {
            "hv_gpa": mat.hardness_hv_gpa,
            "peak_deg_analytic": float(oka_peak_angle_deg(mat.hardness_hv_gpa)),
            "peak_deg_numeric": float(oka_peak_angle_deg_numeric(mat.hardness_hv_gpa)),
        }
        for name, mat in MATERIALS.items()
        if mat.material_class == "ductile"
    }
    oka_peaks[CASE_MATERIAL.name] = {
        "hv_gpa": CASE_MATERIAL.hardness_hv_gpa,
        "peak_deg_analytic": float(oka_peak_angle_deg(CASE_MATERIAL.hardness_hv_gpa)),
        "peak_deg_numeric": float(oka_peak_angle_deg_numeric(CASE_MATERIAL.hardness_hv_gpa)),
    }
    stream = angle_sweep(
        ANGLE_GRID_DEG,
        cfg=StreamConfig(n_particles=N_PARTICLES),
        speed_mean_m_s=30.0,
    )
    stream_peak = peak_angle_from_samples(
        stream["launch_angles_deg"], stream["erosion_ratios_kg_kg"]
    )
    band_table = {
        f"{int(b * 100)}pct": {
            "hv_case_0.50": self_similar_shape_deviation(0.50, b),
            "hv_1.07": self_similar_shape_deviation(1.07, b),
            "hv_2.00": self_similar_shape_deviation(2.00, b),
        }
        for b in (0.20, 0.30, 0.40, 0.50, 0.60)
    }
    shape_dev = band_table["40pct"]["hv_case_0.50"]
    # honest disclosure of the structural divergence at the ends of the range
    divergence = {}
    for ang in (0.0, 10.0, 21.61, 45.0, 60.0, 90.0):
        fv = float(finnie_angle_function(ang)) / float(f.max())
        gv = float(oka_angle_function(ang, CASE_MATERIAL.hardness_hv_gpa)) / float(
            np.max(oka_angle_function(np.linspace(0, 90, 9001), CASE_MATERIAL.hardness_hv_gpa))
        )
        divergence[f"{ang:g}"] = {"finnie_normalised": fv, "oka_normalised": gv}
    return {
        "finnie_peak_deg_analytic": peak_analytic,
        "finnie_peak_deg_numeric": peak_numeric,
        "finnie_analytic_numeric_error": rel_dev(peak_analytic, peak_numeric),
        "finnie_peak_in_window": bool(15.0 <= peak_analytic <= 25.0),
        "oka_peaks_by_material": oka_peaks,
        "oka_peak_in_window_case_material": bool(
            15.0 <= oka_peaks[CASE_MATERIAL.name]["peak_deg_analytic"] <= 25.0
        ),
        "stream": stream,
        "stream_peak": stream_peak,
        "stream_peak_in_window": bool(15.0 <= stream_peak["peak_angle_deg"] <= 25.0),
        "shape_band_sensitivity": band_table,
        "shape_deviation_peak_band_40pct": shape_dev,
        "shape_agreement_under_15pct": bool(shape_dev < 0.15),
        "structural_divergence_table": divergence,
        "pass": bool(
            15.0 <= peak_analytic <= 25.0
            and rel_dev(peak_analytic, peak_numeric) < 1e-3
            and 15.0 <= stream_peak["peak_angle_deg"] <= 25.0
            and shape_dev < 0.15
        ),
    }


# ---------------------------------------------------------------------------
# C2: velocity exponent
# ---------------------------------------------------------------------------
def section_c2() -> dict:
    sweep = velocity_sweep(
        SPEED_GRID_M_S,
        cfg=StreamConfig(n_particles=N_PARTICLES),
        angle_mean_deg=90.0,
    )
    fit = aggregate_exponent(sweep["launch_speeds_m_s"], sweep["erosion_ratios_kg_kg"])
    lit, lit_source = LITERATURE_VELOCITY_EXPONENTS[CASE_MATERIAL.name] if CASE_MATERIAL.name in LITERATURE_VELOCITY_EXPONENTS else (2.2, "Oka et al. (2005) k2 = 2.2, material independent")
    law_k = CASE_LAW.velocity_exponent
    in_window = bool(2.1 <= fit["exponent"] <= 2.5)
    err_lit = rel_dev(fit["exponent"], lit)
    # a second, independent sample set at 45 deg (off-normal) to show the fit is
    # not an artefact of the normal-incidence geometry
    sweep45 = velocity_sweep(
        SPEED_GRID_M_S,
        cfg=StreamConfig(n_particles=N_PARTICLES),
        angle_mean_deg=45.0,
    )
    fit45 = aggregate_exponent(sweep45["launch_speeds_m_s"], sweep45["erosion_ratios_kg_kg"])
    return {
        "speeds_m_s": sweep["launch_speeds_m_s"],
        "erosion_ratios_kg_kg": sweep["erosion_ratios_kg_kg"],
        "samples": len(sweep["launch_speeds_m_s"]),
        "fit": fit,
        "law_velocity_exponent": law_k,
        "fit_vs_law_relative_error": rel_dev(fit["exponent"], law_k),
        "literature_exponent": lit,
        "literature_source": lit_source,
        "fit_vs_literature_relative_error": err_lit,
        "in_window_2p1_2p5": in_window,
        "fit_at_45deg": fit45,
        "pass": bool(in_window and err_lit < 0.10 and len(SPEED_GRID_M_S) >= 5),
    }


# ---------------------------------------------------------------------------
# C3: DNV-RP-O501 per-point comparison
# ---------------------------------------------------------------------------
def section_c3() -> dict:
    pair = DNV_PAIR[CASE_MATERIAL.name]
    # The stream is exactly linear in the anchor (verified by a test), so the
    # anchor of the comparison is fitted on the default-law run instead of
    # re-running the stream: E_fit = (A_fit / A_0) * E_stream(A_0).
    unit = velocity_sweep(
        DNV_SPEED_GRID_M_S,
        cfg=StreamConfig(n_particles=N_PARTICLES),
        angle_mean_deg=90.0,
    )
    # the DNV pair is supplied explicitly (k, n); the material argument only
    # carries the target density into the wall-loss conversion below.
    dnv_rates = [
        dnv_ductile_erosion_ratio(v, 90.0, "carbon_steel", k=pair["k"], n=pair["n"])
        for v in DNV_SPEED_GRID_M_S
    ]
    scale = anchor_fit(dnv_rates, unit["erosion_ratios_kg_kg"])
    a_fit = scale * CASE_LAW.anchor_ratio_kg_kg
    cases = []
    for v, dnv_rate, unit_rate in zip(
        DNV_SPEED_GRID_M_S, dnv_rates, unit["erosion_ratios_kg_kg"]
    ):
        model = scale * unit_rate
        cases.append(
            {
                "impact_speed_m_s": float(v),
                "impact_angle_deg": 90.0,
                "dnv_design_ratio_kg_kg": dnv_rate,
                "stream_ratio_kg_kg": model,
                "relative_deviation": rel_dev(model, dnv_rate),
                "dnv_wall_loss_mm_per_year": dnv_ductile_wall_loss_rate(
                    v, 90.0, "carbon_steel", 0.01, 2.0e-3, k=pair["k"], n=pair["n"]
                )["wall_loss_mm_per_year"],
            }
        )
    max_dev = max(c["relative_deviation"] for c in cases)
    # a second, wider velocity band: reported honestly, not part of the verdict
    wide = velocity_sweep(
        [15.0, 50.0],
        cfg=StreamConfig(n_particles=N_PARTICLES),
        angle_mean_deg=90.0,
    )
    wide_cases = []
    for v, unit_rate in zip([15.0, 50.0], wide["erosion_ratios_kg_kg"]):
        dnv_rate = dnv_ductile_erosion_ratio(v, 90.0, "carbon_steel", k=pair["k"], n=pair["n"])
        model = scale * unit_rate
        wide_cases.append(
            {
                "impact_speed_m_s": float(v),
                "dnv_design_ratio_kg_kg": dnv_rate,
                "stream_ratio_kg_kg": model,
                "relative_deviation": rel_dev(model, dnv_rate),
            }
        )
    # carbon steel: honest cross-material check (DNV n = 2.6 vs plastic k = 2.36)
    steel_pair = DNV_PAIR["carbon_steel"]
    steel_law = OkaAnchoredLaw.for_material(
        "carbon_steel",
        steel_pair["k"] * 30.0 ** steel_pair["n"],
        v_ref_m_s=30.0,
        alpha_ref_deg=90.0,
        d_ref_um=150.0,
        anchor_provenance="DNV steel pair moved to the anchor (shape comparison only)",
    )
    steel = velocity_sweep(
        DNV_SPEED_GRID_M_S,
        cfg=StreamConfig(n_particles=400),
        law=steel_law,
        angle_mean_deg=90.0,
    )
    steel_dnv = [
        dnv_ductile_erosion_ratio(v, 90.0, "carbon_steel", k=steel_pair["k"], n=steel_pair["n"])
        for v in DNV_SPEED_GRID_M_S
    ]
    steel_scale = anchor_fit(steel_dnv, steel["erosion_ratios_kg_kg"])
    steel_devs = [
        rel_dev(steel_scale * r, d) for r, d in zip(steel["erosion_ratios_kg_kg"], steel_dnv)
    ]
    return {
        "material": CASE_MATERIAL.name,
        "dnv_pair": pair,
        "anchor_fit_kg_kg": a_fit,
        "anchor_scale_applied": scale,
        "anchor_linearity": "exact: E is linear in the anchor (asserted by a test)",
        "cases": cases,
        "max_relative_deviation": max_dev,
        "wide_band_cases": wide_cases,
        "wide_band_max_relative_deviation": max(
            c["relative_deviation"] for c in wide_cases
        ),
        "carbon_steel_cases": {
            "speeds_m_s": DNV_SPEED_GRID_M_S,
            "relative_deviations": steel_devs,
            "max_relative_deviation": max(steel_devs),
            "note": "DNV steel exponent n = 2.6 vs the plastic-erosion exponent "
                    "k = 2.3 Hv^0.038 = 2.36: the residual is the published exponent "
                    "gap, not an implementation error",
        },
        "pass": bool(max_dev < 0.15 and len(cases) >= 5),
    }


# ---------------------------------------------------------------------------
# C4: mass ledger
# ---------------------------------------------------------------------------
def section_c4() -> dict:
    cfg = StreamConfig(n_particles=4000)
    out = run_impact_stream(cfg)
    # geometry leg: inverted depth must reproduce the deposited volume
    geo = []
    for v in (10.0, 20.0, 30.0, 50.0):
        c = single_impact_crater(v, 20.0, 150e-6, 2650.0, CASE_MATERIAL, law=CASE_LAW)
        v_back = sphere_cap_volume(c.depth_m, c.particle_radius_m)
        r_back = sphere_cap_radius(c.depth_m, c.particle_radius_m)
        geo.append(
            {
                "speed_m_s": v,
                "removed_volume_m3": c.removed_volume_m3,
                "depth_m": c.depth_m,
                "radius_m": c.radius_m,
                "volume_roundtrip_error": rel_dev(v_back, c.removed_volume_m3),
                "area_check": rel_dev(math.pi * r_back * r_back, c.footprint_area_m2),
                "depth_fraction_of_particle_radius": c.depth_fraction_of_particle_radius,
                "volume_fraction_of_particle": c.volume_fraction_of_particle,
                "admissible": c.physically_admissible,
            }
        )
    removed_mass = out["removed_mass_kg"]
    removed_from_volume = out["removed_volume_m3"] * CASE_MATERIAL.density_kg_m3
    mass_consistency = rel_dev(removed_mass, removed_from_volume)
    # energy leg: removed volume cannot exceed the normal impact work / flow stress
    efficiency = (out["removed_volume_m3"] * CASE_MATERIAL.flow_stress_pa) / max(
        out["kinetic_energy_in_j"], 1e-300
    )
    return {
        "stream": {k: out[k] for k in (
            "impacts", "delivered_mass_kg", "contact_mass_kg", "removed_mass_kg",
            "removed_volume_m3", "erosion_ratio_kg_kg", "closure_residual",
            "wall_integrated_volume_m3", "wall_max_depth_m", "wall_mean_depth_m",
            "crater_depth_mean_m", "crater_ratio_mean_kg_kg", "inadmissible_craters",
            "kinetic_energy_in_j", "kinetic_energy_out_j", "bounce_histogram",
        )},
        "geometry_cases": geo,
        "max_geometry_roundtrip_error": max(g["volume_roundtrip_error"] for g in geo),
        "removed_mass_vs_geometry_relative_error": mass_consistency,
        "masses_non_negative": bool(
            out["delivered_mass_kg"] >= 0.0 and out["removed_mass_kg"] >= 0.0
        ),
        "removal_efficiency": efficiency,
        "removal_efficiency_physical": bool(0.0 <= efficiency <= 1.0),
        "closure_residual": out["closure_residual"],
        "pass": bool(
            out["closure_residual"] < 0.05
            and mass_consistency < 0.05
            and max(g["volume_roundtrip_error"] for g in geo) < 0.05
            and out["delivered_mass_kg"] >= 0.0
            and out["removed_mass_kg"] >= 0.0
            and 0.0 <= efficiency <= 1.0
            and out["inadmissible_craters"] == 0
        ),
    }


# ---------------------------------------------------------------------------
# C5: double monotonicity
# ---------------------------------------------------------------------------
def section_c5() -> dict:
    vel = velocity_sweep(
        [15.0, 20.0, 25.0, 30.0, 40.0, 50.0],
        cfg=StreamConfig(n_particles=N_PARTICLES),
        angle_mean_deg=90.0,
    )
    vel_inc = monotonicity(vel["launch_speeds_m_s"], vel["erosion_ratios_kg_kg"], increasing=True)
    ang = angle_sweep(
        [20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0],
        cfg=StreamConfig(n_particles=N_PARTICLES),
        speed_mean_m_s=30.0,
    )
    ang_dec = monotonicity(
        ang["launch_angles_deg"], ang["erosion_ratios_kg_kg"], increasing=False
    )
    return {
        "velocity": vel,
        "velocity_monotone_increasing": vel_inc,
        "angle_post_peak": ang,
        "angle_monotone_decaying_to_90": ang_dec,
        "pass": bool(vel_inc["monotone"] and ang_dec["monotone"]
                     and vel_inc["samples"] >= 5 and ang_dec["samples"] >= 5),
    }


# ---------------------------------------------------------------------------
# C6: negative controls
# ---------------------------------------------------------------------------
def section_c6() -> dict:
    zero_particles = run_impact_stream(StreamConfig(n_particles=0))
    zero_speed = run_impact_stream(
        StreamConfig(n_particles=50, speed_mean_m_s=0.0, speed_cv=0.0, gravity_m_s2=0.0)
    )
    law_zero = float(CASE_LAW.ratio(0.0, 30.0, 150.0))
    crater_zero = single_impact_crater(0.0, 30.0, 150e-6, 2650.0, CASE_MATERIAL, law=CASE_LAW)
    tiny = 1e-12
    return {
        "zero_particles": {
            "impacts": zero_particles["impacts"],
            "erosion_ratio_kg_kg": zero_particles["erosion_ratio_kg_kg"],
            "removed_mass_kg": zero_particles["removed_mass_kg"],
            "pass": bool(
                zero_particles["impacts"] == 0
                and abs(zero_particles["erosion_ratio_kg_kg"]) < tiny
                and abs(zero_particles["removed_mass_kg"]) < tiny
            ),
        },
        "zero_speed": {
            "impacts": zero_speed["impacts"],
            "erosion_ratio_kg_kg": zero_speed["erosion_ratio_kg_kg"],
            "removed_mass_kg": zero_speed["removed_mass_kg"],
            "pass": bool(
                zero_speed["impacts"] == 0
                and abs(zero_speed["erosion_ratio_kg_kg"]) < tiny
                and abs(zero_speed["removed_mass_kg"]) < tiny
            ),
        },
        "law_at_zero_speed": {"ratio_kg_kg": law_zero, "pass": bool(abs(law_zero) < tiny)},
        "crater_at_zero_speed": {
            "removed_volume_m3": crater_zero.removed_volume_m3,
            "depth_m": crater_zero.depth_m,
            "pass": bool(
                abs(crater_zero.removed_volume_m3) < tiny and abs(crater_zero.depth_m) < tiny
            ),
        },
        "pass": None,  # filled below
    }


# ---------------------------------------------------------------------------
# C7: breakage decoupling
# ---------------------------------------------------------------------------
def section_c7() -> dict:
    base_cfg = StreamConfig(n_particles=600)
    plain = run_impact_stream(base_cfg)
    identity = run_impact_stream(base_cfg, on_impact=fragmentation_hook(1))
    fragmented = run_impact_stream(base_cfg, on_impact=fragmentation_hook(3))
    keys = ("erosion_ratio_kg_kg", "removed_mass_kg", "closure_residual", "impacts")
    diffs = {k: abs(identity[k] - plain[k]) for k in keys}
    static = erosion_core_imports_comminution()
    # erodent bookkeeping of the fragmented run: delivered = carried_end + siblings
    carried = fragmented["carried_mass_end_kg"]
    siblings = fragmented["sibling_mass_kg"]
    delivered = fragmented["delivered_mass_kg"]
    mass_book = abs((carried + siblings) - delivered) / max(delivered, 1e-300)
    # one-way coupling demonstration: at 20 deg the particles bounce repeatedly,
    # so comminution really changes what hits the wall next
    oblique = StreamConfig(n_particles=600, angle_mean_deg=20.0, angle_std_deg=0.0)
    oblique_plain = run_impact_stream(oblique)
    oblique_frag = run_impact_stream(oblique, on_impact=fragmentation_hook(4))
    return {
        "identity_vs_plain": {
            "plain": {k: plain[k] for k in keys},
            "identity": {k: identity[k] for k in keys},
            "absolute_differences": diffs,
            "max_absolute_difference": max(diffs.values()),
            "pass": bool(max(diffs.values()) < 1e-6),
        },
        "fragmentation_3": {
            "erosion_ratio_kg_kg": fragmented["erosion_ratio_kg_kg"],
            "removed_mass_kg": fragmented["removed_mass_kg"],
            "closure_residual": fragmented["closure_residual"],
            "delivered_mass_kg": delivered,
            "carried_mass_kg": carried,
            "sibling_mass_kg": siblings,
            "mass_bookkeeping_residual": mass_book,
            "ratio_change_vs_plain": rel_dev(
                fragmented["erosion_ratio_kg_kg"], plain["erosion_ratio_kg_kg"]
            ),
            "note": "one-way coupling: finer erodent changes the removal through the "
                    "size exponent k3; erosion never feeds back on fragmentation",
        },
        "one_way_coupling_at_20deg": {
            "plain_erosion_ratio_kg_kg": oblique_plain["erosion_ratio_kg_kg"],
            "fragmented_erosion_ratio_kg_kg": oblique_frag["erosion_ratio_kg_kg"],
            "relative_change": rel_dev(
                oblique_frag["erosion_ratio_kg_kg"], oblique_plain["erosion_ratio_kg_kg"]
            ),
            "plain_bounce_histogram": oblique_plain["bounce_histogram"],
            "fragmented_bounce_histogram": oblique_frag["bounce_histogram"],
            "fragmented_mass_bookkeeping_residual": abs(
                oblique_frag["carried_mass_end_kg"]
                + oblique_frag["sibling_mass_kg"]
                - oblique_frag["delivered_mass_kg"]
            ) / max(oblique_frag["delivered_mass_kg"], 1e-300),
        },
        "static_import_check": static,
        "pass": bool(
            max(diffs.values()) < 1e-6
            and static["decoupled"]
            and mass_book < 1e-6
        ),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(results: dict, outdir: pathlib.Path) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    # fig 1: angle curves -- analytic Finnie/Oka and the DEM stream aggregate
    c1 = results["C1"]
    ang = np.linspace(0.0, 90.0, 901)
    f = np.asarray(finnie_angle_function(ang), float)
    g = np.asarray(oka_angle_function(ang, CASE_MATERIAL.hardness_hv_gpa), float)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ang, f / f.max(), label="Finnie cutting (normalised)")
    ax.plot(ang, g / g.max(), label="Oka plastic, Hv=0.5 GPa (normalised)")
    s = c1["stream"]
    er = np.asarray(s["erosion_ratios_kg_kg"], float)
    ax.plot(s["launch_angles_deg"], er / er.max(), "o-", ms=4,
            label="DEM stream aggregate (normalised)")
    ax.axvspan(15.0, 25.0, color="0.85", zorder=0, label="ductile window 15-25 deg")
    ax.axvline(c1["stream_peak"]["peak_angle_deg"], color="k", ls=":", lw=1)
    ax.set_xlabel("impingement angle [deg]")
    ax.set_ylabel("normalised erosion ratio")
    ax.set_title("C1 angle functions and the trajectory-aggregated stream")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    p = outdir / "fig1_angle_functions.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    figs.append({"name": p.name, "path": p, "caption":
                 f"Finnie and Oka angle functions (each normalised to its own maximum) "
                 f"against the DEM stream aggregate; the stream peak is at "
                 f"{c1['stream_peak']['peak_angle_deg']:.2f} deg, inside the ductile "
                 f"window, and the self-similar shape deviation on the +-40% peak band "
                 f"is {c1['shape_deviation_peak_band_40pct']:.3f}."})

    # fig 2: velocity exponent and DNV cross-check
    c2, c3 = results["C2"], results["C3"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    ax1.loglog(c2["speeds_m_s"], c2["erosion_ratios_kg_kg"], "o", label="DEM stream")
    vv = np.linspace(min(c2["speeds_m_s"]), max(c2["speeds_m_s"]), 50)
    ref = c2["erosion_ratios_kg_kg"][0] * (vv / c2["speeds_m_s"][0]) ** c2["fit"]["exponent"]
    ax1.loglog(vv, ref, "-", label=f"fit n = {c2['fit']['exponent']:.3f}")
    ax1.set_xlabel("impact speed [m/s]"); ax1.set_ylabel("E [kg/kg]")
    ax1.set_title("C2 velocity exponent"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3, which="both")
    dnv = [c["dnv_design_ratio_kg_kg"] for c in c3["cases"]]
    mod = [c["stream_ratio_kg_kg"] for c in c3["cases"]]
    spd = [c["impact_speed_m_s"] for c in c3["cases"]]
    ax2.loglog(spd, dnv, "s-", label="DNV-RP-O501 design")
    ax2.loglog(spd, mod, "o--", label="DEM stream (anchored)")
    ax2.set_xlabel("impact speed [m/s]"); ax2.set_ylabel("E [kg/kg]")
    ax2.set_title(f"C3 DNV cross-check (max dev {c3['max_relative_deviation']*100:.1f}%)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")
    p = outdir / "fig2_velocity_and_dnv.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    figs.append({"name": p.name, "path": p, "caption":
                 f"Left: stream E vs speed on log-log axes with the fitted exponent "
                 f"{c2['fit']['exponent']:.3f} (literature {c2['literature_exponent']}). "
                 f"Right: per-point comparison against DNV-RP-O501 with the single "
                 f"fitted anchor; max deviation {c3['max_relative_deviation']*100:.1f}%."})

    # fig 3: single-impact crater geometry + wall depth field
    c4 = results["C4"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    vv = np.linspace(5.0, 60.0, 60)
    depths = [
        single_impact_crater(v, 20.0, 150e-6, 2650.0, CASE_MATERIAL, law=CASE_LAW).depth_m
        for v in vv
    ]
    ind = [
        indentation_depth_m(v * math.sin(math.radians(20.0)), 2650 * math.pi / 6 * 150e-6**3,
                            75e-6, CASE_MATERIAL.flow_stress_pa)
        for v in vv
    ]
    ax1.loglog(vv, depths, label="crater depth (erosion law)")
    ax1.loglog(vv, ind, ls="--", label="plastic indentation depth")
    ax1.set_xlabel("impact speed [m/s]"); ax1.set_ylabel("depth [m]")
    ax1.set_title("C4 single-impact crater geometry"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3, which="both")
    stream = section_c4_stream_for_figure()
    im = ax2.imshow(stream * 1e9, origin="lower", cmap="inferno")
    ax2.set_title("wall depth after 4000 impacts [nm]")
    ax2.set_xlabel("x cell"); ax2.set_ylabel("y cell")
    fig.colorbar(im, ax=ax2, fraction=0.046)
    p = outdir / "fig3_crater_and_wall.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    figs.append({"name": p.name, "path": p, "caption":
                 "Left: single-impact crater depth from the erosion law vs the plastic "
                 "indentation depth (both ~v; the crater volume therefore scales as v^2). "
                 "Right: accumulated wall depth field of the 4000-particle stream."})

    # fig 4: monotonicity panels
    c5 = results["C5"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    ax1.plot(c5["velocity"]["launch_speeds_m_s"], c5["velocity"]["erosion_ratios_kg_kg"], "o-")
    ax1.set_xlabel("impact speed [m/s]"); ax1.set_ylabel("E [kg/kg]")
    ax1.set_title("C5 E increasing in speed"); ax1.grid(alpha=0.3)
    ax2.plot(c5["angle_post_peak"]["launch_angles_deg"], c5["angle_post_peak"]["erosion_ratios_kg_kg"], "s-")
    ax2.set_xlabel("impingement angle [deg]"); ax2.set_ylabel("E [kg/kg]")
    ax2.set_title("C5 monotone decay after the peak"); ax2.grid(alpha=0.3)
    p = outdir / "fig4_monotonicity.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    figs.append({"name": p.name, "path": p, "caption":
                 "C5 double monotonicity: erosion ratio increasing in impact speed (left) "
                 "and decaying monotonically from the angle peak to normal incidence (right)."})
    return figs


_C4_STREAM_CACHE: dict = {}


def section_c4_stream_for_figure() -> np.ndarray:
    if "wall" not in _C4_STREAM_CACHE:
        out = run_impact_stream(StreamConfig(n_particles=4000))
        _C4_STREAM_CACHE["wall"] = out["wall"].depth
    return _C4_STREAM_CACHE["wall"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="artifacts/p176w2-n11")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    global N_PARTICLES
    if args.quick:
        N_PARTICLES = 300
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    c1 = section_c1()
    c2 = section_c2()
    c3 = section_c3()
    c4 = section_c4()
    c5 = section_c5()
    c6 = section_c6()
    c6["pass"] = bool(
        c6["zero_particles"]["pass"]
        and c6["zero_speed"]["pass"]
        and c6["law_at_zero_speed"]["pass"]
        and c6["crater_at_zero_speed"]["pass"]
    )
    c7 = section_c7()
    results = {
        "node": "p176w2-n11-dem-erosion",
        "repository": "TensorDEM",
        "module": "tensordem.erosion_dem",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_s": None,
        "parameters": {
            "n_particles": N_PARTICLES,
            "angle_grid_deg": ANGLE_GRID_DEG,
            "speed_grid_m_s": SPEED_GRID_M_S,
            "dnv_speed_grid_m_s": DNV_SPEED_GRID_M_S,
            "case_material": {
                "name": CASE_MATERIAL.name,
                "hv_gpa": CASE_MATERIAL.hardness_hv_gpa,
                "density_kg_m3": CASE_MATERIAL.density_kg_m3,
                "flow_stress_pa": CASE_MATERIAL.flow_stress_pa,
                "source": CASE_MATERIAL.source,
            },
            "erodent": {"diameter_um": 150.0, "density_kg_m3": 2650.0},
            "law": {
                "velocity_exponent": CASE_LAW.velocity_exponent,
                "peak_angle_deg": CASE_LAW.peak_angle_deg,
                "anchor_ratio_kg_kg": CASE_LAW.anchor_ratio_kg_kg,
                "anchor_provenance": CASE_LAW.anchor_provenance,
                "size_exponent_k3": CASE_LAW.k3,
            },
        },
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "C4": c4,
        "C5": c5,
        "C6": c6,
        "C7": c7,
    }
    results["wall_time_s"] = round(time.time() - t0, 2)
    criteria = {k: bool(results[k]["pass"]) for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")}
    results["criteria"] = criteria
    results["all_pass"] = all(criteria.values())

    figs = make_figures(results, outdir)
    figures_manifest = []
    for f in figs:
        figures_manifest.append(
            {"name": f["name"], "caption": f["caption"],
             "bytes": f["path"].stat().st_size}
        )
    results["figures"] = figures_manifest

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k not in ("wall", "records")}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    payload = _clean(results)
    (outdir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({k: payload[k] for k in ("criteria", "all_pass", "wall_time_s")}, indent=2))
    print(f"wrote {outdir/'results.json'}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())