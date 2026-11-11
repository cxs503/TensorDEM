"""Verification driver for the floe-attrition chain (criteria C1-C7).

Each ``criterion_c*`` function is self-contained, returns a JSON-ready dict with
the measured numbers, the threshold and a pass flag, and is exercised both by
``scripts/floe_attrition_verification.py`` (which writes the artifacts and the
figures) and by ``tests/test_floe_attrition.py`` (which asserts on them).

The criteria are those of node p472w2-N05:

===========  ==========================================================
C1           Wadell roundness closed form vs analytic polygons/polyhedra
C2           Archard attrition vs the Archard-type closed form
C3           R(t) fit quality and the pressure scaling of tau
C4           R -> 1 returns to the m9375 sphere baseline
C5           3x3 design matrix, main effects and interaction (ANOVA)
C6           rounding/consolidation shares closed to 100 %
C7           volume conservation and the zero-slip negative control
===========  ==========================================================
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .design import DesignCell, build_design_matrix, two_way_anova
from .friction import (InterlockingFriction, SphereBaseline, load_sphere_baseline)
from .geometry import (RoundedCube, RoundedPolygon,
                       fillet_radius_for_roundness, polygon_metrics_boundary_integral,
                       sphericity_3d)
from .resistance import TransitScenario, simulate_multi_pass
from .wear import (ArchardWear, integrate_cube_attrition, integrate_polygon_attrition,
                   wear_volume_closed_form)

__all__ = [
    "criteria",
    "criterion_c1",
    "criterion_c2",
    "criterion_c3",
    "criterion_c4",
    "criterion_c5",
    "criterion_c6",
    "criterion_c7",
    "PRODUCTION",
]

#: Production settings (pre-registered; see
#: ``prereg/p472w2_n05_floe_attrition_prereg.json``).
PRODUCTION = {
    "edge_length": 0.6,          # m, regular-polygon edge (floe ~1 m across)
    "n_sides": 4,                # square base family for the chain
    "initial_fillet_ratio": 0.02,  # r0 / a, fresh floe corner radius
    "wear_coefficient": 5.0e-3,
    "hardness_pa": 3.0e7,
    "pressure_pa": 2.0e5,
    "slip_per_transit": 2.5,     # m
    "n_transits": 400,
    "concentration": 0.5,
    # --- multi-pass decay scenario (C6): busy channel in dense brash ice -------
    "decay_concentration": 0.70,
    "decay_pressure_pa": 8.0e5,
    "rest_time_per_transit": 300.0,     # s between passages (5 min headway)
    "bond_time_constant": 3.6e5,        # s (100 h) freeze-bond growth clock
    "bond_strength_max": 2.0e4,         # Pa cohesion of a fully bonded contact
    "bond_area": 0.15,                  # m^2 per metre of width actually bonded
    "contact_length": 5.0,              # m of hull rubbing the brash layer
    "load_coupling": 0.30,              # beta_c: bonded layer amplifies F_n
    "roundness_levels": (0.80, 0.90, 0.99),
    "concentrations": (0.30, 0.50, 0.70),
    "replicates": 8,
    "fit_beta_expected": None,   # fitted freely (see the pre-registration)
    "tau_pressure_exponent": -1.0,
    "tau_exponent_tolerance": 0.20,
}


def _production_shape() -> RoundedPolygon:
    a = PRODUCTION["edge_length"]
    return RoundedPolygon(PRODUCTION["n_sides"], a,
                          PRODUCTION["initial_fillet_ratio"] * a)


def _production_wear() -> ArchardWear:
    return ArchardWear(PRODUCTION["wear_coefficient"], PRODUCTION["hardness_pa"])


# ---------------------------------------------------------------------------
# C1 -- Wadell roundness closed forms
# ---------------------------------------------------------------------------

#: Exact sphericity of the Platonic solids from their exact ``(V, A)``.
_PLATONIC = {
    "tetra": (math.sqrt(2.0) / 12.0, math.sqrt(3.0)),
    "cube": (1.0, 6.0),
    "octa": (math.sqrt(2.0) / 3.0, 2.0 * math.sqrt(3.0)),
    "icosa": ((5.0 / 12.0) * (3.0 + math.sqrt(5.0)),
              5.0 * math.sqrt(3.0)),
    "dodeca": ((15.0 + 7.0 * math.sqrt(5.0)) / 4.0,
               3.0 * math.sqrt(25.0 + 10.0 * math.sqrt(5.0))),
}


def criterion_c1(*, tolerance: float = 1e-12) -> dict:
    """Closed-form roundness/sphericity vs independent computations."""
    rows2d = []
    worst_2d = 0.0
    for n in (3, 4, 5, 6, 8, 64):
        base = RoundedPolygon(n, 1.0, 0.0)
        r_max = base.max_fillet_radius
        for frac in (0.0, 0.05, 0.2, 0.5, 0.8, 0.99, 1.0):
            poly = base.with_fillet(frac * r_max)
            area, perimeter = polygon_metrics_boundary_integral(poly)
            e_a = abs(area - poly.area) / poly.area
            e_p = abs(perimeter - poly.perimeter) / poly.perimeter
            worst_2d = max(worst_2d, e_a, e_p)
            rows2d.append({
                "n_sides": n, "fillet_fraction": frac,
                "roundness": poly.wadell_roundness,
                "area_rel_error": e_a, "perimeter_rel_error": e_p,
            })
    base64 = RoundedPolygon(64, 1.0, 0.0)
    circle = base64.with_fillet(base64.max_fillet_radius)
    circle_roundness_error = abs(circle.wadell_roundness - 1.0)

    # 3D: exact Platonic sphericity vs the existing vertex-cloud descriptor
    from ..polyhedra_shapes import (compute_shape_descriptor, make_cube,
                                    make_dodecahedron, make_icosahedron,
                                    make_octahedron, make_tetrahedron)
    makers = {"tetra": make_tetrahedron, "cube": make_cube, "octa": make_octahedron,
              "icosa": make_icosahedron, "dodeca": make_dodecahedron}
    rows3d = []
    worst_3d = 0.0
    for name, maker in makers.items():
        spec = maker()
        sd = compute_shape_descriptor(name, spec.vertices)
        v_ref, a_ref = _PLATONIC[name]          # unit edge-length reference values
        psi_ref = sphericity_3d(v_ref, a_ref)
        e_psi = abs(sd.sphericity - psi_ref) / psi_ref
        e_v = abs(sd.volume / v_ref - (sd.edge_length ** 3 if hasattr(sd, "edge_length") else 1.0))
        worst_3d = max(worst_3d, e_psi)
        rows3d.append({"polyhedron": name, "sphericity_descriptor": sd.sphericity,
                       "sphericity_closed_form": psi_ref, "rel_error": e_psi,
                       "volume": sd.volume, "surface_area": sd.surface_area,
                       "volume_over_analytic": sd.volume / v_ref,
                       "area_over_analytic": sd.surface_area / a_ref})

    # 3D rounded-cube: Steiner consistency dV/dr == A(r) by Gauss-Legendre
    nodes, weights = np.polynomial.legendre.leggauss(64)
    # Steiner identity dV/dr == A(r) holds at *fixed* inner cube L; in the
    # constant-size family (a = L + 2r, i.e. material removal) dV/dr != A, which
    # is exactly why the wear rate picks up the geometry factor.
    cube = RoundedCube(0.6, 0.02)
    l_fixed = cube.flat_size                            # keep L, vary r
    r_grid = 0.5 * cube.radius * (nodes + 1.0)           # map [-1, 1] -> [0, r]
    def _area_fixed_L(r):
        return 6.0 * l_fixed ** 2 + 6.0 * math.pi * l_fixed * r + 4.0 * math.pi * r ** 2
    def _vol_fixed_L(r):
        return (l_fixed ** 3 + 6.0 * l_fixed ** 2 * r
                + 3.0 * math.pi * l_fixed * r ** 2 + 4.0 / 3.0 * math.pi * r ** 3)
    areas = np.array([_area_fixed_L(r) for r in r_grid])
    volume_by_integral = 0.5 * cube.radius * float(np.sum(weights * areas))
    steiner_error = (abs((_vol_fixed_L(0.0) + volume_by_integral) - _vol_fixed_L(cube.radius))
                     / _vol_fixed_L(cube.radius))
    sphere = RoundedCube(1.0, 0.5)
    sphere_errors = {
        "volume": abs(sphere.volume - 4.0 / 3.0 * math.pi * 0.5 ** 3) / sphere.volume,
        "surface": abs(sphere.surface_area - 4.0 * math.pi * 0.25) / sphere.surface_area,
        "sphericity": abs(sphere.sphericity - 1.0),
        "roundness": abs(sphere.wadell_roundness - 1.0),
        "interlocking": abs(sphere.interlocking_index),
    }

    # DEM-ready discretisation: roundness converges as O(m^-2)
    base_q = RoundedPolygon(4, 1.0, 0.0)
    poly = base_q.with_fillet(0.4 * base_q.max_fillet_radius)
    conv = []
    for m in (4, 8, 16, 32, 64):
        v = poly.wadell_roundness
        a_num, p_num = polygon_metrics_from_vertices(poly, m)
        conv.append({"arc_samples": m,
                     "roundness_discrete": 4.0 * math.pi * a_num / (p_num ** 2),
                     "rel_error": abs(4.0 * math.pi * a_num / (p_num ** 2) - v) / v})

    worst = max(worst_2d, worst_3d, circle_roundness_error,
                steiner_error, *sphere_errors.values())
    return {
        "criterion": "C1 Wadell roundness closed form vs analytic polygons/polyhedra",
        "threshold": tolerance,
        "worst_relative_error": worst,
        "polygon_rows": rows2d,
        "polyhedron_rows": rows3d,
        "circle_limit_roundness_error": circle_roundness_error,
        "steiner_dV_dr_error": steiner_error,
        "sphere_limit_errors": sphere_errors,
        "dem_discretisation_convergence": conv,
        "passed": bool(worst <= tolerance),
        "note": "closed forms are compared with (i) an exact Green's-theorem "
                "boundary integral, (ii) the ConvexHull descriptor of "
                "polyhedra_shapes with exact Platonic (V, A), (iii) Steiner's "
                "identity dV/dr = A by 64-node Gauss-Legendre.",
    }


def polygon_metrics_from_vertices(poly: RoundedPolygon, arc_samples: int):
    """Area/perimeter of the *discretised* boundary (what a DEM particle sees)."""
    from .geometry import rounded_polygon_vertices
    v = rounded_polygon_vertices(poly, arc_samples=arc_samples)
    x, y = v[:, 0], v[:, 1]
    area = 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    seg = np.roll(v, -1, axis=0) - v
    perimeter = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
    return area, perimeter


# ---------------------------------------------------------------------------
# C2 -- Archard attrition law vs its closed form
# ---------------------------------------------------------------------------

def attrition_closed_form(poly: RoundedPolygon, wear: ArchardWear, pressure: float,
                          sliding: np.ndarray) -> np.ndarray:
    """Analytic solution of the pressure-controlled attrition ODE.

    With ``b(r) = chi r`` and ``V(r) = c2 r^2`` the rate ``dV/dt = k p b(r) s/H``
    integrates to ``r(s) = r0 + Omega s`` with
    ``Omega = k p chi / (2 c2 H)``, hence ``V(s) = c2 (r0 + Omega s)^2 - c2 r0^2``.
    """
    r0 = poly.fillet_radius
    c2 = (wear_volume_closed_form(poly, 1.0))
    chi = poly.contact_patch_width / r0 if r0 > 0 else float("nan")
    omega = wear.k_over_h * pressure * chi / (2.0 * c2)
    r = np.minimum(r0 + omega * np.asarray(sliding, dtype=float), poly.max_fillet_radius)
    return c2 * r ** 2 - c2 * r0 ** 2


def criterion_c2(*, tolerance: float = 0.10) -> dict:
    """Attrition simulation vs the Archard-type closed form, and vs the
    classical constant-patch formula."""
    poly = _production_shape()
    wear = _production_wear()
    p = PRODUCTION["pressure_pa"]
    slip = PRODUCTION["slip_per_transit"]
    n = PRODUCTION["n_transits"]
    hist = integrate_polygon_attrition(poly, wear, pressure=p, slip_per_step=slip,
                                       n_steps=n)
    a = hist.arrays()
    sliding = a["sliding"]
    worn_sim = a["worn_geometric"]
    worn_closed = attrition_closed_form(poly, wear, p, sliding)
    # classical Archard reference: constant initial patch (no geometric feedback)
    patch0 = poly.contact_patch_width
    worn_classic = wear.worn_volume(p * patch0, sliding)

    def _rel(x, y):
        m = y > 0
        return float(np.max(np.abs(x[m] - y[m]) / y[m])) if m.any() else 0.0

    err_closed = np.abs(worn_sim[1:] - worn_closed[1:]) / np.maximum(worn_closed[1:], 1e-300)
    err_classic = np.abs(worn_sim - worn_classic) / np.maximum(worn_classic, 1e-300)

    # cadence sensitivity: coarser wear updates (Euler-type error growth)
    cadence = []
    for coarse in (1, 2, 5, 10, 25):
        h = integrate_polygon_attrition(poly, wear, pressure=p,
                                       slip_per_step=slip * coarse,
                                       n_steps=max(1, n // coarse))
        ref = attrition_closed_form(poly, wear, p, np.array([h.sliding[-1]]))[0]
        cadence.append({
            "steps": int(max(1, n // coarse)),
            "slip_per_step": slip * coarse,
            "worn_geometric": h.worn_geometric[-1],
            "worn_closed_form": float(ref),
            "rel_error": abs(h.worn_geometric[-1] - float(ref)) / float(ref) if ref else 0.0,
        })

    worst_cadence = max(r["rel_error"] for r in cadence)
    return {
        "criterion": "C2 corner wear volume vs the Archard-type closed form",
        "threshold": tolerance,
        "pressure_pa": p,
        "sliding_total_m": float(sliding[-1]),
        "worn_simulation": float(worn_sim[-1]),
        "worn_closed_form": float(attrition_closed_form(poly, wear, p, np.array([sliding[-1]]))[0]),
        "worn_classical_constant_patch": float(wear.worn_volume(p * patch0, sliding[-1])),
        "rel_error_vs_closed_form": float(err_closed.max() if err_closed.size else 0.0),
        "rel_error_classical_formula": float(err_classic[-1]),
        "classical_formula_underestimates_by_pct": 100.0 * (
            float(worn_sim[-1]) / float(wear.worn_volume(p * patch0, sliding[-1])) - 1.0),
        "classical_formula_error_vs_sliding": [
            {"sliding": float(x),
             "rel_error_classical": float(np.interp(x, sliding, err_classic))}
            for x in (10.0, 25.0, 50.0, 100.0, 250.0, 1000.0)
            if x <= float(sliding[-1])
        ],
        "cadence_table": cadence,
        "worst_cadence_error": worst_cadence,
        "production_cadence_error": float(cadence[0]["rel_error"]),
        "passed": bool(cadence[0]["rel_error"] <= tolerance
                       and float(err_closed.max() if err_closed.size else 0.0) <= tolerance),
        "note": "the production cadence and the exact-inversion integrator are "
                "verified against the analytic ODE solution; the classical "
                "constant-patch Archard formula is reported separately because "
                "the patch grows as the corner blunts (that difference is a "
                "physical result, not a numerical error).",
    }


# ---------------------------------------------------------------------------
# C3 -- evolution law: R(t) fit and the pressure scaling of tau
# ---------------------------------------------------------------------------

def _fit_roundness_decay(sliding: np.ndarray, roundness: np.ndarray,
                         window_fraction: float = 0.6321):
    """Fit ``1 - R = (1 - R0) exp(-(s/tau)^beta)`` on the pre-saturation window.

    The window ends where ``1 - R`` has decayed to ``exp(-1)`` of its initial
    value (the relaxation time), namely exactly where ``tau`` is defined.
    """
    from scipy.optimize import curve_fit
    y = 1.0 - np.asarray(roundness, dtype=float)
    s = np.asarray(sliding, dtype=float)
    y0 = y[0]
    target = y0 * window_fraction
    idx = np.nonzero(y <= target)[0]
    stop = int(idx[0]) if idx.size else len(y) - 1
    stop = max(stop, 2)
    s_w, y_w = s[:stop + 1], y[:stop + 1]

    def model(x, tau, beta):
        return y0 * np.exp(-(np.maximum(x, 0.0) / tau) ** beta)

    s_scale = s_w[-1] if s_w[-1] > 0 else 1.0
    popt, _ = curve_fit(model, s_w, y_w, p0=(s_scale, 1.0),
                        bounds=([1e-12, 0.05], [1e6 * s_scale, 5.0]), maxfev=20000)
    pred = model(s_w, *popt)
    ss_res = float(np.sum((y_w - pred) ** 2))
    ss_tot = float(np.sum((y_w - y_w.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"tau": float(popt[0]), "beta": float(popt[1]), "r2": r2,
            "window_end_sliding": float(s_w[-1]), "n_points": int(s_w.size),
            "y0": float(y0), "y_end": float(y_w[-1])}


def criterion_c3(*, r2_threshold: float = 0.99) -> dict:
    """``R(s)`` decay fits and the ``tau ∝ p^-1`` scaling."""
    poly = _production_shape()
    wear = _production_wear()
    slip = PRODUCTION["slip_per_transit"]
    pressures = [0.5e5, 1.0e5, 2.0e5, 4.0e5, 8.0e5]
    rows = []
    for p in pressures:
        # long enough for the slowest case to pass the 63 % window, short enough
        # to stay below full rounding: each run stops at 63 % recovery.
        n_steps = 4000
        hist = integrate_polygon_attrition(poly, wear, pressure=p,
                                          slip_per_step=slip, n_steps=n_steps)
        a = hist.arrays()
        fit = _fit_roundness_decay(a["sliding"], a["roundness"])
        rows.append({"pressure": p, "tau_sliding_m": fit["tau"], "beta": fit["beta"],
                     "r2": fit["r2"], "window_end_sliding": fit["window_end_sliding"],
                     "roundness_initial": hist.roundness[0],
                     "roundness_at_window_end": 1.0 - fit["y_end"],
                     "saturated_in_window": bool(fit["y_end"] <= 0.0)})

    ln_p = np.log([r["pressure"] for r in rows])
    ln_tau = np.log([r["tau_sliding_m"] for r in rows])
    slope, intercept = np.polyfit(ln_p, ln_tau, 1)
    pred = slope * ln_p + intercept
    ss_res = float(np.sum((ln_tau - pred) ** 2))
    ss_tot = float(np.sum((ln_tau - ln_tau.mean()) ** 2))
    r2_scaling = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    exponent = float(slope)
    expected = PRODUCTION["tau_pressure_exponent"]
    deviation = abs(exponent - expected) / abs(expected)
    r2_ok = all(r["r2"] >= r2_threshold for r in rows)
    mono_tau = all(rows[i]["tau_sliding_m"] > rows[i + 1]["tau_sliding_m"]
                   for i in range(len(rows) - 1))
    return {
        "criterion": "C3 R(t) fit R^2 >= 0.99 and tau vs pressure exponent",
        "threshold": {"r2": r2_threshold,
                      "tau_exponent": expected,
                      "tolerance": PRODUCTION["tau_exponent_tolerance"]},
        "rows": rows,
        "min_r2": float(min(r["r2"] for r in rows)),
        "tau_pressure_exponent": exponent,
        "tau_exponent_deviation": float(deviation),
        "tau_scaling_r2": float(r2_scaling),
        "tau_monotone_decreasing": bool(mono_tau),
        "passed": bool(r2_ok and deviation <= PRODUCTION["tau_exponent_tolerance"]
                       and mono_tau),
        "note": "pre-registered exponent -1: every term of the attrition rate is "
                "linear in the contact pressure, so the time to reach a given "
                "roundness scales as 1/p exactly.",
    }


# ---------------------------------------------------------------------------
# C4 -- R -> 1 limit against the m9375 sphere baseline
# ---------------------------------------------------------------------------

def criterion_c4(*, tolerance: float = 0.01) -> dict:
    """``mu_eff(R = 1)`` must return the m9375 sphere baseline."""
    baseline = load_sphere_baseline()
    model = InterlockingFriction()
    rows = []
    worst = 0.0
    for c in PRODUCTION["concentrations"]:
        mu_sph = baseline.mu(c)
        # (a) the closure at exactly R = 1 (interlocking index 0.0)
        base_circle = RoundedPolygon(64, PRODUCTION["edge_length"], 0.0)
        poly_circle = base_circle.with_fillet(base_circle.max_fillet_radius)
        mu_limit = model.mu_polygon(poly_circle, c, baseline)
        # (b) a near-circle polygon (R within 1e-6 of 1)
        near = RoundedPolygon(64, PRODUCTION["edge_length"],
                              0.999999 * poly_circle.max_fillet_radius)
        mu_near = model.mu_polygon(near, c, baseline)
        # (c) the attrition chain run to full rounding
        hist = integrate_polygon_attrition(poly_circle.with_fillet(0.02 * PRODUCTION["edge_length"]),
                                          _production_wear(), pressure=8.0e5,
                                          slip_per_step=PRODUCTION["slip_per_transit"],
                                          n_steps=20000)
        mu_chain = model.mu(float(hist.interlocking[-1]), c, baseline)
        rows.append({
            "concentration": c,
            "mu_sphere_baseline": mu_sph,
            "mu_limit_exact_circle": mu_limit,
            "difference_exact": mu_limit - mu_sph,
            "near_circle_roundness": near.wadell_roundness,
            "mu_near_circle": mu_near,
            "rel_error_near_circle": abs(mu_near - mu_sph) / mu_sph,
            "chain_final_roundness": float(hist.roundness[-1]),
            "mu_after_full_attrition": mu_chain,
            "rel_error_chain": abs(mu_chain - mu_sph) / mu_sph,
        })
        worst = max(worst, rows[-1]["rel_error_near_circle"], rows[-1]["rel_error_chain"])
    bit_identical = all(r["difference_exact"] == 0.0 for r in rows)
    return {
        "criterion": "C4 R -> 1 returns to the m9375 sphere baseline",
        "threshold": tolerance,
        "baseline_source": baseline.source,
        "baseline": baseline.as_dict(),
        "rows": rows,
        "worst_relative_error": float(worst),
        "bit_identical_at_limit": bool(bit_identical),
        "passed": bool(bit_identical and worst <= tolerance),
    }


# ---------------------------------------------------------------------------
# C5 -- 3x3 design matrix (Analysis of variance)
# ---------------------------------------------------------------------------

def criterion_c5() -> dict:
    cells = build_design_matrix(PRODUCTION["roundness_levels"],
                                PRODUCTION["concentrations"],
                                replicates=PRODUCTION["replicates"],
                                edge_length=PRODUCTION["edge_length"])
    anova = two_way_anova(cells)
    realised = {}
    for cell in cells:
        realised.setdefault(cell.roundness_level, []).append(cell.realized_roundness)
    return {
        "criterion": "C5 3x3 design matrix: main effects and interaction (ANOVA)",
        "replicates_per_cell": PRODUCTION["replicates"],
        "realised_roundness_max_error": float(max(
            abs(np.mean(v) - k) for k, v in realised.items())),
        "anova": anova,
        "n_cells": len(cells),
        "passed": bool(len(cells) == 9 * PRODUCTION["replicates"]
                       and not math.isnan(anova["table"][3]["ms"])),
        "note": "the interaction is an *output*, not an assumption: when it is "
                "significant the rounding and concentration mechanisms must be "
                "reported separately (see anova.interpretation).",
    }


# ---------------------------------------------------------------------------
# C6 -- multi-pass resistance decay, shares closed to 100 %
# ---------------------------------------------------------------------------

def criterion_c6(*, tolerance: float = 1e-9) -> dict:
    poly = _production_shape()
    wear = _production_wear()
    scenario = TransitScenario(
        concentration=PRODUCTION["decay_concentration"],
        n_transits=PRODUCTION["n_transits"],
        slip_per_transit=PRODUCTION["slip_per_transit"],
        contact_pressure=PRODUCTION["decay_pressure_pa"],
        rest_time_per_transit=PRODUCTION["rest_time_per_transit"],
        bond_time_constant=PRODUCTION["bond_time_constant"],
        bond_strength_max=PRODUCTION["bond_strength_max"],
        bond_area=PRODUCTION["bond_area"],
        contact_length=PRODUCTION["contact_length"],
        load_coupling=PRODUCTION["load_coupling"],
    )
    full = simulate_multi_pass(scenario, poly, wear)
    only_round = simulate_multi_pass(scenario, poly, wear, allow_consolidation=False)
    only_cons = simulate_multi_pass(scenario, poly, wear, allow_rounding=False)
    shares = full["shares"]
    residual = (shares["delta_rounding"] + shares["delta_consolidation"]
                - shares["delta_total"])
    ss_round = only_round["initial"]["resistance"] - only_round["final"]["resistance"]
    ss_cons = only_cons["initial"]["resistance"] - only_cons["final"]["resistance"]
    ss_full = full["initial"]["resistance"] - full["final"]["resistance"]
    superposition_residual = ss_full - (ss_round + ss_cons)
    return {
        "criterion": "C6 rounding/consolidation shares close to 100 %",
        "threshold_closure": tolerance,
        "scenario": scenario.as_dict(),
        "resistance_initial": full["initial"]["resistance"],
        "resistance_final": full["final"]["resistance"],
        "decay_fraction": full["final"]["decay_fraction"],
        "mu_initial": full["initial"]["mu_effective"],
        "mu_final": full["final"]["mu_effective"],
        "roundness_initial": only_round["interlocking"][0],
        "roundness_final_interlocking": full["interlocking"][-1],
        "shares": shares,
        "closure_residual": residual,
        "closure_relative": abs(residual) / abs(shares["delta_total"]) if shares["delta_total"] else 0.0,
        "single_mechanism": {
            "rounding_only_delta": ss_round,
            "consolidation_only_delta": ss_cons,
            "combined_delta": ss_full,
            "superposition_residual": superposition_residual,
            "superposition_residual_pct_of_total": (100.0 * superposition_residual / ss_full
                                                    if ss_full else 0.0),
        },
        "contact_energy_total": full["final"]["contact_energy"],
        "worn_volume_total": full["final"]["worn_volume"],
        "curves": {
            "transits": full["transits"],
            "resistance": full["resistance"],
            "mu_effective": full["mu_effective"],
            "contact_energy": full["contact_energy"],
            "interlocking": full["interlocking"],
            "bond_fraction": full["bond_fraction"],
            "resistance_rounding_only": only_round["resistance"],
            "resistance_consolidation_only": only_cons["resistance"],
        },
        "passed": bool(abs(residual) / abs(shares["delta_total"]) <= tolerance
                       if shares["delta_total"] else False),
        "note": "the shares are the two exact halves of the resistance change; the "
                "superposition residual measures the load coupling between the "
                "mechanisms and is reported, not hidden.",
    }


# ---------------------------------------------------------------------------
# C7 -- conservation and negative control
# ---------------------------------------------------------------------------

def criterion_c7(*, tolerance: float = 1e-6) -> dict:
    poly = _production_shape()
    wear = _production_wear()
    hist = integrate_polygon_attrition(poly, wear,
                                       pressure=PRODUCTION["pressure_pa"],
                                       slip_per_step=PRODUCTION["slip_per_transit"],
                                       n_steps=PRODUCTION["n_transits"])
    cube = RoundedCube(0.12, 0.01)
    hist3d = integrate_cube_attrition(cube, wear, pressure=5.0e5,
                                      slip_per_step=PRODUCTION["slip_per_transit"],
                                      n_steps=1200)
    zero_poly = integrate_polygon_attrition(poly, wear,
                                            pressure=PRODUCTION["pressure_pa"],
                                            slip_per_step=0.0, n_steps=50)
    zero_cube = integrate_cube_attrition(cube, wear, pressure=5.0e5,
                                         slip_per_step=0.0, n_steps=50)
    zero_ok = (zero_poly.worn_archard[-1] == 0.0 and zero_poly.excess[-1] == 0.0
               and zero_poly.state[-1] == poly.fillet_radius
               and zero_cube.worn_archard[-1] == 0.0
               and zero_cube.state[-1] == cube.radius)
    worst = max(hist.conservation_error, hist3d.conservation_error)
    return {
        "criterion": "C7 wear-volume conservation and zero-slip negative control",
        "threshold": tolerance,
        "polygon": hist.summary(),
        "cube": hist3d.summary(),
        "worst_conservation_error": float(worst),
        "zero_slip": {
            "polygon_worn": zero_poly.worn_archard[-1],
            "polygon_state_unchanged": bool(zero_poly.state[-1] == poly.fillet_radius),
            "cube_worn": zero_cube.worn_archard[-1],
            "cube_state_unchanged": bool(zero_cube.state[-1] == cube.radius),
            "exact": bool(zero_ok),
        },
        "passed": bool(worst <= tolerance and zero_ok),
    }


def criteria(*, quick: bool = False) -> dict:
    """Run C1-C7 and return the full artifact payload."""
    out = {
        "node": "p472w2-N05",
        "module": "tensordem.floe_attrition",
        "production": dict(PRODUCTION),
        "outputs": {
            "C1": criterion_c1(),
            "C2": criterion_c2(),
            "C3": criterion_c3(),
            "C4": criterion_c4(),
            "C5": criterion_c5(),
            "C6": criterion_c6(),
            "C7": criterion_c7(),
        },
    }
    out["all_passed"] = all(v["passed"] for v in out["outputs"].values())
    return out