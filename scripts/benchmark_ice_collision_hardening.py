"""p256w1-N04 benchmark: strain-hardening hinges + ISO 19906 ice-pressure indentation.

Runs every machine-checkable acceptance criterion of campaign node
``p256w1-n04-ice-collision-hardening`` and writes a paper-grade evidence bundle
(JSON + CSV + figures, each figure with a SHA-256) into one output directory:

    python scripts/benchmark_ice_collision_hardening.py --output runs/ice_collision_hardening

Criteria (numbers, artefacts and verdicts land in ``report.json``):

(a) hardening -> 0 limit      swaps back to the N01 perfectly-plastic surface:
                              penetration depth / energies / resistance agree
                              with ``resistance.solve_penetration`` to <= 1e-3
                              relative (measured: exact, 0.0).
(b) hardening increment       the closed-form hardening energy increment agrees
                              with an independent fibre-quadrature + Simpson route
                              to <= 2 %, stays below the rigid-hardening bound,
                              and that bound is tight (< 10 % slack).
(c) energy conservation       ``E_k = E_steel + E_friction + E_ice + residual``
                              with |residual|/E_k <= 1e-6 over the whole sweep.
(d) Minorsky recalibration    the collision coefficient ``C(xi, ice face)`` of
                              every case, its uncertainty envelope over the
                              declared parameter box, and the empirical residual
                              of the ship-ship / ice-region subsets.  The case
                              table is NOT in this repository -> both subsets are
                              reported WITHHELD with ``n = 0`` (never invented,
                              never pooled).
(e) pressure-area exponent    alpha = 0.40...0.60 scan; closed-form ice energy
                              checked against Simpson integration, the numerical
                              scaling exponents compared with the ISO 19906
                              closed forms, and the identified exponents must
                              fall inside the ISO family interval; full
                              sensitivity table plus coupled-model exponents
                              (reported, not forced).
(f) negative control          no hardening + no ice side (constant pressure,
                              rigid indenter) reproduces N01 bit-for-bit and the
                              damage geometry coincides with the N01 box
                              ``L_c + 2*span``: the new model *contains* the old.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.hull_collision import (  # noqa: E402
    DATASET_STATUS_WITHHELD,
    DoubleHullSide,
    PressureAreaLaw,
    SideShellLayer,
    coefficient_uncertainty,
    collision_coefficient_from_share,
    collision_coefficient_from_volume,
    damage_geometry,
    damage_geometry_pressure_driven,
    energy_increment_numeric,
    energy_ledger,
    force_history,
    iso19906_scaling_exponents,
    kinetic_energy,
    load_collision_dataset,
    moment_increment_fibre,
    recalibrate_coefficient,
    solve_indentation,
    solve_penetration,
    structural_coefficients_per_length,
)
from tensordem.hull_collision import strain_hardening  # noqa: E402
from tensordem.hull_collision.scenario import (  # noqa: E402
    FRAME_SPAN,
    HARDENING_RATIO,
    ICE_CONTACT_HEIGHT,
    ICE_CONTACT_LENGTH,
    ICE_REFERENCE_PRESSURE,
    SCRATCH_SIGMA_Y,
    design_case,
    effective_mass,
    hardened_design_case,
    ice_pressure_law,
)

VELOCITIES = (2.0, 4.0, 6.0, 8.0)
THRESHOLDS = {
    "index_swap_relative": 1e-3,
    "hardening_increment_relative": 2e-2,
    "hardening_bound_slack": 0.10,
    "energy_residual_relative": 1e-6,
    "minorsky_median_relative": 0.15,
    "scaling_exponent_relative": 2e-2,
}
ICE_FACE_P0 = (2.0e6, 3.0e6, 5.0e6, 7.0e6)
ICE_FACE_ALPHA = (0.40, 0.45, 0.50, 0.55, 0.60)
XI_SWEEP_THICKNESS = (0.011, 0.022, 0.033, 0.044)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(a: float, b: float) -> float:
    if b == 0.0:
        return 0.0 if a == 0.0 else math.inf
    return abs(a - b) / abs(b)


def _with_thickness(structure: DoubleHullSide, thickness: float) -> DoubleHullSide:
    """Rebuild the design case with a different outer-shell thickness (xi sweep)."""
    layers = []
    for layer in structure.layers:
        if layer.name == "outer_shell":
            layers.append(
                SideShellLayer(
                    name=layer.name,
                    thickness=thickness,
                    sigma_y=layer.sigma_y,
                    frame_span=layer.frame_span,
                    effective_width=layer.effective_width,
                    n_longitudinals=layer.n_longitudinals,
                    longitudinal_Z=layer.longitudinal_Z,
                    longitudinal_area=layer.longitudinal_area,
                    crush_depth=layer.crush_depth,
                )
            )
        else:
            layers.append(layer)
    return DoubleHullSide(
        layers=tuple(layers), friction_coefficient=structure.friction_coefficient
    )


def _layer_allocations(side, depth: float) -> dict:
    out = {}
    for layer, u in zip(side.layers, side.local_penetrations(depth)):
        with_hardening = layer.energy(u)
        plastic = layer.base.energy(u)
        out[layer.name] = {
            "local_penetration_m": u,
            "energy_with_hardening_MJ": with_hardening / 1e6,
            "energy_perfectly_plastic_MJ": plastic / 1e6,
            "hardening_energy_MJ": (with_hardening - plastic) / 1e6,
            "hardening_fraction_of_layer": (
                (with_hardening - plastic) / with_hardening if with_hardening else 0.0
            ),
        }
    return out


# ---------------------------------------------------------------------------
# (a) hardening -> 0 index swap
# ---------------------------------------------------------------------------
def _criterion_a(structure: DoubleHullSide, effective: float) -> dict:
    plastic_side = strain_hardening.build_hardened_side(structure, hardening_ratio=0.0)
    rows = []
    for velocity in VELOCITIES:
        e_k = kinetic_energy(effective, velocity)
        n01 = solve_penetration(e_k, structure)
        new = solve_indentation(e_k, plastic_side)
        rows.append(
            {
                "velocity_m_s": velocity,
                "depth_n01_m": n01.penetration_depth,
                "depth_hardening_zero_m": new.penetration_depth,
                "depth_relative_error": _rel(new.penetration_depth, n01.penetration_depth),
                "structural_energy_n01_MJ": n01.structural_energy / 1e6,
                "structural_energy_hardening_zero_MJ": new.structural_energy / 1e6,
                "energy_relative_error": _rel(new.structural_energy, n01.structural_energy),
                "resistance_relative_error": _rel(
                    new.structural_resistance_N, n01.structural_resistance_N
                ),
                "layer_energy_relative_error": max(
                    _rel(a, b) for a, b in zip(new.layer_energies, n01.layer_energies)
                ),
                "hinge_energy_increment_J_per_m": sum(
                    layer.hinge.energy_increment(layer.hinge_rotation(u))
                    for layer, u in zip(plastic_side.layers, new.layer_local_penetrations)
                ),
            }
        )
    worst = max(
        max(
            row["depth_relative_error"],
            row["energy_relative_error"],
            row["resistance_relative_error"],
            row["layer_energy_relative_error"],
        )
        for row in rows
    )
    return {
        "rows": rows,
        "max_relative_error": worst,
        "threshold": THRESHOLDS["index_swap_relative"],
        "pass": worst <= THRESHOLDS["index_swap_relative"]
        and all(row["hinge_energy_increment_J_per_m"] == 0.0 for row in rows),
    }


# ---------------------------------------------------------------------------
# (b) hardening increment: closed form vs fibre/Simpson route vs bound
# ---------------------------------------------------------------------------
def _criterion_b() -> dict:
    side = hardened_design_case()
    rows = []
    for layer in side.layers:
        hinge = layer.hinge
        for angle in (0.10, 0.25, 0.40, 0.60, 0.885, 1.20, 1.60):
            closed = hinge.energy_increment(angle)
            numeric = energy_increment_numeric(hinge, angle, n_steps=96, n_fibres=2001)
            bound = hinge.energy_increment_upper_bound(angle)
            rows.append(
                {
                    "layer": layer.name,
                    "theta_rad": angle,
                    "hinge_length_m": hinge.hinge_length(angle),
                    "curvature_1_per_m": hinge.curvature(angle),
                    "peak_fibre_strain": hinge.peak_fibre_strain(angle),
                    "moment_increment_Nm_per_m": hinge.moment_increment(angle),
                    "moment_increment_ratio": hinge.moment_increment(angle)
                    / hinge.plastic_moment_per_width,
                    "energy_increment_closed_J_per_m": closed,
                    "energy_increment_numeric_J_per_m": numeric,
                    "energy_increment_relative_error": _rel(numeric, closed) if closed else 0.0,
                    "rigid_hardening_bound_J_per_m": bound,
                    "fraction_of_bound": closed / bound if bound else 0.0,
                }
            )
    moment_rows = []
    hinge = side.layers[0].hinge
    for kappa in (0.2, 0.5, 1.0, 5.0, 18.18, 40.0):
        fibre = moment_increment_fibre(
            kappa,
            sigma_y=hinge.sigma_y,
            thickness=hinge.thickness,
            hardening_modulus=hinge.hardening_modulus,
            elastic_modulus=hinge.elastic_modulus,
            n_fibres=8001,
        )
        closed = strain_hardening.bilinear_moment_increment_closed_form(
            kappa,
            sigma_y=hinge.sigma_y,
            thickness=hinge.thickness,
            hardening_modulus=hinge.hardening_modulus,
            elastic_modulus=hinge.elastic_modulus,
        )
        moment_rows.append(
            {
                "kappa_1_per_m": kappa,
                "moment_increment_fibre_Nm_per_m": fibre,
                "moment_increment_closed_Nm_per_m": closed,
                "relative_error": _rel(fibre, closed) if closed else 0.0,
            }
        )
    worst_energy = max(row["energy_increment_relative_error"] for row in rows)
    worst_moment = max(row["relative_error"] for row in moment_rows)
    fractions = [
        row["fraction_of_bound"]
        for row in rows
        if row["rigid_hardening_bound_J_per_m"] > 0.0
    ]
    developed = [
        row["fraction_of_bound"]
        for row in rows
        if row["theta_rad"] >= 0.885 and row["layer"] == "outer_shell"
    ]
    slack = 1.0 - min(developed) if developed else None
    return {
        "rows": rows,
        "moment_rows": moment_rows,
        "max_energy_relative_error": worst_energy,
        "max_moment_relative_error": worst_moment,
        "max_fraction_of_bound": max(fractions),
        "developed_fraction_of_bound": developed,
        "bound_slack": slack,
        "threshold_energy_relative": THRESHOLDS["hardening_increment_relative"],
        "threshold_bound_slack": THRESHOLDS["hardening_bound_slack"],
        "pass": worst_energy <= THRESHOLDS["hardening_increment_relative"]
        and worst_moment <= 1e-3
        and max(fractions) <= 1.0
        and (slack is not None and slack <= THRESHOLDS["hardening_bound_slack"]),
    }


# ---------------------------------------------------------------------------
# (c)+(d) sweep, coefficient table, empirical recalibration status
# ---------------------------------------------------------------------------
def _sweep_and_coefficients(structure, side, law, effective) -> tuple:
    plastic_side = strain_hardening.build_hardened_side(structure, hardening_ratio=0.0)
    sweep = []
    for velocity in VELOCITIES:
        e_k = kinetic_energy(effective, velocity)
        solution = solve_indentation(
            e_k,
            side,
            law=law,
            contact_height=ICE_CONTACT_HEIGHT,
            contact_length=ICE_CONTACT_LENGTH,
        )
        geometry, _ = damage_geometry_pressure_driven(
            side, solution, contact_length=ICE_CONTACT_LENGTH
        )
        ledger = energy_ledger(solution)
        n01 = solve_penetration(e_k, structure)
        n01_geometry = damage_geometry(structure, n01, contact_length=ICE_CONTACT_LENGTH)
        no_ice_depth = solve_indentation(e_k, plastic_side).penetration_depth
        c_model = collision_coefficient_from_volume(
            geometry.volume_m3, effective, velocity, SCRATCH_SIGMA_Y
        )
        c_n01 = collision_coefficient_from_volume(
            n01_geometry.volume_m3, effective, velocity, SCRATCH_SIGMA_Y
        )
        sweep.append(
            {
                "velocity_m_s": velocity,
                "kinetic_energy_MJ": e_k / 1e6,
                "depth_m": solution.penetration_depth,
                "depth_over_span": solution.penetration_depth / FRAME_SPAN,
                "depth_without_hardening_m": no_ice_depth,
                "depth_n01_rigid_m": n01.penetration_depth,
                "ice_load_MN": solution.ice_load_N / 1e6,
                "structural_resistance_MN": solution.structural_resistance_N / 1e6,
                "yield_length_m": solution.yield_length_m,
                "damage_length_m": solution.damage_length_m,
                "damage_length_n01_m": n01_geometry.length_m,
                "damage_volume_m3": geometry.volume_m3,
                "damage_volume_n01_m3": n01_geometry.volume_m3,
                "damage_volume_ratio_to_n01": geometry.volume_m3 / n01_geometry.volume_m3,
                "coefficient_C": c_model,
                "coefficient_C_n01": c_n01,
                "coefficient_ratio_to_n01": c_model / c_n01,
                "inner_shell_intact": solution.inner_shell_intact,
                "breached_layers": list(solution.breached_layers),
                "peak_fibre_strain_outer": solution.peak_fibre_strains[0],
                "ledger": ledger,
                "layer_allocation": _layer_allocations(side, solution.penetration_depth),
            }
        )

    identity_rows = []
    for row, velocity in zip(sweep, VELOCITIES):
        share = row["ledger"]["steel_share"]
        c_share = collision_coefficient_from_share(share, SCRATCH_SIGMA_Y)
        e_steel = row["ledger"]["steel_MJ"] * 1e6
        c_steel_volume = collision_coefficient_from_volume(
            e_steel / 47.1e6, effective, velocity, SCRATCH_SIGMA_Y
        )
        identity_rows.append(
            {
                "velocity_m_s": velocity,
                "coefficient_from_share": c_share,
                "coefficient_from_steel_energy_volume": c_steel_volume,
                "relative_difference": _rel(c_steel_volume, c_share) if c_share else 0.0,
                "coefficient_from_model_volume": row["coefficient_C"],
                "shape_closure_ratio": row["coefficient_C"] / c_share if c_share else None,
            }
        )

    coefficient_grid = []
    for thickness in XI_SWEEP_THICKNESS:
        variant = _with_thickness(structure, thickness)
        variant_side = strain_hardening.build_hardened_side(
            variant, hardening_ratio=HARDENING_RATIO
        )
        xi = thickness / FRAME_SPAN
        n01_variant = damage_geometry(
            variant, solve_penetration(kinetic_energy(effective, 4.0), variant),
            contact_length=ICE_CONTACT_LENGTH,
        )
        for p0 in ICE_FACE_P0:
            for alpha in ICE_FACE_ALPHA:
                face = PressureAreaLaw(
                    reference_pressure=p0, exponent=alpha, reference_area=1.0
                )
                velocity = 4.0
                e_k = kinetic_energy(effective, velocity)
                solution = solve_indentation(
                    e_k,
                    variant_side,
                    law=face,
                    contact_height=ICE_CONTACT_HEIGHT,
                    contact_length=ICE_CONTACT_LENGTH,
                )
                geometry, _ = damage_geometry_pressure_driven(
                    variant_side, solution, contact_length=ICE_CONTACT_LENGTH
                )
                coefficient_grid.append(
                    {
                        "xi_t_over_span": xi,
                        "thickness_m": thickness,
                        "reference_pressure_MPa": p0 / 1e6,
                        "exponent": alpha,
                        "depth_m": solution.penetration_depth,
                        "ice_share": solution.ice_energy / e_k,
                        "yield_length_m": solution.yield_length_m,
                        "damage_volume_m3": geometry.volume_m3,
                        "coefficient_C": collision_coefficient_from_volume(
                            geometry.volume_m3, effective, velocity, SCRATCH_SIGMA_Y
                        ),
                        "coefficient_ratio_to_n01": geometry.volume_m3 / n01_variant.volume_m3,
                        "damage_volume_n01_m3": n01_variant.volume_m3,
                    }
                )
    envelope_all = coefficient_uncertainty(row["coefficient_C"] for row in coefficient_grid)
    envelope_ratio = coefficient_uncertainty(
        row["coefficient_ratio_to_n01"] for row in coefficient_grid
    )
    envelope_by_xi = {}
    for thickness in XI_SWEEP_THICKNESS:
        xi = thickness / FRAME_SPAN
        subset = [
            row["coefficient_C"]
            for row in coefficient_grid
            if abs(row["xi_t_over_span"] - xi) < 1e-15
        ]
        ratio_subset = [
            row["coefficient_ratio_to_n01"]
            for row in coefficient_grid
            if abs(row["xi_t_over_span"] - xi) < 1e-15
        ]
        envelope_by_xi[f"{xi:.5f}"] = {
            "absolute": coefficient_uncertainty(subset),
            "ratio_to_n01": coefficient_uncertainty(ratio_subset),
        }

    dataset = load_collision_dataset(ROOT / "data" / "collision_cases.csv")
    ship_ship = recalibrate_coefficient(
        dataset["cases"], regime="ship_ship", threshold=THRESHOLDS["minorsky_median_relative"]
    )
    ice_region = recalibrate_coefficient(
        dataset["cases"], regime="ice", threshold=THRESHOLDS["minorsky_median_relative"]
    )
    mixed_probe = recalibrate_coefficient(
        [
            {
                "case": "probe_a",
                "regime": "ship_ship",
                "mass_kg": 1.0e7,
                "velocity_m_s": 5.0,
                "sigma_y_Pa": SCRATCH_SIGMA_Y,
                "volume_m3": 20.0,
            },
            {
                "case": "probe_b",
                "regime": "ice",
                "mass_kg": 2.2e6,
                "velocity_m_s": 4.0,
                "sigma_y_Pa": SCRATCH_SIGMA_Y,
                "volume_m3": 3.0,
            },
        ],
        regime="ice",
    )
    no_pooling_ok = (
        mixed_probe["n"] == 1 and mixed_probe["status"] != DATASET_STATUS_WITHHELD
    )

    status = (
        DATASET_STATUS_WITHHELD
        if ship_ship["n"] == 0 and ice_region["n"] == 0
        else "PRESENT"
    )
    minorsky_block = {
        "dataset": {k: v for k, v in dataset.items() if k != "cases"},
        "ship_ship_subset": {k: v for k, v in ship_ship.items() if k != "residuals"},
        "ice_region_subset": {k: v for k, v in ice_region.items() if k != "residuals"},
        "no_pooling_invariant_holds": no_pooling_ok,
        "threshold_median_relative": THRESHOLDS["minorsky_median_relative"],
        "status": status,
        "pass": None,
    }

    identity_worst = max(row["relative_difference"] for row in identity_rows)
    residuals = [row["ledger"]["residual_relative"] for row in sweep]
    criterion_c = {
        "max_residual_relative": max(residuals),
        "threshold": THRESHOLDS["energy_residual_relative"],
        "ledgers_close": all(row["ledger"]["closes"] for row in sweep),
        "pass": max(residuals) <= THRESHOLDS["energy_residual_relative"],
    }
    criterion_d = {
        "coefficient_identity_max_relative_difference": identity_worst,
        "no_pooling_invariant_holds": no_pooling_ok,
        "empirical_residual_status": status,
        "ship_ship_n": ship_ship["n"],
        "ice_region_n": ice_region["n"],
        "withheld_reason": dataset["reason"],
        "pass": None if status == DATASET_STATUS_WITHHELD else ship_ship["meets_threshold"],
        "note": "empirical residual (median |dC|) is WITHHELD: no case table in the repository "
        "and no network access; the coefficient correction form and its uncertainty envelope "
        "are reported instead of a fabricated fit",
    }
    return (
        sweep,
        identity_rows,
        coefficient_grid,
        envelope_all,
        envelope_by_xi,
        envelope_ratio,
        minorsky_block,
        criterion_c,
        criterion_d,
    )

# ---------------------------------------------------------------------------
# (e) pressure-area exponent scan
# ---------------------------------------------------------------------------
def _loglog_slope(xs, ys) -> float:
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )


def _criterion_e(effective: float) -> dict:
    closed_rows, exponent_rows = [], []
    kappa1, kappa2 = structural_coefficients_per_length(design_case().layers[0])
    height = ICE_CONTACT_HEIGHT
    for alpha in ICE_FACE_ALPHA:
        face = PressureAreaLaw(reference_pressure=3.0e6, exponent=alpha, reference_area=1.0)
        depth = 0.25
        closed = face.ice_energy(height, depth)
        numeric = face.ice_energy_numeric(height, depth, n_steps=20000)
        closed_rows.append(
            {
                "exponent": alpha,
                "depth_m": depth,
                "contact_area_m2": face.contact_area(height, depth),
                "pressure_MPa": face.pressure(face.contact_area(height, depth)) / 1e6,
                "ice_load_N": face.ice_load(height, depth),
                "ice_energy_closed_J": closed,
                "ice_energy_simpson_J": numeric,
                "energy_relative_error": _rel(numeric, closed),
            }
        )

        energies = [10.0e6, 14.0e6, 20.0e6, 28.0e6, 40.0e6, 56.0e6]
        depths = []
        for energy in energies:
            lo, hi = 1e-9, 1.0
            while face.ice_energy(height, hi) < energy:  # adaptive bracket
                hi *= 2.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if face.ice_energy(height, mid) < energy:
                    lo = mid
                else:
                    hi = mid
            depths.append(0.5 * (lo + hi))
        xs = [math.log(e) for e in energies]
        slope_depth = _loglog_slope(xs, [math.log(d) for d in depths])
        # hinge-dominated yielding strip: L_y = N_ice/(kappa1*d)
        lengths_hinge = [face.ice_load(height, d) / (kappa1 * d) for d in depths]
        # membrane-dominated yielding strip: L_y = N_ice/(kappa2*d^2)
        lengths_membrane = [face.ice_load(height, d) / (kappa2 * d * d) for d in depths]
        slope_hinge = _loglog_slope(xs, [math.log(v) for v in lengths_hinge])
        slope_membrane = _loglog_slope(xs, [math.log(v) for v in lengths_membrane])
        analytic = iso19906_scaling_exponents(alpha)
        iso_membrane = -(1.0 + alpha) / (2.0 - alpha)
        exponent_rows.append(
            {
                "exponent": alpha,
                "kappa1_N_per_m2": kappa1,
                "kappa2_N_per_m2": kappa2,
                "penetration_exponent_numeric": slope_depth,
                "penetration_exponent_iso": analytic["penetration_vs_energy"],
                "penetration_exponent_relative_error": _rel(
                    slope_depth, analytic["penetration_vs_energy"]
                ),
                "yield_length_exponent_numeric_hinge": slope_hinge,
                "yield_length_exponent_iso_hinge": analytic["yield_length_vs_energy"],
                "yield_length_exponent_relative_error": _rel(
                    slope_hinge, analytic["yield_length_vs_energy"]
                ),
                "yield_length_exponent_numeric_membrane": slope_membrane,
                "yield_length_exponent_iso_membrane": iso_membrane,
                "yield_length_membrane_relative_error": _rel(slope_membrane, iso_membrane),
                "yield_length_hinge_m": lengths_hinge,
                "yield_length_membrane_m": lengths_membrane,
            }
        )

    coupled_rows = []
    side = hardened_design_case()
    for alpha in ICE_FACE_ALPHA:
        face = PressureAreaLaw(
            reference_pressure=ICE_REFERENCE_PRESSURE, exponent=alpha, reference_area=1.0
        )
        energies, depths, lengths, shares, loads = [], [], [], [], []
        for velocity in (2.0, 3.0, 4.0, 6.0, 8.0):
            e_k = kinetic_energy(effective, velocity)
            solution = solve_indentation(
                e_k,
                side,
                law=face,
                contact_height=ICE_CONTACT_HEIGHT,
                contact_length=ICE_CONTACT_LENGTH,
            )
            energies.append(e_k)
            depths.append(solution.penetration_depth)
            lengths.append(solution.yield_length_m)
            shares.append(solution.ice_energy / e_k)
            loads.append(solution.ice_load_N)
        xs = [math.log(e) for e in energies]
        coupled_rows.append(
            {
                "exponent": alpha,
                "penetration_exponent_coupled": _loglog_slope(
                    xs, [math.log(d) for d in depths]
                ),
                "yield_length_exponent_coupled": _loglog_slope(
                    xs, [math.log(v) for v in lengths]
                ),
                "ice_share_min": min(shares),
                "ice_share_max": max(shares),
                "kinetic_energy_MJ": [e / 1e6 for e in energies],
                "depths_m": depths,
                "yield_lengths_m": lengths,
                "ice_loads_MN": [f / 1e6 for f in loads],
            }
        )

    interval = sorted([1.0 / (2.0 - 0.60), 1.0 / (2.0 - 0.40)])
    worst_exponent = max(
        max(
            row["penetration_exponent_relative_error"],
            row["yield_length_exponent_relative_error"],
            row["yield_length_membrane_relative_error"],
        )
        for row in exponent_rows
    )
    worst_closed = max(row["energy_relative_error"] for row in closed_rows)
    inside = all(
        interval[0] - 1e-12 <= row["penetration_exponent_numeric"] <= interval[1] + 1e-12
        for row in exponent_rows
    )
    return {
        "closed_form_rows": closed_rows,
        "exponent_rows": exponent_rows,
        "coupled_rows": coupled_rows,
        "iso_family_interval_penetration_exponent": interval,
        "max_closed_form_relative_error": worst_closed,
        "max_exponent_relative_error": worst_exponent,
        "numerical_exponents_inside_iso_interval": inside,
        "closed_form_threshold": 1e-6,
        "exponent_threshold": THRESHOLDS["scaling_exponent_relative"],
        "pass": worst_closed <= 1e-6
        and worst_exponent <= THRESHOLDS["scaling_exponent_relative"]
        and inside,
        "note": "the ISO exponent family {alpha} -> {1/(2-alpha)} is the reference "
        "interval; the coupled (steel + ice) exponents in coupled_rows are reported "
        "separately because steel membrane absorption shifts them away from the "
        "ice-only asymptote - reported, not forced into the interval",
    }


# ---------------------------------------------------------------------------
# (f) negative control
# ---------------------------------------------------------------------------
def _criterion_f(structure, effective: float) -> dict:
    plastic_side = strain_hardening.build_hardened_side(structure, hardening_ratio=0.0)
    constant_pressure = PressureAreaLaw(
        reference_pressure=3.0e6, exponent=0.0, reference_area=1.0
    )
    box_length = ICE_CONTACT_LENGTH + 2.0 * FRAME_SPAN
    rows = []
    for velocity in VELOCITIES:
        e_k = kinetic_energy(effective, velocity)
        n01 = solve_penetration(e_k, structure)
        n01_geometry = damage_geometry(structure, n01, contact_length=ICE_CONTACT_LENGTH)
        new = solve_indentation(e_k, plastic_side)  # law=None -> rigid indenter
        geometry, _ = damage_geometry_pressure_driven(
            plastic_side, new, contact_length=ICE_CONTACT_LENGTH, length=box_length
        )
        rows.append(
            {
                "velocity_m_s": velocity,
                "depth_relative_error": _rel(new.penetration_depth, n01.penetration_depth),
                "structural_energy_relative_error": _rel(
                    new.structural_energy, n01.structural_energy
                ),
                "friction_energy_relative_error": _rel(
                    new.friction_energy, n01.friction_energy
                ),
                "ice_energy_J": new.ice_energy,
                "damage_length_n01_m": n01_geometry.length_m,
                "damage_length_control_m": geometry.length_m,
                "damage_volume_relative_error": _rel(
                    geometry.volume_m3, n01_geometry.volume_m3
                ),
                "resistance_relative_error": _rel(
                    new.structural_resistance_N, n01.structural_resistance_N
                ),
            }
        )
    worst = max(
        max(
            row["depth_relative_error"],
            row["structural_energy_relative_error"],
            row["friction_energy_relative_error"],
            row["damage_volume_relative_error"],
            row["resistance_relative_error"],
        )
        for row in rows
    )
    height, depth = ICE_CONTACT_HEIGHT, 0.25
    linear_energy = constant_pressure.ice_energy(height, depth)
    analytic_linear = constant_pressure.reference_pressure * height * depth * depth / 2.0
    return {
        "rows": rows,
        "max_relative_error": worst,
        "constant_pressure_energy_J": linear_energy,
        "constant_pressure_energy_analytic_J": analytic_linear,
        "constant_pressure_relative_error": _rel(linear_energy, analytic_linear),
        "threshold": 1e-9,
        "pass": worst <= 1e-9 and _rel(linear_energy, analytic_linear) <= 1e-9,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _figures(output: Path, sweep, hardening, exponent, coupled, grid) -> list:
    figures = []
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    side = hardened_design_case()
    law = ice_pressure_law()

    # 1: force history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for velocity, colour in ((4.0, "tab:blue"), (8.0, "tab:red")):
        history = force_history(
            side,
            kinetic_energy=kinetic_energy(effective_mass(), velocity),
            effective_mass=effective_mass(),
            law=law,
            contact_height=ICE_CONTACT_HEIGHT,
            contact_length=ICE_CONTACT_LENGTH,
            n_points=49,
        )
        axes[0].plot(
            [d * 1e3 for d in history["depth_m"]],
            [f / 1e6 for f in history["load_N"]],
            color=colour,
            label=f"V = {velocity:.0f} m/s",
        )
        axes[1].plot(
            history["time_s"],
            [f / 1e6 for f in history["load_N"]],
            color=colour,
            label=f"V = {velocity:.0f} m/s",
        )
    axes[0].set_xlabel("penetration depth [mm]")
    axes[0].set_ylabel("contact force [MN]")
    axes[1].set_xlabel("quasi-static time [s]")
    axes[1].set_ylabel("contact force [MN]")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("p256w1-N04 force history (hardening + ISO 19906 ice load)")
    fig.tight_layout()
    path = figure_dir / "force_history.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(
        {
            "path": str(path.relative_to(output)),
            "caption": "Left: contact force vs penetration. Right: quasi-static "
            "force-time history (V = 4 / 8 m/s, hardening ratio 1 %, alpha = 0.5).",
        }
    )

    # 2: hardening law
    outer = [row for row in hardening["rows"] if row["layer"] == "outer_shell"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(
        [row["theta_rad"] for row in outer],
        [row["moment_increment_ratio"] for row in outer],
        "o-",
        label="dM_h / M_p (closed form)",
    )
    axes[0].set_xlabel("hinge rotation [rad]")
    axes[0].set_ylabel("hardening moment ratio [-]")
    axes[1].plot(
        [row["theta_rad"] for row in outer],
        [row["energy_increment_closed_J_per_m"] / 1e3 for row in outer],
        "o-",
        label="closed form",
    )
    axes[1].plot(
        [row["theta_rad"] for row in outer],
        [row["energy_increment_numeric_J_per_m"] / 1e3 for row in outer],
        "x--",
        label="fibre quadrature + Simpson",
    )
    axes[1].plot(
        [row["theta_rad"] for row in outer],
        [row["rigid_hardening_bound_J_per_m"] / 1e3 for row in outer],
        ":",
        label="rigid-hardening bound",
    )
    axes[1].set_xlabel("hinge rotation [rad]")
    axes[1].set_ylabel("hardening energy increment [kJ/m]")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("p256w1-N04 bilinear hinge: moment and energy increment")
    fig.tight_layout()
    path = figure_dir / "hardening_law.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(
        {
            "path": str(path.relative_to(output)),
            "caption": "Bilinear hinge: hardening moment ratio dM_h/M_p and hardening "
            "energy increment per unit width, closed form vs independent fibre/Simpson "
            "route vs the rigid-hardening bound.",
        }
    )

    # 3: depth and volume vs velocity, N01 vs hardening + ice
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    velocities = [row["velocity_m_s"] for row in sweep]
    axes[0].plot(velocities, [row["depth_n01_rigid_m"] for row in sweep], "s--",
                 label="N01 (perfectly plastic, rigid ice)")
    axes[0].plot(velocities, [row["depth_without_hardening_m"] for row in sweep], "o:",
                 label="hardening 0 % + ice")
    axes[0].plot(velocities, [row["depth_m"] for row in sweep], "o-",
                 label="hardening 1 % + ice")
    axes[0].set_xlabel("impact velocity [m/s]")
    axes[0].set_ylabel("penetration depth [m]")
    axes[1].plot(velocities, [row["damage_volume_n01_m3"] for row in sweep], "s--",
                 label="N01 box geometry")
    axes[1].plot(velocities, [row["damage_volume_m3"] for row in sweep], "o-",
                 label="pressure-integral geometry")
    axes[1].set_xlabel("impact velocity [m/s]")
    axes[1].set_ylabel("damaged steel volume [m3]")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("p256w1-N04 penetration and damage volume: new model vs N01")
    fig.tight_layout()
    path = figure_dir / "depth_volume_comparison.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(
        {
            "path": str(path.relative_to(output)),
            "caption": "Penetration depth and damaged steel volume vs impact velocity: "
            "N01 rigid-indenter surface, ice side only, and full hardening + ice model.",
        }
    )

    # 4: exponent scan and coefficient envelope
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    alphas = [row["exponent"] for row in exponent["exponent_rows"]]
    axes[0].plot(alphas, [row["penetration_exponent_numeric"] for row in exponent["exponent_rows"]],
                 "o-", label="numerical (ISO asymptote)")
    axes[0].plot(alphas, [row["penetration_exponent_iso"] for row in exponent["exponent_rows"]],
                 "x:", label="1/(2-alpha) closed form")
    axes[0].plot(alphas, [row["penetration_exponent_coupled"] for row in coupled],
                 "^-", label="coupled steel + ice")
    low, high = exponent["iso_family_interval_penetration_exponent"]
    axes[0].axhspan(low, high, color="tab:green", alpha=0.12, label="ISO family interval")
    axes[0].set_xlabel("pressure-area exponent alpha [-]")
    axes[0].set_ylabel("penetration exponent d ~ E_k**beta")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    xis = sorted({row["xi_t_over_span"] for row in grid})
    for xi in xis:
        subset = [row for row in grid if abs(row["xi_t_over_span"] - xi) < 1e-15]
        axes[1].scatter(
            [row["exponent"] for row in subset],
            [row["coefficient_C"] for row in subset],
            s=18,
            label=f"xi = {xi:.4f}",
        )
    axes[1].set_xlabel("pressure-area exponent alpha [-]")
    axes[1].set_ylabel("recalibrated coefficient C [-]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.suptitle("p256w1-N04 exponent scan and C(xi, ice face)")
    fig.tight_layout()
    path = figure_dir / "exponent_and_coefficient.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(
        {
            "path": str(path.relative_to(output)),
            "caption": "Left: penetration-depth scaling exponent vs pressure-area exponent "
            "(numerical ISO asymptote, closed form, coupled model, ISO family interval). "
            "Right: recalibrated collision coefficient C over the xi x ice-face box.",
        }
    )
    return figures


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)

    structure = design_case()
    side = hardened_design_case()
    law = ice_pressure_law()
    effective = effective_mass()

    criterion_a = _criterion_a(structure, effective)
    criterion_b = _criterion_b()
    (
        sweep,
        identity_rows,
        coefficient_grid,
        envelope_all,
        envelope_by_xi,
        envelope_ratio,
        minorsky_block,
        criterion_c,
        criterion_d,
    ) = _sweep_and_coefficients(structure, side, law, effective)
    criterion_e = _criterion_e(effective)
    criterion_f = _criterion_f(structure, effective)

    history_rows = []
    for velocity in (4.0, 8.0):
        history = force_history(
            side,
            kinetic_energy=kinetic_energy(effective, velocity),
            effective_mass=effective,
            law=law,
            contact_height=ICE_CONTACT_HEIGHT,
            contact_length=ICE_CONTACT_LENGTH,
            n_points=41,
        )
        for depth, load, speed, time in zip(
            history["depth_m"], history["load_N"], history["velocity_m_s"], history["time_s"]
        ):
            history_rows.append(
                {
                    "velocity_m_s": velocity,
                    "depth_m": depth,
                    "load_MN": load / 1e6,
                    "velocity_remaining_m_s": speed,
                    "time_s": time,
                }
            )
    with (output / "force_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    with (output / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "velocity_m_s",
            "kinetic_energy_MJ",
            "depth_m",
            "depth_n01_rigid_m",
            "depth_without_hardening_m",
            "ice_load_MN",
            "yield_length_m",
            "damage_length_m",
            "damage_volume_m3",
            "damage_volume_n01_m3",
            "coefficient_C",
            "coefficient_C_n01",
            "steel_share",
            "ice_share",
            "residual_relative",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sweep:
            writer.writerow(
                {
                    "velocity_m_s": row["velocity_m_s"],
                    "kinetic_energy_MJ": row["kinetic_energy_MJ"],
                    "depth_m": row["depth_m"],
                    "depth_n01_rigid_m": row["depth_n01_rigid_m"],
                    "depth_without_hardening_m": row["depth_without_hardening_m"],
                    "ice_load_MN": row["ice_load_MN"],
                    "yield_length_m": row["yield_length_m"],
                    "damage_length_m": row["damage_length_m"],
                    "damage_volume_m3": row["damage_volume_m3"],
                    "damage_volume_n01_m3": row["damage_volume_n01_m3"],
                    "coefficient_C": row["coefficient_C"],
                    "coefficient_C_n01": row["coefficient_C_n01"],
                    "steel_share": row["ledger"]["steel_share"],
                    "ice_share": row["ledger"]["ice_share"],
                    "residual_relative": row["ledger"]["residual_relative"],
                }
            )

    with (output / "coefficient_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coefficient_grid[0].keys()))
        writer.writeheader()
        writer.writerows(coefficient_grid)

    figures = _figures(
        output, sweep, criterion_b, criterion_e, criterion_e["coupled_rows"], coefficient_grid
    )

    criteria = {
        "criterion_a_hardening_zero_index_swap": criterion_a,
        "criterion_b_hardening_increment": criterion_b,
        "criterion_c_energy_conservation": criterion_c,
        "criterion_d_minorsky_recalibration": criterion_d,
        "criterion_e_pressure_area_exponent_scan": criterion_e,
        "criterion_f_negative_control": criterion_f,
    }
    failed = [name for name, result in criteria.items() if result.get("pass") is False]
    withheld = [name for name, result in criteria.items() if result.get("pass") is None]
    verdict = "FAIL" if failed else ("PASS_WITH_WITHHELD" if withheld else "PASS")

    report = {
        "node": "p256w1-n04-ice-collision-hardening",
        "title": "Strain-hardening hinges + ISO 19906 pressure-area indentation: "
        "ice-region collision energy and damage geometry",
        "verdict": verdict,
        "failed_checks": failed,
        "withheld_checks": withheld,
        "thresholds": THRESHOLDS,
        "scenario": {
            "outer_thickness_m": structure.layers[0].thickness,
            "inner_thickness_m": structure.layers[1].thickness,
            "frame_span_m": FRAME_SPAN,
            "sigma_y_MPa": SCRATCH_SIGMA_Y / 1e6,
            "hardening_ratio_Eh_over_E": HARDENING_RATIO,
            "rotation_capacity": side.layers[0].hinge.rotation_capacity,
            "hinge_length_floor_m": side.layers[0].hinge.hinge_length_floor,
            "ice_reference_pressure_MPa": law.reference_pressure / 1e6,
            "ice_pressure_exponent": law.exponent,
            "ice_contact_height_m": ICE_CONTACT_HEIGHT,
            "ice_contact_length_m": ICE_CONTACT_LENGTH,
            "effective_mass_kg": effective,
            "friction_coefficient": side.friction_coefficient,
            "outer_hinge_coefficient_N_per_m": side.layers[0].base.hinge_coefficient,
            "outer_membrane_coefficient_N_per_m2": side.layers[0].base.membrane_coefficient,
            "outer_capacity_with_hardening_MJ": side.layers[0].capacity / 1e6,
            "inner_capacity_with_hardening_MJ": side.layers[1].capacity / 1e6,
            "kappa1_outer_N_per_m2": structural_coefficients_per_length(side.layers[0].base)[0],
            "kappa2_outer_N_per_m2": structural_coefficients_per_length(side.layers[0].base)[1],
        },
        "sweep": sweep,
        "coefficient_identity": identity_rows,
        "coefficient_grid_summary": {
            "envelope_all": envelope_all,
            "envelope_ratio_to_n01": envelope_ratio,
            "envelope_by_xi": envelope_by_xi,
            "n_cases": len(coefficient_grid),
            "minorsky_reference_coefficient": None,
        },
        "minorsky": minorsky_block,
        "criteria": criteria,
        "figures": [
            {"path": figure["path"], "caption": figure["caption"], "sha256": _sha256_file(output / figure["path"])}
            for figure in figures
        ],
        "limitations": [
            "Bilinear hardening applied with the *plating* section ratio on the whole hinge term "
            "of a layer; the deeper longitudinal sections carry a larger relative hardening "
            "reserve, so the layer hardening increment is a lower bound (documented closure).",
            "Hinge length is capped by the rotation capacity (peak fibre strain <= 0.20); beyond "
            "that the hinge spreads instead of hardening further. Fracture/tearing is NOT "
            "modelled, so rotations past the capacity are a modelling extrapolation.",
            "No strain-rate sensitivity on steel or ice, no ice fracture/rubble, no elastic "
            "contact compliance, quasi-static monotone indentation.",
            "The pressure-area law is applied to the nominal contact area of the growing "
            "indentation (global relation); local ice-feature pressure spikes are not resolved.",
            "The empirical Minorsky recalibration residual is WITHHELD: no collision case table "
            "in the repository and no network access; C(xi, ice face) and its envelope are "
            "reported instead of a fitted residual.",
            "Quasi-static force-time history is energy-parameterised, not a dynamic impact "
            "simulation (no added-mass coupling in the deceleration).",
        ],
        "sources": [
            "src/tensordem/hull_collision/strain_hardening.py",
            "src/tensordem/hull_collision/indentation.py",
            "src/tensordem/hull_collision/recalibration.py",
            "src/tensordem/hull_collision/scenario.py",
            "src/tensordem/hull_collision/resistance.py",
            "src/tensordem/hull_collision/plastic_hinge.py",
            "src/tensordem/hull_collision/minorsky.py",
            "scripts/benchmark_ice_collision_hardening.py",
            "tests/test_ice_collision_hardening.py",
        ],
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["audit"] = {
        "report_content_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_sha256": {
            path: _sha256_file(ROOT / path)
            for path in report["sources"]
            if (ROOT / path).exists()
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs" / "ice_collision_hardening"
    )
    arguments = parser.parse_args()
    report = run(arguments.output)
    summary = {
        "verdict": report["verdict"],
        "failed_checks": report["failed_checks"],
        "withheld_checks": report["withheld_checks"],
        "criteria": {
            name: {
                key: value
                for key, value in result.items()
                if key in ("pass", "max_relative_error", "max_energy_relative_error",
                           "max_residual_relative", "max_exponent_relative_error",
                           "threshold", "status")
            }
            for name, result in report["criteria"].items()
        },
        "depths_m": [round(row["depth_m"], 4) for row in report["sweep"]],
        "depths_n01_m": [round(row["depth_n01_rigid_m"], 4) for row in report["sweep"]],
        "coefficient_C": [round(row["coefficient_C"], 4) for row in report["sweep"]],
        "output": str(arguments.output),
    }
    print(json.dumps(summary, indent=2))
    return 0 if report["verdict"] in ("PASS", "PASS_WITH_WITHHELD") else 1


if __name__ == "__main__":
    raise SystemExit(main())
