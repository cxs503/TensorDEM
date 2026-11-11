#!/usr/bin/env python3
"""Assemble the machine-readable maturity evidence for this node.

Two artifacts are produced, both under ``docs/ice_thermal_stress/`` (the only
place the push gate keeps JSON/PNG):

``benchmark_table.json``
    reference vs measured for every quantity the node asked to be *measured*,
    each row carrying its own reference identity and error.

``MATURITY_EVIDENCE.json``
    the six-point delivery manifest (``impl_files``, ``entry``,
    ``run_artifacts``, ``integration_callers``, ``benchmark``, ``repro_cmd``)
    plus the test list and the honest-unknown declarations.

Usage::

    PYTHONPATH=src python scripts/ice_thermal_stress_maturity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.contact_impact import pair_impact                     # noqa: E402
from tensordem.ice_thermal_stress import dem_coupling as dc          # noqa: E402
from tensordem.ice_thermal_stress.properties import ice_properties   # noqa: E402

OUT = ROOT / "docs" / "ice_thermal_stress"
REPORT_SECTION = "docs/ICE_THERMAL_STRESS_COUPLING.md"


def load_validation() -> dict:
    path = OUT / "validation.json"
    if not path.exists():
        raise SystemExit(
            "docs/ice_thermal_stress/validation.json missing -- run "
            "`PYTHONPATH=src python benchmarks/ice_thermal_stress/run_validation.py` first")
    return json.loads(path.read_text())


def pct(x: float) -> float:
    return 100.0 * float(x)


def build_rows(val: dict) -> list[dict]:
    c = val["criteria"]
    impact = val["impact"]
    rows = [
        {"quantity": "C1 max temperature error vs Carslaw & Jaeger constant-flux solution",
         "reference": "analytic semi-infinite solid, q = 1e4 W/m^2",
         "ref": 0.0, "got": c["C1_semi_infinite"]["max_abs_error_k"],
         "unit": "K", "error_pct": pct(c["C1_semi_infinite"]["relative_error"]),
         "tolerance_pct": pct(c["C1_semi_infinite"]["tolerance"])},
        {"quantity": "C2 max temperature error vs heated-slab sine series",
         "reference": "exact series solution, volumetric source 1e6 W/m^3",
         "ref": 0.0, "got": c["C2_slab_source"]["max_abs_error_k"],
         "unit": "K", "error_pct": pct(c["C2_slab_source"]["relative_error"]),
         "tolerance_pct": pct(c["C2_slab_source"]["tolerance"])},
        {"quantity": "C3 FV vs independent nodal FEM temperature",
         "reference": "FE linear elements, lumped mass",
         "ref": 0.0, "got": c["C3_fv_vs_fem"]["max_abs_difference_k"],
         "unit": "K", "error_pct": pct(c["C3_fv_vs_fem"]["relative_to_peak"]),
         "tolerance_pct": pct(c["C3_fv_vs_fem"]["tolerance"])},
        {"quantity": "C4 observed spatial order of accuracy",
         "reference": "2 (formal order of the theta/FV scheme)",
         "ref": 2.0, "got": c["C4_convergence_order"]["mean_order_last_two"],
         "unit": "-", "error_pct": pct(abs(c["C4_convergence_order"]["mean_order_last_two"] - 2.0) / 2.0),
         "tolerance_pct": 0.0},
        {"quantity": "C5 thermal stress field vs analytic restrained gradient",
         "reference": "sigma(z) = -E a_T (T - <T>)/(1-nu)",
         "ref": 0.0, "got": c["C5_stress_field"]["max_abs_difference_pa"],
         "unit": "Pa", "error_pct": pct(c["C5_stress_field"]["relative_to_peak"]),
         "tolerance_pct": pct(c["C5_stress_field"]["tolerance"])},
        {"quantity": "C5b linear-gradient peak stress vs closed form",
         "reference": "E a_T dT/(2(1-nu)), dT = 20 K",
         "ref": c["C5b_linear_gradient_anchor"]["exact_peak_pa"],
         "got": c["C5b_linear_gradient_anchor"]["measured_peak_pa"],
         "unit": "Pa", "error_pct": pct(c["C5b_linear_gradient_anchor"]["relative_error"]),
         "tolerance_pct": pct(c["C5b_linear_gradient_anchor"]["tolerance"])},
        {"quantity": "C6 membrane resultant of the thermal field (static admissibility)",
         "reference": "0 (self-equilibrated temperature field cannot build a net axial force)",
         "ref": 0.0, "got": c["C6_equilibrium"]["residual"], "unit": "-",
         "error_pct": pct(abs(c["C6_equilibrium"]["residual"])), "tolerance_pct": 0.0},
        {"quantity": "C7 restrained/free stress ratio",
         "reference": 2.0, "got": c["C7_restrained_free_ratio"]["measured"], "unit": "-",
         "error_pct": pct(abs(c["C7_restrained_free_ratio"]["measured"] - 2.0) / 2.0),
         "tolerance_pct": 0.0},
        {"quantity": "C8 dissipated energy vs 0.5 m v0^2 (1-e^2)",
         "reference": 0.5 * impact["case"].get("mass_kg", impact.get("mass_kg", 0.0)) or None,
         "got": c["C8_energy_ledger"]["dissipated_relative_error"], "unit": "-",
         "error_pct": pct(c["C8_energy_ledger"]["dissipated_relative_error"]),
         "tolerance_pct": pct(c["C8_energy_ledger"]["tolerance"])},
        {"quantity": "C8 heat -> stored enthalpy closure",
         "reference": 0.0, "got": c["C8_energy_ledger"]["heat_to_enthalpy_relative_error"],
         "unit": "-", "error_pct": pct(c["C8_energy_ledger"]["heat_to_enthalpy_relative_error"]),
         "tolerance_pct": pct(c["C8_energy_ledger"]["tolerance"])},
        {"quantity": "C9 coefficient of restitution, measured vs prescribed",
         "reference": c["C9_restitution"]["prescribed"], "got": c["C9_restitution"]["measured"],
         "unit": "-", "error_pct": pct(c["C9_restitution"]["abs_error"]),
         "tolerance_pct": pct(c["C9_restitution"]["tolerance"])},
        {"quantity": "C10 lumped-capacity vs resolved conduction peak rise",
         "reference": c["C10_lumped_vs_fvm"]["adiabatic_estimate_k"],
         "got": c["C10_lumped_vs_fvm"]["conduction_solution_k"], "unit": "K",
         "error_pct": pct(abs(c["C10_lumped_vs_fvm"]["ratio"] - 1.0)),
         "tolerance_pct": 10.0},
        {"quantity": "C13 mid-contact source vs constant-flux semi-infinite rise",
         "reference": c["C13_midcontact_source"]["analytic_rise_k"],
         "got": c["C13_midcontact_source"]["numeric_rise_k"], "unit": "K",
         "error_pct": pct(c["C13_midcontact_source"]["relative_error"]),
         "tolerance_pct": pct(c["C13_midcontact_source"]["tolerance"])},
    ]
    return rows


def bridge_rows() -> tuple[list[dict], dict]:
    """Solver-integration numbers: IceDEM3D pair impact -> thermal chain."""
    res = pair_impact(e_target=0.5, v0=0.5, radius=0.05, k_n=5.0e4, density=900.0,
                      thermal=True, bulk_temperature_c=-10.0,
                      thermal_cells=60, thermal_dt_s=5.0e-5)
    state = res.thermal_state
    props = ice_properties(-10.0)
    e_star = props.youngs_modulus_pa / (2.0 * (1.0 - props.poisson ** 2))
    closed = res.pair_mass_kg * 0.5 ** 2 * (1.0 - res.e_measured ** 2)
    geo = dc.couple_impact_result(
        res, bulk_temperature_c=-10.0, effective_radius_m=0.025,
        effective_modulus_pa=e_star, peak_overlap_m=res.max_overlap, cells=60, dt_s=5.0e-5)
    hertz = dc.couple_impact_result(
        res, bulk_temperature_c=-10.0, effective_radius_m=0.025,
        effective_modulus_pa=e_star, cells=60, dt_s=5.0e-5)
    rows = [
        {"quantity": "DEM bridge: solver dissipated work vs m v0^2 (1-e^2)",
         "reference": closed, "got": res.dissipated_energy, "unit": "J",
         "error_pct": pct(abs(res.dissipated_energy - closed) / closed),
         "tolerance_pct": 1.0},
        {"quantity": "DEM bridge: heat in ice vs beta*lambda*W",
         "reference": dc.TAYLOR_QUINNEY_BETA * res.dissipated_energy,
         "got": state.heat_ice_j, "unit": "J",
         "error_pct": pct(abs(state.heat_ice_j
                              - dc.TAYLOR_QUINNEY_BETA * res.dissipated_energy)
                          / state.heat_ice_j),
         "tolerance_pct": 1.0},
        {"quantity": "DEM bridge: stored enthalpy vs heat in ice",
         "reference": state.heat_ice_j, "got": state.energy_ledger["block_enthalpy_j"],
         "unit": "J",
         "error_pct": pct(state.energy_ledger["heat_to_enthalpy_relative_error"]),
         "tolerance_pct": 1.0},
        {"quantity": "DEM bridge: peak temperature rise (no latent heat, patch geometry band)",
         "reference": None, "got": state.peak_delta_t_k, "unit": "K",
         "error_pct": None, "tolerance_pct": None,
         "note": ("no experimental calorimetry exists offline; the two admissible patch "
                  "radii differ by 4 orders of magnitude in dT, see patch_geometry_band")},
    ]
    summary = {
        "case": {"e_target": 0.5, "v0_m_s": 0.5, "radius_m": 0.05, "k_n_n_m": 5.0e4,
                 "bulk_temperature_c": -10.0},
        "measured": {
            "restitution": res.e_measured,
            "peak_force_n": res.peak_force,
            "peak_overlap_m": res.max_overlap,
            "contact_duration_s": res.contact_time_force,
            "dissipated_energy_j": res.dissipated_energy,
            "closed_form_energy_j": closed,
            "heat_ice_j": state.heat_ice_j,
            "peak_delta_t_k_geometric_patch": geo.peak_delta_t_k,
            "peak_delta_t_k_hertz_material_patch": hertz.peak_delta_t_k,
            "melting_risk_geometric_patch": geo.melting_risk,
            "melting_risk_hertz_patch": hertz.melting_risk,
        },
        "patch_geometry_band": {
            "geometric_patch_radius_m": (geo.deposition_volume_m3
                                         / (3.141592653589793 * geo.layer_depth_m)) ** 0.5,
            "hertz_patch_radius_m": (hertz.deposition_volume_m3
                                     / (3.141592653589793 * hertz.layer_depth_m)) ** 0.5,
            "dT_ratio_hertz_over_geometric": hertz.peak_delta_t_k / geo.peak_delta_t_k,
            "note": ("a linear spring contact law at grain scale gives a millimetre overlap; "
                     "the patch radius from that overlap is ~100x the Hertz radius built "
                     "from the material modulus, and dT scales as a^-3. The energy ledger "
                     "is unaffected, so the bridge is verified energetically and NOT "
                     "advertised as a grain-scale temperature predictor."),
        },
    }
    return rows, summary


def main() -> int:
    val = load_validation()
    rows = build_rows(val)
    bridge, bridge_summary = bridge_rows()
    table = {
        "node": "p161w2-arb-n07-dem-ice-thermal-stress",
        "mission_id": 14601,
        "validation_digest_sha256": val.get("digest_sha256"),
        "rows": rows,
        "solver_bridge": {"rows": bridge, "summary": bridge_summary},
        "all_rows_within_1pct": all(
            r["error_pct"] is not None and r["error_pct"] < 1.0
            for r in rows + bridge[:3]),
        "no_reference_rows": [r["quantity"] for r in rows + bridge
                              if r["error_pct"] is None],
    }
    (OUT / "benchmark_table.json").write_text(
        json.dumps(table, indent=2, default=float), encoding="utf-8")

    evidence = {
        "node": "p161w2-arb-n07-dem-ice-thermal-stress",
        "mission_id": 14601,
        "repo": "TensorDEM",
        "claimed_level": 4,
        "impl_files": [
            "src/tensordem/ice_thermal_stress/properties.py",
            "src/tensordem/ice_thermal_stress/plastic_heat.py",
            "src/tensordem/ice_thermal_stress/conduction.py",
            "src/tensordem/ice_thermal_stress/thermal_stress.py",
            "src/tensordem/ice_thermal_stress/fracture.py",
            "src/tensordem/ice_thermal_stress/dem_coupling.py",
            "src/tensordem/ice_thermal_stress/collision.py",
            "src/tensordem/ice_thermal_stress/validation.py",
            "src/tensordem/ice_thermal_stress/__main__.py",
        ],
        "entry": [
            "src/tensordem/ice_thermal_stress/__main__.py",
            "benchmarks/ice_thermal_stress/run_validation.py",
        ],
        "run_artifacts": [
            "docs/ice_thermal_stress/validation.json",
            "docs/ice_thermal_stress/sweeps.json",
            "docs/ice_thermal_stress/benchmark_table.json",
            "docs/ice_thermal_stress/fig1_conduction_vs_exact.png",
            "docs/ice_thermal_stress/fig3_impact_temperature.png",
            "docs/ice_thermal_stress/fig4_thermal_stress_field.png",
            "docs/ice_thermal_stress/fig5_significance_map.png",
            "docs/ice_thermal_stress/fig6_breakage_increment.png",
        ],
        "integration_callers": [
            "tensordem.ice_thermal_stress.dem_coupling",
            "src/tensordem/contact_impact.py:pair_impact(thermal=True)",
        ],
        "benchmark": {
            "table": "docs/ice_thermal_stress/benchmark_table.json",
            "reference_case": {
                "impact_speed_m_s": 1.0, "bulk_temperature_c": -10.0,
                "peak_delta_t_k": val["impact"]["peak_delta_t_k"],
                "sigma_thermal_patch_pa": val["impact"]["sigma_thermal_patch_pa"],
                "thermal_fraction_of_strength": val["impact"]["thermal_fraction_of_strength"],
                "breakage_increment": val["impact"]["breakage_coupled"]["increment"],
            },
            "largest_error_pct_C1_C13": max(r["error_pct"] for r in rows),
            "solver_bridge_energy_error_pct": bridge[0]["error_pct"],
        },
        "repro_cmd": (
            "PYTHONPATH=src python benchmarks/ice_thermal_stress/run_validation.py && "
            "PYTHONPATH=src python scripts/ice_thermal_stress_maturity.py && "
            "PYTHONPATH=src python -m pytest tests/test_ice_thermal_stress_*.py -q"),
        "tests": [
            "tests/test_ice_thermal_stress_properties.py",
            "tests/test_ice_thermal_stress_conduction.py",
            "tests/test_ice_thermal_stress_coupling.py",
            "tests/test_ice_thermal_stress_evidence.py",
            "tests/test_ice_thermal_stress_integration.py",
            "tests/test_contact_impact.py",
        ],
        "withheld": {
            "C11_experimental_arm": (
                "no digitised thermal-shock / impact-calorimetry dataset exists in the "
                "repository and none may be fabricated offline; replaced by the falsifiable "
                "critical temperature jump dT_c = sigma_f (1-nu)/(E a_T)"),
            "grain_scale_temperature_prediction": (
                "the DEM bridge transports energy correctly (work -> heat -> enthalpy, "
                "closure to 1e-14) but the absolute temperature rise depends on the "
                "contact patch radius, which differs by ~100x between the geometric and the "
                "Hertz-material routes for a linear contact law; dT is therefore reported as "
                "a band, not a prediction"),
        },
        "report": REPORT_SECTION,
    }
    (OUT / "MATURITY_EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2, default=float), encoding="utf-8")
    print("benchmark rows:", len(rows), "+ solver bridge rows:", len(bridge))
    print("largest ladder error %:", round(max(r["error_pct"] for r in rows), 4))
    print("bridge energy error %:", round(bridge[0]["error_pct"], 9))
    print("wrote", OUT / "benchmark_table.json")
    print("wrote", OUT / "MATURITY_EVIDENCE.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())