"""Command-line entry point: ``python -m tensordem.ice_thermal_stress``.

The package had one way in -- import the functions in a script.  This module
gives it a *programmable* entry that the platform can call without knowing the
API shape:

* ``--case reference``  the reported v = 1 m/s, -10 degC ice block impact,
* ``--case dem``        the same physics driven by a real ``IceDEM3D`` pair
                        impact (the solver-integration path),
* ``--v0``, ``--temp``  any other point of the (v, T) map.

Every run prints a small table and, with ``--json PATH``, writes the same
numbers as JSON.  Nothing is hard-coded: the numbers come from the same
functions the tests and the benchmark use.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

__all__ = ["run_reference_case", "run_dem_case", "main"]


def run_reference_case(*, v0: float = 1.0, temperature_c: float = -10.0,
                       cells: int = 240) -> dict[str, Any]:
    """Analytic Hunt-Crossley driver (the benchmark front end)."""
    from .collision import BlockImpactCase, simulate_block_impact

    case = BlockImpactCase(impact_speed_m_s=v0, bulk_temperature_c=temperature_c,
                           cells=cells)
    result = simulate_block_impact(case)
    flat = result.flat()
    return {
        "entry": "reference",
        "impact_speed_m_s": v0,
        "bulk_temperature_c": temperature_c,
        "numbers": {
            "peak_force_n": flat["peak_force_n"],
            "contact_radius_m": flat["contact_radius_m"],
            "contact_duration_s": flat["contact_duration_s"],
            "dissipated_energy_j": flat["dissipated_energy_j"],
            "plastic_heat_ice_j": flat["plastic_heat_ice_j"],
            "peak_delta_t_k": flat["peak_delta_t_k"],
            "peak_delta_t_adiabatic_k": flat["peak_delta_t_adiabatic_k"],
            "sigma_thermal_patch_pa": flat["sigma_thermal_patch_pa"],
            "sigma_thermal_free_pa": flat["sigma_thermal_free_pa"],
            "sigma_mech_peak_pa": flat["sigma_mech_peak_pa"],
            "thermal_fraction_of_strength": flat["thermal_fraction_of_strength"],
            "breakage_increment": flat["breakage_increment"],
            "breakage_increment_bound": flat["breakage_increment_bound"],
            "melting_risk": flat["melting_risk"],
            "heat_to_enthalpy_relative_error": flat["heat_to_enthalpy_relative_error"],
            "dissipated_energy_relative_error": flat["dissipated_energy_relative_error"],
        },
    }


def run_dem_case(*, v0: float = 0.5, temperature_c: float = -10.0,
                 e_target: float = 0.5, radius_m: float = 0.05,
                 thermal_cells: int = 120, thermal_dt_s: float = 2.0e-5) -> dict[str, Any]:
    """Real ``IceDEM3D`` pair impact -> thermal chain (the integration path)."""
    from ..contact_impact import pair_impact

    res = pair_impact(e_target=e_target, v0=v0, k_n=5.0e4, radius=radius_m,
                      density=900.0, thermal=True, bulk_temperature_c=temperature_c,
                      thermal_cells=thermal_cells, thermal_dt_s=thermal_dt_s)
    state = res.thermal_state
    flat = state.flat()
    work_closed = res.pair_mass_kg * v0 ** 2 * (1.0 - res.e_measured ** 2)
    return {
        "entry": "dem3d_pair_impact",
        "impact_speed_m_s": v0,
        "bulk_temperature_c": temperature_c,
        "numbers": {
            "restitution_measured": res.e_measured,
            "peak_force_n": res.peak_force,
            "contact_duration_s": res.contact_time_force,
            "dissipated_energy_j": res.dissipated_energy,
            "dissipated_energy_closed_form_j": work_closed,
            "dissipated_energy_relative_error": (
                abs(res.dissipated_energy - work_closed) / work_closed
                if work_closed > 0.0 else 0.0),
            "plastic_heat_ice_j": flat["heat_ice_j"],
            "peak_delta_t_k": flat["peak_delta_t_k"],
            "sigma_thermal_patch_pa": flat["sigma_thermal_patch_pa"],
            "thermal_fraction_of_strength": flat["thermal_fraction_of_strength"],
            "breakage_increment": flat["breakage_increment"],
            "melting_risk": flat["melting_risk"],
            "heat_to_enthalpy_relative_error": flat["heat_to_enthalpy_relative_error"],
        },
    }


def _print_table(payload: dict[str, Any]) -> None:
    print(f"entry: {payload['entry']}   v0 = {payload['impact_speed_m_s']} m/s"
          f"   T = {payload['bulk_temperature_c']} degC")
    width = max(len(k) for k in payload["numbers"])
    for key, value in payload["numbers"].items():
        if isinstance(value, bool):
            shown = str(value)
        elif isinstance(value, float):
            shown = f"{value:.6g}"
        else:
            shown = str(value)
        print(f"  {key:<{width}}  {shown}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m tensordem.ice_thermal_stress",
        description="Ice-block collision thermal-mechanical coupling (TensorDEM)")
    ap.add_argument("--case", choices=("reference", "dem"), default="reference")
    ap.add_argument("--v0", type=float, default=None, help="impact speed [m/s]")
    ap.add_argument("--temp", type=float, default=-10.0, help="bulk ice temperature [degC]")
    ap.add_argument("--cells", type=int, default=240, help="conduction cells")
    ap.add_argument("--restitution", type=float, default=0.5, help="DEM target restitution")
    ap.add_argument("--radius", type=float, default=0.05, help="DEM particle radius [m]")
    ap.add_argument("--json", type=Path, default=None, help="write the numbers as JSON")
    args = ap.parse_args(argv)

    if args.case == "reference":
        payload = run_reference_case(
            v0=1.0 if args.v0 is None else args.v0,
            temperature_c=args.temp, cells=args.cells)
    else:
        payload = run_dem_case(
            v0=0.5 if args.v0 is None else args.v0,
            temperature_c=args.temp, e_target=args.restitution, radius_m=args.radius)

    _print_table(payload)

    bad = [k for k, v in payload["numbers"].items()
           if isinstance(v, float) and not math.isfinite(v)]
    if bad:
        print("non-finite numbers:", bad, file=sys.stderr)
        return 2
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True),
                             encoding="utf-8")
        print("wrote", args.json)
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())