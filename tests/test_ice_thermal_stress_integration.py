"""Integration tests: the thermal chain driven by the real DEM solver.

Three things are pinned here, and they are different in kind:

1. **One physics path.**  The analytic driver
   (:func:`tensordem.ice_thermal_stress.collision.simulate_block_impact`) must
   delegate to :func:`tensordem.ice_thermal_stress.dem_coupling.thermal_response`,
   so the solver bridge and the benchmark cannot drift apart.
2. **Energy bookkeeping.**  Work delivered by ``IceDEM3D`` -> heat -> stored
   enthalpy closes to machine precision, and the solver's own dissipated work
   agrees with ``m v0^2 (1 - e^2)``.
3. **The entry point and the patch-geometry sensitivity.**  ``python -m
   tensordem.ice_thermal_stress`` runs and writes JSON; the two admissible ways
   of forming the contact patch differ by orders of magnitude at DEM grain
   scale, which is reported rather than hidden.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tensordem.contact_impact import pair_impact
from tensordem.ice_thermal_stress import dem_coupling as dc
from tensordem.ice_thermal_stress.collision import BlockImpactCase, simulate_block_impact
from tensordem.ice_thermal_stress.properties import ice_properties

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. one physics path
# ---------------------------------------------------------------------------
def test_collision_driver_delegates_to_the_shared_response():
    case = BlockImpactCase(impact_speed_m_s=1.0, bulk_temperature_c=-10.0, cells=80)
    res = simulate_block_impact(case)
    e_eff = res.effective_modulus_pa
    state = dc.thermal_response(
        dissipated_work_j=res.dissipated_energy_j,
        peak_force_n=res.peak_force_n,
        effective_radius_m=case.contact_radius_m,
        effective_modulus_pa=e_eff,
        contact_duration_s=res.contact_duration_s,
        bulk_temperature_c=case.bulk_temperature_c,
        beta=case.beta,
        layer_fraction=case.layer_fraction,
        cells=case.cells,
        grid_depth_m=case.grid_depth_m,
        dt_s=case.dt_s,
        duration_factor=case.duration_factor,
    )
    assert state.peak_delta_t_k == pytest.approx(res.peak_delta_t_k, rel=1e-12)
    assert state.sigma_thermal_patch_pa == pytest.approx(res.sigma_thermal_patch_pa, rel=1e-12)
    assert state.breakage_coupled["increment"] == pytest.approx(
        res.breakage_coupled["increment"], rel=1e-12)


# ---------------------------------------------------------------------------
# 2. energy bookkeeping
# ---------------------------------------------------------------------------
def test_heat_and_enthalpy_close_in_the_shared_response():
    state = dc.thermal_response(
        dissipated_work_j=47.25, peak_force_n=174839.0, effective_radius_m=0.30,
        effective_modulus_pa=1.10e10, contact_duration_s=2.9255e-3,
        bulk_temperature_c=-10.0)
    props = ice_properties(-10.0)
    expected_heat = dc.partition_to_ice(props, None) * dc.TAYLOR_QUINNEY_BETA * 47.25
    assert state.heat_ice_j == pytest.approx(expected_heat, rel=1e-12)
    assert state.energy_ledger["heat_to_enthalpy_relative_error"] < 1e-9
    assert state.breakage_coupled["increment"] <= state.breakage_bound["increment"] + 1e-12
    assert 0.0 < state.peak_delta_t_k < state.peak_delta_t_adiabatic_k * 1.05


def test_dem_pair_impact_drives_the_thermal_chain():
    res = pair_impact(e_target=0.5, v0=0.5, radius=0.05, k_n=5.0e4, density=900.0,
                      thermal=True, bulk_temperature_c=-10.0,
                      thermal_cells=60, thermal_dt_s=2.0e-5)
    state = res.thermal_state
    assert state is not None
    # solver ledger vs the closed form for two equal masses
    closed = res.pair_mass_kg * 0.5 ** 2 * (1.0 - res.e_measured ** 2)
    assert res.dissipated_energy == pytest.approx(closed, rel=1e-9)
    assert res.dissipated_energy > 0.0
    assert state.energy_ledger["dissipated_work_j"] == pytest.approx(
        res.dissipated_energy, rel=1e-12)
    assert state.heat_ice_j == pytest.approx(
        dc.TAYLOR_QUINNEY_BETA * res.dissipated_energy, rel=1e-12)
    assert state.energy_ledger["heat_to_enthalpy_relative_error"] < 1e-9
    assert state.peak_delta_t_k > 0.0
    assert math.isfinite(state.sigma_thermal_patch_pa)


def test_coupler_rejects_negative_work():
    coupler = dc.DEMThermalCoupler(mass_kg=1.0)
    with pytest.raises(ValueError):
        coupler.add_dissipation(-1.0)
    with pytest.raises(ValueError):
        dc.DEMThermalCoupler(mass_kg=0.0)


# ---------------------------------------------------------------------------
# 3. the entry point and the geometry sensitivity
# ---------------------------------------------------------------------------
def test_cli_entry_writes_json(tmp_path: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    out = tmp_path / "entry.json"
    proc = subprocess.run(
        [sys.executable, "-m", "tensordem.ice_thermal_stress",
         "--case", "reference", "--cells", "60", "--json", str(out)],
        capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(out.read_text())
    numbers = payload["numbers"]
    assert payload["entry"] == "reference"
    assert numbers["peak_delta_t_k"] > 5.0
    assert numbers["sigma_thermal_patch_pa"] < 0.0        # compression
    assert numbers["heat_to_enthalpy_relative_error"] < 1e-9
    assert "peak_delta_t_k" in proc.stdout


def test_patch_geometry_is_the_dominant_uncertainty_of_the_bridge():
    """Reported, not hidden: the two admissible patch radii differ hugely.

    At DEM grain scale a soft linear contact law produces a millimetre-scale
    overlap, so the geometric radius ``sqrt(delta (2R* - delta))`` is ~100x the
    Hertz radius built from the *material* modulus.  The squared radius enters
    the deposition volume, so the predicted temperature rise differs by four
    orders of magnitude.  That is exactly the reason the bridge is verified
    energetically (work -> heat -> enthalpy) and *not* advertised as a
    temperature predictor at grain scale.
    """
    res = pair_impact(e_target=0.5, v0=0.5, radius=0.05, k_n=5.0e4, density=900.0,
                      thermal=True, thermal_cells=40, thermal_dt_s=5.0e-5)
    props = ice_properties(-10.0)
    e_star = props.youngs_modulus_pa / (2.0 * (1.0 - props.poisson ** 2))
    geometric = dc.couple_impact_result(
        res, bulk_temperature_c=-10.0, effective_radius_m=0.025,
        effective_modulus_pa=e_star, peak_overlap_m=res.max_overlap, cells=40,
        dt_s=5.0e-5)
    hertz = dc.couple_impact_result(
        res, bulk_temperature_c=-10.0, effective_radius_m=0.025,
        effective_modulus_pa=e_star, cells=40, dt_s=5.0e-5)
    assert geometric.heat_ice_j == pytest.approx(hertz.heat_ice_j, rel=1e-9)
    ratio = hertz.peak_delta_t_k / geometric.peak_delta_t_k
    assert ratio > 100.0
    # both agree on the *energy*, which is the part the solver actually knows
    assert geometric.energy_ledger["block_enthalpy_j"] == pytest.approx(
        geometric.heat_ice_j, rel=1e-9)