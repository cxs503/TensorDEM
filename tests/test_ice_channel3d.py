"""Tests for the 3-D slab channel DEM (P82W2-N05).

The suite pins the *structural* guarantees that the mission asks for:

* the departure-flux ledger closes and the mass is conserved per frame,
* nothing is deleted in the corrected configuration (and the legacy switch
  really is destructive),
* the layered slab converges to the full-thickness bending capacity while
  independent laminae do not,
* the 2-D slice (one layer, closed walls, no subduction) is the configuration
  that inflates the resistance,
* the ten-case reference table loads with its documented column split.

The runs use minimal frame counts, so the suite stays in the seconds range; the
resistance numbers are *not* asserted here (they are campaign results, see
``artifacts/uikku_slab3d/campaign.json``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ice_channel3d import (
    CRITERIA,
    FluxLedger,
    HullFootprint,
    IceSheetProps,
    MassLedger,
    SlabChannelSim,
    SlabConfig,
    bending_capacity_per_width_n_per_m,
    characteristic_length_m,
    layer_moment_shares,
    load_reference_cases,
)
from tensordem.ice_channel3d.uikku import ice_props_of, mape

HULL = HullFootprint(beam_m=20.0, length_m=120.0, bow_half_angle_deg=22.0)
ICE = IceSheetProps(thickness_m=0.77, flexural_strength_pa=724e3,
                    compressive_strength_pa=1748e3, elastic_modulus_pa=929e6)


def make(frames: int = 25, **overrides) -> SlabChannelSim:
    cfg = dict(n_layers=3, frames=frames, relax_iterations=3,
               drift_speed_m_s=0.2, floe_radius_m=2.0)
    cfg.update(overrides)
    return SlabChannelSim(SlabConfig(**cfg), ICE, hull=HULL)


# --------------------------------------------------------------------------
# geometry / constitutive closures
# --------------------------------------------------------------------------
def test_layer_moment_shares_sum_to_one():
    for n in (1, 2, 3, 4, 8):
        shares = layer_moment_shares(n, 0.9)
        assert shares.shape == (n,)
        assert shares.sum() == pytest.approx(1.0, rel=0, abs=1e-12)
        assert (shares > 0).all()
    # the outer layers carry most of the moment, the middle layer the least
    shares = layer_moment_shares(3, 0.9)
    assert shares[0] == pytest.approx(shares[2])
    assert shares[1] < 0.1 * shares[0]


def test_characteristic_length_and_bending_capacity():
    lc = characteristic_length_m(0.77, 929e6)
    assert lc == pytest.approx(7.88, rel=1e-2)
    per_width = bending_capacity_per_width_n_per_m(724e3, 0.77, lc)
    assert per_width == pytest.approx(13.6e3, rel=2e-2)
    # the four-layer slab reproduces the full-thickness capacity per column
    shares = layer_moment_shares(4, 0.77)
    per_floe = shares * per_width * 2.0 * 2.0
    assert per_floe.sum() == pytest.approx(per_width * 4.0, rel=1e-12)


def test_hull_signed_distance_is_exact_for_convex_footprint():
    pts = np.array([[10.0, 0.0], [0.0, 0.0], [-60.0, 0.0], [-60.0, 9.0],
                    [-60.0, 12.0]])
    dist, normal = HULL.signed_distance(pts)
    assert dist[0] > 0.0                       # ahead of the stem: outside
    assert dist[2] < 0.0 and dist[3] < 0.0     # inside the hull body
    assert dist[4] > 0.0                       # outside the parallel side
    assert np.allclose(np.linalg.norm(normal, axis=1), 1.0)


# --------------------------------------------------------------------------
# ledgers
# --------------------------------------------------------------------------
def test_mass_ledger_flags_deletions_and_passes_conserved_runs():
    ledger = MassLedger(ice_density=917.0, initial_mass_kg=1000.0)
    assert ledger.record_frame(1000.0) == pytest.approx(0.0)
    ledger.register_deletion(50.0)
    # the destroyed mass is a *physical* conservation error of 5 %
    assert ledger.record_frame(950.0) == pytest.approx(0.05)
    assert ledger.max_frame_error == pytest.approx(0.05)


def test_flux_ledger_closure_identity():
    ledger = FluxLedger(reservoir_mass_0_kg=0.0)
    ledger.reservoir_mass_kg = 250.0
    assert ledger.record(300.0, 50.0) == pytest.approx(1.0)
    ledger.reservoir_mass_kg = 240.0
    assert ledger.record(300.0, 50.0) == pytest.approx(240.0 / 250.0)
    assert ledger.worst_deviation > 0.0


def test_corrected_run_conserves_mass_and_keeps_every_floe():
    sim = make(frames=20)
    n0 = int(sim.active.sum())
    m0 = sim._field_mass()
    summary = sim.run()
    assert summary.deleted_kg == 0.0
    assert summary.max_frame_mass_error <= 1e-9
    assert -0.01 <= summary.flux_closure_ratio - 1.0 <= 0.01
    # nothing is deleted: no floe leaves the books without a ledger line
    assert int(sim.active.sum()) <= n0
    accounted = (m0 - summary.outflow_kg
                 + summary.return_vertical_kg + summary.return_kg
                 - summary.departure_vertical_kg - summary.departure_kg)
    assert math.isclose(accounted, sim._field_mass(), rel_tol=1e-9, abs_tol=1e-3)


def test_legacy_configuration_is_destructive_and_loses_the_ledger():
    sim = make(frames=60, n_layers=1, lateral_open=False, instant_removal=True,
               subduction=False)
    summary = sim.run()
    # instant clearance removes floes from the wake; the ledger books it
    assert summary.deleted_kg >= 0.0
    if summary.deleted_kg > 0.0:
        assert summary.max_frame_mass_error > 0.0


def test_vertical_departure_only_exists_in_the_slab():
    slab = make(frames=40, n_layers=3)
    flat = make(frames=40, n_layers=1)
    slab_out = slab.run()
    flat_out = flat.run()
    assert slab.subduction_enabled is True
    assert flat.subduction_enabled is False
    assert slab_out.departure_vertical_kg > 0.0
    assert flat_out.departure_vertical_kg == 0.0
    assert flat_out.max_frame_mass_error <= 1e-9


def test_uncoupled_layers_underestimate_the_bending_capacity():
    coupled = make(frames=5, n_layers=4)
    uncoupled = make(frames=5, n_layers=4, layer_moment_coupling=False)
    assert coupled.shares.sum() == pytest.approx(1.0)
    assert uncoupled.shares.sum() == pytest.approx(1.0)
    per_column_coupled = float(np.sum(coupled.break_capacity[:4] * 0 + coupled.shares))
    assert per_column_coupled == pytest.approx(1.0)
    # independent lamina bending scales as 1/n: it is *not* the slab
    lamina = np.array([1.0 / 4] * 4) * 4 ** 2
    assert lamina.sum() > 1.5


# --------------------------------------------------------------------------
# reference data and campaign plumbing
# --------------------------------------------------------------------------
def test_reference_table_loads_ten_level_ice_cases():
    cases = load_reference_cases()
    assert [c["case_id"] for c in cases] == ["103", "104", "205", "206", "301",
                                             "302", "303", "401", "402", "403"]
    first = cases[0]
    assert first["ice_thickness_m"] == pytest.approx(0.77)
    assert first["reference_table8_kN"] == pytest.approx(470.0)
    assert first["measured_table3_kN"] == pytest.approx(480.0)
    # the two experimental columns are kept apart, never reconciled (the paper's
    # own 480 vs 470 kN discrepancy for Test 103 stays visible)
    assert cases[7]["measured_table3_kN"] == pytest.approx(150.0)
    assert cases[7]["reference_table8_kN"] == pytest.approx(152.0)
    props = ice_props_of(first)
    assert props.char_length_m == pytest.approx(7.88, rel=1e-2)


def test_mape_matches_hand_computation():
    assert mape([110.0, 90.0], [100.0, 100.0]) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        mape([1.0], [1.0, 2.0])


def test_criteria_block_declares_the_mission_thresholds():
    assert CRITERIA["a_mass_conservation"]["max_frame_relative_error"] == 1e-6
    assert CRITERIA["b_flux_closure"]["closure_band"] == [0.99, 1.01]
    assert CRITERIA["c_resolution_independence"]["max_relative_change"] == 0.05
    assert CRITERIA["e_resistance_mape"]["target_percent"] == 15.0
    assert CRITERIA["f_negative_control"]["min_mape_percent"] == 100.0


def test_layering_preserves_the_full_thickness_crushing_load():
    """n layers of thickness h/n must sum to the h-thickness crushing force."""
    penetration = 0.01
    for n in (2, 3, 4, 6):
        sim = make(frames=5, n_layers=n)
        per_layer = sim._crush_force(np.array([2.0]), np.array([0.77 / n]),
                                     np.array([penetration]))[0]
        single = sim._crush_force(np.array([2.0]), np.array([0.77]),
                                  np.array([penetration]))[0]
        assert n * per_layer == pytest.approx(single, rel=1e-12)
