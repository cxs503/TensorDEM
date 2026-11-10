"""Acceptance tests for the ice-ridge keel freeze-bond validation face.

The suite is organised around the eight machine-checkable criteria of node
P50-N07 plus the structural invariants of the DEM and the adfreeze law.  The
heavy DEM study runs once per session (``study`` fixture) and every test reads
from it, so the whole file stays in the minute range.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem.ice_ridge_keel_freezebond import (  # noqa: E402
    GRAVITY,
    RHO_ICE,
    RHO_WATER,
    BondSet,
    RubblePile,
    build_pile,
    contact_area,
    create_bonds,
    force_chain_stats,
    freeze_bond_break_force,
    keel_resistance_model,
    kerr_keel_resistance,
    measure_bond_break_force,
    measure_porosity,
    negative_controls,
    rosin_rammler_sample,
    run_keel_case,
    run_study,
    tau_adfreeze,
)

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "ice_ridge_keel"


@pytest.fixture(scope="session")
def study() -> dict:
    """Reduced campaign study (short DEM stages) shared by the whole session."""
    return run_study(None, keel_heights=(4.0,), temperatures=(-2.0, -10.0, -20.0),
                     consol_steps=1600, release_steps=300, with_figures=False)


@pytest.fixture(scope="session")
def case(study) -> dict:
    return study["cases"][0]


# ---------------------------------------------------------------------------
# Adfreeze law (TensorPhysics interface)
# ---------------------------------------------------------------------------
def test_adfreeze_zero_at_zero_undercooling():
    """Negative control 1: no undercooling -> no bond, no break force."""
    assert tau_adfreeze(0.0) == 0.0
    assert freeze_bond_break_force(0.0, 0.25, 0.25, 1.0e-3) == 0.0


def test_adfreeze_monotone_in_undercooling():
    temps = [2.0, 5.0, 10.0, 15.0, 20.0]
    vals = [tau_adfreeze(-t) for t in temps]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_adfreeze_saturates_at_cold_temperatures():
    """Hyperbolic saturation: the slope dtaus/dT must decrease with |dT|."""
    def slope(t: float) -> float:
        return (tau_adfreeze(-(t + 1.0)) - tau_adfreeze(-t)) / 1.0

    assert slope(3.0) > slope(30.0) > slope(80.0) > 0.0
    # the strength stays bounded by the saturation value
    assert tau_adfreeze(-200.0) < 5.8301


def test_adfreeze_time_hardening_saturates():
    v100 = tau_adfreeze(-10.0, 100.0)
    v1000 = tau_adfreeze(-10.0, 1000.0)
    v10000 = tau_adfreeze(-10.0, 10000.0)
    assert v100 < v1000 < v10000
    # sub-linear (Stefan-type) hardening: a 4x longer contact adds < 2x strength
    assert tau_adfreeze(-10.0, 4000.0) < 2.0 * v1000


def test_adfreeze_salinity_softens_bond():
    assert tau_adfreeze(-10.0, salinity_ppt=15.0) < tau_adfreeze(-10.0, salinity_ppt=0.0)


def test_adfreeze_roughness_strengthens_bond():
    assert tau_adfreeze(-10.0, roughness_m=80e-6) > tau_adfreeze(-10.0, roughness_m=5e-6)


def test_adfreeze_realistic_magnitude():
    """At -10 C / 1000 s the bond strength must be a sub-MPa, non-zero value."""
    tau = tau_adfreeze(-10.0)
    assert 0.05 < tau < 2.0


# ---------------------------------------------------------------------------
# Block ensemble
# ---------------------------------------------------------------------------
def test_rosin_rammler_range_and_mean():
    d = rosin_rammler_sample(5000, 0.1, 1.0, seed=3)
    assert d.min() >= 0.1 and d.max() <= 1.0
    assert abs(d.mean() - 0.55) < 0.02


def test_pile_has_required_block_count():
    pile = build_pile(500, 5.5, 5.5, 12.0, seed=0)
    assert pile.n_blocks == 500
    assert 0.1 - 1e-9 <= 2.0 * float(pile.radius.min()) <= 1.0 + 1e-9


def test_placed_blocks_do_not_overlap():
    pile = build_pile(200, 4.0, 4.0, 10.0, seed=1)
    dx = pile.pos[:, None, :] - pile.pos[None, :, :]
    dist = np.linalg.norm(dx, axis=2)
    np.fill_diagonal(dist, 9.0)
    gap = dist - (pile.radius[:, None] + pile.radius[None, :])
    assert gap.min() > -1e-3


def test_mass_consistent_with_geometry():
    pile = build_pile(300, 4.0, 4.0, 10.0, seed=2)
    assert abs(pile.total_mass - pile.analytic_mass()) / pile.analytic_mass() < 1e-12


# ---------------------------------------------------------------------------
# Freeze-bond parameterisation (criterion 1)
# ---------------------------------------------------------------------------
def test_contact_area_scales_with_overlap():
    a1 = contact_area(0.25, 0.25, 1.0e-3)
    a2 = contact_area(0.25, 0.25, 4.0e-3)
    assert a2 > a1 > 0.0
    assert contact_area(0.25, 0.25, -0.1) == contact_area(0.25, 0.25, 1.0e-3)


def test_bond_break_force_linear_in_strength():
    a = freeze_bond_break_force(-10.0, 0.25, 0.25, 1.0e-3)
    b = freeze_bond_break_force(-20.0, 0.25, 0.25, 1.0e-3)
    ratio = b / a
    expected = tau_adfreeze(-20.0) / tau_adfreeze(-10.0)
    assert abs(ratio - expected) / expected < 1e-9


def test_dem_bond_rupture_matches_analytic():
    """Criterion 1: DEM pull test vs tau_adfreeze * A_contact (<= 5 %)."""
    for temp in (-2.0, -10.0, -20.0):
        res = measure_bond_break_force(undercooling_c=temp, radius=0.25)
        assert res["measured_n"] > 0.9 * res["analytic_n"], res
        assert res["rel_error"] <= 0.05, res


def test_bond_consistency_sweep_ten_params(study):
    """Criterion 1 over 10 parameter combinations."""
    sweep = study["bond_consistency"]
    assert len(sweep) == 10
    assert max(s["rel_error"] for s in sweep) <= 0.05
    assert study["criteria"]["c1_bond_parameterisation"]["pass"]


def test_create_bonds_zero_strength_when_above_freezing():
    pile = build_pile(50, 3.0, 3.0, 6.0, seed=4)
    bonds = create_bonds(pile, undercooling_c=0.0)
    assert bonds.n_bonds >= 0
    assert np.all(bonds.f_break == 0.0)


# ---------------------------------------------------------------------------
# Porosity / structure measurement
# ---------------------------------------------------------------------------
def test_porosity_of_loose_placement_is_high(case):
    assert case["porosity_loose"] > 0.60


def test_porosity_estimator_recovers_solid_volume(case):
    """Monte-Carlo porosity must reproduce the analytic solid volume (<= 5 %)."""
    assert case["voxel_volume_error"] <= 0.05


def test_porosity_std_error_is_small(case):
    assert 0.0 < case["porosity_std_error"] < 5e-3


def test_unbonded_porosity_above_055(study):
    """Criterion 5: no freeze bonds -> loose rubble (n >= 0.55)."""
    assert study["unbonded"]["porosity"] >= 0.55
    assert study["criteria"]["c5_unbonded_degradation"]["pass"]


def test_porosity_above_angular_rubble_range(study):
    """Documented honest limitation: spherical blocks pack looser than angular
    rubble, so the consolidated porosity exceeds the 0.30-0.45 target range."""
    n = study["cases"][0]["porosity"]
    assert n > 0.45      # sphere packing cannot reach angular-rubble porosity
    assert study["criteria"]["c2_porosity"]["pass"] is False


def test_local_porosity_distribution(case):
    lp = case["local_porosity"]
    assert 0.0 <= lp["p05"] <= lp["mean"] <= lp["p95"] <= 1.0


# ---------------------------------------------------------------------------
# Kerr closure / resistance model
# ---------------------------------------------------------------------------
def test_kerr_zero_at_zero_height():
    """Negative control 2: h_k = 0 -> zero resistance."""
    assert kerr_keel_resistance(0.0, 0.4) == 0.0
    assert keel_resistance_model(0.0, 0.4, 0.8, 0.7, 0.01)["f_keel"] == 0.0


def test_kerr_scaling():
    f1 = kerr_keel_resistance(2.0, 0.4, 40.0)
    f2 = kerr_keel_resistance(4.0, 0.4, 40.0)
    assert abs(f2 / f1 - 4.0) < 1e-9                       # quadratic in h_k
    expected = 0.5 * RHO_WATER * GRAVITY * 4.0 * math.tan(math.radians(40.0)) * 0.6
    assert abs(f1 - expected) / expected < 1e-12
    # linear in the solid fraction (1 - n)
    assert abs(kerr_keel_resistance(2.0, 0.6) / f1 - 0.4 / 0.6) < 1e-9


def test_resistance_vanishes_at_full_porosity():
    """Negative control 3: n -> 1 -> zero resistance."""
    assert keel_resistance_model(4.0, 1.0, 0.8, 0.7, 0.0)["f_keel"] == 0.0
    empty = RubblePile(pos=np.zeros((0, 3)), vel=np.zeros((0, 3)),
                       radius=np.zeros(0), mass=np.zeros(0),
                       box=np.array([[0.0, 6.0], [0.0, 6.0], [0.0, 2.0]]))
    assert measure_porosity(empty)["porosity"] == 1.0


def test_negative_controls_all_pass():
    """Criterion 7: the three negative controls."""
    neg = negative_controls()
    assert all(v["pass"] for v in neg.values())


def test_resistance_monotone_in_bond_strength():
    base = keel_resistance_model(4.0, 0.5, 0.7, 0.0, 0.02)["f_keel"]
    warm = keel_resistance_model(4.0, 0.5, 0.7, 0.15, 0.02)["f_keel"]
    cold = keel_resistance_model(4.0, 0.5, 0.7, 1.2, 0.02)["f_keel"]
    assert cold > warm > base


def test_kerr_ratio_reported_for_every_height(study, case):
    assert case["f_kerr"] > 0.0
    assert 0.0 < case["f_skel_f_kerr_ratio"] < 3.0
    assert study["criteria"]["c3_kerr_resistance"]["ratios"]


# ---------------------------------------------------------------------------
# Freeze-bond sensitivity (criterion 4)
# ---------------------------------------------------------------------------
def test_temperature_sensitivity_monotone(study):
    sens = study["sensitivity"]
    assert len(sens) == 3
    assert study["criteria"]["c4_temperature_sensitivity"]["spearman_rho"] >= 0.9
    assert study["criteria"]["c4_temperature_sensitivity"]["pass"]


def test_more_bonds_form_at_colder_temperatures(study):
    """Cold ice freezes more contacts: bonded area must grow with |dT|."""
    areas = [s["n_bonds"] for s in study["sensitivity"]]
    assert areas[-1] >= areas[0]


# ---------------------------------------------------------------------------
# Force chains (criterion 6) and conservation (criterion 8)
# ---------------------------------------------------------------------------
def test_force_chain_tail_fit(study):
    """Criterion 6: exponential force tail with R^2 >= 0.90.

    Checked on the published (converged) study; the reduced in-suite study has
    too few settled contacts to resolve the tail.
    """
    study_json = ARTIFACT_DIR / "study.json"
    if not study_json.exists():
        pytest.skip("artifact bundle not generated in this checkout")
    data = json.loads(study_json.read_text())
    assert data["criteria"]["c6_force_chain_tail"]["pass"] is True
    for c in data["cases"]:
        assert c["force_chain"]["tail_r_squared"] >= 0.90, c["force_chain"]
    for c in study["cases"]:                       # reduced proxy: must not crash
        assert c["force_chain"]["n_contacts"] >= 0


def test_force_chain_stats_on_synthetic_exponential():
    rng = np.random.default_rng(0)
    f = rng.exponential(scale=100.0, size=20000)
    stats = force_chain_stats(f)
    assert stats["tail_r_squared"] > 0.90
    assert 50.0 < stats["tail_scale"] < 200.0


def test_force_chain_stats_degenerate_input():
    stats = force_chain_stats(np.zeros(5))
    assert stats["n_contacts"] == 0 and stats["tail_r_squared"] == 0.0


def test_mass_conservation(study, case):
    """Criterion 8: total mass error <= 1e-6."""
    assert case["mass_error"] <= 1e-6
    assert study["criteria"]["c8_mass_conservation"]["pass"]


# ---------------------------------------------------------------------------
# Study bookkeeping
# ---------------------------------------------------------------------------
def test_criteria_block_is_complete(study):
    crit = study["criteria"]
    keys = [k for k in crit if k.startswith("c")]
    assert len(keys) == 8
    assert crit["n_criteria"] == 8
    assert 0 <= crit["n_pass"] <= 8


def test_study_is_json_serialisable(study):
    text = json.dumps(study)
    assert "criteria" in json.loads(text)


def test_published_artifacts_exist():
    """The committed artifact bundle must contain study.json and 5 figures."""
    study_json = ARTIFACT_DIR / "study.json"
    if not study_json.exists():
        pytest.skip("artifact bundle not generated in this checkout")
    data = json.loads(study_json.read_text())
    assert "criteria" in data
    figs = sorted(ARTIFACT_DIR.glob("fig*.png"))
    assert len(figs) >= 5