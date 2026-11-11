"""Acceptance tests for the comminution energy surface (TensorDEM).

Coverage map (the campaign node asks for >= 12 tests, all green):

critical speed
    ``test_critical_speed_constant_from_first_principles``
    ``test_critical_speed_scaling_and_charge_centre_correction``
    ``test_fraction_of_critical_round_trip``
    ``test_dem_detects_the_critical_speed``
mill power
    ``test_hogg_fuerstenau_units_and_fill_optimum``
    ``test_torque_factor_and_power_density``
    ``test_dem_mill_power_matches_correlation``
    ``test_dem_mill_ledgers_close``
Bond
    ``test_bond_third_theory_identities``
    ``test_bond_grindability_closure``
    ``test_work_index_database_bands``
breakage kinetics
    ``test_appearance_function_mass_conservation``
    ``test_appearance_from_t10_round_trip``
    ``test_austin_selection_rate_shape``
    ``test_t10_energy_law_limits``
    ``test_sectional_pbm_matches_matrix_exponential``
    ``test_sectional_pbm_conserves_mass``
    ``test_collision_spectrum_selection_rates``
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from tensordem.breakage_pbm.dpbm import VolumeGrid
from tensordem.comminution.bond import (
    BOND_WORK_INDEX_DB,
    SHORT_TON_TO_TONNE,
    bond_product_size_um,
    bond_specific_energy_kwh_t,
    bond_work_index_from_energy,
    bond_work_index_from_grindability,
    specific_energy_kwh_t,
    work_index_band,
)
from tensordem.comminution.breakage_kinetics import (
    AppearanceFunction,
    austin_selection_rates,
    batch_grind,
    cumulative_passing,
    mass_fraction_matrix,
    p80_from_psd,
    passing_at,
    selection_rates_from_collisions,
    section_edges,
    t10_energy_law,
    t10_from_psd,
    volume_daughter_matrix,
)
from tensordem.comminution.critical_speed import (
    CRITICAL_SPEED_CONSTANT,
    critical_speed_constant_from_first_principles,
    critical_speed_rpm,
    critical_speed_rpm_charge_centre,
    fraction_of_critical,
    froude_from_speed,
    rpm_for_fraction_of_critical,
)
from tensordem.comminution.mill_dem import (
    MillConfig,
    make_mill_state,
    measure_centrifuging_onset_rpm,
    run_mill,
)
from tensordem.comminution.mill_power import (
    charge_bulk_density,
    hogg_fuerstenau_power_kw,
    mill_power_density_kw_m3,
    torque_factor,
)


# ------------------------------------------------------------------ anchor 1
def test_critical_speed_constant_from_first_principles():
    """The textbook 42.3 must be g and pi, not a memorised digit."""
    derived = critical_speed_constant_from_first_principles()
    assert derived == pytest.approx(60.0 / (2.0 * math.pi) * math.sqrt(2.0 * 9.80665), rel=1e-12)
    assert abs(CRITICAL_SPEED_CONSTANT - derived) / derived < 1e-3
    # and the formula itself
    for d in (0.15, 0.30, 0.55, 1.0, 5.5):
        assert critical_speed_rpm(d) == pytest.approx(42.3 / math.sqrt(d), rel=1e-12)


def test_critical_speed_scaling_and_charge_centre_correction():
    """N_c ~ D^-1/2, and the ball-centre variant is the exact single-ball value."""
    d1, d2 = 0.25, 1.0
    ratio = critical_speed_rpm(d1) / critical_speed_rpm(d2)
    assert ratio == pytest.approx(math.sqrt(d2 / d1), rel=1e-12)
    ball = 0.025
    exact = critical_speed_rpm_charge_centre(d2, ball)
    assert exact > critical_speed_rpm(d2)
    # rel=1e-3: the textbook 42.3 is rounded while the charge-centre variant is
    # built from g and pi (see test_critical_speed_constant_from_first_principles)
    assert exact / critical_speed_rpm(d2) == pytest.approx(
        math.sqrt(d2 / (d2 - ball)), rel=1e-3)
    with pytest.raises(ValueError):
        critical_speed_rpm(0.0)
    with pytest.raises(ValueError):
        critical_speed_rpm_charge_centre(0.3, 0.3)


def test_fraction_of_critical_round_trip():
    d = 0.30
    nc = critical_speed_rpm(d)
    assert fraction_of_critical(nc, d) == pytest.approx(1.0, rel=1e-12)
    assert rpm_for_fraction_of_critical(0.7, d) == pytest.approx(0.7 * nc, rel=1e-12)
    # Froude number at the critical speed is exactly 1
    assert froude_from_speed(nc, d) == pytest.approx(1.0, rel=1e-3)


def test_dem_detects_the_critical_speed():
    """DEM tracer onset must land within 2% of 42.3/sqrt(D)."""
    res = measure_centrifuging_onset_rpm(0.20, 0.002, tolerance=2e-3)
    assert abs(res["error_vs_nominal"]) <= 0.02
    assert res["rpm"] > 0.0
    # the ball-centre reference is the tighter physical comparison
    assert abs(res["error_vs_charge_centre"]) <= 0.02
    # detected onset lies between the two references (apex contact happens at a
    # radius between D/2 and (D-d)/2)
    assert (res["critical_speed_rpm_nominal"] <= res["rpm"]
            <= res["critical_speed_rpm_charge_centre"] * 1.001)


# ------------------------------------------------------------------ anchor 2
def test_hogg_fuerstenau_units_and_fill_optimum():
    """Correlation: linear in L, D**2.3, zero at J=0, optimum at J=1/(2*0.937)."""
    base = dict(diameter=0.30, length=0.15, fill=0.35, fraction_critical=0.70,
                bulk_density=4680.0)
    p0 = hogg_fuerstenau_power_kw(**base)
    p1 = hogg_fuerstenau_power_kw(**{**base, "length": 0.30})
    assert p1 == pytest.approx(2.0 * p0, rel=1e-12)
    p2 = hogg_fuerstenau_power_kw(**{**base, "diameter": 0.60})
    assert p2 / p0 == pytest.approx(2.0 ** 2.3, rel=1e-12)
    # fill dependence: zero at J -> 0 and maximum at J = 1/(2*0.937)
    assert hogg_fuerstenau_power_kw(**{**base, "fill": 1e-9}) == pytest.approx(0.0, abs=1e-8)
    j_opt = 1.0 / (2.0 * 0.937)
    grid = np.linspace(0.05, 0.9, 400)
    vals = [hogg_fuerstenau_power_kw(**{**base, "fill": float(j)}) for j in grid]
    assert grid[int(np.argmax(vals))] == pytest.approx(j_opt, abs=5e-3)
    with pytest.raises(ValueError):
        hogg_fuerstenau_power_kw(**{**base, "fill": 1.5})


def test_torque_factor_and_power_density():
    tf = torque_factor(11.4, 28.9, 0.15)
    assert tf == pytest.approx(11.4 / (28.9 * 9.80665 * 0.15), rel=1e-12)
    vol = math.pi * 0.25 * 0.30 ** 2 * 0.15
    assert mill_power_density_kw_m3(0.0646, 0.30, 0.15) == pytest.approx(0.0646 / vol, rel=1e-12)
    rho = charge_bulk_density(7800.0, 0.40)
    assert rho == pytest.approx(4680.0, rel=1e-12)


@pytest.fixture(scope="module")
def small_mill_run():
    cfg = MillConfig(diameter=0.15, length=0.08, lifter_count=6,
                     lifter_height=0.003, lifter_width=0.003,
                     ball_radii=(0.008, 0.005), ball_counts=(8, 12), dt=1.0e-4)
    rpm = rpm_for_fraction_of_critical(0.70, cfg.diameter)
    state = make_mill_state(cfg, fill=0.35 * (1.0 - 0.40), rpm=rpm, seed=3)
    out = run_mill(state, revolutions=1.6, measure_last=0.8, settle_time=0.25,
                   sample_every=200)
    return cfg, rpm, out


def test_dem_mill_ledgers_close(small_mill_run):
    """Energy ledger and angular-momentum ledger must close on a real run."""
    cfg, rpm, out = small_mill_run
    led = out["ledger"]
    assert led["p_in_w"] > 0.0
    assert np.isfinite(led["closure_rel"]) and led["closure_rel"] < 0.25
    assert np.isfinite(led["full_closure_rel"]) and led["full_closure_rel"] < 0.35
    assert led["moment_closure_rel"] < 0.10
    # sign conventions: a CCW shell does positive work on a CCW charge
    assert led["tau_on_charge_nm"] > 0.0
    assert led["x_com_m"] > 0.0
    assert out["max_overlap_rel"] < 0.15


def test_dem_mill_power_matches_correlation(small_mill_run):
    """DEM power within 50% of the empirical correlation (small-mill extrapolation).

    The band is loose on purpose: Hogg & Fuerstenau is a fit to industrial-scale
    mills and this fixture is a 150 mm laboratory mill, so the test protects
    against unit/scale blunders, not against the correlation's own uncertainty.
    """
    cfg, rpm, out = small_mill_run
    fill = out["fill_realised"] / (1.0 - 0.40)
    rho = charge_bulk_density(cfg.density, 0.40)
    p_hf = hogg_fuerstenau_power_kw(cfg.diameter, cfg.length, fill,
                                    fraction_of_critical(rpm, cfg.diameter), rho) * 1000.0
    assert 0.5 < out["ledger"]["p_in_w"] / p_hf < 1.5


# ------------------------------------------------------------------ anchor 3
def test_bond_third_theory_identities():
    wi, f80, p80 = 12.0, 1000.0, 150.0
    e = bond_specific_energy_kwh_t(wi, f80, p80)
    assert e == pytest.approx(10.0 * wi * (1.0 / math.sqrt(p80) - 1.0 / math.sqrt(f80)),
                             rel=1e-12)
    # infinite feed: E = 10 Wi / sqrt(P80)
    assert bond_specific_energy_kwh_t(wi, math.inf, p80) == pytest.approx(
        10.0 * wi / math.sqrt(p80), rel=1e-12)
    # work index = energy for reducing an infinite feed to 100 um
    assert bond_specific_energy_kwh_t(wi, math.inf, 100.0) == pytest.approx(wi, rel=1e-12)
    # inverse round trip
    assert bond_product_size_um(wi, f80, e) == pytest.approx(p80, rel=1e-9)
    assert bond_work_index_from_energy(f80, p80, e) == pytest.approx(wi, rel=1e-9)
    # short-ton conversion is a 10% trap and must be explicit
    assert SHORT_TON_TO_TONNE == pytest.approx(0.90718474, rel=1e-6)
    with pytest.raises(ValueError):
        bond_specific_energy_kwh_t(wi, 100.0, 200.0)


def test_bond_grindability_closure():
    """The grindability equation and the third theory must agree on the same Wi."""
    f80, p80, g_rev = 1000.0, 150.0, 3.0
    pc = p80
    wi = bond_work_index_from_grindability(g_rev, pc, f80, p80)
    e = bond_specific_energy_kwh_t(1.0, f80, p80)      # 10(1/sqrt(P80)-1/sqrt(F80))
    assert 44.5 / ((pc ** 0.23) * (g_rev ** 0.82) * e) == pytest.approx(wi, rel=1e-12)
    # halving the throughput (G) must raise the work index by 2**0.82
    wi2 = bond_work_index_from_grindability(g_rev / 2.0, pc, f80, p80)
    assert wi2 / wi == pytest.approx(2.0 ** 0.82, rel=1e-12)


def test_work_index_database_bands():
    for name, row in BOND_WORK_INDEX_DB.items():
        assert row["low"] <= row["typical"] <= row["high"], name
        assert row["source"]
    band = work_index_band("Limestone")
    assert band["material"] == "limestone"
    assert band["typical"] == pytest.approx(12.0)
    with pytest.raises(KeyError):
        work_index_band("unobtainium")


# ------------------------------------------------------------------ anchor 4
def test_appearance_function_mass_conservation():
    d = VolumeGrid.from_diameters(50e-6, 2000e-6, 18).diameters
    af = AppearanceFunction.from_t10(25.0)
    b = mass_fraction_matrix(d, af)
    assert np.allclose(b.sum(axis=0), 1.0, atol=1e-12)
    # row = daughter section (ascending size), column = parent, so the *lower*
    # triangle is the coarse-fragment block and must be empty
    assert np.allclose(np.tril(b, k=-1), 0.0)          # no fragment coarser than its parent
    edges = section_edges(d)
    assert np.all(np.diff(edges) > 0.0)
    assert float(af.cumulative_finer(1.0, 1.0)) == pytest.approx(1.0, rel=1e-12)
    # the family has no lower cut-off: B(x->0) -> 0 monotonically, but the
    # power-law tail is only reached asymptotically (gamma ~ 0.15 here)
    assert float(af.cumulative_finer(1e-12, 1.0)) < float(af.cumulative_finer(1e-3, 1.0))


def test_appearance_from_t10_round_trip():
    for t10 in (5.0, 12.5, 25.0, 40.0, 60.0):
        af = AppearanceFunction.from_t10(t10)
        assert af.t10 == pytest.approx(t10, rel=1e-9)
        assert af.phi >= min(0.9, 1.05 * t10 / 100.0) - 1e-12
    with pytest.raises(ValueError):
        AppearanceFunction.from_t10(0.0)


def test_austin_selection_rate_shape():
    d_mm = np.array([0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    s = austin_selection_rates(d_mm, a=1.2, alpha=1.0, mu_mm=2.0, Lambda=1.0)
    assert np.all(s > 0.0)
    assert np.all(np.diff(s) > 0.0)
    # small sizes: S ~ a (d/1mm)^alpha
    assert s[3] == pytest.approx(1.2 / (1.0 + (1.0 / 2.0)), rel=1e-12)
    # linear in a
    s2 = austin_selection_rates(d_mm, a=2.4, alpha=1.0, mu_mm=2.0, Lambda=1.0)
    assert np.allclose(s2, 2.0 * s, rtol=1e-12)


def test_t10_energy_law_limits():
    assert float(t10_energy_law(0.0, 45.0, 0.8)) == 0.0
    assert float(t10_energy_law(50.0, 45.0, 0.8)) == pytest.approx(45.0, abs=1e-6)
    mid = float(t10_energy_law(1.0, 45.0, 0.8))
    assert mid == pytest.approx(45.0 * (1.0 - math.exp(-0.8)), rel=1e-12)


def test_sectional_pbm_matches_matrix_exponential():
    """The BDF integration must reproduce expm(A t) p0 of the same operator."""
    grid = VolumeGrid.from_diameters(50e-6, 2000e-6, 20)
    d = grid.diameters
    af = AppearanceFunction.from_t10(20.0)
    b = mass_fraction_matrix(d, af)
    A_op = volume_daughter_matrix(d, b) @ np.diag(austin_selection_rates(d * 1000.0,
                                                                         0.05, 1.0, 1.0, 0.5))
    A_op[np.diag_indices_from(A_op)] -= austin_selection_rates(d * 1000.0, 0.05, 1.0, 1.0, 0.5)
    feed = np.zeros(grid.n_sections)
    feed[-1] = 1.0
    t = 5.0
    S = austin_selection_rates(d * 1000.0, 0.05, 1.0, 1.0, 0.5)
    out = batch_grind(grid, feed, S, af, times=[t])
    # the state variable of the sectional balance is number density, not mass
    # fraction: n = w/v.  Convert, evolve exactly, convert back.
    v = grid.pivots
    exact_n = np.clip(expm(A_op * t) @ (feed / v), 0.0, None)
    exact = v * exact_n
    exact = exact / exact.sum()
    assert np.max(np.abs(out["mass_fractions"][-1] - exact)) < 1e-6
    # grinding must move mass below the feed size (P80 itself is grid-limited
    # here because the feed sits in the top section)
    assert passing_at(d, out["mass_fractions"][-1], 1.0e-3) > passing_at(d, feed, 1.0e-3)
    assert t10_from_psd(d, out["mass_fractions"][-1], d[-1]) > 0.0


def test_sectional_pbm_conserves_mass():
    grid = VolumeGrid.from_diameters(50e-6, 2000e-6, 22)
    af = AppearanceFunction.from_t10(30.0)
    feed = np.zeros(grid.n_sections)
    feed[-1] = 1.0
    S = austin_selection_rates(grid.diameters * 1000.0, 0.2, 1.0, 2.0, 0.7)
    out = batch_grind(grid, feed, S, af, times=[0.0, 1.0, 10.0])
    totals = np.array([float(np.sum(out["mass"][k])) for k in range(out["mass"].shape[0])])
    assert np.allclose(totals, totals[0], rtol=1e-9)
    cum = cumulative_passing(grid.diameters, out["mass_fractions"][-1])
    assert cum[-1] == pytest.approx(1.0, rel=1e-12)
    assert np.all(np.diff(cum) >= -1e-12)
    assert t10_from_psd(grid.diameters, out["mass_fractions"][-1],
                        grid.diameters[-1]) > 0.0


def test_collision_spectrum_selection_rates():
    """Faster impacts and softer ore must both raise the selection rates."""
    rng = np.random.default_rng(0)
    n = 4000
    ra = np.full(n, 0.0125)
    rb = np.full(n, 0.009)
    ore = np.array([0.2e-3, 0.5e-3, 1.0e-3, 2.0e-3])
    slow = selection_rates_from_collisions(rng.uniform(0.1, 0.5, n), ra, rb, ore, window_s=1.0)
    fast = selection_rates_from_collisions(rng.uniform(1.0, 2.0, n), ra, rb, ore, window_s=1.0)
    soft = selection_rates_from_collisions(rng.uniform(0.1, 0.5, n), ra, rb, ore,
                                          window_s=1.0, e50_ref_j_kg=120.0)
    assert np.all(fast["S_per_s"] > slow["S_per_s"])
    assert np.all(soft["S_per_s"] > slow["S_per_s"])
    assert np.all(np.isfinite(slow["S_per_s"]))
    assert np.all(slow["mean_breakage_probability"] <= 1.0)
    # mass-specific collision energy falls as the ore gets coarser
    assert np.all(np.diff(slow["mean_specific_energy_j_kg"]) < 0.0)
    # passing zero energy must give zero rate
    zero = selection_rates_from_collisions(np.zeros(10), ra[:10], rb[:10], ore, window_s=1.0)
    assert np.allclose(zero["S_per_s"], 0.0)


def test_specific_energy_unit_conversion():
    # 1 kW into 1 t for 1 h is 1 kWh/t
    assert specific_energy_kwh_t(1000.0, 3600.0, 1000.0) == pytest.approx(1.0, rel=1e-12)
    assert specific_energy_kwh_t(1000.0, 3600.0, 500.0) == pytest.approx(2.0, rel=1e-12)


def test_mill_config_and_state_guards():
    with pytest.raises(ValueError):
        MillConfig(diameter=0.3, ball_radii=(0.2,), ball_counts=(1,))
    cfg = MillConfig()
    assert cfg.layers == pytest.approx(cfg.length / cfg.slice_thickness, rel=1e-12)
    state = make_mill_state(cfg, fill=0.2, rpm=40.0, seed=1)
    assert state.n > 5
    assert state.charge_mass == pytest.approx(state.charge_mass_2d * cfg.layers, rel=1e-12)
    # all particles start inside the shell
    assert np.all(np.linalg.norm(state.pos, axis=1) + state.radii <= cfg.radius + 1e-9)