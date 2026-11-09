"""Acceptance tests for :mod:`tensordem.ice_ridge_keel_freezebond`.

The suite has three layers, in the order of the physical chain:

* the **adfreeze closure** (mirror of ``tensorphysics.ice.adfreeze``): parity with
  the TensorPhysics source when it is importable, monotonicity, limits;
* the **adfreeze -> DEM bond mapping** (``F_break = tau A_contact``): ten
  parameter sets through a real DEM twin-particle pull test, plus the negative
  control ``dT = 0 -> F_break = 0``;
* the **keel DEM**: a consolidated, freeze-bonded rubble pile (``N >= 500``
  Rosin-Rammler blocks) with Voronoi porosity, force chains, plate drag and the
  rigid-plastic (Kerr-type) reference, plus the negative controls
  ``h_k = 0 -> F_keel = 0`` and ``n -> 1 -> F_keel -> 0``, and mass conservation.

The DEM fixtures are built once per session and reused through deep copies, so
the whole file stays inside a few minutes on one core.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import sys

import numpy as np
import pytest

from tensordem import ice_ridge_keel_freezebond as keel


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
SMALL = dict(n_particles=520, depth_m=4.0, width_m=6.0, length_m=6.0,
             porosity=0.34, dt=6.0e-4, release_steps=200)


def _config(**overrides) -> keel.KeelConfig:
    params = dict(SMALL)
    params.update(overrides)
    return keel.KeelConfig(**params)


@pytest.fixture(scope="module")
def consolidated() -> keel.KeelRubble:
    """One consolidated (sedimenting) rubble pile, reused by the DEM tests."""
    sim = keel.KeelRubble(_config())
    sim.consolidate(preload=1.0, steps=2500, buoyant=False, viscosity=20.0,
                    overlap_guard=0.08, final_relax_steps=400)
    return sim


@pytest.fixture(scope="module")
def floating(consolidated) -> keel.KeelRubble:
    """The same pile handed over to buoyancy, with freeze bonds (-10 degC)."""
    sim = copy.deepcopy(consolidated)
    sim.initialise_bonds()
    sim.float_free_relax(300)
    sim.close_bottom(0.0)
    return sim


# ---------------------------------------------------------------------------
# 1. adfreeze closure
# ---------------------------------------------------------------------------
def test_seawater_freezing_point_freshwater_exact_and_saline_monotone():
    """Liquidus: freshwater is exactly 0 degC and T_f decreases with salinity.

    NOTE (upstream finding): the mirrored TensorPhysics liquidus table maps
    ``S = 35 g/kg`` to ``-15 degC`` instead of the accepted ``-1.9 degC`` (the
    table is a brine-salinity-vs-temperature curve and the inverse is applied to
    the *bulk* salinity).  The mirror reproduces it bit for bit so that the parity
    test stays meaningful; the consequence for this module is that only the
    freshwater limit ``S = 0`` (the campaign case) is thermodynamically exact.
    """
    assert keel.seawater_freezing_point(0.0) == pytest.approx(0.0)
    # monotone over the range where the mirrored brine curve is meaningful
    tf = [keel.seawater_freezing_point(s) for s in (17.5, 35.0, 78.0, 143.0)]
    assert all(b > a for a, b in zip(tf, tf[1:]))
    assert keel.seawater_freezing_point(35.0) == pytest.approx(-15.0)   # as mirrored


def test_reference_strength_is_monotone_and_capped():
    """tau_ref grows with undercooling and never exceeds the ice shear strength."""
    temps = np.linspace(0.0, 60.0, 25)
    tau = [keel.reference_strength_mpa(dT) for dT in temps]
    assert tau[0] == 0.0
    assert all(b >= a for a, b in zip(tau, tau[1:]))          # non-decreasing
    assert all(b > a for a, b in zip(tau[:12], tau[1:13]))    # strictly, pre-cap
    assert max(tau) <= keel.TAU_ICE_CAP_MPA + 1e-12


def test_adfreeze_strength_increases_with_undercooling():
    """tau_adfreeze at fixed contact time grows with |T| over -2 .. -20 degC."""
    closure = keel.AdfreezeClosure()
    taus = [closure.tau_mpa(T) for T in (-2.0, -5.0, -10.0, -20.0)]
    assert all(b > a for a, b in zip(taus, taus[1:]))
    assert 0.05 < taus[0] < taus[-1] < 2.0


def test_adfreeze_zero_at_and_above_freezing_point():
    """Negative control: a joint at its liquidus has no strength and no bond."""
    closure = keel.AdfreezeClosure()
    assert closure.tau_mpa(0.0) == 0.0
    assert closure.tau_mpa(0.5) == 0.0            # warmer than the liquidus
    saline = keel.AdfreezeClosure(salinity_gkg=35.0)
    tf = keel.seawater_freezing_point(35.0)
    assert saline.tau_mpa(tf) == 0.0


def test_adfreeze_contact_time_kinetics():
    """The frozen area fraction (Stefan growth) is monotone in contact time."""
    etas = [keel.frozen_area_fraction(10.0, t) for t in (10.0, 100.0, 1.0e3, 1.0e4)]
    assert all(b > a for a, b in zip(etas, etas[1:]))
    assert keel.frozen_area_fraction(0.0, 1.0e4) == 0.0


def test_adfreeze_salinity_penalty_reduces_strength():
    """Brine volume fraction reduces the joint strength at equal temperature."""
    fresh = keel.AdfreezeClosure(salinity_gkg=0.0).tau_mpa(-10.0)
    saline = keel.AdfreezeClosure(salinity_gkg=32.0).tau_mpa(-10.0)
    assert 0.0 < saline < fresh


def test_tensorphysics_adfreeze_parity():
    """Mirror parity against ``tensorphysics.ice.adfreeze`` when it is available."""
    if importlib.util.find_spec("tensorphysics") is None:
        pytest.skip("TensorPhysics source tree is not importable in this environment")
    from tensorphysics.ice import adfreeze as ref

    for temp in (-2.0, -5.0, -10.0, -20.0):
        for salinity in (0.0, 8.0, 32.0):
            for t_contact in (1.0e3, 3.6e3):
                got = keel.AdfreezeClosure(contact_time_s=t_contact,
                                           salinity_gkg=salinity).tau_mpa(temp)
                want = ref.adfreeze_strength(temp, t_contact, salinity)
                assert got == pytest.approx(want, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. adfreeze -> DEM bond mapping  (criterion 1 + negative control 7a)
# ---------------------------------------------------------------------------
# (temperature, contact_time_s, roughness_um, salinity, r_i, r_j, penetration)
#
# All freshwater: with the mirrored TensorPhysics liquidus a saline joint at these
# temperatures sits *above* its liquidus (``T_f(32 g/kg) = -13 degC`` as mirrored),
# so ``tau_adfreeze`` is identically zero there.  The saline branch is therefore
# exercised by the closure tests, not by the pull test; the upstream liquidus
# finding is reported.
BOND_PARAMS = [
    (-2.0, 3.6e3, 2.5, 0.0, 0.30, 0.30, 2.0e-3),
    (-5.0, 3.6e3, 2.5, 0.0, 0.20, 0.45, 1.0e-3),
    (-10.0, 3.6e3, 2.5, 0.0, 0.25, 0.25, 5.0e-3),
    (-20.0, 3.6e3, 2.5, 0.0, 0.15, 0.50, 3.0e-3),
    (-10.0, 6.0e2, 2.5, 0.0, 0.30, 0.30, 2.0e-3),
    (-10.0, 3.6e3, 20.0, 0.0, 0.30, 0.30, 2.0e-3),
    (-14.0, 7.2e3, 15.0, 0.0, 0.55, 0.30, 2.0e-3),
    (-8.0, 1.8e3, 10.0, 0.0, 0.40, 0.25, 1.5e-3),
    (-16.0, 3.6e3, 5.0, 0.0, 0.10, 0.10, 8.0e-4),
    (-12.0, 3.6e3, 2.5, 0.0, 0.30, 0.20, 2.5e-3),
]


@pytest.mark.parametrize("params", BOND_PARAMS)
def test_dem_bond_break_force_matches_adfreeze_times_area(params):
    """Criterion 1: DEM ``F_break`` vs ``tau_adfreeze * A_contact`` within 5%."""
    temperature, t_contact, roughness, salinity, r_i, r_j, penetration = params
    closure = keel.AdfreezeClosure(contact_time_s=t_contact, salinity_gkg=salinity,
                                   roughness_um=roughness)
    res = keel.bonded_pair_pull(r_i, r_j, penetration, closure, temperature,
                                pull_velocity=1.0e-3, dt=2.0e-4, n_steps=40000)
    analytic = res["f_break_analytic_n"]
    assert analytic > 0.0
    assert res["broke"] is True
    rel = abs(res["f_peak_n"] - analytic) / analytic
    assert rel <= 0.05, f"DEM break force off by {rel:.3%}"
    # and it really is tau_adfreeze * A_contact
    expected = res["tau_adfreeze_pa"] * res["area_m2"]
    assert analytic == pytest.approx(expected, rel=1e-12)


def test_bond_break_force_scales_with_contact_area():
    """F_break is linear in the frozen neck area (same tau, different blocks)."""
    closure = keel.AdfreezeClosure()
    small = keel.freeze_bond_spec(closure, -10.0, 0.10, 0.10, 1e-3)
    big = keel.freeze_bond_spec(closure, -10.0, 0.40, 0.40, 1e-3)
    assert big.f_break_n / small.f_break_n == pytest.approx(16.0, rel=1e-12)


def test_negative_control_zero_undercooling_gives_zero_break_force():
    """Negative control 7a: ``dT = 0 -> F_break = 0`` (strength and area)."""
    closure = keel.AdfreezeClosure()
    spec = keel.freeze_bond_spec(closure, 0.0, 0.3, 0.3, 2e-3)
    assert spec.f_break_n == 0.0
    assert spec.f_shear_break_n == 0.0
    # ... and the geometric half: no contact overlap, no area
    assert keel.neck_area_m2(0.3, 0.3) > 0.0
    assert keel.contact_area_m2(0.3, 0.3, 0.0) == 0.0


# ---------------------------------------------------------------------------
# 3. keel packing, porosity and structure
# ---------------------------------------------------------------------------
def test_packing_has_no_initial_overlap(consolidated):
    """The block cloud handed to the DEM is overlap free by construction."""
    sim = keel.KeelRubble(_config())
    from scipy.spatial import cKDTree
    pr = cKDTree(sim.pos).query_pairs(2.2 * float(sim.rad.max()), output_type="ndarray")
    d = np.linalg.norm(sim.pos[pr[:, 1]] - sim.pos[pr[:, 0]], axis=1)
    overlap = sim.rad[pr[:, 0]] + sim.rad[pr[:, 1]] - d
    assert float(overlap.max()) <= 1e-9
    assert sim.n >= 500                       # criterion: N >= 500 blocks
    assert sim.total_mass() > 0.0


def test_rosin_rammler_size_distribution():
    """The block diameters follow the Rosin-Rammler law they claim to follow."""
    d = keel.rosin_rammler_diameters(20000, 1.6, u_range=(0.30, 0.92), seed=7)
    assert d.min() > 0.0
    # inverse CDF: F(d) = 1 - exp(-(d/d0)^m) with d0 = 1 -> check two quantiles
    for q in (0.25, 0.5, 0.75):
        model = (-math.log(1.0 - (0.30 + q * (0.92 - 0.30)))) ** (1.0 / 1.6)
        assert np.quantile(d, q) == pytest.approx(model, rel=0.03)


def test_consolidation_generates_a_jammed_packing(consolidated):
    """The compaction reaches a dense, low-overlap packing (not a crushed mush)."""
    cd = consolidated.contact_data()
    r_min = np.minimum(consolidated.rad[cd["i"]], consolidated.rad[cd["j"]])
    overlap_ratio = cd["overlap"] / r_min
    assert overlap_ratio.size > 400                     # a real contact network
    assert float(overlap_ratio.mean()) < 0.02           # overlaps stay DEM-legal
    assert float(overlap_ratio.max()) < 0.35
    packing = consolidated.solid_volume() / consolidated.box_volume
    assert 0.55 < packing < 0.72                        # jammed, not loose


def test_voronoi_porosity_is_exact_for_one_sphere():
    """The Voronoi measure reproduces the analytic porosity of a single block."""
    pos = np.array([[3.0, 3.0, 2.0]])
    rad = np.array([1.0])
    out = keel.voronoi_porosity(pos, rad, (0.0, 6.0, 0.0, 6.0, 0.0, 4.0), voxel_m=0.05)
    assert out["porosity"] == pytest.approx(1.0 - (4.0 / 3.0) * np.pi / 144.0, abs=5e-3)


def test_standing_keel_porosity_in_range(floating):
    """Criterion 2: the freeze-bonded standing keel has ``n = 0.30 .. 0.45``."""
    meas = floating.measure(voxel_m=0.12)
    n = meas["porosity"]["porosity"]
    assert 0.30 <= n <= 0.45, f"bonded keel porosity n={n:.3f} outside [0.30, 0.45]"
    assert meas["particles_inside_envelope"] >= 450     # the skeleton holds together


def test_force_chain_tail_is_scale_free(floating):
    """Criterion 6: the strong-force tail is a power law with ``R^2 >= 0.90``."""
    meas = floating.measure(voxel_m=0.15)
    chain = meas["force_chain"]
    assert chain["n_contacts"] > 200
    assert math.isfinite(chain["alpha"]) and 1.0 < chain["alpha"] < 4.0
    assert chain["r2"] >= 0.90, f"force-chain fit R^2={chain['r2']:.3f}"


def test_bonds_form_and_survive_the_float_transition(floating):
    """Freeze bonds are created from the contacts and most of them survive."""
    alive = int(floating.bond_alive.sum())
    formed = int(floating.bond_formed)
    assert formed > 300
    assert alive > 0.4 * formed, f"only {alive}/{formed} bonds survived"


def test_bond_network_carries_cohesion(floating):
    """The smeared Mohr-Coulomb cohesion of the bonded network is positive."""
    alive = floating.bond_alive
    cohesion = keel.bond_cohesion_pa(
        int(alive.sum()), float(np.mean(floating.bond_len[alive])),
        float(np.mean(floating.bond_ft_break[alive])), floating.box_volume,
    )
    assert cohesion > 50.0                             # Pa
    assert cohesion < 5.0e4                            # < 50 kPa, not absurd


# ---------------------------------------------------------------------------
# 4. drag resistance and the Kerr reference
# ---------------------------------------------------------------------------
def test_kerr_reference_limits():
    """Kerr-type reference: linear in width, zero at zero depth, monotone in c."""
    base = dict(depth_m=4.0, width_m=6.0, porosity=0.38, friction_angle_deg=25.0,
                cohesion_pa=1000.0, surcharge_pa=2700.0)
    ref = keel.kerr_keel_resistance(**base)
    wide = keel.kerr_keel_resistance(**{**base, "width_m": 12.0})
    assert wide["f_kerr_N"] == pytest.approx(2.0 * ref["f_kerr_N"], rel=1e-12)
    zero = keel.kerr_keel_resistance(**{**base, "depth_m": 0.0})
    assert zero["f_kerr_N"] == 0.0
    stronger = keel.kerr_keel_resistance(**{**base, "cohesion_pa": 2000.0})
    assert stronger["f_kerr_N"] > ref["f_kerr_N"]
    # buoyant unit weight is negative, so rubble weight *reduces* the resistance
    assert ref["gamma_buoyant_Pa_per_m"] < 0.0
    assert ref["weight_term_N_m"] < 0.0


def test_drag_resistance_is_positive_and_repeatable(floating):
    """The plate drag produces a finite, reproducible resistance on the keel."""
    sim = copy.deepcopy(floating)
    run = sim.drag(distance_m=0.4, velocity=0.45, av_window=0.5)
    assert run.f_keel_n > 0.0
    assert run.state["f_keel_std_N"] >= 0.0
    assert np.all(np.isfinite(run.force_history))
    sim2 = copy.deepcopy(floating)
    run2 = sim2.drag(distance_m=0.4, velocity=0.45, av_window=0.5)
    assert run.f_keel_n == pytest.approx(run2.f_keel_n, rel=1e-9)   # deterministic


def test_bonded_keel_resists_more_than_unbonded(consolidated):
    """Freeze bonds raise the keel resistance: the bonded case must be stronger."""
    bonded = copy.deepcopy(consolidated)
    bonded.initialise_bonds()
    bonded.float_free_relax(250)
    unbonded = copy.deepcopy(consolidated)
    unbonded.bonds_enabled = False
    unbonded.float_free_relax(250)
    f_b = bonded.drag(distance_m=0.35, velocity=0.45).f_keel_n
    f_u = unbonded.drag(distance_m=0.35, velocity=0.45).f_keel_n
    assert f_b > 1.02 * f_u, f"bonded {f_b/1e3:.0f} kN vs unbonded {f_u/1e3:.0f} kN"


def test_negative_control_zero_depth_keel_has_no_resistance():
    """Negative control 7b: ``h_k = 0`` (no rubble) -> ``F_keel = 0``."""
    cfg = keel.KeelConfig(n_particles=4, depth_m=1.0, width_m=2.0, length_m=2.0,
                          porosity=0.4, dt=5e-4)
    sim = keel.KeelRubble(cfg)
    # a keel with no blocks at all: every block is removed from the measurement
    sim.pos[:] = -1.0e3
    run = sim.drag(distance_m=0.3, velocity=0.45)
    assert run.f_keel_n == 0.0
    assert keel.kerr_keel_resistance(
        depth_m=0.0, width_m=6.0, porosity=0.38, friction_angle_deg=25.0,
        cohesion_pa=1000.0, surcharge_pa=2700.0)["f_kerr_N"] == 0.0


def test_negative_control_vanishing_solid_fraction():
    """Negative control 7c: ``n -> 1`` (no solid) -> ``F_keel -> 0``."""
    cfg = keel.KeelConfig(n_particles=520, depth_m=4.0, width_m=6.0, length_m=6.0,
                          porosity=0.34, dt=6e-4)
    sim = keel.KeelRubble(cfg, radius_scale=1.0e-3)
    sim.buoyant = True
    sim.floor_z = 0.0
    run = sim.drag(distance_m=0.4, velocity=0.45)
    assert run.f_keel_n < 1.0e3          # a few newtons, not hundreds of kN
    ref = keel.kerr_keel_resistance(depth_m=4.0, width_m=6.0, porosity=0.999,
                                    friction_angle_deg=25.0, cohesion_pa=0.0,
                                    surcharge_pa=0.0)
    assert ref["f_kerr_N"] < 1.0e3


def test_mass_is_conserved_exactly(floating):
    """Criterion 8: the DEM conserves mass to machine precision."""
    sim = copy.deepcopy(floating)
    m0 = sim.total_mass()
    sim.drag(distance_m=0.3, velocity=0.45)
    drift = abs(sim.total_mass() - m0) / m0
    assert drift <= 1.0e-12
    assert drift <= 1.0e-6               # the acceptance bound, with margin


def test_macroscopic_friction_below_contact_friction():
    """The DEM upscaling gives ``phi_macro < atan(mu)`` (rearrangement costs strength)."""
    out = keel.macroscopic_friction_angle(n_particles=300, size_m=3.0,
                                          confining_pa=5.0e3, axial_rate=0.25,
                                          max_strain=0.12, settle_steps=600)
    phi = out["phi_macro_deg"]
    assert 5.0 < phi < out["contact_friction_angle_deg"]
    assert out["sin_phi"] > 0.05


def test_configuration_validation():
    """Degenerate configurations are rejected instead of silently mis-running."""
    with pytest.raises(ValueError):
        keel.KeelConfig(depth_m=0.0)
    with pytest.raises(ValueError):
        keel.KeelConfig(porosity=1.0)
    with pytest.raises(ValueError):
        keel.AdfreezeClosure(gap_scale_m=0.0)
    with pytest.raises(ValueError):
        keel.contact_area_m2(0.0, 0.1, 1e-3)
    with pytest.raises(ValueError):
        keel.kerr_keel_resistance(depth_m=4.0, width_m=6.0, porosity=0.4,
                                  friction_angle_deg=90.0, cohesion_pa=0.0,
                                  surcharge_pa=0.0)