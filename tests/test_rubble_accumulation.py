"""Unit + integration tests for :mod:`tensordem.rubble_accumulation`.

Case IN69-N06 (rubble accumulation around a fixed cylinder).  The tests cover
the closed-form pieces (SDF, spherical cap, Archimedean equilibrium), the
initial-condition bookkeeping, the growth-law fitter, the geometry extraction,
the two negative controls demanded by the acceptance criteria, mass
conservation, structure impenetrability and determinism.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.rubble_accumulation import (
    RubbleAccumulation,
    momentum_baseline_pressure,
    shielding_ratio,
    RubbleConfig,
    cylinder_sdf,
    extract_pile_geometry,
    fit_exponential_growth,
    floating_equilibrium_height,
    run_rubble_case,
    submerged_volume,
)


def small_config(**kw) -> RubbleConfig:
    """Cheap configuration: D = 1 m cylinder, coarse blocks, short run."""
    base = dict(diameter=1.0, ice_thickness=0.25, concentration=0.4,
                drift_speed=0.3, particle_budget=350, max_steps=400,
                samples=40, driving_pressure=20_000.0)
    base.update(kw)
    return RubbleConfig(**base)


# --------------------------------------------------------------------------
# closed-form geometry
# --------------------------------------------------------------------------
def test_cylinder_sdf_closed_form():
    """sdf = hypot(x, y) - R, with the right sign on both sides."""
    r_struct = 1.5
    pts = np.array([[0.0, 0.0, 0.0], [r_struct, 0.0, 0.0], [r_struct + 2.0, 0.0, 0.0],
                    [0.0, -r_struct, 5.0], [3.0, 4.0, -2.0]])
    sdf = cylinder_sdf(pts, r_struct)
    assert sdf[0] == pytest.approx(-r_struct)
    assert sdf[1] == pytest.approx(0.0, abs=1e-12)
    assert sdf[2] == pytest.approx(2.0)
    assert sdf[3] == pytest.approx(0.0, abs=1e-12)
    assert sdf[4] == pytest.approx(5.0 - r_struct)


def test_submerged_volume_cap_limits():
    """Cap volume: 0 above the waterline, V fully submerged, V/2 at mid-depth."""
    r = 0.1
    v_full = 4.0 / 3.0 * math.pi * r ** 3
    assert submerged_volume(r, r) == pytest.approx(0.0)
    assert submerged_volume(-r, r) == pytest.approx(v_full)
    assert submerged_volume(0.0, r) == pytest.approx(0.5 * v_full)


def test_floating_equilibrium_is_archimedean():
    """The equilibrium centre height reproduces rho_i/rho_w exactly."""
    r = 0.07
    rho_i, rho_w = 900.0, 1025.0
    z = floating_equilibrium_height(r, rho_i, rho_w)
    frac = submerged_volume(z, r) / (4.0 / 3.0 * math.pi * r ** 3)
    assert frac == pytest.approx(rho_i / rho_w, rel=1e-6)
    assert -r < z < 0.0


def test_single_block_floats_at_equilibrium():
    """A free block relaxes onto the Archimedean waterline."""
    cfg = small_config(concentration=0.0, drift_speed=0.0)   # empty, still water
    sim = RubbleAccumulation(cfg)
    sim.pos = np.array([[-cfg.diameter, 0.0, 3.0 * cfg.particle_radius]])
    sim.vel = np.zeros((1, 3))
    sim.omega = np.zeros((1, 3))
    sim.n = 1
    sim._prev_d = np.hypot(sim.pos[:, 0], sim.pos[:, 1])
    for _ in range(3000):
        sim.step()
    assert sim.pos[0, 2] == pytest.approx(cfg.equilibrium_height, abs=0.05 * cfg.particle_radius)


# --------------------------------------------------------------------------
# initial condition
# --------------------------------------------------------------------------
def test_initial_field_matches_concentration_and_has_no_overlap():
    cfg = small_config()
    sim = RubbleAccumulation(cfg)
    assert sim.n > 50
    ice_volume = sim.n * 4.0 / 3.0 * math.pi * cfg.particle_radius ** 3
    target = cfg.concentration * cfg.ice_thickness * cfg.domain_area
    assert ice_volume == pytest.approx(target, rel=0.25)
    pos = sim.pos
    tree_dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(tree_dist, np.inf)
    assert tree_dist.min() > 1.9 * cfg.particle_radius


def test_capture_shell_is_empty_at_start():
    cfg = small_config()
    sim = RubbleAccumulation(cfg)
    d = np.hypot(sim.pos[:, 0], sim.pos[:, 1])
    assert d.min() > cfg.capture_zone_radius
    geo = sim.pile_geometry()
    assert geo.upstream_length == 0.0 and geo.height == 0.0 and geo.mass == 0.0


# --------------------------------------------------------------------------
# growth-law fitter
# --------------------------------------------------------------------------
def test_growth_fit_recovers_known_law():
    t = np.linspace(0.0, 20.0, 120)
    y = 2.5 * (1.0 - np.exp(-t / 3.0))
    y = y + 0.01 * np.random.default_rng(0).normal(size=t.size)
    fit = fit_exponential_growth(t, y)
    assert fit.ok
    assert fit.inf == pytest.approx(2.5, rel=0.02)
    assert fit.tau == pytest.approx(3.0, rel=0.05)
    assert fit.r2 > 0.99


def test_growth_fit_reports_no_growth_instead_of_fabricating():
    t = np.linspace(0.0, 5.0, 40)
    fit = fit_exponential_growth(t, np.zeros_like(t))
    assert not fit.ok
    assert "no growth" in fit.message


def test_growth_fit_rejects_short_series():
    fit = fit_exponential_growth(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert not fit.ok
    assert "too few points" in fit.message


# --------------------------------------------------------------------------
# geometry extraction
# --------------------------------------------------------------------------
def test_pile_geometry_synthetic_cloud():
    cfg = small_config(concentration=0.4)
    r = cfg.particle_radius
    n = 400
    rng = np.random.default_rng(1)
    x = rng.uniform(-cfg.structure_radius - 0.9, -cfg.structure_radius, n)
    y = rng.uniform(-0.8, 0.8, n)
    z = np.full(n, cfg.z_reference - 0.5 * r)
    pos = np.column_stack([x, y, z])
    vel = np.zeros((n, 3))
    geo = extract_pile_geometry(pos, vel, cfg)
    assert geo.count >= int(0.95 * n)
    # 2% / 98% percentiles of the uniform cloud, by construction
    assert geo.upstream_length == pytest.approx(0.9 * 0.96, rel=0.06)
    assert geo.width == pytest.approx(1.6 * 0.98, rel=0.06)
    assert geo.height == pytest.approx(0.5 * r, rel=1e-6)


def test_pile_geometry_ignores_flyers_and_drifters():
    """A fast block and a distant block must not change the pile envelope."""
    cfg = small_config()
    r = cfg.particle_radius
    n = 300
    rng = np.random.default_rng(2)
    x = rng.uniform(-cfg.structure_radius - 1.0, -cfg.structure_radius, n)
    y = rng.uniform(-0.5, 0.5, n)
    z = np.full(n, cfg.z_reference - r + 0.4 * r)
    pos = np.column_stack([x, y, z])
    vel = np.zeros((n, 3))
    geo_base = extract_pile_geometry(pos, vel, cfg)
    extra = np.array([[-cfg.structure_radius - 0.9, 0.0, cfg.z_reference],
                      [5.0 * cfg.diameter, 0.0, cfg.z_reference]])
    vel_extra = np.array([[50.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    geo_extra = extract_pile_geometry(np.vstack([pos, extra]),
                                      np.vstack([vel, vel_extra]), cfg)
    assert geo_extra.upstream_length == pytest.approx(geo_base.upstream_length, rel=0.02)
    assert geo_extra.width == pytest.approx(geo_base.width, rel=0.02)
    assert geo_extra.height == pytest.approx(geo_base.height, rel=0.02)


# --------------------------------------------------------------------------
# negative controls (acceptance criterion (e))
# --------------------------------------------------------------------------
def test_negative_control_zero_concentration():
    cfg = small_config(concentration=0.0)
    sim = RubbleAccumulation(cfg)
    assert sim.n == 0
    sim.run(t_end=0.5)
    geo = sim.pile_geometry()
    assert geo.mass == 0.0 and geo.upstream_length == 0.0 and geo.height == 0.0
    assert sim.mass_ledger()["mass_in_domain"] == 0.0


def test_negative_control_zero_drift():
    cfg = small_config(drift_speed=0.0)
    sim = RubbleAccumulation(cfg)
    assert sim.n > 50
    sim.run(t_end=2.0)
    geo = sim.pile_geometry()
    assert geo.mass == pytest.approx(0.0, abs=1e-9)
    assert geo.height == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# conservation / integrity / determinism
# --------------------------------------------------------------------------
def test_mass_balance_between_inventory_and_outflow():
    cfg = small_config()
    sim = RubbleAccumulation(cfg)
    sim.run(t_end=1.5)
    led = sim.mass_ledger()
    assert led["mass_initial"] + led["mass_inflow"] == pytest.approx(
        led["mass_in_domain"] + led["mass_exited"], rel=1e-10)
    # the pile is a subset of the inventory, never more
    assert led["pile_mass"] <= led["mass_in_domain"] + 1e-9


def test_zone_flux_ledger_small_error():
    cfg = small_config()
    sim = RubbleAccumulation(cfg)
    sim.run(t_end=1.5)
    led = sim.mass_ledger()
    assert led["zone_closure_error"] < 0.05


def test_structure_is_impenetrable():
    cfg = small_config(concentration=0.6, max_steps=600)
    sim = RubbleAccumulation(cfg)
    sim.run(t_end=2.0)
    d = np.hypot(sim.pos[:, 0], sim.pos[:, 1])
    assert d.min() >= cfg.structure_radius + cfg.particle_radius - 1e-9
    assert np.abs(sim.vel).max() < 5.0 * max(cfg.drift_speed, 1e-3) + 1.0


def test_deterministic_from_seed():
    a = run_rubble_case(small_config(seed=3), t_end=1.0)
    b = run_rubble_case(small_config(seed=3), t_end=1.0)
    c = run_rubble_case(small_config(seed=4), t_end=1.0)
    assert a["ledger"]["mass_initial"] == pytest.approx(b["ledger"]["mass_initial"])
    assert a["ledger"]["mass_initial"] != pytest.approx(c["ledger"]["mass_initial"])
    assert a["history"]["n_blocks"] == b["history"]["n_blocks"]


def test_case_record_has_all_acceptance_fields():
    rec = run_rubble_case(small_config(), t_end=1.0)
    for key in ("config", "fits", "steady", "ledger", "history"):
        assert key in rec
    for key in ("L", "B", "H"):
        assert set(rec["fits"][key]) >= {"inf", "tau", "r2", "ok"}
    for key in ("mass_in_domain", "mass_exited", "zone_closure_error",
                "closure_error", "pile_mass"):
        assert key in rec["ledger"]
    assert rec["config"]["froude"] > 0.0


def test_sweeper_baseline_removes_resident_blocks_in_the_shell():
    """Cleared-cylinder baseline: resident blocks in the shell are swept."""
    cfg = small_config(concentration=0.6, sweep_shell_radii=2.0, max_steps=600)
    sim = RubbleAccumulation(cfg)
    sim.run(t_end=2.0)
    assert sim.mass_swept > 0.0
    # a block that is only *passing* is never swept, so the shell may contain
    # blocks in transit; what the sweeper guarantees is that nothing stays
    # resident on the structure: no block keeps its residence timer above the
    # dwell threshold from one step to the next.
    shell = cfg.structure_radius + cfg.sweep_shell_radii * cfg.particle_radius
    d = np.hypot(sim.pos[:, 0], sim.pos[:, 1])
    speed = np.linalg.norm(sim.vel, axis=1)
    resident = (d <= shell) & (speed < cfg.pile_speed_fraction * cfg.drift_speed)
    assert np.all(sim._dwell[resident] < cfg.sweep_dwell_time)


def test_sweeper_baseline_load_is_nonzero():
    """Regression (amendment 1): the cleared-cylinder baseline must load.

    A block parked on the structure delivers its contact impulse for the dwell
    window and is then swept, so ``p_front > 0``; the previous instantaneous
    deletion returned ``p_0 = 0`` in a whole family of cases and made the
    shielding ratio undefined.
    """
    cfg = small_config(concentration=0.0, drift_speed=0.3, sweep_shell_radii=1.02,
                       sweep_dwell_time=0.01, max_steps=40)
    sim = RubbleAccumulation(cfg)
    r = cfg.particle_radius
    sim.pos = np.array([[-(cfg.structure_radius + 0.9 * r), 0.0, cfg.equilibrium_height]])
    sim.vel = np.array([[0.0, 0.0, 0.0]])
    sim.omega = np.zeros((1, 3))
    sim.n = 1
    sim._prev_d = np.hypot(sim.pos[:, 0], sim.pos[:, 1])
    sim._dwell = np.zeros(1)
    sim.run(t_end=20 * cfg.timestep)
    assert sim.mass_swept > 0.0
    assert sim.history["p_front"][-1] > 0.0


def test_dwell_sweep_spares_fast_blocks():
    """A block crossing the shell at drift speed is not swept."""
    cfg = small_config(concentration=0.0, sweep_shell_radii=1.02,
                       sweep_dwell_time=1.0, max_steps=20)
    sim = RubbleAccumulation(cfg)
    r = cfg.particle_radius
    sim.pos = np.array([[-(cfg.structure_radius + 3.0 * r), 0.0, cfg.equilibrium_height]])
    sim.vel = np.array([[cfg.drift_speed, 0.0, 0.0]])
    sim.omega = np.zeros((1, 3))
    sim.n = 1
    sim._prev_d = np.hypot(sim.pos[:, 0], sim.pos[:, 1])
    sim._dwell = np.zeros(1)
    sim.run(t_end=10 * cfg.timestep)
    assert sim.mass_swept == 0.0
    assert sim._dwell[0] == 0.0


def test_thrust_force_is_pressure_times_frontal_area():
    cfg = small_config(thrust_pressure=20_000.0)
    r = cfg.particle_radius
    assert cfg.thrust_force == pytest.approx(20_000.0 * 4.0 * r * r)
    assert cfg.thrust_band_front == pytest.approx(cfg.x_in + cfg.diameter)
    assert RubbleConfig(**{**small_config().__dict__}).thrust_force == 0.0


def test_thrust_acts_only_inside_the_upstream_band():
    """Only blocks in [x_in, x_in + band] feel the sheet pressure."""
    cfg = small_config(concentration=0.0, thrust_pressure=20_000.0,
                       thrust_band_diameters=1.0, max_steps=5)
    sim = RubbleAccumulation(cfg)
    r = cfg.particle_radius
    z = cfg.equilibrium_height
    inside = np.array([[cfg.x_in + r, 0.0, z]])
    outside = np.array([[cfg.x_in + 1.5 * cfg.diameter, 0.0, z]])
    sim.pos = np.vstack([inside, outside])
    sim.vel = np.zeros((2, 3))               # no drag force: target is v_i = 0.3
    sim.omega = np.zeros((2, 3))
    sim.n = 2
    sim._prev_d = np.hypot(sim.pos[:, 0], sim.pos[:, 1])
    sim._dwell = np.zeros(2)
    f = sim._environment_forces()
    drag = cfg.particle_mass * np.array([cfg.drift_speed, 0.0, 0.0]) / cfg.drag_time
    assert f[0, 0] - drag[0] == pytest.approx(cfg.thrust_force)
    assert f[1, 0] == pytest.approx(drag[0])


def test_momentum_baseline_pressure_reference():
    """p_0 = 0.5 C_d rho_i C_i v_i^2 with the declared C_d = 2."""
    p = momentum_baseline_pressure(0.3, 0.3)
    assert p == pytest.approx(0.5 * 2.0 * 900.0 * 0.3 * 0.09)
    assert momentum_baseline_pressure(0.3, 0.6) == pytest.approx(4.0 * p)
    assert momentum_baseline_pressure(0.6, 0.3) == pytest.approx(2.0 * p)
    assert momentum_baseline_pressure(0.0, 0.3) == 0.0
    assert momentum_baseline_pressure(0.3, 0.0) == 0.0


def test_shielding_ratio_is_nan_not_fabricated():
    assert shielding_ratio(100.0, 50.0) == pytest.approx(2.0)
    assert math.isnan(shielding_ratio(100.0, 0.0))
    assert math.isnan(shielding_ratio(float("nan"), 1.0))