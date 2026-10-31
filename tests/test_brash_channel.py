"""Tests for the brash-ice channel DEM engine (W4-N07)."""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.brash_channel import (
    BrashChannelConfig,
    BrashChannelDEM,
    HullSpec,
    buoyancy_force_and_potential,
    generate_field,
    submerged_fraction,
)


def small_config(**overrides) -> BrashChannelConfig:
    params = dict(
        length_m=1.4,
        width_m=1.4,
        concentration=0.5,
        thickness_m=0.4,
        mean_radius_m=0.07,
        radius_spread=0.15,
        mass_scale=1000.0,
        seed=7,
    )
    params.update(overrides)
    return BrashChannelConfig(**params)


def small_hull(**overrides) -> HullSpec:
    params = dict(
        beam_m=1.1,
        bow_half_angle_deg=55.0,
        bow_length_m=0.55 / math.tan(math.radians(55.0)),
        body_length_m=0.4,
        tip_radius_m=0.06,
        y_center_m=0.7,
        x_apex_m=-0.15,
        speed_ms=0.4,
    )
    params.update(overrides)
    return HullSpec(**params)


# ------------------------------------------------------------------ field/config
def test_config_rejects_unphysical_inputs() -> None:
    with pytest.raises(ValueError):
        BrashChannelConfig(concentration=0.0)
    with pytest.raises(ValueError):
        BrashChannelConfig(concentration=0.7)
    with pytest.raises(ValueError):
        BrashChannelConfig(radius_spread=0.6)
    with pytest.raises(ValueError):
        BrashChannelConfig(friction_mu=-0.1)
    with pytest.raises(ValueError):
        HullSpec(bow_half_angle_deg=100.0)


def test_field_matches_target_ice_volume_and_concentration() -> None:
    for concentration in (0.3, 0.6):
        cfg = small_config(concentration=concentration)
        field = generate_field(cfg)
        total = float(field["volumes"].sum())
        target = cfg.target_ice_volume
        assert total == pytest.approx(target, rel=1e-9)
        assert len(field["positions"]) == cfg.particle_count
        # every piece starts inside the channel walls and above the floor
        assert float(field["positions"][:, 1].min()) >= float(field["radii"].min()) - 1e-9
        assert float(field["positions"][:, 1].max()) <= cfg.width_m - float(field["radii"].min()) + 1e-9


def test_field_is_reproducible_for_a_fixed_seed() -> None:
    a = generate_field(small_config())
    b = generate_field(small_config())
    assert torch.allclose(a["positions"], b["positions"])
    assert torch.allclose(a["radii"], b["radii"])


def test_submerged_fraction_endpoints_and_monotonicity() -> None:
    u = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    f = submerged_fraction(u)
    assert float(f[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(f[1]) == pytest.approx(0.0, abs=1e-12)
    assert float(f[2]) == pytest.approx(0.5, abs=1e-12)
    assert float(f[3]) == pytest.approx(1.0, abs=1e-12)
    assert float(f[4]) == pytest.approx(1.0, abs=1e-12)
    assert torch.all(f[1:] >= f[:-1])


def test_buoyancy_equilibrium_draft_matches_density_ratio() -> None:
    """A floating sphere must submerge to rho_ice/rho_water of its volume."""
    rho_ice, rho_w = 917.0, 1025.0
    radius = torch.tensor([0.1], dtype=torch.float64)
    target = rho_ice / rho_w
    u = torch.linspace(-1.0, 1.0, 20001, dtype=torch.float64)
    f = submerged_fraction(u)
    idx = torch.argmin((f - target).abs())
    depth = float(u[idx]) * 0.1
    z = torch.tensor([-depth], dtype=torch.float64)
    force, _ = buoyancy_force_and_potential(z, radius, rho_w)
    weight = rho_ice * (4.0 / 3.0) * math.pi * 0.1**3 * 9.81
    assert float(force[0]) == pytest.approx(weight, rel=2e-3)


def test_buoyancy_force_is_minus_gradient_of_potential() -> None:
    radius = torch.full((5,), 0.1, dtype=torch.float64)
    z = torch.linspace(-0.16, 0.16, 5, dtype=torch.float64)
    h = 1e-6
    force, potential = buoyancy_force_and_potential(z, radius, 1025.0)
    _, p_plus = buoyancy_force_and_potential(z + h, radius, 1025.0)
    _, p_minus = buoyancy_force_and_potential(z - h, radius, 1025.0)
    grad = -(p_plus - p_minus) / (2.0 * h)
    assert torch.allclose(force, grad, rtol=2e-5, atol=1e-3)


# ------------------------------------------------------------------------ engine
def test_engine_initial_state_has_no_deep_overlaps() -> None:
    dem = BrashChannelDEM(small_config(), [small_hull()])
    dem.step(record=False)
    assert dem.max_overlap_ratio < 0.35


def test_energy_ledger_closes_within_two_percent() -> None:
    dem = BrashChannelDEM(small_config(), [small_hull(x_apex_m=0.05)])
    out = dem.run(
        settle_steps=120,
        spinup_travel_m=0.1,
        measure_travel_m=0.2,
        sample_interval=5,
    )
    ledger = out["energy_ledger"]
    assert ledger["bow_work_j"] > 0.0
    assert abs(ledger["relative_residual"]) < 0.02


def test_hull_outside_the_field_produces_no_resistance() -> None:
    cfg = small_config()
    hull = small_hull(x_apex_m=-3.0)
    dem = BrashChannelDEM(cfg, [hull])
    for _ in range(20):
        dem.step(record=True)
    assert abs(dem.last_hull_reaction[0]) < 1e-9


def test_resistance_increases_with_channel_thickness() -> None:
    results = {}
    for thickness in (0.3, 0.6):
        cfg = small_config(thickness_m=thickness)
        dem = BrashChannelDEM(cfg, [small_hull(x_apex_m=0.05)])
        out = dem.run(settle_steps=150, spinup_travel_m=0.08,
                      measure_travel_m=0.2, sample_interval=5)
        results[thickness] = out["mean_resistance_n"]
    assert results[0.6] > results[0.3] > 0.0


def test_measurement_window_reports_sample_count_and_window_steps() -> None:
    dem = BrashChannelDEM(small_config(), [small_hull(x_apex_m=0.05)])
    out = dem.run(settle_steps=100, spinup_travel_m=0.05,
                  measure_travel_m=0.2, sample_interval=5)
    assert out["samples"] > 0
    assert out["steps_measured"] <= out["steps_window"]


def test_convoy_leader_force_is_measured_per_hull() -> None:
    cfg = small_config()
    leader = small_hull(x_apex_m=0.05, name="leader")
    follower = small_hull(x_apex_m=0.05 - 0.9, name="follower")
    dem = BrashChannelDEM(cfg, [leader, follower])
    out = dem.run(settle_steps=80, spinup_travel_m=0.05, measure_travel_m=0.15,
                  sample_interval=5, sample_hull=0)
    assert len(out["hull_resistances_n"]) == 2
    assert out["mean_resistance_n"] == pytest.approx(out["hull_resistances_n"][0])