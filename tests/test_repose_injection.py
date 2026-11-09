"""Tests for the trickle-injection repose driver (``tensordem.repose_injection``).

Fast tests only: the heap campaign lives in
``scripts/verify_repose_injection.py`` and produces
``artifacts/repose_injection/verification.json``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.repose_injection import (
    InjectionCase,
    ReposeInjectionError,
    _floor_contact,
    _remap_history,
    _damping_ratio,
    build_pair_list,
    kinematic_bound_margin,
    simulate_injection,
    slide_threshold,
    surface_flank_angle,
)

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# case definition
# ---------------------------------------------------------------------------

def test_case_validation_rejects_bad_input():
    with pytest.raises(ReposeInjectionError):
        InjectionCase(mu=0.0)
    with pytest.raises(ReposeInjectionError):
        InjectionCase(mu_rolling=-0.1)
    with pytest.raises(ReposeInjectionError):
        InjectionCase(restitution=1.0)
    with pytest.raises(ReposeInjectionError):
        InjectionCase(n_particles=4)
    with pytest.raises(ReposeInjectionError):
        InjectionCase(source_radius=1e-3)          # inside one grain
    with pytest.raises(ReposeInjectionError):
        InjectionCase(polydispersity=0.9)


def test_timestep_resolves_smallest_contact():
    case = InjectionCase(radius=6e-3, kn=6e4)
    r = 6e-3
    m = 2500.0 * 4.0 / 3.0 * math.pi * r ** 3
    period = 2.0 * math.pi * math.sqrt(0.5 * m / 6e4)
    assert case.timestep == pytest.approx(period / case.contact_periods, rel=1e-12)
    # a polydisperse draw uses the smallest radius -> strictly smaller dt
    poly = InjectionCase(radius=6e-3, kn=6e4, polydispersity=0.25)
    assert poly.timestep < case.timestep


def test_polydisperse_radii_are_deterministic_and_bounded():
    case = InjectionCase(n_particles=32, polydispersity=0.25, seed=11)
    a = case.radii
    b = InjectionCase(**{**case.__dict__, "seed": 11}).radii
    assert torch.equal(a, b)
    assert float(a.min()) >= 6e-3 * 0.75 - 1e-15
    assert float(a.max()) <= 6e-3 * 1.25 + 1e-15
    assert torch.equal(InjectionCase(n_particles=32).radii,
                       torch.full((32,), 6.0e-3, dtype=DTYPE))


# ---------------------------------------------------------------------------
# broad phase + history bookkeeping
# ---------------------------------------------------------------------------

def test_pair_list_matches_brute_force():
    gen = torch.Generator().manual_seed(3)
    pos = torch.rand((60, 3), generator=gen, dtype=DTYPE) * 0.1
    rad = torch.full((60,), 6e-3, dtype=DTYPE)
    skin = 3e-3
    i, j = build_pair_list(pos, rad, skin)
    got = {(int(a), int(b)) for a, b in zip(i, j)}
    want = set()
    for a in range(60):
        for b in range(a + 1, 60):
            if float(torch.linalg.vector_norm(pos[a] - pos[b])) < 2 * 6e-3 + skin:
                want.add((a, b))
    assert got == want
    assert bool((i < j).all())


def test_pair_list_empty_when_far_apart():
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=DTYPE)
    rad = torch.tensor([1e-3, 1e-3], dtype=DTYPE)
    i, j = build_pair_list(pos, rad, 1e-4)
    assert len(i) == 0 and len(j) == 0


def test_history_remap_keeps_surviving_pairs_only():
    old_i = torch.tensor([0, 0, 1], dtype=torch.long)
    old_j = torch.tensor([1, 2, 2], dtype=torch.long)
    history = torch.tensor([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]], dtype=DTYPE)
    new_i = torch.tensor([0, 2], dtype=torch.long)      # (0,1) survives, (2,3) is new
    new_j = torch.tensor([1, 3], dtype=torch.long)
    out = _remap_history(history, old_i, old_j, new_i, new_j, 3)
    assert out.shape == (2, 3)
    assert out[0, 0].item() == pytest.approx(1.0)
    assert out[1, 0].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# angle estimation
# ---------------------------------------------------------------------------

def _cone_positions(angle_deg: float, n: int = 400, seed: int = 5,
                    radius: float = 6e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """Grains laid on the surface of a cone of the requested flank angle."""
    gen = torch.Generator().manual_seed(seed)
    tan_a = math.tan(math.radians(angle_deg))
    r_toe = 0.10
    u = torch.rand(n, generator=gen, dtype=DTYPE)
    r = r_toe * torch.sqrt(u)                     # area-uniform on the base
    phi = 2 * math.pi * torch.rand(n, generator=gen, dtype=DTYPE)
    z = (r_toe - r) * tan_a + radius
    pos = torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=1)
    return pos, torch.full((n,), radius, dtype=DTYPE)


def test_surface_flank_angle_recovers_analytic_cone():
    for angle in (8.0, 15.0, 25.0):
        pos, rad = _cone_positions(angle)
        out = surface_flank_angle(pos, rad)
        assert out["theta_deg"] == pytest.approx(angle, rel=0.02)
        assert out["r2"] > 0.98


def test_surface_flank_angle_rejects_flat_layer():
    gen = torch.Generator().manual_seed(7)
    pos = torch.rand((200, 3), generator=gen, dtype=DTYPE) * 0.1
    pos[:, 2] = 0.006
    rad = torch.full((200,), 6e-3, dtype=DTYPE)
    with pytest.raises(ReposeInjectionError):
        surface_flank_angle(pos, rad)


def test_surface_flank_angle_input_guards():
    pos = torch.zeros((20, 2), dtype=DTYPE)
    rad = torch.full((20,), 6e-3, dtype=DTYPE)
    with pytest.raises(ReposeInjectionError):
        surface_flank_angle(pos, rad)
    with pytest.raises(ReposeInjectionError):
        surface_flank_angle(torch.zeros((20, 3), dtype=DTYPE), rad[:5])


def test_kinematic_bound_margin():
    ok = kinematic_bound_margin(20.0, 0.5)
    assert ok["within_bound"] and ok["margin"] == pytest.approx(0.5 - math.tan(
        math.radians(20.0)))
    bad = kinematic_bound_margin(30.0, 0.5)
    assert not bad["within_bound"]


# ---------------------------------------------------------------------------
# single-grain anchors (contact law, not just code execution)
# ---------------------------------------------------------------------------

def test_sliding_onset_brackets_arctan_mu():
    """Irrotational grain: onset is analytic, ``theta = atan(mu)``."""
    out = slide_threshold(0.5, allow_rotation=False)
    assert out["bracketed"], out["rows"]
    lo, hi = out["bracket_deg"]
    assert lo < out["analytic_deg"] < hi
    assert out["travel_at_lo"] < 0.1 * out["travel_at_hi"]


def test_rolling_onset_uses_rolling_threshold_when_rotation_free():
    """Free grain with ``mu_r = 0.2`` stops rolling below ``atan(mu_r)``."""
    out = slide_threshold(0.5, mu_rolling=0.20, allow_rotation=True)
    assert out["analytic_deg"] == pytest.approx(math.degrees(math.atan(0.2)), rel=1e-9)
    assert out["bracketed"], out["rows"]


def test_ideal_rolling_moves_at_any_tilt():
    """``mu_r = 0``: an ideal sphere rolls on a 2 deg slope (no static threshold)."""
    out = slide_threshold(0.5, mu_rolling=0.0, allow_rotation=True,
                          bracket=(0.05, 0.2), dt=2e-5, hold_time=0.2)
    assert out["bracket_deg"][0] < 2.0
    assert out["rows"][-1]["moves"], out["rows"]


def test_floor_spin_term_changes_the_onset_materially():
    """Legacy convention (centre velocity only) pins a rolling grain by friction.

    This is the measured statement of the defect the module fixes: with the spin
    term dropped, a grain with ``mu = 0.5`` and ``mu_r = 0`` needs
    ``theta ~ atan(mu)`` before it moves, instead of rolling at any tilt.
    """
    legacy = slide_threshold(0.5, mu_rolling=0.0, allow_rotation=True,
                             floor_spin_term=False)
    assert legacy["bracketed"], legacy["rows"]
    assert legacy["analytic_deg"] == pytest.approx(26.565, abs=1e-3)
    fixed = slide_threshold(0.5, mu_rolling=0.0, allow_rotation=True,
                            bracket=(0.05, 0.2), dt=2e-5, hold_time=0.2)
    assert fixed["rows"][-1]["moves"]


def test_floor_contact_damps_a_bouncing_grain():
    """No energy creation: a dropped grain loses energy and rests on the plane."""
    r = 6e-3
    mass = torch.tensor([2500.0 * 4 / 3 * math.pi * r ** 3], dtype=DTYPE)
    zeta = _damping_ratio(0.05)
    pos = torch.tensor([[0.0, 0.0, 3 * r]], dtype=DTYPE)
    vel = torch.zeros((1, 3), dtype=DTYPE)
    om = torch.zeros((1, 3), dtype=DTYPE)
    tang = torch.zeros((1, 3), dtype=DTYPE)
    roll = torch.zeros((1, 3), dtype=DTYPE)
    rad = torch.tensor([r], dtype=DTYPE)
    g = torch.tensor([0.0, 0.0, -9.81], dtype=DTYPE)
    dt = 2e-5
    energies = []
    for _ in range(int(0.5 / dt)):
        f, t = _floor_contact(positions=pos, velocities=vel, omega=om, radii=rad,
                              mass=mass, kn=6e4, kt=3e4, zeta=zeta, mu_wall=0.5,
                              mu_roll_wall=0.2, tang=tang, roll=roll, dt=dt,
                              gravity_vec=g, allow_rotation=True)
        vel = vel + f * (dt / mass)[:, None]
        om = om + t * (dt / (0.4 * mass * rad ** 2))[:, None]
        pos = pos + vel * dt
        energies.append(float(0.5 * float(mass[0]) * float(vel.norm()) ** 2))
    assert energies[0] == pytest.approx(0.0)
    assert max(energies) < float(mass[0]) * 9.81 * 3 * r * 1.05
    assert float(vel.norm()) < 1e-3
    assert float(pos[0, 2]) == pytest.approx(r, rel=1e-3)


# ---------------------------------------------------------------------------
# end-to-end smoke (small heap; the production campaign is the script)
# ---------------------------------------------------------------------------

def test_small_injection_run_is_self_describing():
    case = InjectionCase(n_particles=14, injection_rate=400.0, settle_time=0.01,
                         mu=0.5, mu_rolling=0.2, max_steps=3000)
    out = simulate_injection(case)
    assert out["n_particles"] == case.n_particles
    assert out["steps"] > 0
    assert math.isfinite(out["max_speed"])
    assert out["apex_height"] > 0.0
    # a 14-grain heap is below the flank-fit floor: the driver must say so
    # instead of returning a number
    assert (out["theta_r_deg"] is None) or (out["theta_r_deg"] > 0.0)
    assert (out["angle_error"] is None) == (out["theta_r_deg"] is not None)