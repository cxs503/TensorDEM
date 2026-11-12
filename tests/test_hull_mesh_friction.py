"""Tests for the hull-path friction / rolling law (:mod:`hull_mesh_friction`).

The gates scored against closed forms live in
``scripts/verify_hull_mesh_friction.py`` (numbers in
``artifacts/hull_mesh_friction/verification.json``); the tests here pin the
*contract* and the cheap analytic anchors so a regression fails fast in CI.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.hull_mesh_contact import make_flat_mesh, mesh_contact_forces
from tensordem.hull_mesh_friction import (
    HullContactResult,
    HullFrictionConfig,
    HullFrictionState,
    integrate_sphere_on_mesh,
    mesh_contact_forces_frictional,
)
from tensordem.rolling_resistance import (
    critical_incline_angle_deg,
    incline_sliding_acceleration,
)

F64 = torch.float64
G = 9.81
R = 0.05
MASS = 0.52
K_N = 2.0e5
K_T = 2.0e5


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def rot_y(theta_deg: float) -> "object":
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=F64)


def tilted_plane(theta_deg: float, size: float = 3.0, n: int = 4):
    verts, faces = make_flat_mesh(size=size, normal="z", nx=n, ny=n)
    return (rot_y(theta_deg) @ verts.T).T.contiguous(), faces


def place_on_plane(theta_deg: float, radius: float, overlap: float):
    n = rot_y(theta_deg) @ torch.tensor([0.0, 0.0, 1.0], dtype=F64)
    return (radius - overlap) * n


def downhill(theta_deg: float) -> torch.Tensor:
    n = rot_y(theta_deg) @ torch.tensor([0.0, 0.0, 1.0], dtype=F64)
    g = torch.tensor([0.0, 0.0, -G], dtype=F64)
    t = g - (g * n).sum() * n
    return t / torch.linalg.vector_norm(t)


# ---------------------------------------------------------------------------
# configuration / validation
# ---------------------------------------------------------------------------

def test_config_validation():
    with pytest.raises(ValueError):
        HullFrictionConfig(mu=-0.1)
    with pytest.raises(ValueError):
        HullFrictionConfig(mu=0.5, k_t=0.0)
    with pytest.raises(ValueError):
        HullFrictionConfig(mu_r=-1.0)
    with pytest.raises(ValueError):
        HullFrictionConfig(mu_r=0.1, rolling_model="epsd")   # k_r missing
    cfg = HullFrictionConfig(mu=0.4)
    assert cfg.enabled and cfg.friction_enabled
    off = HullFrictionConfig()
    assert not off.enabled and not off.friction_enabled


def test_dt_required_with_friction():
    verts, faces = make_flat_mesh(size=1.0)
    x = torch.tensor([[0.0, 0.0, -0.0499]], dtype=F64)
    with pytest.raises(ValueError):
        mesh_contact_forces_frictional(
            x, torch.zeros(1, 3, dtype=F64), torch.zeros(1, 3, dtype=F64),
            verts, faces, torch.zeros(3, dtype=F64),
            particle_radius=R, config=HullFrictionConfig(mu=0.5),
            state=HullFrictionState(), stiffness=K_N,
        )


def test_shape_validation():
    verts, faces = make_flat_mesh(size=1.0)
    x = torch.tensor([[0.0, 0.0, -0.0499]], dtype=F64)
    with pytest.raises(ValueError):
        mesh_contact_forces_frictional(
            x, torch.zeros(1, 2, dtype=F64), torch.zeros(1, 3, dtype=F64),
            verts, faces, torch.zeros(3, dtype=F64),
            particle_radius=R, config=HullFrictionConfig(), dt=1e-4, stiffness=K_N,
        )


# ---------------------------------------------------------------------------
# the normal limb must be untouched by the friction layer
# ---------------------------------------------------------------------------

def test_mu_zero_reproduces_frictionless_kernel():
    verts, faces = tilted_plane(12.0)
    x = place_on_plane(12.0, R, 3.0e-4).reshape(1, 3)
    v = torch.tensor([[0.2, -0.1, -0.3]], dtype=F64)
    hv = torch.zeros(3, dtype=F64)
    f0, r0 = mesh_contact_forces(x, v, verts, faces, hv,
                                 particle_radius=R, stiffness=K_N, damping=7.0)
    res = mesh_contact_forces_frictional(
        x, v, torch.zeros(1, 3, dtype=F64), verts, faces, hv,
        particle_radius=R, config=HullFrictionConfig(),
        state=HullFrictionState(), dt=1e-4, stiffness=K_N, damping=7.0,
    )
    assert torch.allclose(f0, res.forces, atol=1e-12, rtol=0)
    assert torch.allclose(r0, res.reaction_force, atol=1e-12, rtol=0)
    assert res.tangential_force.abs().max() == 0.0


def test_friction_does_not_change_normal_ordinate():
    verts, faces = tilted_plane(12.0)
    x = place_on_plane(12.0, R, 3.0e-4).reshape(1, 3)
    v = torch.tensor([[0.2, -0.1, -0.3]], dtype=F64)
    hv = torch.zeros(3, dtype=F64)
    f0, _ = mesh_contact_forces(x, v, verts, faces, hv,
                                particle_radius=R, stiffness=K_N, damping=7.0)
    res = mesh_contact_forces_frictional(
        x, v, torch.zeros(1, 3, dtype=F64), verts, faces, hv,
        particle_radius=R, config=HullFrictionConfig(mu=0.6, k_t=K_T),
        state=HullFrictionState(), dt=1e-4, stiffness=K_N, damping=7.0,
    )
    n = res.contacts["normal"][0]
    assert abs(float(f0[0] @ n) - float(res.normal_force[0])) < 1e-12


# ---------------------------------------------------------------------------
# Coulomb cone, direction, saturation
# ---------------------------------------------------------------------------

def test_cone_saturation_and_direction():
    theta, mu = 15.0, 0.4
    verts, faces = tilted_plane(theta)
    x = place_on_plane(theta, R, 2.0e-4).reshape(1, 3)
    t_hat = downhill(theta)
    v = t_hat.reshape(1, 3) * 1.0
    state = HullFrictionState()
    cfg = HullFrictionConfig(mu=mu, k_t=K_T)
    for _ in range(400):
        res = mesh_contact_forces_frictional(
            x, v, torch.zeros(1, 3, dtype=F64), verts, faces,
            torch.zeros(3, dtype=F64),
            particle_radius=R, config=cfg, state=state, dt=1e-4, stiffness=K_N,
        )
    fn = float(res.normal_force[0])
    ft = res.tangential_force[0]
    mag = float(torch.linalg.vector_norm(ft))
    assert abs(mag - mu * fn) / (mu * fn) < 1e-12
    assert bool(res.sliding[0])
    cosang = float((-ft) @ t_hat) / mag
    assert math.degrees(math.acos(max(-1.0, min(1.0, cosang)))) < 1e-6
    # the tangential force must lie in the facet plane
    n = res.contacts["normal"][0]
    assert abs(float(ft @ n)) < 1e-15 * max(mag, 1.0)


def test_sticking_below_cone():
    """A tiny slip velocity must not saturate the cone (F_t = k_t delta)."""
    theta, mu = 5.0, 0.8
    verts, faces = tilted_plane(theta)
    x = place_on_plane(theta, R, 2.0e-4).reshape(1, 3)
    t_hat = downhill(theta)
    v = (1e-4 * t_hat).reshape(1, 3)
    state = HullFrictionState()
    cfg = HullFrictionConfig(mu=mu, k_t=K_T)
    cfg_ = 0
    for _ in range(10):
        res = mesh_contact_forces_frictional(
            x, v, torch.zeros(1, 3, dtype=F64), verts, faces,
            torch.zeros(3, dtype=F64),
            particle_radius=R, config=cfg, state=state, dt=1e-5, stiffness=K_N,
        )
    delta = v.reshape(3) * 1e-5 * 10
    expected = K_T * float(torch.linalg.vector_norm(delta))
    mag = float(torch.linalg.vector_norm(res.tangential_force[0]))
    assert not bool(res.sliding[0])
    assert abs(mag - expected) / expected < 1e-9


def test_cone_radius_matches_saturation_displacement():
    cfg = HullFrictionConfig(mu=0.5, k_t=K_T)
    theta = 15.0
    verts, faces = tilted_plane(theta)
    x = place_on_plane(theta, R, 2.0e-4).reshape(1, 3)
    v = downhill(theta).reshape(1, 3) * 2.0
    state = HullFrictionState()
    for _ in range(200):
        res = mesh_contact_forces_frictional(
            x, v, torch.zeros(1, 3, dtype=F64), verts, faces,
            torch.zeros(3, dtype=F64),
            particle_radius=R, config=cfg, state=state, dt=1e-4, stiffness=K_N,
        )
    delta = state.tangential[(0, int(res.contacts["triangle"][0]))]
    cap = cfg.mu * float(res.normal_force[0]) / cfg.k_t
    assert abs(float(torch.linalg.vector_norm(delta)) - cap) < 1e-12 * cap


# ---------------------------------------------------------------------------
# analytic incline anchors
# ---------------------------------------------------------------------------

def test_sliding_acceleration_matches_closed_form():
    mu = 0.35
    theta = 20.0
    dt = 2e-4
    verts, faces = tilted_plane(theta, size=6.0, n=6)
    overlap = MASS * G * math.cos(math.radians(theta)) / K_N
    x0 = place_on_plane(theta, R, overlap)
    n_steps = int(0.15 / dt)
    run = integrate_sphere_on_mesh(
        x0, torch.zeros(3, dtype=F64), torch.zeros(3, dtype=F64),
        radius=R, mass=MASS, inertia=None, vertices=verts, faces=faces,
        config=HullFrictionConfig(mu=mu, k_t=K_T), dt=dt, n_steps=n_steps,
        stiffness=K_N, damping=0.0, record_every=n_steps // 60,
    )
    import numpy as np
    t = run["t"].numpy()
    vx = (run["v"] @ downhill(theta)).numpy()
    lo = len(t) // 3
    slope = float(np.polyfit(t[lo:], vx[lo:], 1)[0])
    ref = incline_sliding_acceleration(theta, mu, gravity=G)
    assert abs(slope - ref) / abs(ref) < 0.01


def test_static_hold_above_and_below_threshold():
    """Rest is held iff tan(theta) <= mu (single-floe check of G4)."""
    mu = 0.45
    theta_c = math.degrees(math.atan(mu))
    dt = 2e-4
    n_steps = int(0.20 / dt)

    def drift(theta):
        verts, faces = tilted_plane(theta, size=6.0, n=6)
        overlap = MASS * G * math.cos(math.radians(theta)) / K_N
        x0 = place_on_plane(theta, R, overlap)
        run = integrate_sphere_on_mesh(
            x0, torch.zeros(3, dtype=F64), torch.zeros(3, dtype=F64),
            radius=R, mass=MASS, inertia=None, vertices=verts, faces=faces,
            config=HullFrictionConfig(mu=mu, k_t=K_T), dt=dt, n_steps=n_steps,
            stiffness=K_N, damping=0.0, record_every=n_steps // 40,
        )
        x = run["x"].numpy()
        t_hat = downhill(theta).numpy()
        proj = x @ t_hat
        return float(proj[-1] - proj[len(proj) // 2])

    below = drift(theta_c - 8.0)
    above = drift(theta_c + 8.0)
    assert abs(below) < 1e-4            # held (elastic oscillation only)
    assert above > 1e-2                 # slides a measurable distance
    assert above > 100 * max(abs(below), 1e-12)


def test_rolling_threshold_reference_is_used():
    """The rolling branch must be sign-correct: it slows the descent."""
    mu_r = 0.04
    theta = 16.0
    dt = 1e-4
    n_steps = int(0.08 / dt)
    verts, faces = tilted_plane(theta, size=6.0, n=6)
    overlap = MASS * G * math.cos(math.radians(theta)) / K_N
    x0 = place_on_plane(theta, R, overlap)
    inertia = 0.4 * MASS * R * R
    run = integrate_sphere_on_mesh(
        x0, torch.zeros(3, dtype=F64), torch.zeros(3, dtype=F64),
        radius=R, mass=MASS, inertia=inertia, vertices=verts, faces=faces,
        config=HullFrictionConfig(mu=2.0, k_t=K_T, mu_r=mu_r),
        dt=dt, n_steps=n_steps, stiffness=K_N, damping=0.0,
        record_every=n_steps // 40,
    )
    assert math.sin(math.radians(theta)) > mu_r  # above the critical angle...
    v_end = float(run["final"]["v"][0])
    assert v_end > 0.0                            # ...so it accelerates downhill
    assert critical_incline_angle_deg(mu_r) > 0.0


# ---------------------------------------------------------------------------
# state hygiene
# ---------------------------------------------------------------------------

def test_state_reset_on_contact_loss():
    verts, faces = make_flat_mesh(size=1.0)
    cfg = HullFrictionConfig(mu=0.5)
    state = HullFrictionState()
    x = torch.tensor([[0.0, 0.0, -0.0499]], dtype=F64)
    res = mesh_contact_forces_frictional(
        x, torch.tensor([[1.0, 0.0, 0.0]], dtype=F64), torch.zeros(1, 3, dtype=F64),
        verts, faces, torch.zeros(3, dtype=F64),
        particle_radius=R, config=cfg, state=state, dt=1e-4, stiffness=K_N,
    )
    assert len(state) == 1
    far = torch.tensor([[0.0, 0.0, 5.0]], dtype=F64)
    res = mesh_contact_forces_frictional(
        far, torch.zeros(1, 3, dtype=F64), torch.zeros(1, 3, dtype=F64),
        verts, faces, torch.zeros(3, dtype=F64),
        particle_radius=R, config=cfg, state=state, dt=1e-4, stiffness=K_N,
    )
    assert len(state) == 0
    assert res.forces.abs().max() == 0.0


def test_state_copy_is_independent():
    state = HullFrictionState()
    state.tangential[(0, 0)] = torch.tensor([1.0, 0.0, 0.0], dtype=F64)
    state.previous_triangle[0] = 0
    clone = state.copy()
    clone.tangential[(0, 0)] += 1.0
    assert torch.allclose(state.tangential[(0, 0)], torch.tensor([1.0, 0.0, 0.0], dtype=F64))


def test_no_slip_means_no_friction():
    verts, faces = tilted_plane(10.0)
    x = place_on_plane(10.0, R, 2e-4).reshape(1, 3)
    v = torch.tensor([[0.4, 0.1, -0.2]], dtype=F64)
    res = mesh_contact_forces_frictional(
        x, v, torch.zeros(1, 3, dtype=F64), verts, faces, v.reshape(3),
        particle_radius=R, config=HullFrictionConfig(mu=0.7),
        state=HullFrictionState(), dt=1e-4, stiffness=K_N,
    )
    assert float(torch.linalg.vector_norm(res.tangential_force[0])) < 1e-12
    assert float(torch.linalg.vector_norm(res.slip_velocity[0])) < 1e-12


def test_hull_rotation_enters_slip_velocity():
    verts, faces = tilted_plane(10.0)
    x = place_on_plane(10.0, R, 2e-4).reshape(1, 3)
    hw = torch.tensor([0.0, 0.0, 0.3], dtype=F64)
    hv = torch.tensor([0.2, 0.0, 0.0], dtype=F64)
    res = mesh_contact_forces_frictional(
        x, torch.zeros(1, 3, dtype=F64), torch.zeros(1, 3, dtype=F64),
        verts, faces, hv,
        particle_radius=R, config=HullFrictionConfig(mu=0.5),
        state=HullFrictionState(), dt=1e-4, stiffness=K_N,
        hull_angular_velocity=hw,
    )
    c = res.contacts["closest"][0]
    n = res.contacts["normal"][0]
    v_pt = hv + torch.linalg.cross(hw, c)
    v_slip = -(v_pt - (v_pt * n).sum() * n)
    assert torch.allclose(res.slip_velocity[0], v_slip, atol=1e-12, rtol=0)


def test_reaction_is_equal_and_opposite():
    verts, faces = tilted_plane(15.0)
    x = place_on_plane(15.0, R, 2e-4).reshape(1, 3)
    res = mesh_contact_forces_frictional(
        x, torch.tensor([[0.1, 0.0, -0.2]], dtype=F64), torch.zeros(1, 3, dtype=F64),
        verts, faces, torch.zeros(3, dtype=F64),
        particle_radius=R, config=HullFrictionConfig(mu=0.4),
        state=HullFrictionState(), dt=1e-4, stiffness=K_N,
    )
    assert float((res.forces.sum(dim=0) + res.reaction_force).abs().max()) < 1e-12


def test_torque_is_lever_cross_tangential():
    verts, faces = tilted_plane(15.0)
    x = place_on_plane(15.0, R, 2e-4).reshape(1, 3)
    res = mesh_contact_forces_frictional(
        x, downhill(15.0).reshape(1, 3), torch.zeros(1, 3, dtype=F64),
        verts, faces, torch.zeros(3, dtype=F64),
        particle_radius=R, config=HullFrictionConfig(mu=0.4),
        state=HullFrictionState(), dt=1e-4, stiffness=K_N,
    )
    lever = res.contacts["closest"][0] - x[0]
    expected = torch.linalg.cross(lever, res.tangential_force[0])
    assert torch.allclose(res.torques[0], expected, atol=1e-12, rtol=0)


def test_empty_contact_returns_zeros():
    verts, faces = make_flat_mesh(size=1.0)
    x = torch.tensor([[0.0, 0.0, 5.0]], dtype=F64)
    res = mesh_contact_forces_frictional(
        x, torch.zeros(1, 3, dtype=F64), torch.zeros(1, 3, dtype=F64),
        verts, faces, torch.zeros(3, dtype=F64),
        particle_radius=R, config=HullFrictionConfig(mu=0.5),
        state=HullFrictionState(), dt=1e-4, stiffness=K_N,
    )
    assert isinstance(res, HullContactResult)
    assert res.forces.abs().max() == 0.0
    assert res.normal_force.numel() == 0


def test_energy_ledger_closed_and_dissipative():
    theta, mu = 18.0, 0.4
    dt = 2e-4
    verts, faces = tilted_plane(theta, size=6.0, n=6)
    overlap = MASS * G * math.cos(math.radians(theta)) / K_N
    x0 = place_on_plane(theta, R, overlap)
    n_steps = int(0.10 / dt)
    run = integrate_sphere_on_mesh(
        x0, torch.zeros(3, dtype=F64), torch.zeros(3, dtype=F64),
        radius=R, mass=MASS, inertia=None, vertices=verts, faces=faces,
        config=HullFrictionConfig(mu=mu, k_t=K_T), dt=dt, n_steps=n_steps,
        stiffness=K_N, damping=0.0, record_every=n_steps // 100,
    )
    ledger = run["ledger_total"]
    # normalise by the gravitational energy scale, not by the (near-zero) mean
    spread = float((ledger.max() - ledger.min()) / (MASS * G * R))
    assert spread < 1e-2
    diss = run["dissipated"]
    # NOTE: the per-step frictional work is *not* monotone while the contact
    # sticks -- the tangential spring hands energy back (that is why the spring
    # term is part of the ledger).  What must hold is that the closed ledger is
    # constant and that the net frictional work is dissipative.
    assert float(diss[-1]) > 0.0
    assert float(diss[-1]) < 0.1 * MASS * G * R    # bounded by the drop in height