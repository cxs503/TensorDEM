"""Restitution calibration of the 3-D hull-mesh penalty contact law (m16922).

The hull kernel evaluates ``F_n = max(k*overlap - c*vn, 0)`` -- a
*compressive-only* spring-dashpot (src/tensordem/hull_mesh_contact.py).  These
tests pin down the three facts the calibration rests on:

1. the classical closed form ``e = exp(-pi zeta / sqrt(1-zeta^2))`` is **not**
   the restitution of that clamped law (the gap is measured, not assumed);
2. the clamped-model inverse (:func:`tensordem.hull_mesh_contact.
   penalty_damping_for_restitution`) reproduces a requested restitution when the
   shipped kernel is driven directly, at ``dt = t_c/400``, to better than 3 %;
3. the penalties are non-vacuous: the classical inverse misses by more than 3 %
   on the same rig, and the linear speed-independence / elastic-limit
   invariants hold.

Acceptance tolerance: 3 % on the measured restitution (platform accuracy
standard), 0.2 % on the elastic limit.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.contact_restitution import clamped_restitution_ratio
from tensordem.hull_mesh_contact import (
    make_flat_mesh,
    mesh_contact_forces,
    penalty_damping_for_restitution,
)

K = 1.0e6       # N/m
M = 1.0         # kg
R = 0.05        # m
T_C = math.pi * math.sqrt(M / K)
V0 = 0.02       # m/s


def _bounce(c_damp: float, *, dt: float = T_C / 400.0, v_in: float = V0,
            gap: float = 1e-4) -> float:
    """1-DOF impact on the flat mesh, velocity Verlet, returns |v_out|/v_in."""
    torch.set_num_threads(1)
    verts, faces = make_flat_mesh(size=2.0)
    hv = torch.zeros(3, dtype=torch.float64)
    z = torch.tensor([[0.0, 0.0, R + gap]], dtype=torch.float64)
    vel = torch.tensor([[0.0, 0.0, -v_in]], dtype=torch.float64)
    steps = int((gap / v_in + 2.0 * T_C) / dt) + 10
    contacted = False
    for _ in range(steps):
        f, _r = mesh_contact_forces(z, vel, verts, faces, hv,
                                    particle_radius=R, stiffness=K, damping=c_damp)
        if float(torch.linalg.vector_norm(f)) > 0.0:
            contacted = True
        vel = vel + 0.5 * (f / M) * dt
        z = z + vel * dt
        f2, _r = mesh_contact_forces(z, vel, verts, faces, hv,
                                     particle_radius=R, stiffness=K, damping=c_damp)
        vel = vel + 0.5 * (f2 / M) * dt
        if contacted and float(vel[0, 2]) > 0.0 and float(torch.linalg.vector_norm(f2)) == 0.0:
            return float(vel[0, 2]) / v_in
    raise AssertionError("impact did not separate within the step budget")


# ---------------------------------------------------------------------------
# 1. The classical closed form is not the target of this force law
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zeta", [0.1, 0.2, 0.3, 0.4])
def test_clamped_law_exceeds_classical_restitution(zeta: float):
    """Clamping raises the restitution; the gap grows with damping."""
    c = 2.0 * zeta * math.sqrt(K * M)
    e_repo = clamped_restitution_ratio(c, K, M, v_in=V0)["e_reference"]
    e_classic = math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta ** 2))
    assert e_repo > e_classic, (e_repo, e_classic)
    if zeta >= 0.3:
        assert (e_repo - e_classic) / e_classic > 0.15


def test_classical_mapping_is_wrong_on_the_hull_rig():
    """Asking for e = 0.4 through the classical inverse overshoots by > 15 %."""
    from tensordem.contact_tangential import restitution_to_damping

    c_classic = restitution_to_damping(0.4, K, M)
    e_meas = _bounce(c_classic)
    assert e_meas > 0.4 * 1.15, e_meas


# ---------------------------------------------------------------------------
# 2. The clamped inverse reproduces the requested restitution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("e_target", [0.5, 0.4])
def test_closed_loop_restitution_within_3_percent(e_target: float):
    c = penalty_damping_for_restitution(e_target, K, M)
    e_meas = _bounce(c)
    assert abs(e_meas - e_target) / e_target < 0.03, (e_target, e_meas)


def test_inverse_mapping_is_monotone_and_validated():
    lo = penalty_damping_for_restitution(0.8, K, M)
    hi = penalty_damping_for_restitution(0.3, K, M)
    assert hi > lo > 0.0
    with pytest.raises(ValueError):
        penalty_damping_for_restitution(0.0, K, M)
    with pytest.raises(ValueError):
        penalty_damping_for_restitution(0.5, -1.0, M)


# ---------------------------------------------------------------------------
# 3. Invariants of the rig itself
# ---------------------------------------------------------------------------

def test_elastic_limit():
    """No damping -> no dissipation."""
    e = _bounce(0.0)
    assert abs(e - 1.0) < 2e-3, e


def test_speed_independence_of_linear_law():
    c = penalty_damping_for_restitution(0.5, K, M)
    e_fast = _bounce(c, v_in=0.02)
    e_slow = _bounce(c, v_in=0.005)
    assert abs(e_fast - e_slow) / e_fast < 0.01, (e_fast, e_slow)


def test_kernel_force_matches_analytic_penalty_law():
    """F = k*overlap - c*vn is reproduced exactly by the shipped kernel."""
    torch.set_num_threads(1)
    verts, faces = make_flat_mesh(size=2.0)
    hv = torch.zeros(3, dtype=torch.float64)
    c = 500.0
    z = torch.tensor([[0.0, 0.0, R - 2e-3]], dtype=torch.float64)   # 2 mm overlap
    v = torch.tensor([[0.0, 0.0, -0.01]], dtype=torch.float64)      # approaching
    f, _r = mesh_contact_forces(z, v, verts, faces, hv,
                                particle_radius=R, stiffness=K, damping=c)
    expected = K * 2e-3 + c * 0.01
    assert float(f[0, 2]) == pytest.approx(expected, rel=1e-12)