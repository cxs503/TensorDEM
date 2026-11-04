"""Analytic verification of the *wired-in* Cundall-Strack friction history.

`tensordem.dem3d.IceDEM3D.forces` carries a per-pair tangential displacement
``delta_t`` (SIR-PC2|F05).  Before this suite the in-solver friction path was
only covered by ``tests/test_friction_rolling.py``, whose assertions are
"the simulation ran" plus cone saturation; nothing checked the value of the
tangential force, its direction, or the meaning of the stored history when the
contact normal rotates or the contact is lost.

The reference here is the textbook Cundall-Strack update, written for a pair
with normal ``n`` and relative velocity ``v_rel = v_j - v_i``::

    delta_t <- rotate_with_contact_frame(delta_t) + (v_rel - (v_rel.n) n) dt
    F_t      = -k_t delta_t                    (on i),   +k_t delta_t (on j)
    |F_t|   <= mu |F_n|

with two consequences that a frame-aware implementation must satisfy and a
frame-naive one violates:

1. **No normal leakage.**  The friction force is a tangent-plane force, so
   ``F_t . n == 0`` for every configuration, even after the contact normal has
   rotated (the spring lives in the contact frame, not in the global frame).
2. **No inherited history.**  When a pair separates, its tangential
   displacement is forgotten.  Re-contacting must start from ``delta_t = 0``;
   otherwise a bouncing particle re-enters with a saturated friction force it
   never earned, which is neither physical nor energy consistent.

Every test is a closed-form comparison (no tuned constants): the expected
force is computed from the accumulated slip ``m * |v_t| * dt`` and the contact
stiffness, so a tolerance of 1e-12 relative is appropriate.

Run::

    python -m pytest tests/test_dem3d_tangential_frame_transport.py -q
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D

DTYPE = torch.float64
RADIUS = 0.025
KN = 5000.0


def build_pair(*, mu: float = 0.3, overlap: float = 2.0e-3) -> IceDEM3D:
    """Two overlapping, unbonded particles on the x axis, isolated from the rest.

    The remaining lattice nodes are pushed far away and every bond is deleted,
    so ``forces()`` on nodes 0 and 1 sees exactly one particle-particle contact
    and nothing else (no bond force, no drag, no tool, no platen).
    """
    cfg = DEM3DConfig(
        nx=2,
        ny=2,
        nz=2,
        radius=RADIUS,
        density=917.0,
        contact_stiffness=KN,
        contact_damping=0.0,
        contact_damping_model="absolute",
        drag=0.0,
        restitution_coefficient=1.0,
        friction_coefficient=mu,
        rolling_resistance_coefficient=0.0,
        fix_x_edges=False,
        fix_bottom=False,
        tool_gap=1.0,
    )
    sim = IceDEM3D(cfg)
    sim.positions[:] = 0.0
    for idx in range(len(sim.positions)):
        sim.positions[idx] = torch.tensor(
            [10.0 + 0.5 * idx, 10.0, 10.0], dtype=sim.dtype, device=sim.device
        )
    sim.velocities[:] = 0.0
    # Contact pair on the x axis with a prescribed centre distance.
    gap = 2.0 * RADIUS - overlap
    sim.positions[0] = torch.tensor([-0.5 * gap, 0.0, 0.0], dtype=sim.dtype, device=sim.device)
    sim.positions[1] = torch.tensor([+0.5 * gap, 0.0, 0.0], dtype=sim.dtype, device=sim.device)
    # Kill every bond: the topology must not contribute to the force balance.
    sim.alive[:] = False
    sim.bonded[:] = False
    return sim


def contact_normal(sim: IceDEM3D) -> torch.Tensor:
    delta = sim.positions[1] - sim.positions[0]
    return delta / torch.linalg.vector_norm(delta)


def normal_force(sim: IceDEM3D, overlap: float = 2.0e-3) -> float:
    """|F_n| for the frozen pair: linear penalty, zero normal relative velocity."""
    return KN * overlap


def place(sim: IceDEM3D, p0, p1) -> None:
    sim.positions[0] = torch.tensor(p0, dtype=sim.dtype, device=sim.device)
    sim.positions[1] = torch.tensor(p1, dtype=sim.dtype, device=sim.device)


def velocity(sim: IceDEM3D, v0, v1) -> None:
    sim.velocities[0] = torch.tensor(v0, dtype=sim.dtype, device=sim.device)
    sim.velocities[1] = torch.tensor(v1, dtype=sim.dtype, device=sim.device)


def accumulate_slip(sim: IceDEM3D, *, v_t: float, steps: int) -> float:
    """Hold the geometry fixed, give node 1 a +y slip velocity, evaluate forces.

    Returns the accumulated tangential displacement magnitude ``steps * v_t * dt``.
    """
    velocity(sim, (0.0, 0.0, 0.0), (0.0, v_t, 0.0))
    for _ in range(steps):
        sim.forces(update_fracture=False, update_history=True)
    # Freeze the relative velocity so later evaluations do not add slip.
    velocity(sim, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    return steps * abs(v_t) * sim.dt


def contact_force_on(sim: IceDEM3D, index: int) -> torch.Tensor:
    """Contact-only force on a node (bonds deleted, so this is the friction part)."""
    force, _ = sim.forces(update_fracture=False, update_history=False)
    return force[index].clone()


# ---------------------------------------------------------------------------
# 1. Value of the tangential spring before saturation (closed form)
# ---------------------------------------------------------------------------
def test_tangential_force_matches_analytic_ramp() -> None:
    sim = build_pair(mu=1.0, overlap=2.0e-3)
    steps, v_t = 4, 1.0e-3
    slip = accumulate_slip(sim, v_t=v_t, steps=steps)

    force = contact_force_on(sim, 0)
    fn = normal_force(sim)
    # The normal part is the repulsive penalty force -|F_n| n (n points i -> j) and
    # the friction part is +k_t delta_t, because delta_t accumulates v_j - v_i along +y.
    torch.testing.assert_close(
        force, torch.tensor([-fn, KN * slip, 0.0], dtype=DTYPE), rtol=1e-12, atol=1e-12
    )
    # Equal and opposite on the other node: the spring conserves momentum.
    torch.testing.assert_close(
        force, -contact_force_on(sim, 1), rtol=0.0, atol=0.0
    )
    # Well inside the cone for mu = 1, |F_n| = 10 N.
    assert KN * slip < 1.0 * normal_force(sim)


# ---------------------------------------------------------------------------
# 2. Coulomb saturation equals mu * |F_n| exactly
# ---------------------------------------------------------------------------
def test_coulomb_saturation_equals_mu_times_normal_force() -> None:
    mu = 0.3
    sim = build_pair(mu=mu, overlap=2.0e-3)
    slip = accumulate_slip(sim, v_t=5.0, steps=4)
    assert KN * slip > mu * normal_force(sim)  # the ramp must overshoot the cone

    force = contact_force_on(sim, 0)
    fn = normal_force(sim)
    tangential = torch.tensor([0.0, force[1].item(), 0.0], dtype=DTYPE)
    assert torch.linalg.vector_norm(tangential).item() == pytest.approx(mu * fn, rel=1e-12)
    # The saturated force still points along the slip history (+y), not along n (+x).
    torch.testing.assert_close(
        force, torch.tensor([-fn, mu * fn, 0.0], dtype=DTYPE), rtol=1e-12, atol=1e-12
    )


# ---------------------------------------------------------------------------
# 3. The friction spring lives in the contact frame: no normal leakage
# ---------------------------------------------------------------------------
def test_friction_force_is_tangential_after_rigid_contact_rotation() -> None:
    """Rigidly rotate an unbonded pair with frozen slip history.

    Rotating both centres about their midpoint keeps the overlap (hence
    ``F_n``) and the *material* slip history unchanged, so the stored spring
    displacement must be transported with the contact frame: the force has to
    stay in the tangent plane and keep its magnitude.  A global-frame
    implementation instead leaks a normal component and shrinks the force.
    """
    sim = build_pair(mu=1.0, overlap=2.0e-3)
    slip = accumulate_slip(sim, v_t=1.0e-3, steps=4)
    magnitude = KN * slip

    theta = math.radians(45.0)
    c, s = math.cos(theta), math.sin(theta)
    p0 = (-0.5 * (2.0 * RADIUS - 2.0e-3), 0.0, 0.0)
    p1 = (+0.5 * (2.0 * RADIUS - 2.0e-3), 0.0, 0.0)
    place(sim, (c * p0[0], s * p0[0], 0.0), (c * p1[0], s * p1[0], 0.0))

    n = contact_normal(sim)
    force = contact_force_on(sim, 0)

    fn = normal_force(sim)
    # (a) no normal leakage: the normal part is exactly the repulsive penalty force
    normal_component = torch.dot(force, n).item()
    assert normal_component == pytest.approx(-fn, rel=1e-12), (
        f"force normal component {normal_component:.6e} N, expected {-fn:.6e} N"
    )
    # (b) magnitude preserved: the spring is embedded in the frame
    friction = force + fn * n  # remove the normal part
    assert torch.linalg.vector_norm(friction).item() == pytest.approx(magnitude, rel=1e-12)
    # (c) direction is the rotated slip direction
    expected = magnitude * torch.tensor([-s, c, 0.0], dtype=DTYPE)
    torch.testing.assert_close(friction, expected, rtol=1e-12, atol=1e-12)


def test_normal_leakage_absent_for_oblique_rotation() -> None:
    """Same invariant for a rotation that is not about a coordinate axis."""
    sim = build_pair(mu=1.0, overlap=2.0e-3)
    accumulate_slip(sim, v_t=1.0e-3, steps=3)
    p0 = (-0.5 * (2.0 * RADIUS - 2.0e-3), 0.0, 0.0)
    p1 = (+0.5 * (2.0 * RADIUS - 2.0e-3), 0.0, 0.0)
    # Tilt the pair so the normal picks up a z component.
    place(sim, (p0[0] * 0.6, 0.0, p0[0] * 0.8), (p1[0] * 0.6, 0.0, p1[0] * 0.8))
    n = contact_normal(sim)
    fn = normal_force(sim)
    for _ in range(3):  # a few evaluations, geometry frozen
        force = contact_force_on(sim, 0)
        leaked = torch.dot(force, n).item() + fn
        assert abs(leaked) <= 1e-12, f"friction leaked {leaked:.3e} N along the normal"


# ---------------------------------------------------------------------------
# 4. Contact loss forgets the friction history
# ---------------------------------------------------------------------------
def test_friction_history_resets_after_contact_loss() -> None:
    mu = 0.3
    sim = build_pair(mu=mu, overlap=2.0e-3)
    accumulate_slip(sim, v_t=5.0, steps=4)
    saturated = contact_force_on(sim, 0)
    fn = normal_force(sim)
    saturated_friction = saturated + fn * contact_normal(sim)
    assert torch.linalg.vector_norm(saturated_friction).item() == pytest.approx(
        mu * fn, rel=1e-12
    )

    # Separate the pair well beyond the contact diameter; the pair leaves the
    # candidate list, so this evaluation must drop the stored slip history.
    place(sim, (-0.5 * 3.0 * RADIUS, 0.0, 0.0), (+0.5 * 3.0 * RADIUS, 0.0, 0.0))
    sim.forces(update_fracture=False, update_history=True)

    # Re-contact at the original overlap with no relative velocity.
    place(sim, (-0.5 * (2.0 * RADIUS - 2.0e-3), 0.0, 0.0),
          (+0.5 * (2.0 * RADIUS - 2.0e-3), 0.0, 0.0))
    force, _ = sim.forces(update_fracture=False, update_history=True)
    fn = normal_force(sim)
    tangential = force[0] + fn * contact_normal(sim)  # strip the normal part
    assert torch.linalg.vector_norm(tangential).item() <= 1e-12, (
        "a re-formed contact inherited "
        f"{torch.linalg.vector_norm(tangential).item():.3e} N of tangential force"
    )
    assert torch.dot(force[0], contact_normal(sim)).item() == pytest.approx(-fn, rel=1e-12)


def test_friction_history_pruned_when_all_contacts_vanish() -> None:
    sim = build_pair(mu=0.4, overlap=2.0e-3)
    accumulate_slip(sim, v_t=1.0, steps=2)
    assert sim._tangential_state  # history exists
    for idx in range(len(sim.positions)):
        sim.positions[idx] = torch.tensor(
            [10.0 + 0.5 * idx, 20.0, 10.0], dtype=sim.dtype, device=sim.device
        )
    sim.forces(update_fracture=False, update_history=True)
    assert not sim._tangential_state, "stale friction history survived contact loss"


# ---------------------------------------------------------------------------
# 5. Probing must not advance the friction history (frame transport path)
# ---------------------------------------------------------------------------
def test_update_history_false_is_side_effect_free() -> None:
    sim = build_pair(mu=0.5, overlap=2.0e-3)
    accumulate_slip(sim, v_t=1.0e-2, steps=2)
    snapshot = {k: (v[0].clone(), v[1].clone()) for k, v in sim._tangential_state.items()}
    velocity(sim, (0.0, 0.0, 0.0), (0.0, 3.0e-2, 0.0))
    for _ in range(3):
        sim.forces(update_fracture=False, update_history=False)
    assert sim._tangential_state.keys() == snapshot.keys()
    for key, (disp, normal) in snapshot.items():
        torch.testing.assert_close(sim._tangential_state[key][0], disp, rtol=0.0, atol=0.0)
        torch.testing.assert_close(sim._tangential_state[key][1], normal, rtol=0.0, atol=0.0)