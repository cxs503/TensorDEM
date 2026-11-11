"""Verification suite for :mod:`tensordem.dem_vectorized`.

Why this file exists
--------------------
``tests/test_dem_vectorized_100k.py`` (11 tests, green) is named after this engine but
exercises ``tensordem.dem3d.IceDEM3D`` -- it imports nothing from ``dem_vectorized``.
The GPU-100K report (``docs/dem_gpu_100k_report.md``) therefore credited the vectorized
engine with a test suite that never touched it, and the module shipped with **zero**
physics verification.  This file closes that gap with assertions on invariants and
analytic laws, not on "the code ran":

* neighbour search: the cell list must reproduce the O(N^2) pair set exactly
* contact laws: linear ``k*delta`` and Hertz ``(4/3) E* sqrt(R*) delta^1.5``
* third law / momentum: internal contact forces must cancel to machine precision
* integrator: free fall and elastic collisions conserve mechanical energy
* restitution: the configured ``restitution_coefficient`` must be the *realised* one
  (it was not: configured 0.30 was realised as 0.40, +33 %)
* silent-parameter guards: rolling resistance / conflicting damping / unknown contact
  model must raise instead of being ignored

Evidence: ``docs/DEM_VECTORIZED_VERIFICATION.md``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.dem_vectorized import (
    ReferenceDEM,
    VecDEM,
    VecDEMConfig,
    compute_contact_forces,
    find_neighbor_pairs_vectorized,
    realised_restitution_floor,
    realized_restitution_input,
    restitution_response_curve,
)

RADIUS = 0.01
DENSITY = 1000.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def brute_force_pairs(positions: torch.Tensor, radius: float) -> set[tuple[int, int]]:
    """O(N^2) reference neighbour list (strict ``distance < 2*radius``)."""
    cutoff = 2.0 * radius
    out: set[tuple[int, int]] = set()
    n = len(positions)
    for i in range(n):
        d = torch.linalg.vector_norm(positions[i + 1:] - positions[i], dim=1)
        for j in torch.nonzero(d < cutoff).flatten().tolist():
            out.add((i, j + i + 1))
    return out


def pair_set(pairs: torch.Tensor) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j in pairs.tolist()}


def config(**kwargs) -> VecDEMConfig:
    base = dict(n_particles=2, radius=RADIUS, density=DENSITY, friction_coefficient=0.0,
                gravity=0.0, seed=7)
    base.update(kwargs)
    return VecDEMConfig(**base)


def measure_restitution(e_configured: float, *, contact_model: str = "linear",
                        v_impact: float = 1.0, dt_refine: float = 10.0,
                        calibration: bool = True) -> float:
    """Realised restitution of one head-on collision, using the engine kernel.

    The pair starts exactly touching (zero overlap, therefore no spurious spring energy)
    and approaching with +-v_impact.
    """
    cfg = config(contact_model=contact_model, restitution_coefficient=e_configured,
                 restitution_calibration=calibration)
    mass = cfg.mass
    dt = cfg.effective_dt() / dt_refine
    pos = torch.tensor([[-RADIUS, 0.0, 0.0], [RADIUS, 0.0, 0.0]], dtype=torch.float64)
    vel = torch.tensor([[v_impact, 0.0, 0.0], [-v_impact, 0.0, 0.0]], dtype=torch.float64)
    pairs = torch.tensor([[0, 1]])
    for _ in range(2_000_000):
        f = compute_contact_forces(pos, vel, pairs, RADIUS, cfg)
        vel = vel + 0.5 * f / mass * dt
        pos = pos + vel * dt
        f = compute_contact_forces(pos, vel, pairs, RADIUS, cfg)
        vel = vel + 0.5 * f / mass * dt
        if (float(pos[1, 0] - pos[0, 0]) > 2.0 * RADIUS
                and float(vel[1, 0]) > 0.0 and float(vel[0, 0]) < 0.0):
            return abs(float(vel[1, 0] - vel[0, 0]) / (2.0 * v_impact))
    raise AssertionError("collision did not separate within the step cap")


# ---------------------------------------------------------------------------
# 1. neighbour search: cell list == O(N^2) brute force
# ---------------------------------------------------------------------------
def _configurations():
    torch.manual_seed(0)
    dense_side = 4
    g = torch.arange(dense_side, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
    dense = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3) * (2.0 * RADIUS * 0.99)
    return {
        "single": torch.zeros((1, 3), dtype=torch.float64),
        "pair_touching": torch.tensor([[-RADIUS, 0.0, 0.0], [RADIUS, 0.0, 0.0]],
                                      dtype=torch.float64),
        "pair_just_outside": torch.tensor(
            [[-RADIUS * 1.02, 0.0, 0.0], [RADIUS * 1.02, 0.0, 0.0]], dtype=torch.float64),
        "dense_lattice_overlapping": dense,
        "random_dilute": torch.rand(40, 3, dtype=torch.float64) * 0.4,
        "random_clustered": torch.randn(60, 3, dtype=torch.float64) * 0.012,
        "collinear_chain": torch.stack(
            [torch.arange(12, dtype=torch.float64) * 1.9 * RADIUS,
             torch.zeros(12, dtype=torch.float64),
             torch.zeros(12, dtype=torch.float64)], dim=1),
    }


@pytest.mark.parametrize("name", list(_configurations()))
def test_cell_list_equals_brute_force(name):
    positions = _configurations()[name]
    cells = find_neighbor_pairs_vectorized(positions, RADIUS)
    reference = brute_force_pairs(positions, RADIUS)
    assert pair_set(cells) == reference, (
        f"{name}: missing {sorted(reference - pair_set(cells))[:5]}, "
        f"extra {sorted(pair_set(cells) - reference)[:5]}")


@pytest.mark.parametrize("name", list(_configurations()))
def test_cell_list_soundness(name):
    """Every returned pair is i<j, unique, and strictly closer than the cutoff."""
    positions = _configurations()[name]
    pairs = find_neighbor_pairs_vectorized(positions, RADIUS)
    if len(pairs) == 0:
        return
    assert (pairs[:, 0] < pairs[:, 1]).all()
    assert len(pair_set(pairs)) == len(pairs)
    d = torch.linalg.vector_norm(positions[pairs[:, 1]] - positions[pairs[:, 0]], dim=1)
    assert float(d.max()) < 2.0 * RADIUS


# ---------------------------------------------------------------------------
# 2. contact laws against the analytic expressions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("overlap_fraction", [1e-3, 1e-2, 0.1])
def test_linear_contact_law(overlap_fraction):
    delta = overlap_fraction * RADIUS
    pos = torch.tensor([[-RADIUS + delta / 2, 0.0, 0.0],
                        [RADIUS - delta / 2, 0.0, 0.0]], dtype=torch.float64)
    vel = torch.zeros_like(pos)
    cfg = config(contact_stiffness=1e5, restitution_coefficient=1.0)
    f = compute_contact_forces(pos, vel, torch.tensor([[0, 1]]), RADIUS, cfg)
    assert float(f[0, 0]) == pytest.approx(-1e5 * delta, rel=1e-12)
    assert float(f[1, 0]) == pytest.approx(+1e5 * delta, rel=1e-12)


@pytest.mark.parametrize("overlap_fraction", [1e-3, 1e-2, 0.1])
def test_hertz_contact_law(overlap_fraction):
    """F = (4/3) E* sqrt(R*) delta^{3/2} with E* = E/(2(1-nu^2)), R* = R/2."""
    young, nu = 1e9, 0.3
    delta = overlap_fraction * RADIUS
    pos = torch.tensor([[-RADIUS + delta / 2, 0.0, 0.0],
                        [RADIUS - delta / 2, 0.0, 0.0]], dtype=torch.float64)
    cfg = config(contact_model="hertz", young_modulus=young, poisson_ratio=nu,
                 restitution_coefficient=1.0)
    f = compute_contact_forces(pos, torch.zeros_like(pos), torch.tensor([[0, 1]]),
                               RADIUS, cfg)
    E_star = young / (2.0 * (1.0 - nu ** 2))
    R_star = RADIUS / 2.0
    expected = (4.0 / 3.0) * E_star * math.sqrt(R_star) * delta ** 1.5
    assert float(f[0, 0]) == pytest.approx(-expected, rel=1e-12)


def test_no_overlap_no_force():
    pos = torch.tensor([[-1.05 * RADIUS, 0.0, 0.0], [1.05 * RADIUS, 0.0, 0.0]],
                       dtype=torch.float64)
    cfg = config()
    f = compute_contact_forces(pos, torch.zeros_like(pos), torch.tensor([[0, 1]]),
                               RADIUS, cfg)
    assert torch.count_nonzero(f) == 0


def test_action_reaction_machine_precision():
    torch.manual_seed(3)
    pos = torch.rand(30, 3, dtype=torch.float64) * 0.05
    vel = torch.randn(30, 3, dtype=torch.float64)
    cfg = config(contact_stiffness=1e5, friction_coefficient=0.4,
                 restitution_coefficient=0.5)
    pairs = find_neighbor_pairs_vectorized(pos, RADIUS)
    assert len(pairs) > 0
    f = compute_contact_forces(pos, vel, pairs, RADIUS, cfg)
    assert float(torch.abs(f.sum(dim=0)).max()) < 1e-13 * float(torch.abs(f).max())


# ---------------------------------------------------------------------------
# 3. integrator: conservation
# ---------------------------------------------------------------------------
def test_momentum_conservation_multi_particle():
    """Contact-only, no gravity, no walls: total momentum is an exact invariant."""
    torch.manual_seed(11)
    cfg = VecDEMConfig(n_particles=40, radius=RADIUS, density=DENSITY, seed=5,
                       friction_coefficient=0.3, restitution_coefficient=0.5,
                       contact_stiffness=1e5, gravity=0.0,
                       domain_size=(0.06, 0.06, 0.06))
    sim = VecDEM(cfg)
    sim.velocities += 0.2 * torch.randn_like(sim.velocities)
    p0 = sim.total_momentum().clone()
    scale = float(cfg.mass * torch.abs(sim.velocities).sum())
    for _ in range(400):
        sim.step()
    drift = float(torch.abs(sim.total_momentum() - p0).max()) / scale
    assert drift < 1e-12, f"momentum drift {drift:.3e}"


def test_free_fall_energy_conservation():
    """No contacts, gravity only: KE+PE is conserved by the velocity-Verlet integrator."""
    cfg = VecDEMConfig(n_particles=16, radius=RADIUS, density=DENSITY, gravity=9.81,
                       contact_stiffness=1e5, restitution_coefficient=1.0,
                       friction_coefficient=0.0, seed=2,
                       domain_size=(0.5, 0.5, 0.5))
    sim = VecDEM(cfg)
    # well separated particles (gap >> 2R) so that no contact can occur in free fall
    n = cfg.n_particles
    lin = torch.arange(n, dtype=torch.float64)
    sim.positions = torch.stack([lin * 0.02 + 0.05,
                                 torch.zeros(n, dtype=torch.float64),
                                 torch.full((n,), 0.5, dtype=torch.float64)], dim=1)
    sim.velocities[:] = 0.0
    e0 = sim.total_energy()
    for _ in range(1000):
        sim.step()
    e1 = sim.total_energy()
    assert abs(e1 - e0) / max(abs(e0), 1e-30) < 1e-6, (e0, e1)
    assert sim.velocities[:, 2].abs().max() > 0.0  # the cloud really fell


@pytest.mark.parametrize("contact_model", ["linear", "hertz"])
def test_elastic_collision_returns_the_kinetic_energy(contact_model):
    """With e=1 the collision is elastic: the KE lost in compression must come back.

    (Comparing KE *during* contact would be wrong -- the energy sits in the contact
    spring then.  The invariant to assert is the post-separation KE.)
    """
    cfg = config(contact_model=contact_model, restitution_coefficient=1.0)
    mass = cfg.mass
    dt = cfg.effective_dt() / 10.0
    pos = torch.tensor([[-RADIUS, 0.0, 0.0], [RADIUS, 0.0, 0.0]], dtype=torch.float64)
    vel = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
    pairs = torch.tensor([[0, 1]])
    e0 = 0.5 * mass * float((vel ** 2).sum())
    for _ in range(20000):
        f = compute_contact_forces(pos, vel, pairs, RADIUS, cfg)
        vel = vel + 0.5 * f / mass * dt
        pos = pos + vel * dt
        f = compute_contact_forces(pos, vel, pairs, RADIUS, cfg)
        vel = vel + 0.5 * f / mass * dt
        if float(pos[1, 0] - pos[0, 0]) > 2.0 * RADIUS and float(vel[1, 0]) > 0.0:
            break
    e1 = 0.5 * mass * float((vel ** 2).sum())
    assert abs(e1 - e0) / e0 < 5e-3, f"{contact_model}: recovered {e1 / e0:.5f} of KE"


def test_cell_list_engine_matches_reference_engine():
    """VecDEM (cell list) and ReferenceDEM (O(N^2)) must produce the same trajectory."""
    cfg = dict(n_particles=64, radius=RADIUS, density=DENSITY, gravity=9.81,
               contact_stiffness=1e5, restitution_coefficient=0.5,
               friction_coefficient=0.3, seed=4, domain_size=(0.06, 0.06, 0.06))
    a = VecDEM(VecDEMConfig(**cfg))
    b = ReferenceDEM(VecDEMConfig(**cfg))
    assert torch.equal(a.positions, b.positions)
    for _ in range(120):
        a.step()
        b.step()
    scale = float(torch.abs(a.positions).max())
    assert float(torch.abs(a.positions - b.positions).max()) < 1e-12 * max(scale, 1.0)
    assert float(torch.abs(a.velocities - b.velocities).max()) < 1e-12


def test_vectorized_engine_is_deterministic():
    cfg = dict(n_particles=64, radius=RADIUS, density=DENSITY, gravity=9.81,
               contact_stiffness=1e5, seed=9, domain_size=(0.06, 0.06, 0.06))
    a, b = VecDEM(VecDEMConfig(**cfg)), VecDEM(VecDEMConfig(**cfg))
    for _ in range(30):
        a.step()
        b.step()
    assert torch.equal(a.positions, b.positions)
    assert torch.equal(a.velocities, b.velocities)


# ---------------------------------------------------------------------------
# 4. realised restitution (the defect this suite found)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("contact_model", ["linear", "hertz"])
@pytest.mark.parametrize("e_target", [0.2, 0.3, 0.5, 0.7, 0.9])
def test_configured_restitution_is_realised(contact_model, e_target):
    """With calibration on, the realised rebound equals the configured coefficient."""
    realised = measure_restitution(e_target, contact_model=contact_model)
    assert realised == pytest.approx(e_target, rel=0.03), (
        f"{contact_model}: configured e={e_target} realised {realised:.4f}")


def test_legacy_restitution_bias_is_reproducible():
    """Without calibration the clamp over-rebounds by +33 % at e=0.30 (the defect).

    Pinned as a regression so the bias cannot silently come back as "physics".
    """
    legacy = measure_restitution(0.30, calibration=False)
    assert 0.37 < legacy < 0.43, legacy
    assert legacy > 1.15 * 0.30  # >15 % accuracy standard violated


def test_restitution_floor_is_reported_not_hidden():
    """The clamp sets an intrinsic floor; sub-floor targets saturate and are documented."""
    floor = realised_restitution_floor("linear")
    assert 0.12 < floor < 0.20
    realised = measure_restitution(0.05)
    assert realised >= floor - 1e-3
    assert realised == pytest.approx(floor, abs=0.02)


def test_response_curve_is_monotone_and_bounded():
    curve = restitution_response_curve("linear")
    realised = [r for _, r in curve]
    assert all(b > a for a, b in zip(realised, realised[1:]))
    assert realised[-1] == pytest.approx(1.0, abs=2e-5)  # dt/10 discretisation
    for _, r in curve:
        assert 0.0 <= r <= 1.0 + 5e-5  # +: dt/10 discretisation of the elastic limit


def test_linear_restitution_is_velocity_independent():
    """The calibrated linear model realises the same e at 0.2 m/s and 1.0 m/s."""
    slow = measure_restitution(0.5, v_impact=0.2)
    fast = measure_restitution(0.5, v_impact=1.0)
    assert abs(slow - fast) / fast < 0.02, (slow, fast)


def test_restitution_dt_convergence():
    coarse = measure_restitution(0.5, dt_refine=5.0)
    fine = measure_restitution(0.5, dt_refine=20.0)
    assert abs(coarse - fine) / fine < 0.02, (coarse, fine)


def test_realized_restitution_input_monotone():
    ins = [realized_restitution_input(e) for e in (0.2, 0.4, 0.6, 0.8, 0.99)]
    assert all(b > a for a, b in zip(ins, ins[1:]))


# ---------------------------------------------------------------------------
# 5. silent-parameter guards (were accepted and ignored)
# ---------------------------------------------------------------------------
def test_rolling_resistance_is_rejected_not_ignored():
    with pytest.raises(ValueError, match="rolling_resistance_coefficient"):
        config(rolling_resistance_coefficient=0.1)


def test_conflicting_damping_is_rejected():
    with pytest.raises(ValueError, match="contact_damping"):
        config(contact_damping=5.0, contact_damping_model="restitution")


def test_absolute_damping_is_honoured():
    """contact_damping_model='absolute' must actually change the force."""
    pos = torch.tensor([[-RADIUS + 1e-5, 0.0, 0.0], [RADIUS - 1e-5, 0.0, 0.0]],
                       dtype=torch.float64)
    vel = torch.tensor([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=torch.float64)
    cfg = config(contact_damping=5.0, contact_damping_model="absolute",
                 restitution_coefficient=1.0)
    f = compute_contact_forces(pos, vel, torch.tensor([[0, 1]]), RADIUS, cfg)
    assert float(f[0, 0]) < -1e5 * 2e-5  # damping adds to the repulsive force


@pytest.mark.parametrize("bad", ["Hertz", "hooke", ""])
def test_unknown_contact_model_is_rejected(bad):
    with pytest.raises(ValueError, match="contact_model"):
        config(contact_model=bad)


def test_unknown_damping_model_is_rejected():
    with pytest.raises(ValueError, match="contact_damping_model"):
        config(contact_damping_model="hunt_crossley")


def test_invalid_restitution_is_rejected():
    with pytest.raises(ValueError, match="restitution_coefficient"):
        config(restitution_coefficient=1.5)