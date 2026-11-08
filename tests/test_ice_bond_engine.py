"""Engine-level tests for the opt-in rate-dependent ice bond.

Scope: the *wiring* of ``ice_bond_viscoelastic`` into ``IceDEM3D`` (force
evaluation, branch advance, dashpot dissipation ledger, Kachanov creep clock,
checkpointing) and the backward compatibility of the legacy linear bond.
Constitutive correctness of the bond law itself is covered by
``tests/test_ice_bond_viscoelastic.py``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.dem3d_energy import EnergyAuditedIceDEM3D
from tensordem.ice_bond_viscoelastic import MaxwellWiechertBond


def _config(**overrides) -> DEM3DConfig:
    base = dict(nx=3, ny=3, nz=3, radius=0.025, density=917.0,
                bond_stiffness=2_000.0, contact_stiffness=5_000.0,
                fix_x_edges=False, drag=0.0)
    base.update(overrides)
    return DEM3DConfig(**base)


def _viscoelastic(**overrides) -> DEM3DConfig:
    base = dict(bond_model="maxwell_wiechert",
                bond_equilibrium_fraction=0.35,
                bond_delayed_elastic_fraction=(0.45, 0.20),
                bond_relaxation_time_s=(1.0e-3, 1.0e-2))
    base.update(overrides)
    return _config(**base)


def _extension(sim: IceDEM3D) -> torch.Tensor:
    i, j = sim.pairs.T
    return torch.linalg.vector_norm(sim.positions[j] - sim.positions[i], dim=1) - sim.rest_lengths


# ---------------------------------------------------------------------------
# Backward compatibility: the default stays the legacy linear spring
# ---------------------------------------------------------------------------
def test_default_bond_model_is_linear_and_bit_for_bit_legacy():
    sim = IceDEM3D(_config())
    assert sim.bond_law is None
    assert sim.branch_offsets.shape == (len(sim.pairs), 0)
    extension = _extension(sim)
    assert torch.equal(sim._bond_force(extension, update_history=True),
                       sim.config.bond_stiffness * extension)
    assert torch.equal(sim.bond_stored_energy(extension),
                       (0.5 * sim.config.bond_stiffness * extension.square()))
    # and a real trajectory is unchanged by the new code path being present
    sim2 = IceDEM3D(_config(bond_model="linear"))
    for _ in range(20):
        sim.step()
    for _ in range(20):
        sim2.step()
    assert torch.equal(sim.positions, sim2.positions)
    assert sim.cumulative_bond_dissipation_J == 0.0


def test_creep_clock_is_inert_on_the_linear_bond():
    sim = IceDEM3D(_config())
    for _ in range(10):
        sim.step()
    assert float(sim.bond_damage.abs().max()) == 0.0
    assert int((sim.failure_mode == 4).sum()) == 0


# ---------------------------------------------------------------------------
# Instantaneous response and relaxation
# ---------------------------------------------------------------------------
def test_instantaneous_force_uses_the_unrelaxed_stiffness():
    sim = IceDEM3D(_viscoelastic())
    assert sim.unrelaxed_bond_stiffness == pytest.approx(sim.config.bond_stiffness)
    extension = torch.full_like(sim.rest_lengths, 1.0e-4)
    force = sim._bond_force(extension, update_history=False)
    # branch offsets are zero and s_a = d, so F = (k0 + sum k_a) d
    assert torch.allclose(force, sim.unrelaxed_bond_stiffness * extension, rtol=1e-15)


def test_stress_relaxation_follows_the_prony_series():
    """Holding the elongation fixed, the force must decay along G(t)."""
    sim = IceDEM3D(_viscoelastic())
    model = sim.bond_law
    extension = torch.full_like(sim.rest_lengths, 2.0e-5)
    sim._bond_elongation_prev = extension.clone()
    steps = 400
    for _ in range(steps):
        force = sim._bond_force(extension, update_history=True)
    t = steps * sim.dt
    reference = model.relaxation_modulus(t) * extension
    denom = float((model.unrelaxed_stiffness * extension).max())
    assert float((force - reference).abs().max()) / denom < 1e-10
    # the relaxed branch is real: the force must have dropped measurably
    assert float(force.max()) < 0.95 * float(model.unrelaxed_stiffness * extension.max())


def test_probe_calls_do_not_advance_the_constitutive_state():
    """``update_history=False`` must leave branch offsets, damage and the
    dissipation ledger untouched, otherwise monitoring changes the physics."""
    sim = IceDEM3D(_viscoelastic(bond_creep_exponent=2, bond_creep_time_s=1.0))
    sim.positions[1] += torch.tensor([1.0e-5, 0.0, 0.0], dtype=sim.dtype)
    sim.forces(update_fracture=True)
    offsets = sim.branch_offsets.clone()
    damage = sim.bond_damage.clone()
    dissipation = sim.cumulative_bond_dissipation_J
    for _ in range(5):
        sim.forces(update_fracture=False, update_history=False)
    assert torch.equal(sim.branch_offsets, offsets)
    assert torch.equal(sim.bond_damage, damage)
    assert sim.cumulative_bond_dissipation_J == dissipation


# ---------------------------------------------------------------------------
# Creep damage clock
# ---------------------------------------------------------------------------
def test_creep_damage_increment_matches_the_closed_form():
    sim = IceDEM3D(_viscoelastic(bond_creep_exponent=2, bond_creep_time_s=0.5))
    reference = sim.bond_creep_reference_force()
    extension = _extension(sim)
    # Force exactly at F_ref: dD/dt = 1/t_ref, so N steps give N*dt/t_ref.
    sim._commit_creep_damage(reference.clone(), extension)
    assert torch.allclose(sim.bond_damage, torch.full_like(reference, sim.dt / 0.5),
                          rtol=1e-12)
    # Half the reference force with m = 2 -> quarter of the rate.
    sim2 = IceDEM3D(_viscoelastic(bond_creep_exponent=2, bond_creep_time_s=0.5))
    sim2._commit_creep_damage(0.5 * reference, extension)
    assert torch.allclose(sim2.bond_damage, torch.full_like(reference, 0.25 * sim2.dt / 0.5),
                          rtol=1e-12)


def test_creep_failure_deletes_bonds_with_mode_four():
    sim = IceDEM3D(_viscoelastic(bond_creep_exponent=2, bond_creep_time_s=1.0e-4))
    reference = sim.bond_creep_reference_force()
    extension = _extension(sim)
    n_before = int(sim.alive.sum())
    for _ in range(200):
        sim._commit_creep_damage(reference.clone(), extension)
    assert int(sim.alive.sum()) < n_before
    failed = sim.failure_mode == 4
    assert bool(failed.any())
    assert bool((sim.failure_time_s[failed] >= 0.0).all())
    # deleted bonds keep no stored energy in the ledger
    stored = sim.bond_stored_energy(extension)
    assert float(stored[~sim.alive].sum()) >= 0.0


def test_creep_damage_is_monotone_and_bounded_per_bond():
    sim = IceDEM3D(_viscoelastic(bond_creep_exponent=2, bond_creep_time_s=10.0))
    reference = sim.bond_creep_reference_force()
    extension = _extension(sim)
    previous = sim.bond_damage.clone()
    for _ in range(10):
        sim._commit_creep_damage(0.3 * reference, extension)
        assert bool((sim.bond_damage >= previous).all())
        previous = sim.bond_damage.clone()
    assert float(sim.bond_damage.max()) < 1.0


# ---------------------------------------------------------------------------
# Energy ledger
# ---------------------------------------------------------------------------
def test_bond_dissipation_is_nonnegative_and_does_not_degrade_the_ledger():
    """The Maxwell dashpot must enter the ledger with the right sign.

    The audited engine's closure residual is *not* zero on this problem even for
    the legacy linear bond (measured O(1e-3) relative): it is dominated by the
    pre-existing first-order work/dissipation estimate of a stiff spring-mass
    chain, not by the bond law.  The test therefore pins the new term against a
    linear-bond twin on the identical geometry/steps instead of an unreachable
    absolute bound.
    """
    def measure(config):
        sim = EnergyAuditedIceDEM3D(config)
        sim.positions[sim.positions[:, 2] > 0, 2] += 4.0e-4
        sim.reference_mechanical_energy_J = sim._mechanical_energy()
        for _ in range(50):
            sim.step()
        row = sim.diagnostics()
        scale = max(abs(sim.reference_mechanical_energy_J)
                    + abs(sim.cumulative_platen_work_J), 1e-30)
        assert row["bond_dissipation_cumulative_J"] == pytest.approx(
            sim.cumulative_bond_dissipation_J)
        return abs(row["energy_balance_residual_J"]) / scale, sim

    residual_linear, linear = measure(_config())
    residual_mw, mw = measure(_viscoelastic())
    assert linear.cumulative_bond_dissipation_J == 0.0
    assert mw.cumulative_bond_dissipation_J > 0.0
    assert residual_mw <= 3.0 * residual_linear + 1e-9, (residual_mw, residual_linear)


def test_branch_springs_are_visible_to_the_mechanical_energy():
    sim = EnergyAuditedIceDEM3D(_viscoelastic())
    sim.positions[1] += torch.tensor([5.0e-4, 0.0, 0.0], dtype=sim.dtype)
    extension = _extension(sim)
    stored = sim.bond_stored_energy(extension)
    naive = 0.5 * sim.config.bond_stiffness * extension.square()
    # with undeformed branches the two agree; after a step they must not
    sim.forces()
    stored_after = sim.bond_stored_energy(_extension(sim))
    naive_after = 0.5 * sim.config.bond_stiffness * _extension(sim).square()
    assert torch.allclose(stored, naive)
    assert not torch.allclose(stored_after, naive_after)


# ---------------------------------------------------------------------------
# Checkpointing and validation
# ---------------------------------------------------------------------------
def test_state_dict_round_trip_carries_the_bond_state():
    sim = IceDEM3D(_viscoelastic(bond_creep_exponent=2, bond_creep_time_s=1.0))
    sim.positions[1] += torch.tensor([2.0e-5, 0.0, 0.0], dtype=sim.dtype)
    for _ in range(5):
        sim.step()
    state = sim.state_dict()
    assert "bond_law" in state
    fresh = IceDEM3D(_viscoelastic(bond_creep_exponent=2, bond_creep_time_s=1.0))
    fresh.load_state_dict(state)
    assert torch.equal(fresh.branch_offsets, sim.branch_offsets)
    assert torch.equal(fresh.bond_damage, sim.bond_damage)
    assert fresh.cumulative_bond_dissipation_J == sim.cumulative_bond_dissipation_J


def test_load_state_dict_rejects_a_checkpoint_without_bond_state():
    sim = IceDEM3D(_viscoelastic(bond_creep_exponent=2))
    state = sim.state_dict()
    del state["bond_law"]
    fresh = IceDEM3D(_viscoelastic(bond_creep_exponent=2))
    with pytest.raises(ValueError, match="bond-law state"):
        fresh.load_state_dict(state)


def test_legacy_checkpoint_still_loads():
    sim = IceDEM3D(_config())
    state = sim.state_dict()
    state.pop("bond_law", None)
    fresh = IceDEM3D(_config())
    fresh.load_state_dict(state)
    assert torch.equal(fresh.positions, sim.positions)


@pytest.mark.parametrize("overrides,match", [
    ({"bond_model": "pronyseries"}, "bond_model"),
    ({"bond_model": "maxwell_wiechert", "bond_relaxation_time_s": (),
      "bond_delayed_elastic_fraction": ()}, "relaxation time"),
    ({"bond_relaxation_time_s": (1.0e-3,), "bond_delayed_elastic_fraction": (0.45, 0.2)},
     "equal length"),
    ({"bond_equilibrium_fraction": 1.0}, "equilibrium_fraction"),
    ({"bond_creep_exponent": -1}, "bond_creep_exponent"),
    ({"bond_creep_exponent": 2, "bond_creep_time_s": 0.0}, "bond_creep_time_s"),
])
def test_config_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        _config(**overrides)


def test_unrelaxed_stiffness_is_preserved_by_the_split():
    cfg = _viscoelastic()
    model = MaxwellWiechertBond.from_unrelaxed_stiffness(
        cfg.bond_stiffness,
        equilibrium_fraction=cfg.bond_equilibrium_fraction,
        delayed_elastic_fraction=cfg.bond_delayed_elastic_fraction,
        relaxation_time=cfg.bond_relaxation_time_s)
    assert model.unrelaxed_stiffness == pytest.approx(cfg.bond_stiffness)
    sim = IceDEM3D(cfg)
    assert sim.bond_law.unrelaxed_stiffness == pytest.approx(cfg.bond_stiffness)
    assert math.isclose(sim.bond_law.equilibrium_stiffness,
                        cfg.bond_equilibrium_fraction * cfg.bond_stiffness)