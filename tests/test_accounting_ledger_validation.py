"""Close the validation gap on the 2-D impulse/energy ledger.

`tests/test_accounting_output.py` only asserts that the accounting channels
exist and are finite.  This module confronts every channel with an independent,
closed-form budget:

    momentum:  m * sum_i v_i(t) == I_external + I_support + I_drag + I_tool
    energy:    dE_mech == W_external + W_tool - Q_damping - E_fracture
                          + E_pair_contact_acquired

The last term is energy that a bond which breaks while still overlapping keeps
as repulsive pair-contact potential: it never leaves the system and therefore
must be credited back.  Without it the ledger over-reports dissipated fracture
energy by about 2.6 % of the total energy scale on a tool-indentation run, and
that deficit does *not* vanish under time-step refinement (it is bookkeeping,
not discretization).

All cases use the 2-D bonded-disk solver with SI units and are deliberately
small so the suite stays a fast regression check.
"""

from __future__ import annotations

import math

import torch

from tensordem import DEMConfig, IceDEM

HEIGHT_STEPS = 0.12  # simulated seconds for the indentation cases
LATTICE = (7, 3)


def _indentation(dt_scale: float = 1.0, duration: float = HEIGHT_STEPS) -> dict:
    """Run a tool indentation and return the ledger terms for closure checks."""
    nx, ny = LATTICE
    dt0 = IceDEM(DEMConfig(nx=nx, ny=ny, tool_speed=0.2)).dt
    config = DEMConfig(nx=nx, ny=ny, tool_speed=0.2, dt=dt0 * dt_scale)
    sim = IceDEM(config)
    energy0 = sim.mechanical_energy()["mechanical_J"]
    steps = int(round(duration / sim.dt))
    for _ in range(steps):
        sim.step()
    energy1 = sim.mechanical_energy()["mechanical_J"]
    acc = sim.accounting
    raw = (
        (energy1 - energy0)
        - acc["tool_work_J"]
        - acc["external_work_J"]
        + acc["damping_dissipation_J"]
        + acc["fracture_release_J"]
    )
    scale = (
        abs(acc["tool_work_J"])
        + abs(acc["damping_dissipation_J"])
        + abs(energy1 - energy0)
    )
    return dict(
        sim=sim,
        config=config,
        steps=steps,
        energy0=energy0,
        energy1=energy1,
        accounting=dict(acc),
        raw_residual=raw,
        closed_residual=raw - acc["pair_contact_acquired_J"],
        scale=scale,
    )


def _momentum_terms(sim: IceDEM) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (m * sum v, sum of the four impulse channels)."""
    momentum = sim.config.mass * sim.velocities.sum(dim=0)
    channels = ("external_impulse_Ns", "support_impulse_Ns",
                "drag_impulse_Ns", "tool_impulse_on_ice_Ns")
    ledger = sum(torch.as_tensor(sim.accounting[name]) for name in channels)
    return momentum, ledger


def test_integral_momentum_balance_closes_to_roundoff():
    """A uniform nodal load must be reproduced exactly by the impulse ledger."""
    config = DEMConfig(nx=3, ny=3, tool_gap=10.0, tool_speed=1e-9,
                       drag=0.0, contact_damping=0.0)
    sim = IceDEM(config)
    load = torch.zeros_like(sim.positions)
    load[:, 1] = 1e-3
    for _ in range(40):
        sim.step(external_forces=load)

    momentum, ledger = _momentum_terms(sim)
    reference = float(momentum.abs().max())
    assert reference > 0.0
    assert float((momentum - ledger).abs().max()) <= 1e-6 * reference


def test_momentum_balance_covers_drag_tool_and_support_channels():
    """Drag, tool reaction and fixity clamping all appear in the same identity."""
    nx, ny = LATTICE
    config = DEMConfig(nx=nx, ny=ny, tool_speed=0.2, drag=0.5,
                       contact_damping=5.0)
    sim = IceDEM(config)
    steps = int(round(HEIGHT_STEPS / sim.dt))
    for _ in range(steps):
        sim.step()

    momentum, ledger = _momentum_terms(sim)
    reference = max(float(momentum.abs().max()), float(ledger.abs().max()))
    assert reference > 1e-9, "expected non-trivial momentum transfer"
    # Non-zero tool/drag activity is what makes this identity non-trivial.
    assert abs(sim.accounting["drag_impulse_Ns"][1]) > 0.0
    assert abs(sim.accounting["tool_impulse_on_ice_Ns"][1]) > 0.0
    assert float((momentum - ledger).abs().max()) <= 1e-6 * reference


def test_broken_overlapping_bond_retains_pair_contact_energy():
    """The credited term equals the analytic contact energy of broken pairs.

    For a broken pair with stored failure extension e (negative = compression)
    and initial rest length L, the pair becomes a contact whose overlap is
    2R - (L + e); the retained energy is 0.5 * k_contact * overlap**2.
    """
    result = _indentation()
    sim = result["sim"]
    assert sim.broken_bonds > 0, "case must actually break bonds"
    radius = result["config"].radius
    overlap = (2 * radius - (sim.rest_lengths + sim.failure_extension_m)).clamp_min(0.0)
    broken = sim.failure_energy_J > 0.0
    analytic = float(
        (0.5 * result["config"].contact_stiffness * overlap[broken].square()).sum()
    )
    credited = result["accounting"]["pair_contact_acquired_J"]
    assert analytic > 0.0
    assert abs(credited - analytic) <= 1e-12 * max(analytic, 1.0)
    # The retained share is a substantial part of the reported release here.
    assert 0.2 < credited / result["accounting"]["fracture_release_J"] < 0.8


def test_energy_ledger_closes_with_retained_contact_term():
    """Raw fracture release over-reports; crediting retained contact closes it."""
    result = _indentation()
    raw_relative = abs(result["raw_residual"]) / result["scale"]
    closed_relative = abs(result["closed_residual"]) / result["scale"]
    # Documented defect: the untreated ledger loses >1 % of the energy scale.
    assert raw_relative > 0.01
    # After the fix the identity closes to well under half a percent.
    assert closed_relative < 0.005
    assert closed_relative < raw_relative / 5.0


def test_energy_residual_is_bookkeeping_not_timestep_error():
    """The raw deficit is dt-independent; the corrected residue is O(dt)."""
    coarse = _indentation(dt_scale=1.0)
    fine = _indentation(dt_scale=0.5)
    raw_ratio = abs(coarse["raw_residual"]) / abs(fine["raw_residual"])
    assert 0.7 < raw_ratio < 1.4, "raw deficit must not shrink with dt"
    closed_ratio = abs(coarse["closed_residual"]) / abs(fine["closed_residual"])
    assert closed_ratio > 1.2, "corrected residue must vanish first order in dt"


def test_restart_accepts_checkpoint_without_new_ledger_channel():
    """Older snapshots predate pair_contact_acquired_J and must still restore."""
    config = DEMConfig(nx=5, ny=3, tool_speed=0.2)
    sim = IceDEM(config)
    for _ in range(5):
        sim.step()
    state = sim.snapshot()
    del state["accounting"]["pair_contact_acquired_J"]
    restored = IceDEM.from_snapshot(state)
    assert restored.accounting["pair_contact_acquired_J"] == 0.0
    assert math.isfinite(restored.accounting["fracture_release_J"])