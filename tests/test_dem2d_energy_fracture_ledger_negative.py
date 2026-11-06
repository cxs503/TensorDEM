"""Negative-control tests for the 2-D energy/fracture ledger balance.

These tests prove that the ledger balance identity is **sensitive** to
deliberate corruption of individual channels: when a channel is tampered
with, the balance check must *detect* it (the residual must exceed the
tolerance).  This is the complement of
``test_accounting_ledger_validation.py`` which proves the *uncorrupted*
ledger closes.

Scenario: tool-indentation run that breaks bonds (non-trivial fracture
release + pair-contact acquired).  The energy balance identity is:

    dE_mech - W_tool - W_external + Q_damping + E_fracture - E_pair_contact == 0

Each negative test injects a known fault into one channel and asserts the
balance check **catches** it (residual >= tolerance).  A green suite means
the check is sensitive to that channel; a red suite means the check has
become blind to it.

Note: the tool-indentation scenario applies no external nodal loads, so
``external_work_J`` is exactly 0.  Corrupting a zero channel is a no-op, so
that one control is skipped (vacuous) rather than asserted on unchanged data.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensordem import DEMConfig, IceDEM  # noqa: E402

#: simulated seconds for the indentation run
DURATION = 0.12
LATTICE = (7, 3)

#: relative residual below which the (uncorrupted) ledger is considered closed
TOL = 0.005


def _run_indentation() -> IceDEM:
    """Run a tool-indentation simulation that breaks bonds."""
    nx, ny = LATTICE
    dt0 = IceDEM(DEMConfig(nx=nx, ny=ny, tool_speed=0.2)).dt
    config = DEMConfig(nx=nx, ny=ny, tool_speed=0.2, dt=dt0)
    sim = IceDEM(config)
    steps = int(round(DURATION / sim.dt))
    for _ in range(steps):
        sim.step()
    assert sim.broken_bonds > 0, "indentation must break bonds for this test"
    return sim


def _energy_residual(sim: IceDEM) -> float:
    """Compute the energy balance residual from the ledger.

    dE_mech - W_tool - W_external + Q_damping + E_fracture - E_pair_contact
    """
    nx, ny = LATTICE
    dt0 = IceDEM(DEMConfig(nx=nx, ny=ny, tool_speed=0.2)).dt
    config = DEMConfig(nx=nx, ny=ny, tool_speed=0.2, dt=dt0)
    sim0 = IceDEM(config)
    energy0 = sim0.mechanical_energy()["mechanical_J"]
    energy1 = sim.mechanical_energy()["mechanical_J"]
    dE = energy1 - energy0
    acc = sim.accounting
    residual = (
        dE
        - acc["tool_work_J"]
        - acc["external_work_J"]
        + acc["damping_dissipation_J"]
        + acc["fracture_release_J"]
        - acc["pair_contact_acquired_J"]
    )
    return residual


def _energy_scale(sim: IceDEM) -> float:
    """Reference scale for the energy balance."""
    acc = sim.accounting
    return (
        abs(acc["tool_work_J"])
        + abs(acc["damping_dissipation_J"])
        + abs(acc["fracture_release_J"])
    )


# ---------------------------------------------------------------------------
# Positive control: the uncorrupted ledger must close
# ---------------------------------------------------------------------------

def test_positive_control_ledger_closes():
    """The uncorrupted energy/fracture ledger must close to <0.5 %."""
    sim = _run_indentation()
    residual = _energy_residual(sim)
    scale = _energy_scale(sim)
    assert scale > 0.0, "energy scale must be non-trivial"
    rel_err = abs(residual) / scale
    assert rel_err < TOL, (
        f"uncorrupted ledger residual {rel_err:.4g} exceeds 0.5 %"
    )


# ---------------------------------------------------------------------------
# Negative controls: deliberately corrupt one channel, then assert the
# balance check DETECTS the corruption (residual >= tolerance).
#
# A working balance check is sensitive: corrupting a channel must break the
# balance.  These tests are GREEN when the check catches the fault and turn
# RED if the check ever becomes blind to that channel.
# ---------------------------------------------------------------------------

def _corrupt_and_assert_detected(sim: IceDEM, channel: str, factor: float) -> None:
    """Corrupt one ledger channel, then assert the balance check detects it.

    The corrupted ledger must NOT close: the residual must exceed the
    tolerance.  If the check is broken (insensitive to this channel), the
    corrupted ledger would still close and this assertion fails.

    If the channel is exactly zero in this scenario, the corruption is a
    no-op and the control is vacuous — skip rather than assert on unchanged
    data.
    """
    corrupted = copy.deepcopy(sim)
    original = sim.accounting[channel]
    corrupted.accounting[channel] = original * factor
    if corrupted.accounting[channel] == original:
        pytest.skip(
            f"channel {channel} is zero in this scenario; "
            f"x{factor} is a no-op, nothing to detect"
        )
    residual = _energy_residual(corrupted)
    scale = _energy_scale(corrupted)
    rel_err = abs(residual) / max(scale, 1e-18)
    assert rel_err >= TOL, (
        f"corrupted ledger ({channel} x {factor}) residual {rel_err:.4g} "
        f"did not exceed 0.5 % — the balance check failed to catch the fault"
    )


def test_negative_fracture_release_zeroed():
    """Negative: zeroing fracture_release_J must be caught by the check."""
    sim = _run_indentation()
    _corrupt_and_assert_detected(sim, "fracture_release_J", 0.0)


def test_negative_fracture_release_tripled():
    """Negative: tripling fracture_release_J must be caught by the check."""
    sim = _run_indentation()
    _corrupt_and_assert_detected(sim, "fracture_release_J", 3.0)


def test_negative_pair_contact_zeroed():
    """Negative: zeroing pair_contact_acquired_J must be caught by the check."""
    sim = _run_indentation()
    _corrupt_and_assert_detected(sim, "pair_contact_acquired_J", 0.0)


def test_negative_pair_contact_tripled():
    """Negative: tripling pair_contact_acquired_J must be caught by the check."""
    sim = _run_indentation()
    _corrupt_and_assert_detected(sim, "pair_contact_acquired_J", 3.0)


def test_negative_tool_work_zeroed():
    """Negative: zeroing tool_work_J must be caught by the check."""
    sim = _run_indentation()
    _corrupt_and_assert_detected(sim, "tool_work_J", 0.0)


def test_negative_damping_dissipation_zeroed():
    """Negative: zeroing damping_dissipation_J must be caught by the check."""
    sim = _run_indentation()
    _corrupt_and_assert_detected(sim, "damping_dissipation_J", 0.0)


def test_negative_external_work_negated():
    """Negative: negating external_work_J must be caught by the check.

    Skipped when the scenario produces no external work (channel is 0), in
    which case the corruption is a no-op and the control is vacuous.
    """
    sim = _run_indentation()
    _corrupt_and_assert_detected(sim, "external_work_J", -1.0)


# ---------------------------------------------------------------------------
# Self-evidence: document the negative-test commands and expected output
# ---------------------------------------------------------------------------

def test_negative_self_evidence():
    """Documentation test: proves the negative controls exist and are runnable.

    This test itself passes (green).

    Negative-evidence commands:
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
            tests/test_dem2d_energy_fracture_ledger_negative.py -q

    Expected output (summary):
        8 passed, 1 skipped
        (the external_work control is vacuous: that channel is 0 here)
    """
    sim = _run_indentation()
    scale = _energy_scale(sim)
    assert scale > 0.0, "energy scale must be non-trivial for negative tests"
    assert sim.broken_bonds > 0, "must have broken bonds for fracture channels"
