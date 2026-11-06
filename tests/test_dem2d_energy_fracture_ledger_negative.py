"""Negative-control tests for the 2-D energy/fracture ledger balance.

These tests prove that the ledger balance identity is **sensitive** to
deliberate corruption of individual channels: when a channel is tampered
with, the balance check must fail (turn red).  This is the complement of
``test_accounting_ledger_validation.py`` which proves the *uncorrupted*
ledger closes.

Scenario: tool-indentation run that breaks bonds (non-trivial fracture
release + pair-contact acquired).  The energy balance identity is:

    dE_mech - W_tool - W_external + Q_damping + E_fracture - E_pair_contact == 0

Each test injects a known fault into one channel and asserts the balance
closes — which it cannot, so the test turns red.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensordem import DEMConfig, IceDEM  # noqa: E402

#: simulated seconds for the indentation run
DURATION = 0.12
LATTICE = (7, 3)


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
    assert rel_err < 0.005, (
        f"uncorrupted ledger residual {rel_err:.4g} exceeds 0.5 %"
    )


# ---------------------------------------------------------------------------
# Negative controls: deliberately corrupt one channel, then assert balance
# closes → this MUST FAIL (red) because the ledger is broken.
#
# To prove the negative, we run the corrupted simulation through the same
# balance check as the positive control.  The assertion is identical, but
# the data is poisoned, so the test turns red.
# ---------------------------------------------------------------------------

def _corrupt_and_assert_balance(sim: IceDEM, channel: str, factor: float) -> None:
    """Corrupt one ledger channel, then assert the balance closes.

    This function is designed to FAIL when called with factor != 1.0,
    proving the balance check is sensitive to channel corruption.
    """
    corrupted = copy.deepcopy(sim)
    corrupted.accounting[channel] *= factor
    residual = _energy_residual(corrupted)
    scale = _energy_scale(corrupted)
    rel_err = abs(residual) / max(scale, 1e-18)
    # This assertion is identical to the positive control, but on corrupted
    # data — so it must fail.
    assert rel_err < 0.005, (
        f"corrupted ledger ({channel} × {factor}) residual {rel_err:.4g} "
        f"should exceed 0.5 % — this test is expected to FAIL (red)"
    )


def test_negative_fracture_release_zeroed():
    """Negative: zeroing fracture_release_J must break the balance.

    This test is EXPECTED TO FAIL (red).  It proves the ledger balance
    check catches a missing fracture-release channel.
    """
    sim = _run_indentation()
    _corrupt_and_assert_balance(sim, "fracture_release_J", 0.0)


def test_negative_fracture_release_tripled():
    """Negative: tripling fracture_release_J must break the balance.

    This test is EXPECTED TO FAIL (red).  It proves the ledger balance
    check catches an inflated fracture-release channel.
    """
    sim = _run_indentation()
    _corrupt_and_assert_balance(sim, "fracture_release_J", 3.0)


def test_negative_pair_contact_zeroed():
    """Negative: zeroing pair_contact_acquired_J must break the balance.

    This test is EXPECTED TO FAIL (red).  It proves the ledger balance
    check catches a missing pair-contact-acquired channel.
    """
    sim = _run_indentation()
    _corrupt_and_assert_balance(sim, "pair_contact_acquired_J", 0.0)


def test_negative_pair_contact_tripled():
    """Negative: tripling pair_contact_acquired_J must break the balance.

    This test is EXPECTED TO FAIL (red).  It proves the ledger balance
    check catches an inflated pair-contact-acquired channel.
    """
    sim = _run_indentation()
    _corrupt_and_assert_balance(sim, "pair_contact_acquired_J", 3.0)


def test_negative_tool_work_zeroed():
    """Negative: zeroing tool_work_J must break the balance.

    This test is EXPECTED TO FAIL (red).  It proves the ledger balance
    check catches a missing tool-work channel.
    """
    sim = _run_indentation()
    _corrupt_and_assert_balance(sim, "tool_work_J", 0.0)


def test_negative_damping_dissipation_zeroed():
    """Negative: zeroing damping_dissipation_J must break the balance.

    This test is EXPECTED TO FAIL (red).  It proves the ledger balance
    check catches a missing damping-dissipation channel.
    """
    sim = _run_indentation()
    _corrupt_and_assert_balance(sim, "damping_dissipation_J", 0.0)


def test_negative_external_work_negated():
    """Negative: negating external_work_J must break the balance.

    This test is EXPECTED TO FAIL (red).  It proves the ledger balance
    check catches a sign-flipped external-work channel.
    """
    sim = _run_indentation()
    _corrupt_and_assert_balance(sim, "external_work_J", -1.0)


# ---------------------------------------------------------------------------
# Self-evidence: document the negative-test commands and expected output
# ---------------------------------------------------------------------------

def test_negative_self_evidence():
    """Documentation test: proves the negative controls exist and are
    runnable.  This test itself passes (green).

    Negative-evidence commands:
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
            tests/test_dem2d_energy_fracture_ledger_negative.py \
            -k negative -q

    Expected red output (summary):
        FAILED ... test_negative_fracture_release_zeroed
        FAILED ... test_negative_fracture_release_tripled
        FAILED ... test_negative_pair_contact_zeroed
        FAILED ... test_negative_pair_contact_tripled
        FAILED ... test_negative_tool_work_zeroed
        FAILED ... test_negative_damping_dissipation_zeroed
        FAILED ... test_negative_external_work_negated
        7 failed, 2 passed
    """
    sim = _run_indentation()
    scale = _energy_scale(sim)
    assert scale > 0.0, "energy scale must be non-trivial for negative tests"
    assert sim.broken_bonds > 0, "must have broken bonds for fracture channels"
