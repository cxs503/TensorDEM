"""Negative-control tests for the 2-D energy/fracture ledger balance.

These tests prove that the ledger balance identity is **sensitive** to
deliberate corruption of individual channels: when a channel is tampered
with, the balance check must *reject* the data.  This is the complement of
``test_accounting_ledger_validation.py`` which proves the *uncorrupted*
ledger closes.

Scenario: tool-indentation run that breaks bonds (non-trivial fracture
release + pair-contact acquired).  The energy balance identity is:

    dE_mech - W_tool - W_external + Q_damping + E_fracture - E_pair_contact == 0

Each test injects a known fault into exactly one channel and then asserts
that the balance gate (``balance_check``) raises.  The tests are green when
the gate turns red -- i.e. the negative result is the *assertion*, not the
failure mode.  (An earlier revision left the corrupted balance assertion
bare, so the suite was red by construction and could never pass on CI;
see ``docs/dem2d_energy_fracture_ledger_negative_controls.md``.)
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

#: relative tolerance of the balance identity (same as the positive control)
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


def _rel_err(sim: IceDEM) -> float:
    """Relative ledger residual of ``sim``."""
    return abs(_energy_residual(sim)) / max(_energy_scale(sim), 1e-18)


def balance_check(sim: IceDEM, tol: float = TOL) -> float:
    """The ledger gate under test: raise if the balance does not close.

    This is the identical assertion used by the positive control.  Negative
    controls call it through ``pytest.raises`` on corrupted data.
    """
    rel_err = _rel_err(sim)
    assert rel_err < tol, (
        f"energy/fracture ledger residual {rel_err:.4g} exceeds {tol:.4g}"
    )
    return rel_err


def _corrupt(
    sim: IceDEM,
    channel: str,
    factor: float | None = None,
    offset: float | None = None,
) -> IceDEM:
    """Deep-copy ``sim`` and inject a fault into exactly one ledger channel."""
    corrupted = copy.deepcopy(sim)
    assert channel in corrupted.accounting, f"unknown ledger channel {channel!r}"
    if factor is not None:
        corrupted.accounting[channel] *= factor
    if offset is not None:
        corrupted.accounting[channel] += offset
    return corrupted


def _assert_detects(
    sim: IceDEM,
    channel: str,
    factor: float | None = None,
    offset: float | None = None,
) -> float:
    """Assert the balance gate rejects a single-channel fault.

    Requirements for a valid negative control:

    1. the fault must be non-vacuous (the channel value actually changes);
    2. the uncorrupted ledger must close, so the rejection is attributable to
       the fault rather than to a broken baseline;
    3. the gate must raise on the corrupted data.

    Returns the corrupted relative residual for reporting.
    """
    baseline: float = sim.accounting[channel]
    corrupted = _corrupt(sim, channel, factor=factor, offset=offset)
    injected: float = corrupted.accounting[channel]
    assert injected != baseline, (
        f"corruption of {channel!r} (factor={factor}, offset={offset}) is "
        f"vacuous: value stays {baseline!r} -- pick a non-vacuous fault"
    )

    clean_err = balance_check(sim)  # (2) baseline closes
    corrupted_err = _rel_err(corrupted)
    assert corrupted_err > TOL, (
        f"corrupted ledger ({channel}: {baseline:.6g} -> {injected:.6g}) "
        f"residual {corrupted_err:.4g} still within {TOL:.4g}; the balance "
        f"gate is blind to this channel"
    )
    with pytest.raises(AssertionError):  # (3) the gate turns red
        balance_check(corrupted)
    assert corrupted_err > clean_err, (
        f"corruption did not degrade closure: {corrupted_err:.4g} <= {clean_err:.4g}"
    )
    return corrupted_err


# ---------------------------------------------------------------------------
# Positive control: the uncorrupted ledger must close
# ---------------------------------------------------------------------------

def test_positive_control_ledger_closes():
    """The uncorrupted energy/fracture ledger must close to <0.5 %."""
    sim = _run_indentation()
    scale = _energy_scale(sim)
    assert scale > 0.0, "energy scale must be non-trivial"
    rel_err = balance_check(sim)
    assert rel_err < TOL, (
        f"uncorrupted ledger residual {rel_err:.4g} exceeds {TOL:.4g}"
    )


# ---------------------------------------------------------------------------
# Negative controls: deliberately corrupt one channel and require the balance
# gate to reject the data.  Each test is green exactly when the gate is
# sensitive to that channel.
# ---------------------------------------------------------------------------

def test_negative_fracture_release_zeroed():
    """Zeroing fracture_release_J must break the balance."""
    sim = _run_indentation()
    err = _assert_detects(sim, "fracture_release_J", factor=0.0)
    assert err > TOL


def test_negative_fracture_release_tripled():
    """Tripling fracture_release_J must break the balance."""
    sim = _run_indentation()
    err = _assert_detects(sim, "fracture_release_J", factor=3.0)
    assert err > TOL


def test_negative_pair_contact_zeroed():
    """Zeroing pair_contact_acquired_J must break the balance."""
    sim = _run_indentation()
    err = _assert_detects(sim, "pair_contact_acquired_J", factor=0.0)
    assert err > TOL


def test_negative_pair_contact_tripled():
    """Tripling pair_contact_acquired_J must break the balance."""
    sim = _run_indentation()
    err = _assert_detects(sim, "pair_contact_acquired_J", factor=3.0)
    assert err > TOL


def test_negative_tool_work_zeroed():
    """Zeroing tool_work_J must break the balance."""
    sim = _run_indentation()
    err = _assert_detects(sim, "tool_work_J", factor=0.0)
    assert err > TOL


def test_negative_damping_dissipation_zeroed():
    """Zeroing damping_dissipation_J must break the balance."""
    sim = _run_indentation()
    err = _assert_detects(sim, "damping_dissipation_J", factor=0.0)
    assert err > TOL


def test_negative_external_work_fault_detected():
    """A fault in external_work_J must break the balance.

    ``external_work_J`` is identically zero in a bare tool-indentation run
    (no external drive is applied), so a multiplicative sign flip would be a
    no-op and the control would be vacuous.  We therefore inject an additive
    fault with the magnitude of the tool-work channel, which is a fault of the
    same order as the other controls.
    """
    sim = _run_indentation()
    assert sim.accounting["external_work_J"] == 0.0, (
        "external_work_J is expected to be dormant in this scenario; if it is "
        "driven, use a multiplicative sign flip instead of the additive fault"
    )
    fault = sim.accounting["tool_work_J"]
    err = _assert_detects(sim, "external_work_J", offset=fault)
    assert err > 0.1 * TOL


def test_negative_control_guard_rejects_vacuous_corruption():
    """Guard test: a sign flip on a dormant channel is refused as vacuous.

    This pins down the flaw that reduced the original
    ``test_negative_external_work_negated`` control to a no-op: multiplying a
    zero channel by -1 cannot change the ledger, so it proves nothing.
    """
    sim = _run_indentation()
    with pytest.raises(AssertionError, match="vacuous"):
        _assert_detects(sim, "external_work_J", factor=-1.0)


# ---------------------------------------------------------------------------
# Self-evidence: document the negative-test commands and expected output
# ---------------------------------------------------------------------------

def test_negative_self_evidence():
    """Documentation test: proves the negative controls exist and are runnable.

    Negative-evidence commands:
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
            tests/test_dem2d_energy_fracture_ledger_negative.py -q

    Expected (green) output:
        10 passed

    The negative result is asserted *inside* the tests: for every corrupted
    channel ``balance_check`` is required to raise (``pytest.raises``), so a
    green run is proof that the gate is sensitive to every channel.
    """
    sim = _run_indentation()
    scale = _energy_scale(sim)
    assert scale > 0.0, "energy scale must be non-trivial for negative tests"
    assert sim.broken_bonds > 0, "must have broken bonds for fracture channels"
    # the four energy channels that carry this scenario must be non-trivial
    for channel in ("tool_work_J", "damping_dissipation_J", "fracture_release_J",
                    "pair_contact_acquired_J"):
        assert sim.accounting[channel] > 0.0, f"{channel} must be active"