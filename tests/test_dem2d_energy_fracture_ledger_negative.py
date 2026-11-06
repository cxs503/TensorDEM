"""Negative-control tests for the 2-D energy/fracture ledger balance.

These tests prove that the ledger balance identity is **sensitive** to
deliberate corruption of individual channels: when a channel is tampered
with, the balance check must reject the corrupted ledger.  This is the
complement of ``test_accounting_ledger_validation.py`` which proves the
*uncorrupted* ledger closes.

Scenario: tool-indentation run that breaks bonds (non-trivial fracture
release + pair-contact acquired).  The energy balance identity is:

    dE_mech - W_tool - W_external + Q_damping + E_fracture - E_pair_contact == 0

Design note (why this file is green, not red)
---------------------------------------------
The negative controls are expressed as a *rejection* assertion, not as a
deliberately failing assertion::

    assert not _closes(corrupted)      # the check must catch the fault

An earlier revision of this file asserted ``_closes(corrupted)`` in order to
be red on purpose, which made ``python -m pytest -q`` fail on every run and
blocked CI.  Encoding the expected outcome as a predicate keeps the same
evidence (the corrupted ledger is rejected) while the suite stays green, and
it fails exactly when the sensitivity is lost: if the tolerance were loosened
or the corruption became a no-op, ``_closes(corrupted)`` would return true and
these tests would turn red.  ``scripts/negative_controls_m4804.py`` re-proves
that by mutating this file.

Sensitivity is quantified, not just asserted: the observed relative residual
of each corrupted ledger must equal the analytic residual induced by the
tampered channel value (no free parameters), and must exceed the closure
tolerance by a documented margin.
"""
from __future__ import annotations

import copy
import functools
import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensordem import DEMConfig, IceDEM  # noqa: E402

#: closure tolerance of the ledger balance (0.5 % of the energy scale)
TOL = 5.0e-3

#: simulated seconds for the indentation run
DURATION = 0.12
LATTICE = (7, 3)

#: per-particle y load [N] for the scenario carrying non-zero external work
LOAD_Y_N = -0.05

#: sign of every accounting channel in the balance identity
#:      dE - W_tool - W_external + Q_damping + E_fracture - E_pair_contact
CHANNEL_SIGN = {
    "tool_work_J": -1.0,
    "external_work_J": -1.0,
    "damping_dissipation_J": +1.0,
    "fracture_release_J": +1.0,
    "pair_contact_acquired_J": -1.0,
}

#: channels used as the reference scale of the balance
SCALE_CHANNELS = ("tool_work_J", "damping_dissipation_J", "fracture_release_J")


# ---------------------------------------------------------------------------
# Simulation harness
# ---------------------------------------------------------------------------

def _config() -> DEMConfig:
    nx, ny = LATTICE
    dt0 = IceDEM(DEMConfig(nx=nx, ny=ny, tool_speed=0.2)).dt
    return DEMConfig(nx=nx, ny=ny, tool_speed=0.2, dt=dt0)


@functools.lru_cache(maxsize=None)
def _baseline_energy_J() -> float:
    """Mechanical energy of the unloaded lattice at t = 0."""
    return IceDEM(_config()).mechanical_energy()["mechanical_J"]


def _run_indentation(load_y: float | None = None) -> IceDEM:
    """Run a tool-indentation simulation that breaks bonds."""
    sim = IceDEM(_config())
    loads = None
    if load_y is not None:
        loads = torch.zeros_like(sim.positions)
        loads[:, 1] = load_y
    steps = int(round(DURATION / sim.dt))
    for _ in range(steps):
        sim.step(external_forces=loads)
    assert sim.broken_bonds > 0, "indentation must break bonds for this test"
    return sim


# ---------------------------------------------------------------------------
# The balance check shared by the positive and the negative controls
# ---------------------------------------------------------------------------

def _energy_residual(sim: IceDEM) -> float:
    """dE_mech - W_tool - W_external + Q_damping + E_fracture - E_pair_contact."""
    acc = sim.accounting
    dE = sim.mechanical_energy()["mechanical_J"] - _baseline_energy_J()
    return (
        dE
        - acc["tool_work_J"]
        - acc["external_work_J"]
        + acc["damping_dissipation_J"]
        + acc["fracture_release_J"]
        - acc["pair_contact_acquired_J"]
    )


def _energy_scale(sim: IceDEM) -> float:
    """Reference scale for the energy balance."""
    acc = sim.accounting
    return sum(abs(acc[name]) for name in SCALE_CHANNELS)


def _relative_residual(sim: IceDEM) -> float:
    return abs(_energy_residual(sim)) / max(_energy_scale(sim), 1e-18)


def _closes(sim: IceDEM, tol: float = TOL) -> bool:
    """The ledger balance predicate under test."""
    return _relative_residual(sim) < tol


def _corrupt(sim: IceDEM, channel: str, factor: float) -> IceDEM:
    """Return a deep copy of ``sim`` with one ledger channel multiplied."""
    corrupted = copy.deepcopy(sim)
    corrupted.accounting[channel] *= factor
    return corrupted


def _sensitivity_margin(sim: IceDEM, channel: str, factor: float) -> float:
    """Prove the balance check rejects a corrupted ledger; return margin.

    The margin is the corrupted ledger's relative residual divided by the
    closure tolerance (``> 1`` means the fault is detected).

    The residual is also compared with the analytic value induced by the
    tampering, ``sign * (factor - 1) * value``, so the test pins *which*
    channel broke the balance rather than accepting any non-zero residual.
    """
    corrupted = _corrupt(sim, channel, factor)
    pristine = _energy_residual(sim)
    delta = CHANNEL_SIGN[channel] * (factor - 1.0) * sim.accounting[channel]
    expected = abs(pristine + delta) / max(_energy_scale(corrupted), 1e-18)
    observed = _relative_residual(corrupted)
    assert observed == pytest.approx(expected, rel=1e-9, abs=1e-15), (
        f"{channel} × {factor}: residual {observed:.6g} does not match the "
        f"analytic value {expected:.6g} induced by the tampered channel"
    )
    # The negative control: the corrupted ledger must be rejected.
    assert not _closes(corrupted), (
        f"corrupted ledger ({channel} × {factor}) still closes at "
        f"{observed:.4g} < tol {TOL:.4g} — the balance check is insensitive"
    )
    return observed / TOL


# ---------------------------------------------------------------------------
# Scenario fixtures (module scoped: the sims are read-only, corruption works
# on deep copies, so one run per scenario is enough for all controls)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def indentation() -> IceDEM:
    """Tool indentation without external load (W_external == 0)."""
    return _run_indentation()


@pytest.fixture(scope="module")
def loaded_indentation() -> IceDEM:
    """Tool indentation with a uniform nodal load (W_external != 0)."""
    return _run_indentation(LOAD_Y_N)


# ---------------------------------------------------------------------------
# Positive controls: the uncorrupted ledger must close
# ---------------------------------------------------------------------------

def test_positive_control_ledger_closes(indentation: IceDEM) -> None:
    """The uncorrupted energy/fracture ledger must close to <0.5 %."""
    assert _energy_scale(indentation) > 0.0, "energy scale must be non-trivial"
    rel_err = _relative_residual(indentation)
    assert _closes(indentation), f"uncorrupted ledger residual {rel_err:.4g} exceeds {TOL:.4g}"


def test_positive_control_ledger_closes_with_external_load(
    loaded_indentation: IceDEM,
) -> None:
    """The uncorrupted ledger must also close when external work is applied."""
    external_work = loaded_indentation.accounting["external_work_J"]
    assert external_work != 0.0, "the loaded scenario must accumulate external work"
    rel_err = _relative_residual(loaded_indentation)
    assert _closes(loaded_indentation), (
        f"loaded uncorrupted ledger residual {rel_err:.4g} exceeds {TOL:.4g}"
    )
    # ...and the external channel must carry enough weight for its own
    # negative control (sign flip) to be detectable.
    assert abs(external_work) > 0.01 * _energy_scale(loaded_indentation)


# ---------------------------------------------------------------------------
# Negative controls: corrupt one channel, the balance check must reject it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "channel,factor",
    [
        ("fracture_release_J", 0.0),      # missing fracture release
        ("fracture_release_J", 3.0),      # inflated fracture release
        ("fracture_release_J", -1.0),     # sign-flipped fracture release
        ("pair_contact_acquired_J", 0.0),  # missing pair-contact credit
        ("pair_contact_acquired_J", 3.0),  # inflated pair-contact credit
        ("tool_work_J", 0.0),             # missing tool work
        ("tool_work_J", 3.0),             # inflated tool work
        ("damping_dissipation_J", 0.0),   # missing damping dissipation
        ("damping_dissipation_J", 3.0),   # inflated damping dissipation
    ],
    ids=lambda value: f"{value}" if isinstance(value, (int, float)) else value,
)
def test_negative_channel_corruption_is_detected(
    indentation: IceDEM, channel: str, factor: float
) -> None:
    """A corrupted channel must break the balance by a clear margin."""
    assert indentation.accounting[channel] != 0.0, (
        f"{channel} is zero in this scenario — the control would be vacuous"
    )
    margin = _sensitivity_margin(indentation, channel, factor)
    assert margin >= 3.0, (
        f"{channel} × {factor} is caught with margin {margin:.3g}×tol only — "
        "too close to the tolerance to be a meaningful control"
    )


def test_negative_external_work_sign_flip_is_detected(
    loaded_indentation: IceDEM,
) -> None:
    """Sign-flipping external work must break the balance (loaded scenario)."""
    margin = _sensitivity_margin(loaded_indentation, "external_work_J", -1.0)
    assert margin >= 3.0, (
        f"external_work_J sign flip caught with margin {margin:.3g}×tol only"
    )


# ---------------------------------------------------------------------------
# Honest scope limits: which controls this scenario can and cannot exercise
# ---------------------------------------------------------------------------

def test_external_work_sign_flip_is_vacuous_without_load(
    indentation: IceDEM,
) -> None:
    """Documented limitation of the unloaded scenario.

    With no external load ``external_work_J`` is identically zero, so
    multiplying it by -1 changes nothing and the sign-flip control carries no
    evidence there.  That is why it is exercised on ``loaded_indentation``.
    """
    assert indentation.accounting["external_work_J"] == 0.0
    corrupted = _corrupt(indentation, "external_work_J", -1.0)
    assert _energy_residual(corrupted) == _energy_residual(indentation)
    assert _sensitivity_margin(indentation, "pair_contact_acquired_J", 3.0) >= 3.0


# ---------------------------------------------------------------------------
# Self-evidence: the commands behind this file and the margins they yield
# ---------------------------------------------------------------------------

def test_negative_self_evidence(
    indentation: IceDEM, loaded_indentation: IceDEM
) -> None:
    """Record the sensitivity margins of every control (green by design).

    Reproduce with::

        TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
            tests/test_dem2d_energy_fracture_ledger_negative.py -q

    Expected: all tests pass; every negative control reports a margin > 3×tol
    (i.e. the corrupted ledger is rejected), and the positive controls close
    at < 0.5 % of the energy scale.  ``scripts/negative_controls_m4804.py``
    mutates this file and requires it to turn red, proving the controls are
    not vacuous.

    Reference values on the CI lattice (7×3 lattice, 0.12 s, no load;
    energy scale 0.1291 J, pristine relative residual 2.80e-3):

        channel                  corruption  rel. residual  margin/tol
        fracture_release_J       ×0          0.1117         22.3
        fracture_release_J       ×3          0.1731         34.6
        fracture_release_J       ×-1         0.2032         40.6
        pair_contact_acquired_J  ×0          0.0400          8.0
        pair_contact_acquired_J  ×3          0.0715         14.3
        tool_work_J              ×0          3.9882        797.6
        tool_work_J              ×3          0.6140        122.8
        damping_dissipation_J    ×0          0.1056         21.1
        damping_dissipation_J    ×3          0.1663         33.3
        external_work_J (loaded) ×-1         0.0328          6.6

    The same numbers are *asserted analytically* (residual = pristine +
    sign·(factor-1)·value) by each control, not copied from this table.
    """
    margins = {
        (channel, factor): _sensitivity_margin(indentation, channel, factor)
        for channel, factor in (
            ("fracture_release_J", 0.0),
            ("pair_contact_acquired_J", 0.0),
            ("tool_work_J", 0.0),
            ("damping_dissipation_J", 0.0),
        )
    }
    margins[("external_work_J", -1.0)] = _sensitivity_margin(
        loaded_indentation, "external_work_J", -1.0
    )
    for channel, factor in (
        ("fracture_release_J", 3.0),
        ("pair_contact_acquired_J", 3.0),
    ):
        corrupted = _corrupt(indentation, channel, factor)
        margins[(channel, factor)] = _relative_residual(corrupted) / TOL
    assert min(margins.values()) >= 3.0, margins
    assert _closes(indentation) and _closes(loaded_indentation)
    assert _energy_scale(indentation) > 0.0