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

The second half of the module (m4804) quantifies *how much* fault each
control detects.  The gate is linear in the ledger channels, so for every
channel the set of accepted channel values is a closed-form interval

    |r0 + s_ch * (x - x0)| <= tol * S0   ->   x in [x_lo, x_hi]

with ``r0`` the clean residual, ``S0`` the clean reference scale and ``s_ch``
the signed coefficient of the channel in the identity.  The minimal
detectable fault is therefore the distance from ``x0`` to the nearer edge of
that interval -- no fitting, no free parameters.  ``min_detectable_fault``
returns it in closed form, ``bisect_min_detectable`` re-finds the same
boundary by bisection on the gate alone, and the tests require the two to
agree to 1e-9 relative.  That turns the shipped negative controls from "the
gate turned red once" into "the gate is red for a fault at least M times the
smallest fault it could possibly resolve".
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
TOOL_SPEED = 0.2

#: relative tolerance of the balance identity (same as the positive control)
TOL = 0.005

#: energy channels with their signed coefficient in
#: ``dE - W_tool - W_ext + Q_damping + E_fracture - E_pair_contact``
SIGNS: dict[str, float] = {
    "tool_work_J": -1.0,
    "external_work_J": -1.0,
    "damping_dissipation_J": 1.0,
    "fracture_release_J": 1.0,
    "pair_contact_acquired_J": -1.0,
}

#: channels that carry this scenario (``external_work_J`` is dormant)
ACTIVE_CHANNELS = (
    "tool_work_J",
    "damping_dissipation_J",
    "fracture_release_J",
    "pair_contact_acquired_J",
)

#: module-level memo: one 0.12 s run serves the whole module (read-only)
_CACHE: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Scenario and ledger primitives
# ---------------------------------------------------------------------------

def _config() -> DEMConfig:
    nx, ny = LATTICE
    dt0 = IceDEM(DEMConfig(nx=nx, ny=ny, tool_speed=TOOL_SPEED)).dt
    return DEMConfig(nx=nx, ny=ny, tool_speed=TOOL_SPEED, dt=dt0)


def _run_indentation() -> IceDEM:
    """Run a tool-indentation simulation that breaks bonds.

    Memoised: the negative-control and margin tests only *read* the returned
    object (tampering always goes through ``copy.deepcopy``).
    """
    cached = _CACHE.get("sim")
    if cached is not None:
        return cached  # type: ignore[return-value]
    sim = IceDEM(_config())
    for _ in range(int(round(DURATION / sim.dt))):
        sim.step()
    assert sim.broken_bonds > 0, "indentation must break bonds for this test"
    _CACHE["sim"] = sim
    return sim


def _clean() -> IceDEM:
    """The uncorrupted reference run."""
    return _run_indentation()


def _dE(sim: IceDEM) -> float:
    """Change of mechanical energy over the run (independent of accounting)."""
    energy0 = _CACHE.get("energy0")
    if energy0 is None:
        energy0 = IceDEM(_config()).mechanical_energy()["mechanical_J"]
        _CACHE["energy0"] = energy0
    return sim.mechanical_energy()["mechanical_J"] - float(energy0)


def _residual_of(accounting: dict, dE: float) -> float:
    """Residual of the balance identity for an arbitrary channel dict."""
    residual = dE
    for channel, sign in SIGNS.items():
        residual += sign * float(accounting[channel])
    return residual


def _energy_residual(sim: IceDEM) -> float:
    """Compute the energy balance residual from the ledger.

    dE_mech - W_tool - W_external + Q_damping + E_fracture - E_pair_contact
    """
    return _residual_of(sim.accounting, _dE(sim))


def _energy_scale(sim: IceDEM) -> float:
    """Reference scale for the energy balance."""
    acc = sim.accounting
    return (
        abs(acc["tool_work_J"])
        + abs(acc["damping_dissipation_J"])
        + abs(acc["fracture_release_J"])
    )


def _rel_err(sim: IceDEM) -> float:
    """Relative ledger residual of ``sim`` (scale taken from ``sim``)."""
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


# ---------------------------------------------------------------------------
# Closed-form detection margins (m4804)
# ---------------------------------------------------------------------------

def acceptance_band(sim: IceDEM, channel: str, tol: float = TOL) -> tuple[float, float]:
    """Channel values that the balance gate accepts, in closed form.

    The gate accepts data when ``|r0 + s_ch * (x - x0)| <= tol * S0``.  Solve
    that pair of linear inequalities for ``x``; no iteration is involved.
    """
    r0 = _energy_residual(sim)
    budget = tol * _energy_scale(sim)
    sign = SIGNS[channel]
    x0 = float(sim.accounting[channel])
    lo = (-budget - r0) / sign
    hi = (budget - r0) / sign
    if lo > hi:
        lo, hi = hi, lo
    return x0 + lo, x0 + hi


def min_detectable_fault(
    sim: IceDEM, channel: str, mode: str = "additive", tol: float = TOL
) -> dict:
    """Smallest single-channel fault the gate can resolve, in closed form.

    ``mode="additive"``: the fault is an offset in J; ``mode="multiplicative"``:
    the fault is the factor ``alpha`` applied to the channel value and its size
    is reported as ``|alpha - 1|``.

    Returns ``size`` (smallest detectable fault), ``direction`` (+1 = grow,
    -1 = shrink), ``bounds`` (accepted fault interval), ``reachable`` and
    ``reason``.  A dormant (identically zero) channel has no multiplicative
    control: scaling zero changes nothing, so ``reachable=False``.
    """
    x0 = float(sim.accounting[channel])
    band_lo, band_hi = acceptance_band(sim, channel, tol)
    if mode == "additive":
        lo, hi = band_lo - x0, band_hi - x0
        if not (lo <= 0.0 <= hi):
            return {
                "channel": channel,
                "mode": mode,
                "size": 0.0,
                "direction": 1.0,
                "bounds": (lo, hi),
                "reachable": True,
                "reason": "clean ledger already outside the band (|r0| > tol*S0)",
            }
        size, direction = min((abs(lo), -1.0), (abs(hi), +1.0))
        return {
            "channel": channel,
            "mode": mode,
            "size": size,
            "direction": direction,
            "bounds": (lo, hi),
            "reachable": True,
            "reason": "additive fault offsets the channel by J",
        }
    if mode != "multiplicative":
        raise ValueError(f"unknown fault mode {mode!r}")
    if x0 == 0.0:
        return {
            "channel": channel,
            "mode": mode,
            "size": float("nan"),
            "direction": 0.0,
            "bounds": (float("nan"), float("nan")),
            "reachable": False,
            "reason": "channel is dormant (x0 == 0): scaling it is a no-op",
        }
    lo, hi = band_lo / x0, band_hi / x0
    if lo > hi:
        lo, hi = hi, lo
    if not (lo <= 1.0 <= hi):
        return {
            "channel": channel,
            "mode": mode,
            "size": 0.0,
            "direction": 1.0,
            "bounds": (lo, hi),
            "reachable": True,
            "reason": "clean ledger already outside the band (|r0| > tol*S0)",
        }
    candidates = []
    if lo > 0.0:
        candidates.append((abs(lo - 1.0), -1.0))
    if hi > 0.0:
        candidates.append((abs(hi - 1.0), +1.0))
    size, direction = min(candidates)
    return {
        "channel": channel,
        "mode": mode,
        "size": size,
        "direction": direction,
        "bounds": (lo, hi),
        "reachable": True,
        "reason": "multiplicative fault scales the channel",
    }


def fault_size(factor: float | None, offset: float | None) -> tuple[float, str]:
    """Size of an injected fault in the metric its mode uses."""
    if factor is not None:
        return abs(factor - 1.0), "multiplicative"
    assert offset is not None
    return abs(offset), "additive"


def detects(
    sim: IceDEM,
    channel: str,
    factor: float | None = None,
    offset: float | None = None,
    tol: float = TOL,
    scale: str = "clean",
) -> bool:
    """Does the balance gate reject this single-channel fault?

    ``scale="clean"`` pins ``S0`` to the uncorrupted run (the convention the
    closed-form band is derived under); ``scale="data"`` reproduces the shipped
    gate convention, where the scale is recomputed from the data.
    """
    corrupted = _corrupt(sim, channel, factor=factor, offset=offset)
    reference = _energy_scale(sim) if scale == "clean" else _energy_scale(corrupted)
    return abs(_energy_residual(corrupted)) / max(reference, 1e-18) > tol


def _detects_magnitude(
    sim: IceDEM,
    channel: str,
    mode: str,
    direction: float,
    magnitude: float,
    tol: float,
    scale: str,
) -> bool:
    """Apply a fault of ``magnitude`` in ``direction`` and query the gate."""
    if mode == "additive":
        return detects(sim, channel, offset=direction * magnitude, tol=tol, scale=scale)
    return detects(sim, channel, factor=1.0 + direction * magnitude, tol=tol, scale=scale)


def bisect_min_detectable(
    sim: IceDEM,
    channel: str,
    mode: str = "additive",
    tol: float = TOL,
    scale: str = "clean",
    iterations: int = 200,
) -> float:
    """Locate the gate boundary by bisection alone (no closed form used).

    The closed-form result only picks *which* side of the accepted interval is
    nearer; the boundary itself is found by bisecting the gate's boolean
    answer, so agreement between the two is a genuine cross-check.
    """
    analytic = min_detectable_fault(sim, channel, mode, tol)
    if not analytic["reachable"]:
        return float("nan")
    direction = analytic["direction"]
    hi = 2.0 * analytic["size"]
    for _ in range(60):  # guarantee a detecting bracket
        if _detects_magnitude(sim, channel, mode, direction, hi, tol, scale):
            break
        hi *= 2.0
    else:  # pragma: no cover - the gate is linear, this cannot happen
        raise AssertionError(f"no detecting bracket found for {channel!r}")
    lo = 0.0
    assert not _detects_magnitude(sim, channel, mode, direction, lo, tol, scale)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if _detects_magnitude(sim, channel, mode, direction, mid, tol, scale):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def shipped_faults(sim: IceDEM) -> list[dict]:
    """The faults actually asserted by the negative controls in this module."""
    return [
        {"channel": "fracture_release_J", "factor": 0.0, "offset": None},
        {"channel": "fracture_release_J", "factor": 3.0, "offset": None},
        {"channel": "pair_contact_acquired_J", "factor": 0.0, "offset": None},
        {"channel": "pair_contact_acquired_J", "factor": 3.0, "offset": None},
        {"channel": "tool_work_J", "factor": 0.0, "offset": None},
        {"channel": "damping_dissipation_J", "factor": 0.0, "offset": None},
        # dormant channel: additive fault of the tool-work magnitude
        {"channel": "external_work_J", "factor": None,
         "offset": float(sim.accounting["tool_work_J"])},
    ]


def fault_margin(sim: IceDEM, fault: dict, tol: float = TOL) -> dict:
    """Margin of one shipped fault over the smallest detectable fault."""
    size, mode = fault_size(fault["factor"], fault["offset"])
    threshold = min_detectable_fault(sim, fault["channel"], mode, tol)
    assert threshold["reachable"], f"no {mode} control exists for {fault['channel']!r}"
    return {
        "channel": fault["channel"],
        "mode": mode,
        "factor": fault["factor"],
        "offset_J": fault["offset"],
        "fault_size": size,
        "min_detectable": threshold["size"],
        "margin": size / threshold["size"],
        "detected_clean_scale": detects(
            sim, fault["channel"], factor=fault["factor"],
            offset=fault["offset"], tol=tol,
        ),
        "detected_shipped_gate": detects(
            sim, fault["channel"], factor=fault["factor"],
            offset=fault["offset"], tol=tol, scale="data",
        ),
    }


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
# Detection margins: the gate's closed-form resolution (m4804)
# ---------------------------------------------------------------------------

def test_acceptance_band_contains_the_clean_value():
    """The clean ledger sits strictly inside the acceptance band."""
    sim = _clean()
    r0 = _energy_residual(sim)
    budget = TOL * _energy_scale(sim)
    assert budget > 0.0
    assert abs(r0) < budget, f"clean residual {r0:.4g} outside budget {budget:.4g}"
    for channel in SIGNS:
        lo, hi = acceptance_band(sim, channel)
        x0 = float(sim.accounting[channel])
        assert lo <= x0 <= hi, f"{channel} clean value {x0} outside band ({lo}, {hi})"


def test_acceptance_half_width_equals_the_tolerance_budget():
    """Every channel's band is exactly ``2 * tol * S0`` wide (|s_ch| == 1)."""
    sim = _clean()
    budget = TOL * _energy_scale(sim)
    for channel in SIGNS:
        lo, hi = acceptance_band(sim, channel)
        assert abs((hi - lo) - 2.0 * budget) <= 1e-12 * max(2.0 * budget, 1.0), (
            f"{channel}: band width {hi - lo:.12g} != 2*budget {2 * budget:.12g}"
        )


def test_min_detectable_additive_fault_equals_headroom():
    """Additive resolution is the tolerance headroom ``tol*S0 - |r0|``."""
    sim = _clean()
    headroom = TOL * _energy_scale(sim) - abs(_energy_residual(sim))
    assert headroom > 0.0
    for channel in SIGNS:
        fault = min_detectable_fault(sim, channel, "additive")
        assert fault["reachable"]
        assert abs(fault["size"] - headroom) <= 1e-12 * max(headroom, 1.0), (
            f"{channel}: closed-form size {fault['size']:.12g} != headroom "
            f"{headroom:.12g}"
        )


def test_min_detectable_multiplicative_fault_equals_headroom_over_value():
    """Multiplicative resolution is ``(tol*S0 - |r0|) / |x_ch|`` (shrink side)."""
    sim = _clean()
    headroom = TOL * _energy_scale(sim) - abs(_energy_residual(sim))
    for channel in ACTIVE_CHANNELS:
        x0 = float(sim.accounting[channel])
        fault = min_detectable_fault(sim, channel, "multiplicative")
        assert fault["reachable"]
        expected = headroom / abs(x0)
        assert abs(fault["size"] - expected) <= 1e-9 * expected, (
            f"{channel}: |alpha-1| {fault['size']:.12g} != headroom/|x| "
            f"{expected:.12g}"
        )


def test_dormant_channel_has_no_multiplicative_control():
    """A zero channel cannot be corrupted multiplicatively; the API says so."""
    sim = _clean()
    assert sim.accounting["external_work_J"] == 0.0
    fault = min_detectable_fault(sim, "external_work_J", "multiplicative")
    assert fault["reachable"] is False
    assert "dormant" in fault["reason"]
    additive = min_detectable_fault(sim, "external_work_J", "additive")
    assert additive["reachable"] is True
    assert additive["size"] > 0.0


@pytest.mark.parametrize("channel", sorted(SIGNS))
def test_bisection_agrees_with_closed_form_boundary(channel: str):
    """Bisecting the gate finds the same boundary as the closed form (1e-9)."""
    sim = _clean()
    for mode in ("additive", "multiplicative"):
        analytic = min_detectable_fault(sim, channel, mode)
        if not analytic["reachable"]:
            continue  # dormant channel: no multiplicative control exists
        numeric = bisect_min_detectable(sim, channel, mode)
        gap = abs(numeric - analytic["size"])
        assert gap <= 1e-9 * analytic["size"], (
            f"{channel}/{mode}: bisection {numeric:.12g} vs closed form "
            f"{analytic['size']:.12g} (gap {gap:.3g})"
        )


@pytest.mark.parametrize("channel", sorted(SIGNS))
def test_gate_is_a_knife_edge_at_the_analytic_boundary(channel: str):
    """At the closed-form boundary the residual equals the tolerance."""
    sim = _clean()
    budget = TOL * _energy_scale(sim)
    for mode in ("additive", "multiplicative"):
        fault = min_detectable_fault(sim, channel, mode)
        if not fault["reachable"]:
            continue  # dormant channel: no multiplicative control exists
        magnitude = fault["size"]
        direction = fault["direction"]
        if mode == "additive":
            factor, offset = None, direction * magnitude
        else:
            factor, offset = 1.0 + direction * magnitude, None
        corrupted = _corrupt(sim, channel, factor=factor, offset=offset)
        residual = abs(_energy_residual(corrupted))
        assert abs(residual - budget) <= 1e-9 * budget, (
            f"{channel}/{mode}: residual at boundary {residual:.12g} != "
            f"budget {budget:.12g}"
        )
        # the boundary itself is ambiguous to 1 ulp, so probe 1e-9 inside it
        inside = magnitude * (1.0 - 1e-9)
        if mode == "additive":
            near = {"offset": direction * inside}
        else:
            near = {"factor": 1.0 + direction * inside}
        assert not detects(sim, channel, **near), (
            f"{channel}/{mode}: gate rejects data 1e-9 inside its boundary"
        )
        # a 1e-6 relative step past the boundary must be rejected
        step = magnitude * (1.0 + 1e-6)
        if mode == "additive":
            past = {"offset": direction * step}
        else:
            past = {"factor": 1.0 + direction * step}
        assert detects(sim, channel, **past), (
            f"{channel}/{mode}: gate blind 1e-6 past its analytic boundary"
        )


def test_shipped_controls_clear_their_own_resolution():
    """Every shipped fault is at least 1x the smallest detectable fault."""
    sim = _clean()
    for fault in shipped_faults(sim):
        margin = fault_margin(sim, fault)
        assert margin["detected_clean_scale"], margin
        assert margin["detected_shipped_gate"], margin
        assert margin["margin"] >= 1.0, (
            f"{margin['channel']} {margin['mode']} fault is below the gate's "
            f"resolution: margin {margin['margin']:.3g}"
        )


def test_margin_is_monotone_in_fault_amplitude():
    """Half the resolution is invisible; twice the resolution is detected."""
    sim = _clean()
    for channel in ACTIVE_CHANNELS:
        for mode in ("additive", "multiplicative"):
            fault = min_detectable_fault(sim, channel, mode)
            direction, size = fault["direction"], fault["size"]
            for multiplier, expect in ((0.5, False), (2.0, True)):
                magnitude = direction * size * multiplier
                if mode == "additive":
                    kwargs = {"offset": magnitude}
                else:
                    kwargs = {"factor": 1.0 + magnitude}
                assert detects(sim, channel, **kwargs) is expect, (
                    f"{channel}/{mode} at {multiplier}x resolution: expected "
                    f"detected={expect}"
                )


def test_headroom_is_linear_in_the_tolerance():
    """The additive resolution scales as ``tol*S0 - |r0|`` (no hidden floor)."""
    sim = _clean()
    scale = _energy_scale(sim)
    r0 = abs(_energy_residual(sim))
    for tol in (TOL, 2.0 * TOL, 4.0 * TOL):
        expected = tol * scale - r0
        assert expected > 0.0
        for channel in SIGNS:
            got = min_detectable_fault(sim, channel, "additive", tol)["size"]
            assert abs(got - expected) <= 1e-12 * max(expected, 1.0), (
                f"tol={tol}: {channel} resolution {got:.12g} != {expected:.12g}"
            )
    # below the clean residual the ledger cannot close at all: any fault
    # (including none) trips the gate, and the resolution is reported as 0
    for tol in (TOL / 4.0, TOL / 2.0):
        assert tol * scale - r0 < 0.0
        for channel in SIGNS:
            fault = min_detectable_fault(sim, channel, "additive", tol)
            assert fault["size"] == 0.0, fault
            assert "already outside" in fault["reason"], fault


def test_scale_convention_gap_is_negligible():
    """Data-derived scale and clean scale give the same resolution to <1 %.

    The shipped gate recomputes ``S`` from the corrupted data, while the
    closed-form band pins ``S0`` to the clean run.  This test measures the
    resulting boundary shift and requires it to stay far below the margin, so
    the analytic margins remain conservative for the shipped gate.
    """
    sim = _clean()
    for channel in ACTIVE_CHANNELS:
        for mode in ("additive", "multiplicative"):
            clean_boundary = bisect_min_detectable(sim, channel, mode, scale="clean")
            data_boundary = bisect_min_detectable(sim, channel, mode, scale="data")
            gap = abs(data_boundary - clean_boundary) / clean_boundary
            assert gap < 0.01, (
                f"{channel}/{mode}: scale convention shifts the boundary by "
                f"{gap:.2%}"
            )


# ---------------------------------------------------------------------------
# Self-evidence: document the negative-test commands and expected output
# ---------------------------------------------------------------------------

def test_negative_self_evidence():
    """Documentation test: proves the negative controls exist and are runnable.

    Negative-evidence command:
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
            tests/test_dem2d_energy_fracture_ledger_negative.py -q

    Expected (green) output:
        29 passed

    The negative result is asserted *inside* the tests: for every corrupted
    channel ``balance_check`` is required to raise (``pytest.raises``), so a
    green run is proof that the gate is sensitive to every channel.  The
    margin tests add the quantitative counterpart: the fault actually shipped
    must exceed the gate's closed-form resolution
    (``test_shipped_controls_clear_their_own_resolution``), and the closed form
    must match an independent bisection of the gate to 1e-9
    (``test_bisection_agrees_with_closed_form_boundary``).
    """
    sim = _clean()
    scale = _energy_scale(sim)
    assert scale > 0.0, "energy scale must be non-trivial for negative tests"
    assert sim.broken_bonds > 0, "must have broken bonds for fracture channels"
    for channel in ACTIVE_CHANNELS:
        assert sim.accounting[channel] > 0.0, f"{channel} must be active"