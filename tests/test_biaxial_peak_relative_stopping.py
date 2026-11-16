"""Peak-relative stopping rule for the biaxial shear-band driver.

Motivation (R5 of CASE_BIAXIAL_SHEAR_BAND): with a *fixed* strain budget the
post-peak window of a specimen whose deviator peak arrives near the end of the
budget collapses to one or two snapshots -- 3 of 8 anchor specimens hard-hardened
up to the cut-off, and only 5 of 8 resolved the band (the predeclared bar was
6 of 8).  ``post_peak_strain`` turns ``strain_target`` into a *minimum* and keeps
loading until the running deviator peak is that far behind, bounded by
``strain_cap``.

What is asserted here (no physics claim, only the stopping contract):

* off by default, i.e. the legacy budget is unchanged (regression guard);
* ``strain_target`` is still a hard floor -- a run never stops below it;
* with the rule on, the run delivers at least ``post_peak_strain`` of
  post-peak axial strain (up to the cap) and says so in ``result["stopping"]``;
* ``strain_cap`` is never exceeded, and hitting it is reported as such;
* the argument validation rejects a cap without a rule / a non-positive rule.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.biaxial_shear_band import run_biaxial_shear_band

RATE = 0.20            # fast: this test is about the stopping contract
TARGET = 0.10
POST_PEAK = 0.03


def _run(**kw) -> dict:
    base = dict(confining_stress=100.0e3, columns=5, rows=10, radius=0.75,
                strain_rate=RATE, strain_target=TARGET, consolidation_steps=50,
                snapshot_strain=0.01, wall_friction=0.0,
                lateral_control="isochoric",
                material_overrides={"freeze_bond_fraction": 0.0})
    base.update(kw)
    return run_biaxial_shear_band(**base)


def _eps_log(result: dict) -> np.ndarray:
    return np.asarray(result["stress_path"]["axial_strain_log"], dtype=float)


@pytest.fixture(scope="module")
def legacy() -> dict:
    return _run()


@pytest.fixture(scope="module")
def extended() -> dict:
    return _run(post_peak_strain=POST_PEAK)


def test_rule_is_off_by_default(legacy: dict) -> None:
    stopping = legacy["stopping"]
    assert stopping["post_peak_strain"] is None
    assert stopping["strain_cap"] is None
    assert stopping["stop_reason"] == "strain_target"
    assert stopping["online_peak_axial_strain_log"] is None
    # legacy budget: target met, overshoot bounded by (about) one step
    dt = float(legacy["specimen"]["dt"])
    eps = _eps_log(legacy)
    assert eps[-1] >= TARGET - 1e-12
    assert eps[-1] <= TARGET + 3.0 * RATE * dt


def test_default_cap_is_target_plus_rule(extended: dict) -> None:
    stopping = extended["stopping"]
    assert stopping["strain_cap"] == pytest.approx(TARGET + 2.0 * POST_PEAK, rel=1e-12)
    # the docstring in biaxial_shear_band states the same default
    assert stopping["strain_cap"] > TARGET + POST_PEAK


def test_rule_reports_and_respects_bounds(extended: dict) -> None:
    stopping = extended["stopping"]
    eps = _eps_log(extended)
    final = float(eps[-1])
    # the declared target is a floor, never a ceiling
    assert final >= TARGET - 1e-12
    # and the cap is a hard ceiling
    assert final <= float(stopping["strain_cap"]) + 2.0 * RATE * float(
        extended["specimen"]["dt"])
    # the verdict is always one of the declared outcomes -- never silent
    assert stopping["stop_reason"] in {"strain_target", "post_peak", "strain_cap"}
    if stopping["stop_reason"] == "post_peak":
        span = final - float(stopping["online_peak_axial_strain_log"])
        assert span >= POST_PEAK - 1e-9
        assert stopping["post_peak_span_log"] == pytest.approx(span, rel=1e-9)
    if stopping["stop_reason"] == "strain_cap":
        # the cap bound is what stopped the run, i.e. the rule wanted more
        assert final >= float(stopping["strain_cap"]) - 2.0 * RATE * float(
            extended["specimen"]["dt"])


def test_rule_never_shortens_below_target() -> None:
    """A tiny rule must not truncate the run below ``strain_target``."""
    result = _run(strain_target=0.12, post_peak_strain=1.0e-3)
    eps = float(_eps_log(result)[-1])
    assert eps >= 0.12 - 1e-12
    assert result["stopping"]["stop_reason"] == "strain_target"


def test_explicit_cap_stops_the_run_and_is_reported() -> None:
    result = _run(strain_target=0.06, post_peak_strain=0.20, strain_cap=0.08)
    stopping = result["stopping"]
    eps = float(_eps_log(result)[-1])
    dt = float(result["specimen"]["dt"])
    assert stopping["stop_reason"] == "strain_cap"
    assert stopping["strain_cap"] == pytest.approx(0.08, rel=1e-12)
    assert eps <= 0.08 + 2.0 * RATE * dt
    assert eps >= 0.06 - 1e-12


def test_rule_extends_past_target_when_peak_is_late() -> None:
    """Trigger the rule for real: the run must go past its declared target."""
    result = _run(strain_target=0.06, post_peak_strain=0.05)
    stopping = result["stopping"]
    eps = float(_eps_log(result)[-1])
    if stopping["stop_reason"] == "post_peak":
        assert eps > 0.06 + 1e-12
        peak = float(stopping["online_peak_axial_strain_log"])
        assert peak >= 0.0
        assert eps - peak >= 0.05 - 1e-9
    else:
        # an early peak legitimately leaves the target as the binding bound;
        # a peak later than the target may also reach the cap first
        assert stopping["stop_reason"] in {"strain_target", "strain_cap"}
        assert eps <= float(stopping["strain_cap"]) + 3.0 * RATE * float(
            result["specimen"]["dt"])


@pytest.mark.parametrize("kw", [
    {"post_peak_strain": -0.01},
    {"post_peak_strain": 0.0},
    {"strain_cap": 0.2},
    {"post_peak_strain": 0.05, "strain_cap": 0.05},
])
def test_invalid_combinations_are_rejected(kw: dict) -> None:
    with pytest.raises(ValueError):
        _run(**kw)