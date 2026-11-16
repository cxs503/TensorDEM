"""Late-window width of the biaxial shear-band driver, and the protocol arithmetic.

Motivation (R6 of CASE_BIAXIAL_SHEAR_BAND, its own report §⑦): the late window
kept the last 60 % of the post-peak record while the peak-relative rule was only
0.03 wide, so the widest window a perfectly working rule could deliver was
0.6 * 0.03 = 0.018 -- below the 0.02 span the round's own prediction required.
The prediction was arithmetically unsatisfiable and carried no physical
information.  Two contracts are asserted here:

* ``late_window_fraction`` is a real driver argument: default 0.6 reproduces the
  published R4/R5/R6 window, the skipped part of the record is ``1 - f``, and
  the value is reported in ``result["band"]`` / ``result["stopping"]``;
* the reportable span ceiling ``f * post_peak_strain``
  (:func:`tensordem.biaxial_shear_band.late_window_span_cap`) is refused
  (``ValueError``) outside ``f in (0, 1]``, so a protocol that cannot reach its
  own declared minimum span is visible *before* a run, not after (the R6 defect).
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.biaxial_shear_band import (DEFAULT_LATE_WINDOW_FRACTION,
                                          late_window_span_cap,
                                          run_biaxial_shear_band)

RATE = 0.20
TARGET = 0.10
POST_PEAK = 0.05


def _run(**kw) -> dict:
    base = dict(confining_stress=100.0e3, columns=5, rows=10, radius=0.75,
                strain_rate=RATE, strain_target=TARGET, consolidation_steps=50,
                snapshot_strain=0.01, wall_friction=0.0,
                lateral_control="isochoric",
                material_overrides={"freeze_bond_fraction": 0.0})
    base.update(kw)
    return run_biaxial_shear_band(**base)


# ---------------------------------------------------------------------------
# protocol arithmetic
# ---------------------------------------------------------------------------
def test_span_cap_is_fraction_times_rule_width() -> None:
    assert late_window_span_cap(0.8, 0.05) == pytest.approx(0.04)
    # the R6 protocol, reproduced: 0.018 < the 0.02 span R6 demanded
    assert late_window_span_cap(0.6, 0.03) == pytest.approx(0.018)
    assert late_window_span_cap(0.6, 0.03) < 0.02
    # the R7 protocol can satisfy the same demand
    assert late_window_span_cap(0.8, 0.05) >= 0.02


def test_span_cap_is_none_without_a_rule() -> None:
    assert late_window_span_cap(0.6, None) is None


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_span_cap_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError):
        late_window_span_cap(fraction, 0.05)


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
def test_driver_rejects_invalid_late_window_fraction(fraction: float) -> None:
    with pytest.raises(ValueError):
        _run(late_window_fraction=fraction)


# ---------------------------------------------------------------------------
# driver contract
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def default_run() -> dict:
    return _run()


@pytest.fixture(scope="module")
def wide_run() -> dict:
    return _run(post_peak_strain=POST_PEAK, late_window_fraction=0.8)


def test_default_fraction_reproduces_the_published_window(default_run: dict) -> None:
    band = default_run["band"]
    assert band["late_window_fraction"] == pytest.approx(DEFAULT_LATE_WINDOW_FRACTION)
    assert DEFAULT_LATE_WINDOW_FRACTION == pytest.approx(0.6)
    stopping = default_run["stopping"]
    assert stopping["late_window_span_cap_log"] is None      # no rule, no ceiling


def test_wide_window_is_reported_and_caps_the_span(wide_run: dict) -> None:
    stopping = wide_run["stopping"]
    assert stopping["late_window_fraction"] == pytest.approx(0.8)
    assert stopping["late_window_span_cap_log"] == pytest.approx(0.8 * POST_PEAK)
    assert wide_run["band"]["late_window_fraction"] == pytest.approx(0.8)
    # the delivered late window never exceeds its own ceiling: the skipped head
    # of the post-peak record is (1 - f) of the span, so the kept part is at
    # most f * post_peak_strain (plus one snapshot spacing of discretisation)
    history = wide_run["band"]["history"]
    strain = np.asarray([b["axial_strain"] for b in history], dtype=float)
    peak = float(wide_run["peak"]["axial_strain"])
    final = float(strain.max())
    tail_start = peak + (1.0 - 0.8) * max(final - peak, 0.0)
    late = strain[strain >= tail_start]
    if late.size > 1:
        delivered = float(late.max() - late.min())
        assert delivered <= stopping["late_window_span_cap_log"] + 0.011


def test_wide_window_keeps_more_of_the_post_peak_record(wide_run: dict,
                                                        default_run: dict) -> None:
    """The kept window is monotone in the fraction (same stopping rule)."""
    def kept(run: dict) -> int:
        hist = np.asarray(run["stress_path"]["axial_strain"], dtype=float)
        peak = float(run["peak"]["axial_strain"])
        final = float(hist[-1])
        tail = peak + (1.0 - run["band"]["late_window_fraction"]) * max(final - peak, 0.0)
        return int(sum(1 for b in run["band"]["history"]
                       if b["axial_strain"] >= tail))

    narrow = _run(post_peak_strain=POST_PEAK, late_window_fraction=0.6)
    assert kept(wide_run) >= kept(narrow)
