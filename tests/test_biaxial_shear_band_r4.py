"""Tests for the R4 cumulative-window shear-band driver (no DEM integration).

The R4 benchmark (`benchmarks/biaxial_shear_band/run_r4_cumulative_ensemble.py`)
introduces three pieces of logic that the R3 artefacts did not cover:

* the post-peak window selection used to build the cumulative measurement,
* the particle-bootstrap measurement uncertainty ``u`` (new in R4), and
* the ensemble verdict assembly (pass rule, paired P1, confinement P2).

The last test checks the *mechanism* the round is built on: at a fixed
positional noise level, a cumulative window (large tangential slip) returns a
band inclination with a smaller error than a one-step increment (small slip).
That is the claim the R3 noise audit only predicted and R4 measures on DEM data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("torch")

from tensordem.biaxial_band_diagnostics import (  # noqa: E402
    BandMeasurementError, synthetic_band_configuration)
from tensordem.biaxial_shear_band import measure_band  # noqa: E402

import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_DRIVER = (Path(__file__).resolve().parents[1] / "benchmarks" / "biaxial_shear_band"
           / "run_r4_cumulative_ensemble.py")


def _load_driver():
    spec = importlib.util.spec_from_file_location("r4_driver", _DRIVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["r4_driver"] = module
    spec.loader.exec_module(module)
    return module


r4 = _load_driver()


# ---------------------------------------------------------------------------
# window selection
# ---------------------------------------------------------------------------
def _fake_snapshots(strains):
    return [{"strain": float(s), "positions": np.zeros((4, 2)),
             "window": (0.0, 1.0, 0.0, 1.0)} for s in strains]


def test_late_window_starts_at_four_tenths_post_peak():
    snaps = _fake_snapshots([0.0, 0.02, 0.04, 0.06, 0.08, 0.09, 0.10, 0.11, 0.12])
    late = r4._late_snapshots(snaps, peak_strain=0.08)
    # tail_start = 0.08 + 0.4 * (0.12 - 0.08) = 0.096 -> 0.10, 0.11, 0.12
    assert [round(s["strain"], 3) for s in late] == [0.10, 0.11, 0.12]


def test_degenerate_record_yields_no_pairs():
    """Final strain == peak strain -> the late window holds the peak alone, so
    no increment pair exists and the specimen must report zero pairs."""
    snaps = _fake_snapshots([0.0, 0.01, 0.02])
    late = r4._late_snapshots(snaps, peak_strain=0.02)
    assert [round(s["strain"], 3) for s in late] == [0.02]
    assert list(zip(late[:-1], late[1:])) == []


def test_peak_snapshot_is_the_nearest_strain():
    snaps = _fake_snapshots([0.0, 0.05, 0.09, 0.12])
    assert r4._peak_snapshot(snaps, 0.085)["strain"] == pytest.approx(0.09)


# ---------------------------------------------------------------------------
# bootstrap uncertainty
# ---------------------------------------------------------------------------
def _lattice(nx=20, ny=20, spacing=1.5):
    xs, ys = np.meshgrid(np.arange(nx) * spacing, np.arange(ny) * spacing)
    return np.column_stack([xs.ravel(), ys.ravel()])


def test_bootstrap_is_reproducible_seeded_and_finite():
    pos = _lattice()
    now = synthetic_band_configuration(pos, angle_deg=60.0, slip=0.02,
                                       band_width=1.5)
    a = r4._bootstrap_angle(pos, now, (0.0, 30.0, 0.0, 30.0), seed=7,
                            operator="radon", resamples=6)
    b = r4._bootstrap_angle(pos, now, (0.0, 30.0, 0.0, 30.0), seed=7,
                            operator="radon", resamples=6)
    assert a["angles_deg"] == b["angles_deg"]
    assert a["n_resamples"] == 6
    assert math.isfinite(a["std_deg"]) and a["std_deg"] >= 0.0
    # a bootstrap of a well-resolved band must stay near the true inclination
    assert abs(a["mean_deg"] - 60.0) < 12.0


def test_bootstrap_reports_no_resamples_for_a_tiny_window():
    pos = _lattice()
    now = synthetic_band_configuration(pos, angle_deg=60.0, slip=0.02,
                                       band_width=1.5)
    out = r4._bootstrap_angle(pos, now, (0.0, 3.0, 0.0, 3.0), seed=7,
                              operator="radon", resamples=4)
    assert out["n_resamples"] == 0 and out["angles_deg"] == []


def test_cumulative_window_beats_one_step_at_fixed_positional_noise():
    """The SNR mechanism: 6x more slip at the same noise -> smaller error."""
    pos = _lattice(nx=24, ny=24)
    rng = np.random.default_rng(20261113)
    noise = 0.005 * 1.5            # 0.5 % of the particle spacing
    err_one, err_cum = [], []
    for _ in range(3):
        jitter = rng.standard_normal(pos.shape) * noise
        for slip, bucket in ((0.0033, err_one), (0.02, err_cum)):
            now = synthetic_band_configuration(pos, angle_deg=60.0, slip=slip,
                                               band_width=1.5) + jitter
            theta = measure_band(pos, now)["folded_angle_deg"]
            bucket.append(abs(float(theta) - 60.0))
    rms_one = float(np.sqrt(np.mean(np.square(err_one))))
    rms_cum = float(np.sqrt(np.mean(np.square(err_cum))))
    assert math.isfinite(rms_one) and math.isfinite(rms_cum)
    assert rms_cum < rms_one, (rms_one, rms_cum)


# ---------------------------------------------------------------------------
# verdict assembly
# ---------------------------------------------------------------------------
def _fake_specimen(name, seed, theta_cum, theta_inc, *, u_cum=None, u_inc=None,
                   resolved=True, p=100.0e3, axis="anchor"):
    return {
        "name": name, "axis": axis, "seed": seed, "confining_stress": p,
        "one_step": {"theta_deg": theta_inc, "theta_std_deg": u_inc,
                     "resolved": resolved},
        "cumulative": {"late_window": {"radon": theta_cum, "tensor": theta_cum,
                                       "resolved": resolved,
                                       "bootstrap_radon": {"std_deg": u_cum,
                                                           "n_resamples": 30},
                                       "snr_gain_vs_one_step": 6.0},
                       "post_peak": {"radon": theta_cum}},
        "friction_angle_deg": 35.0, "dilation_angle_deg": 10.0,
        "theory_angles_deg": {"mohr_coulomb": 62.5, "roscoe": 50.0,
                              "arthur": 56.25},
    }


def test_ensemble_stats_ci_and_tolerance():
    out = r4._ensemble_stats([58.0, 60.0, 62.0, 64.0])
    assert out["n"] == 4 and out["mean_deg"] == pytest.approx(61.0)
    assert out["std_deg"] > 0 and out["ci95_deg"][0] < out["mean_deg"] < out["ci95_deg"][1]
    assert out["within_tolerance"] is True
    assert r4._ensemble_stats([80.0, 82.0])["within_tolerance"] is False


def test_verdict_passes_only_with_enough_resolved_specimens_and_tolerance():
    anchors = [_fake_specimen(f"a{i}", i, 58.0, 55.0, u_cum=1.0, u_inc=2.0)
               for i in range(8)]
    verdict = r4._analyse(anchors)
    assert verdict["passed"] is True
    assert verdict["specimens_resolved"] == 8
    # five resolved specimens is below the predeclared 6/8 bar
    anchors[5]["cumulative"]["late_window"]["resolved"] = False
    anchors[6]["cumulative"]["late_window"]["resolved"] = False
    anchors[7]["cumulative"]["late_window"]["resolved"] = False
    assert r4._analyse(anchors)["passed"] is False


def test_verdict_fails_when_mean_leaves_the_tolerance_band():
    anchors = [_fake_specimen(f"a{i}", i, 72.0, 55.0) for i in range(8)]
    verdict = r4._analyse(anchors)
    assert verdict["primary_cumulative_late_window_radon"]["within_tolerance"] is False
    assert verdict["passed"] is False


def test_paired_p1_requires_six_of_eight_improvements():
    good = [_fake_specimen(f"a{i}", i, 58.0, 55.0, u_cum=0.5, u_inc=2.0)
            for i in range(8)]
    assert r4._analyse(good)["p1_snr_paired"]["p1_holds"] is True
    for i in range(3):              # only 5/8 improved -> prediction fails
        good[i]["cumulative"]["late_window"]["bootstrap_radon"]["std_deg"] = 3.0
    p1 = r4._analyse(good)["p1_snr_paired"]
    assert p1["n_improved"] == 5 and p1["p1_holds"] is False


def test_confinement_p2_counts_resolved_specimens():
    anchors = [_fake_specimen(f"a{i}", i, 58.0, 55.0) for i in range(8)]
    sens = [_fake_specimen("p200_s53", 53, 61.0, 72.0, resolved=True,
                           p=200.0e3, axis="confining_pressure"),
            _fake_specimen("p200_s67", 67, 59.0, 70.0, resolved=True,
                           p=200.0e3, axis="confining_pressure"),
            _fake_specimen("p200_s79", 79, 63.0, 71.0, resolved=False,
                           p=200.0e3, axis="confining_pressure")]
    verdict = r4._analyse(anchors + sens)
    assert verdict["p2_confinement"]["n_resolved_cumulative"] == 2
    assert verdict["p2_confinement"]["p2_holds"] is True

def test_run_cached_is_spawn_picklable():
    """The parallel path (mp spawn Pool) requires a module-level runner.

    Regression: cj2546 (2026-11-14) failed with
    ``AttributeError: Can't pickle local object 'main.<locals>._one'``
    because the per-specimen runner was a closure inside ``main``.
    """
    import pickle

    from benchmarks.biaxial_shear_band import run_r4_cumulative_ensemble as r4

    assert pickle.dumps(r4._run_cached) is not None
    assert r4._run_cached.__module__ == r4.__name__
