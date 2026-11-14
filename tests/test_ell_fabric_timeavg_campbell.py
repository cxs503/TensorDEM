"""R5 tests: steady-window time-averaged fabric statistics (CASE-DEM-SHEAR-CAMPBELL).

What is asserted (invariants, not just "it runs"):

1. ``fabric_sample`` reproduces hand-computed values for a two-disk geometry,
   including the sign convention of the fabric anisotropy.
2. ``summarise_fabric`` forms ``f_norm`` per sample (ratio of per-sample values),
   never as a ratio of averages.
3. The normal/tangential virial decomposition closes sample by sample
   (``tau_normal + tau_tangential == tau_contact``) on a real run.
4. ``fabric_statistics=False`` leaves every previously shipped number bit-identical
   and returns ``fabric=None``.
5. The R5 benchmark artifacts, when present, satisfy the gated-run and tolerance
   bookkeeping they claim.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.elastic_limit_shear import (  # noqa: E402
    ElasticLimitShearCase, fabric_sample, simulate_plane_shear, summarise_fabric)

RUNS = REPO / "benchmarks" / "case_dem_shear_campbell_elastic_limit" / "runs"
LIGHT = dict(n_particles=200, solid_fraction=0.78, kn=1.0e4, shear_rate=1.0,
             strain_total=0.30, strain_average_from=0.05, dt_factor=0.010,
             seed=20261112)


def test_fabric_sample_two_disks_hand_computed():
    # two equal disks of radius 0.5 whose centres are 0.9 along the 45 degree
    # diagonal: overlap ratio = (1.0 - 0.9)/1.0 = 0.1, n = (1,1)/sqrt(2)
    r = np.array([0.5, 0.5])
    off = 0.9 / math.sqrt(2.0)
    x = np.array([0.0, off])
    y = np.array([0.0, off])
    lx = ly = 100.0
    fab = fabric_sample(x, y, r, lx, ly, 0.0, np.array([0]), np.array([1]), 2)
    assert fab["n_contacts"] == 1
    assert fab["coordination"] == pytest.approx(2.0 * 1 / 2)
    assert fab["mean_overlap_ratio"] == pytest.approx(0.1)
    # n_x n_y = 0.5 for a 45 degree contact; physical anisotropy is -<n_x n_y>
    assert fab["fabric_xy"] == pytest.approx(-0.5)
    assert fab["overlap_weighted_fabric_xy"] == pytest.approx(-0.05)  # ov * n_x n_y
    assert fab["a_xy"] == pytest.approx(-0.5)      # ov-weighted / mean ov = -0.5


def test_fabric_sample_axis_aligned_contact_has_zero_anisotropy():
    r = np.array([0.5, 0.5])
    x = np.array([0.0, 0.9])
    y = np.array([0.0, 0.0])
    fab = fabric_sample(x, y, r, 100.0, 100.0, 0.0, np.array([0]), np.array([1]), 2)
    assert fab["fabric_xy"] == pytest.approx(0.0, abs=1e-12)


def test_fabric_sample_no_contact_is_nan_not_zero():
    r = np.array([0.5, 0.5])
    fab = fabric_sample(np.array([0.0, 5.0]), np.array([0.0, 0.0]), r,
                        100.0, 100.0, 0.0, np.array([0]), np.array([1]), 2)
    assert fab["n_contacts"] == 0
    assert math.isnan(fab["a_xy"])


def test_summarise_fabric_forms_f_norm_per_sample():
    # two samples: f_norm must be mean(94/ (94+28), 1/(1+1)) = mean(0.7705, 0.5)
    samples = [
        {"n_contacts": 4, "coordination": 2.0, "mean_overlap_ratio": 0.04,
         "max_overlap_ratio": 0.05, "fabric_xy": -0.1,
         "overlap_weighted_fabric_xy": -0.004, "a_xy": -0.1,
         "tau_normal": 94.0, "tau_tangential": 28.0, "tau_contact": 122.0},
        {"n_contacts": 5, "coordination": 2.5, "mean_overlap_ratio": 0.05,
         "max_overlap_ratio": 0.06, "fabric_xy": -0.2,
         "overlap_weighted_fabric_xy": -0.010, "a_xy": -0.2,
         "tau_normal": 1.0, "tau_tangential": 1.0, "tau_contact": 2.0},
    ]
    s = summarise_fabric(samples, 0.05, 0.30)
    expected = np.mean([94.0 / 122.0, 1.0 / 2.0])
    assert s["f_norm_mean"] == pytest.approx(float(expected))
    assert s["n_samples"] == 2
    assert s["virial_split_residual_max"] == pytest.approx(0.0, abs=1e-12)
    assert s["coordination_std"] == pytest.approx(0.25)


def test_virial_split_and_flag_off_invariance():
    case = ElasticLimitShearCase(**LIGHT)
    on = simulate_plane_shear(case, fabric_statistics=True)
    off = simulate_plane_shear(case)
    # (4) the flag must not perturb a single number of the physics
    assert off["fabric"] is None
    assert on["tau"] == off["tau"]
    assert on["pressure"] == off["pressure"]
    assert on["tau_star"] == off["tau_star"]
    # (3) the decomposition is an identity, sample by sample
    fab = on["fabric"]
    assert fab is not None and fab["n_samples"] > 0
    assert fab["virial_split_residual_max"] < 1e-9
    # and the time-averaged contact stress must agree with the run's own average
    assert fab["tau_contact_mean"] == pytest.approx(on["tau_contact"], rel=1e-9)
    assert 0.0 <= fab["f_norm_mean"] <= 1.0
    assert -0.5 <= fab["fabric_xy_mean"] <= 0.5
    assert fab["coordination_mean"] > 0.0


ART = RUNS / "fabric_timeavg" / "analysis.json"


@pytest.mark.skipif(not ART.exists(), reason="R5 run artifacts not present")
def test_r5_analysis_bookkeeping():
    res = json.load(open(ART))
    assert res["leg_b_n_gated"] >= 7
    assert res["tolerance"] == 0.15
    assert res["virial_split_residual_absmax"] < 1e-9
    # the pre-registered identity leg must hold on every gated run
    assert res["leg_b_factored_residual_absmax"] <= res["tolerance"]
    # every out-of-sample R2 prediction is a real number
    assert res["leg_c_n_points"] >= 10
    for p in res["leg_c_r2_predictions"]:
        assert math.isfinite(p["residual"]) and p["A_meas"] > 0
    # the epilogue is labelled as such and its LOCO number is the strict one
    ex = res["exploratory_amendment"]
    assert "EXPLORATORY" in ex["status"]
    assert ex["loco_n_points"] >= 10
    assert ex["loco_worst_residual"] <= res["tolerance"] or \
        ex["loco_worst_residual"] != ex["loco_worst_residual"]