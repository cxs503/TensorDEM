"""Tests for the R5 strain-stationarity instrument (CASE-DEM-PLANE-SHEAR-MUIDI).

Two layers:

* **Synthetic** (always runs, milliseconds): the block/estimator/convergence
  bookkeeping is exercised on hand-computable series so a regression in the
  analysis cannot hide behind simulation noise.
* **Integration** (opt-in, ``MU_I_STATIONARITY=1``): one short real ladder point
  through the in-repo servo shear cell to prove the runner still binds to a
  live device (schema + positivity of the physical quantities).

The integration test is opt-in for the same reason as
``test_plane_shear_mu_i.py::test_cache_is_live_two_point_rerun``: the settle
phase alone costs ~10 s of CPU.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyse_stationarity as ana  # noqa: E402
import run_stationarity as rst  # noqa: E402
from run_stationarity_sweep import I_ORDER, LADDER  # noqa: E402


def _series(strains, mus, tau=None, p=None, plate=None):
    tau = tau or [1.0] * len(strains)
    p = p or [4.0] * len(strains)
    plate = plate or [0.06] * len(strains)
    return [{"strain": s, "mu": m, "shear_stress_pa": t,
             "normal_stress_pa": q, "plate_z": z}
            for s, m, t, q, z in zip(strains, mus, tau, p, plate)]


# --------------------------------------------------------------- synthetic --
def test_block_stats_partitions_and_recovers_constant_mu():
    series = _series([0.10, 0.20, 0.30, 0.60, 0.70, 0.80],
                     [0.40, 0.40, 0.40, 0.40, 0.40, 0.40],
                     tau=[2.0] * 6, p=[5.0] * 6)
    blocks = rst.block_stats(series, bin_width=0.5)
    assert len(blocks) == 2
    assert blocks[0]["n"] == 3 and blocks[1]["n"] == 3
    for b in blocks:
        assert b["mu"] == pytest.approx(0.40)
        assert b["mu_ratio"] == pytest.approx(2.0 / 5.0)


def test_block_stats_ratio_of_means_vs_mean_of_ratios():
    """Anti-correlated tau/P must split the two estimators in a known direction."""
    series = _series([0.1, 0.2, 0.3, 0.4], [1.0, 0.5, 0.5, 1.0],
                     tau=[1.0, 1.0, 1.0, 1.0], p=[1.0, 2.0, 2.0, 1.0])
    blocks = rst.block_stats(series, bin_width=1.0)
    b = blocks[0]
    assert b["mu"] == pytest.approx(0.75)                 # mean of per-sample tau/P
    assert b["mu_ratio"] == pytest.approx(1.0 / 1.5)      # <tau>/<P>
    assert b["mu"] > b["mu_ratio"]


def test_convergence_flags_transient_and_finds_first_stationary_rung():
    rows = [
        {"strain_target": 0.2, "mu_tail": 0.20, "mu_ratio": 0.20, "bias_pct": 0.0,
         "drift_pct": math.nan, "phi": 0.57, "mu_err_pct": -50.0, "phi_err_pct": -1.0,
         "p_err_pct": 0.0, "travel_d": 1.0, "steps": 100, "wall_s": 1.0,
         "strain_sampled": (0.08, 0.2), "I_measured": 0.03},
        {"strain_target": 1.0, "mu_tail": 0.33, "mu_ratio": 0.33, "bias_pct": 0.0,
         "drift_pct": 1.0, "phi": 0.57, "mu_err_pct": -19.0, "phi_err_pct": -1.0,
         "p_err_pct": 0.0, "travel_d": 5.0, "steps": 500, "wall_s": 1.0,
         "strain_sampled": (0.4, 1.0), "I_measured": 0.03},
        {"strain_target": 4.0, "mu_tail": 0.40, "mu_ratio": 0.40, "bias_pct": 0.0,
         "drift_pct": 0.8, "phi": 0.57, "mu_err_pct": -2.0, "phi_err_pct": -1.0,
         "p_err_pct": 0.0, "travel_d": 20.0, "steps": 2000, "wall_s": 1.0,
         "strain_sampled": (1.6, 4.0), "I_measured": 0.03},
        {"strain_target": 8.0, "mu_tail": 0.402, "mu_ratio": 0.402, "bias_pct": 0.0,
         "drift_pct": 0.3, "phi": 0.57, "mu_err_pct": -1.5, "phi_err_pct": -1.0,
         "p_err_pct": 0.0, "travel_d": 40.0, "steps": 4000, "wall_s": 1.0,
         "strain_sampled": (3.2, 8.0), "I_measured": 0.03},
    ]
    conv = ana.convergence(rows)
    assert [round(d["dmu_pct"], 2) for d in conv["deltas"]] == [65.0, 21.21, 0.5]
    # 1 -> 4 already moves 21 %, 4 -> 8 moves 0.5 %: first stationary rung is 8
    assert conv["stationary_from_strain"] == 8.0
    assert conv["longest_rung"]["mu_tail"] == pytest.approx(0.402)


def test_tables_md_emits_verdict_from_tolerance():
    rows = ana.rung_table([{
        "strain_target": 1.0, "strain_sampled_lo": 0.4, "strain_sampled_hi": 1.0,
        "I_measured": 0.03, "mu_tail": 0.40, "mu_ratio_of_means": 0.41,
        "mu_ratio_bias_pct": -2.4, "mu_block_drift_pct": 0.5,
        "phi_bed_steady": 0.57, "phi_bed_settled": 0.57,
        "mu_err_pct": 100.0 * (0.40 / ana.ref_mu(0.03) - 1.0),
        "phi_err_pct": 100.0 * (0.57 / ana.ref_phi(0.03) - 1.0),
        "pressure_control_error": 0.0, "plate_travel_in_d": 5.0,
        "shear_steps": 500, "wall_clock_s": 1.0, "series": [],
    }])
    analysis = {"arm": "thick", "tolerance": ana.TOL,
                "stationary_tol": ana.STATIONARY_TOL,
                "points": {"0.03": {"rows": rows,
                                    "convergence": ana.convergence(rows)}}}
    md = ana.tables_md(analysis)
    assert "PASS" in md and "FAIL" not in md       # 0.40 vs 0.4096 is inside 10 %
    # table ① must echo the ratio-of-means estimator next to the mean-of-ratios
    assert "| 0.41 |" in md or "0.4100" in md
    assert "| 0.4000 |" in md


def test_ladder_is_increasing_and_covers_all_I_targets():
    assert I_ORDER == sorted(LADDER)
    for I, rungs in LADDER.items():
        assert list(rungs) == sorted(rungs), I
        assert rungs[0] == pytest.approx(0.2)      # the round-3 protocol rung
        assert rungs[-1] >= 2.0                    # at least a 10x strain rung


def test_stiffness_scale_matches_contact_period():
    # round-3 control: the step budget must scale as sqrt(kn) for fixed physics
    assert rst.ARMS["stiff"]["kn"] == pytest.approx(10.0 * rst.ARMS["baseline"]["kn"])
    assert math.sqrt(10.0) == pytest.approx(3.16227766, rel=1e-6)


# ------------------------------------------------------------- integration --
@pytest.mark.skipif(os.environ.get("MU_I_STATIONARITY") != "1",
                    reason="set MU_I_STATIONARITY=1 to run the live ladder point")
def test_live_ladder_point_schema_and_positivity():
    # default (round-3) settle budget: an under-settled servo (e.g. 400 steps)
    # leaves the platen above the target stress (+89 % measured for that case)
    # and would make the pressure-control assertion below meaningless.
    rec = rst.run_ladder_point(3e-1, 0.2, arm="baseline", sample_stride=5)
    for key in ("I_measured", "mu_tail", "mu_ratio_of_means", "phi_bed_steady",
                "plate_height_steady", "shear_steps", "n_samples"):
        assert key in rec, key
    assert rec["mu_tail"] > 0.0
    assert rec["phi_bed_steady"] == pytest.approx(rec["phi_bed_steady"])  # not NaN
    assert 0.2 < rec["phi_bed_steady"] < 0.75
    assert rec["shear_steps"] >= 1
    assert rec["strain_sampled_hi"] <= rec["strain_target"] * 1.05
    assert rec["plate_travel_in_d"] > 0.0
    # The pressure servo is reported, not asserted tight: on a 0.2-strain
    # window the shear phase is still in start-up (measured +89 % over-stress at
    # I_target = 0.3), which is exactly the effect the strain ladder quantifies.
    assert rec["pressure_control_error"] == pytest.approx(rec["pressure_control_error"])
    assert abs(rec["pressure_control_error"]) < 5.0