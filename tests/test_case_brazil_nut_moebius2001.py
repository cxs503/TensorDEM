"""Fast unit tests for the CASE-DEM-BRAZIL-NUT-SEGREGATION harness.

Only the bookkeeping is tested here (metric definitions on synthetic series,
packing/geometry guards, density overwrite, pre-registered grid).  The physics
runs are the benchmark itself (``sweep.py --grid full``) and are far too long
for a unit test.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CASE = os.path.join(ROOT, "benchmarks", "case_brazil_nut_moebius2001")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, CASE)

import run_case as rc  # noqa: E402
import sweep as sw  # noqa: E402


class _FakeBed:
    """Minimal stand-in for the metric function (radii + species labels only)."""

    def __init__(self, r_small: float, r_large: float) -> None:
        self.radii = np.array([r_small, r_small, r_large])
        self.is_large = np.array([False, False, True])


def _series(z, surface):
    n = len(z)
    return {"tap": list(range(1, n + 1)), "time_s": [0.077 * i for i in range(1, n + 1)],
            "z_intruder_m": list(z), "z_small_mean_m": [s - 0.02 for s in surface],
            "surface_m": list(surface), "ke_J": [0.0] * n, "ke_samples": []}


def test_rise_per_tap_and_layering_index_on_synthetic_series():
    bed = _FakeBed(1.0e-3, 4.0e-3)          # d = 2 mm, D = 8 mm
    z = [0.02, 0.024, 0.028, 0.032]         # 4 mm per tap = 2 d per tap
    surface = [0.05] * 4
    m = rc.summarise(_series(z, surface), bed, {"H0": 0.09})
    assert m["rise_per_tap_over_d_mean"] == pytest.approx(2.0, rel=1e-12)
    assert m["net_rise_over_d"] == pytest.approx(6.0, rel=1e-12)
    assert m["direction"] == "rise"
    # layering index = (z_D - z_small)/H0 = (0.02 - 0.03)/H0
    assert m["layering_index_initial"] == pytest.approx(-0.01 / 0.09, rel=1e-9)
    assert m["surfaced_at_tap"] is None


def test_surfacing_tap_is_detected_when_top_reaches_the_surface():
    bed = _FakeBed(1.0e-3, 4.0e-3)
    z = [0.02, 0.03, 0.045, 0.052]           # top = z + 4 mm
    surface = [0.05] * 4                    # z + R = 0.049 < 0.05 until the 4th
    m = rc.summarise(_series(z, surface), bed, {"H0": 0.09})
    assert m["surfaced_at_tap"] == 4


def test_sinking_is_reported_as_sink():
    bed = _FakeBed(1.0e-3, 4.0e-3)
    m = rc.summarise(_series([0.03, 0.027, 0.024], [0.05] * 3), bed, {"H0": 0.09})
    assert m["direction"] == "sink"
    assert m["rise_per_tap_over_d_mean"] < 0.0


def test_geometry_ratios_follow_the_reference():
    """W/H and the initial depth-of-top must reproduce the reference ratios."""
    assert rc.D_OVER_H == pytest.approx(2.54 / 8.6)
    assert rc.W_OVER_H == pytest.approx(8.2 / 8.6)
    assert rc.TOP_DEPTH_OVER_H == pytest.approx(4.6 / 8.6)


def test_bed_build_embed_and_mass_override():
    """Short build (tiny settle): geometry, intruder mass and removal accounting."""
    state = rc.build_bed(d_small=2.0e-3, size_ratio=8.0, seed=0, settle_time=2.0e-3)
    assert state["bed"].n >= 600
    emb = rc.embed_intruder(state, rho_ratio=3.24)
    bed = emb["bed"]
    assert bed.n == state["bed"].n - emb["removed_small"] + 1
    big = bed.is_large
    assert big.sum() == 1
    # intruder mass from its own density, bed particles from rho_m
    r_l = float(bed.radii[big][0])
    assert float(bed.mass[big][0]) == pytest.approx(
        3.24 * rc.RHO_BED * math.pi * r_l**2 * bed.config.thickness, rel=1e-12)
    assert float(bed.mass[~big][0]) == pytest.approx(
        rc.RHO_BED * math.pi * float(bed.radii[~big][0]) ** 2 * bed.config.thickness,
        rel=1e-12)
    # removed area must be comparable to the intruder area (a hole is carved out)
    intruder_area = math.pi * r_l**2
    assert 0.5 * intruder_area < emb["removed_area"] < 1.5 * intruder_area


def test_prereg_grid_is_complete_and_unique():
    cases = sw.grid_full()
    names = [sw.case_name(c) for c in cases]
    assert len(names) == len(set(names))
    assert len(cases) >= 20
    rhos = sorted({c["rho_ratio"] for c in cases if c["size_ratio"] == 8.0})
    assert 0.47 in rhos and 3.24 in rhos
    assert any(c.get("dt") == 1.0e-5 for c in cases)
    assert len({c.get("gamma", 5.0) for c in cases}) >= 3
    for key in ("H1_direction", "H2_nonmonotonic", "H3_size"):
        assert key in sw.PREREG["hypotheses"]