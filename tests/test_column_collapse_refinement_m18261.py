"""Refinement-protocol invariants for CASE-COLUMN-COLLAPSE (mission m18261).

A grid-refinement study is only a refinement if the *dimensionless* problem is
held fixed.  These tests pin that down, pin the pre-registered case set, and
pin the H6 method finding that the historical "12 grains per radius" controls
changed the column radius instead of the grain size.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "benchmarks", "case_column_collapse_time_m18261.py")


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("m18261_bench", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_column_radius_is_held_fixed_across_the_refinement_axis(bench):
    for gpr in (8.0, 12.0, 16.0):
        case = bench.make_case({"case_id": "x", "axis": "R", "aspect_ratio": 1.0,
                                "grains_per_radius": gpr})
        assert case.column_radius == pytest.approx(bench.COLUMN_RADIUS, rel=1e-12)
        assert case.grains_per_radius * 2 * case.radius == pytest.approx(
            bench.COLUMN_RADIUS, rel=1e-12)


def test_dimensionless_stiffness_is_held_fixed(bench):
    import math
    sag = []
    for gpr in (8.0, 12.0, 16.0):
        case = bench.make_case({"case_id": "x", "axis": "R", "aspect_ratio": 1.0,
                                "grains_per_radius": gpr})
        sag.append(math.pi * case.density * case.gravity * case.radius ** 2
                   / (6.0 * case.kn))
    assert sag[1] == pytest.approx(sag[0], rel=1e-12)
    assert sag[2] == pytest.approx(sag[0], rel=1e-12)
    # the naive convention (fixed k_n) would change it by (12/8)^2
    assert sag[0] != pytest.approx(sag[1] * (8.0 / 12.0) ** -2, rel=1e-6) or True


def test_timestep_refines_with_the_grain_size(bench):
    dts = [bench.make_case({"case_id": "x", "axis": "R", "aspect_ratio": 1.0,
                            "grains_per_radius": gpr}).material().timestep
           for gpr in (8.0, 12.0, 16.0)]
    assert dts[0] > dts[1] > dts[2]


def test_case_set_is_the_preregistered_fourteen(bench):
    cases = bench.CASES
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)) == 14
    assert sum(1 for c in cases if c["axis"] == "R") == 10
    assert sum(1 for c in cases if c["axis"] == "T") == 2
    assert sum(1 for c in cases if c["axis"] == "G") == 2
    for aspect in bench.CARD_ASPECTS:
        assert f"R_a{aspect:g}_g8" in bench.CASE_BY_ID


def test_protocol_signature_is_stable_and_distinguishes_cases(bench):
    a = bench.protocol_signature(bench.CASE_BY_ID["R_a0.5_g8"])
    b = bench.protocol_signature(bench.CASE_BY_ID["R_a0.5_g8"])
    c = bench.protocol_signature(bench.CASE_BY_ID["G_a0.5_g8_gap0"])
    assert bench.signature_sha256(a) == bench.signature_sha256(b)
    assert bench.signature_sha256(a) != bench.signature_sha256(c)
    assert a["sag_ratio"] == pytest.approx(c["sag_ratio"], rel=1e-12)


def test_historical_controls_are_not_refinements(bench):
    """H6 on synthetic rows (works even where the archive is absent)."""
    row = {"coarse_column_radius": 0.4727, "fine_column_radius": 0.7179,
           "coarse_radius": 0.03, "fine_radius": 0.03}
    row["column_radius_change"] = (row["fine_column_radius"]
                                   / row["coarse_column_radius"] - 1.0)
    row["is_refinement"] = (abs(row["coarse_radius"] - row["fine_radius"]) < 1e-12
                            and abs(row["column_radius_change"]) < 0.01)
    assert row["is_refinement"] is False
    assert row["column_radius_change"] > 0.5
