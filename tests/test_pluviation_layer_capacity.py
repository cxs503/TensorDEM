"""Regression tests for the pluviation layer-capacity defect (m14868).

`run_pluviation` generates one rain layer on a staggered square lattice clipped
to the injection disk, but validated the requested ``layer_size`` against a
*hexagon-area estimate* of that disk.  For a small injection radius the estimate
over-predicts the true capacity (7 sites at ``inject_radius_factor = 3.5``
versus an estimate of 10), so the pour silently produced a short array and the
run died in ``pos[layer_idx] = pts[:take]`` with

    ValueError: shape mismatch: value array of shape (7,3) could not be
    broadcast to indexing result of shape (10,3)

which says nothing about the actual problem.  The capacity is now computed
exactly (``pluviation_layer_capacity``) and a short layer raises :class:`HeapError`
naming the capacity.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from tensordem.dem_calibration import heap
from tensordem.dem_calibration.heap import (
    HeapError,
    _injection_positions,
    pluviation_layer_capacity,
    run_pluviation,
)
from tensordem.dem_calibration.materials import spec_by_name

SAND = spec_by_name("sand")


def _case(**kw):
    return dataclasses.replace(SAND.case("calibration"), **kw)


@pytest.mark.parametrize("irf", [3.5, 7.0, 14.0])
def test_capacity_matches_generated_lattice(irf):
    case = dataclasses.replace(SAND.case("calibration"), inject_radius_factor=irf)
    rng = np.random.default_rng(0)
    pts = _injection_positions(case, (0.0, 0.0), 0.0, 10_000, rng=rng,
                               occupied=np.zeros((0, 3)))
    assert pts.shape[0] == pluviation_layer_capacity(case)


def test_area_estimate_overpredicts_small_disk():
    """The historical check was an area estimate; document the discrepancy."""
    case = dataclasses.replace(SAND.case("calibration"), inject_radius_factor=3.5)
    r, gap = case.radius, case.packing_gap
    area = math.pi * (case.inject_radius_factor * r) ** 2
    spacing = 2.0 * r * (1.0 + gap)
    estimate = int(area / (spacing * spacing * math.sqrt(3.0) / 2.0))
    capacity = pluviation_layer_capacity(case)
    assert capacity == 7
    assert estimate == 10            # the over-prediction that caused the crash
    assert capacity < estimate


def test_short_layer_raises_domain_error():
    case = _case(inject_radius_factor=3.5, n_grains=8, layer_size=8, seed=0)
    with pytest.raises(HeapError) as excinfo:
        run_pluviation(case)
    msg = str(excinfo.value)
    assert "7 grains" in msg and "layer_size is 8" in msg
    assert "inject_radius_factor = 3.5" in msg


def test_at_capacity_layer_runs_and_reports_speeds():
    """The capacity itself is a *legal* pour (only capacity+1 is rejected)."""
    case = _case(n_grains=120, layer_size=12, seed=0)
    assert pluviation_layer_capacity(case) == 37 >= case.layer_size
    res = run_pluviation(case, return_positions=True)
    assert len(res["positions"]) == 120
    assert len(res["speeds"]) == 120
    assert all(s >= 0.0 for s in res["speeds"])
    assert res["digest"] and isinstance(res["digest"], str)


def test_default_protocol_capacity_covers_shipped_layer_size():
    """The shipped protocol (layer_size=12, irf=7) must not be affected."""
    case = SAND.case("calibration")
    assert case.layer_size == 12
    assert pluviation_layer_capacity(case) >= case.layer_size