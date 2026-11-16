"""Round 33 tests: the rough wall of the 2-D plane-shear device must move.

Defect under test
-----------------
``RoughWallShearCell`` builds the rough wall out of grain-glued disks and gives
them the wall velocity, but (rounds 24-32) never advanced their positions.  A
"moving rough wall" whose bumps never move is a *frozen corrugated* wall: the
sample is then dragged by the friction law alone instead of being driven by
interlocking, and the cell stops imposing its own shear rate.

These tests pin the three properties the fix has to have, in increasing order of
cost, and they are all cheap (the dynamic one runs a 240-disk smoke cell):

1. ``test_wall_grains_advect_at_wall_speed`` -- the rows' x displacement equals
   ``+-V t`` to machine precision (kinematics, not physics);
2. ``test_advection_never_creates_wall_wall_contacts`` -- modulo advection of a
   cyclic gap sequence may not open a hole a free disk can squeeze through
   (max gap < 2 d) nor make two wall grains overlap (min gap > d);
3. ``test_frozen_wall_is_slip_limited`` -- the dynamics: on the *same* cell, the
   frozen wall leaves the sample nearly plug-flowing (measured/imposed shear-rate
   ratio far below 1) while the advected wall closes most of the gap.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parents[1]
for _p in (str(REPO / "src"), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plane_shear_2d import PlaneShear2DCase, RoughWallShearCell  # noqa: E402
from wall_translation_ablation import run_point  # noqa: E402

SMOKE = dict(n_particles=310, width=0.24, gap=0.14, seed=0)


def _cell(**over) -> RoughWallShearCell:
    case = PlaneShear2DCase(**{**SMOKE, **over})
    return RoughWallShearCell(case)


def test_wall_grains_advect_at_wall_speed():
    cell = _cell()
    case = cell.case
    assert case.wall_translate is True, "the moving-wall path must be the default"
    x0_bot = cell.system.pos[cell.bottom_idx, 0].copy()
    x0_top = cell.system.pos[cell.top_idx, 0].copy()
    n = 400
    cell.advance(n, servo=False)
    expected = cell.v_wall * cell.dt * n
    got_bot = np.mod(cell.system.pos[cell.bottom_idx, 0] - x0_bot, case.width)
    got_top = np.mod(cell.system.pos[cell.top_idx, 0] - x0_top, case.width)
    assert np.allclose(got_bot, np.mod(expected, case.width), atol=1e-12)
    assert np.allclose(got_top, np.mod(-expected, case.width), atol=1e-12)
    # the ablation control must reproduce the archived behaviour exactly
    frozen = _cell(wall_translate=False)
    x0 = frozen.system.pos[frozen.bottom_idx, 0].copy()
    frozen.advance(n, servo=False)
    assert np.array_equal(frozen.system.pos[frozen.bottom_idx, 0], x0)


def test_advection_never_creates_wall_wall_contacts():
    cell = _cell()
    case = cell.case
    for _ in range(12):
        cell.advance(500, servo=True)
        for idx in (cell.bottom_idx, cell.top_idx):
            x = np.sort(cell.system.pos[idx, 0])
            gaps = np.diff(np.append(x, x[0] + case.width))
            assert gaps.min() > case.diameter, "wall grains may never overlap"
            # a free disk of diameter d needs a centre gap of 2 d to pass
            assert gaps.max() < 2.0 * case.diameter, "wall row opened a hole"


SMOKE_REF = (BENCH_DIR / "runs" / "wall_translation_r33"
             / "smoke_filled_n310_wfrozen_I0.05_s0.json")
SMOKE_PROTOCOL = dict(n_particles=310, smoke=True, width=0.24, gap=0.14,
                      settle=20.0, strain_settle=0.4, strain_measure=0.6)


def _frozen_reference() -> dict:
    """The archived-device control point (same cell, same protocol).

    Kept as a run artifact so that each test costs at most one simulation; if the
    artifact is absent it is recomputed, which is slow but still correct.
    """
    if SMOKE_REF.exists():
        return json.loads(SMOKE_REF.read_text())
    return run_point(inertial_number=0.05, mode="frozen", seed=0, **SMOKE_PROTOCOL)


def test_frozen_wall_is_slip_limited():
    """The archived device (wall bumps frozen) does not impose its own shear rate."""
    ref = _frozen_reference()
    assert ref["wall_translate"] is False
    assert float(ref["gamma_dot_ratio"]) < 0.30, ref["gamma_dot_ratio"]
    assert float(ref["p_wall_over_P"]) < 0.30, ref["p_wall_over_P"]


def test_advected_wall_closes_the_rate():
    """Advecting the bumps recovers a large part of the imposed rate and of P."""
    ref = _frozen_reference()
    fixed = run_point(inertial_number=0.05, mode="fixed", seed=0, **SMOKE_PROTOCOL)
    r_fixed = float(fixed["gamma_dot_ratio"])
    r_frozen = float(ref["gamma_dot_ratio"])
    assert math.isfinite(r_fixed)
    assert r_fixed > 2.5 * r_frozen, (r_frozen, r_fixed)
    # the confining servo can only hold its set point on the moving wall
    assert float(fixed["p_wall_over_P"]) > 5.0 * float(ref["p_wall_over_P"]), (
        ref["p_wall_over_P"], fixed["p_wall_over_P"])


def test_frozen_path_reproduces_archived_summary():
    """``wall_translate=False`` must reproduce the archived device exactly.

    The ablation switch has to leave the archived path untouched, so a frozen run
    has to reproduce the archived ``config_digest`` (deterministic final state) --
    bit-for-bit, not approximately.
    """
    ref = _frozen_reference()
    run = run_point(inertial_number=0.05, mode="frozen", seed=0, **SMOKE_PROTOCOL)
    assert run["wall_translate"] is False
    assert run["config_digest"] == ref["config_digest"], (
        "the ablation control drifted from the archived device")
    assert run["phi_gap"] == pytest.approx(ref["phi_gap"], rel=1e-12)
