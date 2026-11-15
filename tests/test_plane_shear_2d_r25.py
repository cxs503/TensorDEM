"""Round-25 gates (m22440): structure diagnostics + the returned-sweep verdict.

Round 24 built the 2-D disk plane-shear device and submitted the sweep; the
round closed before the sweep returned.  This module tests the two things that
round 25 adds:

* ``hexatic_order`` measures what it claims (a triangular lattice is hexatically
  ordered, a disordered packing is not) and ``_random_packing`` produces an
  admissible, overlap-free, *disordered* packing at the lattice's density;
* the round-24 assembly script, pointed at the returned sweep
  (``runs/plane_shear_2d/probe_I*.json``), reproduces the pre-registered verdict
  -- i.e. the numbers in ``docs/CASE_DEM_PLANE_SHEAR_MUIDI_R25.md`` are the
  script's output, not a transcription.

Run: ``pytest -q tests/test_plane_shear_2d_r25.py``
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "plane_shear_mu_i"
for p in (BENCH, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import analyse_2d_r24 as analyse  # noqa: E402
import plane_shear_2d as ps2d  # noqa: E402
import reference_archive  # noqa: E402

RUN_DIR = BENCH / "runs" / "plane_shear_2d"


def _triangular(n_x: int, n_y: int, d: float) -> np.ndarray:
    pts = []
    for k in range(n_y):
        y = k * d * math.sqrt(3.0) / 2.0
        for m in range(n_x):
            pts.append(((m + 0.5 * (k % 2)) * d, y))
    return np.asarray(pts, dtype=float)


def test_hexatic_order_separates_crystal_from_disorder():
    """psi6 must be near 1 on a lattice and clearly lower on a random packing."""
    d = 0.01
    r = 0.5 * d
    lat = _triangular(20, 10, d)
    crystal = ps2d.hexatic_order(lat, r, width=0.0)
    assert crystal["psi6_mean"] > 0.95, crystal
    # the default shell (1.25-1.75 d) is the *second* neighbour shell of a
    # triangular lattice: 6 bonds, fewer at the free boundary of the patch
    assert 4.5 < crystal["z_mean"] <= 6.0, crystal

    rng = np.random.default_rng(7)
    rand = rng.uniform(0.0, 0.20, size=(180, 2))
    disorder = ps2d.hexatic_order(rand, r, width=0.0)
    assert disorder["psi6_mean"] < 0.75, disorder
    assert disorder["psi6_mean"] < crystal["psi6_mean"]


def test_random_packing_is_admissible_and_disordered():
    """The disordered seed must fit the cell, avoid overlaps and stay disordered."""
    case = ps2d.PlaneShear2DCase(I_target=0.05, n_particles=240, width=0.24, gap=0.12,
                                 init_mode="random", seed=3)
    case.validate()
    r = case.radius
    pos, rad = ps2d._random_packing(case, 3.0 * r, case.gap + r)
    assert len(pos) == case.n_particles
    assert np.all(rad > 0)
    assert pos[:, 1].min() >= 3.0 * r - 1e-12
    assert pos[:, 1].max() <= case.gap + r + 1e-12
    # no overlaps in the periodic-x metric
    dx = pos[None, :, 0] - pos[:, None, 0]
    dx -= case.width * np.round(dx / case.width)
    dy = pos[None, :, 1] - pos[:, None, 1]
    dist = np.hypot(dx, dy)
    np.fill_diagonal(dist, np.inf)
    assert dist.min() >= 2.0 * r * (1.0 - 2.0e-3)
    order = ps2d.hexatic_order(pos, r, width=case.width)
    assert order["psi6_mean"] < 0.75, order


def test_lattice_seed_is_the_crystal_round24_shipped():
    """Sanity on the round-24 seed: it *is* hexatically ordered (the defect)."""
    case = ps2d.PlaneShear2DCase(I_target=0.05, n_particles=240, width=0.24, gap=0.12,
                                 init_mode="lattice", seed=3)
    r = case.radius
    pos, _ = ps2d._lattice_packing(case, 3.0 * r, case.gap + r)
    order = ps2d.hexatic_order(pos, r, width=case.width)
    assert order["psi6_mean"] > 0.9, order


@pytest.mark.skipif(not (RUN_DIR / "probe_I0.05.json").exists(),
                    reason="returned sweep artifacts not present")
def test_returned_sweep_reproduces_the_published_verdict():
    """The R25 verdict must be the output of the analysis, not a transcription."""
    law = reference_archive.archived_law()
    result = analyse.assemble(RUN_DIR, law, probe=True)
    v = result["verdict"]
    assert v["n_points"] == 4
    assert v["n_judged"] == 3                     # I = 0.2 is outside the fit
    # P1 (servo calibration) passes; P2 (mu in the 10 % band) and P5 fail
    assert v["P1_calibration"] is True
    assert v["P2_mu_in_band"] is False
    assert v["mu_pass_count"] == 0
    assert 60.0 < v["max_abs_mu_err_pct"] < 80.0
    # phi is *above* the archived dilatancy law everywhere
    assert all(p["phi_err_pct"] > 0.0 for p in result["points"])
    by_I = {round(p["I_target"], 4): p for p in result["points"]}
    low = by_I[0.02]
    assert low["mu_wall"]["mean"] == pytest.approx(0.0963, abs=2e-3)
    assert low["phi_gap"]["mean"] == pytest.approx(0.873, abs=2e-3)
    # the archived law at the *measured* I is what the residual is taken against
    assert low["mu_ref"] == pytest.approx(law.mu(low["I_measured"]), rel=1e-12)


@pytest.mark.skipif(not (RUN_DIR / "analysis_r24_probe.json").exists(),
                    reason="analysis artifact not present")
def test_committed_analysis_matches_a_fresh_assemble():
    """Guard against a hand-edited analysis JSON in the tree."""
    law = reference_archive.archived_law()
    fresh = analyse.assemble(RUN_DIR, law, probe=True)
    committed = json.loads((RUN_DIR / "analysis_r24_probe.json").read_text())
    assert committed["verdict"]["P2_mu_in_band"] == fresh["verdict"]["P2_mu_in_band"]
    for a, b in zip(committed["points"], fresh["points"]):
        assert a["I_target"] == pytest.approx(b["I_target"])
        assert a["mu_wall"]["mean"] == pytest.approx(b["mu_wall"]["mean"], rel=1e-12)