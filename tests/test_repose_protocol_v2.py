"""Regression tests for the repose-protocol validity work (GAP-DEM-CLUMP round 2).

Guards three things that round 1 got wrong:

* the mass-matched fill really equalises the total solid volume across clump
  shapes without introducing initial interpenetration,
* the quiescence gate stops the tail only when the tolerance is met, and reports
  ``quiesced=False`` (not a silently short tail) when it is not,
* the rolling-resistance branch is rate proportional below ``eps_w`` and hence
  has no static (rate independent) branch, which is what the C5 leg claims,
* the round-1 campaign entry point actually writes its outputs (it did not: the
  whole writing block was dead code inside ``derive_failure_reason``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "gap_dem_clump"))

from tensordem.contact.clump_contact import ContactParams  # noqa: E402
from tensordem.particle.clump_dem import (  # noqa: E402
    ClumpDEM,
    lifted_box_repose,
)

import repose_protocol_v2 as v2  # noqa: E402
from run_gap_dem_clump import make_shape  # noqa: E402


# ---------------------------------------------------------------------------
# mass-matched fill
# ---------------------------------------------------------------------------
def test_round1_fill_is_shape_confounded():
    """Establishes the defect the matching has to remove (baseline, not a fix)."""
    counts, vols = {}, {}
    for ar in (1.0, 3.0):
        p = make_shape(ar)()
        n = v2.fill_particles(p, v2.PROTOCOL_V1["fill_height"])
        counts[ar], vols[ar] = n, n * p.volume
    assert counts[3.0] < 0.5 * counts[1.0]
    assert vols[1.0] / vols[3.0] > 1.5


def test_matched_fill_height_equalises_solid_volume():
    target = 2.2695e-06          # smallest round-1 fill (the AR=3 default)
    got = []
    for ar in (1.0, 1.5, 2.0, 3.0):
        m = v2.matched_fill_height(make_shape(ar)(), target)
        assert m["rel_err"] <= 0.05
        got.append(m["solid_volume"])
    assert max(got) / min(got) < 1.10     # within one lattice layer


def test_matched_fill_keeps_the_lattice_non_overlapping():
    """Matching must not buy equal mass with initial interpenetration."""
    p = make_shape(3.0)()
    m = v2.matched_fill_height(p, 2.2695e-06)
    from tensordem.particle.clump_dem import lattice_packing

    ext = v2.particle_extent(p)
    pos, _ = lattice_packing(
        p, (-v2.BOX["box_half_x"], -v2.BOX["slab_half_y"], 0.0),
        (v2.BOX["box_half_x"], v2.BOX["slab_half_y"], m["fill_height"]),
        spacing=v2.SPACING_FACTOR * ext,
    )
    assert len(pos) == m["n_particles"]
    # no two body centres closer than one extent along the densest axis
    d = np.linalg.norm(np.diff(np.sort(pos[:, 2])), axis=0) if len(pos) > 1 else 1.0
    assert d >= 0.99 * v2.SPACING_FACTOR * ext[2]


# ---------------------------------------------------------------------------
# quiescence gate
# ---------------------------------------------------------------------------
def _one_sphere_sim(dt=5e-5):
    sim = ClumpDEM([make_shape(1.0)()], np.array([[0.0, 0.0, 0.0]]),
                   params=ContactParams(mu_roll=0.3), dt=dt)
    return sim


def _repose_fast(**kw):
    args = dict(seed=0, params=ContactParams(mu_roll=0.3), fill_height=0.020,
                wall_height=0.04, settle_steps=150, lift_steps=600, tail_steps=400,
                dt=5e-5, lift_speed=0.30, probe_every=100,
                **v2.BOX)
    args.update(kw)
    return lifted_box_repose(make_shape(1.0), **args)


def test_quiescence_gate_reaches_tolerance():
    res = _repose_fast(min_tail_steps=100, max_tail_steps=3000,
                       quiescence_mean_speed=0.05)
    assert res["quiesced"] is True
    assert res["tail_steps_used"] <= 3000
    assert res["tail_spread"]["mean_speed"] <= 0.05


def test_quiescence_gate_reports_failure_instead_of_short_tail():
    res = _repose_fast(min_tail_steps=100, max_tail_steps=200,
                       quiescence_mean_speed=0.0)
    assert res["quiesced"] is False
    assert res["tail_steps_used"] == 200


def test_fixed_tail_keeps_round1_behaviour():
    res = _repose_fast()                       # no gate -> fixed tail, no new keys set
    assert res["quiesced"] is None
    assert res["tail_steps_used"] == 400


# ---------------------------------------------------------------------------
# rolling-resistance law: rate proportional, no static branch
# ---------------------------------------------------------------------------
def test_rolling_saturation_matches_the_law_and_vanishes_at_rest():
    sim = _one_sphere_sim()
    eps = sim.params.eps_w
    sim.omega[:] = np.array([[0.0, 0.0, 0.0]])
    s0 = sim.rolling_torque_saturation()
    assert s0["median"] == pytest.approx(0.0, abs=1e-12)      # no static branch
    sim.omega[:] = np.array([[eps, 0.0, 0.0]])
    s1 = sim.rolling_torque_saturation()
    assert s1["median"] == pytest.approx(0.5, rel=1e-12)      # half the Coulomb limit
    sim.omega[:] = np.array([[0.0, 100.0 * eps, 0.0]])
    assert sim.rolling_torque_saturation()["median"] > 0.98


def test_rolling_saturation_scales_with_eps_w():
    sim = _one_sphere_sim()
    sim.omega[:] = np.array([[1.0, 0.0, 0.0]])
    a = sim.rolling_torque_saturation(eps_w=1.0)["median"]
    b = sim.rolling_torque_saturation(eps_w=0.1)["median"]
    assert b > a > 0.4


# ---------------------------------------------------------------------------
# round-1 campaign entry point must actually produce its outputs
# ---------------------------------------------------------------------------
def test_round1_campaign_main_writes_outputs(tmp_path, monkeypatch):
    import run_gap_dem_clump as r1

    monkeypatch.setattr(r1, "leg_c1_analytic_anchors", lambda: {"pass": True, "rows": []})
    monkeypatch.setattr(r1, "leg_c2_inertia_scan", lambda: {"pass": True, "rows": [],
                                                            "worst_rel_err_production": 0.0})
    monkeypatch.setattr(r1, "leg_c3_repose_scan",
                        lambda *a, **k: {"pass": False, "rows": [], "by_ar": {},
                                         "monotone_increasing": False,
                                         "max_delta_deg": 0.0,
                                         "sphere_in_literature_band": False,
                                         "deltas_vs_sphere_deg": {}})
    monkeypatch.setattr(r1, "leg_c3_rolling_sweep", lambda *a, **k: {})
    monkeypatch.setattr(sys, "argv", ["run_gap_dem_clump.py", "--out", str(tmp_path)])
    rc = r1.main()
    assert (tmp_path / "results.json").exists(), "campaign wrote no results.json"
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["campaign"] == "GAP-DEM-CLUMP"
    assert (tmp_path / "criteria.json").exists()
    assert rc == 1                      # honest FAIL still exits non-zero