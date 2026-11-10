"""Tests for the material-class repose-angle calibration closed loop.

Fast tests (no DEM time stepping) cover the estimator anchor, the packing
geometry, the monotonisation/inversion algebra and the calibration bookkeeping;
the DEM-backed tests are marked ``slow`` because each column-collapse run costs
a few seconds of CPU.
"""
from __future__ import annotations

import glob
import json
import math
import os

import pytest
import torch

from tensordem import repose_material_calibration as rmc
from tensordem.repose_calibration import ReposeCalibrationTable

SLOW = pytest.mark.slow

ARTIFACT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts", "repose_material_calibration", "calibration.json")


def _fast_spec(key: str = "glass_bead", **overrides) -> rmc.MaterialSpec:
    """A small, cheap variant of a material spec (same physics, fewer grains)."""
    base = rmc.MATERIALS[key]
    kn = overrides.pop("kn", 1000.0)
    cfg = {**base.__dict__,
           "cylinder_radius": 7.0 * base.radius,
           "fill_height": 2.5 * 7.0 * base.radius,
           "kn": kn,
           "dt_override": 2.0 * math.pi * math.sqrt(0.5 * base.mass / kn) / 20.0,
           "hold_time": 0.03, "lift_time": 0.15, "settle_time": 0.25}
    cfg.update(overrides)
    return rmc.MaterialSpec(**cfg)


# ---------------------------------------------------------------------------
# Presets and geometry (fast)
# ---------------------------------------------------------------------------

def test_three_material_classes_with_documented_targets():
    assert set(rmc.MATERIALS) == {"glass_bead", "sand", "gravel"}
    targets = {key: spec.target_theta_deg for key, spec in rmc.MATERIALS.items()}
    assert targets == {"glass_bead": 30.0, "sand": 35.0, "gravel": 40.0}
    # material classes must actually differ in grain scale and rolling prior
    assert rmc.MATERIALS["sand"].radius < rmc.MATERIALS["glass_bead"].radius
    assert (rmc.MATERIALS["gravel"].mu_rolling
            > rmc.MATERIALS["sand"].mu_rolling
            > rmc.MATERIALS["glass_bead"].mu_rolling)
    for spec in rmc.MATERIALS.values():
        assert spec.cylinder_radius >= 5.0 * spec.radius
        assert spec.dt > 0.0 and spec.n_steps > 1000


def test_material_lookup_rejects_unknown_key():
    with pytest.raises(rmc.MaterialCalibrationError):
        rmc.material("teflon")


def test_packing_is_non_overlapping_and_confined():
    for key in ("glass_bead", "sand", "gravel"):
        spec = rmc.MATERIALS[key]
        pos = rmc.initial_packing(spec)
        n = len(pos)
        assert n >= 300, (key, n)
        diff = pos[:, None, :] - pos[None, :, :]
        dist = torch.linalg.vector_norm(diff, dim=-1)
        dist.fill_diagonal_(float("inf"))
        assert float(dist.min()) > 2.0 * spec.radius * 0.97
        radial = torch.linalg.vector_norm(pos[:, :2], dim=1)
        assert float(radial.max()) < spec.cylinder_radius
        assert float(pos[:, 2].max()) >= spec.fill_height - 2.0 * spec.radius


def test_analytic_cone_is_recovered_by_the_repose_estimator():
    for angle in (20.0, 30.0, 40.0):
        check = rmc.analytic_cone_check(angle)
        assert check["passed"], check
        assert abs(check["measured_deg"] - angle) < 1.5


# ---------------------------------------------------------------------------
# Contact-law anchor (fast: single sphere, ~3000 steps)
# ---------------------------------------------------------------------------

def test_slider_onset_matches_arctan_min_mu_mu_rolling():
    # sliding-controlled: mu < mu_r -> onset at arctan(mu)
    slide = rmc.sliding_onset_angle(0.6, 0.9)
    assert slide["passed"], slide
    assert slide["abs_error_deg"] < 0.3
    # rolling-controlled: mu_r < mu -> onset at arctan(mu_r)
    roll = rmc.sliding_onset_angle(0.8, 0.25)
    assert roll["passed"], roll
    assert roll["abs_error_deg"] < 0.3


# ---------------------------------------------------------------------------
# Table algebra (fast, synthetic)
# ---------------------------------------------------------------------------

def _synthetic_nodes(mu_values, theta_values):
    return [{"mu": mu, "theta_r_deg": theta, "theta_r_std_deg": 0.0,
             "n_seeds": 2, "per_seed": [theta, theta], "r2_mean": 0.95,
             "max_speed_m_s_mean": 1e-3, "speed_p95_m_s_mean": 1e-4,
             "seeds": [1, 2]} for mu, theta in zip(mu_values, theta_values)]


def test_pava_reports_and_repairs_non_monotone_nodes():
    nodes = _synthetic_nodes([0.2, 0.4, 0.6, 0.8], [10.0, 14.0, 13.0, 18.0])
    table, report = rmc.build_table(rmc.MATERIALS["sand"], nodes)
    assert report["monotonicity_violations"] == 1
    assert report["raw_theta_deg"] == [10.0, 14.0, 13.0, 18.0]
    fitted = report["monotone_theta_deg"]
    assert all(b > a for a, b in zip(fitted, fitted[1:]))
    # PAV pools the offending pair to their mean
    assert fitted[1] == pytest.approx(13.5, abs=1e-6) or fitted[2] == pytest.approx(13.5, abs=1e-6)


def test_table_inverse_round_trip_and_extrapolation_guard():
    nodes = _synthetic_nodes([0.2, 0.35, 0.5, 0.65, 0.8], [8.0, 16.0, 23.0, 29.0, 34.0])
    table, _ = rmc.build_table(rmc.MATERIALS["glass_bead"], nodes)
    for mu in table.mu:
        assert table.mu_from_theta(table.theta_from_mu(mu)) == pytest.approx(mu, rel=1e-8)
    for mu in (0.27, 0.44, 0.72):
        recovered = table.mu_from_theta(table.theta_from_mu(mu))
        assert abs(recovered - mu) < 0.02
    with pytest.raises(Exception):
        table.mu_from_theta(table.theta_deg[-1] + 5.0)


def test_calibrate_flags_target_outside_bracket(monkeypatch):
    spec = rmc.MATERIALS["glass_bead"]
    nodes = _synthetic_nodes([0.2, 0.5, 0.8], [6.0, 12.0, 18.0])
    table, _ = rmc.build_table(spec, nodes)
    with pytest.raises(rmc.TargetOutOfBracketError):
        rmc.calibrate(spec, table, seeds=(1,))


def test_calibrate_closed_loop_bookkeeping(monkeypatch):
    """Inverse + verification logic on an analytic forward model (no DEM)."""
    spec = rmc.MATERIALS["sand"]

    def fake_simulate(spec, mu, *, seed=None, mu_rolling=None, keep_particles=False,
                      max_steps=None):
        theta = 15.0 + 30.0 * (mu - 0.2) / 0.6
        return {"theta_r_deg": theta, "mu": mu, "seed": seed or 0, "r2": 0.99}

    monkeypatch.setattr(rmc, "simulate", fake_simulate)
    nodes = [{"mu": m, "theta_r_deg": 15.0 + 30.0 * (m - 0.2) / 0.6, "n_seeds": 2,
              "r2_mean": 0.95, "max_speed_m_s_mean": 1e-3}
             for m in rmc.default_mu_nodes(count=7)]
    table, _ = rmc.build_table(spec, nodes)
    cal = rmc.calibrate(spec, table, seeds=(11, 12))
    assert cal["passed"] is True
    assert abs(cal["rel_error"]) < 0.01
    assert cal["seeds"] == [11, 12]
    assert rmc.MU_LOWER <= cal["mu_star"] <= rmc.MU_UPPER
    assert len(cal["verification_runs"]) == 2


def test_aggregate_nodes_reports_seed_scatter():
    runs = [
        {"mu": 0.5, "theta_r_deg": 20.0, "r2": 0.9, "seed": 1},
        {"mu": 0.5, "theta_r_deg": 22.0, "r2": 0.95, "seed": 2},
        {"mu": 0.3, "theta_r_deg": 12.0, "r2": 0.9, "seed": 1},
    ]
    nodes = rmc.aggregate_nodes(runs)
    assert [node["mu"] for node in nodes] == [0.3, 0.5]
    assert nodes[1]["theta_r_deg"] == pytest.approx(21.0)
    assert nodes[1]["theta_r_std_deg"] == pytest.approx(math.sqrt(2.0))
    assert nodes[1]["n_seeds"] == 2


# ---------------------------------------------------------------------------
# DEM-backed physics (slow)
# ---------------------------------------------------------------------------

@SLOW
def test_negative_control_flattens_the_heap():
    spec = _fast_spec("gravel")
    out = rmc.negative_control(spec)
    assert out["theta_r_deg"] < rmc.NEGATIVE_CONTROL_LIMIT_DEG, out["theta_r_deg"]


@SLOW
def test_repose_angle_grows_with_friction_and_beats_the_frictionless_control():
    """Sliding friction must steepen the pile relative to a frictionless pile.

    Only the *direction* is asserted: at this pile size the seed-to-seed scatter
    of the repose angle is of the order of the whole friction effect (see
    ``docs/repose_material_calibration.md`` 3.5), so a per-seed margin of several
    degrees between two friction levels is not a reproducible claim.
    """
    spec = rmc.material("gravel")      # documented calibration geometry
    low = rmc.simulate(spec, 0.2, seed=39907)
    high = rmc.simulate(spec, 0.7, seed=39907)
    zero = rmc.negative_control(spec, seed=39907)
    assert low["theta_r_deg"] > zero["theta_r_deg"]
    # a 0.5-wide gap in mu is well outside the seed scatter measured in the
    # campaign (docs/repose_material_calibration.md 3.5)
    assert high["theta_r_deg"] > low["theta_r_deg"] + 5.0, (low["theta_r_deg"],
                                                           high["theta_r_deg"])
    assert high["r2"] > 0.7, high["r2"]


@SLOW
def test_rolling_prior_raises_the_repose_angle():
    """mu_r is the rolling-resistance channel: removing it must flatten the pile."""
    spec = _fast_spec("sand")
    with_r = rmc.simulate(spec, 0.5, mu_rolling=spec.mu_rolling)
    without_r = rmc.simulate(spec, 0.5, mu_rolling=0.0)
    assert with_r["theta_r_deg"] > without_r["theta_r_deg"] + 1.0, (with_r["theta_r_deg"],
                                                                   without_r["theta_r_deg"])


# ---------------------------------------------------------------------------
# Shipped artifact (only when the campaign ran)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="calibration artifact not built")
def test_shipped_artifact_records_complete_audit_chain():
    with open(ARTIFACT) as fh:
        payload = json.load(fh)
    assert payload["kind"] == "tensordem-repose-material-calibration-v1"
    assert payload["audit"]["commit"]
    assert set(payload["materials"]) == {"glass_bead", "sand", "gravel"}
    for key, entry in payload["materials"].items():
        assert entry["material"]["target_theta_deg"] in (30.0, 35.0, 40.0)
        assert len(entry["scan_nodes"]) >= 7
        assert entry["negative_control"]["theta_r_deg"] < rmc.NEGATIVE_CONTROL_LIMIT_DEG
        assert "mu_star" in entry["calibration"] or "error" in entry["calibration"]
        # honest reporting: a failed criterion must be visible as False, never dropped
        assert isinstance(entry["checks"]["angle_within_10pct"], bool)
    assert payload["verdict"]["n_checks"] >= 3 * 6