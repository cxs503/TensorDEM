"""Geometric-scale invariance of DEM contact parameter sets (M2/M3).

Two levels of evidence:

* **Exact self-similarity of the time integrator** (cheap, machine precision):
  starting from the same lattice, the ``lambda = 2`` case must reproduce the
  ``lambda = 1`` trajectory under ``x -> lam x``, ``v -> lam^0.5 v``,
  ``t -> lam^0.5 t``.  This isolates the transform from any statistical noise
  and fails loudly if a contact law or a boundary condition is not scaled.
* **Macroscopic probes**: a small blade-resistance case is run at both scales
  and the dimensionless resistance must agree within the pre-registered 2 %.

The production campaign (three probes x three stiffness levels x seven
variants) lives in ``scripts/run_contact_scale_ledger.py`` and is analysed in
``artifacts/contact_scale_ledger/``.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from tensordem.contact_scale_invariance import (
    METRIC_KEYS,
    PROBES,
    GranularCase,
    _Engine,
    _box_planes,
    _pack,
    analyse,
    drag_probe,
    scaled_variants,
)


# ---------------------------------------------------------------------------
# transform structure
# ---------------------------------------------------------------------------


def test_scaled_variants_follow_the_ledger_exponents():
    base = GranularCase()
    variants = {v.name: v for v in scaled_variants(base, 2.0)}
    similar = variants["lamN_similar"].case
    assert similar.radius == pytest.approx(2.0 * base.radius)
    assert similar.kn == pytest.approx(4.0 * base.kn)
    assert similar.rolling_lever == pytest.approx(2.0 * base.rolling_lever)
    # the dimensionless rolling ratio must be invariant
    assert similar.mu_rolling == pytest.approx(base.mu_rolling)
    assert similar.mu == pytest.approx(base.mu)
    assert similar.restitution == pytest.approx(base.restitution)
    # protocol scales: lengths x lam, stresses x lam, speeds x lam^0.5
    assert similar.box_width == pytest.approx(2.0 * base.box_width)
    assert similar.normal_stress == pytest.approx(2.0 * base.normal_stress)
    assert similar.shear_velocity == pytest.approx(
        math.sqrt(2.0) * base.shear_velocity)
    assert similar.hold_time == pytest.approx(math.sqrt(2.0) * base.hold_time)
    assert similar.timestep() == pytest.approx(
        math.sqrt(2.0) * base.timestep())


def test_controls_change_exactly_one_parameter():
    base = GranularCase()
    variants = {v.name: v for v in scaled_variants(base, 2.0)}
    similar = variants["lamN_similar"].case
    knob = variants["ctl_kn_constant"].case
    assert knob.kn == pytest.approx(base.kn)
    assert knob.timestep() == pytest.approx(similar.timestep())
    assert knob.rolling_lever == pytest.approx(similar.rolling_lever)
    lever = variants["ctl_lever_constant"].case
    assert lever.rolling_lever == pytest.approx(base.rolling_lever)
    assert lever.kn == pytest.approx(similar.kn)
    assert lever.mu_rolling != pytest.approx(similar.mu_rolling)
    restitution = variants["ctl_restitution_boost"].case
    assert restitution.restitution == pytest.approx(base.restitution + 0.1)
    assert restitution.mu_rolling == pytest.approx(similar.mu_rolling)
    out_of_range = variants["ctl_restitution_out_of_range"].case
    assert out_of_range.restitution > 1.0


def test_rebound_probe_detects_restitution():
    """The rebound apex ratio is the observable that restitution controls."""
    from tensordem.contact_scale_invariance import rebound_probe

    low = rebound_probe(replace(GranularCase(), restitution=0.3))
    mid = rebound_probe(replace(GranularCase(), restitution=0.4))
    high = rebound_probe(replace(GranularCase(), restitution=0.5))
    ratios = [r["rebound_height_ratio"] for r in (low, mid, high)]
    assert ratios[0] < ratios[1] < ratios[2]
    assert (ratios[2] - ratios[1]) / ratios[1] > 0.10


# ---------------------------------------------------------------------------
# exact self-similarity of the integrator
# ---------------------------------------------------------------------------


def _tiny_case() -> GranularCase:
    return replace(GranularCase(), radius=0.004, box_width=0.040,
                   box_depth=0.020, fill_height=0.036, hold_time=0.06,
                   drag_velocity=0.06)


def test_trajectory_is_self_similar_under_the_transform():
    """50 steps at lambda = 1 and lambda = 2 must agree after rescaling."""
    base = _tiny_case()
    variants = {v.name: v for v in scaled_variants(base, 2.0, controls=False)}
    lam = 2.0
    engines = []
    for name in ("lam1_reference", "lamN_similar"):
        case = variants[name].case
        engine = _Engine(case, _pack(case))
        planes = _box_planes(case, floor_mu=case.mu)
        engines.append((case, engine, planes))
    n_steps = 50
    for _ in range(n_steps):
        for case, engine, planes in engines:
            engine.step(planes)
    (base_case, base_engine, _), (_, sim_engine, _) = engines
    pos_err = np.abs(sim_engine.pos / lam - base_engine.pos).max()
    vel_err = np.abs(sim_engine.vel / math.sqrt(lam) - base_engine.vel).max()
    scale = max(np.abs(base_engine.pos).max(), 1e-30)
    assert pos_err / scale < 1e-9
    assert vel_err < 1e-6 * max(np.abs(base_engine.vel).max(), 1e-30) + 1e-12


# ---------------------------------------------------------------------------
# a macroscopic probe at both scales
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("probe", ["drag"])
def test_probe_is_scale_invariant(probe):
    base = _tiny_case()
    variants = {v.name: v for v in scaled_variants(base, 2.0, controls=False)}
    metric = METRIC_KEYS[probe]
    values = {}
    for name in ("lam1_reference", "lamN_similar"):
        values[name] = PROBES[probe](variants[name].case)[metric]
    assert all(np.isfinite(v) and v > 0 for v in values.values())
    deviation = abs(values["lamN_similar"] - values["lam1_reference"]) / abs(
        values["lam1_reference"])
    assert deviation <= 0.02, (values, deviation)


def test_drag_probe_returns_a_physical_record():
    record = drag_probe(_tiny_case())
    assert record["case"] == "drag"
    assert record["n_particles"] > 20
    assert record["resistance_dimensionless"] > 0.0
    assert record["timestep"] > 0.0


# ---------------------------------------------------------------------------
# analysis / verdict logic
# ---------------------------------------------------------------------------


def _record(probe: str, variant: str, value: float, kind: str,
            expected: str) -> dict:
    return {"probe": probe, "metric": METRIC_KEYS[probe], "variant": variant,
            "lam": 1.0 if variant == "lam1_reference" else 2.0, "kind": kind,
            "expected": expected, "note": "", "value": value,
            "raw": {}, "kn_level": 600.0, "case": {}}


def test_analysis_verdicts():
    records = [
        _record("drag", "lam1_reference", 1000.0, "reference", "invariant"),
        _record("drag", "lamN_similar", 1000.0, "similar", "invariant"),
        _record("drag", "ok_invariance", 1010.0, "similar", "invariant"),
        _record("drag", "bad_invariance", 1100.0, "similar", "invariant"),
        _record("drag", "caught_control", 800.0, "control", "violation"),
        _record("drag", "missed_control", 1005.0, "control", "violation"),
    ]
    table = analyse(records)
    variants = table["probes"]["drag"]["variants"]
    assert variants["ok_invariance"]["verdict"] == "PASS"
    assert variants["bad_invariance"]["verdict"] == "FAIL"
    assert variants["caught_control"]["verdict"] == "CAUGHT"
    assert variants["missed_control"]["verdict"] == "MISSED"
    gate = table["gate"]
    assert gate["pass"] is False
    assert gate["n_invariance_fail"] == 1
    assert gate["n_controls_missed"] == 1


def test_analysis_gate_is_green_when_everything_behaves():
    records = [
        _record("shear", "lam1_reference", 0.32, "reference", "invariant"),
        _record("shear", "lamN_similar", 0.32, "similar", "invariant"),
        _record("shear", "ctl_kn_constant", 0.24, "control", "violation"),
    ]
    table = analyse(records)
    assert table["gate"]["pass"] is True