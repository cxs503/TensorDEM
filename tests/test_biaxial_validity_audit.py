"""Invariants for the anchor validity audit (c = 0 test + convergence screen).

These tests do NOT re-run the DEM; they pin the parts that can be wrong silently:

* the 2-D Mohr-Coulomb fit used to extract the apparent cohesion must recover a
  known (c, phi) from synthetic data, and must return c ~ 0 when the data are
  cohesionless -- this is the arithmetic that decides whether the anchor's
  ``sin(phi) = q / (2p)`` reading is a friction angle at all;
* the cell tables in the two audit drivers must stay in step with the tag orders
  the analysis script consumes, so a sharded cluster job can never silently drop
  or mislabel a cell;
* if a committed audit artefact exists, its internal arithmetic must reproduce.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.join(REPO, "benchmarks", "case_dem_biaxial_friction")


def _load(name: str, filename: str):
    path = os.path.join(BENCH, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def analyse():
    return _load("_validity_analyse", "analyse_validity_audit.py")


@pytest.fixture(scope="module")
def screen_mod():
    return _load("_validity_screen", "screen_drivers.py")


@pytest.fixture(scope="module")
def pressure_mod():
    return _load("_validity_pressure", "audit_pressure_bonds.py")


# ---------------------------------------------------------------------------
# linear fit
# ---------------------------------------------------------------------------
def test_linfit_recovers_exact_line(analyse):
    a, b, r2 = analyse._linfit([0.0, 1.0, 2.0], [3.0, 5.0, 7.0])
    assert a == pytest.approx(3.0)
    assert b == pytest.approx(2.0)
    assert r2 == pytest.approx(1.0)


def test_linfit_rejects_degenerate_input(analyse):
    a, b, _ = analyse._linfit([1.0], [1.0])
    assert math.isnan(a) and math.isnan(b)
    a, b, _ = analyse._linfit([2.0, 2.0], [1.0, 3.0])  # zero x-variance
    assert math.isnan(a) and math.isnan(b)


# ---------------------------------------------------------------------------
# Mohr-Coulomb inverse: the decisive arithmetic for the c = 0 assumption
# ---------------------------------------------------------------------------
def _press_rec(tag: str, confining_kPa: float, q: float, p: float) -> dict:
    return {
        "tag": tag,
        "confining_kPa": confining_kPa,
        "q_kPa": q,
        "p_kPa": p,
        "phi_wall_deg": math.degrees(math.asin(min(1.0, q / (2.0 * p)))),
        "phi_internal_deg": float("nan"),
        "specimen": {"n_particles": 1},
        "loading": {},
    }


@pytest.mark.parametrize("phi_true,c_true", [(30.0, 15.0), (38.0, 0.0), (20.0, 40.0)])
def test_pressure_fit_recovers_known_cohesion_and_angle(analyse, phi_true, c_true):
    """q = 2c cos(phi) + 2 p sin(phi) must be inverted exactly."""
    sp, cp = math.sin(math.radians(phi_true)), math.cos(math.radians(phi_true))
    recs = {}
    for tag, p in (("p050", 50.0), ("p100", 100.0), ("p200", 200.0)):
        recs[tag] = _press_rec(tag, p, 2 * c_true * cp + 2 * p * sp, p)
    sec = analyse.pressure_section(recs)
    assert sec["fit"]["phi_fit_deg"] == pytest.approx(phi_true, abs=1e-6)
    assert sec["fit"]["cohesion_c_kPa"] == pytest.approx(c_true, abs=1e-6)
    assert sec["fit"]["r2"] == pytest.approx(1.0, abs=1e-9)


def test_cohesionless_data_gives_zero_cohesion(analyse):
    """A pressure-independent phi (the c = 0 case) must fit c = 0."""
    sp = math.sin(math.radians(32.0))
    recs = {t: _press_rec(t, p, 2 * p * sp, p)
            for t, p in (("p050", 50.0), ("p100", 100.0), ("p200", 200.0))}
    sec = analyse.pressure_section(recs)
    assert sec["fit"]["cohesion_c_kPa"] == pytest.approx(0.0, abs=1e-6)
    assert sec["phi_spread_deg"] == pytest.approx(0.0, abs=1e-6)


def test_pressure_section_reports_spread_and_bond_control(analyse):
    sp = math.sin(math.radians(30.0))
    recs = {t: _press_rec(t, p, 2 * p * sp, p)
            for t, p in (("p050", 50.0), ("p100", 100.0), ("p200", 200.0))}
    bonded = dict(recs["p100"])
    bonded["phi_wall_deg"] = 41.41
    unbonded = dict(recs["p100"])
    unbonded["tag"] = "p100_b0"
    unbonded["phi_wall_deg"] = 30.0
    recs["p100"] = bonded
    recs["p100_b0"] = unbonded
    sec = analyse.pressure_section(recs)
    assert sec["bond_control"]["delta_phi_deg"] == pytest.approx(11.41, abs=1e-9)
    assert sec["bond_control"]["phi_unbonded_error_vs_ref"] == pytest.approx(
        abs(30.0 - 38.0) / 38.0, abs=1e-12)


def test_pressure_section_survives_missing_cells(analyse):
    sec = analyse.pressure_section({"p100": _press_rec("p100", 100.0, 120.0, 150.0)})
    assert math.isnan(sec["fit"]["phi_fit_deg"])
    assert sec["bond_control"] is None


# ---------------------------------------------------------------------------
# shard/tag contract between the drivers and the analysis
# ---------------------------------------------------------------------------
def test_size_tags_match_the_screen_driver(analyse, screen_mod):
    assert analyse.SIZE_ORDER == list(screen_mod.CELLS)
    for tag in analyse.SIZE_ORDER:
        cfg = screen_mod.CELLS[tag]
        # every size cell must share rate and consolidation length, or the
        # "finite-size" reading is confounded
        assert cfg["strain_rate"] == screen_mod.CELLS[analyse.SIZE_ORDER[0]]["strain_rate"]
        assert cfg["consolidation_steps"] == \
            screen_mod.CELLS[analyse.SIZE_ORDER[0]]["consolidation_steps"]


def test_pressure_tags_match_the_pressure_driver(analyse, pressure_mod):
    assert analyse.PRESS_ORDER == [t for t in pressure_mod.CELLS if t != "p100_b0"]
    assert "p100_b0" in pressure_mod.CELLS


def test_anchor_strain_window_does_not_truncate_the_peak(screen_mod, pressure_mod):
    """The peak moves to eps_a ~ 0.08 at 4000 consolidation steps.

    A 0.10 window truncated it in the first sweep attempt, so both drivers must
    keep the anchor's strain target.
    """
    assert screen_mod.STRAIN_TARGET >= 0.25
    assert pressure_mod.STRAIN_TARGET >= 0.25


def test_bonded_and_unbonded_cells_differ_only_in_freeze_bonds(pressure_mod):
    a = dict(pressure_mod.CELLS["p100"])
    b = dict(pressure_mod.CELLS["p100_b0"])
    assert a.pop("freeze_bond_fraction") != b.pop("freeze_bond_fraction")
    assert a == b


# ---------------------------------------------------------------------------
# committed artefact self-consistency
# ---------------------------------------------------------------------------
ARTEFACT = os.path.join(REPO, "docs", "data",
                        "CASE-DEM-BIAXIAL-FRICTION_validity_audit.json")


@pytest.mark.skipif(not os.path.exists(ARTEFACT), reason="audit artefact not committed")
def test_committed_artefact_reproduces_its_own_cohesion_fit():
    with open(ARTEFACT) as f:
        art = json.load(f)
    rows = art["pressure"]["rows"]
    assert len(rows) >= 3
    q = [r["q_kPa"] for r in rows]
    p = [r["p_kPa"] for r in rows]
    a, b, _ = _linfit_local(p, q)
    assert art["pressure"]["fit"]["intercept_kPa"] == pytest.approx(a, rel=1e-9)
    assert art["pressure"]["fit"]["slope"] == pytest.approx(b, rel=1e-9)
    # the reported phi must be the one implied by the reported slope
    assert art["pressure"]["fit"]["phi_fit_deg"] == pytest.approx(
        math.degrees(math.asin(max(-1.0, min(1.0, b / 2.0)))), abs=1e-9)


def _linfit_local(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b, 0.0