"""Gates for the disordered (settle-then-release) column wave (m20811)."""
from __future__ import annotations

import importlib.util
import json
import math
import os

import numpy as np
import pytest

from tensordem.granular_column.pluviation import (EQUILIBRIUM_OVERLAP_FRAC,
                                                  RelaxCase,
                                                  cylinder_wall_contact_forces,
                                                  disorder_metrics,
                                                  prepare_disordered_column,
                                                  settle_report)
from tensordem.granular_column.axisym import (AxisymCase, build_axisymmetric_column,
                                             simulate_axisymmetric_column)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "docs", "artifacts",
                      "column_collapse_disordered_m20811")


def _material(radius=0.03):
    case = RelaxCase(radius=radius)
    return case.material()


# --- cylindrical side wall -------------------------------------------------

def test_wall_force_is_zero_outside_the_container():
    mat = _material()
    pos = np.array([[0.10, 0.0, 0.03]])
    out = cylinder_wall_contact_forces(pos, np.zeros((1, 3)), np.zeros((1, 3)),
                                       mat, 4.8e-4, wall_radius=0.24)
    assert out["n_contacts"] == 0
    assert np.allclose(out["force"], 0.0)


def test_wall_force_points_inward_and_scales_with_penetration():
    mat = _material()
    pos = np.array([[0.215, 0.0, 0.03]])
    pen = 0.215 + 0.03 - 0.24
    out = cylinder_wall_contact_forces(pos, np.zeros((1, 3)), np.zeros((1, 3)),
                                       mat, 4.8e-4, wall_radius=0.24,
                                       contact_mode="one_sided")
    assert out["n_contacts"] == 1
    assert out["force"][0, 0] == pytest.approx(-mat.kn * pen, rel=1e-9)
    assert out["force"][0, 1] == pytest.approx(0.0, abs=1e-12)


def test_wall_rejects_invalid_radius():
    mat = _material()
    pos = np.array([[0.1, 0.0, 0.03]])
    with pytest.raises(ValueError):
        cylinder_wall_contact_forces(pos, np.zeros((1, 3)), np.zeros((1, 3)),
                                     mat, 4.8e-4, wall_radius=0.0)


# --- equilibrium gate ------------------------------------------------------

def test_settle_report_flags_overlap_and_aspect():
    r = 0.03
    pos = np.array([[0.0, 0.0, r], [2.0 * r - 1e-4, 0.0, r], [0.0, 0.0, 3 * r]])
    rep = settle_report(pos, r, 0.24)
    assert rep["max_overlap"] > 0.0
    assert rep["overlap_pairs"] == 1
    assert rep["max_overlap_frac_of_radius"] > 0.0
    assert rep["aspect_measured"] == pytest.approx(rep["column_height_measured"]
                                                  / rep["column_radius_measured"])
    assert rep["tolerance_frac_of_radius"] == EQUILIBRIUM_OVERLAP_FRAC


def test_disorder_metric_separates_lattice_from_random():
    ac = AxisymCase(aspect_ratio=1.0, grains_per_radius=6.0, radius=0.03,
                    density=2500.0, mu=0.5, mu_rolling=0.5, kn=6e4)
    pos, _ = build_axisymmetric_column(ac)
    lat = disorder_metrics(pos, 0.03)
    rng = np.random.default_rng(0)
    rand = disorder_metrics(rng.uniform(-0.5, 0.5, size=pos.shape) + [0, 0, 1.0],
                            0.03)
    assert lat["hex_order"] > rand["hex_order"]
    assert 0.0 <= rand["hex_order"] <= 1.0


# --- preparation -----------------------------------------------------------

def _tiny_relax(**kw):
    base = dict(column_radius=0.12, radius=0.03, aspect_ratio=0.5, seed=11,
                dt=4.8e-4, jitter_diameters=0.1, max_settle_time=1.0,
                min_settle_time=0.02, ke_tol=1e-2)
    base.update(kw)
    return RelaxCase(**base)


def test_preparation_is_deterministic_and_gated():
    rep = prepare_disordered_column(_tiny_relax())
    again = prepare_disordered_column(_tiny_relax())
    assert np.allclose(np.asarray(rep["positions"]),
                       np.asarray(again["positions"]))
    assert rep["n_particles"] > 0
    assert "hex_order" in rep and 0.0 <= rep["hex_order"] <= 1.0
    # every pair is separated by at least a contact overlap carried by gravity
    assert rep["max_overlap_frac_of_radius"] < 0.05
    assert rep["kinetic_energy_at_rest"] >= 0.0


def test_seed_changes_the_prepared_state():
    a = prepare_disordered_column(_tiny_relax(seed=11))
    b = prepare_disordered_column(_tiny_relax(seed=22))
    assert not np.allclose(np.asarray(a["positions"]),
                           np.asarray(b["positions"]))


def test_collapse_accepts_a_prepared_state():
    rep = prepare_disordered_column(_tiny_relax())
    pos = np.asarray(rep["positions"])
    ac = AxisymCase(aspect_ratio=0.5, grains_per_radius=3.0, radius=0.03,
                    density=2500.0, mu=0.5, mu_rolling=0.5, mu_wall=0.8,
                    restitution=0.5, kn=6e4, duration_factor=2.0, min_time=0.05,
                    rest_ke_tol=1e-4, rest_hold=50)
    res = simulate_axisymmetric_column(ac, return_positions=False,
                                       initial_positions=pos, max_steps=60)
    assert res["n_particles"] == pos.shape[0]
    assert res["runout_ratio_p995"] >= 1.0


def test_collapse_rejects_malformed_initial_positions():
    ac = AxisymCase(aspect_ratio=0.5, grains_per_radius=3.0, radius=0.03)
    with pytest.raises(ValueError):
        simulate_axisymmetric_column(ac, initial_positions=np.zeros((3, 2)))


# --- reduction / verdicts --------------------------------------------------

def _load_script():
    path = os.path.join(ROOT, "scripts",
                        "analyze_column_collapse_disordered_m20811.py")
    spec = importlib.util.spec_from_file_location("m20811_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(**kw):
    base = dict(case="x", phase="disordered", aspect_nominal=0.5,
                aspect_measured=0.55, jitter_diameters=0.2, seed=1,
                n_particles=100, packing_fraction=0.60,
                packing_fraction_poured=float("nan"), max_overlap_frac=0.005,
                ke_at_rest=1e-6, at_rest=True, hex_order=0.15, coordination=6.0,
                relax_time=0.3, runout_ratio=2.0, reference=2.0, err=0.0,
                stop_reason="at_rest", lattice_hex_order=0.32, file="x.json")
    base.update(kw)
    return base


def test_verdicts_accept_a_clean_wave():
    mod = _load_script()
    rows = [_row(case="m0", phase="lattice_control", hex_order=0.31,
                 lattice_hex_order=0.32),
            _row(case="d1", runout_ratio=2.05, err=0.025),
            _row(case="d2", runout_ratio=1.98, err=-0.01),
            _row(case="d3", runout_ratio=2.10, err=0.05)]
    out = mod.adjudicate(rows)
    assert out["verdicts"]["C_PHI"] and out["verdicts"]["C_EQ"]
    assert out["verdicts"]["C_DISORDER"] and out["verdicts"]["C_BAND"]
    assert out["verdicts"]["C_SPREAD"]


def test_verdicts_catch_band_and_disorder_failures():
    mod = _load_script()
    rows = [_row(case="m0", phase="lattice_control", hex_order=0.31,
                 lattice_hex_order=0.32),
            _row(case="d1", runout_ratio=2.5, err=0.25, hex_order=0.31),
            _row(case="d2", packing_fraction=0.40)]
    out = mod.adjudicate(rows)
    assert not out["verdicts"]["C_BAND"]
    assert not out["verdicts"]["C_DISORDER"]
    assert not out["verdicts"]["C_PHI"]


def test_spread_and_empty_input():
    mod = _load_script()
    assert mod._spread([1.0, 1.0]) == 0.0
    assert math.isnan(mod._spread([1.0]))
    assert math.isnan(mod._spread([]))
    assert mod.adjudicate([])["incomplete"] is True


@pytest.mark.skipif(not os.path.exists(os.path.join(BUNDLE, "cases_disordered.csv")),
                    reason="wave artifacts not produced yet")
def test_bundled_csv_regression():
    mod = _load_script()
    rows = mod.load_cases()
    assert len(rows) >= 4
    for row in rows:
        assert 0.0 < row["packing_fraction"] < 0.75
        assert row["reference"] == pytest.approx(
            mod.lube_2004_runout_ratio(row["aspect_measured"]))
