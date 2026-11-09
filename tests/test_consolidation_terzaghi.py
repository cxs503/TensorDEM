"""Acceptance tests for the IN99-N08 consolidation chain (TensorDEM).

Each test pins one link of the chain, in the same order the report walks
through it:

1. the Fourier anchor and its short-time branch (independent of everything);
2. the two-field column against that anchor (C1), settlement/inversion (C2);
3. refinement and negative-control discrimination (C3 + gate power);
4. the DEM oedometer primitives and a short integration smoke test;
5. the applicability gates, fail-closed behaviour;
6. the exported ``r_u(T_v)`` table and its downstream consumer (C4).
"""
from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.consolidation import (  # noqa: E402
    TV_AT_U50,
    TV_AT_U90,
    ConsolidationDomainError,
    ColumnParameters,
    OedometerCell,
    OedometerConfig,
    PoroelasticColumn,
    PrototypeBed,
    Tv_of_U,
    U_series,
    U_sqrt_approx,
    U_two_branch,
    build_ru_table,
    carman_kozeny_conductivity,
    check_applicability,
    ci95,
    invert_cv,
    invert_settlement,
    pore_reynolds,
    propagate_cv_uncertainty,
    r_u_at,
    r_u_profile,
    read_ru_table,
    run_chain,
    table_sha256,
    u_profile,
    write_ru_table,
)

#: A cheap, physically plausible DEM result used wherever the DEM itself is not
#: under test (the DEM has its own smoke test below).
SYNTHETIC_DEM = {
    "E_oed": 1.29e6,
    "E_oed_mid": 1.28e6,
    "k_hyd": 1.81e-3,
    "porosity": 0.50,
    "coordination": 2.6,
    "strain_increment": 7.75e-3,
    "total_steps": 65200,
    "stages": [
        {"stress": 5.0e3, "height": 1.159e-3, "porosity": 0.505, "coordination": 2.5},
        {"stress": 15.0e3, "height": 1.150e-3, "porosity": 0.495, "coordination": 2.7},
    ],
    "config": {"diameter": 2.0e-4, "particle_count": 80, "kn": 2000.0, "radius": 1.0e-4,
               "density": 2650.0, "stress_levels": [5.0e3, 15.0e3], "dt": 5.23e-7,
               "box_width": 8.08e-4, "n_row": 4, "n_layer": 5, "seed": 20261108},
}


# ----------------------------------------------------------------- 1. anchor
def test_series_anchors_match_tabulated_time_factors():
    """Terzaghi's tabulated T_v(50 %) = 0.197 and T_v(90 %) = 0.848."""
    assert abs(Tv_of_U(0.5) - TV_AT_U50) / TV_AT_U50 < 0.01
    assert abs(Tv_of_U(0.9) - TV_AT_U90) / TV_AT_U90 < 0.01


def test_series_is_monotone_and_bounded():
    Tv = np.logspace(-5, 1, 200)
    U = np.asarray(U_series(Tv))
    assert np.all(np.diff(U) >= 0.0)          # non-decreasing everywhere
    assert np.all(np.diff(U)[:150] > 0.0)     # strictly increasing while U < 1
    assert 0.0 <= U[0] < 0.01
    assert U[-1] > 0.999
    assert U_series(0.0) == pytest.approx(0.0, abs=1e-12)


def test_sqrt_branch_agrees_below_switch_and_is_monotone():
    """The sqrt(4T_v/pi) branch must be accurate to < 1 % up to U = 0.53."""
    Tv = np.linspace(1e-4, 0.2206, 500)
    ratio = np.abs(U_sqrt_approx(Tv) - U_series(Tv)) / np.asarray(U_series(Tv))
    assert ratio.max() < 0.01
    both = np.asarray(U_two_branch(Tv))
    assert np.all(np.diff(both) > 0.0)
    # continuity of the two-branch rule at the switch
    lo, hi = 0.2206 - 1e-6, 0.2206 + 1e-6
    assert abs(float(U_two_branch(lo)) - float(U_two_branch(hi))) < 1e-3


def test_profile_boundaries_and_mass_consistency():
    """Drained face u = 0, impermeable face du/dz = 0, mean = 1 - U."""
    Tv = 0.5
    z = np.linspace(0.0, 1.0, 2001)
    prof = np.asarray(u_profile(z, Tv))
    assert abs(prof[0]) < 1e-12
    dz = z[1] - z[0]
    assert abs((prof[-3] - prof[-1]) / (2 * dz)) < 5e-3
    mean = np.trapezoid(prof, z)
    assert abs(mean - (1.0 - float(U_series(Tv)))) < 5e-3


# ------------------------------------------------------- 2. column vs anchor
def _column(n_cells: int = 40, E: float = 1.29e6, k: float = 1.81e-3):
    params = ColumnParameters(height=1.0, stress_step=10.0e3, drainage="double")
    return PoroelasticColumn(params=params, k_hyd=k, E_oed=E, n_cells=n_cells)


def test_column_reproduces_terzaghi_c1():
    """C1: five time factors in [0.05, 2], deviation <= 5 % point by point."""
    col = _column(40)
    out = col.run((0.05, 0.2, 0.5, 1.0, 2.0))
    devs = []
    for sample in out["samples"]:
        Tv = sample["T_v"]
        dev = abs(sample["U_simulated"] - float(U_series(Tv))) / float(U_series(Tv))
        devs.append(dev)
        assert dev <= 0.05, (Tv, sample["U_simulated"], float(U_series(Tv)))
    assert max(devs) < 0.01  # in practice far below the 5 % threshold


def test_column_settlement_and_cv_inversion_c2():
    """C2: inverted C_v within 10 % of k E_oed/gamma_w, s_f within 5 %."""
    col = _column(40)
    out = col.run((0.05, 0.2, 0.5, 1.0, 2.0))
    cv_input = col.cv_assembled()
    hist = out["history"]
    inv = invert_cv(hist["t"], hist["U"], col.h_dr, method="t50")
    assert abs(inv["C_v"] - cv_input) / cv_input < 0.10
    s_inv = invert_settlement(hist["t"], hist["settlement"])
    s_analytic = 10.0e3 * 1.0 / 1.29e6
    assert abs(s_inv["s_final"] - s_analytic) / s_analytic < 0.05
    assert out["settlement_final"] == pytest.approx(s_analytic, rel=1e-12)


def test_column_layers_drain_two_way():
    """Half-bed symmetry solve: drained face -> zero, mid-plane -> maximum."""
    col = _column(41)
    col.advance_to(col.time_of(0.2))
    r_u = np.asarray(col.u) / 10.0e3
    assert r_u[0] < 0.05                     # drained face has almost emptied
    assert int(np.argmax(r_u)) == len(r_u) - 1  # impermeable face = full-bed mid-plane
    assert np.all(np.diff(r_u) > -1e-12)     # monotone increasing with depth


def test_heterogeneous_column_conserves_mass_and_stays_monotone():
    params = ColumnParameters(height=1.0, stress_step=10.0e3, drainage="double")
    E = np.array([0.8e6] * 10 + [2.0e6] * 10 + [1.2e6] * 20)
    col = PoroelasticColumn(params=params, k_hyd=1.81e-3, E_oed=E, n_cells=40)
    out = col.run((0.05, 0.5, 2.0))
    U = [s["U_simulated"] for s in out["samples"]]
    assert all(b >= a for a, b in zip(U, U[1:]))
    assert 0.0 < U[0] < U[-1] < 1.0
    # the final settlement is the weighted 1-D compression of the layered column
    expected = 10.0e3 * 1.0 * float(np.mean(1.0 / E))
    assert out["settlement_final"] == pytest.approx(expected, rel=1e-12)


# ------------------------------------------------- 3. refinement / power C3, C5
def test_refinement_drift_is_small_c3():
    runs = []
    for n in (20, 40, 80):
        col = _column(n)
        runs.append([s["U_simulated"] for s in col.run((0.05, 0.2, 0.5, 1.0, 2.0))["samples"]])
    for better in runs[:-1]:
        drift = float(np.max(np.abs(np.asarray(better) - np.asarray(runs[-1]))) * 100.0)
        assert drift < 5.0


def test_negative_control_flips_the_gate():
    """A 20 % error in C_v must break the 5 % C1 gate (the gate has teeth)."""
    col = _column(40)
    ref = [s["U_simulated"] for s in col.run((0.05, 0.2, 0.5, 1.0, 2.0))["samples"]]
    for factor in (0.8, 1.2):
        col_n = _column(40, E=1.29e6 * factor)
        shifts = [Tv * factor for Tv in (0.05, 0.2, 0.5, 1.0, 2.0)]
        shifted = [s["U_simulated"] for s in col_n.run(shifts)["samples"]]
        dev = max(abs(a - b) / b for a, b in zip(shifted, ref)) * 100.0
        assert dev > 5.0, f"factor {factor} only deviates {dev:.2f} %"


def test_chain_reports_every_gate():
    out = run_chain(dem=SYNTHETIC_DEM, prototype=PrototypeBed(height=1.0, stress_step=10e3),
                    dem_ensemble=[SYNTHETIC_DEM])
    assert out["C1"]["pass"] and out["C2"]["pass"] and out["C3"]["pass"]
    assert out["C1"]["max_deviation_pct"] <= 5.0
    assert out["C2"]["C_v_deviation_pct"] <= 10.0
    assert out["C2"]["s_f_deviation_pct"] <= 5.0
    assert out["C3"]["max_drift_pct"] <= 5.0
    assert out["applicability"]["applicable"]
    assert all(n["would_fail_C1"] for n in out["negative_control"])
    assert len(out["C1"]["points"]) == 5


# --------------------------------------------------- 4. DEM oedometer pieces
def test_carman_kozeny_matches_hand_computation():
    n, d = 0.40, 2.0e-4
    k_int = n ** 3 * d ** 2 / (180.0 * (1.0 - n) ** 2)
    K = carman_kozeny_conductivity(n, d)
    assert K == pytest.approx(k_int * 9810.0 / 1.0e-3, rel=1e-12)


def test_pore_reynolds_definition():
    Re = pore_reynolds(conductivity=1.0e-5, gradient=2.0, grain_diameter=1.0e-4)
    assert Re == pytest.approx(1000.0 * 1.0e-5 * 2.0 * 1.0e-4 / 1.0e-3, rel=1e-12)


def test_dem_oedometer_cell_integrates_without_diverging():
    """Short smoke run: finite forces, decreasing porosity under the piston."""
    cfg = OedometerConfig(n_row=3, n_layer=3, unjam_steps=50,
                          max_steps_per_stage=2000, check_every=50)
    cell = OedometerCell(cfg)
    assert cfg.particle_count == cell.n == 27
    n0 = cell.porosity()
    assert 0.6 < n0 < 0.9  # deposited loose bed
    assert np.all(np.isfinite(cell.forces()))
    for _ in range(200):
        cell.step()
    cell.gravity_on = True
    report = cell.load_to(cfg.stress_levels[0])
    assert np.isfinite(report["lid_pressure"])
    assert cell.porosity() < n0  # the piston compacted the bed
    assert np.all(np.isfinite(cell.pos))


# --------------------------------------------------------- 5. domain gates
def _gate(**kw):
    base = dict(saturation=1.0, pore_reynolds=0.05, porosity=0.45,
                chemical_cementation=False, provenance="granular_bed")
    base.update(kw)
    return check_applicability(**base)


def test_gate_accepts_a_darcy_saturated_uncemented_bed():
    report = _gate()
    assert report.applicable
    assert all(g["pass"] for g in report.gates.values())


@pytest.mark.parametrize("kwargs,reason", [
    ({"pore_reynolds": 1.5}, "darcy"),
    ({"saturation": 0.9}, "saturated"),
    ({"chemical_cementation": True}, "no_cementation"),
    ({"provenance": "ice_rubble_thaw"}, "not_thawing_ice_rubble"),
])
def test_gate_fails_closed(kwargs, reason):
    report = _gate(**kwargs)
    assert not report.applicable
    assert any(reason in r for r in report.reasons)
    with pytest.raises(ConsolidationDomainError):
        check_applicability(strict=True, **{**dict(saturation=1.0, pore_reynolds=0.05,
                                                   porosity=0.45), **kwargs})


def test_porosity_outside_carman_kozeny_window_is_advisory_not_blocking():
    report = _gate(porosity=0.60)
    assert report.applicable
    assert report.advisories and "carman_kozeny" in report.advisories[0]


# ------------------------------------------------- 6. r_u table and consumer
def test_ru_table_roundtrip_and_tamper_detection(tmp_path):
    table = build_ru_table(
        Tv_grid=(0.001, 0.01, 0.1, 0.5, 1.0, 2.0),
        drainage="double",
        drainage_path=0.5,
        C_v=0.238,
        C_v_ci=ci95([0.23, 0.24, 0.235]),
        applicability=_gate().as_dict(),
    )
    info = write_ru_table(tmp_path / "t.json", table)
    loaded = read_ru_table(tmp_path / "t.json")
    assert info["sha256"] == table_sha256(table)
    assert loaded["T_v"] == table["T_v"]
    # below the first tabulated T_v the table clamps to its first sample
    assert r_u_at(loaded, 0.0) == pytest.approx(loaded["r_u_average"][0], rel=1e-12)
    # inside the grid it interpolates the published values
    assert r_u_at(loaded, 1.0) == pytest.approx(float(1.0 - U_series(1.0)), rel=1e-6)
    depth, prof = r_u_profile_at_from_table(loaded, 0.5)
    assert len(depth) == len(prof)

    # edit a value -> the stored hash no longer matches
    payload = json.loads((tmp_path / "t.json").read_text())
    payload["r_u_average"][2] = 0.123
    (tmp_path / "t.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        read_ru_table(tmp_path / "t.json")

    # a non-monotone r_u must be refused even if the hash is recomputed
    payload = json.loads((tmp_path / "t.json").read_text())
    payload["r_u_average"] = sorted(payload["r_u_average"])
    payload.pop("integrity", None)
    (tmp_path / "t.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        read_ru_table(tmp_path / "t.json")


def r_u_profile_at_from_table(table, Tv):
    from tensordem.consolidation import r_u_profile_at
    return r_u_profile_at(table, Tv)


def test_consumer_script_end_to_end(tmp_path):
    """C4: the exported table is read by the downstream criterion script."""
    chain = run_chain(dem=SYNTHETIC_DEM,
                      prototype=PrototypeBed(height=1.0, stress_step=10e3),
                      dem_ensemble=[SYNTHETIC_DEM])
    table_path = tmp_path / "ru_table.json"
    write_ru_table(table_path, chain["r_u_table"])
    out_path = tmp_path / "consumer.json"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "in99_n08_liquefaction_consumer.py"),
         "--table", str(table_path), "--out", str(out_path)],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin",
             "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "HOME": "/root"},
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_path.read_text())
    assert payload["all_pass"], payload["assertions"]
    assert payload["table"]["sha256"] == table_sha256(chain["r_u_table"])
    assert payload["liquefaction_window_T_v"] > 0.0


def test_consumer_refuses_a_table_outside_the_domain(tmp_path):
    """Fail-closed: a table stamped 'ice_rubble_thaw' must be refused."""
    bad = build_ru_table(
        Tv_grid=(0.001, 0.01, 0.1, 0.5, 1.0),
        drainage="double",
        drainage_path=0.5,
        C_v=0.238,
        applicability=_gate(provenance="ice_rubble_thaw").as_dict(),
    )
    table_path = tmp_path / "bad.json"
    write_ru_table(table_path, bad)
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "in99_n08_liquefaction_consumer.py"),
         "--table", str(table_path), "--out", str(tmp_path / "out.json")],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin",
             "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "HOME": "/root"},
    )
    assert proc.returncode == 3
    assert "REFUSED" in proc.stderr


# --------------------------------------------------------------- statistics
def test_student_t_confidence_interval():
    stats = ci95([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["se"] == pytest.approx(math.sqrt(2.5) / math.sqrt(5.0))
    # t_{0.975, 4} = 2.776 -> half width 2.776 * se
    assert stats["ci_high"] - stats["mean"] == pytest.approx(2.7764451 * stats["se"], rel=1e-5)


def test_cv_uncertainty_propagation_is_paired():
    E = [1.2e6, 1.3e6, 1.4e6]
    k = [1.8e-3, 1.85e-3, 1.9e-3]
    stats = propagate_cv_uncertainty(E, k, 9810.0)
    cv = [ki * ei / 9810.0 for ei, ki in zip(E, k)]
    assert stats["mean"] == pytest.approx(float(np.mean(cv)))
    assert stats["ci_low"] < stats["mean"] < stats["ci_high"]
    assert stats["n"] == 3