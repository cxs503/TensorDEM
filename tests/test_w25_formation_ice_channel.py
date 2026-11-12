"""Tests for the W25 formation-ice-channel pipeline (library + collector).

The physics tests here are analytic: they pin the arithmetic of the saving, the
Roache GCI, the Lindqvist conversion identities and the geometry rules.  The
end-to-end test feeds the collector a small synthetic raw set (fake loads, real
case metadata) and checks that every verdict is computed, so a broken collector
is caught without a DEM run.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from tensordem.ice_channel_formation import (  # noqa: E402
    FormationCase,
    channel_reclosure_time,
    domain_verdict,
    formation_geometry,
    gci_three_grid,
    lindqvist_reference,
    monotonicity,
    savings_rate,
)
from tensordem.ice_channel_formation_cases import (  # noqa: E402
    BASE_DT_S,
    SPACINGS_L,
    SPEEDS_M_S,
    build_matrix,
    cost_proxy,
    speed_label,
)


# ---------------------------------------------------------------------------
# saving metric
# ---------------------------------------------------------------------------
def test_savings_rate_definition():
    # three ships each carrying 100 kN -> no saving
    assert savings_rate(300.0, 100.0, 3) == pytest.approx(0.0, abs=1e-12)
    # column carries half of three independent ships -> 50 % saving
    assert savings_rate(150.0, 100.0, 3) == pytest.approx(0.5, abs=1e-12)
    # column costs nothing -> 100 % saving
    assert savings_rate(0.0, 100.0, 3) == pytest.approx(1.0, abs=1e-12)
    # column worse than independent ships -> negative saving
    assert savings_rate(450.0, 100.0, 3) == pytest.approx(-0.5, abs=1e-12)


def test_savings_rate_rejects_degenerate_reference():
    with pytest.raises(ValueError):
        savings_rate(10.0, 0.0, 3)
    with pytest.raises(ValueError):
        savings_rate(10.0, -1.0, 3)
    with pytest.raises(ValueError):
        savings_rate(float("nan"), 1.0, 3)
    with pytest.raises(ValueError):
        savings_rate(10.0, 1.0, 0)


# ---------------------------------------------------------------------------
# grid convergence
# ---------------------------------------------------------------------------
def test_gci_recovers_known_order():
    # f(h) = 12 + 3*h^2 with h = 1, 1/2, 1/4 -> observed order must be 2
    exact, slope = 12.0, 3.0
    f = lambda h: exact + slope * h ** 2  # noqa: E731
    fine, medium, coarse = f(0.25), f(0.5), f(1.0)
    out = gci_three_grid(fine, medium, coarse, refinement=2.0)
    assert out["converged"] is True
    assert out["monotone"] is True
    assert out["observed_order"] == pytest.approx(2.0, rel=1e-10)
    # GCI_fine = Fs |(f2-f1)/f1| / (r^p - 1)
    expected = 1.25 * abs(fine - medium) / fine / (2.0 ** 2 - 1.0)
    assert out["gci_pct"] == pytest.approx(100.0 * expected, rel=1e-10)


def test_gci_flags_oscillatory_family():
    out = gci_three_grid(10.0, 12.0, 10.5, refinement=2.0)
    assert out["converged"] is False
    assert math.isnan(out["gci_pct"])
    assert out["monotone"] is False


def test_gci_zero_difference_is_converged():
    out = gci_three_grid(5.0, 5.0, 5.0, refinement=2.0)
    assert out["converged"] is True
    assert out["gci_pct"] == 0.0


def test_gci_rejects_bad_refinement():
    with pytest.raises(ValueError):
        gci_three_grid(1.0, 2.0, 3.0, refinement=1.0)


# ---------------------------------------------------------------------------
# monotonicity
# ---------------------------------------------------------------------------
def test_monotonicity_directions_and_tolerance():
    falling = [0.6, 0.4, 0.2, 0.05]
    assert monotonicity(falling)["monotone"] is True
    assert monotonicity(falling, tolerance=0.0)["n_violations"] == 0
    bumpy = [0.6, 0.4, 0.45, 0.2]
    assert monotonicity(bumpy)["monotone"] is False
    # a bump smaller than the tolerance is not a violation
    assert monotonicity(bumpy, tolerance=0.1)["monotone"] is True
    assert monotonicity([0.1, 0.2, 0.3], sign=1)["monotone"] is True
    assert monotonicity([0.1, 0.2, 0.3])["monotone"] is False


# ---------------------------------------------------------------------------
# Lindqvist conversion
# ---------------------------------------------------------------------------
def test_lindqvist_reference_identities():
    case = FormationCase(label="unit", n_ships=1, speed_m_s=1.0)
    ref = lindqvist_reference(case, 1.0)
    comp = ref["components"]
    total = (comp["crushing_N"] + comp["bending_N"] + comp["submersion_N"]
             + comp["added_drag_N"])
    assert comp["total_N"] == pytest.approx(total, rel=1e-12)
    assert comp["static_N"] == pytest.approx(
        comp["crushing_N"] + comp["bending_N"] + comp["submersion_N"], rel=1e-12)
    for name in ("crushing_N", "bending_N", "submersion_N"):
        assert comp[name] > 0.0
    # coefficient provenance is carried, not invented
    assert comp["provenance"]
    assert comp["ledger_sha256"]


def test_lindqvist_box_hull_is_out_of_domain():
    """The DEM hull is a box (90 deg stem); the formula rejects that angle."""
    from tensordem.level_ice.formula import HullForm, IceState, crushing_resistance

    ice = IceState(thickness_m=0.7)
    with pytest.raises(ValueError):
        HullForm(length_m=80.0, beam_m=20.0, draught_m=0.6, entry_angle_deg=90.0)
    # and the crushing term diverges towards it
    a15 = crushing_resistance(ice, HullForm(entry_angle_deg=15.0, friction_coefficient=0.05))
    a30 = crushing_resistance(ice, HullForm(entry_angle_deg=30.0, friction_coefficient=0.05))
    assert a30 > a15


def test_domain_verdict_grades_and_detects_out_of_domain():
    case = FormationCase(label="unit", n_ships=1, speed_m_s=1.0)
    anchor = lindqvist_reference(case, 1.0)
    exact = domain_verdict(anchor["total_N"], anchor)
    assert exact["relative_deviation_pct"] == pytest.approx(0.0, abs=1e-9)
    assert exact["within_tolerance"] is True

    ten_pct_low = domain_verdict(0.9 * anchor["total_N"], anchor)
    assert ten_pct_low["within_tolerance"] is True
    assert ten_pct_low["relative_deviation_pct"] == pytest.approx(-10.0, abs=1e-9)

    far_low = domain_verdict(0.2 * anchor["total_N"], anchor)
    assert far_low["within_tolerance"] is False
    assert far_low["dem_below_anchor_floor"] is True
    assert far_low["anchor_over_dem_ratio"] == pytest.approx(5.0, rel=1e-9)


# ---------------------------------------------------------------------------
# geometry / case matrix
# ---------------------------------------------------------------------------
def test_field_length_rules():
    single = FormationCase(label="s", n_ships=1, speed_m_s=1.0)
    assert single.field_length_m == pytest.approx(6.0 * single.ship_length_m)
    for spacing in SPACINGS_L:
        col = FormationCase(label=f"c{spacing}", n_ships=3, spacing_L=spacing, speed_m_s=1.0)
        assert col.span_m == pytest.approx(2.0 * spacing * col.ship_length_m)
        # the field always holds the whole column plus two hull lengths of ice
        assert col.field_length_m >= col.span_m + 2.0 * col.ship_length_m - 1e-9
        geom = formation_geometry(col)
        assert geom["window_hi_m"] > geom["window_lo_m"]
        # the trailing hull starts outside the ice and every hull traverses ice
        assert geom["x_trailing_start_m"] < 0.0
        assert geom["travel_m"] == pytest.approx(geom["window_hi_m"] - geom["x_trailing_start_m"])
        assert geom["window_length_m"] >= col.ship_length_m - 1e-9


def test_reclosure_time_and_lag():
    case = FormationCase(label="r", n_ships=3, spacing_L=2.0, speed_m_s=1.0,
                         ship_beam_m=20.0, drift_m_s=0.05)
    out = channel_reclosure_time(case)
    assert out["corridor_half_width_m"] == pytest.approx(10.0)
    assert out["reclosure_time_s"] == pytest.approx(200.0)
    assert out["follower_lag_s"] == pytest.approx(160.0)
    assert out["lag_over_reclosure"] == pytest.approx(0.8)


def test_case_matrix_invariants():
    matrix = build_matrix()
    labels = [item.label for item in matrix]
    assert len(labels) == len(set(labels)), "case labels must be unique"
    roles = {item.role for item in matrix}
    assert roles == {"single", "formation", "gci", "numerics"}
    formations = [item for item in matrix if item.role == "formation"]
    # full spacing x speed surface, three seeds each
    assert {(item.spacing_L, item.speed_m_s) for item in formations} == {
        (s, v) for s in SPACINGS_L for v in SPEEDS_M_S}
    assert len(formations) == len(SPACINGS_L) * len(SPEEDS_M_S) * 3
    singles = [item for item in matrix if item.role == "single"]
    assert len(singles) == len(SPEEDS_M_S) * 5
    # refinement family keeps the contact CFL by scaling dt with the floe radius
    base = min((item for item in matrix
                if item.label == "single_V1p00"), key=lambda item: item.mesh_level)
    for level in (2, 4):
        refined = next(item for item in matrix
                       if item.label == f"single_V1p00_Nx{level}")
        assert refined.dt == pytest.approx(BASE_DT_S / math.sqrt(level))
        assert refined.mean_radius_refined_m == pytest.approx(
            base.mean_radius_m / math.sqrt(level))
    assert all(cost_proxy(item) > 0.0 for item in matrix)


# ---------------------------------------------------------------------------
# collector end-to-end on a synthetic raw set
# ---------------------------------------------------------------------------
def _fake_run(case: FormationCase, loads: list[float]) -> dict:
    per_hull = [
        {
            "index": index,
            "x_start_m": 0.0,
            "samples": 100,
            "mean_upstream_load_N": load,
            "mean_abs_load_N": load,
            "contact_fraction": 0.5,
        }
        for index, load in enumerate(loads)
    ]
    payload = {
        "label": case.label,
        "case": case.to_dict(),
        "geometry": formation_geometry(case),
        "steps_planned": 10,
        "steps_done": 10,
        "sim_time_s": 1.0,
        "dt": case.dt,
        "blew_up": False,
        "max_abs_load_N": max(loads) * 3.0,
        "n_floes": 50,
        "n_floes_carved_at_launch": 4,
        "realised_concentration": 0.8,
        "max_initial_overlap_m": 1e-7,
        "per_hull": per_hull,
        "formation_total_load_N": sum(loads),
        "series": {"t": [0.0], "loads_N": [[load for load in loads]]},
        "wall_time_s": 1.0,
    }
    return payload


def test_collector_end_to_end(tmp_path):
    import w25_formation_collect as collector

    raw = tmp_path / "raw"
    raw.mkdir()
    single_loads = {0.5: 40e3, 1.0: 60e3, 2.0: 150e3}
    for speed, load in single_loads.items():
        for offset in (0, 1, 2, 3, 4):
            case = FormationCase(
                label=f"single_V{speed_label(speed)}"
                      + ("" if offset == 0 else f"_e{offset}"),
                n_ships=1, speed_m_s=speed, seed=20261112 + 1000 * offset, role="single")
            (raw / f"{case.label}.json").write_text(json.dumps(_fake_run(case, [load])))
    # formation: saving falls with spacing, followers load up as spacing grows
    for speed in single_loads:
        for spacing in SPACINGS_L:
            # the lead's corridor re-fills with drift, so the follower share of
            # the lead load GROWS with spacing -> the saving falls with spacing
            fraction = 0.30 + 0.08 * spacing
            lead = single_loads[speed]
            loads = [lead, lead * fraction, lead * fraction * 0.95]
            for offset in (0, 1, 2):
                case = FormationCase(
                    label=f"form_V{speed_label(speed)}_S{spacing:g}L"
                          + ("" if offset == 0 else f"_e{offset}"),
                    n_ships=3, spacing_L=spacing, speed_m_s=speed,
                    seed=20261112 + 1000 * offset, role="formation")
                (raw / f"{case.label}.json").write_text(json.dumps(_fake_run(case, loads)))
    # refinement family
    for level, load in ((1, 60e3), (2, 62e3), (4, 62.5e3)):
        suffix = "" if level == 1 else f"_Nx{level}"
        case = FormationCase(label=f"single_V1p00{suffix}", n_ships=1, speed_m_s=1.0,
                             mesh_level=level, dt=BASE_DT_S / math.sqrt(level), role="gci")
        (raw / f"{case.label}.json").write_text(json.dumps(_fake_run(case, [load])))
        case_f = FormationCase(label=f"form_V1p00_S1L{suffix}", n_ships=3, spacing_L=1.0,
                               speed_m_s=1.0, mesh_level=level,
                               dt=BASE_DT_S / math.sqrt(level), role="gci")
        (raw / f"{case_f.label}.json").write_text(
            json.dumps(_fake_run(case_f, [120e3, 60e3, 55e3])))

    artifact = collector.build_artifact(raw, figures=False)
    assert artifact["task_key"] == "paper-exp-formation-ice-channel-2-rework1"
    assert artifact["geometry_check"]["block_coefficient_Cb"] == 1.0
    assert artifact["case_matrix"]["n_cases_with_results"] > 0
    # every speed present in the synthetic set is solved
    for speed_key in ("0.5", "1", "2"):
        assert speed_key in artifact["single_ship_vs_lindqvist"]
        assert speed_key in artifact["formation_savings"]
    # the synthetic saving falls with spacing -> monotone verdict
    checks = artifact["monotonicity"]
    assert checks["1"]["saving_paired_strict"]["monotone"] is True
    assert checks["1"]["saving_paired"][0] > checks["1"]["saving_paired"][-1]
    statuses = {item["id"]: item["status"] for item in artifact["verdicts"]}
    assert set(statuses) == {"C1", "C2a", "C2b", "C3"}
    assert statuses["C1"] in {"PASS", "FAIL", "WITHHELD"}
    assert artifact["grid_convergence"]["single_ship"]["gci"]["converged"] is True
    assert artifact["overall"]["n_pass"] + artifact["overall"]["n_fail"] + \
        artifact["overall"]["n_withheld"] == 4
    # numbers are JSON-serialisable (no tensors / NaN sentinels leaking)
    json.dumps(artifact)