"""CASE-DEM-PLANE-SHEAR-MUIDI round 7: pre-registration and adjudication tests.

Round 6 (m16858) shipped the bulk Love-Weber estimator plus a pre-registration
file but never ran the experiment, so the R5 verdict "mu(I) WITHHELD at
I >= 0.1" was never adjudicated.  ``benchmarks/plane_shear_mu_i/round7_protocol.py``
executes that pre-registration and adds two explanation arms (layer count,
servo bandwidth).

What is locked in here (cheap, no sweep required):

* the pre-registration is complete, its I points stay inside the case range
  ``I = 1e-3 .. 0.3``, and the reference law it judges against is the
  da Cruz (2005) / GDR MiDi (2004) ``mu(I)`` law;
* the adjudicator returns PASS only when *every* point of an arm satisfies the
  criterion -- the failure mode this case has repeatedly suffered (a marginal
  point reported as a pass) is turned into a test;
* a band verdict is taken on the better of the two stress estimators, so a wall
  traction that happens to land in the band cannot hide a bulk estimate that
  does not (and vice versa);
* the falsification line of the round-5 verdict fires exactly when a bulk point
  lands inside the 10 % band;
* the device contract: ``run_point`` returns every field the tables and the
  adjudicator read, on a deliberately tiny specimen.

Numbers used in the synthetic rows are *not* measurements; they exist only to
exercise the verdict logic.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO / "src"), str(REPO / "benchmarks" / "plane_shear_mu_i")):
    if p not in sys.path:
        sys.path.insert(0, p)

R7 = importlib.import_module("round7_protocol")


# --------------------------------------------------------------- helpers
def synth_row(arm: str = "A_baseline", I: float = 0.1, *, mu_bulk: float,
              mu_wall: float | None = None, p_ratio: float = 1.0,
              rate_ratio: float = 1.0, servo_valid: bool = True,
              recover_steps: int = 0, geom: str = "baseline",
              n_particles: int = 294) -> dict:
    """A row with the same schema ``run_point`` produces (values synthetic)."""
    mu_wall = mu_bulk if mu_wall is None else mu_wall
    ref = R7.mu_ref(I)
    return {
        "arm": arm, "geom": geom, "I_target": I, "mu_ref": ref,
        "n_particles": n_particles,
        "I_measured": I, "I_bulk": I, "I_bulk_final": I,
        "mu_wall": mu_wall, "mu_wall_final": mu_wall,
        "mu_bulk": mu_bulk, "mu_bulk_final": mu_bulk,
        "p_wall": R7.P_APPLIED * p_ratio, "p_bulk": R7.P_APPLIED * p_ratio,
        "p_bulk_final": R7.P_APPLIED * p_ratio, "tau_wall": 0.4 * R7.P_APPLIED,
        "bulk_kinetic_pressure_pa": 100.0,
        "shear_rate_imposed": 50.0,
        "shear_rate_measured": 50.0 * rate_ratio,
        "profile_r2": 0.99,
        "phi_bed_settled": 0.58, "phi_bed_steady": 0.57, "phi_bed_final": 0.56,
        "coordination": 4.0, "dilatancy": -0.01,
        "pressure_control_error": abs(p_ratio - 1.0),
        "pressure_control_error_final": abs(p_ratio - 1.0),
        "servo_valid": servo_valid, "recover_steps": recover_steps,
        "shear_steps": 300, "shear_steps_executed": 300 + recover_steps,
        "shear_strain_achieved": 0.2 + 0.02 * recover_steps,
        "wall_clock_s": 1.0,
        "mu_wall_err_pct": 100.0 * (mu_wall / ref - 1.0),
        "mu_bulk_err_pct": 100.0 * (mu_bulk / ref - 1.0),
        "mu_bulk_final_err_pct": 100.0 * (mu_bulk / ref - 1.0),
        "p_balance_rel": p_ratio - 1.0,
        "mu_bulk_over_wall": mu_bulk / mu_wall if mu_wall else float("nan"),
        "rate_ratio": rate_ratio,
        "pressure_err_final": p_ratio - 1.0,
    }


def armpack(mu_bulk_by_I: dict[float, float], **kw) -> list[dict]:
    return [synth_row(I=I, mu_bulk=mu, **kw) for I, mu in mu_bulk_by_I.items()]


# -------------------------------------------------- pre-registration shape
def test_prereg_covers_the_case_inertial_range():
    I_all = [I for spec in R7.PREREG["arms"].values() for I in spec["I"]]
    assert I_all, "pre-registration has no points"
    assert min(I_all) >= 1e-3 and max(I_all) <= 0.3, I_all
    assert 0.3 in I_all, "the top of the case range is the contested point"


def test_prereg_arms_and_hypotheses_are_declared_up_front():
    arms = R7.PREREG["arms"]
    assert set(arms) == {"A_baseline", "B_layers", "C_servo"}
    # A reproduces the round-6 pre-registration, so it must keep its 4 I points
    assert len(arms["A_baseline"]["I"]) == 4
    # the explanation arms must not silently reuse the same geometry/protocol
    assert arms["B_layers"]["geom"] == "thick"
    assert arms["B_layers"]["geom"] != arms["A_baseline"]["geom"]
    assert arms["C_servo"]["servo"] and arms["A_baseline"]["servo"] is None
    hyps = R7.PREREG["hypotheses"]
    for key in ("H1_pressure_balance", "H2_wall_vs_bulk", "H3_no_localisation",
                "H4_layer_convergence", "H5_servo_bandwidth"):
        assert key in hyps and hyps[key], key
    assert "falsification" in R7.PREREG


def test_prereg_inherits_round6_and_records_its_own_constants():
    common = R7.PREREG["common"]
    assert common["max_recover_steps"] == R7.RECOVER_CAP
    assert common["normal_stress_pa"] == R7.P_APPLIED
    assert common["kn"] == R7.KN
    assert common["shear_strain"] == R7.SHEAR_STRAIN
    assert R7.RADIUS == 0.005
    assert "round 6" in R7.PREREG["inherits"]


def test_reference_law_is_the_dacruz_gdrmidi_law():
    # GDR MiDi / da Cruz 2005 asymptotics and the value at I0
    assert R7.mu_ref(1e-9) == pytest.approx(R7.MU_S, abs=1e-6)
    assert R7.mu_ref(1e6) == pytest.approx(R7.MU_2, abs=1e-3)
    assert R7.mu_ref(R7.I_0) == pytest.approx(0.5 * (R7.MU_S + R7.MU_2))
    grid = [10 ** (-3 + 0.1 * k) for k in range(30)]
    mus = [R7.mu_ref(I) for I in grid]
    assert all(b > a for a, b in zip(mus, mus[1:])), "mu(I) must increase with I"


# ------------------------------------------------------------- adjudicator
def test_h1_h2_h3_pass_on_a_self_consistent_point_set():
    rows = armpack({0.01: 0.3768, 0.03: 0.4096, 0.1: 0.48, 0.3: 0.6})
    v = R7.adjudicate(rows)
    assert v["H1_pressure_balance"]["pass"] is True
    assert v["H2_wall_vs_bulk"]["pass"] is True
    assert v["H3_no_localisation"]["pass"] is True


def test_h1_fails_on_a_single_bad_pressure_point():
    rows = armpack({0.01: 0.3768, 0.03: 0.4096, 0.1: 0.48, 0.3: 0.6})
    rows[3] = synth_row(I=0.3, mu_bulk=0.6, p_ratio=1.45)  # +45 % > 30 % band
    v = R7.adjudicate(rows)
    assert v["H1_pressure_balance"]["pass"] is False
    assert v["H1_pressure_balance"]["values"]["I=0.3"] == pytest.approx(45.0, abs=0.1)


def test_h2_and_h3_fail_independently():
    rows = armpack({0.1: 0.5}, mu_wall=0.35)         # bulk/wall = 1.43 -> H2 FAIL
    v = R7.adjudicate(rows)
    assert v["H2_wall_vs_bulk"]["pass"] is False
    assert v["H3_no_localisation"]["pass"] is True

    rows = armpack({0.1: 0.5}, rate_ratio=0.8)       # -20 % -> H3 FAIL
    v = R7.adjudicate(rows)
    assert v["H3_no_localisation"]["pass"] is False
    assert v["H2_wall_vs_bulk"]["pass"] is True


def test_h4_layer_convergence_uses_the_ten_percent_band_of_mu_ref():
    A = armpack({0.1: 0.48, 0.3: 0.60})
    B = [synth_row("B_layers", I=0.1, mu_bulk=0.49, geom="thick", n_particles=924),
         synth_row("B_layers", I=0.3, mu_bulk=0.58, geom="thick", n_particles=924)]
    v = R7.adjudicate(A + B)
    assert v["H4_layer_convergence"]["pass"] is True
    assert v["H4_layer_convergence"]["values"]["I=0.3"]["delta_mu_bulk"] == (
        pytest.approx(-0.02, abs=1e-9))

    B[1] = synth_row("B_layers", I=0.3, mu_bulk=0.75, geom="thick", n_particles=924)
    v = R7.adjudicate(A + B)
    assert v["H4_layer_convergence"]["pass"] is False


def test_h4_is_not_run_when_the_layer_arm_is_missing():
    v = R7.adjudicate(armpack({0.3: 0.6}))
    # empty dict -> pass must be False, never a silent True
    assert v["H4_layer_convergence"]["pass"] is False
    assert v["H4_layer_convergence"]["values"] == {}


def test_h5_servo_bandwidth_needs_both_speedup_and_pressure_control():
    A = [synth_row(I=0.3, mu_bulk=0.35, recover_steps=20000, p_ratio=1.0)]
    C = [synth_row("C_servo", I=0.3, mu_bulk=0.36, recover_steps=2000,
                   p_ratio=1.03)]
    v = R7.adjudicate(A + C)
    assert v["H5_servo_bandwidth"]["pass"] is True

    # 5x is the boundary: exactly 5x must already pass
    C = [synth_row("C_servo", I=0.3, mu_bulk=0.36, recover_steps=4000,
                   p_ratio=1.03)]
    assert R7.adjudicate(A + C)["H5_servo_bandwidth"]["pass"] is True

    # 4x speed-up is not enough even if the pressure is perfect
    C = [synth_row("C_servo", I=0.3, mu_bulk=0.36, recover_steps=5000,
                   p_ratio=1.0)]
    assert R7.adjudicate(A + C)["H5_servo_bandwidth"]["pass"] is False

    # so is a big speed-up with 8 % residual pressure error
    C = [synth_row("C_servo", I=0.3, mu_bulk=0.36, recover_steps=100,
                   p_ratio=1.08)]
    assert R7.adjudicate(A + C)["H5_servo_bandwidth"]["pass"] is False


def test_falsification_fires_only_inside_the_ten_percent_band():
    ref = R7.mu_ref(0.3)
    inside = armpack({0.3: 1.05 * ref})
    v = R7.adjudicate(inside)
    assert v["falsification"]["triggered"], "+5 % must overturn the WITHHELD verdict"
    assert "overturned" in v["falsification"]["reading"]

    outside = armpack({0.3: 0.65 * ref})
    v = R7.adjudicate(outside)
    assert v["falsification"]["triggered"] == []
    assert "survives" in v["falsification"]["reading"]


def test_band_verdict_uses_the_better_estimator_but_reports_both():
    ref = R7.mu_ref(0.1)
    rows = [synth_row(I=0.1, mu_bulk=0.5 * ref, mu_wall=1.03 * ref)]
    v = R7.adjudicate(rows)
    p = v["points"][0]
    assert p["in_10pct_band"] is True        # wall traction lands in the band
    assert p["mu_bulk_err_pct"] == pytest.approx(-50.0, abs=0.2)
    assert p["mu_wall_err_pct"] == pytest.approx(3.0, abs=0.2)
    # both numbers stay visible: the estimate that fails is never hidden
    assert p["best_abs_err_pct"] == pytest.approx(3.0, abs=0.2)


# ------------------------------------------------------------ deliverables
def test_tables_render_every_arm_and_verdict(tmp_path):
    rows = armpack({0.01: 0.37, 0.3: 0.55}) + [
        synth_row("B_layers", I=0.3, mu_bulk=0.56, geom="thick", n_particles=924),
        synth_row("C_servo", I=0.3, mu_bulk=0.55, recover_steps=100, p_ratio=1.01),
    ]
    v = R7.adjudicate(rows)
    text = R7.write_tables(rows, v, tmp_path / "tables.md")
    assert "A_baseline" in text and "B_layers" in text and "C_servo" in text
    assert "pre-registered verdicts" in text
    assert "PASS" in text or "FAIL" in text
    # programmatic discipline: no literal "nan" leaks into a published table
    assert " nan " not in text.lower().replace("nan%", "nan")
    on_disk = (tmp_path / "tables.md").read_text()
    assert on_disk == text


def test_result_json_roundtrip_is_serialisable(tmp_path):
    rows = armpack({0.3: 0.55})
    payload = {"case": "CASE-DEM-PLANE-SHEAR-MUIDI", "round": 7,
               "prereg": R7.PREREG, "rows": rows,
               "verdicts": R7.adjudicate(rows)}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    back = json.loads(path.read_text())
    assert back["verdicts"]["points"][0]["arm"] == "A_baseline"


# -------------------------------------------------------- device contract
@pytest.mark.slow
def test_run_point_returns_the_published_schema(monkeypatch):
    """Tiny specimen, tiny strain budget: the row must carry every field
    the tables and the adjudicator read (no silent ``nan`` from a renamed
    device key)."""
    monkeypatch.setattr(R7, "RADIUS", 0.01)          # ~37 particles
    monkeypatch.setattr(R7, "SETTLE_STEPS", 200)
    monkeypatch.setattr(R7, "MIN_SHEAR_STEPS", 100)
    monkeypatch.setattr(R7, "SHEAR_STRAIN", 0.02)
    monkeypatch.setattr(R7, "RECOVER_CAP", 200)
    row = R7.run_point(("A_baseline", "baseline", 0.3))
    for key in ("mu_wall", "mu_bulk", "p_wall", "p_bulk", "rate_ratio",
                "p_balance_rel", "mu_bulk_over_wall", "phi_bed_settled",
                "phi_bed_steady", "recover_steps", "shear_strain_achieved"):
        assert key in row, key
        assert isinstance(row[key], (int, float)), (key, type(row[key]))
    assert row["mu_ref"] == pytest.approx(R7.mu_ref(0.3))
    assert row["n_particles"] > 0
    assert row["mu_bulk"] >= 0.0 and row["p_bulk"] > 0.0
    assert row["wall_clock_s"] > 0.0
    # adjudication must survive a finite row from the real device
    R7.adjudicate([row])