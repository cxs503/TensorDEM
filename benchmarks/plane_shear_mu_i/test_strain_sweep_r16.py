"""Round 16 regression tests -- strain-converged mu(I)/phi(I) analysis.

Pins the R16 verdict so a future round cannot silently re-mix the archived
protocol (gamma ~ 0.4 at low I) with the strain-converged sweep (gamma >= 2),
and so the noise-floor argument (10 % band unresolvable at N = 294) stays
measured rather than asserted.

Run:  python -m pytest benchmarks/plane_shear_mu_i/test_strain_sweep_r16.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

RUNS = HERE / "runs" / "strain_sweep_r16"


def _rows() -> list[dict]:
    res = json.loads((RUNS / "result.json").read_text())
    return res["rows"]


def _res() -> dict:
    return json.loads((RUNS / "result.json").read_text())


def test_six_points_present_and_ordered():
    rows = _rows()
    assert [r["I_target"] for r in rows] == [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    assert all(r["n_particles"] == 294 for r in rows)
    assert all(r["strain_achieved"] >= 2.0 for r in rows)
    # the high-I point ran far past the archived protocol's strain budget
    assert rows[-1]["strain_achieved"] > 50.0


def test_mu_deficit_is_a_bias_not_a_transient():
    """At converged strain every point is below the reference on every
    window/abscissa combination; the low-I points are *further* below the
    reference than in the archived gamma~0.4 protocol (rounds 8/14: -0.1 %,
    +1.3 % at I = 0.01/0.03)."""
    for r in _rows():
        assert r["err_mu_20pct_at_target_pct"] < -10.0, r["I_target"]
        assert r["err_mu_25pct_at_target_pct"] < -10.0, r["I_target"]
        assert r["err_mu_25pct_at_bulk_pct"] < -10.0, r["I_target"]
    # the low-I transient: archived I=0.01 was -0.1 %; converged is ~ -18 %
    r01 = next(r for r in _rows() if r["I_target"] == 0.01)
    assert r01["err_mu_20pct_at_target_pct"] < -15.0


def test_phi_flat_and_below_band():
    """The bed is flat (0.55 -> 0.58) where the reference dilates (0.64 ->
    0.61): opposite trend, 5-14 % low; only I = 0.3 is inside the band."""
    rows = _rows()
    phis = [r["phi_container_25pct"] for r in rows]
    assert max(phis) - min(phis) < 0.05          # flat, not dilating
    assert all(r["err_phi_25pct_pct"] < 0.0 for r in rows)
    inside = [r for r in rows if abs(r["err_phi_25pct_pct"]) < 10.0]
    assert [r["I_target"] for r in inside] == [0.3]


def test_noise_floor_exceeds_band_at_five_of_six():
    """The 10 % ledger band is unresolvable at N = 294: the std of the final
    25 % window exceeds the band at five of six points (I = 0.01 is the
    exception, 5.3 %)."""
    rows = _rows()
    exceed = [r for r in rows if r["mu_std_rel_pct"] > 10.0]
    assert len(exceed) == 5
    r01 = next(r for r in rows if r["I_target"] == 0.01)
    assert r01["mu_std_rel_pct"] < 10.0
    # ...yet the bias at I = 0.01 is still -18 %: bias, not noise
    assert r01["err_mu_25pct_at_target_pct"] < -15.0


def test_abscissa_deviation_is_reported():
    """I_bulk deviates from I_target by -50 % to +33 %; the platen rate is
    not the bulk rate (wall slip / localisation)."""
    rows = _rows()
    shifts = [r["abscissa_shift_pct"] for r in rows]
    assert min(shifts) < -40.0
    assert max(shifts) > +20.0
    # re-judging at I_bulk changes no verdict
    for r in rows:
        assert r["err_mu_25pct_at_bulk_pct"] < -10.0


def test_steady_state_flags():
    rows = _rows()
    # phi settles only at the lowest I; mu never settles
    assert [r["steady_phi"] for r in rows] == [True, False, False, False,
                                               False, False]
    assert not any(r["steady_mu"] for r in rows)
    # at I = 0.3 the bed is still compacting at gamma ~ 99
    r03 = rows[-1]
    assert r03["drift_phi_pct"] > 0.0


def test_overall_verdict_is_fail():
    v = _res()["verdict"]
    assert v["overall"] == "FAIL"
    assert v["mu_curve_within_10pct"] is False
    assert v["phi_curve_within_10pct"] is False


def test_reference_provenance_carried_through():
    res = _res()
    assert "PROVENANCE-UNVERIFIED" in res["reference"]["phi"]["status"]
    assert res["reference"]["mu"]["constants"] == {
        "mu_s": 0.36, "mu_2": 0.84, "I_0": 0.30}


def test_artifacts_exist():
    for name in ("result.json", "tables.md", "fig_mu_i_strain_r16.png",
                 "fig_mu_vs_strain_r16.png"):
        assert (RUNS / name).is_file(), name
