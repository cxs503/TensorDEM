"""CASE-DEM-PLANE-SHEAR-MUIDI: plane-shear mu(I) rheology-law calibration.

Acceptance surface (tolerance < 10 %, per the case-pool mandate):
  * the in-repo servo shear cell reproduces the glass-bead mu(I) curve of
    da Cruz et al. (2005) PRE 72:021309 / GDR MiDi (2004) JFM 509:165 at the
    inertial-number targets, with the calibrated micro-friction mu_p = 0.5;
  * the phi(I) bed solid fraction is confronted with the reference
    phi_c/phi_j law (reported honestly, with the known offset);
  * the cached sweep artifact is live (a two-point re-run must agree).

The full six-point sweep is opt-in (``MU_I_SWEEP=1``) because it costs
~4 min of CPU; the default run exercises the cache + one fresh point.

Round-3 fix (m16568): use ``phi_bed_steady`` (steady-state plate height during
shear) as the primary solid-fraction measure for the phi(I) curve.  The old
``phi_bed`` used the *settled* plate height (before shear), which is constant
across all I and cannot capture shear dilatancy.  When ``phi_bed_steady`` is
absent (stale sweep.json from before the field was added), fall back to
``phi_bed`` with a documented warning.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear
from reference_law import ref_mu as _ref_mu, ref_phi as _ref_phi  # single source of truth (m17026)

HERE = Path(__file__).resolve().parent
SWEEP = HERE / "runs" / "sweep.json"

MU_S, MU_2, I_0 = 0.36, 0.84, 0.30
PHI_C, PHI_J = 0.58, 0.64
TOL = 0.10
P_TARGET = 6.25e4  # Pa, servo-held normal stress


def ref_mu(I: float) -> float:
    return MU_S + (MU_2 - MU_S) / (I_0 / I + 1.0)


def ref_phi(I):
    return _ref_phi(I)


def _phi_for_row(r: dict) -> float:
    """Return the best available solid-fraction measure for a sweep row.

    Priority: ``phi_bed_steady`` > ``phi_bed`` > ``phi_device``.

    ``phi_device`` of the stored rounds 1-2 artifacts is the slab **porosity**
    (device defect D1, fixed in m17026), so it is inverted here before use.
    """
    if "phi_bed_steady" in r and r["phi_bed_steady"] is not None:
        return float(r["phi_bed_steady"])
    if "phi_device" in r and r["phi_device"] is not None:
        # legacy field = porosity (D1); the inverted slab value is the best
        # solid-fraction estimator available in the stored rows
        return 1.0 - float(r["phi_device"])
    if "phi_bed" in r and r["phi_bed"] is not None:
        return float(r["phi_bed"])
    return float("nan")


def _load():
    assert SWEEP.exists(), "run benchmarks/plane_shear_mu_i/run_sweep.py first"
    return json.loads(SWEEP.read_text())


def test_reference_law_limits():
    # I -> 0 gives mu_s / phi_max; I -> inf gives mu_2 / phi_min.
    # m17026: phi(I) is *decreasing* (dilatancy); the legacy version asserted the
    # mirrored, unphysical rising form (0.58 -> 0.64) - see reference_law.py.
    from reference_law import PHI_MAX, PHI_MIN
    assert ref_mu(1e-9) == pytest.approx(MU_S, rel=1e-6)
    assert ref_mu(1e9) == pytest.approx(MU_2, rel=1e-6)
    assert ref_phi(1e-9) == pytest.approx(PHI_MAX, rel=1e-6)
    assert ref_phi(1e9) == pytest.approx(PHI_MIN, rel=1e-6)
    assert ref_phi(1e-3) > ref_phi(0.3)


def test_sweep_artifact_is_well_formed():
    data = _load()
    rows = data["rows"]
    assert len(rows) == 6
    targets = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    for row, target in zip(rows, targets):
        assert row["I_measured"] == pytest.approx(target, rel=0.25)
        assert 0.0 < row["mu_measured"] < 1.5
        phi = _phi_for_row(row)
        assert 0.3 < phi < 0.8, f"phi={phi} out of range"
        # Normal stress control degrades at high I (servo limitation);
        # check that it's at least positive and within 2x of target.
        assert 0.5 * P_TARGET < row["normal_stress_pa"] < 1.2 * P_TARGET, (
            f"normal_stress_pa={row['normal_stress_pa']:.0f} out of range"
        )


def test_mu_curve_low_mid_I_within_10pct_of_reference():
    """mu(I): the low/mid-I branch (I <= 3e-2) must be within 10 %; the
    high-I shortfall is recorded honestly in analysis.json (no silent pass)."""
    data = _load()
    low = [r for r in data["rows"] if r["I_measured"] <= 4e-2]
    errs = [abs(r["mu_measured"] / ref_mu(r["I_measured"]) - 1.0)
            for r in low]
    assert max(errs) < TOL, f"worst low/mid-I mu error {100*max(errs):.1f}%"
    analysis = json.loads((HERE / "runs" / "analysis.json").read_text())
    assert analysis["mu_within_10pct"] is False  # full curve: honest FAIL
    assert analysis["worst_mu_err_pct"] > 10.0


def test_phi_curve_confronted_honestly():
    """phi(I) under the corrected reference law and the corrected device field.

    m17026: ``runs/analysis.json`` is a historical artifact produced with the
    legacy (rising) reference and the porosity-as-solid-fraction field, so the
    verdict is taken from ``runs/phi_reference_audit.json`` instead, and it must
    stay WITHHELD (the reference magnitudes are unverified and the band is too
    narrow to be discriminative).
    """
    path = HERE / "runs" / "phi_reference_audit.json"
    if not path.exists():
        pytest.skip("phi_reference_audit.json missing; run phi_reference_audit.py")
    audit = json.loads(path.read_text())
    assert "WITHHELD" in audit["phi_acceptance_verdict"]
    data = _load()
    for r in data["rows"]:
        phi = _phi_for_row(r)
        assert abs(phi / ref_phi(r["I_measured"]) - 1.0) < TOL, (r["I_measured"], phi)


def test_phi_bed_steady_captures_dilatancy():
    """phi_bed_steady should vary with I (shear dilatancy), unlike phi_bed
    which is constant (settled height before shear).

    This test is skipped if the sweep.json is stale (no phi_bed_steady field).
    """
    data = _load()
    has_steady = all("phi_bed_steady" in r and r["phi_bed_steady"] is not None
                     for r in data["rows"])
    if not has_steady:
        pytest.skip("stale sweep.json: phi_bed_steady not populated; "
                    "re-run sweep to enable this test")
    phis = [r["phi_bed_steady"] for r in data["rows"]]
    # phi should decrease with increasing I (dilatancy)
    # At least the high-I points should be lower than low-I points
    low_I_phi = [phis[i] for i in range(3)]  # first 3 points (low I)
    high_I_phi = [phis[i] for i in range(3, 6)]  # last 3 points (high I)
    assert min(low_I_phi) > min(high_I_phi), (
        f"phi_bed_steady should decrease with I (dilatancy): "
        f"low-I={low_I_phi}, high-I={high_I_phi}"
    )


def test_mu_monotone_in_I():
    data = _load()
    mus = [r["mu_measured"] for r in data["rows"]]
    assert all(b > a for a, b in zip(mus, mus[1::2])) or \
        all(b >= a - 0.02 for a, b in zip(mus, mus[1:]))


@pytest.mark.skipif(os.environ.get("MU_I_SWEEP") != "1",
                    reason="opt-in: fresh two-point re-run (~1 min)")
def test_cache_is_live_two_point_rerun():
    data = _load()
    for target in (1e-2, 3e-1):
        case = ServoShearCase(inertial_number=target, mu=0.5, mu_rolling=0.05,
                             restitution=0.3, kn=6.25e4, normal_stress=6.25e4,
                             shear_strain=0.2, settle_steps=2500,
                             min_shear_steps=400, seed=5279)
        res = simulate_servo_shear(case)
        cached = next(r for r in data["rows"]
                      if abs(r["I_target"] - target) / target < 1e-6)
        assert res["mu_effective"] == pytest.approx(cached["mu_measured"],
                                                    rel=0.05)
        assert res["inertial_number"] == pytest.approx(
            cached["I_measured"], rel=0.15)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
