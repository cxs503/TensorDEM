"""Round 19 (m21228) guards for the higher-N sweep's reference handling.

Two defects were found in ``high_n_sweep.py`` as shipped with round 18
(m21111) and are pinned here:

D-A  the file carried its own ``(mu_s, mu_2, I_0) = (0.36, 0.84, 0.30)``
     triple plus its own ``ref_mu``/``ref_phi``, although
     ``reference_law.py`` is documented as the single source of truth for
     this benchmark ("Do not copy this triple into another module").
D-B  that local ``ref_phi`` was ``phi_c + dphi/(I0/I+1)`` -- a *dilatancy
     sign inversion* (phi grows with I), i.e. the round-2 defect D2
     reintroduced.  It would have flipped the sign of every phi residual in
     an assembly; no round-18 row was ever assembled, so no archived number
     is affected.

D-C  the abscissa was ``res["inertial_number"]`` = the *platen-rate* I.  Rounds
     9/10 established that the physical abscissa is ``I`` from the *measured*
     velocity-profile shear rate: wall slip makes the bed shear slower than
     the platen drives it, so the platen-rate abscissa labels a material state
     at a higher I than it actually is and biases mu low at high I.

These tests need no DEM run (``summarise``/``assemble`` are pure).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parents[1] / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import high_n_sweep as H  # noqa: E402
import reference_law as REF  # noqa: E402


# --- D-A: single source of truth -------------------------------------------
def test_reference_law_is_imported_not_copied():
    """The sweep must reuse reference_law's objects, not re-declare them."""
    assert H.ref_mu is REF.ref_mu
    assert H.ref_phi is REF.ref_phi
    assert (H.MU_S, H.MU_2, H.I_0) == (REF.MU_S, REF.MU_2, REF.I_0)
    assert (H.PHI_MAX, H.PHI_MIN) == (REF.PHI_MAX, REF.PHI_MIN)


def test_no_duplicated_reference_triple_in_source():
    """The prose may quote the triple (it explains the defect); the *code* may not.

    Guard is AST-based, so a docstring/comment mentioning the numbers is fine
    while a re-introduced module-level copy is caught.
    """
    import ast
    tree = ast.parse((HERE / "high_n_sweep.py").read_text())
    disputed = {0.36, 0.84, 0.58, 0.64}   # 0.30 is also the restitution E=0.3
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            consts = {n.value for n in ast.walk(node.value)
                      if isinstance(n, ast.Constant)
                      and isinstance(n.value, (int, float))}
            assert not (consts & disputed), ast.unparse(node)
        if isinstance(node, (ast.FunctionDef,)):
            assert node.name not in ("ref_mu", "ref_phi"), node.name


# --- D-B: dilatancy sign ---------------------------------------------------
def test_reference_phi_decreases_with_inertial_number():
    Is = [1e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    phis = [H.ref_phi(I) for I in Is]
    assert all(b < a for a, b in zip(phis, phis[1:])), phis

    # limits and monotone bracket: jamming limit phi_max, asymptotic phi_min
    assert H.ref_phi(1e-9) == pytest.approx(REF.PHI_MAX, abs=1e-6)
    assert H.ref_phi(1e9) == pytest.approx(REF.PHI_MIN, abs=1e-6)
    for I, phi in zip(Is, phis):
        assert REF.PHI_MIN < phi < REF.PHI_MAX


def test_regression_round18_phi_formula_was_inverted():
    """Counter-example: the formula shipped in round 18 grows with I."""
    def old_ref_phi(I):                     # round-18 local definition
        return 0.58 + (0.64 - 0.58) / (0.30 / I + 1.0)

    assert old_ref_phi(0.01) < old_ref_phi(0.3)          # wrong sign
    assert H.ref_phi(0.01) > H.ref_phi(0.3)              # corrected
    # magnitude of the mis-sign at the two ends of the sweep
    assert old_ref_phi(0.01) - H.ref_phi(0.01) < -0.05   # -0.056
    assert old_ref_phi(0.3) - H.ref_phi(0.3) == pytest.approx(0.0, abs=1e-12)


def test_prereg_records_reference_provenance():
    ref = H.PREREG["reference"]
    assert ref["mu"]["mu_s"] == REF.MU_S
    assert "PROVENANCE-UNVERIFIED" in ref["phi"]["status"]
    assert "QUOTED, NOT ARCHIVED" in ref["mu"]["status"]
    assert "reference_law" in ref["origin"]


# --- D-C: measured-rate abscissa -------------------------------------------
def _fake_res(I_imposed, I_measured, mu=0.40, phi_bed=0.60):
    series = [{"mu": mu + 0.002 * (k % 3)} for k in range(40)]
    return {
        "series": series,
        "inertial_number": I_imposed,
        "inertial_number_measured_rate": I_measured,
        "inertial_number_measured_rate_final": I_measured,
        "phi_bed_steady": phi_bed,
        "phi_extent_final": phi_bed,
        "phi_container_final": phi_bed,
        "n_particles": 1276,
        "normal_stress_measured_pa": 62000.0,
        "normal_stress_applied_pa": 62500.0,
        "shear_stress_pa": 25000.0,
        "coordination_number": 4.1,
        "shear_rate_imposed": 5.0,
        "shear_rate_measured": 2.5,
        "shear_strain_achieved": 0.5,
        "dt": 1e-6,
        "shear_steps": 4000,
        "settle_steps": 3000,
        "servo_valid": True,
    }


def test_summarise_uses_measured_rate_as_abscissa():
    rec = H.summarise(_fake_res(0.30, 0.15), 0.30)
    assert rec["I_abscissa"] == pytest.approx(0.15)
    assert rec["I_imposed_rate"] == pytest.approx(0.30)
    assert rec["I_measured_rate"] == pytest.approx(0.15)
    assert rec["mu_ref"] == pytest.approx(REF.ref_mu(0.15))
    assert rec["phi_ref"] == pytest.approx(REF.ref_phi(0.15))
    assert rec["mu_ref_at_imposed_I"] == pytest.approx(REF.ref_mu(0.30))
    # the two abscissae must give materially different residuals
    assert abs(rec["err_mu_pct"] - rec["err_mu_pct_at_imposed_I"]) > 5.0


def test_summarise_falls_back_to_imposed_rate():
    res = _fake_res(0.30, None)
    res["inertial_number_measured_rate_final"] = None
    res["inertial_number_measured_rate"] = None
    rec = H.summarise(res, 0.30)
    assert rec["I_abscissa"] == pytest.approx(0.30)
    assert rec["mu_ref"] == pytest.approx(REF.ref_mu(0.30))


def test_summarise_noise_and_shear_rate_ratio():
    res = _fake_res(0.1, 0.1)
    rec = H.summarise(res, 0.1)
    assert rec["shear_rate_ratio"] == pytest.approx(0.5)
    vals = [s["mu"] for s in res["series"]]
    tail = vals[-max(1, int(0.25 * len(vals))):]        # final-25 % window
    mean = sum(tail) / len(tail)
    std = math.sqrt(sum((v - mean) ** 2 for v in tail) / len(tail))
    assert rec["mu_meas"] == pytest.approx(mean)
    assert rec["mu_std_rel_pct"] == pytest.approx(100.0 * std / mean, rel=1e-9)


# --- assemble(): pre-registered verdict logic ------------------------------
def _write_rows(d, rows):
    for r in rows:
        (d / f"point_{r['I_target']:g}.json").write_text(json.dumps(r))


def _row(I, err_mu, noise, err_phi=5.0, servo=True):
    return {"I_target": I, "err_mu_pct": err_mu, "mu_std_rel_pct": noise,
            "err_phi_pct": err_phi, "err_mu_pct_at_imposed_I": err_mu,
            "n_particles": 1276, "servo_valid": servo}


def test_assemble_pass_case(tmp_path):
    _write_rows(tmp_path, [_row(1e-2, 4.0, 3.0), _row(3e-2, -6.0, 5.0),
                           _row(1e-1, -12.0, 9.0)])
    v = H.assemble(tmp_path)
    assert v["P1_noise_floor"] is True
    assert v["P2_mu_convergence"] is True
    assert v["mu_within_10pct_full"] is False
    assert v["worst_mu_err_pct"] == pytest.approx(12.0)
    assert v["n_particles"] == [1276, 1276, 1276]


def test_assemble_fail_case_is_not_silent(tmp_path):
    _write_rows(tmp_path, [_row(1e-2, -18.0, 3.0), _row(3e-2, -22.0, 14.0)])
    v = H.assemble(tmp_path)
    assert v["P1_noise_floor"] is False        # noise 14 % > band
    assert v["P2_mu_convergence"] is False     # deficits outside band
    assert v["mu_within_10pct_full"] is False
    assert v["worst_mu_err_pct"] == pytest.approx(22.0)
    assert v["P3_honest_verdict"] is True


def test_tolerance_band_is_in_percent_not_fraction(tmp_path):
    """Counter-example: with the old ``TOL = 0.10`` a 9 % error failed the band."""
    _write_rows(tmp_path, [_row(1e-2, 9.0, 3.0), _row(3e-2, -9.0, 3.0)])
    v = H.assemble(tmp_path)
    assert v["tolerance_pct"] == pytest.approx(10.0)
    assert v["P2_mu_convergence"] is True       # 9 % < 10 % band
    assert H.TOL_PCT == pytest.approx(10.0)
    assert abs(9.0) < H.TOL_PCT                # 9 % is inside the band
    # and a genuinely out-of-band point still fails
    _write_rows(tmp_path, [_row(1e-2, 11.0, 3.0)])
    assert H.assemble(tmp_path)["P2_mu_convergence"] is False