"""The integration step must have exactly one spelling (m23261).

Context: `docs/CASE-DEM-SILO-BEVERLOO_m23261_ulp_dt_divergence.md`.  Two drivers of
the *same* frozen silo case wrote `dt` as `0.05 * T_c` (the `Material.dt` property)
and `T_c / 20` (a campaign runner).  Those two expressions differ by one
representable step, and the silo discharge is chaotic on that scale: the paired
probe in `benchmarks/m23261_ulp_dt/probe_divergence.py` shows the trajectories
decorrelating and the cumulative particle count differing after ~0.08 s.

These tests pin the single canonical spelling so the two can never drift apart
again, and pin the pure-arithmetic facts the audit relies on.
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tensordem.silo_flow.beverloo import discharge_status  # noqa: E402
from tensordem.silo_flow.kernel import (  # noqa: E402
    STEPS_PER_CONTACT, Material, contact_time, dt_for_steps,
)

sys.path.insert(0, str(ROOT / "benchmarks" / "m23261_ulp_dt"))
import analyse as A  # noqa: E402


def bits(x: float) -> str:
    return struct.pack(">d", float(x)).hex()


def ulp_gap(a: float, b: float) -> int:
    return (struct.unpack(">q", struct.pack(">d", float(b)))[0]
            - struct.unpack(">q", struct.pack(">d", float(a)))[0])


#: the campaign's frozen material (d = 3 mm, k_n = 5e3 N/m, ρ = 2500 kg/m³)
CAMPAIGN = Material(radius=0.0015, kn=5.0e3, mu=0.50, mu_rolling=0.05)


# --------------------------------------------------------------- canonical dt
def test_default_steps_per_contact_is_twenty():
    assert STEPS_PER_CONTACT == 20


def test_material_dt_is_the_canonical_division():
    t_c = contact_time(CAMPAIGN.mass_eff, CAMPAIGN.kn)
    assert CAMPAIGN.dt == dt_for_steps(CAMPAIGN.mass_eff, CAMPAIGN.kn) == t_c / 20.0
    # and *not* the rounded-reciprocal spelling that produced the archive split
    assert bits(CAMPAIGN.dt) == bits(t_c / 20.0)


def test_rounded_reciprocal_spelling_differs_by_one_ulp_for_the_campaign_material():
    """The historical defect, pinned: 0.05·T != T/20 by exactly 1 ULP here."""
    t_c = contact_time(CAMPAIGN.mass_eff, CAMPAIGN.kn)
    assert abs(ulp_gap(0.05 * t_c, t_c / 20.0)) == 1
    assert 0.05 * t_c != t_c / 20.0
    # the fix is a *spelling* change only: the two agree to 1e-15 relative
    assert math.isclose(0.05 * t_c, CAMPAIGN.dt, rel_tol=1e-15)


@pytest.mark.parametrize("steps", [10, 20, 40, 80])
def test_dt_for_steps_is_exact_division_of_the_contact_period(steps):
    t_c = contact_time(CAMPAIGN.mass_eff, CAMPAIGN.kn)
    assert dt_for_steps(CAMPAIGN.mass_eff, CAMPAIGN.kn, steps) == t_c / float(steps)


def test_dt_for_steps_rejects_nonpositive_step_counts():
    with pytest.raises(ValueError):
        dt_for_steps(CAMPAIGN.mass_eff, CAMPAIGN.kn, 0)
    with pytest.raises(ValueError):
        dt_for_steps(CAMPAIGN.mass_eff, CAMPAIGN.kn, -20)


def test_dt_for_steps_matches_the_stability_limit_used_by_siloflow():
    """20 steps per contact period must stay inside the 0.25·T_c stability bound."""
    dt = dt_for_steps(CAMPAIGN.mass_eff, CAMPAIGN.kn)
    t_c = contact_time(CAMPAIGN.mass_eff, CAMPAIGN.kn)
    assert dt <= 0.25 * t_c


# ------------------------------------------------------------------ pure math
def test_monodisperse_mass_eff_matches_the_kernel():
    for d in (0.003, 0.0045, 0.010):
        r = 0.5 * d
        ref = Material(radius=r).mass_eff
        assert A.monodisperse_mass_eff(d) == pytest.approx(ref, rel=1e-15)


def test_beverloo_helper_matches_the_kernel_correlation():
    from tensordem.silo_flow.beverloo import beverloo_prediction
    for D, d, rho in ((0.015, 0.003, 1323.7847222), (0.020, 0.004, 1500.0)):
        assert A.beverloo(d, D, rho) == pytest.approx(
            beverloo_prediction(D, d, rho), rel=1e-15)


def test_reference_rate_uses_literature_constants_only():
    """C = 0.58, k = 1.4 (Beverloo 1961) — never refitted in the analysis."""
    assert (A.C, A.K, A.G) == (0.58, 1.4, 9.81)


# ------------------------------------------------------------------- verdicts
def test_verdicts_report_not_run_when_every_record_is_missing():
    v = A.verdicts([], {"A_arch_bed": {"present": False},
                        "B_quot_bed": {"present": False}},
                   [])
    assert v["P1_archive_reproduction"]["A_arch_bed"]["verdict"] == "NOT_RUN"
    assert v["P1_archive_reproduction"]["B_quot_bed"]["verdict"] == "NOT_RUN"


def test_verdicts_detect_the_archive_ulp_split_without_any_dem():
    t_c = contact_time(CAMPAIGN.mass_eff, CAMPAIGN.kn)
    archive = [
        {"name": "m22885 fill36_seed1", "present": True, "git": "db2bdaf",
         "spelling": "0.05*T_c", "dt": 0.05 * t_c, "dt_bits": bits(0.05 * t_c),
         "W": 0.030749793997350797, "rho_bulk": 1323.7847222222226,
         "D": 0.015, "d": 0.003, "status": "FLOWING",
         "W_ref": A.beverloo(0.003, 0.015, 1323.7847222222226)},
        {"name": "m23114 dt_nom20_s1", "present": True, "git": "1fb2a46",
         "spelling": "T_c / 20", "dt": t_c / 20.0, "dt_bits": bits(t_c / 20.0),
         "W": 0.0033577361261474805, "rho_bulk": 1323.7847222222226,
         "D": 0.015, "d": 0.003, "status": "ARRESTED",
         "W_ref": A.beverloo(0.003, 0.015, 1323.7847222222226)},
    ]
    for r in archive:
        r["rel"] = r["W"] / r["W_ref"] - 1.0
    v = A.verdicts(archive, {}, [])["H1_archive_dt_fields_differ"]
    assert v["verdict"] == "CONFIRMED"
    assert abs(v["ulp_gap"]) == 1
    assert v["recomputed"]["0.05*Tc_equals_archive_m22885"] is True
    assert v["recomputed"]["Tc_over_20_equals_archive_m23114"] is True


def test_archive_reproduction_verdict_is_strict():
    """A reproduction must be bit-tight (1e-12); a 1 % miss must be REFUTED."""
    t_c = contact_time(CAMPAIGN.mass_eff, CAMPAIGN.kn)
    arch = [{"name": "m22885 fill36_seed1", "present": True, "git": "db2bdaf",
             "spelling": "0.05*T_c", "dt": 0.05 * t_c, "dt_bits": bits(0.05 * t_c),
             "W": 0.030749793997350797, "rho_bulk": 1323.78, "D": 0.015,
             "d": 0.003, "status": "FLOWING", "rel": 0.05,
             "W_ref": A.beverloo(0.003, 0.015, 1323.78)},
            {"name": "m23114 dt_nom20_s1", "present": False}]
    exact = A.verdicts(arch, {"A_arch_bed": {"present": True,
                                            "W": 0.030749793997350797,
                                            "rel": 0.05, "status": "FLOWING"},
                              "B_quot_bed": {"present": False}}, [])
    assert exact["P1_archive_reproduction"]["A_arch_bed"]["verdict"] == "CONFIRMED"
    off = A.verdicts(arch, {"A_arch_bed": {"present": True, "W": 0.031,
                                           "rel": 0.06, "status": "FLOWING"},
                            "B_quot_bed": {"present": False}}, [])
    assert off["P1_archive_reproduction"]["A_arch_bed"]["verdict"] == "REFUTED"


# -------------------------------------------------------------- arrest labels
def test_discharge_status_labels_a_jet_and_a_trickle():
    jet = discharge_status([28, 42, 41, 40, 38, 47, 41, 48],
                           [0.0198, 0.0297, 0.0290, 0.0283, 0.0269, 0.0332, 0.0290, 0.0339],
                           reference_rate=0.02915)
    assert jet["status"] == "FLOWING" and not jet["arrested"]
    trickle = discharge_status([2, 0, 0, 0, 0, 4, 0, 15],
                               [0.00141, 0, 0, 0, 0, 0.00283, 0, 0.01060],
                               reference_rate=0.02915)
    assert trickle["status"] == "ARRESTED" and trickle["arrested"]
    assert trickle["rate_over_reference"] < 0.20


def test_counting_noise_floor_is_the_documented_one():
    """The archived jet record carries N ≈ 330 steady particles ⇒ ≈ 5 % floor."""
    st = discharge_status([28, 42, 41, 40, 38, 47, 41, 48],
                          [0.0198, 0.0297, 0.0290, 0.0283, 0.0269, 0.0332, 0.0290, 0.0339],
                          reference_rate=0.02915)
    n = st["particles_in_steady_windows"]
    assert n == 47 + 41 + 48 + 38
    assert st["counting_noise_rel"] == pytest.approx(1.0 / math.sqrt(n), rel=1e-15)


def test_ulp_shift_helper_round_trips():
    x = 1.86767398828964e-05
    up = struct.unpack(">d", struct.pack(">q", struct.unpack(">q", struct.pack(">d", x))[0] + 1))[0]
    assert ulp_gap(x, up) == 1
    assert np.isfinite(up) and up > x