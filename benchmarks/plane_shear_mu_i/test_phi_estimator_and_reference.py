"""m17026 regression tests: solid-fraction estimator and the phi(I) reference sign.

Two defects are locked in here.

D1 ``tensordem.mu_i_rheology.simulate_servo_shear`` exposed the slab **porosity**
under the name ``solid_fraction`` (``1 - n*V_p/V_slab``).  Effect: every stored
``phi_device`` of rounds 1-3 is a porosity (0.36-0.39) tabulated as a solid
fraction.  Tests: the sample identity ``phi = n_slab*V_p/V_slab``, the
``phi + porosity = 1`` pairing, and the physical range of a dense bed (the old
sign gives ~0.37, which the range test rejects).

D2 the ``phi(I)`` reference law rose with ``I`` (0.58 -> 0.64), opposite to
dilatancy.  Tests: limits, strict monotonicity, and an explicit assertion that
the legacy rising form is no longer what the benchmark uses.

Run: ``pytest benchmarks/plane_shear_mu_i/test_phi_estimator_and_reference.py -q``
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO / "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import reference_law as rl  # noqa: E402
from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear  # noqa: E402

RUNS = HERE / "runs"


# --------------------------------------------------------------- D2: phi(I)
def test_ref_phi_limits():
    assert rl.ref_phi(1e-9) == pytest.approx(rl.PHI_MAX, rel=1e-6)   # I -> 0
    assert rl.ref_phi(1e9) == pytest.approx(rl.PHI_MIN, rel=1e-6)    # I -> inf
    assert rl.PHI_MAX > rl.PHI_MIN


def test_ref_phi_decreases_with_I():
    """Dilatancy: phi must fall as I grows (da Cruz 2005 / GDR MiDi 2004)."""
    Is = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    phis = [rl.ref_phi(I) for I in Is]
    assert all(b < a for a, b in zip(phis, phis[1:])), phis
    # and it must NOT be the legacy rising form
    legacy = rl.PHI_MIN + (rl.PHI_MAX - rl.PHI_MIN) / (rl.I_0 / 1e-3 + 1.0)
    assert abs(legacy - rl.ref_phi(1e-3)) > 0.01


def test_ref_phi_band_is_not_discriminative():
    """The reference band spans only ~5 % of its own value over the whole I
    range, so the 10 % ledger gate cannot discriminate phi - the reason the
    case reports phi as WITHHELD instead of PASS."""
    span = abs(rl.ref_phi(1e-3) - rl.ref_phi(0.3)) / rl.ref_phi(0.3)
    assert span < 0.10


def test_ref_mu_unchanged():
    assert rl.ref_mu(1e-9) == pytest.approx(rl.MU_S, rel=1e-6)
    assert rl.ref_mu(1e9) == pytest.approx(rl.MU_2, rel=1e-6)
    assert rl.ref_mu(0.1) == pytest.approx(0.48, abs=1e-6)


def test_reference_block_declares_provenance():
    ref = rl.reference()
    assert "PROVENANCE-UNVERIFIED" in ref["phi"]["status"]
    assert "monotonicity" in ref["phi"]["status"].lower()


# --------------------------------------------- D1: slab solid fraction
@pytest.fixture(scope="module")
def short_run():
    case = ServoShearCase(
        radius=0.005, density=2500.0, mu=0.5, mu_rolling=0.05, restitution=0.3,
        kn=6.25e4, box_width=0.12, box_depth=0.05, fill_height=0.06,
        normal_stress=6.25e4, inertial_number=0.1, seed=5279,
        settle_steps=400, min_shear_steps=120, sample_stride=20,
        shear_strain=0.02, max_recover_steps=0)
    return simulate_servo_shear(case, return_series=True)


def test_slab_solid_fraction_identity(short_run):
    series = short_run["series"]
    assert series, "no samples"
    for s in series:
        vp = 4.0 / 3.0 * math.pi * short_run["radius"] ** 3
        expected = s["slab_n_particles"] * vp / s["slab_volume"]
        assert s["solid_fraction"] == pytest.approx(expected, rel=1e-9)
        assert s["solid_fraction"] + s["porosity"] == pytest.approx(1.0, rel=1e-12)


def test_solid_fraction_is_not_the_porosity(short_run):
    """A confined granular bed at P = 62.5 kPa is dense: the solid fraction must
    be well above 0.45.  The pre-fix code returned the porosity (~0.37), which
    this range check rejects."""
    for s in short_run["series"]:
        assert 0.45 < s["solid_fraction"] < 0.80, s["solid_fraction"]
        assert 0.20 < s["porosity"] < 0.55, s["porosity"]
    assert short_run["solid_fraction"] > 0.45
    assert short_run["porosity"] == pytest.approx(1.0 - short_run["solid_fraction"])


def test_straddle_fraction_is_reported(short_run):
    """The bulk estimator drops contacts with one endpoint in the slab; the
    dropped fraction is what biases the slab pressure low, so it must be
    exposed for audit."""
    val = short_run["bulk_straddle_fraction"]
    assert val is not None
    assert 0.0 <= val <= 1.0


# --------------------------------------------- stored-artifact consistency
def test_legacy_artifacts_pin_the_definition_defect():
    """The stored round-1/2 ``phi_device`` values are porosities; after the
    documented inversion they must land in the dense-bed band."""
    path = RUNS / "sweep.json"
    if not path.exists():
        pytest.skip("runs/sweep.json not present")
    rows = json.loads(path.read_text())["rows"]
    for r in rows:
        phi_solid = 1.0 - float(r["phi_device"])
        assert 0.5 < phi_solid < 0.8, (r["I_measured"], phi_solid)


def test_phi_audit_artifact_consistent():
    path = RUNS / "phi_reference_audit.json"
    if not path.exists():
        pytest.skip("run phi_reference_audit.py first")
    a = json.loads(path.read_text())
    assert "WITHHELD" in a["phi_acceptance_verdict"]
    for arm, rows in a["arms"].items():
        for r in rows:
            assert r["phi_solid"] == pytest.approx(1.0 - r["phi_legacy_porosity"],
                                                   rel=1e-12)
            assert r["phi_err_pct"] == pytest.approx(
                100.0 * (r["phi_solid"] / r["phi_ref"] - 1.0), rel=1e-9)
            # the mislabeled reading is always worse than the corrected one
            assert abs(r["phi_err_pct_legacy_reading"]) > abs(r["phi_err_pct"])
            assert abs(r["phi_err_pct_legacy_reading"]) > 0.10