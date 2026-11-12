"""Tests for R3 of CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT (kappa-similarity route).

These tests protect three different things and are written to fail loudly if any
of them is overstated:

1. **Design integrity** -- the iso-kappa groups really are iso-kappa, and the
   grid really spans the advertised decades.  If someone edits the grid and
   breaks a pairing, the collapse test becomes meaningless and must not report
   PASS.
2. **The scaling symmetry is an identity, not a measurement** -- the kernel is
   *exactly* invariant under ``(k_n, gamma_dot) -> (lambda^2 k_n, lambda gamma_dot)``
   because the time step is defined as a fraction of the contact time
   ``t_c = sqrt(m/k_n)``.  The tests assert that invariance to ~1e-9, and they
   assert that the published analysis *says so* (``amendments`` entry), so the
   result cannot later be re-sold as empirical confirmation of Campbell (2002).
3. **The analysis arithmetic** -- the two reduction routes must agree
   algebraically: the rate exponent measured along a constant-``k_n`` ladder must
   equal ``2 (1 - beta)`` computed from the same points' ``kappa`` positions.
   A coding error in either route breaks this to first order.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "case_dem_shear_campbell_elastic_limit"))

from tensordem.elastic_limit_shear import (  # noqa: E402
    ElasticLimitShearCase, simulate_plane_shear)

import kappa_collapse as kc  # noqa: E402

RUNS = kc.RUNS
ANALYSIS = RUNS / "analysis.json"
PREREG = RUNS / "prereg.json"

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# ---------------------------------------------------------------------------
# 1. design integrity
# ---------------------------------------------------------------------------
def test_iso_kappa_groups_are_exact_and_span_three_decades():
    rows = kc.design_table()
    groups = {}
    for r in rows:
        if r["group"]:
            groups.setdefault(r["group"], []).append(r)
    assert len(groups) >= 4, f"expected >=4 iso-kappa groups, got {len(groups)}"
    assert max(len(v) for v in groups.values()) >= 3, "largest group must have >=3 members"
    for gk, members in groups.items():
        kappas = np.array([m["kappa_predicted"] for m in members])
        rel_spread = float(np.ptp(kappas) / kappas.mean())
        assert rel_spread < 1e-9, f"group {gk} is not iso-kappa (rel spread {rel_spread:.2e})"
        # distinct apparatus settings, otherwise the pairing tests nothing
        assert len({(m["kn"], m["gdot"]) for m in members}) == len(members)
        assert len({m["kn"] for m in members}) == len(members), \
            "iso-kappa partners must differ in k_n (same k_n means same run)"
    span = math.log10(max(r["kappa_predicted"] for r in rows) /
                      min(r["kappa_predicted"] for r in rows))
    assert span >= 3.0, f"kappa span {span:.2f} decades is too narrow for an exponent fit"


def test_declared_not_run_rungs_are_now_in_the_design():
    """Rounds 1-2 left k_n = 1e5 and 1e6 as not_run; this round must cover them."""
    kns = {r["kn"] for r in kc.design_table()}
    assert 1.0e5 in kns and 1.0e6 in kns


# ---------------------------------------------------------------------------
# 2. the symmetry is exact (short, cheap runs -- not a physics measurement)
# ---------------------------------------------------------------------------
SHORT = dict(n_particles=200, solid_fraction=0.80, strain_total=0.5,
             strain_average_from=0.25, dt_factor=0.004, seed=20261112)


def _tau_star(kn: float, gdot: float) -> float:
    case = ElasticLimitShearCase(kn=kn, shear_rate=gdot, **SHORT)
    return float(simulate_plane_shear(case)["tau_star"])


@pytest.mark.parametrize("base", [(1.0e3, 0.5), (1.0e4, 1.0)])
def test_kernel_is_exactly_scale_invariant(base):
    """(k_n, gdot) -> (4 k_n, 2 gdot) keeps kappa and must keep tau* bit-for-bit."""
    kn, gd = base
    a = _tau_star(kn, gd)
    b = _tau_star(4.0 * kn, 2.0 * gd)
    rel = abs(a - b) / abs(a)
    assert rel < 1e-9, f"scaling symmetry broken: {a!r} vs {b!r} (rel {rel:.2e})"


def test_timestep_scales_with_contact_time():
    """The symmetry holds *because* dt is a fixed fraction of t_c; assert it."""
    c1 = ElasticLimitShearCase(kn=1.0e3, shear_rate=0.5, **SHORT)
    c4 = ElasticLimitShearCase(kn=4.0e3, shear_rate=1.0, **SHORT)
    assert c4.dt / c1.dt == pytest.approx(0.5, rel=1e-12)
    assert c4.kappa == pytest.approx(c1.kappa, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. published analysis: binds the claim, forbids re-selling the identity
# ---------------------------------------------------------------------------
def _load(p: Path) -> dict:
    assert p.exists(), f"missing artefact {p} (run the kappa_collapse driver)"
    return json.loads(p.read_text())


def test_collapse_spread_is_identically_zero_and_declared_as_such():
    a = _load(ANALYSIS)
    assert a["collapse_groups"], "no multi-point kappa group survived the gates"
    for g in a["collapse_groups"]:
        assert g["spread"] < 1e-12, (
            f"collapse at kappa={g['kappa']:.4g} has spread {g['spread']:.3e}; "
            "the kernel is supposed to be exactly scale invariant")
    # the honest classification must travel with the number
    amend = a.get("amendments", {})
    assert "P1" in "".join(amend.keys()) or any("identity" in str(v).lower()
                                                for v in amend.values()), \
        "analysis must declare that the exact collapse is a theoretical identity"


def test_global_fit_is_below_the_elastic_limit_and_approaches_it():
    a = _load(ANALYSIS)
    fit = a["global_fit"]
    assert fit["status"] == "ok"
    beta = fit["beta"]
    assert 0.5 < beta < 1.0, f"beta={beta} outside the physically possible band (0,1)"
    assert fit["r2"] > 0.99, "a single power law must describe the kappa ladder"
    # the local exponent must move toward 1 as kappa grows
    ap = a["approach"]
    assert ap["beta_low_half"] < ap["beta_high_half"], \
        "local exponent must increase with kappa (approach to the elastic limit)"
    assert ap["beta_high_half"] <= 1.0 + 1e-9


def test_rate_exponent_identity_matches_within_family():
    """n from a constant-k_n ladder == 2(1-beta) from the same points' kappa."""
    a = _load(ANALYSIS)
    fams = a["family_rate_exponents"]
    assert fams, "no family had >=3 gated points"
    for kn, fam in fams.items():
        n_meas = fam["n_from_rate_ladder"]
        beta, _ = kc._loglog_slope(
            [float(kn) / (1.0 * g * g) for g in fam["gammas"]],  # kappa up to a constant
            fam["tau_stars"])
        assert n_meas == pytest.approx(2.0 * (1.0 - beta), abs=1e-9), \
            f"family k_n={kn}: n={n_meas} but 2(1-beta)={2.0 * (1.0 - beta)}"


def test_prereg_is_bound_to_the_results():
    pre = _load(PREREG)
    a = _load(ANALYSIS)
    anchor = kc.ANCHOR_FILE
    assert pre["anchor_sha256"] == hashlib.sha256(anchor.read_bytes()).hexdigest()
    assert a["prereg_sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest(), \
        "analysis.json was produced from a different pre-registration than the one on disk"
    for key in ("P1_kappa_collapse", "P2_elastic_exponent", "P3_rate_exponent",
                "P4_monotone_approach", "P5_noise_floor"):
        assert key in pre["predictions"]


def test_rejected_points_are_reported_not_silently_dropped():
    a = _load(ANALYSIS)
    gated = {p["tag"] for p in a["points"] if p["gates"]["all"]}
    rejected = {r["tag"] for r in a["rejected"]}
    assert gated.isdisjoint(rejected)
    for r in a["rejected"]:
        assert r["gates"]["all"] is False
        assert any(v is False for k, v in r["gates"].items() if k != "all")
    assert a["n_gated"] + len(a["rejected"]) == a["n_points"]


# ---------------------------------------------------------------------------
# 4. analysis arithmetic on synthetic data
# ---------------------------------------------------------------------------
def test_slope_recovers_known_exponent():
    xs = np.logspace(1, 4, 12)
    for target in (0.25, 0.5, 1.0, 1.7):
        ys = 3.0 * xs ** target
        beta, r2 = kc._loglog_slope(xs, ys)
        assert beta == pytest.approx(target, abs=1e-12)
        assert r2 == pytest.approx(1.0, abs=1e-12)


def test_constant_y_reports_no_r2():
    """A constant series has no variance: r2 must be 'not defined', not a quotient of noise."""
    beta, r2 = kc._loglog_slope(np.logspace(1, 4, 12), np.full(12, 3.0))
    assert beta == pytest.approx(0.0, abs=1e-9)
    assert math.isnan(r2)


def test_slope_refuses_degenerate_input():
    assert kc._loglog_slope([1.0], [2.0]) == (None, None)
    assert kc._loglog_slope([5.0, 5.0], [2.0, 3.0]) == (None, None)
    # non-positive y is dropped, leaving one usable point -> no slope
    assert kc._loglog_slope([1.0, 10.0], [0.0, 5.0]) == (None, None)


def test_classification_bands():
    assert kc._classify(0.05, 0.15, 0.30) == "PASS"
    assert kc._classify(-0.20, 0.15, 0.30) == "PARTIAL"
    assert kc._classify(0.45, 0.15, 0.30) == "NOT_SUPPORTED"

# ---------------------------------------------------------------------------
# 5. honesty guards: precision must not be claimed beyond the noise floor
# ---------------------------------------------------------------------------
def test_uncertainties_are_propagated_from_the_noise_floor():
    a = _load(ANALYSIS)
    sig = a["noise_floor"]["relative_std"]
    assert 0.0 < sig < 0.3, "a seed noise floor must have been measured"
    assert all(p.get("sigma_beta", 0) > 0 for p in a["local_slopes"]), \
        "every local slope must carry a sigma derived from that noise floor"
    assert a["global_fit"]["sigma_beta_est"] > 0
    ap = a["approach"]
    assert ap["sigma_beta_high_half"] > 0 and ap["sigma_beta_low_half"] > 0


def test_three_percent_claim_is_refused_at_the_stiff_end():
    """The top-decade point estimate is close to beta=1, but not to 3% given the noise.

    This test exists so the '<3%' column of the error table cannot silently become
    a claim: the flag must stay False while 2 sigma exceeds 3%.
    """
    a = _load(ANALYSIS)
    ap = a["approach"]
    two_sigma = 2.0 * ap["sigma_beta_high_half"]
    close = abs(ap["beta_high_half"] - 1.0) <= 0.03
    if close and two_sigma > 0.03:
        assert ap["elastic_limit_demonstrated_3pct"] is False
    doc = (REPO / "docs" / "CASE_DEM_SHEAR_CAMPBELL_ELASTIC_LIMIT_R3_KAPPA_SIMILARITY.md")
    if doc.exists():
        text = doc.read_text()
        assert "not demonstrated" in text, "the doc must say the 3% target is not demonstrated"
        assert str(round(ap["elastic_limit_demonstrated_3pct"])) in text


def test_global_exponent_deficit_is_resolved_not_noise():
    a = _load(ANALYSIS)
    fit = a["global_fit"]
    n_sigma = abs(fit["beta"] - 1.0) / float(fit["sigma_beta_est"])
    assert n_sigma > 3.0, ("the deficit of beta below the elastic limit must be a resolved "
                           f"effect, got {n_sigma:.1f} sigma")
