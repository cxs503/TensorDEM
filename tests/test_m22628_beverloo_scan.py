"""CASE-DEM-SILO-BEVERLOO (m22628): the orifice-scan inversion must be falsifiable.

The point of the scan is to *invert* ``W = C rho_b sqrt(g) (D - k d)**p`` from
DEM data instead of assuming ``k = 1.4`` and ``p = 5/2``.  A fit that cannot
recover a known ``(C, k, p)`` from noiseless synthetic data would make every
verdict downstream meaningless, so these tests pin the *estimator*, and then
pin the *verdict arithmetic* (a fixed-k assumption hides a k error as a
D-dependent amplitude error -- the defect the campaign suffered from).

Nothing here runs a DEM simulation: the acceptance run is the cluster one.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "m22628_beverloo_scan"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

def _load(name: str):
    """Load the scan's module under a mission-scoped name (see m22657's twin).

    Benchmark directories each ship an ``analyse.py``; a bare ``import analyse``
    lets the first-collected test file own the ``sys.modules`` slot and swap
    estimators between missions.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"m22628_{name}", BENCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("analyse")

sys.path.insert(0, str(ROOT / "src"))

D_PART = 0.003
D_ORIFICE = np.array([4.0, 4.5, 5.0, 5.5, 6.0]) * D_PART
RHO = np.full(D_ORIFICE.shape, 1400.0)


def synthetic(c: float, k: float, p: float, *, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    eff = D_ORIFICE - k * D_PART
    w = c * RHO * math.sqrt(A.G) * eff ** p
    if noise:
        rng = np.random.default_rng(seed)
        w = w * (1.0 + noise * rng.standard_normal(w.shape))
    return w


# ---------------------------------------------------------------------------
# estimator recovery
# ---------------------------------------------------------------------------
def test_three_parameter_fit_recovers_a_known_law():
    """Noiseless data from (C=0.58, k=1.4, p=2.5) must be inverted exactly."""
    w = synthetic(0.58, 1.4, 2.5)
    f = A.fit(D_ORIFICE, RHO, D_PART, np.log(w), k_free=True, p_free=True)
    assert f["k"] == pytest.approx(1.4, abs=1e-2)
    assert f["p"] == pytest.approx(2.5, abs=1e-2)
    assert f["C"] == pytest.approx(0.58, rel=1e-2)
    assert f["max_abs_rel_err"] < 1e-6


def test_fit_detects_a_wrong_exponent():
    """If the truth were p = 2.0, the 3-parameter fit must say so (and fail the
    5/2 statement), not absorb the discrepancy into C."""
    w = synthetic(0.58, 1.4, 2.0)
    f = A.fit(D_ORIFICE, RHO, D_PART, np.log(w), k_free=True, p_free=True)
    assert f["p"] == pytest.approx(2.0, abs=1e-2)
    assert abs(f["p"] - A.P_LAW) > A.P_TOL


def test_fixed_k_contaminates_the_amplitude():
    """The campaign's blind spot: with k pinned at 1.4 but the truth at k = 0,
    the residuals are D-dependent and no single amplitude fixes them."""
    w = synthetic(0.58, 0.0, 2.5)
    y = np.log(w)
    pinned = A.fit(D_ORIFICE, RHO, D_PART, y, k_free=False, p_free=False)
    freed = A.fit(D_ORIFICE, RHO, D_PART, y, k_free=True, p_free=False)
    assert pinned["max_abs_rel_err"] > 0.15            # a single C cannot fix it
    assert freed["max_abs_rel_err"] < 1e-6             # a free k can
    assert freed["k"] == pytest.approx(0.0, abs=1e-3)
    # and the pinned fit reports a *wrong* C: forcing k above its true value
    # shrinks (D - k d), so the compensating amplitude is inflated
    assert pinned["C"] > A.C_REF
    assert freed["C"] == pytest.approx(0.58, rel=1e-6)


def test_se_of_the_exponent_is_reported_and_scales_with_noise():
    clean = A.fit(D_ORIFICE, RHO, D_PART, np.log(synthetic(0.58, 1.4, 2.5)),
                  k_free=True, p_free=True)
    noisy = A.fit(D_ORIFICE, RHO, D_PART, np.log(synthetic(0.58, 1.4, 2.5, noise=0.03, seed=7)),
                  k_free=True, p_free=True)
    assert math.isfinite(clean["se_p"]) and math.isfinite(noisy["se_p"])
    assert noisy["se_p"] > clean["se_p"] / 10.0


# ---------------------------------------------------------------------------
# verdict arithmetic
# ---------------------------------------------------------------------------
def _fake_run(case_id, ratio, w, rho=1400.0, wsd=None, fill_height=None, n_bed=3000):
    D_o = ratio * D_PART
    return {
        "case_id": case_id,
        "case": {
            "orifice_diameter": D_o,
            "particle_diameter": D_PART,
            "fill_height": fill_height if fill_height is not None else 3.0 * D_o,
            "seed": 1,
        },
        "domain": {"ratios": {"D_over_d": ratio, "silo_D_over_D": 4.0,
                              "bed_height_over_D": 3.0}, "in_domain": True},
        "result": {"rho_bulk": rho, "flow_rate_kg_s": w,
                   "flow_rate_std_kg_s": wsd if wsd is not None else 0.01 * w,
                   "n_bed": n_bed},
    }


def test_primary_verdict_uses_the_literature_coefficients():
    """The ledger tolerance is against the *law*, so the fitted C must not be
    used to declare a pass."""
    ids = ["D40", "D45", "D50", "D55", "D60"]
    ratios = [4.0, 4.5, 5.0, 5.5, 6.0]
    w = synthetic(0.58, 1.4, 2.5)
    runs = {cid: _fake_run(cid, r, float(wi)) for cid, r, wi in zip(ids, ratios, w)}
    s = A.analyse(runs)
    prim = s["primary"]
    assert prim["available"] and prim["case_id"] == "D50"
    # measured == literature prediction here, so the error is zero
    assert abs(prim["rel_err_literature"]) < 1e-12
    assert prim["pass_10pct_literature"] is True


def test_primary_verdict_reports_a_failure_honestly():
    ids = ["D40", "D45", "D50", "D55", "D60"]
    ratios = [4.0, 4.5, 5.0, 5.5, 6.0]
    w = synthetic(0.58 * 1.2, 1.4, 2.5)     # +20 % everywhere
    runs = {cid: _fake_run(cid, r, float(wi)) for cid, r, wi in zip(ids, ratios, w)}
    s = A.analyse(runs)
    assert s["primary"]["pass_10pct_literature"] is False
    assert s["primary"]["rel_err_literature"] == pytest.approx(0.2, abs=1e-9)


def test_statements_flag_the_exponent_and_the_offset():
    ids = ["D40", "D45", "D50", "D55", "D60"]
    ratios = [4.0, 4.5, 5.0, 5.5, 6.0]
    good = {cid: _fake_run(cid, r, float(wi))
            for cid, r, wi in zip(ids, ratios, synthetic(0.58, 1.4, 2.5))}
    s = A.analyse(good)
    assert s["statements"]["shape_p_equals_2p5"]["pass"] is True
    assert s["statements"]["offset_k_in_1p2_1p6"]["pass"] is True
    assert s["statements"]["amplitude_C_in_0p55_0p60"]["pass"] is True
    assert s["statements"]["residual_collapse_below_10pct"]["pass"] is True

    bad_shape = {cid: _fake_run(cid, r, float(wi))
                 for cid, r, wi in zip(ids, ratios, synthetic(0.58, 1.4, 2.1))}
    s2 = A.analyse(bad_shape)
    assert s2["statements"]["shape_p_equals_2p5"]["pass"] is False

    bad_offset = {cid: _fake_run(cid, r, float(wi))
                  for cid, r, wi in zip(ids, ratios, synthetic(0.58, 0.0, 2.5))}
    s3 = A.analyse(bad_offset)
    assert s3["statements"]["offset_k_in_1p2_1p6"]["pass"] is False


# ---------------------------------------------------------------------------
# fill-height statement
# ---------------------------------------------------------------------------
def test_fill_height_test_detects_independence_and_dependence():
    ids = ["D40", "D45", "D50", "D55", "D60"]
    ratios = [4.0, 4.5, 5.0, 5.5, 6.0]
    w = synthetic(0.58, 1.4, 2.5)
    base = {cid: _fake_run(cid, r, float(wi)) for cid, r, wi in zip(ids, ratios, w)}
    w5 = float(synthetic(0.58, 1.4, 2.5)[2])

    # H-independent flow -> pass
    flat = dict(base)
    flat["D50_H20"] = _fake_run("D50_H20", 5.0, w5, fill_height=2.0 * 5 * D_PART)
    flat["D50_H25"] = _fake_run("D50_H25", 5.0, w5, fill_height=2.5 * 5 * D_PART)
    s = A.analyse(flat)
    assert s["fill_height_test"]["available"] is True
    assert s["fill_height_test"]["pass_10pct"] is True

    # a rate that keeps climbing with H -> fail
    tall = dict(base)
    tall["D50_H20"] = _fake_run("D50_H20", 5.0, w5 * 0.80, fill_height=2.0 * 5 * D_PART)
    tall["D50_H25"] = _fake_run("D50_H25", 5.0, w5 * 0.90, fill_height=2.5 * 5 * D_PART)
    s2 = A.analyse(tall)
    assert s2["fill_height_test"]["pass_10pct"] is False
    assert s2["fill_height_test"]["max_abs_rel_dev"] > 0.10


# ---------------------------------------------------------------------------
# reference provenance + case matrix
# ---------------------------------------------------------------------------
def test_reference_coefficients_are_read_not_hardcoded():
    ref = json.loads((ROOT / "benchmarks/m20055_beverloo/ref_beverloo_1961.json").read_text())
    assert A.C_REF == ref["coefficients"]["C"]
    assert A.K_REF == ref["coefficients"]["k"]
    assert A.C_RANGE == tuple(ref["coefficients"]["C_range_spherical"])
    assert A.K_RANGE == tuple(ref["coefficients"]["k_range_spherical"])


def test_every_scan_case_sits_inside_the_law_domain():
    import scan as S
    from tensordem.silo_flow.beverloo import DischargeCase  # noqa: F401

    ref = json.loads((ROOT / "benchmarks/m20055_beverloo/ref_beverloo_1961.json").read_text())
    for cid in S.case_ids():
        diag = S.domain_check(S.build_case(cid), ref)
        assert diag["in_domain"], (cid, diag)


def test_shard_partition_covers_every_case_exactly_once():
    import scan as S

    ids = S.case_ids()
    seen = []
    for shard in range(1, len(ids) + 1):
        seen += [cid for i, cid in enumerate(ids, start=1)
                 if (i - 1) % len(ids) == shard - 1]
    assert sorted(seen) == sorted(ids)
    assert len(seen) == len(set(seen))