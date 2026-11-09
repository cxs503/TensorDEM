#!/usr/bin/env python3
"""ISO 19906 external-anchor verification for the ``ice-abrasion-extremes`` case.

Task key: ``paper-exp-ice-abrasion-extremes-1`` (W18 paper-exp research plan, item 1:
"external anchor verification -- verbatim transcription + same-caliber conversion +
machine consistency checks").

Scientific question
-------------------
Can the external quantitative anchor of the topic (ISO 19906:2019 Annex A pressure-area
relation and the Clause 14/15 design global ice force) be closed *verbatim* and converted
into the caliber of this case, and once closed, how far is the case's own domain result
from it?

Decision to support (predeclared)
---------------------------------
D1: is the anchor usable as a quantitative acceptance baseline for the case (i.e. may any
    "agreement with ISO 19906" statement be issued)?
D2: which of the two candidate explanations dominates the case-vs-anchor distance --
    (a) the *caliber definition* (pressure on the load-bearing fraction vs on the nominal
    footprint) or (b) the *area dependence* of the anchor law?  D2 is decided on the
    algebra alone (case parameters + the law's form), i.e. without any anchor value.

Predeclared criteria (frozen before execution; a failed criterion is an honest FAIL and the
gap is written into the artifact -- thresholds are never tuned after the fact)
------------------------------------------------------------------------------
K1 transcription: every anchor carries verbatim quote + clause/table/page + raw value in
   the source's own units. Pass iff all anchors are VERIFIED.
K2 recompute: every conversion identity re-evaluated from the artifact round-trips with
   relative residual < 1e-9.
K3 dimensional homogeneity: exponent dimensionless; prefactor length exponent -1-2n for
   p = C A^n; conversion invariance under area-unit and pressure-unit changes < 1e-12.
K4 internal consistency of p(A): p(A0) = p0 (< 1e-12), strict monotone decrease for n < 0,
   log-log slope recovery error < 1e-9, area-weighted integral identity < 1e-9.
K5 caliber conversion re-derivable from the shipped sampler with residual < 1e-9.
K6 provenance: every number below is injected by this script; the shipped source module is
   hashed into the artifact.
K7 no unverified candidate in a compliance claim: if any anchor is not VERIFIED then
   ``domain_comparison.status`` must be NOT_ADJUDICABLE and every agreement flag False.

Everything numeric is computed here and written to
``benchmarks/anchors/ice-abrasion-extremes.json``; nothing is hand-typed.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "benchmarks" / "anchors" / "ice-abrasion-extremes.json"
FIG_DIR = ROOT / "docs" / "figs" / "ice_abrasion_anchor"
SOURCE_MODULE = ROOT / "src" / "tensordem" / "ice_abrasion.py"

REL_TOL = 1e-9          # K2/K5 recompute residual
EXACT_TOL = 1e-12       # K3/K4 algebraic identities
SLOPE_TOL = 1e-9        # K4 log-log slope recovery

# Candidate registry entries are *unverified candidates*: comparison/control only, never
# quotable for a compliance claim (K7).  Registered so the sensitivity study has a band to
# sweep; the band edges are deliberately wide.
CAND_P0_MPA = 3.0       # candidate reference pressure at A = 1 m^2 (sibling repo docstring)
CAND_EXPONENT = 0.45    # candidate |exponent| (sibling repo docstring)
EXPONENT_BAND = (0.0, 0.6)          # sweep band for |n|
AREA_BAND_M2 = (0.1, 10.0)          # plausible nominal contact-area range for a hull belt


# ---------------------------------------------------------------------------
# dimensionless bookkeeping
# ---------------------------------------------------------------------------
PRESSURE_DIMS = {"M": 1.0, "L": -1.0, "T": -2.0}
AREA_DIMS = {"L": 2.0}
M2_PER_FT2 = 0.09290304


def _combine(*terms):
    out = {}
    for dims, k in terms:
        for key, val in dims.items():
            out[key] = out.get(key, 0.0) + val * k
    return out


def prefactor_dims(n):
    """Dimensions of C in p = C A^n: dim(p) - n dim(A)."""
    return _combine((PRESSURE_DIMS, 1.0), (AREA_DIMS, -n))


def p_from_prefactor(C, A, n):
    return C * np.power(np.asarray(A, dtype=float), n)


def p_from_reference(p0, A, n, A0=1.0):
    """Dimensionally anchored form p = p0 (A/A0)^n with A0 = 1 m^2 (n < 0)."""
    return p0 * np.power(np.asarray(A, dtype=float) / A0, n)


def rel_residual(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = np.maximum(np.abs(b), np.finfo(float).tiny)
    return float(np.max(np.abs(a - b) / scale))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# 1. transcription ledger -- the part that cannot be manufactured here
# ---------------------------------------------------------------------------
def anchor_ledger():
    return [
        {
            "id": "iso19906-pA-exponent",
            "name": "ISO 19906 pressure-area relation p(A): prefactor and exponent",
            "source_document": ("ISO 19906:2019 Petroleum and natural gas industries -- "
                                "Arctic offshore structures, Annex A (pressure-area)"),
            "required_fields": ["verbatim_quote", "clause_or_table_or_page",
                                "value_raw_in_source_units"],
            "clause_table_page_target": ("Annex A pressure-area clause/table (exact item not "
                                        "yet identified -- see gap)"),
            "verbatim_quote": None,
            "value_raw": None,
            "unit_raw": None,
            "status": "WITHHELD",
            "retrieval_attempts": [
                {"channel": "local filesystem",
                 "command": "find /root /data /srv /opt -maxdepth 4 -iname '*19906*'",
                 "result": "no copy of the standard"},
                {"channel": "ecosystem repositories",
                 "command": "grep -rIl 'pressure.area|A.8.|Annex A' in TensorPhysics/"
                            "TensorPapers",
                 "result": "only derived model code, no verbatim quotation with table number"},
                {"channel": "publisher (www.iso.org)",
                 "command": "curl -s -o /dev/null -w '%{http_code}' "
                            "https://www.iso.org/standard/65655.html",
                 "result": "HTTP 403; standard is paywalled, no institutional license"},
            ],
            "gap": ("No licensed copy of ISO 19906:2019 is reachable from this sandbox, so the "
                    "prefactor and the exponent cannot be transcribed from the source text. "
                    "Values circulating in sibling repositories are unverified candidates and "
                    "stay in the comparison-only registry."),
        },
        {
            "id": "iso19906-design-global-ice-force",
            "name": "ISO 19906 design global ice force (Clause 14/15)",
            "source_document": ("ISO 19906:2019, Clause 14 / Clause 15 (global ice actions on "
                                "structures)"),
            "required_fields": ["verbatim_quote", "clause_or_table_or_page",
                                "value_raw_in_source_units"],
            "clause_table_page_target": ("Clause 14/15 global ice action formulation (exact "
                                        "clause number not yet identified -- see gap)"),
            "verbatim_quote": None,
            "value_raw": None,
            "unit_raw": None,
            "status": "WITHHELD",
            "retrieval_attempts": [
                {"channel": "local filesystem", "command": "same search as above",
                 "result": "no copy of the standard"},
                {"channel": "publisher (www.iso.org)", "command": "HTTP GET",
                 "result": "HTTP 403"},
            ],
            "gap": ("Same access gap as the p(A) anchor. Additionally the design global force is "
                    "structure-specific (a pressure-area integration over an exposure width), so "
                    "a verbatim clause value would still need a documented structure mapping "
                    "before it means anything for a hull-belt abrasion case."),
        },
    ]


# ---------------------------------------------------------------------------
# 2. structural (value-free) checks of the anchor law -> these CAN be closed now
# ---------------------------------------------------------------------------
def structural_checks():
    n_test = -CAND_EXPONENT
    C_mpa = CAND_P0_MPA
    A = np.array([0.05, 0.2, 1.0, 5.0, 20.0])
    A0 = 1.0

    # K3a dimensionality of the prefactor
    dims = prefactor_dims(n_test)
    length_exp_expected = -1.0 - 2.0 * n_test
    dims_ok = (abs(dims["L"] - length_exp_expected) < 1e-15
               and abs(dims["M"] - 1.0) < 1e-15 and abs(dims["T"] + 2.0) < 1e-15)

    # K3b two parameterizations must agree (A0 = 1 m^2 anchoring)
    p_pref = p_from_prefactor(C_mpa, A, n_test)
    p_ref = p_from_reference(CAND_P0_MPA, A, n_test, A0)
    resid_param = rel_residual(p_pref, p_ref)

    # K3c invariance under area unit change (m^2 -> ft^2): n invariant, C rescales
    C_ft = C_mpa * M2_PER_FT2 ** n_test
    p_ft = p_from_prefactor(C_ft, A / M2_PER_FT2, n_test)
    resid_area_unit = rel_residual(p_ft, p_pref)

    # K3d invariance under pressure unit change (MPa -> Pa)
    C_pa = C_mpa * 1e6
    p_pa = p_from_prefactor(C_pa, A, n_test) / 1e6
    resid_press_unit = rel_residual(p_pa, p_pref)

    # K4a p(A0) = p0
    p_at_A0 = float(p_from_reference(CAND_P0_MPA, np.array([A0]), n_test)[0])
    resid_A0 = rel_residual(p_at_A0, CAND_P0_MPA)

    # K4b strict monotone decrease for n < 0
    grid = np.geomspace(1e-3, 1e3, 4001)
    vals = p_from_reference(CAND_P0_MPA, grid, n_test)
    monotone = bool(np.all(np.diff(vals) < 0.0))

    # K4c log-log slope recovery (central difference -> analytic -n)
    A_mid = 3.7
    h = 1e-5
    lnp_up = math.log(float(p_from_reference(CAND_P0_MPA,
                                            np.array([A_mid * math.exp(h)]), n_test)[0]))
    lnp_dn = math.log(float(p_from_reference(CAND_P0_MPA,
                                            np.array([A_mid * math.exp(-h)]), n_test)[0]))
    slope_hat = (lnp_up - lnp_dn) / (2.0 * h)
    slope_err = abs(slope_hat - n_test)

    # K4d area-weighted mean identity (1/(A2-A1)) int p dA
    A1, A2 = 0.1, 10.0
    xg, wg = np.polynomial.legendre.leggauss(400)
    Aq = 0.5 * (A2 - A1) * xg + 0.5 * (A2 + A1)
    wq = 0.5 * (A2 - A1) * wg
    quad = float(np.sum(wq * p_from_reference(CAND_P0_MPA, Aq, n_test)))
    analytic = (CAND_P0_MPA / (n_test + 1.0)
                * (A2 ** (n_test + 1.0) - A1 ** (n_test + 1.0)))
    resid_integral = rel_residual(quad, analytic)

    # K3e Buckingham: 3 quantities (p, C, A) -> rank of the dimensional matrix
    mat = np.array([
        [PRESSURE_DIMS.get(k, 0.0), dims.get(k, 0.0), AREA_DIMS.get(k, 0.0)]
        for k in ("M", "L", "T")
    ], dtype=float)
    rank = int(np.linalg.matrix_rank(mat, tol=1e-12))
    n_groups = int(3 - rank)

    return {
        "law": "p(A) = C * A^n   <=>   p(A) = p0 * (A/A0)^n,  A0 = 1 m^2 (n < 0)",
        "prefactor_dims_of_C": dims,
        "prefactor_length_exponent_expected": length_exp_expected,
        "prefactor_dimensionally_homogeneous": bool(dims_ok),
        "buckingham_rank": rank,
        "buckingham_dimensionless_groups": n_groups,
        "buckingham_note": (f"{n_groups} dimensionless group for {{p, C, A}}: the law itself, "
                            "p/(C A^n) = 1. Dimensions therefore fix the units of C but never "
                            "its value -- C follows from one (A0, p0) anchoring pair, and n is "
                            "the only transferable dimensionless number of the anchor. Hence an "
                            "anchor exponent may only be compared as a pure number, and an "
                            "anchor prefactor must be re-anchored, not copied."),
        "pair_parameterization_residual": resid_param,
        "area_unit_change_residual": resid_area_unit,
        "pressure_unit_change_residual": resid_press_unit,
        "p_at_A0_residual": resid_A0,
        "strict_monotone_decreasing": monotone,
        "loglog_slope_recovered": slope_hat,
        "loglog_slope_target": n_test,
        "loglog_slope_error": slope_err,
        "area_weighted_mean_quadrature": quad,
        "area_weighted_mean_analytic": analytic,
        "area_weighted_mean_residual": resid_integral,
        "tolerances": {"parameterization": EXACT_TOL, "area_unit": EXACT_TOL,
                       "pressure_unit": EXACT_TOL, "p_at_A0": EXACT_TOL,
                       "slope": SLOPE_TOL, "integral": REL_TOL},
    }


# ---------------------------------------------------------------------------
# 3. the case caliber, read from the shipped sampler (no hand-typed numbers)
# ---------------------------------------------------------------------------
def case_caliber():
    from tensordem.ice_abrasion import ContactEnsembleConfig, sample_contact_ensemble

    config = ContactEnsembleConfig()
    ens = sample_contact_ensemble(config)
    frac = float(config.contact_area_fraction)

    l_f = ens.fragment_size_m.double().numpy()
    load = ens.normal_load_N.double().numpy()
    a_foot = l_f ** 2                       # nominal footprint of one fragment [m^2]
    a_load = frac * a_foot                  # load-bearing (real) contact area [m^2]
    p_load = load / a_load                  # pressure on the load-bearing area [Pa]
    p_foot = load / a_foot                  # pressure on the nominal footprint [Pa]

    # caliber identity: the footprint pressure is the load-bearing pressure times the fraction
    resid_identity = rel_residual(p_load * frac, p_foot)
    resid_load = rel_residual(load, p_foot * a_foot)
    resid_load2 = rel_residual(load, p_load * a_load)

    # case-implied pressure-area exponent: OLS of ln p on ln A (slope = n_case)
    x = np.log(a_foot)
    y = np.log(p_foot)
    xc = x - x.mean()
    slope = float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))
    resid = y - (y.mean() + slope * xc)
    n = x.size
    se = float(math.sqrt(np.dot(resid, resid) / (n - 2) / np.dot(xc, xc)))
    r = float(np.corrcoef(x, y)[0, 1])

    rng = np.random.default_rng(20261108)
    boot = np.empty(2000)
    for k in range(boot.size):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        xb = xb - xb.mean()
        boot[k] = float(np.dot(xb, yb - yb.mean()) / np.dot(xb, xb))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    ln_caliber = math.log(1.0 / frac)
    ln_area_band = math.log(AREA_BAND_M2[1] / AREA_BAND_M2[0]) * EXPONENT_BAND[1]

    return {
        "source_of_parameters": str(SOURCE_MODULE.relative_to(ROOT)),
        "parameters": {
            "n_samples": int(config.n_samples),
            "fragment_size_median_m": float(config.fragment_size_median_m),
            "fragment_size_log_sigma": float(config.fragment_size_log_sigma),
            "nominal_pressure_median_pa": float(config.nominal_pressure_median_pa),
            "nominal_pressure_log_sigma": float(config.nominal_pressure_log_sigma),
            "contact_area_fraction": frac,
            "seed": int(config.seed),
        },
        "medians": {
            "footprint_area_m2": float(np.median(a_foot)),
            "load_bearing_area_m2": float(np.median(a_load)),
            "pressure_on_load_bearing_area_pa": float(np.median(p_load)),
            "pressure_on_footprint_pa": float(np.median(p_foot)),
            "footprint_ensemble_mean_pa": float(p_foot.mean()),
        },
        "identity_residuals": {
            "p_load_bearing_times_fraction_minus_p_footprint": resid_identity,
            "load_minus_p_footprint_times_area": resid_load,
            "load_minus_p_load_bearing_times_area": resid_load2,
            "tolerance": REL_TOL,
        },
        "case_implied_exponent": {
            "quantity": "n_case = d ln p / d ln A over the shipped ensemble (footprint area)",
            "slope": slope,
            "standard_error": se,
            "bootstrap_95_interval": [float(lo), float(hi)],
            "pearson_r": r,
            "n": int(n),
            "interpretation": ("the case draws pressure and fragment size independently, so its "
                               "own pressure-area coupling is statistically zero; a monotone "
                               "decreasing external p(A) law is structurally absent from the "
                               "case"),
        },
        "caliber_ambiguity": {
            "factor_load_bearing_vs_footprint": 1.0 / frac,
            "abs_log_gap": ln_caliber,
            "max_area_band_effect_abs_log": ln_area_band,
            "area_band_m2": list(AREA_BAND_M2),
            "exponent_band_abs": list(EXPONENT_BAND),
            "caliber_dominates_area_band": bool(ln_caliber > ln_area_band),
            "decision_D2": ("caliber definition (load-bearing vs nominal footprint) dominates the "
                            "whole plausible area dependence of the anchor law"
                            if ln_caliber > ln_area_band else
                            "area dependence of the anchor law dominates the caliber choice"),
            "candidate_band_note": ("exponent band and area band are unverified candidates / "
                                    "plausibility windows used only to size the two effects; no "
                                    "compliance claim rests on them (K7)"),
        },
    }


# ---------------------------------------------------------------------------
# 4. conversion chain to the case caliber, recorded as executable steps
# ---------------------------------------------------------------------------
def conversion_chain(case):
    n_test = -CAND_EXPONENT
    C_mpa = CAND_P0_MPA
    frac = case["parameters"]["contact_area_fraction"]
    a_foot = case["medians"]["footprint_area_m2"]
    p_foot = case["medians"]["pressure_on_footprint_pa"] / 1e6      # MPa
    p_load = case["medians"]["pressure_on_load_bearing_area_pa"] / 1e6

    p_anchor = float(p_from_prefactor(C_mpa, np.array([a_foot]), n_test)[0])
    # p0 = p(A0) = C A0^n = C when A0 = 1 m^2; recovered here from the value at A_case
    p0_back = p_anchor * (1.0 / a_foot) ** n_test if n_test else p_anchor
    resid_param = rel_residual(p_from_prefactor(C_mpa, np.array([a_foot]), n_test),
                               p_from_reference(p0_back, np.array([a_foot]), n_test, 1.0))

    return [
        {
            "id": "S1",
            "step": "anchor value at the case footprint area",
            "expression": "p_anchor(A_case) = C * A_case^n, A_case = median footprint area",
            "inputs": {"C_MPa_m-2n": C_mpa, "n": n_test, "A_case_m2": a_foot},
            "value_MPa": p_anchor,
            "status": "CANDIDATE_ONLY (prefactor not transcribed; K1 open)",
        },
        {
            "id": "S2",
            "step": "ratio anchor vs case pressure on the SAME (footprint) caliber",
            "expression": "R = p_anchor(A_case) / p_case,footprint",
            "inputs": {"p_anchor_MPa": p_anchor, "p_case_footprint_MPa": p_foot},
            "value": float(p_anchor / p_foot),
            "status": "CANDIDATE_ONLY",
        },
        {
            "id": "S3",
            "step": ("the caliber trap: comparing the anchor with the case's nominal "
                     "(load-bearing) pressure instead of the footprint pressure"),
            "expression": "R_wrong = p_anchor(A_case) / p_case,load-bearing",
            "inputs": {"p_anchor_MPa": p_anchor, "p_case_load_bearing_MPa": p_load,
                       "factor": 1.0 / frac},
            "value": float(p_anchor / p_load),
            "status": "CANDIDATE_ONLY",
            "note": ("differs from S2 by the contact-area fraction; any comparison that omits "
                     "this step is off by that factor"),
        },
        {
            "id": "S4",
            "step": "recompute check: rebuild S1 from the reference parameterization p0 (A/A0)^-n",
            "expression": "p0 * (A_case/A0)^n vs C * A_case^n (A0 = 1 m^2)",
            "inputs": {"p0_MPa": p0_back, "A0_m2": 1.0},
            "residual": resid_param,
            "status": "RECOMPUTE",
        },
        {
            "id": "S5",
            "step": "round trip Pa <-> MPa on the case calibers",
            "expression": "p_case,footprint*1e6 -> Pa -> /1e6 -> MPa",
            "residual": rel_residual(p_foot * 1e6 / 1e6, p_foot),
            "status": "RECOMPUTE",
        },
        {
            "id": "S6",
            "step": "round trip load-bearing <-> footprint through the fraction",
            "expression": "p_load * fraction -> p_foot -> /fraction -> p_load",
            "residual": rel_residual(p_load * frac / frac, p_load),
            "status": "RECOMPUTE",
        },
    ]


# ---------------------------------------------------------------------------
# 5. figures
# ---------------------------------------------------------------------------
def figures(case):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    A = np.geomspace(1e-2, 1e2, 400)
    a_foot = case["medians"]["footprint_area_m2"]
    p_foot = case["medians"]["pressure_on_footprint_pa"] / 1e6
    p_load = case["medians"]["pressure_on_load_bearing_area_pa"] / 1e6

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for mag in (0.3, CAND_EXPONENT, 0.6):
        ax.loglog(A, p_from_prefactor(CAND_P0_MPA, A, -mag),
                  label=f"candidate p(A), n = -{mag:g} (unverified)")
    ax.axhline(p_foot, color="k", ls="--", lw=1.0, label="case pressure on footprint")
    ax.axhline(p_load, color="0.4", ls=":", lw=1.2,
               label="case pressure on load-bearing area")
    ax.axvspan(AREA_BAND_M2[0], AREA_BAND_M2[1], color="tab:orange", alpha=0.10,
               label="plausible nominal area band")
    ax.plot([a_foot], [p_foot], "ko", ms=5)
    ax.annotate(f"A_case = {a_foot:.3g} m$^2$", (a_foot, p_foot),
                textcoords="offset points", xytext=(8, -14), fontsize=8)
    ax.set_xlabel("contact area A [m$^2$]")
    ax.set_ylabel("ice pressure p [MPa]")
    ax.set_title("ISO 19906 p(A) candidates vs the ice-abrasion case calibers")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)
    path = FIG_DIR / "fig1_pA_vs_case_caliber.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append({"file": str(path.relative_to(ROOT)),
                 "caption": ("ISO 19906 p(A) candidate curves (exponents are unverified "
                             "candidates) against the ice-abrasion case pressures: the two case "
                             "calibers differ by the contact-area fraction, which is larger than "
                             "the whole area-band effect of the law.")})

    from tensordem.ice_abrasion import ContactEnsembleConfig, sample_contact_ensemble
    config = ContactEnsembleConfig()
    ens = sample_contact_ensemble(config)
    l_f = ens.fragment_size_m.double().numpy()
    load = ens.normal_load_N.double().numpy()
    a_foot_s = l_f ** 2
    p_foot_s = load / a_foot_s
    ci = case["case_implied_exponent"]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.loglog(a_foot_s, p_foot_s / 1e6, ".", ms=2.0, alpha=0.35,
              label=f"shipped ensemble ({int(config.n_samples)} contacts)")
    xs = np.geomspace(a_foot_s.min(), a_foot_s.max(), 50)
    bx = np.log(xs)
    by = np.log(p_foot_s).mean() + ci["slope"] * (bx - np.log(a_foot_s).mean())
    ax.loglog(xs, np.exp(by) / 1e6, "r-", lw=1.4,
              label=f"OLS slope n_case = {ci['slope']:+.4f} +/- {ci['standard_error']:.4f}")
    ax.set_xlabel("fragment footprint area A [m$^2$]")
    ax.set_ylabel("contact pressure p [MPa]")
    ax.set_title("Case-implied pressure-area coupling (structurally n = 0)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    path2 = FIG_DIR / "fig2_case_implied_exponent.png"
    fig.tight_layout()
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    made.append({"file": str(path2.relative_to(ROOT)),
                 "caption": ("The case's own pressure-area relation measured on the shipped "
                             "sampler: pressure and fragment area are drawn independently, so the "
                             "case carries no pressure-area coupling at all.")})
    return made


# ---------------------------------------------------------------------------
# 6. assemble
# ---------------------------------------------------------------------------
def evaluate(criteria, anchors, struct, case, steps):
    tol = struct["tolerances"]
    k3_pass = (struct["prefactor_dimensionally_homogeneous"]
               and struct["pair_parameterization_residual"] < tol["parameterization"]
               and struct["area_unit_change_residual"] < tol["area_unit"]
               and struct["pressure_unit_change_residual"] < tol["pressure_unit"])
    k4_pass = (struct["p_at_A0_residual"] < tol["p_at_A0"]
               and struct["strict_monotone_decreasing"]
               and struct["loglog_slope_error"] < tol["slope"]
               and struct["area_weighted_mean_residual"] < tol["integral"])
    idr = case["identity_residuals"]
    k5_pass = (idr["p_load_bearing_times_fraction_minus_p_footprint"] < REL_TOL
               and idr["load_minus_p_footprint_times_area"] < REL_TOL
               and idr["load_minus_p_load_bearing_times_area"] < REL_TOL)
    recompute = [s for s in steps if "residual" in s]
    k2_pass = all(s["residual"] < REL_TOL for s in recompute)
    k1_pass = all(a["status"] == "VERIFIED" for a in anchors)
    domain_status = "NOT_ADJUDICABLE" if not k1_pass else "ADJUDICABLE"
    k7_pass = (not k1_pass) and domain_status == "NOT_ADJUDICABLE"

    med = case["medians"]
    ci = case["case_implied_exponent"]
    cal = case["caliber_ambiguity"]
    criteria += [
        {"id": "K1", "statement": ("anchor transcribed verbatim with quote + clause/table/page "
                                   "+ raw value"),
         "threshold": "all anchors VERIFIED",
         "measured": (f"{sum(a['status'] == 'VERIFIED' for a in anchors)}/{len(anchors)} anchors "
                      f"verified"),
         "passed": k1_pass},
        {"id": "K2", "statement": "conversion chain recomputes from the artifact",
         "threshold": f"max relative residual < {REL_TOL:g}",
         "measured": f"max residual {max(s['residual'] for s in recompute):.3e}",
         "passed": k2_pass},
        {"id": "K3", "statement": "dimensional homogeneity + unit-change invariance",
         "threshold": f"exact < {EXACT_TOL:g}",
         "measured": (f"prefactor L exponent {struct['prefactor_dims_of_C']['L']:.6f} (expected "
                      f"{struct['prefactor_length_exponent_expected']:.6f}); area-unit residual "
                      f"{struct['area_unit_change_residual']:.2e}; pressure-unit residual "
                      f"{struct['pressure_unit_change_residual']:.2e}"),
         "passed": k3_pass},
        {"id": "K4", "statement": ("internal consistency of p(A): p(A0)=p0, monotone, slope "
                                   "recovery, area-weighted identity"),
         "threshold": f"< {EXACT_TOL:g} / < {SLOPE_TOL:g} / < {REL_TOL:g}",
         "measured": (f"p(A0) residual {struct['p_at_A0_residual']:.2e}; monotone "
                      f"{struct['strict_monotone_decreasing']}; slope error "
                      f"{struct['loglog_slope_error']:.2e}; integral residual "
                      f"{struct['area_weighted_mean_residual']:.2e}"),
         "passed": k4_pass},
        {"id": "K5", "statement": "caliber conversion re-derivable from the shipped sampler",
         "threshold": f"< {REL_TOL:g}",
         "measured": (f"max identity residual "
                      f"{max(v for k, v in idr.items() if k != 'tolerance'):.3e}; p_load median "
                      f"{med['pressure_on_load_bearing_area_pa']:.6e} Pa, p_footprint median "
                      f"{med['pressure_on_footprint_pa']:.6e} Pa"),
         "passed": k5_pass},
        {"id": "K6", "statement": "provenance: every number script-injected, source hashed",
         "threshold": "script + source sha256 present",
         "measured": "sha256 recorded for the generator and the sampled source module",
         "passed": True},
        {"id": "K7", "statement": "no unverified candidate used in a compliance claim",
         "threshold": "domain comparison NOT_ADJUDICABLE while K1 open",
         "measured": f"domain_comparison = {domain_status}",
         "passed": k7_pass},
    ]

    open_criteria = [c["id"] for c in criteria if not c["passed"]]
    verdict = {
        "overall": "HONEST_FAIL" if open_criteria else "PASS",
        "open_criteria": open_criteria,
        "headline": ("external anchor could NOT be closed: no licensed ISO 19906:2019 text is "
                     "reachable, so K1 stays open and no agreement with the standard may be "
                     "claimed. The value-free part of the anchor (its algebraic form) was closed "
                     "and checked; the case-vs-anchor distance is dominated by the caliber "
                     "definition, not by the anchor's area dependence."
                     if open_criteria else "anchor closed and converted"),
    }
    domain_comparison = {
        "status": domain_status,
        "reason": ("the p(A) prefactor/exponent and the Clause 14/15 design force are WITHHELD "
                   "(K1 open); quoting a numerical distance to the standard would rest on "
                   "unverified candidates"),
        "what_is_still_quantified": [
            (f"caliber conversion factor between the two case pressure definitions: "
             f"{cal['factor_load_bearing_vs_footprint']:.1f}x (|ln| = {cal['abs_log_gap']:.3f}) vs "
             f"at most {cal['max_area_band_effect_abs_log']:.3f} for the anchor's whole plausible "
             f"area dependence"),
            (f"case-implied exponent n_case = {ci['slope']:+.4f} +/- {ci['standard_error']:.4f} "
             f"(95% bootstrap [{ci['bootstrap_95_interval'][0]:+.4f}, "
             f"{ci['bootstrap_95_interval'][1]:+.4f}]); a monotone decreasing external p(A) law "
             f"implies n < 0, so the case's pressure model is structurally not a p(A) realization"),
        ],
        "what_would_close_it": [
            ("a licensed copy of ISO 19906:2019 (Annex A pressure-area clause/table + Clause "
             "14/15 global ice action), transcribed with quote + table/clause number + page"),
            ("then re-run this script with the anchor entries filled (status VERIFIED); K1 flips "
             "and domain_comparison becomes ADJUDICABLE"),
            ("additionally: state the exposure-width mapping used to turn a global force into a "
             "hull-belt pressure before comparing with a per-contact pressure"),
        ],
        "agreement_declared": False,
    }
    return criteria, verdict, domain_comparison


def build():
    anchors = anchor_ledger()
    struct = structural_checks()
    case = case_caliber()
    steps = conversion_chain(case)
    criteria, verdict, domain_comparison = evaluate([], anchors, struct, case, steps)

    payload = {
        "artifact": "ice-abrasion-extremes external anchor verification",
        "task_key": "paper-exp-ice-abrasion-extremes-1",
        "topic": "ice-abrasion-extremes",
        "experiment": "research_plan item 1 -- external anchor verification",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository": "TensorDEM",
        "git_rev": git_rev(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "torch": torch.__version__},
        "decision_supported": [
            ("D1: may any 'agreement with ISO 19906' statement be issued for this case? -> no, "
             "while K1 is open"),
            ("D2: what dominates the case-vs-anchor distance -> the caliber definition "
             "(load-bearing vs nominal footprint), by the algebra alone"),
        ],
        "criteria": criteria,
        "verdict": verdict,
        "anchors": anchors,
        "structural_form": struct,
        "case_caliber": case,
        "conversion_chain": steps,
        "candidate_registry": [
            {"id": "candidate-iso19906-p0", "value": CAND_P0_MPA, "unit": "MPa at A = 1 m^2",
             "source": ("sibling repository docstring (TensorPhysics "
                        "ice_plasticity.iso19906), not the standard text"),
             "status": "UNVERIFIED_CANDIDATE", "permitted_use": "comparison / sensitivity only"},
            {"id": "candidate-iso19906-exponent", "value": CAND_EXPONENT, "unit": "-",
             "source": "sibling repository docstring, not the standard text",
             "status": "UNVERIFIED_CANDIDATE", "permitted_use": "comparison / sensitivity only"},
        ],
        "domain_comparison": domain_comparison,
        "provenance": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "script_sha256": sha256(Path(__file__).resolve()),
            "source_modules": [{"path": str(SOURCE_MODULE.relative_to(ROOT)),
                                "sha256": sha256(SOURCE_MODULE)}],
            "reproduce": "PYTHONPATH=src python benchmarks/anchors/ice_abrasion_extremes_anchor.py",
            "number_policy": "every numeric field in this file is injected by the script; no "
                             "hand-typed values",
        },
    }
    payload["figures"] = figures(case)
    return payload


def main():
    payload = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(json.dumps({"verdict": payload["verdict"]["overall"],
                      "open_criteria": payload["verdict"]["open_criteria"],
                      "domain_comparison": payload["domain_comparison"]["status"],
                      "criteria": {c["id"]: c["passed"] for c in payload["criteria"]},
                      "p_case_footprint_MPa":
                          payload["case_caliber"]["medians"]["pressure_on_footprint_pa"] / 1e6,
                      "p_case_load_bearing_MPa":
                          payload["case_caliber"]["medians"]
                          ["pressure_on_load_bearing_area_pa"] / 1e6,
                      "n_case": payload["case_caliber"]["case_implied_exponent"]["slope"]},
                     indent=1))


if __name__ == "__main__":
    main()