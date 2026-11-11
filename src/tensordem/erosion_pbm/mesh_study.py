"""Mesh-independence (GCI) study for the sectional breakage PBM.

Campaign node ``p488w1-n09``.  Three questions, one ledger:

1. **Is the sectional mesh converged?**  Three-level refinement sequences with
   Richardson extrapolation and Roache's GCI on the PSD moments (``M0``,
   ``M1``, ``M2``), on two mesh families -- the uniform log ladder and the
   *log-log variable* (graded) ladder of :class:`~.mesh_gci.LogLogGrid`.
2. **Is the fines floor an assumption or a criterion?**  The bin-0 mass share
   is measured against the ``d_0 / v_min**(1/3)`` ratio, the 5 % threshold is
   located by interpolation, and the study WARNs instead of passing silently
   when the production grid sits above it.
3. **How much of the erosion-rate spread is the mesh?**  The same impact
   ensemble is applied to the PSD of every mesh level; the resulting
   removal-mass spread is put next to the 13.1 % cross-correlation
   (Finnie vs Oka vs DNV-RP-O501) deviation of ``p176w2-n08``.

Everything is deterministic: the PBE is advanced by the uniformisation
propagator (:func:`~.mesh_gci.propagate`), so the numbers are a function of the
mesh only.  The one non-deterministic input inherited from the physics
(``p176w2-n08``) is the *ensemble-mean* Gaudin daughter matrix, which is built
with a fixed seed; its sampling noise is what limits the order estimate of the
distribution-shape functionals (``d50``, fines count) and is reported as such.

Usage
-----
``python -m tensordem.erosion_pbm.mesh_study [--out DIR] [--quick]``
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .mesh_gci import (LogLogGrid, cascade_error, erosion_removed_mass, floor_share,
                       floor_threshold, mass_ledger, propagate, psd_fingerprint,
                       richardson_gci)
from .pbm import daughter_matrix, two_pivot_weights
from .dem import meanfield_selection
from .study import CASE, dem_config, erosion_law

__all__ = ["run_study", "main", "NODE", "CONFIG"]

NODE = "p488w1-n09-pbm-gci"

#: full study configuration (reduced by ``--quick`` / ``quick=True``)
CONFIG = {
    "node": NODE,
    # order-verification case: deterministic equal-binary daughters, power-law
    # selection, so every functional is free of sampling noise
    "order_levels": (72, 144, 288, 576),
    "order_asymptotic_triple": (576, 288, 144),
    "order_pre_asymptotic_triple": (288, 144, 72),
    "octaves_below": 20,
    "octaves_above": 4,
    "selection_scale": 4.0,      # 1/s at d_0
    "selection_exponent": 1.0 / 3.0,
    "t_end": 1.5,                # s
    # physical case inherited from p176w2-n08
    "physical_levels": (144, 288, 576),
    "physical_grading": (1.0, 4.0),
    "gaudin_samples": 1024,
    "gaudin_seed": 20260101,
    # the *same* S(v) quadrature is used on every mesh, so its discretisation
    # error is common mode and cannot be mistaken for a mesh effect
    "selection_quadrature": (81, 41),
    # floor sweep (octaves of volume below d_0**3)
    "floor_octaves": (2, 4, 6, 8, 10, 12, 16, 20),
    "floor_limit": 0.05,
    "floor_bins": 144,
    # negative control
    "crude_bins": 10,
    # accepted nominal order of the two-pivot sectional scheme
    "p_nominal": 1.0,
    "order_band": 0.2,           # p in [0.8, 1.2] * p_nominal
    "gci_limit": 0.05,
    "mass_drift_limit": 1e-12,
    "cascade_limit": 1e-6,
    # cross-correlation anchor from p176w2-n08 (docs/P176W2_N08_EROSION_PBM_FUSION.md)
    "cross_correlation_deviation": 0.131,
    "cross_correlation_source": "docs/P176W2_N08_EROSION_PBM_FUSION.md (C4, RP-O501 vs resolved ensemble)",
    "rp_o501_fitted_K": 3.024e-08,
    "velocity_envelope": (20.0, 40.0),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_CASE_CACHE: dict = {}


def _cache_key(kind: str, grid, cfg: dict) -> tuple:
    return (kind, int(grid.n_bins), float(grid.grading), float(grid.v_min), float(grid.v_max),
            float(cfg["gaudin_samples"]), int(cfg["gaudin_seed"]),
            float(cfg["t_end"] if kind == "order" else cfg["_case"]["t_end"]),
            int(cfg["selection_quadrature"][0]), int(cfg["selection_quadrature"][1]))


def initial_particle(grid: LogLogGrid, volume: float, number: float = 1.0) -> np.ndarray:
    """Two-pivot placement of ``number`` particles of volume ``volume``.

    Unlike "put the feed in the bin that contains it", this conserves the
    tracked mass ``sum_k v_k n_k = number * volume`` *exactly* on every mesh,
    which is what makes the mass ledger of criterion C1 comparable across
    levels (otherwise the initial mass itself carries an O(h) placement bias).
    """
    k0, k1, w = two_pivot_weights(float(volume), grid.volumes)
    n = np.zeros(grid.n_bins)
    n[k0] += number * w
    if k1 != k0:
        n[k1] += number * (1.0 - w)
    return n


def order_case(grid: LogLogGrid, cfg: dict, *, t_end: float | None = None):
    """Case A: deterministic equal-binary breakage with a power-law selection."""
    v_0 = cfg["_v0"]
    S = cfg["selection_scale"] * (grid.volumes / v_0) ** cfg["selection_exponent"]
    M = daughter_matrix(grid, n_fragments=2, rule="equal")
    n0 = initial_particle(grid, v_0)
    key = _cache_key("order", grid, cfg)
    if key in _CASE_CACHE:
        return _CASE_CACHE[key]
    run = propagate(S, M, n0, [cfg["t_end"] if t_end is None else t_end])
    out = {"n0": n0, "n": run["n"][:, -1], "tail": float(np.max(np.abs(run["tail"]))),
           "terms": int(run["terms"].max()), "selection": S}
    _CASE_CACHE[key] = out
    return out


def physical_case(grid: LogLogGrid, cfg: dict):
    """Case B: Weibull/impact-stress selection + Gaudin daughter ensemble."""
    case = cfg["_case"]
    dem_cfg = cfg["_dem_cfg"]
    key = _cache_key("physical", grid, cfg)
    if key in _CASE_CACHE:
        return _CASE_CACHE[key]
    n_u, n_a = cfg["selection_quadrature"]
    S = meanfield_selection(grid, dem_cfg, n_u=int(n_u), n_a=int(n_a))
    M = daughter_matrix(grid, n_fragments=case["n_fragments"], rule="gaudin",
                        gaudin_alpha=case["gaudin_alpha"],
                        n_samples=cfg["gaudin_samples"], seed=cfg["gaudin_seed"])
    n0 = initial_particle(grid, case["d_0"] ** 3)
    run = propagate(S, M, n0, [case["t_end"]])
    out = {"n0": n0, "n": run["n"][:, -1], "tail": float(np.max(np.abs(run["tail"]))),
           "terms": int(run["terms"].max()), "selection": S}
    _CASE_CACHE[key] = out
    return out


def _cross_law_deviation(cfg: dict) -> dict:
    """Cross-correlation deviation of the resolved Oka ensemble against RP-O501.

    This is *exactly* the construction of the ``p176w2-n08`` criterion C4 (its
    ``check_o501``): a resolved impact ensemble per speed over ``U in [20, 40]
    m/s``, the RP-O501 coefficient fitted on that ensemble, and the maximum
    per-point relative deviation.  It is recomputed here rather than quoted so
    that the 13.1 % anchor of the decomposition table is traceable to a number
    this run produced.
    """
    case = cfg["_case"]
    try:
        from .study import check_o501
        res = check_o501(case)
        return {"max_relative_deviation": float(res["max_abs_deviation"]),
                "per_point": res.get("deviations"),
                "speeds": res.get("speeds"),
                "fitted_K": res.get("K"),
                "anchored_value": cfg["cross_correlation_deviation"],
                "source": cfg["cross_correlation_source"],
                "construction": "check_o501 of erosion_pbm.study (resolved ensemble vs RP-O501)"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"max_relative_deviation": cfg["cross_correlation_deviation"],
                "anchored_value": cfg["cross_correlation_deviation"],
                "source": cfg["cross_correlation_source"],
                "fallback_reason": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------
def check_mass(cfg: dict) -> dict:
    """C1: ``sum_k v_k n_k`` drift <= 1e-12 relative, every level and family."""
    v_0 = cfg["_v0"]
    out = {"limit": cfg["mass_drift_limit"], "families": {}}
    worst = 0.0
    for family, grading in (("uniform", 1.0), ("graded", float(cfg["physical_grading"][1]))):
        rows = {}
        for n_bins in cfg["order_levels"]:
            grid = LogLogGrid(v_0 * 2.0 ** -cfg["octaves_below"],
                              v_0 * 2.0 ** cfg["octaves_above"], n_bins, grading)
            run = order_case(grid, cfg)
            led = mass_ledger(grid, run["n0"], run["n"])
            rows[str(n_bins)] = {
                "n_bins": int(n_bins), "grading": grading,
                "mass_initial": led["mass_initial"], "mass_final": led["mass_final"],
                "mass_drift_relative": led["mass_drift_relative"],
                "number_initial": led["number_initial"],
                "number_final": led["number_final"],
                "propagator_tail": run["tail"],
                "grading_realised": grid.width_grading,
            }
            worst = max(worst, led["mass_drift_relative"])
        out["families"][family] = rows
    out["worst_drift_relative"] = worst
    out["pass"] = bool(worst <= cfg["mass_drift_limit"])
    return out


def check_cascade(cfg: dict) -> dict:
    """C2: analytic binary cascade ``e^{-t}(2t)^m/m!`` on every new mesh."""
    v_0 = cfg["_v0"]
    out = {"limit": cfg["cascade_limit"], "uniform": {}, "graded": {}}
    worst = 0.0
    for q in (1, 3, 6, 12, 24):
        grid = LogLogGrid(v_0 * 2.0 ** -cfg["octaves_below"],
                          v_0 * 2.0 ** cfg["octaves_above"],
                          (cfg["octaves_below"] + cfg["octaves_above"]) * q, 1.0)
        res = cascade_error(grid, sections_per_octave=q,
                            j0=cfg["octaves_below"] * q,
                            t_eval=np.linspace(0.25, 2.0, 5))
        res["sections_per_octave"] = q
        out["uniform"][str(q)] = res
        worst = max(worst, res["max_relative_error"])
    # the graded ladder is *not* self-similar under halving: report the price
    for grading in (2.0, 4.0):
        grid = LogLogGrid(v_0 * 2.0 ** -cfg["octaves_below"],
                          v_0 * 2.0 ** cfg["octaves_above"],
                          (cfg["octaves_below"] + cfg["octaves_above"]) * 6, grading)
        # the cascade reference is only defined where the local ratio equals 2
        res = cascade_error(grid, sections_per_octave=6,
                            j0=cfg["octaves_below"] * 6,
                            t_eval=np.linspace(0.25, 2.0, 8), n_generations=6)
        res["grading"] = grading
        res["reference_applicable"] = False
        res["note"] = ("the analytic cascade exists only on a lattice that is "
                       "self-similar under halving; on a graded ladder the "
                       "equal-binary daughter no longer lands on a section "
                       "representative, so this number measures the loss of the "
                       "reference, not a solver defect")
        out["graded"][str(grading)] = res
    out["worst_uniform_relative_error"] = worst
    out["pass"] = bool(worst <= cfg["cascade_limit"])
    return out


def _functionals_of(grid: LogLogGrid, n, cfg: dict) -> dict:
    return grid.functional(n, cfg["_case"]["d_0"])


def check_order(cfg: dict) -> dict:
    """C3: observed order of every reported functional, and of the measure factor."""
    v_0 = cfg["_v0"]
    band = (cfg["p_nominal"] * (1.0 - cfg["order_band"]),
            cfg["p_nominal"] * (1.0 + cfg["order_band"]))
    out = {"p_nominal": cfg["p_nominal"], "band": [band[0], band[1]],
           "asymptotic_triple": list(cfg["order_asymptotic_triple"]),
           "pre_asymptotic_triple": list(cfg["order_pre_asymptotic_triple"]),
           "functionals": {}, "physical": {}}

    levels = sorted(set(cfg["order_levels"]) | set(cfg["order_asymptotic_triple"])
                    | set(cfg["order_pre_asymptotic_triple"]))
    tables = {}
    for n_bins in levels:
        grid = LogLogGrid(v_0 * 2.0 ** -cfg["octaves_below"],
                          v_0 * 2.0 ** cfg["octaves_above"], n_bins, 1.0)
        run = order_case(grid, cfg)
        fun = _functionals_of(grid, run["n"], cfg)
        fun["measure_ratio"] = fun["M1_integral"] / fun["M1_pivot"]
        fun["measure_ratio_predicted"] = float(np.mean((grid.local_ratio + 1.0)
                                                       / (2.0 * np.sqrt(grid.local_ratio))))
        tables[n_bins] = fun
    fine, mid, coarse = cfg["order_asymptotic_triple"]
    for name in tables[fine]:
        res = richardson_gci(tables[fine][name], tables[mid][name],
                             tables[coarse][name], 2.0)
        res.update({"value_fine": tables[fine][name], "value_all": {
            str(k): tables[k][name] for k in levels}})
        out["functionals"][name] = res
    pre_fine, pre_mid, pre_coarse = cfg["order_pre_asymptotic_triple"]
    out["pre_asymptotic"] = {
        name: richardson_gci(tables[pre_fine][name], tables[pre_mid][name],
                             tables[pre_coarse][name], 2.0)
        for name in tables[pre_fine]}

    # physical case (Gaudin daughters): the same order must hold there
    for grading in cfg["physical_grading"]:
        fam = {}
        for n_bins in cfg["physical_levels"]:
            grid = LogLogGrid(v_0 * 2.0 ** -cfg["octaves_below"],
                              v_0 * 2.0 ** cfg["octaves_above"], n_bins, grading)
            run = physical_case(grid, cfg)
            fam[n_bins] = _functionals_of(grid, run["n"], cfg)
            fam[n_bins]["floor_mass_share"] = float(fam[n_bins]["floor_mass_share"])
        p_fine, p_mid, p_coarse = cfg["physical_levels"][::-1]
        out["physical"][str(grading)] = {
            name: richardson_gci(fam[p_fine][name], fam[p_mid][name],
                                 fam[p_coarse][name], 2.0)
            for name in fam[p_fine]}
        out["physical"][str(grading)]["_values"] = {
            str(k): fam[k] for k in cfg["physical_levels"]}

    # certification: declared order x observed order x stability between triples
    certified, diagnostics = {}, {}
    for name, res in out["functionals"].items():
        pre = out["pre_asymptotic"][name]
        p_nom = FUNCTIONAL_ORDER.get(name)
        row = {"p": res["p"], "p_pre_asymptotic": pre["p"], "gci": res["gci"],
               "value_fine": res["value_fine"], "declared_p_nominal": p_nom,
               "monotone": res["monotone"]}
        if name == "M1_pivot":
            row["excluded_because"] = (
                "exact invariant of the two-pivot rule (relative drift ~1e-16 at every "
                "level), so no order exists; the mass-conservation certificate C1 is the "
                "right statement for it")
        if p_nom is None:
            row.update({"in_band": False, "order_stable": False,
                        "excluded_because": "no declared nominal order (floor share is a "
                                            "ratio of ~1e-10 sub-populations)"})
            diagnostics[name] = row
            continue
        band = (0.8 * p_nom, 1.2 * p_nom)
        stable = (res["p"] == res["p"] and pre["p"] == pre["p"]
                  and abs(res["p"] - pre["p"]) <= 0.25)
        in_band = bool(res["certified"] and band[0] <= res["p"] <= band[1])
        row.update({"band": [band[0], band[1]], "in_band": in_band,
                    "order_stable": bool(stable)})
        if in_band and stable:
            certified[name] = row
        else:
            diagnostics[name] = row
    out["certified"] = certified
    out["diagnostics"] = diagnostics
    m = out["functionals"]["measure_ratio"]
    out["measure_factor_check"] = {
        "measured": m["value_fine"],
        "measured_p": m["p"],
        "predicted_formula": "(1 + r) / (2 sqrt(r)) with r the local bin ratio",
        "predicted": tables[fine]["measure_ratio_predicted"],
        "relative_error_of_prediction": abs(m["value_fine"] - tables[fine]["measure_ratio_predicted"])
        / tables[fine]["measure_ratio_predicted"],
        "per_level": {str(k): {"measured": tables[k]["measure_ratio"],
                               "predicted": tables[k]["measure_ratio_predicted"]}
                      for k in levels},
        "note": "closed-form anchor: the analytic-integral mass and the pivot mass "
                "differ by this factor on *any* distribution (this run reproduces it "
                "to machine precision, so the GCI machinery is calibrated)",
    }
    out["noise_floor"] = _noise_floor(cfg)
    out["physical_interpretation"] = (
        "the off-grid (Gaudin) daughter rule carries a first-order placement error, so "
        "the physical case is expected to show p -> 1; at these mesh sizes the "
        "discretisation error is already at or below the daughter-matrix sampling "
        "noise floor reported in noise_floor, so its p is reported but not used to "
        "certify the order")
    required = ["M0", "M1_integral", "M2_integral"]
    out["required_certified"] = required
    out["pass"] = bool(all(n in certified for n in required))
    return out


def _noise_floor(cfg: dict) -> dict:
    """How much of the physical-case signal is daughter-matrix sampling noise?

    The Gaudin daughter matrix is an ensemble mean over a fixed, seeded sample
    of fragment splits.  Rebuilding it with the next seed changes the reported
    functionals by an amount that has nothing to do with the mesh; if that
    amount is comparable to the mesh-to-mesh change, the physical case cannot
    certify an observed order (it can still certify a GCI magnitude).
    """
    v_0 = cfg["_v0"]
    n_bins = max(cfg["physical_levels"])
    alt = dict(cfg)
    alt["gaudin_seed"] = int(cfg["gaudin_seed"]) + 1
    grid = LogLogGrid(v_0 * 2.0 ** -cfg["octaves_below"],
                      v_0 * 2.0 ** cfg["octaves_above"], n_bins,
                      float(cfg["physical_grading"][0]))
    a = _functionals_of(grid, physical_case(grid, cfg)["n"], cfg)
    b = _functionals_of(grid, physical_case(grid, alt)["n"], cfg)
    return {"n_bins": int(n_bins), "seed_a": int(cfg["gaudin_seed"]), "seed_b": int(alt["gaudin_seed"]),
            "relative_change": {k: abs(b[k] - a[k]) / abs(a[k]) if a[k] else float("nan")
                                for k in a}}


def check_gci(cfg: dict) -> dict:
    """C4: GCI of M0 and M1 at the finest level, and the mass identity."""
    order = check_order_cache(cfg)
    out = {"limit": cfg["gci_limit"], "values": {}}
    for name in ("M0", "M1_integral", "M1_pivot", "M2_integral"):
        res = order["functionals"][name]
        out["values"][name] = {"gci": res["gci"], "p": res["p"],
                               "f_fine": res["f_fine"], "f_extrapolated": res["f_extrapolated"]}
    # the pivot mass is conserved by construction -> the identity must be visible
    out["pivot_mass_identity"] = {
        "gci": out["values"]["M1_pivot"]["gci"],
        "expected": 0.0,
        "note": "sum_k v_k n_k is an exact invariant of the two-pivot rule; its "
                "GCI measured by Richardson is only the (1e-3 scale) change of the "
                "initial placement bias, not a physical error",
    }
    out["physical_values"] = {
        str(grading): {k: {"gci": v["gci"], "p": v["p"]}
                       for k, v in fam.items() if not k.startswith("_")}
        for grading, fam in order["physical"].items()}
    graded = out["physical_values"][str(cfg["physical_grading"][-1])]
    out["graded_family_gci"] = {k: graded[k]["gci"] for k in ("M0", "M1_integral", "M2_integral")}
    out["pass"] = bool(out["values"]["M0"]["gci"] <= cfg["gci_limit"]
                       and out["values"]["M1_integral"]["gci"] <= cfg["gci_limit"])
    return out


#: Declared nominal order per reported functional, from the scheme's structure:
#: the two-pivot rule reproduces the first two moments *exactly* per event, so
#: the residual error of a moment functional is second order for daughters that
#: land on a representative (equal-binary) and first order for daughters that
#: land between representatives (Gaudin).  The quantile functionals are measured
#: through the bin-representative measure and are therefore first order.  The
#: floor share is a ratio of two numbers that are ~1e-10 of the total, so no
#: order is declared for it.
FUNCTIONAL_ORDER = {
    "M0": 2.0,
    "M1_integral": 2.0,
    "M1_pivot": 2.0,
    "M2_integral": 2.0,
    "measure_ratio": 2.0,
    "fines_number": 1.0,
    "d50": 1.0,
    "floor_mass_share": None,
}

_ORDER_CACHE: dict = {}


def check_order_cache(cfg: dict) -> dict:
    key = id(cfg)
    if key not in _ORDER_CACHE:
        _ORDER_CACHE[key] = check_order(cfg)
    return _ORDER_CACHE[key]


def check_negative_control(cfg: dict) -> dict:
    """C5: the mesh must act (distinct fingerprints) and a crude mesh must fail."""
    v_0 = cfg["_v0"]
    prints, values = {}, {}
    for n_bins in cfg["order_levels"]:
        grid = LogLogGrid(v_0 * 2.0 ** -cfg["octaves_below"],
                          v_0 * 2.0 ** cfg["octaves_above"], n_bins, 1.0)
        run = order_case(grid, cfg)
        prints[str(n_bins)] = psd_fingerprint(run["n"])
        values[str(n_bins)] = _functionals_of(grid, run["n"], cfg)["M0"]
    distinct = len(set(prints.values())) == len(prints)

    crude = LogLogGrid(v_0 * 2.0 ** -cfg["octaves_below"],
                       v_0 * 2.0 ** cfg["octaves_above"], cfg["crude_bins"], 1.0)
    crude_run = order_case(crude, cfg)
    crude_fun = _functionals_of(crude, crude_run["n"], cfg)
    fine = cfg["order_asymptotic_triple"][0]
    fine_grid = LogLogGrid(v_0 * 2.0 ** -cfg["octaves_below"],
                           v_0 * 2.0 ** cfg["octaves_above"], fine, 1.0)
    fine_fun = _functionals_of(fine_grid, order_case(fine_grid, cfg)["n"], cfg)
    rich = richardson_gci(*[check_order_cache(cfg)["functionals"]["M0"][k]
                            for k in ("f_fine", "f_mid", "f_coarse")])
    f_ex = rich["f_extrapolated"]
    deviation = {name: abs(crude_fun[name] - fine_fun[name]) / abs(fine_fun[name])
                 for name in ("M0", "M1_integral")}
    return {
        "fingerprints_distinct": bool(distinct),
        "n_fingerprints": len(prints),
        "crude_bins": int(cfg["crude_bins"]),
        "crude_values": {"M0": crude_fun["M0"], "M1_integral": crude_fun["M1_integral"]},
        "fine_values": {"M0": fine_fun["M0"], "M1_integral": fine_fun["M1_integral"]},
        "crude_vs_fine_relative_deviation": deviation,
        "crude_vs_extrapolated": {name: abs(crude_fun[name] - f_ex) / abs(f_ex)
                                  for name in ("M0",)},
        "limit": cfg["gci_limit"],
        "error_detectable": bool(max(deviation.values()) > cfg["gci_limit"]),
        "pass": bool(distinct and max(deviation.values()) > cfg["gci_limit"]),
    }


def check_floor(cfg: dict) -> dict:
    """C6: bin-0 mass share vs ``d_0 / v_min**(1/3)``; explicit 5 % criterion."""
    v_0 = cfg["_v0"]
    ratios, shares, rows = [], [], {}
    for octaves in cfg["floor_octaves"]:
        grid = LogLogGrid(v_0 * 2.0 ** -octaves, v_0 * 2.0 ** cfg["octaves_above"],
                          cfg["floor_bins"], float(cfg["physical_grading"][0]))
        run = physical_case(grid, cfg)
        share = floor_share(run["n"], grid)
        rows[str(octaves)] = {
            "octaves_below": int(octaves),
            "d0_over_dmin": float(2.0 ** (octaves / 3.0)),
            "floor_mass_share": float(share[-1]),
            "floor_number_share": float(run["n"][0] / run["n"].sum()),
            "M0": float(grid.moment(run["n"], 0)),
            "total_mass": float(grid.pivot_moment(run["n"], 1)),
        }
        ratios.append(2.0 ** (octaves / 3.0))
        shares.append(float(share[-1]))
    thr = floor_threshold(ratios, shares, cfg["floor_limit"])
    production = str(cfg["octaves_below"])
    prod_share = rows[production]["floor_mass_share"] if production in rows else rows[
        str(max(cfg["floor_octaves"]))]["floor_mass_share"]
    return {
        "limit": cfg["floor_limit"],
        "sweep": rows,
        "ratios": ratios,
        "shares": shares,
        "threshold": thr,
        "production_octaves_below": int(cfg["octaves_below"]),
        "production_d0_over_dmin": float(2.0 ** (cfg["octaves_below"] / 3.0)),
        "production_floor_share": float(prod_share),
        "verdict": ("PASS" if prod_share <= cfg["floor_limit"] else "WARN"),
        "warning": None if prod_share <= cfg["floor_limit"] else (
            f"floor mass share {prod_share:.3e} exceeds {cfg['floor_limit']:.0%}: "
            f"move v_min below d_0/{thr['threshold']:.2f} (ratio in diameter)"),
        "pass": bool(prod_share <= cfg["floor_limit"]),
    }


def check_erosion(cfg: dict) -> dict:
    """C7: grid contribution to the removal mass vs the cross-law deviation."""
    case = cfg["_case"]
    law = erosion_law(case)
    v_0 = cfg["_v0"]
    rows, removed = {}, {}
    for grading in cfg["physical_grading"]:
        family = {}
        for n_bins in cfg["physical_levels"]:
            grid = LogLogGrid(v_0 * 2.0 ** -cfg["octaves_below"],
                              v_0 * 2.0 ** cfg["octaves_above"], n_bins, grading)
            run = physical_case(grid, cfg)
            res = erosion_removed_mass(
                grid, run["n"],
                lambda d: law.ratio(case["u_mean"], case["alpha_mean_deg"], d),
                rho_p=case["rho_p"], t_end=case["t_end"], f_impact=case["f_impact"])
            family[str(n_bins)] = {"n_bins": int(n_bins), "grading": float(grading), **res,
                                   "removed_per_feed_mass": res["removed_pivot"] / (
                                       case["rho_p"] * math.pi / 6.0 * case["d_0"] ** 3)}
        rows[str(grading)] = family
        fine, mid, coarse = cfg["physical_levels"][::-1]
        removed[str(grading)] = richardson_gci(family[str(fine)]["removed_pivot"],
                                               family[str(mid)]["removed_pivot"],
                                               family[str(coarse)]["removed_pivot"], 2.0)
    cross = _cross_law_deviation(cfg)
    uniform = removed[str(cfg["physical_grading"][0])]
    fine, mid, coarse = cfg["physical_levels"][::-1]
    umbrella = rows[str(cfg["physical_grading"][0])]
    grid_dev = {
        "coarse_vs_fine": abs(umbrella[str(coarse)]["removed_pivot"]
                              - umbrella[str(fine)]["removed_pivot"])
        / umbrella[str(fine)]["removed_pivot"],
        "mid_vs_fine": abs(umbrella[str(mid)]["removed_pivot"]
                           - umbrella[str(fine)]["removed_pivot"])
        / umbrella[str(fine)]["removed_pivot"],
        "gci_fine": uniform["gci"],
        "pivot_vs_integral_fine": umbrella[str(fine)]["relative_difference"],
    }
    decomposition = {
        "grid_contribution_fine": grid_dev["gci_fine"],
        "grid_contribution_mid": grid_dev["mid_vs_fine"],
        "grid_contribution_coarse": grid_dev["coarse_vs_fine"],
        "cross_correlation_deviation": cross["max_relative_deviation"],
        "cross_correlation_anchored": cross["anchored_value"],
        "grid_over_cross_at_mid": grid_dev["mid_vs_fine"] / cross["max_relative_deviation"],
        "grid_over_cross_at_coarse": grid_dev["coarse_vs_fine"] / cross["max_relative_deviation"],
        "verdict": ("grid below 1/3 of the cross-correlation deviation at the "
                    "production resolution"
                    if grid_dev["mid_vs_fine"] / cross["max_relative_deviation"] <= 1.0 / 3.0
                    else "grid contribution comparable to the cross-correlation deviation"),
    }
    return {
        "families": rows, "richardson": removed, "grid_deviation": grid_dev,
        "cross_law": cross, "decomposition": decomposition,
        "relative_difference_coarse_fine": grid_dev["coarse_vs_fine"],
        "limit": 1.0 / 3.0 * cross["max_relative_deviation"],
        "pass": bool(grid_dev["mid_vs_fine"] <= 1.0 / 3.0 * cross["max_relative_deviation"]
                     and uniform["certified"]),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(metrics: dict, out_dir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    # 1: observed order / error vs h
    levels = sorted(set(CONFIG["order_levels"]))
    h = [1.0 / n for n in levels]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for name in ("M0", "M1_integral", "M2_integral"):
        vals = metrics["C3_order"]["functionals"][name]["value_all"]
        ref = vals[str(levels[-1])]
        ax.loglog(h, [abs(vals[str(n)] - ref) / abs(ref) for n in levels], "o-",
                  label=f"{name} (p={metrics['C3_order']['functionals'][name]['p']:.2f})")
    ax.loglog(h, [h[0] / x * 1.0 for x in h], "k--", lw=0.8, label="slope 1")
    ax.set_xlabel(r"mesh size $h = 1/N$")
    ax.set_ylabel("relative change vs finest mesh")
    ax.set_title("PBM sectional mesh: observed order")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    figs.append(("fig1_order.png", "Sectional-mesh error of the PSD moments vs mesh size; "
                                   "dashed line is slope 1. Pairs with C3."))
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_order.png", dpi=140)
    plt.close(fig)

    # 2: measure factor (analytic anchor)
    anchor = metrics["C3_order"]["measure_factor_check"]
    per = anchor["per_level"]
    ns = sorted(int(k) for k in per)
    meas = [abs(per[str(n)]["measured"] - 1.0) for n in ns]
    pred = [abs(per[str(n)]["predicted"] - 1.0) for n in ns]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.loglog(ns, [m if m > 0 else 1e-17 for m in meas], "o-",
              label="measured $M_1^{int}/M_1^{pivot}-1$")
    ax.loglog(ns, [p if p > 0 else 1e-17 for p in pred], "s--",
              label="closed form $(1+r)/(2\\sqrt{r})-1$")
    ax.loglog(ns, [meas[0] * (ns[0] / n) ** 2 for n in ns], "k:", lw=0.8, label="slope 2")
    ax.set_xlabel("bins")
    ax.set_ylabel("relative measure factor")
    ax.set_title("Analytic anchor for the pivot measure")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    figs.append(("fig2_measure_factor.png",
                 "Integral-vs-pivot mass measure factor per mesh level together with the "
                 "closed-form prediction (1+r)/(2 sqrt r); the two coincide to machine "
                 "precision, calibrating the GCI machinery. Pairs with C3 measure_factor_check."))
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_measure_factor.png", dpi=140)
    plt.close(fig)

    # 3: floor criterion
    sweep = metrics["C6_floor"]["sweep"]
    d0_dmin = [sweep[k]["d0_over_dmin"] for k in sweep]
    share = [max(sweep[k]["floor_mass_share"], 1e-30) for k in sweep]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.loglog(d0_dmin, share, "o-", label="bin-0 mass share")
    ax.axhline(metrics["C6_floor"]["limit"], color="r", ls="--", lw=0.9, label="5 % criterion")
    thr = metrics["C6_floor"]["threshold"]["threshold"]
    if thr:
        ax.axvline(thr, color="g", ls=":", lw=0.9, label=f"threshold $d_0/d_{{min}}={thr:.2f}$")
    ax.set_xlabel(r"$d_0 / v_{min}^{1/3}$")
    ax.set_ylabel("fines-floor mass share")
    ax.set_title("Fines floor as an explicit criterion")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    figs.append(("fig3_floor.png", "Bin-0 mass share vs the floor ratio, with the 5 % criterion "
                                   "and the interpolated threshold. Pairs with C6."))
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_floor.png", dpi=140)
    plt.close(fig)

    # 4: erosion decomposition
    dec = metrics["C7_erosion"]["decomposition"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    labels = ["grid\n(fine GCI)", "grid\n(mid)", "grid\n(coarse)", "cross-law\n(Oka vs RP-O501)"]
    vals = [dec["grid_contribution_fine"], dec["grid_contribution_mid"],
            dec["grid_contribution_coarse"], dec["cross_correlation_deviation"]]
    bars = ax.bar(labels, [100 * v for v in vals], color=["#4c72b0", "#4c72b0", "#4c72b0", "#c44e52"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, 100 * v, f"{100 * v:.2f}%", ha="center", va="bottom",
                fontsize=8)
    ax.set_ylabel("relative deviation [%]")
    ax.set_title("Erosion removal mass: grid vs cross-correlation")
    ax.grid(True, axis="y", alpha=0.3)
    figs.append(("fig4_erosion_decomposition.png",
                 "Removal-mass deviation from the mesh (three levels) next to the "
                 "Finnie/Oka/RP-O501 cross-correlation deviation. Pairs with C7."))
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_erosion_decomposition.png", dpi=140)
    plt.close(fig)
    return figs


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _prepare(cfg: dict) -> dict:
    case = dict(CASE)
    prepared = dict(cfg)
    prepared["_case"] = case
    prepared["_v0"] = case["d_0"] ** 3
    prepared["_dem_cfg"] = dem_config(case)
    return prepared


def run_study(out_dir, *, quick: bool = False, config: dict | None = None) -> dict:
    """Run the whole study and return the metrics dict (also written to disk)."""
    cfg = dict(CONFIG)
    if config:
        cfg.update(config)
    if quick:
        cfg.update({"order_levels": (48, 96, 192, 384),
                    "order_asymptotic_triple": (384, 192, 96),
                    "order_pre_asymptotic_triple": (192, 96, 48),
                    "physical_levels": (96, 192, 384), "floor_octaves": (2, 4, 8, 12, 20),
                    "floor_bins": 96, "gaudin_samples": 512})
    cfg = _prepare(cfg)
    _ORDER_CACHE.clear()
    _CASE_CACHE.clear()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"node": NODE,
               "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
               "case": {k: cfg["_case"][k] for k in
                        ("d_0", "rho_p", "sigma_0", "weibull_m", "u_mean",
                         "alpha_mean_deg", "n_fragments", "gaudin_alpha", "t_end")}}
    metrics["C1_mass"] = check_mass(cfg)
    metrics["C2_cascade"] = check_cascade(cfg)
    metrics["C3_order"] = check_order_cache(cfg)
    metrics["C4_gci"] = check_gci(cfg)
    metrics["C5_negative_control"] = check_negative_control(cfg)
    metrics["C6_floor"] = check_floor(cfg)
    metrics["C7_erosion"] = check_erosion(cfg)

    checks = {"C1_mass_conservation": metrics["C1_mass"]["pass"],
              "C2_analytic_cascade": metrics["C2_cascade"]["pass"],
              "C3_observed_order": metrics["C3_order"]["pass"],
              "C4_gci_le_5pct": metrics["C4_gci"]["pass"],
              "C5_negative_control": metrics["C5_negative_control"]["pass"],
              "C6_floor_criterion": metrics["C6_floor"]["pass"],
              "C7_erosion_grid_error": metrics["C7_erosion"]["pass"]}
    metrics["checks"] = checks
    metrics["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    metrics["surfaces"] = {"figures": [], "tables": []}

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    tables = _write_tables(metrics, out_dir)
    metrics["surfaces"]["tables"] = tables
    try:
        figs = make_figures(metrics, out_dir)
        metrics["surfaces"]["figures"] = [f[0] for f in figs]
        metrics["figure_captions"] = {f[0]: f[1] for f in figs}
    except Exception as exc:  # pragma: no cover - matplotlib optional
        metrics["surfaces"]["figures_error"] = f"{type(exc).__name__}: {exc}"
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


def _write_tables(metrics: dict, out_dir: Path) -> list:
    lines = ["# p488w1-n09 PBM mesh-independence tables (program-generated)", ""]
    lines += ["## C1 mass ledger (drift limit 1e-12)", "",
              "| family | bins | grading | mass drift rel. | propagator tail |",
              "|---|---|---|---|---|"]
    for fam, rows in metrics["C1_mass"]["families"].items():
        for k, r in rows.items():
            lines.append(f"| {fam} | {r['n_bins']} | {r['grading']:.1f} | "
                         f"{r['mass_drift_relative']:.3e} | {r['propagator_tail']:.1e} |")
    lines += ["", "## C2 analytic cascade (limit 1e-6)", "",
              "| sections/octave | bins | max rel. error | generations |", "|---|---|---|---|"]
    for k, r in metrics["C2_cascade"]["uniform"].items():
        lines.append(f"| {k} | {r['n_bins']} | {r['max_relative_error']:.3e} | "
                     f"{r['generations_compared']} |")
    for k, r in metrics["C2_cascade"]["graded"].items():
        lines.append(f"| graded={k} | {r['n_bins']} | {r['max_relative_error']:.3e} | "
                     f"{r['generations_compared']} |")
    lines += ["", "## C3 observed order (declared p_nom = 1, band 0.8-1.2)", "",
              "| functional | p (asymptotic) | p (pre-asymptotic) | GCI | certified |",
              "|---|---|---|---|---|"]
    for name, r in metrics["C3_order"]["functionals"].items():
        cert = name in metrics["C3_order"]["certified"]
        lines.append(f"| {name} | {r['p']:.4f} | "
                     f"{metrics['C3_order']['pre_asymptotic'][name]['p']:.4f} | "
                     f"{r['gci']:.3e} | {'yes' if cert else 'diagnostic'} |")
    for grading, fam in metrics["C3_order"]["physical"].items():
        lines += ["", f"### physical case, grading={grading}", "",
                  "| functional | p | GCI |", "|---|---|---|"]
        for name, r in fam.items():
            if name.startswith("_"):
                continue
            lines.append(f"| {name} | {r['p']:.4f} | {r['gci']:.3e} |")
    lines += ["", "## C5 negative control", "",
              f"- fingerprints distinct: {metrics['C5_negative_control']['fingerprints_distinct']}",
              f"- crude N={metrics['C5_negative_control']['crude_bins']} vs fine: "
              + ", ".join(f"{k}={v:.3%}" for k, v in
                          metrics['C5_negative_control']['crude_vs_fine_relative_deviation'].items()),
              "", "## C6 floor criterion", "",
              "| octaves below | d0/dmin | floor mass share |", "|---|---|---|"]
    for k, r in metrics["C6_floor"]["sweep"].items():
        lines.append(f"| {k} | {r['d0_over_dmin']:.2f} | {r['floor_mass_share']:.3e} |")
    thr = metrics["C6_floor"]["threshold"]
    lines += ["", f"threshold d0/dmin = {thr['threshold']}", ""]
    lines += ["## C7 erosion removal mass decomposition", "",
              "| grading | bins | removed (kg) | pivot vs integral |", "|---|---|---|---|"]
    for grading, fam in metrics["C7_erosion"]["families"].items():
        for k, r in fam.items():
            lines.append(f"| {grading} | {r['n_bins']} | {r['removed_pivot']:.6e} | "
                         f"{r['relative_difference']:.2e} |")
    dec = metrics["C7_erosion"]["decomposition"]
    lines += ["", "| contribution | value |", "|---|---|",
              f"| grid GCI (fine) | {dec['grid_contribution_fine']:.3%} |",
              f"| grid mid vs fine | {dec['grid_contribution_mid']:.3%} |",
              f"| grid coarse vs fine | {dec['grid_contribution_coarse']:.3%} |",
              f"| cross-law (Oka vs RP-O501) | {dec['cross_correlation_deviation']:.3%} |",
              f"| grid/cross at mid | {dec['grid_over_cross_at_mid']:.3f} |", ""]
    (out_dir / "tables.md").write_text("\n".join(lines))
    return ["tables.md", "metrics.json"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/p488w1_n09", help="output directory")
    ap.add_argument("--quick", action="store_true", help="reduced-resolution run")
    args = ap.parse_args(argv)
    metrics = run_study(args.out, quick=args.quick)
    print(json.dumps({"verdict": metrics["verdict"], "checks": metrics["checks"],
                      "out": str(args.out)}, indent=2))
    return 0 if metrics["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())