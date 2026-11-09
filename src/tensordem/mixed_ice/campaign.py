"""Campaign: MY-fraction sweep, FSD sensitivity, DEM strength factorial.

What the campaign computes
-------------------------
1. **Ageing table** -- FY (1 a) through MY (2-5 a, 30 a asymptote): salinity,
   temperature, brine volume, thickness, flexural/compressive strength, modulus,
   toughness, density.
2. **MY-fraction sweep** ``p = 0, 0.1, 0.2, 0.4, 0.6, 0.8`` for both engineering
   anchors (Lindqvist, Riska) at 1, 3 and 5 m/s, plus a strength-parameterisation
   band (MY strength x0.8 .. x1.2).  Answer to "how much does R change from 0 %
   to 80 % MY".
3. **FSD sensitivity** -- one-at-a-time sweep over the four thickness
   distribution parameters plus one material parameter, ranked by span, plus a
   joint Latin-hypercube envelope over the FSD group.
4. **Estimator cross-check** -- deterministic quadrature against a 20 000-floe
   Monte-Carlo sample of the same field (value and realised area fraction).
5. **DEM 2^3 factorial** -- the bonded-particle section model of
   :mod:`tensordem.level_ice` is run at the eight combinations of the FY/MY
   flexural strength, compressive strength and elastic modulus, at ``h = 1 m``.
   A log-linear fit gives the DEM's strength-sensitivity exponents, which are
   compared with the *same* exponents fitted on the engineering anchor.  A
   thickness pair (``h = 1.0`` and ``1.5 m``) gives the DEM thickness exponent,
   and a ``n_z = 6`` pair checks whether the flexure route is resolution-limited.

Every acceptance criterion is declared in :func:`verdicts` with its measured
value and a PASS/FAIL label; a FAIL is never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import aging
from .field import (FloeDistribution, MixedFieldSpec, my_distribution,
                    quadrature_field, sample_field)
from .resistance import (SENSITIVITY_PARAMETERS, class_ratio_decomposition,
                         mixed_resistance, quadrature_vs_sampler,
                         resistance_sensitivity)

__all__ = [
    "MY_FRACTIONS",
    "SPEEDS_M_S",
    "ageing_table",
    "my_fraction_sweep",
    "run_dem_factorial",
    "fit_exponents",
    "verdicts",
    "run_campaign",
    "save_artifact",
]

#: MY area fractions of the primary sweep (the node asks for 0-80 %).
MY_FRACTIONS = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8)
#: Speeds of the sweep [m/s].
SPEEDS_M_S = (1.0, 3.0, 5.0)
#: Primary speed of the reported curve [m/s].
PRIMARY_SPEED_M_S = 3.0
#: MY strength multiplier band (ageing-parameterisation uncertainty).
MY_STRENGTH_BAND = (0.8, 1.0, 1.2)
#: DEM case settings of this campaign (match the level-ice campaign).
DEM_SETTINGS = dict(hull="icebreaker", thickness_m=1.0, speed_m_s=3.0, layers=4,
                    length_m=24.0, advance_m=6.0)
#: DEM resolution check (n_z = 6, same physical set-up).
DEM_RESOLUTION_SETTINGS = dict(DEM_SETTINGS, layers=6)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _design_matrix(levels: dict[str, Sequence[float]]) -> tuple[np.ndarray, list[tuple[str, float]]]:
    """2^k factorial design matrix in log-relative coordinates.

    Returns the matrix ``[1, x_1, ..., x_k]`` and the level list, where
    ``x_i = ln(level_i / low_i)`` is the relative perturbation of factor ``i``.
    """
    keys = list(levels)
    rows: list[list[float]] = []
    for code in range(2 ** len(keys)):
        row = [1.0]
        for j, key in enumerate(keys):
            low, high = levels[key]
            row.append(0.0 if (code >> j) & 1 == 0 else math.log(high / low))
        rows.append(row)
    return np.array(rows, dtype=float), [(k, math.log(levels[k][1] / levels[k][0])) for k in keys]


def fit_exponents(resistances: np.ndarray, levels: dict[str, Sequence[float]]) -> dict[str, Any]:
    """Log-linear least squares fit of ``ln R`` on the factor perturbations.

    With a 2^k design the four fitted parameters (intercept + k exponents) leave
    ``2^k - k - 1`` degrees of freedom, and the residual is the interaction
    magnitude -- i.e. how far the response is from being separable in the
    factors.
    """
    x, scales = _design_matrix(levels)
    y = np.log(np.asarray(resistances, dtype=float))
    if y.size != x.shape[0]:
        raise ValueError("one resistance per design point is required")
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    residual = y - pred
    return {
        "intercept": float(coef[0]),
        "reference_resistance_N": float(math.exp(coef[0])),
        "exponents": {name: float(coef[i + 1]) for i, (name, _s) in enumerate(scales)},
        "levels": {k: list(map(float, v)) for k, v in levels.items()},
        "log_perturbations": {name: float(s) for name, s in scales},
        "residual_rms_log": float(np.sqrt(np.mean(residual ** 2))),
        "residual_max_abs_log": float(np.max(np.abs(residual))),
        "dof": int(y.size - x.shape[1]),
    }


def ageing_table(ages: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 30.0)) -> list[dict[str, Any]]:
    """Material state of every age class of the ageing law."""
    return [aging.material_state(float(a)).as_dict() for a in ages]


# ---------------------------------------------------------------------------
# 1. MY-fraction sweep
# ---------------------------------------------------------------------------
def my_fraction_sweep(*, speeds: Sequence[float] = SPEEDS_M_S,
                      fractions: Sequence[float] = MY_FRACTIONS,
                      models: Sequence[str] = ("lindqvist", "riska"),
                      multipliers: Sequence[float] = MY_STRENGTH_BAND,
                      n_age: int = 8, n_thickness: int = 128) -> dict[str, Any]:
    """R(MY fraction) for every anchor, speed and strength band level."""
    from tensordem.level_ice.formula import ICEBREAKER

    out: dict[str, Any] = {"hull": ICEBREAKER.as_dict(), "curves": {}}
    for model in models:
        for speed in speeds:
            rows = []
            for p in fractions:
                spec = MixedFieldSpec(my_fraction=float(p))
                for mult in multipliers:
                    res = mixed_resistance(spec, hull=ICEBREAKER, speed_m_s=float(speed),
                                           model=model, n_age=n_age,
                                           n_thickness=n_thickness,
                                           my_strength_multiplier=float(mult))
                    row = res.as_dict()
                    row["my_strength_multiplier"] = float(mult)
                    rows.append(row)
            out["curves"][f"{model}_v{speed:g}"] = {
                "model": model, "speed_m_s": float(speed), "rows": rows,
            }
    # endpoints of the primary curve
    primary = [r for r in out["curves"][f"lindqvist_v{PRIMARY_SPEED_M_S:g}"]["rows"]
               if r["my_strength_multiplier"] == 1.0]
    base = [r for r in primary if r["my_fraction"] == 0.0][0]
    end = [r for r in primary if r["my_fraction"] == max(fractions)][0]
    out["primary_endpoints"] = {
        "model": "lindqvist",
        "speed_m_s": PRIMARY_SPEED_M_S,
        "my_fraction_start": base["my_fraction"],
        "my_fraction_end": end["my_fraction"],
        "resistance_start_N": base["resistance_mean_N"],
        "resistance_end_N": end["resistance_mean_N"],
        "relative_change": end["resistance_mean_N"] / base["resistance_mean_N"] - 1.0,
        "class_ratio_start": base["class_ratio"],
        "class_ratio_end": end["class_ratio"],
        "linear_slope_N_per_unit_my": (end["resistance_mean_N"] - base["resistance_mean_N"])
        / (end["my_fraction"] - base["my_fraction"]),
    }
    return out


# ---------------------------------------------------------------------------
# 2. DEM factorial
# ---------------------------------------------------------------------------
def _dem_spec(state: aging.IceAgeState, *, layers: int = 4,
              thickness_m: float = 1.0) -> Any:
    from tensordem.level_ice.campaign import CaseSpec

    return CaseSpec(
        hull=DEM_SETTINGS["hull"], thickness_m=float(thickness_m),
        speed_m_s=DEM_SETTINGS["speed_m_s"], layers=int(layers),
        length_m=DEM_SETTINGS["length_m"], advance_m=DEM_SETTINGS["advance_m"],
        flexural_strength_pa=float(state.flexural_strength_pa),
        compressive_strength_pa=float(state.compressive_strength_pa),
        elastic_modulus_pa=float(state.elastic_modulus_pa),
        label=f"{state.ice_class}-h{thickness_m:g}-n{layers}",
    )


def run_dem_factorial(*, workers: int | None = None,
                      resolution_check: bool = True,
                      my_age_years: float = 3.0) -> dict[str, Any]:
    """Run the DEM cases: 2^3 strength/modulus factorial plus thickness pairs."""
    from tensordem.level_ice.campaign import run_many

    fy = aging.first_year_state()
    my = aging.material_state(my_age_years)
    levels = {
        "flexural_strength_pa": (fy.flexural_strength_pa, my.flexural_strength_pa),
        "compressive_strength_pa": (fy.compressive_strength_pa, my.compressive_strength_pa),
        "elastic_modulus_pa": (fy.elastic_modulus_pa, my.elastic_modulus_pa),
    }
    keys = list(levels)
    specs = []
    codes = []
    for code in range(8):
        values = {k: levels[k][(code >> j) & 1] for j, k in enumerate(keys)}
        # FY state with the factorial levels substituted for the three factors
        state = replace(fy, **values)
        specs.append(_dem_spec(state))
        codes.append(code)

    extras: list[Any] = []
    if resolution_check:
        extras.append(_dem_spec(fy, layers=6))
        extras.append(_dem_spec(my, layers=6))
    extras.append(_dem_spec(fy, thickness_m=1.5))
    extras.append(_dem_spec(my, thickness_m=1.5))

    started = time.time()
    results = run_many(list(specs) + extras, workers=workers)
    factorial = results[: len(specs)]
    extra_results = results[len(specs):]

    # rename for readability in the artifact
    for code, row in zip(codes, factorial):
        row["factorial_code"] = code
        row["factor_levels"] = {
            "flexural_strength_pa": levels["flexural_strength_pa"][code & 1],
            "compressive_strength_pa": levels["compressive_strength_pa"][(code >> 1) & 1],
            "elastic_modulus_pa": levels["elastic_modulus_pa"][(code >> 2) & 1],
        }
    resistances = np.array([r["mean_total_n_per_m"] for r in factorial], dtype=float)
    dem_fit = fit_exponents(resistances, levels)

    # the engineering anchor fitted on exactly the same 2^3 design
    anchor_rows = []
    from tensordem.level_ice.formula import ICEBREAKER, IceState, lindqvist_components, riska_components

    for code in range(8):
        values = {k: levels[k][(code >> j) & 1] for j, k in enumerate(keys)}
        state = IceState(thickness_m=1.0, **values)
        anchor_rows.append({
            "code": code,
            "lindqvist_N": lindqvist_components(state, ICEBREAKER, DEM_SETTINGS["speed_m_s"])["total_N"],
            "riska_N": riska_components(state, ICEBREAKER, DEM_SETTINGS["speed_m_s"])["total_N"],
        })
    anchor_fit = {
        "lindqvist": fit_exponents(np.array([r["lindqvist_N"] for r in anchor_rows]), levels),
        "riska": fit_exponents(np.array([r["riska_N"] for r in anchor_rows]), levels),
    }

    # the anchor's strength-carrying share at the FY state: the section model has
    # no submerged-rubble (submersion) term and almost no added drag, so its
    # strength exponents are expected to be ~1/share times the anchor's.
    base_state = IceState(thickness_m=1.0)
    base_components = lindqvist_components(base_state, ICEBREAKER, DEM_SETTINGS["speed_m_s"])
    strength_share = float(
        (base_components["crushing_N"] + base_components["bending_N"])
        / base_components["total_N"])
    diagnosis: dict[str, Any] = {"anchor_strength_carrying_share": strength_share,
                                 "anchor_component_shares": {
                                     "crushing": base_components["crushing_N"] / base_components["total_N"],
                                     "bending": base_components["bending_N"] / base_components["total_N"],
                                     "submersion": base_components["submersion_N"] / base_components["total_N"],
                                     "added_drag": base_components["added_drag_N"] / base_components["total_N"],
                                 }}
    for key in keys:
        dem_exp = dem_fit["exponents"][key]
        anchor_exp = anchor_fit["lindqvist"]["exponents"][key]
        diagnosis[key] = {
            "dem_exponent": dem_exp,
            "anchor_exponent": anchor_exp,
            "dem_exponent_normalised": dem_exp * strength_share,
            "deviation_raw": (dem_exp - anchor_exp) / anchor_exp if anchor_exp else None,
            "deviation_normalised": ((dem_exp * strength_share - anchor_exp) / anchor_exp
                                     if anchor_exp else None),
        }

    # DEM-domain MY-fraction sweep at fixed thickness: the field's FSD is frozen at
    # h = 1 m and only the ice class (material state) of each floe switches, so this
    # is the DEM analogue of the anchor sweep -- a *section* quantity (N/m), not a
    # ship-scale resistance, and reported for its shape only.
    dem_sweep = {"thickness_m": 1.0, "rows": []}
    for p in MY_FRACTIONS:
        r_dem = (1.0 - p) * resistances[0] + p * resistances[7]
        r_lind = (1.0 - p) * anchor_rows[0]["lindqvist_N"] + p * anchor_rows[7]["lindqvist_N"]
        r_riska = (1.0 - p) * anchor_rows[0]["riska_N"] + p * anchor_rows[7]["riska_N"]
        dem_sweep["rows"].append({
            "my_fraction": float(p),
            "dem_section_N_per_m": float(r_dem),
            "dem_relative_increase": float(r_dem / resistances[0] - 1.0),
            "lindqvist_N": float(r_lind),
            "lindqvist_relative_increase": float(r_lind / anchor_rows[0]["lindqvist_N"] - 1.0),
            "riska_N": float(r_riska),
            "riska_relative_increase": float(r_riska / anchor_rows[0]["riska_N"] - 1.0),
        })
    last = dem_sweep["rows"][-1]
    dem_sweep["at_max_fraction"] = {
        "my_fraction": last["my_fraction"],
        "dem_relative_increase": last["dem_relative_increase"],
        "lindqvist_relative_increase": last["lindqvist_relative_increase"],
        "riska_relative_increase": last["riska_relative_increase"],
        "dem_vs_lindqvist_relative": (
            (last["dem_relative_increase"] - last["lindqvist_relative_increase"])
            / last["lindqvist_relative_increase"]),
    }

    dem_ratio = float(resistances[7] / resistances[0])
    anchor_ratio_ln = float(anchor_rows[7]["lindqvist_N"] / anchor_rows[0]["lindqvist_N"])
    anchor_ratio_rk = float(anchor_rows[7]["riska_N"] / anchor_rows[0]["riska_N"])

    thickness: dict[str, Any] = {}
    if resolution_check:
        fine_fy, fine_my = extra_results[0], extra_results[1]
        thick_fy, thick_my = extra_results[2], extra_results[3]
        thickness = {"fine_layers_fy": fine_fy, "fine_layers_my": fine_my,
                     "h15_fy": thick_fy, "h15_my": thick_my}
        exps = {}
        for label, base, high in (("FY", factorial[0], thick_fy), ("MY", factorial[7], thick_my)):
            exps[label] = math.log(high["mean_total_n_per_m"] / base["mean_total_n_per_m"]) / math.log(1.5)
        thickness["thickness_exponent"] = exps
        thickness["thickness_exponent_anchor"] = {
            "lindqvist": math.log(
                lindqvist_components(IceState(thickness_m=1.5), ICEBREAKER, DEM_SETTINGS["speed_m_s"])["total_N"]
                / lindqvist_components(IceState(thickness_m=1.0), ICEBREAKER, DEM_SETTINGS["speed_m_s"])["total_N"]
            ) / math.log(1.5),
        }
        thickness["flexure_bucket_fraction"] = {
            "fy_coarse": factorial[0]["flexure_fraction"],
            "my_coarse": factorial[7]["flexure_fraction"],
            "fy_fine": fine_fy["flexure_fraction"],
            "my_fine": fine_my["flexure_fraction"],
        }
        thickness["resolution_ratio"] = {
            "fy": fine_fy["mean_total_n_per_m"] / factorial[0]["mean_total_n_per_m"],
            "my": fine_my["mean_total_n_per_m"] / factorial[7]["mean_total_n_per_m"],
        }
    else:
        thick_fy = thick_my = None

    return {
        "settings": dict(DEM_SETTINGS),
        "resolution_settings": dict(DEM_RESOLUTION_SETTINGS) if resolution_check else None,
        "fy_state": fy.as_dict(),
        "my_state": my.as_dict(),
        "my_age_years": float(my_age_years),
        "factorial": factorial,
        "dem_fit": dem_fit,
        "anchor_rows": anchor_rows,
        "anchor_fit": anchor_fit,
        "scaling_diagnosis": diagnosis,
        "my_fraction_sweep": dem_sweep,
        "class_ratio": {
            "dem_full_high_over_low": dem_ratio,
            "lindqvist_full_high_over_low": anchor_ratio_ln,
            "riska_full_high_over_low": anchor_ratio_rk,
            "dem_vs_lindqvist_relative": (dem_ratio - anchor_ratio_ln) / (anchor_ratio_ln - 1.0),
            "dem_vs_riska_relative": (dem_ratio - anchor_ratio_rk) / (anchor_ratio_rk - 1.0),
        },
        "thickness": thickness,
        "wall_clock_s": time.time() - started,
    }


# ---------------------------------------------------------------------------
# 3. joint FSD / material envelope
# ---------------------------------------------------------------------------
def joint_envelope(*, base_my_fraction: float = 0.4, n_fsd: int = 64,
                   n_material: int = 16, seed: int = 20261108,
                   speed_m_s: float = PRIMARY_SPEED_M_S,
                   model: str = "lindqvist") -> dict[str, Any]:
    """Latin-hypercube joint envelope over the FSD and material groups."""
    from tensordem.level_ice.formula import ICEBREAKER

    rng = np.random.default_rng(seed)
    params = SENSITIVITY_PARAMETERS

    def draw(names: Sequence[str], n: int) -> list[dict[str, float]]:
        cols = []
        for name in names:
            levels = params[name][1]
            lo, hi = float(min(levels)), float(max(levels))
            strata = (np.arange(n) + rng.random(n)) / n
            cols.append(lo + (hi - lo) * rng.permutation(strata))
        return [dict(zip(names, [float(c[i]) for c in cols])) for i in range(n)]

    fsd_names = [n for n, (g, _l) in params.items() if g == "fsd"]
    mat_names = [n for n, (g, _l) in params.items() if g == "material"]
    out: dict[str, Any] = {"base_my_fraction": base_my_fraction, "seed": seed,
                           "fsd_parameters": fsd_names, "material_parameters": mat_names,
                           "model": model, "speed_m_s": speed_m_s}
    for label, names, n in (("fsd", fsd_names, n_fsd), ("material", mat_names, n_material)):
        values = []
        for combo in draw(names, n):
            spec = MixedFieldSpec(my_fraction=base_my_fraction, seed=seed)
            mult = 1.0
            for name, level in combo.items():
                if name == "fy_thickness_mean_m":
                    spec = replace(spec, fy=replace(spec.fy, thickness_mean_m=level))
                elif name == "fy_thickness_cv":
                    spec = replace(spec, fy=replace(spec.fy, thickness_cv=level))
                elif name == "my_thickness_scale":
                    spec = replace(spec, my=replace(spec.my_dist,
                                                    thickness_mean_m=spec.my_dist.thickness_mean_m * level))
                elif name == "my_thickness_cv":
                    spec = replace(spec, my=replace(spec.my_dist, thickness_cv=level))
                elif name == "my_strength_multiplier":
                    mult = level
                else:  # pragma: no cover - guarded by SENSITIVITY_PARAMETERS
                    raise ValueError(f"unhandled parameter {name}")
            res = mixed_resistance(spec, hull=ICEBREAKER, speed_m_s=speed_m_s,
                                   model=model, n_age=6, n_thickness=64,
                                   my_strength_multiplier=mult)
            values.append({"combo": combo, "resistance_N": res.resistance_mean_n})
        arr = np.array([v["resistance_N"] for v in values])
        out[label] = {
            "n": int(arr.size),
            "mean_N": float(arr.mean()),
            "std_N": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min_N": float(arr.min()), "max_N": float(arr.max()),
            "p05_N": float(np.percentile(arr, 5)), "p95_N": float(np.percentile(arr, 95)),
            "relative_span": float((arr.max() - arr.min()) / arr.mean()),
            "samples": values,
        }
    return out


# ---------------------------------------------------------------------------
# 4. verdicts
# ---------------------------------------------------------------------------
def verdicts(artifact: dict[str, Any]) -> dict[str, Any]:
    """Machine-checkable acceptance criteria, each with its measured value.

    Criteria (d), (e), (f) and (k) need the DEM factorial; when the campaign is run
    with ``--no-dem`` they are simply absent instead of silently passing.
    """
    sweep = artifact["my_fraction_sweep"]
    sens = artifact["sensitivity"]
    env = artifact["joint_envelope"]
    check = artifact["estimator_check"]
    dem = artifact.get("dem", {})
    out: dict[str, Any] = {}

    def add(key: str, criterion: str, measured: Any, passed: bool,
            extra: dict[str, Any] | None = None) -> None:
        row = {"criterion": criterion, "measured": measured,
               "verdict": "PASS" if passed else "FAIL"}
        if extra:
            row.update(extra)
        out[key] = row

    # (a) MY-fraction sweep is monotone for both anchors at every speed
    monotone = {}
    for key, curve in sweep["curves"].items():
        rows = [r for r in curve["rows"] if r["my_strength_multiplier"] == 1.0]
        values = [r["resistance_mean_N"] for r in rows]
        monotone[key] = all(b > a for a, b in zip(values, values[1:]))
    add("a_my_fraction_sweep_monotone",
        "R(p_MY) strictly increasing over p = 0..0.8 for both anchors at v = 1/3/5 m/s",
        {"monotone": monotone,
         "relative_change_0_to_80pct": sweep["primary_endpoints"]["relative_change"]},
        all(monotone.values()))

    # (b) the mixture is exactly class-additive
    identity_error = 0.0
    for _key, curve in sweep["curves"].items():
        for row in curve["rows"]:
            if row["class_ratio"] is None:
                continue
            predicted = row["my_fraction"] * (row["class_ratio"] - 1.0)
            identity_error = max(identity_error, abs(predicted - row["relative_increase"]))
    add("b_class_weight_identity",
        "|dR/R(0) - p (R_MY/R_FY - 1)| < 1e-9 (the mixture is exactly class-additive)",
        {"max_abs_error": identity_error}, identity_error < 1e-9)

    # (c) quadrature against the Monte-Carlo sampler's own standard error
    add("c_estimator_agreement",
        "|z| < 3 for R_sampler vs R_quadrature (z = difference / Monte-Carlo standard "
        "error of the weighted mean) and |realised MY area fraction - p| < 0.01",
        check, abs(check["z_score"]) < 3.0
        and abs(check["my_fraction_bias"]) < 0.01)

    # (d)-(f), (k): DEM versus anchor
    if "dem_fit" in dem:
        fit = dem["dem_fit"]["exponents"]
        anchor_fit = dem["anchor_fit"]["lindqvist"]["exponents"]
        rel_c = abs(fit["compressive_strength_pa"] - anchor_fit["compressive_strength_pa"]) \
            / abs(anchor_fit["compressive_strength_pa"])
        add("d_dem_sigma_c_exponent",
            "|b_sigmac(DEM) - b_sigmac(anchor)| / b_sigmac(anchor) < 15 %",
            {"dem": fit["compressive_strength_pa"],
             "anchor": anchor_fit["compressive_strength_pa"], "relative_error": rel_c},
            rel_c < 0.15)
        rel_f = abs(fit["flexural_strength_pa"] - anchor_fit["flexural_strength_pa"]) \
            / abs(anchor_fit["flexural_strength_pa"])
        add("e_dem_sigma_f_exponent",
            "|b_sigmaf(DEM) - b_sigmaf(anchor)| / b_sigmaf(anchor) < 15 %",
            {"dem": fit["flexural_strength_pa"],
             "anchor": anchor_fit["flexural_strength_pa"], "relative_error": rel_f,
             "dem_flexure_bucket_fraction": (dem["thickness"]["flexure_bucket_fraction"]
                                             if dem.get("thickness") else None)},
            rel_f < 0.15)
        ratio = dem["class_ratio"]
        add("f_dem_class_ratio_vs_anchor",
            "|R_MY/R_FY (DEM) - R_MY/R_FY (Lindqvist)| / (R_MY/R_FY - 1) < 15 % "
            "(deviation measured against the increment, the quantity the 0-80 % curve uses)",
            {"dem": ratio["dem_full_high_over_low"],
             "lindqvist": ratio["lindqvist_full_high_over_low"],
             "riska": ratio["riska_full_high_over_low"],
             "relative_deviation_vs_lindqvist": ratio["dem_vs_lindqvist_relative"]},
            abs(ratio["dem_vs_lindqvist_relative"]) < 0.15)
        if dem.get("thickness"):
            frac = dem["thickness"]["flexure_bucket_fraction"]
            add("k_dem_flexure_route_open",
                "the DEM flexure bucket must carry > 1 % of the resistance for the "
                "sigma_f exponent to be meaningful",
                frac, max([v for v in frac.values() if v is not None], default=0.0) > 0.01)

    # (l) the mechanism-normalised sigma_c exponent
    if "scaling_diagnosis" in dem:
        diag = dem["scaling_diagnosis"]["compressive_strength_pa"]
        add("l_dem_sigma_c_exponent_normalised",
            "|b_sigmac(DEM) * s - b_sigmac(anchor)| / b_sigmac(anchor) < 25 %, where "
            "s is the anchor's strength-carrying share: the bonded-particle section "
            "has no submersion (displaced-rubble) term, so its strength exponent is "
            "expected to be 1/s times the anchor's",
            {"strength_carrying_share": dem["scaling_diagnosis"]["anchor_strength_carrying_share"],
             "dem_exponent": diag["dem_exponent"],
             "dem_exponent_normalised": diag["dem_exponent_normalised"],
             "anchor_exponent": diag["anchor_exponent"],
             "relative_error_normalised": abs(diag["deviation_normalised"])},
            abs(diag["deviation_normalised"]) < 0.25)

    # (g) the two anchors agree on the 0-80 % increment
    def endpoint(model: str) -> float:
        rows = [r for r in sweep["curves"][f"{model}_v{PRIMARY_SPEED_M_S:g}"]["rows"]
                if r["my_strength_multiplier"] == 1.0]
        return max(rows, key=lambda r: r["my_fraction"])["relative_increase"]

    dl, dr = endpoint("lindqvist"), endpoint("riska")
    add("g_anchor_cross_model",
        "|dR_Lindqvist - dR_Riska| / dR_Lindqvist < 40 % at p = 0.8, v = 3 m/s "
        "(band justified a priori: the anchors' own MAPE against the MT Uikku tests "
        "are 27.9 % and 51.2 %, so 40 % on a relative increment is the resolution of "
        "the comparison)",
        {"lindqvist": dl, "riska": dr, "relative_difference": abs(dl - dr) / abs(dl)},
        abs(dl - dr) / abs(dl) < 0.40)

    # (h) FSD sensitivity reported and ranked
    add("h_fsd_sensitivity_reported",
        "all four FSD parameters swept, ranked by span, joint LHS envelope reported",
        {"rank_by_span": sens["rank_by_span"],
         "fsd_additive_bound_N": sens["fsd_additive_bound_N"],
         "material_additive_bound_N": sens["material_additive_bound_N"],
         "fsd_share_of_bound": sens["fsd_share_of_bound"],
         "joint_fsd_relative_span": env["fsd"]["relative_span"],
         "joint_material_relative_span": env["material"]["relative_span"]},
        len([n for n in sens["parameters"] if sens["parameters"][n]["group"] == "fsd"]) == 4
        and env["fsd"]["n"] > 0)

    # (i) ageing law limits and monotonicity
    table = artifact["ageing_table"]
    mono = all(
        b["flexural_strength_pa"] >= a["flexural_strength_pa"]
        and b["compressive_strength_pa"] >= a["compressive_strength_pa"]
        and b["elastic_modulus_pa"] >= a["elastic_modulus_pa"]
        and b["thickness_m"] >= a["thickness_m"]
        for a, b in zip(table, table[1:])
    )
    fy = aging.first_year_state()
    my = aging.material_state(3.0)
    ratios = {"sigma_f": my.flexural_strength_pa / fy.flexural_strength_pa,
              "sigma_c": my.compressive_strength_pa / fy.compressive_strength_pa,
              "E": my.elastic_modulus_pa / fy.elastic_modulus_pa,
              "thickness": my.thickness_m / fy.thickness_m}
    in_band = (1.3 <= ratios["sigma_f"] <= 2.5 and 1.2 <= ratios["sigma_c"] <= 2.5
               and 1.1 <= ratios["E"] <= 1.7 and ratios["thickness"] > 1.5)
    add("i_ageing_law_limits",
        "sigma_f, sigma_c, E, h monotone in age; MY/FY(3 a) ratios inside the cited "
        "bands sigma_f 1.3-2.5, sigma_c 1.2-2.5, E 1.1-1.7, h > 1.5",
        {"monotone": mono, "my_over_fy_at_3a": ratios}, mono and in_band)

    # (j) provenance
    add("j_provenance_hashed",
        "ageing ledger + FSD spec + sampler seeds + node counts hashed into the artifact",
        {"aging_ledger_sha256": artifact["provenance"]["aging_ledger_sha256"],
         "reference_ledger_sha256": artifact["provenance"]["level_ice_ledger_sha256"],
         "field_spec_sha256": artifact["provenance"]["field_spec_sha256"]},
        bool(artifact["provenance"]["aging_ledger_sha256"]))
    return out


# ---------------------------------------------------------------------------
# 5. campaign driver
# ---------------------------------------------------------------------------
def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str,
                                     separators=(",", ":")).encode()).hexdigest()


def run_campaign(*, workers: int | None = None, with_dem: bool = True,
                 dem_resolution_check: bool = True) -> dict[str, Any]:
    """Run the whole campaign and return the artifact dict."""
    from tensordem.level_ice.formula import ledger_sha256

    started = time.time()
    artifact: dict[str, Any] = {
        "campaign": "P54-W2-N01 mixed FY/MY ice resistance",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "my_fractions": list(MY_FRACTIONS),
        "speeds_m_s": list(SPEEDS_M_S),
        "primary_speed_m_s": PRIMARY_SPEED_M_S,
        "my_strength_band": list(MY_STRENGTH_BAND),
    }
    artifact["ageing_table"] = ageing_table()
    artifact["my_fraction_sweep"] = my_fraction_sweep()
    artifact["class_ratio_decomposition"] = {
        model: class_ratio_decomposition(MixedFieldSpec(my_fraction=0.4), model=model)
        for model in ("lindqvist", "riska")
    }
    artifact["sensitivity"] = resistance_sensitivity(
        MixedFieldSpec(my_fraction=0.4), model="lindqvist",
        speed_m_s=PRIMARY_SPEED_M_S)
    artifact["joint_envelope"] = joint_envelope(base_my_fraction=0.4)
    spec = MixedFieldSpec(my_fraction=0.4)
    artifact["estimator_check"] = quadrature_vs_sampler(
        spec, speed_m_s=PRIMARY_SPEED_M_S, model="lindqvist", n=20_000)
    artifact["field_summary_p04"] = quadrature_field(spec).summary()
    artifact["field_histogram_p04"] = sample_field(spec, n=20_000).as_histogram(bins=30)
    artifact["dem"] = (run_dem_factorial(workers=workers,
                                        resolution_check=dem_resolution_check)
                       if with_dem else {"skipped": True})
    artifact["provenance"] = {
        "aging_ledger_sha256": aging.aging_ledger_sha256(),
        "level_ice_ledger_sha256": ledger_sha256(),
        "field_spec_sha256": _sha({
            "fy": MixedFieldSpec().fy.as_dict(),
            "my": my_distribution().as_dict(),
            "my_age_range": [2.0, 5.0],
            "seed": MixedFieldSpec().seed,
        }),
        "coefficient_ledger": {
            k: [v if math.isfinite(v) else "nan", g, t]
            for k, (v, g, t) in aging.COEFFICIENT_LEDGER.items()
        },
    }
    artifact["verdicts"] = verdicts(artifact)
    artifact["n_pass"] = sum(1 for v in artifact["verdicts"].values() if v["verdict"] == "PASS")
    artifact["n_fail"] = sum(1 for v in artifact["verdicts"].values() if v["verdict"] == "FAIL")
    artifact["wall_clock_s"] = time.time() - started
    return artifact


def _sanitise(obj: Any) -> Any:
    """Replace non-finite floats by ``None`` so the JSON is standards-compliant."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def save_artifact(artifact: dict[str, Any], path: str | Path) -> Path:
    """Write the artifact JSON (stable key order, no NaN/Infinity)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _sanitise(artifact)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target