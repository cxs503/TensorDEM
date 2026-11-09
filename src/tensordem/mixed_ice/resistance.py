"""Mixed-field ice resistance: class-resolved anchors integrated over the FSD.

Model
-----
For a given speed and hull the resistance of one floe is the engineering anchor
of :mod:`tensordem.level_ice.formula` evaluated at **that floe's** thickness and
**that floe's** material state (age-dependent strength, modulus, density):

    R(p) = sum_i w_i * R_anchor(h_i, sigma_c(a_i), sigma_f(a_i), E(a_i), rho(a_i))

with ``w_i`` the integration weight of the FSD node (see
:mod:`tensordem.mixed_ice.field`).  Nothing here is re-implemented: the anchors
are called, so there is a single source of truth for the coefficients and their
ledger hash.

Two consequences are reported explicitly because they bound the claim:

**Additivity.**  ``R(p) = (1 - p) R_FY + p R_MY`` by construction, so

    dR/R(0) = p * (R_MY / R_FY - 1) .

The MY-fraction sensitivity is therefore *entirely* the class ratio
``R_MY/R_FY``; the curve shape carries no independent information.  The class
ratio is decomposed into a thickness part and a strength part (below) so the
report can say which one dominates.

**Class-ratio decomposition.**  Evaluating the anchor for the MY material state
at the *FY* thickness isolates the strength contribution; evaluating the FY
material at the *MY* mean thickness isolates the thickness contribution.  The
product of the two, divided by the FY baseline, is the leading-order
factorisation of ``R_MY/R_FY`` and its residual measures the interaction.

What is deliberately absent: floe-size sub-level ice, floe rotation, broken-ice
channels between floes, and any direct DEM-to-ship calibration.  Every floe is
treated as level ice of its own thickness, which makes the mixed-field number an
*upper* estimate for a broken field (documented limitation).

Sensitivity
-----------
``resistance_sensitivity`` perturbs one field parameter at a time and returns
the resistance envelope.  The parameters are split into an **FSD group**
(thickness distribution of each class) and a **material group** (the MY strength
multiplier that stands for the uncertainty of the ageing parameterisation), so
the report can answer "how much of the spread is ice-thickness distribution,
how much is ice mechanics".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from . import aging
from .field import (FloeDistribution, FieldSample, MixedFieldSpec,
                    my_distribution, quadrature_field, sample_field)

__all__ = [
    "MixedResistance",
    "mixed_resistance",
    "anchor_ratio",
    "class_ratio_decomposition",
    "resistance_sensitivity",
    "SENSITIVITY_PARAMETERS",
]

_SUPPORTED_MODELS = ("lindqvist", "riska")


def _anchor(model: str, ice_state, hull, speed_m_s: float) -> dict[str, Any]:
    from tensordem.level_ice import formula

    if model == "lindqvist":
        return formula.lindqvist_components(ice_state, hull, speed_m_s)
    if model == "riska":
        return formula.riska_components(ice_state, hull, speed_m_s)
    raise ValueError(f"model must be one of {_SUPPORTED_MODELS}")


@dataclass
class MixedResistance:
    """Resistance of a mixed ice field, with the class split kept separate."""

    my_fraction: float
    model: str
    speed_m_s: float
    hull_name: str
    resistance_mean_n: float
    resistance_fy_n: float
    resistance_my_n: float
    class_ratio: float
    relative_increase: float
    component_shares: dict[str, float]
    quantiles_n: dict[str, float]
    field_summary: dict[str, Any]
    nodes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "my_fraction": self.my_fraction,
            "model": self.model,
            "speed_m_s": self.speed_m_s,
            "hull_name": self.hull_name,
            "resistance_mean_N": self.resistance_mean_n,
            "resistance_fy_N": self.resistance_fy_n,
            "resistance_my_N": self.resistance_my_n,
            "class_ratio": self.class_ratio if math.isfinite(self.class_ratio) else None,
            "relative_increase": (self.relative_increase
                                  if math.isfinite(self.relative_increase) else None),
            "component_shares": self.component_shares,
            "quantiles_N": self.quantiles_n,
            "field": self.field_summary,
            "nodes": self.nodes,
        }


def _node_resistance(sample: FieldSample, model: str, hull,
                     speed_m_s: float,
                     my_strength_multiplier: float = 1.0) -> dict[str, np.ndarray]:
    from tensordem.level_ice.formula import IceState

    n = sample.n
    total = np.empty(n)
    crushing = np.empty(n)
    bending = np.empty(n)
    submersion = np.empty(n)
    drag = np.empty(n)
    for i in range(n):
        mult = my_strength_multiplier if sample.is_my[i] else 1.0
        state = IceState(
            thickness_m=float(sample.thickness_m[i]),
            compressive_strength_pa=float(sample.compressive_strength_pa[i]) * mult,
            flexural_strength_pa=float(sample.flexural_strength_pa[i]) * mult,
            elastic_modulus_pa=float(sample.elastic_modulus_pa[i]) * mult,
            density_kg_m3=float(sample.density_kg_m3[i]),
        )
        payload = _anchor(model, state, hull, speed_m_s)
        total[i] = payload["total_N"]
        if model == "lindqvist":
            crushing[i] = payload["crushing_N"]
            bending[i] = payload["bending_N"]
            submersion[i] = payload["submersion_N"]
            drag[i] = payload["added_drag_N"]
        else:
            crushing[i] = payload["crushing_reference_N"]
            bending[i] = payload["bending_reference_N"]
            submersion[i] = payload["submersion_reference_N"]
            drag[i] = payload["total_N"] - (
                payload["crushing_reference_N"] + payload["bending_reference_N"]
                + payload["submersion_reference_N"]
            )
    return {"total": total, "crushing": crushing, "bending": bending,
            "submersion": submersion, "drag": drag}


def _weighted_quantile(values: np.ndarray, weights: np.ndarray,
                       q: Sequence[float]) -> dict[str, float]:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    c = np.cumsum(w)
    c = c / c[-1]
    out = {}
    for level in q:
        idx = int(np.searchsorted(c, level))
        idx = min(idx, v.size - 1)
        out[f"p{int(round(level * 100)):02d}"] = float(v[idx])
    return out


def mixed_resistance(spec: MixedFieldSpec, *, hull=None, speed_m_s: float = 3.0,
                     model: str = "lindqvist",
                     n_age: int = 8, n_thickness: int = 128,
                     my_strength_multiplier: float = 1.0) -> MixedResistance:
    """Resistance of the mixed field of ``spec`` (deterministic quadrature)."""
    from tensordem.level_ice.formula import ICEBREAKER

    if model not in _SUPPORTED_MODELS:
        raise ValueError(f"model must be one of {_SUPPORTED_MODELS}")
    if not math.isfinite(speed_m_s) or speed_m_s < 0.0:
        raise ValueError("speed_m_s must be nonnegative and finite")
    if not math.isfinite(my_strength_multiplier) or my_strength_multiplier <= 0.0:
        raise ValueError("my_strength_multiplier must be positive and finite")
    hull = ICEBREAKER if hull is None else hull

    sample = quadrature_field(spec, n_age=n_age, n_thickness=n_thickness)
    parts = _node_resistance(sample, model, hull, speed_m_s, my_strength_multiplier)
    w = sample.weight
    mask_my, mask_fy = sample.is_my, ~sample.is_my
    w_my, w_fy = float(w[mask_my].sum()), float(w[mask_fy].sum())

    r_mean = float(np.dot(w, parts["total"]))
    r_fy = float(np.dot(w[mask_fy], parts["total"][mask_fy]) / w_fy) if w_fy > 0 else float("nan")
    r_my = float(np.dot(w[mask_my], parts["total"][mask_my]) / w_my) if w_my > 0 else float("nan")
    ratio = r_my / r_fy if (w_fy > 0 and w_my > 0 and r_fy > 0) else float("nan")
    relative = r_mean / r_fy - 1.0 if w_fy > 0 and r_fy > 0 else float("nan")

    shares = {}
    for name in ("crushing", "bending", "submersion", "drag"):
        shares[name] = float(np.dot(w, parts[name]) / r_mean) if r_mean > 0 else float("nan")

    return MixedResistance(
        my_fraction=spec.my_fraction,
        model=model,
        speed_m_s=speed_m_s,
        hull_name=getattr(hull, "name", "hull"),
        resistance_mean_n=r_mean,
        resistance_fy_n=r_fy,
        resistance_my_n=r_my,
        class_ratio=ratio,
        relative_increase=relative,
        component_shares=shares,
        quantiles_n=_weighted_quantile(parts["total"], w, (0.1, 0.5, 0.9)),
        field_summary=sample.summary(),
        nodes=sample.n,
    )


def anchor_ratio(fy_state, my_state, *, hull=None, speed_m_s: float = 3.0,
                 model: str = "lindqvist", thickness_m: float | None = None) -> dict[str, Any]:
    """``R_MY / R_FY`` for two material states, optionally at equal thickness."""
    from tensordem.level_ice.formula import ICEBREAKER

    hull = ICEBREAKER if hull is None else hull
    h = fy_state.thickness_m if thickness_m is None else float(thickness_m)
    r_fy = _anchor(model, fy_state.as_ice_state(h), hull, speed_m_s)["total_N"]
    r_my = _anchor(model, my_state.as_ice_state(h), hull, speed_m_s)["total_N"]
    return {
        "model": model,
        "thickness_m": h,
        "speed_m_s": speed_m_s,
        "R_fy_N": r_fy,
        "R_my_N": r_my,
        "ratio_my_over_fy": r_my / r_fy,
    }


def class_ratio_decomposition(spec: MixedFieldSpec, *, hull=None,
                              speed_m_s: float = 3.0, model: str = "lindqvist",
                              my_age_years: float | None = None) -> dict[str, Any]:
    """Split the class ratio into a thickness part and a strength part.

    ``ratio_strength``  MY material at the FY mean thickness (strength only);
    ``ratio_thickness`` FY material at the MY mean thickness (thickness only);
    ``ratio_full``      MY material at the MY mean thickness (both);
    ``interaction``     ``ratio_full / (ratio_strength * ratio_thickness) - 1``.
    """
    from tensordem.level_ice.formula import ICEBREAKER

    hull = ICEBREAKER if hull is None else hull
    fy = aging.first_year_state()
    age = 0.5 * (spec.my_age_range[0] + spec.my_age_range[1]) if my_age_years is None else my_age_years
    my = aging.material_state(age)
    h_fy = fy.thickness_m
    h_my = spec.my_dist.thickness_mean_m

    def r(state, thickness: float) -> float:
        return _anchor(model, state.as_ice_state(thickness), hull, speed_m_s)["total_N"]

    base = r(fy, h_fy)
    r_strength = r(my, h_fy) / base
    r_thickness = r(fy, h_my) / base
    r_full = r(my, h_my) / base
    return {
        "model": model,
        "speed_m_s": speed_m_s,
        "fy_thickness_m": h_fy,
        "my_thickness_m": h_my,
        "my_age_years": age,
        "ratio_strength_only": r_strength,
        "ratio_thickness_only": r_thickness,
        "ratio_full": r_full,
        "interaction": r_full / (r_strength * r_thickness) - 1.0,
        "strength_share_of_increment": (r_strength - 1.0) / (r_full - 1.0) if r_full != 1.0 else float("nan"),
        "thickness_share_of_increment": (r_thickness - 1.0) / (r_full - 1.0) if r_full != 1.0 else float("nan"),
    }


#: One-at-a-time sensitivity parameters: name -> (group, levels).
SENSITIVITY_PARAMETERS: dict[str, tuple[str, tuple[float, ...]]] = {
    "my_fraction": ("primary", (0.0, 0.2, 0.4, 0.6, 0.8)),
    "fy_thickness_mean_m": ("fsd", (0.8, 1.0, 1.2, 1.5)),
    "fy_thickness_cv": ("fsd", (0.2, 0.35, 0.5)),
    "my_thickness_scale": ("fsd", (0.8, 1.0, 1.2)),
    "my_thickness_cv": ("fsd", (0.15, 0.25, 0.4)),
    "my_strength_multiplier": ("material", (0.8, 1.0, 1.2)),
}


def _spec_with(spec: MixedFieldSpec, name: str, value: float) -> dict[str, Any]:
    """Translate one sensitivity parameter into keyword arguments."""
    fy = spec.fy
    my = spec.my_dist
    kw: dict[str, Any] = {"spec": spec, "n_age": 6, "n_thickness": 64,
                          "my_strength_multiplier": 1.0}
    if name == "my_fraction":
        kw["spec"] = replace(spec, my_fraction=float(value))
    elif name == "fy_thickness_mean_m":
        kw["spec"] = replace(spec, fy=replace(fy, thickness_mean_m=float(value)))
    elif name == "fy_thickness_cv":
        kw["spec"] = replace(spec, fy=replace(fy, thickness_cv=float(value)))
    elif name == "my_thickness_scale":
        kw["spec"] = replace(spec, my=replace(my, thickness_mean_m=my.thickness_mean_m * float(value)))
    elif name == "my_thickness_cv":
        kw["spec"] = replace(spec, my=replace(my, thickness_cv=float(value)))
    elif name == "my_strength_multiplier":
        kw["my_strength_multiplier"] = float(value)
    else:
        raise ValueError(f"unknown sensitivity parameter {name!r}")
    return kw


def resistance_sensitivity(spec: MixedFieldSpec, *, hull=None, speed_m_s: float = 3.0,
                           model: str = "lindqvist",
                           parameters: dict[str, tuple[str, tuple[float, ...]]] | None = None,
                           n_age: int = 6, n_thickness: int = 64) -> dict[str, Any]:
    """One-at-a-time sensitivity of the mixed-field resistance.

    Returns per-parameter levels, the resistance at each level, the absolute
    span, and a group split (``fsd`` versus ``material``).  The joint envelope is
    reported as the conservative additive bound ``sum(max |dR|)``: for monotone,
    independent effects that bound is not exceeded, and it is stated as a bound
    rather than computed from a full cross product.
    """
    params = SENSITIVITY_PARAMETERS if parameters is None else parameters
    base = mixed_resistance(spec, hull=hull, speed_m_s=speed_m_s, model=model,
                            n_age=n_age, n_thickness=n_thickness).resistance_mean_n
    rows: dict[str, Any] = {}
    for name, (group, levels) in params.items():
        values = []
        for level in levels:
            kw = _spec_with(spec, name, level)
            kw["n_age"] = n_age
            kw["n_thickness"] = n_thickness
            result = mixed_resistance(hull=hull, speed_m_s=speed_m_s, model=model, **kw)
            values.append({
                "level": float(level),
                "resistance_N": result.resistance_mean_n,
                "class_ratio": result.class_ratio,
                "relative_increase": result.relative_increase,
            })
        resistances = [v["resistance_N"] for v in values]
        rows[name] = {
            "group": group,
            "levels": values,
            "base_resistance_N": base,
            "span_N": max(resistances) - min(resistances),
            "span_fraction_of_base": (max(resistances) - min(resistances)) / base if base else float("nan"),
            "max_abs_deviation_N": max(abs(r - base) for r in resistances),
        }
    fsd_bound = sum(rows[n]["max_abs_deviation_N"] for n in rows if rows[n]["group"] == "fsd")
    material_bound = sum(rows[n]["max_abs_deviation_N"] for n in rows if rows[n]["group"] == "material")
    ranked = sorted((n for n in rows if rows[n]["group"] != "primary"),
                    key=lambda n: rows[n]["max_abs_deviation_N"], reverse=True)
    return {
        "model": model,
        "speed_m_s": speed_m_s,
        "base_my_fraction": spec.my_fraction,
        "base_resistance_N": base,
        "parameters": rows,
        "rank_by_span": ranked,
        "fsd_additive_bound_N": fsd_bound,
        "material_additive_bound_N": material_bound,
        "fsd_share_of_bound": fsd_bound / (fsd_bound + material_bound) if (fsd_bound + material_bound) else float("nan"),
    }


def quadrature_vs_sampler(spec: MixedFieldSpec, *, hull=None, speed_m_s: float = 3.0,
                          model: str = "lindqvist", n: int = 20_000,
                          seed: int | None = None,
                          n_age: int = 8, n_thickness: int = 128) -> dict[str, Any]:
    """Cross-check: the same field expectation from the sampler and quadrature."""
    from tensordem.level_ice.formula import ICEBREAKER

    hull = ICEBREAKER if hull is None else hull
    quad = mixed_resistance(spec, hull=hull, speed_m_s=speed_m_s, model=model,
                            n_age=n_age, n_thickness=n_thickness)
    sample = sample_field(spec, n=n, seed=seed)
    parts = _node_resistance(sample, model, hull, speed_m_s)
    w = sample.weight
    sw = float(w.sum())
    r_sample = float(np.dot(w, parts["total"]) / sw)
    # standard error of the weighted mean: the estimator's own uncertainty, so the
    # quadrature is tested against it instead of against an arbitrary tolerance
    var = float(np.dot(w ** 2, (parts["total"] - r_sample) ** 2)) / sw ** 2
    se = math.sqrt(var)
    return {
        "quadrature_N": quad.resistance_mean_n,
        "sampler_N": r_sample,
        "sampler_standard_error_N": se,
        "z_score": (r_sample - quad.resistance_mean_n) / se if se > 0 else float("inf"),
        "relative_difference": abs(r_sample - quad.resistance_mean_n) / quad.resistance_mean_n,
        "sampler_realised_my_fraction": sample.realised_my_area_fraction,
        "target_my_fraction": spec.my_fraction,
        "my_fraction_bias": sample.realised_my_area_fraction - spec.my_fraction,
        "n": n,
        "seed": sample.seed,
    }