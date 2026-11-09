"""Bimodal first-year / multi-year floe-thickness distribution (FSD).

Definition of the mixed field
-----------------------------
The ice field is a **mixture of two classes** whose mixing weight is the
multi-year *area* fraction ``p``:

    F(h) = (1 - p) F_FY(h) + p F_MY(h),

with ``F_FY`` a lognormal floe-thickness distribution around a first-year mean
(1 m class, CV 0.35) and ``F_MY`` the thickness distribution of survived ice.
``F_MY`` is *not* a free lognormal: each MY floe carries an age ``a`` (uniform on
``my_age_range``, default 2-5 a) and its thickness is the ageing law of
:mod:`tensordem.mixed_ice.aging` evaluated at that age, times a lognormal spread.
Age therefore drives salinity, temperature, brine volume and strength together,
so a thick MY floe is automatically also the strong, desalinated one.

Why the expectation is the class-weighted mean
----------------------------------------------
Within a class the floe thickness and the floe diameter are treated as
**independent**.  Because a floe contributes resistance in proportion to its
area, area weighting multiplies the thickness marginal by a kernel that depends
only on diameter, so the area-weighted thickness marginal equals the thickness
marginal and the whole area dependence collapses to the class weight.  Hence

    E[R] = (1 - p) E[R | FY] + p E[R | MY],

with ``p`` the MY area fraction.  This exact collapse is a *declared
assumption* (independent thickness/diameter within class); with correlated
thickness and floe size the collapse fails and the sampler below is the honest
estimator instead of the quadrature.

Two estimators of the same field
--------------------------------
``quadrature_field``  deterministic stratified nodes on the exact mixture:
                      the FY class on equal-probability thickness quantiles, the
                      MY class on an age x thickness tensor grid.  Noise-free, so
                      the reported curve has no Monte-Carlo scatter, and its
                      discretisation error is bounded by a node-doubling check.
``sample_field``      a genuine floe sampler.  Class labels are drawn with the
                      *area-tilted* probability

                          q = p W_FY / (p W_FY + (1 - p) W_MY),

                      where ``W`` is the mean floe area of a class.  That tilt is
                      what makes the realised area fraction equal ``p`` when MY
                      floes are larger than FY floes; drawing ``p`` directly
                      would over-represent MY ice.  The realised area fraction
                      of a sample is returned so the sampler is checked against
                      its target and against the quadrature.

Both estimators are used by the campaign: the quadrature produces the reported
curve, the sampler verifies it and produces the floe-thickness histogram.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import aging

__all__ = [
    "FloeDistribution",
    "MixedFieldSpec",
    "FieldSample",
    "inv_norm_cdf",
    "lognormal_params",
    "lognormal_quantile",
    "sample_field",
    "quadrature_field",
    "my_floe_probability",
]

#: Default FY thickness distribution.  The mean is the ageing law's first-year
#: mean thickness (``aging.FY_THICKNESS_M`` = 1.2 m) so the field's FY class and
#: the ageing trajectory cannot drift apart; the CV 0.35 is a pinned field value.
FY_DEFAULT = dict(thickness_mean_m=aging.FY_THICKNESS_M, thickness_cv=0.35,
                  diameter_mean_m=200.0, diameter_cv=0.6, label="FY")


def inv_norm_cdf(p: Any) -> np.ndarray | float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    relative error < 1.2e-8 over ``p in (0, 1)`` (absolute error < 1e-8 in the far
    tail, where the value itself is ~4.75); the two rational branches (lower and
    upper tail plus a central polynomial) are the published coefficients, and
    the symmetry ``Phi^-1(1-p) = -Phi^-1(p)`` is enforced so the tails match.
    """
    arr = np.asarray(p, dtype=float)
    if np.any(arr <= 0.0) or np.any(arr >= 1.0):
        raise ValueError("p must lie strictly inside (0, 1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow = 0.02425
    out = np.empty_like(arr, dtype=float)
    low = arr < plow
    high = arr > 1.0 - plow
    mid = ~(low | high)
    if np.any(low):
        q = np.sqrt(-2.0 * np.log(arr[low]))
        out[low] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - arr[high]))
        out[high] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if np.any(mid):
        q = arr[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r) + 1.0)
    return float(out) if out.ndim == 0 else out


def lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    """``(mu, sigma)`` of the lognormal with the given arithmetic mean and CV."""
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError("mean must be positive and finite")
    if not math.isfinite(cv) or cv < 0.0:
        raise ValueError("cv must be nonnegative and finite")
    sigma2 = math.log1p(cv * cv)
    return math.log(mean) - 0.5 * sigma2, math.sqrt(sigma2)


def lognormal_quantile(q: Any, mean: float, cv: float) -> np.ndarray | float:
    """Quantile function of the lognormal ``(mean, cv)``."""
    mu, sigma = lognormal_params(mean, cv)
    out = np.exp(mu + sigma * np.asarray(inv_norm_cdf(q), dtype=float))
    return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class FloeDistribution:
    """Lognormal floe-thickness and floe-diameter marginals of one ice class."""

    thickness_mean_m: float = 1.0
    thickness_cv: float = 0.35
    diameter_mean_m: float = 200.0
    diameter_cv: float = 0.6
    label: str = "FY"

    def __post_init__(self) -> None:
        for name in ("thickness_mean_m", "diameter_mean_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("thickness_cv", "diameter_cv"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")

    @property
    def mean_area_m2(self) -> float:
        """Mean floe area ``E[pi D^2 / 4]`` of the class."""
        return math.pi / 4.0 * self.diameter_mean_m ** 2 * (1.0 + self.diameter_cv ** 2)

    def thickness_quantile(self, q: Any) -> np.ndarray | float:
        return lognormal_quantile(q, self.thickness_mean_m, self.thickness_cv)

    def diameter_draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        mu, sigma = lognormal_params(self.diameter_mean_m, self.diameter_cv)
        return rng.lognormal(mu, sigma, n)

    def thickness_draw(self, rng: np.random.Generator, n: int,
                       scale: np.ndarray | float = 1.0) -> np.ndarray:
        mu, sigma = lognormal_params(self.thickness_mean_m, self.thickness_cv)
        return np.asarray(scale, dtype=float) * rng.lognormal(mu, sigma, n)

    def as_dict(self) -> dict[str, Any]:
        mu, sigma = lognormal_params(self.thickness_mean_m, self.thickness_cv)
        return {
            "label": self.label,
            "thickness_mean_m": self.thickness_mean_m,
            "thickness_cv": self.thickness_cv,
            "thickness_lognormal_mu": mu,
            "thickness_lognormal_sigma": sigma,
            "diameter_mean_m": self.diameter_mean_m,
            "diameter_cv": self.diameter_cv,
            "mean_area_m2": self.mean_area_m2,
        }


def my_distribution(age_range: Sequence[float] = (2.0, 5.0),
                    thickness_cv: float = 0.25,
                    diameter_mean_m: float = 400.0,
                    diameter_cv: float = 0.5) -> FloeDistribution:
    """MY floe distribution whose mean thickness is the ageing law at the
    mid-age of ``age_range`` (2-5 a -> 2.4 m for the default coefficients)."""
    lo, hi = float(age_range[0]), float(age_range[1])
    if not (1.0 <= lo < hi):
        raise ValueError("age_range must satisfy 1 <= lo < hi")
    # class-mean thickness = mean of the ageing law over the age window (not the
    # law at the mid-age: the growth curve is convex, so the two differ)
    ages = np.linspace(lo, hi, 256)
    mean_h = float(np.mean(aging.class_thickness_m(ages)))
    return FloeDistribution(thickness_mean_m=mean_h, thickness_cv=thickness_cv,
                            diameter_mean_m=diameter_mean_m,
                            diameter_cv=diameter_cv, label="MY")


@dataclass(frozen=True)
class MixedFieldSpec:
    """A mixed FY/MY ice field: class weight, class marginals, ageing window."""

    my_fraction: float = 0.0
    fy: FloeDistribution = field(default_factory=lambda: FloeDistribution(**FY_DEFAULT))
    my_age_range: tuple[float, float] = (2.0, 5.0)
    my: FloeDistribution | None = None
    seed: int = 20261108

    def __post_init__(self) -> None:
        p = float(self.my_fraction)
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError("my_fraction must lie in [0, 1]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        lo, hi = float(self.my_age_range[0]), float(self.my_age_range[1])
        if not (1.0 <= lo <= hi):
            raise ValueError("my_age_range must satisfy 1 <= lo <= hi")

    @property
    def my_dist(self) -> FloeDistribution:
        return self.my if self.my is not None else my_distribution(self.my_age_range)

    def as_dict(self) -> dict[str, Any]:
        return {
            "my_fraction": self.my_fraction,
            "fy": self.fy.as_dict(),
            "my": self.my_dist.as_dict(),
            "my_age_range": list(self.my_age_range),
            "seed": self.seed,
            "my_floe_probability": my_floe_probability(self),
            "aging_ledger_sha256": aging.aging_ledger_sha256(),
        }


def my_floe_probability(spec: MixedFieldSpec) -> float:
    """Per-floe MY label probability that makes the MY *area* fraction ``p``."""
    p = spec.my_fraction
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    w_fy = spec.fy.mean_area_m2
    w_my = spec.my_dist.mean_area_m2
    return p * w_fy / (p * w_fy + (1.0 - p) * w_my)


@dataclass
class FieldSample:
    """A sampled mixed field: per-floe geometry, class and material state.

    ``weight`` is the integration weight of the node: the floe area [m^2] for a
    Monte-Carlo sample, and the normalised mixture probability for the
    deterministic quadrature.  Every field statistic is the ``weight``-weighted
    mean, so one code path serves both estimators.
    """

    is_my: np.ndarray
    thickness_m: np.ndarray
    diameter_m: np.ndarray
    area_m2: np.ndarray
    weight: np.ndarray
    age_years: np.ndarray
    salinity_permille: np.ndarray
    temperature_c: np.ndarray
    brine_volume_fraction: np.ndarray
    compressive_strength_pa: np.ndarray
    flexural_strength_pa: np.ndarray
    elastic_modulus_pa: np.ndarray
    density_kg_m3: np.ndarray
    target_my_fraction: float
    my_label_probability: float
    seed: int
    method: str

    @property
    def n(self) -> int:
        return int(self.is_my.size)

    @property
    def realised_my_area_fraction(self) -> float:
        total = float(self.weight.sum())
        return float(self.weight[self.is_my].sum()) / total if total > 0 else 0.0

    def class_mean(self, values: np.ndarray, is_my: bool) -> float:
        mask = self.is_my if is_my else ~self.is_my
        w = self.weight[mask]
        total = float(w.sum())
        if total <= 0.0:
            return float("nan")
        return float(np.dot(w, values[mask]) / total)

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n": self.n,
            "seed": self.seed,
            "target_my_fraction": self.target_my_fraction,
            "my_label_probability": self.my_label_probability,
            "realised_my_area_fraction": self.realised_my_area_fraction,
            "n_my": int(np.count_nonzero(self.is_my)),
            "fy_thickness_mean_m": self.class_mean(self.thickness_m, False),
            "my_thickness_mean_m": self.class_mean(self.thickness_m, True),
            "fy_compressive_strength_pa": self.class_mean(self.compressive_strength_pa, False),
            "my_compressive_strength_pa": self.class_mean(self.compressive_strength_pa, True),
            "my_flexural_strength_pa": self.class_mean(self.flexural_strength_pa, True),
            "probability_weighted_thickness_mean_m": float(
                (1.0 - self.realised_my_area_fraction)
                * self.class_mean(self.thickness_m, False)
                + self.realised_my_area_fraction * self.class_mean(self.thickness_m, True)
            ),
        }

    def as_histogram(self, bins: int = 40) -> dict[str, Any]:
        lo = float(np.min(self.thickness_m))
        hi = float(np.max(self.thickness_m))
        edges = np.linspace(lo, hi, bins + 1)
        out: dict[str, Any] = {"bin_edges_m": edges.tolist()}
        for label, mask in (("FY", ~self.is_my), ("MY", self.is_my)):
            counts, _ = np.histogram(self.thickness_m[mask], bins=edges)
            out[label] = counts.astype(int).tolist()
        return out


def sample_field(spec: MixedFieldSpec, n: int = 100_000,
                 seed: int | None = None) -> FieldSample:
    """Monte-Carlo floe sample of the mixed field with the area-tilted labels."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(spec.seed if seed is None else seed)
    q_my = my_floe_probability(spec)
    is_my = rng.random(n) < q_my
    my_age_lo, my_age_hi = spec.my_age_range
    thickness = np.empty(n, dtype=float)
    diameter = np.empty(n, dtype=float)
    age = np.ones(n, dtype=float)

    n_fy = int(np.count_nonzero(~is_my))
    if n_fy:
        thickness[~is_my] = spec.fy.thickness_draw(rng, n_fy)
        diameter[~is_my] = spec.fy.diameter_draw(rng, n_fy)
    n_my = int(np.count_nonzero(is_my))
    if n_my:
        my = spec.my_dist
        mu, sigma = lognormal_params(my.thickness_mean_m, my.thickness_cv)
        my_age = rng.uniform(my_age_lo, my_age_hi, n_my) if my_age_hi > my_age_lo else np.full(n_my, my_age_lo)
        base = np.asarray(aging.class_thickness_m(my_age), dtype=float)
        # ``my.thickness_mean_m`` is the class mean thickness: the ageing-law
        # profile carries the *shape* and is rescaled so that E[thickness] equals
        # the declared mean exactly (the lognormal factor has mean one).
        base = base * (my.thickness_mean_m / float(np.mean(aging.class_thickness_m(
            np.linspace(my_age_lo, max(my_age_lo, my_age_hi), 256)))))
        thickness[is_my] = base * rng.lognormal(mu - math.log(my.thickness_mean_m), sigma, n_my)
        diameter[is_my] = my.diameter_draw(rng, n_my)
        age[is_my] = my_age

    state = aging.material_state_arrays(age)
    return FieldSample(
        is_my=is_my,
        thickness_m=thickness,
        diameter_m=diameter,
        area_m2=math.pi / 4.0 * diameter ** 2,
        weight=math.pi / 4.0 * diameter ** 2,
        age_years=state["age_years"],
        salinity_permille=state["salinity_permille"],
        temperature_c=state["temperature_c"],
        brine_volume_fraction=state["brine_volume_fraction"],
        compressive_strength_pa=state["compressive_strength_pa"],
        flexural_strength_pa=state["flexural_strength_pa"],
        elastic_modulus_pa=state["elastic_modulus_pa"],
        density_kg_m3=state["density_kg_m3"],
        target_my_fraction=spec.my_fraction,
        my_label_probability=q_my,
        seed=spec.seed if seed is None else seed,
        method="monte-carlo",
    )


def quadrature_field(spec: MixedFieldSpec, n_age: int = 8,
                     n_thickness: int = 128) -> FieldSample:
    """Deterministic stratified quadrature of the same field (no noise)."""
    if n_age <= 0 or n_thickness <= 0:
        raise ValueError("node counts must be positive")
    p = spec.my_fraction
    nodes: dict[str, list[np.ndarray]] = {k: [] for k in (
        "is_my", "thickness_m", "diameter_m", "area_m2", "age_years",
        "salinity_permille", "temperature_c", "brine_volume_fraction",
        "compressive_strength_pa", "flexural_strength_pa",
        "elastic_modulus_pa", "density_kg_m3", "weight")}
    # ``weight`` is filled by push(); every other key is a per-node field.

    def push(arrays: dict[str, np.ndarray], weights: np.ndarray) -> None:
        for key in nodes:
            nodes[key].append(weights if key == "weight" else arrays[key])

    # FY class: equal-probability thickness quantiles, FY reference age = 1 a
    if p < 1.0:
        q = (np.arange(n_thickness) + 0.5) / n_thickness
        h_fy = np.asarray(spec.fy.thickness_quantile(q), dtype=float)
        state = aging.material_state_arrays(np.ones(n_thickness))
        arrays = {
            "is_my": np.zeros(n_thickness, dtype=bool),
            "thickness_m": h_fy,
            "diameter_m": np.full(n_thickness, spec.fy.diameter_mean_m),
            "area_m2": np.full(n_thickness, spec.fy.mean_area_m2),
            "age_years": state["age_years"],
            "salinity_permille": state["salinity_permille"],
            "temperature_c": state["temperature_c"],
            "brine_volume_fraction": state["brine_volume_fraction"],
            "compressive_strength_pa": state["compressive_strength_pa"],
            "flexural_strength_pa": state["flexural_strength_pa"],
            "elastic_modulus_pa": state["elastic_modulus_pa"],
            "density_kg_m3": state["density_kg_m3"],
        }
        push(arrays, np.full(n_thickness, (1.0 - p) / n_thickness))

    # MY class: age x thickness tensor grid, equal weights
    if p > 0.0:
        my = spec.my_dist
        lo, hi = spec.my_age_range
        ages = (np.arange(n_age) + 0.5) / n_age * (hi - lo) + lo
        q = (np.arange(n_thickness) + 0.5) / n_thickness
        mu, sigma = lognormal_params(my.thickness_mean_m, my.thickness_cv)
        base = np.asarray(aging.class_thickness_m(ages), dtype=float)
        base = base * (my.thickness_mean_m / float(np.mean(base)))
        scale = np.exp(mu - math.log(my.thickness_mean_m) + sigma
                       * np.asarray(inv_norm_cdf(q), dtype=float))
        h_my = (scale[None, :] * base[:, None]).ravel()
        ages_grid = np.repeat(ages, n_thickness)
        state = aging.material_state_arrays(ages_grid)
        m = ages_grid.size
        arrays = {
            "is_my": np.ones(m, dtype=bool),
            "thickness_m": h_my,
            "diameter_m": np.full(m, my.diameter_mean_m),
            "area_m2": np.full(m, my.mean_area_m2),
            "age_years": state["age_years"],
            "salinity_permille": state["salinity_permille"],
            "temperature_c": state["temperature_c"],
            "brine_volume_fraction": state["brine_volume_fraction"],
            "compressive_strength_pa": state["compressive_strength_pa"],
            "flexural_strength_pa": state["flexural_strength_pa"],
            "elastic_modulus_pa": state["elastic_modulus_pa"],
            "density_kg_m3": state["density_kg_m3"],
        }
        push(arrays, np.full(m, p / m))

    concat = {k: np.concatenate(v) for k, v in nodes.items()}
    weights = concat["weight"]
    # normalise the model weights (they sum to 1 by construction up to rounding)
    concat["weight"] = weights / float(weights.sum())
    return FieldSample(
        target_my_fraction=p,
        my_label_probability=my_floe_probability(spec),
        seed=spec.seed,
        method="quadrature",
        **concat,
    )