"""Turbulent fragmentation of ice fragments in a propeller race.

Physics
-------
A floe of characteristic length ``L`` and thickness ``h`` drifting in water whose
turbulent fluctuation velocity at the floe scale is ``u'`` feels a distributed
drag ``q = 0.5 rho_w C_d u'^2 h`` per unit width. Treating the floe as a
free-ended beam of unit width, the peak bending stress is ::

    sigma_b = 3 q L^2 / (4 h^2) = 3 rho_w C_d u'^2 L^2 / (8 h)

so the largest floe that survives the race is ::

    L_crit = sqrt(8 sigma_f h / (3 rho_w C_d u'^2))

Breakup is then a mass-conserving random binary/ternary fragmentation cascade:
every floe whose size exceeds ``L_crit`` is split into ``k`` children whose
volume fractions follow a size-independent power-law kernel, and the cascade is
repeated until all survivors satisfy the stability criterion.  The resulting
number-density of floe sizes is fitted with a two-parameter Weibull law
(:func:`fit_weibull`) and compared with the reference experimental Weibull law
(:func:`weibull_cdf`) used as the gate for fragment-size statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch


def _finite(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class BreakupProperties:
    """Drag / strength parameters used by the turbulent breakup criterion."""

    flexural_strength_pa: float = 0.5e6
    water_density_kg_m3: float = 1025.0
    drag_coefficient: float = 1.2
    thickness_m: float = 0.1
    min_fragment_m: float = 0.01
    ice_density_kg_m3: float = 917.0

    def __post_init__(self) -> None:
        _finite("flexural_strength_pa", self.flexural_strength_pa, positive=True)
        _finite("water_density_kg_m3", self.water_density_kg_m3, positive=True)
        _finite("drag_coefficient", self.drag_coefficient, positive=True)
        _finite("thickness_m", self.thickness_m, positive=True)
        _finite("min_fragment_m", self.min_fragment_m, positive=True)
        _finite("ice_density_kg_m3", self.ice_density_kg_m3, positive=True)


def critical_floe_length(u_rms_m_s: float, props: BreakupProperties) -> float:
    """Largest stable floe length (``inf`` when the race is calm)."""
    u_rms = _finite("u_rms_m_s", u_rms_m_s, nonnegative=True)
    if u_rms <= 0.0:
        return math.inf
    numerator = 8.0 * props.flexural_strength_pa * props.thickness_m
    denominator = 3.0 * props.water_density_kg_m3 * props.drag_coefficient * u_rms**2
    return math.sqrt(numerator / denominator)


def bending_stress_pa(floe_length_m: float, u_rms_m_s: float, props: BreakupProperties) -> float:
    """Peak bending stress of a floe in the turbulent race."""
    length = _finite("floe_length_m", floe_length_m, positive=True)
    u_rms = _finite("u_rms_m_s", u_rms_m_s, nonnegative=True)
    q = 0.5 * props.water_density_kg_m3 * props.drag_coefficient * u_rms**2 * props.thickness_m
    return 0.75 * q * length**2 / props.thickness_m**2


def floe_volume_m3(length_m: float, props: BreakupProperties) -> float:
    """Volume of a square floe of side ``length_m`` and thickness ``h``."""
    length = _finite("length_m", length_m, positive=True)
    return length**2 * props.thickness_m


def fragment(
    sizes_m: Sequence[float],
    rms_velocity_m_s: float,
    props: BreakupProperties,
    *,
    generator: torch.Generator | None = None,
    max_splits: int = 64,
) -> dict[str, Any]:
    """Mass-conserving fragmentation cascade driven by a turbulent rms velocity.

    Returns the final floe list (``length_m``, ``mass_kg``), the number of split
    events, the largest stable length and the total mass (which is conserved by
    construction and re-checked here).
    """
    u_rms = _finite("rms_velocity_m_s", rms_velocity_m_s, nonnegative=True)
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    l_crit = critical_floe_length(u_rms, props)
    rho_i = 917.0
    initial_mass = sum(rho_i * floe_volume_m3(float(s), props) for s in sizes_m)
    active = [float(s) for s in sizes_m if float(s) > 0.0]
    finished: list[float] = []
    splits = 0
    while active:
        size = active.pop()
        if size <= l_crit or size <= 2.0 * props.min_fragment_m or splits >= max_splits * max(len(sizes_m), 1):
            finished.append(size)
            continue
        splits += 1
        children = int(torch.randint(2, 4, (1,), generator=generator).item())
        # Power-law volume kernel: v_k ~ U^(1/3) gives an approximately
        # self-similar size density under repeated splitting.
        weights = torch.rand(children, generator=generator, dtype=torch.float64) ** 3
        weights = weights / weights.sum()
        for weight in weights.tolist():
            child = size * float(weight) ** 0.5
            active.append(child)
        if len(finished) + len(active) > 20000:
            finished.extend(active)
            active = []
    final_mass = sum(rho_i * floe_volume_m3(s, props) for s in finished)
    return {
        "lengths_m": sorted(finished, reverse=True),
        "masses_kg": [rho_i * floe_volume_m3(s, props) for s in finished],
        "split_events": splits,
        "critical_length_m": l_crit,
        "initial_mass_kg": initial_mass,
        "final_mass_kg": final_mass,
        "mass_error_kg": final_mass - initial_mass,
    }


def weibull_cdf(x: float, shape: float, scale: float) -> float:
    """Two-parameter Weibull CDF."""
    x = _finite("x", x, nonnegative=True)
    shape = _finite("shape", shape, positive=True)
    scale = _finite("scale", scale, positive=True)
    if x <= 0.0:
        return 0.0
    return 1.0 - math.exp(-((x / scale) ** shape))


def empirical_cdf(samples: Sequence[float]) -> list[tuple[float, float]]:
    """Sorted sample / plotting-position CDF pairs (Weibull plotting positions)."""
    values = sorted(float(v) for v in samples if float(v) > 0.0)
    n = len(values)
    return [(value, (rank + 1.0) / (n + 1.0)) for rank, value in enumerate(values)]


def fit_weibull(samples: Sequence[float]) -> dict[str, float]:
    """Least-squares Weibull fit on the transformed CDF (ln x vs ln(-ln(1-F)))."""
    pairs = empirical_cdf(samples)
    if len(pairs) < 3:
        raise ValueError("at least three positive samples are required for a Weibull fit")
    xs = [math.log(value) for value, _ in pairs]
    ys = []
    for _, f in pairs:
        f = min(max(f, 1e-9), 1.0 - 1e-9)
        ys.append(math.log(-math.log(1.0 - f)))
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        raise ValueError("degenerate sample set for Weibull fit")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    shape = sxy / sxx
    intercept = mean_y - shape * mean_x
    if shape <= 0.0:
        raise ValueError("fitted Weibull shape is not positive")
    scale = math.exp(-intercept / shape)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + shape * x)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {"shape": shape, "scale": scale, "r_squared": r_squared, "sample_count": n}


def cdf_agreement_r2(
    samples: Sequence[float], shape: float, scale: float
) -> dict[str, float]:
    """Coefficient of determination of the empirical CDF against a Weibull CDF."""
    pairs = empirical_cdf(samples)
    if len(pairs) < 2:
        raise ValueError("at least two positive samples are required")
    observed = [f for _, f in pairs]
    predicted = [weibull_cdf(value, shape, scale) for value, _ in pairs]
    mean_obs = sum(observed) / len(observed)
    ss_tot = sum((f - mean_obs) ** 2 for f in observed)
    ss_res = sum((f - p) ** 2 for f, p in zip(observed, predicted))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    max_dev = max(abs(f - p) for f, p in zip(observed, predicted))
    return {"r_squared": r_squared, "max_cdf_deviation": max_dev, "sample_count": len(pairs)}


def fit_weibull_cdf(samples: Sequence[float]) -> dict[str, float]:
    """Two-parameter Weibull fit minimising the CDF residual (grid + refinement).

    Unlike :func:`fit_weibull` this fit is performed directly in CDF space, which
    is the correct metric for discretised floe censuses that contain repeated
    sizes (a lattice census has many ties that break the double-log transform).
    """
    pairs = empirical_cdf(samples)
    if len(pairs) < 3:
        raise ValueError("at least three positive samples are required")
    xs = [value for value, _ in pairs]
    fs = [f for _, f in pairs]
    mean_x = sum(xs) / len(xs)

    def residual(shape: float, scale: float) -> float:
        return sum((weibull_cdf(x, shape, scale) - f) ** 2 for x, f in zip(xs, fs))

    best = (float("inf"), 1.0, mean_x)
    for k_index in range(60):
        shape = 0.3 * math.exp(k_index * 0.045)
        for s_index in range(50):
            scale = 0.1 * mean_x * math.exp(s_index * 0.075)
            error = residual(shape, scale)
            if error < best[0]:
                best = (error, shape, scale)
    error, shape, scale = best
    # Local refinement around the grid optimum.
    step_k, step_s = 0.02, 0.02 * scale
    for _ in range(60):
        improved = False
        for dk, ds in ((step_k, 0.0), (-step_k, 0.0), (0.0, step_s), (0.0, -step_s)):
            trial_k = max(shape + dk, 1e-3)
            trial_s = max(scale + ds, 1e-6)
            trial = residual(trial_k, trial_s)
            if trial < error:
                error, shape, scale = trial, trial_k, trial_s
                improved = True
        if not improved:
            step_k *= 0.5
            step_s *= 0.5
    mean_f = sum(fs) / len(fs)
    ss_tot = sum((f - mean_f) ** 2 for f in fs)
    return {
        "shape": shape,
        "scale": scale,
        "r_squared": 1.0 - error / ss_tot if ss_tot > 0.0 else 1.0,
        "sample_count": len(pairs),
        "method": "cdf-least-squares",
    }


def kolmogorov_smirnov(samples: Sequence[float], shape: float, scale: float) -> float:
    """Two-sided Kolmogorov-Smirnov statistic against a Weibull reference."""
    pairs = empirical_cdf(samples)
    return max(abs(f - weibull_cdf(value, shape, scale)) for value, f in pairs)