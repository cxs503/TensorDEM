"""Ice-ridge keel grounding: scenario assembly and cross-model comparison metrics.

This module glues the three physical levels of the W5-N03 study together:

* :class:`RidgeGroundingScenario` -- the grounding case (ridge keel geometry and
  material, seabed foundation, hull contact patch, hull panel);
* :func:`kerr_foundation_for_keel` -- the Kerr (three-parameter) foundation that
  the discrete keel corresponds to, including the eccentricity term
  ``g = k_f (H/2)^2`` that a foundation mounted at the keel's lower face
  contributes (the reason a Winkler bed is not enough);
* :func:`compare_keel_with_kerr` -- DEM (discrete bonded-particle) total
  resistance versus the Kerr beam analytic solution, the criterion (b) metric;
* :func:`grounding_probability` -- keel draft versus water depth, the grounding
  probability of criterion (4);
* :func:`fragment_size_distribution` / :func:`lattice_from_fragment_sizes` -- the
  W5-N02 crushed-ice fragment interface;
* :func:`contact_damage` -- contact pressure patch -> local crushing damage,
  the W3-N05 collision-damage interface.

All functions are pure and JSON-serialisable so the campaign script can dump the
numbers straight into an artifact file (no hand-copied digits).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .kerr_foundation import KerrFoundation, BeamSection, characteristic_length
from .local_crushing import PlatePanel, crushing_damage, rigid_plastic_membrane_pressure

SCHEMA = "tensordem.ridge-grounding-scenario/1"


@dataclass(frozen=True)
class RidgeGroundingScenario:
    """Physical parameters of a grounded ice-ridge keel under a hull contact."""

    name: str = "W5-N03 reference ridge keel"
    keel_length: float = 12.0          # m, grounded keel span between supports
    keel_thickness: float = 1.5        # m, consolidated layer thickness
    ice_young_modulus: float = 5.0e9   # Pa (2-D modulus = E * 1 m width)
    ice_density: float = 917.0
    keel_crushing_strength: float = 5.0e6   # Pa, consolidated rubble, elastic limit
    foundation_modulus: float = 2.0e7  # N/m^3 = N/m^2 per m of deflection
    punch_width: float = 0.6           # m, hull contact patch width
    hull_panel: PlatePanel = field(default_factory=PlatePanel)
    water_depth_reference: float = 20.0
    keel_draft_median: float = 9.0     # m, log-normal median keel draft
    keel_draft_log_sigma: float = 0.35
    ship_clearance: float = 1.0        # m, underkeel clearance of the ship

    def __post_init__(self) -> None:
        for name in ("keel_length", "keel_thickness", "ice_young_modulus", "ice_density",
                     "keel_crushing_strength", "punch_width", "water_depth_reference",
                     "keel_draft_median"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.foundation_modulus) or self.foundation_modulus < 0.0:
            raise ValueError("foundation_modulus must be finite and nonnegative")
        if not math.isfinite(self.keel_draft_log_sigma) or self.keel_draft_log_sigma <= 0.0:
            raise ValueError("keel_draft_log_sigma must be finite and positive")
        if not math.isfinite(self.ship_clearance) or self.ship_clearance < 0.0:
            raise ValueError("ship_clearance must be finite and nonnegative")

    @property
    def section(self) -> BeamSection:
        return BeamSection(young_modulus=self.ice_young_modulus,
                           thickness=self.keel_thickness, width=1.0)

    @property
    def target_e2d(self) -> float:
        """Target 2-D modulus of the discrete keel [N/m]."""
        return self.ice_young_modulus

    @property
    def breaking_strain(self) -> float:
        """Bond strain at the keel crushing strength (micro-macro calibration)."""
        return self.keel_crushing_strength / self.ice_young_modulus

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("hull_panel", None)
        data["hull_panel"] = asdict(self.hull_panel)
        data["schema"] = SCHEMA
        data["target_e2d_N_per_m"] = self.target_e2d
        data["beam_ei_Nm2"] = self.section.ei
        data["breaking_strain"] = self.breaking_strain
        return data


def kerr_foundation_for_keel(scenario: RidgeGroundingScenario) -> KerrFoundation:
    """Kerr foundation equivalent to the discrete keel on seabed springs.

    The discrete keel rests on springs at its *lower* face.  Representing that
    support at the beam neutral axis needs (i) the same Winkler stiffness, i.e.
    ``k_u = k_l = 2 k_f`` so that the series combination equals ``k_f``, and
    (ii) a rotational coupling of the springs about the neutral axis, which the
    Kerr shear layer reproduces with ``g = k_f (H/2)^2`` (energy
    ``1/2 k_f e^2 ∫ w'^2`` with the eccentricity ``e = H/2``).  This mapping is
    the reason the three-parameter foundation is used at all.
    """
    kf = scenario.foundation_modulus
    if kf <= 0.0:
        raise ValueError("the Kerr mapping needs a positive foundation modulus")
    eccentricity = 0.5 * scenario.keel_thickness
    return KerrFoundation(upper=2.0 * kf, lower=2.0 * kf, shear=kf * eccentricity**2)


def spearman_monotonic(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Spearman rank correlation of two series (criterion (c) metric)."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size != ya.size or xa.size < 3:
        raise ValueError("both series must have the same length >= 3")
    if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
        raise ValueError("series must be finite")

    def ranks(a: np.ndarray) -> np.ndarray:
        order = a.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(1, a.size + 1, dtype=float)
        # average ties
        _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
        for idx, count in enumerate(counts):
            if count > 1:
                mask = inverse == idx
                r[mask] = r[mask].mean()
        return r

    rx, ry = ranks(xa), ranks(ya)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = math.sqrt(float((rx**2).sum() * (ry**2).sum()))
    rho = float((rx * ry).sum() / denominator) if denominator > 0 else float("nan")
    return {"spearman_rho": rho, "samples": int(xa.size)}


def compare_keel_with_kerr(
    steps: Sequence[Mapping[str, Any]],
    foundation: KerrFoundation,
    ei: float,
    *,
    n_points: int = 401,
) -> dict[str, Any]:
    """Discrete keel resistance versus the Kerr beam analytic solution.

    The discrete observable is the punch reaction ``P_discrete`` at the measured
    foundation-level deflection ``w`` (bottom-row settlement); the continuum
    prediction is the analytic load that produces the same mid-span deflection,
    obtained from the linear relation ``w = C P`` of
    :func:`tensordem.kerr_foundation.kerr_beam_analytic`.  Linear response means
    the comparison is a single stiffness ratio, reported per increment and as the
    mean/max relative deviation -- the criterion (b) metric.
    """
    from .kerr_foundation import kerr_beam_analytic

    if not steps:
        raise ValueError("steps must not be empty")
    unit = kerr_beam_analytic(ei=ei, length=_infer_length(steps), foundation=foundation,
                              load=1.0, n_points=n_points)
    compliance = unit.max_deflection
    rows = []
    for step in steps:
        w = float(step["beam_deflection_m"])
        p_discrete = float(step["reaction_N_per_m"])
        if p_discrete <= 0.0 or w <= 0.0:
            continue
        p_kerr = w / compliance
        rows.append({
            "punch_depth_m": float(step["punch_depth_m"]),
            "beam_deflection_m": w,
            "discrete_resistance_N_per_m": p_discrete,
            "kerr_resistance_N_per_m": p_kerr,
            "relative_error": abs(p_discrete - p_kerr) / p_kerr,
            "discrete_compliance_m_per_N": w / p_discrete,
        })
    if not rows:
        raise ValueError("no usable steps for the comparison")
    errors = [row["relative_error"] for row in rows]
    return {
        "kerr_compliance_m_per_N": compliance,
        "steps": rows,
        "mean_relative_error": float(np.mean(errors)),
        "max_relative_error": float(np.max(errors)),
        "final_relative_error": float(errors[-1]),
        "criterion_15_percent_met": bool(max(errors) < 0.15),
    }


def _infer_length(steps: Sequence[Mapping[str, Any]]) -> float:
    for key in ("keel_length_m", "length_m"):
        if key in steps[0]:
            return float(steps[0][key])
    raise ValueError("steps must carry the keel length")


def grounding_probability(
    water_depths: Iterable[float],
    scenario: RidgeGroundingScenario,
    *,
    penetration_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Probability that a grounded keel touches the seabed at a given depth.

    The keel draft is log-normal (median and log-sigma in the scenario), so
    ``P(grounding | h) = P(draft >= h - clearance + tolerance)`` =
    ``0.5 erfc(z / sqrt(2))`` with ``z = (ln(h) - ln(median)) / sigma``.
    Monotone decreasing in water depth, 0 for deep water and 1 for shallow --
    the grounding criterion (4).  ``penetration_tolerance`` allows the keel to
    be pressed in before the hull is considered to touch.
    """
    depths = [float(h) for h in water_depths]
    if not depths:
        raise ValueError("water_depths must not be empty")
    if not math.isfinite(penetration_tolerance) or penetration_tolerance < 0.0:
        raise ValueError("penetration_tolerance must be finite and nonnegative")
    median = scenario.keel_draft_median
    sigma = scenario.keel_draft_log_sigma
    clearance = scenario.ship_clearance
    rows = []
    for depth in depths:
        if not math.isfinite(depth) or depth <= 0.0:
            raise ValueError("water depths must be finite and positive")
        threshold = depth - clearance - penetration_tolerance
        if threshold <= 0.0:
            probability = 1.0
            z = -math.inf
        else:
            z = (math.log(threshold) - math.log(median)) / sigma
            probability = 0.5 * math.erfc(z / math.sqrt(2.0))
        rows.append({
            "water_depth_m": depth,
            "threshold_draft_m": threshold,
            "z": z if math.isfinite(z) else None,
            "grounding_probability": probability,
        })
    return {
        "median_keel_draft_m": median,
        "log_sigma": sigma,
        "ship_clearance_m": clearance,
        "rows": rows,
        "monotone_decreasing": all(
            rows[i]["grounding_probability"] >= rows[i + 1]["grounding_probability"]
            for i in range(len(rows) - 1)
        ),
    }


def fragment_size_distribution(
    sizes: Sequence[float], counts: Sequence[float] | None = None
) -> dict[str, Any]:
    """Validate/normalise a W5-N02 crushed-fragment size distribution.

    Input is the interface payload of the ridge fragment model: equivalent
    diameters ``sizes`` [m] with optional ``counts``.  The output gives the
    cumulative distribution and the number-median size, which is what the
    discrete keel uses to pick particle radii.
    """
    s = np.asarray(sizes, dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("sizes must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError("sizes must be finite and positive")
    c = np.ones_like(s) if counts is None else np.asarray(counts, dtype=float)
    if c.shape != s.shape or np.any(c < 0.0) or not np.all(np.isfinite(c)):
        raise ValueError("counts must be finite, non-negative and match sizes")
    total = float(c.sum())
    if total <= 0.0:
        raise ValueError("counts must sum to a positive number")
    order = np.argsort(s)
    s, c = s[order], c[order]
    cumulative = np.cumsum(c) / total
    median = float(s[np.searchsorted(cumulative, 0.5)])
    return {
        "sizes_m": s.tolist(),
        "counts": c.tolist(),
        "cumulative": cumulative.tolist(),
        "number_median_m": median,
        "mean_size_m": float((s * c).sum() / total),
        "total_count": total,
    }


def lattice_from_fragment_sizes(
    distribution: Mapping[str, Any],
    *,
    particle_radius_ratio: float = 0.5,
) -> dict[str, Any]:
    """Map a W5-N02 fragment distribution to discrete-keel particle radii.

    The lattice uses a *single* radius per resolution, so the mapping takes
    ``r = ratio * d_median`` (default: half the number-median fragment size) and
    reports the resulting resolution.  Keeping the mapping explicit is what the
    W5-N02 -> W5-N03 interface needs; it is not a hidden constant.
    """
    if "number_median_m" not in distribution:
        raise ValueError("distribution must come from fragment_size_distribution")
    if not math.isfinite(particle_radius_ratio) or particle_radius_ratio <= 0.0:
        raise ValueError("particle_radius_ratio must be finite and positive")
    median = float(distribution["number_median_m"])
    return {
        "fragment_median_m": median,
        "particle_radius_m": particle_radius_ratio * median,
        "particle_diameter_m": 2.0 * particle_radius_ratio * median,
        "interface": "W5-N02 fragment size distribution -> W5-N03 keel lattice",
    }


def contact_damage(
    pressure: float,
    scenario: RidgeGroundingScenario,
    *,
    patch_area: float | None = None,
    load_duration: float | None = None,
) -> dict[str, Any]:
    """Local crushing damage from a hull-keel contact pressure patch (W3-N05)."""
    area = (scenario.punch_width * 1.0) if patch_area is None else patch_area
    return crushing_damage(pressure, scenario.hull_panel, patch_area=area,
                           load_duration=load_duration)


def pressure_from_line_load(line_load: float, punch_width: float) -> float:
    """Convert a 2-D line load [N/m] into a contact pressure [Pa]."""
    if not math.isfinite(punch_width) or punch_width <= 0.0:
        raise ValueError("punch_width must be finite and positive")
    if not math.isfinite(line_load) or line_load < 0.0:
        raise ValueError("line_load must be finite and nonnegative")
    return line_load / punch_width


def kerr_design_summary(scenario: RidgeGroundingScenario, load: float) -> dict[str, Any]:
    """Characteristic length, peak deflection and pressure for a given line load."""
    from .kerr_foundation import kerr_beam_analytic

    foundation = kerr_foundation_for_keel(scenario)
    ei = scenario.section.ei
    solution = kerr_beam_analytic(scenario.keel_length, ei, foundation, load)
    pressure = pressure_from_line_load(load, scenario.punch_width)
    damage = contact_damage(pressure, scenario)
    return {
        "load_N_per_m": load,
        "ei_Nm2": ei,
        "characteristic_length_m": characteristic_length(ei, foundation),
        "keel_length_over_characteristic_length": scenario.keel_length
        / characteristic_length(ei, foundation),
        "peak_deflection_m": solution.max_deflection,
        "peak_pressure_Pa": solution.max_pressure,
        "contact_pressure_Pa": pressure,
        "damage": damage,
    }