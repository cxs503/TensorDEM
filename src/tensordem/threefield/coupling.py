"""Three-field coupling: DEM breaking -> LBM race -> fragmentation -> thermal erosion.

Sequence implemented here (one "pass" of a coupled time step):

1. **DEM hull-ice contact** (:mod:`tensordem.threefield.hull_ice`) produces the
   fragment inventory of the broken ice sheet (mass, equivalent diameter) and
   the contact-pressure field on the hull.
2. **LBM propeller race** (:mod:`tensordem.threefield.wake_lbm`) supplies the
   mean race velocity and the turbulent rms fluctuation at the radial station of
   each fragment.
3. **Turbulent fragmentation** (:mod:`tensordem.threefield.fragmentation`) splits
   the floes that exceed the bending-stability length at that rms velocity.
4. **Thermal erosion** (:mod:`tensordem.threefield.thermal`) melts the survivors
   with the Ranz-Marshall correlation and the Stefan condition over the race
   residence time.
5. **Ledger**: initial ice mass equals surviving fragment mass plus melted mass.

Every stage returns plain JSON-serialisable dictionaries so the whole chain can
be written to an artifact file without post-processing.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Sequence

import torch

from .fragmentation import BreakupProperties, fragment, fit_weibull, cdf_agreement_r2
from .thermal import ThermalProperties, integrate_melting, melting_rate_kg_s


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
class CouplingConfig:
    """Parameters of the fragmentation + thermal stages."""

    residence_time_s: float = 30.0
    melt_dt_s: float = 0.25
    relative_velocity_m_s: float = 0.06
    thermal: ThermalProperties = ThermalProperties()
    breakup: BreakupProperties = BreakupProperties()
    #: Propeller-race velocity at the fragment location (physical units). The
    #: race leaves the propeller at 4-6 m/s for an ice-going ship.
    race_velocity_m_s: float = 5.0
    #: Turbulence intensity of the race used to build the fragmentation driver.
    #: The resolved LBM intensity is reported as an audit; at 10 lattice cells
    #: per propeller diameter the race is under-resolved and its resolved
    #: intensity is roughly an order of magnitude below the physical value.
    turbulence_intensity: float = 0.15
    seed: int = 20261031

    def __post_init__(self) -> None:
        _finite("residence_time_s", self.residence_time_s, positive=True)
        _finite("melt_dt_s", self.melt_dt_s, positive=True)
        _finite("relative_velocity_m_s", self.relative_velocity_m_s, nonnegative=True)
        _finite("race_velocity_m_s", self.race_velocity_m_s, positive=True)
        _finite("turbulence_intensity", self.turbulence_intensity, positive=True)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")


def ice_mass_from_sphere(diameter_m: float, density_kg_m3: float) -> float:
    diameter = _finite("diameter_m", diameter_m, nonnegative=True)
    density = _finite("density_kg_m3", density_kg_m3, positive=True)
    return density * math.pi / 6.0 * diameter**3


def diameter_from_mass(mass_kg: float, density_kg_m3: float) -> float:
    mass = _finite("mass_kg", mass_kg, nonnegative=True)
    density = _finite("density_kg_m3", density_kg_m3, positive=True)
    if mass <= 0.0:
        return 0.0
    return (6.0 * mass / (math.pi * density)) ** (1.0 / 3.0)


def dem_inventory(dem_case: dict[str, Any]) -> list[dict[str, float]]:
    """Fragment inventory (mass, equivalent diameter) from a DEM case result."""
    fragments = dem_case["fragments"]
    masses = fragments["masses_kg"]
    diameters = fragments["diameters_m"]
    if len(masses) != len(diameters):
        raise ValueError("DEM fragment census is inconsistent")
    return [
        {"index": float(index), "mass_kg": float(mass), "diameter_m": float(diameter)}
        for index, (mass, diameter) in enumerate(zip(masses, diameters))
    ]


def race_velocity_at(radius_m: float, profile: dict[str, Any]) -> dict[str, float]:
    """Interpolate mean/rms race velocity of a wake profile at a radius."""
    radius = _finite("radius_m", radius_m, nonnegative=True)
    radii = profile["radius"]
    mean = profile["mean_axial_velocity"]
    rms = profile.get("axial_rms_velocity", [0.0] * len(radii))
    if not radii:
        raise ValueError("wake profile is empty")
    if radius <= radii[0]:
        return {"mean_velocity": mean[0], "rms_velocity": rms[0]}
    if radius >= radii[-1]:
        return {"mean_velocity": mean[-1], "rms_velocity": rms[-1]}
    for index in range(1, len(radii)):
        if radius <= radii[index]:
            r0, r1 = radii[index - 1], radii[index]
            weight = 0.0 if r1 == r0 else (radius - r0) / (r1 - r0)
            return {
                "mean_velocity": mean[index - 1] + weight * (mean[index] - mean[index - 1]),
                "rms_velocity": rms[index - 1] + weight * (rms[index] - rms[index - 1]),
            }
    return {"mean_velocity": mean[-1], "rms_velocity": rms[-1]}


def fragmentation_stage(
    inventory: Sequence[dict[str, float]],
    rms_velocity_m_s: float,
    breakup: BreakupProperties,
    *,
    seed: int = 20261031,
) -> dict[str, Any]:
    """Apply the turbulent breakup cascade to a fragment inventory."""
    # Work in plate-equivalent floe lengths so that the L^2 fragmentation
    # kernel conserves the inventory mass exactly.
    rho_i = breakup.ice_density_kg_m3
    lengths = [
        math.sqrt(max(fragment["mass_kg"], 0.0) / (rho_i * breakup.thickness_m))
        for fragment in inventory
    ]
    lengths = [length for length in lengths if length > 0.0]
    generator = torch.Generator().manual_seed(int(seed))
    result = fragment(lengths, rms_velocity_m_s, breakup, generator=generator)
    lengths_out = result["lengths_m"]
    masses_out = result["masses_kg"]
    initial_mass = sum(fragment["mass_kg"] for fragment in inventory)
    final_mass = sum(masses_out)
    statistics: dict[str, float] = {}
    if len(lengths_out) >= 3:
        statistics = fit_weibull(lengths_out)
    return {
        "fragment_count_in": len(inventory),
        "fragment_count_out": len(lengths_out),
        "critical_length_m": result["critical_length_m"],
        "split_events": result["split_events"],
        "masses_kg": masses_out,
        "lengths_m": lengths_out,
        "initial_mass_kg": initial_mass,
        "final_mass_kg": final_mass,
        "mass_error_kg": final_mass - initial_mass,
        "weibull_fit": statistics,
        "rms_velocity_m_s": rms_velocity_m_s,
    }


def thermal_stage(
    inventory: Sequence[dict[str, float]],
    config: CouplingConfig,
    *,
    relative_velocity_m_s: float | None = None,
) -> dict[str, Any]:
    """Melt a fragment inventory over the race residence time.

    Each fragment is integrated with the Ranz-Marshall + Stefan model until the
    residence time is exhausted; the mass ledger is closed in float64 arithmetic.
    """
    velocity = (
        config.relative_velocity_m_s
        if relative_velocity_m_s is None
        else _finite("relative_velocity_m_s", relative_velocity_m_s, nonnegative=True)
    )
    props = config.thermal
    steps = max(1, int(round(config.residence_time_s / config.melt_dt_s)))
    dt = config.residence_time_s / steps
    survivors: list[dict[str, float]] = []
    melted_total = 0.0
    initial_total = 0.0
    histories: list[dict[str, float]] = []
    for index, item in enumerate(inventory):
        mass = float(item["mass_kg"])
        # Melt the mass-equivalent sphere so the thermal stage can only remove
        # mass: the fragment census carries plate volumes (L^2 h) whose
        # equivalent diameter differs from the floe length.
        diameter = diameter_from_mass(mass, props.ice_density_kg_m3)
        initial_total += mass
        if mass <= 0.0:
            continue
        history = integrate_melting(
            diameter, props, relative_velocity_m_s=velocity, duration_s=config.residence_time_s,
            dt_s=dt, include_sensible=False,
        )
        final = history[-1]
        survivors.append(
            {
                "index": float(index),
                "mass_kg": float(final["mass_kg"]),
                "diameter_m": float(final["diameter_m"]),
                "initial_mass_kg": mass,
                "initial_diameter_m": diameter,
                "melted_kg": mass - float(final["mass_kg"]),
            }
        )
        melted_total += mass - float(final["mass_kg"])
        if index < 8:
            histories.append(
                {
                    "index": float(index),
                    "time_s": [row["time_s"] for row in history],
                    "diameter_m": [row["diameter_m"] for row in history],
                    "melt_fraction": [row["melt_fraction"] for row in history],
                }
            )
    remaining = sum(item["mass_kg"] for item in survivors)
    return {
        "relative_velocity_m_s": velocity,
        "residence_time_s": config.residence_time_s,
        "initial_mass_kg": initial_total,
        "remaining_mass_kg": remaining,
        "melted_mass_kg": melted_total,
        "melt_fraction": 0.0 if initial_total <= 0.0 else melted_total / initial_total,
        "survivors": survivors,
        "histories": histories,
        "mass_error_kg": (remaining + melted_total) - initial_total,
    }


def mass_ledger(
    *,
    initial_mass_kg: float,
    surviving_mass_kg: float,
    melted_mass_kg: float,
    extra_terms: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Close the ice-mass ledger ``M0 = M_fragments + M_melted (+ extra)``.

    The residual is reported relative to ``M0`` so the acceptance gate can be
    stated as ``relative_error < 1e-6``.
    """
    initial = _finite("initial_mass_kg", initial_mass_kg, positive=True)
    surviving = _finite("surviving_mass_kg", surviving_mass_kg, nonnegative=True)
    melted = _finite("melted_mass_kg", melted_mass_kg, nonnegative=True)
    extra_terms = dict(extra_terms or {})
    extra_total = sum(float(value) for value in extra_terms.values())
    balance = surviving + melted + extra_total
    residual = balance - initial
    return {
        "initial_mass_kg": initial,
        "surviving_mass_kg": surviving,
        "melted_mass_kg": melted,
        "extra_terms": extra_terms,
        "balance_kg": balance,
        "residual_kg": residual,
        "relative_error": abs(residual) / initial,
    }


def run_coupled_pass(
    *,
    dem_case: dict[str, Any],
    wake_profile: dict[str, Any],
    config: CouplingConfig,
    weibull_reference: tuple[float, float] = (1.8, 0.12),
) -> dict[str, Any]:
    """Run fragmentation + melting for one DEM fragment inventory and race profile.

    ``weibull_reference`` is the (shape, scale) pair of the experimental
    fragment-size law used as the size-distribution gate.
    """
    inventory = dem_inventory(dem_case)
    initial_mass = sum(item["mass_kg"] for item in inventory)
    # Radial station representative of the fragments: they are entrained in the
    # core of the race, so use the profile value at one disk radius.
    representative = race_velocity_at(0.5 * float(wake_profile["radius"][-1] or 1.0), wake_profile)
    resolved_ti = (
        float(representative["rms_velocity"])
        / max(float(representative["mean_velocity"]), 1e-12)
    )
    rms = config.turbulence_intensity * config.race_velocity_m_s
    breakup = fragmentation_stage(inventory, rms, config.breakup, seed=config.seed)
    sizes = breakup["lengths_m"]
    statistics = breakup["weibull_fit"]
    agreement = (
        cdf_agreement_r2(sizes, *weibull_reference) if len(sizes) >= 2 else {"r_squared": float("nan")}
    )
    post_fragmentation = [
        {
            "index": float(index),
            "mass_kg": float(mass),
            "diameter_m": float(length),
        }
        for index, (mass, length) in enumerate(zip(breakup["masses_kg"], breakup["lengths_m"]))
    ]
    thermal = thermal_stage(post_fragmentation, config)
    ledger = mass_ledger(
        initial_mass_kg=initial_mass,
        surviving_mass_kg=thermal["remaining_mass_kg"],
        melted_mass_kg=thermal["melted_mass_kg"],
    )
    return {
        "initial_mass_kg": initial_mass,
        "initial_fragment_count": len(inventory),
        "fragmentation": {key: value for key, value in breakup.items() if key not in ("masses_kg", "lengths_m")},
        "weibull_reference": {"shape": weibull_reference[0], "scale": weibull_reference[1]},
        "weibull_reference_agreement": agreement,
        "fragment_statistics": statistics,
        "fragment_lengths_m": breakup["lengths_m"],
        "fragment_masses_kg": breakup["masses_kg"],
        "thermal": {key: value for key, value in thermal.items() if key not in ("survivors", "histories")},
        "thermal_histories": thermal["histories"],
        "ledger": ledger,
        "wake_representative": representative,
        "fragmentation_driver": {
            "race_velocity_m_s": config.race_velocity_m_s,
            "turbulence_intensity": config.turbulence_intensity,
            "u_rms_m_s": rms,
            "resolved_lattice_turbulence_intensity": resolved_ti,
        },
    }


def follower_ship_impact(
    *,
    initial_mass_kg: float,
    surviving_mass_kg: float,
    melted_mass_kg: float,
    mean_floe_diameter_m: float,
    reference_floe_diameter_m: float,
    channel_width_m: float,
    ice_thickness_m: float,
    track_length_m: float,
) -> dict[str, Any]:
    """Estimate the ice-related impact of the race on a following ship.

    The melted fraction directly removes ice from the channel; the surviving
    ice is present as smaller floes, which lowers the floe-size dependent share
    of the resistance. Both effects are reported as separate contributions so
    the assumption behind each number is visible.
    """
    channel_width = _finite("channel_width_m", channel_width_m, positive=True)
    thickness = _finite("ice_thickness_m", ice_thickness_m, positive=True)
    track = _finite("track_length_m", track_length_m, positive=True)
    mean_floe = _finite("mean_floe_diameter_m", mean_floe_diameter_m, positive=True)
    reference_floe = _finite("reference_floe_diameter_m", reference_floe_diameter_m, positive=True)
    total = _finite("initial_mass_kg", initial_mass_kg, positive=True)
    surviving = _finite("surviving_mass_kg", surviving_mass_kg, nonnegative=True)
    melted = _finite("melted_mass_kg", melted_mass_kg, nonnegative=True)
    channel_area = channel_width * track
    ice_area_equivalent = surviving / (917.0 * thickness)
    return {
        "melted_fraction": melted / total,
        "ice_area_concentration": ice_area_equivalent / channel_area,
        "mean_floe_diameter_m": mean_floe,
        "floe_size_ratio": mean_floe / reference_floe,
        "resistance_relief_from_melting": melted / total,
        "resistance_relief_from_floe_refinement": max(0.0, 1.0 - mean_floe / reference_floe),
    }