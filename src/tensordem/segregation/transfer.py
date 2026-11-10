"""From a segregated granular cargo bed to ship-stability and structure quantities.

The DEM side of this module knows nothing about ships; this file is the *only*
place where a granular state is turned into the quantities a stability or
structure tool consumes:

===================  ==========================================================
quantity             definition used here
===================  ==========================================================
``mass``             zone solid mass, ``(1 - eps) * rho_solid * V_zone``
``VCG`` (``KG``)     zone mass moment ``sum(m_j z_j) / M``
``LCG`` / ``TCG``    zone mass moments about the hold origin
``dGM``              ``-dKG_total - dFSC`` with ``dKG_total = (M/Delta) dVCG``
``FSC``              free-surface correction ``sum(rho_j I_j) / Delta`` of the
                     cargo surface, with a void-fraction and repose-angle
                     attenuation (two bounds, see :func:`free_surface_bounds`)
``floor pressure``   zone-resolved base load ``F_j / (l_j * b)``, mean and peak
===================  ==========================================================

CI-free, unit-tested closed forms
---------------------------------
All mappings are linear in the zone moments, so they can be assembled from
partial sums (shards) without loss.  :func:`zone_moment_check` proves it: it
recomputes the global moments from a two-way split of the zones and compares
with the direct accumulation.

The DEM record is a **quasi-two-dimensional slice** (unit thickness in the
transverse direction) only when the campaign recorded it as such; the mapping
therefore asks for the transverse breadth explicitly and documents the
uniform-across-breadth assumption instead of hiding it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "HoldGeometry",
    "CargoDistribution",
    "cargo_state",
    "free_surface_bounds",
    "gm_mapping",
    "floor_pressure",
    "zone_moment_check",
    "distribution_from_record",
    "slice_to_hold",
]


@dataclass(frozen=True)
class HoldGeometry:
    """A rectangular cargo-hold compartment (SI units).

    ``n_longitudinal`` x ``n_transverse`` zones tile the floor; the zone grid is
    the only geometric input, so any compartment (hold, hopper, barge tank) can
    be described without a ship-specific database.
    """

    length_m: float
    breadth_m: float
    depth_m: float
    n_longitudinal: int = 8
    n_transverse: int = 1
    repose_angle_deg: float = 35.0

    def __post_init__(self) -> None:
        for name in ("length_m", "breadth_m", "depth_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.n_longitudinal < 1 or self.n_transverse < 1:
            raise ValueError("zone counts must be >= 1")
        if not 0.0 < self.repose_angle_deg < 90.0:
            raise ValueError("repose_angle_deg must lie in (0, 90)")

    @property
    def n_zones(self) -> int:
        return self.n_longitudinal * self.n_transverse

    @property
    def floor_area_m2(self) -> float:
        return self.length_m * self.breadth_m

    @property
    def zone_length_m(self) -> float:
        return self.length_m / self.n_longitudinal

    @property
    def zone_breadth_m(self) -> float:
        return self.breadth_m / self.n_transverse

    @property
    def zone_area_m2(self) -> float:
        return self.floor_area_m2 / self.n_zones


@dataclass
class CargoDistribution:
    """Zone-resolved mass distribution of one compartment.

    ``mass``, ``z_moment``, ``x_moment`` and ``y_moment`` are SI (kg, kg m) and
    are the *only* inputs to the stability mapping; ``z2_moment`` (kg m^2) is
    additionally needed for the vertical spread used by the structure side.
    Zones are ordered ``index = i_longitudinal * n_transverse + j_transverse``.
    """

    geometry: HoldGeometry
    mass_kg: np.ndarray
    z_moment_kg_m: np.ndarray
    x_moment_kg_m: np.ndarray
    y_moment_kg_m: np.ndarray
    z2_moment_kg_m2: np.ndarray
    base_load_N_m: np.ndarray | None = None
    base_load_peak_N_m: np.ndarray | None = None
    label: str = "cargo"

    def __post_init__(self) -> None:
        n = self.geometry.n_zones
        for name in ("mass_kg", "z_moment_kg_m", "x_moment_kg_m",
                     "y_moment_kg_m", "z2_moment_kg_m2"):
            arr = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if arr.size != n:
                raise ValueError(f"{name} must have {n} entries")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must be finite")
            setattr(self, name, arr)
        if (self.mass_kg < 0.0).any():
            raise ValueError("zone masses must be nonnegative")
        for name in ("base_load_N_m", "base_load_peak_N_m"):
            arr = getattr(self, name)
            if arr is not None:
                arr = np.asarray(arr, dtype=np.float64).reshape(-1)
                if arr.size != n:
                    raise ValueError(f"{name} must have {n} entries")
                setattr(self, name, arr)

    @property
    def total_mass_kg(self) -> float:
        return float(self.mass_kg.sum())


def cargo_state(dist: CargoDistribution) -> dict[str, float]:
    """Mass, centres and void-free vertical spread of a zone distribution."""
    m = dist.mass_kg
    total = float(m.sum())
    if total <= 0.0:
        raise ValueError("the compartment has no cargo mass")
    vcg = float(dist.z_moment_kg_m.sum() / total)
    lcg = float(dist.x_moment_kg_m.sum() / total)
    tcg = float(dist.y_moment_kg_m.sum() / total)
    z_var = float(dist.z2_moment_kg_m2.sum() / total) - vcg**2
    return {
        "total_mass_kg": total,
        "vcg_m": vcg,
        "lcg_m": lcg,
        "tcg_m": tcg,
        "z_spread_m": math.sqrt(max(z_var, 0.0)),
        "fill_depth_m": float(np.max(dist.z_moment_kg_m / np.maximum(m, 1e-30))),
    }


def free_surface_bounds(
    dist: CargoDistribution,
    *,
    void_fraction: float,
    solid_density: float = 1000.0,
) -> dict[str, float]:
    """Free-surface correction of a granular cargo: two explicit bounds.

    The liquid free-surface inertia of a rectangular compartment is
    ``I = l b^3 / 12`` (heel about the longitudinal axis).  A granular cargo can
    only rearrange its surface within its angle of repose, which caps the wedge
    that can shift: for a surface slope ``tan(theta_r)`` the effective inertia is
    ``I_g = I * tan(theta_r)`` (equivalence of the repose-limited wedge moment
    with the liquid wedge moment, clamped at the liquid value).  Both numbers are
    returned so a stability assessment can quote the interval instead of a single
    unwarranted number.

    ``void_fraction`` is the measured bed void ratio ``1 - phi``: the void
    volume carries no mass, so it enters the moment through the bulk density
    ``rho_bulk = (1 - eps) rho_solid``.
    """
    geo = dist.geometry
    if not 0.0 <= void_fraction < 1.0:
        raise ValueError("void_fraction must lie in [0, 1)")
    rho_bulk = (1.0 - void_fraction) * solid_density
    inertia = geo.length_m * geo.breadth_m**3 / 12.0
    kappa = min(1.0, math.tan(math.radians(geo.repose_angle_deg)))
    moment_liquid = rho_bulk * inertia * kappa / kappa  # = rho * I  (liquid bound)
    moment_grain = rho_bulk * inertia * kappa
    return {
        "void_fraction": float(void_fraction),
        "bulk_density_kg_m3": float(rho_bulk),
        "free_surface_inertia_m4": float(inertia),
        "repose_attenuation": float(kappa),
        "free_surface_moment_kg_m": float(moment_liquid),
        "free_surface_moment_grain_kg_m": float(moment_grain),
    }


def gm_mapping(
    before: CargoDistribution,
    after: CargoDistribution,
    *,
    displacement_t: float,
    km_m: float,
    solid_density: float = 1000.0,
    void_before: float,
    void_after: float,
) -> dict[str, float]:
    """Closed-form mapping from a segregation state to ``dVCG``/``dTCG``/``dGM``.

    ``before`` is the homogeneous reference fill (same mass, no segregation) and
    ``after`` the segregated state.  The vessel-level consequence of moving cargo
    inside one compartment is

    .. math:: \\Delta KG = \\frac{M}{\\Delta}\\,\\Delta VCG,\\qquad
              \\Delta GM = -\\Delta KG - \\Delta FSC,

    with ``FSC = (rho_bulk I)/\\Delta`` from :func:`free_surface_bounds`.  ``KM``
    is injected (a metacentric height fixed by the hull form) and only enters the
    absolute ``GM``; all reported *changes* are independent of it.
    """
    if not math.isfinite(displacement_t) or displacement_t <= 0.0:
        raise ValueError("displacement_t must be finite and positive")
    if not math.isfinite(km_m):
        raise ValueError("km_m must be finite")
    if abs(before.total_mass_kg - after.total_mass_kg) > 1e-6 * max(before.total_mass_kg, 1.0):
        raise ValueError("before/after must carry the same cargo mass (mass is conserved)")
    s0 = cargo_state(before)
    s1 = cargo_state(after)
    delta = displacement_t * 1000.0  # kg
    m_cargo = s1["total_mass_kg"]
    d_vcg = s1["vcg_m"] - s0["vcg_m"]
    d_tcg = s1["tcg_m"] - s0["tcg_m"]
    d_lcg = s1["lcg_m"] - s0["lcg_m"]
    d_kg = m_cargo / delta * d_vcg
    d_lcg_vessel = m_cargo / delta * d_lcg
    d_tcg_vessel = m_cargo / delta * d_tcg
    fsc0 = free_surface_bounds(before, void_fraction=void_before, solid_density=solid_density)
    fsc1 = free_surface_bounds(after, void_fraction=void_after, solid_density=solid_density)
    d_fsc_solid = (fsc1["free_surface_moment_kg_m"] - fsc0["free_surface_moment_kg_m"]) / delta
    d_fsc_grain = (
        fsc1["free_surface_moment_grain_kg_m"] - fsc0["free_surface_moment_grain_kg_m"]
    ) / delta
    return {
        "cargo_mass_kg": float(m_cargo),
        "d_vcg_m": float(d_vcg),
        "d_tcg_m": float(d_tcg),
        "d_lcg_m": float(d_lcg),
        "d_kg_vessel_m": float(d_kg),
        "d_lcg_vessel_m": float(d_lcg_vessel),
        "d_tcg_vessel_m": float(d_tcg_vessel),
        "fsc_liquid_bound_m": float(fsc1["free_surface_moment_kg_m"] / delta),
        "fsc_grain_bound_m": float(fsc1["free_surface_moment_grain_kg_m"] / delta),
        "d_fsc_liquid_m": float(d_fsc_solid),
        "d_fsc_grain_m": float(d_fsc_grain),
        "d_gm_liquid_m": float(-d_kg - d_fsc_solid),
        "d_gm_grain_m": float(-d_kg - d_fsc_grain),
        "gm_before_m": float(km_m - s0["vcg_m"] - fsc0["free_surface_moment_kg_m"] / delta),
        "gm_after_liquid_m": float(km_m - s1["vcg_m"] - d_kg - fsc1["free_surface_moment_kg_m"] / delta),
        "gm_after_grain_m": float(km_m - s1["vcg_m"] - d_kg - fsc1["free_surface_moment_grain_kg_m"] / delta),
        "z_spread_before_m": s0["z_spread_m"],
        "z_spread_after_m": s1["z_spread_m"],
    }


def floor_pressure(dist: CargoDistribution) -> dict[str, object]:
    """Zone floor pressure from the DEM base load, with the global load check.

    ``base_load_N_m`` is the time-averaged normal load per unit transverse
    length recorded on the floor of each zone (2-D slice convention).  Dividing
    by the zone length gives Pa for a unit-thickness slice, which is what a
    structural panel sees locally; ``total_load_check`` compares the sum with the
    static cargo weight ``M g`` so a missing contact cannot go unnoticed.
    """
    geo = dist.geometry
    if dist.base_load_N_m is None:
        raise ValueError("the distribution carries no floor load")
    load = np.asarray(dist.base_load_N_m, dtype=np.float64)
    peak = (np.asarray(dist.base_load_peak_N_m, dtype=np.float64)
            if dist.base_load_peak_N_m is not None else load)
    length = geo.zone_length_m
    static_weight = dist.total_mass_kg * 9.81
    total = float(load.sum() * length)
    return {
        "zone_mean_pressure_Pa": (load / length).tolist(),
        "zone_peak_pressure_Pa": (peak / length).tolist(),
        "zone_peak_over_mean": (peak / np.maximum(load, 1e-30)).tolist(),
        "total_load_N": total,
        "static_weight_N": static_weight,
        "total_load_rel_error": float((total - static_weight) / static_weight),
        "peak_pressure_Pa": float(np.max(peak / length)),
        "uniform_pressure_Pa": float(static_weight / (geo.length_m)),
    }


def zone_moment_check(dist: CargoDistribution) -> dict[str, float]:
    """Prove the closed-form moments are additive over zone shards.

    The global moments are recomputed from a two-way split of the zone arrays
    and compared with the direct accumulation.  Machine-precision agreement is
    what makes the mapping safe to evaluate on sharded partial results.
    """
    m = dist.mass_kg
    half = len(m) // 2
    parts = [slice(0, half), slice(half, len(m))]
    total_mass = 0.0
    total_z = 0.0
    total_x = 0.0
    total_y = 0.0
    for part in parts:
        total_mass += float(m[part].sum())
        total_z += float(dist.z_moment_kg_m[part].sum())
        total_x += float(dist.x_moment_kg_m[part].sum())
        total_y += float(dist.y_moment_kg_m[part].sum())
    ref_mass = dist.total_mass_kg
    ref_z = float(dist.z_moment_kg_m.sum())
    ref_x = float(dist.x_moment_kg_m.sum())
    ref_y = float(dist.y_moment_kg_m.sum())
    scale = max(abs(ref_mass), abs(ref_z), abs(ref_x), abs(ref_y), 1e-30)
    return {
        "mass_shard_rel": abs(total_mass - ref_mass) / max(abs(ref_mass), 1e-30),
        "z_moment_shard_rel": abs(total_z - ref_z) / max(abs(ref_z), 1e-30),
        "x_moment_shard_rel": abs(total_x - ref_x) / max(abs(ref_x), 1e-30),
        "y_moment_shard_rel": abs(total_y - ref_y) / max(abs(ref_y), 1e-30),
        "max_rel": max(abs(total_mass - ref_mass), abs(total_z - ref_z),
                       abs(total_x - ref_x), abs(total_y - ref_y)) / scale,
    }


def slice_to_hold(
    *,
    zone_mass_slice_kg: np.ndarray,
    zone_z_moment_slice_kg_m: np.ndarray,
    zone_x_moment_slice_kg_m: np.ndarray,
    zone_x_com_m: np.ndarray,
    zone_z_com_m: np.ndarray,
    geometry: HoldGeometry,
    bed_width_m: float,
    base_load_N_m: np.ndarray | None = None,
    base_load_peak_N_m: np.ndarray | None = None,
    label: str = "cargo",
) -> CargoDistribution:
    """Map a 2-D slice distribution onto a 3-D compartment.

    The slice carries mass per unit transverse length; extruding it over the
    breadth ``B`` gives the compartment distribution.  The transverse position is
    taken at the zone centre (``n_transverse == 1`` puts everything at
    ``B/2``), which is exact for a cargo that is uniform across the breadth and
    is the stated assumption whenever the campaign only ran a 2-D slice.
    """
    geo = geometry
    n = geo.n_zones
    if geo.n_transverse != 1:
        raise ValueError(
            "slice_to_hold assumes a single transverse zone; use a 3-D bed for more"
        )
    slice_mass = np.asarray(zone_mass_slice_kg, dtype=np.float64).reshape(-1)
    if slice_mass.size != geo.n_longitudinal:
        raise ValueError(
            f"the slice has {slice_mass.size} zones but the compartment has "
            f"{geo.n_longitudinal} longitudinal zones"
        )
    if abs(float(np.sum(zone_x_com_m)) <= 0.0) and slice_mass.sum() > 0.0:
        raise ValueError("slice zone centres are degenerate")
    scale = geo.breadth_m * (geo.zone_length_m / (bed_width_m / geo.n_longitudinal))
    mass = slice_mass * scale
    z_moment = np.asarray(zone_z_moment_slice_kg_m, dtype=np.float64) * scale
    x_local = (np.asarray(zone_x_com_m, dtype=np.float64) - np.min(zone_x_com_m)) * (
        geo.zone_length_m / (bed_width_m / geo.n_longitudinal)
    )
    x_moment = mass * x_local
    z_local = np.asarray(zone_z_com_m, dtype=np.float64) * scale
    z2 = z_local * np.asarray(zone_z_com_m, dtype=np.float64) * geo.breadth_m
    y_moment = mass * (geo.breadth_m / 2.0)
    return CargoDistribution(
        geometry=geo,
        mass_kg=mass,
        z_moment_kg_m=z_moment,
        x_moment_kg_m=x_moment,
        y_moment_kg_m=y_moment,
        z2_moment_kg_m2=z2,
        base_load_N_m=None if base_load_N_m is None else np.asarray(base_load_N_m, dtype=np.float64),
        base_load_peak_N_m=None if base_load_peak_N_m is None
        else np.asarray(base_load_peak_N_m, dtype=np.float64),
        label=label,
    )


def distribution_from_record(record: dict, geometry: HoldGeometry) -> CargoDistribution:
    """Build a :class:`CargoDistribution` from a campaign record.

    Expects the ``stats`` block written by
    :func:`tensordem.segregation.campaign.run_case` (time-averaged zone area and
    moments for a unit-thickness slice with ``density * thickness = 1000``) and
    converts it to compartment masses.
    """
    stats = record["stats"]
    spec = record["spec"]
    area = np.asarray(stats["x_area"], dtype=np.float64) / stats["n_samples"]
    z_mom = np.asarray(stats["x_moment"], dtype=np.float64) / stats["n_samples"]
    x_mom = np.asarray(stats["x_z_moment"], dtype=np.float64) / stats["n_samples"]
    z2 = np.asarray(stats["x_moment2"], dtype=np.float64) / stats["n_samples"]
    x_edges = np.asarray(stats["x_edges"], dtype=np.float64)
    n_zone = len(area)
    width = float(x_edges[-1] - x_edges[0])
    if geometry.n_longitudinal != n_zone:
        raise ValueError(
            f"the record has {n_zone} zones, the compartment wants {geometry.n_longitudinal}"
        )
    factor = geometry.breadth_m * (width / n_zone) / (width / n_zone)  # = breadth
    mass = area * 1000.0 * factor
    z_moment = z_mom * 1000.0 * factor
    x_local = x_mom / np.maximum(area, 1e-30) - geometry.length_m / 2.0
    x_moment = mass * x_local * (geometry.length_m / width)
    z_com = z_mom / np.maximum(area, 1e-30)
    z2_local = z2 / np.maximum(area, 1e-30)
    y_moment = mass * geometry.breadth_m / 2.0
    base = record.get("base_load", {})
    base_load = (
        np.asarray(base["mean_pressure_Pa"], dtype=np.float64) * geometry.zone_length_m
        if base else None
    )
    base_peak = (
        np.asarray(base["peak_pressure_Pa"], dtype=np.float64) * geometry.zone_length_m
        if base else None
    )
    return CargoDistribution(
        geometry=geometry,
        mass_kg=mass,
        z_moment_kg_m=z_moment,
        x_moment_kg_m=x_moment,
        y_moment_kg_m=y_moment,
        z2_moment_kg_m2=z2_local * mass,
        base_load_N_m=base_load,
        base_load_peak_N_m=base_peak,
        label=f"seg_{spec['name']}",
    )