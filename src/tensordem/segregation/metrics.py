"""Quantities extracted from a bidisperse granular bed: segregation index,
rise velocity, concentration profiles and the mass/moment bookkeeping that
guards against algorithmic artefacts.

Conventions
-----------
* The conserved measure of a particle is its **area** (2-D slice of uniform
  thickness) or its **volume** (3-D box); ``radii`` fixes it.
* ``phi`` always denotes a solid measure fraction: for the quasi-2-D channel it
  is the area fraction, which equals the volume fraction of a uniform-thickness
  slice.
* The segregation index of the campaign definition is

  .. math:: S_{fine} = \\frac{\\phi_{fine}^{top} - \\phi_{fine}^{nom}}
                            {\\phi_{fine}^{nom}}

  Under the Brazil-nut mechanism the *large* species rises, so ``S_fine < 0``.
  The campaign criterion "S increases with d_l/d_s" is therefore evaluated on
  the **Brazil-nut index** ``S_bn = -S_fine``, i.e. the relative enrichment of
  the large species in the top layer, which is exactly

  .. math:: S_{bn} = \\frac{\\phi_{large}^{top} - \\phi_{large}^{nom}}
                          {\\phi_{large}^{nom}}

  only when the normalisation uses the complementary nominal fraction.  Both
  numbers are always reported together so the sign convention cannot be
  mistaken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "particle_measure",
    "BedStatistics",
    "segregation_indices",
    "rise_velocity",
    "mass_moment_audit",
    "zonal_masses",
]


def particle_measure(radii: np.ndarray, dim: int = 2) -> np.ndarray:
    """Conserved per-particle measure: area (2-D) or volume (3-D)."""
    rad = np.asarray(radii, dtype=np.float64).reshape(-1)
    if dim == 2:
        return math.pi * rad**2
    if dim == 3:
        return 4.0 / 3.0 * math.pi * rad**3
    raise ValueError("dim must be 2 or 3")


@dataclass
class BedStatistics:
    """Online, memory-light accumulation of the bed state.

    Sampling many snapshots and *accumulating* (instead of storing positions)
    keeps a 10^5-particle, 10^5-step run inside a few kilobytes of statistics
    while still giving time-averaged profiles: the top-layer fraction of a
    strongly vibrated bed is a violently fluctuating quantity and an
    unaveraged snapshot is worthless as a segregation measure.
    """

    z_edges: np.ndarray
    x_edges: np.ndarray
    n_samples: int = 0
    z_area: np.ndarray = field(default=None)      # area per z bin
    z_large: np.ndarray = field(default=None)     # large-species area per z bin
    z_moment: np.ndarray = field(default=None)    # sum(area * z) per z bin
    x_area: np.ndarray = field(default=None)      # area per x zone
    x_large: np.ndarray = field(default=None)
    x_moment: np.ndarray = field(default=None)    # sum(area * z)
    x_moment2: np.ndarray = field(default=None)   # sum(area * z^2)
    x_z_moment: np.ndarray = field(default=None)  # sum(area * x)
    large_z_sum: float = 0.0
    large_count: int = 0
    small_z_sum: float = 0.0
    small_count: int = 0
    total_measure: float = 0.0
    total_moment: float = 0.0

    def __post_init__(self) -> None:
        nz = len(self.z_edges) - 1
        nx = len(self.x_edges) - 1
        if nz < 2 or nx < 1:
            raise ValueError("need at least 2 z bins and 1 x zone")
        if self.z_area is None:
            self.z_area = np.zeros(nz)
        if self.z_large is None:
            self.z_large = np.zeros(nz)
        if self.z_moment is None:
            self.z_moment = np.zeros(nz)
        if self.x_area is None:
            self.x_area = np.zeros(nx)
        if self.x_large is None:
            self.x_large = np.zeros(nx)
        if self.x_moment is None:
            self.x_moment = np.zeros(nx)
        if self.x_moment2 is None:
            self.x_moment2 = np.zeros(nx)
        if self.x_z_moment is None:
            self.x_z_moment = np.zeros(nx)

    def add(self, pos: np.ndarray, measure: np.ndarray, is_large: np.ndarray) -> None:
        """Accumulate one snapshot."""
        p = np.asarray(pos, dtype=np.float64)
        m = np.asarray(measure, dtype=np.float64).reshape(-1)
        lg = np.asarray(is_large, dtype=bool).reshape(-1)
        if p.shape[0] != m.size or p.shape[0] != lg.size:
            raise ValueError("pos, measure and is_large must agree in length")
        z = p[:, 1]
        x = p[:, 0]
        iz = np.clip(np.digitize(z, self.z_edges) - 1, 0, len(self.z_area) - 1)
        ix = np.clip(np.digitize(x, self.x_edges) - 1, 0, len(self.x_area) - 1)
        np.add.at(self.z_area, iz, m)
        np.add.at(self.z_large, iz, m * lg)
        np.add.at(self.z_moment, iz, m * z)
        np.add.at(self.x_area, ix, m)
        np.add.at(self.x_large, ix, m * lg)
        np.add.at(self.x_moment, ix, m * z)
        np.add.at(self.x_moment2, ix, m * z * z)
        np.add.at(self.x_z_moment, ix, m * x)
        self.large_z_sum += float(z[lg].sum())
        self.large_count += int(lg.sum())
        self.small_z_sum += float(z[~lg].sum())
        self.small_count += int((~lg).sum())
        self.total_measure += float(m.sum())
        self.total_moment += float((m * z).sum())
        self.n_samples += 1

    # ------------------------------------------------------------------ views
    def mean_z(self, is_large: bool) -> float:
        if is_large:
            return self.large_z_sum / max(self.large_count, 1)
        return self.small_z_sum / max(self.small_count, 1)

    def profile(self) -> dict[str, np.ndarray]:
        """Time-averaged solid fraction and fine/large split per z bin."""
        width = np.diff(self.z_edges)
        area = self.z_area / max(self.n_samples, 1)
        large = self.z_large / max(self.n_samples, 1)
        return {
            "z_edges": self.z_edges,
            "z_centres": 0.5 * (self.z_edges[1:] + self.z_edges[:-1]),
            "solid_fraction": area / width,
            "large_fraction_of_solid": np.divide(
                large, area, out=np.zeros_like(large), where=area > 0.0
            ),
            "area": area,
        }


def _cumulative_from_top(z_centres: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Cumulative area measured from the top of the profile downwards."""
    return np.cumsum(area[::-1])[::-1]


def segregation_indices(
    stats: BedStatistics,
    *,
    top_fraction: float = 0.25,
    h_ref_m: float | None = None,
    dilute_factor: float = 0.15,
) -> dict[str, float]:
    """Segregation indices of a bed, from time-averaged statistics.

    ``top_fraction`` is the fraction of the bed height, measured upwards from the
    floor, that defines the "top layer".

    Two ways of fixing the bed height are supported and both are reported:

    ``h_ref_m`` (preferred, used by the campaign)
        the settled bed height measured *before* the drive is switched on.  It is
        a fixed geometric reference, so ``S`` of a dilated bed and ``S`` of a
        compact bed are the same quantity and can be compared across
        ``Gamma``.
    profile detection
        the highest height at which the time-averaged solid fraction still
        exceeds ``dilute_factor`` times the bulk fraction of the lower half of
        the bed.  Used as a cross-check, because it follows the bed while it
        dilates.
    """
    if not 0.0 < top_fraction < 1.0:
        raise ValueError("top_fraction must lie in (0, 1)")
    if stats.n_samples < 1:
        raise ValueError("no samples accumulated")
    area = stats.z_area / stats.n_samples
    large = stats.z_large / stats.n_samples
    total = float(area.sum())
    if total <= 0.0:
        raise ValueError("empty bed")
    z_centres = 0.5 * (stats.z_edges[1:] + stats.z_edges[:-1])
    width = np.diff(stats.z_edges)
    phi = area / width
    occupied = np.flatnonzero(area > 0.0)
    if occupied.size == 0:
        raise ValueError("no occupied bins")
    z_floor = float(stats.z_edges[0])
    # cross-check: profile-detected surface
    z_top_occupied = float(stats.z_edges[occupied[-1] + 1])
    lower = z_centres <= z_floor + 0.5 * (z_top_occupied - z_floor)
    phi_bulk = float(np.max(phi[lower])) if lower.any() else float(np.max(phi))
    solid = np.flatnonzero(phi >= dilute_factor * phi_bulk)
    h_profile = float(stats.z_edges[solid[-1] + 1] - z_floor) if solid.size else z_top_occupied
    if h_ref_m is not None:
        if not math.isfinite(h_ref_m) or h_ref_m <= 0.0:
            raise ValueError("h_ref_m must be finite and positive")
        h_bed = float(h_ref_m)
    else:
        h_bed = h_profile
    z_cut = z_floor + (1.0 - top_fraction) * h_bed
    in_top = z_centres >= z_cut
    top_area = float(area[in_top].sum())
    top_large = float(large[in_top].sum())
    if top_area <= 0.0:
        raise ValueError(
            f"top layer contains no solid measure (h_bed={h_bed:.4f} m, z_cut={z_cut:.4f} m)"
        )
    phi_fine_top = 1.0 - top_large / top_area
    phi_fine_nom = 1.0 - float(large.sum()) / total
    if not 0.0 < phi_fine_nom < 1.0:
        raise ValueError("nominal fine fraction is degenerate")
    s_fine = (phi_fine_top - phi_fine_nom) / phi_fine_nom
    phi_large_nom = 1.0 - phi_fine_nom
    phi_large_top = top_large / top_area
    s_bn = (phi_large_top - phi_large_nom) / phi_large_nom
    return {
        "h_bed_m": float(h_bed),
        "h_profile_m": float(h_profile),
        "z_cut_m": float(z_cut),
        "phi_fine_top": float(phi_fine_top),
        "phi_fine_nominal": float(phi_fine_nom),
        "phi_large_top": float(phi_large_top),
        "phi_large_nominal": float(phi_large_nom),
        "phi_top_over_bulk": float(top_area / (top_fraction * h_bed) / max(phi_bulk, 1e-30)),
        "S_fine": float(s_fine),
        "S_bn": float(s_bn),
        "top_area_share": float(top_area / total),
        "n_top_bins": int(in_top.sum()),
        "bin_width_m": float(np.mean(width)),
    }


def rise_velocity(
    times: np.ndarray,
    mean_z_large: np.ndarray,
    *,
    t_start: float,
    t_end: float | None = None,
) -> dict[str, float]:
    """Least-squares Brazil-nut rise speed ``d<z_large>/dt`` over a window.

    Using the *mean height of the large species* (a Lagrangian average over
    tracers) instead of a per-particle velocity histogram avoids the bias that
    comes from ballistic particles crossing the sampling bins.
    """
    t = np.asarray(times, dtype=np.float64)
    z = np.asarray(mean_z_large, dtype=np.float64)
    if t.shape != z.shape or t.ndim != 1:
        raise ValueError("times and mean_z_large must be 1-D and of equal length")
    if t.size < 3:
        raise ValueError("at least three samples are required")
    if t_end is None:
        t_end = float(t[-1])
    keep = (t >= t_start) & (t <= t_end)
    if keep.sum() < 3:
        raise ValueError("fitting window contains fewer than three samples")
    tt = t[keep]
    zz = z[keep]
    a, b = np.polyfit(tt, zz, 1)
    pred = a * tt + b
    ss_res = float(np.sum((zz - pred) ** 2))
    ss_tot = float(np.sum((zz - zz.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {
        "v_up_m_s": float(a),
        "z0_m": float(b),
        "r2": float(r2),
        "n_samples": int(keep.sum()),
        "t_start_s": float(tt[0]),
        "t_end_s": float(tt[-1]),
        "z_change_m": float(zz[-1] - zz[0]),
    }


def mass_moment_audit(
    stats: BedStatistics,
    *,
    density: float,
    thickness: float = 1.0,
    reference_measure: float | None = None,
) -> dict[str, float]:
    """Mass conservation and centre-of-mass consistency of the accumulation.

    Three independent numbers are compared:

    1. total solid measure accumulated per-snapshot (``sum(area)``),
    2. total solid measure accumulated per z bin,
    3. the prescribed specimen measure, if supplied.

    and the centre of mass is reconstructed from the z-bin moments only
    (``sum(m_z * z) / M`` with the *exact* per-bin moment), which is the closed
    form used downstream by the ship-side zone mapping.
    """
    if stats.n_samples < 1:
        raise ValueError("no samples accumulated")
    snapshot_total = stats.total_measure / stats.n_samples
    per_bin_total = float(stats.z_area.sum()) / stats.n_samples
    per_zone_total = float(stats.x_area.sum()) / stats.n_samples
    scale = density * thickness
    out = {
        "samples": int(stats.n_samples),
        "measure_snapshot": snapshot_total,
        "measure_per_z_bin": per_bin_total,
        "measure_per_x_zone": per_zone_total,
        "mass_kg": snapshot_total * scale,
        "mass_conservation_rel": abs(per_bin_total - snapshot_total) / snapshot_total,
        "mass_conservation_zone_rel": abs(per_zone_total - snapshot_total) / snapshot_total,
    }
    if reference_measure is not None:
        out["mass_conservation_vs_reference_rel"] = (
            abs(snapshot_total - reference_measure) / reference_measure
        )
    z_com_direct = stats.total_moment / stats.total_measure
    z_com_bins = float(stats.z_moment.sum()) / float(stats.z_area.sum())
    x_com_bins = float(stats.x_z_moment.sum()) / float(stats.x_area.sum())
    out.update({
        "z_com_m": z_com_direct,
        "z_com_from_bins_m": z_com_bins,
        "z_com_consistency_rel": abs(z_com_bins - z_com_direct) / abs(z_com_direct),
        "x_com_from_zones_m": x_com_bins,
    })
    # zone-moment reconstruction (what the transfer module consumes)
    mz = float(stats.x_moment.sum()) / stats.n_samples * scale
    mx = float(stats.x_z_moment.sum()) / stats.n_samples * scale
    m_tot = snapshot_total * scale
    out["z_com_from_zone_moments_m"] = mz / m_tot
    out["x_com_from_zone_moments_m"] = mx / m_tot
    out["z_com_zone_rel"] = abs(mz / m_tot - z_com_direct) / abs(z_com_direct)
    return out


def zonal_masses(
    stats: BedStatistics,
    *,
    density: float,
    thickness: float = 1.0,
) -> dict[str, np.ndarray]:
    """Per-zone (x) masses and moments, time-averaged.

    Returns arrays over the ``x`` zones: mass, mean height, second moment
    ``sum(m z^2)`` and the mean longitudinal position, i.e. everything the
    ship-side transfer needs to rebuild ``TCG``/``VCG`` and the free-surface
    inertia of each zone.
    """
    if stats.n_samples < 1:
        raise ValueError("no samples accumulated")
    scale = density * thickness / stats.n_samples
    mass = stats.x_area * scale
    large = stats.x_large * scale
    z_moment = stats.x_moment * scale
    z2 = stats.x_moment2 * scale
    x_moment = stats.x_z_moment * scale
    return {
        "x_edges": stats.x_edges,
        "mass_kg": mass,
        "large_mass_kg": large,
        "total_mass_kg": float(mass.sum()),
        "z_moment_kg_m": z_moment,
        "z2_moment_kg_m2": z2,
        "x_moment_kg_m": x_moment,
        "z_com_m": np.divide(z_moment, mass, out=np.zeros_like(mass), where=mass > 0.0),
        "x_com_m": np.divide(x_moment, mass, out=np.zeros_like(mass), where=mass > 0.0),
    }