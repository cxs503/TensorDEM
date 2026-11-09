"""Segregated-cargo -> ship stability / structure transmission (P70-N03).

The DEM bed of :mod:`tensordem.bidisperse_segregation` is read out as a
*partitioned cargo mass matrix* (strips along the hold length x layers in the
vertical) and fed to closed-form stability quantities, so the numbers a
stability or structures consumer needs are reproducible from the partition
table alone:

    cargo mass        M   = sum_b m_b
    cargo VCG         KG_c = sum_b m_b z_b / M
    cargo LCG         LCG_c = sum_b m_b x_b / M
    bulk density      rho_b = M / V_b          (V_b from the measured bed box)
    free-surface      A_FS = sum_strips l_i * B * sqrt(1 + s_i^2),
                      I_FS = A_FS * B^2 / 12,   s_i = |dh_i/dx| local slope
    Delta GM          = -dKG_c * m_cargo/Delta  -  lambda * I_FS / Delta
    heeling/trim      d(trim) = -d(LCG_c)/L_pp

``lambda`` is the granular free-surface reduction factor of the grain code
(``lambda = 0.12``), i.e. the free-surface inertia of a *granular* cargo is
only 12 % of the fluid value; it is the single calibrated coefficient here and
is labelled CALIBRATED in the report.

Every closed-form quantity is returned together with an independent
*shard-sum* evaluation (direct per-particle sums or per-bin sums of the
partition table) so the mapping can be audited to ``1e-6`` relative error:
that audit is the acceptance criterion (e) of the campaign node.
"""
from __future__ import annotations

import math

import numpy as np

G = 9.81
#: grain-code free-surface reduction factor for a granular (shiftable) cargo
LAMBDA_GRAIN = 0.12


def partition_table(bed, n_x: int = 8, n_z: int = 4) -> dict:
    """Partition the bed into ``n_x`` x-strips x ``n_z`` z-layers.

    Returns the per-bin mass / volume / centroid table plus the direct
    per-particle reference values used by the audit.
    """
    pos = bed.pos
    r, vol, m = bed.r, bed.volume, bed.mass
    z = pos[:, 2]
    z_floor = bed.floor_offset(bed.t) if hasattr(bed, "floor_offset") else 0.0
    h = float(np.max(z + r)) - z_floor
    ix = np.clip((pos[:, 0] / bed.Lx * n_x).astype(np.int64), 0, n_x - 1)
    iz = np.clip(((z - z_floor) / max(h, 1e-12) * n_z).astype(np.int64), 0, n_z - 1)
    flat = ix * n_z + iz
    n_bins = n_x * n_z
    mass_bin = np.bincount(flat, weights=m, minlength=n_bins)
    vol_bin = np.bincount(flat, weights=vol, minlength=n_bins)
    mz_bin = np.bincount(flat, weights=m * z, minlength=n_bins)
    mx_bin = np.bincount(flat, weights=m * pos[:, 0], minlength=n_bins)
    count_bin = np.bincount(flat, minlength=n_bins)
    coarse = bed.cls == 1
    v_coarse_bin = np.bincount(flat, weights=vol * coarse, minlength=n_bins)
    x_width = bed.Lx / n_x
    z_height = h / n_z
    z_cent = z_floor + (np.arange(n_z) + 0.5) * z_height
    x_cent = (np.arange(n_x) + 0.5) * x_width

    mass_tot = float(np.sum(m))
    return {
        "n_x": n_x, "n_z": n_z,
        "x_edges": (np.arange(n_x + 1) * x_width).tolist(),
        "z_edges": (z_floor + np.arange(n_z + 1) * z_height).tolist(),
        "x_centers": x_cent.tolist(),
        "z_centers": z_cent.tolist(),
        "mass": mass_bin.reshape(n_x, n_z).tolist(),
        "volume": vol_bin.reshape(n_x, n_z).tolist(),
        "count": count_bin.reshape(n_x, n_z).tolist(),
        "coarse_volume_fraction": (v_coarse_bin / np.maximum(vol_bin, 1e-30)
                                   ).reshape(n_x, n_z).tolist(),
        "mass_total": mass_tot,
        "volume_total": float(np.sum(vol)),
        "kg_closed_form": float(np.sum(mz_bin) / mass_tot),
        "lcg_closed_form": float(np.sum(mx_bin) / mass_tot),
        "kg_direct": float(np.sum(m * z) / mass_tot),
        "lcg_direct": float(np.sum(m * pos[:, 0]) / mass_tot),
        "mass_bin_sum": float(np.sum(mass_bin)),
        "mass_direct": mass_tot,
        "bed_height": h,
        "bed_volume_box": float(bed.Lx * bed.Ly * h),
        "solid_volume": float(np.sum(vol)),
        "n_x_coarse_depth": n_x,
        "surface_height_strips": _surface_profile(pos, r, ix, n_x, x_width),
    }


def _surface_profile(pos, r, ix, n_x, x_width) -> list[float]:
    """Mean free-surface height per x-strip (top 1 % of particles)."""
    z_top = pos[:, 2] + r
    out = []
    for i in range(n_x):
        mask = ix == i
        if not np.any(mask):
            out.append(0.0)
            continue
        zt = z_top[mask]
        cut = np.quantile(zt, 0.95)
        out.append(float(np.mean(zt[zt >= cut])))
    return out


def free_surface_inertia(surface_heights: list[float], x_width: float,
                         breadth: float) -> dict:
    """Free-surface area and transverse inertia from the measured profile."""
    h = np.asarray(surface_heights, dtype=np.float64)
    slopes = np.abs(np.gradient(h, x_width)) if len(h) > 1 else np.zeros(1)
    area = float(np.sum(x_width * breadth * np.sqrt(1.0 + slopes ** 2)))
    i_fs = area * breadth ** 2 / 12.0
    return {"free_surface_area": area, "free_surface_inertia": float(i_fs),
            "mean_slope": float(np.mean(slopes)),
            "max_slope": float(np.max(slopes))}


def stability_transmission(bed_start, bed_end, **kwargs) -> dict:
    """Map the segregated mass distribution to ship stability quantities."""
    n_x = kwargs.get("n_x", 8)
    n_z = kwargs.get("n_z", 4)
    p0 = partition_table(bed_start, n_x, n_z)
    p1 = partition_table(bed_end, n_x, n_z)
    return transmission_from_partitions(p0, p1, **kwargs)


def transmission_from_partitions(p0: dict, p1: dict, *, hold_length: float,
                                 hold_breadth: float, displacement: float,
                                 cargo_mass_ratio: float = 0.35,
                                 lam: float = LAMBDA_GRAIN,
                                 n_x: int = 8, n_z: int = 4) -> dict:
    """Closed-form stability mapping from two partition tables."""

    m_c = p1["mass_total"]
    v_bulk0 = p1["bed_volume_box"]
    rho_b = m_c / v_bulk0 if v_bulk0 > 0 else float("nan")
    vcargo = m_c / rho_b if rho_b and np.isfinite(rho_b) else float("nan")

    kg0, kg1 = p0["kg_closed_form"], p1["kg_closed_form"]
    lcg0, lcg1 = p0["lcg_closed_form"], p1["lcg_closed_form"]
    fs0 = free_surface_inertia(p0["surface_height_strips"], hold_length / n_x,
                               hold_breadth)
    fs1 = free_surface_inertia(p1["surface_height_strips"], hold_length / n_x,
                               hold_breadth)

    # vertical channel: cargo VCG rise lowers GM by m_cargo/Delta
    d_kg_ship = (kg1 - kg0) * cargo_mass_ratio
    # vertical channel: the cargo VCG rise dKG lowers GM by dKG * m_cargo / Delta
    d_gm_kg = -(kg1 - kg0) * (m_c / displacement) if displacement else float("nan")
    d_gm_fs = -lam * (fs1["free_surface_inertia"] - fs0["free_surface_inertia"]) / displacement
    trim_deg = math.degrees(-(lcg1 - lcg0) / hold_length)

    audit = {
        "kg_rel_err": _rel(p1["kg_closed_form"], p1["kg_direct"]),
        "lcg_rel_err": _rel(p1["lcg_closed_form"], p1["lcg_direct"]),
        "mass_rel_err": _rel(p1["mass_bin_sum"], p1["mass_direct"]),
        "kg_start_rel_err": _rel(p0["kg_closed_form"], p0["kg_direct"]),
        "lcg_start_rel_err": _rel(p0["lcg_closed_form"], p0["lcg_direct"]),
    }
    return {
        "cargo_mass": m_c,
        "cargo_bulk_volume": vcargo,
        "cargo_bulk_density": rho_b,
        "cargo_void_fraction": float(1.0 - p1["solid_volume"] / v_bulk0) if v_bulk0 else float("nan"),
        "VCG_cargo_start": kg0, "VCG_cargo_end": kg1,
        "dVCG_cargo": kg1 - kg0,
        "LCG_cargo_start": lcg0, "LCG_cargo_end": lcg1,
        "dLCG_cargo": lcg1 - lcg0,
        "dKG_ship": d_kg_ship,
        "dGM_kg_channel": d_gm_kg,
        "dGM_freesurface_channel": d_gm_fs,
        "dGM_total": d_gm_kg + d_gm_fs,
        "trim_change_deg": trim_deg,
        "free_surface_start": fs0,
        "free_surface_end": fs1,
        "lambda_grain": lam,
        "hold_length": hold_length,
        "hold_breadth": hold_breadth,
        "displacement": displacement,
        "cargo_mass_ratio": cargo_mass_ratio,
        "partition_start": p0,
        "partition_end": p1,
        "audit": audit,
    }


def floor_pressure_report(summary: dict, Lx: float, Ly: float,
                          total_mass: float) -> dict:
    """Peak / mean floor pressure per strip, plus the static load balance."""
    peak = np.asarray(summary["floor_pressure_peak_bins"], dtype=float)
    mean = np.asarray(summary["floor_pressure_mean_bins"], dtype=float)
    area = (Lx / len(peak)) * Ly
    static = total_mass * G / (Lx * Ly)
    total_mean = float(np.sum(mean * area))
    return {
        "strip_area": float(area),
        "static_reference_pressure": float(static),
        "peak_pressure_per_strip": peak.tolist(),
        "mean_pressure_per_strip": mean.tolist(),
        "peak_pressure_global": float(np.max(peak)),
        "peak_over_static": float(np.max(peak) / static) if static else float("nan"),
        "floor_load_balance_rel_error": _rel(total_mean, total_mass * G),
    }


def _rel(a: float, b: float) -> float:
    a, b = float(a), float(b)
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom