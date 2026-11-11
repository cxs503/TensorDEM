"""Tool-neutral post-processing for the chute cross-validation.

Every tool (TensorDEM, LIGGGHTS, second reference) writes the *same* dump
schema -- CSV with ``t,id,x,y,z,vx,vy,vz`` -- and the *same* functions below
turn it into the comparison quantities.  That is what makes the sampling
convention ("采样口径") identical across tools instead of each code reporting its
own internal averaging.

Definitions (also written into ``result.case_spec.sampling``)
-------------------------------------------------------------
thickness  ``h = N (pi d^3/6) / (lx ly)`` -- volume-equivalent layer thickness,
           independent of binning and of the tool.
profile    arithmetic mean of ``vx`` of the particle centres in bins of 0.5 d
           above ``z_wall`` (top of the rough base), time-averaged over the
           steady window.
Bagnold    ``vx(z) = C [h^p - (h - z)^p]`` fitted for ``p`` in [0.6, 3.0] on the
           flowing part ``0 <= z <= 0.9 h``; ``C`` is eliminated analytically by
           least squares for each ``p`` (one-dimensional search).  Bagnold
           scaling is ``p = 3/2``; Silbert et al. (2001) fit the same form.
arrest     a run is "arrested" when the late-window rms speed satisfies
           ``v_rms / sqrt(g d) < 0.02`` for a whole window; otherwise it is
           "flowing".
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

__all__ = [
    "read_dump",
    "bin_profile",
    "density_profile",
    "bagnold_fit",
    "average_profile_over_window",
    "classify_run",
    "h_stop_from_scan",
]

DUMP_COLUMNS = ("t", "id", "x", "y", "z", "vx", "vy", "vz")


def read_dump(path: str | Path) -> np.ndarray:
    """Read a neutral dump CSV -> (n, 8) float array."""
    arr = np.genfromtxt(path, delimiter=",", names=True)
    if arr.size == 0:
        return np.zeros((0, 8))
    return np.column_stack([arr[c] for c in DUMP_COLUMNS]).astype(np.float64)


def bin_profile(z: np.ndarray, vx: np.ndarray, *, d: float, z_wall: float,
                bin_width_d: float = 0.5) -> dict:
    """Height-binned vx profile (particle-centre sampling)."""
    z_rel = z - z_wall
    if z_rel.size == 0:
        return {"z_mid_d": np.zeros(0), "vx_mean": np.zeros(0),
                "count": np.zeros(0, dtype=int)}
    width = bin_width_d * d
    nb = max(1, int(math.ceil(z_rel.max() / width)))
    idx = np.clip((z_rel / width).astype(int), 0, nb - 1)
    count = np.bincount(idx, minlength=nb)
    vsum = np.bincount(idx, weights=vx, minlength=nb)
    mean = np.where(count > 0, vsum / np.maximum(count, 1), 0.0)
    z_mid = (np.arange(nb) + 0.5) * width
    return {"z_mid_d": z_mid / d, "vx_mean": mean, "count": count}


def density_profile(z: np.ndarray, *, d: float, lx: float, ly: float,
                    z_wall: float, bin_width_d: float = 0.5) -> dict:
    """Packing fraction per height bin (for the bed-height check)."""
    z_rel = z - z_wall
    width = bin_width_d * d
    nb = max(1, int(math.ceil(max(z_rel.max(), 1e-12) / width)))
    idx = np.clip((z_rel / width).astype(int), 0, nb - 1)
    count = np.bincount(idx, minlength=nb)
    vol_bin = lx * ly * width
    phi = count * (math.pi / 6.0) * d ** 3 / vol_bin
    return {"z_mid_d": (np.arange(nb) + 0.5) * width / d, "phi": phi}


def bagnold_fit(z: np.ndarray, vx: np.ndarray, *, d: float, z_wall: float,
                h_d: float, p_grid: np.ndarray | None = None,
                z_max_frac: float = 0.9, min_bins: int = 4) -> dict:
    """Fit the Bagnold-form profile ``vx = C [h^p - (h - z)^p]``.

    ``h_d`` is the thickness in diameters (volume-equivalent).  Only bins below
    ``z_max_frac * h`` are used, because the surface layer is ballistic rather
    than Bagnold (Silbert et al. 2001, Sec. IV).
    """
    z_rel = (z - z_wall) / d
    m = (z_rel >= 0.0) & (z_rel <= z_max_frac * h_d)
    if m.sum() < min_bins or h_d <= 0.0:
        return {"p": float("nan"), "C_over_vref": float("nan"),
                "r2": float("nan"), "n_points": int(m.sum())}
    if p_grid is None:
        p_grid = np.arange(0.60, 3.001, 0.005)
    zz = z_rel[m]
    vv = vx[m]
    h = h_d
    best = None
    for p in p_grid:
        basis = h ** p - np.maximum(h - zz, 0.0) ** p
        denom = float(basis @ basis)
        if denom <= 0.0:
            continue
        c = float(basis @ vv) / denom
        resid = vv - c * basis
        ss = float(resid @ resid)
        if best is None or ss < best[0]:
            best = (ss, p, c)
    if best is None:
        return {"p": float("nan"), "C_over_vref": float("nan"),
                "r2": float("nan"), "n_points": int(m.sum())}
    ss, p, c = best
    sst = float(((vv - vv.mean()) ** 2).sum())
    r2 = 1.0 - ss / sst if sst > 0 else float("nan")
    return {"p": float(p), "C": float(c), "r2": float(r2),
            "n_points": int(m.sum())}


def average_profile_over_window(dumps: list[dict], *, case) -> dict:
    """Time-average binned profiles from a list of snapshots.

    Each snapshot is the dict produced by :func:`read_dump`-style loaders with
    keys ``z``, ``vx``, ``t``.  Returns the profile, the volume-equivalent
    thickness, the 95th-percentile height and the Bagnold fit.
    """
    z_all, vx_all = [], []
    z95 = []
    for snap in dumps:
        z_all.append(snap["z"])
        vx_all.append(snap["vx"])
        z95.append(np.percentile(snap["z"], 95))
    z = np.concatenate(z_all)
    vx = np.concatenate(vx_all)
    n = dumps[0]["z"].size
    h_vol = n * (math.pi / 6.0) * case.d ** 3 / (case.lx * case.ly)
    prof = bin_profile(z, vx, d=case.d, z_wall=case.z_wall)
    fit = bagnold_fit(prof["z_mid_d"] * case.d + case.z_wall, prof["vx_mean"],
                      d=case.d, z_wall=case.z_wall, h_d=h_vol / case.d)
    fit["C_over_vref"] = fit.get("C", float("nan")) / case.v_ref
    dens = density_profile(z, d=case.d, lx=case.lx, ly=case.ly,
                           z_wall=case.z_wall)
    return {
        "h_over_d": h_vol / case.d,
        "z95_over_d": float(np.mean(z95) / case.d),
        "z_mid_d": prof["z_mid_d"].tolist(),
        "vx_mean_over_vref": (prof["vx_mean"] / case.v_ref).tolist(),
        "count": prof["count"].tolist(),
        "phi_profile": dens["phi"].tolist(),
        "bagnold_p": fit["p"],
        "bagnold_r2": fit["r2"],
        "bagnold_C_over_vref": fit["C_over_vref"],
        "bagnold_points": fit["n_points"],
        "vx_bulk_over_vref": float(vx.mean() / case.v_ref),
        "v_rms_over_vref": float(math.sqrt((vx ** 2).mean()) / case.v_ref),
        "n_snapshots": len(dumps),
    }


def classify_run(*, v_rms_over_vref: float, vx_mean_over_vref: float,
                 v_arrest: float = 0.02) -> str:
    """Arrest / flow classification used by the h_stop scan."""
    if v_rms_over_vref < v_arrest:
        return "arrest"
    if vx_mean_over_vref > 2.0 * v_arrest:
        return "flow"
    return "creeping"


def h_stop_from_scan(records: list[dict], *, theta_tol: float = 1e-6) -> dict:
    """Bracket the arrest/flow thickness boundary from a mass scan at fixed theta.

    ``records`` items must carry ``theta_deg``, ``h_over_d`` (volume-equivalent
    thickness) and ``state`` in {"arrest", "flow", "creeping"}.  For each angle
    the function returns the bracket ``[h_arrest_max, h_flow_min]`` and the
    midpoint estimate.  ``creeping`` runs are treated as flowing (they still
    support a sustained slow flow) but flagged.
    """
    from collections import defaultdict
    by_theta = defaultdict(list)
    for rec in records:
        by_theta[round(rec["theta_deg"], 6)].append(rec)
    out = {}
    for theta, recs in sorted(by_theta.items()):
        recs = sorted(recs, key=lambda r: r["h_over_d"])
        arrested = [r for r in recs if r["state"] == "arrest"]
        flowing = [r for r in recs if r["state"] != "arrest"]
        if not arrested or not flowing:
            out[theta] = {
                "theta_deg": theta,
                "h_stop_over_d": float("nan"),
                "bracket": None,
                "states": [(r["h_over_d"], r["state"]) for r in recs],
            }
            continue
        h_a = max(r["h_over_d"] for r in arrested)
        h_f = min(r["h_over_d"] for r in flowing)
        resolved = h_f > h_a
        out[theta] = {
            "theta_deg": theta,
            "h_stop_over_d": 0.5 * (h_a + h_f),
            "h_stop_halfwidth_over_d": 0.5 * (h_f - h_a),
            "bracket": [h_a, h_f],
            "bracket_resolved": bool(resolved),
            "states": [(r["h_over_d"], r["state"]) for r in recs],
        }
    return out