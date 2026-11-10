"""Measurement operators for the chain ledger: speeds, profiles, decay lengths.

Two *independent* routes give the solitary wave speed on the same run:

``arrival_speed``   local pulse velocity ``v_k = (x_{k+1} - x_k)/(t_{k+1} - t_k)``
                    from the peak-force arrival time of every contact, averaged
                    over a window that excludes the impacted end and the far
                    (reflected) end -- the "time of flight" method.

``shape_speed``     the speed for which the analytic Nesterenko profile best
                    matches two snapshots taken ``dt_snap`` apart, i.e. a
                    profile-tracking route that uses the whole waveform instead
                    of a single arrival instant.

Both are reported; the acceptance protocol scores the arrival-time route and
uses the second one as a cross-check of the first.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .nlsm import nesterenko_profile_strain, nesterenko_width


def arrival_speed(peak_time: Sequence[float], contact_x: Sequence[float],
                  i0: int, i1: int) -> dict[str, Any]:
    """Least-squares speed from the peak arrival times over contacts ``[i0, i1)``."""
    t = np.asarray(peak_time, dtype=float)
    x = np.asarray(contact_x, dtype=float)
    t, x = t[i0:i1], x[i0:i1]
    ok = np.isfinite(t)
    t, x = t[ok], x[ok]
    if t.size < 3:
        raise ValueError("need at least three contacts in the fit window")
    slope, intercept = np.polyfit(t, x, 1)     # x = v t + x0
    pred = slope * t + intercept
    ss_res = float(np.sum((x - pred) ** 2))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    local = np.diff(x) / np.diff(t)
    return {
        "speed_m_s": float(slope),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "local_speed_mean_m_s": float(np.mean(local)),
        "local_speed_std_m_s": float(np.std(local)),
        "n_contacts": int(t.size),
    }


def peak_amplitude(peak_force: Sequence[float], i0: int, i1: int) -> float:
    """Mean peak force over a window of contacts (the propagating amplitude)."""
    f = np.asarray(peak_force, dtype=float)[i0:i1]
    f = f[np.isfinite(f) & (f > 0)]
    if f.size == 0:
        return float("nan")
    return float(np.mean(f))


def _resample_profile(x_num: np.ndarray, s_num: np.ndarray, x0: float, x1: float,
                      n: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(x0, x1, n)
    return grid, np.interp(grid, x_num, s_num, left=0.0, right=0.0)


def profile_l1(strain_num: Sequence[float], contact_x: Sequence[float],
               strain_max: float, x_centre: float, spacing: float,
               width: float | None = None, n_grid: int = 400) -> dict[str, Any]:
    """Relative L1 distance between a measured strain profile and ``cos^4``.

    ``L1 = sum|s_num - s_th| / sum s_th`` on a common uniform grid covering
    ``+-1.5`` pulse supports.  Only the pulse *centre* (from the measured peak)
    is transferred from the data; the amplitude is taken from the measured peak
    strain and the width from the analytic prediction, so the residual is a pure
    shape error.
    """
    xs = np.asarray(contact_x, dtype=float)
    ss = np.asarray(strain_num, dtype=float)
    L = nesterenko_width(spacing) if width is None else float(width)
    half = 1.5 * 0.5 * math.pi * L
    grid, s_num_g = _resample_profile(xs, ss, x_centre - half, x_centre + half, n_grid)
    s_th = nesterenko_profile_strain(grid, strain_max, x_centre, spacing, width=width)
    denom = float(np.sum(np.abs(s_th)))
    if denom <= 0:
        raise ValueError("analytic profile vanished on the comparison grid")
    return {
        "l1_relative": float(np.sum(np.abs(s_num_g - s_th)) / denom),
        "width_m": L,
        "support_m": 0.5 * math.pi * L * 2.0,
        "strain_max": float(strain_max),
        "x_centre": float(x_centre),
    }


def profile_l1_force(force: Sequence[float], contact_x: Sequence[float],
                     force_peak: float, x_centre: float, spacing: float,
                     half_window: float = 2.5) -> float:
    """Relative L1 of the *contact force* profile against ``cos^6`` (no resampling).

    The contact force is the directly measured discrete quantity
    (``F = k (a s)^{3/2}``), so this route avoids interpolating a peaked,
    three-point-wide pulse.
    """
    from .nlsm import nesterenko_profile_force

    f = np.asarray(force, dtype=float)
    x = np.asarray(contact_x, dtype=float)
    th = nesterenko_profile_force(x, force_peak, x_centre, spacing)
    win = np.abs(x - x_centre) <= half_window * spacing
    denom = float(np.sum(np.abs(th[win])))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(f[win] - th[win])) / denom)


def best_width_profile_l1(strain_num: Sequence[float], contact_x: Sequence[float],
                          strain_max: float, x_centre: float, spacing: float,
                          widths: Sequence[float] | None = None) -> dict[str, Any]:
    """Same metric but with the ``cos^4`` width swept -- isolates shape vs width.

    The reported ``width_ratio`` is ``width_best / (a sqrt(3))``; a value far
    from one means the leading-order continuum width is inaccurate while the
    ``cos^4`` family may still describe the shape.
    """
    theory = nesterenko_width(spacing)
    if widths is None:
        widths = theory * np.linspace(0.5, 2.0, 61)
    best = None
    for w in widths:
        rec = profile_l1(strain_num, contact_x, strain_max, x_centre, spacing, width=w)
        if best is None or rec["l1_relative"] < best["l1_relative"]:
            best = rec
    best["width_ratio"] = best["width_m"] / theory
    best["l1_at_theory_width"] = profile_l1(
        strain_num, contact_x, strain_max, x_centre, spacing)["l1_relative"]
    return best


def pulse_width_measured(strain_num: Sequence[float], contact_x: Sequence[float],
                         x_centre: float) -> dict[str, float]:
    """Width of the measured pulse: equivalent (integral) width and FWHM.

    ``width_eq = (sum(s) * dx) / max(s)`` is the discretisation-robust width:
    with only ~3 contact samples per pulse the FWHM collapses to one spacing,
    whereas ``width_eq`` reproduces the analytic ``(3 pi / 8) L`` for ``cos^4``
    (checked in the unit tests).
    """
    xs = np.asarray(contact_x, dtype=float)
    ss = np.asarray(strain_num, dtype=float)
    peak = float(np.max(ss))
    if peak <= 0:
        return {"fwhm_m": float("nan"), "support_m": float("nan")}
    above = ss >= 0.5 * peak
    # contiguous run containing the centre
    centre = int(np.argmin(np.abs(xs - x_centre)))
    lo = centre
    while lo - 1 >= 0 and above[lo - 1]:
        lo -= 1
    hi = centre
    while hi + 1 < ss.size and above[hi + 1]:
        hi += 1
    width = float(xs[hi] - xs[lo]) + float(np.mean(np.diff(xs)))
    support = 0.0
    nz = np.nonzero(ss > 0.02 * peak)[0]
    if nz.size:
        support = float(xs[nz[-1]] - xs[nz[0]] + np.mean(np.diff(xs)))
    eq = float(np.sum(np.maximum(ss, 0.0)) * np.mean(np.diff(xs)) / peak)
    return {"fwhm_m": width, "width_eq_m": eq, "support_m": support,
            "peak_strain": peak}


def track_speed(snapshot_a: dict[str, Any], snapshot_b: dict[str, Any],
                i0: int, i1: int) -> dict[str, Any]:
    """Speed from the centre-of-mass displacement of the pulse between snapshots."""
    def centroid(snap: dict[str, Any]) -> float:
        w = np.asarray(snap["strain"])[i0:i1]
        w = np.maximum(w, 0.0)
        x = np.asarray(snap["contact_x"])[i0:i1]
        tot = float(np.sum(w))
        return float(np.sum(w * x) / tot) if tot > 0 else float("nan")

    xa = centroid(snapshot_a)
    xb = centroid(snapshot_b)
    dt = float(snapshot_b["t"]) - float(snapshot_a["t"])
    return {"speed_m_s": (xb - xa) / dt if dt > 0 else float("nan"),
            "x_a": xa, "x_b": xb, "dt": dt}


def window_for(n_contacts: int, skip_head: int = 5, skip_tail: int = 8) -> tuple[int, int]:
    lo = min(skip_head, max(0, n_contacts - 5))
    hi = max(lo + 3, n_contacts - skip_tail)
    return lo, min(hi, n_contacts)


def pick_pulse_snapshot(snapshots: Sequence[dict[str, Any]], fraction: float = 0.5,
                        guard: int = 8) -> dict[str, Any] | None:
    """Snapshot whose pulse centre is closest to the requested chain fraction."""
    best, best_err = None, float("inf")
    for snap in snapshots:
        s = np.asarray(snap["strain"])
        if s.max() <= 0:
            continue
        w = np.maximum(s, 0.0)
        x = np.asarray(snap["contact_x"])
        xc = float(np.sum(w * x) / np.sum(w))
        frac = (xc - x[0]) / (x[-1] - x[0])
        # keep the pulse away from both ends (no reflections in the window)
        if frac < 0.2 or frac > 0.8:
            continue
        err = abs(frac - fraction)
        if err < best_err:
            best, best_err = snap, err
    return best