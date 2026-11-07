"""Jamming volume fraction as an *energy-minimum* property: protocols + FSS.

The parent campaign (IN32-N02) requires that :math:`\\phi_J` stops being a
function of the preparation history and becomes a reproducible material property
of the energy minimum.  This module supplies the machinery:

``periodic_rsa``
    Random sequential adsorption in a **doubly periodic** cell, i.e. without any
    wall layer (the wall layer of the earlier channel model is one of the
    suspects for its low ``phi_J``).

``set_volume_fraction`` / ``run_protocol``
    Four preparation protocols acting on the same potential
    (:mod:`tensordem.granular2d.minimizer`):

    ``growth``           RSA at 0.45, then quasi-static inflation in steps of
                         ``dphi = 0.01`` with an energy minimisation at every step.
    ``decompression``    grow to 0.90, then quasi-static *deflation* in steps of
                         0.01 back to the measurement window (memory of a high
                         pressure history).
    ``rsa_direct``       fresh RSA at 0.45, a *single* instantaneous inflation to
                         the target ``phi`` (a violent quench), CG minimisation.
    ``fire_from_rsa``    identical start as ``rsa_direct`` but FIRE; the pair is
                         therefore a strict minimiser-invariance control.

``fit_jamming_density``
    ``phi_J`` from the equation of state ``P(phi)``: linear extrapolation of the
    first positive-pressure points to ``P = 0`` (the definition of a
    zero-pressure jammed state), with the fit window and its ``R^2`` reported.

``finite_size_scale``
    Fit ``phi_J(N) = phi_J(inf) - c N^{-1/(d nu)}`` with ``d = 2``; returns
    ``phi_J(inf)``, ``nu`` and confidence intervals from the covariance matrix
    *and* from a seed bootstrap.

Reduced units (``kn = 1``, mean radius ``1``) are used throughout: volume
fraction and the *shape* of ``P(phi)`` are dimensionless, and pressure is
reported both in reduced units (``P / kn``) and normalised by the onset value.
"""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from scipy.optimize import curve_fit

from .minimizer import (
    ContactModel,
    MinimizerResult,
    StaticPacking,
    cg_minimize,
    fire_minimize,
)

__all__ = [
    "periodic_rsa",
    "set_volume_fraction",
    "make_packing",
    "run_protocol",
    "PROTOCOLS",
    "fit_jamming_density",
    "finite_size_scale",
    "pressure_reduced",
]

PROTOCOLS = ("growth", "decompression", "rsa_direct", "fire_from_rsa")

#: placement density used by every protocol (safely below the 2D RSA saturation)
PLACEMENT_PHI = 0.45


def _radii(n: int, mean_radius: float, ratio: float, seed: int) -> np.ndarray:
    from .jamming import bidisperse_radii

    return bidisperse_radii(n, mean_radius, ratio=ratio, seed=seed)


def periodic_rsa(n: int, radii: np.ndarray, box: tuple[float, float], *,
                 seed: int = 0, max_tries: int = 4000) -> np.ndarray:
    """Sequential random deposition inside a doubly periodic cell.

    Overlap tests use the minimum-image convention on both axes, so a disk
    deposited near ``x = 0`` also respects the neighbour at ``x = Lx``.  No wall
    margin is used: the cell *is* the system.
    """
    radii = np.asarray(radii, dtype=np.float64)
    if len(radii) != n:
        raise ValueError("radii must have length n")
    lx, ly = float(box[0]), float(box[1])
    if not (lx > 0.0 and ly > 0.0):
        raise ValueError("box must be positive")
    rmax = float(radii.max())
    if 2.0 * rmax >= min(lx, ly):
        raise ValueError("the cell is too small for the largest disk")
    rng = np.random.default_rng(seed)
    cell = 2.0 * rmax * 1.0000001
    nx = max(1, int(math.ceil(lx / cell)))
    ny = max(1, int(math.ceil(ly / cell)))
    grid: dict[tuple[int, int], list[int]] = {}
    pos = np.zeros((n, 2))
    for k in range(n):
        r = float(radii[k])
        placed = False
        base = np.array([r, r]) + (np.array([lx, ly]) - 2.0 * np.array([r, r]))
        for _ in range(max_tries):
            p = np.array([r, r]) + rng.random(2) * base
            p = np.mod(p, np.array([lx, ly]))
            gx, gy = int(p[0] / cell), int(p[1] / cell)
            ok = True
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    key = ((gx + dx) % nx, (gy + dy) % ny)
                    for idx in grid.get(key, ()):
                        d = p - pos[idx]
                        d = d - np.array([lx, ly]) * np.round(d / np.array([lx, ly]))
                        if float(np.hypot(*d)) < r + radii[idx] + 1e-12:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                pos[k] = p
                grid.setdefault((gx % nx, gy % ny), []).append(k)
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"periodic RSA could not place disk {k} of {n} "
                f"(phi_requested={_phi(radii, lx * ly):.3f}); lower the placement phi")
    return pos


def _phi(radii: np.ndarray, area: float) -> float:
    return float(np.sum(math.pi * np.asarray(radii, dtype=np.float64) ** 2) / area)


def make_packing(n: int, phi: float, box: tuple[float, float] | None = None,
                 *, ratio: float = 1.4, seed: int = 0,
                 mean_radius: float = 1.0, placement_phi: float = PLACEMENT_PHI,
                 model: ContactModel | None = None) -> StaticPacking:
    """Build a doubly periodic packing at volume fraction ``phi``.

    The cell is fixed by ``phi`` and the deposition density
    (``placement_phi``), and the radii are inflated instantaneously to ``phi``;
    the packing is therefore *not* yet relaxed (the caller minimises it).
    """
    if not (0.0 < phi < 1.0):
        raise ValueError("phi must lie in (0, 1)")
    if not (0.0 < placement_phi <= phi):
        raise ValueError("placement_phi must be positive and not exceed phi")
    radii0 = _radii(n, mean_radius, ratio, seed)
    if box is None:
        area = float(np.sum(math.pi * radii0**2) / placement_phi)
        box = (math.sqrt(area), math.sqrt(area))
    lx, ly = float(box[0]), float(box[1])
    area = lx * ly
    # radii that realise exactly `placement_phi` in this cell
    radii_place = radii0 * math.sqrt(placement_phi * area / float(np.sum(math.pi * radii0**2)))
    pos = periodic_rsa(n, radii_place, (lx, ly), seed=seed + 1)
    target = radii_place * math.sqrt(phi / placement_phi)
    return StaticPacking(pos, target, box=(lx, ly), model=model or ContactModel())


def set_volume_fraction(packing: StaticPacking, phi: float) -> None:
    """Scale all radii (at fixed cell) so the area fraction becomes ``phi``."""
    current = packing.volume_fraction()
    if current <= 0.0:
        raise ValueError("current volume fraction must be positive")
    packing.radii = packing.radii * math.sqrt(phi / current)
    packing.rebuild(packing.pos)


def pressure_reduced(packing: StaticPacking) -> float:
    """2D pressure in reduced units ``P / (kn * r_mean^0)`` (``kn`` factors out)."""
    return float(packing.pressure() / packing.model.kn)


def _minimise(packing: StaticPacking, method: str, *, tol: float,
              max_iter: int) -> MinimizerResult:
    if method == "fire":
        return fire_minimize(packing, tol=tol, max_iter=max_iter)
    if method in ("cg", "cg_pr_plus"):
        return cg_minimize(packing, tol=tol, max_iter=max_iter)
    raise ValueError(f"unknown minimiser {method!r}")


def run_protocol(name: str, *, n: int, seed: int, phi_grid, dphi: float = 0.01,
                 ratio: float = 1.4, mu: float = 0.0, kt_ratio: float = 0.0,
                 method: str = "fire", tol: float = 1e-7, max_iter: int = 60000,
                 top_phi: float = 0.90, placement_phi: float = PLACEMENT_PHI,
                 mean_radius: float = 1.0) -> dict[str, object]:
    """Run one preparation protocol and return its ``P(phi)`` curve.

    Parameters
    ----------
    phi_grid
        Measurement levels (ascending).  ``growth`` and ``decompression`` sweep
        through them, so one call yields the full curve; the two instantaneous
        protocols are restarted from a fresh RSA state for every level.
    """
    if name not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    grid = sorted(float(p) for p in np.asarray(phi_grid, dtype=np.float64).reshape(-1))
    if not grid:
        raise ValueError("phi_grid must not be empty")
    model = ContactModel(kn=1.0, kt_ratio=kt_ratio, mu=mu)

    def _record(pk, phi, res: MinimizerResult) -> dict[str, float]:
        return {
            "phi": float(phi),
            "pressure": pressure_reduced(pk),
            "energy": float(res.energy),
            "eps": float(res.eps),
            "fmax": float(res.fmax),
            "fmax_reduction": float(res.reduction),
            "iterations": int(res.iterations),
            "converged": bool(res.converged),
            "monotone": bool(res.monotone),
            "coordination": float(res.coordination),
            "n_contacts": int(res.n_contacts),
            "rejected": int(res.rejected),
            "stress_symmetry_rel": float(pk.stress_symmetry_rel()),
        }

    records: list[dict[str, float]] = []
    stages = 0
    if name in ("rsa_direct", "fire_from_rsa"):
        use_method = "fire" if name == "fire_from_rsa" else ("cg" if method == "fire" else method)
        for phi in grid:
            pk = make_packing(n, phi, ratio=ratio, seed=seed, model=model,
                              placement_phi=placement_phi, mean_radius=mean_radius)
            res = _minimise(pk, use_method, tol=tol, max_iter=max_iter)
            stages += 1
            records.append(_record(pk, phi, res))
        return {"protocol": name, "n": int(n), "seed": int(seed), "mu": float(mu),
                "method": use_method, "dphi": float(dphi), "tol": float(tol),
                "records": records}

    # ---- sweep protocols: growth (from below) or decompression (from above)
    if name == "growth":
        levels = np.arange(float(placement_phi) + dphi, max(grid[-1], float(placement_phi)) + 0.5 * dphi,
                           dphi)
        levels = np.round(levels, 12)
        rise, fall = levels, np.zeros(0)
    else:  # decompression: rise to `top_phi`, then measure on the deflation branch
        rise = np.round(np.arange(float(placement_phi) + 0.02, top_phi + 0.5 * 0.02, 0.02), 12)
        fall = np.round(np.arange(top_phi - dphi, min(grid) - 0.5 * dphi, -dphi), 12)
    pk = make_packing(n, float(rise[0]), ratio=ratio, seed=seed, model=model,
                      placement_phi=placement_phi, mean_radius=mean_radius)
    targets = set(np.round(np.asarray(grid), 12).tolist())
    fall_set = set(np.round(np.asarray(fall), 12).tolist())
    for phi in np.concatenate([rise, fall]):
        set_volume_fraction(pk, float(phi))
        res = _minimise(pk, method, tol=tol, max_iter=max_iter)
        stages += 1
        level = round(float(phi), 12)
        # for the decompression protocol only the deflation branch is measured:
        # the rise is preparation, and recording both would duplicate levels
        measured = level in (targets & fall_set) if name == "decompression" else level in targets
        if measured:
            records.append(_record(pk, level, res))
    records.sort(key=lambda r: r["phi"])
    return {"protocol": name, "n": int(n), "seed": int(seed), "mu": float(mu),
            "method": method, "dphi": float(dphi), "tol": float(tol),
            "records": records}


def fit_jamming_density(phi, pressure, *, n_points: int = 3,
                        p_floor: float | None = 1e-4, converged=None) -> dict[str, float]:
    """``phi_J`` from ``P(phi)`` by linear extrapolation of the onset points.

    The onset is the first level with ``P > p_floor``.  ``p_floor`` defaults to
    the absolute reduced pressure ``1e-4``, i.e. ten times the numerical pressure
    floor that a *converged* contact-free state shows (any residual pressure
    below the jamming density is minimisation noise; see the pressure tables in
    ``artifacts/jamming_protocol/jamming_protocol.json``).  ``p_floor_sensitivity``
    records the estimator for 1x/2x/5x that floor so the reader can see how much
    the choice moves ``phi_J`` (it is reported as ``phi_j_floor_band``).  The next
    ``n_points-1``
    positive-pressure levels are included in the least-squares line
    ``P = a (phi - phi_J)``.  A jammed state is defined by ``P -> 0``, so
    ``phi_J = phi - P/a`` at the onset.  If fewer than two positive levels exist
    the density is bracketed instead: ``phi_J`` is reported as the last
    zero-pressure level and ``bracketed`` is set.
    """
    phi = np.asarray(phi, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)
    if phi.shape != p.shape:
        raise ValueError("phi and pressure must have the same shape")
    order = np.argsort(phi)
    phi, p = phi[order], p[order]
    if converged is not None:
        keep = np.asarray(converged, dtype=bool)[order]
        phi, p = phi[keep], p[keep]
    if p_floor is None:
        p_floor = 1e-3 * float(np.max(p)) if len(p) and np.max(p) > 0 else 1e-9
    order = np.argsort(phi)
    phi, p = phi[order], p[order]
    positive = np.flatnonzero(p > p_floor)
    if len(positive) < 2:
        below = phi[p <= p_floor]
        phi_j = float(below.max()) if len(below) else float(phi.min())
        return {"phi_j": phi_j, "slope": float("nan"), "r2": float("nan"),
                "n_fit": int(len(positive)), "bracketed": True,
                "phi_onset": phi_j, "pressure_onset": 0.0}
    start = int(positive[0])
    stop = min(len(phi), start + max(2, int(n_points)))
    sel = slice(start, stop)
    x, y = phi[sel], p[sel]
    a, b = np.polyfit(x, y, 1)
    if a <= 0.0:
        return {"phi_j": float(x[0]), "slope": float(a), "r2": float("nan"),
                "n_fit": int(len(x)), "bracketed": True,
                "phi_onset": float(x[0]), "pressure_onset": float(y[0])}
    pred = a * x + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    out = {"phi_j": float(-b / a), "slope": float(a), "r2": float(r2),
           "n_fit": int(len(x)), "bracketed": False,
           "phi_onset": float(x[0]), "pressure_onset": float(y[0]),
           "p_floor": float(p_floor)}
    # sensitivity to the noise-floor choice: the estimator must not be a knob
    sens = {}
    for factor in (1.0, 2.0, 5.0):
        floor = p_floor * factor
        pos = np.flatnonzero(p > floor)
        if len(pos) < 2:
            continue
        st = int(pos[0])
        sp = min(len(phi), st + max(2, int(n_points)))
        if sp - st < 2:
            continue
        aa, bb = np.polyfit(phi[st:sp], p[st:sp], 1)
        if aa > 0:
            sens[f"{factor:g}x"] = float(-bb / aa)
    if sens:
        vals = np.asarray(list(sens.values()))
        out["p_floor_sensitivity"] = sens
        out["phi_j_floor_band"] = float(vals.max() - vals.min())
    return out


def _fss_model(n, phi_inf, c, nu, d=2.0):
    return phi_inf - c * np.power(np.asarray(n, dtype=np.float64), -1.0 / (d * nu))


def finite_size_scale(sizes, phi_j, *, sigma=None, d: float = 2.0,
                      n_boot: int = 400, seed: int = 0,
                      nu_grid=(0.25, 2.5, 0.005)) -> dict[str, object]:
    """Fit ``phi_J(N) = phi_J(inf) - c N^{-1/(d nu)}`` and report ``nu``.

    The exponent is estimated by **variable projection**: for every trial ``nu``
    on a grid the model is linear in ``(phi_inf, c)``, so the pair is solved by
    weighted least squares and the residual is a one-dimensional function of
    ``nu``; the minimising ``nu`` is then refined on a fine local grid.  This is
    far more robust than a three-parameter ``curve_fit`` when the five ``phi_J``
    values differ by only a few 1e-3 (the usual finite-size window), where a
    simultaneous fit easily stalls at a bound.

    ``sigma`` are per-size standard errors of the mean and are used as weights.
    A Gaussian bootstrap (``n_boot`` resamples of the means within their
    uncertainties) gives the confidence intervals of ``phi_inf`` and ``nu``.
    """
    n = np.asarray(sizes, dtype=np.float64)
    y = np.asarray(phi_j, dtype=np.float64).reshape(-1)
    if n.shape != y.shape:
        raise ValueError("sizes and phi_j must have the same length")
    if len(n) < 4:
        raise ValueError("at least four system sizes are required for a 3-parameter fit")
    if np.any(n <= 0.0):
        raise ValueError("sizes must be positive")
    w = np.ones_like(y) if sigma is None else 1.0 / np.maximum(np.asarray(sigma, dtype=np.float64), 1e-300)

    def _solve(nu: float, target: np.ndarray):
        x = np.power(n, -1.0 / (d * nu))
        # weighted linear least squares for  y = a - c x
        sw = np.sum(w)
        sx = np.sum(w * x)
        sy = np.sum(w * target)
        sxx = np.sum(w * x * x)
        sxy = np.sum(w * x * target)
        den = sw * sxx - sx * sx
        if abs(den) < 1e-300:
            return np.nan, np.nan, np.inf
        slope = (sw * sxy - sx * sy) / den
        intercept = (sy - slope * sx) / sw
        c = -slope
        resid = target - (intercept + slope * x)
        ssr = float(np.sum(w * resid**2))
        return intercept, c, ssr

    grid = np.arange(nu_grid[0], nu_grid[1] + 0.5 * nu_grid[2], nu_grid[2])
    ssrs = np.array([_solve(float(v), y)[2] for v in grid])
    best = float(grid[int(np.argmin(ssrs))])
    fine = np.linspace(max(1e-3, best - 2 * nu_grid[2]), best + 2 * nu_grid[2], 401)
    fine_ssr = np.array([_solve(float(v), y)[2] for v in fine])
    nu_hat = float(fine[int(np.argmin(fine_ssr))])
    phi_inf, c, _ = _solve(nu_hat, y)
    pred = phi_inf - c * np.power(n, -1.0 / (d * nu_hat))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    rng = np.random.default_rng(seed)
    sig = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-6) if sigma is not None else None
    boot_nu, boot_phi, boot_c = [], [], []
    for _ in range(int(n_boot)):
        noise = rng.normal(0.0, sig) if sig is not None else rng.normal(0.0, max(np.std(y - pred), 1e-6))
        yb = pred + noise
        ss = np.array([_solve(float(v), yb)[2] for v in fine])
        nb = float(fine[int(np.argmin(ss))])
        pb, cb, _ = _solve(nb, yb)
        boot_nu.append(nb)
        boot_phi.append(float(pb))
        boot_c.append(float(cb))

    def _ci(values, level=0.95):
        if not values:
            return (float("nan"), float("nan"))
        return (float(np.quantile(values, (1 - level) / 2)),
                float(np.quantile(values, 1 - (1 - level) / 2)))

    return {
        "phi_inf": float(phi_inf),
        "c": float(c),
        "nu": float(nu_hat),
        "d": float(d),
        "r2": float(r2),
        "phi_inf_ci_bootstrap": _ci(boot_phi),
        "c_ci_bootstrap": _ci(boot_c),
        "nu_ci_bootstrap": _ci(boot_nu),
        "nu_ci68": _ci(boot_nu, 0.68),
        "nu_ssr_profile": [[float(a), float(b)] for a, b in zip(fine[::20], fine_ssr[::20])],
        "bootstrap_samples": len(boot_nu),
        "sizes": [float(v) for v in n],
        "phi_j": [float(v) for v in y],
        "phi_j_fit": [float(v) for v in pred],
        "residuals": [float(v) for v in (y - pred)],
    }
