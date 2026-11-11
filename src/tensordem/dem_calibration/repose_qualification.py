"""Qualification gates for a repose forward model used as a calibration target.

`docs/GAP-DEM-AUTOCALIB_round1.md` measured three properties of the shipped
repose forward model (reach the literature band / be monotone in ``mu`` / be at
rest when measured) and concluded that it cannot serve as a calibration target.
All three tests were run at **one** system size (a few hundred grains) and at
**one** set of protocol knobs.

This module turns the missing question into machine-checkable gates: a repose
response may only be used as a calibration target if it is

``G1 flank_quality``      a real heap flank is being fitted (``r2`` high),
``G2 solid_fraction``     the deposit is a three-dimensional heap, not a sheet,
``G3 size_convergence``   ``theta(N_max)`` agrees with ``theta(N_prev)`` inside a
                          pre-declared tolerance (round-1's own 2 deg criterion),
``G4 material_dominance`` the response reacts to the **material** knob (grain
                          friction) at least ``MATERIAL_DOMINANCE_MIN`` times
                          more than to a **protocol geometry** knob, and reacts
                          to friction with signal-to-noise >= 3 against the
                          run-to-run (seed) spread.

Everything here is pure arithmetic over cell records so that the gates can be
unit-tested on synthetic data and re-evaluated on a stored artefact without
re-running DEM.  No threshold is fitted after the fact: they are module
constants, listed in the report before the sweep was launched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "FLANK_R2_MIN",
    "SOLID_FRACTION_MIN",
    "SIZE_PLATEAU_DEG",
    "SIZE_PLATEAU_FRAC",
    "SIGNAL_TO_NOISE_MIN",
    "MATERIAL_DOMINANCE_MIN",
    "Gate",
    "spearman",
    "seed_sigma",
    "dominance",
    "evaluate",
]

#: A flank fit below this is not a flank (round-1 fits landed at 0.68-0.90).
FLANK_R2_MIN = 0.95
#: Bulk solid fraction of a genuine three-dimensional heap deposit.  A loose
#: random packing of spheres is ~0.55-0.60; a deposit measured at 1-3 % is a
#: monolayer sheet and its "slope" is the deposition footprint, not a friction
#: angle.
SOLID_FRACTION_MIN = 0.30
#: Absolute plateau tolerance on theta_r [deg] between the two largest N.
SIZE_PLATEAU_DEG = 2.0
#: Relative plateau tolerance (5 % of the smaller angle) - whichever is looser.
SIZE_PLATEAU_FRAC = 0.05
#: Minimum signal-to-noise of a calibration signal.
SIGNAL_TO_NOISE_MIN = 3.0
#: Material sensitivity must beat protocol-geometry sensitivity by this factor.
MATERIAL_DOMINANCE_MIN = 3.0


@dataclass(frozen=True)
class Gate:
    """One machine-checkable acceptance gate."""

    name: str
    value: float
    threshold: float
    relation: str          # ">=" or "<="
    passed: bool
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    # resolve ties
    vals, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(vals.size)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, 0.0 for a degenerate input."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        raise ValueError("spearman needs two equal-length series of length >= 2")
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if denom == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


def _theta(cell: Mapping) -> float:
    return float(cell["theta_r_deg"])


def seed_sigma(cells: Iterable[Mapping]) -> float:
    """Pooled run-to-run (seed) standard deviation of ``theta_r``.

    Cells are grouped by every field that is *not* the seed; each group with
    more than one member contributes its sample variance (ddof=1), pooled over
    all groups by degrees of freedom.
    """
    groups: dict[tuple, list[float]] = {}
    for c in cells:
        key = tuple(sorted((k, v) for k, v in c.items()
                           if k not in ("seed", "theta_r_deg", "phi_solid", "r2",
                                        "ke_ratio", "wall_s", "digest", "cell")))
        groups.setdefault(key, []).append(_theta(c))
    num = 0.0
    dof = 0
    for vals in groups.values():
        if len(vals) > 1:
            num += float(np.var(vals, ddof=1)) * (len(vals) - 1)
            dof += len(vals) - 1
    if dof == 0:
        return float("nan")
    return math.sqrt(num / dof)


def _group_mean(cells: Sequence[Mapping], **match) -> float:
    """Seed-mean theta of the cells matching every ``match`` key/value."""
    vals = [_theta(c) for c in cells
            if all(c.get(k) == v for k, v in match.items())]
    if not vals:
        raise KeyError(f"no cell matches {match}")
    return float(np.mean(vals))


def dominance(signal_delta: float, geometry_delta: float, sigma: float) -> dict:
    """Compare a material signal against a protocol-geometry signal.

    ``signal_delta``  change of theta_r caused by the material knob,
    ``geometry_delta`` change caused by the protocol-geometry knob,
    ``sigma``         run-to-run (seed) noise of theta_r.

    Returns ``snr_signal``, ``snr_geometry``, ``ratio`` (|signal|/|geometry|)
    and ``passed`` = signal is resolvable (SNR >= 3) **and** dominates the
    geometry by ``MATERIAL_DOMINANCE_MIN``.
    """
    if not sigma or math.isnan(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be a positive finite number")
    sd, gd = abs(float(signal_delta)), abs(float(geometry_delta))
    ratio = float("inf") if gd == 0.0 else sd / gd
    snr_s = sd / sigma
    snr_g = gd / sigma
    passed = bool(snr_s >= SIGNAL_TO_NOISE_MIN and ratio >= MATERIAL_DOMINANCE_MIN)
    return {"delta_signal": sd, "delta_geometry": gd, "sigma": sigma,
            "snr_signal": snr_s, "snr_geometry": snr_g, "ratio": ratio,
            "passed": passed}


def size_plateau(thetas_by_n: Mapping[float, float]) -> tuple[float, float, bool]:
    """Plateau test on the two largest system sizes.

    Returns ``(slope_deg, tolerance_deg, passed)`` where ``slope_deg`` is
    ``|theta(N_max) - theta(N_prev)|`` and the tolerance is
    ``max(SIZE_PLATEAU_DEG, SIZE_PLATEAU_FRAC * min(theta_prev, theta_max))``.
    """
    sizes = sorted(float(n) for n in thetas_by_n)
    if len(sizes) < 2:
        raise ValueError("size_plateau needs at least two system sizes")
    n_prev, n_max = sizes[-2], sizes[-1]
    t_prev, t_max = float(thetas_by_n[n_prev]), float(thetas_by_n[n_max])
    slope = abs(t_max - t_prev)
    tol = max(SIZE_PLATEAU_DEG, SIZE_PLATEAU_FRAC * min(t_prev, t_max))
    return slope, tol, bool(slope <= tol)


def evaluate(cells: Sequence[Mapping], *,
             size_match: Mapping | None = None,
             mu_low: float = 0.30, mu_high: float = 0.70,
             irf_low: float = 3.5, irf_high: float = 14.0) -> dict:
    """Evaluate every gate on a list of cell records.

    Each cell is a mapping with at least ``n_grains``, ``seed``, ``mu``,
    ``inject_radius_factor``, ``theta_r_deg``, ``phi_solid``, ``r2``.
    ``size_match`` restricts the N-sweep to the cells sharing those fields
    (default: the modal parameter set of the with-geometry cells).
    """
    cells = list(cells)
    gates: dict[str, Gate] = {}

    r2_min = min(float(c["r2"]) for c in cells)
    gates["G1_flank_quality"] = Gate(
        "G1_flank_quality", r2_min, FLANK_R2_MIN, ">=",
        bool(r2_min >= FLANK_R2_MIN),
        "min flank fit r2 over all cells")

    phi_med = float(np.median([float(c["phi_solid"]) for c in cells]))
    gates["G2_solid_fraction"] = Gate(
        "G2_solid_fraction", phi_med, SOLID_FRACTION_MIN, ">=",
        bool(phi_med >= SOLID_FRACTION_MIN),
        "median bulk solid fraction of the deposit")

    # --- G3 size convergence on the N sweep -----------------------------
    if size_match is None:
        size_match = {}
    sweep = [c for c in cells
             if all(c.get(k) == v for k, v in size_match.items())]
    by_n: dict[float, list[float]] = {}
    for c in sweep:
        by_n.setdefault(float(c["n_grains"]), []).append(_theta(c))
    thetas_by_n = {n: float(np.mean(v)) for n, v in by_n.items()}
    slope, tol, ok = size_plateau(thetas_by_n)
    gates["G3_size_convergence"] = Gate(
        "G3_size_convergence", slope, tol, "<=", ok,
        f"|theta(N_max)-theta(N_prev)| over N={sorted(thetas_by_n)}")

    # --- G4 material dominance vs protocol geometry ---------------------
    sigma = seed_sigma(cells)
    d_mu = (_group_mean(cells, mu=mu_high, inject_radius_factor=irf_low)
            - _group_mean(cells, mu=mu_low, inject_radius_factor=irf_low))
    d_geo = (_group_mean(cells, mu=mu_low if mu_low is not None else None,
                         inject_radius_factor=irf_high)
             - _group_mean(cells, mu=mu_low, inject_radius_factor=irf_low))
    dom = dominance(d_mu, d_geo, sigma)
    gates["G4a_signal_snr"] = Gate(
        "G4a_signal_snr", dom["snr_signal"], SIGNAL_TO_NOISE_MIN, ">=",
        bool(dom["snr_signal"] >= SIGNAL_TO_NOISE_MIN),
        f"theta(mu={mu_high})-theta(mu={mu_low}) / seed sigma")
    gates["G4b_material_dominance"] = Gate(
        "G4b_material_dominance", dom["ratio"], MATERIAL_DOMINANCE_MIN, ">=",
        bool(dom["ratio"] >= MATERIAL_DOMINANCE_MIN),
        f"|dtheta/dmu| vs |dtheta/d(inject_radius_factor)|")

    failed = sorted(name for name, g in gates.items() if not g.passed)
    return {"gates": {k: v.as_dict() for k, v in gates.items()},
            "size_sweep": thetas_by_n,
            "seed_sigma_deg": sigma,
            "dominance": dom,
            "failed_gates": failed,
            "verdict": "qualified" if not failed else "not_qualified"}