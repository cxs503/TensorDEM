"""Segregation metrics for the rotating-drum bidisperse bed.

Two complementary, independent indices are provided because they answer
different questions and have different failure modes:

``size_position_correlation``
    Pearson correlation between particle diameter and normalised radial
    position.  This is the *primary* direction index: it is dimensionless, it
    needs no binning (so it has no bin-occupancy noise) and it is **identically
    zero for a single size class** (zero variance in diameter), which makes it
    the correct instrument for the monodisperse negative control.

``radial_profile`` / ``concentration_segregation``
    Area-fraction based core-shell measure: large-species area fraction
    ``phi_L(r)`` in annular bins, the deviation ``S = <|phi_L - phi_0|>/phi_0``
    and the core-shell excess ``phi_outer - phi_inner``.  This is the classical
    "how much of the large species sits where" measure, but it has a finite
    sampling floor: even a perfectly mixed bed yields a non-zero ``S`` because
    of finite bin occupancy.  That floor is quantified explicitly by
    ``permutation_null`` and every claim built on ``S`` is reported together
    with its z-score against that null.

Null model (important)
----------------------
For ``S``/excess the null is "the association between species and position
carries no information": the *positions* are permuted among all particles
(species labels and radii stay attached to their index).  That preserves the
position multiset (the geometry of the bed) and the species composition, and
breaks only the species-position association.  Two alternatives were rejected:
permuting labels across particles (breaks the label-radius coupling and
inflates the null) and permuting positions within a size class (the class
position *set* is unchanged, so the null is degenerate: zero spread).
For the correlation index the null is a permutation of diameters across
particles, i.e. the standard permutation test for Pearson's rho.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "radial_profile",
    "size_position_correlation",
    "concentration_segregation",
    "core_shell_excess",
    "permutation_null",
    "classify_mode",
]


def radial_profile(pos, radii, is_large, drum_r, n_bins=8, min_count=3):
    """Large-species area fraction in annular bins.

    Bins span the *occupied* radial range ``[0.98*min(r/R), 1]``: in a partially
    filled drum the region ``r < r_lo`` lies above the free surface and is empty,
    so fixed ``0..1`` bins would spend resolution on empty space and inflate the
    occupancy noise of the metrics.

    Returns dict with ``edges``/``centres`` (normalised radius), ``phi`` (NaN for
    bins with fewer than ``min_count`` particles), ``area_weight`` (annulus area,
    normalised), ``count`` and ``phi_0`` (global large-species area fraction).
    """
    pos = np.asarray(pos, float)
    radii = np.asarray(radii, float)
    is_large = np.asarray(is_large, bool)
    r_norm = np.linalg.norm(pos, axis=1) / drum_r
    r_lo = float(np.min(r_norm)) * 0.98 if r_norm.size else 0.0
    r_lo = min(max(r_lo, 0.0), 0.9)
    edges = np.linspace(r_lo, 1.0, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    area = math.pi * radii ** 2
    phi = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, int)
    for b in range(n_bins):
        m = (r_norm >= edges[b]) & (r_norm < edges[b + 1])
        count[b] = int(m.sum())
        if count[b] >= min_count:
            tot = area[m].sum()
            if tot > 0:
                phi[b] = area[m & is_large].sum() / tot
    w = (edges[1:] ** 2 - edges[:-1] ** 2)
    w = w / w.sum() if w.sum() > 0 else w
    return {"edges": edges, "centres": centres, "phi": phi,
            "area_weight": w, "count": count,
            "phi_0": float(area[is_large].sum() / area.sum()) if area.sum() > 0 else 0.0}


def size_position_correlation(radii, r_norm):
    """Pearson correlation between diameter and normalised radius.

    Returns 0.0 when the diameter variance is zero (single size class): the
    index cannot manufacture segregation out of nothing.
    """
    d = 2.0 * np.asarray(radii, float)
    r = np.asarray(r_norm, float)
    if d.size < 3:
        return 0.0
    sd, sr = d.std(), r.std()
    if sd <= 1e-12 or sr <= 1e-12:
        return 0.0
    return float(np.mean((d - d.mean()) * (r - r.mean())) / (sd * sr))


def concentration_segregation(profile):
    """S = <|phi_L - phi_0|> / phi_0, area weighted, NaN bins dropped."""
    phi = np.asarray(profile["phi"], float)
    w = np.asarray(profile["area_weight"], float)
    phi_0 = float(profile["phi_0"])
    valid = np.isfinite(phi)
    if not valid.any() or phi_0 <= 0.0:
        return 0.0
    w = w[valid] / w[valid].sum()
    return float(np.sum(w * np.abs(phi[valid] - phi_0)) / phi_0)


def core_shell_excess(profile):
    """phi_L(outer half) - phi_L(inner half) of the occupied radial range.

    Positive -> large species enriched at the periphery (Brazil-nut pattern:
    coarse at the wall, fines in the core); negative -> coarse species enriched
    in the core (reverse Brazil-nut, e.g. density driven).
    """
    phi = np.asarray(profile["phi"], float)
    centres = np.asarray(profile["centres"], float)
    valid = np.isfinite(phi)
    r_mid = 0.5 * (centres[0] + 1.0)
    inner = valid & (centres < r_mid)
    outer = valid & (centres > r_mid)
    if not inner.any() or not outer.any():
        return 0.0
    return float(np.mean(phi[outer]) - np.mean(phi[inner]))


def permutation_null(pos, radii, is_large, drum_r, n_bins=8,
                     n_perm=200, seed=0):
    """Null distributions for the segregation indices (see module docstring)."""
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos, float)
    radii = np.asarray(radii, float)
    is_large = np.asarray(is_large, bool)
    r_norm = np.linalg.norm(pos, axis=1) / drum_r

    obs = radial_profile(pos, radii, is_large, drum_r, n_bins)
    S_obs = concentration_segregation(obs)
    ex_obs = core_shell_excess(obs)
    corr_obs = size_position_correlation(radii, r_norm)

    S_null = np.empty(n_perm)
    ex_null = np.empty(n_perm)
    for k in range(n_perm):
        # global permutation of positions among the particles (species labels and
        # radii stay attached to their index).  This preserves the position
        # multiset -- i.e. the geometry of the bed -- and breaks only the
        # association between species and position, which is exactly the null
        # hypothesis of interest.
        p = pos[rng.permutation(pos.shape[0])]
        prof = radial_profile(p, radii, is_large, drum_r, n_bins)
        S_null[k] = concentration_segregation(prof)
        ex_null[k] = core_shell_excess(prof)

    corr_null = np.empty(n_perm)
    for k in range(n_perm):
        corr_null[k] = size_position_correlation(radii[rng.permutation(radii.size)], r_norm)

    sd_s = float(S_null.std(ddof=1)) if n_perm > 1 else 0.0
    sd_e = float(ex_null.std(ddof=1)) if n_perm > 1 else 0.0
    sd_c = float(corr_null.std(ddof=1)) if n_perm > 1 else 0.0
    return {
        "S_observed": S_obs,
        "S_null_mean": float(S_null.mean()),
        "S_null_std": sd_s,
        "S_excess_over_null": float(S_obs - S_null.mean()),
        "S_z": float((S_obs - S_null.mean()) / sd_s) if sd_s > 0 else 0.0,
        "excess_observed": ex_obs,
        "excess_null_mean": float(ex_null.mean()),
        "excess_null_std": sd_e,
        "excess_z": float((ex_obs - ex_null.mean()) / sd_e) if sd_e > 0 else 0.0,
        "corr_observed": corr_obs,
        "corr_null_std": sd_c,
        "corr_z": float(corr_obs / sd_c) if sd_c > 0 else 0.0,
        "n_perm": int(n_perm),
        "profile": obs,
    }


def classify_mode(excess, excess_z, corr, corr_z, z_threshold=3.0):
    """Classify the measured pattern into a segregation mode.

    ``mixed``      no significant species-position structure
    ``periphery``  coarse species enriched at the drum wall (Brazil-nut)
    ``core``       coarse species enriched in the core (reverse Brazil-nut)
    """
    if abs(excess_z) < z_threshold and abs(corr_z) < z_threshold:
        return "mixed"
    if excess > 0.0 or (abs(excess_z) < z_threshold and corr > 0.0):
        return "periphery"
    return "core"