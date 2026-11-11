"""Breakage kinetics: appearance function, selection rates, batch PSD evolution.

The population balance used here is the repository's sectional operator
(:mod:`tensordem.breakage_pbm.dpbm`); this module supplies what comminution
practice needs on top of it:

**Appearance (breakage) function.**  The Austin-Luckie two-parameter family

.. math:: B(x\\,|\\,y) = \\Phi\\,(x/y)^{\\gamma} + (1-\\Phi)\\,(x/y)^{\\beta},
          \\qquad x \\le y,

with ``B(y|y) = 1`` (mass conserving by construction) and ``B`` the *cumulative*
mass fraction of a parent of size ``y`` reporting finer than ``x``.  The shape
parameter of the impact-breakage literature is ``t10 = B(y/10|y)`` (percent
passing one tenth of the parent size); :meth:`AppearanceFunction.from_t10`
inverts ``t10 -> gamma`` so a measured ``t10`` (or a ``t10`` from an energy law)
can be carried into a full size distribution instead of a single index.

**Selection (breakage rate) function.**  Austin's form

.. math:: S_i = \\frac{a\\,(d_i/1\\,\\text{mm})^{\\alpha}}
                     {1 + (d_i/\\mu)^{\\Lambda}}

is provided as the reduced-order/empirical rate, and
:func:`selection_rates_from_collisions` derives ``S_i`` from a *DEM collision
spectrum* instead: every media-media collision recorded by
:mod:`tensordem.comminution.mill_dem` carries an approach speed ``v_n`` and the
two colliding masses, so the mass-specific impact energy delivered to an ore
particle of size ``d_i`` is

.. math:: e_i = \\tfrac12 v_n^2 \\frac{m_{eff}(i)}{m_i},
          \\qquad m_{eff} = \\frac{m_{media}\\,m_i}{m_{media}+m_i},

a velocity/mass-ratio quantity that does **not** depend on the 2-D out-of-plane
thickness convention of the kernel (the leading systematic of a 2-D slice).  The
breakage probability of that event follows the Weibull weakest-link form with a
size-dependent median energy,

.. math:: P_{br}(e_i, d_i) = 1 - \\exp\\left[-\\left(\\frac{e_i}{e_{50}(d_i)}\\right)^{m}\\right],
          \\qquad e_{50}(d) = e_{50,ref}\\,(d/d_{ref})^{-k},

so ``S_i = f_{coll}(d_i) \\, \\langle P_{br}\\rangle_i`` with the collision
frequency scaled by cross-section, ``f_{coll}(d_i) = f_{coll,mean}
(d_i/d_{mean})^2`` -- the documented geometric assumption of this module (a
small particle embedded in a granular medium makes fewer contacts per unit
time).

**Energy -> t10 law.**  The JKMRC impact-breakage relation
``t10 = A (1 - exp(-b E_cs))`` [%] converts a mass-specific comminution energy
``E_cs`` [kWh/t] into the shape parameter, i.e. it is what a mill's power draw
becomes once it is divided by the ore mass it is grinding.

All energies are mass-specific [J/kg] unless a function name says ``kwh_t``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from ..breakage_pbm.dpbm import (
    VolumeGrid,
    cumulative_volume_fraction,
    solve_breakage,
)

__all__ = [
    "AppearanceFunction",
    "austin_selection_rates",
    "batch_grind",
    "cumulative_passing",
    "mass_fraction_matrix",
    "p80_from_psd",
    "passing_at",
    "section_edges",
    "selection_rates_from_collisions",
    "t10_energy_law",
    "t10_from_psd",
    "volume_daughter_matrix",
]

#: Weibull modulus of single-particle strength used by the impact model.
DEFAULT_WEIBULL_MODULUS = 2.5

#: Size exponent of the median breakage energy ``e50 ~ d**(-k)``.
DEFAULT_SIZE_EXPONENT = 0.6


def section_edges(diameters) -> np.ndarray:
    """Cell boundaries of an ascending diameter grid (``n+1`` edges, ascending).

    Interior edges are the geometric means of neighbouring pivots (the natural
    boundary for a geometric grid), the outer ones are extrapolated by the same
    ratio so that every pivot sits inside its cell.
    """
    d = np.asarray(diameters, dtype=np.float64)
    if d.ndim != 1 or d.size < 3:
        raise ValueError("diameters must be a 1-D array with >= 3 entries")
    if not np.all(np.diff(d) > 0.0):
        raise ValueError("diameters must be strictly ascending")
    edges = np.empty(d.size + 1)
    edges[1:-1] = np.sqrt(d[:-1] * d[1:])
    edges[0] = d[0] * d[0] / edges[1]
    edges[-1] = d[-1] * d[-1] / edges[-2]
    return edges


class AppearanceFunction:
    """Cumulative breakage function ``B(x|y)`` (Austin-Luckie, mass conserving)."""

    def __init__(self, phi: float = 0.35, gamma: float = 0.85, beta: float = 2.5) -> None:
        self.phi = float(phi)
        self.gamma = float(gamma)
        self.beta = float(beta)
        if not 0.0 < self.phi < 1.0:
            raise ValueError("phi must be in (0, 1)")
        if self.gamma <= 0.0 or self.beta <= 0.0:
            raise ValueError("gamma and beta must be positive")
        if self.gamma >= self.beta:
            raise ValueError("gamma must be smaller than beta (concave appearance)")

    # -------------------------------------------------------------- shape
    def cumulative_finer(self, x, y):
        """``B(x|y)``: mass fraction of parent size ``y`` finer than ``x``."""
        ratio = np.clip(np.asarray(x, dtype=np.float64) / float(y), 0.0, 1.0)
        return self.phi * ratio ** self.gamma + (1.0 - self.phi) * ratio ** self.beta

    @property
    def t10(self) -> float:
        """``t10`` in percent (``B(y/10|y)*100``)."""
        return 100.0 * float(self.cumulative_finer(0.1, 1.0))

    @classmethod
    def from_t10(cls, t10_percent: float, phi: float = 0.35,
                 beta: float = 2.5) -> "AppearanceFunction":
        """Solve ``gamma`` so that ``B(0.1|1)*100 == t10_percent``.

        ``t10`` physically lies in ``(0, 100``); values outside
        ``(1e-3, 99.9)`` are rejected because ``gamma`` diverges there.
        """
        t = float(t10_percent)
        if not 1e-3 < t < 99.9:
            raise ValueError("t10 must be in (0.001, 99.9) percent")
        target = t / 100.0
        # The fins asymptote of the family is ``phi`` itself (``B -> phi`` as
        # ``x/y -> 0``), so ``t10`` can never exceed ``100*phi`` percent.  When
        # the requested t10 is above the configured asymptote, raise phi to the
        # smallest value that can represent it instead of failing: that keeps
        # ``from_t10`` total over the physically attainable range (a soft ore
        # with t10 = 45% *requires* a high fines asymptote).
        phi = min(0.90, max(float(phi), 1.05 * target))

        def resid(gamma: float) -> float:
            return phi * 0.1 ** gamma + (1.0 - phi) * 0.1 ** beta - target

        lo, hi = 1e-3, 20.0
        if resid(lo) * resid(hi) > 0.0:  # pragma: no cover - guarded by the range check
            raise ValueError("t10 not attainable with the requested phi/beta")
        gamma = brentq(resid, lo, hi, xtol=1e-12, rtol=1e-12)
        return cls(phi=phi, gamma=gamma, beta=beta)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"AppearanceFunction(phi={self.phi:.3f}, gamma={self.gamma:.4f}, "
                f"beta={self.beta:.3f}, t10={self.t10:.2f}%)")


def mass_fraction_matrix(diameters, appearance: AppearanceFunction) -> np.ndarray:
    """Daughter **mass fractions** ``b[i, j]`` on the ascending diameter grid.

    ``b[i, j]`` is the mass fraction of a broken parent in section ``j`` that
    reports to section ``i``; columns sum to 1 (mass conserving) and entries
    above the diagonal are exactly zero (no size growth).  Mass finer than the
    grid floor is dumped into section 0 and the column renormalised, which is
    the only closure available on a finite grid (and is reported by
    :func:`grid_floor_loss`).
    """
    d = np.asarray(diameters, dtype=np.float64)
    edges = section_edges(d)
    n = d.size
    frac = np.zeros((n, n))
    for j in range(n):
        y = d[j]
        cum = np.asarray(appearance.cumulative_finer(edges, y), dtype=np.float64)
        cum = np.clip(cum, 0.0, 1.0)
        # mass in cell i (i <= j) = cum(edge_{i+1}) - cum(edge_i); cell j is
        # capped at the parent size, cum(y) = 1.
        upper = cum[1:].copy()
        upper[j] = 1.0
        lower = cum[:-1]
        col = np.clip(upper - lower, 0.0, None)
        col[j + 1:] = 0.0
        total = col.sum()
        if total <= 0.0:  # pragma: no cover - defensive
            raise ValueError("empty daughter column")
        frac[:, j] = col / total
    return frac


def grid_floor_loss(diameters, appearance: AppearanceFunction) -> float:
    """Worst-case mass fraction that falls below the grid floor (diagnostic)."""
    d = np.asarray(diameters, dtype=np.float64)
    edges = section_edges(d)
    return float(np.max([float(appearance.cumulative_finer(edges[0], y)) for y in d]))


def volume_daughter_matrix(diameters, mass_fractions) -> np.ndarray:
    """Expected daughter *counts* ``N[i, j] = b[i, j] v_j / v_i``.

    This is the form :func:`tensordem.breakage_pbm.dpbm.solve_breakage` consumes
    (its operator acts on number density in volume space).
    """
    d = np.asarray(diameters, dtype=np.float64)
    v = math.pi * d ** 3 / 6.0
    b = np.asarray(mass_fractions, dtype=np.float64)
    if b.shape != (d.size, d.size):
        raise ValueError("mass_fractions must be square with one row per diameter")
    return b * (v[None, :] / v[:, None])


def austin_selection_rates(diameter_mm, a: float, alpha: float, mu_mm: float,
                           Lambda: float) -> np.ndarray:
    """Austin selection function ``S_i`` [same time unit as ``a``].

    ``S_i = a (d_i/1mm)^alpha / (1 + (d_i/mu)^Lambda)``; ``a`` is the rate at
    1 mm, ``mu`` the size at which the rate stops growing, ``Lambda`` the
    sharpness of that roll-over.
    """
    d = np.asarray(diameter_mm, dtype=np.float64)
    if np.any(d <= 0.0):
        raise ValueError("diameters must be positive")
    if a <= 0.0 or mu_mm <= 0.0 or alpha <= 0.0 or Lambda < 0.0:
        raise ValueError("invalid Austin parameters")
    return a * d ** alpha / (1.0 + (d / mu_mm) ** Lambda)


def t10_energy_law(energy_kwh_t, a_percent: float, b_t_per_kwh: float) -> np.ndarray:
    """JKMRC impact law ``t10 = A (1 - exp(-b E_cs))`` [percent].

    ``A`` [%] is the practical maximum ``t10`` of the material, ``b``
    [t/kWh] its impact resistance; both are drop-weight-test material
    parameters.
    """
    e = np.asarray(energy_kwh_t, dtype=np.float64)
    if np.any(e < 0.0):
        raise ValueError("energy must be non-negative")
    a = float(a_percent)
    b = float(b_t_per_kwh)
    if not 0.0 < a <= 100.0:
        raise ValueError("A must be in (0, 100] percent")
    if b <= 0.0:
        raise ValueError("b must be positive")
    return a * (1.0 - np.exp(-b * e))


def selection_rates_from_collisions(approach_speeds, radius_a, radius_b,
                                    ore_diameters, *,
                                    media_density: float = 7800.0,
                                    ore_density: float = 2700.0,
                                    e50_ref_j_kg: float = 1200.0,
                                    d_ref_m: float = 0.010,
                                    weibull_modulus: float = DEFAULT_WEIBULL_MODULUS,
                                    size_exponent: float = DEFAULT_SIZE_EXPONENT,
                                    window_s: float = 1.0,
                                    collided_mass_kg: float | None = None) -> dict:
    """Selection rates of ore size classes from a DEM collision spectrum.

    Parameters
    ----------
    approach_speeds : (N,) array
        Normal approach speed ``v_n`` [m/s] of every recorded media-media
        collision in the measurement window.
    radius_a, radius_b : (N,) arrays
        Radii [m] of the two colliding media.
    ore_diameters : (M,) array
        Ore size classes [m].
    e50_ref_j_kg, d_ref_m, weibull_modulus, size_exponent
        Single-particle impact parameters: median mass-specific breakage energy
        at the reference size and its size scaling ``e50 ~ (d/d_ref)^(-k)``.
    window_s : float
        Length of the measurement window, used to convert collision counts into
        a frequency.
    collided_mass_kg : float, optional
        Total mass of media that participated in the recorded collisions; when
        given, the per-unit-mass collision frequency is used, otherwise the raw
        collision count per second of the *simulated slice*.

    Returns
    -------
    dict with ``diameter_m``, ``S_per_s``, ``collision_frequency_per_s``,
    ``mean_breakage_probability``, ``mean_specific_energy_j_kg``.
    """
    vn = np.asarray(approach_speeds, dtype=np.float64).ravel()
    ra = np.asarray(radius_a, dtype=np.float64).ravel()
    rb = np.asarray(radius_b, dtype=np.float64).ravel()
    d_ore = np.asarray(ore_diameters, dtype=np.float64).ravel()
    if not (vn.size == ra.size == rb.size):
        raise ValueError("approach_speeds, radius_a and radius_b must have equal length")
    if vn.size == 0:
        raise ValueError("empty collision spectrum")
    if np.any(vn < 0.0) or np.any(ra <= 0.0) or np.any(rb <= 0.0):
        raise ValueError("speeds must be non-negative and radii positive")
    if np.any(d_ore <= 0.0):
        raise ValueError("ore diameters must be positive")
    if window_s <= 0.0:
        raise ValueError("window_s must be positive")
    if e50_ref_j_kg <= 0.0 or d_ref_m <= 0.0:
        raise ValueError("e50_ref_j_kg and d_ref_m must be positive")

    # media masses of the recorded pairs (sphere volumes)
    m_a = media_density * 4.0 / 3.0 * math.pi * ra ** 3
    m_b = media_density * 4.0 / 3.0 * math.pi * rb ** 3
    m_media = np.minimum(m_a, m_b)  # the smaller partner governs the impact
    kinetic = 0.5 * m_media * vn ** 2                       # [J] per collision

    n_coll = vn.size / window_s                             # [1/s]
    if collided_mass_kg is not None:
        n_coll = n_coll / float(collided_mass_kg)           # [1/(s kg)]

    m_ore = ore_density * math.pi * d_ore ** 3 / 6.0
    e50 = e50_ref_j_kg * (d_ore / d_ref_m) ** (-size_exponent)

    S = np.zeros(d_ore.size)
    prob = np.zeros(d_ore.size)
    e_mean = np.zeros(d_ore.size)
    for i in range(d_ore.size):
        m_i = m_ore[i]
        m_eff = m_media * m_i / (m_media + m_i)
        e_spec = kinetic * (m_eff / m_i)   # mass-specific impact energy [J/kg]
        p = 1.0 - np.exp(-(e_spec / e50[i]) ** weibull_modulus)
        prob[i] = float(np.mean(p))
        e_mean[i] = float(np.mean(e_spec))
        # collision frequency of a particle of this size, scaled by cross-section
        S[i] = n_coll * prob[i]
    d_mean = float(np.mean(ra + rb))   # mean colliding-pair diameter
    scale = (d_ore / d_mean) ** 2
    return {
        "diameter_m": d_ore,
        "S_per_s": S * scale,
        "collision_frequency_per_s": n_coll * scale,
        "mean_breakage_probability": prob,
        "mean_specific_energy_j_kg": e_mean,
    }


def batch_grind(grid: VolumeGrid, feed_fractions, rate, appearance: AppearanceFunction,
                times) -> dict:
    """Integrate the sectional breakage balance for a batch grind.

    ``feed_fractions`` are mass fractions per section (ascending diameter
    order), ``rate`` the selection rates ``S_i`` [1/s] of those sections.  The
    daughter matrix is built from ``appearance`` so the operator is
    mass-conserving to machine precision.
    """
    d = grid.diameters
    w = np.asarray(feed_fractions, dtype=np.float64)
    if w.shape != (grid.n_sections,):
        raise ValueError("feed_fractions must have one entry per section")
    if np.any(w < 0.0):
        raise ValueError("mass fractions must be non-negative")
    w = w / w.sum()
    v = grid.pivots
    n_density = w / v                       # number density (unnormalised)
    b = mass_fraction_matrix(d, appearance)
    daughter = volume_daughter_matrix(d, b)
    rates = np.broadcast_to(np.asarray(rate, dtype=np.float64), (grid.n_sections,))
    sol = solve_breakage(grid, n_density, rate=rates, daughter=daughter,
                         t_span=(0.0, float(max(times))), t_eval=np.asarray(times, dtype=np.float64),
                         rtol=1e-9, atol=1e-12)
    fractions = np.zeros_like(sol["n"])
    for k in range(sol["n"].shape[0]):
        mass = grid.pivots * sol["n"][k]
        total = mass.sum()
        fractions[k] = mass / total if total > 0.0 else 0.0
    return {
        "grid": grid,
        "diameters_m": d,
        "t": sol["t"],
        "mass_fractions": fractions,
        "mass": grid.pivots * sol["n"],
        "daughter_mass_fractions": b,
        "selection_rates": rates.copy(),
    }


def cumulative_passing(diameters, mass_fractions) -> np.ndarray:
    """Cumulative mass fraction passing each diameter (ascending diameters)."""
    w = np.asarray(mass_fractions, dtype=np.float64)
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("empty distribution")
    return np.cumsum(w) / total


def passing_at(diameters, mass_fractions, size: float) -> float:
    """Interpolated cumulative passing (fraction, 0..1) at ``size``."""
    d = np.asarray(diameters, dtype=np.float64)
    cum = cumulative_passing(d, mass_fractions)
    x = float(size)
    if x <= d[0]:
        return float(cum[0] * (x / d[0]) ** 1.0)
    if x >= d[-1]:
        return 1.0
    return float(np.interp(math.log(x), np.log(d), cum))


def p80_from_psd(diameters, mass_fractions) -> float:
    """80%-passing size (same unit as ``diameters``), log-interpolated."""
    d = np.asarray(diameters, dtype=np.float64)
    cum = cumulative_passing(d, mass_fractions)
    if cum[-1] < 0.8:  # pragma: no cover - defensive
        return float(d[-1])
    logd = np.log(d)
    # cum is strictly increasing where there is mass; guard plateaus
    keep = np.concatenate([[True], np.diff(cum) > 1e-15])
    return float(np.exp(np.interp(0.8, cum[keep], logd[keep])))


def t10_from_psd(diameters, mass_fractions, parent_size: float) -> float:
    """``t10`` [percent] of a product distribution relative to ``parent_size``."""
    return 100.0 * passing_at(diameters, mass_fractions, 0.1 * float(parent_size))