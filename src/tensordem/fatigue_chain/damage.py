"""Rainflow -> Miner damage chain and its statistics-compression surrogates.

The chain is the one the ship side consumes::

    F(t)  --(modal transfer, exact SDOF)-->  sigma(t)
          --(ASTM E1049-85 range pair)-->  (dS_i, n_i)
          --(Miner + DNV-RP-C203 curve)-->  D = sum n_i / N_i

The counting itself is *not* re-implemented here: it is
:func:`tensordem.whipping.rainflow.rainflow` and
:func:`tensordem.whipping.fatigue.miner_damage`, the repository's single source
of truth, so this module cannot drift away from the S-N side of the eco-system.

Two surrogates quantify what is lost when the time history is thrown away and
only *statistics* are kept (the current practice behind ice-zone fatigue
conclusions):

``narrowband_rayleigh_damage``
    (stress RMS, cycle count) -> Rayleigh range distribution -> two-slope
    Miner integral.  This is the classical narrow-band formula
    ``E[S^m] = (2 sqrt(2) sigma)^m Gamma(1 + m/2)``, with the DNV knee handled
    by integrating both branches numerically instead of pretending a single
    slope applies to the whole range distribution.

``statistics_compression``
    the two tabulated reductions: narrow band from RMS, and "all cycles at the
    design peak" (the conservative block-equivalent estimate).  Both are
    compared with the exact rainflow damage of the same record, so the reported
    ``dD/D`` is the information loss of the compression, not of the chain.

Fail-closed: an empty or absent force history raises
:class:`~tensordem.fatigue_chain.contract.MissingForceHistory`; there is no
path in this module that returns a finite damage for no data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Sequence

import numpy as np

from ..whipping.fatigue import SNCurve, miner_damage
from ..whipping.rainflow import rainflow, turning_points
from .contract import MissingForceHistory
from .response import ModalChannel, modal_stress_history

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2 renamed trapz

__all__ = [
    "PowerLawSNCurve",
    "ChainResult",
    "chain_damage",
    "damage_from_stress",
    "stress_history",
    "narrowband_rayleigh_damage",
    "statistics_compression",
    "equal_amplitude_damage",
]


class PowerLawSNCurve:
    """Single-slope S-N curve ``N = a dSigma^-m`` (unit studies, no knee).

    Used where the S-N *slope* is the independent variable (the minimum-sampling
    -rate law).  The DNV curves cannot express a pure slope because their knee
    is defined by ``m2 - m1``; this helper keeps the same interface
    (:meth:`cycles_to_failure`, ``m1``) so it plugs into
    :func:`tensordem.whipping.fatigue.miner_damage` and the equivalent-range
    computation unchanged.  ``a`` is expressed in MPa^m, like the DNV table.
    """

    def __init__(self, slope: float, log_a1: float = 12.164,
                 name: str | None = None) -> None:
        if slope <= 0.0:
            raise ValueError("slope must be positive")
        self.m1 = float(slope)
        self.log_a1 = float(log_a1)
        self.name = name or f"single-slope m={slope:g}"

    @property
    def a1(self) -> float:
        return 10.0 ** self.log_a1

    def cycles_to_failure(self, dsigma) -> np.ndarray:
        s = np.asarray(dsigma, dtype=float) / 1.0e6
        if np.any(s <= 0.0):
            raise ValueError("stress ranges must be positive")
        return self.a1 * s ** (-self.m1)


@dataclass(frozen=True)
class ChainResult:
    """Output of the F(t) -> sigma(t) -> rainflow -> Miner chain."""

    damage: float
    n_cycles: float
    dsigma_eq_Pa: float
    stress_peak_Pa: float
    stress_rms_Pa: float
    n_range_bins: int
    mean_range_Pa: float
    max_range_Pa: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _as_array(force, t) -> tuple[np.ndarray, np.ndarray]:
    if force is None or t is None:
        raise MissingForceHistory("no DEM force history supplied to the fatigue chain")
    tt = np.asarray(t, dtype=float).ravel()
    ff = np.asarray(force, dtype=float).ravel()
    if tt.size != ff.size:
        raise ValueError("t and force must have the same length")
    if tt.size == 0:
        raise MissingForceHistory("empty DEM force history supplied to the fatigue chain")
    return tt, ff


def stress_history(t, force, channels: Sequence[ModalChannel]) -> np.ndarray:
    """Hot-spot stress history from the force history (fail-closed)."""
    tt, ff = _as_array(force, t)
    return modal_stress_history(tt, ff, channels)


def damage_from_stress(sigma, curve: SNCurve) -> ChainResult:
    """Rainflow + Miner damage of a stress history (the counting half-chain)."""
    arr = np.asarray(sigma, dtype=float).ravel()
    if arr.size == 0:
        raise MissingForceHistory("empty stress history supplied to the fatigue chain")
    if arr.size < 2:
        raise ValueError("at least two samples are required for rainflow counting")
    cycles = rainflow(arr)
    if not cycles:
        raise ValueError("rainflow produced no cycles; record too short")
    ranges = np.array([c.range for c in cycles], dtype=float)
    counts = np.array([c.count for c in cycles], dtype=float)
    info = miner_damage(curve, ranges, counts)
    eq = info["equivalent_range_MPa"] * 1.0e6
    return ChainResult(
        damage=float(info["damage"]),
        n_cycles=float(info["cycles"]),
        dsigma_eq_Pa=float(eq),
        stress_peak_Pa=float(np.max(np.abs(arr))),
        stress_rms_Pa=float(np.sqrt(np.mean(arr * arr))),
        n_range_bins=int(ranges.size),
        mean_range_Pa=float(np.mean(ranges)),
        max_range_Pa=float(ranges.max()),
    )


def chain_damage(t, force, channels: Sequence[ModalChannel], curve: SNCurve) -> ChainResult:
    """Full chain: force history -> stress -> rainflow -> Miner damage."""
    sigma = stress_history(t, force, channels)
    return damage_from_stress(sigma, curve)


def equal_amplitude_damage(curve: SNCurve, dsigma_Pa: float, n_cycles: float) -> float:
    """Closed-form Miner damage of ``n`` constant-range cycles."""
    if dsigma_Pa <= 0.0:
        raise ValueError("stress range must be positive")
    return float(n_cycles / curve.cycles_to_failure(np.array([dsigma_Pa]))[0])


def narrowband_rayleigh_damage(curve: SNCurve, stress_rms_Pa: float,
                              n_cycles: float, *, n_points: int = 4000) -> float:
    """Narrow-band Rayleigh two-slope Miner damage from ``(RMS, N)``.

    Range ``S = 2A`` with ``A`` Rayleigh of mode ``sigma``, i.e.
    ``f_S(s) = s / (4 sigma^2) exp(-s^2 / (8 sigma^2))``; the two DNV branches
    are integrated separately at the knee.
    """
    if stress_rms_Pa <= 0.0:
        raise ValueError("stress RMS must be positive")
    if n_cycles <= 0.0:
        return 0.0
    top = 12.0 * stress_rms_Pa
    s = np.linspace(1e-6 * top, top, n_points)
    pdf = s / (4.0 * stress_rms_Pa ** 2) * np.exp(-s * s / (8.0 * stress_rms_Pa ** 2))
    lives = curve.cycles_to_failure(s)
    return float(n_cycles * _trapz(pdf / lives, s))


def statistics_compression(curve: SNCurve, t, force, channels: Sequence[ModalChannel]
                           ) -> Dict[str, Dict[str, float]]:
    """Damage from statistics only, compared with the exact rainflow damage.

    Statistics used (all tabulated in a load report): stress RMS, total
    rainflow cycle count, peak stress.  The exact chain uses the full range
    distribution.
    """
    sigma = stress_history(t, force, channels)
    cycles = rainflow(sigma)
    if not cycles:
        raise ValueError("rainflow produced no cycles; record too short")
    ranges = np.array([c.range for c in cycles])
    counts = np.array([c.count for c in cycles])
    exact = miner_damage(curve, ranges, counts)["damage"]
    mean = float(np.mean(sigma))
    fluct = sigma - mean
    rms_total = float(np.sqrt(np.mean(sigma * sigma)))
    # the narrow-band surrogate must use the *fluctuating* stress: the static
    # part of the load carries no range, and folding it into the RMS inflates
    # the Rayleigh tail (measured: +3839 % instead of +1550 % on the m5405
    # record before this correction)
    rms = float(np.sqrt(np.mean(fluct * fluct)))
    n_tot = float(counts.sum())
    peak = float(np.max(np.abs(fluct)))

    out: Dict[str, Dict[str, float]] = {
        "exact_rainflow": {"damage": float(exact), "n_cycles": n_tot,
                           "stress_rms_Pa": rms_total, "stress_std_Pa": rms,
                           "stress_peak_Pa": float(np.max(np.abs(sigma))),
                           "stress_peak_fluctuating_Pa": peak}
    }
    if exact <= 0.0:
        return out

    d_nb = narrowband_rayleigh_damage(curve, rms, n_tot)
    out["narrowband_rayleigh"] = {
        "damage": d_nb,
        "dD_over_D": d_nb / exact - 1.0,
        "inputs": ["stress_std_fluctuating", "rainflow_cycle_count"],
    }
    # all cycles at the design (peak) range: block-equivalent upper estimate
    d_peak = equal_amplitude_damage(curve, 2.0 * peak, n_tot)
    out["design_peak_block"] = {
        "damage": d_peak,
        "dD_over_D": d_peak / exact - 1.0,
        "inputs": ["stress_peak_fluctuating", "rainflow_cycle_count"],
    }
    # Rayleigh amplitude distribution scaled to the observed peak (a "P95/peak"
    # conditioned narrow band: rms replaced by peak/sqrt(2 ln N))
    if n_tot > 1.0:
        rms_equiv = peak / math.sqrt(2.0 * math.log(n_tot))
        d_pk = narrowband_rayleigh_damage(curve, rms_equiv, n_tot)
        out["rayleigh_peak_conditioned"] = {
            "damage": d_pk,
            "dD_over_D": d_pk / exact - 1.0,
            "inputs": ["stress_peak_fluctuating", "rainflow_cycle_count"],
        }
    return out


def turning_point_count(signal) -> int:
    """Number of rainflow turning points (used by the regression tests)."""
    return int(turning_points(signal).size)