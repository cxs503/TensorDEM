"""Random-field load -> rainflow -> S-N -> Miner fatigue life.

The counting and the S-N curve are **not** re-implemented here: they are
:func:`tensordem.whipping.rainflow.rainflow` and
:func:`tensordem.whipping.fatigue.miner_damage` with the repository's DNV-RP-C203
curve table, the same pair used by
:mod:`tensordem.fatigue_chain` for the point-statistics chain.  This module adds
only the parts specific to a *spatial* load model:

``ice_pressure_load_envelope``
    the temporal envelope ``A(t) >= 0`` of the ice load (Poisson arrivals of
    crushing pulses, log-normal amplitudes, finite rise and decay times),
    normalised to unit time average so that it modulates the pressure field
    without changing its level.

``rainflow_miner_life``
    per-realisation rainflow + Miner damage and the resulting fatigue life
    ``T_f = T_sim / D`` (``D`` is the damage accumulated over ``T_sim``).

``analytic_miner_anchor``
    the independent check: a point-load record whose stress history is known in
    closed form (static Navier transfer times the recorded load), summed by
    hand into Miner's rule, against the full chain (pressure field -> modal
    projection -> modal dynamics -> rainflow -> Miner).

Life definition: ``T_f = T_sim / D`` is the time to reach ``D = 1`` at the same
damage rate; it is the standard linear-damage life and is what the Monte Carlo
distribution of ``T_f`` is built from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from ..whipping.fatigue import DNV_CURVES, SNCurve, miner_damage
from ..whipping.rainflow import rainflow

__all__ = [
    "FatigueMonteCarlo",
    "ice_pressure_load_envelope",
    "rainflow_miner_life",
    "lifetime_from_damage",
    "analytic_miner_anchor",
    "SIGNAL_CYCLES_ANCHOR",
]


def ice_pressure_load_envelope(
    t: np.ndarray,
    *,
    rate_per_s: float,
    tau_rise: float,
    tau_decay: float,
    log_sigma: float = 0.35,
    rng: Optional[np.random.Generator] = None,
    seed: int = 20271109,
) -> np.ndarray:
    """Temporal load envelope ``A(t)`` of an ice-crushing record, mean ``= 1``.

    Pulse ``k`` arrives at ``t_k ~ Poisson(rate_per_s)``, has log-normal
    amplitude, rises linearly over ``tau_rise`` and decays exponentially with
    ``tau_decay`` (a crushing event: fast load build-up, slower release).  The
    superposition is divided by its own time mean, so ``A`` is a pure shape
    function and the pressure level is set by the field, not the envelope.
    """
    tt = np.asarray(t, dtype=float).ravel()
    if tt.size < 2:
        raise ValueError("t must have at least two samples")
    if rate_per_s <= 0.0:
        raise ValueError("rate_per_s must be positive")
    if tau_rise <= 0.0 or tau_decay <= 0.0:
        raise ValueError("rise and decay times must be positive")
    gen = rng if rng is not None else np.random.default_rng(seed)
    duration = float(tt[-1] - tt[0])
    n_events = gen.poisson(rate_per_s * duration)
    if n_events == 0:
        raise ValueError("no crushing event in the record; raise rate_per_s")
    arrivals = gen.uniform(tt[0], tt[-1], n_events)
    amplitudes = gen.lognormal(0.0, log_sigma, n_events)
    env = np.zeros_like(tt)
    for t0, amp in zip(arrivals, amplitudes):
        tau = tt - t0
        rising = (tau >= 0.0) & (tau < tau_rise)
        falling = tau >= tau_rise
        env[rising] += amp * tau[rising] / tau_rise
        env[falling] += amp * np.exp(-(tau[falling] - tau_rise) / tau_decay)
    mean = float(np.mean(env))
    if mean <= 0.0:
        raise ValueError("degenerate envelope")
    return env / mean


def lifetime_from_damage(damage: float, duration_s: float) -> float:
    """``T_f = T_sim / D`` (linear damage; ``inf`` for zero damage)."""
    if damage <= 0.0:
        return float("inf")
    return float(duration_s) / float(damage)


def rainflow_miner_life(
    stress: np.ndarray,
    dt: float,
    *,
    curve: SNCurve = DNV_CURVES["D_air"],
    duration_s: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Rainflow + Miner damage for one or many stress histories.

    ``stress`` shape ``(n_t,)`` or ``(n_samples, n_t)``.  Returns per-sample
    arrays of damage, cycle count, maximum and equivalent stress range, and the
    linear life ``T_f``.
    """
    s = np.asarray(stress, dtype=float)
    single = s.ndim == 1
    if single:
        s = s[None, :]
    if s.ndim != 2:
        raise ValueError("stress must be (n_t,) or (n_samples, n_t)")
    if s.shape[1] < 4:
        raise ValueError("stress history too short for rainflow counting")
    if not np.all(np.isfinite(s)):
        raise ValueError("stress history contains non-finite values")
    n_t = s.shape[1]
    dur = float(duration_s) if duration_s is not None else float(dt) * (n_t - 1)
    out_damage = np.zeros(s.shape[0])
    out_cycles = np.zeros(s.shape[0])
    out_max = np.zeros(s.shape[0])
    out_eq = np.zeros(s.shape[0])
    for i, row in enumerate(s):
        cycles = rainflow(row)
        if not cycles:
            continue
        ranges = np.array([c.range for c in cycles], dtype=float)
        counts = np.array([c.count for c in cycles], dtype=float)
        res = miner_damage(curve, ranges, counts)
        out_damage[i] = res["damage"]
        out_cycles[i] = res["cycles"]
        out_max[i] = res["max_range_MPa"] * 1.0e6
        out_eq[i] = res["equivalent_range_MPa"] * 1.0e6
    life = np.array([lifetime_from_damage(d, dur) for d in out_damage])
    result = {
        "damage": out_damage,
        "cycles": out_cycles,
        "max_range": out_max,
        "equivalent_range": out_eq,
        "life_s": life,
        "duration_s": float(dur),
    }
    if single:
        return {k: (v[0] if np.ndim(v) else v) for k, v in result.items()}
    return result


# ---------------------------------------------------------------------------
# Monte Carlo over random-field realisations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FatigueMonteCarlo:
    """Outcome of the random-field fatigue Monte Carlo."""

    damage: np.ndarray
    life_s: np.ndarray
    cycles: np.ndarray
    equivalent_range: np.ndarray
    max_range: np.ndarray
    stress_peak: np.ndarray
    stress_rms: np.ndarray
    static_stress_mean: np.ndarray
    duration_s: float
    n_samples: int

    def summary(self) -> Dict[str, float]:
        life = self.life_s
        finite = life[np.isfinite(life)]
        cv = float(np.std(finite) / np.mean(finite)) if finite.size > 1 else float("nan")
        return {
            "n_samples": int(self.n_samples),
            "damage_mean": float(np.mean(self.damage)),
            "damage_std": float(np.std(self.damage)),
            "damage_cv": float(np.std(self.damage) / np.mean(self.damage))
            if np.mean(self.damage) > 0 else float("nan"),
            "life_mean_s": float(np.mean(finite)) if finite.size else float("nan"),
            "life_median_s": float(np.median(finite)) if finite.size else float("nan"),
            "life_std_s": float(np.std(finite)) if finite.size else float("nan"),
            "life_cv": cv,
            "life_p05_s": float(np.percentile(finite, 5)) if finite.size else float("nan"),
            "life_p95_s": float(np.percentile(finite, 95)) if finite.size else float("nan"),
            "cycles_mean": float(np.mean(self.cycles)),
            "equivalent_range_mean_MPa": float(np.mean(self.equivalent_range)) / 1.0e6,
            "stress_peak_mean_MPa": float(np.mean(self.stress_peak)) / 1.0e6,
            "stress_rms_mean_MPa": float(np.mean(self.stress_rms)) / 1.0e6,
            "duration_s": float(self.duration_s),
        }


def monte_carlo_fatigue(
    fields: Iterable[np.ndarray],
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    panel,
    *,
    duration_s: float,
    dt: float,
    envelope: Sequence[np.ndarray],
    hot_spot: tuple[float, float],
    curve: SNCurve = DNV_CURVES["D_air"],
    dynamic: bool = True,
    chunk: int = 50,
) -> FatigueMonteCarlo:
    """Run the whole chain for a family of pressure-field realisations.

    ``fields`` is a sequence (or array) of pressure fields ``(ny, nx)`` in Pa,
    ``envelope`` the matching temporal envelopes ``A(t)`` on the same time base.
    The pressure record of realisation ``i`` is ``A_i(t) * p_i(x, y)``; the modal
    forces, modal response, hot-spot stress, rainflow counting and Miner damage
    follow.
    """
    from .plate import modal_forces, stress_history_from_modal_force

    f = np.asarray(list(fields) if not isinstance(fields, np.ndarray) else fields,
                   dtype=float)
    env = np.asarray(list(envelope) if not isinstance(envelope, np.ndarray) else envelope,
                     dtype=float)
    if f.shape[0] != env.shape[0]:
        raise ValueError("need one envelope per field realisation")
    n_samples = f.shape[0]
    n_t = env.shape[1]
    x_hs, y_hs = hot_spot

    Q = modal_forces(panel, f)                      # (S, M)
    damage = np.zeros(n_samples)
    cycles = np.zeros(n_samples)
    eq = np.zeros(n_samples)
    mx = np.zeros(n_samples)
    peak = np.zeros(n_samples)
    rms = np.zeros(n_samples)
    static_mean = np.zeros(n_samples)

    for start in range(0, n_samples, chunk):
        stop = min(n_samples, start + chunk)
        force = Q[start:stop, :, None] * env[start:stop, None, :]
        sigma = stress_history_from_modal_force(panel, force, dt, x_hs, y_hs,
                                                dynamic=dynamic)
        static_sigma = stress_history_from_modal_force(panel, force, dt, x_hs, y_hs,
                                                       dynamic=False)
        static_mean[start:stop] = static_sigma.mean(axis=1)
        res = rainflow_miner_life(sigma, dt, curve=curve, duration_s=duration_s)
        damage[start:stop] = res["damage"]
        cycles[start:stop] = res["cycles"]
        eq[start:stop] = res["equivalent_range"]
        mx[start:stop] = res["max_range"]
        peak[start:stop] = np.max(np.abs(sigma), axis=1)
        rms[start:stop] = np.sqrt(np.mean(sigma ** 2, axis=1))

    life = np.array([lifetime_from_damage(d, duration_s) for d in damage])
    return FatigueMonteCarlo(
        damage=damage, life_s=life, cycles=cycles,
        equivalent_range=eq, max_range=mx,
        stress_peak=peak, stress_rms=rms, static_stress_mean=static_mean,
        duration_s=float(duration_s), n_samples=int(n_samples),
    )


# ---------------------------------------------------------------------------
# analytic anchor
# ---------------------------------------------------------------------------

SIGNAL_CYCLES_ANCHOR = 0.5   # half-sine: one full cycle per period


def analytic_miner_anchor(
    panel,
    *,
    load_amplitude: float,
    frequency_hz: float,
    n_cycles: int,
    dt: float,
    hot_spot: tuple[float, float],
    load_position: tuple[float, float],
    curve: SNCurve = DNV_CURVES["D_air"],
    n_terms: int = 399,
    dynamic: bool = True,
    grid_nx: int = 60,
    grid_ny: int = 30,
) -> Dict[str, float]:
    """Single-point-load fatigue life against the hand-summed Miner rule.

    The load record is a point force at ``load_position`` following
    ``F(t) = F_a (1 - cos(2 pi f t)) / 2`` for ``n_cycles`` periods.  The
    transfer from a point load to the hot-spot stress is the panel's *static*
    modal transfer ``C`` (Pa/N), so the analytic stress history is
    ``sigma(t) = C F(t)``: every cycle has the same range ``C F_a`` and the
    Miner sum is closed form, ``D = n_cycles / N(C F_a)``.

    The chain path is the one used everywhere else: the point load enters as a
    discrete pressure field (one cell of a ``grid_nx x grid_ny`` grid), is
    projected on the panel modes, integrated with the exact modal recurrence
    (``dynamic=True`` gives the real dynamic amplification, which is what makes
    the two paths differ at all), reduced by rainflow and accumulated by Miner.

    The analytic path deliberately uses the *modal* transfer rather than the
    Navier double series: the series for the bending moment under a point load
    converges very slowly (and only logarithmically at the load point), so it is
    not a usable reference at the hot spot; the modal-vs-Navier agreement is
    verified instead on the uniformly loaded panel, where both converge fast
    (see ``navier_static_uniform`` and the plate checks in the study/tests).
    ``navier_transfer`` is still reported for information.
    """
    from .plate import (modal_forces, static_point_load_stress,
                        stress_history_from_modal_force)

    x0, y0 = load_position
    x_hs, y_hs = hot_spot
    n_t = int(round(n_cycles / (frequency_hz * dt))) + 1
    t = np.arange(n_t) * dt
    load = load_amplitude * 0.5 * (1.0 - np.cos(2.0 * math.pi * frequency_hz * t))

    # -- point load as a one-cell pressure field ---------------------------
    field = np.zeros((grid_ny, grid_nx))
    x_centers = (np.arange(grid_nx) + 0.5) * panel.length / grid_nx
    y_centers = (np.arange(grid_ny) + 0.5) * panel.width / grid_ny
    ix = int(np.clip(np.searchsorted(x_centers, x0), 0, grid_nx - 1))
    iy = int(np.clip(np.searchsorted(y_centers, y0), 0, grid_ny - 1))
    cell_area = (panel.length / grid_nx) * (panel.width / grid_ny)
    field[iy, ix] = 1.0 / cell_area          # unit point load; load(t) multiplies below
    Q_unit = modal_forces(panel, field, x_centers, y_centers)

    # static modal transfer, Pa per N
    unit_static = stress_history_from_modal_force(
        panel, Q_unit[None, :, None] * np.ones((1, 1, 2)), 1.0, x_hs, y_hs,
        dynamic=False)[0][0]
    C = float(unit_static)

    # -- analytic path -----------------------------------------------------
    sigma_analytic = C * load
    n_fail = float(curve.cycles_to_failure(np.array([abs(C) * load_amplitude]))[0])
    analytic_damage = float(n_cycles / n_fail)
    cycles_check = rainflow(sigma_analytic)
    analytic_from_rainflow = miner_damage(
        curve,
        np.array([c.range for c in cycles_check]),
        np.array([c.count for c in cycles_check]))["damage"]

    # -- chain path ---------------------------------------------------------
    force = Q_unit[None, :, None] * load[None, None, :]
    sigma_chain = stress_history_from_modal_force(panel, force, dt, x_hs, y_hs,
                                                  dynamic=dynamic)[0]
    res = rainflow_miner_life(sigma_chain, dt, curve=curve)
    chain_damage = float(res["damage"])

    duration = float(t[-1] - t[0])
    navier = static_point_load_stress(panel, x0, y0, x_hs, y_hs, n_terms=n_terms)
    return {
        "static_transfer_Pa_per_N": C,
        "navier_transfer_Pa_per_N": float(navier),
        "analytic_damage": float(analytic_damage),
        "analytic_damage_from_rainflow": float(analytic_from_rainflow),
        "chain_damage": chain_damage,
        "damage_rel_error_chain_vs_analytic": abs(chain_damage - analytic_damage) / analytic_damage,
        "cycle_range_analytic_MPa": abs(C) * load_amplitude / 1e6,
        "chain_range_max_MPa": float(res["max_range"]) / 1e6,
        "chain_range_equivalent_MPa": float(res["equivalent_range"]) / 1e6,
        "analytic_life_s": lifetime_from_damage(analytic_damage, duration),
        "chain_life_s": lifetime_from_damage(chain_damage, duration),
        "life_rel_error": abs(lifetime_from_damage(chain_damage, duration)
                              - lifetime_from_damage(analytic_damage, duration))
        / lifetime_from_damage(analytic_damage, duration),
        "n_cycles": int(n_cycles),
        "duration_s": duration,
        "load_amplitude_N": float(load_amplitude),
        "frequency_hz": float(frequency_hz),
        "n_terms_navier": int(n_terms),
        "dynamic": bool(dynamic),
        "grid": [int(grid_nx), int(grid_ny)],
        "dynamic_amplification": float(np.max(np.abs(sigma_chain))
                                       / max(np.max(np.abs(sigma_analytic)), 1e-30)),
    }


def panel_grid_shape(panel, n_t: int) -> tuple[int, int]:
    """Default (ny, nx) for the equivalent discrete pressure field."""
    nx = max(8, int(round(panel.length / 0.08)))
    ny = max(4, int(round(panel.width / 0.08)))
    return ny, nx