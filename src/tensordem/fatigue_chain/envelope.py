"""dD/D error envelope of load-history compression, and the minimum-rate law.

Two questions are answered here.

1. **Envelope matrix.**  For one load record, the exact chain damage
   ``D(1 kHz)`` is compared with the chain damage of the same record compressed
   to ``fs in {1, 5, 20, 100} Hz``, once with naive point sampling and once with
   peak-preserving min/max sampling.  The matrix is required to be *non-empty*
   and *monotone*: ``|dD/D|`` must fall as ``fs`` rises, and peak-preserving
   must not be worse than naive at the same rate.

2. **Minimum-rate law.**  A family of pulse trains (pulse half width ``t_p``)
   through an SDOF channel tuned to ``f_n = 0.35 / t_p`` is compressed at a
   dense rate grid; the smallest rate that keeps ``|dD/D| < 10 %`` is recorded
   per S-N slope ``m`` and per compression method, and a power law

       ``fs_min = a * m^b / t_p^c``

   is fitted by least squares in the log domain.  The exponent ``c`` is the
   physically expected ``~1`` (the retained peak fraction scales with
   ``fs * t_p``); ``b`` is set by ``dD/D ~ m * dS/S``.

Both outputs are plain dictionaries so they can be dumped into the node's
artifact JSON without a second serialiser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..whipping.fatigue import SNCurve
from .contract import MissingForceHistory
from .damage import chain_damage
from .resample import compress, reconstruct
from .response import ModalChannel

__all__ = [
    "SamplingEnvelope",
    "sampling_envelope",
    "minimum_sampling_rate",
    "fit_min_rate_law",
    "pulse_train",
    "random_pulse_train",
    "SCAN_RATES_HZ",
]

SCAN_RATES_HZ = (1.0, 5.0, 20.0, 100.0)


@dataclass
class SamplingEnvelope:
    """Error envelope of one load record under sampling-rate compression."""

    label: str
    baseline_damage: float
    baseline_cycles: float
    rows: List[Dict[str, float]] = field(default_factory=list)

    def subset(self, method: str) -> List[Dict[str, float]]:
        return [r for r in self.rows if r["method"] == method]

    def monotone(self, *, rtol: float = 1e-9) -> Dict[str, object]:
        """``|dD/D|`` must not increase with the rate, per method."""
        out: Dict[str, object] = {}
        for method in sorted({r["method"] for r in self.rows}):
            rows = sorted(self.subset(method), key=lambda r: r["rate_hz"])
            errs = [abs(r["dD_over_D"]) for r in rows]
            viol = []
            for i in range(1, len(errs)):
                if errs[i] > errs[i - 1] * (1.0 + rtol) + 1e-12:
                    viol.append({"rate_hz": rows[i]["rate_hz"],
                                 "err": errs[i], "prev_err": errs[i - 1]})
            out[method] = {"monotone": not viol, "violations": viol,
                           "errors": errs, "rates": [r["rate_hz"] for r in rows]}
        return out

    def peak_vs_naive(self, *, tol: float = 0.0) -> Dict[str, object]:
        """Peak-preserving must not be worse than naive at every rate."""
        by_rate: Dict[float, Dict[str, float]] = {}
        for r in self.rows:
            by_rate.setdefault(r["rate_hz"], {})[r["method"]] = abs(r["dD_over_D"])
        viol = []
        for rate, errs in sorted(by_rate.items()):
            if "peak" in errs and "naive" in errs:
                if errs["peak"] > errs["naive"] + tol:
                    viol.append({"rate_hz": rate, "peak": errs["peak"],
                                 "naive": errs["naive"]})
        return {"peak_not_worse": not viol, "violations": viol}

    def min_rate(self, method: str, threshold: float = 0.10) -> Optional[float]:
        return minimum_sampling_rate(self.rows, method, threshold)

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "baseline_damage": self.baseline_damage,
            "baseline_cycles": self.baseline_cycles,
            "rows": self.rows,
            "monotone": self.monotone(),
            "peak_vs_naive": self.peak_vs_naive(),
            "min_rate_10pct": {
                method: self.min_rate(method) for method in ("naive", "peak")
            },
        }


def sampling_envelope(t, force, channels: Sequence[ModalChannel], curve: SNCurve,
                      rates: Sequence[float] = SCAN_RATES_HZ,
                      methods: Sequence[str] = ("naive", "peak"),
                      label: str = "") -> SamplingEnvelope:
    """Build the compression error envelope of one force record."""
    if force is None:
        raise MissingForceHistory("no force history supplied to the envelope")
    tt = np.asarray(t, dtype=float).ravel()
    ff = np.asarray(force, dtype=float).ravel()
    if tt.size == 0:
        raise MissingForceHistory("empty force history supplied to the envelope")
    base = chain_damage(tt, ff, channels, curve)
    env = SamplingEnvelope(label=label, baseline_damage=base.damage,
                           baseline_cycles=base.n_cycles)
    for rate in rates:
        for method in methods:
            tc, fc = compress(tt, ff, float(rate), method=method)
            fr = reconstruct(tt, tc, fc)
            res = chain_damage(tt, fr, channels, curve)
            env.rows.append({
                "rate_hz": float(rate),
                "method": method,
                "n_retained": int(tc.size),
                "retention": float(tc.size) / float(tt.size),
                "damage": float(res.damage),
                "dD_over_D": float(res.damage / base.damage - 1.0) if base.damage > 0 else float("nan"),
                "dsigma_eq_Pa": float(res.dsigma_eq_Pa),
                "n_cycles": float(res.n_cycles),
                "stress_peak_Pa": float(res.stress_peak_Pa),
            })
    return env


def minimum_sampling_rate(rows: Sequence[Dict[str, float]], method: str,
                          threshold: float = 0.10) -> Optional[float]:
    """Smallest tested rate whose ``|dD/D|`` is below the threshold.

    Returns ``None`` when no tested rate meets the threshold (which is reported
    as an honest "not reached in this grid", not as a pass).
    """
    ok = [r["rate_hz"] for r in rows
          if r["method"] == method and abs(r["dD_over_D"]) <= threshold]
    if not ok:
        return None
    return float(min(ok))


def pulse_train(t_end: float, rate_hz: float, pulse_width_s: float,
                period_s: float, peak: float = 1.0, t_start: float = 0.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular-pulse train (``pulse_width_s`` = full width at the base)."""
    n = int(round(t_end * rate_hz)) + 1
    t = t_start + np.arange(n) / rate_hz
    phase = np.mod(t, period_s)
    f = np.where(phase <= pulse_width_s, peak, 0.0)
    return t, f


def random_pulse_train(t_end: float, rate_hz: float, pulse_width_s: float,
                       mean_interval_s: float, peak: float = 1.0,
                       amplitude_lo: float = 0.6, seed: int = 11
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Non-periodic rectangular-pulse train (exponential arrivals, random peaks).

    A periodic train phase-locks with the sampling grid and produces
    non-monotone, essentially random ``dD/D`` (measured: peak-preserving error
    oscillating between -2.6 % and +4.7 % for 1000-4000 Hz pulses at 300-500 Hz
    sampling).  Random arrivals remove the phase locking, which is what makes
    the minimum-rate law measurable.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(t_end * rate_hz))) / rate_hz
    f = np.zeros_like(t)
    times_out = []
    now = rng.exponential(mean_interval_s)
    while now < t_end:
        amp = peak * rng.uniform(amplitude_lo, 1.0)
        times_out.append(now)
        f = np.where((t >= now) & (t < now + pulse_width_s),
                     np.maximum(f, amp), f)
        now += rng.exponential(mean_interval_s)
    return t, f


def fit_min_rate_law(samples: Sequence[Dict[str, float]]) -> Dict[str, object]:
    """Least-squares fit of ``fs_min = a * m^b / t_p^c`` in the log domain."""
    if len(samples) < 4:
        raise ValueError("at least four (m, t_p, fs_min) points are required")
    m = np.array([s["m"] for s in samples], dtype=float)
    tp = np.array([s["pulse_width_s"] for s in samples], dtype=float)
    fs = np.array([s["fs_min_hz"] for s in samples], dtype=float)
    if np.any(m <= 0) or np.any(tp <= 0) or np.any(fs <= 0):
        raise ValueError("all fit inputs must be positive")
    design = np.column_stack([np.ones_like(m), np.log(m), -np.log(tp)])
    target = np.log(fs)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    pred = design @ coef
    resid = target - pred
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 1.0
    a = math.exp(coef[0])
    return {
        "form": "fs_min_Hz = a * m^b / pulse_width_s^c",
        "a": a,
        "b": float(coef[1]),
        "c": float(coef[2]),
        "r2_log": r2,
        "max_rel_residual": float(np.max(np.abs(np.exp(pred) / fs - 1.0))),
        "n_points": int(m.size),
        "a_in_formula": a,
        "examples": [
            {"m": float(mm), "pulse_width_s": float(tt), "fs_min_measured_hz": float(ff),
             "fs_min_fit_hz": float(np.exp(pp))}
            for mm, tt, ff, pp in zip(m, tp, fs, pred)
        ],
    }