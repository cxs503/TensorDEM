"""Online comparison of the inverted ice load with the DEM prognosis.

The twin is only useful if it is continuously checked against the independent
DEM forecast: the comparison produces the three acceptance metrics of the node
(correlation, peak error, energy error) and, when a metric exceeds its tolerance,
a *model-correction trigger* together with the correction it implies.

Two corrections are distinguishable and must not be mixed up:

* an amplitude (gain) mismatch — the DEM ice load is too strong/weak by a factor;
  the suggested correction is ``scale = rms(reference) / rms(estimate)``, which
  equals the least-squares gain between the two records;
* a shape/phase mismatch — low correlation, which no scalar gain can repair and
  which is reported as ``shape_mismatch`` so that the operator knows the DEM
  event parametrisation (not the structural model) has to be revisited.

The correlation is always evaluated on the band-limited records (the inversion
bandwidth is finite), while the peak/energy errors are reported both raw and
band-limited, so that a "pass" can never be an artefact of an unstated filter.
"""

from __future__ import annotations

import numpy as np

from .load_inversion import load_error_metrics

__all__ = ["event_metrics", "compare_events", "correction_trigger"]


def event_metrics(estimate, reference, dt: float, *, bandlimit_hz: float | None = None,
                  cut_edges: float = 0.05) -> dict:
    """Metrics of one inverted load against its DEM reference."""
    raw = load_error_metrics(estimate, reference, dt, cut_edges=cut_edges,
                             bandlimit_hz=None)
    band = load_error_metrics(estimate, reference, dt, cut_edges=cut_edges,
                              bandlimit_hz=bandlimit_hz) if bandlimit_hz else raw
    est = np.asarray(estimate, dtype=float).ravel()
    ref = np.asarray(reference, dtype=float).ravel()
    lag = _lag_samples(est, ref, max_lag=int(round(0.5 / dt)))
    return {
        "peak_error": band["peak_error"],
        "energy_error": band["energy_error"],
        "correlation": band["correlation"],
        "rms_error": band["rms_error"],
        "peak_error_raw": raw["peak_error"],
        "energy_error_raw": raw["energy_error"],
        "correlation_raw": raw["correlation"],
        "peak_reference_N": raw["peak_reference_N"],
        "peak_estimate_N": raw["peak_estimate_N"],
        "lag_samples": lag,
        "lag_s": lag * dt,
        "bandlimit_hz": bandlimit_hz,
        "dt_s": dt,
    }


def _lag_samples(estimate: np.ndarray, reference: np.ndarray, max_lag: int) -> int:
    e = estimate - estimate.mean()
    r = reference - reference.mean()
    if e.size < 2 or not np.any(e) or not np.any(r):
        return 0
    n = min(e.size, 4 * max_lag + 1)
    e, r = e[:n], r[:n]
    lags = range(-min(max_lag, n - 2), min(max_lag, n - 2) + 1)
    best, best_lag = -np.inf, 0
    for lag in lags:
        if lag >= 0:
            a, b = e[lag:], r[: r.size - lag]
        else:
            a, b = e[: e.size + lag], r[-lag:]
        if a.size < 3:
            continue
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0.0:
            continue
        value = float(np.dot(a, b) / denom)
        if value > best:
            best, best_lag = value, lag
    return int(best_lag)


def correction_trigger(metrics: dict, *, peak_tolerance: float = 0.30,
                       energy_tolerance: float = 0.30,
                       correlation_floor: float = 0.70) -> dict:
    """Decide whether the DEM prognosis must be corrected after one event."""
    peak_bad = metrics["peak_error"] > peak_tolerance
    energy_bad = metrics["energy_error"] > energy_tolerance
    shape_bad = metrics["correlation"] < correlation_floor
    triggered = bool(peak_bad or energy_bad or shape_bad)
    gain = None
    if triggered and metrics["peak_estimate_N"] > 0.0:
        gain = float(metrics["peak_reference_N"] / metrics["peak_estimate_N"])
    return {
        "triggered": triggered,
        "peak_exceeded": bool(peak_bad),
        "energy_exceeded": bool(energy_bad),
        "shape_mismatch": bool(shape_bad),
        "suggested_amplitude_scale": gain,
        "peak_tolerance": peak_tolerance,
        "energy_tolerance": energy_tolerance,
        "correlation_floor": correlation_floor,
    }


def compare_events(records, *, peak_tolerance: float = 0.30,
                   energy_tolerance: float = 0.30,
                   correlation_floor: float = 0.70,
                   min_events: int = 5) -> dict:
    """Aggregate the per-event comparison into the node's acceptance summary.

    ``records`` is a sequence of dicts with ``estimate``, ``reference``, ``dt``
    and an optional ``label``/``bandlimit_hz``.
    """
    events = []
    for rec in records:
        m = event_metrics(rec["estimate"], rec["reference"], rec["dt"],
                          bandlimit_hz=rec.get("bandlimit_hz"),
                          cut_edges=rec.get("cut_edges", 0.05))
        m["label"] = rec.get("label", f"event_{len(events) + 1}")
        m["correction"] = correction_trigger(
            m, peak_tolerance=peak_tolerance,
            energy_tolerance=energy_tolerance,
            correlation_floor=correlation_floor)
        events.append(m)
    if not events:
        raise ValueError("at least one event is required")
    corr = np.array([e["correlation"] for e in events])
    peak = np.array([e["peak_error"] for e in events])
    energy = np.array([e["energy_error"] for e in events])
    summary = {
        "n_events": len(events),
        "n_events_meets_minimum": bool(len(events) >= min_events),
        "correlation_mean": float(corr.mean()),
        "correlation_min": float(corr.min()),
        "correlation_max": float(corr.max()),
        "peak_error_mean": float(peak.mean()),
        "peak_error_max": float(peak.max()),
        "energy_error_mean": float(energy.mean()),
        "energy_error_max": float(energy.max()),
        "n_corrections_triggered": int(sum(e["correction"]["triggered"] for e in events)),
        "correlation_floor": correlation_floor,
        "peak_tolerance": peak_tolerance,
        "energy_tolerance": energy_tolerance,
        "events": events,
    }
    return summary