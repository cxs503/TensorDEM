"""Two independent wave-speed / attenuation estimators for a granular packing.

Both estimators work on the same physical setup (bottom platen as the source,
two slab-averaged receivers at ``z1 < z2``) so that the only difference between
them is the signal-processing route:

``pulse_wave_speed``   time domain: a short sine-squared velocity pulse is
                       applied to the bottom platen; the transit time is the
                       cross-correlation lag between the two receiver signals
                       (parabolically interpolated to sub-step precision).

``sweep_wave_speed``   frequency domain: the platen is driven sinusoidally at
                       several frequencies; lock-in detection gives the complex
                       amplitude at each receiver, the phase lag between the
                       receivers grows linearly with frequency and the slope
                       yields the phase velocity.

Attenuation is reported as ``Q^-1`` from (i) the peak-amplitude decay between the
two receivers and (ii) the slope of the log spectral ratio (``ln(A1/A2) =
pi f dz /(Q v)``); two *different* estimators on the same data, per the
acceptance protocol.

Design notes
------------
* The receivers average the particle velocity over a slab ~1.5 grain diameters
  thick, which suppresses single-grain rattling without smoothing the arrival.
* A quadratic sponge layer above the upper receiver absorbs the up-going wave
  and removes platen reflections from the lock-in window (the ``Q`` estimators
  would otherwise see a standing wave).
* The drive amplitude keeps the strain at O(1e-4), i.e. the linear regime, and
  the contact forces stay far below the Coulomb cap away from the control case.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Iterable, Sequence

import numpy as np

from .engine import GranularPacking


# --------------------------------------------------------------------------
# Signal helpers
# --------------------------------------------------------------------------
def cross_correlation_lag(a: Sequence[float], b: Sequence[float], dt: float) -> dict[str, Any]:
    """Lag (seconds) such that ``b(t) ~ a(t - lag)``, sub-step interpolated.

    Returns a dict with ``lag_s``, ``lag_steps``, ``peak_correlation`` and
    ``relative_peak_height`` (peak / zero-lag value) so a caller can judge
    whether the pick is meaningful.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size != y.size or x.size < 8:
        raise ValueError("signals must be equal length and at least 8 samples")
    n = x.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    corr = np.fft.irfft(np.fft.rfft(x, nfft).conj() * np.fft.rfft(y, nfft), nfft)
    # corr[k] = sum_t x[t] y[t + k]; negative lags wrap around
    lags = np.concatenate((np.arange(0, n), np.arange(-n + 1, 0)))
    vals = np.concatenate((corr[:n], corr[nfft - n + 1:]))
    order = np.argsort(lags)
    lags, vals = lags[order], vals[order]
    valid = np.abs(lags) < max(3, n // 2)
    k = int(np.argmax(np.where(valid, vals, -np.inf)))
    frac = 0.0
    if 0 < k < len(vals) - 1:
        y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
        denom = (y0 - 2.0 * y1 + y2)
        if abs(denom) > 1e-30:
            frac = 0.5 * (y0 - y2) / denom
            frac = float(np.clip(frac, -0.5, 0.5))
    lag_steps = float(lags[k]) + frac
    zero = float(np.interp(0.0, lags, vals))
    return {
        "lag_s": lag_steps * dt,
        "lag_steps": lag_steps,
        "peak_correlation": float(vals[k]),
        "relative_peak_height": float(vals[k] / zero) if abs(zero) > 1e-30 else float("inf"),
    }


def _slab_indices(sim: GranularPacking, z_lo: float, z_hi: float) -> np.ndarray:
    z = sim.pos[:, 2].detach().cpu().numpy()
    return np.nonzero((z >= z_lo) & (z <= z_hi))[0]


def _receiver_layout(sim: GranularPacking, fractions: Sequence[float]) -> dict[str, Any]:
    z_bottom = float(sim.wall_pos[4]) + float(sim.radius.mean())
    z_top = float(sim.wall_pos[5]) - float(sim.radius.mean())
    height = z_top - z_bottom
    thickness = 1.5 * float(sim.radius.mean())
    centres = [z_bottom + f * height for f in fractions]
    slabs = [(_slab_indices(sim, c - 0.5 * thickness, c + 0.5 * thickness), c)
             for c in centres]
    return {
        "z_bottom": z_bottom,
        "z_top": z_top,
        "height": height,
        "thickness": thickness,
        "centres": centres,
        "slabs": slabs,
        "distance": centres[1] - centres[0],
    }


def _sponge_setup(sim: GranularPacking, layout: dict[str, Any], *, start_fraction: float,
                  v_ref: float) -> None:
    z0 = layout["z_bottom"] + start_fraction * layout["height"]
    length = max(layout["z_top"] - z0, 1e-9)
    sim.sponge = {"z0": z0, "z_top": layout["z_top"],
                  "gamma": 4.0 * v_ref / length}


# --------------------------------------------------------------------------
# Attenuation
# --------------------------------------------------------------------------
def _peak_amplitude(signal: np.ndarray, start: int) -> float:
    seg = np.abs(signal[start:])
    return float(seg.max()) if seg.size else 0.0


def _spectral_ratio_q(sig1: np.ndarray, sig2: np.ndarray, dt: float, arrival1: int,
                      distance: float, speed: float, window: int) -> dict[str, float]:
    """``Q^-1`` from the slope of the log spectral ratio (spectral-ratio method)."""
    n = len(sig1)
    lo = max(0, arrival1 - window // 2)
    hi = min(n, lo + window)
    if hi - lo < 16:
        return {"q_inv_spectral": float("nan"), "spectral_slope": float("nan"),
                "centroid_frequency_hz": float("nan")}
    taper = np.hanning(hi - lo)
    seg1 = sig1[lo:hi] * taper
    seg2 = sig2[lo:hi] * taper
    f = np.fft.rfftfreq(len(seg1), d=dt)
    a1 = np.abs(np.fft.rfft(seg1))
    a2 = np.abs(np.fft.rfft(seg2))
    band = (f > 0.3 * f[np.argmax(a1)]) & (f < 2.0 * f[np.argmax(a1)]) & (a1 > 1e-6 * a1.max()) \
        & (a2 > 1e-6 * a2.max())
    centroid = float((f * a1).sum() / max(a1.sum(), 1e-30))
    if band.sum() < 4:
        return {"q_inv_spectral": float("nan"), "spectral_slope": float("nan"),
                "centroid_frequency_hz": centroid}
    ratio = np.log(np.clip(a1[band], 1e-30, None) / np.clip(a2[band], 1e-30, None))
    slope = float(np.polyfit(f[band], ratio, 1)[0])
    q_inv = slope * speed / (math.pi * distance)
    return {"q_inv_spectral": float(q_inv), "spectral_slope": slope,
            "centroid_frequency_hz": centroid}


@dataclass
class PulseResult:
    """Result of one time-domain (travel-time) wave measurement."""

    mode: str
    speed_m_s: float
    travel_time_s: float
    distance_m: float
    amplitude_ratio: float
    attenuation_1_m: float
    q_inv_peak: float
    q_inv_spectral: float
    centroid_frequency_hz: float
    signal_to_noise: float
    receivers_m: tuple[float, float]
    source_amplitude_m_s: float
    valid: bool = True
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SweepResult:
    """Result of one frequency-domain (lock-in) wave measurement."""

    mode: str
    speed_m_s: float
    phase_lag_rad: tuple[float, ...]
    frequencies_hz: tuple[float, ...]
    amplitude_ratio: tuple[float, ...]
    q_inv: float
    distance_m: float
    source_amplitude_m_s: float
    valid: bool = True
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _drive(sim: GranularPacking, mode: str, value: float) -> None:
    """Set the bottom platen drive (normal for P, tangential-x for S)."""
    sim.wall_vnormal[4] = value if mode == "P" else 0.0
    sim.wall_tangential_drive.zero_()
    if mode == "S":
        sim.wall_tangential_drive[4, 0] = value
        sim.wall_tangential_drive[5, 0] = 0.0


def pulse_wave_speed(sim: GranularPacking, *, mode: str = "P", amplitude: float = 0.02,
                     width_s: float | None = None, fractions: Sequence[float] = (0.30, 0.58),
                     n_steps: int | None = None, sponge_start: float = 0.93,
                     v_ref: float | None = None) -> PulseResult:
    """Time-domain (travel-time) wave speed via a source pulse.

    Args:
        mode:      ``"P"`` (platen normal) or ``"S"`` (platen tangential x).
        amplitude: peak platen velocity, m/s (keep the strain at O(1e-4)).
        width_s:   pulse width; defaults to 1.2 grain diameters / v_ref.
        fractions: receiver positions as fractions of the platen gap.
        n_steps:   record length; defaults to cover the transit plus the sponge.
        v_ref:     rough speed estimate used only to size the pulse / sponge.
    """
    if mode not in ("P", "S"):
        raise ValueError("mode must be 'P' or 'S'")
    r_mean = float(sim.radius.mean())
    v_ref = float(v_ref) if v_ref else 200.0
    layout = _receiver_layout(sim, fractions)
    # 12 grain diameters: the wavelength must be several grain diameters for the
    # affine continuum wave to exist at all (a shorter pulse just rings the
    # individual contacts at their own resonance)
    width = float(width_s) if width_s else 12.0 * 2.0 * r_mean / v_ref
    if any(len(s[0]) == 0 for s in layout["slabs"]):
        raise RuntimeError("receiver slab empty: increase packing height")
    _sponge_setup(sim, layout, start_fraction=sponge_start, v_ref=v_ref)
    if n_steps is None:
        # the record must contain the pulse, the transit *and* the arrival at the
        # far receiver (a truncated record silently corrupts the cross-correlation)
        n_steps = int(math.ceil((width + 2.6 * layout["distance"] / v_ref) / sim.dt)) + 40
    sim.vel.zero_()
    sim.omega.zero_()
    idx1, idx2 = layout["slabs"][0][0], layout["slabs"][1][0]
    comp = 2 if mode == "P" else 0
    sig1 = np.zeros(n_steps)
    sig2 = np.zeros(n_steps)
    for n in range(n_steps):
        t = n * sim.dt
        drive = amplitude * math.sin(math.pi * t / width) ** 2 if t < width else 0.0
        _drive(sim, mode, drive)
        sim.step()
        v = sim.vel.detach().cpu().numpy()
        sig1[n] = v[idx1, comp].mean()
        sig2[n] = v[idx2, comp].mean()
    sim.wall_vnormal.zero_()
    sim.wall_tangential_drive.zero_()
    sim.sponge = None
    sim.rebuild_pairs()

    lag = cross_correlation_lag(sig1, sig2, sim.dt)
    travel = lag["lag_s"]
    if travel <= 0:
        # a non-positive lag means the near receiver did not see a travelling
        # pulse (common when the box is only a few wavelengths long and the
        # response is dominated by a global mode); report it, do not hide it
        return PulseResult(
            mode=mode, speed_m_s=float("nan"), travel_time_s=float(travel),
            distance_m=float(layout["distance"]), amplitude_ratio=float("nan"),
            attenuation_1_m=float("nan"), q_inv_peak=float("nan"),
            q_inv_spectral=float("nan"), centroid_frequency_hz=float("nan"),
            signal_to_noise=float("nan"),
            receivers_m=(layout["centres"][0], layout["centres"][1]),
            source_amplitude_m_s=float(amplitude), valid=False,
            note="non-positive cross-correlation lag: no travelling pulse detected")
    if travel <= 0:
        raise RuntimeError(f"{mode} wave: non-positive travel time ({travel:.3e} s)")
    speed = layout["distance"] / travel
    a1 = _peak_amplitude(sig1, 0)
    a2 = _peak_amplitude(sig2, 0)
    ratio = a2 / a1 if a1 > 0 else float("nan")
    attenuation = -math.log(ratio) / layout["distance"] if ratio > 0 else float("nan")
    arrival1 = int(np.argmax(np.abs(sig1)))
    spec = _spectral_ratio_q(sig1, sig2, sim.dt, arrival1, layout["distance"], speed,
                             window=max(16, int(2.0 * width / sim.dt)))
    f_c = spec["centroid_frequency_hz"]
    q_inv_peak = (attenuation * speed / (math.pi * f_c)
                  if f_c and math.isfinite(f_c) and f_c > 0 else float("nan"))
    pre = np.abs(sig1[: max(4, arrival1 // 2)])
    noise = float(pre.max()) if pre.size else 0.0
    snr = a1 / noise if noise > 0 else float("inf")
    return PulseResult(
        mode=mode,
        speed_m_s=float(speed),
        travel_time_s=float(travel),
        distance_m=float(layout["distance"]),
        amplitude_ratio=float(ratio),
        attenuation_1_m=float(attenuation),
        q_inv_peak=float(q_inv_peak),
        q_inv_spectral=float(spec["q_inv_spectral"]),
        centroid_frequency_hz=float(f_c),
        signal_to_noise=float(snr),
        receivers_m=(layout["centres"][0], layout["centres"][1]),
        source_amplitude_m_s=float(amplitude),
    )


def sweep_wave_speed(sim: GranularPacking, *, mode: str = "P",
                     frequencies: Iterable[float], amplitude: float = 0.02,
                     fractions: Sequence[float] = (0.15, 0.85),
                     warmup_cycles: int = 8, measure_cycles: int = 6,
                     sponge_start: float = 0.93,
                     v_ref: float | None = None) -> SweepResult:
    """Frequency-domain (lock-in) phase velocity and attenuation.

    The platen is driven at each frequency in turn; the complex receiver
    amplitudes are recovered by lock-in over the last ``measure_cycles`` and the
    phase lag between receivers is fitted linearly in frequency, which gives the
    phase velocity without any time-domain picking.
    """
    if mode not in ("P", "S"):
        raise ValueError("mode must be 'P' or 'S'")
    freqs = [float(f) for f in frequencies]
    if len(freqs) < 2:
        raise ValueError("sweep needs at least two frequencies")
    v_ref = float(v_ref) if v_ref else 200.0
    layout = _receiver_layout(sim, fractions)
    if any(len(s[0]) == 0 for s in layout["slabs"]):
        raise RuntimeError("receiver slab empty: increase packing height")
    _sponge_setup(sim, layout, start_fraction=sponge_start, v_ref=v_ref)
    idx1, idx2 = layout["slabs"][0][0], layout["slabs"][1][0]
    comp = 2 if mode == "P" else 0
    phases: list[float] = []
    ratios: list[float] = []
    for freq in freqs:
        period = 1.0 / freq
        n_warm = int(math.ceil(warmup_cycles * period / sim.dt))
        n_meas = int(math.ceil(measure_cycles * period / sim.dt))
        sim.vel.zero_()
        sim.omega.zero_()
        rec1 = np.zeros(n_meas)
        rec2 = np.zeros(n_meas)
        t_start = 0.0
        for n in range(n_warm + n_meas):
            t = n * sim.dt
            _drive(sim, mode, amplitude * math.sin(2.0 * math.pi * freq * t))
            sim.step()
            if n == n_warm:
                t_start = sim.time
            if n >= n_warm:
                v = sim.vel.detach().cpu().numpy()
                rec1[n - n_warm] = v[idx1, comp].mean()
                rec2[n - n_warm] = v[idx2, comp].mean()
        tt = t_start + np.arange(n_meas) * sim.dt
        w = 2.0 * math.pi * freq
        kernel = np.exp(-1j * w * tt)
        z1 = float(np.abs(2.0 * (rec1 * kernel).mean()))
        z2 = float(np.abs(2.0 * (rec2 * kernel).mean()))
        ph1 = float(np.angle((rec1 * kernel).mean()))
        ph2 = float(np.angle((rec2 * kernel).mean()))
        dphi = (ph1 - ph2) % (2.0 * math.pi)     # >0: receiver 2 lags receiver 1
        phases.append(dphi)
        ratios.append(z2 / z1 if z1 > 0 else float("nan"))
    sim.wall_vnormal.zero_()
    sim.wall_tangential_drive.zero_()
    sim.sponge = None
    sim.rebuild_pairs()

    f_arr = np.asarray(freqs)
    dphi_arr = np.asarray(phases)
    slope = float(np.polyfit(f_arr, dphi_arr, 1)[0])     # rad per Hz
    if slope <= 0:
        return SweepResult(
            mode=mode, speed_m_s=float("nan"),
            phase_lag_rad=tuple(float(v) for v in phases),
            frequencies_hz=tuple(float(v) for v in freqs),
            amplitude_ratio=tuple(float(v) for v in ratios), q_inv=float("nan"),
            distance_m=float(layout["distance"]),
            source_amplitude_m_s=float(amplitude), valid=False,
            note="non-positive phase-lag slope: the far receiver does not lag")
    speed = 2.0 * math.pi * layout["distance"] / slope
    # attenuation from the amplitude ratios: ln(A1/A2) = pi f dz / (Q v)
    log_ratio = -np.log(np.clip(np.asarray(ratios), 1e-30, None))
    q_inv_terms = log_ratio * speed / (math.pi * f_arr * layout["distance"])
    q_inv = float(np.nanmean(q_inv_terms))
    return SweepResult(
        mode=mode,
        speed_m_s=float(speed),
        phase_lag_rad=tuple(float(v) for v in phases),
        frequencies_hz=tuple(float(v) for v in freqs),
        amplitude_ratio=tuple(float(v) for v in ratios),
        q_inv=q_inv,
        distance_m=float(layout["distance"]),
        source_amplitude_m_s=float(amplitude),
    )