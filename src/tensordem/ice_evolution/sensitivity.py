"""Parameter sensitivity: temperature, initial porosity, transit interval
(campaign IN155-N09).

Sweeps the three control parameters and emits the resistance-decay envelope
``R_ice(N)/R_ice(1)``.  The DEM calibration is performed once at the
reference temperature; the temperature dependence then enters through the
Frenkel neck growth (viscosity) and the refreezing rate, which is the
dominant physics.

Units are SI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sintering import frenkel_neck_radius
from .refreezing import RefreezeConfig, run_refreezing
from .strength import StrengthConfig, rubble_strength
from .resistance import PlatenConfig
from .channel import ChannelConfig, calibrate


@dataclass(frozen=True)
class SensitivityConfig:
    """Sweep definition."""
    temperatures_k: tuple = (253.15, 258.15, 263.15, 268.15)   # -20..-5 C
    phi_0_values: tuple = (0.2, 0.3, 0.4, 0.5)
    interval_hours: tuple = (1.0, 6.0, 12.0, 24.0)
    n_transits: int = 24
    particle_radius_m: float = 0.010
    resistance_exponent: float = 0.5
    platen: PlatenConfig = field(default_factory=PlatenConfig)
    seed: int = 0


def _ratio_for(temperature_k: float, phi_0: float, interval_s: float,
               n_transits: int, a_cal: np.ndarray, r_cal: np.ndarray,
               particle_radius_m: float, resistance_exponent: float,
               k_ref: float) -> np.ndarray:
    """R(N)/R(1) for one (T, phi_0, interval) combination (no per-case DEM)."""
    total = (n_transits - 1) * interval_s
    refz = run_refreezing(
        RefreezeConfig(thickness_m=0.05,
                        temperature_k=temperature_k, phi_0=phi_0,
                        k_ref=k_ref),
        duration_s=total, record_every_s=max(interval_s, 60.0))
    t_refz, phi_refz = refz.porosity_series()
    ratio = np.empty(n_transits)
    for k in range(n_transits):
        t_heal = k * interval_s
        a = frenkel_neck_radius(t_heal, temperature_k,
                                particle_radius=particle_radius_m)
        phi_k = float(np.interp(t_heal, t_refz, phi_refz))
        r_a = float(np.interp(a, a_cal, r_cal))
        ratio[k] = r_a * ((1.0 - phi_k) / (1.0 - phi_0)) ** resistance_exponent
    return ratio / ratio[0]


def run_sensitivity(cfg: SensitivityConfig) -> dict:
    """Run the sweep and return the table + envelope."""
    # calibrate once at the reference (coldest) temperature
    ref_T = min(cfg.temperatures_k)
    base = cfg.platen
    calib = calibrate(ChannelConfig(
        n_transits=cfg.n_transits,
        interval_s=12.0 * 3600.0,
        temperature_k=ref_T,
        phi_0=0.30,
        particle_radius_m=cfg.particle_radius_m,
        n_calibration=4,
        platen=base))
    a_cal = np.array([lv["a_m"] for lv in calib["levels"]])
    r_cal = np.array([lv["resistance_n"] for lv in calib["levels"]])
    order = np.argsort(a_cal)
    a_cal, r_cal = a_cal[order], r_cal[order]
    r_cal = np.maximum.accumulate(np.maximum(r_cal, 1e-9))

    rows = []
    for T in cfg.temperatures_k:
        for phi_0 in cfg.phi_0_values:
            for ih in cfg.interval_hours:
                interval_s = ih * 3600.0
                # colder -> faster refreezing (Arrhenius anchored at 263 K)
                k_ref = 5.0e-5 * np.exp(-60.0e3 / 8.314462618
                                        * (1.0 / T - 1.0 / 263.15))
                ratio = _ratio_for(T, phi_0, interval_s, cfg.n_transits,
                                    a_cal, r_cal, cfg.particle_radius_m,
                                    cfg.resistance_exponent, float(k_ref))
                N = np.arange(1, n_transits + 1)
                logN = np.log(N)
                A = np.vstack([logN, np.ones_like(logN)]).T
                (alpha, _), *_ = np.linalg.lstsq(A, ratio - 1.0, rcond=None)
                rows.append({
                    "temperature_k": float(T),
                    "phi_0": float(phi_0),
                    "interval_h": float(ih),
                    "alpha": float(alpha),
                    "ratio_final": float(ratio[-1]),
                    "monotone": bool(np.all(np.diff(ratio) >= -1e-12)),
                    "ratio": ratio.tolist(),
                })
    # envelope over the sweep at each N
    n = cfg.n_transits
    all_ratios = np.array([r["ratio"] for r in rows])
    envelope = {
        "n_transits": list(range(1, n + 1)),
        "lower": all_ratios.min(axis=0).tolist(),
        "upper": all_ratios.max(axis=0).tolist(),
        "mean": all_ratios.mean(axis=0).tolist(),
    }
    return {"rows": rows, "envelope": envelope, "n_rows": len(rows)}
