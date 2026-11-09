"""N-transit broken-ice channel: resistance growth with the number of transits
(campaign IN155-N09).

A ship repeatedly transits a channel it has broken into rubble.  Between
transits the rubble heals (sintering + refreezing), so the rubble strength
``sigma(N)`` and the ice resistance ``R_ice(N)`` grow with the transit count.

The driver
----------
1. **DEM calibration** -- the platen-through-rubble DEM is run at several
   sintered-neck levels ``a`` (i.e. several healing times), giving the
   resistance ``R(a)`` and the per-transit force histories.
2. **Healing evolution** -- the refreezing model gives ``phi(t)`` and the
   Frenkel law gives the neck ratio ``a(t)/R``; the strength model gives
   ``sigma(t)``.
3. **Transit loop** -- for transit ``N`` the field has healed for
   ``(N-1) * interval``; the resistance is the calibrated ``R(a)`` with a
   solid-fraction correction ``((1-phi)/(1-phi_0))**m``.
4. **Log fit** -- ``R_ice(N)/R_ice(1) = 1 + alpha log(N)``; ``alpha`` must lie
   in ``[0.05, 0.2]`` and the curve must be monotone (criterion d).

Units are SI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .sintering import frenkel_neck_radius, sinter_bond
from .refreezing import RefreezeConfig, run_refreezing
from .strength import StrengthConfig, rubble_strength
from .resistance import PlatenConfig, run_platen


@dataclass(frozen=True)
class ChannelConfig:
    """Configuration of the N-transit channel study."""
    n_transits: int = 24
    interval_s: float = 6.0 * 3600.0     # healing time between transits
    temperature_k: float = 263.15
    phi_0: float = 0.30
    particle_radius_m: float = 0.010
    # DEM calibration
    n_calibration: int = 5
    platen: PlatenConfig = field(default_factory=PlatenConfig)
    # refreezing
    refreeze: RefreezeConfig = field(
        default_factory=lambda: RefreezeConfig(phi_0=0.30))
    strength: StrengthConfig = field(
        default_factory=lambda: StrengthConfig(phi_0=0.30))
    resistance_exponent: float = 0.5     # R ~ sigma^m
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_transits < 2:
            raise ValueError("n_transits must be >= 2")
        if self.interval_s <= 0.0:
            raise ValueError("interval_s must be positive")


@dataclass
class ChannelResult:
    n_transits: np.ndarray
    resistance: np.ndarray              # R_ice(N), N
    resistance_ratio: np.ndarray        # R_ice(N)/R_ice(1)
    sigma: np.ndarray                   # sigma(N), Pa
    porosity: np.ndarray                # phi at each transit
    neck_ratio: np.ndarray              # a/R at each transit
    alpha: float
    fit_r2: float
    monotone: bool
    calibration: dict = field(default_factory=dict)
    histories: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_transits": self.n_transits.tolist(),
            "resistance_n": self.resistance.tolist(),
            "resistance_ratio": self.resistance_ratio.tolist(),
            "sigma_pa": self.sigma.tolist(),
            "porosity": self.porosity.tolist(),
            "neck_ratio": self.neck_ratio.tolist(),
            "alpha": self.alpha,
            "fit_r2": self.fit_r2,
            "monotone": self.monotone,
        }


def calibrate(cfg: ChannelConfig) -> dict:
    """Run the platen DEM at several neck levels; return the R(a) table."""
    base = cfg.platen
    levels = []
    for i in range(cfg.n_calibration):
        frac = i / max(1, cfg.n_calibration - 1)
        t_heal = frac * cfg.interval_s
        a = frenkel_neck_radius(t_heal, cfg.temperature_k,
                                particle_radius=cfg.particle_radius_m)
        bond = sinter_bond(t_heal, cfg.particle_radius_m, cfg.temperature_k)
        pc = PlatenConfig(
            n_particles=base.n_particles, mean_radius=base.mean_radius,
            box_width=base.box_width, box_height=base.box_height,
            platen_speed=base.platen_speed, duration_s=base.duration_s,
            mu=base.mu, kn=base.kn, damping_ratio=base.damping_ratio,
            dt_s=base.dt_s, seed=base.seed,
            k_bond=bond.bond_stiffness, f_bond=bond.bond_strength,
            thickness_m=base.thickness_m)
        out = run_platen(pc)
        levels.append({
            "a_m": a, "t_heal_s": t_heal,
            "k_bond": bond.bond_stiffness, "f_bond": bond.bond_strength,
            "resistance_n": out["resistance_mean_n"],
            "resistance_peak_n": out["resistance_peak_n"],
            "n_broken": out["n_broken"], "n_bonds": out["n_bonds"],
            "t": out["t"].tolist(), "f_platen": out["f_platen"].tolist(),
        })
    return {"levels": levels}


def run_channel_evolution(cfg: ChannelConfig) -> ChannelResult:
    """Run the N-transit channel evolution."""
    calib = calibrate(cfg)
    a_cal = np.array([lv["a_m"] for lv in calib["levels"]])
    r_cal = np.array([lv["resistance_n"] for lv in calib["levels"]])
    order = np.argsort(a_cal)
    a_cal, r_cal = a_cal[order], r_cal[order]
    # enforce monotone non-decreasing R(a) (numerical noise removal)
    r_cal = np.maximum.accumulate(np.maximum(r_cal, 1e-9))

    total = (cfg.n_transits - 1) * cfg.interval_s
    refz = run_refreezing(cfg.refreeze, duration_s=total,
                           record_every_s=max(cfg.interval_s, 60.0))
    t_refz, phi_refz = refz.porosity_series()

    N = np.arange(1, cfg.n_transits + 1)
    res = np.empty(cfg.n_transits)
    sigma = np.empty(cfg.n_transits)
    phi = np.empty(cfg.n_transits)
    neck = np.empty(cfg.n_transits)
    for k in range(cfg.n_transits):
        t_heal = k * cfg.interval_s
        a = frenkel_neck_radius(t_heal, cfg.temperature_k,
                                particle_radius=cfg.particle_radius_m)
        phi_k = float(np.interp(t_heal, t_refz, phi_refz))
        neck_k = a / cfg.particle_radius_m
        r_a = float(np.interp(a, a_cal, r_cal))
        res[k] = r_a * ((1.0 - phi_k) / (1.0 - cfg.phi_0)) ** cfg.resistance_exponent
        sigma[k] = rubble_strength(t_heal, cfg.strength,
                                   porosity=phi_k, neck_ratio=neck_k)
        phi[k] = phi_k
        neck[k] = neck_k

    ratio = res / res[0]
    # log fit: ratio = 1 + alpha log(N)
    logN = np.log(N)
    A = np.vstack([logN, np.ones_like(logN)]).T
    (alpha, intercept), *_ = np.linalg.lstsq(A, ratio - 1.0, rcond=None)
    pred = 1.0 + alpha * logN
    ss_res = float(np.sum((ratio - pred) ** 2))
    ss_tot = float(np.sum((ratio - ratio.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    monotone = bool(np.all(np.diff(ratio) >= -1e-12))

    return ChannelResult(
        n_transits=N,
        resistance=res,
        resistance_ratio=ratio,
        sigma=sigma,
        porosity=phi,
        neck_ratio=neck,
        alpha=float(alpha),
        fit_r2=float(r2),
        monotone=monotone,
        calibration=calib,
        histories={str(lv["a_m"]): {"t": lv["t"], "f_platen": lv["f_platen"]}
                    for lv in calib["levels"]},
    )
