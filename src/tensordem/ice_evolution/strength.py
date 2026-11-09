"""Macroscopic rubble strength from solid fraction and sintered bonds.

Model
-----
The uniaxial strength of the rubble field is

    sigma(t) = sigma_ice(T) * (1 - phi(t))**n * f_bond(t)

* ``sigma_ice(T)`` -- solid-ice strength (Arrhenius in temperature, anchored
  at 1.5 MPa at -10 C, Timco & Frederking 1996 trend);
* ``(1 - phi)**n`` -- solid-fraction effect, ``n = 2.5`` (porous-ice power
  law; at ``phi -> 0`` the rubble recovers the solid-ice strength, which is
  the literature anchor used by criterion c);
* ``f_bond(t)``    -- bonding factor in ``[f_loose, 1]``:

      f_bond(t) = f_loose + (1 - f_loose) * (a(t)/R)**p,   p = 2

  where ``a(t)/R`` is the Frenkel neck ratio.  Loose rubble (no necks)
  carries load only through frictional contact, hence ``f_loose << 1``.

Exact sequential decomposition (criterion c bookkeeping)
--------------------------------------------------------
    sigma(t) = sigma_0 + d_sigma_sinter(t) + d_sigma_refreeze(t)

with

    sigma_0        = sigma_ice (1 - phi_0)**n f_loose
    d_sigma_sinter = sigma_ice (1 - phi_0)**n (f_bond - f_loose)
    d_sigma_refreeze = sigma_ice f_bond [ (1 - phi)**n - (1 - phi_0)**n ]

i.e. sintering acts first (at the initial porosity) and refreezing second
(at the current bonding level); the three terms add exactly to ``sigma``.

Units are SI.
"""
from __future__ import annotations

from dataclasses import dataclass

from .sintering import (
    frenkel_neck_radius,
    ice_strength,
)


@dataclass(frozen=True)
class StrengthConfig:
    """Parameters of the rubble-strength model."""
    temperature_k: float = 263.15
    phi_0: float = 0.30
    particle_radius_m: float = 0.010
    solid_exponent: float = 2.5
    f_loose: float = 0.25
    neck_exponent: float = 2.0
    sigma_ice: float | None = None    # None -> Arrhenius from temperature

    def __post_init__(self) -> None:
        if not 0.0 < self.phi_0 < 1.0:
            raise ValueError("phi_0 must be in (0, 1)")
        if self.particle_radius_m <= 0.0:
            raise ValueError("particle_radius_m must be positive")
        if not 0.0 < self.f_loose <= 1.0:
            raise ValueError("f_loose must be in (0, 1]")


def _sigma_ice(cfg: StrengthConfig) -> float:
    if cfg.sigma_ice is not None:
        return cfg.sigma_ice
    return ice_strength(cfg.temperature_k)


def rubble_strength(t_s: float, cfg: StrengthConfig, *,
                    porosity: float | None = None,
                    neck_ratio: float | None = None) -> float:
    """Rubble strength at time ``t_s`` (s) after the last transit.

    ``porosity`` and ``neck_ratio`` may be supplied by the evolution driver
    (which owns the refreezing / sintering state); when omitted they are
    evaluated here from the analytic laws.
    """
    if t_s < 0.0:
        raise ValueError("t_s must be >= 0")
    if porosity is None:
        porosity = cfg.phi_0
    if neck_ratio is None:
        a = frenkel_neck_radius(t_s, cfg.temperature_k,
                                particle_radius=cfg.particle_radius_m)
        neck_ratio = a / cfg.particle_radius_m
    f_bond = cfg.f_loose + (1.0 - cfg.f_loose) * neck_ratio ** cfg.neck_exponent
    return _sigma_ice(cfg) * (1.0 - porosity) ** cfg.solid_exponent * f_bond


def strength_growth(t_s: float, cfg: StrengthConfig, *,
                    porosity: float | None = None,
                    neck_ratio: float | None = None) -> dict[str, float]:
    """Decompose ``sigma(t)`` into loose + sinter + refreeze parts.

    Returns a dict with ``sigma_0``, ``d_sigma_sinter``, ``d_sigma_refreeze``
    and ``sigma`` (Pa); the three terms add exactly to ``sigma``.
    """
    sigma_ice = _sigma_ice(cfg)
    if porosity is None:
        porosity = cfg.phi_0
    if neck_ratio is None:
        a = frenkel_neck_radius(t_s, cfg.temperature_k,
                                particle_radius=cfg.particle_radius_m)
        neck_ratio = a / cfg.particle_radius_m

    f_loose = cfg.f_loose
    f_bond = cfg.f_loose + (1.0 - cfg.f_loose) * neck_ratio ** cfg.neck_exponent
    n = cfg.solid_exponent

    sigma_0 = sigma_ice * (1.0 - cfg.phi_0) ** n * f_loose
    d_sinter = sigma_ice * (1.0 - cfg.phi_0) ** n * (f_bond - f_loose)
    d_refreeze = sigma_ice * f_bond * ((1.0 - porosity) ** n - (1.0 - cfg.phi_0) ** n)
    return {
        "sigma_0": sigma_0,
        "d_sigma_sinter": d_sinter,
        "d_sigma_refreeze": d_refreeze,
        "sigma": sigma_0 + d_sinter + d_refreeze,
    }
