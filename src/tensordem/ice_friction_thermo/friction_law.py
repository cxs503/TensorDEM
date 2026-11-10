"""Temperature- and pressure-dependent ice friction: ``mu(T, p)`` with melting.

Constant ``mu`` in a DEM contact means the coefficient is a *label*, not a
state: it does not know that the interface is being heated by its own friction,
nor that pressure lowers the melting point.  This module gives the coefficient a
state:

    ``mu(T, p) = mu_dry(T) * [1 - s(T, p) (1 - r_wet)]``

with

* ``mu_dry(T) = mu_ref * exp(-beta (T - T_ref))`` -- the **exponential
  temperature law** (Barnes, Tabor & Walker 1971; the friction of ice rises
  roughly exponentially as the interface cools, because creep and the
  quasi-liquid layer both slow down).  ``beta > 0`` and temperatures in degC, so
  colder ice rubs harder.  ``mu_ref`` and ``beta`` are the only two friction
  parameters, both determined from a calibration set (see
  :func:`fit_exponential_temperature_law`), never hand-set per case.
* ``s(T, p)`` -- the melt fraction of :mod:`tensordem.ice_friction_thermo.
  clausius_clapeyron`, evaluated at the **real contact pressure**.  This is the
  pressure-melting feedback: ``s`` depends on ``T - T_m(p)`` and ``T_m(p)``
  falls with pressure.
* ``r_wet`` -- the residual fraction of the dry coefficient that a fully melted
  (water-lubricated) interface still carries.  ``mu`` therefore stays positive
  and continuous; taking ``r_wet = 0`` reproduces the "friction vanishes at the
  melting point" idealisation.

All derivatives needed by the coupled solve are analytic
(:meth:`ContactFrictionModel.dmu_d_temperature`), checked against finite
differences in the test suite.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .clausius_clapeyron import CC_SLOPE_K_PER_PA, melt_fraction, melting_temperature_c

__all__ = [
    "ContactFrictionModel",
    "mu_exponential",
    "fit_exponential_temperature_law",
    "MU_FLOOR",
    "BETA_RANGE_PER_K",
]

#: Physical floor of the coefficient (a numerical guard, not physics: it only
#: binds for a fully melted interface with ``r_wet = 0``).  Reported by
#: :meth:`ContactFrictionModel.diagnostics` whenever it binds.
MU_FLOOR: float = 2.0e-3

#: Admissible range of the exponential rate constant.  Published ice friction
#: temperature sweeps give ``beta ~ 0.02 .. 0.04 1/K`` (a factor 1.2 .. 1.5 per
#: 10 K); a fit that lands outside this window is rejected rather than used.
BETA_RANGE_PER_K: tuple[float, float] = (0.005, 0.10)


def mu_exponential(temperature_c: Any, mu_ref: float, beta_per_k: float,
                   t_ref_c: float = -10.0) -> Any:
    """``mu = mu_ref exp(-beta (T - t_ref))`` (scalar or array, degC)."""
    for name, value in (("mu_ref", mu_ref), ("beta_per_k", beta_per_k),
                        ("t_ref_c", t_ref_c)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if mu_ref <= 0.0:
        raise ValueError("mu_ref must be positive")
    if beta_per_k <= 0.0:
        raise ValueError("beta_per_k must be positive (colder ice rubs harder)")
    exponent = -beta_per_k * (temperature_c - t_ref_c)
    if isinstance(temperature_c, (int, float)):
        return mu_ref * math.exp(exponent)
    import numpy as np
    return mu_ref * np.exp(np.asarray(exponent, dtype=float))


@dataclass(frozen=True)
class ContactFrictionModel:
    """Friction law of one ice-structure contact, with its interface state.

    Parameters
    ----------
    mu_ref, beta_per_k, t_ref_c
        The exponential dry-friction law; ``mu_ref`` is the coefficient at
        ``t_ref_c`` with no melt fraction.  Both come from
        :func:`fit_exponential_temperature_law`.
    wet_ratio
        Residual fraction of ``mu_dry`` at complete melting.  ``0`` -> friction
        vanishes on a full water film, ``1`` -> the melt fraction has no effect on
        friction, which is the *control* used to isolate the phase-change
        feedback in the sliding case.
    onset_width_k
        Width of the melting transition (the model's single phenomenological
        parameter; scanned in the sensitivity table).
    hardness_pa
        Ice hardness used to convert nominal into real contact pressure.
    allow_melting
        When ``False`` the melting branch is switched off and ``T_m`` is frozen
        at 0 degC -- this is the *constant-friction* control the campaign asks
        to be beaten, kept in the same code path so the comparison cannot drift.
    """

    mu_ref: float = 0.070
    beta_per_k: float = 0.0272
    t_ref_c: float = -2.0
    wet_ratio: float = 0.05
    onset_width_k: float = 1.0
    hardness_pa: float = 2.0e7
    allow_melting: bool = True
    mu_floor: float = MU_FLOOR

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu_ref) or self.mu_ref <= 0.0:
            raise ValueError("mu_ref must be finite and positive")
        if not math.isfinite(self.beta_per_k) or self.beta_per_k <= 0.0:
            raise ValueError("beta_per_k must be finite and positive")
        if not (BETA_RANGE_PER_K[0] <= self.beta_per_k <= BETA_RANGE_PER_K[1]):
            raise ValueError(
                f"beta_per_k={self.beta_per_k} outside the admissible window "
                f"{BETA_RANGE_PER_K}")
        if not math.isfinite(self.t_ref_c) or self.t_ref_c >= 0.0:
            raise ValueError("t_ref_c must be finite and below the melting point")
        if not (0.0 <= self.wet_ratio <= 1.0):
            raise ValueError("wet_ratio must be in [0, 1]")
        if not math.isfinite(self.hardness_pa) or self.hardness_pa <= 0.0:
            raise ValueError("hardness_pa must be finite and positive")
        if not math.isfinite(self.onset_width_k) or self.onset_width_k <= 0.0:
            raise ValueError("onset_width_k must be finite and positive")
        if not math.isfinite(self.mu_floor) or self.mu_floor <= 0.0:
            raise ValueError("mu_floor must be finite and positive")

    # -- state functions ---------------------------------------------------
    def melting_temperature_c(self, pressure_pa: float) -> float:
        """Melting point at ``pressure_pa``; frozen at 0 degC if melting is off."""
        if not self.allow_melting:
            return 0.0
        return melting_temperature_c(pressure_pa)

    def melt_fraction(self, temperature_c: float, pressure_pa: float) -> float:
        """Water fraction at ``(T, p_real)``; identically 0 with melting off."""
        if not self.allow_melting:
            return 0.0
        return melt_fraction(temperature_c, pressure_pa, self.onset_width_k)

    def mu_dry(self, temperature_c: float) -> float:
        """Dry (no free water) coefficient at ``temperature_c``."""
        return mu_exponential(temperature_c, self.mu_ref, self.beta_per_k, self.t_ref_c)

    def mu_from_state(self, temperature_c: float, melt_fraction: float) -> float:
        """Coefficient from an *explicit* melt fraction (the coupled solver's path).

        The coupled solver of :mod:`tensordem.ice_friction_thermo.selfconsistent`
        knows the melt fraction from the film energy balance, so it passes it in
        instead of letting the algebraic temperature closure guess it.  Keeping
        one evaluation path for both callers is what makes the two closures
        comparable without code duplication.
        """
        s = min(1.0, max(0.0, float(melt_fraction)))
        value = self.mu_dry(temperature_c) * (1.0 - s * (1.0 - self.wet_ratio))
        return max(value, self.mu_floor)

    def mu(self, temperature_c: float, pressure_pa: float) -> float:
        """Coefficient of friction at contact temperature and real pressure.

        Algebraic (no-history) closure: the melt fraction is read off the
        temperature relative to ``T_m(p)``.  This is the form used against
        literature friction data, where no film history exists.
        """
        return self.mu_from_state(temperature_c,
                                  self.melt_fraction(temperature_c, pressure_pa))

    def dmu_d_temperature(self, temperature_c: float, pressure_pa: float) -> float:
        """Analytic ``d mu / dT`` (dry branch; the melting branch is bounded)."""
        s = self.melt_fraction(temperature_c, pressure_pa)
        return -self.beta_per_k * self.mu_dry(temperature_c) * (
            1.0 - s * (1.0 - self.wet_ratio))

    def dmu_d_pressure(self, temperature_c: float, pressure_pa: float) -> float:
        """Analytic ``d mu / dp`` in the melting branch [1/Pa].

        ``ds/dp = -dT_m/dp / w = -|slope|/w`` when ``0 < s < 1`` and zero
        otherwise, so the pressure weakening is exactly the Clausius-Clapeyron
        slope carried into the friction law -- the coupling this node exists to
        quantify.
        """
        if not self.allow_melting:
            return 0.0
        s = self.melt_fraction(temperature_c, pressure_pa)
        if not 0.0 < s < 1.0:
            return 0.0
        dmu_dry = self.mu_dry(temperature_c)
        ds_dp = CC_SLOPE_K_PER_PA / self.onset_width_k      # negative
        return -dmu_dry * (1.0 - self.wet_ratio) * ds_dp

    def diagnostics(self, temperature_c: float, pressure_pa: float) -> dict[str, Any]:
        s = self.melt_fraction(temperature_c, pressure_pa)
        raw = self.mu(temperature_c, pressure_pa)
        return {
            "mu": raw,
            "mu_dry": self.mu_dry(temperature_c),
            "melt_fraction": s,
            "melting_temperature_c": self.melting_temperature_c(pressure_pa),
            "floor_binds": bool(raw <= self.mu_floor + 1e-15),
            "wet_regime": bool(s > 0.0),
        }

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        payload = asdict(self)
        payload["melting_temperature_c_at_20MPa"] = self.melting_temperature_c(2.0e7)
        return payload


def fit_exponential_temperature_law(
    rows: Iterable[Sequence[float] | Any],
    t_ref_c: float | None = None,
) -> dict[str, float]:
    """Closed-form weighted fit of ``ln mu = ln mu_ref - beta (T - t_ref)``.

    ``rows`` is any iterable of objects (or tuples) exposing ``temperature_c``,
    ``mu`` and (optionally) ``sigma``; the repository's
    :class:`tensordem.ice_steel_friction.FrictionAnchor` satisfies that.  A
    straight line in ``(T, ln mu)`` is fitted by weighted least squares: two
    unknowns, no iteration, so the fit cannot hide a non-convergence, and the
    residual of every anchor is reported.

    Returns ``mu_ref, beta_per_k, t_ref_c, rms_log_residual, max_rel_residual``
    (residuals on ``mu`` are relative, which is the quantity the acceptance
    gate uses).
    """
    points: list[tuple[float, float, float]] = []
    for row in rows:
        if hasattr(row, "temperature_c"):
            temperature = float(row.temperature_c)
            mu = float(row.mu)
            sigma = float(getattr(row, "sigma", 1.0))
        else:
            temperature, mu = float(row[0]), float(row[1])
            sigma = float(row[2]) if len(row) > 2 else 1.0
        if mu <= 0.0 or sigma <= 0.0:
            raise ValueError("mu and sigma must be positive")
        points.append((temperature, mu, sigma))
    if len(points) < 3:
        raise ValueError("at least three anchors are required for a 2-parameter fit")

    temperatures = [p[0] for p in points]
    reference = float(sum(temperatures) / len(temperatures) if t_ref_c is None
                      else t_ref_c)
    # weighted linear regression of y = ln mu on x = -(T - T_ref)
    sw = sy = sx = sxx = sxy = 0.0
    for temperature, mu, sigma in points:
        w = 1.0 / (sigma * sigma)
        x = -(temperature - reference)
        y = math.log(mu)
        sw += w
        sy += w * y
        sx += w * x
        sxx += w * x * x
        sxy += w * x * y
    determinant = sw * sxx - sx * sx
    if abs(determinant) < 1e-30:
        raise ValueError("anchors are degenerate: temperature spread is zero")
    beta = (sw * sxy - sx * sy) / determinant
    intercept = (sy - beta * sx) / sw
    mu_ref = math.exp(intercept)
    if beta <= 0.0:
        raise ValueError("fit produced a non-positive beta (data contradict the law)")

    residuals = []
    for temperature, mu, _sigma in points:
        predicted = mu_exponential(temperature, mu_ref, beta, reference)
        residuals.append((predicted - mu) / mu)
    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    return {
        "mu_ref": mu_ref,
        "beta_per_k": beta,
        "t_ref_c": reference,
        "rms_log_residual": rms,
        "max_rel_residual": max(abs(r) for r in residuals),
        "n_anchors": len(points),
    }