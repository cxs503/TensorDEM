"""Melt-film lubrication: how the phase change feeds back into the contact law.

When the interface of a sliding ice structure reaches the melting point, the
water film that forms changes *both* terms of the Coulomb law, ``F_t = mu F_n``:

* **Friction.**  A continuous water film separates the asperities and the
  coefficient of friction collapses.  The model here is a melt-fraction law

      ``s(T) = clip((T - T_onset) / (T_melt - T_onset), 0, 1)``,
      ``mu(T) = mu_base (1 - s)^p``

  with ``mu(T_melt) = 0`` exactly, ``mu`` decreasing monotonically from
  ``mu_base`` at ``T <= T_onset``.  ``mu_base`` is normally the calibrated
  ice-steel law of :mod:`tensordem.ice_steel_friction` evaluated at the cold
  interface temperature, so the repository keeps one calibrated friction source
  and this module only adds the regime change beyond that law's envelope.

* **Normal stiffness.**  A film of thickness ``h`` in series with the elastic
  contact stiffness carries the same force with a larger approach.  With the
  confined-layer stiffness ``k_film = E_w A / h`` the series combination is
  ``k_eff = (1/k_contact + h/(E_w A))^(-1)``, which reduces to ``k_contact`` as
  ``h -> 0`` and to zero as the film thickens -- the documented asymptotic
  behaviour the tests pin down.

The film itself is grown by the Stefan condition, i.e. by the energy balance of
the interface: whatever part of the frictional flux is not conducted into the
solids melts ice at the rate ``dh/dt = q_net / (rho L)``
(:func:`stefan_melt_rate`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "MeltLubricationModel",
    "stefan_melt_rate",
    "film_normal_stiffness",
    "series_normal_stiffness",
    "load_bearing_area_fraction",
]


def _finite(name: str, value: float, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class MeltLubricationModel:
    """Temperature-dependent friction with a melt-film regime change.

    Parameters
    ----------
    mu_base:
        Friction coefficient of the cold interface (e.g. the calibrated
        ice-steel value at the operating speed, temperature and pressure).
    t_melt:
        Melting temperature (degC, 0 for fresh ice).
    t_onset:
        Temperature at which lubrication starts.  Between ``t_onset`` and
        ``t_melt`` the coefficient falls to zero.
    exponent:
        Shape exponent ``p`` of ``(1 - s)^p``.  ``p = 2`` gives a convex decay
        (little change until the film is well developed), ``p = 1`` linear.
    water_bulk_modulus_pa:
        Stiffness of the confined water film used by
        :meth:`effective_normal_stiffness`.
    """

    mu_base: float = 0.10
    t_melt: float = 0.0
    t_onset: float = -3.0
    exponent: float = 2.0
    water_bulk_modulus_pa: float = 2.2e9
    mu_wet: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mu_base",
                           _finite("mu_base", self.mu_base, nonnegative=True))
        object.__setattr__(self, "t_melt", _finite("t_melt", self.t_melt))
        object.__setattr__(self, "t_onset", _finite("t_onset", self.t_onset))
        object.__setattr__(self, "exponent",
                           _finite("exponent", self.exponent, positive=True))
        object.__setattr__(self, "water_bulk_modulus_pa",
                           _finite("water_bulk_modulus_pa",
                                   self.water_bulk_modulus_pa, positive=True))
        object.__setattr__(self, "mu_wet",
                           _finite("mu_wet", self.mu_wet, nonnegative=True))
        if self.t_onset >= self.t_melt:
            raise ValueError("t_onset must be below t_melt")

    # -- melt fraction / friction ---------------------------------------
    def melt_fraction(self, temperature_c: float) -> float:
        """``s(T)`` in [0, 1]: 0 cold, 1 at the melting point."""
        t = _finite("temperature_c", temperature_c)
        if t <= self.t_onset:
            return 0.0
        if t >= self.t_melt:
            return 1.0
        return (t - self.t_onset) / (self.t_melt - self.t_onset)

    def mu(self, temperature_c: float, mu_base: float | None = None) -> float:
        """``mu(T) = mu_base (1 - s)^p``; exactly zero at ``T >= t_melt``."""
        base = self.mu_base if mu_base is None else _finite(
            "mu_base", mu_base, nonnegative=True)
        if self.mu_wet > base:
            raise ValueError("mu_wet must not exceed mu_base")
        s = self.melt_fraction(temperature_c)
        return self.mu_wet + (base - self.mu_wet) * (1.0 - s) ** self.exponent

    def dmu_d_temperature(self, temperature_c: float,
                          mu_base: float | None = None) -> float:
        """``d mu / dT`` (1/K); zero outside the lubrication band."""
        base = self.mu_base if mu_base is None else _finite(
            "mu_base", mu_base, nonnegative=True)
        t = _finite("temperature_c", temperature_c)
        if not self.t_onset < t < self.t_melt:
            return 0.0
        if self.mu_wet > base:
            raise ValueError("mu_wet must not exceed mu_base")
        s = (t - self.t_onset) / (self.t_melt - self.t_onset)
        span = self.t_melt - self.t_onset
        return (-self.exponent * (base - self.mu_wet)
                * (1.0 - s) ** (self.exponent - 1.0) / span)

    # -- contact mechanics ----------------------------------------------
    def film_stiffness(self, area_m2: float, film_thickness_m: float) -> float:
        """Confined-film stiffness ``E_w A / h`` (N/m)."""
        area = _finite("area_m2", area_m2, positive=True)
        h = _finite("film_thickness_m", film_thickness_m, positive=True)
        return self.water_bulk_modulus_pa * area / h

    def effective_normal_stiffness(self, k_contact_n_m: float, area_m2: float,
                                   film_thickness_m: float) -> float:
        """Series stiffness of the elastic contact and the melt film (N/m)."""
        k_contact = _finite("k_contact_n_m", k_contact_n_m, positive=True)
        area = _finite("area_m2", area_m2, positive=True)
        h = _finite("film_thickness_m", film_thickness_m, nonnegative=True)
        if h == 0.0:
            return k_contact
        return 1.0 / (1.0 / k_contact + h / (self.water_bulk_modulus_pa * area))


def film_normal_stiffness(area_m2: float, film_thickness_m: float,
                          modulus_pa: float = 2.2e9) -> float:
    """Confined-film stiffness ``E A / h`` (N/m) without the model object."""
    area = _finite("area_m2", area_m2, positive=True)
    h = _finite("film_thickness_m", film_thickness_m, positive=True)
    modulus = _finite("modulus_pa", modulus_pa, positive=True)
    return modulus * area / h


def series_normal_stiffness(k1_n_m: float, k2_n_m: float) -> float:
    """Two springs in series (N/m); ``k2 -> inf`` returns ``k1``."""
    k1 = _finite("k1_n_m", k1_n_m, positive=True)
    k2 = _finite("k2_n_m", k2_n_m, positive=True)
    return 1.0 / (1.0 / k1 + 1.0 / k2)


def stefan_melt_rate(net_flux_w_m2: float, rho_kg_m3: float,
                     latent_heat_j_kg: float) -> float:
    """Stefan condition ``dh/dt = q_net / (rho L)`` in m/s.

    ``q_net`` is the part of the interface flux that is *not* conducted into the
    solid; a negative value refreezes (the model allows it, the coupled cases
    keep it at zero because the interface never falls below the melting point
    while the frictional source acts).
    """
    flux = _finite("net_flux_w_m2", net_flux_w_m2)
    rho = _finite("rho_kg_m3", rho_kg_m3, positive=True)
    latent = _finite("latent_heat_j_kg", latent_heat_j_kg, positive=True)
    return flux / (rho * latent)


def load_bearing_area_fraction(melt_fraction: float) -> float:
    """Solid-solid load-bearing fraction ``1 - s`` of the nominal contact.

    Reported (not used to scale the friction power, which is spread over the
    nominal contact area to stay consistent with the Jaeger band reference).
    """
    s = _finite("melt_fraction", melt_fraction)
    if not 0.0 <= s <= 1.0:
        raise ValueError("melt_fraction must lie in [0, 1]")
    return 1.0 - s