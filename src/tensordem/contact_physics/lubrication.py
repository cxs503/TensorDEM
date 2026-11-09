"""Lubrication <-> contact two-way coupling: stabilised adapter over N08.

Campaign node: ``P121W5-N08`` (lubrication-contact two-way coupling).  As with
the bond adapter, the numerics stay where they were verified:

* :class:`tensordem.interface_thermal.lubrication.MeltLubricationModel` -- the
  melt-fraction friction law ``mu(T) = mu_base (1 - s)^p`` and the confined-film
  series stiffness;
* :mod:`tensordem.interface_thermal.friction_heat` -- the frictional flux
  ``q = mu p v`` and the Blok effusivity partition;
* :class:`tensordem.interface_thermal.coupling.CoupledInterface` -- the coupled
  marching solver (Newton solve of the DEM force <-> FVM temperature fixed
  point, with the Stefan latent-heat switch).

This wrapper exists so that the integrated contact-physics module can own one
object with one ``step(dt)`` signature and a documented return record, while the
DEM bond module can read the interface temperature from it.  Everything the
wrapper returns is either delegated verbatim or a *renaming* of a delegated
quantity -- the integration test asserts equality with the raw solver at
``1e-10`` relative tolerance.
"""
from __future__ import annotations

import math

from ..interface_thermal.coupling import CoupledInterface, IceSteelContact
from ..interface_thermal.friction_heat import friction_heat_flux, friction_power
from ..interface_thermal.lubrication import (
    MeltLubricationModel,
    load_bearing_area_fraction,
    stefan_melt_rate,
)

__all__ = ["LubricationContactCoupling"]


class LubricationContactCoupling:
    """Two-way coupled melt-lubricated sliding contact (DEM force <-> FVM T).

    Parameters
    ----------
    contact:
        Physical description of the sliding pair
        (:class:`tensordem.interface_thermal.coupling.IceSteelContact`).  The
        default is the module's ice-class hull contact (50 kN on a 5 cm ice
        particle sliding at 0.2 m/s).
    max_iterations, tolerance, allow_melting:
        Forwarded unchanged to :class:`CoupledInterface`.

    Attributes
    ----------
    case:
        The :class:`IceSteelContact` in use (read-only by convention).
    history:
        Per-step records, same dict-of-lists layout as the underlying solver.

    Notes
    -----
    ``step(dt)`` is the only state-changing method; :meth:`reset` rebuilds the
    underlying solver so repeated runs start from exactly the same state (that
    is what makes the standalone-vs-integrated comparison bit-reproducible).
    """

    def __init__(self, contact: IceSteelContact | None = None, *,
                 max_iterations: int = 40, tolerance: float = 1e-6,
                 allow_melting: bool = True) -> None:
        if contact is not None and not isinstance(contact, IceSteelContact):
            raise TypeError("contact must be an IceSteelContact")
        self.case = contact if contact is not None else IceSteelContact()
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.allow_melting = bool(allow_melting)
        self._coupled = CoupledInterface(
            self.case, max_iterations=self.max_iterations,
            tolerance=self.tolerance, allow_melting=self.allow_melting)

    # -- construction / state ------------------------------------------------
    def reset(self) -> "LubricationContactCoupling":
        """Return to the initial (cold, film-free) state; returns ``self``."""
        self._coupled = CoupledInterface(
            self.case, max_iterations=self.max_iterations,
            tolerance=self.tolerance, allow_melting=self.allow_melting)
        return self

    @property
    def model(self) -> MeltLubricationModel:
        """The melt-lubrication law carried by the contact case."""
        return self.case.lubrication

    @property
    def time(self) -> float:
        """Elapsed physical time, s."""
        return float(self._coupled.time)

    @property
    def interface_temperature_c(self) -> float:
        """Current interface temperature, degC."""
        return float(self._coupled.interface_temperature_c)

    @property
    def film_thickness_m(self) -> float:
        """Current melt-film thickness ``h``, m."""
        return float(self._coupled.film_thickness_m)

    @property
    def history(self) -> dict:
        """Per-step history of the underlying coupled solver."""
        return self._coupled.history

    # -- delegated scalar laws ----------------------------------------------
    def melt_fraction(self, temperature_c: float | None = None) -> float:
        """``s(T)`` in [0, 1]; defaults to the current interface temperature."""
        t = self.interface_temperature_c if temperature_c is None else float(temperature_c)
        return self.model.melt_fraction(t)

    def mu(self, temperature_c: float | None = None) -> float:
        """Friction coefficient at ``T`` (cold value from the contact case)."""
        t = self.interface_temperature_c if temperature_c is None else float(temperature_c)
        return self.model.mu(t, mu_base=self.case.mu_cold)

    def load_bearing_fraction(self, temperature_c: float | None = None) -> float:
        """Solid-solid load-bearing fraction ``1 - s`` of the nominal contact."""
        return load_bearing_area_fraction(self.melt_fraction(temperature_c))

    def friction_flux(self, temperature_c: float | None = None) -> float:
        """Frictional heat flux ``q = mu(T) p v`` in W/m^2."""
        mu = self.mu(temperature_c)
        return friction_heat_flux(mu, self.case.mean_pressure_pa,
                                  self.case.slip_velocity_m_s)

    def friction_power_w(self, temperature_c: float | None = None) -> float:
        """Frictional power ``mu F_n v`` in W."""
        mu = self.mu(temperature_c)
        return friction_power(mu, self.case.normal_force_n,
                              self.case.slip_velocity_m_s)

    def melt_rate(self, net_flux_w_m2: float) -> float:
        """Stefan melt rate ``dh/dt = q_net / (rho L)`` in m/s."""
        return stefan_melt_rate(net_flux_w_m2, self.case.ice_rho_kg_m3,
                                self.case.ice_latent_heat_j_kg)

    def normal_stiffness(self) -> float:
        """Series stiffness of the elastic contact and the current melt film, N/m."""
        if self.film_thickness_m <= 0.0:
            return float(self.case.contact_stiffness_n_m)
        return float(self.model.effective_normal_stiffness(
            self.case.contact_stiffness_n_m, self.case.contact_area_m2,
            self.film_thickness_m))

    def normal_force_at_approach(self, approach_m: float) -> float:
        """``F_n = k_eff(h) * delta`` (the mechanical branch of the loop), N."""
        return float(self._coupled.normal_force_at_approach(float(approach_m)))

    # -- the coupled step ----------------------------------------------------
    def step(self, dt: float) -> dict:
        """Advance one coupled step and return the augmented state record.

        The returned dict is the solver's own record plus three derived keys
        (``film_normal_stiffness_n_m``, ``load_bearing_fraction``,
        ``friction_flux_w_m2``) so a caller needs a single object to describe
        the interface.
        """
        record = dict(self._coupled.step(float(dt)))
        # the solver already evaluated mu(T) at the converged interface
        # temperature; reuse it instead of re-evaluating the law (keeps the
        # integration layer's per-step cost to a dict update)
        mu = float(record["mu"])
        melt = self.model.melt_fraction(record["interface_temperature_c"])
        film = float(record["film_thickness_m"])
        record["melt_fraction"] = melt
        record["load_bearing_fraction"] = load_bearing_area_fraction(melt)
        record["friction_flux_w_m2"] = friction_heat_flux(
            mu, self.case.mean_pressure_pa, self.case.slip_velocity_m_s)
        record["film_normal_stiffness_n_m"] = (
            float(self.case.contact_stiffness_n_m) if film <= 0.0 else
            float(self.model.effective_normal_stiffness(
                self.case.contact_stiffness_n_m, self.case.contact_area_m2, film)))
        return record

    def run(self, duration_s: float, dt: float, record_every: int = 1) -> dict:
        """March ``duration_s`` and return the solver's summary dict."""
        return self._coupled.run(float(duration_s), float(dt),
                                 record_every=record_every)