"""Two-way DEM <-> FVM coupling of a sliding ice-structure contact.

The closed loop
---------------
::

    DEM contact            friction heat            FVM conduction
    F_n, v_s  ------------>  q'' = mu p v  ------->  T(x, t)
        ^                    (split by Blok)              |
        |                                                  |
        +--- mu(T), k_n(h) <--- melt film h <--- Stefan condition

Each time step solves the interface temperature *implicitly*: the finite-volume
step is affine in the surface flux (the operator does not depend on it before
phase change), so two trial steps give

    ``T_s(q) = T_a + Z q``,  ``Z = T_s(q=1) - T_s(q=0)``,

with ``T_a`` the surface temperature at zero flux.  The fixed point of the
DEM-FVM loop is then the root of

    ``r(T_s) = T_a + Z q(T_s) - T_s``,  ``q(T_s) = gamma p v mu(T_s)``,

solved by Newton with the analytic derivative ``dq/dT = gamma p v dmu/dT``.
That is a *coupled* solve, not a staggered one: the residual, the iteration
count and its convergence are reported for every step, so criterion "coupled
iteration converges within 5 steps at residual < 1 %" is measured rather than
asserted.

Once the interface reaches the melting point the surface temperature is pinned
(strong Dirichlet) and the Stefan condition distributes the interface energy

    ``rho L dh/dt = q_in - q_face``,

where ``q_face = k (T_s - T_1)/dx`` is what conducts into the ice; the film ``h``
then changes the normal stiffness and, through the melt fraction, the friction
coefficient.  Over the whole run the ice-side energy ledger

    ``int q_in dt  =  sensible enthalpy + rho L h``   (+ film drainage, disabled)

closes, which is the check that the phase-change accounting is conservative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .conduction import Conduction1D, thermal_diffusivity
from .friction_heat import blok_partition_coefficient, friction_heat_flux
from .lubrication import MeltLubricationModel, stefan_melt_rate

__all__ = ["IceSteelContact", "CoupledInterface", "hertz_contact_radius"]


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


def hertz_contact_radius(normal_force_n: float, particle_radius_m: float,
                         e1_pa: float, nu1: float, e2_pa: float,
                         nu2: float) -> float:
    """Hertz radius ``a = (3 F R / (4 E*))^(1/3)`` for a sphere on a flat."""
    force = _finite("normal_force_n", normal_force_n, positive=True)
    radius = _finite("particle_radius_m", particle_radius_m, positive=True)
    e1 = _finite("e1_pa", e1_pa, positive=True)
    e2 = _finite("e2_pa", e2_pa, positive=True)
    inv = (1.0 - _finite("nu1", nu1) ** 2) / e1 + (1.0 - _finite("nu2", nu2) ** 2) / e2
    e_star = 1.0 / inv
    return (3.0 * force * radius / (4.0 * e_star)) ** (1.0 / 3.0)


@dataclass(frozen=True)
class IceSteelContact:
    """Physical description of one sliding ice particle on a steel wall.

    Defaults are a 5 cm ice particle pressed with 50 kN against steel and sliding
    at 0.2 m/s -- an ice-class hull contact, not a laboratory pin-on-disc.
    Thermal properties are the standard fresh-ice / mild-steel values.
    """

    normal_force_n: float = 5.0e4
    slip_velocity_m_s: float = 0.2
    particle_radius_m: float = 0.05
    ice_youngs_modulus_pa: float = 9.0e9
    ice_poisson: float = 0.3
    steel_youngs_modulus_pa: float = 210.0e9
    steel_poisson: float = 0.3
    ice_k_w_m_k: float = 2.2
    ice_rho_kg_m3: float = 917.0
    ice_cp_j_kg_k: float = 2100.0
    ice_latent_heat_j_kg: float = 3.34e5
    steel_k_w_m_k: float = 45.0
    steel_rho_kg_m3: float = 7850.0
    steel_cp_j_kg_k: float = 470.0
    t_melt: float = 0.0
    initial_temperature_c: float = -10.0
    mu_cold: float = 0.10
    lubrication: MeltLubricationModel = field(
        default_factory=lambda: MeltLubricationModel(mu_base=0.10))
    layer_thickness_m: float = 0.02
    cells: int = 241
    grading: float = 1.05
    #: Nominal contact area (m^2).  When ``None`` the Hertz area of the sphere on
    #: a flat is used, which at realistic loads gives very small, very highly
    #: stressed patches; a prescribed patch area (a hull-ice contact of a few
    #: cm^2 at a few MPa) is the more representative sliding-contact setting.
    nominal_contact_area_m2: float | None = None

    def __post_init__(self) -> None:
        for name in ("normal_force_n", "particle_radius_m",
                     "ice_youngs_modulus_pa", "steel_youngs_modulus_pa",
                     "ice_k_w_m_k", "ice_rho_kg_m3", "ice_cp_j_kg_k",
                     "ice_latent_heat_j_kg", "steel_k_w_m_k", "steel_rho_kg_m3",
                     "steel_cp_j_kg_k", "layer_thickness_m"):
            object.__setattr__(self, name, _finite(name, getattr(self, name),
                                                   positive=True))
        object.__setattr__(self, "slip_velocity_m_s",
                           _finite("slip_velocity_m_s", self.slip_velocity_m_s,
                                   nonnegative=True))
        object.__setattr__(self, "mu_cold",
                           _finite("mu_cold", self.mu_cold, nonnegative=True))
        object.__setattr__(self, "t_melt", _finite("t_melt", self.t_melt))
        object.__setattr__(self, "initial_temperature_c",
                           _finite("initial_temperature_c", self.initial_temperature_c))
        if self.initial_temperature_c > self.t_melt:
            raise ValueError("the ice must start at or below the melting point")

    # -- derived quantities ---------------------------------------------
    @property
    def alpha_ice(self) -> float:
        return thermal_diffusivity(self.ice_k_w_m_k, self.ice_rho_kg_m3,
                                   self.ice_cp_j_kg_k)

    @property
    def contact_radius_m(self) -> float:
        return hertz_contact_radius(
            self.normal_force_n, self.particle_radius_m,
            self.ice_youngs_modulus_pa, self.ice_poisson,
            self.steel_youngs_modulus_pa, self.steel_poisson)

    @property
    def contact_area_m2(self) -> float:
        if self.nominal_contact_area_m2 is not None:
            return _finite("nominal_contact_area_m2", self.nominal_contact_area_m2,
                           positive=True)
        return math.pi * self.contact_radius_m ** 2

    @property
    def mean_pressure_pa(self) -> float:
        return self.normal_force_n / self.contact_area_m2

    @property
    def effective_contact_radius_m(self) -> float:
        """Radius of the circle with the nominal contact area."""
        return math.sqrt(self.contact_area_m2 / math.pi)

    @property
    def contact_stiffness_n_m(self) -> float:
        """Hertz stiffness ``2 E* a`` of the sphere-flat pair (N/m)."""
        inv = ((1.0 - self.ice_poisson ** 2) / self.ice_youngs_modulus_pa
               + (1.0 - self.steel_poisson ** 2) / self.steel_youngs_modulus_pa)
        return 2.0 * self.contact_radius_m / inv

    def partition(self) -> dict[str, float]:
        return blok_partition_coefficient(
            self.ice_k_w_m_k, self.ice_rho_kg_m3, self.ice_cp_j_kg_k,
            self.steel_k_w_m_k, self.steel_rho_kg_m3, self.steel_cp_j_kg_k)


class CoupledInterface:
    """Marching solver of the coupled contact: DEM force <-> FVM temperature."""

    def __init__(self, case: IceSteelContact, *, max_iterations: int = 40,
                 tolerance: float = 1e-6, allow_melting: bool = True) -> None:
        self.case = case
        self.max_iterations = int(max_iterations)
        self.tolerance = _finite("tolerance", tolerance, positive=True)
        self.allow_melting = bool(allow_melting)
        if self.max_iterations < 2:
            raise ValueError("max_iterations must be at least 2")
        self.conductor = Conduction1D(
            length_m=case.layer_thickness_m, cells=case.cells,
            k_w_m_k=case.ice_k_w_m_k, rho_kg_m3=case.ice_rho_kg_m3,
            cp_j_kg_k=case.ice_cp_j_kg_k, t_initial=case.initial_temperature_c,
            grading=case.grading, theta=1.0)
        self.partition = case.partition()
        self.gamma_ice = self.partition["partition_1"]
        self.film_thickness_m = 0.0
        self.time = 0.0
        self.interface_temperature_c = case.initial_temperature_c
        self.energy_in_ice_j_m2 = 0.0
        self.history: dict[str, list] = {
            "time": [], "interface_temperature_c": [], "mu": [],
            "friction_force_n": [], "heat_flux_w_m2": [],
            "ice_heat_flux_w_m2": [], "film_thickness_m": [],
            "normal_stiffness_n_m": [], "load_bearing_fraction": [],
            "iterations": [], "residual": [], "mode": [],
            "energy_in_ice_j_m2": [], "latent_energy_j_m2": [],
            "sensible_energy_j_m2": [], "coolest_ice_temperature_c": [],
        }

    # -- helpers --------------------------------------------------------
    def mu(self, temperature_c: float) -> float:
        return self.case.lubrication.mu(temperature_c,
                                        mu_base=self.case.mu_cold)

    def dmu_d_temperature(self, temperature_c: float) -> float:
        return self.case.lubrication.dmu_d_temperature(
            temperature_c, mu_base=self.case.mu_cold)

    def _flux(self, temperature_c: float) -> float:
        """Ice-side interface flux (W/m^2) at a trial interface temperature."""
        mu = self.mu(temperature_c)
        q_total = friction_heat_flux(mu, self.case.mean_pressure_pa,
                                     self.case.slip_velocity_m_s)
        return self.gamma_ice * q_total

    def _trial_step(self, dt: float, *, flux: float = 0.0,
                    t_surface: float | None = None) -> float:
        """Surface temperature after one step from the *current* state."""
        scratch = Conduction1D(
            length_m=self.case.layer_thickness_m, cells=self.case.cells,
            k_w_m_k=self.case.ice_k_w_m_k, rho_kg_m3=self.case.ice_rho_kg_m3,
            cp_j_kg_k=self.case.ice_cp_j_kg_k,
            t_initial=self.case.initial_temperature_c,
            grading=self.case.grading, theta=1.0)
        scratch.t = self.conductor.t.copy()
        scratch.step(dt, q_surface=flux, t_surface=t_surface)
        return float(scratch.t[0])

    def normal_force_at_approach(self, approach_m: float) -> float:
        """Displacement-controlled normal force with the melt film in series.

        ``F_n = k_eff(h) * delta``: the same approach carries less force as the
        film thickens, which is the mechanical branch of the closed loop
        (``k_eff = k_contact`` at ``h = 0``).
        """
        delta = _finite("approach_m", approach_m, positive=True)
        k_contact = self.case.contact_stiffness_n_m
        if self.film_thickness_m <= 0.0:
            return k_contact * delta
        k_eff = self.case.lubrication.effective_normal_stiffness(
            k_contact, self.case.contact_area_m2, self.film_thickness_m)
        return k_eff * delta

    # -- one coupled step -----------------------------------------------
    def step(self, dt: float) -> dict[str, float]:
        """Advance one coupled step; returns the converged interface state."""
        dt = _finite("dt", dt, positive=True)
        case = self.case
        # affine surface response of the FVM step in the ice flux
        t_zero = self._trial_step(dt, flux=0.0)
        t_unit = self._trial_step(dt, flux=1.0)
        impedance = t_unit - t_zero
        if impedance <= 0.0:
            raise RuntimeError("FVM surface impedance must be positive")

        # Newton solve of the coupled fixed point (sensible branch)
        t_s = self.interface_temperature_c
        residual = math.inf
        iterations = 0
        converged = False
        for iterations in range(1, self.max_iterations + 1):
            flux = self._flux(t_s)
            residual_temperature = t_zero + impedance * flux - t_s
            # relative residual on the interface *rise* (the physically
            # meaningful scale); a 1 K floor keeps the first step -- where the
            # rise is still zero -- from dividing by nothing.
            scale = max(abs(t_s - case.initial_temperature_c), 1.0)
            residual = abs(residual_temperature) / scale
            if residual <= self.tolerance:
                converged = True
                break
            if (self.allow_melting and t_s >= case.t_melt - 1e-12
                    and residual_temperature > 0.0):
                # The interface is pinned at the melting point: extra heat
                # cannot raise it, it goes into latent heat (Stefan switch).
                # The constraint fixes the interface temperature, so the
                # iteration residual is zero by construction here; the surplus
                # flux is carried by the Stefan condition below and lands in
                # the (exactly closing) energy ledger.
                residual = 0.0
                converged = True
                break
            derivative = impedance * self.gamma_ice * case.mean_pressure_pa * \
                case.slip_velocity_m_s * self.dmu_d_temperature(t_s) - 1.0
            step = residual_temperature / derivative
            t_s = t_s - step
            # never overshoot far below the coldest ice or above melting
            t_s = min(max(t_s, case.initial_temperature_c - 1.0), case.t_melt)

        flux_ice = self._flux(t_s)
        mode = "sensible"
        latent_increment = 0.0
        if self.allow_melting and t_s >= case.t_melt - 1e-9:
            # interface pinned at the melting point: the surplus becomes latent
            self.conductor.step(dt, t_surface=case.t_melt)
            gradient_flux = self.conductor.conductance[0] * (
                case.t_melt - self.conductor.t[1])
            net = flux_ice - gradient_flux
            if net > 0.0:
                mode = "melting"
                latent_increment = stefan_melt_rate(
                    net, case.ice_rho_kg_m3, case.ice_latent_heat_j_kg) * dt
                self.film_thickness_m += latent_increment
                t_s = case.t_melt
            else:
                # refreezing: fall back to the sensible branch
                self.conductor.step(dt, q_surface=flux_ice)
                mode = "sensible"
        else:
            flux_ice = self._flux(t_s)
            self.conductor.step(dt, q_surface=flux_ice)
            t_s = float(self.conductor.t[0])

        self.interface_temperature_c = t_s
        self.time += dt
        self.energy_in_ice_j_m2 += flux_ice * dt
        self._record(dt, flux_ice, iterations, residual, converged, mode)
        return {
            "time": self.time,
            "interface_temperature_c": self.interface_temperature_c,
            "mu": self.mu(self.interface_temperature_c),
            "friction_force_n": (self.mu(self.interface_temperature_c)
                                 * case.normal_force_n),
            "heat_flux_w_m2": self._flux(self.interface_temperature_c) / self.gamma_ice,
            "film_thickness_m": self.film_thickness_m,
            "iterations": iterations,
            "residual": residual,
            "converged": converged,
            "mode": mode,
        }

    def _record(self, dt: float, flux_ice: float, iterations: int,
                residual: float, converged: bool, mode: str) -> None:
        case = self.case
        mu = self.mu(self.interface_temperature_c)
        k_eff = case.lubrication.effective_normal_stiffness(
            case.contact_stiffness_n_m, case.contact_area_m2,
            self.film_thickness_m + 1e-12) if self.film_thickness_m > 0.0 else \
            case.contact_stiffness_n_m
        self.history["time"].append(self.time)
        self.history["interface_temperature_c"].append(self.interface_temperature_c)
        self.history["mu"].append(mu)
        self.history["friction_force_n"].append(mu * case.normal_force_n)
        self.history["heat_flux_w_m2"].append(flux_ice / self.gamma_ice)
        self.history["ice_heat_flux_w_m2"].append(flux_ice)
        self.history["film_thickness_m"].append(self.film_thickness_m)
        self.history["normal_stiffness_n_m"].append(k_eff)
        self.history["load_bearing_fraction"].append(
            max(0.0, 1.0 - case.lubrication.melt_fraction(
                self.interface_temperature_c)))
        self.history["iterations"].append(iterations)
        self.history["residual"].append(residual)
        self.history["mode"].append(mode)
        self.history["energy_in_ice_j_m2"].append(self.energy_in_ice_j_m2)
        self.history["latent_energy_j_m2"].append(
            case.ice_rho_kg_m3 * case.ice_latent_heat_j_kg * self.film_thickness_m)
        self.history["sensible_energy_j_m2"].append(
            self.conductor.enthalpy_above_initial())
        self.history["coolest_ice_temperature_c"].append(
            float(self.conductor.t[-1]))

    def run(self, duration_s: float, dt: float,
            record_every: int = 1) -> dict[str, np.ndarray | float]:
        """March the coupled problem and return the recorded history."""
        duration_s = _finite("duration_s", duration_s, positive=True)
        dt = _finite("dt", dt, positive=True)
        steps = int(round(duration_s / dt))
        if steps < 1:
            raise ValueError("duration_s must be at least one dt")
        for index in range(steps):
            self.step(dt)
            if record_every > 1 and index % record_every:
                # keep the trajectory dense but trim the per-step diagnostics
                for key in ("time", "interface_temperature_c", "mu",
                            "film_thickness_m", "iterations", "residual"):
                    self.history[key].pop()
        result: dict[str, np.ndarray | float] = {
            key: np.asarray(value) for key, value in self.history.items()
        }
        result["max_iterations_used"] = int(max(self.history["iterations"]))
        result["max_residual"] = float(max(self.history["residual"]))
        result["all_converged"] = all(
            it < self.max_iterations for it in self.history["iterations"])
        ice_energy = float(np.asarray(self.history["energy_in_ice_j_m2"])[-1])
        latent = float(np.asarray(self.history["latent_energy_j_m2"])[-1])
        sensible = float(np.asarray(self.history["sensible_energy_j_m2"])[-1])
        result["energy_ledger"] = {
            "ice_energy_in_j_m2": ice_energy,
            "latent_j_m2": latent,
            "sensible_j_m2": sensible,
            "closure_error": (latent + sensible - ice_energy)
            / max(abs(ice_energy), 1e-30),
        }
        result["mechanical_work_j"] = float(np.trapezoid(
            np.asarray(self.history["friction_force_n"])
            * self.case.slip_velocity_m_s,
            np.asarray(self.history["time"])))
        result["contact"] = {
            "radius_m": self.case.contact_radius_m,
            "area_m2": self.case.contact_area_m2,
            "mean_pressure_pa": self.case.mean_pressure_pa,
            "stiffness_n_m": self.case.contact_stiffness_n_m,
            "partition_ice": self.partition["partition_1"],
            "partition_steel": self.partition["partition_2"],
        }
        result["x_m"] = self.conductor.x.copy()
        result["temperature_profile_c"] = self.conductor.t.copy()
        return result