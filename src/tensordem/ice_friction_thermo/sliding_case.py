"""Ice slab sliding on steel: the time-marched validation case of this package.

The case
--------
A slab of ice slides on a steel plate at a prescribed slip velocity ``v`` under a
normal force ``F_n`` (the geometry of a model ice basin friction test, and the
same loading a DEM ice-structure contact sees).  Every time step:

1. the two-column conductor of :mod:`tensordem.ice_friction_thermo.conduction1d`
   is asked for its affine response ``T(q) = T_a + Z q`` for the coming step;
2. :func:`tensordem.ice_friction_thermo.selfconsistent.solve_contact_state`
   solves the friction-heat / temperature / melting fixed point against that
   response -- so the contact temperature genuinely diffuses (the semi-infinite
   flash formula would ignore the heat still stored in the ice from previous
   steps);
3. the converged conducted flux is committed to the conductor, and the surplus
   flux (the part that cannot be conducted away while the interface is pinned at
   ``T_m(p)``) grows the melt film;
4. the film is drained at the spot-renewal rate ``h/t_c``.

What is checked
---------------
* ``mu`` starts at the dry exponential value and falls only when the interface
  heats or melts -- never the other way round (monotone feedback).
* The friction work is accounted for exactly:
  ``int q dt = sensible + latent(film) + latent(drained)`` to round-off, so no
  energy is created or parked.
* With ``model.allow_melting = False`` (melting branch off, ``T_m`` frozen at
  0 degC) the run reproduces the *constant-friction* baseline: the coefficient
  moves by no more than the dry exponential law allows.  That control is what
  quantifies "what the phase-change closure buys".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .conduction1d import TwoBodyContactConductor
from .friction_law import ContactFrictionModel
from .properties import (
    DEFAULT_FILM_REFERENCE_M,
    DEFAULT_SPOT_RADIUS_M,
    ICE,
    LATENT_HEAT_J_KG,
    STEEL,
    PhaseProperties,
)
from .selfconsistent import ContactConditions, ContactThermoState, solve_contact_state

__all__ = ["SlidingCase", "SlidingResult", "run_sliding_case"]


@dataclass
class SlidingCase:
    """Boundary conditions of the ice-slab-on-steel sliding test."""

    normal_force_n: float = 1.0e5
    nominal_area_m2: float = 0.1
    slip_velocity_m_s: float = 1.0
    bulk_temperature_c: float = -5.0
    model: ContactFrictionModel = field(default_factory=ContactFrictionModel)
    contact_spot_radius_m: float = DEFAULT_SPOT_RADIUS_M
    film_reference_m: float = DEFAULT_FILM_REFERENCE_M
    #: Column depth / resolution.  The flash temperature sets the requirement:
    #: the implicit impedance of the interface cell is within 11 % of the
    #: semi-infinite Blok value once ``dx <= 5e-5 m`` and 46 % high at
    #: ``dx = 2.5e-4 m`` (measured, see ``docs/P272W2_N11_*``), so the defaults
    #: use 5 mm columns at 300 cells (dx = 16.7 um).
    ice_depth_m: float = 5.0e-3
    steel_depth_m: float = 5.0e-3
    ice_cells: int = 300
    steel_cells: int = 300
    steel_temperature_c: float | None = None

    def __post_init__(self) -> None:
        for name in ("normal_force_n", "nominal_area_m2", "slip_velocity_m_s",
                     "contact_spot_radius_m", "film_reference_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def nominal_pressure_pa(self) -> float:
        return self.normal_force_n / self.nominal_area_m2

    def conditions(self) -> ContactConditions:
        return ContactConditions(
            normal_force_n=self.normal_force_n,
            nominal_area_m2=self.nominal_area_m2,
            slip_velocity_m_s=self.slip_velocity_m_s,
            bulk_temperature_c=self.bulk_temperature_c,
            contact_spot_radius_m=self.contact_spot_radius_m,
            steel_temperature_c=self.steel_temperature_c,
        )


@dataclass
class SlidingResult:
    """Recorded history and energy ledger of one sliding run."""

    time_s: np.ndarray
    mu: np.ndarray
    mu_dry: np.ndarray
    interface_temperature_c: np.ndarray
    melting_temperature_c: np.ndarray
    melt_fraction: np.ndarray
    film_thickness_m: np.ndarray
    friction_heat_w: np.ndarray
    heat_flux_w_m2: np.ndarray
    excess_flux_w_m2: np.ndarray
    iterations: np.ndarray
    regime: list[str]
    friction_work_j_m2: float
    sensible_j_m2: float
    latent_melted_j_m2: float
    latent_drained_j_m2: float
    boundary_outflow_j_m2: float
    dt_thermal_s: float
    final_state: ContactThermoState
    case: SlidingCase

    @property
    def energy_residual_j_m2(self) -> float:
        return (self.friction_work_j_m2
                - self.sensible_j_m2
                - self.latent_melted_j_m2
                - self.boundary_outflow_j_m2)

    @property
    def energy_closure_error(self) -> float:
        """``|residual| / friction work`` -- the dimensionless closure error."""
        if self.friction_work_j_m2 == 0.0:
            return 0.0
        return abs(self.energy_residual_j_m2) / self.friction_work_j_m2

    def summary(self) -> dict[str, Any]:
        return {
            "mu_initial": float(self.mu[0]),
            "mu_final": float(self.mu[-1]),
            "mu_min": float(self.mu.min()),
            "mu_dry_initial": float(self.mu_dry[0]),
            "contact_temperature_initial_c": float(self.interface_temperature_c[0]),
            "contact_temperature_final_c": float(self.interface_temperature_c[-1]),
            "melting_temperature_c": float(self.melting_temperature_c[-1]),
            "melt_fraction_final": float(self.melt_fraction[-1]),
            "film_thickness_final_m": float(self.film_thickness_m[-1]),
            "friction_heat_final_w": float(self.friction_heat_w[-1]),
            "iterations_max": int(self.iterations.max()),
            "iterations_mean": float(self.iterations.mean()),
            "regimes": sorted(set(self.regime)),
            "friction_work_j_m2": self.friction_work_j_m2,
            "sensible_j_m2": self.sensible_j_m2,
            "latent_melted_j_m2": self.latent_melted_j_m2,
            "latent_drained_j_m2": self.latent_drained_j_m2,
            "boundary_outflow_j_m2": self.boundary_outflow_j_m2,
            "energy_residual_j_m2": self.energy_residual_j_m2,
            "energy_closure_error": self.energy_closure_error,
            "dt_thermal_s": self.dt_thermal_s,
        }


def run_sliding_case(case: SlidingCase, duration_s: float, dt: float,
                     record_every: int = 1) -> SlidingResult:
    """March the coupled sliding contact for ``duration_s`` with step ``dt``."""
    if not math.isfinite(float(duration_s)) or float(duration_s) <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and positive")
    if record_every < 1:
        raise ValueError("record_every must be at least 1")
    # The resolved thermal step is capped by the *spot residence time*: a real
    # contact patch lives a/v and is then replaced, so stepping the conductor
    # with dt >> a/v would heat a patch for far longer than it exists and
    # inflate the contact temperature (measured: 3.2 K instead of 1.2 K at
    # dt = 1 ms with a = 0.1 mm, v = 1 m/s).  The cap keeps the marching solver
    # consistent with the quasi-static flash estimate of the same contact.
    spot_time = (case.contact_spot_radius_m / case.slip_velocity_m_s
                 if case.slip_velocity_m_s > 0.0 else None)
    dt_thermal = float(dt) if spot_time is None else min(float(dt), spot_time)
    steps = max(1, int(round(float(duration_s) / dt_thermal)))
    dt = dt_thermal

    conductor = TwoBodyContactConductor(
        ice=ICE, steel=STEEL,
        ice_depth_m=case.ice_depth_m, steel_depth_m=case.steel_depth_m,
        ice_cells=case.ice_cells, steel_cells=case.steel_cells,
        initial_temperature_c=case.bulk_temperature_c,
        steel_temperature_c=case.steel_temperature_c,
    )
    conditions = case.conditions()
    model = case.model
    latent_ratio = 1.0 / (ICE.rho_kg_m3 * LATENT_HEAT_J_KG)
    t_c = conditions.contact_time_s

    times: list[float] = []
    mus: list[float] = []
    mu_dry: list[float] = []
    temperatures: list[float] = []
    melt_temperatures: list[float] = []
    melt_fractions: list[float] = []
    films: list[float] = []
    heats: list[float] = []
    fluxes: list[float] = []
    excesses: list[float] = []
    iterations: list[int] = []
    regimes: list[str] = []

    film = 0.0
    friction_work = 0.0
    latent_melted = 0.0
    latent_drained = 0.0
    state: ContactThermoState | None = None

    for step in range(1, steps + 1):
        zero, sensitivity = conductor.trial_response(dt)
        state = solve_contact_state(
            conditions, model,
            film_thickness_m=film,
            film_reference_m=case.film_reference_m,
            response=(float(zero[0]), float(sensitivity[0])),
        )
        total_flux = state.heat_flux_w_m2
        conducted_flux = min(state.conducted_flux_w_m2, total_flux)
        excess_flux = max(0.0, total_flux - conducted_flux)
        # conducted_flux may be negative when the interface is pinned below the
        # bulk temperature (regelation): that heat leaves the interface and the
        # ice column supplies it.
        conductor.commit(dt, conducted_flux)

        # Film renewal, integrated *exactly* in time:
        #   dh/dt = q_excess/(rho L) - h/t_c
        #   h(t+dt) = h e^{-dt/t_c} + q_excess t_c/(rho L) (1 - e^{-dt/t_c})
        # The explicit update is stiff whenever dt >> t_c (the contact spot lives
        # 0.1 ms, the step is 1 ms) and produced a 2-cycle in mu: measured
        # 0.049 -> 0.0038 -> 0.049 before this was switched to the exponential
        # integrator.  The exponential form is unconditionally stable and its
        # dt -> infinity limit is the steady film the quasi-static solver uses.
        melted = excess_flux * latent_ratio * dt          # melted per unit area
        if t_c is not None and t_c > 0.0:
            decay = math.exp(-dt / t_c)
            film_next = film * decay + excess_flux * latent_ratio * t_c * (1.0 - decay)
        else:
            film_next = film + melted
        film_next = max(0.0, film_next)
        drained = max(0.0, melted - (film_next - film))
        film = film_next

        friction_work += total_flux * dt
        latent_melted += melted * ICE.rho_kg_m3 * LATENT_HEAT_J_KG
        latent_drained += drained * ICE.rho_kg_m3 * LATENT_HEAT_J_KG

        if step % record_every == 0 or step == 1 or step == steps:
            times.append(conductor.time_s)
            mus.append(state.mu)
            mu_dry.append(state.mu_dry)
            temperatures.append(conductor.interface_temperature_c)
            melt_temperatures.append(state.melting_temperature_c)
            melt_fractions.append(state.melt_fraction)
            films.append(film)
            heats.append(state.friction_heat_w)
            fluxes.append(total_flux)
            excesses.append(excess_flux)
            iterations.append(state.iterations)
            regimes.append(state.regime)

    assert state is not None
    return SlidingResult(
        time_s=np.asarray(times, dtype=float),
        mu=np.asarray(mus, dtype=float),
        mu_dry=np.asarray(mu_dry, dtype=float),
        interface_temperature_c=np.asarray(temperatures, dtype=float),
        melting_temperature_c=np.asarray(melt_temperatures, dtype=float),
        melt_fraction=np.asarray(melt_fractions, dtype=float),
        film_thickness_m=np.asarray(films, dtype=float),
        friction_heat_w=np.asarray(heats, dtype=float),
        heat_flux_w_m2=np.asarray(fluxes, dtype=float),
        excess_flux_w_m2=np.asarray(excesses, dtype=float),
        iterations=np.asarray(iterations, dtype=int),
        regime=list(regimes),
        friction_work_j_m2=friction_work,
        sensible_j_m2=conductor.sensible_energy_j_m2(),
        latent_melted_j_m2=latent_melted,
        latent_drained_j_m2=latent_drained,
        boundary_outflow_j_m2=conductor.boundary_outflow_j_m2,
        dt_thermal_s=dt_thermal,
        final_state=state,
        case=case,
    )