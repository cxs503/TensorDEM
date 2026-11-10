"""Self-consistent contact state: friction heat, diffusion, melting, ``mu``.

The loop this module closes
---------------------------
A constant-friction DEM contact takes ``mu`` as an input.  Here ``mu`` is an
*output* of a fixed point that couples four pieces of physics:

::

    mu --(Coulomb)--> q = mu p_real v_slip            friction heat      (1)
     |                       |
     |                       v
     |               1D conduction -> T_contact       thermal diffusion  (2)
     |                       |
     |                       v
     |            T_m(p_real) from Clausius-Clapeyron pressure melting   (3)
     |                       |
     |                film h -> melt fraction s       latent heat        (4)
     |                       |
     +--- mu = mu_dry(T_contact) [1 - s (1 - r_wet)] <- exponential law

Each arrow is a separately testable function; the loop is a one-dimensional root
in ``mu`` because the thermal response of an implicit conduction step is
**affine in the flux** (``T = T_a + Z q``).  Two independent solvers are
provided:

* :func:`solve_contact_state` -- the Picard iteration the campaign asks for; it
  reports the iteration count and the relative change of ``mu`` per iteration,
  so "converged in < 10 steps at ``|dmu|/mu < 1 %``" is *measured*.
* :func:`solve_contact_state_bisection` -- a bracketing root find on the same
  residual, used to verify the Picard result independently (agreement to
  ``1e-6`` relative is required by the test suite).

Melt film and its energy bookkeeping
------------------------------------
``q_conducted`` is the flux the interface can conduct away while pinned at
``T_m(p)``; ``q_excess = q - q_conducted`` melts ice.  The film follows the
renewal balance ``dh/dt = q_excess/(rho L) - h/t_c`` with the spot residence
time ``t_c = a_spot / v`` (the contact spot is refreshed every time the ice
slides one spot diameter).  This solver therefore reports the *steady* film
``h = q_excess t_c / (rho L)``; the marching solver in
:mod:`tensordem.ice_friction_thermo.sliding_case` integrates the ODE itself and
passes the carried thickness back in.  ``s = clip(h/h_ref, 0, 1)``.

Regimes
-------
``dry``       ``T_contact < T_m(p)``: no melting.
``pinned``    ``T_contact = T_m(p)``: surplus friction heat becomes latent heat.
``pressure``  ``T_bulk > T_m(p_real)``: the ice is above its *pressure* melting
              point before sliding starts (regelation) -- the regime the
              Clausius-Clapeyron branch exists for, and the reason a cold-looking
              contact can be water-lubricated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .clausius_clapeyron import real_contact_pressure
from .friction_law import ContactFrictionModel
from .properties import (
    DEFAULT_SPOT_RADIUS_M,
    DEFAULT_FILM_REFERENCE_M,
    ICE,
    LATENT_HEAT_J_KG,
    STEEL,
    PhaseProperties,
    blok_partition,
)

__all__ = [
    "ContactConditions",
    "ContactThermoState",
    "solve_contact_state",
    "solve_contact_state_bisection",
    "interface_gain_k_per_mu",
    "DEFAULT_TOLERANCE",
    "MAX_ITERATIONS",
]

#: Convergence criterion of the campaign: relative change of ``mu`` below 1 %.
DEFAULT_TOLERANCE: float = 1.0e-2
#: Iteration cap.  The criterion asks for < 10; the cap is 20 so that a *failure*
#: to converge is reported instead of being silently truncated at 10.
MAX_ITERATIONS: int = 20


def _check(name: str, value: float, *, positive: bool = False,
           nonnegative: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


@dataclass(frozen=True)
class ContactConditions:
    """Loading and kinematics of one sliding ice-structure contact."""

    normal_force_n: float
    nominal_area_m2: float
    slip_velocity_m_s: float
    bulk_temperature_c: float
    contact_spot_radius_m: float = 1.0e-4
    steel_temperature_c: float | None = None

    def __post_init__(self) -> None:
        _check("normal_force_n", self.normal_force_n, nonnegative=True)
        _check("nominal_area_m2", self.nominal_area_m2, positive=True)
        _check("slip_velocity_m_s", self.slip_velocity_m_s, nonnegative=True)
        _check("contact_spot_radius_m", self.contact_spot_radius_m, positive=True)
        if not math.isfinite(self.bulk_temperature_c) or self.bulk_temperature_c >= 0.0:
            raise ValueError("bulk_temperature_c must be finite and below 0 degC")
        if self.steel_temperature_c is not None:
            if not math.isfinite(self.steel_temperature_c) or self.steel_temperature_c >= 0.0:
                raise ValueError("steel_temperature_c must be finite and below 0 degC")

    @property
    def nominal_pressure_pa(self) -> float:
        return self.normal_force_n / self.nominal_area_m2

    @property
    def contact_time_s(self) -> float | None:
        """Residence time of a contact spot ``a / v`` (``None`` at rest)."""
        if self.slip_velocity_m_s <= 0.0:
            return None
        return self.contact_spot_radius_m / self.slip_velocity_m_s


@dataclass
class ContactThermoState:
    """Solved state of one contact: the numbers a DEM step would consume."""

    normal_force_n: float
    slip_velocity_m_s: float
    nominal_pressure_pa: float
    real_pressure_pa: float
    area_fraction: float
    contact_time_s: float | None
    mu: float
    mu_dry: float
    mu_quasi_static: float
    friction_force_n: float
    friction_heat_w: float
    interface_temperature_c: float
    interface_rise_k: float
    melting_temperature_c: float
    melt_fraction: float
    film_thickness_m: float
    heat_flux_w_m2: float
    conducted_flux_w_m2: float
    excess_flux_w_m2: float
    heat_partition_ice: float
    iterations: int
    converged: bool
    tolerance: float
    residual_history: list[float] = field(default_factory=list)
    mu_history: list[float] = field(default_factory=list)
    regime: str = "dry"

    def as_dict(self) -> dict[str, Any]:
        return {
            "normal_force_n": self.normal_force_n,
            "slip_velocity_m_s": self.slip_velocity_m_s,
            "nominal_pressure_mpa": self.nominal_pressure_pa / 1.0e6,
            "real_pressure_mpa": self.real_pressure_pa / 1.0e6,
            "area_fraction": self.area_fraction,
            "contact_time_s": self.contact_time_s,
            "mu": self.mu,
            "mu_dry": self.mu_dry,
            "mu_quasi_static": self.mu_quasi_static,
            "friction_force_n": self.friction_force_n,
            "friction_heat_w": self.friction_heat_w,
            "interface_temperature_c": self.interface_temperature_c,
            "interface_rise_k": self.interface_rise_k,
            "melting_temperature_c": self.melting_temperature_c,
            "melt_fraction": self.melt_fraction,
            "film_thickness_m": self.film_thickness_m,
            "heat_flux_w_m2": self.heat_flux_w_m2,
            "conducted_flux_w_m2": self.conducted_flux_w_m2,
            "excess_flux_w_m2": self.excess_flux_w_m2,
            "heat_partition_ice": self.heat_partition_ice,
            "iterations": self.iterations,
            "converged": self.converged,
            "tolerance": self.tolerance,
            "residual_history": list(self.residual_history),
            "mu_history": list(self.mu_history),
            "regime": self.regime,
        }


def interface_gain_k_per_mu(conditions: ContactConditions, model: ContactFrictionModel,
                            ice: PhaseProperties = ICE,
                            steel: PhaseProperties = STEEL) -> float:
    """``dT_contact/dmu`` for the semi-infinite (flash-temperature) response [K].

    ``q = mu p_real v`` and ``dT = 2 q sqrt(t_c/pi)/(e_i + e_s)``, hence
    ``gain = 2 p_real v sqrt(t_c/pi)/(e_i + e_s)``.  It is the *loop gain* of the
    fixed point: Picard converges in a few steps exactly when
    ``beta * mu * gain < 1``, which the solver reports through its residual
    history.
    """
    p_real, _fraction = real_contact_pressure(conditions.nominal_pressure_pa,
                                              model.hardness_pa)
    v = conditions.slip_velocity_m_s
    if p_real <= 0.0 or v <= 0.0:
        return 0.0
    t_c = conditions.contact_spot_radius_m / v
    effusivity = ice.effusivity_j_m2_k_sqrt_s + steel.effusivity_j_m2_k_sqrt_s
    return 2.0 * p_real * v * math.sqrt(t_c / math.pi) / effusivity


@dataclass(frozen=True)
class _Evaluator:
    """One fixed-point residual evaluation (shared by both solvers)."""

    model: ContactFrictionModel
    p_real: float
    velocity: float
    t_zero: float
    gain: float
    t_melt: float
    film_reference_m: float
    carried_film_m: float
    contact_time_s: float | None

    def __call__(self, mu: float) -> dict[str, float]:
        flux = mu * self.p_real * self.velocity
        t_uncapped = self.t_zero + self.gain * mu
        # Note: no `velocity > 0` guard.  Pressure melting (regelation) exists at
        # rest, so a stationary contact can already be liquid-lubricated; tying
        # the branch to sliding would hide exactly the regime this module is for.
        pinned = bool(self.model.allow_melting and t_uncapped > self.t_melt)
        if pinned:
            t_contact = self.t_melt
            # Flux the interface can conduct away while held at T_m: invert
            # T = t_zero + gain * mu and convert the mu-equivalent back to a flux
            # with the same p_real * v that defines gain.
            mu_capacity = ((self.t_melt - self.t_zero) / self.gain
                           if self.gain > 0.0 else 0.0)
            # mu_capacity may be *negative*: when the bulk ice is already above
            # its pressure melting point (regelation) the interface is held at
            # T_m < T_bulk, so heat must be removed from it -- the cooling is
            # supplied by the ice's own sensible enthalpy, and it melts ice.
            conducted = min(flux, mu_capacity * self.p_real * self.velocity)
            excess = max(0.0, flux - conducted)
            steady_film = 0.0
            if self.contact_time_s is not None:
                steady_film = excess * self.contact_time_s / (ICE.rho_kg_m3 * LATENT_HEAT_J_KG)
            film = max(self.carried_film_m, steady_film)
        else:
            t_contact = t_uncapped
            conducted = flux
            excess = 0.0
            film = self.carried_film_m
        melt = min(1.0, max(0.0, film / self.film_reference_m))
        mu_model = self.model.mu_from_state(t_contact, melt)
        return {
            "mu_model": mu_model,
            "t_contact": t_contact,
            "flux": flux,
            "conducted": conducted,
            "excess": excess,
            "melt": melt,
            "film": film,
            "pinned": pinned,
        }


def _make_evaluator(conditions: ContactConditions, model: ContactFrictionModel,
                    film_thickness_m: float, film_reference_m: float,
                    response: tuple[float, float] | None,
                    ice: PhaseProperties, steel: PhaseProperties) -> tuple[_Evaluator, float, float]:
    p_real, _fraction = real_contact_pressure(conditions.nominal_pressure_pa,
                                              model.hardness_pa)
    v = conditions.slip_velocity_m_s
    if response is None:
        gain = interface_gain_k_per_mu(conditions, model, ice, steel)
        t_zero = conditions.bulk_temperature_c
    else:
        t_zero = float(response[0])
        impedance = float(response[1])
        if impedance < 0.0:
            raise ValueError("thermal response impedance must be non-negative")
        gain = impedance * p_real * v
    evaluator = _Evaluator(
        model=model, p_real=p_real, velocity=v, t_zero=t_zero, gain=gain,
        t_melt=model.melting_temperature_c(p_real),
        film_reference_m=film_reference_m, carried_film_m=film_thickness_m,
        contact_time_s=conditions.contact_time_s,
    )
    return evaluator, p_real, gain


def _assemble(conditions: ContactConditions, model: ContactFrictionModel,
              evaluator: _Evaluator, p_real: float, mu: float,
              iterations: int, converged: bool, tolerance: float,
              residuals: list[float], mus: list[float]) -> ContactThermoState:
    final = evaluator(mu)
    melt = final["melt"]
    pinned = final["pinned"]
    if pinned:
        regime = "pressure" if conditions.bulk_temperature_c > evaluator.t_melt else "pinned"
    else:
        regime = "dry"
    friction_force = mu * conditions.normal_force_n
    return ContactThermoState(
        normal_force_n=conditions.normal_force_n,
        slip_velocity_m_s=conditions.slip_velocity_m_s,
        nominal_pressure_pa=conditions.nominal_pressure_pa,
        real_pressure_pa=p_real,
        area_fraction=(conditions.nominal_pressure_pa / p_real
                       if p_real > 0.0 else 0.0),
        contact_time_s=conditions.contact_time_s,
        mu=mu,
        mu_dry=model.mu_dry(final["t_contact"]),
        mu_quasi_static=model.mu(final["t_contact"], p_real),
        friction_force_n=friction_force,
        friction_heat_w=friction_force * conditions.slip_velocity_m_s,
        interface_temperature_c=final["t_contact"],
        interface_rise_k=final["t_contact"] - conditions.bulk_temperature_c,
        melting_temperature_c=evaluator.t_melt,
        melt_fraction=melt,
        film_thickness_m=final["film"],
        heat_flux_w_m2=final["flux"],
        conducted_flux_w_m2=final["conducted"],
        excess_flux_w_m2=final["excess"],
        heat_partition_ice=blok_partition(ICE, STEEL),
        iterations=iterations,
        converged=converged,
        tolerance=tolerance,
        residual_history=list(residuals),
        mu_history=list(mus),
        regime=regime,
    )


def solve_contact_state(
    conditions: ContactConditions,
    model: ContactFrictionModel,
    *,
    film_thickness_m: float = 0.0,
    film_reference_m: float = DEFAULT_FILM_REFERENCE_M,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = MAX_ITERATIONS,
    response: tuple[float, float] | None = None,
    ice: PhaseProperties = ICE,
    steel: PhaseProperties = STEEL,
) -> ContactThermoState:
    """Picard solution of the coupled contact state.

    Parameters
    ----------
    conditions, model
        Loading/kinematics and the friction law.
    film_thickness_m
        Melt film carried into the step (``0`` for a fresh contact).
    film_reference_m
        Film thickness at which the interface counts as fully separated
        (``s = 1``) -- the model's second phenomenological parameter, scanned in
        the sensitivity table.
    response
        Optional affine thermal response ``(T_zero, impedance)`` from an external
        conductor: ``T_zero`` is the interface temperature the step would reach
        with no friction heat and ``impedance`` its derivative w.r.t. the
        interface flux [K m^2/W] (see
        :meth:`tensordem.ice_friction_thermo.conduction1d.TwoBodyContactConductor.trial_response`).
        When omitted the semi-infinite Blok flash solution is used.
    """
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be at least 1")
    _check("film_thickness_m", film_thickness_m, nonnegative=True)
    _check("film_reference_m", film_reference_m, positive=True)

    evaluator, p_real, _gain = _make_evaluator(
        conditions, model, film_thickness_m, film_reference_m, response, ice, steel)

    mu = max(model.mu_dry(conditions.bulk_temperature_c), model.mu_floor)
    # Seed with the dry coefficient at the (yet unheated) interface.  This is the
    # "constant friction coefficient" the campaign wants to replace, so the loop
    # is seen to *move away* from it rather than to start at its answer.
    residuals: list[float] = []
    mus: list[float] = [mu]
    converged = False
    omega = 1.0
    iterations = 0
    previous = math.inf
    if conditions.slip_velocity_m_s <= 0.0 or p_real <= 0.0:
        # No sliding or no load: nothing to solve, the dry state is exact.
        return _assemble(conditions, model, evaluator, p_real, mu, 0, True,
                         tolerance, residuals, mus)

    for iterations in range(1, int(max_iterations) + 1):
        trial = evaluator(mu)
        mu_new = trial["mu_model"]
        residual = abs(mu_new - mu) / max(mu, 1e-12)
        residuals.append(residual)
        if residual < previous:
            previous = residual
        else:
            # oscillating fixed point: damp it (Anderson-lite).  Reported through
            # the residual history, so a damped run is visible, not hidden.
            omega = max(0.25, 0.5 * omega)
        mu = mu + omega * (mu_new - mu)
        mus.append(mu)
        if residual <= tolerance:
            converged = True
            break
    return _assemble(conditions, model, evaluator, p_real, mu, iterations,
                     converged, tolerance, residuals, mus)


def solve_contact_state_bisection(
    conditions: ContactConditions,
    model: ContactFrictionModel,
    *,
    film_thickness_m: float = 0.0,
    film_reference_m: float = DEFAULT_FILM_REFERENCE_M,
    response: tuple[float, float] | None = None,
    iterations: int = 200,
    ice: PhaseProperties = ICE,
    steel: PhaseProperties = STEEL,
) -> ContactThermoState:
    """Independent root-find solution of the same fixed point.

    The residual ``f(mu) = mu_model(mu) - mu`` is strictly decreasing in ``mu``
    (``mu_model`` is non-increasing because heating can only lower the
    coefficient, and ``-mu`` is decreasing), so bisection on
    ``[0, mu_dry(T_bulk)]`` brackets the unique root.  If it disagrees with
    :func:`solve_contact_state` one of the two is wrong; the test suite requires
    agreement to ``1e-6`` relative.
    """
    evaluator, p_real, _gain = _make_evaluator(
        conditions, model, film_thickness_m, film_reference_m, response, ice, steel)
    if conditions.slip_velocity_m_s <= 0.0 or p_real <= 0.0:
        return _assemble(conditions, model, evaluator, p_real,
                         max(model.mu_dry(conditions.bulk_temperature_c), model.mu_floor),
                         0, True, DEFAULT_TOLERANCE, [], [])

    low = 0.0
    high = max(model.mu_dry(conditions.bulk_temperature_c), model.mu_floor)
    f_low = evaluator(low)["mu_model"] - low
    f_high = evaluator(high)["mu_model"] - high
    if f_low <= 0.0:            # already wetter than any sliding can make it
        return _assemble(conditions, model, evaluator, p_real, low, 0, True,
                         DEFAULT_TOLERANCE, [], [])
    if f_high >= 0.0:           # no heating: the dry value is the root
        return _assemble(conditions, model, evaluator, p_real, high, 0, True,
                         DEFAULT_TOLERANCE, [], [])
    for count in range(1, int(iterations) + 1):
        mid = 0.5 * (low + high)
        f_mid = evaluator(mid)["mu_model"] - mid
        if f_mid > 0.0:
            low = mid
        else:
            high = mid
        if (high - low) <= 1.0e-14 * max(high, 1e-12):
            break
    mu = 0.5 * (low + high)
    return _assemble(conditions, model, evaluator, p_real, mu, count, True,
                     DEFAULT_TOLERANCE, [], [])