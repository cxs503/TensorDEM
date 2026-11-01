"""Thermodynamic melting of ice fragments (Ranz-Marshall + Stefan condition).

Physics
-------
A spherical ice fragment of diameter ``D`` suspended in the propeller race
exchanges heat with the surrounding water at ``T_w`` through the
Ranz-Marshall correlation ::

    Nu = 2 + 0.6 Re^(1/2) Pr^(1/3),    h_c = Nu k_w / D

The Stefan (isothermal) interface condition turns the convective heat flux into
a volume loss ::

    rho_i L_f dV/dt = -h_c A (T_w - T_m)   =>   dD/dt = -2 h_c (T_w - T_m)/(rho_i L_f)

For constant relative velocity the resulting ODE ::

    dD/dt = -A/D - B/sqrt(D),   A = 4 k_w dT/(rho_i L_f),
                                B = 1.2 k_w (u/nu)^(1/2) Pr^(1/3) dT/(rho_i L_f)

is separable and has the closed-form (implicit) solution implemented in
``analytic_ransom_marshall_time``, which is the reference used to grade the
time-integrated melting model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable


def _finite(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
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
class ThermalProperties:
    """Water / sea-ice thermophysical properties (SI, temperatures in deg C)."""

    water_temperature_c: float = 0.5
    melt_temperature_c: float = 0.0
    water_conductivity_w_m_k: float = 0.55
    water_kinematic_viscosity_m2_s: float = 1.79e-6
    water_prandtl: float = 12.5
    latent_heat_j_kg: float = 3.34e5
    ice_density_kg_m3: float = 917.0
    ice_initial_temperature_c: float = -2.0

    def __post_init__(self) -> None:
        _finite("water_temperature_c", self.water_temperature_c)
        _finite("melt_temperature_c", self.melt_temperature_c)
        _finite("ice_initial_temperature_c", self.ice_initial_temperature_c)
        _finite("water_conductivity_w_m_k", self.water_conductivity_w_m_k, positive=True)
        _finite("water_kinematic_viscosity_m2_s", self.water_kinematic_viscosity_m2_s, positive=True)
        _finite("water_prandtl", self.water_prandtl, positive=True)
        _finite("latent_heat_j_kg", self.latent_heat_j_kg, positive=True)
        _finite("ice_density_kg_m3", self.ice_density_kg_m3, positive=True)
        if self.water_temperature_c < self.melt_temperature_c:
            raise ValueError("water must not be colder than the melting temperature")

    @property
    def driving_temperature_k(self) -> float:
        return self.water_temperature_c - self.melt_temperature_c

    @property
    def sensible_heat_j_kg(self) -> float:
        """Heat needed to warm the fragment to the melting point (c_p = 2.1 kJ/kg/K)."""
        delta = max(self.melt_temperature_c - self.ice_initial_temperature_c, 0.0)
        return 2100.0 * delta


def nusselt_ransom_marshall(reynolds: float, prandtl: float) -> float:
    """Ranz-Marshall correlation for forced convection around a sphere."""
    reynolds = _finite("reynolds", reynolds, nonnegative=True)
    prandtl = _finite("prandtl", prandtl, positive=True)
    return 2.0 + 0.6 * math.sqrt(reynolds) * prandtl ** (1.0 / 3.0)


def heat_transfer_coefficient(
    diameter_m: float, relative_velocity_m_s: float, props: ThermalProperties
) -> dict[str, float]:
    """Convective coefficient and the dimensionless groups feeding it."""
    diameter = _finite("diameter_m", diameter_m, positive=True)
    velocity = _finite("relative_velocity_m_s", relative_velocity_m_s, nonnegative=True)
    reynolds = velocity * diameter / props.water_kinematic_viscosity_m2_s
    nusselt = nusselt_ransom_marshall(reynolds, props.water_prandtl)
    return {
        "reynolds": reynolds,
        "nusselt": nusselt,
        "h_w_m2_k": nusselt * props.water_conductivity_w_m_k / diameter,
    }


def diameter_rate(
    diameter_m: float,
    relative_velocity_m_s: float,
    props: ThermalProperties,
    *,
    include_sensible: bool = True,
) -> float:
    """``dD/dt`` in m/s (negative while melting) for a single sphere."""
    diameter = _finite("diameter_m", diameter_m, positive=True)
    groups = heat_transfer_coefficient(diameter, relative_velocity_m_s, props)
    heat_per_volume = (
        props.latent_heat_j_kg + (props.sensible_heat_j_kg if include_sensible else 0.0)
    )
    return -2.0 * groups["h_w_m2_k"] * props.driving_temperature_k / (
        props.ice_density_kg_m3 * heat_per_volume
    )


def _analytic_case_constants(
    props: ThermalProperties, relative_velocity_m_s: float, *, include_sensible: bool
) -> tuple[float, float]:
    d_t = props.driving_temperature_k
    if d_t <= 0.0:
        raise ValueError("no melting without a positive water-melt temperature difference")
    heat_per_volume = props.latent_heat_j_kg + (
        props.sensible_heat_j_kg if include_sensible else 0.0
    )
    denom = props.ice_density_kg_m3 * heat_per_volume
    a = 4.0 * props.water_conductivity_w_m_k * d_t / denom
    b = (
        1.2
        * props.water_conductivity_w_m_k
        * props.water_prandtl ** (1.0 / 3.0)
        * d_t
        / denom
        * math.sqrt(relative_velocity_m_s / props.water_kinematic_viscosity_m2_s)
    )
    return a, b


def analytic_conduction_melt_time(
    diameter_m: float,
    props: ThermalProperties,
    *,
    include_sensible: bool = True,
) -> float:
    """Closed-form melting time in the conduction limit (Nu = 2, no convection)."""
    diameter = _finite("diameter_m", diameter_m, positive=True)
    a, _ = _analytic_case_constants(props, 0.0, include_sensible=include_sensible)
    # dD/dt = -A/D  =>  D^2 = D0^2 - 2 A t
    return diameter**2 / (2.0 * a)


def analytic_diameter_conduction(
    initial_diameter_m: float, time_s: float, props: ThermalProperties, *, include_sensible: bool = True
) -> float:
    """Diameter history of the conduction limit (``Nu = 2``)."""
    d0 = _finite("initial_diameter_m", initial_diameter_m, positive=True)
    t = _finite("time_s", time_s, nonnegative=True)
    a, _ = _analytic_case_constants(props, 0.0, include_sensible=include_sensible)
    return math.sqrt(max(d0 * d0 - 2.0 * a * t, 0.0))


def analytic_ransom_marshall_time(
    initial_diameter_m: float,
    diameter_m: float,
    relative_velocity_m_s: float,
    props: ThermalProperties,
    *,
    include_sensible: bool = True,
) -> float:
    """Closed-form melting time for constant relative velocity.

    With ``dD/dt = -A/D - B/sqrt(D)`` and ``s = sqrt(D)``::

        t(D0) - t(D) = integral_D^{D0} D dD/(A + B sqrt(D))
                     = [ 2 s^3/(3 B) - A s^2/B^2 + 2 A^2 s/B^3
                         - 2 A^3 ln(A + B s)/B^4 ]_{s=sqrt(D)}^{sqrt(D0)}

    (the ``B -> 0`` limit degenerates gracefully to ``(D0^2 - D^2)/(2A)``).
    """
    d0 = _finite("initial_diameter_m", initial_diameter_m, positive=True)
    d = _finite("diameter_m", diameter_m, positive=True)
    velocity = _finite("relative_velocity_m_s", relative_velocity_m_s, nonnegative=True)
    if d > d0:
        raise ValueError("final diameter must not exceed the initial diameter")
    a, b = _analytic_case_constants(props, velocity, include_sensible=include_sensible)
    if b <= 1e-300:
        return (d0 * d0 - d * d) / (2.0 * a)

    def primitive(diameter: float) -> float:
        s = math.sqrt(diameter)
        return (
            2.0 * s**3 / (3.0 * b)
            - a * s**2 / b**2
            + 2.0 * a**2 * s / b**3
            - 2.0 * a**3 * math.log(a + b * s) / b**4
        )

    return primitive(d0) - primitive(d)


def analytic_diameter_ransom_marshall(
    initial_diameter_m: float,
    time_s: float,
    relative_velocity_m_s: float,
    props: ThermalProperties,
    *,
    include_sensible: bool = True,
    iterations: int = 200,
) -> float:
    """Invert the closed-form melting time to get ``D(t)`` (bisection on t(D))."""
    d0 = _finite("initial_diameter_m", initial_diameter_m, positive=True)
    t = _finite("time_s", time_s, nonnegative=True)
    velocity = _finite("relative_velocity_m_s", relative_velocity_m_s, nonnegative=True)
    melt_time = analytic_ransom_marshall_time(
        d0, d0 * 1e-9 + 1e-12, velocity, props, include_sensible=include_sensible
    )
    if t >= melt_time:
        return 0.0
    low, high = 1e-12, d0
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        elapsed = analytic_ransom_marshall_time(
            d0, mid, velocity, props, include_sensible=include_sensible
        )
        if elapsed > t:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def integrate_melting(
    initial_diameter_m: float,
    props: ThermalProperties,
    *,
    relative_velocity_m_s: float
    | Callable[[float, float], float]
    | Callable[[float], float]
    | None = None,
    duration_s: float = 60.0,
    dt_s: float = 0.05,
    include_sensible: bool = True,
) -> list[dict[str, float]]:
    """Explicit (RK2 midpoint) integration of the melting ODE.

    ``relative_velocity_m_s`` may be a constant, ``f(diameter, time)`` or
    ``f(diameter)``. The returned history has one row per step with
    ``time_s``, ``diameter_m``, ``mass_kg``, ``melted_mass_kg`` and
    ``melt_fraction``.
    """
    d0 = _finite("initial_diameter_m", initial_diameter_m, positive=True)
    duration = _finite("duration_s", duration_s, nonnegative=True)
    dt = _finite("dt_s", dt_s, positive=True)
    rho_i = props.ice_density_kg_m3

    def velocity(diameter: float, time_s: float) -> float:
        if relative_velocity_m_s is None:
            return 0.0
        if callable(relative_velocity_m_s):
            try:
                return float(relative_velocity_m_s(diameter, time_s))  # type: ignore[misc]
            except TypeError:
                return float(relative_velocity_m_s(diameter))  # type: ignore[misc]
        return float(relative_velocity_m_s)

    def sphere_mass(diameter: float) -> float:
        return rho_i * math.pi / 6.0 * diameter**3

    initial_mass = sphere_mass(d0)
    rows: list[dict[str, float]] = []
    diameter = d0
    time_s = 0.0
    steps = int(round(duration / dt))
    for step in range(steps + 1):
        mass = sphere_mass(diameter)
        rows.append(
            {
                "time_s": time_s,
                "diameter_m": diameter,
                "mass_kg": mass,
                "melted_mass_kg": initial_mass - mass,
                "melt_fraction": 0.0 if initial_mass <= 0.0 else 1.0 - mass / initial_mass,
            }
        )
        if step == steps or diameter <= 0.0:
            continue
        k1 = diameter_rate(diameter, velocity(diameter, time_s), props,
                           include_sensible=include_sensible)
        mid_diameter = max(diameter + 0.5 * dt * k1, 0.0)
        if mid_diameter <= 0.0:
            diameter = 0.0
        else:
            k2 = diameter_rate(mid_diameter, velocity(mid_diameter, time_s + 0.5 * dt), props,
                               include_sensible=include_sensible)
            diameter = max(diameter + dt * k2, 0.0)
        time_s = (step + 1) * dt
    return rows


def melting_rate_kg_s(
    diameter_m: float, relative_velocity_m_s: float, props: ThermalProperties,
    *, include_sensible: bool = True,
) -> float:
    """Mass melting rate (positive) of a single sphere in kg/s."""
    diameter = _finite("diameter_m", diameter_m, positive=True)
    rate = diameter_rate(diameter, relative_velocity_m_s, props, include_sensible=include_sensible)
    surface = math.pi * diameter**2
    return -props.ice_density_kg_m3 * 0.5 * surface * rate


def aggregate_melt(
    fragments: Iterable[dict[str, float]],
    props: ThermalProperties,
    *,
    relative_velocity_m_s: float,
    dt_s: float,
    include_sensible: bool = True,
) -> dict[str, float]:
    """Advance a fragment population by one step and report the mass ledger.

    ``fragments`` entries need ``mass_kg`` and ``diameter_m``. Mass that falls
    below the numerical floor is booked as melted, so the ledger closes exactly
    up to floating-point round-off.
    """
    dt = _finite("dt_s", dt_s, positive=True)
    rho_i = props.ice_density_kg_m3
    melted = 0.0
    remaining = 0.0
    updated: list[dict[str, float]] = []
    for fragment in fragments:
        mass = float(fragment["mass_kg"])
        diameter = float(fragment["diameter_m"])
        if mass <= 0.0 or diameter <= 0.0:
            melted += max(mass, 0.0)
            updated.append({**fragment, "mass_kg": 0.0, "diameter_m": 0.0, "melted_kg": max(mass, 0.0)})
            continue
        change = diameter_rate(diameter, relative_velocity_m_s, props,
                               include_sensible=include_sensible) * dt
        new_diameter = max(diameter + change, 0.0)
        new_mass = rho_i * math.pi / 6.0 * new_diameter**3
        new_mass = min(new_mass, mass)
        melted += mass - new_mass
        remaining += new_mass
        updated.append(
            {
                **fragment,
                "mass_kg": new_mass,
                "diameter_m": new_diameter,
                "melted_kg": mass - new_mass,
            }
        )
    return {
        "fragments": updated,
        "melted_mass_kg": melted,
        "remaining_mass_kg": remaining,
    }