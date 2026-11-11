"""Mill power anchors: DEM ledger, torque factor and the empirical correlation.

Three independent statements about the same physical quantity are exposed.

1. **Torque factor** (dimensionless, DEM-side)::

       TF = tau / (M_b g R)

   with ``tau`` the shell torque [N m], ``M_b`` the charge mass [kg], ``R`` the
   mill inner radius.  TF is the standard way of reporting tumbling-mill power
   because simulation and pilot data collapse onto it.  In a steady state the
   angular-momentum balance of the charge reads ``<tau_wall> = M_b g <x_cm>``
   (gravity torque about the mill axis, ``x_cm`` the horizontal offset of the
   charge centre of mass), i.e. ``TF = <x_cm>/R``: the DEM torque and the DEM
   charge geometry are two independent estimates of the same number, which is
   what :mod:`tensordem.comminution.mill_dem` cross-checks.

2. **Specific power** [kWh/t of *charge* or of *ore*]: power divided by mass
   flow, the quantity that Bond's law is compared with.

3. **Empirical correlation** of Hogg & Fuerstenau (1972), the classical
   regression on experimental tumbling-mill power data::

       P [kW] = 7.33 * phi_c * J * (1 - 0.937 J) * rho_c[kg/m^3 /1000] * L * D**2.3

   with ``phi_c = N/N_c`` the fraction of critical speed, ``J`` the volumetric
   ball filling, ``rho_c`` the *bulk* charge density and ``D``, ``L`` the mill
   inner diameter and length in metres.

   Validity caveat that the acceptance report must state: it is an empirical fit
   whose data set is dominated by industrial-scale mills (D of a few metres),
   and it is known to *under*-predict large-mill power by 20-40% relative to
   modern models (Morrell); using it at laboratory scale (D ~ 0.3 m) is an
   extrapolation of the fit, so the comparison carries a band, not a point
   truth.  The ``coefficient``/``exponent`` are parameters of this module so the
   sensitivity to the fit constants can be reported instead of hidden.
"""
from __future__ import annotations

import math

__all__ = [
    "HOGG_FUERSTENAU_COEFFICIENT",
    "HOGG_FUERSTENAU_EXPONENT",
    "charge_bulk_density",
    "hogg_fuerstenau_power_kw",
    "mill_power_density_kw_m3",
    "specific_power_kwh_t",
    "torque_factor",
]

#: Leading coefficient of the Hogg & Fuerstenau (1972) power correlation.
HOGG_FUERSTENAU_COEFFICIENT = 7.33

#: Diameter exponent of the same correlation.
HOGG_FUERSTENAU_EXPONENT = 2.3

#: Packing porosity of a static ball charge (bulk/steel density ratio ~0.60).
BALL_CHARGE_POROSITY = 0.40


def charge_bulk_density(solid_density: float, porosity: float = BALL_CHARGE_POROSITY) -> float:
    """Bulk density [kg/m^3] of the ball charge from the solid (steel) density."""
    rho = float(solid_density)
    eps = float(porosity)
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("solid_density must be finite and positive")
    if not 0.0 <= eps < 1.0:
        raise ValueError("porosity must be in [0, 1)")
    return rho * (1.0 - eps)


def hogg_fuerstenau_power_kw(diameter: float, length: float, fill: float,
                             fraction_critical: float, bulk_density: float,
                             coefficient: float = HOGG_FUERSTENAU_COEFFICIENT,
                             exponent: float = HOGG_FUERSTENAU_EXPONENT) -> float:
    """Empirical tumbling-mill power [kW] (Hogg & Fuerstenau 1972).

    Units: ``diameter``/``length`` in m, ``bulk_density`` in kg/m^3 (converted
    internally to t/m^3), ``fill`` and ``fraction_critical`` dimensionless.
    """
    d = float(diameter)
    l = float(length)
    j = float(fill)
    phi = float(fraction_critical)
    rho = float(bulk_density)
    for name, value in (("diameter", d), ("length", l), ("bulk_density", rho)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0.0 < j < 1.0:
        raise ValueError("fill must be in (0, 1)")
    if not 0.0 < phi <= 1.5:
        raise ValueError("fraction_critical must be in (0, 1.5]")
    rho_t = rho / 1000.0
    return (float(coefficient) * phi * j * (1.0 - 0.937 * j) * rho_t
            * l * d ** float(exponent))


def mill_power_density_kw_m3(power_kw: float, diameter: float, length: float) -> float:
    """Power per unit of mill interior volume [kW/m^3]."""
    v = math.pi * 0.25 * float(diameter) ** 2 * float(length)
    if v <= 0.0:
        raise ValueError("mill volume must be positive")
    return float(power_kw) / v


def specific_power_kwh_t(power_w: float, throughput_kg_per_h: float) -> float:
    """Specific grinding energy [kWh per tonne] for a given mass flow."""
    m = float(throughput_kg_per_h)
    if not math.isfinite(m) or m <= 0.0:
        raise ValueError("throughput_kg_per_h must be finite and positive")
    return (float(power_w) / 1000.0) / (m / 1000.0)


def torque_factor(torque_nm: float, charge_mass_kg: float, radius_m: float,
                  gravity: float = 9.80665) -> float:
    """Dimensionless torque factor ``TF = tau / (M_b g R)``."""
    m = float(charge_mass_kg)
    r = float(radius_m)
    g = float(gravity)
    if not math.isfinite(m) or m <= 0.0:
        raise ValueError("charge_mass_kg must be finite and positive")
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError("radius_m must be finite and positive")
    return float(torque_nm) / (m * g * r)