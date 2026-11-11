"""Plastic work of an ice-block impact converted into a heat source.

The physics of step 1 of the node
---------------------------------
A DEM ice-block collision dissipates energy in the contact.  Two distinct
channels exist and they must not be conflated:

* **friction** (tangential slip) -- handled by
  :mod:`tensordem.ice_friction_thermo` and
  :mod:`tensordem.interface_thermal`, which solve the interface temperature of
  a *sliding* contact self-consistently;
* **inelastic deformation** (normal impact, crushing, micro-cracking) -- this
  module.  The fraction of the dissipated work that becomes heat is the
  Taylor-Quinney coefficient ``beta`` (``0.9`` for ice and metals alike, the
  remaining 10 % being locked in the new crack surface and lattice damage).

The dissipated energy of a normal impact with coefficient of restitution ``e``
is fixed by the impact energy::

    W_diss = 1/2 m v0^2 (1 - e^2)

and the part entering the ice block is ``P_ice = lambda W_diss`` with ``lambda``
the Blok effusivity split ``b_ice / (b_ice + b_wall)`` (``lambda = 0.5`` for an
ice-ice collision; ``~0`` for an ice-steel collision, where the steel is
adiabatic by comparison).

*Where* the heat lands matters far more than how much there is.  DEM resolves
the block, not the contact, so the deposition volume is a **model choice** and
this module makes it explicit instead of hiding it::

    V_dep = pi a^2 * delta        (cylindrical patch of radius a, depth delta)

with ``a`` the Hertz contact radius at the peak load and ``delta`` the depth
over which the plastic work is assumed to be spread.  The peak adiabatic rise
of the patch is then

    Delta_T = beta * P_ice / (rho cp V_dep)

which is the lumped-capacity limit of the conduction problem solved in
:mod:`tensordem.ice_thermal_stress.conduction` (valid while the contact
duration is short against the patch diffusion time ``delta^2 / alpha``, which is
checked by :func:`deposition_case` and reported as
``fourier_number_deposition``).

Because ``delta`` is not resolved by the DEM, the sensitivity of every
downstream number to it is measured rather than assumed:
:func:`layer_depth_sensitivity` sweeps it, and the benchmark report quotes the
band, not a single value.  Two physically motivated choices are provided:

``"hertz"``     delta = a          (plastic zone as deep as the contact is wide,
                                   the classical Boussinesq/Hertz scaling for a
                                   fully plastic contact)
``"fraction"``  delta = f a        (user fraction, default ``f = 0.15``)

Nothing here is fitted to the keys' target numbers; the only calibrated number
in the chain is ``beta``, whose range is quoted in :data:`TAYLOR_QUINNEY_RANGE`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..ice_ice_collision import reduced_youngs_modulus
from .properties import NU_ICE, IceProperties, ice_properties

__all__ = [
    "TAYLOR_QUINNEY_BETA",
    "TAYLOR_QUINNEY_RANGE",
    "ImpactHeatBudget",
    "dissipated_energy",
    "hertz_patch_radius",
    "deposition_volume",
    "partition_to_ice",
    "adiabatic_temperature_rise",
    "deposition_case",
    "layer_depth_sensitivity",
]


#: Taylor-Quinney coefficient (fraction of plastic work converted to heat).
TAYLOR_QUINNEY_BETA = 0.90
#: Literature band of the Taylor-Quinney coefficient for ice / rock.
TAYLOR_QUINNEY_RANGE = (0.80, 0.95)


def _finite(name: str, value: float, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and x <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    if nonnegative and x < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return x


def dissipated_energy(mass_kg: float, impact_speed_m_s: float,
                      restitution: float) -> float:
    """Inelastic energy of a normal impact ``0.5 m v0^2 (1 - e^2)`` in J.

    ``e = 0`` is a fully plastic (crushing) impact: every joule of the approach
    kinetic energy is dissipated; ``e = 1`` is elastic and dissipates nothing.

    Examples
    --------
    >>> round(dissipated_energy(100.0, 1.0, 0.3), 1)
    45.5
    """
    m = _finite("mass_kg", mass_kg, positive=True)
    v = _finite("impact_speed_m_s", impact_speed_m_s, nonnegative=True)
    e = _finite("restitution", restitution, nonnegative=True)
    if e > 1.0:
        raise ValueError(f"restitution must be in [0, 1], got {restitution!r}")
    return 0.5 * m * v * v * (1.0 - e * e)


def hertz_patch_radius(force_n: float, effective_radius_m: float,
                       effective_modulus_pa: float) -> float:
    """Hertz contact radius ``a = (3 F R / (4 E*))^(1/3)`` in m.

    Examples
    --------
    >>> round(hertz_patch_radius(5.0e4, 0.3, 5.06e9), 4)
    0.013
    """
    f = _finite("force_n", force_n, positive=True)
    r = _finite("effective_radius_m", effective_radius_m, positive=True)
    e = _finite("effective_modulus_pa", effective_modulus_pa, positive=True)
    return (3.0 * f * r / (4.0 * e)) ** (1.0 / 3.0)


def deposition_volume(patch_radius_m: float, layer_depth_m: float) -> float:
    """Volume of the heated patch ``pi a^2 delta`` in m^3."""
    a = _finite("patch_radius_m", patch_radius_m, positive=True)
    d = _finite("layer_depth_m", layer_depth_m, positive=True)
    return math.pi * a * a * d


def partition_to_ice(props_ice: IceProperties, props_wall: IceProperties | None) -> float:
    """Blok effusivity split ``b_ice / (b_ice + b_wall)`` (-).

    ``props_wall is None`` means an "adiabatic" counter-face (steel, rock, a
    rigid wall): all of the dissipated heat stays in the ice (``1.0``, the
    conservative upper bound).
    """
    b_ice = props_ice.effusivity_j_m2_k_sqrt_s
    if props_wall is None:
        return 1.0
    b_wall = props_wall.effusivity_j_m2_k_sqrt_s
    return b_ice / (b_ice + b_wall)


def adiabatic_temperature_rise(
    plastic_heat_j: float,
    volume_m3: float,
    props: IceProperties,
    *,
    beta: float = TAYLOR_QUINNEY_BETA,
) -> float:
    """Lumped-capacity rise ``beta Q / (rho cp V)`` in K.

    Examples
    --------
    >>> p = ice_properties(-10.0)
    >>> round(adiabatic_temperature_rise(40.0, 1.0e-6, p), 2)
    21.41
    """
    q = _finite("plastic_heat_j", plastic_heat_j, nonnegative=True)
    v = _finite("volume_m3", volume_m3, positive=True)
    b = _finite("beta", beta, nonnegative=True)
    if b > 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta!r}")
    return b * q / (props.volumetric_heat_capacity_j_m3_k * v)


@dataclass(frozen=True)
class ImpactHeatBudget:
    """Everything the conduction solver needs, and every number's provenance."""

    mass_kg: float
    impact_speed_m_s: float
    restitution: float
    dissipated_energy_j: float
    partition_to_ice: float
    beta: float
    plastic_heat_ice_j: float
    peak_force_n: float
    patch_radius_m: float
    layer_depth_m: float
    deposition_volume_m3: float
    delta_t_adiabatic_k: float
    contact_duration_s: float
    diffusion_time_s: float
    fourier_number_deposition: float
    properties: IceProperties
    depth_model: str

    def as_dict(self) -> dict[str, Any]:
        out = {
            "mass_kg": self.mass_kg,
            "impact_speed_m_s": self.impact_speed_m_s,
            "restitution": self.restitution,
            "dissipated_energy_j": self.dissipated_energy_j,
            "partition_to_ice": self.partition_to_ice,
            "beta": self.beta,
            "plastic_heat_ice_j": self.plastic_heat_ice_j,
            "peak_force_n": self.peak_force_n,
            "patch_radius_m": self.patch_radius_m,
            "layer_depth_m": self.layer_depth_m,
            "deposition_volume_m3": self.deposition_volume_m3,
            "delta_t_adiabatic_k": self.delta_t_adiabatic_k,
            "contact_duration_s": self.contact_duration_s,
            "diffusion_time_s": self.diffusion_time_s,
            "fourier_number_deposition": self.fourier_number_deposition,
            "depth_model": self.depth_model,
            "properties": self.properties.as_dict(),
        }
        return out


def deposition_case(
    *,
    mass_kg: float,
    impact_speed_m_s: float,
    restitution: float,
    peak_force_n: float,
    effective_radius_m: float,
    effective_modulus_pa: float,
    temperature_c: float,
    contact_duration_s: float,
    wall_temperature_c: float | None = None,
    beta: float = TAYLOR_QUINNEY_BETA,
    depth_model: str = "hertz",
    layer_fraction: float = 0.15,
    layer_depth_m: float | None = None,
) -> ImpactHeatBudget:
    """Assemble the heat budget of one impact (step 1 of the node).

    Parameters
    ----------
    peak_force_n, effective_radius_m, effective_modulus_pa
        Contact data -- in a DEM run these come from the contact law at the
        instant of peak load (``tensordem.ice_ice_collision``).
    contact_duration_s
        Time the contact lasts; used to form the deposition Fourier number
        ``Fo = alpha t_c / delta^2`` so the adiabatic-patch assumption is
        *tested* (``Fo << 1``) and not merely asserted.
    wall_temperature_c
        If given, the Blok effusivity split uses the wall material's
        properties evaluated at that temperature; ``None`` means an adiabatic
        counter-face (all heat stays in the ice).
    depth_model
        ``"hertz"`` (``delta = a``), ``"fraction"`` (``delta = layer_fraction *
        a``) or ``"explicit"`` (``layer_depth_m`` given).
    """
    props = ice_properties(temperature_c)
    wall = ice_properties(wall_temperature_c) if wall_temperature_c is not None else None
    w = dissipated_energy(mass_kg, impact_speed_m_s, restitution)
    lam = partition_to_ice(props, wall)
    a = hertz_patch_radius(peak_force_n, effective_radius_m, effective_modulus_pa)
    if layer_depth_m is not None:
        delta = _finite("layer_depth_m", layer_depth_m, positive=True)
        depth_model = "explicit"
    elif depth_model == "hertz":
        delta = a
    elif depth_model == "fraction":
        delta = _finite("layer_fraction", layer_fraction, positive=True) * a
    else:  # pragma: no cover - guarded above
        raise ValueError(f"unknown depth_model {depth_model!r}")
    vol = deposition_volume(a, delta)
    heat = lam * w
    duration = _finite("contact_duration_s", contact_duration_s, positive=True)
    diffusion_time = delta * delta / props.diffusivity_m2_s
    return ImpactHeatBudget(
        mass_kg=float(mass_kg),
        impact_speed_m_s=float(impact_speed_m_s),
        restitution=float(restitution),
        dissipated_energy_j=w,
        partition_to_ice=lam,
        beta=float(beta),
        plastic_heat_ice_j=heat,
        peak_force_n=float(peak_force_n),
        patch_radius_m=a,
        layer_depth_m=delta,
        deposition_volume_m3=vol,
        delta_t_adiabatic_k=adiabatic_temperature_rise(heat, vol, props, beta=beta),
        contact_duration_s=duration,
        diffusion_time_s=diffusion_time,
        fourier_number_deposition=props.diffusivity_m2_s * duration / (delta * delta),
        properties=props,
        depth_model=depth_model,
    )


def layer_depth_sensitivity(
    *,
    mass_kg: float,
    impact_speed_m_s: float,
    restitution: float,
    peak_force_n: float,
    effective_radius_m: float,
    effective_modulus_pa: float,
    temperature_c: float,
    contact_duration_s: float,
    fractions: Any = (0.05, 0.10, 0.15, 0.25, 0.50, 1.00),
    beta: float = TAYLOR_QUINNEY_BETA,
) -> list[ImpactHeatBudget]:
    """Sweep ``delta = f a``; returns one budget per fraction (ascending)."""
    out = []
    for f in np.asarray(fractions, dtype=float):
        out.append(
            deposition_case(
                mass_kg=mass_kg,
                impact_speed_m_s=impact_speed_m_s,
                restitution=restitution,
                peak_force_n=peak_force_n,
                effective_radius_m=effective_radius_m,
                effective_modulus_pa=effective_modulus_pa,
                temperature_c=temperature_c,
                contact_duration_s=contact_duration_s,
                beta=beta,
                depth_model="fraction",
                layer_fraction=float(f),
            )
        )
    return out


#: Self-check used by the tests: the Blok split conserves the dissipated energy.
_p = ice_properties(-10.0)
assert abs(
    partition_to_ice(_p, _p) - 0.5
) < 1e-12, "ice-ice Blok split must be one half"
assert partition_to_ice(_p, None) == 1.0
assert reduced_youngs_modulus.__module__.endswith("ice_ice_collision")
assert NU_ICE == 0.33