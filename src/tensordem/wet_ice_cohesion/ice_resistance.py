"""Ice-resistance enhancement factor of wet (cohesive) ice vs dry ice.

What is computed
----------------
``enhancement_factor(tau_wet, tau_dry)`` -- the ratio of the shear resistance of
a liquid-bridge-bearing granular ice assembly to the same assembly without
liquid.  In a Mohr-Coulomb rubble description

    tau = c + sigma_n tan(phi)

the dry assembly has ``c = 0`` (no tensile capacity) and the wet one has the
capillary cohesion ``c_cap`` delivered by the bridge network, so

    F_enh = 1 + c_cap / (sigma_n tan(phi)).

The campaign evaluates this quantity along three independent routes and reports
all three:

1. **DEM** (:mod:`tensordem.wet_ice_cohesion.shear_cell`) -- measured ``tau``.
2. **Micro-mechanical (Rumpf-type)** -- ``c_cap`` from the *measured* coordination
   number of the same cell and the validated bridge force; this is the
   prediction a capillary-cohesion theory makes for that very packing.
3. **Literature relation (Timco & Weeks 2010)** -- the sensitivity of granular
   ice strength to pore liquid content.  See the evidence note below.

Evidence note on route 3 (read this before quoting a number)
------------------------------------------------------------
Timco & Weeks (2010, *A review of the engineering properties of sea ice*,
Cold Reg. Sci. Technol. 60) is a compendium of *sea-ice material* properties:
brine volume, density, compressive/shear strength, friction.  It does **not**
tabulate a "wet vs dry granular ice resistance ratio".  What it does provide,
and what is used here, is

* the brine-volume relation ``v_b = S_a (49.185/T + 0.532)`` (Frankenstein &
  Garner 1967, reproduced in Timco & Weeks 2010 eq. for ``v_b``), with salinity
  ``S_a`` in ppt and ``T`` in degC, and
* the porosity-limited compressive strength ``sigma_c = 37 * eps_dot**0.22 *
  (1 - sqrt(v_b / 0.29))**2`` (MPa, ``eps_dot`` in 1/s), valid for
  ``v_b < 0.29``.

Two model inputs are therefore graded differently, and the campaign JSON says so:

* the *capillary* branch of ``F_enh`` is REAL (validated force law + DEM),
* the *ice-matrix* factor ``f_TW = (1 - sqrt(v_b/0.29))**2`` is a literature
  relation (coefficients transcribed from the published fit, not re-digitised
  here), so any number that depends on it is graded WITHHELD as experimental
  evidence and is reported as a literature-relation comparison.

``timco_weeks_reference`` returns that branch explicitly, including the
``matrix_strength_ratio = f_TW(wet) / f_TW(dry)`` which is *below* one: pore
liquid weakens the solid ice matrix.  This is the physical counter-term to
capillary cohesion and the reason the campaign reports the particle-size window
in which cohesion can matter at all (Bond number).
"""
from __future__ import annotations

import math

from .capillary import capillary_force

__all__ = [
    "GAMMA_WATER_0C",
    "bond_number",
    "capillary_cohesion",
    "enhancement_factor",
    "mohr_coulomb_fit",
    "rumpf_cohesion",
    "timco_weeks_brine_volume",
    "timco_weeks_reference",
    "timco_weeks_strength_factor",
]

#: surface tension of (fresh) water at 0 degC, N/m (IAPWS R1-76)
GAMMA_WATER_0C = 0.0759


def bond_number(radius: float, gamma: float = GAMMA_WATER_0C,
                density: float = 917.0, gravity: float = 9.81) -> float:
    """``Bo = rho g R^2 / gamma`` -- capillary force over particle weight.

    ``Bo << 1`` (``R < sqrt(gamma/(rho g)) ~ 2.9 mm`` for water/ice) is the window
    in which pendular bridges can carry the assembly; above it gravity wins and
    capillary cohesion becomes a small perturbation of the dry strength.
    """
    if radius <= 0 or gamma <= 0:
        raise ValueError("radius and gamma must be positive")
    return density * gravity * radius ** 2 / gamma


def capillary_cohesion(
    radius: float,
    gamma: float,
    coordination: float,
    bridge_volume_fraction: float = 0.02,
    contact_angle_deg: float = 0.0,
    gap: float = 0.0,
) -> float:
    """Isotropic capillary cohesion of a bridge network (stress units).

    ``c = 0.5 * Z * F_bridge / A_particle`` with
    ``A_particle = pi R^2 / phi`` the area (volume in 3D) per particle and ``phi``
    the solid fraction of the *packing*; here the plain particle footprint is
    used for the 2D cell (``A = 4 R^2``) so the formula matches the shear cell it
    is compared with.  ``gamma = 0`` returns exactly ``0.0``.
    """
    if gamma == 0.0:
        return 0.0
    if coordination < 0.0:
        raise ValueError("coordination must be non-negative")
    force = capillary_force(
        radius, gamma, bridge_volume_fraction * radius ** 3, gap, contact_angle_deg
    )
    area = (2.0 * radius) ** 2
    return 0.5 * coordination * force / area


def rumpf_cohesion(
    radius: float,
    gamma: float,
    coordination: float,
    bridge_volume_fraction: float = 0.02,
    contact_angle_deg: float = 0.0,
    solid_fraction: float = 0.8,
) -> float:
    """3D Rumpf (1962) granule tensile-strength estimate ``sigma_t``.

    ``sigma_t = (1 - phi)/phi * Z F / (pi d^2)`` with ``d = 2R``; the 3D analogue
    of :func:`capillary_cohesion`, used for the continuum cross-check.
    """
    if gamma == 0.0:
        return 0.0
    force = capillary_force(
        radius, gamma, bridge_volume_fraction * radius ** 3, 0.0, contact_angle_deg
    )
    d = 2.0 * radius
    return (1.0 - solid_fraction) / solid_fraction * coordination * force / (math.pi * d * d)


def enhancement_factor(tau_wet: float, tau_dry: float) -> float:
    """``F_wet / F_dry`` -- the ice-resistance enhancement factor."""
    if tau_dry <= 0.0:
        raise ValueError("the dry reference stress must be positive")
    return tau_wet / tau_dry


def mohr_coulomb_fit(sigma_n, tau):
    """Least-squares ``tau = c + sigma_n tan(phi)``.

    Returns ``{"cohesion", "tan_phi", "phi_deg", "rms"}``.  Two points give an
    exact interpolant, so the caller must check ``rms`` (a 2-point fit has
    ``rms = 0`` by construction and is *not* evidence of linearity).
    """
    import numpy as np

    s = np.asarray(sigma_n, dtype=float)
    t = np.asarray(tau, dtype=float)
    if s.size < 2:
        raise ValueError("need at least two normal stresses")
    A = np.vstack([np.ones_like(s), s]).T
    (c, tan_phi), *_ = np.linalg.lstsq(A, t, rcond=None)
    resid = t - A @ np.array([c, tan_phi])
    return {
        "cohesion": float(c),
        "tan_phi": float(tan_phi),
        "phi_deg": math.degrees(math.atan(float(tan_phi))),
        "rms": float(math.sqrt(float(np.mean(resid ** 2)))),
    }


# ---------------------------------------------------------------------------
# Timco & Weeks (2010) branch
# ---------------------------------------------------------------------------

def timco_weeks_brine_volume(salinity_ppt: float, temperature_c: float) -> float:
    """Brine volume fraction ``v_b = S_a (49.185/|T| + 0.532) / 1000``.

    Frankenstein & Garner (1967) as reproduced in Timco & Weeks (2010): the
    bracket is written with the *magnitude* of the temperature in degC and the
    result is per mille of bulk volume, hence the ``/1000``.  ``T`` must be
    negative (the relation is fitted for ``-30 <= T <= -2`` degC, ``S_a`` in ppt).

    Spot check used in the tests: ``S_a = 5 ppt``, ``T = -5 degC`` gives
    ``v_b = 0.0519`` (5.2 % brine), the value quoted for first-year sea ice at
    that temperature.
    """
    if temperature_c >= 0.0:
        raise ValueError("the relation is valid for T < 0 degC")
    if salinity_ppt < 0.0:
        raise ValueError("salinity must be non-negative")
    return salinity_ppt * (49.185 / abs(temperature_c) + 0.532) / 1000.0


def timco_weeks_strength_factor(brine_volume: float) -> float:
    """Porosity factor ``(1 - sqrt(v_b/0.29))**2`` of the Timco-Weeks strength law."""
    if brine_volume < 0.0:
        raise ValueError("brine volume must be non-negative")
    if brine_volume >= 0.29:
        return 0.0
    return (1.0 - math.sqrt(brine_volume / 0.29)) ** 2


def timco_weeks_reference(
    salinity_ppt: float,
    temperature_c: float,
    liquid_saturation: float,
    cohesion: float,
    tan_phi: float,
    normal_stress: float,
) -> dict:
    """Literature-relation branch of the enhancement factor.

    * dry state: ``v_b = v_b(S_a, T)`` (pore liquid is only the brine)
    * wet state: the pore space additionally carries liquid water so the
      effective liquid fraction is scaled by ``liquid_saturation`` and capped just
      below the 0.29 pore-liquid limit
    * the *matrix* strength ratio is ``f_TW(wet)/f_TW(dry)`` (< 1: liquid weakens
      solid ice)
    * the *capillary* part is the measured DEM/Mohr-Coulomb cohesion, evaluated
      as an added resistance

    Returns the ratio with its two branches separated so the reader can see which
    part is literature and which part is model.
    """
    if not (0.0 <= liquid_saturation <= 1.0):
        raise ValueError("liquid_saturation must be in [0, 1]")
    vb_dry = timco_weeks_brine_volume(salinity_ppt, temperature_c)
    # wet state: pore liquid grows with the added water, capped just below the
    # 0.29 pore-liquid limit of the strength relation
    vb_wet = min(0.28, vb_dry + liquid_saturation * (1.0 - vb_dry))
    f_dry = timco_weeks_strength_factor(vb_dry)
    f_wet = timco_weeks_strength_factor(vb_wet)
    matrix_ratio = (f_wet / f_dry) if f_dry > 0 else float("nan")
    tau_dry = max(1e-12, normal_stress * tan_phi * f_dry)
    tau_wet = normal_stress * tan_phi * f_wet + cohesion
    return {
        "brine_volume_dry": vb_dry,
        "brine_volume_wet": vb_wet,
        "strength_factor_dry": f_dry,
        "strength_factor_wet": f_wet,
        "matrix_strength_ratio": matrix_ratio,
        "capillary_cohesion": cohesion,
        "tau_dry_estimate": tau_dry,
        "tau_wet_estimate": tau_wet,
        "enhancement_factor": tau_wet / tau_dry,
        "grade": (
            "literature-relation (Timco & Weeks 2010 strength/brine relations); "
            "NOT an independent measurement of a wet/dry granular resistance ratio"
        ),
    }