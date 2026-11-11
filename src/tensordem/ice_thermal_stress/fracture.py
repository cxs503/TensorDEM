"""Thermal-mechanical superposition and the thermal contribution to breakage.

Step 4 of the node: ``sigma_total = sigma_mechanical + sigma_thermal`` against a
temperature dependent strength ``sigma_f(T)``.  Two things make this more than a
sum:

**Sign matters.**  Ice is much weaker in tension than in compression, so the
two branches are treated separately:

* **compression branch** (contact crushing): both ``sigma_mech`` and a *heating*
  ``sigma_thermal`` are negative and add; the relevant strength is
  ``sigma_f_c(T)`` (:func:`tensordem.ice_material.pure_ice_strength`, i.e. the
  Timco & Frederking temperature law already used by the rest of the library).
* **tension branch** (surface cracking on rapid *cooling*, or the release of the
  compressive patch at the edge of the contact): the tensile strength is
  ``sigma_f_t = RATIO_COMPRESSION_TO_TENSION * sigma_f_c`` with the classical
  ice ratio 3.0 (Schulson & Duval 2009 report 2-5; the value is a module
  constant so it can be swept, not a hidden number).

**Strength is evaluated at the *heated* temperature** ``T_bulk + dT`` -- that is
what "``sigma_fracture(T)``" in the node means, and it is why heating both adds
stress *and* softens the material: the two effects work in the same direction
for the compression branch and are accounted for separately here.

Breakage increment
------------------
The DEM-side quantity of interest is not "does one point fail" but "how much
*mores* of the contact population fails when the thermal term is switched on".
:func:`thermal_breakage_increment` takes the mechanical stress of a contact
population, adds the thermal term and returns the *change in failing fraction*.
For a deterministic threshold the increment is reported both as a count
fraction and as the equivalent strength margin
:func:`thermal_margin_fraction`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .properties import IceProperties, compressive_strength, ice_properties
from .thermal_stress import (
    field_stress_from_temperature,
    restrained_uniform_stress,
)

__all__ = [
    "RATIO_COMPRESSION_TO_TENSION",
    "FractureState",
    "superpose",
    "tensile_strength",
    "failure_margin",
    "thermal_margin_fraction",
    "thermal_breakage_increment",
    "breakage_increment_curve",
    "critical_heating_for_crushing",
    "strength_budget_bound",
]


#: ``sigma_f_compression / sigma_f_tension`` for ice (Schulson & Duval 2009).
RATIO_COMPRESSION_TO_TENSION = 3.0

#: Strength is evaluated no closer than this to the melting point (degC).
#: Phase change is deliberately outside this module.
MELTING_GUARD_C = -0.05


def _finite(name: str, value: float) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return x


def superpose(sigma_mech_pa: Any, sigma_thermal_pa: Any) -> Any:
    """``sigma_mech + sigma_thermal`` with compression negative.

    Examples
    --------
    >>> superpose(-40.0e6, -5.0e6) / 1e6
    -45.0
    """
    out = np.asarray(sigma_mech_pa, float) + np.asarray(sigma_thermal_pa, float)
    return float(out) if np.ndim(out) == 0 else out


def tensile_strength(
    temperature_c: Any,
    *,
    ratio: float = RATIO_COMPRESSION_TO_TENSION,
    salinity_ppt: float | None = None,
) -> Any:
    """Ice tensile strength (Pa) from the compressive law divided by ``ratio``."""
    r = _finite("ratio", ratio)
    if r <= 0.0:
        raise ValueError(f"ratio must be > 0, got {ratio!r}")
    sigma_c_mpa = compressive_strength(temperature_c, salinity_ppt=salinity_ppt)
    return np.asarray(sigma_c_mpa, float) * 1.0e6 / r


@dataclass(frozen=True)
class FractureState:
    """Failing / not failing, with the numbers that produced the verdict."""

    sigma_mech_pa: float
    sigma_thermal_pa: float
    sigma_total_pa: float
    temperature_c: float
    strength_pa: float
    branch: str
    failed: bool
    melted: bool = False

    @property
    def margin_pa(self) -> float:
        """``|sigma_total| - strength``: positive means failed (Pa)."""
        return abs(self.sigma_total_pa) - self.strength_pa

    def as_dict(self) -> dict[str, Any]:
        return {
            "sigma_mech_pa": self.sigma_mech_pa,
            "sigma_thermal_pa": self.sigma_thermal_pa,
            "sigma_total_pa": self.sigma_total_pa,
            "temperature_c": self.temperature_c,
            "strength_pa": self.strength_pa,
            "branch": self.branch,
            "failed": self.failed,
            "melted": self.melted,
            "margin_pa": self.margin_pa,
        }


def failure_margin(
    sigma_mech_pa: float,
    sigma_thermal_pa: float,
    temperature_c: float,
    *,
    salinity_ppt: float | None = None,
    ratio: float = RATIO_COMPRESSION_TO_TENSION,
    delta_t_k: float = 0.0,
) -> FractureState:
    """Evaluate the superposed criterion on one branch of the stress state.

    The branch is chosen by the **sign of the total stress**: ``sigma_total < 0``
    is compression (compare with ``sigma_f_c``), ``> 0`` is tension (compare
    with ``sigma_f_c / ratio``).  Failure is ``|sigma_total| >= strength``.

    ``delta_t_k`` (the local heating that produced ``sigma_thermal_pa``) shifts
    the temperature at which the strength is evaluated, so heating both adds
    stress *and* softens the ice -- the two effects of ``sigma_f(T)`` are not
    double counted, they are the same loading path seen twice.  The shifted
    temperature is clamped just below the melting point; a patch that would
    actually melt is **out of scope** (no latent heat, no melt film here) and is
    reported as a limitation by the benchmark rather than silently modelled.
    """
    mech = _finite("sigma_mech_pa", sigma_mech_pa)
    therm = _finite("sigma_thermal_pa", sigma_thermal_pa)
    temp = float(temperature_c)
    props = ice_properties(temp)
    heated = max(min(temp + _finite("delta_t_k", delta_t_k), MELTING_GUARD_C),
                 props.temperature_c)
    total = superpose(mech, therm)
    if total <= 0.0:
        branch = "compression"
        strength = float(np.asarray(compressive_strength(heated,
                                                         salinity_ppt=salinity_ppt))) * 1.0e6
    else:
        branch = "tension"
        strength = float(np.asarray(tensile_strength(heated, ratio=ratio,
                                                     salinity_ppt=salinity_ppt)))
    return FractureState(
        sigma_mech_pa=mech,
        sigma_thermal_pa=therm,
        sigma_total_pa=total,
        temperature_c=props.temperature_c,
        strength_pa=strength,
        branch=branch,
        failed=bool(abs(total) >= strength),
        melted=bool(heated >= MELTING_GUARD_C),
    )


def thermal_margin_fraction(
    sigma_mech_pa: Any,
    sigma_thermal_pa: Any,
    temperature_c: float,
) -> Any:
    """How much of the strength budget the *thermal* term consumes (-).

    ``|sigma_thermal| / (sigma_f(T) - |sigma_mech|)``: values ``>= 1`` mean the
    thermal term alone is enough to close the remaining margin.  Returns ``inf``
    where the mechanical term has already exhausted the strength.
    """
    mech = np.asarray(sigma_mech_pa, float)
    therm = np.asarray(sigma_thermal_pa, float)
    sigma_f = float(np.asarray(compressive_strength(temperature_c))) * 1.0e6
    rem = sigma_f - np.abs(mech)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(rem > 0.0, np.abs(therm) / np.where(rem > 0.0, rem, 1.0), np.inf)
    return float(out) if np.ndim(out) == 0 else out


def thermal_breakage_increment(
    sigma_mech_pa: Sequence[float] | np.ndarray,
    sigma_thermal_pa: float,
    temperature_c: float,
    *,
    delta_t_k: float = 0.0,
    salinity_ppt: float | None = None,
    ratio: float = RATIO_COMPRESSION_TO_TENSION,
) -> dict[str, float]:
    """Change in the failing fraction of a contact population when thermal adds.

    Parameters
    ----------
    sigma_mech_pa
        Mechanical stress of each contact in the population (Pa, compression
        negative -- the convention of the rest of the ice chain).
    sigma_thermal_pa
        Thermal stress superposed on every contact (Pa).  A scalar keeps the
        function pure; the collision driver passes the patch-averaged value.

    Returns
    -------
    dict with ``fraction_mech``, ``fraction_total``, ``increment`` (the
    difference), ``count_mech``, ``count_total`` and ``n``.
    """
    mech = np.atleast_1d(np.asarray(sigma_mech_pa, dtype=float))
    if mech.size == 0:
        raise ValueError("sigma_mech_pa must not be empty")
    therm = _finite("sigma_thermal_pa", sigma_thermal_pa)
    n = mech.size
    failed_mech = 0
    failed_total = 0
    for value in mech:
        if failure_margin(float(value), 0.0, temperature_c,
                          salinity_ppt=salinity_ppt, ratio=ratio).failed:
            failed_mech += 1
        if failure_margin(float(value), therm, temperature_c,
                          salinity_ppt=salinity_ppt, ratio=ratio,
                          delta_t_k=delta_t_k).failed:
            failed_total += 1
    return {
        "n": float(n),
        "count_mech": float(failed_mech),
        "count_total": float(failed_total),
        "fraction_mech": failed_mech / n,
        "fraction_total": failed_total / n,
        "increment": (failed_total - failed_mech) / n,
    }


def breakage_increment_curve(
    sigma_mech_pa: Sequence[float] | np.ndarray,
    delta_t_k: Any,
    temperature_c: float,
    *,
    restraint: str = "restrained",
    salinity_ppt: float | None = None,
    ratio: float = RATIO_COMPRESSION_TO_TENSION,
) -> dict[str, np.ndarray]:
    """Sweep the temperature rise and return the breakage increment curve.

    The thermal stress is computed with :func:`restrained_uniform_stress` (the
    patch-scale bound, see :mod:`tensordem.ice_thermal_stress.thermal_stress`);
    ``restraint="free"`` halves it.
    """
    props = ice_properties(temperature_c)
    dts = np.atleast_1d(np.asarray(delta_t_k, dtype=float))
    increments = np.empty_like(dts)
    stress = np.empty_like(dts)
    for i, dt in enumerate(dts):
        sigma_th = float(restrained_uniform_stress(float(dt), props))
        if restraint == "free":
            sigma_th *= 0.5
        elif restraint != "restrained":
            raise ValueError(f"unknown restraint {restraint!r}")
        stress[i] = sigma_th
        increments[i] = thermal_breakage_increment(
            sigma_mech_pa, sigma_th, temperature_c, delta_t_k=float(dt),
            salinity_ppt=salinity_ppt, ratio=ratio)["increment"]
    return {"delta_t_k": dts, "sigma_thermal_pa": stress, "increment": increments}


def critical_heating_for_crushing(
    sigma_mech_pa: float,
    temperature_c: float,
    *,
    salinity_ppt: float | None = None,
    restraint: str = "restrained",
) -> float:
    """Temperature rise that makes the superposed compression reach strength (K).

    ``dT_c = (sigma_f(T) - |sigma_mech|) (1 - nu) / (E alpha_T)`` (doubled for
    the free branch).  Negative input (``|sigma_mech| > sigma_f``) returns 0.
    """
    mech = abs(_finite("sigma_mech_pa", sigma_mech_pa))
    sigma_f = float(np.asarray(compressive_strength(temperature_c,
                                                    salinity_ppt=salinity_ppt))) * 1.0e6
    props = ice_properties(temperature_c)
    coeff = props.thermal_stress_coefficient_pa_k
    remaining = sigma_f - mech
    if remaining <= 0.0:
        return 0.0
    value = remaining / coeff
    return value if restraint == "restrained" else 2.0 * value


def strength_budget_bound(
    sigma_thermal_pa: float,
    temperature_c: float,
    *,
    delta_t_k: float = 0.0,
    salinity_ppt: float | None = None,
) -> float:
    """Upper bound on the breakage increment for *any* stress population.

    Two mechanisms consume the strength budget, and both are thermal:

    ``softening``   ``1 - sigma_f(T + dT) / sigma_f(T)`` -- heating melts the
                    ice's strength away;
    ``stress``      ``|sigma_thermal| / sigma_f(T)`` -- the superposed stress.

    For a population spread uniformly over ``[0, sigma_f]`` the failing fraction
    grows by exactly the sum of the two (clipped at 1), so this is a *tight*
    bound for that population and an upper bound in general.  It is what the
    coupled impact driver reports next to the geometry-resolved increment.
    """
    sigma_th = abs(_finite("sigma_thermal_pa", sigma_thermal_pa))
    t = float(temperature_c)
    s_bulk = float(np.asarray(compressive_strength(t, salinity_ppt=salinity_ppt))) * 1.0e6
    heated = max(min(t + _finite("delta_t_k", delta_t_k), MELTING_GUARD_C), t)
    s_heated = float(np.asarray(compressive_strength(heated, salinity_ppt=salinity_ppt))) * 1.0e6
    softening = max(0.0, 1.0 - s_heated / s_bulk)
    return float(min(1.0, softening + sigma_th / s_bulk))


def thermal_stress_from_field(
    temperature_profile: Any, temperature_c: float, *, restraint: str = "restrained",
) -> np.ndarray:
    """Convenience wrapper: stress field from a computed ``T(z)`` profile."""
    return field_stress_from_temperature(temperature_profile, ice_properties(temperature_c),
                                         restraint=restraint)


#: module self-checks
assert abs(RATIO_COMPRESSION_TO_TENSION - 3.0) < 1e-12
assert failure_margin(-40.0e6, -5.0e6, -10.0).branch == "compression"
assert failure_margin(+10.0e6, 0.0, -10.0).branch == "tension"
assert critical_heating_for_crushing(0.0, -10.0) > 0.0
assert critical_heating_for_crushing(1.0e9, -10.0) == 0.0