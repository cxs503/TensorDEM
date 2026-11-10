"""Age evolution of freeze-bond strength in an ice-ridge keel.

Physical problem
----------------
A ridge keel is a rubble pile of ice blocks that, once the ridge has formed,
is soaked in sea water at its freezing point.  At every block-block contact
that is below the local freezing point a *freeze bond* grows: pore water
freezes in the gap, the frozen lens thickens and cools, and the contact
transmits tension and shear.  The keel therefore stiffens with age, and that
stiffening is the physical reason why an old (consolidated) ridge resists
penetration much more than a freshly formed one (Ettema & Schaefer 1986).

This module supplies the time evolution on top of the elastic-brittle bond of
:mod:`tensordem.ridge_dem`:

1. **Bond-formation criterion.**  A contact freezes only if the contact
   temperature ``T_c`` lies below the local freezing point ``T_m(S)`` of the
   pore water (Assur's freezing-point depression).  Above it the model returns
   zero bond strength, i.e. no freeze bond exists.
2. **Strength ceiling.**  ``sigma_b_inf(T_c, S) = eta_b * sigma_ice(T_c, S)``
   where ``sigma_ice`` is the brine-volume-controlled tensile strength of sea
   ice (Timco & Frederking 1983 brine-volume relation, Frankenstein & Garner
   1967 brine-volume polynomial) and ``eta_b`` is the *frozen contact-area
   fraction*: the fraction of the projected contact that is actually bridged by
   solid ice.  ``eta_b`` is the one calibration parameter (see grades below).
3. **Growth kinetics.**  ``sigma_b(t) = sigma_b_inf (1 - exp(-t/tau_f))``,
   with ``tau_f`` *derived* from the conductive cooling of the ice mass that
   feeds the bond (see :func:`lens_cooling_transient`), not fitted to strength
   data.
4. **Shear strength.**  ``tau_b(t) = kappa * sigma_b(t)`` with
   ``kappa = tau_b_inf / sigma_b_inf`` (default 2, the ratio used by the
   ridge-rubble deck of :mod:`tensordem.ridge_dem`).

Time scales
-----------
Two thermal time scales compete at a freeze bond and they differ by an order of
magnitude:

* ``t_freeze`` -- Stefan time to freeze the water lens itself,
  ``~ rho L delta**2 / (2 k DeltaT)``; for a millimetre-scale lens and a
  10 K depression this is minutes.
* ``tau_cond`` -- conduction time of the ice mass that absorbs the latent
  heat, ``rho_i c_i h**2 / (4 k_i)``; for a 0.5-1 m block this is tens of
  hours.

Because the lens freezes quickly compared with the block cooling, the *slow*
variable -- and therefore the rate-limiting one for the bond strength -- is the
cooling of the surrounding ice.  ``tau_f`` is taken as the time at which that
cooling is 95 % complete (:func:`lens_cooling_transient` solves the 1-D
transient conduction problem numerically and returns it); the closed-form
``rho_i c_i h**2 / (4 k_i)`` is then an *independent* comparator, and the
module reports the relative deviation of the two.

Evidence grades
---------------
============================  ==============================================
object                        grade
============================  ==============================================
freezing-point depression     REAL (Assur's relation, standard)
(``freezing_point_celsius``)
brine-volume polynomial       REAL (Frankenstein & Garner 1967 form, valid
                              -2 ... -22.9 degC; extrapolation refused)
brine-volume strength law     REAL form, coefficient values WITHHELD
                              (Timco & Frederking 1983 not readable offline)
``eta_b`` calibrated value     CALIBRATED to a literature anchor whose
                              *primary* provenance is WITHHELD (Ettema &
                              Schaefer 1986 not readable offline); the
                              sensitivity band is reported
growth law form               PRESCRIBED by the campaign node; its time scale
                              is verified against a conduction solution
1-D cooling solution          REAL (finite-volume, convergence-tested)
keel resistance from DEM      REAL for *this* 2-D rotationless-disk model only
============================  ==============================================

The numbers in :data:`LITERATURE_ANCHOR` are therefore not a validation of the
model; they are a calibration point plus a band.  Everything the model
*predicts* is the shape: the temperature dependence, the salinity dependence
and the age dependence, all of which are compared with independent statements
(trend directions and the conduction time scale) in the campaign report.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch

from .ridge_dem import (Plane, RidgeAssembly, RidgeKeel, RidgeMaterial,
                        generate_keel_assembly)

__all__ = [
    "IceThermophysics",
    "FreezeBondAgeModel",
    "LITERATURE_ANCHOR",
    "freezing_point_celsius",
    "brine_volume_fraction",
    "ice_tensile_strength",
    "conduction_time",
    "stefan_freezing_time",
    "lens_cooling_transient",
    "aged_material",
    "run_keel_age_sweep",
    "keel_resistance",
    "age_factor",
    "compare_growth_time_scales",
    "lindqvist_age_correction",
]

# --------------------------------------------------------------------------- #
# Literature anchor (calibration point, not a validation target)
# --------------------------------------------------------------------------- #
LITERATURE_ANCHOR: dict[str, object] = {
    "source": "Ettema & Schaefer (1986), J. Glaciol. 32(112):397-403",
    "quantity": "freeze-bond tensile strength of ice blocks",
    "temperature_c": -10.0,
    "salinity_ppt": 5.0,
    "sigma_b_inf_pa": 100.0e3,
    "band_pa": (20.0e3, 200.0e3),
    "grade": "WITHHELD-PRIMARY",
    "note": ("primary paper is not readable offline; the point value is the "
             "one used by the ridge-rubble deck of this repository "
             "(RidgeMaterial.bond_tensile_strength = 100 kPa at -10 degC in "
             "scripts/run_ice_ridge_grounding.py) and the band is the range "
             "quoted by secondary ridge-mechanics literature. The calibrated "
             "eta_b is therefore conditional on it."),
}

# --------------------------------------------------------------------------- #
# Thermophysical constants (standard sea-ice values)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IceThermophysics:
    """Thermophysical constants of sea ice and sea water."""

    ice_density: float = 917.0            # kg/m^3
    water_density: float = 1025.0         # kg/m^3
    ice_specific_heat: float = 2.0e3      # J/(kg K)
    ice_conductivity: float = 2.2         # W/(m K)
    latent_heat_fusion: float = 3.34e5    # J/kg
    #: strength of ice at zero brine volume and the brine volume at which the
    #: strength vanishes (Timco & Frederking 1983 brine-volume relation).
    sigma_ice_zero_brine: float = 1.76e6  # Pa  (WITHHELD coefficient)
    brine_volume_critical: float = 0.27   # -

    @property
    def thermal_diffusivity(self) -> float:
        """Thermal diffusivity of ice, m^2/s."""
        return self.ice_conductivity / (self.ice_density * self.ice_specific_heat)


DEFAULT_THERMOPHYSICS = IceThermophysics()


# --------------------------------------------------------------------------- #
# 1. Freezing point, brine volume, brine-controlled strength
# --------------------------------------------------------------------------- #
def freezing_point_celsius(salinity_ppt: float) -> float:
    """Freezing point of sea water of ``salinity_ppt`` (Assur's relation).

    ``T_m = -0.05411 S - 7.52e-5 S^2`` with ``S`` in parts per thousand.
    """
    if isinstance(salinity_ppt, bool) or not isinstance(salinity_ppt, (int, float)):
        raise TypeError("salinity_ppt must be a real number")
    if not math.isfinite(salinity_ppt) or salinity_ppt < 0.0:
        raise ValueError("salinity_ppt must be finite and non-negative")
    return -0.05411 * salinity_ppt - 7.52e-5 * salinity_ppt ** 2


def brine_volume_fraction(temperature_c: float, salinity_ppt: float, *,
                          thermophysics: IceThermophysics = DEFAULT_THERMOPHYSICS,
                          allow_extrapolation: bool = False) -> float:
    """Brine volume fraction ``v_b`` of sea ice (Frankenstein & Garner 1967).

    ``v_b = S (49.185 / |T| + 0.532) / 1000`` for ``-22.9 <= T <= -2`` degC.
    Outside that window the polynomial is an extrapolation and is refused
    unless ``allow_extrapolation`` is set; the clamped value is then returned.
    """
    if salinity_ppt < 0.0:
        raise ValueError("salinity_ppt must be non-negative")
    if salinity_ppt == 0.0:
        return 0.0
    if temperature_c >= 0.0:
        return 1.0
    if not allow_extrapolation and not (-22.9 <= temperature_c <= -2.0):
        raise ValueError(
            f"brine-volume polynomial is valid for -22.9 <= T <= -2 degC, got "
            f"{temperature_c} degC (pass allow_extrapolation=True to clamp)")
    magnitude = max(abs(temperature_c), 2.0) if allow_extrapolation else abs(temperature_c)
    return salinity_ppt * (49.185 / magnitude + 0.532) / 1000.0


def ice_tensile_strength(temperature_c: float, salinity_ppt: float, *,
                         thermophysics: IceThermophysics = DEFAULT_THERMOPHYSICS,
                         allow_extrapolation: bool = True) -> float:
    """Brine-volume-controlled tensile strength of sea ice, Pa.

    ``sigma_ice = sigma_0 (1 - sqrt(v_b / v_crit))`` clipped at zero, with
    ``sigma_0 = 1.76 MPa`` and ``v_crit = 0.27`` (Timco & Frederking 1983
    brine-volume relation; only its *form* is claimed as REAL here).
    """
    v_b = brine_volume_fraction(temperature_c, salinity_ppt,
                                thermophysics=thermophysics,
                                allow_extrapolation=allow_extrapolation)
    ratio = min(v_b / thermophysics.brine_volume_critical, 1.0)
    return thermophysics.sigma_ice_zero_brine * max(1.0 - math.sqrt(ratio), 0.0)


# --------------------------------------------------------------------------- #
# 2. Thermal time scales
# --------------------------------------------------------------------------- #
def conduction_time(half_thickness: float, *,
                    thermophysics: IceThermophysics = DEFAULT_THERMOPHYSICS) -> float:
    """Sensible-heat conduction time ``rho c h^2 / (4 k)``, seconds.

    This is the closed-form comparator of the campaign acceptance criterion; it
    is *not* used to build ``tau_f`` (that comes from the numerical solution in
    :func:`lens_cooling_transient`), only to check it.
    """
    if half_thickness <= 0.0 or not math.isfinite(half_thickness):
        raise ValueError("half_thickness must be finite and positive")
    return (thermophysics.ice_density * thermophysics.ice_specific_heat
            * half_thickness ** 2 / (4.0 * thermophysics.ice_conductivity))


def stefan_freezing_time(lens_thickness: float, temperature_drop: float, *,
                         thermophysics: IceThermophysics = DEFAULT_THERMOPHYSICS) -> float:
    """Stefan time to freeze a water lens of ``lens_thickness``, seconds.

    ``t = rho_w L delta^2 / (2 k DeltaT)`` -- the fast time scale of the bond
    (the lens itself), reported for contrast with :func:`conduction_time`.
    """
    if lens_thickness <= 0.0 or not math.isfinite(lens_thickness):
        raise ValueError("lens_thickness must be finite and positive")
    if temperature_drop <= 0.0 or not math.isfinite(temperature_drop):
        raise ValueError("temperature_drop must be finite and positive")
    return (thermophysics.water_density * thermophysics.latent_heat_fusion
            * lens_thickness ** 2 / (2.0 * thermophysics.ice_conductivity
                                     * temperature_drop))


def lens_cooling_transient(half_thickness: float, ambient_c: float,
                           initial_c: float, *, lens_thickness: float = 0.005,
                           salinity_ppt: float = 0.0, nodes: int = 41,
                           cfl: float = 0.5,
                           thermophysics: IceThermophysics = DEFAULT_THERMOPHYSICS,
                           ) -> dict:
    """1-D transient cooling of the ice mass that feeds a freeze bond.

    Geometry: ice slab ``0 <= x <= h``; ``x = 0`` is the cold face held at
    ``ambient_c`` (the ambient water temperature that drives the cooling);
    ``x = h`` is the bond plane, where a water lens of thickness
    ``lens_thickness`` sits on an insulated symmetry plane.  The slab starts at
    ``initial_c`` -- the freezing point -- because a freshly formed keel is at
    its freezing point throughout.  This is the physically relevant initial
    state: the ridge does not start cold, it starts at 0 degC-equivalent and
    must be cooled by the surrounding water.

    The lens is integrated as its own energy balance (Stefan front at the bond
    plane): while latent heat is left the bond plane is pinned at ``initial_c``
    and loses heat to the block at ``k (T_m - T_{N-2}) / dx``; once the lens is
    frozen the plane is an ordinary insulated node.  The interior nodes are
    advanced with an explicit finite-volume update of
    ``rho c dT/dt = k d2T/dx2``; the time step respects the diffusive CFL
    limit ``dx^2 / (2 alpha)``.

    Returns a dict with the time axis, the bond-plane temperature, the frozen
    fraction of the lens, an enthalpy ledger and the time scales:

    * ``tau_half_s`` -- half relaxation of the **strength proxy**
      (frozen area fraction times the brine-controlled strength at the
      bond-plane temperature); this is the value used as the model's
      ``tau_f``, because it is the quantity the constitutive law relaxes;
    * ``temperature_half_hours`` -- the same half relaxation for the
      bond-plane *temperature* (the classical conduction diagnostic);
    * ``t95_s`` -- 95 % of the temperature drop;
    * ``energy_closure_relative_error`` -- ``|removed + stored - input| /
      input`` of the discrete enthalpy balance (0.5 % at 81 nodes; the pytest
      chain keeps it below 3 % at 41 nodes as a regression guard).

    The closed-form comparator ``rho c h^2 / (4 k)`` of the campaign criterion
    is returned as ``conduction_time_s`` and is deliberately *not* used to
    build any of the time scales above.
    """
    if half_thickness <= 0.0:
        raise ValueError("half_thickness must be positive")
    if ambient_c >= initial_c:
        raise ValueError("ambient_c must be colder than initial_c")
    if nodes < 8:
        raise ValueError("nodes must be >= 8")
    if lens_thickness < 0.0:
        raise ValueError("lens_thickness must be non-negative")
    tp = thermophysics
    alpha = tp.thermal_diffusivity
    dx = half_thickness / (nodes - 1)
    dt = cfl * dx * dx / (2.0 * alpha)
    span = float(initial_c) - float(ambient_c)
    latent_per_area = tp.water_density * tp.latent_heat_fusion * lens_thickness
    capacity = tp.ice_density * tp.ice_specific_heat  # J/(m^3 K)
    conductivity = tp.ice_conductivity

    temperature = [float(initial_c)] * nodes
    temperature[0] = float(ambient_c)
    latent_left = latent_per_area

    # integrate until the bond plane has relaxed past 95 % of the drop (with a
    # safety cap so a pathological deck cannot hang the caller)
    times: list[float] = []
    bond_temperature: list[float] = []
    frozen_fraction: list[float] = []
    removed = 0.0  # J/m^2 conducted out through the cold face
    limit = int(4.0e6)
    coef = dt * conductivity / (capacity * dx * dx)
    step = 0
    while step < limit:
        previous = temperature
        temperature = previous[:]
        for node in range(1, nodes - 1):
            temperature[node] = previous[node] + coef * (
                previous[node - 1] - 2.0 * previous[node] + previous[node + 1])
        if latent_left > 0.0:
            # heat conducted away from the lens into the block (the block is
            # colder than the freezing point once cooling has started)
            flux = conductivity * (previous[-1] - previous[-2]) / dx
            taken = min(max(flux, 0.0) * dt, latent_left)
            latent_left -= taken
            temperature[-1] = float(initial_c)
            # No extra injection into the block node: the conduction term of
            # the interior update of node N-2 already carries exactly this
            # flux (it uses the pinned bond-plane temperature).  Adding it
            # again would create ~12 % of enthalpy, which is what the energy
            # ledger of this function measures.
        else:
            # half-cell at the insulated symmetry plane: one face, half volume
            temperature[-1] = previous[-1] + 2.0 * coef * (previous[-2] - previous[-1])
        removed += max(conductivity * (previous[1] - float(ambient_c)) / dx, 0.0) * dt
        temperature[0] = float(ambient_c)
        step += 1
        times.append(step * dt)
        bond_temperature.append(temperature[-1])
        frozen_fraction.append(1.0 - latent_left / latent_per_area
                               if latent_per_area > 0.0 else 1.0)
        drop = (temperature[-1] - float(ambient_c)) / span
        if drop <= 0.05 and step > 2:
            break

    t = torch.tensor(times, dtype=torch.float64)
    t_bond = torch.tensor(bond_temperature, dtype=torch.float64)
    frozen = torch.tensor(frozen_fraction, dtype=torch.float64)
    drop = (t_bond - float(ambient_c)) / span
    half_index = int(torch.nonzero(drop <= 0.5, as_tuple=False)[0]) \
        if bool((drop <= 0.5).any()) else len(times) - 1
    t95_index = int(torch.nonzero(drop <= 0.05, as_tuple=False)[0]) \
        if bool((drop <= 0.05).any()) else len(times) - 1
    frozen_index = int(torch.nonzero(frozen >= 0.95, as_tuple=False)[0]) \
        if bool((frozen >= 0.95).any()) else len(times) - 1
    # physically modelled bond-strength proxy: frozen area fraction times the
    # brine-controlled strength of the ice at the bond-plane temperature
    strength = torch.tensor(
        [ice_tensile_strength(float(value), salinity_ppt) for value in t_bond],
        dtype=torch.float64) * frozen
    asymptote = float(strength[-1]) if float(strength[-1]) > 0.0 else 1.0
    relative = strength / asymptote
    strength_index = int(torch.nonzero(relative >= 0.5, as_tuple=False)[0]) \
        if bool((relative >= 0.5).any()) else len(times) - 1
    comparator = conduction_time(half_thickness, thermophysics=tp)
    tau_half = float(t[strength_index])
    sensible = capacity * half_thickness * span
    stored = capacity * dx * (
        0.5 * (temperature[0] + temperature[-1])
        + sum(temperature[1:-1])) - capacity * half_thickness * float(ambient_c)
    energy_in = sensible + latent_per_area
    energy_accounted = removed + stored
    return {
        "half_thickness": float(half_thickness),
        "ambient_c": float(ambient_c),
        "initial_c": float(initial_c),
        "lens_thickness": float(lens_thickness),
        "salinity_ppt": float(salinity_ppt),
        "nodes": nodes,
        "dt": dt,
        "dx": dx,
        "steps": step,
        "time_s": t,
        "bond_temperature_c": t_bond,
        "frozen_fraction": frozen,
        "tau_half_s": tau_half,
        "tau_half_hours": tau_half / 3600.0,
        "temperature_half_hours": float(t[half_index]) / 3600.0,
        "strength_proxy": strength.tolist(),
        "t95_s": float(t[t95_index]),
        "t95_hours": float(t[t95_index]) / 3600.0,
        "conduction_time_s": comparator,
        "conduction_time_hours": comparator / 3600.0,
        "tau_half_relative_deviation": abs(tau_half - comparator) / comparator,
        "lens_freeze_time_s": float(t[frozen_index]),
        "lens_freeze_time_hours": float(t[frozen_index]) / 3600.0,
        "fundamental_mode_s": half_thickness ** 2 / (math.pi ** 2 * alpha) * 4.0,
        "energy_initial_j_per_m2": energy_in,
        "energy_removed_j_per_m2": removed,
        "energy_closure_relative_error": abs(energy_accounted - energy_in) / energy_in,
    }


# --------------------------------------------------------------------------- #
# 3. The age model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FreezeBondAgeModel:
    """Freeze-bond strength as a function of age, temperature and salinity.

    Parameters
    ----------
    temperature_c
        Contact temperature ``T_c`` of the rubble (degC).
    salinity_ppt
        Bulk salinity of the pore water (parts per thousand).
    thermal_length
        Half-thickness ``h`` of the ice mass that feeds the bond, m.  It is the
        length that enters the conduction time scale; the default equals the
        ridge-DEM particle radius used by
        :class:`tensordem.ridge_dem.RidgeMaterial`.
    lens_thickness
        Thickness of the frozen lens, m (latent-heat bookkeeping).
    area_fraction
        ``eta_b``, the frozen fraction of the projected contact area.  The one
        calibration parameter of the model (see ``LITERATURE_ANCHOR``).
    shear_ratio
        ``kappa = tau_b_inf / sigma_b_inf``.
    tau_f_seconds
        Growth time constant.  ``None`` derives it from
        :func:`lens_cooling_transient`; a value may be supplied to override it
        (and then ``tau_source`` says so).
    """

    temperature_c: float = -10.0
    salinity_ppt: float = 5.0
    thermal_length: float = 0.75
    lens_thickness: float = 0.005
    area_fraction: float = 0.0
    shear_ratio: float = 2.0
    thermophysics: IceThermophysics = DEFAULT_THERMOPHYSICS
    tau_f_seconds: float | None = None
    formation_ratio: float = 0.02

    # ---------------------------------------------------------------- checks
    def __post_init__(self) -> None:
        for name in ("temperature_c", "salinity_ppt", "thermal_length",
                     "lens_thickness", "area_fraction", "shear_ratio",
                     "formation_ratio"):
            value = getattr(self, name)
            if not isinstance(value, float) and not isinstance(value, int):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.salinity_ppt < 0.0:
            raise ValueError("salinity_ppt must be non-negative")
        if self.thermal_length <= 0.0:
            raise ValueError("thermal_length must be positive")
        if self.lens_thickness < 0.0:
            raise ValueError("lens_thickness must be non-negative")
        if self.area_fraction < 0.0 or self.area_fraction > 1.0:
            raise ValueError("area_fraction must lie in [0, 1]")
        if self.shear_ratio <= 0.0:
            raise ValueError("shear_ratio must be positive")
        if self.formation_ratio <= 0.0 or self.formation_ratio >= 1.0:
            raise ValueError("formation_ratio must lie in (0, 1)")
        if self.tau_f_seconds is not None and self.tau_f_seconds <= 0.0:
            raise ValueError("tau_f_seconds must be positive")

    # ------------------------------------------------------------ properties
    @property
    def freezing_point_c(self) -> float:
        return freezing_point_celsius(self.salinity_ppt)

    @property
    def bond_active(self) -> bool:
        """True when the contact is cold enough for a freeze bond to exist."""
        return self.temperature_c < self.freezing_point_c

    @property
    def brine_fraction(self) -> float:
        if not self.bond_active:
            return 1.0
        return brine_volume_fraction(self.temperature_c, self.salinity_ppt)

    @property
    def ice_tensile_strength_pa(self) -> float:
        if not self.bond_active:
            return 0.0
        return ice_tensile_strength(self.temperature_c, self.salinity_ppt)

    @property
    def sigma_inf_pa(self) -> float:
        """Asymptotic (fully grown) freeze-bond tensile strength, Pa."""
        return self.area_fraction * self.ice_tensile_strength_pa

    @property
    def tau_inf_pa(self) -> float:
        """Asymptotic freeze-bond shear strength, Pa."""
        return self.shear_ratio * self.sigma_inf_pa

    @property
    def cooling_solution(self) -> dict:
        """1-D cooling solution used to derive ``tau_f`` (cached per instance)."""
        cached = getattr(self, "_cooling_cache", None)
        if cached is not None:
            return cached
        solution = lens_cooling_transient(
            self.thermal_length, ambient_c=self.temperature_c,
            initial_c=self.freezing_point_c, lens_thickness=self.lens_thickness,
            salinity_ppt=self.salinity_ppt, thermophysics=self.thermophysics)
        object.__setattr__(self, "_cooling_cache", solution)
        return solution

    @property
    def tau_f_seconds_effective(self) -> float:
        if self.tau_f_seconds is not None:
            return float(self.tau_f_seconds)
        return float(self.cooling_solution["tau_half_s"])

    @property
    def tau_f_hours(self) -> float:
        return self.tau_f_seconds_effective / 3600.0

    @property
    def tau_source(self) -> str:
        return ("prescribed" if self.tau_f_seconds is not None
                else "1-D cooling half-relaxation time")

    @property
    def formation_time_seconds(self) -> float:
        return self.formation_ratio * self.tau_f_seconds_effective

    # -------------------------------------------------------------- kinetics
    def sigma_at(self, age_hours: float) -> float:
        """Freeze-bond tensile strength at ``age_hours``, Pa."""
        if age_hours < 0.0 or not math.isfinite(age_hours):
            raise ValueError("age_hours must be finite and non-negative")
        if not self.bond_active:
            return 0.0
        return self.sigma_inf_pa * (1.0 - math.exp(-age_hours * 3600.0
                                                  / self.tau_f_seconds_effective))

    def tau_at(self, age_hours: float) -> float:
        """Freeze-bond shear strength at ``age_hours``, Pa."""
        return self.shear_ratio * self.sigma_at(age_hours)

    def bond_fraction_at(self, age_hours: float, *, maximum: float = 0.65) -> float:
        """Fraction of contacts carrying a freeze bond at ``age_hours``."""
        if age_hours < 0.0 or not math.isfinite(age_hours):
            raise ValueError("age_hours must be finite and non-negative")
        if not self.bond_active or maximum <= 0.0:
            return 0.0
        if maximum > 1.0:
            raise ValueError("maximum freeze-bond fraction must be <= 1")
        growth = 1.0 - math.exp(-age_hours * 3600.0 / self.formation_time_seconds)
        return maximum * growth

    def strength_at(self, age_hours: float) -> tuple[float, float]:
        """``(sigma_b(t), tau_b(t))`` in Pa."""
        return self.sigma_at(age_hours), self.tau_at(age_hours)

    # ------------------------------------------------------------- reporting
    def as_dict(self) -> dict:
        solution = self.cooling_solution
        comparator = conduction_time(self.thermal_length,
                                     thermophysics=self.thermophysics)
        return {
            "temperature_c": self.temperature_c,
            "salinity_ppt": self.salinity_ppt,
            "freezing_point_c": self.freezing_point_c,
            "bond_active": self.bond_active,
            "brine_fraction": self.brine_fraction,
            "ice_tensile_strength_pa": self.ice_tensile_strength_pa,
            "area_fraction": self.area_fraction,
            "shear_ratio": self.shear_ratio,
            "sigma_inf_pa": self.sigma_inf_pa,
            "tau_inf_pa": self.tau_inf_pa,
            "thermal_length_m": self.thermal_length,
            "lens_thickness_m": self.lens_thickness,
            "tau_f_seconds": self.tau_f_seconds_effective,
            "tau_f_hours": self.tau_f_hours,
            "tau_f_source": self.tau_source,
            "formation_time_hours": self.formation_time_seconds / 3600.0,
            "conduction_time_hours": comparator / 3600.0,
            "tau_f_vs_conduction_relative_deviation": abs(
                self.tau_f_seconds_effective - comparator) / comparator,
            "lens_cooling": {
                key: solution[key] for key in
                ("tau_half_hours", "t95_hours", "conduction_time_hours",
                 "tau_half_relative_deviation", "lens_freeze_time_hours",
                 "fundamental_mode_s", "steps", "dt", "nodes", "dx")
            },
            "stefan_lens_time_hours": stefan_freezing_time(
                self.lens_thickness,
                self.freezing_point_c - self.temperature_c,
                thermophysics=self.thermophysics) / 3600.0
            if self.bond_active else None,
        }

    # ------------------------------------------------------------ constructors
    @classmethod
    def calibrated(cls, *, sigma_inf_target_pa: float | None = None,
                   **kwargs) -> "FreezeBondAgeModel":
        """Model whose ``eta_b`` reproduces a target strength at ``T_c, S``.

        With no target the literature anchor of :data:`LITERATURE_ANCHOR` is
        used, so the returned model reproduces that anchor *by construction*.
        """
        probe = cls(area_fraction=0.0, **kwargs)
        if not probe.bond_active:
            raise ValueError("no freeze bond forms at this temperature/salinity")
        target = (float(LITERATURE_ANCHOR["sigma_b_inf_pa"])
                  if sigma_inf_target_pa is None else float(sigma_inf_target_pa))
        if target < 0.0:
            raise ValueError("sigma_inf_target_pa must be non-negative")
        frac = target / probe.ice_tensile_strength_pa
        if frac > 1.0:
            raise ValueError("target strength exceeds the brine-controlled ice "
                             "strength: the freeze bond cannot be stronger than "
                             "solid ice")
        return cls(area_fraction=frac, **kwargs)


# --------------------------------------------------------------------------- #
# 4. Coupling to the DEM keel
# --------------------------------------------------------------------------- #
def aged_material(material: RidgeMaterial, model: FreezeBondAgeModel,
                  age_hours: float, *, maximum_fraction: float | None = None,
                  ) -> RidgeMaterial:
    """Ridge material with the freeze-bond state of ``age_hours``.

    The bond strength is replaced by the age-dependent value and the fraction
    of frozen contacts by :meth:`FreezeBondAgeModel.bond_fraction_at`.  At
    ``t = 0`` the fraction is exactly zero, so the assembly is a genuinely
    *unbonded* keel -- that is the ``R_keel(0)`` control of the campaign
    criterion.
    """
    sigma, tau = model.strength_at(age_hours)
    fraction = model.bond_fraction_at(
        age_hours, maximum=material.freeze_bond_fraction
        if maximum_fraction is None else maximum_fraction)
    if fraction <= 0.0:
        return replace(material, freeze_bond_fraction=0.0)
    return replace(material, freeze_bond_fraction=fraction,
                   bond_tensile_strength=max(sigma, 1e-6),
                   bond_shear_strength=max(tau, 1e-6))


def _smooth(values: torch.Tensor, window: int) -> torch.Tensor:
    """Centred moving average that preserves the record length."""
    if window <= 1 or values.numel() < 3:
        return values.clone()
    kernel = torch.ones(window, dtype=torch.float64) / window
    left = window // 2
    padded = torch.nn.functional.pad(values.reshape(1, 1, -1),
                                     (left, window - 1 - left), mode="replicate")
    return torch.nn.functional.conv1d(padded, kernel.reshape(1, 1, -1)).reshape(-1)


def keel_resistance(keel: RidgeKeel, material: RidgeMaterial, *,
                    press_velocity: float = 0.02, stroke: float = 0.10,
                    bond_contact_mode: str = "parallel") -> dict:
    """Peak and residual resistance of a confined keel to a rigid press.

    Apparatus (all rigid planes, penalty contact with friction):

    * two vertical side walls and a rigid bottom plane (the seabed) -- the keel
      is *confined*, so a pushed keel cannot simply float away.  A free keel
      pushed by a kinematic platen is buoyancy-controlled (the net buoyant
      force is 0.83 kN per 0.5 m disk, i.e. 45 kN per metre of ridge for this
      deck) and shows almost no freeze-bond sensitivity;
    * a rigid top press moving down at ``press_velocity`` for ``stroke``
      metres, whose line reaction is the resistance.

    The press is placed exactly in contact at ``t = 0`` (no initial
    penetration, so there is no start-up spike) and the reaction is reported as
    ``peak`` (maximum of the 5 %-window moving average -- the load that breaks
    the bonded structure), ``residual`` (mean over the last half of the stroke
    -- friction only once the bonds are broken) and ``work`` per metre of
    ridge.  ``bond_contact_mode="parallel"`` keeps the frictional capacity of a
    bonded contact, which is what makes the peak strength-controlled.
    """
    if press_velocity <= 0.0 or stroke <= 0.0:
        raise ValueError("press_velocity and stroke must be positive")
    generated = generate_keel_assembly(keel, material, tip_height=1.0)
    assembly: RidgeAssembly = generated["assembly"]
    assembly.bond_contact_mode = bond_contact_mode
    wall_stiffness = 2.0 * material.contact_stiffness
    radius = material.particle_radius
    half_width = float(assembly.pos[:, 0].max()) + radius
    bottom = float(assembly.pos[:, 1].min()) - radius
    top = float(assembly.pos[:, 1].max()) + radius
    planes = [
        Plane((1.0, 0.0), (-half_width, 0.0), stiffness=wall_stiffness,
              damping=material.contact_damping, friction=0.25),
        Plane((-1.0, 0.0), (half_width, 0.0), stiffness=wall_stiffness,
              damping=material.contact_damping, friction=0.25),
        Plane((0.0, 1.0), (0.0, bottom), stiffness=wall_stiffness,
              damping=material.contact_damping, friction=0.40),
    ]
    press = Plane((0.0, -1.0), (0.0, top), velocity=-press_velocity,
                  stiffness=wall_stiffness, damping=material.contact_damping,
                  friction=0.35)
    assembly.planes = planes + [press]
    assembly.driven = torch.zeros(assembly.count, dtype=torch.bool)
    assembly.set_platen_velocity(torch.tensor((0.0, 0.0), dtype=torch.float64))
    dt = material.recommended_dt
    steps = max(int(round(stroke / (press_velocity * dt))), 10)
    history = []
    for _ in range(steps):
        out = assembly.step(dt)
        press = replace(press, point=(0.0, press.point[1] - press_velocity * dt))
        assembly.planes = planes + [press]
        # reaction on the press: its normal points down, so a positive
        # magnitude gives a +y force on the plane
        history.append(float(out["plane_forces"][3][1]))
    force = torch.tensor(history, dtype=torch.float64)
    window = max(int(0.05 * steps), 1)
    smoothed = _smooth(force, window)
    residual = force[steps // 2:]
    return {
        "particles": generated["info"]["particles"],
        "bonds": generated["info"]["bonds"],
        "bond_contact_mode": bond_contact_mode,
        "press_velocity": press_velocity,
        "stroke": stroke,
        "steps": steps,
        "dt": dt,
        "peak_n_per_m": float(smoothed.max()),
        "peak_raw_n_per_m": float(force.max()),
        "residual_n_per_m": float(residual.mean()),
        "work_j_per_m": float(force.mean() * stroke),
        "broken_bonds": assembly.broken_bonds,
        "force_history_n_per_m": [float(value) for value in force],
        "displacement_m": [press_velocity * dt * (index + 1) for index in range(steps)],
    }


def run_keel_age_sweep(ages_hours, *, keel: RidgeKeel | None = None,
                       material: RidgeMaterial | None = None,
                       model: FreezeBondAgeModel | None = None,
                       seeds=(20261031,), press_velocity: float = 0.02,
                       stroke: float = 0.10,
                       bond_contact_mode: str = "parallel") -> dict:
    """DEM keel resistance ``R_keel`` at each age of ``ages_hours``.

    For every age the freeze-bond state of :func:`aged_material` is installed
    (strength ``sigma_b(t)``, ``tau_b(t)`` and the frozen-contact fraction) and
    :func:`keel_resistance` measures the confined-press resistance.  Several
    ``seeds`` may be given; the reported ``peak`` is their mean and the spread
    is kept so the monotonicity claim can be qualified.
    """
    keel = keel or RidgeKeel(8.0, 12.0, 0.25)
    base_material = material or RidgeMaterial(particle_radius=0.5,
                                              freeze_bond_fraction=0.65)
    if model is None:
        raise ValueError("a FreezeBondAgeModel is required")
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    records = []
    for age in ages_hours:
        runs = []
        for seed in seeds:
            aged = aged_material(replace(base_material, seed=int(seed)), model, age)
            run = keel_resistance(keel, aged, press_velocity=press_velocity,
                                  stroke=stroke, bond_contact_mode=bond_contact_mode)
            runs.append(run)
        peaks = torch.tensor([r["peak_n_per_m"] for r in runs], dtype=torch.float64)
        residuals = torch.tensor([r["residual_n_per_m"] for r in runs],
                                 dtype=torch.float64)
        works = torch.tensor([r["work_j_per_m"] for r in runs], dtype=torch.float64)
        records.append({
            "age_hours": float(age),
            "sigma_b_pa": model.sigma_at(age),
            "tau_b_pa": model.tau_at(age),
            "bond_fraction": aged.freeze_bond_fraction,
            "particles": runs[0]["particles"],
            "bonds": runs[0]["bonds"],
            "peak_force_n_per_m": float(peaks.mean()),
            "peak_force_seed_spread_n_per_m": float(peaks.std(unbiased=False))
            if len(seeds) > 1 else 0.0,
            "peak_forces_n_per_m": [float(value) for value in peaks],
            "residual_force_n_per_m": float(residuals.mean()),
            "residual_forces_n_per_m": [float(value) for value in residuals],
            "work_j_per_m": float(works.mean()),
            "broken_bonds": [r["broken_bonds"] for r in runs],
            "steps": runs[0]["steps"],
            "dt": runs[0]["dt"],
            "runs": [{"seed": int(seed), **{k: v for k, v in run.items()
                                            if k not in ("force_history_n_per_m",
                                                         "displacement_m")}}
                     for seed, run in zip(seeds, runs)],
            "history": [{"seed": int(seed),
                         "force": run["force_history_n_per_m"],
                         "displacement": run["displacement_m"]}
                        for seed, run in zip(seeds, runs)],
        })
    reference = records[0]["peak_force_n_per_m"] if records else 0.0
    for record in records:
        record["age_factor"] = (record["peak_force_n_per_m"] / reference
                                if reference else float("nan"))
    return {"keel": {"depth_m": keel.keel_depth, "width_m": keel.keel_width,
                     "porosity": keel.porosity},
            "material": {"particle_radius": base_material.particle_radius,
                         "freeze_bond_fraction_max":
                             base_material.freeze_bond_fraction,
                         "friction_coefficient": base_material.friction_coefficient},
            "apparatus": {"press_velocity": press_velocity, "stroke": stroke,
                          "bond_contact_mode": bond_contact_mode,
                          "seeds": list(seeds)},
            "records": records}


def compare_growth_time_scales(*, thermal_lengths=(0.05, 0.25, 0.5, 1.0),
                               temperature_c: float = -10.0,
                               salinity_ppt: float = 5.0,
                               lens_thickness: float = 0.005) -> dict:
    """Conduction-derived growth time vs the prescribed one of the sibling module.

    :mod:`tensordem.ridge_freeze_bond_aging` builds the same constitutive law
    with a *declared* Arrhenius growth time ``tau_g`` (its apparent activation
    energy is a parametric model, chosen so that a bond at -10 degC saturates
    in hours).  The 1-D cooling solution of this module derives the same time
    scale from the thermal mass that feeds the bond instead.  The two routes
    are compared here, which is the only place where the declared parameter is
    checked against an independent physical model.

    The comparison matters because the two routes scale differently: the
    prescribed ``tau_g`` is a material constant, while the conduction time
    grows with the **square** of the block size, so a centimetre-scale
    laboratory bond and a half-metre field block cannot share one time scale.

    Returns a dict with one record per ``thermal_length``:
    ``tau_derived_hours``, ``conduction_time_hours``, ``ratio_to_conduction``,
    and -- when the sibling module is importable -- ``tau_prescribed_hours``
    and ``ratio_prescribed_over_derived``.
    """
    try:  # the sibling module is optional: the comparison degrades to "n/a"
        from .ridge_freeze_bond_aging import SaturationFreezeBondModel
    except Exception:  # noqa: BLE001 - any import problem means "not available"
        prescribed_model = None
    else:
        prescribed_model = SaturationFreezeBondModel()
    rows = []
    for half in thermal_lengths:
        model = FreezeBondAgeModel(temperature_c=temperature_c,
                                   salinity_ppt=salinity_ppt,
                                   thermal_length=half,
                                   lens_thickness=lens_thickness)
        derived = model.tau_f_hours
        comparator = conduction_time(half) / 3600.0
        row = {
            "thermal_length_m": float(half),
            "tau_derived_hours": derived,
            "conduction_time_hours": comparator,
            "ratio_to_conduction": derived / comparator,
            "tau_prescribed_hours": None,
            "ratio_prescribed_over_derived": None,
        }
        if prescribed_model is not None:
            # the sibling module takes kelvin and percent-by-weight
            prescribed = prescribed_model.growth_time_s(
                salinity_percent=salinity_ppt / 10.0,
                temperature_K=temperature_c + 273.15) / 3600.0
            row["tau_prescribed_hours"] = prescribed
            row["ratio_prescribed_over_derived"] = prescribed / derived
        rows.append(row)
    ratios = [row["ratio_prescribed_over_derived"] for row in rows
              if row["ratio_prescribed_over_derived"] is not None]
    return {
        "temperature_c": float(temperature_c),
        "salinity_ppt": float(salinity_ppt),
        "lens_thickness_m": float(lens_thickness),
        "records": rows,
        "prescribed_over_derived": {
            "min": min(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
            "scaling": ("tau_derived ~ h^2, tau_prescribed is scale free: the "
                        "ratio grows with the block size"),
        },
    }


def age_factor(records) -> list[tuple[float, float]]:
    """``(age_hours, R_keel(t)/R_keel(0))`` pairs from a sweep record list."""
    return [(float(r["age_hours"]), float(r["age_factor"])) for r in records]


def lindqvist_age_correction(records, *, reference_keel_resistance: float = 1.0,
                             ) -> dict:
    """Fold the DEM age trend into Lindqvist's ridge keel term.

    ``R_keel(t) = R_keel^Lindqvist * k_age(t)`` with
    ``k_age(t) = R^DEM(t) / R^DEM(0)`` (dimensionless, ``k_age(0) = 1``).

    This is an *interface*: the Lindqvist ridge-keel resistance itself is
    supplied by the caller (its coefficient values are graded WITHHELD in this
    repository, see ``docs/LEVEL_ICE_RESISTANCE_DEM.md``), so no literature
    coefficient is invented here.
    """
    factors = age_factor(records)
    if not factors:
        raise ValueError("records must not be empty")
    return {
        "reference_keel_resistance_n_per_m": reference_keel_resistance,
        "k_age": [{"age_hours": age, "k_age": factor,
                   "R_keel_n_per_m": reference_keel_resistance * factor}
                  for age, factor in factors],
        "coefficient_provenance": "WITHHELD (Lindqvist 1989 not readable "
                                 "offline); k_age multiplies the caller's keel "
                                 "term and contains no literature coefficient",
    }