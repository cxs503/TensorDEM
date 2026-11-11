"""Bridge between a DEM impact run and the thermal chain.

Why this module exists
----------------------
:mod:`tensordem.ice_thermal_stress.collision` integrates its *own* Hunt-Crossley
law and then feeds the resulting dissipated work into the thermal chain.  That is
the right tool for the benchmark and the sweeps, but it is not the library's
contact solver.  A user who runs :class:`tensordem.dem3d.IceDEM3D` gets a
*different* number for the dissipated energy -- the solver integrates the actual
contact model, including its ``clamp_min(0)`` normal-force truncation, its
friction model and however many contacts are simultaneously open.

This module is the **single conversion** from "work the solver actually
dissipated" to "thermal state of the ice":

``thermal_response(...)``
    work -> heat -> FV conduction -> thermal stress -> superposed fracture.
    Extracted verbatim from the collision driver so there is exactly one
    implementation of the physics in the package (the collision driver calls it
    too, which is why its 41 tests guard this code path as well).

``DEMThermalCoupler``
    the solver-facing object: a DEM loop calls :meth:`DEMThermalCoupler.add_dissipation`
    with the dissipated work of each contact step and asks for
    :meth:`DEMThermalCoupler.state` once the contact has closed.

``couple_impact_result(...)``
    convenience wrapper: convert a :class:`tensordem.contact_impact.PairImpactResult`
    (a real ``IceDEM3D`` pair impact) into the same thermal state.

Scope and honesty notes
-----------------------
* Only the *normal* (Hertzian) patch is heated.  Frictional heating on the
  tangential branch is the subject of :mod:`tensordem.ice_friction_thermo` and is
  not added here; for pure normal impact it is zero.
* The dissipated work is bookkeeping from the solver, its conversion factor is
  the Taylor-Quinney ``beta`` (0.9, with a declared band) times the Blok
  effusivity split ``lambda`` (1.0 for an adiabatic counter-face).
* No latent heat: a patch that would cross 0 degC is flagged ``melting_risk``
  and its thermal stress is additionally reported capped at ``dT = |T_bulk|``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .conduction import SlabHeat1D
from .fracture import strength_budget_bound, thermal_breakage_increment
from .plastic_heat import (
    TAYLOR_QUINNEY_BETA,
    hertz_patch_radius,
    partition_to_ice,
)
from .properties import IceProperties, compressive_strength, ice_properties
from .thermal_stress import restrained_gradient_stress, restrained_uniform_stress

__all__ = [
    "ThermalState",
    "thermal_response",
    "DEMThermalCoupler",
    "couple_impact_result",
]

#: Seed of the contact-population Monte-Carlo inside the fracture layer.  Kept
#: here (not in :mod:`collision`) so the analytic and the DEM-driven paths draw
#: the *same* population and their breakage increments are directly comparable.
POPULATION_SEED = 20261110
N_CONTACTS = 1024


@dataclass
class ThermalState:
    """Thermal answer for one closed contact."""

    heat_ice_j: float
    layer_depth_m: float
    deposition_volume_m3: float
    peak_delta_t_k: float
    peak_delta_t_adiabatic_k: float
    sigma_thermal_patch_pa: float
    sigma_thermal_free_pa: float
    sigma_thermal_profile_pa: np.ndarray = field(repr=False, default=None)
    sigma_mech_peak_pa: float = 0.0
    thermal_to_mechanical_ratio: float = 0.0
    thermal_fraction_of_strength: float = 0.0
    sigma_f_bulk_pa: float = 0.0
    sigma_f_heated_pa: float = 0.0
    breakage_plain: dict[str, float] = field(default_factory=dict)
    breakage_coupled: dict[str, float] = field(default_factory=dict)
    breakage_bound: dict[str, float] = field(default_factory=dict)
    melting_risk: bool = False
    melt_cap_delta_t_k: float = 0.0
    sigma_thermal_melt_capped_pa: float = 0.0
    energy_ledger: dict[str, float] = field(default_factory=dict)
    time_s: np.ndarray = field(repr=False, default=None)
    surface_history_k: np.ndarray = field(repr=False, default=None)
    power_history_w_m2: np.ndarray = field(repr=False, default=None)
    delta_t_profile_k: np.ndarray = field(repr=False, default=None)
    depth_m: np.ndarray = field(repr=False, default=None)
    cell_widths_m: np.ndarray = field(repr=False, default=None)

    def flat(self) -> dict[str, Any]:
        """Scalar-only view (JSON/report friendly)."""
        out = {
            k: v for k, v in self.__dict__.items()
            if not isinstance(v, np.ndarray) and not isinstance(v, dict)
        }
        out.update(self.energy_ledger)
        out["breakage_increment"] = self.breakage_coupled.get("increment")
        out["breakage_increment_bound"] = self.breakage_bound.get("increment")
        out["breakage_fraction_coupled"] = self.breakage_coupled.get("fraction_total")
        out["breakage_fraction_mech"] = self.breakage_plain.get("fraction_mech")
        return out


def thermal_response(
    *,
    dissipated_work_j: float,
    peak_force_n: float,
    effective_radius_m: float,
    effective_modulus_pa: float,
    contact_duration_s: float,
    bulk_temperature_c: float,
    patch_radius_m: float | None = None,
    beta: float = TAYLOR_QUINNEY_BETA,
    layer_fraction: float = 0.15,
    wall_temperature_c: float | None = None,
    cells: int = 240,
    grid_depth_m: float = 0.05,
    dt_s: float = 2.0e-6,
    duration_factor: float = 3.0,
    layer_depth_m: float | None = None,
) -> ThermalState:
    """Convert dissipated contact work into the thermal state of the ice.

    Parameters
    ----------
    dissipated_work_j
        Work dissipated by the contact during the collision, as *delivered by the
        solver* (in the analytic driver: ``int F_damp ddelta``; in the DEM bridge:
        the kinetic-energy loss of the impact pair).
    peak_force_n, effective_radius_m, effective_modulus_pa
        Contact data at peak load.  The Hertz patch radius is formed here from
        ``a = (3 F R / (4 E*))^(1/3)`` so callers cannot accidentally pass the
        *body* radius where the *radius of curvature* belongs.
    contact_duration_s
        Time the contact lasted; forms the Fourier number of the deposition
        layer, i.e. it is what makes the adiabatic-patch assumption testable.
    layer_fraction
        Deposition depth as a fraction of the Hertz patch radius (0.15 here,
        varied in the sensitivity band).
    """
    work = max(float(dissipated_work_j), 0.0)
    props: IceProperties = ice_properties(bulk_temperature_c)
    lam = partition_to_ice(
        props, None if wall_temperature_c is None else ice_properties(wall_temperature_c))
    heat_ice = lam * float(beta) * work

    if patch_radius_m is None:
        contact_radius = float(hertz_patch_radius(
            max(float(peak_force_n), 1e-12), float(effective_radius_m),
            float(effective_modulus_pa)))
    else:
        # the caller measured the contact patch (geometric Hertz relation
        # ``a = sqrt(delta (2R* - delta))`` from the solver's own overlap)
        contact_radius = float(patch_radius_m)
    layer_depth = (contact_radius if layer_fraction is None
                   else float(layer_fraction) * contact_radius)
    if layer_depth_m is not None:
        layer_depth = float(layer_depth_m)
    deposition_volume = math.pi * contact_radius ** 2 * layer_depth
    contact_area = math.pi * contact_radius ** 2

    # -- conduction --------------------------------------------------
    grid = SlabHeat1D(cells=cells, thickness_m=grid_depth_m,
                      properties=props, theta=0.5, grading=50.0)
    power_per_area = heat_ice / (float(contact_duration_s) * contact_area)
    q_vol = grid.source_profile(power_per_area, layer_depth)
    n_steps = max(int(round(float(contact_duration_s) / dt_s)), 8)
    dt = float(contact_duration_s) / n_steps
    n_tot = max(int(round(duration_factor * float(contact_duration_s) / dt)), n_steps)
    time_hist = np.empty(n_tot)
    surface_hist = np.empty(n_tot)
    power_hist = np.zeros(n_tot)
    for i in range(n_tot):
        active = i < n_steps
        grid.step(dt, q_vol=q_vol if active else None)
        time_hist[i] = (i + 1) * dt
        surface_hist[i] = grid.t[0] - grid.t_initial
        power_hist[i] = power_per_area if active else 0.0

    delta_t_profile = grid.t - grid.t_initial
    peak_delta_t = float(np.max(delta_t_profile))
    peak_delta_t_adiabatic = heat_ice / (
        props.volumetric_heat_capacity_j_m3_k * deposition_volume)
    sigma_patch = float(restrained_uniform_stress(peak_delta_t, props))
    sigma_free = 0.5 * sigma_patch
    sigma_field = restrained_gradient_stress(delta_t_profile, props, dz=grid.dx)
    sigma_mech_peak = -float(peak_force_n) / contact_area

    # -- fracture ----------------------------------------------------
    sigma_f_bulk = float(compressive_strength(bulk_temperature_c)) * 1.0e6
    heated_t = min(bulk_temperature_c + peak_delta_t, -0.05)
    sigma_f_heated = float(compressive_strength(heated_t)) * 1.0e6

    rng = np.random.default_rng(POPULATION_SEED)
    frac_r = np.sqrt(rng.random(N_CONTACTS))            # area-weighted radius
    p_ratio = np.sqrt(np.clip(1.0 - frac_r ** 2, 0.0, 1.0))
    sigma_pop = -np.minimum(abs(sigma_mech_peak) * p_ratio, sigma_f_bulk)
    breakage_plain = thermal_breakage_increment(sigma_pop, 0.0, bulk_temperature_c)
    breakage_coupled = thermal_breakage_increment(
        sigma_pop, sigma_patch, bulk_temperature_c, delta_t_k=peak_delta_t)
    bound = strength_budget_bound(sigma_patch, bulk_temperature_c, delta_t_k=peak_delta_t)
    breakage_bound = {
        "increment": bound,
        "fraction_mech": 0.0,
        "fraction_total": bound,
        "note": "uniform population over the whole strength budget; upper bound",
    }

    melt_cap = abs(float(bulk_temperature_c))
    stored_block = float(np.sum(grid.capacity * delta_t_profile)) * contact_area
    ledger = {
        "plastic_heat_ice_j": heat_ice,
        "block_enthalpy_j": stored_block,
        "heat_to_enthalpy_relative_error": (
            abs(stored_block - heat_ice) / heat_ice if heat_ice > 0.0 else 0.0),
        "effusivity_partition_lambda": lam,
        "taylor_quinney_beta": float(beta),
        "fourier_number_layer": (
            props.diffusivity_m2_s * float(contact_duration_s) / layer_depth ** 2),
    }

    return ThermalState(
        heat_ice_j=heat_ice,
        layer_depth_m=layer_depth,
        deposition_volume_m3=deposition_volume,
        peak_delta_t_k=peak_delta_t,
        peak_delta_t_adiabatic_k=peak_delta_t_adiabatic,
        sigma_thermal_patch_pa=sigma_patch,
        sigma_thermal_free_pa=sigma_free,
        sigma_thermal_profile_pa=sigma_field,
        sigma_mech_peak_pa=sigma_mech_peak,
        thermal_to_mechanical_ratio=abs(sigma_patch) / abs(sigma_mech_peak),
        thermal_fraction_of_strength=abs(sigma_patch) / sigma_f_bulk,
        sigma_f_bulk_pa=sigma_f_bulk,
        sigma_f_heated_pa=sigma_f_heated,
        breakage_plain=breakage_plain,
        breakage_coupled=breakage_coupled,
        breakage_bound=breakage_bound,
        melting_risk=bool(bulk_temperature_c + peak_delta_t > -0.05),
        melt_cap_delta_t_k=melt_cap,
        sigma_thermal_melt_capped_pa=float(restrained_uniform_stress(melt_cap, props)),
        energy_ledger=ledger,
        time_s=time_hist,
        surface_history_k=surface_hist,
        power_history_w_m2=power_hist,
        delta_t_profile_k=delta_t_profile,
        depth_m=grid.x,
        cell_widths_m=grid.dx,
    )


class DEMThermalCoupler:
    """Solver-facing accumulator: dissipated work in, thermal state out.

    The DEM loop is expected to call :meth:`add_dissipation` once per contact
    step (or once per contact, with the total), then :meth:`state` after the
    contact closes::

        coupler = DEMThermalCoupler(mass_kg=m, bulk_temperature_c=-10.0)
        for step in impact:
            coupler.add_dissipation(step.dissipated_work_j)
        state = coupler.state(peak_force_n=F_max,
                              effective_radius_m=R,
                              effective_modulus_pa=e_star,
                              contact_duration_s=t_c)

    Nothing is stored per particle: the thermal patch of one closed contact is
    the resolved scale here.  Multi-contact averaging belongs to the caller and
    is deliberately *not* hidden inside this class.
    """

    def __init__(
        self,
        *,
        mass_kg: float,
        bulk_temperature_c: float = -10.0,
        beta: float = TAYLOR_QUINNEY_BETA,
        layer_fraction: float = 0.15,
        wall_temperature_c: float | None = None,
        cells: int = 240,
        grid_depth_m: float = 0.05,
        dt_s: float = 2.0e-6,
        duration_factor: float = 3.0,
    ) -> None:
        if not math.isfinite(mass_kg) or mass_kg <= 0.0:
            raise ValueError("mass_kg must be finite and positive")
        self.mass_kg = float(mass_kg)
        self.bulk_temperature_c = float(bulk_temperature_c)
        self.beta = float(beta)
        self.layer_fraction = float(layer_fraction)
        self.wall_temperature_c = wall_temperature_c
        self.cells = int(cells)
        self.grid_depth_m = float(grid_depth_m)
        self.dt_s = float(dt_s)
        self.duration_factor = float(duration_factor)
        self._work_j = 0.0
        self._n_samples = 0

    # -- accounting ---------------------------------------------------
    def add_dissipation(self, work_j: float) -> float:
        """Accumulate dissipated work; returns the running total."""
        w = float(work_j)
        if not math.isfinite(w):
            raise ValueError("dissipated work must be finite")
        if w < 0.0:
            raise ValueError("dissipated work cannot be negative")
        self._work_j += w
        self._n_samples += 1
        return self._work_j

    @property
    def dissipated_work_j(self) -> float:
        return self._work_j

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def reset(self) -> None:
        self._work_j = 0.0
        self._n_samples = 0

    # -- conversion ---------------------------------------------------
    def state(
        self,
        *,
        peak_force_n: float,
        effective_radius_m: float,
        effective_modulus_pa: float,
        contact_duration_s: float,
        patch_radius_m: float | None = None,
    ) -> ThermalState:
        """Thermal state of the accumulated work (does not reset the ledger)."""
        return thermal_response(
            dissipated_work_j=self._work_j,
            peak_force_n=peak_force_n,
            effective_radius_m=effective_radius_m,
            effective_modulus_pa=effective_modulus_pa,
            contact_duration_s=contact_duration_s,
            patch_radius_m=patch_radius_m,
            bulk_temperature_c=self.bulk_temperature_c,
            beta=self.beta,
            layer_fraction=self.layer_fraction,
            wall_temperature_c=self.wall_temperature_c,
            cells=self.cells,
            grid_depth_m=self.grid_depth_m,
            dt_s=self.dt_s,
            duration_factor=self.duration_factor,
        )


def couple_impact_result(
    result: Any,
    *,
    bulk_temperature_c: float = -10.0,
    effective_radius_m: float | None = None,
    effective_modulus_pa: float | None = None,
    peak_overlap_m: float | None = None,
    **kwargs: Any,
) -> ThermalState:
    """Thermal state of a :class:`tensordem.contact_impact.PairImpactResult`.

    ``result`` must expose ``impact_speed``, ``rebound_speed``, ``pair_mass_kg``,
    ``peak_force`` and ``contact_time_force``; those are exactly the fields the
    ``IceDEM3D`` impact harness measures, so the conversion here is
    solver-measured work -> thermal chain, with no analytic shortcut.

    The contact patch radius is taken from the *measured* overlap
    (``a = sqrt(delta (2R* - delta))``) when ``peak_overlap_m`` is given, and
    from the Hertz formula with ``effective_modulus_pa`` otherwise.  The overlap
    route is the consistent one for a solver whose contact law is linear: the
    Hertz formula would combine the solver's soft spring force with the
    material's stiffness and collapse the heat flux into a far too small patch
    (a 30x temperature over-prediction was measured before this branch existed).
    """
    e_measured = abs(float(result.rebound_speed)) / float(result.impact_speed)
    work = float(result.dissipated_energy)
    if effective_radius_m is None or effective_modulus_pa is None:
        raise ValueError("effective_radius_m and effective_modulus_pa are required")
    patch = None
    if peak_overlap_m is not None:
        delta = min(max(float(peak_overlap_m), 0.0), 2.0 * float(effective_radius_m))
        patch = math.sqrt(max(delta * (2.0 * float(effective_radius_m) - delta), 0.0))
    coupler = DEMThermalCoupler(mass_kg=float(result.pair_mass_kg),
                                bulk_temperature_c=bulk_temperature_c, **kwargs)
    coupler.add_dissipation(work)
    state = coupler.state(peak_force_n=float(result.peak_force),
                          effective_radius_m=float(effective_radius_m),
                          effective_modulus_pa=float(effective_modulus_pa),
                          contact_duration_s=float(result.contact_time_force),
                          patch_radius_m=patch)
    state.energy_ledger.update({
        "dissipated_work_j": work,
        # pair kinetic energy before/after: m v0^2 (1 - e^2) for two equal masses
        "kinetic_energy_lost_closed_form_j": (
            float(result.pair_mass_kg) * float(result.impact_speed) ** 2
            * (1.0 - e_measured ** 2)),
        "restitution_measured": e_measured,
        "solver_dt_s": float(result.dt),
    })
    return state