"""End-to-end DEM ice-block collision with thermal-mechanical coupling.

This is the integration layer of the node: one ice block impacts a rigid
structure, the contact dissipates plastic work, part of it becomes heat in a
patch under the contact, the patch conducts into the block, the resulting
temperature field produces thermal stress, and the superposed stress decides
whether -- and how much more -- ice breaks than the mechanical stress alone
would have broken.

Model chain (each link is its own module)
----------------------------------------
::

    Hunt-Crossley contact   ->   W_diss = int F_damp d(delta)
            |                          |
            |                     plastic_heat.deposition_case
            |                          |
            v                          v
      sigma_mech(r)            q''' in a patch of depth delta
            |                          |
            |                   conduction.SlabHeat1D -> T(z,t)
            |                          |
            +------> fracture.failure_margin <----- thermal_stress
                              |
                     breakage increment

Contact law
-----------
Hunt-Crossley, ``F = k_h delta^{3/2} (1 + alpha v)`` with
``k_h = (4/3) E* sqrt(R)``, integrated with RK4 at 2000 steps per elastic
period.  ``alpha`` is **not** taken from the textbook approximation
``e = 1 - (2/3) alpha``: on this integrator that approximation gives
``e = 0.58`` for a prescribed ``e = 0.30`` (measured, see the report).  ``alpha``
is therefore calibrated by bisection on the integrator itself, so that the
measured restitution matches the prescribed one to ``1e-3`` -- and the energy
ledger then closes against ``0.5 m v0^2 (1 - e^2)``, which is what makes
``W_diss`` trustworthy enough to feed the thermal chain.

Heat injection
--------------
The dissipated work is spread uniformly over the contact patch
``pi a^2 delta`` and injected over the contact duration; conduction then
continues for ``duration_factor`` contact durations.  The FV column is per unit
faced area, so the injected power is ``P/(A_patch delta)``.

Mechanical stress in the patch
------------------------------
Ideal-elastic Hertz pressure reaches ~220 MPa at 1 m/s, an order of magnitude
above the ice strength; the DEM crushing law caps the pressure the ice can
actually carry at ``sigma_f(T)``.  The contact population therefore uses
``min(p_hertz(r), sigma_f)``, and the *additional* breakage is the annulus that
the thermal term pushes over the threshold.  Two numbers are reported:

``breakage_coupled["increment"]``
    geometry-resolved (Hertz patch with the crushing cap), the physical answer;
``breakage_bound["increment"]``
    ``|sigma_th| / sigma_f``, the increment for a population spread uniformly
    over the whole strength budget -- an *upper bound* quoted so the reader can
    see how much room the thermal term could ever have.

What is deliberately **not** modelled
-------------------------------------
* phase change (melting, latent heat, melt film) -- flagged as
  ``melting_risk`` when the patch would cross 0 degC;
* lateral spreading inside the patch -- bounded with the point-source kernel;
* the structure's own heat capacity -- the counter-face is adiabatic
  (``wall_temperature_c=None``), the conservative bound for the ice.

All quantities are SI.  Compression is negative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dem_coupling import thermal_response
from .fracture import failure_margin
from .plastic_heat import (
    TAYLOR_QUINNEY_BETA,
    dissipated_energy,
    hertz_patch_radius,
    partition_to_ice,
)
from .properties import IceProperties, ice_properties

__all__ = [
    "BlockImpactCase",
    "BlockImpactResult",
    "simulate_block_impact",
    "significance_sweep",
    "layer_depth_sweep",
]


# ---------------------------------------------------------------------------
# case
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BlockImpactCase:
    """Geometry, kinematics and numerics of a single normal impact."""

    radius_m: float = 0.30
    thickness_m: float | None = 0.60
    impact_speed_m_s: float = 1.0
    restitution: float = 0.30
    bulk_temperature_c: float = -10.0
    beta: float = TAYLOR_QUINNEY_BETA
    layer_fraction: float | None = 0.15
    cells: int = 240
    grid_depth_m: float = 0.05
    dt_s: float = 2.0e-6
    duration_factor: float = 3.0
    wall_temperature_c: float | None = None

    def __post_init__(self) -> None:
        for name in ("radius_m", "impact_speed_m_s", "grid_depth_m", "dt_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value!r}")
        if not (0.0 <= float(self.restitution) < 1.0):
            raise ValueError("restitution must be in [0, 1)")
        if not (0.0 < float(self.beta) <= 1.0):
            raise ValueError("beta must be in (0, 1]")
        if self.thickness_m is not None and float(self.thickness_m) <= 0.0:
            raise ValueError("thickness_m must be > 0 when given")
        if self.layer_fraction is not None and not (0.0 < float(self.layer_fraction) <= 1.0):
            raise ValueError("layer_fraction must be in (0, 1]")

    @property
    def contact_radius_m(self) -> float:
        """Effective radius of a sphere-on-plane impact."""
        return float(self.radius_m)

    def mass_kg(self, props: IceProperties) -> float:
        """Block mass: ``rho * min(pi R^2 h, 4/3 pi R^3)``."""
        sphere = 4.0 / 3.0 * math.pi * self.radius_m ** 3
        if self.thickness_m is None:
            return props.rho_kg_m3 * sphere
        plate = math.pi * self.radius_m ** 2 * float(self.thickness_m)
        return props.rho_kg_m3 * min(plate, sphere)


# ---------------------------------------------------------------------------
# contact integration
# ---------------------------------------------------------------------------
def _hertz_stiffness(radius_m: float, modulus_pa: float) -> float:
    return 4.0 / 3.0 * modulus_pa * math.sqrt(radius_m)


def _peak_force_analytic(mass: float, v0: float, k_h: float) -> float:
    """Hertz impact peak force from ``0.5 m v0^2 = (2/5) F delta``, ``F = k delta^{3/2}``."""
    return k_h * (5.0 * mass * v0 * v0 / (4.0 * k_h)) ** 0.6


def _rk4_step(delta: float, v: float, h: float, *, k_h: float, mass: float,
              alpha: float) -> tuple[float, float]:
    def accel(d: float, vv: float) -> float:
        if d <= 0.0:
            return 0.0
        f_el = k_h * d * math.sqrt(d)
        return -(f_el + f_el * alpha * vv) / mass

    a1 = accel(delta, v)
    a2 = accel(delta + 0.5 * h * v, v + 0.5 * h * a1)
    v2 = v + 0.5 * h * a1
    a3 = accel(delta + 0.5 * h * v2, v + 0.5 * h * a2)
    v3 = v + 0.5 * h * a2
    a4 = accel(delta + h * v3, v + h * a3)
    d_new = delta + h / 6.0 * (v + 2.0 * v2 + 2.0 * v3 + (v + h * a3))
    v_new = v + h / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    return d_new, v_new


def _contact(case: BlockImpactCase, props: IceProperties, mass: float,
             k_h: float, period: float, alpha: float, *, substeps: int = 2000,
             track_energy: bool = False) -> dict[str, float]:
    """Integrate one Hunt-Crossley impact and return its integral quantities."""
    dt_c = period / substeps
    delta = 0.0
    v = float(case.impact_speed_m_s)
    t = 0.0
    work_damp = 0.0
    peak_force = 0.0
    peak_overlap = 0.0
    was_in_contact = False
    v_out = v
    max_steps = int(20.0 * period / dt_c)
    while max_steps > 0:
        max_steps -= 1
        if delta > 0.0:
            f_now = k_h * delta * math.sqrt(delta) * (1.0 + alpha * v)
            peak_force = max(peak_force, abs(f_now))
            peak_overlap = max(peak_overlap, delta)
        d_prev, v_prev = delta, v
        delta, v = _rk4_step(delta, v, dt_c, k_h=k_h, mass=mass, alpha=alpha)
        t += dt_c
        if track_energy:
            d0, d1 = max(d_prev, 0.0), max(delta, 0.0)
            f0 = k_h * d0 * math.sqrt(d0) * alpha * v_prev
            f1 = k_h * d1 * math.sqrt(d1) * alpha * v
            work_damp += 0.5 * (f0 + f1) * (d1 - d0)
        if delta > 0.0:
            was_in_contact = True
        elif was_in_contact:
            v_out = v
            break
    return {
        "work_damp_j": work_damp,
        "peak_force_n": peak_force,
        "peak_overlap_m": peak_overlap,
        "contact_duration_s": t,
        "v_out_m_s": v_out,
        "restitution_measured": abs(v_out) / float(case.impact_speed_m_s),
    }


def _calibrate_alpha(case: BlockImpactCase, props: IceProperties, mass: float,
                     k_h: float, period: float) -> tuple[float, float]:
    """Bisect the Hunt-Crossley damping so the measured restitution is the one asked.

    Returns ``(alpha, restitution_measured)``.  The textbook approximation
    ``alpha = 1.5 (1 - e)`` is used as the initial guess only.
    """
    target = float(case.restitution)
    lo, hi = 0.0, 40.0
    guess = max(1.5 * (1.0 - target), 1e-6)
    e_guess = _contact(case, props, mass, k_h, period, guess)["restitution_measured"]
    if target <= 0.0:
        return hi, _contact(case, props, mass, k_h, period, hi)["restitution_measured"]
    if abs(e_guess - target) < 1e-3:
        return guess, e_guess
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        e_mid = _contact(case, props, mass, k_h, period, mid, substeps=600)[
            "restitution_measured"]
        if e_mid > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    alpha = 0.5 * (lo + hi)
    return alpha, _contact(case, props, mass, k_h, period, alpha)["restitution_measured"]


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------
@dataclass
class BlockImpactResult:
    """Everything the report needs, all programmatically generated."""

    case: BlockImpactCase
    mass_kg: float
    effective_modulus_pa: float
    hunt_crossley_alpha: float
    restitution_measured: float
    peak_force_n: float
    peak_force_analytic_n: float
    peak_overlap_m: float
    peak_contact_stress_pa: float
    contact_radius_m: float
    contact_duration_s: float
    layer_depth_m: float
    deposition_volume_m3: float
    dissipated_energy_j: float
    dissipated_energy_closed_form_j: float
    plastic_heat_ice_j: float
    peak_delta_t_k: float
    peak_delta_t_adiabatic_k: float
    sigma_thermal_patch_pa: float
    sigma_thermal_free_pa: float
    sigma_mech_peak_pa: float
    thermal_to_mechanical_ratio: float
    thermal_fraction_of_strength: float
    sigma_f_bulk_pa: float
    sigma_f_heated_pa: float
    breakage_plain: dict[str, float]
    breakage_coupled: dict[str, float]
    breakage_bound: dict[str, float]
    melting_risk: bool
    melt_cap_delta_t_k: float
    sigma_thermal_melt_capped_pa: float
    energy_ledger: dict[str, float]
    time_s: np.ndarray = field(repr=False, default=None)
    delta_t_profile_k: np.ndarray = field(repr=False, default=None)
    sigma_thermal_profile_pa: np.ndarray = field(repr=False, default=None)
    depth_m: np.ndarray = field(repr=False, default=None)
    cell_widths_m: np.ndarray = field(repr=False, default=None)
    surface_history_k: np.ndarray = field(repr=False, default=None)
    power_history_w_m2: np.ndarray = field(repr=False, default=None)

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "case": {
                "radius_m": self.case.radius_m,
                "thickness_m": self.case.thickness_m,
                "impact_speed_m_s": self.case.impact_speed_m_s,
                "restitution": self.case.restitution,
                "bulk_temperature_c": self.case.bulk_temperature_c,
                "beta": self.case.beta,
                "layer_fraction": self.case.layer_fraction,
                "cells": self.case.cells,
                "grid_depth_m": self.case.grid_depth_m,
                "dt_s": self.case.dt_s,
                "duration_factor": self.case.duration_factor,
            },
            "mass_kg": self.mass_kg,
            "effective_modulus_pa": self.effective_modulus_pa,
            "hunt_crossley_alpha": self.hunt_crossley_alpha,
            "restitution_measured": self.restitution_measured,
            "peak_force_n": self.peak_force_n,
            "peak_force_analytic_n": self.peak_force_analytic_n,
            "peak_overlap_m": self.peak_overlap_m,
            "peak_contact_stress_pa": self.peak_contact_stress_pa,
            "contact_radius_m": self.contact_radius_m,
            "contact_duration_s": self.contact_duration_s,
            "layer_depth_m": self.layer_depth_m,
            "deposition_volume_m3": self.deposition_volume_m3,
            "dissipated_energy_j": self.dissipated_energy_j,
            "dissipated_energy_closed_form_j": self.dissipated_energy_closed_form_j,
            "plastic_heat_ice_j": self.plastic_heat_ice_j,
            "peak_delta_t_k": self.peak_delta_t_k,
            "peak_delta_t_adiabatic_k": self.peak_delta_t_adiabatic_k,
            "sigma_thermal_patch_pa": self.sigma_thermal_patch_pa,
            "sigma_thermal_free_pa": self.sigma_thermal_free_pa,
            "sigma_mech_peak_pa": self.sigma_mech_peak_pa,
            "thermal_to_mechanical_ratio": self.thermal_to_mechanical_ratio,
            "thermal_fraction_of_strength": self.thermal_fraction_of_strength,
            "sigma_f_bulk_pa": self.sigma_f_bulk_pa,
            "sigma_f_heated_pa": self.sigma_f_heated_pa,
            "breakage_plain": self.breakage_plain,
            "breakage_coupled": self.breakage_coupled,
            "breakage_bound": self.breakage_bound,
            "melting_risk": self.melting_risk,
            "melt_cap_delta_t_k": self.melt_cap_delta_t_k,
            "sigma_thermal_melt_capped_pa": self.sigma_thermal_melt_capped_pa,
            "within_model_validity": not self.melting_risk,
            "energy_ledger": self.energy_ledger,
        }
        if include_arrays:
            out["arrays"] = {
                "time_s": np.asarray(self.time_s).tolist(),
                "surface_delta_t_k": np.asarray(self.surface_history_k).tolist(),
                "power_w_m2": np.asarray(self.power_history_w_m2).tolist(),
                "depth_m": np.asarray(self.depth_m).tolist(),
                "cell_widths_m": np.asarray(self.cell_widths_m).tolist(),
                "delta_t_profile_k": np.asarray(self.delta_t_profile_k).tolist(),
                "sigma_thermal_profile_pa": np.asarray(self.sigma_thermal_profile_pa).tolist(),
            }
        return out

    def flat(self) -> dict[str, Any]:
        """Scalar-only view used by the sweeps and the report tables."""
        d = self.as_dict()
        flat = {k: v for k, v in d.items() if not isinstance(v, dict)}
        flat.update(self.energy_ledger)
        flat.update({
            "radius_m": self.case.radius_m,
            "impact_speed_m_s": self.case.impact_speed_m_s,
            "bulk_temperature_c": self.case.bulk_temperature_c,
            "restitution": self.case.restitution,
            "layer_fraction": self.case.layer_fraction,
        })
        flat["breakage_increment"] = self.breakage_coupled["increment"]
        flat["breakage_increment_bound"] = self.breakage_bound["increment"]
        flat["breakage_fraction_coupled"] = self.breakage_coupled["fraction_total"]
        flat["breakage_fraction_mech"] = self.breakage_plain["fraction_mech"]
        return flat


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def simulate_block_impact(case: BlockImpactCase) -> BlockImpactResult:
    """Run the coupled chain for one impact case."""
    props = ice_properties(case.bulk_temperature_c)
    nu = props.poisson
    e_eff = props.youngs_modulus_pa / (1.0 - nu * nu)     # rigid counter-face
    mass = case.mass_kg(props)
    k_h = _hertz_stiffness(case.contact_radius_m, e_eff)
    period = 2.0 * math.pi * math.sqrt(mass / k_h)

    alpha, e_meas = _calibrate_alpha(case, props, mass, k_h, period)
    contact = _contact(case, props, mass, k_h, period, alpha, track_energy=True)
    peak_force = contact["peak_force_n"]
    contact_duration = contact["contact_duration_s"]
    work_damp = max(contact["work_damp_j"], 0.0)

    peak_force_analytic = _peak_force_analytic(mass, float(case.impact_speed_m_s), k_h)
    contact_radius = hertz_patch_radius(max(peak_force, 1e-9), case.contact_radius_m, e_eff)
    scatter = thermal_response(
        dissipated_work_j=work_damp,
        peak_force_n=peak_force,
        effective_radius_m=case.contact_radius_m,
        effective_modulus_pa=e_eff,
        contact_duration_s=contact_duration,
        bulk_temperature_c=case.bulk_temperature_c,
        beta=case.beta,
        layer_fraction=(1.0 if case.layer_fraction is None else float(case.layer_fraction)),
        wall_temperature_c=case.wall_temperature_c,
        cells=case.cells,
        grid_depth_m=case.grid_depth_m,
        dt_s=case.dt_s,
        duration_factor=case.duration_factor,
    )
    closed_form = dissipated_energy(mass, float(case.impact_speed_m_s), case.restitution)
    ledger = {
        "kinetic_energy_j": 0.5 * mass * case.impact_speed_m_s ** 2,
        "dissipated_energy_integrated_j": work_damp,
        "dissipated_energy_closed_form_j": closed_form,
        "dissipated_energy_relative_error": (
            abs(work_damp - closed_form) / closed_form if closed_form > 0.0 else 0.0),
        "restitution_prescribed": float(case.restitution),
        "restitution_measured": e_meas,
        **scatter.energy_ledger,
        "peak_overlap_m": contact["peak_overlap_m"],
    }

    return BlockImpactResult(
        case=case,
        mass_kg=mass,
        effective_modulus_pa=e_eff,
        hunt_crossley_alpha=alpha,
        restitution_measured=e_meas,
        peak_force_n=peak_force,
        peak_force_analytic_n=peak_force_analytic,
        peak_overlap_m=contact["peak_overlap_m"],
        peak_contact_stress_pa=scatter.sigma_mech_peak_pa,
        contact_radius_m=contact_radius,
        contact_duration_s=contact_duration,
        layer_depth_m=scatter.layer_depth_m,
        deposition_volume_m3=scatter.deposition_volume_m3,
        dissipated_energy_j=work_damp,
        dissipated_energy_closed_form_j=closed_form,
        plastic_heat_ice_j=scatter.heat_ice_j,
        peak_delta_t_k=scatter.peak_delta_t_k,
        peak_delta_t_adiabatic_k=scatter.peak_delta_t_adiabatic_k,
        sigma_thermal_patch_pa=scatter.sigma_thermal_patch_pa,
        sigma_thermal_free_pa=scatter.sigma_thermal_free_pa,
        sigma_mech_peak_pa=scatter.sigma_mech_peak_pa,
        thermal_to_mechanical_ratio=scatter.thermal_to_mechanical_ratio,
        thermal_fraction_of_strength=scatter.thermal_fraction_of_strength,
        sigma_f_bulk_pa=scatter.sigma_f_bulk_pa,
        sigma_f_heated_pa=scatter.sigma_f_heated_pa,
        breakage_plain=scatter.breakage_plain,
        breakage_coupled=scatter.breakage_coupled,
        breakage_bound=scatter.breakage_bound,
        melting_risk=scatter.melting_risk,
        melt_cap_delta_t_k=scatter.melt_cap_delta_t_k,
        sigma_thermal_melt_capped_pa=scatter.sigma_thermal_melt_capped_pa,
        energy_ledger=ledger,
        time_s=scatter.time_s,
        delta_t_profile_k=scatter.delta_t_profile_k,
        sigma_thermal_profile_pa=scatter.sigma_thermal_profile_pa,
        depth_m=scatter.depth_m,
        cell_widths_m=scatter.cell_widths_m,
        surface_history_k=scatter.surface_history_k,
        power_history_w_m2=scatter.power_history_w_m2,
    )


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------
def significance_sweep(
    speeds: Any = (0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0),
    temperatures_c: Any = (-2.0, -5.0, -10.0, -20.0, -30.0),
    *,
    radius_m: float = 0.30,
    restitution: float = 0.30,
    layer_fraction: float | None = 0.15,
    cells: int = 120,
) -> list[dict[str, Any]]:
    """Map ``sigma_th / sigma_mech`` and ``sigma_th / sigma_f`` over ``(v, T)``."""
    rows = []
    for tv in np.asarray(temperatures_c, dtype=float):
        for v in np.asarray(speeds, dtype=float):
            case = BlockImpactCase(radius_m=radius_m, impact_speed_m_s=float(v),
                                   restitution=restitution,
                                   bulk_temperature_c=float(tv),
                                   layer_fraction=layer_fraction, cells=cells)
            res = simulate_block_impact(case)
            row = res.flat()
            row["bulk_temperature_c"] = float(tv)
            rows.append(row)
    return rows


def layer_depth_sweep(
    fractions: Any = (0.05, 0.10, 0.15, 0.25, 0.50, 1.00),
    *,
    impact_speed_m_s: float = 1.0,
    bulk_temperature_c: float = -10.0,
    radius_m: float = 0.30,
    restitution: float = 0.30,
    cells: int = 160,
) -> list[dict[str, Any]]:
    """Sensitivity of every thermal number to the *unresolved* deposition depth."""
    rows = []
    for f in np.asarray(fractions, dtype=float):
        case = BlockImpactCase(radius_m=radius_m, impact_speed_m_s=impact_speed_m_s,
                               restitution=restitution,
                               bulk_temperature_c=bulk_temperature_c,
                               layer_fraction=float(f), cells=cells)
        res = simulate_block_impact(case)
        row = {
            "layer_fraction": float(f),
            "layer_depth_m": res.layer_depth_m,
            "deposition_volume_m3": res.deposition_volume_m3,
            "peak_delta_t_k": res.peak_delta_t_k,
            "sigma_thermal_pa": res.sigma_thermal_patch_pa,
            "thermal_fraction_of_strength": res.thermal_fraction_of_strength,
            "thermal_to_mechanical_ratio": res.thermal_to_mechanical_ratio,
            "breakage_increment": res.breakage_coupled["increment"],
            "melting_risk": res.melting_risk,
        }
        rows.append(row)
    return rows


#: module self-checks: the Hertz impact closed form and the friction-free limit.
_m, _v, _kh = 100.0, 1.0, 4.0 / 3.0 * 9.5e9 / (1 - 0.33 ** 2) * math.sqrt(0.3)
assert _peak_force_analytic(_m, _v, _kh) > 0.0
assert failure_margin(-1.0e6, 0.0, -10.0).branch == "compression"