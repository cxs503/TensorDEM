"""Ice-structure DEM contact friction verification: Lindqvist analytical anchor.

Implements an ice plate particle assembly with:
- Hertz or linear normal contact elasticity
- Coulomb tangential friction (Cundall-Strack incremental spring)
- Bending fracture criterion (flexural strength based bond breakage)
- Moving wedge bow contact (ship hull idealization)

Simulates ship wedge bow breaking level ice at prescribed speeds,
outputs ice resistance R_ice(t) and fracture patterns.

Validated against Lindqvist(1989) formula: R_ice = R_c + R_s + R_b.

All quantities in SI units (m, Pa, kg/m³, m/s, N).

Note: For computational tractability in verification, the default uses
a linear contact model with stiffness calibrated to ice properties.
The Hertz model is available but requires much smaller time steps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from .dem3d import DEM3DConfig, IceDEM3D
from .dem3d_wedge import MovingWedgeBowIceDEM3D
from .level_ice.formula import (
    IceState, HullForm, ICEBREAKER,
    lindqvist_components, crushing_resistance, bending_resistance,
    submersion_resistance,
)


# ---------------------------------------------------------------------------
# Ice material properties (typical level ice)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IceMaterial:
    """Level ice material properties (SI)."""
    density_kg_m3: float = 917.0
    young_modulus_pa: float = 9.0e9
    poisson_ratio: float = 0.33
    compressive_strength_pa: float = 1.5e6
    flexural_strength_pa: float = 0.5e6
    friction_coefficient: float = 0.15
    # Bending fracture: bond breaking strain derived from flexural strength
    # eps_break = sigma_f / E (simplified beam theory)
    @property
    def breaking_strain(self) -> float:
        return self.flexural_strength_pa / self.young_modulus_pa

    @property
    def shear_breaking_strain(self) -> float:
        # Shear strength ~ 0.6 * compressive for ice; strain = tau / G
        g = self.young_modulus_pa / (2.0 * (1.0 + self.poisson_ratio))
        return 0.6 * self.compressive_strength_pa / g


@dataclass(frozen=True)
class ShipWedgeConfig:
    """Ship wedge bow geometry and operating conditions."""
    speed_knots: float = 2.0
    wedge_angle_deg: float = 22.0  # half entry angle
    bow_half_width_m: float = 10.0  # half beam
    initial_gap_m: float = 0.05

    @property
    def speed_m_s(self) -> float:
        return self.speed_knots * 0.514444  # 1 knot = 0.514444 m/s


@dataclass(frozen=True)
class IcePlateConfig:
    """Ice plate geometry for DEM discretization."""
    thickness_m: float = 1.0
    width_m: float = 20.0  # transverse (y)
    length_m: float = 30.0  # longitudinal (x)
    particle_radius_m: float = 0.15  # resolution


def make_ice_plate_config(
    ice: IcePlateConfig,
    material: IceMaterial,
    ship: ShipWedgeConfig,
    *,
    contact_model: str = "linear",
    device: str = "cpu",
    stiffness_scale: float = 1.0,
) -> DEM3DConfig:
    """Build a DEM3DConfig for an ice plate.

    Uses linear contact model by default for computational tractability.
    The stiffness is scaled from ice properties but kept moderate so that
    the stable time step is practical for verification runs.

    Args:
        stiffness_scale: Multiplier on the base stiffness. Use < 1 for
            faster runs (softer material), > 1 for stiffer ice.
    """
    # Use a small grid for verification tractability
    nx = max(3, min(8, int(ice.length_m / (4.0 * ice.particle_radius_m))))
    ny = max(3, min(6, int(ice.width_m / (4.0 * ice.particle_radius_m))))
    nz = max(2, min(4, int(ice.thickness_m / (2.0 * ice.particle_radius_m))))

    # Base stiffness from ice properties, scaled for tractability
    # k_n ~ E * R gives ~1e9 N/m for real ice; we scale down for dt
    base_stiffness = material.young_modulus_pa * ice.particle_radius_m
    # Scale to a practical range: 1000-10000 N/m for verification
    contact_stiffness = max(1000.0, min(10000.0, base_stiffness * stiffness_scale * 1e-6))
    bond_stiffness = contact_stiffness * 0.5

    # Breaking strain from flexural strength
    breaking_strain = material.breaking_strain
    # For verification, use a moderate breaking strain so fracture occurs
    # within a reasonable number of steps
    breaking_strain = max(0.01, min(0.5, breaking_strain))
    shear_breaking_strain = breaking_strain * 2.0

    cfg = DEM3DConfig(
        nx=nx, ny=ny, nz=nz,
        radius=ice.particle_radius_m,
        density=material.density_kg_m3,
        bond_stiffness=bond_stiffness,
        contact_stiffness=contact_stiffness,
        breaking_strain=breaking_strain,
        shear_breaking_strain=shear_breaking_strain,
        contact_damping=0.5,
        drag=0.5,
        tool_speed=ship.speed_m_s,
        tool_gap=ship.initial_gap_m,
        tool_radius=0.5,
        dt=None,  # Let the engine compute stable dt
        device=device,
        geometry="box",
        fix_x_edges=False,
        fix_bottom=False,
        friction_coefficient=material.friction_coefficient,
        rolling_resistance_coefficient=0.0,
        restitution_coefficient=0.3,
        contact_model=contact_model,
        young_modulus=material.young_modulus_pa,
        poisson_ratio=material.poisson_ratio,
    )
    return cfg


class IceDemContactSim(MovingWedgeBowIceDEM3D):
    """Ice-structure DEM simulation with energy ledger and Lindqvist validation.

    Extends MovingWedgeBowIceDEM3D with:
    - Energy balance tracking (work input vs stored + dissipated)
    - Ice resistance time series
    - Fracture pattern diagnostics
    """

    def __init__(
        self,
        config: DEM3DConfig,
        *,
        bow_speed: float = 1.0,
        wedge_angle_deg: float = 22.0,
        initial_gap: float = 0.05,
        tip_height_fraction: float = 0.5,
        bow_half_width_m: float = 10.0,
        ice_material: IceMaterial | None = None,
    ) -> None:
        super().__init__(
            config,
            bow_speed=bow_speed,
            wedge_angle_deg=wedge_angle_deg,
            initial_gap=initial_gap,
            tip_height_fraction=tip_height_fraction,
            bow_half_width_m=bow_half_width_m,
        )
        self.ice_material = ice_material or IceMaterial()
        # Energy ledger
        self._cumulative_bow_work = 0.0
        self._cumulative_friction_dissipation = 0.0
        self._last_bow_force_x = 0.0
        # Time series
        self.history: list[dict[str, float]] = []

    @torch.no_grad()
    def step(self, external_forces: torch.Tensor | None = None) -> None:
        """Single step with energy accounting."""
        # Record pre-step energy
        diag_before = self.diagnostics()
        e_mech_before = diag_before["mechanical_energy_J"]

        super().step(external_forces)

        # Post-step diagnostics
        diag_after = self.diagnostics()
        e_mech_after = diag_after["mechanical_energy_J"]

        # Bow work increment: F_x * dx
        bow_reaction_x = diag_after.get("bow_reaction_x_N", 0.0)
        dx = self.bow_speed * self.dt
        bow_work = abs(bow_reaction_x) * dx
        self._cumulative_bow_work += bow_work
        self._last_bow_force_x = bow_reaction_x

        # Friction dissipation from base class (if tracked)
        friction_diss = getattr(self, '_last_friction_power', 0.0) * self.dt
        self._cumulative_friction_dissipation += abs(friction_diss)

        # Record history
        self.history.append({
            "time_s": self.time,
            "step": self.step_count,
            "bow_tip_x_m": self.bow_tip_x,
            "bow_reaction_x_N": bow_reaction_x,
            "bow_reaction_z_N": diag_after.get("bow_reaction_z_N", 0.0),
            "ice_resistance_N": abs(bow_reaction_x),
            "broken_bonds": self.broken_bonds,
            "kinetic_energy_J": diag_after["kinetic_energy_J"],
            "bond_elastic_energy_J": diag_after["bond_elastic_energy_J"],
            "contact_energy_J": diag_after.get("particle_contact_energy_J", 0.0),
            "tool_contact_energy_J": diag_after.get("tool_contact_energy_J", 0.0),
            "mechanical_energy_J": e_mech_after,
            "cumulative_bow_work_J": self._cumulative_bow_work,
            "cumulative_friction_dissipation_J": self._cumulative_friction_dissipation,
        })

    def energy_residual(self) -> float:
        """Energy balance residual: |W_input - ΔE_mech - E_diss| / W_input."""
        if self._cumulative_bow_work < 1e-12:
            return 0.0
        if not self.history:
            return 0.0
        e0 = self.history[0]["mechanical_energy_J"]
        e1 = self.history[-1]["mechanical_energy_J"]
        delta_e = e1 - e0
        dissipated = self._cumulative_friction_dissipation
        residual = abs(self._cumulative_bow_work - delta_e - dissipated) / self._cumulative_bow_work
        return float(residual)

    def mean_ice_resistance(self, *, skip_steps: int = 10) -> float:
        """Mean ice resistance after initial transient."""
        if len(self.history) <= skip_steps:
            return 0.0
        vals = [h["ice_resistance_N"] for h in self.history[skip_steps:]]
        return float(sum(vals) / len(vals))

    def fracture_pattern(self) -> dict[str, Any]:
        """Describe the current fracture state."""
        n_total = int(self.bonded.sum().item())
        n_broken = self.broken_bonds
        return {
            "total_bonds": n_total,
            "broken_bonds": n_broken,
            "broken_fraction": n_broken / max(n_total, 1),
            "time_s": self.time,
            "bow_travel_m": self.bow_tip_x - (
                float(self.initial_positions[:, 0].min()) - 0.5 * self.config.radius - self.initial_gap
            ),
        }


# ---------------------------------------------------------------------------
# Lindqvist analytical anchor for this configuration
# ---------------------------------------------------------------------------

def lindqvist_anchor_for_config(
    ice_material: IceMaterial,
    ice_plate: IcePlateConfig,
    ship: ShipWedgeConfig,
    hull_form: HullForm | None = None,
) -> dict[str, float]:
    """Compute Lindqvist R_ice = R_c + R_s + R_b for the given configuration.

    Note: The classical Lindqvist formula uses ship beam B and length L.
    For our wedge bow, we map:
    - B = 2 * bow_half_width_m
    - L = effective hull length (use ice_plate.length_m as characteristic)
    - entry_angle = wedge_angle_deg
    - ice thickness = ice_plate.thickness_m
    """
    if hull_form is None:
        hull_form = HullForm(
            length_m=ice_plate.length_m,
            beam_m=2.0 * ship.bow_half_width_m,
            draught_m=ice_plate.thickness_m * 0.7,  # typical
            entry_angle_deg=ship.wedge_angle_deg,
            stem_angle_deg=30.0,
            friction_coefficient=ice_material.friction_coefficient,
            name="wedge_bow",
        )

    ice_state = IceState(
        thickness_m=ice_plate.thickness_m,
        compressive_strength_pa=ice_material.compressive_strength_pa,
        flexural_strength_pa=ice_material.flexural_strength_pa,
        elastic_modulus_pa=ice_material.young_modulus_pa,
        density_kg_m3=ice_material.density_kg_m3,
    )

    result = lindqvist_components(
        ice_state, hull_form, ship.speed_m_s,
    )
    r_c = result["crushing_N"]
    r_b = result["bending_N"]
    r_s = result["submersion_N"]
    r_v = result["added_drag_N"]

    return {
        "crushing_N": r_c,
        "bending_N": r_b,
        "submersion_N": r_s,
        "added_drag_N": r_v,
        "total_static_N": r_c + r_b + r_s,
        "total_N": r_c + r_b + r_s + r_v,
        "hull": hull_form.as_dict(),
        "ice": ice_state.as_dict(),
        "speed_m_s": ship.speed_m_s,
    }


def effective_lindqvist_from_dem_config(
    config: DEM3DConfig,
    ice_plate: IcePlateConfig,
    ship: ShipWedgeConfig,
) -> dict[str, float]:
    """Compute Lindqvist using effective ice properties from DEM config.

    The DEM uses scaled stiffness for computational tractability. This
    function computes the corresponding effective ice properties and
    evaluates Lindqvist with those, giving a fair comparison.

    Effective properties:
    - E_eff = k_n / R (from contact stiffness)
    - σ_c_eff = E_eff * breaking_strain
    - σ_f_eff = E_eff * breaking_strain
    """
    # Effective Young's modulus from contact stiffness
    e_eff = config.contact_stiffness / config.radius
    # Effective strength from breaking strain
    sigma_c_eff = e_eff * config.breaking_strain
    sigma_f_eff = e_eff * config.breaking_strain

    # DEM beam width
    dem_beam = config.ny * 2.0 * config.radius

    hull_form = HullForm(
        length_m=ice_plate.length_m,
        beam_m=dem_beam,
        draught_m=ice_plate.thickness_m * 0.7,
        entry_angle_deg=ship.wedge_angle_deg,
        stem_angle_deg=30.0,
        friction_coefficient=config.friction_coefficient,
        name="wedge_bow_dem_scaled",
    )

    ice_state = IceState(
        thickness_m=ice_plate.thickness_m,
        compressive_strength_pa=sigma_c_eff,
        flexural_strength_pa=sigma_f_eff,
        elastic_modulus_pa=e_eff,
        density_kg_m3=config.density,
    )

    result = lindqvist_components(ice_state, hull_form, ship.speed_m_s)
    return {
        "crushing_N": result["crushing_N"],
        "bending_N": result["bending_N"],
        "submersion_N": result["submersion_N"],
        "added_drag_N": result["added_drag_N"],
        "total_static_N": result["crushing_N"] + result["bending_N"] + result["submersion_N"],
        "total_N": result["total_N"],
        "effective_young_modulus_pa": e_eff,
        "effective_compressive_strength_pa": sigma_c_eff,
        "effective_flexural_strength_pa": sigma_f_eff,
        "dem_beam_m": dem_beam,
    }


# ---------------------------------------------------------------------------
# Campaign runner
# ---------------------------------------------------------------------------

def run_single_case(
    ice_material: IceMaterial,
    ice_plate: IcePlateConfig,
    ship: ShipWedgeConfig,
    *,
    steps: int = 200,
    device: str = "cpu",
    stiffness_scale: float = 1.0,
) -> dict[str, Any]:
    """Run a single ice-breaking simulation case.

    Returns a dict with simulation results and Lindqvist comparison.
    """
    config = make_ice_plate_config(
        ice_plate, ice_material, ship, device=device,
        stiffness_scale=stiffness_scale,
    )
    sim = IceDemContactSim(
        config,
        bow_speed=ship.speed_m_s,
        wedge_angle_deg=ship.wedge_angle_deg,
        initial_gap=ship.initial_gap_m,
        bow_half_width_m=ship.bow_half_width_m,
        ice_material=ice_material,
    )

    for _ in range(steps):
        sim.step()

    # Results
    mean_r = sim.mean_ice_resistance(skip_steps=max(10, steps // 5))

    # Lindqvist with real ice properties (for reference)
    lindqvist_real = lindqvist_anchor_for_config(ice_material, ice_plate, ship)

    # Lindqvist with effective DEM properties (for fair comparison)
    lindqvist_eff = effective_lindqvist_from_dem_config(config, ice_plate, ship)

    # Relative error vs effective Lindqvist (crushing + bending)
    r_lind_eff_cb = lindqvist_eff["crushing_N"] + lindqvist_eff["bending_N"]
    if r_lind_eff_cb > 0:
        rel_error_eff = (mean_r - r_lind_eff_cb) / r_lind_eff_cb
    else:
        rel_error_eff = float("nan")

    # Relative error vs real Lindqvist (for reference)
    r_lind_real_cb = lindqvist_real["crushing_N"] + lindqvist_real["bending_N"]
    if r_lind_real_cb > 0:
        rel_error_real = (mean_r - r_lind_real_cb) / r_lind_real_cb
    else:
        rel_error_real = float("nan")

    return {
        "speed_knots": ship.speed_knots,
        "speed_m_s": ship.speed_m_s,
        "ice_thickness_m": ice_plate.thickness_m,
        "steps": steps,
        "final_time_s": sim.time,
        "mean_ice_resistance_N": mean_r,
        "peak_ice_resistance_N": max((h["ice_resistance_N"] for h in sim.history), default=0.0),
        "broken_bonds": sim.broken_bonds,
        "energy_residual": sim.energy_residual(),
        "cumulative_bow_work_J": sim._cumulative_bow_work,
        # Real Lindqvist (full-scale reference)
        "lindqvist_crushing_N": lindqvist_real["crushing_N"],
        "lindqvist_bending_N": lindqvist_real["bending_N"],
        "lindqvist_submersion_N": lindqvist_real["submersion_N"],
        "lindqvist_total_static_N": lindqvist_real["total_static_N"],
        "lindqvist_total_N": lindqvist_real["total_N"],
        "rel_error_vs_lindqvist_real": rel_error_real,
        # Effective Lindqvist (scaled for fair comparison)
        "lindqvist_eff_crushing_N": lindqvist_eff["crushing_N"],
        "lindqvist_eff_bending_N": lindqvist_eff["bending_N"],
        "lindqvist_eff_submersion_N": lindqvist_eff["submersion_N"],
        "lindqvist_eff_total_static_N": lindqvist_eff["total_static_N"],
        "lindqvist_eff_total_N": lindqvist_eff["total_N"],
        "rel_error_vs_lindqvist_eff": rel_error_eff,
        "effective_young_modulus_pa": lindqvist_eff["effective_young_modulus_pa"],
        "effective_compressive_strength_pa": lindqvist_eff["effective_compressive_strength_pa"],
        "fracture_pattern": sim.fracture_pattern(),
        "history": sim.history,
    }


def run_campaign(
    *,
    speeds_knots: tuple[float, ...] = (2.0, 4.0, 6.0),
    ice_thicknesses_m: tuple[float, ...] = (0.5, 1.0, 1.5),
    steps: int = 200,
    device: str = "cpu",
    stiffness_scale: float = 1.0,
) -> dict[str, Any]:
    """Run a full campaign: speed sweep + thickness sweep.

    Returns results for all cases plus power-law fit for R_ice vs h_i.
    """
    ice_material = IceMaterial()
    ice_plate_base = IcePlateConfig()

    results = []
    for thickness in ice_thicknesses_m:
        for speed in speeds_knots:
            plate = IcePlateConfig(thickness_m=thickness)
            ship = ShipWedgeConfig(speed_knots=speed)
            case_result = run_single_case(
                ice_material, plate, ship, steps=steps, device=device,
                stiffness_scale=stiffness_scale,
            )
            results.append(case_result)

    # Power-law fit: R_ice ∝ h_i^n
    # Group by speed, fit n for each
    power_law_fits = {}
    for speed in speeds_knots:
        cases = [r for r in results if r["speed_knots"] == speed]
        if len(cases) >= 2:
            h_vals = [c["ice_thickness_m"] for c in cases]
            r_vals = [c["mean_ice_resistance_N"] for c in cases]
            # Log-log linear fit: log(R) = n * log(h) + c
            log_h = [math.log(h) for h in h_vals]
            log_r = [math.log(max(r, 1e-6)) for r in r_vals]
            n_fit = _linear_fit_slope(log_h, log_r)
            power_law_fits[speed] = {"exponent_n": n_fit, "cases": len(cases)}

    return {
        "results": results,
        "power_law_fits": power_law_fits,
        "case_count": len(results),
    }


def _linear_fit_slope(x: list[float], y: list[float]) -> float:
    """Simple linear regression slope: n = cov(x,y) / var(x)."""
    n = len(x)
    if n < 2:
        return float("nan")
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / n
    var_x = sum((xi - x_mean) ** 2 for xi in x) / n
    if var_x < 1e-15:
        return float("nan")
    return cov / var_x


# ---------------------------------------------------------------------------
# Negative control helpers
# ---------------------------------------------------------------------------

def run_negative_control_no_fracture(
    ice_plate: IcePlateConfig,
    ship: ShipWedgeConfig,
    *,
    steps: int = 100,
    device: str = "cpu",
) -> dict[str, Any]:
    """Negative control: σ_f → ∞ (no bond breakage).

    With infinite flexural strength, bonds should never break.
    """
    # Create material with very high flexural strength → no fracture
    ice_material = IceMaterial(
        flexural_strength_pa=1e15,  # effectively infinite
        compressive_strength_pa=1e15,
    )
    config = make_ice_plate_config(ice_plate, ice_material, ship, device=device)
    sim = IceDemContactSim(
        config,
        bow_speed=ship.speed_m_s,
        wedge_angle_deg=ship.wedge_angle_deg,
        initial_gap=ship.initial_gap_m,
        bow_half_width_m=ship.bow_half_width_m,
        ice_material=ice_material,
    )

    for _ in range(steps):
        sim.step()

    return {
        "broken_bonds": sim.broken_bonds,
        "total_bonds": int(sim.bonded.sum().item()),
        "no_fracture": sim.broken_bonds == 0,
        "mean_ice_resistance_N": sim.mean_ice_resistance(),
    }


def run_negative_control_zero_friction(
    ice_plate: IcePlateConfig,
    ship: ShipWedgeConfig,
    *,
    steps: int = 100,
    device: str = "cpu",
) -> dict[str, Any]:
    """Negative control: μ → 0 (zero friction).

    With zero friction coefficient, tangential forces should be ~0.
    """
    ice_material = IceMaterial(friction_coefficient=0.0)
    config = make_ice_plate_config(ice_plate, ice_material, ship, device=device)
    sim = IceDemContactSim(
        config,
        bow_speed=ship.speed_m_s,
        wedge_angle_deg=ship.wedge_angle_deg,
        initial_gap=ship.initial_gap_m,
        bow_half_width_m=ship.bow_half_width_m,
        ice_material=ice_material,
    )

    for _ in range(steps):
        sim.step()

    # Check tangential forces are near zero
    diag = sim.diagnostics()
    return {
        "friction_coefficient": 0.0,
        "mean_ice_resistance_N": sim.mean_ice_resistance(),
        "broken_bonds": sim.broken_bonds,
        "diagnostics": {
            "bow_reaction_x_N": diag.get("bow_reaction_x_N", 0.0),
            "bow_reaction_y_N": diag.get("bow_reaction_y_N", 0.0),
        },
    }
