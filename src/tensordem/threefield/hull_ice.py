"""DEM hull-ice contact: bow penetration, contact pressure field, fragment census.

The case is the moving-wedge bow of :mod:`tensordem.dem3d_wedge` applied to a
bonded ice sheet, run as a *unit-width* bow section: the sheet spans
``ny * 2 radius`` in the transverse direction, so the resistance can be reduced to
newtons per metre of bow width and compared with the per-unit-width engineering
baselines of :mod:`tensordem.threefield.resistance` without a geometric fudge
factor.

Outputs per case:

* ``resistance_history`` -- bow reaction vs time (ice resistance),
* ``contact_pressure`` -- mean/peak normal pressure on the hull contact patch,
  the contact area and the number of particles in contact,
* ``fragments`` -- mass-conserving connected-component census of the fragmented
  ice sheet (cluster mass, equivalent diameter, size distribution),
* the calibrated DEM parameter set actually used (for the audit chain).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Sequence

import torch

from ..dem3d import DEM3DConfig
from ..dem3d_wedge import MovingWedgeBowIceDEM3D


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


class BowContactDEM(MovingWedgeBowIceDEM3D):
    """Moving-wedge bow DEM with an exposed hull contact-pressure field."""

    @torch.no_grad()
    def contact_field(self) -> dict[str, Any]:
        nodal_force, reaction, overlap = self._wedge_contact()
        active = overlap > 0.0
        contact_count = int(active.sum().item())
        area = math.pi * self.config.radius**2
        magnitude = torch.linalg.vector_norm(nodal_force, dim=1)
        pressures = magnitude[active] / area
        return {
            "reaction_N": [float(value) for value in reaction.tolist()],
            "contact_particle_count": contact_count,
            "contact_area_m2": contact_count * area,
            "mean_pressure_pa": float(pressures.mean().item()) if contact_count else 0.0,
            "max_pressure_pa": float(pressures.max().item()) if contact_count else 0.0,
            "total_normal_force_N": float(magnitude.sum().item()),
            "overlap_max_m": float(overlap.max().item()) if overlap.numel() else 0.0,
        }

    @torch.no_grad()
    def fragment_clusters(self) -> dict[str, Any]:
        """Connected components of the surviving bond network."""
        count = len(self.positions)
        parent = list(range(count))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        pairs = self.pairs[self.alive].tolist()
        for a, b in pairs:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra
        clusters: dict[int, list[int]] = {}
        for index in range(count):
            clusters.setdefault(find(index), []).append(index)
        radius = self.config.radius
        particle_mass = self.config.density * 4.0 / 3.0 * math.pi * radius**3
        masses = [len(members) * particle_mass for members in clusters.values()]
        diameters = [2.0 * radius * len(members) ** (1.0 / 3.0) for members in clusters.values()]
        order = sorted(range(len(masses)), key=lambda index: -masses[index])
        return {
            "cluster_count": len(masses),
            "particle_mass_kg": particle_mass,
            "masses_kg": [masses[index] for index in order],
            "diameters_m": [diameters[index] for index in order],
            "largest_mass_kg": masses[order[0]] if order else 0.0,
            "total_mass_kg": float(sum(masses)),
            "broken_bonds": self.broken_bonds,
            "initial_bonds": int(len(self.pairs)),
        }


@dataclass(frozen=True)
class HullIceCase:
    """Configuration of one unit-width bow/ice case."""

    thickness_m: float = 0.30
    particle_radius_m: float = 0.025
    n_length: int = 9
    n_width: int = 4
    bow_speed_m_s: float = 0.10
    wedge_angle_deg: float = 45.0
    tip_height_fraction: float = 0.5
    bond_stiffness: float = 1200.0
    contact_stiffness: float = 3500.0
    breaking_strain: float = 0.008
    shear_breaking_strain: float = 0.025
    contact_damping: float = 0.5
    drag: float = 0.1
    density_kg_m3: float = 917.0
    steps: int = 400

    def __post_init__(self) -> None:
        _finite("thickness_m", self.thickness_m, positive=True)
        _finite("particle_radius_m", self.particle_radius_m, positive=True)
        _finite("bow_speed_m_s", self.bow_speed_m_s, positive=True)
        _finite("wedge_angle_deg", self.wedge_angle_deg, positive=True)
        for name in ("n_length", "n_width", "steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if not 5.0 <= self.wedge_angle_deg <= 85.0:
            raise ValueError("wedge_angle_deg must be in [5, 85]")
        if not 0.0 <= self.tip_height_fraction <= 1.0:
            raise ValueError("tip_height_fraction must be in [0, 1]")

    @property
    def layers(self) -> int:
        """Number of particle layers that discretise the ice thickness."""
        return max(2, int(round(self.thickness_m / (2.0 * self.particle_radius_m))))

    @property
    def realised_thickness_m(self) -> float:
        return self.layers * 2.0 * self.particle_radius_m

    @property
    def model_width_m(self) -> float:
        return self.n_width * 2.0 * self.particle_radius_m

    def to_dem_config(self) -> DEM3DConfig:
        base = DEM3DConfig(
            nx=self.n_length,
            ny=self.n_width,
            nz=self.layers,
            radius=self.particle_radius_m,
            density=self.density_kg_m3,
            bond_stiffness=self.bond_stiffness,
            contact_stiffness=self.contact_stiffness,
            breaking_strain=self.breaking_strain,
            shear_breaking_strain=self.shear_breaking_strain,
            contact_damping=self.contact_damping,
            drag=self.drag,
            tool_speed=self.bow_speed_m_s,
            fix_x_edges=False,
            fix_bottom=False,
        )
        return DEM3DConfig(**{**base.__dict__, "dt": base.recommended_dt})


def run_hull_ice_case(case: HullIceCase) -> dict[str, Any]:
    """Run one unit-width bow/ice case and return the diagnostics bundle."""
    config = case.to_dem_config()
    sim = BowContactDEM(
        config,
        bow_speed=case.bow_speed_m_s,
        wedge_angle_deg=case.wedge_angle_deg,
        initial_gap=0.0,
        tip_height_fraction=case.tip_height_fraction,
        bow_half_width_m=None,
    )
    history: list[dict[str, float]] = []
    first_fracture: float | None = None
    for step in range(case.steps + 1):
        if step:
            sim.step()
        diag = sim.diagnostics()
        contact = sim.contact_field()
        resistance = max(0.0, -float(diag["bow_reaction_x_N"]))
        if first_fracture is None and int(diag["broken_bonds"]) > 0:
            first_fracture = float(diag["time"])
        history.append(
            {
                "step": step,
                "time_s": float(diag["time"]),
                "bow_tip_x_m": float(diag["bow_tip_x_m"]),
                "ice_resistance_N": resistance,
                "contact_particles": float(contact["contact_particle_count"]),
                "contact_area_m2": float(contact["contact_area_m2"]),
                "mean_contact_pressure_pa": float(contact["mean_pressure_pa"]),
                "max_contact_pressure_pa": float(contact["max_pressure_pa"]),
                "broken_bonds": float(diag["broken_bonds"]),
                "kinetic_energy_J": float(diag["kinetic_energy_J"]),
            }
        )
        if not all(math.isfinite(value) for value in history[-1].values()):
            raise FloatingPointError(f"non-finite hull-ice history at step {step}")
    fragments = sim.fragment_clusters()
    # Sample the crushing branch once fracture has started, so the mean is not
    # diluted by the elastic ramp-up before first contact failure.
    start = 0
    if first_fracture is not None:
        start = min(
            range(len(history)),
            key=lambda index: abs(history[index]["time_s"] - first_fracture),
        )
    steady = [row["ice_resistance_N"] for row in history[start:]]
    if not steady:
        steady = [row["ice_resistance_N"] for row in history]
    peak = max(row["ice_resistance_N"] for row in history)
    mean_resistance = sum(steady) / len(steady)
    # Crushing phase: from first contact failure to the breakthrough collapse.
    peak_index = max(range(len(history)), key=lambda index: history[index]["ice_resistance_N"])
    end_index = len(history) - 1
    for index in range(peak_index, len(history)):
        if history[index]["ice_resistance_N"] < 0.1 * peak:
            end_index = index
            break
    crushing_rows = [
        row["ice_resistance_N"] for row in history[start:end_index + 1]
    ] or steady
    crushing_resistance = sum(crushing_rows) / len(crushing_rows)
    contact_rows = [row for row in history if row["contact_particles"] > 0.0]
    mean_pressure = (
        sum(row["mean_contact_pressure_pa"] for row in contact_rows) / len(contact_rows)
        if contact_rows
        else 0.0
    )
    return {
        "case": {
            "thickness_m": case.thickness_m,
            "realised_thickness_m": case.realised_thickness_m,
            "layers": case.layers,
            "model_width_m": case.model_width_m,
            "particle_radius_m": case.particle_radius_m,
            "particle_count": int(len(sim.positions)),
            "bow_speed_m_s": case.bow_speed_m_s,
            "wedge_angle_deg": case.wedge_angle_deg,
            "dt_s": config.dt,
            "steps": case.steps,
            "simulated_time_s": float(sim.time),
            "bow_travel_m": float(case.bow_speed_m_s * sim.time),
        },
        "history": history,
        "mean_ice_resistance_N": mean_resistance,
        "crushing_phase_resistance_N": crushing_resistance,
        "crushing_phase_resistance_per_width_N_m": crushing_resistance / case.model_width_m,
        "crushing_phase_duration_s": float(history[end_index]["time_s"] - history[start]["time_s"]),
        "peak_ice_resistance_N": peak,
        "resistance_per_width_N_m": mean_resistance / case.model_width_m,
        "peak_resistance_per_width_N_m": peak / case.model_width_m,
        "mean_contact_pressure_pa": mean_pressure,
        "max_contact_pressure_pa": max(row["max_contact_pressure_pa"] for row in history),
        "first_fracture_time_s": first_fracture,
        "fragments": fragments,
        "dem_parameters": {
            "bond_stiffness": case.bond_stiffness,
            "contact_stiffness": case.contact_stiffness,
            "breaking_strain": case.breaking_strain,
            "shear_breaking_strain": case.shear_breaking_strain,
            "contact_damping": case.contact_damping,
            "density_kg_m3": case.density_kg_m3,
        },
    }


def run_thickness_sweep(
    thicknesses: Iterable[float], base: HullIceCase | None = None
) -> dict[str, Any]:
    """Run the unit-width bow/ice case at several ice thicknesses."""
    base = base or HullIceCase()
    cases: list[dict[str, Any]] = []
    for thickness in thicknesses:
        case = HullIceCase(**{**base.__dict__, "thickness_m": float(thickness)})
        cases.append(run_hull_ice_case(case))
    return {
        "case_count": len(cases),
        "thicknesses_m": [case["case"]["thickness_m"] for case in cases],
        "resistance_per_width_N_m": [case["resistance_per_width_N_m"] for case in cases],
        "crushing_phase_resistance_per_width_N_m": [
            case["crushing_phase_resistance_per_width_N_m"] for case in cases
        ],
        "peak_resistance_per_width_N_m": [
            case["peak_resistance_per_width_N_m"] for case in cases
        ],
        "mean_contact_pressure_pa": [case["mean_contact_pressure_pa"] for case in cases],
        "fragment_counts": [case["fragments"]["cluster_count"] for case in cases],
        "cases": cases,
    }