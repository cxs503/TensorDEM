"""Capped/frictional wedge-bow DEM and the 3-D contact-to-resistance driver.

The driver runs the existing 3-D bonded-particle ice model
(:mod:`tensordem.dem3d_wedge`) with four additions that are needed before a
contact-force decomposition can be interpreted at all, and exposes the
per-contact records the closure layer consumes:

1. **Plastic crushing cap.**  The penalty normal law ``k_n d - c v_n`` is capped
   at ``p_crush * chord(d) * w_trib`` with ``chord(d) = 2 sqrt(2 r d - d^2)`` the
   circle chord at penetration ``d``.  Without the cap the contact force is set
   by an arbitrary penalty stiffness; with it the force is set by the physical
   contact strength, which is what makes the magnitude comparable with an
   engineering formula at all.  ``p_crush`` is the *mean* pressure of a crushing
   wedge; the triangular-profile mean ``sigma_c / 2`` of
   ``docs/threefield/FORMULAS.md`` is used by default so DEM and Lindqvist share
   the same crushing assumption (this is a consistency choice, stated as such --
   it is not an independent validation).
2. **Coulomb friction on the hull face.**  ``|F_t| <= mu N`` opposing the
   tangential slip of the ice relative to the hull, ``tanh``-regularised in slip
   speed so the direction is defined at zero slip.  Lindqvist's
   ``tan(alpha) + mu / cos(alpha)`` bracket needs this term to be present in the
   DEM.
3. **Buoyancy / gravity.**  The base solver already carries Archimedes buoyancy
   against a free surface (``DEM3DConfig.water_level``) plus gravity; the case
   builder sets the waterline so that the lattice floats at
   ``f_sub = rho_ice / rho_water``.  This is what makes detached ice cost work to
   push down, i.e. it is the mechanism behind the submersion component.
4. **Bond-graph connectivity.**  ``cluster_labels()`` labels the surviving bond
   components; ice not connected to the largest (held) cluster is *detached*,
   which is the classifier input for the submersion mechanism.

Every addition is opt-in through this module: the base classes are untouched, so
no existing solver trajectory changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from ..dem3d import DEM3DConfig
from ..dem3d_wedge import MovingWedgeBowIceDEM3D
from .mechanisms import (
    HullContactSet,
    MechanismDecomposition,
    average_decompositions,
    decompose,
    plate_characteristic_length,
    flexural_demand_ratio,
)

ICE_DENSITY_KG_M3 = 917.0
WATER_DENSITY_KG_M3 = 1025.0
GRAVITY_M_S2 = 9.80665


def cap_engagement_stiffness(crushing_pressure_pa: float, particle_radius_m: float,
                             engagement_fraction: float = 0.1) -> float:
    """Penalty stiffness at which the plastic cap engages at ``fraction * r``.

    The cap ``p * 2 r * chord(d)`` with ``chord(d) = 2 sqrt(2 r d - d^2)`` is
    sub-linear in penetration while the penalty law ``k d`` is linear, so the two
    cross where ``k d = 4 p r sqrt(2 r d - d^2)``.  Requiring the crossing at
    ``d = fraction * r`` gives, exactly,

        ``k = 4 p r sqrt(2 f - f^2) / f``      (``f`` = engagement fraction).
    This is a *numerical* parameter choice, documented and checked by a stiffness
    sweep: above ``k`` the reported resistance is cap controlled and therefore
    insensitive to ``k``.
    """
    for name, value in (("crushing_pressure_pa", crushing_pressure_pa),
                        ("particle_radius_m", particle_radius_m),
                        ("engagement_fraction", engagement_fraction)):
        _check(name, value, positive=True)
    if engagement_fraction >= 1.0:
        raise ValueError("engagement_fraction must be < 1")
    return (
        4.0
        * crushing_pressure_pa
        * particle_radius_m
        * math.sqrt(2.0 * engagement_fraction - engagement_fraction**2)
        / engagement_fraction
    )


def _check(name: str, value: Any, *, positive: bool = False, nonnegative: bool = False) -> float:
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


class CappedBowDEM(MovingWedgeBowIceDEM3D):
    """Wedge-bow DEM with a plastic crushing cap and Coulomb hull friction.

    ``_wedge_contact`` is overridden (not the force integrator), so the inherited
    ``forces``/``diagnostics`` paths pick up the capped, frictional contact and
    ``bow_reaction_x_N`` stays the honest hull reaction of this model.
    """

    #: Slip-speed scale [m/s] of the friction regularisation.  Velocities below
    #: it are treated as stick; the value is a numerical device, not a physical
    #: parameter (reported in every case bundle).
    DEFAULT_SLIP_SCALE = 1.0e-3

    def __init__(
        self,
        config: DEM3DConfig,
        *,
        crushing_pressure_pa: float,
        friction_coefficient: float = 0.05,
        bow_speed: float = 0.1,
        wedge_angle_deg: float = 30.0,
        initial_gap: float = 0.0,
        tip_height_fraction: float = 0.5,
        bow_half_width_m: float | None = None,
        tributary_width_m: float | None = None,
        slip_scale_m_s: float | None = None,
    ) -> None:
        _check("crushing_pressure_pa", crushing_pressure_pa, positive=True)
        _check("friction_coefficient", friction_coefficient, nonnegative=True)
        if tributary_width_m is not None:
            _check("tributary_width_m", tributary_width_m, positive=True)
        super().__init__(
            config,
            bow_speed=bow_speed,
            wedge_angle_deg=wedge_angle_deg,
            initial_gap=initial_gap,
            tip_height_fraction=tip_height_fraction,
            bow_half_width_m=bow_half_width_m,
        )
        self.crushing_pressure_pa = float(crushing_pressure_pa)
        self.friction_coefficient = float(friction_coefficient)
        self.tributary_width_m = (
            2.0 * float(config.radius) if tributary_width_m is None else float(tributary_width_m)
        )
        self.slip_scale_m_s = (
            self.DEFAULT_SLIP_SCALE if slip_scale_m_s is None else float(slip_scale_m_s)
        )
        #: per-contact state of the most recent ``_wedge_contact`` evaluation
        self.last_contact: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # contact law
    # ------------------------------------------------------------------
    def crushing_chord(self, penetration: torch.Tensor) -> torch.Tensor:
        """Circle chord ``2 sqrt(2 r d - d^2)`` at normal penetration ``d``, [m]."""
        radius = self.config.radius
        argument = (2.0 * radius * penetration - penetration.square()).clamp_min(0.0)
        return 2.0 * torch.sqrt(argument)

    def plastic_cap(self, penetration: torch.Tensor) -> torch.Tensor:
        """Plastic crushing cap ``p_crush * chord(d) * w_trib``, [N]."""
        return self.crushing_pressure_pa * self.crushing_chord(penetration) * self.tributary_width_m

    @torch.no_grad()
    def _wedge_contact(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Capped normal + Coulomb tangential bow forces (force on the ice)."""
        theta = math.radians(self.wedge_angle_deg)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        dx = self.positions[:, 0] - self.bow_tip_x
        dz = self.positions[:, 2] - self.bow_tip_z
        # float64 explicitly: ``torch.where(cond, 1.0, -1.0)`` would build a
        # float32 tensor and silently round the face normal to single precision.
        side = torch.where(dz >= 0, torch.ones_like(dz), -torch.ones_like(dz))
        wedge_distance = sin_t * dx + cos_t * dz.abs()
        normal = torch.zeros_like(self.positions)
        normal[:, 0] = sin_t
        normal[:, 2] = side * cos_t
        signed_distance = wedge_distance
        if self.bow_half_width_m is not None:
            dy = self.positions[:, 1] - self.bow_center_y_m
            side_distance = dy.abs() - self.bow_half_width_m
            side_active = side_distance > wedge_distance
            signed_distance = torch.maximum(wedge_distance, side_distance)
            normal[:, 0] = torch.where(side_active, 0.0, normal[:, 0])
            normal[:, 2] = torch.where(side_active, 0.0, normal[:, 2])
            normal[:, 1] = torch.where(side_active, torch.where(dy >= 0, 1.0, -1.0), 0.0)
        overlap = (self.config.radius - signed_distance).clamp_min(0.0)
        contact = overlap > 0.0
        relative_velocity = self.velocities.clone()
        relative_velocity[:, 0] -= self.bow_speed
        normal_speed = (relative_velocity * normal).sum(dim=1)
        penalty = (
            self.config.contact_stiffness * overlap - self.config.contact_damping * normal_speed
        ).clamp_min(0.0)
        cap = self.plastic_cap(overlap)
        normal_magnitude = torch.where(contact, torch.minimum(penalty, cap), torch.zeros_like(cap))
        # Tangential slip of the ice relative to the face, in the face plane.
        tangential = relative_velocity - normal_speed[:, None] * normal
        slip = torch.linalg.vector_norm(tangential, dim=1)
        friction_magnitude = (
            self.friction_coefficient
            * normal_magnitude
            * torch.tanh(slip / self.slip_scale_m_s)
        )
        friction = -friction_magnitude[:, None] * tangential / (slip[:, None] + 1e-30)
        force = normal_magnitude[:, None] * normal + torch.where(contact[:, None], friction, 0.0)
        reaction = -force.sum(dim=0)
        self.last_contact = {
            "force_on_particle": force,
            "normal": normal,
            "overlap": overlap,
            "penalty": torch.where(contact, penalty, torch.zeros_like(penalty)),
            "cap": cap,
            "normal_magnitude": normal_magnitude,
            "friction": torch.where(contact[:, None], friction, torch.zeros_like(friction)),
            "contact": contact,
            "slip": slip,
        }
        return force, reaction, overlap

    # ------------------------------------------------------------------
    # bond-graph connectivity
    # ------------------------------------------------------------------
    @torch.no_grad()
    def cluster_labels(self, *, max_sweeps: int = 4096) -> torch.Tensor:
        """Connected-component label per particle (min-label propagation)."""
        count = len(self.positions)
        labels = torch.arange(count, dtype=torch.int64, device=self.device)
        alive = self.alive
        if int(alive.sum()) == 0:
            return labels
        pairs = self.pairs[alive]
        if pairs.numel() == 0:
            return labels
        first = pairs[:, 0]
        second = pairs[:, 1]
        for _ in range(max_sweeps):
            previous = labels.clone()
            labels.scatter_reduce_(0, first, labels[second], reduce="amin", include_self=True)
            labels.scatter_reduce_(0, second, labels[first], reduce="amin", include_self=True)
            if bool(torch.equal(labels, previous)):
                break
        return labels

    @torch.no_grad()
    def detached_mask(self, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Particles not connected to the largest surviving bond cluster."""
        if labels is None:
            labels = self.cluster_labels()
        counts = torch.bincount(labels)
        if int(counts.numel()) == 0:
            return torch.zeros(len(self.positions), dtype=torch.bool, device=self.device)
        largest = int(torch.argmax(counts))
        return labels != largest

    # ------------------------------------------------------------------
    # contact records for the closure layer
    # ------------------------------------------------------------------
    @torch.no_grad()
    def contact_set(self, *, detached: torch.Tensor | None = None) -> HullContactSet:
        """Per-contact :class:`HullContactSet` of the current configuration."""
        state = self.last_contact
        if not state:
            self._wedge_contact()
            state = self.last_contact
        contact = state["contact"]
        indices = torch.nonzero(contact, as_tuple=False).flatten()
        if detached is None:
            detached = self.detached_mask()
        submerged = self.submerged_fraction()
        cap = state["cap"]
        return HullContactSet(
            positions=self.positions[indices],
            normals=state["normal"][indices],
            normal_force=state["normal_magnitude"][indices],
            friction_force=state["friction"][indices],
            penetration=state["overlap"][indices],
            cap_force=torch.where(
                torch.isfinite(cap[indices]), cap[indices], torch.full_like(cap[indices], math.inf)
            ),
            detached=detached[indices],
            submerged_fraction=submerged[indices],
        )


# ---------------------------------------------------------------------------
# case configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClosureCase:
    """Configuration of one 3-D hull/ice case for the closure study."""

    thickness_m: float = 0.40
    particle_radius_m: float = 0.025
    n_length: int = 16
    n_width: int = 4
    bow_speed_m_s: float = 0.10
    wedge_angle_deg: float = 30.0
    tip_height_fraction: float = 0.5
    bond_stiffness: float = 1200.0
    contact_stiffness: float = 3500.0
    breaking_strain: float = 0.008
    shear_breaking_strain: float = 0.025
    contact_damping: float = 0.5
    drag: float = 0.1
    density_kg_m3: float = ICE_DENSITY_KG_M3
    friction_coefficient: float = 0.05
    compressive_strength_pa: float = 1.5e6
    flexural_strength_pa: float = 0.5e6
    elastic_modulus_pa: float = 5.0e9
    #: Fraction of the crushing cap: Lindqvist's triangular-profile mean is 0.5.
    crushing_efficiency: float = 0.5
    gravity_m_s2: float = GRAVITY_M_S2
    water_density_kg_m3: float = WATER_DENSITY_KG_M3
    submerged_fraction_rest: float = ICE_DENSITY_KG_M3 / WATER_DENSITY_KG_M3
    wet_drag_coefficient: float = 0.0
    #: Quasi-static mass scaling: both the ice and the water density are
    #: multiplied by it, so the floating equilibrium (``f_sub = rho_i/rho_w``) and
    #: every contact/bond force are unchanged while the inertial time scale grows
    #: by ``sqrt(mass_scale)``.  Standard DEM device (``docs/LEVEL_ICE_RESISTANCE_DEM.md``
    #: uses the same device); its convergence must be demonstrated by a sweep and
    #: it is reported in every case bundle.
    mass_scale: float = 1.0
    #: Initial bow embedding [m]: the wedge starts already in contact with the
    #: sheet instead of spending the approach transient (the wedge tip starts
    #: outside the sheet, and the gap is filled at ``bow_speed``).  Purely an
    #: initial-condition device; the recorded contact state is the simulated one.
    initial_gap_m: float = 0.0
    embed_m: float = 0.010
    steps: int = 600
    contact_mask_every: int = 10
    steady_window_fraction: float = 0.4

    def __post_init__(self) -> None:
        for name in (
            "thickness_m",
            "particle_radius_m",
            "bow_speed_m_s",
            "wedge_angle_deg",
            "bond_stiffness",
            "contact_stiffness",
            "contact_damping",
            "density_kg_m3",
            "compressive_strength_pa",
            "flexural_strength_pa",
            "elastic_modulus_pa",
            "water_density_kg_m3",
        ):
            _check(name, getattr(self, name), positive=True)
        for name in ("friction_coefficient", "drag", "gravity_m_s2", "wet_drag_coefficient",
                     "embed_m", "mass_scale", "initial_gap_m"):
            _check(name, getattr(self, name), nonnegative=True)
        for name in ("crushing_efficiency", "submerged_fraction_rest", "steady_window_fraction"):
            value = _check(name, getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        _check("mass_scale", self.mass_scale, positive=True)
        for name in ("n_length", "n_width", "steps", "contact_mask_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 5.0 <= self.wedge_angle_deg <= 85.0:
            raise ValueError("wedge_angle_deg must be in [5, 85]")

    @property
    def layers(self) -> int:
        return max(2, int(round(self.thickness_m / (2.0 * self.particle_radius_m))))

    @property
    def realised_thickness_m(self) -> float:
        return self.layers * 2.0 * self.particle_radius_m

    @property
    def model_width_m(self) -> float:
        return self.n_width * 2.0 * self.particle_radius_m

    @property
    def model_length_m(self) -> float:
        return (self.n_length - 1) * 2.0 * self.particle_radius_m

    @property
    def crushing_pressure_pa(self) -> float:
        """Mean crushing pressure of the contact cap, ``eff * sigma_c``."""
        return self.crushing_efficiency * self.compressive_strength_pa

    @property
    def water_level_m(self) -> float:
        """Free-surface height giving the lattice ``submerged_fraction_rest``."""
        return self.realised_thickness_m * self.submerged_fraction_rest

    @property
    def effective_density_kg_m3(self) -> float:
        return self.density_kg_m3 * self.mass_scale

    @property
    def effective_water_density_kg_m3(self) -> float:
        return self.water_density_kg_m3 * self.mass_scale

    @classmethod
    def physical(cls, **changes: Any) -> "ClosureCase":
        """Case whose bond law is derived from the ice material properties.

        Offline, the lattice carries no calibrated bond law, and the inherited
        defaults (``bond_stiffness = 1200 N/m`` -> a bond breaking force of
        ``~10 N``) are three orders below the contact forces a crushing cap
        produces, so any capped contact shreds the sheet.  This constructor
        removes that internal inconsistency with two textbook lattice estimates:

        * axial bond stiffness of a cubic lattice with spacing ``2 r``:
          ``k = E * 2r`` (so ``E = k / (2r)`` recovers the ice modulus);
        * bond breaking strain ``eps_b = sigma_f / E`` (the material failure
          strain), which makes the bond breaking force ``sigma_f * (2r)^2``.

        ``mass_scale`` then sets the inertial time scale; the contact penalty is
        the cap-engagement stiffness of :func:`cap_engagement_stiffness`.
        """
        radius = float(changes.get("particle_radius_m", cls.particle_radius_m))
        modulus = float(changes.get("elastic_modulus_pa", cls.elastic_modulus_pa))
        flexural = float(changes.get("flexural_strength_pa", cls.flexural_strength_pa))
        compressive = float(changes.get("compressive_strength_pa", cls.compressive_strength_pa))
        efficiency = float(changes.get("crushing_efficiency", cls.crushing_efficiency))
        engagement = float(changes.get("cap_engagement_fraction", 0.1))
        derived = {
            "bond_stiffness": modulus * 2.0 * radius,
            "breaking_strain": flexural / modulus,
            "shear_breaking_strain": 2.0 * flexural / modulus,
            "contact_stiffness": cap_engagement_stiffness(
                efficiency * compressive, radius, engagement
            ),
        }
        base = {**cls().__dict__, **derived, **changes}
        return cls(**base)

    def to_config(self) -> DEM3DConfig:
        base = DEM3DConfig(
            nx=self.n_length,
            ny=self.n_width,
            nz=self.layers,
            radius=self.particle_radius_m,
            density=self.effective_density_kg_m3,
            bond_stiffness=self.bond_stiffness,
            contact_stiffness=self.contact_stiffness,
            breaking_strain=self.breaking_strain,
            shear_breaking_strain=self.shear_breaking_strain,
            contact_damping=self.contact_damping,
            drag=self.drag,
            tool_speed=self.bow_speed_m_s,
            gravity=self.gravity_m_s2,
            water_density=self.effective_water_density_kg_m3,
            water_level=self.water_level_m,
            wet_drag_coefficient=self.wet_drag_coefficient,
            fix_x_edges=True,
            fix_bottom=False,
        )
        return DEM3DConfig(**{**base.__dict__, "dt": base.recommended_dt})

    def steps_for_travel(self, travel_m: float) -> int:
        """Steps needed to advance the bow by ``travel_m`` at the stable step.

        Used to compare two discretisations at the *same* advance instead of at
        the same step count -- a refinement study at equal step count would
        compare different physical states.
        """
        _check("travel_m", travel_m, positive=True)
        dt = self.to_config().recommended_dt
        return max(1, int(round(travel_m / (self.bow_speed_m_s * dt))))

    def with_changes(self, **changes: Any) -> "ClosureCase":
        return ClosureCase(**{**self.__dict__, **changes})


def build_simulation(case: ClosureCase) -> CappedBowDEM:
    """Instantiate the capped wedge-bow DEM for ``case`` (bow pre-embedded)."""
    sim = CappedBowDEM(
        case.to_config(),
        crushing_pressure_pa=case.crushing_pressure_pa,
        friction_coefficient=case.friction_coefficient,
        bow_speed=case.bow_speed_m_s,
        wedge_angle_deg=case.wedge_angle_deg,
        initial_gap=case.initial_gap_m,
        tip_height_fraction=case.tip_height_fraction,
        bow_half_width_m=None,
    )
    sim.bow_start_x += case.embed_m
    return sim


def run_case(case: ClosureCase) -> dict[str, Any]:
    """Run one closure case and return the per-step series and the averaged split.

    The steady window is the trailing ``steady_window_fraction`` of the run after
    first hull contact, which excludes the elastic ramp-up and the initial
    stress-wave transient.  Nothing is fitted: the reported numbers are the
    window means of the measured contact decomposition.
    """
    sim = build_simulation(case)
    heading = (1.0, 0.0, 0.0)
    series: list[dict[str, float]] = []
    snapshots: list[MechanismDecomposition] = []
    labels: torch.Tensor | None = None
    first_contact_step: int | None = None
    max_relative_residual = 0.0
    first_fracture_time: float | None = None
    for step in range(case.steps + 1):
        if step:
            sim.step()
        if step % max(1, case.contact_mask_every) == 0 or step == case.steps:
            labels = sim.cluster_labels()
        detached = sim.detached_mask(labels)
        contacts = sim.contact_set(detached=detached)
        decomposition = decompose(contacts, heading=heading)
        max_relative_residual = max(max_relative_residual, decomposition.relative_residual)
        if contacts.count and first_contact_step is None:
            first_contact_step = step
        diagnostics = sim.diagnostics()
        if first_fracture_time is None and int(diagnostics["broken_bonds"]) > 0:
            first_fracture_time = float(sim.time)
        series.append(
            {
                "step": step,
                "time_s": float(sim.time),
                "bow_tip_x_m": float(sim.bow_tip_x),
                "contacts": float(decomposition.contact_count),
                "capped_contacts": float(decomposition.capped_count),
                "detached_contacts": float(decomposition.detached_count),
                "r_total_N": decomposition.r_total,
                "r_crushing_N": decomposition.components["crushing"],
                "r_bending_N": decomposition.components["bending"],
                "r_submersion_N": decomposition.components["submersion"],
                "residual_N": decomposition.residual,
                "relative_residual": decomposition.relative_residual,
                "reaction_x_N": decomposition.reaction_vector[0],
                "reaction_z_N": decomposition.reaction_vector[2],
                "broken_bonds": float(diagnostics["broken_bonds"]),
                "detached_particles": float(int(detached.sum())),
                "contact_area_m2": _contact_patch_area(sim),
            }
        )
        snapshots.append(decomposition)
        if not all(math.isfinite(value) for value in series[-1].values()):
            raise FloatingPointError(f"non-finite closure history at step {step}")
    start = 0 if first_contact_step is None else first_contact_step
    window_length = max(1, int(round(case.steady_window_fraction * (len(snapshots) - start))))
    window = snapshots[len(snapshots) - window_length:]
    averaged = average_decompositions(window)
    peak = max(snapshot.r_total for snapshot in snapshots)
    width = case.model_width_m
    characteristic = plate_characteristic_length(
        case.elastic_modulus_pa, case.realised_thickness_m,
        case.water_density_kg_m3, case.gravity_m_s2,
    )
    vertical_per_width = averaged.vertical_reaction / width
    return {
        "case": {
            "thickness_m": case.thickness_m,
            "realised_thickness_m": case.realised_thickness_m,
            "layers": case.layers,
            "particle_radius_m": case.particle_radius_m,
            "particle_count": int(len(sim.positions)),
            "model_length_m": case.model_length_m,
            "model_width_m": width,
            "bow_speed_m_s": case.bow_speed_m_s,
            "wedge_angle_deg": case.wedge_angle_deg,
            "friction_coefficient": case.friction_coefficient,
            "crushing_pressure_pa": case.crushing_pressure_pa,
            "crushing_efficiency": case.crushing_efficiency,
            "compressive_strength_pa": case.compressive_strength_pa,
            "elastic_modulus_pa": case.elastic_modulus_pa,
            "flexural_strength_pa": case.flexural_strength_pa,
            "water_level_m": case.water_level_m,
            "mass_scale": case.mass_scale,
            "effective_density_kg_m3": case.effective_density_kg_m3,
            "bond_stiffness": case.bond_stiffness,
            "breaking_strain": case.breaking_strain,
            "contact_stiffness": case.contact_stiffness,
            "gravity_m_s2": case.gravity_m_s2,
            "dt_s": float(sim.dt),
            "steps": case.steps,
            "simulated_time_s": float(sim.time),
            "bow_travel_m": float(sim.bow_speed * sim.time),
            "steady_window_steps": window_length,
            "steady_window_start_step": len(snapshots) - window_length,
            "first_contact_step": first_contact_step,
            "first_fracture_time_s": first_fracture_time,
            "final_broken_bonds": int(diagnostics["broken_bonds"]),
            "final_broken_bond_fraction": float(
                diagnostics["broken_bonds"] / max(1, len(sim.alive))
            ),
            "final_detached_particles": int(sim.detached_mask(sim.cluster_labels()).sum()),
            "slip_scale_m_s": sim.slip_scale_m_s,
            "tributary_width_m": sim.tributary_width_m,
        },
        "series": series,
        "steady": averaged.as_dict(),
        "steady_per_width": {
            "r_total_N_m": averaged.r_total / width,
            **{name: averaged.components[name] / width for name in averaged.components},
        },
        "peak_total_N": peak,
        "max_relative_residual": max_relative_residual,
        "characteristic_length_m": characteristic,
        "sheet_length_over_characteristic_length": case.model_length_m / characteristic,
        "flexural_demand_ratio": flexural_demand_ratio(
            vertical_per_width, characteristic,
            case.flexural_strength_pa, case.realised_thickness_m,
        ),
        "average_contact_pressure_pa": _average_contact_pressure(sim),
        "contact_patch_area_m2": _contact_patch_area(sim),
        "contact_patch_area_ratio": _contact_patch_area(sim)
        / (case.realised_thickness_m * case.model_width_m),
    }


def _contact_patch_area(sim: CappedBowDEM) -> float:
    """Sum of the spherical-cap contact areas of the active contacts [m^2].

    ``A = sum_i pi (2 r d_i - d_i^2)``.  Comparing it with the nominal Lindqvist
    crushing band ``h * B`` is the diagnostic that explains any magnitude gap:
    the DEM can only transmit the crushing pressure over the area it actually
    resolves.
    """
    state = sim.last_contact
    if not state:
        return 0.0
    contact = state["contact"]
    if int(contact.sum()) == 0:
        return 0.0
    penetration = state["overlap"][contact]
    area = math.pi * (2.0 * sim.config.radius * penetration - penetration.square()).clamp_min(0.0)
    return float(area.sum())


def _average_contact_pressure(sim: CappedBowDEM) -> float:
    """Mean normal contact pressure over the active contact patches [Pa]."""
    state = sim.last_contact
    if not state:
        return 0.0
    contact = state["contact"]
    if int(contact.sum()) == 0:
        return 0.0
    penetration = state["overlap"][contact]
    area = math.pi * (2.0 * sim.config.radius * penetration - penetration.square()).clamp_min(0.0)
    force = state["normal_magnitude"][contact]
    total_area = float(area.sum())
    if total_area <= 0.0:
        return 0.0
    return float(force.sum()) / total_area