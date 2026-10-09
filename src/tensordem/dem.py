"""Two-dimensional central-force bonded disks, using SI units."""

from dataclasses import asdict, dataclass
import math

import torch

from .hull_contact import polyline_contact_forces


@dataclass(frozen=True)
class DEMConfig:
    nx: int = 21
    ny: int = 5
    radius: float = 0.025
    thickness: float = 0.2
    density: float = 917.0
    bond_stiffness: float = 2_000.0
    contact_stiffness: float = 2_000.0
    breaking_strain: float = 0.015
    shear_breaking_strain: float = 0.03
    contact_damping: float = 5.0
    drag: float = 2.0
    tool_radius: float = 0.1
    tool_speed: float = 1.0
    tool_gap: float = 0.01
    dt: float | None = None
    device: str = "cpu"
    fix_edges: bool = True

    def __post_init__(self) -> None:
        for name in ("nx", "ny"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        for name in (
            "radius", "thickness", "density", "bond_stiffness",
            "contact_stiffness", "breaking_strain", "shear_breaking_strain",
            "tool_radius", "tool_speed",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("contact_damping", "drag", "tool_gap"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.dt is not None and (not math.isfinite(self.dt) or self.dt <= 0):
            raise ValueError("dt must be finite and positive")

    @property
    def mass(self) -> float:
        return self.density * math.pi * self.radius**2 * self.thickness

    @property
    def recommended_dt(self) -> float:
        # Bound all possible pair contacts plus the moving tool, and dashpots.
        neighbors = self.nx * self.ny - 1
        stiffness = (2 * neighbors + 1) * max(
            self.bond_stiffness, self.contact_stiffness
        )
        damping = (2 * neighbors + 1) * self.contact_damping + self.drag
        elastic_dt = 0.2 * math.sqrt(self.mass / stiffness)
        damping_dt = 0.2 * self.mass / damping if damping else math.inf
        travel_dt = 0.1 * self.radius / self.tool_speed
        return min(elastic_dt, damping_dt, travel_dt)


class IceDEM:
    """A small ice lattice; only tensile axial strain breaks bonds.

    Positions, velocities and externally applied forces have shape (N, 2).
    External forces use SI newtons in the global x/y frame and are evaluated
    at the current step (not accumulated internally). Bond and contact forces
    are central and equal/opposite. Tensile failure uses axial bond strain;
    shear failure uses an objective local Green-Lagrange strain invariant.
    The model still has no particle rotations or explicit shear-bond moments.
    """

    def __init__(
        self,
        config: DEMConfig = DEMConfig(),
        *,
        hull_profile: torch.Tensor | None = None,
        hull_start: tuple[float, float] = (0.0, 0.0),
        hull_velocity: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and CUDA devices are supported")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.dt = config.dt if config.dt is not None else config.recommended_dt
        if self.dt > config.recommended_dt:
            raise ValueError(
                f"dt exceeds conservative limit {config.recommended_dt:.6g} s"
            )
        x = torch.arange(config.nx, dtype=torch.float64, device=self.device)
        y = torch.arange(config.ny, dtype=torch.float64, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.positions = torch.stack(
            (xx.flatten() - (config.nx - 1) / 2, yy.flatten()), dim=1
        ) * (2 * config.radius)
        self.initial_positions = self.positions.clone()
        self.velocities = torch.zeros_like(self.positions)
        self.fixed = torch.zeros(len(self.positions), dtype=torch.bool, device=self.device)
        if config.fix_edges:
            self.fixed = (xx.flatten() == 0) | (xx.flatten() == config.nx - 1)
        self.pairs = torch.triu_indices(
            len(self.positions), len(self.positions), offset=1, device=self.device
        ).T
        delta = self.positions[self.pairs[:, 1]] - self.positions[self.pairs[:, 0]]
        distances = torch.linalg.vector_norm(delta, dim=1)
        self.initial_normals = delta / distances[:, None]
        # Square-grid nearest neighbors and diagonals form the bonded lattice.
        self.bonded = distances <= 2 * config.radius * math.sqrt(2) * (1 + 1e-10)
        self.bond_indices = self.pairs[self.bonded]
        self.reference_bond_vectors = (
            self.initial_positions[self.bond_indices[:, 1]]
            - self.initial_positions[self.bond_indices[:, 0]]
        )
        self.rest_lengths = distances.clone()
        self.alive = self.bonded.clone()
        self.time = 0.0
        self.step_count = 0
        self.accounting = dict(external_work_J=0.0, fracture_release_J=0.0,
                               tool_work_J=0.0, damping_dissipation_J=0.0,
                               external_impulse_Ns=[0.0, 0.0],
                               support_impulse_Ns=[0.0, 0.0], drag_impulse_Ns=[0.0, 0.0],
                               tool_impulse_on_ice_Ns=[0.0, 0.0])
        self.last_step = {}
        self.tool_start = torch.tensor(
            [0.0, (config.ny - 1) * 2 * config.radius
             + config.radius + config.tool_radius + config.tool_gap],
            dtype=torch.float64, device=self.device,
        )
        self.tool_velocity = torch.tensor(
            [0.0, -config.tool_speed], dtype=torch.float64, device=self.device
        )
        self.hull_profile = None
        self.hull_start = torch.tensor(hull_start, dtype=torch.float64, device=self.device)
        self.hull_velocity = torch.tensor(hull_velocity, dtype=torch.float64, device=self.device)
        if not bool(torch.isfinite(self.hull_start).all() and torch.isfinite(self.hull_velocity).all()):
            raise ValueError("hull_start and hull_velocity must be finite")
        if hull_profile is not None:
            if not isinstance(hull_profile, torch.Tensor):
                raise TypeError("hull_profile must be a torch.Tensor with shape (M, 2)")
            if hull_profile.ndim != 2 or hull_profile.shape[1] != 2 or hull_profile.shape[0] < 2:
                raise ValueError("hull_profile must have shape (M, 2), M >= 2")
            self.hull_profile = hull_profile.to(device=self.device, dtype=torch.float64).clone()
            if not bool(torch.isfinite(self.hull_profile).all()):
                raise ValueError("hull_profile must contain only finite values")
            if bool((torch.linalg.vector_norm(self.hull_profile[1:] - self.hull_profile[:-1], dim=1)
                     <= torch.finfo(self.hull_profile.dtype).eps).any()):
                raise ValueError("hull_profile must not contain zero-length segments")

    @property
    def tool_position(self) -> torch.Tensor:
        """Return the circular tool position or prescribed hull translation."""
        if self.hull_profile is not None:
            return self.hull_start + self.time * self.hull_velocity
        return self.tool_start + self.time * self.tool_velocity

    @property
    def broken_bonds(self) -> int:
        return int((self.bonded & ~self.alive).sum().item())

    @torch.no_grad()
    def _local_objective_shear_strain(self) -> torch.Tensor:
        """Return a rotation-invariant local shear strain estimate per particle.

        A least-squares deformation gradient is reconstructed from the initial
        bonded neighborhood. The Green-Lagrange strain removes rigid rotation;
        the equivalent in-plane shear measure is sqrt((E_xx-E_yy)^2 + 4 E_xy^2).
        """
        n = len(self.positions)
        i, j = self.bond_indices.T
        reference = self.reference_bond_vectors
        current = self.positions[j] - self.positions[i]
        reference_outer = reference[:, :, None] * reference[:, None, :]
        current_reference = current[:, :, None] * reference[:, None, :]
        covariance = torch.zeros((n, 2, 2), dtype=self.positions.dtype, device=self.device)
        cross = torch.zeros_like(covariance)
        covariance.index_add_(0, i, reference_outer)
        covariance.index_add_(0, j, reference_outer)
        cross.index_add_(0, i, current_reference)
        cross.index_add_(0, j, current_reference)
        deformation = cross @ torch.linalg.pinv(covariance)
        identity = torch.eye(2, dtype=self.positions.dtype, device=self.device)
        strain = 0.5 * (deformation.transpose(1, 2) @ deformation - identity)
        return torch.sqrt(
            (strain[:, 0, 0] - strain[:, 1, 1]).square()
            + 4.0 * strain[:, 0, 1].square()
        )

    def _validate_external_forces(
        self, external_forces: torch.Tensor | None
    ) -> torch.Tensor | None:
        if external_forces is None:
            return None
        if not isinstance(external_forces, torch.Tensor):
            raise TypeError("external_forces must be a torch.Tensor with shape (N, 2)")
        if tuple(external_forces.shape) != tuple(self.positions.shape):
            raise ValueError(
                f"external_forces must have shape {tuple(self.positions.shape)}"
            )
        loads = external_forces.to(device=self.device, dtype=self.positions.dtype)
        if not bool(torch.isfinite(loads).all()):
            raise ValueError("external_forces must contain only finite values")
        return loads

    @torch.no_grad()
    def forces(
        self, external_forces: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return total particle force and ice-on-tool reaction.

        external_forces is an optional instantaneous nodal load array in N.
        It is additive to DEM bond/contact forces and linear drag. The returned
        tool reaction retains its existing sign convention (ice on tool).
        """
        cfg = self.config
        loads = self._validate_external_forces(external_forces)
        i, j = self.pairs.T
        delta = self.positions[j] - self.positions[i]
        distance = torch.linalg.vector_norm(delta, dim=1)
        normal = delta / distance.clamp_min(torch.finfo(delta.dtype).eps)[:, None]
        normal = torch.where((distance > 0)[:, None], normal, self.initial_normals)
        extension = distance - self.rest_lengths
        tensile_strain = extension / self.rest_lengths
        local_shear = self._local_objective_shear_strain()
        i_bond, j_bond = self.bond_indices.T
        bond_shear = 0.5 * (local_shear[i_bond] + local_shear[j_bond])
        shear_failed = bond_shear > cfg.shear_breaking_strain
        tensile_failed = tensile_strain[self.bonded] > cfg.breaking_strain
        previous = self.alive.clone()
        self.alive[self.bonded] &= ~(tensile_failed | shear_failed)
        newly_broken = previous & ~self.alive
        # Actual spring energy removed at failure, not a calibrated fracture Gc.
        released = 0.5 * cfg.bond_stiffness * extension[newly_broken].square().sum()
        self.accounting["fracture_release_J"] += float(released)
        normal_speed = ((self.velocities[j] - self.velocities[i]) * normal).sum(dim=1)
        overlap = (2 * cfg.radius - distance).clamp_min(0)
        repulsion = (
            cfg.contact_stiffness * overlap - cfg.contact_damping * normal_speed
        ).clamp_min(0)
        repulsion = torch.where(
            (overlap > 0) & ~self.alive, repulsion, torch.zeros_like(repulsion)
        )
        magnitude = torch.where(
            self.alive, cfg.bond_stiffness * extension, -repulsion
        )
        pair_force = magnitude[:, None] * normal
        force = -cfg.drag * self.velocities
        force.index_add_(0, i, pair_force)
        force.index_add_(0, j, -pair_force)

        if self.hull_profile is None:
            tool_delta = self.positions - self.tool_position
            tool_distance = torch.linalg.vector_norm(tool_delta, dim=1)
            tool_normal = tool_delta / tool_distance.clamp_min(
                torch.finfo(tool_delta.dtype).eps
            )[:, None]
            fallback = torch.zeros_like(tool_normal)
            fallback[:, 1] = -1
            tool_normal = torch.where((tool_distance > 0)[:, None], tool_normal, fallback)
            tool_overlap = (cfg.radius + cfg.tool_radius - tool_distance).clamp_min(0)
            tool_speed = ((self.velocities - self.tool_velocity) * tool_normal).sum(dim=1)
            tool_magnitude = (
                cfg.contact_stiffness * tool_overlap - cfg.contact_damping * tool_speed
            ).clamp_min(0)
            tool_magnitude = torch.where(
                tool_overlap > 0, tool_magnitude, torch.zeros_like(tool_magnitude)
            )
            tool_force = tool_magnitude[:, None] * tool_normal
            tool_reaction = -tool_force.sum(dim=0)
        else:
            vertices = self.hull_profile + self.tool_position
            tool_force, tool_reaction = polyline_contact_forces(
                self.positions,
                self.velocities,
                vertices,
                self.hull_velocity,
                particle_radius=cfg.radius,
                hull_radius=cfg.tool_radius,
                stiffness=cfg.contact_stiffness,
                damping=cfg.contact_damping,
            )
        # Dissipation is elastic-force minus actual force dotted into relative
        # motion, including the non-tensile dashpot clipping branch.
        pair_dissipation = ((repulsion - cfg.contact_stiffness * overlap) * (-normal_speed))
        pair_dissipation = torch.where((overlap > 0) & ~self.alive, pair_dissipation, torch.zeros_like(pair_dissipation))
        # Recover conservative tool force from the same geometry with damping 0.
        if self.hull_profile is None:
            conservative_tool = cfg.contact_stiffness * tool_overlap[:, None] * tool_normal
            tool_v = self.tool_velocity
        else:
            conservative_tool, _ = polyline_contact_forces(
                self.positions, self.velocities, vertices, self.hull_velocity,
                particle_radius=cfg.radius, hull_radius=cfg.tool_radius,
                stiffness=cfg.contact_stiffness, damping=0.0)
            tool_v = self.hull_velocity
        tool_dissipation = -((tool_force - conservative_tool) * (self.velocities - tool_v)).sum()
        self._damping_power = float(pair_dissipation.sum() + tool_dissipation + cfg.drag * self.velocities.square().sum())
        self._tool_power = float(-tool_reaction @ tool_v)
        force += tool_force
        if loads is not None:
            force += loads
        return force, tool_reaction

    @torch.no_grad()
    def step(self, external_forces: torch.Tensor | None = None, *, load_time: float | None = None) -> None:
        """Semi-implicit Euler; optional nodal loads are supplied in newtons."""
        if load_time is not None and (not math.isfinite(load_time) or
                abs(load_time - self.time) > 1e-12 * max(1.0, abs(self.time))):
            raise ValueError("load_time must equal current DEM time in seconds")
        loads = self._validate_external_forces(external_forces)
        old_positions = self.positions.clone()
        force, reaction = self.forces(external_forces=loads)
        support = -force[self.fixed].sum(0)
        external = torch.zeros(2, dtype=self.positions.dtype, device=self.device) if loads is None else loads.sum(0)
        for name, vector in (("external_impulse_Ns", external),
                             ("support_impulse_Ns", support),
                             ("drag_impulse_Ns", -self.config.drag * self.velocities.sum(0)),
                             ("tool_impulse_on_ice_Ns", -reaction)):
            self.accounting[name] = [a + self.dt * float(b) for a, b in zip(self.accounting[name], vector)]
        self.velocities += self.dt * force / self.config.mass
        self.velocities[self.fixed] = 0
        self.positions += self.dt * self.velocities
        self.positions[self.fixed] = self.initial_positions[self.fixed]
        displacement = self.positions - old_positions
        work = 0.0 if loads is None else float((loads * displacement).sum())
        self.accounting["external_work_J"] += work
        self.accounting["tool_work_J"] += self.dt * self._tool_power
        self.accounting["damping_dissipation_J"] += self.dt * self._damping_power
        self.last_step = dict(time_start_s=self.time, time_end_s=self.time + self.dt,
                              external_work_J=work, support_force_N=support.tolist(),
                              tool_reaction_N=reaction.tolist(), external_force_N=external.tolist(),
                              external_moment_Nm=0.0 if loads is None else float(
                                  (old_positions[:, 0] * loads[:, 1] - old_positions[:, 1] * loads[:, 0]).sum()))
        self.time += self.dt
        self.step_count += 1
        if not bool(torch.isfinite(self.positions).all()) or not bool(
            torch.isfinite(self.velocities).all()
        ):
            raise FloatingPointError("Non-finite simulation state; reduce dt")

    @torch.no_grad()
    def diagnostics(
        self, external_forces: torch.Tensor | None = None
    ) -> dict[str, float | int]:
        force, reaction = self.forces(external_forces=external_forces)
        boundary_reaction = -force[self.fixed].sum(dim=0)
        return {
            "time": self.time,
            "tool_x": float(self.tool_position[0].item()),
            "tool_y": float(self.tool_position[1].item()),
            "reaction_x": float(reaction[0].item()),
            "reaction_y": float(reaction[1].item()),
            "boundary_reaction_x": float(boundary_reaction[0].item()),
            "boundary_reaction_y": float(boundary_reaction[1].item()),
            "broken_bonds": self.broken_bonds,
            "kinetic_energy": float(
                (0.5 * self.config.mass * self.velocities.square().sum()).item()
            ),
        }

    def exchange_state(self) -> dict:
        """SI global x/y state at t_n; supplied forces are held over [t_n,t_n+dt].

        Mass is a physical disk mass (finite extrusion thickness), not mass per
        unit depth. Planar moments use x*Fy-y*Fx, positive counterclockwise.
        No rotational DOFs, bending stiffness or friction are implied.
        """
        return dict(schema="tensordem.exchange/1", time_s=self.time, dt_s=self.dt,
                    positions_m=self.positions.tolist(), velocities_m_s=self.velocities.tolist(),
                    masses_kg=[self.config.mass] * len(self.positions),
                    fixed=self.fixed.tolist(), particle_radius_m=self.config.radius,
                    thickness_m=self.config.thickness)

    def fragments(self) -> list[dict]:
        """Connected components of surviving bonds, including isolated disks."""
        neighbors = [set() for _ in self.positions]
        for i, j in self.pairs[self.alive].tolist():
            neighbors[i].add(j)
            neighbors[j].add(i)
        unseen = set(range(len(neighbors)))
        result = []
        while unseen:
            todo = [min(unseen)]
            component = []
            unseen.remove(todo[0])
            while todo:
                i = todo.pop()
                component.append(i)
                for j in sorted(neighbors[i] & unseen):
                    unseen.remove(j)
                    todo.append(j)
            ids = sorted(component)
            result.append(dict(particle_ids=ids, mass_kg=len(ids) * self.config.mass,
                               centroid_m=self.positions[ids].mean(0).tolist(),
                               mean_velocity_m_s=self.velocities[ids].mean(0).tolist()))
        return result

    def snapshot(self) -> dict:
        """Complete JSON restart; derived lattice topology is rebuilt and checked."""
        import copy
        return dict(schema="tensordem.restart/1", config=asdict(self.config),
                    dt_s=self.dt, time_s=self.time, step_count=self.step_count,
                    positions_m=self.positions.tolist(), velocities_m_s=self.velocities.tolist(),
                    initial_positions_m=self.initial_positions.tolist(), fixed=self.fixed.tolist(),
                    alive=self.alive.tolist(), tool_start_m=self.tool_start.tolist(),
                    tool_velocity_m_s=self.tool_velocity.tolist(),
                    hull_profile_m=None if self.hull_profile is None else self.hull_profile.tolist(),
                    hull_start_m=self.hull_start.tolist(), hull_velocity_m_s=self.hull_velocity.tolist(),
                    accounting=copy.deepcopy(self.accounting), last_step=copy.deepcopy(self.last_step))

    @classmethod
    def from_snapshot(cls, state: dict, *, device: str | None = None) -> "IceDEM":
        """Fail closed on missing fields, incompatible topology or nonfinite data."""
        import copy
        if not isinstance(state, dict) or state.get("schema") != "tensordem.restart/1":
            raise ValueError("unsupported DEM restart schema")
        required = set(cls(DEMConfig(nx=2, ny=2)).snapshot())
        if set(state) != required:
            raise ValueError("restart fields missing or unknown")
        cfg = dict(state["config"])
        if device is not None:
            cfg["device"] = device
        config = DEMConfig(**cfg)
        hull = state["hull_profile_m"]
        result = cls(config, hull_profile=None if hull is None else torch.tensor(hull, dtype=torch.float64),
                     hull_start=state["hull_start_m"], hull_velocity=state["hull_velocity_m_s"])
        if state["dt_s"] != result.dt or not math.isfinite(state["time_s"]) or state["time_s"] < 0:
            raise ValueError("invalid restart time or timestep")
        count = state["step_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("invalid restart step_count")
        if abs(state["time_s"] - count * result.dt) > 1e-10 * max(1.0, state["time_s"]):
            raise ValueError("restart time inconsistent with step count")
        for key, attribute in (("positions_m", "positions"), ("velocities_m_s", "velocities"),
                               ("initial_positions_m", "initial_positions"),
                               ("tool_start_m", "tool_start"), ("tool_velocity_m_s", "tool_velocity")):
            original = getattr(result, attribute)
            tensor = torch.tensor(state[key], dtype=torch.float64, device=result.device)
            if tensor.shape != original.shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"invalid restart {key}")
            if attribute == "initial_positions" and not torch.equal(tensor, original):
                raise ValueError("reference lattice mismatch")
            setattr(result, attribute, tensor)
        for key in ("fixed", "alive"):
            values = state[key]
            if not isinstance(values, list) or any(type(v) is not bool for v in values):
                raise ValueError(f"invalid boolean restart {key}")
            tensor = torch.tensor(values, dtype=torch.bool, device=result.device)
            if tensor.shape != getattr(result, key).shape:
                raise ValueError(f"invalid restart {key} shape")
            if key == "alive" and bool((tensor & ~result.bonded).any()):
                raise ValueError("nonbonded pair cannot be alive")
            if key == "fixed" and not torch.equal(tensor, result.fixed):
                raise ValueError("fixed boundary mismatch")
            setattr(result, key, tensor)
        if not torch.equal(result.positions[result.fixed], result.initial_positions[result.fixed]) or bool((result.velocities[result.fixed] != 0).any()):
            raise ValueError("fixed particle state mismatch")
        accounting = state["accounting"]
        if set(accounting) != set(result.accounting):
            raise ValueError("restart accounting fields mismatch")
        for key, value in accounting.items():
            tensor = torch.as_tensor(value, dtype=torch.float64)
            expected_shape = (2,) if "impulse" in key else ()
            if tuple(tensor.shape) != expected_shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError("invalid restart accounting")
            if key == "fracture_release_J" and value < 0:
                raise ValueError("negative fracture release")
        # last_step is diagnostic only; nevertheless reject nonfinite JSON numbers.
        import json
        json.dumps(state["last_step"], allow_nan=False)
        result.accounting = copy.deepcopy(accounting)
        result.last_step = copy.deepcopy(state["last_step"])
        result.time, result.step_count = state["time_s"], count
        return result

    def mechanical_energy(self) -> dict:
        """Actual current spring/contact potential; does not advance failure."""
        delta = self.positions[self.pairs[:, 1]] - self.positions[self.pairs[:, 0]]
        distances = torch.linalg.vector_norm(delta, dim=1)
        extension = distances - self.rest_lengths
        overlap = (2 * self.config.radius - distances).clamp_min(0)
        bond = 0.5 * self.config.bond_stiffness * extension[self.alive].square().sum()
        contact = 0.5 * self.config.contact_stiffness * overlap[~self.alive].square().sum()
        if self.hull_profile is None:
            distance = torch.linalg.vector_norm(self.positions - self.tool_position, dim=1)
        else:
            vertices = self.hull_profile + self.tool_position
            segments = vertices[1:] - vertices[:-1]
            rel = self.positions[:, None] - vertices[None, :-1]
            fraction = ((rel * segments).sum(2) / segments.square().sum(1)).clamp(0, 1)
            nearest = vertices[None, :-1] + fraction[:, :, None] * segments
            distance = torch.linalg.vector_norm(self.positions[:, None] - nearest, dim=2).min(1).values
        tool_overlap = (self.config.radius + self.config.tool_radius - distance).clamp_min(0)
        tool = 0.5 * self.config.contact_stiffness * tool_overlap.square().sum()
        kinetic = 0.5 * self.config.mass * self.velocities.square().sum()
        return dict(kinetic_J=float(kinetic), bond_J=float(bond), pair_contact_J=float(contact),
                    tool_contact_J=float(tool), mechanical_J=float(kinetic + bond + contact + tool))
