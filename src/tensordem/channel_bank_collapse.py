"""Channel bank collapse & width evolution via DEM multi-body ice–bankwall interaction.

Physical model
--------------
1. **Bankwall discretisation**: The bankwall is represented as a grid of bonded
   DEM blocks (2-D central-force network). Each block carries mass, cohesion *c*,
   and internal friction angle *φ*.

2. **Mohr–Coulomb shear failure**: For each inter-block bond the shear strength
   is  τ_max = c + σ_n · tan(φ),  where σ_n is the normal stress on the bond
   plane.  When the resolved shear traction exceeds τ_max the bond breaks
   irreversibly (block detachment).

3. **Ice–bankwall contact**: Floating ice blocks (concentration, size distribution)
   are advected by a prescribed ship-induced current.  Contact forces between ice
   particles and bankwall boundary nodes accumulate lateral pressure.

4. **Rankine earth-pressure anchor**: The at-rest lateral earth-pressure
   coefficient  K₀ = 1 − sin(φ)  provides the analytical baseline for the
   initial (pre-collapse) bankwall stress state.

5. **Channel width evolution**: Each detached block reduces the effective channel
   width.  The time history W(t) is tracked; physically W(t) must be monotonically
   non-increasing.

All quantities use SI units.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Rankine earth-pressure utilities (analytical anchor)
# ---------------------------------------------------------------------------

def rankine_k0(phi_deg: float) -> float:
    """At-rest earth-pressure coefficient  K₀ = 1 − sin(φ)."""
    phi = math.radians(phi_deg)
    return 1.0 - math.sin(phi)


def rankine_ka(phi_deg: float) -> float:
    """Active earth-pressure coefficient  Kₐ = tan²(45° − φ/2)."""
    phi = math.radians(phi_deg)
    return math.tan(math.pi / 4 - phi / 2) ** 2


def rankine_kp(phi_deg: float) -> float:
    """Passive earth-pressure coefficient  Kₚ = tan²(45° + φ/2)."""
    phi = math.radians(phi_deg)
    return math.tan(math.pi / 4 + phi / 2) ** 2


def rankine_lateral_pressure(z: float, gamma: float, phi_deg: float,
                             mode: str = "at_rest") -> float:
    """Lateral earth pressure σₕ = K · γ · z at depth *z*.

    Parameters
    ----------
    z : depth below ground surface (m), must be ≥ 0
    gamma : unit weight of soil (N/m³)
    phi_deg : internal friction angle (degrees)
    mode : 'at_rest', 'active', or 'passive'
    """
    if z < 0:
        raise ValueError("depth z must be non-negative")
    if gamma <= 0:
        raise ValueError("unit weight gamma must be positive")
    if mode == "at_rest":
        K = rankine_k0(phi_deg)
    elif mode == "active":
        K = rankine_ka(phi_deg)
    elif mode == "passive":
        K = rankine_kp(phi_deg)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return K * gamma * z


# ---------------------------------------------------------------------------
# Mohr–Coulomb shear failure
# ---------------------------------------------------------------------------

def mohr_coulomb_shear_strength(sigma_n: float, c: float, phi_deg: float) -> float:
    """Return τ_max = c + σ_n · tan(φ).

    σ_n is the normal stress (positive = compression).  Tensile σ_n reduces
    the shear capacity linearly; if σ_n is sufficiently negative the strength
    goes to zero (no tensile cohesion beyond *c*).
    """
    if c < 0:
        raise ValueError("cohesion c must be non-negative")
    phi = math.radians(phi_deg)
    tau = c + sigma_n * math.tan(phi)
    return max(tau, 0.0)


def mohr_coulomb_critical_normal_pressure(c: float, phi_deg: float,
                                           tau_applied: float) -> float:
    """Return the critical normal pressure σ_n,crit at which τ_applied = τ_max.

    From  τ = c + σ_n·tan(φ)  →  σ_n = (τ − c) / tan(φ).
    """
    if tau_applied < 0:
        raise ValueError("applied shear must be non-negative")
    phi = math.radians(phi_deg)
    tan_phi = math.tan(phi)
    if tan_phi < 1e-12:
        # Frictionless: any shear > c causes failure immediately
        return 0.0 if tau_applied <= c else float("-inf")
    return (tau_applied - c) / tan_phi


# ---------------------------------------------------------------------------
# Bankwall DEM block network
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BankwallConfig:
    """Configuration for the bankwall DEM block network.

    Parameters
    ----------
    nx, ny : number of blocks in horizontal (across-channel) and vertical directions
    block_size : characteristic block side length (m)
    density : block material density (kg/m³)
    E : Young's modulus for inter-block bonds (Pa)
    thickness : out-of-plane thickness (m)
    cohesion : Mohr–Coulomb cohesion *c* (Pa)
    friction_angle : internal friction angle φ (degrees)
    channel_width : initial channel width (m)
    water_depth : water depth adjacent to bankwall (m)
    """
    nx: int = 10
    ny: int = 8
    block_size: float = 0.5
    density: float = 2000.0
    E: float = 1e7
    thickness: float = 1.0
    cohesion: float = 5000.0
    friction_angle: float = 30.0
    channel_width: float = 50.0
    water_depth: float = 5.0

    def __post_init__(self) -> None:
        for name in ("nx", "ny"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 2:
                raise ValueError(f"{name} must be int >= 2")
        for name in ("block_size", "density", "E", "thickness",
                      "cohesion", "channel_width", "water_depth"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not (0 < self.friction_angle < 90):
            raise ValueError("friction_angle must be in (0, 90) degrees")


class BankwallNetwork:
    """2-D bonded block network representing one side of a channel bankwall.

    The network is a square central-force lattice (same topology as
    ``BoundaryNetwork``) with Mohr–Coulomb shear failure on each bond.
    The rightmost column of nodes represents the bankwall face exposed to
    ice pressure; the leftmost column is fixed (far-field ground).
    """

    def __init__(self, config: BankwallConfig) -> None:
        self.config = config
        dx = config.block_size
        dy = config.block_size
        length = config.nx * dx
        height = config.ny * dy

        # Node positions
        self.x = torch.tensor(
            [[i * dx, j * dy] for j in range(config.ny + 1) for i in range(config.nx + 1)],
            dtype=torch.float64,
        )
        self.u = torch.zeros_like(self.x)  # displacement
        self.v = torch.zeros_like(self.x)  # velocity

        # Mass (tributary area weighting)
        w = torch.ones(len(self.x), dtype=torch.float64)
        w[(self.x[:, 0] == 0) | (self.x[:, 0] == length)] *= 0.5
        w[(self.x[:, 1] == 0) | (self.x[:, 1] == height)] *= 0.5
        self.mass = w * config.density * config.thickness * dx * dy

        # Edges (axial + diagonal)
        edges: list[tuple[int, int]] = []
        ks: list[float] = []
        for j in range(config.ny + 1):
            for i in range(config.nx + 1):
                a = j * (config.nx + 1) + i
                if i < config.nx:
                    edges.append((a, a + 1))
                    edge_k = 3 * config.E * config.thickness / 4
                    if j in (0, config.ny):
                        edge_k *= 0.5
                    ks.append(edge_k)
                if j < config.ny:
                    edges.append((a, a + config.nx + 1))
                    edge_k = 3 * config.E * config.thickness / 4
                    if i in (0, config.nx):
                        edge_k *= 0.5
                    ks.append(edge_k)
                if i < config.nx and j < config.ny:
                    edges.extend(((a, a + config.nx + 2), (a + 1, a + config.nx + 1)))
                    diag_k = 3 * config.E * config.thickness / 8
                    ks.extend((diag_k, diag_k))

        self.edges = torch.tensor(edges, dtype=torch.long)
        self.k = torch.tensor(ks, dtype=torch.float64)

        # Edge normals
        d = self.x[self.edges[:, 1]] - self.x[self.edges[:, 0]]
        self.n = d / torch.linalg.vector_norm(d, dim=1)[:, None]

        # Fixed boundary (left edge + bottom)
        self.fixed = torch.zeros_like(self.x, dtype=torch.bool)
        self.fixed[self.x[:, 0] == 0, 0] = True  # fix x on left
        self.fixed[self.x[:, 1] == 0, 1] = True  # fix y on bottom

        # Right-face nodes (bankwall face)
        self.right = torch.where(self.x[:, 0] == length)[0]

        # Tributary areas for right-face nodes
        self.areas = torch.full((config.ny + 1,), dy * config.thickness, dtype=torch.float64)
        self.areas[0] *= 0.5
        self.areas[-1] *= 0.5

        # Mohr–Coulomb parameters per bond
        self.cohesion = config.cohesion
        self.friction_angle = config.friction_angle
        self.phi_rad = math.radians(config.friction_angle)
        self.tan_phi = math.tan(self.phi_rad)

        # Bond alive status
        self.bond_alive = torch.ones(len(self.edges), dtype=torch.bool)

        # Fracture tracking
        self.bond_failure_time = torch.full((len(self.edges),), -1.0, dtype=torch.float64)
        self.bond_failure_mode = torch.zeros(len(self.edges), dtype=torch.uint8)  # 0=ok, 1=shear, 2=tension

        # Energy accounting
        self.time = 0.0
        self.step_count = 0
        self.total_external_work = 0.0  # work done by external forces
        self.total_fracture_energy = 0.0  # energy released by bond breaking
        self.kinetic_energy = 0.0
        self.strain_energy = 0.0

        # Detached blocks tracking
        self.detached_blocks: list[dict[str, Any]] = []
        self.channel_width_history: list[tuple[float, float]] = [(0.0, config.channel_width)]

        # Penalty for ice contact
        self.penalty = 50 * config.E / length

        # Timestep
        rows = torch.zeros(len(self.x), dtype=torch.float64)
        rows.index_add_(0, self.edges[:, 0], self.k)
        rows.index_add_(0, self.edges[:, 1], self.k)
        self.dt = 0.15 * float(torch.sqrt(self.mass / rows.clamp_min(1e-12)).min())

    # ------------------------------------------------------------------
    # Internal force computation with Mohr–Coulomb check
    # ------------------------------------------------------------------

    def _compute_forces(self, ice_pressure: torch.Tensor | None = None) -> torch.Tensor:
        """Compute internal + external forces on all nodes.

        Parameters
        ----------
        ice_pressure : shape (N_right,) lateral pressure on right-face nodes (Pa)
        """
        f = torch.zeros_like(self.x)

        # Bond forces
        alive = self.bond_alive
        if alive.any():
            e = ((self.u[self.edges[:, 1]] - self.u[self.edges[:, 0]]) * self.n).sum(1)
            e = e * alive.to(torch.float64)
            z = (self.k * e)[:, None] * self.n
            f.index_add_(0, self.edges[:, 0], z)
            f.index_add_(0, self.edges[:, 1], -z)

        # Ice pressure on right face
        if ice_pressure is not None:
            f[self.right, 0] += ice_pressure * self.areas

        # Fixed node reactions (zero displacement enforced)
        f = torch.where(self.fixed, 0.0, f)
        return f

    def _compute_strain_energy(self) -> float:
        """Compute current strain energy stored in alive bonds."""
        e = ((self.u[self.edges[:, 1]] - self.u[self.edges[:, 0]]) * self.n).sum(1)
        e = e * self.bond_alive.to(torch.float64)
        return float(0.5 * (self.k * e.square()).sum())

    def _check_mohr_coulomb(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check Mohr–Coulomb shear failure on all alive bonds.

        Returns
        -------
        fail_mask : boolean mask of bonds that should break
        energy_release : energy released by each breaking bond (J)
        """
        n_edges = len(self.edges)
        fail_mask = torch.zeros(n_edges, dtype=torch.bool)
        energy_release = torch.zeros(n_edges, dtype=torch.float64)

        if not self.bond_alive.any():
            return fail_mask, energy_release

        # Normal and shear components of bond deformation
        du = self.u[self.edges[:, 1]] - self.u[self.edges[:, 0]]
        normal_disp = (du * self.n).sum(1)  # axial extension
        shear_disp_vec = du - normal_disp[:, None] * self.n
        shear_disp = torch.linalg.vector_norm(shear_disp_vec, dim=1)

        # Bond length
        bond_len = torch.linalg.vector_norm(
            self.x[self.edges[:, 1]] - self.x[self.edges[:, 0]], dim=1
        )

        # Normal stress (compression positive)
        sigma_n = -self.k * normal_disp / (self.config.block_size * self.config.thickness)

        # Shear stress
        tau = self.k * shear_disp / (self.config.block_size * self.config.thickness)

        # Mohr–Coulomb strength
        tau_max = self.cohesion + sigma_n.clamp_min(0) * self.tan_phi

        # Failure: shear stress exceeds strength (only for alive bonds)
        shear_fail = (tau > tau_max) & self.bond_alive

        # Also check tensile failure (extension > 1% of block size)
        tensile_fail = (normal_disp > 0.01 * self.config.block_size) & self.bond_alive

        fail_mask = shear_fail | tensile_fail

        # Energy release for breaking bonds: strain energy stored in the bond
        # E_bond = 0.5 * k * extension²
        extension = torch.abs(normal_disp) + shear_disp
        energy_release = 0.5 * self.k * extension.square()

        return fail_mask, energy_release

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def step(self, ice_pressure: torch.Tensor | None = None) -> dict[str, Any]:
        """Advance one explicit time step.

        Parameters
        ----------
        ice_pressure : lateral pressure on right-face nodes (Pa), shape (N_right,)
        """
        dt = self.dt

        # Store pre-step strain energy
        strain_before = self._compute_strain_energy()

        f1 = self._compute_forces(ice_pressure)
        a1 = f1 / self.mass[:, None]
        v_half = self.v + 0.5 * dt * a1
        du = dt * v_half
        self.u += du

        # External work done this step
        if ice_pressure is not None:
            ext_work = float((ice_pressure * self.areas * du[self.right, 0]).sum())
            self.total_external_work += ext_work

        # Check Mohr–Coulomb failure
        fail_mask, energy_release = self._check_mohr_coulomb()
        newly_broken = fail_mask & self.bond_alive
        if newly_broken.any():
            # Energy released by breaking bonds
            released = float(energy_release[newly_broken].sum())
            self.total_fracture_energy += released

            self.bond_alive[newly_broken] = False
            self.bond_failure_time[newly_broken] = self.time
            self.bond_failure_mode[newly_broken] = 1  # shear failure

        f2 = self._compute_forces(ice_pressure)
        a2 = f2 / self.mass[:, None]
        self.v = v_half + 0.5 * dt * a2

        self.time += dt
        self.step_count += 1

        # Energy
        self.kinetic_energy = float(0.5 * (self.mass[:, None] * self.v.square()).sum())
        self.strain_energy = self._compute_strain_energy()

        # Track detached blocks (bonds broken on right face)
        right_edges = torch.isin(self.edges[:, 0], self.right) | torch.isin(self.edges[:, 1], self.right)
        new_detached = newly_broken & right_edges
        if new_detached.any():
            n_detached = int(new_detached.sum().item())
            block_area = self.config.block_size ** 2 * self.config.thickness
            block_mass = block_area * self.config.density
            # Approximate drop height = half block size
            drop_h = 0.5 * self.config.block_size
            pe = block_mass * 9.81 * drop_h * n_detached
            self.detached_blocks.append({
                "time": self.time,
                "n_blocks": n_detached,
                "potential_energy_J": pe,
                "fracture_energy_J": float(energy_release[new_detached].sum()),
            })
            # Reduce channel width
            width_reduction = n_detached * self.config.block_size * 0.1  # empirical factor
            current_width = self.channel_width_history[-1][1] - width_reduction
            self.channel_width_history.append((self.time, max(current_width, 0.0)))

        return {
            "time": self.time,
            "n_broken": int((~self.bond_alive).sum().item()),
            "n_newly_broken": int(newly_broken.sum().item()),
            "kinetic_energy_J": self.kinetic_energy,
            "strain_energy_J": self.strain_energy,
            "channel_width_m": self.channel_width_history[-1][1],
        }

    def run(self, n_steps: int,
            ice_pressure_fn: callable | None = None) -> list[dict[str, Any]]:
        """Run simulation for *n_steps* steps.

        Parameters
        ----------
        ice_pressure_fn : callable(step, time) -> torch.Tensor of shape (N_right,)
            If None, a linearly increasing pressure profile is used.
        """
        history = []
        for s in range(n_steps):
            if ice_pressure_fn is not None:
                ip = ice_pressure_fn(s, self.time)
            else:
                # Linear ramp: 0 → 10 kPa over the simulation
                max_p = 10000.0
                ip = torch.full((len(self.right),), max_p * min(1.0, self.time / 10.0),
                                dtype=torch.float64)
            info = self.step(ip)
            history.append(info)
        return history

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "tensordem.bankwall-network/1",
            "config": {
                "nx": self.config.nx, "ny": self.config.ny,
                "block_size": self.config.block_size,
                "density": self.config.density, "E": self.config.E,
                "thickness": self.config.thickness,
                "cohesion": self.config.cohesion,
                "friction_angle": self.config.friction_angle,
                "channel_width": self.config.channel_width,
                "water_depth": self.config.water_depth,
            },
            "time": self.time,
            "step_count": self.step_count,
            "n_broken_bonds": int((~self.bond_alive).sum().item()),
            "channel_width_history": self.channel_width_history,
            "detached_blocks": self.detached_blocks,
            "energy": {
                "external_work_J": self.total_external_work,
                "fracture_energy_J": self.total_fracture_energy,
                "kinetic_energy_J": self.kinetic_energy,
                "strain_energy_J": self.strain_energy,
            },
        }


# ---------------------------------------------------------------------------
# Ice block field (simplified DEM particles)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IceFieldConfig:
    """Configuration for the floating ice field."""
    n_ice: int = 20
    ice_radius: float = 1.0
    ice_density: float = 917.0
    concentration: float = 0.3  # area fraction
    current_speed: float = 0.5  # ship-induced current (m/s)
    channel_width: float = 50.0
    water_depth: float = 5.0


class IceField:
    """Simplified ice block field advected by a prescribed current.

    Each ice block is a circle that exerts contact pressure on the bankwall
    when it overlaps the bankwall face.
    """

    def __init__(self, config: IceFieldConfig) -> None:
        self.config = config
        rng = torch.Generator().manual_seed(42)
        # Random positions within channel
        self.positions = torch.zeros(config.n_ice, 2, dtype=torch.float64)
        self.positions[:, 0] = torch.rand(config.n_ice, generator=rng) * config.channel_width
        self.positions[:, 1] = torch.rand(config.n_ice, generator=rng) * config.water_depth
        self.velocities = torch.full((config.n_ice, 2), config.current_speed, dtype=torch.float64)
        self.velocities[:, 1] = 0.0  # only x-direction current

    def compute_bankwall_pressure(self, bankwall_x: float,
                                   bankwall_nodes_y: torch.Tensor,
                                   node_areas: torch.Tensor) -> torch.Tensor:
        """Compute lateral pressure on bankwall nodes from ice contact.

        Parameters
        ----------
        bankwall_x : x-coordinate of bankwall face
        bankwall_nodes_y : y-coordinates of bankwall face nodes
        node_areas : tributary areas of bankwall face nodes
        """
        pressure = torch.zeros(len(bankwall_nodes_y), dtype=torch.float64)
        for i in range(self.config.n_ice):
            dx = bankwall_x - self.positions[i, 0]
            if dx < self.config.ice_radius:  # ice block touching bankwall
                overlap = self.config.ice_radius - dx
                if overlap > 0:
                    # Distribute force to nearby nodes
                    dy = (bankwall_nodes_y - self.positions[i, 1]).abs()
                    influence = torch.exp(-dy ** 2 / (2 * self.config.ice_radius ** 2))
                    force = overlap * 1e5  # simple linear contact law
                    pressure += force * influence / node_areas.clamp_min(1e-12)
        return pressure

    def step(self, dt: float) -> None:
        """Advect ice blocks."""
        self.positions += self.velocities * dt
        # Wrap around channel
        self.positions[:, 0] = self.positions[:, 0] % self.config.channel_width


# ---------------------------------------------------------------------------
# Coupled simulation: ice field + bankwall
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelBankCollapseConfig:
    """Full simulation configuration."""
    bankwall: BankwallConfig = field(default_factory=BankwallConfig)
    ice_field: IceFieldConfig = field(default_factory=IceFieldConfig)
    n_steps: int = 500
    output_interval: int = 10


class ChannelBankCollapseSim:
    """Coupled ice–bankwall simulation with channel width evolution tracking."""

    def __init__(self, config: ChannelBankCollapseConfig) -> None:
        self.config = config
        if isinstance(config.bankwall, dict):
            self.bankwall = BankwallConfig(**config.bankwall)
        else:
            self.bankwall = BankwallConfig(
                nx=config.bankwall.nx, ny=config.bankwall.ny,
                block_size=config.bankwall.block_size,
                density=config.bankwall.density, E=config.bankwall.E,
                thickness=config.bankwall.thickness,
                cohesion=config.bankwall.cohesion,
                friction_angle=config.bankwall.friction_angle,
                channel_width=config.bankwall.channel_width,
                water_depth=config.bankwall.water_depth,
            )
        self.bw = BankwallNetwork(self.bankwall)
        self.ice = IceField(config.ice_field)
        self.history: list[dict[str, Any]] = []

    def run(self) -> list[dict[str, Any]]:
        """Run the coupled simulation."""
        n_steps = self.config.n_steps
        interval = self.config.output_interval

        for s in range(n_steps):
            # Compute ice pressure on bankwall
            bw_x = self.bankwall.nx * self.bankwall.block_size
            bw_nodes_y = self.bw.x[self.bw.right, 1]
            ice_pressure = self.ice.compute_bankwall_pressure(
                bw_x, bw_nodes_y, self.bw.areas
            )

            info = self.bw.step(ice_pressure)
            self.ice.step(self.bw.dt)

            if s % interval == 0 or s == n_steps - 1:
                self.history.append({
                    "step": s,
                    **info,
                    "n_ice_blocks": self.config.ice_field.n_ice,
                })

        return self.history

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "tensordem.channel-bank-collapse/1",
            "bankwall": self.bw.snapshot(),
            "history": self.history,
        }
