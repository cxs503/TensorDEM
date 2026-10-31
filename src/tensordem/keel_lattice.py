"""Bonded-particle (lattice) ice-ridge keel on a seabed foundation.

The keel is discretised as a 2-D triangular lattice of bonded disks (unit
out-of-plane thickness, plane-strain idealisation of a keel ribbon between two
grounding points).  It is the discrete-level counterpart of the Kerr beam model
in :mod:`tensordem.kerr_foundation`:

* **bonds** carry a central normal force ``k_n (l - l0)`` and break irreversibly
  once the bond strain exceeds ``breaking_strain``;
* **seabed foundation** springs act on the bottom row, with stiffness per
  particle ``k_f * s`` so that the distributed modulus equals ``k_f``;
* **hull contact** is a rigid flat punch (the hull/STL cross-section) advanced
  into the top surface under *displacement control*, the total reaction force
  and the contact-pressure distribution are recorded.

Micro-parameters are calibrated, not assumed: :func:`uniaxial_strain_test` and
:func:`cantilever_test` solve the linear elastic lattice exactly (sparse static
solve of the bond Hessian) and return the effective 2-D modulus ``E_2D`` and the
bending stiffness ``EI`` of the *discretised* keel.  Those calibrated numbers
are what the continuum model is fed with, so the DEM-vs-Kerr comparison is a
cross-model check rather than a tautology.

Units: SI, forces in N, lengths in m, masses in kg; all state in
``torch.float64``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch

KEEL_SCHEMA = "tensordem.keel-lattice/1"


# ---------------------------------------------------------------------------
# lattice geometry
# ---------------------------------------------------------------------------
def triangular_lattice(length: float, thickness: float, radius: float,
                       *, jitter: float = 0.0, seed: int = 0) -> tuple[np.ndarray, float]:
    """Triangular lattice of touching disks filling ``length x thickness``.

    Returns particle centres ``(N, 2)`` and the realised bond length ``s``.
    """
    for name, value in (("length", length), ("thickness", thickness), ("radius", radius)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not (0.0 <= jitter < 0.25):
        raise ValueError("jitter must lie in [0, 0.25)")
    s = 2.0 * radius
    dy = s * math.sqrt(3.0) / 2.0
    n_row = max(2, int(round(thickness / dy)) + 1)
    n_col = max(2, int(math.ceil(length / s)) + 1)
    rows = []
    for j in range(n_row):
        x0 = 0.0 if j % 2 == 0 else 0.5 * s
        x = x0 + np.arange(n_col) * s
        x = x[x <= length + 1e-12]
        if x.size == 0:
            continue
        rows.append(np.column_stack([x, np.full_like(x, j * dy)]))
    pos = np.vstack(rows)
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        pos = pos + rng.normal(scale=jitter * s * 0.5, size=pos.shape)
    return pos, s


def neighbour_pairs(pos: np.ndarray, bond_length: float, *, tolerance: float = 0.05) -> np.ndarray:
    """Neighbour list: pairs closer than ``(1 + tolerance) * bond_length``."""
    from scipy.spatial import cKDTree

    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("pos must have shape (N, 2)")
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=bond_length * (1.0 + tolerance), output_type="ndarray")
    if pairs.size == 0:
        raise ValueError("no bonds found -- check the lattice spacing")
    pairs = np.sort(pairs.astype(np.int64), axis=1)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


# ---------------------------------------------------------------------------
# exact linear calibration of the lattice
# ---------------------------------------------------------------------------
def _bond_hessian(pos: torch.Tensor, pairs: torch.Tensor, stiffness: float):
    """Sparse bond Hessian ``d2E/dx2`` of the axial (central-force) bonds."""
    from scipy.sparse import coo_matrix

    d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    length = torch.linalg.norm(d, dim=1)
    n = d / length[:, None]
    block = (stiffness * torch.einsum("pi,pj->pij", n, n)).numpy()
    idx_i = pairs[:, 0].numpy()
    idx_j = pairs[:, 1].numpy()
    rows, cols, vals = [], [], []
    for a in range(2):
        for c in range(2):
            b = block[:, a, c]
            rows += [2 * idx_i + a, 2 * idx_j + a, 2 * idx_i + a, 2 * idx_j + a]
            cols += [2 * idx_i + c, 2 * idx_j + c, 2 * idx_j + c, 2 * idx_i + c]
            vals += [b, b, -b, -b]
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    return coo_matrix((vals, (rows, cols)), shape=(2 * pos.shape[0], 2 * pos.shape[0])).tocsr()


def _static_solve(hessian, rhs: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Solve ``K u = f`` on the free DOFs (others prescribed to zero)."""
    from scipy.sparse.linalg import spsolve

    u = np.zeros(hessian.shape[0])
    k_ff = hessian[free][:, free].tocsc()
    sol = spsolve(k_ff, rhs[free])
    if not np.all(np.isfinite(sol)):
        raise ValueError("singular lattice stiffness -- check the bonded network")
    u[free] = sol
    return u


def uniaxial_strain_test(pos: np.ndarray, pairs: np.ndarray, stiffness: float,
                         strain: float = 1e-4) -> dict[str, Any]:
    """Exact effective 2-D modulus ``E_2D`` of the bonded lattice.

    Left column clamped, right column moved by ``strain * width``, top/bottom
    free.  ``E_2D = sigma / strain`` with ``sigma = sum(Fx_right) / height`` is
    the plane-stress modulus (force per unit width) used by the continuum beam.
    """
    if not math.isfinite(strain) or strain <= 0.0:
        raise ValueError("strain must be finite and positive")
    pos_t = torch.as_tensor(pos, dtype=torch.float64)
    pairs_t = torch.as_tensor(pairs, dtype=torch.int64)
    x_min, x_max = float(pos[:, 0].min()), float(pos[:, 0].max())
    width = x_max - x_min
    height = float(pos[:, 1].max() - pos[:, 1].min())
    if width <= 0.0 or height <= 0.0:
        raise ValueError("lattice must have positive extent")
    left = np.where(np.abs(pos[:, 0] - x_min) < 1e-9)[0]
    right = np.where(np.abs(pos[:, 0] - x_max) < 1e-9)[0]
    hessian = _bond_hessian(pos_t, pairs_t, stiffness)
    rhs = np.zeros(2 * pos.shape[0])
    move = strain * width
    # the right-column displacement is imposed by elimination (no penalty bias)
    freed = np.ones(2 * pos.shape[0], dtype=bool)
    for node in left:
        freed[2 * node] = False
        freed[2 * node + 1] = False
    for node in right:
        freed[2 * node] = False          # horizontal pull is prescribed
    low_right = right[np.argmin(pos[right, 1])]
    freed[2 * low_right + 1] = False      # remove the remaining y translation
    free_idx = np.where(freed)[0]
    fixed_idx = np.where(~freed)[0]
    u_fixed = np.zeros(2 * pos.shape[0])
    for node in right:
        u_fixed[2 * node] = move
    q = np.zeros(2 * pos.shape[0])
    rhs_r = -hessian[free_idx][:, fixed_idx] @ u_fixed[fixed_idx]
    from scipy.sparse.linalg import spsolve

    sol = spsolve(hessian[free_idx][:, free_idx].tocsc(), rhs_r)
    if not np.all(np.isfinite(sol)):
        raise ValueError("singular lattice stiffness -- check the bonded network")
    u = u_fixed
    u[free_idx] = sol
    residual = hessian @ u - q
    reaction_x = float(residual[2 * right].sum())
    stress = reaction_x / height
    return {
        "test": "uniaxial_strain",
        "strain": strain,
        "width_m": width,
        "height_m": height,
        "reaction_force_N": reaction_x,
        "stress_N_per_m": stress,
        "E_2D_N_per_m": stress / strain,
        "bond_stiffness_N_per_m": stiffness,
    }


def cantilever_test(pos: np.ndarray, pairs: np.ndarray, stiffness: float,
                    tip_load: float = 1.0) -> dict[str, Any]:
    """Exact bending stiffness ``EI`` of the discretised keel ribbon.

    Left edge clamped; vertical point load ``tip_load`` (per metre width) on the
    top node of the free right edge.  ``EI = P L^3 / (3 delta)`` and
    ``E_bend = EI / (thickness^3 / 12)``.
    """
    if not math.isfinite(tip_load) or tip_load == 0.0:
        raise ValueError("tip_load must be finite and nonzero")
    pos_t = torch.as_tensor(pos, dtype=torch.float64)
    pairs_t = torch.as_tensor(pairs, dtype=torch.int64)
    x_min, x_max = float(pos[:, 0].min()), float(pos[:, 0].max())
    length = x_max - x_min
    thickness = float(pos[:, 1].max() - pos[:, 1].min())
    left = np.where(np.abs(pos[:, 0] - x_min) < 1e-9)[0]
    freed = np.ones(2 * pos.shape[0], dtype=bool)
    for node in left:
        freed[2 * node] = False
        freed[2 * node + 1] = False
    free_idx = np.where(freed)[0]
    tip_nodes = np.where(np.abs(pos[:, 0] - x_max) < 1e-9)[0]
    top = int(tip_nodes[np.argmax(pos[tip_nodes, 1])])
    rhs = np.zeros(2 * pos.shape[0])
    rhs[2 * top + 1] = -abs(tip_load)
    hessian = _bond_hessian(pos_t, pairs_t, stiffness)
    u = _static_solve(hessian, rhs, free_idx)
    tip_deflection = abs(float(u[2 * top + 1]))
    ei = abs(tip_load) * length**3 / (3.0 * tip_deflection)
    return {
        "test": "cantilever",
        "length_m": length,
        "thickness_m": thickness,
        "tip_load_N": abs(tip_load),
        "tip_deflection_m": tip_deflection,
        "EI_Nm2": ei,
        "E_bend_N_per_m": ei / (thickness**3 / 12.0),
        "tip_node": top,
    }


def calibrate_bond_stiffness(length: float, thickness: float, radius: float,
                             target_E_2D: float) -> dict[str, Any]:
    """Find ``k_n`` so that the lattice uniaxial modulus equals ``target_E_2D``.

    Starts from the affine (Cauchy-Born) estimate ``E_2D ~ 0.87 k_n`` of a
    triangular central-force lattice; one exact uniaxial solve rescales it (the
    response is linear in ``k_n``) and a second solve verifies the result.  The
    cantilever test then returns the bending ``EI`` of the same discretisation.
    """
    pos, s = triangular_lattice(length, thickness, radius)
    pairs = neighbour_pairs(pos, s)
    k_guess = target_E_2D / 0.87
    first = uniaxial_strain_test(pos, pairs, k_guess)
    k_n = k_guess * target_E_2D / first["E_2D_N_per_m"]
    check = uniaxial_strain_test(pos, pairs, k_n)
    cant = cantilever_test(pos, pairs, k_n)
    return {
        "test": "calibrate_bond_stiffness",
        "bond_length_m": s,
        "particles": int(pos.shape[0]),
        "bonds": int(pairs.shape[0]),
        "target_E_2D_N_per_m": target_E_2D,
        "affine_guess_N_per_m": k_guess,
        "bond_stiffness_N_per_m": k_n,
        "realised_E_2D_N_per_m": check["E_2D_N_per_m"],
        "relative_error": abs(check["E_2D_N_per_m"] - target_E_2D) / target_E_2D,
        "cantilever": cant,
    }


# ---------------------------------------------------------------------------
# dynamic bonded-particle keel
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KeelLatticeConfig:
    """Configuration of the bonded-particle keel + rigid hull punch."""

    length: float = 12.0
    thickness: float = 1.5
    radius: float = 0.0102
    density: float = 917.0
    bond_stiffness: float = 5.75e9
    breaking_strain: float = 0.010
    foundation_modulus: float = 2.0e7
    contact_stiffness_factor: float = 20.0
    contact_damping_factor: float = 2.0
    local_damping: float = 0.70
    viscous_damping: float = 0.30
    punch_width: float = 1.0
    punch_center: float | None = None
    mass_scale: float = 1.0
    dt: float | None = None
    jitter: float = 0.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("length", "thickness", "radius", "density", "bond_stiffness",
                     "breaking_strain", "punch_width"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.foundation_modulus) or self.foundation_modulus < 0.0:
            raise ValueError("foundation_modulus must be finite and nonnegative")
        if not math.isfinite(self.contact_stiffness_factor) or self.contact_stiffness_factor <= 0:
            raise ValueError("contact_stiffness_factor must be finite and positive")
        for name in ("local_damping", "viscous_damping"):
            if not (0.0 <= getattr(self, name) < 2.0):
                raise ValueError(f"{name} must lie in [0, 2)")
        if not math.isfinite(self.mass_scale) or self.mass_scale <= 0.0:
            raise ValueError("mass_scale must be finite and positive")
        if self.dt is not None and (not math.isfinite(self.dt) or self.dt <= 0.0):
            raise ValueError("dt must be finite and positive")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")

    @property
    def particle_mass(self) -> float:
        return self.density * math.pi * self.radius**2

    @property
    def support_mass(self) -> float:
        return self.particle_mass * self.mass_scale


class KeelLatticeDEM:
    """Bonded triangular lattice keel with seabed springs and a rigid punch."""

    def __init__(self, config: KeelLatticeConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.float64
        pos_np, s = triangular_lattice(config.length, config.thickness, config.radius,
                                       jitter=config.jitter, seed=config.seed)
        pairs_np = neighbour_pairs(pos_np, s)
        self.bond_length = s
        self.pos = torch.as_tensor(pos_np, dtype=self.dtype, device=self.device)
        self.pairs = torch.as_tensor(pairs_np, dtype=torch.int64, device=self.device)
        n = int(self.pos.shape[0])
        self.vel = torch.zeros_like(self.pos)
        self.mass = torch.full((n,), config.support_mass, dtype=self.dtype, device=self.device)
        self.bond_rest = torch.linalg.norm(
            self.pos[self.pairs[:, 1]] - self.pos[self.pairs[:, 0]], dim=1).clone()
        self.alive = torch.ones(int(self.pairs.shape[0]), dtype=torch.bool, device=self.device)
        y_min = float(self.pos[:, 1].min())
        self.bottom = (self.pos[:, 1] - y_min).abs() < 1e-9
        ground_k = torch.zeros(n, dtype=self.dtype, device=self.device)
        ground_k[self.bottom] = config.foundation_modulus * s
        self.ground_stiffness = ground_k
        self.ground_rest = self.pos[:, 1].clone()
        self.ground_damping = 2.0 * config.viscous_damping * torch.sqrt(ground_k * self.mass)
        self.fixed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.k_contact = config.contact_stiffness_factor * config.bond_stiffness
        self.c_contact = config.contact_damping_factor * math.sqrt(
            config.particle_mass * config.bond_stiffness)
        self.c_bond = 2.0 * config.viscous_damping * math.sqrt(
            0.5 * config.particle_mass * config.bond_stiffness)
        x_mid = (float(self.pos[:, 0].min()) + float(self.pos[:, 0].max())) / 2.0
        self.punch_center = x_mid if config.punch_center is None else float(config.punch_center)
        self.punch_face_y = float(self.pos[:, 1].max()) + config.radius + 1e-3
        self.initial_face_y = self.punch_face_y
        self.reaction = 0.0
        self.contact_forces = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.contact_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.broken_bonds = 0
        self.step_count = 0
        self.external_forces = torch.zeros_like(self.pos)
        self.pin_center = False
        if config.dt is None:
            # explicit stability: stiffest local mode of the bond network *and*
            # of the punch penalty contact, whichever governs
            k_max = max(7.0 * config.bond_stiffness, self.k_contact)
            dt = 0.1 * math.sqrt(config.support_mass / k_max)
            ground_max = float(ground_k.max())
            if ground_max > 0.0:
                dt = min(dt, 0.1 * math.sqrt(config.support_mass / ground_max))
            self.dt = dt
        else:
            self.dt = float(config.dt)

    # -- diagnostics -------------------------------------------------------
    @property
    def particles(self) -> int:
        return int(self.pos.shape[0])

    @property
    def bonds(self) -> int:
        return int(self.pairs.shape[0])

    @property
    def total_bond_strain(self) -> float:
        d = self.pos[self.pairs[:, 1]] - self.pos[self.pairs[:, 0]]
        strain = (torch.linalg.norm(d, dim=1) - self.bond_rest) / self.bond_rest
        return float(strain[self.alive].abs().max()) if bool(self.alive.any()) else 0.0

    @property
    def penetration(self) -> float:
        """Punch face advance below its initial position [m]."""
        return max(0.0, self.initial_face_y - self.punch_face_y)

    @property
    def peak_velocity(self) -> float:
        return float(self.vel.abs().max())

    def elastic_energy(self) -> float:
        d = self.pos[self.pairs[:, 1]] - self.pos[self.pairs[:, 0]]
        ext = torch.linalg.norm(d, dim=1) - self.bond_rest
        e_bond = 0.5 * self.config.bond_stiffness * (ext**2)[self.alive].sum()
        dy = self.pos[:, 1] - self.ground_rest
        e_ground = 0.5 * (self.ground_stiffness * dy**2).sum()
        return float(e_bond + e_ground)

    # -- force evaluation --------------------------------------------------
    def _bond_forces(self) -> torch.Tensor:
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        d = self.pos[j] - self.pos[i]
        dist = torch.linalg.norm(d, dim=1).clamp_min(1e-30)
        n = d / dist[:, None]
        ext = dist - self.bond_rest
        strain = ext / self.bond_rest
        intact = self.alive
        vrel = ((self.vel[j] - self.vel[i]) * n).sum(dim=1)
        magnitude = self.config.bond_stiffness * ext + self.c_bond * vrel
        magnitude = torch.where(intact, magnitude, torch.zeros_like(magnitude))
        overlap = (self.bond_rest - dist).clamp_min(0.0)
        force_pair = torch.where(intact, magnitude, -self.k_contact * overlap)
        f = torch.zeros_like(self.pos)
        f = f.index_add(0, i, force_pair[:, None] * n)
        f = f.index_add(0, j, -force_pair[:, None] * n)
        if bool(intact.any()):
            broken = intact & (strain > self.config.breaking_strain)
            if bool(broken.any()):
                self.alive = self.alive & ~broken
                self.broken_bonds += int(broken.sum())
        return f

    def _foundation_forces(self) -> torch.Tensor:
        f = torch.zeros_like(self.pos)
        if self.config.foundation_modulus <= 0.0:
            return f
        dy = self.pos[:, 1] - self.ground_rest
        fy = -(self.ground_stiffness * dy + self.ground_damping * self.vel[:, 1])
        fy = torch.where(self.ground_stiffness > 0, fy, torch.zeros_like(fy))
        f[:, 1] += fy
        return f

    def punch_mask(self) -> torch.Tensor:
        """Particles inside the punch footprint and above its face."""
        r = self.config.radius
        inside = (self.pos[:, 0] - self.punch_center).abs() <= (
            self.config.punch_width / 2.0 + r)
        # tolerance so that particles projected exactly onto the face stay active
        tolerance = 1e-12 + 1e-12 * abs(self.punch_face_y)
        above = (self.pos[:, 1] + r) >= (self.punch_face_y - tolerance)
        return inside & above

    def _apply_punch_constraint(self) -> None:
        """Rigid punch: project penetrating particles onto the face, block upward.

        The punch is a rigid body moving downwards, so a penalty spring is
        avoided (it would dominate the explicit stable time step).  The reaction
        force is read from the *conservative* forces of the constrained
        particles once the relaxation has converged (see :meth:`measure_reaction`).
        """
        mask = self.punch_mask()
        self.contact_mask = mask
        if not bool(mask.any()):
            return
        face = self.punch_face_y - self.config.radius
        self.pos[mask, 1] = torch.minimum(self.pos[mask, 1], torch.full_like(
            self.pos[mask, 1], face))
        self.vel[mask, 1] = torch.minimum(self.vel[mask, 1],
                                          torch.zeros_like(self.vel[mask, 1]))

    def measure_reaction(self) -> float:
        """Punch reaction [N per m width] from the constrained particles."""
        mask = self.contact_mask
        if not bool(mask.any()):
            self.reaction = 0.0
            return 0.0
        conservative = self._bond_forces() + self._foundation_forces()
        self.contact_forces = torch.zeros(self.particles, dtype=self.dtype, device=self.device)
        # the ice pushes the punch up with the force that the constraint has to
        # balance, i.e. the net conservative force on the contact particles
        self.contact_forces[mask] = conservative[mask, 1]
        self.reaction = float(conservative[mask, 1].sum())
        return self.reaction

    def _damping_forces(self, conservative: torch.Tensor) -> torch.Tensor:
        alpha = self.config.local_damping
        if alpha == 0.0:
            return torch.zeros_like(self.pos)
        direction = torch.where(self.vel >= 0, 1.0, -1.0)
        return -alpha * conservative.abs() * direction

    def forces(self) -> torch.Tensor:
        conservative = self._bond_forces() + self._foundation_forces()
        total = conservative + self._damping_forces(conservative) + self.external_forces
        total[self.fixed] = 0.0
        return total

    def step(self, steps: int = 1) -> None:
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        for _ in range(steps):
            accel = self.forces() / self.mass[:, None]
            self.vel = self.vel + accel * self.dt
            self.vel[self.fixed] = 0.0
            self.pos = self.pos + self.vel * self.dt
            self._apply_punch_constraint()
            self.step_count += 1

    def kinetic_energy(self) -> float:
        return float(0.5 * (self.mass[:, None] * self.vel**2).sum())

    def relax(self, max_steps: int = 600, energy_tolerance: float = 1e-5,
              min_steps: int = 10) -> dict[str, Any]:
        """Dynamic relaxation with kinetic damping (Barnes-style).

        Velocities are zeroed whenever the total kinetic energy passes a local
        maximum, which damps every mode without needing the stiffness spectrum;
        convergence is declared when ``KE / elastic energy <= energy_tolerance``.
        """
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not math.isfinite(energy_tolerance) or energy_tolerance <= 0.0:
            raise ValueError("energy_tolerance must be finite and positive")
        used = 0
        converged = False
        best = math.inf
        resets = 0
        for _ in range(max_steps):
            self.step(1)
            used += 1
            if used < min_steps:
                continue
            ke = self.kinetic_energy()
            if ke < best:
                best = ke
            else:
                self.vel.zero_()
                best = math.inf
                resets += 1
            if ke <= energy_tolerance * max(self.elastic_energy(), 1e-30):
                converged = True
                break
        self.measure_reaction()
        return {
            "steps": used,
            "kinetic_resets": resets,
            "peak_velocity_m_per_s": self.peak_velocity,
            "converged": converged,
            "kinetic_energy_J": self.kinetic_energy(),
            "elastic_energy_J": self.elastic_energy(),
        }

    # -- loading -----------------------------------------------------------
    def set_punch_face(self, face_y: float) -> None:
        self.punch_face_y = float(face_y)

    def advance_punch(self, depth: float, relax_steps: int = 400,
                      energy_tolerance: float = 1e-5) -> dict[str, Any]:
        """Displacement-controlled punch advance of ``depth`` then relaxation."""
        if not math.isfinite(depth) or depth < 0.0:
            raise ValueError("depth must be finite and nonnegative")
        self.punch_face_y -= depth
        info = self.relax(max_steps=relax_steps, energy_tolerance=energy_tolerance)
        return {
            "penetration_m": self.penetration,
            "reaction_N_per_m": self.reaction,
            "broken_bonds": self.broken_bonds,
            "relax": info,
        }

    def pressure_profile(self, bins: int = 61) -> dict[str, list[float]]:
        """Contact pressure of the punch [N/m^2] binned along the keel."""
        if not isinstance(bins, int) or bins < 2:
            raise ValueError("bins must be an integer >= 2")
        lo = float(self.pos[:, 0].min())
        hi = float(self.pos[:, 0].max())
        edges = torch.linspace(lo, hi, bins + 1, dtype=self.dtype, device=self.device)
        centres = 0.5 * (edges[:-1] + edges[1:])
        width = float(edges[1] - edges[0])
        idx = torch.bucketize(self.pos[:, 0], edges[1:-1])
        weight = self.contact_forces.clamp_min(0.0)
        sums = torch.zeros(bins, dtype=self.dtype, device=self.device)
        sums = sums.index_add(0, idx, weight)
        return {
            "x_m": centres.tolist(),
            "pressure_Pa": (sums / width).tolist(),
            "bin_width_m": [width] * bins,
        }

    def bottom_settlement_profile(self) -> dict[str, Any]:
        """Settlement of the bottom row (the foundation deflection ``w``) [m].

        The bottom row rests directly on the seabed springs, so its settlement is
        the quantity that the continuum beam model calls the beam deflection at
        the foundation level; at the punch centre (zero slope) it equals the
        neutral-axis deflection, which makes it the right DEM observable for the
        DEM-vs-Kerr comparison -- it is *not* contaminated by the local
        indentation of the hull into the top surface.
        """
        idx = self.bottom.nonzero().flatten()
        x = self.pos[idx, 0]
        w = self.ground_rest[idx] - self.pos[idx, 1]
        order = torch.argsort(x)
        return {"x_m": x[order].tolist(), "settlement_m": w[order].tolist()}

    def beam_deflection(self, x: float) -> float:
        """Interpolated bottom-row settlement at ``x`` [m]."""
        prof = self.bottom_settlement_profile()
        return float(np.interp(float(x), prof["x_m"], prof["settlement_m"]))

    def midplane_settlement_profile(self, bins: int = 61) -> dict[str, Any]:
        """Mean settlement of the particles nearest mid-thickness (beam axis)."""
        if not isinstance(bins, int) or bins < 2:
            raise ValueError("bins must be an integer >= 2")
        y_mid = 0.5 * (float(self.pos[:, 1].max()) + float(self.pos[:, 1].min()))
        band = (self.pos[:, 1] - y_mid).abs() < 0.5 * self.bond_length * math.sqrt(3.0)
        if not bool(band.any()):
            band = torch.ones(self.particles, dtype=torch.bool, device=self.device)
        lo, hi = float(self.pos[:, 0].min()), float(self.pos[:, 0].max())
        edges = torch.linspace(lo, hi, bins + 1, dtype=self.dtype, device=self.device)
        centres = 0.5 * (edges[:-1] + edges[1:])
        idx = torch.bucketize(self.pos[:, 0], edges[1:-1])
        disp = (float(self.pos[:, 1].max()) - self.pos[:, 1])
        sums = torch.zeros(bins, dtype=self.dtype, device=self.device)
        counts = torch.zeros(bins, dtype=self.dtype, device=self.device)
        sums = sums.index_add(0, idx[band], disp[band])
        counts = counts.index_add(0, idx[band], torch.ones_like(disp[band]))
        mean = torch.where(counts > 0, sums / counts.clamp_min(1.0), torch.zeros_like(sums))
        return {"x_m": centres.tolist(), "settlement_m": mean.tolist()}

    def fragment_size_distribution(self, *, min_size: float | None = None) -> dict[str, Any]:
        """Connected-cluster fragment sizes after bond breaking [W5-N02 interface].

        Broken bonds split the lattice into clusters; equivalent diameters
        ``d = 2 sqrt(A / pi)`` with ``A`` the cluster area, so the output can be
        fed straight into the ridge fragment-size model.
        """
        alive_pairs = self.pairs[self.alive].cpu().numpy()
        n = self.particles
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        if alive_pairs.size == 0:
            labels = np.arange(n)
        else:
            data = np.ones(alive_pairs.shape[0])
            graph = coo_matrix((data, (alive_pairs[:, 0], alive_pairs[:, 1])), shape=(n, n))
            _, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels)
        s = self.bond_length
        areas = sizes * (math.sqrt(3.0) / 2.0) * s * s
        diameter = 2.0 * np.sqrt(areas / math.pi)
        if min_size is not None:
            diameter = diameter[diameter >= float(min_size)]
        return {
            "clusters": int(sizes.size),
            "particle_counts": sizes.tolist(),
            "equivalent_diameter_m": sorted(float(d) for d in diameter),
            "largest_cluster_particles": int(sizes.max()),
        }

    def state(self) -> dict[str, Any]:
        return {
            "schema": KEEL_SCHEMA,
            "config": asdict(self.config),
            "particles": self.particles,
            "bonds": self.bonds,
            "bond_length_m": self.bond_length,
            "dt_s": self.dt,
            "steps": self.step_count,
            "penetration_m": self.penetration,
            "reaction_N_per_m": self.reaction,
            "broken_bonds": self.broken_bonds,
            "max_bond_strain": self.total_bond_strain,
            "elastic_energy_J": self.elastic_energy(),
        }

# ---------------------------------------------------------------------------
# quasi-static (exact linear) indentation of the same bonded-particle keel
# ---------------------------------------------------------------------------
def _static_hessian(pos_np: np.ndarray, pairs_np: np.ndarray, bond_stiffness: float,
                    ground_stiffness: np.ndarray):
    """Bond Hessian plus the diagonal seabed springs."""
    from scipy.sparse import diags as sp_diags

    hessian = _bond_hessian(torch.as_tensor(pos_np, dtype=torch.float64),
                            torch.as_tensor(pairs_np, dtype=torch.int64), bond_stiffness)
    diag = np.zeros(2 * pos_np.shape[0])
    diag[1::2] = ground_stiffness
    return (hessian + sp_diags(diag, format="csr")).tocsr()


def quasi_static_indentation(
    length: float,
    thickness: float,
    radius: float,
    bond_stiffness: float,
    foundation_modulus: float,
    *,
    punch_width: float = 0.6,
    punch_center: float | None = None,
    depths: Sequence[float] = (),
    initial_gap: float = 1e-3,
    max_active_iterations: int = 60,
) -> dict[str, Any]:
    """Exact linear equilibrium of the bonded keel under a rigid punch.

    The bonded-particle keel is the same discrete model as
    :class:`KeelLatticeDEM`; here the *elastic* (no bond breaking) equilibrium is
    solved directly (sparse stiffness solve with an active set for the unilateral
    punch contact) instead of by explicit time integration.  This removes the
    time-stepping bias from the DEM-vs-beam comparison: the residual is machine
    precision, so any difference to the continuum solution is model difference.

    Returns the punch reaction ``P`` [N/m], the bottom-row settlement profile
    (the foundation-level deflection ``w``), bond strains and the contact set.
    """
    if not depths:
        raise ValueError("depths must not be empty")
    for name, value in (("length", length), ("thickness", thickness), ("radius", radius),
                        ("bond_stiffness", bond_stiffness), ("punch_width", punch_width)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(foundation_modulus) or foundation_modulus < 0.0:
        raise ValueError("foundation_modulus must be finite and nonnegative")

    from scipy.sparse.linalg import splu

    pos, s = triangular_lattice(length, thickness, radius)
    pairs = neighbour_pairs(pos, s)
    n = pos.shape[0]
    x_min, x_max = float(pos[:, 0].min()), float(pos[:, 0].max())
    y_min, y_max = float(pos[:, 1].min()), float(pos[:, 1].max())
    x_center = 0.5 * (x_min + x_max) if punch_center is None else float(punch_center)
    bottom = np.where(np.abs(pos[:, 1] - y_min) < 1e-9)[0]
    ground = np.zeros(n)
    ground[bottom] = foundation_modulus * s
    # x of two bottom corner particles is pinned (seabed friction) to remove the
    # otherwise free horizontal translation mode
    pinned = [2 * int(bottom[np.argmin(pos[bottom, 0])]),
              2 * int(bottom[np.argmax(pos[bottom, 0])])]
    face0 = y_max + radius + initial_gap
    hessian = _static_hessian(pos, pairs, bond_stiffness, ground)
    footprint = np.abs(pos[:, 0] - x_center) <= punch_width / 2.0 + radius
    rows: list[dict[str, Any]] = []
    active_from_previous = np.zeros(n, dtype=bool)
    use_previous = False

    for depth in depths:
        if not math.isfinite(depth) or depth < 0.0:
            raise ValueError("depths must be finite and nonnegative")
        face = face0 - float(depth)
        active = active_from_previous.copy() if use_previous else np.zeros(n, dtype=bool)
        u = np.zeros(2 * n)
        contact_force = np.zeros(n)
        for _ in range(max_active_iterations):
            fixed_dofs = np.array(
                sorted(set(pinned) | {2 * i + 1 for i in np.where(active)[0]}), dtype=np.int64)
            free = np.setdiff1d(np.arange(2 * n), fixed_dofs)
            u = np.zeros(2 * n)
            if active.any():
                u[2 * np.where(active)[0] + 1] = (face - radius) - pos[active, 1]
            rhs = np.zeros(2 * n)
            if fixed_dofs.size:
                rhs = -hessian[:, fixed_dofs] @ u[fixed_dofs]
            sol = splu(hessian[free][:, free].tocsc()).solve(rhs[free])
            if not np.all(np.isfinite(sol)):
                raise ValueError("singular keel stiffness matrix")
            u[free] = sol
            residual = hessian @ u
            deformed = pos[:, 1] + u[1::2]
            penetrating = footprint & ((deformed + radius) > face)
            force_down = -(residual[2 * np.where(active)[0] + 1]) if active.any() else np.array([])
            released = np.zeros(n, dtype=bool)
            if active.any():
                released[np.where(active)[0][force_down < 0.0]] = True
            updated = (active & ~released) | penetrating
            contact_force = np.zeros(n)
            if updated.any():
                idx = 2 * np.where(updated)[0] + 1
                contact_force[np.where(updated)[0]] = -residual[idx]
            if np.array_equal(updated, active):
                active = updated
                break
            active = updated
        active_from_previous = active.copy()
        use_previous = True
        strain = _bond_strain(pos, pairs, u)
        rows.append({
            "punch_depth_m": float(depth),
            "punch_face_m": face,
            "reaction_N_per_m": float(contact_force.sum()),
            "contact_particles": int(active.sum()),
            "max_bond_strain": float(np.abs(strain).max()) if strain.size else 0.0,
            "bottom_x_m": pos[bottom, 0].tolist(),
            "bottom_settlement_m": (-u[2 * bottom + 1]).tolist(),
            "beam_deflection_m": float(np.interp(x_center, pos[bottom, 0],
                                                 -u[2 * bottom + 1])),
            "midplane_settlement_m": float(np.mean(-u[2 * np.where(
                np.abs(pos[:, 1] - 0.5 * (y_min + y_max)) < 0.5 * s * math.sqrt(3.0))[0] + 1])),
            "particles": int(n),
            "bonds": int(pairs.shape[0]),
        })
    return {
        "test": "quasi_static_indentation",
        "bond_length_m": s,
        "particles": int(n),
        "bonds": int(pairs.shape[0]),
        "punch_width_m": punch_width,
        "punch_center_m": x_center,
        "foundation_modulus_N_per_m2": foundation_modulus,
        "bond_stiffness_N_per_m": bond_stiffness,
        "steps": rows,
    }


def _bond_strain(pos: np.ndarray, pairs: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Engineering bond strain for a displacement field ``u`` (2N vector)."""
    d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    length = np.linalg.norm(d, axis=1)
    n = d / length[:, None]
    i, j = 2 * pairs[:, 0], 2 * pairs[:, 1]
    du = np.stack([u[j] - u[i], u[j + 1] - u[i + 1]], axis=1)
    return (du * n).sum(axis=1) / length
