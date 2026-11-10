"""Binary-particle-bed segregation dynamics for TensorDEM.

Implements spontaneous Brazil-nut / reverse-Brazil-nut segregation under
vertical vibration and gravity pouring, with:

* Binary spherical particles (d_large / d_small ∈ {1.5, 2, 3, 4})
* Two driving modes: vertical vibration (Γ = Aω²/g) and gravity pouring
* Segregation index S = (φ_top − φ_nominal) / φ_nominal
* Brazil-nut rise velocity v_up
* Mass-conservation audit (relative error ≤ 1e-9)
* TCG / ΔGM propagation mapping for cargo-hold applications

All quantities SI.  The engine uses a spatial-hash neighbour search that
supports polydisperse radii, with vectorised numpy contact laws.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Polydisperse contact kernel
# ---------------------------------------------------------------------------

def _damping_from_restitution(restitution: float) -> float:
    """zeta = |ln e| / sqrt(pi^2 + ln^2 e)."""
    e = max(min(restitution, 1.0 - 1e-12), 1e-6)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)


@dataclass(frozen=True)
class BinaryMaterial:
    """Two-size spherical particle material (SI)."""
    r_small: float = 0.005          # m
    r_large: float = 0.010          # m
    density: float = 2500.0         # kg/m³
    mu: float = 0.5                 # Coulomb friction
    mu_rolling: float = 0.05        # rolling resistance
    restitution: float = 0.3        # coefficient of restitution
    kn: float = 1.0e5               # normal stiffness (N/m)
    kt_ratio: float = 0.5           # kt / kn
    gravity: float = 9.81           # m/s²

    @property
    def diameter_ratio(self) -> float:
        return self.r_large / self.r_small

    def mass(self, r: float) -> float:
        return self.density * 4.0 / 3.0 * math.pi * r ** 3

    def inertia(self, r: float) -> float:
        return 0.4 * self.mass(r) * r ** 2

    def kt(self) -> float:
        return self.kt_ratio * self.kn

    def kr(self, r: float) -> float:
        return self.kt() * r ** 2

    def gamma_pair(self, r_i: float, r_j: float) -> float:
        """Pair dashpot for two spheres of radii r_i, r_j."""
        m_i = self.mass(r_i)
        m_j = self.mass(r_j)
        m_eff = m_i * m_j / (m_i + m_j)
        zeta = _damping_from_restitution(self.restitution)
        return 2.0 * zeta * math.sqrt(self.kn * m_eff)

    def gamma_wall(self, r: float) -> float:
        """Rigid-wall dashpot."""
        zeta = _damping_from_restitution(self.restitution)
        return 2.0 * zeta * math.sqrt(self.kn * self.mass(r))

    def critical_dt(self) -> float:
        """Stable timestep estimate (smallest particle, stiffest spring)."""
        m = self.mass(self.r_small)
        period = 2.0 * math.pi * math.sqrt(0.5 * m / self.kn)
        return period / 25.0


def build_pairs_poly(pos: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All pairs within r_i + r_j via cKDTree with max-radius bound."""
    n = len(pos)
    if n < 2:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    r_max = float(radii.max())
    tree = cKDTree(pos)
    pairs = tree.query_pairs(2.0 * r_max, output_type="ndarray")
    if len(pairs) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    iu = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    ju = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    dist_cutoff = radii[iu] + radii[ju]
    dvec = pos[ju] - pos[iu]
    dist = np.linalg.norm(dvec, axis=1)
    mask = dist < dist_cutoff
    return iu[mask], ju[mask]


def contact_forces_poly(
    pos: np.ndarray, vel: np.ndarray, omega: np.ndarray,
    radii: np.ndarray, masses: np.ndarray, inertias: np.ndarray,
    iu: np.ndarray, ju: np.ndarray,
    mat: BinaryMaterial, dt: float,
    state_tang: dict | None = None,
    state_roll: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Vectorised polydisperse contact forces and torques."""
    n = len(pos)
    force = np.zeros((n, 3), dtype=np.float64)
    torque = np.zeros((n, 3), dtype=np.float64)
    m_pairs = len(iu)
    if m_pairs == 0:
        return force, torque, state_tang or {}, state_roll or {}

    if state_tang is None:
        state_tang = {}
    if state_roll is None:
        state_roll = {}

    kn = mat.kn
    kt = mat.kt()
    mu = mat.mu
    mu_r = mat.mu_rolling

    r_i = radii[iu]
    r_j = radii[ju]
    m_i = masses[iu]
    m_j = masses[ju]

    dvec = pos[ju] - pos[iu]
    dist = np.linalg.norm(dvec, axis=1)
    overlap = (r_i + r_j) - dist
    active = overlap > 0.0
    if not active.any():
        return force, torque, state_tang, state_roll

    iu_a, ju_a = iu[active], ju[active]
    dvec_a = dvec[active]
    overlap_a = overlap[active]
    r_i_a = r_i[active]
    r_j_a = r_j[active]
    dist_a = dist[active]

    nvec = dvec_a / np.maximum(dist_a, 1e-30)[:, None]

    # Normal force
    v_rel = vel[ju_a] - vel[iu_a]
    v_n = (v_rel * nvec).sum(axis=1)
    m_eff_a = (m_i[active] * m_j[active]) / (m_i[active] + m_j[active])
    zeta = _damping_from_restitution(mat.restitution)
    gamma_a = 2.0 * zeta * np.sqrt(kn * m_eff_a)
    f_n = np.maximum(kn * overlap_a - gamma_a * v_n, 0.0)

    # Tangential
    v_t = v_rel - v_n[:, None] * nvec
    if mu > 0.0:
        surf_i = np.cross(omega[iu_a], nvec * r_i_a[:, None])
        surf_j = np.cross(omega[ju_a], -nvec * r_i_a[:, None])
        v_t_surf = v_t + (surf_j - surf_i)
        keys = (iu_a * n + ju_a).tolist()
        tang = np.array([state_tang.get(k, np.zeros(3)) for k in keys], dtype=np.float64)
        tang = tang + v_t_surf * dt
        tnorm = np.linalg.norm(tang, axis=1)
        cap = mu * f_n / kt
        scale = np.where(tnorm > cap, cap / np.maximum(tnorm, 1e-30), 1.0)
        tang = tang * scale[:, None]
        f_t = -kt * tang
        for idx, key in enumerate(keys):
            state_tang[key] = tang[idx]
    else:
        f_t = np.zeros_like(v_t)
        keys = []

    f_total = f_n[:, None] * nvec + f_t
    np.add.at(force, iu_a, -f_total)
    np.add.at(force, ju_a, f_total)

    if mu > 0.0:
        t_i = np.cross(-nvec * r_i_a[:, None], f_t)
        np.add.at(torque, iu_a, t_i)
        np.add.at(torque, ju_a, t_i)

    # Rolling resistance
    if mu_r > 0.0:
        spin = omega[ju_a] - omega[iu_a]
        keys_r = (iu_a * n + ju_a).tolist()
        r_eff_a = r_i_a * r_j_a / (r_i_a + r_j_a)
        kr_a = kt * r_eff_a ** 2
        roll = np.array([state_roll.get(k, np.zeros(3)) for k in keys_r], dtype=np.float64)
        roll = roll + spin * dt
        rnorm = np.linalg.norm(roll, axis=1)
        cap_r = (mu_r * r_eff_a * f_n) / np.maximum(kr_a, 1e-30)
        scale_r = np.where(rnorm > cap_r, cap_r / np.maximum(rnorm, 1e-30), 1.0)
        roll = roll * scale_r[:, None]
        for idx, key in enumerate(keys_r):
            state_roll[key] = roll[idx]
        torque_r = -kr_a[:, None] * roll
        np.add.at(torque, iu_a, -torque_r)
        np.add.at(torque, ju_a, torque_r)

    # Clean up inactive state
    active_keys = set((iu_a * n + ju_a).tolist())
    for key in list(state_tang.keys()):
        if key not in active_keys:
            del state_tang[key]
    for key in list(state_roll.keys()):
        if key not in active_keys:
            del state_roll[key]

    return force, torque, state_tang, state_roll


# ---------------------------------------------------------------------------
# Particle initialisation
# ---------------------------------------------------------------------------

def init_binary_bed(
    n_small: int, n_large: int,
    r_small: float, r_large: float,
    box_w: float, box_d: float, fill_h: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly place binary particles in a box, resolving overlaps."""
    rng = np.random.default_rng(seed)
    N = n_small + n_large
    radii = np.empty(N, dtype=np.float64)
    types = np.empty(N, dtype=np.int8)
    radii[:n_small] = r_small
    radii[n_small:] = r_large
    types[:n_small] = 0
    types[n_small:] = 1

    pos = np.empty((N, 3), dtype=np.float64)
    for i in range(N):
        r = radii[i]
        pos[i, 0] = rng.uniform(r + 1e-4, box_w - r - 1e-4)
        pos[i, 1] = rng.uniform(r + 1e-4, box_d - r - 1e-4)
        pos[i, 2] = rng.uniform(r + 1e-4, fill_h - r - 1e-4)

    # Overlap relaxation
    for _ in range(30):
        tree = cKDTree(pos)
        pairs = tree.query_pairs(2.0 * r_large, output_type="ndarray")
        if len(pairs) == 0:
            break
        iu = pairs[:, 0]
        ju = pairs[:, 1]
        dvec = pos[ju] - pos[iu]
        dist = np.linalg.norm(dvec, axis=1)
        overlap = (radii[iu] + radii[ju]) - dist
        active = overlap > 0.0
        if not active.any():
            break
        iu_a, ju_a = iu[active], ju[active]
        dvec_a = dvec[active]
        dist_a = dist[active]
        overlap_a = overlap[active]
        nvec = dvec_a / np.maximum(dist_a, 1e-30)[:, None]
        push = 0.5 * overlap_a[:, None] * nvec
        np.add.at(pos, iu_a, -push)
        np.add.at(pos, ju_a, push)

    for i in range(N):
        r = radii[i]
        pos[i, 0] = np.clip(pos[i, 0], r + 1e-4, box_w - r - 1e-4)
        pos[i, 1] = np.clip(pos[i, 1], r + 1e-4, box_d - r - 1e-4)
        pos[i, 2] = np.clip(pos[i, 2], r + 1e-4, fill_h - r - 1e-4)

    return pos, radii, types


# ---------------------------------------------------------------------------
# Wall forces (vectorised)
# ---------------------------------------------------------------------------

def wall_forces(
    pos: np.ndarray, vel: np.ndarray, radii: np.ndarray, masses: np.ndarray,
    box_w: float, box_d: float,
    base_z: float, base_vel: float,
    mat: BinaryMaterial, dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forces from rigid box walls and vibrating base (vectorised)."""
    n = len(pos)
    force = np.zeros((n, 3), dtype=np.float64)
    torque = np.zeros((n, 3), dtype=np.float64)
    kn = mat.kn
    mu = mat.mu
    kt = mat.kt()

    # x walls
    for wall_x, sign in [(0.0, 1.0), (box_w, -1.0)]:
        overlap = radii - sign * (pos[:, 0] - wall_x)
        active = overlap > 0
        if active.any():
            v_n = sign * vel[active, 0]
            gamma = np.array([mat.gamma_wall(radii[i]) for i in np.where(active)[0]])
            f_n = np.maximum(kn * overlap[active] - gamma * v_n, 0.0)
            force[active, 0] += sign * f_n
            if mu > 0:
                v_t = vel[active, 1]
                f_t = np.minimum(kt * np.abs(v_t) * dt, mu * f_n)
                force[active, 1] -= np.sign(v_t) * f_t

    # y walls
    for wall_y, sign in [(0.0, 1.0), (box_d, -1.0)]:
        overlap = radii - sign * (pos[:, 1] - wall_y)
        active = overlap > 0
        if active.any():
            v_n = sign * vel[active, 1]
            gamma = np.array([mat.gamma_wall(radii[i]) for i in np.where(active)[0]])
            f_n = np.maximum(kn * overlap[active] - gamma * v_n, 0.0)
            force[active, 1] += sign * f_n

    # base
    overlap = radii - (pos[:, 2] - base_z)
    active = overlap > 0
    if active.any():
        v_n = vel[active, 2] - base_vel
        gamma = np.array([mat.gamma_wall(radii[i]) for i in np.where(active)[0]])
        f_n = np.maximum(kn * overlap[active] - gamma * v_n, 0.0)
        force[active, 2] += f_n
        if mu > 0:
            v_t = vel[active, :2]
            v_t_norm = np.linalg.norm(v_t, axis=1)
            f_t_mag = np.minimum(kt * v_t_norm * dt, mu * f_n)
            active_idx = np.where(active)[0]
            mask2 = v_t_norm > 1e-12
            force[active_idx[mask2], :2] -= (f_t_mag[mask2] / v_t_norm[mask2])[:, None] * v_t[mask2]

    return force, torque


# ---------------------------------------------------------------------------
# Time integrator (Velocity-Verlet)
# ---------------------------------------------------------------------------

def step_vv(
    pos: np.ndarray, vel: np.ndarray, omega: np.ndarray,
    radii: np.ndarray, masses: np.ndarray, inertias: np.ndarray,
    types: np.ndarray,
    mat: BinaryMaterial,
    box_w: float, box_d: float,
    base_z: float, base_vel: float,
    dt: float,
    state_tang: dict | None = None,
    state_roll: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    """One Velocity-Verlet step."""
    n = len(pos)
    g = mat.gravity

    # Half-step velocity (gravity)
    vel[:, 2] -= 0.5 * g * dt

    # Full-step position
    pos += vel * dt

    # Compute forces
    iu, ju = build_pairs_poly(pos, radii)
    f_contact, t_contact, state_tang, state_roll = contact_forces_poly(
        pos, vel, omega, radii, masses, inertias,
        iu, ju, mat, dt, state_tang, state_roll,
    )
    f_wall, t_wall = wall_forces(
        pos, vel, radii, masses, box_w, box_d,
        base_z, base_vel, mat, dt,
    )

    # Total acceleration
    acc = (f_contact + f_wall) / masses[:, None]

    # Half-step velocity (complete)
    vel += acc * dt
    vel[:, 2] -= 0.5 * g * dt

    # Angular update
    ang_acc = t_contact / inertias[:, None]
    omega += ang_acc * dt

    # Clamp positions
    for i in range(n):
        r = radii[i]
        pos[i, 0] = np.clip(pos[i, 0], r + 1e-6, box_w - r - 1e-6)
        pos[i, 1] = np.clip(pos[i, 1], r + 1e-6, box_d - r - 1e-6)
        pos[i, 2] = np.maximum(pos[i, 2], base_z + r + 1e-6)

    return pos, vel, omega, state_tang, state_roll


# ---------------------------------------------------------------------------
# Vibration driver
# ---------------------------------------------------------------------------

def vibration_base_z(t: float, amplitude: float, frequency: float) -> float:
    """Base position: z = A sin(ωt)."""
    omega = 2.0 * math.pi * frequency
    return amplitude * math.sin(omega * t)


def vibration_base_vel(t: float, amplitude: float, frequency: float) -> float:
    """Base velocity: v = Aω cos(ωt)."""
    omega = 2.0 * math.pi * frequency
    return amplitude * omega * math.cos(omega * t)


def compute_gamma(amplitude: float, frequency: float, g: float = 9.81) -> float:
    """Dimensionless vibration intensity Γ = Aω²/g."""
    omega = 2.0 * math.pi * frequency
    return amplitude * omega ** 2 / g


def amplitude_from_gamma(gamma: float, frequency: float, g: float = 9.81) -> float:
    """Compute amplitude A from Γ and frequency."""
    omega = 2.0 * math.pi * frequency
    return gamma * g / omega ** 2


# ---------------------------------------------------------------------------
# Segregation analysis
# ---------------------------------------------------------------------------

def compute_segregation_index(
    pos: np.ndarray, types: np.ndarray, radii: np.ndarray,
    density: float = 2500.0,
    n_layers: int = 10,
) -> dict:
    """Compute segregation metrics from particle positions."""
    n_small = int((types == 0).sum())
    n_large = int((types == 1).sum())
    N = n_small + n_large

    if N == 0:
        return {"S": 0.0, "v_up_est": 0.0, "layer_composition": [],
                "z_com": 0.0, "total_mass": 0.0, "phi_nom_small": 0.5, "phi_top_small": 0.5}

    r_small = radii[0] if n_small > 0 else 0.005
    r_large = radii[n_small] if n_large > 0 else 0.010

    vol_small = n_small * (4.0 / 3.0 * math.pi * r_small ** 3)
    vol_large = n_large * (4.0 / 3.0 * math.pi * r_large ** 3)
    vol_total = vol_small + vol_large
    phi_nom_small = vol_small / vol_total if vol_total > 0 else 0.5

    z_min = float(pos[:, 2].min())
    z_max = float(pos[:, 2].max())
    z_range = z_max - z_min

    masses = np.where(types == 0,
                      density * 4.0 / 3.0 * math.pi * r_small ** 3,
                      density * 4.0 / 3.0 * math.pi * r_large ** 3)
    z_com = float(np.sum(pos[:, 2] * masses) / np.sum(masses))

    if z_range < 1e-12:
        return {"S": 0.0, "v_up_est": 0.0, "layer_composition": [],
                "z_com": z_com, "total_mass": float(np.sum(masses)),
                "phi_nom_small": phi_nom_small, "phi_top_small": phi_nom_small}

    layer_thickness = z_range / n_layers
    layer_composition = []
    top_small_vol = 0.0
    top_total_vol = 0.0

    for layer in range(n_layers):
        z_lo = z_min + layer * layer_thickness
        z_hi = z_lo + layer_thickness
        mask = (pos[:, 2] >= z_lo) & (pos[:, 2] < z_hi)
        n_in_layer = int(mask.sum())
        if n_in_layer == 0:
            layer_composition.append((layer, 0.0, 0.0))
            continue
        n_s = int((types[mask] == 0).sum())
        n_l = int((types[mask] == 1).sum())
        vol_s = n_s * (4.0 / 3.0 * math.pi * r_small ** 3)
        vol_l = n_l * (4.0 / 3.0 * math.pi * r_large ** 3)
        vol_layer = vol_s + vol_l
        phi_s = vol_s / vol_layer if vol_layer > 0 else 0.0
        phi_l = vol_l / vol_layer if vol_layer > 0 else 0.0
        layer_composition.append((layer, phi_s, phi_l))

        if layer == n_layers - 1:
            top_small_vol = vol_s
            top_total_vol = vol_l + vol_s

    # Use top layers that actually contain particles (non-empty)
    non_empty_layers = [(l, ps, pl) for l, ps, pl in layer_composition if ps + pl > 0.01]
    if non_empty_layers:
        # Use top 3 non-empty layers
        top_non_empty = non_empty_layers[-3:] if len(non_empty_layers) >= 3 else non_empty_layers
        top_phi_small_values = [ps for _, ps, _ in top_non_empty]
        phi_top_small = float(np.mean(top_phi_small_values))
    else:
        phi_top_small = phi_nom_small
    S = (phi_top_small - phi_nom_small) / phi_nom_small if phi_nom_small > 0 else 0.0

    return {
        "S": S,
        "v_up_est": 0.0,
        "layer_composition": layer_composition,
        "z_com": z_com,
        "total_mass": float(np.sum(masses)),
        "phi_nom_small": phi_nom_small,
        "phi_top_small": phi_top_small,
    }


def compute_mass_conservation(
    pos: np.ndarray, radii: np.ndarray, types: np.ndarray,
    density: float = 2500.0,
) -> dict:
    """Audit mass conservation."""
    n_small = int((types == 0).sum())
    n_large = int((types == 1).sum())
    r_small = radii[0] if n_small > 0 else 0.0
    r_large = radii[n_small] if n_large > 0 else 0.0

    mass_small = n_small * density * 4.0 / 3.0 * math.pi * r_small ** 3
    mass_large = n_large * density * 4.0 / 3.0 * math.pi * r_large ** 3
    total_mass = mass_small + mass_large

    actual_n = len(pos)
    expected_n = n_small + n_large

    return {
        "expected_n": expected_n,
        "actual_n": actual_n,
        "mass_small": mass_small,
        "mass_large": mass_large,
        "total_mass": total_mass,
        "n_conserved": actual_n == expected_n,
        "relative_error": abs(actual_n - expected_n) / max(expected_n, 1),
    }


def compute_tcg_gm_offset(
    pos: np.ndarray, radii: np.ndarray, types: np.ndarray,
    density: float = 2500.0,
    hold_width: float = 10.0,
    hold_length: float = 20.0,
    initial_tcg: float = 0.0,
    initial_kg: float = 5.0,
    initial_km: float = 4.0,
) -> dict:
    """Map segregated mass distribution to cargo-hold TCG and GM offsets."""
    n = len(pos)
    if n == 0:
        return {"delta_tcg": 0.0, "delta_kg": 0.0, "delta_gm": 0.0,
                "final_tcg": 0.0, "final_kg": 0.0, "final_gm": 0.0}

    n_small = int((types == 0).sum())
    r_small = radii[0] if n_small > 0 else 0.005
    r_large = radii[n_small] if n > n_small else 0.010

    masses = np.where(types == 0,
                      density * 4.0 / 3.0 * math.pi * r_small ** 3,
                      density * 4.0 / 3.0 * math.pi * r_large ** 3)

    tcg = float(np.sum(pos[:, 0] * masses) / np.sum(masses))
    delta_tcg = tcg - initial_tcg

    kg = float(np.sum(pos[:, 2] * masses) / np.sum(masses))
    delta_kg = kg - initial_kg

    delta_gm = -delta_kg

    return {
        "delta_tcg": delta_tcg,
        "delta_kg": delta_kg,
        "delta_gm": delta_gm,
        "final_tcg": tcg,
        "final_kg": kg,
        "final_gm": initial_kg + initial_km + delta_gm,
        "total_cargo_mass": float(np.sum(masses)),
    }


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

@dataclass
class SegregationResult:
    """Container for a complete segregation simulation result."""
    diameter_ratio: float
    phi_large_nominal: float
    driving_mode: str
    gamma: float = 0.0
    frequency: float = 0.0
    amplitude: float = 0.0
    n_small: int = 0
    n_large: int = 0
    n_total: int = 0
    S: float = 0.0
    v_up: float = 0.0
    z_com: float = 0.0
    total_mass: float = 0.0
    mass_conservation_error: float = 0.0
    delta_tcg: float = 0.0
    delta_gm: float = 0.0
    layer_composition: list = field(default_factory=list)
    seed: int = 42
    steps: int = 0
    wallclock_s: float = 0.0
    config_2d: bool = False


def run_segregation(
    diameter_ratio: float = 2.0,
    phi_large: float = 0.3,
    n_total: int = 5000,
    driving_mode: str = "vibration",
    gamma: float = 2.0,
    frequency: float = 30.0,
    n_steps: int = 5000,
    seed: int = 42,
    config_2d: bool = False,
    sample_interval: int = 100,
) -> SegregationResult:
    """Run a full segregation simulation."""
    import time
    t0 = time.time()

    r_small = 0.005
    r_large = r_small * diameter_ratio
    density = 2500.0

    vol_ratio = (r_large / r_small) ** 3
    n_large = int(round(n_total * phi_large * vol_ratio / (phi_large * vol_ratio + (1 - phi_large))))
    n_small = n_total - n_large

    mat = BinaryMaterial(
        r_small=r_small, r_large=r_large, density=density,
        mu=0.5, mu_rolling=0.05, restitution=0.3,
        kn=1.0e5, kt_ratio=0.5, gravity=9.81,
    )

    if config_2d:
        box_w = 0.2
        box_d = 3.0 * r_large * 2
    else:
        box_w = 0.15
        box_d = 0.15

    vol_particles = n_small * 4.0 / 3.0 * math.pi * r_small ** 3 + \
                    n_large * 4.0 / 3.0 * math.pi * r_large ** 3
    packing_frac = 0.6
    fill_h = vol_particles / (box_w * box_d * packing_frac) + 0.02

    pos, radii, types = init_binary_bed(
        n_small, n_large, r_small, r_large,
        box_w, box_d, fill_h, seed=seed,
    )

    masses = np.where(types == 0, mat.mass(r_small), mat.mass(r_large))
    inertias = np.where(types == 0, mat.inertia(r_small), mat.inertia(r_large))
    vel = np.zeros((len(pos), 3), dtype=np.float64)
    omega = np.zeros((len(pos), 3), dtype=np.float64)

    dt = mat.critical_dt()
    state_tang: dict = {}
    state_roll: dict = {}

    # Settling phase
    settle_steps = min(2000, n_steps // 2)
    for step in range(settle_steps):
        pos, vel, omega, state_tang, state_roll = step_vv(
            pos, vel, omega, radii, masses, inertias, types,
            mat, box_w, box_d, 0.0, 0.0, dt, state_tang, state_roll,
        )

    # Driving phase
    amplitude = amplitude_from_gamma(gamma, frequency) if driving_mode == "vibration" else 0.0
    history_S = []
    history_z_com = []
    history_time = []

    drive_steps = n_steps - settle_steps
    for step in range(drive_steps):
        t = step * dt
        if driving_mode == "vibration":
            base_z = vibration_base_z(t, amplitude, frequency)
            base_v = vibration_base_vel(t, amplitude, frequency)
        else:
            base_z = 0.0
            base_v = 0.0

        pos, vel, omega, state_tang, state_roll = step_vv(
            pos, vel, omega, radii, masses, inertias, types,
            mat, box_w, box_d, base_z, base_v, dt, state_tang, state_roll,
        )

        if step % sample_interval == 0:
            metrics = compute_segregation_index(pos, types, radii, density)
            history_S.append(metrics["S"])
            history_z_com.append(metrics["z_com"])
            history_time.append(t)

    final_metrics = compute_segregation_index(pos, types, radii, density)
    mass_audit = compute_mass_conservation(pos, radii, types, density)
    tcg_gm = compute_tcg_gm_offset(pos, radii, types, density)

    v_up = 0.0
    if len(history_z_com) >= 3:
        # Use linear regression to filter out oscillations
        t_arr = np.array(history_time)
        z_arr = np.array(history_z_com)
        # Use second half of data (after initial transients)
        half = len(t_arr) // 2
        coeffs = np.polyfit(t_arr[half:], z_arr[half:], 1)
        v_up = coeffs[0]  # slope = velocity

    wallclock = time.time() - t0

    return SegregationResult(
        diameter_ratio=diameter_ratio,
        phi_large_nominal=phi_large,
        driving_mode=driving_mode,
        gamma=gamma,
        frequency=frequency,
        amplitude=amplitude,
        n_small=n_small,
        n_large=n_large,
        n_total=n_total,
        S=final_metrics["S"],
        v_up=v_up,
        z_com=final_metrics["z_com"],
        total_mass=final_metrics["total_mass"],
        mass_conservation_error=mass_audit["relative_error"],
        delta_tcg=tcg_gm["delta_tcg"],
        delta_gm=tcg_gm["delta_gm"],
        layer_composition=final_metrics["layer_composition"],
        seed=seed,
        steps=n_steps,
        wallclock_s=wallclock,
        config_2d=config_2d,
    )
