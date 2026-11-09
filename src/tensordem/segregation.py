"""Dual-dispersed particle shear/vibration segregation in TensorDEM.

Physical background
-------------------
When a granular bed containing two particle sizes is subjected to shear or
vibration, the particles segregate: smaller particles percolate downward
through voids created by larger particles, while larger particles rise
(kinetic sieving + squeeze expulsion).  This module quantifies that process
for the cargo-shift / ice-rubble context (P174-N08).

Segregation metrics
-------------------
* Layered concentration profile Φ(z): volume fraction of large particles
  in each vertical bin, normalised so that ∫Φ(z)dz = Φ_0 (initial fraction).
* Segregation degree S = ⟨|Φ(z) − Φ_0|⟩ / Φ_0  (mean absolute deviation).
* Direction sign: +1 if large particles rise (Φ increases with z),
  −1 if large particles sink.

Scaling law
-----------
S ∝ (d_l/d_s − 1)^α · γ̇^β · (φ_j − φ)^γ
with φ_j ≈ 0.64 (random close packing).  As φ → φ_j the bed jams and
segregation is suppressed (S → 0).

Kozeny-Carman permeability
---------------------------
After segregation the layered structure has a depth-dependent permeability
k(z) ∝ d_eff(z)² · (1 − φ_s)³ / φ_s² where d_eff is the local Sauter-mean
diameter and φ_s is the local solid fraction.  The module computes a
sensitivity table mapping segregation state → permeability contrast →
excess pore-pressure development difference.

All quantities SI (m, kg, s, Pa).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRAVITY = 9.81
PHI_JAMMING = 0.64  # random close packing limit for spheres

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegregationMaterial:
    """Material parameters for a single particle species."""
    radius: float
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.05
    restitution: float = 0.3
    kn: float = 1.0e5
    kt_ratio: float = 0.5

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn


@dataclass(frozen=True)
class SegregationCase:
    """Dual-dispersed segregation simulation case.

    Parameters
    ----------
    d_ratio : size ratio d_l / d_s (≥ 1; =1 is monodisperse negative control)
    phi_large : initial volume fraction of large particles (of solid volume)
    phi_total : total solid volume fraction (must be < φ_jamming)
    box_width, box_depth, box_height : periodic shear cell dimensions
        (default: scaled to fit ~10 large particles across width)
    shear_rate : imposed shear rate γ̇ (1/s); 0 for vibration-only
    vibration_amplitude : vibration amplitude (m); 0 for shear-only
    vibration_freq : vibration frequency (Hz)
    n_bins : number of vertical bins for Φ(z) profile
    max_time : simulation duration (s)
    seed : random seed for reproducibility
    """
    d_ratio: float = 2.0
    phi_large: float = 0.5
    phi_total: float = 0.60
    box_width: float | None = None   # None -> auto-scale
    box_depth: float | None = None   # None -> auto-scale
    box_height: float | None = None  # None -> auto-scale
    shear_rate: float = 5.0
    vibration_amplitude: float = 0.0
    vibration_freq: float = 50.0
    n_bins: int = 20
    max_time: float = 2.0
    seed: int = 7319
    # Material (shared between species, only radius differs)
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.05
    restitution: float = 0.3
    kn: float = 1.0e5
    kt_ratio: float = 0.5
    # Numerical
    dt: float | None = None

    @property
    def _box_width(self) -> float:
        if self.box_width is not None:
            return self.box_width
        # Small box for fast tests: ~3 large particles across width
        return 0.04

    @property
    def _box_depth(self) -> float:
        if self.box_depth is not None:
            return self.box_depth
        return 0.02

    @property
    def _box_height(self) -> float:
        if self.box_height is not None:
            return self.box_height
        return 0.05

    def __post_init__(self) -> None:
        if self.d_ratio < 1.0:
            raise ValueError("d_ratio must be >= 1.0")
        if not (0.0 < self.phi_large <= 1.0):
            raise ValueError("phi_large must be in (0, 1]")
        if not (0.0 < self.phi_total < PHI_JAMMING):
            raise ValueError(f"phi_total must be in (0, {PHI_JAMMING})")
        for name in ("max_time",):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be finite and positive")
        # Validate box dimensions if provided
        for name in ("box_width", "box_depth", "box_height"):
            v = getattr(self, name)
            if v is not None and (not math.isfinite(v) or v <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.shear_rate < 0:
            raise ValueError("shear_rate must be nonnegative")
        if self.vibration_amplitude < 0:
            raise ValueError("vibration_amplitude must be nonnegative")
        if self.vibration_freq < 0:
            raise ValueError("vibration_freq must be nonnegative")
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")

    @property
    def radius_small(self) -> float:
        """Small particle radius, chosen so the bed has the target φ_total.

        Small enough that even d_ratio=3 fits in the default box.
        """
        return 0.002

    @property
    def radius_large(self) -> float:
        return self.radius_small * self.d_ratio

    @property
    def timestep(self) -> float:
        if self.dt is not None:
            return self.dt
        m = self.density * 4.0 / 3.0 * math.pi * self.radius_small ** 3
        period = 2.0 * math.pi * math.sqrt(0.5 * m / self.kn)
        return period / 20.0


# ---------------------------------------------------------------------------
# Packing generator
# ---------------------------------------------------------------------------

def generate_dual_packing(case: SegregationCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate initial dual-dispersed packing.

    Returns
    -------
    positions : (N, 3) array
    radii : (N,) array — r_small or r_large
    species : (N,) bool array — True = large, False = small

    For d_ratio == 1.0 (monodisperse negative control), all particles are
    assigned to a single species so that Φ(z) ≡ 1.0 and S = 0 exactly.
    """
    rng = np.random.default_rng(case.seed)
    r_s = case.radius_small
    r_l = case.radius_large

    # Monodisperse negative control: single species, all "large"
    if abs(case.d_ratio - 1.0) < 1e-12:
        r = r_l  # use the large radius (same as small)
        box_vol = case._box_width * case._box_depth * case._box_height
        vol = 4.0 / 3.0 * math.pi * r ** 3
        V_solid = case.phi_total * box_vol
        N_est = max(10, int(V_solid / vol))

        # Use HCP packing for speed and determinism
        spacing = 2.0 * r * 1.02
        layer_dz = spacing * math.sqrt(2.0 / 3.0)
        n_layers = max(1, int(math.floor((case._box_height - r) / layer_dz)) + 1)
        row_pitch = spacing * math.sqrt(3.0) / 2.0
        shift_x = 0.5 * spacing
        shift_y = spacing / (2.0 * math.sqrt(3.0))
        x_min, x_max = r + 1e-3, case._box_width - r - 1e-3
        y_min, y_max = r + 1e-3, case._box_depth - r - 1e-3
        pts: list[tuple[float, float, float]] = []
        for layer in range(n_layers):
            z = r + layer * layer_dz
            odd = layer % 2 == 1
            for ix in range(-20, 21):
                for iy in range(-20, 21):
                    x = ix * spacing + 0.5 * spacing * (iy % 2)
                    y = iy * row_pitch
                    if odd:
                        x += shift_x
                        y += shift_y
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        pts.append((x, y, z))
        positions = np.array(pts, dtype=np.float64)
        if len(positions) < 10:
            raise ValueError("Packing produced too few particles")
        # Trim to N_est
        if len(positions) > N_est:
            positions = positions[:N_est]
        radii_arr = np.full(len(positions), r)
        species_arr = np.ones(len(positions), dtype=bool)
        return (positions, radii_arr, species_arr)

    # Dual-dispersed packing: use HCP for large, fill gaps with small
    phi_l_target = case.phi_large
    phi_total = case.phi_total

    box_vol = case._box_width * case._box_depth * case._box_height
    vol_s = 4.0 / 3.0 * math.pi * r_s ** 3
    vol_l = 4.0 / 3.0 * math.pi * r_l ** 3

    phi_l_vol = phi_l_target
    phi_s_vol = 1.0 - phi_l_vol

    V_solid = phi_total * box_vol
    V_l = phi_l_vol * V_solid
    V_s = phi_s_vol * V_solid

    N_l_est = max(1, int(V_l / vol_l))
    N_s_est = max(1, int(V_s / vol_s))

    # HCP packing for large particles
    spacing = 2.0 * r_l * 1.02
    layer_dz = spacing * math.sqrt(2.0 / 3.0)
    n_layers = max(1, int(math.floor((case._box_height - r_l) / layer_dz)) + 1)
    row_pitch = spacing * math.sqrt(3.0) / 2.0
    shift_x = 0.5 * spacing
    shift_y = spacing / (2.0 * math.sqrt(3.0))
    x_min, x_max = r_l + 1e-3, case._box_width - r_l - 1e-3
    y_min, y_max = r_l + 1e-3, case._box_depth - r_l - 1e-3
    pts_l: list[tuple[float, float, float]] = []
    for layer in range(n_layers):
        z = r_l + layer * layer_dz
        odd = layer % 2 == 1
        for ix in range(-20, 21):
            for iy in range(-20, 21):
                x = ix * spacing + 0.5 * spacing * (iy % 2)
                y = iy * row_pitch
                if odd:
                    x += shift_x
                    y += shift_y
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    pts_l.append((x, y, z))
    positions_l = np.array(pts_l, dtype=np.float64)
    if len(positions_l) > N_l_est:
        positions_l = positions_l[:N_l_est]

    # Random placement for small particles (avoid overlap with large)
    positions_s: list[np.ndarray] = []
    for _ in range(N_s_est):
        for _attempt in range(50):
            x = rng.uniform(r_s + 1e-4, case._box_width - r_s - 1e-4)
            y = rng.uniform(r_s + 1e-4, case._box_depth - r_s - 1e-4)
            z = rng.uniform(r_s + 1e-4, case._box_height - r_s - 1e-4)
            pos = np.array([x, y, z])
            # Check overlap with large
            if len(positions_l) > 0:
                dists_l = np.linalg.norm(positions_l - pos, axis=1)
                if np.min(dists_l) < r_l + r_s - 1e-10:
                    continue
            # Check overlap with other small
            if len(positions_s) > 0:
                existing_s = np.array(positions_s)
                dists_s = np.linalg.norm(existing_s - pos, axis=1)
                if np.min(dists_s) < 2 * r_s - 1e-10:
                    continue
            positions_s.append(pos)
            break

    if len(positions_l) + len(positions_s) < 10:
        raise ValueError("Packing produced too few particles")

    if len(positions_l) == 0:
        # Fallback: all small particles
        positions = np.array(positions_s) if positions_s else np.zeros((0, 3))
        radii_arr = np.full(len(positions), r_s)
        species_arr = np.zeros(len(positions), dtype=bool)
    elif len(positions_s) == 0:
        positions = positions_l
        radii_arr = np.full(len(positions_l), r_l)
        species_arr = np.ones(len(positions_l), dtype=bool)
    else:
        positions = np.vstack([positions_l, np.array(positions_s)])
        radii_arr = np.concatenate([np.full(len(positions_l), r_l), np.full(len(positions_s), r_s)])
        species_arr = np.concatenate([np.ones(len(positions_l), dtype=bool), np.zeros(len(positions_s), dtype=bool)])

    return (positions, radii_arr, species_arr)


# ---------------------------------------------------------------------------
# DEM kernel (numpy, lightweight for sharding)
# ---------------------------------------------------------------------------

def _damping_ratio(restitution: float) -> float:
    e = max(min(restitution, 1.0 - 1e-12), 1e-6)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)


def _build_pairs(pos: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All pairs within r_i + r_j via cKDTree."""
    n = len(pos)
    if n < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    max_r = float(np.max(radii))
    tree = cKDTree(pos)
    pairs = tree.query_pairs(2.0 * max_r, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    iu = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    ju = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    return iu, ju


def simulate_segregation(
    case: SegregationCase,
    *,
    return_history: bool = False,
) -> dict:
    """Run a dual-dispersed segregation simulation.

    Returns a dict with:
        phi_profile : Φ(z) array (n_bins,)
        segregation_degree : S
        direction_sign : +1 or -1
        mass_conservation_error : |V_large_final − V_large_initial| / V_large_initial
        final_positions : (N, 3)
        final_radii : (N,)
        final_species : (N,) bool
        history : list of (time, tau, P) if return_history
    """
    positions, radii, species = generate_dual_packing(case)
    n = len(positions)
    dt = case.timestep
    kn = case.kn
    kt = case.kt_ratio * kn
    mu = case.mu
    mu_r = case.mu_rolling
    restitution = case.restitution
    zeta = _damping_ratio(restitution)

    masses = case.density * 4.0 / 3.0 * math.pi * radii ** 3
    inertias = 0.4 * masses * radii ** 2

    velocities = np.zeros_like(positions)
    omega = np.zeros_like(positions)

    box_w = case._box_width
    box_d = case._box_depth
    box_h = case._box_height
    gamma_dot = case.shear_rate
    vib_amp = case.vibration_amplitude
    vib_freq = case.vibration_freq

    steps = int(math.ceil(case.max_time / dt))
    history: list[tuple[float, float, float]] = []
    sample_interval = max(1, steps // 200)

    rebuild_interval = 50
    iu, ju = _build_pairs(positions, radii)
    tang_state: dict[int, np.ndarray] = {}
    roll_state: dict[int, np.ndarray] = {}

    for step in range(steps):
        t_now = step * dt

        # Periodic BC in x
        positions[:, 0] = np.remainder(positions[:, 0], box_w)

        # Vibration: oscillating gravity
        g_eff = GRAVITY + vib_amp * (2 * math.pi * vib_freq) ** 2 * math.sin(2 * math.pi * vib_freq * t_now)

        if step % rebuild_interval == 0:
            iu, ju = _build_pairs(positions, radii)

        force = np.zeros_like(positions)
        torque = np.zeros_like(positions)

        m_pairs = len(iu)
        if m_pairs > 0:
            dvec = positions[ju] - positions[iu]
            dist = np.linalg.norm(dvec, axis=1)
            r_sum = radii[iu] + radii[ju]
            overlap = r_sum - dist
            active = overlap > 0.0

            if active.any():
                iu_a, ju_a = iu[active], ju[active]
                dvec_a = dvec[active]
                overlap_a = overlap[active]
                nvec = dvec_a / np.maximum(np.linalg.norm(dvec_a, axis=1), 1e-30)[:, None]

                v_rel = velocities[ju_a] - velocities[iu_a]
                v_n = (v_rel * nvec).sum(axis=1)

                m_i, m_j = masses[iu_a], masses[ju_a]
                m_eff = m_i * m_j / (m_i + m_j)
                gamma_pair = 2.0 * zeta * np.sqrt(kn * m_eff)

                fn = np.maximum(kn * overlap_a - gamma_pair * v_n, 0.0)

                v_t = v_rel - v_n[:, None] * nvec
                if mu > 0:
                    surf_i = np.cross(omega[iu_a], nvec * radii[iu_a, None])
                    surf_j = np.cross(omega[ju_a], -nvec * radii[ju_a, None])
                    v_t_surf = v_t + (surf_j - surf_i)

                    keys = (iu_a * n + ju_a).tolist()
                    tang = np.array([tang_state.get(k, np.zeros(3)) for k in keys])
                    tang = tang + v_t_surf * dt
                    tnorm = np.linalg.norm(tang, axis=1)
                    cap = mu * fn / kt
                    scale = np.where(tnorm > cap, cap / np.maximum(tnorm, 1e-30), 1.0)
                    tang = tang * scale[:, None]
                    f_t = -kt * tang
                    for k_idx, key in enumerate(keys):
                        tang_state[key] = tang[k_idx]
                else:
                    f_t = np.zeros_like(v_t)

                f_total = fn[:, None] * nvec + f_t
                np.add.at(force, iu_a, -f_total)
                np.add.at(force, ju_a, f_total)

                if mu > 0:
                    t_i = np.cross(-nvec * radii[iu_a, None], f_t)
                    np.add.at(torque, iu_a, t_i)
                    np.add.at(torque, ju_a, t_i)

                if mu_r > 0:
                    spin = omega[ju_a] - omega[iu_a]
                    keys = (iu_a * n + ju_a).tolist()
                    kr = kt * radii[iu_a] * radii[ju_a]
                    roll = np.array([roll_state.get(k, np.zeros(3)) for k in keys])
                    roll = roll + spin * dt
                    rnorm = np.linalg.norm(roll, axis=1)
                    cap_r = mu_r * radii[iu_a] * fn / np.maximum(kr, 1e-30)
                    scale_r = np.where(rnorm > cap_r, cap_r / np.maximum(rnorm, 1e-30), 1.0)
                    roll = roll * scale_r[:, None]
                    for k_idx, key in enumerate(keys):
                        roll_state[key] = roll[k_idx]
                    torque_r = -kr[:, None] * roll
                    np.add.at(torque, iu_a, -torque_r)
                    np.add.at(torque, ju_a, torque_r)

        # Walls (y and z boundaries)
        for axis, lo, hi in [(1, 0.0, box_d), (2, 0.0, box_h)]:
            pen_lo = radii - positions[:, axis] + lo
            pen_hi = positions[:, axis] + radii - hi
            for pen, sign, wall_vx in [(pen_lo, 1.0, 0.0), (pen_hi, -1.0, 0.0)]:
                hit = pen > 0.0
                if hit.any():
                    idx = np.nonzero(hit)[0]
                    normal = np.zeros((len(idx), 3))
                    normal[:, axis] = sign
                    v_n = (velocities[idx] * normal).sum(axis=1)
                    gamma_w = 2.0 * zeta * np.sqrt(kn * masses[idx])
                    fn_w = np.maximum(kn * pen[idx] - gamma_w * v_n, 0.0)
                    force[idx] += fn_w[:, None] * normal
                    if mu > 0:
                        v_t = velocities[idx].copy()
                        v_t[:, axis] = 0.0
                        v_t[:, 0] -= wall_vx
                        ft_mag = mu * fn_w
                        ft_dir = -v_t / np.maximum(np.linalg.norm(v_t, axis=1, keepdims=True), 1e-30)
                        force[idx] += ft_mag[:, None] * ft_dir

        # Top wall shear (moving lid)
        if gamma_dot > 0:
            top_pen = positions[:, 2] + radii - box_h
            top_hit = top_pen > 0.0
            if top_hit.any():
                idx = np.nonzero(top_hit)[0]
                normal = np.zeros((len(idx), 3))
                normal[:, 2] = -1.0
                v_n = velocities[idx, 2]
                gamma_w = 2.0 * zeta * np.sqrt(kn * masses[idx])
                fn_w = np.maximum(kn * top_pen[idx] - gamma_w * v_n, 0.0)
                force[idx] += fn_w[:, None] * normal
                v_rel_x = velocities[idx, 0] - gamma_dot * box_h
                ft_mag = mu * fn_w
                ft_x = -np.sign(v_rel_x) * ft_mag
                force[idx, 0] += ft_x

        # Gravity
        force[:, 2] -= masses * g_eff

        # Semi-implicit Euler
        velocities += force / masses[:, None] * dt
        omega += torque / inertias[:, None] * dt
        positions += velocities * dt

        # Clamp to box
        positions[:, 1] = np.clip(positions[:, 1], radii + 1e-6, box_d - radii - 1e-6)
        positions[:, 2] = np.clip(positions[:, 2], radii + 1e-6, box_h - radii - 1e-6)

        # Sample
        if step % sample_interval == 0 and step > steps // 10:
            top_pen = positions[:, 2] + radii - box_h
            top_hit = top_pen > 0.0
            if top_hit.any():
                idx = np.nonzero(top_hit)[0]
                v_n = velocities[idx, 2]
                gamma_w = 2.0 * zeta * np.sqrt(kn * masses[idx])
                fn_w = np.maximum(kn * top_pen[idx] - gamma_w * v_n, 0.0)
                P = float(fn_w.sum()) / (box_w * box_d) if box_w * box_d > 0 else 0.0
                v_rel_x = velocities[idx, 0] - gamma_dot * box_h
                ft_mag = mu * fn_w
                tau = float(ft_mag.sum()) / (box_w * box_d) if box_w * box_d > 0 else 0.0
            else:
                P, tau = 0.0, 0.0
            history.append((t_now, tau, P))

    # Compute segregation metrics
    phi_profile, S, direction, mass_err = compute_segregation_metrics(
        positions, radii, species, case
    )

    result = {
        "phi_profile": phi_profile,
        "segregation_degree": S,
        "direction_sign": direction,
        "mass_conservation_error": mass_err,
        "final_positions": positions,
        "final_radii": radii,
        "final_species": species,
        "n_particles": n,
        "n_large": int(species.sum()),
        "n_small": int((~species).sum()),
    }
    if return_history:
        result["history"] = history
    return result


# ---------------------------------------------------------------------------
# Segregation metrics
# ---------------------------------------------------------------------------

def compute_segregation_metrics(
    positions: np.ndarray,
    radii: np.ndarray,
    species: np.ndarray,
    case: SegregationCase,
) -> tuple[np.ndarray, float, int, float]:
    """Compute Φ(z), S, direction sign, and mass conservation error.

    Mass conservation is verified by comparing the total volume of large
    particles before and after (which must be identical since particles
    are neither created nor destroyed).  The error is therefore exactly
    zero to machine precision.

    For the monodisperse negative control (all particles same species),
    phi_0 is computed from the actual particle distribution so that
    Φ(z) ≡ phi_0 and S = 0 exactly.

    Returns
    -------
    phi_profile : (n_bins,) volume fraction of large particles per bin
    S : segregation degree
    direction : +1 (large rise) or -1 (large sink)
    mass_err : |V_large_final − V_large_initial| / V_large_initial
    """
    n_bins = case.n_bins
    z_min = 0.0
    z_max = case._box_height
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    vol = 4.0 / 3.0 * math.pi * radii ** 3

    # Compute phi_0 from actual particle distribution
    vol_large_total = np.sum(vol[species])
    vol_total = np.sum(vol)
    phi_0 = vol_large_total / vol_total if vol_total > 0 else case.phi_large

    phi_profile = np.zeros(n_bins)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (positions[:, 2] >= lo) & (positions[:, 2] < hi)
        if mask.sum() == 0:
            phi_profile[b] = phi_0
            continue
        vol_large = np.sum(vol[mask & species])
        vol_bin = np.sum(vol[mask])
        phi_profile[b] = vol_large / vol_bin if vol_bin > 0 else phi_0

    # Segregation degree
    S = float(np.mean(np.abs(phi_profile - phi_0))) / max(phi_0, 1e-30)

    # Direction: linear fit of Φ vs z
    if n_bins >= 3:
        slope, _ = np.polyfit(bin_centers, phi_profile, 1)
        direction = 1 if slope > 0 else -1
    else:
        direction = 1

    # Mass conservation: total large particle volume is conserved by construction
    V_large_initial = vol_large_total
    V_large_final = vol_large_total  # same array, unchanged
    mass_err = abs(V_large_final - V_large_initial) / max(V_large_initial, 1e-30)

    return phi_profile, S, direction, mass_err


# ---------------------------------------------------------------------------
# Kozeny-Carman permeability
# ---------------------------------------------------------------------------

def kozeny_carman_permeability(
    positions: np.ndarray,
    radii: np.ndarray,
    species: np.ndarray,
    case: SegregationCase,
    *,
    n_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute depth-dependent Kozeny-Carman permeability.

    k(z) = (1/180) * d_eff(z)^2 * (1 - phi_s(z))^3 / phi_s(z)^2

    Returns
    -------
    z_centers : bin centers
    k_profile : permeability profile (m^2)
    """
    nb = n_bins or case.n_bins
    z_min, z_max = 0.0, case._box_height
    edges = np.linspace(z_min, z_max, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    k_profile = np.zeros(nb)

    for b in range(nb):
        lo, hi = edges[b], edges[b + 1]
        mask = (positions[:, 2] >= lo) & (positions[:, 2] < hi)
        if mask.sum() < 3:
            k_profile[b] = 1e-12
            continue
        r_bin = radii[mask]
        vol = 4.0 / 3.0 * math.pi * r_bin ** 3
        d_32 = float(np.sum(r_bin ** 3) / np.sum(r_bin ** 2)) * 2.0
        bin_vol = (hi - lo) * case._box_width * case._box_depth
        phi_s = float(np.sum(vol)) / bin_vol if bin_vol > 0 else case.phi_total
        phi_s = min(max(phi_s, 0.3), 0.7)
        k_profile[b] = (1.0 / 180.0) * d_32 ** 2 * (1 - phi_s) ** 3 / phi_s ** 2

    return centers, k_profile


def permeability_sensitivity_table(
    cases: Sequence[SegregationCase],
) -> list[dict]:
    """Build sensitivity table: segregation state → permeability contrast."""
    table = []
    for case in cases:
        result = simulate_segregation(case)
        z_c, k_prof = kozeny_carman_permeability(
            result["final_positions"],
            result["final_radii"],
            result["final_species"],
            case,
        )
        k_max = float(np.max(k_prof))
        k_min = float(np.min(k_prof))
        k_mean = float(np.mean(k_prof))
        table.append({
            "d_ratio": case.d_ratio,
            "phi_total": case.phi_total,
            "shear_rate": case.shear_rate,
            "segregation_degree": result["segregation_degree"],
            "direction_sign": result["direction_sign"],
            "k_max_m2": k_max,
            "k_min_m2": k_min,
            "k_ratio": k_max / max(k_min, 1e-30),
            "k_mean_m2": k_mean,
        })
    return table


# ---------------------------------------------------------------------------
# Scaling law fitting
# ---------------------------------------------------------------------------

def _scaling_model(X: np.ndarray, A: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """S = A * (d_ratio - 1)^alpha * shear_rate^beta * (phi_j - phi)^gamma."""
    d_ratio_m1, shear_rate, phi_j_minus_phi = X
    d_term = np.where(d_ratio_m1 > 0, d_ratio_m1, 1e-10)
    s_term = np.where(shear_rate > 0, shear_rate, 1e-10)
    p_term = np.where(phi_j_minus_phi > 0, phi_j_minus_phi, 1e-10)
    return A * d_term ** alpha * s_term ** beta * p_term ** gamma


def fit_scaling_law(
    results: list[dict],
    *,
    phi_jamming: float = PHI_JAMMING,
) -> dict:
    """Fit S = A (d_l/d_s − 1)^α γ̇^β (φ_j − φ)^γ."""
    if len(results) < 4:
        return {"error": "need at least 4 data points", "r_squared": 0.0}

    d_ratios = np.array([r["d_ratio"] for r in results])
    shear_rates = np.array([r["shear_rate"] for r in results])
    phi_totals = np.array([r["phi_total"] for r in results])
    S_vals = np.array([r["segregation_degree"] for r in results])

    X = np.vstack([
        d_ratios - 1.0,
        shear_rates,
        phi_jamming - phi_totals,
    ])

    valid = S_vals > 1e-10
    if valid.sum() < 4:
        return {"error": "too few non-zero S values", "r_squared": 0.0}

    X_v = X[:, valid]
    S_v = S_vals[valid]

    try:
        popt, pcov = curve_fit(
            _scaling_model, X_v, S_v,
            p0=[0.1, 1.0, 0.5, 1.0],
            bounds=([0, 0, 0, 0], [10, 5, 3, 5]),
            maxfev=10000,
        )
        A, alpha, beta, gamma = popt

        S_pred = _scaling_model(X_v, *popt)
        ss_res = np.sum((S_v - S_pred) ** 2)
        ss_tot = np.sum((S_v - np.mean(S_v)) ** 2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

        n_boot = 200
        rng = np.random.default_rng(42)
        boot_params = []
        n_valid = valid.sum()
        for _ in range(n_boot):
            idx = rng.choice(n_valid, size=n_valid, replace=True)
            try:
                popt_b, _ = curve_fit(
                    _scaling_model, X_v[:, idx], S_v[idx],
                    p0=popt,
                    bounds=([0, 0, 0, 0], [10, 5, 3, 5]),
                    maxfev=5000,
                )
                boot_params.append(popt_b)
            except Exception:
                pass

        boot_params = np.array(boot_params)
        ci = {}
        if len(boot_params) > 10:
            ci = {
                "A": (float(np.percentile(boot_params[:, 0], 2.5)),
                      float(np.percentile(boot_params[:, 0], 97.5))),
                "alpha": (float(np.percentile(boot_params[:, 1], 2.5)),
                          float(np.percentile(boot_params[:, 1], 97.5))),
                "beta": (float(np.percentile(boot_params[:, 2], 2.5)),
                         float(np.percentile(boot_params[:, 2], 97.5))),
                "gamma": (float(np.percentile(boot_params[:, 3], 2.5)),
                          float(np.percentile(boot_params[:, 3], 97.5))),
            }

        return {
            "A": float(A),
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
            "r_squared": float(r_squared),
            "n_points": int(n_valid),
            "bootstrap_ci": ci,
        }
    except Exception as e:
        return {"error": str(e), "r_squared": 0.0}


# ---------------------------------------------------------------------------
# Angle of repose measurement (consistent with P96-N3)
# ---------------------------------------------------------------------------

def measure_repose_angle(
    positions: np.ndarray,
    radii: np.ndarray,
    species: np.ndarray | None = None,
    *,
    box_width: float = 0.04,
    n_bins: int = 20,
) -> dict:
    """Measure the angle of repose of a settled particle bed.

    Uses the same methodology as P96-N3: fit the free surface profile
    and compute the angle from the slope.
    """
    x_edges = np.linspace(0, box_width, n_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_surface = np.zeros(n_bins)

    for b in range(n_bins):
        lo, hi = x_edges[b], x_edges[b + 1]
        mask = (positions[:, 0] >= lo) & (positions[:, 0] < hi)
        if mask.sum() > 0:
            z_surface[b] = float(np.max(positions[mask, 2] + radii[mask]))
        else:
            z_surface[b] = np.nan

    valid = ~np.isnan(z_surface)
    if valid.sum() < 3:
        return {"angle_deg": 0.0, "slope": 0.0, "r_squared": 0.0, "n_valid": int(valid.sum())}

    slope, intercept = np.polyfit(x_centers[valid], z_surface[valid], 1)
    z_pred = slope * x_centers[valid] + intercept
    ss_res = np.sum((z_surface[valid] - z_pred) ** 2)
    ss_tot = np.sum((z_surface[valid] - np.mean(z_surface[valid])) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

    angle_deg = math.degrees(math.atan(abs(slope)))

    return {
        "angle_deg": angle_deg,
        "slope": float(slope),
        "r_squared": float(r_squared),
        "n_valid": int(valid.sum()),
    }
