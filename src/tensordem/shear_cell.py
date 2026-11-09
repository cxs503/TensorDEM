"""Direct-shear cell for coarse-graining verification (numpy kernel).

Drives a rectangular shear box with a moving top platen to measure the
macroscopic shear response.  Uses the lightweight ``granular3d_core`` kernel
so runs can be sharded across cluster nodes without a torch dependency.

Protocol
--------
1. Fill box with HCP packing, settle under gravity.
2. Apply normal load via top platen (constant stress or constant height).
3. Shear: move top platen laterally at constant velocity.
4. Record shear stress vs normal stress → Mohr-Coulomb fit → friction angle.

All quantities are SI (m, kg, s, N, Pa).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .granular3d_core import (
    Material3D,
    ContactState,
    build_pairs,
    contact_forces,
    hcp_positions_in_box,
)


@dataclass(frozen=True)
class ShearCellCase:
    """Direct-shear cell parameters (SI)."""
    # material
    radius: float = 0.005
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.05
    restitution: float = 0.3
    kn: float = 1.0e5
    kt_ratio: float = 0.5
    gravity: float = 9.81
    # geometry
    box_width: float = 0.1      # x direction (shear direction)
    box_depth: float = 0.05     # y direction
    fill_height: float = 0.1    # z direction
    packing_gap: float = 0.02
    jitter: float = 0.01
    # numerics
    dt: Optional[float] = None
    seed: int = 6107
    # protocol
    settle_time: float = 0.5
    normal_stress: float = 5000.0   # Pa, applied by top platen
    shear_velocity: float = 0.01    # m/s, top platen lateral speed
    shear_strain_target: float = 0.2  # total engineering shear strain
    wall_mu: float = 0.0            # side wall friction

    @property
    def mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    @property
    def kr(self) -> float:
        return self.kt * self.radius ** 2

    @property
    def zeta(self) -> float:
        e = max(min(self.restitution, 1.0 - 1e-12), 1e-6)
        ln_e = math.log(e)
        return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)

    @property
    def gamma_pair(self) -> float:
        return 2.0 * self.zeta * math.sqrt(self.kn * 0.5 * self.mass)

    @property
    def gamma_wall(self) -> float:
        return 2.0 * self.zeta * math.sqrt(self.kn * self.mass)

    @property
    def timestep(self) -> float:
        if self.dt is not None:
            return self.dt
        period = 2.0 * math.pi * math.sqrt((0.5 * self.mass) / self.kn)
        return period / 20.0

    @property
    def shear_time(self) -> float:
        """Time to reach target shear strain."""
        return self.shear_strain_target * self.fill_height / self.shear_velocity


def _damping_ratio(restitution: float) -> float:
    e = max(min(restitution, 1.0 - 1e-12), 1e-6)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)


def simulate_shear_cell(case: ShearCellCase) -> dict:
    """Run a direct-shear simulation and return macroscopic response.

    Returns
    -------
    dict with:
        normal_stress_applied: Pa
        shear_stress_history: list of Pa
        normal_stress_history: list of Pa (measured)
        shear_strain_history: list
        friction_angle_deg: float (from Mohr-Coulomb fit)
        r_squared: float (linear fit quality)
        n_particles: int
        total_steps: int
    """
    r = case.radius
    mass = case.mass
    inertia = case.inertia
    kn = case.kn
    kt = case.kt
    kr = case.kr
    mu = case.mu
    mu_r = case.mu_rolling
    g = case.gravity
    dt = case.timestep
    zeta = _damping_ratio(case.restitution)
    gamma = 2.0 * zeta * math.sqrt(kn * 0.5 * mass)
    gamma_w = 2.0 * zeta * math.sqrt(kn * mass)

    # Initial packing
    pos = hcp_positions_in_box(
        r, case.box_width, case.box_depth, case.fill_height,
        packing_gap=case.packing_gap, jitter=case.jitter, seed=case.seed
    )
    n = len(pos)
    vel = np.zeros((n, 3), dtype=np.float64)
    omega = np.zeros((n, 3), dtype=np.float64)

    state = ContactState()

    # Identify top and bottom particles for platen control
    z_max = pos[:, 2].max()
    z_min = pos[:, 2].min()
    top_mask = pos[:, 2] > z_max - 2.5 * r
    bottom_mask = pos[:, 2] < z_min + 2.5 * r
    top_indices = np.where(top_mask)[0]
    bottom_indices = np.where(bottom_mask)[0]

    # Normal force from applied stress
    area = case.box_width * case.box_depth
    normal_force = case.normal_stress * area
    # Distribute among top particles
    n_top = len(top_indices)
    force_per_top = normal_force / max(n_top, 1)

    # Shear protocol
    total_time = case.settle_time + case.shear_time
    total_steps = int(math.ceil(total_time / dt))
    settle_steps = int(math.ceil(case.settle_time / dt))

    shear_stress_history = []
    normal_stress_history = []
    shear_strain_history = []
    top_displacement_x = 0.0

    # Platen positions
    platen_z = z_max + r  # just above top particles
    platen_x_offset = 0.0

    for step in range(total_steps):
        t_now = step * dt
        is_shearing = t_now > case.settle_time

        if is_shearing:
            shear_t = t_now - case.settle_time
            platen_x_offset = case.shear_velocity * shear_t
            top_displacement_x = platen_x_offset

        # Build pairs
        iu, ju = build_pairs(pos, r)

        # Contact forces
        force, torque = contact_forces(
            pos, vel, omega, iu, ju, state,
            Material3D(
                radius=r, density=case.density, mu=mu, mu_rolling=mu_r,
                restitution=case.restitution, kn=kn, kt_ratio=case.kt_ratio,
                gravity=g
            ),
            dt
        )

        # Gravity
        force[:, 2] -= mass * g

        # Bottom wall: fixed
        bottom_z = r
        for i in bottom_indices:
            if pos[i, 2] < bottom_z + r:
                overlap = bottom_z + r - pos[i, 2]
                if overlap > 0:
                    f_n = kn * overlap - gamma_w * vel[i, 2]
                    f_n = max(f_n, 0.0)
                    force[i, 2] += f_n
                    # Friction on bottom
                    if mu > 0:
                        v_t = vel[i, :2].copy()
                        f_t = -kt * v_t * dt
                        f_t_mag = np.linalg.norm(f_t)
                        f_limit = mu * f_n
                        if f_t_mag > f_limit:
                            f_t = f_t * f_limit / f_t_mag
                        force[i, :2] += f_t

        # Side walls
        for dim in [0, 1]:
            lo = r if dim == 0 else r
            hi = case.box_width - r if dim == 0 else case.box_depth - r
            for i in range(n):
                if i in top_indices or i in bottom_indices:
                    continue
                if pos[i, dim] < lo + r:
                    overlap = lo + r - pos[i, dim]
                    if overlap > 0:
                        f_n = kn * overlap - gamma_w * vel[i, dim]
                        f_n = max(f_n, 0.0)
                        force[i, dim] += f_n
                elif pos[i, dim] > hi - r:
                    overlap = pos[i, dim] - (hi - r)
                    if overlap > 0:
                        f_n = kn * overlap + gamma_w * vel[i, dim]
                        f_n = max(f_n, 0.0)
                        force[i, dim] -= f_n

        # Top platen: apply normal force + shear displacement
        for i in top_indices:
            # Normal force downward
            force[i, 2] -= force_per_top
            # Shear: impose x velocity
            if is_shearing:
                vel[i, 0] = case.shear_velocity
                # Also constrain z to maintain contact
                vel[i, 2] = min(vel[i, 2], 0.0)

        # Integration (Velocity Verlet)
        acc = force / mass[:, None]
        ang_acc = torque / inertia

        pos = pos + vel * dt + 0.5 * acc * dt ** 2
        vel = vel + acc * dt
        omega = omega + ang_acc * dt

        # Record during shearing
        if is_shearing and step % 50 == 0:
            shear_strain = top_displacement_x / case.fill_height
            # Measure shear stress from total x-force on top particles
            shear_force = sum(force[i, 0] for i in top_indices)
            shear_stress = abs(shear_force) / area
            # Measured normal stress from z-force balance
            normal_force_measured = sum(abs(force[i, 2]) for i in top_indices) / max(n_top, 1)
            normal_stress_measured = normal_force_measured * n_top / area

            shear_stress_history.append(float(shear_stress))
            normal_stress_history.append(float(normal_stress_measured))
            shear_strain_history.append(float(shear_strain))

    # Fit Mohr-Coulomb: τ = c + σ·tan(φ)
    if len(shear_stress_history) > 10:
        # Use steady-state portion (last 60%)
        n_ss = int(0.6 * len(shear_stress_history))
        ss_tau = np.array(shear_stress_history[-n_ss:])
        ss_sigma = np.array(normal_stress_history[-n_ss:])

        # Linear fit
        A = np.vstack([ss_sigma, np.ones_like(ss_sigma)]).T
        result = np.linalg.lstsq(A, ss_tau, rcond=None)
        tan_phi = result[0][0]
        cohesion = result[0][1]
        friction_angle = math.degrees(math.atan(tan_phi))

        # R²
        predicted = A @ result[0]
        ss_res = np.sum((ss_tau - predicted) ** 2)
        ss_tot = np.sum((ss_tau - np.mean(ss_tau)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        friction_angle = 0.0
        r_squared = 0.0
        cohesion = 0.0
        tan_phi = 0.0

    return {
        "normal_stress_applied": case.normal_stress,
        "shear_stress_history": shear_stress_history,
        "normal_stress_history": normal_stress_history,
        "shear_strain_history": shear_strain_history,
        "friction_angle_deg": friction_angle,
        "cohesion_pa": cohesion,
        "tan_phi": tan_phi,
        "r_squared": r_squared,
        "n_particles": n,
        "total_steps": total_steps,
        "top_displacement_x": top_displacement_x,
    }
