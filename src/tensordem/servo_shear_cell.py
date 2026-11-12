"""Servo-controlled plane-shear cell for mu(I) rheology calibration.

CASE-DEM-PLANE-SHEAR-MUIDI: implements a constant-pressure servo shear cell
that reproduces the glass-bead mu(I) curve of da Cruz et al. (2005) PRE
72:021309 / GDR MiDi (2004) JFM 509:165.

Physics
-------
The mu(I) rheology law:

    mu(I) = mu_s + (mu_2 - mu_s) / (I_0 / I + 1)

where I = gamma_dot * d / sqrt(P / rho_p) is the inertial number.

The servo cell maintains a target normal stress P_target by adjusting the
top wall position via a PID-like controller:

    v_wall = K_p * (P_target - P_measured) + K_d * dP/dt

This eliminates the fixed-wall artefact where the actual pressure drifts
far from the intended value, which was the primary source of the >20%
errors in the previous implementation.

Stress measurement uses the wall-force estimator:

    tau = F_x_top / A,  P = F_z_top / A

Reference
---------
da Cruz, F., Emam, S., Prochnow, M., Roux, J.-N. & Chevoir, F. (2005).
Rheophysics of dense granular materials: Discrete simulation of plane
shear flows. Physical Review E, 72, 021309.
DOI: 10.1103/PhysRevE.72.021309

GDR MiDi (2004). On dense granular flows. J. Fluid Mech. 509, 165-175.
DOI: 10.1017/S002211200400933X
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "ServoShearCase",
    "ServoShearResult",
    "simulate_servo_shear",
    "mu_i_law",
    "phi_i_law",
    "inertial_number",
]

# ---------------------------------------------------------------------------
# Reference mu(I) constants (da Cruz et al. 2005, glass beads)
# ---------------------------------------------------------------------------
MU_S_REF = 0.36
MU_2_REF = 0.84
I_0_REF = 0.30
PHI_C_REF = 0.58
PHI_J_REF = 0.64


def mu_i_law(I: float, mu_s: float = MU_S_REF, mu_2: float = MU_2_REF,
             I_0: float = I_0_REF) -> float:
    if I <= 0:
        return mu_s
    return mu_s + (mu_2 - mu_s) / (I_0 / I + 1.0)


def phi_i_law(I: float, phi_c: float = PHI_C_REF, phi_j: float = PHI_J_REF,
              I_0: float = I_0_REF) -> float:
    if I <= 0:
        return phi_c
    return phi_c + (phi_j - phi_c) / (I_0 / I + 1.0)


def inertial_number(gamma_dot: float, d: float, P: float, rho_p: float) -> float:
    if P <= 0 or rho_p <= 0:
        return 0.0
    return gamma_dot * d / math.sqrt(P / rho_p)


@dataclass(frozen=True)
class ServoShearCase:
    radius: float = 0.005
    density: float = 2500.0
    mu_particle: float = 0.5
    mu_rolling: float = 0.05
    restitution: float = 0.3
    kn: float = 1.0e5
    kt_ratio: float = 0.5
    box_width: float = 0.06
    box_depth: float = 0.03
    n_particles: int = 200
    shear_rate: float = 5.0
    P_target: float = 62500.0
    servo_Kp: float = 0.01
    servo_Kd: float = 0.001
    max_time: float = 2.0
    dt: Optional[float] = None
    sample_every: int = 50
    settle_fraction: float = 0.25
    seed: int = 5279

    @property
    def particle_diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def particle_mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    @property
    def timestep(self) -> float:
        if self.dt is not None:
            return self.dt
        period = 2.0 * math.pi * math.sqrt(0.5 * self.particle_mass / self.kn)
        return period / 20.0


@dataclass
class ServoShearResult:
    mu_measured: float
    P_measured: float
    tau_measured: float
    I_measured: float
    phi_bed_steady: float
    phi_bed_initial: float
    coordination_number: float
    wall_clock_s: float
    history: list


def _initial_positions(case: ServoShearCase) -> np.ndarray:
    rng = np.random.default_rng(case.seed)
    r = case.radius
    d = 2.0 * r
    area = case.box_width * case.box_depth
    phi_init = 0.55
    vol_p = 4.0 / 3.0 * math.pi * r ** 3
    h_init = case.n_particles * vol_p / (area * phi_init)

    pos = np.zeros((case.n_particles, 3), dtype=np.float64)
    pos[:, 0] = rng.uniform(r, case.box_width - r, case.n_particles)
    pos[:, 1] = rng.uniform(r, case.box_depth - r, case.n_particles)
    pos[:, 2] = rng.uniform(r, h_init - r, case.n_particles)

    # Quick overlap removal
    for _ in range(100):
        for i in range(case.n_particles):
            for j in range(i + 1, case.n_particles):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                dx -= case.box_width * round(dx / case.box_width)
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < d and dist > 1e-12:
                    ov = (d - dist) * 0.5
                    nx, ny, nz = dx / dist, dy / dist, dz / dist
                    pos[i] -= ov * np.array([nx, ny, nz])
                    pos[j] += ov * np.array([nx, ny, nz])
    return pos


def _damping_from_restitution(e: float) -> float:
    e = max(min(e, 1.0 - 1e-12), 1e-6)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)


def simulate_servo_shear(case: ServoShearCase) -> ServoShearResult:
    t0 = time.perf_counter()
    pos = _initial_positions(case)
    n = case.n_particles
    r = case.radius
    d = 2.0 * r
    mass = case.particle_mass
    dt = case.timestep
    kn = case.kn
    kt = case.kt
    mu_p = case.mu_particle

    zeta = _damping_from_restitution(case.restitution)
    gamma_damp = 2.0 * zeta * math.sqrt(kn * 0.5 * mass)

    vel = np.zeros_like(pos)
    box_w = case.box_width
    box_d = case.box_depth
    gamma_dot = case.shear_rate

    top_z = float(pos[:, 2].max()) + r + 0.002
    prev_P = case.P_target

    steps = int(math.ceil(case.max_time / dt))
    settle_steps = int(case.settle_fraction * steps)
    history: list = []
    n_contacts = 0

    for step in range(steps):
        t = step * dt
        pos[:, 0] = np.remainder(pos[:, 0], box_w)

        force = np.zeros_like(pos)

        # Cell list for O(n) contact detection
        cell_size = d * 1.5
        nx_cells = max(1, int(math.ceil(box_w / cell_size)))
        ny_cells = max(1, int(math.ceil(box_d / cell_size)))
        nz_cells = max(1, int(math.ceil((top_z - r) / cell_size)))
        cells: dict[tuple[int, int, int], list[int]] = {}
        for i in range(n):
            cx = int(pos[i, 0] / cell_size) % nx_cells
            cy = int(pos[i, 1] / cell_size) % ny_cells
            cz = int(pos[i, 2] / cell_size)
            cells.setdefault((cx, cy, cz), []).append(i)

        nc = 0
        for cx in range(nx_cells):
            for cy in range(ny_cells):
                for cz in range(nz_cells):
                    key = (cx, cy, cz)
                    if key not in cells:
                        continue
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for dz in (-1, 0, 1):
                                nkey = ((cx + dx) % nx_cells, (cy + dy) % ny_cells, cz + dz)
                                if nkey not in cells:
                                    continue
                                for i in cells[key]:
                                    for j in cells[nkey]:
                                        if j <= i:
                                            continue
                                        ddx = pos[j, 0] - pos[i, 0]
                                        ddy = pos[j, 1] - pos[i, 1]
                                        ddz = pos[j, 2] - pos[i, 2]
                                        ddx -= box_w * round(ddx / box_w)
                                        dist2 = ddx * ddx + ddy * ddy + ddz * ddz
                                        if dist2 < d * d and dist2 > 1e-30:
                                            nc += 1
                                            dist = math.sqrt(dist2)
                                            ov = d - dist
                                            nx_c = ddx / dist
                                            ny_c = ddy / dist
                                            nz_c = ddz / dist
                                            v_rel = vel[j] - vel[i]
                                            v_n = v_rel[0] * nx_c + v_rel[1] * ny_c + v_rel[2] * nz_c
                                            fn = max(0.0, kn * ov - gamma_damp * v_n)
                                            # Tangential
                                            v_tx = v_rel[0] - v_n * nx_c
                                            v_ty = v_rel[1] - v_n * ny_c
                                            v_tz = v_rel[2] - v_n * nz_c
                                            vt_mag = math.sqrt(v_tx * v_tx + v_ty * v_ty + v_tz * v_tz)
                                            if mu_p > 0 and vt_mag > 1e-15:
                                                ft = min(mu_p * fn, kt * vt_mag * dt)
                                                fx = -ft * v_tx / vt_mag
                                                fy = -ft * v_ty / vt_mag
                                                fz = -ft * v_tz / vt_mag
                                            else:
                                                fx = fy = fz = 0.0
                                            fnx = fn * nx_c + fx
                                            fny = fn * ny_c + fy
                                            fnz = fn * nz_c + fz
                                            force[i, 0] += fnx; force[i, 1] += fny; force[i, 2] += fnz
                                            force[j, 0] -= fnx; force[j, 1] -= fny; force[j, 2] -= fnz
        n_contacts = nc

        # Wall contacts (y walls)
        for wy, sign in [(0.0, 1.0), (box_d, -1.0)]:
            pen = r - (pos[:, 1] - wy) * sign
            hit = pen > 0
            if hit.any():
                idx = np.where(hit)[0]
                fn_w = kn * pen[idx]
                force[idx, 1] += fn_w * sign

        # Top wall
        top_pen = pos[:, 2] + r - top_z
        top_hit = top_pen > 0
        top_fz = 0.0
        top_fx = 0.0
        if top_hit.any():
            idx = np.where(top_hit)[0]
            fn_w = kn * top_pen[idx]
            top_fz = float(fn_w.sum())
            force[idx, 2] -= fn_w
            if mu_p > 0:
                v_rel_x = vel[idx, 0] - gamma_dot * top_z
                ft_w = mu_p * fn_w
                for k, ii in enumerate(idx):
                    s = -1.0 if v_rel_x[k] > 0 else 1.0
                    force[ii, 0] += ft_w[k] * s
                    top_fx += ft_w[k] * s

        # Bottom wall
        bot_pen = r - pos[:, 2]
        bot_hit = bot_pen > 0
        if bot_hit.any():
            idx = np.where(bot_hit)[0]
            fn_w = kn * bot_pen[idx]
            force[idx, 2] += fn_w

        # Servo control
        area = box_w * box_d
        P_meas = top_fz / area if area > 0 else 0.0
        dP = P_meas - prev_P
        top_vz = case.servo_Kp * (case.P_target - P_meas) - case.servo_Kd * dP / max(dt, 1e-12)
        top_z += top_vz * dt
        top_z = max(top_z, float(pos[:, 2].max()) + r + 1e-6)
        prev_P = P_meas

        # Integration
        vel += force * (dt / mass)
        pos += vel * dt

        # Sample
        if step % case.sample_every == 0 and step > settle_steps:
            tau = abs(top_fx) / area if area > 0 else 0.0
            mu_eff = tau / P_meas if P_meas > 1.0 else 0.0
            history.append((t, tau, P_meas, mu_eff))

    wall_clock = time.perf_counter() - t0

    if len(history) >= 3:
        ns = max(1, int(0.6 * len(history)))
        steady = history[-ns:]
        avg_tau = sum(h[1] for h in steady) / ns
        avg_P = sum(h[2] for h in steady) / ns
        avg_mu = sum(h[3] for h in steady) / ns
    else:
        avg_tau = avg_P = avg_mu = 0.0

    vol_p = n * 4.0 / 3.0 * math.pi * r ** 3
    phi_steady = vol_p / (area * (top_z - r)) if top_z > r else 0.0
    phi_init = vol_p / (area * (float(pos[:, 2].max()) + r)) if n > 0 else 0.0
    avg_Z = 2.0 * n_contacts / n if n > 0 else 0.0
    I_meas = inertial_number(gamma_dot, d, avg_P, case.density)

    return ServoShearResult(
        mu_measured=avg_mu, P_measured=avg_P, tau_measured=avg_tau,
        I_measured=I_meas, phi_bed_steady=phi_steady, phi_bed_initial=phi_init,
        coordination_number=avg_Z, wall_clock_s=wall_clock, history=history,
    )
