"""μ(I) rheology benchmark for TensorDEM granular flow.

Physical background
-------------------
Dense granular flows exhibit a rate-dependent friction law known as the μ(I)
rheology (Jop, Forterre & Chaucha, Nature 2006; GDR MiDi, JFM 2004):

    μ(I) = μ_s + (μ_2 - μ_s) / (I_0 / I + 1)

where:
    μ(I)  : effective friction coefficient (τ/P)
    I     : inertial number = γ̇ d / √(P/ρ_p)
    μ_s   : static friction coefficient (I → 0 limit)
    μ_2   : dynamic friction coefficient (I → ∞ limit)
    I_0   : characteristic inertial number (typically ~0.3)
    γ̇    : shear rate (1/s)
    d     : particle diameter (m)
    P     : confining pressure (Pa)
    ρ_p   : particle material density (kg/m³)

Implementation
--------------
A direct shear box is simulated: a rectangular cell filled with granular
particles, with periodic boundary conditions in the shear direction (x),
rigid walls in the transverse direction (y), and a moving top wall that
applies a constant shear rate γ̇ while maintaining a constant normal stress P.

All quantities are SI (m, kg, s, Pa).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch

DTYPE = torch.float64


class MuIRheologyError(ValueError):
    """Base class for μ(I) rheology errors."""


@dataclass(frozen=True)
class MuICase:
    radius: float = 0.005
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.05
    restitution: float = 0.3
    kn: float = 1.0e5
    kt_ratio: float = 0.5
    box_width: float = 0.06
    box_depth: float = 0.03
    fill_height: float = 0.04
    shear_rate: float = 5.0
    normal_stress: float = 5000.0
    packing_gap: float = 0.02
    jitter: float = 0.01
    dt: Optional[float] = None
    seed: int = 5279
    max_time: float = 2.0
    sample_interval: int = 50

    @property
    def particle_mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.particle_mass * self.radius ** 2

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    @property
    def timestep(self) -> float:
        if self.dt is not None:
            return self.dt
        period = 2.0 * math.pi * math.sqrt((0.5 * self.particle_mass) / self.kn)
        return period / 20.0

    @property
    def particle_diameter(self) -> float:
        return 2.0 * self.radius

    def __post_init__(self) -> None:
        for name in ("radius", "density", "kn", "kt_ratio",
                     "box_width", "box_depth", "fill_height",
                     "shear_rate", "normal_stress", "max_time"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value > 0.0):
                raise MuIRheologyError(f"{name} must be finite and positive")
        for name in ("mu", "mu_rolling", "restitution", "packing_gap", "jitter"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0.0):
                raise MuIRheologyError(f"{name} must be finite and nonnegative")
        if not 0.0 <= self.restitution < 1.0:
            raise MuIRheologyError("restitution must lie in [0, 1)")
        if self.dt is not None and (not math.isfinite(self.dt) or self.dt <= 0.0):
            raise MuIRheologyError("dt must be finite and positive")


def _hcp_packing_in_box(case: MuICase) -> torch.Tensor:
    r = case.radius
    spacing = 2.0 * r * (1.0 + case.packing_gap)
    layer_dz = spacing * math.sqrt(2.0 / 3.0)
    n_layers = max(1, int(math.floor((case.fill_height - r) / layer_dz)) + 1)
    generator = torch.Generator().manual_seed(case.seed)
    row_pitch = spacing * math.sqrt(3.0) / 2.0
    shift_x = 0.5 * spacing
    shift_y = spacing / (2.0 * math.sqrt(3.0))
    points: list[tuple[float, float, float]] = []
    x_min = r + 0.001
    x_max = case.box_width - r - 0.001
    y_min = r + 0.001
    y_max = case.box_depth - r - 0.001
    for layer in range(n_layers):
        z = r + layer * layer_dz
        odd = layer % 2 == 1
        span_x = int(math.ceil((case.box_width + spacing) / spacing))
        span_y = int(math.ceil((case.box_depth + spacing) / spacing))
        for ix in range(-span_x, span_x + 1):
            for iy in range(-span_y, span_y + 1):
                x = ix * spacing + 0.5 * spacing * (iy % 2)
                y = iy * row_pitch
                if odd:
                    x += shift_x
                    y += shift_y
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    points.append((x, y, z))
    positions = torch.tensor(points, dtype=DTYPE)
    if len(positions) < 8:
        raise MuIRheologyError("packing produced fewer than 8 particles")
    if case.jitter:
        noise = torch.randn(positions.shape, dtype=DTYPE, generator=generator)
        positions = positions + noise * (case.jitter * r)
    positions[:, 2].clamp_(min=case.radius * 1.001)
    return positions


def _damping_ratio(restitution: float) -> float:
    e = max(min(restitution, 1.0 - 1e-12), 1e-6)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)


def _pair_index(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.triu_indices(n, n, offset=1)


def _safe_normalize(v: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    norm = torch.linalg.vector_norm(v, dim=1, keepdim=True)
    return torch.where(norm > eps, v / norm, torch.zeros_like(v))


def simulate_shear(case: MuICase, *, return_history: bool = False) -> dict:
    positions = _hcp_packing_in_box(case)
    n = len(positions)
    r = case.radius
    mass = case.particle_mass
    inertia = case.inertia
    dt = case.timestep
    kn, kt = case.kn, case.kt
    zeta = _damping_ratio(case.restitution)
    gamma_damp = 2.0 * zeta * math.sqrt(kn * 0.5 * mass)
    gamma_wall = 2.0 * zeta * math.sqrt(kn * mass)

    velocities = torch.zeros_like(positions)
    omega = torch.zeros_like(positions)

    box_w = case.box_width
    box_d = case.box_depth
    gamma_dot = case.shear_rate
    top_z = case.fill_height
    top_vx = gamma_dot * top_z

    steps = int(math.ceil(case.max_time / dt))
    history: list[tuple[float, float, float, float]] = []
    settle_steps = min(500, steps // 4)

    for step in range(steps):
        # Periodic BC in x
        positions[:, 0] = torch.remainder(positions[:, 0], box_w)

        force = torch.zeros_like(positions)
        torque = torch.zeros_like(positions)

        # Particle-particle contacts with periodic x
        # all distinct pairs (i < j), the upstream ordering of the pair forces
        aiu, aju = _pair_index(n)
        dvec = positions[aju] - positions[aiu]
        # Apply minimum image in x
        dvec[:, 0] = dvec[:, 0] - box_w * torch.round(dvec[:, 0] / box_w)
        dist = torch.linalg.vector_norm(dvec, dim=1)
        overlap = 2.0 * r - dist
        touching = overlap > 0.0

        if touching.any():
            t_idx = torch.nonzero(touching, as_tuple=False).flatten()
            n_vec = _safe_normalize(dvec[t_idx])
            v_rel = velocities[aju[t_idx]] - velocities[aiu[t_idx]]
            v_n = (v_rel * n_vec).sum(dim=1)
            fn = torch.clamp(kn * overlap[t_idx] - gamma_damp * v_n, min=0.0)

            v_t = v_rel - v_n[:, None] * n_vec
            if case.mu > 0:
                ft_mag = case.mu * fn
                ft_dir = _safe_normalize(-v_t)
                ft = ft_mag[:, None] * ft_dir
            else:
                ft = torch.zeros_like(v_t)

            f_total = fn[:, None] * n_vec + ft
            force.index_add_(0, aiu[t_idx], f_total)
            force.index_add_(0, aju[t_idx], -f_total)

            if case.mu_rolling > 0:
                omega_rel = omega[aju[t_idx]] - omega[aiu[t_idx]]
                tr_mag = case.mu_rolling * r * fn
                tr_dir = _safe_normalize(-omega_rel)
                tr = tr_mag[:, None] * tr_dir
                torque.index_add_(0, aiu[t_idx], tr)
                torque.index_add_(0, aju[t_idx], -tr)

        # Side walls (y direction)
        y_pen_left = r - positions[:, 1]
        y_pen_right = positions[:, 1] + r - box_d
        for pen, sign in [(y_pen_left, 1.0), (y_pen_right, -1.0)]:
            hit = pen > 0.0
            if hit.any():
                idx = torch.nonzero(hit, as_tuple=False).flatten()
                normal = torch.tensor([[0.0, sign, 0.0]], dtype=DTYPE).expand(len(idx), 3)
                v_n = (velocities[idx] * normal).sum(dim=1)
                fn_w = torch.clamp(kn * pen[idx] - gamma_wall * v_n, min=0.0)
                force.index_add_(0, idx, fn_w[:, None] * normal)
                if case.mu > 0:
                    v_t = velocities[idx].clone()
                    v_t[:, 1] = 0.0
                    ft_w = case.mu * fn_w
                    ft_dir = _safe_normalize(-v_t)
                    force.index_add_(0, idx, ft_w[:, None] * ft_dir)

        # Top wall
        top_pen = positions[:, 2] + r - top_z
        top_hit = top_pen > 0.0
        top_force_z = 0.0
        top_force_x = 0.0
        if top_hit.any():
            idx = torch.nonzero(top_hit, as_tuple=False).flatten()
            normal = torch.tensor([[0.0, 0.0, -1.0]], dtype=DTYPE).expand(len(idx), 3)
            v_n = (velocities[idx] * normal).sum(dim=1)
            fn_w = torch.clamp(kn * top_pen[idx] - gamma_wall * v_n, min=0.0)
            force.index_add_(0, idx, fn_w[:, None] * normal)
            top_force_z = float(fn_w.sum())
            if case.mu > 0:
                v_rel_x = velocities[idx, 0] - top_vx
                ft_w = case.mu * fn_w
                ft_dir_x = torch.where(v_rel_x > 0, -1.0, 1.0)
                fx = ft_w * ft_dir_x
                force.index_add_(0, idx, fx[:, None] * torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE))
                top_force_x = float(fx.sum())

        # Bottom wall
        bottom_pen = r - positions[:, 2]
        bottom_hit = bottom_pen > 0.0
        if bottom_hit.any():
            idx = torch.nonzero(bottom_hit, as_tuple=False).flatten()
            normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE).expand(len(idx), 3)
            v_n = velocities[idx, 2]
            fn_w = torch.clamp(kn * bottom_pen[idx] - gamma_wall * v_n, min=0.0)
            force.index_add_(0, idx, fn_w[:, None] * normal)
            if case.mu > 0:
                v_t = velocities[idx].clone()
                v_t[:, 2] = 0.0
                ft_w = case.mu * fn_w
                ft_dir = _safe_normalize(-v_t)
                force.index_add_(0, idx, ft_w[:, None] * ft_dir)

        # Gravity
        force[:, 2] -= mass * 9.81

        # Semi-implicit Euler
        velocities = velocities + force * (dt / mass)
        omega = omega + torque * (dt / inertia)
        positions = positions + velocities * dt

        # Sample stress
        if step % case.sample_interval == 0 and step > settle_steps:
            area = box_w * box_d
            tau = top_force_x / area if area > 0 else 0.0
            P = top_force_z / area if area > 0 else 0.0
            mu_eff = tau / P if P > 0 else 0.0
            history.append((t_now := step * dt, tau, P, mu_eff))

    # Compute averages
    if len(history) >= 3:
        n_steady = max(1, int(0.6 * len(history)))
        steady = history[-n_steady:]
        avg_tau = sum(h[1] for h in steady) / n_steady
        avg_P = sum(h[2] for h in steady) / n_steady
        avg_mu = sum(h[3] for h in steady) / n_steady
    else:
        avg_tau = avg_P = avg_mu = 0.0

    d = 2.0 * case.radius
    rho_p = case.density
    I = gamma_dot * d / math.sqrt(avg_P / rho_p) if avg_P > 0 else 0.0

    mu_s = case.mu
    mu_2 = 0.7
    I_0 = 0.3
    mu_theoretical = mu_s + (mu_2 - mu_s) / (I_0 / I + 1) if I > 0 else mu_s

    result = {
        "n_particles": n,
        "mu_effective": avg_mu,
        "shear_stress_Pa": avg_tau,
        "normal_stress_Pa": avg_P,
        "shear_rate_imposed": gamma_dot,
        "inertial_number": I,
        "mu_theoretical": mu_theoretical,
        "mu_s": mu_s, "mu_2": mu_2, "I_0": I_0,
        "dt": dt, "steps": steps,
        "mu": case.mu, "mu_rolling": case.mu_rolling,
        "radius": r, "density": case.density,
    }
    if return_history:
        result["history"] = history
    return result


def jop_mu_I(I: float, mu_s: float, mu_2: float, I_0: float) -> float:
    if I <= 0:
        return mu_s
    return mu_s + (mu_2 - mu_s) / (I_0 / I + 1)


def fit_jop_law(I_values: list[float], mu_values: list[float]) -> dict[str, float]:
    import numpy as np
    I_arr = np.array(I_values)
    mu_arr = np.array(mu_values)
    best_rmse = float("inf")
    best_params = {}
    for mu_s in np.arange(0.2, 0.5, 0.02):
        for mu_2 in np.arange(0.5, 0.9, 0.02):
            for I_0 in np.arange(0.1, 0.6, 0.05):
                mu_pred = np.array([jop_mu_I(I, mu_s, mu_2, I_0) for I in I_arr])
                rmse = float(np.sqrt(np.mean((mu_arr - mu_pred) ** 2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {"mu_s": mu_s, "mu_2": mu_2, "I_0": I_0, "rmse": rmse}
    return best_params


__all__ = [
    "MuIRheologyError", "MuICase", "simulate_shear",
    "jop_mu_I", "fit_jop_law",
]


# ===========================================================================
# Servo-controlled (constant-stress) shear box — coarse-graining macro anchors
# ===========================================================================
#
# ``simulate_shear`` above drives a *fixed-height* lid, so the confining
# pressure is an output rather than a control.  For coarse-graining (CG) macro
# invariance studies the confining stress has to be a controlled variable: the
# inertial number I = gamma_dot d / sqrt(P/rho_p) mixes the prescribed shear
# rate with the emergent pressure, so an uncontrolled P leaves I ambiguous.
#
# ``simulate_servo_shear`` keeps the same direct-shear-box geometry (periodic
# x, walled y, frictional floor, rough top platen) but drives the platen with a
# constant-stress servo, and additionally returns the two microstructure
# anchors of the CG verification (coordination number Z, solid fraction phi).
# It is a standalone numpy kernel (scipy cKDTree + ghost images for the
# periodic direction) so a k-sweep can be sharded across cores.
#
# Conventions (all SI):
#   * normal      F_n = max(k_n delta - gamma_p v_n, 0)
#   * tangential  Coulomb sliding |F_t| = mu_i |F_n|, opposing slip
#   * rolling     |T_r| = mu_r R_eff |F_n|, opposing relative spin
#   * walls       floor + platen frictional (mu_wall), side walls smooth
#
# Servo law: plate_z += relax * (F_n^meas - F^target) / (k_n n_c) because
# F_n^meas ~ k_n n_c delta.

import numpy as np

from .granular3d_core import hcp_positions_in_box


@dataclass(frozen=True)
class ServoShearCase:
    """Constant-stress direct-shear box (CG macro-anchor device), SI units."""

    # material
    radius: float = 0.005
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.05
    restitution: float = 0.3
    kn: float = 1.0e3
    kt_ratio: float = 0.5
    gravity: float = 0.0
    # geometry (x periodic, y/z walled)
    box_width: float = 0.12
    box_depth: float = 0.05
    fill_height: float = 0.06
    # protocol
    normal_stress: float = 2.0e3
    shear_rate: float = 0.0
    inertial_number: Optional[float] = None
    shear_strain: float = 0.25
    wall_mu: float = 1.5
    # numerics / sampling
    packing_gap: float = 0.02
    jitter: float = 0.01
    seed: int = 5279
    dt: Optional[float] = None
    settle_steps: int = 3000
    sample_stride: int = 5
    min_shear_steps: int = 2000
    servo_relax: float = 0.2
    servo_stride: int = 10
    servo_max_step_d: float = 0.002
    bulk_margin_z: float = 0.25
    # constant-stress validity gate (m16671): the shear phase may run past the
    # requested strain budget until the servo has held the applied normal
    # stress within ``pressure_control_tol`` for ``pressure_recover_window``
    # steps.  Without it the platen stress drifts (measured +58 % / -17 % at
    # I >= 0.1) and the point is not a constant-stress measurement.
    pressure_control_tol: float = 0.05
    pressure_recover_window: int = 200
    max_recover_steps: int = 20000
    # bulk rheometry (round 6): the wall traction mu = tau_wall/p_wall is a
    # boundary quantity; this slab Virial estimate gives the interior stress so
    # the two can be compared on the same point.
    bulk_stress: bool = True
    bulk_y_margin: float = 0.2
    bulk_bins: int = 6
    # cross-validation diagnostic (m17500): also accumulate the *contact-local*
    # inertial-number proxy I_c = |v_t|/sqrt(f_n/(rho d^2)) and mu(I_c) on the
    # same samples.  Diagnostic only -- the integration (Coulomb cap at
    # ``mu``) is untouched, so every historical case is bit-for-bit unchanged.
    contact_proxy: bool = False

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
    def zeta(self) -> float:
        return _damping_ratio(self.restitution)

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
        return 2.0 * math.pi * math.sqrt(0.5 * self.mass / self.kn) / 20.0

    @property
    def particle_diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def kappa(self):
        """Campbell stiffness number ``k_n/(rho d^3 gdot^2)`` of this shear point.

        ``None`` before a shear rate is imposed (a static box has no kappa).
        """
        if not self.shear_rate > 0.0:
            return None
        from .elastic_limit_shear import kappa_of
        return kappa_of(self.kn, self.density, self.particle_diameter, self.shear_rate)

    @property
    def elastic_limit_regime(self):
        """Where this device sits on Campbell's stiffness-number map.

        A rate-independent (mu(I)) reading of the measured stress is only valid
        in the ``elastic-quasi-static`` band; at finite stiffness the friction
        coefficient keeps a shear-rate dependence and the measurement must be
        reported with its kappa.  This is a diagnostic, it does not change the
        integration.
        """
        kappa = self.kappa
        if kappa is None:
            return None
        from .elastic_limit_shear import elastic_limit_regime
        return elastic_limit_regime(kappa)

    @property
    def area(self) -> float:
        return self.box_width * self.box_depth

    def __post_init__(self) -> None:
        for name in ("radius", "density", "kn", "kt_ratio", "box_width",
                     "box_depth", "fill_height", "normal_stress"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value)
                    and value > 0.0):
                raise MuIRheologyError(f"{name} must be finite and positive")
        if not 0.0 <= self.restitution < 1.0:
            raise MuIRheologyError("restitution must lie in [0, 1)")
        if self.inertial_number is not None and not self.inertial_number > 0.0:
            raise MuIRheologyError("inertial_number must be positive when given")
        if self.shear_rate < 0.0:
            raise MuIRheologyError("shear_rate must be nonnegative")


def _periodic_pair_list(flat_pos: np.ndarray, n: int, radius: float):
    """Minimum-image pair list over a ghost-triplicated position array.

    ``flat_pos`` stacks the base cell and its +Lx / -Lx images.  Only pairs with
    the first endpoint in the base copy are kept, so every physical contact of
    the periodic cell appears exactly once.
    """
    from scipy.spatial import cKDTree

    pairs = cKDTree(flat_pos).query_pairs(2.0 * radius, output_type="ndarray")
    if len(pairs) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    a, b = pairs[:, 0], pairs[:, 1]
    keep = (a // n) == 0
    return a[keep].astype(np.int64), b[keep].astype(np.int64)


def _wall_spring(dxi: np.ndarray, v_t: np.ndarray, dt: float, kt: float,
                 fn: np.ndarray, mu_w: float, idx: np.ndarray):
    """Cundall-Strack tangential spring for a wall, saturated at mu_w |F_n|.

    ``dxi`` is the per-particle accumulated tangential offset (n, 2); it is
    updated in place for the contacting particles ``idx``.
    """
    mask = np.ones(dxi.shape[0], dtype=bool)
    mask[idx] = False
    dxi[mask] = 0.0
    dxi[idx] += v_t[idx] * dt
    f_t = -kt * dxi[idx]
    mag = np.linalg.norm(f_t, axis=1)
    cap = mu_w * fn
    over = mag > cap
    if over.any():
        scale = np.where(over, cap / np.maximum(mag, 1e-30), 1.0)
        f_t *= scale[:, None]
        dxi[idx] = -f_t / kt
    return f_t


def bulk_virial_stress(pos, vel, pairs, *, z_lo, z_hi,
                       y_lo, y_hi, box_width, n, mass, rule="both"):
    """Love-Weber stress of the slab ``z in [z_lo, z_hi]`` (tension positive).

    The wall traction the device reports is ``tau_wall / p_wall`` on the top
    platen; a slab-rheometry reading needs the *bulk* stress, which is not the
    same quantity when the boundary layer carries a different stress than the
    interior.  This estimator is the standard discrete Love-Weber sum

        sigma_ij = (1/V) [ sum_{c in slab} f_i^c r_j^c + sum_{p in slab} m v'_i v'_j ]

    with ``f^c`` the force exerted on the *first* endpoint of the pair and
    ``r^c = x_j - x_i`` the branch vector taken from the ghost-triplicated
    positions (so the periodic x images are correct), ``v'`` the velocity
    fluctuation about the slab mean.  Compressive states give negative normal
    stresses, so the pressure is ``p = -trace(sigma)/3`` and the shear stress
    magnitude is ``|sigma_xz|``.

    ``rule`` selects which contacts enter the sum (m17126):

    * ``"both"`` (legacy, rounds 6-7): only contacts whose *both* endpoints lie
      in the slab.  On a 2-4 diameter slab this discards 40-50 % of the
      load-bearing chains (the ``straddle_fraction`` diagnostic), so the slab
      stress reads systematically low.
    * ``"midpoint"``: a contact enters when the *midpoint* of the branch vector
      lies in the slab.  This is the standard slab-averaging rule; it counts
      each crossing chain once and is unbiased for a homogeneous slab.

    Wall contacts are not part of the pair list, so the estimator is a pure
    particle-particle virial (see :func:`full_cell_virial_stress` for the
    momentum-balance-normalised domain average).
    """
    import numpy as _np

    n = int(n)
    sel = ((pos[:, 2] >= z_lo) & (pos[:, 2] <= z_hi)
           & (pos[:, 1] >= y_lo) & (pos[:, 1] <= y_hi))
    n_sel = int(_np.count_nonzero(sel))
    sigma = _np.zeros((3, 3))
    volume = box_width * (y_hi - y_lo) * (z_hi - z_lo)
    n_contacts = 0
    if n_sel < 2 or volume <= 0.0:
        return {"sigma": sigma, "pressure": float("nan"),
                "shear_xz": float("nan"), "mu": float("nan"),
                "volume": float(volume), "n_particles": n_sel,
                "n_contacts": 0, "kinetic_pressure": float("nan"),
                "contact_pressure": float("nan"), "sigma_zz": float("nan"),
                "sigma_yy": float("nan"), "sigma_xx": float("nan")}
    in_i = sel[_np.mod(pairs["iu"], n)]
    in_j = sel[_np.mod(pairs["ju"], n)]
    endpoint_in = in_i | in_j
    both_in = in_i & in_j
    if rule == "midpoint":
        mid = pos[_np.mod(pairs["iu"], n)] + 0.5 * pairs["branch"]
        mid[:, 0] = _np.mod(mid[:, 0], box_width)      # undo the periodic image
        mid[:, 1] += 0.0
        inside = ((mid[:, 2] >= z_lo) & (mid[:, 2] <= z_hi)
                  & (mid[:, 1] >= y_lo) & (mid[:, 1] <= y_hi))
    elif rule == "both":
        inside = both_in
    else:
        raise ValueError(f"unknown virial rule {rule!r}")
    # contacts with exactly one endpoint in the slab: the both-endpoints rule
    # drops them, and for a thin slab (a few particle diameters) they are a
    # large fraction of the load-bearing chains, so the estimate reads low.
    # Reported as a diagnostic; it does not enter the sum (m17026).
    n_straddle = int(_np.count_nonzero(endpoint_in) - _np.count_nonzero(both_in))
    if _np.any(inside):
        rvec = pairs["branch"][inside]
        fvec = pairs["force_on_i"][inside]
        sigma += _np.einsum("ni,nj->ij", fvec, rvec) / volume
        n_contacts = int(_np.count_nonzero(inside))
    contact_sigma = sigma.copy()
    v = vel[sel] - vel[sel].mean(axis=0, keepdims=True)
    sigma -= mass * (v.T @ v) / volume   # kinetic stress: -rho <v'v'>
    kinetic_sigma = sigma - contact_sigma
    pressure = -float(_np.trace(sigma)) / 3.0
    kinetic_pressure = -float(_np.trace(kinetic_sigma)) / 3.0
    shear_xz = float(sigma[0, 2])
    return {"sigma": sigma, "pressure": pressure, "shear_xz": shear_xz,
            "mu": abs(shear_xz) / pressure if pressure > 0 else float("nan"),
            "volume": float(volume), "n_particles": n_sel,
            "n_contacts": n_contacts,
            "n_contacts_straddle": n_straddle,
            "straddle_fraction": (n_straddle / (n_straddle + n_contacts)
                                  if (n_straddle + n_contacts) > 0 else float("nan")),
            "kinetic_pressure": kinetic_pressure,
            "contact_pressure": pressure - kinetic_pressure,
            "sigma_zz": float(sigma[2, 2]), "sigma_yy": float(sigma[1, 1]),
            "sigma_xx": float(sigma[0, 0])}


def full_cell_virial_stress(pos, vel, pairs, *, box_width, box_depth,
                             height, radius, mass):
    """Domain-average Love-Weber stress over the *whole* cell (m17126).

    Why this estimator exists: it is the only one in this device with an
    **exact** analytic reference.  For a packing in static equilibrium, zero
    gravity, with all internal contacts counted,

        sum_pairs f_ij (x) (x_j - x_i) = -sum_i F_i^ext (x) x_i

    (this follows from per-particle force balance).  The external loads sit on
    the floor (z ~ r) and the platen (z ~ plate_z - r), so the sum of pair
    virials has a ``zz`` component of magnitude ``P A (height - 2 r)``.  Dividing
    by the *centre-accessible* volume ``V = A (height - 2 r)`` therefore gives

        sigma_zz = -P_applied      (exactly, up to the dynamic/kinetic term)

    which makes ``sigma_zz / P_applied = -1`` a machine-checkable momentum
    balance invariant, independent of any literature constant.  Rounds 6-7 used
    only the thin-slab estimator, whose truncation (40-50 % of the chains
    dropped) could not be distinguished from a genuine stress deficit; this
    function supplies the check.

    The kinetic part uses the fluctuation about the measured affine shear
    profile ``v_x = a + b z`` (a bulk sheared cell has no other mean field in
    x/y/z beyond the platen drift).
    """
    import numpy as _np

    n = int(pos.shape[0])
    volume = box_width * box_depth * max(height - 2.0 * radius, 1e-12)
    sigma = _np.zeros((3, 3))
    n_contacts = 0
    if n < 2 or pairs.get("branch") is None or len(pairs["branch"]) == 0:
        return {"sigma": sigma, "pressure": float("nan"),
                "shear_xz": float("nan"), "mu": float("nan"),
                "volume": float(volume), "n_contacts": 0,
                "kinetic_pressure": float("nan"), "contact_pressure": float("nan"),
                "sigma_zz": float("nan"), "sigma_yy": float("nan"),
                "sigma_xx": float("nan"), "shear_rate": float("nan"),
                "profile_r2": float("nan")}
    # every pair of the periodic pair list is one physical contact of the cell
    sigma += _np.einsum("ni,nj->ij", pairs["force_on_i"], pairs["branch"]) / volume
    n_contacts = int(len(pairs["branch"]))
    contact_sigma = sigma.copy()

    z = pos[:, 2]
    zc = z - z.mean()
    vx = vel[:, 0]
    denom = float(zc @ zc)
    slope = float(zc @ (vx - vx.mean()) / denom) if denom > 1e-30 else 0.0
    intercept = float(vx.mean() - slope * z.mean())
    fluc = vel.copy()
    fluc[:, 0] -= slope * z + intercept
    fluc -= fluc.mean(axis=0, keepdims=True)
    sigma -= mass * (fluc.T @ fluc) / volume
    kinetic_sigma = sigma - contact_sigma
    pressure = -float(_np.trace(sigma)) / 3.0
    shear_xz = float(sigma[0, 2])
    return {"sigma": sigma, "pressure": pressure, "shear_xz": shear_xz,
            "mu": abs(shear_xz) / pressure if pressure > 0 else float("nan"),
            "volume": float(volume), "n_contacts": n_contacts,
            "kinetic_pressure": -float(_np.trace(kinetic_sigma)) / 3.0,
            "contact_pressure": pressure + float(_np.trace(kinetic_sigma)) / 3.0,
            "sigma_zz": float(sigma[2, 2]), "sigma_yy": float(sigma[1, 1]),
            "sigma_xx": float(sigma[0, 0]),
            "shear_rate": slope, "profile_r2": float("nan")}


def inertial_number(shear_rate, particle_diameter, pressure, density):
    """``I = gdot d / sqrt(P / rho_p)`` — the *definition*, in one place.

    ``shear_rate`` is whatever rate the caller means (imposed platen rate or the
    volume-averaged measured rate), ``pressure`` the normal stress that pairs
    with it and ``density`` the *grain* density (not the bulk density: the law's
    ``rho_p`` is the material density of the particles).  Returns ``nan`` when
    the pressure is not positive, so a lost pressure can never masquerade as a
    small inertial number.
    """
    if not pressure > 0.0:
        return float("nan")
    return abs(shear_rate) * particle_diameter / math.sqrt(pressure / density)


def shear_rate_profile(pos, vel, *, z_lo, z_hi, y_lo, y_hi, n_bins=6):
    """Least-squares slope ``d<v_x>/dz`` of the slab velocity profile.

    The imposed ``gamma_dot`` is the *platen* rate; localisation or wall slip
    makes the strain rate actually seen by the bulk different, and ``I`` must
    use the measured rate if the mu(I) abscissa is to mean anything.
    """
    import numpy as _np

    sel = ((pos[:, 2] >= z_lo) & (pos[:, 2] <= z_hi)
           & (pos[:, 1] >= y_lo) & (pos[:, 1] <= y_hi))
    n_sel = int(_np.count_nonzero(sel))
    if n_sel < 4 * n_bins:
        return {"shear_rate": float("nan"), "n_bins": 0, "r2": float("nan"),
                "profile": [], "slip_top": float("nan")}
    z = pos[sel, 2]
    vx = vel[sel, 0]
    edges = _np.linspace(z_lo, z_hi, n_bins + 1)
    centers, means, counts = [], [], []
    for k in range(n_bins):
        m = (z >= edges[k]) & (z <= edges[k + 1] if k == n_bins - 1
                               else z < edges[k + 1])
        c = int(_np.count_nonzero(m))
        if c == 0:
            continue
        centers.append(0.5 * (edges[k] + edges[k + 1]))
        means.append(float(vx[m].mean()))
        counts.append(c)
    if len(centers) < 2:
        return {"shear_rate": float("nan"), "n_bins": len(centers),
                "r2": float("nan"), "profile": [], "slip_top": float("nan")}
    centers_a = _np.asarray(centers)
    means_a = _np.asarray(means)
    w = _np.asarray(counts, dtype=float)
    slope, intercept = _np.polyfit(centers_a, means_a, 1, w=_np.sqrt(w))
    pred = slope * centers_a + intercept
    ss_res = float(_np.sum(w * (means_a - pred) ** 2))
    ss_tot = float(_np.sum(w * (means_a - _np.average(means_a, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"shear_rate": float(slope), "n_bins": len(centers), "r2": r2,
            "profile": [[float(c), float(m), int(k)] for c, m, k
                        in zip(centers, means, counts)],
            "intercept": float(intercept)}


def simulate_servo_shear(case: ServoShearCase, *, return_series: bool = False) -> dict:
    """Constant-stress direct shear of a confined sphere packing.

    Protocol: HCP fill -> constant-stress servo phase (no shear) -> shear phase
    at ``gamma_dot`` (from ``inertial_number`` if given, else ``shear_rate``)
    to a target strain, sampling stress and microstructure after a 40 % warm-up.

    Solid-fraction bookkeeping (important, see m16427 / plane-shear round 2):
    ``plate_height`` is the **settle-phase** platen height, frozen when the
    shear phase starts, so a bed solid fraction built from it
    (``phi_bed_settled``) is identical for every shear rate by construction and
    must not be read as an ``phi(I)`` curve.  The steady-state counterparts
    ``plate_height_steady`` / ``phi_bed_steady`` are tail averages over the
    shear-phase samples (``plate_z`` is servo-updated during shear), and
    ``dilatancy`` is their relative height change against the settled bed.

    Particle contacts use :func:`tensordem.granular3d_core.contact_forces`
    (Cundall-Strack tangential spring, velocity-independent -- no regularised
    Coulomb, so the low-I limit is not smeared by a slip-velocity cutoff).
    The periodic x direction is handled with ghost images whose forces are
    folded back onto the base cell.
    """
    import time as _time

    from .granular3d_core import ContactState, Material3D, contact_forces

    r = case.radius
    dt = case.timestep
    kn = case.kn
    kt = case.kt
    gamma_w = case.gamma_wall
    mass, inertia = case.mass, case.inertia
    mu_w = case.wall_mu
    W, D = case.box_width, case.box_depth
    area = case.area
    rho = case.density
    d = 2.0 * r
    volume_particle = 4.0 / 3.0 * math.pi * r ** 3
    g = case.gravity

    mat = Material3D(radius=r, density=rho, mu=case.mu, mu_rolling=case.mu_rolling,
                     restitution=case.restitution, kn=kn, kt_ratio=case.kt_ratio,
                     gravity=g)

    pos = hcp_positions_in_box(r, W, D, case.fill_height,
                               packing_gap=case.packing_gap,
                               jitter=case.jitter, seed=case.seed,
                               margin=0.2 * r)
    n = pos.shape[0]
    vel = np.zeros((n, 3))
    omega = np.zeros((n, 3))
    state = ContactState()
    xi_floor = np.zeros((n, 2))
    xi_top = np.zeros((n, 2))

    target_force = case.normal_stress * area
    plate_z = float(pos[:, 2].max() + r)
    plate_vx = 0.0
    plate_vz = 0.0
    series: list[dict] = []
    proxy_series: list[dict] = []
    wall_clock = _time.perf_counter()

    ghost_shift = np.array([[0.0, 0.0, 0.0],
                            [W, 0.0, 0.0],
                            [-W, 0.0, 0.0]])
    base_key = np.arange(n, dtype=np.int64)

    def _step_forces(plate_vx_local: float, plate_vz_local: float):
        nonlocal xi_floor, xi_top
        pos[:, 0] = np.mod(pos[:, 0], W)
        flat_pos = pos[None, :, :] + ghost_shift[:, None, :]
        flat_pos = flat_pos.reshape(-1, 3)
        flat_vel = np.tile(vel, (3, 1))
        flat_omega = np.tile(omega, (3, 1))
        iu, ju = _periodic_pair_list(flat_pos, n, r)
        state_keys = (iu % n) * n + (ju % n)
        f_flat, t_flat, pairs = contact_forces(
            flat_pos, flat_vel, flat_omega, iu, ju, state, mat, dt,
            state_keys=state_keys, return_pairs=True)
        force = f_flat[:n] + f_flat[n:2 * n] + f_flat[2 * n:]
        torque = t_flat[:n] + t_flat[n:2 * n] + t_flat[2 * n:]
        n_contacts = len(iu)

        # side walls (y): smooth, normal + dashpot only
        pen_lo = r - pos[:, 1]
        pen_hi = pos[:, 1] + r - D
        for pen, sign in ((pen_lo, 1.0), (pen_hi, -1.0)):
            hit = pen > 0.0
            if hit.any():
                v_n = vel[hit, 1] * sign
                f_n = np.maximum(kn * pen[hit] - gamma_w * v_n, 0.0)
                force[hit, 1] += sign * f_n

        # floor (z = 0): normal + tangential spring
        pen_b = r - pos[:, 2]
        hit = pen_b > 0.0
        if hit.any():
            idx = np.nonzero(hit)[0]
            v_n = vel[idx, 2]
            f_n = np.maximum(kn * pen_b[idx] - gamma_w * v_n, 0.0)
            force[idx, 2] += f_n
            if mu_w > 0.0:
                f_t = _wall_spring(xi_floor, vel[:, :2].copy(), dt, kt, f_n, mu_w, idx)
                force[idx, :2] += f_t

        # top platen: constant-stress normal + rough tangential spring
        pen_t = np.minimum(pos[:, 2] + r - plate_z, r)
        hit = pen_t > 0.0
        fn_sum = 0.0
        ftx_sum = 0.0
        if hit.any():
            idx = np.nonzero(hit)[0]
            v_n = vel[idx, 2] - plate_vz_local
            f_n = np.maximum(kn * pen_t[idx] - gamma_w * v_n, 0.0)
            force[idx, 2] -= f_n
            fn_sum = float(f_n.sum())
            if mu_w > 0.0:
                v_t = vel[:, :2].copy()
                v_t[:, 0] -= plate_vx_local
                f_t = _wall_spring(xi_top, v_t, dt, kt, f_n, mu_w, idx)
                force[idx, :2] += f_t
                ftx_sum = float(f_t[:, 0].sum())

        if g != 0.0:
            force[:, 2] -= mass * g
        contacts = pairs
        return force, torque, fn_sum, ftx_sum, n_contacts, contacts

    # ---- phase 1: constant-stress servo, no shear -------------------------
    fn_window: list[float] = []
    fn_avg = 0.0
    settle_steps = case.settle_steps
    max_settle = max(6 * case.settle_steps, 12000)
    for step in range(max_settle):
        force, torque, fn_sum, _, n_contacts, _ = _step_forces(0.0, plate_vz)
        fn_avg += (fn_sum - fn_avg) * 0.01
        if step % case.servo_stride == 0:
            n_c = max(1, int(np.count_nonzero(pos[:, 2] + r > plate_z)))
            delta = case.servo_relax * (fn_avg - target_force) / (kn * n_c)
            delta = max(min(delta, case.servo_max_step_d * d),
                        -case.servo_max_step_d * d)
            plate_z += delta
            plate_vz = delta / (case.servo_stride * dt)
        vel += force / mass * dt
        omega += torque / inertia * dt
        pos += vel * dt
        pos[:, 2] = np.maximum(pos[:, 2], 0.0)
        over = pos[:, 2] > plate_z
        if over.any():
            pos[over, 2] = plate_z
            vel[over, 2] = np.minimum(vel[over, 2], 0.0)
        if step >= 200 and abs(fn_avg - target_force) < 0.08 * target_force:
            fn_window.append(fn_sum)
            if len(fn_window) >= 400:
                settle_steps = step + 1
                break
        if os.environ.get("SERVO_DEBUG") and step % 200 == 0:
            print(f"[settle {step}] plate={plate_z:.5f} fn={fn_sum:.2f} "
                  f"target={target_force:.2f} vmax={np.abs(vel).max():.3e} "
                  f"zmax={pos[:,2].max():.5f} nc={n_contacts}")

    p_settle = float(np.mean(fn_window)) / area if fn_window else 0.0
    if not fn_window:
        raise MuIRheologyError("servo did not reach the target normal stress")
    plate_height = plate_z

    if case.inertial_number is not None:
        if p_settle <= 0.0:
            raise MuIRheologyError("settle pressure is zero; cannot set inertial number")
        # the servo holds normal_stress, so the I-control uses the applied value
        gamma_dot = case.inertial_number * math.sqrt(case.normal_stress / rho) / d
    else:
        gamma_dot = case.shear_rate
    if gamma_dot <= 0.0:
        raise MuIRheologyError("shear rate resolved to zero")

    shear_steps = int(max(case.min_shear_steps,
                          math.ceil(case.shear_strain / (gamma_dot * dt))))
    warmup = int(0.4 * shear_steps)
    strain = 0.0
    max_shear_steps = shear_steps + max(0, int(case.max_recover_steps))
    recover_streak = 0
    step = -1
    while step + 1 < max_shear_steps:
        step += 1
        if step % case.servo_stride == 0:
            plate_vx = gamma_dot * plate_z
        force, torque, fn_sum, ftx_sum, n_contacts, contacts = _step_forces(plate_vx, plate_vz)
        fn_avg += (fn_sum - fn_avg) * 0.01
        if step % case.servo_stride == 0:
            n_c = max(1, int(np.count_nonzero(pos[:, 2] + r > plate_z)))
            delta = case.servo_relax * (fn_avg - target_force) / (kn * n_c)
            delta = max(min(delta, case.servo_max_step_d * d),
                        -case.servo_max_step_d * d)
            plate_z += delta
            plate_vz = delta / (case.servo_stride * dt)
        vel += force / mass * dt
        omega += torque / inertia * dt
        pos += vel * dt
        pos[:, 2] = np.maximum(pos[:, 2], 0.0)
        over = pos[:, 2] > plate_z
        if over.any():
            pos[over, 2] = plate_z
            vel[over, 2] = np.minimum(vel[over, 2], 0.0)
        strain += gamma_dot * dt

        if step >= warmup and step % case.sample_stride == 0:
            p_now = fn_sum / area
            tau_now = abs(ftx_sum) / area
            margin = case.bulk_margin_z * plate_z
            bulk = ((pos[:, 2] > margin) & (pos[:, 2] < plate_z - margin)
                    & (pos[:, 1] > 0.2 * D) & (pos[:, 1] < 0.8 * D))
            n_bulk = int(np.count_nonzero(bulk))
            v_bulk = W * 0.6 * D * max(plate_z - 2.0 * margin, 1e-12)
            # solid fraction of the *interior* slab: n*V_p/V_slab.  Until
            # m17026 this line read ``1.0 - n*V_p/V_slab``, i.e. it returned the
            # slab POROSITY under the name ``solid_fraction`` -- a sign
            # inversion that made every reported ``phi`` ~0.37-0.42 for a dense
            # bed (and made the case look 30+ % low against the reference law).
            phi = (n_bulk * volume_particle / v_bulk) if v_bulk > 0 else float("nan")
            sample = {
                "step": step,
                "strain": strain,
                "plate_z": plate_z,
                "normal_stress_pa": p_now,
                "shear_stress_pa": tau_now,
                "mu": tau_now / p_now if p_now > 0 else float("nan"),
                "coordination": 2.0 * n_contacts / n,
                "solid_fraction": phi,
                "porosity": 1.0 - phi,
                # audit fields so the estimator can be re-derived from a sample
                "slab_n_particles": n_bulk,
                "slab_volume": v_bulk,
            }
            if case.bulk_stress:
                z_lo = max(margin, 0.0)
                z_hi = plate_z - margin
                y_lo = case.bulk_y_margin * D
                y_hi = (1.0 - case.bulk_y_margin) * D
                if z_hi > z_lo:
                    vir = bulk_virial_stress(
                        pos, vel, contacts,
                        z_lo=z_lo, z_hi=z_hi, y_lo=y_lo, y_hi=y_hi,
                        box_width=W, n=n, mass=mass)
                    prof = shear_rate_profile(
                        pos, vel, z_lo=z_lo, z_hi=z_hi, y_lo=y_lo, y_hi=y_hi,
                        n_bins=case.bulk_bins)
                    p_b = vir["pressure"]
                    sample.update({
                        "bulk_pressure_pa": p_b,
                        "bulk_shear_pa": abs(vir["shear_xz"]),
                        "bulk_mu": vir["mu"],
                        "bulk_sigma_zz": vir["sigma_zz"],
                        "bulk_kinetic_pressure_pa": vir["kinetic_pressure"],
                        "bulk_n_particles": vir["n_particles"],
                        "bulk_n_contacts": vir["n_contacts"],
                        "bulk_straddle_fraction": vir["straddle_fraction"],
                        "shear_rate_measured": prof["shear_rate"],
                        "profile_r2": prof["r2"],
                        "bulk_inertial_number": (
                            abs(prof["shear_rate"]) * d
                            / math.sqrt(p_b / rho)
                            if p_b and p_b > 0 and math.isfinite(
                                prof["shear_rate"]) else float("nan")),
                    })
                    # m17126: the R7-recommended midpoint rule and the exact
                    # momentum-balance estimator on the same sample.
                    vir_mid = bulk_virial_stress(
                        pos, vel, contacts, z_lo=z_lo, z_hi=z_hi,
                        y_lo=y_lo, y_hi=y_hi, box_width=W, n=n, mass=mass,
                        rule="midpoint")
                    full = full_cell_virial_stress(
                        pos, vel, contacts, box_width=W, box_depth=D,
                        height=plate_z, radius=r, mass=mass)
                    sample.update({
                        "bulk_mid_pressure_pa": vir_mid["pressure"],
                        "bulk_mid_shear_pa": abs(vir_mid["shear_xz"]),
                        "bulk_mid_mu": vir_mid["mu"],
                        "bulk_mid_n_contacts": vir_mid["n_contacts"],
                        "full_pressure_pa": full["pressure"],
                        "full_shear_pa": abs(full["shear_xz"]),
                        "full_mu": full["mu"],
                        "full_mu_zz": (abs(full["shear_xz"])
                                       / abs(full["sigma_zz"])
                                       if full["sigma_zz"] else float("nan")),
                        "full_sigma_zz": full["sigma_zz"],
                        "full_kinetic_pressure_pa": full["kinetic_pressure"],
                        "full_n_contacts": full["n_contacts"],
                        "full_momentum_balance": (
                            full["sigma_zz"] / (-case.normal_stress)
                            if case.normal_stress > 0 else float("nan")),
                    })
            if case.contact_proxy:
                # m17500 cross-validation: the contact gauge of this very state
                pair = contacts or {}
                branch = pair.get("branch")
                if branch is not None and len(branch) > 0:
                    from .mu_i_contact_proxy import (
                        contact_proxy_stats, slip_speed_at_contacts)
                    iu_p = np.asarray(pair["iu"]) % n
                    ju_p = np.asarray(pair["ju"]) % n
                    # ``branch`` comes from the ghost-triplicated positions, so
                    # the normal direction is aliasing-free for the periodic x
                    slip = slip_speed_at_contacts(vel, omega, iu_p, ju_p,
                                                  branch, r)
                    proxy_series.append(contact_proxy_stats(
                        np.asarray(pair["f_n"]), slip, d, rho))
            series.append(sample)

        # constant-stress validity gate: once the requested strain budget is
        # spent, keep shearing until the servo holds the applied pressure
        if step + 1 >= shear_steps and target_force > 0.0:
            err_now = abs(fn_avg / target_force - 1.0)
            if err_now <= case.pressure_control_tol:
                recover_streak += 1
            else:
                recover_streak = 0
            if recover_streak >= case.pressure_recover_window:
                break

    shear_steps_executed = step + 1
    wall_clock = _time.perf_counter() - wall_clock
    if not series:
        raise MuIRheologyError("no samples collected in the shear phase")

    keys = ["normal_stress_pa", "shear_stress_pa", "mu",
            "coordination", "solid_fraction", "porosity"]
    if case.bulk_stress and "bulk_mu" in series[0]:
        keys += ["bulk_pressure_pa", "bulk_shear_pa", "bulk_mu",
                 "bulk_kinetic_pressure_pa", "bulk_sigma_zz",
                 "shear_rate_measured", "profile_r2", "bulk_inertial_number",
                 "bulk_straddle_fraction", "bulk_mid_pressure_pa",
                 "bulk_mid_shear_pa", "bulk_mid_mu", "bulk_mid_n_contacts",
                 "full_pressure_pa", "full_shear_pa", "full_mu",
                 "full_sigma_zz", "full_kinetic_pressure_pa",
                 "full_n_contacts", "full_momentum_balance", "full_mu_zz"]
    arr = {key: np.array([s.get(key, float("nan")) for s in series], dtype=float)
           for key in keys}
    n_tail = max(1, int(0.6 * len(series)))
    mean = {key: float(np.nanmean(value[-n_tail:])) for key, value in arr.items()}
    # steady-state bed bookkeeping: the platen keeps moving during shear, so the
    # settled height alone cannot express a shear-rate dependent solid fraction
    plate_series = np.array([s["plate_z"] for s in series], dtype=float)
    plate_height_steady = float(np.nanmean(plate_series[-n_tail:]))
    phi_bed_settled = (n * volume_particle / (W * D * plate_height)
                       if plate_height > 0.0 else float("nan"))
    phi_bed_steady = (n * volume_particle / (W * D * plate_height_steady)
                      if plate_height_steady > 0.0 else float("nan"))
    dilatancy = ((plate_height_steady - plate_height) / plate_height
                 if plate_height > 0.0 else float("nan"))
    p_mean = mean["normal_stress_pa"]
    # round 10 (m17694): ``gamma_dot`` is the *imposed platen* rate, so what the
    # historic key ``inertial_number`` carries is I at the imposed rate with the
    # measured pressure -- not a measured I (rounds 6-8 read it as one).  The
    # value is unchanged; the explicit taxonomy is added below.
    i_imposed_rate = inertial_number(gamma_dot, d, p_mean, rho)
    # final-window (last 20 % of samples) counterparts: with a drifting servo the
    # 60 % tail average mixes the transient, so the validity verdict is taken on
    # the final window (this is the window the recovery gate settles on).
    n_final = max(1, int(0.2 * len(series)))
    p_final = float(np.nanmean(arr["normal_stress_pa"][-n_final:]))
    mu_final = float(np.nanmean(arr["mu"][-n_final:]))
    plate_height_final = float(np.nanmean(plate_series[-n_final:]))
    phi_bed_final = (n * volume_particle / (W * D * plate_height_final)
                     if plate_height_final > 0.0 else float("nan"))

    result = {
        "device": "servo_shear_cell",
        "bulk_stress_enabled": bool(case.bulk_stress),
        "n_particles": int(n),
        "radius": r,
        "particle_diameter": d,
        "density": rho,
        "kn": kn,
        "mu_contact": case.mu,
        "mu_rolling": case.mu_rolling,
        "restitution": case.restitution,
        "normal_stress_applied_pa": case.normal_stress,
        "normal_stress_measured_pa": p_mean,
        "shear_stress_pa": mean["shear_stress_pa"],
        "mu_effective": mean["mu"],
        "shear_rate_imposed": gamma_dot,
        "inertial_number": i_imposed_rate,
        "inertial_number_target": case.inertial_number,
        "coordination_number": mean["coordination"],
        "solid_fraction": mean["solid_fraction"],
        "porosity": mean["porosity"],
        "plate_height": plate_height,
        "plate_height_settled": plate_height,
        "plate_height_steady": plate_height_steady,
        "phi_bed_settled": phi_bed_settled,
        "phi_bed_steady": phi_bed_steady,
        "dilatancy": dilatancy,
        "dt": dt,
        "settle_steps": settle_steps,
        "shear_steps": shear_steps,
        "shear_strain_achieved": strain,
        "n_samples": len(series),
        "shear_steps_executed": shear_steps_executed,
        "recover_steps": shear_steps_executed - shear_steps,
        "pressure_control_error": (p_mean / case.normal_stress - 1.0
                                   if case.normal_stress > 0 else float('nan')),
        "pressure_control_error_final": (
            p_final / case.normal_stress - 1.0
            if case.normal_stress > 0 else float('nan')),
        "mu_final": mu_final,
        "phi_bed_final": phi_bed_final,
        "servo_valid": bool(case.normal_stress > 0 and
                            abs(p_final / case.normal_stress - 1.0)
                            <= case.pressure_control_tol),
        "wall_clock_s": wall_clock,
        "bulk_mu": (mean["bulk_mu"] if "bulk_mu" in mean else None),
        "bulk_mu_final": (float(np.nanmean(arr["bulk_mu"][-n_final:]))
                          if "bulk_mu" in arr else None),
        "bulk_pressure_pa": (mean["bulk_pressure_pa"]
                            if "bulk_pressure_pa" in mean else None),
        "bulk_shear_pa": (mean["bulk_shear_pa"]
                          if "bulk_shear_pa" in mean else None),
        "bulk_pressure_final_pa": (
            float(np.nanmean(arr["bulk_pressure_pa"][-n_final:]))
            if "bulk_pressure_pa" in arr else None),
        "bulk_kinetic_pressure_pa": (
            mean["bulk_kinetic_pressure_pa"]
            if "bulk_kinetic_pressure_pa" in mean else None),
        "bulk_sigma_zz_final": (float(np.nanmean(arr["bulk_sigma_zz"][-n_final:]))
                                if "bulk_sigma_zz" in arr else None),
        "bulk_straddle_fraction": (mean["bulk_straddle_fraction"]
                                   if "bulk_straddle_fraction" in mean else None),
        "shear_rate_measured": (mean["shear_rate_measured"]
                                if "shear_rate_measured" in mean else None),
        "profile_r2": (mean["profile_r2"] if "profile_r2" in mean else None),
        "inertial_number_bulk": (mean["bulk_inertial_number"]
                                 if "bulk_inertial_number" in mean else None),
        "inertial_number_bulk_final": (
            float(np.nanmean(arr["bulk_inertial_number"][-n_final:]))
            if "bulk_inertial_number" in arr else None),
        "bulk_mid_mu": (mean["bulk_mid_mu"] if "bulk_mid_mu" in mean else None),
        "bulk_mid_mu_final": (float(np.nanmean(arr["bulk_mid_mu"][-n_final:]))
                              if "bulk_mid_mu" in arr else None),
        "bulk_mid_pressure_pa": (mean["bulk_mid_pressure_pa"]
                                 if "bulk_mid_pressure_pa" in mean else None),
        "full_mu": (mean["full_mu"] if "full_mu" in mean else None),
        "full_mu_final": (float(np.nanmean(arr["full_mu"][-n_final:]))
                          if "full_mu" in arr else None),
        "full_pressure_pa": (mean["full_pressure_pa"]
                             if "full_pressure_pa" in mean else None),
        "full_shear_pa": (mean["full_shear_pa"]
                          if "full_shear_pa" in mean else None),
        "full_mu_zz": (mean["full_mu_zz"] if "full_mu_zz" in mean else None),
        "full_mu_zz_final": (float(np.nanmean(arr["full_mu_zz"][-n_final:]))
                             if "full_mu_zz" in arr else None),
        "full_anisotropy": ((mean["full_pressure_pa"]
                             / abs(np.nanmean(arr["full_sigma_zz"][-n_tail:])))
                            if "full_pressure_pa" in mean else None),
        "full_sigma_zz_final": (float(np.nanmean(arr["full_sigma_zz"][-n_final:]))
                                if "full_sigma_zz" in arr else None),
        "full_kinetic_pressure_pa": (mean["full_kinetic_pressure_pa"]
                                     if "full_kinetic_pressure_pa" in mean else None),
        "full_n_contacts": (mean["full_n_contacts"]
                            if "full_n_contacts" in mean else None),
        "full_momentum_balance": (mean["full_momentum_balance"]
                                  if "full_momentum_balance" in mean else None),
        "full_momentum_balance_final": (
            float(np.nanmean(arr["full_momentum_balance"][-n_final:]))
            if "full_momentum_balance" in arr else None),
        "box_volume": W * D * plate_height,
        "settle_pressure_pa": p_settle,
        "max_velocity": float(np.abs(vel).max()),
    }
    # stiffness-number diagnostic (Campbell 2002 elastic limit): a mu(I)
    # (rate-independent) reading is only licensed in the quasi-static band.
    regime = case.elastic_limit_regime
    if regime is not None:
        result["kappa"] = regime["kappa"]
        result["elastic_limit"] = regime
    # --- round 10 (m17694): explicit inertial-number taxonomy -----------------
    # The mu(I) ordinate is a *volume* average (full_cell_virial_stress), so the
    # abscissa must be the volume-averaged measured rate, not the platen rate.
    # Nothing above changes: this block only names the two other abscissae.
    _rate = result.get("shear_rate_measured")
    result["inertial_number_imposed_rate"] = i_imposed_rate
    result["inertial_number_measured_rate"] = (
        inertial_number(_rate, d, p_mean, rho) if _rate is not None else None)
    if "shear_rate_measured" in arr:
        _rate_final = float(np.nanmean(arr["shear_rate_measured"][-n_final:]))
        result["inertial_number_measured_rate_final"] = inertial_number(
            _rate_final, d, p_final, rho)
        result["shear_rate_measured_final"] = _rate_final
    else:
        result["inertial_number_measured_rate_final"] = None
        result["shear_rate_measured_final"] = None

    if case.bulk_stress and result.get("bulk_mu") is not None:
        mu_b, mu_w = result["bulk_mu"], result["mu_effective"]
        p_b = result["bulk_pressure_pa"]
        result["bulk_vs_wall_mu_ratio"] = (mu_b / mu_w
                                           if mu_w else float("nan"))
        result["bulk_pressure_balance_error"] = (
            p_b / case.normal_stress - 1.0 if case.normal_stress else float("nan"))
        result["bulk_pressure_vs_wall"] = (p_b / result["normal_stress_measured_pa"]
                                           if result["normal_stress_measured_pa"]
                                           else float("nan"))
        if result.get("full_mu") is not None:
            result["full_cell_volume_proxy"] = W * D * (plate_height - 2.0 * r)
            result["mu_full_over_wall"] = (
                result["full_mu"] / mu_w if mu_w else float("nan"))
        result["shear_rate_ratio"] = (
            result["shear_rate_measured"] / gamma_dot
            if gamma_dot else float("nan"))
    if case.contact_proxy:
        from .mu_i_contact_proxy import aggregate_proxy
        result["contact_proxy"] = aggregate_proxy(proxy_series)
        result["contact_proxy_series"] = proxy_series
    if return_series:
        result["series"] = series
    return result


__all__ += ["ServoShearCase", "simulate_servo_shear", "inertial_number"]


@dataclass(frozen=True)
class ReposeCollapseCase:
    """Gravity-driven column collapse (repose-angle anchor), SI units."""

    radius: float = 0.005
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.05
    restitution: float = 0.3
    kn: float = 6.25e4
    kt_ratio: float = 0.5
    gravity: float = 9.81
    cylinder_radius: float = 0.05
    fill_height: float = 0.06
    floor_radius: float = 0.18
    wall_mu: float = 0.5
    hold_time: float = 0.10
    settle_time: float = 0.05
    collapse_time: float = 0.55
    packing_gap: float = 0.02
    jitter: float = 0.01
    seed: int = 4970
    dt: Optional[float] = None
    sample_stride: int = 20

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
    def zeta(self) -> float:
        return _damping_ratio(self.restitution)

    @property
    def gamma_wall(self) -> float:
        return 2.0 * self.zeta * math.sqrt(self.kn * self.mass)

    @property
    def timestep(self) -> float:
        if self.dt is not None:
            return self.dt
        return 2.0 * math.pi * math.sqrt(0.5 * self.mass / self.kn) / 20.0

    def __post_init__(self) -> None:
        for name in ("radius", "density", "kn", "cylinder_radius", "fill_height",
                     "floor_radius", "gravity"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value)
                    and value > 0.0):
                raise MuIRheologyError(f"{name} must be finite and positive")
        if self.floor_radius <= self.cylinder_radius:
            raise MuIRheologyError("floor_radius must exceed the cylinder radius")


def _repose_angle_from_positions(pos: np.ndarray, radius: float,
                                 floor_radius: float) -> tuple[float, float, int]:
    """Cone fit of the heap free surface; returns (deg, stderr_deg, n_cells).

    The surface is the per-column height ``h(x,y)`` (a grid of one-particle
    cells), fitted as ``h = h0 - tan(theta) r`` over the radial band
    ``[0.2, 0.9] floor_radius``.  Using every occupied column (instead of a few
    radial bins) keeps the slope error at the level of the surface roughness
    divided by sqrt(n_cells).
    """
    cell = 2.0 * radius
    # integer grid keys; the +offset keeps negative coordinates unambiguous
    io = 1024
    ix = np.floor(pos[:, 0] / cell).astype(np.int64) + io
    iy = np.floor(pos[:, 1] / cell).astype(np.int64) + io
    keys = ix * 4096 + iy
    z = pos[:, 2]
    order = np.argsort(keys, kind="stable")
    keys_s, z_s = keys[order], z[order]
    uniq, starts = np.unique(keys_s, return_index=True)
    top = np.maximum.reduceat(z_s, starts)
    gx = (uniq // 4096 - io).astype(float) * cell + 0.5 * cell
    gy = (uniq % 4096 - io).astype(float) * cell + 0.5 * cell
    rad = np.hypot(gx, gy)
    # heap toe: outer radius where the surface still stands more than one cell
    # above the floor; the cone fit is restricted to the slope between the crest
    # and the toe (the flat apron beyond the toe has no slope information).
    standing = top > 1.2 * cell
    if not standing.any():
        return float("nan"), float("nan"), 0
    toe = float(rad[standing].max())
    sel = (rad > 0.15 * toe) & (rad < 0.85 * toe)
    n_cells = int(np.count_nonzero(sel))
    if n_cells < 8:
        return float("nan"), float("nan"), n_cells
    r_sel, h_sel = rad[sel], top[sel]
    # robust cone fit: two IRLS passes with Huber weights
    A = np.vstack([r_sel, np.ones_like(r_sel)]).T
    w = np.ones_like(r_sel)
    for _ in range(2):
        Aw = A * w[:, None]
        hw = h_sel * w
        slope, intercept = np.linalg.lstsq(Aw, hw, rcond=None)[0]
        res = h_sel - (slope * r_sel + intercept)
        scale = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-12
        w = 1.0 / np.sqrt(1.0 + (res / (1.5 * scale)) ** 2)
    dof = max(1, n_cells - 2)
    sigma = math.sqrt(float(np.sum(w ** 2 * res ** 2)) / dof)
    sxx = float(np.sum(w ** 2 * (r_sel - np.average(r_sel, weights=w)) ** 2))
    stderr_slope = sigma / math.sqrt(max(sxx, 1e-30))
    theta = math.degrees(math.atan(abs(slope)))
    stderr = math.degrees(stderr_slope / (1.0 + slope ** 2))
    return theta, stderr, n_cells


def simulate_repose_collapse(case: ReposeCollapseCase, *, seeds: tuple[int, ...] = (0, 1),
                             return_state: bool = False) -> dict:
    """Column collapse of a confined cylinder -> angle of repose.

    The hollow cylinder wall is removed at ``hold_time``; the material spreads
    into a heap whose free-surface slope is the repose angle.  Gravity is kept
    fixed across coarse-graining factors while lengths scale as k, so the
    collapse time scales as sqrt(k) (Froude similarity) -- the campaign passes
    ``collapse_time * sqrt(k)``.
    """
    import time as _time

    from .granular3d_core import (ContactState, Material3D, build_pairs,
                                  contact_forces, hcp_positions_in_cylinder)

    angles, stderrs, seed_reports = [], [], []
    wall_clock = _time.perf_counter()
    for seed in seeds:
        r = case.radius
        dt = case.timestep
        kn = case.kn
        kt = case.kt
        g = case.gravity
        mass, inertia = case.mass, case.inertia
        mat = Material3D(radius=r, density=case.density, mu=case.mu,
                         mu_rolling=case.mu_rolling, restitution=case.restitution,
                         kn=kn, kt_ratio=case.kt_ratio, gravity=g)
        pos = hcp_positions_in_cylinder(r, case.cylinder_radius, case.fill_height,
                                        packing_gap=case.packing_gap,
                                        jitter=case.jitter, seed=case.seed + seed)
        n = pos.shape[0]
        vel = np.zeros((n, 3))
        omega = np.zeros((n, 3))
        state = ContactState()
        xi_floor = np.zeros((n, 2))
        total_time = case.settle_time + case.hold_time + case.collapse_time
        steps = int(math.ceil(total_time / dt))

        for step in range(steps):
            t_now = step * dt
            force = np.zeros((n, 3))
            torque = np.zeros((n, 3))
            iu, ju = build_pairs(pos, r)
            f_pair, t_pair = contact_forces(pos, vel, omega, iu, ju, state, mat, dt)
            force += f_pair
            torque += t_pair

            # floor
            pen_b = r - pos[:, 2]
            hit = pen_b > 0.0
            if hit.any():
                idx = np.nonzero(hit)[0]
                v_n = vel[idx, 2]
                f_n = np.maximum(kn * pen_b[idx] - case.gamma_wall * v_n, 0.0)
                force[idx, 2] += f_n
                if case.wall_mu > 0.0:
                    f_t = _wall_spring(xi_floor, vel[:, :2].copy(), dt, kt, f_n,
                                       case.wall_mu, idx)
                    force[idx, :2] += f_t

            # confining cylinder, removed after hold_time
            if t_now < case.settle_time + case.hold_time:
                rad = np.hypot(pos[:, 0], pos[:, 1])
                pen = rad + r - case.cylinder_radius
                hit = pen > 0.0
                if hit.any():
                    idx = np.nonzero(hit)[0]
                    nvec = pos[idx, :2] / np.maximum(rad[idx], 1e-30)[:, None]
                    v_n = (vel[idx, :2] * nvec).sum(axis=1)
                    f_n = np.maximum(kn * pen[idx] - case.gamma_wall * v_n, 0.0)
                    force[idx, :2] -= nvec * f_n[:, None]

            force[:, 2] -= mass * g
            vel += force / mass * dt
            omega += torque / inertia * dt
            pos += vel * dt
            pos[:, 2] = np.maximum(pos[:, 2], 0.0)
            rad = np.hypot(pos[:, 0], pos[:, 1])
            outside = rad > case.floor_radius
            if outside.any():
                pos[outside, :2] *= (case.floor_radius / rad[outside])[:, None]
                vel[outside, :2] *= 0.0

        theta, stderr, n_bins = _repose_angle_from_positions(pos, r, case.floor_radius)
        angles.append(theta)
        stderrs.append(stderr)
        seed_reports.append({"seed": case.seed + seed, "theta_deg": theta,
                             "stderr_deg": stderr, "n_bins": n_bins,
                             "n_particles": int(n), "steps": steps,
                             "dt": dt, "max_velocity": float(np.abs(vel).max())})

    wall_clock = _time.perf_counter() - wall_clock
    result = {
        "device": "repose_column_collapse",
        "repose_angle_deg": float(np.mean(angles)),
        "repose_angle_stderr_deg": float(np.std(angles, ddof=1) / math.sqrt(len(angles)))
        if len(angles) > 1 else float(stderrs[0]),
        "repose_angle_seeds": angles,
        "n_particles": seed_reports[0]["n_particles"],
        "radius": case.radius,
        "kn": case.kn,
        "cylinder_radius": case.cylinder_radius,
        "fill_height": case.fill_height,
        "dt": seed_reports[0]["dt"],
        "steps": seed_reports[0]["steps"],
        "wall_clock_s": wall_clock,
        "seeds": seed_reports,
    }
    if return_state:
        result["final_positions"] = pos.tolist()
        result["final_velocities"] = vel.tolist()
    return result


__all__ += ["ReposeCollapseCase", "simulate_repose_collapse"]
