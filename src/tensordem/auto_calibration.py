"""DEM parameter auto-calibration framework (TensorDEM p284w3-n05).

Implements a complete inverse-calibration pipeline for DEM contact parameters
(friction coefficient mu, rolling-resistance mu_r, restitution e, stiffness ratio
kt/kn) using three macroscopic benchmarks:

1. **Angle of repose** (column collapse): measures heap slope vs mu, mu_r
2. **Direct shear box**: measures internal friction angle phi and cohesion c
3. **Uniaxial compression**: measures elastic modulus E and Poisson ratio nu

The framework combines:
- Forward solver (DEM simulation of each benchmark)
- Bayesian optimisation (Gaussian-process surrogate + EI/UCB acquisition)
- Uncertainty quantification (MCMC posterior sampling for parameter CIs)

All quantities are SI (m, kg, s, rad; angles in degrees).

References
----------
- Potyondy & Cundall (2004), "A bonded-particle model for rock"
- Jones et al. (2011), "Bayesian calibration of DEM parameters"
- Kennedy & O'Hagan (2001), "Bayesian calibration of computer models"
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from scipy.special import erf as _scipy_erf

import numpy as np

from .granular3d_core import (
    Material3D,
    ContactState,
    build_pairs,
    contact_forces,
    hcp_positions_in_cylinder,
    hcp_positions_in_box,
    damping_ratio,
)


# ============================================================================
# 1. Forward problem: DEM benchmarks
# ============================================================================

@dataclass(frozen=True)
class ReposeBenchmark:
    """Angle-of-repose column-collapse benchmark configuration."""
    cylinder_radius: float = 0.05
    fill_height: float = 0.08
    packing_gap: float = 0.02
    jitter: float = 0.02
    hold_time: float = 0.05
    lift_time: float = 0.15
    settle_time: float = 0.30
    seed: int = 4970


@dataclass(frozen=True)
class ShearBoxBenchmark:
    """Direct shear box benchmark configuration."""
    box_width: float = 0.04
    box_depth: float = 0.04
    fill_height: float = 0.03
    packing_gap: float = 0.02
    jitter: float = 0.01
    normal_stress_Pa: float = 5000.0  # applied normal stress
    shear_displacement: float = 0.008  # total shear displacement
    seed: int = 7701


@dataclass(frozen=True)
class CompressionBenchmark:
    """Uniaxial compression benchmark configuration."""
    specimen_radius: float = 0.025
    specimen_height: float = 0.05
    packing_gap: float = 0.02
    jitter: float = 0.01
    strain_rate: float = 0.01  # engineering strain rate (1/s)
    max_strain: float = 0.005
    seed: int = 8801


def _simulate_repose(mat: Material3D, bench: ReposeBenchmark | None = None,
                     *, max_steps: int | None = None) -> dict:
    """Simulate column collapse and measure repose angle.

    Uses a simplified quasi-static relaxation: particles settle under gravity
    with strong damping, then the repose angle is measured from the final heap
    profile via linear regression on the flank.

    Returns dict with repose_angle_deg, n_particles, ke_final, diagnostics.
    """
    bench = bench or ReposeBenchmark()
    r = mat.radius
    pos = hcp_positions_in_cylinder(
        r, bench.cylinder_radius, bench.fill_height,
        packing_gap=bench.packing_gap, jitter=bench.jitter, seed=bench.seed,
    )
    n = len(pos)
    vel = np.zeros((n, 3), dtype=np.float64)
    omega = np.zeros((n, 3), dtype=np.float64)

    dt = mat.timestep
    state = ContactState()
    g = mat.gravity

    # Strong damping for quasi-static relaxation
    damping_factor = 0.95  # velocity damping per step

    # Phase 1: pre-settle with wall
    n_pre = int(bench.hold_time / dt) + 1
    max_p = min(n_pre, 200) if max_steps is None else min(n_pre, max_steps)
    for _ in range(max_p):
        vel[:, 2] -= g * dt
        iu, ju = build_pairs(pos, r)
        f, t = contact_forces(pos, vel, omega, iu, ju, state, mat, dt)
        vel += f / mat.mass * dt
        omega += t / mat.inertia * dt
        vel *= damping_factor
        omega *= damping_factor
        pos += vel * dt

    # Phase 2: wall removal - particles flow out to form heap
    n_settle = int(bench.settle_time / dt) + 1
    max_s = min(n_settle, 500) if max_steps is None else min(n_settle, max_steps)
    wall_radius = bench.cylinder_radius  # wall starts at cylinder radius
    wall_lift_rate = bench.cylinder_radius / max(1, int(bench.lift_time / dt))  # wall lifts up

    for step in range(max_s):
        vel[:, 2] -= g * dt
        iu, ju = build_pairs(pos, r)
        f, t = contact_forces(pos, vel, omega, iu, ju, state, mat, dt)

        # Wall constraint: wall lifts during lift_time, then fully removed
        if step * dt < bench.lift_time:
            # Wall still present - constrain particles
            radial = np.linalg.norm(pos[:, :2], axis=1)
            outside = radial > bench.cylinder_radius - r
            if outside.any():
                dirs = pos[outside, :2] / radial[outside, None]
                pos[outside, :2] = dirs * (bench.cylinder_radius - r - 0.001)
                vel[outside, :2] *= 0.0
        # After lift_time, wall is removed - particles can flow freely

        # Floor constraint
        below = pos[:, 2] < r
        if below.any():
            pos[below, 2] = r
            vel[below, 2] = np.maximum(vel[below, 2], 0.0)
            vel[below, :2] *= 0.8

        vel += f / mat.mass * dt
        omega += t / mat.inertia * dt
        vel *= damping_factor
        omega *= damping_factor
        pos += vel * dt

    # Measure repose angle from heap profile
    ke = 0.5 * mat.mass * np.sum(vel ** 2)
    # Use particles above mid-height to fit flank angle
    z_mid = bench.fill_height * 0.3
    above = pos[:, 2] > z_mid
    if above.sum() < 5:
        # fallback: use all particles
        above = np.ones(n, dtype=bool)

    # Radial distance vs height for flank particles
    radial_pos = np.linalg.norm(pos[above, :2], axis=1)
    z_pos = pos[above, 2]

    # Fit linear: z = a * r + b; repose angle = arctan(-a)
    if len(radial_pos) > 2:
        A = np.vstack([radial_pos, np.ones(len(radial_pos))]).T
        result = np.linalg.lstsq(A, z_pos, rcond=None)
        slope = result[0][0]
        repose_angle = math.degrees(math.atan(abs(slope)))
    else:
        repose_angle = 0.0

    return {
        "repose_angle_deg": repose_angle,
        "n_particles": n,
        "ke_final_J": float(ke),
        "n_steps": n_pre + n_settle,
        "dt": dt,
    }


def _simulate_shear_box(mat: Material3D, bench: ShearBoxBenchmark | None = None,
                        *, max_steps: int | None = None) -> dict:
    """Simulate direct shear box test.

    The lower half of the specimen is fixed; the upper half is sheared
    horizontally under constant normal stress. The shear stress vs displacement
    curve yields the peak shear strength, from which the internal friction angle
    and cohesion are derived.

    Returns dict with phi_deg, cohesion_Pa, peak_shear_Pa, normal_stress_Pa.
    """
    bench = bench or ShearBoxBenchmark()
    r = mat.radius
    pos = hcp_positions_in_box(
        r, bench.box_width, bench.box_depth, bench.fill_height,
        packing_gap=bench.packing_gap, jitter=bench.jitter, seed=bench.seed,
    )
    n = len(pos)
    vel = np.zeros((n, 3), dtype=np.float64)
    omega = np.zeros((n, 3), dtype=np.float64)

    dt = mat.timestep
    state = ContactState()

    # Split into lower (fixed) and upper (mobile) halves
    z_mid = bench.fill_height / 2.0
    lower = pos[:, 2] < z_mid
    upper = ~lower

    # Pre-consolidation phase
    max_c = 200 if max_steps is None else min(200, max_steps)
    for _ in range(max_c):
        # Apply normal stress as body force on upper particles
        normal_force = bench.normal_stress_Pa * bench.box_width * bench.box_depth
        n_upper = upper.sum()
        if n_upper > 0:
            vel[upper, 2] -= (normal_force / n_upper / mat.mass) * dt

        iu, ju = build_pairs(pos, r)
        f, t = contact_forces(pos, vel, omega, iu, ju, state, mat, dt)

        # Floor constraint
        below = pos[:, 2] < r
        if below.any():
            pos[below, 2] = r
            vel[below, 2] = np.maximum(vel[below, 2], 0.0)

        vel[lower] *= 0.9  # damp lower half
        vel += f / mat.mass * dt
        omega += t / mat.inertia * dt
        vel *= 0.95
        omega *= 0.95
        pos += vel * dt

    # Shear phase: displace upper half horizontally
    n_shear = int(bench.shear_displacement / (dt * 0.01)) + 1
    n_shear = min(n_shear, 300) if max_steps is None else min(n_shear, max_steps)
    shear_vel = 0.01  # m/s

    shear_stress_history: list[float] = []
    displacement_history: list[float] = []

    for step in range(n_shear):
        # Impose shear velocity on upper half
        vel[upper, 0] = shear_vel

        iu, ju = build_pairs(pos, r)
        f, t = contact_forces(pos, vel, omega, iu, ju, state, mat, dt)

        # Floor constraint
        below = pos[:, 2] < r
        if below.any():
            pos[below, 2] = r
            vel[below, 2] = np.maximum(vel[below, 2], 0.0)

        vel[lower] *= 0.9
        vel += f / mat.mass * dt
        omega += t / mat.inertia * dt
        vel *= 0.95
        omega *= 0.95
        pos += vel * dt

        # Measure shear stress from contact forces on lower half
        shear_force = f[lower, 0].sum()
        area = bench.box_width * bench.box_depth
        shear_stress_history.append(abs(shear_force) / area)
        displacement_history.append(step * shear_vel * dt)

    peak_shear = max(shear_stress_history) if shear_stress_history else 0.0
    normal_stress = bench.normal_stress_Pa

    # Mohr-Coulomb: tau = c + sigma * tan(phi)
    # With one normal stress, assume c=0 for cohesionless material
    # phi = arctan(tau / sigma)
    if normal_stress > 0:
        phi = math.degrees(math.atan(peak_shear / normal_stress))
        cohesion = 0.0
    else:
        phi = 0.0
        cohesion = peak_shear

    return {
        "phi_deg": phi,
        "cohesion_Pa": cohesion,
        "peak_shear_Pa": peak_shear,
        "normal_stress_Pa": normal_stress,
        "n_particles": n,
        "shear_stress_history": shear_stress_history,
        "displacement_history": displacement_history,
    }


def _simulate_compression(mat: Material3D, bench: CompressionBenchmark | None = None,
                          *, max_steps: int | None = None) -> dict:
    """Simulate uniaxial compression test.

    A cylindrical specimen is compressed between two platens at a constant
    strain rate. The stress-strain curve yields the elastic modulus and
    Poisson ratio.

    Returns dict with E_Pa, nu, peak_stress_Pa, stress_strain_curve.
    """
    bench = bench or CompressionBenchmark()
    r = mat.radius
    pos = hcp_positions_in_cylinder(
        r, bench.specimen_radius, bench.specimen_height,
        packing_gap=bench.packing_gap, jitter=bench.jitter, seed=bench.seed,
    )
    n = len(pos)
    vel = np.zeros((n, 3), dtype=np.float64)
    omega = np.zeros((n, 3), dtype=np.float64)

    dt = mat.timestep
    state = ContactState()

    # Pre-consolidation
    max_cp = 100 if max_steps is None else min(100, max_steps)
    for _ in range(max_cp):
        vel[:, 2] -= mat.gravity * dt
        iu, ju = build_pairs(pos, r)
        f, t = contact_forces(pos, vel, omega, iu, ju, state, mat, dt)

        below = pos[:, 2] < r
        if below.any():
            pos[below, 2] = r
            vel[below, 2] = np.maximum(vel[below, 2], 0.0)

        vel *= 0.95
        omega *= 0.95
        pos += vel * dt

    # Compression phase
    initial_height = bench.specimen_height
    n_steps = int(bench.max_strain / bench.strain_rate / dt) + 1
    n_steps = min(n_steps, 300) if max_steps is None else min(n_steps, max_steps)

    stress_history: list[float] = []
    strain_history: list[float] = []
    lateral_strain_history: list[float] = []

    for step in range(n_steps):
        # Impose platen velocity
        platen_vel = -bench.strain_rate * initial_height
        vel[:, 2] += platen_vel * dt * 0.1  # gradual application

        iu, ju = build_pairs(pos, r)
        f, t = contact_forces(pos, vel, omega, iu, ju, state, mat, dt)

        # Floor constraint
        below = pos[:, 2] < r
        if below.any():
            pos[below, 2] = r
            vel[below, 2] = np.maximum(vel[below, 2], 0.0)

        # Top constraint (platen)
        current_height = pos[:, 2].max() - pos[:, 2].min()
        target_height = initial_height * (1 - bench.strain_rate * step * dt)
        above = pos[:, 2] > target_height + r
        if above.any():
            pos[above, 2] = target_height + r
            vel[above, 2] = np.minimum(vel[above, 2], 0.0)

        vel *= 0.95
        omega *= 0.95
        pos += vel * dt

        # Measure stress
        axial_force = f[:, 2].sum()
        area = math.pi * bench.specimen_radius ** 2
        stress = abs(axial_force) / area
        strain = (initial_height - current_height) / initial_height

        stress_history.append(stress)
        strain_history.append(max(strain, 1e-10))

        # Lateral strain
        radial = np.linalg.norm(pos[:, :2], axis=1)
        avg_radial = radial.mean()
        lateral_strain = (avg_radial - bench.specimen_radius) / bench.specimen_radius
        lateral_strain_history.append(lateral_strain)

    # Compute elastic modulus from initial linear portion
    if len(stress_history) > 2 and strain_history[1] > 0:
        # Use first 30% of data for linear fit
        n_linear = max(3, len(stress_history) // 3)
        s = np.array(stress_history[:n_linear])
        e = np.array(strain_history[:n_linear])
        if e[-1] > 0:
            A = np.vstack([e, np.ones(n_linear)]).T
            result = np.linalg.lstsq(A, s, rcond=None)
            E = result[0][0]
        else:
            E = 0.0
    else:
        E = 0.0

    # Poisson ratio from lateral vs axial strain
    if len(lateral_strain_history) > 2 and strain_history[-1] > 0:
        n_linear = max(3, len(lateral_strain_history) // 3)
        ls = np.array(lateral_strain_history[:n_linear])
        es = np.array(strain_history[:n_linear])
        if es[-1] > 0:
            A = np.vstack([es, np.ones(n_linear)]).T
            result = np.linalg.lstsq(A, ls, rcond=None)
            nu = abs(result[0][0])
        else:
            nu = 0.0
    else:
        nu = 0.0

    peak_stress = max(stress_history) if stress_history else 0.0

    return {
        "E_Pa": E,
        "nu": nu,
        "peak_stress_Pa": peak_stress,
        "stress_strain_curve": list(zip(strain_history, stress_history)),
        "n_particles": n,
    }


# ============================================================================
# 2. Parameter space and objective function
# ============================================================================

@dataclass(frozen=True)
class CalibrationParameter:
    """A single calibration parameter with bounds."""
    name: str
    low: float
    high: float
    prior_mean: float | None = None
    prior_std: float | None = None


# Default parameter space for DEM calibration
DEFAULT_PARAMETERS = [
    CalibrationParameter("mu", 0.1, 0.8, prior_mean=0.4, prior_std=0.15),
    CalibrationParameter("mu_rolling", 0.0, 0.15, prior_mean=0.05, prior_std=0.03),
    CalibrationParameter("restitution", 0.1, 0.6, prior_mean=0.3, prior_std=0.1),
    CalibrationParameter("kt_ratio", 0.3, 0.8, prior_mean=0.5, prior_std=0.1),
]


@dataclass
class ExperimentalData:
    """Experimental benchmark data for calibration."""
    repose_angle_deg: float | None = None
    repose_angle_std: float = 2.0  # measurement uncertainty
    shear_phi_deg: float | None = None
    shear_phi_std: float = 1.5
    shear_cohesion_Pa: float | None = None
    shear_cohesion_std: float = 500.0
    compression_E_Pa: float | None = None
    compression_E_std: float = 1e6
    compression_nu: float | None = None
    compression_nu_std: float = 0.05


def build_material(params: dict, *, radius: float = 0.005, density: float = 2500.0,
                   kn: float = 1e5, gravity: float = 9.81) -> Material3D:
    """Build a Material3D from calibration parameters."""
    return Material3D(
        radius=radius,
        density=density,
        mu=params.get("mu", 0.5),
        mu_rolling=params.get("mu_rolling", 0.05),
        restitution=params.get("restitution", 0.3),
        kn=kn,
        kt_ratio=params.get("kt_ratio", 0.5),
        gravity=gravity,
    )


def objective_function(
    params: dict,
    exp_data: ExperimentalData,
    *,
    radius: float = 0.005,
    density: float = 2500.0,
    kn: float = 1e5,
    repose_bench: ReposeBenchmark | None = None,
    shear_bench: ShearBoxBenchmark | None = None,
    compression_bench: CompressionBenchmark | None = None,
    max_steps: int | None = None,
) -> dict:
    """Compute the objective (misfit) between simulation and experiment.

    Returns dict with total_objective, component_objectives, and simulation results.
    """
    mat = build_material(params, radius=radius, density=density, kn=kn)
    components: dict[str, float] = {}
    total = 0.0

    # Repose angle component
    if exp_data.repose_angle_deg is not None:
        sim = _simulate_repose(mat, repose_bench, max_steps=max_steps)
        angle = sim["repose_angle_deg"]
        diff = angle - exp_data.repose_angle_deg
        components["repose"] = (diff / exp_data.repose_angle_std) ** 2
        total += components["repose"]

    # Shear box component
    if exp_data.shear_phi_deg is not None:
        sim_shear = _simulate_shear_box(mat, shear_bench, max_steps=max_steps)
        phi = sim_shear["phi_deg"]
        diff_phi = phi - exp_data.shear_phi_deg
        components["shear_phi"] = (diff_phi / exp_data.shear_phi_std) ** 2
        total += components["shear_phi"]

        if exp_data.shear_cohesion_Pa is not None:
            c = sim_shear["cohesion_Pa"]
            diff_c = c - exp_data.shear_cohesion_Pa
            components["shear_c"] = (diff_c / exp_data.shear_cohesion_std) ** 2
            total += components["shear_c"]

    # Compression component
    if exp_data.compression_E_Pa is not None:
        sim_comp = _simulate_compression(mat, compression_bench, max_steps=max_steps)
        E = sim_comp["E_Pa"]
        diff_E = E - exp_data.compression_E_Pa
        components["compression_E"] = (diff_E / exp_data.compression_E_std) ** 2
        total += components["compression_E"]

        if exp_data.compression_nu is not None:
            nu = sim_comp["nu"]
            diff_nu = nu - exp_data.compression_nu
            components["compression_nu"] = (diff_nu / exp_data.compression_nu_std) ** 2
            total += components["compression_nu"]

    return {
        "total_objective": total,
        "components": components,
        "params": params,
    }


# ============================================================================
# 3. Bayesian Optimization (GP surrogate + EI/UCB acquisition)
# ============================================================================

class GaussianProcessSurrogate:
    """Simple GP surrogate model for Bayesian optimization.

    Uses a squared-exponential kernel with automatic relevance determination
    (ARD) for multi-dimensional parameter spaces.
    """

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6):
        self.length_scale = length_scale
        self.noise = noise
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.K_inv: np.ndarray | None = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Squared-exponential kernel with ARD."""
        # Normalize by length scale
        X1s = X1 / self.length_scale
        X2s = X2 / self.length_scale
        # Compute pairwise squared distances
        sq_dist = (
            np.sum(X1s ** 2, axis=1, keepdims=True)
            + np.sum(X2s ** 2, axis=1, keepdims=True).T
            - 2 * X1s @ X2s.T
        )
        sq_dist = np.maximum(sq_dist, 0.0)
        return np.exp(-0.5 * sq_dist)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP to observed data."""
        self.X = X.copy()
        self.y = y.copy()
        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        self.K_inv = np.linalg.solve(K, np.eye(len(X)))

    def predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at new points."""
        if self.X is None or self.K_inv is None:
            return np.zeros(len(X_new)), np.ones(len(X_new))

        K_star = self._kernel(X_new, self.X)
        mu = K_star @ self.K_inv @ self.y

        K_ss = self._kernel(X_new, X_new)
        var = np.diag(K_ss - K_star @ self.K_inv @ K_star.T)
        var = np.maximum(var, 1e-10)

        return mu, var

    def expected_improvement(self, X_new: np.ndarray, y_best: float,
                             xi: float = 0.01) -> np.ndarray:
        """Expected Improvement acquisition function."""
        mu, var = self.predict(X_new)
        sigma = np.sqrt(var)

        with np.errstate(divide="ignore", invalid="ignore"):
            Z = (y_best - mu - xi) / sigma
            ei = (y_best - mu - xi) * _norm_cdf(Z) + sigma * _norm_pdf(Z)
            ei = np.where(sigma > 0, ei, 0.0)

        return ei

    def ucb(self, X_new: np.ndarray, kappa: float = 2.0) -> np.ndarray:
        """Upper Confidence Bound acquisition function (for minimization, use negative)."""
        mu, var = self.predict(X_new)
        return -(mu - kappa * np.sqrt(var))  # negative for minimization


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF approximation."""
    return 0.5 * (1.0 + _scipy_erf(x / np.sqrt(2.0)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)


def latin_hypercube_sample(n: int, dim: int, *, seed: int = 42) -> np.ndarray:
    """Latin hypercube sampling for initial design."""
    rng = np.random.default_rng(seed)
    samples = np.zeros((n, dim))
    for d in range(dim):
        perm = rng.permutation(n)
        samples[:, d] = (perm + rng.uniform(0, 1, n)) / n
    return samples


def bayesian_optimize(
    objective: Callable[[dict], float],
    parameters: list[CalibrationParameter],
    n_initial: int = 10,
    n_iterations: int = 50,
    acquisition: str = "EI",
    xi: float = 0.01,
    kappa: float = 2.0,
    seed: int = 42,
) -> dict:
    """Run Bayesian optimization to find optimal DEM parameters.

    Parameters
    ----------
    objective : callable
        Function that takes a dict of parameter values and returns the objective.
    parameters : list of CalibrationParameter
        Parameter definitions with bounds.
    n_initial : int
        Number of initial Latin hypercube samples.
    n_iterations : int
        Number of BO iterations.
    acquisition : str
        Acquisition function: "EI" or "UCB".
    xi : float
        Exploration parameter for EI.
    kappa : float
        Exploration parameter for UCB.
    seed : int
        Random seed.

    Returns
    -------
    dict with best_params, best_objective, history, gp_model.
    """
    dim = len(parameters)
    rng = np.random.default_rng(seed)

    # Scale parameters to [0, 1]
    def to_scaled(params: dict) -> np.ndarray:
        arr = np.zeros(dim)
        for i, p in enumerate(parameters):
            arr[i] = (params[p.name] - p.low) / (p.high - p.low)
        return arr

    def from_scaled(scaled: np.ndarray) -> dict:
        params = {}
        for i, p in enumerate(parameters):
            params[p.name] = p.low + scaled[i] * (p.high - p.low)
        return params

    # Initial design
    X_init = latin_hypercube_sample(n_initial, dim, seed=seed)
    y_init = np.zeros(n_initial)
    for i in range(n_initial):
        params = from_scaled(X_init[i])
        y_init[i] = objective(params)

    X = X_init.copy()
    y = y_init.copy()

    # GP surrogate
    gp = GaussianProcessSurrogate(length_scale=0.5, noise=1e-4)
    gp.fit(X, y)

    history = [{"iteration": i, "params": from_scaled(X[i]), "objective": float(y[i])}
               for i in range(n_initial)]

    y_best = float(np.min(y))
    best_idx = int(np.argmin(y))
    best_params = from_scaled(X[best_idx])

    # BO iterations
    n_candidates = 500
    for it in range(n_iterations):
        # Generate candidates
        X_cand = rng.uniform(0, 1, size=(n_candidates, dim))

        # Acquisition
        if acquisition == "EI":
            acq = gp.expected_improvement(X_cand, y_best, xi=xi)
        else:
            acq = gp.ucb(X_cand, kappa=kappa)

        # Select best candidate
        next_idx = int(np.argmax(acq))
        x_next = X_cand[next_idx]
        params_next = from_scaled(x_next)
        y_next = objective(params_next)

        # Update
        X = np.vstack([X, x_next.reshape(1, -1)])
        y = np.append(y, y_next)
        gp.fit(X, y)

        if y_next < y_best:
            y_best = float(y_next)
            best_params = params_next

        history.append({
            "iteration": n_initial + it,
            "params": params_next,
            "objective": float(y_next),
        })

    return {
        "best_params": best_params,
        "best_objective": y_best,
        "history": history,
        "n_evaluations": len(X),
        "converged": y_best < 1.0,  # heuristic: objective < 1 means good fit
    }


# ============================================================================
# 4. Uncertainty Quantification (MCMC posterior sampling)
# ============================================================================

def log_posterior(
    params: dict,
    exp_data: ExperimentalData,
    parameters: list[CalibrationParameter],
    *,
    radius: float = 0.005,
    density: float = 2500.0,
    kn: float = 1e5,
    repose_bench: ReposeBenchmark | None = None,
    shear_bench: ShearBoxBenchmark | None = None,
    compression_bench: CompressionBenchmark | None = None,
    max_steps: int | None = None,
) -> float:
    """Compute log-posterior for MCMC sampling.

    log p(theta | data) ∝ log p(data | theta) + log p(theta)
    """
    # Prior: Gaussian around prior_mean with prior_std
    log_prior = 0.0
    for p in parameters:
        if p.prior_mean is not None and p.prior_std is not None:
            val = params.get(p.name, p.prior_mean)
            log_prior -= 0.5 * ((val - p.prior_mean) / p.prior_std) ** 2
        else:
            # Uniform prior within bounds
            val = params.get(p.name, (p.low + p.high) / 2)
            if val < p.low or val > p.high:
                return -np.inf

    # Likelihood: Gaussian error model
    mat = build_material(params, radius=radius, density=density, kn=kn)
    log_likelihood = 0.0

    if exp_data.repose_angle_deg is not None:
        sim = _simulate_repose(mat, repose_bench, max_steps=max_steps)
        angle = sim["repose_angle_deg"]
        diff = angle - exp_data.repose_angle_deg
        log_likelihood -= 0.5 * (diff / exp_data.repose_angle_std) ** 2

    if exp_data.shear_phi_deg is not None:
        sim_shear = _simulate_shear_box(mat, shear_bench, max_steps=max_steps)
        phi = sim_shear["phi_deg"]
        diff_phi = phi - exp_data.shear_phi_deg
        log_likelihood -= 0.5 * (diff_phi / exp_data.shear_phi_std) ** 2

        if exp_data.shear_cohesion_Pa is not None:
            c = sim_shear["cohesion_Pa"]
            diff_c = c - exp_data.shear_cohesion_Pa
            log_likelihood -= 0.5 * (diff_c / exp_data.shear_cohesion_std) ** 2

    if exp_data.compression_E_Pa is not None:
        sim_comp = _simulate_compression(mat, compression_bench, max_steps=max_steps)
        E = sim_comp["E_Pa"]
        diff_E = E - exp_data.compression_E_Pa
        log_likelihood -= 0.5 * (diff_E / exp_data.compression_E_std) ** 2

        if exp_data.compression_nu is not None:
            nu = sim_comp["nu"]
            diff_nu = nu - exp_data.compression_nu
            log_likelihood -= 0.5 * (diff_nu / exp_data.compression_nu_std) ** 2

    return log_prior + log_likelihood


def metropolis_hastings(
    exp_data: ExperimentalData,
    parameters: list[CalibrationParameter],
    n_samples: int = 1000,
    burn_in: int = 200,
    step_size: float = 0.05,
    seed: int = 42,
    max_steps: int | None = None,
    **sim_kwargs,
) -> dict:
    """Run Metropolis-Hastings MCMC for parameter uncertainty quantification.

    Returns posterior samples, acceptance rate, and summary statistics.
    """
    rng = np.random.default_rng(seed)
    dim = len(parameters)

    # Initialize at prior means
    current = {}
    for p in parameters:
        current[p.name] = p.prior_mean if p.prior_mean is not None else (p.low + p.high) / 2

    current_lp = log_posterior(current, exp_data, parameters, max_steps=max_steps, **sim_kwargs)
    if not math.isfinite(current_lp):
        current_lp = -1e10

    samples: list[dict] = []
    n_accepted = 0

    for i in range(n_samples + burn_in):
        # Propose
        proposed = {}
        for p in parameters:
            val = current[p.name] + rng.normal(0, step_size * (p.high - p.low))
            proposed[p.name] = max(p.low, min(p.high, val))

        proposed_lp = log_posterior(proposed, exp_data, parameters, max_steps=max_steps, **sim_kwargs)

        # Accept/reject
        log_alpha = proposed_lp - current_lp
        if math.isfinite(log_alpha) and rng.uniform() < min(1.0, math.exp(log_alpha)):
            current = proposed
            current_lp = proposed_lp
            n_accepted += 1

        if i >= burn_in:
            samples.append(dict(current))

    acceptance_rate = n_accepted / (n_samples + burn_in)

    # Compute summary statistics
    param_names = [p.name for p in parameters]
    posterior_summary = {}
    for name in param_names:
        values = [s[name] for s in samples]
        values_arr = np.array(values)
        posterior_summary[name] = {
            "mean": float(np.mean(values_arr)),
            "std": float(np.std(values_arr)),
            "median": float(np.median(values_arr)),
            "ci_95_low": float(np.percentile(values_arr, 2.5)),
            "ci_95_high": float(np.percentile(values_arr, 97.5)),
            "ci_90_low": float(np.percentile(values_arr, 5.0)),
            "ci_90_high": float(np.percentile(values_arr, 95.0)),
            "samples": values,
        }

    return {
        "posterior_summary": posterior_summary,
        "acceptance_rate": acceptance_rate,
        "n_samples": n_samples,
        "burn_in": burn_in,
        "samples": samples,
    }


# ============================================================================
# 5. Complete calibration pipeline
# ============================================================================

def run_calibration(
    exp_data: ExperimentalData,
    parameters: list[CalibrationParameter] | None = None,
    *,
    n_bo_initial: int = 10,
    n_bo_iterations: int = 50,
    n_mcmc_samples: int = 500,
    n_mcmc_burn_in: int = 100,
    acquisition: str = "EI",
    seed: int = 42,
    radius: float = 0.005,
    density: float = 2500.0,
    kn: float = 1e5,
    max_steps: int | None = None,
) -> dict:
    """Run the complete auto-calibration pipeline.

    1. Bayesian optimization to find best-fit parameters
    2. MCMC to quantify parameter uncertainty

    Returns comprehensive results dict.
    """
    parameters = parameters or DEFAULT_PARAMETERS
    t0 = time.time()

    # Build objective function
    def obj(params: dict) -> float:
        result = objective_function(
            params, exp_data,
            radius=radius, density=density, kn=kn,
            max_steps=max_steps,
        )
        return result["total_objective"]

    # Step 1: Bayesian optimization
    bo_result = bayesian_optimize(
        obj, parameters,
        n_initial=n_bo_initial,
        n_iterations=n_bo_iterations,
        acquisition=acquisition,
        seed=seed,
    )

    # Step 2: MCMC uncertainty quantification
    mcmc_result = metropolis_hastings(
        exp_data, parameters,
        n_samples=n_mcmc_samples,
        burn_in=n_mcmc_burn_in,
        seed=seed,
        max_steps=max_steps,
        radius=radius, density=density, kn=kn,
    )

    # Step 3: Forward validation with best parameters
    best_params = bo_result["best_params"]
    validation = objective_function(
        best_params, exp_data,
        radius=radius, density=density, kn=kn,
        max_steps=max_steps,
    )

    elapsed = time.time() - t0

    return {
        "best_params": best_params,
        "best_objective": bo_result["best_objective"],
        "bo_history": bo_result["history"],
        "bo_converged": bo_result["converged"],
        "bo_n_evaluations": bo_result["n_evaluations"],
        "mcmc_posterior": mcmc_result["posterior_summary"],
        "mcmc_acceptance_rate": mcmc_result["acceptance_rate"],
        "validation": validation,
        "elapsed_seconds": elapsed,
        "parameters": [p.name for p in parameters],
    }


# ============================================================================
# 6. Material-specific calibration presets
# ============================================================================

# Glass beads: smooth, low friction, low rolling resistance
GLASS_BEADS_EXP = ExperimentalData(
    repose_angle_deg=22.0,
    repose_angle_std=1.5,
    shear_phi_deg=22.0,
    shear_phi_std=1.5,
    shear_cohesion_Pa=0.0,
    shear_cohesion_std=200.0,
)

GLASS_BEADS_PARAMS = [
    CalibrationParameter("mu", 0.1, 0.5, prior_mean=0.25, prior_std=0.08),
    CalibrationParameter("mu_rolling", 0.0, 0.08, prior_mean=0.02, prior_std=0.02),
    CalibrationParameter("restitution", 0.5, 0.9, prior_mean=0.7, prior_std=0.1),
    CalibrationParameter("kt_ratio", 0.3, 0.7, prior_mean=0.5, prior_std=0.1),
]

# Sand: higher friction, moderate rolling resistance
SAND_EXP = ExperimentalData(
    repose_angle_deg=34.0,
    repose_angle_std=2.0,
    shear_phi_deg=34.0,
    shear_phi_std=2.0,
    shear_cohesion_Pa=0.0,
    shear_cohesion_std=300.0,
)

SAND_PARAMS = [
    CalibrationParameter("mu", 0.3, 0.8, prior_mean=0.55, prior_std=0.1),
    CalibrationParameter("mu_rolling", 0.02, 0.12, prior_mean=0.06, prior_std=0.02),
    CalibrationParameter("restitution", 0.1, 0.5, prior_mean=0.3, prior_std=0.1),
    CalibrationParameter("kt_ratio", 0.3, 0.7, prior_mean=0.5, prior_std=0.1),
]

# Gravel: high friction, high rolling resistance
GRAVEL_EXP = ExperimentalData(
    repose_angle_deg=40.0,
    repose_angle_std=2.5,
    shear_phi_deg=40.0,
    shear_phi_std=2.5,
    shear_cohesion_Pa=500.0,
    shear_cohesion_std=200.0,
)

GRAVEL_PARAMS = [
    CalibrationParameter("mu", 0.4, 0.8, prior_mean=0.6, prior_std=0.1),
    CalibrationParameter("mu_rolling", 0.05, 0.15, prior_mean=0.08, prior_std=0.03),
    CalibrationParameter("restitution", 0.1, 0.4, prior_mean=0.2, prior_std=0.08),
    CalibrationParameter("kt_ratio", 0.3, 0.7, prior_mean=0.5, prior_std=0.1),
]

MATERIAL_PRESETS = {
    "glass_beads": (GLASS_BEADS_EXP, GLASS_BEADS_PARAMS),
    "sand": (SAND_EXP, SAND_PARAMS),
    "gravel": (GRAVEL_EXP, GRAVEL_PARAMS),
}
