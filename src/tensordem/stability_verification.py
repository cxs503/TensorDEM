"""Engine-level verification of the explicit time-step stability threshold.

Why this module exists
----------------------
:mod:`tensordem.stability` supplies the *closed-form* amplification factor of the
semi-implicit Euler scheme and a Gershgorin bound on the highest eigenfrequency
of the assembled bonded-particle system.  Its evidence was structural only:
``DEM3DConfig.recommended_dt`` caps the usable step at ~2% of the audited limit
and ``IceDEM3D.__init__`` refuses anything above it, so the **failure side** of
the threshold (``dt > dt_crit`` -> perturbations grow) had never been observed in
the solver itself.  ``docs/dem3d_timestep_stability.md`` recorded that gap as
``Engine-level threshold crossing verified dynamically  【未验证】``.

This module closes it by driving the *actual* solver across its own threshold:

1. :func:`assemble_stiffness` re-derives the translational stiffness matrix of
   the bonded lattice from the bond list and the central-force law.  It never
   calls the solver, so it is an independent reference.
2. :func:`top_mode_analytic` diagonalises that matrix (``torch.linalg.eigh``) and
   returns ``omega_max = sqrt(lambda_max / m)`` -- the frequency that fixes
   ``dt_crit = 2 / omega_max``.
3. :func:`stiffness_apply` applies the *solver's own* force routine to a
   displacement field, so :func:`engine_rayleigh_omega` measures the same
   frequency from inside the engine.  Agreement between (2) and (3) shows the
   engine force law is the gradient of the lattice energy it claims to be.
4. :func:`engine_amplification_matrix` submits two basis states to
   ``IceDEM3D.step`` and reconstructs the 2x2 per-step amplification matrix in
   the top-mode subspace, giving a *measured* spectral radius ``rho``.
5. :func:`threshold_sweep` compares ``rho`` with
   :func:`tensordem.stability.symplectic_euler_growth_factor` across
   ``dt / dt_crit``; :func:`bisect_critical_step` locates the ``rho = 1``
   crossing from engine measurements alone.
6. :func:`long_run_growth` integrates the engine for many steps on either side
   of the threshold.  Above it the amplitude follows the closed-form ``rho**n``;
   below it the scheme is *marginal* (``rho = 1`` exactly for the undamped
   oscillator), so boundedness is measured with the exact quadratic invariant of
   the scheme rather than with the physical energy, which oscillates by an
   O(1) fraction at ``dt -> dt_crit``.

Scope, stated honestly
----------------------
The demonstration is *single-mode*: it shows that the step the engine actually
takes obeys the closed-form threshold for the highest eigenmode of its own
lattice, which is exactly what the Gershgorin bound claims.  It is a
numerical-stability statement about the integrator, not a statement about
material behaviour, and it does not claim that every nonlinear engine
trajectory above ``dt_crit`` is eventually non-finite.  Steps above the
threshold are reached through ``DEM3DConfig.allow_dt_above_recommended``, a
verification-only escape hatch; physical runs must leave it ``False``.

All quantities are SI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import torch

from . import stability
from .dem3d import DEM3DConfig, IceDEM3D

PROTOCOL = "tensordem-engine-stability-threshold-v1"

# Amplitude of the displacement probes.  Small enough that the bond law is
# linear to ~1e-9 relative (strain ~2e-5 against breaking_strain = 1.0) and no
# pair can come into overlap, yet far above float64 cancellation in
# ``|delta| - rest_length``.
PROBE_AMPLITUDE = 1.0e-6

# The undamped semi-implicit Euler scheme is *marginally* stable below the
# threshold (rho = 1 exactly), so a measured rho has to be compared with
# 1 + MARGINAL_TOLERANCE to decide "grows" versus "bounded".  The tolerance is
# ~1e3 times the observed float64 noise floor of the measurement.
MARGINAL_TOLERANCE = 1.0e-6


def reference_config(**overrides: Any) -> DEM3DConfig:
    """Harmonic reference configuration: bonded lattice only.

    Drag, pair damping, friction, rolling resistance, the indenter and the
    platens are all switched off and the strain thresholds are raised out of
    reach, so the only force is the linear bond spring.  The model is then an
    exactly linear oscillator network and ``dt_crit`` is a property of the
    scheme, not of an event.
    """
    defaults: dict[str, Any] = dict(
        nx=3,
        ny=3,
        nz=3,
        radius=0.025,
        density=917.0,
        bond_stiffness=2_000.0,
        contact_stiffness=5_000.0,
        contact_damping=0.0,
        drag=0.0,
        tool_speed=1.0e-6,
        tool_radius=1.0e-6,
        tool_gap=1.0,
        friction_coefficient=0.0,
        rolling_resistance_coefficient=0.0,
        restitution_coefficient=1.0,
        fix_x_edges=False,
        fix_bottom=False,
        breaking_strain=1.0,
        shear_breaking_strain=1.0,
        allow_dt_above_recommended=True,
    )
    defaults.update(overrides)
    return DEM3DConfig(**defaults)


def reference_model(dt: float | None = None, **overrides: Any) -> IceDEM3D:
    """Build the harmonic reference model, optionally at an audited step ``dt``."""
    if dt is not None:
        overrides["dt"] = float(dt)
    return IceDEM3D(reference_config(**overrides))


# ---------------------------------------------------------------------------
# 1-2. Independent (solver-free) stiffness assembly and diagonalisation
# ---------------------------------------------------------------------------
def assemble_stiffness(model: IceDEM3D) -> torch.Tensor:
    """Translational stiffness matrix of the alive bond network (``3N x 3N``).

    For a central-force bond ``(i, j)`` with unit normal ``n`` and stiffness
    ``k`` the linearised force on ``i`` is ``k n n^T (u_j - u_i)``, hence

        ``K_ii += k n n^T``,  ``K_jj += k n n^T``,  ``K_ij -= k n n^T``.

    Fixed particles are projected out; all particles are free in
    :func:`reference_config`.
    """
    cfg = model.config
    n = len(model.positions)
    free = ~model.fixed
    dofs = int(free.sum().item())
    index = torch.full((n,), -1, dtype=torch.long)
    index[free.flatten()] = torch.arange(dofs)
    k_matrix = torch.zeros((3 * dofs, 3 * dofs), dtype=torch.float64)
    ref = model.initial_positions
    i, j = model.pairs.T
    delta = ref[j] - ref[i]
    length = torch.linalg.vector_norm(delta, dim=1).clamp_min(torch.finfo(torch.float64).eps)
    normals = delta / length[:, None]
    for bond in range(len(model.pairs)):
        if not bool(model.alive[bond]):
            continue
        outer = cfg.bond_stiffness * torch.outer(normals[bond], normals[bond])
        row_i = int(index[int(i[bond])].item())
        row_j = int(index[int(j[bond])].item())
        if row_i >= 0:
            block = slice(3 * row_i, 3 * row_i + 3)
            k_matrix[block, block] += outer
        if row_j >= 0:
            block = slice(3 * row_j, 3 * row_j + 3)
            k_matrix[block, block] += outer
        if row_i >= 0 and row_j >= 0:
            bi = slice(3 * row_i, 3 * row_i + 3)
            bj = slice(3 * row_j, 3 * row_j + 3)
            k_matrix[bi, bj] -= outer
            k_matrix[bj, bi] -= outer
    return k_matrix


@dataclass(frozen=True)
class AnalyticMode:
    """Top eigenpair of the independently assembled stiffness matrix."""

    omega_rad_s: float
    eigenvalue: float
    eigenvector: torch.Tensor = field(repr=False)
    n_free: int
    eigenvalue_gap: float


def top_mode_analytic(model: IceDEM3D) -> AnalyticMode:
    """Highest eigenpair of :func:`assemble_stiffness` (no solver call)."""
    k_matrix = assemble_stiffness(model)
    if k_matrix.numel() == 0:
        raise ValueError("model has no free degrees of freedom")
    eigenvalues, eigenvectors = torch.linalg.eigh(k_matrix)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    top = float(eigenvalues[0].item())
    gap = float((eigenvalues[0] - eigenvalues[1]).item()) if len(eigenvalues) > 1 else math.inf
    vector = eigenvectors[:, 0].reshape(-1, 3).clone()
    return AnalyticMode(
        omega_rad_s=math.sqrt(max(top, 0.0) / model.config.mass),
        eigenvalue=top,
        eigenvector=vector,
        n_free=len(vector),
        eigenvalue_gap=gap,
    )


# ---------------------------------------------------------------------------
# 3. Same operator through the solver's own force routine
# ---------------------------------------------------------------------------
def stiffness_apply(model: IceDEM3D, displacement: torch.Tensor) -> torch.Tensor:
    """``K u`` evaluated by ``IceDEM3D.forces`` (no history, no fracture).

    Note: this writes ``model.positions`` and zeroes ``model.velocities``; it is
    a probe, not a step.  Callers that care about the velocity state restore it.
    """
    model.positions = model.initial_positions + displacement
    model.velocities = torch.zeros_like(model.positions)
    force, _ = model.forces(None, update_fracture=False, update_history=False)
    return -force


def engine_rayleigh_omega(
    model: IceDEM3D,
    displacement: torch.Tensor | None = None,
    *,
    iterations: int = 400,
    tolerance: float = 1.0e-14,
    seed: int = 20261103,
) -> dict[str, Any]:
    """Highest eigenfrequency as seen by the engine, plus the mode residual.

    Power iteration on ``K`` applied through :func:`stiffness_apply`; the
    frequency is the Rayleigh quotient ``sqrt(u.Ku / (m u.u))``.  The residual
    ``||K u - lambda u|| / (lambda ||u||)`` measures how close the returned field
    is to an exact mode -- the growth experiment below is only meaningful when
    it is small.
    """
    generator = torch.Generator().manual_seed(seed)
    if displacement is None:
        state = torch.randn(model.initial_positions.shape, generator=generator, dtype=torch.float64)
    else:
        state = displacement.clone()
    state = state / torch.linalg.vector_norm(state)
    mass = model.config.mass
    previous = 0.0
    used = 0
    for used in range(1, iterations + 1):
        applied = stiffness_apply(model, PROBE_AMPLITUDE * state) / PROBE_AMPLITUDE
        rayleigh = float((state * applied).sum().item())
        norm = float(torch.linalg.vector_norm(applied).item())
        if norm <= 0.0:
            raise ValueError("stiffness operator is identically zero")
        state = applied / norm
        if abs(rayleigh - previous) <= tolerance * max(abs(rayleigh), 1.0):
            previous = rayleigh
            break
        previous = rayleigh
    applied = stiffness_apply(model, PROBE_AMPLITUDE * state) / PROBE_AMPLITUDE
    rayleigh = float((state * applied).sum().item())
    residual = float(
        torch.linalg.vector_norm(applied - rayleigh * state).item() / max(abs(rayleigh), 1.0)
    )
    return {
        "omega_rad_s": math.sqrt(max(rayleigh, 0.0) / mass),
        "eigenvalue": rayleigh,
        "residual": residual,
        "iterations": used,
        "eigenvector": state,
    }


# ---------------------------------------------------------------------------
# Closed-form single-mode algebra (independent of the solver)
# ---------------------------------------------------------------------------
def closed_form_step_matrix(dt: float, omega: float, damping_rate: float = 0.0) -> torch.Tensor:
    """Amplification matrix of semi-implicit Euler on ``x'' = -w^2 x - g x'``."""
    d2 = (dt * omega) ** 2
    gamma = dt * damping_rate
    return torch.tensor([[1.0 - d2, dt], [-d2 / dt, 1.0 - gamma]], dtype=torch.float64)


def closed_form_rho(dt: float, omega: float, damping_rate: float = 0.0) -> float:
    """Alias of :func:`tensordem.stability.symplectic_euler_growth_factor`."""
    return stability.symplectic_euler_growth_factor(dt, omega, damping_rate)


def dominant_direction(dt: float, omega: float) -> tuple[float, float]:
    """Unit-norm dominant eigenvector ``(x0, y0)`` of the closed-form step map.

    Starting the engine on this direction makes the mode amplitude grow as
    ``rho**n`` from the first step, so the long run tests the *rate* rather than
    a transient.  Below the threshold the map has no real eigenvector; the real
    part of the complex one is returned (a rotating state, fine for the
    boundedness measurement).
    """
    matrix = closed_form_step_matrix(dt, omega)
    values, vectors = torch.linalg.eig(matrix)
    pick = int(torch.argmax(values.abs()).item())
    vector = vectors[:, pick]
    if float(vector.imag.abs().max()) > 1.0e-14:
        vector = vector.real if float(vector.real.abs().sum()) > 1.0e-14 else vector.imag
    vector = vector.real / torch.linalg.vector_norm(vector.real)
    return float(vector[0]), float(vector[1])


def invariant_form(dt: float, omega: float) -> torch.Tensor:
    """Exact quadratic invariant ``P`` of the step map: ``A^T P A = P``.

    For the undamped oscillator below the threshold ``P`` is positive definite,
    so ``[x y] P [x y]^T`` is conserved to round-off while the *physical*
    energy oscillates by an O(1) fraction as ``dt -> dt_crit``.  ``p22`` is
    normalised to 1 to fix the scale, which leaves the 2x2 system solved here;
    at or above the threshold no such form exists and the caller must not use
    it (the matrix becomes indefinite and the solve is meaningless).
    """
    matrix = closed_form_step_matrix(dt, omega)
    a11, a12 = float(matrix[0, 0]), float(matrix[0, 1])
    a21, a22 = float(matrix[1, 0]), float(matrix[1, 1])
    if abs(a11 + a22) >= 2.0 - 1.0e-12:
        raise ValueError("no positive definite invariant exists at or above the threshold")
    # A^T P A - P = 0 for symmetric P = [[p11, p12], [p12, 1]]: three equations
    # reduce to two unknowns once p22 = 1 is fixed.
    coefficients = [
        [a11 * a11 - 1.0, 2.0 * a11 * a21],
        [a11 * a12, a11 * a22 + a21 * a12 - 1.0],
        [a12 * a12, 2.0 * a12 * a22],
    ]
    offsets = [a21 * a21, a21 * a22, a22 * a22 - 1.0]
    lhs = torch.tensor(coefficients, dtype=torch.float64)
    rhs = -torch.tensor(offsets, dtype=torch.float64)
    solution, *_ = torch.linalg.lstsq(lhs, rhs)
    residual = float((lhs @ solution - rhs).abs().max().item())
    if residual > 1.0e-9:
        raise ValueError("invariant system is inconsistent; dt is at or above the threshold")
    p11, p12 = float(solution[0]), float(solution[1])
    return torch.tensor([[p11, p12], [p12, 1.0]], dtype=torch.float64)


def quadratic(form: torch.Tensor, x: float, y: float) -> float:
    """``[x y] form [x y]^T``."""
    return float(form[0, 0]) * x * x + 2.0 * float(form[0, 1]) * x * y + float(form[1, 1]) * y * y


# ---------------------------------------------------------------------------
# 4. Measured per-step amplification in the top-mode subspace
# ---------------------------------------------------------------------------
def _project(state: torch.Tensor, unit_mode: torch.Tensor) -> float:
    return float((state * unit_mode).sum().item())


def engine_amplification_matrix(
    model: IceDEM3D,
    unit_mode: torch.Tensor,
    dt: float,
    *,
    amplitude: float = PROBE_AMPLITUDE,
) -> dict[str, Any]:
    """2x2 semi-implicit Euler map of the engine in the ``unit_mode`` subspace.

    Two engine runs with the basis states ``(x=1, y=0)`` and ``(x=0, y=1)``
    supply the four entries ``a11, a21, a12, a22``; the spectral radius follows
    from ``trace`` and ``det`` exactly as in the closed-form derivation.
    """

    def one_step(x: float, y: float) -> tuple[float, float]:
        # Same configuration; the audited step enters through the supported path
        # (``allow_dt_above_recommended`` belongs to the reference config).
        probe = IceDEM3D(replace(model.config, dt=float(dt)))
        probe.positions = probe.initial_positions + amplitude * x * unit_mode
        probe.velocities = amplitude * y * unit_mode.clone()
        probe.step()
        moved = probe.positions - probe.initial_positions
        return (_project(moved, unit_mode) / amplitude, _project(probe.velocities, unit_mode) / amplitude)

    a11, a21 = one_step(1.0, 0.0)
    a12, a22 = one_step(0.0, 1.0)
    trace = a11 + a22
    det = a11 * a22 - a12 * a21
    discriminant = trace * trace - 4.0 * det
    if discriminant >= 0.0:
        root = math.sqrt(discriminant)
        rho = max(abs(0.5 * (trace + root)), abs(0.5 * (trace - root)))
    else:
        rho = math.sqrt(abs(det))
    return {
        "dt_s": float(dt),
        "matrix": [[a11, a12], [a21, a22]],
        "trace": trace,
        "det": det,
        "rho_measured": rho,
        "grows": bool(rho > 1.0 + MARGINAL_TOLERANCE),
    }


# ---------------------------------------------------------------------------
# 5. Sweep across the threshold and independent location of the crossing
# ---------------------------------------------------------------------------
DEFAULT_FRACTIONS: tuple[float, ...] = (
    0.10, 0.50, 0.90, 0.99, 0.999, 1.001, 1.01, 1.05, 1.20, 1.50,
)


def threshold_sweep(
    model: IceDEM3D | None = None,
    *,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    analytic: AnalyticMode | None = None,
) -> dict[str, Any]:
    """``rho`` measured by the engine versus the closed form across ``dt/dt_crit``."""
    model = model or reference_model()
    analytic = analytic or top_mode_analytic(model)
    engine = engine_rayleigh_omega(model, analytic.eigenvector)
    unit_mode = analytic.eigenvector / torch.linalg.vector_norm(analytic.eigenvector)
    critical = stability.elastic_timestep_limit(analytic.omega_rad_s)
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        dt = fraction * critical
        measured = engine_amplification_matrix(model, unit_mode, dt)
        reference = closed_form_rho(dt, analytic.omega_rad_s)
        rows.append({
            "dt_over_critical": float(fraction),
            "dt_s": float(dt),
            "rho_measured": measured["rho_measured"],
            "rho_closed_form": reference,
            "relative_error": abs(measured["rho_measured"] - reference) / max(reference, 1e-30),
            "grows": measured["grows"],
            "matrix": measured["matrix"],
            "timestep_guard_would_refuse": bool(dt > model.config.recommended_dt),
        })
    crossing = bisect_critical_step(model, unit_mode, analytic.omega_rad_s)
    return {
        "protocol": PROTOCOL,
        "n_particles": int(len(model.positions)),
        "n_bonds": int(len(model.pairs)),
        "mass_kg": float(model.config.mass),
        "bond_stiffness_N_per_m": float(model.config.bond_stiffness),
        "analytic_omega_rad_s": analytic.omega_rad_s,
        "analytic_eigenvalue_gap": analytic.eigenvalue_gap,
        "engine_omega_rad_s": engine["omega_rad_s"],
        "engine_omega_relative_error": abs(engine["omega_rad_s"] - analytic.omega_rad_s) / analytic.omega_rad_s,
        "engine_mode_residual": engine["residual"],
        "gershgorin_omega_rad_s": stability.stability_report(model)["omega_upper_rad_s"],
        "critical_dt_analytic_s": critical,
        "engine_recommended_dt_s": float(model.config.recommended_dt),
        "marginal_tolerance": MARGINAL_TOLERANCE,
        "rows": rows,
        "crossing": crossing,
    }


def measured_rho(model: IceDEM3D, unit_mode: torch.Tensor, dt: float) -> float:
    """Engine spectral radius at ``dt`` (see :func:`engine_amplification_matrix`)."""
    return engine_amplification_matrix(model, unit_mode, dt)["rho_measured"]


def bisect_critical_step(
    model: IceDEM3D,
    unit_mode: torch.Tensor,
    omega: float,
    *,
    lower_fraction: float = 0.5,
    upper_fraction: float = 1.5,
    tolerance: float = 1.0e-9,
    max_iterations: int = 60,
) -> dict[str, Any]:
    """Locate the ``rho = 1`` crossing from engine measurements only.

    Bisection on ``sign(rho_measured - (1 + MARGINAL_TOLERANCE))``.  No analytic
    number enters the search -- only the bracket -- and the answer is compared
    with the closed-form ``2 / omega`` afterwards.
    """
    critical = stability.elastic_timestep_limit(omega)
    low = lower_fraction * critical
    high = upper_fraction * critical
    if measured_rho(model, unit_mode, low) > 1.0 + MARGINAL_TOLERANCE:
        raise ValueError("lower bracket is already unstable")
    if measured_rho(model, unit_mode, high) <= 1.0 + MARGINAL_TOLERANCE:
        raise ValueError("upper bracket is still stable")
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        middle = 0.5 * (low + high)
        if measured_rho(model, unit_mode, middle) > 1.0 + MARGINAL_TOLERANCE:
            high = middle
        else:
            low = middle
        if high - low <= tolerance * critical:
            break
    crossing = 0.5 * (low + high)
    return {
        "dt_crossing_measured_s": crossing,
        "dt_crit_closed_form_s": critical,
        "relative_error": abs(crossing - critical) / critical,
        "bracket_width_s": high - low,
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# 6. Long runs on either side of the threshold
# ---------------------------------------------------------------------------
def long_run_growth(
    dt: float,
    mode: torch.Tensor,
    omega: float,
    *,
    steps: int = 600,
    amplitude: float = PROBE_AMPLITUDE,
    amplitude_limit: float = 1.0e3,
    **config_overrides: Any,
) -> dict[str, Any]:
    """Integrate the engine in the top mode and report the amplitude history.

    The state is projected on the mode each step, so the recorded growth is the
    engine's own amplification of that mode rather than a norm over a mixture.
    The run is truncated once the mode amplitude exceeds ``amplitude_limit``,
    which keeps every reported number inside the linear regime the threshold
    statement refers to.
    """
    unit_mode = mode / torch.linalg.vector_norm(mode)
    model = IceDEM3D(replace(reference_config(**config_overrides), dt=float(dt)))
    x0, y0 = dominant_direction(dt, omega)
    model.positions = model.initial_positions + amplitude * x0 * unit_mode
    model.velocities = amplitude * y0 * unit_mode.clone()
    stable = not (closed_form_rho(dt, omega) > 1.0 + MARGINAL_TOLERANCE)
    form = invariant_form(dt, omega) if stable else None
    invariant0 = quadratic(form, x0, y0) if form is not None else float("nan")
    history: list[float] = []
    invariants: list[float] = []
    peak = 0.0
    for _ in range(steps):
        model.step()
        moved = model.positions - model.initial_positions
        x = _project(moved, unit_mode) / amplitude
        y = _project(model.velocities, unit_mode) / amplitude
        history.append(math.hypot(x, y / max(omega, 1e-30)))
        if form is not None:
            invariants.append(quadratic(form, x, y) / invariant0)
        peak = max(peak, abs(x))
        if not math.isfinite(history[-1]) or peak > amplitude_limit:
            break
    reference = closed_form_rho(dt, omega)
    count = len(history)
    # Per-step growth of the projection for the unstable branch: with the
    # analytic dominant direction the projection is C * rho**n from step one.
    if count >= 2 and abs(x0) > 0.0:
        final_x = _project(model.positions - model.initial_positions, unit_mode) / amplitude
        growth_rate = (abs(final_x) / abs(x0)) ** (1.0 / count) if abs(final_x) > 0.0 else float("nan")
    else:
        final_x, growth_rate = float("nan"), float("nan")
    return {
        "dt_s": float(dt),
        "steps": count,
        "rho_closed_form": reference,
        "grows": bool(reference > 1.0 + MARGINAL_TOLERANCE),
        "amplitude_start": history[0] if history else float("nan"),
        "amplitude_peak_abs_projection": peak,
        "predicted_amplitude_after_steps": abs(x0) * reference ** count,
        "measured_amplitude_final": abs(final_x),
        "growth_per_step_measured": growth_rate,
        "growth_per_step_relative_error": (
            abs(growth_rate - reference) / reference if reference > 0 and math.isfinite(growth_rate) else float("nan")
        ),
        "invariant_peak": max(invariants) if invariants else float("nan"),
        "invariant_min": min(invariants) if invariants else float("nan"),
        "invariant_relative_drift": (
            max(abs(max(invariants) - 1.0), abs(min(invariants) - 1.0)) if invariants else float("nan")
        ),
        "history": history,
        "invariant_history": invariants,
    }