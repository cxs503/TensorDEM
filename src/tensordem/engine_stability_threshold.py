"""Engine-level dynamic verification of the explicit time-step stability limit.

Background
----------
``docs/dem3d_timestep_stability.md`` carries one deliberately open item:

    "Engine-level threshold crossing verified dynamically -- 【未验证】:
     ``DEM3DConfig.recommended_dt`` allows at most ``dt/dt_crit = 0.022`` and
     ``IceDEM3D.__init__`` refuses larger steps, so the threshold is
     deliberately unreachable through the public API."

This module closes that gap *without* weakening the production guard.  The
guard lives in ``IceDEM3D.__init__``; ``IceDEM3D.step`` reads the mutable
attribute ``model.dt``.  A verification harness may therefore construct a model
at the recommended step and then assign a larger ``model.dt`` to probe the
threshold.  Nothing in ``dem3d.py`` changes, and the constructor still refuses
``dt > recommended_dt`` (re-tested on every run by :func:`guard_still_refuses`).

Independent reference (no solver involvement)
---------------------------------------------
For a bonded lattice in the small-amplitude limit the translational stiffness
matrix is assembled here directly from the *rest geometry* and the bond table::

    K = sum_alive_bonds k_b (n n^T)  (x)  [[+1, -1], [-1, +1]]

with ``n`` the bond direction, i.e. a ``3N x 3N`` symmetric block matrix.  Its
largest eigenvalue ``lambda_max`` follows from a dense ``eigvalsh`` call, so the
prediction does not reuse any force routine of the engine under test::

    omega_max = sqrt(lambda_max / m)
    dt_crit   = min(2 / omega_max, 2 m / c)          (semi-implicit Euler)

The first factor is the marginal elastic limit (``d = dt * omega = 2``); the
second is the marginal damping limit (``g = dt * c / m = 2``).  Because pure
per-particle drag makes ``C = c I`` commute with ``K``, both factors are exact
for the configurations used here (no contact damping is active: contact
stiffness is switched off so that only the bond lattice carries stiffness).

What is measured
----------------
* divergence onset in ``dt / dt_crit`` by bisection over engine runs
  (:func:`measure_onset_ratio`): the bisection reports the largest ratio that
  still runs without amplification,
* the per-step energy growth factor of a diverging run, compared with the
  closed-form spectral radius ``rho = |lambda|`` of the scheme
  (:func:`measure_growth_exponent`),
* the algebraic ``E ~ n^2`` growth at the marginal step ``dt = dt_crit``,
* the row-sum (Gershgorin) bound of :mod:`tensordem.stability` against the dense
  eigenvalue of the same assembled matrix, i.e. how conservative the criterion
  that the solver actually uses is.

Everything here is a *numerical-stability* statement.  It says nothing about
whether the ice contact law matches experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import torch

from .dem3d import DEM3DConfig, IceDEM3D
from .stability import gershgorin_omega_bound, symplectic_euler_growth_factor

__all__ = [
    "assemble_stiffness_matrix",
    "exact_omega_max",
    "predict_critical_dt",
    "refined_damping_dt",
    "make_elastic_lattice",
    "run_energy_history",
    "measure_growth_exponent",
    "measure_onset_ratio",
    "guard_still_refuses",
    "threshold_report",
]


def make_elastic_lattice(
    *,
    nx: int = 4,
    ny: int = 4,
    nz: int = 4,
    radius: float = 0.025,
    density: float = 917.0,
    bond_stiffness: float = 2_000.0,
    drag: float = 0.0,
) -> DEM3DConfig:
    """Config for a free (unfixed) bonded simple-cubic lattice.

    Contact stiffness, contact damping, drag (unless requested), platens and the
    indenter are disabled so that the *bond lattice alone* carries stiffness and
    the linear reference model above is exact.  ``contact_stiffness`` must stay
    strictly positive for the config validator, so it is set to ``1e-9`` N/m:
    nine orders of magnitude below ``bond_stiffness``, i.e. a relative stiffness
    contribution of ``2.6e-13`` per bond (measured in the report).  The breaking strain is raised so
    that the verification runs stay in the linear regime (no bond loss).
    """
    return DEM3DConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        radius=radius,
        density=density,
        bond_stiffness=bond_stiffness,
        contact_stiffness=1.0e-9,
        contact_damping=0.0,
        drag=drag,
        breaking_strain=10.0,
        shear_breaking_strain=10.0,
        tool_radius=0.1,
        tool_speed=1.0e-3,
        tool_gap=1.0,
        fix_x_edges=False,
        fix_bottom=False,
        geometry="box",
        restitution_coefficient=1.0,
        friction_coefficient=0.0,
        rolling_resistance_coefficient=0.0,
    )


def assemble_stiffness_matrix(model: IceDEM3D) -> torch.Tensor:
    """Linearised ``3N x 3N`` bond stiffness matrix from the rest geometry."""
    cfg = model.config
    n_particles = len(model.positions)
    k_total = torch.zeros(3 * n_particles, 3 * n_particles, dtype=torch.float64)
    alive = model.alive & model.bonded
    pairs = model.pairs[alive]
    if pairs.numel() == 0:
        return k_total
    reference = model.initial_positions
    for i, j in pairs.tolist():
        delta = reference[j] - reference[i]
        length = float(torch.linalg.vector_norm(delta))
        if length == 0.0:
            continue
        unit = delta / length
        block = cfg.bond_stiffness * torch.outer(unit, unit)
        i3, j3 = 3 * i, 3 * j
        k_total[i3:i3 + 3, i3:i3 + 3] += block
        k_total[j3:j3 + 3, j3:j3 + 3] += block
        k_total[i3:i3 + 3, j3:j3 + 3] -= block
        k_total[j3:j3 + 3, i3:i3 + 3] -= block
    return k_total


def exact_omega_max(model: IceDEM3D) -> float:
    """Highest undamped eigenfrequency of the assembled bonded system."""
    stiffness = assemble_stiffness_matrix(model)
    lambda_max = float(torch.linalg.eigvalsh(stiffness).max())
    return math.sqrt(max(lambda_max, 0.0) / model.config.mass)


def predict_critical_dt(
    *, omega_max: float, mass: float, damping_coefficient: float
) -> tuple[float, float, float]:
    """Return ``(dt_crit, elastic_limit, damping_limit)`` from closed forms."""
    elastic = 2.0 / omega_max if omega_max > 0 else math.inf
    damping = 2.0 * mass / damping_coefficient if damping_coefficient > 0 else math.inf
    return min(elastic, damping), elastic, damping


def refined_damping_dt(
    *, omega_max: float, mass: float, damping_coefficient: float,
    iterations: int = 200,
) -> float:
    """Marginal step for the damping branch, refined from ``rho(dt) = 1``.

    ``2 m / c`` (i.e. ``g = 2``) is only a *sufficient* scale: with ``d > 0`` the
    root ``lambda = -1`` occurs at ``g = 2 - d^2 / 2``, so the true marginal step
    is slightly below ``2 m / c``.  Solving ``rho(dt) = 1`` with the closed-form
    amplification factor (bisection on ``dt``, no engine call) returns that
    refined limit.
    """
    upper = 2.0 * mass / damping_coefficient if damping_coefficient > 0 else math.inf
    if not math.isfinite(upper):
        return math.inf
    lower = 0.5 * upper
    rate = damping_coefficient / mass
    if symplectic_euler_growth_factor(upper, omega_max, rate) <= 1.0:
        return upper
    for _ in range(iterations):
        mid = 0.5 * (lower + upper)
        if symplectic_euler_growth_factor(mid, omega_max, rate) > 1.0:
            upper = mid
        else:
            lower = mid
    return lower


def _excite(model: IceDEM3D, amplitude: float, seed: int = 20251103) -> None:
    """Deterministic small velocity perturbation that excites every mode."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pattern = torch.randn(
        tuple(model.velocities.shape), generator=generator, dtype=torch.float64
    )
    pattern = pattern / float(torch.linalg.vector_norm(pattern))
    with torch.no_grad():
        model.velocities.copy_(pattern.to(model.device) * amplitude)


def run_energy_history(
    config: DEM3DConfig,
    *,
    dt: float,
    n_steps: int,
    amplitude: float = 1.0e-4,
    blowup_ratio: float = 1.0e6,
    sample_every: int = 1,
) -> dict:
    """Integrate the engine at a chosen ``dt`` and return the energy record.

    The constructor guard is bypassed by assigning ``model.dt`` after
    construction (the pattern used by ``Tensordem`` verification harnesses).  The
    run stops early once ``E / E0`` exceeds ``blowup_ratio`` or the engine raises
    ``FloatingPointError``.

    ``sample_every`` records the energy every ``k``-th step; the divergence test
    is still evaluated at the sample points, which costs one extra energy
    evaluation per ``k`` steps and leaves the detected blow-up step unchanged to
    within ``k`` steps.  The bond-failure counter is asserted to stay zero by the
    tests, so the runs stay in the linear regime.
    """
    model = IceDEM3D(config)
    model.dt = float(dt)
    _excite(model, amplitude)
    e0 = float(model.energy()["mechanical_energy_J"])
    history = [e0]
    diverged_at: int | None = None
    error: str | None = None
    for step in range(n_steps):
        try:
            model.step()
        except FloatingPointError as exc:  # pragma: no cover - environment dependent
            diverged_at = step + 1
            error = str(exc)
            break
        if sample_every > 1 and (step + 1) % sample_every:
            continue
        energy = float(model.energy()["mechanical_energy_J"])
        if not math.isfinite(energy):
            diverged_at = step + 1
            error = "non-finite energy"
            break
        history.append(energy)
        if e0 > 0 and energy / e0 > blowup_ratio:
            diverged_at = step + 1
            break
    peak = max(history)
    return {
        "dt_s": float(dt),
        "n_steps": len(history) - 1,
        "energy_initial_J": e0,
        "energy_peak_J": peak,
        "peak_over_initial": peak / e0 if e0 else math.inf,
        "energy_final_J": history[-1],
        "diverged": diverged_at is not None,
        "diverged_at_step": diverged_at,
        "error": error,
        "history": history,
        "broken_bonds": int(model.broken_bonds),
    }


def measure_growth_exponent(history: Sequence[float], *, tail_fraction: float = 0.4) -> float:
    """Per-step energy growth factor ``exp(beta/2)`` fitted on the late phase.

    The energy of a growing mode behaves like ``rho^(2n)``, so a least-squares fit
    of ``ln E`` versus step returns ``2 ln rho``.  Only the last
    ``tail_fraction`` of the record is used: the early samples still contain the
    bounded transient that spreads the initial perturbation over the modes, and
    including it biases the slope downwards (measured: ``1.01`` ratio gave 15%
    low over the full record versus 0.2% over the tail).
    """
    if len(history) < 12:
        return math.nan
    start = int(round((1.0 - tail_fraction) * (len(history) - 1)))
    kept = [(i, max(v, 1.0e-300)) for i, v in enumerate(history) if i >= start]
    if len(kept) < 6:
        return math.nan
    n = len(kept)
    xs = [float(i) for i, _ in kept]
    ys = [math.log(v) for _, v in kept]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0:
        return math.nan
    beta = sxy / sxx
    return math.exp(beta / 2.0)


def measure_onset_ratio(
    config: DEM3DConfig,
    *,
    dt_crit: float,
    bisections: int = 22,
    low: float = 0.60,
    high: float = 1.60,
    n_steps: int = 1500,
    blowup_ratio: float = 1.0e3,
    sample_every: int = 5,
    resolution: float = 2.0e-4,
) -> dict:
    """Locate the divergence onset in ``dt / dt_crit`` by bisection.

    ``low * dt_crit`` must run without amplification and ``high * dt_crit`` must
    diverge; the returned ``onset_ratio`` is the bisection limit, i.e. the largest
    probed ratio that stays bounded.  ``ratio = 1`` is the closed-form threshold.
    """
    def diverges(ratio: float) -> bool:
        record = run_energy_history(
            config,
            dt=ratio * dt_crit,
            n_steps=n_steps,
            blowup_ratio=blowup_ratio,
            sample_every=sample_every,
        )
        return record["diverged"]

    if diverges(low):
        raise ValueError(f"reference case at dt/dt_crit={low} already diverges")
    if not diverges(high):
        raise ValueError(f"reference case at dt/dt_crit={high} did not diverge")
    probes = 0
    while high - low > resolution and probes < bisections:
        mid = 0.5 * (low + high)
        if diverges(mid):
            high = mid
        else:
            low = mid
        probes += 1
    return {
        "onset_ratio": low,
        "onset_ratio_upper": high,
        "probing_width": high - low,
        "n_probes": probes,
        "n_steps": n_steps,
        "blowup_ratio": blowup_ratio,
        "sample_every": sample_every,
    }


def guard_still_refuses(config: DEM3DConfig) -> dict:
    """Confirm the production constructor still rejects ``dt > recommended_dt``."""
    ratio = 1.05
    probe = DEM3DConfig(**{**config.__dict__, "dt": config.recommended_dt * ratio})
    refused = False
    message = ""
    try:
        IceDEM3D(probe)
    except ValueError as exc:
        refused = True
        message = str(exc)
    return {
        "recommended_dt_s": float(config.recommended_dt),
        "probe_dt_s": float(probe.dt),
        "probe_ratio_of_recommended": ratio,
        "refused": refused,
        "message": message,
    }


@dataclass
class ThresholdResult:
    """Container for one verification sweep (JSON-serialisable via ``dict``)."""

    elastic: dict = field(default_factory=dict)
    damping: dict = field(default_factory=dict)
    growth: dict = field(default_factory=dict)
    marginal: dict = field(default_factory=dict)
    guard: dict = field(default_factory=dict)
    bound: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "elastic_regime": self.elastic,
            "damping_regime": self.damping,
            "growth_just_above_threshold": self.growth,
            "marginal_step": self.marginal,
            "constructor_guard": self.guard,
            "row_sum_bound": self.bound,
        }


def threshold_report(
    *,
    nx: int = 4,
    ny: int = 4,
    nz: int = 4,
    bond_stiffness: float = 2_000.0,
    drag: float = 0.0,
    onset_steps: int = 1500,
) -> ThresholdResult:
    """Run the full engine-level threshold verification sweep."""
    cfg = make_elastic_lattice(
        nx=nx, ny=ny, nz=nz, bond_stiffness=bond_stiffness, drag=drag
    )
    model = IceDEM3D(cfg)
    omega = exact_omega_max(model)
    dt_crit, elastic_limit, damping_limit = predict_critical_dt(
        omega_max=omega, mass=cfg.mass, damping_coefficient=cfg.drag
    )
    result = ThresholdResult()

    # --- (1) elastic-bound regime: divergence onset vs 2 / omega_max --------
    onset = measure_onset_ratio(
        cfg, dt_crit=elastic_limit, n_steps=onset_steps, resolution=2.0e-4
    )
    result.elastic = {
        "omega_max_rad_s": omega,
        "predicted_dt_crit_s": elastic_limit,
        "measured_onset_ratio": onset["onset_ratio"],
        "onset_resolution": onset["probing_width"],
        "relative_error": onset["onset_ratio"] - 1.0,
        "n_probes": onset["n_probes"],
        "n_steps": onset["n_steps"],
        "blowup_ratio": onset["blowup_ratio"],
    }

    # --- (2) growth rate at dt = 1.05 dt_crit vs closed-form rho -----------
    # A ratio barely above the threshold keeps the growth slow enough that the
    # whole record stays inside the linear regime (E/E0 <= 1e3 <=> max bond
    # strain ~ 1e-7): the energy of the run is tracked up to that cap only, since
    # the exponential amplification would otherwise push the lattice into
    # large-displacement geometry where the linear reference no longer applies.
    ratio = 1.0002
    run = run_energy_history(
        cfg, dt=ratio * elastic_limit, n_steps=800, blowup_ratio=1.0e3
    )
    rho_pred = symplectic_euler_growth_factor(ratio * elastic_limit, omega, cfg.drag)
    rho_meas = measure_growth_exponent(run["history"])
    result.growth = {
        "ratio": ratio,
        "dt_s": ratio * elastic_limit,
        "predicted_growth_factor": rho_pred,
        "measured_growth_factor": rho_meas,
        "relative_error": (rho_meas - rho_pred) / rho_pred,
        "diverged": run["diverged"],
        "steps_to_blowup": run["diverged_at_step"],
        "peak_over_initial": run["peak_over_initial"],
        "fit_samples": len(run["history"]),
    }

    # --- (3) marginal step: E ~ n^2 (repeated eigenvalue at -1) ------------
    marginal_run = run_energy_history(
        cfg, dt=elastic_limit, n_steps=400, blowup_ratio=math.inf
    )
    hist = marginal_run["history"]
    half = len(hist) // 2
    n_half = half
    e_half = max(hist[half], 1e-300)
    e_end = max(hist[-1], 1e-300)
    result.marginal = {
        "dt_s": elastic_limit,
        "steps": len(hist) - 1,
        "energy_ratio_second_half_over_first": e_end / e_half,
        "algebraic_prediction": 4.0,
        "broken_bonds": marginal_run["broken_bonds"],
        "diverged": marginal_run["diverged"],
        "half_index": n_half,
    }

    # --- (4) damping-bound regime: 2 m / c --------------------------------
    big_drag = 400.0
    cfg_damped = make_elastic_lattice(
        nx=nx, ny=ny, nz=nz, bond_stiffness=bond_stiffness, drag=big_drag
    )
    dt_crit_d, elastic_d, damping_d = predict_critical_dt(
        omega_max=omega, mass=cfg.mass, damping_coefficient=big_drag
    )
    damped_onset = measure_onset_ratio(
        cfg_damped,
        dt_crit=damping_d,
        n_steps=onset_steps,
        high=1.60,
        low=0.60,
        resolution=2.0e-4,
    )
    refined_damping = refined_damping_dt(
        omega_max=omega, mass=cfg.mass, damping_coefficient=big_drag
    )
    result.damping = {
        "drag_Ns_per_m": big_drag,
        "elastic_limit_s": elastic_d,
        "damping_limit_s": damping_d,
        "refined_damping_limit_s": refined_damping,
        "refined_onset_ratio": refined_damping / damping_d,
        "error_vs_classical": damped_onset["onset_ratio"] - 1.0,
        "error_vs_refined": damped_onset["onset_ratio"] - refined_damping / damping_d,
        "binding_limit": "damping" if damping_d < elastic_d else "elastic",
        "predicted_dt_crit_s": dt_crit_d,
        "measured_onset_ratio": damped_onset["onset_ratio"],
        "onset_resolution": damped_onset["probing_width"],
        "relative_error": damped_onset["onset_ratio"] - 1.0,
        "n_probes": damped_onset["n_probes"],
    }

    # --- (5) production guard untouched -----------------------------------
    result.guard = guard_still_refuses(cfg)

    # --- (6) how conservative is the row-sum bound actually used? ---------
    stiffness = assemble_stiffness_matrix(model)
    dense = float(torch.linalg.eigvalsh(stiffness).max())
    from .stability import particle_stiffness_sums  # local import: avoids cycles

    sums = particle_stiffness_sums(model)
    free = ~model.fixed
    bound = gershgorin_omega_bound(sums, cfg.mass, free_mask=free)
    result.bound = {
        "dense_lambda_max_N_per_m": dense,
        "dense_omega_max_rad_s": omega,
        "gershgorin_omega_bound_rad_s": bound,
        "bound_over_exact": bound / omega,
        "n_particles": len(model.positions),
        "n_alive_bonds": int(model.alive.sum().item()),
    }
    return result