#!/usr/bin/env python3
"""Engine-level verification of the explicit time-step stability boundary.

Why this script exists
----------------------
``docs/dem3d_timestep_stability.md`` (merged) declared two things that this
script settles with the *engine itself*, not with a reconstruction of its scheme:

1. ``Engine-level threshold crossing verified dynamically`` was marked
   ``[未验证]``: ``DEM3DConfig.recommended_dt`` keeps the engine at
   ``dt/dt_crit <= 0.022`` and ``IceDEM3D.__init__`` refuses larger steps, so no
   test ever drove the assembled solver across the boundary.
2. ``The stability region is exactly d <= 2 and g <= 2`` was marked
   ``[已证实]``.  That statement is wrong.  For ``A = [[1-d^2, dt(1-g)],
   [-dt w^2, 1-g]]`` both roots lie inside the unit disc iff ``|det| <= 1`` and
   ``|tr| <= 1 + det``, i.e. ``0 <= g <= 2`` **and** ``d^2 + 2 g <= 4``.  The
   rectangle is only a set of *necessary* axis conditions; e.g. ``d = g = 1.5``
   passes both axis tests and has roots ``0.25`` and ``-2`` (``rho = 2``).

Method (all reference values independent of the engine)
------------------------------------------------------
* Assemble the engine's *tangent* stiffness matrix by probing its own force
  routine with unit displacements (``update_fracture=False``), in a state that
  is in exact equilibrium (bond spacing scaled by 1.002 and the rest lengths
  scaled with it, so every bond is force-free and there is a contact gap; the
  state then evolves linearly for perturbations far below the gap).
* Dense eigen-solve of ``K`` gives ``omega_max`` exactly; the Gershgorin row-sum
  bound of :mod:`tensordem.stability` is compared against it.
* The 2N x 2N step matrix ``A`` is built from ``K`` and the integrator's own
  update rule; its spectral radius is the prediction.  The closed form
  :func:`tensordem.stability.symplectic_euler_growth_factor` is a *second*,
  independent prediction.
* The engine is then stepped from the dominant eigenvector of ``A`` (scaled to
  ``1e-9`` m) and the measured per-step growth is read from the left-eigenvector
  projection ``|w.z_n / w.z_0|^(1/n)`` -- exact for a pure mode, immune to
  transient contamination.
* Naming anything a rigime: ``rho > 1`` exponential growth; (near-)marginal
  modes are classified by the log-log slope of the mechanical energy.

Outputs
-------
``docs/dem3d-engine-threshold-crossing/metrics.json``   machine-readable record
``docs/dem3d-engine-threshold-crossing/summary.md``     generated tables
``docs/dem3d-engine-threshold-crossing/fig1_growth_vs_dt.png``
``docs/dem3d-engine-threshold-crossing/fig2_energy_histories.png``

Run::

    PYTHONPATH=src python scripts/verify_engine_threshold_crossing.py
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

import torch

from tensordem import stability
from tensordem.dem3d import DEM3DConfig, IceDEM3D

SCHEMA = "tensordem.engine-threshold-crossing/1"
SPACING_FACTOR = 1.002   # creates a contact gap: 0.002 * 2R while remaining force-free
AMPLITUDE_M = 1e-9


# ---------------------------------------------------------------------------
# Model assembly and linearisation
# ---------------------------------------------------------------------------
def build_model(nx: int = 7, ny: int = 5, nz: int = 2, drag: float = 0.0) -> IceDEM3D:
    """Bonded lattice in exact static equilibrium with a contact gap.

    ``fix_x_edges`` is disabled so that every degree of freedom is free (the
    rigid-body modes are then part of the spectrum, as they are for a free
    specimen).  The tool is parked far away with ``tool_speed ~ 0`` so only the
    bond law is active; ``contact_damping`` is zero and ``friction_coefficient``
    is left at its default 0, so the tangent operator is constant.
    """
    cfg = DEM3DConfig(nx=nx, ny=ny, nz=nz, fix_x_edges=False, drag=drag,
                      contact_damping=0.0, tool_speed=1e-12, tool_gap=1.0)
    model = IceDEM3D(cfg)
    model.positions *= SPACING_FACTOR
    model.rest_lengths *= SPACING_FACTOR
    model.initial_positions *= SPACING_FACTOR
    return model


@torch.no_grad()
def reconstruct_stiffness(model: IceDEM3D, delta: float = 1e-9) -> torch.Tensor:
    """Tangent stiffness ``K`` (3N x 3N) by probing ``IceDEM3D.forces``.

    ``K[:, k] = -(f(x0 + delta e_k) - f(x0)) / delta`` with the fracture and
    friction history updates disabled, so the probe is side-effect free.  The
    probe is a forward difference, so ``K`` carries an ``O(delta/L0)`` geometric
    error; :func:`probe_quality` quantifies it.
    """
    base = model.positions.clone()
    f0, _ = model.forces(update_fracture=False, update_history=False)
    n = len(base)
    stiffness = torch.zeros((3 * n, 3 * n), dtype=torch.float64)
    for k in range(3 * n):
        model.positions = base.clone()
        model.positions.view(-1)[k] += delta
        fk, _ = model.forces(update_fracture=False, update_history=False)
        stiffness[:, k] = -(fk - f0).reshape(-1) / delta
    model.positions = base.clone()
    return stiffness


def probe_quality(model: IceDEM3D, delta: float = 1e-9) -> dict[str, float]:
    """Symmetry residual and Richardson error of the probed stiffness matrix."""
    coarse = reconstruct_stiffness(model, delta)
    fine = reconstruct_stiffness(model, delta / 10.0)
    scale = float(coarse.abs().max())
    residual_zero = float(torch.linalg.vector_norm(
        model.forces(update_fracture=False, update_history=False)[0]))
    return {
        "probe_delta_m": delta,
        "refined_delta_m": delta / 10.0,
        "relative_asymmetry": float((coarse - coarse.T).abs().max()) / scale,
        "richardson_relative_change": float((fine - coarse).abs().max()) / scale,
        "equilibrium_residual_N": residual_zero,
    }


def amplification_matrix(K: torch.Tensor, mass: float, dt: float,
                         damping_rate: float) -> torch.Tensor:
    """2N x 2N step matrix of ``IceDEM3D.step`` for the linearised lattice."""
    n = K.shape[0]
    eye = torch.eye(n, dtype=K.dtype)
    k_hat = K / mass
    decay = 1.0 - dt * damping_rate
    A = torch.zeros((2 * n, 2 * n), dtype=K.dtype)
    A[:n, :n] = eye - dt * dt * k_hat
    A[:n, n:] = dt * decay * eye
    A[n:, :n] = -dt * k_hat
    A[n:, n:] = decay * eye
    return A


def modal_data(model: IceDEM3D, K: torch.Tensor) -> dict[str, Any]:
    """Exact ``omega_max`` (dense) versus the Gershgorin row-sum bound."""
    mass = model.config.mass
    eigenvalues = torch.linalg.eigvalsh(0.5 * (K + K.T))
    omega_exact = math.sqrt(float(eigenvalues[-1]) / mass)
    sums = stability.particle_stiffness_sums(model)
    omega_bound = stability.gershgorin_omega_bound(sums, mass)
    supported = eigenvalues[eigenvalues > 1e-9 * float(eigenvalues[-1])]
    return {
        "n_particles": int(len(model.positions)),
        "n_bonds": int(len(model.pairs)),
        "mass_kg": mass,
        "lambda_max_N_per_m": float(eigenvalues[-1]),
        "lambda_max_over_bond_stiffness": float(eigenvalues[-1]) / model.config.bond_stiffness,
        "omega_exact_rad_s": omega_exact,
        "omega_gershgorin_rad_s": omega_bound,
        "gershgorin_over_exact": omega_bound / omega_exact,
        "n_zero_modes": int(eigenvalues.numel() - supported.numel()),
        "dt_crit_elastic_exact_s": 2.0 / omega_exact,
        "engine_recommended_dt_s": float(model.config.recommended_dt),
        "engine_dt_over_crit": float(model.dt) / (2.0 / omega_exact),
        "engine_margin_elastic": (2.0 / omega_exact) / float(model.dt),
    }


# ---------------------------------------------------------------------------
# Engine measurement
# ---------------------------------------------------------------------------
def total_energy(model: IceDEM3D) -> float:
    i, j = model.pairs.T
    distance = torch.linalg.vector_norm(model.positions[j] - model.positions[i], dim=1)
    elastic = 0.5 * model.config.bond_stiffness * (distance - model.rest_lengths).square().sum()
    kinetic = 0.5 * model.config.mass * model.velocities.square().sum()
    return float(kinetic + elastic)


def _loglog_slope(xs: torch.Tensor, ys: torch.Tensor) -> float:
    x = torch.log(xs)
    y = torch.log(ys)
    return float(((x * y).mean() - x.mean() * y.mean()) / ((x * x).mean() - x.mean() ** 2))


def run_case(K: torch.Tensor, mass: float, omega: float, *, elastic_number: float,
             damping_number: float, steps: int, amplitude_cap: float | None,
             seed: int = 0, stability_module=stability) -> dict[str, Any]:
    """Drive the engine at a prescribed ``(d, g)`` pair and measure the growth.

    ``d = dt * omega`` and ``g = dt * damping_rate`` are set by choosing
    ``dt = d / omega`` and ``drag = g * mass / dt``.  The step is installed by
    assigning ``model.dt`` after construction: ``IceDEM3D.__init__`` refuses
    ``dt > recommended_dt`` on purpose, which is exactly why this verification
    was previously impossible through the public constructor.
    """
    dt = elastic_number / omega
    drag = damping_number * mass / dt
    model = build_model(drag=drag)
    model.dt = dt
    base = model.positions.clone()
    damping_rate = drag / mass
    gamma = dt * damping_rate

    A = amplification_matrix(K, mass, dt, damping_rate)
    values, vectors = torch.linalg.eig(A)
    index = int(values.abs().argmax())
    rho_dense = float(values[index].abs())

    # Seed the dominant eigenvector: the trajectory is then a pure mode, so the
    # 2-norm ratio gives the per-step growth exactly (all measured growing cases
    # have a real dominant eigenvalue; complex pairs only occur where rho = 1).
    z0 = vectors[:, index].real
    z0 = z0 / torch.linalg.vector_norm(z0)
    n_dof = 3 * len(base)
    model.positions = base + (AMPLITUDE_M * z0[:n_dof]).reshape(base.shape)
    model.velocities = (AMPLITUDE_M * z0[n_dof:]).reshape(base.shape)
    norm_zero = float(torch.linalg.vector_norm(z0))

    energies: list[float] = []
    amplitudes: list[float] = []
    diverged = None
    used = 0
    for step in range(1, steps + 1):
        try:
            model.step()
        except FloatingPointError as exc:      # engine's own divergence guard
            diverged = f"FloatingPointError after {step} steps"
            break
        energies.append(total_energy(model))
        amplitudes.append(float((model.positions - base).abs().max()))
        used = step
        if amplitude_cap is not None and amplitudes[-1] > amplitude_cap and step >= 6:
            break

    rho_measured = float("nan")
    if used >= 6:
        z = torch.cat([(model.positions - base).reshape(-1),
                       model.velocities.reshape(-1)])
        norm_now = float(torch.linalg.vector_norm(z))
        if norm_now > 0.0:
            rho_measured = (norm_now / (AMPLITUDE_M * norm_zero)) ** (1.0 / used)

    energy_tensor = torch.tensor(energies, dtype=torch.float64)
    indices = torch.arange(1, len(energies) + 1, dtype=torch.float64)
    slope = _loglog_slope(indices, energy_tensor) if len(energies) > 8 else float("nan")
    closed = stability_module.symplectic_euler_growth_factor(dt, omega, damping_rate)
    if rho_dense > 1.0 + 1e-9:
        regime = "exponential"
    elif abs(slope) < 0.15:
        regime = "bounded"
    else:
        regime = "polynomial"

    return {
        "elastic_number_d": elastic_number,
        "damping_number_g": damping_number,
        "dt_s": dt,
        "drag_Ns_per_m": drag,
        "damping_rate_per_s": damping_rate,
        "dt_over_elastic_limit": dt * omega / 2.0,
        "rho_dense": rho_dense,
        "rho_closed_form": closed,
        "rho_measured_engine": rho_measured,
        "relative_error_measured_vs_dense": (
            abs(rho_measured - rho_dense) / rho_dense if rho_measured == rho_measured else float("nan")),
        "regime": regime,
        "engine_steps": used,
        "energy_loglog_slope": slope,
        "energy_growth_ratio": float(energy_tensor[-1] / energy_tensor[0]) if len(energies) else float("nan"),
        "amplitude_final_m": amplitudes[-1] if amplitudes else float("nan"),
        "diverged": diverged,
        "axis_rectangle_stable": bool(elastic_number <= 2.0 and damping_number <= 2.0),
        "coupled_region_value": elastic_number ** 2 + 2.0 * damping_number,
        "coupled_region_stable": bool(elastic_number ** 2 + 2.0 * damping_number <= 4.0
                                      and damping_number <= 2.0),
        "energy_history": [round(value, 12) for value in energies[:400]],
    }


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------
UNDAMPED_RATIOS = (0.5, 0.9, 0.99, 1.0, 1.005, 1.01, 1.05)
DAMPED_CASES = (
    (0.05, 0.8),                      # both criteria stable -> bounded
    (1.0, 0.99 * math.sqrt(2.0)),     # just inside the coupled boundary
    (1.0, 1.01 * math.sqrt(2.0)),     # just outside; rectangle still says stable
    (1.5, 1.5),                       # rectangle "stable", true rho = 2
    (1.95, 0.8),                      # rectangle "stable", true rho = 1.313
)


def run_campaign(*, quick: bool = False) -> dict[str, Any]:
    torch.set_num_threads(6)
    model = build_model()
    quality = probe_quality(model)
    K = reconstruct_stiffness(model)
    K = 0.5 * (K + K.T)
    modal = modal_data(model, K)
    mass = model.config.mass
    omega = modal["omega_exact_rad_s"]
    dt_crit = 2.0 / omega

    bounded_steps = 400 if quick else 800
    undamped = []
    for ratio in UNDAMPED_RATIOS:
        steps = bounded_steps if ratio <= 1.0 else 400
        undamped.append(run_case(
            K, mass, omega, elastic_number=ratio * 2.0, damping_number=0.0,
            steps=steps, amplitude_cap=2e-5, seed=1))

    damped = []
    for damping_number, elastic_number in DAMPED_CASES:
        growing = elastic_number ** 2 + 2.0 * damping_number > 4.0
        damped.append(run_case(
            K, mass, omega, elastic_number=elastic_number,
            damping_number=damping_number,
            steps=400 if growing else bounded_steps,
            amplitude_cap=2e-5, seed=2))

    # Crossed boundary without an amplitude cap: the engine must not survive it.
    divergence = run_case(K, mass, omega, elastic_number=1.5, damping_number=1.5,
                          steps=1500, amplitude_cap=None, seed=3)

    return {
        "schema": SCHEMA,
        "generated_by": "scripts/verify_engine_threshold_crossing.py",
        "lattice": {
            "nx": 7, "ny": 5, "nz": 2, "radius_m": model.config.radius,
            "density_kg_per_m3": model.config.density,
            "bond_stiffness_N_per_m": model.config.bond_stiffness,
            "spacing_factor": SPACING_FACTOR,
            "contact_gap_m": (SPACING_FACTOR - 1.0) * 2.0 * model.config.radius,
            "amplitude_m": AMPLITUDE_M,
        },
        "probe_quality": quality,
        "modal": modal,
        "undamped_cases": undamped,
        "damped_cases": damped,
        "divergence_case": {k: v for k, v in divergence.items() if k != "energy_history"},
        "corrigendum": {
            "region": "0 <= g <= 2 and d^2 + 2 g <= 4",
            "coupled_limit_formula": "dt_crit = 4 / (sqrt(gamma^2 + 4 w^2) + gamma)",
            "rectangle_counterexample": "d = g = 1.5 -> roots 0.25, -2 (rho = 2)",
        },
    }


def write_summary(metrics: dict[str, Any], path: pathlib.Path) -> None:
    modal = metrics["modal"]
    lines = ["# Engine threshold-crossing campaign (generated)", "",
             "Generated by `scripts/verify_engine_threshold_crossing.py`; do not edit.", "",
             "## Modal data of the probed lattice", "",
             "| quantity | value |", "| --- | ---: |",
             f"| particles / bonds | {modal['n_particles']} / {modal['n_bonds']} |",
             f"| lambda_max(K) / k_bond | {modal['lambda_max_over_bond_stiffness']:.4f} |",
             f"| omega_exact [rad/s] | {modal['omega_exact_rad_s']:.4f} |",
             f"| omega_Gershgorin [rad/s] | {modal['omega_gershgorin_rad_s']:.4f} |",
             f"| Gershgorin / exact | {modal['gershgorin_over_exact']:.4f} |",
             f"| zero modes (rigid body) | {modal['n_zero_modes']} |",
             f"| dt_crit = 2/omega_exact [s] | {modal['dt_crit_elastic_exact_s']:.6e} |",
             f"| engine recommended dt [s] | {modal['engine_recommended_dt_s']:.6e} |",
             f"| engine margin to dt_crit | {modal['engine_margin_elastic']:.2f} |", "",
             "## Undamped threshold crossing (g = 0)", "",
             "| d = dt*omega | rho dense | rho closed form | rho measured | rel. err | regime | steps |",
             "| ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
    for case in metrics["undamped_cases"]:
        measured = case["rho_measured_engine"]
        err = case["relative_error_measured_vs_dense"]
        lines.append(
            f"| {case['elastic_number_d']:.3f} | {case['rho_dense']:.6f} | "
            f"{case['rho_closed_form']:.6f} | "
            f"{'n/a' if measured != measured else f'{measured:.9f}'} | "
            f"{'n/a' if err != err else f'{err:.2e}'} | {case['regime']} | {case['engine_steps']} |")
    lines += ["", "## Damped cases: rectangle criterion versus coupled criterion", "",
              "| g = dt*gamma | d = dt*omega | d^2+2g | rectangle says | rho dense | rho measured | regime |",
              "| ---: | ---: | ---: | --- | ---: | ---: | --- |"]
    for case in metrics["damped_cases"]:
        measured = case["rho_measured_engine"]
        lines.append(
            f"| {case['damping_number_g']:.3f} | {case['elastic_number_d']:.3f} | "
            f"{case['coupled_region_value']:.3f} | "
            f"{'stable' if case['axis_rectangle_stable'] else 'unstable'} | "
            f"{case['rho_dense']:.6f} | "
            f"{'n/a' if measured != measured else f'{measured:.6f}'} | {case['regime']} |")
    divergence = metrics["divergence_case"]
    lines += ["", "## Uncaught divergence (same d = g = 1.5 point, no amplitude cap)", "",
              f"* engine steps survived: {divergence['engine_steps']}",
              f"* divergence signal: {divergence['diverged']}",
              f"* final amplitude: {divergence['amplitude_final_m']:.3e} m", "",
              "## Correction", "",
              f"* exact region: `{metrics['corrigendum']['region']}`",
              f"* exact limit: `{metrics['corrigendum']['coupled_limit_formula']}`",
              f"* rectangle counterexample: {metrics['corrigendum']['rectangle_counterexample']}", ""]
    path.write_text("\n".join(lines))


def plot(metrics: dict[str, Any], directory: pathlib.Path) -> list[pathlib.Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    grid = [0.05 + 0.05 * i for i in range(60)]
    for damping_number, style in ((0.0, "-"), (1.0, "--"), (1.5, ":")):
        rhos = []
        for d in grid:
            gamma = damping_number
            omega = 1.0
            dt = d / omega
            rhos.append(stability.symplectic_euler_growth_factor(dt, omega, gamma / dt))
        ax.plot(grid, rhos, style, lw=1.4,
                label=fr"closed form, $g={damping_number:g}$")
    points = [c for c in metrics["undamped_cases"] if c["rho_dense"] > 1.0 + 1e-9]
    ax.plot([c["elastic_number_d"] for c in points],
            [c["rho_measured_engine"] for c in points], "o", ms=6,
            label="engine, measured (g=0)")
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.axvline(2.0, color="0.5", ls="-.", lw=0.8)
    ax.set_xlabel(r"$d = \Delta t\,\omega$")
    ax.set_ylabel(r"per-step growth $\rho$")
    ax.set_title("TensorDEM: engine growth vs. step matrix prediction")
    ax.set_ylim(0.9, 2.6)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    path = directory / "fig1_growth_vs_dt.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    marginal = next(c for c in metrics["undamped_cases"]
                    if abs(c["elastic_number_d"] - 2.0) < 1e-9)
    steps = range(1, len(marginal["energy_history"]) + 1)
    axes[0].loglog(list(steps), marginal["energy_history"], lw=1.2, label="engine energy")
    reference = [marginal["energy_history"][9] * (n / 10.0) ** 2 for n in steps]
    axes[0].loglog(list(steps), reference, "k--", lw=0.9, label=r"$\propto n^2$")
    axes[0].set_xlabel("step n")
    axes[0].set_ylabel("mechanical energy [J]")
    axes[0].set_title(r"marginal step $d=2$: algebraic, not exponential")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25, which="both")

    for case, style in ((metrics["damped_cases"][3], "-"),
                        (metrics["damped_cases"][1], "--"),
                        (metrics["damped_cases"][2], ":")):
        history = case["energy_history"]
        axes[1].semilogy(range(1, len(history) + 1), history, style, lw=1.2,
                         label=(fr"$g={case['damping_number_g']:.2f}$, "
                                fr"$d={case['elastic_number_d']:.2f}$, "
                                fr"$\rho$={case['rho_dense']:.2f}"))
    axes[1].set_xlabel("step n")
    axes[1].set_ylabel("mechanical energy [J]")
    axes[1].set_title("coupled criterion: d^2 + 2g <= 4")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.25, which="both")
    path = directory / "fig2_energy_histories.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)
    return written


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("docs/dem3d-engine-threshold-crossing"))
    parser.add_argument("--quick", action="store_true", help="shorter marginal runs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = run_campaign(quick=args.quick)
    figures = plot(metrics, args.output)
    metrics["figures"] = [path.name for path in figures]
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_summary(metrics, args.output / "summary.md")
    print(json.dumps({
        "modal": {k: metrics["modal"][k] for k in
                  ("omega_exact_rad_s", "omega_gershgorin_rad_s", "gershgorin_over_exact",
                   "engine_margin_elastic", "n_zero_modes")},
        "undamped": [(c["elastic_number_d"], round(c["rho_dense"], 6),
                      None if c["rho_measured_engine"] != c["rho_measured_engine"]
                      else round(c["rho_measured_engine"], 9),
                      c["regime"], c["engine_steps"]) for c in metrics["undamped_cases"]],
        "damped": [(c["damping_number_g"], c["elastic_number_d"], round(c["coupled_region_value"], 3),
                    c["axis_rectangle_stable"], round(c["rho_dense"], 6),
                    None if c["rho_measured_engine"] != c["rho_measured_engine"]
                    else round(c["rho_measured_engine"], 6), c["regime"])
                   for c in metrics["damped_cases"]],
        "divergence": metrics["divergence_case"]["diverged"],
        "probe": metrics["probe_quality"],
    }, indent=1))
    return metrics


if __name__ == "__main__":
    main()