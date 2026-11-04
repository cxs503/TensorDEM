"""Engine-level audit of the explicit time-step stability threshold.

Closes the row ``Engine-level threshold crossing verified dynamically
【未验证】`` of ``docs/dem3d_timestep_stability.md``: the solver is driven across
its own CFL-type limit and the measured growth is compared with the closed form.

Writes (all numbers programmatic, nothing hand-copied):

* ``audit.json``           -- omega chain, rho table, bisected crossing, long runs;
* ``summary.md``           -- the same numbers as markdown tables (included
  verbatim in ``docs/dem3d_engine_stability_threshold.md``);
* ``engine_threshold.png`` -- rho(dt/dt_crit) with engine markers, and the
  long-run amplitude histories on both sides of the threshold.

Run from the repository root::

    PYTHONPATH=src python3 benchmarks/engine_stability_threshold_audit.py \
        --output artifacts/engine_stability_threshold
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensordem import stability_verification as sv  # noqa: E402
from tensordem.dem3d import IceDEM3D  # noqa: E402


def generic_perturbation(model: IceDEM3D, dt: float, steps: int, seed: int, omega: float) -> dict:
    """Random (non-eigen) perturbation: trajectory-level boundedness check."""
    probe = IceDEM3D(replace(model.config, dt=float(dt)))
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(probe.initial_positions.shape, generator=generator, dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction) * sv.PROBE_AMPLITUDE
    probe.positions = probe.initial_positions + direction
    probe.velocities = torch.zeros_like(probe.positions)
    mass = probe.config.mass
    history: list[float] = []
    for _ in range(steps):
        probe.step()
        moved = probe.positions - probe.initial_positions
        velocity = probe.velocities.clone()
        applied = sv.stiffness_apply(probe, moved)  # probe: restores nothing itself
        probe.velocities = velocity  # keep the integrator state intact
        energy = 0.5 * (float((moved * applied).sum()) / mass + float(velocity.square().sum()))
        history.append(energy)
        if not math.isfinite(energy):
            break
    ratio = [value / history[0] for value in history]
    exponent = range(1, len(ratio) + 1)
    rho = sv.closed_form_rho(dt, omega)
    return {
        "dt_over_critical": dt / (2.0 / omega),
        "steps": len(history),
        "energy_ratio_end": ratio[-1],
        "energy_ratio_peak": max(ratio),
        "energy_ratio_min": min(ratio),
        "rho_closed_form": rho,
        "rho_closed_form_power": rho ** len(history),
        "bounded": bool(max(ratio) < 1.0e3),
    }


def run_audit(*, steps_calm: int = 2000, steps_unstable: int = 400, amplitude_limit: float = 10.0) -> dict:
    model = sv.reference_model()
    analytic = sv.top_mode_analytic(model)
    sweep = sv.threshold_sweep(model, analytic=analytic)
    unit_mode = analytic.eigenvector / torch.linalg.vector_norm(analytic.eigenvector)
    critical = sweep["critical_dt_analytic_s"]

    calm = sv.long_run_growth(0.95 * critical, unit_mode, analytic.omega_rad_s,
                              steps=steps_calm, amplitude_limit=amplitude_limit)
    hot = sv.long_run_growth(1.05 * critical, unit_mode, analytic.omega_rad_s,
                             steps=steps_unstable, amplitude_limit=amplitude_limit)

    generic_calm = generic_perturbation(model, 0.95 * critical, min(steps_calm, 800), 7, analytic.omega_rad_s)
    generic_hot = generic_perturbation(model, 1.05 * critical, min(steps_unstable, 200), 11, analytic.omega_rad_s)

    return {
        "sweep": sweep,
        "long_run": {"calm": calm, "unstable": hot},
        "generic_perturbation": {"calm": generic_calm, "unstable": generic_hot},
        "torch_version": torch.__version__,
    }


def _summary_markdown(audit: dict) -> str:
    sweep = audit["sweep"]
    critical = sweep["critical_dt_analytic_s"]
    lines: list[str] = []
    lines.append("### 1. Highest eigenfrequency: three routes to the same number\n")
    lines.append("| route | omega_max (rad/s) | relative deviation |")
    lines.append("|---|---:|---:|")
    lines.append(f"| bond-lattice assembly + `eigh` (no solver call) | {sweep['analytic_omega_rad_s']:.6f} | reference |")
    lines.append(f"| engine `forces()` power iteration (Rayleigh) | {sweep['engine_omega_rad_s']:.6f} "
                 f"| {sweep['engine_omega_relative_error']:.2e} |")
    lines.append(f"| Gershgorin upper bound (`stability.stability_report`) | {sweep['gershgorin_omega_rad_s']:.6f} "
                 f"| {(sweep['gershgorin_omega_rad_s'] - sweep['analytic_omega_rad_s']) / sweep['analytic_omega_rad_s']:+.2e} |")
    lines.append("")
    lines.append(f"Engine mode residual `||K u - lambda u|| / (lambda ||u||)` = "
                 f"{sweep['engine_mode_residual']:.2e}; particles = {sweep['n_particles']}, "
                 f"bonds = {sweep['n_bonds']}, m = {sweep['mass_kg']:.9f} kg, "
                 f"k_b = {sweep['bond_stiffness_N_per_m']:.0f} N/m, "
                 f"dt_crit = 2/omega_max = {critical:.9e} s, "
                 f"engine `recommended_dt` = {sweep['engine_recommended_dt_s']:.9e} s "
                 f"(= {sweep['engine_recommended_dt_s'] / critical:.3f} dt_crit), "
                 f"top-eigenvalue degeneracy gap = {sweep['analytic_eigenvalue_gap']:.1e}.")
    lines.append("")
    lines.append("### 2. Measured per-step amplification versus the closed form\n")
    lines.append(f"`grows` means rho > 1 + {sweep['marginal_tolerance']:.0e} "
                 "(the undamped scheme is marginally stable below the threshold, so rho = 1 exactly).\n")
    lines.append("| dt/dt_crit | dt (s) | rho engine | rho closed form | rel. error | engine verdict | guard would refuse |")
    lines.append("|---:|---:|---:|---:|---:|:--|:--|")
    for row in sweep["rows"]:
        lines.append(
            f"| {row['dt_over_critical']:.3f} | {row['dt_s']:.6e} | {row['rho_measured']:.9f} "
            f"| {row['rho_closed_form']:.9f} | {row['relative_error']:.2e} "
            f"| {'grows' if row['grows'] else 'bounded'} "
            f"| {'yes' if row['timestep_guard_would_refuse'] else 'no'} |"
        )
    crossing = sweep["crossing"]
    lines.append("")
    lines.append("### 3. Threshold located from engine measurements alone\n")
    lines.append(
        f"Bisection on `sign(rho_measured - 1)` over {crossing['iterations']} engine evaluations gives "
        f"dt* = {crossing['dt_crossing_measured_s']:.9e} s against the closed-form "
        f"2/omega_max = {crossing['dt_crit_closed_form_s']:.9e} s "
        f"(relative error {crossing['relative_error']:.2e}, bracket width "
        f"{crossing['bracket_width_s']:.2e} s).")
    lines.append("")
    calm, hot = audit["long_run"]["calm"], audit["long_run"]["unstable"]
    lines.append("### 4. Long runs on both sides of the threshold\n")
    lines.append("| case | dt/dt_crit | steps | rho closed form | measured per-step growth | rel. error | statement |")
    lines.append("|---|---:|---:|---:|---:|---:|:--|")
    lines.append(
        f"| 0.95 dt_crit | {calm['dt_s'] / critical:.3f} | {calm['steps']} | {calm['rho_closed_form']:.9f} "
        f"| n/a (marginal) | — "
        f"| no growth: exact quadratic-invariant drift {calm['invariant_relative_drift']:.2e} over "
        f"{calm['steps']} steps |")
    lines.append(
        f"| 1.05 dt_crit | {hot['dt_s'] / critical:.3f} | {hot['steps']} | {hot['rho_closed_form']:.9f} "
        f"| {hot['growth_per_step_measured']:.9f} | {hot['growth_per_step_relative_error']:.2e} "
        f"| amplitude 1.000e+00 -> {hot['measured_amplitude_final']:.4e} "
        f"(closed-form prediction {hot['predicted_amplitude_after_steps']:.4e}) |")
    lines.append("")
    lines.append("The 1.05 run stops itself once the mode amplitude leaves the linear regime "
                 f"(peak projection > {10.0:.0f} probe units, i.e. bond strain > 2e-4); an unbounded "
                 "exponential excursion is a *linear-algebra* statement and is not claimed beyond that window.")
    lines.append("")
    gen = audit["generic_perturbation"]
    lines.append("### 5. Generic random perturbation (not the eigenvector)\n")
    lines.append("Mechanical-energy ratio to its own initial value, `0.5 (u.Ku/m + |v|^2)`:\n")
    lines.append("| case | dt/dt_crit | steps | energy ratio end | energy ratio peak | energy ratio min | bounded |")
    lines.append("|---|---:|---:|---:|---:|---:|:--|")
    for key, label in (("calm", "0.95 dt_crit"), ("unstable", "1.05 dt_crit")):
        row = gen[key]
        lines.append(f"| {label} | {row['dt_over_critical']:.3f} | {row['steps']} | "
                     f"{row['energy_ratio_end']:.6e} | {row['energy_ratio_peak']:.6e} | "
                     f"{row['energy_ratio_min']:.6e} | {'yes' if row['bounded'] else 'no'} |")
    lines.append("")
    lines.append("On the stable side the plain energy is *not* a conserved quantity of the scheme "
                 "(the modified energy is); it oscillates, which is why boundedness is asserted with the "
                 "quadratic invariant of section 4, not with this table.")
    lines.append("")
    return "\n".join(lines)


def write_figure(audit: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = audit["sweep"]
    critical = sweep["critical_dt_analytic_s"]
    omega = sweep["analytic_omega_rad_s"]
    fractions = [0.05 + 0.01 * step for step in range(0, 196)]
    closed = [sv.closed_form_rho(fraction * critical, omega) for fraction in fractions]
    measured = [row["dt_over_critical"] for row in sweep["rows"]]
    rho = [row["rho_measured"] for row in sweep["rows"]]

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.6, 4.7))

    left.plot(fractions, closed, "-", color="tab:blue", lw=2.2,
              label=r"closed form $\rho(dt\,\omega_{max})$")
    left.plot(measured, rho, "o", color="tab:red", ms=7, mfc="none", mew=1.7,
              label="engine measurement (2x2 map from `IceDEM3D.step`)")
    left.axhline(1.0, color="0.4", lw=1.0, ls=":")
    left.axvline(1.0, color="0.4", lw=1.0, ls="--")
    left.axvspan(1.0, fractions[-1], color="tab:red", alpha=0.07)
    crossing = sweep["crossing"]
    left.plot([crossing["dt_crossing_measured_s"] / critical], [1.0 + sweep["marginal_tolerance"]],
              "k*", ms=13, label="bisected engine threshold")
    left.annotate("dt > dt_crit: perturbations grow", xy=(1.42, 3.0), color="tab:red",
                  fontsize=9, ha="center")
    left.annotate("dt < dt_crit: bounded", xy=(0.5, 3.0), color="tab:green", fontsize=9, ha="center")
    left.set_xlabel(r"$dt / dt_{crit}$,   $dt_{crit} = 2/\omega_{max}$")
    left.set_ylabel(r"per-step amplification $\rho$")
    left.set_yscale("log")
    left.set_title("Engine reproduces the closed-form threshold")
    left.legend(fontsize=8, loc="upper left")
    left.grid(alpha=0.25)

    calm, hot = audit["long_run"]["calm"], audit["long_run"]["unstable"]
    right.semilogy(range(1, len(calm["invariant_history"]) + 1), calm["invariant_history"],
                   color="tab:green", lw=1.2,
                   label=f"0.95 dt_crit: quadratic invariant ({calm['steps']} steps, stays 1)")
    right.semilogy(range(1, len(hot["history"]) + 1), hot["history"], color="tab:red", lw=1.2,
                   label=f"1.05 dt_crit ({hot['steps']} steps, grows)")
    exponent = list(range(1, len(hot["history"]) + 1))
    right.semilogy(exponent, [1.0 * hot["rho_closed_form"] ** step for step in exponent],
                   "k--", lw=1.0, label=r"closed-form $\rho^{\,n}$")
    right.set_xlabel("engine step n")
    right.set_ylabel("normalised quantity (start = 1)")
    right.set_title("Long run: bounded below, exponential above")
    right.legend(fontsize=8, loc="lower right")
    right.grid(alpha=0.25, which="both")

    figure.suptitle("TensorDEM: engine-level verification of the explicit time-step threshold", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/engine_stability_threshold")
    parser.add_argument("--steps-calm", type=int, default=2000)
    parser.add_argument("--steps-unstable", type=int, default=400)
    parser.add_argument("--amplitude-limit", type=float, default=10.0)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    audit = run_audit(steps_calm=args.steps_calm, steps_unstable=args.steps_unstable,
                      amplitude_limit=args.amplitude_limit)
    payload = dict(audit)
    payload["long_run"] = {
        key: {inner: value for inner, value in row.items()
              if inner not in ("history", "invariant_history")}
        for key, row in audit["long_run"].items()
    }
    (out / "audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = _summary_markdown(audit)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    write_figure(audit, out / "engine_threshold.png")
    print(summary)
    print(f"\nartifacts written to {out}")


if __name__ == "__main__":
    main()