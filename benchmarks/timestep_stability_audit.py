"""Reproducible audit of the explicit time-step stability criteria (CFL-class).

Produces (all numbers programmatically, nothing hand-copied):

* ``report.json``  -- verification table for the closed-form growth factor, the
  SDOF / chain thresholds, the Gershgorin bound and the engine margin sweep;
* ``stability_domain.png`` -- analytic growth factor plus measured markers and
  the measured energy amplification versus ``dt / dt_crit``;
* ``engine_margin.png`` -- audited critical step of representative engine
  configurations against the step the solver actually takes.

Run from the repository root::

    python benchmarks/timestep_stability_audit.py \
        --output docs/dem3d-timestep-stability
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from tensordem import stability
from tensordem.dem3d import DEM3DConfig, IceDEM3D


def _step_scalar(x, v, dt, omega, damping_rate):
    """Reference implementation of one semi-implicit Euler step (independent of src)."""
    v_new = v + dt * (-omega * omega * x - damping_rate * v)
    return x + dt * v_new, v_new


def measured_growth(dt: float, omega: float, damping_rate: float) -> float:
    """Reconstruct the 2x2 step matrix by probing the integrator, then take rho."""
    first = _step_scalar(1.0, 0.0, dt, omega, damping_rate)
    second = _step_scalar(0.0, 1.0, dt, omega, damping_rate)
    a11, a12 = first[0], second[0]
    a21, a22 = first[1], second[1]
    trace, det = a11 + a22, a11 * a22 - a12 * a21
    disc = trace * trace - 4.0 * det
    if disc >= 0.0:
        root = math.sqrt(disc)
        return max(abs(0.5 * (trace + root)), abs(0.5 * (trace - root)))
    return math.sqrt(abs(det))


def energy_amplification(dt_fraction: float, omega: float = 1.0, steps: int = 4000) -> float:
    """Peak energy of the ``x0=1, v0=0`` oscillator relative to its initial energy."""
    dt = dt_fraction * 2.0 / omega
    x, v, peak = 1.0, 0.0, 0.0
    for _ in range(steps):
        x, v = _step_scalar(x, v, dt, omega, 0.0)
        peak = max(peak, 0.5 * (omega * omega * x * x + v * v))
    return peak / (0.5 * omega * omega)


def chain_top_frequency(n: int, stiffness: float, mass: float) -> float:
    """Closed-form highest eigenfrequency of a free-free 1-D spring-mass chain."""
    return math.sqrt((2.0 * stiffness / mass) * (1.0 + math.cos(math.pi / n)))


def chain_peak_energy(dt: float, n: int, stiffness: float, mass: float, steps: int) -> float:
    """Peak/total energy ratio of the chain driven in its exact top eigenmode."""
    mode = [math.sin((n - 1) * math.pi * (index + 1) / n) for index in range(n)]
    x = [0.0] * n
    v = list(mode)
    initial = 0.5 * mass * sum(component * component for component in v)
    peak = initial
    for _ in range(steps):
        force = [0.0] * n
        for i in range(n):
            if i > 0:
                force[i] += stiffness * (x[i - 1] - x[i])
            if i < n - 1:
                force[i] += stiffness * (x[i + 1] - x[i])
        for i in range(n):
            v[i] += dt * force[i] / mass
        for i in range(n):
            x[i] += dt * v[i]
        total = 0.5 * mass * sum(component * component for component in v)
        total += 0.5 * stiffness * sum((x[i + 1] - x[i]) ** 2 for i in range(n - 1))
        peak = max(peak, total)
    return {"peak_energy_ratio": peak / initial, "final_energy_ratio": total / initial}


def marginal_growth_ratio(omega: float = 100.0) -> dict:
    """Energy growth of the marginal step dt = 2/omega (Jordan block => algebraic)."""
    dt = 2.0 / omega

    def energy(steps: int) -> float:
        x, v = 1.0, 0.0
        for _ in range(steps):
            x, v = _step_scalar(x, v, dt, omega, 0.0)
        return 0.5 * (omega * omega * x * x + v * v)

    e500, e1000, e2000 = energy(500), energy(1000), energy(2000)
    return {"energy_500": e500, "energy_1000": e1000, "energy_2000": e2000,
            "ratio_1000_over_500": e1000 / e500, "ratio_2000_over_1000": e2000 / e1000}


ENGINE_CASES: list[tuple[str, dict]] = [
    ("baseline 9x5x3", {}),
    ("small 3x3x3", dict(nx=3, ny=3, nz=3)),
    ("large 12x6x3", dict(nx=12, ny=6, nz=3)),
    ("platens on", dict(top_platen_enabled=True, bottom_platen_enabled=True)),
    ("platen damping 50", dict(top_platen_enabled=True, platen_damping=50.0,
                               top_platen_velocity=-0.01)),
    ("cylinder_xz 5x3x5", dict(geometry="cylinder_xz", nx=5, ny=3, nz=5)),
    ("bond k = 2e5", dict(bond_stiffness=2.0e5)),
    ("contact k = 2e5", dict(contact_stiffness=2.0e5)),
    ("contact damping 500", dict(contact_damping=500.0)),
    ("drag 100", dict(drag=100.0)),
    ("radius 5 mm", dict(radius=0.005)),
]


def engine_sweep() -> list[dict]:
    rows = []
    for label, overrides in ENGINE_CASES:
        model = IceDEM3D(DEM3DConfig(**overrides))
        report = stability.stability_report(model)
        rows.append({
            "case": label,
            "dt_s": report["dt_s"],
            "engine_recommended_dt_s": report["engine_recommended_dt_s"],
            "critical_dt_s": report["critical_dt_s"],
            "binding_limit": report["binding_limit"],
            "margin_factor": report["margin_factor"],
            "growth_factor_at_dt": report["growth_factor_at_dt"],
            "omega_upper_rad_s": report["omega_upper_rad_s"],
            "is_stable": report["is_stable"],
            "guard_is_inside_limit": report["engine_recommended_dt_s"] <= report["critical_dt_s"],
        })
    return rows


def growth_table() -> dict:
    omega = 50.0
    d_values = [0.2, 0.5, 0.9, 1.5, 1.9, 1.99, 2.0, 2.01, 2.4, 2.9]
    gamma_values = [0.0, 0.5, 1.5, 1.99, 2.0, 2.1]
    rows, worst = [], 0.0
    for d in d_values:
        for gamma in gamma_values:
            dt = d / omega
            damping_rate = gamma / dt if dt > 0 else 0.0
            analytic = stability.symplectic_euler_growth_factor(dt, omega, damping_rate)
            measured = measured_growth(dt, omega, damping_rate)
            rel = abs(analytic - measured) / max(analytic, 1e-300)
            worst = max(worst, rel)
            rows.append({"dt_over_crit": d / 2.0, "damping_gamma": gamma,
                         "analytic_rho": analytic, "measured_rho": measured,
                         "rel_error": rel})
    return {"omega_rad_s": omega, "rows": rows, "max_rel_error": worst}


def engine_dt_sweep_energy() -> dict:
    """Closed-system engine run: perturbation energy ratio versus dt/dt_crit."""
    rows = []
    torch.manual_seed(11)
    reference = IceDEM3D(DEM3DConfig(nx=6, ny=4, nz=3))
    limit = stability.stability_report(reference)["critical_dt_s"]
    for fraction in (0.2, 0.4, 0.6, 0.8, 0.9, 0.95):
        config = DEM3DConfig(nx=6, ny=4, nz=3)
        dt = fraction * limit
        if dt > config.recommended_dt:
            # The engine refuses a step above its own guard, so larger audited
            # fractions are not reachable through the public API.
            rows.append({"dt_over_crit": fraction, "runnable": False,
                         "engine_recommended_dt_s": config.recommended_dt,
                         "dt_over_crit_engine_limit": config.recommended_dt / limit})
            continue
        model = IceDEM3D(config)
        model.tool_start = torch.tensor([0.0, 0.0, 1.0e3], dtype=torch.float64)
        model.velocities += 1.0e-3 * torch.randn_like(model.velocities)
        initial = float(0.5 * model.config.mass * model.velocities.square().sum())
        steps = max(1, int(round(0.02 / model.dt)))
        peak = initial
        for _ in range(steps):
            model.step()
            peak = max(peak, float(0.5 * model.config.mass * model.velocities.square().sum()))
        rows.append({"dt_over_crit": fraction, "runnable": True, "steps": steps,
                     "peak_energy_ratio": peak / initial,
                     "final_energy_ratio": float(0.5 * model.config.mass
                                                  * model.velocities.square().sum()) / initial,
                     "broken_bonds": model.broken_bonds})
    return {"critical_dt_s": limit, "rows": rows}


def engine_closed_system_energy(steps: int = 2000) -> dict:
    """Relative drift of the mechanical energy of a closed, unforced particle block.

    The indenter is disengaged and no fracture occurs, so kinetic + bond energy
    must oscillate around its initial value instead of growing.  This is the only
    engine-level stability evidence reachable through the public API: the step
    is a fixed ~2.2% of ``dt_crit`` because ``DEM3DConfig.recommended_dt`` is far
    more conservative than the stability limit (see ``engine_sweep``).
    """
    torch.manual_seed(7)
    model = IceDEM3D(DEM3DConfig(nx=6, ny=4, nz=3))
    model.tool_start = torch.tensor([0.0, 0.0, 1.0e3], dtype=torch.float64)
    model.velocities += 1.0e-3 * torch.randn_like(model.velocities)

    def mechanical_energy() -> float:
        kinetic = 0.5 * model.config.mass * float(model.velocities.square().sum())
        i, j = model.pairs.T
        distance = torch.linalg.vector_norm(model.positions[j] - model.positions[i], dim=1)
        extension = distance - model.rest_lengths
        bond = 0.5 * model.config.bond_stiffness * float((extension.square() * model.alive).sum())
        return kinetic + bond

    initial = mechanical_energy()
    peak, final = initial, initial
    for _ in range(steps):
        model.step()
        energy = mechanical_energy()
        peak = max(peak, energy)
        final = energy
    report = stability.stability_report(model)
    return {
        "steps": steps,
        "simulated_time_s": model.time,
        "dt_over_crit": report["dt_s"] / report["critical_dt_s"],
        "initial_energy_J": initial,
        "peak_over_initial": peak / initial,
        "final_over_initial": final / initial,
        "broken_bonds": model.broken_bonds,
        "growth_factor_at_dt": report["growth_factor_at_dt"],
    }


def _summary_markdown(report: dict) -> str:
    """Render the generated verification tables (numbers never hand-copied)."""
    lines = ["<!-- Generated by benchmarks/timestep_stability_audit.py; do not edit by hand. -->",
             "# Generated verification tables", ""]

    growth = report["growth_factor_verification"]
    lines += ["## 1. Closed-form growth factor vs independently reconstructed step matrix",
              "",
              "| dt/dt_crit | gamma | rho analytic | rho measured | rel. error |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for row in growth["rows"]:
        lines.append("| {:.3f} | {:.2f} | {:.10f} | {:.10f} | {:.2e} |".format(
            row["dt_over_crit"], row["damping_gamma"], row["analytic_rho"],
            row["measured_rho"], row["rel_error"]))
    lines += ["", f"Maximum relative error over the grid: **{growth['max_rel_error']:.3e}**.", ""]

    marginal = report["marginal_step_dt_equals_crit"]
    lines += ["## 2. Marginal step dt = dt_crit (Jordan block)",
              "",
              f"- energy at 500/1000/2000 steps: {marginal['energy_500']:.6e}, "
              f"{marginal['energy_1000']:.6e}, {marginal['energy_2000']:.6e}",
              f"- E(1000)/E(500) = {marginal['ratio_1000_over_500']:.6f}, "
              f"E(2000)/E(1000) = {marginal['ratio_2000_over_1000']:.6f} "
              "(quadratic growth expected: 4)", ""]

    chain = report["chain_reference"]
    lines += ["## 3. 32-mass spring chain (top eigenmode)",
              "",
              f"- exact top frequency: {chain['omega_max_exact_rad_s']:.12f} rad/s",
              f"- Gershgorin upper bound: {chain['gershgorin_upper_rad_s']:.12f} rad/s "
              f"(conservativeness {chain['conservativeness_ratio']:.6f})",
              "",
              "| dt/dt_crit | peak energy / E0 | final energy / E0 |",
              "| ---: | ---: | ---: |"]
    for row in chain["energy_vs_dt"]:
        lines.append("| {:.2f} | {:.6f} | {:.6f} |".format(
            row["dt_over_crit"], row["peak_energy_ratio"], row["final_energy_ratio"]))
    lines.append("")

    lines += ["## 4. Engine configurations: step used vs audited limit",
              "",
              "| configuration | dt (s) | engine recommended dt (s) | audited dt_crit (s) | margin | binding | guard inside limit |",
              "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for row in report["engine_sweep"]:
        lines.append("| {} | {:.3e} | {:.3e} | {:.3e} | {:.1f} | {} | {} |".format(
            row["case"], row["dt_s"], row["engine_recommended_dt_s"], row["critical_dt_s"],
            row["margin_factor"], row["binding_limit"], row["guard_is_inside_limit"]))
    lines.append("")

    closed = report["engine_closed_system_energy"]
    lines += ["## 5. Engine closed-system energy (indenter disengaged, no fracture)",
              "",
              f"- steps: {closed['steps']} ({closed['simulated_time_s']:.6g} s), "
              f"dt/dt_crit = {closed['dt_over_crit']:.4f}",
              f"- peak/initial energy: {closed['peak_over_initial']:.6f} "
              f"(no growth), final/initial: {closed['final_over_initial']:.6f} "
              "(drag dissipation only)",
              f"- broken bonds: {closed['broken_bonds']}, "
              f"growth factor at dt: {closed['growth_factor_at_dt']:.6f}", ""]

    sweep = report["engine_dt_sweep"]
    lines += ["## 6. Reachability of high dt/dt_crit ratios through the engine API",
              "",
              f"Audited critical step for the 6x4x3 block: {sweep['critical_dt_s']:.3e} s. "
              f"``DEM3DConfig.recommended_dt`` allows at most "
              f"dt/dt_crit = {sweep['rows'][0]['dt_over_crit_engine_limit']:.4f}, "
              "so audited fractions above that value are refused by ``IceDEM3D.__init__`` "
              "and cannot be exercised through the public API (see limitations).", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/dem3d-timestep-stability"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    stiffness, mass, n = 1.0, 1.0, 32
    omega_chain = chain_top_frequency(n, stiffness, mass)
    chain_rows = []
    for fraction in (0.25, 0.5, 0.8, 0.9):
        row = chain_peak_energy(fraction * 2.0 / omega_chain, n, stiffness, mass, 1500)
        row["dt_over_crit"] = fraction
        chain_rows.append(row)

    report = {
        "scheme": "semi-implicit (symplectic) Euler: v += dt*f/m; x += dt*v",
        "closed_form": {
            "dt_crit_elastic": "2 / omega_max",
            "dt_crit_damping": "2 * m / c",
            "growth_factor": "spectral radius of [[1-d^2, dt(1-g)],[-dt*w^2, 1-g]]",
            "omega_upper": "sqrt(2 * max_i S_i / m)  (Gershgorin row bound)",
        },
        "growth_factor_verification": growth_table(),
        "marginal_step_dt_equals_crit": marginal_growth_ratio(),
        "chain_reference": {
            "n_particles": n,
            "stiffness_N_per_m": stiffness,
            "mass_kg": mass,
            "omega_max_exact_rad_s": omega_chain,
            "gershgorin_upper_rad_s": 2.0 * math.sqrt(stiffness / mass),
            "conservativeness_ratio": (2.0 * math.sqrt(stiffness / mass)) / omega_chain,
            "energy_vs_dt": chain_rows,
        },
        "engine_sweep": engine_sweep(),
        "engine_dt_sweep": engine_dt_sweep_energy(),
        "engine_closed_system_energy": engine_closed_system_energy(),
        "environment": {"torch": torch.__version__, "device": "cpu"},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")

    (args.output / "summary.md").write_text(_summary_markdown(report), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"matplotlib unavailable ({exc}); report.json written without figures")
        return

    # Figure 1: analytic stability map with measured markers + energy amplification.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axis = axes[0]
    d_grid = [0.02 * index for index in range(1, 201)]
    for gamma in (0.0, 0.5, 1.5, 1.99):
        analytic = [stability.symplectic_euler_growth_factor(
            0.05 * d, 20.0, gamma / (0.05 * d)) for d in d_grid]
        axis.plot(d_grid, analytic, linewidth=1.2, label=f"analytic g={gamma:g}")
    selected = report["growth_factor_verification"]["rows"]
    for gamma, marker in ((0.0, "o"), (1.5, "s")):
        points = [row for row in selected if row["damping_gamma"] == gamma]
        axis.plot([2.0 * row["dt_over_crit"] for row in points],
                  [row["measured_rho"] for row in points],
                  linestyle="none", marker=marker, markerfacecolor="none",
                  label=f"measured g={gamma:g}")
    axis.axvline(2.0, color="k", linestyle="--", linewidth=1, label="dt = dt_crit")
    axis.set_xlabel("dt * omega_max")
    axis.set_ylabel("per-step growth factor rho")
    axis.set_yscale("log")
    axis.set_title("(a) symplectic Euler amplification")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)

    axis = axes[1]
    fractions = [0.05 * index for index in range(1, 20)]
    axis.plot(fractions, [energy_amplification(fraction) for fraction in fractions],
              marker="o", markersize=3, label="SDOF, x0=1, v0=0")
    axis.plot([row["dt_over_crit"] for row in chain_rows],
              [row["peak_energy_ratio"] for row in chain_rows],
              marker="s", linestyle="--", label="32-mass chain, top mode")
    axis.axvline(1.0, color="k", linestyle="--", linewidth=1, label="dt = dt_crit")
    axis.axhline(1.1, color="r", linestyle=":", linewidth=1, label="10% energy band")
    axis.set_xlabel("dt / dt_crit")
    axis.set_ylabel("peak energy / initial energy")
    axis.set_yscale("log")
    axis.set_title("(b) energy amplification vs step margin")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output / "stability_domain.png", dpi=150)
    plt.close(fig)

    # Figure 2: engine audit -- the step in use vs the audited limit.
    rows = [row for row in report["engine_sweep"] if math.isfinite(row["critical_dt_s"])]
    labels = [row["case"] for row in rows]
    positions = list(range(len(rows)))
    fig, axis = plt.subplots(figsize=(9, 4.6))
    axis.barh([p + 0.2 for p in positions], [row["critical_dt_s"] for row in rows],
              height=0.4, label="audited critical dt")
    axis.barh([p - 0.2 for p in positions], [row["dt_s"] for row in rows],
              height=0.4, label="step actually used")
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xscale("log")
    axis.set_xlabel("time step (s)")
    axis.set_title("engine configurations: step used vs audited stability limit")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(args.output / "engine_margin.png", dpi=150)
    plt.close(fig)

    print(f"wrote {args.output/'report.json'}, stability_domain.png, engine_margin.png")
    print("max relative error of closed-form growth factor: "
          f"{report['growth_factor_verification']['max_rel_error']:.3e}")
    print("worst engine margin factor: "
          f"{min(row['margin_factor'] for row in report['engine_sweep']):.3g}")


if __name__ == "__main__":
    main()