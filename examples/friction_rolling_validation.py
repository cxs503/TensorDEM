"""Run the DEM3D friction/rolling verification campaign and write artifacts.

Writes into ``outputs/friction_rolling_validation/``:

* ``results.json``  - full campaign record (measured vs reference per scenario)
* ``tables.md``     - markdown tables rendered from ``results.json``
* ``fig_stick_slip.png``  - tangential force vs driven slip with the analytic
  elastic line and the Coulomb limit
* ``fig_reversal.png``    - slip needed to reverse a saturated friction force
  vs sliding history, current model (flat) against the unclamped model (linear)
* ``fig_rolling.png``     - rolling-resistance force vs mu_r, plus the lateral
  force produced by the pre-fix couple model
* ``fig_work_convergence.png`` - sliding-work error vs time step (log-log)

Usage:
    PYTHONPATH=src python examples/friction_rolling_validation.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.friction_rolling_verification import (  # noqa: E402
    ContactPatch,
    PatchConfig,
    _stick_limit,
    run_campaign,
)

DEFAULT_OUT = pathlib.Path("outputs/friction_rolling_validation")


def _fmt(value: float, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def render_tables(results: dict) -> str:
    lines = ["# Friction / rolling verification - programmatic tables", ""]
    cfg = results["config"]
    lines += [
        "## Contact patch (analytic normal force)",
        "",
        "| quantity | value |", "|---|---|",
        f"| radius R (m) | {_fmt(cfg['radius_m'])} |",
        f"| contact stiffness k_n (N/m) | {_fmt(cfg['contact_stiffness_N_per_m'])} |",
        f"| overlap (m) | {_fmt(cfg['overlap_m'])} |",
        f"| normal force F_n (N) | {_fmt(cfg['normal_force_N'])} |",
        f"| time step dt (s) | {_fmt(cfg['dt_s'])} |",
        "",
        "## Elastic (sticking) regime",
        "",
        "| quantity | measured | reference | rel. error |", "|---|---|---|---|",
    ]
    stick = results["stick"]
    lines += [
        f"| tangential force (N) | {_fmt(stick['f_tangent_measured_N'])} | "
        f"{_fmt(stick['f_tangent_reference_N'])} | {_fmt(stick['rel_error'], 3)} |",
        f"| stored slip (m) | {_fmt(stick['stored_slip_measured_m'])} | "
        f"{_fmt(stick['stored_slip_reference_m'])} | {_fmt(stick['stored_slip_rel_error'], 3)} |",
        f"| stick ratio slip/a_stick | {_fmt(stick['stick_ratio'], 4)} | < 1 | - |",
        "",
        "## Sliding (saturated) regime",
        "",
        "| mu | F_limit (N) | measured |F_t| (N) | rel. error | stored slip (m) | "
        "a_stick (m) | stored/a_stick | driven slip (m) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in results["sliding"]["rows"]:
        lines.append(
            f"| {row['mu']} | {_fmt(row['f_limit_N'])} | "
            f"{_fmt(-row['f_tangent_measured_N'])} | {_fmt(row['f_tangent_rel_error'], 3)} | "
            f"{_fmt(row['stored_slip_measured_m'])} | {_fmt(row['a_stick_m'])} | "
            f"{_fmt(row['stored_slip_ratio'], 6)} | {_fmt(row['unclamped_slip_m'])} |")
    rev = results["reversal"]
    lines += [
        "",
        "## Friction-force reversal (history independence)",
        "",
        f"Reference: reversing a saturated force needs one ``a_stick`` = "
        f"{_fmt(rev['a_stick_m'])} m of reversed slip.",
        "",
        "| sliding duration (s) | force evaluations | slip before reversal (m) | "
        "steps to flip | reversal slip (m) | excess (quanta) |",
        "|---|---|---|---|---|---|",
    ]
    for row in rev["rows"]:
        lines.append(
            f"| {row['sliding_duration_s']} | {row['sliding_force_evaluations']} | "
            f"{_fmt(row['slip_before_reversal_m'])} | {row['reversal_flip_step']} | "
            f"{_fmt(row['reversal_slip_m'])} | {_fmt(row['reversal_slip_excess_quanta'], 3)} |")
    lines += [f"\nSpread across histories: {_fmt(rev['reversal_slip_spread_m'])} m "
              f"(history independent: {rev['history_independent']})."]
    nr = results["normal_rotation"]
    lines += [
        "",
        "## Contact-normal rotation",
        "",
        "| quantity | value |", "|---|---|",
        f"| normal force measured (N) | {_fmt(nr['f_normal_measured_N'])} |",
        f"| normal force reference (N) | {_fmt(nr['f_normal_reference_N'])} |",
        f"| spurious normal component of friction (N) | {_fmt(nr['spurious_normal_component_N'], 3)} |",
        f"| as fraction of the friction limit | {_fmt(nr['spurious_fraction_of_limit'], 3)} |",
        "",
        "## Separation memory",
        "",
        "| quantity | value |", "|---|---|",
        f"| stored slip before separation (m) | {_fmt(results['separation']['stored_slip_before_m'])} |",
        f"| pairs retained after separation | {results['separation']['states_retained_after_separation']} |",
        f"| tangential force on re-contact (N) | {_fmt(results['separation']['recontact_tangential_force_N'], 3)} |",
        "",
        "## Rolling resistance",
        "",
        "| quantity | measured | reference | rel. error |", "|---|---|---|---|",
    ]
    ro = results["rolling"]
    lines += [
        f"| tangential force (N) | {_fmt(ro['f_tangential_measured_N'])} | "
        f"{_fmt(ro['f_tangential_reference_N'])} | {_fmt(ro['rel_error'], 3)} |",
        f"| lateral force (N) | {_fmt(ro['f_lateral_measured_N'], 3)} | 0 | - |",
        f"| dissipation rate (W) | {_fmt(ro['dissipation_rate_W'])} | "
        f"{_fmt(ro['dissipation_rate_reference_W'])} | "
        f"{_fmt(abs((ro['dissipation_rate_W'] - ro['dissipation_rate_reference_W']) / ro['dissipation_rate_reference_W']), 3)} |",
        f"| control mu_r = 0 force (N) | {_fmt(ro['control_mu_r0_f_tangential_N'], 3)} | 0 | - |",
        "",
        "## Sliding work",
        "",
        "| quantity | measured | reference | rel. error |", "|---|---|---|---|",
    ]
    sw = results["sliding_work"]
    lines += [
        f"| total work (J) | {_fmt(sw['work_measured_J'])} | {_fmt(sw['work_reference_J'])} | "
        f"{_fmt(sw['rel_error'], 3)} |",
        f"| dissipated (J) | {_fmt(sw['dissipation_measured_J'])} | "
        f"{_fmt(sw['dissipation_reference_J'])} | {_fmt(sw['dissipation_rel_error'], 3)} |",
        "",
        "## Time-step convergence of the sliding work",
        "",
        "| dt (s) | steps | slip (m) | work measured (J) | work reference (J) | rel. error |",
        "|---|---|---|---|---|---|",
    ]
    for row in results["work_convergence"]["rows"]:
        lines.append(
            f"| {_fmt(row['dt_s'])} | {row['steps']} | {_fmt(row['slip_m'])} | "
            f"{_fmt(row['work_measured_J'])} | {_fmt(row['work_reference_J'])} | "
            f"{_fmt(row['rel_error'], 3)} |")
    lines += ["", f"Error ratio first/last: "
                  f"{_fmt(results['work_convergence']['error_ratio_first_to_last'], 4)} "
                  f"(first order expected: 4)."]
    return "\n".join(lines) + "\n"


def fig_stick_slip(out: pathlib.Path, cfg: PatchConfig) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mu, colour in ((0.1, "tab:blue"), (0.2, "tab:orange")):
        patch = ContactPatch(cfg, mu=mu)
        records = patch.drive(speed=1.0, steps=60)
        slip = [1.0 * cfg.dt * (k + 1) for k in range(len(records))]
        force = [-r["f_tangent"] for r in records]
        ax.plot([s * 1e3 for s in slip], force, "o", ms=3, color=colour,
                label=f"measured, mu = {mu}")
        a_stick = _stick_limit(cfg, mu)
        ax.plot([0, a_stick * 1e3], [0, mu * cfg.normal_force], "-", color=colour,
                lw=1.2, label=f"analytic k_t*slip, mu = {mu}")
        ax.axhline(mu * cfg.normal_force, color=colour, ls="--", lw=1.0,
                   label=f"mu*F_n = {mu * cfg.normal_force:.2f} N")
    ax.set_xlabel("driven relative slip (mm)")
    ax.set_ylabel("friction force magnitude |F_t| (N)")
    ax.set_title("Coulomb branch: elastic ramp then saturation")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_stick_slip.png", dpi=130)
    plt.close(fig)


def fig_reversal(out: pathlib.Path, results: dict, cfg: PatchConfig) -> None:
    rev = results["reversal"]
    rows = rev["rows"]
    durations = [r["sliding_duration_s"] for r in rows]
    measured = [r["reversal_slip_m"] * 1e3 for r in rows]
    speed, mu = 1.0, 0.1
    unclamped = [speed * d * 1e3 for d in durations]   # model of the old defect
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(durations, measured, "o-", color="tab:green",
            label="measured (persisted slip clamped)")
    ax.plot(durations, unclamped, "s--", color="tab:red",
            label="unclamped-state model (pre-fix behaviour)")
    ax.axhline(rev["a_stick_m"] * 1e3, color="k", ls=":", lw=1.0,
               label=f"a_stick = {rev['a_stick_m'] * 1e3:.3f} mm")
    ax.set_yscale("log")
    ax.set_xlabel("sliding duration before reversal (s)")
    ax.set_ylabel("slip needed to reverse F_t (mm)")
    ax.set_title("Friction memory must not grow with sliding history")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig_reversal.png", dpi=130)
    plt.close(fig)


def fig_rolling(out: pathlib.Path, cfg: PatchConfig) -> None:
    mu_rs = [0.01, 0.02, 0.05, 0.1]
    measured, reference, lateral = [], [], []
    for mu_r in mu_rs:
        patch = ContactPatch(cfg, mu=0.0, mu_r=mu_r)
        patch.set_velocity(tangent_axis=1, speed=1.0)
        force, _ = patch.sim.forces()
        comp = force[1].tolist()
        measured.append(-comp[1])
        lateral.append(abs(comp[2]))
        reference.append(mu_r * cfg.normal_force)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(mu_rs, reference, "k-", lw=1.4, label="analytic mu_r*F_n")
    ax.plot(mu_rs, measured, "o", ms=6, color="tab:green", label="measured |F_t|")
    ax.plot(mu_rs, lateral, "x--", color="tab:red",
            label="pre-fix model: lateral force (does no work)")
    ax.set_xlabel("rolling-resistance coefficient mu_r")
    ax.set_ylabel("force on particle 1 (N)")
    ax.set_title("Rolling resistance must be tangential and dissipative")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_rolling.png", dpi=130)
    plt.close(fig)


def fig_convergence(out: pathlib.Path, results: dict) -> None:
    rows = results["work_convergence"]["rows"]
    dts = [r["dt_s"] for r in rows]
    errs = [abs(r["rel_error"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(dts, errs, "o-", label="measured relative error")
    ax.loglog(dts, [errs[0] * dts[0] / d for d in dts], "k--", lw=1.0,
              label="first-order reference (slope -1)")
    ax.set_xlabel("time step dt (s)")
    ax.set_ylabel("relative error of sliding work")
    ax.set_title("Sliding-work residual is a discretisation effect")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig_work_convergence.png", dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results = run_campaign()
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "tables.md").write_text(render_tables(results))

    cfg = PatchConfig()
    fig_stick_slip(args.out, cfg)
    fig_reversal(args.out, results, cfg)
    fig_rolling(args.out, cfg)
    fig_convergence(args.out, results)

    print(f"wrote artifacts to {args.out}")
    print(f"  sliding rel. error (mu=0.2): "
          f"{results['sliding']['rows'][2]['f_tangent_rel_error']:.3e}")
    print(f"  reversal spread: {results['reversal']['reversal_slip_spread_m']:.3e} m")
    print(f"  spurious normal component: "
          f"{results['normal_rotation']['spurious_normal_component_N']:.3e} N")
    print(f"  rolling rel. error: {results['rolling']['rel_error']:.3e}")
    print(f"  sliding work rel. error: {results['sliding_work']['rel_error']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())