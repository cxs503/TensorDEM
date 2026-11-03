"""Engine-level time-step threshold benchmark for the 3-D DEM.

Generates the evidence for ``docs/dem3d_engine_threshold_verification.md``:
divergence onset versus the closed-form limit, measured growth factor versus the
closed-form spectral radius, marginal ``E ~ n^2`` growth, the damping-bound
regime, the row-sum conservativeness, and the untouched production guard.

Run::

    PYTHONPATH=src python3 scripts/benchmark_dem3d_engine_threshold.py

Outputs (all inside the repository):
    docs/dem3d-engine-threshold/threshold_report.json
    docs/dem3d-engine-threshold/threshold_summary.csv
    docs/dem3d-engine-threshold/energy_histories.png
    docs/dem3d-engine-threshold/onset_comparison.png
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.dem3d import IceDEM3D  # noqa: E402
from tensordem.engine_stability_threshold import (  # noqa: E402
    exact_omega_max,
    make_elastic_lattice,
    predict_critical_dt,
    run_energy_history,
    threshold_report,
)
from tensordem.stability import symplectic_euler_growth_factor  # noqa: E402

OUT = Path("docs/dem3d-engine-threshold")

# (name, config kwargs, divergence-search step budget per probe).  The budgets are
# sized so that one full run of this script stays inside a two-minute slot on a
# loaded CPU; the falsification power is unaffected because the onset is located
# by bisection to 2e-4 in dt/dt_crit either way.
CONFIGS = [
    ("cubic4_bond2000", dict(nx=4, ny=4, nz=4, bond_stiffness=2_000.0), 700),
    ("cubic3_bond1000", dict(nx=3, ny=3, nz=3, bond_stiffness=1_000.0), 250),
]


def history_panel(kind: str, ratios: list[float], config, dt_crit: float) -> dict:
    """Record energy histories for the requested ``dt / dt_crit`` ratios."""
    curves = {}
    for ratio in ratios:
        record = run_energy_history(
            config,
            dt=ratio * dt_crit,
            n_steps=300,
            blowup_ratio=1.0e6,
            sample_every=1,
        )
        e0 = record["energy_initial_J"]
        curves[f"{ratio:g}"] = {
            "ratio": ratio,
            "energies_over_initial": [value / e0 for value in record["history"]],
            "diverged": record["diverged"],
            "diverged_at_step": record["diverged_at_step"],
            "peak_over_initial": record["peak_over_initial"],
            "broken_bonds": record["broken_bonds"],
        }
    return {"kind": kind, "dt_crit_s": dt_crit, "curves": curves}


def _partial_path(name: str) -> Path:
    return OUT / f"_partial_{name}.json"


def run_one(name: str, kwargs: dict, onset_steps: int) -> None:
    """Run one configuration and dump its partial record (restartable)."""
    OUT.mkdir(parents=True, exist_ok=True)
    report = threshold_report(onset_steps=onset_steps, **kwargs)
    data = report.as_dict()
    data["name"] = name
    data["config"] = {k: v for k, v in make_elastic_lattice(**kwargs).__dict__.items()}
    cfg = make_elastic_lattice(**kwargs)
    model = IceDEM3D(cfg)
    omega = exact_omega_max(model)
    _, elastic_limit, _ = predict_critical_dt(
        omega_max=omega, mass=cfg.mass, damping_coefficient=cfg.drag
    )
    data["energy_histories"] = history_panel(
        name, [0.9, 1.0, 1.0002, 1.05], cfg, elastic_limit
    )
    with _partial_path(name).open("w") as handle:
        json.dump(data, handle, indent=1, default=float, sort_keys=True)
    el = data["elastic_regime"]
    print(
        f"[{name}] omega={el['omega_max_rad_s']:.1f} rad/s "
        f"dt_crit={el['predicted_dt_crit_s']:.6g} s "
        f"onset={el['measured_onset_ratio']:.6f} "
        f"growth={data['growth_just_above_threshold']['measured_growth_factor']:.6f}"
        f"/{data['growth_just_above_threshold']['predicted_growth_factor']:.6f} "
        f"damped={data['damping_regime']['measured_onset_ratio']:.6f} "
        f"refined={data['damping_regime']['refined_onset_ratio']:.6f} "
        f"bound/exact={data['row_sum_bound']['bound_over_exact']:.4f}",
        flush=True,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict = {"configurations": [], "energy_histories": [], "figures": []}
    rows: list[list] = []

    for name, _, _ in CONFIGS:
        path = _partial_path(name)
        if not path.exists():
            raise SystemExit(
                f"missing {path}; run: python3 {__file__} --only {name}"
            )
        with path.open() as handle:
            data = json.load(handle)
        payload["configurations"].append(data)
        payload["energy_histories"].append(data["energy_histories"])

        el = data["elastic_regime"]
        gr = data["growth_just_above_threshold"]
        ma = data["marginal_step"]
        da = data["damping_regime"]
        bo = data["row_sum_bound"]
        rows.append([
            name,
            f"{el['omega_max_rad_s']:.6g}",
            f"{el['predicted_dt_crit_s']:.6g}",
            f"{el['measured_onset_ratio']:.6f}",
            f"{el['relative_error'] * 100:+.4f}",
            f"{gr['predicted_growth_factor']:.6f}",
            f"{gr['measured_growth_factor']:.6f}",
            f"{gr['relative_error'] * 100:+.4f}",
            f"{ma['energy_ratio_second_half_over_first']:.4f}",
            f"{da['refined_onset_ratio']:.6f}",
            f"{da['error_vs_refined'] * 100:+.4f}",
            f"{bo['bound_over_exact']:.4f}",
            str(data["constructor_guard"]["refused"]),
        ])


    # --- figure 1: energy histories --------------------------------------
    panel = payload["energy_histories"][0]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    styles = {0.9: "-", 1.0: "--", 1.0002: "-.", 1.05: ":"}
    for label, curve in panel["curves"].items():
        energies = curve["energies_over_initial"]
        steps = list(range(len(energies)))
        ax.semilogy(
            steps,
            [max(v, 1e-12) for v in energies],
            styles.get(curve["ratio"], "-"),
            linewidth=1.4,
            label=f"dt/dt_crit = {label}",
        )
    ax.axhline(1.0, color="k", linewidth=0.6)
    ax.set_xlabel("step")
    ax.set_ylabel(r"mechanical energy / $E_0$")
    ax.set_title("Engine energy at fixed ratios of the closed-form threshold\n"
                 "(4$\\times$4$\\times$4 bonded lattice, 64 particles, no damping)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "energy_histories.png", dpi=140)
    plt.close(fig)
    payload["figures"].append("energy_histories.png")

    # --- figure 2: measured onset / growth vs closed form -----------------
    names = [row[0] for row in rows]
    propagations = [float(row[3]) for row in rows]
    ref_onset = [1.0 for _ in rows]
    refined_damped = [float(row[9]) for row in rows]
    bound = [float(row[11]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    ax = axes[0]
    index = range(len(names))
    ax.bar([i - 0.22 for i in index], propagations, width=0.22,
           label="measured elastic onset / dt_crit")
    ax.bar([i for i in index], ref_onset, width=0.22, label="closed form (2/omega_max)")
    ax.bar([i + 0.22 for i in index], refined_damped, width=0.22,
           label="refined damping onset / (2m/c)")
    ax.set_xticks(list(index))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0.95, 1.01)
    ax.set_ylabel("onset ratio")
    ax.set_title("Divergence onset vs closed form")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(list(index), bound, width=0.4, color="tab:orange",
           label="row-sum bound / exact omega_max")
    ax.axhline(1.0, color="k", linewidth=0.8, label="exact")
    ax.set_xticks(list(index))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("bound / exact")
    ax.set_title("Conservativeness of the criterion the solver uses")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "onset_comparison.png", dpi=140)
    plt.close(fig)
    payload["figures"].append("onset_comparison.png")

    with (OUT / "threshold_report.json").open("w") as handle:
        json.dump(payload, handle, indent=1, default=float, sort_keys=True)

    header = [
        "config", "omega_max_rad_s", "dt_crit_s", "measured_onset_ratio",
        "onset_error_pct", "growth_predicted", "growth_measured",
        "growth_error_pct", "marginal_energy_ratio", "refined_damping_onset",
        "error_vs_refined_pct", "row_sum_over_exact", "guard_refused",
    ]
    with (OUT / "threshold_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    import sys

    if "--only" in sys.argv:
        wanted = sys.argv[sys.argv.index("--only") + 1]
        for name, kwargs, steps in CONFIGS:
            if name == wanted:
                run_one(name, kwargs, steps)
                break
        else:
            raise SystemExit(f"unknown configuration {wanted!r}")
    else:
        main()