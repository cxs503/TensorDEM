"""Re-audit of the 4x4x3 opposed-platen specimen: what is the peak made of?

Purpose
-------
``docs/ice_bond_specimen_rate_dependence.md`` (mission m6464) treated the maximum
axial stress of an opposed-platen run as the specimen strength and concluded the
emergent exponent was 0.244 (linear+creep) / 0.198 (maxwell_wiechert+creep)
versus the analytic 1/(m+1) = 1/3, with the purely elastic negative control not
flat either (+0.117).  It attributed the discrepancy to inertia and recorded an
honest FAIL.

Nothing measured where in the history that maximum occurs, how many bonds the
gauge section actually has, or whether the stress is carried by bonds at all.
This script measures all three, at the same cost as the original campaign:

Part A -- load-path integrity.  Drive one specimen step by step and compare the
    platen reaction with the axial force transmitted by the bonds crossing the
    gauge plane.  A bonded specimen has no other load path, so the ratio must be
    1 while the bonds are intact; the crossing-bond count and their axial
    projections also give the *exact* rigid-plate bound
    (:func:`tensordem.specimen_attribution.bond_path_capacity`).

Part B -- peak attribution.  Re-run the campaign with the recorded
    ``broken_bonds`` counter attached to every stress sample and separate the
    pre-failure service strength from the post-failure jamming peak.

Part C -- the inertia numbers.  Report the local platen-speed ratio the original
    report used *and* the specimen-scale wave-transit ratio, because they
    disagree and only the latter decides whether the test is quasi-static.

Every number in the JSON and in the report is produced here.

Usage
-----
    PYTHONPATH=src:. python scripts/verify_ice_bond_specimen_attribution.py
    PYTHONPATH=src:. python scripts/verify_ice_bond_specimen_attribution.py --quick
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensordem.dem3d_triaxial import TriaxialRunner
from tensordem.specimen_attribution import (
    bond_path_capacity,
    classify_peak,
    inertia_numbers,
    load_path_integrity_ratio,
)
from scripts.verify_ice_bond_specimen_rate import (
    BREAKING_STRAIN,
    BOND_STIFFNESS,
    NX,
    NY,
    NZ,
    RADIUS,
    STRAIN_RATES,
    arm_configs,
)

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "ice_bond_specimen_attribution"
ARMS = ("linear", "linear+creep", "maxwell_wiechert", "maxwell_wiechert+creep")
DENSITY = 917.0                      # kg/m^3, ice
DT_HINT = 3.4e-5                     # s, same hint as the original campaign
GAUGE_PLANE_Z = 1.5 * RADIUS         # between layer 1 (z=0.05) and 2 (z=0.10)
SPECIMEN_HEIGHT = NZ * 2.0 * RADIUS  # m
GROSS_AREA = (NX * 2.0 * RADIUS) * (NY * 2.0 * RADIUS)


def crossing_bonds(sim) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask and axial projections of the bonds crossing the gauge plane."""
    i, j = sim.pairs.T
    z = sim.initial_positions[:, 2]
    dz = (z[j] - z[i])
    mask = ((z[i] - GAUGE_PLANE_Z) * (z[j] - GAUGE_PLANE_Z)) < 0
    return mask, dz


def capacity(sim):
    mask, dz = crossing_bonds(sim)
    return bond_path_capacity(
        dz[mask].tolist(), bond_stiffness_N_m=BOND_STIFFNESS,
        breaking_strain=BREAKING_STRAIN, gross_area_m2=GROSS_AREA)


def naive_m6464_bound_Pa() -> float:
    """The bound used by m6464: nx * ny bonds of k_u eps_b L0 over the gross area."""
    return BOND_STIFFNESS * BREAKING_STRAIN * 2.0 * RADIUS * (NX * NY) / GROSS_AREA


def load_path_probe(rate: float, stride: int = 40) -> dict:
    """Part A: platen reaction vs the axial bond force across the gauge plane."""
    cfg = arm_configs("linear", rate, dt_hint=DT_HINT)
    runner = TriaxialRunner(cfg)
    sim = runner.sim
    i, j = sim.pairs.T
    mask, dz = crossing_bonds(sim)
    ratios: list[float] = []
    samples: list[dict] = []
    for step in range(cfg.max_axial_steps):
        sim.step()
        intact = bool(sim.alive.all())
        if step < max(200, cfg.max_axial_steps // 20) or step % stride or not intact:
            continue          # skip the platen take-up transient
        d = sim.positions[j] - sim.positions[i]
        length = torch.linalg.vector_norm(d, dim=1)
        extension = length - sim.rest_lengths
        cos = (d[:, 2] / length).abs()
        axial = float((BOND_STIFFNESS * extension * cos)[mask].sum())
        reaction = float(sim.last_top_platen_reaction_z)
        if axial == 0.0:
            continue
        ratios.append(load_path_integrity_ratio(reaction, axial))
        if len(samples) < 400:
            samples.append({"step": step, "time_s": float(sim.time),
                            "platen_reaction_N": reaction, "bond_axial_N": axial,
                            "max_bond_strain": float((extension / sim.rest_lengths).max())})
    cap = capacity(sim)
    n_broken_first = None
    return {
        "strain_rate_s-1": float(rate),
        "n_bonds_total": int(len(sim.pairs)),
        "n_bonds_crossing_gauge": cap.n_bonds,
        "bond_axial_projection_m": sorted(
            {round(abs(float(v)), 6) for v in dz[mask]}),
        "capacity_force_N": cap.capacity_force_N,
        "capacity_stress_Pa": cap.capacity_stress_Pa,
        "naive_m6464_bound_Pa": naive_m6464_bound_Pa(),
        "bound_undercount_factor": cap.capacity_stress_Pa / naive_m6464_bound_Pa(),
        "integrity_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "integrity_ratio_min": min(ratios) if ratios else None,
        "integrity_ratio_max": max(ratios) if ratios else None,
        "integrity_n_samples": len(ratios),
        "samples": samples,
        "first_break_bonds": n_broken_first,
    }


def run_one(arm: str, rate: float) -> dict:
    cfg = arm_configs(arm, rate, dt_hint=DT_HINT)
    runner = TriaxialRunner(cfg)
    record = runner.run()
    rows = [r for r in record["rows"] if r["phase"] == "axial"] or record["rows"]
    history = [
        {"axial_engineering_strain": float(r["axial_engineering_strain"]),
         "axial_stress_Pa": float(r["axial_stress_Pa"]),
         "broken_bonds": int(r["broken_bonds"])}
        for r in rows
    ]
    cap = capacity(runner.sim)
    verdict = classify_peak(history, cap.capacity_stress_Pa)
    inertia = inertia_numbers(
        strain_rate=rate, breaking_strain=BREAKING_STRAIN,
        specimen_height_m=SPECIMEN_HEIGHT, rest_length_m=2.0 * RADIUS,
        bond_stiffness_N_m=BOND_STIFFNESS, radius_m=RADIUS, density=DENSITY)
    return {
        "arm": arm,
        "strain_rate_s-1": float(rate),
        "n_particles": int(len(runner.sim.positions)),
        "n_bonds": int(len(runner.sim.pairs)),
        "n_steps": int(record["n_steps"]),
        "dt_s": float(runner.sim.dt),
        "platen_speed_m_s": float(cfg.loading_speed),
        "sigma_m6464_Pa": float(record["sigma1_Pa"]),
        "strain_at_m6464_peak": float(record["strain_at_peak"]),
        "verdict": verdict,
        "m6464_bound_over_capacity": naive_m6464_bound_Pa() / cap.capacity_stress_Pa,
        "inertia": {
            "bond_wave_speed_m_s": inertia.bond_wave_speed_m_s,
            "bond_period_s": inertia.bond_period_s,
            "platen_speed_ratio": inertia.platen_speed_ratio,
            "loading_time_ratio": inertia.loading_time_ratio,
            "transit_ratio": inertia.transit_ratio,
            "quasi_static": inertia.quasi_static,
        },
        "history": history[:: max(1, len(history) // 120)],
        "history_full_points": len(history),
    }


def power_law_slope(rates, values) -> float:
    xs = [math.log10(r) for r in rates]
    ys = [math.log10(v) for v in values]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")


def summarise(runs: list[dict], probe: dict | None) -> dict:
    by_arm: dict[str, dict] = {}
    for arm in ARMS:
        rows = sorted([r for r in runs if r["arm"] == arm],
                      key=lambda r: r["strain_rate_s-1"])
        if not rows:
            continue
        rates = [r["strain_rate_s-1"] for r in rows]
        peaks = [r["verdict"]["peak_stress_Pa"] for r in rows]
        pre = [r["verdict"]["prefailure_stress_Pa"] for r in rows]
        complete = all(v is not None and v > 0 for v in pre)
        by_arm[arm] = {
            "strain_rates_s-1": rates,
            "peak_stress_Pa": peaks,
            "broken_bonds_at_peak": [r["verdict"]["broken_bonds_at_peak"] for r in rows],
            "peak_over_capacity": [r["verdict"]["peak_over_capacity"] for r in rows],
            "prefailure_stress_Pa": pre,
            "prefailure_over_capacity": [r["verdict"]["prefailure_over_capacity"] for r in rows],
            "all_peaks_postfailure": all(not r["verdict"]["peak_is_prefailure"] for r in rows),
            "m6464_exponent": power_law_slope(rates, peaks),
            "prefailure_exponent": power_law_slope(rates, pre) if complete else None,
            "prefailure_spread_relative": (
                None if not complete else (max(pre) - min(pre)) / min(pre)),
        }
    slopes = [v["m6464_exponent"] for v in by_arm.values()]
    pre_slopes = [v["prefailure_exponent"] for v in by_arm.values()
                  if v["prefailure_exponent"] is not None]
    return {
        "per_arm": by_arm,
        "exponent_spread_across_arms": max(slopes) - min(slopes) if slopes else None,
        "prefailure_exponent_spread_across_arms": (
            None if len(pre_slopes) < 2 else max(pre_slopes) - min(pre_slopes)),
        "every_peak_postfailure": all(v["all_peaks_postfailure"] for v in by_arm.values()),
        "load_path": (None if probe is None else {
            "n_bonds_crossing_gauge": probe["n_bonds_crossing_gauge"],
            "capacity_stress_Pa": probe["capacity_stress_Pa"],
            "naive_m6464_bound_Pa": probe["naive_m6464_bound_Pa"],
            "bound_undercount_factor": probe["bound_undercount_factor"],
            "integrity_ratio_mean": probe["integrity_ratio_mean"],
            "integrity_ratio_min": probe["integrity_ratio_min"],
            "integrity_ratio_max": probe["integrity_ratio_max"],
        }),
    }


def figures(runs: list[dict], summary: dict, probe: dict, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    cap = probe["capacity_stress_Pa"]
    naive = probe["naive_m6464_bound_Pa"]
    arms = [a for a in ARMS if any(r["arm"] == a for r in runs)]

    # 1) Part A: load-path integrity
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    s = probe["samples"]
    ax.plot([p["time_s"] for p in s], [p["platen_reaction_N"] for p in s], lw=1.5,
            label="platen reaction (measured load)")
    ax.plot([p["time_s"] for p in s], [p["bond_axial_N"] for p in s], lw=1.2, ls="--",
            label="axial force of bonds crossing gauge")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("axial force (N)")
    ax.set_title(f"Load path is the bond network only: ratio "
                 f"{probe['integrity_ratio_mean']:.4f}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "load_path_integrity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    names.append(p.name)

    # 2) stress-strain with the correct and the m6464 bound
    n = len(arms)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.0), sharex=True, squeeze=False)
    for ax, arm in zip(axes[0], arms):
        for r in sorted([x for x in runs if x["arm"] == arm],
                        key=lambda x: x["strain_rate_s-1"]):
            h = r["history"]
            ax.plot([q["axial_engineering_strain"] for q in h],
                    [q["axial_stress_Pa"] for q in h], lw=1.2,
                    label=f"edot={r['strain_rate_s-1']:g}")
        ax.axhline(cap, color="k", ls="--", lw=1.2, label=f"bond bound {cap:.0f} Pa")
        ax.axhline(naive, color="r", ls=":", lw=1.2, label=f"m6464 bound {naive:.0f} Pa")
        ax.axvline(BREAKING_STRAIN, color="grey", ls=":", lw=1.0)
        ax.set_title(arm, fontsize=9)
        ax.set_xlabel("reported strain")
        ax.set_ylabel("axial stress (Pa)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5)
    fig.suptitle("Peaks sit above the bond bound because they are post-failure "
                 "jamming peaks", fontsize=11)
    fig.tight_layout()
    p = out_dir / "stress_strain_bounds.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    names.append(p.name)

    # 3) broken bonds at the peak
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for arm, style in zip(arms, ("-", "--", "-.", ":")):
        r = max([x for x in runs if x["arm"] == arm], key=lambda x: x["strain_rate_s-1"])
        h = r["history"]
        ax.plot([q["axial_engineering_strain"] for q in h],
                [q["broken_bonds"] for q in h], style, lw=1.4,
                label=f"{arm} (edot={r['strain_rate_s-1']:g})")
        v = r["verdict"]
        ax.plot([v["peak_strain"]], [v["broken_bonds_at_peak"]], "o", ms=5)
    ax.set_xlabel("reported strain")
    ax.set_ylabel("broken bonds")
    ax.set_title("Bonds already consumed when the recorded peak occurs "
                 "(dots = peaks)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "broken_bonds_at_peak.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    names.append(p.name)

    # 4) peak vs pre-failure strength
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for arm, marker in zip(arms, ("o", "s", "^", "D")):
        rows = sorted([x for x in runs if x["arm"] == arm],
                      key=lambda x: x["strain_rate_s-1"])
        ax.loglog([r["strain_rate_s-1"] for r in rows],
                  [r["verdict"]["peak_stress_Pa"] for r in rows], marker + "-",
                  label=f"{arm}: peak (m6464 variable)")
        pre = [(r["strain_rate_s-1"], r["verdict"]["prefailure_stress_Pa"]) for r in rows
               if r["verdict"]["prefailure_stress_Pa"]]
        if pre:
            ax.loglog([x[0] for x in pre], [x[1] for x in pre], marker + ":",
                      alpha=0.75, label=f"{arm}: pre-failure strength")
    ax.axhline(cap, color="k", ls="--", lw=1.2, label=f"bond bound {cap:.0f} Pa")
    ax.set_xlabel("strain rate (1/s)")
    ax.set_ylabel("axial stress (Pa)")
    ax.set_title("Dependent variable of m6464 vs the pre-failure service stress",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    p = out_dir / "peak_vs_prefailure.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    names.append(p.name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    arms = ARMS[:1] if args.quick else ARMS
    rates = STRAIN_RATES[-1:] if args.quick else STRAIN_RATES

    # Probe in the quasi-static window: at the fastest rate the specimen
    # needs >4 wave transits, so single-sample ratios carry the inertia.
    probe = load_path_probe(min(rates))
    print(f"Part A: {probe['n_bonds_crossing_gauge']} bonds cross the gauge plane, "
          f"capacity {probe['capacity_stress_Pa']:.0f} Pa "
          f"({probe['bound_undercount_factor']:.2f}x the m6464 bound), "
          f"integrity {probe['integrity_ratio_mean']:.4f}")

    runs: list[dict] = []
    for arm in arms:
        for rate in rates:
            row = run_one(arm, rate)
            runs.append(row)
            v = row["verdict"]
            pre = v["prefailure_stress_Pa"]
            print(f"{arm:26s} edot={rate:6.3f}  peak={v['peak_stress_Pa']:9.1f} Pa "
                  f"({v['peak_over_capacity']:5.2f}x cap, broken={v['broken_bonds_at_peak']:3d})"
                  f"  pre-failure={pre and round(pre, 1)} Pa "
                  f"({(v['prefailure_over_capacity'] or 0):5.2f}x cap)")

    summary = summarise(runs, probe)
    out_dir = OUT / "quick" if args.quick else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_names = [] if args.no_figures else figures(runs, summary, probe, out_dir)
    payload = {
        "campaign": "ice_bond_specimen_attribution",
        "specimen": {"nx": NX, "ny": NY, "nz": NZ, "radius_m": RADIUS,
                     "unrelaxed_bond_stiffness_N_m": BOND_STIFFNESS,
                     "breaking_strain": BREAKING_STRAIN, "density_kg_m3": DENSITY,
                     "dt_hint_s": DT_HINT, "gauge_plane_z_m": GAUGE_PLANE_Z,
                     "gross_area_m2": GROSS_AREA},
        "load_path": probe,
        "summary": summary,
        "runs": runs,
        "figures": fig_names,
    }
    path = out_dir / ("quick.json" if args.quick else "verification.json")
    path.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {path}")
    print(f"every peak post-failure: {summary['every_peak_postfailure']}")
    for arm, s in summary["per_arm"].items():
        print(f"  {arm:26s} m6464 exponent {s['m6464_exponent']:+.3f}   "
              f"pre-failure exponent {s['prefailure_exponent']}   "
              f"broken@peak {s['broken_bonds_at_peak']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())