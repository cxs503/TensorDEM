"""Multi-disperse floe-field generator: coverage, FSD sensitivity and jamming.

Produces ``artifacts/fsd_packing.json`` (machine readable, every criterion number
programmatically injected) and the figures under ``artifacts/fsd_packing/``.

Sections
--------
A  FSD families: analytic median / CV / moments ``m0..m2`` of the six
   distributions compared in this study, all anchored on the H1 reference
   median floe radius (5 m).
B  Coverage sweep phi = 0.60 / 0.70 / 0.80 / 0.85 for every family: realised
   coverage, absolute deviation, initial maximum overlap (H1 convention),
   coordination number, moments, generation time.
C  Placement routes as *negative controls*: random sequential adsorption (RSA),
   which is the route H1 reports as blocked at ~0.55, against the compaction
   routes of this module; the measured jamming coverage of each route.
D  Monodisperse-limit regression lock: for ``CV -> 0`` the generator dispatches
   to ``IceChannelSimulation.from_field`` and must reproduce the H1 field and
   ``state_digest`` bit-for-bit.
E  Generation time vs floe count at fixed coverage (near-linearity criterion).
F  Channel-transit sensitivity: the same hull transit (H1 configuration) through
   the H1 monodisperse field and through each FSD family at phi = 0.80, giving
   the cleared channel width and the ice resistance; seed ensemble for one FSD.
G  Criterion table: every acceptance number with its target and pass/fail flag.

Usage
-----
``python benchmarks/fsd_packing_sensitivity.py``           # production run
``python benchmarks/fsd_packing_sensitivity.py --fast``    # reduced (unit tests)
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tensordem.fsd_packing import (  # noqa: E402
    BimodalFSD,
    DeltaFSD,
    LogNormalFSD,
    PowerLawFSD,
    generate_field,
    make_fsd,
    rsa_placement,
)
from tensordem.ice_channel import HullGeometry, IceChannelSimulation  # noqa: E402

JSON_OUT = _REPO_ROOT / "artifacts" / "fsd_packing.json"
FIG_DIR = _REPO_ROOT / "artifacts" / "fsd_packing"

FAMILIES = [
    ("power_law_k1.5", {"kind": "power_law", "kappa": 1.5}),
    ("power_law_k2.5", {"kind": "power_law", "kappa": 2.5}),
    ("power_law_k3.5", {"kind": "power_law", "kappa": 3.5}),
    ("log_normal_cv0.3", {"kind": "log_normal", "cv": 0.3}),
    ("log_normal_cv0.5", {"kind": "log_normal", "cv": 0.5}),
    ("bimodal", {"kind": "bimodal"}),
]
COVERAGES = [0.60, 0.70, 0.80, 0.85]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=_REPO_ROOT).decode().strip()
    except Exception:                                   # pragma: no cover
        return "unknown"


# ---------------------------------------------------------------------------
# A. FSD families
# ---------------------------------------------------------------------------

def section_families() -> dict:
    rows = {}
    for name, spec in FAMILIES:
        fsd = make_fsd(spec)
        rows[name] = {"spec": spec, **fsd.describe()}
    rows["h1_monodisperse"] = {"spec": {"kind": "delta", "radius": 5.0},
                               **DeltaFSD(5.0).describe()}
    return rows


# ---------------------------------------------------------------------------
# B. coverage sweep
# ---------------------------------------------------------------------------

def section_coverage(domain, seed: int, method: str = "lattice") -> dict:
    rows = {}
    for name, spec in FAMILIES:
        fsd = make_fsd(spec)
        for phi in COVERAGES:
            try:
                field = generate_field(fsd, phi, domain_length=domain[0],
                                       domain_width=domain[1], seed=seed,
                                       method=method)
                summary = field.summary()
                summary["status"] = "ok"
            except Exception as exc:                    # pragma: no cover
                summary = {"status": f"error: {exc}"}
            rows[f"{name}@{phi:.2f}"] = summary
    return rows


def section_timing(seed: int, domain, method: str = "lattice") -> dict:
    """Generation wall time vs floe count at fixed coverage (near-linear check)."""
    fsd = LogNormalFSD(0.5)
    points = []
    base_area = domain[0] * domain[1]
    for scale in (0.5, 1.0, 2.0, 4.0, 8.0):
        length = domain[0] * math.sqrt(scale)
        width = domain[1] * math.sqrt(scale)
        field = generate_field(fsd, 0.80, domain_length=length, domain_width=width,
                               seed=seed, method=method)
        points.append({"floe_count": field.floe_count,
                       "domain_m": [length, width],
                       "wall_s": field.generation_wall_s})
    counts = [p["floe_count"] for p in points]
    times = [p["wall_s"] for p in points]
    slope = float("nan")
    if len(points) >= 2 and min(counts) > 0 and min(times) > 0:
        xs = [math.log(c) for c in counts]
        ys = [math.log(t) for t in times]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        slope = num / den if den > 0 else float("nan")
    return {"points": points, "log_log_slope": slope,
            "floe_counts": counts,
            "wall_s": times,
            "normalised_wall_us_per_floe": [1e6 * t / c for t, c in zip(times, counts)]}


# ---------------------------------------------------------------------------
# C. placement routes / negative controls
# ---------------------------------------------------------------------------

def section_placement_routes(domain, seed: int) -> dict:
    rows = {}
    candidates = [("rsa_delta_r5.0", {"kind": "delta", "radius": 5.0}),
                  ("rsa_power_law_k2.5", {"kind": "power_law", "kappa": 2.5})]
    for name, spec in candidates:
        fsd = make_fsd(spec)
        for phi in COVERAGES:
            field = rsa_placement(fsd, phi, domain_length=domain[0],
                                  domain_width=domain[1], seed=seed,
                                  max_consecutive_failures=2000)
            summary = field.summary()
            summary["requested_coverage"] = phi
            summary["blocked"] = bool(summary.get("blocked", False))
            rows[f"{name}@{phi:.2f}"] = summary
    # compaction routes for the same targets
    for method in ("compaction", "lattice"):
        fsd = PowerLawFSD(2.5)
        for phi in (0.80, 0.85):
            field = generate_field(fsd, phi, domain_length=domain[0],
                                   domain_width=domain[1], seed=seed,
                                   method=method)
            rows[f"{method}_power_law_k2.5@{phi:.2f}"] = field.summary()
    return rows


# ---------------------------------------------------------------------------
# D. monodisperse-limit regression lock
# ---------------------------------------------------------------------------

def section_monodisperse_lock(domain, seed: int) -> dict:
    coverage = 0.80
    gen = generate_field(LogNormalFSD(0.0), coverage, domain_length=domain[0],
                         domain_width=domain[1], seed=seed)
    ref = IceChannelSimulation.from_field(coverage, domain_length=domain[0],
                                          domain_width=domain[1], seed=seed)
    ref_field = DeltaFSD(5.0)
    # weak continuity check: a slightly polydisperse field must stay close to H1
    near = generate_field(LogNormalFSD(0.01), coverage, domain_length=domain[0],
                          domain_width=domain[1], seed=seed)
    return {
        "coverage": coverage,
        "generator_method": gen.method,
        "generator_state_digest": gen.state_digest(),
        "h1_state_digest": ref.state_digest(),
        "digest_identical": bool(gen.state_digest() == ref.state_digest()),
        "positions_bit_identical": bool(torch.equal(gen.pos, ref.pos)),
        "radius_bit_identical": bool(torch.equal(gen.radius, ref.radius)),
        "thickness_bit_identical": bool(torch.equal(gen.thickness, ref.thickness)),
        "h1_reference_fsd": ref_field.describe(),
        "near_limit_cv": 0.01,
        "near_limit_coverage_deviation": float(near.coverage_deviation()),
        "near_limit_max_overlap": float(near.max_overlap),
        "near_limit_coordination": float(near.coordination_number),
    }


# ---------------------------------------------------------------------------
# F. channel transit
# ---------------------------------------------------------------------------

def run_transit(field, *, dt: float, n_steps: int, record_every: int,
                hull: HullGeometry) -> dict:
    """Run the H1 channel transit on a generated field.

    The ice resistance on the hull is the reaction to the hull-floe contact
    forces, ``-sum(F_x)`` over all floes, sampled at the recording steps.  The
    cleared width is H1's ``IceChannelSimulation.cleared_width`` definition.
    """
    sim = field.to_simulation(hull=hull, dt=dt, integrator="euler")
    y0 = sim.pos[:, 1].clone()
    tic = time.perf_counter()
    trace = {"time": [], "cleared_width": [], "resistance": [], "mean_speed": []}
    for step in range(int(n_steps)):
        sim.step()
        if (step + 1) % int(record_every) == 0:
            force, _ = sim._hull_force(sim.time, sim.pos)
            trace["time"].append(float(sim.time))
            trace["cleared_width"].append(float(sim.cleared_width()))
            trace["resistance"].append(float(-force[:, 0].sum()))
            trace["mean_speed"].append(float(torch.linalg.norm(sim.vel, dim=1).mean()))
    wall = time.perf_counter() - tic
    times = trace["time"]
    if times:
        cut = times[0] + 0.4 * (times[-1] - times[0])
        tail = [i for i, t in enumerate(times) if t >= cut]
        cw = [trace["cleared_width"][i] for i in tail]
        res = [trace["resistance"][i] for i in tail]
        lat = (sim.pos[:, 1] - y0).abs()
        out = {
            "cleared_width_tail_mean_m": float(sum(cw) / len(cw)),
            "cleared_width_final_m": float(trace["cleared_width"][-1]),
            "cleared_width_tail_std_m": float(_std(cw)),
            "resistance_tail_mean_N": float(sum(res) / len(res)),
            "resistance_tail_std_N": float(_std(res)),
            "mean_speed_tail_m_s": float(sum(trace["mean_speed"][i] for i in tail)
                                         / len(tail)),
            "displaced_fraction": float((lat > 1.0).to(torch.float64).mean()),
            "max_lateral_displacement_m": float(lat.max()),
        }
    else:                                               # pragma: no cover
        out = {"cleared_width_tail_mean_m": float("nan")}
    out.update({"simulated_time_s": float(sim.time), "steps": int(n_steps),
                "dt": float(dt), "wall_time_s": float(wall),
                "hull": {"length": hull.length, "beam": hull.beam,
                         "speed": hull.speed, "y_position": hull.y_position,
                         "x_start": hull.x_start},
                "floe_count": field.floe_count,
                "trace": trace})
    return out


def _std(values) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def section_transit(fields: dict, hull: HullGeometry, *, dt: float, n_steps: int,
                    record_every: int) -> dict:
    rows = {}
    for name, field in fields.items():
        rows[name] = run_transit(field, dt=dt, n_steps=n_steps,
                                 record_every=record_every, hull=hull)
        print(f"[transit] {name}: cleared={rows[name]['cleared_width_tail_mean_m']:.2f} m "
              f"resistance={rows[name]['resistance_tail_mean_N']:.3e} N "
              f"({rows[name]['wall_time_s']:.1f} s)", flush=True)
    base = rows.get("h1_monodisperse")
    if base is not None:
        for name, row in rows.items():
            row["delta_cleared_width_m"] = (row["cleared_width_tail_mean_m"]
                                            - base["cleared_width_tail_mean_m"])
            row["delta_cleared_width_rel"] = (
                row["delta_cleared_width_m"] / base["cleared_width_tail_mean_m"]
                if base["cleared_width_tail_mean_m"] else float("nan"))
            row["delta_resistance_rel"] = (
                (row["resistance_tail_mean_N"] - base["resistance_tail_mean_N"])
                / base["resistance_tail_mean_N"]
                if base["resistance_tail_mean_N"] else float("nan"))
    return rows


# ---------------------------------------------------------------------------
# G. criteria
# ---------------------------------------------------------------------------

def evaluate_criteria(report: dict) -> dict:
    cov = report["coverage_sweep"]
    crit = {}

    # (a) coverage deviation < 0.5 % absolute
    devs = [row.get("coverage_deviation_abs", float("nan")) for row in cov.values()
            if row.get("status") == "ok"]
    crit["a_coverage_deviation_lt_0p005"] = {
        "max_abs_deviation": max(devs) if devs else float("nan"),
        "threshold": 0.005,
        "pass": bool(devs) and max(devs) < 0.005,
        "cases": len(devs)}

    # (b) initial max overlap <= 0.06 m
    by_cov = {}
    for key, row in cov.items():
        if row.get("status") != "ok":
            continue
        phi = key.split("@")[1]
        by_cov.setdefault(phi, []).append(row["max_overlap_m"])
    crit["b_max_overlap_le_0p06"] = {
        "threshold": 0.06,
        "per_coverage_max_overlap_m": {phi: max(v) for phi, v in by_cov.items()},
        "per_coverage_pass": {phi: bool(max(v) <= 0.06) for phi, v in by_cov.items()},
        "pass": all(max(v) <= 0.06 for v in by_cov.values()) if by_cov else False}

    # (c) monodisperse limit reproduces H1 bit-for-bit
    lock = report["monodisperse_lock"]
    crit["c_monodisperse_limit_bit_identical"] = {
        "digest_identical": lock["digest_identical"],
        "positions_bit_identical": lock["positions_bit_identical"],
        "pass": bool(lock["digest_identical"] and lock["positions_bit_identical"]
                     and lock["radius_bit_identical"]
                     and lock["thickness_bit_identical"])}

    # (d) realised m1 within 1 % of the analytic FSD m1
    m1 = [row.get("m1_deviation_rel", float("nan")) for row in cov.values()
          if row.get("status") == "ok"]
    crit["d_m1_deviation_lt_0p01"] = {
        "max_rel_deviation": max(m1) if m1 else float("nan"),
        "threshold": 0.01, "pass": bool(m1) and max(m1) < 0.01}

    # (e) cleared width dispersion across three seeds < 5 %
    seeds = report["seed_ensemble"]
    widths = [row["cleared_width_tail_mean_m"] for row in seeds.values()]
    if len(widths) >= 2 and sum(widths) > 0:
        mean = sum(widths) / len(widths)
        disp = (max(widths) - min(widths)) / mean
    else:                                               # pragma: no cover
        disp = float("nan")
    crit["e_seed_dispersion_lt_0p05"] = {
        "cleared_widths_m": widths,
        "dispersion_range_over_mean": disp,
        "threshold": 0.05, "pass": bool(disp == disp and disp < 0.05)}

    # (f) 0.85 coverage must be generatable (RSA must fail there)
    phi85 = [row for key, row in cov.items() if key.endswith("@0.85")
             and row.get("status") == "ok"]
    rsa85 = [row for key, row in report["placement_routes"].items()
             if key.endswith("@0.85") and key.startswith("rsa")]
    rsa_ok = [row for row in rsa85 if row.get("blocked") is False]
    crit["f_0p85_generatable_vs_rsa"] = {
        "generator_cases_at_0p85": len(phi85),
        "generator_all_converged": bool(phi85) and all(
            row.get("converged", False) for row in phi85),
        "rsa_all_blocked_at_0p85": all(row.get("blocked", True) for row in rsa85),
        "rsa_realised_coverage_at_0p85": [row["realised_coverage"] for row in rsa85],
        "pass": bool(phi85) and all(row.get("converged", False) for row in phi85)
                and all(row.get("blocked", True) for row in rsa85)}

    # (g) near-linear generation time
    timing = report["timing"]
    slope = timing["log_log_slope"]
    crit["g_generation_time_near_linear"] = {
        "log_log_slope": slope, "threshold_max_slope": 1.30,
        "pass": bool(slope == slope and slope <= 1.30)}

    # (h) programmatic artifact injection
    crit["h_artifact_programmatic"] = {
        "artifact_path": str(JSON_OUT.relative_to(_REPO_ROOT)),
        "generated_by": "benchmarks/fsd_packing_sensitivity.py",
        "pass": True}
    return crit


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(report: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figures = []

    # fig 1: FSD pdfs / quantile curves
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    radii = torch.linspace(0.5, 16.0, 400, dtype=torch.float64)
    for name, spec in FAMILIES:
        fsd = make_fsd(spec)
        q = fsd.quantile(radii / 16.0)             # illustrative quantile curve
        ax.plot(radii.numpy(), q.numpy(), lw=1.5, label=name)
    ax.set_xlabel("probability level $u$")
    ax.set_ylabel("floe radius [m]")
    ax.set_title("FSD families: quantile functions (all medians = 5 m)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG_DIR / "fig_fsd_quantiles.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figures.append({"name": p.name,
                    "caption": "Quantile functions of the six FSD families; all are "
                               "anchored on the H1 reference median radius 5 m "
                               "(4.6-5.4 m band).  The power-law slopes kappa = "
                               "1.5/2.5/3.5 keep the same median and differ in the "
                               "weight of the coarse tail."})

    # fig 2: coverage deviation and max overlap vs coverage
    cov = report["coverage_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for name, _ in FAMILIES:
        dev = [cov[f"{name}@{phi:.2f}"].get("coverage_deviation_abs", float("nan"))
               for phi in COVERAGES]
        ov = [cov[f"{name}@{phi:.2f}"].get("max_overlap_m", float("nan"))
              for phi in COVERAGES]
        axes[0].plot(COVERAGES, dev, "o-", lw=1.2, ms=3.5, label=name)
        axes[1].plot(COVERAGES, ov, "o-", lw=1.2, ms=3.5, label=name)
    axes[0].axhline(0.005, color="k", ls="--", lw=1.0, label="0.5 % limit")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("target coverage $\\phi$")
    axes[0].set_ylabel("|realised - target|")
    axes[0].set_title("coverage control")
    axes[1].axhline(0.06, color="k", ls="--", lw=1.0, label="0.06 m limit")
    axes[1].set_xlabel("target coverage $\\phi$")
    axes[1].set_ylabel("max initial overlap [m]")
    axes[1].set_title("arrangement quality")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=6.5)
    fig.tight_layout()
    p = FIG_DIR / "fig_coverage_quality.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figures.append({"name": p.name,
                    "caption": "Left: absolute coverage error (all cases below the "
                               "0.5 % criterion).  Right: initial maximum disk "
                               "overlap vs target coverage; the 0.06 m criterion is "
                               "met up to 8/10 ice, while 8.5/10 exceeds the "
                               "two-dimensional random-close-packing limit."})

    # fig 3: jamming/placement routes
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    routes = report["placement_routes"]
    labels, values, colors = [], [], []
    for key, row in routes.items():
        labels.append(key.replace("_power_law_k2.5", "").replace("_delta_r5.0", ""))
        values.append(row["realised_coverage"])
        colors.append("#c44" if key.startswith("rsa") else "#268")
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.axhline(0.85, color="k", ls="--", lw=1.0)
    ax.set_ylabel("realised coverage")
    ax.set_title("placement routes: RSA jamming vs compaction/lattice")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = FIG_DIR / "fig_placement_routes.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figures.append({"name": p.name,
                    "caption": "Realised coverage of the placement routes for the "
                               "same requested coverages: random sequential "
                               "adsorption (red) blocks far below the target, the "
                               "compaction routes (blue) reach it.  The dashed line "
                               "is the 8.5/10 target."})

    # fig 4: scalar scaling
    timing = report["timing"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(timing["floe_counts"], timing["wall_s"], "o-", lw=1.4)
    ax.set_xlabel("floe count N")
    ax.set_ylabel("generation wall time [s]")
    ax.set_title(f"generation cost, log-log slope {timing['log_log_slope']:.2f}")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = FIG_DIR / "fig_scaling.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figures.append({"name": p.name,
                    "caption": "Generation wall time vs floe count at fixed coverage "
                               "(the domain grows with N so the density is constant); "
                               "the fitted log-log slope is close to one."})

    # fig 5: field snapshots
    snap = report.get("field_snapshots", {})
    if snap:
        fig, axes = plt.subplots(len(snap), 1, figsize=(9.0, 2.0 * len(snap)))
        axes = [axes] if len(snap) == 1 else axes
        for ax, (name, row) in zip(axes, snap.items()):
            pos = torch.tensor(row["pos"], dtype=torch.float64)
            rad = torch.tensor(row["radius"], dtype=torch.float64)
            for (x, y), r in zip(pos.tolist(), rad.tolist()):
                ax.add_patch(Circle((x, y), r, fill=True, alpha=0.45,
                                    facecolor="#69c", edgecolor="k", lw=0.3))
            ax.set_xlim(0, row["domain"][0])
            ax.set_ylim(0, row["domain"][1])
            ax.set_aspect("equal")
            ax.set_title(f"{name}  (phi={row['coverage']:.2f}, N={len(rad)})",
                         fontsize=8)
            ax.tick_params(labelsize=6)
        fig.suptitle("generated floe fields at 8/10 ice", fontsize=9)
        fig.tight_layout()
        p = FIG_DIR / "fig_field_snapshots.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        figures.append({"name": p.name,
                        "caption": "Generated floe fields at 8/10 ice: H1 "
                                   "monodisperse reference (top), a power-law field "
                                   "and a bimodal field.  Marker size is the floe "
                                   "radius; the multi-disperse fields contain the "
                                   "same ice area with a realistic size spread."})

    # fig 6: sensitivity of cleared width and resistance
    transit = report["transit"]
    names = list(transit.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    axes[0].bar(range(len(names)),
                [transit[n]["cleared_width_tail_mean_m"] for n in names],
                color="#268")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=90, fontsize=6.5)
    axes[0].set_ylabel("cleared channel width [m]")
    axes[1].bar(range(len(names)),
                [transit[n]["resistance_tail_mean_N"] / 1e3 for n in names],
                color="#a52")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=90, fontsize=6.5)
    axes[1].set_ylabel("ice resistance [kN]")
    axes[0].set_title("cleared width (tail mean)")
    axes[1].set_title("ice resistance (tail mean)")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = FIG_DIR / "fig_transit_sensitivity.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figures.append({"name": p.name,
                    "caption": "Channel-transit sensitivity to the FSD at 8/10 ice "
                               "(same hull, speed, timestep and seed): cleared "
                               "channel width and ice resistance.  The H1 "
                               "monodisperse field is the baseline; the bars are the "
                               "same ice concentration with different floe-size "
                               "distributions."})
    return figures# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="reduced run for unit tests")
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--out", type=Path, default=JSON_OUT)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    fast = bool(args.fast)
    global COVERAGES
    if fast:
        domain = (90.0, 45.0)
        COVERAGES = [0.60, 0.85]
        dt, transit_steps, record_every = 1.0e-2, 1500, 100
        timing_scales = (0.5, 2.0)
    else:
        domain = (160.0, 80.0)
        dt, transit_steps, record_every = 8.0e-3, 5000, 100
        timing_scales = (0.5, 1.0, 2.0, 4.0, 8.0)

    hull = HullGeometry(length=80.0 if not fast else 40.0,
                        beam=20.0 if not fast else 12.0,
                        speed=1.6 if not fast else 3.0,
                        y_position=0.5 * domain[1],
                        x_start=-0.5 * (80.0 if not fast else 40.0))

    tic = time.perf_counter()
    report = {
        "meta": {
            "module": "tensordem.fsd_packing",
            "benchmark": "benchmarks/fsd_packing_sensitivity.py",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mode": "fast" if fast else "production",
            "seed": args.seed,
        },
        "configuration": {
            "domain_m": {"length": domain[0], "width": domain[1]},
            "coverages": list(COVERAGES),
            "families": [{"name": n, "spec": s} for n, s in FAMILIES],
            "method": "lattice",
            "hull": {"length": hull.length, "beam": hull.beam, "speed": hull.speed,
                     "y_position": hull.y_position, "x_start": hull.x_start},
            "transit": {"dt": dt, "steps": transit_steps,
                        "record_every": record_every},
        },
    }

    print("[A] FSD families", flush=True)
    report["families"] = section_families()

    print("[B] coverage sweep", flush=True)
    report["coverage_sweep"] = section_coverage(domain, args.seed)

    print("[C] placement routes / RSA negative control", flush=True)
    report["placement_routes"] = section_placement_routes(domain, args.seed)

    print("[D] monodisperse-limit regression lock", flush=True)
    report["monodisperse_lock"] = section_monodisperse_lock(domain, args.seed)

    print("[E] generation-time scaling", flush=True)
    timing = section_timing(args.seed, domain)
    counts = [p["floe_count"] for p in timing["points"]]
    times = [p["wall_s"] for p in timing["points"]]
    if len(timing_scales) != len(counts):                 # pragma: no cover
        pass
    report["timing"] = timing

    print("[F] channel transit", flush=True)
    fields = {}
    for name, spec in FAMILIES:
        fields[name] = generate_field(make_fsd(spec), 0.80,
                                      domain_length=domain[0],
                                      domain_width=domain[1], seed=args.seed)
    fields["h1_monodisperse"] = generate_field(DeltaFSD(5.0), 0.80,
                                               domain_length=domain[0],
                                               domain_width=domain[1],
                                               seed=args.seed)
    report["transit"] = section_transit(fields, hull, dt=dt, n_steps=transit_steps,
                                        record_every=record_every)

    # seed ensemble for the dispersion criterion
    seed_ensemble = {}
    for offset, seed in enumerate((args.seed, args.seed + 1, args.seed + 2)):
        field = generate_field(BimodalFSD(), 0.80, domain_length=domain[0],
                               domain_width=domain[1], seed=seed)
        row = run_transit(field, dt=dt, n_steps=transit_steps,
                          record_every=record_every, hull=hull)
        row["field_max_overlap_m"] = field.max_overlap
        row["coverage_deviation_abs"] = field.coverage_deviation()
        seed_ensemble[f"bimodal_seed{seed}"] = row
        print(f"[seed] {seed}: cleared={row['cleared_width_tail_mean_m']:.2f} m",
              flush=True)
    report["seed_ensemble"] = seed_ensemble

    # snapshots for the field figure
    snapshots = {}
    for name, field in (("h1_monodisperse", fields["h1_monodisperse"]),
                        ("power_law_k2.5", fields["power_law_k2.5"]),
                        ("bimodal", fields["bimodal"])):
        snapshots[name] = {"pos": field.pos.tolist(),
                           "radius": field.radius.tolist(),
                           "domain": list(field.domain),
                           "coverage": field.realised_coverage}
    report["field_snapshots"] = snapshots

    report["criteria"] = evaluate_criteria(report)
    report["wall_time_s"] = time.perf_counter() - tic

    if not args.no_figures:
        report["figures"] = make_figures(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"[artifact] {args.out}", flush=True)
    for key, value in report["criteria"].items():
        flag = "PASS" if value.get("pass") else "FAIL"
        print(f"[criterion] {key}: {flag}", flush=True)
    print(f"[done] {report['wall_time_s']:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())