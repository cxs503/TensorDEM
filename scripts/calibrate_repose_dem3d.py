"""Generate the TensorDEM friction-angle -> repose-angle calibration artifact.

Runs the column-collapse forward model on a fixed friction grid (multi-realisation
averaging), assembles the monotone PCHIP calibration table, closes the loop by
recovering the friction coefficient from off-node simulated angles, and writes

* ``artifacts/repose_calibration/repose_calibration.json``  (machine-readable)
* ``artifacts/repose_calibration/repose_calibration.md``    (number table)
* ``docs/repose_angle_calibration/fig_*.png``               (figures)

Usage::

    python scripts/calibrate_repose_dem3d.py --jobs 8
    python scripts/calibrate_repose_dem3d.py --smoke

The full grid is ~30 runs of ~30 s each on one core; ``--jobs`` parallelises the
independent cases (each worker is pinned to one BLAS thread).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import tensordem.repose_calibration as rc  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "repose_calibration"
FIG_DIR = ROOT / "docs" / "repose_angle_calibration"

# ---------------------------------------------------------------------------
# Case definition (documented in docs/repose_angle_calibration.md)
# ---------------------------------------------------------------------------
CASE = dict(
    radius=0.008,            # m
    density=2500.0,          # kg/m^3
    kn=6.0e4,                # N/m
    kt_ratio=0.5,
    restitution=0.05,
    gravity=9.81,
    cylinder_radius=0.035,   # m
    fill_height=0.14,        # m
    packing_gap=0.01,
    jitter=0.002,
    hold_time=0.03,
    lift_time=0.35,
    settle_time=0.70,
)

SMOKE_CASE = dict(
    radius=0.009, kn=8.0e4, cylinder_radius=0.030, fill_height=0.11,
    packing_gap=0.01, jitter=0.002, hold_time=0.03, lift_time=0.22,
    settle_time=0.22, restitution=0.05,
)

MU_ROLLING = 0.2
MU_NODES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
CHECK_MU = (0.275, 0.325, 0.375, 0.425, 0.475, 0.525)
SEEDS = (4970, 4971, 4972)

SMOKE_NODES = (0.35, 0.70)
SMOKE_CHECK = (0.50, 0.25)


def _run(job: dict) -> dict:
    """One forward run: (mu, mu_rolling, seed) -> measured repose angle."""
    torch.set_num_threads(1)
    case = rc.ReposeCase(mu=job["mu"], mu_rolling=job["mu_rolling"],
                         seed=job["seed"], **job["case"])
    try:
        out = rc.simulate(case, return_particles=True)
        out["seed"] = job["seed"]
        out["role"] = job["role"]
        bound = rc.kinematic_bound_margin(out["theta_r_deg"], job["mu"])
        out["kinematic_bound"] = bound
        out["ok"] = True
        return {k: (round(v, 9) if isinstance(v, float) else v) for k, v in out.items()}
    except rc.ReposeCalibrationError as exc:      # e.g. no measurable flank
        return {"mu": job["mu"], "mu_rolling": job["mu_rolling"], "seed": job["seed"],
                "role": job["role"], "ok": False, "error": str(exc)}


def _mean(values):
    return sum(values) / len(values)


def _aggregate(runs, role):
    grouped: dict[float, list[dict]] = {}
    for run in runs:
        if run["role"] != role:
            continue
        grouped.setdefault(run["mu"], []).append(run)
    rows = []
    for mu in sorted(grouped):
        entries = [r for r in grouped[mu] if r.get("ok")]
        if not entries:
            rows.append({"mu": mu, "ok": False, "runs": grouped[mu]})
            continue
        thetas = [r["theta_r_deg"] for r in entries]
        rows.append({
            "mu": mu,
            "ok": True,
            "theta_r_deg": _mean(thetas),
            "theta_std_deg": (statistics.pstdev(thetas) if len(thetas) > 1 else 0.0),
            "theta_min_deg": min(thetas),
            "theta_max_deg": max(thetas),
            "n_realisations": len(entries),
            "r2": _mean([r["r2"] for r in entries]),
            "apex_z": _mean([r["apex_z"] for r in entries]),
            "toe_radius": _mean([r["toe_radius"] for r in entries]),
            "max_speed": max(r["max_speed"] for r in entries),
            "seconds": _mean([r.get("elapsed_s", 0.0) for r in entries]),
            "n_particles": entries[0]["n_particles"],
            "steps": entries[0]["steps"],
            "runs": entries,
        })
    return rows


def _isotonic(values):
    """Pool-adjacent-violators isotonic regression (nondecreasing fit)."""
    blocks = [[v, 1] for v in values]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            total = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            weight = blocks[i][1] + blocks[i + 1][1]
            blocks[i:i + 2] = [[total / weight, weight]]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted = []
    for value, weight in blocks:
        fitted.extend([value] * weight)
    return fitted


def _anchors(case: dict) -> dict:
    """Closed-form anchors A1/A2 measured with the same contact driver."""
    a1 = []
    for mu in (0.3, 0.6):
        a1.append(rc.sliding_onset_angle(mu, radius=case["radius"] * 0.75,
                                         kn=case["kn"], restitution=case["restitution"],
                                         duration=0.15, slide_fraction=0.25))
    a2 = []
    for mu, mu_r in ((0.6, 0.2), (0.2, 0.2), (0.6, 0.6)):
        critical = math.degrees(math.atan(min(mu, mu_r)))
        below = rc.sliding_displacement(
            mu, 0.8 * critical, radius=case["radius"] * 0.75, kn=case["kn"],
            restitution=case["restitution"], duration=0.08,
            allow_rotation=True, mu_rolling=mu_r)
        above = rc.sliding_displacement(
            mu, 1.4 * critical, radius=case["radius"] * 0.75, kn=case["kn"],
            restitution=case["restitution"], duration=0.08,
            allow_rotation=True, mu_rolling=mu_r)
        a2.append({"mu": mu, "mu_rolling": mu_r, "critical_deg": critical,
                   "displacement_below_m": below["displacement"],
                   "displacement_above_m": above["displacement"],
                   "static_below": below["displacement"] < 1.0e-5})
    return {"A1_irrotational_onset": a1, "A2_rolling_onset": a2}


def _figures(table, node_rows, check_rows, inverse_rows, anchors, case, mu_rolling):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    # --- Figure 1: calibration curve + inverse recovery ---------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    grid = [table.mu[0] + i * (table.mu[-1] - table.mu[0]) / 200.0 for i in range(201)]
    ax[0].plot(grid, [table.theta_from_mu(m) for m in grid], "-",
               color="C0", label="monotone PCHIP interpolant")
    ax[0].errorbar([r["mu"] for r in node_rows], [r["theta_r_deg"] for r in node_rows],
                   yerr=[r["theta_std_deg"] for r in node_rows], fmt="o", color="C0",
                   capsize=3, label=f"DEM nodes ({node_rows[0]['n_realisations']} seeds)")
    ax[0].plot([r["mu"] for r in check_rows], [r["theta_r_deg"] for r in check_rows],
               "s", color="C1", label="off-node verification runs")
    ax[0].set_xlabel(r"Coulomb friction coefficient $\mu$ [-]")
    ax[0].set_ylabel(r"repose angle $\theta_r$ [deg]")
    ax[0].set_title(fr"DEM column collapse, $\mu_r$={mu_rolling}")
    ax[0].grid(alpha=0.3)
    ax[0].legend(loc="lower right", fontsize=8)

    ax[1].axhline(2.0, color="C3", ls="--", lw=1, label="2 % acceptance limit")
    scored = [r for r in inverse_rows if r["rel_error"] is not None]
    if scored:
        ax[1].plot([r["mu_true"] for r in scored],
                   [100.0 * abs(r["rel_error"]) for r in scored], "s-", color="C1",
                   label="closed loop (DEM -> table -> mu)")
    interp = [r for r in inverse_rows if r.get("table_theta_deg") is not None]
    if interp:
        ax[1].plot([r["mu_true"] for r in interp],
                   [100.0 * abs(table.mu_from_theta(r["table_theta_deg"]) - r["mu_true"])
                    / r["mu_true"] for r in interp], "^--", color="C0",
                   label="interpolation only")
    ax[1].set_xlabel(r"true $\mu$ of the verification run [-]")
    ax[1].set_ylabel(r"inverse-lookup error on $\mu$ [%]")
    ax[1].set_title("closed-loop inverse recovery")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "fig_repose_calibration_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # --- Figure 2: contact-law anchors --------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    mus = [a["mu"] for a in anchors["A1_irrotational_onset"]]
    meas = [a["measured_deg"] for a in anchors["A1_irrotational_onset"]]
    pred = [a["predicted_deg"] for a in anchors["A1_irrotational_onset"]]
    ax[0].plot(pred, meas, "o", color="C0", label="measured onset")
    lim = [0, max(pred) * 1.15]
    ax[0].plot(lim, lim, "--", color="0.5", label=r"$\theta=\arctan\mu$")
    ax[0].set_xlabel(r"$\arctan\mu$ [deg]")
    ax[0].set_ylabel("measured sliding onset [deg]")
    ax[0].set_title("anchor A1: irrotational slider")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    labels = [rf"$\mu$={a['mu']:.1f}, $\mu_r$={a['mu_rolling']:.1f}" for a in anchors["A2_rolling_onset"]]
    below = [a["displacement_below_m"] for a in anchors["A2_rolling_onset"]]
    above = [a["displacement_above_m"] for a in anchors["A2_rolling_onset"]]
    idx = range(len(labels))
    ax[1].bar([i - 0.2 for i in idx], below, width=0.4, label="0.8 x critical tilt")
    ax[1].bar([i + 0.2 for i in idx], above, width=0.4, label="1.4 x critical tilt")
    ax[1].set_xticks(list(idx))
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("down-slope displacement after 80 ms [m]")
    ax[1].set_title(r"anchor A2: onset at $\arctan(\min(\mu,\mu_r))$")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = FIG_DIR / "fig_repose_contact_anchors.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # --- Figure 3: settled heap silhouettes ---------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for i, row in enumerate(node_rows):
        run = row["runs"][0]
        pos = run.get("positions")
        if pos is None:
            continue
        pos = pos.numpy()
        centre = pos[:, :2].mean(axis=0)
        radial = ((pos[:, :2] - centre) ** 2).sum(axis=1) ** 0.5
        ax.scatter(radial * 1e3, pos[:, 2] * 1e3, s=3, color=f"C{i}", alpha=0.5)
        theta = row["theta_r_deg"]
        r_shoulder = 0.0
        r_toe = row["toe_radius"]
        h = r_toe * math.tan(math.radians(theta))
        ax.plot([r_shoulder * 1e3, r_toe * 1e3], [h * 1e3, 0.0], "--", color=f"C{i}",
                lw=1.2, label=rf"$\mu$={row['mu']:.2f}: $\theta_r$={theta:.1f}$^\circ$")
    ax.set_xlabel("radial distance from deposit centroid [mm]")
    ax.set_ylabel("elevation [mm]")
    ax.set_title("settled heap cross-sections and fitted flank slopes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "fig_repose_heap_sections.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--smoke", action="store_true",
                        help="two-node reduced grid, writes to a scratch subdirectory")
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    args = parser.parse_args(argv)

    case = dict(SMOKE_CASE if args.smoke else CASE)
    nodes = list(SMOKE_NODES if args.smoke else MU_NODES)
    checks = list(SMOKE_CHECK if args.smoke else CHECK_MU)
    seeds = list(SEEDS[: max(1, args.seeds)])

    jobs = [{"mu": mu, "mu_rolling": MU_ROLLING, "seed": seed, "case": case,
             "role": "table"} for mu in nodes for seed in seeds]
    jobs += [{"mu": mu, "mu_rolling": MU_ROLLING, "seed": seed, "case": case,
              "role": "check"} for mu in checks for seed in seeds]

    print(f"running {len(jobs)} forward cases with {args.jobs} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        runs = list(pool.map(_run, jobs))

    node_rows = _aggregate(runs, "table")
    check_rows = _aggregate(runs, "check")

    good_nodes = [row for row in node_rows if row["ok"]]
    if len(good_nodes) < 2:
        print("fewer than two usable nodes -- cannot build a table", file=sys.stderr)
        return 2
    raw_theta = [row["theta_r_deg"] for row in good_nodes]
    raw_monotone = all(b > a for a, b in zip(raw_theta, raw_theta[1:]))
    # A calibration table may only be built where the simulated curve is
    # monotone: keep the longest strictly-increasing prefix of the raw nodes and
    # report the excluded nodes as the non-monotone region (no silent smoothing).
    n_keep = len(raw_theta)
    for i in range(1, len(raw_theta)):
        if raw_theta[i] <= raw_theta[i - 1]:
            n_keep = i
            break
    excluded_nodes = [{"mu": row["mu"], "theta_r_deg": raw,
                       "theta_std_deg": row["theta_std_deg"], "r2": row["r2"],
                       "reason": "breaks monotonicity of the simulated curve"}
                      for row, raw in zip(good_nodes[n_keep:], raw_theta[n_keep:])]
    good_nodes = good_nodes[:n_keep]
    raw_theta = raw_theta[:n_keep]
    fitted = list(raw_theta)
    # Strictly increasing requirement of the table: a plateau produced by the
    # isotonic fit is separated by a numerical epsilon (documented).
    for i in range(1, len(fitted)):
        if fitted[i] <= fitted[i - 1]:
            fitted[i] = fitted[i - 1] + 1e-9
    table = rc.ReposeCalibrationTable(
        mu=tuple(row["mu"] for row in good_nodes), theta_deg=tuple(fitted),
        mu_rolling=MU_ROLLING,
        provenance={"case": case, "fit": "isotonic (PAVA)" if not raw_monotone else "raw",
                    "raw_theta_deg": raw_theta})
    monotone = all(b > a for a, b in zip(table.theta_deg, table.theta_deg[1:]))
    theta_of = {row["mu"]: table.theta_from_mu(row["mu"]) for row in good_nodes}
    table_vs_raw = [{"mu": row["mu"], "raw_deg": raw, "table_deg": fit,
                     "delta_deg": fit - raw}
                    for row, raw, fit in zip(good_nodes, raw_theta, table.theta_deg)]

    inverse_rows = []
    for row in check_rows:
        if not row["ok"]:
            continue
        entry = {
            "mu_true": row["mu"],
            "theta_r_deg": row["theta_r_deg"],
            "theta_std_deg": row["theta_std_deg"],
            "n_realisations": row["n_realisations"],
        }
        try:
            entry["table_theta_deg"] = table.theta_from_mu(row["mu"])
        except rc.ReposeExtrapolationError:
            # verification run outside the calibrated friction bracket: the
            # forward query is refused, which is itself the expected behaviour.
            entry["table_theta_deg"] = None
        try:
            recovered = table.mu_from_theta(row["theta_r_deg"])
        except rc.ReposeExtrapolationError as exc:
            entry.update({"mu_recovered": None, "abs_error": None, "rel_error": None,
                          "status": "rejected_outside_bracket", "detail": str(exc)})
        else:
            entry.update({"mu_recovered": recovered, "abs_error": recovered - row["mu"],
                          "rel_error": (recovered - row["mu"]) / row["mu"],
                          "status": "recovered"})
        # local invertibility: dmu/dtheta of the table at this angle
        eps = 1e-3
        lo = max(table.theta_deg[0], row["theta_r_deg"] - eps)
        hi = min(table.theta_deg[-1], row["theta_r_deg"] + eps)
        if hi > lo:
            slope = (table.mu_from_theta(hi) - table.mu_from_theta(lo)) / (hi - lo)
            entry["dmu_dtheta"] = slope
            entry["sigma_mu_from_seed_noise"] = abs(slope) * row["theta_std_deg"]
        inverse_rows.append(entry)

    bound_rows = [{"mu": mu, "theta_r_deg": theta,
                   **rc.kinematic_bound_margin(theta, mu)} for mu, theta in theta_of.items()]

    recovered_rows = [r for r in inverse_rows if r["rel_error"] is not None]
    worst_rel = max((abs(r["rel_error"]) for r in recovered_rows), default=float("inf"))
    # also score the interpolation-only error (truth = table's own forward value)
    worst_interp = max((abs(table.mu_from_theta(r["table_theta_deg"]) - r["mu_true"])
                        / r["mu_true"] for r in recovered_rows
                        if r["table_theta_deg"] is not None),
                       default=None)
    payload = {
        "kind": "tensordem-repose-calibration-v1",
        "module": "tensordem.repose_calibration",
        "forward_model": {
            "protocol": "lifting-cylinder column collapse, monodisperse spheres",
            "contact_laws": ["linear spring-dashpot normal (e from restitution)",
                             "Cundall-Strack tangential, saturated at mu|Fn|",
                             "rolling spring, saturated at mu_r R |Fn|",
                             "rough cylindrical wall (mu_wall = mu)"],
            "case": case,
            "mu_rolling": MU_ROLLING,
            "seeds": seeds,
            "n_particles": good_nodes[0]["n_particles"],
            "steps": good_nodes[0]["steps"],
        },
        "table": table.to_json(),
        "node_rows": [{k: v for k, v in row.items() if k != "runs"} for row in node_rows],
        "kinematic_bound": bound_rows,
        "inverse_recovery": inverse_rows,
        "anchors": {},
        "table_vs_raw": table_vs_raw,
        "excluded_nodes": excluded_nodes,
        "verdict": {
            "raw_nodes_monotone": raw_monotone,
            "monotone": monotone,
            "worst_inverse_rel_error_interpolation_only": worst_interp,
            "n_verification_recovered": len(recovered_rows),
            "worst_inverse_rel_error": worst_rel,
            "inverse_recovery_under_2pct": worst_rel < 0.02,
            "kinematic_bound_all_satisfied": all(b["satisfied"] for b in bound_rows),
            "n_table_nodes": len(good_nodes),
            "n_excluded_nodes": len(excluded_nodes),
            "n_verification_runs": len(inverse_rows),
        },
    }

    out_dir = ARTIFACT_DIR / ("smoke" if args.smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "repose_calibration.json"
    json_path.write_text(json.dumps(payload, indent=2))

    try:                                     # anchors are cheap but not critical
        anchors = _anchors(case)
    except Exception as exc:                 # pragma: no cover - diagnostic only
        anchors = {"error": str(exc)}
        print(f"anchor evaluation failed: {exc}", file=sys.stderr)
    payload["anchors"] = anchors
    json_path.write_text(json.dumps(payload, indent=2))

    figures = []
    try:
        figures = _figures(table, good_nodes, check_rows, inverse_rows, anchors, case,
                           MU_ROLLING)
    except Exception as exc:                       # pragma: no cover - plotting only
        print(f"figure generation skipped: {exc}", file=sys.stderr)

    md = ["# Repose-angle calibration (generated)", "",
          f"case: `{json.dumps(case)}`, mu_rolling={MU_ROLLING}, seeds={seeds}", "",
          "| mu | theta_r [deg] | std | R_toe [mm] | apex [mm] | r2 | max_speed |",
          "|---|---|---|---|---|---|---|"]
    for row in good_nodes:
        md.append(f"| {row['mu']:.2f} | {row['theta_r_deg']:.2f} | {row['theta_std_deg']:.2f} | "
                  f"{row['toe_radius']*1e3:.1f} | {row['apex_z']*1e3:.1f} | {row['r2']:.3f} | "
                  f"{row['max_speed']:.3f} |")
    md += ["", "| mu_true | theta_r [deg] | mu_recovered | rel. error |",
           "|---|---|---|---|"]
    for row in inverse_rows:
        if row["mu_recovered"] is None:
            md.append(f"| {row['mu_true']:.3f} | {row['theta_r_deg']:.2f} | "
                      f"rejected | outside bracket |")
        else:
            md.append(f"| {row['mu_true']:.3f} | {row['theta_r_deg']:.2f} | "
                      f"{row['mu_recovered']:.4f} | {100*row['rel_error']:+.2f}% |")
    (out_dir / "repose_calibration.md").write_text("\n".join(md) + "\n")

    print(json.dumps(payload["verdict"], indent=2))
    print(f"wrote {json_path}")
    for figure in figures:
        print(f"wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())