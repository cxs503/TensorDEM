"""Repeated-impact verification of the TensorDEM3D contact law (bounce chain).

Every other restitution audit in this repository measures **one** impact.  This
script drives a single free particle so that it hits two rigid planes several
times in a row and reduces the chain to closure metrics:

* per-episode restitution vs the reference solution of the *same* clipped law
  (``tensordem.restitution.unilateral_contact_detail``);
* chain propagation ``v_n`` vs ``e**n v_0`` (accumulation / state release);
* force-window duration vs the reference contact time;
* per-episode dissipation vs ``(1 - e_ref**2) KE``;
* free-flight invariance (no gravity, no drag => ``v_z`` must be exactly
  constant between episodes) and the "no reaction outside the plane" invariant;
* time-step sensitivity, so the agreement is not an artefact of one ``dt``.

A bilateral (textbook) damping-to-restitution map is carried along as a
negative control: agreement with the clipped law is only informative if a
*plausible but wrong* law fails the same tolerance.

Usage
-----
    python scripts/verify_dem3d_bounce_chain.py [--output artifacts/dem3d_bounce_chain]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensordem.bounce_chain import BounceChainConfig, run_bounce_chain  # noqa: E402

# (damping [N s/m], bounces).  The chain decays geometrically, so the ballistic
# transit between episodes grows like 1/e**n: a high-damping chain in a fixed gap
# would need ~1e7 steps for the fifth bounce.  The bounce count is therefore cut
# where the tail would dominate the run time; this is stated in the doc.
CASES = ((2.0, 4), (5.0, 4), (10.0, 3), (20.0, 2))
TRAVEL = 0.015
IMPACT_SPEED = 1.0


def _run(damping: float, bounces: int = 5, **overrides) -> dict:
    cfg = BounceChainConfig(
        damping=damping, bounces=bounces, travel=TRAVEL,
        impact_speed=IMPACT_SPEED, **{"dt_fraction": 0.5, **overrides}
    )
    return run_bounce_chain(cfg).to_dict()


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0].keys())
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(
            f"{row[f]:.12g}" if isinstance(row[f], float) else str(row[f]) for f in fields
        ))
    path.write_text("\n".join(lines) + "\n")


def _figures(outdir: Path, cases: list[dict], dt_case: dict) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover - optional dep
        print(f"[warn] matplotlib unavailable ({exc}); figures skipped")
        return []

    made = []

    # Figure 1 -- chain propagation against the analytic geometric decay.
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for case in cases:
        events = case["events"]
        e0 = events[0]["restitution"]
        idx = [ev["index"] + 1 for ev in events]
        meas = [ev["rebound_speed"] for ev in events]
        pred = [case["config"]["impact_speed"] * e0 ** n for n in idx]
        line, = ax.plot(idx, meas, "o-", ms=5, label=f"DEM  c_n={case['config']['damping']:g} N s/m")
        ax.plot(idx, pred, "--", color=line.get_color(), lw=1.0, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("bounce index n")
    ax.set_ylabel("separation speed $v_n$ [m/s]")
    ax.set_title("Repeated impacts: measured $v_n$ (markers) vs $e^n v_0$ (dashed)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = outdir / "fig1_chain_propagation.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    made.append(path.name)

    # Figure 2 -- trajectory with contact windows and the residual force.
    case = cases[1] if len(cases) > 1 else cases[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 5.6), sharex=False,
                                   gridspec_kw={"height_ratios": [2, 1]})
    times, heights, speeds = [], [], []
    for ev in case["events"]:
        times.append(ev["t_end"])
        speeds.append(ev["rebound_speed"])
        heights.append(ev["max_overlap"])
    ax1.stem(times, heights, basefmt=" ")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("peak overlap [m]")
    ax1.set_title(f"Contact episodes, $c_n$={case['config']['damping']:g} N s/m "
                  f"(peak overlap shrinks as the impact speed decays)")
    ax1.grid(alpha=0.3)
    labels = [f"#{ev['index']}\n{ev['plane'][:3]}" for ev in case["events"]]
    ax2.bar(range(len(case["events"])), [ev["force_duration"] for ev in case["events"]],
            color="#3b6ea5")
    ax2.plot(range(len(case["events"])), [ev["reference"]["duration"] for ev in case["events"]],
             "r_", ms=18, mew=2, label="reference contact time")
    ax2.set_xticks(range(len(case["events"])))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("force window [s]")
    ax2.set_xlabel("contact episode")
    ax2.set_ylim(0, 1.35 * max(ev["force_duration"] for ev in case["events"]))
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    path = outdir / "fig2_contact_episodes.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    made.append(path.name)

    # Figure 3 -- closure errors per case plus the bilateral negative control.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = [f"c={c['config']['damping']:g}" for c in cases]
    clipped = [100 * c["checks"]["restitution_worst_rel_error"] for c in cases]
    chain = [100 * c["checks"]["chain_worst_rel_error"] for c in cases]
    bilateral = [100 * c["checks"]["bilateral_worst_rel_error"] for c in cases]
    x = range(len(cases))
    ax.bar([i - 0.25 for i in x], clipped, width=0.25, label="per-episode vs clipped law")
    ax.bar(list(x), chain, width=0.25, label="chain vs $e^n v_0$")
    ax.bar([i + 0.25 for i in x], bilateral, width=0.25, label="vs bilateral map (negative control)")
    ax.axhline(3.0, color="k", ls="--", lw=1.0, label="3 % acceptance")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("worst relative error [%]")
    ax.set_title("Bounce-chain closure errors")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    path = outdir / "fig3_closure_errors.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    made.append(path.name)
    return made


def summarise(cases: list[dict], dt_case: dict) -> dict:
    rows = []
    for case in cases:
        chk = case["checks"]
        rows.append({
            "damping_N_s_per_m": case["config"]["damping"],
            "bounces": chk["n_closed_events"],
            "restitution_first": chk["restitution_first_event"],
            "restitution_mean": chk["restitution_mean"],
            "clipped_worst_rel_error_pct": 100 * chk["restitution_worst_rel_error"],
            "chain_worst_rel_error_pct": 100 * chk["chain_worst_rel_error"],
            "force_duration_worst_rel_error_pct": 100 * chk["force_duration_worst_rel_error"],
            "energy_loss_worst_rel_error_pct": 100 * chk["energy_loss_worst_rel_error"],
            "bilateral_restitution": chk["bilateral_restitution"],
            "bilateral_worst_rel_error_pct": 100 * chk["bilateral_worst_rel_error"],
            "free_flight_velocity_leak": case["free_flight_velocity_leak"],
            "separated_force_leak_steps": chk["separated_force_leak_steps"],
            "geometric_duration_spread_pct": 100 * chk["geometric_overlap_duration_spread"],
        })
    # dt sensitivity: the *matched* case only (same c_n, halved dt).  Comparing
    # unrelated dampings against each other would just measure the damping.
    drift = None
    if dt_case:
        matched = [c for c in cases
                   if abs(c["config"]["damping"] - dt_case["config"]["damping"]) < 1e-12]
        if matched:
            base = matched[0]["checks"]["restitution_mean"]
            drift = abs(base - dt_case["checks"]["restitution_mean"]) / base
    return {"cases": rows, "dt_sensitivity": drift}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "dem3d_bounce_chain"))
    args = parser.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    cases = []
    for damping, bounces in CASES:
        print(f"[run] c_n = {damping:g} N s/m, {bounces} bounces", flush=True)
        case = _run(damping, bounces)
        case["name"] = f"c{int(damping)}"
        cases.append(case)
        print("      restitution_worst_rel_error = "
              f"{100 * case['checks']['restitution_worst_rel_error']:.3f} %", flush=True)

    print("[run] dt sensitivity: c_n = 5 with dt halved", flush=True)
    dt_case = _run(5.0, 4, dt_fraction=0.25)
    dt_case["name"] = "c5_dt_half"

    rows = summarise(cases, dt_case)
    figures = _figures(outdir, cases, dt_case)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "method": "TensorDEM3D single particle between two rigid planes, bonds disabled, "
                  "drag=0, no gravity; clipped unilateral contact law; reference = "
                  "tensordem.restitution.unilateral_contact_detail",
        "cases": cases,
        "dt_sensitivity": dt_case,
        "summary": rows,
        "figures": figures,
    }
    (outdir / "chain.json").write_text(json.dumps(payload, indent=2))

    csv_rows = []
    for case in cases:
        for ev in case["events"]:
            csv_rows.append({
                "damping_N_s_per_m": case["config"]["damping"],
                "bounce": ev["index"],
                "plane": ev["plane"],
                "impact_speed": ev["impact_speed"],
                "rebound_speed": ev["rebound_speed"],
                "restitution": ev["restitution"],
                "reference_restitution": ev["reference"]["restitution"],
                "force_duration": ev["force_duration"],
                "reference_duration": ev["reference"]["duration"],
                "geometric_overlap_duration": ev["t_end"] - ev["t_start"],
                "peak_overlap": ev["max_overlap"],
                "free_flight_to_next": (ev["free_flight_to_next"]
                                        if ev["free_flight_to_next"] is not None else float("nan")),
            })
    _write_csv(outdir / "chain_events.csv", csv_rows)

    print(json.dumps(rows, indent=2))
    print(f"artifacts -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())