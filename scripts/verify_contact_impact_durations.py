"""Contact-impact dashpot-convention campaign: restitution and contact-duration baselines.

Reproduces the numbers in ``docs/dem3d_contact_impact_dashpot_convention.md``:

* measured ``e`` vs the clamped spring-dashpot closed form (the reference the
  engine's truncated normal force obeys),
* force-clamped duration ``theta*/omega_d`` and geometric overlap duration
  (contact time incl. the free rebound inside the remaining overlap),
* a ``dt`` refinement sweep (``dt_scale`` in 1, 1/2, 1/4) that shows the residual
  is discretisation-limited.

Artefacts: ``artifacts/contact_impact_dashpot/{impact_duration_convergence.json,csv,png}``.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensordem.contact_impact import pair_impact  # noqa: E402

OUT = ROOT / "artifacts" / "contact_impact_dashpot"
E_TARGETS = (0.7, 0.5, 0.3)
DT_SCALES = (1.0, 0.5, 0.25)


def run_campaign() -> list[dict]:
    rows: list[dict] = []
    for e_target in E_TARGETS:
        for scale in DT_SCALES:
            r = pair_impact(e_target=e_target, dt_scale=scale)
            rows.append(
                {
                    "e_target": e_target,
                    "dt_scale": scale,
                    "dt": r.dt,
                    "e_measured": r.e_measured,
                    "e_clamped": r.e_clamped,
                    "e_error_vs_clamped": r.e_error_vs_clamped,
                    "e_error_vs_unclamped": r.e_error_vs_unclamped,
                    "contact_time_force": r.contact_time_force,
                    "contact_time_force_analytic": r.contact_time_force_analytic,
                    "contact_time_force_error": r.contact_time_force_error,
                    "contact_time": r.contact_time,
                    "contact_time_analytic": r.contact_time_analytic,
                    "contact_time_error": r.contact_time_error,
                    "force_steps": r.force_steps,
                    "contact_steps": r.contact_steps,
                    "steps_per_contact": r.steps_per_contact,
                    "momentum_error": r.momentum_error,
                    "force_law_residual": r.force_law_residual,
                }
            )
    return rows


def figure(rows: list[dict], png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    for e_target in E_TARGETS:
        sel = [r for r in rows if r["e_target"] == e_target]
        ax.plot([r["dt"] for r in sel], [100 * r["e_error_vs_clamped"] for r in sel],
                "o-", label=f"measured e vs clamped form, $e_t$={e_target}")
    ax.axhline(3.0, color="r", ls="--", lw=1, label="3 % band")
    ax.axhline(-3.0, color="r", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("solver dt [s]")
    ax.set_ylabel("restitution error [%]")
    ax.set_title("(a) restitution vs clamped closed form")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for e_target in E_TARGETS:
        sel = [r for r in rows if r["e_target"] == e_target]
        ax.plot([r["dt"] for r in sel], [100 * r["contact_time_force_error"] for r in sel],
                "s-", label=f"force duration, $e_t$={e_target}")
        ax.plot([r["dt"] for r in sel], [100 * r["contact_time_error"] for r in sel],
                "^--", label=f"overlap duration, $e_t$={e_target}")
    ax.axhline(3.0, color="r", ls="--", lw=1)
    ax.axhline(-3.0, color="r", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("solver dt [s]")
    ax.set_ylabel("contact-duration error [%]")
    ax.set_title("(b) duration baselines under dt refinement")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for e_target in E_TARGETS:
        sel = [r for r in rows if r["e_target"] == e_target]
        ax.plot([r["dt"] for r in sel],
                [r["contact_time_force"] for r in sel], "o-", label="simulated force duration")
        ax.axhline(sel[0]["contact_time_force_analytic"], color="k", ls=":", lw=1,
                   label=f"analytic $\\theta^*/\\omega_d$, $e_t$={e_target}")
        ax.plot([r["dt"] for r in sel],
                [r["contact_time"] for r in sel], "s--", label="simulated overlap duration")
    ax.set_xscale("log")
    ax.set_xlabel("solver dt [s]")
    ax.set_ylabel("contact duration [s]")
    ax.set_title("(c) absolute durations")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    fig.suptitle("DEM3D equal-mass impact: absolute dashpot convention ($c_n$ from restitution mapping)")
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = run_campaign()
    (OUT / "impact_duration_convergence.json").write_text(json.dumps(rows, indent=1))
    with (OUT / "impact_duration_convergence.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure(rows, OUT / "impact_duration_convergence.png")

    worst_e = max(abs(r["e_error_vs_clamped"]) for r in rows)
    worst_f = max(abs(r["contact_time_force_error"]) for r in rows)
    worst_c = max(abs(r["contact_time_error"]) for r in rows)
    print(f"{'e_t':>5} {'dt_scale':>8} {'e_meas':>8} {'e_err_vs_clamped':>16} "
          f"{'Tforce_err':>11} {'Toverlap_err':>13} {'steps':>6}")
    for r in rows:
        print(f"{r['e_target']:>5} {r['dt_scale']:>8} {r['e_measured']:>8.5f} "
              f"{100 * r['e_error_vs_clamped']:>15.3f}% {100 * r['contact_time_force_error']:>10.3f}% "
              f"{100 * r['contact_time_error']:>12.3f}% {r['steps_per_contact']:>6}")
    print(f"\nworst |e error| = {100 * worst_e:.3f} %  "
          f"worst |force-duration error| = {100 * worst_f:.3f} %  "
          f"worst |overlap-duration error| = {100 * worst_c:.3f} %")
    print(f"artefacts -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())