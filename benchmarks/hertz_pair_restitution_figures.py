"""Figures for the Hertz pair-restitution audit (matplotlib, no hand-typed numbers)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def write_figures(audit: dict, outdir: Path) -> list[str]:
    rows = audit["pair_rows"]
    speeds = audit["speed_rows"]
    cal = audit["calibration"]
    names = []

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))

    ax = axes[0][0]
    e_t = [r["e_target"] for r in rows]
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", lw=1.0, label="declared = realised")
    ax.plot(e_t, [r["e_reference_ode"] for r in rows], "o-", label="independent ODE reference")
    ax.plot(e_t, [r["e_engine_dt_scale_0.0625"] for r in rows], "s-", label="IceDEM3D (dt/16)")
    ax.plot(e_t, [r["e_reference_first_order"] for r in rows], "^:", label="1st-order quadrature")
    ax.set_xlabel("declared restitution_coefficient")
    ax.set_ylabel("realised restitution")
    ax.set_title("Hertz pair impact: declared vs realised restitution")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(e_t, [r["bias_vs_target_pct"] for r in rows], "o-", label="engine bias vs declared")
    ax.plot(e_t, [r["reference_vs_target_pct"] for r in rows], "x--",
            label="ODE bias vs declared")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.axhline(3.0, color="r", lw=0.8, ls=":")
    ax.axhline(-3.0, color="r", lw=0.8, ls=":")
    ax.set_xlabel("declared restitution_coefficient")
    ax.set_ylabel("relative bias [%]")
    ax.set_title("Systematic bias of the linear Kelvin-Voigt map on a Hertz law")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot([r["impact_speed"] for r in speeds], [r["e_reference_ode"] for r in speeds],
            "o-", label="ODE reference")
    ax.plot([r["impact_speed"] for r in speeds], [r["e_engine"] for r in speeds],
            "s-", label="IceDEM3D (dt/4)")
    ax.set_xscale("log")
    ax.set_xlabel("per-particle impact speed [m/s]")
    ax.set_ylabel("realised restitution")
    ax.set_title("Speed independence at declared e = 0.5")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1][1]
    ax.plot([c["e_desired"] for c in cal],
            [c["declared_restitution_coefficient"] for c in cal], "o-",
            label="value to declare")
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", lw=1.0)
    ax.set_xlabel("desired physical restitution e")
    ax.set_ylabel("restitution_coefficient to declare")
    ax.set_title("Calibration table (Hertz pair law, dt-converged)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("TensorDEM - Hertz pair-contact restitution verification "
                 f"(E*={audit['parameters']['e_star_pair_Pa']:.3e} Pa, "
                 f"R*={audit['parameters']['r_star_pair_m']:.4f} m)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    name = "hertz_pair_restitution.png"
    fig.savefig(outdir / name, dpi=160)
    plt.close(fig)
    names.append(name)
    return names