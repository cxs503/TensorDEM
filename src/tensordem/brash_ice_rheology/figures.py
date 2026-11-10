"""Matplotlib figures for the brash-channel report.

Four figures (PNG, base64 for the report):
``fig_components``  stacked component breakdown vs beam, with the anchor total;
``fig_speed``       R(v) and the log-log power-law fit;
``fig_porosity``    component response to porosity;
``fig_mu_i``        measured DEM steady-shear points vs the Jop law +
                    the resistance model's own operating point.
"""
from __future__ import annotations

import base64
import io
import math
from typing import Dict, List, Optional, Sequence

__all__ = ["fig_components", "fig_speed", "fig_porosity", "fig_mu_i", "png_base64"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .anchors import reference_anchor  # noqa: E402
from .material import STANDARD_BRASH, BrashIceState  # noqa: E402
from .model import (Closure, STANDARD_BEAMS_M, brash_resistance,  # noqa: E402
                    porosity_sweep, speed_sweep, standard_hull)
from .mu_i import JopParameters, SWEEP_TARGETS  # noqa: E402


def png_base64(fig) -> str:
    """Encode a matplotlib figure as base64 PNG (report ``figures`` payload)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fig_components(state: BrashIceState = STANDARD_BRASH, speed: float = 3.0,
                   beams: Sequence[float] = STANDARD_BEAMS_M,
                   closure: Closure = None):
    """Stacked three-component breakdown with the reference anchor total."""
    closure = closure or Closure.from_ledger()
    shear, buoy, drag, anchor = [], [], [], []
    for beam in beams:
        hull = standard_hull(beam)
        res = brash_resistance(state, hull, speed, closure)
        shear.append(res.F_shear_N / 1e3)
        buoy.append(res.F_buoy_N / 1e3)
        drag.append(res.F_drag_N / 1e3)
        anchor.append(float(reference_anchor(state, hull, speed)["total_N"]) / 1e3)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(beams, shear, width=3.0, label=r"granular shear $\mu(I)$")
    ax.bar(beams, buoy, width=3.0, bottom=shear, label="buoyancy displacement")
    ax.bar(beams, drag, width=3.0,
           bottom=[a + b for a, b in zip(shear, buoy)], label="underwater keel drag")
    ax.plot(beams, anchor, "k--o", label="reference anchor (in-repo brash)")
    ax.set_xlabel("beam B [m]")
    ax.set_ylabel("resistance [kN]")
    ax.set_title("Brash-channel resistance decomposition, H=1 m, phi=0.3, v=3 m/s")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def fig_speed(state: BrashIceState = STANDARD_BRASH, beam: float = 25.0,
              closure: Closure = None):
    """R(v) on linear axes, and the log-log power-law fit."""
    closure = closure or Closure.from_ledger()
    hull = standard_hull(beam)
    speeds: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    records = speed_sweep(state, hull, speeds, closure)
    totals = [r["total_N"] / 1e3 for r in records]
    fit_v = [v for v in speeds if v >= 1.0]
    fit_r = [r["total_N"] / 1e3 for v, r in zip(speeds, records) if v >= 1.0]
    n = (math.log(fit_r[-1] / fit_r[0]) / math.log(fit_v[-1] / fit_v[0]))
    ref = fit_r[0] * (1.0 / fit_v[0]) ** n

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].plot(speeds, totals, "o-", label="model total")
    axes[0].set_xlabel("speed v [m/s]")
    axes[0].set_ylabel("resistance [kN]")
    axes[0].set_title(f"R(v), B={beam:.0f} m (mu(I) rate dependence)")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].loglog(fit_v, fit_r, "o", label="model, 1-5 m/s")
    axes[1].loglog(fit_v, [ref * v ** n for v in fit_v], "k--",
                   label=f"fit R ~ v^{n:.2f}")
    axes[1].set_xlabel("v [m/s]"); axes[1].set_ylabel("R [kN]")
    axes[1].set_title("power-law exponent (criterion d)")
    axes[1].grid(alpha=0.3, which="both"); axes[1].legend(fontsize=8)
    return fig


def fig_porosity(state: BrashIceState = STANDARD_BRASH, beam: float = 25.0,
                 speed: float = 3.0, closure: Closure = None):
    """Component response to porosity (criterion e)."""
    closure = closure or Closure.from_ledger()
    hull = standard_hull(beam)
    porosities = (0.2, 0.25, 0.3, 0.35, 0.4)
    records = porosity_sweep(state, hull, speed, porosities, closure)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(porosities, [r["F_shear_N"] / 1e3 for r in records], "o-", label="shear")
    ax.plot(porosities, [r["F_buoy_N"] / 1e3 for r in records], "s-", label="buoyancy")
    ax.plot(porosities, [r["F_drag_N"] / 1e3 for r in records], "^-", label="keel drag")
    ax.plot(porosities, [r["total_N"] / 1e3 for r in records], "k--", label="total")
    ax.set_xlabel("layer porosity phi [-]")
    ax.set_ylabel("resistance [kN]")
    ax.set_title("Porosity sensitivity, B=25 m, H=1 m, v=3 m/s")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    return fig


def fig_mu_i(mu_calibration: Optional[Dict[str, object]] = None,
             state: BrashIceState = STANDARD_BRASH, beam: float = 25.0,
             speed: float = 3.0, closure: Closure = None):
    """mu(I): measured DEM points, the Jop law, and the model's operating point."""
    closure = closure or Closure.from_ledger()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    i_grid = [10 ** (-3 + 4.0 * k / 120.0) for k in range(121)]
    law = JopParameters()
    ax.semilogx(i_grid, [law.mu(i) for i in i_grid], "k-",
                label=r"Jop law $\mu_s=%.2f,\mu_2=%.2f,I_0=%.2f$"
                      % (law.mu_s, law.mu_2, law.i_0))
    if mu_calibration:
        params = JopParameters(**mu_calibration["parameters"])
        ax.semilogx(i_grid, [params.mu(i) for i in i_grid], "b--",
                    label="DEM steady-shear fit")
        pts = mu_calibration.get("points") or []
        if pts:
            ax.semilogx([p["inertial_number_measured"] for p in pts],
                        [p["mu_measured"] for p in pts], "bo",
                        label="DEM servo-shear measurements")
    hull = standard_hull(beam)
    res = brash_resistance(state, hull, speed, closure)
    ax.plot([res.inertial_number], [res.mu_effective], "r*", markersize=14,
            label=f"brash channel operating point (I={res.inertial_number:.2f})")
    ax.set_xlabel("inertial number I [-]")
    ax.set_ylabel(r"effective friction $\mu=\tau/P$ [-]")
    ax.set_title("mu(I) closure: calibration and the brash-channel regime")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    return fig
