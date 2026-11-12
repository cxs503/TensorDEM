"""Hasimoto array-baseline audit of the point-particle CFD-DEM self-response.

Motivation
----------
``tensordem.point_cfd_dem`` passes its single-particle gate against the
*unbounded* Stokes terminal velocity ``v_t`` by means of a "self-response"
correction ``C_eff = C / (1 - C R)`` (:meth:`SpectralPeriodicFluid.self_response`).
The module docstring of :func:`verification.stokes_terminal_velocity_case`
asserts that the physical finite-domain effect of the periodic box is
``3 pi d / L ~ 0.3 %`` at ``L = 32 d`` and therefore negligible.

That assertion is testable against an independent literature anchor: the
dilute limit of a simple cubic array of equal spheres (Hasimoto 1959;
coefficients re-derived by Sangani & Acrivos 1982) gives a *drag enhancement*

    F / (6 pi mu a U) = 1 + 1.7601 phi^(1/3) + phi - 1.5593 phi^2 + O(phi^(8/3))

so a sphere in a periodic cube of side ``L`` (periodic images = a simple cubic
array with lattice constant ``L``) settles at ``U = v_t / (1 + C_H)``, i.e.
4.43 % below ``v_t`` at ``L = 32 d``, 17.8 % below at ``L = 8 d``.  This is a
*physical* effect of the discretised domain, not a numerical artefact, and it
is one to two orders of magnitude larger than the 0.3 % claimed in the module.

The audit therefore asks three questions, all answered by measurement:

Q1 (operator)   How large is the module's own steady self-response ``C R``
                compared with the literature array enhancement ``C_H`` at the
                same solid fraction, and how does it scale with the grid
                (``h``) and the box (``L``)?
Q2 (dynamics)   Once the periodic back-flow is fully developed (``t >> L^2/nu``,
                which the existing gate never reaches: it stops at ``t = 2``
                while ``L^2/nu = 2048``), what settling velocity does the
                *uncorrected* point-particle model give?  Does it reproduce
                the Hasimoto array solution, i.e. does the raw model already
                contain the correct dilute-array physics?
Q3 (correction) Does the module's ``self_response="spectral"`` mode reproduce
                the array solution, or does it remove the physical array
                back-flow together with the numerical self-field?  Is a
                "subtract the physical part only" variant
                (``R_eff = R - C_H / C``, mode ``array_subtracted``) closer?

Outputs (this directory)
------------------------
``result.json``  every measured number (operator sweep + settling runs)
``tables.md``    same numbers as markdown tables
``figures/*.png`` operator comparison, settling transients, steady comparison

Usage::

    PYTHONPATH=src python benchmarks/point_particle_cfd_dem/hasimoto_array.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.point_cfd_dem import closures  # noqa: E402
from tensordem.point_cfd_dem.coupling import PointParticleCFDDEM  # noqa: E402
from tensordem.point_cfd_dem.fluid import SpectralPeriodicFluid  # noqa: E402
from tensordem.point_cfd_dem.particles import (  # noqa: E402
    ParticleSet,
    PointParticleModel,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = HERE / "hasimoto_array"


# ---------------------------------------------------------------------------
# literature anchor
# ---------------------------------------------------------------------------
def hasimoto_drag_enhancement(phi: float) -> float:
    """``C_H = F/(6 pi mu a U) - 1`` for a simple cubic array (Hasimoto 1959).

    Three terms of the lattice-sum expansion are kept.  Over the solid
    fractions audited here (``phi <= 8.2e-3``) the neglected ``O(phi^(8/3))``
    term is below ``1e-6`` and even the retained ``phi^2`` term is below
    ``1.1e-4``, so the series value is exact to better than 0.1 % of ``C_H``.
    """
    if phi <= 0.0:
        return 0.0
    c = phi ** (1.0 / 3.0)
    return 1.7601 * c + phi - 1.5593 * phi * phi


def hasimoto_settling_ratio(phi: float) -> float:
    """``U_array / v_t = 1 / (1 + C_H)`` in the container frame."""
    return 1.0 / (1.0 + hasimoto_drag_enhancement(phi))


def hasimoto_series_truncation(phi: float) -> float:
    """Magnitude of the first neglected term, ``3.979 phi^(8/3)``."""
    return 3.979 * phi ** (8.0 / 3.0)


# ---------------------------------------------------------------------------
# Q1: operator sweep (no dynamics)
# ---------------------------------------------------------------------------
def operator_point(h_over_d: float, l_over_d: float, mu: float = 0.5,
                   rho_f: float = 1.0, dt: float = 0.01, diameter: float = 1.0,
                   frac=(0.5, 0.5, 0.5)):
    """``C R`` at one grid/box pair, with the drag coefficient ``C = 3 pi mu d``."""
    h = h_over_d * diameter
    cells = int(round(l_over_d * diameter / h))
    fluid = SpectralPeriodicFluid((cells,) * 3, h, rho=rho_f, mu=mu, advect=True)
    pos = np.array(frac, dtype=float) * fluid.extent
    R = float(fluid.self_response(np.atleast_2d(pos), dt, elapsed=None)[0])
    C = float(closures.stokes_drag_coefficient(mu, diameter))
    phi = (math.pi / 6.0) * diameter ** 3 / (l_over_d * diameter) ** 3
    return {
        "h_over_d": h_over_d,
        "L_over_d": l_over_d,
        "cells": cells,
        "phi": phi,
        "C_R": C * R,
        "C_H": hasimoto_drag_enhancement(phi),
        "ratio_to_hasimoto": (C * R) / hasimoto_drag_enhancement(phi),
        "frac": list(frac),
    }


def operator_sweep(quick: bool = False):
    boxes = (4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
    if quick:
        boxes = (8.0, 32.0)
    rows = []
    for h in (2.0, 1.0):
        for L in boxes:
            for frac in ((0.5, 0.5, 0.5), (0.23, 0.37, 0.11)):
                rows.append(operator_point(h, L, frac=frac))
    return rows


# ---------------------------------------------------------------------------
# Q2/Q3: long-time settling runs
# ---------------------------------------------------------------------------
def settle_case(l_over_d: float, h_over_d: float, mu: float, dt: float,
                n_viscous_times: float, mode: str, rho_f: float = 1.0,
                rho_p: float = 2.0, g: float = 0.5, diameter: float = 1.0,
                samples: int = 240):
    """Settle one sphere until the periodic back-flow is fully developed.

    ``mode``:
      ``none``              no self-correction at all (raw point-particle model)
      ``spectral``          module default ``self_response="spectral"``
      ``array_subtracted``  ``R_eff = max(R - C_H/C, 0)``: remove only the
                            numerical self-field, keep the physical array
                            back-flow in the drag
    """
    h = h_over_d * diameter
    cells = int(round(l_over_d * diameter / h))
    fluid = SpectralPeriodicFluid((cells,) * 3, h, rho=rho_f, mu=mu, advect=True)
    centre = np.array(fluid.extent) / 2.0
    part = ParticleSet.create([centre], [[0.0, 0.0, 0.0]],
                              radius=diameter / 2.0, density=rho_p)
    model = PointParticleModel(rho_f=rho_f, mu=mu,
                               gravity=np.array([0.0, 0.0, -g]))
    v_t = float(closures.terminal_velocity_stokes(rho_p, rho_f, g, diameter, mu))
    L = fluid.extent[0]
    phi = float(part.volume[0]) / L ** 3
    C = float(closures.stokes_drag_coefficient(mu, diameter))
    C_H = hasimoto_drag_enhancement(phi)
    R_array = C_H / C
    nu = mu / rho_f
    t_visc = L * L / nu
    t_end = n_viscous_times * t_visc
    dt = min(dt, 0.30 * float(closures.relaxation_time_stokes(rho_p, diameter, mu)))
    n_steps = int(round(t_end / dt))
    every = max(1, n_steps // samples)

    sim = PointParticleCFDDEM(
        fluid=fluid, particles=part, model=model, zero_net_flux=True,
        self_disturbance=(mode != "none"), exchange_terms=("drag",),
        drag_filter_width=None,
        self_response=None if mode == "none" else "spectral",
    )
    if mode == "array_subtracted":
        original = fluid.self_response

        def patched(positions, dt, elapsed=None):
            R = original(positions, dt, elapsed=elapsed)
            return np.maximum(R - R_array, 0.0)

        fluid.self_response = patched
    elif mode not in ("none", "spectral"):
        raise ValueError(f"unknown mode {mode!r}")

    t_hist, vz_hist = [], []
    t0 = time.time()
    for k in range(n_steps):
        sim.step(dt)
        if k % every == 0 or k == n_steps - 1:
            t_hist.append(sim.time)
            vz_hist.append(float(part.velocity[0, 2]))
    wall = time.time() - t0

    # steady window: last 20 % of the run
    track = np.array(vz_hist)
    n_tail = max(3, len(track) // 5)
    u_meas = abs(float(np.mean(track[-n_tail:])))
    return {
        "mode": mode,
        "L_over_d": l_over_d,
        "h_over_d": h_over_d,
        "cells": cells,
        "mu": mu,
        "dt": dt,
        "phi": phi,
        "C_H": C_H,
        "R_array_subtracted_over_C": R_array * C,
        "v_t": v_t,
        "t_viscous": t_visc,
        "t_end": t_end,
        "n_viscous_times": n_viscous_times,
        "n_steps": n_steps,
        "u_measured": u_meas,
        "u_over_vt": u_meas / v_t,
        "hasimoto_u_over_vt": hasimoto_settling_ratio(phi),
        "error_vs_stokes": abs(u_meas - v_t) / v_t,
        "error_vs_hasimoto": abs(u_meas - hasimoto_settling_ratio(phi) * v_t)
        / (hasimoto_settling_ratio(phi) * v_t),
        "t": t_hist,
        "vz_over_vt": [v / v_t for v in vz_hist],
        "wall_seconds": wall,
    }


DEFAULT_CASES = (
    # (L/d, h/d, mu, dt, n_viscous_times)
    (8.0, 2.0, 5.0, 0.005, 6.0),
    (12.0, 2.0, 5.0, 0.005, 6.0),
)

EXTRA_CASES = (
    (16.0, 2.0, 5.0, 0.005, 4.0),
)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def markdown_tables(result):
    lines = []
    lines.append("### Q1 operator sweep — module self-response vs Hasimoto array\n")
    lines.append("`C R` is the module's steady self-response in drag units; "
                 "`C_H` is the literature simple-cubic array enhancement at the "
                 "same solid fraction.\n")
    lines.append("| L/d | h/d | cells | frac | phi | C R | C_H | C R / C_H |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in result["operator_sweep"]:
        frac = "centre" if r["frac"][0] == 0.5 else "off-centre"
        lines.append(f"| {r['L_over_d']:.1f} | {r['h_over_d']:.1f} | {r['cells']} | "
                     f"{frac} | {r['phi']:.4e} | {r['C_R']:.4f} | {r['C_H']:.4f} | "
                     f"{r['ratio_to_hasimoto']:.1f} |")
    lines.append("")
    lines.append("### Q2/Q3 long-time settling — measured vs Stokes and Hasimoto\n")
    lines.append("| L/d | mode | phi | C_H | U/v_t measured | Stokes 1.000 | "
                 "Hasimoto U/v_t | error vs Stokes | error vs Hasimoto | wall (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in result["settling"]:
        lines.append(
            f"| {r['L_over_d']:.1f} | {r['mode']} | {r['phi']:.4e} | {r['C_H']:.4f} | "
            f"{r['u_over_vt']:.4f} | 1.0000 | {r['hasimoto_u_over_vt']:.4f} | "
            f"{100 * r['error_vs_stokes']:.2f} % | {100 * r['error_vs_hasimoto']:.2f} % | "
            f"{r['wall_seconds']:.1f} |")
    return "\n".join(lines) + "\n"


def figures(result, out_dir: Path):
    figs = []
    op = result["operator_sweep"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for h, marker in ((2.0, "o"), (1.0, "s")):
        sel = [r for r in op if r["h_over_d"] == h and r["frac"][0] == 0.5]
        ax.loglog([r["phi"] for r in sel], [r["C_R"] for r in sel], marker + "-",
                  label=f"module C R (h = {h:g} d, centre)")
    sel = [r for r in op if r["h_over_d"] == 1.0 and r["frac"][0] == 0.5]
    ax.loglog([r["phi"] for r in sel], [r["C_H"] for r in sel], "k--",
              label="Hasimoto array C_H")
    ax.set_xlabel(r"solid fraction $\phi = (\pi/6)(d/L)^3$")
    ax.set_ylabel(r"drag enhancement / self-response")
    ax.set_title("Module self-response vs literature array correction")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    p = out_dir / "fig1_self_response_vs_hasimoto.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({"name": "fig1_self_response_vs_hasimoto.png",
                 "caption": "C R of the module's self-response against the "
                            "Hasimoto simple-cubic array enhancement at the same "
                            "solid fraction (log-log)."})

    big = [r for r in result["settling"] if r["L_over_d"] == max(
        s["L_over_d"] for s in result["settling"])]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r in big:
        ax.plot(r["t"], r["vz_over_vt"], label=f"{r['mode']}")
        ax.axhline(r["hasimoto_u_over_vt"], ls=":", lw=1.0,
                   color=ax.lines[-1].get_color())
    first = big[0]
    ax.axhline(1.0, color="k", lw=1.0, label="Stokes (unbounded)")
    ax.axhline(first["hasimoto_u_over_vt"], color="k", ls="--", lw=1.0,
               label=f"Hasimoto array (phi={first['phi']:.2e})")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$|u_p| / v_t$")
    ax.set_title(f"Steady settling in a periodic box, L = {first['L_over_d']:g} d "
                 f"(t = {first['n_viscous_times']:g} L^2/nu)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    p = out_dir / "fig2_settling_transients.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({"name": "fig2_settling_transients.png",
                 "caption": "Container-frame settling velocity history for the "
                            "three self-response modes, compared with the Stokes "
                            "and Hasimoto anchors."})

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for mode, marker in (("none", "o"), ("spectral", "s"),
                         ("array_subtracted", "^")):
        sel = [r for r in result["settling"] if r["mode"] == mode]
        if sel:
            ax.plot([r["phi"] for r in sel], [r["u_over_vt"] for r in sel],
                    marker + "-", label=mode)
    phis = np.linspace(2e-4, 5e-3, 60)
    ax.plot(phis, [hasimoto_settling_ratio(p) for p in phis], "k--",
            label="Hasimoto array")
    ax.axhline(1.0, color="k", lw=1.0, label="Stokes")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$U/v_t$")
    ax.set_title("Steady settling vs solid fraction")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    p = out_dir / "fig3_settling_vs_phi.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({"name": "fig3_settling_vs_phi.png",
                 "caption": "Steady container-frame settling velocity against "
                            "solid fraction for the three modes, with the "
                            "Hasimoto and Stokes anchors."})
    return figs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="operator sweep only")
    ap.add_argument("--extra", action="store_true", help="also the L = 12 d case")
    ap.add_argument("--modes", default="none,spectral,array_subtracted")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    result = {
        "node": "point_cfd_dem_hasimoto_array_audit",
        "references": {
            "hasimoto_1959": "F = 6 pi mu a U (1 + 1.7601 phi^(1/3) + phi "
                             "- 1.5593 phi^2 + ...)",
            "sangani_acrivos_1982": "simple cubic array coefficients",
            "stokes": "v_t = (rho_p - rho_f) g d^2 / (18 mu)",
        },
        "operator_sweep": operator_sweep(quick=args.quick),
        "settling": [],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if not args.quick:
        cases = list(DEFAULT_CASES) + (list(EXTRA_CASES) if args.extra else [])
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        for (L, h, mu, dt, nvt) in cases:
            for mode in modes:
                rec = settle_case(L, h, mu, dt, nvt, mode)
                result["settling"].append(rec)
                print(f"L/d={L:g} h/d={h:g} mode={mode:15s} "
                      f"U/v_t={rec['u_over_vt']:.4f} "
                      f"Hasimoto={rec['hasimoto_u_over_vt']:.4f} "
                      f"err_H={100 * rec['error_vs_hasimoto']:.2f}% "
                      f"({rec['wall_seconds']:.1f} s)", flush=True)
                (OUT / "result.json").write_text(json.dumps(result, indent=1))
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    result["figures"] = figures(result, OUT / "figures") if result["settling"] else []
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    (OUT / "tables.md").write_text(markdown_tables(result))
    print(markdown_tables(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())