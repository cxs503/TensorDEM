#!/usr/bin/env python3
"""Figures for the m4729 test-expansion mission.

Two figures are produced from the production code paths exercised by the new
tests (no hand-copied numbers):

1. ``fig1_shear_convergence.png`` -- DEM3D direct-shear cross-section series
   ``d(n) = 1 - G_eff/C44`` with the boundary-layer fits of
   :func:`tensordem.shear_convergence.boundary_layer_fit` (models A/B/C) and the
   extrapolated limit.
2. ``fig2_free_float_decay.png`` -- RK4 free-float heave from
   :func:`tensordem.buoyancy_verification.integrate_free_float` against the
   linearised reference/envelope of the same module.

Usage: ``TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 scripts/plot_m4729_test_coverage.py``
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem import buoyancy_verification as bv  # noqa: E402
from tensordem import shear_convergence as sc  # noqa: E402

OUT = ROOT / "artifacts" / "m4729_negative_controls"


def figure_shear_convergence() -> dict[str, float]:
    sizes = [5, 7, 9, 11, 13]
    series = sc.cross_section_series(sizes)
    rows = series["rows"]
    fit = series["fit"]
    n = np.array([row["n"] for row in rows], dtype=float)
    deficit = np.array([row["deficit"] for row in rows])

    dense = np.linspace(4.0, 60.0, 400)
    x = 1.0 / dense

    fig, ax = plt.subplots(2, 1, figsize=(6.2, 6.4), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(n, deficit, "o", color="k", label="DEM3D measured  $1-G_{eff}/C_{44}$")
    for name, colour in (("A", "tab:blue"), ("B", "tab:orange"), ("C", "tab:green")):
        powers = fit["models"][name]["powers"]
        coef = [fit["models"][name]["coefficients"][f"c{p}"] for p in powers]
        curve = sum(c * x ** p for p, c in zip(powers, coef))
        ax[0].plot(dense, curve, "--", color=colour,
                   label=f"model {name} (rms {fit['models'][name]['rms_residual']:.2e})")
    ax[0].axhline(fit["gate"], color="tab:red", lw=1.0, ls=":",
                  label=f"3 % gate; predicted $n={fit['predicted_n_for_gate']:.1f}$")
    ax[0].axhline(fit["asymptotic_deficit_model_C"], color="tab:green", lw=0.8, alpha=0.6)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("deficit  $1-G_{eff}/C_{44}$  [-]")
    ax[0].set_title("DEM3D direct shear: cross-section convergence and boundary-layer fits")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=7.5, loc="upper right")

    powers = fit["models"]["B"]["powers"]
    coef = [fit["models"]["B"]["coefficients"][f"c{p}"] for p in powers]
    residual = np.array([sum(c * (1.0 / nn) ** p for p, c in zip(powers, coef)) - d
                         for nn, d in zip(n, deficit)])
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].plot(n, residual, "s", color="tab:orange",
               label=f"model B residual (max {np.max(np.abs(residual)):.1e})")
    ax[1].legend(fontsize=7.5, loc="upper right")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("cross-section  n  (particles per side, $h/a=2$)")
    ax[1].set_ylabel("model B residual  [-]")
    ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig1_shear_convergence.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)

    return {
        "observed_order": fit["observed_order"],
        "extrapolated_G_over_C44_model_C": fit["extrapolated_G_over_C44_model_C"],
        "richardson_extrapolated_G_over_C44": fit["richardson_extrapolated_G_over_C44"],
        "predicted_n_for_gate": fit["predicted_n_for_gate"],
        "deficit_at_largest_n": fit["deficit_at_largest_n"],
        "model_B_rms": fit["models"]["B"]["rms_residual"],
        "figure": path.name,
    }


def figure_free_float() -> dict[str, float]:
    params = bv.FreeFloatParameters(duration=6.0, dt=1e-2)
    h_eq = params.equilibrium_height()
    history = bv.integrate_free_float(params, h_eq + params.amplitude, sample_every=5)
    t = np.array(history["time"])
    z = np.array(history["height"]) - h_eq
    analytic = np.array(bv.analytic_displacement(t.tolist(), params))
    envelope = np.array([bv.analytic_envelope(ti, params) for ti in t])

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(t, z * 1e3, color="k", lw=1.4, label="RK4 on the production force law")
    ax.plot(t, analytic * 1e3, "--", color="tab:red", lw=1.1, label="linearised reference")
    ax.plot(t, envelope * 1e3, ":", color="tab:blue", label=r"envelope $A e^{-\zeta\omega_n t}$")
    ax.plot(t, -envelope * 1e3, ":", color="tab:blue")
    ax.set_xlabel("time  [s]")
    ax.set_ylabel(r"heave about the draft  $(y-h_{eq})$  [mm]")
    ax.set_title("Free-float heave: simulated vs linearised reference "
                 f"($\\omega_n$={params.omega_n():.3f} rad/s, $\\zeta$={params.damping_ratio():.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig2_free_float_decay.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)

    tail = t >= 4.0
    peak = float(np.max(np.abs(z)))
    return {
        "h_eq_m": float(h_eq),
        "omega_n_rad_s": float(params.omega_n()),
        "zeta": float(params.damping_ratio()),
        "waveform_max_rel_error_tail": float(
            np.max(np.abs(z[tail] - analytic[tail])) / params.amplitude),
        "peak_amplitude_mm": peak * 1e3,
        "figure": path.name,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {"shear": figure_shear_convergence(), "free_float": figure_free_float()}
    (OUT / "figure_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())