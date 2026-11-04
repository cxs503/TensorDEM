"""Cross-section convergence study of the DEM3D direct-shear apparent modulus.

Closes the debt left open by ``docs/dem3d_shear_static_reference.md`` §5: a
*shape-independent* shear modulus could not be extracted from the direct-shear
specimen because the 9x5 -> 21x21 trend was too short to separate a lateral
free-surface boundary layer (``O(1/n)``, extrapolable) from a measurement bias.

Writes ``report.json``, ``series.csv``, ``summary.md`` and
``figures/convergence.png`` under ``docs/dem3d-shear-cross-section-convergence/``.

    PYTHONPATH=src python scripts/audit_dem3d_shear_cross_section_convergence.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensordem.shear_convergence import (
    ACCURACY_GATE,
    cross_section_series,
    surface_lag_profile,
)
from tensordem.shear_specimen import shear_modulus_closed_form
from tensordem.dem3d import DEM3DConfig

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "dem3d-shear-cross-section-convergence"
SIZES = [5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 29, 33, 37, 41, 45, 49, 55, 61]
LAG_SIZES = [17, 41, 61]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    series = cross_section_series(SIZES, nz=3, gate=ACCURACY_GATE)
    profiles = [surface_lag_profile(n=n) for n in LAG_SIZES]

    rows = series["rows"]
    with (OUT / "series.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema": "tensordem-dem3d-shear-cross-section-convergence-v1",
        "description": "Square cross-section series at fixed h/a = 2 (nz = 3), no fracture, "
                       "linear elastic static equilibrium of the direct-shear specimen.",
        "config": {
            "nx": "swept", "ny": "= nx", "nz": 3,
            "bond_stiffness_N_per_m": DEM3DConfig().bond_stiffness,
            "radius_m": DEM3DConfig().radius,
            "density_kg_per_m3": DEM3DConfig().density,
            "gamma": series["gamma"],
            "C44_Pa": shear_modulus_closed_form(DEM3DConfig()),
            "accuracy_gate": ACCURACY_GATE,
        },
        "series": series,
        "surface_lag": profiles,
    }
    (OUT / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fit = series["fit"]
    model_b = fit["models"]["B"]
    model_c = fit["models"]["C"]
    finite = [row for row in rows if row["deficit"] < ACCURACY_GATE]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    inv_n = np.array([1.0 / row["n"] for row in rows])
    g_eff = np.array([row["G_effective_over_C44"] for row in rows])
    area = np.array([row["area_ratio"] for row in rows])
    grid = np.linspace(0.0, inv_n.max(), 200)
    c1, c2 = model_b["coefficients"]["c1"], model_b["coefficients"]["c2"]
    axes[0].plot(inv_n, g_eff, "o", label=r"$G_\mathrm{eff}/C_{44}$ (static solve)")
    axes[0].plot(inv_n, area, "s", mfc="none", label=r"$A_\mathrm{eff}/A_\mathrm{nom}$ (bond census)")
    axes[0].plot(grid, 1.0 - c1 * grid - c2 * grid ** 2, "-",
                 label=r"fit $1-d=1-c_1/n-c_2/n^2$")
    axes[0].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[0].axhline(1.0 - ACCURACY_GATE, color="tab:red", lw=0.9, ls=":")
    axes[0].text(0.012, 1.0 - ACCURACY_GATE - 0.035, "3 % gate (red dotted)", color="tab:red", fontsize=8)
    axes[0].set_xlabel(r"$1/n$  (square cross-section $n\times n$)")
    axes[0].set_ylabel("normalised modulus / area")
    axes[0].set_title("Cross-section convergence at fixed $h/a=2$")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(alpha=0.3)

    for profile in profiles:
        bins = profile["bins"]
        k = np.array([b["distance_over_spacing"] for b in bins], dtype=float)
        lag = np.array([b["mean_lag"] for b in bins], dtype=float)
        axes[1].plot(k, lag, "o-", ms=3.5, label=f"n={profile['n']} (lambda/a={profile['decay_length_over_spacing']:.2f})")
    axes[1].axhline(0.0, color="k", lw=0.8, ls="--")
    axes[1].set_xlabel("distance from nearest lateral face  [lattice spacings $a$]")
    axes[1].set_ylabel(r"mid-layer deviation $\delta=u_x/(\gamma z)-1$")
    axes[1].set_title("Boundary layer: 1.5$a$ compliant skin")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "convergence.png", dpi=150)
    plt.close(fig)

    summary = [
        "# DEM3D direct-shear cross-section convergence — generated summary",
        "",
        f"Square cross-sections `n x n`, `nz = 3` (`h/a = 2`), no fracture, gamma = {series['gamma']}; "
        f"`C44 = 7k/(3a) = {payload['config']['C44_Pa']:.4f} Pa`.",
        "",
        "| n | particles | G_eff/C44 | deficit | A_eff/A_nom |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        summary.append(
            f"| {row['n']} | {row['particle_count']} | {row['G_effective_over_C44']:.5f} | "
            f"{row['deficit']:.5f} | {row['area_ratio']:.5f} |")
    summary += [
        "",
        "## Boundary-layer fit  d(n) = c1/n + c2/n^2",
        "",
        f"- c1 = {c1:.4f}, c2 = {c2:.4f}",
        f"- RMS residual = {model_b['rms_residual']:.2e}, max |residual| = {model_b['max_abs_residual']:.2e}",
        f"- 3-parameter model bias c0 = {model_c['coefficients']['c0']:.2e} "
        f"(=> G_inf/C44 = {fit['extrapolated_G_over_C44_model_C']:.5f})",
        f"- observed order from the three finest sizes: p = {fit['observed_order']:.3f}",
        f"- predicted n for the {ACCURACY_GATE:.0%} gate: {fit['predicted_n_for_gate']:.1f}",
        f"- largest measured n = {fit['largest_n_measured']}, deficit {fit['deficit_at_largest_n']:.5f}"
        f" (gate met: {fit['gate_met_by_measurement']}); measured n below gate: "
        f"{[int(r['n']) for r in finite]}",
        "",
        "## Surface lag (mid layer)",
        "",
        "| n | lambda/a | delta at d=1a | delta at d=2a | centre |",
        "| --- | --- | --- | --- | --- |",
    ]
    for profile in profiles:
        by_k = {b["distance_over_spacing"]: b["mean_lag"] for b in profile["bins"]}
        summary.append(
            f"| {profile['n']} | {profile['decay_length_over_spacing']:.3f} | "
            f"{by_k.get(1.0, float('nan')):.4f} | {by_k.get(2.0, float('nan')):.4f} | "
            f"{profile['lag_at_center']:.2e} |")
    (OUT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"wrote {OUT/'report.json'}, series.csv, summary.md, figures/convergence.png")
    print(f"c1={c1:.4f} c2={c2:.4f} rms={model_b['rms_residual']:.2e} "
          f"c0={model_c['coefficients']['c0']:.2e} Ginf/C44={fit['extrapolated_G_over_C44_model_C']:.5f} "
          f"p={fit['observed_order']:.3f} n_gate={fit['predicted_n_for_gate']:.1f}")
    print("deficits:", " ".join(f"{r['n']}:{r['deficit']:.4f}" for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())