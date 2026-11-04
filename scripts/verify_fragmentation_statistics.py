#!/usr/bin/env python3
"""Independent verification of the ``threefield`` fragmentation / Weibull layer.

Protocol id: ``tensordem-fragmentation-statistics-verification-v1``

The three-field report grades fragment-size statistics through
``fragmentation.fit_weibull`` / ``fit_weibull_cdf`` / ``kolmogorov_smirnov``.
None of those estimators was ever checked against an *independent*
implementation, so their evidence grade was ``WITHHELD``: the numbers in the
report could have been produced by any estimator at all.

This script closes that debt with references that are not the module itself:

* the bending criterion is re-derived from the beam closed form
  ``sigma_b = 3 rho_w C_d u'^2 L^2 / (8 h)`` and ``L_crit`` is obtained as the
  root of ``sigma_b(L) - sigma_f = 0`` (bisection, independent of the module);
* the Weibull CDF is compared with ``scipy.stats.weibull_min.cdf``;
* both Weibull fits are compared with the ``scipy`` maximum-likelihood fit;
* the KS statistic is compared with ``scipy.stats.kstest`` for the same
  reference CDF;
* mass conservation of the fragmentation cascade is measured against a mass
  computed outside the module (``rho * L^2 * h``), for several ice densities.

Outputs (``artifacts/threefield_fragmentation/``):

* ``verification.json`` — every number quoted in the doc / report;
* ``error_table.csv`` — the per-anchor error table;
* ``ks_bias.csv`` — plotting-position vs exact KS statistic versus sample size;
* ``figures/fragmentation_statistics_verification.png`` — three-panel figure.

Run: ``python scripts/verify_fragmentation_statistics.py``
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import scipy.stats as st
import torch

from tensordem.threefield.coupling import fragmentation_stage
from tensordem.threefield.fragmentation import (
    BreakupProperties,
    bending_stress_pa,
    critical_floe_length,
    fit_weibull,
    cdf_agreement_r2,
    fit_weibull_cdf,
    fragment,
    kolmogorov_smirnov,
    kolmogorov_smirnov_plotting_position,
    weibull_cdf,
)

OUT = Path("artifacts/threefield_fragmentation")
REFERENCE_WEIBULL = {"shape": 1.8, "scale": 0.12}  # report gate, docs/threefield


def _rel(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference) if reference else float("nan")


def bending_anchor() -> list[dict[str, float]]:
    props = BreakupProperties()
    rows = []
    for length, u_rms in [(0.5, 3.0), (2.0, 1.0), (5.0, 0.2), (12.0, 0.05)]:
        measured = bending_stress_pa(length, u_rms, props)
        closed = (
            3.0 * props.water_density_kg_m3 * props.drag_coefficient * u_rms**2 * length**2
            / (8.0 * props.thickness_m)
        )
        rows.append(
            {
                "anchor": "bending_stress",
                "case": f"L={length}_u={u_rms}",
                "measured": measured,
                "reference": closed,
                "relative_error": _rel(measured, closed),
            }
        )
    return rows


def critical_length_anchor() -> list[dict[str, float]]:
    """Solve sigma_b(L) - sigma_f = 0 by bisection, without touching the module."""
    props = BreakupProperties()
    rows = []
    for u_rms in [0.05, 0.25, 1.0, 2.5]:
        low, high = 1e-6, 1e3
        for _ in range(200):
            mid = 0.5 * (low + high)
            load = 3.0 * props.water_density_kg_m3 * props.drag_coefficient * u_rms**2 * mid**2
            stress = load / (8.0 * props.thickness_m)
            if stress > props.flexural_strength_pa:
                high = mid
            else:
                low = mid
        root = 0.5 * (low + high)
        measured = critical_floe_length(u_rms, props)
        rows.append(
            {
                "anchor": "critical_length",
                "case": f"u_rms={u_rms}",
                "measured": measured,
                "reference": root,
                "relative_error": _rel(measured, root),
            }
        )
    return rows


def weibull_cdf_anchor() -> list[dict[str, float]]:
    rows = []
    for shape, scale in [(1.0, 0.2), (1.8, 0.12), (3.5, 2.0)]:
        for x in [0.01, 0.05, 0.12, 0.5, 2.0]:
            measured = weibull_cdf(x, shape, scale)
            reference = float(st.weibull_min.cdf(x, shape, scale=scale))
            rows.append(
                {
                    "anchor": "weibull_cdf",
                    "case": f"k={shape}_lambda={scale}_x={x}",
                    "measured": measured,
                    "reference": reference,
                    "relative_error": _rel(measured, reference) if reference else 0.0,
                }
            )
    return rows


def estimator_anchor() -> tuple[list[dict[str, float]], dict[str, object]]:
    """Fit both estimators to synthetic Weibull samples and to a cascade census."""
    rows = []
    details: dict[str, object] = {"samples": {}, }
    for size in [60, 150, 400]:
        samples = st.weibull_min.rvs(
            REFERENCE_WEIBULL["shape"], scale=REFERENCE_WEIBULL["scale"], size=size, random_state=20261031
        )
        mle_shape, _, mle_scale = st.weibull_min.fit(samples, floc=0)
        ls = fit_weibull(list(samples))
        cdf_ls = fit_weibull_cdf(list(samples))
        ks_ls = kolmogorov_smirnov(list(samples), ls["shape"], ls["scale"])
        ks_cdf = kolmogorov_smirnov(list(samples), cdf_ls["shape"], cdf_ls["scale"])
        details["samples"][str(size)] = {
            "mle_shape": float(mle_shape),
            "mle_scale": float(mle_scale),
            "log_ls": {k: float(v) for k, v in ls.items()},
            "cdf_ls": {
                k: float(v) for k, v in cdf_ls.items() if isinstance(v, (int, float))
            },
            "ks_log_ls": ks_ls,
            "ks_cdf_ls": ks_cdf,
        }
        for name, fit in (("log_ls", ls), ("cdf_ls", cdf_ls)):
            rows.append(
                {
                    "anchor": f"weibull_shape_{name}",
                    "case": f"n={size}",
                    "measured": fit["shape"],
                    "reference": float(mle_shape),
                    "relative_error": _rel(fit["shape"], float(mle_shape)),
                }
            )
            rows.append(
                {
                    "anchor": f"weibull_scale_{name}",
                    "case": f"n={size}",
                    "measured": fit["scale"],
                    "reference": float(mle_scale),
                    "relative_error": _rel(fit["scale"], float(mle_scale)),
                }
            )

    # Cascade census from the module's own breakup model (43-300 survivors).
    props = BreakupProperties()
    cascade = fragment(
        [20.0, 12.0, 5.0, 1.0, 0.4], 2.0, props, generator=torch.Generator().manual_seed(7)
    )
    lengths = cascade["lengths_m"]
    mle_shape, _, mle_scale = st.weibull_min.fit(np.asarray(lengths) * 1e3, floc=0)
    ls = fit_weibull(lengths)
    cdf_ls = fit_weibull_cdf(lengths)
    details["cascade"] = {
        "survivors": len(lengths),
        "mle_shape_mm": float(mle_shape),
        "mle_scale_mm": float(mle_scale),
        "log_ls": {k: float(v) for k, v in ls.items()},
        "cdf_ls": {k: float(v) for k, v in cdf_ls.items() if isinstance(v, (int, float))},
        "ks_log_ls": kolmogorov_smirnov(lengths, ls["shape"], ls["scale"]),
        "ks_cdf_ls": kolmogorov_smirnov(lengths, cdf_ls["shape"], cdf_ls["scale"]),
        "cdf_agreement": {
            "cdf_ls": cdf_agreement_r2(lengths, cdf_ls["shape"], cdf_ls["scale"]),
            "log_ls": cdf_agreement_r2(lengths, ls["shape"], ls["scale"]),
            "reference_gate": cdf_agreement_r2(
                lengths, REFERENCE_WEIBULL["shape"], REFERENCE_WEIBULL["scale"]
            ),
        },
    }
    return rows, details


def ks_anchor() -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Exact KS statistic vs SciPy, and the plotting-position bias versus n."""
    rows: list[dict[str, float]] = []
    bias_rows: list[dict[str, float]] = []
    for size in [20, 50, 100, 200, 400, 800]:
        samples = st.weibull_min.rvs(
            REFERENCE_WEIBULL["shape"], scale=REFERENCE_WEIBULL["scale"], size=size, random_state=size
        )
        measured = kolmogorov_smirnov(list(samples), REFERENCE_WEIBULL["shape"], REFERENCE_WEIBULL["scale"])
        reference = float(
            st.kstest(samples, lambda x: st.weibull_min.cdf(x, REFERENCE_WEIBULL["shape"],
                                                           scale=REFERENCE_WEIBULL["scale"])).statistic
        )
        rows.append(
            {
                "anchor": "ks_statistic",
                "case": f"n={size}",
                "measured": measured,
                "reference": reference,
                "relative_error": _rel(measured, reference),
            }
        )
        plotting = kolmogorov_smirnov_plotting_position(
            list(samples), REFERENCE_WEIBULL["shape"], REFERENCE_WEIBULL["scale"]
        )
        bias_rows.append(
            {
                "n": size,
                "ks_exact": measured,
                "ks_scipy": reference,
                "ks_plotting_position": plotting,
                "plotting_position_deficit": (measured - plotting) / measured,
            }
        )
    return rows, bias_rows


def mass_anchor() -> list[dict[str, float]]:
    """Cascade mass conservation, including non-default ice densities."""
    rows = []
    for density in [917.0, 900.0, 950.0]:
        props = BreakupProperties(ice_density_kg_m3=density, thickness_m=0.10)
        cascade = fragment(
            [20.0, 12.0, 5.0, 1.0, 0.4], 2.0, props, generator=torch.Generator().manual_seed(11)
        )
        reference = sum(density * L**2 * props.thickness_m for L in cascade["lengths_m"])
        rows.append(
            {
                "anchor": "cascade_mass",
                "case": f"rho={density}",
                "measured": cascade["final_mass_kg"],
                "reference": reference,
                "relative_error": _rel(cascade["final_mass_kg"], reference),
            }
        )
        # Coupled-stage ledger: inventory masses are built with the same density.
        inventory = [
            {"mass_kg": density * 4.0 * props.thickness_m, "diameter_m": 2.0},
            {"mass_kg": density * 1.0 * props.thickness_m, "diameter_m": 1.0},
        ]
        stage = fragmentation_stage(inventory, 2.0, props)
        rows.append(
            {
                "anchor": "coupling_ledger_residual",
                "case": f"rho={density}",
                "measured": abs(stage["mass_error_kg"]),
                "reference": stage["initial_mass_kg"],
                "relative_error": abs(stage["mass_error_kg"]) / stage["initial_mass_kg"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def figure(path: Path, details: dict, bias_rows: list[dict[str, float]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - viz extra
        return False

    cascade = details["cascade"]
    props = BreakupProperties()
    cascade_runs = fragment(
        [20.0, 12.0, 5.0, 1.0, 0.4], 2.0, props, generator=torch.Generator().manual_seed(7)
    )
    lengths = np.asarray(cascade_runs["lengths_m"])
    order = np.argsort(lengths)
    lengths = lengths[order]
    plotting = (np.arange(1, len(lengths) + 1)) / (len(lengths) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    axes[0].plot(lengths * 1e3, plotting, "o", ms=3.5, label="cascade census (Weibull plotting positions)")
    grid = np.linspace(lengths.min(), lengths.max(), 300)
    axes[0].plot(
        grid * 1e3,
        st.weibull_min.cdf(grid, cascade["cdf_ls"]["shape"], scale=cascade["cdf_ls"]["scale"]),
        "-",
        label=f"CDF-LS fit k={cascade['cdf_ls']['shape']:.2f}, lambda={cascade['cdf_ls']['scale']*1e3:.0f} mm",
    )
    axes[0].plot(
        grid * 1e3,
        st.weibull_min.cdf(grid, cascade["log_ls"]["shape"], scale=cascade["log_ls"]["scale"]),
        "--",
        label=f"log-LS fit k={cascade['log_ls']['shape']:.2f}",
    )
    axes[0].set_xlabel("floe length L [mm]")
    axes[0].set_ylabel("cumulative fraction")
    axes[0].set_title(f"(a) cascade census vs Weibull fits (n={cascade['survivors']})")
    axes[0].legend(fontsize=7)

    sizes = [int(row["n"]) for row in bias_rows]
    axes[1].loglog(sizes, [row["ks_scipy"] for row in bias_rows], "o-", label="exact KS (scipy.kstest)")
    axes[1].loglog(sizes, [row["ks_plotting_position"] for row in bias_rows], "s--",
                   label="plotting-position deviation")
    axes[1].set_xlabel("sample size n")
    axes[1].set_ylabel("statistic")
    axes[1].set_title("(b) KS estimator: exact vs plotting position")
    axes[1].legend(fontsize=7)

    for density, marker in [(917.0, "o"), (900.0, "s"), (950.0, "^")]:
        props_d = BreakupProperties(ice_density_kg_m3=density, thickness_m=0.10)
        sweep = []
        for u_rms in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            res = fragment([20.0, 12.0, 5.0, 1.0, 0.4], u_rms, props_d,
                           generator=torch.Generator().manual_seed(11))
            sweep.append(abs(res["mass_error_kg"]) / res["initial_mass_kg"])
        axes[2].semilogy([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], np.maximum(sweep, 1e-18), marker + "-",
                         label=f"rho_i={density:.0f} kg/m3")
    axes[2].axhline(1e-12, color="k", lw=0.8, ls=":", label="round-off floor 1e-12")
    axes[2].set_xlabel("race rms velocity u' [m/s]")
    axes[2].set_ylabel("|mass error| / initial mass")
    axes[2].set_title("(c) cascade mass conservation")
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = bending_anchor() + critical_length_anchor() + weibull_cdf_anchor()
    estimator_rows, details = estimator_anchor()
    ks_rows, bias_rows = ks_anchor()
    mass_rows = mass_anchor()
    rows += estimator_rows + ks_rows + mass_rows

    by_anchor: dict[str, dict[str, float]] = {}
    for row in rows:
        entry = by_anchor.setdefault(row["anchor"], {"rows": 0, "worst_relative_error": 0.0})
        entry["rows"] += 1
        entry["worst_relative_error"] = max(entry["worst_relative_error"], float(row["relative_error"]))

    payload = {
        "protocol": "tensordem-fragmentation-statistics-verification-v1",
        "reference_weibull": REFERENCE_WEIBULL,
        "summary_by_anchor": by_anchor,
        "rows": rows,
        "details": details,
        "ks_bias": bias_rows,
    }
    (OUT / "verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_csv(OUT / "error_table.csv", rows)
    write_csv(OUT / "ks_bias.csv", bias_rows)
    made_figure = figure(OUT / "figures" / "fragmentation_statistics_verification.png", details, bias_rows)

    print(f"{'anchor':32s} {'rows':>4s} {'worst rel.err':>14s}")
    for anchor, entry in sorted(by_anchor.items()):
        print(f"{anchor:32s} {entry['rows']:4d} {entry['worst_relative_error']:14.3e}")
    print(f"figure: {made_figure}; outputs in {OUT}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())