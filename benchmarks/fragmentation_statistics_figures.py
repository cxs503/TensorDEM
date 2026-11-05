"""Figures and machine-generated tables for the fragmentation-statistics audit.

Reads ``artifacts/fragmentation_statistics/validation_report.json`` (produced by
``benchmarks/fragmentation_statistics.py``) and writes

* ``artifacts/fragmentation_statistics/fragmentation_statistics.png`` (4 panels),
* ``docs/fragmentation_statistics_tables.md`` (every table quoted in the doc,
  generated programmatically so no number is transcribed by hand).

The Monte-Carlo histograms are re-sampled here rather than stored in the JSON;
the generator seed is the same one the harness uses, so the panels agree with the
reported means.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORT = ROOT / "artifacts" / "fragmentation_statistics" / "validation_report.json"
PNG = ROOT / "artifacts" / "fragmentation_statistics" / "fragmentation_statistics.png"
TABLES = ROOT / "docs" / "fragmentation_statistics_tables.md"


def _ks_arrays(n: int, trials: int, rng: np.random.Generator, shape: float, scale: float):
    out_exact, out_legacy = [], []
    for _ in range(trials):
        x = np.sort(scale * np.power(-np.log1p(-rng.random(n)), 1.0 / shape))
        reference = 1.0 - np.exp(-np.power(x / scale, shape))
        ranks = np.arange(1, n + 1)
        out_exact.append(max((ranks / n - reference).max(), (reference - (ranks - 1) / n).max()))
        out_legacy.append(np.abs(ranks / (n + 1.0) - reference).max())
    return np.asarray(out_exact), np.asarray(out_legacy)


def make_figure(report: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.2))

    # panel 1: what actually controls L_crit -- the support assumption
    axis = axes[0][0]
    rows = report["support_conditions"]["rows"]
    labels = [row["support"].split(" (")[0] for row in rows]
    values = [row["critical_length_m"] for row in rows]
    colours = ["#c26b4f", "#3d6fa8", "#5f9b62"]
    bars = axis.barh(labels, values, color=colours)
    for bar, row in zip(bars, rows):
        axis.text(
            bar.get_width() + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"coeff {row['stress_coefficient']:.2f} → {row['critical_length_m']:.2f} m",
            va="center",
            fontsize=9,
        )
    axis.axvline(report["critical_length"]["critical_length_m"], ls=":", color="k", lw=1)
    axis.set_xlim(0, max(values) * 1.55)
    axis.set_xlabel("critical floe length $L_{crit}$ [m]")
    axis.set_title(
        "Beam coefficient verified to machine precision;\n"
        f"end condition alone spans a factor {report['support_conditions']['spread_factor']:.2f} in $L_{{crit}}$",
        fontsize=10,
    )
    axis.grid(axis="x", alpha=0.3)

    # panel 2: KS statistic -- exact vs the pre-fix plotting-position proxy
    axis = axes[0][1]
    rng = np.random.default_rng(20261103)
    for n, colour in ((20, "#3d6fa8"), (200, "#c26b4f")):
        exact, legacy = _ks_arrays(n, 3000, rng, 1.8, 1.0)
        axis.hist(exact, bins=45, alpha=0.55, color=colour, density=True, label=f"exact, n={n}")
        axis.hist(legacy, bins=45, alpha=0.35, color="grey", density=True, label=f"proxy, n={n}")
        axis.axvline(1.358 / math.sqrt(n), ls="--", color=colour, lw=1.2)
    axis.set_xlabel("KS statistic $D$")
    axis.set_ylabel("density")
    axis.set_title("Exact statistic (colour) vs old plotting-position proxy (grey);\ndashed = 5 % asymptotic critical value", fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    # panel 3: gate power gained by using the exact statistic
    axis = axes[1][0]
    ks_rows = report["ks_calibration"]["rows"]
    ns = [row["n"] for row in ks_rows]
    axis.plot(ns, [row["power_exact"] for row in ks_rows], "o-", color="#3d6fa8", label="exact (fixed)")
    axis.plot(ns, [row["power_legacy"] for row in ks_rows], "s--", color="#c26b4f", label="proxy (pre-fix)")
    axis.set_xscale("log")
    axis.set_xlabel("sample size n")
    axis.set_ylabel("power at nominal 5 % (Weibull shape 1.8 → 2.4)")
    axis.set_title("Where the fix buys power; false-positive rate stays nominal", fontsize=10)
    for row in ks_rows:
        axis.annotate(
            f"{row['fpr_exact_asymptotic']*100:.1f}%",
            (row["n"], row["power_exact"]),
            textcoords="offset points",
            xytext=(0, -12),
            fontsize=7,
            ha="center",
        )
    axis.legend(fontsize=9)
    axis.grid(alpha=0.3)

    # panel 4: the density bug -- ledger error before and after the fix
    axis = axes[1][1]
    ledger_rows = report["mass_ledger"]["rows"]
    densities = [row["ice_density_kg_m3"] for row in ledger_rows]
    legacy = [row["legacy_hardcode_rel_error"] for row in ledger_rows]
    fixed = [max(row["initial_mass_rel_error"], row["conservation_rel_error"], 1e-17) for row in ledger_rows]
    axis.semilogy(densities, [v for v in legacy], "s--", color="#c26b4f", label=r"before: $\rho=917$ hard-coded")
    axis.semilogy(densities, fixed, "o-", color="#3d6fa8", label="after: uses props.ice_density_kg_m3")
    axis.axhspan(1e-17, 1e-12, color="#3d6fa8", alpha=0.08)
    axis.set_xlabel(r"ice density $\rho_i$ [kg/m$^3$]")
    axis.set_ylabel("relative mass-ledger error")
    axis.set_title("Cascade mass ledger: 1.9 % drift at 900 kg/m$^3$ before the fix", fontsize=10)
    axis.legend(fontsize=8, loc="center right")
    axis.grid(alpha=0.3, which="both")

    figure.suptitle(
        "TensorDEM threefield fragmentation: statistics-layer verification "
        f"(report {report.get('git_revision', 'unknown')})",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG, dpi=140)
    plt.close(figure)


def write_tables(report: dict) -> None:
    lines: list[str] = []
    add = lines.append

    add("# Fragmentation statistics — machine-generated tables\n")
    add(
        "Generated by `benchmarks/fragmentation_statistics_figures.py` from "
        "`artifacts/fragmentation_statistics/validation_report.json` "
        f"(schema `{report['schema']}`, git `{report.get('git_revision', 'unknown')}`). "
        "Do not edit by hand.\n"
    )

    add("## 1. Bending-stress coefficient against the beam ODE\n")
    beam = report["beam_reference"]
    add("| cells | $M_{max}/(qL^2)$ | rel. error |")
    add("|---|---|---|")
    for row in beam["fd_rows"]:
        add(f"| {row['cells']} | {row['moment_ratio']:.12f} | {row['moment_ratio_rel_error']:.3e} |")
    add("")
    add("| point loads | $M_{max}/(qL^2)$ | rel. error |")
    add("|---|---|---|")
    for row in beam["lumped_rows"]:
        add(f"| {row['loads']} | {row['moment_ratio']:.12f} | {row['moment_ratio_rel_error']:.3e} |")
    add("")
    add(
        f"Module stress coefficient (recovered from `bending_stress_pa`) = "
        f"{beam['module_stress_coefficient']:.12f}, relative error "
        f"{beam['module_coefficient_rel_error']:.3e} against 3/4. "
        "Finite-difference route: machine precision limited by conditioning "
        f"({beam['fd_machine_precision_rel_error']:.3e} at the finest mesh); "
        f"sampled statics: exact on {beam['lumped_exact_meshes']} of {len(beam['lumped_rows'])} meshes "
        f"(worst {beam['lumped_worst_rel_error']:.3e}. No grid-convergence index is claimed: "
        "the exact solution is a quartic, so a consistent scheme reproduces it exactly.\n"
    )

    add("## 2. Critical floe length\n")
    crit = report["critical_length"]
    add("| quantity | value |")
    add("|---|---|")
    add(f"| reference case | $u'={crit['case']['rms_velocity_m_s']}$ m/s, $h={crit['case']['thickness_m']}$ m, "
        f"$\\sigma_f={crit['case']['flexural_strength_pa']:.3g}$ Pa |")
    add(f"| `critical_floe_length` | {crit['critical_length_m']:.9f} m |")
    add(f"| bisection on `bending_stress_pa` | {crit['bisection_length_m']:.9f} m (rel. error {crit['bisection_rel_error']:.3e}) |")
    add(f"| calm race ($u'=0$) | `inf` |")
    add(f"| measured exponent, $u'$ | {crit['velocity_exponent']:.6f} (closed form −1) |")
    add(f"| measured exponent, $h$ | {crit['thickness_exponent']:.6f} (closed form 0.5) |")
    add(f"| measured exponent, $\\sigma_f$ | {crit['strength_exponent']:.6f} (closed form 0.5) |")
    add("")

    add("## 3. Support-condition sensitivity of $L_{crit}$\n")
    sup = report["support_conditions"]
    add("| support | $M_{max}/(qL^2)$ | stress coefficient | $L_{crit}$ [m] | ratio to cantilever |")
    add("|---|---|---|---|---|")
    for row in sup["rows"]:
        add(f"| {row['support']} | {row['moment_ratio']:.6f} | {row['stress_coefficient']:.3f} | "
            f"{row['critical_length_m']:.4f} | {row['ratio_to_cantilever']:.4f} |")
    add("")
    add(f"Module value matches the simply supported row: {sup['module_matches_simply_supported']}. "
        f"Cantilever-to-clamped spread in $L_{{crit}}$ = {sup['spread_factor']:.4f} "
        f"(drag load {sup['drag_per_metre_n_per_m']:.3f} N/m).\n")

    add("## 4. Cascade mass ledger vs ice density\n")
    ledger = report["mass_ledger"]
    add("| $\\rho_i$ | expected mass [kg] | reported initial [kg] | rel. error | conservation rel. error | error of the 917 hard-code |")
    add("|---|---|---|---|---|---|")
    for row in ledger["rows"]:
        add(f"| {row['ice_density_kg_m3']:.0f} | {row['expected_mass_kg']:.6f} | "
            f"{row['reported_initial_mass_kg']:.6f} | {row['initial_mass_rel_error']:.3e} | "
            f"{row['conservation_rel_error']:.3e} | {row['legacy_hardcode_rel_error']:.4%} |")
    add("")
    add(f"Calm (no-breakup) branch: worst relative error {ledger['calm_case_mass_rel_error']:.3e}. "
        f"Verdict: {'PASS' if ledger['verdict_pass'] else 'FAIL'}.\n")

    add("## 5. Coupled pass ledger (`coupling.fragmentation_stage`)\n")
    coupling = report["coupling_ledger"]
    add("| $\\rho_i$ | inventory [kg] | initial [kg] | final [kg] | splits | ledger rel. error | old drift |")
    add("|---|---|---|---|---|---|---|")
    for row in coupling["rows"]:
        add(f"| {row['ice_density_kg_m3']:.0f} | {row['inventory_mass_kg']:.1f} | "
            f"{row['reported_initial_mass_kg']:.1f} | {row['reported_final_mass_kg']:.6f} | "
            f"{row['split_events']} | {row['ledger_rel_error']:.3e} | {row['legacy_hardcode_rel_error']:.4%} |")
    add("")

    add("## 6. Kolmogorov-Smirnov statistic: exact vs the pre-fix proxy\n")
    ks = report["ks_calibration"]
    add("| n | mean $D$ exact | mean $D$ proxy | proxy bias | mean $D$ scipy | FPR exact (5 %) | FPR proxy | FPR exact, finite-$n$ crit | power exact | power proxy |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for row in ks["rows"]:
        add(f"| {row['n']} | {row['mean_d_exact']:.4f} | {row['mean_d_legacy']:.4f} | "
            f"{row['legacy_bias_relative']:+.1%} | {row['mean_d_scipy']:.4f} | "
            f"{row['fpr_exact_asymptotic']:.3f} | {row['fpr_legacy_asymptotic']:.3f} | "
            f"{row['fpr_exact_finite_n']:.3f} | {row['power_exact']:.2f} | {row['power_legacy']:.2f} |")
    add("")
    add(f"Reference law {ks['truth_weibull']}, alternative {ks['alternative_weibull']}, "
        f"{ks['rows'][0]['trials']} trials per cell. Wilson 95 % intervals of the exact FPR are in the JSON "
        "(`fpr_exact_asymptotic_wilson95`).\n")

    add("## 7. Weibull estimator bias against the scipy MLE\n")
    wb = report["weibull_bias"]
    add("| n | double-log shape bias | double-log scale bias | CDF-LS shape bias | CDF-LS scale bias | MLE shape bias | MLE scale bias |")
    add("|---|---|---|---|---|---|---|")
    for row in wb["rows"]:
        add(f"| {row['n']} | {row['double_log_shape_bias']:+.1%} | {row['double_log_scale_bias']:+.1%} | "
            f"{row['cdf_shape_bias']:+.1%} | {row['cdf_scale_bias']:+.1%} | "
            f"{row['mle_shape_bias']:+.1%} | {row['mle_scale_bias']:+.1%} |")
    add("")
    add(f"Truth shape {wb['rows'][0]['truth_shape']}, scale {wb['rows'][0]['truth_scale']}; "
        f"{wb['rows'][0]['trials']} trials (CDF fit: {wb['rows'][0]['cdf_fit_trials']}, it is the expensive estimator). "
        "`fit_weibull` is the estimator the coupled report quotes.\n")

    add("## 8. Cascade post-conditions\n")
    cascade = report["cascade"]
    for row in cascade["rows"]:
        add(f"- 60 floes, $u'=0.75$ m/s: {row['floes_out']} fragments, {row['split_events']} split events, "
            f"largest survivor {row['largest_survivor_m']:.3f} m vs $L_{{crit}}$ {row['critical_length_m']:.3f} m, "
            f"mass rel. error {row['mass_rel_error']:.3e}, deterministic replay {row['deterministic']}.")
    add("")
    add("| $u'$ [m/s] | $L_{crit}$ [m] | split events | fragments |")
    add("|---|---|---|---|")
    for row in cascade["velocity_sweep"]:
        add(f"| {row['rms_velocity_m_s']:.2f} | {row['critical_length_m']:.3f} | {row['split_events']} | {row['floes_out']} |")
    add("")

    add("## 9. Is the reference Weibull law commensurate with a field census?\n")
    ref = report["reference_law_check"]
    fitted = ref["fitted_weibull"]
    agree = ref["agreement_with_reference_law"]
    add("| quantity | value |")
    add("|---|---|")
    add(f"| assumed field population | {ref['population']['floes_in']} floes, "
        f"{ref['population']['range_m'][0]:.2f}–{ref['population']['range_m'][1]:.2f} m |")
    add(f"| split events / fragments | {ref['split_events']} / {ref['fragment_count']} |")
    add(f"| fitted shape / scale | {fitted['shape']:.4f} / {fitted['scale']:.4f} m (R² {fitted['r_squared']:.4f}) |")
    add(f"| reference law | shape {ref['reference_law']['shape']}, scale {ref['reference_law']['scale_m']} m |")
    add(f"| agreement with reference | R² {agree['r_squared']:.4f}, max CDF deviation {agree['max_cdf_deviation']:.4f} |")
    add(f"| reference/census scale ratio | {ref['scale_ratio_reference_to_census']:.4f} |")
    add("")
    TABLES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    make_figure(report)
    write_tables(report)
    print(f"figure: {PNG}")
    print(f"tables: {TABLES}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())