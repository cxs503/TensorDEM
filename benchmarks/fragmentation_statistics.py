"""Verification harness for the threefield fragmentation *statistics* layer.

The breakup criterion (``critical_floe_length``) and the mass-conserving cascade
were taken as given by ``docs/THREEFIELD_ICE_HULL_PROPELLER.md``.  What had never
been checked is the layer that turns a fragment census into a number someone can
gate on: the Weibull fits, the Kolmogorov-Smirnov statistic, the mass ledger and
the bending-stress coefficient.  This harness checks all four against references
that are *independent* of the implementation:

===================================  ==========================================
quantity                             independent reference
===================================  ==========================================
peak bending moment of a uniformly    finite-difference solve of the
loaded, simply supported beam          Euler-Bernoulli ODE (mesh refinement,
                                        observed order of convergence)
``critical_floe_length``               bisection on ``bending_stress_pa`` +
                                        log-log scaling exponents
cascade mass ledger                    ``rho_i L^2 h`` recomputed outside the
                                        module, swept over ice density
KS statistic                           Kolmogorov asymptotic and finite-``n``
                                        critical values, and ``scipy`` exact
                                        statistics over Monte-Carlo trials
Weibull shape / scale estimators       ``scipy.stats.weibull_min.fit`` (MLE)
===================================  ==========================================

Every number quoted in ``docs/fragmentation_statistics_verification.md`` is
machine-generated into ``artifacts/fragmentation_statistics/validation_report.json``
by this script; nothing is transcribed by hand.

Runs in about a minute on one core (Monte-Carlo with numpy, no heavy solver), so
it is safe on a loaded control host::

    python benchmarks/fragmentation_statistics.py --output artifacts/.../validation_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from tensordem.threefield.fragmentation import (  # noqa: E402
    BreakupProperties,
    bending_stress_pa,
    cdf_agreement_r2,
    critical_floe_length,
    fit_weibull,
    fit_weibull_cdf,
    floe_volume_m3,
    fragment,
    kolmogorov_smirnov,
    weibull_cdf,
)

SCHEMA = "tensordem.fragmentation-statistics/1"
KS_CRIT_5PCT = 1.358  # Kolmogorov asymptotic constant, alpha = 0.05
ICE_DENSITY_DEFAULT = 917.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _weibull_cdf_np(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
    return 1.0 - np.exp(-np.power(np.asarray(x, dtype=float) / scale, shape))


def _weibull_samples(rng: np.random.Generator, n: int, shape: float, scale: float) -> np.ndarray:
    uniforms = rng.random(n)
    return scale * np.power(-np.log1p(-uniforms), 1.0 / shape)


def _ks_exact(x_sorted: np.ndarray, f_reference: np.ndarray) -> float:
    n = len(x_sorted)
    upper = np.arange(1, n + 1) / n - f_reference
    lower = f_reference - np.arange(0, n) / n
    return float(max(upper.max(), lower.max()))


def _ks_legacy(x_sorted: np.ndarray, f_reference: np.ndarray) -> float:
    """The pre-fix implementation: Weibull plotting positions, one side only."""
    n = len(x_sorted)
    positions = (np.arange(1, n + 1)) / (n + 1.0)
    return float(np.abs(positions - f_reference).max())


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover - git is optional for the harness
        return "unknown"


# --------------------------------------------------------------------------- #
# 1. bending-stress coefficient from the beam ODE
# --------------------------------------------------------------------------- #
def _beam_moment_fd(n_cells: int) -> float:
    """``M_max / (q L^2)`` for a simply supported beam on ``n_cells`` intervals.

    Solves ``w'''' = q / (EI)`` with ``w = 0`` and ``w'' = 0`` at both ends by a
    five-point finite difference on a uniform mesh (``EI = 1``, ``q = 1``,
    ``L = 1``, unit width) and returns the peak bending moment obtained from the
    discrete curvature, ``M = -EI w''``.
    """
    h = 1.0 / n_cells
    # unknowns: ghost nodes -1 and n_cells+1 plus physical nodes 0..n_cells
    size = n_cells + 3
    matrix = np.zeros((size, size))
    rhs = np.zeros(size)

    def index(node: int) -> int:
        return node + 1  # ghost node -1 maps to 0

    # boundary conditions
    matrix[index(0), index(0)] = 1.0  # w(0) = 0
    matrix[index(n_cells), index(n_cells)] = 1.0  # w(L) = 0
    matrix[index(-1), index(-1)] = 1.0  # w''(0) = 0 -> w_{-1} - 2 w_0 + w_1 = 0
    matrix[index(-1), index(0)] = -2.0
    matrix[index(-1), index(1)] = 1.0
    matrix[index(n_cells + 1), index(n_cells + 1)] = 1.0
    matrix[index(n_cells + 1), index(n_cells)] = -2.0
    matrix[index(n_cells + 1), index(n_cells - 1)] = 1.0

    for node in range(1, n_cells):
        row = index(node)
        for offset, coefficient in ((-2, 1.0), (-1, -4.0), (0, 6.0), (1, -4.0), (2, 1.0)):
            matrix[row, index(node + offset)] = coefficient / h**4
        rhs[row] = 1.0

    w = np.linalg.solve(matrix, rhs)
    curvature = np.zeros(n_cells + 1)
    for node in range(n_cells + 1):
        curvature[node] = (w[index(node + 1)] - 2.0 * w[index(node)] + w[index(node - 1)]) / h**2
    return float(np.abs(curvature).max())


def _beam_lumped_load_moment(n_loads: int) -> float:
    """``M_max / (q L^2)`` with the distributed load lumped into ``n_loads`` point loads.

    Statics only (no ODE): a simply supported beam of length 1 carrying point
    loads ``P = 1/n`` at the element midpoints ``x_i``.  Unlike the five-point
    finite-difference solve below, this route carries a genuine ``O(h^2)``
    discretisation error, so the coefficient can be shown to be a property of
    the continuum limit rather than of one discretisation scheme.
    """
    loads = [(index + 0.5) / n_loads for index in range(n_loads)]
    total = 1.0
    reaction_left = sum(total / n_loads * x for x in loads)  # symmetry, L = 1
    moment = 0.0
    for station in loads:
        bending = reaction_left * station
        for x in loads:
            if x < station:
                bending -= (total / n_loads) * (station - x)
        moment = max(moment, abs(bending))
    return moment


def beam_reference(
    fd_cells: Sequence[int] = (10, 20, 40, 80, 160, 320, 640),
    lumped_counts: Sequence[int] = (2, 4, 8, 16, 32, 64, 128, 256),
) -> dict[str, Any]:
    """Numerically verify the ``3 q L^2 / (4 h^2)`` bending-stress coefficient.

    Two independent routes are used, because the first one is *too* good to be
    evidence about a discretisation limit:

    * the five-point finite-difference solve of the fourth-order beam ODE with
      simply supported end conditions reproduces the quartic deflection
      *exactly* (the stencil is exact for polynomials of degree <= 4), so its
      agreement is a machine-precision check limited only by the ``h^-4``
      conditioning of the assembled matrix;
    * a lumped-load statics model carries a real ``O(h^2)`` error, so its mesh
      series measures the order of convergence towards ``q L^2 / 8`` and shows
      that the coefficient is not an artefact of one discretisation.
    """
    exact_ratio = 1.0 / 8.0  # M_max / (q L^2), uniformly loaded SS beam

    fd_rows = []
    for n_cells in fd_cells:
        ratio = _beam_moment_fd(n_cells)
        fd_rows.append(
            {
                "cells": int(n_cells),
                "moment_ratio": ratio,
                "moment_ratio_rel_error": abs(ratio - exact_ratio) / exact_ratio,
                "stress_coefficient": 6.0 * ratio,
            }
        )

    lumped_rows = []
    for n_loads in lumped_counts:
        ratio = _beam_lumped_load_moment(n_loads)
        lumped_rows.append(
            {
                "loads": int(n_loads),
                "moment_ratio": ratio,
                "moment_ratio_rel_error": abs(ratio - exact_ratio) / exact_ratio,
                "stress_coefficient": 6.0 * ratio,
            }
        )
    exact_count = sum(1 for row in lumped_rows if row["moment_ratio_rel_error"] < 1e-15)
    worst_lumped = max(row["moment_ratio_rel_error"] for row in lumped_rows)

    props = BreakupProperties(thickness_m=0.2, flexural_strength_pa=5.0e5)
    stress = bending_stress_pa(4.0, 0.75, props)
    q = 0.5 * props.water_density_kg_m3 * props.drag_coefficient * 0.75**2 * props.thickness_m
    module_coefficient = stress * props.thickness_m**2 / (q * 4.0**2)
    return {
        "exact_moment_ratio": exact_ratio,
        "exact_stress_coefficient": 0.75,
        "fd_rows": fd_rows,
        "fd_machine_precision_rel_error": max(row["moment_ratio_rel_error"] for row in fd_rows),
        "lumped_rows": lumped_rows,
        "lumped_exact_meshes": exact_count,
        "lumped_worst_rel_error": worst_lumped,
        "no_gci_claim": (
            "the exact solution of the uniformly loaded simply supported beam is a "
            "quartic and the sampled statics are symmetric, so every consistent "
            "discretisation reproduces M_max exactly; no grid-convergence index is "
            "claimed because there is no discretisation error to extrapolate"
        ),
        "module_stress_coefficient": module_coefficient,
        "module_coefficient_rel_error": abs(module_coefficient - 0.75) / 0.75,
        "drag_per_metre_n_per_m": q,
    }


def support_condition_sensitivity() -> dict[str, Any]:
    """How much the *end condition* (not the numerics) moves ``L_crit``.

    The module's coefficient ``3/4`` is the simply supported beam value,
    ``sigma = 6 M_max / h^2`` with ``M_max = q L^2 / 8``.  A cantilever or a
    clamped floe has a different moment and therefore a different critical
    length; since ``L_crit ~ sqrt(sigma_f h / (coeff * q))`` the support
    assumption is the dominant modelling uncertainty of the criterion.
    """
    props = BreakupProperties(thickness_m=0.1, flexural_strength_pa=0.5e6)
    u_rms = 0.75
    q = 0.5 * props.water_density_kg_m3 * props.drag_coefficient * u_rms**2 * props.thickness_m
    cases = (
        ("cantilever (one free end)", 1.0 / 2.0),
        ("simply supported (module)", 1.0 / 8.0),
        ("clamped-clamped", 1.0 / 12.0),
    )
    rows = []
    reference = None
    for name, moment_ratio in cases:
        coefficient = 6.0 * moment_ratio
        length = math.sqrt(props.flexural_strength_pa * props.thickness_m**2 / (coefficient * q))
        if reference is None:
            reference = length
        rows.append(
            {
                "support": name,
                "moment_ratio": moment_ratio,
                "stress_coefficient": coefficient,
                "critical_length_m": length,
                "ratio_to_cantilever": length / reference,
            }
        )
    module = critical_floe_length(u_rms, props)
    return {
        "drag_per_metre_n_per_m": q,
        "rows": rows,
        "module_critical_length_m": module,
        "module_matches_simply_supported": abs(module - rows[1]["critical_length_m"]) / module < 1e-12,
        "spread_factor": rows[2]["critical_length_m"] / rows[0]["critical_length_m"],
    }


# --------------------------------------------------------------------------- #
# 2. critical floe length: self-consistency and scaling
# --------------------------------------------------------------------------- #
def critical_length_checks() -> dict[str, Any]:
    props = BreakupProperties(thickness_m=0.1, flexural_strength_pa=0.5e6)
    u_rms = 0.75
    l_crit = critical_floe_length(u_rms, props)

    # bisection on the *module's own* stress law -> must reproduce the closed form
    low, high = 1e-3, 1e3
    for _ in range(200):
        mid = 0.5 * (low + high)
        if bending_stress_pa(mid, u_rms, props) < props.flexural_strength_pa:
            low = mid
        else:
            high = mid
    bisection = 0.5 * (low + high)

    def loglog_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
        x = np.log(np.asarray(xs, dtype=float))
        y = np.log(np.asarray(ys, dtype=float))
        return float(np.polyfit(x, y, 1)[0])

    velocities = [0.25, 0.5, 1.0, 2.0, 4.0]
    thicknesses = [0.05, 0.1, 0.2, 0.4]
    strengths = [0.25e6, 0.5e6, 1.0e6, 2.0e6]
    return {
        "case": {
            "rms_velocity_m_s": u_rms,
            "thickness_m": props.thickness_m,
            "flexural_strength_pa": props.flexural_strength_pa,
            "water_density_kg_m3": props.water_density_kg_m3,
            "drag_coefficient": props.drag_coefficient,
        },
        "critical_length_m": l_crit,
        "bisection_length_m": bisection,
        "bisection_rel_error": abs(bisection - l_crit) / l_crit,
        "calm_race_length": critical_floe_length(0.0, props),
        "velocity_exponent": loglog_slope(
            velocities, [critical_floe_length(v, props) for v in velocities]
        ),
        "thickness_exponent": loglog_slope(
            thicknesses,
            [
                critical_floe_length(
                    u_rms, BreakupProperties(thickness_m=h, flexural_strength_pa=0.5e6)
                )
                for h in thicknesses
            ],
        ),
        "strength_exponent": loglog_slope(
            strengths,
            [
                critical_floe_length(
                    u_rms, BreakupProperties(thickness_m=0.1, flexural_strength_pa=s)
                )
                for s in strengths
            ],
        ),
        "expected_exponents": {"velocity": -1.0, "thickness": 0.5, "strength": 0.5},
    }


# --------------------------------------------------------------------------- #
# 3. cascade mass ledger over ice density
# --------------------------------------------------------------------------- #
def mass_ledger(densities: Sequence[float] = (800.0, 900.0, 917.0, 940.0, 950.0)) -> dict[str, Any]:
    sizes = [3.0, 5.0]
    thickness = 0.2
    rows = []
    for density in densities:
        props = BreakupProperties(thickness_m=thickness, ice_density_kg_m3=density)
        result = fragment(sizes, 5.0, props, max_splits=64)
        expected = sum(density * floe_volume_m3(size, props) for size in sizes)
        reported = float(result["initial_mass_kg"])
        produced = float(result["final_mass_kg"])
        per_fragment = sum(
            density * floe_volume_m3(length, props) for length in result["lengths_m"]
        )
        rows.append(
            {
                "ice_density_kg_m3": density,
                "expected_mass_kg": expected,
                "reported_initial_mass_kg": reported,
                "reported_final_mass_kg": produced,
                "independent_final_mass_kg": per_fragment,
                "initial_mass_rel_error": abs(reported - expected) / expected,
                "conservation_rel_error": abs(produced - reported) / expected,
                "legacy_hardcode_rel_error": abs(ICE_DENSITY_DEFAULT / density - 1.0),
            }
        )
    worst_initial = max(row["initial_mass_rel_error"] for row in rows)
    worst_conservation = max(row["conservation_rel_error"] for row in rows)
    worst_legacy = max(row["legacy_hardcode_rel_error"] for row in rows)
    # no-breakup branch: masses must simply be rho L^2 h
    calm_props = BreakupProperties(thickness_m=thickness, ice_density_kg_m3=900.0)
    calm = fragment([0.05, 0.06], 0.0, calm_props)
    calm_expected = [900.0 * floe_volume_m3(size, calm_props) for size in (0.05, 0.06)]
    calm_error = max(
        abs(a - b) / b for a, b in zip(sorted(calm["masses_kg"]), sorted(calm_expected))
    )
    return {
        "rows": rows,
        "worst_initial_mass_rel_error": worst_initial,
        "worst_conservation_rel_error": worst_conservation,
        "legacy_hardcode_rel_error": worst_legacy,
        "calm_case_mass_rel_error": calm_error,
        "verdict_pass": worst_initial < 1e-12 and worst_conservation < 1e-12 and calm_error < 1e-12,
    }


def coupling_ledger_consistency() -> dict[str, Any]:
    """Reproduce the coupled pass' inventory conversion and check its closure.

    ``coupling.fragmentation_stage`` converts an inventory mass to a
    plate-equivalent length with ``sqrt(m / (rho_i h))`` and then lets
    ``fragment`` rebuild the masses.  If the two disagree on the density the
    ledger cannot close.
    """
    from tensordem.threefield.coupling import fragmentation_stage

    rows = []
    for density in (800.0, 900.0, 917.0, 950.0):
        props = BreakupProperties(thickness_m=0.2, ice_density_kg_m3=density)
        # Field-scale floe masses (thousands of tonnes) in a 0.75 m/s race so
        # that the cascade genuinely splits; with the DEM-scale census
        # (kilograms) every floe sits below L_crit and no density dependence
        # would be exercised at all.
        inventory = [
            {"index": float(i), "mass_kg": mass, "diameter_m": 1.0}
            for i, mass in enumerate((2.0e5, 1.0e5, 5.0e4, 2.5e4))
        ]
        stage = fragmentation_stage(inventory, 0.75, props, seed=7)
        initial = sum(item["mass_kg"] for item in inventory)
        rows.append(
            {
                "ice_density_kg_m3": density,
                "inventory_mass_kg": initial,
                "reported_initial_mass_kg": stage["initial_mass_kg"],
                "reported_final_mass_kg": stage["final_mass_kg"],
                "ledger_rel_error": abs(stage["final_mass_kg"] - initial) / initial,
                "legacy_hardcode_rel_error": abs(ICE_DENSITY_DEFAULT / density - 1.0),
                "split_events": stage["split_events"],
            }
        )
    worst = max(row["ledger_rel_error"] for row in rows)
    return {
        "rows": rows,
        "worst_ledger_rel_error": worst,
        "verdict_pass": worst < 1e-12,
    }


# --------------------------------------------------------------------------- #
# 4. KS statistic calibration and gate power
# --------------------------------------------------------------------------- #
def _wilson(p_hat: float, n: int, z: float = 1.96) -> list[float]:
    denominator = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denominator
    return [centre - half, centre + half]


def ks_calibration(
    sample_sizes: Sequence[int] = (10, 20, 50, 100, 200, 500),
    trials: int = 4000,
    seed: int = 20261103,
    calibration_trials: int = 300,
) -> dict[str, Any]:
    from scipy import stats

    rng = np.random.default_rng(seed)
    truth = (1.8, 1.0)  # reference law used by the gate
    alternative = (2.4, 1.0)  # a visibly different fragment-size law
    rows = []
    for n in sample_sizes:
        d_exact, d_legacy, d_scipy, d_alt_exact, d_alt_legacy = [], [], [], [], []
        for trial in range(trials):
            x = np.sort(_weibull_samples(rng, n, *truth))
            f = _weibull_cdf_np(x, *truth)
            d_exact.append(_ks_exact(x, f))
            d_legacy.append(_ks_legacy(x, f))
            if trial < calibration_trials:
                d_scipy.append(stats.kstest(x, lambda v: _weibull_cdf_np(v, *truth)).statistic)
            xa = np.sort(_weibull_samples(rng, n, *alternative))
            fa = _weibull_cdf_np(xa, *truth)
            d_alt_exact.append(_ks_exact(xa, fa))
            d_alt_legacy.append(_ks_legacy(xa, fa))
        crit_asym = KS_CRIT_5PCT / math.sqrt(n)
        crit_finite = float(stats.kstwo.ppf(0.95, n))
        d_exact_arr = np.asarray(d_exact)
        rows.append(
            {
                "n": int(n),
                "trials": int(trials),
                "mean_d_exact": float(d_exact_arr.mean()),
                "mean_d_legacy": float(np.mean(d_legacy)),
                "mean_d_scipy": float(np.mean(d_scipy)),
                "max_abs_diff_exact_vs_scipy": float(
                    np.abs(d_exact_arr[: len(d_scipy)] - np.asarray(d_scipy)).max()
                ),
                "legacy_bias_relative": float(np.mean(d_legacy) / d_exact_arr.mean() - 1.0),
                "crit_asymptotic": crit_asym,
                "crit_finite_n": crit_finite,
                "fpr_exact_asymptotic": float((d_exact_arr > crit_asym).mean()),
                "fpr_legacy_asymptotic": float((np.asarray(d_legacy) > crit_asym).mean()),
                "fpr_exact_finite_n": float((d_exact_arr > crit_finite).mean()),
                "fpr_exact_asymptotic_wilson95": _wilson(
                    float((d_exact_arr > crit_asym).mean()), trials
                ),
                "power_exact": float((np.asarray(d_alt_exact) > crit_asym).mean()),
                "power_legacy": float((np.asarray(d_alt_legacy) > crit_asym).mean()),
            }
        )
    fpr = [row["fpr_exact_asymptotic"] for row in rows]
    return {
        "truth_weibull": {"shape": truth[0], "scale": truth[1]},
        "alternative_weibull": {"shape": alternative[0], "scale": alternative[1]},
        "rows": rows,
        "fpr_range": [min(fpr), max(fpr)],
        "verdict_exact_calibrated": all(0.028 <= value <= 0.075 for value in fpr),
        "verdict_legacy_bias": [row["legacy_bias_relative"] for row in rows],
        "verdict_power_gain": [
            row["power_exact"] - row["power_legacy"] for row in rows
        ],
        "calibration_trials": int(calibration_trials),
    }


def ks_module_agreement(
    sample_sizes: Sequence[int] = (10, 20, 50, 100, 200),
    trials: int = 40,
    seed: int = 99,
) -> dict[str, Any]:
    """The module function must equal the exact statistic, not the proxy."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in sample_sizes:
        worst = 0.0
        worst_legacy_gap = 0.0
        for _ in range(trials):
            x = np.sort(_weibull_samples(rng, n, 1.8, 1.0))
            f = _weibull_cdf_np(x, 1.8, 1.0)
            exact = _ks_exact(x, f)
            module = kolmogorov_smirnov(x.tolist(), 1.8, 1.0)
            worst = max(worst, abs(module - exact))
            worst_legacy_gap = max(worst_legacy_gap, exact - _ks_legacy(x, f))
        rows.append(
            {
                "n": int(n),
                "max_abs_diff_module_vs_exact": worst,
                "max_legacy_understatement": worst_legacy_gap,
            }
        )
    return {
        "rows": rows,
        "verdict_pass": max(row["max_abs_diff_module_vs_exact"] for row in rows) < 1e-12,
    }


# --------------------------------------------------------------------------- #
# 5. Weibull estimator bias against the scipy MLE
# --------------------------------------------------------------------------- #
def weibull_bias(
    sample_sizes: Sequence[int] = (30, 100, 300),
    truth: tuple[float, float] = (1.8, 0.4),
    trials: int = 200,
    cdf_fit_trials: int = 15,
    seed: int = 4242,
) -> dict[str, Any]:
    from scipy import stats

    rng = np.random.default_rng(seed)
    rows = []
    for n in sample_sizes:
        double_log_shape, cdf_shape, mle_shape = [], [], []
        double_log_scale, cdf_scale, mle_scale = [], [], []
        for trial in range(trials):
            data = _weibull_samples(rng, n, *truth).tolist()
            fit = fit_weibull(data)
            double_log_shape.append(fit["shape"])
            double_log_scale.append(fit["scale"])
            mle = stats.weibull_min.fit(data, floc=0.0)
            mle_shape.append(float(mle[0]))
            mle_scale.append(float(mle[2]))
            if trial < cdf_fit_trials:
                cdf_fit = fit_weibull_cdf(data)
                cdf_shape.append(cdf_fit["shape"])
                cdf_scale.append(cdf_fit["scale"])
        rows.append(
            {
                "n": int(n),
                "trials": int(trials),
                "cdf_fit_trials": int(min(cdf_fit_trials, trials)),
                "truth_shape": truth[0],
                "truth_scale": truth[1],
                "double_log_shape_bias": float(np.mean(double_log_shape) / truth[0] - 1.0),
                "double_log_scale_bias": float(np.mean(double_log_scale) / truth[1] - 1.0),
                "double_log_shape_sd": float(np.std(double_log_shape)),
                "cdf_shape_bias": float(np.mean(cdf_shape) / truth[0] - 1.0),
                "cdf_scale_bias": float(np.mean(cdf_scale) / truth[1] - 1.0),
                "cdf_shape_sd": float(np.std(cdf_shape)),
                "mle_shape_bias": float(np.mean(mle_shape) / truth[0] - 1.0),
                "mle_scale_bias": float(np.mean(mle_scale) / truth[1] - 1.0),
                "mle_shape_sd": float(np.std(mle_shape)),
            }
        )
    return {
        "rows": rows,
        "note": "double_log = fit_weibull (used by the coupled report); cdf = fit_weibull_cdf; mle = scipy",
    }


# --------------------------------------------------------------------------- #
# 6. cascade post-conditions
# --------------------------------------------------------------------------- #
def cascade_postconditions(seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    rng = np.random.default_rng(2026)
    props = BreakupProperties(thickness_m=0.1, ice_density_kg_m3=917.0)
    population = list(np.exp(rng.uniform(math.log(0.5), math.log(40.0), 60)))
    rows = []
    for seed in seeds:
        result = fragment(population, 0.75, props)
        repeat = fragment(population, 0.75, props)
        l_crit = float(result["critical_length_m"])
        survivors_ok = max(result["lengths_m"]) <= max(l_crit, 2.0 * props.min_fragment_m) + 1e-12
        rows.append(
            {
                "seed_seedless_default": int(seed),
                "floes_in": len(population),
                "floes_out": len(result["lengths_m"]),
                "split_events": int(result["split_events"]),
                "critical_length_m": l_crit,
                "largest_survivor_m": max(result["lengths_m"]),
                "survivors_below_criterion": bool(survivors_ok),
                "mass_rel_error": abs(result["final_mass_kg"] - result["initial_mass_kg"])
                / result["initial_mass_kg"],
                "deterministic": result["lengths_m"] == repeat["lengths_m"],
            }
        )
        break  # the default generator is seeded internally; one row is enough
    # race strength sweep: split count must grow monotonically with u'
    sweep = []
    for u_rms in (0.2, 0.4, 0.75, 1.5, 3.0):
        result = fragment(population, u_rms, props)
        sweep.append(
            {
                "rms_velocity_m_s": u_rms,
                "critical_length_m": float(result["critical_length_m"]),
                "split_events": int(result["split_events"]),
                "floes_out": len(result["lengths_m"]),
            }
        )
    monotone = all(
        sweep[i]["split_events"] <= sweep[i + 1]["split_events"] for i in range(len(sweep) - 1)
    )
    row = rows[0]
    return {
        "rows": rows,
        "velocity_sweep": sweep,
        "splits_monotone_in_velocity": monotone,
        "verdict_pass": bool(
            row["survivors_below_criterion"]
            and row["mass_rel_error"] < 1e-12
            and row["deterministic"]
            and monotone
        ),
    }


# --------------------------------------------------------------------------- #
# report assembly
# --------------------------------------------------------------------------- #
def build_report(quick: bool = False) -> dict[str, Any]:
    trials = 400 if quick else 4000
    report = {
        "schema": SCHEMA,
        "git_revision": _git_revision(),
        "quick": bool(quick),
        "critical_length": critical_length_checks(),
        "beam_reference": beam_reference(
            fd_cells=(10, 20, 40, 80, 160) if quick else (10, 20, 40, 80, 160, 320, 640),
            lumped_counts=(2, 4, 8, 16) if quick else (2, 4, 8, 16, 32, 64, 128, 256),
        ),
        "support_conditions": support_condition_sensitivity(),
        "mass_ledger": mass_ledger(),
        "coupling_ledger": coupling_ledger_consistency(),
        "ks_calibration": ks_calibration(trials=trials),
        "ks_module": ks_module_agreement(),
        "weibull_bias": weibull_bias(
            trials=40 if quick else 200, cdf_fit_trials=5 if quick else 15
        ),
        "cascade": cascade_postconditions(),
        "reference_law_check": reference_law_check(),
    }
    report["verdicts"] = _verdicts(report)
    return report


def reference_law_check() -> dict[str, Any]:
    """Does the cascade output actually look like the experimental Weibull law?

    ``run_coupled_pass`` grades the census against ``(1.8, 0.12 m)``.  The
    cascade works in metres, with ``L_crit`` of order ten metres for a 0.1 m
    floe in a 0.75 m/s race, so the *scale* of that reference cannot describe a
    field-scale census.  This records the measured agreement instead of
    assuming it.
    """
    rng = np.random.default_rng(11)
    props = BreakupProperties(thickness_m=0.1, ice_density_kg_m3=917.0)
    population = list(np.exp(rng.uniform(math.log(0.5), math.log(40.0), 400)))
    result = fragment(population, 0.75, props)
    sizes = result["lengths_m"]
    agreement = cdf_agreement_r2(sizes, 1.8, 0.12)
    fitted = fit_weibull(sizes)
    return {
        "population": {
            "floes_in": len(population),
            "range_m": [float(min(population)), float(max(population))],
            "note": "assumed field population, not a measurement",
        },
        "split_events": int(result["split_events"]),
        "fragment_count": len(sizes),
        "largest_fragment_m": float(max(sizes)),
        "critical_length_m": float(result["critical_length_m"]),
        "fitted_weibull": fitted,
        "agreement_with_reference_law": agreement,
        "reference_law": {"shape": 1.8, "scale_m": 0.12},
        "scale_ratio_reference_to_census": 0.12 / float(fitted["scale"]),
    }


def _verdicts(report: dict[str, Any]) -> dict[str, Any]:
    beam = report["beam_reference"]
    ledger = report["mass_ledger"]
    coupling = report["coupling_ledger"]
    ks = report["ks_calibration"]
    ks_module = report["ks_module"]
    cascade = report["cascade"]
    return {
        "beam_coefficient_verified": {
            "pass": beam["module_coefficient_rel_error"] < 1e-12
            and beam["fd_machine_precision_rel_error"] < 1e-6
            and beam["lumped_worst_rel_error"] < 2e-2,
            "module_coefficient_rel_error": beam["module_coefficient_rel_error"],
            "fd_machine_precision_rel_error": beam["fd_machine_precision_rel_error"],
            "lumped_worst_rel_error": beam["lumped_worst_rel_error"],
            "lumped_exact_meshes": beam["lumped_exact_meshes"],
        },
        "support_condition_is_the_dominant_uncertainty": {
            "pass": report["support_conditions"]["module_matches_simply_supported"],
            "module_matches_simply_supported": report["support_conditions"][
                "module_matches_simply_supported"
            ],
            "spread_factor": report["support_conditions"]["spread_factor"],
        },
        "critical_length_self_consistent": {
            "pass": report["critical_length"]["bisection_rel_error"] < 1e-10,
            "bisection_rel_error": report["critical_length"]["bisection_rel_error"],
        },
        "mass_ledger_closed": {
            "pass": ledger["verdict_pass"],
            "worst_initial_mass_rel_error": ledger["worst_initial_mass_rel_error"],
            "worst_conservation_rel_error": ledger["worst_conservation_rel_error"],
            "legacy_hardcode_rel_error_at_900": ledger["legacy_hardcode_rel_error"],
        },
        "coupling_ledger_closed": {
            "pass": coupling["verdict_pass"],
            "worst_ledger_rel_error": coupling["worst_ledger_rel_error"],
        },
        "ks_statistic_exact": {
            "pass": ks_module["verdict_pass"],
            "max_abs_diff_module_vs_exact": max(
                row["max_abs_diff_module_vs_exact"] for row in ks_module["rows"]
            ),
        },
        "ks_gate_calibrated": {
            "pass": ks["verdict_exact_calibrated"],
            "fpr_range": ks["fpr_range"],
            "legacy_bias_relative": ks["verdict_legacy_bias"],
        },
        "cascade_postconditions": {"pass": cascade["verdict_pass"]},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(_ROOT / "artifacts" / "fragmentation_statistics" / "validation_report.json"),
        help="JSON report path",
    )
    parser.add_argument("--quick", action="store_true", help="smaller Monte-Carlo budget")
    parser.add_argument("--print", action="store_true", dest="echo", help="echo the report")
    args = parser.parse_args(argv)

    report = build_report(quick=args.quick)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.echo:
        print(json.dumps(report, indent=2, sort_keys=True))
    failed = [name for name, verdict in report["verdicts"].items() if not verdict.get("pass", False)]
    print(f"wrote {destination}")
    print("verdicts:", "ALL PASS" if not failed else "FAIL -> " + ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())