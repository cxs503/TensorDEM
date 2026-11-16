"""Pin the m22657 ensemble estimator before any DEM run is read.

The point of these tests is that the verdict machinery cannot be quietly tuned to
whatever the runs return: the statistics, the sign conventions, the frozen
configuration and the honest-FAIL / NOT_DECIDABLE branches are all exercised on
synthetic artefacts.

Covers:
  * Student-t CDF / quantile against published table values;
  * window-drift estimator on exact linear and on trend-free series;
  * counting-noise budget ``1/sqrt(N)``;
  * ensemble mean / sd / 95 % CI;
  * the four verdict branches (PASS / FAIL / NOT_DECIDABLE / NOT_RUN);
  * the frozen configuration is inside the law's domain and identical for all
    seeds;
  * a synthetic ensemble built with a planted -6.9 % bias is recovered to 1e-9.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "m22657_beverloo_ensemble"
for p in (ROOT / "src", BENCH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

def _load(name: str):
    """Load a benchmark module under a mission-scoped name.

    Every benchmark directory ships its own ``analyse.py``/``run_case.py``;
    importing them by bare module name makes whichever test file pytest collects
    first win the ``sys.modules`` slot and silently swaps estimators between
    missions.  Loading by path keeps this suite order-independent.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"m22657_{name}", BENCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("analyse")
R = _load("run_case")


# ------------------------------------------------------------------ t distribution
@pytest.mark.parametrize("df,p,expected", [
    (1, 0.975, 12.706),
    (2, 0.975, 4.303),
    (5, 0.975, 2.571),
    (9, 0.975, 2.262),
    (30, 0.95, 1.697),
])
def test_student_quantile_matches_tables(df, p, expected):
    assert A.student_quantile(p, df) == pytest.approx(expected, abs=5e-3)


def test_student_cdf_is_antisymmetric_and_monotone():
    for df in (1, 3, 8, 40):
        assert A.student_cdf(0.0, df) == pytest.approx(0.5, abs=1e-12)
        assert A.student_cdf(-1.3, df) == pytest.approx(1.0 - A.student_cdf(1.3, df), abs=1e-10)
        assert A.student_cdf(-2.0, df) < A.student_cdf(0.0, df) < A.student_cdf(2.0, df)


# ----------------------------------------------------------------- window drift
def test_window_trend_recovers_exact_linear_ramp():
    rates = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    tr = A.window_trend(rates)
    assert tr["slope_kg_s_per_window"] == pytest.approx(0.1, rel=1e-12)
    # drift across the counted set (4 intervals) relative to the mean 1.2
    assert tr["rel_drift"] == pytest.approx(0.4 / 1.2, rel=1e-12)
    assert tr["n"] == 5


def test_window_trend_flat_series_has_no_drift():
    tr = A.window_trend(np.full(6, 0.02))
    assert tr["rel_drift"] == pytest.approx(0.0, abs=1e-15)
    assert not math.isfinite(tr["t_stat"]) or abs(tr["t_stat"]) < 1e-9


def test_window_trend_declining_bed_is_negative():
    tr = A.window_trend(np.array([0.05, 0.045, 0.04, 0.035]))
    assert tr["rel_drift"] < 0.0
    assert tr["t_stat"] < 0.0


# ------------------------------------------------------------- noise budget
def test_poisson_budget_is_one_over_sqrt_n():
    b = A.poisson_noise_budget([100.0] * 4, mass_per_particle=1e-5, window_time=0.05)
    assert b["n_exit_counted"] == pytest.approx(400.0)
    assert b["expected_rel_noise"] == pytest.approx(1.0 / 20.0)
    assert b["mean_rate_kg_s"] == pytest.approx(400 * 1e-5 / 0.2)


def test_poisson_budget_floor_for_the_frozen_case():
    """prereg.md P1: ~340 particles during the counted windows -> ~5.4 % noise."""
    b = A.poisson_noise_budget([85.0] * 4, mass_per_particle=3.53e-5, window_time=0.05)
    assert 0.04 < b["expected_rel_noise"] < 0.07


# -------------------------------------------------------------- ensemble stats
def test_ensemble_stats_interval_contains_mean_and_scales_with_noise():
    v = np.array([0.098, 0.102, 0.100, 0.101, 0.097, 0.102])
    s = A.ensemble_stats(v)
    assert s["n"] == 6
    assert s["mean_kg_s"] == pytest.approx(v.mean())
    lo, hi = s["ci95_kg_s"]
    assert lo < s["mean_kg_s"] < hi
    assert s["ci95_kg_s"][1] - s["ci95_kg_s"][0] == pytest.approx(2 * s["t_crit"] * s["se_kg_s"])
    noisy = A.ensemble_stats(v + np.array([0.0, 0.01, -0.01, 0.0, 0.0, 0.0]))
    assert noisy["sd_kg_s"] > s["sd_kg_s"]
    assert noisy["cv"] > s["cv"]


def test_ensemble_stats_single_seed_is_not_faked():
    s = A.ensemble_stats(np.array([0.1]))
    assert s["n"] == 1 and math.isnan(s["sd_kg_s"])


# ---------------------------------------------------------------- reference law
def test_reference_uses_literature_coefficients_only():
    W = A.beverloo_reference(0.015, 0.003, 1375.0)
    expect = 0.58 * 1375.0 * math.sqrt(9.81) * (0.015 - 1.4 * 0.003) ** 2.5
    assert W == pytest.approx(expect, rel=1e-12)
    assert A.beverloo_reference(0.003, 0.003, 1375.0) == 0.0  # orifice below k*d


# --------------------------------------------------------------- synthetic cases
PARTICLE_MASS = 3.53e-5  # kg, d = 3 mm glass


def _write_case(run_dir: Path, seed: int, W: float, rho_b: float = 1375.0,
                n_windows: int = 8, trend: float = 0.0, n_bed: int = 2530) -> None:
    """Write one artefact in the schema ``study beverloo`` produces."""
    window_time = 0.05
    rates = [W * (1.0 + trend * (i - (n_windows - 1) / 2) / max(n_windows - 1, 1))
             for i in range(n_windows)]
    payload = {
        "case": {"orifice_diameter": 0.015, "particle_diameter": 0.003,
                 "silo_radius": 0.024, "fill_height": 0.036,
                 "wall_mu": 0.0, "seed": seed, "kn": 5000.0, "mu": 0.5,
                 "mu_rolling": 0.05},
        "dt": 2.3e-5,
        "n_bed": n_bed,
        "bed_mass_kg": n_bed * PARTICLE_MASS,
        "rho_bulk": rho_b,
        "flow_rate_windows_kg_s": rates,
        "flow_rate_kg_s": float(np.mean(rates[-n_windows // 2:])),
        "flow_rate_std_kg_s": 0.0,
        "window_time_s": window_time,
        "particles_exited": 400,
        "n_final": n_bed - 400,
    }
    (run_dir / f"seed{seed}.json").write_text(json.dumps(payload))


def test_planted_bias_is_recovered(tmp_path):
    rho = 1375.0
    W_ref = A.beverloo_reference(0.015, 0.003, rho)
    for seed in range(1, 7):
        _write_case(tmp_path, seed, W_ref * 0.931, rho_b=rho)
    s = A.analyse(tmp_path)
    assert s["n_cases"] == 6
    assert s["ensemble"]["rel_err_ensemble"] == pytest.approx(-0.069, abs=1e-9)
    assert s["ensemble"]["cv"] == pytest.approx(0.0, abs=1e-12)
    assert s["verdicts"]["V1_ledger"] == "PASS"       # -6.9 % inside ±10 %, CI ~ 0
    assert s["verdicts"]["V2_reproducibility"] == "PASS"


def test_out_of_tolerance_is_honest_fail(tmp_path):
    rho = 1375.0
    W_ref = A.beverloo_reference(0.015, 0.003, rho)
    for seed in range(1, 7):
        _write_case(tmp_path, seed, W_ref * 1.20, rho_b=rho)
    s = A.analyse(tmp_path)
    assert s["ensemble"]["rel_err_ensemble"] == pytest.approx(0.20, abs=1e-9)
    assert s["verdicts"]["V1_ledger"] == "FAIL"


def test_wide_scatter_is_not_decidable(tmp_path):
    rho = 1375.0
    W_ref = A.beverloo_reference(0.015, 0.003, rho)
    spread = [0.80, 1.05, 0.90, 1.15, 0.95, 1.10]
    for k, f in enumerate(spread, start=1):
        _write_case(tmp_path, k, W_ref * f, rho_b=rho)
    s = A.analyse(tmp_path)
    rel = s["ensemble"]["rel_err_ensemble"]
    lo, hi = s["ensemble"]["rel_err_ci95"]
    assert abs(rel) < 0.10                     # point estimate is inside the band
    assert hi > 0.10 or lo < -0.10             # but the interval is not
    assert s["verdicts"]["V1_ledger"] == "NOT_DECIDABLE"
    assert s["ensemble"]["cv"] > 0.10
    assert s["verdicts"]["V2_reproducibility"] == "FAIL"


def test_no_runs_means_not_run(tmp_path):
    s = A.analyse(tmp_path)
    assert s["verdicts"]["V1_ledger"] == "NOT_RUN"
    assert s["n_cases"] == 0


def test_rho_spread_gate(tmp_path):
    rho = 1375.0
    W_ref = A.beverloo_reference(0.015, 0.003, rho)
    for seed in range(1, 7):
        _write_case(tmp_path, seed, W_ref, rho_b=rho * (1.0 + 0.02 * seed))
    s = A.analyse(tmp_path)
    assert s["verdicts"]["V4_rho_control"] == "FAIL"
    assert s["verdicts"]["supporting"]["V4_relative_rho_spread"] > 0.03


def test_persistent_drift_trips_v3(tmp_path):
    rho = 1375.0
    W_ref = A.beverloo_reference(0.015, 0.003, rho)
    for seed in range(1, 7):
        _write_case(tmp_path, seed, W_ref, rho_b=rho, trend=0.30)
    s = A.analyse(tmp_path)
    assert s["verdicts"]["V3_steady_state"] == "FAIL"
    assert s["verdicts"]["supporting"]["V3_max_abs_rel_drift"] > 0.10


# ----------------------------------------------------------- frozen configuration
def test_frozen_configuration_is_inside_the_law_domain():
    dom = R.dimensionless(R.FROZEN["ratio"] * R.FROZEN["particle_diameter"],
                          R.FROZEN["particle_diameter"], R.FROZEN["silo_radius"],
                          R.FROZEN["fill_height"])
    assert dom["in_domain"] is True
    assert dom["D_over_d"] == pytest.approx(5.0)
    assert dom["silo_diameter_over_D"] >= 2.5
    assert dom["fill_over_D"] >= 1.5


def test_frozen_configuration_is_seed_independent():
    cmds = [R.command(s, f"/tmp/x{s}.json") for s in (1, 2, 6)]
    stripped = []
    for c in cmds:
        c = list(c)
        i_seed = c.index("--seed")
        i_out = c.index("--out")
        stripped.append([x for k, x in enumerate(c) if k not in (i_seed + 1, i_out + 1)])
    assert stripped[0] == stripped[1] == stripped[2]


def test_command_points_at_the_campaign_kernel_driver():
    cmd = R.command(4, "/tmp/y.json")
    assert "tensordem.silo_flow.study" in cmd
    assert cmd[cmd.index("--n-windows") + 1] == "8"
    assert cmd[cmd.index("--window-time") + 1] == "0.05"


def test_reference_json_is_machine_readable():
    ref = json.loads((ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json").read_text())
    assert 0.55 <= float(ref["coefficients"]["C"]) <= 0.60
    assert 1.2 <= float(ref["coefficients"]["k"]) <= 1.6