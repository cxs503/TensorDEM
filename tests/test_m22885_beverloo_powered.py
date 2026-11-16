"""Pins the m22885 Beverloo ensemble estimator on synthetic ground truth.

Nothing here needs a DEM run: every branch of `analyse.py` (PASS / FAIL /
NOT_DECIDABLE / NOT_RUN), the Student-t quantiles, the CI arithmetic, the drift
regression and the counting-noise budget are exercised on planted numbers, so a
verdict computed from real artefacts cannot be a coding accident.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str):
    """Load a benchmark module under a unique name.

    Several benchmark directories ship their own ``analyse.py``; importing them as
    the bare module name ``analyse`` makes whichever test file is collected first
    win, which silently mixes estimators between missions (a real cross-file
    pollution bug already visible between ``test_m22657`` and ``test_m22628``).
    Loading by path under a mission-scoped name keeps this suite order-independent.
    """
    import importlib.util
    path = ROOT / f"benchmarks/m22885_beverloo_powered/{name}.py"
    spec = importlib.util.spec_from_file_location(f"m22885_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("analyse")

D = 0.015
d = 0.003
RHO = 1375.0
W_REF = A.beverloo_ref(D, d, RHO)


def synth(seed: int, scale: float = 1.0, fill: float = 0.036, rho_meas: float = RHO,
          rho_nom: float = RHO, n_exit: int = 340, drift: float = 0.0) -> dict:
    """A discharge JSON shaped like `beverloo.run_discharge` output."""
    n = 8
    base = A.beverloo_ref(D, d, rho_meas) * scale
    rates = [base * (1.0 - drift * i / (n - 1)) for i in range(n)]
    per_window_mass = [r * 0.05 for r in rates]
    heights = [fill - 0.001 * i for i in range(n)]
    return {
        "case": {"orifice_diameter": D, "particle_diameter": d, "orifice_over_particle": D / d,
                 "silo_radius": 0.024, "fill_height": fill, "wall_mu": 0.0, "seed": seed,
                 "kn": 5e3, "mu": 0.5, "mu_rolling": 0.05},
        "dt": 1.866e-5, "n_bed": 2534, "bed_mass_kg": 0.09,
        "rho_bulk": rho_nom, "rho_bulk_measured": rho_meas, "surface_height_m": fill - 0.004,
        "n_windows": n, "window_time_s": 0.05, "discard_time_s": 0.12,
        "flow_rate_windows_kg_s": rates,
        "flow_rate_kg_s": float(np.mean(rates[n // 2:])),
        "flow_rate_std_kg_s": 0.0, "flow_rate_steady_windows": n // 2,
        "particles_exited": n_exit,
        "window_particles": [n_exit // n + (1 if i < n_exit % n else 0) for i in range(n)],
        "window_surface_height_m": heights,
        "n_final": 2000, "energy": {}, "wall_clock_s": 1.0, "steps": 31000,
        "_per_window_mass": per_window_mass,
    }


# --------------------------------------------------------------------- unit bits
def test_t_quantiles_match_published_table():
    assert A.t975(1) == pytest.approx(12.706)
    assert A.t975(5) == pytest.approx(2.571)
    assert A.t975(11) == pytest.approx(2.201)
    assert A.t975(20) == pytest.approx(2.086)
    assert A.t975(1000) == pytest.approx(1.960)
    # monotone decreasing in dof, and continuous across the fallback seam at 30/31
    prev = float("inf")
    for dof in (1, 2, 5, 10, 20, 29, 30, 31, 60, 61, 120, 121):
        cur = A.t975(dof)
        assert cur <= prev + 1e-12
        prev = cur
    assert A.t975(31) == pytest.approx(A.t975(30), rel=0.02)


def test_counting_noise_floor():
    assert A.counting_noise(100) == pytest.approx(0.1)
    assert A.counting_noise(340) == pytest.approx(1 / math.sqrt(340))
    assert math.isnan(A.counting_noise(0))


def test_beverloo_reference_uses_literature_coefficients():
    assert W_REF == pytest.approx(0.58 * RHO * math.sqrt(9.81) * (D - 1.4 * d) ** 2.5)
    assert W_REF == pytest.approx(0.0303, rel=0.02)          # ~30 g/s at D/d = 5
    assert A.beverloo_ref(1.0 * d, d, RHO) == 0.0            # effective orifice <= 0


def test_mean_ci_matches_hand_arithmetic():
    stats = A.mean_ci([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    assert stats["n"] == 6
    assert stats["mean"] == pytest.approx(0.035)
    assert stats["sd"] == pytest.approx(float(np.std([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], ddof=1)))
    assert stats["ci95"] == pytest.approx(A.t975(5) * stats["sd"] / math.sqrt(6))
    assert A.mean_ci([])["mean"] is None
    assert A.mean_ci([0.5])["ci95"] is None       # n = 1 -> no interval, not a fake zero


# ------------------------------------------------------------------- verdicts
def test_verdict_band_all_three_branches():
    assert A.verdict_band({"mean": 0.02, "ci95": 0.02}) == "PASS"            # [-0.0, 0.04]
    assert A.verdict_band({"mean": 0.25, "ci95": 0.05}) == "FAIL"
    assert A.verdict_band({"mean": 0.13, "ci95": 0.05}) == "NOT_DECIDABLE"
    assert A.verdict_band({"mean": None, "ci95": None}) == "NOT_RUN"


def test_ensemble_recovers_planted_scale_and_interval():
    runs = [synth(s, scale=1.06) for s in range(1, 13)]
    res = A.ensemble(runs)
    assert res["rel"]["mean"] == pytest.approx(0.06, abs=0.005)
    assert res["V1_ledger"] == "NOT_DECIDABLE" or res["V1_ledger"] == "PASS"
    # zero scatter -> CI is 0 and 6 % is inside the band
    assert res["rel"]["ci95"] == pytest.approx(0.0)
    assert res["V1_ledger"] == "PASS"
    assert res["V6_noise_floor"] == "FAIL"        # 340 particles -> 5.4 % floor


def test_ensemble_flags_honest_fail_when_bias_exceeds_tolerance():
    runs = [synth(s, scale=1.25) for s in range(1, 13)]
    res = A.ensemble(runs)
    assert res["V1_ledger"] == "FAIL"
    assert res["rel"]["mean"] == pytest.approx(0.25, abs=0.005)


def test_ensemble_requires_data_for_not_run():
    res = A.ensemble([])
    assert res["V1_ledger"] == "NOT_RUN"
    assert res["V2_reproducibility"] is None and res["V3_steady_state"] is None


def test_reproducibility_and_drift_verdicts():
    tight = [synth(s, scale=1.0 + 0.005 * (s - 6)) for s in range(1, 13)]
    assert A.ensemble(tight)["V2_reproducibility"] == "PASS"
    loose = [synth(s, scale=1.0 + 0.10 * (s - 6)) for s in range(1, 13)]
    assert A.ensemble(loose)["V2_reproducibility"] == "FAIL"
    steady = [synth(s, drift=0.02) for s in range(1, 13)]
    assert A.ensemble(steady)["V3_steady_state"] == "PASS"
    dropping = [synth(s, drift=0.30) for s in range(1, 13)]
    assert A.ensemble(dropping)["V3_steady_state"] == "FAIL"


def test_drift_fraction_recovers_planted_slope():
    rates = [1.0 - 0.05 * i for i in range(8)]
    heights = [0.036 - 0.0005 * i for i in range(8)]
    got = A.drift_fraction(rates, heights)
    assert got == pytest.approx(0.05 * 7 / np.mean(rates), rel=1e-6)
    with pytest.raises(ValueError):
        A.drift_fraction([1.0, 2.0], [0.1, 0.2])


def test_rho_convention_verdict():
    # convention matters (2 % shift) and scatter is small -> PASS
    runs = [synth(s, rho_meas=RHO * 1.02, rho_nom=RHO) for s in range(1, 13)]
    res = A.ensemble(runs)
    assert res["rho_convention_shift"] == pytest.approx(0.02, rel=1e-6)
    assert res["V4_rho_control"] == "PASS"
    # the fix is moot if the conventions coincide -> FAIL (do not report a vacuous pass)
    same = [synth(s, rho_meas=RHO, rho_nom=RHO) for s in range(1, 13)]
    assert A.ensemble(same)["V4_rho_control"] == "FAIL"


def test_ledger_uses_measured_bulk_density_not_nominal():
    """The convention shift is one-directional: nominal rho_b biases W/W_ref HIGH."""
    runs = [synth(s, scale=1.0, rho_meas=RHO * 1.05, rho_nom=RHO) for s in range(1, 13)]
    measured = A.ensemble(runs, rho_key="rho_bulk_measured")
    # rates were planted at the measured-rho_b reference -> zero deviation
    assert measured["rel"]["mean"] == pytest.approx(0.0, abs=1e-9)
    # with the old nominal convention W_ref is 5 % low -> every ratio reads +5 %
    nominal = A.ensemble(runs, rho_key="rho_bulk")
    assert nominal["rel"]["mean"] == pytest.approx(0.05, abs=1e-9)


def test_h_independence_verdict():
    base = A.ensemble([synth(s, fill=0.036, scale=1.0) for s in range(1, 5)])
    tall = A.ensemble([synth(s, fill=0.048, scale=1.02) for s in range(1, 5)])
    got = A.h_independence(base, tall)
    assert got["ratio"] == pytest.approx(1.02, rel=1e-9)
    assert got["V5_H_independence"] == "PASS"
    bad = A.h_independence(base, A.ensemble([synth(s, fill=0.048, scale=1.4) for s in range(1, 5)]))
    assert bad["V5_H_independence"] == "FAIL"
    assert A.h_independence(A.ensemble([]), tall)["V5_H_independence"] == "NOT_RUN"


def test_analyse_dir_groups_by_fill_height(tmp_path):
    for s in range(1, 5):
        (tmp_path / f"fill36_seed{s}.json").write_text(json.dumps(synth(s, fill=0.036)))
    for s in range(1, 3):
        (tmp_path / f"fill48_seed{s}.json").write_text(json.dumps(synth(s, fill=0.048, scale=1.03)))
    res = A.analyse_dir(tmp_path)
    assert res["n_runs"] == 6
    assert set(res["groups"]) == {"fill_36.0mm", "fill_48.0mm"}
    assert res["groups"]["fill_36.0mm"]["n"] == 4
    assert res["V5_H_independence"]["ratio"] == pytest.approx(1.03, rel=1e-6)
    # an empty directory must yield NOT_RUN everywhere, never a silent pass
    empty = A.analyse_dir(tmp_path / "nope") if (tmp_path / "nope").mkdir() is None else None
    assert empty["n_runs"] == 0 and empty["V5_H_independence"]["V5_H_independence"] == "NOT_RUN"


# ------------------------------------------------- per-run reduction integrity
def test_per_run_reduces_frozen_quantities():
    run = synth(7, scale=1.04)
    row = A.per_run(run)
    assert row["seed"] == 7
    assert row["rel"] == pytest.approx(0.04, abs=1e-9)
    assert row["w_s"] == pytest.approx(run["flow_rate_kg_s"])
    assert row["n_exit"] == run["particles_exited"]
    assert row["noise"] == pytest.approx(1 / math.sqrt(run["particles_exited"]))
    assert row["c_recovered"] == pytest.approx(0.58 * 1.04, rel=1e-9)


def test_wrapper_freezes_the_preregistered_configuration():
    run_case = _load("run_case")
    assert run_case.FROZEN["ratio"] * run_case.FROZEN["particle_diameter"] == pytest.approx(0.015)
    assert run_case.FROZEN["silo_radius"] * 2 / 0.015 == pytest.approx(3.2)
    dom = run_case.dimensionless(36.0)
    assert dom["in_domain"] is True
    assert dom["D_over_d"] == pytest.approx(5.0)
    cmd = run_case.command(36.0, 3, Path("/tmp/x.json"))
    assert "--fill-height" in cmd and cmd[cmd.index("--fill-height") + 1] == "0.036"
    assert cmd[cmd.index("--seed") + 1] == "3"
    # the secondary axis is still inside Beverloo's validity window
    assert run_case.dimensionless(48.0)["in_domain"] is True


def test_run_discharge_reports_per_window_diagnostics():
    """The measurement-reporting fix is present in the kernel path (no run needed)."""
    src = (ROOT / "src/tensordem/silo_flow/beverloo.py").read_text()
    for field in ("rho_bulk_measured", "surface_height_m", "window_particles",
                  "window_surface_height_m"):
        assert field in src
    from tensordem.silo_flow.kernel import Material, SiloGeometry, SiloFlow
    geom = SiloGeometry(cylinder_radius=0.012, cylinder_height=0.05, outlet_radius=0.0075)
    mat = Material(radius=0.0015, mu=0.5, kn=5e3)
    flow = SiloFlow(np.array([[0.0, 0.0, 0.010], [0.0, 0.0, 0.012], [0.001, 0.0, 0.008]]),
                    mat, geom)
    h = flow.surface_height()
    assert 0.009 <= h <= 0.0145         # mean top layer (0.010, 0.012) + r = 0.0125
    assert flow.bulk_density(0.012) > flow.bulk_density(0.020)   # nominal line over-states volume