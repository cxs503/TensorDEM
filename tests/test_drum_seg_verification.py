"""Tests for the p176w2-n08 rotating-drum bidisperse segregation verification surface.

Coverage map
------------
A. metrics (binning, indices, permutation null, mode classification)
B. DEM kernel (packing, wall geometry, static control, rotation drive, determinism)
C. axial reduced-order model (analytic growth rate, wavelength, conservation, limits)
D. criteria evaluation + scaling fit
E. end-to-end case record (schema of the artefact consumed by the report)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.drum_segregation import axial, metrics, sweeps, verification
from tensordem.drum_segregation.kernel2d import (
    DrumConfig, coulomb_damping_ratio, make_bed, simulate,
)

R_DEF = 0.15


# ---------------------------------------------------------------- helpers
def _ring_bed(n=300, r_lo=0.45, seed=0, drum_r=R_DEF):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(r_lo ** 2, 1.0, n)) * drum_r
    th = rng.uniform(0.0, 2 * math.pi, n)
    pos = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    radii = np.full(n, 0.005)
    is_large = np.zeros(n, bool)
    return pos, radii, is_large


def _tiny_cfg(**kw):
    base = dict(radius=R_DEF, fill=0.30, d_small=0.025, d_large=0.040,
                large_area_fraction=0.30, omega=2.0, dt=1.5e-4, mu_rolling=0.10,
                settle_steps=800, revs=0.25, sample_revs=0.1, n_bins=6,
                seed=0)
    base.update(kw)
    return DrumConfig(**base)


# ---------------------------------------------------------------- A metrics
def test_radial_profile_occupancy_and_phi0():
    pos, radii, is_large = _ring_bed()
    is_large[:90] = True                       # 30 % of particles
    prof = metrics.radial_profile(pos, radii, is_large, R_DEF, n_bins=6)
    assert prof["phi"].shape == (6,)
    assert prof["edges"][0] > 0.0              # bins start at the occupied radius
    assert abs(prof["phi_0"] - 0.30) < 0.02
    assert np.isclose(prof["area_weight"].sum(), 1.0)


def test_radial_profile_empty_bins_are_nan():
    pos = np.array([[0.1, 0.0], [0.11, 0.0], [0.12, 0.0], [0.149, 0.0]])
    radii = np.full(4, 0.005)
    is_large = np.array([True, False, False, True])
    prof = metrics.radial_profile(pos, radii, is_large, R_DEF, n_bins=8, min_count=3)
    assert np.isnan(prof["phi"]).sum() >= 5    # sparse bins must be dropped


def test_size_position_correlation_zero_for_monodisperse():
    pos, radii, is_large = _ring_bed()
    r_norm = np.linalg.norm(pos, axis=1) / R_DEF
    assert metrics.size_position_correlation(radii, r_norm) == 0.0   # exact zero


def test_size_position_correlation_detects_perfect_ordering():
    r_norm = np.linspace(0.5, 1.0, 50)
    radii = np.linspace(0.002, 0.006, 50)
    assert metrics.size_position_correlation(radii, r_norm) > 0.999
    assert metrics.size_position_correlation(radii[::-1], r_norm) < -0.999


def test_concentration_segregation_zero_for_uniform_profile():
    prof = {"phi": np.full(8, 0.3), "area_weight": np.full(8, 1 / 8), "phi_0": 0.3}
    assert metrics.concentration_segregation(prof) == pytest.approx(0.0, abs=1e-15)


def test_core_shell_excess_sign():
    phi = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
    centres = np.array([0.55, 0.62, 0.70, 0.80, 0.90, 0.98])
    prof = {"phi": phi, "centres": centres, "area_weight": np.ones(6) / 6, "phi_0": 0.3}
    assert metrics.core_shell_excess(prof) > 0                     # periphery mode
    prof["phi"] = phi[::-1]
    assert metrics.core_shell_excess(prof) < 0                     # core mode


def test_permutation_null_detects_artificial_segregation():
    pos, radii, is_large = _ring_bed(400)
    r_norm = np.linalg.norm(pos, axis=1) / R_DEF
    is_large = r_norm > np.quantile(r_norm, 0.7)          # coarse pushed outward
    radii = np.where(is_large, 0.006, 0.003)
    null = metrics.permutation_null(pos, radii, is_large, R_DEF, n_bins=6, n_perm=80)
    assert null["excess_observed"] > 0.3
    assert null["excess_z"] > 5.0
    assert null["corr_observed"] > 0.5
    assert metrics.classify_mode(null["excess_observed"], null["excess_z"],
                                 null["corr_observed"], null["corr_z"]) == "periphery"


def test_permutation_null_is_flat_for_random_arrangement():
    pos, radii, is_large = _ring_bed(400, seed=3)
    radii = np.where(is_large, 0.006, 0.003)
    null = metrics.permutation_null(pos, radii, is_large, R_DEF, n_bins=6, n_perm=120)
    assert abs(null["excess_z"]) < 3.0
    assert metrics.classify_mode(null["excess_observed"], null["excess_z"],
                                 null["corr_observed"], null["corr_z"]) == "mixed"


def test_classify_mode_core():
    assert metrics.classify_mode(-0.3, -6.0, -0.4, -5.0) == "core"
    assert metrics.classify_mode(0.01, 0.5, 0.02, 0.4) == "mixed"


# ---------------------------------------------------------------- B kernel
def test_make_bed_places_all_particles_inside_drum_without_overlap():
    cfg = _tiny_cfg()
    pos, radii, is_large = make_bed(cfg)
    assert len(pos) == len(radii) == len(is_large)
    r = np.linalg.norm(pos, axis=1)
    assert np.all(r + radii <= cfg.radius + 2e-3)
    from scipy.spatial import cKDTree
    pairs = cKDTree(pos).query_pairs(2 * radii.max())
    if len(pairs):
        i, j = np.array(sorted(pairs)).T
        d = np.linalg.norm(pos[j] - pos[i], axis=1)
        overlap = (radii[i] + radii[j]) - d
        assert overlap.max() < 0.02 * (radii[i] + radii[j]).max()


def test_bed_is_well_mixed_initially():
    """Species must be uncorrelated with position in the initial state."""
    cfg = _tiny_cfg()
    pos, radii, is_large = make_bed(cfg)
    r_norm = np.linalg.norm(pos, axis=1) / cfg.radius
    assert abs(metrics.size_position_correlation(radii, r_norm)) < 0.35


def test_static_drum_is_a_negative_control():
    cfg = _tiny_cfg(omega=0.0, revs=0.5, settle_steps=1500)
    run = simulate(cfg)
    assert run.diagnostics["settle_rms_speed"] < 2e-2
    assert run.diagnostics["initial_ke"] < 1e-3
    run = simulate(cfg)
    null = metrics.permutation_null(run.pos, run.radii, run.is_large,
                                    cfg.radius, cfg.n_bins, n_perm=60)
    assert abs(null["excess_z"]) < 3.0
    assert abs(null["corr_observed"]) < 0.25


def test_bed_stays_inside_rotating_drum():
    cfg = _tiny_cfg()
    run = simulate(cfg)
    r = np.linalg.norm(run.pos, axis=1)
    assert np.all(r + run.radii <= cfg.radius + 5e-3)
    assert run.diagnostics["max_overlap_ratio"] < 0.05


def test_rotation_drives_the_bed():
    """A rotating wall must put kinetic energy into the bed."""
    cfg_static = _tiny_cfg(omega=0.0, revs=0.6, settle_steps=1500)
    cfg_drive = _tiny_cfg(omega=3.0, revs=0.6, settle_steps=1500)
    r_static = simulate(cfg_static)
    r_drive = simulate(cfg_drive)
    v_static = abs(r_static.diagnostics["mean_azimuthal_speed"])
    v_drive = abs(r_drive.diagnostics["mean_azimuthal_speed"])
    assert v_drive > 3.0 * max(v_static, 1e-9)
    # the rotating bed must develop a dynamic repose slope (rolling regime)
    assert abs(r_drive.diagnostics["surface_slope_deg"]) > 5.0


def test_monodisperse_run_has_zero_size_position_index():
    cfg = _tiny_cfg(d_large=0.025)
    run = simulate(cfg)
    r_norm = np.linalg.norm(run.pos, axis=1) / cfg.radius
    assert metrics.size_position_correlation(run.radii, r_norm) == 0.0


def test_simulation_is_deterministic_for_a_fixed_seed():
    a = simulate(_tiny_cfg(revs=0.15))
    b = simulate(_tiny_cfg(revs=0.15))
    assert np.allclose(a.pos, b.pos, atol=1e-12)


def test_coulomb_damping_ratio_limits():
    assert coulomb_damping_ratio(1.0) < 1e-3          # (e clamped to 0.999)
    assert 0.2 < coulomb_damping_ratio(0.3) < 0.6


def test_series_is_monotone_in_time_length():
    run = simulate(_tiny_cfg())
    for key in ("rev", "segregation", "size_pos_corr", "kinetic_energy"):
        assert len(run.series[key]) == len(run.series["rev"])
    assert run.series["rev"][-1] > run.series["rev"][0]


# ---------------------------------------------------------------- C axial
@pytest.fixture(scope="module")
def axial_chk():
    return axial.axial_verification(n=128, kappa=5e-4)


def test_axial_growth_rate_matches_linear_theory(axial_chk):
    chk = axial_chk
    for mode_row in chk["A1_growth_rate"].values():
        assert mode_row["rel_error"] < 0.03, mode_row


def test_axial_wavelength_selection(axial_chk):
    chk = axial_chk
    assert chk["A2_wavelength"]["rel_error"] < 0.10
    assert abs(chk["A2_wavelength"]["k_max_measured"] - chk["A2_wavelength"]["k_max_theory"]) \
        / chk["A2_wavelength"]["k_max_theory"] < 0.25


def test_axial_conservation(axial_chk):
    chk = axial_chk
    assert chk["A3_conservation"]["value"] <= 1e-10


def test_axial_spinodal_negative_control_and_kappa_limit(axial_chk):
    chk = axial_chk
    assert chk["A4_spinodal_negative_control"]["pass"]        # phi0=0.8 decays
    assert chk["A5_no_kappa_limit"]["pass"]                   # no scale selection
    assert chk["all_pass"]


# ---------------------------------------------------------------- D criteria
def test_fit_scaling_recovers_known_exponent():
    ratios = np.array([1.2, 1.5, 2.0, 3.0])
    amp = 0.4 * (ratios - 1.0) ** 0.7
    fit = verification.fit_scaling(ratios, amp)
    assert fit["alpha"] == pytest.approx(0.7, abs=1e-6)
    assert fit["A"] == pytest.approx(0.4, rel=1e-6)
    assert fit["R2"] > 0.999


def test_criteria_reject_when_direction_is_wrong():
    """A 'core' pattern (reverse Brazil-nut) for every ratio must fail C1."""
    cases = []
    for ratio in (1.2, 1.5, 2.0, 3.0):
        cases.append({
            "name": f"ratio_{ratio:g}", "group": "ratio", "mode": "core",
            "null": {"excess_observed": -0.3, "excess_z": -8.0,
                     "corr_observed": -0.4, "corr_z": -7.0,
                     "S_observed": 0.6, "S_z": 9.0, "S_null_mean": 0.05},
        })
    out = verification.evaluate(cases, axial_checks=None, n_pytest=20)
    assert out["criteria"]["C1"]["n_compliant"] == 0
    assert out["criteria"]["C1"]["pass"] is False


def test_criteria_pass_on_consistent_periphery_case_set():
    """Synthetic consistent data set: all six criteria must be satisfiable."""
    cases = []
    ratios = (1.2, 1.5, 2.0, 3.0)
    amps = (0.10, 0.18, 0.30, 0.55)
    for ratio, amp in zip(ratios, amps):
        cases.append({
            "name": f"ratio_{ratio:g}", "group": "ratio", "mode": "periphery",
            "null": {"excess_observed": amp, "excess_z": 8.0,
                     "corr_observed": 0.2, "corr_z": 6.0,
                     "S_observed": 0.5, "S_z": 8.0, "S_null_mean": 0.05},
        })
    for nm, rat in (("mono_1.00_s0", 1.0), ("mono_1.00_s1", 1.0),
                    ("mono_1.05", 1.05), ("mono_1.10", 1.10)):
        cases.append({"name": nm, "group": "mono", "mode": "mixed",
                      "null": {"excess_observed": 0.02 * (rat - 1.0), "excess_z": 0.4,
                               "corr_observed": 0.0, "corr_z": 0.0,
                               "S_observed": 0.02, "S_z": 0.1, "S_null_mean": 0.02}})
    for om in (0.0, 0.5, 1.0, 4.0):
        cases.append({"name": f"omega_{om:g}", "group": "omega", "mode": "mixed",
                      "null": {"excess_observed": 0.01 + 0.12 * om, "excess_z": 1.0,
                               "corr_observed": 0.02 * om, "corr_z": 1.0,
                               "S_observed": 0.05, "S_z": 1.0, "S_null_mean": 0.03}})
    for fl in (0.15, 0.5, 0.65):
        amp = 0.4 if fl < 0.5 else 0.05
        cases.append({"name": f"fill_{fl:g}", "group": "fill", "mode": "mixed",
                      "null": {"excess_observed": amp, "excess_z": 0.5,
                               "corr_observed": 0.01, "corr_z": 0.5,
                               "S_observed": 0.05, "S_z": 1.0, "S_null_mean": 0.03}})
    axial_chk = {"A2_wavelength": {"pass": True}}
    out = verification.evaluate(cases, axial_checks=axial_chk, n_pytest=20)
    assert out["criteria"]["C1"]["pass"]
    assert out["criteria"]["C2"]["pass"]
    assert out["criteria"]["C3"]["pass"]
    assert out["criteria"]["C4"]["pass"]
    assert out["criteria"]["C5"]["pass"]
    assert out["criteria"]["C6"]["pass"]
    assert out["criteria"]["C6"]["n_pytest"] == 20


# ---------------------------------------------------------------- E matrix / e2e
def test_case_matrix_contains_required_ratios_and_controls():
    cases = {c["name"]: c for c in sweeps.build_cases()}
    for ratio in (1.2, 1.5, 2.0, 3.0):
        assert f"ratio_{ratio:g}" in cases
        assert cases[f"ratio_{ratio:g}"]["cfg"]["d_large"] == \
            pytest.approx(ratio * cases[f"ratio_{ratio:g}"]["cfg"]["d_small"])
    assert cases["mono_1.00_s0"]["cfg"]["d_large"] == cases["mono_1.00_s0"]["cfg"]["d_small"]
    assert cases["omega_0"]["cfg"]["omega"] == 0.0
    assert cases["fill_0.65"]["cfg"]["fill"] == 0.65
    ref = cases["ratio_2"]["cfg"]
    assert 0.02 < ref["omega"] ** 2 * ref["radius"] / 9.81 < 0.3   # rolling regime
    assert cases["sat_ratio2_rev28"]["cfg"]["revs"] == 28.0


def test_end_to_end_case_record(tmp_path):
    from scripts.run_drum_seg_verification import run_case
    case = {"name": "e2e", "group": "ratio",
            "seed": 0, "cfg": dict(_tiny_cfg().__dict__)}
    rec = run_case(case)
    for key in ("name", "group", "cfg", "null", "profile", "series", "mode",
                "n_particles", "froude", "final"):
        assert key in rec
    assert rec["mode"] in ("periphery", "core", "mixed")
    assert len(rec["series"]["segregation"]) == len(rec["series"]["rev"]) > 0
    assert math.isfinite(rec["null"]["excess_observed"])