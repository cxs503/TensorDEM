"""Tests for the joint three-benchmark DEM calibration (node P284W3-N08).

Covers the contact engine (walls, bonds), the three benchmark drivers, the
response features / grouped objective, the posterior-predictive coverage check
and the campaign acceptance thresholds.  The DEM cases in the regression tests
are deliberately tiny so the suite stays fast; the production protocol lives in
``scripts/run_joint_calibration.py`` and is recorded in
``docs/P284W3_N08_JOINT_CALIBRATION.md``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.dem_calibration.contact_engine import (
    PARAMETER_BOUNDS,
    PARAMETER_NAMES,
    ContactParams,
    GrainMaterial,
    WallState,
    bond_forces,
    make_bonds,
    moving_average,
    plane_forces,
)
from tensordem.dem_calibration.joint_benchmarks import (
    ReposeCase,
    ShearCase,
    UniaxialCase,
    repose_angle,
    run_repose,
    run_shear,
    run_uniaxial,
)
from tensordem.dem_calibration import joint_calibration as jc
from tensordem.dem_calibration.inverse import ParameterSpace

MAT = GrainMaterial(radius=0.004)
#: the repose test runs 6 mm grains (see joint_calibration.ExperimentSpec)
MAT_REPOSE = GrainMaterial(radius=0.006)
PARAMS = ContactParams(kn=900.0, kt=450.0, mu=0.45, restitution=0.30, sigma_bond=6.0e3)


# --------------------------------------------------------------------- params
def test_contact_params_validation_and_roundtrip():
    with pytest.raises(ValueError):
        ContactParams(kn=-1.0)
    with pytest.raises(ValueError):
        ContactParams(restitution=1.0)
    with pytest.raises(ValueError):
        ContactParams(mu=-0.1)
    d = PARAMS.as_dict()
    assert set(d) == set(PARAMETER_NAMES)
    assert ContactParams.from_mapping(d) == PARAMS
    with pytest.raises(KeyError):
        ContactParams.from_mapping({"kn": 1.0})


def test_parameter_space_matches_bounds():
    space = jc.PARAMETER_SPACE
    assert space.dim == 5
    assert space.names == PARAMETER_NAMES
    for name, (lo, hi) in PARAMETER_BOUNDS.items():
        assert space.lower[space.names.index(name)] == pytest.approx(lo)
        assert space.upper[space.names.index(name)] == pytest.approx(hi)
    truth = jc.TRUTH["sand"].as_dict()
    for name, value in truth.items():
        lo, hi = PARAMETER_BOUNDS[name]
        assert lo < value < hi, name          # truth strictly inside the box


# ------------------------------------------------------------- repose estimator
def test_repose_angle_recovers_an_analytic_cone():
    """A synthetic cone of grains must give back its slope angle."""
    rng = np.random.default_rng(0)
    theta_true = 28.0
    r = 0.004
    n = 3000
    x = 0.06 * np.sqrt(rng.random(n)) * np.cos(2 * np.pi * rng.random(n))
    y = 0.06 * np.sqrt(rng.random(n)) * np.sin(2 * np.pi * rng.random(n))
    rad = np.hypot(x, y)
    z = 0.05 - math.tan(math.radians(theta_true)) * rad
    pos = np.column_stack([x, y, z + rng.normal(0.0, 0.2 * r, n)])
    fit = repose_angle(pos, r)
    # the top-surface estimator is a *max* statistic, so it is biased towards
    # the upper grain envelope: the recovered angle is flatter by ~1-3 deg
    assert fit["theta_deg"] == pytest.approx(theta_true, abs=3.0)
    assert fit["theta_deg"] < theta_true
    assert fit["r2"] > 0.9


def test_repose_angle_rejects_a_rising_surface():
    rng = np.random.default_rng(1)
    r = 0.004
    rad = np.linspace(0.0, 0.05, 600)
    pos = np.column_stack([rad * np.cos(2 * np.pi * rng.random(600)),
                           rad * np.sin(2 * np.pi * rng.random(600)),
                           rad * 0.7 + rng.normal(0.0, 0.2 * r, 600)])
    with pytest.raises(ValueError):
        repose_angle(pos, r)


def test_moving_average_preserves_the_mean():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sm = moving_average(x, 3)
    assert sm.shape == x.shape
    assert sm.mean() == pytest.approx(x.mean(), rel=1e-12)
    assert moving_average(x, 1).tolist() == x.tolist()


# ------------------------------------------------------------- plane wall law
def test_plane_friction_is_bounded_by_mu_times_normal_force():
    r = 0.004
    pos = np.array([[0.0, 0.0, r - 1e-5]])      # pressed into the floor
    vel = np.zeros((1, 3))
    omega = np.zeros((1, 3))
    state = WallState()
    kn, kt, mu = 900.0, 450.0, 0.4
    for _ in range(600):                        # drag the wall under the grain
        force, torque, f_n = plane_forces(
            pos, vel, omega, state, normal=np.array([0.0, 0.0, 1.0]),
            point=np.zeros(3), surf_vel=np.array([0.5, 0.0, 0.0]), radius=r,
            kn=kn, kt=kt, kr=kt * r * r, mu=mu, mu_roll=0.0, gamma_wall=0.1,
            dt=1e-4)
    ft = math.hypot(force[0][0], force[0][1])
    fn = force[0][2]
    assert fn > 0.0
    assert ft <= mu * fn * 1.05                 # Coulomb cap respected
    assert ft > 0.5 * mu * fn                   # and actually mobilised


# --------------------------------------------------------------- uniaxial bond
def test_bond_breaks_in_tension_above_sigma_bond():
    r = 0.004
    area = math.pi * r ** 2
    params = ContactParams(kn=900.0, kt=450.0, mu=0.4, restitution=0.3,
                           sigma_bond=6.0e3)
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0 * r + 1e-4]])
    vel = np.zeros_like(pos)
    omega = np.zeros_like(pos)
    bonds = make_bonds(pos, r, params, scatter=0.0)
    assert bonds.n_intact == 1
    # small separation -> tension below the strength -> intact
    bond_forces(pos, vel, omega, bonds, params, r, dt=1e-4)
    assert bonds.n_intact == 1
    # large separation -> tensile failure
    pos[1, 2] += 5.0 * params.sigma_bond * area / params.kn
    bond_forces(pos, vel, omega, bonds, params, r, dt=1e-4)
    assert bonds.n_intact == 0


def test_bond_scatter_keeps_the_mean_strength():
    r = 0.004
    n = 200
    pos = np.zeros((n, 3))
    pos[:, 0] = np.arange(n) * 2.0 * r
    pos[1:, 2] = 0.0
    params = ContactParams(sigma_bond=1.0e4)
    bonds = make_bonds(pos, r, params, scatter=0.35, seed=1)
    expected = params.sigma_bond * math.pi * r ** 2
    assert bonds.tensile_strength.mean() == pytest.approx(expected, rel=0.1)
    assert bonds.tensile_strength.std() > 0.0


# ------------------------------------------------------------- benchmark drivers
def _tiny_repose(seed: int = 5) -> ReposeCase:
    return ReposeCase(material=MAT_REPOSE, block_width=7, block_layers=4,
                      settle_steps=1200, seed=seed)


def test_repose_is_deterministic_and_settles():
    a = run_repose(_tiny_repose(), PARAMS)
    b = run_repose(_tiny_repose(), PARAMS)
    assert a["digest"] == b["digest"]
    assert a["theta_deg"] > 0.0
    assert a["ke_ratio"] < 0.5
    assert 0.0 <= a["phi_solid"] <= 1.0


def test_repose_angle_grows_with_friction():
    low = run_repose(_tiny_repose(), ContactParams(kn=900, kt=450, mu=0.20,
                                                   restitution=0.3))
    high = run_repose(_tiny_repose(), ContactParams(kn=900, kt=450, mu=0.65,
                                                    restitution=0.3))
    assert high["theta_deg"] > low["theta_deg"] + 2.0


def _tiny_shear(seed: int = 7) -> ShearCase:
    return ShearCase(material=MAT, box_width=0.032, box_depth=0.024,
                     fill_height=0.016, shear_strain_target=0.04,
                     shear_velocity=0.03, settle_steps=40, consol_steps=40,
                     consolidation_strain=0.004, seed=seed)


def test_shear_curve_is_well_formed():
    out = run_shear(_tiny_shear(), PARAMS)
    assert out["tau"].shape == out["strain"].shape == out["tau_smooth"].shape
    assert out["tau"].size >= 8
    assert np.all(np.isfinite(out["tau"]))
    assert np.all(out["tau"] >= 0.0)
    assert out["peak_tau"] >= out["residual_tau"] * 0.5
    assert out["escaped"] is False
    assert out["sigma_n_measured"] > 0.0


def test_shear_is_deterministic():
    a = run_shear(_tiny_shear(), PARAMS)
    b = run_shear(_tiny_shear(), PARAMS)
    assert a["digest"] == b["digest"]
    assert np.allclose(a["tau"], b["tau"])


def _tiny_uniaxial(seed: int = 3, sigma_bond: float = 6.0e3) -> tuple:
    case = UniaxialCase(material=MAT, specimen_radius=0.012, fill_height=0.024,
                        strain_target=0.03, loading_velocity=0.03,
                        settle_steps=40, seed=seed)
    return case, ContactParams(kn=900.0, kt=450.0, mu=0.45, restitution=0.3,
                               sigma_bond=sigma_bond)


def test_uniaxial_reports_a_peak_and_breaks_bonds():
    case, params = _tiny_uniaxial()
    out = run_uniaxial(case, params)
    assert out["stress"].shape == out["strain"].shape
    assert out["peak_stress"] > 0.0
    assert out["peak_strain"] > 0.0
    assert out["n_bonds_initial"] > 0
    assert out["intact_fraction"] <= 1.0


def test_uniaxial_peak_stress_scales_with_bond_strength():
    low = run_uniaxial(*_tiny_uniaxial(sigma_bond=3.0e3))
    high = run_uniaxial(*_tiny_uniaxial(sigma_bond=1.2e4))
    assert high["peak_stress"] > 1.5 * low["peak_stress"]


# ------------------------------------------------------- features and objective
def _fake_raw(theta: float = 25.0, tau_scale: float = 1.0, peak: float = 3000.0) -> dict:
    strain = np.linspace(0.0, 0.12, 40)
    tau = tau_scale * 1000.0 * (1.0 - np.exp(-strain / 0.03))
    sig = np.array([0.005, 0.010, 0.015, 0.020, 0.030, 0.050])
    return {
        "repose": {"theta_deg": theta, "phi_solid": 0.55},
        "shear": {"strain": strain, "tau": tau, "tau_smooth": tau,
                  "sigma_n_measured": 2000.0, "peak_tau": float(tau.max()),
                  "residual_tau": float(tau[-1])},
        "uniaxial": {"strain": sig, "stress": peak * sig / sig.max(),
                     "stress_smooth": peak * sig / sig.max(),
                     "peak_stress": peak, "modulus": peak / sig.max()},
    }


def test_feature_vector_is_fixed_length_and_positive():
    feats = jc.feature_vector(_fake_raw())
    expected = 2 + len(jc.SHEAR_STRAIN_POINTS) + 1 + len(jc.UNIAXIAL_STRAIN_POINTS) + 2
    assert len(feats) == expected
    assert feats["repose.theta"] == pytest.approx(25.0)
    assert feats["uniaxial.peak"] == pytest.approx(3000.0)
    assert 0.0 <= feats["shear.tau@0.050"] <= 1.0


def test_objective_is_zero_at_the_target_and_grouped():
    feats = jc.feature_vector(_fake_raw())
    zero = jc.objective_from_features(feats, feats)
    assert zero["total"] == pytest.approx(0.0, abs=1e-15)
    worse = jc.objective_from_features(jc.feature_vector(_fake_raw(theta=32.0)), feats)
    assert worse["total"] > 0.0
    assert set(worse["parts"]) == set(jc.BENCHMARK_WEIGHTS)
    assert worse["parts"]["repose"] > 0.0
    assert worse["parts"]["shear"] == pytest.approx(0.0, abs=1e-15)


def test_objective_weights_normalise_to_one():
    assert sum(jc.BENCHMARK_WEIGHTS.values()) == pytest.approx(1.0)


# ---------------------------------------------------- posterior predictive check
def test_posterior_predictive_coverage_flags_uncovered_targets():
    rng = np.random.default_rng(0)
    space = ParameterSpace(names=("a",), lower=(0.0,), upper=(1.0,))
    x = rng.random((30, 1))
    y = 2.0 * x[:, 0]
    responses = {"f": y}
    samples = rng.random((200, 1))
    covered = jc.posterior_predictive_coverage(x, responses, {"f": 1.0},
                                              {"f": 0.5}, samples)
    missed = jc.posterior_predictive_coverage(x, responses, {"f": 1000.0},
                                             {"f": 0.1}, samples)
    assert 0.0 <= covered["coverage"] <= 1.0
    assert covered["coverage"] == 1.0        # target inside the predictive band
    assert missed["coverage"] == 0.0         # target far outside -> not covered
    assert covered["n_features"] == 1 and space.dim == 1


# ------------------------------------------------------------- campaign guards
def test_thresholds_match_the_node_card():
    assert jc.THRESHOLDS["repose_error_deg"] == 3.0
    assert jc.THRESHOLDS["shear_rmse_rel"] == 0.05
    assert jc.THRESHOLDS["peak_stress_rel"] == 0.05
    assert jc.THRESHOLDS["objective_drop"] == 0.90
    assert jc.THRESHOLDS["coverage"] == 0.90
    assert jc.THRESHOLDS["max_evaluations"] == 50


def test_calibration_refuses_to_exceed_the_evaluation_budget():
    with pytest.raises(ValueError):
        jc.calibrate_material("sand", n_init=30, n_iter=30)


def test_validation_spec_uses_independent_seeds():
    spec = jc.ExperimentSpec(name="sand")
    val = spec.with_validation_seeds(100)
    assert val.repose.seed == spec.repose.seed + 100
    assert val.shear.seed == spec.shear.seed + 100
    assert val.uniaxial.seed == spec.uniaxial.seed + 100
    assert val.repose.settle_steps == spec.repose.settle_steps