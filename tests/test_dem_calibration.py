"""Tests for the DEM contact-parameter auto-calibration package.

Two layers:

* **machinery** -- the estimator, the forward model's determinism, the GP
  surrogate, BayesOpt and the posterior sampler.  The calibration *loop* is
  exercised against an analytic test double (``toy_forward``) so that the unit
  tests stay seconds long and do not depend on cluster capacity; the test double
  is labelled as such everywhere and is never used to claim a physical result.
* **acceptance** -- the campaign criteria (a)-(e) are checked against the JSON
  record produced by the production DEM run
  (``artifacts/dem_auto_calibration/*.json``).  Numbers are read from that file
  only; no hand-copied value appears in this file.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
import pytest

from tensordem.dem_calibration.heap import (
    HeapCase,
    HeapError,
    build_cone,
    build_pile,
    hash_positions,
    repose_angle,
    run_heap,
    run_pluviation,
)
from tensordem.dem_calibration.inverse import (
    ParameterSpace,
    bayesian_optimize,
    posterior_sample,
)
from tensordem.dem_calibration.materials import (
    MATERIALS,
    OBSERVATION_SIGMA,
    RESPONSE_WEIGHTS,
    case_for,
    spec_by_name,
)
from tensordem.dem_calibration.pipeline import (
    THRESHOLDS,
    ForwardModel,
    calibrate_material,
    calibration_verdicts,
    objective_function,
)
from tensordem.dem_calibration.surrogate import (
    GaussianProcess,
    expected_improvement,
    latin_hypercube,
)

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "artifacts", "dem_auto_calibration")


# ---------------------------------------------------------------------------
# Analytic test double (NOT physics): a smooth surrogate of the DEM response
# used to exercise the inverse machinery in seconds instead of minutes.
# ---------------------------------------------------------------------------

def toy_forward(case: HeapCase) -> dict:
    mu, e, sr = float(case.mu), float(case.restitution), float(case.stiffness_ratio)
    theta = 6.0 + 34.0 * mu - 6.0 * e + 1.4 * (sr - 2.0) + 1.1 * mu * sr
    phi = 0.60 - 0.08 * mu + 0.05 * e - 0.012 * (sr - 2.0)
    digest = hashlib.sha256(
        f"{mu:.12f}|{e:.12f}|{sr:.12f}|{case.seed}".encode()).hexdigest()
    return {"theta_r_deg": theta, "phi_solid": phi, "digest": digest,
            "r2": 0.90, "ke_ratio": 1e-3}


def _small_case(**kwargs) -> HeapCase:
    """Small but protocol-identical heap case for the heavy tests (seconds, 1 core)."""
    base = dict(radius=0.005, kn=800.0, mu_rolling=0.10, shape="pluviation",
                n_grains=120, layer_size=12, settle_per_layer=150,
                inject_radius_factor=7.0, drop_height_factor=1.5, seed=4242)
    base.update(kwargs)
    return HeapCase(**base)


# ===========================================================================
# 1-2. Repose estimator against a synthetic cone
# ===========================================================================

def _cone_positions(angle_deg: float, radius: float = 0.06, r: float = 0.004,
                    n: int = 40000, seed: int = 3) -> np.ndarray:
    """Deterministic random volume filling of a cone of known apex slope.

    The estimator is validated against *this* geometry because it samples the
    free surface continuously; a lattice-filled cone is a staircase with 3-4
    steps at these sizes and cannot resolve an angle to better than ~10 deg
    (measured), which says nothing about the estimator.
    """
    rng = np.random.default_rng(seed)
    slope = math.tan(math.radians(angle_deg))
    rad = radius * np.sqrt(rng.random(n))
    ang = 2.0 * math.pi * rng.random(n)
    z_surface = r + slope * (radius - rad)
    z = r + (z_surface - r) * rng.random(n)
    return np.column_stack([rad * np.cos(ang), rad * np.sin(ang), z])


def test_repose_estimator_recovers_analytic_cone():
    """Analytic cone anchor, including the known max-statistics bias.

    The sector-maximum estimator is biased low by construction: the highest
    grain of an annulus of width ``r`` sits at the annulus' inner edge, so the
    fitted slope is ~5 % flatter than the true cone.  The bias is a property of
    the *estimator*, identical for the twin-experiment target and for every
    prediction, so it cancels in the calibration; it is pinned here as a
    regression guard instead of being hidden.
    """
    for angle in (20.0, 30.0, 40.0):
        est = repose_angle(_cone_positions(angle), 0.004)
        assert abs(est["theta_r_deg"] - angle) < 2.5, (angle, est)
        ratio = math.tan(math.radians(est["theta_r_deg"])) / math.tan(math.radians(angle))
        assert 0.90 <= ratio <= 1.02, (angle, ratio)
        assert est["r2"] > 0.98


def test_repose_estimator_robust_to_scattered_tail():
    """A handful of grains ejected to 2-3x the pile radius must not flatten the fit.

    Observed on the real deposits: with the outermost grain defining the span
    the reported angle collapsed from ~39 deg to ~1 deg, because the flank
    window then covers the scattered tail instead of the pile.
    """
    pos = _cone_positions(35.0, radius=0.05)
    rng = np.random.default_rng(11)
    tail = np.column_stack([
        rng.uniform(0.10, 0.15, 12),
        rng.uniform(-0.02, 0.02, 12),
        rng.uniform(0.004, 0.02, 12),
    ])
    est_clean = repose_angle(pos, 0.004)
    est_tail = repose_angle(np.vstack([pos, tail]), 0.004)
    assert abs(est_clean["theta_r_deg"] - 35.0) < 2.5, est_clean
    assert abs(est_tail["theta_r_deg"] - est_clean["theta_r_deg"]) < 3.0, (est_clean, est_tail)


# ===========================================================================
# 3-5. Forward model: packing, validation, timestep
# ===========================================================================

def test_build_pile_deterministic_and_seeded():
    case = _small_case(shape="column", cylinder_radius=0.030, fill_height=0.060)
    a = build_pile(case)
    b = build_pile(case)
    c = build_pile(_small_case(shape="column", cylinder_radius=0.030,
                               fill_height=0.060, seed=999))
    assert np.array_equal(a, b)
    assert a.shape == c.shape and not np.array_equal(a, c)
    # no initial overlap: nearest-neighbour distance must exceed 2r
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(a).query(a, k=2)
    nn = dist[:, 1]
    # the lattice gap (3 %) dominates the seeded jitter (2 % x r), so residual
    # overlaps stay below 2 % of a diameter and cannot inject energy
    assert nn.min() >= 2.0 * case.radius * 0.98
    assert abs(nn.mean() - 2.0 * case.radius * (1.0 + case.packing_gap)) < 0.05 * case.radius
    # the column fits inside its container
    assert np.hypot(a[:, 0], a[:, 1]).max() < case.wall_radius


@pytest.mark.parametrize("field,value", [
    ("restitution", 1.2), ("mu", -0.1), ("stiffness_ratio", 5.0),
    ("jitter", 0.7), ("contact_mode", "nonsense"),
])
def test_heap_case_rejects_invalid_parameters(field, value):
    kwargs = {"mu": 0.4, "restitution": 0.3, "stiffness_ratio": 2.0, field: value}
    with pytest.raises(HeapError):
        HeapCase(**kwargs)


def test_heap_timestep_resolves_contact_period():
    """dt must resolve the linear contact period by 20 steps (stability rule)."""
    for radius in (0.003, 0.005):
        case = _small_case(radius=radius)
        period = 2.0 * math.pi * math.sqrt(0.5 * case.mass / case.kn)
        assert case.timestep == pytest.approx(period / 20.0, rel=1e-12)


# ===========================================================================
# 6-9. Surrogate
# ===========================================================================

def test_gp_interpolates_smooth_function():
    rng = np.random.default_rng(0)
    x = rng.random((25, 2))
    y = np.sin(3 * x[:, 0]) + np.cos(2 * x[:, 1])
    gp = GaussianProcess().fit(x, y, seed=0)
    xs = rng.random((20, 2))
    truth = np.sin(3 * xs[:, 0]) + np.cos(2 * xs[:, 1])
    mean, std = gp.predict(xs)
    assert np.max(np.abs(mean - truth)) < 0.3
    assert np.all(std > 0.0)


def test_gp_uncertainty_shrinks_near_data():
    rng = np.random.default_rng(1)
    x = rng.random((12, 1))
    y = x[:, 0] ** 2
    gp = GaussianProcess().fit(x, y, seed=0)
    near = gp.predict(np.array([[0.5]]))[1][0]
    far = gp.predict(np.array([[5.0]]))[1][0]
    assert far > near


def test_latin_hypercube_stratified():
    n, d = 12, 3
    x = latin_hypercube(n, d, seed=5)
    assert x.shape == (n, d)
    assert x.min() >= 0.0 and x.max() <= 1.0
    for j in range(d):
        bins = np.floor(x[:, j] * n).astype(int)
        assert sorted(bins.tolist()) == list(range(n))
    assert np.array_equal(x, latin_hypercube(n, d, seed=5))


def test_expected_improvement_positive_near_incumbent():
    mean = np.array([0.0, 0.5, 2.0])
    std = np.array([0.1, 0.1, 0.1])
    ei = expected_improvement(mean, std, best=0.6)
    assert ei[1] > ei[2]                      # near the incumbent beats far
    assert np.all(ei >= 0.0)


# ===========================================================================
# 10-14. BayesOpt, objective, posterior, pipeline (analytic test double)
# ===========================================================================

def test_bayesian_optimize_finds_analytic_minimum():
    space = ParameterSpace()
    target = np.array([0.3, 0.7, 0.5])

    def objective(params):
        u = space.to_unit([params[n] for n in space.names])
        return float(np.sum((u - target) ** 2))

    result = bayesian_optimize(objective, space, n_init=8, n_iter=20, seed=0,
                               y_stop=1e-4)
    assert result["best_objective"] < 5e-3
    u_best = space.to_unit([result["best_params"][n] for n in space.names])
    assert np.max(np.abs(u_best - target)) < 0.12
    assert result["n_evaluations"] == 28


def test_objective_function_is_relative_and_weighted():
    targets = {"theta_r_deg": 30.0, "phi_solid": 0.50}
    sim = {"theta_r_deg": 33.0, "phi_solid": 0.525}
    expected = (0.8 * 3.0 / 30.0) + (0.2 * 0.025 / 0.50)
    assert objective_function(sim, targets) == pytest.approx(expected)
    assert objective_function(None, targets) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        objective_function(sim, {"theta_r_deg": 0.0})


def test_posterior_covers_truth_on_toy():
    space = ParameterSpace()
    spec = spec_by_name("sand")
    forward = ForwardModel(case_for(spec, "calibration", spec.truth), forward=toy_forward)
    design = latin_hypercube(40, space.dim, seed=2)
    rows = []
    for u in design:
        params = space.to_physical(u)
        rows.append((u, forward.responses(params)))
    x_unit = np.array([r[0] for r in rows])
    targets = forward.responses(spec.truth)
    responses = {name: np.array([r[1][name] for r in rows]) for name in targets
                 if name in RESPONSE_WEIGHTS}
    post = posterior_sample(x_unit, responses,
                            {k: targets[k] for k in responses}, OBSERVATION_SIGMA,
                            space, n_samples=500, seed=1)
    for name, value in spec.truth.items():
        assert post["summary"][name]["lo95"] <= value <= post["summary"][name]["hi95"]


def test_calibration_pipeline_on_analytic_double():
    spec = spec_by_name("gravel")
    record = calibrate_material(spec, n_init=6, n_iter=12, seed=4,
                                forward=toy_forward, n_posterior=400)
    verdicts = record["verdicts"]
    assert verdicts["angle_within_5deg"]
    assert verdicts["generalisation_within_10pct"]
    assert verdicts["reproducible"]
    assert record["search"]["n_evaluations"] <= THRESHOLDS["max_iterations"]
    json.dumps(record)                        # the record must be serialisable


def test_calibration_is_reproducible_on_analytic_double():
    spec = spec_by_name("glass_beads")
    a = calibrate_material(spec, n_init=4, n_iter=4, seed=7, forward=toy_forward,
                           n_posterior=200)
    b = calibrate_material(spec, n_init=4, n_iter=4, seed=7, forward=toy_forward,
                           n_posterior=200)
    assert a["search"]["best_params"] == b["search"]["best_params"]
    assert a["search"]["best_objective"] == b["search"]["best_objective"]
    assert a["posterior"]["summary"] == b["posterior"]["summary"]


# ===========================================================================
# 15-16. Real DEM forward model (small case, seconds)
# ===========================================================================

@pytest.fixture(scope="module")
def small_dem_run():
    case = _small_case(mu=0.5, restitution=0.3, stiffness_ratio=2.0)
    return run_pluviation(case)


def test_forward_deterministic_digest(small_dem_run):
    """Identical case -> bit-identical final configuration (criterion (e))."""
    case = _small_case(mu=0.5, restitution=0.3, stiffness_ratio=2.0)
    again = run_pluviation(case)
    assert again["digest"] == small_dem_run["digest"]
    assert again["theta_r_deg"] == small_dem_run["theta_r_deg"]
    assert hash_positions(build_pile(case)) == hash_positions(build_pile(case))


def test_forward_angle_physical_range(small_dem_run):
    out = small_dem_run
    if "theta_r_deg" not in out:
        pytest.skip("no measurable heap in the small configuration")
    assert 0.0 <= out["theta_r_deg"] < 60.0
    assert out["n_particles"] > 50
    assert 0.0 < out["phi_solid"] < 0.7
    # the heap must be at rest: residual kinetic energy well below the
    # released potential energy
    assert out["ke_ratio"] < 0.25


# ===========================================================================
# 17-21. Acceptance criteria on the production artifact
# ===========================================================================

def _load_records():
    if not os.path.isdir(ARTIFACT_DIR):
        pytest.skip(f"production artifact missing: {ARTIFACT_DIR}")
    records = []
    for spec in MATERIALS:
        path = os.path.join(ARTIFACT_DIR, f"{spec.name}.json")
        if not os.path.exists(path):
            pytest.skip(f"missing per-material record {path}")
        with open(path, "r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records


def test_calibration_accuracy():
    """(a) calibrated parameters reproduce the repose angle within 5 deg."""
    records = _load_records()
    for record in records:
        err = record["verification"]["angle_error_deg"]
        assert err < THRESHOLDS["angle_error_deg"], (record["material"], err)


def test_calibration_convergence():
    """(b) objective < 0.05 within at most 50 forward evaluations."""
    records = _load_records()
    for record in records:
        search = record["search"]
        assert search["n_evaluations"] <= THRESHOLDS["max_iterations"], record["material"]
        assert search["best_objective"] < THRESHOLDS["objective"], (
            record["material"], search["best_objective"])


def test_generalization():
    """(c) independent-condition prediction within 10 % relative error."""
    records = _load_records()
    for record in records:
        rel = record["generalisation"]["relative_error"]
        assert rel < THRESHOLDS["generalisation_rel"], (record["material"], rel)


def test_uncertainty_quantification():
    """(d) the 95 % credible interval contains the twin-experiment truth."""
    records = _load_records()
    for record in records:
        summary = record["posterior"]["summary"]
        for name, value in record["truth"].items():
            lo, hi = summary[name]["lo95"], summary[name]["hi95"]
            assert lo <= value <= hi, (record["material"], name, lo, value, hi)


def test_reproducibility():
    """(e) the production run is bit-identical and flags it explicitly."""
    records = _load_records()
    for record in records:
        assert record["reproducible"] is True, record["material"]
        assert record["verification"]["responses"]["digest"]
        assert record["posterior"]["acceptance_rate"] > 0.0