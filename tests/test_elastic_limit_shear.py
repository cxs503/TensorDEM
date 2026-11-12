"""Verification surface for the elastic-limit (Campbell) shear module.

The module is a *diagnostic-grade* prototype: the physics gates it carries
(energy balance, overlap, steadiness) are part of the contract and are pinned
here, including the fact that the current integrator does **not** reach the
gated elastic-limit steady state (honest-FAIL, see
``docs/CASE_DEM_SHEAR_CAMPBELL_ELASTIC_LIMIT.md``).  A future repair that makes
the runs gated must flip ``test_current_integrator_is_rejected_by_gates``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.elastic_limit_shear import (
    ElasticLimitShearCase, ElasticLimitShearError, build_configuration,
    configuration_report, kappa_of, shear_stress_scaling, simulate_couette_shear,
    tau_star_of)


def test_dimensionless_groups_match_campbell_definition():
    # kappa = k_n / (rho d^3 gdot^2); doubling gdot quarters kappa
    k1 = kappa_of(1.0e4, 1.0, 1.2, 0.5)
    k2 = kappa_of(1.0e4, 1.0, 1.2, 1.0)
    assert k1 == pytest.approx(4.0 * k2)
    assert tau_star_of(10.0, 1.0, 1.2, 0.5) == pytest.approx(
        10.0 / (1.2 ** 2 * 0.25))


def test_case_rejects_impossible_parameters():
    with pytest.raises(ElasticLimitShearError):
        ElasticLimitShearCase(n_particles=15)
    with pytest.raises(ElasticLimitShearError):
        ElasticLimitShearCase(n_particles=20, solid_fraction=0.95)
    with pytest.raises(ElasticLimitShearError):
        ElasticLimitShearCase(n_particles=20, restitution=1.0)


def test_starting_configuration_is_overlap_free():
    case = ElasticLimitShearCase(n_particles=64, solid_fraction=0.78)
    cfg = build_configuration(case)
    rep = configuration_report(case, cfg["pos"])
    assert rep["max_initial_overlap_over_d"] <= 0.05
    assert rep["width"] > 0.0 and rep["gap"] > 0.0
    # every centre keeps its surface inside the wall faces
    r = case.radii
    assert np.all(cfg["pos"][:, 1] > case.wall_radius + r - 1e-6)
    assert np.all(cfg["pos"][:, 1] < case.gap + case.wall_radius - r + 1e-6)


def test_couette_point_is_deterministic_and_reports_finite_numbers():
    case = ElasticLimitShearCase(n_particles=64, solid_fraction=0.78, kn=1.0e3,
                                 total_strain=0.05, average_from=0.2,
                                 sample_every=20)
    cfg = build_configuration(case)
    a = simulate_couette_shear(case, config=cfg)
    b = simulate_couette_shear(case, config=cfg)
    assert a["tau"] == b["tau"] and a["pressure"] == b["pressure"]
    for key in ("tau", "pressure", "temperature", "tau_star", "kappa"):
        assert math.isfinite(a[key])
    assert a["kappa"] == pytest.approx(case.kappa)


def test_current_integrator_is_rejected_by_the_physics_gates():
    """Honest pin of the open defect: the gates must reject today's runs."""
    case = ElasticLimitShearCase(n_particles=64, solid_fraction=0.78, kn=1.0e3,
                                 total_strain=0.3, average_from=0.3,
                                 sample_every=25)
    res = simulate_couette_shear(case)
    assert res["gate_pass"] is False
    assert (res["gate"]["energy_balance_rel"] > case.gate_tol
            or res["gate"]["max_overlap_over_d"] > case.max_overlap)


def test_scaling_fit_reports_slopes_and_refuses_short_ladders():
    pts = [{"kappa": k, "tau_star": k / 1e2, "gate_pass": True}
           for k in (1e2, 1e3, 1e4, 1e5)]
    fit = shear_stress_scaling(pts)
    # tau* proportional to kappa -> beta = 1 -> n = 0 (elastic-quasistatic)
    assert fit["beta_global"] == pytest.approx(1.0, abs=0.05)
    assert fit["n_global"] == pytest.approx(0.0, abs=0.1)
    with pytest.raises(ElasticLimitShearError):
        shear_stress_scaling(pts[:2])
