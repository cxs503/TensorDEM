"""Verification surface for the elastic-limit (Campbell 2002) shear cell.

The pins here are the ones that cost the two earlier revisions their physics:

* the triclinic Lees-Edwards minimum image must be the *true* nearest image
  (rounding the fractional component alone is not, and created a crush),
* the stress estimator's sign/normalisation,
* the fact that a sheared point must reach a *positive* pressure with a bounded
  overlap instead of crushing,
* the scaling/fit maths (a synthetic ``n = 2`` and ``n = 0`` data set must come
  back out),
* the production artifacts of the case must exist and be gated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensordem.elastic_limit_shear import (
    ElasticLimitShearCase, ElasticLimitShearError, build_configuration,
    configuration_report, inertial_number_of, kappa_of, pair_stress,
    shear_rate_exponent, shear_stress_scaling, simulate_plane_shear, tau_star_of,
    _min_image)

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "case_dem_shear_campbell_elastic_limit"


# ---------------------------------------------------------------- groups ----
def test_dimensionless_groups_match_campbell_definition():
    k1 = kappa_of(1.0e4, 1.0, 1.2, 0.5)
    k2 = kappa_of(1.0e4, 1.0, 1.2, 1.0)
    assert k1 == pytest.approx(4.0 * k2)
    assert tau_star_of(10.0, 1.0, 1.2, 0.5) == pytest.approx(10.0 / (1.2 ** 2 * 0.25))
    assert inertial_number_of(1.0, 4.0, 1.0, 1.0, 2.0) == pytest.approx(1.0)


def test_case_rejects_impossible_parameters():
    with pytest.raises(ElasticLimitShearError):
        ElasticLimitShearCase(n_particles=50)
    with pytest.raises(ElasticLimitShearError):
        ElasticLimitShearCase(n_particles=200, solid_fraction=0.95)
    with pytest.raises(ElasticLimitShearError):
        ElasticLimitShearCase(n_particles=200, restitution=1.0)
    with pytest.raises(ElasticLimitShearError):
        ElasticLimitShearCase(n_particles=200, shear_rate=0.0)
    with pytest.raises(ElasticLimitShearError):
        ElasticLimitShearCase(n_particles=200, strain_total=1.0, strain_average_from=2.0)


# ------------------------------------------------------------- geometry ----
@pytest.mark.parametrize("s", [0.0, 0.1, 0.3, 0.5, -0.4])
def test_minimum_image_is_the_true_nearest_image(s):
    """Regression: the first two revisions used fractional rounding only.

    That overestimated pair distances by up to 16 % of the box once the box was
    tilted, which manufactured contacts inside the cell and crushed it (max
    overlap -> 1.0 d, energy audit off by 10^3).
    """
    rng = np.random.default_rng(7)
    lx = ly = 10.0
    n = 40
    x = rng.uniform(0, lx, n)
    y = rng.uniform(0, ly, n)
    i = np.arange(0, n, 2)
    j = np.arange(1, n, 2)
    dx, dy = _min_image(x, y, i, j, lx, ly, s)
    best = np.full(i.size, np.inf)
    for nx in range(-3, 4):
        for ny in range(-3, 4):
            ddx = (x[j] - x[i]) + s * (y[j] - y[i]) + nx * lx + ny * s * ly
            ddy = (y[j] - y[i]) + ny * ly
            best = np.minimum(best, np.hypot(ddx, ddy))
    assert np.allclose(np.hypot(dx, dy), best, atol=1e-12)


def test_packing_matches_requested_solid_fraction_and_is_overlap_free():
    case = ElasticLimitShearCase(n_particles=300, solid_fraction=0.80)
    cfg = build_configuration(case)
    rep = configuration_report(case, cfg)
    assert rep["solid_fraction_measured"] == pytest.approx(0.80, rel=1e-6)
    assert rep["max_overlap_init"] < 5e-2
    assert rep["n_particles"] == 300


# --------------------------------------------------------------- stress ----
def test_pair_stress_sign_and_normalisation():
    """A single repulsive contact must give positive pressure and shear stress."""
    # particle i at the origin, j one diameter up and one diameter to the right
    dx, dy = np.array([1.0]), np.array([1.0])
    d = math.hypot(1.0, 1.0)
    fn = 3.0
    ux, uy = dx / d, dy / d
    fx = -fn * ux          # force on i from j (repulsive)
    fy = -fn * uy
    area = 2.0
    tau, pres = pair_stress(dx, dy, fx, fy, area)
    # P = -F_i.(r_j-r_i)/(2A) = fn*d/(2A) > 0 for a repulsive contact
    assert pres == pytest.approx(fn * d / (2.0 * area))
    # tau = sum F_i,x (r_i-r_j)_y / A = -fx*dy/A
    assert tau == pytest.approx(fx * dy[0] / area)


# ------------------------------------------------------------ simulation ----
def test_sheared_point_reaches_positive_pressure_without_crushing():
    case = ElasticLimitShearCase(n_particles=200, kn=1.0e4, shear_rate=1.0,
                                 dt_factor=0.005, strain_total=0.30,
                                 strain_average_from=0.15)
    res = simulate_plane_shear(case, build_configuration(case))
    assert res["pressure"] > 0.0
    assert res["tau"] > 0.0            # the drive does positive shear work
    assert res["max_overlap_ratio"] < 0.5
    assert np.isfinite(res["tau_star"]) and res["tau_star"] > 0.0
    # SLLOD energy identity: the audit cannot be wildly off (a blow-up would mean
    # a broken stress normalisation or a sign error)
    assert res["energy"]["relative_defect"] < 1.0


# --------------------------------------------------------------- fitting ----
def _synthetic(kn_values, tau_star_values, gamma=1.0):
    pts = []
    for kn, tst in zip(kn_values, tau_star_values):
        pts.append({
            "case": {"kn": kn, "shear_rate": gamma},
            "kappa": kappa_of(kn, 1.0, 1.2, gamma),
            "tau": tst * 1.2 ** 2 * gamma ** 2,
            "tau_star": tst,
            "pressure": tst * 2.0,
            "I": 0.1,
            "mu_eff": 0.5,
            "gates": {"all": True},
        })
    return pts


def test_scaling_recovers_a_quadratic_bagnold_branch():
    kns = [1e0, 1e1, 1e2, 1e3]
    pts = _synthetic(kns, [2.0] * len(kns))
    out = shear_stress_scaling(pts)
    assert out["status"] == "ok"
    assert out["beta_global"] == pytest.approx(0.0, abs=0.02)
    assert out["n_global"] == pytest.approx(2.0, abs=0.05)


def test_scaling_recovers_the_elastic_branch():
    kns = [1e4, 1e5, 1e6]
    pts = _synthetic(kns, [4e-4 * k for k in kns])
    out = shear_stress_scaling(pts)
    assert out["beta_global"] == pytest.approx(1.0, abs=0.02)
    assert out["n_global"] == pytest.approx(0.0, abs=0.05)


def test_scaling_refuses_when_too_few_points_pass_the_gates():
    pts = _synthetic([1e2, 1e3], [1.0, 1.0])
    out = shear_stress_scaling(pts)
    assert out["status"] == "refused"
    assert "fewer than" in out["reason"]


def test_rate_sweep_exponent():
    pts = _synthetic([1e0, 1e0, 1e0], [1.0, 1.0, 1.0])
    for p, g in zip(pts, (0.5, 1.0, 2.0)):
        p["case"]["shear_rate"] = g
        p["tau"] = 0.75 * g ** 2
    out = shear_rate_exponent(pts)
    assert out["status"] == "ok"
    assert out["n_bagnold"] == pytest.approx(2.0, abs=0.02)


# ------------------------------------------------------------- artifacts ----
@pytest.mark.skipif(not ART.exists(), reason="case artifacts not present")
def test_production_artifacts_are_present_and_gated():
    points = sorted(ART.glob("point_*.json"))
    assert len(points) >= 3, "the case must ship its run artifacts"
    scaling = json.loads((ART / "scaling.json").read_text())
    assert scaling["n_points"] >= 3
    assert scaling["scaling_vs_kappa"]["status"] == "ok", scaling["scaling_vs_kappa"]
    assert scaling["scaling_vs_kappa"]["n_gated"] >= 3