"""Tests for the granular parameter-identifiability face.

Fast tests cover the machinery (packing, budget, analytic Jacobian/FIM,
inversion, propagation) and the two single-grain observables, which cost
milliseconds.  The bulk observable and the full acceptance study are marked
``slow`` because they integrate a granular bed for several thousand steps.
"""

from __future__ import annotations

import dataclasses
import glob
import math
import os
import re

import numpy as np
import pytest

from tensordem.granular_identifiability import (
    PARAMETER_SPECS,
    BudgetExceeded,
    CaseBudget,
    ForwardMap,
    GranularSystemSpec,
    TEST_GRANULAR_01,
    TARGET_NAMES,
    bed_observables,
    central_jacobian,
    conditioning,
    drop_observables,
    evaluate_targets,
    grain_observables,
    hex_pack,
    invert_parameters,
    params_from_unit,
    propagation_linearity,
    restitution_to_zeta,
    target_sigma,
    unit_from_params,
    zeta_to_restitution,
)
from tensordem.granular_identifiability.inversion import recovery
from tensordem.granular_identifiability.study import REFERENCE_PARAMS

PACKAGE_DIR = os.path.dirname(
    __import__("tensordem.granular_identifiability", fromlist=["__file__"]).__file__
)

#: material/application vocabulary that must never appear in this package: the
#: face is defined on a synthetic granular system only
FORBIDDEN_TOKENS = (
    "ice",
    "floe",
    "hull",
    "ship",
    "ridge",
    "arctic",
    "glacier",
    "sea_",
    "seawater",
    "propeller",
    "bergy",
)


def _fast_spec(**kwargs) -> GranularSystemSpec:
    """Cheap variant of the case (used by the end-to-end test only)."""
    base = dict(bed_settle=0.02, bed_fill_height=0.10, drop_height=0.02,
                tilt_bisections=2)
    base.update(kwargs)
    return dataclasses.replace(TEST_GRANULAR_01, **base)


# --------------------------------------------------------------------------
# domain neutrality
# --------------------------------------------------------------------------
def test_package_has_no_application_semantics():
    """The identifiability face must stay domain neutral (synthetic case only)."""
    hits = []
    for path in glob.glob(os.path.join(PACKAGE_DIR, "*.py")):
        text = open(path).read().lower()
        for token in FORBIDDEN_TOKENS:
            if re.search(rf"\b{re.escape(token)}", text):
                hits.append((os.path.basename(path), token))
    assert not hits, f"application vocabulary leaked into the synthetic face: {hits}"


def test_case_is_the_synthetic_system():
    assert TEST_GRANULAR_01.name == "test_granular_01"
    assert TEST_GRANULAR_01.seed == 20261110


# --------------------------------------------------------------------------
# system construction
# --------------------------------------------------------------------------
def test_hex_pack_is_deterministic_and_non_overlapping():
    spec = TEST_GRANULAR_01
    pos1, rad1 = hex_pack(spec, 0.0, 0.2, 0.0, 0.2, seed_offset=0)
    pos2, rad2 = hex_pack(spec, 0.0, 0.2, 0.0, 0.2, seed_offset=0)
    assert np.array_equal(pos1, pos2) and np.array_equal(rad1, rad2)
    assert np.all(rad1 > 0.0)
    # nearest-neighbour distance never below the sum of radii by more than the
    # 1 % jitter allowance
    d = np.linalg.norm(pos1[:, None, :] - pos1[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    overlap = rad1[:, None] + rad1[None, :] - d
    assert overlap.max() < 0.03 * spec.radius


def test_hex_pack_rejects_degenerate_region():
    with pytest.raises(ValueError):
        hex_pack(TEST_GRANULAR_01, 0.0, 0.001, 0.0, 0.001, seed_offset=0)


def test_spec_dt_divisor_keeps_physics():
    refined = TEST_GRANULAR_01.with_dt_divisor(4)
    assert refined.dt == pytest.approx(TEST_GRANULAR_01.dt / 4.0)
    assert refined.radius == TEST_GRANULAR_01.radius
    with pytest.raises(ValueError):
        TEST_GRANULAR_01.with_dt_divisor(0)


# --------------------------------------------------------------------------
# budget discipline
# --------------------------------------------------------------------------
def test_budget_rejects_step_overrun():
    spec = _fast_spec()
    with pytest.raises(BudgetExceeded):
        evaluate_targets(
            REFERENCE_PARAMS, spec=spec, budget=CaseBudget(max_steps=10)
        )


def test_budget_rejects_particle_overrun():
    budget = CaseBudget(max_particles=2).start()
    with pytest.raises(BudgetExceeded):
        budget.charge_particles(3)


def test_budget_report_fields():
    report = CaseBudget(max_particles=5, max_steps=5, max_wall_seconds=5.0).report()
    assert set(report) == {
        "max_particles", "max_steps", "max_wall_seconds",
        "particles", "steps", "wall_seconds",
    }


# --------------------------------------------------------------------------
# parameter <-> unit cube
# --------------------------------------------------------------------------
def test_unit_cube_roundtrip():
    u = unit_from_params(REFERENCE_PARAMS)
    assert np.all((u >= 0.0) & (u <= 1.0))
    back = params_from_unit(u)
    for name in REFERENCE_PARAMS:
        assert back[name] == pytest.approx(REFERENCE_PARAMS[name])
    for name, lo, hi, _, _ in PARAMETER_SPECS:
        assert lo <= REFERENCE_PARAMS[name] <= hi


def test_unit_cube_rejects_out_of_box():
    with pytest.raises(ValueError):
        params_from_unit([1.5, 0.5, 0.5])


def test_restitution_damping_roundtrip():
    for e in (0.05, 0.2, 0.45, 0.8, 0.99):
        assert zeta_to_restitution(restitution_to_zeta(e)) == pytest.approx(e, rel=1e-12)
    with pytest.raises(ValueError):
        restitution_to_zeta(1.5)
    with pytest.raises(ValueError):
        zeta_to_restitution(1.0)


# --------------------------------------------------------------------------
# single-grain observables (cheap)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mu", [0.2, 0.35, 0.55, 0.8])
def test_grain_friction_ratio_measures_mu(mu):
    result = grain_observables({**REFERENCE_PARAMS, "friction": mu})
    assert result["grain_friction_ratio"] == pytest.approx(mu, rel=2e-3)
    assert result["normal_load"] > 0.0
    assert result["slip_samples"] > 10


def test_grain_friction_ratio_independent_of_other_parameters():
    base = grain_observables({**REFERENCE_PARAMS})
    other = grain_observables({**REFERENCE_PARAMS, "restitution": 0.8, "stiffness": 1.4e4})
    assert other["grain_friction_ratio"] == pytest.approx(
        base["grain_friction_ratio"], rel=1e-6
    )


def test_drop_observables_are_reproducible_and_physical():
    a = drop_observables(REFERENCE_PARAMS)
    b = drop_observables(REFERENCE_PARAMS)
    assert a["restitution"] == b["restitution"]
    assert a["impact_duration_s"] == b["impact_duration_s"]
    assert 0.0 < a["restitution"] < 1.0
    assert a["impact_duration_s"] > 0.0
    assert a["impact_speed"] == pytest.approx(
        math.sqrt(2.0 * TEST_GRANULAR_01.gravity * TEST_GRANULAR_01.drop_height),
        rel=0.02,
    )


def test_impact_duration_decreases_with_stiffness():
    soft = drop_observables({**REFERENCE_PARAMS, "stiffness": 7.0e3})
    stiff = drop_observables({**REFERENCE_PARAMS, "stiffness": 1.3e4})
    assert stiff["impact_duration_s"] < soft["impact_duration_s"]
    assert soft["impact_duration_s"] / stiff["impact_duration_s"] == pytest.approx(
        math.sqrt(1.3e4 / 7.0e3), rel=0.05
    )


def test_restitution_increases_with_target_restitution():
    low = drop_observables({**REFERENCE_PARAMS, "restitution": 0.25})
    high = drop_observables({**REFERENCE_PARAMS, "restitution": 0.70})
    assert high["restitution"] > low["restitution"]


# --------------------------------------------------------------------------
# identifiability machinery on an analytic forward model
# --------------------------------------------------------------------------
#: deliberately well-conditioned analytic forward operator (4 observables, 3 params)
_ANALYTIC_A = np.asarray(
    [[1.0, 0.2, 0.0], [0.0, 1.0, 0.3], [0.1, 0.0, 1.0], [0.05, 0.1, 0.9]]
)


def _analytic_model(u):
    """``y = A u``: a stand-in for the DEM map so the UQ machinery is testable."""
    return _ANALYTIC_A @ np.asarray(u, dtype=float)


def test_central_jacobian_recovers_linear_operator():
    J = central_jacobian(_analytic_model, [0.5, 0.5, 0.5], h=1e-3)
    assert np.allclose(J, _ANALYTIC_A, atol=1e-6)


def test_central_jacobian_mirrors_at_the_box_faces():
    J = central_jacobian(_analytic_model, [1.0, 0.0, 0.5], h=0.05)
    assert np.all(np.isfinite(J))


def test_conditioning_reports_fim_spectrum_and_correlation():
    J = central_jacobian(_analytic_model, [0.5, 0.5, 0.5], h=1e-3)
    out = conditioning(J, np.asarray([0.1, 0.1, 0.1, 0.1]))
    assert out["condition_number"] == pytest.approx(
        np.linalg.cond(out["fim"]), rel=1e-9
    )
    assert len(out["eigenvalues"]) == 3
    corr = np.asarray(out["correlation"])
    assert np.allclose(np.diag(corr), 1.0)
    assert np.all(np.abs(corr) <= 1.0 + 1e-12)


def test_conditioning_rejects_bad_sigma():
    J = central_jacobian(_analytic_model, [0.5, 0.5, 0.5], h=1e-3)
    with pytest.raises(ValueError):
        conditioning(J, np.asarray([0.1, 0.0, 0.1, 0.1]))


def test_inversion_recovers_analytic_parameters():
    """Inversion recovers the truth from synthetic data -- correctly synthetic.

    An identifiability test *must* generate its observations from a model with a
    known parameter vector; the point is the inversion, not physical agreement.
    Declared in-band so tools/self_verification_scan.py keeps it visible: this
    test carries no evidence grade for any experimental claim.
    """
    truth = np.asarray([0.30, 0.60, 0.75])
    sigma = np.asarray([0.01, 0.01, 0.01, 0.01])
    rng = np.random.default_rng(7)
    # self-verification: synthetic surrogate -- y_obs is model output + noise by design
    y_obs = _analytic_model(truth) + rng.normal(0.0, 1.0, 4) * sigma
    result = invert_parameters(_analytic_model, y_obs, sigma, [0.5, 0.5, 0.5])
    assert np.allclose(result["unit"], truth, atol=2e-2)
    assert result["chi2"] < 20.0


def test_recovery_reports_relative_errors():
    est = {"friction": 0.58, "restitution": 0.45, "stiffness": 1.05e4}
    out = recovery(est, REFERENCE_PARAMS)
    assert out["friction"]["rel_error"] == pytest.approx(0.03 / 0.55)
    assert out["max_rel_error"] == pytest.approx(0.03 / 0.55)
    assert out["stiffness"]["rel_error"] == pytest.approx(0.05)


def test_propagation_linearity_on_analytic_model():
    truth = {"friction": 0.55, "restitution": 0.45, "stiffness": 1.0e4}

    def model(u):
        return _analytic_model(u)

    out = propagation_linearity(model, truth, min_relative_response=1e-6)
    assert out["worst"]["r2"] is None or out["worst"]["r2"] > 0.95
    for name, row in out["parameters"].items():
        for target, entry in row["observables"].items():
            if entry["sensitive"]:
                assert entry["r2"] and entry["r2"] > 0.95


# --------------------------------------------------------------------------
# end to end on the synthetic case (slow)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_evaluate_targets_end_to_end():
    spec = _fast_spec()
    result = evaluate_targets(REFERENCE_PARAMS, spec=spec, detail=True)
    assert set(TARGET_NAMES) <= set(result["targets"])
    for name in TARGET_NAMES:
        value = result["targets"][name]
        assert np.isfinite(value), name
        assert target_sigma(name, value) > 0.0
    assert result["usage"]["steps"] > 0
    assert result["usage"]["particles"] > 0


@pytest.mark.slow
def test_bed_packing_fraction_is_deterministic():
    spec = _fast_spec()
    a = bed_observables(REFERENCE_PARAMS, spec=spec)
    b = bed_observables(REFERENCE_PARAMS, spec=spec)
    assert a["packing_fraction"] == b["packing_fraction"]
    assert 0.5 < a["packing_fraction"] < 0.95


@pytest.mark.slow
def test_forward_map_caches_evaluations():
    spec = _fast_spec()
    forward = ForwardMap(spec=spec, budget=CaseBudget(max_steps=20000))
    u = unit_from_params(REFERENCE_PARAMS)
    first = forward(u)
    second = forward(u)
    assert np.array_equal(first, second)
    assert forward.stats["calls"] == 1
    assert forward.stats["cache_hits"] == 1


@pytest.mark.slow
def test_study_smoke_produces_verdicts(tmp_path):
    from tensordem.granular_identifiability.study import run_study

    report = run_study(
        out_dir=str(tmp_path),
        spec=_fast_spec(),
        workers=2,
        dt_divisors=(1,),
        starts=[{"name": "centre", "unit": [0.5, 0.5, 0.5]}],
        max_nfev=12,
        write_figures=False,
    )
    assert set(report["verdicts"]) == {
        "a_discretisation_independence",
        "b_recovery",
        "c_fim_condition_number",
        "d_uncertainty_propagation",
        "e_zero_domain_semantics",
        "f_case_budget",
    }
    assert os.path.exists(os.path.join(str(tmp_path), "study.json"))
    # the case cost must stay inside the declared budget
    assert report["verdicts"]["f_case_budget"]["pass"]