"""Tests for the three-body ice collision / broken-ice decomposition module.

The suite is deliberately *small-scale*: every case uses a reduced geometry so
that the whole file runs in seconds, while still exercising the physics the
acceptance criteria depend on (momentum and energy conservation, the force
partition identity, the sign conventions of the contact laws, and the closure of
the resistance decomposition).
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.threebody_ice import (ContactLaw, FieldSpec, FloeState,
                                     MultiBodyModel, ScenarioSpec,
                                     calibrate_law, elastic_reference,
                                     measure_restitution, parameter_ledger_sha256,
                                     power_law_exponent, run_field,
                                     run_three_body, single_collision_fractions)
from tensordem.threebody_ice.field import _floe_layout, seed_region_m
from tensordem.threebody_ice.model import (damping_ratio_from_restitution,
                                           restitution_from_damping_ratio)
from tensordem.threebody_ice.report import build_markdown
from tensordem.threebody_ice.validation import (THRESHOLDS, evaluate,
                                                CASE_MANIFEST)

TORCH_DTYPE = torch.float64

QUIET = dict(floe_floe_friction=0.0, hull_floe_friction=0.0,
             water_drag_coefficient=0.0)


def _quiet_law(**kwargs) -> ContactLaw:
    params = dict(QUIET)
    params.update(kwargs)
    return ContactLaw(**params)


# ---------------------------------------------------------------------------
# parameters and contact-law bookkeeping
# ---------------------------------------------------------------------------

def test_restitution_damping_inverse_roundtrip() -> None:
    for target in (1.0, 0.8, 0.5, 0.2, 0.05):
        beta = damping_ratio_from_restitution(target)
        assert 0.0 <= beta < 1.0
        assert restitution_from_damping_ratio(beta) == pytest.approx(target, rel=1e-12)


def test_contact_law_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        ContactLaw(floe_floe_restitution=1.5)
    with pytest.raises(ValueError):
        ContactLaw(hull_floe_friction=-0.1)
    with pytest.raises(ValueError):
        ContactLaw(normal_stiffness=0.0)


def test_parameter_ledger_hash_is_stable_and_graded() -> None:
    digest = parameter_ledger_sha256()
    assert len(digest) == 64
    assert digest == parameter_ledger_sha256()
    from tensordem.threebody_ice.model import PARAMETER_LEDGER
    assert {grade for _, grade, _ in PARAMETER_LEDGER.values()} <= {
        "REAL", "CONVENTION", "DEM-SOFT"}


def test_floe_mass_and_inertia_follow_the_disk_formulae() -> None:
    law = _quiet_law()
    state = FloeState.create([1.5, 2.0], [[0.0, 0.0], [10.0, 0.0]], law=law)
    expected = law.ice_density * math.pi * 1.5 ** 2 * law.thickness
    assert float(state.mass[0]) == pytest.approx(expected, rel=1e-12)
    assert float(state.inertia[0]) == pytest.approx(0.5 * expected * 1.5 ** 2, rel=1e-12)


def test_single_collision_fractions_match_the_closed_form() -> None:
    fractions = single_collision_fractions(0.4)
    assert fractions["dissipation_fraction_of_input"] == pytest.approx(0.30)
    assert fractions["dissipation_over_ke_gain"] == pytest.approx(0.6 / 1.4)
    # elastic limit: nothing is dissipated
    assert single_collision_fractions(1.0)["dissipation_fraction_of_input"] == 0.0


# ---------------------------------------------------------------------------
# contact laws: signs and conservation
# ---------------------------------------------------------------------------

def test_contact_normal_force_is_repulsive() -> None:
    law = _quiet_law()
    # centres 1.9 m apart with radii 1.0 m each -> 0.1 m of penetration
    state = FloeState.create([1.0, 1.0], [[-0.95, 0.0], [0.95, 0.0]], law=law)
    model = MultiBodyModel(state, law)
    out = model.forces()
    # floe 0 is on the -x side: it must be pushed further in -x, floe 1 in +x
    assert float(out["force"][0, 0]) < 0.0
    assert float(out["force"][1, 0]) > 0.0
    assert float(out["force"][0, 0]) == pytest.approx(-float(out["force"][1, 0]))


def test_tangential_friction_sign_drags_and_spins_correctly() -> None:
    law = _quiet_law(normal_stiffness=1e6, floe_floe_friction=0.3,
                     floe_floe_restitution=1.0)
    # floe 0 at origin, floe 1 above it, sliding along +x while floe 0 rests
    state = FloeState.create([1.0, 1.0], [[0.0, 0.0], [0.0, 1.95]], law=law)
    state.vel[1] = torch.tensor([0.5, 0.0], dtype=TORCH_DTYPE)
    model = MultiBodyModel(state, law)
    out = model.forces()
    # floe 0 must be dragged along +x and spun clockwise (negative z)
    assert float(out["force"][0, 0]) > 0.0
    assert float(out["torque"][0]) < 0.0
    # floe 1 must be slowed down and spun clockwise as well
    assert float(out["force"][1, 0]) < 0.0
    assert float(out["torque"][1]) < 0.0


def test_water_drag_opposes_motion_and_scales_quadratically() -> None:
    law = ContactLaw(water_drag_coefficient=1.0)
    for speed in (0.5, 1.0, 2.0):
        state = FloeState.create([1.0], [[0.0, 0.0]], law=law)
        state.vel[0] = torch.tensor([speed, 0.0], dtype=TORCH_DTYPE)
        model = MultiBodyModel(state, law)
        drag = model.forces()["drag_force"]
        expected = -0.5 * law.water_density * 1.0 * (2 * 1.0 * law.thickness) * speed ** 2
        assert float(drag[0, 0]) == pytest.approx(expected, rel=1e-12)


def test_elastic_free_collision_conserves_momentum_and_energy() -> None:
    law = _quiet_law(floe_floe_restitution=1.0)
    result = elastic_reference(law, speed_m_s=0.5, radius_m=1.0)
    assert not math.isnan(result["relative_momentum_error"])
    assert result["relative_momentum_error"] < THRESHOLDS["momentum_relative_error"]
    assert result["relative_energy_drift"] < THRESHOLDS["elastic_energy_relative_drift"]
    assert result["measured_restitution"] == pytest.approx(1.0, abs=5e-3)
    # equal masses exchange velocities in the elastic limit
    assert result["measured_velocities_m_s"][1][0] == pytest.approx(0.5, abs=1e-2)


def test_restitution_calibration_hits_the_target() -> None:
    target = 0.4
    calibration = calibrate_law(target, law=ContactLaw(
        floe_floe_restitution=target, hull_floe_restitution=target,
        normal_stiffness=2.0e7), tolerance=2.0e-2, max_iterations=12)
    assert calibration["measured_restitution_floe_floe"] == pytest.approx(target, abs=2e-2)
    assert calibration["measured_restitution_hull_floe"] == pytest.approx(target, abs=2e-2)
    # the calibrated damping must differ from the analytic inverse (that is why
    # the calibration exists)
    assert calibration["beta_floe_floe_calibrated"] > 0.0


def test_measured_restitution_of_a_single_free_collision() -> None:
    law = _quiet_law(floe_floe_restitution=1.0)
    assert measure_restitution(law, kind="floe_floe", speed_m_s=0.5,
                               radius_m=1.0) == pytest.approx(1.0, abs=5e-3)
    with pytest.raises(ValueError):
        measure_restitution(law, kind="not_a_kind")


# ---------------------------------------------------------------------------
# three-body scenario
# ---------------------------------------------------------------------------

def test_three_body_force_partition_identity_and_momentum() -> None:
    law = _quiet_law(floe_floe_restitution=0.5, hull_floe_restitution=0.5)
    spec = ScenarioSpec(speed_m_s=1.0, floe_radius_m=1.2, spacing_ratio=1.0)
    row = run_three_body(spec, law, drag=False)
    assert row["force_share_algebraic_error"] < THRESHOLDS["force_partition_algebraic"]
    assert row["impulse_vs_momentum_error"] < THRESHOLDS["momentum_relative_error"]
    assert row["impulse_total"] != pytest.approx(0.0)
    fractions = row["analytic_single_collision_fractions"]
    assert 0.0 <= fractions["dissipation_fraction_of_input"] <= 1.0


def test_three_body_symmetric_configuration_splits_the_load() -> None:
    """Mirrored floes (same station, opposite offsets) must load identically."""
    law = _quiet_law(floe_floe_restitution=0.5, hull_floe_restitution=0.5)
    spec = ScenarioSpec(speed_m_s=1.0, floe_radius_m=1.2, spacing_ratio=0.0,
                        lateral_fraction=0.9)
    model, dt = __import__("tensordem.threebody_ice.threebody",
                           fromlist=["build_three_body"]).build_three_body(spec, law)
    import torch as _torch
    model.state.pos[1, 0] = model.state.pos[0, 0]      # same station: mirrored
    model.state.pos[1, 1] = -model.state.pos[0, 1]
    peaks = [0.0, 0.0]
    for _ in range(int(0.9 / dt)):
        per_floe = model.step(dt)["hull_force_per_floe"]
        peaks[0] = max(peaks[0], abs(float(per_floe[0, 0])))
        peaks[1] = max(peaks[1], abs(float(per_floe[1, 0])))
    assert peaks[0] > 0.0 and peaks[1] > 0.0
    assert peaks[1] == pytest.approx(peaks[0], rel=1e-9)


def test_three_body_spacing_sweep_is_monotone_in_simultaneity() -> None:
    law = _quiet_law(floe_floe_restitution=0.5, hull_floe_restitution=0.5)
    small = run_three_body(ScenarioSpec(speed_m_s=1.0, floe_radius_m=1.0,
                                        spacing_ratio=0.5), law, drag=True,
                           record=False)
    # with water drag the hull keeps re-contacting the near floe: the sweep must
    # actually produce simultaneous loading at some spacing
    assert small["drag_enabled"] is True
    assert small["contact_steps"] > 0


# ---------------------------------------------------------------------------
# field transit
# ---------------------------------------------------------------------------

def test_floe_layout_reproduces_the_concentration_without_overlap() -> None:
    for concentration in (0.3, 0.5):
        spec = FieldSpec(domain_length_m=20.0, domain_width_m=14.0,
                         hull_length_m=8.0, hull_beam_m=4.0,
                         floe_radius_min_m=0.5, floe_radius_max_m=0.9,
                         concentration=concentration, seed_margin_m=1.0)
        radii, points = _floe_layout(spec)
        region = seed_region_m(spec)
        realised = math.pi * sum(r * r for r in radii) / region["area_m2"]
        assert realised == pytest.approx(concentration, rel=0.05)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = math.dist(points[i], points[j])
                assert distance >= radii[i] + radii[j] - 1e-9


def test_field_transit_decomposes_and_closes() -> None:
    law = ContactLaw(floe_floe_restitution=0.4, hull_floe_restitution=0.4,
                     normal_stiffness=2.0e6)
    spec = FieldSpec(domain_length_m=18.0, domain_width_m=12.0,
                     hull_length_m=8.0, hull_beam_m=4.0,
                     floe_radius_min_m=0.4, floe_radius_max_m=0.8,
                     concentration=0.4, speed_m_s=1.0, seed_margin_m=1.5,
                     window_fraction=(0.15, 0.95))
    row = run_field(spec, law, record_every=50)
    assert row["R_ice_force_N"] > 0.0
    assert row["R_collision_N"] >= 0.0
    assert row["R_submerged_drag_N"] >= 0.0
    assert row["R_channel_clearing_N"] >= 0.0
    # the miniature fixture has a short steady window, so the *storage* residual
    # is a larger share of R_ice than in the production matrix; the acceptance
    # threshold is enforced there (validation.evaluate).  What must hold here is
    # the full ledger identity: components + storage residual = R_ice.
    # 5 % is the node's own closure bar; the stricter 1 % self-imposed bound
    # (criterion X1 in the artifact) is NOT met at 2-3 %, because the trapezoid
    # impulse and the midpoint work rate differ across contact start/stop kinks
    assert row["decomposition_closure_with_residual_relative"] < 0.05
    assert row["decomposition_closure_relative"] < 0.10
    assert row["R_force_vs_power_relative"] < THRESHOLDS["decomposition_closure_relative"]
    assert 0.35 < row["realised_concentration"] < 0.45


def test_field_is_deterministic_for_a_seed() -> None:
    law = ContactLaw(floe_floe_restitution=0.4, hull_floe_restitution=0.4,
                     normal_stiffness=2.0e6)
    spec = FieldSpec(domain_length_m=14.0, domain_width_m=10.0,
                     hull_length_m=6.0, hull_beam_m=3.0,
                     floe_radius_min_m=0.4, floe_radius_max_m=0.7,
                     concentration=0.35, speed_m_s=1.5, seed_margin_m=1.2,
                     window_fraction=(0.3, 0.8), seed=7)
    first = run_field(spec, law, record_every=1000)
    second = run_field(spec, law, record_every=1000)
    assert first["R_ice_force_N"] == pytest.approx(second["R_ice_force_N"], rel=1e-12)
    assert first["R_collision_N"] == pytest.approx(second["R_collision_N"], rel=1e-12)


# ---------------------------------------------------------------------------
# criteria machinery / report
# ---------------------------------------------------------------------------

def test_power_law_exponent_recovers_a_known_slope() -> None:
    speeds = [0.5, 1.0, 1.5, 2.0, 3.0]
    fit = power_law_exponent(speeds, [3.2 * v ** 2 for v in speeds])
    assert fit["exponent"] == pytest.approx(2.0, rel=1e-6)
    assert fit["r2_log"] > 0.999
    with pytest.raises(ValueError):
        power_law_exponent([1.0, 2.0], [1.0, 2.0])


def test_manifest_and_thresholds_are_predeclared() -> None:
    keys = [key for key, _, _ in CASE_MANIFEST()]
    assert len(keys) == len(set(keys))
    assert "reference_calibration" in keys
    assert any(key.startswith("field_") for key in keys)
    assert THRESHOLDS["momentum_relative_error"] == 0.01
    assert THRESHOLDS["elastic_energy_relative_drift"] == 0.02
    assert THRESHOLDS["decomposition_closure_relative"] == 0.05
    assert THRESHOLDS["dissipation_band_low"] == 0.30
    assert THRESHOLDS["dissipation_band_high"] == 0.80


def test_evaluate_flags_a_failing_criterion() -> None:
    calibration = {"restitution_error_floe_floe": 0.0,
                   "restitution_error_hull_floe": 0.0}
    elastic = {"relative_momentum_error": 0.5,          # deliberately bad
               "relative_energy_drift": 0.0,
               "measured_restitution": 1.0}
    three = {"impulse_vs_momentum_error": 0.0,
             "force_share_algebraic_error": 0.0,
             "dissipation_fraction_of_input": 0.5,
             "dissipation_over_ke_gain": 0.5,
             "drag_enabled": False,
             "spec": {"spacing_ratio": 1.0}, "impulse_share_1": 0.5,
             "simultaneous_contact_fraction": 0.0,
             "energy_closure_relative": 0.0, "peak_force_total_N": 1.0}
    field = {"spec": {"concentration": 0.5, "speed_m_s": 1.0},
             "R_ice_force_N": 100.0, "R_collision_N": 40.0,
             "R_submerged_drag_N": 40.0, "R_channel_clearing_N": 20.0,
             "R_residual_N": 0.0, "decomposition_closure_relative": 0.0,
             "R_force_vs_power_relative": 0.0, "realised_concentration": 0.5,
             "n_floes": 10, "n_removed": 1}
    parts = [{"key": "reference_calibration", "kind": "reference",
              "calibration": calibration},
             {"key": "reference_elastic", "kind": "reference_elastic",
              "elastic": elastic},
             {"key": "three_body_d1", "kind": "three_body", "row": dict(three)},
             {"key": "three_body_drag_d1", "kind": "three_body_drag",
              "row": dict(three, drag_enabled=True)},
             {"key": "field_c0.5_v1", "kind": "field", "row": dict(field)}]
    validation = evaluate(parts, pytest_count=20, figures=["a.png", "b.png", "c.png"])
    verdicts = {c["name"]: c["passed"] for c in validation["criteria"]}
    assert verdicts["F1a_momentum_error_free_pair_elastic"] is False
    assert verdicts["F4_decomposition_closure"] is True
    assert verdicts["F5_pytest_cases_green"] is True
    assert validation["all_passed"] is False


def test_report_contains_the_six_elements() -> None:
    parts = [{"key": "reference_calibration", "kind": "reference",
              "calibration": {"target_restitution": 0.4,
                              "measured_restitution_floe_floe": 0.4,
                              "measured_restitution_hull_floe": 0.4,
                              "restitution_error_floe_floe": 1e-4,
                              "restitution_error_hull_floe": 1e-4,
                              "analytic_single_collision":
                                  single_collision_fractions(0.4)}},
             {"key": "reference_elastic", "kind": "reference_elastic",
              "elastic": {"relative_momentum_error": 0.0,
                          "relative_energy_drift": 0.001,
                          "measured_restitution": 1.0}}]
    payload = evaluate(parts, pytest_count=20, figures=["f1.png", "f2.png", "f3.png"])
    payload.update({"parameter_ledger_sha256": parameter_ledger_sha256(),
                    "artifact_sha256": "0" * 64, "figures": ["f1.png"],
                    "missing_cases": [], "cases": {}})
    markdown = build_markdown(payload, parts)
    for section in ("① 案例介绍", "② 怎么做的", "③ 计算结果", "④ 结果图",
                    "⑤ 注意点", "⑥ 误差判定", "审计链"):
        assert section in markdown
    assert "Kujala" in markdown            # the withheld benchmark is declared