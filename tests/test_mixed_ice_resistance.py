"""Mixed-field resistance: additivity, class decomposition, sensitivity, DEM wiring.

The additivity identity is algebra, so it is checked to machine precision against
an independent evaluation (anchor over explicit FY thickness quantiles).  The DEM
side is checked for the interface it needs (an age-dependent elastic modulus
reaching ``SheetConfig``) and for the exponent fit on a synthetic exactly-
separable response, not by re-running the campaign.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.mixed_ice import aging
from tensordem.mixed_ice.campaign import fit_exponents
from tensordem.mixed_ice.field import MixedFieldSpec, quadrature_field
from tensordem.mixed_ice.resistance import (SENSITIVITY_PARAMETERS,
                                            _node_resistance, class_ratio_decomposition,
                                            mixed_resistance, quadrature_vs_sampler,
                                            resistance_sensitivity)
from tensordem.level_ice.campaign import CaseSpec
from tensordem.level_ice.formula import ICEBREAKER, IceState, lindqvist_components


def test_pure_first_year_field_matches_an_independent_quadrature():
    """At p = 0 the field expectation is the FY thickness average alone."""
    spec = MixedFieldSpec(my_fraction=0.0)
    result = mixed_resistance(spec, speed_m_s=3.0)
    sample = quadrature_field(spec)
    fy = aging.first_year_state()
    expected = float(np.dot(
        sample.weight,
        [lindqvist_components(fy.as_ice_state(float(h)), ICEBREAKER, 3.0)["total_N"]
         for h in sample.thickness_m]))
    assert result.resistance_mean_n == pytest.approx(expected, rel=1e-12)
    assert math.isnan(result.class_ratio)
    assert result.relative_increase == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("model", ["lindqvist", "riska"])
def test_class_additivity_identity(model):
    for p in (0.2, 0.4, 0.8):
        res = mixed_resistance(MixedFieldSpec(my_fraction=p), speed_m_s=3.0, model=model)
        assert res.relative_increase == pytest.approx(p * (res.class_ratio - 1.0), rel=1e-12)
        # the field value is the class-weighted mean of the conditional means
        assert res.resistance_mean_n == pytest.approx(
            (1 - p) * res.resistance_fy_n + p * res.resistance_my_n, rel=1e-12)


def test_my_fraction_increases_resistance_monotonically():
    values = [mixed_resistance(MixedFieldSpec(my_fraction=p), speed_m_s=3.0).resistance_mean_n
              for p in (0.0, 0.2, 0.4, 0.6, 0.8)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_class_ratio_decomposition_factorises():
    spec = MixedFieldSpec(my_fraction=0.4)
    dec = class_ratio_decomposition(spec)
    assert dec["ratio_full"] == pytest.approx(
        dec["ratio_strength_only"] * dec["ratio_thickness_only"] * (1.0 + dec["interaction"]),
        rel=1e-12)
    # both parts push the resistance up, and thickness dominates this field
    assert dec["ratio_strength_only"] > 1.0
    assert dec["ratio_thickness_only"] > dec["ratio_strength_only"]
    assert 0.0 < dec["thickness_share_of_increment"] < 1.0


def test_strength_multiplier_scales_only_the_my_class():
    spec = MixedFieldSpec(my_fraction=0.4)
    base = mixed_resistance(spec)
    stronger = mixed_resistance(spec, my_strength_multiplier=1.2)
    assert stronger.resistance_my_n > base.resistance_my_n
    assert stronger.resistance_fy_n == pytest.approx(base.resistance_fy_n, rel=1e-12)


def test_component_shares_sum_to_one():
    res = mixed_resistance(MixedFieldSpec(my_fraction=0.4))
    assert sum(res.component_shares.values()) == pytest.approx(1.0, abs=1e-12)


def test_quantiles_are_ordered():
    res = mixed_resistance(MixedFieldSpec(my_fraction=0.4))
    q = res.quantiles_n
    assert q["p10"] <= q["p50"] <= q["p90"]


def test_sensitivity_covers_four_fsd_parameters_and_ranks_them():
    sens = resistance_sensitivity(MixedFieldSpec(my_fraction=0.4),
                                  parameters={k: v for k, v in SENSITIVITY_PARAMETERS.items()})
    groups = {name: row["group"] for name, row in sens["parameters"].items()}
    assert sum(1 for g in groups.values() if g == "fsd") == 4
    assert sens["rank_by_span"][0] in groups
    assert sens["fsd_additive_bound_N"] > 0.0
    assert 0.0 <= sens["fsd_share_of_bound"] <= 1.0
    # ranks are sorted by span, i.e. the first entry dominates
    spans = [sens["parameters"][n]["max_abs_deviation_N"] for n in sens["rank_by_span"]]
    assert spans == sorted(spans, reverse=True)


def test_estimator_agreement_is_within_three_standard_errors():
    check = quadrature_vs_sampler(MixedFieldSpec(my_fraction=0.4), n=20_000,
                                  speed_m_s=3.0)
    assert abs(check["z_score"]) < 3.0
    assert abs(check["my_fraction_bias"]) < 0.01


def test_unknown_model_and_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        mixed_resistance(MixedFieldSpec(my_fraction=0.4), model="not-a-model")
    with pytest.raises(ValueError):
        mixed_resistance(MixedFieldSpec(my_fraction=0.4), speed_m_s=-1.0)
    with pytest.raises(ValueError):
        mixed_resistance(MixedFieldSpec(my_fraction=0.4), my_strength_multiplier=0.0)


def test_fit_exponents_recovers_a_separable_response():
    """On a response that is exactly a product of powers the fit is exact."""
    levels = {"a": (1.0, 2.0), "b": (3.0, 5.0), "c": (0.5, 0.75)}
    exponents = {"a": 0.7, "b": -0.4, "c": 1.5}
    resistances = []
    for code in range(8):
        value = 10.0
        for j, key in enumerate(levels):
            low, high = levels[key]
            level = high if (code >> j) & 1 else low
            value *= (level / low) ** exponents[key]
        resistances.append(value)
    fit = fit_exponents(np.array(resistances), levels)
    assert fit["residual_max_abs_log"] < 1e-12
    for key, expected in exponents.items():
        assert fit["exponents"][key] == pytest.approx(expected, rel=1e-10)
    assert fit["reference_resistance_N"] == pytest.approx(10.0, rel=1e-12)


def test_dem_case_spec_accepts_an_age_dependent_modulus():
    """The interface the DEM cross-check needs: aged E reaches SheetConfig."""
    state = aging.material_state(3.0)
    spec = CaseSpec(elastic_modulus_pa=state.elastic_modulus_pa,
                    compressive_strength_pa=state.compressive_strength_pa,
                    flexural_strength_pa=state.flexural_strength_pa)
    config = spec.sheet_config()
    assert config.elastic_modulus_pa == pytest.approx(state.elastic_modulus_pa)
    assert config.crush_cap_pa == pytest.approx(state.compressive_strength_pa)
    # the default keeps the historical 5 GPa sheet
    assert CaseSpec().sheet_config().elastic_modulus_pa == pytest.approx(5.0e9)
    assert spec.as_dict()["elastic_modulus_pa"] == pytest.approx(state.elastic_modulus_pa)


def test_node_resistance_reproduces_the_canonical_anchor():
    """The per-node evaluation must call the canonical anchor, not a copy."""
    spec = MixedFieldSpec(my_fraction=0.2, seed=2)
    sample = quadrature_field(spec)
    parts = _node_resistance(sample, "lindqvist", ICEBREAKER, 3.0)
    state = IceState(
        thickness_m=float(sample.thickness_m[0]),
        compressive_strength_pa=float(sample.compressive_strength_pa[0]),
        flexural_strength_pa=float(sample.flexural_strength_pa[0]),
        elastic_modulus_pa=float(sample.elastic_modulus_pa[0]),
        density_kg_m3=float(sample.density_kg_m3[0]),
    )
    expected = lindqvist_components(state, ICEBREAKER, 3.0)
    assert parts["total"][0] == pytest.approx(expected["total_N"], rel=1e-12)
    assert parts["crushing"][0] == pytest.approx(expected["crushing_N"], rel=1e-12)
    assert parts["crushing"][0] + parts["bending"][0] + parts["submersion"][0] + parts["drag"][0] \
        == pytest.approx(parts["total"][0], rel=1e-12)