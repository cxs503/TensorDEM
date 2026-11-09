"""Ageing parameterisation: physical limits, calibration and provenance.

Every expected number here is either an exact algebraic consequence of the
declared laws or the published limit of the cited relation; nothing is a
regression snapshot of the implementation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.mixed_ice import aging
from tensordem.level_ice.formula import IceState


def test_brine_volume_matches_the_closed_form():
    # v_b = S (1 + 49.185/|T|)/1000: hand-computed for S = 5 permille, T = -10 degC
    expected = 0.005 * (1.0 + 4.9185)
    assert aging.brine_volume_fraction(5.0, -10.0) == pytest.approx(expected, rel=1e-12)
    # and it is monotonically decreasing in |T| and increasing in S
    cold = aging.brine_volume_fraction(5.0, -20.0)
    warm = aging.brine_volume_fraction(5.0, -5.0)
    assert cold < warm
    assert aging.brine_volume_fraction(2.0, -10.0) < aging.brine_volume_fraction(6.0, -10.0)


def test_brine_volume_rejects_melting_ice_and_percolation():
    with pytest.raises(ValueError):
        aging.brine_volume_fraction(5.0, 0.5)
    with pytest.raises(ValueError):
        aging.brine_volume_fraction(30.0, -2.0)  # far above the percolation threshold


def test_flexural_strength_limits_of_the_cited_relation():
    # v_b -> 0 gives the near-pure-ice value 1.76 MPa
    assert aging.flexural_strength_pa(0.0) == pytest.approx(1.76e6, rel=1e-12)
    # the relation's stated validity edge (v_b <= 0.27)
    assert aging.flexural_strength_pa(0.27) == pytest.approx(
        1.76e6 * math.exp(-5.88 * math.sqrt(0.27)), rel=1e-12)
    assert aging.flexural_strength_pa(0.0) > aging.flexural_strength_pa(0.1)
    with pytest.raises(ValueError):
        aging.flexural_strength_pa(0.30)


def test_first_year_state_reproduces_the_repository_reference_values():
    """The calibration makes age 1 exactly consistent with the FY model."""
    fy = aging.first_year_state()
    assert fy.compressive_strength_pa == pytest.approx(1.5e6, rel=1e-12)
    assert fy.elastic_modulus_pa == pytest.approx(5.0e9, rel=1e-12)
    assert fy.salinity_permille == pytest.approx(aging.FY_SALINITY_PERMILLE)
    assert fy.temperature_c == pytest.approx(aging.FY_TEMPERATURE_C)
    assert fy.thickness_m == pytest.approx(aging.FY_THICKNESS_M)
    assert fy.ice_class == "FY"


def test_first_year_state_bridges_to_level_ice_ice_state():
    fy = aging.first_year_state()
    state = fy.as_ice_state()
    assert isinstance(state, IceState)
    assert state.thickness_m == pytest.approx(fy.thickness_m)
    assert state.flexural_strength_pa == pytest.approx(fy.flexural_strength_pa)
    # an explicit thickness override is honoured (per-floe FSD use)
    assert fy.as_ice_state(2.5).thickness_m == pytest.approx(2.5)


def test_ageing_is_monotone_in_every_material_property():
    ages = np.linspace(1.0, 40.0, 40)
    states = [aging.material_state(float(a)) for a in ages]
    for key in ("salinity_permille", "temperature_c"):
        values = [getattr(s, key) for s in states]
        if key == "salinity_permille":
            assert all(b <= a for a, b in zip(values, values[1:]))
        else:
            assert all(b <= a for a, b in zip(values, values[1:]))  # colder
    for key in ("thickness_m", "flexural_strength_pa", "compressive_strength_pa",
                "elastic_modulus_pa", "fracture_toughness_pa_sqrt_m"):
        values = [getattr(s, key) for s in states]
        assert all(b >= a - 1e-12 for a, b in zip(values, values[1:])), key
    # density falls slightly with age: brine (denser than ice) drains out
    density = [s.density_kg_m3 for s in states]
    assert all(b <= a + 1e-12 for a, b in zip(density, density[1:]))


def test_my_over_fy_ratios_inside_the_cited_bands():
    fy = aging.first_year_state()
    my = aging.material_state(3.0)
    assert 1.3 <= my.flexural_strength_pa / fy.flexural_strength_pa <= 2.5
    assert 1.2 <= my.compressive_strength_pa / fy.compressive_strength_pa <= 2.5
    assert 1.1 <= my.elastic_modulus_pa / fy.elastic_modulus_pa <= 1.7
    assert my.thickness_m / fy.thickness_m > 1.5
    assert my.ice_class == "MY"


def test_asymptotic_state_is_finite_and_physically_plausible():
    far = aging.material_state(200.0)
    assert far.salinity_permille == pytest.approx(aging.MY_SALINITY_ASYMPTOTE_PERMILLE, rel=1e-6)
    assert far.temperature_c == pytest.approx(aging.MY_TEMPERATURE_ASYMPTOTE_C, rel=1e-6)
    assert far.thickness_m == pytest.approx(aging.MY_THICKNESS_ASYMPTOTE_M, rel=1e-6)
    assert 3.0e9 < far.elastic_modulus_pa < 1.0e10
    assert 0.5e6 < far.flexural_strength_pa < 2.0e6
    assert 0.0 < far.density_kg_m3 < 930.0


def test_density_is_between_pure_ice_and_brine():
    state = aging.material_state(2.0)
    rho_brine = aging.BRINE_DENSITY_BASE_KG_M3 + aging.BRINE_DENSITY_SLOPE * state.salinity_permille
    assert aging.PURE_ICE_DENSITY_KG_M3 < state.density_kg_m3 < rho_brine


def test_age_below_one_year_is_rejected():
    with pytest.raises(ValueError):
        aging.material_state(0.5)


def test_ledger_grades_and_hash():
    grades = {grade for _v, grade, _t in aging.COEFFICIENT_LEDGER.values()}
    assert grades <= {"DERIVED", "CONVENTION", "CITED", "WITHHELD"}
    assert aging.COEFFICIENT_LEDGER["literature_agreement"][1] == "WITHHELD"
    first = aging.aging_ledger_sha256()
    assert first == aging.aging_ledger_sha256()
    assert len(first) == 64


def test_fracture_toughness_is_flagged_as_not_consumed_by_the_dem():
    state = aging.material_state(3.0).as_dict()
    assert state["consumed_by_dem"]["fracture_toughness_pa_sqrt_m"] is False
    assert state["consumed_by_dem"]["compressive_strength_pa"] is True