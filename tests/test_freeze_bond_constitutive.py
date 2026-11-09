"""Freeze-bond constitutive law and cohesive element: self-consistency tests.

These tests check *implementation* consistency, not ice physics:

* the element's peak traction equals the closed-form power law it is given,
  to machine precision, for tension and shear, over a grid of
  ``(Delta T, S, t)``;
* the softening branch spends exactly the declared fracture energy
  ``G_f = 0.5 * sigma * delta_f`` and the dissipation is a monotone,
  irreversible state function of the history variable;
* a broken bond can be re-frozen (strength restored, dissipated energy *not*
  recovered).

The law itself is a declared hypothesis (see the module docstring); no test here
claims it matches measured freeze-bond strength.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.freeze_bond import FREEZE_BOND_LAW_REFERENCE, FreezeBondLaw
from tensordem.freeze_bond.cohesive import BondEnvironment, CohesiveBondSet


def test_power_law_matches_hand_computation():
    law = FreezeBondLaw(
        coefficient_pa=1.0e4,
        temperature_exponent=1.0,
        salinity_coefficient_per_ppt=0.05,
        consolidation_exponent=0.5,
        shear_to_tension_ratio=0.5,
        fracture_energy_J_m2=3.0,
    )
    dT, s, t = 4.0, 10.0, 3600.0
    expected = 1.0e4 * 4.0 * (1.0 + 0.05 * 10.0) ** (-1.0) * math.sqrt(t)
    assert law.tension_strength(dT, s, t) == pytest.approx(expected, rel=1e-15)
    assert law.shear_strength(dT, s, t) == pytest.approx(0.5 * expected, rel=1e-15)


def test_power_law_monotonicity_and_limits():
    law = FREEZE_BOND_LAW_REFERENCE
    assert law.tension_strength(5.0, 0.0, 3600.0) == pytest.approx(3.0e4, rel=1e-12)
    # colder -> stronger, saltier -> weaker, older -> stronger
    assert law.tension_strength(10.0, 0.0, 3600.0) > law.tension_strength(5.0, 0.0, 3600.0)
    assert law.tension_strength(5.0, 20.0, 3600.0) < law.tension_strength(5.0, 0.0, 3600.0)
    assert law.tension_strength(5.0, 0.0, 7200.0) > law.tension_strength(5.0, 0.0, 3600.0)
    # no supercooling -> no bond at all
    assert law.tension_strength(0.0, 0.0, 3600.0) == 0.0
    assert law.shear_strength(0.0, 0.0, 3600.0) == 0.0
    with pytest.raises(ValueError):
        law.tension_strength(5.0, -1.0, 3600.0)
    with pytest.raises(ValueError):
        law.tension_strength(5.0, 0.0, 0.0)


def test_reference_anchor_reproduces_strength():
    law = FreezeBondLaw.from_reference_strength(
        1.23e4, reference_delta_T_K=3.0, reference_salinity_ppt=7.0,
        reference_consolidation_s=1800.0, salinity_coefficient_per_ppt=0.04,
        temperature_exponent=1.2, consolidation_exponent=0.4)
    assert law.tension_strength(3.0, 7.0, 1800.0) == pytest.approx(1.23e4, rel=1e-12)


def test_softening_ratio_is_energetically_consistent():
    law = FREEZE_BOND_LAW_REFERENCE
    k0 = 1.5e9
    for sigma in (1.0e3, 3.0e4, 2.0e5):
        gf = law.fracture_energy_at(sigma, k0)
        delta_0 = sigma / k0
        delta_f = 2.0 * gf / sigma
        assert delta_f > delta_0          # the soft branch exists for every strength
        assert gf == pytest.approx(0.5 * law.softening_ratio * sigma ** 2 / k0, rel=1e-12)


@pytest.mark.parametrize("dT,salt,time_s", [(1.0, 0.0, 600.0), (5.0, 10.0, 3600.0),
                                            (10.0, 20.0, 86400.0)])
def test_element_peak_traction_matches_analytic_law(dT, salt, time_s):
    law = FREEZE_BOND_LAW_REFERENCE
    k0 = 1.5e9
    env = BondEnvironment.uniform(1, delta_T_K=dT, salinity_ppt=salt,
                                 consolidation_time_s=time_s)
    sigma = float(law.tension_strength(dT, salt, time_s))
    d0 = sigma / k0
    for mode in ("tension", "shear"):
        cs = CohesiveBondSet(law, env, stiffness_pa_per_m=k0)
        target = sigma if mode == "tension" else float(law.shear_strength(dT, salt, time_s))
        d_lim = target / k0
        peak = 0.0
        grid = np.union1d(np.linspace(0.0, 20.0 * d0, 500), [d_lim])
        for delta in grid:
            dn = np.array([delta]) if mode == "tension" else np.array([0.0])
            dt = np.array([0.0]) if mode == "tension" else np.array([delta])
            state = cs.trial(dn, dt)
            peak = max(peak, float(state["traction"][0]))
            cs.commit(state)
        assert peak == pytest.approx(target, rel=1e-12), mode


def test_fracture_energy_is_spent_exactly_and_dissipation_is_irreversible():
    law = FREEZE_BOND_LAW_REFERENCE
    k0 = 1.5e9
    sigma = 3.0e4
    gf = float(law.fracture_energy_at(sigma, k0))
    delta_f = 2.0 * gf / sigma
    cs = CohesiveBondSet(law, BondEnvironment.uniform(1), stiffness_pa_per_m=k0)
    previous = -1.0
    for delta in np.linspace(0.0, 1.2 * delta_f, 1201):
        state = cs.trial(np.array([delta]), np.array([0.0]))
        cs.commit(state)
        assert cs.dissipated_J_m2[0] >= previous - 1e-15   # monotone
        previous = cs.dissipated_J_m2[0]
    assert cs.dissipated_J_m2[0] == pytest.approx(gf, rel=1e-12)
    assert cs.work_J_m2[0] == pytest.approx(gf, rel=1e-12)
    assert cs.trial(np.array([delta_f]), np.array([0.0]))["broken"][0]
    # unloading must not dissipate
    for delta in np.linspace(delta_f, 0.0, 201):
        state = cs.trial(np.array([delta]), np.array([0.0]))
        cs.commit(state)
    assert cs.dissipated_J_m2[0] == pytest.approx(gf, rel=1e-12)


def test_refreeze_restores_strength_but_keeps_dissipation():
    law = FREEZE_BOND_LAW_REFERENCE
    k0 = 1.5e9
    sigma = 3.0e4
    gf = float(law.fracture_energy_at(sigma, k0))
    cs = CohesiveBondSet(law, BondEnvironment.uniform(1, consolidation_time_s=3600.0),
                         stiffness_pa_per_m=k0)
    delta_f = 2.0 * gf / sigma
    state = cs.trial(np.array([2.0 * delta_f]), np.array([0.0]))
    cs.commit(state)
    assert state["broken"][0]
    dissipated_before = float(cs.dissipated_J_m2[0])
    healed = cs.refreeze(np.array([True]), hold_time_s=86400.0)
    assert healed == 1
    assert cs.dissipated_J_m2[0] == pytest.approx(dissipated_before, rel=1e-12)
    fresh = cs.trial(np.array([0.0]), np.array([0.0]))
    assert not fresh["broken"][0]
    # the regrown neck follows the same kinetic law with the new consolidation age
    expected = float(law.tension_strength(5.0, 0.0, 86400.0))
    assert expected > sigma     # gamma > 0: a longer contact gives a stronger neck
    assert cs.strengths()[0][0] == pytest.approx(expected, rel=1e-12)


def test_zero_strength_bond_carries_nothing():
    law = FREEZE_BOND_LAW_REFERENCE
    cs = CohesiveBondSet(law, BondEnvironment.uniform(1, delta_T_K=0.0),
                         stiffness_pa_per_m=1.5e9)
    state = cs.trial(np.array([1e-4]), np.array([0.0]))
    assert state["traction"][0] == 0.0
    assert state["damage"][0] == pytest.approx(1.0)
    assert state["broken"][0]


def test_closing_bond_carries_no_normal_traction():
    """The cohesive element is one-sided: compression is the contacts' job."""
    law = FREEZE_BOND_LAW_REFERENCE
    cs = CohesiveBondSet(law, BondEnvironment.uniform(1), stiffness_pa_per_m=1.5e9)
    state = cs.trial(np.array([-5e-5]), np.array([0.0]))
    assert state["t_e" if "t_e" in state else "traction"][0] == 0.0
    assert state["t_n"][0] == 0.0
    assert state["damage"][0] == 0.0