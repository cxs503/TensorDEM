"""Tests for :mod:`tensordem.ice_ice_collision` (node P202W1-N10).

Coverage map (machine criteria of the node):

* C1 -- Hertz contact force-displacement law vs the analytic impact solution,
* C2 -- freeze-bond normal rupture threshold vs ``sigma_b``,
* C4/C7 -- ``mu_eff(C)`` monotone increase and >= 10x growth,
* C5 -- ``mu_eff(C = 0) = 0``,
* C6 -- closed energy ledger ``E_kin + E_spring + E_diss = E_kin(0)``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ice_ice_collision import (
    FloeFieldConfig,
    IceIceCollisionError,
    bond_area,
    bond_normal_stress,
    bond_ruptures,
    bond_tensile_test,
    bonded_pair_rupture_test,
    hertz_contact_force,
    hertz_contact_radius,
    hertz_impact_peak,
    hertz_secant_stiffness,
    hunt_crossley_beta,
    mindlin_tangential_stiffness,
    reduced_radius,
    reduced_shear_modulus,
    reduced_youngs_modulus,
    simulate_floe_field,
    two_floe_collision,
)


# ---------------------------------------------------------------------------
# contact primitives
# ---------------------------------------------------------------------------
def test_reduced_quantities_equal_spheres():
    e_star = reduced_youngs_modulus(5.0e9, 5.0e9, 0.3, 0.3)
    assert e_star == pytest.approx(5.0e9 / (2.0 * (1.0 - 0.3 ** 2)), rel=1e-12)
    g = 5.0e9 / (2.0 * (1.0 + 0.3))
    g_star = reduced_shear_modulus(g, g, 0.3, 0.3)
    assert g_star == pytest.approx(g / (2.0 * (2.0 - 0.3)), rel=1e-12)
    assert reduced_radius(0.6, 0.6) == pytest.approx(0.3, rel=1e-15)
    assert reduced_radius(0.6, math.inf) == pytest.approx(0.6, rel=1e-15)


def test_hertz_force_is_three_halves_power_law():
    r_eff, e_eff = 0.3, 5.5e6
    delta = np.array([1e-4, 1e-3, 1e-2])
    force = hertz_contact_force(delta, r_eff, e_eff)
    assert force == pytest.approx((4.0 / 3.0) * e_eff * math.sqrt(r_eff)
                                  * delta ** 1.5, rel=1e-14)
    # secant stiffness is the analytic derivative dF/ddelta = 2 E* a
    h = 1e-9
    num = (hertz_contact_force(delta + h, r_eff, e_eff)
           - hertz_contact_force(delta - h, r_eff, e_eff)) / (2.0 * h)
    assert num == pytest.approx(hertz_secant_stiffness(delta, r_eff, e_eff),
                                rel=1e-6)
    assert hertz_contact_radius(delta, r_eff) == pytest.approx(
        np.sqrt(r_eff * delta), rel=1e-14)


def test_mindlin_stiffness_ratio_for_equal_spheres():
    e_eff, g_eff = 5.5e6, 2.1e6
    delta = 1e-3
    ratio = mindlin_tangential_stiffness(delta, 0.3, g_eff) \
        / hertz_secant_stiffness(delta, 0.3, e_eff)
    assert ratio == pytest.approx(4.0 * g_eff / e_eff, rel=1e-12)


def test_thickness_scales_contact_force_linearly():
    """A thicker plate carries proportionally larger contact forces."""
    f1 = hertz_contact_force(1e-3, 0.3, 5.5e6, 0.5)
    f2 = hertz_contact_force(1e-3, 0.3, 5.5e6, 1.0)
    assert f2 == pytest.approx(2.0 * f1, rel=1e-15)


# ---------------------------------------------------------------------------
# C1: Hertz impact vs analytic solution
# ---------------------------------------------------------------------------
def test_c1_hertz_impact_matches_analytic():
    res = two_floe_collision(restitution=1.0, friction=0.0, impact_speed=1.0,
                             dt_factor=0.005)
    assert res["force_error"] <= 0.05
    assert res["overlap_error"] <= 0.05
    # the analytic peak is the Hertz energy-balance solution of the same pair
    mass = 900.0 * math.pi * 0.6 ** 2 * 0.5
    e_eff = reduced_youngs_modulus(1.0e7, 1.0e7, 0.3, 0.3)
    analytic = hertz_impact_peak(0.5 * mass, 1.0, 0.3, e_eff, 0.5)
    assert analytic["delta_max"] == pytest.approx(res["analytic"]["delta_max"],
                                                  rel=1e-9)


# ---------------------------------------------------------------------------
# C6: energy ledger
# ---------------------------------------------------------------------------
def test_c6_elastic_impact_conserves_energy():
    res = two_floe_collision(restitution=1.0, friction=0.0, impact_speed=1.0,
                             dt_factor=0.005)
    assert res["ledger_relative_error"] <= 0.01
    assert res["measured_restitution"] == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("restitution,friction", [(0.7, 0.0), (0.5, 0.0),
                                                  (0.5, 0.3), (0.3, 0.3)])
def test_c6_dissipative_impact_ledger_closes(restitution, friction):
    res = two_floe_collision(restitution=restitution, friction=friction,
                             impact_speed=1.0, dt_factor=0.005)
    assert res["ledger_relative_error"] <= 0.05
    assert res["dissipated"] > 0.0
    # measured restitution must be below the elastic value and above zero
    assert 0.0 < res["measured_restitution"] < 1.0


def test_hunt_crossley_beta_limits():
    assert hunt_crossley_beta(1.0, 1.0) == pytest.approx(0.0, abs=1e-9)
    assert hunt_crossley_beta(0.5, 1.0) > 0.0
    with pytest.raises(IceIceCollisionError):
        hunt_crossley_beta(0.5, 0.0)


# ---------------------------------------------------------------------------
# C2: freeze-bond rupture
# ---------------------------------------------------------------------------
def test_bond_area_and_stress_helpers():
    area = bond_area(0.6, 0.6, 0.25)
    assert area == pytest.approx(math.pi * (0.15 ** 2), rel=1e-12)
    assert bond_normal_stress(1000.0, area) == pytest.approx(1000.0 / area)
    assert bool(bond_ruptures(1000.0, area, sigma_b=1.0e6)) is False
    assert bool(bond_ruptures(1000.0, area, sigma_b=1.0)) is True


def test_bond_quasi_static_tensile_test():
    res = bond_tensile_test(sigma_b=1.0e5)
    assert res["ruptured"]
    assert res["relative_error"] <= 0.01


@pytest.mark.parametrize("sigma_b", [5.0e4, 1.0e5, 3.0e5])
def test_c2_bond_rupture_threshold_in_simulator(sigma_b):
    res = bonded_pair_rupture_test(sigma_b=sigma_b)
    assert res["ruptured"]
    assert res["relative_error"] <= 0.10
    # after rupture the pair has no tensile capacity
    assert res["post_rupture_bond_active"] is False


# ---------------------------------------------------------------------------
# C5: no ice, no viscosity
# ---------------------------------------------------------------------------
def test_c5_zero_concentration_gives_zero_viscosity():
    out = simulate_floe_field(FloeFieldConfig(concentration=0.0, shear_steps=10,
                                              settle_steps=0))
    assert out["n_floes"] == 0
    assert out["mu_eff"] == 0.0
    assert out["tau_total"] == 0.0
    assert out["diss_total"] == 0.0


# ---------------------------------------------------------------------------
# C4/C7: concentration dependence of the equivalent viscosity
# ---------------------------------------------------------------------------
def test_c4_c7_mu_eff_increases_with_concentration():
    """Coarse sweep in a small cell: monotone increase and >= 10x growth."""
    base = dict(box_width=10.0, box_height=10.0, shear_steps=1500,
                sample_steps=700, settle_steps=800)
    mus = []
    for c in (0.2, 0.5, 0.8):
        out = simulate_floe_field(FloeFieldConfig(concentration=c, **base))
        mus.append(out["mu_eff"])
    assert mus[0] < mus[1] < mus[2]
    assert mus[-1] / mus[0] >= 10.0


# ---------------------------------------------------------------------------
# configuration guards and field invariants
# ---------------------------------------------------------------------------
def test_invalid_configuration_raises():
    with pytest.raises(IceIceCollisionError):
        FloeFieldConfig(concentration=1.5)
    with pytest.raises(IceIceCollisionError):
        FloeFieldConfig(restitution=1.0)
    with pytest.raises(IceIceCollisionError):
        FloeFieldConfig(mean_radius=-1.0)
    with pytest.raises(IceIceCollisionError):
        FloeFieldConfig(radius_spread=0.5)


def test_field_diagnostics_are_physical():
    out = simulate_floe_field(FloeFieldConfig(concentration=0.6, box_width=10.0,
                                              box_height=10.0, shear_steps=800,
                                              sample_steps=400, settle_steps=600))
    assert out["n_floes"] > 0
    assert out["mean_contacts"] > 0.0
    assert out["mu_eff"] > 0.0
    assert out["diss_total"] > 0.0
    assert out["pressure"] > 0.0          # compressed (not cohesive) field
    assert 0.0 <= out["settle_max_overlap"] < 1.0