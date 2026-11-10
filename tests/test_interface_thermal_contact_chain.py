"""Acceptance tests for the contact-conduction / particle-chain verification face.

Criteria covered (campaign node p40w2-n06, thermal half):

* (a) single-pair steady thermal resistance from the Hertz patch agrees with an
  independent axisymmetric finite-volume solution of the half-space problem;
* (b) particle-chain steady and transient temperature fields agree with a fine
  one-dimensional finite-volume reference rod to better than 15 %;
* the ice-bond gate: ``k_ice(T)`` against the tabulated Ih values, the bridge
  resistance against a slab solution, and the parallel-combination bounds.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.interface_thermal.contact_chain import (
    ICE_CONDUCTIVITY_TABLE,
    HalfSpacePatch,
    chain_lumped_steady,
    chain_steady_analytic,
    chain_transient_fvm,
    chain_transient_lumped,
    grain_area,
    hertz_patch_radius,
    ice_bond_gate,
    ice_bridge_resistance,
    ice_conductivity,
    pair_contact_resistance,
    rod_temperature_fvm,
)

RADIUS = 0.005
K_ICE = 2.2
RHO = 917.0
CP = 2100.0
AREA = grain_area(RADIUS)
GRAIN = 2.0 * RADIUS / (K_ICE * AREA)
R_CONTACT = 3.0 * GRAIN
N_PARTICLES = 20


def test_hertz_patch_radius_follows_analytic_scaling():
    """a ~ F^(1/3), a ~ R*^(1/3), a ~ (1/E*)^(1/3) (Hertz, no fitting)."""
    a1 = hertz_patch_radius(10.0, 0.005, 0.005, 9.0e9, 9.0e9, 0.33, 0.33)
    a8 = hertz_patch_radius(80.0, 0.005, 0.005, 9.0e9, 9.0e9, 0.33, 0.33)
    assert a8 / a1 == pytest.approx(8.0 ** (1.0 / 3.0), rel=1e-12)
    a_soft = hertz_patch_radius(10.0, 0.005, 0.005, 9.0e8, 9.0e8, 0.33, 0.33)
    assert a_soft / a1 == pytest.approx(10.0 ** (1.0 / 3.0), rel=1e-12)
    # equal spheres: R* = R/2 and E* = E/(2(1-nu^2)) -> closed form
    closed = (3.0 * 10.0 * (0.005 / 2.0)
              / (4.0 * 9.0e9 / (2.0 * (1.0 - 0.33 ** 2)))) ** (1.0 / 3.0)
    assert a1 == pytest.approx(closed, rel=1e-12)


def test_pair_contact_resistance_is_two_maxwell_constrictions():
    a = 1.5e-4
    r_eq = pair_contact_resistance(K_ICE, K_ICE, a)
    assert r_eq == pytest.approx(1.0 / (2.0 * K_ICE * a), rel=1e-12)
    r_split = pair_contact_resistance(K_ICE, 4.0 * K_ICE, a)
    assert r_split == pytest.approx(1.0 / (4.0 * K_ICE * a) + 1.0 / (16.0 * K_ICE * a), rel=1e-12)


def test_half_space_patch_reproduces_maxwell_constriction():
    """Criterion (a): mesh solution of div(k grad T)=0 vs 1/(4 k a), gate < 10 %."""
    a = hertz_patch_radius(10.0, 0.005, 0.005, 9.0e9, 9.0e9, 0.33, 0.33)
    mesh = HalfSpacePatch(K_ICE, a, cells_per_radius=16)
    assert abs(mesh.relative_error) < 0.10
    assert mesh.resistance_k_w > 0.0


def test_half_space_patch_converges_with_mesh_refinement():
    a = 2.0e-4
    errors = [abs(HalfSpacePatch(K_ICE, a, cells_per_radius=n).relative_error)
              for n in (8, 16)]
    assert errors[1] < errors[0]


def test_rod_fvm_steady_state_matches_piecewise_linear_analytic():
    fvm = rod_temperature_fvm(N_PARTICLES, RADIUS, K_ICE, RHO, CP, R_CONTACT,
                              cells_per_particle=24, time_steps=200, total_time=1.0e9)
    analytic = chain_steady_analytic(N_PARTICLES, RADIUS, K_ICE, R_CONTACT)
    centres = (np.arange(N_PARTICLES) + 0.5) * 2.0 * RADIUS
    idx = [int(np.argmin(np.abs(fvm["x"] - x))) for x in centres]
    reference = np.interp(centres, analytic["x"], analytic["temperature"])
    assert np.max(np.abs(fvm["final"][idx] - reference)) < 0.01


def test_lumped_chain_steady_field_matches_continuum():
    """Criterion (b): lumped DEM thermal network vs continuum rod, steady state."""
    lumped = chain_lumped_steady(N_PARTICLES, RADIUS, K_ICE, RHO, CP, R_CONTACT)
    fvm = rod_temperature_fvm(N_PARTICLES, RADIUS, K_ICE, RHO, CP, R_CONTACT,
                              cells_per_particle=24, time_steps=200, total_time=1.0e9)
    centres = (np.arange(N_PARTICLES) + 0.5) * 2.0 * RADIUS
    idx = [int(np.argmin(np.abs(fvm["x"] - x))) for x in centres]
    error = float(np.max(np.abs(lumped - fvm["final"][idx]))) / 1.0
    assert error < 0.05, error


def test_lumped_chain_transient_field_within_fifteen_percent():
    """Criterion (b): transient T(x,t) of the lumped chain vs continuum rod."""
    diffusion_time = 0.5 * (2.0 * RADIUS * N_PARTICLES) ** 2 / (K_ICE / (RHO * CP))
    fvm = chain_transient_fvm(N_PARTICLES, RADIUS, K_ICE, RHO, CP, R_CONTACT,
                              total_time=diffusion_time, steps=400)
    lumped = chain_transient_lumped(N_PARTICLES, RADIUS, K_ICE, RHO, CP, R_CONTACT,
                                    total_time=diffusion_time, steps=400)
    fh = fvm["centre_history"]
    lh = lumped["node_history"]
    assert fh.shape == lh.shape
    error = float(np.max(np.abs(fh - lh)))
    assert error < 0.15, error
    # the fields must also be monotone in x at the end of the run
    final = fvm["centre_history"][-1]
    assert np.all(np.diff(final) < 0.0)


def test_ice_conductivity_matches_tabulated_values_and_is_monotone():
    for temperature, reference in ICE_CONDUCTIVITY_TABLE.items():
        assert ice_conductivity(temperature) == pytest.approx(reference, rel=0.05)
    colder = [ice_conductivity(t) for t in (0.0, -10.0, -20.0, -40.0)]
    assert colder == sorted(colder)
    with pytest.raises(ValueError):
        ice_conductivity(5.0)


def test_ice_bridge_resistance_matches_slab_solution():
    area, length = 4.9e-6, 0.01
    gate = ice_bond_gate(force_n=10.0, radius_m=RADIUS, temperature_c=-10.0)
    assert abs(gate["ice_bridge_relative_error"]) < 0.01
    direct = ice_bridge_resistance(-10.0, area, length)
    assert direct == pytest.approx(length / (ice_conductivity(-10.0) * area), rel=1e-12)


def test_ice_bond_gate_mechanics_and_thermal_limits():
    gate = ice_bond_gate(force_n=10.0, radius_m=RADIUS, temperature_c=-10.0,
                         mesh_cells_per_radius=16)
    assert gate["gate_pass"] is True
    assert gate["mesh_relative_error"] < 0.10
    assert gate["ice_bond_dominates"] is True
    assert 0.5 <= gate["parallel_bounds_ok"] or gate["parallel_bounds_ok"] is True
    # the ice path improves (resistance drops) as the ice cools
    warm = ice_bond_gate(force_n=10.0, radius_m=RADIUS, temperature_c=-0.5)["ice_bridge_resistance_k_w"]
    cold = ice_bond_gate(force_n=10.0, radius_m=RADIUS, temperature_c=-40.0)["ice_bridge_resistance_k_w"]
    assert cold < warm
    # a heavier contact flattens the patch: the dry path gets better
    light = ice_bond_gate(force_n=1.0, radius_m=RADIUS)["contact_resistance_k_w"]
    heavy = ice_bond_gate(force_n=100.0, radius_m=RADIUS)["contact_resistance_k_w"]
    assert heavy < light


def test_chain_inputs_are_validated():
    with pytest.raises(ValueError):
        hertz_patch_radius(-1.0, 0.005, 0.005, 9.0e9, 9.0e9)
    with pytest.raises(ValueError):
        rod_temperature_fvm(0, RADIUS, K_ICE, RHO, CP, R_CONTACT)
    with pytest.raises(ValueError):
        ice_bridge_resistance(-10.0, -1.0, 0.01)
    with pytest.raises(ValueError):
        chain_lumped_steady(1, RADIUS, K_ICE, RHO, CP, R_CONTACT)