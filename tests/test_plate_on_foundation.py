"""Verification of the ice-sheet plate-on-elastic-foundation module.

Every tight-tolerance check is against an *independently* coded reference: the
Kelvin-function closed form, the Hankel-transform quadrature route, a global
equilibrium identity, or a published constant (Westergaard 1926 / Kerr 1979).
No check compares the module with itself.
"""
import math

import numpy as np
import pytest

from tensordem.plate_on_foundation import (
    GRAVITY,
    SEA_WATER_DENSITY,
    WESTERGAARD_CONSTANT,
    _hankel_integral,
    characteristic_length,
    foundation_modulus,
    ice_sheet_bearing_capacity,
    kelvin_kei,
    patch_centre_moment,
    patch_load_deflection,
    patch_load_moments,
    plate_rigidity,
    point_load_deflection,
    point_load_moments,
    westergaard_max_stress,
)

YOUNG = 5.0e9
THICK = 1.0
NU = 0.3
D = plate_rigidity(YOUNG, THICK, NU)
K = foundation_modulus()
L = characteristic_length(D, K)


# ------------------------------------------------------------------ mathematics
def test_kei_matches_classical_origin_value():
    kei, _, _ = kelvin_kei(np.array([0.0]))
    assert kei[0] == pytest.approx(-math.pi / 4.0, rel=1e-12)


def test_kei_second_derivative_satisfies_kelvin_ode():
    """``kei''`` from the Kelvin ODE must match a central difference of ``kei``."""
    for x in (0.5, 1.0, 2.0, 4.0):
        _, _, second = kelvin_kei(np.array([x]))
        h = 1e-4
        lo = kelvin_kei(np.array([x - h]))[0][0]
        mid = kelvin_kei(np.array([x]))[0][0]
        hi = kelvin_kei(np.array([x + h]))[0][0]
        numeric = (hi - 2.0 * mid + lo) / h**2
        assert second[0] == pytest.approx(numeric, rel=1e-5, abs=1e-6)


def test_point_load_centre_deflection_is_classical_identity():
    """``w(0) = P / (8 sqrt(k D))`` exactly (Hetenyi 1946)."""
    load = 1.0e5
    value = point_load_deflection(0.0, load=load, rigidity=D, modulus=K)[0]
    assert value == pytest.approx(load / (8.0 * math.sqrt(K * D)), rel=1e-12)
    assert value == pytest.approx(load * L**2 / (8.0 * D), rel=1e-12)


def test_point_load_moment_at_origin_sign_and_divergence():
    """Point-load moments are sagging and grow like ``ln(l/r)`` as ``r -> 0``."""
    small = point_load_moments(1e-6 * L, load=1.0e5, rigidity=D, modulus=K)[0][0]
    large = point_load_moments(1e-1 * L, load=1.0e5, rigidity=D, modulus=K)[0][0]
    assert small > 0.0 and small > large
    at_tenth = point_load_moments(0.1 * L, load=1.0e5, rigidity=D, modulus=K)[0][0]
    assert at_tenth > 0.0 and 4.0 * at_tenth < small < 8.0 * at_tenth  # ln(1/rho) growth


def test_point_load_global_equilibrium():
    """Foundation reaction integrates back to the applied point load."""
    radial = np.linspace(0.0, 60.0 * L, 240001)
    w = point_load_deflection(radial, load=1.0, rigidity=D, modulus=K)
    reaction = K * np.trapezoid(w * 2.0 * math.pi * radial, radial)
    assert reaction == pytest.approx(1.0, rel=1e-6)


def test_hankel_identity_for_kelvin_derivative():
    """``int s**2 J1(eps s)/(1+s**4) ds = kei'(eps)`` -- the bridge between routes."""
    for eps in (0.02, 0.1, 0.4):
        integral = -2.0 * _hankel_integral(eps, 0.0, "drr")
        reference = kelvin_kei(np.array([eps]))[1][0]
        assert integral == pytest.approx(reference, rel=1e-5)


def test_patch_centre_moment_two_independent_routes():
    """Closed form (ring superposition of the Kelvin field) vs Hankel quadrature."""
    for radius in (0.05 * L, 0.2 * L):
        closed = patch_centre_moment(load=1.0e5, radius=radius, rigidity=D, modulus=K)
        radial, hoop = patch_load_moments(
            0.0, load=1.0e5, radius=radius, rigidity=D, modulus=K)
        assert radial[0] == pytest.approx(closed, rel=1e-5)
        assert hoop[0] == pytest.approx(closed, rel=1e-5)


def test_patch_centre_state_is_equibiaxial():
    radial, hoop = patch_load_moments(0.0, load=1.0e5, radius=0.1 * L, rigidity=D, modulus=K)
    assert radial[0] == pytest.approx(hoop[0], rel=1e-6)


def test_patch_deflection_reaches_point_load_limit():
    """``radius -> 0`` at fixed load must converge to the exact point solution."""
    load = 1.0e5
    point = point_load_deflection(0.0, load=load, rigidity=D, modulus=K)[0]
    small = patch_load_deflection(
        0.0, load=load, radius=0.01 * L, rigidity=D, modulus=K)[0]
    assert small == pytest.approx(point, rel=1e-3)


def test_patch_load_global_equilibrium():
    """Foundation reaction of a patch load integrates back to the applied load."""
    load = 1.0e5
    # Far-field oscillatory Bessel quadrature limits this route to ~1 % (the
    # Kelvin point route holds the same identity to 1e-6, see the test above).
    radial = np.linspace(0.0, 30.0 * L, 121)
    w = patch_load_deflection(radial, load=load, radius=0.1 * L, rigidity=D, modulus=K)
    reaction = K * np.trapezoid(w * 2.0 * math.pi * radial, radial)
    assert reaction == pytest.approx(load, rel=1e-2)


# ------------------------------------------------------- published literature
def test_westergaard_constant_emerges_from_the_exact_solution():
    """``(2 l/a) kei'(a/l) - ln(l/a) -> 0.6159`` (Westergaard 1926 / Kerr 1979)."""
    constants = []
    for eps in (0.01, 0.005, 0.002):
        kei_prime = kelvin_kei(np.array([eps]))[1][0]
        constants.append(2.0 / eps * kei_prime - math.log(1.0 / eps))
    assert constants[0] == pytest.approx(WESTERGAARD_CONSTANT, abs=1e-3)
    assert constants[-1] == pytest.approx(WESTERGAARD_CONSTANT, abs=1e-4)
    assert constants[0] > constants[1] > constants[2]  # monotone approach


def test_published_asymptote_reproduces_exact_peak_stress():
    """The published formula must match the exact patch solution to well inside 3 %."""
    for radius_over_l in (0.01, 0.03, 0.1):
        radius = radius_over_l * L
        exact = 6.0 * patch_centre_moment(
            load=1.0, radius=radius, rigidity=D, modulus=K) / THICK**2
        published = westergaard_max_stress(
            1.0, radius=radius, thickness=THICK, young_modulus=YOUNG,
            poisson_ratio=NU, modulus=K)
        assert abs(published - exact) / exact < 0.03


# ------------------------------------------------------------- bearing capacity
def test_bearing_capacity_scales_with_thickness_squared():
    thin = ice_sheet_bearing_capacity(thickness=0.5)
    thick = ice_sheet_bearing_capacity(thickness=1.0)
    assert thin["characteristic_length_m"] == pytest.approx(
        (plate_rigidity(YOUNG, 0.5, NU) / (SEA_WATER_DENSITY * GRAVITY)) ** 0.25, rel=1e-12)
    ratio = thick["allowable_load_N"] / thin["allowable_load_N"]
    assert 3.0 < ratio < 4.5  # h^2 scaling plus the log(l/a) correction


def test_bearing_capacity_reference_numbers():
    """Snapshot of the documented 1 m ice-sheet case (0.5 MPa flexural strength)."""
    sheet = ice_sheet_bearing_capacity(thickness=1.0)
    assert sheet["characteristic_length_m"] == pytest.approx(14.6079, rel=1e-4)
    assert sheet["allowable_load_N"] == pytest.approx(155_100.0, rel=1e-2)
    assert abs(sheet["westergaard_relative_error"]) < 0.03
    shallow = ice_sheet_bearing_capacity(thickness=0.3)
    assert shallow["allowable_load_N"] == pytest.approx(16_900.0, rel=2e-2)
    assert shallow["thickness_over_l"] < 0.1  # thin-plate regime holds


def test_hydrostatic_foundation_modulus():
    assert foundation_modulus() == pytest.approx(SEA_WATER_DENSITY * GRAVITY)


# ------------------------------------------------------------------- validation
def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        plate_rigidity(YOUNG, -1.0, NU)
    with pytest.raises(ValueError):
        plate_rigidity(YOUNG, THICK, 0.6)
    with pytest.raises(ValueError):
        characteristic_length(0.0, K)
    with pytest.raises(ValueError):
        point_load_moments(0.0, load=1.0, rigidity=D, modulus=K)
    with pytest.raises(ValueError):
        kelvin_kei(np.array([-0.5]))
    with pytest.raises(ValueError):
        westergaard_max_stress(1.0, radius=2.0 * L, thickness=THICK,
                               young_modulus=YOUNG, poisson_ratio=NU, modulus=K)
    with pytest.raises(TypeError):
        foundation_modulus(water_density="sea")