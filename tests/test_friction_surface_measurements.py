"""Measurement-layer tests: heap angle, contact stress, Mohr-Coulomb, binning."""
import math

import numpy as np
import pytest

from tensordem.friction_surface.repose import heap_surface_angle
from tensordem.friction_surface.shear import (
    PeriodicShearCase, bin_history, build_specimen, mohr_coulomb_fit,
    solid_fraction, stress_tensor,
)


def test_heap_angle_of_a_synthetic_cone_with_known_slope():
    """A synthetic cone of slope ``tan(30 deg)`` must be measured as 30 deg."""
    rng = np.random.default_rng(0)
    n = 20000
    radius = 1.0e-3
    rho = 0.06 * np.sqrt(rng.random(n))
    # a heap: the surface falls away from the apex at the centre
    z = 0.06 * math.tan(math.radians(30.0)) - rho * math.tan(math.radians(30.0))
    ang = rng.random(n) * 2 * math.pi
    pos = np.column_stack([rho * np.cos(ang), rho * np.sin(ang), z])
    measured = heap_surface_angle(pos, radius, n_bins=26)
    assert abs(measured["angle_deg"] - 30.0) < 2.0, measured


def test_heap_angle_of_a_flat_layer_is_near_zero():
    rng = np.random.default_rng(1)
    n = 8000
    rho = 0.05 * np.sqrt(rng.random(n))
    ang = rng.random(n) * 2 * math.pi
    pos = np.column_stack([rho * np.cos(ang), rho * np.sin(ang),
                           np.full(n, 1.0e-3)])
    measured = heap_surface_angle(pos, 1.0e-3, n_bins=20)
    assert abs(measured["angle_deg"]) < 3.0


def test_contact_stress_of_a_single_axial_pair():
    """One compressive pair along z carries exactly ``sigma = f/(2A)`` per area."""
    area = 1.0e-4
    volume = 1.0e-6
    contacts = {"i": np.array([0]), "j": np.array([1]),
                "branch": np.array([[0.0, 0.0, 2.0e-3]]),
                "f_total": np.array([[0.0, 0.0, 1.0]])}
    s = stress_tensor(contacts, volume)
    assert s[2, 2] == pytest.approx(1.0 * 2.0e-3 / volume)
    assert s[0, 2] == pytest.approx(0.0, abs=1e-12)


def test_mohr_coulomb_fit_recovers_a_known_envelope():
    sig = np.array([2000.0, 5000.0, 10000.0, 20000.0])
    tau = 800.0 + sig * 0.42
    fit = mohr_coulomb_fit(sig, tau)
    assert fit["cohesion_pa"] == pytest.approx(800.0, rel=1e-6)
    assert fit["tan_phi"] == pytest.approx(0.42, rel=1e-6)
    assert fit["max_relative_deviation"] < 1e-9
    assert fit["r_squared"] > 0.999999


def test_bin_history_averages_within_bins():
    strains = np.array([0.0, 0.005, 0.012, 0.016, 0.030])
    taus = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    sigmas = np.array([100.0, 100.0, 200.0, 200.0, 300.0])
    b = bin_history(strains, taus, sigmas, width=0.01)
    # bins with a single sample are dropped: [0.000,0.005] and [0.012,0.016] stay
    assert b["tau"].size == 2
    assert b["tau"][0] == pytest.approx(15.0)
    assert b["sigma"][0] == pytest.approx(100.0)
    assert b["tau"][1] == pytest.approx(35.0)


def test_shear_cell_geometry_is_commensurate_and_dense():
    """The cell must be an exact multiple of the lattice pitch (no wrap overlaps)."""
    case = PeriodicShearCase(nx=6, ny=6, nz=4)
    pos = build_specimen(case)
    assert pos.shape[0] == 6 * 6 * 4 * 2
    assert solid_fraction(case, pos) == pytest.approx(0.7405 / (1.05 ** 3), rel=0.02)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 2.0 * case.radius
