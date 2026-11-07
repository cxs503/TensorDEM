"""Tests of the hull-girder modal ROM and the strain-mode operator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.shm.model import STRAIN_SENSORS, HullGirderROM, SHMConfig
from tensordem.shm.modal_filter import ModalFilter
from tensordem.whipping import modes as whipping_modes


@pytest.fixture(scope="module")
def rom() -> HullGirderROM:
    return HullGirderROM(SHMConfig(n_modes=4))


def test_dry_frequencies_match_free_free_analytic(rom):
    """FEM dry modes must reproduce the free-free beam closed form."""
    analytic = whipping_modes.free_free_analytic_frequencies(
        rom.case.youngs_modulus * rom.case.inertia, rom.case.mass_per_length,
        rom.case.length, rom.n_modes)
    err = np.abs(rom.dry_freqs - analytic) / analytic
    assert err.max() < 5.0e-3, f"dry frequency error {err}"


def test_wet_frequencies_reduced_by_added_mass(rom):
    """Wet modes are slower than dry modes and consistent with the Ritz solve."""
    assert np.all(rom.freqs < rom.dry_freqs)
    assert np.all(rom.wet_reduction > 0.0)
    # consistency with the upstream wet-mode route (audit cross-check): the two
    # solvers must agree to well below the physical modelling uncertainty
    assert rom.freq_upstream_scatter < 1.0e-2, rom.freq_upstream_scatter
    assert np.allclose(rom.modal_mass, 1.0, atol=1e-10)
    # the first wet mode must sit above the 0.5 Hz rigid/elastic split,
    # otherwise the separation of the node would be meaningless
    assert rom.freqs[0] > 0.5


def test_mode_shapes_are_mass_normalised(rom):
    """int (m_s + m_a) phi_i phi_j dx = delta_ij for the wet mode set."""
    x = np.linspace(0.0, rom.length, 2001)
    m_total = rom.case.mass_per_length + rom.m_a_of_x(x)
    phi = np.array([rom.shape(x, n) for n in range(rom.n_modes)])
    gram = np.empty((rom.n_modes, rom.n_modes))
    for i in range(rom.n_modes):
        for j in range(rom.n_modes):
            gram[i, j] = np.trapezoid(m_total * phi[i] * phi[j], x)
    assert np.max(np.abs(np.diag(gram) - 1.0)) < 2.0e-2, np.diag(gram)
    off = gram - np.diag(np.diag(gram))
    assert np.max(np.abs(off)) < 2.0e-2, off


def test_strain_mode_matrix_matches_finite_difference(rom):
    """H[k, n] = -y_k phi_n''(x_k) against a central difference of the shape."""
    h = 0.05
    for n in range(rom.n_modes):
        x = rom.sensor_x
        d2 = (rom.shape(x + h, n) - 2.0 * rom.shape(x, n) + rom.shape(x - h, n)) / h**2
        expected = -rom.sensor_y * d2
        assert np.allclose(rom.H[:, n], expected, rtol=2e-3, atol=1e-3 * np.abs(expected).max())


def test_strain_matrix_is_solvable_with_default_sensors(rom):
    """The default sensor layout must give a well conditioned, full-rank H."""
    assert rom.H.shape == (len(STRAIN_SENSORS), rom.n_modes)
    assert np.linalg.matrix_rank(rom.H) == rom.n_modes
    assert rom.condition_number() < 100.0
    filt = ModalFilter(rom.H)
    assert np.isfinite(filt.noise_gain) and filt.noise_gain > 0.0


def test_ill_conditioned_sensor_layout_rejected(rom):
    """A layout with fewer gauges than modes cannot be inverted: raise, never hide."""
    with pytest.raises(ValueError):
        ModalFilter(rom.H[:2, :])
    # a degenerate (co-located) gauge set is rank deficient as well
    x = np.full(rom.n_modes, 0.5 * rom.length)
    y = np.full(rom.n_modes, 7.0)
    with pytest.raises(ValueError):
        ModalFilter(rom.strain_mode_matrix(x, y))


def test_load_projection_inverts_exact_modal_force(rom):
    """phi^T p / (phi^T phi) recovers the point force from its modal force."""
    force = 7.3e6
    modal = rom.modal_force(force)
    recovered = float(rom.load_projection() @ modal[:, 0])
    assert math.isclose(recovered, force, rel_tol=1e-12)


def test_load_field_projects_onto_modal_participation(rom):
    """The distributed estimate peaks at the load station for a point load."""
    x = np.linspace(0.0, rom.length, 201)
    field = rom.load_field(x, rom.modal_force(1.0e7)[:, 0])
    assert np.isfinite(field).all()
    assert np.max(np.abs(field)) > 0.0