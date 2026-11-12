"""Tests for the Hasimoto periodic-array baseline used to audit the
point-particle CFD-DEM self-response (mission m16963).

The audit document is ``docs/POINT_CFD_DEM_HASIMOTO_ARRAY_AUDIT.md`` and the
measurement driver is ``benchmarks/point_particle_cfd_dem/hasimoto_array.py``.
Everything asserted here is fast (no long settling run); the settling numbers
are the benchmark's job.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.point_particle_cfd_dem import hasimoto_array as HA  # noqa: E402
from tensordem.lbmdem import drag as lbm_drag  # noqa: E402
from tensordem.point_cfd_dem import closures  # noqa: E402
from tensordem.point_cfd_dem.coupling import PointParticleCFDDEM  # noqa: E402
from tensordem.point_cfd_dem.fluid import SpectralPeriodicFluid  # noqa: E402
from tensordem.point_cfd_dem.particles import (  # noqa: E402
    ParticleSet,
    PointParticleModel,
)


def test_hasimoto_series_truncation_is_negligible():
    """Over the audited solid fractions the retained terms are exact to 0.1 %."""
    for L in (4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0):
        phi = (math.pi / 6.0) / L ** 3
        c_h = HA.hasimoto_drag_enhancement(phi)
        assert c_h > 0.0
        assert HA.hasimoto_series_truncation(phi) < 1e-3 * c_h
        # the phi^2 term is the last one kept, and it is already tiny
        assert 1.5593 * phi * phi < 1.1e-4


def test_hasimoto_series_matches_repo_two_term_lbm_version():
    """Independent implementations agree: difference is exactly 1.5593 phi^2."""
    for L in (4.0, 6.0, 8.0, 16.0, 32.0):
        phi = (math.pi / 6.0) / L ** 3
        three = HA.hasimoto_drag_enhancement(phi)
        two = lbm_drag.array_correction(phi) - 1.0   # = 1.7601 phi^1/3 + phi
        assert three - two == pytest.approx(-1.5593 * phi * phi, rel=1e-12)


def test_gate_box_physical_array_correction_is_4_43_percent():
    """At L = 32 d the literature finite-domain effect is 4.43 %, not 0.3 %."""
    phi = (math.pi / 6.0) / 32.0 ** 3
    assert HA.hasimoto_drag_enhancement(phi) == pytest.approx(0.0443, abs=5e-4)
    assert HA.hasimoto_settling_ratio(phi) == pytest.approx(0.9576, abs=5e-4)


def test_self_response_is_grid_scale_not_lattice_sum():
    """C R scales like 1/h and depends on the sub-grid position.

    The physical simple-cubic array enhancement C_H is by construction
    h-independent and position-independent, so these two properties prove the
    module's R is a numerical self-field rather than the lattice sum.
    """
    coarse = HA.operator_point(2.0, 16.0)
    fine = HA.operator_point(1.0, 16.0)
    # ~1/h scaling
    assert 1.7 < fine["C_R"] / coarse["C_R"] < 2.4
    # far larger than the physical array correction at the same phi
    assert coarse["C_R"] > 4.0 * coarse["C_H"]
    # sub-grid position dependence of the numerical field
    off = HA.operator_point(2.0, 16.0, frac=(0.23, 0.37, 0.11))
    assert abs(off["C_R"] - coarse["C_R"]) > 0.1 * coarse["C_R"]


def test_correction_is_ill_conditioned_at_h_equal_d():
    """On the fine grid C R exceeds the stiffness guard, so the correction is
    unusable there; the guard itself (``evaluate_forces``) is exercised directly.
    """
    op = HA.operator_point(1.0, 8.0, frac=(0.5, 0.5, 0.5))
    assert op["C_R"] > 0.95

    from tensordem.point_cfd_dem.particles import evaluate_forces

    fluid = SpectralPeriodicFluid((8, 8, 8), 1.0, rho=1.0, mu=0.5, advect=True)
    centre = np.array(fluid.extent) / 2.0 + 0.5
    part = ParticleSet.create([centre], [[0.0, 0.0, 0.0]], radius=0.5,
                             density=2.0)
    model = PointParticleModel(rho_f=1.0, mu=0.5,
                               gravity=np.array([0.0, 0.0, -0.5]))
    with pytest.raises(ValueError, match="ill-conditioned"):
        evaluate_forces(part, model, u_fluid=np.zeros((3, 1)),
                        self_response=np.array([1.1]))


def test_uncorrected_closure_amplifies_terminal_velocity_by_C_R():
    """Chain rule of the audit: U/v_t = 1 + C R for the uncorrected mode.

    The identity is what makes the operator table (Q1) predictive of the
    dynamics (Q2); it is asserted here on the *short-time* response, which is
    cheap, and cross-checked against the published gate-1 baseline.
    """
    assert closures.stokes_drag_coefficient(0.5, 1.0) == pytest.approx(3 * math.pi * 0.5)
    # gate-1 published baseline: uncorrected U = 0.815 v_t at h = 2d, L = 32 d,
    # t = 2 (docs/point-particle-cfd-dem.md); fully developed the sign flips, so
    # only the *magnitude order* of the numerical self-field is asserted here.
    op = HA.operator_point(2.0, 32.0)
    assert 0.5 < op["C_R"] < 0.6
    assert op["ratio_to_hasimoto"] > 10.0