"""Regression tests for the *dynamic* moving-plane tool-impulse reference.

The companion script ``scripts/verify_moving_plane_dynamic_impulse.py`` builds a
closed-form (modal) reference for the dynamic ``tool_impulse`` of
``tensordem.moving_plane_boundary.MovingPlaneNetwork`` and compares it with the
engine ledger.  These tests pin the parts that must never regress silently:

* the reference model really is the model the engine integrates
  (``K`` vs ``forces()``, eigen-residual, mass-orthonormality,
  closed-form ODE residual of the particular solution);
* the engine ledger agrees with the closed-form dynamic impulse to well below
  the 3 % acceptance bar, and the agreement improves under time-step refinement;
* the *quasi-static* reference (the only one available before this work) does
  **not** agree -- i.e. the check has teeth and is not passing by construction;
* the contact-set validity flag is honest: a fast ramp loses contact (the exact
  linear reference is then out of its envelope) and is reported as such.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from tensordem.moving_plane_boundary import MovingPlaneNetwork  # noqa: E402

import verify_moving_plane_dynamic_impulse as V  # noqa: E402


@pytest.fixture(scope="module")
def system():
    net = MovingPlaneNetwork(8, 2)
    return net, V.modal_system(net)


def test_reference_model_is_the_engine_model(system):
    """Assembled K reproduces forces(); eigen-pairs and the ODE residual are exact."""
    net, sy = system
    from verify_moving_plane_ledger import jacobian_check

    jac = jacobian_check(net)
    assert jac["rel"] < 1e-14
    assert sy["eig_residual"] < 1e-12
    assert sy["orth_residual"] < 1e-12
    # the polynomial particular solution satisfies p'' + w^2 p = c s identically
    for T in (0.05, 0.25, 0.5):
        ex = V.ExactResponse(sy, ramp=T)
        assert ex.residual(0.5 * T) < 1e-12 * max(1.0, float(np.abs(sy["c"]).max()))


def test_closed_form_impulse_matches_engine(system):
    """Dynamic tool_impulse vs the closed-form integral: well inside the 3 % bar."""
    _, sy = system
    for ramp, tol in ((0.5, 1e-6), (0.25, 1e-5)):
        run = V.run_engine(ramp, substeps=4)
        ex = V.ExactResponse(sy, ramp=ramp)
        rel = abs(run["tool_impulse"] - ex.impulse(run["t_end"])) / ex.impulse(run["t_end"])
        assert rel < tol, (ramp, rel)


def test_impulse_error_decreases_with_time_step(system):
    """Two-level refinement: the ledger error shrinks at (at least) ~2nd order."""
    _, sy = system
    ex = V.ExactResponse(sy, ramp=0.5)
    errs = []
    for s in (1, 2, 4):
        run = V.run_engine(0.5, substeps=s)
        ref = ex.impulse(run["t_end"])
        errs.append(abs(run["tool_impulse"] - ref) / ref)
    assert errs[0] > errs[1] > errs[2], errs
    order = np.log2(errs[0] / errs[1])
    assert order > 1.2, order


def test_quasi_static_reference_lacks_teeth(system):
    """The pre-existing static reference is off by >> 3 % at this ramp -- by design."""
    net, sy = system
    run = V.run_engine(0.5, substeps=2)
    i_qs = V.quasi_static_impulse(net, 0.5, run["t_end"], samples=201)
    rel_qs = abs(run["tool_impulse"] - i_qs) / i_qs
    ex = V.ExactResponse(sy, ramp=0.5)
    rel_ex = abs(run["tool_impulse"] - ex.impulse(run["t_end"])) / ex.impulse(run["t_end"])
    assert rel_qs > 100.0 * rel_ex  # the exact reference is orders of magnitude tighter


def test_contact_set_validity_flag(system):
    """Fast ramps lose contact; the reference must not be used there without saying so."""
    net, _ = system
    slow = V.run_engine(0.5, substeps=2)
    fast = V.run_engine(0.1, substeps=2)
    assert slow["min_gap"] > 0.0        # contact intact -> closed-form reference valid
    assert fast["min_gap"] < 0.0        # contact lost  -> exact linear reference invalid


def test_impulse_is_positive_and_finite(system):
    run = V.run_engine(0.5, substeps=1)
    assert np.isfinite(run["tool_impulse"])
    assert run["tool_impulse"] > 0.0