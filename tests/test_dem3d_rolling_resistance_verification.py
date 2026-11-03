"""Analytic verification of the DEM3D rolling-resistance couple (SIR-PC2|F05).

These tests bind the solver to closed-form references, not merely to "the
simulation runs".  The reference case is a single unbonded contact between a
fixed sphere and a free sphere with a prescribed overlap, so the normal force
is exactly ``k_c * overlap`` and the expected tangential resistance is
``(mu + mu_r) |F_n|``.

Historical context: before this increment the couple was built from the torque
and re-applied as ``force = torque / R``, i.e. ``f = n x f_res`` -- perpendicular
to the slip direction.  That version neither decelerated slip nor dissipated
energy; it pumped energy into the binormal direction.  ``test_legacy_negative_control``
keeps the pre-fix law alive as a negative control so the checks are known to have
teeth.

Verification report: ``docs/dem3d_rolling_resistance_verification.md``.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
import torch

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "verify_dem3d_rolling_resistance.py"
_spec = importlib.util.spec_from_file_location("verify_dem3d_rolling_resistance", _SCRIPT)
assert _spec is not None and _spec.loader is not None
vrr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vrr)


@pytest.fixture(scope="module")
def law():
    torch.set_num_threads(1)
    return vrr.check_law(mu_r=0.10, mu=0.20)


# ---------------------------------------------------------------------------
# Contact-law level
# ---------------------------------------------------------------------------
def test_single_contact_isolation():
    """The reference case must contain exactly one active contact pair."""
    sim, anchor, slider = vrr.build_single_contact(0.2, 0.1)
    assert vrr.contact_pairs_realised(sim) == 1
    assert vrr.normal_force(sim, anchor, slider) == pytest.approx(5.0, rel=1e-9)


def test_law_magnitude_is_mu_r_times_normal_force(law):
    """L1: |f_rr| = mu_r |F_n| exactly."""
    assert law["L1_rel_err"] < 1e-12
    assert law["L1_force_mag"] == pytest.approx(0.10 * 5.0, rel=1e-12)


def test_law_opposes_relative_slip(law):
    """L2: the force on the moving particle is anti-parallel to its slip."""
    assert law["L2_cos_to_slip"] == pytest.approx(-1.0, abs=1e-12)


def test_couple_conserves_linear_momentum(law):
    """L3: f_i + f_j = 0, so the couple cannot create net momentum."""
    assert law["L3_net_force"] < 1e-12


def test_torque_matches_declared_lever_and_ai_factor(law):
    """L4: |T| = mu_r R |F_n|; the Ai et al. (2011) equal-sphere R* = R/2
    convention is exactly a factor two below the declared lever arm."""
    assert law["L4_rel_err"] < 1e-12
    assert law["L4_impl_over_ai_factor"] == pytest.approx(2.0, rel=1e-12)


def test_additive_with_saturated_coulomb_spring(law):
    """L5: saturated sliding + rolling resistance = (mu + mu_r)|F_n|."""
    assert law["L5_rel_err"] < 1e-12
    assert law["L5_total_tangential"] == pytest.approx(0.30 * 5.0, rel=1e-12)


def test_rolling_disabled_is_bit_identical_to_friction_only(law):
    """L6: mu_r = 0 must not perturb the force field or the torque ledger."""
    assert law["L6_bit_identical_to_friction_only"] is True


# ---------------------------------------------------------------------------
# Integration level
# ---------------------------------------------------------------------------
def test_sliding_deceleration_matches_analytic():
    """I1-I4: a = mu_r |F_n| / m, no reversal, monotone energy, no transverse drift."""
    run = vrr.run_sliding(0.0, 0.10)
    assert run["a_rel_err"] < 0.03, run
    assert run["no_reversal"]
    assert run["ke_monotone"]
    assert run["max_abs_vy"] < 1e-6
    assert run["overlap_drift_rel"] < 1e-3


def test_sliding_plus_rolling_superposition():
    """I1 with both mechanisms active: a = (mu + mu_r) |F_n| / m within 3%."""
    run = vrr.run_sliding(0.20, 0.10)
    assert run["a_rel_err"] < 0.03, run
    assert run["max_abs_vy"] < 1e-6


def test_legacy_negative_control():
    """N1: the pre-fix orthogonal couple fails these checks.

    It leaves the slip almost undamped and injects the whole initial velocity
    into the binormal direction, so kinetic energy is not monotone.
    """
    legacy = vrr.run_sliding(0.0, 0.10, legacy=True)
    assert legacy["a_rel_err"] > 0.20
    assert legacy["max_abs_vy"] > 0.5 * legacy["v0"]
    assert not legacy["ke_monotone"]