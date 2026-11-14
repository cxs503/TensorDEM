"""Tests for friction-dependent RCP reference comparison in the aggregate gates.

This module verifies that the G12/G13/G18 gates in
``benchmarks/case_rcp_deposition/aggregate.py`` correctly use friction-dependent
references from ``tensordem.rcp_references`` rather than comparing all tiers
against the frictionless reference (a category error that was fixed in m20258).

The bug: G13 compared the mu=0.1 extrapolated limit (≈0.617) against the
frictionless reference (0.6366), yielding a 3% deviation that failed the gate.
The fix: G13 now compares against the mu=0.1 reference (0.6250), yielding
a 1.2% deviation that passes.
"""
import pytest

from tensordem.rcp_references import get_reference
from tensordem.deposition_packing import stiffness_limit


# ---------------------------------------------------------------------------
# Reference anchors
# ---------------------------------------------------------------------------

def test_frictionless_reference_is_06366():
    ref = get_reference(0.0)
    assert ref.phi_rcp == pytest.approx(0.6366, abs=1e-4)


def test_low_friction_reference_is_0625():
    ref = get_reference(0.1)
    assert ref.phi_rcp == pytest.approx(0.6250, abs=1e-4)


# ---------------------------------------------------------------------------
# Gate logic: friction-dependent comparison
# ---------------------------------------------------------------------------

class TestFrictionDependentGates:
    """Reproduce the gate logic from aggregate.py with known inputs."""

    TOL_REL = 0.03  # 3%

    def _gate_pass(self, phi_measured, mu):
        """Reproduce the G12/G13/G18 gate logic."""
        ref = get_reference(mu)
        dev = abs(phi_measured / ref.phi_rcp - 1.0)
        return dev < self.TOL_REL, dev, ref.phi_rcp

    def test_mu0_production_passes(self):
        passed, dev, ref = self._gate_pass(0.64237, 0.0)
        assert passed, f"mu=0 production should pass: dev={dev:.4f}, ref={ref}"
        assert dev < 0.01

    def test_mu01_production_passes_friction_ref(self):
        passed, dev, ref = self._gate_pass(0.62628, 0.1)
        assert passed, f"mu=0.1 production should pass vs friction ref: dev={dev:.4f}, ref={ref}"
        assert dev < 0.005

    def test_mu01_vs_frictionless_ref_is_larger(self):
        """mu=0.1 vs frictionless ref has larger deviation than vs friction ref."""
        ref_frictionless = get_reference(0.0)
        ref_friction = get_reference(0.1)
        dev_fl = abs(0.62628 / ref_frictionless.phi_rcp - 1.0)
        dev_f = abs(0.62628 / ref_friction.phi_rcp - 1.0)
        assert dev_fl > dev_f, (
            f"mu=0.1 vs frictionless ref ({dev_fl:.4f}) should be larger "
            f"than vs friction ref ({dev_f:.4f})"
        )

    def test_mu0_hard_sphere_limit_passes(self):
        passed, dev, ref = self._gate_pass(0.63535, 0.0)
        assert passed, f"mu=0 limit should pass: dev={dev:.4f}"
        assert dev < 0.005

    def test_mu01_hard_sphere_limit_passes_friction_ref(self):
        passed, dev, ref = self._gate_pass(0.61739, 0.1)
        assert passed, f"mu=0.1 limit should pass vs friction ref: dev={dev:.4f}, ref={ref}"
        assert dev < 0.015

    def test_mu01_limit_vs_frictionless_is_larger(self):
        """mu=0.1 limit vs frictionless ref has larger deviation."""
        ref_frictionless = get_reference(0.0)
        ref_friction = get_reference(0.1)
        dev_fl = abs(0.61739 / ref_frictionless.phi_rcp - 1.0)
        dev_f = abs(0.61739 / ref_friction.phi_rcp - 1.0)
        assert dev_fl > dev_f, (
            f"mu=0.1 limit vs frictionless ref ({dev_fl:.4f}) should be larger "
            f"than vs friction ref ({dev_f:.4f})"
        )


# ---------------------------------------------------------------------------
# Stiffness extrapolation sanity checks
# ---------------------------------------------------------------------------

class TestStiffnessExtrapolation:
    """Verify the Richardson extrapolation used in the gates."""

    def test_mu0_extrapolation_from_runs(self):
        pts = [
            (4000.0, 0.64261),
            (16000.0, 0.63791),
            (64000.0, 0.63680),
            (256000.0, 0.63571),
        ]
        result = stiffness_limit(pts)
        assert result["limit"] == pytest.approx(0.63535, abs=0.001)
        # order may be None if deltas are not monotonic; order_used defaults to 1.0
        assert result["order_used"] == 1.0
        assert result["n_levels"] == 4

    def test_mu01_extrapolation_from_runs(self):
        pts = [
            (4000.0, 0.62671),
            (16000.0, 0.62159),
            (64000.0, 0.61928),
        ]
        result = stiffness_limit(pts)
        assert result["limit"] == pytest.approx(0.6174, abs=0.001)
        assert result["n_levels"] == 3
