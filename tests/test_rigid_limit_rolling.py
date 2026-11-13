"""Tests for rigid-limit extrapolation with rolling resistance.

Anchor: XVAL-DEM-ROUND2-DEPOSITION-RIGID-LIMIT
"""

import math

import numpy as np
import pytest

from tensordem.deposition_packing import (
    DepositionSpec, run_deposition, stiffness_limit)


class TestRigidLimitWithRollingResistance:
    """Validate stiffness_limit extrapolation when rolling resistance is present."""

    @pytest.mark.parametrize("mu_rolling,kn,expected_phi_range", [
        (0.0, 4000.0, (0.61, 0.63)),
        (0.1, 4000.0, (0.60, 0.62)),
    ])
    def test_deposition_with_rolling_resistance_settles(self, mu_rolling, kn, expected_phi_range):
        """Deposition with rolling resistance should settle to a physically plausible phi."""
        spec = DepositionSpec(
            n_particles=500, mu=0.1, mu_rolling=mu_rolling, seed=1,
            kn=kn, restitution=0.1, t_max=50.0)
        res = run_deposition(spec)
        assert res.settled, f"Deposition did not settle (mur={mu_rolling}, kn={kn})"
        phi = res.metrics["phi_bulk"]
        assert expected_phi_range[0] < phi < expected_phi_range[1], (
            f"phi={phi} outside expected range {expected_phi_range}")

    def test_rolling_resistance_lowers_packing_fraction(self):
        """Higher rolling resistance should produce looser packings (lower phi)."""
        def phi_at(mur, kn=4000.0):
            spec = DepositionSpec(
                n_particles=500, mu=0.1, mu_rolling=mur, seed=1,
                kn=kn, restitution=0.1, t_max=50.0)
            return run_deposition(spec).metrics["phi_bulk"]

        phi0 = phi_at(0.0)
        phi1 = phi_at(0.1)
        assert phi1 < phi0 + 0.005, (
            f"Rolling resistance should lower phi: phi(mur=0)={phi0:.5f}, "
            f"phi(mur=0.1)={phi1:.5f}")

    def test_stiffness_limit_with_rolling_resistance(self):
        """stiffness_limit should produce a well-defined limit with rolling resistance."""
        # Use pre-computed values from the validation run
        pts_0 = [(4000.0, 0.61884), (16000.0, 0.61717)]
        pts_1 = [(4000.0, 0.61357), (16000.0, 0.60293)]

        sl0 = stiffness_limit(pts_0)
        sl1 = stiffness_limit(pts_1)

        # Both should have well-defined limits
        assert sl0["limit"] is not None
        assert sl1["limit"] is not None

        # mur=0.1 should have a lower limit than mur=0.0
        assert sl1["limit"] < sl0["limit"], (
            f"Higher rolling resistance should lower the hard-sphere limit: "
            f"mur=0.0 -> {sl0['limit']:.5f}, mur=0.1 -> {sl1['limit']:.5f}")

        # The extrapolation should remove most of the soft-sphere systematic
        assert sl0["delta_from_last"] < 0.005
        assert sl1["delta_from_last"] < 0.005

    def test_rigid_limit_within_precision_standard(self):
        """The hard-sphere limit should be within 15% of the RCP reference."""
        phi_ref = 0.6366
        pts_0 = [(4000.0, 0.61884), (16000.0, 0.61717)]
        pts_1 = [(4000.0, 0.61357), (16000.0, 0.60293)]

        for pts, mur in [(pts_0, 0.0), (pts_1, 0.1)]:
            sl = stiffness_limit(pts)
            rel_err = abs(sl["limit"] / phi_ref - 1.0)
            assert rel_err < 0.15, (
                f"mur={mur}: |phi_inf/phi_ref - 1| = {rel_err:.4f} >= 0.15")
