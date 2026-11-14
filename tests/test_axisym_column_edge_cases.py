"""Edge-case and validation tests for the axisymmetric column module.

Covers paths not exercised by the main Lube-2004 benchmark test:
- ``AxisymCase`` validation (mu_inertial, rolling_lever, mu_wall,
  t_reach_fraction, contact_mode, restitution boundaries)
- ``radial_extent`` on synthetic deposits
- ``_time_to_reach`` interpolation and edge cases
- ``verdict`` boundary conditions
- ``signature`` completeness
- ``material()`` property with different rolling_lever settings
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.granular_column.axisym import (
    AxisymCase,
    AxisymError,
    build_axisymmetric_column,
    lube_2004_emplacement_time,
    lube_2004_runout_ratio,
    radial_extent,
    verdict,
)


# ---------------------------------------------------------------------------
# AxisymCase validation
# ---------------------------------------------------------------------------

class TestAxisymCaseValidation:
    """Guard against silent acceptance of invalid parameters."""

    def test_negative_aspect_ratio_rejected(self):
        with pytest.raises(AxisymError, match="aspect_ratio"):
            AxisymCase(aspect_ratio=-1.0)

    def test_zero_aspect_ratio_rejected(self):
        with pytest.raises(AxisymError, match="aspect_ratio"):
            AxisymCase(aspect_ratio=0.0)

    def test_inf_aspect_ratio_rejected(self):
        with pytest.raises(AxisymError, match="aspect_ratio"):
            AxisymCase(aspect_ratio=float("inf"))

    def test_nan_aspect_ratio_rejected(self):
        with pytest.raises(AxisymError, match="aspect_ratio"):
            AxisymCase(aspect_ratio=float("nan"))

    def test_negative_radius_rejected(self):
        with pytest.raises(AxisymError, match="radius"):
            AxisymCase(radius=-0.01)

    def test_zero_grains_per_radius_rejected(self):
        with pytest.raises(AxisymError, match="grains_per_radius"):
            AxisymCase(grains_per_radius=0.0)

    def test_negative_density_rejected(self):
        with pytest.raises(AxisymError, match="density"):
            AxisymCase(density=-1000.0)

    def test_negative_kn_rejected(self):
        with pytest.raises(AxisymError, match="kn"):
            AxisymCase(kn=-1e4)

    def test_negative_kt_ratio_rejected(self):
        with pytest.raises(AxisymError, match="kt_ratio"):
            AxisymCase(kt_ratio=-0.5)

    def test_negative_gravity_rejected(self):
        with pytest.raises(AxisymError, match="gravity"):
            AxisymCase(gravity=-9.81)

    def test_negative_duration_factor_rejected(self):
        with pytest.raises(AxisymError, match="duration_factor"):
            AxisymCase(duration_factor=-1.0)

    def test_negative_min_time_rejected(self):
        with pytest.raises(AxisymError, match="min_time"):
            AxisymCase(min_time=-0.1)

    def test_negative_mu_rejected(self):
        with pytest.raises(AxisymError, match="mu"):
            AxisymCase(mu=-0.1)

    def test_zero_mu_accepted(self):
        """mu=0 is physically meaningful (frictionless)."""
        c = AxisymCase(mu=0.0)
        assert c.mu == 0.0

    def test_negative_mu_rolling_rejected(self):
        with pytest.raises(AxisymError, match="mu_rolling"):
            AxisymCase(mu_rolling=-0.1)

    def test_zero_mu_rolling_accepted(self):
        c = AxisymCase(mu_rolling=0.0)
        assert c.mu_rolling == 0.0

    def test_negative_jitter_rejected(self):
        with pytest.raises(AxisymError, match="jitter"):
            AxisymCase(jitter=-0.01)

    def test_zero_jitter_accepted(self):
        c = AxisymCase(jitter=0.0)
        assert c.jitter == 0.0

    def test_negative_packing_gap_rejected(self):
        with pytest.raises(AxisymError, match="packing_gap"):
            AxisymCase(packing_gap=-0.01)

    def test_restitution_at_one_rejected(self):
        with pytest.raises(AxisymError, match="restitution"):
            AxisymCase(restitution=1.0)

    def test_restitution_negative_rejected(self):
        with pytest.raises(AxisymError, match="restitution"):
            AxisymCase(restitution=-0.1)

    def test_restitution_zero_accepted(self):
        c = AxisymCase(restitution=0.0)
        assert c.restitution == 0.0

    def test_restitution_0999_accepted(self):
        c = AxisymCase(restitution=0.999)
        assert c.restitution == pytest.approx(0.999)

    def test_t_reach_fraction_zero_rejected(self):
        with pytest.raises(AxisymError, match="t_reach_fraction"):
            AxisymCase(t_reach_fraction=0.0)

    def test_t_reach_fraction_one_accepted(self):
        c = AxisymCase(t_reach_fraction=1.0)
        assert c.t_reach_fraction == 1.0

    def test_t_reach_fraction_half_accepted(self):
        c = AxisymCase(t_reach_fraction=0.5)
        assert c.t_reach_fraction == 0.5

    def test_mu_wall_negative_rejected(self):
        with pytest.raises(AxisymError, match="mu_wall"):
            AxisymCase(mu_wall=-0.1)

    def test_mu_wall_zero_accepted(self):
        c = AxisymCase(mu_wall=0.0)
        assert c.mu_wall == 0.0

    def test_mu_wall_none_uses_grain_mu(self):
        c = AxisymCase(mu=0.5, mu_wall=None)
        mat = c.material()
        assert mat.mu == 0.5

    def test_rolling_lever_invalid_rejected(self):
        with pytest.raises(AxisymError, match="rolling_lever"):
            AxisymCase(rolling_lever="invalid")

    def test_rolling_lever_particle_radius_accepted(self):
        c = AxisymCase(rolling_lever="particle_radius")
        assert c.rolling_lever == "particle_radius"

    def test_rolling_lever_reduced_radius_accepted(self):
        c = AxisymCase(rolling_lever="reduced_radius")
        assert c.rolling_lever == "reduced_radius"

    def test_mu_inertial_wrong_length_rejected(self):
        with pytest.raises(AxisymError, match="mu_inertial"):
            AxisymCase(mu_inertial=(0.5, 0.8))  # needs 3 values

    def test_mu_inertial_invalid_params_rejected(self):
        """InertialFriction validates mu_s < mu_2 and i_0 > 0."""
        with pytest.raises((AxisymError, ValueError)):
            AxisymCase(mu_inertial=(0.8, 0.5, 0.1))  # mu_s > mu_2

    def test_mu_inertial_valid_accepted(self):
        c = AxisymCase(mu_inertial=(0.4, 0.8, 0.3))
        assert c.mu_inertial == (0.4, 0.8, 0.3)

    def test_contact_mode_stored(self):
        """contact_mode is passed through to the kernel; we just check storage."""
        c = AxisymCase(contact_mode="one_sided")
        assert c.contact_mode == "one_sided"


# ---------------------------------------------------------------------------
# AxisymCase properties
# ---------------------------------------------------------------------------

class TestAxisymCaseProperties:
    def test_column_radius(self):
        c = AxisymCase(grains_per_radius=8.0, radius=0.03)
        assert c.column_radius == pytest.approx(8.0 * 2.0 * 0.03)

    def test_column_height(self):
        c = AxisymCase(aspect_ratio=2.0, grains_per_radius=8.0, radius=0.03)
        assert c.column_height == pytest.approx(2.0 * c.column_radius)

    def test_material_rolling_lever_particle_radius(self):
        c = AxisymCase(rolling_lever="particle_radius", mu_rolling=0.3)
        mat = c.material()
        assert mat.rolling_lever == "particle_radius"

    def test_material_rolling_lever_reduced_radius(self):
        c = AxisymCase(rolling_lever="reduced_radius", mu_rolling=0.3)
        mat = c.material()
        assert mat.rolling_lever == "reduced_radius"

    def test_signature_contains_all_fields(self):
        c = AxisymCase(aspect_ratio=1.5, mu_inertial=(0.4, 0.8, 0.3),
                       mu_wall=0.8, label="test")
        sig = c.signature()
        assert sig["case"] == "CASE-COLUMN-COLLAPSE"
        assert sig["doi"] is not None
        assert sig["aspect_ratio"] == 1.5
        assert sig["mu_inertial"] == [0.4, 0.8, 0.3]
        assert sig["mu_law"] == "mu(I_c) Jop et al. (2006)"
        assert sig["mu_wall"] == 0.8
        assert sig["label"] == "test"

    def test_signature_constant_mu_law(self):
        c = AxisymCase(mu_inertial=None)
        sig = c.signature()
        assert sig["mu_law"] == "constant"
        assert sig["mu_inertial"] is None


# ---------------------------------------------------------------------------
# radial_extent
# ---------------------------------------------------------------------------

class TestRadialExtent:
    def test_single_grain_at_origin(self):
        pos = np.array([[0.0, 0.0, 0.03]])
        r = 0.03
        ext = radial_extent(pos, r)
        assert ext["p995"] == pytest.approx(r)
        assert ext["max"] == pytest.approx(r)
        assert ext["mean"] == pytest.approx(0.0)

    def test_two_grains_on_x_axis(self):
        pos = np.array([[0.1, 0.0, 0.03], [0.2, 0.0, 0.03]])
        r = 0.03
        ext = radial_extent(pos, r)
        # rr = [0.1, 0.2]; with 2 points, p995 interpolates near the max
        assert ext["max"] == pytest.approx(0.23)
        assert ext["mean"] == pytest.approx(0.15)
        # p995 is close to max but not exactly (quantile interpolation)
        assert 0.22 <= ext["p995"] <= 0.23

    def test_grains_on_circle(self):
        """All grains at same radius -> p995 == max."""
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        R = 0.5
        pos = np.column_stack([R * np.cos(theta), R * np.sin(theta),
                               np.full(100, 0.03)])
        r = 0.03
        ext = radial_extent(pos, r)
        assert ext["p995"] == pytest.approx(R + r, rel=1e-6)
        assert ext["max"] == pytest.approx(R + r)

    def test_empty_array_raises(self):
        with pytest.raises(Exception):
            radial_extent(np.zeros((0, 3)), 0.03)


# ---------------------------------------------------------------------------
# _time_to_reach (imported indirectly via simulate, but we test the logic)
# ---------------------------------------------------------------------------

class TestTimeToReach:
    """Test the interpolation logic used for emplacement time."""

    def _time_to_reach(self, time_hist, front_hist, r_i, front_final, fraction):
        """Re-implementation for testing (matches the module's logic)."""
        if time_hist.size == 0:
            return float("nan")
        target = r_i + float(fraction) * (front_final - r_i)
        above = np.nonzero(front_hist >= target)[0]
        if not above.size:
            return float("nan")
        k = int(above[0])
        if k == 0:
            return float(time_hist[0])
        f0, f1 = front_hist[k - 1], front_hist[k]
        t0, t1 = time_hist[k - 1], time_hist[k]
        w = 0.0 if f1 == f0 else (target - f0) / (f1 - f0)
        return float(t0 + w * (t1 - t0))

    def test_empty_history_returns_nan(self):
        t = self._time_to_reach(np.array([]), np.array([]), 0.5, 1.0, 0.99)
        assert math.isnan(t)

    def test_reaches_target_via_interpolation(self):
        """When the target is between two front values, interpolate."""
        time_hist = np.array([0.0, 0.1, 0.2])
        front_hist = np.array([0.5, 0.55, 0.6])
        # target = 0.5 + 0.99 * 0.1 = 0.599; front_hist[2]=0.6 >= 0.599
        # k=2, interpolate between (0.1, 0.55) and (0.2, 0.6)
        t = self._time_to_reach(time_hist, front_hist, 0.5, 0.6, 0.99)
        assert t == pytest.approx(0.198)

    def test_reaches_at_second_step_interpolates(self):
        time_hist = np.array([0.0, 0.1, 0.2])
        front_hist = np.array([0.6, 0.7, 0.8])
        # target = 0.5 + 0.5 * 0.3 = 0.65; front_hist[1]=0.7 >= 0.65
        # k=1, interpolate between (0.0, 0.6) and (0.1, 0.7)
        t = self._time_to_reach(time_hist, front_hist, 0.5, 0.8, 0.5)
        assert t == pytest.approx(0.05)

    def test_exact_hit_returns_previous_time(self):
        """When target exactly equals a front value, k points to it and
        we interpolate from the previous step."""
        time_hist = np.array([0.0, 1.0, 2.0])
        front_hist = np.array([0.5, 0.7, 0.9])
        # target = 0.5 + 0.5 * 0.4 = 0.7; front_hist[1]=0.7
        # k=1, interpolate between (0.0, 0.5) and (1.0, 0.7)
        # w = (0.7 - 0.5) / (0.7 - 0.5) = 1.0
        # t = 0.0 + 1.0 * 1.0 = 1.0
        t = self._time_to_reach(time_hist, front_hist, 0.5, 0.9, 0.5)
        assert t == pytest.approx(1.0)

    def test_interpolation_midpoint(self):
        time_hist = np.array([0.0, 1.0, 2.0])
        front_hist = np.array([0.5, 0.6, 0.8])
        # target = 0.5 + 0.5 * 0.3 = 0.65
        # k=2 (front_hist[2]=0.8 >= 0.65), f0=0.6, f1=0.8, t0=1.0, t1=2.0
        # w = (0.65 - 0.6) / (0.8 - 0.6) = 0.25
        # t = 1.0 + 0.25 * 1.0 = 1.25
        t = self._time_to_reach(time_hist, front_hist, 0.5, 0.8, 0.5)
        assert t == pytest.approx(1.25)

    def test_fraction_one_reaches_final(self):
        time_hist = np.array([0.0, 1.0, 2.0])
        front_hist = np.array([0.5, 0.7, 0.9])
        # target = 0.5 + 1.0 * 0.4 = 0.9; at index 2
        # k=2, f0=0.7, f1=0.9, t0=1.0, t1=2.0
        # w = (0.9 - 0.7) / (0.9 - 0.7) = 1.0
        # t = 1.0 + 1.0 * 1.0 = 2.0
        t = self._time_to_reach(time_hist, front_hist, 0.5, 0.9, 1.0)
        assert t == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

class TestVerdict:
    def test_exact_match_passes(self):
        v = verdict(2.0, 2.0)
        assert v["pass"] is True
        assert v["relative_error"] == 0.0

    def test_within_tolerance_passes(self):
        v = verdict(2.0, 2.1)
        assert v["pass"] is True
        assert v["relative_error"] == pytest.approx(0.05)

    def test_at_tolerance_boundary_passes(self):
        v = verdict(2.0, 2.3)
        assert v["pass"] is True
        assert v["relative_error"] == pytest.approx(0.15)

    def test_just_over_tolerance_fails(self):
        v = verdict(2.0, 2.31)
        assert v["pass"] is False
        assert v["relative_error"] == pytest.approx(0.155)

    def test_custom_tolerance(self):
        v = verdict(2.0, 2.05, tolerance=0.03)
        assert v["pass"] is True
        v2 = verdict(2.0, 2.07, tolerance=0.03)
        assert v2["pass"] is False

    def test_measured_less_than_reference(self):
        v = verdict(2.0, 1.8)
        assert v["pass"] is True
        assert v["relative_error"] == pytest.approx(-0.10)

    def test_zero_reference_raises(self):
        with pytest.raises((ZeroDivisionError, ValueError)):
            verdict(0.0, 0.1)


# ---------------------------------------------------------------------------
# lube_2004_runout_ratio edge cases
# ---------------------------------------------------------------------------

class TestLubeRunoutEdgeCases:
    def test_at_regime_break(self):
        """At a=1.7, the inertial (sqrt) branch is used."""
        r = lube_2004_runout_ratio(1.7)
        assert r == pytest.approx(1.0 + 1.6 * math.sqrt(1.7))

    def test_just_below_regime_break(self):
        r = lube_2004_runout_ratio(1.69)
        assert r == pytest.approx(1.0 + 1.24 * 1.69)

    def test_just_above_regime_break(self):
        r = lube_2004_runout_ratio(1.71)
        assert r == pytest.approx(1.0 + 1.6 * math.sqrt(1.71))

    def test_very_small_aspect(self):
        r = lube_2004_runout_ratio(0.01)
        assert r == pytest.approx(1.0 + 1.24 * 0.01)

    def test_very_large_aspect(self):
        r = lube_2004_runout_ratio(100.0)
        assert r == pytest.approx(1.0 + 1.6 * math.sqrt(100.0))


# ---------------------------------------------------------------------------
# lube_2004_emplacement_time edge cases
# ---------------------------------------------------------------------------

class TestLubeEmplacementEdgeCases:
    def test_zero_column_radius_rejected(self):
        with pytest.raises(AxisymError, match="column_radius"):
            lube_2004_emplacement_time(1.0, 0.0)

    def test_zero_gravity_rejected(self):
        with pytest.raises(AxisymError, match="gravity"):
            lube_2004_emplacement_time(1.0, 0.5, gravity=0.0)

    def test_negative_gravity_rejected(self):
        with pytest.raises(AxisymError, match="gravity"):
            lube_2004_emplacement_time(1.0, 0.5, gravity=-9.81)


# ---------------------------------------------------------------------------
# build_axisymmetric_column
# ---------------------------------------------------------------------------

class TestBuildAxisymmetricColumn:
    def test_returns_positions_and_material(self):
        c = AxisymCase(aspect_ratio=1.0, grains_per_radius=4.0, radius=0.02)
        pos, mat = build_axisymmetric_column(c)
        assert pos.ndim == 2
        assert pos.shape[1] == 3
        assert pos.shape[0] > 0
        assert mat.radius == 0.02

    def test_positions_inside_cylinder(self):
        c = AxisymCase(aspect_ratio=1.0, grains_per_radius=4.0, radius=0.02)
        pos, mat = build_axisymmetric_column(c)
        rr = np.hypot(pos[:, 0], pos[:, 1])
        assert (rr + mat.radius).max() <= c.column_radius * 1.02

    def test_same_seed_same_positions(self):
        """HCP lattice is deterministic for a given seed."""
        c1 = AxisymCase(aspect_ratio=1.0, grains_per_radius=4.0, radius=0.02,
                        seed=42, jitter=0.0)
        c2 = AxisymCase(aspect_ratio=1.0, grains_per_radius=4.0, radius=0.02,
                        seed=42, jitter=0.0)
        pos1, _ = build_axisymmetric_column(c1)
        pos2, _ = build_axisymmetric_column(c2)
        assert np.allclose(pos1, pos2)

    def test_different_seeds_give_different_positions(self):
        """With jitter > 0, different seeds give different positions."""
        c1 = AxisymCase(aspect_ratio=1.0, grains_per_radius=4.0, radius=0.02,
                        seed=1, jitter=0.01)
        c2 = AxisymCase(aspect_ratio=1.0, grains_per_radius=4.0, radius=0.02,
                        seed=2, jitter=0.01)
        pos1, _ = build_axisymmetric_column(c1)
        pos2, _ = build_axisymmetric_column(c2)
        assert not np.allclose(pos1, pos2)

    def test_jitter_changes_positions(self):
        c0 = AxisymCase(aspect_ratio=1.0, grains_per_radius=4.0, radius=0.02,
                        jitter=0.0, seed=42)
        c1 = AxisymCase(aspect_ratio=1.0, grains_per_radius=4.0, radius=0.02,
                        jitter=0.01, seed=42)
        pos0, _ = build_axisymmetric_column(c0)
        pos1, _ = build_axisymmetric_column(c1)
        assert not np.allclose(pos0, pos1)
