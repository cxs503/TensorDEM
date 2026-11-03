"""Regression guard for the restitution -> normal-damping mapping of DEM3D.

The solver integrates a compressive-only normal law, so the legacy free-oscillator
identity overshoots a requested coefficient of restitution (up to +63 % at e=0.2).
``DEM3DConfig.restitution_mapping="clamped"`` (the default) inverts the law the
solver really integrates. These tests measure the *shipped* solver, not the
formula against itself.
"""
from __future__ import annotations

import math

import pytest

from tensordem.contact_impact import clamped_restitution, restitution_to_damping_clamped
from tensordem.contact_restitution import measure_dem3d_impact
from tensordem.dem3d import DEM3DConfig, IceDEM3D

K_N = 5_000.0
RADIUS = 0.025
DENSITY = 917.0
CRITERION = 0.03


def _cfg(**kwargs) -> DEM3DConfig:
    base = dict(nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY,
                contact_stiffness=K_N, contact_damping=0.0, drag=0.0,
                fix_x_edges=False)
    base.update(kwargs)
    return DEM3DConfig(**base)


def _measure(e: float, mapping: str, **kwargs) -> dict:
    kwargs.setdefault("both_free", True)
    kwargs.setdefault("drag", 0.0)
    return measure_dem3d_impact(
        restitution_coefficient=e, restitution_mapping=mapping,
        k_n=K_N, radius=RADIUS, density=DENSITY, **kwargs)


class TestConfigContract:
    def test_default_mapping_is_clamped(self):
        assert _cfg().restitution_mapping == "clamped"

    def test_invalid_mapping_rejected(self):
        with pytest.raises(ValueError, match="restitution_mapping"):
            _cfg(restitution_mapping="hertzian")

    def test_unit_restitution_gives_zero_damping_in_both_mappings(self):
        for mapping in ("clamped", "classic"):
            assert _cfg(restitution_mapping=mapping).pair_normal_damping == 0.0

    def test_clamped_damping_is_larger_than_classic_for_lossy_contacts(self):
        # The clamped law releases the contact early, so it needs more damping to
        # remove the same amount of kinetic energy.
        for e in (0.2, 0.3, 0.5, 0.8):
            assert _cfg(restitution_coefficient=e).pair_normal_damping > \
                _cfg(restitution_coefficient=e, restitution_mapping="classic").pair_normal_damping

    def test_classic_mapping_reproduces_the_legacy_formula(self):
        e = 0.4
        cfg = _cfg(restitution_coefficient=e, restitution_mapping="classic")
        abs_ln_e = -math.log(e)
        zeta = abs_ln_e / math.sqrt(math.pi ** 2 + abs_ln_e ** 2)
        expected = 2.0 * zeta * math.sqrt(K_N * cfg.mass / 2.0)
        assert cfg.pair_normal_damping == pytest.approx(expected, rel=1e-12)

    def test_clamped_damping_matches_the_analytic_inverse(self):
        e = 0.35
        cfg = _cfg(restitution_coefficient=e)
        expected = restitution_to_damping_clamped(e, K_N, cfg.mass / 2.0)
        assert cfg.pair_normal_damping == pytest.approx(expected, rel=1e-12)

    def test_solver_uses_the_config_damping(self):
        for e in (0.45, 0.85):
            sim = IceDEM3D(_cfg(restitution_coefficient=e))
            assert sim._pair_normal_damping == pytest.approx(
                _cfg(restitution_coefficient=e).pair_normal_damping)

    def test_recommended_dt_shrinks_with_contact_damping(self):
        # A soft restitution is heavily damped; the stability bound must see it.
        dt_lossy = _cfg(restitution_coefficient=0.2).recommended_dt
        dt_elastic = _cfg(restitution_coefficient=1.0).recommended_dt
        assert dt_lossy < dt_elastic

    def test_recommended_dt_does_not_exceed_its_own_bound(self):
        cfg = _cfg(restitution_coefficient=0.2)
        assert IceDEM3D(cfg).dt <= cfg.recommended_dt


class TestSolverAccuracy:
    @pytest.mark.parametrize("e_target", [0.2, 0.3, 0.5, 0.7, 0.9])
    def test_clamped_mapping_meets_the_three_percent_band(self, e_target):
        result = _measure(e_target, "clamped")
        rel = result["e_measured"] / e_target - 1.0
        assert abs(rel) < CRITERION, (
            f"e_target={e_target}: realised {result['e_measured']:.5f} "
            f"({rel * 100:+.2f} %) exceeds the {CRITERION * 100:.0f} % band")

    @pytest.mark.parametrize("e_target", [0.2, 0.3, 0.5])
    def test_classic_mapping_overshoots_negative_control(self, e_target):
        """Anti-tautology: the legacy mapping must fail where it is known to bias."""
        result = _measure(e_target, "classic")
        rel = result["e_measured"] / e_target - 1.0
        assert rel > CRITERION, (
            f"e_target={e_target}: legacy mapping realised {rel * 100:+.2f} %, "
            "expected a positive overshoot above the 3 % band")

    def test_time_step_refinement_keeps_the_target(self):
        coarse = _measure(0.3, "clamped", dt_scale=1.0)
        fine = _measure(0.3, "clamped", dt_scale=0.25)
        for result in (coarse, fine):
            assert abs(result["e_measured"] / 0.3 - 1.0) < CRITERION

    def test_fixed_wall_case_is_documented_not_silently_wrong(self):
        """One frozen sphere is an infinite-mass wall: m_eff = mass, not mass/2.

        The configuration keeps the equal-mass convention, so the realised
        rebound is higher; the test pins that documented behaviour instead of
        letting it drift.
        """
        result = _measure(0.3, "clamped", both_free=False)
        assert result["e_measured"] / 0.3 - 1.0 > CRITERION

    def test_validity_band_lower_edge_is_documented(self):
        """At e=0.1 the contact is nearly critical and the band is NOT met.

        The mapping itself is exact; the discrete integrator is not: at zeta ~ 0.9
        the force truncation happens within a few steps per contact, so the realised
        rebound is ~+34 % high. The band stated in docs/dem3d_restitution_mapping.md
        is therefore e >= 0.2 (zeta <= 0.75); this test pins the edge so the claim
        cannot silently widen.
        """
        result = _measure(0.1, "clamped")
        assert result["e_measured"] / 0.1 - 1.0 > CRITERION

    def test_measured_matches_the_analytic_clamped_value(self):
        """The solver must reproduce the clamped closed form, not just the target."""
        e = 0.5
        cfg = _cfg(restitution_coefficient=e)
        measured = _measure(e, "clamped")
        zeta = measured["zeta"]
        assert measured["e_measured"] == pytest.approx(clamped_restitution(zeta), rel=CRITERION)