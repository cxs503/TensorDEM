"""Tests for the engine-coupled BPM identifiability harness (P143W2C-N02).

Coverage map:

* geometry / Latin design / quasi-static rate estimate (pure functions),
* the size-effect estimator and its pre-registered identifiability rule
  (positive control with known m = 6, negative control with m = infinity),
* the UCS literature cross-check rule,
* an end-to-end engine smoke run of a bonded specimen (small lattice, few
  increments) so the driver is exercised against the real ``IceDEM3D``,
* the static/dynamic dual friction law: legacy bit-for-bit equivalence with
  ``friction_regularization="none"``, the analytic cone limits of all three laws,
  and the measured residual friction of a slip cell against the analytic
  Coulomb caps.
"""
import math

import numpy as np
import pytest
import torch

from tensordem import bpm_identifiability as bi
from tensordem.dem3d import DEM3DConfig, IceDEM3D


# ---------------------------------------------------------------------------
# Geometry and design helpers
# ---------------------------------------------------------------------------
class TestGeometry:
    def test_gauge_length_and_area(self):
        g = bi.SpecimenGeometry(n_layers=5, cross_n=4, radius=0.0125)
        assert g.thickness == pytest.approx(0.1)
        assert g.area == pytest.approx(0.01)
        assert g.n_particles == 80
        assert g.volume == pytest.approx(0.001)

    def test_from_thickness_hits_the_requested_families(self):
        for thickness in (0.1, 0.2, 0.4):
            g = bi.SpecimenGeometry.from_thickness(thickness)
            assert g.thickness == pytest.approx(thickness)

    def test_rejects_degenerate_geometry(self):
        with pytest.raises(ValueError):
            bi.SpecimenGeometry(n_layers=1)
        with pytest.raises(ValueError):
            bi.SpecimenGeometry(radius=0.0)


class TestLatinDesign:
    def test_strata_are_covered_once_per_beta(self):
        betas = (100.0, 400.0, 1600.0, 3600.0)
        design = bi.latin_rate_design(betas, n_points=3, rate_ref=1e-2, seed=1)
        assert set(design) == set(betas)
        # The rate factors of each beta fall in distinct strata of [0.5, 2.0].
        edges = np.linspace(0.5, 2.0, 4)
        for beta, rates in design.items():
            factors = sorted(r / (1e-2 * math.sqrt(3600.0 / beta)) for r in rates)
            strata = [int(np.searchsorted(edges, f, side="right") - 1) for f in factors]
            assert sorted(strata) == [0, 1, 2], (beta, factors)

    def test_rates_scale_with_beta(self):
        design = bi.latin_rate_design((100.0, 3600.0), n_points=1, rate_ref=1.0,
                                      seed=0)
        assert design[3600.0][0] == pytest.approx(design[100.0][0] / 6.0)


class TestQuasistaticRate:
    def test_rate_falls_with_mass_scaling(self):
        g = bi.SpecimenGeometry()
        r100 = bi.quasistatic_strain_rate(1e7, g, mass_scaling=100.0)
        r3600 = bi.quasistatic_strain_rate(1e7, g, mass_scaling=3600.0)
        assert r100 == pytest.approx(6.0 * r3600, rel=1e-9)

    def test_rate_grows_with_stiffness(self):
        g = bi.SpecimenGeometry()
        soft = bi.quasistatic_strain_rate(1e7, g, mass_scaling=400.0)
        stiff = bi.quasistatic_strain_rate(4e7, g, mass_scaling=400.0)
        assert stiff > soft


# ---------------------------------------------------------------------------
# Size effect, identifiability and the controls
# ---------------------------------------------------------------------------
class TestSizeEffect:
    def test_positive_control_recovers_m_within_10_percent(self):
        # Four thickness families and n = 96 specimens per family: the design at
        # which the bootstrap CI of m is both centred and narrow (see the design
        # curve in docs/bpm_identifiability_P143W2C_N02.md).
        ctrl = bi.positive_control(m_true=6.0, n_specimens=96,
                                   thicknesses=(0.05, 0.1, 0.2, 0.4),
                                   n_bootstrap=300)
        assert ctrl["m_relative_error"] <= 0.10, ctrl
        assert ctrl["identifiable"], ctrl
        assert ctrl["m_ci95"][0] < 6.0 < ctrl["m_ci95"][1]

    def test_positive_control_minimum_design_is_wide_but_centred(self):
        """The pre-registered minimum design (n = 8) is a CI-width statement."""
        ctrl = bi.positive_control(m_true=6.0, n_specimens=8, n_bootstrap=300)
        assert ctrl["m_ci95"][0] < 6.0 < ctrl["m_ci95"][1]
        assert ctrl["m_ci95"][1] - ctrl["m_ci95"][0] > bi.M_CI_WIDTH_MAX

    def test_negative_control_constant_strength_is_unidentifiable(self):
        neg = bi.negative_control_strengths(jitter_rel=0.0, n_bootstrap=200)
        assert neg["degenerate"]
        assert not neg["identifiable"]
        assert neg["m_ci95"] == list(bi.DEGENERATE_M_BAND)
        assert neg["pass"]

    def test_verdict_rule_rejects_the_sentinel_band(self):
        sentinel = {"m_ci95": list(bi.DEGENERATE_M_BAND), "m_ci_width": 100.0,
                    "degenerate": True}
        assert not bi.identifiability_verdict(sentinel)["identifiable"]
        narrow = {"m_ci95": [5.2, 5.9], "m_ci_width": 0.7, "degenerate": False}
        assert bi.identifiability_verdict(narrow)["identifiable"]
        wide = {"m_ci95": [3.0, 8.5], "m_ci_width": 5.5, "degenerate": False}
        assert not bi.identifiability_verdict(wide)["identifiable"]

    def test_bootstrap_rejects_a_single_family(self):
        with pytest.raises(ValueError):
            bi.bootstrap_size_effect_m({0.1: [1e6, 2e6]}, n_bootstrap=10)


class TestUcsCrossCheck:
    def test_central_value_hits_all_three_intervals(self):
        out = bi.ucs_literature_cross_check(5.5e6)
        assert out["in_window"]
        assert out["n_hits"] == 3
        assert out["pass"]

    def test_out_of_window_value_fails(self):
        low = bi.ucs_literature_cross_check(1.0e6)
        assert not low["in_window"]
        assert not low["pass"]
        high = bi.ucs_literature_cross_check(30e6)
        assert not high["pass"]

    def test_two_hits_are_required(self):
        strict = {"a": {"interval_MPa": (0.0, 10.0), "provenance": "synthetic",
                        "digitized": False},
                  "b": {"interval_MPa": (100.0, 200.0), "provenance": "synthetic",
                        "digitized": False}}
        out = bi.ucs_literature_cross_check(5.0e6, intervals=strict, min_hits=2)
        assert out["n_hits"] == 1
        assert not out["pass"]


# ---------------------------------------------------------------------------
# Engine-coupled specimen driver (smoke, small lattice)
# ---------------------------------------------------------------------------
class TestSpecimenDriver:
    def test_small_specimen_produces_finite_strength_and_modulus(self):
        geometry = bi.SpecimenGeometry(n_layers=3, cross_n=2)
        micro = bi.MicroParameters(bond_stiffness=2.56e5)
        protocol = bi.LoadProtocol(mode="tension", protocol="staged",
                                   mass_scaling=100.0, target_strain=0.02,
                                   increments=6, ramp_steps=8, hold_steps=12,
                                   decay_steps=10, settle_steps=10)
        result = bi.run_specimen(micro, geometry, protocol, bond_seed=0)
        assert result.n_particles == 12
        assert result.n_bonds > 0
        assert math.isfinite(result.sigma_peak) and result.sigma_peak > 0.0
        assert math.isfinite(result.youngs_modulus) and result.youngs_modulus > 0.0
        assert result.strain.size == result.stress.size
        assert result.ke_ratio >= 0.0
        assert result.dt > 0.0

    def test_compression_mode_stresses_the_specimen_inward(self):
        geometry = bi.SpecimenGeometry(n_layers=3, cross_n=2)
        micro = bi.MicroParameters()
        protocol = bi.LoadProtocol(mode="compression", protocol="staged",
                                   mass_scaling=100.0, target_strain=0.02,
                                   increments=6, ramp_steps=8, hold_steps=12,
                                   decay_steps=10, settle_steps=10)
        result = bi.run_specimen(micro, geometry, protocol, bond_seed=0)
        # Compression drives the top layer down: the gauge shortens, the nominal
        # stress magnitude is positive and the trace is recorded under the same
        # convention as tension.
        assert result.mode == "compression"
        assert result.sigma_peak > 0.0
        assert np.all(np.diff(result.strain) >= -1e-12)

    def test_protocol_validation(self):
        with pytest.raises(ValueError):
            bi.LoadProtocol(mode="shear")
        with pytest.raises(ValueError):
            bi.LoadProtocol(protocol="creep")
        with pytest.raises(ValueError):
            bi.LoadProtocol(strain_rate=0.0)


# ---------------------------------------------------------------------------
# Static/dynamic dual friction law
# ---------------------------------------------------------------------------
def _engine(regularization="none", mu_s=None, mu_d=None, mu=0.0, v_c=1e-3):
    return IceDEM3D(DEM3DConfig(
        nx=2, ny=2, nz=2, radius=0.0125, friction_coefficient=mu,
        friction_regularization=regularization,
        static_friction_coefficient=mu_s, dynamic_friction_coefficient=mu_d,
        friction_transition_speed=v_c))


class TestDualFriction:
    def test_default_pair_reproduces_the_single_coefficient(self):
        cfg = DEM3DConfig(friction_coefficient=0.37)
        assert cfg.friction_coefficient_pair == (0.37, 0.37)
        cfg = DEM3DConfig(friction_coefficient=0.1, static_friction_coefficient=0.6,
                          dynamic_friction_coefficient=0.4)
        assert cfg.friction_coefficient_pair == (0.6, 0.4)

    def test_dynamic_may_not_exceed_static(self):
        with pytest.raises(ValueError):
            DEM3DConfig(static_friction_coefficient=0.3,
                        dynamic_friction_coefficient=0.5)

    def test_unknown_regularization_rejected(self):
        with pytest.raises(ValueError):
            DEM3DConfig(friction_regularization="smooth")

    def test_cone_limits_match_the_three_laws(self):
        normal = torch.full((4,), 2.0, dtype=torch.float64)
        tan_force = torch.tensor([0.1, 0.5, 1.0, 1.5], dtype=torch.float64)
        speed = torch.tensor([0.0, 0.5e-3, 1.0e-3, 1.0e-1], dtype=torch.float64)
        legacy = _engine(mu=0.5).config
        legacy_limit = legacy.friction_cone_limit(tan_force, normal, speed)
        assert np.allclose(legacy_limit.numpy(), 0.5 * normal.numpy(), rtol=1e-12)

        tanh_cfg = _engine("tanh", mu_s=0.6, mu_d=0.4).config
        limits = tanh_cfg.friction_cone_limit(tan_force, normal, speed)
        # mu(v) = 0.4 + 0.2 * (1 - tanh(v/vc)) / 2, times N = 2
        expected = np.array([0.4 + 0.2 * 0.5 * (1 - math.tanh(v / 1e-3))
                             for v in speed.tolist()]) * 2.0
        assert np.allclose(limits.numpy(), expected, rtol=1e-12)

        dual_cfg = _engine("dual", mu_s=0.6, mu_d=0.4).config
        limits = dual_cfg.friction_cone_limit(tan_force, normal, speed)
        # Stick while |F_t| <= mu_s N = 1.2, then the cone drops to mu_d N = 0.8.
        assert np.allclose(limits.numpy(), [1.2, 1.2, 1.2, 0.8], rtol=1e-12)

    def test_legacy_trajectory_is_unchanged_by_the_new_switch(self):
        """``regularization="none"`` must reproduce the pre-existing dynamics."""
        def run(**kwargs):
            sim = IceDEM3D(DEM3DConfig(nx=3, ny=3, nz=3, radius=0.0125,
                                       friction_coefficient=0.5,
                                       gravity=9.81, **kwargs))
            for _ in range(30):
                sim.step()
            return sim

        default = run()
        explicit = run(friction_regularization="none",
                       static_friction_coefficient=0.5,
                       dynamic_friction_coefficient=0.5)
        assert torch.equal(default.positions, explicit.positions)
        assert torch.equal(default.velocities, explicit.velocities)


class TestSlipCell:
    @pytest.mark.parametrize("slip_speed", [0.5e-3, 5.0e-3])
    def test_dual_law_residual_equals_the_dynamic_cap(self, slip_speed):
        out = bi.shear_cell_residual(regularization="dual", mu_s=0.6, mu_d=0.4,
                                     transition_speed=1.0e-3,
                                     slip_speed=slip_speed, n_steps=500)
        assert out["mu_measured"] == pytest.approx(out["mu_reference"], rel=0.02)
        assert out["residual_over_prediction_vs_dynamic_pct"] < 2.0

    def test_tanh_law_over_predicts_the_residual_in_the_transition_band(self):
        dual = bi.shear_cell_residual(regularization="dual", slip_speed=0.5e-3,
                                      n_steps=500)
        tanh = bi.shear_cell_residual(regularization="tanh", slip_speed=0.5e-3,
                                      n_steps=500)
        assert tanh["mu_measured"] > 1.05 * dual["mu_measured"]
        assert tanh["bias_pct"] == pytest.approx(0.0, abs=2.0)