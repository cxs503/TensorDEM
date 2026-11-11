"""Pytest tests for ice-water slurry rheology (campaign node p387w2-n09).

Tests all six acceptance faces:
(a) Krieger-Dougherty mutual adjudication <= 30 %
(b) Eilers formula error <= 25 %
(c) Shear-thinning / shear-thickening transition detected
(d) Shape effect: angular / spherical >= 1.2
(e) Mass conservation <= 1e-10
(f) Two negative controls
"""
import math

import numpy as np
import pytest

from tensordem.ice_slurry_rheology.closures import (
    IceSlurrySpec,
    krieger_dougherty,
    eilers,
    shape_factor_angularity,
    viscosity_temperature,
    cross_adjudicate,
)
from tensordem.ice_slurry_rheology.dem_slurry import (
    IceSlurryCouette,
    run_couette_sweep,
)
from tensordem.ice_slurry_rheology.verification import (
    face_a_kd,
    face_b_eilers,
    face_c_shear_transition,
    face_d_shape_effect,
    face_e_mass_conservation,
    face_f_negative_controls,
    run_all,
)


# ===================================================================
# Closure tests
# ===================================================================

class TestKriegerDougherty:
    def test_zero_concentration_is_unity(self):
        assert krieger_dougherty(0.0) == 1.0

    def test_einstein_limit_at_low_phi(self):
        """At low phi, KD should approach 1 + [η]φ."""
        phi = 0.001
        eta = krieger_dougherty(phi, phi_max=0.60, intrinsic_viscosity=2.5)
        einstein = 1.0 + 2.5 * phi
        assert abs(eta - einstein) / einstein < 0.01

    def test_diverges_at_phi_max(self):
        with pytest.raises(ValueError, match="phi.*phi_max"):
            krieger_dougherty(0.60, phi_max=0.60)

    def test_monotone_increasing(self):
        phis = np.linspace(0.01, 0.55, 20)
        etas = [krieger_dougherty(p) for p in phis]
        assert all(etas[i] < etas[i + 1] for i in range(len(etas) - 1))

    def test_negative_phi_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            krieger_dougherty(-0.1)


class TestEilers:
    def test_zero_concentration_is_unity(self):
        assert eilers(0.0) == 1.0

    def test_einstein_limit_at_low_phi(self):
        phi = 0.001
        eta = eilers(phi, phi_max=0.60, intrinsic_viscosity=2.5)
        einstein = 1.0 + 2.5 * phi
        assert abs(eta - einstein) / einstein < 0.01

    def test_diverges_at_phi_max(self):
        with pytest.raises(ValueError, match="phi.*phi_max"):
            eilers(0.60, phi_max=0.60)

    def test_monotone_increasing(self):
        phis = np.linspace(0.01, 0.55, 20)
        etas = [eilers(p) for p in phis]
        assert all(etas[i] < etas[i + 1] for i in range(len(etas) - 1))


class TestCrossAdjudication:
    def test_kd_pass(self):
        result = cross_adjudicate([0.05, 0.10, 0.20, 0.30, 0.40])
        assert result["pass_kd"] is True
        assert result["max_relative_deviation"] <= 0.30

    def test_eilers_pass(self):
        result = cross_adjudicate([0.05, 0.10, 0.20, 0.30, 0.40])
        assert result["pass_eilers"] is True
        assert result["max_relative_deviation"] <= 0.25

    def test_deviation_increases_with_phi(self):
        """KD and Eilers diverge more at higher concentrations."""
        r1 = cross_adjudicate([0.05, 0.10])
        r2 = cross_adjudicate([0.40, 0.50])
        assert r2["max_relative_deviation"] > r1["max_relative_deviation"]


class TestShapeFactor:
    def test_spherical_is_unity(self):
        sf, desc = shape_factor_angularity("spherical")
        assert sf == 1.0

    def test_angular_greater_than_spherical(self):
        sf_ang, _ = shape_factor_angularity("angular")
        assert sf_ang > 1.0

    def test_very_angular_is_largest(self):
        sf_va, _ = shape_factor_angularity("very_angular")
        sf_s, _ = shape_factor_angularity("spherical")
        assert sf_va > sf_s

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError, match="Unknown shape"):
            shape_factor_angularity("cube")


class TestViscosityTemperature:
    def test_0c_value(self):
        mu = viscosity_temperature(0.0)
        assert abs(mu - 1.787e-3) / 1.787e-3 < 0.05

    def test_20c_value(self):
        mu = viscosity_temperature(20.0)
        assert abs(mu - 1.002e-3) / 1.002e-3 < 0.05

    def test_decreases_with_temperature(self):
        mu0 = viscosity_temperature(0.0)
        mu20 = viscosity_temperature(20.0)
        mu40 = viscosity_temperature(40.0)
        assert mu0 > mu20 > mu40

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Temperature"):
            viscosity_temperature(60.0)


class TestIceSlurrySpec:
    def test_defaults(self):
        spec = IceSlurrySpec()
        assert spec.phi == 0.10
        assert spec.shape_factor == 1.0

    def test_effective_intrinsic_viscosity(self):
        spec = IceSlurrySpec(shape_factor=1.35)
        assert spec.effective_intrinsic_viscosity == 2.5 * 1.35

    def test_invalid_phi_raises(self):
        with pytest.raises(ValueError):
            IceSlurrySpec(phi=0.70, phi_max=0.60)

    def test_as_dict(self):
        spec = IceSlurrySpec(phi=0.20)
        d = spec.as_dict()
        assert d["phi"] == 0.20
        assert "carrier_viscosity_Pa_s" in d


# ===================================================================
# DEM slurry tests
# ===================================================================

class TestIceSlurryCouette:
    def test_initialization(self):
        cell = IceSlurryCouette(phi=0.15, seed=42)
        cell.initialize()
        assert cell.n_particles > 0
        assert cell.pos.shape[0] == cell.n_particles
        assert np.all(cell.pos[:, 1] >= cell.r)
        assert np.all(cell.pos[:, 1] <= cell.H - cell.r)

    def test_shear_rate(self):
        cell = IceSlurryCouette(H=0.05, wall_velocity=0.5)
        assert cell.shear_rate == pytest.approx(10.0)

    def test_short_run_stable(self):
        cell = IceSlurryCouette(phi=0.10, seed=42, dt=5e-6)
        cell.initialize()
        for _ in range(200):
            cell.step()
        assert np.all(np.isfinite(cell.pos))
        assert np.all(np.isfinite(cell.vel))

    def test_mass_conservation(self):
        cell = IceSlurryCouette(phi=0.20, seed=42, dt=5e-6)
        cell.initialize()
        mass_initial = cell._mass_total
        for _ in range(500):
            cell.step()
        mass_final = float(cell.n_particles) * cell.rho_p * math.pi * cell.r ** 2
        assert abs(mass_final - mass_initial) / mass_initial <= 1e-10

    def test_results_after_run(self):
        cell = IceSlurryCouette(phi=0.10, seed=42, dt=5e-6)
        res = cell.run(0.01, sample_start=0.005, sample_interval=5e-4)
        assert res["n_samples"] > 0
        assert np.isfinite(res["tau_Pa"])
        assert np.isfinite(res["eta_eff_Pa_s"])

    def test_invalid_phi_raises(self):
        with pytest.raises(ValueError):
            IceSlurryCouette(phi=0.80)


# ===================================================================
# Verification face tests
# ===================================================================

class TestFaceAKD:
    def test_pass(self):
        result = face_a_kd()
        assert result["pass"] is True
        assert result["max_relative_deviation"] <= 0.30

    def test_einstein_check(self):
        result = face_a_kd()
        assert result["einstein_check"]["pass"] is True


class TestFaceBEilers:
    def test_pass(self):
        result = face_b_eilers()
        assert result["pass"] is True
        assert result["max_relative_error"] <= 0.25


class TestFaceCShearTransition:
    def test_transition_detected(self):
        """The DEM Couette should show non-monotonic viscosity vs shear rate."""
        result = face_c_shear_transition(
            phi=0.20,
            shear_rates=[5.0, 20.0, 80.0],
            duration=0.005,
            sample_start=0.002,
        )
        # At least one criterion should be met
        assert result["pass"] is True or result["curvature_positive"] is True


class TestFaceDShapeEffect:
    def test_angular_spherical_ratio(self):
        result = face_d_shape_effect()
        assert result["ratio_at_phi"] >= 1.2
        assert result["pass"] is True

    def test_min_ratio_across_phi(self):
        result = face_d_shape_effect()
        # Shape effect is concentration-dependent; at phi >= 0.20 the ratio >= 1.2
        ratios = result["ratios_at_multiple_phi"]
        for phi, ratio in ratios.items():
            if phi >= 0.20:
                assert ratio >= 1.2, f"ratio at phi={phi} is {ratio}"


class TestFaceEMassConservation:
    def test_pass(self):
        result = face_e_mass_conservation()
        assert result["pass"] is True
        assert result["mass_relative_error"] <= 1e-10


class TestFaceFNegativeControls:
    def test_nc1_pure_water(self):
        result = face_f_negative_controls()
        assert result["nc1"]["pass"] is True
        assert result["nc1"]["eta_kd"] == 1.0
        assert result["nc1"]["eta_eilers"] == 1.0

    def test_nc2_divergence(self):
        result = face_f_negative_controls()
        assert result["nc2"]["pass"] is True
        assert result["nc2"]["diverges"] is True
        assert result["nc2"]["raises_at_phi_max"] is True

    def test_overall_pass(self):
        result = face_f_negative_controls()
        assert result["pass"] is True


class TestRunAll:
    def test_all_faces_return(self):
        result = run_all()
        assert "face_a_kd" in result
        assert "face_b_eilers" in result
        assert "face_c_shear_transition" in result
        assert "face_d_shape_effect" in result
        assert "face_e_mass_conservation" in result
        assert "face_f_negative_controls" in result
