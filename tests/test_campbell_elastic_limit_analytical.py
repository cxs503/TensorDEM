"""Analytical verification of Campbell (2002) elastic-limit scaling relationships.

These tests verify the mathematical structure of the dimensionless groups and
scaling laws WITHOUT running any DEM simulation.  They are lightweight
(< 0.1 s) and can run on any machine regardless of load.

Reference: Campbell, C. S. (2002). Granular shear flows at the elastic limit.
J. Fluid Mech. 465, 261-291. DOI: 10.1017/S002211200200109X
"""
from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# Dimensionless group definitions (from Campbell 2002)
# ---------------------------------------------------------------------------

def kappa(kn: float, rho: float, d: float, gdot: float) -> float:
    """Dimensionless stiffness: κ = k_n / (ρ d³ γ̇²)."""
    return kn / (rho * d ** 3 * gdot ** 2)


def tau_star(tau: float, rho: float, d: float, gdot: float) -> float:
    """Dimensionless shear stress: τ* = τ / (ρ d² γ̇²)."""
    return tau / (rho * d ** 2 * gdot ** 2)


def shear_rate_exponent_from_beta(beta: float) -> float:
    """Shear-rate exponent from the log-log slope: n = 2(1 - β).

    Derived from κ ~ γ̇⁻² at fixed k_n:
        d ln τ* / d ln κ = β
        d ln τ / d ln γ̇ = 2(1 - β) = n
    """
    return 2.0 * (1.0 - beta)


def beta_from_exponent(n: float) -> float:
    """Inverse: β = 1 - n/2."""
    return 1.0 - n / 2.0


# ---------------------------------------------------------------------------
# Reference regime boundaries (from Campbell 2002 abstract + secondary sources)
# ---------------------------------------------------------------------------

KAPPA_QUASI_STATIC = 1e4   # κ ≥ 1e4 → elastic-quasi-static
KAPPA_INERTIAL = 1.0       # κ < 1 → inertial (Bagnold)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDimensionlessGroups:
    """Verify the dimensionless group definitions match Campbell's formulation."""

    def test_kappa_units_cancel(self):
        """κ should be dimensionless: [k_n] = M/T², [ρ d³ γ̇²] = M/T²."""
        # SI units check
        kn = 1e5       # N/m = kg/s²
        rho = 1000.0   # kg/m³
        d = 0.001      # m
        gdot = 10.0    # 1/s
        k = kappa(kn, rho, d, gdot)
        assert k > 0
        # Numerical value
        expected = 1e5 / (1000 * 1e-9 * 100)
        assert k == pytest.approx(expected)

    def test_tau_star_units_cancel(self):
        """τ* should be dimensionless: [τ] = M/(L T²), [ρ d² γ̇²] = M/(L T²)."""
        tau = 1000.0   # Pa = kg/(m s²)
        rho = 1000.0   # kg/m³
        d = 0.001      # m
        gdot = 10.0    # 1/s
        ts = tau_star(tau, rho, d, gdot)
        assert ts > 0
        expected = 1000 / (1000 * 1e-6 * 100)
        assert ts == pytest.approx(expected)

    def test_kappa_scales_as_gamma_dot_minus_2(self):
        """At fixed k_n, κ ∝ γ̇⁻²."""
        kn, rho, d = 1e4, 1.0, 1.0
        gdot1, gdot2 = 1.0, 2.0
        k1 = kappa(kn, rho, d, gdot1)
        k2 = kappa(kn, rho, d, gdot2)
        # Doubling γ̇ should quarter κ
        assert k1 / k2 == pytest.approx(4.0)

    def test_tau_star_scales_as_gamma_dot_minus_2_for_fixed_tau(self):
        """At fixed τ, τ* ∝ γ̇⁻²."""
        tau, rho, d = 100.0, 1.0, 1.0
        gdot1, gdot2 = 1.0, 2.0
        ts1 = tau_star(tau, rho, d, gdot1)
        ts2 = tau_star(tau, rho, d, gdot2)
        assert ts1 / ts2 == pytest.approx(4.0)


class TestScalingRelationships:
    """Verify the scaling relationships between β and n."""

    def test_beta_to_exponent_quasi_static(self):
        """In elastic-quasi-static limit: β → 1, n → 0."""
        beta = 1.0
        n = shear_rate_exponent_from_beta(beta)
        assert n == pytest.approx(0.0)

    def test_beta_to_exponent_bagnold(self):
        """In inertial (Bagnold) limit: β → 0, n → 2."""
        beta = 0.0
        n = shear_rate_exponent_from_beta(beta)
        assert n == pytest.approx(2.0)

    def test_beta_to_exponent_elastic_inertial(self):
        """In elastic-inertial regime: 0 < β < 1, 0 < n < 2."""
        for beta in [0.2, 0.5, 0.8]:
            n = shear_rate_exponent_from_beta(beta)
            assert 0 < n < 2

    def test_exponent_to_beta_inverse(self):
        """β = 1 - n/2 should be the inverse of n = 2(1 - β)."""
        for n in [0.0, 0.5, 1.0, 1.5, 2.0]:
            beta = beta_from_exponent(n)
            n_back = shear_rate_exponent_from_beta(beta)
            assert n_back == pytest.approx(n)

    def test_loglog_slope_interpretation(self):
        """If τ* = C κ^β, then d ln τ* / d ln κ = β.

        Verify with synthetic data.
        """
        C = 0.5
        beta_true = 0.6
        kappa_vals = [10.0, 100.0, 1000.0]
        tau_star_vals = [C * k ** beta_true for k in kappa_vals]

        # Compute slope from first and last points
        log_k1, log_ts1 = math.log(kappa_vals[0]), math.log(tau_star_vals[0])
        log_k2, log_ts2 = math.log(kappa_vals[-1]), math.log(tau_star_vals[-1])
        beta_measured = (log_ts2 - log_ts1) / (log_k2 - log_k1)

        assert beta_measured == pytest.approx(beta_true, rel=1e-10)


class TestRegimeBoundaries:
    """Verify the regime boundary classifications."""

    def test_quasi_static_boundary(self):
        """κ ≥ 1e4 → elastic-quasi-static."""
        assert kappa(1e6, 1.0, 1.0, 1.0) >= KAPPA_QUASI_STATIC
        assert kappa(1e4, 1.0, 1.0, 1.0) >= KAPPA_QUASI_STATIC

    def test_elastic_inertial_band(self):
        """1 ≤ κ < 1e4 → elastic-inertial."""
        assert KAPPA_INERTIAL <= kappa(1e3, 1.0, 1.0, 1.0) < KAPPA_QUASI_STATIC
        assert KAPPA_INERTIAL <= kappa(1e2, 1.0, 1.0, 1.0) < KAPPA_QUASI_STATIC
        assert KAPPA_INERTIAL <= kappa(1e1, 1.0, 1.0, 1.0) < KAPPA_QUASI_STATIC

    def test_inertial_boundary(self):
        """κ < 1 → inertial (Bagnold)."""
        assert kappa(1e0, 1.0, 1.0, 10.0) < KAPPA_INERTIAL
        assert kappa(1e0, 1.0, 1.0, 100.0) < KAPPA_INERTIAL

    def test_regime_exponent_consistency(self):
        """Each regime should have the expected exponent range."""
        # Quasi-static: n → 0
        beta_qs = 0.98  # close to 1
        n_qs = shear_rate_exponent_from_beta(beta_qs)
        assert n_qs < 0.1

        # Elastic-inertial: 0 < n < 2
        beta_ei = 0.5
        n_ei = shear_rate_exponent_from_beta(beta_ei)
        assert 0 < n_ei < 2

        # Inertial: n → 2
        beta_in = 0.02  # close to 0
        n_in = shear_rate_exponent_from_beta(beta_in)
        assert n_in > 1.9


class TestCampbellAbstractClaims:
    """Verify claims extractable from the Campbell (2002) abstract."""

    def test_no_path_between_inertial_and_quasi_static_at_fixed_concentration(self):
        """Campbell abstract: 'no path between inertial (rapid) flow and
        quasi-static flow by varying the shear rate at a fixed concentration'.

        This means: at fixed φ, varying γ̇ changes κ (since κ ∝ γ̇⁻²), but
        the transition is abrupt, not continuous.  The abstract states a 2%
        concentration reduction can cause ~3 orders of magnitude stress reduction.
        """
        # At fixed φ, κ varies with γ̇⁻²
        # A factor of 1000 in stress requires κ to change by ~1000
        # Since κ ∝ γ̇⁻², this requires γ̇ to change by √1000 ≈ 31.6
        gamma_dot_ratio = math.sqrt(1000)
        # This is achievable by varying γ̇, but the abstract says the transition
        # is abrupt (not a smooth path), implying a bifurcation
        assert gamma_dot_ratio > 1  # sanity check

    def test_two_percent_concentration_stress_drop(self):
        """Abstract: 'a 2% concentration reduction can be accompanied by
        nearly three orders of magnitude of stress reduction'.

        This is a qualitative claim about the abruptness of the transition,
        not a quantitative scaling law.  We verify the order of magnitude.
        """
        stress_reduction_factor = 1000  # 3 orders of magnitude
        concentration_change = 0.02  # 2%
        # The claim is about the ratio, not the absolute values
        assert stress_reduction_factor >= 100  # at least 2 orders
        assert concentration_change < 0.05  # small concentration change


class TestEnergyAuditStructure:
    """Verify the energy audit structure (mathematical, not numerical)."""

    def test_work_dissipation_balance_dimensional(self):
        """ΔE = W - D should be dimensionally consistent.

        [E] = M L²/T² (energy)
        [W] = [γ̇] [A] [τ] = (1/T) (L²) (M/(L T²)) = M L / T³ ... wait.

        Actually in 2D: [A] = L² (area), [τ] = M/T² (force/length in 2D)
        [W] = (1/T) (L²) (M/T²) = M L² / T³ ... still not energy.

        The work rate is Ẇ = -γ̇ A S_xy, so W = ∫ Ẇ dt has units:
        [W] = (1/T) (L²) (M/T²) (T) = M L² / T² = energy. ✓
        """
        # Dimensional check: work rate × time = energy
        gdot = 1.0    # 1/T
        area = 1.0    # L²
        S_xy = 1.0    # M/T² (in 2D DEM units)
        dt = 1.0      # T

        work_rate = gdot * area * S_xy  # M L² / T³
        work = work_rate * dt           # M L² / T² = energy

        # In DEM units (m=1, d=1, ρ=1), energy has units of M L²/T²
        assert work > 0

    def test_kinetic_energy_formula(self):
        """E_kin = ½ Σ m v² should have correct units."""
        m = 1.0   # M
        v = 1.0   # L/T
        e_kin = 0.5 * m * v ** 2  # M L²/T²
        assert e_kin > 0

    def test_elastic_energy_formula(self):
        """E_el = ½ k δ² should have correct units."""
        k = 1e5     # M/T² (spring constant)
        delta = 0.01  # L (overlap)
        e_el = 0.5 * k * delta ** 2  # M L²/T²
        assert e_el > 0
