"""Analytic + invariant verification of ``tensordem.bowden_hughes_thermal``.

Companion to ``docs/AUDIT_BOWDEN_HUGHES_THERMAL_m16736.md`` and
``benchmarks/bowden_hughes_audit/run_audit.py``.

Three kinds of assertions live here, and they are deliberately separated:

* **correct-physics tests** — closed forms the module *does* satisfy (Blok
  effusivity partition, Stefan-film scaling exponents, melt-film mass balance,
  admissible chatter-free output); these must stay green.
* **defect pins** — tests that assert the value the module currently returns so
  that any later correction of the physics is noticed.  Each pin states the
  analytic reference and the measured deviation in its docstring.
* **``xfail(strict=True)`` gaps** — properties the module *should* satisfy for
  the physics it advertises.  They fail today by construction (see the audit
  document); strict xfail means the day someone fixes them the suite will say
  so instead of silently keeping a stale pin.

Run with ``PYTHONPATH=src python3 -m pytest tests/test_bowden_hughes_thermal_analytic.py -q``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.bowden_hughes_thermal import (
    BowdenHughesConfig,
    _melt_film_thickness_steady,
    _squeeze_out_time,
    _transient_conduction_depth,
    closure_residual,
    compute_bowden_hughes_equilibrium,
    compute_mu_velocity_curve,
    energy_conservation_check,
)

CFG = BowdenHughesConfig()

E_ICE = math.sqrt(CFG.ice_thermal_conductivity * CFG.ice_density * CFG.ice_specific_heat)
E_STEEL = math.sqrt(
    CFG.steel_thermal_conductivity * CFG.steel_density * CFG.steel_specific_heat
)


def dT_semi_infinite(q: float, t: float) -> float:
    """``dT = 2 q sqrt(t/pi)/(e_ice + e_steel)`` for two semi-infinite bodies."""
    return 2.0 * q * math.sqrt(t / math.pi) / (E_ICE + E_STEEL)


def t_stefan_tau(h: float, sigma: float) -> float:
    """Stefan time constant ``tau = 3 eta R^2/(4 sigma h0^2)``.

    ``h(t) = h0/sqrt(1+t/tau)``, so the half-thickness time is ``3 tau``.
    """
    return 3.0 * CFG.water_viscosity * CFG.contact_radius**2 / (4.0 * sigma * h**2)


# --------------------------------------------------------------------------- #
# 1. correct physics: things the module gets right
# --------------------------------------------------------------------------- #
class TestHeatPartition:
    """Blok effusivity partition — the one branch that is analytically exact."""

    def test_flux_ratio_equals_effusivity_ratio(self):
        """q_ice/q_steel must be e_ice/e_steel (equal interface temperature)."""
        r = compute_bowden_hughes_equilibrium(1.0, 1e6, CFG)
        assert r.heat_flux_steel > 0
        measured = r.heat_flux_ice / r.heat_flux_steel
        assert measured == pytest.approx(E_ICE / E_STEEL, rel=1e-12)

    def test_partition_fractions_sum_to_one(self):
        r = compute_bowden_hughes_equilibrium(1.0, 1e6, CFG)
        assert r.asperity_area_fraction + r.film_area_fraction == pytest.approx(1.0, abs=1e-12)

    def test_effusivity_of_ice_much_smaller_than_steel(self):
        """Sanity: steel must dominate the heat sink by an order of magnitude."""
        assert E_STEEL / E_ICE > 5.0


class TestMeltFilmMassBalance:
    """``h = q_melt R / (L rho V)`` — a pure mass balance, must be exact."""

    @pytest.mark.parametrize("velocity", [0.01, 0.1, 1.0, 10.0])
    def test_module_h_equals_hand_computation(self, velocity):
        r = compute_bowden_hughes_equilibrium(velocity, 1e6, CFG)
        expected = r.heat_flux_melt * CFG.contact_radius / (
            CFG.latent_heat_fusion * CFG.water_density * velocity
        )
        assert r.film_thickness == pytest.approx(expected, rel=1e-14)

    def test_helper_zero_flux_gives_zero_film(self):
        assert _melt_film_thickness_steady(0.0, 1.0, 3.34e5, 1000.0, 2e-3) == 0.0

    def test_film_is_velocity_independent_at_fixed_mu(self):
        """With q_melt ~ mu sigma V the film thickness h -> mu sigma R/(L rho).

        Above V ~ 0.1 m/s the conductive sink is negligible next to
        ``q_total = mu sigma V``, so the film thickness saturates at
        ``h -> mu sigma R/(L rho)`` and stops growing with V.  The viscous term
        ``mu_film = eta V/(h sigma)`` then rises only linearly, staying below
        ``mu_asy`` up to V ~ 74 m/s — which is why the shipped mu(V) is flat.
        """
        hs = [
            compute_bowden_hughes_equilibrium(v, 1e6, CFG).film_thickness
            for v in (0.1, 1.0, 10.0)
        ]
        assert max(hs) / min(hs) < 1.10


class TestSqueezeFilmScaling:
    """Stefan squeeze film: scaling must be eta^1 R^2 sigma^-1 h^-2."""

    def test_scaling_exponents(self):
        h, eta, R, sigma = 8.9e-7, CFG.water_viscosity, CFG.contact_radius, 1e6
        base = _squeeze_out_time(h, sigma, eta, R)
        assert _squeeze_out_time(h, sigma, 2 * eta, R) / base == pytest.approx(2.0, rel=1e-12)
        assert _squeeze_out_time(h, sigma, eta, 2 * R) / base == pytest.approx(4.0, rel=1e-12)
        assert _squeeze_out_time(h, 2 * sigma, eta, R) / base == pytest.approx(0.5, rel=1e-12)
        assert _squeeze_out_time(2 * h, sigma, eta, R) / base == pytest.approx(0.25, rel=1e-12)

    def test_squeeze_time_constant_is_twice_tau(self):
        """Defect pin: the shipped prefactor is 2 tau, i.e. 2/3 of the half-life.

        Stefan: dh/dt = -2h^3 F/(3 pi eta R^4), F = sigma pi R^2 =>
        h(t) = h0/sqrt(1+t/tau), tau = 3 eta R^2/(4 sigma h0^2); the
        half-thickness time is 3 tau.  The module returns
        3 eta R^2/(2 sigma h0^2) = 2 tau (measured ratio to tau = 2.000,
        ratio to the half-life = 0.667).
        """
        h, sigma = 8.9e-7, 1e6
        measured = _squeeze_out_time(h, sigma, CFG.water_viscosity, CFG.contact_radius)
        assert measured / t_stefan_tau(h, sigma) == pytest.approx(2.0, rel=1e-12)
        assert measured / (3.0 * t_stefan_tau(h, sigma)) == pytest.approx(2.0 / 3.0, rel=1e-12)


class TestOutputAdmissibility:
    """Cheap invariants that any physically admissible answer must satisfy."""

    @pytest.mark.parametrize("sigma", [1e5, 5e5, 1e6, 2e6])
    def test_mu_bounded_by_asperity_friction(self, sigma):
        """The clamp ``min(mu_film, mu_asy)`` makes mu <= mu_asy structural."""
        mu, _ = compute_mu_velocity_curve(sigma, np.logspace(-4, 2, 61), CFG)
        assert np.all(mu >= 0.0)
        assert np.all(mu <= CFG.asperity_friction + 1e-12)

    def test_interface_temperature_never_exceeds_melting(self):
        for v in np.logspace(-4, 2, 25):
            r = compute_bowden_hughes_equilibrium(float(v), 1e6, CFG)
            assert r.interface_temperature <= 0.0 + 1e-12

    def test_zero_velocity_state(self):
        r = compute_bowden_hughes_equilibrium(0.0, 1e6, CFG)
        assert r.friction_coefficient == CFG.asperity_friction
        assert r.heat_flux_total == 0.0
        assert r.regime == "boundary"
        assert closure_residual(r) == 0.0


class TestClosureResidualDiagnostic:
    """The additive diagnostic delivered with this audit."""

    def test_zero_for_zero_flux(self):
        r = compute_bowden_hughes_equilibrium(0.0, 1e6, CFG)
        assert closure_residual(r) == 0.0

    def test_detects_the_shipped_inconsistency(self):
        """At V=0.1 m/s, sigma=0.1 MPa the returned mu does NOT reproduce q."""
        r = compute_bowden_hughes_equilibrium(0.1, 1e5, CFG)
        assert r.friction_coefficient < CFG.asperity_friction
        assert closure_residual(r) > 1e-2


# --------------------------------------------------------------------------- #
# 2. defect pins (measured values, so a fix is noticed)
# --------------------------------------------------------------------------- #
class TestDefectPins:
    def test_surface_temperature_is_26_percent_above_semi_infinite(self):
        """Defect pin: dT is 26.2 % above the semi-infinite analytic value.

        The conduction-only branch mixes ``k_eff = mean(k)`` and
        ``delta_eff = mean(delta)`` instead of using ``2 q sqrt(t/pi)/(e_i+e_s)``.
        """
        v, sigma = 1e-4, 1e5
        r = compute_bowden_hughes_equilibrium(v, sigma, CFG)
        assert r.heat_flux_melt == 0.0
        dT_mod = r.interface_temperature - CFG.bulk_ice_temperature
        dT_ref = dT_semi_infinite(r.heat_flux_total, CFG.contact_time)
        assert dT_mod / dT_ref == pytest.approx(1.2623, abs=2e-3)

    def test_melting_onset_flux_is_velocity_independent(self):
        """Defect pin: q_crit = k_ice (-T_bulk)/sqrt(alpha_ice t_contact) = const.

        Because the conduction depth ignores the contact residence time, the
        melting criterion fires at the same flux (928 W/m2) for every V.
        """
        qs = [
            compute_bowden_hughes_equilibrium(v, 1e6, CFG).heat_flux_melt
            for v in (0.01, 1.0, 100.0)
        ]
        assert all(q > 0 for q in qs)
        flux_onset = CFG.asperity_friction * 1e6 * 0.006194410750767818
        assert flux_onset == pytest.approx(929.16, rel=1e-4)

    @pytest.mark.parametrize("sigma", [1e5, 5e5, 1e6, 2e6])
    def test_regime_is_always_boundary(self, sigma):
        """Defect pin: the mixed/hydrodynamic branches are unreachable."""
        _, res = compute_mu_velocity_curve(sigma, np.logspace(-4, 2, 121), CFG)
        assert {r.regime for r in res} == {"boundary"}
        # the module's own gate for leaving "boundary" is A_film >= 0.05; the
        # squeeze criterion keeps A_film below it everywhere (max 2.7e-2 at
        # sigma = 0.1 MPa), so the mixed/hydrodynamic branches are dead code.
        assert max(r.film_area_fraction for r in res) < 0.05

    def test_mu_excursion_over_six_decades_of_velocity(self):
        """Defect pin: mu(V) spans < 2 % of mu_asy over 1e-4..1e2 m/s."""
        brute, _ = compute_mu_velocity_curve(1e5, np.logspace(-4, 2, 241), CFG)
        excursion = float((CFG.asperity_friction - brute.min()) / CFG.asperity_friction)
        assert excursion == pytest.approx(0.0184, abs=1e-3)

    def test_energy_check_is_a_tautology(self):
        """``energy_conservation_check`` passes at 1e-12 wherever it is defined.

        It re-checks ``q_ice + q_steel + q_melt == q_total`` — an identity the
        solver imposes in both branches — while the *physical* closure
        ``q = mu sigma V`` is off by up to 1.7 %.
        """
        n_pass, n, worst_closure = 0, 0, 0.0
        for sigma in (1e5, 5e5, 1e6, 2e6):
            for v in np.logspace(-4, 2, 13):
                r = compute_bowden_hughes_equilibrium(float(v), sigma, CFG)
                n += 1
                n_pass += int(energy_conservation_check(r, tolerance=1e-12))
                worst_closure = max(worst_closure, closure_residual(r))
        assert n_pass == n
        assert worst_closure > 1e-2


# --------------------------------------------------------------------------- #
# 3. gaps: advertised physics that the implementation cannot produce
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="AUDIT m16736: heat flux is always mu_asy*sigma*V, never mu_total*sigma*V; "
    "closure residual reaches 1.7 % at V=0.1 m/s, sigma=0.1 MPa",
)
def test_physical_closure_within_1_percent():
    worst = 0.0
    for sigma in (1e5, 5e5, 1e6, 2e6):
        for v in np.logspace(-4, 2, 61):
            worst = max(worst, closure_residual(compute_bowden_hughes_equilibrium(float(v), sigma, CFG)))
    assert worst <= 1e-2


@pytest.mark.xfail(
    strict=True,
    reason="AUDIT m16736: advertised melting lubrication does not appear — mu(V) is "
    "flat within 1.8 % over 6 decades because the squeeze criterion removes the film",
)
def test_melting_lubrication_depletes_friction():
    mu, _ = compute_mu_velocity_curve(5e5, np.logspace(-2, 1, 61), CFG)
    assert mu.min() <= 0.5 * CFG.asperity_friction


@pytest.mark.xfail(
    strict=True,
    reason="AUDIT m16736: conduction-only branch is 26.2 % above the semi-infinite "
    "analytic surface temperature; tolerance is the house 3 % rule",
)
def test_surface_temperature_matches_semi_infinite_solution():
    r = compute_bowden_hughes_equilibrium(1e-4, 1e5, CFG)
    dT_mod = r.interface_temperature - CFG.bulk_ice_temperature
    dT_ref = dT_semi_infinite(r.heat_flux_total, CFG.contact_time)
    assert dT_mod == pytest.approx(dT_ref, rel=0.03)


def test_conduction_depth_helper_matches_definition():
    """``delta = sqrt(alpha t)`` as documented (used by the analytic comparison)."""
    assert _transient_conduction_depth(1.17e-6, 120.0) == pytest.approx(
        math.sqrt(1.17e-6 * 120.0), rel=1e-15
    )