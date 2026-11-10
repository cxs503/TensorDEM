"""Tests for the freeze-bond mode-transition node (P46W3-N07).

Every acceptance criterion of the campaign node is exercised here:

(a) ``σ_bond = σ_0·exp(-β·ΔT)`` with ``β > 0`` (log-linear recovery of β),
(b) failure-mode transition temperature with ``|ΔT| ≤ 2 °C`` against the anchor,
(c) ``F-u`` curves against the anchor with ``R² ≥ 0.90``,
(d) negative control ``T → T_melt ⇒ σ_bond → 0``,

plus the REAL-tier checks that need no literature: the fracture-energy ledger
closes at machine precision, the discrete 1024-fibre pull-off reproduces the
closed-form curve, and the transition is a *kink* (not a jump) whose slope ratio
matches the analytic prediction ``β·ΔT_trans/n``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.freeze_bond.mode_transition import (
    ACCEPTANCE,
    MODE_ADHESIVE,
    MODE_COHESIVE,
    MODE_NONE,
    BondInterfaceParams,
    bond_area_fraction,
    bond_strength,
    bond_strength_temperature,
    failure_mode,
    failure_mode_map,
    failure_stress,
    fiber_bundle_pull_off,
    ice_tensile_strength,
    load_anchor_set,
    melt_ramp,
    peak_force,
    peak_jump_ratio,
    peak_slope_kink,
    pull_off_curve,
    r_squared,
    run_validation_report,
    traction_separation,
    transition_temperature,
    validate_against_anchor,
)

P = BondInterfaceParams()
T_C = 600.0  # saturated contact time used by the anchor set


# ---------------------------------------------------------------------------
# (a) exponential strength law
# ---------------------------------------------------------------------------
class TestStrengthLaw:
    def test_exponential_form_and_positive_beta(self):
        """Values follow σ_0·exp(-βΔT) and β > 0."""
        temps = np.linspace(-1.0, -25.0, 25)
        sigma = np.array([bond_strength_temperature(t, P) for t in temps])
        assert np.all(sigma > 0.0)
        # log-linear regression recovers beta and sigma_0
        slope, intercept = np.polyfit(temps, np.log(sigma), 1)
        assert -slope == pytest.approx(P.beta_per_C, rel=1e-10)
        assert math.exp(intercept) == pytest.approx(P.sigma_0_pa, rel=1e-9)
        assert P.beta_per_C > 0.0

    def test_strength_increases_with_subcooling(self):
        """Monotone in temperature: colder interface ⇒ stronger bond."""
        temps = np.linspace(-0.5, -25.0, 50)
        sigma = np.array([bond_strength_temperature(t, P) for t in temps])
        assert np.all(np.diff(sigma) > 0.0)

    def test_contact_time_growth_saturates(self):
        """Bonded area fraction: 0 at t=0, → 1 for t >> τ."""
        assert bond_area_fraction(0.0, P) == 0.0
        assert bond_area_fraction(P.tau_growth_s, P) == pytest.approx(1.0 - math.exp(-1.0))
        assert bond_area_fraction(50.0 * P.tau_growth_s, P) == pytest.approx(1.0, abs=1e-12)
        # a short contact gives a strictly weaker bond at the same temperature
        assert bond_strength(-10.0, 5.0, P) < bond_strength(-10.0, T_C, P)

    def test_ice_strength_weak_temperature_dependence(self):
        """Bulk ice strength follows ΔT^n with n≈0.25 and vanishes at the melt point."""
        assert ice_tensile_strength(-10.0, P) == pytest.approx(P.sigma_ice_ref_pa)
        assert ice_tensile_strength(-40.0, P) / ice_tensile_strength(-10.0, P) == pytest.approx(4.0 ** 0.25)
        assert ice_tensile_strength(0.0, P) == 0.0

    def test_invalid_parameters_rejected(self):
        """Negative β (bond weakening with sub-cooling) must not be constructible."""
        with pytest.raises(ValueError):
            BondInterfaceParams(beta_per_C=-0.1)
        with pytest.raises(ValueError):
            BondInterfaceParams(area_m2=0.0)
        with pytest.raises(ValueError):
            BondInterfaceParams(softening="cubic")


# ---------------------------------------------------------------------------
# (b) failure-mode transition
# ---------------------------------------------------------------------------
class TestModeTransition:
    def test_transition_bracketed_by_labels(self):
        """Adhesive above T_trans, cohesive below it."""
        t_trans = transition_temperature(P, contact_time_s=T_C)
        assert -30.0 < t_trans < -1.0
        assert peak_force(t_trans + 0.5, T_C, P)["mode"] == MODE_ADHESIVE
        assert peak_force(t_trans - 0.5, T_C, P)["mode"] == MODE_COHESIVE

    def test_peak_force_continuous_at_transition(self):
        """Pure stress criterion: the branches meet — a kink, not a jump."""
        t_trans = transition_temperature(P, contact_time_s=T_C)
        eps = 1e-4
        f_warm = peak_force(t_trans + eps, T_C, P)["peak_force_n"]
        f_cold = peak_force(t_trans - eps, T_C, P)["peak_force_n"]
        assert abs(f_cold - f_warm) / f_warm < 1e-4

    def test_transition_error_against_anchor(self):
        """Criterion (b): |T_trans,model - T_trans,anchor| ≤ 2 °C."""
        anchor = load_anchor_set()
        res = validate_against_anchor(anchor, P, T_C)["transition_validation"]
        assert res["abs_error_c"] <= ACCEPTANCE["max_transition_error_c"], res

    def test_transition_moves_colder_with_shorter_contact(self):
        """Less bonded area ⇒ the cohesive branch needs deeper sub-cooling."""
        t_long = transition_temperature(P, contact_time_s=600.0)
        t_short = transition_temperature(P, contact_time_s=5.0)
        assert t_short < t_long

    def test_no_transition_when_bond_never_reaches_ice_strength(self):
        """A 1e-3 s contact cannot produce a cohesive failure anywhere."""
        with pytest.raises(ValueError):
            transition_temperature(P, contact_time_s=1.0e-3)

    def test_failure_mode_map_consistency(self):
        """Map codes agree with the point-wise mode selection."""
        temps = np.linspace(-1.0, -25.0, 49)
        times = [5.0, 60.0, 600.0]
        mp = failure_mode_map(temps, times, P)
        assert mp["mode_codes"].shape == (len(times), temps.size)
        for i, t in enumerate(times):
            for j, temp in enumerate(temps):
                expect = failure_mode(
                    bond_strength(temp, t, P), ice_tensile_strength(temp, P), P
                )
                code = {MODE_NONE: 0, MODE_ADHESIVE: 1, MODE_COHESIVE: 2}[expect]
                assert mp["mode_codes"][i, j] == code

    def test_kink_slope_ratio_matches_analytic_prediction(self):
        """The adhesive branch is ≈ β·ΔT_trans/n times steeper than the cohesive one."""
        kink = peak_slope_kink(P, T_C)
        assert kink["slope_ratio_predicted"] == pytest.approx(
            P.beta_per_C * (P.t_melt_c - kink["transition_temperature_c"]) / P.ice_exponent
        )
        assert kink["slope_ratio_rel_error"] < 0.05
        assert kink["slope_ratio"] > 2.0

    def test_peak_jump_only_with_reinforced_cohesion(self):
        """k_coh=1 → continuous; k_coh=2 → peak step of k_coh²."""
        j1 = peak_jump_ratio(P, T_C)
        assert j1["jump_ratio"] == pytest.approx(1.0, abs=1e-3)
        p2 = BondInterfaceParams(k_cohesive=2.0)
        j2 = peak_jump_ratio(p2, T_C)
        assert j2["jump_ratio"] == pytest.approx(4.0, rel=5e-3)


# ---------------------------------------------------------------------------
# (c) force-displacement curves and the energy ledger
# ---------------------------------------------------------------------------
class TestForceDisplacement:
    @pytest.mark.parametrize("temperature", [-5.0, -15.0])
    def test_curve_shape_and_peak(self, temperature):
        """Linear rise with slope k, peak at F_p = σ_fail·A, zero after failure."""
        cd = pull_off_curve(temperature, T_C, P)
        info = peak_force(temperature, T_C, P)
        assert cd.peak_force_n == pytest.approx(info["peak_force_n"], rel=1e-9)
        assert cd.peak_displacement_m == pytest.approx(
            info["peak_force_n"] / P.stiffness_n_per_m, rel=1e-9
        )
        elastic = cd.u_m < 0.9 * cd.peak_displacement_m
        slope = np.polyfit(cd.u_m[elastic], cd.force_n[elastic], 1)[0]
        assert slope == pytest.approx(P.stiffness_n_per_m, rel=1e-6)
        assert cd.force_n[-1] == 0.0

    @pytest.mark.parametrize("temperature", [-5.0, -15.0])
    @pytest.mark.parametrize("softening", ["bilinear", "exponential"])
    def test_fracture_energy_ledger(self, temperature, softening):
        """Work minus the elastic triangle equals G_f·A exactly."""
        params = BondInterfaceParams(softening=softening)
        cd = pull_off_curve(temperature, T_C, params)
        assert cd.energy_error < ACCEPTANCE["max_energy_rel_error"]
        assert cd.dissipated_energy_J == pytest.approx(cd.fracture_energy_J, rel=1e-6)

    def test_cohesive_curve_is_stiffer_stronger_and_tougher(self):
        """Cohesive failure: higher peak, same stiffness, larger fracture energy."""
        ad = pull_off_curve(-5.0, T_C, P)
        co = pull_off_curve(-15.0, T_C, P)
        assert ad.mode == MODE_ADHESIVE and co.mode == MODE_COHESIVE
        assert co.peak_force_n > ad.peak_force_n
        assert co.G_f_J_m2 > ad.G_f_J_m2
        assert co.stiffness_n_per_m == pytest.approx(ad.stiffness_n_per_m)

    def test_unbonded_state_gives_zero_force(self):
        """Above the melt point there is no bond and no force."""
        cd = pull_off_curve(2.0, T_C, P)
        assert cd.mode == MODE_NONE
        assert cd.peak_force_n == 0.0 and np.all(cd.force_n == 0.0)
        f, mode = traction_separation(np.linspace(0.0, 1e-5, 11), 2.0, T_C, P)
        assert mode == MODE_NONE and np.all(f == 0.0)

    def test_curve_r2_against_anchor(self):
        """Criterion (c): R² ≥ 0.9 for both anchor curves."""
        res = validate_against_anchor(load_anchor_set(), P, T_C)["curve_validation"]
        assert res["min_r2"] >= ACCEPTANCE["min_curve_r2"], res
        assert res["max_peak_rel_error"] <= ACCEPTANCE["max_peak_rel_error"], res
        assert res["max_fracture_energy_rel_error"] <= ACCEPTANCE["max_fracture_energy_rel_error"], res

    def test_r_squared_definition(self):
        """R² = 1 for a perfect fit, 0 for the mean predictor."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        assert r_squared(obs, obs) == pytest.approx(1.0)
        assert r_squared(np.full(4, obs.mean()), obs) == pytest.approx(0.0)
        assert r_squared([0.0, 1.0], [0.0, 2.0]) == pytest.approx(1.0 - 1.0 / 2.0)


# ---------------------------------------------------------------------------
# Discrete (DEM-style) pull-off
# ---------------------------------------------------------------------------
class TestDiscreteBundle:
    @pytest.mark.parametrize("temperature", [-5.0, -15.0])
    def test_bundle_energy_closes_and_matches_continuum(self, temperature):
        """1024 independent bonds: exact G_f·A closure and the same F-u shape."""
        bd = fiber_bundle_pull_off(temperature, T_C, P)
        assert bd.energy_error <= ACCEPTANCE["max_energy_rel_error"]
        f_cont, _ = traction_separation(bd.u_m, temperature, T_C, P)
        r2 = r_squared(bd.force_n / bd.peak_force_n, f_cont / f_cont.max())
        assert r2 >= ACCEPTANCE["min_bundle_r2"], r2

    def test_bundle_mean_peak_below_single_bond_strength(self):
        """Strength scatter spreads the peak, so the ensemble peak is below σ_fail·A."""
        bd = fiber_bundle_pull_off(-15.0, T_C, P)
        info = peak_force(-15.0, T_C, P)
        assert 0.9 * info["peak_force_n"] < bd.peak_force_n < info["peak_force_n"]

    def test_bundle_rejects_single_fibre(self):
        with pytest.raises(ValueError):
            fiber_bundle_pull_off(-10.0, T_C, P, n_fibers=1)


# ---------------------------------------------------------------------------
# (d) negative control + anchor bookkeeping
# ---------------------------------------------------------------------------
class TestNegativeControlAndAnchor:
    def test_bond_strength_vanishes_at_melt(self):
        """Criterion (d): T → T_melt ⇒ σ_bond → 0 (continuously, monotonically)."""
        series = [bond_strength(P.t_melt_c - dt, T_C, P) for dt in (1.0, 0.1, 0.01, 0.001)]
        assert all(b < a for a, b in zip(series, series[1:]))
        assert series[-1] / P.sigma_0_pa < 2e-3
        assert bond_strength(P.t_melt_c, T_C, P) == 0.0
        assert bond_strength(P.t_melt_c + 5.0, T_C, P) == 0.0
        assert melt_ramp(P.t_melt_c - 0.5, P) == pytest.approx(0.5)

    def test_asymptotic_law_recovered_deep_in_subcooling(self):
        """For ΔT ≥ ΔT_ramp the physical strength equals the exponential law."""
        for t in (-2.0, -10.0, -25.0):
            assert bond_strength(t, T_C, P) == pytest.approx(bond_strength_temperature(t, P), rel=1e-6)

    def test_anchor_set_schema_and_provenance(self):
        """The anchor set is loadable and declares its evidence grade."""
        anchor = load_anchor_set()
        assert anchor["provenance"]["evidence_grade"] in ("REAL", "WITHHELD")
        assert anchor["provenance"]["digitization"] in ("reconstructed", "point_digitized")
        assert len(anchor["peak_tests"]) >= 5
        assert len(anchor["curves"]) >= 2
        for curve in anchor["curves"]:
            assert len(curve["u_um"]) == len(curve["F_N"])
            assert curve["mode"] in (MODE_ADHESIVE, MODE_COHESIVE)

    def test_peak_force_error_against_anchor(self):
        """Peak force vs temperature stays inside the 15 % accuracy standard."""
        res = validate_against_anchor(load_anchor_set(), P, T_C)["peak_validation"]
        assert res["max_rel_error"] <= ACCEPTANCE["max_peak_rel_error"], res
        assert res["mode_agreement"] >= 0.99, res

    def test_full_report_passes_all_checks(self):
        """The machine-checkable acceptance report is green end to end."""
        report = run_validation_report(P, T_C)
        failed = {
            k: v
            for k, v in report["checks"].items()
            if isinstance(v, dict) and v.get("passed") is False
        }
        assert not failed, failed
        assert report["all_checks_passed"]