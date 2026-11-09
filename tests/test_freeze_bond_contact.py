"""Acceptance tests for the frozen-contact-bond / failure-mode-transition node.

Campaign node IN13-W4-N03.  The six machine-checkable criteria are implemented as
one test class each:

(a) ``sigma_bond(T)`` vs Sastry & Metzger (1986): every point <= 20 % relative
    error (in-sample *and* leave-one-out);
(b) failure-mode switch: ``T_transition`` from the bisection root agrees with the
    closed-form analytic estimate within 2 C, and the mode flips across it;
(c) peak-load jump: ``F_cohesive/F_adhesive >= 2`` across the transition window;
(d) temperature monotonicity: Spearman rho <= -0.95;
(e) negative controls: ``T >= T_freeze`` -> ``sigma_bond = 0``, ``t_c -> 0`` ->
    ``sigma_bond -> 0``;
(f) energy ledger: complete thermal and mechanical budgets close to <= 5 %.

A regression test also pins the module-shadowing fix (the strength module used to
be unreachable behind the ``freeze_bond`` package).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.freeze_bond import FreezeBondConfig, sigma_bond_temperature
from tensordem.freeze_bond.contact import (
    ICE_HEAT_CAPACITY,
    ICE_CONDUCTIVITY,
    LATENT_HEAT_FUSION,
    REPETTO_ADHESION_WINDOW_MPA,
    WATER_DENSITY,
    ContactBondConfig,
    FrozenBondAssembly,
    FrozenContactBond,
    bond_strengths,
    break_event,
    energy_ledger,
    failure_mode_at,
    freezing_front_ledger,
    leave_one_out_validation,
    mechanical_energy_ledger,
    mode_transition_scan,
    multi_contact_slide,
    peak_contact_force,
    peak_load_jump,
    sastry_metzger_contact_table,
    single_contact_pulloff,
    spearman_rho,
    transition_temperature_analytic,
    transition_temperature_numeric,
)

CFG = ContactBondConfig()


# ---------------------------------------------------------------------------
# Regression: the strength module must be importable (shadowing fix)
# ---------------------------------------------------------------------------
class TestModuleWiring:
    def test_strength_api_reachable_from_package(self):
        """``FreezeBondConfig``/``sigma_bond_temperature`` live behind the package."""
        assert FreezeBondConfig is not None
        assert sigma_bond_temperature(-10.0) > 0.0

    def test_contact_module_is_part_of_package(self):
        from tensordem.freeze_bond import contact
        assert hasattr(contact, "FrozenContactBond")

    def test_config_validation(self):
        with pytest.raises(ValueError):
            ContactBondConfig(area_m2=0.0)
        with pytest.raises(ValueError):
            ContactBondConfig(film_thickness_m=-1.0)
        with pytest.raises(ValueError):
            ContactBondConfig(friction_coefficient=-0.1)


# ---------------------------------------------------------------------------
# (a) sigma_bond(T) vs Sastry & Metzger (1986)
# ---------------------------------------------------------------------------
class TestCriterionA_TemperatureCurve:
    def test_all_points_within_20pct(self):
        table = sastry_metzger_contact_table()
        assert table["n_points"] == 6
        for p in table["points"]:
            assert p["relative_error"] <= 0.20, p
        assert table["criterion_20pct_all_points"] is True
        assert table["max_relative_error"] <= 0.20

    def test_curve_monotone_and_zero_above_freezing(self):
        temps = np.linspace(-20.0, -1.0, 60)
        values = [single_contact_pulloff(float(t))["pulloff_stress_mpa"] for t in temps]
        assert np.all(np.diff(values) < 0.0)          # colder -> stronger
        assert single_contact_pulloff(0.5)["pulloff_stress_mpa"] == 0.0

    def test_leave_one_out_generalisation(self):
        loo = leave_one_out_validation()
        assert loo["max_relative_error"] <= 0.20, loo
        assert loo["mean_relative_error"] <= 0.15


# ---------------------------------------------------------------------------
# (b) failure-mode switch and transition temperature
# ---------------------------------------------------------------------------
class TestCriterionB_ModeSwitch:
    @pytest.mark.parametrize("pressure,time", [(2.0, 1.0), (5.0, 1.0), (1.0, 10.0)])
    def test_transition_matches_analytic(self, pressure, time):
        num = transition_temperature_numeric(pressure, time)
        ana = transition_temperature_analytic(pressure, time)
        assert num is not None and ana is not None
        assert abs(num - ana) <= 2.0, (num, ana)

    def test_residual_vanishes_at_root(self):
        t_tr = transition_temperature_numeric(2.0, 1.0)
        sb, sc = bond_strengths(t_tr, 2.0, 1.0)
        assert abs(sb - sc) <= 1.0e-9

    def test_mode_flips_across_transition(self):
        t_tr = transition_temperature_numeric(2.0, 1.0)
        assert failure_mode_at(t_tr - 0.05, 2.0, 1.0) == "cohesive"
        assert failure_mode_at(t_tr + 0.05, 2.0, 1.0) == "adhesive"

    def test_break_event_uses_first_threshold(self):
        ev = peak_contact_force(-5.0, 0.1, 1.0)
        below = break_event(0.5 * ev["f_peak_N"], -5.0, 0.1, 1.0)
        above = break_event(1.5 * ev["f_peak_N"], -5.0, 0.1, 1.0)
        assert below["broken"] is False
        assert above["broken"] is True
        assert above["failure_mode"] == ev["failure_mode"]
        assert above["threshold_N"] == pytest.approx(ev["f_peak_N"])


# ---------------------------------------------------------------------------
# (c) peak-load jump
# ---------------------------------------------------------------------------
class TestCriterionC_PeakLoadJump:
    @pytest.mark.parametrize("pressure,time", [(2.0, 1.0), (5.0, 1.0), (0.1, 1.0)])
    def test_window_ratio_at_least_two(self, pressure, time):
        jump = peak_load_jump(pressure, time)
        assert jump["defined"] is True
        assert jump["window_ratio"] >= 2.0, jump

    def test_peak_load_continuous_at_transition(self):
        """The mode switch is a kink: ``min(F_a, F_c)`` is continuous."""
        jump = peak_load_jump(2.0, 1.0)
        assert abs(jump["kink_ratio"] - 1.0) <= 1.0e-2

    def test_reserve_factor_above_one_in_adhesive_regime(self):
        jump = peak_load_jump(2.0, 1.0)
        assert jump["branch_strength_ratio"] > 1.0

    def test_peak_is_min_of_both_thresholds(self):
        ev = peak_contact_force(-3.0, 2.0, 1.0)
        assert ev["f_peak_N"] == min(ev["f_adhesive_N"], ev["f_cohesive_N"])


# ---------------------------------------------------------------------------
# (d) temperature monotonicity
# ---------------------------------------------------------------------------
class TestCriterionD_Monotonicity:
    def test_spearman_strongly_negative(self):
        temps = np.linspace(-1.0, -20.0, 25)
        sig = [bond_strengths(float(t), 0.1, 1.0)[0] for t in temps]
        rho = spearman_rho(temps, sig)
        assert rho <= -0.95, rho

    def test_spearman_cross_check_scipy(self):
        from scipy.stats import spearmanr
        x = np.linspace(-1.0, -30.0, 40)
        y = [bond_strengths(float(t), 5.0, 100.0)[0] for t in x]
        assert spearman_rho(x, y) == pytest.approx(float(spearmanr(x, y).statistic), abs=1e-9)

    def test_deep_subcooling_still_monotone(self):
        temps = np.linspace(-1.0, -30.0, 60)
        sig = np.array([bond_strengths(float(t), 0.1, 1.0)[0] for t in temps])
        assert np.all(np.diff(sig) > 0.0)


# ---------------------------------------------------------------------------
# (e) negative controls
# ---------------------------------------------------------------------------
class TestCriterionE_NegativeControls:
    def test_no_bond_above_freezing(self):
        for t in (0.0, 0.5, 5.0):
            assert bond_strengths(t, 0.1, 1.0)[0] == 0.0
            assert failure_mode_at(t, 0.1, 1.0) == "none"
            assert peak_contact_force(t, 0.1, 1.0)["f_peak_N"] == 0.0

    def test_bond_vanishes_with_contact_time(self):
        """``t_c -> 0`` drives ``sigma_bond -> 0`` (as ``t^alpha``, alpha ~ 0.2)."""
        full = bond_strengths(-5.0, 0.1, 1.0)[0]
        small = bond_strengths(-5.0, 0.1, 1.0e-9)[0]
        tiny = bond_strengths(-5.0, 0.1, 1.0e-30)[0]
        assert tiny < 1.0e-5 * full
        assert tiny < small < full
        assert small == pytest.approx(full * (1.0e-9) ** CFG.strength.alpha, rel=1e-9)

    def test_zero_pressure_gives_no_bond(self):
        assert bond_strengths(-5.0, 0.0, 1.0)[0] == 0.0

    def test_form_returns_false_above_freezing(self):
        bond = FrozenContactBond(CFG, temperature_c=2.0)
        assert bond.form() is False
        assert bond.state.bonded is False
        assert bond.load(1.0)["event"] == "no_bond"


# ---------------------------------------------------------------------------
# (f) energy ledger
# ---------------------------------------------------------------------------
class TestCriterionF_EnergyLedger:
    def test_thermal_budget_closes(self):
        therm = freezing_front_ledger(-10.0)
        assert therm["converged"] is True
        assert therm["residual"] <= 0.05, therm["residual"]
        assert therm["residual_balance"] <= 0.05, therm["residual_balance"]
        assert therm["freeze_time_error"] <= 0.05

    def test_mechanical_budget_closes(self):
        mech = mechanical_energy_ledger(-5.0)
        assert mech["residual"] <= 0.05
        assert mech["criterion_5pct"] is True

    def test_literal_two_term_budget_is_falsified(self):
        """Q_lat = E_fracture + E_friction does **not** hold for a 0.1 um film."""
        led = energy_ledger(-10.0)
        assert led["two_term_residual"] > 0.05
        assert led["latent_over_mechanical"] < 1.0
        # ... it would only close at a ~0.1 um film
        assert 5.0e-8 <= led["closure_thickness_m"] <= 2.0e-6

    def test_complete_ledger_passes(self):
        led = energy_ledger(-10.0)
        assert led["thermal_residual"] <= 0.05
        assert led["mechanical_residual"] <= 0.05
        assert led["criterion_5pct"] is True

    def test_energy_scales_with_area(self):
        small = energy_ledger(-10.0, config=ContactBondConfig(area_m2=1.0e-4))
        big = energy_ledger(-10.0, config=ContactBondConfig(area_m2=2.0e-4))
        assert big["q_latent_J"] == pytest.approx(2.0 * small["q_latent_J"], rel=1e-9)


# ---------------------------------------------------------------------------
# DEM contact life cycle and multi-contact sliding
# ---------------------------------------------------------------------------
class TestContactLifeCycle:
    def test_form_grow_break_slide(self):
        bond = FrozenContactBond(CFG, temperature_c=-5.0,
                                 contact_pressure_mpa=0.1, contact_time_s=1.0)
        assert bond.form() is True
        s1 = bond.strengths()[0]
        bond.advance(99.0)
        s100 = bond.strengths()[0]
        assert s100 > s1
        assert s100 / s1 == pytest.approx((100.0 / 1.0) ** CFG.strength.alpha, rel=1e-9)
        ev = bond.load(1.0e-3 * bond.peak()["f_peak_N"])
        assert ev["event"] == "elastic"
        ev = bond.load(1.0e6)
        assert ev["event"] == "break"
        assert bond.state.bonded is False
        work = bond.slide(1.0e-3, 10.0)
        assert work == pytest.approx(CFG.friction_coefficient * 10.0 * 1.0e-3)
        assert bond.state.e_friction_J == pytest.approx(work)

    def test_latent_heat_matches_film(self):
        bond = FrozenContactBond(CFG, temperature_c=-5.0)
        bond.form()
        expected = WATER_DENSITY * LATENT_HEAT_FUSION * CFG.film_thickness_m * CFG.area_m2
        assert bond.state.q_latent_J == pytest.approx(expected, rel=1e-12)


class TestMultiContactSlide:
    def test_breakaway_inside_declared_window(self):
        res = multi_contact_slide(-5.0)
        lo, hi = REPETTO_ADHESION_WINDOW_MPA
        assert lo <= res["breakaway_stress_mpa"] <= hi
        assert res["inside_repetto_window"] is True
        assert "WITHHELD" in res["evidence_grade"]

    def test_mode_counts_sum(self):
        res = multi_contact_slide(-15.0, n_contacts=32)
        assert sum(res["mode_counts"].values()) == 32

    def test_friction_stress_scales_with_mu(self):
        base = multi_contact_slide(-5.0)
        doubled = multi_contact_slide(-5.0, config=ContactBondConfig(friction_coefficient=0.4))
        assert doubled["friction_stress_mpa"] == pytest.approx(
            2.0 * base["friction_stress_mpa"], rel=1e-9)

    def test_assembly_requires_consistent_lengths(self):
        with pytest.raises(ValueError):
            FrozenBondAssembly([0.1, 0.2], [1.0e-4], -5.0)


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------
class TestSweep:
    def test_scan_shape_and_modes(self):
        rows = mode_transition_scan([-1.0, -5.0, -20.0], [0.1, 2.0, 5.0], [1.0, 100.0])
        assert len(rows) == 3 * 3 * 2
        modes = {r["failure_mode"] for r in rows}
        assert {"adhesive", "cohesive"} <= modes
        for r in rows:
            assert math.isfinite(r["sigma_bond_mpa"])
            assert r["f_peak_N"] >= 0.0

    def test_transition_inside_window_for_reference_condition(self):
        t_tr = transition_temperature_numeric(2.0, 1.0)
        assert -20.0 < t_tr < -1.0

    def test_transition_moves_colder_at_low_pressure(self):
        """At vanishing contact pressure the bond only wins at deep subcooling."""
        cold = transition_temperature_numeric(0.001, 1.0)
        warm = transition_temperature_numeric(2.0, 1.0)
        assert cold is None or cold < -30.0
        assert warm > cold if cold is not None else True