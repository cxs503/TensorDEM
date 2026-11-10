"""Tests for moist-particle drying in the rotary dryer (mission p380w2-n05).

Coverage maps 1:1 onto the machine-checkable acceptance criteria:

(a) wet-bulb pinning in the constant-rate period
(b) Ranz-Marshall mass transfer: diffusive limit, Re=100 anchor, analogy
(c) water ledger closure
(d) energy ledger closure + latent-sink negative control (the check has teeth)
(e) falling-rate kinetics against the Page model
(f) degenerate limit w -> 0, Y_g -> 0 reproducible against the dry-particle
    baseline (plan 380 n02 thermo-mechanical limit)
(g) parametric monotonicity of the outlet moisture

plus property-table validation of the psychrometric closures.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from tensordem.rotary_dryer import psychrometrics as psy
from tensordem.rotary_dryer.moist_bed import (
    DryerBedConfig,
    bed_ledger,
    simulate_dry_heating,
    simulate_moist_drying,
    solve_steady_profile,
)
from tensordem.rotary_dryer.moist_transfer import (
    driving_force,
    film_properties,
    fit_thin_layer,
    sherwood_ranz_marshall,
    sherwood_whitaker,
    simulate_single_particle,
    sorption_activity,
    transfer_coefficients,
    wet_bulb_from_closures,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
BASELINE = REPO / "artifacts" / "moist_drying" / "dry_baseline.json"

MAIN = dict(
    gas_inlet_temperature=400.0,
    gas_inlet_humidity=0.010,
    gas_inlet_velocity=3.0,
    solid_feed_dry=0.25,
    residence_time=600.0,
    particle_diameter=0.004,
    particle_density=1250.0,
    particle_cp=1500.0,
    initial_moisture=0.35,
    critical_moisture=0.15,
    equilibrium_moisture=0.02,
    sorption_kappa=6.0,
    particle_initial_temperature=293.15,
)


# ---------------------------------------------------------------------------
# psychrometric property closures
# ---------------------------------------------------------------------------
class TestPsychrometrics:
    @pytest.mark.parametrize("t_celsius,ref_pa", [(0.0, 611.2), (20.0, 2339.0),
                                                  (50.0, 12352.0), (100.0, 101325.0)])
    def test_saturation_pressure_table(self, t_celsius, ref_pa):
        """Hyland-Wexler against the IAPWS table (tolerance 0.2%)."""
        p = psy.saturation_pressure(t_celsius + 273.15)
        assert abs(p - ref_pa) / ref_pa < 2e-3

    def test_latent_heat_ashrae(self):
        assert abs(psy.latent_heat(373.15) - 2257.0e3) / 2257.0e3 < 1e-2
        assert abs(psy.latent_heat(293.15) - 2454.0e3) / 2454.0e3 < 1e-2

    def test_latent_heat_is_enthalpy_difference(self):
        for T in (280.0, 320.0, 360.0):
            assert psy.vapour_enthalpy(T) - psy.liquid_enthalpy(T) == pytest.approx(
                psy.latent_heat(T), rel=1e-14
            )

    def test_enthalpy_inversion_roundtrip(self):
        for T, Y in ((300.0, 0.0), (400.0, 0.01), (350.0, 0.05)):
            assert psy.temperature_from_enthalpy(psy.moist_enthalpy(T, Y), Y) == pytest.approx(T, abs=1e-12)

    def test_vapour_density_humidity_ratio_identity(self):
        """Differential identity: rho_da dY = (1 - p_v/p)^-1 d(rho_v), isothermal."""
        T, p, Y1 = 350.0, psy.P_STD, 0.01
        Y2 = Y1 * (1.0 + 1e-6)
        p_v = psy.vapour_pressure_from_humidity_ratio(Y1, p)
        rho_da = psy.dry_air_density(p_v, T, p)
        d_rho = psy.vapour_density(psy.vapour_pressure_from_humidity_ratio(Y2, p), T) - \
            psy.vapour_density(p_v, T)
        factor = p / (p - p_v)
        assert rho_da * (Y2 - Y1) == pytest.approx(d_rho * factor, rel=1e-6)

    def test_wetbulb_room_conditions(self):
        """Classical psychrometric wet bulb at 20 degC, Y = 0.01 (chart ~15.8 degC)."""
        T_wb = psy.wet_bulb_temperature(293.15, 0.01)
        assert T_wb == pytest.approx(289.0, abs=1.0)

    def test_wetbulb_saturated_air_returns_drybulb(self):
        Y_sat = psy.saturation_humidity_ratio(300.0, psy.P_STD)
        assert psy.wet_bulb_temperature(300.0, Y_sat) == pytest.approx(300.0, abs=1e-9)

    def test_wetbulb_rejects_negative_humidity(self):
        with pytest.raises(ValueError):
            psy.wet_bulb_temperature(300.0, -0.001)


# ---------------------------------------------------------------------------
# (b) mass-transfer closure
# ---------------------------------------------------------------------------
class TestMassTransferClosure:
    def test_diffusive_limit_sh_equals_two(self):
        """Sh -> 2 as Re -> 0 (steady diffusion from a sphere)."""
        for Re in (1e-4, 1e-3):
            Sh = sherwood_ranz_marshall(Re, 0.6)
            assert abs(Sh - 2.0) / 2.0 < 0.02

    def test_diffusive_limit_by_extrapolation(self):
        Re = np.array([1e-6, 4e-6, 1e-5, 4e-5, 1e-4])
        Sh = np.array([sherwood_ranz_marshall(r, 0.6) for r in Re])
        coef = np.polyfit(np.sqrt(Re), Sh, 1)
        assert coef[1] == pytest.approx(2.0, abs=1e-6)

    def test_re100_matches_independent_reference(self):
        """Re = 100 anchor evaluated through an independent arithmetic path."""
        from decimal import Decimal, getcontext

        getcontext().prec = 40
        ref = Decimal(2) + Decimal("0.6") * Decimal(100).sqrt() * (
            Decimal("0.6") ** (Decimal(1) / Decimal(3))
        )
        model = sherwood_ranz_marshall(100.0, 0.6)
        assert abs(float(ref) - model) / float(ref) < 0.05

    def test_re100_literature_spread(self):
        """Whitaker (1972) is within 15% of Ranz-Marshall at Re=100 (documented spread)."""
        assert abs(sherwood_whitaker(100.0, 0.6) - sherwood_ranz_marshall(100.0, 0.6)) / \
            sherwood_ranz_marshall(100.0, 0.6) < 0.15

    def test_heat_mass_analogy_consistency(self):
        tr = transfer_coefficients(400.0, 320.0, 0.01, 3.0, 0.004)
        ratio = (tr["Sh"] / tr["Nu"]) / (tr["Sc"] / tr["Pr"]) ** (1.0 / 3.0)
        assert abs(ratio - 1.0) < 0.05

    def test_coefficients_positive_and_scaling(self):
        tr = transfer_coefficients(400.0, 310.0, 0.01, 2.5, 0.004)
        assert tr["h"] > 0 and tr["h_m"] > 0
        assert tr["h_m"] == pytest.approx(tr["Sh"] * tr["D_ab"] / 0.004)
        assert tr["h"] == pytest.approx(tr["Nu"] * tr["k"] / 0.004)

    def test_negative_re_speed_uses_magnitude(self):
        a = transfer_coefficients(400.0, 320.0, 0.01, 3.0, 0.004)
        b = transfer_coefficients(400.0, 320.0, 0.01, -3.0, 0.004)
        assert a["Re"] == pytest.approx(b["Re"])


# ---------------------------------------------------------------------------
# (a) wet-bulb pinning
# ---------------------------------------------------------------------------
class TestWetBulbPinning:
    def test_lewis_one_limit_reproduces_chart_wetbulb(self):
        """With D_AB = alpha the dual closure collapses onto the psychrometric value."""
        T_g, Y_g = 400.0, 0.01
        T_wb_le1 = wet_bulb_from_closures(
            T_g, Y_g, 2.5, 0.004, d_ab_override="alpha",
            driving_force_form="humidity_ratio",
        )
        T_wb_chart = psy.wet_bulb_temperature(T_g, Y_g)
        assert abs(T_wb_le1 - T_wb_chart) < 0.01

    def test_constant_rate_particle_temperature_pinned(self):
        """Particle temperature plateau = closed-form wet bulb, slope ~ 0."""
        T_g, Y_g, u, d = 305.0, 0.005, 1.0, 0.004
        T_wb = wet_bulb_from_closures(T_g, Y_g, u, d)
        sim = simulate_single_particle(T_g, Y_g, u, d, 1250.0, 1500.0, 293.15,
                                       0.35, 0.02, 0.15, 6.0, t_end=250.0, dt=0.01)
        t, T, X = sim["t"], sim["T_p"], sim["X"]
        const = (X > 0.15) & (t > 120.0)
        assert const.sum() > 100
        assert np.abs(T[const] - T_wb).max() < 1.0
        # slope over any 30-step window inside the plateau
        slope = np.abs(np.diff(T, 30)) / 0.30
        assert slope[const[30:]].max() < 1e-2

    def test_wetbulb_is_stable_fixed_point_from_below_and_above(self):
        T_g, Y_g, u, d = 340.0, 0.02, 1.0, 0.004
        T_wb = wet_bulb_from_closures(T_g, Y_g, u, d)
        for T0 in (T_wb - 15.0, T_wb + 15.0):
            sim = simulate_single_particle(T_g, Y_g, u, d, 1250.0, 1500.0, T0,
                                           1.0, 0.02, 0.40, 6.0, t_end=120.0, dt=0.01)
            assert abs(sim["T_p"][-1] - T_wb) < 1.0

    def test_bed_constant_rate_pinning(self):
        """Bed-level: constant-rate cells sit on the local wet bulb."""
        cfg = DryerBedConfig(**{**MAIN, "gas_inlet_velocity": 2.0})
        prof = solve_steady_profile(cfg, nz=200)
        mask = (prof["a_w"] > 0.999) & (prof["z"] > 1.0)
        assert mask.sum() > 20
        assert np.abs(prof["T_p"] - prof["T_wb"])[mask].max() < 1.0

    def test_latent_cooling_absent_without_sink(self):
        """Sanity: switching the latent sink off removes the wet-bulb plateau."""
        cfg = dict(gas_inlet_temperature=340.0, gas_inlet_humidity=0.02,
                   gas_inlet_velocity=1.0, solid_feed_dry=0.02, n_cells=6,
                   dt=5.0, t_end=300.0, include_latent_sink=False)
        res = simulate_moist_drying(DryerBedConfig(**cfg))
        assert res["final"]["T_p_out"] > 330.0


# ---------------------------------------------------------------------------
# sorption isotherm
# ---------------------------------------------------------------------------
class TestSorptionIsotherm:
    def test_bounds_and_continuity(self):
        X_e, X_c, kappa = 0.02, 0.15, 6.0
        assert sorption_activity(X_e, X_e, X_c, kappa) == 0.0
        assert sorption_activity(0.0, X_e, X_c, kappa) == 0.0
        assert sorption_activity(X_c, X_e, X_c, kappa) == 1.0
        assert sorption_activity(0.5, X_e, X_c, kappa) == 1.0
        assert sorption_activity(X_c - 1e-9, X_e, X_c, kappa) == pytest.approx(1.0, abs=1e-6)
        assert sorption_activity(X_e + 1e-9, X_e, X_c, kappa) < 1e-6

    def test_monotone_increasing_in_X(self):
        vals = [sorption_activity(x, 0.02, 0.15, 6.0) for x in np.linspace(0.02, 0.15, 50)]
        assert np.all(np.diff(vals) > 0)


# ---------------------------------------------------------------------------
# (c) + (d) ledgers
# ---------------------------------------------------------------------------
class TestLedgers:
    @staticmethod
    @pytest.fixture(scope="class")
    def res():
        cfg = DryerBedConfig(**MAIN, n_cells=8, dt=5.0, t_end=300.0)
        return simulate_moist_drying(cfg)

    def test_water_ledger_closes(self, res):
        led = bed_ledger(res)
        assert abs(led["water_residual_rel"]) < 1e-9

    def test_energy_ledger_closes(self, res):
        led = bed_ledger(res)
        assert abs(led["energy_residual_rel"]) < 1e-2

    def test_ledger_books_are_non_trivial(self, res):
        led = res["ledger"]
        assert led["water_evaporated_kg"] > 0.1
        assert led["energy_in_gas_J"] > 0.0
        assert abs(led["water_in_gas_kg"] + led["water_in_solid_kg"]) > 0.0

    def test_latent_sink_removal_breaks_energy_ledger(self):
        """Negative control: without the latent sink the energy ledger screams."""
        cfg = dict(MAIN, n_cells=8, dt=5.0, t_end=300.0, include_latent_sink=False)
        bad = simulate_moist_drying(DryerBedConfig(**cfg))
        assert abs(bad["ledger"]["energy_residual_rel"]) > 0.01

    def test_wall_heat_source_is_booked(self):
        cfg = DryerBedConfig(**MAIN, n_cells=6, dt=5.0, t_end=120.0,
                            wall_temperature=500.0)
        res = simulate_moist_drying(cfg)
        assert res["ledger"]["energy_wall_source_J"] > 0.0
        assert abs(res["ledger"]["energy_residual_rel"]) < 1e-2


# ---------------------------------------------------------------------------
# (e) thin-layer kinetics
# ---------------------------------------------------------------------------
class TestKinetics:
    def test_page_fit_of_falling_rate(self):
        sim = simulate_single_particle(340.0, 0.02, 1.0, 0.004, 1250.0, 1500.0,
                                       293.15, 0.35, 0.02, 0.15, 6.0,
                                       t_end=900.0, dt=0.1)
        t, X = sim["t"], sim["X"]
        i_c = int(np.argmax(X < 0.15))
        assert 0 < i_c < len(t) - 10
        fit = fit_thin_layer(t[i_c:] - t[i_c], X[i_c:])
        assert fit["page"]["r2"] > 0.95
        assert fit["lewis"]["r2"] > 0.9
        assert 0.9 < fit["page"]["n"] < 1.6  # between Lewis (1.0) and Page shape

    def test_drying_rate_sign_and_monotonicity(self):
        sim = simulate_single_particle(340.0, 0.02, 1.0, 0.004, 1250.0, 1500.0,
                                       293.15, 0.35, 0.02, 0.15, 6.0,
                                       t_end=300.0, dt=0.1)
        dX = np.diff(sim["X"])
        assert np.all(dX <= 1e-15)
        assert sim["X"][-1] < sim["X"][0]

    def test_equilibrium_moisture_is_a_floor(self):
        sim = simulate_single_particle(340.0, 0.02, 1.0, 0.004, 1250.0, 1500.0,
                                       293.15, 0.35, 0.10, 0.15, 6.0,
                                       t_end=3000.0, dt=0.2)
        assert sim["X"][-1] >= 0.10 - 1e-12


# ---------------------------------------------------------------------------
# (f) degenerate dry limit
# ---------------------------------------------------------------------------
class TestDegenerateDryLimit:
    def _cfg(self):
        deg = dict(MAIN)
        deg.update(initial_moisture=0.0, equilibrium_moisture=0.0,
                   critical_moisture=0.0, gas_inlet_humidity=0.0)
        return DryerBedConfig(**deg, n_cells=8, dt=2.0, t_end=300.0)

    def test_moist_model_reduces_to_dry_model(self):
        cfg = self._cfg()
        moist = simulate_moist_drying(cfg)
        dry = simulate_dry_heating(cfg)
        assert np.abs(dry["T_p"] - moist["final"]["T_p"]).max() < 1e-9
        assert abs(dry["T_g_out"] - moist["final"]["T_g_out"]) < 1e-9

    def test_degenerate_run_has_zero_evaporation(self):
        res = simulate_moist_drying(self._cfg())
        assert res["ledger"]["water_evaporated_kg"] == 0.0
        assert abs(res["ledger"]["energy_residual_rel"]) < 1e-9

    @pytest.mark.skipif(not BASELINE.exists(), reason="baseline artifact not generated")
    def test_reproduces_committed_dry_baseline(self):
        """Bit-comparison against the committed baseline (<1e-9 absolute)."""
        import dataclasses

        base = json.loads(BASELINE.read_text())
        fields = {f.name for f in dataclasses.fields(DryerBedConfig)}
        cfg = DryerBedConfig(**{k: v for k, v in base["config"].items() if k in fields})
        dry = simulate_dry_heating(cfg)
        ref = np.array([float(v) for v in base["T_p_profile"]])
        assert np.abs(dry["T_p"] - ref).max() < 1e-9
        assert abs(dry["T_g_out"] - float(base["T_g_out"])) < 1e-9


# ---------------------------------------------------------------------------
# (g) monotonicity
# ---------------------------------------------------------------------------
class TestMonotonicity:
    @pytest.mark.parametrize("key,direction", [
        ("gas_inlet_temperature", -1),
        ("gas_inlet_velocity", -1),
        ("solid_feed_dry", +1),
    ])
    def test_outlet_moisture_monotone(self, key, direction):
        grid = {
            "gas_inlet_temperature": np.linspace(380.0, 460.0, 5),
            "gas_inlet_velocity": np.linspace(1.8, 4.0, 5),
            "solid_feed_dry": np.linspace(0.18, 0.36, 5),
        }[key]
        xs = []
        for v in grid:
            cfg = DryerBedConfig(**{**MAIN, key: float(v)})
            xs.append(solve_steady_profile(cfg, nz=120)["X_out"])
        d = np.diff(xs)
        assert np.all(np.sign(d) == direction), (key, xs)


# ---------------------------------------------------------------------------
# numerical consistency and guards
# ---------------------------------------------------------------------------
class TestNumerics:
    def test_cell_cascade_converges_towards_plug_flow(self):
        ref = solve_steady_profile(DryerBedConfig(**MAIN), nz=300)
        errs = []
        for nc in (10, 20):
            cfg = DryerBedConfig(**MAIN, n_cells=nc, dt=5.0, t_end=1200.0)
            res = simulate_moist_drying(cfg)
            X_ref = np.interp(res["final"]["z"], ref["z"], ref["X"])
            errs.append(np.abs(res["final"]["X"] - X_ref).max())
        assert errs[1] < errs[0]
        assert abs(res["final"]["X_out"] - ref["X_out"]) < 0.02

    def test_config_rejects_bad_equilibrium(self):
        with pytest.raises(ValueError):
            DryerBedConfig(equilibrium_moisture=0.3, critical_moisture=0.2)

    def test_config_rejects_zero_cells(self):
        with pytest.raises(ValueError):
            DryerBedConfig(n_cells=0)

    def test_specific_area_scaling(self):
        from tensordem.rotary_dryer.moist_transfer import specific_surface_area

        assert specific_surface_area(0.004, 1250.0) == pytest.approx(
            6.0 / (1250.0 * 0.004), rel=1e-12
        )

    def test_film_properties_are_arithmetic_mean(self):
        fp = film_properties(400.0, 300.0, 0.01)
        assert fp["T_f"] == pytest.approx(350.0)
        assert fp["Pr"] == pytest.approx(
            psy.air_viscosity(350.0) * psy.moist_air_cp(0.01) / psy.air_conductivity(350.0)
        )