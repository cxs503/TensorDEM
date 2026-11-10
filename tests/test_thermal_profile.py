"""Acceptance tests for the freezing-front thermal profile node (P316-W1-N03).

Covers the machine-checkable acceptance criteria of the node:

(a) Stefan analytical anchor — numerical freezing front vs
    ``x_f = sqrt(2 k DeltaT t/(rho L))`` (<= 2 %) and the exact Neumann
    similarity solution (<= 0.5 %);
(b) energy conservation — stored enthalpy == boundary flux integral, with the
    latent heat released in full (the defect of the predecessor module);
(c) uniform (m7439) limit — ``k -> inf`` (``Bi -> inf``) collapses the profile
    onto the ambient temperature and ``tau_eff -> tau_uniform`` (<= 0.1 %);
(d) deviation quantification — baseline ``tau_eff/tau_uniform`` for
    k = 2.2 W/mK, n = 0.3, DeltaT = 15 K, H = 1 m, t = 24 h and its monotone
    growth with the Biot number (Spearman rho = 1 over >= 5 levels);
(e) brine reduction — monotone decrease with the brine fraction and a
    bit-for-bit return to the fresh-water value at ``nu_b = 0``;
(f) negative control — DeltaT = 0 leaves the front displacement < 1e-9 m;
(g) engineering surface — the ``run_case`` result contract and the analytic
    helpers.

All tests are deterministic (no randomness) and use reduced meshes so the whole
file runs in well under a minute.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import spearmanr

from tensordem.freeze_bond.thermal_profile import (
    BondStrengthLaw,
    EnthalpyFrontSolver,
    IceThermal,
    deviation_ratio,
    effective_strength,
    frozen_depth_from_profile,
    front_position,
    local_brine_fraction,
    neumann_front,
    neumann_lambda,
    neumann_temperature_profile,
    run_biot_sweep,
    run_brine_sweep,
    run_case,
    solid_fraction,
    stefan_quasi_steady_front,
    uniform_limit_study,
    uniform_strength,
)

DT = 15.0


def _solve(delta_T=DT, n_cells=400, dt=10.0, t_end=3600.0, k=2.2, rec=None):
    th = IceThermal(k=k)
    solver = EnthalpyFrontSolver(n_cells=n_cells, domain_length=0.3, thermal=th)
    res = solver.run(T_env=th.liquidus_K - delta_T, t_end=t_end, dt=dt, h_conv=None,
                     record_times=rec or [t_end])
    return th, solver, res


# ---------------------------------------------------------------------------
# (a) Stefan / Neumann anchor
# ---------------------------------------------------------------------------
class TestCriterionA_StefanAnchor:
    def test_neumann_lambda_small_ste_asymptote(self):
        """For Ste << 1, lambda -> sqrt(Ste/2) (Neumann quasi-steady limit)."""
        for ste in (1e-3, 1e-2, 3e-2):
            lam = neumann_lambda(ste)
            assert abs(lam / math.sqrt(ste / 2.0) - 1.0) < 0.01

    def test_neumann_lambda_zero(self):
        assert neumann_lambda(0.0) == 0.0

    def test_quasi_steady_front_sqrt_law(self):
        """x_quasi = sqrt(2 k DeltaT t /(rho L)) and x ∝ sqrt(t)."""
        th = IceThermal(k=2.2)
        t = np.array([100.0, 400.0, 900.0, 1600.0])
        x = stefan_quasi_steady_front(t, th, DT)
        expected = np.sqrt(2.0 * th.k * DT * t / (th.rho * th.L))
        assert np.allclose(x, expected, rtol=1e-15)
        assert np.allclose(x / np.sqrt(t), x[0] / np.sqrt(t[0]), rtol=1e-14)

    def test_front_matches_stefan_formula(self):
        """(a) numerical front vs the quasi-steady Stefan formula, <= 2 %."""
        th, _, res = _solve(delta_T=5.0, n_cells=400, dt=10.0)
        x_num = float(res["front"][-1])
        x_qs = float(stefan_quasi_steady_front(3600.0, th, 5.0))
        assert abs(x_num - x_qs) / x_qs < 0.02

    def test_front_matches_exact_neumann(self):
        """(a) numerical front vs the exact Neumann solution, <= 0.5 %."""
        th, _, res = _solve(delta_T=5.0, n_cells=800, dt=5.0)
        x_num = float(res["front"][-1])
        x_nm = float(neumann_front(3600.0, th, 5.0))
        assert abs(x_num - x_nm) / x_nm < 0.005

    def test_front_sqrt_t_scaling(self):
        """The numerical front follows the self-similar sqrt(t) law."""
        th, _, res = _solve(delta_T=DT, n_cells=400, dt=10.0,
                            rec=[900.0, 1600.0, 2500.0, 3600.0])
        ratio = res["front"] / np.sqrt(res["time"])
        assert np.std(ratio) / np.mean(ratio) < 0.03
        # and the absolute level is anchored
        x_nm = neumann_front(3600.0, th, DT)
        assert abs(res["front"][-1] / x_nm - 1.0) < 0.01

    def test_solid_fraction_and_enthalpy_inverse(self):
        """s(T) is 0/1 outside the band and e(T) is invertible to round-off."""
        th = IceThermal()
        solver = EnthalpyFrontSolver(n_cells=50, domain_length=0.1, thermal=th, band=0.2)
        T_f = th.liquidus_K
        assert solid_fraction(T_f + 1.0, T_f, 0.2) == 0.0
        assert solid_fraction(T_f - 1.0, T_f, 0.2) == 1.0
        assert abs(solid_fraction(T_f - 0.1, T_f, 0.2) - 0.5) < 1e-12
        T = np.linspace(T_f - 12.0, T_f + 3.0, 97)
        assert np.allclose(solver.temperature(solver.enthalpy(T)), T, atol=1e-9)

    def test_enthalpy_is_monotone_in_temperature(self):
        """de/dT > 0 everywhere: the phase-change map is invertible."""
        th = IceThermal()
        solver = EnthalpyFrontSolver(n_cells=40, domain_length=0.1, thermal=th, band=0.05)
        T = np.linspace(th.liquidus_K - 20.0, th.liquidus_K + 5.0, 5001)
        e = solver.enthalpy(T)
        assert np.all(np.diff(e) > 0.0)


# ---------------------------------------------------------------------------
# (b) Energy conservation
# ---------------------------------------------------------------------------
class TestCriterionB_EnergyConservation:
    def test_stored_enthalpy_equals_flux_integral(self):
        """(b) |stored - flux| / |flux| < 1e-4 at every record time."""
        _, _, res = _solve(n_cells=400, dt=10.0, rec=[600.0, 1800.0, 3600.0])
        rel = np.abs(res["energy_stored"] - res["energy_flux"]) / np.abs(res["energy_flux"])
        assert np.max(rel) < 1e-4
        assert np.max(rel) < 1e-9           # in practice: round-off

    def test_latent_heat_released_in_full(self):
        """(b) latent = rho L x_f and latent + sensible == flux (<= 1e-4)."""
        th, solver, res = _solve(n_cells=400, dt=10.0)
        T = res["profiles"][-1]
        s = solid_fraction(T, th.liquidus_K, solver.band)
        latent = th.rho * th.L * float(np.sum(s) * solver.dz)
        sensible = th.rho * th.cp * float(np.sum(T - th.liquidus_K) * solver.dz)
        flux = float(np.abs(res["energy_flux"][-1]))
        assert abs((latent - sensible) - flux) / flux < 1e-4
        # the released latent heat per unit area equals rho L x_f: no L/2 deficit
        x_int = float(res["front_integral"][-1])
        assert abs(latent - th.rho * th.L * x_int) / latent < 1e-12
        # sanity: latent dominates sensible at Ste ~ 0.09
        assert latent > 10.0 * abs(sensible)

    def test_energy_monotone_and_negative(self):
        """Cooling a warm layer removes energy: stored enthalpy decreases."""
        _, _, res = _solve(n_cells=200, dt=10.0, rec=[600.0, 1800.0, 3600.0])
        assert np.all(np.diff(res["energy_stored"]) < 0.0)


# ---------------------------------------------------------------------------
# (c) Uniform (m7439) limit
# ---------------------------------------------------------------------------
class TestCriterionC_UniformLimit:
    def test_flat_profile_calibers_agree_bitwise(self):
        """(c) a flat profile gives tau_eff == tau_uniform to round-off."""
        th = IceThermal(k=2.2)
        law = BondStrengthLaw()
        T_env = th.liquidus_K - DT
        z = np.linspace(0.0, 1.0, 2001)
        tau_eff = effective_strength(z, np.full_like(z, T_env), 1.0, law, th)
        tau_uni = uniform_strength(T_env, law, th)
        assert abs(tau_eff - tau_uni) / tau_uni < 1e-12

    def test_uniform_limit_k_to_infinity(self):
        """(c) tau_eff/tau_uniform -> 1 within 0.1 % as k -> inf."""
        ul = uniform_limit_study()
        assert abs(ul["rows"][-1]["ratio"] - 1.0) < 1e-3
        assert abs(ul["flat_profile"]["rel_diff"]) < 1e-12

    def test_uniform_limit_is_monotone(self):
        """The ratio increases monotonically with k (no overshoot)."""
        ul = uniform_limit_study(k_levels=(2.2, 22.0, 220.0, 2200.0, 2.2e4, 2.2e6))
        ratios = [r["ratio"] for r in ul["rows"]]
        assert all(b > a for a, b in zip(ratios, ratios[1:]))

    def test_numerical_matches_neumann_profile(self):
        """(c) solver vs exact Neumann profile at a resolvable k (<= 2 %)."""
        th = IceThermal(k=22.0)
        out = run_case(k=22.0, h_conv=None, delta_T_K=DT, H=1.0, t_end=86400.0,
                       n_cells=400, dt=60.0, domain_length=4.0)
        z = np.linspace(0.0, 1.0, 2001)
        T_an = neumann_temperature_profile(z, 86400.0, th, DT)
        ratio_an = effective_strength(z, T_an, 1.0, BondStrengthLaw(), th) \
            / out["strength"]["tau_uniform_Pa"]
        assert abs(out["strength"]["ratio"] / ratio_an - 1.0) < 0.02


# ---------------------------------------------------------------------------
# (d) Baseline deviation and Biot monotonicity
# ---------------------------------------------------------------------------
class TestCriterionD_Deviation:
    def test_baseline_case_value(self):
        """(d) 24 h baseline: the ratio is a definite number ~0.10."""
        out = run_case(k=2.2, exponent_n=0.3, delta_T_K=15.0, H=1.0, t_end=86400.0,
                       n_cells=400, dt=60.0)
        ratio = out["strength"]["ratio"]
        assert 0.09 < ratio < 0.115
        assert out["strength"]["tau_eff_Pa"] < out["strength"]["tau_uniform_Pa"]
        # the uniform caliper overestimates by ~1/ratio
        assert 8.0 < 1.0 / ratio < 12.0

    def test_baseline_front_and_energy(self):
        """(d) baseline front ~0.134 m at 24 h, energy closes."""
        out = run_case(k=2.2, exponent_n=0.3, delta_T_K=15.0, H=1.0, t_end=86400.0,
                       n_cells=400, dt=60.0)
        assert abs(out["front_m"]["numerical"] / out["front_m"]["neumann"] - 1.0) < 0.01
        assert abs(out["energy"]["rel_residual"]) < 1e-9

    def test_ratio_monotone_in_biot(self):
        """(d) Spearman rho = 1 over >= 5 Bi levels."""
        sweep = run_biot_sweep(biot_levels=(0.0044, 0.0088, 0.0176, 0.0352, 0.0704),
                               h_conv=500.0, exponent_n=0.3, delta_T_K=15.0, H=1.0,
                               t_end=86400.0, n_cells=300, dt=60.0)
        ratios = [r["ratio"] for r in sweep["rows"]]
        bis = [r["biot"] for r in sweep["rows"]]
        assert sweep["monotone_increasing"]
        assert spearmanr(bis, ratios)[0] == pytest.approx(1.0, abs=1e-12)

    def test_ratio_grid_convergence(self):
        """The reported ratio is grid-converged to a few 1e-3."""
        r400 = run_case(k=2.2, exponent_n=0.3, delta_T_K=15.0, H=1.0, t_end=86400.0,
                        n_cells=400, dt=60.0)["strength"]["ratio"]
        r800 = run_case(k=2.2, exponent_n=0.3, delta_T_K=15.0, H=1.0, t_end=86400.0,
                        n_cells=800, dt=60.0)["strength"]["ratio"]
        assert abs(r800 / r400 - 1.0) < 0.02

    def test_tau_uniform_is_the_m7439_caliper(self):
        """tau_uniform is sigma_b at the ambient temperature (definition)."""
        th = IceThermal(k=2.2)
        law = BondStrengthLaw(exponent_n=0.3)
        T_env = th.liquidus_K - 15.0
        expected = law.sigma_0 * (DT / law.T_ref) ** law.exponent_n
        assert uniform_strength(T_env, law, th) == pytest.approx(expected, rel=1e-14)


# ---------------------------------------------------------------------------
# (e) Brine reduction
# ---------------------------------------------------------------------------
class TestCriterionE_BrineReduction:
    def test_brine_reduces_strength_monotonically(self):
        """(e) tau_eff decreases strictly with the brine fraction."""
        sweep = run_brine_sweep(brine_levels=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10),
                                k=2.2, exponent_n=0.3, delta_T_K=15.0, H=1.0,
                                t_end=86400.0, n_cells=300, dt=60.0)
        assert sweep["monotone_decreasing"]
        tau = sweep["tau_eff_Pa"]
        assert all(b < a for a, b in zip(tau, tau[1:]))

    def test_zero_brine_is_bitwise_fresh_water(self):
        """(e) nu_b = 0 returns the fresh-water value (< 1e-12)."""
        kw = dict(k=2.2, exponent_n=0.3, delta_T_K=15.0, H=1.0, t_end=86400.0,
                  n_cells=300, dt=60.0)
        pure = run_case(**kw)["strength"]["tau_eff_Pa"]
        zero = run_case(brine_override=0.0, **kw)["strength"]["tau_eff_Pa"]
        assert abs(zero - pure) / pure < 1e-12

    def test_cox_weeks_liquid_fraction(self):
        """nu_b grows with temperature and vanishes for fresh water."""
        th = IceThermal(salinity_ppt=5.0)
        T_f = th.liquidus_K
        nu = [local_brine_fraction(T_f + t, th) for t in (-20.0, -10.0, -2.0)]
        assert nu[0] < nu[1] < nu[2]
        assert local_brine_fraction(T_f - 5.0, IceThermal(salinity_ppt=0.0)) == 0.0

    def test_saline_liquidus(self):
        """Liquidus from the repo's Cox & Weeks (1983) constants.

        ``T_f = -(0.05411 S + 7.594e-5 S^2)`` degC (same constants as
        ``tensordem.salinity_strength``): -1.81 degC at S = 32 ppt,
        -1.99 degC at S = 35 ppt.
        """
        assert IceThermal(salinity_ppt=32.0).liquidus_K - 273.15 == pytest.approx(-1.81, abs=0.03)
        assert IceThermal(salinity_ppt=35.0).liquidus_K - 273.15 == pytest.approx(-1.99, abs=0.03)

    def test_saline_case_is_weaker_than_fresh(self):
        """Coupled salinity lowers both the liquidus and the strength."""
        kw = dict(k=2.2, exponent_n=0.3, delta_T_K=15.0, H=1.0, t_end=86400.0,
                  n_cells=400, dt=60.0)
        fresh = run_case(**kw)
        saline = run_case(salinity_ppt=5.0, **kw)
        assert saline["strength"]["ratio"] < fresh["strength"]["ratio"]
        assert saline["parameters"]["liquidus_K"] < fresh["parameters"]["liquidus_K"]


# ---------------------------------------------------------------------------
# (f) Negative control
# ---------------------------------------------------------------------------
class TestCriterionF_NegativeControl:
    def test_zero_delta_T_leaves_front_at_zero(self):
        """(f) DeltaT = 0: front displacement < 1e-9 m, no strength."""
        th = IceThermal(k=2.2)
        solver = EnthalpyFrontSolver(n_cells=200, domain_length=0.3, thermal=th)
        res = solver.run(T_env=th.liquidus_K, t_end=86400.0, dt=60.0, h_conv=None,
                         record_times=[86400.0])
        assert abs(float(res["front"][-1])) < 1e-9
        assert np.max(np.abs(res["profiles"][-1] - th.liquidus_K)) < 1e-9
        assert float(res["energy_flux"][-1]) == pytest.approx(0.0, abs=1e-9)

    def test_zero_stefan_number_has_no_front(self):
        th = IceThermal()
        assert float(stefan_quasi_steady_front(3600.0, th, 0.0)) == 0.0
        assert float(neumann_front(3600.0, th, 0.0)) == 0.0

    def test_front_position_is_monotone_and_subcell(self):
        """Only positive (cold) surface cells produce a front."""
        z = np.array([0.5, 1.5, 2.5, 3.5]) * 1e-3
        s = np.array([1.0, 0.75, 0.25, 0.0])
        x = front_position(z, s, 1e-3)
        assert 1.5e-3 < x < 2.5e-3
        assert front_position(z, np.zeros(4), 1e-3) == 0.0


# ---------------------------------------------------------------------------
# (g) Engineering surface / contract
# ---------------------------------------------------------------------------
class TestCriterionG_Contract:
    def test_run_case_contract(self):
        out = run_case(k=2.2, delta_T_K=DT, H=1.0, t_end=3600.0, n_cells=200, dt=30.0)
        for key in ("parameters", "z", "T_profile", "sigma_profile", "front_m",
                    "energy", "strength"):
            assert key in out
        for key in ("numerical", "integral", "quasi_steady", "neumann",
                    "rel_err_vs_quasi", "rel_err_vs_neumann"):
            assert key in out["front_m"]
        for key in ("tau_eff_Pa", "tau_uniform_Pa", "ratio", "tau_frozen_caliber_Pa"):
            assert key in out["strength"]
        assert out["sigma_profile"].shape == out["T_profile"].shape
        assert np.all(out["sigma_profile"] >= 0.0)

    def test_deviation_ratio_helper_matches_run_case(self):
        th = IceThermal(k=2.2)
        law = BondStrengthLaw(exponent_n=0.3)
        out = run_case(k=2.2, exponent_n=0.3, delta_T_K=DT, H=1.0, t_end=3600.0,
                       n_cells=200, dt=30.0)
        r = deviation_ratio(out["z"], out["T_profile"], 1.0,
                            th.liquidus_K - DT, law, th)
        assert r == pytest.approx(out["strength"]["ratio"], rel=1e-12)

    def test_frozen_depth_from_profile(self):
        th = IceThermal(k=2.2)
        solver = EnthalpyFrontSolver(n_cells=200, domain_length=0.3, thermal=th)
        res = solver.run(T_env=th.liquidus_K - DT, t_end=3600.0, dt=10.0, h_conv=None,
                         record_times=[3600.0])
        x_int = frozen_depth_from_profile(res["z"], res["profiles"][-1], th, solver.band)
        assert abs(x_int / float(res["front_integral"][-1]) - 1.0) < 1e-9

    def test_front_reaching_wall_is_flagged(self):
        """A run whose front reaches the far wall stops with a flag, not a crash."""
        th = IceThermal(k=2.2)
        solver = EnthalpyFrontSolver(n_cells=100, domain_length=0.1, thermal=th)
        res = solver.run(T_env=th.liquidus_K - DT, t_end=86400.0, dt=60.0,
                         h_conv=None, record_times=[86400.0])
        assert res["front_reached_wall"] is True
        # the recorded state is at the wall, and it is not the requested t_end
        assert float(res["front"][-1]) > 0.9 * 0.1
        assert float(res["time"][-1]) < 86400.0