"""Tests for the ice-hull-propeller three-field coupling package."""
from __future__ import annotations

import math
import unittest

import torch

from tensordem.threefield import (
    BreakupProperties,
    CouplingConfig,
    Environment,
    HullGeometry,
    HullIceCase,
    IceProperties,
    LBMConfig,
    ThermalProperties,
    analytic_diameter_ransom_marshall,
    analytic_ransom_marshall_time,
    cdf_agreement_r2,
    critical_floe_length,
    fit_weibull,
    fit_weibull_cdf,
    fragment,
    gaussian_race_profile,
    heat_transfer_coefficient,
    integrate_melting,
    lindqvist_resistance,
    mass_ledger,
    nusselt_ransom_marshall,
    profile_error,
    relative_error_pct,
    resistance_band,
    riska_resistance,
    run_coupled_pass,
    run_hull_ice_case,
    weibull_cdf,
)


class ResistanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ice = IceProperties(thickness_m=0.6)
        self.hull = HullGeometry()
        self.env = Environment()

    def test_lindqvist_components_sum_to_total(self) -> None:
        result = lindqvist_resistance(self.ice, self.hull, 0.5, self.env)
        components = (
            result["crushing_N"]
            + result["bending_N"]
            + result["submersion_N"]
            + result["speed_dependent_N"]
        )
        self.assertAlmostEqual(result["total_N"], components, places=6)
        for key in ("crushing_N", "bending_N", "submersion_N"):
            self.assertGreater(result[key], 0.0)

    def test_riska_is_speed_linear(self) -> None:
        slow = riska_resistance(self.ice, self.hull, 0.0, self.env)
        fast = riska_resistance(self.ice, self.hull, 1.0, self.env)
        self.assertAlmostEqual(fast["total_N"] - slow["total_N"], fast["c2_N_per_m_s"], places=6)
        self.assertGreater(fast["c1_N"], 0.0)

    def test_resistance_band_orders_models(self) -> None:
        band = resistance_band(self.ice, self.hull, 0.5, self.env)
        self.assertLessEqual(band["band_low_N"], band["band_centre_N"])
        self.assertLessEqual(band["band_centre_N"], band["band_high_N"])
        self.assertGreater(band["band_spread_pct"], 0.0)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            IceProperties(thickness_m=-0.1)
        with self.assertRaises(ValueError):
            HullGeometry(waterline_entry_angle_deg=95.0)
        with self.assertRaises(ValueError):
            lindqvist_resistance(self.ice, self.hull, -1.0, self.env)

    def test_relative_error_is_symmetric(self) -> None:
        self.assertAlmostEqual(relative_error_pct(110.0, 100.0), 10.0, places=9)
        self.assertAlmostEqual(relative_error_pct(90.0, 100.0), 10.0, places=9)


class ThermalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.props = ThermalProperties()

    def test_nusselt_limits(self) -> None:
        self.assertAlmostEqual(nusselt_ransom_marshall(0.0, 12.5), 2.0, places=12)
        self.assertGreater(nusselt_ransom_marshall(1000.0, 12.5), nusselt_ransom_marshall(10.0, 12.5))

    def test_heat_transfer_matches_correlation(self) -> None:
        groups = heat_transfer_coefficient(0.05, 0.2, self.props)
        expected = groups["nusselt"] * self.props.water_conductivity_w_m_k / 0.05
        self.assertAlmostEqual(groups["h_w_m2_k"], expected, places=10)
        self.assertAlmostEqual(
            groups["reynolds"], 0.2 * 0.05 / self.props.water_kinematic_viscosity_m2_s, places=10
        )

    def test_conduction_limit_matches_square_law(self) -> None:
        d0 = 0.02
        melt_time = analytic_ransom_marshall_time(d0, d0 * 1e-6, 0.0, self.props, include_sensible=False)
        history = integrate_melting(
            d0, self.props, relative_velocity_m_s=0.0,
            duration_s=melt_time * 0.5, dt_s=melt_time / 400.0, include_sensible=False,
        )
        mid = history[-1]
        self.assertTrue(math.isclose(mid["time_s"], melt_time * 0.5, rel_tol=1e-12))
        # D^2 = D0^2 - 2 A t  =>  at t = T/2 the sphere area is exactly halved.
        expected = d0 / math.sqrt(2.0)
        self.assertLess(abs(mid["diameter_m"] - expected) / expected, 0.01)

    def test_ransom_marshall_numeric_matches_closed_form(self) -> None:
        d0 = 0.03
        velocity = 0.25
        duration = analytic_ransom_marshall_time(
            d0, d0 * 0.9, velocity, self.props, include_sensible=False
        )
        history = integrate_melting(
            d0, self.props, relative_velocity_m_s=velocity,
            duration_s=duration, dt_s=duration / 200.0, include_sensible=False,
        )
        numeric = history[-1]["diameter_m"]
        analytic = analytic_diameter_ransom_marshall(
            d0, duration, velocity, self.props, include_sensible=False
        )
        self.assertLess(abs(numeric - analytic) / analytic, 0.05)

    def test_melting_reduces_mass_monotonically(self) -> None:
        history = integrate_melting(
            0.02, self.props, relative_velocity_m_s=0.3, duration_s=2.0, dt_s=0.02,
            include_sensible=False,
        )
        masses = [row["mass_kg"] for row in history]
        self.assertTrue(all(b <= a + 1e-18 for a, b in zip(masses, masses[1:])))
        self.assertGreater(history[-1]["melt_fraction"], 0.0)
        self.assertAlmostEqual(history[0]["melt_fraction"], 0.0, places=12)

    def test_cold_water_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ThermalProperties(water_temperature_c=-1.0)


class FragmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.props = BreakupProperties(thickness_m=0.2, flexural_strength_pa=5.0e5)

    def test_critical_length_scales_with_inverse_velocity(self) -> None:
        slow = critical_floe_length(0.1, self.props)
        fast = critical_floe_length(0.2, self.props)
        self.assertAlmostEqual(slow / fast, 2.0, places=9)
        self.assertEqual(critical_floe_length(0.0, self.props), math.inf)

    def test_fragmentation_conserves_mass(self) -> None:
        sizes = [2.0, 3.0, 4.5, 7.0]
        result = fragment(sizes, 5.0, self.props, generator=torch.Generator().manual_seed(3))
        self.assertLess(abs(result["mass_error_kg"]) / result["initial_mass_kg"], 1e-12)
        self.assertGreater(result["split_events"], 0)

    def test_stable_floes_are_untouched(self) -> None:
        sizes = [0.05, 0.06]
        result = fragment(sizes, 0.01, self.props, generator=torch.Generator().manual_seed(1))
        self.assertEqual(result["split_events"], 0)
        self.assertEqual(len(result["lengths_m"]), len(sizes))

    def test_weibull_fit_recovers_parameters(self) -> None:
        samples = [
            weibull_cdf(0.0, 1.0, 1.0)  # touch the API to keep the import meaningful
        ]
        self.assertEqual(samples, [0.0])
        generator = torch.Generator().manual_seed(11)
        uniforms = torch.rand(4000, generator=generator, dtype=torch.float64)
        shape, scale = 1.8, 0.4
        data = [scale * (-math.log(1.0 - float(u))) ** (1.0 / shape) for u in uniforms]
        fit = fit_weibull(data)
        self.assertLess(abs(fit["shape"] - shape) / shape, 0.15)
        self.assertGreater(fit["r_squared"], 0.95)
        cdf_fit = fit_weibull_cdf(data)
        self.assertGreater(cdf_fit["r_squared"], 0.99)
        agreement = cdf_agreement_r2(data, shape, scale)
        self.assertGreater(agreement["r_squared"], 0.99)


class WakeTests(unittest.TestCase):
    def test_equilibrium_recovers_moments(self) -> None:
        config = LBMConfig(nx=16, ny=12, nz=12, disk_x=8, disk_radius=2.5, dtype="float64")
        sim = __import__(
            "tensordem.threefield.wake_lbm", fromlist=["WakeLBM"]
        ).WakeLBM(config)
        rho = torch.full((4, 5, 6), 1.3, dtype=torch.float64)
        u = torch.zeros((4, 5, 6, 3), dtype=torch.float64)
        u[..., 0] = 0.05
        u[..., 1] = -0.01
        f = sim._equilibrium(rho.unsqueeze(-1), u)
        rho_back, u_back = sim._macroscopic(f)
        self.assertTrue(torch.allclose(rho_back, rho, atol=1e-12))
        self.assertTrue(torch.allclose(u_back, u, atol=1e-12))

    def test_short_run_is_stable_and_flux_conserving(self) -> None:
        from tensordem.threefield.wake_lbm import WakeLBM

        config = LBMConfig(nx=32, ny=12, nz=12, disk_x=8, disk_radius=2.5, dtype="float64",
                           inlet_perturbation=0.0)
        sim = WakeLBM(config)
        sim.run(40)
        self.assertTrue(bool(torch.isfinite(sim.f).all()))
        mass_fluxes = [sim.mass_flux(station) for station in (2, 16, 30)]
        spread = (max(mass_fluxes) - min(mass_fluxes)) / max(mass_fluxes)
        self.assertLess(spread, 0.05)
        self.assertGreater(sim.thrust_lattice(), 0.0)

    def test_gaussian_model_conserves_thrust_momentum(self) -> None:
        thrust = 0.05
        model = gaussian_race_profile(
            [0.0, 1.0, 2.0, 3.0, 4.0], inflow_velocity=0.04, thrust=thrust,
            distance=20.0, spreading_rate=0.085, sigma_floor=1.0,
        )
        self.assertAlmostEqual(model["momentum_flux"], thrust, places=12)
        self.assertGreater(model["centreline_excess"], 0.0)

    def test_profile_error_metrics(self) -> None:
        reference = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(profile_error(reference, reference)["relative_l2_error"], 0.0)
        shifted = [2.0, 3.0, 4.0]
        self.assertGreater(profile_error(shifted, reference)["relative_l2_error"], 0.0)
        with self.assertRaises(ValueError):
            profile_error([1.0], [1.0, 2.0])

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            LBMConfig(tau0=0.4)
        with self.assertRaises(ValueError):
            LBMConfig(jet_velocity=0.3)
        with self.assertRaises(ValueError):
            LBMConfig(nx=4)


class HullIceTests(unittest.TestCase):
    def _case(self, **kwargs) -> HullIceCase:
        defaults = dict(
            thickness_m=0.15, particle_radius_m=0.025, n_length=5, n_width=2, steps=40
        )
        defaults.update(kwargs)
        return HullIceCase(**defaults)

    def test_geometry_bookkeeping(self) -> None:
        case = self._case()
        self.assertEqual(case.layers, 3)
        self.assertAlmostEqual(case.realised_thickness_m, 0.15, places=12)
        self.assertAlmostEqual(case.model_width_m, 0.1, places=12)

    def test_case_runs_and_conserves_ice_mass(self) -> None:
        result = run_hull_ice_case(self._case())
        fragments = result["fragments"]
        expected = fragments["cluster_count"] * fragments["particle_mass_kg"] * 0.0 + sum(
            fragments["masses_kg"]
        )
        self.assertAlmostEqual(fragments["total_mass_kg"], expected, places=9)
        self.assertAlmostEqual(
            fragments["total_mass_kg"],
            result["case"]["particle_count"] * fragments["particle_mass_kg"],
            places=9,
        )
        self.assertGreaterEqual(result["peak_ice_resistance_N"], 0.0)
        self.assertGreater(result["case"]["bow_travel_m"], 0.0)

    def test_case_validation(self) -> None:
        with self.assertRaises(ValueError):
            self._case(wedge_angle_deg=95.0)
        with self.assertRaises(ValueError):
            self._case(steps=1)


class CouplingTests(unittest.TestCase):
    def test_mass_ledger_is_exact(self) -> None:
        ledger = mass_ledger(
            initial_mass_kg=10.0, surviving_mass_kg=7.25, melted_mass_kg=2.75
        )
        self.assertLess(ledger["relative_error"], 1e-15)
        with self.assertRaises(ValueError):
            mass_ledger(initial_mass_kg=1.0, surviving_mass_kg=2.0, melted_mass_kg=-1.0)

    def test_coupled_pass_closes_the_ledger(self) -> None:
        case = run_hull_ice_case(
            HullIceCase(thickness_m=0.15, particle_radius_m=0.025, n_length=5,
                        n_width=2, steps=40)
        )
        profile = {
            "radius": [0.0, 0.5, 1.0, 2.0],
            "mean_axial_velocity": [0.12, 0.10, 0.07, 0.045],
            "axial_rms_velocity": [0.006, 0.005, 0.004, 0.002],
        }
        coupled = run_coupled_pass(
            dem_case=case, wake_profile=profile, config=CouplingConfig(residence_time_s=5.0)
        )
        self.assertLess(coupled["ledger"]["relative_error"], 1e-6)
        self.assertGreater(coupled["initial_mass_kg"], 0.0)
        self.assertLessEqual(
            coupled["thermal"]["remaining_mass_kg"] + coupled["thermal"]["melted_mass_kg"],
            coupled["initial_mass_kg"] + 1e-9,
        )
        self.assertIn("weibull_reference_agreement", coupled)


if __name__ == "__main__":
    unittest.main()