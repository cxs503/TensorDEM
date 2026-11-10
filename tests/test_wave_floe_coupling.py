"""Wave-ice coupling verification suite (p348w2-n07).

Every test here corresponds to one of the campaign criteria (C1..C7 in
``tensordem.wave_floe.validation``), or to the defect it exposed in the shipped
wet-coupling kernel.  The heavy end-to-end campaign (full RAO sweep, ensemble
spectral study) lives in ``scripts/run_wave_floe_validation.py``; these tests keep
the same physics at a smaller size so they run in CI.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.buoyancy_wet import (
    WaveForceConfig,
    WetCouplingConfig,
    compute_wave_force,
    wave_kinematics,
    wave_kinematics_full,
)
from tensordem.wave_floe.array import (
    FloeArrayTransmission,
    single_interface_transmission,
    slab_transmission_analytic,
)
from tensordem.wave_floe.breakup import (
    WinklerBeam,
    characteristic_wavenumber,
    critical_wave_height,
    theory_bending_stress,
)
from tensordem.wave_floe.response import (
    FloeLattice,
    FloeResponseModel,
    cell_potential,
    cell_submerged_area,
    continuum_coefficients,
    discrete_coefficients,
    frequency_response,
    rao_from_time_series,
)
from tensordem.wave_floe.spectra import (
    discretize,
    ensemble_psd,
    jonswap,
    pierson_moskowitz,
    spectral_fidelity,
    spectrum_grid,
    synthesize_eta,
    welch_psd,
)
from tensordem.wave_floe.waves import RegularWave, solve_wavenumber, wave_energy_flux

G = 9.81


# ---------------------------------------------------------------------------
# Waves and dispersion
# ---------------------------------------------------------------------------

class TestDispersion:
    def test_deep_water_limit(self):
        for T in (2.0, 5.0, 12.0):
            omega = 2.0 * math.pi / T
            assert solve_wavenumber(omega, None) == pytest.approx(omega ** 2 / G, rel=1e-15)

    @pytest.mark.parametrize("depth", [2.0, 10.0, 50.0])
    def test_finite_depth_residual(self, depth):
        for T in (3.0, 7.0, 15.0):
            omega = 2.0 * math.pi / T
            k = solve_wavenumber(omega, depth)
            assert abs(omega ** 2 - G * k * math.tanh(k * depth)) / omega ** 2 < 1e-12

    def test_shallow_water_limit(self):
        # kd -> 0: w^2 = g k^2 d  =>  k = w/sqrt(g d)
        depth, T = 0.5, 60.0
        omega = 2.0 * math.pi / T
        k = solve_wavenumber(omega, depth)
        assert k * depth < 0.05
        assert k == pytest.approx(omega / math.sqrt(G * depth), rel=2e-3)

    def test_energy_flux_matches_pressure_work(self):
        for T in (4.0, 8.0, 14.0):
            out = wave_energy_flux(RegularWave(height=0.4, period=T))
            assert out["rel_error"] < 1e-4

    def test_group_velocity_deep_limit(self):
        wave = RegularWave(height=0.1, period=8.0)
        assert wave.group_velocity == pytest.approx(0.5 * wave.phase_speed, rel=1e-12)

    def test_kinematics_free_surface_relations(self):
        """Deep water: ``w(z=0) = d(eta)/dt`` and ``dw/dt(z=0) = -w^2 eta``."""
        wave = RegularWave(height=0.2, period=7.0)
        eps = 1e-6
        for t in (0.0, 1.3, 2.9):
            kin = wave.kinematics(np.array([1.0]), np.array([0.0]), t)
            eta = wave.surface_elevation(np.array([1.0]), t)
            d_eta = (wave.surface_elevation(np.array([1.0]), t + eps)
                     - wave.surface_elevation(np.array([1.0]), t - eps)) / (2 * eps)
            assert kin["w"][0] == pytest.approx(d_eta[0], abs=1e-6)
            assert kin["dw_dt"][0] == pytest.approx(-wave.omega ** 2 * eta[0], rel=1e-10)


# ---------------------------------------------------------------------------
# Spectra (C4)
# ---------------------------------------------------------------------------

class TestSpectra:
    def test_pierson_moskowitz_normalisation(self):
        grid = spectrum_grid(8.0, n=4000)
        spec = pierson_moskowitz(grid, significant_height=2.0, peak_period=8.0)
        assert spec.m0 == pytest.approx(2.0 ** 2 / 16.0, rel=2e-3)

    def test_jonswap_normalisation(self):
        grid = spectrum_grid(9.0, n=4000)
        for gamma in (1.0, 3.3, 6.0):
            spec = jonswap(grid, significant_height=1.5, peak_period=9.0, gamma=gamma)
            assert spec.significant_height == pytest.approx(1.5, rel=2e-3)
            assert spec.peak_period == pytest.approx(9.0, rel=2e-2)

    def test_discretisation_keeps_variance(self):
        grid = spectrum_grid(8.0, n=200)
        spec = jonswap(grid, 2.0, 8.0, 3.3)
        _, amp, _ = discretize(spec, phase_seed=1)
        assert float(np.sum(amp ** 2) / 2.0) == pytest.approx(spec.m0, rel=1e-2)

    def test_realisation_variance_matches_spectrum(self):
        grid = spectrum_grid(8.0, n=200)
        spec = jonswap(grid, 2.0, 8.0, 3.3)
        omega, amp, _ = discretize(spec, phase_seed=2)
        t = np.arange(0.0, 600.0, 0.2)
        eta = synthesize_eta(omega, amp, np.random.default_rng(3).uniform(0, 2 * math.pi,
                                                                          omega.size), t)
        assert eta.var() == pytest.approx(spec.m0, rel=0.1)

    def test_welch_recover_variance(self):
        """Parseval: the one-sided PSD must integrate back to the variance."""
        rng = np.random.default_rng(7)
        dt = 0.05
        t = np.arange(0.0, 400.0, dt)
        eta = 0.3 * np.sin(0.8 * t + 0.4) + 0.1 * rng.standard_normal(t.size)
        omega, S = welch_psd(eta, dt, segment=1024)
        assert float(np.trapezoid(S, omega)) == pytest.approx(eta.var(), rel=0.08)

    def test_spectral_fidelity_of_jonswap_realisation(self):
        """C4: R^2 >= 0.95 between the realised sea and the target JONSWAP."""
        grid = spectrum_grid(8.0, n=200)
        spec = jonswap(grid, 2.0, 8.0, 3.3)
        omega, amp, _ = discretize(spec, phase_seed=0)
        om, S = ensemble_psd(omega, amp, np.array([0.0]), dt=8.0 / 32.0,
                             duration=300 * 8.0, n_realisations=8,
                             segment=int(16 * 8.0 / (8.0 / 32.0)), seed0=11)
        fid = spectral_fidelity(spec, om, S)
        assert fid["r2"] >= 0.95
        assert fid["hs_ratio"] == pytest.approx(1.0, abs=0.02)
        assert fid["peak_omega_error"] < 0.05


# ---------------------------------------------------------------------------
# Floe response (C1, C5, C6)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def floe():
    lat = FloeLattice(length=20.0, thickness=1.0, n_x=40, n_y=2)
    return lat, lat.equilibrium_height()


class TestFloeGeometry:
    def test_equilibrium_draft(self, floe):
        lat, z_eq = floe
        assert lat.draft == pytest.approx(917.0 / 1025.0 * 1.0, rel=1e-12)
        # the centroid sits (draft - h/2) *below* the waterline, so the offset is
        # negative (z is up)
        assert z_eq == pytest.approx(0.5 * lat.thickness - lat.draft, abs=1e-9)

    def test_potential_derivative_is_area(self, floe):
        lat, _ = floe
        b = np.linspace(-1.5 * lat.dy, 0.5 * lat.dy, 41)
        eps = 1e-7
        dphi = (cell_potential(b + eps, lat.dx, lat.dy)
                - cell_potential(b - eps, lat.dx, lat.dy)) / (2 * eps)
        np.testing.assert_allclose(dphi, -cell_submerged_area(b, lat.dx, lat.dy), atol=1e-6)

    def test_waterplane_stiffness_is_exact(self, floe):
        lat, z_eq = floe
        coeffs = discrete_coefficients(lat, z_eq)
        assert coeffs.k33 == pytest.approx(1025.0 * 9.81 * lat.length, rel=1e-12)


class TestFloeResponse:
    def test_long_wave_limit_rao_unity(self, floe):
        """C1 anchor: a floe much shorter than the wave rides the surface."""
        lat, z_eq = floe
        for T in (60.0, 120.0):
            wave = RegularWave(height=0.2, period=T)
            fd = frequency_response(wave, lat, z_eq, reference="continuum")
            assert fd.rao_heave == pytest.approx(1.0, abs=5e-3)
            assert fd.rao_pitch == pytest.approx(1.0, abs=5e-3)

    def test_heave_natural_period_matches_stiffness_ratio(self, floe):
        lat, z_eq = floe
        coeffs = continuum_coefficients(lat)
        expected = 2 * math.pi * math.sqrt((coeffs.mass + coeffs.a33) / coeffs.k33)
        assert coeffs.to_dict()["heave_natural_period"] == pytest.approx(expected, rel=1e-12)

    def test_time_domain_matches_discrete_transfer_function(self, floe):
        """C1: the RK4 force path reproduces the linearised transfer function."""
        lat, z_eq = floe
        T = 8.0
        wave = RegularWave(height=0.02, period=T)
        fd = frequency_response(wave, lat, z_eq, reference="discrete")
        model = FloeResponseModel(lat, wave, z_eq=z_eq)
        trace = model.run(dt=T / 240, n_steps=int(240 * 12))
        fit = rao_from_time_series(trace.time, trace.heave - z_eq, wave.omega, skip=0.5)
        assert fit["amplitude"] / wave.amplitude == pytest.approx(fd.rao_heave, rel=0.02)
        assert fit["residual_fraction"] < 0.02

    def test_time_domain_matches_continuum_within_5pct(self, floe):
        lat, z_eq = floe
        T = 6.0
        wave = RegularWave(height=0.02, period=T)
        fd = frequency_response(wave, lat, z_eq, reference="continuum")
        model = FloeResponseModel(lat, wave, z_eq=z_eq)
        trace = model.run(dt=T / 240, n_steps=int(240 * 12))
        fit = rao_from_time_series(trace.time, trace.heave - z_eq, wave.omega, skip=0.5)
        assert fit["amplitude"] / wave.amplitude == pytest.approx(fd.rao_heave, rel=0.05)

    def test_energy_conserved_without_wave_and_damping(self, floe):
        """C5: released displacement, no wave, no damping -> energy conserved."""
        lat, z_eq = floe
        model = FloeResponseModel(lat, RegularWave(height=0.0, period=8.0),
                                  z_eq=z_eq, damping_ratio_heave=0.0,
                                  damping_ratio_pitch=0.0)
        trace = model.run(dt=8.0 / 240, n_steps=2400, z0=0.05)
        drift = trace.energy["delta_energy"] / trace.energy["energy_start"]
        assert abs(drift) < 1e-6

    def test_energy_ledger_closes_with_wave(self, floe):
        """C5: work of the explicit load = dKE + dU + dissipation."""
        lat, z_eq = floe
        model = FloeResponseModel(lat, RegularWave(height=0.1, period=8.0), z_eq=z_eq)
        # Same resolution as the campaign (C5): the residual is dominated by the
        # frozen added-mass operator plus the start-up transient, which shrinks
        # with dt, so a coarser march is not a fair test of the ledger.
        trace = model.run(dt=8.0 / 600, n_steps=3600)
        assert trace.energy["residual_fraction"] < 0.05

    def test_rao_fit_recovers_synthetic_sinusoid(self):
        t = np.arange(0.0, 40.0, 0.01)
        omega = 0.8
        y = 0.037 * np.cos(omega * t) - 0.021 * np.sin(omega * t) + 0.5
        fit = rao_from_time_series(t, y, omega, skip=0.0)
        assert fit["amplitude"] == pytest.approx(math.hypot(0.037, 0.021), rel=1e-9)
        assert fit["mean"] == pytest.approx(0.5, abs=1e-9)

    def test_refinement_changes_rao_by_less_than_5pct(self):
        """C6: refine the column lattice x2 (40x2 -> 80x4)."""
        coarse = FloeLattice(length=20.0, thickness=1.0, n_x=40, n_y=2)
        fine = FloeLattice(length=20.0, thickness=1.0, n_x=80, n_y=4)
        wave = RegularWave(height=0.4, period=8.0)
        c = frequency_response(wave, coarse, coarse.equilibrium_height(), reference="discrete")
        f = frequency_response(wave, fine, fine.equilibrium_height(), reference="discrete")
        assert abs(f.rao_heave - c.rao_heave) / c.rao_heave < 0.05


# ---------------------------------------------------------------------------
# Array transmission (C2)
# ---------------------------------------------------------------------------

class TestArrayTransmission:
    def test_empty_channel_transmits_everything(self):
        arr = FloeArrayTransmission(period=8.0, depth=1.0, floe_length=5.0,
                                    gap=5.0, n_floes=0, thickness=0.2)
        assert arr.transmission == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("period", [6.0, 8.0, 12.0, 20.0])
    def test_energy_closure_and_reciprocity(self, period):
        arr = FloeArrayTransmission(period=period, depth=1.0, floe_length=5.0,
                                    gap=5.0, n_floes=8, thickness=0.2)
        assert arr.energy_closure == pytest.approx(1.0, abs=1e-12)
        t_l, t_r = arr.coefficients()
        t_rev, _ = arr.coefficients(from_right=True)
        assert abs(abs(t_l) - abs(t_rev)) < 1e-12

    def test_transfer_matrix_matches_closed_form_slab(self):
        """Independent closed form for one homogeneous slab (two interfaces)."""
        for period in (4.0, 6.0, 8.0, 12.0):
            arr = FloeArrayTransmission(period=period, depth=1.0, floe_length=5.0,
                                        gap=0.0, n_floes=1, thickness=0.2)
            tmm = abs(arr.coefficients()[0])
            ana = slab_transmission_analytic(period, 1.0, 5.0, 0.2)["transmission"]
            assert tmm == pytest.approx(ana, rel=1e-10)

    def test_single_interface_energy_balance(self):
        out = single_interface_transmission(period=8.0, depth=1.0, thickness=0.2)
        assert out["energy_closure_flux"] == pytest.approx(1.0, abs=1e-12)
        assert out["impedance_floe"] < out["impedance_open"]

    def test_bragg_gap_is_present(self):
        """A spaced array attenuates near k0 (L_f + L_g) = pi far more than far off."""
        near = FloeArrayTransmission(period=7.0, depth=1.0, floe_length=5.0,
                                     gap=5.0, n_floes=12, thickness=0.2)
        far = FloeArrayTransmission(period=20.0, depth=1.0, floe_length=5.0,
                                    gap=5.0, n_floes=12, thickness=0.2)
        assert near.transmission < far.transmission
        assert near.reflection > far.reflection


# ---------------------------------------------------------------------------
# Wave-induced breakup (C3)
# ---------------------------------------------------------------------------

class TestBreakup:
    def test_fem_matches_plate_theory(self):
        E, h, nu, L = 9.0e9, 1.0, 0.3, 200.0
        k_c = characteristic_wavenumber(E, h, nu)
        beam = WinklerBeam(length=L, thickness=h, youngs_modulus=E, poisson=nu,
                           n_elements=80)
        fem = beam.solve(0.1, k_c)
        theory = theory_bending_stress(E, h, k_c, 0.1, nu)
        assert fem["sigma_max"] == pytest.approx(theory["sigma"], rel=0.05)
        assert fem["w_mid"] == pytest.approx(theory["w"], rel=0.02)

    def test_critical_wave_height_matches_theory(self):
        E, h, nu, sigma_f = 9.0e9, 1.0, 0.3, 0.5e6
        k_c = characteristic_wavenumber(E, h, nu)
        beam = WinklerBeam(length=200.0, thickness=h, youngs_modulus=E, poisson=nu,
                           n_elements=80)
        fem = beam.critical_wave_height(sigma_f, k_c)
        theory = critical_wave_height(sigma_f, E, h, k_c, nu)
        assert fem["critical_wave_height"] == pytest.approx(
            theory["critical_wave_height"], rel=0.05)
        # the tabulated engineering value for a 1 m floe is O(0.1 m)
        assert 0.05 < theory["critical_wave_height"] < 0.3

    def test_stress_peaks_at_characteristic_wavenumber(self):
        E, h, nu = 9.0e9, 1.0, 0.3
        k_c = characteristic_wavenumber(E, h, nu)
        at_kc = theory_bending_stress(E, h, k_c, 0.1, nu)["sigma"]
        for factor in (0.3, 0.6, 2.0, 4.0):
            assert theory_bending_stress(E, h, factor * k_c, 0.1, nu)["sigma"] < at_kc

    def test_beam_mesh_refinement(self):
        E, h, k_c = 9.0e9, 1.0, characteristic_wavenumber(9.0e9, 1.0, 0.3)
        s_coarse = WinklerBeam(length=200.0, n_elements=40).solve(0.1, k_c)["sigma_max"]
        s_fine = WinklerBeam(length=200.0, n_elements=120).solve(0.1, k_c)["sigma_max"]
        assert abs(s_fine - s_coarse) / s_coarse < 0.02


# ---------------------------------------------------------------------------
# Defect fix in the shipped Morison kernel
# ---------------------------------------------------------------------------

class TestVerticalWaveInertiaDefect:
    """The shipped kernel had no vertical inertia term (vertical load = drag only).

    Regression: a submerged particle in a wave must feel
    ``rho_w C_M A_p du_y/dt thickness`` upward/downward, which for a particle at
    the surface is the classic ``-rho_w C_M A_p w^2 eta``.
    """

    def _force(self, y, t, c_d=0.0):
        positions = torch_tensor([[0.0, y]])
        velocities = torch_tensor([[0.0, 0.0]])
        wet = WetCouplingConfig(enabled=True, water_density=1025.0, gravity=9.81)
        wave = WaveForceConfig(enabled=True, amplitude=0.1, wavelength=40.0,
                               period=6.0, water_depth=1000.0, inertia_coeff=2.0,
                               drag_coeff_morison=c_d)
        return compute_wave_force(positions, velocities, 0.25, 1.0, t, wave, wet)

    def test_vertical_inertia_is_present_and_analytic(self):
        radius, thickness, C_M, rho_w = 0.25, 1.0, 2.0, 1025.0
        wave = RegularWave(height=0.2, period=6.0)  # same wave as _force (H=0.2)
        for t in (0.0, 0.7, 1.9):
            f = self._force(0.0, t)
            kin = wave.kinematics(np.array([0.0]), np.array([0.0]), t)
            # a particle centred on the free surface is half submerged
            f_sub = 0.5
            expected = (rho_w * C_M * math.pi * radius ** 2
                        * float(kin["dw_dt"][0]) * thickness * f_sub)
            assert float(f[0, 1]) == pytest.approx(expected, rel=2e-3)

    def test_heave_load_is_not_drag_only(self):
        """Baseline comparison: without the fix the vertical load would be zero."""
        t = 0.0
        with_drag = float(self._force(0.0, t, c_d=1.2)[0, 1])
        no_drag = float(self._force(0.0, t, c_d=0.0)[0, 1])
        assert abs(no_drag) > 1e-3, "vertical inertia term missing"
        # the inertia part dominates the drag part at the crest
        assert abs(no_drag) > abs(with_drag - no_drag)

    def test_kinematics_full_matches_kinematics(self):
        x = torch_tensor([[0.0, 0.5, 1.0]])
        y = torch_tensor([[-0.5, 0.0, -1.0]])
        cfg = WaveForceConfig(enabled=True, amplitude=0.2, wavelength=30.0,
                             period=5.0, water_depth=1000.0)
        ux, uy, dux, duy = wave_kinematics_full(x, y, 0.3, cfg)
        ux0, uy0, dux0 = wave_kinematics(x, y, 0.3, cfg)
        assert np.allclose(ux.numpy(), ux0.numpy())
        assert np.allclose(uy.numpy(), uy0.numpy())
        assert np.allclose(dux.numpy(), dux0.numpy())
        # deep-water surface value: du_y/dt(z=0) = -w^2 eta
        omega = 2.0 * math.pi / 5.0
        i = 1
        assert float(duy.numpy().ravel()[i]) == pytest.approx(
            -omega ** 2 * cfg.amplitude
            * math.cos(2 * math.pi / 30.0 * 0.5 - omega * 0.3),
            rel=1e-6)


def torch_tensor(data):
    import torch
    return torch.tensor(data, dtype=torch.float64)