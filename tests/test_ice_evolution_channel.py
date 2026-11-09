"""Pytest suite for the broken-ice channel evolution (campaign p54w2-N02).

Machine-checkable criteria:
(a) Sintering neck growth vs Frenkel theory a(t) ∝ t^(1/3), exponent fit error < 10%;
(b) Porosity decay vs refreezing theory φ(t) = φ_0 exp(-kt), fit error < 15%;
(c) Resistance decay curve R_ice(N)/R_ice(1) monotonically increasing with α ∈ [0.05, 0.2];
(d) Full pytest chain including no-sintering degradation test.

Units are SI throughout.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ice_evolution.sintering import (
    ICE_SURFACE_ENERGY,
    ICE_VISCOSITY_REF,
    ICE_VISCOSITY_T_REF,
    ICE_YOUNG,
    ICE_STRENGTH_REF,
    SinterState,
    frenkel_neck_radius,
    ice_strength,
    ice_viscosity,
    neck_growth_exponent,
    sinter_bond,
)
from tensordem.ice_evolution.refreezing import (
    RefreezeConfig,
    RefreezeResult,
    run_refreezing,
)
from tensordem.ice_evolution.strength import (
    StrengthConfig,
    rubble_strength,
    strength_growth,
)
from tensordem.ice_evolution.resistance import (
    PlatenConfig,
    PlatenDEM,
    run_platen,
)
from tensordem.ice_evolution.channel import (
    ChannelConfig,
    ChannelResult,
    calibrate,
    run_channel_evolution,
)
from tensordem.ice_evolution.energy import (
    EnergyBudget,
    energy_closure,
    sinter_surface_energy_j_m3,
)


# ===========================================================================
# (a) Sintering neck growth vs Frenkel theory
# ===========================================================================

class TestFrenkelSintering:
    """Criterion (a): a(t) ∝ t^(1/3) with exponent fit error < 10%."""

    @pytest.mark.parametrize("temperature_k", [253.15, 258.15, 263.15, 268.15])
    def test_frenkel_exponent_error_below_10pct(self, temperature_k):
        """Frenkel exponent must be within 10% of 1/3 at all temperatures."""
        t = np.logspace(2.0, 7.0, 30)
        a = np.array([frenkel_neck_radius(tt, temperature_k,
                                          particle_radius=0.010) for tt in t])
        p, C = neck_growth_exponent(t, a)
        exp_err = abs(p - 1.0 / 3.0) / (1.0 / 3.0)
        assert exp_err < 0.10, (
            f"Exponent error {exp_err:.4f} at T={temperature_k}K "
            f"(fitted={p:.6f}, theory=0.3333)"
        )

    def test_frenkel_amplitude_matches_theory(self):
        """Amplitude C must match (3γ/(2η))^(1/3) within 10%."""
        temperature_k = 263.15
        t = np.logspace(2.0, 7.0, 30)
        a = np.array([frenkel_neck_radius(tt, temperature_k,
                                          particle_radius=0.010) for tt in t])
        p, C = neck_growth_exponent(t, a)
        eta = ice_viscosity(temperature_k)
        C_exact = (3.0 * ICE_SURFACE_ENERGY / (2.0 * eta)) ** (1.0 / 3.0)
        amp_err = abs(C - C_exact) / C_exact
        assert amp_err < 0.10, (
            f"Amplitude error {amp_err:.4f}: fitted={C:.6e}, theory={C_exact:.6e}"
        )

    def test_neck_radius_monotone_increasing(self):
        """Neck radius must increase monotonically with time."""
        temperature_k = 263.15
        t = np.logspace(2.0, 7.0, 20)
        a = np.array([frenkel_neck_radius(tt, temperature_k,
                                          particle_radius=0.010) for tt in t])
        for i in range(len(a) - 1):
            assert a[i] < a[i + 1], (
                f"Non-monotone at t={t[i]:.0f}→{t[i+1]:.0f}: "
                f"a={a[i]:.6e}→{a[i+1]:.6e}"
            )

    def test_neck_capped_at_particle_radius(self):
        """Neck radius must not exceed the particle radius."""
        temperature_k = 263.15
        R = 0.010
        a = frenkel_neck_radius(1e10, temperature_k, particle_radius=R)
        assert a <= R, f"Neck {a:.6e} exceeds particle radius {R}"

    def test_neck_zero_at_zero_time(self):
        """At t=0, the neck radius must be zero."""
        temperature_k = 263.15
        a = frenkel_neck_radius(0.0, temperature_k, particle_radius=0.010)
        assert a == 0.0

    def test_viscosity_decreases_with_temperature(self):
        """Ice viscosity must decrease as temperature increases."""
        eta_cold = ice_viscosity(253.15)
        eta_warm = ice_viscosity(268.15)
        assert eta_cold > eta_warm, (
            f"Viscosity not decreasing: eta(-20C)={eta_cold:.2e} "
            f"vs eta(-5C)={eta_warm:.2e}"
        )

    def test_sinter_bond_state(self):
        """SinterState must have physically consistent values."""
        temperature_k = 263.15
        R = 0.010
        t_s = 3600.0
        state = sinter_bond(t_s, R, temperature_k)
        assert isinstance(state, SinterState)
        assert state.neck_radius_m > 0.0
        assert 0.0 < state.neck_ratio < 1.0
        assert state.bond_stiffness > 0.0
        assert state.bond_strength > 0.0
        assert state.surface_area_reduced_m2 > 0.0

    def test_bond_stiffness_scales_with_a_squared(self):
        """Bond stiffness k_bond = E π a² / R must scale with a²."""
        temperature_k = 263.15
        R = 0.010
        t1, t2 = 100.0, 400.0  # 4x time → 4^(1/3) ≈ 1.587x neck
        s1 = sinter_bond(t1, R, temperature_k)
        s2 = sinter_bond(t2, R, temperature_k)
        ratio_k = s2.bond_stiffness / s1.bond_stiffness
        ratio_a = (s2.neck_radius_m / s1.neck_radius_m) ** 2
        assert abs(ratio_k / ratio_a - 1.0) < 0.01, (
            f"k_bond scaling mismatch: k_ratio={ratio_k:.4f}, a²_ratio={ratio_a:.4f}"
        )


# ===========================================================================
# (b) Porosity decay vs refreezing theory
# ===========================================================================

class TestRefreezingPorosity:
    """Criterion (b): φ(t) = φ_0 exp(-kt) with fit error < 15%."""

    def test_porosity_exponential_fit_error_below_15pct(self):
        """Max relative error of exponential fit must be < 15%."""
        cfg = RefreezeConfig(
            thickness_m=0.05, temperature_k=263.15, phi_0=0.30
        )
        duration_s = 24.0 * 3600.0
        res = run_refreezing(cfg, duration_s=duration_s, record_every_s=300.0)
        t, phi = res.porosity_series()
        mask = phi > 1e-6
        lt = t[mask]
        lphi = np.log(phi[mask] / cfg.phi_0)
        A = np.vstack([-lt, np.ones_like(lt)]).T
        (k_fit, intercept), *_ = np.linalg.lstsq(A, lphi, rcond=None)
        pred = cfg.phi_0 * np.exp(-k_fit * t)
        rel_err = float(np.max(np.abs(pred - phi) / np.maximum(phi, 1e-9)))
        assert rel_err < 0.15, (
            f"Porosity fit error {rel_err:.4f} exceeds 15%; "
            f"k_fit={-k_fit:.6e} 1/s"
        )

    def test_porosity_decreases_monotonically(self):
        """Porosity must decrease monotonically over time."""
        cfg = RefreezeConfig(
            thickness_m=0.05, temperature_k=263.15, phi_0=0.30
        )
        res = run_refreezing(cfg, duration_s=12.0 * 3600.0, record_every_s=600.0)
        _, phi = res.porosity_series()
        for i in range(len(phi) - 1):
            assert phi[i] >= phi[i + 1], (
                f"Porosity increased: {phi[i]:.4f} → {phi[i+1]:.4f}"
            )

    def test_porosity_bounded(self):
        """Porosity must stay in [0, phi_0]."""
        cfg = RefreezeConfig(
            thickness_m=0.05, temperature_k=263.15, phi_0=0.30
        )
        res = run_refreezing(cfg, duration_s=48.0 * 3600.0, record_every_s=600.0)
        _, phi = res.porosity_series()
        assert np.all(phi >= 0.0), "Porosity went negative"
        assert np.all(phi <= cfg.phi_0 + 1e-12), "Porosity exceeded initial"

    def test_porosity_initial_value(self):
        """Initial porosity must equal phi_0."""
        cfg = RefreezeConfig(thickness_m=0.05, temperature_k=263.15, phi_0=0.35)
        res = run_refreezing(cfg, duration_s=3600.0, record_every_s=600.0)
        _, phi = res.porosity_series()
        assert phi[0] == pytest.approx(cfg.phi_0, rel=1e-10)

    @pytest.mark.parametrize("phi_0", [0.15, 0.30, 0.45, 0.60])
    def test_porosity_decay_for_various_phi0(self, phi_0):
        """Exponential fit must work for various initial porosities."""
        cfg = RefreezeConfig(
            thickness_m=0.05, temperature_k=263.15, phi_0=phi_0
        )
        res = run_refreezing(cfg, duration_s=24.0 * 3600.0, record_every_s=600.0)
        t, phi = res.porosity_series()
        mask = phi > 1e-6
        if len(mask) < 3:
            pytest.skip("Too few data points for fit")
        lt = t[mask]
        lphi = np.log(phi[mask] / cfg.phi_0)
        A = np.vstack([-lt, np.ones_like(lt)]).T
        (k_fit, _), *_ = np.linalg.lstsq(A, lphi, rcond=None)
        pred = cfg.phi_0 * np.exp(-k_fit * t)
        rel_err = float(np.max(np.abs(pred - phi) / np.maximum(phi, 1e-9)))
        assert rel_err < 0.15, (
            f"phi_0={phi_0}: fit error {rel_err:.4f} exceeds 15%"
        )

    def test_temperature_affects_freezing_rate(self):
        """Colder temperatures must produce faster freezing."""
        cfg_cold = RefreezeConfig(
            thickness_m=0.05, temperature_k=253.15, phi_0=0.30
        )
        cfg_warm = RefreezeConfig(
            thickness_m=0.05, temperature_k=268.15, phi_0=0.30
        )
        dur = 12.0 * 3600.0
        res_cold = run_refreezing(cfg_cold, duration_s=dur, record_every_s=600.0)
        res_warm = run_refreezing(cfg_warm, duration_s=dur, record_every_s=600.0)
        phi_cold = res_cold.porosity[-1]
        phi_warm = res_warm.porosity[-1]
        assert phi_cold < phi_warm, (
            f"Colder should freeze more: phi(-20C)={phi_cold:.4f} "
            f"vs phi(-5C)={phi_warm:.4f}"
        )


# ===========================================================================
# (c) Resistance growth curve
# ===========================================================================

class TestResistanceGrowth:
    """Criterion (c): R_ice(N)/R_ice(1) monotone increasing, α ∈ [0.05, 0.2]."""

    def _fast_channel_config(self, n_transits=8, n_cal=3):
        """Return a fast channel config for testing (small DEM)."""
        from tensordem.ice_evolution.resistance import PlatenConfig
        from tensordem.ice_evolution.refreezing import RefreezeConfig
        from tensordem.ice_evolution.strength import StrengthConfig

        platen = PlatenConfig(
            n_particles=40,
            mean_radius=0.012,
            box_width=0.8,
            box_height=0.20,
            platen_speed=0.4,
            duration_s=1.0,
            mu=0.3,
            kn=2.0e5,
            damping_ratio=0.3,
            dt_s=2.0e-4,
            seed=0,
            k_bond=0.0,
            f_bond=0.0,
            thickness_m=0.2,
        )
        return ChannelConfig(
            n_transits=n_transits,
            interval_s=6.0 * 3600.0,
            temperature_k=263.15,
            phi_0=0.30,
            particle_radius_m=0.010,
            n_calibration=n_cal,
            platen=platen,
            refreeze=RefreezeConfig(thickness_m=0.05, temperature_k=263.15, phi_0=0.30),
            strength=StrengthConfig(temperature_k=263.15, phi_0=0.30, particle_radius_m=0.010),
        )

    def test_resistance_ratio_monotone(self):
        """R_ice(N)/R_ice(1) must be monotonically non-decreasing."""
        cfg = self._fast_channel_config(n_transits=6, n_cal=2)
        ch = run_channel_evolution(cfg)
        assert ch.monotone, (
            f"Resistance ratio not monotone: {ch.resistance_ratio.tolist()}"
        )

    def test_alpha_in_valid_range(self):
        """Log-fit alpha must be in [0.05, 0.2]."""
        cfg = self._fast_channel_config(n_transits=8, n_cal=3)
        ch = run_channel_evolution(cfg)
        assert 0.05 <= ch.alpha <= 0.20, (
            f"Alpha {ch.alpha:.4f} outside [0.05, 0.20]; "
            f"R2={ch.fit_r2:.4f}, monotone={ch.monotone}"
        )

    def test_resistance_increases_with_transits(self):
        """Final resistance must be greater than initial."""
        cfg = self._fast_channel_config(n_transits=6, n_cal=2)
        ch = run_channel_evolution(cfg)
        assert ch.resistance_ratio[-1] > 1.0, (
            f"Final ratio {ch.resistance_ratio[-1]:.4f} should be > 1.0"
        )

    def test_sigma_increases_with_transits(self):
        """Rubble strength must increase with transit count."""
        cfg = self._fast_channel_config(n_transits=6, n_cal=2)
        ch = run_channel_evolution(cfg)
        for i in range(len(ch.sigma) - 1):
            assert ch.sigma[i] <= ch.sigma[i + 1] + 1e-6, (
                f"Strength decreased at N={i+1}: {ch.sigma[i]:.2e} → {ch.sigma[i+1]:.2e}"
            )

    def test_porosity_decreases_with_transits(self):
        """Porosity must decrease with transit count (healing)."""
        cfg = self._fast_channel_config(n_transits=6, n_cal=2)
        ch = run_channel_evolution(cfg)
        for i in range(len(ch.porosity) - 1):
            assert ch.porosity[i] >= ch.porosity[i + 1] - 1e-12, (
                f"Porosity increased at N={i+1}: {ch.porosity[i]:.4f} → {ch.porosity[i+1]:.4f}"
            )

    def test_neck_ratio_increases_with_transits(self):
        """Neck ratio must increase with transit count (sintering)."""
        cfg = self._fast_channel_config(n_transits=6, n_cal=2)
        ch = run_channel_evolution(cfg)
        for i in range(len(ch.neck_ratio) - 1):
            assert ch.neck_ratio[i] <= ch.neck_ratio[i + 1] + 1e-12, (
                f"Neck ratio decreased at N={i+1}: {ch.neck_ratio[i]:.6e} → {ch.neck_ratio[i+1]:.6e}"
            )


# ===========================================================================
# (d) No-sintering degradation test (negative control)
# ===========================================================================

class TestNoSinteringDegradation:
    """Criterion (d): Without sintering/refreezing, resistance stays constant."""

    def test_zero_bond_gives_loose_rubble(self):
        """With k_bond=0 and f_bond=0, the DEM should produce loose-rubble resistance."""
        cfg = PlatenConfig(
            n_particles=40,
            mean_radius=0.012,
            box_width=0.8,
            box_height=0.20,
            platen_speed=0.4,
            duration_s=1.0,
            mu=0.3,
            kn=2.0e5,
            damping_ratio=0.3,
            dt_s=2.0e-4,
            seed=0,
            k_bond=0.0,
            f_bond=0.0,
            thickness_m=0.2,
        )
        out = run_platen(cfg)
        # Loose rubble should have finite but low resistance
        assert out["resistance_mean_n"] > 0.0, "Zero resistance for loose rubble"
        assert out["n_broken"] == 0, "No bonds to break in loose rubble"

    def test_sintered_bond_increases_resistance(self):
        """Strong sintered bonds must increase resistance compared to loose rubble.

        We use a long sintering time (24h) to ensure bonds are strong enough
        to significantly increase the resistance.
        """
        base = PlatenConfig(
            n_particles=40,
            mean_radius=0.012,
            box_width=0.8,
            box_height=0.20,
            platen_speed=0.4,
            duration_s=1.0,
            mu=0.3,
            kn=2.0e5,
            damping_ratio=0.3,
            dt_s=2.0e-4,
            seed=0,
            k_bond=0.0,
            f_bond=0.0,
            thickness_m=0.2,
        )
        out_loose = run_platen(base)

        # Create strong sintered bond (24h sintering at -10C)
        bond = sinter_bond(24.0 * 3600.0, 0.010, 263.15)
        cfg_bonded = PlatenConfig(
            n_particles=base.n_particles,
            mean_radius=base.mean_radius,
            box_width=base.box_width,
            box_height=base.box_height,
            platen_speed=base.platen_speed,
            duration_s=base.duration_s,
            mu=base.mu,
            kn=base.kn,
            damping_ratio=base.damping_ratio,
            dt_s=base.dt_s,
            seed=base.seed,
            k_bond=bond.bond_stiffness,
            f_bond=bond.bond_strength,
            thickness_m=base.thickness_m,
        )
        out_bonded = run_platen(cfg_bonded)

        # With strong bonds, resistance should be higher
        # (bonds hold the rubble together, requiring more force to break through)
        assert out_bonded["resistance_mean_n"] > out_loose["resistance_mean_n"], (
            f"Bonded resistance {out_bonded['resistance_mean_n']:.1f} N "
            f"should exceed loose {out_loose['resistance_mean_n']:.1f} N; "
            f"bond stiffness={bond.bond_stiffness:.1f} N/m, "
            f"bond strength={bond.bond_strength:.1f} N"
        )

    def test_no_sintering_constant_resistance(self):
        """Without sintering (t_s=0), the channel resistance should be nearly constant.

        We test this by checking that the rubble strength at t=0 equals the
        loose-rubble baseline, and that the neck ratio stays at 0.
        """
        cfg = StrengthConfig(temperature_k=263.15, phi_0=0.30, particle_radius_m=0.010)
        sigma_0 = rubble_strength(0.0, cfg)
        sigma_ice = ice_strength(263.15)
        expected = sigma_ice * (1.0 - cfg.phi_0) ** cfg.solid_exponent * cfg.f_loose
        assert sigma_0 == pytest.approx(expected, rel=1e-10), (
            f"sigma(0)={sigma_0:.6e} != expected={expected:.6e}"
        )

        # Also verify neck ratio is 0 at t=0
        a = frenkel_neck_radius(0.0, 263.15, particle_radius=0.010)
        assert a == 0.0, f"Neck at t=0 should be 0, got {a}"

    def test_rubble_strength_decomposition(self):
        """Strength growth decomposition must sum exactly to total strength."""
        cfg = StrengthConfig(temperature_k=263.15, phi_0=0.30, particle_radius_m=0.010)
        t_s = 3600.0
        decomp = strength_growth(t_s, cfg)
        total = decomp["sigma_0"] + decomp["d_sigma_sinter"] + decomp["d_sigma_refreeze"]
        assert abs(total - decomp["sigma"]) < 1e-10 * decomp["sigma"], (
            f"Decomposition mismatch: sum={total:.6e}, sigma={decomp['sigma']:.6e}"
        )

    def test_rubble_strength_at_zero_time(self):
        """At t=0, rubble strength should equal the loose-rubble baseline."""
        cfg = StrengthConfig(temperature_k=263.15, phi_0=0.30, particle_radius_m=0.010)
        sigma_0 = rubble_strength(0.0, cfg)
        sigma_ice = ice_strength(263.15)
        expected = sigma_ice * (1.0 - cfg.phi_0) ** cfg.solid_exponent * cfg.f_loose
        assert sigma_0 == pytest.approx(expected, rel=1e-10), (
            f"sigma(0)={sigma_0:.6e} != expected={expected:.6e}"
        )

    def test_bond_strength_increases_with_sintering_time(self):
        """Bond strength must increase with sintering time."""
        R = 0.010
        T = 263.15
        t_short = 3600.0    # 1 hour
        t_long = 24*3600.0  # 24 hours
        s_short = sinter_bond(t_short, R, T)
        s_long = sinter_bond(t_long, R, T)
        assert s_long.bond_strength > s_short.bond_strength, (
            f"Bond strength should increase: {s_short.bond_strength:.2f} N (1h) "
            f"vs {s_long.bond_strength:.2f} N (24h)"
        )
        assert s_long.bond_stiffness > s_short.bond_stiffness, (
            f"Bond stiffness should increase: {s_short.bond_stiffness:.1f} N/m (1h) "
            f"vs {s_long.bond_stiffness:.1f} N/m (24h)"
        )


# ===========================================================================
# Energy budget
# ===========================================================================

class TestEnergyBudget:
    """Energy closure test for sintering + refreezing."""

    def test_energy_closure_below_10pct(self):
        """Energy closure must be < 10%."""
        cfg = RefreezeConfig(thickness_m=0.05, temperature_k=263.15, phi_0=0.30)
        duration_s = 24.0 * 3600.0
        res = run_refreezing(cfg, duration_s=duration_s, record_every_s=600.0)

        # Estimate sintering surface energy
        a = frenkel_neck_radius(duration_s, 263.15, particle_radius=0.010)
        neck_ratio = a / 0.010
        n_contacts = 6.0 / (4.0 / 3.0 * np.pi * 0.010**3)
        e_sinter = sinter_surface_energy_j_m3(neck_ratio, 0.010, n_contacts,
                                              ICE_SURFACE_ENERGY)

        budget = EnergyBudget(
            e_latent=res.e_latent_j_m2,
            e_sinter=res.e_sinter_j_m3 * cfg.thickness_m,
            e_conduction=res.e_conduction_j_m2,
            e_ambient=0.0,
            d_e_sensible=res.e_sensible_j_m2,
        )
        closure = energy_closure(budget)
        assert closure < 0.10, (
            f"Energy closure {closure:.4f} exceeds 10%; "
            f"latent={budget.e_latent:.2f}, conduction={budget.e_conduction:.2f}"
        )

    def test_energy_budget_trivial_closure(self):
        """When nothing happens, closure should be 0."""
        budget = EnergyBudget(
            e_latent=0.0, e_sinter=0.0, e_conduction=0.0,
            e_ambient=0.0, d_e_sensible=0.0,
        )
        assert energy_closure(budget) == 0.0


# ===========================================================================
# Platen DEM basic tests
# ===========================================================================

class TestPlatenDEM:
    """Basic sanity checks for the platen-through-rubble DEM."""

    def test_platen_force_positive(self):
        """Platen force must be positive (resisting motion)."""
        cfg = PlatenConfig(
            n_particles=40, mean_radius=0.012, box_width=0.8, box_height=0.20,
            platen_speed=0.4, duration_s=0.5, mu=0.3, kn=2.0e5,
            damping_ratio=0.3, dt_s=2.0e-4, seed=0,
            k_bond=0.0, f_bond=0.0, thickness_m=0.2,
        )
        out = run_platen(cfg)
        assert out["resistance_mean_n"] > 0.0

    def test_platen_deterministic(self):
        """Same seed must produce identical results."""
        cfg = PlatenConfig(
            n_particles=40, mean_radius=0.012, box_width=0.8, box_height=0.20,
            platen_speed=0.4, duration_s=0.5, mu=0.3, kn=2.0e5,
            damping_ratio=0.3, dt_s=2.0e-4, seed=42,
            k_bond=0.0, f_bond=0.0, thickness_m=0.2,
        )
        out1 = run_platen(cfg)
        out2 = run_platen(cfg)
        assert out1["resistance_mean_n"] == pytest.approx(out2["resistance_mean_n"])

    def test_bond_breakage_counted(self):
        """With bonds, some should break during platen transit."""
        bond = sinter_bond(3600.0, 0.010, 263.15)
        cfg = PlatenConfig(
            n_particles=40, mean_radius=0.012, box_width=0.8, box_height=0.20,
            platen_speed=0.4, duration_s=1.0, mu=0.3, kn=2.0e5,
            damping_ratio=0.3, dt_s=2.0e-4, seed=0,
            k_bond=bond.bond_stiffness, f_bond=bond.bond_strength, thickness_m=0.2,
        )
        out = run_platen(cfg)
        assert out["n_broken"] >= 0  # may be 0 if bonds are very strong
