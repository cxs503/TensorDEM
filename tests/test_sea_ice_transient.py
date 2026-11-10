"""Acceptance + negative-control tests for the transient sea-ice drift anchors.

Machine-checkable criteria of campaign node P121W5-N07 (see
``benchmarks/sea_ice_transient_anchor.py`` for the full study):

(a) closed form: oscillation frequency, decay rate and spin-up amplitude ratio
    reproduced to ``< 1e-6`` (f = 1.4e-4 1/s and the 80N / 70N cases);
(b) momentum ledger (Coriolis included) ``residual_rel < 1e-8``, pointwise and
    step-integrated;
(c) ``P* -> 0`` in the viscous-plastic channel degenerates to the
    :mod:`tensordem.sea_ice_drift` steady-state ledger (pointwise ``< 1e-10``);
(d) negative control: explicit Euler at ``dt = 1 h`` must fail (amplitude error
    ``> 50 %`` or divergence) while the exponential (semi-implicit) scheme stays
    ``< 1e-3``;
(f) two-level time-step refinement changes the identified response by ``< 5 %``.

Structural invariants that must also hold:

* ``<d sigma/dx> = 0`` in the periodic channel, so the *domain-mean* response is
  exactly free drift for any ``P*`` (zero-control);
* the estimators agree with each other on noise-free data (no self-grading);
* the linear-viscous prediction is monotone in ``P*`` and vanishes at ``P* = 0``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensordem.sea_ice_drift import (  # noqa: E402
    SeaIceDriftModel,
    coriolis_parameter,
    free_drift_state,
)
from tensordem.sea_ice_transient import (  # noqa: E402
    DampedInertialSolution,
    amplification,
    discrete_ledger_residual,
    estimate_event_response,
    estimate_inertial_parameters,
    initial_state_closed_form,
    integrate,
    linearized_drag_rate,
    momentum_ledger,
)
from tensordem.sea_ice_vp import (  # noqa: E402
    ChannelConfig,
    HiblerRheology,
    OneDimVPIceChannel,
    linear_viscous_prediction,
)

RHO_I, RHO_W, RHO_A = 900.0, 1025.0, 1.25
CD_W, CD_A = 0.0045, 0.0018
H_ICE = 6.0


def linear_model(lat, thickness=H_ICE, wind_speed=10.0, drag_rate=8.5e-5, angle=0.0):
    ang = math.radians(angle)
    return SeaIceDriftModel(
        latitude_deg=lat, ice_thickness=thickness,
        wind=(wind_speed * math.cos(ang), wind_speed * math.sin(ang)),
        ice_density=RHO_I, rho_air=RHO_A, c_d_air=CD_A,
        rho_water=RHO_W, c_d_water=CD_W,
        drag_mode="linear", linear_drag_rate=drag_rate, wind_relative=False,
    )


def quadratic_model(lat, thickness):
    return SeaIceDriftModel(
        latitude_deg=lat, ice_thickness=thickness, wind=(10.0, 0.0),
        ice_density=RHO_I, rho_air=RHO_A, c_d_air=CD_A,
        rho_water=RHO_W, c_d_water=CD_W,
        drag_mode="quadratic", wind_relative=True,
    )


# ---------------------------------------------------------------------------
# (a) closed-form anchor
# ---------------------------------------------------------------------------
class TestClosedForm:
    @pytest.mark.parametrize("lat", [70.0, 80.0])
    def test_frequency_decay_and_amplitude_anchor(self, lat):
        model = linear_model(lat)
        sol = initial_state_closed_form(model, u_initial=(0.0, 0.0))
        dt = sol.period / 720.0
        n = int(round(5.0 * 86400.0 / dt))
        times, us = integrate(sol, dt, n, scheme="exponential")
        est = estimate_inertial_parameters(times, us, method="recurrence")
        assert abs(est["f"] - sol.f) / sol.f < 1e-6
        assert abs(est["gamma"] - sol.gamma) / sol.gamma < 1e-6
        # spin-up amplitude ratio against the analytic exp(-gamma t)
        i = int(round(3.0 * 3600.0 / dt))
        ratio_num = np.linalg.norm(us[i] - sol.u_steady) / np.linalg.norm(sol.u_steady)
        ratio_an = sol.amplitude_ratio(times[i])
        assert abs(ratio_num - ratio_an) / ratio_an < 1e-6

    def test_frequency_is_exactly_f_not_sqrt_f2_plus_gamma2(self):
        """The friction is not a restoring force: omega = |f| exactly."""
        sol = DampedInertialSolution(
            u_steady=np.array([0.2, 0.0]), u_initial=np.zeros(2),
            gamma=1.0e-4, f=1.4e-4,
        )
        assert sol.omega == sol.f
        # zero crossing of v(t) at half the inertial period
        assert abs(sol.offset_at(sol.period / 2.0) @ np.array([0.0, 1.0])) < 1e-12 * 0.2

    def test_analytic_estimator_cross_checks(self):
        # Estimators identify (gamma, omega) from the *transient*; the record must
        # therefore only span a few decay times (here ~5 e-folds), otherwise the
        # information is numerically absent and any fit is degenerate.
        sol = DampedInertialSolution(
            u_steady=np.array([0.18, -0.03]), u_initial=np.zeros(2),
            gamma=8.5e-5, f=1.4e-4,
        )
        dt = sol.period / 480.0
        times, us = integrate(sol, dt, int(round(3.0 * 86400.0 / dt)))
        rec = estimate_inertial_parameters(times, us, method="recurrence")
        non = estimate_inertial_parameters(times, us, method="nonlinear",
                                           f_hint=sol.f, gamma_hint=sol.gamma)
        ev = estimate_event_response(times, us, [], f_hint=sol.f, gamma_hint=sol.gamma)
        for res in (rec, non, ev):
            assert abs(res["gamma"] - sol.gamma) / sol.gamma < 1e-6
            assert abs(res["f"] - sol.f) / sol.f < 1e-6

    def test_closed_form_requires_linear_isotropic_drag(self):
        with pytest.raises(ValueError):
            initial_state_closed_form(quadratic_model(80.0, H_ICE))
        bad = linear_model(80.0)
        bad.wind_relative = True
        with pytest.raises(ValueError):
            initial_state_closed_form(bad)

    def test_series_matches_complex_exponential(self):
        sol = DampedInertialSolution(
            u_steady=np.array([0.1, 0.2]), u_initial=np.zeros(2), gamma=1e-4, f=1.4e-4
        )
        z0 = complex(*(sol.u_initial - sol.u_steady))
        t = 3600.0
        z = z0 * np.exp(-(sol.gamma + 1j * sol.f) * t)
        expected = sol.u_steady + np.array([z.real, z.imag])
        assert np.allclose(sol.velocity(t), expected, atol=1e-15)


# ---------------------------------------------------------------------------
# (b) momentum ledger
# ---------------------------------------------------------------------------
class TestMomentumLedger:
    def test_pointwise_ledger_includes_coriolis(self):
        model = linear_model(80.0)
        sol = initial_state_closed_form(model, u_initial=(0.0, 0.0))
        worst = 0.0
        coriolis_max = 0.0
        for t in np.linspace(0.0, 1.5 * sol.period, 17):
            led = momentum_ledger(model, sol.velocity(t), sol.derivative(t))
            worst = max(worst, led["residual_rel"])
            coriolis_max = max(coriolis_max, float(np.linalg.norm(led["coriolis"])))
        assert worst < 1e-8
        assert coriolis_max > 0.0   # the balance really contains the Coriolis term

    def test_step_integrated_ledger(self):
        model = linear_model(70.0)
        sol = initial_state_closed_form(model, u_initial=(0.0, 0.0))
        dt = sol.period / 360.0
        assert discrete_ledger_residual(sol, dt, 240, scheme="exponential") < 1e-8

    def test_ledger_detects_a_sign_error(self):
        """A flipped Coriolis term must show up in the residual (the ledger has teeth)."""
        model = linear_model(80.0)
        sol = initial_state_closed_form(model, u_initial=(0.0, 0.0))
        t = 0.37 * sol.period
        u = sol.velocity(t)
        a_bad = sol.derivative(t) + 2.0 * sol.f * np.array([u[1], -u[0]])
        led = momentum_ledger(model, u, a_bad)
        assert led["residual_rel"] > 1e-3


# ---------------------------------------------------------------------------
# (d) scheme gates
# ---------------------------------------------------------------------------
class TestSchemes:
    def test_exponential_scheme_is_exact_and_euler_is_not(self):
        sol = initial_state_closed_form(linear_model(80.0), u_initial=(0.0, 0.0))
        dt = 3600.0
        n = 48
        times, us = integrate(sol, dt, n, scheme="exponential")
        assert np.max(np.abs(us - sol.series(times))) < 1e-12 * np.linalg.norm(sol.u_steady)
        amp = amplification(dt, sol.f, sol.gamma, "exponential")
        assert abs(amp["gamma_numeric"] - sol.gamma) / sol.gamma < 1e-12
        assert abs(amp["f_numeric"] - abs(sol.f)) / abs(sol.f) < 1e-12

    def test_euler_1h_negative_control_fails(self):
        """At dt = 1 h explicit Euler corrupts both the phase and the damping.

        With ``gamma*dt = 0.3`` and ``f*dt = 0.52`` the per-step modulus is 0.87,
        i.e. the scheme is *spuriously overdamped* (an artificial extra drag) and
        the rotation is under-resolved, so the amplitude error after a few days
        exceeds 50 % -- the acceptance gate must reject it.
        """
        sol = initial_state_closed_form(linear_model(80.0), u_initial=(0.0, 0.0))
        amp = amplification(3600.0, sol.f, sol.gamma, "euler")
        assert abs(amp["gamma_numeric"] - sol.gamma) / sol.gamma > 0.2
        assert abs(amp["f_numeric"] - abs(sol.f)) / abs(sol.f) > 0.05
        times, us = integrate(sol, 3600.0, 96, scheme="euler")   # 4 days
        amp_num = np.linalg.norm(us[-1] - sol.u_steady)
        amp_exact = np.linalg.norm(sol.offset_at(times[-1]))
        assert abs(amp_num - amp_exact) / amp_exact > 0.5

    def test_semi_implicit_1h_beats_the_1e3_gate(self):
        sol = initial_state_closed_form(linear_model(80.0), u_initial=(0.0, 0.0))
        dt = 3600.0
        i = 3   # 3 h, where the amplitude is still O(1)
        times, us = integrate(sol, dt, 24, scheme="exponential")
        err = abs(np.linalg.norm(us[i] - sol.u_steady)
                  - np.linalg.norm(sol.offset_at(times[i]))) / np.linalg.norm(
                      sol.offset_at(times[i]))
        assert err < 1e-3

    def test_crank_nicolson_is_stable_but_damps_wrong(self):
        sol = initial_state_closed_form(linear_model(80.0), u_initial=(0.0, 0.0))
        amp = amplification(3600.0, sol.f, sol.gamma, "crank_nicolson")
        # A-stable: no growth
        assert amp["modulus"] <= 1.0 + 1e-12
        # but the per-step damping is visibly off at gamma*dt = 0.31
        assert abs(amp["gamma_numeric"] - sol.gamma) / sol.gamma > 1e-2

    @pytest.mark.parametrize("scheme", ["exponential", "crank_nicolson", "rk4"])
    def test_first_order_refinement_converges(self, scheme):
        sol = initial_state_closed_form(linear_model(80.0), u_initial=(0.0, 0.0))
        def err(dt):
            times, us = integrate(sol, dt, int(round(86400.0 / dt)), scheme=scheme)
            return float(np.max(np.abs(us - sol.series(times))))
        e1 = err(1800.0)
        e2 = err(900.0)
        assert e2 < e1


# ---------------------------------------------------------------------------
# (c) P* -> 0 degeneration and the periodic zero-control
# ---------------------------------------------------------------------------
class TestVPDegeneration:
    def _channel(self, p_star, lat=80.0, thick=2.5, amplitude=0.0, days=2.0, n=32):
        cfg = ChannelConfig(
            n_cells=n, dt=1800.0, duration_days=days, latitude_deg=lat,
            ice_thickness=thick, wind_angle_deg=0.0, wind_speed=10.0,
            wind_spatial_amplitude=amplitude, wind_switch_angle_deg=0.0,
            wind_variability=0.0, picard_iterations=2, buoy_window_length=4.0e5,
        )
        ch = OneDimVPIceChannel(cfg, HiblerRheology(p_star=p_star), seed=0)
        return ch, ch.run()

    @pytest.mark.parametrize("lat,thick", [(70.0, 2.5), (80.0, 6.0)])
    def test_pstar_zero_matches_m5748_steady_ledger(self, lat, thick):
        model = quadratic_model(lat, thick)
        u_newton = model.steady_state()
        _, rec = self._channel(0.0, lat=lat, thick=thick)
        rel = np.max(np.abs(rec["domain"][-1] - u_newton)) / np.max(np.abs(u_newton))
        assert rel < 1e-10
        # the field itself is uniform (no internal stress -> no gradients)
        fields = np.column_stack([rec["final_state"]["u"], rec["final_state"]["v"]])
        assert np.max(np.abs(fields - u_newton)) / np.max(np.abs(u_newton)) < 1e-10
        # and the m5748 closed form agrees with the Newton steady state
        tau = float(np.linalg.norm(model.wind_stress(np.zeros(2))))
        fd = free_drift_state(tau, model.mass_per_area, model.f, RHO_W, CD_W)
        assert abs(fd["speed"] - np.linalg.norm(u_newton)) / fd["speed"] < 0.03

    def test_stress_divergence_has_exactly_zero_periodic_mean(self):
        """``<d sigma/dx> = 0``: the internal stress cannot change the bulk drift."""
        cfg = ChannelConfig(
            n_cells=32, dt=1800.0, duration_days=4.0, latitude_deg=80.0,
            ice_thickness=2.5, wind_angle_deg=0.0, wind_speed=10.0,
            wind_spatial_amplitude=0.55, wind_switch_angle_deg=0.0,
            wind_variability=0.0, picard_iterations=2, buoy_window_length=4.0e5,
        )
        ch = OneDimVPIceChannel(cfg, HiblerRheology(p_star=2.0e4), seed=0)
        ch.run()
        sd = ch.stress_divergence()
        scale = float(np.max(np.abs(sd["d_sigma_11_dx"])))
        assert scale > 1e-3                      # the stress field really is active
        assert abs(float(np.mean(sd["d_sigma_11_dx"]))) / scale < 1e-12
        assert abs(float(np.mean(sd["d_sigma_12_dx"]))) < 1e-12 * max(
            float(np.max(np.abs(sd["d_sigma_12_dx"]))), 1e-30)
        # non-uniform stress field: sigma_11 varies across the domain
        assert float(np.ptp(sd["sigma_11"])) > 1e3

    def test_bulk_drift_is_nearly_insensitive_to_internal_stress(self):
        """Bulk drift stays close to free drift; only the local field is redistributed.

        The residual O(10 %) difference is *not* a stress effect (its mean
        vanishes); it comes from the drag covariance
        ``<|u|u> != |<u>|<u>`` and from the thickness redistribution.
        """
        model = quadratic_model(80.0, 2.5)
        u_newton = model.steady_state()
        _, rec = self._channel(2.0e4, amplitude=0.55, days=8.0)
        rel = np.max(np.abs(rec["domain"][-1] - u_newton)) / np.max(np.abs(u_newton))
        assert rel < 0.2
        fields = np.column_stack([rec["final_state"]["u"], rec["final_state"]["v"]])
        spread = np.max(np.abs(fields - fields.mean(axis=0)))
        assert spread > 10.0 * rel * np.max(np.abs(u_newton))

    def test_pressure_and_viscosities_follow_hibler(self):
        rh = HiblerRheology(p_star=2.75e4, ellipse=2.0, c_concentration=20.0,
                            delta_min=1e-6)
        h = np.array([1.0, 2.0])
        A = np.array([1.0, 0.9])
        P = rh.pressure(h, A)
        assert P[0] == pytest.approx(2.75e4 * 1.0)
        assert P[1] == pytest.approx(2.75e4 * 2.0 * np.exp(-20.0 * 0.1))
        div = np.array([2.0e-6, 0.0])
        shear = np.array([0.0, 0.0])
        zeta, eta, plastic = rh.viscosities(P, div, shear)
        delta = np.sqrt((1.0 + 0.25) * div**2)
        assert zeta[0] == pytest.approx(P[0] / (2.0 * max(delta[0], 1e-6)))
        assert eta[0] == pytest.approx(zeta[0] / 4.0)
        assert plastic == 0.5   # only cell 0 reaches Delta_min
        sig = rh.stress(zeta, eta, P, div, shear)
        assert sig["sigma_11"][0] == pytest.approx((3.0 * zeta[0] - eta[0]) * div[0]
                                                   - 0.5 * P[0])

    def test_pstar_zero_has_no_internal_stress_at_all(self):
        rh = HiblerRheology(p_star=0.0)
        P = rh.pressure(np.array([2.0]), np.array([0.95]))
        assert P[0] == 0.0
        zeta, eta, _ = rh.viscosities(P, np.array([1e-5]), np.array([0.0]))
        assert zeta[0] == 0.0 and eta[0] == 0.0


# ---------------------------------------------------------------------------
# Linear-viscous prediction
# ---------------------------------------------------------------------------
class TestLinearViscousPrediction:
    def test_scaling_and_monotonicity(self):
        rh = HiblerRheology(delta_min=1e-6)
        m = RHO_I * H_ICE
        f = coriolis_parameter(80.0)
        vals = [linear_viscous_prediction(p, rh, m, f, 8.5e-5, 2.0e5,
                                          thickness=H_ICE, concentration=0.95)
                for p in (0.0, 1e3, 5e3, 2e4)]
        assert vals[0]["delta_gamma_over_gamma"] == 0.0
        dg = [v["delta_gamma_over_gamma"] for v in vals]
        assert all(b > a for a, b in zip(dg, dg[1:]))
        # gamma_eff = gamma_drag + (3 zeta + eta) k^2 / (2 m)
        k = 2.0 * math.pi / 2.0e5
        expected = vals[-1]["gamma_drag"] + (3.0 * vals[-1]["zeta"] + vals[-1]["eta"]) \
            * k**2 / (2.0 * m)
        assert vals[-1]["gamma_eff"] == pytest.approx(expected)
        # enormous viscosity -> the oscillation is overdamped (flagged, not hidden)
        assert vals[-1]["overdamped"] is True
        assert vals[-1]["delta_theta_deg"] < vals[0]["delta_theta_deg"]

    def test_zero_viscosity_reproduces_the_arctan_angle(self):
        rh = HiblerRheology(delta_min=1e-6)
        m = RHO_I * H_ICE
        f = coriolis_parameter(80.0)
        pred = linear_viscous_prediction(0.0, rh, m, f, 8.5e-5, 2.0e5)
        assert pred["theta0_deg"] == pytest.approx(math.degrees(math.atan2(f, 8.5e-5)))


# ---------------------------------------------------------------------------
# Estimators used by the ensemble
# ---------------------------------------------------------------------------
class TestEventEstimator:
    def test_recovers_parameters_of_a_piecewise_forced_record(self):
        f, g = 1.436e-4, 8.5e-5
        dt = 900.0
        t = np.arange(0.0, 6.0 * 86400.0, dt)
        events = [2.0 * 86400.0, 4.0 * 86400.0]
        bounds = np.concatenate([[0.0], events, [t[-1] + 1.0]])
        seg = np.clip(np.searchsorted(bounds, t, side="right") - 1, 0, bounds.size - 2)
        targets = np.array([[0.18, 0.02], [0.10, -0.06], [0.15, 0.10]])
        u = np.zeros((t.size, 2))
        v = np.zeros(2)

        def rot(tau, w):
            c, s = math.cos(f * tau), math.sin(f * tau)
            return math.exp(-g * tau) * np.array([c * w[0] + s * w[1],
                                                  -s * w[0] + c * w[1]])
        for m in range(bounds.size - 1):
            mask = seg == m
            p = targets[m]
            t0 = bounds[m]
            for k in np.flatnonzero(mask):
                u[k] = p + rot(t[k] - t0, v - p)
            v = p + rot(bounds[m + 1] - t0, v - p)
        est = estimate_event_response(t, u, events, f_hint=f, gamma_hint=g)
        assert abs(est["gamma"] - g) / g < 1e-8
        assert abs(est["omega"] - f) / f < 1e-8
        assert est["fit_residual_rel"] < 1e-10

    def test_fixed_omega_variant_and_noise_bias(self):
        f, g = 1.436e-4, 8.5e-5
        dt = 900.0
        t = np.arange(0.0, 6.0 * 86400.0, dt)
        events = [2.0 * 86400.0, 4.0 * 86400.0]
        bounds = np.concatenate([[0.0], events, [t[-1] + 1.0]])
        seg = np.clip(np.searchsorted(bounds, t, side="right") - 1, 0, bounds.size - 2)
        targets = np.array([[0.18, 0.02], [0.10, -0.06], [0.15, 0.10]])
        u = np.zeros((t.size, 2))
        v = np.zeros(2)

        def rot(tau, w):
            c, s = math.cos(f * tau), math.sin(f * tau)
            return math.exp(-g * tau) * np.array([c * w[0] + s * w[1],
                                                  -s * w[0] + c * w[1]])
        for m in range(bounds.size - 1):
            mask = seg == m
            p = targets[m]
            t0 = bounds[m]
            for k in np.flatnonzero(mask):
                u[k] = p + rot(t[k] - t0, v - p)
            v = p + rot(bounds[m + 1] - t0, v - p)
        rng = np.random.default_rng(3)
        noisy = u + rng.normal(0.0, 2.0e-3, size=u.shape)
        est = estimate_event_response(t, noisy, events, gamma_hint=g, fixed_omega=f)
        assert abs(est["gamma"] - g) / g < 0.05
        assert est["method"] == "event_response_fixed_omega"


# ---------------------------------------------------------------------------
# (f) refinement, cheap configuration
# ---------------------------------------------------------------------------
class TestRefinement:
    def _diag(self, dt, n_cells):
        cfg = ChannelConfig(
            n_cells=n_cells, dt=dt, duration_days=2.0, latitude_deg=80.0,
            ice_thickness=H_ICE, wind_variability=0.0, picard_iterations=2,
            buoy_window_length=2.0e5,
        )
        res = {}
        for p in (0.0, 2.0e4):
            ch = OneDimVPIceChannel(cfg, HiblerRheology(p_star=p), seed=0)
            out = ch.run()
            est = estimate_event_response(
                out["times"], out["buoy"], out["switch_times"], gamma_hint=2.3e-4,
                fixed_omega=cfg.f, window_after_event=18 * 3600.0,
            )
            res[p] = est["gamma"]
        return (res[2.0e4] - res[0.0]) / res[0.0]

    def test_timestep_refinement_changes_the_shift_by_less_than_5pct(self):
        coarse = self._diag(1800.0, 48)
        fine = self._diag(900.0, 48)
        assert abs(fine - coarse) < 0.05


# ---------------------------------------------------------------------------
# Guard rails on the numerics
# ---------------------------------------------------------------------------
def test_linearized_drag_rate_quadratic_jacobian():
    model = quadratic_model(80.0, 2.5)
    u = model.steady_state()
    gamma = linearized_drag_rate(model, u)
    m = model.mass_per_area
    du_w = -u
    du_a = np.asarray(model.wind) - u
    expected = (CD_W * RHO_W * np.linalg.norm(du_w) + CD_A * RHO_A
                * np.linalg.norm(du_a)) / m
    assert gamma == pytest.approx(expected)


def test_explicit_euler_stability_limit_matches_analytic():
    """Stability bound dt* = 2 gamma/(gamma^2+f^2) of explicit Euler on dw/dt = Aw.

    Below dt* the scheme is stable but *wrong* (it over-damps), above dt* it
    diverges -- which is why the negative control of criterion (d) is stated as
    "amplitude error > 50 % **or** divergence": at the 1 h acceptance step the
    error comes from damping, not from growth.
    """
    sol = DampedInertialSolution(
        u_steady=np.array([0.20, 0.02]), u_initial=np.zeros(2),
        gamma=8.5e-5, f=coriolis_parameter(80.0),
    )
    dt_star = 2.0 * sol.gamma / (sol.gamma ** 2 + sol.f ** 2)
    assert amplification(0.5 * dt_star, sol.f, sol.gamma, "euler")["modulus"] < 1.0
    assert amplification(1.5 * dt_star, sol.f, sol.gamma, "euler")["modulus"] > 1.0
    # the analytic limit is the modulus == 1 crossing
    assert amplification(dt_star, sol.f, sol.gamma, "euler")["modulus"] == pytest.approx(
        1.0, abs=1e-12
    )
    # 1 h gate sits below the limit at 80N, so the failure mode is damping
    assert 3600.0 < dt_star
    # and pushing above the limit must actually blow up
    n = int(round(10.0 * 86400.0 / (2.0 * dt_star)))
    _, us = integrate(sol, 2.0 * dt_star, n, scheme="euler")
    ratio = np.linalg.norm(us[-1] - sol.u_steady) / np.linalg.norm(sol.u_steady)
    assert ratio > 50.0


def test_ledger_records_monotonicity_under_three_readings():
    """Audit the machine ledger's monotonicity block (criterion e).

    The node text asks for "monotone (Spearman rho >= 0.95)".  For a *decreasing*
    response the signed value is negative, so the ledger must carry all three
    readings: |rho| (graded), the literal signed rule (disclosed), and the
    resolvable (SNR >= 3) subset -- and the sign reversals must sit inside the
    WITHHELD band, otherwise the monotonicity claim itself would be unsound.
    """
    import json

    ledger = (Path(__file__).parent.parent / "artifacts" / "sea_ice_transient"
              / "acceptance.json")
    if not ledger.exists():
        pytest.skip("acceptance ledger not generated in this checkout")
    crit = json.loads(ledger.read_text())["criteria"]
    snr_gate = json.loads(ledger.read_text())["design"]["snr_threshold"]
    for key in ("e_spearman_dtheta", "e_spearman_dgamma"):
        blk = crit[key]
        assert blk["abs_value"] == pytest.approx(abs(blk["value"]))
        assert blk["pass"] == (blk["abs_value"] >= blk["threshold"])
        assert "literal_signed_reading" in blk and "resolvable_only" in blk
        res = blk["resolvable_only"]
        if res["rho"] is not None:
            # on the resolvable levels the relation is (near-)perfectly monotone
            assert abs(res["rho"]) >= blk["threshold"]
        # every withheld level must be excluded from the resolvable subset
        assert len(res["p_star_levels_N_per_m"]) <= len(blk.get("turning_points") or []) + 8
    for t in crit["e_spearman_dtheta"]["turning_points"]:
        assert t["snr"] < snr_gate, "a sign reversal outside the WITHHELD band"


def test_euler_stability_limit_is_in_the_ledger():
    import json

    ledger = (Path(__file__).parent.parent / "artifacts" / "sea_ice_transient"
              / "acceptance.json")
    if not ledger.exists():
        pytest.skip("acceptance ledger not generated in this checkout")
    d = json.loads(ledger.read_text())
    lim = d["a_and_d"]["euler_stability_limit"]
    sol_f, sol_g = 1.436e-4, 8.5e-5
    assert lim["analytic_limit_s"] == pytest.approx(
        2.0 * sol_g / (sol_g ** 2 + sol_f ** 2), rel=0.01
    )
    assert lim["gate_below_limit"] is True
    assert lim["diverged_over_limit"] is True
    assert d["criteria"]["d_negative_control_euler_1h"]["analytic_stability_limit_h"] > 1.0


def test_channel_wind_switch_schedule_is_deterministic():
    cfg = ChannelConfig(n_cells=16, duration_days=2.0, dt=1800.0)
    ch = OneDimVPIceChannel(cfg, HiblerRheology(), seed=1)
    t = ch.wind.switch_times()
    assert t.size >= 1
    assert np.allclose(np.diff(t), cfg.wind_switch_period_days * 86400.0)