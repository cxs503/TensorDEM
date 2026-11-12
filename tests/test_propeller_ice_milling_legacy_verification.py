"""Verification gates for the legacy IN27-S4 propeller-ice milling module.

``src/tensordem/propeller_ice_milling.py`` (1140 lines, campaign node IN27-S4)
shipped with **no test module, no driver and no documentation reference** while
the same physics is implemented and tested a second time in
``tensordem.propeller_ice`` (IN34-W3-N05).  These gates pin the legacy module's
closed-form claims against independent references and record the three defects
that the exercise uncovered (see ``docs/propeller_ice_milling_legacy_verification.md``
and ``artifacts/propeller_ice_milling_legacy/verification.json``).

Green tests = claims that hold.  ``xfail(strict=True)`` tests = claims that do
*not* hold today; they are written as the acceptance criterion of the fix, so
they will turn red (unexpectedly passing) the moment the defect is repaired,
which is the intended signal.  ``test_defect_*`` tests characterise the current
behaviour so that a fix is visible in the diff.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordem.propeller_ice_milling import (
    PropellerGeometry,
    RaceModel,
    blade_element_open_water,
    integrate_debris,
    torque_spectrum,
)

RHO_W = 1025.0


def _beta(e: float) -> float:
    return math.gamma(1.0 + e) ** 2 / math.gamma(2.0 + 2.0 * e)


# ---------------------------------------------------------------------------
# geometry: planform and expanded-area closure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("exponent", [0.20, 0.35, 0.50])
def test_chord_scale_matches_closed_form_beta_integral(exponent):
    """``c_max = A_blade / (B(1+e,1+e)/0.615 * (R - r_h))`` to 1e-4."""
    g = PropellerGeometry(planform_exponent=exponent)
    span = g.radius - g.hub_radius
    exact = g.blade_area / (_beta(exponent) / 0.615 * span)
    assert g.chord_scale() == pytest.approx(exact, rel=1.0e-4)


@pytest.mark.parametrize("exponent", [0.20, 0.35, 0.50])
def test_panel_set_reproduces_expanded_area_ratio(exponent):
    """Sum of contact-panel areas reproduces ``EAR * pi D^2 / 4`` (2 %)."""
    g = PropellerGeometry(planform_exponent=exponent)
    assert float(g.panels()["area"].sum()) == pytest.approx(g.total_blade_area, rel=2.0e-2)


def test_pitch_angle_law_is_helicoidal():
    g = PropellerGeometry()
    r = 0.7 * g.radius
    assert g.pitch_angle(r) == pytest.approx(math.atan(g.pitch / (2.0 * math.pi * r)),
                                             rel=1.0e-12)


# ---------------------------------------------------------------------------
# torque spectrum: analytic harmonic recovery
# ---------------------------------------------------------------------------
def _harmonic_train():
    rpm, n_blades, fs, duration = 90.0, 4, 2048.0, 2.0
    bpf = rpm / 60.0 * n_blades
    t = np.arange(0.0, duration, 1.0 / fs)
    amps = {1: 120.0, 2: 40.0, 3: 15.0}
    signal = 1000.0 + sum(amps[k] * np.cos(2.0 * math.pi * k * bpf * t) for k in amps)
    return bpf, t, signal, amps


def test_torque_spectrum_recovers_analytic_harmonics():
    bpf, t, signal, amps = _harmonic_train()
    res = torque_spectrum(torch.tensor(t), torch.tensor(signal), bpf, n_harmonics=2)
    assert res.peak_frequency == pytest.approx(bpf, rel=5.0e-3)
    assert res.bpf_amplitude == pytest.approx(amps[1], rel=5.0e-3)
    assert res.harmonic_amplitudes[0] == pytest.approx(amps[2], rel=2.0e-2)
    assert res.harmonic_amplitudes[1] == pytest.approx(amps[3], rel=2.0e-2)


def test_torque_spectrum_white_noise_control():
    """Negative control: pure noise must not concentrate its peak on the BPF."""
    bpf, t, _, _ = _harmonic_train()
    rng = np.random.default_rng(20261112)
    noise = torch.tensor(1000.0 + 50.0 * rng.standard_normal(t.size))
    res = torque_spectrum(torch.tensor(t), noise, bpf, n_harmonics=2)
    assert abs(res.peak_frequency - bpf) > 1.0


# ---------------------------------------------------------------------------
# open-water: internal identities and monotonicity
# ---------------------------------------------------------------------------
def test_open_water_efficiency_identity_and_monotone_thrust():
    g = PropellerGeometry()
    kts = []
    for J in (0.2, 0.4, 0.6, 0.8):
        res = blade_element_open_water(g, 1.0, J * g.diameter, n_strips=24)
        assert res["K_T"] > 0.0 and res["K_Q"] > 0.0
        eta_id = res["advance_ratio"] * res["K_T"] / (2.0 * math.pi * res["K_Q"])
        assert res["efficiency"] == pytest.approx(eta_id, rel=1.0e-10)
        kts.append(res["K_T"])
    assert all(a > b for a, b in zip(kts, kts[1:]))


# ---------------------------------------------------------------------------
# debris drag: integrator vs closed form
# ---------------------------------------------------------------------------
def _drag_setup():
    race = RaceModel(thrust=5.0e4, diameter=3.0, spreading_rate=1.0e-12)
    mass, diam, c_d = 0.5, 0.05, 1.10
    k = 0.5 * RHO_W * c_d * math.pi * (0.5 * diam) ** 2 / mass
    return race, mass, diam, c_d, k


def _analytic_crossing(u: float, v0: float, k: float, x_exit: float) -> tuple[float, float]:
    w0 = v0 - u
    x_of_t = lambda tt: u * tt + math.log1p(k * w0 * tt) / k
    lo, hi = 0.0, 1.0
    while x_of_t(hi) < x_exit:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if x_of_t(mid) < x_exit else (lo, mid)
    t_cross = 0.5 * (lo + hi)
    return t_cross, u + w0 / (1.0 + k * w0 * t_cross)


def test_defect_d2_debris_carrier_width_differs_from_race_model():
    """DEFECT-D2 characterisation: ``integrate_debris`` uses ``sigma = R``, not ``R/2``.

    With the *integrator's own* wake width (``sigma = 0.5 D``) the RK2 transport
    reproduces the closed-form quadratic-drag solution, which isolates the defect
    to the width/normalisation convention rather than to the ODE integration.
    """
    race, mass, diam, c_d, k = _drag_setup()
    sigma_int = 0.5 * race.diameter                     # what integrate_debris assumes
    u_int = float(race.centreline(0.0)) * (race.sigma(0.0) / sigma_int)  # u_c ~ 1/sigma
    _, v_ref = _analytic_crossing(u_int, 20.0, k, 1.0)
    out = integrate_debris([{"mass": mass, "d": diam, "x": 0.0, "r": 0.0, "vx": 20.0, "vr": 0.0}],
                           race, x_exit=1.0, dt=2.5e-4, c_d=c_d)
    assert out[0]["exit_speed"] == pytest.approx(v_ref, rel=2.0e-2)
    # ... and that width is twice the RaceModel's own sigma
    assert sigma_int == pytest.approx(2.0 * race.sigma(0.0), rel=1.0e-12)


@pytest.mark.xfail(strict=True, reason="DEFECT-D2: integrate_debris uses sigma=R while "
                                      "RaceModel.sigma()=R/2 -> chips see a 4x weaker jet")
def test_debris_exit_speed_matches_closed_form_in_race_model_carrier():
    race, mass, diam, c_d, k = _drag_setup()
    u = race.axial_velocity(0.0, 0.0)                   # module's own carrier speed
    _, v_ref = _analytic_crossing(u, 20.0, k, 1.0)
    out = integrate_debris([{"mass": mass, "d": diam, "x": 0.0, "r": 0.0, "vx": 20.0, "vr": 0.0}],
                           race, x_exit=1.0, dt=2.5e-4, c_d=c_d)
    assert out[0]["exit_speed"] == pytest.approx(v_ref, rel=2.0e-2)


# ---------------------------------------------------------------------------
# propeller race: momentum conservation
# ---------------------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason="DEFECT-D1: u_c = sqrt(M/(2 pi sigma^2)) delivers "
                                      "int rho u^2 2 pi r dr = T/2, not T")
def test_race_conserves_momentum_flux():
    thrust, diameter = 2.0e4, 3.0
    race = RaceModel(thrust=thrust, diameter=diameter)
    sigma = race.sigma(1.0)
    r = np.linspace(0.0, 8.0 * sigma, 4001)
    u = np.array([race.axial_velocity(1.0, float(rr)) for rr in r])
    flux = float(np.trapezoid(RHO_W * u ** 2 * 2.0 * math.pi * r, r))
    assert flux == pytest.approx(thrust, rel=1.0e-2)


def test_defect_d1_race_momentum_flux_is_half_the_thrust():
    """DEFECT-D1 characterisation: the delivered flux is ``T/2`` to quadrature accuracy."""
    thrust, diameter = 2.0e4, 3.0
    race = RaceModel(thrust=thrust, diameter=diameter)
    sigma = race.sigma(0.0)
    r = np.linspace(0.0, 8.0 * sigma, 4001)
    u = np.array([race.axial_velocity(0.0, float(rr)) for rr in r])
    flux = float(np.trapezoid(RHO_W * u ** 2 * 2.0 * math.pi * r, r))
    assert flux == pytest.approx(0.5 * thrust, rel=1.0e-2)


def test_race_induced_velocity_matches_actuator_disk_closed_form():
    thrust, diameter, v_ship = 2.0e4, 3.0, 0.5
    race = RaceModel(thrust=thrust, diameter=diameter)
    area = math.pi * (0.5 * diameter) ** 2
    ref = -0.5 * v_ship + math.sqrt(0.25 * v_ship ** 2 + (thrust / RHO_W) / (2.0 * area))
    assert race.induced_velocity(v_ship) == pytest.approx(ref, rel=1.0e-12)


# ---------------------------------------------------------------------------
# cross-implementation: package BEM saturates its induction cap at low J
# ---------------------------------------------------------------------------
def test_defect_d3_package_bem_saturates_induction_cap_at_low_J():
    """DEFECT-D3 characterisation: every strip sits on ``a = 0.85`` at J = 0.2.

    The blade-element load is then unbalanced by the annulus momentum while the
    returned residual (``max|delta a|``) is ~1e-8, so a residual-based acceptance
    test cannot see it.  ``_buhl`` is never called by ``solve_open_water``.
    """
    from tensordem.propeller_ice.geometry import PropellerGeometry as PkgGeometry
    from tensordem.propeller_ice.hydro import OpenWaterConfig, solve_open_water

    geom = PkgGeometry(diameter=3.0, n_blades=4, hub_ratio=0.40, pitch_ratio=0.95,
                       chord_ratio=0.539, skew_tip_deg=0.0, thickness_ratio=0.05)
    state = solve_open_water(geom, 1.0, 0.2 * 3.0, config=OpenWaterConfig(n_strips=24))
    assert int(np.sum(np.asarray(state.a) >= 0.849)) == np.asarray(state.a).size
    assert state.residual < 1.0e-6            # converged by its own metric
    assert state.thrust_coefficient > 0.6      # far above the legacy strip result (0.315)