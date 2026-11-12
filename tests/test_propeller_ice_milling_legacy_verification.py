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


def test_debris_carrier_is_the_race_model_field():
    """D2 repair: the transport carrier is ``RaceModel``'s own field (width ``R/2``).

    The old defect used a local ``sigma = 0.5 D`` closure (= ``R``, twice the wake
    width and hence a 4x weaker ``u_c``).  This test is written so that the old
    closure *fails* it: the analytic crossing with the module's carrier reproduces
    the RK2 exit speed, and the same closed form evaluated with the old width
    differs by far more than the 2 % tolerance.
    """
    race, mass, diam, c_d, k = _drag_setup()
    u = float(race.axial_velocity(0.0, 0.0))            # module's own carrier speed
    _, v_ref = _analytic_crossing(u, 20.0, k, 1.0)
    out = integrate_debris([{"mass": mass, "d": diam, "x": 0.0, "r": 0.0, "vx": 20.0, "vr": 0.0}],
                           race, x_exit=1.0, dt=2.5e-4, c_d=c_d)
    assert out[0]["exit_speed"] == pytest.approx(v_ref, rel=2.0e-2)
    # discrimination: the pre-repair sigma = 0.5 D carrier predicts something else
    sigma_old = 0.5 * race.diameter
    u_old = float(race.centreline(0.0)) * (race.sigma(0.0) / sigma_old)
    _, v_old = _analytic_crossing(u_old, 20.0, k, 1.0)
    assert abs(v_old - out[0]["exit_speed"]) / out[0]["exit_speed"] > 0.2


def test_axial_velocity_tensor_agrees_with_scalar_field():
    """The vectorised race field is the same field as the scalar API (1e-12)."""
    race = RaceModel(thrust=3.0e4, diameter=2.4, spreading_rate=0.085)
    x = torch.linspace(0.0, 6.0, 13, dtype=torch.float64)
    r = torch.linspace(0.0, 2.0, 11, dtype=torch.float64)
    xx, rr = torch.meshgrid(x, r, indexing="ij")
    field = race.axial_velocity_tensor(xx, rr)
    ref = torch.tensor([[race.axial_velocity(float(a), float(b)) for b in r] for a in x],
                       dtype=torch.float64)
    assert torch.allclose(field, ref, rtol=0.0, atol=1.0e-12 * float(ref.max()))
    sig = race.sigma_tensor(x)
    assert torch.allclose(sig, torch.tensor([race.sigma(float(a)) for a in x],
                                            dtype=torch.float64), rtol=1.0e-12)


# ---------------------------------------------------------------------------
# propeller race: momentum conservation (D1 repair)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x", [0.0, 1.0, 4.0])
def test_race_conserves_momentum_flux(x):
    """D1 repair: ``int_0^inf rho u^2 2 pi r dr = T`` at every station (1 %).

    Pre-repair the ratio was ``0.5`` exactly (``u_c = sqrt(M/(2 pi sigma^2))``);
    the repair is ``u_c = sqrt(M/(pi sigma^2))``, i.e. a factor ``sqrt(2)``.
    """
    thrust, diameter = 2.0e4, 3.0
    race = RaceModel(thrust=thrust, diameter=diameter)
    sigma = race.sigma(x)
    r = np.linspace(0.0, 8.0 * sigma, 4001)
    u = np.array([race.axial_velocity(x, float(rr)) for rr in r])
    flux = float(np.trapezoid(RHO_W * u ** 2 * 2.0 * math.pi * r, r))
    assert flux == pytest.approx(thrust, rel=1.0e-2)
    assert abs(flux / thrust - 0.5) > 0.1            # the old half-flux cannot pass


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
    """DEFECT-D3 repair guard (mission m16448): the cap is now *reported*.

    Pre-repair the solver sat on ``a = 0.85`` for every strip, returned a residual
    of ~1e-8 and said nothing else: the blade-element thrust was unbalanced by the
    annulus momentum and no residual-based test could see it.  The repair adds an
    independent momentum residual plus a model-validity flag, so this test asserts
    both halves -- the registered numbers still reproduce, and the state now
    carries the diagnosis the pre-repair code did not have (which is why the
    pre-repair code fails the last four assertions).
    """
    from tensordem.propeller_ice.geometry import PropellerGeometry as PkgGeometry
    from tensordem.propeller_ice.hydro import OpenWaterConfig, solve_open_water

    geom = PkgGeometry(diameter=3.0, n_blades=4, hub_ratio=0.40, pitch_ratio=0.95,
                       chord_ratio=0.539, skew_tip_deg=0.0, thickness_ratio=0.05)
    state = solve_open_water(geom, 1.0, 0.2 * 3.0, config=OpenWaterConfig(n_strips=24))
    assert int(np.sum(np.asarray(state.a) >= 0.849)) == np.asarray(state.a).size
    assert state.residual < 1.0e-6             # converged by its own metric
    assert state.thrust_coefficient > 0.6      # far above the legacy strip result (0.315)
    # --- what the repair added (these fail on the pre-repair code) ---------
    assert state.momentum_residual > 1.0e-2    # ~7 % of the thrust is unbalanced
    assert not state.model_valid
    assert state.validity_flag == "thrust_closure_exceeded"
    assert state.ct_max_loading > 10.0         # ~19x above the closure bound


