"""Tests for the planar-Poiseuille precision face (IN52-N02).

The reference channel has an analytic profile; the bounded variants differ from
it only in the boundary rules, so the measured deviation *is* the boundary error.
"""
from __future__ import annotations

import math

import pytest

from tensordem.threefield.channel_poiseuille import (
    ChannelConfig,
    ChannelFlow,
    plug_index,
    poiseuille_flux,
    poiseuille_peak,
    poiseuille_profile,
    profile_error,
    run_channel,
)
from tensordem.threefield.wake_lbm import LBMConfig, WakeLBM


def _force(tau0: float, peak: float = 0.02, ny: int = 16) -> float:
    nu = (tau0 - 0.5) / 3.0
    return 8.0 * nu * peak / float(ny) ** 2


# -- analytic reference ------------------------------------------------------
def test_profile_peak_and_flux_match_the_closed_forms():
    force, nu, ny = 6.25e-5, 0.1, 16
    profile = poiseuille_profile(force, nu, ny)
    # the row maximum sits half a cell off the centreline: 1 - (1/2)^2/(H/2)^2
    assert max(profile) == pytest.approx(poiseuille_peak(force, nu, ny), rel=5e-3)
    # the row sum is the trapezoidal/mid-point sum of the parabola on this grid
    assert sum(profile) == pytest.approx(poiseuille_flux(force, nu, ny), rel=1e-2)
    assert profile[0] == pytest.approx(profile[-1], rel=1e-12)


def test_plug_index_distinguishes_a_parabola_from_a_plug():
    # the parabola's largest deviation from its mean sits at the wall
    profile = poiseuille_profile(6.25e-5, 0.1, 16)
    mean = sum(profile) / len(profile)
    assert plug_index(profile) == pytest.approx(1.0 - (mean - profile[0]) / mean, rel=1e-12)
    assert plug_index(profile) == pytest.approx(0.181, rel=1e-2)
    assert plug_index([0.02] * 16) == pytest.approx(1.0, rel=1e-12)


def test_profile_error_is_normalised_by_the_peak():
    reference = poiseuille_profile(6.25e-5, 0.1, 16)
    peak = max(reference)
    doubled = [2.0 * value for value in reference]
    error = profile_error(doubled, reference, peak=peak)
    assert error["linf"] == pytest.approx(1.0, rel=1e-12)
    rms = math.sqrt(sum(value * value for value in reference) / len(reference))
    assert error["l2"] == pytest.approx(rms / peak, rel=1e-12)


# -- the shipped boundary rules ----------------------------------------------
def test_uniform_stream_is_an_exact_fixed_point_of_the_bounded_channel():
    """Mirrors ``boundary_verification.uniform_stream_check`` in the channel solver.

    Free-slip walls plus a zero-gradient outlet must carry a uniform stream
    without drift, a spurious boundary layer or a transverse velocity.
    """
    config = ChannelConfig(
        ny=16, nx=24, tau0=0.8, force=0.0, axial="bounded", outlet="zero_gradient",
        wall="free_slip", forcing="guo", inlet_profile="uniform", inlet_speed=0.02,
        initial="inlet",
    )
    flow = ChannelFlow(config)
    mass0 = flow.mass()
    flow.step(200)
    velocity = flow.velocity()
    assert float(velocity[..., 0].max().item()) == pytest.approx(0.02, rel=1e-12)
    assert float(velocity[..., 0].min().item()) == pytest.approx(0.02, rel=1e-12)
    assert float(velocity[..., 1].abs().max().item()) < 1e-14
    # the outlet mirror still lets mass escape from the last plane only
    assert abs(flow.mass() - mass0) < 1e-6 * mass0


def test_free_slip_channel_cannot_hold_a_poiseuille_profile():
    """No wall shear, no steady channel: the plug accelerates at exactly ``G``.

    This is the model-form statement behind "planar Poiseuille is not
    representable by the shipped lateral walls", measured rather than asserted:
    the momentum source is the body force alone, so ``du/dt = G`` to rounding.
    """
    force = _force(0.8)
    config = ChannelConfig(ny=16, nx=8, tau0=0.8, force=force, axial="periodic",
                           wall="free_slip", forcing="guo")
    flow = ChannelFlow(config)
    flow.step(400)
    peak = float(flow.velocity()[..., 0].max().item())
    assert peak / 400.0 == pytest.approx(force, rel=1e-6)   # du/dt = G
    assert plug_index(flow.profile()) == pytest.approx(1.0, rel=1e-9)  # a plug, not a parabola
    # the parabola's peak is fixed by G and nu while the plug ramps linearly, so
    # the deviation from the analytic channel grows without bound
    assert max(flow.profile()) / poiseuille_peak(force, 0.1, 16) > 1.2


# -- the two force schemes ---------------------------------------------------
def test_consistent_guo_source_matches_the_parabola():
    """With the (1 - omega/2) prefactor the discrete momentum balance reads
    ``div(Pi) = F``, so the periodic no-slip channel reproduces the flyspeck
    parabola to well inside the criterion band."""
    force = _force(0.8)
    result = run_channel(
        ChannelConfig(ny=16, nx=8, tau0=0.8, force=force, axial="periodic",
                      wall="no_slip", forcing="guo"),
        steps=3000, sample_every=500,
    )
    assert 100.0 * result.error["linf"] < 1.5
    assert abs(result.mass_drift_rate) < 1e-12


def test_shipped_simple_source_overdrives_by_one_plus_omega_half():
    """The historical ``3 w (c.F)`` source is over-driven by ``1 + omega/2``.

    Kept as a *characterisation* test: it books the behaviour of the production
    path (``force_scheme="simple"``) so nobody can silently "fix" it without
    also updating the wake-benchmark diff table.
    """
    tau0 = 0.8
    force = _force(tau0)
    result = run_channel(
        ChannelConfig(ny=16, nx=8, tau0=tau0, force=force, axial="periodic",
                      wall="no_slip", forcing="simple"),
        steps=3000, sample_every=500,
    )
    predicted = 1.0 + 0.5 / tau0
    measured = max(result.profile) / poiseuille_peak(force, (tau0 - 0.5) / 3.0, 16)
    assert measured == pytest.approx(predicted, rel=0.02)
    assert measured > 1.5


def test_zero_gradient_outlet_piles_up_mass_during_the_fill_transient():
    """From a rest start the zero-gradient outlet cannot evacuate what the
    Dirichlet inlet injects, so mass accumulates; the density anchor does not.
    (Started in the developed state both close the books to rounding -- the
    production consequence is a start-up drift, not a steady-state leak.)"""
    force = _force(0.8)
    results = {}
    for outlet in ("zero_gradient", "pressure_anchor"):
        results[outlet] = run_channel(
            ChannelConfig(ny=16, nx=40, tau0=0.8, force=force, axial="bounded",
                          outlet=outlet, wall="no_slip", forcing="guo"),
            steps=3000, sample_every=500, developed_fraction=0.3,
        )
    anchor = results["pressure_anchor"]
    gradient = results["zero_gradient"]
    assert abs(anchor.mass_drift_rate) < 1e-10
    assert abs(anchor.mass_drift_rate) < abs(gradient.mass_drift_rate) / 10.0
    assert 100.0 * gradient.error["linf"] < 5.0
    assert math.isfinite(anchor.error["linf"])


# -- the opt-in wake-solver scheme -------------------------------------------
def test_default_force_scheme_keeps_the_legacy_update():
    base = LBMConfig(nx=16, ny=16, nz=16, tau0=0.5005, smagorinsky_cs=0.14,
                     inflow_velocity=0.04, jet_velocity=0.12, disk_x=8,
                     disk_radius=3.0, disk_thickness=2, quarter_domain=True)
    assert base.force_scheme == "simple"
    legacy = WakeLBM(base)
    explicit = WakeLBM(
        LBMConfig(**{**base.__dict__, "force_scheme": "simple"})
    )
    consistent = WakeLBM(LBMConfig(**{**base.__dict__, "force_scheme": "guo"}))
    legacy.step(5)
    explicit.step(5)
    consistent.step(5)
    assert legacy.thrust_impulse == explicit.thrust_impulse
    assert abs(consistent.thrust_impulse - legacy.thrust_impulse) > 1e-6 * legacy.thrust_impulse


def test_unknown_force_scheme_is_rejected():
    with pytest.raises(ValueError):
        LBMConfig(force_scheme="ladd")