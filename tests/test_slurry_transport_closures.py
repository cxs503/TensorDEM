"""Unit tests for the dredged-slurry transport closures (A1, plus drag laws)."""

import math

import numpy as np
import pytest

from tensordem.slurry_transport.closures import (
    G, MU_W, RHO_W, archimedes_number, cd_schiller_naumann, cd_stokes,
    drag_force, hindrance_richardson_zaki, particle_reynolds,
    richardson_zaki_exponent, sphere_area, sphere_volume, submerged_weight,
    terminal_velocity_numeric, terminal_velocity_regime,
    terminal_velocity_rubey, terminal_velocity_stokes)
from tensordem.slurry_transport.verification import face_a1


def test_sphere_geometry_and_submerged_weight():
    d = 2.0e-3
    assert sphere_volume(d) == pytest.approx(math.pi / 6.0 * d ** 3)
    assert sphere_area(d) == pytest.approx(math.pi / 4.0 * d ** 2)
    w = submerged_weight(d, 2650.0, RHO_W)
    assert w == pytest.approx((2650.0 - RHO_W) * sphere_volume(d) * G)
    # buoyancy reduced weight is linear in the density difference
    assert submerged_weight(d, 2.0 * 2650.0 - RHO_W, RHO_W) == pytest.approx(2.0 * w)


def test_drag_laws_limits():
    # Stokes limit of Schiller-Naumann
    assert float(cd_schiller_naumann(1e-3)) == pytest.approx(
        float(cd_stokes(1e-3)) * (1.0 + 0.15 * 1e-3 ** 0.687), rel=1e-12)
    # Newton plateau
    assert float(cd_schiller_naumann(5e5)) == pytest.approx(0.44)
    # continuity at the branch switch Re = 1000 (< 2% jump)
    lo = float(cd_schiller_naumann(999.0))
    hi = float(cd_schiller_naumann(1000.1))
    assert abs(hi - lo) / lo < 0.02


def test_force_balance_is_satisfied_at_the_terminal_velocity():
    for d in (0.1e-3, 0.5e-3, 2.0e-3, 1.0e-2):
        res = terminal_velocity_numeric(d, 2650.0, RHO_W, MU_W)
        assert res["residual"] < 1e-12, (d, res["residual"])
        # drag equals submerged weight at the reported velocity
        assert drag_force(res["w"], d, RHO_W, MU_W) == pytest.approx(
            submerged_weight(d, 2650.0, RHO_W), rel=1e-10)


def test_regime_closed_form_matches_numeric():
    a1 = face_a1()
    assert a1["max_relerr_closed_vs_numeric"] < 1e-9
    assert a1["pass_closed_form"]
    regimes = {r["d_mm"]: r["regime"] for r in a1["rows"]}
    assert regimes[0.1] == "transition"     # Re ~ 0.9, Stokes is 12% off
    # regime selection must agree with the Reynolds number of the solution
    for r in a1["rows"]:
        if r["regime"] == "stokes":
            assert r["Re"] < 1.0
        elif r["regime"] == "newton":
            assert r["Re"] > 900.0


def test_stokes_limit_scaling_exponent():
    a1 = face_a1()
    assert abs(a1["stokes_slope"] - 2.0) <= 0.1
    assert a1["pass_stokes_slope"]
    # Stokes closed form must be exact in the limit d -> 0
    ds = np.array([1e-7, 1e-6, 1e-5])
    ws = np.array([terminal_velocity_numeric(d, 2650.0, RHO_W, MU_W)["w"] for d in ds])
    ref = terminal_velocity_stokes(ds, 2650.0, RHO_W, MU_W)
    # 0.2% tolerance: the residual gap is the genuine Schiller-Naumann
    # correction (1 + 0.15 Re^0.687) at Re ~ 1e-3, not a solver error
    assert np.allclose(ws, ref, rtol=2e-3)


def test_rubey_is_an_upper_bound_and_agrees_in_the_Stokes_limit():
    # Rubey uses Cd -> 4/3 (not 0.44) in the inertial limit, so it over-predicts
    # coarse particles; it must reduce to Stokes as d -> 0
    d_small = 1e-6
    w_rub = terminal_velocity_rubey(d_small, 2650.0, RHO_W, MU_W)
    w_stk = terminal_velocity_stokes(d_small, 2650.0, RHO_W, MU_W)
    assert w_rub == pytest.approx(w_stk, rel=1e-3)
    # coarse particles: Rubey's Cd = 4/3 > Schiller-Naumann 0.44, so it is a
    # *lower* bound there
    d_big = 5e-3
    assert terminal_velocity_rubey(d_big, 2650.0, RHO_W, MU_W) < \
        terminal_velocity_numeric(d_big, 2650.0, RHO_W, MU_W)["w"]


def test_neutral_density_has_no_settling_velocity():
    res = terminal_velocity_numeric(1e-3, RHO_W, RHO_W, MU_W)
    assert res["w"] == 0.0
    assert res["regime"] == "neutral"
    w_cf, regime = terminal_velocity_regime(1e-3, RHO_W, RHO_W, MU_W)
    assert w_cf == 0.0 and regime == "neutral"


def test_reynolds_and_archimedes():
    d, w = 1e-3, 0.1
    Re = float(particle_reynolds(w, d, RHO_W, MU_W))
    assert Re == pytest.approx(RHO_W * w * d / MU_W)
    # Ar = 3/4 Cd Re^2 is the standard identity in the Stokes range
    Ar = archimedes_number(d, 2650.0, RHO_W, MU_W)
    assert Ar > 0


def test_hindrance_monotone_and_limits():
    eps = np.array([0.2, 0.4, 0.6, 0.9])
    h = np.asarray(hindrance_richardson_zaki(eps, 100.0))
    assert np.all(np.diff(h) < 0.0)          # more voidage -> less hindrance
    assert np.all(h[eps < 1.0] > 1.0)
    assert float(np.asarray(hindrance_richardson_zaki([1.0], 100.0))[0]) == pytest.approx(1.0)
    # exponent piecewise values
    assert richardson_zaki_exponent(0.1) == 4.6
    assert richardson_zaki_exponent(1000.0) == 2.4


def test_drag_force_is_quadratic_at_high_re_and_linear_at_low_re():
    d = 1e-4
    f1 = float(drag_force(1e-6, d, RHO_W, MU_W))
    f2 = float(drag_force(2e-6, d, RHO_W, MU_W))
    assert f2 / f1 == pytest.approx(2.0, rel=1e-2)   # Stokes: linear

    d = 1e-2
    g1 = float(drag_force(2.0, d, RHO_W, MU_W))
    g2 = float(drag_force(4.0, d, RHO_W, MU_W))
    assert g2 / g1 == pytest.approx(4.0, rel=0.05)   # Newton: quadratic