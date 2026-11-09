"""Analytic anchors for the dense particle--fluid drag closures.

Each test pins a closed form or a limiting case that the closure must reproduce
exactly; none of them depends on the bed-scale assembly (that is
``test_bedcoupling_bed.py``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.bedcoupling.closures import (
    GIDASPOW_SWITCH,
    buoyant_weight,
    drag_coefficient,
    drag_force,
    ergun_pressure_gradient,
    gidaspow_branch_jump,
    hindered_slip_velocity,
    hydrostatic_pressure_gradient,
    pressure_gradient_force,
    rz_exponent,
    rz_slip_velocity,
    rz_superficial_velocity,
    sphere_volume,
    terminal_velocity_sn,
    terminal_velocity_stokes,
    wen_yu_cd,
)

D = 1.0e-3
MU = 1.0e-3
RHO_F = 998.0
RHO_P = 2650.0


def test_wen_yu_cd_stokes_limit() -> None:
    # C_D -> 24/Re as Re -> 0
    # 24/Re plus the O(Re^0.687) correction of the correlation
    assert float(wen_yu_cd(1e-8)) == pytest.approx(24.0e8, rel=1e-5)


def test_wen_yu_stokes_coefficient_has_eps_minus_2_65() -> None:
    """Stokes limit of the Wen--Yu branch: ``F = 3 pi mu d u eps^-2.65``."""
    for eps in (0.6, 0.8, 0.95):
        c_zero = float(
            drag_coefficient(0.0, diameter=D, porosity=eps, rho_fluid=RHO_F, mu=MU, branch="wen_yu")
        )
        assert c_zero == pytest.approx(3.0 * math.pi * MU * D * eps**-2.65, rel=1e-15)
        c_small = float(
            drag_coefficient(1e-9, diameter=D, porosity=eps, rho_fluid=RHO_F, mu=MU, branch="wen_yu")
        )
        assert c_small == pytest.approx(c_zero, rel=1e-4)


def test_wen_yu_single_particle_limit() -> None:
    """At ``eps -> 1`` the drag must collapse to the single-sphere value."""
    u = 0.05
    c = float(drag_coefficient(u, diameter=D, porosity=1.0, rho_fluid=RHO_F, mu=MU, branch="wen_yu"))
    re = RHO_F * u * D / MU
    expected = 0.5 * RHO_F * float(wen_yu_cd(re)) * u * u * math.pi * D**2 / 4.0
    assert c * u == pytest.approx(expected, rel=1e-12)


def test_ergun_viscous_and_inertial_terms() -> None:
    u, eps = 0.01, 0.4
    drop = ergun_pressure_gradient(u, diameter=D, porosity=eps, rho_fluid=RHO_F, mu=MU)
    assert drop.viscous == pytest.approx(150.0 * MU * (1 - eps) ** 2 / (eps**3 * D**2) * u)
    assert drop.inertial == pytest.approx(1.75 * RHO_F * (1 - eps) / (eps**3 * D) * u * u)
    assert drop.total == pytest.approx(drop.viscous + drop.inertial)


def test_ergun_regime_labels() -> None:
    low = ergun_pressure_gradient(1e-4, diameter=D, porosity=0.38, rho_fluid=RHO_F, mu=MU)
    high = ergun_pressure_gradient(1.0, diameter=D, porosity=0.38, rho_fluid=RHO_F, mu=MU)
    assert low.regime == "laminar"
    assert high.regime == "inertial"


def test_gidaspow_switches_branch_at_eighty_percent() -> None:
    u = 0.05
    below = float(drag_coefficient(u, diameter=D, porosity=GIDASPOW_SWITCH - 1e-9, rho_fluid=RHO_F, mu=MU))
    above = float(drag_coefficient(u, diameter=D, porosity=GIDASPOW_SWITCH + 1e-9, rho_fluid=RHO_F, mu=MU))
    ergun = float(drag_coefficient(u, diameter=D, porosity=GIDASPOW_SWITCH, rho_fluid=RHO_F, mu=MU, branch="ergun"))
    wen = float(drag_coefficient(u, diameter=D, porosity=GIDASPOW_SWITCH, rho_fluid=RHO_F, mu=MU, branch="wen_yu"))
    assert below == pytest.approx(ergun)
    assert above == pytest.approx(wen, rel=1e-6)


def test_gidaspow_branch_jump_is_measured_not_hidden() -> None:
    jump = gidaspow_branch_jump(0.05, diameter=D, rho_fluid=RHO_F, mu=MU)
    # the two branches are known to be discontinuous; the jump must be reported
    assert jump["relative_jump"] < -0.5
    assert jump["coefficient_ergun"] > jump["coefficient_wen_yu"] > 0.0


def test_pressure_gradient_force_is_buoyancy() -> None:
    vol = np.array([sphere_volume(D)])
    grad_p = hydrostatic_pressure_gradient(RHO_F)
    f = pressure_gradient_force(vol, grad_p)
    assert f[0, 2] == pytest.approx(vol[0] * RHO_F * 9.80665)
    assert f[0, 0] == 0.0 and f[0, 1] == 0.0


def test_buoyant_weight_matches_density_contrast() -> None:
    assert buoyant_weight(D, rho_particle=RHO_P, rho_fluid=RHO_F) == pytest.approx(
        sphere_volume(D) * (RHO_P - RHO_F) * 9.80665
    )


def test_terminal_velocity_balances_schiller_naumann() -> None:
    res = terminal_velocity_sn(D, rho_particle=RHO_P, rho_fluid=RHO_F, mu=MU)
    u = res["u_terminal"]
    re = RHO_F * u * D / MU
    drag = 0.5 * RHO_F * float(wen_yu_cd(re)) * u * u * math.pi * D**2 / 4.0
    assert drag == pytest.approx(res["buoyant_weight"], rel=1e-12)
    assert res["re_terminal"] == pytest.approx(re, rel=1e-12)


def test_terminal_velocity_reduces_to_stokes_at_low_re() -> None:
    # very viscous fluid, tiny particle -> Stokes regime
    d, mu = 1e-4, 0.5
    res = terminal_velocity_sn(d, rho_particle=RHO_P, rho_fluid=RHO_F, mu=mu)
    stokes = terminal_velocity_stokes(d, rho_particle=RHO_P, rho_fluid=RHO_F, mu=mu)
    assert res["re_terminal"] < 1e-3
    assert res["u_terminal"] == pytest.approx(stokes, rel=1e-4)


def test_rz_exponent_piecewise_values() -> None:
    assert float(rz_exponent(0.1)) == pytest.approx(4.65)
    assert float(rz_exponent(500.0)) == pytest.approx(2.4)
    assert float(rz_exponent(1.0)) == pytest.approx(4.4)
    # monotonically decreasing with Re
    res = rz_exponent(np.array([0.1, 0.5, 5.0, 50.0, 5000.0]))
    assert np.all(np.diff(res) < 0.0)


def test_rz_velocity_laws_are_consistent() -> None:
    eps, ut, n = 0.8, 0.01, 4.65
    assert float(rz_superficial_velocity(eps, ut, n)) == pytest.approx(ut * eps**n)
    assert float(rz_slip_velocity(eps, ut, n)) == pytest.approx(ut * eps ** (n - 1.0))


def test_hindered_slip_velocity_balances_the_weight() -> None:
    eps = np.array([0.8, 0.9, 1.0])
    u = hindered_slip_velocity(eps, diameter=D, rho_particle=RHO_P, rho_fluid=RHO_F, mu=MU)
    weight = buoyant_weight(D, rho_particle=RHO_P, rho_fluid=RHO_F)
    for e, ui in zip(eps, u):
        c = float(drag_coefficient(ui, diameter=D, porosity=e, rho_fluid=RHO_F, mu=MU))
        assert c * ui == pytest.approx(weight, rel=1e-9)


def test_hindered_slip_recovers_terminal_velocity_at_eps_one() -> None:
    res = terminal_velocity_sn(D, rho_particle=RHO_P, rho_fluid=RHO_F, mu=MU)
    u = hindered_slip_velocity(
        np.array([1.0]), diameter=D, rho_particle=RHO_P, rho_fluid=RHO_F, mu=MU
    )
    assert float(u[0]) == pytest.approx(res["u_terminal"], rel=1e-6)


def test_drag_force_is_antiparallel_to_and_linear_in_slip() -> None:
    u = np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.02]])
    eps = np.array([0.4, 0.9])
    f = drag_force(u, diameter=D, porosity=eps, rho_fluid=RHO_F, mu=MU)
    assert np.all(np.sign(f[:, 2]) == np.sign(u[:, 2]))
    for k in range(2):
        c = float(drag_coefficient(abs(u[k, 2]), diameter=D, porosity=eps[k], rho_fluid=RHO_F, mu=MU))
        assert f[k, 2] == pytest.approx(c * u[k, 2], rel=1e-14)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        ergun_pressure_gradient(0.1, diameter=D, porosity=1.2, rho_fluid=RHO_F, mu=MU)
    with pytest.raises(ValueError):
        drag_coefficient(0.1, diameter=D, porosity=-0.1, rho_fluid=RHO_F, mu=MU)
    with pytest.raises(ValueError):
        drag_coefficient(0.1, diameter=D, porosity=0.5, rho_fluid=RHO_F, mu=MU, branch="nope")
    with pytest.raises(ValueError):
        rz_exponent(0.0)
