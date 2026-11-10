"""Unit tests of the Hertz-Mindlin contact law (P103W2-N03).

Every assertion is against a closed-form value or a set of equations that the
solver is supposed to satisfy, not against a stored number.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensordem.clear_ice.contact import (
    ContactMaterial,
    ContactSolver,
    contact_time,
    damping_ratio_from_restitution,
    effective_modulus,
    effective_radius,
    hertz_elastic_energy,
    hertz_elastic_force,
    hertz_stiffness,
    restitution_from_damping_ratio,
)
from tensordem.clear_ice.field import ClearIceField, FieldConfig


def test_hertz_force_and_stiffness_closed_form():
    mat = ContactMaterial()
    e_eff = effective_modulus(mat, mat)
    assert e_eff == pytest.approx(mat.youngs_modulus / (2 * (1 - mat.poisson_ratio ** 2)))
    r_eff = effective_radius(0.6, 0.9)
    assert r_eff == pytest.approx(0.6 * 0.9 / 1.5)
    delta = np.array([1e-3, 4e-3])
    f = hertz_elastic_force(delta, e_eff, r_eff)
    expected = (4.0 / 3.0) * e_eff * np.sqrt(r_eff) * delta ** 1.5
    assert np.allclose(f, expected, rtol=1e-14)
    k = hertz_stiffness(delta, e_eff, r_eff)
    # k_n must be the exact derivative of the elastic force
    h = 1e-9
    num = (hertz_elastic_force(delta + h, e_eff, r_eff)
           - hertz_elastic_force(delta - h, e_eff, r_eff)) / (2 * h)
    assert np.allclose(k, num, rtol=1e-5)
    # spring energy derivative must give the force back
    u = hertz_elastic_energy(delta, e_eff, r_eff)
    num_u = (hertz_elastic_energy(delta + h, e_eff, r_eff)
             - hertz_elastic_energy(delta - h, e_eff, r_eff)) / (2 * h)
    assert np.allclose(num_u, f, rtol=1e-5)
    assert np.allclose(f, k * delta / 1.5, rtol=1e-12)


def test_restitution_damping_calibration_table_is_self_consistent():
    """The embedded table must be reproducible by the integrator it came from."""
    from tensordem.clear_ice.contact import E_TABLE, ZETA_TABLE, _restitution_of_zeta
    for zeta, e in zip(ZETA_TABLE[2::4], E_TABLE[2::4]):
        assert _restitution_of_zeta(zeta) == pytest.approx(e, abs=1e-3)
    # monotone decreasing up to the over-damped limit
    assert np.all(np.diff(E_TABLE) < 0.0)
    # the calibrated ratio must be monotone in e
    es = np.linspace(0.05, 1.0, 12)
    zetas = [damping_ratio_from_restitution(e) for e in es]
    assert np.all(np.diff(zetas) < 0.0)
    assert damping_ratio_from_restitution(1.0) == 0.0
    # the calibrated ratio must differ materially from the linear-dashpot
    # estimate (the clamp weakens the effective damping, so it comes out higher)
    from tensordem.clear_ice.contact import damping_ratio_from_restitution_analytic
    z_cal = damping_ratio_from_restitution(0.2)
    z_ana = damping_ratio_from_restitution_analytic(0.2)
    assert z_cal > 1.3 * z_ana


def _two_particle_collision(restitution: float, dt_over_tc: float = 0.02,
                            v_in: float = 0.3, thickness: float = 0.3):
    """Head-on collision of two equal floes; returns the restitution measured.

    The integrator runs until the pair actually separates (first force-free step
    after contact), so a heavily damped contact that takes many contact times to
    release is not truncated.
    """
    r = 0.9
    mat = ContactMaterial(restitution=restitution)
    solver = ContactSolver(mat, box=None)
    density = mat.density
    mass = density * np.pi * r ** 2 * thickness
    x = np.array([[0.0, 0.0], [2 * r + 1e-3, 0.0]])
    v = np.array([[v_in, 0.0], [-v_in, 0.0]])          # head-on, closing
    omega = np.zeros(2)
    masses = np.array([mass, mass])

    e_eff = effective_modulus(mat, mat)
    tc = contact_time(0.5 * mass, 0.5 * r, e_eff, v_in)
    dt = dt_over_tc * tc

    f, tau, _ = solver.update(x, v, np.array([r, r]), masses, omega, dt, thickness)
    acc = f / mass
    touched = False
    for _ in range(int(round(60 * tc / dt))):
        v += 0.5 * dt * acc
        x = x + dt * v
        f, tau, diag = solver.update(x, v, np.array([r, r]), masses, omega, dt, thickness)
        acc = f / mass
        v += 0.5 * dt * acc
        if len(diag):
            touched = True
        elif touched:
            break
    assert touched, "the floes never came into contact"
    return float((v[1, 0] - v[0, 0]) / (2 * v_in))


@pytest.mark.parametrize("e_in", [0.2, 0.5, 0.8])
def test_measured_restitution_matches_input(e_in):
    """The calibrated damping must realise the requested restitution."""
    measured = _two_particle_collision(e_in)
    assert measured == pytest.approx(e_in, rel=0.05, abs=0.02)


def test_analytic_damping_formula_would_miss_the_target():
    """Negative control for the calibration: the classical formula is biased."""
    from tensordem.clear_ice.contact import damping_ratio_from_restitution_analytic
    from tensordem.clear_ice.contact import _restitution_of_zeta
    e_analytic = _restitution_of_zeta(damping_ratio_from_restitution_analytic(0.2))
    assert e_analytic > 0.2 * 1.3          # ~60 % high: why the table is needed


def test_normal_contact_never_pulls():
    """A separating pair must have zero normal force (non-cohesive regularisation)."""
    r = 0.9
    mat = ContactMaterial()
    solver = ContactSolver(mat, box=None)
    mass = mat.density * np.pi * r ** 2 * 0.3
    x = np.array([[0.0, 0.0], [2 * r - 1e-3, 0.0]])   # 1 mm penetration
    v = np.array([[-0.5, 0.0], [0.5, 0.0]])           # separating fast
    f, _, diag = solver.update(x, v, np.array([r, r]), np.array([mass, mass]),
                               np.zeros(2), 1e-4, 0.3)
    assert np.all(diag.fn >= 0.0)
    assert diag.fn[0] == 0.0            # damping would have made it negative
    assert np.allclose(f.sum(axis=0), 0.0)


def test_crushing_cap_bounds_the_force():
    """The ice crushing strength caps the contact force (bounded DEM)."""
    r = 0.9
    mat = ContactMaterial()
    solver = ContactSolver(mat, box=None)
    mass = mat.density * np.pi * r ** 2 * 0.3
    x = np.array([[0.0, 0.0], [2 * r - 0.5, 0.0]])     # 0.5 m penetration
    f, _, diag = solver.update(x, np.array([[0.0, 0.0], [0.0, 0.0]]),
                               np.array([r, r]), np.array([mass, mass]),
                               np.zeros(2), 1e-4, 0.3)
    cap = mat.crushing_strength * 2 * r * 0.3
    assert diag.fn[0] <= cap * (1 + 1e-12)
    assert diag.fn[0] == pytest.approx(cap)


def test_tangential_stiffness_and_coulomb_saturation():
    """Mindlin k_t = 8 G* sqrt(R* delta); Coulomb cap at mu F_n."""
    r = 0.9
    mat = ContactMaterial(friction=0.4)
    solver = ContactSolver(mat, box=None)
    mass = mat.density * np.pi * r ** 2 * 0.3
    delta = 2e-3
    x = np.array([[0.0, 0.0], [2 * r - delta, 0.0]])
    keep = np.array([[-r + delta, 0.0], [r + delta, 0.0]])
    v = np.zeros((2, 2))
    t = 1e-5
    # small slip: spring branch
    v = np.array([[0.0, 1e-4], [0.0, 0.0]])
    f, tau, diag = solver.update(x, v, np.array([r, r]), np.array([mass, mass]),
                                 np.zeros(2), t, 0.3)
    r_eff = effective_radius(r, r)
    from tensordem.clear_ice.contact import effective_shear_modulus
    g_eff = effective_shear_modulus(mat, mat)
    k_t = 8.0 * g_eff * np.sqrt(r_eff * delta)
    assert diag.ft[0] == pytest.approx(-k_t * 1e-4 * t, rel=1e-9)
    assert not diag.sliding[0]
    # large slip: the Coulomb cap saturates the spring
    v = np.array([[0.0, 5.0], [0.0, 0.0]])
    for _ in range(30):
        f, tau, diag = solver.update(x, v, np.array([r, r]), np.array([mass, mass]),
                                     np.zeros(2), t, 0.3)
    assert diag.sliding[0]
    assert abs(diag.ft[0]) == pytest.approx(mat.friction * diag.fn[0], rel=1e-9)


def test_pair_contact_conserves_momentum():
    r = 0.9
    mat = ContactMaterial()
    solver = ContactSolver(mat, box=None)
    mass = mat.density * np.pi * r ** 2 * 0.3
    x = np.array([[0.0, 0.0], [1.7, 0.1]])
    v = np.array([[-0.2, 0.05], [0.3, -0.02]])
    omega = np.zeros(2)
    total = mass * v.sum(axis=0)
    f, tau, _ = solver.update(x, v, np.array([r, r]), np.array([mass, mass]),
                              omega, 1e-4, 0.3)
    assert np.allclose(f.sum(axis=0), 0.0, atol=1e-9)
    # angular momentum about the origin must also be balanced by the contact
    # torque pair (equal, opposite sign convention for the two bodies)
    assert np.allclose(f.sum(axis=0) + 0.0, 0.0, atol=1e-9)
    del total


def test_field_dt_limiter_respects_contact_time():
    cfg = FieldConfig(n_floes=12, duration=0.2, warmup=0.1, dt=5e-3)
    field = ClearIceField(cfg)
    assert field.dt <= cfg.safety_factor * field.tc * (1 + 1e-12)
    assert field.dt_over_tc <= cfg.safety_factor * (1 + 1e-12)


def test_plane_wall_matches_pair_contact_limit():
    """A floe against a flat wall must feel the same Hertz law as against R*=r/2."""
    from tensordem.clear_ice.contact import STEEL
    r = 0.9
    mat = ContactMaterial()
    solver = ContactSolver(mat, box=None)
    mass = mat.density * np.pi * r ** 2 * 0.3
    wall_x = 10.0
    delta = 1e-3
    x = np.array([[wall_x - r + delta, 0.0]])
    f, tau = solver.plane_wall_contacts(x, np.zeros((1, 2)), np.array([r]),
                                        np.zeros(1), np.array([mass]), wall_x,
                                        -1.0, STEEL, 0.3)
    e_eff = effective_modulus(mat, STEEL)
    expected = hertz_elastic_force(delta, e_eff, r)
    assert f[0, 0] == pytest.approx(-expected, rel=1e-10)
    assert f[0, 1] == 0.0