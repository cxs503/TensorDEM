"""Analytic and integration tests for the Mindlin--Deresiewicz partial-slip law.

Case anchor: **CASE-DEM-MINDLIN-TANGENTIAL** (Mindlin & Deresiewicz 1953,
J. Appl. Mech. 20:327, DOI 10.1115/1.4010702).

Three independent layers are checked:

1. *Closed-form anchors* of :mod:`tensordem.mindlin_partial_slip`:
   zero at no load, ``mu F_n`` at ``delta_s``, initial slope ``k_t = 8 G* a``,
   invertibility, stick-circle radius, reversal branch endpoints/slope.
2. *Cross-implementation* agreement with
   :func:`tensordem.contact_tangential.mindlin_backbone` (a separately written,
   config-driven envelope) -- two code paths, one law.
3. *Solver integration*: :class:`tensordem.dem3d.IceDEM3D` with
   ``tangential_model="mindlin_partial_slip"`` reproduces the analytic curve on
   a real sphere/plane contact, and the default ``cundall_strack`` path is left
   untouched.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem import hertz_mindlin, mindlin_partial_slip
from tensordem.contact_tangential import MindlinTangentialConfig, mindlin_backbone
from tensordem.dem3d import (
    DEM3DConfig,
    IceDEM3D,
    hertz_effective_radius,
    hertz_reduced_modulus,
)

E = 1.0e9          # Pa
NU = 0.3
R = 0.025          # m
MU = 0.3
FN = 1.0           # N

E_STAR_SPHERE = hertz_reduced_modulus(E, NU)                       # two spheres
R_STAR_SPHERE = hertz_effective_radius(R)                          # two spheres
G_STAR_SPHERE = hertz_mindlin.hertz_reduced_shear_modulus(E, NU)
E_STAR_PLANE = hertz_reduced_modulus(E, NU, rigid_counterpart=True)
R_STAR_PLANE = hertz_effective_radius(R, rigid_counterpart=True)
G_STAR_PLANE = hertz_mindlin.hertz_reduced_shear_modulus(E, NU, rigid_counterpart=True)


# ---------------------------------------------------------------------------
# 1. Closed-form anchors
# ---------------------------------------------------------------------------
def test_stiffness_matches_hertz_mindlin_through_the_overlap():
    """k_t = 8 G* a must equal the overlap-parameterised Hertz-Mindlin k_t."""
    delta_n = 2.6510734941148934e-06
    for e_star, g_star, r_eff in ((E_STAR_SPHERE, G_STAR_SPHERE, R_STAR_SPHERE),
                                  (E_STAR_PLANE, G_STAR_PLANE, R_STAR_PLANE)):
        f_n = (4.0 / 3.0) * e_star * math.sqrt(r_eff) * delta_n ** 1.5
        k_from_force = float(mindlin_partial_slip.mindlin_no_slip_stiffness(
            f_n, e_star, g_star, r_eff))
        k_from_overlap = float(hertz_mindlin.hertz_mindlin_tangential_stiffness(
            delta_n, e_star, g_star, r_eff))
        assert k_from_force == pytest.approx(k_from_overlap, rel=1e-12)


def test_gross_slip_displacement_classical_form():
    """delta_s = 3 mu F_n/(2 k_t) equals 3 mu F_n (2-nu)/(8 G a) for equal spheres."""
    a = float(mindlin_partial_slip.hertz_contact_radius(FN, E_STAR_SPHERE, R_STAR_SPHERE))
    k_t = float(mindlin_partial_slip.mindlin_no_slip_stiffness(
        FN, E_STAR_SPHERE, G_STAR_SPHERE, R_STAR_SPHERE))
    d_s = float(mindlin_partial_slip.gross_slip_displacement(FN, MU, k_t))
    g = E / (2.0 * (1.0 + NU))
    classical = 3.0 * MU * FN * (2.0 - NU) / (8.0 * g * a)
    assert d_s == pytest.approx(classical, rel=1e-12)
    # and the generic 3 mu Fn/(16 G* a) form
    assert d_s == pytest.approx(3.0 * MU * FN / (16.0 * G_STAR_SPHERE * a), rel=1e-12)


def test_partial_slip_curve_anchors():
    """F(0) = 0, F(delta_s) = mu F_n, monotone, and slope k_t at the origin."""
    k_t = float(mindlin_partial_slip.mindlin_no_slip_stiffness(
        FN, E_STAR_PLANE, G_STAR_PLANE, R_STAR_PLANE))
    d_s = float(mindlin_partial_slip.gross_slip_displacement(FN, MU, k_t))
    assert float(mindlin_partial_slip.partial_slip_magnitude(0.0, FN, MU, k_t)) == 0.0
    assert float(mindlin_partial_slip.partial_slip_magnitude(d_s, FN, MU, k_t)) == \
        pytest.approx(MU * FN, rel=1e-14)
    # beyond delta_s the force is the Coulomb plateau
    assert float(mindlin_partial_slip.partial_slip_magnitude(
        5.0 * d_s, FN, MU, k_t)) == pytest.approx(MU * FN, rel=1e-14)
    # monotone increasing
    grid = torch.linspace(0.0, 1.2 * d_s, 501, dtype=torch.float64)
    f = mindlin_partial_slip.partial_slip_magnitude(grid, FN, MU, k_t)
    assert bool(torch.all(f[1:] >= f[:-1]))
    # initial slope: central difference at delta_s/1000
    h = d_s * 1e-3
    slope = (float(mindlin_partial_slip.partial_slip_magnitude(2 * h, FN, MU, k_t))
             - float(mindlin_partial_slip.partial_slip_magnitude(0.0, FN, MU, k_t))) / (2 * h)
    assert slope == pytest.approx(k_t, rel=2e-3)


def test_inverse_and_stick_radius():
    """delta(F_t) inverts F_t(delta) and c/a = (1 - F_t/mu F_n)^{1/3}."""
    k_t = float(mindlin_partial_slip.mindlin_no_slip_stiffness(
        FN, E_STAR_PLANE, G_STAR_PLANE, R_STAR_PLANE))
    d_s = float(mindlin_partial_slip.gross_slip_displacement(FN, MU, k_t))
    grid = torch.linspace(0.0, 0.999 * d_s, 257, dtype=torch.float64)
    f = mindlin_partial_slip.partial_slip_magnitude(grid, FN, MU, k_t)
    back = mindlin_partial_slip.inverse_displacement(f, FN, MU, k_t)
    assert float(torch.max(torch.abs(back - grid))) <= 1e-12 * d_s
    assert float(mindlin_partial_slip.stick_radius_ratio(0.0, FN, MU)) == \
        pytest.approx(1.0, rel=1e-15)
    assert float(mindlin_partial_slip.stick_radius_ratio(MU * FN, FN, MU)) == \
        pytest.approx(0.0, abs=1e-15)
    # at half the Coulomb load the stick circle covers 79 % of the patch
    c_over_a = float(mindlin_partial_slip.stick_radius_ratio(0.5 * MU * FN, FN, MU))
    assert c_over_a == pytest.approx(0.5 ** (1.0 / 3.0), rel=1e-12)


def test_reversal_branch_endpoints_and_slope():
    """Reversal branch: initial slope k_t, full reversal at 2 delta_s -> -mu F_n."""
    k_t = float(mindlin_partial_slip.mindlin_no_slip_stiffness(
        FN, E_STAR_PLANE, G_STAR_PLANE, R_STAR_PLANE))
    d_s = float(mindlin_partial_slip.gross_slip_displacement(FN, MU, k_t))
    peak = MU * FN
    h = d_s * 1e-4
    slope = (float(mindlin_partial_slip.reversal_branch(h, peak, FN, MU, k_t))
             - float(mindlin_partial_slip.reversal_branch(0.0, peak, FN, MU, k_t))) / h
    assert slope == pytest.approx(-k_t, rel=2e-3)
    end = float(mindlin_partial_slip.reversal_branch(2.0 * d_s, peak, FN, MU, k_t))
    assert end == pytest.approx(-MU * FN, rel=1e-12)
    area = float(mindlin_partial_slip.reversal_loop_area(FN, MU, k_t, n_quad=8001))
    assert area > 0.0
    # the loop area scales like mu^2 F_n^2 / k_t (dimensional analysis)
    area2 = float(mindlin_partial_slip.reversal_loop_area(
        FN, 2.0 * MU, k_t, n_quad=8001))
    assert area2 / area == pytest.approx(4.0, rel=2e-3)


def test_vector_force_alignment_and_inactive_contact():
    """F_t points along delta; an unloaded contact carries no force and no slip."""
    k_t = float(mindlin_partial_slip.mindlin_no_slip_stiffness(
        FN, E_STAR_PLANE, G_STAR_PLANE, R_STAR_PLANE))
    disp = torch.tensor([[3e-7, 4e-7, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    force, slip = mindlin_partial_slip.partial_slip_force(disp, FN, MU, k_t)
    mag = float(torch.linalg.vector_norm(force[0]))
    assert mag == pytest.approx(float(
        mindlin_partial_slip.partial_slip_magnitude(5e-7, FN, MU, k_t)), rel=1e-15)
    unit = force[0] / mag
    assert float(unit[0]) == pytest.approx(0.6, rel=1e-12)
    assert float(unit[1]) == pytest.approx(0.8, rel=1e-12)
    assert float(torch.linalg.vector_norm(force[1])) == 0.0
    # zero normal load: no force even with a large tangential displacement
    f0, _ = mindlin_partial_slip.partial_slip_force(disp, 0.0, MU, k_t)
    assert float(torch.max(torch.abs(f0))) == 0.0


# ---------------------------------------------------------------------------
# 2. Cross-implementation agreement
# ---------------------------------------------------------------------------
def test_matches_contact_tangential_backbone():
    """Two separately written implementations of the same law agree exactly."""
    k_t = float(mindlin_partial_slip.mindlin_no_slip_stiffness(
        FN, E_STAR_SPHERE, G_STAR_SPHERE, R_STAR_SPHERE))
    cfg = MindlinTangentialConfig(k_t0=k_t, mu_s=MU, backbone="mindlin")
    grid = torch.linspace(0.0, 2.0 * float(
        mindlin_partial_slip.gross_slip_displacement(FN, MU, k_t)), 331,
        dtype=torch.float64)
    mine = mindlin_partial_slip.partial_slip_magnitude(grid, FN, MU, k_t)
    theirs = mindlin_backbone(grid, torch.full_like(grid, FN), cfg)
    assert float(torch.max(torch.abs(mine - theirs))) <= 1e-14 * MU * FN


# ---------------------------------------------------------------------------
# 3. Solver integration
# ---------------------------------------------------------------------------
def _single_contact_config(**overrides):
    kwargs = dict(
        nx=2, ny=2, nz=2, radius=R, density=917.0,
        bond_stiffness=1e-6, fix_x_edges=False, fix_bottom=False,
        contact_model="hertz", young_modulus=E, poisson_ratio=NU,
        friction_coefficient=MU, platen_friction_coefficient=MU,
        bottom_platen_enabled=True, bottom_platen_gap=0.0,
        tangential_model="mindlin_partial_slip",
        restitution_coefficient=1.0, platen_damping=2.0,
    )
    kwargs.update(overrides)
    return DEM3DConfig(**kwargs)


def _run_single_contact(model="mindlin_partial_slip", steps=1600, mu=MU, fn=FN,
                        drive_v=5.0e-3):
    cfg = _single_contact_config(tangential_model=model, friction_coefficient=mu,
                                 platen_friction_coefficient=mu)
    e_star = hertz_reduced_modulus(E, NU, rigid_counterpart=True)
    r_eff = hertz_effective_radius(R, rigid_counterpart=True)
    delta_n = (3.0 * fn / (4.0 * e_star * math.sqrt(r_eff))) ** (2.0 / 3.0)
    sim = IceDEM3D(cfg)
    for idx in range(1, sim.positions.shape[0]):
        sim.positions[idx, 0] += 5.0 * idx
        sim.positions[idx, 1] += 5.0 * idx
    sim.positions[0, 0] = 0.0
    sim.positions[0, 1] = 0.0
    sim.positions[0, 2] = -delta_n
    sim.velocities[:] = 0.0
    ext = torch.zeros_like(sim.positions)
    ext[0, 2] = -fn
    drive_k = 2.0e4
    out = []
    t = 0.0
    for _ in range(steps):
        x = float(sim.positions[0, 0])
        ext[0, 0] = drive_k * (drive_v * t - x)
        sim.step(ext)
        t += sim.dt
        out.append((abs(float(sim._bottom_platen_tang_disp[0, 0])),
                    abs(float(sim.last_bottom_platen_friction[0, 0]))))
    d, f = zip(*out)
    return torch.tensor(d, dtype=torch.float64), torch.tensor(f, dtype=torch.float64)


def test_dem3d_mindlin_branch_reproduces_the_analytic_curve():
    """The solver's F_t(delta) follows the analytic law through both segments."""
    delta, f_t = _run_single_contact()
    k_t = float(mindlin_partial_slip.mindlin_no_slip_stiffness(
        FN, E_STAR_PLANE, G_STAR_PLANE, R_STAR_PLANE))
    d_s = float(mindlin_partial_slip.gross_slip_displacement(FN, MU, k_t))
    ref = mindlin_partial_slip.partial_slip_magnitude(delta, FN, MU, k_t)
    err = float(torch.max(torch.abs(f_t - ref))) / (MU * FN)
    assert err < 1e-9, f"max deviation {err:.3e} of mu F_n"
    # both segments are exercised: micro-slip and the Coulomb plateau
    assert float(delta.max()) > 1.1 * d_s
    assert float(f_t[-1]) == pytest.approx(MU * FN, rel=1e-9)


def test_dem3d_legacy_default_is_cundall_strack():
    """The opt-in flag defaults off, and the legacy branch keeps its linear law."""
    assert DEM3DConfig().tangential_model == "cundall_strack"
    delta, f_t = _run_single_contact(model="cundall_strack", steps=200,
                                     drive_v=1.0e-3)
    k_platen = _single_contact_config().platen_stiffness
    ref = torch.clamp(k_platen * delta, max=MU * FN)
    err = float(torch.max(torch.abs(f_t - ref))) / (MU * FN)
    assert err < 1e-12, f"legacy platen law changed (max {err:.3e} of mu F_n)"


def test_dem3d_rejects_unsupported_combinations():
    with pytest.raises(ValueError):
        _single_contact_config(contact_model="linear")
    with pytest.raises(ValueError):
        _single_contact_config(friction_regularization="dual")
    with pytest.raises(ValueError):
        _single_contact_config(tangential_model="hooke")

def test_reversal_loop_area_absolute_closed_form():
    """The full double-amplitude hysteresis loop area equals 6 (mu F_n)^2 / k_t.

    This is the *absolute-value* verification of :func:`reversal_loop_area`
    (the existing suite only checks the mu^2 F_n^2 / k_t *scaling*).  For a
    full Mindlin--Deresiewicz cycle (virgin loading 0 -> delta_s, then a full
    reversal to -delta_s), the area enclosed by the loading branch and the
    reversal branch is exactly

        W_loop = 6 (mu F_n)^2 / k_t

    (derived by direct quadrature of the two closed-form branches; the factor
    6 is the dimensionless loop-area coefficient).  This is the frictional
    energy dissipated per full cycle -- the thermodynamic consistency anchor
    of the partial-slip law.
    """
    for mu, fn in [(0.3, 1.0), (0.5, 2.0), (0.1, 0.5), (0.7, 3.0)]:
        k_t = float(mindlin_partial_slip.mindlin_no_slip_stiffness(
            fn, E_STAR_PLANE, G_STAR_PLANE, R_STAR_PLANE))
        area = float(mindlin_partial_slip.reversal_loop_area(
            fn, mu, k_t, n_quad=200001))
        closed = 6.0 * (mu * fn) ** 2 / k_t
        assert area == pytest.approx(closed, rel=1e-6), \
            f"mu={mu} Fn={fn}: loop area {area:.6e} != 6(mu Fn)^2/k_t {closed:.6e}"
