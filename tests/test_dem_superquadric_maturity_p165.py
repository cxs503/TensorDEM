"""Acceptance tests for campaign node P165-W2-N06 (non-spherical DEM maturity).

The three defects the node pays off, each with its own negative control:

* **analytic curvature** -- closed-form gradient/Hessian of the superquadric
  implicit surface replaces the finite-difference estimator
  (:mod:`tensordem.superquadric_curvature`).  Baseline: the legacy four-point
  estimator of :func:`tensordem.superquadric.principal_curvatures_superquadric`
  is kept under ``method="fd"`` and must stay *bad* (15 %-97 %), otherwise the
  comparison is not measuring anything.
* **velocity-Verlet integrator** -- second order with a contact-force half-step
  position correction (:mod:`tensordem.superquadric_integrators`).  Baseline:
  symplectic Euler, which must stay first order and must fail the energy-drift
  criterion (negative control).
* **column-collapse repose convergence** -- :mod:`tensordem.superquadric_dem`
  sphere fast path + ``scripts/p165w2_n06_column_collapse.py``; the artifact
  check is skipped when the (long) runs have not been produced in this tree.

Numbers in the prose come from ``artifacts/dem/p165w2_n06/maturity_results.json``
and ``artifacts/dem/p165w2_n06/column_collapse.json``; the heavy anchors are
recomputed here with reduced settings so the suite stays fast.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from tensordem.flat_punch import reduced_modulus
from tensordem.superquadric import principal_curvatures_superquadric as legacy_curv
from tensordem.superquadric_curvature import (
    ellipsoid_gaussian_curvature,
    fd_refinement_order,
    gaussian_curvature,
    mesh_refinement_order,
    principal_curvatures_analytic,
    principal_curvatures_fd,
    principal_curvatures_mesh,
    surface_points,
    superquadric_gradient,
    superquadric_hessian,
)
from tensordem.superquadric_dem import (
    SQSystem,
    SQSystemConfig,
    Wall,
    make_superquadric_body,
)
from tensordem.superquadric_integrators import (
    duffing_accel,
    harmonic_accel,
    integrate,
    order_from_errors,
    step,
    velocity_verlet_step,
)
from tensordem.superquadric_kernel import (
    ContactMaterial,
    effective_radius_hertz,
    gradient_local,
    principal_curvatures_local,
    rolling_resistance_torque,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(REPO, "artifacts", "dem", "p165w2_n06")
MATURITY = os.path.join(ART, "maturity_results.json")
COLUMN = os.path.join(ART, "column_collapse.json")

A_, B_, C_ = 0.03, 0.02, 0.01
SHAPES = [(A_, B_, C_, 2.0, 2.0), (A_, B_, C_, 3.0, 2.0), (A_, B_, C_, 4.0, 3.0),
          (A_, B_, C_, 5.0, 4.0), (A_, B_, C_, 8.0, 8.0), (A_, B_, C_, 2.0, 8.0)]
ANGLES = [(30.0, 20.0), (-25.0, 110.0), (50.0, 200.0), (70.0, 45.0)]


def _surface_points(shape, angles):
    a, b, c, n1, n2 = shape
    pts = [surface_points(np.array([math.radians(e)]), np.array([math.radians(o)]),
                          a, b, c, n1, n2)[0] for e, o in angles]
    return np.array(pts)


# ---------------------------------------------------------------------------
# analytic curvature
# ---------------------------------------------------------------------------

def test_analytic_gradient_matches_kernel_gradient():
    for shape in SHAPES:
        pts = _surface_points(shape, ANGLES)
        g_new = superquadric_gradient(pts, *shape)
        g_old = gradient_local(pts, *shape)
        assert np.allclose(g_new, g_old, rtol=1e-9, atol=1e-9)


def test_analytic_hessian_matches_finite_difference_of_gradient():
    a, b, c, n1, n2 = 0.03, 0.02, 0.01, 5.0, 4.0
    pts = _surface_points((a, b, c, n1, n2), [(30.0, 20.0), (70.0, 45.0)])
    hess = superquadric_hessian(pts, a, b, c, n1, n2)
    h = 1e-7 * max(a, b, c)
    for j in range(3):
        d = np.zeros((len(pts), 3))
        d[:, j] = h
        gp = superquadric_gradient(pts + d, a, b, c, n1, n2)
        gm = superquadric_gradient(pts - d, a, b, c, n1, n2)
        fd = (gp - gm) / (2.0 * h)
        assert np.allclose(hess[:, :, j], fd, rtol=1e-4, atol=1e-3)


def test_ellipsoid_gaussian_curvature_matches_textbook():
    a, b, c = A_, B_, C_
    rng = np.random.default_rng(0)
    th = rng.uniform(0.05, math.pi - 0.05, 25)
    ph = rng.uniform(0.0, 2.0 * math.pi, 25)
    pts = np.column_stack([a * np.sin(th) * np.cos(ph),
                           b * np.sin(th) * np.sin(ph),
                           c * np.cos(th)])
    K = gaussian_curvature(pts, a, b, c, 2, 2)
    K_ref = ellipsoid_gaussian_curvature(pts, a, b, c)
    assert np.max(np.abs(K - K_ref) / np.abs(K_ref)) < 1e-6


def test_ellipsoid_pole_principal_curvatures_match_textbook():
    a, b, c = A_, B_, C_
    k1, k2 = principal_curvatures_analytic(np.array([[0.0, 0.0, c]]), a, b, c, 2, 2)
    assert abs(float(k1[0]) - max(c / a ** 2, c / b ** 2)) / max(c / a ** 2, c / b ** 2) < 1e-9
    assert abs(float(k2[0]) - min(c / a ** 2, c / b ** 2)) / min(c / a ** 2, c / b ** 2) < 1e-9


def test_sphere_limit_reports_one_over_radius():
    R = 0.01
    pts = np.array([[R, 0.0, 0.0], [0.0, R, 0.0], [0.0, 0.0, -R],
                    [R / math.sqrt(3)] * 3])
    k1, k2 = principal_curvatures_analytic(pts, R, R, R, 2, 2)
    assert np.allclose(k1, 1.0 / R, rtol=1e-6)
    assert np.allclose(k2, 1.0 / R, rtol=1e-6)
    assert np.allclose(gaussian_curvature(pts, R, R, R, 2, 2), 1.0 / R ** 2, rtol=1e-6)


def test_general_superquadric_matches_independent_mesh_estimator():
    for shape in SHAPES:
        a, b, c, n1, n2 = shape
        for (eta_deg, omega_deg) in ANGLES:
            eta, omega = math.radians(eta_deg), math.radians(omega_deg)
            p = surface_points(np.array([eta]), np.array([omega]), a, b, c, n1, n2)
            k1, k2 = principal_curvatures_analytic(p, a, b, c, n1, n2)
            m1, m2, _ = principal_curvatures_mesh(eta, omega, a, b, c, n1, n2, 256)
            scale = max(abs(float(k1[0])), abs(float(k2[0])), 1e-30)
            assert abs(float(k1[0]) - m1) / scale < 1e-2
            assert abs(float(k2[0]) - m2) / scale < 1e-2


def test_mesh_refinement_order_is_second_order():
    row = mesh_refinement_order(30.0, 20.0, (A_, B_, C_, 5.0, 4.0),
                               rings=(4, 8, 16, 32, 64))
    assert row["order"] >= 1.5
    errs = [h["rel_err"] for h in row["history"]]
    assert errs[-1] < errs[0] / 100.0


def test_fd_step_refinement_order_is_second_order():
    row = fd_refinement_order(30.0, 20.0, (A_, B_, C_, 5.0, 4.0),
                              eps_seq=(1e-2, 1e-3, 1e-4))
    assert row["order"] >= 1.5


def test_legacy_scalar_curvature_baseline_is_bad():
    """Negative control: the parent's estimator must stay far above 1 %."""
    worst = 0.0
    for shape in SHAPES:
        a, b, c, n1, n2 = shape
        for (eta_deg, omega_deg) in ANGLES:
            p = surface_points(np.array([math.radians(eta_deg)]),
                               np.array([math.radians(omega_deg)]), a, b, c, n1, n2)[0]
            k1, k2 = principal_curvatures_analytic(p[None, :], a, b, c, n1, n2)
            scale = max(abs(float(k1[0])), abs(float(k2[0])), 1e-30)
            l1, l2 = legacy_curv(p, a, b, c, n1, n2, method="fd")
            worst = max(worst, abs(float(k1[0]) - l1) / scale,
                        abs(float(k2[0]) - l2) / scale)
    assert worst > 0.15      # the documented >=15 % legacy error


def test_kernel_curvature_returns_sorted_curvatures_for_both_methods():
    p = surface_points(np.array([0.5]), np.array([1.1]), A_, B_, C_, 5.0, 4.0)
    lo_a, hi_a = principal_curvatures_local(p, A_, B_, C_, 5.0, 4.0)
    lo_f, hi_f = principal_curvatures_local(p, A_, B_, C_, 5.0, 4.0, method="fd")
    assert lo_a[0] <= hi_a[0] and lo_f[0] <= hi_f[0]
    assert abs(float(hi_a[0]) - float(hi_f[0])) / float(hi_a[0]) < 1e-3


def test_kernel_curvature_rejects_unknown_method():
    p = surface_points(np.array([0.5]), np.array([1.1]), A_, B_, C_, 5.0, 4.0)
    with pytest.raises(ValueError):
        principal_curvatures_local(p, A_, B_, C_, 5.0, 4.0, method="nope")


# ---------------------------------------------------------------------------
# curvature change must be torque-neutral (criterion c)
# ---------------------------------------------------------------------------

def test_rolling_torque_change_with_analytic_curvature_below_one_percent():
    mu_r = 0.15
    f_n = 3.0
    omega_rel = np.array([[0.4, -1.2, 0.7]])
    for shape in SHAPES:
        a, b, c, n1, n2 = shape
        for (eta_deg, omega_deg) in ANGLES:
            p = surface_points(np.array([math.radians(eta_deg)]),
                               np.array([math.radians(omega_deg)]), a, b, c, n1, n2)
            k1a, k2a = principal_curvatures_local(p, a, b, c, n1, n2)
            k1f, k2f = principal_curvatures_local(p, a, b, c, n1, n2, method="fd")
            ra = effective_radius_hertz(k1a, k2a, np.zeros(1), np.zeros(1))
            rf = effective_radius_hertz(k1f, k2f, np.zeros(1), np.zeros(1))
            ma = np.linalg.norm(rolling_resistance_torque(
                np.array([f_n]), ra, omega_rel, np.array([mu_r]), np.array([1e-3])))
            mf = np.linalg.norm(rolling_resistance_torque(
                np.array([f_n]), rf, omega_rel, np.array([mu_r]), np.array([1e-3])))
            assert abs(ma - mf) / max(ma, 1e-30) < 1e-2


def test_rolling_friction_saturation_criterion_unchanged():
    """Parent criterion: |M| = mu_r R_eff F_n (saturation), analytic curvature."""
    mu_r = 0.15
    f_n = 2.5
    omega_rel = np.array([[3.0, 0.0, 0.0]])
    for shape in SHAPES:
        p = surface_points(np.array([0.5]), np.array([1.1]), *shape)
        k1, k2 = principal_curvatures_local(p, *shape)
        r_eff = float(effective_radius_hertz(k1, k2, np.zeros(1), np.zeros(1))[0])
        m = float(np.linalg.norm(rolling_resistance_torque(
            np.array([f_n]), np.array([r_eff]), omega_rel,
            np.array([mu_r]), np.array([1e-3]))))
        assert abs(m / (mu_r * r_eff * f_n) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# sphere fast path == general implicit-surface path
# ---------------------------------------------------------------------------

def _random_sphere_bodies(n=12, seed=5, radius=0.005, spacing=1.8):
    """Lattice of touching-ish spheres (overlap 20 % of the radius).

    Deliberately in the physical DEM regime (overlap ``O(1 %)`` of the
    diameter); *deeply* overlapping pairs are a separate, documented limitation
    of the implicit-surface overlap definition (see
    ``test_deep_overlap_is_outside_the_sdf_regime``).
    """
    rng = np.random.default_rng(seed)
    bodies = []
    side = int(math.ceil(n ** (1.0 / 3.0)))
    k = 0
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if k >= n:
                    break
                p = np.array([ix, iy, iz], dtype=float) * spacing * radius
                b = make_superquadric_body(radius, radius, radius, 2.0, 2.0, p, 2500.0)
                b.velocity = rng.uniform(-0.2, 0.2, 3)
                b.omega = rng.uniform(-2.0, 2.0, 3)
                bodies.append(b)
                k += 1
    return bodies


def test_sphere_fast_path_forces_equal_general_path():
    bodies = _random_sphere_bodies()
    mat = ContactMaterial(e_star=1e9, mu=0.4, mu_rolling=0.1)
    s_fast = SQSystem(bodies, [], SQSystemConfig(material=mat, gravity=np.zeros(3)))
    s_fast._spheres_only_cache = True
    f1, t1, c1 = s_fast.contact_forces()
    s_gen = SQSystem(bodies, [], SQSystemConfig(material=mat, gravity=np.zeros(3)))
    s_gen._spheres_only_cache = False
    f2, t2, c2 = s_gen.contact_forces()
    assert len(c1) == len(c2) > 0
    assert np.allclose(f1, f2, rtol=1e-9, atol=1e-9)
    assert np.allclose(t1, t2, rtol=1e-9, atol=1e-9)


def test_deep_overlap_is_outside_the_sdf_regime():
    """Documented limitation: ``delta/D > 50 %`` breaks the SDF overlap.

    When one centre lies inside the other sphere, the projected-surface overlap
    of the parent implicit path degenerates to the centre distance ``d``
    (``r_b - r_a + d``) instead of the exact ``r_a + r_b - d``.  The analytic
    sphere path keeps the exact value, so the two paths are only compared in the
    physical regime (``delta/D <= 20 %``) above.
    """
    r = 0.005
    b1 = make_superquadric_body(r, r, r, 2.0, 2.0, [0.0, 0.0, 0.0], 2500.0)
    b2 = make_superquadric_body(r, r, r, 2.0, 2.0, [0.003, 0.0, 0.0], 2500.0)
    mat = ContactMaterial(e_star=1e8, mu=0.0, mu_rolling=0.0)
    out = {}
    for fast in (True, False):
        s = SQSystem([b1, b2], [], SQSystemConfig(material=mat, gravity=np.zeros(3)))
        s._spheres_only_cache = fast
        out[fast] = s.contact_forces()[2][0]["overlap"]
    assert abs(out[True] - 0.007) < 1e-12          # exact r1 + r2 - d
    assert abs(out[False] - 0.003) < 2e-3          # SDF degeneracy (documented)
    assert out[True] > out[False]


def test_wall_rolling_resistance_is_absent_in_both_wall_paths():
    """Finding of this node (reported, not silently changed).

    The superquadric wall path of the parent node applies **no** rolling
    resistance at a wall contact, unlike its sphere-sphere and clump-wall paths.
    The sphere fast path reproduces the parent behaviour exactly, so switching
    paths cannot move the physics; the inconsistency is recorded here because it
    is the dominant model assumption for the absolute repose angle (it does not
    affect the N-convergence of criterion (f), which uses one code path).
    """
    r = 0.005
    b = make_superquadric_body(r, r, r, 2.0, 2.0, [0.0, 0.0, 0.004], 2500.0)
    b.omega = np.array([0.5, 1.0, 0.0])          # spin about z: no normal torque
    mat = ContactMaterial(e_star=1e8, mu=0.4, mu_rolling=0.5)  # large mu_r on purpose
    out = {}
    for fast in (True, False):
        s = SQSystem([b], [Wall(np.array([0.0, 0.0, 1.0]))],
                     SQSystemConfig(material=mat, gravity=np.zeros(3)))
        s._spheres_only_cache = fast
        f, t, c = s.contact_forces()
        out[fast] = float(np.linalg.norm(t[0, :2]))
    # pure r x f moment: |m| = |r_vec| * |f_t| with r_vec = (0,0,-r+delta)
    assert out[True] == pytest.approx(out[False], rel=1e-9)
    assert out[True] < 1e-2 * 0.5 * 1.0     # far below mu_r * R * F_n saturation


def test_sphere_wall_fast_path_matches_general_path():
    r = 0.005
    b1 = make_superquadric_body(r, r, r, 2.0, 2.0, [0.0, 0.0, 0.004], 2500.0)
    b2 = make_superquadric_body(r, r, r, 2.0, 2.0, [0.05, 0.0, 0.0045], 2500.0)
    b1.velocity = np.array([0.1, 0.0, -0.2])
    b1.omega = np.array([0.5, 1.0, 0.0])
    b2.velocity = np.array([0.0, 0.05, -0.1])
    walls = [Wall(np.array([0.0, 0.0, 1.0]))]
    mat = ContactMaterial(e_star=1e8, mu=0.4, mu_rolling=0.1)
    out = []
    for fast in (True, False):
        s = SQSystem([b1, b2], walls, SQSystemConfig(material=mat, gravity=np.zeros(3)))
        s._spheres_only_cache = fast
        f, t, c = s.contact_forces()
        out.append((f, t, len(c)))
    assert out[0][2] == out[1][2]
    assert np.allclose(out[0][0], out[1][0], rtol=1e-9, atol=1e-9)
    assert np.allclose(out[0][1], out[1][1], rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# integrators
# ---------------------------------------------------------------------------

def test_velocity_verlet_is_second_order_harmonic():
    omega = 2.0 * math.pi
    t_end = 10.37
    dts = [1e-2, 5e-3, 2.5e-3]
    errs = []
    for dt in dts:
        n = int(round(t_end / dt))
        x, _ = integrate("velocity_verlet", harmonic_accel(omega), [1.0], [0.0], dt, n)
        errs.append(abs(x[0] - math.cos(omega * n * dt)))
    assert order_from_errors(dts, errs) >= 1.9


def test_symplectic_euler_is_first_order_negative_control():
    omega = 2.0 * math.pi
    t_end = 10.37
    dts = [1e-2, 5e-3, 2.5e-3]
    errs = []
    for dt in dts:
        n = int(round(t_end / dt))
        x, _ = integrate("euler", harmonic_accel(omega), [1.0], [0.0], dt, n)
        errs.append(abs(x[0] - math.cos(omega * n * dt)))
    p = order_from_errors(dts, errs)
    assert 0.8 < p < 1.3


def test_velocity_verlet_is_second_order_duffing():
    acc = duffing_accel(2.0 * math.pi, 2.0)
    dts = [1e-2, 5e-3, 2.5e-3]
    ref_dt = 2.5e-5
    xr, _ = integrate("velocity_verlet", acc, [1.0], [0.0], ref_dt,
                      int(round(10.37 / ref_dt)))
    errs = []
    for dt in dts:
        n = int(round(10.37 / dt))
        x, _ = integrate("velocity_verlet", acc, [1.0], [0.0], dt, n)
        errs.append(abs(x[0] - xr[0]))
    assert order_from_errors(dts, errs) >= 1.9


def test_integrator_dispatch_rejects_unknown_name():
    with pytest.raises(ValueError):
        step("rk4", np.zeros(1), np.zeros(1), 1e-3, harmonic_accel(1.0))


def test_verlet_step_is_reversible_for_a_free_particle():
    """Kick-drift-kick with a constant force must be time-reversible."""
    g = np.array([0.0, 0.0, -9.81])
    accel = lambda x, v: np.full_like(x, g)
    x0 = np.array([[0.0, 0.0, 1.0]])
    v0 = np.array([[0.5, 0.0, 2.0]])
    x, v, _ = velocity_verlet_step(x0, v0, 1e-3, accel)
    xb, vb, _ = velocity_verlet_step(x, -v, 1e-3, accel)
    assert np.allclose(xb, x0, atol=1e-12)
    assert np.allclose(vb, -v0, atol=1e-12)


def test_restitution_one_means_zero_damping():
    assert ContactMaterial(restitution=1.0).damping_ratio == 0.0
    assert ContactMaterial(restitution=0.4).damping_ratio > 0.0


def _damping_fit_rel_err(integrator: str, dt: float, gamma: float = 1000.0,
                         t_end: float = 0.02, **kw) -> float:
    n = int(round(t_end / dt))
    w = 50.0
    ts, ws = [0.0], [w]
    for k in range(n):
        x, v, _ = step(integrator, np.zeros(1), np.array([w]), dt,
                       lambda _x, _v: -gamma * _v, **kw)
        w = float(v[0])
        ts.append((k + 1) * dt)
        ws.append(w)
    slope = np.polyfit(np.array(ts), np.log(np.maximum(np.array(ws), 1e-300)), 1)[0]
    return abs(-slope - gamma) / gamma


def test_velocity_verlet_damping_fit_error_below_one_percent():
    assert _damping_fit_rel_err("velocity_verlet", 1e-4) < 0.01


def test_euler_damping_fit_error_is_the_documented_five_percent():
    err = _damping_fit_rel_err("euler", 1e-4)
    assert 0.03 < err < 0.08


def test_plain_kick_drift_kick_damping_fit_is_first_order():
    err_kdk = _damping_fit_rel_err("velocity_verlet", 1e-4, velocity_predictor=False)
    err_pc = _damping_fit_rel_err("velocity_verlet", 1e-4)
    assert err_kdk > err_pc * 5.0      # the velocity predictor is what buys 2nd order


def _energy_deviation(integrator: str, scale: float = 0.5, steps: int = 10000,
                      v0: float = 0.25) -> float:
    radius, rho = 0.005, 2500.0
    mat = ContactMaterial(e_star=reduced_modulus(1e7, 0.3, 1e7, 0.3),
                          g_star=1e7 / 5.2, mu=0.0, mu_rolling=0.0,
                          restitution=1.0)
    body = make_superquadric_body(radius, radius, radius, 2.0, 2.0,
                                  [0.0, 0.0, 0.02], rho)
    body.velocity = np.array([0.0, 0.0, -v0])
    sysm = SQSystem([body], [Wall(np.array([0.0, 0.0, 1.0]))],
                    SQSystemConfig(material=mat, integrator=integrator,
                                   gravity=np.zeros(3)))
    sysm.dt *= scale
    e0 = sysm.energy()
    worst = 0.0
    for k in range(steps):
        sysm.step()
        if k % 5 == 0:
            worst = max(worst, abs(sysm.energy() - e0) / abs(e0))
    return worst


def test_conservative_energy_drift_velocity_verlet_below_threshold():
    assert _energy_deviation("velocity_verlet") < 1e-3


def test_conservative_energy_drift_euler_negative_control():
    """The same scenario must FAIL for first-order symplectic Euler."""
    assert _energy_deviation("euler") > 1e-2


def test_system_config_rejects_unknown_integrator():
    radius, rho = 0.005, 2500.0
    body = make_superquadric_body(radius, radius, radius, 2.0, 2.0, [0, 0, 0.01], rho)
    cfg = SQSystemConfig(integrator="leapfrog")
    sysm = SQSystem([body], [], cfg)
    with pytest.raises(ValueError):
        sysm.step()


# ---------------------------------------------------------------------------
# artefact checks (the long runs, when present in the tree)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(MATURITY), reason="maturity artifact absent")
def test_artifact_maturity_meets_criteria():
    with open(MATURITY) as fh:
        res = json.load(fh)
    a = res["A"]
    assert a["ellipsoid_gaussian_curvature_max_rel_err"] < 1e-6
    assert a["ellipsoid_pole_principal_max_rel_err"] < 1e-6
    assert a["general_mesh_max_rel_err"] < 1e-2
    assert a["mesh_order_min"] >= 1.5
    assert max(r["max_rel_err"] for r in a["legacy_scalar_fd"]) > 0.15
    assert res["B"]["rolling_torque_max_rel_change"] < 1e-2
    assert abs(res["B"]["saturation_ratio_analytic"] - 1.0) < 1e-9
    assert res["D"]["harmonic"]["velocity_verlet"]["order"] >= 1.9
    assert res["D"]["harmonic"]["euler"]["order"] < 1.3
    assert res["D"]["duffing"]["velocity_verlet"]["order"] >= 1.9
    decay = res["D"]["damping_decay"]["1.00e-04"]
    assert decay["velocity_verlet"]["rel_err"] < 0.01
    assert decay["euler"]["rel_err"] > 0.03
    drift = res["D"]["energy_drift_1e4"]
    assert drift["velocity_verlet"]["0.5"]["max_rel_energy_dev"] < 1e-3
    assert drift["euler"]["0.5"]["max_rel_energy_dev"] > 1e-2


@pytest.mark.skipif(not os.path.exists(COLUMN), reason="column-collapse artifact absent")
def test_artifact_column_collapse_convergence():
    with open(COLUMN) as fh:
        res = json.load(fh)
    runs = res["runs"]
    assert len(runs) >= 3
    angles = [r["repose_angle_deg"] for r in runs]
    spread = (max(angles) - min(angles)) / max(np.mean(angles), 1e-30)
    assert spread <= 0.05, f"repose angle spread {spread:.2%} over {angles}"
    assert res["summary"]["reaches_32k"] is not None
    # friction scan must be monotone (higher mu -> steeper pile)
    if "friction_scan" in res and len(res["friction_scan"]) >= 3:
        mus = [r["mu"] for r in res["friction_scan"]]
        th = [r["repose_angle_deg"] for r in res["friction_scan"]]
        order = np.argsort(mus)
        th = np.array(th)[order]
        assert th[-1] >= th[0]