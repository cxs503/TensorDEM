"""Verification tests for the sea-ice drift dynamics closure (Coriolis + added mass + drag).

Every assertion is against an independently derived reference: closed-form
solutions, a matrix-exponential ODE solution, or a scaling law.  Run with::

    python -m pytest tests/test_sea_ice_drift.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from scipy.linalg import expm

from tensordem.sea_ice_drift import (
    OMEGA_EARTH,
    AddedMassTensor,
    QuadraticDrag,
    SeaIceDriftModel,
    coriolis_parameter,
    deflection_angle,
    ekman_decay_depth,
    ekman_depth_from_profile,
    ekman_drag_coefficient,
    ekman_frictional_depth,
    ekman_spiral,
    ekman_spiral_residual,
    free_drift_state,
    k_cross,
    linear_slab_solution,
    nansen_angle,
    rotate_right,
)

RHO_W = 1025.0
RHO_I = 900.0
CD_W = 0.0045
CD_A = 0.0018
RHO_A = 1.25


def nominal_model(**kw) -> SeaIceDriftModel:
    params = dict(
        latitude_deg=80.0,
        ice_thickness=2.5,
        wind=(6.0, 0.0),
        ice_density=RHO_I,
        rho_air=RHO_A,
        rho_water=RHO_W,
        c_d_air=CD_A,
        c_d_water=CD_W,
    )
    params.update(kw)
    return SeaIceDriftModel(**params)


# ---------------------------------------------------------------------------
# Coriolis parameter and geometry helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "lat,f_ref",
    [(0.0, 0.0), (30.0, 7.2921e-5), (60.0, 2.0 * 7.2921e-5 * math.sin(math.radians(60))),
     (90.0, 2.0 * 7.2921e-5)],
)
def test_coriolis_parameter(lat, f_ref):
    assert coriolis_parameter(lat) == pytest.approx(f_ref, rel=1e-12, abs=1e-18)
    # 2*Omega*sin(phi) must reproduce the textbook 1.458e-4 1/s at the pole
    assert coriolis_parameter(lat) == pytest.approx(
        2 * OMEGA_EARTH * math.sin(math.radians(lat)), rel=1e-15, abs=1e-20
    )


def test_coriolis_parameter_rejects_bad_latitude():
    with pytest.raises(ValueError):
        coriolis_parameter(91.0)


def test_k_cross_and_deflection_conventions():
    # motion due east, Northern Hemisphere: Coriolis force must point south
    f = coriolis_parameter(80.0)
    m = 1000.0
    u = np.array([1.0, 0.0])
    force = -m * f * k_cross(u)
    assert force[1] < 0.0  # south = to the right of the eastward motion
    assert deflection_angle([0.0, -1.0], [1.0, 0.0]) == pytest.approx(90.0)
    assert deflection_angle([0.0, 1.0], [1.0, 0.0]) == pytest.approx(-90.0)
    assert deflection_angle([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0, abs=1e-12)
    assert rotate_right([1.0, 0.0], math.pi / 2) == pytest.approx([0.0, -1.0])


# ---------------------------------------------------------------------------
# Added-mass tensor
# ---------------------------------------------------------------------------

def test_added_mass_tensor_validation_and_properties():
    with pytest.raises(ValueError):
        AddedMassTensor(np.array([[1.0, 2.0], [0.0, 1.0]]))  # not symmetric
    with pytest.raises(ValueError):
        AddedMassTensor(np.array([[-1.0, 0.0], [0.0, 1.0]]))  # negative eigenvalue
    with pytest.raises(ValueError):
        AddedMassTensor(np.zeros((3, 3)))
    iso = AddedMassTensor.isotropic(500.0)
    assert iso.matrix == pytest.approx(np.eye(2) * 500.0)
    assert iso.isotropic_equivalent == pytest.approx(500.0)
    assert iso.anisotropy_ratio == pytest.approx(1.0)
    rot = AddedMassTensor.from_principals(400.0, 900.0, 30.0)
    assert rot.anisotropy_ratio == pytest.approx(900.0 / 400.0)
    # trace is rotation invariant
    assert rot.isotropic_equivalent == pytest.approx(650.0)
    assert float(np.linalg.det(rot.matrix)) == pytest.approx(400.0 * 900.0)
    # force is -M_a @ a
    assert iso.force([1.0, 0.0]) == pytest.approx([-500.0, 0.0])
    with pytest.raises(ValueError):
        AddedMassTensor.from_coefficient(RHO_W, draught=-1.0)


def test_added_mass_enters_inertia_exactly():
    """du/dt at u=0 equals M^{-1} tau_a with M = m I + M_a (no drag force yet)."""
    ma = AddedMassTensor.from_principals(600.0, 1400.0, angle_deg=25.0)
    model = nominal_model(added_mass=ma, latitude_deg=0.0)
    tau_a = model.wind_stress(np.zeros(2))
    M = model.mass_per_area * np.eye(2) + ma.matrix
    ref = np.linalg.solve(M, tau_a)
    assert model.rhs(np.zeros(2)) == pytest.approx(ref, rel=1e-13)
    # and it differs measurably from the no-added-mass acceleration
    plain = nominal_model(latitude_deg=0.0)
    assert np.linalg.norm(model.rhs(np.zeros(2)) - plain.rhs(np.zeros(2))) > 0.1 * np.linalg.norm(ref)


@pytest.mark.parametrize("c_m", [0.0, 0.5, 1.0, 2.0])
def test_spin_up_time_constant_scales_with_added_mass(c_m):
    """f=0, linear drag: u(t) = u_inf (1 - exp(-t/tau)), tau = (m + m_a)/(m C_D)."""
    rate = 1.0e-3
    ma = AddedMassTensor.from_coefficient(RHO_W, 2.25, c_m=c_m)
    model = nominal_model(
        added_mass=ma, latitude_deg=0.0, drag_mode="linear",
        linear_drag_rate=rate, wind_relative=False,
    )
    m = model.mass_per_area
    tau_inf = (m + ma.isotropic_equivalent) / (m * rate)
    u_inf = model.wind_stress(np.zeros(2)) / (m * rate)
    assert model.steady_state() == pytest.approx(u_inf, rel=1e-11)
    t, u, _ = model.integrate((0.0, 0.0), t_end=3.0 * tau_inf, dt=1.0, method="rk4")
    ref = u_inf[None, :] * (1.0 - np.exp(-t[:, None] / tau_inf))
    assert np.max(np.abs(u - ref)) / np.max(np.abs(u_inf)) < 5e-5
    # the fit of the simulated curve returns the analytic time constant
    growth = u[:, 0]
    k = (growth[1] - growth[0]) / (u_inf[0] - growth[0])  # per dt
    tau_fit = -1.0 / math.log(1.0 - k)
    assert tau_fit == pytest.approx(tau_inf, rel=2e-2)


def test_added_mass_leaves_steady_state_unchanged():
    """Added mass is inertia only: the steady force balance must be unaffected."""
    plain = nominal_model()
    heavy = nominal_model(added_mass=AddedMassTensor.isotropic(5000.0))
    u0, u1 = plain.steady_state(), heavy.steady_state()
    assert u1 == pytest.approx(u0, rel=1e-12)
    # ... but the transient to reach it is much slower for heavy added mass
    t, u_p, _ = plain.integrate((0.0, 0.0), t_end=3000.0, dt=1.0)
    t, u_h, _ = heavy.integrate((0.0, 0.0), t_end=3000.0, dt=1.0)
    err_p = np.linalg.norm(u_p[-1] - u0) / np.linalg.norm(u0)
    err_h = np.linalg.norm(u_h[-1] - u0) / np.linalg.norm(u0)
    assert err_h > err_p


def test_anisotropic_added_mass_matches_matrix_exponential():
    """Rotated, non-diagonal M_a + Coriolis + linear drag vs expm(A t) solution."""
    ma = AddedMassTensor.from_principals(800.0, 1600.0, angle_deg=35.0)
    model = nominal_model(
        added_mass=ma, drag_mode="linear", linear_drag_rate=2.0e-3, wind_relative=False
    )
    tau_a = model.wind_stress(np.zeros(2))
    M = model.added_mass.total_mass_matrix(model.mass_per_area)
    Kk = np.array([[0.0, -1.0], [1.0, 0.0]])
    A = np.linalg.solve(
        M, -model.mass_per_area * model.linear_drag_rate * np.eye(2)
        - model.mass_per_area * model.f * Kk,
    )
    b = np.linalg.solve(M, tau_a)
    u_star = -np.linalg.solve(A, b)
    t, u, _ = model.integrate((0.0, 0.0), t_end=400.0, dt=0.05, method="rk4")
    ref = np.array([u_star + expm(A * ti) @ (np.zeros(2) - u_star) for ti in t])
    assert np.max(np.abs(u - ref)) / np.max(np.abs(ref)) < 1e-7
    # second independent check: the model's own steady state
    assert model.steady_state() == pytest.approx(u_star, rel=1e-10)


# ---------------------------------------------------------------------------
# Analytic anchors: linear-friction slab and quadratic free drift
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lat", [60.0, 70.0, 80.0, 85.0, 90.0])
@pytest.mark.parametrize("rate", [1.0e-4, 3.0e-4, 1.0e-3])
def test_linear_slab_angle_is_arctan_f_over_cd(lat, rate):
    model = nominal_model(
        latitude_deg=lat, drag_mode="linear", linear_drag_rate=rate, wind_relative=False
    )
    u = model.steady_state()
    closed = linear_slab_solution(model.wind_stress(np.zeros(2)), model.mass_per_area, model.f, rate)
    assert u == pytest.approx(closed["u"], rel=1e-11, abs=1e-14)
    assert np.linalg.norm(u) == pytest.approx(closed["speed_closed_form"], rel=1e-11)
    assert deflection_angle(u, model.wind) == pytest.approx(
        math.degrees(math.atan2(model.f, rate)), abs=1e-10
    )
    assert deflection_angle(u, model.wind) == pytest.approx(nansen_angle(model.f, rate), abs=1e-10)
    assert model.report(u)["residual_magnitude"] < 1e-12 * np.linalg.norm(model.wind_stress(u))


def test_nansen_angle_scales_with_f_and_drag_rate():
    f = coriolis_parameter(80.0)
    assert nansen_angle(f, f) == pytest.approx(45.0)  # f = C_D -> 45 deg
    angles = [nansen_angle(coriolis_parameter(lat), 3e-4) for lat in (60, 70, 80, 85)]
    assert all(b > a for a, b in zip(angles, angles[1:]))  # more f -> larger deflection
    rates = [1e-4, 3e-4, 1e-3]
    ang = [nansen_angle(f, r) for r in rates]
    assert all(b < a for a, b in zip(ang, ang[1:]))  # more drag -> smaller deflection


@pytest.mark.parametrize("lat", [60.0, 75.0, 85.0])
@pytest.mark.parametrize("wind_speed", [5.0, 10.0, 15.0])
def test_quadratic_free_drift_matches_closed_form(lat, wind_speed):
    model = nominal_model(latitude_deg=lat, wind=(wind_speed, 0.0), wind_relative=False)
    u = model.steady_state()
    tau_a = model.wind_stress(np.zeros(2))
    closed = free_drift_state(
        float(np.linalg.norm(tau_a)), model.mass_per_area, model.f, RHO_W, CD_W
    )
    assert np.linalg.norm(u) == pytest.approx(closed["speed"], rel=1e-11)
    assert deflection_angle(u, model.wind) == pytest.approx(closed["angle_deg"], abs=1e-9)
    # the two equivalent expressions for the speed agree
    assert closed["speed_check"] == pytest.approx(closed["speed"], rel=1e-10)
    # along-wind / transverse momentum balance holds to round-off
    c_w = RHO_W * CD_W
    speed = float(np.linalg.norm(u))
    theta = math.radians(closed["angle_deg"])
    along = float(np.linalg.norm(tau_a)) * math.cos(theta) - c_w * speed**2
    trans = float(np.linalg.norm(tau_a)) * math.sin(theta) - model.mass_per_area * model.f * speed
    scale = float(np.linalg.norm(tau_a))
    assert abs(along) / scale < 1e-10 and abs(trans) / scale < 1e-10


def test_relative_wind_drag_is_a_small_correction_to_the_anchor():
    """Switching from prescribed to relative wind stress changes the answer by <2%."""
    fixed = nominal_model(wind_relative=False)
    relative = nominal_model(wind_relative=True)
    u_f, u_r = fixed.steady_state(), relative.steady_state()
    ang_f = deflection_angle(u_f, fixed.wind)
    ang_r = deflection_angle(u_r, relative.wind)
    assert abs(ang_r - ang_f) / ang_f < 2e-2
    assert abs(np.linalg.norm(u_r) - np.linalg.norm(u_f)) / np.linalg.norm(u_f) < 2e-2


def test_time_integration_reproduces_newton_steady_state():
    model = nominal_model()
    u_ss = model.steady_state()
    t, u, _ = model.integrate((0.0, 0.0), t_end=4.0e4, dt=5.0, method="rk4")
    assert np.linalg.norm(u[-1] - u_ss) / np.linalg.norm(u_ss) < 1e-6
    assert np.linalg.norm(model.forces(u_ss)["net"]) < 1e-12 * np.linalg.norm(model.wind_stress(u_ss))


def test_steady_state_solver_reports_non_convergence():
    """A deliberately vandalised Jacobian must raise, never silently return."""
    model = nominal_model()

    def bad_forces(u, include_inertia=False):
        return {"terms": {}, "net": np.array([1.0, 1.0])}

    original = model.forces
    model.forces = bad_forces  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError):
            model.steady_state(max_iter=3)
    finally:
        model.forces = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Nansen angle band, drift-speed ratio, sweep matrix, negative control
# ---------------------------------------------------------------------------

NANSEN_CASES = [
    # (latitude_deg, wind_speed, ice_thickness)  -- machine-checkable sweep
    (60.0, 5.0, 2.5),
    (60.0, 15.0, 2.5),
    (85.0, 5.0, 2.5),
    (85.0, 15.0, 2.5),
    (70.0, 10.0, 2.5),
    (80.0, 6.0, 2.5),   # nominal validation case
    (80.0, 6.0, 3.0),   # thick multiyear ice
]


@pytest.mark.parametrize("lat,wind_speed,thickness", NANSEN_CASES)
def test_nansen_angle_and_speed_ratio_bounds(lat, wind_speed, thickness):
    model = nominal_model(latitude_deg=lat, wind=(wind_speed, 0.0), ice_thickness=thickness)
    u = model.steady_state()
    rep = model.report(u)
    # (b) drift/wind speed ratio in [1%, 3%] for all cases
    assert 0.01 <= rep["speed_ratio"] <= 0.03, rep["speed_ratio"]
    # deflection to the right in the Northern Hemisphere
    assert 0.0 < rep["deflection_deg"] < 90.0
    # the free-drift angle must also satisfy the closed form to <1%
    tau_a = np.linalg.norm(model.wind_stress(np.zeros(2)))
    closed = free_drift_state(tau_a, model.mass_per_area, model.f, RHO_W, CD_W)
    assert abs(rep["deflection_deg"] - closed["angle_deg"]) / closed["angle_deg"] < 1e-2
    # (a) Nansen band, restricted to the moderate-wind regime (see docs)
    if wind_speed <= 7.0 or thickness >= 4.0:
        assert 20.0 <= rep["deflection_deg"] <= 40.0, rep["deflection_deg"]


def test_nansen_angle_decreases_with_wind_speed_and_thickens_with_ice():
    speeds = [3.0, 5.0, 7.0, 10.0, 15.0]
    angles = [
        nominal_model(wind=(w, 0.0)).report(nominal_model(wind=(w, 0.0)).steady_state())[
            "deflection_deg"
        ]
        for w in speeds
    ]
    assert all(b < a for a, b in zip(angles, angles[1:]))
    thick = [
        nominal_model(ice_thickness=h).report(nominal_model(ice_thickness=h).steady_state())[
            "deflection_deg"
        ]
        for h in (0.5, 1.0, 2.0, 3.0)
    ]
    assert all(b > a for a, b in zip(thick, thick[1:]))
    # alpha -> 0 as f -> 0 (no rotation)
    assert thick[0] > 5.0


@pytest.mark.parametrize("rate", [1e-4, 5e-4, 1e-3])
def test_zero_coriolis_negative_control(rate):
    """(d) f = 0: deflection must vanish and the drift must align with the wind."""
    model = nominal_model(latitude_deg=0.0, drag_mode="linear", linear_drag_rate=rate)
    assert model.f == 0.0
    u = model.steady_state()
    assert deflection_angle(u, model.wind) == pytest.approx(0.0, abs=1e-10)
    assert abs(u[1]) < 1e-12 * abs(u[0])
    # quadratic model at the equator must also be undeflected, and must satisfy
    # the pure drag balance T_a = rho_w C_w U^2
    q = nominal_model(latitude_deg=0.0)
    uq = q.steady_state()
    assert deflection_angle(uq, q.wind) == pytest.approx(0.0, abs=1e-10)
    speed = float(np.linalg.norm(uq))
    # with f = 0 the balance is exactly wind stress = rho_w C_w U^2 (relative drag)
    assert RHO_W * CD_W * speed**2 == pytest.approx(
        float(np.linalg.norm(q.wind_stress(uq))), rel=1e-10
    )


# ---------------------------------------------------------------------------
# Ekman spiral / boundary-layer closure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("f", [1.0e-4, 1.4e-4, 2.0e-4])
@pytest.mark.parametrize("K", [0.005, 0.02, 0.1])
def test_ekman_decay_depth_from_profile(f, K):
    spiral = ekman_spiral((0.05, 0.0), f, K, RHO_W, n_layers=1201)  # grid hits z = D
    D_ref = math.sqrt(2.0 * K / f)
    assert spiral["decay_depth"] == pytest.approx(D_ref, rel=1e-14)
    assert ekman_decay_depth(f, K) == pytest.approx(D_ref, rel=1e-14)
    assert ekman_frictional_depth(f, K) == pytest.approx(math.pi * D_ref, rel=1e-14)
    # e-folding depth recovered by an independent log-linear fit of the profile
    assert ekman_depth_from_profile(spiral["z"], spiral["amplitude"]) == pytest.approx(
        D_ref, rel=1e-9
    )
    # amplitude at z = D_E is exactly 1/e of the surface value
    i = int(round((spiral["z"].size - 1) / 3.0))
    assert spiral["z"][i] == pytest.approx(D_ref, rel=1e-12)
    assert spiral["amplitude"][i] / spiral["surface_speed"] == pytest.approx(
        math.exp(-1.0), rel=1e-12
    )


def test_ekman_depth_scalings():
    """D_E ~ sqrt(K) and ~ 1/sqrt(f)."""
    d1 = ekman_decay_depth(1.4e-4, 0.02)
    assert ekman_decay_depth(1.4e-4, 0.08) == pytest.approx(2.0 * d1, rel=1e-12)
    assert ekman_decay_depth(5.6e-4, 0.02) == pytest.approx(0.5 * d1, rel=1e-12)


def test_ekman_spiral_momentum_balance():
    """The sampled profile satisfies f k x u = K u_zz (central differences)."""
    f, K = coriolis_parameter(80.0), 0.02
    spiral = ekman_spiral((0.05, 0.0), f, K, RHO_W, n_layers=801)
    assert ekman_spiral_residual(spiral, f, K) < 1e-3


def test_ekman_surface_45deg_and_transport_90deg():
    f, K = coriolis_parameter(80.0), 0.02
    tau = np.array([0.05, 0.0])
    spiral = ekman_spiral(tau, f, K, RHO_W, n_layers=1201, depth_factor=6.0)
    assert spiral["surface_deflection_from_stress_deg"] == pytest.approx(45.0, abs=1e-10)
    assert spiral["surface_speed"] == pytest.approx(
        float(np.linalg.norm(tau)) / (RHO_W * math.sqrt(f * K)), rel=1e-12
    )
    # total (depth-integrated) transport is 90 deg right with magnitude tau/(rho f)
    ref_mag = float(np.linalg.norm(tau)) / (RHO_W * f)
    assert np.linalg.norm(spiral["transport"]) == pytest.approx(ref_mag, rel=5e-3)
    assert deflection_angle(spiral["transport"], tau) == pytest.approx(90.0, abs=0.2)
    assert spiral["transport_error"] / ref_mag < 5e-3


def test_ekman_closure_drag_reproduces_complex_analytic_steady_state():
    """Ekman drag law: u = tau_a / (i m f + rho_w sqrt(fK) e^{i pi/4})."""
    K = 0.02
    model = nominal_model(drag_mode="ekman", eddy_viscosity=K, wind_relative=False)
    u = model.steady_state()
    tau_a = model.wind_stress(np.zeros(2))
    tau_c = tau_a[0] + 1j * tau_a[1]
    coef = ekman_drag_coefficient(model.f, K, RHO_W)
    u_ref = tau_c / (1j * model.mass_per_area * model.f + coef * np.exp(1j * math.pi / 4.0))
    assert u[0] == pytest.approx(u_ref.real, rel=1e-10)
    assert u[1] == pytest.approx(u_ref.imag, rel=1e-10)
    assert model.report(u)["residual_magnitude"] < 1e-12 * float(np.linalg.norm(tau_a))
    # closure drag coefficient is exactly rho_w sqrt(f K)
    assert ekman_drag_coefficient(model.f, K, RHO_W) == pytest.approx(
        RHO_W * math.sqrt(model.f * K), rel=1e-14
    )


def test_ekman_closure_requires_viscosity():
    with pytest.raises(ValueError):
        nominal_model(drag_mode="ekman")
    with pytest.raises(ValueError):
        ekman_decay_depth(0.0, 0.02)


def test_drag_law_quadratic_relative_velocity():
    law = QuadraticDrag(RHO_W, CD_W)
    u_body = np.array([0.1, 0.0])
    u_fluid = np.array([0.3, 0.0])
    tau = law.stress(u_body, u_fluid)
    assert tau == pytest.approx([RHO_W * CD_W * 0.2**2, 0.0])  # opposes relative motion
    assert law.stress_magnitude(u_body, u_fluid) == pytest.approx(RHO_W * CD_W * 0.04)
    absolute = QuadraticDrag(RHO_W, CD_W, relative=False)
    assert absolute.stress(u_body, u_fluid) == pytest.approx([RHO_W * CD_W * 0.09, 0.0])
    with pytest.raises(ValueError):
        QuadraticDrag(-1.0, CD_W)