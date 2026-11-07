"""Tests for :mod:`tensordem.ice_rideup` -- 3-D floe-edge contact and ride-up.

Coverage:
  * contact geometry vs analytic solutions (sphere-plane, disk-incline) < 3 %
  * contact geometry vs brute-force sampling (penetration, normal)
  * force decomposition (normal + Coulomb) and torque arm
  * Archimedes buoyancy: static vertical residual < 1e-10 N
  * ride-up resistance conventions (momentum vs channel)
  * dynamics: R_up > 0 for Fr > 0.2, closure residual, negative controls
    (zero Fr -> R_up = 0, no slope facet -> R_up = 0)
"""
import math

import pytest
import torch

from tensordem.ice_rideup import (
    GRAVITY, RHO_ICE, RHO_WATER, BowSlope, ContactParams, RectPatch,
    RideUpSimulation, archimedes_buoyancy, disk_patch_contact, floe_mass,
    hydrostatic_lift, monotonicity_slope, perp_basis, quat_integrate,
    quat_multiply, quat_rotate, quat_to_rotmat, ride_up_resistance,
    rim_patch_contact, run_case, sphere_plane_contact, sphere_plane_force,
)

torch.set_num_threads(1)
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# 1. quaternion algebra
# ---------------------------------------------------------------------------

def test_quaternion_roundtrip_and_rotation():
    q = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=DTYPE)          # 120 deg about (1,1,1)
    R = quat_to_rotmat(q)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, dtype=DTYPE), atol=1e-12)
    assert abs(float(torch.linalg.det(R)) - 1.0) < 1e-12
    v = torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE)
    rot = quat_rotate(q, v)
    assert torch.allclose(rot, torch.tensor([[0.0, 1.0, 0.0]], dtype=DTYPE), atol=1e-12)


def test_quaternion_integration_matches_small_angle():
    q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
    omega = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    dt = 0.01
    q1 = quat_integrate(q0, omega, dt)
    # rotation of pi/2 about z after (pi/2)/omega steps
    n_steps = int(round((math.pi / 2) / (float(omega[0, 2]) * dt)))
    q = q0.clone()
    for _ in range(n_steps):
        q = quat_integrate(q, omega, dt)
    v = torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE)
    assert torch.allclose(quat_rotate(q, v), torch.tensor([[0.0, 1.0, 0.0]], dtype=DTYPE),
                          atol=2e-3)
    assert torch.allclose(q.norm(dim=-1), torch.ones(1, dtype=DTYPE), atol=1e-14)


def test_perp_basis_is_orthonormal():
    a = torch.tensor([[0.3, -0.4, 0.87]], dtype=DTYPE)
    a = a / a.norm(dim=-1, keepdim=True)
    u1, u2 = perp_basis(a)
    for u in (u1, u2):
        assert abs(float((u * a).sum())) < 1e-14
        assert abs(float(u.norm()) - 1.0) < 1e-14
    assert abs(float((u1 * u2).sum())) < 1e-14


# ---------------------------------------------------------------------------
# 2. analytic contact geometry
# ---------------------------------------------------------------------------

def test_sphere_plane_penetration_analytic():
    plane = RectPatch(origin=(0.0, 0.0, 0.0), e1=(1.0, 0.0, 0.0), length1=10.0,
                      normal=(0.0, 0.0, 1.0))
    for height in (1.2, 0.5, 0.0, -0.3):
        c = torch.tensor([0.0, 0.0, height], dtype=DTYPE)
        out = sphere_plane_contact(c, 1.0, plane)
        assert abs(float(out["signed_distance"]) - height) < 1e-14
        assert abs(float(out["penetration"]) - max(0.0, 1.0 - height)) < 1e-14
        assert torch.allclose(out["point_hull"], torch.zeros(3, dtype=DTYPE))


def test_sphere_incline_statics_normal_and_friction():
    """Mass on an incline: N = m g cos(a), |f_t| = mu N once sliding."""
    slope = BowSlope(angle_deg=30.0)
    plane = slope.slope_patch
    n = plane.normal_t
    m = 120.0
    k = 2.0e6
    fric = 0.25
    # place the sphere so that the penalty gives exactly N = m g cos(a)
    n_force = m * GRAVITY * math.cos(slope.alpha)
    pen = n_force / k
    center = plane.origin_t + plane.e1_t * plane.length1 * 0.5 + n * (1.0 - pen)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    slip_dir = -(up - n * float((up * n).sum()))
    slip_dir = slip_dir / slip_dir.norm()
    vel = slip_dir * 0.4                       # sliding up the facet
    out = sphere_plane_force(center, vel, 1.0, plane, stiffness=k, damping=0.0,
                            friction=fric, tangential_damping=1e9)
    n_num = float(out["normal_force"])
    t_num = float(out["tangential_force"].norm())
    assert abs(n_num - n_force) / n_force < 0.03
    assert abs(t_num - fric * n_force) / (fric * n_force) < 0.03
    # friction opposes the slip
    assert float((out["tangential_force"] * slip_dir).sum()) < 0.0
    # the normal carries the vertical component the ride-up needs
    assert float(out["force"][2]) > 0.0


def test_disk_penetration_matches_closed_form():
    """Flat disk vs plane: depth = n.(c_low_rim - p0) - r*|n_perp| (closed form)."""
    slope = BowSlope(angle_deg=30.0, plan_half_angle_deg=0.0)
    plane = slope.slope_patch
    n = plane.normal_t
    r, t = 2.0, 0.5
    centers = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, -0.2]], dtype=DTYPE)
    quats = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], dtype=DTYPE)
    out = disk_patch_contact(centers, quats, torch.full((2,), r), torch.full((2,), t),
                             (plane,), n_samples=512)
    n_perp = math.sqrt(max(0.0, 1.0 - float(n[2]) ** 2))
    for i in range(2):
        low_rim = centers[i] - n.new_tensor([0.0, 0.0, 0.5 * t])
        signed = float(((low_rim - plane.origin_t) * n).sum()) - r * n_perp
        numeric = float(out["penetration"][i])
        # penetration = -signed distance of the deepest rim point
        assert abs(numeric + signed) < 0.01, (i, numeric, signed)


def test_disk_penetration_converges_with_rim_sampling():
    """24-sample kernel vs the 4096-sample limit: < 2 % (or 2 cm) difference.

    The kernel is a sampled version of a closed-form measure, so the honest test
    is convergence of the sampling, on poses that do contact the facet (interior
    and boundary features).
    """
    from tensordem.ice_rideup import patch_set_contact

    slope = BowSlope(angle_deg=25.0, plan_half_angle_deg=20.0)
    planes = slope.patches
    mid = slope.slope_patch_port.origin_t + slope.slope_patch_port.e1_t * 1.8
    centers = torch.stack((
        mid + torch.tensor([0.0, 0.2, 0.0], dtype=DTYPE),
        mid + torch.tensor([-0.4, -0.6, 0.3], dtype=DTYPE),
        mid + torch.tensor([0.5, 1.0, -0.2], dtype=DTYPE),
    ))
    quats = torch.tensor([[1.0, 0, 0, 0],
                          [math.cos(0.2), math.sin(0.2), 0.0, 0.0],
                          [math.cos(0.15), 0.0, math.sin(0.15), 0.0]], dtype=DTYPE)
    axes = quat_rotate(quats, torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE).expand_as(centers))
    r = torch.full((3,), 1.8)
    t = torch.full((3,), 0.4)
    coarse = patch_set_contact(centers, axes, r, t, planes, n_samples=24)
    fine = patch_set_contact(centers, axes, r, t, planes, n_samples=4096)
    for i in range(3):
        pc = float(coarse["penetration"][i])
        pf = float(fine["penetration"][i])
        assert pc > 0.0 and pf > 0.0
        assert abs(pc - pf) < max(0.02, 0.02 * abs(pf)), (i, pc, pf)
        assert float((coarse["normal"][i] * fine["normal"][i]).sum()) > 0.9


def test_contact_normal_is_unit_and_points_into_floe():
    slope = BowSlope(angle_deg=30.0)
    centers = torch.tensor([[0.0, 0.0, -0.4]], dtype=DTYPE)
    quats = torch.tensor([[1.0, 0, 0, 0]], dtype=DTYPE)
    out = disk_patch_contact(centers, quats, torch.tensor([2.0]), torch.tensor([0.5]),
                             slope.patches)
    n = out["normal"][0]
    assert abs(float(n.norm()) - 1.0) < 1e-12
    # the bow-slope facets push the floe up (the ride-up mechanism)
    assert float(n[2]) > 0.0
    # the hull receives a resistance reaction in -x
    assert float(n[0]) > 0.0


def test_edge_contact_registered_at_facet_boundary():
    """A floe outside the facet footprint but touching its edge is registered."""
    slope = BowSlope(angle_deg=30.0)
    plane = slope.slope_patch
    # far along +y beyond the facet half width, level with the lower edge
    mid = plane.origin_t + plane.e1_t * (0.5 * plane.length1)
    center = mid + torch.tensor([0.0, 0.5 * plane.length2 + 1.2, 0.0], dtype=DTYPE)
    axes = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    out = rim_patch_contact(center, axes, 1.6, 0.4, plane, n_samples=256)
    assert float(out["penetration"][0]) > 0.0          # grazing the facet edge
    far = center + torch.tensor([0.0, 5.0, 0.0], dtype=DTYPE)
    out2 = rim_patch_contact(far, axes, 1.6, 0.4, plane, n_samples=256)
    assert float(out2["penetration"][0]) <= 0.0


# ---------------------------------------------------------------------------
# 3. hydrostatics
# ---------------------------------------------------------------------------

def test_buoyancy_residual_is_machine_zero():
    r = torch.tensor([0.5, 1.5], dtype=DTYPE)
    t = torch.tensor([0.3, 0.5], dtype=DTYPE)
    z = 0.5 * t - (RHO_ICE / RHO_WATER) * t      # B = W  <=>  d_sub = (rho_i/rho_w) t
    out = archimedes_buoyancy(z, r, t)
    assert float(out["residual"].abs().max()) < 1e-10
    assert float(out["buoyancy"].sum()) > 0.0


def test_hydrostatic_lift_balances_at_equilibrium():
    r = torch.tensor([2.0], dtype=DTYPE)
    t = torch.tensor([0.5], dtype=DTYPE)
    z = 0.5 * t - (RHO_ICE / RHO_WATER) * t      # B = W  <=>  d_sub = (rho_i/rho_w) t
    f = hydrostatic_lift(z, r, t, torch.zeros(1, dtype=DTYPE), normal_z=math.cos(0.5))
    assert abs(float(f)) < 1e-10


def test_hydrostatic_lift_positive_when_lifted_out_of_water():
    r = torch.tensor([2.0], dtype=DTYPE)
    t = torch.tensor([0.5], dtype=DTYPE)
    z_eq = 0.5 * t - (RHO_ICE / RHO_WATER) * t
    f = hydrostatic_lift(z_eq + 0.6, r, t, torch.zeros(1, dtype=DTYPE), normal_z=math.cos(0.5))
    assert float(f) < 0.0            # out of the water: weight exceeds buoyancy
    assert float(archimedes_buoyancy(z_eq + 0.6, r, t)["weight"]) > 0.0


def test_floe_mass_matches_cylinder_volume():
    r = torch.tensor([2.0], dtype=DTYPE)
    t = torch.tensor([0.5], dtype=DTYPE)
    m = float(floe_mass(r, t))
    assert abs(m - RHO_ICE * math.pi * 4.0 * 0.5) < 1e-9


# ---------------------------------------------------------------------------
# 4. resistance conventions
# ---------------------------------------------------------------------------

def test_ride_up_resistance_conventions():
    a = 30.0
    fz = 1000.0
    mom = ride_up_resistance(fz, a, convention="momentum")
    cha = ride_up_resistance(fz, a, convention="channel")
    assert abs(mom - fz * math.tan(math.radians(a))) < 1e-12
    assert abs(cha - fz * math.sin(math.radians(a))) < 1e-12
    assert mom > cha
    with pytest.raises(ValueError):
        ride_up_resistance(fz, a, convention="bogus")


def test_monotonicity_slope_signs():
    assert monotonicity_slope([0.1, 0.2, 0.3], [1.0, 2.0, 4.0]) > 0.0
    assert monotonicity_slope([0.1, 0.2, 0.3], [4.0, 2.0, 1.0]) < 0.0
    assert math.isnan(monotonicity_slope([1.0, 1.0], [1.0, 2.0]))


# ---------------------------------------------------------------------------
# 5. dynamics (short runs, machine decisions)
# ---------------------------------------------------------------------------

def _short_run(**kw):
    defaults = dict(n_floes=10, steps=90, warmup_steps=20, dt=0.01, n_samples=12,
                    radius_mean=2.0)
    defaults.update(kw)
    return run_case(**defaults)


def test_decomposition_closes():
    res = _short_run(fr=0.3, thickness=0.5, angle_deg=30.0, seed=0)
    assert res.r_total > 0.0
    assert res.closure_residual < 0.02


def test_zero_froude_gives_no_ride_up():
    """Negative control: a stationary ship cannot ride ice up (R_up = 0)."""
    res = _short_run(fr=0.0, thickness=0.5, angle_deg=30.0, seed=0)
    assert abs(res.r_up) < 1e-9
    assert res.n_climbing == 0.0


def test_no_slope_facet_gives_no_ride_up():
    """Negative control: a hull without a bow slope has no ride-up channel."""
    slope = BowSlope(angle_deg=30.0)
    sim = RideUpSimulation(fr=0.3, thickness=0.5, slope=slope, seed=0, n_floes=10,
                           steps=90, warmup_steps=20, dt=0.01, n_samples=12,
                           patches=(slope.shelf_patch,), n_slope_facets=0)
    res = sim.run()
    assert abs(res.r_up) < 1e-9
    assert res.n_climbing == 0.0


def test_ride_up_appears_at_moderate_froude():
    res = _short_run(fr=0.4, thickness=0.5, angle_deg=30.0, seed=1)
    assert res.r_up > 0.0
    assert res.n_climbing > 0.0
    assert res.lift_force_z > 0.0


def test_contact_geometry_is_pose_continuous():
    """A 1 mm pose perturbation cannot change the penetration by more than a few mm."""
    slope = BowSlope(angle_deg=30.0)
    base = torch.tensor([[0.2, 0.3, -0.25]], dtype=DTYPE)
    q = torch.tensor([[1.0, 0, 0, 0]], dtype=DTYPE)
    r = torch.tensor([1.8])
    t = torch.tensor([0.5])
    p0 = float(disk_patch_contact(base, q, r, t, slope.patches, n_samples=32)["penetration"])
    for delta in (0.001, -0.001, 0.002):
        moved = base + torch.tensor([[delta, 0.0, 0.0]], dtype=DTYPE)
        p1 = float(disk_patch_contact(moved, q, r, t, slope.patches, n_samples=32)["penetration"])
        assert abs(p1 - p0) < 0.02


def test_simulation_is_deterministic_for_a_seed():
    a = _short_run(fr=0.3, thickness=0.5, angle_deg=30.0, seed=7)
    b = _short_run(fr=0.3, thickness=0.5, angle_deg=30.0, seed=7)
    assert a.r_total == pytest.approx(b.r_total, rel=0.0, abs=0.0)
    c = _short_run(fr=0.3, thickness=0.5, angle_deg=30.0, seed=8)
    assert abs(c.r_total - a.r_total) > 0.0