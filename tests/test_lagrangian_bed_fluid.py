"""Unit and consistency tests for :mod:`tensordem.lagrangian_bed`.

The tests are deliberately fast (no long deposition): they pin the closure
algebra, the contact-law energy behaviour (a sign slip in the normal damping is
an energy pump -- see ``test_free_space_collision_is_dissipative``), the
action-reaction ledger of the 1-D column and the reproducibility of the seeded
packing/integration.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.lagrangian_bed import bed as bd
from tensordem.lagrangian_bed import closures as cl
from tensordem.lagrangian_bed import column as col
from tensordem.lagrangian_bed import mechanics as me
from tensordem.lagrangian_bed.study import StudyOptions, _ergun_reference

DIAMETER = 1.0e-3
RHO_P = 2500.0
RHO_F = 1000.0
MU = 1.0e-3


# ---------------------------------------------------------------------------
# geometry / closure algebra
# ---------------------------------------------------------------------------
def test_sphere_slab_fraction_partitions_the_volume():
    r = 0.5e-3
    assert cl.sphere_slab_fraction(r, r, 0.0, r) == pytest.approx(0.5, rel=1e-12)
    assert cl.sphere_slab_fraction(r, r, 0.0, 2 * r) == pytest.approx(1.0, rel=1e-12)
    assert cl.sphere_slab_fraction(r, r, -2 * r, 0.0) == pytest.approx(0.0, abs=1e-18)
    total = sum(
        float(cl.sphere_slab_fraction(r, r, z, z + 0.1 * r)) for z in np.linspace(0.0, 2 * r, 20, endpoint=False)
    )
    assert total == pytest.approx(1.0, rel=1e-10)


def test_cell_voidage_conserves_solid_volume():
    cfg = bd.BedConfig(n_particles=64, size_xy=6e-3, polydispersity=0.1)
    state = bd.initial_lattice(64, cfg.diameter, cfg.size_xy, 0.0, cfg.seed, density=cfg.rho_particle)
    profile = col.ColumnProfile(0.0, 0.012, 8, area=cfg.size_xy**2)
    cs = col.column_state(profile, state.pos, state.vel, state.radius, cfg.diameter)
    solid = float(np.sum(cs["solid_volume"]))
    expected = float(np.sum(4.0 / 3.0 * np.pi * state.radius**3))
    assert solid == pytest.approx(expected, rel=1e-10)


def test_gidaspow_dense_branch_reproduces_ergun():
    eps = np.array([0.38])
    u = np.array([0.004])
    beta, dense = cl.gidaspow_beta(eps, DIAMETER, RHO_F, MU, u / eps)
    assert bool(dense[0])
    dpdz = beta * (u / eps) / eps  # -dp/dz = beta * u_f / eps
    expected = _ergun_reference(eps[0], DIAMETER, RHO_F, MU, u[0])
    assert float(dpdz[0]) == pytest.approx(expected, rel=1e-12)


def test_gidaspow_branch_switch_is_continuous_enough():
    jump = cl.gidaspow_branch_jump(DIAMETER, RHO_F, MU, slip=0.01)
    assert jump < 0.25  # documented discontinuity of the piecewise closure


def test_wen_yu_limit_returns_isolated_schiller_naumann_drag():
    eps = np.array([0.999])
    slip = 0.05
    force, _, _ = cl.particle_drag_force(eps, DIAMETER, RHO_F, MU, np.array([[0.0, 0.0, slip]]), np.zeros((1, 3)))
    isolated = float(cl.schiller_naumann_drag(slip, DIAMETER, RHO_F, MU))
    assert float(force[0, 2]) == pytest.approx(isolated, rel=0.02)


def test_terminal_velocity_reproduces_stokes_regime():
    d = 2e-5  # Re_t ~ 0.02, where the Oseen correction is below 2 %
    ref = cl.terminal_velocity(d, RHO_P, RHO_F, MU)
    stokes = (RHO_P - RHO_F) * 9.81 * d**2 / (18.0 * MU)
    assert ref["re_terminal"] < 0.05
    assert ref["u_terminal"] == pytest.approx(stokes, rel=0.02)


# ---------------------------------------------------------------------------
# Richardson-Zaki
# ---------------------------------------------------------------------------
def test_rz_exponent_is_monotonically_decreasing():
    re = np.array([0.1, 0.5, 5.0, 50.0, 150.0, 300.0, 400.0, 800.0])
    n = cl.rz_exponent(re)
    assert np.all(np.diff(n) < 0.0)


def test_rz_hindrance_reproduces_the_correlation():
    eps = np.array([0.45, 0.6, 0.75, 0.9])
    ut = cl.terminal_velocity(DIAMETER, RHO_P, RHO_F, MU)
    n = float(cl.rz_exponent(ut["re_terminal"]))
    weight = -cl.submerged_weight(DIAMETER, RHO_P, RHO_F)
    for e in eps:
        s = np.linspace(1e-6, 2.0 * ut["u_terminal"], 400000)
        f = cl.rz_hindrance_drag(np.full_like(s, e), DIAMETER, RHO_F, MU, s, n)
        s_ss = float(s[np.argmin(np.abs(f - weight))])
        assert s_ss == pytest.approx(cl.rz_slip_velocity(ut["u_terminal"], e, n), rel=2e-3)


def test_uniform_suspension_settling_matches_rz_scaling():
    cfg = bd.BedConfig()
    ut = cl.terminal_velocity(cfg.diameter, cfg.rho_particle, cfg.rho_fluid, cfg.mu)
    n = float(cl.rz_exponent(ut["re_terminal"]))
    for eps in (0.55, 0.85):
        run = bd.uniform_suspension_settling(eps, cfg, closure="rz")
        expected = cl.rz_slip_velocity(ut["u_terminal"], run["eps"], n) / ut["u_terminal"]
        assert run["u_over_ut"] == pytest.approx(expected, rel=0.02)


# ---------------------------------------------------------------------------
# mechanics: the contact law must be dissipative
# ---------------------------------------------------------------------------
def test_free_space_collision_is_dissipative():
    box = me.Box(size_xy=1.0, height=1.0, floor_z=-1.0, periodic_xy=True)
    params = me.ContactParameters()
    state = me.make_particles(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.02 * DIAMETER]]),
        0.5 * DIAMETER,
        RHO_P,
        vel=np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]]),
    )
    dt = params.contact_time(float(state.mass[0])) / 50.0
    ke0 = me.kinetic_energy(state)
    for _ in range(400):
        state = me.step(state, box, params, dt, gravity=0.0)
    assert me.kinetic_energy(state) < ke0  # anti-damping would increase it


def test_single_particle_rests_on_the_floor():
    box = me.Box(size_xy=4e-3, height=40e-3, floor_z=0.0, periodic_xy=True)
    params = me.ContactParameters()
    state = me.make_particles(np.array([[2e-3, 2e-3, 5e-3]]), 0.5 * DIAMETER, RHO_P)
    dt = params.contact_time(float(state.mass[0])) / 50.0
    for _ in range(4000):
        state = me.step(state, box, params, dt, max_speed=1.0)
    assert state.pos[0, 2] == pytest.approx(0.5 * DIAMETER, abs=0.05 * DIAMETER)
    assert abs(float(state.vel[0, 2])) < 1e-2


def test_static_grain_equilibrium_residual_is_machine_level():
    audit = bd.static_grain_equilibrium(bd.BedConfig(), steps=4000)
    assert audit["residual_over_weight"] < 1e-6
    assert audit["floor_force"] == pytest.approx(audit["submerged_weight"], rel=1e-5)


# ---------------------------------------------------------------------------
# column: action-reaction ledger
# ---------------------------------------------------------------------------
def _lattice_bed(n_particles=125, size_xy=5e-3):
    cfg = bd.BedConfig(n_particles=n_particles, size_xy=size_xy, polydispersity=0.05)
    state = bd.initial_lattice(
        n_particles, cfg.diameter, size_xy, 0.0, cfg.seed, density=cfg.rho_particle
    )
    return cfg, state


def test_column_voidage_and_drag_ledger_close():
    """The divided-volume drag must equal the fluid-side cell integral exactly."""
    cfg, state = _lattice_bed()
    profile = col.ColumnProfile(0.0, 0.008, 4, area=cfg.size_xy**2)
    cs = col.column_state(profile, state.pos, state.vel, state.radius, cfg.diameter)
    fluid = col.FluidProperties(rho=cfg.rho_fluid, mu=cfg.mu)
    u = 0.01
    steady = col.steady_flow(profile, cs["eps"], cs["v_p"], u, cfg.diameter, fluid)
    part = col.drag_force_partitioned(
        profile, state.pos, state.radius, cs["eps"], cfg.diameter, fluid, u, v_p_cells=cs["v_p"]
    )
    particle_side = float(np.sum(part["force"][:, 2])) / profile.area
    fluid_side = float(np.sum(steady["drag_per_volume"]) * profile.dz)
    assert particle_side == pytest.approx(fluid_side, rel=1e-10)
    # pressure drop carries the 1/eps of the fluid momentum balance
    dp_drag = float(np.sum(part["cell_drag"] / cs["eps"]))
    assert dp_drag > fluid_side  # eps < 1 in a packed column


def test_centre_cell_drag_differs_only_by_the_volume_partition():
    """Documented diagnostic: the simpler centre-cell assignment is a few % off."""
    cfg, state = _lattice_bed()
    profile = col.ColumnProfile(0.0, 0.008, 4, area=cfg.size_xy**2)
    cs = col.column_state(profile, state.pos, state.vel, state.radius, cfg.diameter)
    fluid = col.FluidProperties(rho=cfg.rho_fluid, mu=cfg.mu)
    u = 0.01
    eps_i = cs["eps"][cl.cell_index(state.pos[:, 2], 0.0, 0.008, 4)]
    u_f = np.zeros_like(state.pos)
    u_f[:, 2] = u / eps_i
    force, _, _ = cl.particle_drag_force(
        eps_i, cfg.diameter, fluid.rho, fluid.mu, u_f, np.zeros_like(state.pos),
        volume=cl.sphere_volume(2.0 * state.radius),
    )
    centre = float(np.sum(force[:, 2])) / profile.area
    part = col.drag_force_partitioned(
        profile, state.pos, state.radius, cs["eps"], cfg.diameter, fluid, u, v_p_cells=cs["v_p"]
    )
    exact = float(np.sum(part["force"][:, 2])) / profile.area
    assert abs(centre / exact - 1.0) < 0.25  # 3.7 % in the dense study bed, up to ~20 % on a coarse lattice


def test_at_rest_pressure_is_hydrostatic_without_motion():
    cfg, state = _lattice_bed()
    profile = col.ColumnProfile(0.0, 0.008, 4, area=cfg.size_xy**2)
    cs = col.column_state(profile, state.pos, state.vel, state.radius, cfg.diameter)
    fluid = col.FluidProperties(rho=cfg.rho_fluid, mu=cfg.mu)
    pr = col.at_rest_pressure(profile, cs["eps"], cs["v_p"], cfg.diameter, fluid)
    hydrostatic = fluid.rho * fluid.gravity * profile.length
    assert pr["dp_total"] == pytest.approx(hydrostatic, rel=0.05)


# ---------------------------------------------------------------------------
# numerical certificates
# ---------------------------------------------------------------------------
def test_gci_recovers_a_second_order_sequence():
    exact = 100.0
    f1 = exact + 0.25  # dt/4  (error ~ dt^2, refinement ratio 2)
    f2 = exact + 1.0   # dt/2
    f3 = exact + 4.0   # dt
    out = bd.gci([f1, f2, f3])
    assert out["p"] == pytest.approx(2.0, rel=1e-6)
    assert out["extrapolated"] == pytest.approx(exact, rel=1e-6)


def test_settling_is_bit_reproducible():
    cfg = bd.BedConfig(n_particles=60, size_xy=4e-3)
    a = bd.group_settling(cfg, initial_porosity=0.9, steps=60, sample_every=20)
    b = bd.group_settling(cfg, initial_porosity=0.9, steps=60, sample_every=20)
    assert np.array_equal(a["dp_drag"], b["dp_drag"])
    assert np.array_equal(a["u_mean"], b["u_mean"])


def test_packing_statistics_are_finite_on_a_synthetic_bed():
    cfg, state = _lattice_bed()
    box = me.Box(size_xy=cfg.size_xy, height=0.02, periodic_xy=True)
    stats = bd.porosity_statistics(state, 1.0, box)
    assert 0.0 < stats["mean"] < 1.0
    assert np.isfinite(stats["std"])


def test_study_options_are_json_serialisable():
    import json

    from tensordem.lagrangian_bed.study import json_default

    payload = json.dumps({"opts": StudyOptions().__dict__, "eps": np.array([1.0, 2.0])}, default=json_default)
    assert "packing_seeds" in payload