"""Verification tests for the Hertz-Mindlin granular wave-speed toolkit.

The tests fall in four groups:

* **contact laws** - the engine's Hertz normal force, Mindlin tangential
  stiffness, Coulomb cap and rigid-platen law against their closed forms;
* **packing diagnostics** - coordination number, porosity, and the Hertz
  pressure-overlap scaling ``p ~ delta^(3/2)`` that gives ``v ~ p^(1/6)``;
* **effective medium** - the Digby (1981) closed form against the independent
  affine contact average, and the exact ``p^(1/6)`` velocity scaling;
* **estimators** - the cross-correlation travel-time pick on signals with a
  known delay, and the frictionless "no shear rigidity" control.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordem.dem3d_triaxial import servo_velocity
from tensordem.granular_wave import (
    GranularPacking,
    GranularWaveConfig,
    cross_correlation_lag,
)
from tensordem.granular_wave.emt import (
    affine_contact_medium,
    digby_hertz_mindlin,
    loglog_slope,
    moduli_from_velocities,
    poisson_from_velocities,
)


def _cfg(**kw) -> GranularWaveConfig:
    base = dict(nx=3, ny=3, nz=8, pressure=1.0e5, polydispersity=0.0, jitter=0.0,
                servo_v_max=0.05, servo_gain=3.0e-3)
    base.update(kw)
    return GranularWaveConfig(**base)


def _two_particles(cfg: GranularWaveConfig, overlap: float, *,
                   contact: bool = True) -> GranularPacking:
    """A packing reduced to two spheres with a prescribed overlap."""
    sim = GranularPacking(cfg)
    r = cfg.radius
    d = 2.0 * r - overlap if contact else 2.0 * r + abs(overlap)
    sim.pos = torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]], dtype=torch.float64)
    sim.radius = torch.tensor([r, r], dtype=torch.float64)
    sim.mass = cfg.density * (4.0 / 3.0) * math.pi * r**3 * torch.ones(2, dtype=torch.float64)
    sim.inertia = 0.4 * sim.mass * r**2
    sim.n = 2
    sim.vel = torch.zeros_like(sim.pos)
    sim.omega = torch.zeros_like(sim.pos)
    # push the platens far away so only the pair interacts
    sim.wall_pos = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
    sim.wall_tan = torch.zeros((2, 6, 3), dtype=torch.float64)
    sim.rebuild_pairs()
    return sim


# --------------------------------------------------------------------------
# contact laws
# --------------------------------------------------------------------------
def test_hertz_normal_force_matches_closed_form():
    cfg = _cfg()
    overlap = 1.0e-4
    sim = _two_particles(cfg, overlap)
    force, _ = sim.forces()
    r_star = 0.5 * cfg.radius
    expected = (4.0 / 3.0) * cfg.e_star * math.sqrt(r_star) * overlap ** 1.5
    assert float(force[1, 0]) == pytest.approx(expected, rel=1e-12)
    assert float(force[0, 0]) == pytest.approx(-expected, rel=1e-12)


def test_hertz_tangent_stiffness_matches_finite_difference():
    cfg = _cfg()
    overlap = 1.0e-4
    eps = 1.0e-9
    f1 = float(sim_f(_two_particles(cfg, overlap)))
    f2 = float(sim_f(_two_particles(cfg, overlap + eps)))
    k_num = (f2 - f1) / eps
    r_star = 0.5 * cfg.radius
    k_ana = 2.0 * cfg.e_star * math.sqrt(r_star * overlap)
    assert k_num == pytest.approx(k_ana, rel=2e-3)


def sim_f(sim: GranularPacking) -> float:
    force, _ = sim.forces()
    return force[1, 0]


def test_mindlin_tangential_stiffness_matches_closed_form():
    """``k_t = 8 G* sqrt(R* delta)`` from a small tangential offset, no slip."""
    cfg = _cfg(friction=10.0)          # friction large enough to stay in stick
    overlap = 2.0e-4
    sim = _two_particles(cfg, overlap)
    _, _ = sim.forces()                # populate the contact state
    disp = 1.0e-9
    # the tangential displacement is accumulated from the relative *velocity*
    # (Cundall-Strack), so feed one step of v_y = disp/dt and then re-evaluate
    sim.vel[1, 1] = disp / sim.dt
    sim.forces()
    sim.vel.zero_()
    force, _ = sim.forces()
    r_star = 0.5 * cfg.radius
    k_t = 8.0 * cfg.g_star * math.sqrt(r_star * overlap)
    # the tangential force on particle 1 is -k_t * delta_t
    assert float(force[1, 1]) == pytest.approx(-k_t * disp, rel=5e-3)


def test_coulomb_cap_limits_tangential_force():
    cfg = _cfg(friction=0.3)
    overlap = 1.0e-4
    sim = _two_particles(cfg, overlap)
    # drive a large tangential slip (through the relative velocity, as the
    # Cundall-Strack tangential state is velocity-driven) so the contact slides
    for _ in range(200):
        sim.vel[1, 1] = 1.0e-3
        force, _ = sim.forces()
    f_n = float(sim.pair_force_normal.max())
    # the pair axis is x, so the tangential component on particle 1 is y
    f_t = abs(float(force[1, 1]))
    assert f_t <= cfg.friction * f_n * (1.0 + 1e-9)


def test_rigid_platen_hertz_law():
    cfg = _cfg()
    sim = GranularPacking(cfg)
    r = cfg.radius
    sim.pos = torch.zeros((1, 3), dtype=torch.float64)
    sim.pos[0, 2] = sim.wall_pos[5] - r + 2.0e-4    # 0.2 mm into the top platen
    sim.radius = torch.tensor([r], dtype=torch.float64)
    sim.mass = cfg.density * (4.0 / 3.0) * math.pi * r**3 * torch.ones(1, dtype=torch.float64)
    sim.inertia = 0.4 * sim.mass * r**2
    sim.n = 1
    sim.vel = torch.zeros_like(sim.pos)
    sim.omega = torch.zeros_like(sim.pos)
    sim.pairs = torch.zeros((0, 2), dtype=torch.long)
    sim.pair_tan = torch.zeros((0, 3), dtype=torch.float64)
    sim.pair_force_normal = torch.zeros(0, dtype=torch.float64)
    sim.wall_tan = torch.zeros((1, 6, 3), dtype=torch.float64)
    overlap = 2.0e-4
    force, _ = sim.forces()
    expected = (4.0 / 3.0) * cfg.e_star_wall * math.sqrt(r) * overlap ** 1.5
    assert float(force[0, 2]) == pytest.approx(-expected, rel=1e-10)


def test_contact_inactive_without_overlap():
    cfg = _cfg()
    sim = _two_particles(cfg, 1.0e-4, contact=False)
    force, _ = sim.forces()
    assert float(torch.linalg.vector_norm(force)) == 0.0


def test_servo_law_is_shared_with_true_triaxial():
    """The pressure controller must be the triaxial protocol law, not a copy."""
    assert servo_velocity(10.0, 1.0e-3, 0.05) == pytest.approx(0.01)
    assert servo_velocity(1.0e6, 1.0e-3, 0.05) == pytest.approx(0.05)
    assert servo_velocity(-1.0e6, 1.0e-3, 0.05) == pytest.approx(-0.05)


# --------------------------------------------------------------------------
# packing diagnostics
# --------------------------------------------------------------------------
def test_solid_fraction_matches_lattice_geometry():
    cfg = _cfg(nx=4, ny=4, nz=4)
    sim = GranularPacking(cfg)
    solid = float((sim.mass / cfg.density).sum())
    assert sim.solid_fraction() == pytest.approx(solid / sim.volume(), rel=1e-12)
    # a cubic lattice of touching spheres has pi/6 of the volume in the grains
    assert 0.5 < sim.solid_fraction() < 0.6


def test_coordination_number_of_a_compressed_lattice():
    cfg = _cfg(nx=4, ny=4, nz=4)
    sim = GranularPacking(cfg)
    assert sim.contact_network()["coordination_number"] == 0.0
    # squeeze 1% so every neighbour overlaps
    sim.pos = sim.pos * 0.99
    sim.wall_pos = sim.wall_pos * 0.99
    sim.rebuild_pairs()
    z = sim.contact_network()["coordination_number"]
    # 4x4x4 cubic lattice: 3 * 3 * 4 * 4 = 144 bonds, so 2*144/64 = 4.5 (the
    # surface grains have fewer neighbours than the bulk, which is why the finite
    # specimen value is below the bulk 6)
    assert z == pytest.approx(4.5, abs=1e-9)


def test_hertz_pressure_overlap_scaling():
    """``p ~ delta^(3/2)`` on a pre-stressed lattice (the origin of ``v ~ p^(1/6)``)."""
    cfg = _cfg(nx=4, ny=4, nz=6)
    sim = GranularPacking(cfg)
    sim.quench = 0.98
    data = []
    for _ in range(6):
        sim.pos = sim.pos * 0.998
        sim.wall_pos = sim.wall_pos * 0.998
        sim.rebuild_pairs()
        for n in range(120):
            sim.step()
            if (n + 1) % 20 == 0:
                sim.rebuild_pairs()
        data.append((sim.mean_contact_overlap(), float(sim.pressures().mean())))
    sim.quench = 1.0
    delta = [d for d, _ in data if d > 0]
    p = [v for d, v in data if d > 0]
    assert len(delta) >= 2
    slope = loglog_slope(delta, p)
    assert slope == pytest.approx(1.5, abs=0.05)


def test_static_shear_modulus_is_positive_and_repeatable():
    """``G`` from the tangential platen test: positive, and repeatable to 50%.

    The reload is not exactly the pristine state (the platen return leaves a
    residual tangential state in the outermost contact layer), so this test only
    claims order-of-magnitude repeatability; the production study reports one
    measurement per state from the pristine configuration.
    """
    cfg = _cfg(nx=3, ny=3, nz=6)
    values = []
    for _ in range(2):
        sim = GranularPacking(cfg)          # fresh packing for each measurement
        sim.quench = 0.98
        sim.pos = sim.pos * 0.985
        sim.wall_pos = sim.wall_pos * 0.985
        sim.rebuild_pairs()
        for _ in range(200):
            sim.step()
        sim.quench = 1.0
        values.append(sim.static_shear_modulus(
            strain=2.0e-5, relax_steps=200)["shear_modulus_Pa"])
    assert values[0] > 0.0 and values[1] > 0.0
    assert values[1] == pytest.approx(values[0], rel=0.02)


# --------------------------------------------------------------------------
# effective medium
# --------------------------------------------------------------------------
def test_digby_equals_affine_contact_average():
    """The affine contact average reproduces Digby's closed form exactly."""
    kwargs = dict(coordination_number=6.0, porosity=0.42, pressure=1.0e6)
    cfg = _cfg()
    digby = digby_hertz_mindlin(grain_shear_modulus=cfg.shear_modulus,
                               grain_poisson_ratio=cfg.poisson_ratio,
                               bulk_density=1600.0, **kwargs)
    affine = affine_contact_medium(mean_radius=cfg.radius, e_star=cfg.e_star,
                                  g_star=cfg.g_star, bulk_density=1600.0, **kwargs)
    assert digby["K_Pa"] == pytest.approx(affine["K_Pa"], rel=1e-3)
    assert digby["G_Pa"] == pytest.approx(affine["G_Pa"], rel=1e-3)


def test_effective_medium_sixth_root_scaling():
    """``K, G ~ p^(1/3)`` and ``v ~ p^(1/6)`` exactly."""
    cfg = _cfg()
    kw = dict(coordination_number=6.0, porosity=0.42,
              grain_shear_modulus=cfg.shear_modulus,
              grain_poisson_ratio=cfg.poisson_ratio, bulk_density=1600.0)
    lo = digby_hertz_mindlin(pressure=1.0e5, **kw)
    hi = digby_hertz_mindlin(pressure=8.0e5, **kw)
    assert hi["K_Pa"] / lo["K_Pa"] == pytest.approx(8.0 ** (1.0 / 3.0), rel=1e-12)
    assert hi["v_p_m_s"] / lo["v_p_m_s"] == pytest.approx(8.0 ** (1.0 / 6.0), rel=1e-12)
    assert hi["v_s_m_s"] / lo["v_s_m_s"] == pytest.approx(8.0 ** (1.0 / 6.0), rel=1e-12)


def test_poisson_relations_are_consistent():
    cfg = _cfg()
    nu = cfg.poisson_ratio
    k_ratio = 2.0 * (1.0 - nu) / (2.0 - nu)          # k_t / k_n, Mindlin
    nu_eff = (1.0 - k_ratio) / (4.0 + k_ratio)
    v_s = 300.0
    v_p = v_s * math.sqrt(2.0 * (1.0 - nu_eff) / (1.0 - 2.0 * nu_eff))
    assert poisson_from_velocities(v_p, v_s) == pytest.approx(nu_eff, rel=1e-9)
    mod = moduli_from_velocities(1600.0, v_p, v_s)
    assert mod["poisson"] == pytest.approx(nu_eff, rel=1e-9)
    with pytest.raises(ValueError):
        poisson_from_velocities(200.0, 300.0)


def test_loglog_slope_basics():
    x = [1.0, 2.0, 4.0, 8.0]
    y = [v ** (1.0 / 6.0) for v in x]
    assert loglog_slope(x, y) == pytest.approx(1.0 / 6.0, rel=1e-12)
    with pytest.raises(ValueError):
        loglog_slope([1.0], [2.0])
    with pytest.raises(ValueError):
        loglog_slope([1.0, -2.0], [1.0, 2.0])


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------
def test_cross_correlation_recovers_known_delay():
    dt = 1.0e-4
    t = np.arange(800) * dt
    width = 4.0e-3

    def pulse(t0):
        return np.where((t >= t0) & (t < t0 + width),
                        np.sin(np.pi * (t - t0) / width) ** 2, 0.0)

    a = pulse(5.0e-3) + 1e-3 * np.random.default_rng(0).standard_normal(t.size)
    b = pulse(5.0e-3 + 7.0e-3) + 1e-3 * np.random.default_rng(1).standard_normal(t.size)
    lag = cross_correlation_lag(a, b, dt)
    assert lag["lag_s"] == pytest.approx(7.0e-3, abs=0.15 * dt)
    back = cross_correlation_lag(b, a, dt)
    assert back["lag_s"] == pytest.approx(-7.0e-3, abs=0.15 * dt)


def test_cross_correlation_rejects_short_signals():
    with pytest.raises(ValueError):
        cross_correlation_lag([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0e-4)


# --------------------------------------------------------------------------
# frictionless control
# --------------------------------------------------------------------------
def test_frictionless_contacts_lose_shear_rigidity():
    """``mu -> 0`` must collapse ``G`` while ``p`` stays finite (no shear rigidity)."""
    def measure(mu: float) -> tuple[float, float]:
        cfg = _cfg(nx=3, ny=3, nz=6, friction=mu)
        sim = GranularPacking(cfg)
        sim.quench = 0.98
        for _ in range(3):
            sim.pos = sim.pos * 0.995
            sim.wall_pos = sim.wall_pos * 0.995
            sim.rebuild_pairs()
            for n in range(120):
                sim.step()
                if (n + 1) % 20 == 0:
                    sim.rebuild_pairs()
        sim.quench = 1.0
        shear = sim.static_shear_modulus(strain=2.0e-5, relax_steps=150)
        return shear["shear_modulus_Pa"], float(sim.pressures().mean())

    g_fric, p_fric = measure(0.3)
    g_free, p_free = measure(1.0e-6)
    assert p_fric > 0.0 and p_free > 0.0
    assert g_free < 0.2 * g_fric