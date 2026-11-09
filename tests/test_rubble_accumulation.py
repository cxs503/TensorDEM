"""Tests for the IN69-N06 rubble-accumulation model.

The suite is organised in three layers:

1. **Analytic anchors** -- things that must be exact or near-exact for the
   physics to be trustworthy at all: the drag relaxation time, the concentration
   -> thickness map, the pile-geometry estimators on synthetic configurations,
   the growth-law fit recovery, and the log-log exponent estimator;
2. **Conservation and numerical guards** -- mass ledger closure, no piece lost
   or trapped, penetration bounded, determinism for a given seed;
3. **Regression / controls** -- the pre-registered negative controls (``C_i = 0``
   and ``v_i = 0`` produce no pile) and the load bookkeeping.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from tensordem import rubble_accumulation as ra


SMALL_DOMAIN = ra.DomainSpec(length=8.0, width=6.0)


def small_config(**kwargs) -> ra.AccumulationConfig:
    domain = kwargs.pop("domain", SMALL_DOMAIN)
    structure = kwargs.pop("structure", ra.StructureSpec(diameter=1.0, x_center=5.0,
                                                        y_center=3.0))
    defaults = dict(domain=domain, structure=structure,
                    conditions=ra.IceConditions(velocity=0.3, thickness=0.2,
                                                concentration=0.4),
                    duration=6.0, sample_dt=1.0, d_min=0.1, d_max=0.25)
    defaults.update(kwargs)
    return ra.AccumulationConfig(**defaults)


# --------------------------------------------------------------------------- #
# 1. analytic anchors
# --------------------------------------------------------------------------- #
def test_drag_time_matches_quadratic_drag_linearisation():
    """``tau_d = rho_i pi r / (rho_w C_d v)`` from ``m/(0.5 rho_w C_d A v)``."""
    mat = ra.RubbleMaterial()
    cfg = small_config()
    field = ra.generate_piece_field(cfg.domain, cfg.structure, concentration=0.4,
                                    d_min=cfg.d_min, d_max=cfg.d_max, seed=1)
    sim = ra.RubbleAccumulation(field, cfg)
    r = float(sim.radii.mean())
    expected = (mat.density_ice * math.pi * r
                / (mat.density_water * mat.drag_coefficient * cfg.conditions.velocity))
    assert sim.drag_time == pytest.approx(expected, rel=1e-12)
    # the same constant written through mass and frontal area
    mass = mat.density_ice * math.pi * r ** 2 * cfg.conditions.thickness
    frontal = 2.0 * r * cfg.conditions.thickness
    quadratic = mass / (0.5 * mat.density_water * mat.drag_coefficient * frontal
                        * cfg.conditions.velocity)
    assert sim.drag_time == pytest.approx(quadratic, rel=1e-12)


def test_single_piece_relaxes_onto_drift_speed_exponentially():
    """A lone fragment must approach ``v_i`` with the analytic time constant."""
    domain = ra.DomainSpec(length=20.0, width=8.0)
    structure = ra.StructureSpec(diameter=0.5, x_center=18.0, y_center=7.5)
    cfg = small_config(domain=domain, structure=structure, duration=6.0)
    pos = np.array([[1.0, 4.0]])
    rad = np.array([0.1])
    sim = ra.RubbleAccumulation(ra.PieceField(pos, rad, domain), cfg)
    tau = sim.drag_time
    steps = int(round(2.5 * tau / sim.dt))
    sim.step(steps)
    expected = cfg.conditions.velocity * (1.0 - math.exp(-sim.time / tau))
    assert sim.vel[0, 0] == pytest.approx(expected, rel=2e-2)
    assert abs(sim.vel[0, 1]) < 1e-9


def test_areal_thickness_mapping_is_concentration_over_solid_fraction():
    h, n = 0.3, 0.32
    phi = np.array([0.0, 0.34, 0.68])
    expected = phi * h / (1.0 - n)
    assert np.allclose(ra.areal_thickness_mapping(phi, h, n), expected)
    # full solid packing gives exactly the piece thickness per layer
    assert ra.areal_thickness_mapping(np.array([1.0 - n]), h, n)[0] == pytest.approx(h)
    with pytest.raises(ValueError):
        ra.areal_thickness_mapping(phi, h, 1.0)


def test_pile_geometry_estimators_on_synthetic_configuration():
    """``B_p``, ``L_p`` recover the imposed pile shape exactly."""
    domain = ra.DomainSpec(length=20.0, width=20.0)
    structure = ra.StructureSpec(diameter=2.0, x_center=12.0, y_center=10.0)
    cfg = small_config(domain=domain, structure=structure, d_min=0.1, d_max=0.25,
                       conditions=ra.IceConditions(velocity=0.0, thickness=0.3,
                                                   concentration=0.4))
    # a rectangular slab of touching disks just upstream of the structure face
    radius = 0.25
    xs = np.arange(10.2, 11.9, 2 * radius * 0.999)
    ys = np.arange(8.5, 11.5, 2 * radius * 0.999)
    grid = np.array([[x, y] for x in xs for y in ys])
    rad = np.full(len(grid), radius)
    sim = ra.RubbleAccumulation(ra.PieceField(grid, rad, domain), cfg)
    sim.forces()  # build the contact list for the cluster search
    geom = sim.geometry()
    assert geom["L_p"] == pytest.approx(12.0 - 1.0 - xs.min(), abs=0.02)
    assert geom["B_p"] == pytest.approx(ys.max() - ys.min(), abs=0.02)
    # 4x8 touching disks in one cell -> areal fraction -> thickness
    expected_phi = len(grid) * math.pi * radius ** 2 / (domain.area)
    assert geom["H_p"] >= ra.areal_thickness_mapping(np.array([expected_phi]), 0.3, 0.32)[0]


def test_empty_field_has_zero_geometry_and_zero_load():
    domain = ra.DomainSpec(length=10.0, width=8.0)
    structure = ra.StructureSpec(diameter=1.0, x_center=6.0, y_center=4.0)
    cfg = small_config(domain=domain, structure=structure,
                       conditions=ra.IceConditions(velocity=0.3, thickness=0.2,
                                                   concentration=0.0))
    field = ra.generate_piece_field(domain, structure, concentration=0.0)
    assert field.n == 0
    sim = ra.RubbleAccumulation(field, cfg)
    sim.step(200)
    snap = sim.snapshot()
    assert snap["pile_mass"] == 0.0 and snap["L_p"] == 0.0
    assert sim.structure_force.tolist() == [0.0, 0.0]


def test_exponential_growth_fit_recovers_known_parameters():
    t = np.linspace(0.0, 40.0, 81)
    target = {"X_inf": 3.2, "tau": 7.5}
    y = target["X_inf"] * (1.0 - np.exp(-t / target["tau"]))
    fit = ra.exponential_growth_fit(t, y)
    assert fit["X_inf"] == pytest.approx(target["X_inf"], rel=1e-3)
    assert fit["tau"] == pytest.approx(target["tau"], rel=1e-3)
    assert fit["r2"] > 0.999


def test_loglog_exponent_recovers_slope_with_standard_error():
    x = np.array([0.1, 0.2, 0.4, 0.8, 1.6])
    y = 2.5 * x ** 2.0
    out = ra.loglog_exponent(x, y)
    assert out["exponent"] == pytest.approx(2.0, abs=1e-9)
    assert out["r2"] == pytest.approx(1.0, abs=1e-12)
    assert out["stderr"] < 1e-9
    noisy = ra.loglog_exponent(x, y * np.array([1.0, 1.1, 0.9, 1.05, 0.95]))
    assert noisy["stderr"] > 0.0
    assert math.isnan(ra.loglog_exponent(np.array([0.0, -1.0]), np.array([1.0, 2.0]))["exponent"])


def test_momentum_balance_reference_is_internally_consistent():
    cond = ra.IceConditions(velocity=0.3, thickness=0.3, concentration=0.3)
    ref = ra.momentum_balance_reference(cond)
    mat = ra.RubbleMaterial()
    # H_inf/h_i = C Fr^2 / (mu (1-n) (rho_w/rho_i - 1)) recomputed by hand
    manual = (cond.concentration * cond.froude ** 2
              / (mat.mu_ice * (1.0 - mat.porosity) * (mat.density_water / mat.density_ice - 1.0)))
    assert ref["H_inf_over_h_i"] == pytest.approx(manual, rel=1e-12)
    # doubling the speed quadruples the predicted pile height (Fr^2 scaling)
    ref2 = ra.momentum_balance_reference(
        ra.IceConditions(velocity=2 * cond.velocity, thickness=cond.thickness,
                         concentration=cond.concentration))
    assert ref2["H_inf_over_h_i"] == pytest.approx(4.0 * ref["H_inf_over_h_i"], rel=1e-12)
    # the strength stress is the stress the pile can carry
    assert ref["strength_stress"] == pytest.approx(
        mat.mu_ice * (1.0 - mat.porosity) * (mat.density_water - mat.density_ice)
        * ra.GRAVITY * ref["H_inf_over_h_i"] * cond.thickness, rel=1e-12)


# --------------------------------------------------------------------------- #
# 2. conservation and numerical guards
# --------------------------------------------------------------------------- #
def test_mass_ledger_closes_and_loses_no_piece():
    cfg = small_config(duration=8.0, sample_dt=1.0)
    run = ra.run_accumulation(cfg)
    last = run.samples[-1]
    assert abs(last["mass_total_drift"]) <= 1e-9 * max(last["mass_total_initial"], 1.0)
    closures = [s["mass_closure_relative"] for s in run.samples
                if np.isfinite(s["mass_closure_relative"])]
    assert closures, "the ledger must have produced a closure number"
    assert max(closures) < 0.02
    assert last["mass_trapped"] == 0.0


def test_penetration_guard_stays_bounded():
    # the report's runs use kn = 3e5 N/m, the value the pre-registered 5%
    # penetration guard was checked against
    cfg = small_config(duration=8.0, sample_dt=1.0,
                       material=ra.RubbleMaterial(kn=3.0e5))
    run = ra.run_accumulation(cfg)
    assert run.samples[-1]["max_penetration_ratio"] < 0.05


def test_run_is_deterministic_for_a_fixed_seed():
    cfg = small_config(duration=4.0, sample_dt=1.0, seed=7)
    first = ra.run_accumulation(cfg)
    second = ra.run_accumulation(cfg)
    assert first.samples[-1]["pile_mass"] == second.samples[-1]["pile_mass"]
    assert first.samples[-1]["p_front"] == second.samples[-1]["p_front"]
    other = ra.run_accumulation(replace(cfg, seed=8))
    assert other.samples[-1]["pile_mass"] != first.samples[-1]["pile_mass"]


def test_generated_field_has_no_initial_overlap_and_hits_the_target_fraction():
    domain = ra.DomainSpec(length=10.0, width=8.0)
    structure = ra.StructureSpec(diameter=1.0, x_center=6.0, y_center=4.0)
    for concentration in (0.1, 0.3, 0.5):
        field = ra.generate_piece_field(domain, structure, concentration=concentration,
                                        d_min=0.1, d_max=0.3, seed=3)
        assert field.areal_fraction == pytest.approx(concentration, rel=1e-6)
        # no overlap beyond the relaxation tolerance, including across the seam
        aug = np.concatenate([field.positions,
                              field.positions + np.array([domain.length, 0.0]),
                              field.positions - np.array([domain.length, 0.0])])
        from scipy.spatial import cKDTree
        tree = cKDTree(aug)
        pairs = tree.query_pairs(2.0 * field.radii.max(), output_type="ndarray")
        bad = 0
        for i, j in pairs:
            if i >= field.n:
                continue
            gap = (field.radii[i] + field.radii[j % field.n]
                   - np.linalg.norm(aug[j] - field.positions[i]))
            if gap > 1e-2:
                bad += 1
        assert bad == 0


def test_wall_and_structure_contacts_are_repulsive():
    """A disk pushed into the bank or the cylinder must be pushed back out."""
    domain = ra.DomainSpec(length=10.0, width=6.0)
    structure = ra.StructureSpec(diameter=1.0, x_center=5.0, y_center=3.0)
    cfg = small_config(domain=domain, structure=structure,
                       conditions=ra.IceConditions(velocity=0.0, thickness=0.2,
                                                   concentration=0.4))
    pos = np.array([[5.2, 3.0], [5.0, 0.01], [5.0, 5.99]])
    rad = np.array([0.2, 0.2, 0.2])
    sim = ra.RubbleAccumulation(ra.PieceField(pos, rad, domain), cfg)
    force = sim.forces()
    assert force[0, 0] > 0.0, "disk inside the cylinder must be pushed downstream"
    assert force[1, 1] > 0.0, "disk at the lower bank must be pushed up"
    assert force[2, 1] < 0.0, "disk at the upper bank must be pushed down"


def test_pair_dashpot_opposes_the_relative_velocity():
    """Sign convention guard: the normal dashpot must damp, never drive.

    A wrong dashpot sign (a classic DEM bug) makes contact forces positive while
    a pair separates, i.e. it *injects* energy.  Here the pair force must have a
    negative projection on the relative velocity.
    """
    domain = ra.DomainSpec(length=10.0, width=6.0)
    structure = ra.StructureSpec(diameter=1.0, x_center=9.0, y_center=5.0)
    cfg = small_config(domain=domain, structure=structure,
                       conditions=ra.IceConditions(velocity=0.0, thickness=0.2,
                                                   concentration=0.4))
    pos = np.array([[3.0, 3.0], [3.0 + 0.2 + 0.2 - 0.005, 3.0]])
    rad = np.array([0.2, 0.2])
    sim = ra.RubbleAccumulation(ra.PieceField(pos, rad, domain), cfg)
    sim.vel[0] = [0.1, 0.0]
    sim.vel[1] = [-0.1, 0.0]
    i, j, normal, overlap, fn, ft, tangent = sim._pair_contacts()
    assert len(i) == 1 and fn[0] > 0.0
    f_pair = fn[0] * normal[0] + ft[0] * tangent[0]
    power = float(np.dot(-f_pair, sim.vel[i[0]]) + np.dot(f_pair, sim.vel[j[0]]))
    assert power < 0.0, "contact power must be dissipative"


def test_total_mechanical_energy_never_grows_in_a_closed_pair():
    """Spring + kinetic energy of an isolated pair must not increase."""
    domain = ra.DomainSpec(length=10.0, width=6.0)
    structure = ra.StructureSpec(diameter=0.2, x_center=9.7, y_center=5.7)
    cfg = small_config(domain=domain, structure=structure,
                       conditions=ra.IceConditions(velocity=0.0, thickness=0.2,
                                                   concentration=0.4))
    pos = np.array([[3.0, 3.0], [3.0 + 0.2 + 0.2 - 0.005, 3.0]])
    rad = np.array([0.2, 0.2])
    sim = ra.RubbleAccumulation(ra.PieceField(pos, rad, domain), cfg)

    def total_energy() -> float:
        ke = float(np.sum(0.5 * sim.mass * np.sum(sim.vel ** 2, axis=1)))
        _i, _j, _n, ov, _fn, _ft, _t = sim._pair_contacts()
        spring = 0.5 * sim.material.kn * float(np.sum(ov ** 2))
        return ke + spring

    sim.vel[0] = [0.1, 0.0]
    sim.vel[1] = [-0.1, 0.0]
    e0 = total_energy()
    for _ in range(4):
        sim.step(25)
        assert total_energy() <= e0 * (1.0 + 1e-9)


# --------------------------------------------------------------------------- #
# 3. controls and regression
# --------------------------------------------------------------------------- #
def test_negative_control_zero_concentration_gives_no_pile():
    cfg = small_config(duration=4.0, sample_dt=1.0,
                       conditions=ra.IceConditions(velocity=0.3, thickness=0.2,
                                                   concentration=0.0))
    run = ra.run_accumulation(cfg)
    assert run.field["n_pieces"] == 0.0
    assert max(s["pile_mass"] for s in run.samples) == 0.0


def test_negative_control_zero_velocity_gives_no_pile():
    cfg = small_config(duration=6.0, sample_dt=1.0,
                       conditions=ra.IceConditions(velocity=0.0, thickness=0.2,
                                                   concentration=0.4))
    run = ra.run_accumulation(cfg)
    assert max(s["pile_discs"] for s in run.samples) == 0.0
    assert max(s["L_p"] for s in run.samples) == 0.0
    assert abs(run.samples[-1]["p_front"]) < 1e-9


def test_pile_grows_when_the_field_drifts():
    cfg = small_config(duration=10.0, sample_dt=1.0,
                       conditions=ra.IceConditions(velocity=0.3, thickness=0.2,
                                                   concentration=0.5))
    run = ra.run_accumulation(cfg)
    masses = [s["pile_mass"] for s in run.samples]
    assert max(masses[-3:]) > 0.0
    assert masses[-1] >= masses[0]


def test_load_definition_is_width_and_thickness_normalised():
    cfg = small_config(duration=2.0, sample_dt=1.0)
    field = ra.generate_piece_field(cfg.domain, cfg.structure, concentration=0.4,
                                    d_min=cfg.d_min, d_max=cfg.d_max, seed=0)
    sim = ra.RubbleAccumulation(field, cfg)
    sim.step(50)
    sim.structure_force_x_history = [10.0] * 10
    sim.dt = 0.01
    p = sim.load(window=0.1)
    assert p == pytest.approx(10.0 / cfg.conditions.thickness / cfg.structure.diameter)


def test_sweep_rows_are_one_factor_at_a_time_and_unique():
    rows = ra.sweep_rows()
    keys = [(r["v_i"], r["h_i"], r["C_i"], r["D"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert {"v_i": 0.3, "h_i": 0.3, "C_i": 0.3, "D": 3.0} in rows
    changed = [sum(1 for k in ("v_i", "h_i", "C_i", "D") if abs(r[k] - 0.3) > 1e-12
                   and not (k == "D" and abs(r[k] - 3.0) < 1e-12)) for r in rows]
    assert max(changed) <= 1