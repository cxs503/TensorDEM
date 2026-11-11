"""Tests for the breakable buoyant floe DEM (ice ridge rubbling)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ice_rafting.mechanics import IceProperties
from tensordem.ice_ridge_formation.rubbling import (
    RidgeDEMConfig,
    RubbleFloe,
    RubbleWorld,
    fit_porosity_law,
    generate_floe_field,
    polygon_area,
    polygon_moments,
    ridge_porosity,
    run_aspect_sweep,
    run_rubbling_ridge,
    sail_keel_geometry,
    split_convex_polygon,
)


# ---------------------------------------------------------------- geometry --
def test_polygon_area_rectangle():
    poly = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 0.5], [0.0, 0.5]])
    assert polygon_area(poly) == pytest.approx(2.0)
    assert polygon_area(poly[::-1]) == pytest.approx(-2.0)


def test_polygon_moments_centroid_and_inertia():
    poly = np.array([[-2.0, -0.25], [2.0, -0.25], [2.0, 0.25], [-2.0, 0.25]])
    area, centroid, i_c = polygon_moments(poly)
    assert area == pytest.approx(2.0)
    assert centroid == pytest.approx([0.0, 0.0], abs=1e-12)
    # polar second moment per unit mass of a rectangle about its centre
    assert i_c == pytest.approx((16.0 + 0.25) / 12.0, rel=1e-12)


def test_split_conserves_area_and_keeps_convexity():
    poly = np.array([[-3.0, -0.25], [3.0, -0.25], [3.0, 0.25], [-3.0, 0.25]])
    a, b = split_convex_polygon(poly, [0.3, 0.0], [1.0, 0.0])
    assert len(a) >= 3 and len(b) >= 3
    assert polygon_area(a) + polygon_area(b) == pytest.approx(polygon_area(poly), rel=1e-12)
    # both halves stay convex (all cross products keep one sign)
    for piece in (a, b):
        p = np.asarray(piece)
        cr = np.cross(np.roll(p, -1, axis=0) - p, np.roll(p, -2, axis=0) - np.roll(p, -1, axis=0))
        assert np.all(cr >= -1e-12) or np.all(cr <= 1e-12)


def test_body_properties_and_float_equilibrium():
    ice = IceProperties()
    h = 0.5
    poly = np.array([[-2.0, -h / 2], [2.0, -h / 2], [2.0, h / 2], [-2.0, h / 2]])
    body = RubbleFloe(poly, ice, node_spacing=0.05)
    assert body.area == pytest.approx(2.0)
    assert body.mass == pytest.approx(ice.rho_ice * 2.0)
    assert body.thickness == pytest.approx(h, rel=1e-9)
    assert len(body.nodes_local) > 10
    assert body.node_radius == pytest.approx(0.025)


# --------------------------------------------------------------- statics --
def test_hydrostatic_equilibrium_is_exact_and_stable():
    """A floating floe must sit at the analytic draft and not move."""
    ice = IceProperties()
    h = 0.5
    sub = h * ice.draft_ratio
    poly = np.array([[-2.0, -h / 2], [2.0, -h / 2], [2.0, h / 2], [-2.0, h / 2]])
    poly = poly + np.array([0.0, 0.5 * h - sub])
    body = RubbleFloe(poly, ice, node_spacing=0.03)
    world = RubbleWorld([body], ice, x_left=-1e6, piston_x=1e6,
                        piston_velocity=0.0, node_spacing=0.03)
    z0 = body.pos[1]
    f_b, _, area_sub = world._buoyancy(body)
    assert f_b[1] == pytest.approx(body.mass * ice.gravity, rel=1e-12)
    assert area_sub / body.area == pytest.approx(ice.draft_ratio, rel=1e-9)
    world.step(300)
    assert body.pos[1] == pytest.approx(z0, abs=1e-9)
    assert world.kinetic_energy() == pytest.approx(0.0, abs=1e-9)


def test_buoyancy_centre_moves_when_the_floe_tilts():
    ice = IceProperties()
    body = generate_floe_field(RidgeDEMConfig(), lengths=[4.0])[0]
    world = RubbleWorld([body], ice, x_left=-1e6, piston_x=1e6,
                        piston_velocity=0.0, node_spacing=0.05)
    _, tq0, _ = world._buoyancy(body)
    assert tq0 == pytest.approx(0.0, abs=1e-9)
    body.theta = 0.2
    f_b, tq1, area_sub = world._buoyancy(body)
    assert area_sub < body.area        # part of the floe lifts out of the water
    assert abs(tq1) > 1e-6             # the centre of buoyancy is off the centroid


# -------------------------------------------------------------- bending ----
def test_bending_stress_matches_beam_formula():
    """sigma = 6 M / t^2 for a simply loaded beam of known moment."""
    ice = IceProperties()
    body = RubbleFloe(np.array([[-4.0, -0.25], [4.0, -0.25], [4.0, 0.25], [-4.0, 0.25]]),
                      ice, node_spacing=0.1)
    world = RubbleWorld([body], ice, x_left=-1e6, piston_x=1e6,
                        piston_velocity=0.0, node_spacing=0.1)
    F = 1.0e4
    # two contacts, one at each end, acting downwards -> moment F * L / 2 at the centre
    log = [(np.array([-4.0, 0.25]), np.array([0.0, -F])),
           (np.array([4.0, 0.25]), np.array([0.0, -F]))]
    sigma = world.bending_stress(body, log)
    expected = 6.0 * (F * 4.0) / body.thickness ** 2
    assert sigma == pytest.approx(expected, rel=1e-9)


def test_single_point_contact_does_not_bend():
    ice = IceProperties()
    body = RubbleFloe(np.array([[-4.0, -0.25], [4.0, -0.25], [4.0, 0.25], [-4.0, 0.25]]),
                      ice, node_spacing=0.1)
    world = RubbleWorld([body], ice, x_left=-1e6, piston_x=1e6,
                        piston_velocity=0.0, node_spacing=0.1)
    log = [(np.array([3.9, 0.25]), np.array([0.0, -1e6]))]
    assert world.bending_stress(body, log) == 0.0


def test_fracture_splits_area_and_keeps_momentum():
    ice = IceProperties()
    poly = np.array([[-4.0, -0.25], [4.0, -0.25], [4.0, 0.25], [-4.0, 0.25]])
    body = RubbleFloe(poly, ice, node_spacing=0.1)
    body.vel = np.array([0.3, -0.1])
    world = RubbleWorld([body], ice, x_left=-1e6, piston_x=1e6,
                        piston_velocity=0.0, node_spacing=0.1,
                        min_block_area=0.01, min_block_thickness=0.05)
    area0 = world.total_area()
    momentum0 = body.mass * body.vel.copy()
    a, b = world._split_body(body, [], np.random.default_rng(1))
    assert a.area + b.area == pytest.approx(area0, rel=1e-12)
    assert a.thickness == pytest.approx(0.5, rel=2e-3)
    mom = sum(p.mass * p.vel for p in (a, b))
    assert mom == pytest.approx(momentum0, rel=1e-2, abs=1e-2)


# -------------------------------------------------------------- measures ---
def test_ridge_porosity_of_a_known_packing():
    """Two stacked blocks: measured porosity must equal the geometric vacancy."""
    ice = IceProperties()
    cfg = RidgeDEMConfig()
    lower = RubbleFloe(np.array([[0.0, -0.25], [2.0, -0.25], [2.0, 0.25], [0.0, 0.25]]),
                       ice, node_spacing=0.01)
    upper = RubbleFloe(np.array([[0.0, 1.0], [2.0, 1.0], [2.0, 1.5], [0.0, 1.5]]),
                       ice, node_spacing=0.01)
    por = ridge_porosity([lower, upper], 0.5, dx=0.05)
    # envelope = 1.75 m tall x 2 m wide, ice = 2 x 1 m^2 -> n = 1 - 2 / 3.5
    assert por["porosity"] == pytest.approx(1.0 - 2.0 / 3.5, abs=0.02)
    assert por["ice_area"] == pytest.approx(2.0, rel=1e-6)
    assert por["ridge_width"] > 0.0


def test_porosity_law_fit_recovers_parameters():
    eps = np.linspace(0.0, 0.3, 12)
    n = 0.45 * np.exp(-eps / 0.18)
    fit = fit_porosity_law(eps, n)
    assert fit["n0"] == pytest.approx(0.45, rel=1e-6)
    assert fit["eps_c"] == pytest.approx(0.18, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)


def test_porosity_fit_does_not_invent_points():
    fit = fit_porosity_law([0.0, 0.1], [0.4, float("nan")])
    assert fit["n_points"] == 1
    assert math.isnan(fit["r2"])


# ---------------------------------------------------------------- runner ---
def test_config_validation():
    with pytest.raises(ValueError):
        RidgeDEMConfig(n_floes=1)
    with pytest.raises(ValueError):
        RidgeDEMConfig(target_convergence=0.95)


def test_floe_field_is_power_law_sized_and_floating():
    cfg = RidgeDEMConfig(n_floes=10, min_floe_length=2.0, max_floe_length=8.0)
    floes = generate_floe_field(cfg)
    lengths = np.array([f.length for f in floes])
    assert lengths.min() >= 1.99 and lengths.max() <= 8.01
    assert np.all(np.diff(lengths) >= 0)
    z0 = 0.5 * cfg.ice_thickness - cfg.ice_thickness * cfg.ice.draft_ratio
    # the bevelled bottom corners shift the centroid up by < 2 % of the thickness
    assert all(abs(f.pos[1] - z0) < 0.05 * cfg.ice_thickness for f in floes)


def test_short_compression_conserves_volume_and_stays_finite():
    res = run_rubbling_ridge(
        n_floes=4, min_floe_length=2.0, max_floe_length=4.0,
        servo_force=2.0e4, servo_velocity=0.02, mass_scale=4.0,
        settle_steps=200, sample_every=400, max_steps=2000,
        target_convergence=0.5)
    assert res["n_steps"] <= 2000
    assert res["volume"]["rel_error"] <= 1e-12
    assert res["final"]["total_area"] == pytest.approx(res["volume"]["initial_area"])
    assert np.isfinite(res["final"]["sail_height"])
    assert np.isfinite(res["final"]["keel_depth"])
    assert res["final"]["n_blocks"] >= 4


def test_sail_keel_geometry_is_measured_from_blocks():
    ice = IceProperties()
    cfg = RidgeDEMConfig()
    a = generate_floe_field(cfg, lengths=[2.0])[0]
    b = RubbleFloe(a.vertices() + np.array([0.0, 3.0]), ice, node_spacing=0.05)
    geom = sail_keel_geometry([a, b])
    assert geom["sail_height"] > 2.0
    assert geom["keel_depth"] > 0.0
    assert geom["n_blocks"] == 2
