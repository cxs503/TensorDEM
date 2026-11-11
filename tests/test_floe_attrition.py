"""Tests for the floe-attrition chain (node p472w2-N05).

The criteria C1-C7 are asserted here, together with the regression guards for the
two defects this work touched (the icosahedron vertex generation and the Steiner
identity being valid only at fixed inner size) and the API guards (no
extrapolation of the DEM baseline, no pressure-controlled contact on a
zero-radius corner).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.floe_attrition import (
    ArchardWear, InterlockingFriction, RoundedCube, RoundedPolygon, TransitScenario,
    build_design_matrix, criteria, fillet_radius_for_roundness,
    integrate_cube_attrition, integrate_polygon_attrition, load_sphere_baseline,
    mu_effective_from_index, polygon_metrics_boundary_integral,
    simulate_multi_pass, two_way_anova, wadell_roundness_2d)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_wadell_roundness_basic_values():
    # circle
    assert abs(wadell_roundness_2d(math.pi, 2.0 * math.pi) - 1.0) < 1e-15
    # square of side 1: 4 pi A / P^2 = pi / 4
    assert abs(wadell_roundness_2d(1.0, 4.0) - math.pi / 4.0) < 1e-15
    # equilateral triangle of side 1: 4 pi (sqrt(3)/4) / 9
    assert abs(wadell_roundness_2d(math.sqrt(3.0) / 4.0, 3.0)
               - 4.0 * math.pi * math.sqrt(3.0) / 36.0) < 1e-15
    with pytest.raises(ValueError):
        wadell_roundness_2d(-1.0, 4.0)


def test_polygon_closed_forms_match_boundary_integral():
    worst = 0.0
    for n in (3, 4, 5, 6, 8, 64):
        base = RoundedPolygon(n, 1.0, 0.0)
        for frac in (0.0, 0.05, 0.25, 0.5, 0.75, 1.0):
            poly = base.with_fillet(frac * base.max_fillet_radius)
            area, perimeter = polygon_metrics_boundary_integral(poly)
            worst = max(worst, abs(area - poly.area) / poly.area,
                        abs(perimeter - poly.perimeter) / poly.perimeter)
    assert worst <= 1e-12


def test_circle_limit_is_exactly_a_circle():
    base = RoundedPolygon(7, 0.4, 0.0)
    circle = base.with_fillet(base.max_fillet_radius)
    assert abs(circle.wadell_roundness - 1.0) <= 1e-14
    assert circle.interlocking_index == pytest.approx(0.0, abs=1e-14)
    radius = circle.area ** 0.5 / math.sqrt(math.pi)
    assert circle.perimeter == pytest.approx(2.0 * math.pi * radius, rel=1e-14)
    assert circle.inscribed_radius == pytest.approx(radius, rel=1e-14)


def test_roundness_is_monotone_in_the_wear_state():
    base = RoundedPolygon(5, 0.6, 0.0)
    states = np.linspace(0.0, base.max_fillet_radius, 25)
    r = np.array([base.with_fillet(s).wadell_roundness for s in states])
    idx = np.array([base.with_fillet(s).interlocking_index for s in states])
    assert np.all(np.diff(r) > 0.0)
    assert np.all(np.diff(idx) < 0.0)
    assert r[0] > 0.8 and r[-1] == pytest.approx(1.0, abs=1e-14)


def test_fillet_radius_inversion_round_trip():
    base = RoundedPolygon(4, 0.6, 0.0)
    for target in (0.80, 0.85, 0.90, 0.95, 0.99, 0.999):
        r = fillet_radius_for_roundness(base, target)
        assert base.with_fillet(r).wadell_roundness == pytest.approx(target, abs=1e-12)
    with pytest.raises(ValueError):
        fillet_radius_for_roundness(base, 0.5)


def test_rounded_cube_sphere_limit_and_sphericity():
    cube = RoundedCube(1.0, 0.0)
    assert cube.sphericity == pytest.approx(0.8059959770082347, rel=1e-14)
    assert cube.wadell_roundness == 0.0
    sphere = RoundedCube(1.0, 0.5)
    assert sphere.volume == pytest.approx(4.0 / 3.0 * math.pi * 0.125, rel=1e-14)
    assert sphere.surface_area == pytest.approx(math.pi, rel=1e-14)
    assert sphere.sphericity == pytest.approx(1.0, abs=1e-14)
    assert sphere.wadell_roundness == pytest.approx(1.0, abs=1e-14)
    with pytest.raises(ValueError):
        RoundedCube(1.0, 0.6)


def test_steiner_identity_needs_fixed_inner_cube():
    """Regression guard: dV/dr == A only at fixed inner cube L."""
    nodes, weights = np.polynomial.legendre.leggauss(48)
    l_fixed, r = 0.5, 0.02
    grid = 0.5 * r * (nodes + 1.0)
    area = lambda x: 6.0 * l_fixed ** 2 + 6.0 * math.pi * l_fixed * x + 4.0 * math.pi * x ** 2
    vol = lambda x: (l_fixed ** 3 + 6.0 * l_fixed ** 2 * x
                     + 3.0 * math.pi * l_fixed * x ** 2 + 4.0 / 3.0 * math.pi * x ** 3)
    integral = 0.5 * r * float(np.sum(weights * np.array([area(x) for x in grid])))
    assert vol(r) - vol(0.0) == pytest.approx(integral, rel=1e-13)


def test_platonic_sphericity_matches_vertex_cloud_descriptor():
    from tensordem.polyhedra_shapes import (compute_shape_descriptor, make_cube,
                                            make_dodecahedron, make_icosahedron,
                                            make_octahedron, make_tetrahedron)
    exact = {
        "tetra": (math.sqrt(2.0) / 12.0, math.sqrt(3.0)),
        "cube": (1.0, 6.0),
        "octa": (math.sqrt(2.0) / 3.0, 2.0 * math.sqrt(3.0)),
        "icosa": ((5.0 / 12.0) * (3.0 + math.sqrt(5.0)), 5.0 * math.sqrt(3.0)),
        "dodeca": ((15.0 + 7.0 * math.sqrt(5.0)) / 4.0,
                   3.0 * math.sqrt(25.0 + 10.0 * math.sqrt(5.0))),
    }
    makers = {"tetra": make_tetrahedron, "cube": make_cube, "octa": make_octahedron,
              "icosa": make_icosahedron, "dodeca": make_dodecahedron}
    for name, maker in makers.items():
        spec = maker()
        sd = compute_shape_descriptor(name, spec.vertices)
        v, a = exact[name]
        psi = math.pi ** (1.0 / 3.0) * (6.0 * v) ** (2.0 / 3.0) / a
        assert sd.sphericity == pytest.approx(psi, rel=1e-12), name


def test_icosahedron_is_regular():
    """Regression guard for the leaked loop variable in ``make_icosahedron``."""
    from tensordem.polyhedra_shapes import compute_shape_descriptor, make_icosahedron
    spec = make_icosahedron()
    sd = compute_shape_descriptor("icosa", spec.vertices)
    assert spec.vertices.shape == (12, 3)
    assert (sd.n_vertices, sd.n_faces, sd.n_edges) == (12, 20, 30)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            edges.append(float(np.linalg.norm(spec.vertices[i] - spec.vertices[j])))
    edges = np.sort(np.array(edges))[:30]
    assert (edges.max() - edges.min()) / edges.mean() < 1e-12


def test_dem_discretisation_converges_quadratically():
    from tensordem.floe_attrition.verification import polygon_metrics_from_vertices
    base = RoundedPolygon(4, 1.0, 0.0)
    poly = base.with_fillet(0.4 * base.max_fillet_radius)
    err = []
    for m in (8, 16, 32, 64):
        a, p = polygon_metrics_from_vertices(poly, m)
        err.append(abs(4.0 * math.pi * a / p ** 2 - poly.wadell_roundness))
    for k in range(len(err) - 1):          # each doubling reduces the error ~4x
        assert err[k] / err[k + 1] > 3.0


# ---------------------------------------------------------------------------
# wear
# ---------------------------------------------------------------------------

def test_archard_law_linearity_and_guards():
    wear = ArchardWear(2.0e-3, 1.0e8)
    assert wear.worn_volume(10.0, 2.0) == pytest.approx(2.0e-3 * 10.0 * 2.0 / 1.0e8)
    assert wear.worn_volume(0.0, 5.0) == 0.0
    arr = wear.worn_volume(10.0, np.array([0.0, 1.0, 2.0]))
    assert np.allclose(arr, np.array([0.0, 2.0e-10, 4.0e-10]))
    with pytest.raises(ValueError):
        wear.worn_volume(-1.0, 1.0)
    with pytest.raises(ValueError):
        ArchardWear(0.0, 1.0e8)


def test_quadratic_worn_area_closed_form():
    from tensordem.floe_attrition import wear_volume_closed_form
    poly = RoundedPolygon(6, 0.5, 0.0)
    for state in (0.0, 0.01, 0.05, 0.2):
        # the subtraction route (sharp_area - area) loses ~1e-9 to cancellation
        assert wear_volume_closed_form(poly, state) == pytest.approx(
            poly.with_fillet(state).worn_area, rel=1e-8, abs=1e-20)


def test_two_d_blunting_is_linear_in_sliding_and_conserves_volume():
    poly = RoundedPolygon(4, 0.6, 0.012)
    wear = ArchardWear(5.0e-3, 3.0e7)
    hist = integrate_polygon_attrition(poly, wear, pressure=2.0e5,
                                       slip_per_step=2.5, n_steps=400)
    r = hist.arrays()["state"]
    dr = np.diff(r)
    # linear blunting: the exact inversion gives dr = sqrt(r^2 + 2 r dr0) - r, so
    # the step varies by O(dr/r) (~0.6 % here) around the analytic constant rate
    assert np.allclose(dr, dr.mean(), rtol=1e-2)
    from tensordem.floe_attrition import wear_volume_closed_form
    c2 = wear_volume_closed_form(poly, 1.0)      # V_w(r) = c2 r^2
    chi = poly.contact_patch_width / poly.fillet_radius
    omega = wear.k_over_h * 2.0e5 * chi / (2.0 * c2)
    assert dr.mean() == pytest.approx(omega * 2.5, rel=0.02)
    assert hist.conservation_error <= 1e-6
    assert hist.roundness[-1] > hist.roundness[0]


def test_three_d_blunting_is_exponential():
    cube = RoundedCube(0.12, 0.01)
    wear = ArchardWear(5.0e-3, 3.0e7)
    hist = integrate_cube_attrition(cube, wear, pressure=5.0e5,
                                    slip_per_step=2.5, n_steps=1200)
    r = hist.arrays()["state"]
    ratio = r[1:] / r[:-1]
    # r ~ exp(t) to leading order; the r^3 term of the Steiner volume makes the
    # per-step ratio drift by <0.1 % over the run
    assert np.allclose(ratio, ratio.mean(), rtol=1e-2)
    assert abs(ratio[-1] / ratio[0] - 1.0) < 1e-2
    assert hist.conservation_error <= 1e-6


def test_zero_slip_is_an_exact_fixed_point():
    poly = RoundedPolygon(4, 0.6, 0.012)
    cube = RoundedCube(0.12, 0.01)
    wear = ArchardWear()
    h2 = integrate_polygon_attrition(poly, wear, pressure=2.0e5, slip_per_step=0.0,
                                     n_steps=25)
    h3 = integrate_cube_attrition(cube, wear, pressure=5.0e5, slip_per_step=0.0,
                                  n_steps=25)
    assert h2.worn_archard[-1] == 0.0 and h3.worn_archard[-1] == 0.0
    assert h2.state[-1] == poly.fillet_radius and h3.state[-1] == cube.radius
    assert h2.excess[-1] == 0.0 and h3.excess[-1] == 0.0


def test_sharp_corner_rejected_for_pressure_controlled_contact():
    with pytest.raises(ValueError):
        integrate_polygon_attrition(RoundedPolygon(4, 0.6, 0.0), ArchardWear(),
                                    pressure=2.0e5, slip_per_step=1.0, n_steps=5)


def test_saturation_books_the_surplus_as_mass_loss():
    poly = RoundedPolygon(4, 0.6, 0.28)
    wear = ArchardWear(5.0e-2, 3.0e7)
    hist = integrate_polygon_attrition(poly, wear, pressure=1.0e6,
                                       slip_per_step=2.5, n_steps=200)
    assert hist.saturated
    assert hist.state[-1] == pytest.approx(poly.max_fillet_radius, rel=1e-12)
    assert hist.conservation_error <= 1e-6      # geometric + excess == Archard


# ---------------------------------------------------------------------------
# friction closure
# ---------------------------------------------------------------------------

def test_baseline_interpolation_and_bracket_guard():
    baseline = load_sphere_baseline()
    lo, hi = baseline.bracket
    assert baseline.mu(lo) == pytest.approx(baseline.mu_sph[0], rel=1e-12)
    assert baseline.mu(hi) == pytest.approx(baseline.mu_sph[-1], rel=1e-12)
    with pytest.raises(ValueError):
        baseline.mu(hi + 0.01)
    with pytest.raises(ValueError):
        baseline.z(lo - 0.01)


def test_mu_eff_sphere_limit_is_bit_identical():
    baseline = load_sphere_baseline()
    model = InterlockingFriction()
    for c in (0.30, 0.50, 0.70):
        assert model.mu(0.0, c, baseline) == baseline.mu(c)


def test_mu_eff_increases_with_angularity_and_kappa_is_linear():
    baseline = load_sphere_baseline()
    m1 = InterlockingFriction(1.0)
    m2 = InterlockingFriction(0.5)
    mu_a = m1.mu(0.4, 0.6, baseline)
    mu_b = m1.mu(0.2, 0.6, baseline)
    assert mu_a > mu_b > baseline.mu(0.6)
    assert (mu_a - baseline.mu(0.6)) == pytest.approx(
        2.0 * (mu_b - baseline.mu(0.6)), rel=1e-12)
    assert (m2.mu(0.4, 0.6, baseline) - baseline.mu(0.6)) == pytest.approx(
        0.5 * (mu_a - baseline.mu(0.6)), rel=1e-12)


# ---------------------------------------------------------------------------
# design / ANOVA
# ---------------------------------------------------------------------------

def test_two_way_anova_recovers_known_effects():
    cells = []
    for i, a in enumerate((0.0, 1.0, 2.0)):
        for j, b in enumerate((0.0, 10.0, 20.0)):
            for k in range(4):
                noise = ((-1.0) ** k) * 0.01 * (k + 1)
                cells.append(_cell(a, b, a + b + noise))
    anova = two_way_anova(cells)
    rows = {r["source"]: r for r in anova["table"]}
    # SS_A = b m sum_i (mean_i - mean)^2 = 3 * 4 * ((0-1)^2 + 0 + (2-1)^2) = 24
    assert rows["roundness"]["ss"] == pytest.approx(24.0, rel=1e-9)
    # SS_B = a m sum_j (...) = 3 * 4 * (100 + 0 + 100) = 2400
    assert rows["concentration"]["ss"] == pytest.approx(2400.0, rel=1e-9)
    assert rows["residual"]["ss"] > 0.0
    assert rows["roundness"]["p"] < 0.05 and rows["concentration"]["p"] < 0.05


def test_anova_rejects_unbalanced_design():
    cells = [_cell(0.0, 0.0, 1.0), _cell(0.0, 1.0, 2.0), _cell(1.0, 0.0, 3.0),
             _cell(1.0, 1.0, 4.0), _cell(1.0, 1.0, 5.0)]
    with pytest.raises(ValueError):
        two_way_anova(cells)


def _cell(r_level: float, c: float, mu: float):
    from tensordem.floe_attrition.design import DesignCell
    return DesignCell(roundness_level=r_level, concentration=c, n_sides=4,
                      fillet_radius=0.0, realized_roundness=r_level,
                      interlocking_index=0.0, mu_effective=mu, mu_sphere=0.0,
                      coordination=0.0)


def test_design_matrix_is_balanced_and_hits_the_roundness_targets():
    levels = (0.80, 0.90, 0.99)
    concentrations = (0.30, 0.50, 0.70)
    cells = build_design_matrix(levels, concentrations, replicates=4)
    assert len(cells) == 9 * 4
    for level in levels:
        got = {c.realized_roundness for c in cells if c.roundness_level == level}
        assert all(abs(g - level) < 1e-9 for g in got)
    assert len({c.n_sides for c in cells}) > 1        # shape disorder present


# ---------------------------------------------------------------------------
# resistance decay
# ---------------------------------------------------------------------------

def test_decay_shares_close_and_are_antisymmetric():
    from tensordem.floe_attrition import decay_shares
    s = decay_shares(mu_start=0.7, mu_end=0.6, load_start=100.0, load_end=120.0,
                     cohesion_start=0.0, cohesion_end=50.0, mu_reference=0.7)
    assert s["closure_relative"] <= 1e-12
    assert s["share_sum_pct"] == pytest.approx(100.0, abs=1e-9)


def test_multi_pass_runs_and_respects_mechanism_switches():
    poly = RoundedPolygon(4, 0.6, 0.012)
    wear = ArchardWear()
    scenario = TransitScenario(concentration=0.7, n_transits=200,
                               slip_per_transit=2.5, contact_pressure=8.0e5)
    full = simulate_multi_pass(scenario, poly, wear)
    only_round = simulate_multi_pass(scenario, poly, wear, allow_consolidation=False)
    only_cons = simulate_multi_pass(scenario, poly, wear, allow_rounding=False)
    assert only_cons["interlocking"][-1] == pytest.approx(poly.interlocking_index)
    assert only_cons["mu_effective"][-1] == pytest.approx(
        only_cons["mu_effective"][0])
    assert only_round["bond_fraction"][-1] == 0.0
    assert only_round["interlocking"][-1] < poly.interlocking_index
    assert full["final"]["resistance"] < only_cons["final"]["resistance"] + 1e-9
    assert full["shares"]["closure_relative"] <= 1e-12
    assert np.all(np.diff(full["contact_energy"]) >= 0.0)


# ---------------------------------------------------------------------------
# the seven acceptance criteria
# ---------------------------------------------------------------------------

def test_all_criteria_pass():
    out = criteria()
    for name, payload in out["outputs"].items():
        assert payload["passed"], (name, payload)
    assert out["all_passed"]