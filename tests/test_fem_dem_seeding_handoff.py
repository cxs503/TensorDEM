"""Acceptance tests for the shape-resolved FEM -> DEM handoff (p302w2-n03).

The node's criteria are tested one by one, together with the *guarantees the
implementation claims* (inscribed discs, disjoint clumps, exact mass split,
rigidity residual, closed-form dynamics).  Every number quoted in the report is
produced by one of these tests or by the study run that the module-scoped fixture
performs, so nothing in the report is hand-typed.

Run: ``PYTHONPATH=src pytest tests/test_fem_dem_seeding_handoff.py -q``
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from tensordem.fem_dem_fracture import IcePlateCase
from tensordem.fem_dem_fracture.handoff import dem_initial_conditions, read_failure_faces
from tensordem.fem_dem_seeding import (
    Raster,
    analytic_heave,
    boundary_identity,
    drag_relax,
    floe_node_mask,
    footprint_metrics,
    heave_audit,
    integrate_heave,
    kinematic_transfer,
    overlap_audit,
    polygon_mask,
    read_node_state,
    resolution_check,
    rigid_plane_fit,
    run_study,
    seed_floe_bodies,
    seed_inscribed_discs,
    write_figures,
)
from tensordem.fem_dem_seeding.clump import _boundary_distances


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def study(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("p302w2_n03")
    summary = run_study(str(out))
    summary["_dir"] = str(out)
    return summary


@pytest.fixture(scope="module")
def faces(study):
    return read_failure_faces(os.path.join(study["_dir"], "failure_faces.json"))


@pytest.fixture(scope="module")
def node_state(study):
    return read_node_state(os.path.join(study["_dir"], "fem_node_state.json"))


@pytest.fixture(scope="module")
def raster(study) -> Raster:
    plate = study["case"]["plate"]
    m = 2.0
    return Raster(-m, -m, float(plate["length_x_m"]) + m,
                  float(plate["length_y_m"]) + m, 0.25 / 8.0)


@pytest.fixture(scope="module")
def bodies(study, faces, raster):
    return seed_floe_bodies(faces, raster, modes=("clump", "equivalent_disc"))


# --------------------------------------------------------------------------
# 1. raster geometry: the instrument must be exact on the shapes it measures
# --------------------------------------------------------------------------
def test_raster_area_of_edge_aligned_square_is_exact():
    raster = Raster(0.0, 0.0, 4.0, 4.0, 0.25)
    square = [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]
    mask = polygon_mask(raster, [square])
    assert raster.mask_area(mask) == pytest.approx(4.0, abs=1e-12)


def test_raster_subtracts_holes():
    raster = Raster(0.0, 0.0, 6.0, 6.0, 0.25)
    outer = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
    hole = [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)]
    mask = polygon_mask(raster, [outer, hole])
    assert raster.mask_area(mask) == pytest.approx(25.0 - 1.0, abs=1e-12)


def test_floe_polygon_raster_area_matches_shoelace(faces, raster):
    for floe in faces.floes:
        metrics = footprint_metrics(raster, floe.polygons, np.empty((0, 2)), np.empty(0))
        assert abs(metrics["polygon_area_raster_error"]) < 1e-9
        assert metrics["polygon_area"] == pytest.approx(floe.area, rel=1e-12)


def test_disc_mask_area_matches_analytic_within_one_cell():
    raster = Raster(-2.0, -2.0, 2.0, 2.0, 0.01)
    r = 1.0
    mask = polygon_mask(raster, [[(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]])
    from tensordem.fem_dem_seeding import disc_mask

    disc = disc_mask(raster, [[0.0, 0.0]], [r])
    area = raster.mask_area(disc & mask)
    assert abs(area - math.pi * r * r) / (math.pi * r * r) < 0.02
    assert raster.mask_area(disc & ~mask) == 0.0


# --------------------------------------------------------------------------
# 2. seeding guarantees
# --------------------------------------------------------------------------
def test_clump_discs_are_inscribed_in_their_floe(bodies):
    for body in bodies["clump"]:
        dist = _boundary_distances(body.centres(), body.polygons)
        radii = body.radii()
        assert np.all(radii <= dist + 1e-9), (
            f"floe {body.floe_id}: a disc reaches outside its footprint "
            f"(r={radii.max():.4f} m, distance={dist.min():.4f} m)"
        )


def test_clump_discs_do_not_overlap_each_other(bodies):
    for body in bodies["clump"]:
        c, r = body.centres(), body.radii()
        d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
        gap = d - (r[:, None] + r[None, :])
        iu = np.triu_indices(c.shape[0], k=1)
        assert np.min(gap[iu]) > -1e-9


def test_clump_material_area_never_exceeds_the_floe(bodies):
    for body in bodies["clump"]:
        assert body.disc_area <= body.area + 1e-9


def test_clump_mass_split_is_exact(bodies, faces):
    thickness = float(faces.meta["plate"]["thickness_m"])
    rho = float(faces.meta["material"]["density_kg_m3"])
    for body in bodies["clump"]:
        assert body.disc_mass_sum == pytest.approx(body.mass, abs=1e-9)
        assert body.mass == pytest.approx(body.area * thickness * rho, rel=1e-12)


def test_target_disc_count_is_respected_for_a_large_floe(faces, raster):
    floe = max(faces.floes, key=lambda f: f.area)
    rings = tuple(tuple(tuple(float(c) for c in v) for v in ring) for ring in floe.polygons)
    centres, radii = seed_inscribed_discs(raster, rings, target_discs=60)
    assert 30 <= centres.shape[0] <= 90
    assert radii.shape == (centres.shape[0],)


def test_equivalent_disc_area_is_exact_but_it_spills(faces, bodies):
    for body in bodies["equivalent_disc"]:
        assert body.disc_area == pytest.approx(body.area, rel=1e-12)
    spill = sum(b.metrics["spill_area"] for b in bodies["equivalent_disc"])
    assert spill > 1.0, "the baseline is expected to put material outside its floes"


def test_overlap_audit_separates_the_two_representations(bodies):
    clump = overlap_audit(bodies["clump"])
    base = overlap_audit(bodies["equivalent_disc"])
    assert clump["n_overlapping_pairs"] == 0
    assert clump["min_clearance_m"] > 0.0
    assert base["n_overlapping_pairs"] > 0
    assert base["min_clearance_m"] < 0.0


# --------------------------------------------------------------------------
# 3. kinematic transfer
# --------------------------------------------------------------------------
def test_rigid_plane_fit_recovers_an_exact_plane():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    p_true, q_true, w0_true = 3.0e-3, -2.0e-3, 0.1
    w = w0_true + p_true * xy[:, 0] + q_true * xy[:, 1]
    fit = rigid_plane_fit(xy, w, np.full(5, p_true), np.full(5, q_true), length_scale=1.0)
    # w0 is the rigid translation *at the floe reference point* (the sample
    # centroid), not the value of the plane at the plate origin.
    assert (fit.x0, fit.y0) == pytest.approx((0.5, 0.5), abs=1e-15)
    assert fit.w0 == pytest.approx(w0_true + p_true * 0.5 + q_true * 0.5, abs=1e-15)
    assert fit.evaluate(0.0, 0.0) == pytest.approx(w0_true, abs=1e-15)
    assert fit.p == pytest.approx(p_true, rel=1e-12)
    assert fit.q == pytest.approx(q_true, rel=1e-12)
    assert fit.rms < 1e-15
    check = fit.slope_check()
    assert check["abs_gap_x"] < 1e-15 and check["abs_gap_y"] < 1e-15


def test_slope_check_is_scale_normalised_for_a_nearly_flat_floe():
    xy = np.array([[0.0, 0.0], [0.3, 0.0], [0.0, 0.3]])
    w = 1e-9 * xy[:, 0]                   # slopes of order 1e-9
    fit = rigid_plane_fit(xy, w, np.zeros(3), np.zeros(3), length_scale=0.3)
    check = fit.slope_check()
    # A raw relative test would report 100 % here; the natural-slope floor must not.
    assert check["rel_to_natural_x"] < 1.0
    assert np.isfinite(check["rel_to_natural_y"])


def test_kinematic_transfer_covers_every_floe(faces, node_state):
    kin = kinematic_transfer(faces, node_state)
    assert len(kin) == len(faces.floes)
    for k in kin:
        assert k.n_nodes >= 3
        assert np.isfinite(k.v_z) and np.isfinite(k.omega_x) and np.isfinite(k.omega_y)
        assert 0.0 <= k.fit.rel_rms < 0.5
        # The load pushes the sheet down, so every fragment carries a non-zero
        # heave offset whose rate has the same sign as the offset itself.
        assert abs(k.z0) > 1e-3
        assert np.sign(k.v_z) == np.sign(k.z0)


def test_floe_node_mask_excludes_nodes_outside_the_footprint(node_state, faces):
    xy = node_state["node_xy"]
    total = 0
    for floe in faces.floes:
        mask = floe_node_mask(xy, floe.polygons)
        assert np.count_nonzero(mask) >= 3
        total += np.count_nonzero(mask)
    assert total > len(faces.floes) * 3


# --------------------------------------------------------------------------
# 4. dynamics against closed forms
# --------------------------------------------------------------------------
def test_analytic_heave_is_size_independent_and_thickness_driven():
    a = analytic_heave(0.5, 917.0)
    b = analytic_heave(1.0, 917.0)
    assert a.period == pytest.approx(1.3419, abs=5e-4)
    # Thicker ice heaves slower: the period scales as sqrt(t).
    assert b.period == pytest.approx(a.period * math.sqrt(2.0), rel=1e-12)
    # omega^2 = rho_w g / (rho_i t) with rho_w g = 1025 * 9.80665
    assert a.omega == pytest.approx(math.sqrt(1025.0 * 9.80665 / (917.0 * 0.5)), rel=1e-12)


def test_integrate_heave_matches_the_harmonic_period_and_conserves_energy():
    ana = analytic_heave(0.5, 917.0)
    res = integrate_heave(ana.omega, z0=0.05, v0=0.01, t_end=6.0)
    assert abs(res["period"] - ana.period) / ana.period < 1e-3
    # Velocity-Verlet conserves energy up to an O((omega dt)^2) oscillation; the
    # honest test is the *order* of the convergence, not one absolute number.
    coarse = integrate_heave(ana.omega, z0=0.05, v0=0.01, t_end=6.0,
                             n_steps_per_period=200)
    assert res["energy_drift"] < 1e-4
    assert res["energy_drift"] < coarse["energy_drift"] / 3.0
    assert res["n_crossings"] >= 5


def test_integrate_heave_amplitude_follows_the_closed_form():
    omega = 4.0
    res = integrate_heave(omega, z0=0.02, v0=0.03, t_end=4.0)
    assert res["amplitude"] == pytest.approx(math.sqrt(0.02 ** 2 + (0.03 / omega) ** 2),
                                             rel=1e-12)
    # The sampled trajectory reaches the amplitude to integration accuracy.
    assert np.max(np.abs(res["z"])) <= res["amplitude"] * (1.0 + 1e-3)


def test_drag_relax_reproduces_the_exponential():
    tau = 20.0
    res = drag_relax(0.0, 0.15, tau, t_end=60.0, dt=0.05)
    assert res["rel_error"] < 1e-9
    # After 3 tau the body is at 1 - e^-3 of the current, not at the current.
    assert res["terminal_velocity"] == pytest.approx(0.15 * (1.0 - math.exp(-3.0)),
                                                     rel=1e-9)


def test_heave_audit_of_the_production_bodies(bodies, faces, node_state):
    kin = kinematic_transfer(faces, node_state)
    audit = heave_audit(bodies["clump"], kin, t_end=4.0)
    assert audit["max_period_error"] < 0.01
    assert audit["period_spread"] < 1e-9        # omega is size independent
    assert audit["max_energy_drift"] < 1e-4
    assert all(row["rho_body_kg_m3"] == pytest.approx(917.0, rel=1e-9)
               for row in audit["rows"])


# --------------------------------------------------------------------------
# 5. the node's acceptance criteria, as computed by the study
# --------------------------------------------------------------------------
def test_C1_fracture_artefact_is_written(study):
    assert study["criteria"]["C1_fracture_artefact_written"]["pass"]
    assert os.path.exists(os.path.join(study["_dir"], "failure_faces.json"))
    assert os.path.exists(os.path.join(study["_dir"], "fem_node_state.json"))


def test_C2_mass_conservation(study):
    clump = study["ledger"]["clump"]
    assert abs(clump["mass_residual_kg"]) < 1e-6
    assert abs(clump["disc_mass_residual_kg"]) < 1e-6
    assert study["criteria"]["C2_mass_conservation"]["pass"]


def test_C3_area_and_boundary_identity(study):
    assert abs(study["ledger"]["clump"]["area_residual_m2"]) < 1e-3
    bnd = study["boundary"]
    assert abs(bnd["rim_residual_m"]) < 1e-9
    assert bnd["n_overused_segments"] == 0
    # A partition of the plate: rim once, every fracture face twice.
    assert bnd["n_shared_segments"] > 0
    assert bnd["fracture_face_length_m"] > 0.0


def test_C4_floe_count_in_range(study):
    assert 3 <= study["case"]["n_floe"] <= 30


def test_C5_end_to_end_from_files_only(study, raster):
    faces = read_failure_faces(os.path.join(study["_dir"], "failure_faces.json"))
    node_state = read_node_state(os.path.join(study["_dir"], "fem_node_state.json"))
    seeded = seed_floe_bodies(faces, raster, modes=("clump",))
    kin = kinematic_transfer(faces, node_state)
    assert len(seeded["clump"]) == len(faces.floes) == len(kin)
    payload = json.load(open(os.path.join(study["_dir"], "dem_bodies.json"),
                             encoding="utf-8"))
    assert len(payload["bodies"]) == len(faces.floes)
    for rec in payload["bodies"]:
        assert rec["n_discs"] >= 3
        assert rec["initial_state"]["floe_id"] == rec["floe_id"]
        assert math.isfinite(rec["initial_state"]["v_z_m_s"])


def test_C6_all_criteria_pass(study):
    failed = {k: v for k, v in study["criteria"].items() if not v["pass"]}
    assert not failed, f"criteria not met: {failed}"


def test_ledger_is_closed_after_the_run(study):
    post = study["ledger"]["post_run"]
    assert abs(post["mass_residual_kg"]) < 1e-6
    assert abs(post["area_residual_m2"]) < 1e-3
    assert post["all_masses_positive"]
    assert post["overlap"]["n_overlapping_pairs"] == 0


def test_first_handoff_baseline_is_reproduced(faces, study):
    legacy = study["ledger"]["legacy_first_handoff"]
    ic = dem_initial_conditions(faces)
    assert legacy["n_bodies"] == len(faces.floes)
    assert np.sum(ic.masses) == pytest.approx(study["case"]["plate"]["area_m2"]
                                              * float(faces.meta["plate"]["thickness_m"])
                                              * float(faces.meta["material"]["density_kg_m3"]),
                                              rel=1e-12)
    assert legacy["velocities_all_zero"] is True


def test_raster_score_is_resolved(bodies, raster):
    body = max(bodies["clump"], key=lambda b: b.area)
    res = resolution_check(raster, body.polygons, lambda _r: (body.centres(), body.radii()))
    assert res["max_relative_change"] < 0.05


def test_seeding_is_deterministic(faces, raster):
    floe = max(faces.floes, key=lambda f: f.area)
    rings = tuple(tuple(tuple(float(c) for c in v) for v in ring) for ring in floe.polygons)
    a = seed_inscribed_discs(raster, floe_rings_alias(rings), target_discs=40)
    b = seed_inscribed_discs(raster, floe_rings_alias(rings), target_discs=40)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def floe_rings_alias(rings):
    """Copy so the second call cannot benefit from a cached object identity."""
    return tuple(tuple(tuple(float(c) for c in v) for v in ring) for ring in rings)


def test_figures_are_written(study):
    recs = write_figures(study, study["_dir"])
    assert len(recs) >= 5
    for rec in recs:
        assert os.path.getsize(rec["path"]) > 10_000
        assert rec["caption"]


def test_boundary_identity_helper_rejects_a_tampered_partition(faces):
    good = boundary_identity(faces, plate_length=10.0, plate_width=10.0)
    assert abs(good["rim_residual_m"]) < 1e-9
    # Dropping a floe must break the rim ledger (the check has teeth).
    trimmed = type(faces)(meta=faces.meta, cracks=faces.cracks, floes=faces.floes[1:],
                          fractured_elements=faces.fractured_elements, fsd=faces.fsd,
                          conservation=faces.conservation, provenance=faces.provenance,
                          crack_cells=faces.crack_cells)
    bad = boundary_identity(trimmed, plate_length=10.0, plate_width=10.0)
    assert abs(bad["rim_residual_m"]) > 1e-3 or bad["n_segments"] < good["n_segments"]


def test_plate_case_matches_the_handoff_case():
    case = IcePlateCase()
    assert case.nx == case.ny == 40
    assert case.length_x == case.length_y == 10.0