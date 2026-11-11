"""Acceptance tests for the dynamic FEM -> DEM fracture handoff (p302w2-n03).

Every number the report quotes for wave 2 is produced by one of these tests or by
the CLI run they drive; nothing is asserted that is not checked here.

The static chain (schema 1) is covered by ``test_fem_dem_fracture_handoff.py``;
this file covers the wave-2 additions: the inherited velocity field, the rigid
body projection and its conservation identities, the schema-2 artefact with its
tamper proof, the DEM-side intake, and the end-to-end CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os

import numpy as np
import pytest

from tensordem.fem_dem_fracture import (
    IcePlateCase,
    PlateMesh,
    cell_centroid_unit_rates,
    dem_initial_conditions_dynamic,
    dem_intake,
    momentum_ledger,
    project_point_masses,
    project_rigid_body_motion,
    quasi_static_ledger,
    read_dynamic_failure_faces,
    read_failure_faces,
    run_dynamic_handoff,
    run_handoff,
    solve_case_field,
    velocity_field,
)
from tensordem.fem_dem_fracture.dynamics import (
    SCHEMA_DEM_IC,
    SCHEMA_DYNAMIC,
    elastic_energy,
    floe_cell_membership,
    kinetic_energy,
    linearity_check,
)
from tensordem.fem_dem_fracture.study_dynamic import main as cli_main


# --------------------------------------------------------------------------
# fixtures: one production run shared by the module (the CLI writes the bundle)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def case() -> IcePlateCase:
    return IcePlateCase()


@pytest.fixture(scope="module")
def unit(case):
    return solve_case_field(case)


@pytest.fixture(scope="module")
def static_faces(case, tmp_path_factory):
    out = tmp_path_factory.mktemp("static")
    summary = run_handoff(case, str(out))
    faces = read_failure_faces(os.path.join(str(out), "failure_faces.json"))
    return {"summary": summary, "faces": faces, "dir": str(out)}


@pytest.fixture(scope="module")
def field(case, unit, static_faces):
    return velocity_field(unit, pressure_rate_pa_s=1000.0,
                          fracture_pressure_pa=float(
                              static_faces["summary"]["criteria"]["splitting_pressure_pa"]))


@pytest.fixture(scope="module")
def kinetic(case, unit, static_faces, field):
    faces = static_faces["faces"]
    kinematics = project_rigid_body_motion(faces, field)
    ledger = momentum_ledger(faces, field, kinematics)
    quasi = quasi_static_ledger(unit, list(case.patches), field,
                                density=case.material.density,
                                thickness=case.material.thickness)
    return {"faces": faces, "field": field, "kinematics": kinematics,
            "ledger": ledger, "quasi": quasi}


@pytest.fixture(scope="module")
def run(case, tmp_path_factory):
    out = tmp_path_factory.mktemp("dynamic")
    summary = run_dynamic_handoff(case, str(out), pressure_rate_pa_s=1000.0,
                                  make_figures=False, heave_periods=2.0)
    return {"summary": summary, "dir": str(out)}


# --------------------------------------------------------------------------
# 1. inherited velocity field
# --------------------------------------------------------------------------
def test_cell_rates_are_exact_for_a_linear_field():
    """A rigid (plane) deflection field must be reproduced exactly at cell centres."""

    @dataclass
    class FakeSolution:
        mesh: PlateMesh
        w: np.ndarray
        theta_x: np.ndarray
        theta_y: np.ndarray

    mesh = PlateMesh(nx=6, ny=5, length_x=3.0, length_y=2.5)
    xy = mesh.node_xy()
    a, b, c = 2.0e-5, 1.0e-5, -3.0e-6       # w = a + b x + c y
    w = a + b * xy[:, 0] + c * xy[:, 1]
    theta_x = np.full_like(w, b)
    theta_y = np.full_like(w, c)
    rates = cell_centroid_unit_rates(FakeSolution(mesh, w, theta_x, theta_y))
    cx = (np.arange(mesh.nx) + 0.5) * mesh.hx
    cy = (np.arange(mesh.ny) + 0.5) * mesh.hy
    XX, YY = np.meshgrid(cx, cy, indexing="xy")
    expected = (a + b * XX + c * YY).ravel()
    assert np.allclose(rates, expected, rtol=1e-12, atol=1e-18)
    # and the reconstruction really is nonzero where the field is
    assert np.max(np.abs(rates)) > 0.0


def test_loading_path_is_linear_in_the_pressure_scale(case, unit):
    """``wdot = qdot * w_hat``: a two-level solve must reproduce the unit field."""
    report = linearity_check(unit, list(case.patches), delta=1.0)
    assert report["nodal_rel_residual"] < 1e-9
    assert report["cell_rel_residual"] < 1e-9
    assert report["nodal_max_abs_fd_m_per_pa"] == pytest.approx(
        report["nodal_max_abs_unit_m_per_pa"], rel=1e-12)


def test_velocity_field_is_linear_in_the_pressure_rate(unit, static_faces):
    q_f = float(static_faces["summary"]["criteria"]["splitting_pressure_pa"])
    a = velocity_field(unit, pressure_rate_pa_s=1000.0, fracture_pressure_pa=q_f)
    b = velocity_field(unit, pressure_rate_pa_s=2500.0, fracture_pressure_pa=q_f)
    assert np.allclose(b.rates, a.rates * 2.5, rtol=1e-13, atol=0.0)
    assert np.allclose(b.unit_rates, a.unit_rates, rtol=0.0, atol=0.0)
    assert a.max_abs_rate > 0.0


def test_kinetic_energy_scales_with_the_square_of_the_rate(case, unit, static_faces):
    q_f = float(static_faces["summary"]["criteria"]["splitting_pressure_pa"])
    f1 = velocity_field(unit, pressure_rate_pa_s=1000.0, fracture_pressure_pa=q_f)
    f2 = velocity_field(unit, pressure_rate_pa_s=2000.0, fracture_pressure_pa=q_f)
    ke1 = kinetic_energy(f1, density=case.material.density,
                         thickness=case.material.thickness)
    ke2 = kinetic_energy(f2, density=case.material.density,
                         thickness=case.material.thickness)
    assert ke2 == pytest.approx(4.0 * ke1, rel=1e-12)


def test_quasi_static_ledger_reports_the_closed_form_critical_rate(case, unit,
                                                                  static_faces):
    q_f = float(static_faces["summary"]["criteria"]["splitting_pressure_pa"])
    field = velocity_field(unit, pressure_rate_pa_s=1000.0, fracture_pressure_pa=q_f)
    led = quasi_static_ledger(unit, list(case.patches), field,
                              density=case.material.density,
                              thickness=case.material.thickness)
    assert led["elastic_energy_j"] > 0.0
    assert led["critical_rate_pa_s"] > 0.0
    # KE/U is quadratic, so doubling the rate quadruples the ratio: the closed-form
    # critical rate must therefore satisfy KE(qdot_crit)/U == 0.1 by construction.
    ratio = led["kinetic_energy_j"] / led["elastic_energy_j"]
    implied = led["critical_rate_pa_s"] / 1000.0
    assert ratio * implied ** 2 == pytest.approx(0.1, rel=1e-12)
    assert led["quasi_static"] is True


# --------------------------------------------------------------------------
# 2. rigid-body projection: exact identities
# --------------------------------------------------------------------------
def _lattice(n=11, length=1.0):
    coords = (np.arange(n) + 0.5) * (length / n)
    XX, YY = np.meshgrid(coords, coords, indexing="xy")
    x = XX.ravel() - length / 2
    y = YY.ravel() - length / 2
    return x, y, np.full(x.size, 2.5)          # uniform point mass [kg]


def test_rigid_field_is_recovered_exactly():
    x, y, m = _lattice()
    v, wx, wy = 3.0e-3, -1.2e-4, 4.5e-4
    u = v + wx * y - wy * x
    fit = project_point_masses(x, y, u, m)
    assert fit.rigid_fraction == pytest.approx(1.0, rel=0, abs=1e-14)
    assert fit.max_abs_deform < 1e-18
    assert fit.v_z == pytest.approx(v, rel=1e-14)
    assert fit.omega_x == pytest.approx(wx, rel=1e-12)
    assert fit.omega_y == pytest.approx(wy, rel=1e-12)
    assert fit.ke_deform < 1e-30


def test_momentum_and_angular_momentum_are_conserved_exactly():
    rng = np.random.default_rng(20261111)
    x, y, m = _lattice(n=13)
    u = rng.normal(scale=1e-3, size=x.size)
    fit = project_point_masses(x, y, u, m)
    assert fit.momentum_z == pytest.approx(float(np.sum(m * u)), rel=1e-14)
    l_ref = np.array([float(np.sum(m * (y - fit.centroid[1]) * u)),
                      float(-np.sum(m * (x - fit.centroid[0]) * u))])
    assert np.allclose(fit.angular_momentum, l_ref, rtol=1e-12, atol=1e-18)


def test_kinetic_energy_splits_without_a_residual():
    rng = np.random.default_rng(7)
    x, y, m = _lattice(n=9)
    u = rng.normal(scale=1e-3, size=x.size)
    fit = project_point_masses(x, y, u, m)
    ke_ref = 0.5 * float(np.sum(m * u * u))
    assert fit.ke_total == pytest.approx(ke_ref, rel=1e-14)
    assert fit.ke_rigid + fit.ke_deform == pytest.approx(fit.ke_total, rel=1e-13)
    assert 0.0 <= fit.rigid_fraction <= 1.0


def test_isotropic_bending_collapses_to_the_mean_translation():
    """Symmetry kills the tilt: the rigid fit of kappa/2 (x^2+y^2) is its mean."""
    x, y, m = _lattice(n=12)
    kappa = 1.0e-3
    u = 0.5 * kappa * (x ** 2 + y ** 2)
    fit = project_point_masses(x, y, u, m)
    assert max(abs(fit.omega_x), abs(fit.omega_y)) < 1e-15
    ref = float(np.mean(u) ** 2 / np.mean(u ** 2))
    assert fit.rigid_fraction == pytest.approx(ref, rel=1e-12)
    assert 0.0 < fit.rigid_fraction < 1.0


def test_projection_rejects_invalid_masses():
    x, y, _ = _lattice(n=4)
    with pytest.raises(ValueError):
        project_point_masses(x, y, np.zeros_like(x), np.zeros_like(x))
    with pytest.raises(ValueError):
        project_point_masses(x, y, np.zeros_like(x), -np.ones_like(x))


# --------------------------------------------------------------------------
# 3. the artefact and its ledger
# --------------------------------------------------------------------------
def test_floe_membership_covers_the_plate_exactly(kinetic):
    faces = kinetic["faces"]
    mesh = kinetic["field"].mesh
    cells = floe_cell_membership(faces, mesh)
    assert len(cells) == len(faces.floes)
    assert sum(c.size for c in cells) == mesh.n_cells
    for floe, idx in zip(faces.floes, cells):
        assert idx.size == floe.n_cells
    covered = np.concatenate(cells)
    assert np.array_equal(np.sort(covered), np.arange(mesh.n_cells))


def test_transfer_conserves_mass_momentum_and_energy(kinetic, case):
    faces, ledger = kinetic["faces"], kinetic["ledger"]
    plate_area = faces.meta["plate"]["area_m2"]
    plate_mass = plate_area * case.material.thickness * case.material.density
    assert ledger.mass_field == pytest.approx(plate_mass, rel=1e-12)
    assert abs(ledger.mass_residual) < 1e-6
    assert abs(ledger.momentum_residual) < 1e-9 * max(1.0, abs(ledger.momentum_field))
    assert np.max(np.abs(ledger.angular_residual)) < 1e-9 * max(
        1.0, float(np.max(np.abs(ledger.angular_field))))
    assert abs(ledger.ke_sum_residual) < 1e-9 * max(1.0, ledger.ke_field)
    assert 0.0 <= ledger.rigid_fraction <= 1.0


def test_schema_two_roundtrip_recomputes_the_ledger(run):
    summary = run["summary"]
    path = summary["artifacts"]["failure_faces_dynamic"]
    dynamic = read_dynamic_failure_faces(path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["schema"] == SCHEMA_DYNAMIC
    assert dynamic.n_bodies == len(dynamic.faces.floes)
    assert dynamic.ledger.mass_residual == pytest.approx(
        payload["momentum_ledger"]["mass_residual_kg"], abs=1e-12)
    assert dynamic.ledger.momentum_residual == pytest.approx(
        payload["momentum_ledger"]["momentum_residual_kg_m_s"], abs=1e-15)
    # the static reader still validates the schema-2 file (superset contract)
    assert read_failure_faces(payload).conservation.n_floe == len(dynamic.faces.floes)


def test_schema_two_detects_a_tampered_velocity(run, tmp_path):
    path = run["summary"]["artifacts"]["failure_faces_dynamic"]
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["velocity_field"]["cells"][3]["v_z_m_s"] *= 1.5
    with pytest.raises(ValueError, match="SHA-256|disagrees|qdot"):
        read_dynamic_failure_faces(payload)
    # a payload whose stored hash was also updated must still fail the recomputation
    from tensordem.fem_dem_fracture import write_dynamic_failure_faces

    tampered = dict(payload)
    tampered["provenance"] = {k: v for k, v in payload["provenance"].items()
                              if k != "payload_sha256"}
    write_dynamic_failure_faces(tampered, str(tmp_path / "tampered.json"))
    with pytest.raises(ValueError, match="disagrees|not qdot"):
        read_dynamic_failure_faces(str(tmp_path / "tampered.json"))


def test_dem_initial_conditions_carry_the_inherited_velocity(run):
    dynamic = read_dynamic_failure_faces(
        run["summary"]["artifacts"]["failure_faces_dynamic"])
    ic = dem_initial_conditions_dynamic(dynamic)
    assert ic.as_payload()["schema"] == SCHEMA_DEM_IC
    assert len(ic) == dynamic.n_bodies
    assert ic.total_mass == pytest.approx(dynamic.faces.meta["plate"]["area_m2"]
                                          * dynamic.faces.meta["plate"]["thickness_m"]
                                          * dynamic.faces.meta["material"]["density_kg_m3"],
                                          rel=1e-12)
    # vertical velocities are exactly the kinematics handed over, in-plane motion is zero
    assert np.allclose(ic.velocities[:, 2],
                       [k.v_z for k in dynamic.kinematics], rtol=0.0, atol=0.0)
    assert np.allclose(ic.velocities[:, :2], 0.0)
    assert np.allclose(ic.angular_velocities[:, 2], 0.0)
    # the positions are the FEM cell mass centroids the fit is conservative about
    for i, k in enumerate(dynamic.kinematics):
        assert ic.positions[i, 0] == pytest.approx(k.centroid[0], rel=1e-12)
        assert ic.positions[i, 1] == pytest.approx(k.centroid[1], rel=1e-12)


# --------------------------------------------------------------------------
# 4. DEM-side intake (the real cross-module call)
# --------------------------------------------------------------------------
def test_dem_intake_conserves_mass_and_reproduces_the_heave_period(run):
    intake = json.load(open(run["summary"]["artifacts"]["dem_intake"], encoding="utf-8"))
    seeded = intake["seeded_ledger"]
    assert intake["n_bodies"] == len(run["summary"]["per_floe"])
    assert abs(seeded["mass_residual_kg"]) < 1e-6
    assert abs(seeded["area_residual_m2"]) < 1e-9
    assert intake["heave"]["max_period_error"] < 1e-3
    thickness = run["summary"]["case"]["thickness"]
    for row in intake["heave"]["rows"]:
        # the seeded body really carries the floe density (mass = area * t * rho)
        assert row["rho_body_kg_m3"] == pytest.approx(
            row["mass_kg"] / (row["area_m2"] * thickness), rel=1e-9)


def test_dem_intake_is_deterministic(kinetic, tmp_path):
    """Reading the artefact twice must give the same bodies and the same period."""
    from tensordem.fem_dem_fracture.dynamics import (
        build_dynamic_failure_faces,
        write_dynamic_failure_faces,
    )

    payload = build_dynamic_failure_faces(kinetic["faces"], kinetic["field"],
                                          kinetic["kinematics"], kinetic["ledger"])
    path = str(tmp_path / "ff.json")
    write_dynamic_failure_faces(payload, path)
    a = dem_intake(read_dynamic_failure_faces(path), heave_periods=3.0)
    b = dem_intake(read_dynamic_failure_faces(path), heave_periods=3.0)
    assert np.isfinite(a["heave"]["max_period_error"])
    assert a["n_bodies"] == b["n_bodies"]
    assert a["seeded_ledger"]["body_mass_sum_kg"] == pytest.approx(
        b["seeded_ledger"]["body_mass_sum_kg"], rel=0.0, abs=0.0)
    assert a["heave"]["max_period_error"] == pytest.approx(
        b["heave"]["max_period_error"], rel=0.0, abs=0.0)


# --------------------------------------------------------------------------
# 5. criteria and the CLI entry point
# --------------------------------------------------------------------------
def test_all_dynamic_criteria_pass(run):
    criteria = run["summary"]["criteria"]
    failures = {k: v for k, v in criteria.items()
                if isinstance(v, dict) and "pass" in v and not v["pass"]}
    assert failures == {}
    assert criteria["D4_kinetic_energy_split"]["rigid_fraction_plate"] > 0.9
    assert criteria["D6_mass_conservation_after_transfer"]["residual_kg"] < 1e-6


def test_benchmark_rows_are_within_tolerance(run):
    table = json.load(open(run["summary"]["artifacts"]["benchmark"], encoding="utf-8"))
    assert table["all_within_tolerance"] is True
    assert table["n_failing"] == 0
    assert any(r["benchmark"].startswith("SSSS square plate") for r in table["rows"])
    assert any(r["benchmark"].startswith("rigid projection") for r in table["rows"])


def test_cli_entry_point_writes_the_bundle(tmp_path):
    out = tmp_path / "cli"
    code = cli_main(["--out", str(out), "--pressure-rate", "1000",
                     "--nx", "40", "--ny", "40", "--no-figures", "--quiet"])
    assert code == 0
    for name in ("failure_faces.json", "failure_faces_dynamic.json",
                 "dem_initial_conditions.json", "dem_initial_conditions_dynamic.json",
                 "dem_intake.json", "benchmark.json", "summary_dynamic.json",
                 "manifest.json"):
        assert (out / name).exists(), name
    manifest = json.load(open(out / "manifest.json", encoding="utf-8"))
    assert manifest["criteria_all_pass"] is True
    assert "--pressure-rate 1000" in manifest["repro_cmd"]
    assert set(manifest["artifacts"]) >= {"failure_faces_dynamic", "dem_intake"}
    for record in manifest["artifacts"].values():
        assert len(record["sha256"]) == 64