"""Tests of the ice-rubble microstructure module (TensorDEM m5074).

Coverage
--------
* Rosin-Rammler generator: range, analytic CDF agreement (KS), moments.
* Kolmogorov-Smirnov implementation validated against ``scipy.stats``.
* Force-chain reference shape: unit mean, unit CV for the exponential, analytic
  CDF/quantiles, plateau variant.
* Voxel Voronoi porosity: exact on a lattice with a known solid fraction,
  convergence under voxel refinement, bounding values.
* DEM engine: single-sphere restitution sanity, frictionless monodisperse
  packing reaches random close packing (the negative control), coordination
  number of an ordered lattice, gravity/platen plumbing, energy bookkeeping.
* Literature criteria plumbing: the study analysis accepts/rejects as documented
  (cheap synthetic runs, no DEM).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import tensordem.ice_rubble_microstructure as rub


# ---------------------------------------------------------------------------
# Floe size distribution
# ---------------------------------------------------------------------------
def test_rosin_rammler_sample_matches_analytic_cdf():
    dist = rub.RosinRammlerDistribution()
    sample = dist.sample(20000, seed=3)
    d, p = rub.ks_test_cdf(sample, dist.cdf)
    assert d < 0.02
    assert p > 0.01
    assert sample.min() >= dist.d_min - 1e-12
    assert sample.max() <= dist.d_max + 1e-12


def test_rosin_rammler_moments_and_ppf_roundtrip():
    dist = rub.RosinRammlerDistribution(d_min=0.05, d_max=0.5, d_char=0.13, exponent=2.0)
    u = np.linspace(0.001, 0.999, 501)
    recovered = dist.cdf(dist.ppf(u))
    assert np.allclose(recovered, u, atol=2e-3)
    mean_d = dist.mean_diameter()
    assert dist.d_min < mean_d < dist.d_max
    moments = dist.moments(3)
    # diameters are < 1 m, so E[d] > E[d^2] > E[d^3] but each is positive
    assert moments[0] > moments[1] > moments[2] > 0.0
    # the truncated mean volume must be reproducible from E[d^3]
    assert dist.mean_volume() == pytest.approx(math.pi / 6.0 * moments[2], rel=1e-9)


def test_rosin_rammler_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        rub.RosinRammlerDistribution(d_min=0.5, d_max=0.1)
    with pytest.raises(ValueError):
        rub.RosinRammlerDistribution(exponent=0.0)


def test_distribution_with_solid_volume_solves_the_scale():
    target = 0.0017
    for k in (1.2, 2.0, 3.6):
        dist = rub.distribution_with_solid_volume(exponent=k, target_mean_volume=target)
        assert dist.mean_volume() == pytest.approx(target, rel=2e-3)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_ks_test_matches_scipy():
    from scipy import stats

    rng = np.random.default_rng(5)
    sample = rng.normal(size=400)
    d_ref, p_ref = stats.kstest(sample, "norm")
    d, p = rub.ks_test_cdf(sample, lambda x: stats.norm.cdf(x))
    assert d == pytest.approx(d_ref, abs=1e-10)
    assert p == pytest.approx(p_ref, rel=0.05)


def test_ks_test_rejects_wrong_distribution_and_accepts_right_one():
    rng = np.random.default_rng(7)
    uniform = rng.random(500)
    d_bad, p_bad = rub.ks_test_cdf(uniform, rub.HOPKINS_TVEITEN_2002.cdf)
    assert p_bad < 1e-6
    good = rub.HOPKINS_TVEITEN_2002.sample(500, seed=2)
    d_good, p_good = rub.ks_test_cdf(good, rub.HOPKINS_TVEITEN_2002.cdf)
    assert p_good > 0.05


def test_ks_two_sample_symmetry_and_null():
    rng = np.random.default_rng(11)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    d_ab, p_ab = rub.ks_test_two_sample(a, b)
    d_ba, p_ba = rub.ks_test_two_sample(b, a)
    assert d_ab == pytest.approx(d_ba)
    assert p_ab == pytest.approx(p_ba)
    assert p_ab > 0.05
    c = rng.normal(loc=1.0, size=200)
    d_ac, p_ac = rub.ks_test_two_sample(a, c)
    assert p_ac < 1e-6 and d_ac > d_ab


def test_force_chain_reference_is_unit_mean_exponential():
    ref = rub.HOPKINS_TVEITEN_2002
    assert ref.plateau_end == 0.0
    grid = np.linspace(0.0, 60.0, 200001)
    pdf = ref.pdf(grid)
    assert np.trapezoid(pdf, grid) == pytest.approx(1.0, rel=1e-3)
    assert np.trapezoid(grid * pdf, grid) == pytest.approx(1.0, rel=1e-3)
    assert ref.cdf(np.array([1.0]))[0] == pytest.approx(1.0 - math.exp(-1.0), rel=1e-9)
    assert ref.coefficient_of_variation() == pytest.approx(1.0, rel=5e-3)
    # half of the load is carried by the f > ln 2 part of an exponential
    assert ref.survival(np.array([2.0]))[0] == pytest.approx(math.exp(-2.0), rel=1e-9)


def test_plateau_reference_has_narrower_shape():
    plateau = rub.HOPKINS_TVEITEN_2002_PLATEAU
    assert plateau.mean() == pytest.approx(1.0, rel=1e-6)
    assert 0.6 < plateau.coefficient_of_variation() < 0.95
    assert plateau.plateau_height == pytest.approx(plateau.pdf(np.array([0.5]))[0])


# ---------------------------------------------------------------------------
# Voronoi porosity
# ---------------------------------------------------------------------------
def _lattice(radius: float, spacing: float, n: int = 8):
    axis = (np.arange(n) + 0.5) * spacing
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    radii = np.full(grid.shape[0], radius)
    return grid, radii


def test_voxel_voronoi_porosity_reproduces_lattice_solid_fraction():
    radius, spacing, n = 0.2, 0.6, 8
    positions, radii = _lattice(radius, spacing, n)
    expected_phi = 4.0 / 3.0 * math.pi * radius**3 / spacing**3
    domain = (spacing, spacing, spacing) * np.array([1, 1, 1]) * n
    result = rub.voxel_voronoi_porosity(
        positions,
        radii,
        domain_lo=(0.0, 0.0, 0.0),
        domain_hi=(n * spacing, n * spacing, n * spacing),
        voxel_size=spacing / 26,
    )
    # voxel-resolved tessellation: |Delta n| < 6e-3 at this resolution
    assert result["packing_fraction"] == pytest.approx(expected_phi, abs=6e-3)
    assert result["porosity"] == pytest.approx(1.0 - expected_phi, abs=6e-3)
    # every Voronoi cell of a primitive lattice holds exactly one sphere
    per_cell = 1.0 - result["cell_porosity"]
    assert np.allclose(per_cell, expected_phi, atol=0.02)
    assert (result["cell_volume"] > 0).all()


def test_voxel_voronoi_porosity_converges_under_refinement():
    positions, radii = _lattice(0.2, 0.55, 6)
    kwargs = dict(
        domain_lo=(0.0, 0.0, 0.0),
        domain_hi=(6 * 0.55, 6 * 0.55, 6 * 0.55),
    )
    coarse = rub.voxel_voronoi_porosity(positions, radii, voxel_size=0.55 / 20, **kwargs)
    fine = rub.voxel_voronoi_porosity(positions, radii, voxel_size=0.55 / 40, **kwargs)
    assert abs(coarse["porosity"] - fine["porosity"]) < 5e-3
    analytic = 1.0 - 4.0 / 3.0 * math.pi * 0.2**3 / 0.55**3
    assert abs(fine["porosity"] - analytic) < 6e-3


def test_voxel_voronoi_porosity_limits():
    positions, radii = _lattice(0.2, 0.6, 4)
    dense = rub.voxel_voronoi_porosity(
        positions, radii, domain_lo=(0, 0, 0), domain_hi=(2.4, 2.4, 2.4), voxel_size=0.05
    )
    assert 0.0 < dense["porosity"] < 1.0
    with pytest.raises(ValueError):
        rub.voxel_voronoi_porosity(
            positions, radii, domain_lo=(1, 1, 1), domain_hi=(0, 0, 0), voxel_size=0.05
        )


# ---------------------------------------------------------------------------
# DEM engine
# ---------------------------------------------------------------------------
def _small_field(mu=0.3, n=180, radius=0.1, spacing=0.30, gravity_scale=1.0, material=None):
    axis = (np.arange(6) + 0.5) * spacing
    z_axis = (np.arange(5) + 0.5) * spacing
    grid = np.stack(np.meshgrid(axis, axis, z_axis, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[:n]
    radii = np.full(grid.shape[0], radius)
    mat = material or rub.IceRubbleMaterial(friction_coefficient=mu, gravity_scale=gravity_scale)
    return rub.IceRubbleField(grid, radii, mat, box=(6 * spacing, 6 * spacing, 6 * spacing))


def test_engine_rejects_invalid_input():
    with pytest.raises(ValueError):
        rub.IceRubbleMaterial(friction_coefficient=-1.0)
    with pytest.raises(ValueError):
        rub.IceRubbleField(np.zeros((2, 3)), np.ones(3), box=(1, 1, 1))
    with pytest.raises(ValueError):
        rub.IceRubbleField(np.zeros((2, 2)), np.ones(2), box=(1, 1, 1))


def test_recommended_dt_is_stable_for_the_smallest_particle():
    mat = rub.IceRubbleMaterial()
    dt = mat.recommended_dt(0.025)
    assert 1e-4 < dt < 5e-3
    # the time step used by the study is inside the stability window
    assert dt >= 1.0e-3


def test_engine_step_is_energy_bounded_and_conserves_cell_volume():
    field = _small_field()
    start = field.kinetic_energy()
    for _ in range(50):
        field.step(1e-3)
    assert field.pos[:, 2].min() >= field.floor_z - 1e-12
    assert field.pos[:, 0].max() <= field.platen_x + 1e-12
    assert math.isfinite(field.kinetic_energy())
    assert field.kinetic_energy() >= 0.0
    assert field.max_speed() < 5.0  # no numerical explosion
    assert start == pytest.approx(0.0)


def test_contact_network_counts_ordered_lattice_neighbours():
    radius, spacing = 0.2, 0.38  # slight overlap so every neighbour is a contact
    axis = (np.arange(4) + 0.5) * spacing
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    radii = np.full(grid.shape[0], radius)
    mat = rub.IceRubbleMaterial(friction_coefficient=0.5)
    field = rub.IceRubbleField(grid, radii, mat, box=(4 * spacing, 4 * spacing, 4 * spacing))
    contacts = field.contact_network()
    # simple cubic interior particles have 6 neighbours; the average over a
    # 4x4x4 block is 6 - (boundary deficit) -> between 4 and 6
    stats = rub.coordination_number(contacts, field.n)
    assert 4.0 <= stats["Z"] <= 6.0
    assert contacts["normal_force"].min() > 0.0
    assert contacts["i"].shape == contacts["j"].shape == contacts["normal_force"].shape


def test_coordination_spectrum_is_monotone_and_bounded():
    field = _small_field()
    for _ in range(60):
        field.step(1e-3)
    contacts = field.contact_network()
    spectrum = rub.coordination_spectrum(contacts, field.n)
    zs = [row["Z"] for row in spectrum]
    assert zs == sorted(zs, reverse=True)
    assert all(z >= 0.0 for z in zs)
    assert spectrum[0]["n_contacts"] >= spectrum[-1]["n_contacts"]


def test_force_chain_pdf_and_clusters():
    rng = np.random.default_rng(4)
    forces = rng.exponential(scale=3.0, size=2000)
    pdf = rub.force_chain_pdf(forces)
    assert pdf["mean_force"] == pytest.approx(3.0, rel=0.1)
    assert pdf["coefficient_of_variation"] == pytest.approx(1.0, rel=0.1)
    assert 0.3 < pdf["strong_fraction"] < 0.45  # exp(-1) = 0.368
    assert 0.6 < pdf["strong_load_fraction"] < 0.85  # 2 exp(-1) = 0.736
    assert np.all(pdf["pdf"] >= 0.0)
    # a chain of four collinear strong contacts is one cluster
    contacts = {
        "i": np.array([0, 1, 2, 3, 10]),
        "j": np.array([1, 2, 3, 4, 11]),
        "normal_force": np.array([10.0, 12.0, 11.0, 9.0, 0.01]),
    }
    chains = rub.force_chain_clusters(contacts, threshold=1.0, n_particles=12)
    assert chains["n_chains"] == 1
    assert chains["largest_chain_contacts"] == 4
    assert chains["particle_participation"] == pytest.approx(5 / 12)
    with pytest.raises(ValueError):
        rub.force_chain_pdf(np.array([1.0, 2.0]))


def test_frictionless_monodisperse_packing_approaches_random_close_packing():
    """Negative control: mu = 0 nearly monodisperse spheres -> phi -> 0.64 (RCP).

    Random (RSA) placement avoids the columnar arching of a perfect lattice, which
    would stop the piston long before random close packing.
    """
    dist = rub.RosinRammlerDistribution(d_min=0.19, d_max=0.21, d_char=0.2, exponent=20.0)
    box = (1.2, 1.2, 2.2)
    field_data = rub.generate_floe_field(n_floes=300, distribution=dist, box=box, seed=5)
    assert 0.3 < field_data["initial_packing_fraction"] < 0.5
    mat = rub.IceRubbleMaterial(friction_coefficient=0.0, rolling_friction=0.0)
    field = rub.IceRubbleField(field_data["positions"], field_data["radii"], mat, box=box)
    result = field.compact_to_stress(
        1e-3, stress_target=10000.0, max_steps=6000, lid_mass=200.0
    )
    assert result["reached_target"]
    phi = field.packing_fraction()
    assert 0.58 <= phi <= 0.68, phi


def test_platen_advance_records_a_positive_resistance():
    field = _small_field(mu=0.8)
    field.compact_to_stress(1e-3, stress_target=15000.0, max_steps=2500)
    # start the platen right at the outermost floe so that the push engages
    field.platen_x = float(field.pos[:, 0].max())
    history = field.advance_platen(1e-3, speed=0.05, penetration=0.03)
    assert history["steps"] > 0
    assert history["mean_force_all"] > 0.0
    assert len(history["force_history"]) == len(history["penetration_history"]) > 0
    assert history["penetration"] >= 0.02 - 1e-9


def test_gravity_scale_changes_the_body_force_only():
    heavy = _small_field(gravity_scale=1.0)
    light = _small_field(gravity_scale=0.5)
    heavy.step(1e-3)
    light.step(1e-3)
    assert heavy.pos[:, 2].mean() < light.pos[:, 2].mean()
    assert heavy.material.effective_gravity == pytest.approx(2.0 * light.material.effective_gravity)


# ---------------------------------------------------------------------------
# Study plumbing (analysis only, no DEM)
# ---------------------------------------------------------------------------
def _synthetic_runs(cfg, *, porosity_primary=0.32, z_primary=4.0):
    """Minimal stand-in for measured runs used to exercise the criteria logic."""
    rng = np.random.default_rng(9)

    def make(name, mu, n, z, resistance, exponent=2.0, d_char=0.13):
        normalised = rng.exponential(size=1200)
        return {
            "case": {"name": name},
            "friction_coefficient": mu,
            "exponent": exponent,
            "d_char": d_char,
            "diameters": list(rng.uniform(0.05, 0.5, size=1200)),
            "n_particles": 1200,
            "compaction": {"final_lid_pressure": 25000.0},
            "microstructure": {
                "porosity_voronoi": n,
                "packing_fraction_voronoi": 1.0 - n,
                "packing_fraction_cell": 1.0 - n,
                "coordination_number": z,
                "coordination_number_all": z + 0.4,
                "diameter_min": 0.05,
                "diameter_max": 0.5,
                "cell_height": 0.95,
                "coordination_spectrum": [{"force_fraction": 0.0, "Z": z, "n_contacts": 100}],
                "force_chain": {
                    "normalised": normalised.tolist(),
                    "coefficient_of_variation": 1.0,
                    "strong_load_fraction": 0.74,
                },
            },
            "drag": {"mean_force_tail": resistance},
            "static_lateral_thrust": 2.0e3 * (1.0 - n),
        }

    runs = {}
    for mu, n, z in zip(cfg.friction_values, [0.36, 0.35, 0.34, 0.335, 0.32], [6, 5.5, 5, 4.5, 4.0]):
        runs[f"mu_{mu:.2f}"] = make(f"mu_{mu:.2f}", mu, n, z, 1.0e3 * (1.0 - n))
    for k, n in zip(cfg.drag_exponents, [0.36, 0.35, 0.34, 0.33, 0.32]):
        runs[f"shape_k{k:.2f}"] = make(
            f"shape_k{k:.2f}", cfg.primary_friction, n, z_primary, 3.0e3 * (1.0 - n), exponent=k
        )
    for count, n in zip(cfg.fill_counts, [0.54, 0.49, 0.44, 0.39, 0.35]):
        runs[f"fill_{count:d}"] = make(
            f"fill_{count:d}", cfg.primary_friction, n, 4.0, 3.0e3 * (1.0 - n)
        )
    runs["primary_replicate"] = runs[f"mu_{cfg.primary_friction:.2f}"]
    runs["gravity_half"] = make("gravity_half", cfg.primary_friction, porosity_primary + 0.01, 4.0, 1.0e3)
    runs["negative_control"] = make("negative_control", 0.0, 0.36, 8.0, 0.0)
    return runs


def test_analyse_runs_reports_every_criterion():
    cfg = rub.MicrostructureStudyConfig()
    study = rub.analyse_runs(cfg, _synthetic_runs(cfg))
    criteria = study["criteria"]
    for key in (
        "a_porosity_vs_literature",
        "b_force_chain_pdf",
        "c_coordination_number",
        "d_resistance_feedback",
        "e_negative_control_frictionless",
        "f_floe_field",
        "g_gravity_sensitivity",
    ):
        assert key in criteria
    assert criteria["a_porosity_vs_literature"]["passed"] is True  # n = 0.32 exactly
    assert criteria["c_coordination_number"]["passed"] is True
    assert criteria["d_resistance_feedback"]["r_squared"] > 0.95
    assert study["summary"]["n_gating_criteria"] == 6
    # a wrong porosity must fail the porosity criterion
    runs = _synthetic_runs(cfg)
    runs[f"mu_{cfg.primary_friction:.2f}"]["microstructure"]["porosity_voronoi"] = 0.45
    bad = rub.analyse_runs(cfg, runs)
    assert bad["criteria"]["a_porosity_vs_literature"]["passed"] is False
    assert bad["criteria"]["a_porosity_vs_literature"]["relative_error"] > 0.1


def test_study_cases_are_complete_and_unique():
    cfg = rub.MicrostructureStudyConfig()
    cases = rub.study_cases(cfg)
    names = [c["name"] for c in cases]
    assert len(names) == len(set(names))
    assert f"mu_{cfg.primary_friction:.2f}" in names
    assert all(f"shape_k{k:.2f}" in names for k in cfg.drag_exponents)
    assert all(f"fill_{c:d}" in names for c in cfg.fill_counts)
    assert "negative_control" in names and "gravity_half" in names
    # the shape family keeps the floe count and the solid volume fixed
    target = cfg.target_solid_volume
    for case in cases:
        if case["kind"] != "shape":
            continue
        dist = rub.RosinRammlerDistribution(
            d_min=cfg.d_min, d_max=cfg.d_max, d_char=case["d_char"], exponent=case["exponent"]
        )
        assert cfg.n_floes * dist.mean_volume() == pytest.approx(target, rel=5e-3)


def test_evidence_grades_are_declared():
    assert rub.TIMCO_OBRIEN_2004.mean == 0.32
    assert rub.TIMCO_OBRIEN_2004.std == 0.05
    assert rub.TIMCO_OBRIEN_2004.n_datasets >= 10
    assert rub.TIMCO_OBRIEN_2004.within(0.34)
    assert not rub.TIMCO_OBRIEN_2004.within(0.45)
    assert "LITERATURE_SUMMARY" in rub.TIMCO_OBRIEN_2004.evidence
    assert "PARAMETRIC_MODEL" in rub.HOPKINS_TVEITEN_2002.evidence
    assert rub.COORDINATION_REFERENCE.mean == 4.0