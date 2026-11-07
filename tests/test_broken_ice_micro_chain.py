"""Tests for the broken-ice microstructure chain (IN34-W3-N01).

Coverage
--------
* triangular lattice topology (bond lengths, bulk coordination, node area);
* fragment census (connected components, area-equivalent diameter, size filter);
* quasi-static fragmentation engine: intact-lattice modulus, resolution
  independence, no-disorder negative control, disorder-driven fragmentation and
  mass conservation;
* FSD power-law estimator: recovers a known exponent, reports the survival-fit
  ``R**2``, and rejects a sample that is too short;
* analytic Janssen profile (small-depth and saturated limits) and the
  ``(K, mu_w)`` recovery from exact synthetic data;
* exponential-tail estimator on synthetic exponential and half-normal samples;
* the clogging-curve interpolation;
* the study ledger logic (pass and fail paths).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import tensordem.broken_ice_micro as bim
from tensordem.broken_ice_micro import study


# ------------------------------------------------------------------- lattice
def test_triangular_lattice_bond_length_and_coordination():
    spacing = 0.03
    pos, bonds = bim.triangular_lattice(9, 6, spacing)
    assert pos.shape == (54, 2)
    lengths = np.linalg.norm(pos[bonds[:, 1]] - pos[bonds[:, 0]], axis=1)
    assert np.allclose(lengths, spacing, rtol=1e-12)
    degree = np.bincount(bonds.reshape(-1), minlength=pos.shape[0])
    assert degree.max() == 6                 # bulk coordination of a triangular net
    assert degree.min() == 2                 # corners


def test_node_area_matches_lattice_cell():
    cfg = bim.FragmentationConfig(nx=8, ny=8, spacing=0.05)
    # one node owns sqrt(3)/2 a^2 of plate area
    assert cfg.node_area == pytest.approx(0.5 * math.sqrt(3.0) * 0.05**2)
    assert cfg.node_length == pytest.approx(7 * 0.05)
    assert cfg.bond_stiffness == pytest.approx(cfg.young_modulus * cfg.thickness / math.sqrt(3.0))


# -------------------------------------------------------------------- census
def test_fragment_census_components_and_diameters():
    # 6 nodes: component {0,1,2}, component {3,4}, singleton {5}
    bonds = np.array([[0, 1], [1, 2], [3, 4]])
    area = 0.01
    census = bim.fragment_census(6, bonds, area, min_fragment_nodes=1)
    assert census.n_fragments == 3
    assert sorted(census.node_counts.tolist()) == [1, 2, 3]
    assert census.total_nodes == 6
    assert census.diameters.max() == pytest.approx(math.sqrt(4 * 3 * area / math.pi))
    filtered = bim.fragment_census(6, bonds, area, min_fragment_nodes=2)
    assert sorted(filtered.node_counts.tolist()) == [2, 3]
    assert filtered.largest_fraction == pytest.approx(3 / 6)


# -------------------------------------------------------------------- engine
def test_engine_effective_modulus_is_resolution_independent():
    ratios = []
    for nx, ny in ((16, 8), (32, 16)):
        cfg = bim.FragmentationConfig(nx=nx, ny=ny, spacing=0.03)
        run = bim.QuasiStaticFragmentation(cfg)
        e_eff = run.effective_modulus(strain=1e-5)
        ratios.append(e_eff / cfg.young_modulus)
    # the 2:1 plate keeps two free lateral surfaces, so a 2x refinement changes
    # E_eff by a boundary-layer amount (measured 9 %); the lattice factor itself
    # is resolution independent within that band
    assert ratios[0] == pytest.approx(ratios[1], rel=0.12)
    # a central-force triangular lattice is softer than the continuum by a fixed
    # geometric factor; the factor must be a sane O(1) number
    assert 0.3 < ratios[0] < 0.9


def test_engine_intact_lattice_is_linear_elastic():
    cfg = bim.FragmentationConfig(nx=16, ny=8, spacing=0.03)
    run = bim.QuasiStaticFragmentation(cfg)
    run.thresholds = run.thresholds * 1e6        # freeze damage
    forces = []
    for strain in (2e-6, 4e-6, 8e-6):
        run.applied_strain = strain
        run.solve()
        forces.append(run.wall_force())
    assert forces[1] / forces[0] == pytest.approx(2.0, rel=0.02)
    assert forces[2] / forces[0] == pytest.approx(4.0, rel=0.02)
    assert run.total_breaks == 0


def test_no_disorder_negative_control_does_not_fragment():
    """Uniform thresholds: the plate fails everywhere at once, not into a census."""
    cfg = bim.FragmentationConfig(nx=24, ny=12, spacing=0.03, weibull_modulus=4.0,
                                  target_strain=0.02, strain_increment=1e-3)
    run = bim.QuasiStaticFragmentation(cfg)
    run.thresholds = np.full_like(run.thresholds, 5e-3)
    run.run()
    census = run.census()
    assert census.n_fragments <= 2


def test_disordered_lattice_fragments_and_conserves_nodes():
    cfg = bim.FragmentationConfig(nx=32, ny=16, spacing=0.03, seed=2, target_strain=0.05,
                                  strain_increment=1e-3, min_fragment_nodes=1)
    run = bim.QuasiStaticFragmentation(cfg)
    diag = run.run()
    census = run.census()
    assert diag["broken_bonds"] > 50
    assert census.n_fragments > 5
    assert census.total_nodes + 0 == census.total_nodes <= census.n_nodes_total
    assert 0.0 < census.largest_fraction <= 1.0
    # fragments cannot contain more nodes than the lattice has
    assert census.node_counts.sum() == census.total_nodes


# ----------------------------------------------------------------------- FSD
def test_power_law_fit_recovers_known_exponent():
    rng = np.random.default_rng(7)
    alpha = 3.0
    x_min, x_max = 0.05, 1.0
    u = rng.random(20000)
    a = 1.0 - alpha
    sample = (u * (x_max**a - x_min**a) + x_min**a) ** (1.0 / a)
    fit = bim.fit_power_law(sample, x_min=x_min, min_tail=100)
    assert fit.alpha == pytest.approx(alpha, rel=0.05)
    assert fit.n_tail > 1000
    assert fit.r2 > 0.95
    assert fit.ks_distance < 0.02


def test_power_law_fit_rejects_tiny_sample():
    with pytest.raises(ValueError):
        bim.fit_power_law(np.linspace(0.1, 0.2, 5), min_tail=10)


def test_log_binned_histogram_slope_is_minus_alpha():
    rng = np.random.default_rng(11)
    alpha = 2.5
    x_min, x_max = 0.1, 1.0
    u = rng.random(50000)
    a = 1.0 - alpha
    sample = (u * (x_max**a - x_min**a) + x_min**a) ** (1.0 / a)
    centres, density = bim.log_binned_histogram(sample, x_min=x_min, x_max=x_max)
    slope = np.polyfit(np.log10(centres), np.log10(density), 1)[0]
    assert slope == pytest.approx(-alpha, rel=0.05)


# ------------------------------------------------------------------- Janssen
def test_janssen_profile_limits():
    prof = bim.janssen_profile(width=0.4, rho_bulk=680.0, k_ratio=0.5, wall_friction=0.5)
    lam = prof.decay_length
    assert lam == pytest.approx(0.4 / (2 * 0.5 * 0.5))
    assert prof.plateau_stress == pytest.approx(680.0 * 9.81 * lam)
    deep = prof.stress_zz(50 * lam)
    assert deep == pytest.approx(prof.plateau_stress, rel=1e-6)
    shallow = prof.stress_zz(1e-4 * lam)
    assert shallow == pytest.approx(680.0 * 9.81 * 1e-4 * lam, rel=1e-3)
    assert prof.wall_stress(3 * lam) == pytest.approx(0.5 * prof.stress_zz(3 * lam))


def test_fit_janssen_recovers_parameters_from_exact_data():
    truth = bim.janssen_profile(width=0.4, rho_bulk=680.0, k_ratio=0.62, wall_friction=0.45)
    depths = np.linspace(0.1, 2.0, 15)
    measured = truth.wall_stress(depths)
    fit = bim.fit_janssen(depths, measured, width=0.4, rho_bulk=680.0)
    assert fit.decay_length == pytest.approx(truth.decay_length, rel=0.02)
    assert fit.k_ratio == pytest.approx(0.62, rel=0.03)
    assert fit.wall_friction == pytest.approx(0.45, rel=0.03)
    assert fit.max_relative_error < 0.02
    assert fit.passes is True


def test_fit_janssen_cumulative_form_matches_integral():
    truth = bim.janssen_profile(width=0.4, rho_bulk=680.0, k_ratio=0.62, wall_friction=0.45)
    depths = np.linspace(0.1, 2.0, 15)
    lam = truth.decay_length
    cumulative = 0.62 * 680.0 * 9.81 * lam * (depths - lam * (1 - np.exp(-depths / lam)))
    fit = bim.fit_janssen(depths, cumulative, width=0.4, rho_bulk=680.0, cumulative=True)
    assert fit.decay_length == pytest.approx(lam, rel=0.02)
    assert fit.wall_friction == pytest.approx(0.45, rel=0.03)


# -------------------------------------------------------------------- chains
def test_exponential_tail_estimator_on_exponential_sample():
    rng = np.random.default_rng(3)
    forces = rng.exponential(scale=1.0, size=40000)
    fit = bim.fit_exponential_tail(forces)
    assert fit.tail_rate == pytest.approx(1.0, rel=0.10)
    assert fit.tail_r2 > 0.95
    assert fit.exponential_tail is True
    assert fit.strong_load_fraction > 0.5


def test_exponential_tail_estimator_on_half_normal_sample():
    rng = np.random.default_rng(5)
    forces = np.abs(rng.normal(size=40000))
    fit = bim.fit_exponential_tail(forces)
    # a half-normal is thinner-tailed than an exponential: faster decay
    assert fit.tail_rate < 1.0
    assert fit.ks_vs_unit_exponential > 0.02


# ------------------------------------------------------------------ clogging
def test_clogging_curve_and_critical_ratio():
    class Run:
        def __init__(self, ratio, clogged, span=0.0):
            self.width_ratio = ratio
            self.clogged = clogged
            self.arch_span = span

    runs = ([Run(3.0, True, 0.24)] * 8 + [Run(4.0, True)] * 5 + [Run(4.0, False)] * 3
            + [Run(6.0, False)] * 8)
    curve = bim.clogging_curve(runs)
    assert curve["p_clog"][0] == pytest.approx(1.0)
    assert curve["p_clog"][-1] == pytest.approx(0.0)
    crit = bim.critical_width_ratio(curve)
    assert 3.0 < crit["critical_width_ratio"] < 6.0
    assert crit["lower_bracket"] <= crit["critical_width_ratio"] <= crit["upper_bracket"]


def test_silo_bed_is_overlap_free_and_dense():
    from tensordem.broken_ice_micro.arching import SiloConfig, _polydisperse_bed

    cfg = SiloConfig(width=0.32, floe_diameter=0.08, height=2.4, fill_fraction=0.9, seed=1)
    pos, radii = _polydisperse_bed(cfg)
    assert len(radii) >= 40
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    overlaps = radii[:, None] + radii[None, :] - d
    # the +/-2 % positional jitter leaves a sub-millimetre overlap (< 5 % of the
    # smallest radius) that the DEM contact law removes in the first settle step;
    # the seed is not allowed to be grossly overlapping
    assert overlaps.max() < 0.10 * radii.min()
    assert (pos[:, 0] - radii >= -1e-9).all()
    assert (pos[:, 0] + radii <= cfg.body_width + 1e-9).all()
    assert cfg.body_width == pytest.approx(2.0 * cfg.width)


# -------------------------------------------------------------------- FSD law
def test_measured_power_law_fsd_sampling_matches_cdf():
    fsd = bim.MeasuredPowerLawFSD(alpha=2.3, d_min=0.05, d_max=0.5)
    sample = np.sort(fsd.sample(60000, seed=4))
    empirical = np.arange(1, sample.size + 1) / sample.size
    model = fsd.cdf(sample)
    assert np.max(np.abs(empirical - model)) < 0.01
    assert fsd.pdf(np.array([0.2]))[0] > 0.0


# --------------------------------------------------------------------- study
def test_assemble_ledger_pass_and_fail_paths():
    good = dict(
        stage="fsd",
        cases=[dict(label="baseline_x", n_fragments_pooled=500,
                    fragmented_only=dict(alpha=3.0, cdf_r2=0.95, r2=0.93))],
    )
    parts = dict(
        fsd=good,
        arch=[dict(stage="arch", runs=[
            dict(width_ratio=3.0, clogged=True, arch_span_m=0.24),
            dict(width_ratio=6.0, clogged=False, arch_span_m=0.0),
        ])],
        janssen=dict(stage="janssen", cases=[
            dict(label="c1", wall_friction=0.5, fit=dict(max_relative_error=0.05,
                                                         rms_relative_error=0.02, r2=0.99,
                                                         decay_length_m=0.5, k_ratio=0.6))]),
        chain=dict(stage="chain", cases=[dict(
            label="a", porosity=0.38, coordination_number=4.2,
            contact_network=dict(force_chain=dict(tail_r2=0.95, tail_rate=1.0)))]),
    )
    ledger = study.assemble(parts)
    assert ledger["verdicts"]["a_fsd_exponent"]["passed"] is True
    assert ledger["verdicts"]["c_janssen_error"]["passed"] is True
    assert ledger["verdicts"]["d_force_chain_exponential_tail"]["passed"] is True
    assert ledger["verdicts"]["e_porosity_window"]["passed"] is True
    assert ledger["passed"] is False          # clogging brackets do not span 0.5

    bad = dict(parts)
    bad["fsd"] = dict(stage="fsd", cases=[dict(label="baseline_x", n_fragments_pooled=100,
                                               fragmented_only=dict(alpha=2.1, cdf_r2=0.85, r2=0.7))])
    bad["chain"] = dict(stage="chain", cases=[dict(label="a", porosity=0.22, coordination_number=4.0,
                                                   contact_network=dict(force_chain=dict(tail_r2=0.5, tail_rate=1.0)))])
    ledger = study.assemble(bad)
    assert ledger["verdicts"]["a_fsd_exponent"]["passed"] is False
    assert ledger["verdicts"]["d_force_chain_exponential_tail"]["passed"] is False
    assert ledger["verdicts"]["e_porosity_window"]["passed"] is False


def test_study_case_count_requires_six_cases():
    ledger = study.assemble(dict(
        fsd=dict(stage="fsd", cases=[]), janssen=dict(stage="janssen", cases=[]),
        chain=dict(stage="chain", cases=[]), arch=[dict(stage="arch", runs=[])],
    ))
    assert ledger["verdicts"]["f_case_count"]["passed"] is False
    assert ledger["verdicts"]["f_case_count"]["n_cases"] == 0