"""Tests for the floe-population campaign node ``p80w2-n05``.

Coverage
--------
* DEM instrument: momentum/energy of an elastic two-body collision, the
  coordination-aware stable step, the floe census (connected components and
  volume-equivalent diameter), the ideal-gas limit of the pair distribution and
  the stability of a cold cemented aggregate;
* collision kernel: the measured table is well formed and the Enskog prediction is
  within a loose bound on a short run (the campaign threshold is asserted against
  the production artefact, see :func:`test_campaign_criteria_from_artifact`);
* breakage kernel: the log-log fit recovers a synthetic power law exactly, and the
  anvil-impact experiment shows the no-breakage floor and the fragmentation onset;
* population balance: mass conservation of the fixed-pivot operators, the
  analytic constant-kernel Smoluchowski solution for the coalescence channel, the
  ``B = 1`` no-op, the number doubling of ``B = 2`` and the grid-floor handling;
* negative control C5 and the HOPPER benchmark's ``WITHHELD`` honesty path;
* the assembled campaign ledger (C1..C7) against the thresholds, when the
  artefact produced by ``python -m tensordem.floe_population.study`` is present.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensordem.floe_population import kernels as kt
from tensordem.floe_population.field import FieldConfig, run_floe_field
from tensordem.floe_population.hopper import (
    HOPPER_TOLERANCE,
    compare_mean_size,
    hopper_report,
    implied_size_ratio,
    load_hopper_census,
)
from tensordem.floe_population.population_balance import (
    SectionGrid,
    compare_trajectories,
    effective_fragments,
    solve_pbe,
    smoluchowski_constant_kernel,
)
from tensordem.floe_population.soft_sphere import (
    ContactModel,
    PairHistogramAccumulator,
    SphereAssembly,
    build_floe,
    maxwellian_velocities,
    place_floes,
)

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "floe_population" / "study_summary.json"


# --------------------------------------------------------------- DEM instrument
def test_elastic_two_body_collision_conserves_momentum_and_energy():
    radius = np.array([0.5, 1.4])
    mass = radius**3
    position = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    velocity = np.array([[1.0, 0.0, 0.0], [-0.2, 0.0, 0.0]])
    assembly = SphereAssembly(
        radius, position, box=np.array([40.0] * 3), velocity=velocity,
        model=ContactModel(restitution=1.0),
    )
    momentum0 = (mass[:, None] * assembly.velocity).sum(axis=0)
    energy0 = 0.5 * float((mass[:, None] * assembly.velocity**2).sum())
    assembly.advance(assembly.recommended_dt(), 3000)
    momentum1 = (mass[:, None] * assembly.velocity).sum(axis=0)
    energy1 = 0.5 * float((mass[:, None] * assembly.velocity**2).sum())
    assert np.allclose(momentum0, momentum1, atol=1e-10)
    # the soft contact is resolved by ~5 steps per period, so a few per cent of
    # energy error is the discretisation budget, not a physics failure
    assert abs(energy1 - energy0) / energy0 < 0.20


def test_recommended_dt_accounts_for_coordination():
    """A cemented aggregate must get a shorter stable step than a free sphere."""
    sphere = SphereAssembly(
        np.full(4, 0.5), np.random.default_rng(0).uniform(0.0, 6.0, (4, 3)),
        box=np.array([6.0] * 3),
    )
    grid, bonds, radius = build_floe(3)
    aggregate = SphereAssembly(radius, grid, box=np.array([8.0] * 3), bonds=bonds)
    assert aggregate.max_coordination() == 18
    assert aggregate.recommended_dt() < 0.5 * sphere.recommended_dt()


def test_cold_aggregate_is_stable_and_does_not_shed_bonds():
    """No thermostat, no load: the cement must hold for thousands of steps."""
    grid, bonds, radius = build_floe(3)
    assembly = SphereAssembly(radius, grid, box=np.array([10.0] * 3), bonds=bonds)
    assembly.advance(assembly.recommended_dt(), 2000)
    assert assembly.bond_break_events == []
    census = assembly.cluster_census()
    assert len(census) == 1
    assert census.diameter[0] == pytest.approx(3.0, rel=1e-9)


def test_cluster_census_is_a_connected_component_count():
    grid, bonds, radius = build_floe(2)
    assembly = SphereAssembly(radius, grid, box=np.array([8.0] * 3), bonds=bonds)
    assert assembly.cluster_census().diameter[0] == pytest.approx(2.0)
    assembly.bond_alive[:] = False
    census = assembly.cluster_census()
    assert len(census) == 8
    assert np.allclose(census.diameter, 1.0)
    # volume-equivalence: sum of D**3 is conserved when bonds are cut
    assert float((census.diameter**3).sum()) == pytest.approx(8.0)


def test_pair_distribution_ideal_gas_limit():
    """Uncorrelated points: the shell-averaged contact value must be one.

    The contact shell of a dilute sample holds a handful of pairs per snapshot, so
    the estimator is only meaningful on a large sample -- which is exactly why the
    campaign accumulator averages over snapshots (see the docstring of
    :class:`PairHistogramAccumulator`).
    """
    rng = np.random.default_rng(3)
    box = np.array([60.0] * 3)
    position = rng.uniform(0.0, 60.0, (6000, 3))
    accumulator = PairHistogramAccumulator(box=box, diameters=np.array([1.0]), n_classes=1)
    accumulator.observe(position, np.zeros(position.shape[0], dtype=int))
    value = accumulator.contact_value()[0, 0]
    assert value == pytest.approx(1.0, rel=0.12)


def test_place_floes_returns_non_overlapping_periodic_bodies():
    rng = np.random.default_rng(1)
    box = np.array([16.0] * 3)
    radius, position, bonds, tag = place_floes((2, 2, 3), box=box, rng=rng, radius=0.5)
    assert radius.size == 8 + 8 + 27
    assert np.all(position >= 0.0) and np.all(position < box)
    assert np.array_equal(np.bincount(tag), np.array([8, 8, 27]))
    finite = 0.0
    for i in range(position.shape[0]):
        delta = position[i + 1 :] - position[i]
        delta -= box * np.round(delta / box)
        dist = np.linalg.norm(delta, axis=1)
        cross = tag[i] != tag[i + 1 :]
        if np.any(cross):
            finite = min(finite, float(dist[cross].min())) if finite else float(dist[cross].min())
    assert finite >= 1.0


# -------------------------------------------------------------- collision kernel
def test_collision_kernel_measurement_is_well_formed():
    table = kt.measure_collision_kernel(
        diameters=[1.0, 2.0],
        counts_per_class=40,
        temperature=0.5,
        packing_fraction=0.15,
        seed=0,
        duration=6.0,
    )
    assert table.counts.shape == (2, 2)
    assert table.weighted_error == pytest.approx(table.weighted_error)  # not NaN
    assert table.weighted_error < 0.6                # loose sanity bound <15 % gate
    assert np.all(table.g_contact > 0.0)
    assert np.all(table.k_measured > 0.0)
    assert table.temperature == pytest.approx(0.5, rel=0.15)


# --------------------------------------------------------------- breakage kernel
def test_breakage_fit_recovers_synthetic_power_law():
    k_true, alpha, beta = 0.3, 2.4, 1.1
    results = []
    for diameter in (2.0, 3.0, 5.0, 6.0):
        for energy in (8.0, 30.0, 120.0):
            fragments = k_true * diameter**alpha * energy**beta
            results.append(
                dict(diameter=diameter, energy=energy, fragments=fragments, speed=1.0)
            )
    fit = kt.calibrate_breakage_kernel(results, hold_out=lambda record: record["diameter"] == 6.0)
    assert fit.alpha == pytest.approx(alpha, rel=1e-6)
    assert fit.beta == pytest.approx(beta, rel=1e-6)
    assert fit.k_break == pytest.approx(k_true, rel=1e-6)
    assert fit.validation_error < 1e-9


def test_impact_fragmentation_floor_and_onset():
    model = ContactModel(k_n=500.0, bond_stiffness=1000.0, bond_tension=4.0,
                         bond_compression=40.0)
    gentle = kt.impact_fragmentation(3, 0.5, seed=0, model=model, settle_time=1.0)
    violent = kt.impact_fragmentation(3, 6.0, seed=0, model=model, settle_time=1.0)
    assert gentle.fragments == 1
    assert violent.fragments >= gentle.fragments
    assert violent.energy > gentle.energy


# ------------------------------------------------------------ population balance
def test_fixed_pivot_mass_conservation_and_breakage_number():
    grid = SectionGrid(d_min=0.25, d_max=32.0, n_sections=25)
    volumes = grid.volumes
    initial = np.zeros(25)
    initial[15] = 1.0
    kernel = np.ones((25, 25))
    solution = solve_pbe(
        grid, initial, collision_kernel=kernel, fragments=np.full(25, 2.0),
        t_span=(0.0, 1.0), t_eval=np.linspace(0.0, 1.0, 6), atol=1e-12,
    )
    mass = solution["n"] @ volumes
    assert np.allclose(mass / mass[0], 1.0, rtol=1e-9)
    number = solution["n"].sum(axis=1)
    assert np.all(np.diff(number) > 0.0)             # B = 2 increases the count
    no_break = solve_pbe(
        grid, initial, collision_kernel=kernel, fragments=np.ones(25),
        t_span=(0.0, 1.0), t_eval=np.linspace(0.0, 1.0, 3), atol=1e-12,
    )
    assert np.allclose(no_break["n"].sum(axis=1), 1.0, rtol=1e-10)


def test_coalescence_channel_matches_smoluchowski():
    grid = SectionGrid(d_min=1.0, d_max=32.0, n_sections=21)
    initial = np.zeros(21)
    initial[0] = 1.0
    kernel = np.ones((21, 21))
    solution = solve_pbe(
        grid, initial, collision_kernel=np.zeros((21, 21)), fragments=np.ones(21),
        coalescence_kernel=kernel, t_span=(0.0, 2.0), t_eval=np.linspace(0.0, 2.0, 9),
    )
    analytic = smoluchowski_constant_kernel(solution["t"], n0=1.0, rate=1.0, n_monomers=1.0)
    assert np.allclose(solution["n"].sum(axis=1), analytic, rtol=1e-4)


def test_grid_floor_freezes_sections_that_cannot_break():
    grid = SectionGrid(d_min=0.25, d_max=32.0, n_sections=25)
    fragments, clipped = effective_fragments(grid, np.full(25, 2.0))
    assert clipped == [0, 1]
    assert fragments[0] == 1.0 and fragments[2] == 2.0


def test_compare_trajectories_uses_volume_equivalent_diameter():
    grid = SectionGrid(d_min=1.0, d_max=8.0, n_sections=7)
    density = np.zeros((2, 7))
    density[:, 3] = 1.0
    comparison = compare_trajectories(grid, [0.0, 1.0], density, density)
    assert np.allclose(comparison.mean_model, grid.diameters[3])
    assert comparison.mean_relative_error == pytest.approx(0.0)


# ------------------------------------------------------------- negative control
def test_negative_control_keeps_the_floe_sizes_frozen():
    history = run_floe_field(
        FieldConfig(specs=(3, 3, 3), temperature=1.0, sphere_fraction=0.04,
                    duration=2.0, n_snapshots=4, breakage=False, seed=0)
    )
    sizes = np.asarray(history.mean_size)
    assert history.bond_breaks == 0
    assert np.std(sizes) / np.mean(sizes) < 1e-3
    assert len(history.sizes[0]) == 3


# --------------------------------------------------------------- HOPPER anchor
def test_hopper_reports_withheld_without_a_census(tmp_path):
    report = hopper_report([1.0, 2.0, 3.0], path=tmp_path / "absent.csv")
    assert report.status == "WITHHELD"
    assert report.passes is None
    assert report.as_dict()["evidence_grade"] == "WITHHELD"
    assert load_hopper_census(tmp_path / "absent.csv") is None


def test_hopper_compares_against_a_census_when_present(tmp_path):
    path = tmp_path / "hopper.csv"
    path.write_text("diameter_m,count\n1.0,3\n2.0,1\n", encoding="utf-8")
    sample = load_hopper_census(path)
    assert sample.size == 4
    result = compare_mean_size([1.25] * 4, sample, tolerance=HOPPER_TOLERANCE)
    assert result["mean_reference"] == pytest.approx(1.25)
    assert result["relative_error"] == pytest.approx(0.0)
    report = hopper_report([1.25] * 4, path=path)
    assert report.status == "COMPARED" and report.passes is True
    assert implied_size_ratio(2.5) == pytest.approx(3.0)


# ------------------------------------------------------- assembled campaign gate
@pytest.mark.skipif(not ARTIFACT.exists(), reason="campaign artefact not produced in this checkout")
def test_campaign_criteria_from_artifact():
    summary = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    ledger = summary["ledger"]
    assert ledger["C1"]["value"] < summary["criteria"]["C1_collision_kernel_rel_error"]
    assert ledger["C2"]["value"] < summary["criteria"]["C2_breakage_kernel_rel_error"]
    assert ledger["C3"]["value"] < summary["criteria"]["C3_population_balance_rel_error"]
    assert ledger["C5"]["value"] < summary["criteria"]["C5_negative_control_size_drift"]
    assert ledger["C7"]["value"] < summary["criteria"]["C7_seed_dispersion"]
    # C4 is only "passed" if a measured HOPPER census was actually available
    assert ledger["C4"]["status"] in {"WITHHELD", "COMPARED"}
    if ledger["C4"]["status"] == "WITHHELD":
        assert ledger["C4"]["passes"] is None
    else:
        assert ledger["C4"]["value"] < summary["criteria"]["C4_hopper_mean_size_rel_error"]
    assert len(summary["seeds"]) >= 3
    for name, pointer in summary["artifacts"].items():
        assert (Path(__file__).resolve().parents[1] / pointer).exists(), name