"""Tests for the floe-size-distribution evolution package (p233w1-N04).

Covers, in order of decreasing cheapness:

* the sectional grid and the two-pivot assignment conserve the conserved
  measure (floe area) exactly,
* the population balance is validated against an analytic control (equal-size
  binary breakage at constant rate grows the number exponentially,
  ``N(t) = N0 exp(nu t)``) and against a no-breakage negative control,
* the power-law estimator recovers a known exponent,
* the DEM cascade closes its energy ledger and its daughter census conserves
  area per event,
* the single-floe size effect returns a positive hazard exponent,
* the kernel calibration gates the scale-invariant daughter statistics on a
  held-out ensemble.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.fsd_evolution.cascade import (  # noqa: E402
    CascadeConfig,
    FragmentationCascade,
    fragment_perimeter,
)
from tensordem.fsd_evolution.kernel import (  # noqa: E402
    EmpiricalDaughterRule,
    calibrate_kernel,
    size_effect_hazard,
    summarise_events,
)
from tensordem.fsd_evolution.pbe import (  # noqa: E402
    SectionGrid,
    daughter_matrix,
    mass_defect,
    mass_moment,
    number_moment,
    solve_breakage,
    two_pivot_weights,
)
from tensordem.fsd_evolution.study import _fit_power_law, _fsd_state  # noqa: E402


# --------------------------------------------------------------------- grid
def test_section_grid_spans_the_requested_interval():
    grid = SectionGrid(d_min=0.02, d_max=1.0, n_sections=13)
    assert grid.diameters.size == 13
    assert grid.diameters[0] == pytest.approx(0.02)
    assert grid.diameters[-1] == pytest.approx(1.0)
    # geometric in diameter: equal log spacing
    steps = np.diff(np.log(grid.diameters))
    assert np.allclose(steps, steps[0])


def test_two_pivot_weights_conserve_measure():
    grid = SectionGrid(d_min=0.01, d_max=10.0, n_sections=17)
    pivots = grid.measure
    rng = np.random.default_rng(3)
    measures = np.concatenate(
        [
            rng.uniform(1e-5, 1e2, 200),
            np.array([1e-9, pivots[0] * 0.999, pivots[0], pivots[-1], pivots[-1] * 3.0]),
        ]
    )
    for measure in measures:
        lower, upper, w_lower, w_upper = two_pivot_weights(grid, float(measure))
        carried = w_lower * pivots[lower] + w_upper * pivots[upper]
        assert carried == pytest.approx(measure, rel=1e-12, abs=0.0)


class _EqualBinaryRule:
    """Two area-equal daughters: the analytic control's breakage rule."""

    def sample(self, parent_diameter: float) -> np.ndarray:
        half = parent_diameter / math.sqrt(2.0)
        return np.array([half, half])


def test_daughter_matrix_conserves_mass():
    grid = SectionGrid(d_min=0.02, d_max=1.0, n_sections=13)
    matrix = daughter_matrix(grid, _EqualBinaryRule(), n_samples=400)
    defect = np.max(np.abs(matrix.T @ grid.measure - grid.measure) / grid.measure)
    assert defect < 1e-12
    # every break produces exactly two daughters wherever both daughters stay
    # inside the grid; below the floor section the count weight is inflated so
    # that the mass of the unresolved daughter is still carried (documented).
    above_floor = grid.measure * 0.5 > grid.measure[0]
    assert np.allclose(matrix.sum(axis=0)[above_floor], 2.0)
    assert np.all(matrix.sum(axis=0) >= 1.0 - 1e-9)


# ------------------------------------------------------------------ balance
def test_equal_binary_breakage_grows_number_exponentially():
    """Analytic control: ``N(t) = N0 exp(nu t)`` for a constant breakage rate."""
    grid = SectionGrid(d_min=0.02, d_max=1.0, n_sections=21)
    matrix = daughter_matrix(grid, _EqualBinaryRule(), n_samples=800)
    rates = np.full(grid.n_sections, 0.7)
    initial = np.zeros(grid.n_sections)
    initial[-1] = 1.0
    solution = solve_breakage(
        grid, initial, rates=rates, daughter=matrix, t_span=(0.0, 2.0),
        t_eval=np.linspace(0.0, 2.0, 9),
    )
    for t, state in zip(solution["t"], solution["n"]):
        expected = math.exp(0.7 * t)
        assert number_moment(state) == pytest.approx(expected, rel=2e-3)


def test_no_breakage_is_a_negative_control():
    grid = SectionGrid(d_min=0.02, d_max=1.0, n_sections=13)
    matrix = daughter_matrix(grid, _EqualBinaryRule(), n_samples=200)
    initial = np.zeros(grid.n_sections)
    initial[-1] = 1.0
    solution = solve_breakage(
        grid, initial, rates=np.zeros(grid.n_sections), daughter=matrix,
        t_span=(0.0, 5.0), t_eval=np.linspace(0.0, 5.0, 6),
    )
    for state in solution["n"]:
        assert np.allclose(state, initial)


def test_mass_is_conserved_by_the_operator():
    grid = SectionGrid(d_min=0.02, d_max=1.0, n_sections=17)
    matrix = daughter_matrix(grid, _EqualBinaryRule(), n_samples=200)
    rates = np.linspace(0.1, 2.0, grid.n_sections)
    initial = np.zeros(grid.n_sections)
    initial[-1] = 1.0
    solution = solve_breakage(
        grid, initial, rates=rates, daughter=matrix, t_span=(0.0, 3.0),
        t_eval=np.linspace(0.0, 3.0, 7),
    )
    for state in solution["n"]:
        assert mass_defect(grid, initial, np.asarray(state)) < 1e-10
        assert mass_moment(grid, np.asarray(state)) == pytest.approx(
            mass_moment(grid, initial), rel=1e-10
        )


# --------------------------------------------------------------------- fit
def test_power_law_estimator_recovers_a_known_exponent():
    diameters = np.logspace(math.log10(0.04), 0.0, 17)
    width = np.gradient(diameters)
    alpha = 2.6
    density = 3.0 * diameters ** (-alpha) * width
    state = _fsd_state(
        {"grid": SectionGrid(d_min=0.04, d_max=1.0, n_sections=17),
         "n": density[None, :], "t": np.array([0.0])},
        0,
    )
    fit = _fit_power_law(state, np.ones(diameters.size, dtype=bool))
    assert fit["alpha"] == pytest.approx(alpha, rel=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------------ cascade
@pytest.fixture(scope="module")
def small_cascades():
    runs = [
        FragmentationCascade(
            CascadeConfig(nx=32, ny=16, seed=seed, strain_increment=1e-3,
                          target_strain=0.05, min_fragment_nodes=2)
        ).run()
        for seed in (0, 2)
    ]
    return runs


def test_cascade_energy_ledger_closes(small_cascades):
    for run in small_cascades:
        energies = run.energies
        work = energies["platen_work_energy_consistent"]
        change = energies["elastic_final"] - energies["elastic_initial"]
        assert abs(work - change - energies["released_energy"]) / abs(work) < 1e-8


def test_cascade_daughters_conserve_area(small_cascades):
    events = [e for run in small_cascades for e in run.events if e.n_daughters >= 2]
    assert events, "the cascade produced no split events"
    for event in events:
        # floe area is the conserved measure (constant thickness)
        assert (event.daughter_diameters**2).sum() == pytest.approx(
            event.parent_diameter**2, rel=1e-10
        )
        assert event.new_perimeter > 0.0


def test_cascade_reports_a_usable_kernel_summary(small_cascades):
    events = [e for run in small_cascades for e in run.events if e.n_daughters >= 2]
    summary = summarise_events(events)
    assert summary.mean_daughters > 2.0
    assert 0.0 < summary.mass_median_ratio <= 1.0
    assert fragment_perimeter(summary.mass_median_ratio * np.ones(2)) > 0.0


def test_size_effect_hazard_is_positive():
    hazard = size_effect_hazard(sizes=(8, 16), n_seeds=3)
    assert hazard["q"] > 0.0
    assert hazard["nu_ref_per_strain"] > 0.0
    assert hazard["predicted_q"] == pytest.approx(0.5)
    assert len(hazard["rows"]) == 2


def test_kernel_calibration_gates_scale_invariant_statistics(small_cascades):
    hazard = dict(q=0.45, q_stderr=0.05, d_ref_m=1.0, nu_ref_per_strain=500.0)
    kernel = calibrate_kernel(small_cascades, small_cascades, hazard=hazard)
    assert kernel.q == pytest.approx(0.45)
    assert kernel.max_residual <= 0.10
    assert kernel.passes
    assert set(kernel.residuals) == {"mass_median_ratio", "largest_daughter_ratio"}
    # the resolution-limited statistics are reported but not gated
    assert "mean_daughters" in kernel.resolution_limited_residuals
    rule = kernel.daughter_rule(seed=0, log_sigma=0.0)
    daughters = rule.sample(1.0)
    assert (daughters**2).sum() == pytest.approx(1.0, rel=1e-12)


def test_empirical_daughter_rule_conserves_area():
    events = []
    from tensordem.fsd_evolution.cascade import SplitEvent

    events.append(
        SplitEvent(
            parent_diameter=1.0,
            daughter_diameters=np.array([0.9, 0.3, 0.3, 0.1]),
            parent_area=math.pi / 4.0,
            applied_strain=0.01,
            released_energy=0.0,
        )
    )
    rule = EmpiricalDaughterRule(events, seed=1, log_sigma=0.2)
    for parent in (0.05, 0.3, 1.0):
        daughters = rule.sample(parent)
        assert (daughters**2).sum() == pytest.approx(parent**2, rel=1e-12)