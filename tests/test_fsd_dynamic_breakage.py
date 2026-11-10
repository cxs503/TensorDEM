"""Tests for the dynamic (inertial) DEM breakage kernel and FSD evolution.

Node P233W2-N04.  The tests are deliberately small (short runs, coarse grids) and
check the *physics bookkeeping* rather than the campaign numbers:

* dynamic instrument: fragment mass is conserved exactly, the inertial energy
  ledger closes and its residual converges with the time step, the daughter
  census conserves floe area;
* kernel: daughter statistics are scale invariant, the catalogue sampler
  conserves area, the breadth exponent is recovered from a planted power law;
* field: the population balance mass budget closes, the power-law fit recovers a
  planted exponent, and the melt sink follows the ``2 v_lat / d`` geometry;
* driver: the full chain runs end to end on a synthetic ensemble and reports every
  acceptance criterion.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.fsd_evolution.dynamic_breakage import (
    DynamicImpactConfig,
    run_impact,
)
from tensordem.fsd_evolution.dynamic_kernel import (
    Event,
    CatalogueRule,
    daughter_summary,
    fit_rate_law,
    number_scaling_exponent,
    weakest_link_prediction,
)
from tensordem.fsd_evolution.field import (
    FieldConfig,
    FieldState,
    build_field,
    evolve_field,
    fit_power_law,
    mass_budget_residual,
)
from tensordem.fsd_evolution.study_dynamic import run_study


# --------------------------------------------------------------- instrument
@pytest.fixture(scope="module")
def short_run():
    cfg = DynamicImpactConfig(
        nx=16, ny=8, loading="platen", strain_rate=1.0,
        dt=4.0e-6, t_end=0.02, damping_ratio=0.0, seed=0,
    )
    return run_impact(cfg)


def test_fragment_mass_is_conserved_exactly(short_run):
    assert short_run.mass_residual <= 1.0e-12
    assert short_run.parent_area == pytest.approx(float(short_run.config.nx * short_run.config.ny)
                                                 * short_run.config.node_area)


def test_daughter_area_normalisation(short_run):
    x = short_run.daughter_ratios
    assert np.sum(x**2) == pytest.approx(1.0, rel=1e-12)


def test_energy_ledger_closes(short_run):
    e = short_run.energies
    assert e["input"] > 0.0
    assert abs(short_run.ledger_residual) <= 0.10
    total = (e["ke"] + e["u_elastic"] + e["u_wall"] + e["e_dissipated"]
             + e["e_wall_dissipated"] + e["e_fracture"])
    assert total == pytest.approx(e["input"], rel=0.10)


def test_energy_ledger_residual_decreases_with_time_step():
    base = dict(nx=16, ny=8, loading="platen", strain_rate=1.0,
                t_end=0.02, damping_ratio=0.0, seed=0)
    coarse = run_impact(DynamicImpactConfig(dt=4.0e-6, **base))
    fine = run_impact(DynamicImpactConfig(dt=2.0e-6, **base))
    assert abs(fine.ledger_residual) <= abs(coarse.ledger_residual) + 1e-3
    assert abs(fine.ledger_residual) <= 0.10


def test_config_rejects_unstable_time_step():
    with pytest.raises(ValueError):
        DynamicImpactConfig(dt=1.0e-4)  # way above the leapfrog limit
    with pytest.raises(ValueError):
        DynamicImpactConfig(loading="nonsense")


def test_impact_loading_produces_bounded_speeds():
    cfg = DynamicImpactConfig(nx=12, ny=6, loading="stretch", impact_velocity=2.0,
                              dt=2.0e-6, t_end=1.0e-3)
    result = run_impact(cfg)
    assert result.energies["peak_speed_ratio"] < 50.0
    assert result.mass_residual <= 1e-12


# -------------------------------------------------------------------- kernel
def test_daughter_summary_is_scale_invariant():
    events = [Event(parent=1.0, daughters=np.array([0.6, 0.4, 0.1])),
              Event(parent=4.0, daughters=np.array([2.4, 1.6, 0.4]))]
    summary = daughter_summary(events)
    assert summary["mean_daughters"] == pytest.approx(3.0)
    assert summary["largest_daughter_ratio"] == pytest.approx(0.6)


def test_number_scaling_exponent_recovers_planted_slope():
    # C(x) ~ x**r families: daughters uniformly filling x in (0, 1) gives r = 1
    rng = np.random.default_rng(0)
    events = []
    for _ in range(200):
        x = np.sort(rng.random(6))[::-1]
        x = x / math.sqrt(float(np.sum(x**2)))
        events.append(Event(parent=1.0, daughters=x))
    fitted = number_scaling_exponent(events, x_min=0.1, x_max=0.9)
    assert fitted["r"] == pytest.approx(1.0, abs=0.15)


def test_catalogue_rule_conserves_area():
    catalogue = [np.array([0.7, 0.7]), np.array([0.6, 0.5, 0.3])]
    rule = CatalogueRule(catalogue, seed=1, log_sigma=0.2)
    for parent in (0.1, 1.0, 10.0):
        daughters = rule.sample(parent)
        assert np.sum((daughters / parent) ** 2) == pytest.approx(1.0, rel=1e-12)


def test_fit_rate_law_recovers_exponent():
    d = np.array([0.25, 0.5, 1.0, 2.0])
    nu = 0.7 * d**1.5
    fit = fit_rate_law(d, nu)
    assert fit["q"] == pytest.approx(1.5, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)


def test_weakest_link_prediction():
    assert weakest_link_prediction(4.0) == pytest.approx(0.5)
    assert weakest_link_prediction(2.0, area_exponent=3.0) == pytest.approx(1.5)


# --------------------------------------------------------------------- field
class _BinaryKernel:
    """Minimal kernel stand-in: binary daughters, power-law rate."""

    def __init__(self, q=1.0, nu_ref=0.5, d_ref=1.0):
        self.q = q
        self.nu_ref = nu_ref
        self.d_ref = d_ref

    def rate(self, diameter):
        d = np.asarray(diameter, dtype=float)
        return self.nu_ref * (d / self.d_ref) ** self.q

    def daughter_rule(self, seed=0, log_sigma=0.0):
        return CatalogueRule([np.array([0.7071, 0.7071])], seed=seed, log_sigma=log_sigma)


def test_field_mass_budget_closes():
    cfg = FieldConfig(n_sections=14, initial_area_fraction=0.05, t_end=200.0,
                      n_samples=800)
    built = build_field(_BinaryKernel(), cfg)
    evolved = evolve_field(built, t_eval=np.linspace(0.0, cfg.t_end, 401))
    assert built["daughter_mass_defect"] <= 1e-12
    assert mass_budget_residual(built, evolved["states"]) <= 1e-6


def test_melt_sink_is_inverse_diameter():
    built = build_field(_BinaryKernel(), FieldConfig(n_sections=12, n_samples=800))
    d = built["diameters"]
    mu = built["sink"]
    ratio = mu * d
    assert np.allclose(ratio, ratio[0])  # mu = 2 v_lat / d exactly


def test_power_law_fit_recovers_planted_exponent():
    grid = FieldConfig(n_sections=20, n_samples=500).grid
    d = grid.diameters
    alpha = 2.5
    density = 1e-3 * (d / d[0]) ** (-alpha)  # n(d) = density/width ~ d**-alpha
    widths = np.gradient(d)
    state = FieldState(t=0.0, density=density * widths, diameters=d,
                       widths=widths, areas=grid.measure)
    mask = state.scaling_mask()
    fit = fit_power_law(state, mask)
    assert fit["differential"]["alpha"] == pytest.approx(alpha, abs=1e-6)
    assert fit["cumulative"]["alpha"] == pytest.approx(alpha - 1.0, abs=0.05)
    assert fit["differential"]["r2"] > 0.99


def test_field_needs_strictly_positive_geometry():
    with pytest.raises(ValueError):
        FieldConfig(d_min=0.0)
    with pytest.raises(ValueError):
        FieldConfig(n_sections=4)


# -------------------------------------------------------------------- driver
def _synthetic_ensemble() -> dict:
    """A miniature impact ensemble with the same schema as the real one."""
    rows = []
    for nx, d_parent, tag, dt in ((12, 0.267, "production", 2e-6),
                                  (32, 0.713, "production", 2e-6),
                                  (48, 1.07, "production", 2e-6),
                                  (32, 0.713, "convergence", 1e-6),
                                  (48, 1.07, "convergence", 1e-6)):
        mass = 100.0 * d_parent**2
        for v0 in (0.4, 0.6, 0.8, 1.2):
            for seed in (0, 1):
                broke = v0 >= (0.6 if nx > 12 else 0.5)
                daughters = [0.7071 * d_parent, 0.7071 * d_parent] if broke else [d_parent]
                events = ([dict(t=5e-6, parent=0.7071 * d_parent,
                                daughters=[0.5 * d_parent, 0.5 * d_parent])] if broke else [])
                rows.append(dict(
                    tag=tag, loading="stretch", nx=nx, ny=nx // 2, seed=seed,
                    dt_s=dt, impact_velocity_m_s=v0, parent_diameter_m=d_parent,
                    parent_mass_kg=mass, parent_area_m2=3.0, broke=broke,
                    n_fragments=len(daughters),
                    largest_fraction=float(max(np.asarray(daughters) / d_parent) ** 2),
                    n_broken_bonds=40, t_first_break_s=(4e-6 if broke else None),
                    t_split_s=(5e-6 if broke else None), mass_residual=0.0,
                    ledger_residual=0.004, floor_mass_fraction=0.0,
                    energies=dict(input=0.5 * mass * v0**2, ke=0.3 * mass * v0**2,
                                  u_elastic=0.01 * mass * v0**2, u_wall=0.0,
                                  e_fracture=0.05 * mass * v0**2, e_dissipated=0.0,
                                  e_wall_dissipated=0.0),
                    daughters=daughters, events=events,
                ))
    return dict(dt_s=2e-6, dt_fine_s=1e-6, t_end_s=1.5e-3, sizes=[12, 32, 48],
                velocities=[0.4, 0.6, 0.8, 1.2], seeds=[0, 1],
                rows=rows, wall_time_s=1.0)


def test_study_runs_end_to_end(tmp_path):
    ensemble = _synthetic_ensemble()
    config = FieldConfig(n_sections=16, t_end=300.0, n_samples=800)
    payload = run_study(ensemble, platen=None, out_dir=tmp_path,
                        figures_dir=tmp_path / "fig", field_config=config,
                        sensitivity_sink_factors=(0.5, 1.0),
                        grid_sections=(14, 16), write_report=True)
    for key in ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6"):
        assert payload["acceptance"][key]["verdict"] in ("PASS", "FAIL")
    assert payload["acceptance"]["AC2"]["alpha"] == pytest.approx(
        payload["field"]["fit"]["differential"]["alpha"])
    assert payload["field"]["breadth"]["r"] >= 0.0
    assert payload["field"]["mass_budget_residual"] <= 1e-6
    assert payload["acceptance"]["AC4"]["verdict"] == "PASS"
    ledger = tmp_path / "P233W2_N04_ledger.json"
    assert ledger.exists()
    report = tmp_path / "P233W2_N04_dynamic_fsd.md"
    assert report.exists()
    text = report.read_text()
    assert "AC2" in text and "Error assessment" in text