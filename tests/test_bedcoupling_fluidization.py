"""Unit and acceptance tests for the fluidisation / three-force-ledger layer.

Structure (each item is one behaviour, thresholds pre-registered in
``prereg/P142W3-N03_fluidization_prereg.md``):

* analytic anchors -- Wen-Yu quadratic, Ergun closed form, Stokes and
  Schiller-Naumann terminal velocities, RZ piecewise exponents;
* physics acceptance -- Ergun sweep ``Re_s in [1,1000]``, onset vs Wen-Yu, RZ
  curve, ledger residuals, resolution convergence;
* negative controls -- dropped drag term, corrupted ledger record, hydrostatic
  buoyancy, and the wrong (Wen-Yu) drag branch: none of these may pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensordem.bedcoupling.bed import FluidProperties
from tensordem.bedcoupling.closures import (
    ergun_pressure_gradient,
    rz_exponent,
    terminal_velocity_sn,
    terminal_velocity_stokes,
)
from tensordem.bedcoupling.fluidization import (
    WEN_YU_COEFFICIENTS,
    archimedes_number,
    bed_weight_gradient,
    column_dynamics,
    ergun_minimum_fluidization,
    ledger_residuals,
    resolution_convergence,
    simulated_minimum_fluidization,
    wen_yu_minimum_fluidization,
)
from tensordem.bedcoupling.packing import (
    PackedBed,
    local_porosity,
    make_jittered_cubic_array,
    make_random_close_packing,
)

WATER = FluidProperties(rho_fluid=998.0, rho_particle=2650.0, mu=1.0e-3)
GLYCERIN = FluidProperties(rho_fluid=1200.0, rho_particle=2650.0, mu=0.05)
D = 1.0e-3


# ---------------------------------------------------------------------------
# fixtures (module scoped: one packing reused by every geometric test)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bed() -> PackedBed:
    return make_random_close_packing(400, D, 0.38, seed=20261110)


@pytest.fixture(scope="module")
def eps_local(bed: PackedBed) -> np.ndarray:
    return local_porosity(bed.positions, bed.diameter, bed.box_length, 1.5 * D)


@pytest.fixture(scope="module")
def single_transition() -> PackedBed:
    return PackedBed(
        diameter=D,
        box_length=20.0 * D,
        positions=np.array([[10.0 * D, 10.0 * D, 10.0 * D]]),
        kind="single-particle",
    )


@pytest.fixture(scope="module")
def single_stokes() -> PackedBed:
    d = 0.2e-3
    return PackedBed(
        diameter=d,
        box_length=20.0 * d,
        positions=np.array([[10.0 * d, 10.0 * d, 10.0 * d]]),
        kind="single-particle",
    )


# ---------------------------------------------------------------------------
# analytic anchors
# ---------------------------------------------------------------------------
def test_bed_weight_gradient_is_submerged_bed_weight() -> None:
    """W/(A L) = (1-eps)(rho_p - rho_f) g -- buoyancy counted once."""
    eps, rho_p, rho_f, g = 0.38, 2650.0, 998.0, 9.80665
    expected = (1.0 - eps) * (rho_p - rho_f) * g
    assert bed_weight_gradient(eps, rho_particle=rho_p, rho_fluid=rho_f, gravity=g) == pytest.approx(
        expected, rel=1e-15
    )
    # an extra Archimedes term (the classic double count) would add eps*rho_f*g,
    # i.e. ~37 % of the required gradient at this density ratio: detectable, not subtle
    double_count = eps * rho_f * g
    assert 0.3 < double_count / expected < 0.45


def test_archimedes_number_definition() -> None:
    ar = archimedes_number(
        2.0e-3, rho_particle=2500.0, rho_fluid=1000.0, mu=1.0e-3, gravity=9.81
    )
    expected = 1000.0 * (2.0e-3) ** 3 * 1500.0 * 9.81 / (1.0e-3) ** 2
    assert ar == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("diameter", [0.1e-3, 0.5e-3, 1.0e-3, 5.0e-3])
def test_wen_yu_root_solves_its_own_quadratic(diameter: float) -> None:
    out = wen_yu_minimum_fluidization(
        diameter, rho_particle=2650.0, rho_fluid=998.0, mu=1.0e-3
    )
    c1 = WEN_YU_COEFFICIENTS["laminar"]
    c2 = WEN_YU_COEFFICIENTS["inertial"]
    residual = (c1 * out["re_mf"] + c2 * out["re_mf"] ** 2) / out["ar"] - 1.0
    assert abs(residual) < 1e-12
    assert out["re_mf"] > 0.0
    # the two published spellings of the same equation must agree (their
    # coefficients are rounded, so a sub-percent mismatch is expected)
    assert abs(out["re_form_mismatch"]) < 5e-3


def test_wen_yu_asymptotes() -> None:
    """Laminar limit U = Ar/c1 * mu/(rho_f d); Newton limit U = sqrt(Ar/c2) * mu/(rho_f d)."""
    fine = wen_yu_minimum_fluidization(
        0.02e-3, rho_particle=2650.0, rho_fluid=998.0, mu=1.0e-3
    )
    coarse = wen_yu_minimum_fluidization(
        8.0e-3, rho_particle=2650.0, rho_fluid=998.0, mu=1.0e-3
    )
    assert fine["ar"] < 1.0 and coarse["ar"] > 1.0e5
    assert fine["u_mf"] == pytest.approx(fine["u_mf_laminar_limit"], rel=1e-4)
    assert coarse["u_mf"] == pytest.approx(coarse["u_mf_newton_limit"], rel=0.15)


def test_ergun_minimum_fluidization_closed_form_balance() -> None:
    out = ergun_minimum_fluidization(
        0.38, diameter=D, rho_particle=2650.0, rho_fluid=998.0, mu=1.0e-3
    )
    dp = ergun_pressure_gradient(
        out["u_mf"], diameter=D, porosity=0.38, rho_fluid=998.0, mu=1.0e-3
    )
    assert dp.total == pytest.approx(out["weight_gradient"], rel=1e-12)
    assert out["viscous_part"] + out["inertial_part"] == pytest.approx(
        out["weight_gradient"], rel=1e-12
    )
    assert 0.0 < out["inertial_fraction"] < 0.5


def test_rz_exponent_piecewise_anchors() -> None:
    assert float(rz_exponent(0.1)) == pytest.approx(4.65)
    assert float(rz_exponent(0.5)) == pytest.approx(4.4 * 0.5**-0.03)
    assert float(rz_exponent(100.0)) == pytest.approx(4.4 * 100.0**-0.1)
    assert float(rz_exponent(600.0)) == pytest.approx(2.4)


# ---------------------------------------------------------------------------
# onset of fluidisation
# ---------------------------------------------------------------------------
def test_simulated_onset_reproduces_the_ergun_closed_form(bed: PackedBed) -> None:
    """Mean-voidage closure and the Ergun quadratic are the same equation."""
    sim = simulated_minimum_fluidization(bed, WATER)
    erg = ergun_minimum_fluidization(
        bed.porosity, diameter=bed.diameter, rho_particle=WATER.rho_particle,
        rho_fluid=WATER.rho_fluid, mu=WATER.mu,
    )
    assert sim["u_mf"] == pytest.approx(erg["u_mf"], rel=1e-9)
    assert sim["max_local_voidage_gap"] < 1e-9


def test_simulated_onset_local_voidage_within_10pct_of_wen_yu(
    bed: PackedBed, eps_local: np.ndarray
) -> None:
    sim = simulated_minimum_fluidization(bed, WATER, eps_local=eps_local)
    wy = wen_yu_minimum_fluidization(
        bed.diameter, rho_particle=WATER.rho_particle, rho_fluid=WATER.rho_fluid, mu=WATER.mu
    )
    assert abs(sim["u_mf"] / wy["u_mf"] - 1.0) <= 0.10
    # settling-type Reynolds number at onset, for the report
    assert 1.0 < sim["re_mf"] < 100.0


def test_simulated_onset_rejects_a_non_bracketing_interval(bed: PackedBed) -> None:
    with pytest.raises(ValueError):
        simulated_minimum_fluidization(bed, WATER, u_lo=1e-6, u_hi=2e-6)


def test_local_voidage_onset_is_not_the_mean_voidage_onset(
    bed: PackedBed, eps_local: np.ndarray
) -> None:
    """Heterogeneity must move the onset: the two routes are not the same number."""
    local = simulated_minimum_fluidization(bed, WATER, eps_local=eps_local)["u_mf"]
    mean = simulated_minimum_fluidization(bed, WATER)["u_mf"]
    assert local != mean
    assert abs(local / mean - 1.0) < 0.05


# ---------------------------------------------------------------------------
# terminal velocity (criterion 2a/2b)
# ---------------------------------------------------------------------------
def test_stokes_settling_matches_analytic(single_stokes: PackedBed) -> None:
    run = column_dynamics(single_stokes, GLYCERIN, n_steps=800, fluid_mode="imposed")
    u_stokes = terminal_velocity_stokes(
        single_stokes.diameter,
        rho_particle=GLYCERIN.rho_particle,
        rho_fluid=GLYCERIN.rho_fluid,
        mu=GLYCERIN.mu,
    )
    re_t = GLYCERIN.rho_fluid * u_stokes * single_stokes.diameter / GLYCERIN.mu
    assert re_t < 0.01
    assert -run["steady_mean_vz"] == pytest.approx(u_stokes, rel=0.05)


def test_transition_settling_matches_schiller_naumann(single_transition: PackedBed) -> None:
    run = column_dynamics(single_transition, WATER, n_steps=1200, fluid_mode="imposed")
    ref = terminal_velocity_sn(
        D, rho_particle=WATER.rho_particle, rho_fluid=WATER.rho_fluid, mu=WATER.mu
    )
    assert ref["re_terminal"] > 10.0
    assert -run["steady_mean_vz"] == pytest.approx(ref["u_terminal"], rel=0.10)


# ---------------------------------------------------------------------------
# three-force ledger (criterion 3) and its negative controls
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ledger_run():
    cell = make_jittered_cubic_array(4, D, porosity=0.80, seed=7)
    run = column_dynamics(cell, GLYCERIN, n_steps=200, fluid_mode="closed_box")
    return cell, run


def test_ledger_bookkeeping_residual_is_roundoff(ledger_run) -> None:
    _, run = ledger_run
    assert run["finite"]
    assert run["bookkeeping_residual"] < 1e-10


def test_ledger_equilibrium_residual_is_below_one_percent(ledger_run) -> None:
    _, run = ledger_run
    assert run["equilibrium_residual"] <= 0.01


def test_ledger_three_forces_cancel_in_the_steady_window(ledger_run) -> None:
    _, run = ledger_run
    total = (
        run["steady_mean_drag_N"] + run["steady_mean_pressure_N"] + run["steady_mean_gravity_N"]
    )
    assert abs(total) / run["weight_total_N"] < 1e-9
    # and the body force is the mixture buoyancy, i.e. gravity + pressure != gravity
    assert run["steady_mean_pressure_N"] > 0.0
    assert run["steady_mean_gravity_N"] < 0.0


def test_ledger_detects_a_dropped_drag_term(ledger_run) -> None:
    """Negative control N2: if drag is applied but not recorded, the ledger must fail."""
    cell, _ = ledger_run
    broken = column_dynamics(
        cell, GLYCERIN, n_steps=200, fluid_mode="closed_box", applied_terms=("pressure",)
    )
    assert broken["bookkeeping_residual"] > 10.0  # O(1) in bed weights, i.e. detected
    assert broken["equilibrium_residual"] > 0.1


def test_ledger_residual_detects_a_corrupted_record(ledger_run) -> None:
    """A ledger whose pressure record is tampered with must no longer close."""
    _, run = ledger_run
    corrupted = dict(run)
    corrupted["f_pressure"] = run["f_pressure"] + 0.01 * run["weight_total_N"]
    out = ledger_residuals(corrupted)
    assert out["bookkeeping_residual"] > 1e-3


def test_timestep_refinement_reduces_the_discretisation_residual(ledger_run) -> None:
    """Negative control N4: the dt-dependent residual must scale with dt."""
    cell, coarse = ledger_run
    fine = column_dynamics(cell, GLYCERIN, n_steps=400, dt=0.5 * coarse["dt"], fluid_mode="closed_box")
    assert fine["discretisation_residual"] < 0.7 * coarse["discretisation_residual"]
    assert coarse["bookkeeping_residual"] < 1e-10 and fine["bookkeeping_residual"] < 1e-10


@pytest.mark.parametrize(
    "kwargs",
    [{"applied_terms": ("drag", "nonsense")}, {"fluid_mode": "sieve"}],
)
def test_column_dynamics_rejects_unknown_options(kwargs) -> None:
    cell = make_jittered_cubic_array(3, D, porosity=0.86, seed=3)
    with pytest.raises(ValueError):
        column_dynamics(cell, GLYCERIN, n_steps=2, **kwargs)


# ---------------------------------------------------------------------------
# Ergun sweep (criterion 1) and the branch negative control
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("re_target", [1.0, 10.0, 100.0, 1000.0])
def test_ergun_sweep_inside_10_percent(bed: PackedBed, eps_local: np.ndarray, re_target: float) -> None:
    from tensordem.bedcoupling.bed import bed_pressure_drop

    u_sup = re_target * WATER.mu / (WATER.rho_fluid * bed.diameter)
    ref = ergun_pressure_gradient(
        u_sup, diameter=bed.diameter, porosity=bed.porosity,
        rho_fluid=WATER.rho_fluid, mu=WATER.mu,
    )
    measured = bed_pressure_drop(bed, u_sup, WATER, eps_local=eps_local)
    assert abs(measured / ref.total - 1.0) <= 0.10


def test_wrong_drag_branch_is_rejected_at_dense_porosity(
    bed: PackedBed, eps_local: np.ndarray
) -> None:
    """Negative control N3: the Wen-Yu branch is the wrong closure for eps = 0.38."""
    from tensordem.bedcoupling.bed import bed_pressure_drop

    u_sup = 5e-2
    ref = ergun_pressure_gradient(
        u_sup, diameter=bed.diameter, porosity=bed.porosity,
        rho_fluid=WATER.rho_fluid, mu=WATER.mu,
    )
    wen_yu = bed_pressure_drop(bed, u_sup, WATER, eps_local=eps_local, branch="wen_yu")
    assert abs(wen_yu / ref.total - 1.0) > 0.10


# ---------------------------------------------------------------------------
# resolution convergence (criterion 5)
# ---------------------------------------------------------------------------
def test_resolution_convergence_within_five_percent() -> None:
    conv = resolution_convergence(
        WATER,
        particle_counts=(300, 600, 1200),
        settling_n_side=(4, 5, 6),
        settling_steps=200,
        sweep_velocities=(2e-3, 5e-2),
    )
    assert conv["worst_spread"] <= 0.05
    for name, item in conv["quantities"].items():
        assert item["spread"] <= 0.05, (name, item)


def test_resolution_convergence_reports_the_local_voidage_spread() -> None:
    """The kernel statistic is reported, and it is *not* claimed to be converged."""
    conv = resolution_convergence(
        WATER,
        particle_counts=(300, 600),
        settling_n_side=(4,),
        settling_steps=100,
        sweep_velocities=(5e-2,),
    )
    stds = [r["porosity_local_std"] for r in conv["resolutions"]]
    assert len(stds) == 2
    assert all(0.0 < s < 0.1 for s in stds)
    mean_eps = [r["porosity_volume_balance"] for r in conv["resolutions"]]
    assert all(abs(e - 0.38) < 1e-9 for e in mean_eps)


def test_resolution_convergence_rejects_too_small_a_cell() -> None:
    with pytest.raises(ValueError):
        resolution_convergence(WATER, particle_counts=(40,), settling_n_side=(3,), settling_steps=10)