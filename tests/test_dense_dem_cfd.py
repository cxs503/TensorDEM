"""Tests for the dense unresolved DEM--CFD closure package.

Coverage map (criterion -> test):

* closures and switches ....... test_schiller_naumann_*, test_di_felice_*,
  test_beta_*, test_force_model_validation, test_pressure_gradient_*,
  test_added_mass_*, test_terminal_velocity_*
* criterion 1 (momentum) ...... test_momentum_audit_conserves,
  test_momentum_audit_detects_disabled_reaction
* criterion 2 (Richardson--Zaki) test_richardson_zaki_dynamic_index,
  test_equilibrium_velocity_monotone
* criterion 3 (Ergun) ......... test_ergun_pressure_drop_two_regimes
* criterion 4 (u_mf) .......... test_minimum_fluidization_two_sided,
  test_ergun_umf_matches_gradient
* criterion 5 (grid) .......... test_grid_convergence_is_second_order
* criterion 6 (conservation) .. test_mass_conservation_flux_form,
  test_energy_ledger_identities
* criterion 7 (negative) ...... test_nc1_suspending_fluid_reaction,
  test_nc2_doubled_drag, test_nc3_suspending_pressure_gradient
* end-to-end ................. test_study_all_criteria_pass,
  test_study_json_and_figures
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensordem.cfd_dem_dense import (
    ColumnConfig,
    ForceModel,
    added_mass_force,
    beta,
    buoyant_weight,
    di_felice_chi,
    drag_density,
    equilibrium_superficial_velocity,
    ergun_gradient,
    ergun_minimum_fluidization_velocity,
    fixed_bed_pressure_drop,
    fluid_pressure_gradient,
    graded_bed_pressure_drop,
    grid_convergence_study,
    jittered_lattice_positions,
    minimum_fluidization_study,
    momentum_exchange_audit,
    newton_cd,
    per_particle_drag,
    pressure_gradient_force,
    relax_column,
    relaxation_energy_ledger,
    richardson_zaki_sweep,
    run_acceptance_study,
    schiller_naumann_cd,
    solid_mass_advection_test,
    sphere_projected_area,
    sphere_volume,
    terminal_velocity,
    write_results,
    StudyOptions,
    THRESHOLDS,
    rz_correlation_exponent,
)


QUICK = StudyOptions(
    rz_points=5,
    n_particles=32,
    n_cells=32,
    grid_cells=(20, 40, 80),
    ledger_dts=(2.0e-5, 1.0e-5),
    sensitivity_closures=False,
)

SAND = dict(diameter=1.0e-4, rho_particle=2500.0, rho_fluid=1000.0, mu=1.0e-3)


@pytest.fixture(scope="module")
def quick_study() -> dict:
    return run_acceptance_study(QUICK)


# ---------------------------------------------------------------------------
# closures
# ---------------------------------------------------------------------------
def test_schiller_naumann_stokes_limit() -> None:
    cd = float(schiller_naumann_cd(1e-8))
    assert cd == pytest.approx(24.0 / 1e-8, rel=1e-6)


def test_schiller_naumann_matches_sphere_drag_data() -> None:
    # Schlichting's tabulated sphere drag coefficients; the Schiller--Naumann
    # fit is quoted as valid to a few percent for Re < 800.
    table = {1.0: 26.5, 10.0: 4.1, 100.0: 1.07, 1000.0: 0.46}
    for re, cd_measured in table.items():
        cd_model = float(schiller_naumann_cd(re))
        assert abs(cd_model - cd_measured) / cd_measured < 0.12


def test_newton_regime_coefficient_is_constant() -> None:
    assert newton_cd() == pytest.approx(0.44)
    assert newton_cd(cd_scale=2.0) == pytest.approx(0.88)


def test_di_felice_exponent_range() -> None:
    # chi = 3.7 - 0.65 exp(...) lies in (3.1, 3.7)
    for re in (1e-2, 1.0, 10.0, 1e2, 1e4, 1e6):
        chi = float(di_felice_chi(re))
        assert 3.10 <= chi <= 3.70
    assert float(di_felice_chi(10.0)) == pytest.approx(3.127, abs=1e-3)


def test_voidage_exponent_of_empty_felice_closure_is_zero() -> None:
    from tensordem.cfd_dem_dense import voidage_exponent

    assert voidage_exponent("wen_yu", 1.0) == pytest.approx(2.65)
    assert voidage_exponent("schiller_naumann", 1.0) == 0.0
    with pytest.raises(ValueError):
        voidage_exponent("gidaspow", 1.0)


def test_ergun_beta_reproduces_classic_porous_media_form() -> None:
    """beta_modelA * eps must be the classic Ergun coefficient."""
    eps = 0.58
    u = 4.0e-4
    classic = ergun_gradient(
        u, diameter=SAND["diameter"], porosity=eps, rho_fluid=SAND["rho_fluid"], mu=SAND["mu"]
    )
    beta_model = float(
        beta(u, eps, diameter=SAND["diameter"], rho_fluid=SAND["rho_fluid"], mu=SAND["mu"], model="ergun")
    )
    assert beta_model * u / classic["total"] == pytest.approx(eps, rel=1e-12)


def test_wen_yu_beta_equals_per_particle_drag_over_volume() -> None:
    eps, u = 0.90, 3.0e-3
    dens = float(
        drag_density(u, eps, diameter=SAND["diameter"], rho_fluid=SAND["rho_fluid"], mu=SAND["mu"], model="wen_yu")
    )
    f_particle = float(
        per_particle_drag(u, eps, diameter=SAND["diameter"], rho_fluid=SAND["rho_fluid"], mu=SAND["mu"], model="wen_yu")
    )
    e = sphere_volume(SAND["diameter"]) * dens / (1.0 - eps)
    assert f_particle == pytest.approx(e, rel=1e-12)
    single = 0.5 * SAND["rho_fluid"] * float(schiller_naumann_cd(SAND["rho_fluid"] * u * SAND["diameter"] * eps / SAND["mu"])) * u * u * sphere_projected_area(SAND["diameter"]) * eps**-2.65
    assert f_particle == pytest.approx(single, rel=1e-12)


def test_gidaspow_branches_are_selected_at_the_switch() -> None:
    kw = dict(diameter=SAND["diameter"], rho_fluid=SAND["rho_fluid"], mu=SAND["mu"])
    u = 1.0e-3
    below = float(beta(u, 0.79, model="gidaspow", **kw))
    ergun_below = float(beta(u, 0.79, model="ergun", **kw))
    above = float(beta(u, 0.81, model="gidaspow", **kw))
    wen_above = float(beta(u, 0.81, model="wen_yu", **kw))
    assert below == pytest.approx(ergun_below, rel=1e-12)
    assert above == pytest.approx(wen_above, rel=1e-12)
    assert abs(above - below) / below > 0.05


def test_force_model_validation_and_description() -> None:
    fm = ForceModel(drag_model="di_felice", added_mass=False, cd_scale=2.0)
    described = fm.describe()
    assert described["drag_model"] == "di_felice"
    assert described["double_counting_risk"] is False
    with pytest.raises(ValueError):
        ForceModel(drag_model="not_a_closure")
    with pytest.raises(ValueError):
        ForceModel(cd_scale=0.0)
    with pytest.raises(ValueError):
        ForceModel(cd_model="mars")
    assert fm.replace(added_mass=True).added_mass is True
    assert fm.added_mass is False  # immutable


# ---------------------------------------------------------------------------
# pressure gradient and added mass
# ---------------------------------------------------------------------------
def test_pressure_gradient_is_hydrostatic_in_empty_cells() -> None:
    grad = float(fluid_pressure_gradient(1.0, 0.0, rho_fluid=1000.0))
    assert grad == pytest.approx(-1000.0 * 9.80665, rel=1e-12)


def test_pressure_gradient_force_sign_supports_the_bed() -> None:
    eps = 0.58
    drag = float(drag_density(1.0e-4, eps, diameter=SAND["diameter"], rho_fluid=1000.0, mu=1e-3, model="gidaspow"))
    grad = float(fluid_pressure_gradient(eps, drag, rho_fluid=1000.0))
    assert grad < 0.0
    force = float(pressure_gradient_force(sphere_volume(SAND["diameter"]), grad))
    assert force > 0.0  # upward


def test_added_mass_is_half_the_displaced_fluid_mass() -> None:
    vol = sphere_volume(SAND["diameter"])
    force = float(added_mass_force(vol, 1000.0, 2.0))
    assert force == pytest.approx(0.5 * 1000.0 * vol * 2.0)


def test_added_mass_changes_the_transient_not_the_steady_state() -> None:
    base = ColumnConfig(solid_fraction=0.5, n_particles=32, n_cells=32, length=0.02, model=ForceModel(drag_model="gidaspow"))
    without = base.with_(model=ForceModel(drag_model="gidaspow", added_mass=False))
    with_am = relax_column(base, 2.0e-2, t_end=2.0e-3, dt=2.0e-5)
    no_am = relax_column(without, 2.0e-2, t_end=2.0e-3, dt=2.0e-5)
    # added mass = 0.5 rho_f / rho_p = 0.2 of the solid mass
    assert with_am["m_added_fraction"] == pytest.approx(0.2, rel=1e-12)

    def time_to_half(hist) -> float:
        i = int(np.argmax(hist["v"] >= 0.5 * hist["v"][-1]))
        return float(hist["t"][i])

    # the transient is slower (added mass adds inertia) ...
    assert time_to_half(with_am) > 1.1 * time_to_half(no_am)
    # ... but the steady state is identical
    assert with_am["v"][-1] == pytest.approx(no_am["v"][-1], rel=1e-6)


def test_gidaspow_dilute_branch_carries_the_wen_yu_voidage_factor() -> None:
    kw = dict(diameter=SAND["diameter"], rho_fluid=1000.0, mu=1e-3)
    u = 5.0e-3
    eps = 0.90
    gid = float(beta(u, eps, model="gidaspow", **kw))
    wen = float(beta(u, eps, model="wen_yu", **kw))
    bare = float(beta(u, eps, model="schiller_naumann", **kw))
    assert gid == pytest.approx(wen, rel=1e-12)
    assert gid / bare == pytest.approx(eps**-2.65, rel=1e-12)


def test_terminal_velocity_stokes_limit() -> None:
    d = 1.0e-6
    u = terminal_velocity(diameter=d, rho_particle=2500.0, rho_fluid=1000.0, mu=1e-3)
    stokes = d * d * (2500.0 - 1000.0) * 9.80665 / (18.0 * 1e-3)
    assert u == pytest.approx(stokes, rel=5e-3)


def test_terminal_velocity_reproduces_buoyant_weight() -> None:
    u = terminal_velocity(**{k: SAND[k] for k in ("diameter", "rho_particle", "rho_fluid", "mu")})
    drag = float(
        per_particle_drag(u, 1.0 - 1e-12, diameter=SAND["diameter"], rho_fluid=1000.0, mu=1e-3, model="schiller_naumann")
    )
    weight = buoyant_weight(SAND["diameter"], rho_particle=2500.0, rho_fluid=1000.0)
    # eps -> 1 is singular in the per-particle conversion; compare with the
    # single-sphere drag law instead
    cd = float(schiller_naumann_cd(1000.0 * u * SAND["diameter"] / 1e-3))
    single = 0.5 * 1000.0 * cd * u * u * sphere_projected_area(SAND["diameter"])
    assert single == pytest.approx(weight, rel=1e-9)
    assert math.isfinite(drag)


# ---------------------------------------------------------------------------
# criterion 1: momentum
# ---------------------------------------------------------------------------
def test_momentum_audit_conserves_at_two_solid_fractions() -> None:
    rng = np.random.default_rng(3)
    z = jittered_lattice_positions(2000, 5.0e-3, jitter_fraction=1.0, seed=5)
    v = 4.0e-3 * (1.0 + 0.5 * (rng.random(2000) - 0.5))
    for eps_s in (0.30, 0.50):
        audit = momentum_exchange_audit(
            z_particle=z,
            v_particle=v,
            n_cells=40,
            length=5.0e-3,
            diameter=SAND["diameter"],
            rho_particle=2500.0,
            rho_fluid=1000.0,
            mu=1e-3,
            solid_fraction=eps_s,
            du_rel_dt=10.0,
            fluid_velocity=0.02,
        )
        assert audit["force_l1"] > 0.0
        assert audit["reaction_residual"] <= THRESHOLDS["momentum_residual"]
        assert audit["sum_force_particle"] != 0.0


def test_momentum_audit_detects_disabled_reaction() -> None:
    z = jittered_lattice_positions(500, 5.0e-3, jitter_fraction=1.0)
    v = np.full(500, 4.0e-3)
    common = dict(
        z_particle=z,
        v_particle=v,
        n_cells=40,
        length=5.0e-3,
        diameter=SAND["diameter"],
        rho_particle=2500.0,
        rho_fluid=1000.0,
        mu=1e-3,
        solid_fraction=0.3,
        fluid_velocity=0.02,
    )
    audit = momentum_exchange_audit(model=ForceModel(drag_model="gidaspow"), **common)
    broken = momentum_exchange_audit(
        model=ForceModel(drag_model="gidaspow", fluid_reaction=False), **common
    )
    assert audit["reaction_residual"] <= 1e-12
    assert broken["reaction_residual"] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# criterion 2: Richardson-Zaki
# ---------------------------------------------------------------------------
def test_equilibrium_velocity_monotone_in_porosity() -> None:
    porosities = [0.55, 0.65, 0.75, 0.85, 0.95]
    u = [
        equilibrium_superficial_velocity(
            e, model="wen_yu", **{k: SAND[k] for k in ("diameter", "rho_particle", "rho_fluid", "mu")}
        )
        for e in porosities
    ]
    assert np.all(np.diff(u) > 0.0)


def test_richardson_zaki_dynamic_index(quick_study: dict) -> None:
    rz = quick_study["criteria"]["2_richardson_zaki"]
    assert rz["passed"]
    dyn = rz["dynamic"]
    assert dyn["monotonic_increasing"]
    assert dyn["r_squared"] > 0.999
    assert dyn["n_deviation_relative"] <= THRESHOLDS["rz_index_deviation"]
    # the dynamic and algebraic routes are independent implementations
    assert rz["dynamic_vs_algebraic_relative"] < 1e-3
    # the fitted index sits in the physically expected band
    assert 3.5 < dyn["n_fit"] < 5.5


def test_rz_correlation_piecewise() -> None:
    assert rz_correlation_exponent(0.1) == pytest.approx(4.65)
    assert rz_correlation_exponent(1000.0) == pytest.approx(2.4)
    assert rz_correlation_exponent(1.0) == pytest.approx(4.4, rel=1e-6)
    assert rz_correlation_exponent(10.0) == pytest.approx(4.4 * 10.0**-0.1, rel=1e-12)


# ---------------------------------------------------------------------------
# criterion 3: Ergun
# ---------------------------------------------------------------------------
def test_ergun_pressure_drop_two_regimes(quick_study: dict) -> None:
    eg = quick_study["criteria"]["3_ergun_pressure_drop"]
    assert eg["passed"]
    assert set(eg["regimes_exercised"]) == {"viscous", "inertial"}
    low, high = eg["rows"]
    assert low["inertial_fraction"] < 0.10
    assert high["inertial_fraction"] > 0.50
    assert high["uniform"]["dp_total"] > low["uniform"]["dp_total"]


def test_fixed_bed_pressure_drop_equals_bed_weight() -> None:
    cfg = ColumnConfig(
        diameter=2.0e-4, solid_fraction=0.5, n_cells=64, length=0.2, n_particles=32,
        model=ForceModel(drag_model="gidaspow"),
    )
    u_mf = ergun_minimum_fluidization_velocity(
        diameter=cfg.diameter, porosity=cfg.porosity, rho_particle=cfg.rho_particle,
        rho_fluid=cfg.rho_fluid, mu=cfg.mu,
    )
    dp = fixed_bed_pressure_drop(cfg, u_mf)["dp_total"]
    bed_weight = (cfg.solid_fraction * cfg.rho_particle + cfg.porosity * cfg.rho_fluid) * cfg.gravity * cfg.length
    assert dp == pytest.approx(bed_weight, rel=2e-3)


# ---------------------------------------------------------------------------
# criterion 4: minimum fluidisation
# ---------------------------------------------------------------------------
def test_minimum_fluidization_two_sided(quick_study: dict) -> None:
    umf = quick_study["criteria"]["4_minimum_fluidization"]
    assert umf["passed"]
    st = umf["study"]
    assert st["two_sided"]
    assert st["drift_below_u_mf"] < 0.0 < st["drift_above_u_mf"]
    assert st["u_mf_deviation_relative"] <= THRESHOLDS["umf_deviation"]


def test_ergun_umf_matches_gradient_inversion() -> None:
    cfg = dict(diameter=SAND["diameter"], porosity=0.42, rho_particle=2500.0, rho_fluid=1000.0, mu=1e-3)
    u = ergun_minimum_fluidization_velocity(**cfg)
    grad = ergun_gradient(
        u, diameter=cfg["diameter"], porosity=cfg["porosity"], rho_fluid=1000.0, mu=1e-3
    )["total"]
    assert grad == pytest.approx((1.0 - 0.42) * (2500.0 - 1000.0) * 9.80665, rel=1e-10)


# ---------------------------------------------------------------------------
# criterion 5: grid convergence
# ---------------------------------------------------------------------------
def test_grid_convergence_is_second_order() -> None:
    study = grid_convergence_study(n_cells=(20, 40, 80, 160), length=0.2, diameter=2.0e-4)
    assert study["all_errors_nonzero"]
    assert study["min_error_ratio"] >= THRESHOLDS["grid_error_ratio_min"]
    assert min(study["observed_orders"]) > 1.5


def test_graded_bed_error_is_small_on_a_resolved_grid() -> None:
    coarse = graded_bed_pressure_drop(n_cells=20, length=0.2)
    fine = graded_bed_pressure_drop(n_cells=320, length=0.2)
    assert fine["relative_error"] < coarse["relative_error"]
    assert fine["relative_error"] < 1e-4


# ---------------------------------------------------------------------------
# criterion 6: conservation
# ---------------------------------------------------------------------------
def test_mass_conservation_flux_form_has_teeth() -> None:
    conservative = solid_mass_advection_test(t_end=0.2, dt=1e-4)
    advective = solid_mass_advection_test(t_end=0.2, dt=1e-4, conservative=False)
    assert conservative["relative_drift"] <= THRESHOLDS["conservation"]
    assert advective["relative_drift"] > 1e-2


def test_energy_ledger_identities() -> None:
    cfg = ColumnConfig(
        diameter=2.0e-4, solid_fraction=0.5, n_cells=32, length=0.2, n_particles=16,
        model=ForceModel(drag_model="gidaspow"),
    )
    led = relaxation_energy_ledger(cfg, 0.02, t_end=1.0e-3, dt=1.0e-5)
    assert led["dissipation_sign_ok"]
    assert led["power_pair_relative_residual"] <= THRESHOLDS["conservation"]
    assert led["ledger_close_rk4_relative"] <= THRESHOLDS["conservation"]
    assert led["ledger_close_potential_relative"] <= THRESHOLDS["conservation"]
    # the independent (trapezoidal) mechanical closure is quadrature limited
    assert led["mechanical_closure_relative"] < 1e-3


def test_energy_ledger_is_second_order_in_time() -> None:
    cfg = ColumnConfig(
        diameter=2.0e-4, solid_fraction=0.5, n_cells=32, length=0.2, n_particles=16,
        model=ForceModel(drag_model="gidaspow"),
    )
    coarse = relaxation_energy_ledger(cfg, 0.02, t_end=1.0e-3, dt=2.0e-5)
    fine = relaxation_energy_ledger(cfg, 0.02, t_end=1.0e-3, dt=1.0e-5)
    ratio = coarse["mechanical_closure_relative"] / fine["mechanical_closure_relative"]
    assert 3.0 < ratio < 5.0


# ---------------------------------------------------------------------------
# criterion 7: negative controls
# ---------------------------------------------------------------------------
def test_nc1_suspending_fluid_reaction(quick_study: dict) -> None:
    """NC1: without the particle->fluid reaction the bed pressure drop is wrong."""
    nc1 = quick_study["criteria"]["7_negative_controls"]["nc1_reaction_off"]
    assert nc1["deviation_relative"] >= THRESHOLDS["nc1_reaction_off_deviation_min"]
    assert nc1["dp_without_reaction"] < nc1["dp_with_reaction"]


def test_nc2_doubled_drag(quick_study: dict) -> None:
    """NC2: doubling C_D in the inertial regime scales u_t by 2**-0.5."""
    nc2 = quick_study["criteria"]["7_negative_controls"]["nc2_drag_doubled"]
    assert nc2["deviation_relative"] <= THRESHOLDS["nc2_terminal_velocity_deviation"]
    assert nc2["ratio"] == pytest.approx(2.0**-0.5, rel=THRESHOLDS["nc2_terminal_velocity_deviation"])
    # the Stokes-regime cross-check is reported and is *not* 2**-0.5
    stokes = nc2["stokes_regime_cross_check"]
    assert abs(stokes["ratio"] - 2.0**-0.5) > 0.05


def test_nc3_suspending_pressure_gradient(quick_study: dict) -> None:
    """NC3: without -V grad p the submerged weight balance is broken."""
    nc3 = quick_study["criteria"]["7_negative_controls"]["nc3_pressure_gradient_off"]
    assert nc3["deviation_relative"] >= THRESHOLDS["nc3_pressure_gradient_off_deviation_min"]
    assert nc3["u_mf_without_pressure_gradient"] > nc3["u_mf_with_pressure_gradient"]


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
def test_study_all_criteria_pass(quick_study: dict) -> None:
    failed = {k: c for k, c in quick_study["criteria"].items() if not c["passed"]}
    assert not failed, failed
    assert quick_study["all_criteria_passed"]
    assert len(quick_study["criteria"]) == 7


def test_study_json_and_figures(tmp_path, quick_study: dict) -> None:
    written = write_results(quick_study, tmp_path, figures=True)
    payload = json.loads(written["json"].read_text())
    assert payload["node"] == "p50w2-arb7250-n03-dense-dem-cfd"
    assert payload["all_criteria_passed"] is True
    assert len(written["figures"]) == 5
    for name in written["figures"]:
        path = tmp_path / name
        assert path.stat().st_size > 5_000