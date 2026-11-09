"""Bed-scale coupling: Ergun identity, static balance, settling and RZ fitting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.bedcoupling.bed import (
    FluidProperties,
    bed_force_assembly,
    bed_pressure_drop,
    body_force_per_particle,
    fit_power_law,
    relaxation_time,
    simulate_settling,
    settling_sweep,
    static_bed_force_audit,
)
from tensordem.bedcoupling.closures import (
    drag_coefficient,
    ergun_pressure_gradient,
)
from tensordem.bedcoupling.packing import (
    PackedBed,
    local_porosity,
    make_jittered_cubic_array,
    make_random_close_packing,
)

D = 1.0e-3
FLUID = FluidProperties(rho_fluid=998.0, rho_particle=2650.0, mu=1.0e-3)


@pytest.fixture(scope="module")
def dense_bed() -> PackedBed:
    return make_random_close_packing(600, D, 0.40, seed=424242, final_relax_iterations=4000)


def test_mean_voidage_pressure_drop_reproduces_ergun(dense_bed: PackedBed) -> None:
    """``sum(F_drag)/V_bed`` with the mean voidage must be Ergun to round-off."""
    for u_sup in (1e-3, 1e-2, 5e-2):
        ref = ergun_pressure_gradient(
            u_sup, diameter=D, porosity=dense_bed.porosity, rho_fluid=FLUID.rho_fluid, mu=FLUID.mu
        )
        measured = bed_pressure_drop(dense_bed, u_sup, FLUID)
        assert measured == pytest.approx(ref.total, rel=1e-12)


def test_local_voidage_pressure_drop_is_within_fifteen_percent(dense_bed: PackedBed) -> None:
    eps_local = local_porosity(dense_bed.positions, D, dense_bed.box_length, 2.5 * D)
    for u_sup in (2e-3, 5e-2):
        ref = ergun_pressure_gradient(
            u_sup, diameter=D, porosity=dense_bed.porosity, rho_fluid=FLUID.rho_fluid, mu=FLUID.mu
        )
        measured = bed_pressure_drop(dense_bed, u_sup, FLUID, eps_local=eps_local)
        assert abs(measured / ref.total - 1.0) <= 0.15


def test_dilute_branch_is_not_interchangeable_with_ergun(dense_bed: PackedBed) -> None:
    """The Wen--Yu branch is a *dilute* closure: it must fail a dense-bed check.

    This is the negative control that makes the packing criterion
    discriminating: a bed-scale pressure drop computed with the wrong branch is
    off by tens of percent, so no unit-conversion slip can hide behind it.
    """
    eps_local = local_porosity(dense_bed.positions, D, dense_bed.box_length, 2.5 * D)
    u_sup = 5e-2
    ref = ergun_pressure_gradient(
        u_sup, diameter=D, porosity=dense_bed.porosity, rho_fluid=FLUID.rho_fluid, mu=FLUID.mu
    )
    wrong = bed_pressure_drop(dense_bed, u_sup, FLUID, eps_local=eps_local, branch="wen_yu")
    assert wrong / ref.total - 1.0 < -0.5


def test_static_bed_residual_is_at_round_off(dense_bed: PackedBed) -> None:
    audit = static_bed_force_audit(dense_bed, FLUID)
    assert audit["residual_over_total_gravity"] <= 1e-12
    assert audit["max_per_particle_residual_over_weight"] <= 1e-12
    # buoyancy equals rho_f g V_solid, and the net weight is the submerged one
    assert audit["buoyancy_N"] == pytest.approx(audit["expected_buoyancy_N"], rel=1e-12)
    assert audit["effective_weight_N"] == pytest.approx(audit["effective_weight_analytic_N"], rel=1e-12)
    assert audit["drag_at_rest_N"] == 0.0


def test_double_counting_buoyancy_would_be_caught(dense_bed: PackedBed) -> None:
    """A spurious Archimedes term shifts the net force by O(1) of the weight."""
    audit = static_bed_force_audit(dense_bed, FLUID)
    extra = audit["buoyancy_N"]
    assert extra / audit["total_gravity_N"] > 0.1


def test_body_force_buoyancy_models_differ_by_one_power_of_voidage() -> None:
    bed = make_jittered_cubic_array(5, D, porosity=0.8)
    mix = body_force_per_particle(bed, FLUID, buoyancy_model="mixture")
    hyd = body_force_per_particle(bed, FLUID, buoyancy_model="hydrostatic")
    vp = bed.particle_volume
    assert mix[0, 2] == pytest.approx(-vp * 0.8 * FLUID.density_contrast * FLUID.gravity)
    assert hyd[0, 2] == pytest.approx(-vp * FLUID.density_contrast * FLUID.gravity)
    assert mix[0, 2] == pytest.approx(0.8 * hyd[0, 2])


def test_relaxation_time_scales_with_mass_over_coefficient() -> None:
    bed = make_jittered_cubic_array(4, D, porosity=0.9)
    tau = relaxation_time(bed, FLUID, slip_velocity=1e-2)
    coeff = float(
        drag_coefficient(1e-2, diameter=D, porosity=0.9, rho_fluid=FLUID.rho_fluid, mu=FLUID.mu)
    )
    assert tau == pytest.approx(FLUID.rho_particle * bed.particle_volume / coeff)


def test_single_particle_terminal_velocity_matches_schiller_naumann() -> None:
    bed = PackedBed(
        diameter=D, box_length=20 * D, positions=np.array([[10 * D, 10 * D, 10 * D]]), kind="single"
    )
    run = simulate_settling(bed, FLUID, n_steps=4000, fluid_mode="at_rest")
    weight = bed.particle_volume * FLUID.density_contrast * FLUID.gravity
    coeff = float(
        drag_coefficient(
            abs(run["steady_mean_vz"]), diameter=D, porosity=1.0, rho_fluid=FLUID.rho_fluid, mu=FLUID.mu
        )
    )
    assert coeff * abs(run["steady_mean_vz"]) == pytest.approx(weight, rel=5e-3)


def test_settling_is_contact_free_and_stable() -> None:
    bed = make_jittered_cubic_array(5, D, porosity=0.8)
    run = simulate_settling(bed, FLUID, n_steps=800)
    assert run["dt_over_tau"] < 0.2  # explicit Euler stability margin
    assert np.isfinite(run["mean_vz"]).all()
    assert bed.minimum_separation() >= D * (1 - 1e-12)


def test_fit_power_law_recovers_a_known_exponent() -> None:
    eps = np.array([0.5, 0.6, 0.7, 0.8])
    y = 2.5e-3 * eps**4.65
    fit = fit_power_law(eps, y)
    assert fit["exponent"] == pytest.approx(4.65, rel=1e-10)
    assert fit["prefactor"] == pytest.approx(2.5e-3, rel=1e-10)
    assert fit["r_squared"] == pytest.approx(1.0)


def test_settling_sweep_reproduces_the_closure_exponent() -> None:
    """The DEM route and the algebraic route must give the same RZ index."""
    packings = {e: make_jittered_cubic_array(5, D, porosity=e) for e in (0.8, 0.86, 0.92, 0.96)}
    sweep = settling_sweep(packings, FLUID, n_steps=1500, buoyancy_model="mixture")
    assert sweep["fit"]["r_squared"] > 0.999
    # Gidaspow/Wen-Yu at Re_t ~ 146 gives an index a little above 3
    assert 2.8 <= sweep["fit"]["exponent"] <= 3.3
    hydro = settling_sweep(packings, FLUID, n_steps=1500, buoyancy_model="hydrostatic")
    # the two buoyancy conventions differ by ~one exponent unit
    assert sweep["fit"]["exponent"] - hydro["fit"]["exponent"] == pytest.approx(0.66, abs=0.15)


def test_invalid_modes_raise() -> None:
    bed = make_jittered_cubic_array(3, D, porosity=0.9)
    with pytest.raises(ValueError):
        simulate_settling(bed, FLUID, n_steps=10, fluid_mode="nope")
    with pytest.raises(ValueError):
        simulate_settling(bed, FLUID, n_steps=10, buoyancy_model="nope")
    with pytest.raises(ValueError):
        fit_power_law(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
