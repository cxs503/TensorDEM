"""Quasi-static single-bond / bonded-column rate tests.

Mission m6464 left the specimen-level rate exponent as an honest FAIL: the
platen specimen was inertia dominated (non-flat negative control, strength 7-9x
the rigid-plate bound).  ``tensordem.bond_quasistatic`` isolates the constitutive
response instead: a 2 x 2 bonded column in *tension* driven by a prescribed
displacement, with ``n_layers = 2`` leaving no free particle.  These tests pin
the claims that the campaign report makes:

* the closed forms are self-consistent (damage integral = 1 at the analytic
  failure time, Prony force = the engine's exact branch advance),
* the engine reproduces the closed form for an isolated bond,
* the brittle control is rate independent (the control that failed before),
* the emergent exponent is 1/(m+1),
* free layers add inertia and the strength error grows with it,
* the damage clock is first order in dt,
* invalid configurations are rejected.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.bond_quasistatic import (
    ChainTensionDriver,
    QuasiStaticBondConfig,
    affine_network_factor,
    brittle_force_scale,
    crossover_strain_rate,
    ductile_failure_strain,
    elastic_creep_reference,
    gauge_length,
    maxwell_wiechert_force,
    maxwell_wiechert_reference,
)
from tensordem.ice_bond_viscoelastic import MaxwellWiechertBond, advance_branches


def config(**overrides) -> QuasiStaticBondConfig:
    base = dict(
        n_layers=2,
        bond_creep_exponent=2,
        bond_creep_time_s=5.0e-3,
        bond_relaxation_time_s=(5.0e-3, 5.0e-2),
    )
    base.update(overrides)
    return QuasiStaticBondConfig(**base)


def dt_for(cfg: QuasiStaticBondConfig, rate: float, steps: int = 1000) -> float:
    """Step that lands ``steps`` integration points on the analytic failure."""
    reference = elastic_creep_reference(cfg, rate)
    stable = ChainTensionDriver(cfg).sim.dt
    return min(0.95 * stable, reference["failure_time_s"] / steps)


def run(cfg: QuasiStaticBondConfig, rate: float, steps: int = 1000,
        **run_kwargs) -> dict:
    """Run one pull test with a dt that puts ``steps`` points on the ramp."""
    dt = run_kwargs.pop("dt", None) or dt_for(cfg, rate, steps)
    driver = ChainTensionDriver(QuasiStaticBondConfig(**{**cfg.__dict__, "dt": dt}))
    return driver.run(rate, **run_kwargs)


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("exponent", [1, 2, 4])
def test_closed_form_damage_integral_is_one_at_failure(exponent):
    cfg = config(bond_creep_exponent=exponent)
    for rate in (0.5, 0.05):
        reference = elastic_creep_reference(cfg, rate)
        assert reference["regime"] == "ductile"
        assert reference["damage_at_failure"] == pytest.approx(1.0, rel=1e-7)


def test_ductile_root_meets_brittle_cap_at_crossover():
    cfg = config(bond_creep_exponent=2)
    rate_c = crossover_strain_rate(creep_time_s=cfg.bond_creep_time_s,
                                   exponent=cfg.bond_creep_exponent,
                                   breaking_strain=cfg.breaking_strain)
    assert ductile_failure_strain(
        rate_c, creep_time_s=cfg.bond_creep_time_s,
        exponent=cfg.bond_creep_exponent,
        breaking_strain=cfg.breaking_strain) == pytest.approx(
            cfg.breaking_strain, rel=1e-12)
    # Above the crossover the brittle branch must win, below it the ductile one.
    assert ductile_failure_strain(
        2.0 * rate_c, creep_time_s=cfg.bond_creep_time_s,
        exponent=cfg.bond_creep_exponent,
        breaking_strain=cfg.breaking_strain) == cfg.breaking_strain
    assert ductile_failure_strain(
        0.5 * rate_c, creep_time_s=cfg.bond_creep_time_s,
        exponent=cfg.bond_creep_exponent,
        breaking_strain=cfg.breaking_strain) < cfg.breaking_strain


def test_prony_force_matches_engine_branch_advance():
    """The closed-form ramp force must equal the engine's exact branch update."""
    cfg = config(bond_model="maxwell_wiechert")
    law = cfg.bond_law
    assert isinstance(law, MaxwellWiechertBond)
    rate = 0.2
    L0 = cfg.bond_rest_length
    dt = 2.0e-6
    elongation = torch.zeros(1, dtype=torch.float64)
    offsets = torch.zeros((1, law.n_branches), dtype=torch.float64)
    time = 0.0
    for _ in range(2000):
        time += dt
        extension = torch.full((1,), rate * L0 * time, dtype=torch.float64)
        offsets, force, _work, _dissipation = advance_branches(
            law, offsets, elongation, extension, dt)
        elongation = extension.clone()
        assert float(force[0]) == pytest.approx(
            maxwell_wiechert_force(rate, time, L0, law), rel=1e-10, abs=1e-14)


# ---------------------------------------------------------------------------
# Engine vs closed form: the isolated bond
# ---------------------------------------------------------------------------
def test_isolated_bond_reproduces_closed_form():
    cfg = config()
    rate = 0.5
    record = run(cfg, rate)
    reference = elastic_creep_reference(cfg, rate)
    assert record["failed"] is True
    assert record["failure_mode"] == 4              # creep damage, not cleavage
    assert record["inertia_index"] == 0.0           # no free particle
    assert record["failure_strain"] == pytest.approx(
        reference["failure_strain"], rel=5e-3)
    assert record["axial_bond_force_N"] == pytest.approx(
        reference["failure_force_N"], rel=5e-3)
    assert record["broken_bonds"] >= cfg.n_columns  # every axial bond


def test_brittle_control_is_rate_independent():
    """The control that invalidated the specimen campaign must be flat here."""
    cfg = config(bond_creep_exponent=0)
    fast = run(cfg, 0.5)
    slow = run(cfg, 0.05)
    for record in (fast, slow):
        assert record["failed"] is True
        assert record["failure_mode"] == 1          # tensile cleavage
        assert record["failure_strain"] == pytest.approx(
            cfg.breaking_strain, rel=2e-3)
        assert record["axial_bond_force_N"] == pytest.approx(
            brittle_force_scale(cfg), rel=2e-3)
    assert abs(fast["failure_strain"] - slow["failure_strain"]) < 2e-3 * cfg.breaking_strain


def test_emergent_exponent_is_one_over_m_plus_one():
    cfg = config(bond_creep_exponent=2)
    rates = (0.3, 0.05)
    strengths = []
    for rate in rates:
        record = run(cfg, rate)
        reference = elastic_creep_reference(cfg, rate)
        assert record["failure_mode"] == 4
        assert record["failure_strain"] == pytest.approx(
            reference["failure_strain"], rel=5e-3)
        strengths.append(math.log(record["axial_bond_force_N"]))
    slope = (strengths[0] - strengths[1]) / (math.log(rates[0]) - math.log(rates[1]))
    assert slope == pytest.approx(1.0 / (cfg.bond_creep_exponent + 1.0), abs=1e-2)


@pytest.mark.parametrize("exponent", [1, 4])
def test_exponent_follows_the_clock_exponent(exponent):
    cfg = config(bond_creep_exponent=exponent)
    rate = 0.1
    record = run(cfg, rate)
    reference = elastic_creep_reference(cfg, rate)
    assert reference["regime"] == "ductile"
    assert record["failure_strain"] == pytest.approx(
        reference["failure_strain"], rel=5e-3)


def test_maxwell_wiechert_engine_matches_reference():
    cfg = config(bond_model="maxwell_wiechert")
    rate = 0.1
    record = run(cfg, rate)
    reference = maxwell_wiechert_reference(cfg, rate)
    assert reference["regime"] == "ductile"
    assert record["failure_mode"] == 4
    assert record["failure_strain"] == pytest.approx(
        reference["failure_strain"], rel=5e-3)


# ---------------------------------------------------------------------------
# Discretisation and inertia
# ---------------------------------------------------------------------------
def test_damage_clock_is_first_order_in_dt():
    cfg = config()
    rate = 0.2
    base = dt_for(cfg, rate, steps=500)
    errors = []
    for factor in (1.0, 0.5):
        record = ChainTensionDriver(
            QuasiStaticBondConfig(**{**cfg.__dict__, "dt": base * factor})
        ).run(rate, stop_at_strain=0.5 * elastic_creep_reference(cfg, rate)["failure_strain"])
        reference = elastic_creep_reference(cfg, rate)
        closed = ((record["final_strain"] / cfg.breaking_strain) ** 2 * record["final_strain"]
                  / (3.0 * cfg.bond_creep_time_s * rate))
        # Half the failure strain carries (1/2)**(m+1) = 1/8 of the damage.
        assert closed == pytest.approx(0.125 * reference["damage_at_failure"], rel=2e-2)
        errors.append(abs(record["final_max_bond_damage"] - closed) / closed)
    assert errors[0] > errors[1]
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.35)


def test_free_layers_add_inertia_and_strength_error():
    rate = 0.1
    bounded = run(config(n_layers=2), rate)
    free = run(config(n_layers=4), rate, steps=600)
    assert bounded["inertia_index"] == 0.0
    assert free["inertia_index"] > 0.0
    assert free["max_strain_nonuniformity"] > bounded["max_strain_nonuniformity"]
    bounded_error = abs(bounded["failure_strain"]
                        / elastic_creep_reference(config(n_layers=2), rate)["failure_strain"] - 1.0)
    free_error = abs(free["failure_strain"]
                     / elastic_creep_reference(config(n_layers=4), rate)["failure_strain"] - 1.0)
    assert free_error > 10.0 * bounded_error


def test_lattice_amplification_matches_affine_geometry():
    cfg = config()
    factor = affine_network_factor(cfg)["net_per_axial_bond"]
    assert factor == pytest.approx(28.0 / 3.0, rel=1e-12)
    record = run(cfg, 0.5)
    # Within the run the bond direction cosine changes with strain, so the
    # measured amplification sits a few tenths of a percent above the rest-state
    # value; 2 % is a convergence-free bound.
    assert record["peak_force_N"] / record["axial_bond_force_N"] == pytest.approx(
        factor, rel=2e-2)


def test_gauge_length_and_brittle_scale():
    cfg = config(n_layers=3)
    assert gauge_length(cfg) == pytest.approx((cfg.n_layers - 1) * cfg.bond_rest_length)
    assert gauge_length(cfg) == pytest.approx((cfg.n_layers - 1) * 2.0 * cfg.radius)
    assert brittle_force_scale(cfg) == pytest.approx(
        cfg.bond_stiffness * cfg.breaking_strain * cfg.bond_rest_length)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        QuasiStaticBondConfig(n_layers=1)
    with pytest.raises(ValueError):
        QuasiStaticBondConfig(bond_creep_exponent=-1)
    with pytest.raises(ValueError):
        QuasiStaticBondConfig(dt=0.0)
    with pytest.raises(ValueError):
        QuasiStaticBondConfig(bond_model="maxwell_wiechert", bond_relaxation_time_s=())


def test_invalid_run_arguments_are_rejected():
    driver = ChainTensionDriver(config())
    with pytest.raises(ValueError):
        driver.run(0.0)
    with pytest.raises(ValueError):
        driver.run(-1.0)
    with pytest.raises(ValueError):
        driver.run(0.1, max_strain=1e-3)      # below the cleavage strain