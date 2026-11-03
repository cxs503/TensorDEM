"""Tests for the periodic granular-gas Haff's-law verification harness.

Physics anchors:
* single-impact restitution must reproduce the requested coefficient (two-body
  verification already covered by ``tensordem.contact_restitution``);
* granular temperature in the homogeneous cooling state must follow
  ``1/sqrt(T) = 1/sqrt(T0) + (A/2) t`` (Haff 1983), i.e. ``alpha = 2``;
* the cooling coefficient must scale with the dissipation factor ``1-e**2``;
* with ``e = 1`` nothing dissipates (negative control);
* the kinetic-energy loss must equal the contact-damper work (ledger).
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.granular_gas import (GranularGas, GranularGasConfig,
                                    fit_haff_exponent,
                                    fit_inverse_sqrt_temperature,
                                    kinetic_theory_cooling_coefficient,
                                    maxwell_2d_diagnostics)


@pytest.fixture(autouse=True)
def _single_thread():
    """Small dense tensors: thread oversubscription costs ~40x."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _cool(phi: float, e: float, *, n_particles: int = 120, target_ratio: float = 0.55,
          seed: int = 3) -> dict:
    cfg = GranularGasConfig(n_particles=n_particles, packing_fraction=phi,
                            restitution=e, target_overlap_fraction=0.05,
                            steps_per_contact=25, seed=seed)
    gas = GranularGas(cfg)
    A_kt = kinetic_theory_cooling_coefficient(e, cfg.number_density, cfg.diameter, cfg.mass)
    tau_kt = 2.0 / (A_kt * math.sqrt(cfg.initial_temperature))
    T0, ke0 = gas.kinetic_temperature(), gas.kinetic_energy()
    times, temps = [], []
    slab = max(1, int(0.01 * tau_kt / cfg.dt))
    while gas.kinetic_temperature() > target_ratio * T0 and gas.time < 40.0 * tau_kt:
        gas.run(slab, sample_every=1)
        times.append(gas.time)
        temps.append(gas.kinetic_temperature())
    lo = next(i for i, T in enumerate(temps) if T <= 0.85 * T0)
    inv = fit_inverse_sqrt_temperature(times[lo:], temps[lo:])
    exp = fit_haff_exponent(times[lo:], temps[lo:], tau=inv["tau_haff"])
    nu_measured = 2.0 * gas.collision_onsets / (cfg.n_particles * gas.time)
    nu_kt = cfg.number_density * cfg.diameter * math.sqrt(
        math.pi * 0.5 * (T0 + temps[-1]) / cfg.mass)
    return {
        "config": cfg, "gas": gas, "A": inv["A"], "A_kt": A_kt,
        "inverse_sqrt_r2": inv["r_squared"], "alpha": exp["alpha"],
        "exp_r2": exp["r_squared"], "temperature_ratio": temps[-1] / T0,
        "ledger_gap": abs((ke0 - gas.kinetic_energy()) - gas.damping_work)
        / max(ke0 - gas.kinetic_energy(), 1e-300),
        "nu_ratio": nu_measured / nu_kt,
    }


def test_pair_restitution_matches_requested_coefficient() -> None:
    """The gas contact law must realise the requested restitution (e = 0.70)."""
    for e_target in (0.70, 0.90):
        cfg = GranularGasConfig(n_particles=2, packing_fraction=0.02,
                                restitution=e_target, target_overlap_fraction=0.05,
                                steps_per_contact=40, seed=0)
        gas = GranularGas(cfg)
        gas.positions = torch.tensor([[0.0, 0.0], [cfg.diameter * 3.0, 0.0]],
                                     dtype=torch.float64)
        gas.velocities = torch.tensor([[0.05, 0.0], [-0.05, 0.0]], dtype=torch.float64)
        v_in = 0.10
        realized = None
        for _ in range(int(40 * cfg.contact_time / cfg.dt)):
            gas.step()
            rel = float(gas.velocities[1][0] - gas.velocities[0][0])
            if rel > 0.0 and float(gas.positions[1][0] - gas.positions[0][0]) > cfg.diameter:
                realized = rel / v_in
                break
        assert realized is not None, "particles never separated"
        assert abs(realized - e_target) < 0.02, (realized, e_target)


def test_haff_law_exponent_and_inverse_sqrt_linearity() -> None:
    """Haff's law: 1/sqrt(T) is linear in time and the exponent is 2."""
    case = _cool(0.30, 0.7)
    assert case["inverse_sqrt_r2"] > 0.98, case["inverse_sqrt_r2"]
    assert 1.75 <= case["alpha"] <= 2.15, case["alpha"]
    assert case["exp_r2"] > 0.98, case["exp_r2"]
    assert 0.30 <= case["temperature_ratio"] <= 0.85


def test_cooling_rate_scales_with_dissipation_factor() -> None:
    """``A`` must scale with ``1-e**2`` within the fitted window."""
    slow = _cool(0.30, 0.8, seed=7)
    fast = _cool(0.30, 0.6, seed=7)
    ratio = (fast["A"] / slow["A"])
    expected = ((1 - 0.6 ** 2) / (1 - 0.8 ** 2))
    assert abs(ratio / expected - 1.0) < 0.30, (ratio, expected)


def test_energy_ledger_closure() -> None:
    """Contact-damper work must account for the lost kinetic energy.

    The residual is the O(dt) symplectic-Euler ledger error, not physics: the
    benchmark script reports 3-5% at ``steps_per_contact=25`` and the fitted
    cooling coefficient is invariant to within 0.6% between 25 and 50 steps per
    contact (see ``docs/granular_gas_haff_law.md``).
    """
    case = _cool(0.30, 0.7, target_ratio=0.6)
    assert case["ledger_gap"] < 0.15, case["ledger_gap"]
    assert case["gas"].damping_work > 0.0


def test_elastic_gas_conserves_energy() -> None:
    """Negative control: ``e = 1`` has no dissipation channel at all."""
    cfg = GranularGasConfig(n_particles=100, packing_fraction=0.25, restitution=1.0,
                            steps_per_contact=25, seed=5)
    gas = GranularGas(cfg)
    assert cfg.damping == 0.0
    ke0 = gas.kinetic_energy()
    gas.run(800)
    assert gas.damping_work == 0.0
    # Explicit symplectic Euler on stiff elastic contacts gives a bounded
    # energy oscillation of order (omega*dt)**2 ~ 4%; nothing is dissipated.
    assert abs(gas.kinetic_energy() - ke0) / ke0 < 3e-2
    assert gas.collision_onsets > 0


def test_collision_frequency_exceeds_dilute_estimate() -> None:
    """Characterisation lock, not validation: the measured event rate sits
    ~2-3x above the dilute Boltzmann estimate, which is exactly why the
    prefactor comparison is reported as WITHHELD rather than validated."""
    case = _cool(0.30, 0.7)
    assert 1.5 < case["nu_ratio"] < 3.5, case["nu_ratio"]


def test_maxwell_diagnostics_are_isotropic() -> None:
    case = _cool(0.10, 0.7, target_ratio=0.7)
    diag = maxwell_2d_diagnostics(case["gas"].velocities, case["config"].mass)
    assert diag["anisotropy"] < 0.15, diag
    assert 2.4 < diag["kurtosis_x"] < 3.6, diag
    assert abs(diag["temperature"] / case["gas"].kinetic_temperature() - 1.0) < 1e-9


def test_analysis_helpers_recover_synthetic_haff_law() -> None:
    tau, T0, A = 3.0, 2.0e-4, 2.0 / (3.0 * math.sqrt(2.0e-4))
    times = [0.05 * i for i in range(200)]
    temps = [T0 / (1.0 + t / tau) ** 2 for t in times]
    inv = fit_inverse_sqrt_temperature(times, temps)
    assert inv["r_squared"] > 1 - 1e-9
    assert abs(inv["A"] / A - 1.0) < 1e-6
    assert abs(inv["tau_haff"] / tau - 1.0) < 1e-6
    exp = fit_haff_exponent(times, temps, tau=tau)
    assert abs(exp["alpha"] - 2.0) < 1e-6
    assert abs(exp["T0_fit"] / T0 - 1.0) < 1e-6


def test_configuration_validation() -> None:
    for kwargs in ({"restitution": 0.0}, {"restitution": 1.5},
                   {"packing_fraction": 0.0}, {"packing_fraction": 0.7},
                   {"target_overlap_fraction": 0.5}, {"steps_per_contact": 5},
                   {"n_particles": 1}):
        with pytest.raises(ValueError):
            GranularGasConfig(**kwargs)


def test_box_length_and_mass_are_consistent() -> None:
    cfg = GranularGasConfig(n_particles=100, packing_fraction=0.2)
    occupied = cfg.n_particles * math.pi * cfg.radius ** 2
    assert abs(occupied / cfg.box_length ** 2 - 0.2) < 1e-12
    assert abs(cfg.mass - cfg.density * math.pi * cfg.radius ** 2) < 1e-18
    assert cfg.dt < 0.1 * cfg.contact_time