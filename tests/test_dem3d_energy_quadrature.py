"""Exact-quadrature ledger arms for ``EnergyAuditedIceDEM3D``.

The historical work accounting evaluates actuator power at the start of the
step (``reaction * v * dt``).  For a penalty spring whose stored energy is
quadratic in the boundary position, that left-endpoint rule misses the
``1/2 k dx^2`` term of the potential increment: a systematic first-order bias
that dominates the ledger residual of platen-driven runs (measured, see
``docs/dem3d_energy_quadrature_closure.md``).

These tests pin the exact arm (potential increment at fixed particle
positions) to the engine's own stored energy, and pin the *sign and size* of
the legacy deficit so a future change cannot silently reintroduce the bias.
"""
from __future__ import annotations

import pytest
import torch

from tensordem.dem3d import DEM3DConfig
from tensordem.dem3d_energy import EnergyAuditedIceDEM3D

torch.set_num_threads(1)


def _sim(**kwargs) -> EnergyAuditedIceDEM3D:
    """Small two-layer block resting between two driven platens, no tool."""
    cfg = DEM3DConfig(nx=4, ny=4, nz=3, radius=0.025, top_platen_enabled=True,
                      bottom_platen_enabled=True, top_platen_velocity=-2.0e-3,
                      **kwargs)
    sim = EnergyAuditedIceDEM3D(cfg)
    return sim


def _warm(sim: EnergyAuditedIceDEM3D, steps: int = 20) -> None:
    for _ in range(steps):
        sim.step()


def test_platen_potential_matches_stored_energy():
    """The exact arm reuses the engine's stored platen energy (no second model)."""
    sim = _sim()
    _warm(sim)
    cfg = sim.config
    top_z = sim.top_platen_start_z + sim.time * cfg.top_platen_velocity
    bottom_z = sim.bottom_platen_start_z + sim.time * cfg.bottom_platen_velocity
    inline = 0.0
    top_ov = (sim.positions[:, 2] + cfg.radius - top_z).clamp_min(0.0)
    inline += 0.5 * cfg.platen_stiffness * (top_ov ** 2).sum().item()
    bot_ov = (bottom_z + cfg.radius - sim.positions[:, 2]).clamp_min(0.0)
    inline += 0.5 * cfg.platen_stiffness * (bot_ov ** 2).sum().item()
    helper = sim._platen_potential(sim.positions, top_z=top_z, bottom_z=bottom_z)
    assert helper == pytest.approx(inline, rel=1e-13, abs=1e-18)


def test_lateral_potential_ignored_when_lateral_platens_disabled():
    """Guard: a stale ``lateral_platen_pos`` must not leak energy.

    ``lateral_platen_pos`` is initialised on every instance but only driven when
    ``config.lateral_platens`` is set; counting it unconditionally adds a
    spurious 1/2 k dx^2 for every particle (measured 1500 J on the axial-only
    triaxial case, i.e. four orders of magnitude above the real scale).
    """
    sim = _sim()
    _warm(sim, 5)
    assert not sim.config.lateral_platens
    with_pos = sim._platen_potential(sim.positions, lateral_pos=sim.lateral_platen_pos)
    without = sim._platen_potential(sim.positions)
    assert with_pos == pytest.approx(without, rel=0.0, abs=1e-18)


def test_exact_work_equals_potential_increment_of_one_step():
    """One step: the exact arm must equal -(U(t+dt) - U(t)) at fixed positions."""
    sim = _sim()
    _warm(sim, 30)
    cfg = sim.config
    top_z0 = sim.top_platen_start_z + sim.time * cfg.top_platen_velocity
    bottom_z0 = sim.bottom_platen_start_z + sim.time * cfg.bottom_platen_velocity
    u0 = sim._platen_potential(sim.positions, top_z=top_z0, bottom_z=bottom_z0)
    top_z1 = top_z0 + sim.dt * cfg.top_platen_velocity
    bottom_z1 = bottom_z0 + sim.dt * cfg.bottom_platen_velocity
    u1 = sim._platen_potential(sim.positions, top_z=top_z1, bottom_z=bottom_z1)
    w_before = sim.cumulative_platen_work_exact_J
    sim.step()
    w_step = sim.cumulative_platen_work_exact_J - w_before
    assert w_step == pytest.approx(u1 - u0, rel=1e-12, abs=1e-20)


def test_legacy_platen_work_underestimates_by_half_k_dx2():
    """Quantify the legacy deficit and its first-order scaling in ``dt``.

    For a quadratic penalty the exact increment is ``-R*dx + 1/2 k n dx^2``
    (sign per the ledger convention), so the legacy left-endpoint rule is
    biased low by ``1/2 k n dx^2``: quadratic in ``dx`` for one step, hence
    first order in ``dt`` once accumulated.  Halving ``dt`` must quarter the
    one-step deficit.
    """
    def one_step_deficit(dt: float) -> tuple[float, float]:
        sim = _sim()
        sim.dt = dt
        _warm(sim, 30)
        legacy0 = sim.cumulative_platen_work_J
        exact0 = sim.cumulative_platen_work_exact_J
        sim.step()
        return (sim.cumulative_platen_work_exact_J - exact0,
                sim.cumulative_platen_work_J - legacy0)

    exact_a, legacy_a = one_step_deficit(4.0e-5)
    exact_b, legacy_b = one_step_deficit(2.0e-5)
    deficit_a = exact_a - legacy_a
    deficit_b = exact_b - legacy_b
    assert deficit_a > 0.0 and deficit_b > 0.0, "legacy must be biased low"
    # second-order in dt for a single step -> factor 4 on halving
    assert deficit_a / deficit_b == pytest.approx(4.0, rel=0.25)


def test_triaxial_summary_reports_both_ledger_arms():
    """The triaxial summary must expose the exact arm next to the legacy one."""
    from tensordem.dem3d_triaxial import TriaxialConfig, run_triaxial

    res = run_triaxial(TriaxialConfig(loading_speed=0.002, confine_steps=0,
                                      max_axial_steps=60, peak_min_stress=1e9))
    led = res["energy_ledger"]
    for key in ("input_work_J", "input_work_exact_J", "closure_residual_J",
                "closure_residual_exact_J", "closure_residual_relative_exact",
                "pass_exact", "quadrature"):
        assert key in led, key
    assert led["pass"] is False or led["pass"] is True  # boolean, not numpy
    # The two arms are close but not identical: the exact one carries the
    # additional 1/2 k dx^2 term of every step.
    assert led["input_work_exact_J"] != led["input_work_J"]
