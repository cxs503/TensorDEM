"""Tests for the 3D level-ice sheet with mass-conserving lateral escape.

Verifies:
(a) Mass conservation: relative error ≤ 1e-6 per frame
(b) Lateral flux closure rate is finite and reasonable
(c) Negative control: disabling reservoir changes resistance
(d) Hu & Zhou MT Uikku case loading and single-case execution
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from tensordem.level_ice.sheet import SheetConfig
from tensordem.level_ice.sheet3d import (
    LevelIceSheet3D,
    MassLedger,
    LateralFluxAccount,
    Sheet3DConfig,
    map_3d_to_ship,
    load_uikku_cases,
    run_uikku_case,
)
from tensordem.level_ice.formula import HullForm


# ---------------------------------------------------------------------------
# Unit tests: MassLedger
# ---------------------------------------------------------------------------
class TestMassLedger:
    def test_initial_state(self):
        ledger = MassLedger()
        assert ledger.total_kg == 0.0
        assert ledger.relative_error == 0.0

    def test_conservation_with_transfers(self):
        ledger = MassLedger()
        ledger.mass_intact_kg = 1000.0
        ledger.mass_initial_kg = 1000.0
        ledger.mass_intact_kg -= 100.0
        ledger.mass_reservoir_kg += 100.0
        assert ledger.relative_error < 1e-15
        assert ledger.total_kg == pytest.approx(1000.0)

    def test_snapshot_records_history(self):
        ledger = MassLedger()
        ledger.mass_intact_kg = 500.0
        ledger.mass_initial_kg = 500.0
        ledger.snapshot(0.0)
        ledger.mass_intact_kg = 400.0
        ledger.mass_reservoir_kg = 100.0
        ledger.snapshot(1.0)
        assert len(ledger.history["time_s"]) == 2
        assert ledger.history["total_kg"][-1] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Unit tests: LateralFluxAccount
# ---------------------------------------------------------------------------
class TestLateralFluxAccount:
    def test_no_outflow_closure_is_one(self):
        flux = LateralFluxAccount()
        assert flux.closure_rate == 1.0

    def test_with_outflow(self):
        flux = LateralFluxAccount()
        flux.outflow_kg = 100.0
        flux.return_impulse_Ns = 100.0
        assert flux.closure_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration tests: LevelIceSheet3D
# ---------------------------------------------------------------------------
def _fast_sheet_config(thickness_m=1.0, layers=4, speed=1.0):
    """Create a small, fast sheet config for testing."""
    return SheetConfig(
        thickness_m=thickness_m,
        particle_size_m=thickness_m / layers,
        length_m=10.0,
        flexural_strength_pa=0.5e6,
        compressive_strength_pa=1.5e6,
        shear_strength_pa=1.0e6,
        buoyancy=True,
        bow_angle_deg=30.0,
        hull_speed_m_s=speed,
        hull_friction=0.05,
        crush_cap_pa=1.5e6,
        mass_scale=200.0,
    )


class TestLevelIceSheet3D:
    """Integration tests for the 3D sheet simulation."""

    def test_mass_conservation_with_reservoir(self):
        """(a) Mass conservation: relative error ≤ 1e-6 per frame."""
        sheet_cfg = _fast_sheet_config()
        cfg = Sheet3DConfig(
            sheet_config=sheet_cfg,
            channel_width_m=20.0,
            beam_m=20.0,
            n_lateral_layers=2,
            enable_reservoir=True,
        )
        sim = LevelIceSheet3D(cfg)
        result = sim.run(distance_m=1.5, record_stride=50)

        ledger = result["mass_ledger"]
        max_error = ledger["max_rel_error"]
        assert max_error <= 1e-6, (
            f"Mass conservation error {max_error:.2e} exceeds 1e-6"
        )
        # Also check total mass equals initial mass
        assert ledger["mass_total_kg"] == pytest.approx(
            ledger["mass_initial_kg"], rel=1e-10
        )

    def test_mass_not_conserved_without_reservoir(self):
        """Negative control: without reservoir, mass is NOT conserved."""
        sheet_cfg = _fast_sheet_config()
        cfg_no_res = Sheet3DConfig(
            sheet_config=sheet_cfg,
            channel_width_m=20.0,
            beam_m=20.0,
            enable_reservoir=False,
        )
        sim = LevelIceSheet3D(cfg_no_res)
        result = sim.run(distance_m=1.5, record_stride=50)

        ledger = result["mass_ledger"]
        # Without reservoir, particles are deleted → mass decreases
        assert ledger["mass_total_kg"] < ledger["mass_initial_kg"]

    def test_lateral_flux_finite(self):
        """(b) Lateral flux closure rate should be finite."""
        sheet_cfg = _fast_sheet_config()
        cfg = Sheet3DConfig(
            sheet_config=sheet_cfg,
            channel_width_m=20.0,
            beam_m=20.0,
            n_lateral_layers=2,
            enable_reservoir=True,
        )
        sim = LevelIceSheet3D(cfg)
        result = sim.run(distance_m=1.5, record_stride=50)

        flux = result["lateral_flux"]
        assert math.isfinite(flux["closure_rate"])
        assert flux["n_escape_events"] >= 0

    def test_reservoir_produces_back_pressure(self):
        """Reservoir should produce non-zero back-pressure when mass escapes."""
        sheet_cfg = _fast_sheet_config()
        cfg = Sheet3DConfig(
            sheet_config=sheet_cfg,
            channel_width_m=20.0,
            beam_m=20.0,
            enable_reservoir=True,
        )
        sim = LevelIceSheet3D(cfg)
        result = sim.run(distance_m=1.5, record_stride=50)

        # If any mass escaped, there should be back-pressure
        if result["mass_ledger"]["mass_reservoir_kg"] > 0:
            assert result["lateral_flux"]["back_pressure_N"] > 0


# ---------------------------------------------------------------------------
# Hu & Zhou MT Uikku cases
# ---------------------------------------------------------------------------
class TestUikkuCases:
    def test_load_uikku_cases(self):
        cases = load_uikku_cases()
        assert len(cases) == 10
        assert all("test_id" in c for c in cases)
        assert all("ice_thickness_m" in c for c in cases)
        case_103 = next(c for c in cases if c["test_id"] == "103")
        assert case_103["ice_thickness_m"] == pytest.approx(0.77)
        assert case_103["drift_speed_m_s"] == pytest.approx(0.2)

    def test_single_uikku_case_runs(self):
        """Run one MT Uikku case and verify mass conservation."""
        cases = load_uikku_cases()
        case = cases[0]  # Test 103
        result = run_uikku_case(case, wb_ratio=1.5, layers=4, advance_m=1.5)

        assert result["test_id"] == "103"
        assert result["predicted_kN"] > 0
        assert result["mass_ledger"]["max_rel_error"] <= 1e-6
        assert result["wb_ratio"] == 1.5
