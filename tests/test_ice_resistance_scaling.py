"""Tests for :mod:`tensordem.ice_resistance_scaling` (P121W5-N04).

Covers:
- UnitCase geometry and derived properties
- Design cases (full factorial)
- Scale law fit structure
- Ship case and engineering anchors
- Energy budget closure
- FSD transfer
- Criteria evaluation structure
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def module():
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from tensordem import ice_resistance_scaling as m
    return m


class TestUnitCase:
    """UnitCase dataclass and derived geometry."""

    def test_default_construction(self, module):
        case = module.UnitCase()
        assert case.beam_m == 0.4
        assert case.thickness_m == 0.5
        assert case.speed_m_s == 1.0
        assert case.concentration == 0.7

    def test_floe_radius(self, module):
        case = module.UnitCase(beam_m=0.8, floe_radius_ratio=0.35, mesh_refinement=1.0)
        assert abs(case.floe_radius_m - 0.28) < 1e-12

    def test_hull_length(self, module):
        case = module.UnitCase(beam_m=1.6, hull_length_ratio=4.0)
        assert abs(case.hull_length_m - 6.4) < 1e-12

    def test_domain_dimensions(self, module):
        case = module.UnitCase(beam_m=0.4, domain_length_ratio=14.0,
                               domain_width_ratio=6.0)
        length, width = case.domain_m
        assert abs(length - 5.6) < 1e-12
        assert abs(width - 2.4) < 1e-12

    def test_auto_dt_positive(self, module):
        case = module.UnitCase()
        dt = case.auto_dt()
        assert dt > 0
        assert math.isfinite(dt)

    def test_contact_stiffness(self, module):
        case = module.UnitCase(strength_pa=1.5e6, thickness_m=0.5)
        k = case.contact_stiffness_n_m()
        assert k > 0
        # k_n = 2 * sigma_c * h / sqrt(frac) with frac=0.01
        expected = 2.0 * 1.5e6 * 0.5 / math.sqrt(0.01)
        assert abs(k - expected) / expected < 1e-10

    def test_transit_time(self, module):
        case = module.UnitCase(beam_m=0.4, speed_m_s=1.0)
        t = case.transit_time_s
        assert t > 0
        assert math.isfinite(t)

    def test_steady_window(self, module):
        case = module.UnitCase()
        t0, t1 = case.steady_window_s
        assert t0 < t1
        assert t0 >= 0

    def test_label(self, module):
        case = module.UnitCase(beam_m=0.4, thickness_m=0.5,
                               speed_m_s=1.0, concentration=0.7)
        assert case.label() == "B0.4_h0.5_v1_c0.7"

    def test_invalid_concentration(self, module):
        with pytest.raises(ValueError):
            module.UnitCase(concentration=1.5)

    def test_invalid_beam(self, module):
        with pytest.raises(ValueError):
            module.UnitCase(beam_m=-1.0)


class TestDesignCases:
    """Full factorial design."""

    def test_108_cases(self, module):
        cases = module.design_cases()
        assert len(cases) == 108

    def test_beam_levels(self, module):
        cases = module.design_cases()
        beams = sorted({c.beam_m for c in cases})
        assert beams == [0.2, 0.4, 0.8, 1.6]

    def test_thickness_levels(self, module):
        cases = module.design_cases()
        thicknesses = sorted({c.thickness_m for c in cases})
        assert thicknesses == [0.3, 0.5, 0.8]

    def test_speed_levels(self, module):
        cases = module.design_cases()
        speeds = sorted({c.speed_m_s for c in cases})
        assert speeds == [0.5, 1.0, 2.0]

    def test_concentration_levels(self, module):
        cases = module.design_cases()
        concentrations = sorted({c.concentration for c in cases})
        assert concentrations == [0.6, 0.7, 0.8]

    def test_geometric_similarity(self, module):
        """All cases maintain geometric similarity (L/B, R_floe/B constant)."""
        cases = module.design_cases()
        ratios = {(c.hull_length_m / c.beam_m,
                   c.floe_radius_m / c.beam_m) for c in cases}
        assert len(ratios) == 1  # all same ratio


class TestShipCase:
    """ShipCase dataclass."""

    def test_default_ship(self, module):
        ship = module.ShipCase()
        assert ship.length_m == 150.0
        assert ship.beam_m == 22.0
        assert ship.speed_m_s == 5.0

    def test_floe_diameter(self, module):
        ship = module.ShipCase(floe_diameter_m=10.0)
        assert ship.floe_diameter_m == 10.0


class TestEnergyBudget:
    """Energy budget closure."""

    def test_closure_ratio(self, module):
        budget = module.energy_budget(
            w_ext=1000.0, w_ff_normal=-400.0, w_ff_tangent=-300.0,
            w_drag=-200.0, kinetic_energy=100.0, scheme_error=0.0)
        # W_ext + W_drag + W_ff = dKE
        # 1000 + (-200) + (-400) + (-300) = 100 = dKE
        assert abs(budget["closure_ratio"]) < 1e-10

    def test_three_term_ratio(self, module):
        budget = module.energy_budget(
            w_ext=1000.0, w_ff_normal=-400.0, w_ff_tangent=-300.0,
            w_drag=-200.0, kinetic_energy=-100.0, scheme_error=0.0)
        # E_break + E_push + E_friction = 400 + 200 + 300 = 900
        # W_ext = 1000
        # three_term_ratio = |1000 - 900| / 1000 = 0.1
        assert abs(budget["three_term_ratio"] - 0.1) < 1e-10


class TestBrokenIceAnchor:
    """Broken-ice engineering anchor."""

    def test_anchor_structure(self, module):
        ship = module.ShipCase()
        anchor = module.broken_ice_anchor(ship)
        assert "crushing_N" in anchor
        assert "bending_N" in anchor
        assert "submersion_N" in anchor
        assert "momentum_N" in anchor
        assert "total_N" in anchor
        assert anchor["total_N"] > 0

    def test_ledger_sha256(self, module):
        h1 = module.broken_ice_ledger_sha256()
        h2 = module.broken_ice_ledger_sha256()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex


class TestBendingLength:
    """Characteristic bending length."""

    def test_positive(self, module):
        ship = module.ShipCase()
        lam = module.bending_length_m(ship)
        assert lam > 0

    def test_thickness_scaling(self, module):
        ship1 = module.ShipCase(thickness_m=0.5)
        ship2 = module.ShipCase(thickness_m=1.0)
        lam1 = module.bending_length_m(ship1)
        lam2 = module.bending_length_m(ship2)
        # lambda ~ h * (E h^3 / ...)^(1/4) ~ h^(1+3/4) = h^(7/4)
        ratio = lam2 / lam1
        expected = (1.0 / 0.5) ** 1.75
        assert abs(ratio - expected) / expected < 0.01


class TestFSDTransfer:
    """Floe size distribution transfer."""

    def test_power_law_fsd(self, module):
        diameters = module.power_law_fsd(d_min_m=1.0, d_max_m=10.0,
                                         exponent=2.0, n=100, seed=42)
        assert len(diameters) == 100
        assert float(diameters.min()) >= 1.0
        assert float(diameters.max()) <= 10.0

    def test_fsd_transfer_structure(self, module):
        ship = module.ShipCase(floe_diameter_m=10.0, length_m=150.0)
        model_diameters = [2.0, 3.0, 4.0, 5.0, 6.0]
        result = module.fsd_transfer(model_diameters=model_diameters,
                                     ship=ship, length_scale=10.0,
                                     exponent=2.0)
        assert "n_floes" in result
        assert "hypotheses" in result
        assert "sqrt_length" in result["hypotheses"]
        assert "geometric" in result["hypotheses"]
        assert "fitted" in result["hypotheses"]


class TestScaleToShip:
    """Scale law application to full ship."""

    def test_scale_structure(self, module):
        fit = {
            "exponents": {
                "beam": {"value": 1.0},
                "speed": {"value": 2.0},
                "thickness": {"value": 1.0},
                "concentration": {"value": 1.0},
            }
        }
        unit_record = {
            "case": {"beam_m": 0.4, "speed_m_s": 1.0,
                     "thickness_m": 0.5, "concentration": 0.7,
                     "hull_length_m": 1.6},
            "R_mean_N": 1000.0,
        }
        ship = module.ShipCase(beam_m=22.0, speed_m_s=5.0,
                               thickness_m=0.5, concentration=0.7,
                               length_m=150.0)
        result = module.scale_to_ship(fit=fit, unit_record=unit_record,
                                      ship=ship)
        assert "R_scale_N" in result
        assert result["R_scale_N"] > 0
        # With α=1, β=2, γ=1, δ=1:
        # R_ship = 1000 * (22/0.4)^1 * (5/1)^2 * (0.5/0.5)^1 * (0.7/0.7)^1
        #        = 1000 * 55 * 25 * 1 * 1 = 1,375,000
        assert abs(result["R_scale_N"] - 1375000.0) / 1375000.0 < 1e-10


class TestArtifactIntegrity:
    """Verify the produced artifact on disk is self-consistent."""

    @pytest.fixture
    def artifact(self):
        path = REPO / "artifacts" / "ice_resistance_scaling" / "ice_resistance_scaling.json"
        if not path.exists():
            pytest.skip("artifact not yet generated")
        return json.loads(path.read_text())

    def test_has_criteria(self, artifact):
        assert "criteria" in artifact

    def test_has_fit(self, artifact):
        assert "fit" in artifact
        assert "exponents" in artifact["fit"]
        assert "r2_log" in artifact["fit"]

    def test_has_sweep(self, artifact):
        assert "sweep" in artifact
        assert len(artifact["sweep"]) >= 100

    def test_has_figures(self, artifact):
        assert "figures" in artifact
        assert len(artifact["figures"]) >= 4

    def test_r2_above_threshold(self, artifact):
        assert artifact["fit"]["r2_log"] > 0.95

    def test_energy_closure(self, artifact):
        assert artifact["criteria"]["d_energy_budget"]["pass"] is True

    def test_negative_controls(self, artifact):
        assert artifact["criteria"]["f_negative_controls"]["pass"] is True
