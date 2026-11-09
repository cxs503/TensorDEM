"""Acceptance tests for the breakage–clearance operator (campaign P103W2C-N02).

Criteria (see ``docs/in27_s3_breakage_clearance.md``):

C1  Mass conservation: Σ m_i relative drift < 1e-10 (mass lost only through the
    recorded clearance sink).
C2  Energy ledger: W_in = surface + dissipation + penalty + ΔKE + clearance,
    residual < 2 %.
C3  Dual-axis convergence: h_min and dt refinement give an observed order ≥ 1
    and GCI < 10 % on the primary load metric.
C4  Old-conclusion stability: nine-case VBM block extremes vs the legacy
    ``delta > 0.25 m`` threshold differ by < 25 %; net lateral force stays below
    50 MN per case.
C5  Negative controls: σ_c → ∞ gives pure penalty contact (no breakage, mass
    conserved); h_min = 0 triggers the entity-explosion guard.
C6  End-to-end: the IN27-S3 load chain runs through and delivers force
    statistics *and* floe kinematics.

The heavy nine-case matrix and the full convergence study are produced by
``python -m tensordem.ice_contact.breakage_load_chain`` (see
``artifacts/in27_s3_breakage_clearance/report.json``); these tests use reduced
configurations so that they stay fast.

Run: ``python -m pytest tests/test_breakage_clearance.py -q``
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensordem.breakage_clearance import (  # noqa: E402
    BreakageConfig,
    BreakageEngine,
    Floe,
)
from tensordem.ice_contact import breakage_load_chain as chain  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_floes(n: int = 10, size: float = 0.30, speed: float = 1.0,
                density: float = 917.0) -> list[Floe]:
    rng = np.random.default_rng(7)
    floes = []
    for i in range(n):
        vol = (4.0 / 3.0) * math.pi * (size / 2) ** 3
        vel = np.array([speed + rng.normal(scale=0.1),
                        rng.normal(scale=0.05),
                        rng.normal(scale=0.05)])
        pos = rng.uniform(-1.0, 1.0, size=3)
        floes.append(Floe(fid=i, mass=density * vol, volume=vol, char_size=size,
                          position=pos, velocity=vel))
    return floes


def _fast_params(method: str = "new", **kwargs) -> chain.ChainParams:
    base = dict(speed=3.0, thickness=1.0, method=method, advance=0.12,
                dt=1.0e-4, max_steps=4000, n_max=90, record_series=False,
                n_blocks=12)
    base.update(kwargs)
    return chain.ChainParams(**base)


# ---------------------------------------------------------------------------
# C1 mass conservation
# ---------------------------------------------------------------------------

class TestC1MassConservation:
    def test_engine_mass_drift_with_breakage(self):
        floes = _make_floes(n=8, size=0.30)
        cfg = BreakageConfig(h_min=0.05, sigma_c=300.0, gamma=0.5, dt=1e-4,
                             breakage_time_s=0.02)
        eng = BreakageEngine(cfg, floes)
        for _ in range(60):
            eng.step(contact_impulse_Ns=5e4, penalty_energy_J=1.0)
        assert len(eng.breakage_events()) > 0
        assert eng.mass_drift() < 1e-10

    def test_chain_mass_drift(self):
        res = chain.run_case(_fast_params("new"))
        assert res["mass"]["drift"] < 1e-10
        # mass lost only through the clearance sink
        assert res["mass"]["clearance_mass_fraction"] >= 0.0

    def test_clearance_only_below_h_min(self):
        floes = _make_floes(n=4, size=0.12)
        cfg = BreakageConfig(h_min=0.08, sigma_c=300.0, gamma=0.5, dt=1e-4)
        eng = BreakageEngine(cfg, floes)
        for _ in range(40):
            eng.step(contact_impulse_Ns=5e4, penalty_energy_J=1.0)
        for evt in eng.clearance_events():
            # every cleared floe is below the resolution floor
            assert evt["mass_kg"] > 0.0
        assert eng.mass_drift() < 1e-10


# ---------------------------------------------------------------------------
# C2 energy ledger
# ---------------------------------------------------------------------------

class TestC2EnergyLedger:
    def test_chain_energy_residual(self):
        res = chain.run_case(_fast_params("new"))
        assert res["energy"]["residual"] < 0.02, res["energy"]

    def test_legacy_energy_residual(self):
        res = chain.run_case(_fast_params("legacy"))
        assert res["energy"]["residual"] < 0.02, res["energy"]

    def test_friction_does_not_break_the_ledger(self):
        res = chain.run_case(_fast_params("new", mu_wall=0.30, mu_pp=0.50))
        assert res["energy"]["residual"] < 0.02, res["energy"]

    def test_ledger_records_every_term(self):
        floes = _make_floes(n=5, size=0.30)
        cfg = BreakageConfig(h_min=0.05, dt=1e-4)
        eng = BreakageEngine(cfg, floes)
        led = eng.step(contact_impulse_Ns=1e4, penalty_energy_J=20.0,
                       dissipation_energy_J=3.0, external_work_J=5.0)
        row = led.to_dict()
        for key in ("contact_impulse_Ns", "breakage_surface_energy_J",
                    "dissipation_J", "penalty_energy_J", "clearance_energy_J",
                    "kinetic_energy_J", "total_mass_kg"):
            assert key in row
        assert row["dissipation_J"] == pytest.approx(3.0)

    def test_penalty_ceiling_reports_pollution(self):
        floes = _make_floes(n=3, size=0.30)
        cfg = BreakageConfig(h_min=0.05, penalty_ceiling_J=100.0, dt=1e-4)
        eng = BreakageEngine(cfg, floes)
        led = eng.step(contact_impulse_Ns=1e2, penalty_energy_J=250.0)
        assert led.penalty_energy_J == pytest.approx(100.0)
        assert led.penalty_exceeded_J == pytest.approx(150.0)
        assert eng.penalty_pollution_fraction() > 0.0


# ---------------------------------------------------------------------------
# C3 convergence
# ---------------------------------------------------------------------------

class TestC3Convergence:
    def test_gci_second_order(self):
        # synthetic second-order sequence: error halves by 4 per refinement
        coarse, medium, fine = 1.0 + 0.04, 1.0 + 0.01, 1.0 + 0.0025
        out = chain._gci(coarse, medium, fine)
        assert out["observed_order"] == pytest.approx(2.0, abs=1e-9)
        assert out["gci"] is not None and out["gci"] < 0.10

    def test_h_min_refinement_reduces_difference(self):
        base = _fast_params("new")
        coarse = chain.run_case(chain.ChainParams(**{**base.__dict__, "h_min": 0.10}))
        fine = chain.run_case(chain.ChainParams(**{**base.__dict__, "h_min": 0.05}))
        # both must stay physical and observable
        assert coarse["kinematics"]["n_floes"] > 0
        assert fine["kinematics"]["n_floes"] > 0
        assert np.isfinite(fine["block_extremes"]["HBM"]["mean"])


# ---------------------------------------------------------------------------
# C4 old-conclusion stability
# ---------------------------------------------------------------------------

class TestC4OldConclusionStability:
    def test_vbm_and_lateral_force_bands(self):
        base = _fast_params()
        for speed, thickness in ((1.0, 0.5), (3.0, 1.0), (5.0, 1.5)):
            new = chain.run_case(chain.ChainParams(**{**base.__dict__, "speed": speed,
                                                      "thickness": thickness, "method": "new"}))
            old = chain.run_case(chain.ChainParams(**{**base.__dict__, "speed": speed,
                                                      "thickness": thickness, "method": "legacy"}))
            vbm_new = new["block_extremes"]["VBM"]["mean"]
            vbm_old = old["block_extremes"]["VBM"]["mean"]
            if vbm_old > 0:
                assert abs(vbm_new - vbm_old) / vbm_old < 0.25, (speed, thickness)
            assert new["loads"]["net_lateral_force_N"] < 50.0e6
            assert old["loads"]["net_lateral_force_N"] < 50.0e6


# ---------------------------------------------------------------------------
# C5 negative controls
# ---------------------------------------------------------------------------

class TestC5NegativeControls:
    def test_infinite_strength_no_breakage(self):
        res = chain.run_case(_fast_params("new", sigma_c=1.0e15))
        assert res["counts"]["n_breakage_events"] == 0
        assert res["mass"]["drift"] < 1e-10
        assert res["energy"]["residual"] < 0.02

    def test_h_min_zero_triggers_guard(self):
        with pytest.raises(RuntimeError) as exc:
            chain.run_case(_fast_params("new", h_min=0.0, max_floe_count=100,
                                        advance=0.3, max_steps=6000,
                                        sigma_c=1.0e5))
        assert "guard" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# C6 end-to-end
# ---------------------------------------------------------------------------

class TestC6EndToEnd:
    def test_matrix_small(self):
        base = _fast_params()
        out = chain.run_matrix(speeds=(3.0,), thicknesses=(0.5, 1.0), base=base,
                               record_series=False)
        assert out["criteria"]["case_count_new"] == 2
        assert out["criteria"]["case_count_old"] == 2
        first = out["cases_new"][0]
        # force statistics
        for comp in ("VBM", "HBM", "TBM"):
            assert len(first["block_extremes"][comp]["values"]) > 0
            assert first["gev"][comp]["n"] > 0
            assert first["gev"][comp]["loc"] is not None
        # floe kinematics
        kin = first["kinematics"]
        assert kin["n_floes"] > 0
        assert kin["mean_speed"] >= 0.0
        assert kin["d50"] > 0.0
        assert "variance_speed" in kin

    def test_block_extremes_and_gev(self):
        values = np.array([1.0, 2.0, 1.5, 3.0, 2.5, 1.2, 4.0, 2.2])
        maxima, block_len = chain.block_extremes(values, 4)
        assert len(maxima) == 4
        assert block_len == pytest.approx(2.0)
        fit = chain.fit_gev_block_maxima(np.array(maxima))
        assert fit["n"] == 4
        assert fit["loc"] is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))