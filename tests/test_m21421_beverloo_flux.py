"""Tests for the m21421 orifice-flux meter and the wall-proximity scan matrix.

Two things are asserted:

1. ``SiloCase.measure_orifice_flux`` is a *bookkeeping-only* instrument: with the
   flag off the discharge result is unchanged and the flux block reports
   ``measured: False``; with the flag on the two independent estimates of the
   steady mass flow rate (aperture-plane crossings vs. particles that ended
   below the plate) agree, and mass is conserved (crossings >= discharges).
2. The pre-registered scan of ``benchmarks/m21421_beverloo_wall/prereg.md`` is
   internally consistent: one particle size, one D/d, H/D = 4.0 in every case,
   the same random draw, and the silo_D/D column of the table.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "benchmarks" / "m21421_beverloo_wall"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tensordem.silo_discharge import SiloCase, simulate_silo  # noqa: E402

TINY = dict(
    radius=0.005, silo_radius=0.030, fill_height=0.040,
    orifice_diameter=0.030, plate_thickness=0.005, kn=5.0e5,
    mu=0.30, packing="settled", plug_during_settle=True,
    settle_loose_gap=0.30, jitter=0.05, seed=21421,
    settle_time=0.15, discharge_time=0.25, window=0.04, n_windows=5,
)


@pytest.fixture(scope="module")
def flux_run() -> dict:
    res = simulate_silo(SiloCase(**TINY, measure_orifice_flux=True))
    res.pop("positions", None)
    return res


@pytest.fixture(scope="module")
def plain_run() -> dict:
    return simulate_silo(SiloCase(**TINY, measure_orifice_flux=False))


def test_flux_flag_off_reports_not_measured(plain_run):
    assert plain_run["orifice_flux"] == {"measured": False}
    assert plain_run["steady_flow_rate_kg_s"] > 0.0


def test_flux_meter_is_bookkeeping_only(flux_run, plain_run):
    """The meter must not perturb the trajectory: same discharges, same rate."""
    assert flux_run["n_initial"] == plain_run["n_initial"]
    assert flux_run["n_discharged_total"] == plain_run["n_discharged_total"]
    assert flux_run["steady_flow_rate_kg_s"] == pytest.approx(
        plain_run["steady_flow_rate_kg_s"], rel=1e-12, abs=0.0)


def test_mass_conservation_crossings_ge_discharges(flux_run):
    fx = flux_run["orifice_flux"]
    assert fx["measured"] is True
    assert fx["n_crossed"] >= flux_run["n_discharged_total"]


def test_two_estimators_agree(flux_run):
    fx = flux_run["orifice_flux"]
    W_count = flux_run["steady_flow_rate_kg_s"]
    W_flux = fx["steady_flux_kg_s"]
    assert math.isfinite(W_flux) and W_flux > 0.0
    assert abs(W_flux - W_count) / W_count < 0.10, (W_flux, W_count)


def test_flux_windows_shape(flux_run):
    fx = flux_run["orifice_flux"]
    assert len(fx["steady_flux_windows_kg_s"]) == 5
    assert fx["steady_flux_interval_s"][1] > fx["steady_flux_interval_s"][0]


# ---------------------------------------------------------------------------
# pre-registration consistency of the scan matrix
# ---------------------------------------------------------------------------
def test_scan_matrix_matches_preregistration():
    import run_case as scan

    assert scan.ORDER == ["S26", "S35", "S40", "F26"]
    seeds = {scan.BASE["seed"]}
    expected_silo_over_D = {"S26": 2.6, "S35": 3.5, "S40": 4.0, "F26": 2.6}
    for cid in scan.ORDER:
        case = scan.build_case(cid)
        assert case.diameter == pytest.approx(2.0 * scan.BASE["radius"])
        assert case.orifice_diameter / case.diameter == pytest.approx(5.0)
        assert case.fill_height / case.orifice_diameter == pytest.approx(4.0, abs=0.01)
        assert 2.0 * case.silo_radius / case.orifice_diameter == pytest.approx(
            expected_silo_over_D[cid], abs=0.02)
        assert case.measure_orifice_flux is True
        seeds.add(case.seed)
    assert seeds == {scan.BASE["seed"]}, "scan must be paired: one seed only"
    assert scan.build_case("F26").plug_during_settle is False
    for cid in ("S26", "S35", "S40"):
        assert scan.build_case(cid).plug_during_settle is True


def test_scan_cases_are_valid_silo_cases():
    """Every pre-registered geometry must pass SiloCase validation."""
    import run_case as scan

    for cid in scan.ORDER:
        case = scan.build_case(cid)
        assert case.orifice_diameter < 2.0 * case.silo_radius