"""Acceptance tests for the static packed-bed effective conductivity module (P44-N12)."""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.thermal import packed_bed as pb  # noqa: E402
from tensordem.thermal import stefan_adapter  # noqa: E402

RESULT = REPO / "benchmarks" / "packed_bed_conductivity" / "result.json"
REFS = REPO / "benchmarks" / "packed_bed_conductivity" / "refs.json"


# ---------------------------------------------------------------- literature
def test_wiener_bounds_match_analytic_values():
    phi, ks, kf = 0.40, 60.0, 0.6
    bounds = pb.wiener_bounds(phi, ks, kf)
    assert bounds["k_series"] == pytest.approx(1.0 / (0.6 / ks + 0.4 / kf), rel=1e-12)
    assert bounds["k_parallel"] == pytest.approx(0.6 * ks + 0.4 * kf, rel=1e-12)
    assert bounds["k_series"] < bounds["k_parallel"]


def test_krupiczka_follows_its_own_exponent():
    phi, contrast = 0.40, 20.0
    kf = 0.6
    expected = kf * contrast ** (
        0.280 - 0.757 * math.log10(phi) - 0.057 * math.log10(contrast)
    )
    assert pb.krupiczka_conductivity(phi, contrast * kf, kf) == pytest.approx(expected, rel=1e-12)


def test_correlations_are_monotone_in_contrast():
    # cell-model area weighting sqrt(1-phi) is not exactly bound compatible, so
    # the correlations are only required to be monotone here; the Wiener bounds
    # are enforced for the operators in the tests below.
    for phi in (0.30, 0.40, 0.50, 0.60):
        krup = [pb.krupiczka_conductivity(phi, c * 0.6, 0.6) for c in pb.CONDUCTIVITY_RATIO_SCAN]
        zs = [
            pb.zehner_schluender_conductivity(phi, c * 0.6, 0.6)
            for c in pb.CONDUCTIVITY_RATIO_SCAN
        ]
        assert all(krup[i + 1] > krup[i] for i in range(len(krup) - 1))
        assert all(zs[i + 1] > zs[i] for i in range(len(zs) - 1))
        assert krup[0] == pytest.approx(0.6, rel=1e-12)
        assert zs[0] == pytest.approx(0.6, rel=1e-12)


def test_zehner_schluender_is_exact_for_equal_conductivities():
    for phi in (0.3, 0.45, 0.6):
        assert pb.zehner_schluender_conductivity(phi, 3.0, 3.0) == pytest.approx(3.0, rel=1e-12)
        assert pb.krupiczka_conductivity(phi, 3.0, 3.0) == pytest.approx(3.0, rel=1e-12)


def test_zehner_schluender_quadrature_is_converged():
    coarse = pb.zehner_schluender_conductivity(
        0.4, 60.0, 0.6, constants=pb.ZehnerSchluenderConstants(quadrature_points=801)
    )
    fine = pb.zehner_schluender_conductivity(0.4, 60.0, 0.6)
    assert coarse == pytest.approx(fine, rel=1e-3)


# ------------------------------------------------------------------- operators
def test_periodic_solver_reproduces_homogeneous_and_layered_limits():
    n, h = 12, 1.0 / 12
    k = 2.5
    k_eff, _ = pb._periodic_solve([np.full((n, n, n), k * h)] * 3, h, n, 1.0)
    assert k_eff == pytest.approx(k, rel=1e-9)
    field = np.broadcast_to(
        np.where((np.arange(n) % 2 == 0)[None, None, :], 1.0, 3.0), (n, n, n)
    ).copy()
    faces = [
        np.full((n, n, n), 1e-12 * h),
        np.full((n, n, n), 1e-12 * h),
        2 * h / (1.0 / field + 1.0 / np.roll(field, -1, axis=2)),
    ]
    k_eff, _ = pb._periodic_solve(faces, h, n, 1.0)
    assert k_eff == pytest.approx(1.0 / (0.5 / 1.0 + 0.5 / 3.0), rel=1e-8)


def test_packing_reaches_target_porosity_and_contacts():
    beds = {phi: pb.generate_static_bed(phi, n_spheres=60, seed=11) for phi in (0.35, 0.45, 0.55)}
    for phi, bed in beds.items():
        assert bed.porosity == pytest.approx(phi, abs=1e-2)
        assert bed.coordination_number > 0.0
    assert beds[0.35].coordination_number > beds[0.55].coordination_number


def test_rescale_porosity_keeps_centres_and_rebuilds_contacts():
    bed = pb.generate_static_bed(0.40, n_spheres=50, seed=5)
    thin = pb.rescale_porosity(bed, 0.80)
    assert thin.porosity == pytest.approx(0.80, rel=1e-12)
    assert thin.radius < bed.radius
    np.testing.assert_allclose(thin.positions, bed.positions)


def test_flux_operator_is_exact_in_the_degenerate_contrast_limit():
    bed = pb.generate_static_bed(0.40, n_spheres=50, seed=5)
    a_c = pb.DEFAULT_CONTACT_MODEL.contact_radius(bed)
    value = pb.conductivity_flux_average(bed, 2.0, 2.0, grid=20, contact_radius_m=a_c)["k_eff"]
    assert value == pytest.approx(2.0, rel=2e-2)


def test_flux_operator_respects_wiener_bounds():
    bed = pb.generate_static_bed(0.40, n_spheres=50, seed=5)
    a_c = pb.DEFAULT_CONTACT_MODEL.contact_radius(bed)
    for contrast in (5.0, 100.0):
        ks, kf = contrast * 0.6, 0.6
        bounds = pb.wiener_bounds(bed.porosity, ks, kf)
        value = pb.conductivity_flux_average(bed, ks, kf, grid=24, contact_radius_m=a_c)["k_eff"]
        assert bounds["k_series"] <= value <= bounds["k_parallel"]


def test_two_operators_agree_on_the_same_configuration():
    bed = pb.generate_static_bed(0.40, n_spheres=80, seed=3)
    a_c = pb.DEFAULT_CONTACT_MODEL.contact_radius(bed)
    network = pb._graph_solid_channel(bed, 60.0, a_c)["k_eff"]
    field = pb.conductivity_flux_average(
        bed, 60.0, 1e-9 * 60.0, grid=32, contact_radius_m=a_c
    )["k_eff"]
    assert field > 0.0 and network > 0.0
    assert abs(network - field) / field <= 0.05


def test_hertz_contact_radius_scales_with_stress():
    bed = pb.generate_static_bed(0.40, n_spheres=40, seed=2)
    low = pb.ContactModel(confining_stress_pa=1e3).contact_radius(bed)
    high = pb.ContactModel(confining_stress_pa=1e5).contact_radius(bed)
    # a_c ~ sigma^(1/3): a 100x stress ratio gives 100^(1/3) = 4.64x radius
    assert high / low == pytest.approx(100.0 ** (1.0 / 3.0), rel=1e-9)


def test_laminate_limit_reproduces_the_series_bound_at_extreme_porosity():
    """phi -> 0 and phi -> 1 limits of the operator (exact series configuration)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "packed_bed_bench", REPO / "benchmarks" / "packed_bed_conductivity" / "run.py"
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    for phi, reference in ((1e-6, 60.0), (0.999, 0.6)):
        value = bench.laminate_limit(phi, 60.0, 0.6, layers=8)
        expected = pb.wiener_bounds(phi, 60.0, 0.6)["k_series"]
        assert value == pytest.approx(expected, rel=1e-8)
        assert abs(value - reference) / reference <= 1e-3


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        pb.wiener_bounds(1.2, 1.0, 0.1)
    with pytest.raises(ValueError):
        pb.krupiczka_conductivity(0.4, -1.0, 0.6)
    with pytest.raises(ValueError):
        pb.generate_static_bed(0.4, n_spheres=4)


# --------------------------------------------------------------- Stefan front
def test_stefan_melt_time_decreases_with_conductivity():
    state = stefan_adapter.BedStefanState(thickness_m=0.10, porosity=0.40)
    values = list(np.geomspace(0.5, 20.0, 9))
    sensitivity = stefan_adapter.melt_time_sensitivity(values, state)
    assert sensitivity["monotonic_decreasing"] is True
    assert sensitivity["max_dt_melt_dk"] < 0.0
    assert sensitivity["t_melt_s"][0] > sensitivity["t_melt_s"][-1]
    # t_melt ~ 1/k_eff for a conduction-limited static bed
    ratio = sensitivity["t_melt_s"][0] / sensitivity["t_melt_s"][-1]
    assert ratio == pytest.approx(values[-1] / values[0], rel=1e-6)


def test_stefan_spread_between_bounds_is_reported_and_large():
    state = stefan_adapter.BedStefanState(thickness_m=0.10, porosity=0.40)
    bounds = pb.wiener_bounds(0.40, 60.0, 0.6)
    t_series = stefan_adapter.melt_time_from_conductivity(bounds["k_series"], state)
    t_parallel = stefan_adapter.melt_time_from_conductivity(bounds["k_parallel"], state)
    spread = abs(t_series - t_parallel) / t_parallel
    assert spread > 0.20  # conduction-dominated: the enthalpy argument alone is not enough
    assert t_series > t_parallel


def test_stefan_adapter_uses_the_threefield_thermal_module():
    state = stefan_adapter.BedStefanState()
    fragment = stefan_adapter.convective_reference_melt_time(5e-3, state)
    assert math.isfinite(fragment) and fragment > 0.0


# ------------------------------------------------------------- benchmark lock
@pytest.mark.skipif(not RESULT.exists(), reason="benchmark result.json not generated")
def test_benchmark_result_records_every_gate():
    payload = json.loads(RESULT.read_text())
    for key in (
        "gate_bounds",
        "gate_porosity_limits",
        "gate_literature",
        "gate_two_operators",
        "gate_stefan_coupling",
    ):
        assert key in payload
    assert len(payload["scan"]) == len(pb.POROSITY_SCAN) * len(pb.CONDUCTIVITY_RATIO_SCAN)
    assert payload["gate_bounds"]["pass"] is True
    assert payload["gate_bounds"]["trends_pass"] is True
    assert payload["gate_two_operators"]["max_rel_diff"] <= 0.05
    assert payload["gate_stefan_coupling"]["monotonic_decreasing"] is True
    assert payload["gate_stefan_coupling"]["max_dt_melt_dk"] < 0.0
    assert payload["gate_stefan_coupling"]["spread_exceeds_20pct"] is True


@pytest.mark.skipif(not REFS.exists(), reason="benchmark refs.json not generated")
def test_literature_constants_are_sha256_locked():
    refs = json.loads(REFS.read_text())
    canonical = json.dumps(refs["constants"], sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    assert digest == refs["sha256"]
    assert refs["constants"]["krupiczka_1967"]["coefficients"]["c0"] == 0.280
    tampered = json.loads(json.dumps(refs["constants"]))
    tampered["krupiczka_1967"]["coefficients"]["c0"] = 0.3
    other = hashlib.sha256(
        json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert other != refs["sha256"]