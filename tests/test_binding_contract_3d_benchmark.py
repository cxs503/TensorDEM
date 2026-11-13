"""3-D numeric benchmark of the frozen binding contract (``IceDEM3D``).

Anchor: XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT.  Round 1 anchored the 2-D
surface (``benchmarks`` rows in :func:`tensordem.binding_contract.benchmark_physics`);
the 3-D entry the binders also consume had a frozen *name* surface with a
fingerprint but no numeric anchor.  These tests pin the newly added
:func:`tensordem.binding_contract.benchmark_physics_3d`:

* every row must carry ``ref`` / ``got`` / ``rel_error`` / ``tolerance`` / ``pass``
  and the declared ``pass`` must be exactly ``rel_error <= tolerance``;
* the tight rows (impulse-momentum, symplectic closed form, clock, restart,
  Hertz kernel) must sit at machine precision;
* the continuum rows must converge at first order in ``dt``;
* the benchmark must *have teeth*: at a coarse time step, the naive
  continuum-only reference (no ``n(n+1)/2``) is off by more than the owner 15 %
  line while the correct closed form is exact -- i.e. a wrong reference cannot
  slip through the 15 % gate.

The model is deliberately tiny (60 particles); no simulation campaign.
"""

import math

import pytest

from tensordem import binding_contract as bc

TORCH = pytest.importorskip("torch", reason="the DEM solver surface needs torch")

OWNER_TOLERANCE = 0.15

EXPECTED_ROWS = {
    "impulse_momentum_3d",
    "loads_not_accumulated_3d",
    "free_flight_symplectic_3d",
    "free_flight_continuum_3d_dt_x1",
    "free_flight_continuum_3d_dt_x0.5",
    "free_flight_continuum_3d_dt_x0.25",
    "free_flight_convergence_order_3d",
    "energy_drift_conservative_3d",
    "hertz_reduced_modulus_3d",
    "hertz_effective_radius_3d",
    "hertz_normal_force_3d",
    "clock_n_dt_3d",
    "restart_roundtrip_3d",
    "restart_continuation_3d",
    "shape_guard_3d",
}


@pytest.fixture(scope="module")
def report():
    return bc.benchmark_physics_3d(n_steps_energy=200)


def _by_name(report):
    return {row["name"]: row for row in report["rows"]}


def test_benchmark_schema_and_rows(report):
    assert report["schema"] == "tensordem.binding-contract.benchmark-3d/1"
    assert report["entry"] == "IceDEM3D"
    assert report["rows_total"] == len(report["rows"])
    assert report["rows_total"] >= len(EXPECTED_ROWS)
    names = set(_by_name(report))
    missing = EXPECTED_ROWS - names
    assert not missing, f"benchmark lost rows: {sorted(missing)}"
    for row in report["rows"]:
        for key in ("name", "metric", "ref", "got", "ref_source", "rel_error",
                    "tolerance", "pass", "detail"):
            assert key in row, f"{row.get('name')} misses {key}"
        assert row["ref_source"].strip(), f"{row['name']} has no reference source"


def test_declared_pass_matches_the_tolerance_arithmetic(report):
    """The row verdict is the pre-registered inequality, not a separate claim."""
    for row in report["rows"]:
        assert row["pass"] == (row["rel_error"] <= row["tolerance"]), row["name"]


def test_benchmark_meets_owner_accuracy_standard(report):
    assert report["verdict"] == "MET"
    assert report["rows_passed"] == report["rows_total"]
    assert report["worst_rel_error"] < OWNER_TOLERANCE, report["worst_rel_error"]


def test_identity_rows_are_at_machine_precision(report):
    rows = _by_name(report)
    assert rows["impulse_momentum_3d"]["rel_error"] < 1e-12
    assert rows["loads_not_accumulated_3d"]["rel_error"] < 1e-12
    assert rows["free_flight_symplectic_3d"]["rel_error"] < 1e-11
    assert rows["clock_n_dt_3d"]["rel_error"] < 1e-12
    assert rows["hertz_reduced_modulus_3d"]["rel_error"] < 1e-12
    assert rows["hertz_effective_radius_3d"]["rel_error"] < 1e-12
    assert rows["hertz_normal_force_3d"]["rel_error"] < 1e-12
    # bitwise restart: the payload is exact, not merely close
    assert rows["restart_roundtrip_3d"]["got"] == 0.0
    assert rows["restart_continuation_3d"]["got"] == 0.0
    assert rows["shape_guard_3d"]["got"] == 1.0


def test_continuum_rows_converge_at_first_order(report):
    rows = _by_name(report)
    e1 = rows["free_flight_continuum_3d_dt_x1"]["rel_error"]
    e2 = rows["free_flight_continuum_3d_dt_x0.5"]["rel_error"]
    e4 = rows["free_flight_continuum_3d_dt_x0.25"]["rel_error"]
    assert e1 > e2 > e4 > 0.0
    p1 = math.log(e1 / e2) / math.log(2.0)
    p2 = math.log(e2 / e4) / math.log(2.0)
    for order in (p1, p2):
        assert 0.85 <= order <= 1.15, f"observed order {order} is not first order"
    assert all(0.85 <= o <= 1.15 for o in report["free_flight_convergence_order"])


def test_energy_row_is_tighter_than_the_owner_line(report):
    row = _by_name(report)["energy_drift_conservative_3d"]
    assert row["tolerance"] == pytest.approx(0.02)
    assert row["tolerance"] < OWNER_TOLERANCE
    assert row["pass"]


def test_reference_has_teeth_against_the_naive_continuum_form():
    """A wrong reference must not pass the 15 % line.

    At ``n = 4`` steps the semi-implicit Euler closed form and the continuum
    answer differ by ``1/n = 25 %``.  So the benchmark can distinguish the
    integrator-exact reference (which the code matches to ~1e-13) from the naive
    ``0.5*a*t^2`` reference: had the row used the latter, it would be rejected at
    the owner 15 % line.
    """
    case = bc.free_flight_case_3d(dt_factor=1.0, n_steps=4)
    # the correct closed form is matched exactly by the frozen surface ...
    assert case["symplectic_rel_error"] < 1e-10
    # ... while the continuum-only reference is off by more than the 15 % line
    naive = max(
        bc._rel(case["got_displacement_m"][i], case["ref_continuum_m"][i])
        for i in range(3) if case["ref_continuum_m"][i] != 0.0
    )
    assert naive > OWNER_TOLERANCE, naive


def test_reference_model_is_force_free():
    """Analytic rows are only valid if the isolated lattice really is force-free."""
    model = bc.isolated_3d_model()
    total, tool = model.forces()
    assert float(total.abs().max()) == 0.0
    assert float(tool.abs().max()) == 0.0


def test_benchmark_anchors_are_recorded(report):
    anchors = report["anchors"]
    for key in ("impulse_momentum", "symplectic_euler", "continuum", "hertz_contact",
                "energy", "restart"):
        assert key in anchors and len(anchors[key]) > 20
    assert "Hertz" in anchors["hertz_contact"] or "hertz" in anchors["hertz_contact"].lower()