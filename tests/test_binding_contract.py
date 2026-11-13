"""Frozen binding-contract tests (provider side of the cross-repo binding).

Anchor: XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT.  The reference practice is
the *frozen contract* that ``TensorPhysics`` publishes for its solver
adapters (``docs/adapter-contracts.md``, ``docs/adapter-inventory.md``,
``tests/test_adapter_inventory_meta.py``); here TensorDEM freezes the surface
it *provides*, data-driven and fingerprinted.

The tests are deliberately cheap: small lattices, no simulation campaign.
"""

import json
import os
from pathlib import Path

import pytest

from tensordem import binding_contract as bc

TORCH = pytest.importorskip("torch", reason="the DEM solver surface needs torch")

#: Fields the TensorPhysics adapter contract consumes from the 2-D surface
#: (see TensorPhysics/docs/adapter-contracts.md, "TensorDEM" section).  If any
#: of these disappears from the frozen contract, a cross-repo binder can break
#: without anyone noticing.
BINDER_CONSUMED_FIELDS = {
    "state": {"dt", "time", "step_count", "positions", "velocities"},
    "methods": {"step", "diagnostics", "forces"},
}


def test_frozen_fingerprint_is_pinned():
    """The contract hash is pinned; editing the frozen surface must be explicit."""
    assert bc.FROZEN_FINGERPRINT.startswith("sha256:")
    assert bc.fingerprint() == bc.FROZEN_FINGERPRINT


def test_contract_is_json_serialisable_and_hash_stable():
    text = json.dumps(bc.BINDING_CONTRACT, sort_keys=True)
    round_tripped = json.loads(text)
    assert bc.fingerprint(round_tripped) == bc.fingerprint()
    assert bc.canonical_payload() == bc.canonical_payload(round_tripped)


def test_live_surface_conforms_to_contract():
    report = bc.assert_conforms()
    assert report["status"] == "CONFORMS"
    assert report["breaking"] == 0
    assert report["checks"] > 0
    assert report["matched"] == report["checks"]
    assert report["frozen_fingerprint_match"] is True


def test_contract_covers_the_binder_consumed_surface():
    entry = bc.BINDING_CONTRACT["entries"]["IceDEM"]
    assert BINDER_CONSUMED_FIELDS["state"].issubset(entry["state"])
    assert BINDER_CONSUMED_FIELDS["methods"].issubset(entry["methods"])
    step_params = entry["methods"]["step"]["params"]
    assert step_params["external_forces"]["shape"] == "(N, 2)"
    assert step_params["external_forces"]["units"] == "N"
    assert "NOT accumulated" in step_params["external_forces"]["semantics"]
    assert step_params["load_time"]["kind"] == "keyword_only"


def test_2d_and_3d_restart_surfaces_are_frozen_apart():
    """Frozen structural difference: the 3-D restart API is not the 2-D one."""
    entries = bc.BINDING_CONTRACT["entries"]
    assert "snapshot" in entries["IceDEM"]["methods"]
    assert "state_dict" in entries["IceDEM3D"]["methods"]
    for absent in ("snapshot", "from_snapshot", "exchange_state"):
        assert absent in entries["IceDEM3D"]["explicitly_absent"]
        assert absent not in entries["IceDEM3D"]["methods"]


def test_negative_assertions_hold_on_live_classes():
    from tensordem.dem import IceDEM
    from tensordem.dem3d import IceDEM3D

    assert not hasattr(IceDEM, "restore"), "checkpoint must not be advertised as restart"
    assert not hasattr(IceDEM3D, "restore")


@pytest.mark.parametrize("spec", bc.MUTATION_SPECS, ids=lambda s: s["key"])
def test_each_drift_class_is_detected(spec):
    rows = {row["key"]: row for row in bc.mutation_matrix()}
    row = rows[spec["key"]]
    assert row["detected"], row
    assert row["observed_change"] == spec["expect_change"]
    assert row["observed_severity"] == spec["expect_severity"]
    if spec["expect_severity"] == "breaking":
        assert row["status"] == "DRIFT"
        assert row["breaking_findings"] >= 1


def test_mutation_detection_rate_is_complete():
    rows = bc.mutation_matrix()
    assert len(rows) == len(bc.MUTATION_SPECS)
    assert all(row["detected"] for row in rows), rows


def test_clean_surface_reports_no_findings():
    report = bc.verify(bc.default_instances())
    assert report["findings"] == []


def test_substep_rule_acceptance_boundary():
    rows = {row["ratio"]: row for row in bc.substep_rule_check()}
    for exact in (1.0, 2.0, 4.0, 16.0):
        assert rows[exact]["accepted"] is True
        assert rows[exact]["n_substeps_accepted"] == int(exact)
    assert rows[4.000001]["accepted"] is False
    assert rows[4.5]["accepted"] is False


def test_semantic_probes_all_pass():
    records = bc.probe_semantics()
    failures = [r for r in records if r["status"] != "PASS"]
    assert not failures, failures
    assert len(records) >= 19


def test_impulse_momentum_error_is_machine_precision():
    """Owner accuracy criterion (<15 %) is met by orders of magnitude."""
    records = bc.probe_semantics()
    errors = [r["error"] for r in records
              if r["name"].startswith("impulse_momentum_identity") and r["error"] is not None]
    assert errors, "impulse probes did not report errors"
    assert max(errors) < 0.15
    assert max(errors) < 1e-12


def test_loads_are_not_accumulated_across_calls():
    """H1 (instantaneous loads) vs H0 (accumulating binder) are separable."""
    ratios = [r["got"] for r in bc.probe_semantics() if r["name"] == "loads_not_accumulated"]
    assert ratios
    for got in ratios:
        ratio = float(got.split("=")[1])
        assert abs(ratio - 1.0) < 1e-12, got


def test_contract_module_imports_without_torch(monkeypatch):
    """The declarative contract must be readable even where torch is absent."""
    import importlib
    import sys

    for name in list(sys.modules):
        if name == "tensordem.binding_contract":
            monkeypatch.delitem(sys.modules, name)
    module = importlib.import_module("tensordem.binding_contract")
    assert module.fingerprint() == bc.FROZEN_FINGERPRINT
    report = module.verify({"IceDEM": object, "IceDEM3D": object})
    # An untyped object drifts, and the report must say so rather than crash.
    assert report["status"] == "DRIFT"
    assert report["breaking"] > 0

# ---------------------------------------------------------------------------
# Integration callers: the shipped solver export paths must enforce the gate
# ---------------------------------------------------------------------------
def test_exchange_state_calls_the_frozen_gate(monkeypatch):
    """``IceDEM.exchange_state`` is a real caller of the contract gate."""
    from tensordem import binding_contract as bc
    from tensordem.dem import DEMConfig, IceDEM

    calls = []
    original = bc.assert_entry_conforms

    def spy(target, entry_name, **kwargs):
        calls.append(entry_name)
        return original(target, entry_name, **kwargs)

    monkeypatch.setattr("tensordem.dem.assert_entry_conforms", spy)
    model = IceDEM(DEMConfig(nx=4, ny=3))
    payload = model.exchange_state()
    assert payload["schema"] == "tensordem.exchange/1"
    assert calls == ["IceDEM"]

    # Break a frozen negative assertion (IceDEM must not grow ``restore``):
    # the platform-facing export must fail closed, not emit a drifted surface.
    monkeypatch.setattr(IceDEM, "restore", lambda self: None, raising=False)
    bc.clear_entry_cache()
    with pytest.raises(bc.BindingContractError):
        model.exchange_state()


def test_state_dict_calls_the_frozen_gate():
    """The 3-D restart export is wired to the frozen gate.

    Structural assertion (the behavioural 2-D proof is
    ``test_exchange_state_calls_the_frozen_gate``): ``IceDEM3D.state_dict``
    must call the gate before emitting the binding surface, and a deliberately
    drifted subclass must be refused when the gate is consulted directly.
    """
    import inspect

    from tensordem.dem3d import DEM3DConfig, IceDEM3D

    source = inspect.getsource(IceDEM3D.state_dict)
    assert "assert_entry_conforms" in source
    assert "IceDEM3D" in source

    class DriftedIceDEM3D(IceDEM3D):
        restore = lambda self: None  # noqa: E731 - deliberate negative-assertion breach

    drifted = DriftedIceDEM3D(DEM3DConfig(nx=3, ny=3, nz=3))
    with pytest.raises(bc.BindingContractError):
        bc.assert_entry_conforms(drifted, "IceDEM3D", refresh=True)


def test_verify_entry_detects_instance_level_drift():
    from tensordem import binding_contract as bc
    from tensordem.dem import DEMConfig, IceDEM

    model = IceDEM(DEMConfig(nx=4, ny=3))
    assert bc.verify_entry(model, "IceDEM")["status"] == "CONFORMS"
    del model.positions
    drifted = bc.verify_entry(model, "IceDEM")
    assert drifted["status"] == "DRIFT"
    assert any(f["field"] == "state.positions" for f in drifted["findings"])


def test_assert_entry_conforms_is_cached_but_refresh_works():
    from tensordem import binding_contract as bc
    from tensordem.dem import DEMConfig, IceDEM

    model = IceDEM(DEMConfig(nx=4, ny=3))
    first = bc.assert_entry_conforms(model, "IceDEM", refresh=True)
    second = bc.assert_entry_conforms(model, "IceDEM")
    assert first["cached"] is False
    assert second["cached"] is True
    del model.positions
    with pytest.raises(bc.BindingContractError):
        bc.assert_entry_conforms(model, "IceDEM", refresh=True)


def test_cli_reports_contract_fingerprint_and_benchmark(tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    out = tmp_path / "contract"
    proc = subprocess.run(
        [sys.executable, "-m", "tensordem", "binding-contract",
         "--out", str(out), "--benchmark-energy-steps", "100"],
        cwd=root, env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads((out / "binding_contract_cli.json").read_text(encoding="utf-8"))
    assert report["contract_fingerprint"] == bc.fingerprint()
    assert report["verify"]["status"] == "CONFORMS"
    assert report["benchmark"]["verdict"] == "MET"
    assert report["benchmark"]["worst_rel_error"] < bc.BENCHMARK_ACCEPT_REL_ERROR


# ---------------------------------------------------------------------------
# Numeric benchmark rows
# ---------------------------------------------------------------------------
def test_benchmark_rows_are_ref_vs_got_under_the_owner_line():
    bench = bc.benchmark_physics(n_steps_energy=200)
    assert bench["rows_total"] >= 10
    for row in bench["rows"]:
        assert set(row) >= {"name", "metric", "ref", "got", "rel_error", "pass"}
        assert row["rel_error"] <= bc.BENCHMARK_ACCEPT_REL_ERROR, row
        assert row["pass"] is True, row
    assert bench["verdict"] == "MET"
    names = {row["name"] for row in bench["rows"]}
    assert {"impulse_momentum_2d", "free_flight_vs_symplectic",
            "energy_drift_conservative", "hertz_normal_force",
            "clock_n_dt", "restart_roundtrip_drift"} <= names


def test_free_flight_continuum_error_converges_at_first_order():
    bench = bc.benchmark_physics(n_steps_energy=10)
    orders = bench["free_flight_convergence_order"]
    assert len(orders) == 2
    for order in orders:
        assert 0.9 <= order <= 1.1, orders


def test_energy_drift_is_reported_against_zero():
    bench = bc.benchmark_physics(n_steps_energy=500)
    row = next(r for r in bench["rows"] if r["name"] == "energy_drift_conservative")
    assert row["ref"] == 0.0
    assert row["got"] == row["rel_error"]
    assert row["rel_error"] < 0.15


def test_module_entry_point_runs_without_torch_json_side_effects():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    proc = subprocess.run(
        [sys.executable, "-m", "tensordem.binding_contract", "--verify"],
        cwd=root, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["status"] == "CONFORMS"
    assert report["frozen_fingerprint_match"] is True
