"""Frozen binding-contract tests (provider side of the cross-repo binding).

Anchor: XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT.  The reference practice is
the *frozen contract* that ``TensorPhysics`` publishes for its solver
adapters (``docs/adapter-contracts.md``, ``docs/adapter-inventory.md``,
``tests/test_adapter_inventory_meta.py``); here TensorDEM freezes the surface
it *provides*, data-driven and fingerprinted.

The tests are deliberately cheap: small lattices, no simulation campaign.
"""

import json

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