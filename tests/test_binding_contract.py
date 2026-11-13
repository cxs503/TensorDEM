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


# ---------------------------------------------------------------------------
# Round 2: the contract must still cover what the binder actually calls
# ---------------------------------------------------------------------------
def test_pinned_consumer_surface_is_covered_by_the_frozen_contract():
    """Every name the TensorPhysics adapter calls is frozen (7/7 here).

    This is the round-2 gap the round-1 contract left open: the frozen entries
    could stay self-consistent while the binder grew calls the contract never
    described.
    """
    report = bc.consumer_coverage()  # pinned surface, no binder checkout needed
    assert report["status"] == "COVERED", report["gaps"]
    assert report["checks"] >= 7
    assert report["covered"] == report["checks"]
    fields = {row.get("contract_field") for row in report["rows"]}
    assert "IceDEM.methods.step.external_forces" in fields
    assert "IceDEM.methods.step.load_time" in fields
    assert {"IceDEM.state.dt", "IceDEM.state.time",
            "IceDEM.state.step_count"} <= fields


def test_live_binder_surface_is_covered():
    """Audit the binder as it is today, when a TensorPhysics checkout is around."""
    live = bc.live_consumer_surface()
    if live is None:
        pytest.skip("no TensorPhysics checkout (set TENSORPHYSICS_SRC to audit)")
    report = bc.consumer_coverage(live)
    assert report["status"] == "COVERED", report["gaps"]
    assert bc.assert_consumer_covered(live)["covered"] == report["checks"]


def test_live_binder_surface_matches_the_pin():
    """Drift alarm: a changed binder source must be re-pinned deliberately."""
    live = bc.live_consumer_surface()
    if live is None:
        pytest.skip("no TensorPhysics checkout (set TENSORPHYSICS_SRC to audit)")
    assert live["source_sha256"] == bc.CONSUMER_SURFACE_PIN["source_sha256"], (
        "the TensorPhysics adapter changed since the surface was pinned; re-run "
        "`python -m tensordem.binding_contract --consumer-surface`, confirm the "
        "coverage is still complete, and update CONSUMER_SURFACE_PIN in the same "
        "commit")


def test_consumer_detector_flags_new_binder_calls(tmp_path):
    """Mutation check: the extractor is not a stub, it sees *new* calls."""
    source = '''
class Binding:
    def advance(self, model, n):
        for _ in range(n):
            model.step(external_forces=None, load_time=0.0, bogus_load=1.0)
        dt = model.dt
        dx = model.wobble  # a name the contract has never seen
        return dt, dx
'''
    surface = bc.consumer_surface(source)
    assert surface["attributes"] == ["dt", "step", "wobble"]
    assert surface["methods"]["step"]["keyword_params"] == [
        "bogus_load", "external_forces", "load_time"]
    report = bc.consumer_coverage(surface)
    assert report["status"] == "GAP"
    assert report["gaps"] == ["wobble", "step(bogus_load=...)"]
    with pytest.raises(bc.BindingContractError):
        bc.assert_consumer_covered(surface)


def test_consumer_detector_handles_getattr_and_private_names():
    source = '''
def probe(model):
    ok = hasattr(model, "time")
    positions = getattr(model, "positions", None)
    hidden = model._internal_cache
    return ok, positions, hidden
'''
    surface = bc.consumer_surface(source)
    assert surface["attributes"] == ["positions", "time"]
    assert surface["methods"] == {}


def test_consumer_pin_is_separate_from_coverage():
    """A binder edited to call *fewer* frozen names stays COVERED (the provider
    may keep the surface); ``pin_match=False`` is what tells the operator that
    the pinned snapshot is stale and needs a deliberate re-pin."""
    shrunk = dict(bc.CONSUMER_SURFACE_PIN, source_sha256="sha256:" + "0" * 64,
                  attributes=["dt"], methods={})
    report = bc.consumer_coverage(shrunk)
    assert report["status"] == "COVERED"
    assert report["pin_match"] is False
    assert report["covered"] == report["checks"] == 1


def test_verify_accepts_class_targets_without_false_drift():
    """Documented class-target path: instance attributes are proven statically."""
    from tensordem.dem import IceDEM
    from tensordem.dem3d import IceDEM3D

    report = bc.verify({"IceDEM": IceDEM, "IceDEM3D": IceDEM3D})
    assert report["status"] == "CONFORMS", report["findings"]
    assert report["breaking"] == 0
    assert report["matched"] == report["checks"]


def test_verify_still_rejects_a_genuinely_missing_state_attribute():
    contract = {
        "schema": bc.CONTRACT_SCHEMA,
        "contract_version": bc.CONTRACT_VERSION,
        "entries": {"Dummy": {"state": {"alpha": {}, "beta": {}},
                              "properties": {}, "methods": {},
                              "explicitly_absent": []}},
    }

    class Complete:
        def __init__(self) -> None:
            self.alpha = 1.0
            self.beta = 2.0

    class Missing:
        def __init__(self) -> None:
            self.alpha = 1.0

    ok = bc.verify({"Dummy": Complete}, contract=contract)
    assert ok["status"] == "CONFORMS"
    assert ok["matched"] == ok["checks"] == 2

    drifted = bc.verify({"Dummy": Missing}, contract=contract)
    assert drifted["status"] == "DRIFT"
    assert [f["field"] for f in drifted["findings"]] == ["state.beta"]


# ---------------------------------------------------------------------------
# Round 3: surface accounting (no silent surface)
# ---------------------------------------------------------------------------
def test_surface_ledger_classifies_every_public_member():
    """Round-1 limitation: 12 of the 20 public 3-D members were unmentioned."""
    from tensordem.dem import IceDEM
    from tensordem.dem3d import IceDEM3D

    for entry_name, cls in (("IceDEM", IceDEM), ("IceDEM3D", IceDEM3D)):
        audit = bc.audit_surface(cls, entry_name)
        assert audit["status"] == "COMPLETE", audit["gaps"]
        assert audit["gaps"] == []
        assert audit["classified_ratio"] == 1.0
        assert audit["public_count"] == len(bc.public_members(cls))
        assert audit["frozen_count"] + audit["excluded_count"] == audit["public_count"]

    assert bc.audit_surface(IceDEM3D, "IceDEM3D")["excluded_count"] == 12
    assert bc.audit_surface(IceDEM, "IceDEM")["excluded_count"] == 0


def test_every_exclusion_carries_a_reason_and_stays_out_of_the_frozen_fields():
    entry = bc.BINDING_CONTRACT["entries"]["IceDEM3D"]
    frozen_fields = set(entry["state"]) | set(entry["properties"]) | set(entry["methods"])
    excluded = {name: spec for name, spec in entry["surface_ledger"].items()
                if spec["disposition"] == "excluded"}
    assert excluded, "the 3-D diagnostics must be classified, not dropped"
    for name, spec in excluded.items():
        assert spec.get("reason", "").strip(), name
        assert name not in frozen_fields, name


def test_new_public_member_is_a_breaking_finding():
    """A public member nobody classified may not ride along silently."""
    from tensordem.dem import IceDEM

    class GrownIceDEM(IceDEM):
        def thermal_flux(self) -> float:  # noqa: D102
            return 0.0

    audit = bc.audit_surface(GrownIceDEM, "IceDEM")
    assert audit["status"] == "GAPS"
    gap = next(g for g in audit["gaps"] if g["name"] == "thermal_flux")
    assert gap["finding"] == "public_member_unclassified"
    with pytest.raises(bc.BindingContractError):
        bc.assert_surface_classified(GrownIceDEM, "IceDEM")


def test_stale_ledger_entry_and_misclassification_are_reported():
    import copy

    stale = copy.deepcopy(bc.BINDING_CONTRACT)
    stale["entries"]["IceDEM"]["surface_ledger"]["ghost_method"] = {"disposition": "frozen"}
    audit = bc.audit_surface(None, "IceDEM", stale)
    assert any(g["finding"] == "surface_ledger_stale" and g["name"] == "ghost_method"
               for g in audit["gaps"])

    mislabelled = copy.deepcopy(bc.BINDING_CONTRACT)
    mislabelled["entries"]["IceDEM"]["surface_ledger"]["step"] = {
        "disposition": "excluded", "reason": "pretend it is a diagnostic"}
    audit = bc.audit_surface(None, "IceDEM", mislabelled)
    assert any(g["finding"] == "surface_ledger_misclassified" and g["name"] == "step"
               for g in audit["gaps"])

    unreasoned = copy.deepcopy(bc.BINDING_CONTRACT)
    unreasoned["entries"]["IceDEM3D"]["surface_ledger"]["body_forces"] = {
        "disposition": "excluded"}
    audit = bc.audit_surface(None, "IceDEM3D", unreasoned)
    assert any(g["finding"] == "surface_ledger_misclassified" and g["name"] == "body_forces"
               for g in audit["gaps"])


def test_surface_accounting_is_wired_into_verify_and_the_export_gate():
    report = bc.verify(bc.default_instances())
    assert report["status"] == "CONFORMS"
    surface = report["surface"]
    assert set(surface) == {"IceDEM", "IceDEM3D"}
    assert all(entry["status"] == "COMPLETE" for entry in surface.values())
    # The frozen-field checks are unchanged: the accounting is additive.
    accounting = sum(entry["public_count"] for entry in surface.values())
    assert report["checks"] == 42 + accounting
    assert report["matched"] == report["checks"]

    entry_report = bc.verify_entry(bc.default_instances()["IceDEM3D"], "IceDEM3D")
    assert entry_report["surface"]["status"] == "COMPLETE"
    assert entry_report["surface"]["public_count"] == 20


def test_surface_mutation_check_is_not_decorative():
    rows = bc.surface_mutation_check()
    assert {row["case"] for row in rows} == {
        "public_member_added", "ledger_entry_missing_for_frozen_member",
        "frozen_member_marked_excluded", "unmodified_contract"}
    assert all(row["detected"] for row in rows), rows


def test_class_target_is_supported_by_verify_entry():
    """Documented class path must not report instance state as removed.

    Round 3 defect fix: ``verify_entry(IceDEM, "IceDEM")`` reported 10 bogus
    ``attribute_removed`` findings because the class-level source fallback was
    only wired into :func:`verify`.  A thin subclass (state inherited from
    another module) needs the fallback to walk the MRO.
    """
    from tensordem.dem import IceDEM
    from tensordem.dem3d import IceDEM3D

    for entry_name, cls in (("IceDEM", IceDEM), ("IceDEM3D", IceDEM3D)):
        report = bc.verify_entry(cls, entry_name)
        assert report["status"] == "CONFORMS", report["findings"]
        assert report["surface"]["status"] == "COMPLETE"

    class Thin(IceDEM):
        pass

    assert bc.verify_entry(Thin, "IceDEM")["status"] == "CONFORMS"

    class NoStep(IceDEM):
        step = None  # the member the binder calls disappears

    drift = bc.verify_entry(NoStep, "IceDEM")
    assert drift["status"] == "DRIFT"
    assert any(f["field"] == "methods.step" for f in drift["findings"])


def test_surface_mutation_check_reports_the_pre_round3_baseline():
    """The accounting is the increment: the round-1/2 checks saw nothing."""
    growth = next(row for row in bc.surface_mutation_check()
                  if row["case"] == "public_member_added")
    assert growth["baseline_findings"] == 0
    assert growth["baseline_detected"] is False
    assert growth["detected"] is True
