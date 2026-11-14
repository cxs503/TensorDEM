"""Round 4: contract *lineage* -- version discipline for the frozen contract.

The round 1-3 tests freeze the surface and detect drift between the frozen data
and the live classes.  They cannot see an edit of the frozen data itself: the
payload may be rewritten (``sha256:0cbfed8c...`` -> ``sha256:048e4cf9...`` in
round 3) while ``CONTRACT_VERSION`` stays put and no record exists.  These tests
pin the missing half:

* the release chain must end at the pinned fingerprint *and* the declared
  version, with a bump that covers the change classes actually observed;
* the change classes of a re-freeze are *derived* from
  :func:`tensordem.binding_contract.diff_contracts` against the recorded
  baseline manifest, never hand-written;
* a binder's pinned manifest can be diffed against the provider contract
  offline (no solver, no torch on the diff path).
"""

import json
from pathlib import Path

import pytest

from tensordem import binding_contract as bc

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "xval_library_binding_contract"
BASELINE_1_0_0 = ARTIFACTS / "manifest_v1.0.0.json"
ROUND3_UNRECORDED = ARTIFACTS / "manifest_round3_unrecorded.json"


def _payload() -> dict:
    return json.loads(bc.canonical_payload())


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The chain itself
# ---------------------------------------------------------------------------
def test_history_tail_is_the_pinned_freeze():
    tail = bc.CONTRACT_HISTORY[-1]
    assert tail["fingerprint"] == bc.FROZEN_FINGERPRINT
    assert tail["fingerprint"] == bc.fingerprint()
    assert tail["version"] == bc.CONTRACT_VERSION
    assert bc.check_lineage()["status"] == "CONFORMS"


def test_lineage_checks_are_counted_not_merely_run():
    report = bc.check_lineage()
    assert report["checks"] == report["matched"]
    assert report["breaking"] == 0
    assert report["entries"] == len(bc.CONTRACT_HISTORY) >= 2


def test_change_classes_are_derived_from_the_recorded_baseline():
    """The declared 1.1.0 changes must equal the diff against the 1.0.0 manifest."""
    baseline = _load(BASELINE_1_0_0)
    assert baseline["fingerprint"] == bc.CONTRACT_HISTORY[0]["fingerprint"]
    findings = bc.diff_contracts(baseline["payload"], _payload())
    summary = bc.summarise_findings(findings)
    tail = bc.CONTRACT_HISTORY[-1]
    assert tail["changes"] == summary["change_classes"]
    assert tail["bump"] == bc.required_bump(findings) == "minor"
    assert summary["by_severity"]["breaking"] == 0


def test_round3_refreeze_without_a_record_is_visible():
    """Round 3 moved the payload and left the version at 1.0.0: that is the hole."""
    unrecorded = _load(ROUND3_UNRECORDED)
    assert unrecorded["fingerprint"] != bc.CONTRACT_HISTORY[0]["fingerprint"]
    assert unrecorded["contract_version"] == bc.CONTRACT_HISTORY[0]["version"]
    report = bc.check_consumer_manifest(unrecorded)
    assert report["status"] == "DRIFT"
    assert "manifest_version_not_in_history" in {f["change"] for f in report["findings"]}


@pytest.mark.parametrize("row", bc.lineage_matrix(), ids=lambda r: f"{r['kind']}-{r['key']}")
def test_every_lineage_hole_is_detected(row):
    assert row["detected"], row
    if row["expect_change"] is None:
        assert row["status"] == "CONFORMS"


def test_lineage_matrix_has_a_clean_control_per_kind():
    kinds = {row["kind"] for row in bc.lineage_matrix() if row["expect_change"] is None}
    assert kinds == {"lineage", "manifest"}


def test_record_history_entry_refuses_an_understated_bump():
    baseline = _load(BASELINE_1_0_0)
    with pytest.raises(bc.BindingContractError) as excinfo:
        bc.record_history_entry(baseline, version="9.9.9", bump="patch", reason="x")
    assert excinfo.value.report["status"] == "UNDERSTATED_BUMP"
    assert excinfo.value.report["required_bump"] == "minor"

    entry = bc.record_history_entry(baseline, version="9.9.9", reason="x")
    assert entry["bump"] == "minor"
    assert entry["fingerprint"] == bc.FROZEN_FINGERPRINT
    assert entry["changes"] == ["surface_ledger_member_added"]


# ---------------------------------------------------------------------------
# Contract-to-contract diff: the metadata verify() cannot see
# ---------------------------------------------------------------------------
def test_contract_version_is_lineage_bookkeeping_not_a_drift_finding():
    mutated = _payload()
    mutated["contract_version"] = "7.7.7"
    assert bc.diff_contracts(_payload(), mutated) == []
    # ... but an unrecorded version in the chain is still caught
    report = bc.check_lineage(
        [dict(entry) for entry in bc.CONTRACT_HISTORY[:-1]]
        + [dict(bc.CONTRACT_HISTORY[-1], version="7.7.7")])
    assert "history_version_not_bumped" in {f["change"] for f in report["findings"]}


def _mutate(path_fn) -> dict:
    mutated = _payload()
    path_fn(mutated)
    return mutated


@pytest.mark.parametrize(
    "mutator,expected_change,expected_severity",
    [
        (lambda c: c["entries"]["IceDEM"]["state"]["dt"].__setitem__("units", "ms"),
         "attribute_units_changed", "breaking"),
        (lambda c: c["entries"]["IceDEM"]["state"]["positions"].__setitem__("shape", "(N, 3)"),
         "attribute_shape_changed", "breaking"),
        (lambda c: c["entries"]["IceDEM"]["methods"]["step"]["params"].pop("load_time"),
         "param_removed", "breaking"),
        (lambda c: c["entries"]["IceDEM"]["methods"].pop("exchange_state"),
         "method_removed", "breaking"),
        (lambda c: c["entries"]["IceDEM"]["methods"]["forces"].__setitem__("returns", "float"),
         "method_returns_changed", "breaking"),
        (lambda c: c["entries"]["IceDEM"]["clock"].__setitem__(
            "load_time_guard_rel_tol", 1e-6), "clock_rule_changed", "breaking"),
        (lambda c: c["entries"]["IceDEM"]["capabilities"].pop("restart"),
         "capability_removed", "breaking"),
        (lambda c: c["entries"]["IceDEM"]["surface_ledger"]["fragments"].__setitem__(
            "disposition", "excluded"), "surface_ledger_disposition_tightened", "breaking"),
        (lambda c: c["entries"]["IceDEM"]["methods"]["step"]["params"].__setitem__(
            "bending_stiffness",
            {"kind": "keyword_only", "required": False}), "param_added_optional", "additive"),
        (lambda c: c["entries"]["IceDEM"]["capabilities"].__setitem__(
            "tau_relaxation", "new"), "capability_added", "additive"),
        (lambda c: c["entries"]["IceDEM"]["explicitly_absent"].append("set_loads"),
         "negative_assertion_added", "narrowing"),
    ],
)
def test_diff_classifies_contract_metadata_drift(mutator, expected_change, expected_severity):
    findings = bc.diff_contracts(_payload(), _mutate(mutator))
    hits = [f for f in findings if f["change"] == expected_change]
    assert hits, [f["change"] for f in findings]
    assert all(f["severity"] == expected_severity for f in hits)


def test_widening_and_narrowing_of_parameter_kinds_are_opposite():
    widened = _mutate(lambda c: c["entries"]["IceDEM"]["methods"]["step"]["params"]
                      ["load_time"].__setitem__("kind", "positional_or_keyword"))
    narrowed = _mutate(lambda c: c["entries"]["IceDEM"]["methods"]["step"]["params"]
                       ["external_forces"].__setitem__("kind", "keyword_only"))
    widened_changes = {f["change"] for f in bc.diff_contracts(_payload(), widened)}
    narrowed_changes = {f["change"] for f in bc.diff_contracts(_payload(), narrowed)}
    assert widened_changes == {"param_kind_widened"}
    assert narrowed_changes == {"param_kind_narrowed"}


def test_units_retune_is_invisible_to_the_introspection_verifier():
    """The round-3 blind spot: a silently retyped unit matched no live check.

    :func:`verify` compares the contract against the live classes, so contract
    metadata that has no live counterpart (units) can be rewritten -- including
    by mistake -- without a single finding.  :func:`diff_contracts` is the only
    detector for it, and it must stay that way.
    """
    pytest.importorskip("torch")
    mutated = _mutate(lambda c: c["entries"]["IceDEM"]["state"]["dt"].__setitem__("units", "ms"))
    targets = bc.default_instances()
    blind = bc.verify(targets, contract=mutated)
    assert blind["breaking"] == 0
    assert blind["status"] == "CONFORMS"
    seen = {f["change"] for f in bc.diff_contracts(_payload(), mutated)}
    assert seen == {"attribute_units_changed"}


# ---------------------------------------------------------------------------
# Binder side: pinned manifest, offline
# ---------------------------------------------------------------------------
def test_manifest_roundtrip_is_conformant():
    report = bc.check_consumer_manifest(bc.export_manifest())
    assert report["status"] == "CONFORMS"
    assert report["breaking"] == 0
    assert report["checks"] == report["matched"] >= 5


def test_manifest_with_a_rewritten_payload_is_flagged():
    manifest = bc.export_manifest()
    manifest["payload"] = bc.export_manifest(
        {**_payload(), "units": "CGS"})["payload"]
    report = bc.check_consumer_manifest(manifest)
    changes = {f["change"] for f in report["findings"]}
    assert "manifest_payload_tampered" in changes
    assert "units_changed" in changes


def test_manifest_of_an_unknown_schema_is_rejected():
    report = bc.check_consumer_manifest({"schema": "something/1"})
    assert report["status"] == "DRIFT"
    assert report["findings"][0]["change"] == "manifest_schema_unknown"


def test_older_manifest_degrades_to_compatible_additive():
    baseline = _load(BASELINE_1_0_0)
    report = bc.check_consumer_manifest(baseline)
    assert report["status"] == "COMPATIBLE_ADDITIVE"
    assert report["required_bump"] == "minor"
    assert report["pinned_version"] == "1.0.0"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_lineage_and_manifest_roundtrip(tmp_path, capsys):
    assert bc._main(["--lineage"]) == 0
    printed = capsys.readouterr().out
    assert '"CONFORMS"' in printed

    manifest_path = tmp_path / "manifest.json"
    assert bc._main(["--export-manifest", str(manifest_path)]) == 0
    assert bc._main(["--check-manifest", str(manifest_path)]) == 0
    assert '"CONFORMS"' in capsys.readouterr().out

    assert bc._main(["--record-history", str(manifest_path),
                     "--record-version", "9.9.9", "--record-reason", "cli"]) == 0
    entry = json.loads(capsys.readouterr().out)
    # the exported manifest *is* the current payload: a version-only re-freeze
    assert entry["bump"] == "patch"
    assert entry["changes"] == ["lineage_version_bump"]
    assert entry["fingerprint"] == bc.FROZEN_FINGERPRINT