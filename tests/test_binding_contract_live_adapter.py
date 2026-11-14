"""Live cross-repository binding check: real TensorPhysics consumer vs real IceDEM.

Anchor: XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT (round 4 / m19466).

The supplier-side contract (``tensordem.binding_contract``) is verified by
``tests/test_binding_contract.py``; this file verifies the *consumer* side by
executing the real ``tensorphysics.adapters.tensordem.TensorDEMAdapter`` inside
a real ``tensorphysics.Simulation`` against a real ``tensordem.dem.IceDEM``.
When the upstream checkout is absent the test skips (never fakes numbers).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "xval_library_binding_contract_live.py"


def _load_entry():
    """Import the entry script by path so the test cannot silently drift from it."""
    spec = importlib.util.spec_from_file_location("xval_live_binding_entry", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report():
    entry = _load_entry()
    src = entry._find_tensorphysics()
    if src is None:
        pytest.skip("tensorphysics checkout not available (set TENSORPHYSICS_SRC)")
    sys.path.insert(0, str(src))
    return entry.collect()


def _row(report, name):
    for row in report["rows"]:
        if row["name"] == name:
            return row
    raise AssertionError(f"arm {name} missing from the live binding report")


def test_every_pre_registered_arm_passes(report):
    assert report["verdict"] == "MET", [
        (r["name"], r["ref"], r["got"]) for r in report["rows"] if not r["pass"]
    ]
    assert report["rows_passed"] == report["rows_total"]


def test_fingerprint_matches_the_pinned_contract(report):
    from tensordem import binding_contract as bc

    assert report["contract_fingerprint"] == bc.FROZEN_FINGERPRINT
    assert report["fingerprint_is_pinned"] is True


def test_consumer_requests_exactly_one_load_per_native_substep(report):
    row = _row(report, "substep_count")
    assert row["ref"] == row["got"] == 12, row


def test_load_time_is_stamped_at_each_substep_start(report):
    row = _row(report, "load_time_stamp_max_dev_s")
    assert row["got"] <= 1e-12, row
    assert _row(report, "loads_passed_verbatim_calls")["pass"] is True


def test_external_impulse_identity_holds_for_the_real_consumer(report):
    row = _row(report, "external_impulse_rel_error")
    assert row["rel_error"] <= 1e-12, row
    assert _row(report, "impulse_ratio_got_over_ref")["rel_error"] <= 1e-12


def test_accumulation_drift_is_separated_from_the_contract_prediction(report):
    """H0 (accumulated loads) predicts (n+1)/2 = 6.5x; H1 predicts 1.0x."""
    sep = _row(report, "h0_separation")
    assert sep["got"] >= 0.15, sep
    drift = report["drift"]["accumulating_supplier"]
    assert drift["detected"] is True
    assert all(abs(r - drift["predicted_by_H0"]) < 1e-9 for r in drift["ratio"]), drift


def test_non_accumulation_is_flat_across_substep_counts(report):
    """Measured impulse ratio must stay 1.0 for n=1..16 while H0 predicts (n+1)/2."""
    row = _row(report, "non_accumulation_sweep_max_dev")
    assert row["pass"] is True, row
    sweep = report["sweep"]
    assert [leg["substeps"] for leg in sweep] == [1, 2, 4, 8, 12, 16]
    for leg in sweep:
        assert max(abs(x - 1.0) for x in leg["ratio_got_over_ref"]) <= 1e-12, leg
        assert leg["state"] == "FINISHED", leg
    assert sweep[-1]["h0_prediction"] == 8.5


def test_substep_rule_boundary_rejects_through_the_real_adapter(report):
    assert _row(report, "substep_boundary[ratio+1e-10]")["pass"] is True
    assert _row(report, "substep_boundary[ratio+1e-6]")["pass"] is True
    assert _row(report, "substep_boundary[ratio 4.5]")["pass"] is True


def test_consumer_does_not_claim_restartability(report):
    assert _row(report, "no_restore_claim")["pass"] is True
    assert report["drift"]["stale_native_clock"]["detected"] is True


def test_committed_artifact_matches_a_fresh_run(report):
    """The committed JSON must be a real run of this script, not a transcription."""
    import json

    path = ROOT / "artifacts" / "xval_library_binding_contract_live" / "live_binding.json"
    if not path.is_file():
        pytest.skip("committed artifact missing in this checkout")
    committed = json.loads(path.read_text())
    assert committed["contract_fingerprint"] == report["contract_fingerprint"]
    assert committed["rows_total"] == report["rows_total"]
    fresh = {r["name"]: r["got"] for r in report["rows"]}
    for row in committed["rows"]:
        assert row["pass"] is True, row
        if isinstance(row["got"], float) and isinstance(fresh[row["name"]], float):
            assert row["got"] == pytest.approx(fresh[row["name"]], rel=1e-9, abs=1e-18)