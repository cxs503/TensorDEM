"""Guard the PR #216 cure: the merge of ``agent/m3558`` into ``main`` is lossless.

The heavy lifting lives in ``scripts/verify_pr216_merge_semantics.py``; this test
only makes sure the recorded decision cannot rot silently.  On a shallow clone
(CI checkouts) the pinned parent commits are absent, in which case the check is
skipped rather than faked.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "verify_pr216_merge_semantics.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_pr216", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _has_history(mod) -> bool:
    return all(
        subprocess.run(["git", "cat-file", "-e", f"{ref}^{{commit}}"], cwd=ROOT,
                       capture_output=True).returncode == 0
        for ref in (mod.MERGE_BASE, mod.MAIN, mod.BRANCH, mod.MERGE)
    )


def test_cure_verification_passes(tmp_path):
    mod = _load()
    if not _has_history(mod):
        pytest.skip("pinned cure commits not present (shallow clone)")
    out = tmp_path / "verify.json"
    assert mod.main(["--json", str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["overall"] == "PASS"
    assert all(c["ok"] for c in record["checks"].values())


def test_committed_artefact_matches_verifier(tmp_path):
    """The checked-in decision record must be what the verifier produces now."""
    mod = _load()
    if not _has_history(mod):
        pytest.skip("pinned cure commits not present (shallow clone)")
    committed = ROOT / "artifacts" / "pr216" / "verify_pr216.json"
    assert committed.exists(), "cure decision record missing"
    recorded = json.loads(committed.read_text())
    out = tmp_path / "verify.json"
    assert mod.main(["--json", str(out)]) == 0
    fresh = json.loads(out.read_text())
    assert fresh["shas"] == recorded["shas"]
    assert fresh["overall"] == recorded["overall"] == "PASS"
    for name, check in fresh["checks"].items():
        assert check["ok"] == recorded["checks"][name]["ok"], name


def test_conflict_marker_check_is_not_vacuous():
    """C4 must actually inspect files (a passing no-op check would be worthless)."""
    mod = _load()
    result = mod.c4_conflict_markers()
    assert result["ok"] is True
    assert result["hits"] == []
    assert (ROOT / "docs" / "doc_evidence_ledger.json").exists()
    ledger = json.loads((ROOT / "docs" / "doc_evidence_ledger.json").read_text())
    assert ledger["documents_scanned"] >= 107  # union of both parents (106 + 106)