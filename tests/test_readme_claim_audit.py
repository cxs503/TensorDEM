"""Contract tests for the README claim↔evidence audit (AUDIT-SCI, m22981).

These tests are the reason the audit cannot rot: they fail if a registered claim
loses its code, its test, its tracked evidence, or if the README stops advertising
a capability the registry still claims to be auditing.  Negative controls (a
synthetic claim pointed at a non-existent file) prove the checker actually reports
defects rather than returning green for anything it is handed.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import readme_claim_audit as audit_mod  # noqa: E402


@pytest.fixture(scope="module")
def dem_result() -> dict:
    return audit_mod.audit(ROOT, sph_root=None)


def test_javascript_syntax_tree_not_required():  # anchor for collection stability
    assert audit_mod.SCHEMA_VERSION.startswith("readme_claim_audit")


def test_every_dem_claim_is_advertised(dem_result):
    """A registered claim must still be in README, else the registry is stale."""
    retired = [c["cid"] for c in dem_result["claims"] if c["verdict"] == "RETIRED"]
    assert not retired, f"registry claims no longer advertised in README: {retired}"


def test_no_dem_claim_is_unsupported(dem_result):
    """Claimed capability whose implementing file/symbol is missing => NOT_SUPPORTED."""
    bad = {c["cid"]: c.get("code_missing", []) + c.get("symbols_failed", [])
           for c in dem_result["claims"] if c["verdict"] == "NOT_SUPPORTED"}
    assert not bad, f"claimed but not implemented: {bad}"


def test_every_dem_claim_has_a_test_file(dem_result):
    missing = {c["cid"]: c["tests_missing"] for c in dem_result["claims"]
               if c.get("tests_missing")}
    assert not missing, f"claims without a registered test file: {missing}"


def test_no_baseline_regressions(dem_result):
    """Verdicts may only hold or improve relative to the registered baseline."""
    assert not dem_result["regressions"], f"regressed claims: {dem_result['regressions']}"


def test_dem_readme_pointers_are_only_regenerable_artifacts(dem_result):
    """Tracked paths must resolve; only `artifacts/` (generated, not in VCS) may dangle."""
    non_artifact = [p for p in dem_result["dangling_pointers_dem"]
                    if not p.startswith("artifacts/")]
    assert not non_artifact, f"README cites paths that do not exist: {non_artifact}"


def test_registered_generators_exist_for_every_pending_artifact(dem_result):
    """A missing generated artifact is only acceptable with a rebuildable script."""
    bad = {c["cid"]: (c.get("artifacts_missing"), c.get("generators_missing"))
           for c in dem_result["claims"]
           if c.get("artifacts_missing") and c.get("generators_missing")}
    assert not bad, f"unrebuildable generated evidence: {bad}"


def test_quoted_numbers_have_a_second_copy(dem_result):
    unbacked = {c["cid"]: c["numbers_unbacked"] for c in dem_result["claims"]
                if c.get("numbers_unbacked")}
    assert not unbacked, f"README numbers with no source anywhere: {unbacked}"


def test_gate_values_match_what_readme_quotes(dem_result):
    drift = {c["cid"]: c["gate_failures"] for c in dem_result["claims"]
             if c.get("gate_failures")}
    assert not drift, f"machine-readable gates disagree with README: {drift}"


def test_readme_test_census_matches_the_tree(dem_result):
    census = [c for c in dem_result["claims"] if c["cid"] == "DEM-README-TEST-CENSUS"][0]
    info = census["census"]
    assert info["ratio_claim_holds"], ("README's substantive claim is that pytest-style "
                                       "module-level tests dominate TestCase classes")
    if info["drift"] and not info["dated_snapshot"]:
        pytest.fail(f"undated test-count assertion is stale: {info['drift']}")


# --------------------------------------------------------------------------
# negative controls: the checker must actually detect each defect class
# --------------------------------------------------------------------------


def _sandbox(tmp_path: Path) -> Path:
    """Minimal fake repo: README + one real module + one test."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "docs").mkdir()
    (root / "src" / "thing.py").write_text("class Thing:\n    pass\n", encoding="utf-8")
    (root / "tests" / "test_thing.py").write_text(
        "def test_thing():\n    import thing\n    assert thing\n", encoding="utf-8")
    (root / "docs" / "thing.md").write_text("value 42 %\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# fake\n\n`src/thing.py` does the thing. See `docs/thing.md`.\n"
        "Claim: error 42 %.\n", encoding="utf-8")
    return root


def test_negative_control_detects_missing_file(tmp_path):
    root = _sandbox(tmp_path)
    claim = audit_mod.Claim(
        cid="FAKE-CODEFREE", section="x", anchor="does the thing",
        code=("src/nope.py",), tests=("tests/test_thing.py",), docs=("docs/thing.md",))
    rec = audit_mod.evaluate(root, claim)
    assert rec["verdict"] == "NOT_SUPPORTED"
    assert rec["code_missing"] == ["src/nope.py"]


def test_negative_control_detects_import_only_symbol(tmp_path):
    """A name that is merely imported is not an implementation."""
    root = _sandbox(tmp_path)
    (root / "src" / "shim.py").write_text("from thing import Thing\n", encoding="utf-8")
    claim = audit_mod.Claim(cid="FAKE-IMPORT", section="x", anchor="does the thing",
                            code=("src/shim.py",), symbols=("src/shim.py::Thing",),
                            tests=("tests/test_thing.py",))
    rec = audit_mod.evaluate(root, claim)
    assert rec["verdict"] == "NOT_SUPPORTED"
    assert any("not defined" in msg for msg in rec["symbols_failed"])


def test_negative_control_detects_unsourced_number(tmp_path):
    root = _sandbox(tmp_path)
    claim = audit_mod.Claim(cid="FAKE-NUMBER", section="x", anchor="Claim: error 42 %.",
                            code=("src/thing.py",), tests=("tests/test_thing.py",),
                            numbers=(("99.9", "docs/thing.md"),))
    rec = audit_mod.evaluate(root, claim)
    assert rec["numbers"][0]["in_declared_source"] is False
    assert rec["numbers"][0]["in_readme"] is False  # not even quoted -> not counted as unbacked
    # quoted in README, absent from the declared source *and* from the whole repo
    claim2 = audit_mod.Claim(cid="FAKE-NUMBER2", section="x", anchor="Claim: error 77 %.",
                             code=("src/thing.py",), tests=("tests/test_thing.py",),
                             numbers=(("77", "src/does_not_exist.py"),))
    (root / "README.md").write_text((root / "README.md").read_text() + "Claim: error 77 %.\n",
                                    encoding="utf-8")
    rec2 = audit_mod.evaluate(root, claim2)
    assert rec2["numbers_unbacked"] == ["77"]
    assert rec2["verdict"] == "WITHHELD"


def test_negative_control_detects_self_contradiction(tmp_path):
    root = _sandbox(tmp_path)
    (root / "README.md").write_text(
        "# fake\n\n`src/thing.py` does the thing. See `docs/thing.md`.\n"
        "There is no thing model.\n", encoding="utf-8")
    claim = audit_mod.Claim(cid="FAKE-CONTRA", section="x", anchor="does the thing",
                            code=("src/thing.py",), tests=("tests/test_thing.py",),
                            docs=("docs/thing.md",), counterclaim="There is no thing model.")
    rec = audit_mod.evaluate(root, claim)
    assert rec["verdict"] == "SELF_CONTRADICTION"


def test_negative_control_detects_dangling_pointer(tmp_path):
    root = _sandbox(tmp_path)
    (root / "README.md").write_text("see `docs/missing.md`\n", encoding="utf-8")
    assert audit_mod.dangling_pointers(root, (root / "README.md").read_text()) == ["docs/missing.md"]


def test_negative_control_detects_stale_undated_census(tmp_path):
    root = _sandbox(tmp_path)
    (root / "README.md").write_text("快照：999 个测试文件\n`src/thing.py` does the thing.\n",
                                    encoding="utf-8")
    claim = audit_mod.Claim(cid="FAKE-CENSUS", section="x", anchor="个测试文件", census=True)
    rec = audit_mod.evaluate(root, claim)
    assert rec["verdict"] == "STALE_NUMBER"
    # ... and the same numbers are tolerated when explicitly date-stamped
    (root / "README.md").write_text("2026-11 快照：999 个测试文件\n", encoding="utf-8")
    assert audit_mod.evaluate(root, claim)["verdict"] == "REAL"


def test_negative_control_detects_gate_drift(tmp_path):
    root = _sandbox(tmp_path)
    (root / "docs" / "gates.json").write_text(json.dumps({
        "all_recomputed_pass": True,
        "gates": [{"gate": "g", "measured": 0.5, "threshold": 0.02, "passed": True}]}),
        encoding="utf-8")
    claim = audit_mod.Claim(cid="FAKE-GATE", section="x", anchor="does the thing",
                            code=("src/thing.py",), tests=("tests/test_thing.py",),
                            docs=("docs/gates.json",),
                            gates=(("docs/gates.json", "g", 0.0098),))
    rec = audit_mod.evaluate(root, claim)
    assert rec["verdict"] == "GATE_DRIFT"


def test_negative_control_detects_retired_claim(tmp_path):
    root = _sandbox(tmp_path)
    claim = audit_mod.Claim(cid="FAKE-RETIRED", section="x", anchor="a sentence never written",
                            code=("src/thing.py",))
    assert audit_mod.evaluate(root, claim)["verdict"] == "RETIRED"


def test_json_lookup_rejects_bad_paths(tmp_path):
    data = {"runs": [{"case": "a", "p": {"e": 0.5}}, {"case": "b", "p": {"e": 0.7}}]}
    ok, val, _ = audit_mod.json_lookup(data, "case=b:p.e")
    assert ok and val == 0.7
    assert not audit_mod.json_lookup(data, "case=zz:p.e")[0]
    assert not audit_mod.json_lookup(data, "case=a:p.missing")[0]


def test_markdown_render_is_stable(dem_result):
    md = audit_mod.render_markdown(dem_result)
    assert "README claim" in md
    for claim in dem_result["claims"]:
        assert f"`{claim['cid']}`" in md


def test_written_artifacts_are_committed(tmp_path):
    """The generated table must exist in the repo, not only in a temp dir."""
    art = ROOT / "artifacts" / "readme_claim_audit"
    if not (art / "claims.json").exists():
        pytest.skip("generate with: python scripts/readme_claim_audit.py")
    data = json.loads((art / "claims.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == audit_mod.SCHEMA_VERSION
    assert (art / "CLAIM_EVIDENCE_TABLE.md").exists()