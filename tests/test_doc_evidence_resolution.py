"""Regression gate for :mod:`scripts.audit_doc_evidence` pointer resolution.

Why this test exists
--------------------
The doc-evidence ledger is only useful as a *debt list* if every entry it calls
``code:MISSING`` is genuinely absent.  Before the fix exercised here, 61 of the
88 dangling ``.py`` pointers were false positives: documentation cites Python
modules by their **import path** (``tensordem/freeze_bond.py``) or by a
package-relative path (``freeze_bond/__init__.py``, ``contact_physics/...``),
while this repository keeps them under ``src/tensordem/``; modules that were
later promoted to packages were still cited by their old ``module.py`` path.
Those forms resolved nowhere, so real debt was buried in noise.

The test pins the three resolution rules and puts a ceiling on the number of
dangling code pointers so that new documentation cannot silently outrun the
code it claims to evidence.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Debt ceiling measured on 2026-11-12 after the resolution fix (see
# docs/DOC_EVIDENCE_POINTER_RESOLUTION_AUDIT.md for the triage of every entry
# that is still counted).  Lower it when pointers are repaired; never raise it
# without re-running the ledger and explaining the new debt in that document.
CODE_MISSING_CEILING = 27


def _audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_doc_evidence", ROOT / "scripts" / "audit_doc_evidence.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return _audit_module()


def test_import_path_shorthand_resolves(audit):
    """`tensordem/x.py` is an import path; the file is `src/tensordem/x.py`."""
    status, resolved = audit.classify("tensordem/ice_creep.py", "docs/x.md", audit.git_tracked())
    assert status == "TRACKED"
    assert resolved == "src/tensordem/ice_creep.py"


def test_package_shorthand_resolves(audit):
    """A package-internal file is cited relative to `tensordem/`."""
    status, resolved = audit.classify(
        "contact_physics/__init__.py", "docs/x.md", audit.git_tracked()
    )
    assert status == "TRACKED"
    assert resolved == "src/tensordem/contact_physics/__init__.py"


def test_module_promoted_to_package_resolves(audit):
    """`src/tensordem/freeze_bond.py` is now the package `freeze_bond/__init__.py`."""
    status, resolved = audit.classify(
        "src/tensordem/freeze_bond.py", "docs/x.md", audit.git_tracked()
    )
    assert status == "TRACKED"
    assert resolved == "src/tensordem/freeze_bond/__init__.py"


def test_resolution_never_invents_paths(audit):
    """A pointer that exists nowhere stays MISSING under every new candidate."""
    status, resolved = audit.classify(
        "src/tensordem/definitely_not_a_module.py", "docs/x.md", audit.git_tracked()
    )
    assert status == "MISSING"
    assert resolved.endswith("definitely_not_a_module.py")


def test_dangling_code_pointer_budget(audit):
    """The committed ledger must not have grown its dangling-code backlog."""
    report = audit.scan()
    counts = report["counts"]
    dangling = counts.get("code:MISSING", 0)
    assert dangling <= CODE_MISSING_CEILING, (
        f"{dangling} dangling .py pointers (> ceiling {CODE_MISSING_CEILING}); "
        "add the producer/file or justify the debt in "
        "docs/DOC_EVIDENCE_POINTER_RESOLUTION_AUDIT.md"
    )
    # the fix must actually be in force: the pre-fix count was 88
    assert dangling < 88