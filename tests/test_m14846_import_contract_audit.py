"""Gate: static import-contract audit must stay calibrated (m14846).

The audit (``scripts/m14846_import_contract_audit.py``) is a *triage* tool: it
finds test-module imports whose names are not statically exported, which is the
defect class that makes a merged verification module impossible to collect
(package directory shadowing a same-named module file, m5754/m14450).

Two guards are asserted here:

1. **Shadow guard** — the set of modules where ``x.py`` and ``x/`` coexist must
   not grow.  A new shadow silently deletes a module's public API, so the test
   fails on any addition beyond the documented baseline.
2. **Detection guard** — as long as ``tensordem.ice_structure_contact`` does not
   export ``ContactPressureField`` at runtime, the static audit must report
   exactly that contract as broken.  The guard is *conditional on the runtime
   state* so it keeps working after the upstream fix (m14450) lands instead of
   turning into a false failure.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import m14846_import_contract_audit as audit_mod  # noqa: E402

# modules that already carry the file/package shadow on this trunk; a new entry
# means a newly broken (un-collectable) verification module
KNOWN_SHADOWS = {"tensordem.brash_ice_rheology", "tensordem.ice_structure_contact"}


@pytest.fixture(scope="module")
def result():
    return audit_mod.audit()


def test_schema_and_scale(result):
    assert result["protocol"] == "tensordem-import-contract-audit-v1"
    assert result["n_test_modules"] > 400, "audit must actually see the test tree"
    assert isinstance(result["broken"], list)


def test_no_new_shadowed_modules(result):
    found = set(result["shadowed_modules"])
    new = found - KNOWN_SHADOWS
    assert not new, f"new file/package shadow deletes public API: {sorted(new)}"


def test_ice_structure_contact_defect_is_detected_while_it_exists(result):
    try:
        pkg = importlib.import_module("tensordem.ice_structure_contact")
    except ImportError:  # pragma: no cover - whole package missing
        pytest.skip("package itself unavailable")
    if hasattr(pkg, "ContactPressureField"):
        pytest.skip("defect already fixed upstream (m14450); nothing to detect")

    flagged = {r["test"] for r in result["broken"]
               if r["module"] == "tensordem.ice_structure_contact"}
    assert "test_ice_structure_contact.py" in flagged
    assert any("ContactPressureField" in r["missing"]
               for r in result["broken"]
               if r["module"] == "tensordem.ice_structure_contact")


def test_known_dynamic_exports_are_the_only_ice_steel_false_positives(result):
    """Calibration note codified: ice_steel_friction exports its anchor tables
    dynamically, so the static scan lists it even though runtime import works."""
    mod = importlib.import_module("tensordem.ice_steel_friction")
    assert hasattr(mod, "CALIBRATION_ANCHORS"), "runtime export disappeared"
    static = [r for r in result["broken"] if r["module"] == "tensordem.ice_steel_friction"]
    # the static scan may flag it (documented limitation) but it must stay scoped
    # to the anchor-table names, never to the module being absent
    assert all(not r["missing_module"] for r in static)


def test_markdown_render_is_complete(result):
    md = audit_mod.to_markdown(result)
    assert result["protocol"] in md
    assert "局限" in md
    for r in result["broken"]:
        assert r["test"] in md