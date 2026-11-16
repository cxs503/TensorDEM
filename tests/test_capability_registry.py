"""Tests for the TensorDEM capability registry (dual-pilot spec m23186/m23215).

Covers: decorator self-registration, anti-overclaim gates, the static AST
dump contract (no package import / no torch), static-vs-import parity,
inventory completeness, and the platform sync payload shape.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PKG = SRC / "tensordem"

sys.path.insert(0, str(SRC))

from tensordem import registry  # noqa: E402
from tensordem.registry import (  # noqa: E402
    Capability,
    OverclaimError,
    REGISTRY,
    capability,
    static_scan,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    REGISTRY.clear()
    yield
    # restore the real inventory for other tests (reload re-runs the
    # decorators; a plain import would be a no-op once the module is cached)
    import importlib
    import tensordem.capabilities
    importlib.reload(tensordem.capabilities)


# ---------------------------------------------------------------- decorator

def test_decorator_registers_into_singleton():
    @capability(id="dem.test_fixture", module_path="tensordem.dem",
                entry="IceDEM", wired=True, tested=True, verified=True)
    class _X:
        pass

    cap = REGISTRY.get("dem.test_fixture")
    assert cap is not None
    assert cap.entry == "IceDEM"
    assert cap.maturity == "developing"  # default, not self-declared mature
    assert _X is not None  # object returned unchanged


def test_mature_requires_verified_and_wired():
    with pytest.raises(OverclaimError):
        @capability(id="dem.bad1", module_path="tensordem.dem",
                    maturity="mature", verified=True, wired=False)
        class _A:
            pass
    with pytest.raises(OverclaimError):
        @capability(id="dem.bad2", module_path="tensordem.dem",
                    maturity="mature", verified=False, wired=True)
        class _B:
            pass


def test_endpoint_requires_wired():
    with pytest.raises(OverclaimError):
        @capability(id="dem.bad3", module_path="tensordem.dem",
                    endpoint="cli:python -m tensordem dem-2d", wired=False)
        class _C:
            pass


def test_mature_ok_with_full_evidence():
    @capability(id="dem.good", module_path="tensordem.dem",
                maturity="mature", wired=True, tested=True, verified=True)
    class _D:
        pass
    assert REGISTRY.get("dem.good").maturity == "mature"


# ---------------------------------------------------------------- static dump

def _static_dump() -> dict:
    caps = static_scan(PKG)
    return {"source_commit": registry.source_commit(), "capabilities": caps}


def test_static_dump_contract():
    d = _static_dump()
    assert set(d) == {"source_commit", "capabilities"}
    assert isinstance(d["capabilities"], list) and d["capabilities"]
    for c in d["capabilities"]:
        for k in ("id", "kind", "maturity", "module_path", "entry", "params",
                  "wired", "tested", "verified", "test_path", "evidence",
                  "endpoint"):
            assert k in c, f"missing field {k} in {c.get('id')}"
        assert c["maturity"] in ("mature", "developing", "planned")
        assert isinstance(c["wired"], bool)
        assert isinstance(c["tested"], bool)
        assert isinstance(c["verified"], bool)
        assert c["evidence"].get("grade") in ("A", "B", "C")


def test_static_mode_does_not_import_package():
    """--static must work without the package being importable: run the
    module file directly (no tensordem on sys.path)."""
    code = (
        "import runpy, sys; "
        "sys.argv = ['registry.py', '--static']; "
        "runpy.run_path(r'%s', run_name='__main__')" % (PKG / "registry.py")
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=300,
                         env={"PATH": "/usr/bin:/bin", "PYTHONPATH": ""})
    assert out.returncode == 0, out.stderr
    d = json.loads(out.stdout)
    assert d["capabilities"]


def test_static_import_parity():
    import tensordem.capabilities  # noqa: F401
    imp = {c["id"]: c.to_dict() for c in REGISTRY.all()}
    st = {c["id"]: c for c in static_scan(PKG)}
    assert set(imp) == set(st)
    for i in imp:
        assert imp[i] == st[i], f"parity mismatch for {i}"


# ---------------------------------------------------------------- inventory

def _caps() -> list[dict]:
    return _static_dump()["capabilities"]


def test_inventory_covers_all_subpackages_and_modules():
    caps = {c["id"]: c for c in _caps()}
    subpkgs = sorted(
        d.name for d in PKG.iterdir()
        if d.is_dir() and (d / "__init__.py").exists()
    )
    infra = {"__init__", "__main__", "cli", "registry", "capabilities"}
    tops = sorted(p.stem for p in PKG.glob("*.py") if p.stem not in infra)
    for name in subpkgs:
        assert f"dem.{name}" in caps, f"subpackage {name} not registered"
    for name in tops:
        assert (f"dem.{name}" in caps) or (f"dem.{name}_mod" in caps), \
            f"module {name} not registered"


def test_flags_are_honest():
    for c in _caps():
        if c["maturity"] == "mature":
            assert c["verified"] and c["wired"] and c["tested"], c["id"]
        if c["endpoint"]:
            assert c["wired"], c["id"]
        if c["test_path"]:
            assert (ROOT / c["test_path"]).exists(), c["test_path"]
        if c["wired"]:
            assert c["entry"], c["id"]
        if c["evidence"]["grade"] == "A":
            assert c["verified"], c["id"]
        if c["evidence"]["grade"] == "B":
            assert c["tested"] and not c["verified"], c["id"]


def test_no_planned_without_ledger_anchor():
    # TensorDEM has no feature_ledger 'planned' anchors -> no planned entries
    assert all(c["maturity"] != "planned" for c in _caps())


def test_endpoints_are_the_real_cli_subcommands():
    eps = {c["endpoint"] for c in _caps() if c["endpoint"]}
    assert eps == {
        "cli:python -m tensordem dem-2d",
        "cli:python -m tensordem verify-friction-polyhedron",
        "cli:python -m tensordem elastic-limit-shear",
        "cli:python -m tensordem h1-broken-ice",
        "cli:python -m tensordem binding-contract",
    }


def test_sync_payload_shape():
    """The dump must be POST-able to POST /tensor/capabilities/sync."""
    d = _static_dump()
    body = {"repo": "TensorDEM", "source_commit": d["source_commit"],
            "capabilities": d["capabilities"]}
    assert 0 < len(body["capabilities"]) <= 2000
    for c in body["capabilities"]:
        assert 2 <= len(c["id"]) <= 160
        assert len(c["module_path"]) <= 240
        if c["endpoint"]:
            assert len(c["endpoint"]) <= 240
        if c["maturity"] == "mature":
            assert c["verified"] and c["wired"]
        if c["endpoint"]:
            assert c["wired"]
