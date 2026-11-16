"""Tests for the TensorDEM capability registry (dual-pilot spec m23186/m23215).

Covers: decorator self-registration, the anti-overclaim gates (mature /
planned / endpoint / verified), pointer resolution (missing module/entry/
test = problem, never a silent drop), the single-JSON-object stdout contract
(source_commit + capabilities + summary + problems), static-vs-import parity
(in-process and via the ``--parity`` CLI), direct file-path invocation on a
torch-free host, inventory completeness, and the platform sync payload shape.
"""
from __future__ import annotations

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
    ENDPOINT_PREFIXES,
    OverclaimError,
    REGISTRY,
    VERIFIED_LEVELS,
    capability,
    check_gates,
    resolve_pointers,
    static_scan,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    REGISTRY.clear()
    yield
    # restore the real inventory for other tests (reload re-runs the
    # decorators; a plain import would be a no-op once the module is cached).
    # Clear first: the duplicate-registration gate forbids re-registering ids
    # that are still present.
    import importlib
    import tensordem.capabilities
    REGISTRY.clear()
    importlib.reload(tensordem.capabilities)


# ---------------------------------------------------------------- decorator

def test_decorator_registers_into_singleton():
    @capability(id="dem.test_fixture", module_path="tensordem.dem",
                entry="IceDEM", wired=True, tested=True,
                verified=True, evidence={"level": "ANALYTIC"})
    class _X:
        pass

    cap = REGISTRY.get("dem.test_fixture")
    assert cap is not None
    assert cap.entry == "IceDEM"
    assert cap.maturity == "developing"  # default, not self-declared mature
    assert _X is not None  # object returned unchanged


def test_duplicate_registration_rejected():
    @capability(id="dem.dup")
    class _A:
        pass
    with pytest.raises(OverclaimError, match="duplicate"):
        @capability(id="dem.dup")
        class _B:
            pass


# ---------------------------------------------------------------- gates

def test_mature_requires_verified_and_wired():
    with pytest.raises(OverclaimError, match="mature"):
        @capability(id="dem.bad1", module_path="tensordem.dem",
                    maturity="mature", verified=True, wired=False,
                    evidence={"level": "ANALYTIC"})
        class _A:
            pass
    with pytest.raises(OverclaimError, match="mature"):
        @capability(id="dem.bad2", module_path="tensordem.dem",
                    maturity="mature", verified=False, wired=True)
        class _B:
            pass


def test_planned_requires_feature_ledger_anchor():
    with pytest.raises(OverclaimError, match="feature_ledger"):
        @capability(id="dem.bad3", maturity="planned")
        class _A:
            pass
    with pytest.raises(OverclaimError, match="feature_ledger"):
        @capability(id="dem.bad4", maturity="planned",
                    evidence={"feature_ledger": "   "})
        class _B:
            pass
    @capability(id="dem.planned_ok", maturity="planned",
                evidence={"feature_ledger": "CAP-REG-DEM"})
    class _C:
        pass
    assert REGISTRY.get("dem.planned_ok").maturity == "planned"


def test_endpoint_requires_wired_and_prefix():
    with pytest.raises(OverclaimError, match="wired"):
        @capability(id="dem.bad5", module_path="tensordem.dem",
                    endpoint="cli:python -m tensordem dem-2d", wired=False)
        class _A:
            pass
    with pytest.raises(OverclaimError, match="prefix"):
        @capability(id="dem.bad6", module_path="tensordem.dem",
                    endpoint="python -m tensordem dem-2d", wired=True)
        class _B:
            pass
    for prefix in ENDPOINT_PREFIXES:
        @capability(id=f"dem.ep_{prefix.rstrip(':')}", module_path="tensordem.dem",
                    endpoint=f"{prefix}x", wired=True)
        class _C:
            pass
        assert REGISTRY.get(f"dem.ep_{prefix.rstrip(':')}") is not None


def test_verified_requires_analytic_or_xvalue_level():
    with pytest.raises(OverclaimError, match="level"):
        @capability(id="dem.bad7", module_path="tensordem.dem",
                    verified=True, wired=True,
                    evidence={"grade": "A", "note": "self-declared"})
        class _A:
            pass
    with pytest.raises(OverclaimError, match="level"):
        @capability(id="dem.bad8", module_path="tensordem.dem",
                    verified=True, wired=True,
                    evidence={"level": "FEELING"})
        class _B:
            pass
    for level in VERIFIED_LEVELS:
        @capability(id=f"dem.ver_{level}", module_path="tensordem.dem",
                    verified=True, wired=True, evidence={"level": level})
        class _C:
            pass
        assert REGISTRY.get(f"dem.ver_{level}").verified


def test_mature_ok_with_full_evidence():
    @capability(id="dem.good", module_path="tensordem.dem",
                 maturity="mature", wired=True, tested=True, verified=True,
                 evidence={"level": "ANALYTIC"})
    class _D:
        pass
    assert REGISTRY.get("dem.good").maturity == "mature"


# ---------------------------------------------------------------- pointers

def test_pointer_missing_module_entry_test_are_problems():
    caps = [
        {"id": "dem.p_missing_mod", "maturity": "developing",
         "module_path": "tensordem.no_such_module", "entry": "X",
         "wired": False, "tested": False, "verified": False,
         "test_path": None, "evidence": {}, "endpoint": None},
        {"id": "dem.p_missing_entry", "maturity": "developing",
         "module_path": "tensordem.dem", "entry": "NoSuchSymbol",
         "wired": False, "tested": False, "verified": False,
         "test_path": None, "evidence": {}, "endpoint": None},
        {"id": "dem.p_missing_test", "maturity": "developing",
         "module_path": "tensordem.dem", "entry": "IceDEM",
         "wired": True, "tested": True, "verified": False,
         "test_path": "tests/test_no_such_file.py",
         "evidence": {}, "endpoint": None},
        {"id": "dem.p_ok", "maturity": "developing",
         "module_path": "tensordem.dem", "entry": "IceDEM",
         "wired": True, "tested": False, "verified": False,
         "test_path": None, "evidence": {}, "endpoint": None},
    ]
    problems = resolve_pointers(caps, pkg_root=PKG, repo_root=ROOT)
    codes = {(p["capability"], p["code"]) for p in problems}
    assert ("dem.p_missing_mod", "pointer.module_missing") in codes
    assert ("dem.p_missing_entry", "pointer.entry_missing") in codes
    assert ("dem.p_missing_test", "pointer.test_missing") in codes
    assert not any(c == "dem.p_ok" for c, _ in codes)


def test_check_gates_reports_violations_as_problems():
    caps = [
        {"id": "dem.g1", "maturity": "mature", "wired": True,
         "verified": False, "evidence": {}, "endpoint": None},
        {"id": "dem.g2", "maturity": "planned", "wired": False,
         "verified": False, "evidence": {}, "endpoint": None},
        {"id": "dem.g3", "maturity": "developing", "wired": False,
         "verified": False, "evidence": {}, "endpoint": "api:/x"},
        {"id": "dem.g4", "maturity": "developing", "wired": True,
         "verified": True, "evidence": {"grade": "A"}, "endpoint": None},
    ]
    codes = {p["code"] for p in check_gates(caps)}
    assert "gate.mature_requires_verified_wired" in codes
    assert "gate.planned_requires_feature_ledger" in codes
    assert "gate.endpoint_requires_wired" in codes
    assert "gate.verified_requires_analytic_xvalue" in codes


# ---------------------------------------------------------------- dump CLI

def _run_cli(args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": ""}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(PKG / "registry.py"), *args],
        capture_output=True, text=True, timeout=600, env=env,
    )


def test_stdout_is_single_json_object_with_contract_keys():
    for args in (["--static"], []):
        out = _run_cli(args)
        assert out.returncode == 0, out.stderr
        d = json.loads(out.stdout)  # exactly one JSON object on stdout
        assert set(d) == {"source_commit", "capabilities", "summary", "problems"}
        assert isinstance(d["capabilities"], list) and d["capabilities"]
        s = d["summary"]
        assert s["total"] == len(d["capabilities"])
        assert s["problems"] == len(d["problems"])
        assert sum(s["by_maturity"].values()) == s["total"]
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
    module file directly (no tensordem on sys.path, no torch)."""
    out = _run_cli(["--static"])
    assert out.returncode == 0, out.stderr
    d = json.loads(out.stdout)
    assert d["capabilities"]


def test_parity_cli_mode():
    out = _run_cli(["--parity"])
    assert out.returncode == 0, out.stderr
    d = json.loads(out.stdout)
    assert d["parity"] is True
    assert d["static_count"] == d["import_count"] > 0
    assert d["differences"] == []


def test_static_import_parity_in_process():
    import importlib
    import tensordem.capabilities
    import tensordem.registry as reg_mod
    REGISTRY.clear()
    importlib.reload(tensordem.capabilities)  # re-run the decorators
    imp = {c.id: c.to_dict() for c in reg_mod.REGISTRY.all()}
    st = {c["id"]: c for c in static_scan(PKG)}
    assert set(imp) == set(st)
    for cid in imp:
        assert imp[cid] == st[cid], f"parity mismatch for {cid}"


# ---------------------------------------------------------------- inventory

def _caps() -> list[dict]:
    return static_scan(PKG)


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
            assert c["endpoint"].startswith(ENDPOINT_PREFIXES), c["id"]
        if c["test_path"]:
            assert (ROOT / c["test_path"]).exists(), c["test_path"]
        if c["wired"]:
            assert c["entry"], c["id"]
        if c["evidence"]["grade"] == "A":
            assert c["verified"], c["id"]
        if c["evidence"]["grade"] == "B":
            assert c["tested"] and not c["verified"], c["id"]
        if c["verified"]:
            assert c["evidence"].get("level") in VERIFIED_LEVELS, c["id"]


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
    d = json.loads(_run_cli(["--static"]).stdout)
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
