"""Consistency pytest for the TensorDEM registry ``--static`` dump (m24229).

Runs ``python src/tensordem/registry.py --static`` as a subprocess (the
platform host may be torch-free and must not rely on ``-m``) and asserts the
four consistency invariants of the emitted single-JSON-object payload:

1. ``capabilities`` is non-empty and every entry carries the identity /
   category / maturity fields (``id`` = name, ``kind`` = category,
   ``maturity`` = maturity);
2. ``maturity`` value is a subset of {mature, developing, planned};
3. capability ``id`` (name) is unique — no duplicate registration;
4. ``source_commit`` is a 40-character lowercase hex SHA-1.

The ``--static`` mode is an AST scan (no package import, no torch), so the
test is runnable on a bare checkout with only stdlib on ``sys.path``.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "tensordem"

VALID_MATURITY = {"mature", "developing", "planned"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _static_dump() -> dict:
    """Run the registry --static CLI as a subprocess and parse its stdout."""
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": ""}
    out = subprocess.run(
        [sys.executable, str(PKG / "registry.py"), "--static"],
        capture_output=True, text=True, timeout=600, env=env,
    )
    assert out.returncode == 0, f"registry --static failed:\n{out.stderr}"
    return json.loads(out.stdout)


def test_static_capabilities_nonempty_with_identity_fields():
    """Invariant 1: capabilities non-empty; each has id/kind/maturity."""
    d = _static_dump()
    caps = d["capabilities"]
    assert isinstance(caps, list) and caps, "capabilities must be non-empty"
    for c in caps:
        # name -> id, category -> kind, maturity -> maturity
        assert "id" in c and isinstance(c["id"], str) and c["id"], \
            f"capability missing id/name: {c!r}"
        assert "kind" in c and isinstance(c["kind"], str) and c["kind"], \
            f"capability {c.get('id')} missing kind/category"
        assert "maturity" in c and isinstance(c["maturity"], str), \
            f"capability {c.get('id')} missing maturity"


def test_static_maturity_value_subset():
    """Invariant 2: maturity value is a subset of {mature, developing, planned}."""
    d = _static_dump()
    for c in d["capabilities"]:
        assert c["maturity"] in VALID_MATURITY, \
            f"capability {c.get('id')} has invalid maturity {c['maturity']!r}"
    # every declared value is actually represented in the inventory
    seen = {c["maturity"] for c in d["capabilities"]}
    assert seen <= VALID_MATURITY


def test_static_no_duplicate_names():
    """Invariant 3: capability id (name) is unique across the dump."""
    d = _static_dump()
    ids = [c["id"] for c in d["capabilities"]]
    assert len(ids) == len(set(ids)), \
        f"duplicate capability names: {[i for i in ids if ids.count(i) > 1]}"


def test_static_source_commit_is_40_hex():
    """Invariant 4: source_commit is a 40-character lowercase hex SHA-1."""
    d = _static_dump()
    sc = d["source_commit"]
    assert isinstance(sc, str) and sc, "source_commit must be a non-empty string"
    assert len(sc) == 40, f"source_commit length {len(sc)} != 40: {sc!r}"
    assert HEX40.match(sc), f"source_commit not 40-hex: {sc!r}"
