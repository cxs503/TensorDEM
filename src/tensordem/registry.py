"""TensorDEM capability registry (dual-pilot spec, m23186/m23215 contract).

Microservice-style self-registration: importing :mod:`tensordem.capabilities`
registers every capability into the :data:`REGISTRY` singleton, the same way
FastAPI route decorators register routes at import time.

Dump CLI (two modes, identical JSON contract on stdout)::

    python -m tensordem.registry            # import mode (needs the package env)
    python -m tensordem.registry --static   # AST scan: no package import, no torch

Both modes emit exactly::

    {"source_commit": "<git HEAD or null>", "capabilities": [ ... ]}

The ``--static`` mode is the one the platform capability_sync daemon calls
(it must work on a bare checkout without torch).  The two modes must agree
on the capability set (enforced by ``tests/test_capability_registry.py``).

Anti-overclaim gates (client-side mirror of the platform 422 gate on
``POST /api/v1/tensor/capabilities/sync``):

* ``maturity == "mature"`` requires ``verified and wired``;
* a non-null ``endpoint`` (TensorPhysics callable) requires ``wired``.

``verified`` may not be self-declared: the generator
(``scripts/gen_dem_capabilities.py``) only sets it when a machine-checkable
reference-comparison artifact exists (``*_verification.py`` module, root
``verdict_*.json``, or an xval/verification test binding).
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_NAME = "TensorDEM"

VALID_MATURITY = ("mature", "developing", "planned")


class OverclaimError(ValueError):
    """Raised when a registration violates the anti-overclaim gates."""


@dataclass
class Capability:
    id: str
    kind: str = "dem"
    maturity: str = "developing"
    module_path: str = ""
    entry: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    wired: bool = False
    tested: bool = False
    verified: bool = False
    test_path: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    endpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "maturity": self.maturity,
            "module_path": self.module_path,
            "entry": self.entry,
            "params": self.params,
            "wired": self.wired,
            "tested": self.tested,
            "verified": self.verified,
            "test_path": self.test_path,
            "evidence": self.evidence,
            "endpoint": self.endpoint,
        }


class Registry:
    """Singleton capability registry with anti-overclaim gates."""

    def __init__(self) -> None:
        self._items: dict[str, Capability] = {}

    def register(self, cap: Capability) -> Capability:
        if cap.maturity not in VALID_MATURITY:
            raise OverclaimError(
                f"capability {cap.id}: maturity {cap.maturity!r} not in {VALID_MATURITY}"
            )
        if cap.maturity == "mature" and not (cap.verified and cap.wired):
            raise OverclaimError(
                f"capability {cap.id}: maturity=mature requires verified+wired "
                "machine evidence (anti-overclaim)"
            )
        if cap.endpoint and not cap.wired:
            raise OverclaimError(
                f"capability {cap.id}: endpoint (TensorPhysics callable) requires wired=true"
            )
        if not cap.module_path:
            raise OverclaimError(f"capability {cap.id}: module_path is required")
        self._items[cap.id] = cap
        return cap

    def get(self, cap_id: str) -> Capability | None:
        return self._items.get(cap_id)

    def all(self) -> list[Capability]:
        return sorted(self._items.values(), key=lambda c: c.id)

    def clear(self) -> None:
        self._items.clear()

    def dump(self, source_commit: str | None) -> dict[str, Any]:
        return {
            "source_commit": source_commit,
            "capabilities": [c.to_dict() for c in self.all()],
        }


REGISTRY = Registry()


def capability(
    id: str,
    kind: str = "dem",
    maturity: str = "developing",
    module_path: str = "",
    entry: str | None = None,
    params: dict[str, Any] | None = None,
    wired: bool = False,
    tested: bool = False,
    verified: bool = False,
    test_path: str | None = None,
    evidence: dict[str, Any] | None = None,
    endpoint: str | None = None,
):
    """Decorator: register the decorated object as a capability at import time.

    The decorated object is returned unchanged; the metadata is the contract.
    """

    def decorator(obj: Any) -> Any:
        REGISTRY.register(
            Capability(
                id=id,
                kind=kind,
                maturity=maturity,
                module_path=module_path,
                entry=entry,
                params=dict(params or {}),
                wired=wired,
                tested=tested,
                verified=verified,
                test_path=test_path,
                evidence=dict(evidence or {}),
                endpoint=endpoint,
            )
        )
        return obj

    return decorator


# ---------------------------------------------------------------------------
# static (AST) scan — no package import, no torch
# ---------------------------------------------------------------------------

def _decorator_is_capability(node: ast.AST) -> bool:
    dec = node.decorator_list
    for d in dec:
        if isinstance(d, ast.Call):
            f = d.func
            if isinstance(f, ast.Name) and f.id == "capability":
                return True
            if isinstance(f, ast.Attribute) and f.attr == "capability":
                return True
    return False


def _literal_or_raw(value: ast.expr) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        try:
            return ast.unparse(value)
        except Exception:
            return None


def static_scan(package_dir: Path | None = None) -> list[dict[str, Any]]:
    """AST-scan the package source tree for ``@capability(...)`` decorators.

    Never imports the package (no torch required).  Returns capability dicts
    in the same shape as :meth:`Capability.to_dict`.
    """
    if package_dir is None:
        package_dir = Path(__file__).resolve().parent
    out: list[dict[str, Any]] = []
    for path in sorted(package_dir.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _decorator_is_capability(node):
                continue
            for d in node.decorator_list:
                if not isinstance(d, ast.Call):
                    continue
                f = d.func
                if not ((isinstance(f, ast.Name) and f.id == "capability")
                        or (isinstance(f, ast.Attribute) and f.attr == "capability")):
                    continue
                kw: dict[str, Any] = {}
                for k in d.keywords:
                    if k.arg is None:
                        continue
                    kw[k.arg] = _literal_or_raw(k.value)
                if "id" not in kw:
                    continue
                kw.setdefault("kind", "dem")
                kw.setdefault("maturity", "developing")
                kw.setdefault("module_path", "")
                kw.setdefault("entry", None)
                kw.setdefault("params", {})
                kw.setdefault("wired", False)
                kw.setdefault("tested", False)
                kw.setdefault("verified", False)
                kw.setdefault("test_path", None)
                kw.setdefault("evidence", {})
                kw.setdefault("endpoint", None)
                out.append(kw)
    return out


def _repo_root() -> Path | None:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return parent
    return None


def source_commit() -> str | None:
    root = _repo_root()
    if root is None:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def dump(static: bool = False) -> dict[str, Any]:
    if static:
        caps = static_scan()
    else:
        import tensordem.capabilities  # noqa: F401  (triggers registration)
        caps = [c.to_dict() for c in REGISTRY.all()]
    return {"source_commit": source_commit(), "capabilities": caps}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tensordem.registry",
        description="Dump the TensorDEM capability registry as JSON on stdout.",
    )
    parser.add_argument(
        "--static", action="store_true",
        help="AST-scan the source tree (no package import, no torch)",
    )
    args = parser.parse_args(argv)
    payload = dump(static=args.static)
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
