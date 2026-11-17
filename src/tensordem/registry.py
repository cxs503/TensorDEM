"""TensorDEM capability registry (dual-pilot spec, m23186/m23215 contract).

Microservice-style self-registration: importing :mod:`tensordem.capabilities`
registers every capability into the :data:`REGISTRY` singleton, the same way
FastAPI route decorators register routes at import time.

Dump CLI (three modes, single JSON object on stdout)::

    python -m tensordem.registry            # import mode (needs the package env)
    python -m tensordem.registry --static   # AST scan: no package import, no torch
    python -m tensordem.registry --parity   # static vs import, field-by-field

Direct file-path invocation is supported (the platform host may be torch-free
and must not rely on ``-m``)::

    python src/tensordem/registry.py --static

All modes emit exactly one JSON object on stdout::

    {"source_commit": ..., "capabilities": [...], "summary": {...},
     "problems": [...]}

``--parity`` emits ``{"source_commit", "parity", "static_count",
"import_count", "differences", "problems"}`` and exits non-zero on mismatch.

``problems`` is the honest-failure channel: a pointer that does not resolve
(module file, entry symbol, test file) is reported here, never silently
dropped.  Gate violations are reported the same way in static mode; in
import mode they raise :class:`OverclaimError` at registration time.

Anti-overclaim gates (client-side mirror of the platform 422 gate on
``POST /api/v1/tensor/capabilities/sync``):

* ``maturity == "mature"``   requires ``verified and wired``;
* ``maturity == "planned"``  requires an ``evidence.feature_ledger`` anchor;
* a non-null ``endpoint``    requires ``wired`` and a ``cli:`` / ``route:`` /
  ``api:`` prefix;
* ``verified == True``       requires ``evidence.level`` in
  ``ANALYTIC`` / ``XVALUE`` (self-declared verification is not accepted).

The generator (``scripts/gen_dem_capabilities.py``) only sets ``verified``
when a machine-checkable reference-comparison artifact exists and stamps the
level: ``ANALYTIC`` for closed-form reference comparisons, ``XVALUE`` for
cross-validation against external/literature data.
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
ENDPOINT_PREFIXES = ("cli:", "route:", "api:")
VERIFIED_LEVELS = ("ANALYTIC", "XVALUE")


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


def check_gates(caps: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Pure-data gate check over capability dicts (no import required).

    Mirrors :meth:`Registry.register` so the static (AST) dump can report
    violations as problems instead of crashing.
    """
    problems: list[dict[str, str]] = []
    for c in caps:
        cid = c.get("id", "?")
        ev = c.get("evidence") or {}
        if c.get("maturity") not in VALID_MATURITY:
            problems.append({
                "capability": cid, "code": "gate.maturity_invalid",
                "detail": f"maturity={c.get('maturity')!r} not in {VALID_MATURITY}",
            })
        if c.get("maturity") == "mature" and not (c.get("verified") and c.get("wired")):
            problems.append({
                "capability": cid, "code": "gate.mature_requires_verified_wired",
                "detail": f"verified={c.get('verified')} wired={c.get('wired')}",
            })
        if c.get("maturity") == "planned" and not str(ev.get("feature_ledger") or "").strip():
            problems.append({
                "capability": cid, "code": "gate.planned_requires_feature_ledger",
                "detail": "evidence.feature_ledger anchor missing",
            })
        ep = c.get("endpoint")
        if ep:
            if not c.get("wired"):
                problems.append({
                    "capability": cid, "code": "gate.endpoint_requires_wired",
                    "detail": f"endpoint={ep!r} but wired=False",
                })
            elif not str(ep).startswith(ENDPOINT_PREFIXES):
                problems.append({
                    "capability": cid, "code": "gate.endpoint_prefix",
                    "detail": f"endpoint={ep!r} must start with one of {ENDPOINT_PREFIXES}",
                })
        if c.get("verified") and ev.get("level") not in VERIFIED_LEVELS:
            problems.append({
                "capability": cid, "code": "gate.verified_requires_analytic_xvalue",
                "detail": f"evidence.level={ev.get('level')!r} not in {VERIFIED_LEVELS}",
            })
    return problems


class Registry:
    """Singleton capability registry with anti-overclaim gates."""

    def __init__(self) -> None:
        self._items: dict[str, Capability] = {}

    def register(self, cap: Capability) -> Capability:
        if cap.id in self._items:
            raise OverclaimError(f"capability {cap.id}: duplicate registration")
        for p in check_gates([cap.to_dict()]):
            raise OverclaimError(f"capability {cap.id}: {p['code']} ({p['detail']})")
        self._items[cap.id] = cap
        return cap

    def get(self, cap_id: str) -> Capability | None:
        return self._items.get(cap_id)

    def all(self) -> list[Capability]:
        return sorted(self._items.values(), key=lambda c: c.id)

    def clear(self) -> None:
        self._items.clear()


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
    Gate violations raise :class:`OverclaimError` at import time (fail loud).
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
    for d in node.decorator_list:
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
        except (SyntaxError, UnicodeDecodeError, OSError):
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


# ---------------------------------------------------------------------------
# pointer resolution — module / entry / test_path must exist, else problem
# ---------------------------------------------------------------------------

def _module_file(module_path: str, pkg_root: Path) -> Path | None:
    """Resolve ``tensordem.x.y`` to a file under the package root."""
    parts = module_path.split(".")
    if not parts or parts[0] != "tensordem":
        return None
    rel = Path(*parts[1:]) if len(parts) > 1 else Path()
    base = pkg_root / rel
    if base.is_dir():
        # a directory only shadows the module file when it is a real package
        init = base / "__init__.py"
        if init.is_file():
            return init
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return None


def _top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                names.add("*" if a.name == "*" else (a.asname or a.name))
    return names


def _module_names(mod_file: Path) -> set[str]:
    """Names available from a module file (packages: __init__ + submodules)."""
    try:
        tree = ast.parse(mod_file.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return set()
    names = _top_level_names(tree)
    if mod_file.name == "__init__.py":
        for py in sorted(mod_file.parent.glob("*.py")):
            if py.name == "__init__.py":
                continue
            try:
                sub = ast.parse(py.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue
            names |= _top_level_names(sub)
    return names


def resolve_pointers(
    caps: list[dict[str, Any]],
    pkg_root: Path | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, str]]:
    """Check module/entry/test_path pointers; missing = problem, not a drop.

    File/AST based (no import, no torch) so both CLI modes share it.
    """
    if pkg_root is None:
        pkg_root = Path(__file__).resolve().parent
    alt_root: Path | None = None
    if repo_root is None:
        git_root = _repo_root()
        if git_root is not None:
            repo_root = git_root
        else:
            # No .git (bare export / clean room): the checkout may be flat
            # (package at repo root) or src/-laid-out; accept either.
            repo_root = pkg_root.parent
            alt_root = pkg_root.parent.parent
    problems: list[dict[str, str]] = []
    for c in caps:
        cid = c.get("id", "?")
        mod = c.get("module_path") or ""
        mod_file = _module_file(mod, pkg_root) if mod else None
        if mod and mod_file is None:
            problems.append({
                "capability": cid, "code": "pointer.module_missing",
                "detail": f"module file not found for {mod!r}",
            })
            continue
        entry = c.get("entry")
        if entry and mod_file is not None and entry not in _module_names(mod_file):
            problems.append({
                "capability": cid, "code": "pointer.entry_missing",
                "detail": f"entry {entry!r} not defined in {mod}",
            })
        tp = c.get("test_path")
        if tp:
            ok = (repo_root / tp).is_file()
            if not ok and alt_root is not None:
                ok = (alt_root / tp).is_file()
            if not ok:
                problems.append({
                    "capability": cid, "code": "pointer.test_missing",
                    "detail": f"test file not found: {tp}",
                })
    return problems


# ---------------------------------------------------------------------------
# dump / parity / CLI
# ---------------------------------------------------------------------------

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


def _summary(caps: list[dict[str, Any]], problems: list[dict[str, str]]) -> dict[str, Any]:
    by_maturity = {m: 0 for m in VALID_MATURITY}
    for c in caps:
        m = c.get("maturity")
        if m in by_maturity:
            by_maturity[m] += 1
    return {
        "total": len(caps),
        "by_maturity": by_maturity,
        "wired": sum(1 for c in caps if c.get("wired")),
        "tested": sum(1 for c in caps if c.get("tested")),
        "verified": sum(1 for c in caps if c.get("verified")),
        "endpoints": sum(1 for c in caps if c.get("endpoint")),
        "problems": len(problems),
    }


def _import_caps() -> list[dict[str, Any]]:
    _ensure_package_importable()
    import tensordem.capabilities  # noqa: F401  (triggers registration)
    # Read the registry from the *imported* module: in direct file-path mode
    # this file runs as __main__ (its own REGISTRY), while capabilities.py
    # registers into the imported tensordem.registry singleton.
    import tensordem.registry as reg_mod
    return [c.to_dict() for c in reg_mod.REGISTRY.all()]


def _ensure_package_importable() -> None:
    """Make ``import tensordem`` resolve to the package co-located with this file.

    The platform host may carry a stale editable install of tensordem from
    another worktree (or none at all); the checkout that ships this
    registry.py must win, so its src dir is forced to the front of
    sys.path and any foreign cached modules are dropped.
    """
    src_dir = Path(__file__).resolve().parent.parent
    for name in [m for m in list(sys.modules)
                 if m == "tensordem" or m.startswith("tensordem.")]:
        f = getattr(sys.modules.get(name), "__file__", None) or ""
        if not Path(f).is_relative_to(src_dir):
            del sys.modules[name]
    if str(src_dir) in sys.path:
        sys.path.remove(str(src_dir))
    sys.path.insert(0, str(src_dir))


def dump(static: bool = False) -> dict[str, Any]:
    """Build the single-JSON-object payload (contract on stdout)."""
    if static:
        caps = static_scan()
    else:
        caps = _import_caps()
    caps = sorted(caps, key=lambda c: c.get("id", ""))
    problems = resolve_pointers(caps)
    problems += check_gates(caps)
    return {
        "source_commit": source_commit(),
        "capabilities": caps,
        "summary": _summary(caps, problems),
        "problems": problems,
    }


def _diff_caps(static: list[dict[str, Any]], imp: list[dict[str, Any]]) -> list[dict[str, Any]]:
    s = {c["id"]: c for c in static}
    i = {c["id"]: c for c in imp}
    diffs: list[dict[str, Any]] = []
    for cid in sorted(set(s) - set(i)):
        diffs.append({"id": cid, "where": "static_only"})
    for cid in sorted(set(i) - set(s)):
        diffs.append({"id": cid, "where": "import_only"})
    for cid in sorted(set(s) & set(i)):
        fields = [k for k in s[cid] if s[cid].get(k) != i[cid].get(k)]
        if fields:
            diffs.append({"id": cid, "where": "field_mismatch", "fields": fields})
    return diffs


def parity_report() -> dict[str, Any]:
    """Compare the static (AST) and import (REGISTRY) capability sets."""
    static = sorted(static_scan(), key=lambda c: c.get("id", ""))
    try:
        imp = sorted(_import_caps(), key=lambda c: c.get("id", ""))
        import_error: str | None = None
    except Exception as e:  # torch-free host without the package env
        imp, import_error = [], f"{type(e).__name__}: {e}"
    diffs = _diff_caps(static, imp)
    problems = resolve_pointers(static) + check_gates(static)
    if import_error is not None:
        problems.append({
            "capability": "-", "code": "parity.import_failed",
            "detail": import_error,
        })
    parity = import_error is None and not diffs
    return {
        "source_commit": source_commit(),
        "parity": parity,
        "static_count": len(static),
        "import_count": len(imp),
        "differences": diffs,
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tensordem-registry",
        description="Dump the TensorDEM capability registry as one JSON object on stdout.",
    )
    parser.add_argument(
        "--static", action="store_true",
        help="AST-scan the source tree (no package import, no torch)",
    )
    parser.add_argument(
        "--parity", action="store_true",
        help="compare static vs import mode field-by-field (exit 1 on mismatch)",
    )
    parser.add_argument(
        "--out", metavar="PATH", default=None,
        help="also write the JSON payload to PATH (stdout stays the contract)",
    )
    args = parser.parse_args(argv)
    if args.parity:
        payload = parity_report()
        rc = 0 if payload["parity"] else 1
    else:
        payload = dump(static=args.static)
        rc = 0
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(text + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
