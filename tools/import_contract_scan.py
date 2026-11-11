#!/usr/bin/env python3
"""Repo-wide import-contract scan: do the tests' imports actually resolve?

Why this exists
---------------
The merge gate has repeatedly filtered delivery files while keeping the tests that
import them (commit subjects end in "N 个文件被门禁滤除").  The suite then fails at
*collection*, and a filtered test module is invisible: no red test, just a silent
hole in the verification surface.  This scan makes the hole machine-visible.

Two checks:

``tests``
    For every ``from tensordem.X import a, b`` in ``tests/**`` check that module
    ``X`` imports and that the name ``a`` is an attribute of it (or a submodule
    ``tensordem.X.a``).

``shadows``
    Package/module same-name collisions (``src/tensordem/X.py`` next to
    ``src/tensordem/X/``).  Python resolves the package first, so the module file
    becomes **unreachable** and every symbol it defines silently disappears.  This
    has bitten TensorDEM three times (freeze_bond m14449, ice_structure_contact
    m14450, shape_representation audit F3); here it is detected structurally
    instead of waiting for a test to fail at collection.  A collision is *closed*
    when every public symbol of the legacy module is reachable from the package
    (re-exported); otherwise it fails the scan.  Legacy files that are byte
    duplicates of an in-package module are reported as stale duplicates (advisory).

Usage:  PYTHONPATH=src python3 tools/import_contract_scan.py [--json PATH]
Exit code 0 = clean, 1 = unresolved test imports and/or shadowed symbols.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def scan(root: Path) -> list[dict]:
    broken: list[dict] = []
    for path in sorted(root.glob("tests/**/*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:                      # pragma: no cover
            broken.append({"test": str(path), "module": "<parse>", "name": "",
                           "kind": "syntax_error", "detail": str(exc)})
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.module
                    and node.module.startswith("tensordem")):
                continue
            try:
                mod = importlib.import_module(node.module)
            except Exception as exc:                    # noqa: BLE001
                broken.append({"test": str(path), "module": node.module, "name": "*",
                               "kind": "module_import", "detail": repr(exc)[:160]})
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                if hasattr(mod, alias.name):
                    continue
                try:
                    importlib.import_module(f"{node.module}.{alias.name}")
                except Exception as exc:                # noqa: BLE001
                    broken.append({"test": str(path), "module": node.module,
                                   "name": alias.name, "kind": "missing_symbol",
                                   "detail": type(exc).__name__})
    return broken


def _public_names(path: Path) -> list[str]:
    """Top-level public names defined by a module file (no dunders)."""
    names: list[str] = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return sorted({n for n in names if not n.startswith("__")})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_shadows(src_root: Path) -> list[dict]:
    """Package/module same-name collisions and the symbols they swallow."""
    rows: list[dict] = []
    dirs = {p.name: p for p in src_root.iterdir()
            if p.is_dir() and (p / "__init__.py").exists()}
    for legacy in sorted(src_root.glob("*.py")):
        stem = legacy.stem
        if stem not in dirs:
            continue
        pkg_name = f"tensordem.{stem}"
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as exc:                        # noqa: BLE001
            rows.append({"package": pkg_name, "legacy_file": str(legacy),
                         "status": "package_import_error", "lost": [],
                         "detail": repr(exc)[:160]})
            continue
        public = _public_names(legacy)
        lost = [n for n in public if not hasattr(pkg, n)]
        legacy_sha = _sha(legacy)
        duplicate_of = next(
            (str(p.relative_to(src_root)) for p in sorted(dirs[stem].glob("*.py"))
             if _sha(p) == legacy_sha), None)
        if lost:
            status = "shadowed_symbols_lost"
        elif duplicate_of:
            status = "stale_duplicate"
        else:
            status = "closed_by_reexport"
        rows.append({"package": pkg_name, "legacy_file": str(legacy),
                     "legacy_lines": len(legacy.read_text().splitlines()),
                     "public_names": len(public), "lost": lost,
                     "duplicate_of": duplicate_of, "status": status})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="artifacts/import_contract/broken_imports.json")
    ap.add_argument("--root", default=".")
    ap.add_argument("--src", default="src/tensordem")
    args = ap.parse_args(argv)

    broken = scan(Path(args.root))
    shadows = scan_shadows(Path(args.src))
    failing_shadows = [r for r in shadows if r["status"] in
                       ("shadowed_symbols_lost", "package_import_error")]

    by_pkg: dict[str, list[dict]] = defaultdict(list)
    for row in broken:
        by_pkg[row["module"]].append(row)

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_broken_symbols": len(broken),
        "n_test_modules_affected": len({r["test"] for r in broken}),
        "packages": {k: sorted({r["name"] for r in v}) for k, v in by_pkg.items()},
        "rows": broken,
        "shadow_collisions": {
            "n_collisions": len(shadows),
            "n_lost_symbols": sum(len(r["lost"]) for r in shadows),
            "n_failing": len(failing_shadows),
            "rows": shadows,
        },
    }, indent=1) + "\n")

    print(f"unresolved test imports: {len(broken)} "
          f"in {len({r['test'] for r in broken})} test modules")
    for pkg, rows in sorted(by_pkg.items()):
        print(f"  {pkg:45s} {len(rows):3d}  ({len({r['test'] for r in rows})} test module(s))")
    print(f"package/module shadow collisions: {len(shadows)} "
          f"({len(failing_shadows)} failing, "
          f"{sum(len(r['lost']) for r in shadows)} lost symbols)")
    for r in shadows:
        note = f" lost={r['lost']}" if r["lost"] else ""
        dup = f" dup_of={r['duplicate_of']}" if r.get("duplicate_of") else ""
        print(f"  {r['package']:45s} {r['status']:22s}{note}{dup}")
    print(f"artifact: {out}")
    return 1 if (broken or failing_shadows) else 0


if __name__ == "__main__":
    sys.exit(main())