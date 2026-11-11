#!/usr/bin/env python3
"""Repo-wide import-contract scan: do the tests' imports actually resolve?

Why this exists
---------------
The merge gate has repeatedly filtered delivery files while keeping the tests that
import them (commit subjects end in "N 个文件被门禁滤除").  The suite then fails at
*collection*, and a filtered test module is invisible: no red test, just a silent
hole in the verification surface.  This scan makes the hole machine-visible.

For every ``from tensordem.X import a, b`` in ``tests/**`` it checks that module
``X`` imports and that the name ``a`` is an attribute of it (or a submodule
``tensordem.X.a``).  Output: JSON + exit code 1 when anything is unresolved.

Usage:  PYTHONPATH=src python3 tools/import_contract_scan.py [--json PATH]
"""
from __future__ import annotations

import argparse
import ast
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="artifacts/import_contract/broken_imports.json")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    broken = scan(Path(args.root))
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
    }, indent=1) + "\n")

    print(f"unresolved test imports: {len(broken)} "
          f"in {len({r['test'] for r in broken})} test modules")
    for pkg, rows in sorted(by_pkg.items()):
        print(f"  {pkg:45s} {len(rows):3d}  ({len({r['test'] for r in rows})} test module(s))")
    print(f"artifact: {out}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
