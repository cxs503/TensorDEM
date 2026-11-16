#!/usr/bin/env python3
"""Generate src/tensordem/capabilities.py — the TensorDEM capability inventory.

Dual-pilot spec (m23186/m23215): every capability is registered with
machine-checkable evidence only.  Rules (documented, auditable):

* scope: every subpackage of tensordem + every top-level module
  (infrastructure modules __init__/__main__/cli/registry/capabilities excluded).
* wired: the entry symbol is *defined* in the module (AST check,
  environment-independent).  No entry found -> wired=False.
* tested: a test file tests/test_<stem>[_*].py exists.
* verified (grade A): a quantitative reference-comparison artifact exists:
    - a ``*verification*.py`` module inside the package (packages) or a
      sibling ``<stem>_verification.py`` (top-level modules), or the module
      itself is a verification/xval module;
    - a root-level ``verdict_*.json`` names the stem;
    - a bound test file is named *xval*/*verification*.
  Grade B = test binding only; grade C = declaration only.
* evidence.level (required by the registry gate for verified=True):
    - ANALYTIC: closed-form reference comparison (verification module);
    - XVALUE: cross-validation against external/literature data (xval
      module/test, campaign verdict artifact).
* maturity: mature iff verified AND wired AND tested (stricter than the
  platform 422 gate, which requires verified+wired); else developing.
  planned: only feature_ledger-anchored roadmap items (none for TensorDEM
  as of this generation -> no planned entries).
* endpoint: only the five real ``python -m tensordem`` CLI subcommands
  (cli.py), and only when the target module is wired.

Regenerate with:  python scripts/gen_dem_capabilities.py
Do not edit the generated file by hand.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "tensordem"
TESTS = ROOT / "tests"

INFRA_MODULES = {"__init__", "__main__", "cli", "registry", "capabilities"}

KIND_RULES: list[tuple[str, str]] = [
    ("ice", "ice"), ("floe", "ice"), ("brash", "ice"), ("clear_ice", "ice"),
    ("level_ice", "ice"), ("mixed_ice", "ice"), ("pancake", "ice"),
    ("myi", "ice"), ("threebody_ice", "ice"), ("propeller_ice", "ice"),
    ("wave_ice", "ice"), ("wave_floe", "ice"), ("adfreeze", "ice"),
    ("freeze_bond", "ice"), ("wet_ice", "ice"), ("broken_ice", "ice"),
    ("hull_collision", "ice"), ("ridge", "ice"), ("salinity", "ice"),
    ("random_ice", "ice"), ("icebreaker", "ice"),
    ("granular", "granular"), ("chute", "granular"), ("drum", "granular"),
    ("silo", "granular"), ("rotary", "granular"), ("segregation", "granular"),
    ("bulk_cargo", "granular"), ("cargo", "granular"), ("biaxial", "granular"),
    ("ball_", "granular"), ("bounce", "granular"), ("column", "granular"),
    ("erosion", "fracture"), ("comminution", "fracture"), ("breakage", "fracture"),
    ("single_particle_crushing", "fracture"), ("fatigue", "fracture"),
    ("fem_dem", "fem_dem"), ("lefm", "fem_dem"),
    ("thermal", "thermal"), ("interface_thermal", "thermal"), ("bowden", "thermal"),
    ("acoustic", "acoustics"), ("ae", "acoustics"),
    ("cfd_dem", "coupling"), ("point_cfd_dem", "coupling"), ("two_way", "coupling"),
    ("lbmdem", "coupling"), ("coupling_kernel", "coupling"),
    ("threefield", "coupling"), ("bedcoupling", "coupling"),
    ("lagrangian_bed", "coupling"), ("seepage", "coupling"),
    ("polyhedron", "shape"), ("polyhedral", "shape"),
    ("shape_representation", "shape"),
    ("calibration", "calibration"), ("auto_calibration", "calibration"),
    ("bpm", "calibration"), ("sensitivity", "calibration"),
    ("identifiability", "calibration"), ("uncertainty", "calibration"),
    ("upscaling", "upscaling"), ("nonlocal", "upscaling"),
    ("resistance_closure", "upscaling"), ("scale_ceiling", "upscaling"),
    ("xval", "verification"), ("verification", "verification"),
]

# real CLI subcommands in src/tensordem/cli.py (checked against the source)
ENDPOINTS = {
    "tensordem.dem": "cli:python -m tensordem dem-2d",
    "tensordem.friction_polyhedron_real_verification":
        "cli:python -m tensordem verify-friction-polyhedron",
    "tensordem.elastic_limit_shear": "cli:python -m tensordem elastic-limit-shear",
    "tensordem.polyhedron_floe3d": "cli:python -m tensordem h1-broken-ice",
    "tensordem.binding_contract": "cli:python -m tensordem binding-contract",
}


def kind_for(name: str) -> str:
    for prefix, kind in KIND_RULES:
        if name == prefix or name.startswith(prefix + "_"):
            return kind
    return "dem"


def parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None


def first_public_entry(tree: ast.Module) -> str | None:
    """First public class (preferred) or function defined in the module."""
    classes = [n.name for n in tree.body
               if isinstance(n, ast.ClassDef) and not n.name.startswith("_")]
    if classes:
        return classes[0]
    funcs = [n.name for n in tree.body
             if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
    return funcs[0] if funcs else None


def entry_for_package(pkgdir: Path) -> str | None:
    init = parse(pkgdir / "__init__.py")
    if init is not None:
        e = first_public_entry(init)
        if e:
            return e
    for py in sorted(pkgdir.glob("*.py")):
        if py.name == "__init__.py":
            continue
        t = parse(py)
        if t is not None:
            e = first_public_entry(t)
            if e:
                return e
    return None


def entry_for_module(path: Path) -> str | None:
    t = parse(path)
    return first_public_entry(t) if t is not None else None


def bound_tests(stem: str) -> list[str]:
    pat = re.compile(rf"^test_{re.escape(stem)}(_.*|)\.py$")
    return sorted(p.name for p in TESTS.glob("test_*.py") if pat.match(p.name))


def verification_evidence(stem: str, pkgdir: Path | None) -> tuple[str, str] | None:
    """Machine-checkable reference-comparison artifact, or None.

    Returns ``(level, note)`` where level is ANALYTIC (closed-form reference
    comparison) or XVALUE (cross-validation vs external/literature data).
    """
    if stem.endswith("_xval"):
        return ("XVALUE", f"module itself is an xval module ({stem}.py)")
    if stem.endswith("_verification"):
        return ("ANALYTIC", f"module itself is a verification module ({stem}.py)")
    if pkgdir is not None:
        for py in sorted(pkgdir.rglob("*.py")):
            if py.name.endswith("_xval.py"):
                return ("XVALUE", f"xval module in package: {py.name}")
        for py in sorted(pkgdir.rglob("*.py")):
            if "verification" in py.name:
                return ("ANALYTIC", f"verification module in package: {py.name}")
    else:
        sib = PKG / f"{stem}_verification.py"
        if sib.exists():
            return ("ANALYTIC", f"sibling verification module: {sib.name}")
    for v in sorted(ROOT.glob("verdict_*.json")):
        try:
            if stem in v.read_text(encoding="utf-8", errors="ignore"):
                return ("XVALUE", f"root verdict artifact names the module: {v.name}")
        except OSError:
            continue
    for t in bound_tests(stem):
        if "xval" in t:
            return ("XVALUE", f"xval test binding: tests/{t}")
        if "verification" in t:
            return ("ANALYTIC", f"verification test binding: tests/{t}")
    return None


def main() -> int:
    items: list[dict] = []
    subpackages = sorted(
        d.name for d in PKG.iterdir()
        if d.is_dir() and (d / "__init__.py").exists()
    )
    top_modules = sorted(
        p.stem for p in PKG.glob("*.py")
        if p.stem not in INFRA_MODULES
    )

    for name in subpackages:
        pkgdir = PKG / name
        entry = entry_for_package(pkgdir)
        tests = bound_tests(name)
        ev = verification_evidence(name, pkgdir)
        wired = entry is not None
        tested = bool(tests)
        verified = ev is not None
        grade = "A" if verified else ("B" if tested else "C")
        maturity = "mature" if (verified and wired and tested) else "developing"
        mod = f"tensordem.{name}"
        evidence = {"grade": grade,
                    "note": (ev[1] if ev else ("test binding" if tested else "declaration only"))}
        if ev:
            evidence["level"] = ev[0]
        items.append({
            "id": f"dem.{name}", "kind": kind_for(name), "maturity": maturity,
            "module_path": mod, "entry": entry, "params": {},
            "wired": wired, "tested": tested, "verified": verified,
            "test_path": f"tests/{tests[0]}" if tests else None,
            "evidence": evidence,
            "endpoint": ENDPOINTS.get(mod) if wired else None,
        })

    for name in top_modules:
        path = PKG / f"{name}.py"
        entry = entry_for_module(path)
        tests = bound_tests(name)
        ev = verification_evidence(name, None)
        wired = entry is not None
        tested = bool(tests)
        verified = ev is not None
        grade = "A" if verified else ("B" if tested else "C")
        maturity = "mature" if (verified and wired and tested) else "developing"
        mod = f"tensordem.{name}"
        cap_id = f"dem.{name}"
        if name in subpackages:  # package and module share the stem
            cap_id = f"dem.{name}_mod"
        evidence = {"grade": grade,
                    "note": (ev[1] if ev else ("test binding" if tested else "declaration only"))}
        if ev:
            evidence["level"] = ev[0]
        items.append({
            "id": cap_id, "kind": kind_for(name), "maturity": maturity,
            "module_path": mod, "entry": entry, "params": {},
            "wired": wired, "tested": tested, "verified": verified,
            "test_path": f"tests/{tests[0]}" if tests else None,
            "evidence": evidence,
            "endpoint": ENDPOINTS.get(mod) if wired else None,
        })

    lines = [
        '"""TensorDEM capability inventory (GENERATED — do not edit by hand).',
        "",
        "Generated by scripts/gen_dem_capabilities.py from machine-checkable",
        "evidence (AST entry scan, tests/ bindings, verification artifacts).",
        "Regenerate after adding/removing modules; the static registry dump",
        "(``python -m tensordem.registry --static``) must stay in sync.",
        '"""',
        "from .registry import capability",
        "",
        "",
    ]
    for it in items:
        lines.append("@capability(")
        for k in ("id", "kind", "maturity", "module_path", "entry", "params",
                  "wired", "tested", "verified", "test_path", "evidence", "endpoint"):
            lines.append(f"    {k}={repr(it[k])},")
        lines.append(")")
        marker = "_cap_" + it["id"].replace(".", "_")
        lines.append(f"class {marker}:")
        lines.append(f'    """capability marker for {it["id"]}"""')
        lines.append("")
    out = PKG / "capabilities.py"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    n = len(items)
    n_mature = sum(1 for i in items if i["maturity"] == "mature")
    n_dev = sum(1 for i in items if i["maturity"] == "developing")
    n_a = sum(1 for i in items if i["evidence"]["grade"] == "A")
    n_b = sum(1 for i in items if i["evidence"]["grade"] == "B")
    n_c = sum(1 for i in items if i["evidence"]["grade"] == "C")
    n_wired = sum(1 for i in items if i["wired"])
    n_tested = sum(1 for i in items if i["tested"])
    n_ep = sum(1 for i in items if i["endpoint"])
    print(f"capabilities.py: {n} entries "
          f"(mature={n_mature}, developing={n_dev}, planned=0)")
    print(f"grade A={n_a} B={n_b} C={n_c}; wired={n_wired} tested={n_tested} "
          f"endpoints={n_ep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
