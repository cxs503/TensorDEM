#!/usr/bin/env python3
"""README claim <-> code evidence contract check (AUDIT-SCI).

Why this exists
---------------
A README is a *claim set*.  Every capability sentence is either backed by code
that exists, a test that exercises that code, an archived number, or it is backed
by nothing.  For a tree this size, prose review cannot keep that in sync, so the
registry below is **executed**: CI (and ``tests/test_readme_claim_audit.py``)
fails when a claim loses its evidence or when README starts advertising something
the tree cannot back.

What is checked, per registered claim
-------------------------------------
* ``anchor``   — the exact sentence/fragment must still be in ``README.md``
                 (a silent README rewrite retires the entry instead of leaving a
                 stale audit behind);
* ``code`` / ``symbols`` — the implementing files must exist, and ``path::Name``
                 must be *defined* in that file's AST (not imported, not a
                 comment);
* ``tests``    — the registered test files must exist and at least one of them
                 must reference the module under test;
* ``docs`` / ``schemas`` — tracked evidence (must be in the tree);
* ``artifacts`` — README-cited *generated* evidence.  This repo's convention is
                 that generated outputs are **not** in VCS
                 (``scripts/readme_api_audit.py:_test_style`` neighbourhood,
                 "生成物不入 VCS ... 干净检出里这类断言会让 pytest -q 固定红"), so a
                 missing ``artifacts/...`` path is not by itself a defect — it is
                 only acceptable if a registered ``generator`` in the same repo can
                 rebuild it.  Missing artifact *and* missing generator = WITHHELD;
* ``numbers``  — every number README quotes must have an independent copy in a
                 named source file; if the declared source lacks it, the whole
                 repo is searched so "wrong pointer" is separated from "number
                 exists nowhere";
* ``gates``    — machine-readable criteria (``gates.json`` style) must still hold
                 the value the README quotes, and the pass/fail flag the README
                 quotes must still be that flag (this is how a silent honest-FAIL
                 → pass flip, or a quiet threshold move, gets caught);
* ``counterclaim`` — a sentence elsewhere in the same README denying the
                 capability, while the code and docs for it exist, is a
                 SELF_CONTRADICTION (readers get opposite statements);
* ``census``   — README's self-reported test-suite counts are recounted.

Two repositories are covered because the audit anchor is the pair: TensorDEM
(this checkout) and TensorSPH (read-only external root via ``--sph-root``; absent
root reports NOT_AUDITED rather than a pass).

Usage
-----
    python scripts/readme_claim_audit.py                  # DEM + SPH (if present)
    python scripts/readme_claim_audit.py --no-external     # DEM only
    python scripts/readme_claim_audit.py --check-entrypoints
    python scripts/readme_claim_audit.py --strict          # exit 1 on regression

Outputs: ``artifacts/readme_claim_audit/{claims.json,CLAIM_EVIDENCE_TABLE.md}``.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = "readme_claim_audit.v2"

#: top-level directories whose relative paths are treated as repo paths
TOP_DIRS = ("docs", "src", "tests", "scripts", "benchmarks", "artifacts", "examples",
            "schemas", "tools", "results")


# ---------------------------------------------------------------------------
# registry: one entry per auditable README sentence
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    cid: str
    section: str
    anchor: str  # exact fragment that must still appear in README.md
    repo: str = "dem"  # "dem" | "sph"
    code: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()  # "path::Name" that must be *defined* there
    tests: tuple[str, ...] = ()
    test_must_reference: str | None = None
    docs: tuple[str, ...] = ()
    schemas: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()  # generated, not in VCS by convention
    generators: tuple[str, ...] = ()  # scripts that rebuild the artifacts
    numbers: tuple[tuple[str, str], ...] = ()  # (literal, independent source)
    gates: tuple[tuple[str, str, object], ...] = ()  # (json, gate key, expected measured)
    json_checks: tuple[tuple[str, str, object], ...] = ()  # (json, dotted key, expected value)
    qualifications: tuple[tuple[str, str, object], ...] = ()  # (json, dotted key, expected)
    counterclaim: str | None = None
    census: bool = False
    note: str = ""
    baseline: str = "REAL"


DEM_CLAIMS: tuple[Claim, ...] = (
    Claim(
        cid="DEM-CORE-2D",
        section="模型与单位",
        anchor="中心力弹簧黏结",
        code=("src/tensordem/dem.py",),
        symbols=("src/tensordem/dem.py::IceDEM", "src/tensordem/dem.py::DEMConfig"),
        tests=("tests/test_binding_contract.py",),
        test_must_reference="IceDEM",
        docs=("docs/tutorials/first_verified_run.md",),
        note="2-D bonded lattice, penalty contact, prescribed indenter; tutorial is "
             "executed by tests/test_tutorial_first_verified_run.py.",
    ),
    Claim(
        cid="DEM-EXTERNAL-LOAD",
        section="外部载荷接口",
        anchor="IceDEM.forces(external_forces=loads)",
        code=("src/tensordem/dem.py",),
        symbols=("src/tensordem/dem.py::IceDEM",),
        tests=("tests/test_load_interface.py",),
        test_must_reference="external_forces",
        note="README itself states this is a load entry point, not a coupling.",
    ),
    Claim(
        cid="DEM-LBMDEM-DRAG",
        section="两向 LBM–DEM 曳力闭合",
        anchor="D3Q19 流体求解器与两种浸没边界",
        code=("src/tensordem/lbmdem/__init__.py", "src/tensordem/lbmdem/drag.py",
              "src/tensordem/lbmdem/ibm.py"),
        tests=("tests/test_lbmdem_drag_calibration.py",),
        test_must_reference="lbmdem",
        docs=("docs/tensordem_coupling_p1.md",),
        schemas=("benchmarks/verified/lbm_dem_drag_calibration/gates.json",
                 "benchmarks/verified/lbm_dem_drag_calibration/result.json"),
        numbers=(("0.98", "docs/tensordem_coupling_p1.md"),
                 ("1.19", "docs/tensordem_coupling_p1.md")),
        gates=(("benchmarks/verified/lbm_dem_drag_calibration/gates.json",
                "stokes_drag_analytic", 0.009756625767396221),),
        note="README's honest-FAIL sentence (all_recomputed_pass = false) is asserted "
             "separately via the gate flag below.",
    ),
    Claim(
        cid="DEM-SENSITIVITY-SCAN",
        section="网格/分辨率敏感性扫描",
        anchor="tensordem.sensitivity",
        code=("src/tensordem/sensitivity.py",),
        tests=("tests/test_sensitivity.py",),
        test_must_reference="sensitivity",
        note="README explicitly denies it is a convergence-order estimator.",
    ),
    Claim(
        cid="DEM-POLYLINE-HULL",
        section="Prescribed polyline hull mode (prototype)",
        anchor="examples/hull_profile.csv",
        code=("src/tensordem/dem.py",),
        tests=("tests/test_hull_profile_cli.py", "tests/test_hull_contact_integration.py"),
        test_must_reference="hull_profile",
        docs=("examples/hull_profile.csv",),
        note="Explicitly a prototype; no calibrated resistance claim.",
    ),
    Claim(
        cid="DEM-DEM3D",
        section="3-D bonded-sphere solver (new prototype)",
        anchor="`DEM3DConfig` and `IceDEM3D`",
        code=("src/tensordem/dem3d.py",),
        symbols=("src/tensordem/dem3d.py::IceDEM3D", "src/tensordem/dem3d.py::DEM3DConfig"),
        tests=("tests/test_dem3d.py",),
        test_must_reference="dem3d",
        note="Labelled 'new prototype' in README.",
    ),
    Claim(
        cid="DEM-RIDGE-GROUNDING",
        section="冰脊触底与海底管道",
        anchor="ridge_dem` / `ridge_geotech` / `ridge_grounding`",
        code=("src/tensordem/ridge_dem.py", "src/tensordem/ridge_geotech.py",
              "src/tensordem/ridge_grounding.py"),
        tests=("tests/test_ridge_grounding.py",
               "tests/test_ridge_geotech_capacity_verification.py",
               "tests/test_ice_ridge_grounding_rederivation.py"),
        test_must_reference="ridge_grounding",
        docs=("docs/ice_ridge_pipeline_grounding.md",),
        artifacts=("artifacts/ridge_grounding/",),
        generators=("scripts/run_ice_ridge_grounding.py",),
        numbers=(("DNV-OS-F101", "docs/ice_ridge_pipeline_grounding.md"),
                 ("Mohr-Coulomb", "src/tensordem/ridge_geotech.py")),
        baseline="REAL_REGENERABLE",
        note="Anchor of this audit round (DEM half 1/3).",
    ),
    Claim(
        cid="DEM-ICE-ABRASION",
        section="冰–船体水线带冲蚀磨损",
        anchor="ice_abrasion_archard_wear",
        code=("src/tensordem/ice_abrasion.py",),
        tests=("tests/test_ice_abrasion.py", "tests/test_ice_abrasion_field.py",
               "tests/test_ice_abrasion_extremes_anchor.py"),
        test_must_reference="ice_abrasion",
        docs=("docs/ice_abrasion_archard_wear.md",),
        artifacts=("artifacts/ice_abrasion/",),
        numbers=(("V = K W L / H", "docs/ice_abrasion_archard_wear.md"),
                 ("CALIBRATED", "docs/ice_abrasion_archard_wear.md")),
        baseline="REAL_REGENERABLE",
        note="Anchor of this audit round (DEM half 2/3). Single-parameter k is flagged "
             "CALIBRATED by the README, so no uncalibrated-magnitude claim is made.",
    ),
    Claim(
        cid="DEM-ICE-TRANSIT",
        section="稳态碎冰场直航阻力与接触级能量账本",
        anchor="tensordem.ice_transit",
        code=("src/tensordem/ice_transit/__init__.py", "src/tensordem/ice_transit/transit.py",
              "src/tensordem/ice_transit/laws.py"),
        tests=("tests/test_ice_transit.py",),
        test_must_reference="ice_transit",
        docs=("docs/ICE_TRANSIT_RESISTANCE_SURFACE.md",),
        artifacts=("artifacts/ice_transit/",),
        generators=("scripts/run_ice_transit_campaign.py",),
        numbers=(("W_thrust", "docs/ICE_TRANSIT_RESISTANCE_SURFACE.md"),
                 ("signed_distance", "src/tensordem/ice_transit/laws.py")),
        baseline="REAL_REGENERABLE",
        note="Anchor of this audit round (DEM half 3/3). Energy ledger is an identity "
             "asserted per contact, not a post-hoc residual.",
    ),
    Claim(
        cid="DEM-PRESSURE-BAND",
        section="冰压带相关长度 λ_x 的标定估计量",
        anchor="ice_pressure_band`",
        code=("src/tensordem/ice_pressure_band/__init__.py",),
        tests=("tests/test_ice_pressure_band.py", "tests/test_ice_pressure_band_maturation.py"),
        test_must_reference="ice_pressure_band",
        docs=("docs/ice_pressure_band_lambda_x.md",
              "docs/ice_pressure_band_lambda_x_maturation.md"),
        schemas=("schemas/lambda_x_estimate.v1.json",),
        artifacts=("artifacts/pressure_band_lambda/",
                   "artifacts/pressure_band_lambda_maturation/criteria.json"),
        generators=("scripts/pressure_band_lambda_calibration.py",
                    "scripts/pressure_band_maturation.py"),
        numbers=(("0.058", "docs/ice_pressure_band_lambda_x_maturation.md"),
                 ("1.5–3", "docs/ice_pressure_band_lambda_x.md")),
        baseline="REAL_REGENERABLE",
        note="README carries the honest FAIL (native λ_x/h 0.058 vs literature 1.5-3).",
    ),
    Claim(
        cid="DEM-README-TEST-CENSUS",
        section="安装与运行",
        anchor="个模块级 `test_*` 函数",
        census=True,
        note="Self-census.  The anchor is stable prose, not the count itself, so the entry "
             "survives the next recount; a date-stamped snapshot that has drifted is "
             "reported as info, an undated mismatch is a verdict.",
    ),
)

SPH_CLAIMS: tuple[Claim, ...] = (
    Claim(
        cid="SPH-2D-CASES",
        section="Implemented",
        anchor="2-D dam break (B2)",
        repo="sph",
        code=("src/tensorsph/droplet_impact.py",),
        tests=("tests/test_dam_break.py",),
        test_must_reference="dam_break",
        docs=("docs/benchmarks/dam-break-2d.md",),
        numbers=(("0.96", "docs/benchmarks/dam-break-2d.md"),),
    ),
    Claim(
        cid="SPH-3D-OPERATORS",
        section="Implemented",
        anchor="3-D operators (`tensorsph.threed`)",
        repo="sph",
        code=("src/tensorsph/threed/__init__.py", "src/tensorsph/threed/kernel3.py"),
        tests=("tests/test_threed_operators.py", "tests/test_threed_kernel.py"),
        test_must_reference="threed",
        docs=("docs/verification_threed.md",),
        numbers=(("2.7", "docs/verification_threed.md"),),
        json_checks=(
            ("benchmarks/threed_shear/report.json", "resolution_ladder.-1.decay.rel_error", -0.0268),
            ("benchmarks/threed_hydrostatic/report.json",
             "case=adami-c0_20-nz16:pressure.bottom_rel_error", -0.0249),
        ),
        qualifications=(
            ("benchmarks/threed_shear/report.json", "verdict", "fail"),
            ("benchmarks/threed_hydrostatic/report.json",
             "case=adami-c0_40-nz16:pressure.bottom_rel_error", -0.0121),
        ),
        counterclaim="no 3-D free surface, surface tension or multiphase",
        baseline="WITHHELD",
        note="Anchor of this round (SPH half). README routes both 3-D quantitative "
             "claims (shear decay, hydrostatic pressure) to one doc file.",
    ),
    Claim(
        cid="SPH-SURFACE-TENSION",
        section="Implemented",
        anchor="mixture-material surface tension",
        repo="sph",
        code=("src/tensorsph/csf_mixture.py", "src/tensorsph/surface_tension.py"),
        tests=("tests",),
        docs=("docs/verification_mixture_surface_tension.md",
              "docs/verification_surface_tension_curvature.md"),
        numbers=(("1.4", "docs/verification_mixture_surface_tension.md"),),
        counterclaim="No surface-tension model.",
        baseline="SELF_CONTRADICTION",
        note="README 'Current limitations' still denies the capability its "
             "'Implemented' section verifies.",
    ),
    Claim(
        cid="SPH-WALL-MODEL",
        section="Implemented",
        anchor="static wall particles",
        repo="sph",
        code=("src/tensorsph",),
        tests=("tests",),
        docs=("docs/benchmarks/dam-break-2d.md",),
        numbers=(("8.6", "docs/benchmarks/dam-break-2d.md"),),
        note="Adami eq. (11) completion is the stated reason the B2 density gate "
             "moved from fail to pass.",
    ),
)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
BACKTICK_RE = re.compile(r"`([^`\n]+)`")


def readme_pointers(text: str) -> set[str]:
    """Relative paths the README promises exist (markdown links + repo-y backticks)."""
    out: set[str] = set()
    for m in LINK_RE.finditer(text):
        tgt = m.group(1)
        if tgt.startswith(("http://", "https://", "#", "mailto:")):
            continue
        out.add(tgt.split("#")[0])
    for m in BACKTICK_RE.finditer(text):
        cand = m.group(1).strip()
        if not cand or "://" in cand or cand.startswith("-") or " " in cand:
            continue
        cand = cand.split("#")[0].split(":")[0]  # `file.py:symbol` -> `file.py`
        for expanded in _expand_braces(cand):
            if expanded.split("/")[0] in TOP_DIRS:
                out.add(expanded)
    return out


def _expand_braces(token: str) -> list[str]:
    """``dir/{a.json,b.json}`` -> the two real paths (docs use brace shorthand)."""
    m = re.search(r"\{([^{}]+)\}", token)
    if not m:
        return [token]
    out = []
    for alt in m.group(1).split(","):
        out.extend(_expand_braces(token[:m.start()] + alt + token[m.end():]))
    return out


def dangling_pointers(root: Path, text: str) -> list[str]:
    return [p for p in sorted(readme_pointers(text)) if not (root / p).exists()]


def top_level_names(root: Path, rel: str) -> tuple[bool, set[str], str]:
    path = root / rel
    if not path.exists():
        return False, (set(), set()), "missing"
    if not rel.endswith(".py"):
        return True, (set(), set()), ""
    try:
        tree = ast.parse(_read(path))
    except SyntaxError as exc:  # pragma: no cover
        return False, (set(), set()), f"unparsable: {exc}"
    names: set[str] = set()
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])
    # an imported name is *not* an implementation: `from x import Thing` does not
    # make this file implement Thing, and a README claim backed only by a re-export
    # must be reported rather than silently accepted.
    return True, (names, imported), ""


def test_files_referencing(root: Path, needle: str) -> list[str]:
    tests = root / "tests"
    if not tests.is_dir():
        return []
    hits = []
    for f in sorted(tests.glob("test_*.py")):
        try:
            if needle in _read(f):
                hits.append(str(f.relative_to(root)))
        except OSError:  # pragma: no cover
            continue
    return hits


def repo_grep(root: Path, needle: str, limit: int = 6) -> list[str]:
    """Where else does ``needle`` appear (tracked files first, capped)?"""
    files: list[Path] = []
    try:
        out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True,
                             text=True, timeout=120)
        if out.returncode == 0:
            files = [root / line for line in out.stdout.splitlines() if line]
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        files = []
    if not files:
        files = [p for p in root.rglob("*") if p.is_file()
                 and p.suffix in {".md", ".py", ".json", ".csv", ".yaml", ".yml", ".toml"}]
    hits: list[str] = []
    for f in files:
        if f.suffix not in {".md", ".py", ".json", ".csv", ".yaml", ".yml", ".toml"}:
            continue
        if f.name == "README.md" and f.parent == root:
            continue  # the README itself is the place the number came *from*
        try:
            if needle in _read(f):
                hits.append(str(f.relative_to(root)))
                if len(hits) >= limit:
                    break
        except OSError:
            continue
    return hits


def read_gate(root: Path, rel: str, key: str) -> dict:
    path = root / rel
    if not path.exists():
        return {"found": False, "reason": "file missing"}
    try:
        data = json.loads(_read(path))
    except json.JSONDecodeError as exc:
        return {"found": False, "reason": f"unparsable: {exc}"}
    for g in data.get("gates", []):
        if g.get("gate") == key:
            return {"found": True, "measured": g.get("measured"),
                    "threshold": g.get("threshold"), "passed": g.get("passed"),
                    "all_recomputed_pass": data.get("all_recomputed_pass")}
    return {"found": False, "reason": f"gate '{key}' absent"}


def json_lookup(data, key: str):
    """Resolve ``case=NAME:a.b.-1.c`` against a benchmark report.

    ``case=`` filters the report's ``runs`` list (TensorSPH hydrostatic report);
    ``-1`` indexes a list from the end.  Returns ``(found, value, reason)``.
    """
    cur = data
    rest = key
    if rest.startswith("case="):
        name, _, rest = rest[len("case="):].partition(":")
        runs = data.get("runs") if isinstance(data, dict) else None
        if not isinstance(runs, list):
            return False, None, "report has no 'runs' list"
        match = [r for r in runs if isinstance(r, dict) and r.get("case") == name]
        if not match:
            return False, None, f"case '{name}' absent"
        cur = match[0]
    for part in [p for p in rest.split(".") if p]:
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return False, None, f"bad list index '{part}'"
        elif isinstance(cur, dict):
            if part not in cur:
                return False, None, f"key '{part}' absent"
            cur = cur[part]
        else:
            return False, None, f"cannot descend into {type(cur).__name__} at '{part}'"
    return True, cur, ""


def run_json_checks(root: Path, rel: str, checks) -> list[dict]:
    out: list[dict] = []
    path = root / rel
    data = None
    if path.exists():
        try:
            data = json.loads(_read(path))
        except json.JSONDecodeError:
            data = None
    for key, expected in checks:
        rec = {"json": rel, "key": key, "expected": expected}
        if data is None:
            rec.update({"found": False, "reason": "file missing or unparsable"})
        else:
            found, value, why = json_lookup(data, key)
            rec.update({"found": found, "value": value, "reason": why})
            if found:
                if isinstance(expected, (int, float)) and isinstance(value, (int, float)) \
                        and not isinstance(expected, bool):
                    rec["matches"] = abs(value - expected) <= max(1e-9, 0.02 * abs(expected))
                else:
                    rec["matches"] = value == expected
        out.append(rec)
    return out


def census(root: Path) -> dict:
    """Recount the README's self-census with the repo's own definition
    (module-level ``test_*`` functions and ``unittest.TestCase`` subclasses; see
    ``scripts/readme_api_audit.py:_test_style``)."""
    tests = root / "tests"
    files = sorted(tests.glob("test_*.py")) if tests.is_dir() else []
    module_level = cases = 0
    for f in files:
        try:
            tree = ast.parse(_read(f))
        except SyntaxError:  # pragma: no cover
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                module_level += 1
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                    getattr(b, "attr", getattr(b, "id", "")) == "TestCase" for b in node.bases):
                cases += 1
    return {"test_files": len(files), "module_level_test_functions": module_level,
            "unittest_testcase_classes": cases}


_ORDER = {"REAL": 0, "REAL_REGENERABLE": 1, "STALE_NUMBER": 2, "WITHHELD": 3,
          "NO_TEST": 4, "GATE_DRIFT": 5, "SELF_CONTRADICTION": 6, "NOT_SUPPORTED": 7,
          "RETIRED": 8, "NOT_AUDITED": 0}


def _sev(v: str) -> int:
    return _ORDER.get(v, 9)


# ---------------------------------------------------------------------------
# per-claim evaluation
# ---------------------------------------------------------------------------


def evaluate(root: Path, claim: Claim, run_entrypoints: bool = False) -> dict:
    text = _read(root / "README.md") if (root / "README.md").exists() else ""
    rec: dict = {"cid": claim.cid, "repo": claim.repo, "section": claim.section,
                 "anchor": claim.anchor, "baseline": claim.baseline, "note": claim.note}
    rec["anchor_present"] = claim.anchor in text

    # code + symbols
    rec["code_missing"] = [p for p in claim.code if not (root / p).exists()]
    sym_fail = []
    for spec in claim.symbols:
        rel, _, name = spec.partition("::")
        ok, payload, why = top_level_names(root, rel)
        if not ok:
            sym_fail.append(f"{rel}: {why}")
            continue
        names, imported = payload
        if name and name not in names:
            sym_fail.append(f"{rel}: '{name}' not defined here"
                            + (" (imported only)" if name in imported else " (absent)"))
    rec["symbols_failed"] = sym_fail

    # tests
    rec["tests_missing"] = [p for p in claim.tests if not (root / p).exists()]
    rec["tests_referencing_code"] = (test_files_referencing(root, claim.test_must_reference)
                                     if claim.test_must_reference else [])

    # tracked evidence vs generated evidence
    rec["tracked_missing"] = [p for p in (*claim.docs, *claim.schemas) if not (root / p).exists()]
    rec["artifacts_missing"] = [p for p in claim.artifacts if not (root / p).exists()]
    rec["generators_missing"] = [p for p in claim.generators if not (root / p).exists()]
    rec["generators"] = {}
    if run_entrypoints and claim.generators:
        for g in claim.generators:
            if not (root / g).exists():
                rec["generators"][g] = "missing"
                continue
            try:
                proc = subprocess.run([sys.executable, g, "--help"], cwd=root,
                                      capture_output=True, text=True, timeout=120)
                rec["generators"][g] = ("help-ok" if proc.returncode == 0
                                        else f"help-exit={proc.returncode}")
            except subprocess.TimeoutExpired:
                rec["generators"][g] = "helped-timeout"

    # numbers
    nums = []
    for literal, src in claim.numbers:
        in_readme = literal in text
        src_has = (root / src).exists() and literal in _read(root / src)
        entry = {"literal": literal, "declared_source": src, "in_readme": in_readme,
                 "in_declared_source": src_has}
        if in_readme and not src_has:
            entry["elsewhere"] = repo_grep(root, literal)
        nums.append(entry)
    rec["numbers"] = nums
    rec["numbers_unbacked"] = [n["literal"] for n in nums
                               if n["in_readme"] and not n["in_declared_source"]
                               and not n.get("elsewhere")]

    # gates
    gate_recs = []
    for rel, key, expected in claim.gates:
        g = read_gate(root, rel, key)
        g.update({"json": rel, "gate": key})
        if g.get("found") and isinstance(expected, (int, float)):
            meas = g.get("measured")
            g["expected_measured"] = expected
            g["matches_expected"] = (
                isinstance(meas, (int, float))
                and abs(meas - expected) <= max(1e-12, 0.02 * abs(expected)))
        gate_recs.append(g)
    rec["gates"] = gate_recs
    rec["gate_failures"] = [f"{g['json']}#{g['gate']}" for g in gate_recs
                            if not g.get("found") or g.get("matches_expected") is False]

    # machine-readable JSON checks (numbers quoted by README vs the artifact they come from)
    rec["json_checks"] = []
    for rel in sorted({j for j, _k, _e in claim.json_checks}):
        rec["json_checks"] += run_json_checks(
            root, rel, [(k, e) for j, k, e in claim.json_checks if j == rel])
    rec["qualifications"] = []
    for rel in sorted({j for j, _k, _e in claim.qualifications}):
        rec["qualifications"] += run_json_checks(
            root, rel, [(k, e) for j, k, e in claim.qualifications if j == rel])
    rec["json_check_failures"] = [f"{c['json']}#{c['key']}" for c in rec["json_checks"]
                                  if not c.get("found") or c.get("matches") is False]

    # contradiction with a denying sentence in the same README
    rec["contradiction"] = None
    if claim.counterclaim and claim.counterclaim in text and rec["anchor_present"]:
        if not rec["code_missing"] and not rec["tracked_missing"]:
            rec["contradiction"] = claim.counterclaim

    # census drift
    rec["census"] = None
    if claim.census:
        rec["census"] = census_drift(root, text)

    # verdict
    if not rec["anchor_present"]:
        verdict = "RETIRED"
    elif rec["code_missing"] or rec["symbols_failed"]:
        verdict = "NOT_SUPPORTED"
    elif rec["contradiction"]:
        verdict = "SELF_CONTRADICTION"
    elif (rec["census"] is not None and rec["census"]["drift"]
          and not rec["census"]["dated_snapshot"]):
        verdict = "STALE_NUMBER"
    elif rec["tests_missing"]:
        verdict = "NO_TEST"
    elif rec["tracked_missing"] or rec["numbers_unbacked"] or rec["json_check_failures"]:
        verdict = "WITHHELD"
    elif rec["gate_failures"]:
        verdict = "GATE_DRIFT"
    elif rec["artifacts_missing"] and rec["generators_missing"]:
        verdict = "WITHHELD"
    elif rec["artifacts_missing"]:
        verdict = "REAL_REGENERABLE"
    else:
        verdict = "REAL"
    rec["verdict"] = verdict
    rec["regression"] = _sev(verdict) > _sev(claim.baseline)
    return rec


def census_drift(root: Path, text: str) -> dict:
    live = census(root)
    quoted = {}
    for pattern, key in ((r"(\d+)\s*个测试文件", "test_files"),
                         (r"(\d+)\s*个模块级", "module_level_test_functions"),
                         (r"(\d+)\s*个\s*`?unittest\.TestCase`?\s*类", "unittest_testcase_classes")):
        m = re.search(pattern, text)
        if m:
            quoted[key] = int(m.group(1))
    drift = {k: {"quoted": v, "live": live.get(k), "delta": live.get(k, 0) - v}
             for k, v in quoted.items() if live.get(k) != v}
    # A dated snapshot ("2026-11 巡检快照") that has since drifted is honest record-keeping;
    # an undated assertion that no longer matches the tree is a false statement.  Only the
    # latter is allowed to drive the verdict.
    dated = bool(re.search(r"20\d\d[-年]\d{1,2}", text))
    return {"quoted": quoted, "live": live, "drift": drift, "dated_snapshot": dated,
            "ratio_claim_holds": live["module_level_test_functions"] > live["unittest_testcase_classes"]}


def audit(root: Path, sph_root: Path | None, run_entrypoints: bool = False) -> dict:
    claims = [evaluate(root, c, run_entrypoints) for c in DEM_CLAIMS]
    sph_present = bool(sph_root and (sph_root / "README.md").exists())
    if sph_present:
        claims += [evaluate(sph_root, c, run_entrypoints) for c in SPH_CLAIMS]
    else:
        claims += [{"cid": c.cid, "repo": "sph", "verdict": "NOT_AUDITED",
                    "baseline": c.baseline, "regression": False,
                    "note": "external root unavailable"} for c in SPH_CLAIMS]
    counts: dict[str, int] = {}
    for c in claims:
        counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "dem_root": str(root),
        "sph_root": str(sph_root) if sph_root else None,
        "claims": claims,
        "dangling_pointers_dem": dangling_pointers(root, _read(root / "README.md")),
        "dangling_pointers_sph": (dangling_pointers(sph_root, _read(sph_root / "README.md"))
                                  if sph_present else []),
        "verdict_counts": counts,
        "regressions": [c["cid"] for c in claims if c.get("regression")],
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_markdown(result: dict) -> str:
    L = [
        "# README claim ↔ code evidence table (generated, do not edit)",
        "",
        f"schema `{result['schema_version']}` · TensorDEM `{result['dem_root']}` · "
        f"TensorSPH `{result['sph_root']}`",
        "",
        "Verdict key: `REAL` code+test+tracked evidence · `REAL_REGENERABLE` same, but the "
        "numerical evidence is a generated artifact that this repo deliberately does not "
        "version (rebuildable by the registered script) · `STALE_NUMBER` README quotes a "
        "count that no longer matches the tree · `WITHHELD` a cited tracked path or a quoted "
        "number cannot be found anywhere · `NO_TEST` claimed without a registered test file · "
        "`GATE_DRIFT` machine-readable criterion no longer matches what README quotes · "
        "`SELF_CONTRADICTION` README both claims and denies · `NOT_SUPPORTED` claimed but code "
        "missing · `RETIRED` no longer advertised in README.",
        "",
        "| id | repo | verdict | baseline | code | symbols | tests | tracked evid. | generated evid. | numbers | gates |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in result["claims"]:
        if c["verdict"] == "NOT_AUDITED":
            L.append(f"| `{c['cid']}` | {c['repo']} | _not audited_ | {c['baseline']} | – | – | – | – | – | – | – |")
            continue
        nums = c.get("numbers", [])
        nbacked = sum(1 for n in nums if n["in_readme"] and n["in_declared_source"])
        gates = c.get("gates", [])
        gok = f"{sum(1 for g in gates if g.get('found') and g.get('matches_expected') in (True, None))}/{len(gates)}"
        L.append(
            f"| `{c['cid']}` | {c['repo']} | **{c['verdict']}** | {c['baseline']} | "
            f"{'✓' if not c['code_missing'] else '✗'} | "
            f"{'✓' if not c['symbols_failed'] else '✗'} | "
            f"{'✓' if not c['tests_missing'] else '✗'} | "
            f"{'✓' if not c['tracked_missing'] else '✗' + str(len(c['tracked_missing']))} | "
            f"{'✓' if not c.get('artifacts_missing') else 'pending ' + str(len(c['artifacts_missing']))} | "
            f"{nbacked}/{len(nums) if nums else '–'} | "
            f"{gok if gates else '–'} |")
    L += ["", "## Verdict counts", ""]
    L += [f"- `{k}`: {v}" for k, v in sorted(result["verdict_counts"].items(),
                                             key=lambda kv: _sev(kv[0]))]
    L += ["", "## Dangling README pointers — TensorDEM", ""]
    L += ([f"- `{p}`" for p in result["dangling_pointers_dem"]] or ["- none"])
    if result["dangling_pointers_sph"]:
        L += ["", "## Dangling README pointers — TensorSPH", ""]
        L += [f"- `{p}`" for p in result["dangling_pointers_sph"]]
    L += ["", "## Detail", ""]
    for c in result["claims"]:
        if c["verdict"] == "NOT_AUDITED":
            continue
        L.append(f"### `{c['cid']}` — {c['verdict']}")
        L.append(f"- anchor `{c['anchor']}` present={c['anchor_present']}")
        for key, label in (("code_missing", "code missing"), ("symbols_failed", "symbol check failed"),
                           ("tests_missing", "test files missing"),
                           ("tracked_missing", "tracked evidence missing"),
                           ("artifacts_missing", "generated evidence missing (not in VCS by convention)"),
                           ("generators_missing", "generator missing")):
            if c.get(key):
                L.append(f"- {label}: " + ", ".join(f"`{x}`" for x in c[key]))
        if c.get("generators"):
            L.append("- generator entry points: " + ", ".join(
                f"`{k}` → {v}" for k, v in c["generators"].items()))
        if c.get("tests_referencing_code") and c.get("test_must_reference"):
            tr = c["tests_referencing_code"]
            L.append(f"- tests referencing `{c.get('test_must_reference')}`: "
                     + ", ".join(f"`{x}`" for x in tr[:3]) + (f" (+{len(tr)-3})" if len(tr) > 3 else ""))
        for n in c.get("numbers", []):
            extra = f", found elsewhere: {', '.join('`'+e+'`' for e in n['elsewhere'][:3])}" if n.get("elsewhere") else ""
            L.append(f"- number `{n['literal']}`: README={n['in_readme']}, "
                     f"declared source `{n['declared_source']}`={n['in_declared_source']}{extra}")
        for g in c.get("gates", []):
            if not g.get("found"):
                L.append(f"- gate `{g['gate']}`: **not found** ({g.get('reason')})")
            else:
                L.append(f"- gate `{g['gate']}`: measured {g.get('measured')} vs expected "
                         f"{g.get('expected_measured')} → "
                         f"{'match' if g.get('matches_expected') else '**drift**'}; "
                         f"passed={g.get('passed')}, all_recomputed_pass={g.get('all_recomputed_pass')}")
        for chk in c.get("json_checks", []):
            if not chk.get("found"):
                L.append(f"- json check `{chk['json']}#{chk['key']}`: **not found** ({chk.get('reason')})")
            else:
                L.append(f"- json check `{chk['json']}#{chk['key']}`: value {chk.get('value')} vs "
                         f"README-implied {chk.get('expected')} → "
                         f"{'match' if chk.get('matches') else '**mismatch**'}")
        for chk in c.get("qualifications", []):
            if chk.get("found"):
                L.append(f"- qualification `{chk['json']}#{chk['key']}` = {chk.get('value')} "
                         f"(recorded, expected {chk.get('expected')}; "
                         f"{'as registered' if chk.get('matches') else '**changed**'})")
        if c.get("contradiction"):
            L.append(f"- **contradicted by**: “{c['contradiction']}”")
        if c.get("census"):
            cq = c["census"]
            for k, v in cq["quoted"].items():
                live = cq["live"].get(k)
                tag = "match" if k not in cq["drift"] else (
                    "drift, but the sentence is date-stamped (info only)"
                    if cq["dated_snapshot"] else "**drift**")
                L.append(f"- census `{k}`: quoted {v} / live {live} → {tag}")
            L.append(f"- census ratio claim (module-level `test_*` > `TestCase`): "
                     f"{'holds' if cq['ratio_claim_holds'] else '**violated**'}")
        if c.get("note"):
            L.append(f"- note: {c['note']}")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--sph-root", default="/root/tensor-ecosystem/TensorSPH")
    ap.add_argument("--no-external", action="store_true")
    ap.add_argument("--check-entrypoints", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root).resolve()
    sph_root = None if args.no_external else Path(args.sph_root)
    result = audit(root, sph_root, args.check_entrypoints)
    out_dir = Path(args.out_dir) if args.out_dir else root / "artifacts" / "readme_claim_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(result)
    (out_dir / "claims.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    (out_dir / "CLAIM_EVIDENCE_TABLE.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"[written] {out_dir / 'claims.json'}")
    print(f"[written] {out_dir / 'CLAIM_EVIDENCE_TABLE.md'}")
    if args.strict and result["regressions"]:
        print("[strict] regressions vs baseline: " + ", ".join(result["regressions"]),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())