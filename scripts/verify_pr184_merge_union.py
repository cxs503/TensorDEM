"""Machine audit of the PR #184 (m3129) merge union.

The merge of ``agent/m3129`` into ``main`` had exactly one conflicting file,
``src/tensordem/contact_restitution.py``: both sides added the same functional
line ``contact_damping_model="absolute"`` to :func:`measure_dem3d_impact`, each
carrying its own rationale comment, so git produced a comment-block conflict on
a shared code line.  A mechanical overlay (take ours / take theirs) would have
silently dropped one side's argument; the resolution is a *semantic union*.

This script audited that union at merge time and re-audits it in CI:

1. ``source`` - no conflict markers, exactly one ``contact_damping_model``
   assignment inside ``measure_dem3d_impact``, and both sides' rationale
   keywords survive in the merged comment block.
2. ``deliverables`` - m3129's non-conflicting additions (its verification
   script) are still present and parseable (nothing dropped by the merge).
3. ``physics`` - closed-loop re-verification: the merged module, driven through
   the real 3-D solver, reproduces the clamped analytic impact model.

Every number below is written to ``artifacts/pr184/union_audit.json`` and is read
back from there when the report tables are produced (zero hand-copying).

Run:  python3 scripts/verify_pr184_merge_union.py
"""
from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.contact_restitution import (  # noqa: E402
    damping_for_target_restitution,
    measure_dem3d_impact,
)

TARGET = ROOT / "src/tensordem/contact_restitution.py"
M3129_DELIVERABLE = ROOT / "scripts/verify_restitution_closed_loop.py"

# Rationale keywords: left column = main's argument, right column = m3129's.
MAIN_KEYWORDS = ("clipped", "restitution\" dashpot convention", "contact_damping`")
M3129_KEYWORDS = ("restitution_coefficient", "contact_impact._build_pair", "elastic")

RADIUS = 0.025
DENSITY = 917.0
K_N = 5_000.0
E_TARGET = 0.6
TOLERANCE = 0.03  # 3 % acceptance gate used throughout this repository


def audit_source() -> dict:
    text = TARGET.read_text()
    tree = ast.parse(text)  # SyntaxError would abort the audit, as intended
    markers = [m for m in ("<<<<<<<", "=======", ">>>>>>>") if m in text]

    kwarg_sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg == "contact_damping_model"
    ]
    measure_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "measure_dem3d_impact"
    )
    in_measure = [
        node
        for node in ast.walk(measure_fn)
        if isinstance(node, ast.keyword) and node.arg == "contact_damping_model"
    ]
    conventions = sorted(
        {
            node.value.value
            for node in in_measure
            if isinstance(node.value, ast.Constant)
        }
    )

    # Comment block directly above the kwarg line inside measure_dem3d_impact.
    lines = text.splitlines()
    idx = next(
        i
        for i, line in enumerate(lines)
        if line.strip().startswith("contact_damping_model=")
        and lines[i].strip() == 'contact_damping_model="absolute",'
    )
    comment_lines = []
    j = idx - 1
    while j >= 0 and lines[j].strip().startswith("#"):
        comment_lines.append(lines[j])
        j -= 1
    comment = "\n".join(reversed(comment_lines))

    missing_main = [k for k in MAIN_KEYWORDS if k not in comment]
    missing_m3129 = [k for k in M3129_KEYWORDS if k not in comment]
    return {
        "conflict_markers": markers,
        "contact_damping_model_keywords_total": len(kwarg_sites),
        "contact_damping_model_in_measure": len(in_measure),
        "conventions_in_measure": conventions,
        "comment_lines": len(comment_lines),
        "main_rationale_missing": missing_main,
        "m3129_rationale_missing": missing_m3129,
        "ok": (
            not markers
            and len(in_measure) == 1
            and conventions == ["absolute"]
            and not missing_main
            and not missing_m3129
        ),
    }


def audit_deliverables() -> dict:
    present = M3129_DELIVERABLE.is_file()
    parses = False
    if present:
        try:
            ast.parse(M3129_DELIVERABLE.read_text())
            parses = True
        except SyntaxError:
            parses = False
    return {
        "m3129_script": str(M3129_DELIVERABLE.relative_to(ROOT)),
        "present": present,
        "parses": parses,
        "ok": present and parses,
    }


def audit_physics() -> dict:
    mass = DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3
    m_eff = mass  # second sphere frozen -> infinite-mass wall, m_eff = m
    cal = damping_for_target_restitution(E_TARGET, K_N, m_eff)
    gamma = float(cal["gamma_n"])
    zeta = float(cal["zeta"])
    result = measure_dem3d_impact(gamma, k_n=K_N, radius=RADIUS, density=DENSITY)
    e_solver = float(result["e_measured"])
    e_reference = float(cal["e_reference"])
    rel = abs(e_solver - e_reference) / e_reference
    return {
        "e_target": E_TARGET,
        "gamma_n": gamma,
        "zeta": zeta,
        "e_solver": e_solver,
        "e_reference_clamped": e_reference,
        "rel_vs_reference": rel,
        "within_3pct": rel < TOLERANCE,
        "keys_available": sorted(result.keys()),
        "ok": rel < TOLERANCE,
    }


def main() -> int:
    audit = {
        "pr": 184,
        "branch": "agent/m3129",
        "conflict_files": ["src/tensordem/contact_restitution.py"],
        "resolution": "semantic union: one kwarg, both rationale comments merged",
        "source": audit_source(),
        "deliverables": audit_deliverables(),
        "physics": audit_physics(),
    }
    audit["ok"] = all(
        audit[k]["ok"] for k in ("source", "deliverables", "physics")
    )
    out_dir = ROOT / "artifacts/pr184"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "union_audit.json"
    out_file.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"\nwrote {out_file.relative_to(ROOT)}")
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())