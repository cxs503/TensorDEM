"""Regression pin for the semantic-union resolution of PR #184 (m3129).

Background: merging ``agent/m3129`` into ``main`` conflicted on a single file.
Both sides had *added the same functional line* to
``measure_dem3d_impact``::

    contact_damping_model="absolute",

each with its own rationale comment, so git reported a content conflict over the
shared code line.  A mechanical ``--ours`` / ``--theirs`` overlay would have
silently discarded one side's argument (and, if the sides had disagreed on the
value, one side's behaviour).  The merge therefore kept exactly one kwarg and
folded both rationales into one comment block.

These tests fail loudly if a future edit re-introduces a duplicate kwarg,
conflict markers, or drops either parent's rationale, and they re-check that the
surviving convention is the physically correct one (absolute dashpot) by running
the real 3-D solver against the clamped analytic impact model.
"""
from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path

import torch

from tensordem.contact_restitution import (
    clamped_restitution_ratio,
    damping_for_target_restitution,
    measure_dem3d_impact,
)

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src/tensordem/contact_restitution.py"
M3129_SCRIPT = ROOT / "scripts/verify_restitution_closed_loop.py"

# Rationale fingerprints: one from main's comment, two from m3129's comment.
MAIN_RATIONALE = ("clipped", "the reference integration in this module solves")
M3129_RATIONALE = (
    "restitution_coefficient",
    "contact_impact._build_pair",
    "silently make every measured impact",
)

RADIUS = 0.025
DENSITY = 917.0
K_N = 5_000.0


def _measure_fn_keywords():
    tree = ast.parse(TARGET.read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "measure_dem3d_impact"
    )
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.keyword) and node.arg == "contact_damping_model"
    ]


def _kwarg_comment() -> str:
    lines = TARGET.read_text().splitlines()
    idx = next(
        i for i, line in enumerate(lines) if line.strip() == 'contact_damping_model="absolute",'
    )
    out = []
    j = idx - 1
    while j >= 0 and lines[j].strip().startswith("#"):
        out.append(lines[j])
        j -= 1
    return "\n".join(reversed(out))


class MergeUnionTests(unittest.TestCase):
    def test_no_conflict_markers_anywhere_in_module(self):
        text = TARGET.read_text()
        for marker in ("<<<<<<<", "=======", ">>>>>>>"):
            self.assertNotIn(marker, text)

    def test_single_absolute_dashpot_convention(self):
        keywords = _measure_fn_keywords()
        self.assertEqual(len(keywords), 1, "duplicated contact_damping_model kwarg")
        self.assertEqual(keywords[0].value.value, "absolute")

    def test_both_parents_rationale_survive(self):
        comment = _kwarg_comment()
        self.assertTrue(comment.strip(), "explanatory comment block was dropped")
        for needle in MAIN_RATIONALE:
            self.assertIn(needle, comment, f"main's rationale lost: {needle!r}")
        for needle in M3129_RATIONALE:
            self.assertIn(needle, comment, f"m3129's rationale lost: {needle!r}")

    def test_m3129_deliverable_survives_merge(self):
        self.assertTrue(M3129_SCRIPT.is_file(), "m3129 verification script was lost")
        ast.parse(M3129_SCRIPT.read_text())  # SyntaxError => corrupted by merge

    def test_absolute_convention_reproduces_clamped_reference(self):
        torch.set_num_threads(1)
        m_eff = DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3
        cal = damping_for_target_restitution(0.6, K_N, m_eff)
        result = measure_dem3d_impact(
            cal["gamma_n"], k_n=K_N, radius=RADIUS, density=DENSITY
        )
        e_solver = result["e_measured"]
        e_ref = float(cal["e_reference"])
        self.assertAlmostEqual(e_ref, 0.6, places=6)
        rel = abs(e_solver - e_ref) / e_ref
        self.assertLess(rel, 0.03, f"solver vs clamped reference: {rel:.2%}")
        # The legacy "restitution" convention would have ignored the dashpot and
        # returned an elastic impact (e ~ 1.0); pin that this cannot happen.
        self.assertLess(e_solver, 0.9)
        # Same dashpot through the clamped analytic integrator as cross-check.
        zeta = result["zeta"]
        self.assertAlmostEqual(
            clamped_restitution_ratio(cal["gamma_n"], K_N, m_eff)["e_reference"],
            e_ref,
            places=9,
        )
        self.assertAlmostEqual(zeta, cal["zeta"], places=12)


if __name__ == "__main__":
    unittest.main()