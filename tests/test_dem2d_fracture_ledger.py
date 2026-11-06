"""Closed-form regression tests for the 2-D fracture ledger channels.

Companion to ``scripts/verify_dem2d_fracture_ledger.py``.  The balance
identity (``tests/test_accounting_ledger_validation.py``) cannot catch a
channel scaled by a wrong constant; these tests anchor
``fracture_release_J`` and ``pair_contact_acquired_J`` to independently coded
closed forms and to a k*xi^2 scaling law.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scripts.verify_dem2d_fracture_ledger import (  # noqa: E402
    case_a_controlled_break,
    case_b_scaling,
    case_c_indentation_events,
    case_d_subthreshold,
    check_provenance,
)


ROOT = Path(__file__).resolve().parents[1]


class Dem2DFractureLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_provenance(self):
        # The engine must come from *this* checkout, not from an editable install
        # pointing at another worktree.  Anchor on the repository root: an earlier
        # version asserted on os.sep + "work" + os.sep, which only probed the
        # runner's mount layout and goes red on runners that check out under
        # ``_work`` (self-hosted CI), for a tree that is perfectly clean.
        resolved = Path(check_provenance()).resolve()
        self.assertEqual(resolved.parent.parent, ROOT / "src", str(resolved))
        self.assertEqual(resolved.name, "__init__.py")

    def test_provenance_rejects_a_foreign_tree(self):
        # Negative control: without it the positive test above could pass on a
        # gate that never fails.
        outside = os.path.join(tempfile.gettempdir(), "elsewhere", "tensordem",
                               "__init__.py")
        with self.assertRaises(RuntimeError):
            check_provenance(outside)

    def test_controlled_break_matches_closed_form(self):
        row = case_a_controlled_break()
        self.assertTrue(row["controlled_bond_broken"])
        # Ledger equals 0.5 * k * xi0^2 to machine precision.
        self.assertLess(row["rel_err_released"], 1e-12)
        # Tension break: no overlap, acquired channel exactly zero.
        self.assertEqual(row["ledger_acquired_J"], 0.0)
        # Recorded extension equals the imposed pre-strain.
        self.assertAlmostEqual(row["recorded_extension_m"], row["xi0"],
                              delta=1e-12)
        # One-step total-energy residual is the O((omega dt)^2) integrator
        # error, bounded by the released energy scale.
        self.assertLess(abs(row["energy_residual_J"]),
                        0.05 * row["closed_released_J"])

    def test_channel_scales_as_k_xi_squared(self):
        rows = case_b_scaling()
        base = rows[0]["ledger_released_J"]
        for row in rows:
            expected = base * row["expected_ratio"]
            self.assertAlmostEqual(row["ledger_released_J"], expected,
                                   delta=1e-12 * max(1.0, expected))

    def test_indentation_events_match_closed_forms(self):
        c = case_c_indentation_events()
        self.assertGreaterEqual(c["n_events"], 1)
        # The audit must be non-vacuous for the acquired channel.
        self.assertGreaterEqual(c["n_overlap_events"], 1)
        self.assertLess(c["rel_err_released"], 1e-12)
        self.assertLess(c["rel_err_acquired"], 1e-12)
        # Per-event recorded energy equals the closed form.
        for e in c["events"]:
            self.assertAlmostEqual(e["recorded_energy_J"],
                                   e["closed_released_J"], delta=1e-15)

    def test_subthreshold_negative_control(self):
        d = case_d_subthreshold()
        self.assertTrue(d["controlled_bond_alive"])
        self.assertEqual(d["ledger_released_J"], 0.0)
        self.assertEqual(d["ledger_acquired_J"], 0.0)


if __name__ == "__main__":
    unittest.main()
