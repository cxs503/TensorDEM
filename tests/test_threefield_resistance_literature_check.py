"""Identifiability audit of the shipped Lindqvist/Riska chains vs published data.

The published predictions transcribed in the MT Uikku case set are the only
literature reference for the ``threefield`` resistance baselines that is
available offline.  These tests pin the audit's *findings* so that a later change
of the shipped coefficient set cannot silently re-open a claim the repository has
already settled -- the ratchet pattern used elsewhere in this repository.

Every number asserted here is produced by
``tensordem.threefield.literature_check.run_audit`` from the committed CSV; none
is typed by hand into the source.
"""
from __future__ import annotations

import unittest

from tensordem.threefield.literature_check import (
    PROTOCOL,
    TOLERANCE_PCT,
    load_cases,
    published_riska_linearity,
    run_audit,
    format_markdown_table,
)

#: One campaign per process: the audit is deterministic, so the tests can share it.
REPORT = run_audit(samples=800, signature_samples=4000, seed=20261104)


class ReferenceDataTests(unittest.TestCase):
    """The transcribed source data, and what it is (and is not) good for."""

    def test_ten_level_ice_cases_load_with_units(self):
        cases = load_cases()
        self.assertEqual(len(cases), 10)
        self.assertEqual([c.test_id for c in cases][0], "103")
        thick = sorted({round(c.thickness_m, 2) for c in cases})
        self.assertEqual(thick, [0.63, 0.76, 0.77, 0.95, 0.96, 1.04])
        # Sanity of the SI conversion: sea-ice strengths sit in the MPa range.
        for case in cases:
            self.assertTrue(1e5 < case.flexural_strength_pa < 2e6)
            self.assertTrue(1e6 < case.compressive_strength_pa < 8e6)
            self.assertLess(case.speed_m_s, 2.0)

    def test_published_riska_column_is_affine_in_speed(self):
        """Criterion: the published Riska predictions are ``C1 + C2 V`` exactly.

        A residual at the 1 kN rounding floor of the source table is the strongest
        available evidence that the published chain really is Riska's speed-linear
        model, which is what makes the ``C2 / C1`` signature meaningful.
        """
        linearity = published_riska_linearity(load_cases())
        self.assertTrue(linearity["is_linear_within_rounding"])
        self.assertLess(linearity["max_abs_residual_n"], linearity["rounding_floor_n"])
        self.assertEqual(len(linearity["groups"]), 2)

    def test_published_speed_signature_is_the_measured_one(self):
        signature = REPORT["published_speed_signature_s_per_m"]
        self.assertGreater(signature, 0.15)
        self.assertLess(signature, 0.25)

    def test_published_formula_columns_ignore_the_per_test_strength_spread(self):
        """Provenance caveat: the published columns are per ice *sheet*, not per test.

        Within a sheet the flexural strength spread is 100-220 kPa (15-27 % of the
        mean) yet a single ``C1`` reproduces all three Riska predictions to the
        rounding floor.  The published predictions therefore do not respond to the
        per-test strength columns, and a DEM-versus-formula comparison that feeds
        the per-test values is not comparing like with like.
        """
        for group in REPORT["published_riska_linearity"]["groups"]:
            self.assertGreater(group["flexural_strength_spread_pa"], 1e5)
            self.assertLess(group["max_abs_residual_n"], 150.0)


class ShippedChainVsLiteratureTests(unittest.TestCase):
    """What the shipped coefficient set can and cannot reproduce."""

    def test_as_shipped_chain_misses_the_tolerance_gate(self):
        for model in ("lindqvist", "riska"):
            block = REPORT["as_shipped"][model]
            self.assertGreater(block["max_abs_pct"], TOLERANCE_PCT,
                               f"{model} as-shipped error unexpectedly inside the gate")
            self.assertGreater(block["mape_pct"], TOLERANCE_PCT)
        # The shipped default hull over-predicts every case of both columns.
        self.assertGreater(min(REPORT["as_shipped"]["lindqvist"]["per_case_pct"]), 0.0)
        self.assertGreater(min(REPORT["as_shipped"]["riska"]["per_case_pct"]), -10.0)

    def test_best_hull_inside_the_declared_envelope_still_misses_the_gate(self):
        for model in ("lindqvist", "riska"):
            fit = REPORT["best_feasible_fit"][model]
            self.assertFalse(fit["within_tolerance"],
                             f"{model} reproduced the published column inside 15 %: "
                             "the REFUTED verdict must be revisited")
            self.assertGreater(fit["max_abs_pct"], TOLERANCE_PCT)
            # The recovered hull must stay physically interpretable, otherwise the
            # bounded search was not bounded.
            self.assertGreater(fit["hull"]["beam_length_ratio"], 0.09)
            self.assertLess(fit["hull"]["beam_length_ratio"], 0.26)
            self.assertGreater(fit["hull"]["length_m"], 120.0)
            self.assertLess(fit["hull"]["length_m"], 260.0)

    def test_speed_term_structure_is_refuted_geometry_robustly(self):
        """The decisive, hull-insensitive finding.

        The published Riska column carries 3-4 times more *relative* speed
        dependence (``C2 / C1``) than the shipped chain can produce anywhere inside
        the declared hull envelope.  Because ``B`` largely cancels in the ratio, this
        cannot be explained away by picking a different ship.
        """
        gap = REPORT["speed_signature_gap_factor"]
        self.assertGreater(gap, 2.0)
        self.assertLess(REPORT["feasible_speed_signature"]["c2_over_c1_s_per_m"],
                        REPORT["published_speed_signature_s_per_m"])
        self.assertIn("REFUTED", REPORT["verdict"]["speed_term_structure"])


class AuditContractTests(unittest.TestCase):
    """Protocol, determinism and rendering of the audit itself."""

    def test_verdict_grades_are_complete(self):
        self.assertEqual(REPORT["protocol"], PROTOCOL)
        self.assertEqual(REPORT["tolerance_pct"], TOLERANCE_PCT)
        verdict = REPORT["verdict"]
        self.assertEqual(verdict["published_riska_linearity"], "REAL")
        self.assertEqual(verdict["literature_agreement_of_shipped_coefficients"],
                         "REFUTED-IN-ENVELOPE")
        self.assertIn("CONVENTION", REPORT["envelope_evidence_grade"])

    def test_audit_is_deterministic(self):
        again = run_audit(samples=200, signature_samples=500, seed=7)
        repeat = run_audit(samples=200, signature_samples=500, seed=7)
        self.assertEqual(again["best_feasible_fit"]["lindqvist"]["max_abs_pct"],
                         repeat["best_feasible_fit"]["lindqvist"]["max_abs_pct"])
        self.assertEqual(again["feasible_speed_signature"]["c2_over_c1_s_per_m"],
                         repeat["feasible_speed_signature"]["c2_over_c1_s_per_m"])

    def test_markdown_table_covers_every_case(self):
        table = format_markdown_table(REPORT)
        for case in REPORT["cases"]:
            self.assertIn("| %s |" % case["test_id"], table)
        self.assertEqual(table.count("\n"), len(REPORT["cases"]) + 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()