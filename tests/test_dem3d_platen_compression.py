from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_dem3d_platen_compression import run_platen_compression


class DEM3DPlatenCompressionTests(unittest.TestCase):
    def test_opposed_platens_activate_and_emit_finite_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "platen"
            report = run_platen_compression(output, steps=30)
            self.assertEqual(report["verdict"], "PASS", report)
            self.assertTrue(all(report["checks"].values()))
            self.assertGreater(report["peak_top_platen_reaction_N"], 0.0)
            self.assertGreater(report["peak_bottom_platen_reaction_abs_N"], 0.0)
            self.assertTrue((output / "platen_compression_report.json").is_file())
            self.assertTrue((output / "platen_compression_history.csv").is_file())
            saved = json.loads((output / "platen_compression_report.json").read_text())
            self.assertEqual(len(saved["rows"]), 31)
            self.assertEqual(saved["signature_sha256"], report["signature_sha256"])

    def test_nonpositive_step_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_platen_compression(Path(tmp), steps=0)


if __name__ == "__main__":
    unittest.main()
