from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
from scripts.benchmark_dem3d_three_point_bending import run_three_point_bending

class DEM3DThreePointBendingTests(unittest.TestCase):
    def test_load_transfer_is_finite_and_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a=run_three_point_bending(root/"a",steps=24); b=run_three_point_bending(root/"b",steps=24)
            self.assertEqual(a["verdict"],"PASS",a); self.assertTrue(all(a["checks"].values()))
            self.assertEqual(a["signature_sha256"],b["signature_sha256"])
            self.assertGreater(a["peak_support_reaction_abs_N"],0); self.assertGreater(a["peak_midspan_deflection_abs_m"],0)
            self.assertTrue((root/"a"/"three_point_bending_history.csv").is_file())
            saved=json.loads((root/"a"/"three_point_bending_report.json").read_text())
            self.assertEqual(saved["protocol"],"tensordem-dem3d-three-point-bending-v1"); self.assertEqual(len(saved["rows"]),25)
    def test_invalid_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): run_three_point_bending(Path(tmp)/"zero",steps=0)
            with self.assertRaises(ValueError): run_three_point_bending(Path(tmp)/"negative",peak_load_N=-1)
if __name__=="__main__": unittest.main()
