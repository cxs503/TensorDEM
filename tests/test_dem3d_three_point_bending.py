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
            self.assertGreater(saved["dt_s"],0); self.assertAlmostEqual(saved["simulated_duration_s"],24*saved["dt_s"],places=15)
    def test_fixed_duration_time_step_refinement_preserves_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            reference=run_three_point_bending(root/"reference",steps=24)
            coarse=run_three_point_bending(root/"coarse",dt_s=reference["dt_s"],duration_s=24*reference["dt_s"])
            fine=run_three_point_bending(root/"fine",dt_s=0.5*coarse["dt_s"],duration_s=24*coarse["dt_s"])
            self.assertEqual(coarse["verdict"],"PASS",coarse)
            self.assertEqual(fine["verdict"],"PASS",fine)
            self.assertAlmostEqual(coarse["simulated_duration_s"],fine["simulated_duration_s"],places=12)
            self.assertEqual(fine["steps"],2*coarse["steps"])
            self.assertAlmostEqual(fine["dt_s"],0.5*coarse["dt_s"],places=15)

    def test_duration_must_be_integer_multiple_of_dt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_three_point_bending(Path(tmp)/"invalid",dt_s=1e-6,duration_s=1.5e-6)

    def test_invalid_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): run_three_point_bending(Path(tmp)/"zero",steps=0)
            with self.assertRaises(ValueError): run_three_point_bending(Path(tmp)/"negative",peak_load_N=-1)
if __name__=="__main__": unittest.main()
