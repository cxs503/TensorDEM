"""Integration tests for the complete DEM3D validation campaign runner."""
import json
import tempfile
import unittest
from pathlib import Path

from tensordem.dem3d import DEM3DConfig
from scripts.run_dem3d_validation_campaign import run_validation_campaign


class DEM3DValidationCampaignRunnerTests(unittest.TestCase):
    def test_short_campaign_emits_all_stages_and_hashed_manifest(self):
        config = DEM3DConfig(
            nx=3, ny=3, nz=2, tool_radius=0.03, tool_gap=0.0,
            drag=0.0, contact_damping=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "campaign"
            manifest = run_validation_campaign(
                config, output=output, base_steps=2,
                timestep_factors=(1, 2), resolution_factors=(1, 2),
                sample_every=1,
            )
            self.assertEqual(
                manifest["protocol"], "tensordem-dem3d-validation-campaign-v1"
            )
            self.assertIn(manifest["screening_verdict"], {"PASS", "WARN", "FAIL"})
            expected = (
                "timestep/convergence_3d.json",
                "resolution/resolution_sensitivity_3d.json",
                "cross_resolution/cross_resolution_summary_3d.json",
                "screening/engineering_screen_3d.json",
                "screening/engineering_screen_3d.csv",
            )
            for relative in expected:
                self.assertTrue((output / relative).is_file(), relative)
            saved = json.loads(
                (output / "validation_campaign_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["campaign"]["timestep_factors"], [1, 2])
            self.assertEqual(saved["campaign"]["resolution_factors"], [1, 2])
            artifact_paths = {entry["path"] for entry in saved["artifacts"]}
            self.assertIn("screening/engineering_screen_3d.json", artifact_paths)
            self.assertTrue(all(
                len(entry["sha256"]) == 64 for entry in saved["artifacts"]
            ))

    def test_invalid_campaign_schedule_rejected_before_running(self):
        config = DEM3DConfig(nx=3, ny=3, nz=2)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_validation_campaign(
                    config, output=Path(tmp) / "campaign",
                    base_steps=0, timestep_factors=(1, 2),
                    resolution_factors=(1, 2),
                )
            with self.assertRaises(ValueError):
                run_validation_campaign(
                    config, output=Path(tmp) / "campaign",
                    base_steps=1, timestep_factors=(1, 1),
                    resolution_factors=(1, 2),
                )


if __name__ == "__main__":
    unittest.main()
