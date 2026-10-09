"""Regression tests for cumulative 3-D DEM energy accounting and benchmark."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from tensordem.dem3d import DEM3DConfig
from tensordem.dem3d_energy import EnergyAuditedIceDEM3D
from scripts.benchmark_dem3d import run_benchmark


class DEM3DEnergyAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return DEM3DConfig(nx=3, ny=3, nz=2, drag=0.0,
                           fix_x_edges=False, tool_speed=0.5)

    def test_cumulative_energy_terms_are_finite_and_nonnegative(self):
        sim = EnergyAuditedIceDEM3D(self.config())
        for _ in range(12):
            sim.step()
        row = sim.diagnostics()
        keys = (
            "tool_work_cumulative_J", "external_work_cumulative_J",
            "drag_dissipation_cumulative_J", "particle_damping_cumulative_J",
            "tool_damping_cumulative_J", "fracture_release_cumulative_J",
            "energy_balance_residual_J", "energy_balance_residual_relative",
        )
        for key in keys:
            self.assertTrue(torch.isfinite(torch.tensor(row[key])), key)
        for key in (
            "drag_dissipation_cumulative_J", "particle_damping_cumulative_J",
            "tool_damping_cumulative_J", "fracture_release_cumulative_J",
        ):
            self.assertGreaterEqual(row[key], 0.0, key)
        self.assertAlmostEqual(
            row["fracture_release_cumulative_J"],
            float(sim.failure_energy_J.sum()), places=12,
        )

    def test_energy_audit_checkpoint_continues_identically(self):
        uninterrupted = EnergyAuditedIceDEM3D(self.config())
        for _ in range(8):
            uninterrupted.step()
        state = uninterrupted.state_dict()
        for _ in range(7):
            uninterrupted.step()

        resumed = EnergyAuditedIceDEM3D(self.config())
        resumed.load_state_dict(state)
        for _ in range(7):
            resumed.step()
        torch.testing.assert_close(resumed.positions, uninterrupted.positions,
                                   atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(resumed.velocities, uninterrupted.velocities,
                                   atol=1e-12, rtol=1e-12)
        self.assertEqual(resumed.cumulative_tool_work_J,
                         uninterrupted.cumulative_tool_work_J)
        self.assertEqual(resumed.cumulative_fracture_release_J,
                         uninterrupted.cumulative_fracture_release_J)
        self.assertEqual(resumed.energy_balance_residual_J,
                         uninterrupted.energy_balance_residual_J)

    def test_benchmark_repeat_signature_and_artifacts(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = run_benchmark(output, config, steps=12, sample_every=3,
                                    verify_repeat=True)
            self.assertTrue(summary["repeat_verified"])
            self.assertEqual(summary["signature_sha256"],
                             summary["repeat_signature_sha256"])
            self.assertTrue((output / "history_3d.csv").is_file())
            self.assertTrue((output / "final_checkpoint_3d.pt").is_file())
            saved = json.loads((output / "summary_3d.json").read_text())
            self.assertEqual(saved["protocol"], "tensordem-dem3d-indenter-v1")
            self.assertEqual(saved["signature_sha256"], summary["signature_sha256"])


if __name__ == "__main__":
    unittest.main()
