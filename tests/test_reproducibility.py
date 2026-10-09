"""Deterministic regression checks for the 2-D indentation benchmark."""

import unittest
import torch
from tensordem import DEMConfig, IceDEM

class ReproducibleIndentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def run_case():
        config = DEMConfig(nx=7, ny=3, tool_speed=0.2, breaking_strain=0.008, shear_breaking_strain=0.02)
        sim = IceDEM(config)
        history = []
        for step in range(160):
            sim.step()
            if step % 10 == 0 or step == 159:
                diag = sim.diagnostics()
                history.append((diag["time"], diag["reaction_y"], diag["broken_bonds"], diag["kinetic_energy"]))
        return sim.positions.clone(), sim.velocities.clone(), sim.alive.clone(), history

    def test_identical_runs_produce_identical_histories_and_final_state(self):
        first = self.run_case()
        second = self.run_case()
        torch.testing.assert_close(first[0], second[0], atol=0.0, rtol=0.0)
        torch.testing.assert_close(first[1], second[1], atol=0.0, rtol=0.0)
        torch.testing.assert_close(first[2], second[2], atol=0.0, rtol=0.0)
        self.assertEqual(first[3], second[3])

    def test_benchmark_history_is_finite_and_damage_is_irreversible(self):
        _, _, _, history = self.run_case()
        previous_broken = 0
        for time, reaction_y, broken_bonds, kinetic_energy in history:
            values = torch.tensor([time, reaction_y, kinetic_energy], dtype=torch.float64)
            self.assertTrue(bool(torch.isfinite(values).all()))
            self.assertGreaterEqual(broken_bonds, previous_broken)
            self.assertGreaterEqual(kinetic_energy, 0.0)
            previous_broken = broken_bonds

if __name__ == "__main__":
    unittest.main()