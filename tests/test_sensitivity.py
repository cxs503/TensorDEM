import math
import unittest

from tensordem.sensitivity import build_config, simulate_case


class SensitivitySweepTests(unittest.TestCase):
    def test_resolution_keeps_width_fixed_and_records_discretized_height(self):
        coarse = build_config(5, width=0.3, thickness=0.2)
        fine = build_config(9, width=0.3, thickness=0.2)
        self.assertAlmostEqual(2 * coarse.radius * (coarse.nx - 1), 0.3)
        self.assertAlmostEqual(2 * fine.radius * (fine.nx - 1), 0.3)
        self.assertNotEqual(coarse.radius, fine.radius)
        self.assertGreaterEqual(coarse.ny, 2)
        self.assertGreaterEqual(fine.ny, 2)

    def test_invalid_resolution_is_rejected(self):
        for nx in (True, 2, 2.5, 0):
            with self.subTest(nx=nx), self.assertRaises(ValueError):
                build_config(nx)

    def test_short_case_produces_finite_summary_and_history(self):
        config = build_config(3, width=0.2, thickness=0.1, speed=0.2)
        summary, history = simulate_case(config, duration=1e-7)
        self.assertEqual(summary["steps"], 1)
        self.assertEqual(len(history), 2)
        for value in summary.values():
            if isinstance(value, (float, int)):
                self.assertTrue(math.isfinite(float(value)))
        self.assertGreaterEqual(summary["final_broken_bonds"], 0)
        self.assertGreaterEqual(summary["peak_abs_reaction_y_N"], 0.0)

    def test_nonpositive_duration_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate_case(build_config(3), duration=0.0)


if __name__ == "__main__":
    unittest.main()
