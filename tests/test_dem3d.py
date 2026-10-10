import csv
import math
import unittest
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import torch
from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.dem3d_cli import main as dem3d_main


class DEM3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, **kwargs):
        return IceDEM3D(DEM3DConfig(nx=3, ny=3, nz=2, drag=0.0,
                                    fix_x_edges=False, **kwargs))

    def test_initial_equilibrium_and_force_balance(self):
        sim = self.model()
        force, reaction = sim.forces()
        self.assertTrue(torch.isfinite(force).all())
        torch.testing.assert_close(force.sum(0), -reaction, atol=1e-10, rtol=0)
        self.assertLess(float(force.abs().max()), 1e-8)

    def test_rigid_rotation_does_not_break_bonds(self):
        sim = self.model()
        a, b = 0.41, -0.27
        rx = torch.tensor([[1., 0., 0.], [0., math.cos(a), -math.sin(a)],
                           [0., math.sin(a), math.cos(a)]], dtype=torch.float64)
        ry = torch.tensor([[math.cos(b), 0., math.sin(b)], [0., 1., 0.],
                           [-math.sin(b), 0., math.cos(b)]], dtype=torch.float64)
        sim.positions = sim.initial_positions @ (rx @ ry).T + torch.tensor([2., -3., 4.])
        force, _ = sim.forces()
        self.assertEqual(sim.broken_bonds, 0)
        self.assertLess(float(force.abs().max()), 1e-7)

    def test_tension_failure_is_irreversible(self):
        sim = self.model(breaking_strain=0.01, shear_breaking_strain=0.5)
        sim.positions = sim.initial_positions * 1.03
        sim.forces()
        self.assertGreater(sim.broken_bonds, 0)
        broken = sim.broken_bonds
        sim.positions = sim.initial_positions.clone()
        sim.forces()
        self.assertEqual(sim.broken_bonds, broken)

    def test_external_load_shape_and_finiteness(self):
        sim = self.model()
        with self.assertRaises(ValueError):
            sim.forces(torch.zeros((len(sim.positions), 2)))
        loads = torch.zeros_like(sim.positions)
        loads[0, 0] = 1.0
        force, _ = sim.forces(loads)
        self.assertTrue(torch.isfinite(force).all())
        with self.assertRaises(ValueError):
            loads[0, 0] = float("nan")
            sim.forces(loads)

    def test_step_fixed_boundary_and_finite_state(self):
        sim = IceDEM3D(DEM3DConfig(nx=4, ny=3, nz=2))
        fixed0 = sim.positions[sim.fixed].clone()
        for _ in range(20):
            sim.step()
        torch.testing.assert_close(sim.positions[sim.fixed], fixed0, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(sim.positions).all())
        self.assertEqual(sim.step_count, 20)

    def test_restart_snapshot_and_validation(self):
        sim = self.model()
        snap = sim.state_dict()
        self.assertEqual(snap["dimension"], 3)
        self.assertEqual(tuple(snap["positions"].shape), (18, 3))
        with self.assertRaises(ValueError):
            DEM3DConfig(nx=1)
        with self.assertRaises(ValueError):
            DEM3DConfig(dt=math.nan)


    def test_cell_list_covers_all_overlapping_pairs_without_duplicates(self):
        sim = self.model()
        # Force a non-grid configuration with several deliberately overlapping
        # pairs, then compare the broad-phase list against brute-force truth.
        generator = torch.Generator().manual_seed(73)
        sim.positions = torch.randn(sim.positions.shape, generator=generator,
                                    dtype=torch.float64) * 0.035
        candidates = sim._contact_candidate_pairs()
        keys = candidates[:, 0] * len(sim.positions) + candidates[:, 1]
        self.assertEqual(len(keys), len(torch.unique(keys)))
        delta = sim.positions[:, None, :] - sim.positions[None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        truth = torch.triu(distance < 2 * sim.config.radius, diagonal=1).nonzero()
        candidate_keys = set(keys.tolist())
        truth_keys = (truth[:, 0] * len(sim.positions) + truth[:, 1]).tolist()
        self.assertTrue(set(truth_keys).issubset(candidate_keys))

    def test_broken_bond_is_still_eligible_for_contact(self):
        sim = self.model()
        # Move a bonded pair into overlap and mark its bond broken. The cell
        # list must still find the pair and contact lookup must not call it alive.
        pair = sim.pairs[0].clone()
        i, j = int(pair[0]), int(pair[1])
        sim.positions[j] = sim.positions[i] + torch.tensor(
            [sim.config.radius, 0.0, 0.0], dtype=torch.float64)
        sim.alive[0] = False
        candidates = sim._contact_candidate_pairs()
        keys = candidates[:, 0] * len(sim.positions) + candidates[:, 1]
        key = i * len(sim.positions) + j
        self.assertIn(key, keys.tolist())
        idx = (candidates[:, 0] == i) & (candidates[:, 1] == j)
        self.assertFalse(bool(sim._alive_bond_for_pairs(candidates[idx]).any()))

    def test_energy_diagnostics_and_cli_restart_workflow(self):
        sim = self.model()
        row = sim.diagnostics()
        for key in ("kinetic_energy_J", "bond_elastic_energy_J",
                    "particle_contact_energy_J", "tool_contact_energy_J",
                    "released_bond_energy_J", "mechanical_energy_J"):
            self.assertIn(key, row)
            self.assertTrue(math.isfinite(row[key]))
            self.assertGreaterEqual(row[key], 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with redirect_stdout(StringIO()):
                dem3d_main(["--nx", "3", "--ny", "3", "--nz", "2",
                            "--steps", "2", "--save-every", "1",
                            "--output", str(output)])
            checkpoint_path = output / "checkpoint_3d.pt"
            self.assertTrue(checkpoint_path.is_file())
            first = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(first["state"]["step_count"], 2)
            with redirect_stdout(StringIO()):
                dem3d_main(["--steps", "2", "--save-every", "1",
                            "--output", str(output), "--restart", str(checkpoint_path)])
            second = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(second["state"]["step_count"], 4)
            self.assertAlmostEqual(second["state"]["time"], 4 * second["state"]["dt"])

    def test_checkpoint_round_trip_continues_identically(self):
        first = self.model()
        for _ in range(5):
            first.step()
        snapshot = first.state_dict()
        for _ in range(5):
            first.step()

        resumed = self.model()
        resumed.load_state_dict(snapshot)
        for _ in range(5):
            resumed.step()
        torch.testing.assert_close(resumed.positions, first.positions, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(resumed.velocities, first.velocities, atol=1e-12, rtol=1e-12)
        self.assertEqual(resumed.step_count, first.step_count)
        self.assertEqual(resumed.time, first.time)
        self.assertTrue(torch.equal(resumed.alive, first.alive))

    def test_checkpoint_rejects_incompatible_or_corrupt_state(self):
        sim = self.model()
        snapshot = sim.state_dict()
        broken = dict(snapshot)
        broken["positions"] = torch.zeros((1, 3))
        with self.assertRaises(ValueError):
            sim.load_state_dict(broken)
        incompatible = IceDEM3D(DEM3DConfig(nx=4, ny=3, nz=2, drag=0.0,
                                             fix_x_edges=False))
        with self.assertRaises(ValueError):
            incompatible.load_state_dict(snapshot)

    def test_diagnostics_is_pure_even_when_pending_fracture_exceeds_threshold(self):
        sim = self.model()
        sim.positions = sim.initial_positions * 1.03
        sim.last_reaction = torch.tensor([1., 2., 3.], dtype=torch.float64)
        before = sim.state_dict()
        reaction_before = sim.last_reaction.clone()
        for _ in range(3):
            row = sim.diagnostics()
            self.assertEqual(row['broken_bonds'], 0)
            self.assertGreater(row['bond_elastic_energy_J'], 0)
        after = sim.state_dict()
        for key, value in before.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, after[key]), key)
            else:
                self.assertEqual(value, after[key], key)
        self.assertTrue(torch.equal(sim.last_reaction, reaction_before))
        sim.forces()  # Existing default API still commits the pending fracture.
        self.assertGreater(sim.broken_bonds, 0)

    def test_observation_frequency_preserves_fracture_history_and_force_evolution(self):
        unobserved, observed = self.model(), self.model()
        # Start below failure, then apply a reproducible deformation schedule.
        for scale in (1.005, 1.01, 1.03, 1.0, 1.04):
            for sim in (unobserved, observed):
                sim.positions = sim.initial_positions * scale
            for _ in range(5):
                observed.diagnostics()
            unobserved.step(); observed.step()
            for key in ('positions', 'velocities', 'alive', 'failure_mode',
                        'failure_time_s', 'failure_extension_m', 'failure_energy_J'):
                self.assertTrue(torch.equal(getattr(unobserved, key), getattr(observed, key)), key)
            first = unobserved.forces(update_fracture=False)
            second = observed.forces(update_fracture=False)
            for a, b in zip(first, second):
                self.assertTrue(torch.equal(a, b))
        self.assertGreater(unobserved.broken_bonds, 0)

    def test_checkpoint_dtype_corruption_rejected_atomically_before_conversion(self):
        sim = self.model()
        original = sim.state_dict()
        corruptions = {
            'alive': torch.full(sim.alive.shape, float('nan'), dtype=torch.float64),
            'pairs': sim.pairs.to(torch.float64) + .2,
            'failure_mode': sim.failure_mode.to(torch.int64) + 256,
            'positions': sim.positions.to(torch.float32),
        }
        for key, tensor in corruptions.items():
            corrupt = dict(original); corrupt[key] = tensor
            with self.assertRaisesRegex(ValueError, 'dtype'):
                sim.load_state_dict(corrupt)
            after = sim.state_dict()
            for field, value in original.items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, after[field]), field)
                else:
                    self.assertEqual(value, after[field], field)


    def test_moving_plane_platens_report_action_reaction_and_energy(self):
        sim = self.model(
            top_platen_enabled=True, bottom_platen_enabled=True,
            platen_stiffness=4000.0, platen_damping=0.0,
            top_platen_velocity=-0.1, bottom_platen_velocity=0.1,
        )
        sim.time = 0.01
        force, _ = sim.forces(update_fracture=False)
        self.assertTrue(torch.isfinite(force).all())
        self.assertGreater(float(sim.last_top_platen_reaction_z), 0.0)
        self.assertLess(float(sim.last_bottom_platen_reaction_z), 0.0)
        row = sim.diagnostics()
        self.assertGreater(row["top_platen_reaction_z_N"], 0.0)
        self.assertLess(row["bottom_platen_reaction_z_N"], 0.0)
        self.assertGreater(row["platen_contact_energy_J"], 0.0)
        self.assertTrue(math.isfinite(row["mechanical_energy_J"]))

    def test_plane_platen_configuration_validation(self):
        with self.assertRaises(ValueError):
            self.model(platen_stiffness=0.0)
        with self.assertRaises(ValueError):
            self.model(top_platen_gap=-1e-3)
        with self.assertRaises(ValueError):
            self.model(bottom_platen_velocity=float("nan"))



    def test_checkpoint_without_new_platen_keys_remains_loadable(self):
        original = self.model()
        snapshot = original.state_dict()
        for key in (
            "top_platen_enabled", "bottom_platen_enabled", "platen_stiffness",
            "platen_damping", "top_platen_gap", "bottom_platen_gap",
            "top_platen_velocity", "bottom_platen_velocity",
        ):
            snapshot["config"].pop(key, None)
        resumed = self.model()
        resumed.load_state_dict(snapshot)
        torch.testing.assert_close(resumed.positions, original.positions, atol=0, rtol=0)

    def test_cli_can_run_with_moving_plane_platens(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "platen-cli"
            with redirect_stdout(StringIO()):
                dem3d_main([
                    "--nx", "3", "--ny", "3", "--nz", "2",
                    "--steps", "2", "--save-every", "1",
                    "--top-platen", "--bottom-platen",
                    "--top-platen-velocity", "-0.05",
                    "--bottom-platen-velocity", "0.05",
                    "--output", str(output),
                ])
            with (output / "history.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertIn("top_platen_reaction_z_N", rows[0])
            self.assertIn("bottom_platen_reaction_z_N", rows[0])
            self.assertIn("platen_contact_energy_J", rows[0])

if __name__ == "__main__":
    unittest.main()
