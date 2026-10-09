"""Mechanics invariants for the staged ice-breaking DEM validation."""

import unittest

import torch

from tensordem import DEMConfig, IceDEM


class MechanicsInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_internal_forces_conserve_resultant_and_moment_with_external_loads(self):
        sim = IceDEM(
            DEMConfig(
                nx=4,
                ny=3,
                drag=0.0,
                fix_edges=False,
                breaking_strain=10.0,
                shear_breaking_strain=10.0,
                tool_gap=10.0,
                tool_speed=0.01,
            )
        )
        # Exercise a deformed configuration while keeping the tool out of range.
        sim.positions[5, 0] += 1.0e-3
        loads = torch.arange(
            2 * len(sim.positions), dtype=torch.float64
        ).reshape(-1, 2) * 0.25 - 1.0

        force, reaction = sim.forces(external_forces=loads)
        torch.testing.assert_close(force.sum(0), loads.sum(0), atol=1e-10, rtol=1e-10)
        self.assertTrue(bool(torch.isfinite(reaction).all()))

        positions = sim.positions
        moment = (positions[:, 0] * force[:, 1] - positions[:, 1] * force[:, 0]).sum()
        load_moment = (
            positions[:, 0] * loads[:, 1] - positions[:, 1] * loads[:, 0]
        ).sum()
        torch.testing.assert_close(moment, load_moment, atol=1e-10, rtol=1e-10)

    def test_support_reaction_is_negative_fixed_node_force_residual(self):
        sim = IceDEM(
            DEMConfig(
                nx=5,
                ny=3,
                drag=0.0,
                breaking_strain=10.0,
                shear_breaking_strain=10.0,
                tool_gap=10.0,
                tool_speed=0.01,
            )
        )
        sim.positions[7, 0] += 5.0e-4
        loads = torch.zeros_like(sim.positions)
        loads[sim.fixed, 0] = 2.0
        loads[2, 1] = 1.5

        force, _ = sim.forces(external_forces=loads)
        expected = -force[sim.fixed].sum(0)
        diag = sim.diagnostics(external_forces=loads)
        torch.testing.assert_close(
            torch.tensor(
                [diag["boundary_reaction_x"], diag["boundary_reaction_y"]],
                dtype=torch.float64,
            ),
            expected,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_zero_external_load_is_identical_to_omitted_load(self):
        sim = IceDEM(
            DEMConfig(
                nx=4,
                ny=3,
                drag=0.0,
                fix_edges=False,
                tool_gap=10.0,
                tool_speed=0.01,
            )
        )
        sim.positions[4, 0] += 2.0e-4
        zero_load = torch.zeros_like(sim.positions)
        force_none, reaction_none = sim.forces()
        force_zero, reaction_zero = sim.forces(external_forces=zero_load)
        torch.testing.assert_close(force_zero, force_none, atol=0.0, rtol=0.0)
        torch.testing.assert_close(reaction_zero, reaction_none, atol=0.0, rtol=0.0)

    def test_halving_timestep_reduces_trajectory_error(self):
        base_config = DEMConfig(
            nx=4,
            ny=3,
            drag=0.0,
            breaking_strain=10.0,
            shear_breaking_strain=10.0,
            tool_gap=10.0,
            tool_speed=0.01,
        )
        coarse_dt = 0.5 * base_config.recommended_dt
        horizon = 8 * coarse_dt
        loads = torch.zeros((base_config.nx * base_config.ny, 2), dtype=torch.float64)
        loads[5, 0] = 0.02

        def final_positions(dt):
            sim = IceDEM(DEMConfig(
                nx=base_config.nx,
                ny=base_config.ny,
                drag=base_config.drag,
                breaking_strain=base_config.breaking_strain,
                shear_breaking_strain=base_config.shear_breaking_strain,
                tool_gap=base_config.tool_gap,
                tool_speed=base_config.tool_speed,
                dt=dt,
            ))
            steps = round(horizon / dt)
            for _ in range(steps):
                sim.step(external_forces=loads)
            self.assertAlmostEqual(sim.time, horizon, places=12)
            return sim.positions

        coarse = final_positions(coarse_dt)
        medium = final_positions(coarse_dt / 2)
        fine = final_positions(coarse_dt / 4)
        coarse_error = torch.linalg.vector_norm(coarse - fine)
        medium_error = torch.linalg.vector_norm(medium - fine)
        self.assertTrue(bool(torch.isfinite(fine).all()))
        self.assertGreater(float(coarse_error), 0.0)
        self.assertLess(float(medium_error), float(coarse_error))


if __name__ == "__main__":
    unittest.main()
