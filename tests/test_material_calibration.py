import math
import unittest
import torch
from tensordem.dem import DEMConfig, IceDEM
from tensordem.material_calibration import lattice, energy, qualify, relaxed_tension


class MaterialCalibrationTests(unittest.TestCase):
    def test_actual_dem_topology_and_axial_energy(self):
        x, bonds, _ = lattice(2)
        case = qualify(2)
        dem = IceDEM(DEMConfig(nx=5, ny=3, radius=0.1, thickness=0.2,
                    bond_stiffness=case['stiffness_N_per_m'], breaking_strain=1,
                    shear_breaking_strain=1, contact_damping=0, drag=0, fix_edges=False,
                    tool_gap=10))
        actual = {tuple(pair) for pair in dem.bond_indices.tolist()}
        expected = {tuple(sorted(pair)) for pair in bonds.tolist()}
        self.assertEqual(actual, expected)
        displacement = torch.stack((1e-5 * x[:, 0], torch.zeros(len(x))), 1)
        dem.positions += displacement
        delta = dem.positions[dem.bond_indices[:, 1]] - dem.positions[dem.bond_indices[:, 0]]
        extension = torch.linalg.vector_norm(delta, dim=1) - dem.rest_lengths[dem.bonded]
        nonlinear = float(0.5 * dem.config.bond_stiffness * extension.square().sum())
        linear = energy(x, bonds, displacement, dem.config.bond_stiffness)
        self.assertLess(abs(nonlinear / linear - 1), 1e-5)

    def test_fixed_physical_mass_and_material_failure(self):
        cases = [qualify(n) for n in (4, 8, 16)]
        for case in cases:
            self.assertAlmostEqual(case['mass_kg'], 917 * 0.8 * 0.4 * 0.2)
            self.assertLess(case['relative_errors']['C11'], 1e-12)
            self.assertFalse(case['material_qualified'])
        self.assertGreater(cases[-1]['relative_errors']['shear'], cases[0]['relative_errors']['shear'])
        self.assertLess(cases[-1]['relative_errors']['bending'], cases[0]['relative_errors']['bending'])

    def test_rigid_motion_energy(self):
        x, bonds, _ = lattice(3)
        translation = torch.ones_like(x)
        rotation = torch.stack((-x[:, 1], x[:, 0]), 1)
        self.assertEqual(energy(x, bonds, translation, 1000), 0)
        self.assertLess(energy(x, bonds, rotation, 1000), 1e-27)

    def test_free_transverse_equilibrium(self):
        case = relaxed_tension(4)
        self.assertLess(case['free_dof_residual_N'], 1e-10)
        self.assertGreater(case['E_relative_error'], 0.1)
        self.assertGreater(case['apparent_nu'], 0.4)

    def test_invalid_inputs(self):
        for n in (True, 1, 2.5):
            with self.assertRaises(ValueError):
                lattice(n)
        with self.assertRaises(ValueError):
            qualify(4, poisson_ratio=0.5)
        with self.assertRaises(ValueError):
            lattice(4, density=math.nan)


if __name__ == '__main__':
    unittest.main()
