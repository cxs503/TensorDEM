import copy
import unittest
import torch
from tensordem.dem import DEMConfig
from tensordem.calibrated_dem import CalibratedIceDEM
from tensordem.material_calibration import corrected_stiffness


def make(dt=None, threshold=1):
    cfg=DEMConfig(nx=5,ny=3,radius=.1,thickness=.2,bond_stiffness=150000,
                  contact_damping=0,drag=0,fix_edges=False,tool_gap=10,
                  breaking_strain=threshold,shear_breaking_strain=1,dt=dt)
    from tensordem.dem import IceDEM
    proxy=IceDEM(cfg)
    return CalibratedIceDEM(cfg,bond_stiffness=corrected_stiffness(proxy.positions,proxy.bond_indices,1e6,.2))


class CalibratedTests(unittest.TestCase):
    def test_force_energy_and_release(self):
        sim=make(threshold=.001)
        sim.positions[:,0]*=1.01
        old_energy=sim.mechanical_energy()['bond_J']
        forces,_=sim.forces()
        self.assertLess(float(forces.sum(0).abs().max()),1e-10)
        self.assertAlmostEqual(sim.accounting['fracture_release_J'],old_energy,places=10)
        sim.forces()
        self.assertAlmostEqual(sim.accounting['fracture_release_J'],old_energy,places=10)

    def test_energy_gradient(self):
        sim=make()
        sim.positions[6,0]+=.0001
        force,_=sim.forces()
        eps=1e-8
        sim.positions[6,0]+=eps
        plus=sim.mechanical_energy()['bond_J']
        sim.positions[6,0]-=2*eps
        minus=sim.mechanical_energy()['bond_J']
        self.assertAlmostEqual(float(force[6,0]),-(plus-minus)/(2*eps),places=5)

    def test_dynamic_restart_and_bound(self):
        sim=make()
        load=torch.zeros_like(sim.positions);load[:,0]=.01
        for _ in range(5): sim.step(load,load_time=sim.time)
        restored=CalibratedIceDEM.from_snapshot(sim.snapshot())
        for _ in range(5):
            sim.step(load);restored.step(load)
        self.assertEqual(sim.snapshot(),restored.snapshot())
        self.assertLessEqual(sim.dt,sim.config.recommended_dt)
        invalid=copy.deepcopy(sim.snapshot());invalid['bond_stiffness_N_per_m'][0]=float('nan')
        with self.assertRaises(ValueError):CalibratedIceDEM.from_snapshot(invalid)
        invalid=copy.deepcopy(sim.snapshot());invalid['bond_stiffness_N_per_m'][0]*=.5
        with self.assertRaises(ValueError):CalibratedIceDEM.from_snapshot(invalid)
        sim.coefficients[0]*=.5
        with self.assertRaises(ValueError):sim.step(load)


if __name__=='__main__':unittest.main()
