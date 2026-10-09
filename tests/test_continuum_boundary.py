import copy
import math
import unittest
import torch
from tensordem.continuum_boundary_dem import BoundaryNetwork


class BoundaryTests(unittest.TestCase):
    def test_invalid_properties(self):
        for args in ((True,2),(8.5,2),(8,1)):
            with self.assertRaises(ValueError):BoundaryNetwork(*args)
        for E in (True,0.,float('nan')):
            with self.assertRaises(ValueError):BoundaryNetwork(8,2,E=E)

    def test_control_mass_and_fixed_span(self):
        for n in (8,16,32):
            b=BoundaryNetwork(n,n//4)
            self.assertAlmostEqual(float(b.mass.sum()),22.5,places=12)
            self.assertEqual(float(b.x[b.right,0].min()),1.)
            self.assertEqual(float(b.x[b.fixed[:,0],0].max()),0.)
            self.assertAlmostEqual(float(b.areas.sum()),.025,places=14)

    def test_affine_plane_stress_E_nu_patch(self):
        for n in (8,16,32):
            b=BoundaryNetwork(n,n//4);eps=.001
            b.u[:,0]=eps*b.x[:,0];b.u[:,1]=-eps/3*b.x[:,1]
            f,_,_,_=b.forces()
            # Summing right traction independently recovers Young modulus.
            right_force=-float(f[b.right,0].sum())
            self.assertAlmostEqual(right_force/(.025*eps),1e5,places=7)
            top=b.x[:,1]==.25
            self.assertLess(abs(float(f[top,1].sum())),1e-11)
            self.assertAlmostEqual(b.energy(),.5*1e5*eps**2*.025,places=12)
            interior=(b.x[:,0]>0)&(b.x[:,0]<1)&(b.x[:,1]>0)&(b.x[:,1]<.25)
            self.assertLess(float(f[interior].abs().max()),1e-11)

    def test_contact_force_distribution_potential(self):
        for n in (8,16):
            b=BoundaryNetwork(n,n//4)
            f,external,_,potential=b.forces(plane=.999)
            self.assertAlmostEqual(-float(external.sum()),125.,places=9)
            self.assertAlmostEqual(potential,.0625,places=12)
            self.assertTrue(torch.equal(b.forces(plane=1.001)[1],torch.zeros_like(b.x)))

    def test_restart_and_dynamic_conservation(self):
        b=BoundaryNetwork(8,2)
        for _ in range(20):b.step(b.dt,100.)
        r=BoundaryNetwork.restore(copy.deepcopy(b.snapshot()))
        for _ in range(20):b.step(b.dt,100.);r.step(r.dt,100.)
        self.assertEqual(b.snapshot(),r.snapshot())
        self.assertLess(float(((b.mass[:,None]*b.v).sum(0)-b.impulse).abs().max()),1e-12)
        self.assertLess(abs(b.energy()-b.work),1e-5)
        with self.assertRaises(ValueError):b.step(2*b.dt)
        invalid=b.snapshot();invalid['u'][0][0]=float('nan')
        with self.assertRaises(ValueError):BoundaryNetwork.restore(invalid)

if __name__=='__main__':unittest.main()
