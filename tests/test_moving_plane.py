import json
import unittest
from tensordem.moving_plane_boundary import MovingPlaneNetwork,smooth_plane

class SmoothPlaneTests(unittest.TestCase):
    def test_smooth_motion_endpoints(self):
        self.assertEqual(smooth_plane(0),1.)
        self.assertEqual(smooth_plane(.5),.999)
        self.assertEqual(smooth_plane(1.),.999)
        self.assertLess(abs(smooth_plane(1e-5)-1.),1e-12)

    def test_tool_work_sign_restart_energy(self):
        b=MovingPlaneNetwork(8,2)
        for i in range(200):b.moving_step(b.dt,smooth_plane(i*b.dt),smooth_plane((i+1)*b.dt))
        self.assertGreater(b.tool_work,0)
        r=MovingPlaneNetwork.restore(json.loads(json.dumps(b.snapshot())))
        for i in range(200,300):
            a=smooth_plane(i*b.dt);z=smooth_plane((i+1)*b.dt)
            b.moving_step(b.dt,a,z);r.moving_step(r.dt,a,z)
        self.assertEqual(b.snapshot(),r.snapshot())
        error=b.energy()+b.forces(plane=z)[3]-b.tool_work
        self.assertLess(abs(error)/b.tool_work,.01)

if __name__=='__main__':unittest.main()
