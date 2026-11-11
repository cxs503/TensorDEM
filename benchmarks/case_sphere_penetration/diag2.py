import sys, numpy as np
sys.path.insert(0, "src")
from tensordem.sphere_pack_dem import SpherePack, SpherePackConfig, SphereWall
from tensordem.superquadric_kernel import ContactMaterial
from tensordem.granular3d_core import hcp_positions_in_box

d = 0.004; r = d / 2
Lx = Ly = 24 * d; H = 14 * d
pos = hcp_positions_in_box(r, Lx, Ly, H, packing_gap=0.02, jitter=0.0)
print("N", len(pos), "phi", len(pos)*4/3*np.pi*r**3/(Lx*Ly*H))
walls = [SphereWall(np.array([1.,0,0]), 0.0), SphereWall(np.array([-1.,0,0]), -Lx),
         SphereWall(np.array([0.,1,0]), 0.0), SphereWall(np.array([0.,-1,0]), -Ly),
         SphereWall(np.array([0.,0,1]), 0.0)]
mat = ContactMaterial(e_star=1e9, g_star=4e8, mu=0.4, mu_rolling=0.05, restitution=0.4)
for fac in (1.0, 0.25):
    cfg = SpherePackConfig(material=mat, damping_global=200.0, damping_rotational=200.0, spin_slip=True)
    sp = SpherePack(pos.copy(), np.full(len(pos), r), 2500.0, walls, cfg)
    sp.dt *= fac
    ke0 = sp.kinetic_energy()
    for k in range(150):
        sp.step()
        if k % 30 == 0:
            print(f"fac={fac} k={k} KE={sp.kinetic_energy():.3e} maxpen={sp.max_penetration_ratio:.3e} "
                  f"vmax={np.max(np.linalg.norm(sp.vel,axis=1)):.3e} nc={sp.n_contacts} dt={sp.dt:.2e}")