import sys, numpy as np
sys.path.insert(0, "src")
sys.path.insert(0, "benchmarks/case_sphere_penetration")
from run_case import fcc_bed, make_walls, SpherePack, SpherePackConfig, ContactMaterial

rng = np.random.default_rng(1)
d = 0.004
pos, rad, Lx, Ly, H = fcc_bed(8, 6, d, 0.60, rng)
print("N", len(pos), "Lx", Lx, "H", H, "phi_lattice",
      len(pos)*4/3*np.pi*(d/2)**3/(Lx*Ly*H))
mat = ContactMaterial(e_star=1e9, g_star=4e8, mu=0.4, mu_rolling=0.05, restitution=0.4)
cfg = SpherePackConfig(material=mat, damping_global=200.0, damping_rotational=200.0, spin_slip=True)
sp = SpherePack(pos, rad, 2500.0, make_walls(Lx, Ly), cfg)
print("dt", sp.dt, "contacts", sp.n_contacts)
for k in range(200):
    sp.step()
    if k % 20 == 0:
        print(k, "KE %.3e" % sp.kinetic_energy(), "maxpen %.3e" % sp.max_penetration_ratio,
              "vmax %.3e" % np.max(np.linalg.norm(sp.vel, axis=1)),
              "nc", sp.n_contacts)