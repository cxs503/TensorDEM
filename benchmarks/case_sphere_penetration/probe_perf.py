"""Timing/API probe for the sphere-penetration case (not part of the deliverable)."""
import sys, time
import numpy as np
sys.path.insert(0, "src")
from tensordem.sphere_pack_dem import SpherePack, SpherePackConfig, SphereWall
from tensordem.superquadric_kernel import ContactMaterial


def build(n_wide, n_deep, d, phi_target=0.60, seed=0):
    rng = np.random.default_rng(seed)
    r = d / 2.0
    # FCC lattice: nearest-neighbour spacing L from target packing fraction
    # phi = 0.7405 * (2r/L)^3  ->  L = 2r * (0.7405/phi)^(1/3)
    L = 2 * r * (0.7405 / phi_target) ** (1.0 / 3.0)
    a = L * np.sqrt(2.0)
    pts = []
    nx = int(n_wide * d / a) + 1
    ny = int(n_wide * d / a) + 1
    nz = int(n_deep * d / a) + 1
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                base = np.array([i * a, j * a, k * a])
                for off in ([0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]):
                    p = base + a * np.array(off)
                    if p[0] < n_wide * d and p[1] < n_wide * d and p[2] < n_deep * d:
                        pts.append(p)
    pts = np.array(pts)
    pts = pts[rng.permutation(len(pts))]
    pts[:] += rng.normal(0.0, 0.05 * d, pts.shape)
    radii = np.full(len(pts), r)
    return pts, radii, r


def main():
    d = 0.004
    for n_wide, n_deep in ((22, 12), (24, 14)):
        pos, rad, r = build(n_wide, n_deep, d)
        Lx, Ly = n_wide * d, n_wide * d
        walls = [SphereWall(np.array([0.0, 0, 0]), 0.0, label="x0"),
                 SphereWall(np.array([1.0, 0, 0]), -Lx, label="xL"),
                 SphereWall(np.array([0.0, 1, 0]), 0.0, label="y0"),
                 SphereWall(np.array([0.0, 1, 0]), -Ly, label="yL"),
                 SphereWall(np.array([0.0, 0, 1]), 0.0, label="floor")]
        cfg = SpherePackConfig(damping_global=200.0, damping_rotational=200.0)
        sp = SpherePack(pos, rad, 2500.0, walls, cfg)
        n = len(pos)
        t0 = time.time()
        sp.run(20)
        t1 = time.time()
        phi = n * (4.0 / 3.0 * np.pi * r ** 3) / (Lx * Ly * n_deep * d)
        print(f"N={n:6d} dt={sp.dt:.3e} phi0={phi:.3f}  {(t1-t0)/20*1000:.0f} ms/step  "
              f"contacts={sp.n_contacts}")


if __name__ == "__main__":
    main()