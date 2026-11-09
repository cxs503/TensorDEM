"""Diagnostic 4: which wall-contact array goes non-finite first?"""
import sys, math
sys.path.insert(0, "src")
import numpy as np
import tensordem.bidisperse_segregation as bs

cfg = bs.BedConfig(size_ratio=2.0, n_particles=20000, d_small=0.006, width=0.15,
                   geometry="box3d", large_volume_fraction=0.2, seed=1)
bed = bs.BidisperseBed(cfg)
bed.extra_damping = 120.0

orig = bed._wall_contacts


def checked(t, dt):
    n = bed.n
    pos, r, vel, om = bed.pos, bed.r, bed.vel, bed.omega
    zf = bed.floor_offset(t)
    vzf = bed.floor_velocity(t)
    walls = [(0, 2, +1.0, zf, np.array([0.0, 0.0, vzf])), (1, 0, +1.0, 0.0, np.zeros(3)),
             (2, 0, -1.0, bed.Lx, np.zeros(3)), (3, 1, +1.0, 0.0, np.zeros(3)),
             (4, 1, -1.0, bed.Ly, np.zeros(3))]
    for wid, axis, sign, coord, v_wall in walls:
        nvec = np.zeros(3)
        nvec[axis] = sign
        pen = -sign * (pos[:, axis] - coord) + r
        act = pen > 0
        idx = np.flatnonzero(act)
        if len(idx) == 0:
            continue
        for name, arr in (("pen", pen[idx]), ("wall_tang", bed.wall_tang[wid][idx]),
                          ("vel", vel[idx]), ("om", om[idx])):
            if not np.isfinite(arr).all():
                k = idx[0]
                print("WALL", wid, name, "non-finite ; particle", k, "pos", pos[k],
                      "vel", vel[k], "om", om[k], "r", r[k], flush=True)
                return orig(t, dt)
        cap = 0.35 * np.maximum(bed.kn * pen[idx] - 2 * bed.zeta *
                                np.sqrt(bed.kn * bed.mass[idx]) *
                                ((vel[idx] - v_wall) @ nvec), 0.0) / bed.kt
        tg = np.linalg.norm(bed.wall_tang[wid][idx], axis=1)
        if np.any(tg > cap + 1e-12):
            print("WALL", wid, "tang over cap", float(np.max(tg - cap)), "n", len(idx), flush=True)
    return orig(t, dt)


bed._wall_contacts = checked
for i in range(1, 1200):
    bed.step()
    if not np.isfinite(bed.pos).all():
        print("first non-finite pos at step", i, "t=%.4g" % bed.t, flush=True)
        bad = np.flatnonzero(~np.isfinite(bed.pos).any(axis=1))
        print("bad particles", bad[:10], flush=True)
        k = bad[0]
        print("pos", bed.pos[k], "vel", bed.vel[k], "om", bed.omega[k], "r", bed.r[k], flush=True)
        break
    if i % 200 == 0:
        print("step", i, "vmax %.3g" % np.abs(bed.vel).max(), "ommax %.3g" % np.abs(bed.omega).max(),
              "tang_pairs", len(bed._tang_keys), flush=True)
else:
    print("no NaN in 1200 steps", flush=True)