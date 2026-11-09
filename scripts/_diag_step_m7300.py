"""Diagnostic 2: localise the first non-finite array inside one DEM step."""
import sys, math, time
sys.path.insert(0, "src")
import numpy as np
from tensordem.bidisperse_segregation import BedConfig, BidisperseBed

s = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
cfg = BedConfig(size_ratio=s, n_particles=n, d_small=0.006, width=0.15,
                geometry="box3d", large_volume_fraction=0.2, seed=1)
bed = BidisperseBed(cfg)
bed.extra_damping = 120.0
dt = bed.dt

def ok(name, a):
    if not np.isfinite(a).all():
        bad = np.argwhere(~np.isfinite(a))
        print("FAIL", name, "n_bad", len(bad), "first", bad[:5], flush=True)
        return False
    return True

for i in range(1, 800):
    iu, ju = bed._pairs()
    if not ok("pairs", bed.pos):
        break
    bed.last_max_overlap = 0.0
    f_p, tq_p = bed._particle_forces(iu, ju, dt)
    if not ok("f_p", f_p) or not ok("tq_p", tq_p):
        print("pairs", len(iu), "ov", bed.last_max_overlap, flush=True)
        k = np.argmax(np.abs(f_p).max(axis=1))
        print("worst force particle", k, f_p[k], bed.vel[k], bed.pos[k], bed.r[k], flush=True)
        break
    f_w, tq_w, ffz = bed._wall_contacts(bed.t, dt)
    if not ok("f_w", f_w) or not ok("tq_w", tq_w):
        print("wall fail", flush=True)
        break
    acc = (f_p + f_w) * bed.inv_m[:, None]
    acc[:, 2] -= 9.81
    bed.vel += acc * dt
    decay = math.exp(-bed.extra_damping * dt)
    bed.vel *= decay
    bed.omega += (tq_p + tq_w) * bed.inv_i[:, None] * dt
    bed.omega *= decay
    if not ok("vel", bed.vel) or not ok("omega", bed.omega):
        print("step", i, "vmax", np.nanmax(np.abs(bed.vel)), "omax", np.nanmax(np.abs(bed.omega)), flush=True)
        break
    bed.pos += bed.vel * dt
    bed.t += dt
    if not ok("pos", bed.pos):
        print("step", i, "vmax", np.abs(bed.vel).max(), flush=True)
        k = np.argmax(np.abs(bed.vel).max(axis=1))
        print("worst", k, bed.vel[k], bed.pos[k], bed.r[k], flush=True)
        break
print("done", flush=True)