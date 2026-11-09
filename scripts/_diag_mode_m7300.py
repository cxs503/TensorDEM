"""Diagnostic 3: does the divergence depend on friction / timestep?"""
import sys, math, os, time
sys.path.insert(0, "src")
import numpy as np
from tensordem.bidisperse_segregation import BedConfig, BidisperseBed

mode = sys.argv[1] if len(sys.argv) > 1 else "base"
kw = {}
if mode == "nofric":
    kw = dict(mu=0.0, mu_wall=0.0, mu_rolling=0.0)
elif mode == "noroll":
    kw = dict(mu_rolling=0.0)
elif mode == "smalldt":
    kw = dict(dt_safety=0.012)
elif mode == "softkn":
    kw = dict(kn=800.0)
cfg = BedConfig(size_ratio=2.0, n_particles=20000, d_small=0.006, width=0.15,
                geometry="box3d", large_volume_fraction=0.2, seed=1, **kw)
bed = BidisperseBed(cfg)
bed.extra_damping = 120.0
t0 = time.time()
vmax_hist = []
for i in range(1, 2500):
    bed.step()
    v = float(np.abs(bed.vel).max())
    vmax_hist.append(v)
    if not np.isfinite(bed.pos).all() or v > 20.0:
        print("%s DIVERGED step %d vmax=%.3g ov=%.3g (%.1fs)" % (mode, i, v, bed.last_max_overlap, time.time() - t0), flush=True)
        break
    if i % 500 == 0:
        print("%s step %d vmax=%.3g ov=%.3g zmax=%.3f (%.1f ms/step)" % (
            mode, i, v, bed.last_max_overlap, bed.pos[:, 2].max(),
            1000 * (time.time() - t0) / i), flush=True)
else:
    print("%s OK 2500 steps vmax_final=%.3g ov=%.3g" % (mode, vmax_hist[-1], bed.last_max_overlap), flush=True)
print("vmax trace:", ["%.2f" % x for x in vmax_hist[::200]], flush=True)