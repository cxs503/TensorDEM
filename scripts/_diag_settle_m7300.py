"""Diagnostic: monitor the settle phase for divergence (campaign debugging)."""
import sys, math, time
sys.path.insert(0, "src")
import numpy as np
np.seterr(divide="raise", invalid="raise", over="raise")
from tensordem.bidisperse_segregation import BedConfig, BidisperseBed

s = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
cfg = BedConfig(size_ratio=s, n_particles=n, d_small=0.006, width=0.15,
                geometry="box3d", large_volume_fraction=0.2, seed=1)
t0 = time.time()
bed = BidisperseBed(cfg)
print("build %.1fs n=%d n_l=%d Lx=%.3f Ly=%.3f Lz=%.3f packed_h=%.3f kn=%.0f dt=%.3e"
      % (time.time() - t0, bed.n, int(bed.cls.sum()), bed.Lx, bed.Ly, bed.Lz,
         bed.bed_height_packed, bed.kn, bed.dt), flush=True)
print("initial overlaps:", bed.last_max_overlap, flush=True)
bed.extra_damping = 120.0
t0 = time.time()
for i in range(1, 900):
    bed.step()
    if not np.isfinite(bed.pos).all():
        print("NONFINITE at step", i, flush=True); break
    if i % 100 == 0:
        v = float(np.abs(bed.vel).max())
        print("step %d ov=%.3g vmax=%.3g zmax=%.3f xmin=%.3f xmax=%.3f t=%.3fs %.1f ms/step"
              % (i, bed.last_max_overlap, v, bed.pos[:, 2].max(), bed.pos[:, 0].min(),
                 bed.pos[:, 0].max(), bed.t, 1000 * (time.time() - t0) / i), flush=True)
        t0 = time.time()
        if not np.isfinite(bed.pos).all() or v > 50:
            print("DIVERGENCE", flush=True)
            bad = np.argmax(np.abs(bed.vel).max(axis=1))
            print("worst particle", bad, bed.pos[bad], bed.vel[bad], bed.r[bad], flush=True)
            break