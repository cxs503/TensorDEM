"""Production-scale (2e4) spot check with the overlap-free packing (m7300)."""
import sys, math, time, json
sys.path.insert(0, "src")
from tensordem.bidisperse_segregation import BedConfig, BidisperseBed, summarize_shake

s = float(sys.argv[1])
gam = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
cfg = BedConfig(size_ratio=s, n_particles=20000, d_small=0.006, width=0.15,
                geometry="box3d", large_volume_fraction=0.2, seed=1, kn=900.0)
t0 = time.time()
bed = BidisperseBed(cfg)
bed.settle(0.05)
m0 = bed.measure()
A = gam * 9.81 / (2 * math.pi * 25.0) ** 2
res = bed.run_shaking(30 / 25.0, A, 25.0, sample_dt=0.04)
sm = summarize_shake(res)
print(json.dumps({"s": s, "gamma": gam, "diverged": res["diverged"],
                  "n_steps": res["n_steps"], "S": sm["S"], "S_init": sm["S_initial"],
                  "v_up": sm["v_up"], "ov": sm["max_overlap"],
                  "mass_err": sm["mass_rel_error"], "wall_s": time.time() - t0,
                  "h": m0["bed_height"], "dt": bed.dt, "kn": bed.kn}), flush=True)
print("S trace:", ["%.3f" % h["S"] for h in res["history"]], flush=True)