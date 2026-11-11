import sys, time
sys.path.insert(0, '/root/tp-agent/work/m14721/src')
import numpy as np
from tensordem.shear_cell import ShearCellCase, simulate_shear_cell

pad = float(sys.argv[1]) if len(sys.argv) > 1 else 400.0
case = ShearCellCase(normal_stress=pad, diagnostics=True, sample_every=40)
t0 = time.time()
out = simulate_shear_cell(case)
print("wall %.1fs valid=%s sigma_meas=%.3f target=%.1f tau=%.4f mu=%s n=%d steps=%d"
      % (time.time() - t0, out["valid"], out["normal_stress_measured"],
         case.normal_stress, out["shear_stress"], out["mu_macro"],
         out["n_particles"], out["total_steps"]))
d = out["diagnostics"]
print("phase      step  sigma   bed_top platen_z  gap_mm ncont nabove  vmax   mean_z     vz")
for i in range(0, len(d["step"]), max(1, len(d["step"]) // 40)):
    ph = d["phase"][i]
    print("%-7s %6d %8.2f %8.4f %8.4f %7.2f %5d %6d %6.3f %8.4f" % (
        ph, d["step"][i], d["sigma"][i], d["bed_top"][i], d["platen_z"][i],
        1e3 * (d["platen_z"][i] - d["bed_top"][i]), d["n_contact"][i],
        d["n_above"][i], d["vmax"][i], d["bed_mean_z"][i]) + " %8.2e" % d["vz"][i])
print("n_above max", max(d["n_above"]), " mean_z first/last",
      d["bed_mean_z"][0], d["bed_mean_z"][-1],
      " bed_top first/last", d["bed_top"][0], d["bed_top"][-1])