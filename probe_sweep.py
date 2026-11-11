import sys, time
sys.path.insert(0, '/root/tp-agent/work/m14721/src')
from dataclasses import replace
from tensordem.shear_cell import ShearCellCase, prepare_bed, simulate_shear_cell

levels = [float(x) for x in (sys.argv[1:] or [2000.0, 4000.0, 8000.0])]
base = ShearCellCase(consol_time=0.04, shear_strain_target=0.30, sample_every=25, diagnostics=True)
t0 = time.time()
bed = prepare_bed(base)
print("bed: steps=%d bed_top=%.4f phi=%.3f n=%d (%.1fs)"
      % (bed["steps"], bed["bed_top"], bed["packing_fraction"], bed["n_particles"],
         time.time() - t0))
for s in levels:
    t1 = time.time()
    out = simulate_shear_cell(replace(base, normal_stress=s), bed=bed)
    print("sigma %6.0f -> meas %8.2f (rel %+6.2f%%) tau %8.2f mu %6.4f valid %s "
          "sigma_sem %.2f tau_sem %.2f steps %d (%.0fs)"
          % (s, out["normal_stress_measured"], 100 * out["normal_stress_rel_err"],
             out["shear_stress"], out["mu_macro"], out["valid"],
             out["normal_stress_sem"], out["shear_stress_sem"],
             out["total_steps"], time.time() - t1))
    d = out["diagnostics"]
    nc = [c for c, ph in zip(d["n_contact"], d["phase"]) if ph == "shear"]
    print("    platen contacts: mean %.1f  zero-fraction %.2f  tau_floor/tau %.3f"
          % (sum(nc) / max(len(nc), 1), sum(1 for x in nc if x == 0) / max(len(nc), 1),
             out["tau_platen_vs_floor_rel"]))