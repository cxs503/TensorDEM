import sys, time
sys.path.insert(0, '/root/tp-agent/work/m14721/src')
from dataclasses import replace
from tensordem.shear_cell import ShearCellCase, prepare_bed, simulate_shear_cell

mus = [float(x) for x in (sys.argv[1:] or [0.1, 0.3, 0.5])]
base = ShearCellCase(consol_time=0.04, shear_strain_target=0.30, sample_every=25,
                     normal_stress=4000.0)
t0 = time.time()
bed = prepare_bed(base)
print("bed: steps=%d bed_top=%.4f phi=%.3f (%.1fs)"
      % (bed["steps"], bed["bed_top"], bed["packing_fraction"], time.time() - t0))
for m in mus:
    t1 = time.time()
    out = simulate_shear_cell(replace(base, mu=m), bed=bed)
    print("mu_p %.2f -> sigma %8.2f (rel %+6.2f%%) tau %8.2f mu_macro %6.4f "
          "valid %s tau_floor %8.2f (%.0fs)"
          % (m, out["normal_stress_measured"], 100 * out["normal_stress_rel_err"],
             out["shear_stress"], out["mu_macro"], out["valid"],
             out["floor_shear_stress"], time.time() - t1))