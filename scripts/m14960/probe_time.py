import sys, time
sys.path.insert(0, 'src')
from dataclasses import replace
from tensordem.shear_cell import ShearCellCase, prepare_bed, simulate_shear_cell

for (w,d,h) in [(0.08,0.032,0.05),(0.08,0.064,0.08),(0.16,0.08,0.10)]:
    c = ShearCellCase(box_width=w, box_depth=d, fill_height=h, normal_stress=4000.0,
                      shear_strain_target=0.1, consol_time=0.04, sample_every=50)
    t0=time.time(); bed=prepare_bed(c); t1=time.time()
    out=simulate_shear_cell(replace(c), bed=bed); t2=time.time()
    print("box %.3f %.3f %.3f n=%d bed %.0fs  shear(gamma=0.1) %.0fs steps=%d sigma_meas=%.1f mu=%.3f valid=%s"
          % (w,d,h,bed["n_particles"],t1-t0,t2-t1,out["total_steps"],out["normal_stress_measured"],out["mu_macro"],out["valid"]), flush=True)
