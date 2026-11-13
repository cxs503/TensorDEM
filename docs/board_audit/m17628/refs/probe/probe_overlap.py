import numpy as np, math
from tensordem.mts_contact import _gas_system, pressure_and_shear
for lee in (False, True):
    sysm, box = _gas_system(0.80, 64, 0.10, n_sub=5, mu=0.5, kn=1e5, zeta=0.4,
                            density=2000.0, radius=5e-3, contact_model="linear",
                            bidisperse=True, random_positions=True, shear_rate=10.0,
                            scheme="mts", seed=0, lees_edwards=lee)
    sysm.step(909)
    c = sysm.contacts; pair = c["kind"] == 0
    L = np.linalg.norm(c["branch"][pair], axis=1)
    rsum = sysm.radii[c["i"][pair]] + sysm.radii[c["j"][pair]]
    ov = rsum - L
    st = pressure_and_shear(sysm, box*box)
    print(f"lee={lee} n_pair={pair.sum()} P={st['pressure']:.4g} tau={st['shear']:.4g} "
          f"tau/P={st['shear']/st['pressure']:+.4f} coord={st['coord']:.2f} "
          f"ov_max={100*ov.max()/5e-3:.2f}%rad ov_med={100*np.median(ov)/5e-3:.3f}% "
          f"n_ov>10%={int((ov>0.1*5e-3).sum())} strain={sysm.strain:.3f}")
