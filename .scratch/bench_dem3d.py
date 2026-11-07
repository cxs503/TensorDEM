"""Quick per-step cost benchmark for the ice-beam bending DEM (d=5mm lattice)."""
import sys, time
sys.path.insert(0, "/root/tp-agent/work/m5073/src")
import torch
from tensordem.dem3d import DEM3DConfig, IceDEM3D

torch.set_num_threads(128)

# same density as the full beam: d=5mm, E=8.9GPa target
a = 0.005
k = 8.9e9 * a * 45.0 / 142.0
print(f"k_bond={k:.4g} N/m")

for (nx, ny, nz) in [(40, 8, 4), (100, 20, 10)]:
    cfg = DEM3DConfig(nx=nx, ny=ny, nz=nz, radius=a / 2, density=917.0,
                      bond_stiffness=k, contact_stiffness=k,
                      breaking_strain=1.0, shear_breaking_strain=1.0,
                      contact_damping=0.0, drag=1.0e-3,
                      tool_speed=1.0, tool_gap=1.0, tool_radius=0.01,
                      fix_x_edges=False, fix_bottom=False)
    t0 = time.time()
    sim = IceDEM3D(cfg)
    t1 = time.time()
    print(f"grid {nx}x{ny}x{nz}: N={len(sim.positions)} bonds={len(sim.pairs)} "
          f"build={t1-t0:.2f}s dt={sim.dt:.3e}")
    loads = torch.zeros_like(sim.positions)
    for _ in range(3):
        sim.step(external_forces=loads)
    t0 = time.time()
    n = 20
    for _ in range(n):
        sim.step(external_forces=loads)
    t1 = time.time()
    per = (t1 - t0) / n
    print(f"  step={per*1e3:.2f} ms  -> 1.5M steps = {1.5e6*per/3600:.1f} h")
