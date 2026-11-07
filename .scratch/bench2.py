"""Measure pure-torch bond-force cost (no contact search)."""
import sys, time
sys.path.insert(0, "/root/tp-agent/work/m5073/src")
import torch
from tensordem.dem3d import DEM3DConfig, IceDEM3D

torch.set_num_threads(128)
a = 0.005
k = 8.9e9 * a * 45.0 / 142.0

for (nx, ny, nz) in [(100, 20, 10), (200, 40, 20)]:
    cfg = DEM3DConfig(nx=nx, ny=ny, nz=nz, radius=a / 2, density=917.0,
                      bond_stiffness=k, contact_stiffness=k,
                      breaking_strain=1.0, shear_breaking_strain=1.0,
                      contact_damping=0.0, drag=1.0e-3,
                      tool_speed=1.0, tool_gap=1.0, tool_radius=0.01,
                      fix_x_edges=False, fix_bottom=False)
    t0 = time.time()
    sim = IceDEM3D(cfg)
    t1 = time.time()
    print(f"grid {nx}x{ny}x{nz}: N={len(sim.positions)} bonds={len(sim.pairs)} build={t1-t0:.2f}s")
    # disable contact search: intact lattice has no overlaps (proved by geometry)
    empty = torch.empty((0, 2), dtype=torch.long)
    sim._contact_candidate_pairs = lambda: empty
    loads = torch.zeros_like(sim.positions)
    for _ in range(3):
        sim.step(external_forces=loads)
    t0 = time.time()
    n = 30
    for _ in range(n):
        sim.step(external_forces=loads)
    t1 = time.time()
    per = (t1 - t0) / n
    print(f"  step(no contacts)={per*1e3:.2f} ms  -> 1.5M steps = {1.5e6*per/3600:.1f} h")
