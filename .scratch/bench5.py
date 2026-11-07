import sys, time
sys.path.insert(0, "src")
import torch
from tensordem.dem3d import DEM3DConfig, IceDEM3D
torch.set_num_threads(64)
a=0.02  # d=40mm? no. use d=20mm -> a=0.01
a=0.01
k=8.9e9*a*45.0/142.0
cfg=DEM3DConfig(nx=50,ny=10,nz=5,radius=a/2,density=917.0,bond_stiffness=k,contact_stiffness=k,breaking_strain=1.0,shear_breaking_strain=1.0,contact_damping=0.0,drag=1e-3,tool_speed=1.0,tool_gap=1.0,tool_radius=0.01,fix_x_edges=False,fix_bottom=False)
sim=IceDEM3D(cfg)
empty=torch.empty((0,2),dtype=torch.long); sim._contact_candidate_pairs=lambda: empty
loads=torch.zeros_like(sim.positions)
for _ in range(3): sim.step(external_forces=loads)
t0=time.time()
for _ in range(20): sim.step(external_forces=loads)
t1=time.time()
print(f"N={len(sim.positions)} bonds={len(sim.pairs)} dt={sim.dt:.2e} step={(t1-t0)/20*1e3:.1f} ms")
