"""Profile one step of the 160k-particle beam to find the bottleneck."""
import sys, time
sys.path.insert(0, "/root/tp-agent/work/m5073/src")
import torch
from tensordem.dem3d import DEM3DConfig, IceDEM3D

torch.set_num_threads(128)
a = 0.005
k = 8.9e9 * a * 45.0 / 142.0
cfg = DEM3DConfig(nx=200, ny=40, nz=20, radius=a / 2, density=917.0,
                  bond_stiffness=k, contact_stiffness=k,
                  breaking_strain=1.0, shear_breaking_strain=1.0,
                  contact_damping=0.0, drag=1.0e-3,
                  tool_speed=1.0, tool_gap=1.0, tool_radius=0.01,
                  fix_x_edges=False, fix_bottom=False)
sim = IceDEM3D(cfg)
empty = torch.empty((0, 2), dtype=torch.long)
sim._contact_candidate_pairs = lambda: empty
loads = torch.zeros_like(sim.positions)
sim.step(external_forces=loads)

# manual step decomposition
i, j = sim.pairs.T
t0 = time.time()
for _ in range(5):
    delta = sim.positions[j] - sim.positions[i]
    distance = torch.linalg.vector_norm(delta, dim=1)
    normal = delta / distance.clamp_min(1e-30)[:, None]
    extension = distance - sim.rest_lengths
    bond_mag = k * extension
    bond_force = bond_mag[:, None] * normal
    force = torch.zeros_like(sim.positions)
    force.index_add_(0, i, bond_force)
    force.index_add_(0, j, -bond_force)
t1 = time.time()
print(f"bond part: {(t1-t0)/5*1e3:.1f} ms")

t0 = time.time()
for _ in range(5):
    acc = force / sim.config.mass
    v = sim.velocities + acc * sim.dt
    p = sim.positions + v * sim.dt
t1 = time.time()
print(f"integration part: {(t1-t0)/5*1e3:.1f} ms")

# full step() for reference
t0 = time.time()
for _ in range(5):
    sim.step(external_forces=loads)
t1 = time.time()
print(f"full step(): {(t1-t0)/5*1e3:.1f} ms")
