"""Generate figures for the m669 report."""
import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
sys.path.insert(0, "src")

from tensordem.hull_mesh_contact import mesh_contact_forces, make_flat_mesh

# ---- Figure 1: Force error vs analytical ----
R = 0.05
k = 10000.0
heights = np.linspace(0.001, R * 0.99, 50)
errors = []
forces_got = []
forces_expected = []
for h in heights:
    verts, faces = make_flat_mesh(size=2.0, normal="z", offset=0.0, nx=8, ny=8)
    pos = torch.tensor([[0.0, 0.0, h]], dtype=torch.float64)
    vel = torch.zeros(1, 3, dtype=torch.float64)
    f, _ = mesh_contact_forces(pos, vel, verts, faces,
                                torch.zeros(3, dtype=torch.float64),
                                particle_radius=R, stiffness=k)
    expected = k * (R - h)
    got = float(f[0, 2].item())
    forces_got.append(got)
    forces_expected.append(expected)
    errors.append(abs(got - expected) / max(expected, 1e-15))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(heights, forces_expected, "k--", label="Analytical $F = k(R-h)$", linewidth=2)
ax1.plot(heights, forces_got, "r.", markersize=4, label="Mesh contact (this work)")
ax1.set_xlabel("Sphere height $h$ [m]", fontsize=12)
ax1.set_ylabel("Contact force $F_z$ [N]", fontsize=12)
ax1.set_title("Sphere on Flat Plane: Force vs Height", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.semilogy(heights, np.array(errors) + 1e-16, "b-", linewidth=2)
ax2.axhline(0.1, color="r", linestyle="--", label="Threshold (10%)")
ax2.set_xlabel("Sphere height $h$ [m]", fontsize=12)
ax2.set_ylabel("Relative force error $|F_{\\rm mesh} - F_{\\rm anal}| / F_{\\rm anal}$", fontsize=12)
ax2.set_title("Force Error vs Analytical Solution", fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(1e-16, 1)

plt.tight_layout()
plt.savefig("artifacts/fig1_force_error.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- Figure 2: Complexity scaling ----
verts, faces = make_flat_mesh(size=10.0, normal="z", offset=0.0, nx=10, ny=10)
torch.manual_seed(42)
N_vals = [100, 200, 500, 1000, 2000, 5000]
times = []
for n in N_vals:
    pos = torch.rand(n, 3, dtype=torch.float64)
    pos[:, 0] = (pos[:, 0] - 0.5) * 8.0
    pos[:, 1] = (pos[:, 1] - 0.5) * 8.0
    pos[:, 2] = 0.03
    vel = torch.zeros_like(pos)
    # Average over 3 runs
    t_list = []
    for _ in range(3):
        t0 = time.perf_counter()
        mesh_contact_forces(pos, vel, verts, faces,
                            torch.zeros(3, dtype=torch.float64),
                            particle_radius=0.05, stiffness=10000.0)
        t_list.append(time.perf_counter() - t0)
    times.append(np.median(t_list))

fig2, ax3 = plt.subplots(figsize=(7, 5))
ax3.loglog(N_vals, times, "bo-", markersize=8, linewidth=2, label="Measured")
# O(N) reference line
t_ref = times[0]
n_ref = N_vals[0]
ax3.loglog(N_vals, [t_ref * (n / n_ref) for n in N_vals], "g--", linewidth=2, label="$O(N)$ reference")
# O(N log N) reference
ax3.loglog(N_vals, [t_ref * (n / n_ref) * np.log2(n / n_ref + 1) for n in N_vals],
           "r:", linewidth=2, label="$O(N \\log N)$ reference")
ax3.set_xlabel("Number of spheres $N$", fontsize=12)
ax3.set_ylabel("Contact detection time [s]", fontsize=12)
ax3.set_title("Scaling: Sphere–Triangle Mesh Contact", fontsize=13)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("artifacts/fig2_complexity.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- Figure 3: 3D contact visualization ----
fig3 = plt.figure(figsize=(8, 6))
ax4 = fig3.add_subplot(111, projection="3d")

# Plot a bowl mesh
n_theta = 24
n_phi = 12
radius_hull = 0.5
verts_list = []
for j in range(n_phi + 1):
    phi = math.pi / 2 * j / n_phi
    for i in range(n_theta):
        theta = 2 * math.pi * i / n_theta
        x = radius_hull * math.sin(phi) * math.cos(theta)
        y = radius_hull * math.sin(phi) * math.sin(theta)
        z = -radius_hull * math.cos(phi)
        verts_list.append([x, y, z])
verts_bowl = np.array(verts_list)

# Plot mesh wireframe
for j in range(n_phi):
    for i in range(n_theta):
        i_next = (i + 1) % n_theta
        idx0 = j * n_theta + i
        idx1 = j * n_theta + i_next
        idx2 = (j + 1) * n_theta + i
        idx3 = (j + 1) * n_theta + i_next
        ax4.plot3D(*zip(verts_bowl[idx0], verts_bowl[idx1]), color="steelblue", alpha=0.3, linewidth=0.5)
        ax4.plot3D(*zip(verts_bowl[idx0], verts_bowl[idx2]), color="steelblue", alpha=0.3, linewidth=0.5)

# Plot spheres
sphere_positions = [
    [0.0, 0.0, -0.45],
    [0.15, 0.1, -0.40],
    [-0.1, 0.2, -0.35],
]
R_vis = 0.05
for sp in sphere_positions:
    u = np.linspace(0, 2 * np.pi, 15)
    v = np.linspace(0, np.pi, 10)
    xs = R_vis * np.outer(np.cos(u), np.sin(v)) + sp[0]
    ys = R_vis * np.outer(np.sin(u), np.sin(v)) + sp[1]
    zs = R_vis * np.outer(np.ones_like(u), np.cos(v)) + sp[2]
    ax4.plot_surface(xs, ys, zs, color="coral", alpha=0.7)

# Force arrows
for sp in sphere_positions:
    ax4.quiver(sp[0], sp[1], sp[2], 0, 0, 0.15, color="red", arrow_length_ratio=0.2, linewidth=2)

ax4.set_xlabel("X [m]")
ax4.set_ylabel("Y [m]")
ax4.set_zlabel("Z [m]")
ax4.set_title("Sphere–Hull Mesh Contact (3D)", fontsize=13)

plt.tight_layout()
plt.savefig("artifacts/fig3_3d_contact.png", dpi=150, bbox_inches="tight")
plt.close()

print("Figures saved to artifacts/")
