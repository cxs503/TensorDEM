"""Diagnose the axial stress measurement paths (temporary, not a deliverable)."""
import math
import numpy as np
import sys
sys.path.insert(0, "src")

from tensordem.friction_surface.kernel import (ContactModel, PairState, PlaneWall,
                                               apply_walls, integrate)
from tensordem.friction_surface.shear import (build_pairs_periodic, build_specimen,
                                              pair_forces_periodic, stress_tensor)
from tensordem.triaxial_cd import CDTriaxialCase

case = CDTriaxialCase(sigma3=1.0e5, nx=5, ny=5, nz=4)
model = case.model
geom = case.geometry_case
dt = case.dt_value
pos = build_specimen(geom)
width, depth = float(geom.width), float(geom.depth)
n = len(pos)
vel = np.zeros_like(pos); omega = np.zeros_like(pos); ps = PairState(); contacts = {}
floor = PlaneWall((0, 0, 0), (0, 0, 1), mu=0.0, mu_rolling=0.0)
z_top = float(pos[:, 2].max() + model.radius)
lid = PlaneWall((0, 0, z_top), (0, 0, -1), mu=0.0, mu_rolling=0.0)
area = width * depth

for step in range(30000):
    iu, ju, keys = build_pairs_periodic(pos, model.radius, width, depth)
    f, tq = pair_forces_periodic(pos, vel, omega, iu, ju, ps, model, dt, width,
                                depth, contacts=contacts)
    fw, tw = apply_walls([floor, lid], pos, vel, omega, model, dt, step*dt)
    f += fw
    pos, vel, omega = integrate(pos, vel, omega, f, tq+tw, model, dt)
    pos[:, 0] %= width; pos[:, 1] %= depth
    vol = width*depth*max(z_top, 1e-9)
    st = stress_tensor(contacts, vol)
    p_lat = 0.5*(st[0, 0]+st[1, 1])
    err = (case.sigma3 - p_lat)/case.sigma3
    rate = float(np.clip(20.0*err, -3.0, 3.0))
    sc = 1.0 - rate*dt
    pos[:, 0] *= sc; pos[:, 1] *= sc; vel[:, 0] /= sc; vel[:, 1] /= sc
    width *= sc; depth *= sc; area = width*depth
    target = case.sigma3*area
    dz = -0.5*(target - float(-lid.traction[2]))/(model.kn*8.0)
    dz = float(np.clip(dz, -0.2*dt, 0.2*dt))
    z_top = float(z_top + dz); lid.point[2] = z_top

st = stress_tensor(contacts, width*depth*z_top)
zmin, zmax = float(pos[:, 2].min()), float(pos[:, 2].max())
vol_eff = width*depth*(zmax - zmin + 2*model.radius)
st_eff = stress_tensor(contacts, vol_eff)
lid_p = float(-lid.traction[2])/area
floor_p = float(-floor.traction[2])/area
print(f"n={n} steps=30000 sf={n*4/3*math.pi*model.radius**3/(width*depth*z_top):.3f}")
print(f"lid_p={lid_p/1e3:.2f} kPa  floor_p={floor_p/1e3:.2f} kPa")
print(f"interior sigma_zz(V=area*z_top)={st[2,2]/1e3:.2f} kPa  p_lat={0.5*(st[0,0]+st[1,1])/1e3:.2f} kPa")
print(f"interior sigma_zz(V_eff)={st_eff[2,2]/1e3:.2f} kPa  p_lat={0.5*(st_eff[0,0]+st_eff[1,1])/1e3:.2f} kPa")
print(f"z_top={z_top:.4f} zmin={zmin:.4f} zmax={zmax:.4f} r={model.radius}")
print(f"ratio lid/interior={lid_p/float(st[2,2]):.3f}  lid/interior_eff={lid_p/float(st_eff[2,2]):.3f}")
print(f"ke/particle={0.5*model.mass*np.sum(vel**2)/n:.3e}")
print(f"p_lat err rel={(case.sigma3 - 0.5*(st[0,0]+st[1,1]))/case.sigma3:.4f}")
print("n_contacts", len(contacts['i']))