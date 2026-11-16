"""Does build_bed() give a bit-identical bed for the same seed in two processes?

Probe for the m23184 reproducibility audit of CASE-DEM-SILO-BEVERLOO:
two archived records of the *same* frozen case and the same seed disagree by a
factor 9 in the discharge rate but by 1e-6 m in the settled surface, which is
only possible if either (a) the initial state is not reproducible across
processes, or (b) the discharge is chaotically bistable.
"""
import hashlib
import json
import sys

sys.path.insert(0, "/root/tp-agent/work/m23184/src")
from tensordem.silo_flow.beverloo import DischargeCase
from tensordem.silo_flow.kernel import Material, SiloGeometry, contact_time
from tensordem.silo_flow.packing import build_bed

case = DischargeCase(
    particle_diameter=0.003, orifice_diameter=0.015, silo_radius=0.024,
    fill_height=0.036, wall_mu=0.0, seed=1, settle_time=0.06,
    discard_time=0.12, window_time=0.05, n_windows=8, kn=5.0e3, mu=0.50,
)
mat = Material(radius=0.5 * case.particle_diameter, mu=case.mu,
               mu_rolling=case.mu_rolling, kn=case.kn)
geo = SiloGeometry(cylinder_radius=case.silo_radius,
                   cylinder_height=case.fill_height + 0.02,
                   outlet_radius=0.5 * case.orifice_diameter)
dt = contact_time(mat.mass_eff, case.kn) / 20.0
flow = build_bed(geo, mat, case.fill_height, wall_mu=case.wall_mu, seed=case.seed,
                 settle_time=case.settle_time, dt=dt)
flow.plugged = False
for _ in range(400):
    flow.step()
print(json.dumps({
    "n": flow.n,
    "dt": flow.dt,
    "pos_sha": hashlib.sha256(flow.pos.tobytes()).hexdigest()[:16],
    "vel_sha": hashlib.sha256(flow.vel.tobytes()).hexdigest()[:16],
    "bed_mass": flow.bed_mass,
    "surface": flow.surface_height(),
    "mass_exited": flow.mass_exited,
    "initial_overlap_over_radius": flow.initial_overlap_over_radius,
}, indent=1))