"""Real per-bond disk dynamics under small bilateral tensile traction."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tensordem.dem import DEMConfig, IceDEM
from tensordem.calibrated_dem import CalibratedIceDEM
from tensordem.material_calibration import corrected_stiffness

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--audit',action='store_true')
args=parser.parse_args()
cases=[]
for factor in (1,2,4):
    cfg=DEMConfig(nx=5,ny=3,radius=.1,thickness=.2,bond_stiffness=150000,
                  contact_damping=0,drag=100,fix_edges=False,tool_gap=10,
                  breaking_strain=1,shear_breaking_strain=1)
    proxy=IceDEM(cfg)
    from dataclasses import replace
    steps=round(1/cfg.recommended_dt)*factor
    cfg=replace(cfg,dt=1/steps)
    sim=CalibratedIceDEM(cfg,bond_stiffness=corrected_stiffness(proxy.positions,proxy.bond_indices,1e6,.2))
    reference=sim.positions.clone()
    left=reference[:,0]==reference[:,0].min();right=reference[:,0]==reference[:,0].max()
    bottom=reference[:,1]==reference[:,1].min();top=reference[:,1]==reference[:,1].max()
    load=torch.zeros_like(reference)
    for mask,sign in ((left,-1),(right,1)):
        weights=torch.ones(int(mask.sum()),dtype=torch.float64);weights[[0,-1]]=.5
        load[mask,0]=sign*8*weights/weights.sum()
    history=[]
    for step in range(steps):
        sim.step(load,load_time=sim.time)
        if step%max(1,steps//100)==0 or step==steps-1:
            extension=float((sim.positions[right,0]-reference[right,0]).mean()-(sim.positions[left,0]-reference[left,0]).mean())
            transverse=float((sim.positions[top,1]-reference[top,1]).mean()-(sim.positions[bottom,1]-reference[bottom,1]).mean())
            history.append(dict(time_s=sim.time,axial_strain=extension/.8,transverse_strain=transverse/.4,**sim.mechanical_energy()))
    last=history[-1]; measured_E=100/last['axial_strain'];nu=-last['transverse_strain']/last['axial_strain']
    cases.append(dict(dt_s=sim.dt,steps=steps,measured_E_Pa=measured_E,apparent_nu=nu,E_relative_error=abs(measured_E/1e6-1),nu_abs_error=abs(nu-1/3),history=history,snapshot=sim.snapshot(),
                      discrete_energy_residual_J=sim.mechanical_energy()['mechanical_J']-sim.accounting['external_work_J']+sim.accounting['damping_dissipation_J']))
record=dict(schema='tensordem.calibrated-dynamic-patch/1',source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['src/tensordem/calibrated_dem.py','src/tensordem/dem.py','src/tensordem/material_calibration.py']},cases=cases,
            scope='1 second bilateral 8 N traction, actual disk masses, drag100 N*s/m relaxation; no continuum mass, fracture or coupled wet qualification',
            patch_passed=all(c['E_relative_error']<.03 and c['nu_abs_error']<.01 for c in cases))
output=root/'docs/material-calibration/dynamic.json'
if args.audit:
    if json.loads(output.read_text()) != record:
        raise SystemExit('FAIL source/state/history/metrics exact replay')
    print('PASS dynamic raw-state exact replay')
else:
    output.write_text(json.dumps(record,indent=2)+'\n')
for c in cases:print({k:v for k,v in c.items() if k not in ('history','snapshot')})
print('patch_passed',record['patch_passed'])
