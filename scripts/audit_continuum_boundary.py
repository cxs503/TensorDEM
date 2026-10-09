"""Reconstruct final fields and analytic bar response without report badges."""
import hashlib,json,math
from pathlib import Path
import torch
from tensordem.continuum_boundary_dem import BoundaryNetwork
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/coupling/continuum-boundary'
s=json.loads((OUT/'study.json').read_text())
assert hashlib.sha256((ROOT/'src/tensordem/continuum_boundary_dem.py').read_bytes()).hexdigest()==s['source_sha256']
rows=[]
for name,digest in s['files_sha256'].items():
    p=OUT/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    r=json.loads(p.read_text());b=BoundaryNetwork.restore(r['final'])
    assert torch.equal(b.x,torch.tensor(r['positions_m'],dtype=torch.float64))
    assert torch.equal(b.mass,torch.tensor(r['mass_nodes_kg'],dtype=torch.float64))
    assert torch.equal(b.edges,torch.tensor(r['edges']))
    assert torch.equal(b.k,torch.tensor(r['stiffness_N_per_m'],dtype=torch.float64))
    assert abs(float(b.mass.sum())-22.5)<1e-12
    assert float(b.u[b.fixed].abs().max())==0 and float(b.v[b.fixed].abs().max())==0
    momentum=(b.mass[:,None]*b.v).sum(0)
    assert float((momentum-b.impulse).abs().max())<1e-12
    end=float((b.u[b.right,0]*b.areas).sum()/b.areas.sum())
    assert abs(end-r['end_displacement_m'])<1e-15
    assert abs(b.energy()-r['history'][-1]['energy_J'])<1e-14
    assert abs(b.time-r['duration_s'])<1e-12
    assert r['restart_exact']
    if r['mode']=='load':
        c=math.sqrt(9*b.config['E']/8/b.config['density']);t=b.time
        exact=100/(9*b.config['E']/8)*(1-sum(8/math.pi**2/(2*j+1)**2*math.cos((2*j+1)*math.pi*c*t/2) for j in range(20000)))
        assert abs(exact-r['analytic_end_displacement_m'])<1e-12
        assert abs(end/exact-1)<.03
        assert abs(b.energy()-b.work)<1e-6
    else:
        potential=b.forces(plane=r['plane_m'])[3]
        assert abs(b.energy()+potential-r['initial_contact_energy_J'])<.01*max(1e-12,r['initial_contact_energy_J'])
        if r['mode']=='separation':assert b.energy()==0 and r['tool_impulse_Ns']==0
    rows.append(dict(file=name,raw_field_audit_passed=True))
print(json.dumps(dict(cases=rows,physical_accuracy_qualified=False),indent=2))
