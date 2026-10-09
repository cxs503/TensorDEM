"""Audit tool quadrature and actual final fields of smooth plane refinement."""
import hashlib,json
from pathlib import Path
import torch
from tensordem.moving_plane_boundary import MovingPlaneNetwork,smooth_plane
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/coupling/smooth-plane'
s=json.loads((OUT/'study.json').read_text());rows=[]
for name,digest in s['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
for name,digest in s['raw_sha256'].items():
    p=OUT/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    r=json.loads(p.read_text());b=MovingPlaneNetwork.restore(r['final'])
    for field,actual in [('positions_m',b.x),('mass_nodes_kg',b.mass),('edges',b.edges),('stiffness_N_per_m',b.k)]:assert torch.equal(actual,torch.tensor(r[field],dtype=actual.dtype))
    assert abs(float(b.mass.sum())-22.5)<1e-12
    assert float(b.u[b.fixed].abs().max())==0 and float(b.v[b.fixed].abs().max())==0
    assert float(((b.mass[:,None]*b.v).sum(0)-b.impulse).abs().max())<1e-12
    energy=b.energy()+b.forces(plane=smooth_plane(r['duration_s']))[3]
    assert abs(energy-r['history'][-1]['mechanical_and_contact_energy_J'])<1e-14
    q=r['tool_quadrature_force_displacement_dt'];work=0.;impulse=0.
    assert len(q)==r['steps']
    for i,(reaction,dplane,dt) in enumerate(q):
        assert abs(dplane-(smooth_plane((i+1)*dt)-smooth_plane(i*dt)))<1e-15
        work-=reaction*dplane;impulse+=reaction*dt
    assert abs(work-b.tool_work)<1e-14 and abs(impulse-b.tool_impulse)<1e-13
    assert abs(energy-work)/abs(work)<.01
    assert r['restart_exact']
    end=float((b.u[b.right,0]*b.areas).sum()/b.areas.sum())
    assert abs(end-r['end_displacement_m'])<1e-15
    rows.append(dict(file=name,raw_field_and_tool_work_audit_passed=True,final_energy_error_J=energy-work))
print(json.dumps(dict(cases=rows,dynamic_3_percent_passed=s['dynamic_3_percent_passed'],physical_accuracy_qualified=False),indent=2))
