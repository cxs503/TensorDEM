"""Actual dynamic fixed-span load and stationary-plane contact refinement."""
import hashlib,json,math
from pathlib import Path
import torch
from tensordem.continuum_boundary_dem import BoundaryNetwork

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/coupling/continuum-boundary'

def run(nx,mode,dt_factor=1.):
    b=BoundaryNetwork(nx,nx//4);duration=.12;steps=math.ceil(duration/(b.dt*dt_factor));dt=duration/steps
    traction=100. if mode=='load' else 0.
    plane=.999 if mode=='contact' else (1.001 if mode=='separation' else None)
    initial_contact=b.forces(traction,plane)[3]
    history=[];restart=None;max_energy=max_momentum=0.;peak=0.;tool_impulse=0.
    for i in range(steps):
        reaction=b.step(dt,traction,plane);tool_impulse+=reaction*dt
        if i==steps//2: restart=BoundaryNetwork.restore(json.loads(json.dumps(b.snapshot())))
        elif restart is not None:restart.step(dt,traction,plane)
        energy=b.energy();contact=b.forces(traction,plane)[3]
        max_energy=max(max_energy,abs(energy+contact-initial_contact) if mode!='load' else abs(energy-b.work))
        momentum=(b.mass[:,None]*b.v).sum(0)
        max_momentum=max(max_momentum,float((momentum-b.impulse).abs().max()))
        peak=max(peak,abs(reaction))
        if i%max(1,steps//100)==0 or i==steps-1:
            history.append(dict(time_s=b.time,end_displacement_m=float((b.u[b.right,0]*b.areas).sum()/b.areas.sum()),reaction_N=reaction,energy_J=energy,contact_energy_J=contact,external_work_J=b.work))
    exact=None
    if mode=='load':
        c=math.sqrt(9*b.config['E']/8/b.config['density'])
        series=sum(8/math.pi**2/(2*j+1)**2*math.cos((2*j+1)*math.pi*c*duration/2) for j in range(20000))
        exact=traction/(9*b.config['E']/8)*(1-series)
    end=float((b.u[b.right,0]*b.areas).sum()/b.areas.sum())
    return dict(schema='tensordem.boundary-study/1',mode=mode,nx=nx,ny=nx//4,dt_factor=dt_factor,tool_impulse_Ns=tool_impulse,dt_s=dt,steps=steps,mass_kg=float(b.mass.sum()),duration_s=duration,peak_force_N=peak,end_displacement_m=end,analytic_end_displacement_m=exact,analytic_error_relative=None if exact is None else abs(end-exact)/abs(exact),maximum_energy_residual_J=max_energy,maximum_momentum_residual_Ns=max_momentum,restart_exact=bool(torch.equal(b.u,restart.u) and torch.equal(b.v,restart.v) and b.snapshot()==restart.snapshot()),history=history,final=b.snapshot(),positions_m=b.x.tolist(),mass_nodes_kg=b.mass.tolist(),edges=b.edges.tolist(),stiffness_N_per_m=b.k.tolist(),plane_m=plane,initial_contact_energy_J=initial_contact,physical_accuracy_qualified=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True);records=[]
    for mode in ('load','contact','separation'):
        for nx in (16,32,64):
            r=run(nx,mode);p=OUT/f'{mode}-{nx}.json';p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');records.append({k:r[k] for k in ('mode','nx','tool_impulse_Ns','end_displacement_m','mass_kg','dt_s','steps','peak_force_N','analytic_error_relative','maximum_energy_residual_J','maximum_momentum_residual_Ns','restart_exact')})
    half=run(64,'contact',.5);(OUT/'contact-64-half.json').write_text(json.dumps(half,indent=2,allow_nan=False)+'\n')
    summary=dict(records=records,source_sha256=hashlib.sha256((ROOT/'src/tensordem/continuum_boundary_dem.py').read_bytes()).hexdigest(),files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*-*.json'))},scope='fixed reference boundary control-volume linear network; no moving disk or fracture qualification',physical_accuracy_qualified=False)
    contact=[r['peak_force_N'] for r in records if r['mode']=='contact'];summary['contact_peak_refinement_relative']=[abs(contact[i]/contact[i-1]-1) for i in (1,2)];summary['contact_peak_3_percent_passed']=summary['contact_peak_refinement_relative'][-1]<.03
    impulse=[r['tool_impulse_Ns'] for r in records if r['mode']=='contact'];summary['contact_impulse_refinement_relative']=[abs(impulse[i]/impulse[i-1]-1) for i in (1,2)]
    summary['contact_impulse_3_percent_passed']=summary['contact_impulse_refinement_relative'][-1]<.03
    summary['contact_time_half_impulse_relative']=abs(half['tool_impulse_Ns']/impulse[-1]-1)
    (OUT/'study.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
