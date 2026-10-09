import hashlib,json,math
from pathlib import Path
import torch
from tensordem.moving_plane_boundary import MovingPlaneNetwork,smooth_plane
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/coupling/smooth-plane'

def run(nx,factor=1.):
    b=MovingPlaneNetwork(nx,nx//4);duration=.8;steps=math.ceil(duration/(b.dt*factor));dt=duration/steps
    history=[];quadrature=[];restart=None;maximum_error=0.;peak=0.
    for i in range(steps):
        a=smooth_plane(i*dt);z=smooth_plane((i+1)*dt)
        reaction=b.moving_step(dt,a,z);quadrature.append([reaction,z-a,dt])
        if i==steps//2:restart=MovingPlaneNetwork.restore(json.loads(json.dumps(b.snapshot())))
        elif restart is not None:restart.moving_step(dt,a,z)
        energy=b.energy()+b.forces(plane=z)[3];error=energy-b.tool_work
        maximum_error=max(maximum_error,abs(error));peak=max(peak,reaction)
        if i%max(1,steps//100)==0 or i==steps-1:
            history.append(dict(time_s=b.time,plane_m=z,tool_force_N=reaction,tool_work_J=b.tool_work,mechanical_and_contact_energy_J=energy,energy_error_J=error))
    end=float((b.u[b.right,0]*b.areas).sum()/b.areas.sum())
    return dict(nx=nx,dt_factor=factor,steps=steps,dt_s=dt,duration_s=duration,ramp_s=.5,depth_m=.001,mass_kg=float(b.mass.sum()),tool_impulse_Ns=b.tool_impulse,tool_work_J=b.tool_work,end_displacement_m=end,peak_force_N=peak,maximum_energy_residual_J=maximum_error,maximum_energy_residual_relative_to_final_work=maximum_error/abs(b.tool_work),momentum_residual_Ns=float(((b.mass[:,None]*b.v).sum(0)-b.impulse).abs().max()),restart_exact=b.snapshot()==restart.snapshot(),history=history,tool_quadrature_force_displacement_dt=quadrature,final=b.snapshot(),positions_m=b.x.tolist(),mass_nodes_kg=b.mass.tolist(),edges=b.edges.tolist(),stiffness_N_per_m=b.k.tolist(),physical_accuracy_qualified=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for n,f in [(16,1.),(32,1.),(64,1.),(64,.5)]:
        r=run(n,f);name=f'grid-{n}'+('-half' if f==.5 else '')+'.json';(OUT/name).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
        rows.append({k:r[k] for k in ('nx','dt_factor','steps','dt_s','tool_impulse_Ns','tool_work_J','end_displacement_m','maximum_energy_residual_J','maximum_energy_residual_relative_to_final_work','momentum_residual_Ns','restart_exact')})
    s=dict(records=rows,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/tensordem/continuum_boundary_dem.py',ROOT/'src/tensordem/moving_plane_boundary.py']},raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('grid-*.json')},physical_accuracy_qualified=False)
    for key in ('tool_impulse_Ns','end_displacement_m'):
        s[key+'_refinement_relative']=[abs(rows[i][key]/rows[i-1][key]-1) for i in (1,2)]
        s[key+'_time_half_relative']=abs(rows[3][key]/rows[2][key]-1)
    s['dynamic_3_percent_passed']=all(s[k+'_refinement_relative'][-1]<.03 for k in ('tool_impulse_Ns','end_displacement_m'))
    (OUT/'study.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
if __name__=='__main__':main()
