"""Execute material-consistent sensitivity, preserving failed comparisons."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import torch
from tensorlbm.ice_coupling_2d import CoupledIceConfig
from tensordem.calibrated_wet_coupling import CalibratedWetCoupling
import tensordem.calibrated_wet_coupling as cw
import tensordem.calibrated_dem as cd
import tensordem.dem as dm
import tensordem.material_calibration as mc
import tensorlbm.ice_coupling_2d as lc
import tensorlbm.solver as ls
import tensorlbm.d2q9 as dq
import tensorlbm.icebreaking as ib

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():
    modules=[cw,cd,dm,mc,lc,ls,dq,ib]
    return {m.__name__:sha(m.__file__) for m in modules}|{'tensordem.hull_contact':sha(Path(dm.__file__).with_name('hull_contact.py')), 'scripts/calibrated_wet_study.py':sha(__file__)}
def main():
    out=Path('docs/calibrated-wet');out.mkdir(exist_ok=True,parents=True)
    cases={}
    for nx in (6,12,18):
        c=CoupledIceConfig(ice_nx=nx,ice_ny=nx//3,ice_radius_m=.45/(2*nx),duration_s=.03,
              tool_gap_m=.0001,breaking_strain=100.,shear_breaking_strain=100.,exchange_steps=2)
        cases['wet_'+str(nx)]=c
    cases['wet_12_time_half']=replace(cases['wet_12'],fluid_dt_s=.00025,exchange_steps=4)
    for nx in (6,12,18):cases['dry_'+str(nx)]=replace(cases['wet_'+str(nx)],wet=False)
    cases['fracture_12']=replace(cases['wet_12'],duration_s=.15,breaking_strain=.015,shear_breaking_strain=.03)
    study=dict(schema='tensordem.calibrated-wet-study/1',source_sha256=sources(),cases={},artifacts_sha256={},physical_accuracy_pass=False,
       scope='Actual per-bond E=1000Pa nu=1/3 dynamics, point friction periodic liquid, artificial viscosity .02m2/s; no impermeable boundary or calibrated fracture',
       invariants=dict(outer_disk_envelope_m=[.45,.15],total_mass_kg=12.3795,thickness_m=.2,young_modulus_Pa=1000.,poisson_ratio=1/3),
       boundary_caveat='fixed first/last center columns move inward by radius; central spring material spans shrink by 2radius; homogeneous disk mass, density compensates overlap convention; contact damping/stiffness fixed but particle contact discretization changes')
    for name,c in cases.items():
        sim=CalibratedWetCoupling(c)
        count=round(c.duration_s/c.fluid_dt_s)
        for i in range(count):
            sim.step()
            if i==count//2:
                restarted=CalibratedWetCoupling.from_snapshot(json.loads(json.dumps(sim.snapshot())))
            elif i>count//2:
                restarted.step()
        restart=sim.snapshot()==restarted.snapshot()
        h=sim.history
        result=dict(peak_force_N=max(abs(v['total_fy_n']) for v in h),impulse_Ns=sum(v['total_fy_n']*c.fluid_dt_s for v in h),
              broken_bonds=sim.dem.broken_bonds,substeps=sim.substeps,restart_bitwise=restart,
              max_momentum_error_kg_m_s=max(max(abs(x) for x in v['momentum_residual_kg_m_s']) for v in h),
              max_mass_relative_error=max(v['mass_relative_error'] for v in h),
              final_dem_energy_residual_J=h[-1]['dem_discrete_energy_residual_J'],
              final_interface_work_residual_J=h[-1]['staggered_interface_work_residual_J'],
              max_force_mapping_error_N=max(v['force_mapping_error_N'] for v in h),
              max_moment_mapping_error_Nm=max(v['moment_mapping_error_Nm'] for v in h),
              max_adjoint_power_error_W=max(v['adjoint_power_error_W'] for v in h))
        assert restart
        path=out/(name+'.json');path.write_text(json.dumps(dict(summary=result,state=sim.snapshot()),allow_nan=False)+'\n')
        study['cases'][name]=result;study['artifacts_sha256'][path.name]=sha(path)
        print(name,result,flush=True)
    study['sensitivity']={name:{k:abs(study['cases'][name][k]/study['cases'][base][k]-1) for k in ('peak_force_N','impulse_Ns')}
       for name,base in [('wet_12','wet_6'),('wet_18','wet_12'),('wet_12_time_half','wet_12')]}
    study['three_percent_sensitivity_pass']=all(v<.03 for r in study['sensitivity'].values() for v in r.values())
    (out/'study.json').write_text(json.dumps(study,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
