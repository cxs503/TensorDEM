"""Independent field/hash audit; numerical sensitivity is never promoted."""
import json
from pathlib import Path
import torch
from calibrated_wet_study import sources,sha
from tensordem.calibrated_wet_coupling import CalibratedWetCoupling

def main():
    out=Path('docs/calibrated-wet');study=json.loads((out/'study.json').read_text())
    assert study['source_sha256']==sources()
    for name,summary in study['cases'].items():
        path=out/(name+'.json');assert sha(path)==study['artifacts_sha256'][path.name]
        raw=json.loads(path.read_text());assert raw['summary']==summary
        sim=CalibratedWetCoupling.from_snapshot(raw['state'])
        assert sim.dem.config.mass*len(sim.dem.positions)==study['invariants']['total_mass_kg'] or abs(sim.dem.config.mass*len(sim.dem.positions)-study['invariants']['total_mass_kg'])<1e-12
        h=sim.history;c=sim.config
        assert max(abs(v['total_fy_n']) for v in h)==summary['peak_force_N']
        assert sum(v['total_fy_n']*c.fluid_dt_s for v in h)==summary['impulse_Ns']
        assert sim.dem.broken_bonds==summary['broken_bonds']
        p=sim.total_momentum()-sim.total_initial_momentum-sim.external_impulse
        assert torch.allclose(p,torch.tensor(h[-1]['momentum_residual_kg_m_s'],dtype=torch.float64),atol=1e-14,rtol=0)
        energy=sim.dem.mechanical_energy()['mechanical_J']+sim.dem.accounting['damping_dissipation_J']+sim.dem.accounting['fracture_release_J']-sim.initial_dem_energy-sim.dem.accounting['external_work_J']-sim.dem.accounting['tool_work_J']
        assert abs(energy-summary['final_dem_energy_residual_J'])<1e-14
        assert summary['restart_bitwise'] and summary['max_momentum_error_kg_m_s']<1e-9
        print(name,'field/hash audit passed')
    for name,base in [('wet_12','wet_6'),('wet_18','wet_12'),('wet_12_time_half','wet_12')]:
        for k in ('peak_force_N','impulse_Ns'):
            assert study['sensitivity'][name][k]==abs(study['cases'][name][k]/study['cases'][base][k]-1)
    assert study['physical_accuracy_pass'] is False
    print('Audit passed; three-percent sensitivity:',study['three_percent_sensitivity_pass'])
if __name__=='__main__':main()
