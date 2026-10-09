import copy
import json
import pytest
import torch
from tensorlbm.ice_coupling_2d import CoupledIceConfig
from tensordem.calibrated_wet_coupling import CalibratedWetCoupling

def test_held_exchange_restart():
    c=CoupledIceConfig(ice_nx=6,ice_ny=2,ice_radius_m=.0375,exchange_steps=3,tool_gap_m=.0001,breaking_strain=100.,shear_breaking_strain=100.)
    a=CalibratedWetCoupling(c)
    a.step();a.step()
    b=CalibratedWetCoupling.from_snapshot(json.loads(json.dumps(a.snapshot())))
    for _ in range(5):
        assert a.step()==b.step()
        assert torch.equal(a.f,b.f)
        assert torch.equal(a.dem.positions,b.dem.positions)
    assert a.dem.coefficient_hash==b.dem.coefficient_hash
    assert a.exchange_sample_time==b.exchange_sample_time
    assert a.dem.config.mass*len(a.dem.positions)==pytest.approx(12.3795)

@pytest.mark.parametrize('field',['coeff','clock','substeps','population','exchange_clock'])
def test_corrupt_restart(field):
    a=CalibratedWetCoupling(CoupledIceConfig(ice_nx=6,ice_ny=2,ice_radius_m=.0375))
    a.step();state=copy.deepcopy(a.snapshot())
    if field=='coeff':state['coupled']['dem']['bond_stiffness_N_per_m'][0]*=1.1
    if field=='clock':state['coupled']['dem']['base']['time_s']+=1.
    if field=='substeps':state['coupled']['substeps']+=1
    if field=='exchange_clock':state['coupled']['scalars']['exchange_sample_time']+=1.
    if field=='population':state['coupled']['f'][0][0][0]=-1.
    with pytest.raises(ValueError):CalibratedWetCoupling.from_snapshot(state)
