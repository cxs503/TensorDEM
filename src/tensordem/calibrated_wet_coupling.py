"""Opt-in actual per-bond DEM + legacy permeable point-friction LBM."""
from dataclasses import asdict, replace
import copy
import math
import torch
from tensorlbm.ice_coupling_2d import CoupledIce2D, CoupledIceConfig
from .dem import DEMConfig, IceDEM
from .calibrated_dem import CalibratedIceDEM
from .material_calibration import corrected_stiffness

class CalibratedWetCoupling(CoupledIce2D):
    def __init__(self, config=CoupledIceConfig(), *, young_modulus_Pa=1000., total_mass_kg=12.3795):
        for v in (young_modulus_Pa,total_mass_kg):
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
                raise ValueError('material and mass must be finite positive SI')
        super().__init__(config)
        self.material=dict(young_modulus_Pa=float(young_modulus_Pa),total_mass_kg=float(total_mass_kg),poisson_ratio=1/3)
        density=total_mass_kg/(config.ice_nx*config.ice_ny*math.pi*config.ice_radius_m**2*.2)
        dc=DEMConfig(nx=config.ice_nx,ny=config.ice_ny,radius=config.ice_radius_m,thickness=.2,
            density=density,bond_stiffness=.75*young_modulus_Pa*.2,contact_stiffness=150.,
            contact_damping=.1,drag=0.,tool_speed=config.tool_speed_m_s,tool_gap=config.tool_gap_m,
            breaking_strain=config.breaking_strain,shear_breaking_strain=config.shear_breaking_strain)
        self.substeps=math.ceil(config.fluid_dt_s/dc.recommended_dt)
        dc=replace(dc,dt=config.fluid_dt_s/self.substeps)
        prototype=IceDEM(dc)
        coeff=corrected_stiffness(prototype.positions,prototype.bond_indices,young_modulus_Pa,dc.thickness)
        self.dem=CalibratedIceDEM(dc,bond_stiffness=coeff)
        # Keep the actual disk exterior at y in [0,.15] across refinement.
        shift=torch.tensor([0.,config.ice_radius_m],dtype=torch.float64)
        self.origin_offset+=shift
        self.grid_xy-=shift
        self.total_initial_momentum=self.total_momentum().clone()
        self.initial_dem_energy=self.dem.mechanical_energy()['mechanical_J']
        self.initial_fluid_energy=self.fluid_energy()

    def snapshot(self):
        return dict(schema='tensordem.calibrated-wet-restart/1',material=self.material,
                    coupled=super().snapshot())

    @classmethod
    def from_snapshot(cls, record):
        if not isinstance(record,dict) or set(record)!={'schema','material','coupled'} or record['schema']!='tensordem.calibrated-wet-restart/1':
            raise ValueError('invalid calibrated wet restart')
        if set(record['material'])!={'young_modulus_Pa','total_mass_kg','poisson_ratio'} or record['material']['poisson_ratio']!=1/3:
            raise ValueError('invalid calibrated material')
        config=CoupledIceConfig(**record['coupled']['config'])
        obj=cls(config,young_modulus_Pa=record['material']['young_modulus_Pa'],total_mass_kg=record['material']['total_mass_kg'])
        dem=CalibratedIceDEM.from_snapshot(record['coupled']['dem'])
        if asdict(dem.config)!=asdict(obj.dem.config) or dem.coefficient_hash!=obj.dem.coefficient_hash or record['coupled']['substeps']!=obj.substeps:
            raise ValueError('material/subcycling mismatch')
        # Use the established complete fluid/held-map/clock validator with its
        # own uncalibrated initialization count, then restore the actual backend.
        legacy=copy.deepcopy(record['coupled'])
        legacy['dem']=legacy['dem']['base']
        legacy['substeps']=CoupledIce2D(config).substeps
        restored=CoupledIce2D.from_snapshot(legacy)
        material,substeps=obj.material,obj.substeps
        origin_offset,grid_xy=obj.origin_offset,obj.grid_xy
        obj.__dict__.update(restored.__dict__)
        obj.dem,obj.material,obj.substeps=dem,material,substeps
        obj.origin_offset,obj.grid_xy=origin_offset,grid_xy
        expected_sample=(max(obj.step_index-1,0)//config.exchange_steps)*config.exchange_steps*config.fluid_dt_s
        if abs(obj.exchange_sample_time-expected_sample)>1e-10:
            raise ValueError('held exchange clock mismatch')
        return obj
