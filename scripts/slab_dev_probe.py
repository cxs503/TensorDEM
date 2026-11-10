"""Run one slab-slice configuration for MT Uikku case 103 and print the numbers.

Used during development; the campaign driver is
``scripts/run_ice_channel3d_uikku.py``.
"""
import json
import sys
import time

from tensordem.ice_channel3d.slab import HullFootprint, SlabChannelSim, SlabConfig
from tensordem.ice_channel3d.uikku import ice_props_of, load_reference_cases

CONFIGS = {
    "corr": {},
    "single_layer": {"n_layers": 1},
    "legacy": {"lateral_open": False, "n_layers": 1, "instant_removal": True},
    "no_opening": {"lateral_open": False},
    "r1.5": {"floe_radius_m": 1.5},
    "r3.0": {"floe_radius_m": 3.0},
    "layers2": {"n_layers": 2},
    "layers4": {"n_layers": 4},
    "WB1": {"channel_width_over_beam": 1.0},
    "WB2": {"channel_width_over_beam": 2.0},
    "WB0.3": {"channel_width_over_beam": 0.3},
    "no_subduction": {"subduction": False},
    "uncoupled": {"layer_moment_coupling": False},
}


def main(names):
    case = load_reference_cases()[0]
    hull = HullFootprint(beam_m=20.0, length_m=120.0, bow_half_angle_deg=22.0)
    base = dict(n_layers=3, frames=int(sys.argv[2]) if len(sys.argv) > 2 else 200,
                drift_speed_m_s=case["drift_speed_m_s"], relax_iterations=6)
    for name in names:
        cfg = SlabConfig(**{**base, **CONFIGS[name]})
        t0 = time.time()
        s = SlabChannelSim(cfg, ice_props_of(case), hull=hull, case_id="103").run()
        print("%-13s R=%8.0f tot=%8.0f err=%+7.0f%% brk=%6.0f crs=%7.0f fr=%5.0f "
              "cont=%5.1f nb=%4d ns=%4d del=%7.0f masserr=%.1e clos=%.3f t=%.1fs"
              % (name, s.resistance_mean_kN, s.total_with_closures_kN,
                 (s.total_with_closures_kN - 470) / 470 * 100, s.break_component_kN,
                 s.crush_component_kN, s.friction_component_kN,
                 s.contacts_in_contact_set, s.breaking_events, s.subduction_events,
                 s.deleted_kg, s.max_frame_mass_error, s.flux_closure_ratio,
                 time.time() - t0), flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","))