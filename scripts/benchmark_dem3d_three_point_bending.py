"""DEM3D three-point-bending numerical integration fixture (not material calibration)."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys
from pathlib import Path
from typing import Any
import torch
ROOT=Path(__file__).resolve().parents[1]
for candidate in (ROOT,ROOT/"src"):
    if str(candidate) not in sys.path: sys.path.insert(0,str(candidate))
from tensordem.dem3d import DEM3DConfig, IceDEM3D

def run_three_point_bending(output: Path, *, steps: int|None=None, peak_load_N: float|None=None)->dict[str,Any]:
    torch.set_num_threads(1)
    manifest=json.loads((ROOT/"benchmarks/dem3d/three_point_bending_cases.json").read_text(encoding="utf-8"))
    if manifest.get("protocol")!="tensordem-dem3d-three-point-bending-v1": raise ValueError("unsupported three-point-bending protocol")
    case=manifest["cases"][0]; p=case["config"]
    nsteps=int(p["steps"] if steps is None else steps); load_max=float(p["peak_load_N"] if peak_load_N is None else peak_load_N)
    if nsteps<1: raise ValueError("steps must be positive")
    if not math.isfinite(load_max) or load_max<=0: raise ValueError("peak_load_N must be finite and positive")
    cfg=DEM3DConfig(nx=int(p["nx"]),ny=int(p["ny"]),nz=int(p["nz"]),radius=float(p["radius_m"]),density=float(p["density_kg_m3"]),bond_stiffness=float(p["bond_stiffness_N_m"]),contact_stiffness=float(p["contact_stiffness_N_m"]),breaking_strain=float(p["breaking_strain"]),shear_breaking_strain=float(p["shear_breaking_strain"]),contact_damping=float(p["contact_damping_Ns_m"]),drag=float(p["drag"]),fix_x_edges=False,fix_bottom=False,tool_gap=1.0,tool_radius=0.01,tool_speed=0.01)
    sim=IceDEM3D(cfg); sim.tool_start[:]=torch.tensor([1e6,1e6,1e6],dtype=sim.dtype); sim.tool_velocity.zero_()
    x=sim.initial_positions[:,0]; z=sim.initial_positions[:,2]; xv=torch.unique(x); zb,zt=z.min(),z.max()
    lx=xv[max(1,len(xv)//4)]; rx=xv[min(len(xv)-2,(3*len(xv))//4)]
    supports=(z==zb)&((x==lx)|(x==rx)); loaded=(z==zt)&(x==xv[len(xv)//2])
    if int(supports.sum())<2 or int(loaded.sum())<1: raise ValueError("invalid support/loading geometry")
    sim.fixed[:]=supports; sim.positions[sim.fixed]=sim.initial_positions[sim.fixed]
    rows=[]; digest=hashlib.sha256(); nload=int(loaded.sum())
    for step in range(nsteps+1):
        applied=load_max*step/nsteps; loads=torch.zeros_like(sim.positions); loads[loaded,2]=-applied/nload
        if step: sim.step(external_forces=loads)
        force,_=sim.forces(loads,update_fracture=False); reaction=-force[sim.fixed].sum(dim=0)
        deflection=float((sim.positions[loaded,2]-sim.initial_positions[loaded,2]).mean()); d=sim.diagnostics()
        row={"step":step,"time_s":float(d["time"]),"applied_load_N":applied,"support_reaction_z_N":float(reaction[2]),"vertical_force_balance_residual_N":float(reaction[2]-applied),"midspan_deflection_m":deflection,"broken_bonds":int(d["broken_bonds"]),"kinetic_energy_J":float(d["kinetic_energy_J"]),"bond_elastic_energy_J":float(d["bond_elastic_energy_J"]),"mechanical_energy_J":float(d["mechanical_energy_J"])}
        if not all(math.isfinite(float(v)) for v in row.values()): raise FloatingPointError(f"non-finite bending output at step {step}")
        rows.append(row); digest.update(json.dumps(row,sort_keys=True,separators=(",",":")).encode())
    peak_reaction=max(abs(float(r["support_reaction_z_N"])) for r in rows); peak_deflection=max(abs(float(r["midspan_deflection_m"])) for r in rows)
    checks={"support_load_transfer":peak_reaction>float(case["acceptance"]["minimum_peak_support_reaction_N"]),"nonzero_midspan_deflection":peak_deflection>float(case["acceptance"]["minimum_peak_deflection_m"]),"finite_history":all(math.isfinite(float(r["mechanical_energy_J"])) for r in rows),"fracture_count_monotonic":all(int(rows[i]["broken_bonds"])<=int(rows[i+1]["broken_bonds"]) for i in range(len(rows)-1))}
    failed=[k for k,v in checks.items() if not v]
    report={"protocol":manifest["protocol"],"case_id":case["id"],"verdict":"PASS" if not failed else "FAIL","checks":checks,"failed_checks":failed,"steps":nsteps,"dt_s":sim.dt,"particle_count":len(sim.positions),"bond_count":len(sim.pairs),"support_particle_count":int(supports.sum()),"loading_particle_count":nload,"peak_applied_load_N":load_max,"peak_support_reaction_abs_N":peak_reaction,"peak_midspan_deflection_abs_m":peak_deflection,"broken_bonds":sim.broken_bonds,"signature_sha256":digest.hexdigest(),"rows":rows,"interpretation":"PASS means finite numerical load transfer only. Supports are fixed particles and the nodal load is distributed over the center top row; no experimental validation or calibrated flexural strength is implied."}
    output.mkdir(parents=True,exist_ok=True)
    with (output/"three_point_bending_history.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/"three_point_bending_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if failed: raise RuntimeError(f"three-point-bending benchmark failed: {failed}")
    return report

def main()->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,default=Path("results-dem3d-three-point-bending")); parser.add_argument("--steps",type=int,default=None); parser.add_argument("--peak-load-N",type=float,default=None); a=parser.parse_args()
    r=run_three_point_bending(a.output,steps=a.steps,peak_load_N=a.peak_load_N); print(json.dumps({k:v for k,v in r.items() if k!="rows"},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
