"""Exact Neumann slab mode superposed on spatially uniform ramp heating."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from tensordem.thermal import affine_contact_interval, harmonic_contact_interval


def measure(n,harmonic=False):
    dx,k,c,dt = 1/n,1.2,2.1,.19
    faces = np.linspace(0.,1.,n+1)
    initial = np.diff(np.sin(np.pi*faces))/(np.pi*dx)
    capacity = np.full(n,c*dx)
    heating,slope = .7,-.4
    links = np.full(n-1,dx/k)
    rate = k*np.pi**2/c
    mode_integral = -np.expm1(-rate*dt)/rate
    end_mode = np.exp(-rate*dt)
    if harmonic:
        omega,start = 6.,.14
        uniform,modal = .6+.4j,-.7+.5j
        phase = np.exp(1j*omega*start)
        phase_change = phase*np.expm1(1j*omega*dt)
        integrated_phase = phase_change/(1j*omega)
        source = capacity*(uniform+modal*initial)
        end,heat,mean = harmonic_contact_interval(links,capacity,omega,start,dt,initial,
                    t_left=None,t_right=None,nodal_power=source,return_interval_temperature=True)
        final_rise = (uniform*integrated_phase).real
        mean_rise = (uniform/(1j*omega)*(integrated_phase/dt-phase)).real
        amplitude = modal/(1j*omega+rate)
        transient = 1-(amplitude*phase).real
        mode_integral = transient*mode_integral+(amplitude*integrated_phase).real
        end_mode = transient*end_mode+(amplitude*(phase+phase_change)).real
        supplied = (source*integrated_phase).real
    else:
        end,heat,mean = affine_contact_interval(links,capacity,dt,initial,t_left=None,t_right=None,
                    nodal_power_start=capacity*heating,nodal_power_end=capacity*(heating+slope*dt),
                    return_interval_temperature=True)
        final_rise = dt*(heating+.5*slope*dt)
        mean_rise = dt*(heating/2+slope*dt/6)
        supplied = capacity*final_rise
    stored = capacity*(end-initial)
    return dict(cells=n,temperature_error_K=float(abs(end-final_rise-end_mode*initial).max()),
                interval_temperature_error_K=float(abs(mean-mean_rise-mode_integral/dt*initial).max()),
                link_heat_error_J=float(abs(heat-k*np.pi*np.sin(np.pi*faces[1:-1])*mode_integral).max()),
                node_balance_J=float(abs(stored-supplied+np.diff(np.r_[0.,heat,0.])).max()),
                enthalpy_balance_J=float(abs(stored.sum()-supplied.sum())))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmonic",action="store_true",help="drive uniform and cosine modes with sinusoidal power")
    args = parser.parse_args()
    rows = []
    for n in (8,16,32,64,128):
        row = measure(n,args.harmonic)
        orders = {key:float(np.log2(rows[-1][key]/value)) for key,value in row.items()
                  if rows and "_error_" in key}
        row.update(orders=orders,passed=max(row["node_balance_J"],row["enthalpy_balance_J"])<1e-12
                   and all(p>1.9 for p in orders.values()))
        rows.append(row)
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,harmonic=args.harmonic,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
