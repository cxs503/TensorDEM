"""Exact Neumann slab mode superposed on spatially uniform ramp heating."""
import json
from pathlib import Path
import subprocess

import numpy as np

from tensordem.thermal import affine_contact_interval


def measure(n):
    dx,k,c,dt = 1/n,1.2,2.1,.19
    faces = np.linspace(0.,1.,n+1)
    initial = np.diff(np.sin(np.pi*faces))/(np.pi*dx)
    capacity = np.full(n,c*dx)
    heating,slope = .7,-.4
    end,heat,mean = affine_contact_interval(np.full(n-1,dx/k),capacity,dt,initial,
                t_left=None,t_right=None,nodal_power_start=capacity*heating,
                nodal_power_end=capacity*(heating+slope*dt),return_interval_temperature=True)
    rate = k*np.pi**2/c
    mode_integral = -np.expm1(-rate*dt)/rate
    final_rise = dt*(heating+.5*slope*dt)
    mean_rise = dt*(heating/2+slope*dt/6)
    return dict(cells=n,temperature_error_K=float(abs(end-final_rise-np.exp(-rate*dt)*initial).max()),
                interval_temperature_error_K=float(abs(mean-mean_rise-mode_integral/dt*initial).max()),
                link_heat_error_J=float(abs(heat-k*np.pi*np.sin(np.pi*faces[1:-1])*mode_integral).max()),
                node_balance_J=float(abs(capacity*(end-initial-final_rise)+np.diff(np.r_[0.,heat,0.])).max()),
                enthalpy_balance_J=float(abs(capacity@(end-initial-final_rise))))


if __name__=="__main__":
    rows = []
    for n in (8,16,32,64,128):
        row = measure(n)
        orders = {key:float(np.log2(rows[-1][key]/value)) for key,value in row.items()
                  if rows and "_error_" in key}
        row.update(orders=orders,passed=max(row["node_balance_J"],row["enthalpy_balance_J"])<1e-12
                   and all(p>1.9 for p in orders.values()))
        rows.append(row)
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
