"""Continuum decay of two material layers with a nonzero interface temperature jump."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from tensordem.thermal import cell_centered_slab_network, constant_contact_interval


def measure(n):
    # Both sine branches vanish at the exterior walls. Select the right
    # diffusivity for equal decay rates and match the rightward heat flux.
    a,b = math.pi/2,1.4*math.pi
    k_left,k_right = 1.,.25
    rate = a*a
    cv_right = k_right*b*b/rate
    interface_power = -k_left*a*math.cos(a/2)
    amplitude = interface_power/(k_right*b*math.cos(b/2))
    traces = np.array([math.sin(a/2),amplitude*math.sin(b/2)])
    contact = (traces[0]-traces[1])/interface_power
    widths = np.full(n,1/n)
    x = (np.arange(n)+.5)/n
    means = np.where(x<.5,np.sin(a*x)*np.sinc(a/(2*math.pi*n)),
                     amplitude*np.sin(b*(1-x))*np.sinc(b/(2*math.pi*n)))
    resistances = np.zeros(n-1)
    resistances[n//2-1] = contact
    r,c = cell_centered_slab_network(widths,np.where(x<.5,k_left,k_right),
                  np.where(x<.5,1.,cv_right),contact_resistance=resistances,resolve_contacts=True)
    initial = np.zeros_like(c)
    initial[c>0] = means
    dt = .35/rate
    end,heat = constant_contact_interval(r,c,dt,initial,t_left=0.,t_right=0.,allow_massless=True)
    decay = math.exp(-rate*dt)
    interface = n//2+1
    observed_traces = end[interface:interface+2]
    return dict(cells=n,resistance_m2_K_W=contact,
                cell_error_K=float(abs(end[c>0]-decay*means).max()),
                trace_error_K=float(abs(observed_traces-decay*traces).max()),
                interface_heat_error_J=abs(float(heat[interface])-interface_power*(1-decay)/rate),
                node_balance_J=float(abs((c*(end-initial))[1:-1]+np.diff(heat)).max()))


if __name__=="__main__":
    rows = []
    previous = {}
    for n in (8,16,32,64,128):
        row = measure(n)
        errors = {key:val for key,val in row.items() if "_error_" in key}
        orders = {key:math.log2(previous[key]/val) for key,val in errors.items()} if previous else {}
        row.update(orders=orders,passed=row["node_balance_J"]<1e-11 and
                   (not orders or min(orders.values())>1.8))
        rows.append(row)
        previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
