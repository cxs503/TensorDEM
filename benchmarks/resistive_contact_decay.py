"""Continuum decay of two material layers with a nonzero interface temperature jump."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from tensordem.thermal import affine_contact_interval, cell_centered_slab_network, constant_contact_interval


def measure(n, ramp=False):
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
    decay = math.exp(-rate*dt)
    interface = n//2+1
    extra = {}
    if ramp:
        # Superpose a rightward flux q(t)=.2-.7t on the decay mode.
        # A linear background in resistance coordinates has exactly this
        # flux across every link, including the two massless contact traces.
        distance = np.r_[0.,np.cumsum(r)]
        base = .4-.2*distance
        slope = .15+.7*distance
        initial += base
        end,heat,mean = affine_contact_interval(r,c,dt,initial,t_left=(base[0],base[0]+dt*slope[0]),
                      t_right=(base[-1],base[-1]+dt*slope[-1]),nodal_power_start=c*slope,
                      allow_massless=True,return_interval_temperature=True)
        avg_mode = -math.expm1(-rate*dt)/(rate*dt)
        observed_mean_mode = mean-(base+.5*dt*slope)
        extra = dict(mean_cell_error_K=float(abs(observed_mean_mode[c>0]-avg_mode*means).max()),
                     mean_trace_error_K=float(abs(observed_mean_mode[interface:interface+2]-avg_mode*traces).max()))
        balance = (c*(end-initial-dt*slope))[1:-1]+np.diff(heat)
        end = end-(base+dt*slope)
        heat = heat-dt*(.2-.35*dt)
    else:
        end,heat = constant_contact_interval(r,c,dt,initial,t_left=0.,t_right=0.,allow_massless=True)
        balance = (c*(end-initial))[1:-1]+np.diff(heat)
    observed_traces = end[interface:interface+2]
    return dict(cells=n,resistance_m2_K_W=contact,
                cell_error_K=float(abs(end[c>0]-decay*means).max()),
                trace_error_K=float(abs(observed_traces-decay*traces).max()),
                interface_heat_error_J=abs(float(heat[interface])-interface_power*(1-decay)/rate),
                node_balance_J=float(abs(balance).max()),**extra)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ramp",action="store_true",help="superpose moving reservoirs and verify time-mean contact traces")
    args = parser.parse_args()
    rows = []
    previous = {}
    for n in (8,16,32,64,128):
        row = measure(n,args.ramp)
        errors = {key:val for key,val in row.items() if "_error_" in key}
        orders = {key:math.log2(previous[key]/val) for key,val in errors.items()} if previous else {}
        row.update(orders=orders,passed=row["node_balance_J"]<1e-11 and
                   (not orders or min(orders.values())>1.8))
        rows.append(row)
        previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,ramp=args.ramp,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
