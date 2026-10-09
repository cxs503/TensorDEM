"""Smooth moving-plane opt-in for the fixed control-volume network.

The legacy impulsive-onset module is unchanged. Contact force uses the same
50 E/L pressure penalty and reference boundary tributary areas. Tool work
is integrated with endpoint force quadrature and exposed energy error.
"""
import math
import torch
from .continuum_boundary_dem import BoundaryNetwork


class MovingPlaneNetwork(BoundaryNetwork):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.tool_work=0.;self.tool_impulse=0.

    def moving_step(self,dt,plane_start,plane_end):
        if not math.isfinite(dt) or dt<=0 or dt>self.dt*(1+1e-12):raise ValueError('invalid timestep')
        if any(not math.isfinite(z) for z in (plane_start,plane_end)):raise ValueError('invalid plane')
        f,ext,sup,_=self.forces(plane=plane_start)
        a=torch.where(self.fixed,0.,f/self.mass[:,None]);vhalf=self.v+.5*dt*a
        du=dt*vhalf;self.u+=du
        f2,ext2,sup2,_=self.forces(plane=plane_end)
        self.v=vhalf+.5*dt*torch.where(self.fixed,0.,f2/self.mass[:,None])
        self.impulse+=.5*dt*(ext+sup+ext2+sup2).sum(0)
        self.work+=float((.5*(ext+ext2)*du).sum())
        # ext is force on the network. Opposite reaction is on the tool;
        # actuator work into the network/contact system is ext*d(plane).
        mean_force=float(.5*(ext+ext2).sum())
        self.tool_work+=mean_force*(plane_end-plane_start)
        self.tool_impulse-=mean_force*dt
        self.time+=dt
        return -mean_force

    def snapshot(self):
        return dict(schema='tensordem.smooth-plane/1',base=super().snapshot(),tool_work_J=self.tool_work,tool_impulse_Ns=self.tool_impulse)

    @classmethod
    def restore(cls,state):
        if set(state)!={'schema','base','tool_work_J','tool_impulse_Ns'} or state['schema']!='tensordem.smooth-plane/1':raise ValueError('restart schema')
        base=BoundaryNetwork.restore(state['base']);r=cls(**base.config)
        r.__dict__.update(base.__dict__)
        for key,target in [('tool_work_J','tool_work'),('tool_impulse_Ns','tool_impulse')]:
            if not math.isfinite(state[key]):raise ValueError('restart tool ledger')
            setattr(r,target,state[key])
        return r


def smooth_plane(time,ramp=.5,depth=.001):
    """C2 quintic approach, starting in zero-force contact, then hold."""
    s=min(1.,max(0.,time/ramp));q=s**3*(10+s*(-15+6*s))
    return 1.-depth*q
