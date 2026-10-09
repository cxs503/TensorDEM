"""Linear central-spring control-volume network, isolated from disk DEM.

Square axial/diagonal network represents plane stress nu=1/3. Boundary
springs parallel to outer faces have half weight. Contact acts on fixed
reference boundary nodes with tributary areas, not radius-shifted disk centers.
"""
import math
from numbers import Real
import torch


class BoundaryNetwork:
    def __init__(self, nx, ny, *, length=1., height=.25, thickness=.1, E=1e5, density=900.):
        if type(nx) is not int or type(ny) is not int or nx < 2 or ny < 2:
            raise ValueError('grid dimensions >=2 required')
        vals=(length,height,thickness,E,density)
        if any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) or v<=0 for v in vals): raise ValueError('positive finite properties required')
        self.config=dict(nx=nx,ny=ny,length=length,height=height,thickness=thickness,E=E,density=density)
        dx=length/nx; dy=height/ny
        if abs(dx-dy)>1e-12: raise ValueError('square network required')
        self.x=torch.tensor([[i*dx,j*dy] for j in range(ny+1) for i in range(nx+1)],dtype=torch.float64)
        self.u=torch.zeros_like(self.x);self.v=torch.zeros_like(self.x)
        w=torch.ones(len(self.x),dtype=torch.float64)
        w[(self.x[:,0]==0)|(self.x[:,0]==length)]*=.5
        w[(self.x[:,1]==0)|(self.x[:,1]==height)]*=.5
        self.mass=w*density*thickness*dx*dy
        edges=[];ks=[]
        for j in range(ny+1):
            for i in range(nx+1):
                a=j*(nx+1)+i
                if i<nx: edges.append((a,a+1));ks.append(3*E*thickness/4*(.5 if j in (0,ny) else 1))
                if j<ny: edges.append((a,a+nx+1));ks.append(3*E*thickness/4*(.5 if i in (0,nx) else 1))
                if i<nx and j<ny:
                    edges.extend(((a,a+nx+2),(a+1,a+nx+1)));ks.extend((3*E*thickness/8,)*2)
        self.edges=torch.tensor(edges);self.k=torch.tensor(ks,dtype=torch.float64)
        d=self.x[self.edges[:,1]]-self.x[self.edges[:,0]];self.n=d/torch.linalg.vector_norm(d,dim=1)[:,None]
        self.fixed=torch.zeros_like(self.x,dtype=torch.bool);self.fixed[:,1]=True;self.fixed[self.x[:,0]==0,0]=True
        self.right=torch.where(self.x[:,0]==length)[0]
        self.areas=torch.full((ny+1,),dy*thickness,dtype=torch.float64);self.areas[0]*=.5;self.areas[-1]*=.5
        self.penalty=50*E/length
        rows=torch.zeros(len(self.x),dtype=torch.float64);rows.index_add_(0,self.edges[:,0],self.k);rows.index_add_(0,self.edges[:,1],self.k)
        rows[self.right]+=self.penalty*self.areas
        self.dt=.15*float(torch.sqrt(self.mass/rows).min())
        self.time=0.;self.impulse=torch.zeros(2,dtype=torch.float64);self.work=0.

    def forces(self,traction=0.,plane=None):
        e=((self.u[self.edges[:,1]]-self.u[self.edges[:,0]])*self.n).sum(1)
        f=torch.zeros_like(self.x);z=(self.k*e)[:,None]*self.n
        f.index_add_(0,self.edges[:,0],z);f.index_add_(0,self.edges[:,1],-z)
        external=torch.zeros_like(f);external[self.right,0]=traction*self.areas
        gap=torch.zeros(len(self.right),dtype=torch.float64)
        if plane is not None:
            gap=(self.x[self.right,0]+self.u[self.right,0]-plane).clamp_min(0)
            external[self.right,0]-=self.penalty*self.areas*gap
        f+=external
        support=torch.where(self.fixed,-f,0.)
        return f,external,support,float(.5*(self.penalty*self.areas*gap.square()).sum())

    def energy(self):
        e=((self.u[self.edges[:,1]]-self.u[self.edges[:,0]])*self.n).sum(1)
        return float(.5*(self.mass[:,None]*self.v.square()).sum()+.5*(self.k*e.square()).sum())

    def step(self,dt,traction=0.,plane=None):
        if isinstance(dt,bool) or not isinstance(dt,Real) or not math.isfinite(dt) or dt<=0 or dt>self.dt*(1+1e-12):raise ValueError('invalid timestep')
        f,ext,sup,_=self.forces(traction,plane)
        a=torch.where(self.fixed,0.,f/self.mass[:,None]);vhalf=self.v+.5*dt*a
        du=dt*vhalf;self.u+=du
        f2,ext2,sup2,_=self.forces(traction,plane)
        self.v=vhalf+.5*dt*torch.where(self.fixed,0.,f2/self.mass[:,None])
        self.impulse+=.5*dt*(ext+sup+ext2+sup2).sum(0)
        # Trapezoidal force displacement gives exact quadratic work for fixed plane.
        self.work+=float((.5*(ext+ext2)*du).sum());self.time+=dt
        return float(-.5*(ext[self.right,0]+ext2[self.right,0]).sum())

    def snapshot(self):
        return dict(schema='tensordem.control-volume-boundary/1',config=self.config,u=self.u.tolist(),v=self.v.tolist(),time=self.time,impulse=self.impulse.tolist(),work=self.work)

    @classmethod
    def restore(cls,s):
        if set(s)!={'schema','config','u','v','time','impulse','work'} or s['schema']!='tensordem.control-volume-boundary/1':raise ValueError('restart schema')
        r=cls(**s['config'])
        for key in ('u','v','impulse'):
            z=torch.tensor(s[key],dtype=torch.float64)
            if z.shape!=getattr(r,key).shape or not torch.isfinite(z).all():raise ValueError('restart fields')
            setattr(r,key,z)
        if float(r.u[r.fixed].abs().max())!=0 or float(r.v[r.fixed].abs().max())!=0:raise ValueError('restart fixed constraint violation')
        if not math.isfinite(s['time']) or s['time']<0 or not math.isfinite(s['work']):raise ValueError('restart clock/work')
        r.time=s['time'];r.work=s['work'];return r
