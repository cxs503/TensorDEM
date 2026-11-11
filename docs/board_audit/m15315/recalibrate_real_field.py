import math, sys, json
import numpy as np
sys.path.insert(0,'/root/tp-agent/work/m15279/src')
from tensordem.biaxial_shear_band import radon_band_angle
z=np.load('/root/tp-agent/work/m15279/artifacts/biaxial_shear_band/final_field.npz')
x,y,gamma=z['x'],z['y'],z['gamma']; stored=float(z['angle_deg'])
print(f"particles={x.size} window x[{x.min():.1f},{x.max():.1f}] y[{y.min():.1f},{y.max():.1f}] aspect={(y.max()-y.min())/(x.max()-x.min()):.2f}")
print(f"stored angle_deg = {stored:.2f}  (cumulative-field estimate of m15279)")
lib=radon_band_angle(x,y,gamma,resolution_deg=1.0)
print(f"library radon on real gamma field -> {lib['folded_angle_deg']:.1f} deg (eta2={lib['eta2']:.2f}, z={lib['z_score']:.2f})")

def radon_fixed_bin(x,y,w,bin_width,resolution_deg=0.5):
    x=np.asarray(x,float).ravel(); y=np.asarray(y,float).ravel(); w=np.clip(np.asarray(w,float).ravel(),0,None); w=w/w.sum()
    scale=max(np.ptp(x),np.ptp(y)); x=(x-x.min())/scale; y=(y-y.min())/scale
    nbins=max(8,int(math.ceil(max(np.ptp(x),np.ptp(y))/(bin_width/scale))))
    mu=float(w.mean()); total=float(np.sum((w-mu)**2)); angles=np.arange(0.,180.,resolution_deg); eta=np.empty_like(angles)
    for i,a in enumerate(angles):
        r=math.radians(a); s=x*(-math.sin(r))+y*math.cos(r); s=s-s.min()
        idx=np.minimum((s/(bin_width/scale)).astype(np.int64),nbins-1)
        cnt=np.bincount(idx,minlength=nbins).astype(float); ws=np.bincount(idx,weights=w,minlength=nbins)
        occ=cnt>0; bm=np.zeros(nbins); bm[occ]=ws[occ]/cnt[occ]
        eta[i]=float(np.sum(cnt[occ]*(bm[occ]-mu)**2))/total
    b=int(np.argmax(eta)); return float(angles[b]),float(eta[b])

# calibrated bin width: one local-neighbourhood diameter ~ 2 particle spacings; test 1.5/2/3
print('--- calibration on the REAL window geometry (synthetic ridge of known theta) ---')
cal=[]
for theta in (50.,55.,60.,65.,70.):
    th=math.radians(theta); s=-x*math.sin(th)+y*math.cos(th); w=np.exp(-(s/2.0)**2)
    l=radon_band_angle(x,y,w,resolution_deg=1.0)['folded_angle_deg']
    row={'theta_true':theta,'library':round(l,2),'library_bias':round(l-theta,2)}
    for bw in (1.5,2.0,3.0):
        f,_=radon_fixed_bin(x,y,w,bw); row[f'fixed_bw{bw}']=round(f,2); row[f'fixed_bw{bw}_bias']=round(f-theta,2)
    cal.append(row); print(json.dumps(row))
best=2.0
corr,_=radon_fixed_bin(x,y,gamma,best)
print(f"CORRECTED real-field angle (fixed bin {best} spacings) = {corr:.1f} deg")
json.dump({'stored_angle':stored,'library_on_real_field':lib['folded_angle_deg'],
           'corrected_real_field':corr,'calibration':cal},
          open('/root/tp-agent/work/m15315/docs/board_audit/m15315/real_field_recalibration.json','w'),indent=1)
