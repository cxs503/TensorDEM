import math, sys, json
import numpy as np, torch
sys.path.insert(0,'/root/tp-agent/work/m15279/src')
from tensordem.biaxial_shear_band import radon_band_angle, local_shear_strain
torch.set_num_threads(1)

def lattice(columns, rows, spacing=1.0, jitter=0.03, seed=11):
    rng=np.random.default_rng(seed); pts=[]
    for r in range(rows):
        for c in range(columns):
            pts.append((c*spacing+0.5*(r%2)*spacing, r*spacing*math.sqrt(3)/2))
    pts=np.asarray(pts,float); pts+=jitter*spacing*rng.standard_normal(pts.shape); return pts

def radon_fixed_bin(x, y, w, bin_width, resolution_deg=0.5):
    """Same eta2 criterion, but with a FIXED physical bin width and a COMMON
    (domain-wide) normalisation, so eta2 is comparable across alpha."""
    x=np.asarray(x,float).ravel(); y=np.asarray(y,float).ravel(); w=np.clip(np.asarray(w,float).ravel(),0,None)
    w=w/w.sum()
    # common metric: isotropic normalisation by the domain scale, not per-alpha ptp
    scale=max(np.ptp(x), np.ptp(y))
    x=(x-x.min())/scale; y=(y-y.min())/scale
    nbins=max(8, int(math.ceil(max(np.ptp(x),np.ptp(y))/ (bin_width/scale))))
    mu=float(w.mean()); total=float(np.sum((w-mu)**2))
    angles=np.arange(0.,180.,resolution_deg); eta=np.empty_like(angles)
    for i,a in enumerate(angles):
        r=math.radians(a); s=x*(-math.sin(r))+y*math.cos(r)
        s=s-s.min()
        idx=np.minimum((s/ (bin_width/scale)).astype(np.int64), nbins-1)
        cnt=np.bincount(idx,minlength=nbins).astype(float); ws=np.bincount(idx,weights=w,minlength=nbins)
        occ=cnt>0; bm=np.zeros(nbins); bm[occ]=ws[occ]/cnt[occ]
        eta[i]=float(np.sum(cnt[occ]*(bm[occ]-mu)**2))/total
    b=int(np.argmax(eta)); return float(angles[b]), float(eta[b])

print('--- aspect sweep: library radon vs fixed-bin radon (weight ridge, known theta) ---')
for (c,r) in [(24,48),(48,48),(24,24),(48,24),(16,32)]:
    ref=lattice(c,r)
    line=[]
    for theta in (45.,60.,70.):
        th=math.radians(theta)
        s=-ref[:,0]*math.sin(th)+ref[:,1]*math.cos(th)
        w=np.exp(-(s/2.0)**2)
        lib=radon_band_angle(ref[:,0],ref[:,1],w,resolution_deg=0.5)['folded_angle_deg']
        fx,feta=radon_fixed_bin(ref[:,0],ref[:,1],w,bin_width=2.0)
        line.append(f"th={theta:.0f}: lib {lib:6.2f} ({lib-theta:+5.2f}) | fix {fx:6.2f} ({fx-theta:+5.2f}) eta2={feta:.2f}")
    print(f"lattice {c}x{r} (aspect {r*math.sqrt(3)/2/c:.2f}): " + " || ".join(line))
