import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
here=os.path.dirname(os.path.abspath(__file__))
d=json.load(open(os.path.join(here,"audit_biaxial_estimator.json")))
p1=d["probes"]["P1_operator_recovery"]; p3=d["probes"]["P3_real_field"]
fig,ax=plt.subplots(1,3,figsize=(15,4.4))
# (a) recovery on the audited 24x48 window
r=[q for q in p1 if q["window"]=="24x48"]
t=[q["theta_true"] for q in r]
ax[0].plot(t,[q["library_deg"] for q in r],"o-",label="m15279 operator")
ax[0].plot(t,[q["isotropic_deg"] for q in r],"s-",label="isotropic fix (audit)")
ax[0].plot(t,t,"k--",lw=1,label="y = theta_true")
ax[0].set_xlabel("true band inclination [deg]"); ax[0].set_ylabel("recovered [deg]")
ax[0].set_title("(a) operator recovery, 24x48 biaxial window\nm15279 bias -15 deg")
ax[0].legend(); ax[0].grid(alpha=.3)
# (b) root cause: arctan(tan theta / aspect)
asp=np.linspace(0.4,2.2,200); th=np.array([45,50,55,60,65,70],float)
for q in th:
    ax[1].plot(asp,np.degrees(np.arctan(np.tan(np.radians(q))/asp)),lw=1,
               label=f"theta={q:.0f} deg" if q in (45,60,70) else None)
ax[1].axvline(1.404,color="r",ls=":",label="real window A=1.404")
ax[1].axvline(1.732,color="g",ls=":",label="24x48 lattice A=1.73")
ax[1].set_xlabel("window aspect ratio A = h/w"); ax[1].set_ylabel("reported angle [deg]")
ax[1].set_title("(b) root cause: arctan(tan theta / A)\nmax residual vs library output 0.5 deg")
ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
# (c) filed vs re-measured
ax[2].bar(["m15279 filed\nincremental mean","m15279 filed\nr24x48_p200","library radon\non stored field","isotropic fix\non stored field"],
          [44.25,44.67,p3["library_radon_deg"],p3["isotropic_radon_deg"]],
          color=["#b03a2e","#b03a2e","#7f8c8d","#1e8449"])
ax[2].axhspan(51,69,color="tab:blue",alpha=.15,label="accept window 51-69 deg")
ax[2].axhline(60,color="tab:blue",ls="--",label="anchor 60 deg")
for i,v in enumerate([44.25,44.67,p3["library_radon_deg"],p3["isotropic_radon_deg"]]):
    ax[2].text(i,v+1,f"{v:.1f}",ha="center")
ax[2].set_ylim(0,80); ax[2].set_ylabel("band inclination [deg]")
ax[2].set_title("(c) same stored DEM field, two estimators"); ax[2].legend(fontsize=7)
plt.tight_layout(); out=os.path.join(here,"m15315_estimator_bias.png"); plt.savefig(out,dpi=140)
print("saved",out, os.path.getsize(out))
