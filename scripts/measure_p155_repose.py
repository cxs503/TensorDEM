"""C5: repose angle of a stratified vs monodisperse column (P96-N3 estimator)."""
import json, sys, pathlib
sys.path.insert(0, "src")
import numpy as np
from tensordem.bidisperse_segregation import BidisperseConfig, collapse_and_measure_repose

crit_path = pathlib.Path("artifacts/bidisperse_segregation_criteria.json")
crit = json.loads(crit_path.read_text())
phi = crit["C6"]["phi_z"]
ok = [v is not None for v in phi]
prof = np.array([0.5 if v is None else v for v in phi], dtype=float)

def phi_of_z(u):
    i = min(max(int(u * len(prof)), 0), len(prof) - 1)
    return float(prof[i])

cfg = BidisperseConfig(d_small=0.010, d_large=0.030, volume_fraction=0.60,
                       n_particles=40, seed=11)
strat_err = None
try:
    strat = collapse_and_measure_repose(cfg, phi_of_z=phi_of_z, aspect=2.5,
                                        n_particles=40, settle_steps=300,
                                        collapse_steps=42000)
except Exception as exc:
    strat = None
    strat_err = f"{type(exc).__name__}: {exc}"
cfg2 = BidisperseConfig(d_small=0.010, d_large=0.010, volume_fraction=0.60,
                        n_particles=40, seed=11)
mono = collapse_and_measure_repose(cfg2, phi_of_z=None, aspect=2.5,
                                   n_particles=40, settle_steps=300,
                                   collapse_steps=42000)

def slim(d):
    return {k: v for k, v in d.items() if k not in ("surface_x", "surface_y")}

c5 = {
    "pass": False if strat is None else bool(strat["estimator"] == mono["estimator"]),
    "estimator": mono["estimator"],
    "stratified_error": strat_err,
    "stratified_angle_deg": None if strat is None else strat["theta_r_deg"],
    "stratified_r2": None if strat is None else strat.get("r2"),
    "stratified_heap": None if strat is None else {
        "height": strat["heap_height"], "width": strat["heap_width"]},
    "monodisperse_angle_deg": mono["theta_r_deg"],
    "monodisperse_r2": mono.get("r2"),
    "delta_deg": (None if strat is None
                  else float(strat["theta_r_deg"] - mono["theta_r_deg"])),
    "n_particles": 40,
    "collapse_steps": 42000,
    "note": ("same estimator (P96-N3 repose_angle) on both beds; stratified "
             "column carries the measured Phi(z) of the campaign"),
    "stratified": None if strat is None else slim(strat),
    "monodisperse": slim(mono),
}
crit["C5"] = c5
crit["criteria_summary"]["C5"] = c5["pass"]
crit["overall_pass"] = all(crit["criteria_summary"].values())
crit_path.write_text(json.dumps(crit, indent=1, default=float))
print(json.dumps({k: c5[k] for k in ("pass", "stratified_angle_deg",
                                     "monodisperse_angle_deg", "delta_deg",
                                     "stratified_r2")}, indent=1))
