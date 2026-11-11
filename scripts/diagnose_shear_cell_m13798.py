"""Regenerate artifacts/shear_cell_friction/shear_cell_diagnosis.json (m13798)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataclasses import replace
from tensordem.shear_cell import ShearCellCase, simulate_shear_cell, _fit_mohr_coulomb

c = ShearCellCase(settle_time=0.02, consol_time=0.03, shear_strain_target=0.02, sample_every=20)
runs = {}
for s in (400.0, 1600.0):
    out = simulate_shear_cell(replace(c, normal_stress=s))
    runs[str(int(s))] = {k: v for k, v in out.items() if not isinstance(v, list)}
    runs[str(int(s))]["trace"] = {k: out[k] for k in
        ("shear_strain_history", "shear_stress_history", "normal_stress_history",
         "bed_top_history", "platen_z_history")}
payload = {"mission": 13798, "anchor": "GAP-DEM-FRICTION", "runs": runs,
           "fitter_check": _fit_mohr_coulomb([400., 800., 1600.], [200., 400., 800.]),
           "verdict": "honest FAIL: platen loses contact, sigma_n -> 0, mu undefined"}
out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "shear_cell_friction")
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "shear_cell_diagnosis.json")
with open(path, "w") as fh:
    json.dump(payload, fh, indent=2)
print("wrote", os.path.normpath(path))
