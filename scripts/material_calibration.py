"""Generate or independently reconstruct the material patch-test evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from tensordem.material_calibration import qualify, relaxed_tension, corrected_patch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs/material-calibration/study.json'
parser = argparse.ArgumentParser()
parser.add_argument('--audit', action='store_true')
args = parser.parse_args()
source = ROOT / 'src/tensordem/material_calibration.py'
record = dict(schema='tensor-dem.material-patch/1', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              cases=[qualify(n) for n in (8, 16, 32)], relaxed_tension_cases=[relaxed_tension(n) for n in (4, 8, 16)], corrected_cases=[corrected_patch(n) for n in (4, 8, 16)], physical_fracture_accuracy_qualified=False,
              interpretation='Same central-bond topology as IceDEM; material proxy only. Calibrating C11 alone cannot qualify isotropic shear/cross stiffness. Disk mass and contact/damage are excluded.')
if args.audit:
    saved = json.loads(OUTPUT.read_text())
    if saved != record:
        raise SystemExit('FAIL source/raw geometry/mass/stiffness/material metrics reconstruction')
    print('PASS exact reconstruction of all three grids; material_qualified remains false')
else:
    OUTPUT.write_text(json.dumps(record, indent=2) + '\n')
    for case in record['cases']:
        print(case['cells_y'], case['stiffness_N_per_m'], case['relative_errors'], case['material_qualified'])
