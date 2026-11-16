#!/usr/bin/env python3
"""Post the Round 28 comprehensive validation report."""
import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
if not BASE.endswith("/api/v1"):
    BASE += "/api/v1"

# Read the markdown report
md_path = ROOT / "docs" / "CASE_DEM_PLANE_SHEAR_MUIDI_R28.md"
markdown = md_path.read_text()

# Read figures
fig_dir = ROOT / "docs" / "figures"
figures = []
for fig_name in ['r28_mu_i_comparison.png', 'r28_phi_i_comparison.png', 'r28_estimator_spread.png', 'r28_momentum_balance.png']:
    fig_path = fig_dir / fig_name
    if fig_path.exists():
        with open(fig_path, 'rb') as f:
            png_data = f.read()
        figures.append({
            'name': fig_name,
            'caption': fig_name.replace('.png', '').replace('r28_', '').replace('_', ' ').title(),
            'png_base64': base64.b64encode(png_data).decode('utf-8')
        })

report = {
    'title': 'CASE-DEM-PLANE-SHEAR-MUIDI Round 28: Comprehensive Validation Report',
    'summary': 'Comprehensive validation of plane shear mu(I) rheology benchmark after 28 rounds. Status: WITHHELD. 3D servo shear gives mu 18-45% below da Cruz/GDR MiDi reference. Full-cell virial captures 93% of applied stress. Issues: reference parameters unverified, stationarity problems, stress estimator ambiguity. Needed: longer strains (5-10), larger system (1000+ particles), parameter calibration, archive original reference data.',
    'scenario_key': 'CASE-DEM-PLANE-SHEAR-MUIDI',
    'mission_id': 22696,
    'tags': ['DEM', 'plane-shear', 'mu-I', 'rheology', 'validation', 'round-28'],
    'markdown': markdown,
    'figures': figures
}

# Login and post
user = os.environ.get("TENSOR_AGENT_USER")
passwd = os.environ.get("TENSOR_AGENT_PASS")
creds = json.dumps({"username": user, "password": passwd}).encode()

print(f"Logging in to {BASE}...")
req = urllib.request.Request(BASE + "/auth/login", data=creds,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=180) as resp:
    tok = json.loads(resp.read())
token = tok.get("access_token") or tok.get("token")
print(f"Token obtained: {len(token)} chars")

# Post report
payload = json.dumps(report).encode()
req = urllib.request.Request(
    BASE + "/tensor/reports", data=payload,
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    method="POST"
)
with urllib.request.urlopen(req, timeout=300) as resp:
    result = json.loads(resp.read())
    print(f"Report posted: {result}")
