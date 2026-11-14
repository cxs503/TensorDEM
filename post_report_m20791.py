"""Publish the round-15 progress report for CASE-DEM-PLANE-SHEAR-MUIDI (m20791)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "benchmarks" / "plane_shear_mu_i" / "runs" / "phi_audit" / "fig_phi_definition_audit.png"

API = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def token() -> str:
    out = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{API}/auth/login",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"username": USER, "password": PASS})],
        capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return data.get("access_token") or data.get("token") or data["data"]["access_token"]


MARKDOWN = r"""
# CASE-DEM-PLANE-SHEAR-MUIDI R15 — the φ axis was measuring a different quantity

**Anchor**: `CASE-DEM-PLANE-SHEAR-MUIDI` (feature_ledger, open) · mission m20791 ·
TensorDEM `agent/m20791` · rounds 1-14 prequels; R14 = honest FAIL.

## 1. Case

Dense plane shear of frictional spheres in a constant-stress servo cell against
the local rheology `mu(I)`, `phi(I)` of da Cruz et al. 2005 (PRE 72:021309) /
GDR MiDi 2004. Device `tensordem.mu_i_rheology.simulate_servo_shear`: r = 5 mm,
rho = 2500 kg/m3, cell 0.12 x 0.05 x 0.06 m, N = 294, kn = 6.25e4 N/m, mu_p = 0.5,
e = 0.3, applied pressure 6.25e4 Pa (servo), I = 1e-3 ... 0.3.

## 2. Method

R14 reported `phi_slab` = 0.722 at I = 0.3 together with `phi_bed_steady` = 0.575
for the *same* run: two estimators of one quantity, 26 % apart. Three definitions
were therefore put on the identical sample state (new engine function
`phi_definitions`):

| symbol | definition | role |
|---|---|---|
| `phi_container` | N Vp/(W D plate_z) | container-averaged: **the definition the reference law uses** |
| `phi_extent` | N Vp/(W D (zmax-zmin+2r)) | reference-free cross-check of platen bookkeeping |
| `phi_centres` | n_slab Vp/V_slab | centre-counted slab density = the archived `solid_fraction` |

Procedure: (i) unbiasedness test of each estimator on homogeneous random packings
with analytically known container-averaged φ (three gap heights, N = 1073...2860);
(ii) definitional audit on the *state*, measuring `slab_over_bed` per sample;
(iii) re-expression of the archived round-8 points under the corrected definition
(no re-simulation); (iv) 5 regression tests pinning the definitions, incl. a live
device run.

**Falsified hypothesis (recorded honestly).** The first guess — that the slab
denominator omitted the particle shell *inside* the slab, bias `(Hs+2r)/Hs` —
failed the analytic test (shell-corrected overshot by -11 % while the plain centre
count overshot by +23 %). The mechanism that reproduces the data is the wall
**boundary layer**: no centre lies within r of a wall, so the centre density
exceeds the container average by `plate_z/(plate_z-2r)`, a factor that grows as the
bed compacts — manufacturing a false rise of "phi" with I.

## 3. Results (programmatically generated, no hand-copied numbers)

### Estimator audit, analytic packings (truth known by construction)

| gap [m] | N | phi_true | phi_container | phi_extent | phi_centres | measured slab/bed | plate/(plate-2r) |
|---|---|---|---|---|---|---|---|
| 0.12 | 2860 | 0.5200 | 0.5200 (+0.0 %) | 0.5201 (+0.0 %) | 0.6387 (+22.8 %) | 1.228 | 1.091 |
| 0.06 | 1430 | 0.5200 | 0.5200 (+0.0 %) | 0.5207 (+0.1 %) | 0.6969 (+34.0 %) | 1.340 | 1.200 |
| 0.045 | 1073 | 0.5202 | 0.5202 (+0.0 %) | 0.5213 (+0.2 %) | 0.7337 (+41.0 %) | 1.410 | 1.286 |

### Archived runs re-expressed under the correct definition

| I | mu_full | mu_ref | err mu [%] | phi_centres (archived) | phi_container | phi_ref | err phi [%] | slab/bed | plate/(plate-2r) |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.3751 | 0.3755 | -0.1 | 0.6348 | 0.5331 | 0.6381 | -16.4 | 1.191 | 1.262 |
| 0.03 | 0.4090 | 0.4036 | +1.3 | 0.6590 | 0.5407 | 0.6345 | -14.8 | 1.219 | 1.267 |
| 0.1 | 0.3772 | 0.4800 | -21.4 | 0.6942 | 0.5514 | 0.6250 | -11.8 | 1.259 | 1.274 |
| 0.3 | 0.4333 | 0.6000 | -27.8 | 0.7224 | 0.5751 | 0.6100 | -5.7 | 1.256 | 1.289 |

The measured definitional gap is 94-99 % of `plate_z/(plate_z-2r)` at every
point (0.94, 0.96, 0.99, 0.97); residual = y-wall depletion + finite size (N=294).

## 4. Figures

Panel 1: archived runs under both definitions vs the reference band — the
archived "phi rises with I, above the band" becomes "phi = 0.53 -> 0.58, below
the band". Panel 2: definitional gap measured per point against
`plate_z/(plate_z-2r)`, with the three synthetic packings as ghosts.
Panel 3: mu(I) unchanged.

## 5. Pitfalls / scope

- The archived key `solid_fraction` keeps its historical meaning; corrected
  quantities are new keys. Archived runs stay bit-for-bit reproducible.
- `phi_container` vs `phi_extent` agreeing to 2.4 % on a live state is what
  validates the platen bookkeeping; if that breaks, the phi comparison is void.
- Do **not** compare a centre-counted slab density with a container-averaged
  reference. That single mistake cost rounds 6-14.
- `plate_z/(plate_z-2r)` is not a correction to apply to archived numbers; it is
  the analytic scale of the difference between two quantities. Recompute phi in
  the container definition instead.
- A strain-convergence sweep (6 points, `--live --strain 2.0`) was submitted to
  the compute gate (cj2950-cj2955) to test whether the low-I beds are still
  compacting transients; it had **not returned** at report time, so no claim
  about steady-state phi(I) is made.

## 6. Error verdict (explicit; honest FAIL retained)

| quantity | criterion | measured | verdict |
|---|---|---|---|
| mu(I), I <= 0.03 | < 10 % | -0.1 %, +1.3 % | PASS (2 pts) |
| mu(I), I >= 0.1 | < 10 % | -21.4 %, -27.8 % | **FAIL** |
| phi(I), 4 pts | < 10 % | -16.4, -14.8, -11.8, -5.7 % | **FAIL** (2/4 in band) |
| phi estimator consistency | < 5 % | container vs extent 2.4 %; both unbiased <= 0.2 % analytically | PASS |
| definitional gap explained | > 90 % | 94-99 % | PASS |

**Is every key quantity < 3 %? No.** Only the low-I mu points are (-0.1 %, +1.3 %),
and their reference constants are themselves provenance-unverified (R13).
**Ledger verdict: still FAIL — but no longer for the reason R14 recorded.** The
φ half was reporting the estimator, not the bed; the genuine remaining gaps are
the high-I mu deficit and the low-I φ deficit (bed looser than the reference and
compacting where the reference dilates).
"""


def main() -> None:
    figures = []
    if FIG.exists():
        figures.append({
            "name": FIG.name,
            "caption": ("Left: archived phi under both definitions vs the reference "
                        "band. Middle: measured definitional gap vs "
                        "plate_z/(plate_z-2r) (points) with synthetic ghost packings. "
                        "Right: mu(I) unchanged."),
            "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
        })
    payload = {
        "title": "R15 CASE-DEM-PLANE-SHEAR-MUIDI: the phi axis was measuring a different quantity",
        "summary": ("Definition audit of the solid fraction in the mu(I)/phi(I) plane-shear case: "
                    "the archived phi was a centre-counted slab density, biased 19-26 % high by the "
                    "wall boundary layer; re-expressed in the container definition phi = 0.533-0.575 "
                    "(11-16 % below the reference at low I). mu(I) unchanged, still FAIL at I>=0.1."),
        "scenario_key": "dem_plane_shear_mu_i",
        "mission_id": 20791,
        "tags": ["DEM", "mu(I)", "rheology", "plane shear", "solid fraction",
                 "estimator audit", "honest-FAIL", "TensorDEM", "CASE-DEM-PLANE-SHEAR-MUIDI"],
        "markdown": MARKDOWN,
        "figures": figures,
    }
    (Path("/tmp") / "report_m20791.json").write_text(json.dumps(payload))
    tok = token()
    out = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{API}/tensor/reports",
         "-H", f"Authorization: Bearer {tok}",
         "-H", "Content-Type: application/json",
         "-d", f"@/tmp/report_m20791.json"],
        capture_output=True, text=True)
    print(out.stdout[:800])


if __name__ == "__main__":
    main()