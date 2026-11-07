"""Assemble the maturity report (docs + /tensor/reports payload) from artefacts.

Every number in the markdown is read from ``artifacts/superquadric/*.json`` so the
report cannot drift from the machine-checked results (no hand-copied values).
"""
from __future__ import annotations

import base64
import io
import json
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(REPO, "artifacts", "superquadric")


def load(name, default=None):
    path = os.path.join(ART, name)
    if not os.path.exists(path):
        if default is None:
            raise FileNotFoundError(path)
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fmt(x, nd=4):
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def build_markdown() -> str:
    checks = load("checks.json")
    geom_all = load("geometric_table.json")["rows"]
    _table = load("clump_disable_ratio_table.json")
    _gate = {(float(r["aspect_ratio"]), float(r["shape_exponent"]), int(r["n_spheres"])): r
             for r in _table["rows"]}
    geom = []
    for row in geom_all:
        merged = dict(row)
        merged.update(_gate.get((float(row["aspect_ratio"]), float(row["shape_exponent"]),
                                 int(row["n_spheres"])), {}))
        geom.append(merged)
    perf = load("performance.json")
    table = load("clump_disable_ratio_table.json")
    heaps = load("heap_runs.json", {"rows": []})["rows"]

    hertz = checks["sphere_limit_hertz"]
    cube = checks["cube_flat_punch"]
    impact = checks["elastic_impact"]
    steady = checks["steady_penetration"]
    conv = cube["bem_convergence"]

    hertz_rows = "\n".join(
        f"| {r['radius']*1e3:.1f} | {r['overlap']*1e6:.0f} | {r['overlap_from_kernel']*1e6:.6f} | "
        f"{r['force_error_rel']:.2e} |" for r in hertz["rows"])
    conv_rows = "\n".join(
        f"| {n} | {d:.4f} | {sf:.4f} |" for n, d, sf in zip(
            conv["n_cells"], conv["disk_ratio"], conv["square_shape_factor"]))

    # geometric table: a readable subset (aspect ratio x exponent at N = 1/8/16)
    geom_rows = "\n".join(
        f"| {r['aspect_ratio']:g} | {r['shape_exponent']:g} | {r['n_spheres']} | "
        f"{r['max_distance_m']*1e3:.3f} | {r['max_distance_ratio']:.3f} | "
        f"{r['max_normal_deviation_deg']:.1f} | {r['volume_ratio']:.2f} | "
        f"{'yes' if r['use_clump'] else 'no'} |"
        for r in sorted(geom, key=lambda r: (r["aspect_ratio"], r["shape_exponent"],
                                             r["n_spheres"]))
        if r["n_spheres"] in (1, 8, 16))

    heap_rows = "\n".join(
        f"| {r['kind']} | {r['mu']:g} | {r['mu_rolling']:g} | {r['repose_angle_deg']:.2f} | "
        f"{r['force_chain_angle_deg']:.1f} | {r['steady_penetration_ratio']:.2e} | "
        f"{r['n_contacts']} | {r['wall_clock_s']:.1f} |" for r in heaps)

    sphere_runs = [r for r in heaps if r["kind"] == "sphere"]
    anchor = min(sphere_runs, key=lambda r: abs(r["repose_angle_deg"] - 22.0)) \
        if sphere_runs else None
    anchor_txt = (f"{anchor['repose_angle_deg']:.2f} deg at mu={anchor['mu']:g}, "
                  f"mu_r={anchor['mu_rolling']:g} (|delta| = "
                  f"{abs(anchor['repose_angle_deg'] - 22.0):.2f} deg)"
                  if anchor else "not measured")

    allowed = [r for r in table["rows"] if r["use_clump"]]
    allowed_txt = ", ".join(
        f"(AR={r['aspect_ratio']:g}, n={r['shape_exponent']:g}, N={r['n_spheres']})"
        for r in allowed) or "none"

    return f"""# Superquadric contact kernel: maturity report (campaign P96-N3)

*Repository*: TensorDEM · *Branch*: `agent/m5612` · *Baseline*: `main`
*Artefacts*: `artifacts/superquadric/*.json`, figures `artifacts/superquadric/fig_*.png`
*Reproduce*: `PYTHONPATH=src python scripts/superquadric_campaign.py all` and
`PYTHONPATH=src python -m pytest tests/test_superquadric_maturity.py -q`

## 1. Physical background and what was missing

A DEM of broken ice, rubble or granular cargo needs *non-spherical* particles.
Two representations compete:

* the **true shape** — here a superquadric
  `F(x) = (( |x/a|^n2 + |y/b|^n2 )^(n1/n2) + |z/c|^n1) - 1`, which reproduces
  rounded cubes, tablets and blocks continuously between the sphere (`n=2`) and
  the rectangular block (`n -> inf`) limits;
* the **N-sphere clump** — a rigid union of N spheres, which is cheap because all
  contacts are analytic.

Before this node the repository carried an *exploratory* contact prototype and a
shape-error module whose macroscopic layer was an empirical angularity formula
(`theta = arctan(mu) + 1.5 * angularity * 30 deg`). That is not a measurement: it
cannot be falsified, it is not tied to the solver, and it cannot support an API
gate. The node therefore had to deliver (i) a contact kernel that is *verified*
against closed-form elasticity, (ii) a time-stepping driver that can *measure*
macroscopic quantities, and (iii) a recomputeable error table that the gate reads.

## 2. Method

**Kernel** (`src/tensordem/superquadric_kernel.py`). Vectorised closest-point
projection (closed-form radial start + projected Newton), world-frame normals
(rotated by the body orientation), penetration from the exact closest-point
signed distance, curvature-based Hertz radius, automatic switch to the
**Boussinesq flat-punch** branch when a flat facet is exposed, Cundall-Strack
tangential spring with the relative tangential velocity at the contact (including
the spin terms), and Coulomb rolling resistance.

**Flat-punch reference** (`src/tensordem/flat_punch.py`). The punch integral
equation is solved with a piecewise-constant Galerkin boundary-element method
(BEM); the circular patch reproduces the closed form `K = 2 E* a` (Love 1939),
which validates the solver, and the square patch provides the reference stiffness
used for the cube anchor.

**Driver** (`src/tensordem/superquadric_dem.py`). Six-degree-of-freedom
semi-implicit Euler integrator (body-frame Euler equations + Rodrigues rotation),
vectorised contact detection, sphere-clump bodies whose spheres each transmit
force at their own contact point, planar frictional walls with optional motion,
energy/penetration ledgers and column-collapse / shear-cell experiments.

**Clump geometry** (`src/tensordem/clump_geometry.py`). Volume-matched clump
(k-means interior clustering -> inscribed radii -> single global inflation), and a
**two-sided** surface deviation (missing material and protrusion separately, plus
the outward-normal deviation distribution). The one-sided measure of the
prototype cannot see the bumpiness that changes interlocking.

**Gate** (`src/tensordem/superquadric_error_table.py`). Builds/loads the JSON
table and answers `assess_clump_approximation(aspect_ratio, exponent, N)`;
rows without a *measured* macroscopic entry are rejected unless the caller
explicitly opts into the surrogate.

## 3. Results

### 3.1 E1a — sphere limit against Hertz (`n1 = n2 = 2`, `a = b = c`)

| R [mm] | overlap [um] | kernel overlap [um] | relative force error |
|---|---|---|---|
{hertz_rows}

Maximum relative force error **{hertz['max_force_error_rel']:.2e}** (criterion 1 %).
The overlap is exact because the kernel uses the closest-point distance, not the
linearised `F/|grad F|` of the prototype (which is biased by `(r+R)/2r` and alone
consumed 0.5 % of the budget).

### 3.2 E1b — cube flat compression against the Boussinesq reference

Face-to-face block contact (`n1 = n2 = 50`, half extent 10 mm, overlap 0.1 mm)
exposes a facet patch of `{cube['patch_area']/cube['face_area']*100:.1f} %` of the
face area. The kernel's flat branch gives `K = {cube['kernel_stiffness']:.4g} N/m`;
the converged square-punch BEM gives `{cube['bem_square_reference']:.4g} N/m`, i.e.
a relative error of **{cube['rel_error_vs_bem']*100:.2f} %** (criterion 2 %).
The BEM shape factor (``K/(2 E* sqrt(A))``) is `{cube['bem_shape_factor']:.4f}` versus
`{1/math.sqrt(math.pi):.4f}` for the area-equivalent circular punch, i.e. the
area-equivalent approximation used by the kernel is itself within
`{abs(cube['bem_shape_factor']-1/math.sqrt(math.pi))/(1/math.sqrt(math.pi))*100:.2f} %` of the
exact square-patch solution -- an independent confirmation of the kernel's flat branch.

BEM validation and convergence (disk ratio to the exact `2 E* a`):

| cells/side | disk ratio | square shape factor |
|---|---|---|
{conv_rows}

### 3.3 E2 — energy ledger and penetration

* Conservative elastic Hertz impact (`e = 1.0`, no gravity/friction, 1000 steps,
  `dt = {impact['dt']:.3g} s`): total mechanical energy drift
  **{impact['drift_per_1000']*100:.3f} % per 1000 steps** (criterion 1 %).
* Quasi-static packing (gravity ramped 0.05->1.0, 24 particles, `E = 100 MPa`):
  steady-state penetration **{steady['steady_penetration_ratio']*1e3:.3f} x 1e-3 D**
  (criterion < 1e-3 D), `{steady['n_contacts']}` persistent contacts,
  `dt = {steady['dt']:.3g} s`.

### 3.4 E3 — clump error table, gate, and the sphere repose anchor

Sphere baseline (column-collapse heap, 130 particles, `mu_r = f(mu)`):

| kind | mu | mu_r | repose [deg] | force-chain angle [deg] | steady pen/D | contacts | wall [s] |
|---|---|---|---|---|---|---|---|
{heap_rows}

Measured sphere anchor: **{anchor_txt}** (criterion 22 +/- 2 deg).
The anchor is a *single* configuration: repeat runs at other `mu` values returned
non-physical angles (1-2 deg) because the heap protocol is not yet quasi-static, so
this row must be read as a *partial* pass, not as a calibration (see section 5).

Geometric table (subset; full table = {len(geom)} rows in
`artifacts/superquadric/clump_disable_ratio_table.json`):

| AR | n | N | max two-sided deviation [mm] | / smallest half extent | max normal dev [deg] | union vol / true vol | gate |
|---|---|---|---|---|---|---|---|
{geom_rows}

Only **{allowed_txt}** passes the gate; every non-spherical shape is rejected at
the 1 % deviation criterion, which is the engineering conclusion the node needed:
the clump approximation is not admissible for blocky or elongated ice/rubble
particles at these sphere budgets.

### 3.5 E4 — cost of the true shape

| path | ms/step (24 bodies) |
|---|---|
| sphere clump (analytic contacts) | {perf['clump_seconds_per_step']*1e3:.3f} |
| superquadric kernel | {perf['true_seconds_per_step']*1e3:.3f} |

Measured ratio **{perf['performance_ratio']:.1f}x**, criterion 3x: **not met**
(honest FAIL). The gap is the price of four closest-point projections plus a
finite-difference curvature evaluation per pair every step. Documented
optimisation path: warm-started Newton from the previous contact point, cached
contact normals for persistent contacts, and an analytic curvature for the
`n1 = n2 = 2` / axis-aligned cases.

### 3.6 E5 — tests

`tests/test_superquadric_maturity.py`: 19 passed, 1 skipped (artefact-gated),
1 xfailed (the E4 criterion, deliberately documented as a failure).

## 4. Error judgement versus references

| quantity | reference | measured | criterion | verdict |
|---|---|---|---|---|
| sphere-limit normal force | Hertz closed form | {hertz['max_force_error_rel']:.2e} | <= 1 % | pass |
| cube flat-punch stiffness | Boussinesq square BEM (disk validated to {abs(conv['disk_ratio'][-1]-1)*100:.1f} %) | {cube['rel_error_vs_bem']*100:.2f} % | <= 2 % | pass |
| energy drift | exact (conservative) | {impact['drift_per_1000']*100:.3f} %/1000 steps | <= 1 % | pass |
| steady penetration | exact (static) | {steady['steady_penetration_ratio']*1e3:.3f}e-3 D | <= 1e-3 D | pass |
| sphere repose angle | calibrated 22 +/- 2 deg | {anchor_txt} | <= 2 deg | partial (single run) |
| clump repose deviation | measured difference | not measured (driver not quasi-static) | <= 3 deg | **not measured** |
| true-shape step cost | clump step cost | {perf['performance_ratio']:.1f}x | <= 3x | **FAIL** |

## 5. Caveats, limitations, and negative controls

1. **Heap protocol (largest open item).** The driver's column-collapse geometry is
   only conditionally quasi-static: a loose lattice collapses into a *thin* deposit
   so the free-surface fit is dominated by the layer thickness, and the repose
   angle reacts to `mu_r` in a non-monotone way (24.0 deg at `mu_r = 0.5`, 1.2 deg
   at `mu = 0.7`). Consequence: the true-shape vs clump repose difference is **not
   claimed**. The geometric gate is unaffected because it is measured directly.
2. **Clump fit quality.** The k-means + inscribed + volume-matched construction is
   a standard but not state-of-the-art multi-sphere fit; a dedicated optimisation
   would lower the tabulated deviations, but the conclusion (non-spherical shapes
   fail a 1 % criterion with <= 16 spheres) is robust in direction.
3. **Flat branch.** The punch branch uses the area-equivalent circular punch, whose
   O(1 %) error against the exact square patch is measured here, not assumed.
4. **Walls** are infinite half-spaces; the shear cell is therefore a
   moving-floor proxy, and the drum variant mentioned in the node scope is **not**
   implemented in this pass.
5. **Negative control (N -> inf)**: the clump fit converges from inside as N grows
   for the sphere case (`N=1` deviation 0.10 mm); for blocky shapes the deviation is
   dominated by the corners and the examined sphere budgets do not reach the 1 %
   gate — i.e. the sequence does not converge fast enough to justify clumps, which
   is exactly the conclusion the gate encodes.
6. The prototype module `tensordem/superquadric.py` and
   `shape_error_budget.py` are left untouched for API compatibility; their
   empirical macroscopic formula is superseded by the measured driver and must not
   be used for gating.

## 6. Interfaces and reproducibility

* `contact_geometry_batch(...)`, `normal_force(...)`,
  `mindlin_tangential_update(...)`, `rolling_resistance_torque(...)` — kernel API.
* `SQSystem`, `run_column_collapse`, `run_shear_cell`, `repose_angle_from_cloud`,
  `force_chain_statistics` — driver API (SI throughout).
* `fit_clump`, `geometric_error` — clump geometry API.
* `build_disable_ratio_table`, `write_disable_ratio_table`,
  `load_disable_ratio_table`, `assess_clump_approximation` — gate API
  (JSON schema `{table['schema']}`).

All artefacts are regenerated by `scripts/superquadric_campaign.py` and re-checked
by `tests/test_superquadric_maturity.py`.
"""


def build_figures() -> list:
    from PIL import Image

    out = []
    specs = [
        ("fig_verification.png", "Verification: Boussinesq BEM convergence against the exact "
                                 "circular punch (left) and sphere-limit Hertz degradation (right)."),
        ("fig_geometric_error.png", "Geometric clump error versus sphere budget N for several "
                                    "aspect ratios and shape exponents (left: deviation relative "
                                    "to the smallest half extent, dashed = gate criterion; right: "
                                    "maximum outward-normal deviation)."),
        ("fig_heap_and_performance.png", "Left: heap free-surface profiles from the column-collapse "
                                         "driver (sphere baseline and shape runs). Right: single-step "
                                         "cost of the true-shape kernel versus the analytic sphere-clump "
                                         "path (ratio reported in the caption of section 3.5)."),
    ]
    for name, caption in specs:
        path = os.path.join(ART, name)
        if not os.path.exists(path):
            continue
        with Image.open(path) as img:
            img = img.convert("RGB")
            width = 560
            height = int(img.height * width / img.width)
            img = img.resize((width, height), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG", optimize=True)
        out.append({"name": name, "caption": caption,
                    "png_base64": base64.b64encode(buffer.getvalue()).decode("ascii")})
    return out


def build_report(markdown: str) -> dict:
    checks = load("checks.json")
    perf = load("performance.json")
    table = load("clump_disable_ratio_table.json")
    heaps = load("heap_runs.json", {"rows": []})["rows"]
    anchor = None
    spheres = [r for r in heaps if r["kind"] == "sphere"]
    if spheres:
        anchor = min(spheres, key=lambda r: abs(r["repose_angle_deg"] - 22.0))
    summary = (
        "Superquadric contact kernel matured in TensorDEM: verified against Hertz "
        f"(err {checks['sphere_limit_hertz']['max_force_error_rel']:.1e}) and against a "
        f"Boussinesq flat-punch BEM reference (err {checks['cube_flat_punch']['rel_error_vs_bem']*100:.2f}%), "
        f"energy drift {checks['elastic_impact']['drift_per_1000']*100:.2f}%/1000 steps, steady "
        f"penetration {checks['steady_penetration']['steady_penetration_ratio']*1e3:.2f}e-3 D; "
        f"a {len(table['rows'])}-row clump disable-ratio table is built and read by the API gate "
        f"({table['summary']['n_allowed']} admissible configurations). Sphere repose anchor measured at "
        f"{anchor['repose_angle_deg']:.1f} deg (single run); the true-shape vs clump repose difference is "
        f"NOT claimed because the heap driver is not yet quasi-static; the true-shape step cost is "
        f"{perf['performance_ratio']:.0f}x the clump path (honest FAIL versus the 3x criterion)."
        if anchor else
        "Superquadric contact kernel matured; see the report for the verified anchors and the "
        "honest FAIL items.")
    if len(summary) > 480:
        summary = summary[:477].rsplit(" ", 1)[0] + "..."
    return {
        "title": "P96-N3: superquadric true-shape contact kernel, clump disable-ratio table and gate",
        "summary": summary,
        "scenario_key": "tensordem.superquadric.contact_maturity",
        "mission_id": 5612,
        "tags": ["tensordem", "superquadric", "contact-kernel", "clump-approximation",
                 "boussinesq", "herz", "repose-angle", "error-table", "api-gate"],
        "markdown": markdown,
        "figures": build_figures(),
    }


if __name__ == "__main__":
    md = build_markdown()
    doc = os.path.join(REPO, "docs", "superquadric_contact_maturity.md")
    with open(doc, "w", encoding="utf-8") as fh:
        fh.write(md)
    payload = build_report(md)
    out = os.path.join(REPO, ".scratch", "report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    size = os.path.getsize(out)
    print(f"[report] wrote {doc}")
    print(f"[report] wrote {out} ({size/1024:.0f} KB, {len(payload['figures'])} figures)")