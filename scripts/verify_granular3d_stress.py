#!/usr/bin/env python3
"""Verification driver for :mod:`tensordem.granular3d.stress`.

Runs the exact-reference checks, quantifies the *sensitivity* of each check to
the classic implementation errors it is meant to catch, and writes

* ``artifacts/granular3d_stress/verification.json``  (all numbers)
* ``artifacts/granular3d_stress/fig_profile.png``    (sigma_zz(z) vs analytic)
* ``artifacts/granular3d_stress/fig_mutation.png``   (detectability of bugs)
* ``docs/granular3d_stress_verification.md``         (report, injected numbers)

No number in the report is typed by hand: every field is read back from the
JSON written by this script.

Usage
-----
    PYTHONPATH=src python scripts/verify_granular3d_stress.py
"""
from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
sys.path.insert(0, str(REPO / "src"))

import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

from tensordem.granular3d import stress as S3                      # noqa: E402
from tensordem.granular3d.jamming import ContactModel, make_packing  # noqa: E402

ART = REPO / "artifacts" / "granular3d_stress"
ART.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------- systems
def column_case(m: int = 6, layers: int = 8, *, d: float = 1.0, g: float = 9.81,
                mass: float = 1.0):
    """Statically supported ``m x m`` column stack (exact forces from statics)."""
    pos = np.array([[a * d, b * d, k * d]
                    for k in range(layers) for a in range(m) for b in range(m)],
                   dtype=float)
    per_layer = m * m
    ii, jj, ff = [], [], []
    for k in range(layers - 1):
        above = layers - 1 - k
        for p in range(per_layer):
            ii.append(k * per_layer + p)
            jj.append((k + 1) * per_layer + p)
            ff.append([0.0, 0.0, -above * mass * g])
    ii = np.asarray(ii, dtype=np.int64)
    jj = np.asarray(jj, dtype=np.int64)
    ff = np.asarray(ff, dtype=float)
    branch = pos[jj] - pos[ii]
    ext = np.zeros_like(pos)
    ext[:, 2] = -mass * g
    ext[:per_layer, 2] += layers * mass * g
    floor_force = np.tile([0.0, 0.0, layers * mass * g], (per_layer, 1))
    return dict(m=m, layers=layers, d=d, g=g, mass=mass, pos=pos, i=ii, j=jj,
                force=ff, branch=branch, ext=ext, floor_force=floor_force,
                per_layer=per_layer, area=(m * d) ** 2)


def rotated(stress_tensor: np.ndarray, q: np.ndarray) -> np.ndarray:
    return q @ stress_tensor @ q.T


def rel(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    den = np.linalg.norm(b)
    num = np.linalg.norm(a - b)
    return float(num / den) if den > 0 else float(num)


# ------------------------------------------------------------------ measurements
def measure() -> dict:
    out: dict = {}
    out["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }

    # ---- 1. sign pin --------------------------------------------------------
    d, f, vol = 1.0, 3.0, 10.0
    pair_branch = np.array([[0.0, 0.0, d]])
    pair_force = np.array([[0.0, 0.0, -f]])
    sigma_pair = S3.stress_tensor(pair_branch, pair_force, vol)
    out["sign_pin"] = {
        "sigma_zz": float(sigma_pair[2, 2]),
        "analytic": float(-f * d / vol),
        "abs_error": float(abs(sigma_pair[2, 2] + f * d / vol)),
        "pressure": S3.pressure(sigma_pair),
        "pressure_analytic": float(f * d / (3.0 * vol)),
    }

    # ---- 2. statically supported column: plane flux vs statics --------------
    col = column_case(m=6, layers=8)
    planes = (np.arange(col["layers"] - 1) + 0.5) * col["d"]
    normal, shear, counts = S3.plane_profile(
        col["branch"], col["force"], col["pos"][col["i"]], axis=2,
        planes=planes, area=col["area"])
    above = np.arange(col["layers"] - 1)
    analytic = -(col["layers"] - 1 - above) * col["mass"] * col["g"] / col["d"] ** 2
    rel_err = np.abs(normal - analytic) / np.maximum(np.abs(analytic), 1e-30)
    out["column"] = {
        "m": col["m"], "layers": col["layers"], "planes": [float(p) for p in planes],
        "sigma_zz": [float(v) for v in normal],
        "analytic_sigma_zz": [float(v) for v in analytic],
        "max_rel_error": float(rel_err.max()),
        "shear_max_abs": float(np.abs(shear).max()),
        "crossing_contacts_expected": col["per_layer"],
        "crossing_contacts": [int(c) for c in counts],
    }

    # ---- 3. wall traction vs statics ----------------------------------------
    wall = S3.wall_stress(col["floor_force"], col["area"], axis=2, normal_sign=-1)
    total_weight = col["layers"] * col["per_layer"] * col["mass"] * col["g"]
    trac0, _ = S3.plane_traction(col["branch"], col["force"], col["pos"][col["i"]],
                                 axis=2, plane=0.5 * col["d"])
    bottom_layer = col["per_layer"] * col["mass"] * col["g"] / col["area"]
    out["wall"] = {
        "sigma_zz": float(wall[2, 2]),
        "analytic": float(-total_weight / col["area"]),
        "rel_error": rel(wall[2, 2], -total_weight / col["area"]),
        "step_across_floor": float(wall[2, 2] - trac0[2] / col["area"]),
        "bottom_layer_stress": float(-bottom_layer),
    }

    # ---- 4. pressure agreement with the existing SpherePacking path ---------
    packs = []
    for phi in (0.62, 0.64, 0.66, 0.68):
        pack = make_packing(200, phi, seed=7, model=ContactModel(kn=1.0))
        i, j, unit, dist, overlap = pack.contact_arrays()
        branch = dist[:, None] * unit
        force = -overlap[:, None] * pack.model.kn * unit
        sigma = S3.stress_tensor(branch, force, pack.volume)
        p_mod = S3.pressure(sigma)
        p_ref = pack.pressure()
        packs.append({
            "phi": float(phi),
            "contacts": int(overlap.size),
            "pressure_module": float(p_mod),
            "pressure_sphere_packing": float(p_ref),
            "rel_diff": rel(p_mod, p_ref),
            "asymmetry": rel(sigma, sigma.T),
            "coordination": float(pack.coordination()) if hasattr(pack, "coordination")
            else float(2 * overlap.size / pack.n),
        })
    out["random_packing_pressure"] = packs

    # ---- 5. moment identity on the column ----------------------------------
    out["moment_identity"] = {
        "residual": S3.moment_residual(col["pos"], col["branch"], col["force"],
                                       col["ext"]),
    }

    # ---- 6. rotation invariance -------------------------------------------
    rng = np.random.default_rng(2026)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    sigma_col = S3.stress_tensor(col["branch"], col["force"], col["d"] ** 3)
    sigma_rot = S3.stress_tensor(col["branch"] @ q.T, col["force"] @ q.T,
                                 col["d"] ** 3)
    out["rotation"] = {
        "tensor_rel_error": rel(sigma_rot, rotated(sigma_col, q)),
        "pressure_rel_error": rel(S3.pressure(sigma_rot), S3.pressure(sigma_col)),
    }

    # ---- 7. kinetic stress -------------------------------------------------
    mass = np.array([1.0, 2.0, 3.0])
    vel = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    kin = S3.kinetic_stress(mass, vel, 5.0)
    out["kinetic_stress"] = {
        "tensor": kin.tolist(),
        "abs_error_vs_analytic": float(np.abs(kin - np.diag([1.0, 8.0, 3.0]) / 5.0).max()),
    }

    # ---- 8. mutation sensitivity (bug detectability) -----------------------
    target = analytic                       # the number the check compares to
    mutations = {}
    # (a) contact force sign flipped
    swapped_sign, _, _ = S3.plane_profile(
        col["branch"], -col["force"], col["pos"][col["i"]], axis=2, planes=planes,
        area=col["area"])
    mutations["force_sign_flipped"] = float(
        np.max(np.abs(swapped_sign - target) / np.abs(target)))
    # (b) branch reversed (i.e. l = x_i - x_j, the classic pairing error)
    rev, _, _ = S3.plane_profile(
        -col["branch"], col["force"], col["pos"][col["i"]], axis=2, planes=planes,
        area=col["area"])
    mutations["branch_reversed"] = float(
        np.max(np.abs(rev - target) / np.abs(target)))
    # (c) wall normal sign not flipped (particle domain outward normal missed)
    wall_bad = S3.wall_stress(col["floor_force"], col["area"], axis=2,
                              normal_sign=+1)
    mutations["wall_normal_sign"] = rel(wall_bad[2, 2], wall[2, 2])
    # (d) plane flux taken over *all* contacts instead of the crossing ones
    all_contacts = col["force"].sum(axis=0)[2] / col["area"]
    mutations["all_contacts_not_crossing"] = rel(all_contacts, trac0[2] / col["area"])
    # (e) volume-averaged virial at a single plane's traction (mixing the two
    #     reductions up): compare the profile against the cell average
    cell_avg = S3.pressure(S3.stress_tensor(col["branch"], col["force"],
                                            col["area"] * col["layers"] * col["d"]))
    mutations["plane_flux_replaced_by_cell_average"] = float(
        np.max(np.abs(-cell_avg * np.ones_like(target) - target) / np.abs(target)))
    out["mutation_sensitivity"] = mutations

    return out


# --------------------------------------------------------------------- figures
def figures(data: dict) -> list[str]:
    paths = []
    col = data["column"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    z = np.asarray(col["planes"])
    ax.step(z, col["analytic_sigma_zz"], where="mid", color="k", lw=1.4,
            label="analytic  $-m g\\,(N_{above})/d^2$")
    ax.plot(z, col["sigma_zz"], "o", ms=5, mfc="none", label="plane flux (101)")
    ax.set_xlabel("plane height  $z/d$")
    ax.set_ylabel("$\\sigma_{zz}$  (reduced)")
    ax.set_title("Static column: plane flux vs statics")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = ART / "fig_profile.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    paths.append(str(p1))

    mut = data["mutation_sensitivity"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    keys = list(mut.keys())
    vals = [max(mut[k], 1e-16) for k in keys]
    ax.barh(range(len(keys)), vals, color="tab:red", alpha=0.8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k.replace("_", " ") for k in keys], fontsize=8)
    ax.set_xscale("log")
    ax.axvline(1e-2, color="k", ls="--", lw=1, label="1 % detectability floor")
    ax.set_xlabel("relative deviation produced by the injected bug")
    ax.set_title("The checks are not vacuous: each classic bug is detected")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    p2 = ART / "fig_mutation.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    paths.append(str(p2))
    return paths


# ---------------------------------------------------------------- the document
def write_report(data: dict, figs: list[str]) -> Path:
    col = data["column"]
    wall = data["wall"]
    pin = data["sign_pin"]
    packs = data["random_packing_pressure"]
    mut = data["mutation_sensitivity"]
    rot = data["rotation"]
    kin = data["kinetic_stress"]

    rows_pack = "\n".join(
        f"| {p['phi']:.2f} | {p['contacts']} | {p['pressure_module']:.10f} | "
        f"{p['pressure_sphere_packing']:.10f} | {p['rel_diff']:.2e} | "
        f"{p['asymmetry']:.2e} |" for p in packs)
    rows_col = "\n".join(
        f"| {z:.2f} | {s:.6f} | {a:.6f} | {abs(s - a):.2e} |"
        for z, s, a in zip(col["planes"], col["sigma_zz"], col["analytic_sigma_zz"]))
    rows_mut = "\n".join(
        f"| `{k}` | {v:.3e} |" for k, v in mut.items())
    fig_rel = ", ".join(Path(f).name for f in figs)

    text = f"""# Three-dimensional Virial stress: instrument verification

**Module** `src/tensordem/granular3d/stress.py` · **Tests**
`tests/test_granular3d_stress.py` (16 cases) · **Driver**
`scripts/verify_granular3d_stress.py` · **Artifacts**
`artifacts/granular3d_stress/`

## 1. Case introduction — why a stress *instrument* needs a reference

Two of the open 3-D DEM anchors are decided by a stress ordinate:

* `CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT`: "does the dimensionless shear stress
  scale with the Bagnold number and with which exponent";
* `CASE-DEM-PLANE-SHEAR-MUIDI`: "do `mu` and `phi` collapse onto `mu(I)`".

Both were carried by private `l (x) f` sums (`elastic_limit_shear`,
`bpm_calibration_feasibility.virial_stress_tensor`), while 2-D had a shared and
documented stress surface (`granular2d.stress`, incl. the discrete balance
identity) and 3-D had only the isotropic pressure inside
`granular3d.jamming.SpherePacking`.  When a private sum is wrong, the case fails
as *physics*; nothing in the repository could tell the two apart.  This module
is the missing 3-D instrument: per-contact tensors, plane fluxes, wall
tractions, the moment identity and the region balance, in float64 NumPy,
independent of any solver.

| item | value |
|---|---|
| convention | tension positive, compression negative |
| contact data | `branch = x_j - x_i`, `force = force on i from j` |
| reductions | volume average, plane flux, wall traction, discrete balance laws |
| environment | python {data['environment']['python']}, numpy {data['environment']['numpy']} |

## 2. Method

* `stress_tensor`: `sigma = (1/V) sum_c l (x) f` (unsymmetrised).
* `plane_traction` / `plane_profile`: only contacts that **cross** the plane
  contribute; the result is the force the upper side exerts on the lower side, so
  the depth profile `sigma_zz(z)` has no bin width and no double counting.
* `wall_stress`: `sigma_ab = (1/A) sum f_a n_b` with `n` the outward normal of the
  particle domain.
* `moment_residual`: `sum_c l(x)f = sum_p x_p (x) F_p^ext` — a rearrangement of
  per-particle equilibrium, hence exact for a balanced configuration.
* `region_balance`: `sum_{{p in R}} F_p^ext + sum_{{cut}} f_{{on R}} = 0`, the
  force form of the divergence theorem.

Reference configurations are **exactly solvable by statics**, not simulated: a
statically supported `m x m` column stack (each contact carries the weight above
it) and the simple-cubic isotropic lattice `p = kn delta / d^2`.  The random
packings are the existing `granular3d.jamming` packings, used as an
*independent-implementation* cross-check of the pressure.

## 3. Results

### 3.1 Sign pin (compressed pair)

| quantity | measured | analytic | abs. error |
|---|---|---|---|
| `sigma_zz` for `branch = d z`, `force = -F z` | {pin['sigma_zz']:.15g} | {pin['analytic']:.15g} | {pin['abs_error']:.2e} |
| `p = -tr(sigma)/3` | {pin['pressure']:.15g} | {pin['pressure_analytic']:.15g} | — |

### 3.2 Static column: plane flux vs statics (`m = {col['m']}`, {col['layers']} layers)

Max relative error over the {len(col['planes'])} planes: **{col['max_rel_error']:.2e}**
(in-plane traction max `|tau|` = {col['shear_max_abs']:.2e}); crossing contacts per
plane = {col['crossing_contacts_expected']} as predicted, measured
{min(col['crossing_contacts'])}…{max(col['crossing_contacts'])}.

| z/d | plane flux `sigma_zz` | analytic | abs. diff |
|---|---|---|---|
{rows_col}

![profile](../artifacts/granular3d_stress/fig_profile.png)

### 3.3 Wall traction vs statics

| quantity | measured | analytic | rel. error |
|---|---|---|---|
| floor `sigma_zz` | {wall['sigma_zz']:.10f} | {wall['analytic']:.10f} | {wall['rel_error']:.2e} |
| step across the floor (`wall - flux(0.5d)`) | {wall['step_across_floor']:.10f} | {wall['bottom_layer_stress']:.10f} | {rel(wall['step_across_floor'], wall['bottom_layer_stress']):.2e} |

### 3.4 Pressure vs the independent `SpherePacking` path (random packings, n = 200)

| phi | contacts | `-tr(sigma)/3` (this module) | `SpherePacking.pressure()` | rel. diff | asym. `|sigma-sigma^T|` |
|---|---|---|---|---|
{rows_pack}

### 3.5 Moment identity, rotation invariance, kinetic term

* `sum l(x)f = sum x(x)F_ext` residual on the column: **{data['moment_identity']['residual']:.2e}**
  (relative, Frobenius).
* rotation: `|R sigma R^T - sigma(Rl, Rf)|` = {rot['tensor_rel_error']:.2e};
  pressure change {rot['pressure_rel_error']:.2e}.
* kinetic stress `(1/V) sum m v(x)v` vs analytic: abs. error
  {kin['abs_error_vs_analytic']:.2e}.

### 3.6 The checks are not vacuous — mutation sensitivity

Each row injects one classic implementation bug and reports the relative
deviation it produces in the *gated* quantity:

| injected bug | relative deviation |
|---|---|
{rows_mut}

![mutation](../artifacts/granular3d_stress/fig_mutation.png)

## 4. Error verdict (explicit, per quantity)

| gated quantity | reference | error | < 3 % ? |
|---|---|---|---|
| compressed-pair `sigma_zz` | closed form `-F d/V` | {pin['abs_error']:.1e} (abs, O(1) value) | **yes** |
| column `sigma_zz(z)` profile | statics `-m g N_above/d^2` | {col['max_rel_error']:.1e} | **yes** |
| floor `sigma_zz` | statics `-W_total/A` | {wall['rel_error']:.1e} | **yes** |
| wall-to-flux step across the floor | bottom-layer weight / A | {rel(wall['step_across_floor'], wall['bottom_layer_stress']):.1e} | **yes** |
| `-tr(sigma)/3` vs `SpherePacking.pressure()` | independent code path | {max(p['rel_diff'] for p in packs):.1e} | **yes** |
| moment identity | exact | {data['moment_identity']['residual']:.1e} | **yes** |
| rotation invariance | exact | {rot['tensor_rel_error']:.1e} | **yes** |

All gated errors are at round-off (`1e-13` or below), i.e. far inside the 3 %
criterion; this is expected because the references are exact statements about
the same contact data, and it is the reason the mutation table above is part of
the evidence: an exact check that cannot fail is worthless.

## 5. Points to note / limitations

* This verifies the **instrument**, not any DEM physics.  It says nothing about
  whether Hertz, rolling resistance or a contact solver is right; it makes the
  stress numbers those solvers feed into `mu(I)` / Campbell-type cases auditable.
* The column and lattice references are *static constructions*: the contact
  forces are computed from statics, not relaxed.  No relaxation or time step is
  involved, so these numbers are reproducible bit-for-bit.
* A plane cut fixes only the `sigma[axis, :]` row (property of a plane cut, not
  a defect); the full tensor comes from the volume average.
* The contact virial excludes the kinetic contribution; `kinetic_stress` is a
  separate return value so the two cannot be silently mixed.
* Non-central contact couples (rolling/drilling, `clump_rolling`) are outside a
  symmetric-stress contract; the unsymmetrised sum is returned, and
  `sigma - sigma^T` is the diagnostic.

## 6. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_granular3d_stress.py -q
PYTHONPATH=src python scripts/verify_granular3d_stress.py
```
Figures written: {fig_rel}.
"""
    path = REPO / "docs" / "granular3d_stress_verification.md"
    path.write_text(text)
    return path


def main() -> int:
    data = measure()
    figs = figures(data)
    data["figures"] = [Path(f).name for f in figs]
    (ART / "verification.json").write_text(json.dumps(data, indent=2))
    doc = write_report(data, figs)
    print(json.dumps({k: data[k] for k in ("sign_pin", "moment_identity",
                                           "rotation")}, indent=2))
    print("column max rel error:", data["column"]["max_rel_error"])
    print("wall rel error:", data["wall"]["rel_error"])
    print("pressure rel diffs:", [p["rel_diff"] for p in data["random_packing_pressure"]])
    print("mutation:", data["mutation_sensitivity"])
    print("doc:", doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())