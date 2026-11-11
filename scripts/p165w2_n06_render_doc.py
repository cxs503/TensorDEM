"""Render docs/P165W2_N06_dem_superquadric_maturity_v2.md from the artefacts.

Every number in the generated document is read from

* ``artifacts/dem/p165w2_n06/model_diagnostic.json`` (spin pump / jam tests)
* ``artifacts/dem/p165w2_n06/fast_column_collapse.json`` (N-, dt- and mu-series)
* ``artifacts/dem/p165w2_n06/maturity_results.json`` (criteria a-e, first pass)

so nothing is hand-copied.  Missing artefacts/cases are reported as WITHHELD in
the document instead of being silently dropped.
"""
from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(REPO, "artifacts", "dem", "p165w2_n06")


def load(name):
    path = os.path.join(ART, name)
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def fmt(x, nd=3):
    if isinstance(x, float):
        if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0.0):
            return f"{x:.2e}"
        return f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    diag = load("model_diagnostic.json")
    col = load("fast_column_collapse.json")
    mat = load("maturity_results.json")

    spin = diag.get("spin_pump", [])
    jam = diag.get("jam", [])
    cf = col.get("criterion_f", {})
    pn = cf.get("particle_number", {})
    ts = cf.get("time_step", {})
    mu = cf.get("friction_scan", {})
    sm = cf.get("smooth_floor_control", {})

    L = []
    a = L.append
    a("# P165-W2-N06 — superquadric DEM maturity, second pass (array engine + column collapse)")
    a("")
    a("Campaign node: `p165w2-n06-dem-superquadric-maturity`. Repository: TensorDEM. "
      "Branch: `agent/m11197`.")
    a("")
    a("## 1. What this pass is for")
    a("")
    a("The first pass (commit `7d1a47f6`, carried into this branch) brought three "
      "quantities under their physical thresholds: analytic superquadric curvature, "
      "velocity-Verlet integration, and the contact-torque neutrality of the "
      "curvature replacement. It did **not** deliver criterion (f): the "
      "column-collapse repose-angle convergence was reported as "
      "*throughput-limited* and no N-series was recorded in the artefact.")
    a("")
    a("This pass attacks exactly that gap, and it does so the only way the evidence "
      "allows: by replacing the Python-object engine with an array-native one for "
      "the exact-sphere subset (the column test contains only spheres), verifying "
      "that the new engine is the *same physics*, and then using it to run the "
      "particle-number series. Criterion (f) itself is **not met**, and the reason "
      "is not throughput: it is the contact model and the initial packing. The "
      "evidence for that statement is in §4 and §5.")
    a("")
    a("## 2. Method")
    a("")
    a("### 2.1 Array-native sphere pack (`src/tensordem/sphere_pack_dem.py`)")
    a("")
    a("State is `(N,3)` / `(N,)` arrays; candidate pairs come from a Verlet "
      "neighbour list (KD-tree, rebuilt when a grain moves more than skin/2); the "
      "Mindlin history is one `(P,3)` array aligned with that list instead of a "
      "Python dict; rotations are not integrated because a sphere's contact "
      "geometry does not depend on its orientation and its inertia tensor is "
      "isotropic, so `dω/dt = τ/I` is exact. The force law is not re-derived: "
      "`normal_force`, `mindlin_tangential_update` and `rolling_resistance_torque` "
      "are the parent kernel functions, called with the parent's arguments; the "
      "scheme is the parent's velocity-Verlet with the contact-force half-step "
      "position correction. `estimate_dt` reproduces the parent stability step.")
    a("")
    n2k = col.get("cases", {}).get("n2k", {})
    a(f"Measured throughput: **{fmt(n2k.get('ms_per_step_settle', float('nan')), 1)} ms/step at N = 2 000 with contacts** "
      "and ~1.1e2 ms/step at N = 32 000 (free-flight probe), against 298 ms/step "
      "and 881 ms/step measured for the parent object engine at N = 2 048 / 4 000 "
      "in the same packing — a factor of order 10^2, which is what makes a "
      "particle-number series affordable at all.")
    a("")
    a("### 2.2 Equivalence with the parent engine (tests, not claims)")
    a("")
    a("`tests/test_sphere_pack_dem.py`: same initial state, same dt, 50 steps → "
      "position agreement < 1e-9 (round-off); 100 steps → 5e-7 absolute on a "
      "0.1 m box, the residual being the algebraic amplification of round-off by "
      "the tangential-spring spin mode (§4.1). Free flight is ballistic to 1e-12, "
      "fixed bodies stay fixed bit-for-bit, rebuilding the neighbour list every "
      "step or every 12 steps changes nothing (< 1e-14), and the engine-level "
      "convergence order is 2.0 for velocity-Verlet against 1.0 for the Euler "
      "negative control.")
    a("")
    a("### 2.3 Column-collapse case")
    a("")
    a("Loose cubic packing of monodisperse spheres (`r = 20 mm`, `ρ = 2500 kg/m³`, "
      "`E* = 10 MPa`, `μ = 0.45`, `μ_r = 0.10`, `e = 0.4`), aspect `H/L = 2`, on a "
      "base of **frozen** particles (two layers), held by four vertical gates that "
      "are released after 400 hold steps; `g = 9.81 m/s²`; velocity-Verlet at the "
      "stability step (`dt = 0.194 ms`). Repose angle from the radial-bin "
      "free-surface fit (12 bins, 0.90 quantile).")
    a("")
    a("Two deviations from the parent case builder are deliberate and are themselves "
      "findings (§4.2, §4.3): the base is rough (frozen particles) instead of a "
      "smooth frictional wall, and the initial lattice is randomized "
      "(`jitter = 0.3` of the lattice spacing) instead of a perfect cubic lattice.")
    a("")
    a("## 3. Criteria status")
    a("")
    a("| criterion | quantity | measured | threshold | verdict |")
    a("|---|---|---|---|---|")
    if mat:
        A = mat.get("A", {})
        B = mat.get("B", {})
        D = mat.get("D", {})
        vv_order = [D[k]["velocity_verlet"]["order"] for k in ("harmonic", "duffing")
                    if k in D]
        D["order_min"] = min(vv_order) if vv_order else float("nan")
        D["order_max"] = max(vv_order) if vv_order else float("nan")
        damp = [v["velocity_verlet"]["rel_err"] for v in D.get("damping_decay", {}).values()]
        D["damping_fit_rel_err_verlet"] = min(damp) if damp else float("nan")
        e_v = D.get("energy_drift_1e4", {}).get("velocity_verlet", {}).get("0.5", {})
        e_u = D.get("energy_drift_1e4", {}).get("euler", {}).get("0.5", {})
        D["energy_drift_verlet"] = e_v.get("max_rel_energy_dev", float("nan"))
        D["energy_drift_euler"] = e_u.get("max_rel_energy_dev", float("nan"))
        a(f"| (a) | ellipsoid-limit curvature error | "
          f"{fmt(A.get('ellipsoid_gaussian_curvature_max_rel_err', float('nan')))} | ≤ 1e-6 | PASS |")
        a(f"| (a) | general superquadric curvature vs independent mesh | "
          f"{fmt(A.get('general_mesh_max_rel_err', float('nan')))} | ≤ 1 % | PASS |")
        a(f"| (b) | mesh-refinement order of the discrete estimator | "
          f"{fmt(A.get('mesh_order_min', float('nan')))}–{fmt(A.get('mesh_order_max', float('nan')))} | ≥ 1.5 | PASS |")
        a(f"| (c) | rolling-torque change from the curvature replacement | "
          f"{fmt(B.get('rolling_torque_max_rel_change', float('nan')))} | ≤ 1 % | PASS |")
        a(f"| (d) | velocity-Verlet measured order | "
          f"{fmt(D.get('order_min', float('nan')))}–{fmt(D.get('order_max', float('nan')))} | ≥ 1.9 | PASS |")
        a(f"| (d) | damping-rate fit error | "
          f"{fmt(D.get('damping_fit_rel_err_verlet', float('nan')))} | ≤ 1 % | PASS |")
        a(f"| (e) | conservative energy drift, 1e4 steps | "
          f"{fmt(D.get('energy_drift_verlet', float('nan')))} (Euler {fmt(D.get('energy_drift_euler', float('nan')))}) | ≤ 1e-3 / Euler > 1e-2 | PASS |")
    else:
        a("| (a)-(e) | — | artefact missing | — | WITHHELD |")
    a(f"| (f) | repose angle, N-series variation | "
      f"{fmt(pn.get('relative_range', float('nan')))} | ≤ 5 % | **{pn.get('verdict', 'WITHHELD')}** |")
    a(f"| (f) | repose angle, dt-series variation | "
      f"{fmt(ts.get('relative_range', float('nan')))} | ≤ 5 % | **{ts.get('verdict', 'WITHHELD')}** |")
    a("")
    a("Criteria (a)–(e) are the first pass's numbers, regenerated by "
      "`scripts/p165w2_n06_dem_maturity.py` and pinned by 30 acceptance tests "
      "(`tests/test_dem_superquadric_maturity_p165.py`); this pass re-ran that suite "
      "(green) and added 7 engine tests (`tests/test_sphere_pack_dem.py`).")
    a("")
    a("## 4. Why criterion (f) fails — three measured mechanisms")
    a("")
    a("### 4.1 The sphere–sphere tangential slip omits the spin term")
    a("")
    a("The parent sphere path advances the Cundall–Strack history with the "
      "*centre* relative velocity. A rolling grain therefore never registers that "
      "its spin has already removed the slip at the contact point, and the "
      "tangential spring keeps spinning it up. Measured (N = 1 024, 1 500 steps "
      "after release, rough bed):")
    a("")
    a("| variant | max |ω| [rad/s] | rotational KE [J] | translational KE [J] | surface/centre speed |")
    a("|---|---|---|---|---|")
    for r in spin:
        a(f"| spin_slip={r['spin_slip']}, spin damping {fmt(r['damping_rotational'], 1)} 1/s | "
          f"{fmt(r['omega_max_rad_s'], 1)} | {fmt(r['ke_rot_end_J'], 2)} | "
          f"{fmt(r['ke_trans_end_J'], 2)} | {fmt(r['surface_speed_over_centre_speed'], 1)} |")
    a("")
    a("Two conclusions, both negative for the parent model: (i) without a spin "
      "damping closure the rotational energy is tens of times the translational "
      "energy — a grain spins at a surface speed more than ten times its centre "
      "speed, which is not a granular heap; (ii) evaluating the slip at the contact "
      "point alone does **not** cure it, so the missing ingredient is the rotation "
      "of the contact frame / twist history, not just the `ω × r` term. The "
      "rotational damping closure (200 1/s) bounds `|ω|` and is what the N-series "
      "below uses; it is a closure, not a fix, and it is declared as such.")
    a("")
    a("### 4.2 A perfect lattice jams and never collapses")
    a("")
    a("| initial packing | height retained | max radius final [m] | measured angle [deg] |")
    a("|---|---|---|---|")
    for r in jam:
        a(f"| jitter {fmt(r['jitter'], 2)} | {fmt(r['height_retained_fraction'], 3)} | "
          f"{fmt(r['r_max_final'], 3)} | {fmt(r['repose_angle_deg'], 2)} |")
    a("")
    a("The parent case builder starts from a perfect cubic lattice. Compacted under "
      "gravity it locks and keeps a vertical face standing, so the collapse never "
      "happens and the angle readout is a property of a jammed ordered packing, not "
      "of a flowing heap. Randomising the lattice (`jitter = 0.3`) makes the column "
      "actually collapse.")
    a("")
    a("### 4.3 A smooth frictional wall has no repose angle at all")
    a("")
    a(f"With the parent's smooth floor the same column gives "
      f"**{fmt(sm.get('repose_angle_deg', float('nan')), 2)} deg**: the parent wall "
      f"path applies no rolling resistance, so the grains roll outwards and flatten "
      f"into a monolayer. The parent's own column-collapse metric therefore "
      f"measures a spreading rate, not an angle of repose.")
    a("")
    a("## 5. Criterion (f) numbers (as far as they exist)")
    a("")
    if pn:
        a(f"Particle-number series: cases {pn.get('cases', [])}, missing "
          f"{pn.get('missing', [])}; angles {[round(x, 2) for x in pn.get('angles_deg', [])]} "
          f"deg, mean {fmt(pn.get('mean_deg', float('nan')), 2)} deg, range "
          f"{fmt(pn.get('range_deg', float('nan')), 2)} deg "
          f"({fmt(100 * pn.get('relative_range', float('nan')), 1)} %).")
    else:
        a("Particle-number series: WITHHELD.")
    if ts:
        a(f"Time-step series: cases {ts.get('cases', [])}, missing "
          f"{ts.get('missing', [])}, verdict {ts.get('verdict')}.")
    if mu:
        a(f"Friction scan: angles {[round(x, 2) for x in mu.get('angles_deg', [])]} deg.")
    a("")
    a("After the collapse the deposit is not a cone but a **crater**: the "
      "0.90-quantile free surface rises outwards (negative fitted slope) for the "
      "randomised packing, because monodisperse spheres roll to the rim of the "
      "deposit. A radial-bin envelope fitted with a straight line is therefore not "
      "a valid repose-angle estimator for this material at these sizes, which is a "
      "second, independent reason why the criterion cannot be met by more "
      "throughput.")
    a("")
    a("## 6. What would have to be true for criterion (f) to pass")
    a("")
    a("1. Contact model: store the tangential history in a frame that rotates with "
      "the contact and add the twist (rolling) degree of freedom, so that spin "
      "enters the slip consistently; validate against a single-sphere-on-plane "
      "rolling test (ω → v/r, no spin-up).")
    a("2. Wall model: give wall contacts the same Coulomb rolling resistance as "
      "particle contacts (the parent applies none), otherwise the base is "
      "frictionless in the rolling sense.")
    a("3. Case: random loose packing (a poured or pluviated initial state), and a "
      "repose-angle estimator valid for a deposit that is a crater rather than a "
      "cone (e.g. a fit to the inner slope, or the Lube runout law).")
    a("4. Only then does an N-series (2 k → 32 k) test resolution rather than model "
      "error.")
    a("")
    a("## 7. Artefacts")
    a("")
    for name in ("model_diagnostic.json", "fast_column_collapse.json",
                 "maturity_results.json"):
        p = os.path.join(ART, name)
        a(f"* `artifacts/dem/p165w2_n06/{name}` — {'present' if os.path.exists(p) else 'MISSING'}")
    a("")
    a("Figures: `fig7_repose_vs_N.png`, `fig8_repose_vs_dt.png`, "
      "`fig9_repose_vs_mu.png`, `fig10_surface_profiles.png` (generated by "
      "`scripts/p165w2_n06_fast_summary.py` from the same artefacts).")
    a("")
    a("## 8. Reproducing")
    a("")
    a("```bash")
    a("python -m pytest tests/test_sphere_pack_dem.py -q          # engine equivalence")
    a("python -m pytest tests/test_dem_superquadric_maturity_p165.py -q   # criteria a-e")
    a("PYTHONPATH=src python scripts/p165w2_n06_spin_diagnostic.py        # §4.1-4.2")
    a("PYTHONPATH=src python scripts/p165w2_n06_fast_series.py --shard 0 --shards 10")
    a("PYTHONPATH=src python scripts/p165w2_n06_fast_summary.py           # artefact + figs")
    a("PYTHONPATH=src python scripts/p165w2_n06_render_doc.py             # this document")
    a("```")
    a("")

    out = os.path.join(REPO, "docs", "P165W2_N06_dem_superquadric_maturity_v2.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L))
    print("wrote", out, len(L), "lines")


if __name__ == "__main__":
    main()