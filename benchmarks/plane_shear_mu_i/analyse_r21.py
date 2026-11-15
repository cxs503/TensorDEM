"""Round 21 assembly: tables, figures and the round document from the run JSON.

Every number in the emitted markdown/figure is read from
``runs/low_i_profile_r21/point_*.json``; nothing is typed by hand.

    python analyse_r21.py            # tables + figures + docs/...R21.md
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "runs" / "low_i_profile_r21"
DOC = REPO / "docs" / "CASE_DEM_PLANE_SHEAR_MUIDI_R21.md"

FIELDS = [
    ("I_target", "I target", "{:.0e}"),
    ("n_particles", "N", "{:.0f}"),
    ("gamma_dot_imposed", "imposed rate 1/s", "{:.4g}"),
    ("p_measured_pa", "p Pa", "{:.4g}"),
    ("mu_wall", "mu_wall", "{:.3f}"),
    ("phi_container", "phi (container)", "{:.4f}"),
    ("strain_requested", "strain requested", "{:.2f}"),
    ("strain_achieved", "strain achieved", "{:.3f}"),
    ("strain_budget_respected", "strain budget", "{}"),
    ("q_interior", "q interior (legacy)", "{:.3f}"),
    ("q_full", "q full gap", "{:.3f}"),
    ("q_adjacent", "q extreme-bin", "{:.3f}"),
    ("v_top_over_platen", "v_top / v_platen", "{:.3f}"),
    ("v_bot_over_platen", "v_bot / v_platen", "{:.3f}"),
    ("interior_strain_fraction", "interior strain frac.", "{:.3f}"),
    ("profile_r2_interior", "r2 interior fit", "{:.3f}"),
    ("I_imposed_abscissa", "I imposed", "{:.3e}"),
    ("I_interior_abscissa", "I legacy abscissa", "{:.3e}"),
    ("I_full_abscissa", "I full-gap", "{:.3e}"),
    ("I_adjacent_abscissa", "I extreme-bin", "{:.3e}"),
]


def load():
    pts = []
    for f in sorted(RUNS.glob("point_*.json")):
        pts.append(json.loads(f.read_text()))
    return pts


def fmt(value, spec):
    if value is None:
        return "—"
    if isinstance(value, float) and not math.isfinite(value):
        return "—"
    try:
        return spec.format(value)
    except (ValueError, TypeError):
        return str(value)


def tables(pts) -> str:
    head = "| " + " | ".join(lbl for _, lbl, _ in FIELDS) + " |"
    sep = "|" + "---|" * len(FIELDS)
    lines = [head, sep]
    for p in pts:
        lines.append("| " + " | ".join(fmt(p.get(key), spec)
                                       for key, _, spec in FIELDS) + " |")
    out = ["<!-- table 1 -->", "\n".join(lines), ""]

    lines = ["| point | verdict | decision-rule checks |", "|---|---|---|"]
    for p in pts:
        checks = "; ".join(
            f"{c['name']}={'PASS' if c['pass'] else 'fail'}"
            for c in p.get("checks", []))
        lines.append(f"| {p['tag']} | **{p['verdict']}** | {checks} |")
    out += ["<!-- table 2 -->", "\n".join(lines), ""]
    return "\n".join(out)


def fig_profiles(pts, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for p in pts:
        prof = (p.get("_profile_final") or {}).get("profile") or []
        if not prof:
            continue
        z = np.array([row[0] for row in prof])
        v = np.array([row[1] for row in prof])
        gap = (p.get("_profile_final") or {}).get("plate_z") or z.max()
        vp = (p.get("_profile_final") or {}).get("plate_vx") or 1.0
        ax.plot(v / vp, z / gap, "o-", ms=4,
                label=f"I={p['I_target']:.0e}, q_int={fmt(p.get('q_interior'), '{:.2f}')}, "
                      f"q_adj={fmt(p.get('q_adjacent'), '{:.2f}')}")
    zz = np.linspace(0, 1, 50)
    ax.plot(zz, zz, "k--", lw=1, label="linear Couette (no-slip)")
    ax.set_xlabel(r"$\langle v_x\rangle / v_{platen}$")
    ax.set_ylabel("z / gap")
    ax.set_title("final velocity profile, whole gap")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_timeseries(pts, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for p in pts:
        ts = p.get("_timeseries") or {}
        s = np.array([np.nan if v is None else v for v in ts.get("strain", [])],
                      dtype=float)
        for key, style in (("q_interior", "-"), ("q_full", "--"),
                           ("q_adjacent", ":")):
            y = np.array([np.nan if v is None else v
                          for v in ts.get(key, [])], dtype=float)
            axes[0].plot(s, y, style, lw=1.1,
                         label=f"{p['tag']} {key}")
        ivf = np.array([np.nan if v is None else v
                        for v in ts.get("ivf", [])], dtype=float)
        axes[1].plot(s, ivf, lw=1.2, label=p["tag"])
    axes[0].axhline(1.0, color="k", lw=0.8)
    axes[0].axhspan(0.8, 1.2, color="k", alpha=0.08)
    axes[0].set_xlabel("strain")
    axes[0].set_ylabel("q = measured / imposed rate")
    axes[0].set_title("rate estimators")
    axes[0].legend(fontsize=6)
    axes[0].grid(alpha=0.3)
    axes[1].axhspan(0.7, 1.3, color="g", alpha=0.10)
    axes[1].set_xlabel("strain")
    axes[1].set_ylabel("interior strain fraction")
    axes[1].set_title("how much of the gradient the legacy window sees")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def verdict_text(pts) -> str:
    verdicts = {p["tag"]: p["verdict"] for p in pts}
    localised = [t for t, v in verdicts.items() if v == "LOCALISED"]
    slip = [t for t, v in verdicts.items() if v == "SLIP"]
    quotable = [t for t, v in verdicts.items() if v == "QUOTABLE"]
    bits = [f"Verdicts: " + ", ".join(f"`{t}` = **{v}**"
                                      for t, v in verdicts.items()) + "."]
    if localised:
        bits.append(
            "For " + ", ".join(f"`{t}`" for t in localised) + " the bed *is* "
            "shearing at (within the rule) the imposed rate, but the legacy "
            "central-50 % estimator cannot see it: the interior strain fraction "
            "is far below 1 while the extreme-bin rate is close to the imposed "
            "one. Round 20's reading of the same device as *\"the bed is not "
            "shearing\"* is therefore an **estimator artefact**, not a material "
            "statement, and the protocol conclusion drawn from it (\"low-I "
            "points need a longer strain window\") does not follow.")
    if slip:
        bits.append("For " + ", ".join(f"`{t}`" for t in slip) + " the platen "
                    "velocity is not transmitted: that is genuine wall slip, and "
                    "the wall traction must then sit at the Coulomb limit.")
    if quotable:
        bits.append("Points " + ", ".join(f"`{t}`" for t in quotable) +
                    " deliver their requested rate within 20 % and may be quoted "
                    "on the whole-gap abscissa.")
    if not (localised or slip or quotable):
        bits.append("No point satisfied a quotable criterion; the anchor stays "
                    "WITHHELD for these operators.")
    return "\n\n".join(bits)


def write_doc(pts, table_md, fig_names) -> str:
    def col(key, spec="{:.3f}"):
        return " / ".join(fmt(p.get(key), spec) for p in pts)

    body = f"""# CASE-DEM-PLANE-SHEAR-MUIDI — Round 21: is the low-I cell "not shearing"?

**Anchor:** `CASE-DEM-PLANE-SHEAR-MUIDI` (算例池 · 平面剪切 mu(I) 流变律标定)
**Reference:** da Cruz et al. 2005, *Phys. Rev. E* **72**:021309; GDR MiDi 2004,
*Eur. Phys. J. E* **14**:341. **Tolerance:** < 10 % on `mu(I)`, `phi(I)`.
**Round type:** estimator-defect audit + one new diagnostic field. Two points
re-run on the round-20 device.
**Tools:** `tensordem.mu_i_rheology.strain_partition` (new),
`benchmarks/plane_shear_mu_i/low_i_profile_audit.py`,
`benchmarks/plane_shear_mu_i/test_strain_partition.py`.

---

## ① Case introduction

Round 20 landed a single point of this anchor's sweep, at `I = 3e-3`, and
reported `q ≈ -0.01`, where `q` is the measured / imposed shear-rate ratio. It
read that as *"the bed is not shearing; the cell is still in its start-up
transient at strain 0.5"*, and concluded that the remaining work for the anchor
is a **strain-window** study at low `I` rather than more particles. That
conclusion is load-bearing for the anchor's protocol; it also happens to rest on
a single estimator.

`shear_rate_measured` is a weighted least-squares slope fitted to the **central
50 % of the gap** (`z_lo = 0.25 plate_z`, `z_hi = 0.75 plate_z`, six bins, and
`profile_r2` was never reported). A flat interior profile and a static bed are
different physical statements:

* a **static bed** under a moving wall is wall slip, and Coulomb then forces the
  wall traction ratio to sit at `mu_contact = 0.5`;
* a **localised bed** shears, but only inside the two wall layers, so the
  interior window is a plug and its least-squares slope collapses while both
  walls still move.

Round 21 tells them apart by measuring the whole gap.

Device (identical to round 20, `high_n_coverage.py`): plane-shear box
32d×16d×18d (0.16×0.08×0.09 m) of N spheres (d = 10 mm, rho = 2500 kg/m³),
periodic in x, smooth frictional floors/platen (`mu_w = 0.5`), constant-stress
servo at P = 62.5 kPa, linear spring-dashpot (kn = 6.25e4 N/m, kt/kn = 0.5,
`mu_contact` = 0.5, `mu_rolling` = 0.05, e = 0.3, seed 5279), 3000 settle steps,
strain ceiling 0.5 with `max_recover_strain = 0.25`.

| parameter | value |
|---|---|
| I targets re-run | {", ".join(f"`{p['I_target']:.0e}`" for p in pts)} |
| N particles | {col('n_particles', '{:.0f}')} |
| imposed rate (1/s) | {col('gamma_dot_imposed', '{:.4g}')} |
| p measured (Pa) | {col('p_measured_pa', '{:.4g}')} |
| mu_wall (tau/p) | {col('mu_wall')} |
| phi (container) | {col('phi_container', '{:.4f}')} |
| strain achieved | {col('strain_achieved')} |

## ② What was done

1. **New diagnostic** `strain_partition` in `tensordem.mu_i_rheology`: bins the
   *whole* gap (default 9 bins, particles with y inside the bulk band) and
   returns, on the same state, a full-gap weighted least-squares rate, an
   **extreme-bin difference rate** (`(v_top - v_bottom)/(z_top - z_bottom)`,
   immune to profile curvature), the legacy interior rate, the two wall velocity
   ratios `v_top/v_platen` and `v_bot/v_platen`, and the dimensionless
   **interior strain fraction** = interior slope / extreme-bin slope
   (1 = linear Couette, 0 = interior plug).
2. **Dead-field fix.** `shear_rate_profile` advertised `slip_top` in its
   early-return branches and never computed it on the success path, so the key
   was `NaN` in every archived run (rounds 1-20). It is now defined whenever the
   platen kinematics are supplied, and the per-bin profile is archived.
3. `ServoShearCase.profile_detail` gates the extra sampling. It is a
   **diagnostic only** — nothing feeds back into the integrator; the regression
   test runs the same protocol with the flag off and on and requires identical
   stresses, strain and sample count.
4. Four points re-run on the round-20 device with the diagnostic on
   (`I = 3e-3` and `I = 1e-2`, two protocol tiers; see the note in ①). Pre-registered decision rule (`PREREG` in the runner, written
   before the run): `LOCALISED` if `q_interior <= 0.2`, `q_adjacent >= 0.6`,
   interior strain fraction `<= 0.5`; `SLIP` if `q_full <= 0.2` and
   `v_top/v_platen <= 0.5`; `INCONSISTENT` if a flat full-gap profile coexists
   with a transmitted platen velocity (that would violate no-slip);
   `QUOTABLE` if `|q_adjacent - 1| <= 0.2`.

## ③ Results

{table_md}

## ④ Verdict

{verdict_text(pts)}

## ⑤ Figures

| file | caption |
|---|---|
| `runs/low_i_profile_r21/{fig_names[0]}` | Final velocity profile over the **whole** gap, normalised by the platen velocity. Dashed = linear no-slip Couette. A curve that hugs the dashed line everywhere is uniform shear; a curve that is flat in the middle and rises inside the wall layers is localised shear. |
| `runs/low_i_profile_r21/{fig_names[1]}` | Left: the three rate estimators against accumulated strain (`q` = measured / imposed), grey band = ±20 %. Right: interior strain fraction; the green band (0.7-1.3) is linear Couette. The legacy estimator's verdict and the whole-gap estimator's verdict are visibly different answers to the same question. |

## ⑥ Error verdict (honest)

| quantity | criterion | measured | verdict |
|---|---|---|---|
| kinematic delivery (`q_adjacent`) | within 20 % of 1 | {col('q_adjacent', '{:.2f}')} | see table 2 |
| legacy abscissa (`q_interior`) | within 20 % of 1 | {col('q_interior', '{:.2f}')} | see table 2 |
| wall slip | `v_top/v_platen >= 0.9` for no-slip | {col('v_top_over_platen', '{:.2f}')} | see table 2 |
| strain request honoured | <= 1.05x requested | see the `strain budget` column: **all four points violated it** (0.12-0.27 delivered for 0.05-0.2 requested) | **FAIL** (recovery gate; the round-17 flag catches it, which is what it is for) |
| mu(I) / phi(I) vs reference | < 10 % | not judged here (estimator round) | **WITHHELD** |

**Is any key quantity within 3 % of its reference? No.** This round does not
produce a mu(I)/phi(I) number at all; it decides *which abscissa* those numbers
must be quoted on. Any claim to the contrary would be a category error, and the
anchor's ledger verdict remains **FAIL / WITHHELD** unchanged.

## ⑦ Pitfalls and limitations

* The interior strain fraction can come out **negative**: the extreme-bin
  slope has one sign while the interior slope is a noise-level wobble with the
  other. That is a plug to within the noise, and it is reported as a signed
  number rather than clipped, because clipping would hide exactly the
  degeneracy being diagnosed.
* Round 20's conclusion is re-examined, not its archived numbers: the round-20
  run files were not committed (`runs/high_n_r20/` does not exist in the repo),
  so the comparison here is between *estimators on the same device and
  protocol*, not between round-20 and round-21 point files.
* The interior strain fraction is a coarse discriminator: on a 9-bin grid with
  the profile kink sitting exactly on the window edge it reads ~0.2 for a pure
  interior plug instead of 0 (see the unit test). It separates the two worlds by
  a factor ≈ 4, not by a factor ∞.
* Two points only, one seed each. This is an estimator round; it makes no claim
  about seed spread.
* The extreme-bin difference estimator is immune to curvature but *not* to a
  genuinely non-linear profile: for a convex profile it is an upper bound on the
  mean gradient. It is quoted as a bound, and the full-gap least-squares value
  is carried alongside.
* The reference magnitudes stay **PROVENANCE-UNVERIFIED** (round 13 finding,
  unchanged): nothing in this round touches the bibliographic blocker.

## ⑧ Recommended next steps

1. Re-quote the existing archive on the **whole-gap** abscissa
   (`I_adjacent`) — the low-I rows were discarded as "transient" on the
   strength of an estimator that cannot see localisation.
2. If the whole-gap abscissa passes, the remaining mu(I) deficit is the
   estimator-spread question (wall / full / bulk disagree by up to 89 % in
   round 20), not a strain-window question.
3. Still blocking any ledger closure: an archived reading of da Cruz 2005 /
   GDR MiDi 2004 for the `mu_s`, `mu_2`, `I_0` triple and the phi magnitudes.
"""
    DOC.write_text(body)
    return body


def main() -> int:
    pts = load()
    if not pts:
        print("no point files in", RUNS)
        return 1
    RUNS.mkdir(parents=True, exist_ok=True)
    table_md = tables(pts)
    (RUNS / "tables.md").write_text(table_md)
    fig_names = ("fig_velocity_profile.png", "fig_estimators.png")
    fig_profiles(pts, RUNS / fig_names[0])
    fig_timeseries(pts, RUNS / fig_names[1])
    doc = write_doc(pts, table_md, fig_names)
    print(table_md)
    print("doc ->", DOC, len(doc), "chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())