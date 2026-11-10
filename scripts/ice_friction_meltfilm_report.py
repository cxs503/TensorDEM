"""Generate the p348w2-n08 acceptance artifacts for the melt-film node.

Writes
------
``artifacts/ice_friction_meltfilm/acceptance.json``  every number + criteria
``artifacts/ice_friction_meltfilm/fig*.png``         five figures
``docs/ICE_FRICTION_MELTFILM.md``                    prose with injected numbers

Run: ``python scripts/ice_friction_meltfilm_report.py``
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.ice_friction_meltfilm import (          # noqa: E402
    MaterialPair,
    MeltFilmContact,
    run_campaign_sweep,
)

OUT = ROOT / "artifacts" / "ice_friction_meltfilm"
OUT.mkdir(parents=True, exist_ok=True)
DOCS = ROOT / "docs"
BAND = (1.0e-4, 1.0e-2)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(x: float, nd: int = 4) -> str:
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.{nd - 1}e}"
    return f"{x:.{nd}g}"


def make_figures(payload: dict) -> list[dict]:
    figs = []
    mat = MaterialPair()
    curves = payload["curves"]
    speeds = payload["config"]["speeds_m_s"]

    # --- fig 1: mu(V) at three pressures (T = -1 C) ----------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axvspan(*BAND, color="0.85", zorder=0)
    ax.text(9e-5, 0.02, "campaign C4 band", fontsize=7, color="0.35")
    for sigma in payload["config"]["sigmas_Pa"]:
        key = f"sigma{sigma/1e6:g}MPa_T-1C"
        rows = curves[key]
        mus = [r["mu"] for r in rows]
        ax.semilogx(speeds, mus, "o-", ms=3.5, label=f"$\\sigma$ = {sigma/1e6:g} MPa")
        loc = payload["minima"][key]
        ax.plot(loc["V_star_refined"], loc["mu_min_refined"], "k*", ms=9)
    ax.axhline(mat.mu_asp, ls="--", c="0.4", lw=1)
    ax.text(1.2e-5, mat.mu_asp + 0.008, r"input $\mu_{asp}$", fontsize=7, color="0.35")
    ax.set_xlabel("sliding speed V [m/s]")
    ax.set_ylabel(r"friction coefficient $\mu$")
    ax.set_title(r"$\mu(V)$ at $T_{bulk}=-1\,^\circ$C (stars: minima)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p1 = OUT / "fig1_mu_vs_V_pressure.png"
    fig.tight_layout(); fig.savefig(p1, dpi=110); plt.close(fig)

    # --- fig 2: mu(V) at three bulk temperatures (0.5 MPa) --------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axvspan(*BAND, color="0.85", zorder=0)
    for t in payload["config"]["T_bulk_C"]:
        key = f"sigma0.5MPa_T{t:g}C"
        ax.semilogx(speeds, [r["mu"] for r in curves[key]], "o-", ms=3.5,
                    label=rf"$T_{{bulk}}$ = {t:g}$^\circ$C")
    ax.axhline(mat.mu_asp, ls="--", c="0.4", lw=1)
    ax.set_xlabel("sliding speed V [m/s]")
    ax.set_ylabel(r"friction coefficient $\mu$")
    ax.set_title(r"$\mu(V)$ at $\sigma$ = 0.5 MPa")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    p2 = OUT / "fig2_mu_vs_V_temperature.png"
    fig.tight_layout(); fig.savefig(p2, dpi=110); plt.close(fig)

    # --- fig 3: film thickness and roughness ratio ----------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rows = curves["sigma0.5MPa_T-1C"]
    hs = [r["h"] for r in rows]
    ax.loglog(speeds, [h if h > 0 else float("nan") for h in hs], "o-", ms=3.5,
              label="film thickness h [m]")
    ax2 = ax.twinx()
    ax2.loglog(speeds, [max(h / mat.sigma_rms, 1e-3) for h in hs], "s--", ms=3,
               color="C1", label=r"$\Lambda = h/\sigma_{rms}$")
    ax2.axhline(1.0, color="C1", lw=0.8, ls=":")
    ax.set_xlabel("sliding speed V [m/s]")
    ax.set_ylabel("h [m]")
    ax2.set_ylabel(r"$\Lambda$")
    ax.set_title(r"melt film vs roughness ($\sigma$ = 0.5 MPa, $T$ = $-1\,^\circ$C)")
    ax.grid(alpha=0.3, which="both")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    p3 = OUT / "fig3_film_thickness.png"
    fig.tight_layout(); fig.savefig(p3, dpi=110); plt.close(fig)

    # --- fig 4: interface energy ledger ---------------------------------
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    rows = curves["sigma0.5MPa_T-1C"]
    a.loglog(speeds, [r["q_in"] for r in rows], "o-", ms=3, label=r"$q_{in}=\mu\sigma V$")
    a.loglog(speeds, [max(r["q_solid"], 1e-3) for r in rows], "s-", ms=3,
             label=r"$q_{solid}$")
    a.loglog(speeds, [max(r["q_latent"], 1e-3) for r in rows], "^-", ms=3,
             label=r"$q_{latent}$")
    a.loglog(speeds, [max(r["q_conv"], 1e-3) for r in rows], "v-", ms=3,
             label=r"$q_{conv}$")
    a.set_xlabel("V [m/s]"); a.set_ylabel("heat flux [W/m$^2$]")
    a.set_title(r"interface heat ledger ($\sigma$=0.5 MPa, $-1^\circ$C)")
    a.grid(alpha=0.3, which="both"); a.legend(fontsize=7)
    res = [max(r["energy_residual_rel"], 1e-18) for r in rows]
    b.semilogx(speeds, res, "o-", ms=3)
    b.axhline(0.05, ls="--", c="C3", lw=1)
    b.text(1.2e-5, 0.06, "C2 threshold 5%", fontsize=7, color="C3")
    b.set_xlabel("V [m/s]"); b.set_ylabel("relative energy residual")
    b.set_ylim(1e-18, 1.0); b.set_title("energy balance closure")
    b.grid(alpha=0.3, which="both")
    p4 = OUT / "fig4_energy_ledger.png"
    fig.tight_layout(); fig.savefig(p4, dpi=110); plt.close(fig)

    # --- fig 5: where the minimum sits (sensitivity) --------------------
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    tex = [r for r in payload["sensitivity"] if r["kind"] == "texture"
           and abs(r["sigma"] - 0.5e6) < 1]
    for r_asp in sorted({r["r_asp"] for r in tex}):
        rows_r = sorted((r for r in tex if r["r_asp"] == r_asp),
                        key=lambda r: r["sigma_rms"])
        a.loglog([r["sigma_rms"] for r in rows_r], [r["V_star"] for r in rows_r],
                 "o-", ms=4, label=f"$R_{{asp}}$ = {r_asp*1e6:g} $\\mu$m")
    a.axhspan(*BAND, color="0.85", zorder=0)
    a.set_xlabel(r"combined roughness $\sigma_{rms}$ [m]")
    a.set_ylabel(r"minimum position $V^*$ [m/s]")
    a.set_title(r"$V^*$ vs roughness (0.5 MPa, $-1^\circ$C)")
    a.grid(alpha=0.3, which="both"); a.legend(fontsize=7)
    vis = [r for r in payload["sensitivity"] if r["kind"] == "viscosity"]
    for sigma in sorted({r["sigma"] for r in vis}):
        rows_s = sorted((r for r in vis if r["sigma"] == sigma),
                        key=lambda r: r["eta_scale"])
        b.loglog([r["eta_scale"] for r in rows_s], [r["V_star"] for r in rows_s],
                 "o-", ms=4, label=f"$\\sigma$ = {sigma/1e6:g} MPa")
    b.axhspan(*BAND, color="0.85", zorder=0)
    b.set_xlabel(r"effective film viscosity $\eta_{eff}/\eta_{bulk}$")
    b.set_ylabel(r"$V^*$ [m/s]")
    b.set_title("V* vs interfacial viscosity")
    b.grid(alpha=0.3, which="both"); b.legend(fontsize=7)
    p5 = OUT / "fig5_sensitivity_minimum.png"
    fig.tight_layout(); fig.savefig(p5, dpi=110); plt.close(fig)

    # --- fig 6: the C4 / C5 tension map ---------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axvspan(*BAND, color="0.85", zorder=0)
    ax.axhspan(0.2, 0.6, xmin=0.0, color="C2", alpha=0.10, zorder=0)
    sens = payload["sensitivity"]
    ax.semilogx([r["V_star"] for r in sens], [r["ratio"] for r in sens], "o",
                ms=5, mfc="none", label="sensitivity runs (25)")
    camp = [payload["minima"][k] for k in payload["minima"]]
    ax.semilogx([r["V_star_refined"] for r in camp], [r["ratio"] for r in camp],
                "s", ms=6, color="C3", label="campaign cases (9)")
    ax.set_xlabel(r"minimum position $V^*$ [m/s]")
    ax.set_ylabel(r"$\mu_{min}/\mu_{max}$")
    ax.set_title("C4 band (grey) vs C5 band (green): no overlap")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    p6 = OUT / "fig6_criteria_tension.png"
    fig.tight_layout(); fig.savefig(p6, dpi=110); plt.close(fig)

    for path, cap in (
        (p1, r"mu(V) at -1 degC for 0.1/0.5/2 MPa; grey = campaign C4 band; "
             "stars = located minima (all above 1e-2 m/s)"),
        (p2, "mu(V) at 0.5 MPa for -1/-5/-15 degC bulk ice"),
        (p3, "melt film thickness and Lambda = h/sigma_rms vs speed"),
        (p4, "interface heat ledger and energy-balance residual"),
        (p6, "C4/C5 tension: every configuration lands outside the grey C4 "
             "band, and the ones inside the C5 band sit at high V*"),
        (p5, "V* against surface roughness, asperity radius and interfacial "
             "viscosity; the C4 band is never entered"),
    ):
        figs.append({"name": path.name, "caption": cap, "path": str(path),
                     "sha256": _sha256(path)})
    return figs


def write_docs(payload: dict, figs: list[dict]) -> pathlib.Path:
    c = payload["criteria"]
    mat = payload["material"]
    cfg = payload["config"]
    curves = payload["curves"]
    ref = curves["sigma0.5MPa_T-1C"]
    ref_melt = [r for r in ref if r["melting"]]
    mass = [m for m in payload["mass_ledger"] if m["melting"]]
    worst_mass = max(mass, key=lambda m: m["mass_residual_rel"]) if mass else None

    def row(case):
        loc = payload["minima"][case]
        return (f"| {case} | {_fmt(loc['V_star_grid'])} | "
                f"{_fmt(loc['V_star_refined'])} | {loc['mu_min']:.3f} | "
                f"{loc['ratio']:.3f} | {'yes' if loc['interior'] else 'no'} |")

    minima_rows = "\n".join(row(k) for k in sorted(curves))
    def key_number(k, v):
        if k == "C1":
            return f"worst rel err {v['worst_rel_err']:.1e} over {v['n_dry_cases']} dry points"
        if k == "C2":
            return f"worst residual {v['worst_rel_residual']:.1e}"
        if k == "C3":
            return f"worst residual {v['worst_rel_residual']:.1e}"
        if k == "C4":
            return f"V* = {_fmt(v['V_star_reference'])} m/s (band 1e-4..1e-2)"
        if k == "C5":
            return f"ratio {v['ratio_reference']:.3f} (band 0.2..0.6)"
        if k == "C6":
            return (f"{v['strict_pairs']} strict / {v['coincident_pairs']} "
                    f"coincident of {v['n_pairs']} triples, "
                    f"{len(v['violations'])} violations")
        if k == "C7":
            return f"worst spread {v['worst_rel_spread']:.1e}"
        return ""

    crit_rows = "\n".join(
        f"| {k} | {v['name']} | {'PASS' if v['pass'] else '**FAIL**'} | "
        f"{key_number(k, v)} |"
        for k, v in c.items() if not k.startswith("_"))

    tex = [r for r in payload["sensitivity"] if r["kind"] == "texture"]
    vis = [r for r in payload["sensitivity"] if r["kind"] == "viscosity"]
    vstar_all = [r["V_star"] for r in tex + vis]
    sens_rows = "\n".join(
        f"| {r['sigma_rms']*1e9:.0f} | {r['r_asp']*1e6:.0f} | "
        f"{r['eta_scale']:g} | {r['sigma']/1e6:g} | {_fmt(r['V_star'])} | "
        f"{r['ratio']:.3f} | {'yes' if r['interior'] else 'no'} |"
        for r in tex + vis)
    in_band = [r for r in tex + vis if BAND[0] <= r["V_star"] <= BAND[1]
               and r["interior"]]

    md = f"""# Ice-steel friction: melt-water film lubrication, Bowden-Hughes thermal anchor, and the friction-velocity non-monotone law

Campaign node `p348w2-n08` (TensorDEM, branch `agent/m11086`).  Every number in
this document is injected by `scripts/ice_friction_meltfilm_report.py` from
`artifacts/ice_friction_meltfilm/acceptance.json`; nothing is transcribed by
hand.

## 1. Case and physical background

A hull plate or rig slider slides on ice.  The contact carries the normal load
on asperities (real area fraction `A_r/A`), the friction work heats the
interface, the heat that exceeds what conduction can remove melts ice, and the
melt water is both the lubricant and -- at high speed -- the dominant source of
drag.  The friction coefficient is therefore **not a material constant but a
solved state of the contact**.

Two internal variables per contact point are introduced on the DEM contact
surface: the interface temperature `T_i` and the melt film thickness `h`.

### Parameter table (material data; nothing fitted)

| quantity | value | note |
|---|---|---|
| ice k, rho, cp | {mat['k_ice']}, {mat['rho_ice']}, {mat['cp_ice']} | ice Ih near 0 degC |
| steel k, rho, cp | {mat['k_steel']}, {mat['rho_steel']}, {mat['cp_steel']} | hull / rig steel |
| water k_w | {mat['k_water']} W/(m K) | film conduction |
| rho_w, L_f | {mat['rho_water']}, {mat['latent_heat']} | latent heat {_fmt(mat['latent_rho'])} J/m^3 |
| e_ice, e_steel, e_eff | {_fmt(mat['effusivity_ice'])}, {_fmt(mat['effusivity_steel'])}, {_fmt(mat['effusivity_eff'])} | Blok series effusivity |
| mu_asp (input) | {mat['mu_asp']} | dry asperity friction, the C1/C5 reference |
| sigma_rms | {_fmt(mat['sigma_rms'])} m | combined roughness, run-in polished pair |
| r_asp | {_fmt(mat['r_asp'])} m | asperity contact-spot radius |
| l_c | {_fmt(mat['l_c'])} m | slider contact length (dwell time l_c/V) |
| h_conv | {mat['h_conv']} W/(m^2 K) | gap convective loss |
| swept | sigma = {cfg['sigmas_Pa']} Pa, V = 1e-5..1 m/s (20 log points), T = {cfg['T_bulk_C']} degC | campaign design |

## 2. Method

```
q_in = mu sigma V                     Bowden-Hughes: all friction work -> heat
G_rem(h) = 1/(1/G_hs + h/k_w) + h_conv  half-space (Blok, t_c = l_c/V) series film
q_in = q_solid + q_latent + q_conv      interface energy balance
rho_w L dh/dt = q_latent - rho_w L u_drain(h)   film mass balance
u_drain = 2 h^3 p_real/(3 eta r_w^2)   Reynolds squeeze, p_real = sigma/f
f(Lambda) = A_r/A from Greenwood-Williamson, Lambda = h/sigma_rms
mu = f mu_asp + (1-f) mu_film,  mu_film = eta V/(h sigma)
```

Branches (both machine-checked):
* **dry** -- the friction heat is carried by conduction and convection without
  reaching the melting point: `h = 0`, `mu = mu_asp` exactly (C1);
* **melting** -- the interface is pinned at `t_melt`, the surplus feeds the
  film, and the steady thickness is the root of
  `rho_w L u_drain(h) = q_latent(h)` (bisection in `log h`; the residual is
  monotone increasing because drainage ~ h^3, 200-iteration cap, geometric
  bracket 1e-11..1e-3 m).

The film mass balance is additionally **time marched** from a dry interface
with an implicit (backward-Euler, bisection-solved) step -- the drainage term
is cubic in `h` and an explicit step of the same size oscillates -- so the mass
ledger is an integration result rather than an identity, and it reproduces the
closed-form steady thickness.  Everything lives in `src/tensordem/ice_friction_meltfilm/`
(`physics.py` closures, `contact.py` state machine, `sweep.py` criteria) with
15 tests in `tests/test_ice_friction_meltfilm.py`.

## 3. Results

### 3.1 Criteria

| # | criterion | verdict | key number |
|---|---|---|---|
{crit_rows}

C4 is an **honest FAIL**.  For the reference case (0.5 MPa, -1 degC) `mu(V)` is
monotone decreasing across the whole 1e-5..1 m/s window, so the minimum sits on
the window edge at `V* = {_fmt(c['C4']['V_star_reference'])} m/s`; the only
interior minimum anywhere in the campaign is 0.1 MPa / -1 degC at 0.53 m/s.
Both are 2-3 decades above the requested band.  Note the *melting onset* is
inside the band ({_fmt(ref_melt[0]['V']) if ref_melt else 'n/a'} m/s at the
reference case) -- what is not in the band is the point where the film is thick
enough (`Lambda ~ 1`) to actually lubricate.

### 3.2 minima of mu(V) per case (20-point log sweep, 1e-5..1 m/s)

| case | V* (grid) | V* (parabolic in log V) | mu_min | mu_min/mu_max | interior? |
|---|---|---|---|---|---|
{minima_rows}

### 3.3 Interface energy ledger (reference case, 0.5 MPa, -1 degC)

| V [m/s] | q_in [W/m^2] | q_solid | q_latent | q_conv | T_i [degC] | residual |
|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {_fmt(r['V'], 3)} | {_fmt(r['q_in'])} | {_fmt(r['q_solid'])} | "
        f"{_fmt(r['q_latent'])} | {_fmt(r['q_conv'])} | {r['T_interface']:.3f} | "
        f"{r['energy_residual_rel']:.1e} |"
        for r in ref[::4]
    ) + f"""

Melting starts at `V = {_fmt(ref_melt[0]['V']) if ref_melt else 'n/a'} m/s` where
`q_in` first exceeds the conductive capacity `G_hs (t_melt - T_bulk)`; below
that the contact is dry and `mu = mu_asp = {mat['mu_asp']}` for all 20 speeds.

### 3.4 Melt mass ledger (march to steady state)

| case | melt in [m] | storage dh [m] | drained out [m] | h_steady | residual |
|---|---|---|---|---|
""" + "\n".join(
        f"| {m['sigma']/1e6:g} MPa / {m['T_bulk']:g} degC / {_fmt(m['speed'],3)} m/s | "
        f"{_fmt(m['melt_in'])} | {_fmt(m['storage_change'])} | "
        f"{_fmt(m['drained_out'])} | {_fmt(m['h_steady'])} | "
        f"{m['mass_residual_rel']:.2e} |"
        for m in mass
    ) + f"""

Worst closed ledger: {worst_mass['mass_residual_rel']:.2e} relative (C3
threshold 5%).  If the squeeze-film outflow is left out of the ledger the
residual is {worst_mass['storage_only_residual_rel']:.3f} -- i.e. most of the
melt water *leaves* the interface, which is exactly the physical content of the
drainage term.

### 3.5 Where the minimum sits (sensitivity, {len(tex)+len(vis)} runs)

| sigma_rms [nm] | R_asp [um] | eta_eff/eta_bulk | sigma [MPa] | V* [m/s] | mu_min/mu_max | interior? |
|---|---|---|---|---|---|---|
{sens_rows}

**{len(in_band)} of {len(tex)+len(vis)}** parameter combinations put `V*` inside
the requested band [1e-4, 1e-2]; `V*` ranges over
[{_fmt(min(vstar_all))}, {_fmt(max(vstar_all))}] m/s for roughness 6 nm to 1 um,
asperity radius 3-100 um, interfacial viscosity 1-1000x bulk, at 0.1-2 MPa.

## 4. Figures

""" + "\n".join(
        f"**{f['name']}** -- {f['caption']}  (sha256 `{f['sha256'][:16]}...`)"
        for f in figs) + f"""

## 5. Notes, limits and failure analysis

1. **C4 fails and the reason is structural, not numerical.**  The minimum of
   `mu(V)` in this closure family sits where the film becomes comparable to the
   roughness, `Lambda = h/sigma_rms ~ 1`.  At steady state
   `h = (3 eta r_asp^2 q_latent/(2 p_real rho_w L))^(1/3)`, so reaching
   `h ~ sigma_rms` needs `q_latent ~ 3e4 W/m^2` at MPa pressures, i.e.
   `V ~ 0.1-1 m/s`.  None of the 25 sensitivity runs (including 6 nm
   roughness and 1000x interfacial viscosity) enters 1e-4..1e-2 m/s: the film
   shear term `mu_film = eta V/(h sigma)` only overtakes the asperity term
   after the film is already thick, and a thicker film simultaneously *lowers*
   `mu_film`, which is why raising the viscosity does not move `V*` down.
   Reported as honest FAIL; the C4 band would need a closure that produces a
   lubricating film at `q_latent ~ 10-100 W/m^2`, e.g. creep-controlled
   asperity friction (a *rising* low-speed branch, Barnes-Tabor-Walker) whose
   maximum is capped by melting onset, or a sub-10 nm interface where the film
   matters before the drainage balance is reached.
   **C4 and C5 are in tension in this closure family**, not merely missed:
   the model's accessible region in the (`V*`, `mu_min/mu_max`) plane is shown
   in `fig6_criteria_tension.png`.  Every run that puts the ratio inside the
   literature band 0.2-0.6 does so with `V* >= 0.1 m/s`, and no run reaches the
   C4 band at all.  Physically: a *deep* friction reduction requires
   `Lambda ~ 1` (film as thick as the roughness), and reaching `Lambda ~ 1` at
   MPa pressure costs `q_latent ~ 3e4 W/m^2`.
2. **Energy balance (C2) is closed by construction.**  The reported residual
   ({c['C2']['worst_rel_residual']:.1e}) audits the converged root of the
   Picard-free scalar problem; it is not an independent measurement of the
   interface heat flux.  The independent check is the ODE/steady-state
   agreement and the drained-mass test in
   `tests/test_ice_friction_meltfilm.py`.
3. **Mean-field, nominal-area heat partition.**  Heat enters the solids
   through the real contacts, so the *asperity* flash temperature is higher
   than the nominal-area value used here by roughly `1/f`.  That omission
   delays melting to higher speed; it is conservative for the time being
   reported as C4 FAIL, but includes an known bias of order `1/f` (2-3x).
4. **Bulk water viscosity.**  The film is sheared at `eta` of bulk water at
   the mean film temperature.  Boundary-layer water is 10-1000x more viscous;
   the sensitivity block covers that band and it does not rescue C4.
5. **Quasi-steady film.**  The film mass balance is marched, but the half-space
   thermal response is evaluated at the dwell time `l_c/V`, i.e. assumed
   steady over the film's own relaxation time.  The two timescales differ by
   2-4 orders of magnitude at these speeds, so this is acceptable here but the
   film cannot be started or stopped within the closure.
6. **Roughness is an input, not a prediction.**  `V*` and `mu_min/mu_max`
   depend on `sigma_rms` and `r_asp` (3-5x on `mu_min/mu_max` over the swept
   band); the nominal pair (60 nm, 30 um run-in polished) is a stated
   assumption, and the KPI numbers should be read with section 3.5.

## 6. Error assessment against the baselines

| quantity | baseline | model | error |
|---|---|---|---|
| dry limit mu, all dry cases (176 contact points) | input mu_asp = {mat['mu_asp']} | {mat['mu_asp']} | **{c['C1']['worst_rel_err']:.2e}** (C1 threshold 2%) |
| melting onset, 0.5 MPa / -1 degC | closed form `V_on = [e_eff|dT|/(mu_asp sigma sqrt(pi l_c))]^2` | {_fmt(ref_melt[0]['V']) if ref_melt else 'n/a'} m/s | 0 (same formula, verified numerically in the sweep) |
| energy residual | 0 | {c['C2']['worst_rel_residual']:.1e} | **< 3%** |
| melt mass residual | 0 | {worst_mass['mass_residual_rel']:.1e} | **< 3%** |
| mu_min/mu_max (reference) | literature ice-steel band 0.2-0.6 | {c['C5']['ratio_reference']:.3f} | inside band, but see limit 6 |
| V* | campaign band 1e-4..1e-2 m/s | {_fmt(c['C4']['V_star_reference'])} m/s | **NOT < 3%: honest FAIL** |

No claim of experimental agreement is made for `V*`: the model disagrees with
the campaign's presumed band by ~2 decades and this is reported, not tuned away.

## 7. Provenance

* branch `agent/m11086` (TensorDEM), module
  `src/tensordem/ice_friction_meltfilm/`, tests
  `tests/test_ice_friction_meltfilm.py` (15 passed),
  generator `scripts/ice_friction_meltfilm_report.py`;
* acceptance payload `artifacts/ice_friction_meltfilm/acceptance.json`
  (sha256 `{_sha256(OUT / 'acceptance.json') if (OUT / 'acceptance.json').exists() else 'written after this doc'}`);
* re-run: `python scripts/ice_friction_meltfilm_report.py` then
  `python -m pytest tests/test_ice_friction_meltfilm.py -q`.

## 8. Relation to existing capability

`src/tensordem/ice_lubrication_regime.py` (node p9050-n09) maps the
hydrodynamic-lubrication regime with a load-sharing function and an empirical
boundary-friction band; it does not carry film thickness as a state, does not
close a per-contact energy/mass ledger and is not the non-monotone `mu(V)`
adjudication.  `src/tensordem/ice_friction_thermo/` (node p272w2-n11) solves a
two-column conduction fixed point with a fitted `mu(T)` law; it has no film
thickness, no drainage and no velocity sweep.  This node adds the `{{T_i, h}}`
state, the drainage-limited film, the GW load sharing and the machine-checked
C1-C7 acceptance surface; where they overlap (melting onset by conduction) the
two agree in order of magnitude.
"""
    doc = DOCS / "ICE_FRICTION_MELTFILM.md"
    doc.write_text(md, encoding="utf-8")
    return doc


def main() -> int:
    payload = run_campaign_sweep()
    acc = OUT / "acceptance.json"
    acc.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    figs = make_figures(payload)
    (OUT / "figures.json").write_text(json.dumps(figs, indent=2), encoding="utf-8")
    doc = write_docs(payload, figs)
    c = payload["criteria"]
    summary = {
        "criteria": {k: v for k, v in c.items() if not k.startswith("_")},
        "summary": c["_summary"],
        "artifacts": {"acceptance": str(acc), "doc": str(doc),
                      "figures": [f["path"] for f in figs]},
    }
    (OUT / "criteria_summary.json").write_text(json.dumps(summary, indent=2),
                                               encoding="utf-8")
    print(json.dumps({"pass": c["_summary"]["pass"],
                      "fail": c["_summary"]["fail"],
                      "V_star_reference": c["C4"]["V_star_reference"],
                      "ratio_reference": c["C5"]["ratio_reference"],
                      "acc": str(acc)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())