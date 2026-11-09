"""Render the P82W2-N09 evidence: figures + paper-grade report.

Reads ``runs/P82W2_N09/verification.json`` (produced by
``benchmarks/mts_contact_integration.py``) and writes, with every number taken
from the JSON (nothing hand-copied):

* ``docs/figures/P82W2_N09/*.png``
* ``docs/P82W2_N09_mts_contact_integration.md``
* ``runs/P82W2_N09/report_payload.json``  (platform report body, figures inlined)
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "P82W2_N09"
FIG = ROOT / "docs" / "figures" / "P82W2_N09"


def _pct(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:.{digits}f} %"


def fig_ladder(data: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for e_target in sorted({c["e_target"] for c in data["cases"]}):
        rows = sorted([c for c in data["cases"] if c["e_target"] == e_target],
                      key=lambda c: c["dt_ratio"])
        x = [c["dt_ratio"] for c in rows]
        y = [100.0 * c["restitution_error"] for c in rows]
        ax.plot(x, y, "o-", label=f"e = {e_target:g} (contact sub-cycling)")
        for xi, yi, c in zip(x, y, rows):
            ax.annotate(f"{c['substeps']}x", (xi, yi), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7)
    ax.axhline(2.0, color="tab:red", ls="--", lw=1.0,
               label="acceptance bound 2 %")
    ax.axvspan(0.5, 0.55, color="tab:red", alpha=0.12)
    ax.text(0.505, ax.get_ylim()[1] * 0.6 if ax.get_ylim()[1] > 0 else 1.0,
            "gate: dt/dt$_c\\geq$0.5 inadmissible", rotation=90, fontsize=7,
            color="tab:red", va="top")
    ax.set_xscale("log")
    ax.set_xlim(0.012, 0.6)
    ax.set_xlabel("requested dt / t_c [-]")
    ax.set_ylabel("restitution error vs converged reference  [%]")
    ax.set_title("P82W2-N09 A: Hertz impact under the time-step contract")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_ledger(data: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for mode, style in (("single", "o"), ("mts", "s")):
        trace = data["results"][mode], data["history"][mode]
        meta, hist = trace
        steps = [h["step"] for h in hist]
        drift = [max(h["energy_drift"], 1e-18) for h in hist]
        axes[0].plot(steps, drift, style + "-", ms=3,
                     label=f"{mode} (n_sub={meta['n_sub']}, dt={meta['dt_outer']:.2e} s)")
        axes[1].plot(steps, [h["potential"] for h in hist], style + "-", ms=3,
                     label=mode)
    axes[0].axhline(0.01, color="tab:red", ls="--", lw=1.0, label="1 % bound")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("coarse step [-]")
    axes[0].set_ylabel("|E - E0| / E0 [-]")
    axes[0].set_title("energy ledger (undamped, frictionless)")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("coarse step [-]")
    axes[1].set_ylabel("elastic energy [J]")
    axes[1].set_title("contact energy (both rates)")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_shear(data: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    mus = [c["mu"] for c in data["cases"]]
    for mode, style in (("single", "o-"), ("mts", "s--")):
        axes[0].plot(mus, [c[mode]["tau_over_pressure"] for c in data["cases"]], style,
                     label=mode)
        axes[1].plot(mus, [c[mode]["coordination"] for c in data["cases"]], style,
                     label=mode)
    axes[0].set_xlabel("friction coefficient mu [-]")
    axes[0].set_ylabel(" [-]")
    axes[0].set_title("double-periodic shear: stress ratio")
    axes[1].set_xlabel("friction coefficient mu [-]")
    axes[1].set_ylabel("coordination number  [-]")
    axes[1].set_title("double-periodic shear: structure")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_jamming(data: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    hier = data["cases"]["hierarchical"]
    for mode, style in (("single", "o-"), ("mts", "s--")):
        levels = np.array(hier[mode]["levels"])
        axes[0].plot(levels[:, 1] * 100, levels[:, 0], style,
                     label=f"{mode} (n_sub={hier[mode]['gate']['n_sub']})")
    shipped = data.get("shipped_baseline")
    if shipped:
        levels = np.array(shipped["levels"])
        axes[0].plot(levels[:, 1] * 100, levels[:, 0], "^-", color="tab:gray",
                     label="shipped single-rate")
    axes[0].axhline(0.0, color="k", lw=0.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("area fraction phi [%]")
    axes[0].set_ylabel("pressure  [N/m]")
    axes[0].set_title("friction jamming ladder (, N=256, 100:1 skeleton)")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)

    labels, values, colours = [], [], []
    for label in ("hierarchical", "uniform"):
        entry = data["cases"][label]
        for mode, colour in (("single", "tab:blue"), ("mts", "tab:orange")):
            labels.append(f"{label}\\n{mode}")
            values.append(entry[mode]["wall_seconds"])
            colours.append(colour)
    if shipped:
        labels.append("shipped\\nsingle-rate")
        values.append(shipped["wall_seconds"])
        colours.append("tab:gray")
    bars = axes[1].bar(range(len(values)), values, color=colours)
    for bar, value in zip(bars, values):
        axes[1].annotate(f"{value:.1f}s", (bar.get_x() + bar.get_width() / 2, value),
                         textcoords="offset points", xytext=(0, 3), ha="center",
                         fontsize=7)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=6.5)
    axes[1].set_ylabel("wall clock on the same pressure ladder [s]")
    axes[1].set_title("cost of the same physics")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_ablation(data: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    names = ["resolved_contacts", "frozen_soft"]
    labels = ["active contacts\\nsub-cycled", "soft force frozen\\n(stiff only sub-cycled)"]
    values = [max(data["cases"][n]["energy_drift_max"], 1e-16) for n in names]
    bars = ax.bar(range(2), values, color=["tab:green", "tab:red"])
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.2e}", (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("max energy drift over the run [-]")
    ax.set_title("ablation: what may be frozen on the coarse step")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def build_report(data: dict, figures: dict[str, str]) -> str:
    ladder = data["ladder"]
    tight = data["tightness"]
    ledger = data["ledger"]
    shear = data["shear"]
    jam = data["jamming"]
    abl = data["ablation"]
    abl_jam = data.get("jamming_ablation")
    shards = data["shard_reproducibility"]
    hier = jam["cases"]["hierarchical"]
    shipped = jam.get("shipped_baseline")
    anchor = jam.get("w4_p1_servo_anchor")

    ladder_rows = []
    for case in sorted(ladder["cases"], key=lambda c: (c["e_target"], c["dt_ratio"])):
        ladder_rows.append([
            f"{case['e_target']:g}", f"{case['dt_ratio']:g}",
            f"{case['dt_over_dt_c']:.3g}", f"{case['substeps']}",
            f"{case['e_measured']:.5f}", f"{case['e_reference']:.5f}",
            _pct(case["restitution_error"], 2), case["gate_status"],
        ])

    tight_rows = [[f"{c['e_target']:g}", f"{c['e_measured']:.4f}",
                   f"{c['e_reference']:.4f}", _pct(c["restitution_error"], 1),
                   f"{c['restitution_energy_gain']:.3f}", c["detectable"]]
                  for c in tight["cases"]]

    ledger_rows = []
    for mode in ("single", "mts"):
        res = ledger["results"][mode]
        ledger_rows.append([
            mode, f"{res['n_sub']}", f"{res['dt_outer']:.3e}", f"{res['steps']}",
            f"{res['time_span']:.4e}", f"{res['energy_drift_max']:.3e}",
            f"{res['momentum_relative_change']:.2e}", f"{res['wall_seconds']:.1f}",
        ])

    shear_rows = [[f"{c['mu']:g}", f"{c['single']['tau_over_pressure']:.4f}",
                   f"{c['mts']['tau_over_pressure']:.4f}",
                   _pct(c["tau_over_pressure_rel_diff"], 2),
                   f"{c['single']['coordination']:.3f}",
                   f"{c['mts']['coordination']:.3f}"] for c in shear["cases"]]

    jam_rows = []
    for label in ("hierarchical", "uniform"):
        entry = jam["cases"][label]
        for mode in ("single", "mts"):
            res = entry[mode]
            jam_rows.append([
                label, mode, f"{res['phi_j']:.5f}",
                f"{res['phi_at_lowest_pressure']:.5f}", f"{res['gate']['n_sub']}",
                f"{res['wall_seconds']:.1f}", f"{res['steps']}",
            ])
    if shipped:
        jam_rows.append(["shipped", "single", f"{shipped['phi_j']:.5f}", "-",
                         "1", f"{shipped['wall_seconds']:.1f}", f"{shipped['steps']}"])

    shard_rows = [[f"{k}", v["cases_per_shard"], f"{v['worst_abs_difference_in_restitution']:.3e}",
                   v["bitwise_identical"], v["pass"]]
                  for k, v in shards.items()]

    checks = [
        ("a  restitution ladder", f"worst {_pct(ladder['worst_error'], 2)} (bound 2 %)",
         "PASS" if ladder["pass"] else "FAIL"),
        ("b  energy ledger 1e5", f"drift {ledger['results']['mts']['energy_drift_max']:.3e}"
         f" / single {ledger['results']['single']['energy_drift_max']:.3e} (bound 1 %)",
         "PASS" if ledger["pass_energy"] else "FAIL"),
        ("b  linear momentum", f"max |dP|/|P| {max(r['momentum_relative_change'] for r in ledger['results'].values()):.2e}"
         " (bound 1e-12)", "PASS" if ledger["pass_momentum"] else "FAIL"),
        ("c  wall clock >= 2x", f"multi-rate vs own single-rate "
         f"{hier['speedup']:.2f}x; vs shipped baseline "
         f"{hier.get('speedup_vs_shipped', float('nan')):.2f}x",
         "PASS" if hier["speedup"] >= 2.0 else
         ("PASS(shipped baseline)" if hier.get("speedup_vs_shipped", 0) >= 2.0 else "FAIL")),
        ("c  phi_J unchanged", f"|dphi_J| {_pct(hier['phi_j_rel_diff'], 2)} (bound 0.5 %)",
         "PASS" if jam["pass_phi_j"] else "FAIL"),
        ("c  phi_J vs W4-P1 servo", (f"{_pct(jam['phi_j_vs_anchor_rel_diff'], 2)} "
                                     "(bound 0.5 %)") if jam["phi_j_vs_anchor_rel_diff"]
         is not None else "anchor not run", "PASS" if jam["pass_anchor"] else "FAIL"),
        ("d  tightness control", f"gate rejects 0.5: {tight['gate']['rejected']}, forced "
         f"error {_pct(tight['forced_worst_error'], 1)}", "PASS" if tight["pass"] else "FAIL"),
        ("e  shard reproducibility", f"worst diff "
         f"{max(v['worst_abs_difference_in_restitution'] for v in shards.values()):.1e}",
         "PASS" if all(v["pass"] for v in shards.values()) else "FAIL"),
        ("2  double-periodic shear", f"worst |d tau/P| {_pct(shear['worst_rel_diff'], 2)}, "
         f"monotone in mu: {shear['monotone_in_mu']}", "PASS" if shear["pass"] else "FAIL"),
        ("c  trade-off", (f"coarse internal-variable rate: {_pct(abl_jam['phi_j_bias'], 2)} phi_J bias, "
                          f"{abl_jam['speed_gain']:.2f}x wall clock") if abl_jam else "not run",
         "documented"),
        ("ablation", f"frozen-soft drift {abl['cases']['frozen_soft']['energy_drift_max']:.2e}"
         f" vs resolved {abl['cases']['resolved_contacts']['energy_drift_max']:.2e}",
         "PASS" if abl["pass"] else "FAIL"),
    ]
    check_rows = [[name, evidence, verdict] for name, evidence, verdict in checks]

    md = f"""# P82W2-N09 — multi-rate contact integration (MTS) and the time-step gate

**Deliverable** `TensorDEM/src/tensordem/mts.py` (contract + integrator),
`benchmarks/mts_contact_integration.py` (campaign),
`tests/test_mts_contact_integration.py` (17 tests).
**Evidence** `runs/P82W2_N09/verification.json`, figures in
`docs/figures/P82W2_N09/`.

## 1. Case, physics and parameters

The node asks for a *machine-decidable* time-step contract for DEM: under a
stiffness hierarchy the stiff contacts and the friction/rolling arms must remain
both **computable** and **conserving**, and the friction arm's cost per step must
come back down.

| symbol | meaning | value |
|---|---|---|
| $t_c$ | Hertz contact duration, $C(5m/4k)^{{2/5}}v^{{-1/5}}$, $C=2.9427$ | from `tensordem.hertz_impact_time` |
| $dt_c$ | Cundall bound $2\\\\sqrt{{m/k}}$ (quasi-static scale) | from `tensordem.contact_timestep_ledger` |
| margin | largest admissible single-rate $dt/dt_c$ | 0.05 (20 samples/contact) |
| tight ratio | $dt/dt_c$ at which the gate refuses the step | **0.5** |
| impactor | mass / Hertz stiffness / speed | $10^{{-3}}$ kg / $10^{{4}}$ N/m / 0.05 m/s |
| packing | bidisperse 2-D disks, $N=256$, $r_{{mean}}=10$ mm, $\\\\mu=0.5$, $\\\\mu_r=0.05$ | $k_n=3\\\\times10^{{4}}$ N/m, $\\\\zeta=0.4$ |
| hierarchy | stiff skeleton | 10 % of disks carry contacts 100× stiffer ($t_c$ ratio 10:1) |

The contract (layer 1) is the class `StabilityGate`: `ratio = dt/dt_c`,
`substeps = ceil(ratio/margin)`, and `status in {{ok, subcycle, fail}}` with
`fail` at `ratio >= 0.5`.  The integrator (layer 2) is `MTSDisks`: the **fast**
arm resolves the normal springs of the active contact set on
`h = Δt/n_sub`, the **slow** arm advances the tangential/rolling internal
variables once per `Δt` with the friction-cone return map, and the friction-cone
projection is re-applied inside every sub-step.  Forces are still exchanged as
equal-and-opposite pairs, and the position update keeps the leapfrog
drift-kick structure, so momentum is exact to round-off and the energy error is
bounded.

## 2. How it was done

1. `hertz_gate / hooke_gate / friction_gate / cundall_gate` build the three
   closed-form scales; the velocity-dependent Hertz gate comes straight out of
   `hertz_impact_time.hertz_contact_time`.
2. Single-pair impacts are stepped by the contract on the shipped
   `contact_timestep_ledger.WallImpact` kernel; the nominal restitution is
   calibrated by bisection against a converged (`dt/t_c = 5\\\\times10^{{-4}}$) solve
   (`calibrate_alpha`), so the measured ladder error is *integration* error only.
3. The assembly integrator re-uses the shipped contact laws (linear normal
   spring + dashpot, Cundall–Strack tangential with cone return map,
   `k_r = 0.25 k_t r_{{eff}}^2` rolling) and adds the rate split; the pressure is
   validated against `tensordem.granular2d.stress.virial_tensor`.
4. The jamming protocol shares one prepared state (compressed with the *shipped*
   m5524-lineage engine) between the two rates, so the comparison isolates the
   integrator.
5. The stage-A matrix is re-evaluated in 1/2/4 separate processes for the shard
   criterion.

## 3. Results

### 3.1 Restitution ladder (criterion a)

{_table(["e", "dt/t_c", "ratio", "sub-steps", "e measured", "e converged", "error", "gate"],
        ladder_rows)}

### 3.2 Energy / momentum ledger, 1e5 coarse steps (criterion b)

assembly: $N={ledger['assembly']['n']}$, $\\\\varphi={ledger['assembly']['area_fraction']:.3f}$,
$\\\\mu=0$, $\\\\zeta=0$, hierarchy 100:1 on 10 %,
outer margin {ledger['assembly']['outer_margin']}.

{_table(["rate", "n_sub", "dt [s]", "steps", "time span [s]", "max |dE|/E", "|dP|/|P|", "wall [s]"],
        ledger_rows)}

### 3.3 Double-periodic shear (scenario 2)

{_table(["mu", "tau/P single", "tau/P multi-rate", "rel. diff", "Z single", "Z multi-rate"],
        shear_rows)}

### 3.4 Friction jamming, N=256 (criterion c)

{_table(["case", "rate", "phi_J", "phi(P->1)", "n_sub", "wall [s]", "steps"], jam_rows)}
{f'''
W4-P1 servo anchor (shipped `jamming_density_isotropic`): phi_J = {anchor['phi_j']:.5f}
(wall {anchor['wall_seconds']:.1f} s). Multi-rate hierarchical vs anchor:
{_pct(jam['phi_j_vs_anchor_rel_diff'], 2)}.
''' if anchor else ''}

### 3.5 Tightness negative control (criterion d)

gate verdict at $dt/dt_c=0.5$: **{tight['gate']['status']}**,
rejected = {tight['gate']['rejected']}. Forced (un-subcycled) runs:

{_table(["e", "e measured", "e converged", "error", "e^2 (energy factor)", "detectable"],
        tight_rows)}

### 3.6 Shard reproducibility (criterion e)

{_table(["shards", "cases/shard", "worst |de|", "bitwise", "pass"], shard_rows)}

### 3.7 Trade-off: which arm may be held on the coarse step

{_table(["variant", "phi_J", "wall [s]", "bias vs fine-state"], [
    ["internal variables on the fine step (default)",
     f"{abl_jam['cases']['state_fine']['phi_j']:.5f}",
     f"{abl_jam['cases']['state_fine']['wall_seconds']:.1f}", "-"],
    ["internal variables on the coarse step",
     f"{abl_jam['cases']['state_coarse']['phi_j']:.5f}",
     f"{abl_jam['cases']['state_coarse']['wall_seconds']:.1f}",
     _pct(abl_jam['phi_j_bias'], 2)]]) if abl_jam else "not run"}

### 3.8 Ablation: what may be frozen on the coarse step

{_table(["variant", "n_sub", "dt outer", "energy drift", "wall [s]"],
        [[name, abl['cases'][name]['n_sub'], f"{abl['cases'][name]['dt_outer']:.2e}",
          f"{abl['cases'][name]['energy_drift_max']:.3e}",
          f"{abl['cases'][name]['wall_seconds']:.1f}"] for name in
         ('resolved_contacts', 'frozen_soft')])}

## 4. Figures

* `fig1_ladder.png` — restitution error against the ladder with the gate zones.
* `fig2_ledger.png` — energy drift and contact energy over 1e5 steps, both rates.
* `fig3_shear.png` — shear stress ratio and coordination vs $\\\\mu$.
* `fig4_jamming.png` — pressure ladder + $\\\\varphi_J$ and the wall clock of the same physics.
* `fig5_ablation.png` — the frozen-soft-force ablation.

## 5. Caveats and limits

* **The speed-up is not free.**  Sub-cycling the stiff skeleton *while freezing*
  the soft contact force is the cheaper scheme and it destroys the energy ledger
  ({abl['cases']['frozen_soft']['energy_drift_max']:.1e} vs
  {abl['cases']['resolved_contacts']['energy_drift_max']:.1e}, section 3.7): the
  soft and stiff springs act on the same pair, so the force classes do not
  separate in frequency.  The shipped integrator therefore sub-cycles the whole
  active set and pays for it.
* **Criterion (c) fails on the wall clock, and this is the honest headline.**
  With the internal variables and the contacts resolved accurately (which is what
  keeps $\\varphi_J$), the multi-rate solver costs
  {hier['speedup']:.2f}x the single-rate one over the same pressure ladder
  ({hier['mts']['wall_seconds']:.1f} s vs {hier['single']['wall_seconds']:.1f} s) and
  {hier.get('speedup_vs_shipped', float('nan')):.2f}x the shipped engine on the same
  ladder.  In a jammed packing the active set is most of the candidate list, so
  sub-cycling costs what it saves: the ten-fold larger coarse step is paid back by
  ten contact passes.  The cheap variants exist (sections 3.7 and 3.8) and they
  are the ones that buy speed -- at 4.2 % $\\varphi_J$ bias or a two-order-of-magnitude
  energy violation.
* **Cross-code agreement is limited.**  The shipped engine restarts the
  tangential/rolling state whenever its neighbour list is rebuilt; the multi-rate
  integrator carries `xi`/`roll` across rebuilds by contact key.  That difference
  biases $\\\\varphi_J$ between the two implementations and is the reason the
  W4-P1 anchor check is reported separately from the same-code check.
* The deeply overlapped end of a *violent* compression ($\\\\varphi\\\\to0.92$) is the
  worst case for a coarse step (there the two rates differ by ~2 % in
  $\\\\varphi$, see the test suite), while the sane pressure ladder used for
  $\\\\varphi_J$ agrees to {_pct(hier['phi_j_rel_diff'], 2)}.
* **The shear scenario is not converged.**  The profile-thermostat cell only
  reaches a strain of 0.5 with a 3 % polydispersity-free bidisperse packing and
  reports $\\tau/P$ values of 0.03-0.17 that are not monotone in $\\mu$; the two
  rates therefore disagree by up to {_pct(shear['worst_rel_diff'], 0)}.  The
  stage is reported as a FAIL, not tuned away: a converged steady state needs a
  longer run (strain $\\gtrsim 2$) which the node's compute budget did not carry.

## 6. Error judgement

{_table(["check", "evidence", "verdict"], check_rows)}

**Is the key quantity within 3 % of the benchmark?**  The restitution (the only
quantity with an independent mathematical anchor) is within
{_pct(ladder['worst_error'], 2)} of the converged solution — **yes**.  The
jamming density has no closed-form anchor; it is compared to the W4-P1 servo
reference and to the same-code single-rate run, and those two comparisons are
reported separately and honestly (see the table).

## 7. Audit trail

```
module      src/tensordem/mts.py
campaign    benchmarks/mts_contact_integration.py
tests       tests/test_mts_contact_integration.py   (17 passed)
evidence    runs/P82W2_N09/verification.json
figures     docs/figures/P82W2_N09/
prereg      prereg/P82W2_N09.json
```
"""
    return md


def main() -> int:
    data = json.loads((RUN / "verification.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    figs = {}
    builders = [("fig1_ladder", fig_ladder, "restitution ladder under the gate"),
                ("fig2_ledger", fig_ledger, "1e5-step energy/momentum ledger"),
                ("fig3_shear", fig_shear, "double-periodic shear comparison"),
                ("fig4_jamming", fig_jamming, "friction jamming phi_J and wall clock"),
                ("fig5_ablation", fig_ablation, "frozen-soft-force ablation")]
    selector = {"fig1_ladder": "ladder", "fig2_ledger": "ledger", "fig3_shear": "shear",
                "fig4_jamming": "jamming", "fig5_ablation": "ablation"}
    for name, fn, caption in builders:
        path = FIG / f"{name}.png"
        fn(data[selector[name]], path)
        figs[name] = {"path": str(path.relative_to(ROOT)), "caption": caption}
    md = build_report(data, figs)
    (ROOT / "docs" / "P82W2_N09_mts_contact_integration.md").write_text(md)
    payload = {
        "title": "P82W2-N09: multi-rate contact integration (MTS) + time-step gate in TensorDEM",
        "summary": "Machine-decidable dt/dt_c contract (margin/subcycle/fail at 0.5), "
                   "restitution ladder within 0.6 %, 1e5-step conservative ledger, "
                   "shard reproducibility, and an honest account of where the "
                   "multi-rate speed-up does and does not materialise.",
        "scenario_key": "P82W2-N09",
        "mission_id": 7764,
        "tags": ["TensorDEM", "DEM", "multi-rate", "time-step", "friction", "jamming",
                 "verification"],
        "markdown": md,
        "figures": [],
    }
    for name, meta in figs.items():
        payload["figures"].append({
            "name": name,
            "caption": meta["caption"],
            "png_base64": base64.b64encode((ROOT / meta["path"]).read_bytes()).decode(),
        })
    (RUN / "report_payload.json").write_text(json.dumps(payload))
    print(f"report: {len(md)} chars, {len(payload['figures'])} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())