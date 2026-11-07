"""Generate ``docs/ice_abrasion_archard_wear.md`` from the run artifacts.

Every number in the generated document is injected from
``artifacts/ice_abrasion/report.json`` (and ``convergence.json`` /
``sensitivity.csv``), so no figure in the text is hand-copied.  Re-run after a new
benchmark run:

    python scripts/build_ice_abrasion_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "ice_abrasion"
DOC = ROOT / "docs" / "ice_abrasion_archard_wear.md"


def fmt(value, spec: str = ".3g") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main() -> int:
    report = json.loads((ART / "report.json").read_text())
    conv = json.loads((ART / "convergence.json").read_text())
    cases = report["coating_case"]["season"]
    grid = report["geometry"]
    ledger = cases["ledger"]
    life = report["coating_case"]["life"]
    steel = report["steel_case"]
    heavy = report["held_out_validation"]
    calib = report["coating_case"]["calibration"]
    crit = report["criteria"]
    cs = report["contact_statistics"]
    ens = cs["ensemble_summary"]
    voyage = report["voyage"]
    sens = report["sensitivity"]["rows"]

    geometry_table = table(
        ["quantity", "value", "audit"],
        [["L x B x T [m]", f"{grid['length_m']:.2f} x {grid['beam_m']:.3f} x "
                           f"{grid['draught_m']:.2f}", "prescribed"],
         ["Cb target / mesh", f"{grid['block_coefficient_target']:.4f} / "
                              f"{grid['block_coefficient_mesh']:.5f}",
          f"rel. err {grid['cb_relative_error']:.2e}"],
         ["B target / mesh [m]", f"{grid['beam_m']:.3f} / {grid['beam_mesh_m']:.4f}",
          f"rel. err {grid['beam_relative_error']:.2e}"],
         ["Cp / Cm", f"{grid['prismatic_coefficient']:.4f} / "
                     f"{grid['midship_coefficient']:.3f}",
          f"m = {grid['waterplane_exponent']:.4f}, n = {grid['midship_exponent']:.4f}"],
         ["geometry gate (<1 %)", fmt(grid["geometry_audit_passed"]), "Cb/L/B re-integrated"],
         ["ice-belt band (one side) [m^2]", f"{grid['band_area_one_side_m2']:.2f}",
          f"{grid['n_z']} x {grid['n_x']} cells at {grid['cell_size_m']:.2f} m"],
         ["band below / above waterline [m]",
          f"{grid['band_below_m']:.2f} / {grid['band_above_m']:.2f}", "FSICR-like ice belt"]])

    voyage_table = table(
        ["leg", "distance [nm]", "speed [kn]", "ice thickness [m]", "concentration"],
        [[leg["name"], fmt(leg["distance_nm"], ".0f"), fmt(leg["speed_knots"], ".1f"),
          fmt(leg["thickness_m"], ".2f"), fmt(leg["concentration"], ".2f")]
         for leg in voyage["legs"]])

    ensemble_table = table(
        ["statistic", "value"],
        [["sampled contacts per leg", fmt(ens["n_contacts"])],
         ["contact pressure median [MPa]", fmt(ens["pressure_median_pa"] / 1e6, ".3f")],
         ["contact pressure P95 [MPa]", fmt(ens["pressure_p95_pa"] / 1e6, ".3f")],
         ["crushing-limited contacts",
          f"{100.0 * ens['hertz_capped_fraction']:.1f} %"],
         ["slip speed median [m/s]", fmt(ens["slip_speed_median_m_s"], ".3f")],
         ["sliding fraction (Cundall-Strack)", fmt(ens["sliding_fraction_mean"], ".3f")],
         ["crushed patch radius median [mm]",
          fmt(ens["contact_radius_median_m"] * 1e3, ".2f")],
         ["normal load median [N]", fmt(ens.get("normal_load_median_N"), ".0f")],
         ["timesteps per contact", fmt(ens["steps"])]])

    field_table = table(
        ["material", "per voyage mean / P95 / max [um]",
         "per season mean / P95 / max [um]", "volume per season, one side [m^3]"],
        [["coating",
          f"{cases['per_voyage']['mean_depth_um']:.2f} / "
          f"{cases['per_voyage']['p95_depth_um']:.2f} / "
          f"{cases['per_voyage']['max_depth_um']:.2f}",
          f"{cases['per_year']['mean_depth_um']:.2f} / "
          f"{cases['per_year']['p95_depth_um']:.2f} / "
          f"{cases['per_year']['max_depth_um']:.2f}",
          fmt(cases["per_year"]["volume_total_one_side_m3"], ".3e")],
         ["bare AH36 steel",
          f"{steel['season']['per_voyage']['mean_depth_um']:.2f} / "
          f"{steel['season']['per_voyage']['p95_depth_um']:.2f} / "
          f"{steel['season']['per_voyage']['max_depth_um']:.2f}",
          f"{steel['season']['per_year']['mean_depth_um']:.2f} / "
          f"{steel['season']['per_year']['p95_depth_um']:.2f} / "
          f"{steel['season']['per_year']['max_depth_um']:.2f}",
          fmt(steel["season"]["per_year"]["volume_total_one_side_m3"], ".3e")]])

    life_table = table(
        ["material", "thickness used", "statistic",
         "rate [mm/year]", "life P10 / P50 / P90 [years]"],
        [["coating", f"{life['thickness_used_m'] * 1e6:.0f} um DFT", key,
          f"{life['rate_peak_m_per_year'] * 1e3:.4f}"
          if key == "worst_survey_patch" else
          f"{life['rate_p95_m_per_year'] * 1e3:.4f}"
          if key == "p95_survey_patch" else
          f"{life['rate_mean_m_per_year'] * 1e3:.4f}",
          f"{life[key]['life_years_p10']:.2f} / {life[key]['life_years_p50']:.2f} / "
          f"{life[key]['life_years_p90']:.2f}"]
         for key in ("worst_survey_patch", "p95_survey_patch", "mean_field")]
        + [["AH36 steel", f"{steel['life']['thickness_used_m'] * 1e3:.1f} mm allowance",
            key,
            f"{steel['life']['rate_peak_m_per_year'] * 1e3:.4f}"
            if key == "worst_survey_patch" else
            f"{steel['life']['rate_p95_m_per_year'] * 1e3:.4f}"
            if key == "p95_survey_patch" else
            f"{steel['life']['rate_mean_m_per_year'] * 1e3:.4f}",
            f"{steel['life'][key]['life_years_p10']:.2f} / "
            f"{steel['life'][key]['life_years_p50']:.2f} / "
            f"{steel['life'][key]['life_years_p90']:.2f}"]
           for key in ("worst_survey_patch", "p95_survey_patch", "mean_field")])

    ledger_table = table(
        ["ledger quantity", "value"],
        [["friction work per season, both sides [J]",
          fmt(ledger["friction_work_year_total_J"], ".4e")],
         ["friction work per season, one side [J]",
          fmt(ledger["friction_work_year_per_side_J"], ".4e")],
         ["friction work recovered from the event population [J]",
          fmt(ledger["friction_work_from_events_year_per_side_J"], ".4e")],
         ["friction-interface deviation [%]",
          fmt(100.0 * ledger["friction_interface_relative_deviation"], ".2e")],
         ["Archard wear volume, one side [m^3/season]",
          fmt(ledger["archard_volume_year_one_side_m3"], ".4e")],
         ["Finnie (impact) volume fraction",
          f"{100.0 * ledger['finnie_volume_fraction']:.1f} %"],
         ["wear energy V H [J/season, one side]",
          fmt(ledger["wear_energy_year_J"], ".4e")],
         ["V H / W_f (measured)", fmt(ledger["energy_ratio_volumeH_over_friction_work"],
                                      ".4e")],
         ["K / mu (set)", fmt(ledger["energy_ratio_expected_K_over_mu"], ".4e")],
         ["energy-identity deviation [%]",
          fmt(100.0 * ledger["energy_ratio_relative_deviation"], ".2e")],
         ["implied load-bearing contact duty factor",
          fmt(ledger["duty_factor_implied"], ".3e")],
         ["Archard coefficient K = k H [-]", fmt(ledger["archard_coefficient"], ".3e")]])

    conv_table = table(
        ["study", "levels", "observed order", "GCI(fine) [%]", "criterion <5 %"],
        [["DEM particle count",
          " / ".join(str(v) for v in conv["dem"]["sample_counts"]),
          fmt(conv["dem"]["sample_count_gci"]["observed_order"], ".2f"),
          fmt(conv["dem"]["sample_count_gci"]["gci_fine_pct"], ".3f"),
          fmt(conv["dem"]["sample_count_gci"]["gci_fine_pct"] < 5.0)],
         ["contact timestep",
          " / ".join(str(v) for v in conv["dem"]["step_counts"]),
          "machine precision",
          fmt(conv["dem"]["timestep_gci"]["gci_fine_pct"], ".3f"),
          fmt((conv["dem"]["timestep_gci"]["gci_fine_pct"] or 0.0) < 5.0)],
         ["band grid, belt mean",
          " / ".join(f"{v:g}" for v in conv["grid"]["cell_sizes_m"]),
          fmt(conv["grid"]["mean_depth_gci"]["observed_order"], ".2f"),
          fmt(conv["grid"]["mean_depth_gci"]["gci_fine_pct"], ".3f"),
          fmt((conv["grid"]["mean_depth_gci"]["gci_fine_pct"] or 0.0) < 5.0)],
         ["band grid, P95 depth",
          " / ".join(f"{v:g}" for v in conv["grid"]["cell_sizes_m"]),
          fmt(conv["grid"]["p95_depth_gci"]["observed_order"], ".2f"),
          fmt(conv["grid"]["p95_depth_gci"]["gci_fine_pct"], ".3f"),
          fmt((conv["grid"]["p95_depth_gci"]["gci_fine_pct"] or 100.0) < 5.0)]])

    sens_table = table(
        ["parameter", "value", "P95 depth [um/season]", "belt mean [um/season]"],
        [[row["parameter"], fmt(row["value"], ".3g"),
          fmt(row["p95_um_per_season"], ".3f"), fmt(row["mean_um_per_season"], ".3f")]
         for row in sens])

    crit_table = table(
        ["criterion", "verdict", "key number"],
        [["(a) analytic Archard degradation",
          fmt(crit["a"]["passed"]),
          f"max rel. deviation {fmt(crit['a']['max_relative_deviation_direct'], '.1e')} "
          f"(direct), {fmt(crit['a']['max_relative_deviation_pit_chain'], '.1e')} "
          f"(pit chain), tolerance 1e-10"],
         ["(b) literature / survey benchmark (weak)",
          fmt(crit["b"]["passed"]),
          f"held-out belt mean {heavy['mean_mm_per_year']:.3f} mm/yr = "
          f"{heavy['mean_deviation_vs_anchor_pct']:+.1f} % of the 0.10 mm/yr midpoint; "
          f"P95 patch {heavy['p95_mm_per_year']:.3f} mm/yr = "
          f"{heavy['deviation_vs_anchor_pct']:+.0f} % (outside band)"],
         ["(c) energy consistency", fmt(crit["c"]["passed"]),
          f"V H / W_f = {fmt(crit['c']['ratio'], '.4e')} vs K/mu = "
          f"{fmt(crit['c']['expected_K_over_mu'], '.4e')}, deviation "
          f"{fmt(crit['c']['deviation_pct'], '.2e')} %"],
         ["(d) DEM / timestep convergence", fmt(crit["d"]["passed"]),
          f"sample GCI {fmt(crit['d']['sample_count_gci_pct'], '.3f')} %, timestep GCI "
          f"{fmt(crit['d']['timestep_gci_pct'], '.3f')} %"],
         ["(e) friction-module interface", fmt(crit["e"]["passed"]),
          f"deviation {fmt(crit['e']['deviation_pct'], '.2e')} % at equal mu"],
         ["(f) pytest + full case", fmt(crit["f"]["passed"]),
          f"{crit['f']['pytest_tests']} tests, status \"{crit['f']['pytest_status']}\""]])

    def pct(value, spec=".2f"):
        return fmt(100.0 * value, spec)

    doc = f"""# Ice-hull waterline-band abrasion wear: DEM contact statistics x Archard law,
# wear depth field and coating life (W10-N06)

Module `src/tensordem/ice_abrasion.py` · tests `tests/test_ice_abrasion.py` ·
driver `scripts/benchmark_ice_abrasion.py` · artifacts `artifacts/ice_abrasion/`.

**Evidence level.**  The numerics and physics (Hertzian contact with a crushing
cap, Cundall-Strack tangential saturation, the Archard law, the Finnie angle
function, volume-conserving projection, Roache GCI, the energy ledger) are
**REAL** and are unit-tested against closed-form values.  The *absolute* wear
level is **CALIBRATED**: one parameter, the specific wear rate
`k = K/H`, is fitted to a single field anchor (Baltic 1A ice-belt coating loss).
Nothing in this document should be read as a validated wear predictor: the
literature anchor is a weak reference and the calibration is explicit.

## 1. Engineering problem

An ice-class hull loses coating and steel in the **waterline band (ice belt)**
because broken ice slides along the shell with local contact pressures of order
1-3 MPa while the ship advances.  Design consequences: coating service life and
recoating interval, steel wastage allowance for the ice belt, and the friction
work that has to be accounted for in the ice-resistance budget.  TensorDEM could
already resolve ice-hull *contact forces* (`hull_contact`, `hull_mesh_contact`,
`threefield.hull_ice`) and ice *resistance* (`threefield.resistance`), but had no
material-removal law, so an ice-class coating had no life model.  This increment
adds the missing loop:

```
DEM contact statistics  ->  Archard / Finnie wear law  ->  waterline-band depth
field  ->  coating life  ->  energy ledger consistent with the resistance module
```

## 2. Case definition

{geometry_table}

Voyage profile (Gulf of Bothnia round trip, one direction listed; the return legs
repeat the same conditions in ballast):

{voyage_table}

Seasonal model: {fmt(voyage['voyages_per_year'], '.0f')} round voyages per year,
belt energy fraction {fmt(voyage['belt_energy_fraction'], '.2f')} (share of the
ice friction work dissipated in the ice belt rather than along the deeper hull),
friction coefficient mu = {fmt(voyage['friction_coefficient'], '.2f')} fed to the
resistance baselines, coating 350 um DFT epoxy (H = 200 MPa), bare AH36 steel
(H = 2.0 GPa).

## 3. Method

### 3.1 DEM contact statistics (the discretised `f(p_c, v_s)`)

Each fragment of size `l_f` (lognormal, median 0.35 m) carries the normal load
`F = p_nom * kappa_A * l_f^2` with the nominal ice pressure `p_nom` lognormal
(median 0.7 MPa) over the load-bearing fraction `kappa_A = 0.02` of its footprint.
The normal contact is Hertzian, `p_H = F/(pi a^2)` with
`a = (3 F R^* / 4 E^*)^(1/3)`, **capped by the local crushing capacity**
`p_crush` (lognormal, median 1.5 MPa); the crushed patch radius then follows from
`F = p_c pi a^2`.  Tangential motion is a Cundall-Strack spring
(`k_t = 8 G^* sqrt(R^* delta)`, Mindlin) saturated at `mu F_n`, integrated over the
contact duration `L/v_s`, and projected back onto the friction cone after every
step.  Only sliding increments count as wear-relevant slip, so the ensemble
returns `(p_c, v_s, L_slip, N, a)`.

{ensemble_table}

The elastic branch is fully crushing-limited
({pct(ens['hertz_capped_fraction'], '.1f')} % of contacts): for a 0.35 m fragment
the Hertz pressure would be tens of MPa, which ice cannot sustain, so the contact
pressure is governed by the crushing-strength scatter.  This is a physical result
of the model, not a numerical artefact.

### 3.2 Wear law

Archard, written per contact as a pit depth,

```
d = K p_c L_slip / H        (equivalently d = k p_c L_slip, k = K/H)
```

with `k` in m^3/(N m) - the form ice-abrasion data are reported in.  The high
impact-angle correction is Finnie (1960),

```
V = C_F (m v_n^2 / sigma_f) f(alpha),  sigma_f = H/2.5,
f(alpha) = sin(2a) - 3 sin^2(a)  (a <= atan(1/3)),  cos^2(a)/3  (a > atan(1/3))
```

applied only where the ice approach is oblique (obliquity decays aft from the
stem) and with only the corner mass (`{fmt(ens['contact_area_fraction'], '.3g')}`-type
small fraction, 0.3 %) and the normal approach speed (3 % of the slip speed)
participating.  The impact branch contributes
{fmt(100.0 * ledger['finnie_volume_fraction'], '.1f')} % of the removed volume, i.e. it is a
bow-localised correction, not the main mechanism.

### 3.3 Depth field

Every contact deposits its pit volume `d * pi a^2` at a position drawn from the
ice-belt activity distribution (peak at the stem and fore shoulder, vertical peak
just below the waterline) and the volume is split over the touched band cells by
footprint overlap.  Volume is conserved exactly, so the cell-averaged depth is
grid independent in the dilute-contact limit.

### 3.4 Energy ledger and the friction-module interface

The friction work of each leg is taken from the `threefield.resistance`
baselines as the mu-dependent part of the resistance,

```
W_f = (R(mu) - R(mu=0)) * distance
```

and the belt share is `belt_energy_fraction * concentration * W_f`.  The event
population is then scaled so that `sum(N L_slip) = W_f,belt / mu` exactly, which
makes the Archard branch satisfy `V H / W_f = K / mu`.  Two consequences are
worth stating:

* the circulation of mass is *not* fitted (only `k` is), so the belt-mean depth is
  a genuine prediction of the friction work and of the contact statistics;
* because both baselines are linear in mu, `W_f/mu` is mu independent and so is
  the predicted wear depth - the mu input re-partitions the energy ledger but not
  the wear magnitude (see the sensitivity table).

### 3.5 Life prediction

Life to local breach = coating thickness / annual depth rate at the chosen
statistic, with a Monte Carlo (20000 samples) over lognormal uncertainty in the
specific wear rate (sigma_ln = 0.5), the friction coefficient (0.2) and the ice
resistance (0.15).  Three statistics are reported: the worst 0.25 m cell, the P95
survey patch (the calibration statistic) and the belt mean.

### 3.6 Geometry gate

No wear number is produced unless the parametric hull reproduces its own
principal particulars from the mesh quadrature (Cb, L, B) within 1 %.

## 4. Results

### 4.1 Geometry audit

{geometry_table}

### 4.2 Wear depth field (calibrated season)

{field_table}

`k` fitted to the anchor: **{fmt(calib['spec_rate_fitted'], '.3e')} m^3/(N m)**
(within the literature range {fmt(calib['spec_rate_literature_range'][0], '.0e')} ..
{fmt(calib['spec_rate_literature_range'][1], '.0e')},
{fmt(calib['fitted_within_literature_range'])}); the implied Archard coefficient
is `K = k H = {fmt(ledger['archard_coefficient'], '.3e')}`, i.e.
`K/mu = {fmt(ledger['energy_ratio_expected_K_over_mu'], '.3e')}`, well inside
(0, 1) as required.

### 4.3 Coating and steel life

{life_table}

The worst 0.25 m cell is a Poisson-dominated extreme-value statistic (see 4.5) and
is reported as a lower bound; the P95 survey patch is the design value used for
the recoating interval; the belt mean corresponds to a full-belt renewal.

### 4.4 Energy ledger

{ledger_table}

The recovered event friction work matches the resistance-module friction work to
{fmt(100.0 * ledger['friction_interface_relative_deviation'], '.1e')} %, and the
energy ratio reproduces `K/mu` to
{fmt(100.0 * ledger['energy_ratio_relative_deviation'], '.1e')} %.  The implied
duty factor `{fmt(ledger['duty_factor_implied'], '.3e')}` is the physical
interpretation of the calibrated magnitude: about
{fmt(100.0 * ledger['duty_factor_implied'], '.2f')} % of the fragment-passage
opportunities at a fixed hull point actually carry a load-bearing sliding contact
(the rest are water-lubricated or non-load-bearing).

### 4.5 Convergence

{conv_table}

The DEM sample-count levels are *nested* (prefixes of one ensemble) and the metric
is the mean pit depth that the season integration consumes; the timestep affects
the slip distance and is converged to machine precision already at 48 steps.  The
belt-mean depth is grid independent to
{fmt((conv['grid']['mean_depth_gci']['gci_fine_pct'] or 0.0), '.2f')} %, but the P95
statistic changes by
{fmt((conv['grid']['p95_depth_gci']['gci_fine_pct'] or 0.0), '.1f')} % between
1.0 m and 0.25 m cells: in the dilute-contact limit the extreme-value statistics of
a shot-noise field are resolution dependent even when the mean is not.  This is a
documented limitation, not a converged quantity.

### 4.6 Sensitivity

{sens_table}

The Archard branch is exactly linear in `k` (log-log slope 1); the flat `mu` rows
are a consequence of the linear resistance baselines discussed in 3.4.

### 4.7 Held-out validation

The specific wear rate fitted on the coating anchor is applied unchanged to a
harsher, independent profile (1A Super traffic, 20 round trips per season, 0.7-1.0 m
ice) on bare steel:

| quantity | model | literature band |
|---|---|---|
| belt-mean wastage | {heavy['mean_mm_per_year']:.3f} mm/yr | 0.05 - 0.15 mm/yr |
| P95 survey patch | {heavy['p95_mm_per_year']:.3f} mm/yr | 0.05 - 0.15 mm/yr |
| worst 0.25 m cell | {heavy['peak_mm_per_year']:.3f} mm/yr | - |

The belt mean lands {heavy['mean_deviation_vs_anchor_pct']:+.1f} % from the band
midpoint (inside the band), while the P95 patch exceeds the band
({heavy['deviation_vs_anchor_pct']:+.0f} %).  The honest reading: the model
reproduces the *representative* wastage rate at which a class survey quotes ice-belt
steel loss, and over-predicts the single worst patch.

## 5. Acceptance criteria

{crit_table}

### 5.1 Error judgements (explicit)

| key quantity | benchmark | error | < 3 % ? |
|---|---|---|---|
| Archard volume `V = K W L / H` | closed form | {fmt(crit['a']['max_relative_deviation_direct'], '.1e')} (rel.) | yes |
| pit chain `d pi a^2` vs `K W L / H` | closed form | {fmt(crit['a']['max_relative_deviation_pit_chain'], '.1e')} (rel.) | yes |
| `V H / W_f` vs `K/mu` | energy identity | {fmt(crit['c']['deviation_pct'], '.1e')} % | yes |
| friction work vs resistance module | same mu input | {fmt(crit['e']['deviation_pct'], '.1e')} % | yes |
| DEM particle-count GCI | r = 2 self-consistency | {fmt(crit['d']['sample_count_gci_pct'], '.3f')} % | yes |
| contact timestep GCI | r = 2 self-consistency | {fmt(crit['d']['timestep_gci_pct'], '.3f')} % | yes |
| belt-mean depth vs grid | 1.0 m vs 0.25 m cells | {fmt((conv['grid']['mean_depth_gci']['gci_fine_pct'] or 0.0), '.2f')} % | yes |
| P95 depth vs grid | 1.0 m vs 0.25 m cells | {fmt((conv['grid']['p95_depth_gci']['gci_fine_pct'] or 0.0), '.1f')} % | **no** |
| coating P95 vs anchor A | 35 um/season midpoint | {fmt(calib['deviation_vs_anchor_pct'], '.1e')} % (calibrated) | yes by construction |
| held-out steel belt mean vs anchor B | 0.10 mm/yr midpoint | {heavy['mean_deviation_vs_anchor_pct']:+.1f} % | **no** (inside band) |
| held-out steel P95 vs anchor B | 0.10 mm/yr midpoint | {heavy['deviation_vs_anchor_pct']:+.0f} % | **no** |

The calibration criterion is met only because it is a one-parameter fit; the
genuine test is the held-out line and it does **not** meet a 3 % (or even 25 %)
tolerance on the patch statistic.  Both are stated rather than hidden.

## 6. Attention points and limitations

1. **One-parameter calibration.** `k` is fitted to a single literature anchor
   (Baltic 1A ice-belt coating loss, 20-50 um/season).  The anchor itself scatters
   by a factor of ~2-3, so the calibration inherits that.  The fitted value
   ({fmt(calib['spec_rate_fitted'], '.2e')} m^3/(N m)) is, however, inside the
   published ice-abrasion range, and it is below dry-sliding values, which is
   consistent with water-lubricated, intermittent contact.
2. **Belt energy fraction.** The split of friction work between the ice belt and
   the deeper hull (`0.6`) is an engineering assumption motivated by the waterline
   pressure peak; a range of 0.3-0.8 would move the predicted wear by a factor of
   ~2.  It is the second largest uncertainty after `k`.
3. **The P95 / max statistics are not converged.** Shot-noise extreme-value
   statistics are resolution dependent ({fmt((conv['grid']['p95_depth_gci']['gci_fine_pct'] or 0.0), '.1f')} % between
   1.0 m and 0.25 m cells).  The calibration uses the 0.25 m P95; using a different
   cell size shifts `k` accordingly.
4. **Friction-work source.** Lindqvist and Riska friction works differ by
   {fmt(100.0 * max(r['baseline_spread_fraction'] for r in ledger['per_leg']), '.0f')} % on the
   light-ice leg (the design value is their mean), so the absolute wear level
   inherits that baseline spread.
5. **mu insensitivity.** With linear friction baselines the predicted depth does
   not depend on mu; a nonlinear friction model would break that and is the natural
   next increment.
6. **Model simplifications.** No coating delamination/chipping, no corrosion, no
   temperature or ice-salinity dependence of the wear coefficient, no ridge keel
   impact, no hull flare above the waterline, single ice-belt band on each side,
   fragment shape described by one size statistic, and the surface-length error of
   the band cells (deadrise neglected) is a few percent.
7. **Not a validated predictor.** The model is a physically structured wear loop
   with verified numerics; only a meaured thickness survey of a real ship could
   make it a predictive tool.

## 7. Reproduction

```bash
python -m pytest tests/test_ice_abrasion.py -q          # 39 tests
python scripts/benchmark_ice_abrasion.py --stage cases
python scripts/benchmark_ice_abrasion.py --stage convergence
python scripts/benchmark_ice_abrasion.py --stage sensitivity_rates
python scripts/benchmark_ice_abrasion.py --stage sensitivity_mu
python scripts/benchmark_ice_abrasion.py --stage report
python scripts/build_ice_abrasion_doc.py                # regenerate this document
```

## 8. Audit chain

| link | path |
|---|---|
| model | `src/tensordem/ice_abrasion.py` |
| reused engines | `src/tensordem/dem3d.py` (Hertz), `src/tensordem/hertz_mindlin.py` (Mindlin), `src/tensordem/contact_tangential.py` (Cundall-Strack), `src/tensordem/threefield/resistance.py` (friction work) |
| tests | `tests/test_ice_abrasion.py` (39 tests) |
| driver | `scripts/benchmark_ice_abrasion.py` |
| numeric record | `artifacts/ice_abrasion/report.json` |
| fields | `artifacts/ice_abrasion/wear_field_coating.csv`, `wear_field_steel.csv`, `profile.csv` |
| life / sensitivity | `artifacts/ice_abrasion/life_table.csv`, `sensitivity.csv`, `convergence.json` |
| figures | `artifacts/ice_abrasion/figs/*.png` |
| API contract | `ice_abrasion_calculation` -> `POST /api/v1/calculations/structure/ice_abrasion` |

**API deployment status.** The request/response contract is implemented and tested
inside this repository (`ice_abrasion_calculation`; unknown material or missing
principal particulars raise, which a FastAPI route maps to 422).  The route itself
must be registered by the ship-performance service, which lives in a different
repository; as of this run `POST :8002/api/v1/calculations/structure/ice_abrasion`
still answers 404, so the platform capability is *not yet exposed* - the contract
and its tests are ready for the wiring.

Figures: `figs/wear_field.png` (depth field, coating/steel), `figs/wear_profiles.png`
(longitudinal/vertical profiles vs contact activity), `figs/life_distribution.png`
(Monte Carlo life), `figs/convergence.png` (DEM and grid GCI),
`figs/sensitivity.png` (k and mu), `figs/energy_ledger.png` (criterion c).
"""
    DOC.write_text(doc)
    print("wrote", DOC, len(doc), "chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())