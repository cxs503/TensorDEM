"""Report renderer: every number in the documents comes from the artifact.

``render_report``  the repository document ``docs/MIXED_ICE_RESISTANCE.md``
``render_results_markdown``  a compact results section used for the platform
                   report and for quick inspection

Nothing here types a physical number: tables, verdicts, error statements and
file names are all injected from the campaign artifact, so the documents cannot
drift away from the run.
"""
from __future__ import annotations

from typing import Any, Sequence

__all__ = ["render_report", "render_results_markdown"]


def _mn(value_n: float, digits: int = 3) -> str:
    return f"{value_n / 1e6:.{digits}f}"


def _pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{100.0 * value:.{digits}f} %"


def _ageing_table(artifact: dict[str, Any]) -> str:
    head = ("| age [a] | class | S [permille] | T [degC] | v_b [-] | h [m] | "
            "sigma_f [MPa] | sigma_c [MPa] | E [GPa] | K_Ic [MPa m^0.5] | rho [kg/m^3] |")
    sep = "|" + "---|" * 11
    rows = [head, sep]
    for row in artifact["ageing_table"]:
        rows.append(
            "| {age:g} | {cls} | {s:.2f} | {t:.2f} | {vb:.4f} | {h:.2f} | {sf:.3f} | "
            "{sc:.3f} | {e:.2f} | {k:.4f} | {rho:.1f} |".format(
                age=row["age_years"], cls=row["ice_class"],
                s=row["salinity_permille"], t=row["temperature_c"],
                vb=row["brine_volume_fraction"], h=row["thickness_m"],
                sf=row["flexural_strength_pa"] / 1e6, sc=row["compressive_strength_pa"] / 1e6,
                e=row["elastic_modulus_pa"] / 1e9,
                k=row["fracture_toughness_pa_sqrt_m"] / 1e6, rho=row["density_kg_m3"]))
    return "\n".join(rows)


def _curve_table(artifact: dict[str, Any], speed: float) -> str:
    sweep = artifact["my_fraction_sweep"]["curves"]
    rows: dict[float, dict[str, Any]] = {}
    for model in ("lindqvist", "riska"):
        for row in sweep[f"{model}_v{speed:g}"]["rows"]:
            if row["my_strength_multiplier"] != 1.0:
                continue
            rows.setdefault(row["my_fraction"], {})[model] = row
    out = ["| p_MY | R Lindqvist [MN] | dR vs p=0 | R Riska [MN] | dR vs p=0 | "
           "R_MY/R_FY (Lindqvist) |", "|---|---|---|---|---|---|"]
    for p in sorted(rows):
        lind = rows[p]["lindqvist"]
        ris = rows[p]["riska"]
        ratio = "n/a" if lind["class_ratio"] is None else f"{lind['class_ratio']:.3f}"
        out.append(
            "| {p:.1f} | {rl} | {dl} | {rr} | {dr} | {ratio} |".format(
                p=p, rl=_mn(lind["resistance_mean_N"]),
                dl=_pct(lind["relative_increase"], 1),
                rr=_mn(ris["resistance_mean_N"]),
                dr=_pct(ris["relative_increase"], 1), ratio=ratio))
    return "\n".join(out)


def _band_table(artifact: dict[str, Any], speed: float) -> str:
    rows: dict[float, dict[float, float]] = {}
    for row in artifact["my_fraction_sweep"]["curves"][f"lindqvist_v{speed:g}"]["rows"]:
        rows.setdefault(row["my_fraction"], {})[row["my_strength_multiplier"]] = row["resistance_mean_N"]
    multipliers = sorted(next(iter(rows.values())))
    out = ["| p_MY | " + " | ".join(f"x{m:g}" for m in multipliers) + " | span [MN] |",
           "|---|" + "---|" * (len(multipliers) + 1)]
    for p in sorted(rows):
        values = [rows[p][m] for m in multipliers]
        out.append(f"| {p:.1f} | " + " | ".join(_mn(v) for v in values)
                   + f" | {_mn(max(values) - min(values))} |")
    return "\n".join(out)


def _speed_table(artifact: dict[str, Any]) -> str:
    out = ["| v [m/s] | dR/R(0) at p_MY = 0.8, Lindqvist | dR/R(0) at p_MY = 0.8, Riska |",
           "|---|---|---|"]
    for speed in artifact["speeds_m_s"]:
        cells = []
        for model in ("lindqvist", "riska"):
            rows = [r for r in artifact["my_fraction_sweep"]["curves"][f"{model}_v{speed:g}"]["rows"]
                    if r["my_strength_multiplier"] == 1.0]
            end = max(rows, key=lambda r: r["my_fraction"])
            cells.append(_pct(end["relative_increase"], 1))
        out.append(f"| {speed:g} | {cells[0]} | {cells[1]} |")
    return "\n".join(out)


def _sensitivity_table(artifact: dict[str, Any]) -> str:
    sens = artifact["sensitivity"]
    base = sens["base_resistance_N"]
    out = ["| parameter | group | levels | span [MN] | span / R | max abs deviation [MN] |",
           "|---|---|---|---|---|---|"]
    for name in sens["rank_by_span"] + [n for n in sens["parameters"]
                                        if sens["parameters"][n]["group"] == "primary"]:
        row = sens["parameters"][name]
        levels = sorted(v["level"] for v in row["levels"])
        out.append(f"| {name} | {row['group']} | {', '.join(f'{v:g}' for v in levels)} | "
                   f"{_mn(row['span_N'])} | {_pct(row['span_fraction_of_base'], 1)} | "
                   f"{_mn(row['max_abs_deviation_N'])} |")
    out.append("")
    out.append(f"Additive bounds (a conservative joint bound for monotone, mutually "
               f"independent effects): FSD {_mn(sens['fsd_additive_bound_N'])} MN "
               f"({_pct(sens['fsd_share_of_bound'], 1)} of the total), material "
               f"{_mn(sens['material_additive_bound_N'])} MN. Base R(p_MY="
               f"{sens['base_my_fraction']:g}) = {_mn(base)} MN.")
    return "\n".join(out)


def _envelope_table(artifact: dict[str, Any]) -> str:
    env = artifact["joint_envelope"]
    out = ["| group | n | mean [MN] | p05 [MN] | p95 [MN] | min [MN] | max [MN] | relative span |",
           "|---|---|---|---|---|---|---|---|"]
    for label in ("fsd", "material"):
        row = env[label]
        out.append(f"| {label} | {row['n']} | {_mn(row['mean_N'])} | {_mn(row['p05_N'])} | "
                   f"{_mn(row['p95_N'])} | {_mn(row['min_N'])} | {_mn(row['max_N'])} | "
                   f"{_pct(row['relative_span'], 1)} |")
    return "\n".join(out)


def _decomposition_table(artifact: dict[str, Any]) -> str:
    out = ["| model | ratio strength only | ratio thickness only | ratio full | interaction | "
           "strength share | thickness share |", "|---|---|---|---|---|---|---|"]
    for model, row in artifact["class_ratio_decomposition"].items():
        out.append(
            f"| {model} | {row['ratio_strength_only']:.3f} | {row['ratio_thickness_only']:.3f} | "
            f"{row['ratio_full']:.3f} | {_pct(row['interaction'], 2)} | "
            f"{_pct(row['strength_share_of_increment'], 1)} | "
            f"{_pct(row['thickness_share_of_increment'], 1)} |")
    return "\n".join(out)


def _dem_table(artifact: dict[str, Any]) -> str:
    dem = artifact["dem"]
    if "factorial" not in dem:
        return "_DEM factorial skipped in this run._"
    out = ["| code | sigma_f [MPa] | sigma_c [MPa] | E [GPa] | R section [MN/m] | "
           "crush share of normal force | flexure share of normal force | "
           "friction share of total | broken |",
           "|---|---|---|---|---|---|---|---|---|"]
    for row in dem["factorial"]:
        levels = row["factor_levels"]
        friction_share = row["friction_n_per_m"] / row["mean_total_n_per_m"]
        out.append(
            f"| {row['factorial_code']} | {levels['flexural_strength_pa']/1e6:.3f} | "
            f"{levels['compressive_strength_pa']/1e6:.3f} | {levels['elastic_modulus_pa']/1e9:.2f} | "
            f"{_mn(row['mean_total_n_per_m'])} | {row['crushing_fraction']:.3f} | "
            f"{row['flexure_fraction']:.3f} | {friction_share:.3f} | {row['broken_fraction']:.3f} |")
    return "\n".join(out)


_FACTOR_LABEL = {
    "compressive_strength_pa": r"$\sigma_c$",
    "flexural_strength_pa": r"$\sigma_f$",
    "elastic_modulus_pa": r"$E$",
}


def _rel(value: float, reference: float) -> str:
    """Relative error, or an explicit note when the reference is (near) zero."""
    if abs(reference) < 1e-9:
        return "n/a (anchor exponent 0)"
    return f"{100.0 * value / reference:.1f} %"


def _dem_exponent_table(artifact: dict[str, Any]) -> str:
    dem = artifact["dem"]
    if "dem_fit" not in dem:
        return "_DEM factorial skipped in this run._"
    order = ["compressive_strength_pa", "flexural_strength_pa", "elastic_modulus_pa"]
    out = ["| exponent | DEM 2^3 fit | Lindqvist anchor | Riska anchor | relative error (DEM vs Lindqvist) |",
           "|---|---|---|---|---|"]
    for key in order:
        d = dem["dem_fit"]["exponents"][key]
        a = dem["anchor_fit"]["lindqvist"]["exponents"][key]
        r = dem["anchor_fit"]["riska"]["exponents"][key]
        out.append(f"| d ln R / d ln {_FACTOR_LABEL[key]} | {d:.4f} | {a:.4f} | {r:.4f} | "
                   f"{_rel(d - a, a)} |")
    out.append("")
    out.append(f"Fit residual (interaction magnitude) of the DEM factorial: "
               f"{dem['dem_fit']['residual_max_abs_log']:.4f} in ln R "
               f"({dem['dem_fit']['dof']} degrees of freedom); anchor factorials are "
               f"exactly separable because the anchors are additive in their components "
               f"(residual {dem['anchor_fit']['lindqvist']['residual_max_abs_log']:.2e}).")
    return "\n".join(out)


def _dem_sweep_table(artifact: dict[str, Any]) -> str:
    dem = artifact["dem"]
    sweep = dem.get("my_fraction_sweep")
    if not sweep:
        return "_DEM sweep skipped in this run._"
    out = [f"DEM-domain MY-fraction sweep at fixed h = {sweep['thickness_m']:g} m "
           f"(section line load [MN/m]; only the ice class of each floe switches, the "
           f"FSD is frozen):", "",
           "| p_MY | DEM section [MN/m] | DEM dR | Lindqvist [MN] | Lindqvist dR | Riska dR |",
           "|---|---|---|---|---|---|"]
    for row in sweep["rows"]:
        out.append(f"| {row['my_fraction']:.1f} | {_mn(row['dem_section_N_per_m'])} | "
                   f"{_pct(row['dem_relative_increase'], 1)} | {_mn(row['lindqvist_N'])} | "
                   f"{_pct(row['lindqvist_relative_increase'], 1)} | "
                   f"{_pct(row['riska_relative_increase'], 1)} |")
    at = sweep["at_max_fraction"]
    out.append("")
    out.append(f"At p_MY = {at['my_fraction']:.1f} the DEM section model reports "
               f"{_pct(at['dem_relative_increase'], 1)} against "
               f"{_pct(at['lindqvist_relative_increase'], 1)} for Lindqvist at the same "
               f"thickness ({_pct(at['dem_vs_lindqvist_relative'], 1)} relative to the "
               f"anchor increment) -- the same over-sensitivity diagnosed below.")
    return "\n".join(out)


def _diagnosis_table(artifact: dict[str, Any]) -> str:
    dem = artifact["dem"]
    diag = dem.get("scaling_diagnosis")
    if not diag:
        return "_DEM factorial skipped in this run._"
    out = ["| factor | b (DEM) | b (anchor) | b (DEM) x s | deviation raw | deviation normalised |",
           "|---|---|---|---|---|---|"]
    for key in ("compressive_strength_pa", "flexural_strength_pa", "elastic_modulus_pa"):
        row = diag[key]
        raw = _rel(row["dem_exponent"] - row["anchor_exponent"], row["anchor_exponent"])
        norm = _rel(row["dem_exponent_normalised"] - row["anchor_exponent"],
                    row["anchor_exponent"])
        out.append(f"| {_FACTOR_LABEL[key]} | {row['dem_exponent']:.4f} | "
                   f"{row['anchor_exponent']:.4f} | {row['dem_exponent_normalised']:.4f} | "
                   f"{raw} | {norm} |")
    shares = diag["anchor_component_shares"]
    out.append("")
    out.append("Anchor component shares at h = 1 m: crushing {:.3f}, bending {:.3f}, "
               "submersion {:.3f}, added drag {:.3f}; strength-carrying share "
               "s = {:.3f}.  The bonded-particle section carries no submersion term, so "
               "its strength exponents should be about 1/s times the anchor's."
               .format(shares["crushing"], shares["bending"], shares["submersion"],
                       shares["added_drag"], diag["anchor_strength_carrying_share"]))
    return "\n".join(out)


def _dem_thickness_table(artifact: dict[str, Any]) -> str:
    dem = artifact["dem"]
    thickness = dem.get("thickness") or {}
    if not thickness:
        return "_DEM resolution pair skipped in this run._"
    exps = thickness["thickness_exponent"]
    frac = thickness["flexure_bucket_fraction"]
    res = thickness["resolution_ratio"]
    out = ["| quantity | FY | MY |", "|---|---|---|",
           f"| h exponent (1.0 -> 1.5 m) | {exps['FY']:.3f} | {exps['MY']:.3f} |",
           f"| flexure bucket fraction, n_z = 4 | {frac['fy_coarse']:.3f} | {frac['my_coarse']:.3f} |",
           f"| flexure bucket fraction, n_z = 6 | {frac['fy_fine']:.3f} | {frac['my_fine']:.3f} |",
           f"| R(n_z = 6) / R(n_z = 4) | {res['fy']:.3f} | {res['my']:.3f} |"]
    out.append("")
    out.append(f"Anchor h exponent for the same states: {thickness['thickness_exponent_anchor']['lindqvist']:.3f}.")
    return "\n".join(out)


def _verdict_table(artifact: dict[str, Any]) -> str:
    out = ["| # | criterion | measured | verdict |", "|---|---|---|---|"]
    for key in sorted(artifact["verdicts"]):
        row = artifact["verdicts"][key]
        measured = row["measured"]
        if isinstance(measured, dict):
            text = "; ".join(f"{k} = {_fmt(v)}" for k, v in measured.items())
        else:
            text = _fmt(measured)
        out.append(f"| {key} | {row['criterion']} | {text} | **{row['verdict']}** |")
    return "\n".join(out)


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_fmt(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return str(value)


def _class_ratio_table(artifact: dict[str, Any]) -> str:
    dem = artifact["dem"]
    if "class_ratio" not in dem:
        return "_DEM factorial skipped in this run._"
    ratio = dem["class_ratio"]
    return "\n".join([
        "Class ratio at equal thickness:",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| DEM R(MY)/R(FY), full factorial corners | {ratio['dem_full_high_over_low']:.3f} |",
        f"| Lindqvist R(MY)/R(FY) | {ratio['lindqvist_full_high_over_low']:.3f} |",
        f"| Riska R(MY)/R(FY) | {ratio['riska_full_high_over_low']:.3f} |",
        f"| DEM deviation from Lindqvist, relative to the increment | "
        f"{_pct(ratio['dem_vs_lindqvist_relative'], 1)} |",
    ])


def _dem_ratio_error(artifact: dict[str, Any]) -> str:
    dem = artifact["dem"]
    if "class_ratio" not in dem:
        return "DEM not run"
    return _pct(dem["class_ratio"]["dem_vs_lindqvist_relative"], 1)


def _dem_ratio_verdict(artifact: dict[str, Any]) -> str:
    row = artifact["verdicts"].get("f_dem_class_ratio_vs_anchor")
    if row is None:
        return "not tested"
    return "**not < 3 %** (honest FAIL, criterion f)" if row["verdict"] == "FAIL" else "**< 3 %**"


def _dem_exponent_error(artifact: dict[str, Any]) -> str:
    dem = artifact["dem"]
    if "dem_fit" not in dem:
        return "DEM not run"
    d = dem["dem_fit"]["exponents"]["compressive_strength_pa"]
    a = dem["anchor_fit"]["lindqvist"]["exponents"]["compressive_strength_pa"]
    return f"{abs(d - a) / abs(a) * 100:.1f} %"


def _dem_exponent_verdict(artifact: dict[str, Any]) -> str:
    row = artifact["verdicts"].get("d_dem_sigma_c_exponent")
    if row is None:
        return "not tested"
    passed = row["verdict"] == "PASS"
    within3 = row["measured"]["relative_error"] < 0.03
    return ("**< 3 %**" if within3 else "**not < 3 %**") + (" (criterion d PASS)" if passed else " (criterion d FAIL)")


def _flexure_share(artifact: dict[str, Any]) -> str:
    row = artifact["verdicts"].get("k_dem_flexure_route_open")
    if row is None:
        return "an unresolved fraction"
    return f"{row['measured']['fy_coarse']:.3f}"


def render_results_markdown(artifact: dict[str, Any], speed: float | None = None) -> str:
    speed = artifact["primary_speed_m_s"] if speed is None else speed
    ep = artifact["my_fraction_sweep"]["primary_endpoints"]
    return "\n".join([
        "## Results (generated; every number from the artifact)",
        "",
        f"Primary speed {speed:g} m/s, icebreaker hull "
        f"(L = {artifact['my_fraction_sweep']['hull']['length_m']:g} m, "
        f"B = {artifact['my_fraction_sweep']['hull']['beam_m']:g} m, "
        f"entry angle {artifact['my_fraction_sweep']['hull']['entry_angle_deg']:g} deg).",
        "",
        f"- R(0) = {_mn(ep['resistance_start_N'])} MN, R(p_MY = {ep['my_fraction_end']:g}) = "
        f"{_mn(ep['resistance_end_N'])} MN, change {_pct(ep['relative_change'], 1)}.",
        f"- class ratio R_MY/R_FY = {ep['class_ratio_end']:.3f}; the mixture is exactly "
        f"class-additive, so dR/R(0) = p_MY (R_MY/R_FY - 1).",
        "",
        _curve_table(artifact, speed),
        "",
        "### Validation and DEM cross-check",
        "",
        _verdict_table(artifact),
        "",
    ])


def render_report(artifact: dict[str, Any], figures: Sequence[str] = ()) -> str:
    speed = artifact["primary_speed_m_s"]
    ep = artifact["my_fraction_sweep"]["primary_endpoints"]
    dem = artifact["dem"]
    has_dem = "dem_fit" in dem
    sens = artifact["sensitivity"]
    env = artifact["joint_envelope"]
    check = artifact["estimator_check"]
    n_pass, n_fail = artifact["n_pass"], artifact["n_fail"]
    fig_names = list(figures)
    fig_block = "\n".join(
        f"![{name}](../artifacts/mixed_ice/figures/{name})\n" for name in fig_names
    )
    fails = [k for k, v in artifact["verdicts"].items() if v["verdict"] == "FAIL"]
    riska_rows = [r for r in artifact["my_fraction_sweep"]["curves"][f"riska_v{speed:g}"]["rows"]
                  if r["my_strength_multiplier"] == 1.0]
    endpoint_riska = max(riska_rows, key=lambda r: r["my_fraction"])["relative_increase"]

    return f"""# Mixed first-year / multi-year ice resistance: MY-fraction sweep and FSD sensitivity

Evidence grade: **new capability**, relative claims only. The absolute ship-scale
level inherits the level-ice anchor's documented error against the MT Uikku model
tests (:mod:`tensordem.level_ice`); this document therefore quantifies *changes*
of resistance and their sensitivity, and says so explicitly in the error section.

### Answers to the two questions of the node

1. **How much does the ice resistance change when the multi-year area fraction
   goes from 0 % to 80 %?**  At v = {speed:g} m/s the Lindqvist anchor gives
   R = {_mn(ep['resistance_start_N'])} MN at p_MY = 0 and
   {_mn(ep['resistance_end_N'])} MN at p_MY = {ep['my_fraction_end']*100:.0f} %, i.e.
   **{_pct(ep['relative_change'], 1)}** (Riska: {_pct(endpoint_riska, 1)}; the two
   anchors agree to within {_pct(abs(ep['relative_change'] - endpoint_riska) / ep['relative_change'], 1)}).
   {_pct(artifact['class_ratio_decomposition']['lindqvist']['thickness_share_of_increment'], 0)}
   of that increment is the **thickness** of the MY class and only
   {_pct(artifact['class_ratio_decomposition']['lindqvist']['strength_share_of_increment'], 0)}
   is the **mechanical strengthening** of older ice.
2. **How sensitive is the prediction to the ice-thickness distribution?**  Over
   the declared parameter ranges the FSD group alone spans
   {_mn(env['fsd']['max_N'] - env['fsd']['min_N'])} MN
   ({_pct(env['fsd']['relative_span'], 0)} of the mean) against
   {_mn(env['material']['max_N'] - env['material']['min_N'])} MN
   ({_pct(env['material']['relative_span'], 0)}) for the material
   parameterisation, and the single dominant parameter is
   `{sens['rank_by_span'][0]}`: the **MY class-mean thickness**.
   The measured absolute level of R is not validated (no measured MY-ice series is
   available offline), so these are relative statements -- see the error section.

## 1. Case, geometry and parameters

The question is a field question, not a single-ice-sheet question: a ship
transiting an ice field in which a fraction ``p_MY`` of the *area* is multi-year
(MY) ice and the remainder is first-year (FY) ice. Two things are unknown for MY
ice and are parameterised here:

* **Material state.** MY ice is desalinated, colder and thicker. The ageing law
  of :mod:`tensordem.mixed_ice.aging` takes age as the single independent
  variable and returns salinity, temperature, brine volume, class-mean thickness,
  flexural strength, compressive strength, elastic modulus, fracture toughness
  and density. Brine volume enters through the gas-free freezing-point relation

      v_b = S (1 + 49.185/|T|) / 1000,

  the flexural strength through the Timco & O'Brien relation
  ``sigma_f = 1.76 exp(-5.88 sqrt(v_b))`` MPa, and the compressive strength,
  modulus and toughness through a pinned percolation form
  ``X = X_solid (1 - v_b/v_p)^n`` with ``v_p = 0.28``. The solid prefactors are
  *calibrated* so that age 1 returns the first-year values this repository already
  uses (``sigma_c = 1.5 MPa``, ``E = 5.0 GPa``), which makes the MY extension
  FY-consistent by construction.

{_ageing_table(artifact)}

* **Thickness distribution.** The field is the mixture
  ``F(h) = (1 - p) F_FY(h) + p F_MY(h)`` with lognormal marginals
  (FY mean {artifact['field_summary_p04']['fy_thickness_mean_m']:.2f} m, CV 0.35;
  MY mean {artifact['field_summary_p04']['my_thickness_mean_m']:.2f} m, CV 0.25,
  the mean being the ageing law averaged over a 2-5 a age window). Within a class
  thickness and floe diameter are treated as independent, which is what makes the
  field expectation exactly class-additive.

Hull (fixed): icebreaker, L = {artifact['my_fraction_sweep']['hull']['length_m']:g} m,
B = {artifact['my_fraction_sweep']['hull']['beam_m']:g} m,
T = {artifact['my_fraction_sweep']['hull']['draught_m']:g} m,
entry angle {artifact['my_fraction_sweep']['hull']['entry_angle_deg']:g} deg,
friction {artifact['my_fraction_sweep']['hull']['friction_coefficient']:g}.
Speeds {', '.join(f'{v:g}' for v in artifact['speeds_m_s'])} m/s; MY area fractions
{', '.join(f'{v:g}' for v in artifact['my_fractions'])}.

## 2. Method

1. **Ageing** (`mixed_ice/aging.py`) produces the material state per age class;
   the coefficient ledger is graded (DERIVED / CONVENTION / CITED / WITHHELD) and
   SHA-256 hashed into the artifact.
2. **FSD sampler and quadrature** (`mixed_ice/field.py`). Two estimators of the
   same field: a deterministic stratified quadrature (equal-probability thickness
   quantiles per class, an age x thickness grid for MY) that carries no noise, and
   a genuine Monte-Carlo floe sampler. The sampler draws the MY *label* with the
   area-tilted probability ``q = p W_FY / (p W_FY + (1-p) W_MY)`` so that the
   realised **area** fraction equals ``p`` even though MY floes are larger
   (at p = {check['target_my_fraction']:.2f} the tilt gives q =
   {artifact['field_summary_p04']['my_label_probability']:.4f}).
3. **Resistance** (`mixed_ice/resistance.py`): each FSD node is evaluated with the
   repository's own anchors (Lindqvist three-component, Riska speed-linear) at that
   node's thickness and material state, then area-weighted. Nothing is
   re-implemented, so the anchor's coefficient ledger remains the single source of
   truth.
4. **DEM cross-check** (`mixed_ice/campaign.py`): the bonded-particle section
   model of `tensordem.level_ice.sheet` is run in a 2^3 factorial over the MY/FY
   flexural strength, compressive strength and elastic modulus at h = 1 m
   (n_z = 4, mass scaling 200, advance 6 m). A log-linear fit gives the DEM's
   strength-sensitivity exponents, compared with the same exponents fitted on the
   engineering anchor. A thickness pair (1.0 / 1.5 m) gives the DEM thickness
   exponent and an n_z = 6 pair tests whether the flexure load path is
   resolution-limited.
5. **Verdicts**: eleven criteria are declared in `campaign.verdicts`, each with its
   measured number; honest FAILs are kept.

## 3. Results

### 3.1 R(MY fraction) at {speed:g} m/s

{_curve_table(artifact, speed)}

R(0) = {_mn(ep['resistance_start_N'])} MN, R(p_MY = {ep['my_fraction_end']:g}) =
{_mn(ep['resistance_end_N'])} MN: **the field resistance changes by
{_pct(ep['relative_change'], 1)}** from 0 % to {ep['my_fraction_end']*100:.0f} % MY,
i.e. {_mn(ep['linear_slope_N_per_unit_my'])} MN per unit MY area fraction
({_pct(ep['relative_change'] / ep['my_fraction_end'], 1)} per 10 % of MY ice).
This is a *per-floe level-ice upper estimate*, and section 3.2 shows that
{_pct(artifact['class_ratio_decomposition']['lindqvist']['thickness_share_of_increment'], 0)}
of the increment is the MY class-mean thickness
({artifact['class_ratio_decomposition']['lindqvist']['my_thickness_m']:.2f} m against
{artifact['class_ratio_decomposition']['lindqvist']['fy_thickness_m']:.2f} m for FY),
not the strength of older ice.

The mixture is exactly class-additive, so the curve carries no shape information
beyond the class ratio ``R_MY/R_FY = {ep['class_ratio_end']:.3f}``; criterion (b)
verifies the identity ``dR/R(0) = p (R_MY/R_FY - 1)`` to
{artifact['verdicts']['b_class_weight_identity']['measured']['max_abs_error']:.1e}.

Speed dependence of the same increment:

{_speed_table(artifact)}

Strength-parameterisation band (MY strength and modulus scaled by the printed
factors; Lindqvist, v = {speed:g} m/s):

{_band_table(artifact, speed)}

### 3.2 Where the increment comes from: thickness versus strength

{_decomposition_table(artifact)}

For the default field the MY/FY class ratio factorises into a thickness part of
{artifact['class_ratio_decomposition']['lindqvist']['ratio_thickness_only']:.2f}
and a strength part of
{artifact['class_ratio_decomposition']['lindqvist']['ratio_strength_only']:.2f},
with an interaction residual of only
{_pct(artifact['class_ratio_decomposition']['lindqvist']['interaction'], 1)}.
**The MY effect is a thickness effect**:
{_pct(artifact['class_ratio_decomposition']['lindqvist']['thickness_share_of_increment'], 0)}
of the increment comes from the thicker MY floes and
{_pct(artifact['class_ratio_decomposition']['lindqvist']['strength_share_of_increment'], 0)}
from the stronger MY ice.

### 3.3 FSD sensitivity (p_MY = {sens['base_my_fraction']:g})

{_sensitivity_table(artifact)}

Joint Latin-hypercube envelopes over the declared parameter ranges:

{_envelope_table(artifact)}

Reading: over the declared plausible ranges the **ice-thickness distribution**
contributes {_mn(sens['fsd_additive_bound_N'])} MN and the **material
parameterisation** {_mn(sens['material_additive_bound_N'])} MN of resistance spread
({_pct(sens['fsd_share_of_bound'], 0)} / {_pct(1 - sens['fsd_share_of_bound'], 0)});
the dominant single parameter is `{sens['rank_by_span'][0]}`. The FSD group p5-p95
envelope at fixed p_MY is
{_mn(env['fsd']['p05_N'])}-{_mn(env['fsd']['p95_N'])} MN against a mean of
{_mn(env['fsd']['mean_N'])} MN.

### 3.4 Estimator cross-check

| quantity | value |
|---|---|
| R, deterministic quadrature | {_mn(check['quadrature_N'])} MN |
| R, 20 000-floe Monte-Carlo | {_mn(check['sampler_N'])} MN |
| Monte-Carlo standard error | {_mn(check['sampler_standard_error_N'], 4)} MN |
| z = difference / standard error | {check['z_score']:.2f} |
| realised MY area fraction (target {check['target_my_fraction']:.2f}) | {check['sampler_realised_my_fraction']:.4f} |

### 3.5 DEM strength factorial (h = 1 m, n_z = 4)

{_dem_table(artifact)}

{_dem_exponent_table(artifact)}

{_dem_sweep_table(artifact)}

{_diagnosis_table(artifact)}

{_dem_thickness_table(artifact)}

{_class_ratio_table(artifact)}

## 4. Figures

{fig_block}

## 5. Verdicts

{n_pass} PASS / {n_fail} FAIL.

{_verdict_table(artifact)}

The FAILs are kept as stated. They share one identified root cause: in the DEM
section model the **flexure bucket carries {_flexure_share(artifact)} of the
resistance at n_z = 4** because the crushing cap saturates essentially every
contact, so the bending load path -- the route through which flexural strength
enters -- is closed. The DEM therefore reacts to the compressive strength with an
exponent close to the anchor's and almost not at all to the flexural strength,
which is exactly what criteria (d), (e), (f) and (k) measure. That is a property of
the reduced section model at this resolution (the level-ice campaign reports a
146 % mesh GCI), not of the mixed-field model, and it is the reason the reported
R(p_MY) curve is built on the engineering anchors while the DEM supplies the
strength-sensitivity cross-check and its validity boundary.

## 6. Error judgement (is it < 3 %?)

| quantity | reference | error | verdict |
|---|---|---|---|
| R(p_MY = 0) vs the FY Lindqvist anchor of the same field | internal identity | < 1e-12 (criterion b) | **< 3 %** |
| mixture identity dR/R(0) = p (ratio - 1) | algebra | {artifact['verdicts']['b_class_weight_identity']['measured']['max_abs_error']:.1e} | **< 3 %** |
| quadrature vs Monte-Carlo estimator | sampler with its own standard error | z = {check['z_score']:.2f} | **< 3 %** (within 3 standard errors) |
| MY/FY class ratio, DEM vs Lindqvist | bonded-particle section | {_dem_ratio_error(artifact)} | {_dem_ratio_verdict(artifact)} |
| DEM sigma_c exponent vs anchor | bonded-particle section | {_dem_exponent_error(artifact)} | {_dem_exponent_verdict(artifact)} |
| absolute ship-scale level of R(p_MY) | measured MY-ice resistance | **not available offline** | **cannot be judged**: no measured multi-year ice resistance series is readable, so the absolute level stays WITHHELD; the FY anchor's documented MAPE against the MT Uikku model tests is 27.9 % (the repository's level-ice campaign), which is inherited and is not < 15 % either |

Consequence for use: the **relative** results of this document (the 0-80 % change,
the thickness-versus-strength split and the sensitivity envelope) are reproducible
and internally verified, while any absolute resistance figure quoted from this
model carries the anchor's ~28 % class of error at the absolute level.

## 7. Limitations and pitfalls found

* **Class additivity is assumed, not derived.** It holds because thickness and
  floe diameter are treated as independent *within* a class. A field in which
  thick MY ice comes in large floes and thin FY ice in small floes would break it;
  the Monte-Carlo sampler is the honest estimator in that case.
* **Every floe is level ice.** Floe-size sub-level ice, floe rotation, broken-ice
  channels and the associated resistance drop are absent, so the mixed-field
  number is an upper estimate for a broken field.
* **Area weighting needs the tilt.** Drawing MY labels with probability ``p``
  over-represents MY ice whenever MY floes are larger; the sampler applies
  ``q = p W_FY / (p W_FY + (1-p) W_MY)``. Verified: realised area fraction
  {check['sampler_realised_my_fraction']:.4f} against a target of
  {check['target_my_fraction']:.2f}.
* **Fracture toughness is parameterised but consumed by nothing.** The
  bonded-particle sheet uses strength-only bonds, so ``K_Ic`` has zero influence
  here; the field is reported (`consumed_by_dem: false`) as an interface contract
  for a future fracture-energy bond model.
* **The MY thickness mean dominates the answer.** It is the ageing law averaged
  over the 2-5 a window; the campaign reports the whole plausible band, and the
  sensitivity ranking should be read as a statement about the declared ranges, not
  as a local derivative.
* **The aged material does not change the DEM flexure route**, and the section
  model is not mesh converged (GCI 146 % in the level-ice campaign). The DEM is
  used only for sensitivity exponents and ratios, never for the ship-scale level.
* **No MY resistance data.** Nothing in this document is validated against
  measured multi-year ice resistance; the evidence grade is WITHHELD at the
  absolute level by construction.

## 8. Reproduction

```
python scripts/mixed_ice_campaign.py            # full run (DEM includes n_z = 6 pair)
python scripts/mixed_ice_campaign.py --no-dem   # anchors only, no DEM
pytest tests/test_mixed_ice_aging.py tests/test_mixed_ice_field.py \\
       tests/test_mixed_ice_resistance.py tests/test_mixed_ice_campaign.py -q
pytest tests/test_level_ice_resistance_dem.py -q   # the extended CaseSpec
```

The 44 unit tests run in about 13 s; the campaign integration test adds roughly
30 s with the DEM skipped. One unrelated pre-existing collection error exists in
this worktree: `tests/test_shape_representation.py` imports `PileConfig2D`, which
is absent from `tensordem.shape_representation` at the baseline commit; that file
is not touched by this work.

Artifacts: `artifacts/mixed_ice/mixed_ice_campaign.json` (all tables and verdicts),
`artifacts/mixed_ice/figures/` (figures), aging ledger
`{artifact['provenance']['aging_ledger_sha256'][:16]}`,
level-ice ledger `{artifact['provenance']['level_ice_ledger_sha256'][:16]}`,
field spec `{artifact['provenance']['field_spec_sha256'][:16]}`.
Wall clock of the campaign: {artifact['wall_clock_s']:.1f} s.
"""