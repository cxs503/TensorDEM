"""Round 31 analysis: judge the two R30 open items with the measured profile.

Reads the ``prof_r31_*`` runs, decides the pre-registered questions of
``prereg_r31.json`` (abscissa validity, flow-stress magnitude), re-judges ``mu``
against the 2-D law of ``reference_2d`` (R30's yardstick) on both the nominal
and the measured abscissa, and writes every number used in
``docs/CASE_DEM_PLANE_SHEAR_MUIDI_R31.md`` into ``outputs/r31/`` (tables) and
``docs/figures/`` (png) - programmatically, so nothing is hand-copied.

    python benchmarks/plane_shear_mu_i/analyse_r31_profile.py            # tables + png
    python benchmarks/plane_shear_mu_i/analyse_r31_profile.py --doc      # + the round doc

A run that is missing is reported as ``not_run``; it is never silently skipped
and never counted as a pass.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent
for _p in (str(REPO / "src"), str(BENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import reference_2d as ref  # noqa: E402

RUNS = BENCH / "runs" / "plane_shear_2d"
OUT = BENCH / "outputs" / "r31"
FIGS = REPO / "docs" / "figures"
TOL = 0.10                      # the anchor's tolerance
ABSCISSA_BAND = (0.90, 1.10)    # prereg_r31: nominal abscissa is valid
ABSCISSA_FAIL = 0.80            # prereg_r31: below this the abscissa is broken
R2_VALID = 0.99
R2_FAIL = 0.95
KIN_NEGLIGIBLE = 0.01
KIN_MATERIAL = 0.10

COMBOS = [(0.5, 0.02, 0), (0.5, 0.05, 0), (0.5, 0.05, 1),
          (0.5, 0.20, 0), (0.5, 0.20, 1), (0.0, 0.05, 0)]

# programmatic colour system (never rely on matplotlib defaults for meaning)
C_LAW = "#1b3a6b"
C_NOMINAL = "#c1272d"
C_MEASURED = "#2e8b57"
C_TOTAL = "#b8860b"
C_CRYSTAL = "#8e8e8e"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_runs(suffix: str = "") -> list[dict]:
    rows = []
    for w, inertial, seed in COMBOS:
        path = RUNS / f"prof_r31_w{w:g}_I{inertial:g}_s{seed}{suffix}.json"
        if not path.exists():
            rows.append({"w": w, "I_target": inertial, "seed": seed,
                         "state": "not_run", "file": path.name})
            continue
        d = json.loads(path.read_text())
        prof = d.get("profile") or {}
        der = d.get("profile_derived") or {}
        rows.append({
            "state": "ok", "file": path.name, "w": w, "I_target": inertial,
            "seed": seed,
            "I_nominal": d.get("I_bulk"),
            "gamma_dot_nominal": der.get("gamma_dot_nominal"),
            "gamma_dot_measured": der.get("gamma_dot_measured"),
            "gamma_dot_ratio": der.get("gamma_dot_ratio"),
            "r2_linear": (prof.get("fit") or {}).get("r2"),
            "slip_b_bottom": der.get("slip_length_bottom_over_h"),
            "slip_b_top": der.get("slip_length_top_over_h", float("nan")),
            "mu_bulk": d.get("mu_bulk"),
            "mu_wall": d.get("mu_wall"),
            "mu_total": prof.get("mu_total"),
            "phi_gap": d.get("phi_gap"),
            "p_wall_over_P": d.get("p_wall_over_P"),
            "p_kin_share": prof.get("kinetic_share_p"),
            "tau_kin_share": prof.get("kinetic_share_tau"),
            "p_kin_total": prof.get("p_kinetic_total"),
            "tau_kin_total": prof.get("tau_kinetic_total"),
            "profile": prof,
            "duration_s": d.get("duration_s"),
            "samples": prof.get("samples"),
            "psi6": d.get("psi6_mean"),
            "diameter": (d.get("case") or {}).get("radius", 5e-3) * 2.0,
            "pressure": (d.get("case") or {}).get("normal_stress", 1.0e4),
            "density": (d.get("case") or {}).get("density", 2500.0),
        })
    return rows


def ok_rows(rows):
    return [r for r in rows if r["state"] == "ok"]


# --------------------------------------------------------------------------- #
# judging
# --------------------------------------------------------------------------- #
def judge(rows: list[dict]) -> dict:
    """Every pre-registered decision, computed from the runs (no hand input)."""
    good = ok_rows(rows)
    fluid = [r for r in good if r["w"] > 0.0]
    out: dict = {"n_expected": len(rows), "n_run": len(good),
                 "n_fluid": len(fluid)}

    ratios = np.array([r["gamma_dot_ratio"] for r in fluid], dtype=float)
    r2s = np.array([r["r2_linear"] for r in fluid], dtype=float)
    kin_p = np.array([r["p_kin_share"] for r in fluid], dtype=float)
    kin_t = np.array([r["tau_kin_share"] for r in fluid], dtype=float)

    if len(fluid):
        out["abscissa"] = {
            "ratio_min": float(np.nanmin(ratios)),
            "ratio_max": float(np.nanmax(ratios)),
            "r2_min": float(np.nanmin(r2s)),
            "valid": bool((ratios.min() >= ABSCISSA_BAND[0])
                          and (ratios.max() <= ABSCISSA_BAND[1])
                          and (r2s.min() >= R2_VALID)),
            "broken": bool((ratios.min() < ABSCISSA_FAIL) or (r2s.min() < R2_FAIL)),
        }
        out["flow_stress"] = {
            "p_share_max": float(np.nanmax(kin_p)),
            "tau_share_max": float(np.nanmax(kin_t)),
            "negligible": bool(np.nanmax(kin_p) < KIN_NEGLIGIBLE
                               and np.nanmax(kin_t) < KIN_NEGLIGIBLE),
            "material": bool(np.nanmax(kin_t) > KIN_MATERIAL),
        }

    # --- mu against the 2-D law, on three abscissae -------------------------
    judge_rows = []
    for r in good:
        I_in = float(r["I_nominal"]) if r.get("I_nominal") else float("nan")
        I_me = (float(r["gamma_dot_measured"]) * r["diameter"]
                / math.sqrt(r["pressure"] / r["density"])
                if r.get("gamma_dot_measured") else float("nan"))
        mu = float(r["mu_bulk"] or float("nan"))
        mu_tot = float(r["mu_total"] or float("nan"))
        phi = float(r["phi_gap"] or float("nan"))
        row = {"w": r["w"], "seed": r["seed"], "I_nominal": I_in,
               "I_measured": I_me, "mu": mu, "mu_total": mu_tot, "phi": phi,
               "gamma_dot_ratio": r["gamma_dot_ratio"], "r2": r["r2_linear"],
               "slip_bottom": r["slip_b_bottom"], "slip_top": r["slip_b_top"],
               "p_kin_share": r["p_kin_share"], "tau_kin_share": r["tau_kin_share"],
               "psi6": r["psi6"]}
        for tag, I in (("nominal", I_in), ("measured", I_me)):
            if not (I and math.isfinite(I) and I > 0):
                row[f"mu_ref_{tag}"] = float("nan")
                row[f"err_{tag}"] = float("nan")
                row[f"phi_ref_{tag}"] = float("nan")
                row[f"err_phi_{tag}"] = float("nan")
                continue
            m_ref = float(ref.mu_2d(I))
            n_ref = float(ref.nu_2d(I))
            row[f"mu_ref_{tag}"] = m_ref
            row[f"err_{tag}"] = 100.0 * (mu - m_ref) / m_ref
            row[f"phi_ref_{tag}"] = n_ref
            row[f"err_phi_{tag}"] = 100.0 * (phi - n_ref) / n_ref
        judge_rows.append(row)
    out["rows"] = judge_rows

    # --- rate sensitivity b on both abscissae ------------------------------
    fluid_j = [r for r in judge_rows if r["w"] > 0.0
               and math.isfinite(r["mu"]) and math.isfinite(r["I_nominal"])]
    if len(fluid_j) >= 2:
        I_nom = np.array([r["I_nominal"] for r in fluid_j])
        mu = np.array([r["mu"] for r in fluid_j])
        fit_nom = ref.fit_linear(I_nom, mu)
        out["fit_nominal"] = fit_nom
        meas = [r for r in fluid_j if math.isfinite(r["I_measured"])]
        if len(meas) >= 2:
            out["fit_measured"] = ref.fit_linear(
                np.array([r["I_measured"] for r in meas]),
                np.array([r["mu"] for r in meas]))
        tot = [r for r in fluid_j if math.isfinite(r["mu_total"])]
        if len(tot) >= 2:
            out["fit_total"] = ref.fit_linear(
                np.array([r["I_nominal"] for r in tot]),
                np.array([r["mu_total"] for r in tot]))

    # --- worst errors -------------------------------------------------------
    fluid_j_all = [r for r in judge_rows if r["w"] > 0.0]
    def worst(key):
        vals = [abs(r[key]) for r in fluid_j_all if math.isfinite(r.get(key, float('nan')))]
        return max(vals) if vals else float("nan")
    out["worst_err_mu_nominal"] = worst("err_nominal")
    out["worst_err_mu_measured"] = worst("err_measured")
    out["worst_err_phi_nominal"] = worst("err_phi_nominal")

    # --- localisation / layering -------------------------------------------
    loc = []
    for r in ok_rows(rows):
        prof = r.get("profile") or {}
        mu_loc = np.asarray(prof.get("mu_local", []), dtype=float)
        phi_b = np.asarray(prof.get("phi", []), dtype=float)
        fin = mu_loc[np.isfinite(mu_loc)]
        loc.append({
            "w": r["w"], "I_target": r["I_target"], "seed": r["seed"],
            "n_empty": int(np.isnan(mu_loc).sum()),
            "mu_local_min": float(fin.min()) if fin.size else float("nan"),
            "mu_local_max": float(fin.max()) if fin.size else float("nan"),
            "mu_local_spread": (float(fin.max() / fin.min())
                                if fin.size and fin.min() > 0 else float("nan")),
            "phi_min": float(np.nanmin(phi_b)) if phi_b.size else float("nan"),
            "phi_max": float(np.nanmax(phi_b)) if phi_b.size else float("nan"),
        })
    out["localisation"] = loc
    if loc:
        spreads = [l["mu_local_spread"] for l in loc if math.isfinite(l["mu_local_spread"])]
        out["localisation_max_spread"] = max(spreads) if spreads else float("nan")
        out["localised"] = bool(spreads and max(spreads) > 3.0)
    return out


# --------------------------------------------------------------------------- #
# writing: tables
# --------------------------------------------------------------------------- #
def fmt(x, digits=4, dash="–"):
    if x is None:
        return dash
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return dash
    return f"{v:.{digits}f}"


def tables_md(rows: list[dict], j: dict) -> str:
    L = []
    L.append("### 表 1 — 预注册项判定（`prereg_r31.json`）\n")
    L.append("| 量 | 判据 | 本轮实测 | 判定 |")
    L.append("|---|---|---|---|")
    ab = j.get("abscissa")
    if ab:
        L.append(f"| $\\dot\\gamma_{{meas}}/\\dot\\gamma_{{nom}}$ | 0.90–1.10 | "
                 f"{fmt(ab['ratio_min'],3)} – {fmt(ab['ratio_max'],3)} | "
                 f"{'PASS' if ab['valid'] else ('FAIL（坐标轴需重标）' if ab['broken'] else 'MARGINAL')} |")
        L.append(f"| 剖面线性度 $r^2$ | ≥ 0.99 | ≥ {fmt(ab['r2_min'],3)} | "
                 f"{'PASS' if ab['r2_min'] >= R2_VALID else 'FAIL'} |")
    fs = j.get("flow_stress")
    if fs:
        L.append(f"| $p_{{kin}}/p_{{tot}}$ | < 1 % | ≤ {fmt(100*fs['p_share_max'],3)} % | "
                 f"{'PASS（流项可忽略）' if fs['negligible'] else 'FAIL（流项不可忽略）'} |")
        L.append(f"| $\\tau_{{kin}}/\\tau_{{tot}}$ | < 1 % | ≤ {fmt(100*fs['tau_share_max'],3)} % | "
                 f"{'PASS（流项可忽略）' if fs['negligible'] else 'FAIL'} |")
    L.append("")
    L.append("### 表 2 — 逐点结果（mu/phi 分别对 2-D 论文律，名义/实测两种 $I$）\n")
    L.append("| w | seed | $I_{nom}$ | $I_{meas}$ | 比值 | $r^2$ | $\\mu$ | $\\mu_{ref}(I_{nom})$ | err % | "
             "$\\mu_{ref}(I_{meas})$ | err % | $\\phi$ | $\\phi_{ref}$ | err % | $p_{kin}$ 份额 % | $\\psi_6$ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in j.get("rows", []):
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            fmt(r["w"], 2), r["seed"], fmt(r["I_nominal"], 4), fmt(r["I_measured"], 4),
            fmt(r["gamma_dot_ratio"], 3), fmt(r["r2"], 4), fmt(r["mu"], 4),
            fmt(r["mu_ref_nominal"], 4), fmt(r["err_nominal"], 1),
            fmt(r["mu_ref_measured"], 4), fmt(r["err_measured"], 1),
            fmt(r["phi"], 4), fmt(r["phi_ref_nominal"], 4), fmt(r["err_phi_nominal"], 1),
            fmt(100 * (r["p_kin_share"] or float("nan")), 3), fmt(r["psi6"], 3)))
    L.append("")
    L.append("### 表 3 — 律系数（2-D 论文律 $\\mu^*=\\mu^*_{min}+bI$，$\\nu=\\nu_{max}-aI$）\n")
    L.append("| 拟合基 | 截距 $\\mu^*_{min}$ | 斜率 $b$ | 点数 | 备注 |")
    L.append("|---|---|---|---|---|")
    L.append(f"| 论文（da Cruz 2005 Eq. 14） | {ref.MU_MIN_2D:.3f} | {ref.B_2D:.3f} | – | 基准 |")
    for tag, key, note in (("名义 $I$", "fit_nominal", "接触应力 $\\mu$"),
                           ("实测 $I$", "fit_measured", "接触应力 $\\mu$，横坐标按剖面斜率重算"),
                           ("名义 $I$（含流项）", "fit_total", "$\\mu_{tot}=(\\tau_c+\\tau_k)/(p_c+p_k)$")):
        f = j.get(key)
        if f:
            L.append(f"| {tag} | {fmt(f.get('intercept'),4)} | {fmt(f.get('slope'),4)} | "
                     f"{f.get('n','–')} | {note} |")
        else:
            L.append(f"| {tag} | – | – | – | 点数不足 |")
    L.append("")
    L.append("### 表 4 — 分层应力比（局域化/分层检测器）\n")
    L.append("| w | $I_{target}$ | seed | 空 bin 数 | $\\mu_{local}$ 最小 | 最大 | 最大/最小 | $\\phi$ 最小 | 最大 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for l in j.get("localisation", []):
        L.append("| %s | %s | %d | %d | %s | %s | %s | %s | %s |" % (
            fmt(l["w"], 2), fmt(l["I_target"], 3), l["seed"], l["n_empty"],
            fmt(l["mu_local_min"], 3), fmt(l["mu_local_max"], 3),
            fmt(l["mu_local_spread"], 2), fmt(l["phi_min"], 3), fmt(l["phi_max"], 3)))
    L.append("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# writing: figures
# --------------------------------------------------------------------------- #
def make_figures(rows: list[dict], j: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    paths = []
    good = [r for r in ok_rows(rows) if (r.get("profile") or {}).get("samples")]
    if not good:
        return paths

    # --- fig 1: the measured mean velocity profile --------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cmap = plt.get_cmap("viridis")
    for k, r in enumerate(good):
        prof = r["profile"]
        y = np.asarray(prof["y_centre"], float)
        ux = np.asarray(prof["ux"], float)
        h = prof["bin_height"] * prof["n_bins"]
        yn = (y - (y[-1] - h / 2 + prof["bin_height"] / 2)) / h
        yn = (y - y[0]) / (y[-1] - y[0])
        v = prof["v_wall"]
        ax.plot(yn, ux / v, "o-", ms=3.5, lw=1.2,
                color=cmap(k / max(1, len(good) - 1)) if r["w"] > 0 else C_CRYSTAL,
                label=f"w={r['w']:g}, I={r['I_target']:g}, s{r['seed']} "
                      f"(r²={fmt(r['r2_linear'],3)})")
    ax.plot([0, 1], [1, -1], "k--", lw=1.6, label="ideal Couette $u_x/v_{wall}=1-2y/h$")
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel("normalised height $y/h$ (bottom → top)")
    ax.set_ylabel("$u_x/v_{wall}$")
    ax.set_title("R31: measured mean velocity profile (8-strain window)")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p1 = FIGS / "r31_velocity_profile.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    paths.append(p1)

    # --- fig 2: mu vs I on both abscissae, against the 2-D law --------------
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    Iline = np.geomspace(1e-3, 0.35, 200)
    ax.plot(Iline, ref.mu_2d(Iline), "-", color=C_LAW, lw=1.8,
            label="da Cruz 2005 2-D law (Eq. 14)")
    jr = [r for r in j.get("rows", []) if r["w"] > 0.0 and math.isfinite(r["mu"])]
    for r in jr:
        m = r["mu"]
        if math.isfinite(r["I_nominal"]):
            ax.plot(r["I_nominal"], m, "o", ms=7, color=C_NOMINAL,
                    label="measured, nominal $I$" if r is jr[0] else None)
        if math.isfinite(r["I_measured"]):
            ax.annotate("", xy=(r["I_measured"], m), xytext=(r["I_nominal"], m),
                        arrowprops=dict(arrowstyle="->", color="0.5", lw=1.0))
            ax.plot(r["I_measured"], m, "s", ms=6, color=C_MEASURED,
                    label="same point, profile-measured $I$" if r is jr[0] else None)
        if math.isfinite(r["mu_total"] or float("nan")) and math.isfinite(r["I_nominal"]):
            ax.plot(r["I_nominal"], r["mu_total"], "^", ms=6, color=C_TOTAL,
                    label="with kinetic stress $\\mu_{tot}$" if r is jr[0] else None)
    ax.set_xscale("log")
    ax.set_xlabel("inertial number $I$")
    ax.set_ylabel("effective friction $\\mu$")
    ax.set_title("R31: $\\mu(I)$ on both abscissae (10 % band would be invisible here)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    p2 = FIGS / "r31_mu_vs_I_abscissa.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    paths.append(p2)

    # --- fig 3: per-bin mu and phi (layering / localisation) ----------------
    sel = [r for r in good if r["w"] > 0.0]
    sel.sort(key=lambda r: (r["I_target"], r["seed"]))
    if sel:
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
        for k, r in enumerate(sel):
            prof = r["profile"]
            y = np.asarray(prof["y_centre"], float)
            yn = (y - y[0]) / (y[-1] - y[0])
            axes[0].plot(yn, np.asarray(prof["mu_local"], float), "o-", ms=3, lw=1.1,
                         label=f"w={r['w']:g}, I={r['I_target']:g}, s{r['seed']}")
            axes[1].plot(yn, np.asarray(prof["phi"], float), "o-", ms=3, lw=1.1,
                         label=f"w={r['w']:g}, I={r['I_target']:g}, s{r['seed']}")
        axes[0].set_xlabel("$y/h$")
        axes[0].set_ylabel("local $\\mu=\\tau_c/p_c$")
        axes[0].set_title("R31: depth-resolved stress ratio")
        axes[1].set_xlabel("$y/h$")
        axes[1].set_ylabel("local area fraction $\\phi$")
        axes[1].set_title("R31: depth-resolved packing fraction")
        for a in axes:
            a.grid(alpha=0.25)
            a.legend(fontsize=6)
        fig.tight_layout()
        p3 = FIGS / "r31_depth_profiles.png"
        fig.savefig(p3, dpi=140)
        plt.close(fig)
        paths.append(p3)
    return paths


# --------------------------------------------------------------------------- #
# doc
# --------------------------------------------------------------------------- #
def doc_md(rows, j, figs, *, probe_note: str = "") -> str:
    ab = j.get("abscissa", {})
    fs = j.get("flow_stress", {})
    L = []
    L.append("# CASE-DEM-PLANE-SHEAR-MUIDI — round 31 (m22979): 速度剖面与流应力直测，坐标轴与失效签名分离\n")
    L.append("**Anchor** `CASE-DEM-PLANE-SHEAR-MUIDI`（算例池·平面剪切 mu(I) 流变律标定，status `open`，容差 10 %）  ")
    L.append("**Repo** TensorDEM · **branch** `agent/m22979` · **device** `tensordem.granular2d` 2-D rough-wall plane shear"
             "（与 R24–R30 **逐位相同**的算例 + 分层剖面仪）\n")
    L.append("## ① 案例介绍\n")
    L.append("致密颗粒流的局部流变律 $\\mu(I)$、$\\phi(I)$ 是 DEM→连续介质本构的桥梁。锚点要求二者对 "
             "da Cruz et al. 2005 (PRE 72:021309) / GDR MiDi 2004 的曲线落在 **10 %** 内。R30 把维度错配"
             "（2-D 圆盘 vs 3-D 玻璃珠读数）改正后，失效签名变了：$\\nu_{max}$ 合格（+1.7 %），但"
             "**率敏感度 $b$ 只有论文的 1/3–1/4**（0.279 vs 1.0）、截距低 29 %。R30 留下两个**它自己无法回答**的问题：\n")
    L.append("1. **横坐标是否可信？** 装置用名义剪切率 $2V/h_{eff}$ 构造 $I$；R30 发现由动能反推的颗粒速度只有 ~0.1 m/s，"
             "而 $I=0.19$ 需要 $\\dot\\gamma d\\approx0.39$ m/s，两者不能同时成立。若存在壁面滑移/剪切局域化，"
             "**整条 $I$ 轴都是错的**。\n")
    L.append("2. **率相关（流）应力到底多大？** R30 只能给上界 $p_{kin}\\le KE/A\\le0.132\\,\\%\\,p_{bulk}$，"
             "据此否定了「补上流项就能补回 $b$」，但从未**直接测量**该应力项。\n")
    L.append("本轮只做这两件事：给装置加**分层速度/应力分解仪**（`profile_r31.py`），一次测量同时回答 (1)(2)。"
             "接触物理、算例参数、8 应变窗、判据尺子（`reference_2d`）全部不动，因此与归档轮次逐点可比。\n")
    L.append("## ② 怎么做的\n")
    L.append("1. `profile_r31.ProfileAccumulator`：把**每一个采样时刻**的自由盘速度与全部接触按 $y$ 分入 10 个等厚层"
             "（层区间 $[y_{bottom}+r,\\ y_{top}-r]$；接触按**中点** $y$ 归层，分支向量穿周期像，与 `measure()` 的整胞维里同一约定），"
             "**池化**累加原始和（不是逐样本均值），于是层内方差既含剪切梯度又含真实颗粒温度。\n")
    L.append("2. 层内量：$u_x,u_y$、$\\phi$、接触应力 $\\sigma_{ab}=\\sum r_af_b/(W h_{bin}t)$、动能应力 "
             "$p_{kin}=\\rho\\phi(\\mathrm{var}_x+\\mathrm{var}_y)/2$、$\\tau_{kin}=-\\rho\\phi\\,\\mathrm{cov}_{xy}$、"
             "以及层局部应力比 $\\mu_{local}=\\tau_c/p_c$。\n")
    L.append("3. 剪切率由**中心 60 % 层带的加权最小二乘斜率**给出（不受壁面层影响），$r^2$ 作局域化诊断；"
             "壁面滑移 = 最靠壁层的 $u_x$ 与施加壁速之差，滑移长度 = 该差 / 实测斜率。\n")
    L.append("4. `prereg_r31.json` **先定后跑**：bin 数（10）、拟合带（中心 60 %）、8 应变窗、判定阈值（比值 0.90–1.10、$r^2\\ge0.99$、"
             "流项份额 <1 %）都在结果回来之前写死；`run_profile_r31.py` 递交 6 个组合（与 `cj4332` 同名同参）。\n")
    L.append("5. `analyse_r31_profile.py` 生成本文所有数字（零手抄），并对 `reference_2d` 的 2-D 律**在名义 $I$ 与实测 $I$ 上各判一次**。\n")
    L.append("## ③ 计算结果\n")
    if j.get("n_run") == 0:
        L.append("**本轮运行未回**（`not_run`）：下列各节为空缺，判定不作改变。\n")
    L.append(f"回执：预期 {j['n_expected']} 个组合，实回 **{j['n_run']}** 个（流体点 {j['n_fluid']} 个）。\n")
    if probe_note:
        L.append(probe_note.strip() + "\n")
    L.append("**关键判定（预注册）**\n")
    L.append(f"* 横坐标：$\\dot\\gamma_{{meas}}/\\dot\\gamma_{{nom}}\\in[{fmt(ab.get('ratio_min'),3)},\\ {fmt(ab.get('ratio_max'),3)}]$，"
             f"$r^2\\ge{fmt(ab.get('r2_min'),3)}$ ⇒ **{'名义坐标轴成立' if ab.get('valid') else '坐标轴需重标'}**。\n")
    L.append(f"* 流应力：$p_{{kin}}$ 份额 ≤ {fmt(100*fs.get('p_share_max', float('nan')),3)} %、"
             f"$\\tau_{{kin}}$ 份额 ≤ {fmt(100*fs.get('tau_share_max', float('nan')),3)} % ⇒ "
             f"**{'流项可忽略（直接测量证实 R30 的上界推断）' if fs.get('negligible') else '流项不可忽略'}**。\n")
    L.append(f"* 最差误差：$\\mu$（名义 $I$）{fmt(j.get('worst_err_mu_nominal'),1)} %；"
             f"$\\mu$（实测 $I$）{fmt(j.get('worst_err_mu_measured'),1)} %；$\\phi$ {fmt(j.get('worst_err_phi_nominal'),1)} %。\n")
    L.append("")
    L.append(tables_md(rows, j))
    if figs:
        L.append("### 图\n")
        for p in figs:
            L.append(f"![{p.stem}](figures/{p.name})\n")
    L.append("## ④ 注意点 / 局限\n")
    L.append("* 剖面仪只测**平均**速度与速度涨落的二阶矩；颗粒温度的各向异性更高阶矩、以及接触力链方向分布未测。\n")
    L.append("* 分层把接触按中点归层，靠近壁面的一层因此含少量「壁粒—自由盘」接触，$\\mu_{local}$ 在边界层不可与体相直接比较。\n")
    L.append("* 8 应变窗内的池化统计假设过程近似平稳；R30 的预注册稳态判据在此装置上并未全部通过，故 $\\mu$ 的系统误差不因样本量增大而消失。\n")
    L.append("* 基准律仍是论文自述的线性近似（$I\\gtrsim0.2$ 饱和），$I=0.1935$ 那点标 `beyond_linear_range`，不作定量结论。\n")
    L.append("* 晶体对照 $w=0$ 只作机制对照（基面滑移），不进拟合。\n")
    L.append("## ⑤ 误差判定（honest）\n")
    L.append("| 量 | 判据 | 本轮 | 判定 |")
    L.append("|---|---|---|---|")
    L.append(f"| $\\mu$ vs 2-D 论文律（名义 $I$） | <10 % | {fmt(j.get('worst_err_mu_nominal'),1)} % | "
             f"{'PASS' if (j.get('worst_err_mu_nominal') or 1e9) < 10 else 'FAIL'} |")
    L.append(f"| $\\mu$ vs 2-D 论文律（实测 $I$） | <10 % | {fmt(j.get('worst_err_mu_measured'),1)} % | "
             f"{'PASS' if (j.get('worst_err_mu_measured') or 1e9) < 10 else 'FAIL'} |")
    L.append(f"| $\\phi$ vs 2-D 论文律 | <10 % | {fmt(j.get('worst_err_phi_nominal'),1)} % | "
             f"{'PASS' if (j.get('worst_err_phi_nominal') or 1e9) < 10 else 'FAIL'} |")
    L.append(f"| 关键量 <3 % | <3 % | 最好 {fmt(min([abs(j.get('worst_err_mu_nominal') or 1e9), abs(j.get('worst_err_phi_nominal') or 1e9)]),1)} % | **NO** |")
    L.append("")
    L.append("**锚点判定：** 由上述证据决定（见上表）；本轮不因仪器改进而放宽任何阈值。\n")
    L.append("## ⑥ 产物\n")
    L.append("* `benchmarks/plane_shear_mu_i/profile_r31.py`（分层剖面 + 接触/动能应力分解仪）")
    L.append("* `benchmarks/plane_shear_mu_i/run_profile_r31.py`、`analyse_r31_profile.py`、`test_r31_profile.py`")
    L.append("* `benchmarks/plane_shear_mu_i/prereg_r31.json`（先定后跑）")
    L.append("* `benchmarks/plane_shear_mu_i/runs/plane_shear_2d/prof_r31_*.json`、`outputs/r31/`")
    L.append("* `docs/figures/r31_velocity_profile.png`、`r31_mu_vs_I_abscissa.png`、`r31_depth_profiles.png`")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc", action="store_true", help="also write the round doc")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--suffix", default="",
                    help="run-file suffix, e.g. _smoke for the reduced-cell probe")
    ap.add_argument("--probe-note", default="",
                    help="markdown paragraph injected under the run receipt")
    args = ap.parse_args(argv)

    rows = load_runs(args.suffix)
    j = judge(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "r31_tables" + (args.suffix or "")
    (OUT / f"{stem}.md").write_text(tables_md(rows, j))
    (OUT / f"r31_summary{args.suffix or ''}.json").write_text(
        json.dumps(j, indent=1, default=float))
    figs = [] if args.no_figures else make_figures(rows, j)
    if args.doc:
        doc = doc_md(rows, j, figs, probe_note=args.probe_note)
        (REPO / "docs" / "CASE_DEM_PLANE_SHEAR_MUIDI_R31.md").write_text(doc)
        print("wrote docs/CASE_DEM_PLANE_SHEAR_MUIDI_R31.md")
    print(json.dumps({k: v for k, v in j.items()
                      if k not in ("rows", "localisation", "fit_nominal",
                                   "fit_measured", "fit_total")},
                     indent=1, default=float))
    for k in ("fit_nominal", "fit_measured", "fit_total"):
        if k in j:
            print(k, json.dumps(j[k], default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())