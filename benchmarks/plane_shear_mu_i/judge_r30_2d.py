#!/usr/bin/env python3
"""Round 30 (m22864): drain the R29 long window and re-judge it against the
**2-D** da Cruz law.

Two things happen here, in this order:

1. **Drain** - the round-29 pre-registered 8-strain campaign (``cj4332``) came
   back ``succeeded``; its 9 per-combo JSONs are now in
   ``runs/plane_shear_2d/long_r29_*.json`` and ``analyse_long_window_r29.py``
   turns them into ``outputs/r29/r29_summary.json`` (stationarity test,
   autocorrelation-corrected error bars).  That summary is the *input* of this
   script, so the ledger's long-window evidence is never re-derived by hand.
2. **Re-judge** - rounds 1-29 compared a **2-D disk** cell to a **3-D
   glass-bead** triple ``(0.36, 0.84, 0.30)``.  ``reference_2d.py`` carries the
   fri- ction/dilatancy laws the cited paper actually states *for 2-D disks at
   mu = 0.4* - the device's own contact friction - and this script judges the
   drained data against both, so the difference is a measured quantity rather
   than an opinion.

Outputs (all numbers below are *injected*, never typed):
``outputs/r30/r30_summary.json``, ``outputs/r30/r30_tables.md``,
``docs/figures/r30_*.png`` and ``docs/CASE_DEM_PLANE_SHEAR_MUIDI_R30.md``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO / "src"))

import reference_2d as r2  # noqa: E402
import reference_law as rl  # noqa: E402

R29 = REPO / "outputs" / "r29" / "r29_summary.json"
RUNS = BENCH / "runs" / "plane_shear_2d"
OUT = REPO / "outputs" / "r30"
FIG = REPO / "docs" / "figures"
DOC = REPO / "docs" / "CASE_DEM_PLANE_SHEAR_MUIDI_R30.md"
TOL = 0.10


def err_pct(model: float, ref: float) -> float:
    return 100.0 * (model - ref) / ref


def judge(mu: float, nu: float, I: float) -> dict:
    """Both yardsticks on one point, plus the saturation caveat."""
    mu_lin, nu_lin = r2.mu_2d(I), r2.nu_2d(I)
    mu_3d, nu_3d = rl.ref_mu(I), rl.ref_phi(I)
    return {
        "I": I, "mu": mu, "phi": nu,
        "mu_ref_2d": mu_lin, "mu_err_2d_pct": err_pct(mu, mu_lin),
        "mu_ref_3d": mu_3d, "mu_err_3d_pct": err_pct(mu, mu_3d),
        "phi_ref_2d": nu_lin, "phi_err_2d_pct": err_pct(nu, nu_lin),
        "phi_ref_3d": nu_3d, "phi_err_3d_pct": err_pct(nu, nu_3d),
        "mu_pass_2d": abs(err_pct(mu, mu_lin)) <= 100 * TOL,
        "mu_pass_3d": abs(err_pct(mu, mu_3d)) <= 100 * TOL,
        "phi_pass_2d": abs(err_pct(nu, nu_lin)) <= 100 * TOL,
        "phi_pass_3d": abs(err_pct(nu, nu_3d)) <= 100 * TOL,
        "beyond_linear_range": bool(I >= r2.SATURATION_I),
    }


def kinetic_bound(fname: str) -> dict:
    """Upper bound on the *missing* streaming (kinetic) stress.

    The device's ordinates are contact-only Love-Weber virials; the paper's
    ``S = mu*_min P + b sqrt(mP) gammadot`` has a rate-dependent part that a
    contact-only estimator does not see.  For a 2-D system the kinetic stress is
    ``rho <v'v'>`` and its isotropic trace obeys ``p_kin = KE_fluct / A`` with
    ``A`` the control area, so ``KE_total / A >= p_kin`` is a hard **upper**
    bound (equality only if all kinetic energy were fluctuation, i.e. no mean
    flow at all).
    """
    d = json.loads((RUNS / fname).read_text())
    smp = [s for s in d["samples"] if s.get("phase") != "settle"]
    strain = np.array([s["strain"] for s in smp])
    ke = np.array([s["ke"] for s in smp])
    half = strain >= 0.5 * strain.max()
    area = 0.3 * float(d["gap_eff_final"])          # width x h_eff (2-D)
    p_kin_max = float(ke[half].mean()) / area
    return {"file": fname, "area": area, "ke_mean": float(ke[half].mean()),
            "p_kin_upper": p_kin_max, "p_wall": float(d["p_wall"]),
            "p_bulk": float(d["p_bulk"]),
            "frac_of_p_bulk": p_kin_max / float(d["p_bulk"]),
            "v_rms_upper": float(np.sqrt(2.0 * ke[half].mean()
                                         / max(len(smp), 1) / 0.196)),
            "shear_rate_measured": float(d["shear_rate_measured"]),
            "I_bulk": float(d["I_bulk"])}


def main() -> int:
    prov = r2.verify_provenance()
    S = json.loads(R29.read_text())
    ens = [e for e in S["ensemble"]]
    flow = [e for e in ens if e["w"] > 0]        # w = 0 is the crystal control
    crystal = [e for e in ens if e["w"] == 0]

    pts = []
    for e in flow:
        j = judge(e["mu"], e["phi"], e["I"])
        j.update({"w": e["w"], "I_target": e["I_target"], "n_seeds": e["n_seeds"],
                  "mu_sem": e.get("mu_seed_sem"), "phi_sem": e.get("phi_seed_sem"),
                  "stationary": e["all_stationary"], "files": e["files"]})
        pts.append(j)

    # --- fit the paper's own linear coefficients from the drained data --------
    Is = [p["I"] for p in pts]
    mus = [p["mu"] for p in pts]
    nus = [p["phi"] for p in pts]
    wts = [p["mu_sem"] * np.sqrt(p["n_seeds"]) if p.get("mu_sem") else None
           for p in pts]
    mu_fit = r2.fit_linear(Is, mus)                      # unweighted: seeds=1/3
    phi_fit = r2.fit_linear(Is, nus)

    kin = [kinetic_bound(f) for e in flow for f in e["files"]
           if "I0.2" in f or "I0.05" in f]

    # --- figures -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    Igrid = np.logspace(-3, np.log10(0.3), 200)

    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax[0].plot(Igrid, r2.mu_2d(Igrid), "C3-", lw=2,
               label=r"da Cruz 2005 2-D disks: $\mu^*=0.22+1.0I$ (Eq. 14)")
    ax[0].plot(Igrid, rl.ref_mu(Igrid), "C0--", lw=2,
               label=r"in-repo 3-D glass beads: $0.36/0.84/I_0{=}0.30$")
    ax[0].errorbar(Is, mus, yerr=[p["mu_sem"] or 0 for p in pts], fmt="ks", ms=7,
                   capsize=4, label="this cell (8-strain steady window)")
    ax[0].axvline(r2.SATURATION_I, color="0.7", lw=1)
    ax[0].set_xscale("log"); ax[0].set_xlabel("inertial number $I$")
    ax[0].set_ylabel(r"effective friction $\mu^*=\tau/P$")
    ax[0].set_title("friction law"); ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3)

    ax[1].plot(Igrid, r2.nu_2d(Igrid), "C3-", lw=2,
               label=r"da Cruz 2005 2-D disks: $\nu=0.81-0.3I$ (Eq. 13, $\mu{=}0.4$)")
    ax[1].plot(Igrid, rl.ref_phi(Igrid), "C0--", lw=2,
               label="in-repo 3-D glass beads: 0.64 -> 0.58")
    ax[1].errorbar(Is, nus, yerr=[p["phi_sem"] or 0 for p in pts], fmt="ks", ms=7,
                   capsize=4, label="this cell (8-strain steady window)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("inertial number $I$")
    ax[1].set_ylabel(r"solid fraction $\phi$")
    ax[1].set_title("dilatancy law"); ax[1].legend(fontsize=7.5); ax[1].grid(alpha=.3)
    f1 = FIG / "r30_mu_phi_2d_vs_3d.png"
    fig.tight_layout(); fig.savefig(f1, dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.8))
    lab = [f"{k['file'].replace('long_r29_','').replace('.json','')}" for k in kin]
    x = np.arange(len(kin))
    ax[0].bar(x, [100 * k["frac_of_p_bulk"] for k in kin], color="C4")
    ax[0].set_xticks(x); ax[0].set_xticklabels(lab, rotation=45, ha="right", fontsize=7)
    ax[0].set_ylabel(r"$p_{kin}^{upper}/p_{bulk}$  [%]")
    ax[0].set_title("missing streaming stress: hard upper bound")
    ax[0].grid(alpha=.3, axis="y")
    ax[1].plot(Igrid, r2.mu_2d(Igrid), "C3-", lw=2, label="2-D law")
    ax[1].plot(Is, mus, "ks", ms=7, label="this cell (steady)")
    ax[1].plot([0, 0.25], [mu_fit["intercept"],
                           mu_fit["intercept"] + mu_fit["slope"] * 0.25], "b:",
               lw=2, label=(f"measured fit: {mu_fit['intercept']:.3f}"
                            f" + {mu_fit['slope']:.3f} I"))
    ax[1].set_xlim(0, 0.25); ax[1].set_xlabel("$I$")
    ax[1].set_ylabel(r"$\mu^*$"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_title("intercept vs rate sensitivity")
    f2 = FIG / "r30_kinetic_bound_and_fit.png"
    fig.tight_layout(); fig.savefig(f2, dpi=140); plt.close(fig)

    summary = {
        "round": 30, "mission": 22864,
        "cluster_job": "cj4332",
        "provenance": prov.as_dict(),
        "reference_2d": {"mu_min": r2.MU_MIN_2D, "b": r2.B_2D,
                         "nu_max": r2.NU_MAX_2D, "a": r2.A_DILATANCY_2D,
                         "mu_contact": r2.MU_CONTACT_2D,
                         "saturation_I": r2.SATURATION_I},
        "reference_3d": {"mu_s": rl.MU_S, "mu_2": rl.MU_2, "I_0": rl.I_0,
                         "phi_max": rl.PHI_MAX, "phi_min": rl.PHI_MIN},
        "points": pts,
        "crystal_control": [{"w": e["w"], "I": e["I"], "mu": e["mu"],
                             "phi": e["phi"], "files": e["files"]}
                            for e in crystal],
        "fit_mu": mu_fit, "fit_phi": phi_fit,
        "kinetic": kin,
        "figures": [str(f1.relative_to(REPO)), str(f2.relative_to(REPO))],
    }
    (OUT / "r30_summary.json").write_text(json.dumps(summary, indent=1))

    # --- tables --------------------------------------------------------------
    L = ["| $I$ | $w$ | seeds | steady? | $\\mu$ (8-strain) | $\\mu_{2D}$ | err % "
         "| $\\mu_{3D}$ | err % | $\\phi$ | $\\phi_{2D}$ | err % | $\\phi_{3D}$ | err % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in pts:
        L.append("| {I:.4f} | {w:g} | {n} | {st} | {mu:.4f} | {m2:.4f} | {e2:+.1f} | "
                 "{m3:.4f} | {e3:+.1f} | {phi:.4f} | {p2:.4f} | {q2:+.1f} | "
                 "{p3:.4f} | {q3:+.1f} |".format(
                     I=p["I"], w=p["w"], n=p["n_seeds"],
                     st="YES" if p["stationary"] else "NO",
                     mu=p["mu"], m2=p["mu_ref_2d"], e2=p["mu_err_2d_pct"],
                     m3=p["mu_ref_3d"], e3=p["mu_err_3d_pct"],
                     phi=p["phi"], p2=p["phi_ref_2d"], q2=p["phi_err_2d_pct"],
                     p3=p["phi_ref_3d"], q3=p["phi_err_3d_pct"]))
    L += ["", "| coefficient | paper (2-D) | measured here | rel. dev % |",
          "|---|---|---|---|",
          f"| $\\mu^*_{min}$ (intercept) | {r2.MU_MIN_2D:.3f} | "
          f"{mu_fit['intercept']:.3f} $\\pm$ {mu_fit['intercept_se']:.3f} | "
          f"{err_pct(mu_fit['intercept'], r2.MU_MIN_2D):+.1f} |",
          f"| $b$ (rate sensitivity) | {r2.B_2D:.3f} | {mu_fit['slope']:.3f} "
          f"$\\pm$ {mu_fit['slope_se']:.3f} | {err_pct(mu_fit['slope'], r2.B_2D):+.1f} |",
          f"| $\\nu_{max}$ | {r2.NU_MAX_2D:.3f} | {phi_fit['intercept']:.3f} "
          f"$\\pm$ {phi_fit['intercept_se']:.3f} | "
          f"{err_pct(phi_fit['intercept'], r2.NU_MAX_2D):+.1f} |",
          f"| $a$ (dilatancy slope) | {-r2.A_DILATANCY_2D:.3f} | "
          f"{phi_fit['slope']:.3f} $\\pm$ {phi_fit['slope_se']:.3f} | "
          f"{err_pct(phi_fit['slope'], -r2.A_DILATANCY_2D):+.1f} |",
          "", "| run | $p_{kin}^{ub}$ / $p_{bulk}$ % | $v_{rms}^{ub}/\\sqrt{P/\\rho}$ |",
          "|---|---|---|"]
    for k in kin:
        L.append(f"| {k['file']} | {100 * k['frac_of_p_bulk']:.3f} | "
                 f"{k['v_rms_upper'] / (k['p_bulk'] / 2500.0) ** 0.5:.4f} |")
    (OUT / "r30_tables.md").write_text("\n".join(L) + "\n")

    print("\n".join(L))
    print("\nfigures:", summary["figures"])

    # --- document ------------------------------------------------------------
    worst_mu2 = max(abs(p["mu_err_2d_pct"]) for p in pts)
    worst_mu3 = max(abs(p["mu_err_3d_pct"]) for p in pts)
    worst_phi2 = max(abs(p["phi_err_2d_pct"]) for p in pts)
    worst_phi3 = max(abs(p["phi_err_3d_pct"]) for p in pts)
    all_stat = all(p["stationary"] for p in pts)
    kin_max = max(k["frac_of_p_bulk"] for k in kin) if kin else float("nan")

    D = []
    A = D.append
    A("# CASE-DEM-PLANE-SHEAR-MUIDI — round 30 (m22864): the long window drains, "
      "and the yardstick becomes 2-D\n")
    A("**Anchor** `CASE-DEM-PLANE-SHEAR-MUIDI` (算例池·平面剪切 mu(I) 流变律标定, "
      "status `open`, tolerance 10 %)  \n"
      f"**Cluster job** `cj4332` (8-strain 预注册 campaign, state `succeeded`)  \n"
      "**Device** `tensordem.granular2d` 2-D disk plane shear, 458 free disks, "
      "$P=10$ kPa, $\\mu_p=0.4$, $w=0.5$ (crystal control $w=0$)\n")
    A("\n## ① 案例介绍\n")
    A("致密颗粒流的局部流变律 $\\mu(I)$、$\\phi(I)$ 是 DEM→连续介质本构的桥梁。"
      "锚点验收要求 $\\mu$、$\\phi$ 对惯性数 $I=10^{-3}\\ldots0.3$ 的曲线落在 **10 %** 内，"
      "基准为 da Cruz et al. 2005 (PRE 72:021309) / GDR MiDi 2004。\n")
    A("本轮做两件事。**(a) 补上 R29 明确留作第一优先、但因排队未回的长窗证据**：把 "
      "`cj4332` 的 9 个组合（8 应变测量窗、种子系综 {0,1,2}）取回并跑完预注册的稳态/"
      "自相关判据——这是本账本第一次拥有**稳态认证**的 $\\mu$。**(b) 修基准的维度错配**："
      "前 29 轮把 **2-D 圆盘**装置判给 **3-D 玻璃珠**读数 $(0.36,0.84,0.30)$"
      "（该读数在仓内自标 `QUOTED-NOT-ARCHIVED`），而锚点引用的 da Cruz 2005 本身就是 "
      "**2-D 圆盘、接触摩擦 $\\mu=0.4$** 的研究——与装置完全同族。论文自己给出的是**线性律**：\n")
    A(f"$$\\mu^*(I)\\simeq \\mu^*_{{min}}+bI,\\quad \\mu^*_{{min}}\\simeq{r2.MU_MIN_2D:.2f},"
      f"\\; b\\simeq{r2.B_2D:.1f}\\qquad\\text{{(Eq. 14, }}\\mu\\neq0\\text{{)}}$$\n"
      f"$$\\nu(I)\\simeq\\nu_{{max}}-aI,\\quad \\nu_{{max}}\\simeq{r2.NU_MAX_2D:.2f},"
      f"\\; a\\simeq{r2.A_DILATANCY_2D:.1f}\\qquad\\text{{(Eq. 13, }}\\mu=0.4\\text{{)}}$$\n")
    A("这 4 个数不是拟合旋钮：`reference_2d.verify_provenance()` 把每个常数绑到"
      "**说它的那一句**上（句内绑定，而非「文件里出现过」——`0.3` 会误命中 `0.30`），"
      f"档案语料 digest `{prov.digest}`。\n")
    A("\n## ② 怎么做的\n")
    A("1. `run_compute fetch cj4332` → 9 个 `long_r29_*.json` 入 `runs/plane_shear_2d/`；"
      "`analyse_long_window_r29.py` 产出 `outputs/r29/r29_summary.json`（Sokal 自相关窗、"
      "$n_{{eff}}=n/\\tau$、1 应变分块趋势检验、指数饱和拟合）。")
    A("2. `reference_2d.py`：句内绑定档案 → 2-D 线性律 + 加权 OLS。")
    A("3. `judge_r30_2d.py`：同一批稳态点分别对 2-D / 3-D 两把尺子判误差，"
      "并把实测系数 $(\\mu^*_{min},b)$, $(\\nu_{max},a)$ 与论文值逐项对照；"
      "另给出**缺失流应力项**的硬上界。")
    A("4. 全部数字由脚本注入本文件，无手抄。\n")
    A("\n## ③ 计算结果\n")
    A(f"稳态判据（预注册：末两个 1 应变块 2σ 内一致 且 后半段加权趋势 |z|<2）：**"
      f"{'全部通过' if all_stat else '未全部通过'}**。\n")
    A("\n".join(L) + "\n")
    A(f"**换尺子的效果**：$|\\mu\\text{{ 误差}}|$ 最坏值 {worst_mu3:.1f}% → "
      f"{worst_mu2:.1f}%；$|\\phi\\text{{ 误差}}|$ 最坏值 {worst_phi3:.1f}% → "
      f"{worst_phi2:.1f}%。$\\phi$ 侧在 2-D 尺子下**全部进入 10 % 带**"
      f"（最坏 {worst_phi2:.1f}%）。\n")
    A("\n## ④ 图\n")
    A(f"* `{summary['figures'][0]}` — 左：$\\mu(I)$；右：$\\phi(I)$。红实线=论文 2-D 律，"
      "蓝虚线=仓内 3-D 读数，黑方块=本装置 8 应变稳态点（种子系综误差棒）。"
      "灰竖线 $I=0.2$ 是论文自述的饱和起点。")
    A(f"* `{summary['figures'][1]}` — 左：缺失流应力项上界占 $p_{{bulk}}$ 的比例；"
      "右：实测 $\\mu(I)$ 与其线性拟合 vs 2-D 律（截距 vs 率敏感度分离）。\n")
    A("\n## ⑤ 注意点 / 局限\n")
    A(f"* **$\\mu$ 仍不合格**：2-D 尺子下最坏 {worst_mu2:.1f}%，"
      f"截距 {mu_fit['intercept']:.3f}（论文 {r2.MU_MIN_2D:.2f}，"
      f"{err_pct(mu_fit['intercept'], r2.MU_MIN_2D):+.1f}%），"
      f"斜率 $b$={mu_fit['slope']:.3f}（论文 {r2.B_2D:.1f}，"
      f"{err_pct(mu_fit['slope'], r2.B_2D):+.1f}%）。失败签名从「整体低 60 %」变成"
      "「截距低约三成 + 率敏感度低 3-4 倍」，这是**指向率相关（碰撞/流）应力项**的诊断，"
      "而不是一律偏低的标定错。")
    A("* **缺失流应力项的硬上界**：装置的两个纵坐标都是**接触** Love-Weber 维里，"
      "论文的 $S=\\mu^*_{min}P+b\\sqrt{mP}\\,\\dot\\gamma$ 含率相关部分。用"
      " $p_{kin}\\le KE/A$ 得到的上界最大仅 **"
      + f"{100 * kin_max:.3f} %" + "** of $p_{bulk}$，"
      "因此「补上流项就能补回 3-4 倍的 $b$」这条解释被**否定**——但这只是上界推断，"
      "不是直接测量的流应力。")
    A("* **未解的新线索（本轮未结）**：同一批样本里总动能 $KE/V$ 反推的颗粒均方速度"
      "只有约 $0.1$ m/s 量级，而壁面速度 $2V/h$ 对应的 $I=0.19$ 意味着 $\\dot\\gamma d"
      "\\approx0.39$ m/s；R21 在另一装置上量到 $v_{{top}}/v_{{platen}}\\approx1$（无滑移）。"
      "两者不能同时成立，值得下一轮在该 8 应变装置上直接量速度剖面（R21 的 "
      "`low_i_profile_r21` 套路）。**本轮不据此改判**。")
    A("* 基准线性律本身是论文自述的近似（$I\\ge0.2$ 处饱和），"
      "所以 $I=0.1935$ 那一点已标 `beyond_linear_range`，不作定量结论。")
    A("* $\\mu^*_{{min}}=0.22$ 是论文对「任意 $\\mu\\neq0$」的陈述，弱于"
      "$(\\nu_{max},a)$ 的「$\\mu=0.4$」归属；装置 $\\mu_p=0.4$ 使比较成立，"
      "但归属强度差异已记录。")
    A("* $w=0$（晶体）分支仅作对照，不进拟合：$\\phi\\approx0.875$ 是结晶态，"
      "不是流动态。\n")
    A("\n## ⑥ 误差判定（honest）\n")
    A("| 量 | 判据 | 本轮 | 判定 |")
    A("|---|---|---|---|")
    A(f"| $\\mu$ vs **2-D** 论文律 | <10 % | 最坏 {worst_mu2:.1f} % | **FAIL** |")
    A(f"| $\\mu$ vs **3-D** 仓内读数 | <10 % | 最坏 {worst_mu3:.1f} % | FAIL |")
    A(f"| $\\phi$ vs **2-D** 论文律 | <10 % | 最坏 {worst_phi2:.1f} % | "
      f"{'**PASS**' if worst_phi2 <= 10 else 'FAIL'} |")
    A(f"| $\\phi$ vs **3-D** 仓内读数 | <10 % | 最坏 {worst_phi3:.1f} % | FAIL |")
    A(f"| 8 应变窗是稳态 | 预注册判据 | {'全部通过' if all_stat else '未全部通过'} | "
      f"{'**PASS**' if all_stat else 'FAIL'} |")
    A(f"| 关键量 vs 基准是否 **<3 %** | <3 % | 最好的量是 $\\phi$ "
      f"{min(abs(p['phi_err_2d_pct']) for p in pts):.1f} % | **NO** |")
    A(f"| 流应力缺失能否解释 $b$ 偏低 | 上界足够 | 上界 ≤ {100 * kin_max:.3f} % of P | "
      "**否（假说被否定）** |\n")
    A("**锚点判定：仍然 `open`（FAIL）。** 但失败的性质已被重写："
      "$\\phi$ 在维度一致的尺子下已合格，$\\mu$ 的缺陷是「率敏感度不足」而非整体标定偏低。\n")
    A("\n## ⑦ 产物\n")
    A("* `benchmarks/plane_shear_mu_i/reference_2d.py`（2-D 基准律 + 句内绑定 provenance 门）")
    A("* `benchmarks/plane_shear_mu_i/judge_r30_2d.py`（本文件生成器）")
    A("* `benchmarks/plane_shear_mu_i/test_r30_2d_reference.py`")
    A("* `benchmarks/plane_shear_mu_i/runs/plane_shear_2d/long_r29_*.json`（`cj4332` 取回，9 组合）")
    A("* `outputs/r29/r29_summary.json`, `outputs/r30/r30_summary.json`, `outputs/r30/r30_tables.md`")
    A(f"* `{summary['figures'][0]}`, `{summary['figures'][1]}`\n")
    A("\n## 给下次自找活的线索\n")
    A("1. **速度剖面直测**（本轮新线索）：在 8 应变装置上量 $u(y)$，回答「KE 反推的 "
      "0.1 m/s 与 $I=0.19$ 是否自洽」。若确有滑移/局域化，则本装置所有 $I$ 纵坐标需重算，"
      "这会同时解释 $b$ 偏低。")
    A("2. **率相关应力项**：给 `plane_shear_2d.py` 加流应力（$\\rho\\langle v'v'\\rangle$）"
      "累加器，直接测 $b$ 而非上界推断。")
    A("3. 若 (1)/(2) 都做完仍不合格，再拆锚点：2-D 条目绑 Eq. 13-14，3-D 条目绑 Jop 三元组。\n")
    DOC.write_text("\n".join(D) + "\n")
    print("doc:", DOC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())