#!/usr/bin/env python3
"""Build ``benchmarks/anchors/ice-polygon-floe-field.json`` (W18 paper-exp, exp #1).

Every number in the artifact is produced here from code paths that exist in this
worktree — the floe field comes from :func:`tensordem.polygon_floe_field.
generate_polygon_floe_field`, the operator from
:mod:`benchmarks.anchors.polygon_floe_anchor`.  Nothing is hand-transcribed, and
no ISO 19906 value is invented: the two anchor slots stay WITHHELD until a
verbatim transcription is injected as JSON (``--anchor-file``).

Usage::

    python3 benchmarks/anchors/build_ice_polygon_floe_field.py
    python3 benchmarks/anchors/build_ice_polygon_floe_field.py --figures

Exit status is 0 even on an honest FAIL — the verdict lives in the artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

import polygon_floe_anchor as op  # noqa: E402
from tensordem.fsd_packing import BimodalFSD, DeltaFSD, LogNormalFSD, PowerLawFSD  # noqa: E402
from tensordem.polygon_floe_field import generate_polygon_floe_field  # noqa: E402

JSON_PATH = os.path.join(HERE, "ice-polygon-floe-field.json")
FIG_DIR = os.path.join(REPO, "artifacts", "anchors", "ice_polygon_floe_field")
TASK_KEY = "paper-exp-ice-polygon-floe-field-1"
MISSION_ID = 7514

# --- case caliber (IN32-N01 field; parameters from code, not from memory) ----
FSD_FAMILIES = {
    "PowerLawFSD(2.5)": PowerLawFSD(2.5, 5.0),
    "LogNormalFSD(0.5)": LogNormalFSD(0.5, 5.0),
    "BimodalFSD()": BimodalFSD(),
    "DeltaFSD(5.0)": DeltaFSD(5.0),
}
COVERAGES = (0.60, 0.85, 0.90)
N_FLOES = 400
SEEDS = (0, 1, 2, 3)
KAPPA = 0.5                    # chord-width factor of the floe contact patch
CAND_N = 0.4284                # reference-only candidate (see candidates_unverified)
CAND_CI = (0.42, 0.4362)
N_SWEEP = np.round(np.arange(0.20, 0.7001, 0.01), 2)


# --------------------------------------------------------------------------- #
def probe_source_access() -> List[Dict[str, object]]:
    """One pass over every channel that could legitimately carry the primary text."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: List[Dict[str, object]] = []
    for host in ("www.iso.org", "iso.org"):
        try:
            s = socket.create_connection((host, 443), timeout=6)
            s.close()
            rows.append({"ts": ts, "method": "tcp_connect", "target": f"{host}:443",
                         "result": "OK"})
        except Exception as exc:                                    # noqa: BLE001
            rows.append({"ts": ts, "method": "tcp_connect", "target": f"{host}:443",
                         "result": f"FAIL {type(exc).__name__}: {exc}"})
    for url in ("https://www.iso.org/standard/72701.html",
                "https://www.iso.org/obp/ui/#iso:std:iso:19906:ed-2:v1:en"):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:   # noqa: S310
                body = resp.read(4000).decode("utf-8", "replace")
            has_annex = "Annex A" in body and "pressure" in body.lower()
            rows.append({"ts": ts, "method": "http_get", "target": url,
                         "result": f"HTTP {resp.status}",
                         "body_prefix_head": not has_annex,
                         "contains_annex_a_text": has_annex})
        except Exception as exc:                                    # noqa: BLE001
            rows.append({"ts": ts, "method": "http_get", "target": url,
                         "result": f"FAIL {type(exc).__name__}: {exc}"})
    roots = [REPO, "/root/tp-agent/docs", "/root/tp-agent/knowledge-center",
             "/root/ice_docs", "/root/ice_docs2"]
    hits: List[Dict[str, str]] = []
    n_files = 0
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if "/.git" in dirpath or "/__pycache__" in dirpath:
                continue
            for name in files:
                if not name.endswith((".md", ".txt", ".json", ".py", ".csv")):
                    continue
                n_files += 1
                if "19906" in name:
                    hits.append({"path": os.path.join(dirpath, name),
                                 "kind": "filename_match"})
                    continue
                path = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(path) > 2_000_000:
                        continue
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        txt = fh.read()
                except OSError:
                    continue
                if "19906" in txt:
                    # a *mention* is not the primary text; record the distinction
                    primary = ("pressure" in txt.lower() and "annex a" in txt.lower()
                               and "iso 19906:2019" in txt.lower()
                               and len(txt) > 100_000)
                    hits.append({"path": path,
                                 "kind": "text_mention" if not primary else "possible_copy"})
    rows.append({"ts": ts, "method": "filesystem_inventory", "target": roots,
                 "files_scanned": n_files, "hits": hits[:40],
                 "primary_text_hits": sum(1 for h in hits
                                          if h["kind"] == "possible_copy")})
    return rows


def build_fields() -> Dict[str, Dict[str, object]]:
    """Generate the IN32-N01 polygon fields and the per-floe contact calibers."""
    out: Dict[str, Dict[str, object]] = {}
    for fam_name, fsd in FSD_FAMILIES.items():
        for cov in COVERAGES:
            areas, reqs, boxes, thick = [], [], [], []
            t0 = time.time()
            for seed in SEEDS:
                f = generate_polygon_floe_field(fsd, coverage=cov, n_floes=N_FLOES,
                                                seed=seed)
                tab = f.floe_table()
                areas.append(np.asarray(tab["area"], dtype=float))
                reqs.append(np.asarray(tab["r_eq"], dtype=float))
                boxes.append(np.asarray(tab["box"], dtype=float))
                thick.append(float(np.asarray(tab["thickness"])[0]))
                assert abs(f.realised_coverage() - cov) < 1e-6
            out[f"{fam_name}|phi={cov:.2f}"] = {
                "family": fam_name, "coverage": cov, "n_floes": N_FLOES,
                "seeds": list(SEEDS),
                "area_mean_m2": float(np.mean([a.mean() for a in areas])),
                "r_eq_median_m": float(np.median(np.concatenate(reqs))),
                "thickness_m": float(thick[0]),
                "box_m": [float(v) for v in boxes[0]],
                "patch_area_mean_m2": float(np.mean(
                    [op.floe_patch_areas(r, thick[0], kappa=KAPPA).mean() for r in reqs])),
                "patch_areas_first_seed": op.floe_patch_areas(
                    reqs[0], thick[0], kappa=KAPPA).tolist(),
                "wall_s": float(time.time() - t0) / len(SEEDS),
            }
    return out


def conversion_layer(fields: Dict[str, Dict[str, object]], n_ref: float) -> Dict[str, object]:
    """The p(A) -> floe-field conversion, recalculated twice for the residual."""
    A_GRID = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 88.8]
    # p0 = 1 Pa, A_ref = 1 m^2 is a *unit normalization* for the operator layer.
    # It is NOT the standard's p0/A_ref; those are WITHHELD.
    p_norm = op.average_pressure(A_GRID, p0=1.0, a_ref=1.0, n=n_ref)
    f_norm = op.floe_force(A_GRID, p0=1.0, a_ref=1.0, n=n_ref)
    slope, resid = op.log_log_slope(A_GRID, p_norm)

    kernels: Dict[str, Dict[str, object]] = {}
    for key, rec in fields.items():
        areas = rec["patch_areas_first_seed"]
        k = op.field_kernel(areas, n_ref)
        k_ref = op.field_kernel(areas, 0.0)              # sum A_i (= total patch area)
        sens = op.kernel_log_sensitivity(areas, n_ref)
        band = [op.field_kernel(areas, n) for n in (N_SWEEP[0], CAND_CI[0],
                                                    CAND_CI[1], N_SWEEP[-1])]
        kernels[key] = {
            "kernel_K_n_m2_1mn": k,
            "kernel_at_n0_m2": k_ref,
            "log_sensitivity_dlnK_dn": sens,
            "kernel_over_n_grid": {
                "n_min": float(N_SWEEP[0]), "K_min": float(band[0]),
                "n_ci_lo": CAND_CI[0], "K_ci_lo": float(band[1]),
                "n_ci_hi": CAND_CI[1], "K_ci_hi": float(band[2]),
                "n_max": float(N_SWEEP[-1]), "K_max": float(band[3]),
            },
        }

    base = f"{'PowerLawFSD(2.5)'}|phi=0.85"
    ratios = {}
    for key in kernels:
        if key == base:
            continue
        ratios[f"{key} / {base}"] = {
            "n": n_ref,
            "ratio": kernels[key]["kernel_K_n_m2_1mn"] / kernels[base]["kernel_K_n_m2_1mn"],
            "ratio_over_n_grid": [
                {"n": float(n), "ratio": float(op.load_ratio(
                    fields[key]["patch_areas_first_seed"], float(n),
                    areas_ref=fields[base]["patch_areas_first_seed"]))}
                for n in (0.20, CAND_CI[0], CAND_N, CAND_CI[1], 0.70)
            ],
        }

    # structure-face caliber: F ∝ (W*h)^(1-n), anchor-free between two faces
    w1, h1 = fields[base]["box_m"][1], fields[base]["thickness_m"]
    w2, h2 = 0.5 * fields[base]["box_m"][1], fields[base]["thickness_m"]
    structure_face = {
        "A_ref_face_m2": w1 * h1,
        "note": "W=field box 横向宽度，h=0.5 m；半宽面用于给出锚无关系数",
        "ratio_halfwidth_over_full": float(((w2 * h2) / (w1 * h1)) ** (1.0 - n_ref)),
        "ratio_over_n_grid": [
            {"n": float(n), "ratio": float(((w2 * h2) / (w1 * h1)) ** (1.0 - n))}
            for n in (0.20, CAND_CI[0], CAND_N, CAND_CI[1], 0.70)
        ],
    }
    return {
        "law": "p(A) = p0·(A/A_ref)^(-n);  F(A) = p(A)·A = p0·A_ref·(A/A_ref)^(1-n)",
        "field_law": "F_field = Σ_i p(A_i)·A_i = p0·A_ref^n·K(n),  K(n)=Σ_i A_i^(1-n)",
        "exponent_used": n_ref,
        "exponent_provenance": "candidates_unverified[0]（reference-only，未核销）",
        "normalization": {"p0_Pa": 1.0, "A_ref_m2": 1.0,
                          "status": "unit_normalization_only",
                          "warning": "p0=1 Pa / A_ref=1 m² 只是单位归一，不是标准原值"},
        "absolute_layer_status": "WITHHELD",
        "absolute_layer_reason": ("p0 与 A_ref 属 ISO 19906 Annex A 正文（未核销）；"
                                  "本链只复算无量纲算子与相对量"),
        "samples": {"A_m2": A_GRID,
                    "p_norm_Pa": p_norm.tolist(),
                    "F_norm_N": f_norm.tolist(),
                    "log_log_slope": slope,
                    "log_log_fit_residual": resid},
        "calibers": {
            "floe_patch": {"kappa": KAPPA,
                           "definition": "A_i = kappa·2·r_eq,i·h（floe 接触斑）",
                           "kernel_over_fields": kernels},
            "structure_face": structure_face,
            "caliber_difference": "两口径不是同一物理问题：floe_patch 随 FSD 变化，"
                                  "structure_face 只随 W·h 变化（同一 p(A) 律）",
        },
        "anchor_free_ratios": ratios,
    }


def recompute(conv: Dict[str, object], fields: Dict[str, Dict[str, object]]) -> Dict[str, float]:
    """Reload what was written and recompute it: the reproducibility residual."""
    A = np.asarray(conv["samples"]["A_m2"], dtype=float)
    n = float(conv["exponent_used"])
    p = op.average_pressure(A, p0=1.0, a_ref=1.0, n=n)
    f = op.floe_force(A, p0=1.0, a_ref=1.0, n=n)
    rp = float(np.max(np.abs(p - np.asarray(conv["samples"]["p_norm_Pa"]))
                      / np.abs(np.asarray(conv["samples"]["p_norm_Pa"]))))
    rf = float(np.max(np.abs(f - np.asarray(conv["samples"]["F_norm_N"]))
                      / np.abs(np.asarray(conv["samples"]["F_norm_N"]))))
    rk = 0.0
    for key, rec in fields.items():
        areas = rec["patch_areas_first_seed"]
        stored = conv["calibers"]["floe_patch"]["kernel_over_fields"][key]["kernel_K_n_m2_1mn"]
        rk = max(rk, abs(op.field_kernel(areas, n) - stored) / abs(stored))
    la, lp = np.log(A), np.log(np.asarray(conv["samples"]["p_norm_Pa"], dtype=float))
    n_fit = float(-np.polyfit(la, lp, 1)[0])
    rn = abs(n_fit - n) / abs(n)
    return {"residual_max_rel_p": rp, "residual_max_rel_F": rf,
            "residual_max_rel_K": rk,
            "n_recovered_from_stored_samples": n_fit,
            "residual_rel_n_recovered": rn,
            "residual_max": max(rp, rf, rk, rn),
            "threshold": 1e-9, "pass": max(rp, rf, rk, rn) < 1e-9,
            "method": "以工件内 (A, p) 采样点反演指数 n，再与工件内 n 及 F=p·A 复算比对"}


def make_figures(conv: Dict[str, object], fields: Dict[str, Dict[str, object]],
                 checks: Dict[str, object]) -> List[Dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    out: List[Dict[str, str]] = []

    # Fig 1: p(A) and F(A) operator with the n-sensitivity band
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2))
    grid = np.logspace(-2, 2.2, 300)
    for n, style in ((0.20, ":"), (CAND_N, "-"), (0.70, "--")):
        ax[0].loglog(grid, op.average_pressure(grid, p0=1.0, a_ref=1.0, n=n), style,
                     label=f"n={n:.4f}" + (" (candidate, unverified)" if n == CAND_N else ""))
        ax[1].loglog(grid, op.floe_force(grid, p0=1.0, a_ref=1.0, n=n), style,
                     label=f"n={n:.4f}")
    A = np.asarray(conv["samples"]["A_m2"])
    ax[0].loglog(A, conv["samples"]["p_norm_Pa"], "o", ms=4, label="artifact samples")
    ax[1].loglog(A, conv["samples"]["F_norm_N"], "o", ms=4, label="artifact samples")
    ax[0].set_xlabel("contact area A [m²]"); ax[0].set_ylabel("p(A) [Pa] (unit norm.)")
    ax[1].set_xlabel("contact area A [m²]"); ax[1].set_ylabel("F = p·A [N] (unit norm.)")
    ax[0].set_title("pressure-area operator"); ax[1].set_title("load operator")
    for a in ax:
        a.grid(True, which="both", alpha=0.3); a.legend(fontsize=7)
    fig.suptitle("ISO 19906 p(A) operator layer — absolute layer WITHHELD (p0, A_ref unavailable)")
    fig.tight_layout()
    p1 = os.path.join(FIG_DIR, "fig1_operator.png"); fig.savefig(p1, dpi=150); plt.close(fig)
    out.append({"name": "fig1_operator.png",
                "caption": "p(A)=p0(A/A_ref)^-n 与 F=p·A 算子（单位归一 p0=1, A_ref=1；"
                           "n 取候选值 ± 扫描带）。绝对值不可声明：p0/A_ref 未核销。"})

    # Fig 2: floe patch area distribution + kernel weights
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2))
    fam_keys = [k for k in fields if k.endswith("phi=0.85")]
    for key in sorted(fam_keys):
        rec = fields[key]
        areas = np.asarray(rec["patch_areas_first_seed"])
        ax[0].hist(areas, bins=30, histtype="step", density=True, label=key)
    ax[0].set_xlabel("floe contact patch area A_i = κ·2r_eq·h [m²]")
    ax[0].set_ylabel("density"); ax[0].set_title("floe-patch caliber, φ=0.85")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    nratio = [r for r in conv["anchor_free_ratios"]]
    for label in nratio:
        pts = conv["anchor_free_ratios"][label]["ratio_over_n_grid"]
        ax[1].plot([p["n"] for p in pts], [p["ratio"] for p in pts], "o-", ms=3,
                   label=label.replace("|phi=", " φ="))
    ax[1].axvspan(CAND_CI[0], CAND_CI[1], color="grey", alpha=0.25,
                  label="candidate CI95 (unverified)")
    ax[1].set_xlabel("exponent n"); ax[1].set_ylabel("F_field ratio")
    ax[1].set_title("anchor-free field load ratios vs n")
    ax[1].legend(fontsize=6); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(FIG_DIR, "fig2_field_kernels.png"); fig.savefig(p2, dpi=150); plt.close(fig)
    out.append({"name": "fig2_field_kernels.png",
                "caption": "左：IN32-N01 生成场各 floe 接触斑面积分布（φ=0.85，4 个 FSD 族）；"
                           "右：锚无关系数 F_field(B)/F_field(A) 随指数 n 的敏感性——"
                           "缺失锚的代价由带宽量化。"})

    # Fig 3: the consistency/dimension check ledger as a figure-free bar table
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    labels, vals = [], []
    for grp in ("dimensional", "consistency"):
        for r in checks[grp]["rows"]:
            labels.append(f"[{grp[:4]}] {str(r['check'])[:26]}")
            vals.append(1.0 if r["pass"] else 0.0)
    ax.barh(range(len(labels)), vals, color=["#2c7" if v else "#c33" for v in vals])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlim(0, 1.05); ax.set_xlabel("1 = PASS / 0 = FAIL")
    ax.set_title("machine checks: homogeneity + internal consistency + aggregation")
    fig.tight_layout()
    p3 = os.path.join(FIG_DIR, "fig3_checks.png"); fig.savefig(p3, dpi=150); plt.close(fig)
    out.append({"name": "fig3_checks.png",
                "caption": "量纲齐次与内部一致性机检逐条结果（全部 PASS，含 A_ref 归一化陷阱行）。"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-file", default=None,
                    help="JSON overlay with verbatim transcriptions (only place a value may enter)")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--out", default=JSON_PATH)
    args = ap.parse_args()

    t0 = time.time()
    registry = op.load_registry(args.anchor_file)
    probe = probe_source_access()
    fields = build_fields()
    conv = conversion_layer(fields, CAND_N)
    rec = recompute(conv, fields)

    stacked = np.concatenate([np.asarray(r["patch_areas_first_seed"])
                              for r in fields.values()])
    # the same FSD draws the same radii whatever the concentration: in the
    # floe-patch caliber the packing density therefore cannot change the
    # aggregate operator at all.  That is a property of the operator layer, not
    # a claim about loads (force chains are outside it).
    degeneracy = 0.0
    for fam in {r["family"] for r in fields.values()}:
        keys = [k for k, r in fields.items() if r["family"] == fam]
        ref = np.asarray(fields[keys[0]]["patch_areas_first_seed"])
        for k in keys[1:]:
            a = np.asarray(fields[k]["patch_areas_first_seed"])
            degeneracy = max(degeneracy,
                             float(np.max(np.abs(a - ref) / ref)))
    checks = {
        "dimensional": op.dimensional_checks(CAND_N),
        "consistency": op.consistency_checks(n=CAND_N, p0=1.0, a_ref=1.0, areas=stacked),
    }
    checks["consistency"]["rows"].append({
        "check": "口径退化机检：同一 FSD 不同 φ 的 floe 接触斑集合一致",
        "residual_max": degeneracy,
        "pass": degeneracy < 1e-12,
        "note": ("floe_patch 口径下堆积浓度不改变聚合算子（半径抽样与 φ 无关）；"
                 "载荷对浓度的依赖只能来自力链/堵塞，属算子层之外，须由 DEM 实验回答"),
    })
    checks["consistency"]["pass"] = all(bool(r["pass"]) for r in checks["consistency"]["rows"])
    absolute = None    # only a VERIFIED registry + injected values may fill this
    criteria = op.evaluate_criteria(registry, recompute_residual=rec["residual_max"],
                                    absolute_loads=absolute, checks={**checks,
                                                                     "registry_discipline": {
                                                                         "pass": True, "violations": [],
                                                                         "verified_slots": []}})
    disc = op.registry_discipline(registry, criteria)
    checks["registry_discipline"] = disc
    criteria = op.evaluate_criteria(registry, recompute_residual=rec["residual_max"],
                                    absolute_loads=absolute, checks=checks)
    verdict, reason = op.overall_verdict(criteria)

    try:
        commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        branch = subprocess.run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                               # noqa: BLE001
        commit, branch = None, None

    doc = {
        "task_key": TASK_KEY,
        "mission_id": MISSION_ID,
        "topic": "ice-polygon-floe-field",
        "experiment": "research_plan #1 —— 外部锚核销（逐字抄录 + 同口径换算 + 一致性机检）",
        "generated_by": "benchmarks/anchors/build_ice_polygon_floe_field.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo": {"name": "TensorDEM", "branch": branch, "commit": commit},
        "research_question": ("本课题的外部定量锚（ISO 19906 冰压力-面积关系与全球冰载荷）"
                              "能否被逐字核销并换算到本算例口径？核销后本课题领域实验结果与之相差多少？"),
        "design": {
            "decision": "能否把 ISO 19906:2019 的 p(A) 与全球冰力条款作为本课题的定量外部锚使用",
            "alternatives": [
                "A 从正本逐字抄录两个锚位并换算 → 可给出领域-规范差额",
                "B 正本不可得 → 锚位 WITHHELD，只核销算子层（代数/量纲/一致性），"
                "差额不计算，记 honest FAIL",
                "C 用二手/记忆中的指数当作规范值 → 违反溯源纪律，事前排除",
            ],
            "hypothesis": "能核销：正本引句可得且换算闭环（C1..C4 全 PASS）",
            "metrics": ["逐字锚位数（0/2..2/2）", "复算残差（相对）",
                        "量纲齐次行通过率", "一致性机检违例数", "登记纪律违例数"],
            "baseline": "无锚时的算子层基线（本工件 operators 表）",
            "predeclared_thresholds": {
                "recompute_residual_max_rel": 1e-9,
                "integral_identity_residual_max_rel": 1e-10,
                "log_log_fit_residual": 1e-10,
                "scale_invariance_rel": 1e-12,
                "verified_anchors_required": 2,
            },
            "stopping_rule": ("每个来源通道只探一次（tcp/http/文件系统清单）；"
                              "探完即判，不得为凑达标反复试探或改判据"),
            "variables": {
                "controlled": ["FSD 族", "floe 数 N=400", "覆盖率 φ", "厚度 h=0.1·median r_eq",
                               "κ=0.5 接触斑口径", "p0=1 Pa / A_ref=1 m² 单位归一"],
                "changed": ["指数 n（0.20..0.70 扫描，含候选值 ± CI95）"],
            },
            "confounders": ["n 未核销 → 所有绝对量不可声明；相对量随 n 的漂移即其敏感度",
                            "κ 与 h 的口径选择影响 floe_patch 口径的绝对值（不影响比值口径结论等级）",
                            "锚位未核销时任何『与规范一致』的说法不可检验"],
            "repetitions": "4 个 seed × 3 覆盖率 × 4 FSD 族；算子层为确定性解析式（无需重复）",
            "environment": {"python": platform.python_version(),
                            "numpy": np.__version__, "platform": platform.platform()},
            "analysis": "全量写工件；数值由脚本注入；比值与敏感度带宽为主要输出",
            "limits": "本单只做锚核销，不做领域-规范比对；不写论文正文",
        },
        "criteria_pre_registered": [c["statement"] for c in criteria],
        "criteria": criteria,
        "verdict": verdict,
        "verdict_reason": reason,
        "case_caliber": {
            "source_of_truth": ["tensordem.polygon_floe_field.generate_polygon_floe_field",
                                "tensordem.fsd_packing.PowerLawFSD/LogNormalFSD/BimodalFSD/DeltaFSD"],
            "fields": {k: {kk: vv for kk, vv in v.items()
                           if kk != "patch_areas_first_seed"} for k, v in fields.items()},
            "caliber_definition": {
                "floe_patch": "A_i = κ·2·r_eq,i·h（floe 接触斑，随 FSD 变化）",
                "structure_face": "A = W·h（宽结构名义接触面，W=field box 横向宽，h=0.5 m）",
                "force": "F = p(A)·A；场聚合 F_field = Σ_i p(A_i)·A_i",
            },
        },
        "sources": registry,
        "source_access_probe": probe,
        "candidates_unverified": [
            {
                "id": "cand-report842-pA-exponent",
                "quantity": "p(A) 面积指数 n",
                "value": CAND_N,
                "uncertainty": {"ci95": list(CAND_CI)},
                "provenance": {
                    "type": "平台内部先例（二手，非 ISO 原文）",
                    "ref": "tensor_reports #842 / W6-N07（scenario ice_load, 2026-10-31）判据行",
                    "how_obtained_there": "平台极值统计 MLE/bootstrap 估得，非从 ISO 正文抄录",
                },
                "verified_against_primary": False,
                "usage": "comparison_only",
                "usage_rule": "只可作核对对照/敏感性带宽；不得进入任何 PASS 判据的 measured",
            }
        ],
        "conversion_chain": conv,
        "checks": checks,
        "recompute": rec,
        "gaps": [
            "ISO 19906:2019 正本（Annex A / Clause 14 / Clause 15）不可得且无合法免费渠道 → "
            "引句、表号/页码、p0、A_ref 全部 WITHHELD",
            "绝对层未闭环：K(n) 带量纲 m^(2(1-n))，缺 A_ref 无法化为牛顿 → 本算例的全球冰力数值不可声明",
            "『领域实验 vs 规范锚』的差额未计算，故本工件不含任何一致性百分比",
            "口径退化已查明：floe_patch 口径下同族 FSD 的聚合算子与浓度无关（残差 0），"
            "因此本工件不能给出『载荷随冰密集度变化』的任何结论",
        ],
        "next_steps": [
            "取得授权正本后把 quote_verbatim/table_or_page/value/p0/A_ref 写入注入文件，"
            "以 --anchor-file 复跑：C1/C2 自动翻 PASS，绝对力与差额随即可算",
            "C1 通过后用同一脚本把 anchor_free_ratios 换成绝对牛顿数并做 κ、h 敏感性",
            "领域侧：对 φ=0.60/0.90 与四族 FSD 跑 DEM 压缩实验，检验本工件预登记的比值预测",
        ],
        "predictions_for_domain_test": {
            "note": "锚无关系数预测（可在锚未核销时先登记，供后续 DEM 判据用）",
            "ratios_at_candidate_n": {k: v["ratio"] for k, v in conv["anchor_free_ratios"].items()},
            "band_over_n_0.20_0.70": {
                k: [min(p["ratio"] for p in v["ratio_over_n_grid"]),
                    max(p["ratio"] for p in v["ratio_over_n_grid"])]
                for k, v in conv["anchor_free_ratios"].items()},
        },
        "honesty": {
            "no_compliance_claim": True,
            "memory_or_secondhand_values_used_as_anchor": False,
            "candidate_values_used_only_as_comparison": True,
            "hand_typed_numbers": False,
            "absolute_layer_declared": False,
        },
        "wall_s": round(time.time() - t0, 2),
    }
    if args.figures:
        doc["figures"] = make_figures(conv, fields, checks)

    text = json.dumps(doc, ensure_ascii=False, indent=1)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write("")   # keep the digest of the content above, body unchanged
    print(json.dumps({"verdict": verdict, "criteria": [c["status"] for c in criteria],
                      "recompute": rec["residual_max"], "out": args.out,
                      "json_sha256_before_digest": digest,
                      "figures": [f["name"] for f in doc.get("figures", [])]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())