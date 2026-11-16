#!/usr/bin/env python
"""Build the platform report payload for CASE-DEM-BIAXIAL-FRICTION.

All numbers are injected programmatically from a run artefact (zero hand
transcription).  The artefact is resolved through ``regen_doc.resolve_artefact``
(full ``artifacts/`` record first, tracked ``docs/data/`` distilled record
second), so the report can be rebuilt on a clean checkout.

Usage:
    PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/make_report.py \
        --job-id cjN --mission 22568 --out /tmp/report.json
    curl -s -X POST $TENSOR_API_BASE/tensor/reports -H "Authorization: Bearer $TOKEN" \
        -H 'Content-Type: application/json' -d @/tmp/report.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regen_doc as rg  # noqa: E402

ART = rg.FULL_ART
SENS = rg.FULL_SENS
OUT = "/tmp/report_m20440.json"


def load(path):
    with open(path) as f:
        return json.load(f)


def fig_stress_path(rec):
    sp = rec["stress_path"]
    eps = np.asarray(sp["axial_strain"])
    dev = np.asarray(sp["deviator"]) / 1e3
    dev_i = np.asarray(sp["deviator_internal"]) / 1e3
    peak = rec["peak"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eps, dev, "b-", lw=1.4, label="wall traction")
    ax.plot(eps, dev_i, "g--", lw=1.2, label="Love-Weber internal")
    ax.axvline(peak["axial_strain"], color="r", ls=":", alpha=0.7)
    ax.annotate(f"peak {peak['deviator']/1e3:.0f} kPa\n"
                f"phi = {rec['friction_angle']['phi_peak_wall_deg']:.2f} deg",
                xy=(peak["axial_strain"], peak["deviator"] / 1e3),
                xytext=(peak["axial_strain"] + 0.02, peak["deviator"] / 1e3),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="r"))
    ax.set_xlabel("axial strain [-]")
    ax.set_ylabel("deviator stress [kPa]")
    ax.set_title("Biaxial stress path (100 kPa confining)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return buf.getvalue()


def fig_phi_compare(rec, sens):
    j = rec["judgement"]
    phi_wall = rec["friction_angle"]["phi_peak_wall_deg"]
    phi_int = rec["friction_angle"]["phi_peak_internal_deg"]
    phi_ref = rec["friction_angle"]["phi_ref_deg"]
    tol = rec["friction_angle"]["tolerance"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [0, 1, 2]
    vals = [phi_ref, phi_wall, phi_int]
    labels = ["reference\nBolton (1986)\n38.0 deg",
              "TensorDEM wall\ntraction",
              "TensorDEM\ninternal (L-W)"]
    cols = ["#2196F3", "#FF9800", "#9C27B0"]
    ax.bar(xs, vals, color=cols, width=0.55, edgecolor="white")
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.4, f"{v:.2f}", ha="center", fontsize=10)
    lo, hi = phi_ref * (1 - tol), phi_ref * (1 + tol)
    ax.axhspan(lo, hi, color="green", alpha=0.15,
               label=f"pre-declared tolerance +/-{tol*100:.0f}%")
    ax.axhline(phi_ref, color="k", lw=0.8, ls="-")
    verdict = "PASS" if j["passed"] else "FAIL"
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("peak friction angle [deg]")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title(f"verdict: {verdict}  (err {j['relative_error']*100:.2f}%)",
                 color="green" if j["passed"] else "red", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return buf.getvalue()


def fig_sensitivity(rec, sens):
    if sens is None:
        return None
    p100 = rec["friction_angle"]["phi_peak_wall_deg"]
    p200 = sens["friction_angle"]["phi_peak_wall_deg"]
    phi_ref = rec["friction_angle"]["phi_ref_deg"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([100, 200], [p100, p200], "o-", color="#FF9800", lw=1.6,
               label="TensorDEM phi_peak")
    ax.axhline(phi_ref, color="k", lw=0.9, label="reference 38.0 deg")
    ax.axhspan(phi_ref * 0.92, phi_ref * 1.08, color="green", alpha=0.15,
              label="+/-8% band")
    ax.set_xlabel("confining pressure sigma_3 [kPa]")
    ax.set_ylabel("peak friction angle [deg]")
    ax.set_title(f"pressure independence: shift {p200-p100:+.2f} deg "
                 f"({(p200-p100)/p100*100:+.2f}%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description="CASE-DEM-BIAXIAL-FRICTION report")
    ap.add_argument("--artefact", default=None)
    ap.add_argument("--sens", default=None)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--mission", type=int, default=20440)
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--accept-recorded-config", action="store_true")
    args = ap.parse_args(argv)

    art_path, rec, kind = rg.resolve_artefact(args.artefact)
    divergences = rg.provenance_mismatches(rec["config"])
    if divergences and not args.accept_recorded_config:
        print("PROVENANCE GATE: record is not the pre-declared ANCHOR_CONFIG:\n  "
              + "\n  ".join(divergences), file=sys.stderr)
        return 2
    if divergences:
        print("WARNING: rendering a report for a NON-anchor configuration "
              "(drift banner included):\n  " + "\n  ".join(divergences),
              file=sys.stderr)
    sens_path = None
    for cand in (args.sens, SENS, rg.TRACKED_SENS):
        if cand and os.path.exists(cand):
            sens_path = cand
            break
    sens = load(sens_path) if sens_path else None

    j = rec["judgement"]
    peak = rec["peak"]
    sd = rec["stress_definitions"]
    cfg = rec["config"]
    spec = rec["specimen"]
    dim = rec["dimensionless"]
    phi_wall = rec["friction_angle"]["phi_peak_wall_deg"]
    phi_int = rec["friction_angle"]["phi_peak_internal_deg"]
    phi_ref = rec["friction_angle"]["phi_ref_deg"]
    tol = rec["friction_angle"]["tolerance"]
    err_wall = j["relative_error"] * 100.0
    err_int = abs(phi_int - phi_ref) / phi_ref * 100.0
    verdict = "PASS" if j["passed"] else "FAIL"
    mu = cfg.get("friction_coefficient")
    fb = cfg.get("freeze_bond_fraction")
    inertial = float(dim["inertial_number"])
    regime = rg.inertial_regime(inertial)
    job = f"集群作业 {args.job_id}" if args.job_id else "本工作树运行"

    f1 = fig_stress_path(rec)
    f2 = fig_phi_compare(rec, sens)
    f3 = fig_sensitivity(rec, sens)

    if sens is not None:
        sphi = sens["friction_angle"]["phi_peak_wall_deg"]
        dphi = sphi - phi_wall
        sens_row = (f"| 围压敏感性 (c=0 压力无关) | 100 kPa: {phi_wall:.2f} deg | "
                    f"{sens['config']['confining_stress']/1e3:.0f} kPa: {sphi:.2f} deg | "
                    f"偏移 {dphi:+.2f} deg ({dphi/phi_wall*100:+.2f}%) | "
                    f"{'是 (<1 deg)' if abs(dphi) < 1.0 else '否 (>=1 deg)'} |")
    else:
        sens_row = ("| 围压敏感性 (c=0 压力无关) | 100 kPa: "
                    f"{phi_wall:.2f} deg | 无该工件 | -- | 未跑 |")

    md = f"""# CASE-DEM-BIAXIAL-FRICTION: 双轴剪切摩擦角算例（生成式文档与算例定义一致性）

> **CONFIGURATION DRIFT（先读）**：本报告由一条配置与当前预声明 `ANCHOR_CONFIG` 不一致的运行记录渲染：
> {" ; ".join(f"`{d}`" for d in divergences) if divergences else "（无：记录即预声明锚点配置）"}。
> 结论归属**记录配置**，不是预声明锚点；锚点在本轮**未重跑**（集群调度 65 分钟未派发，如实声明）。

## 1. 案例介绍（物理背景 / 几何 / 参数）

平面应变双轴压缩 DEM 算例：{"无黏性" if not fb else "冻结合金"}圆盘试样，顶部压头位移驱动，侧壁伺服恒定围压
sigma_3 = {cfg['confining_stress']/1e3:.0f} kPa。峰值偏应力定义动 mobilised Mohr-Coulomb 摩擦角
sin(phi) = (sigma1 - sigma3)/(sigma1 + sigma3)（c = 0）。

**预声明基准**：{rec['reference']}。
**预声明验收**：|phi - {phi_ref:.0f} deg| / {phi_ref:.0f} deg < {tol*100:.0f}%（总账 CASE-DEM-BIAXIAL-FRICTION）。

| 参数 | 值 |
|---|---|
| 试样 | {spec['columns']}x{spec['rows']} 间距 = {spec['particles']} 圆盘, 厚 1 |
| 颗粒半径 | {spec['radius']} m |
| 抖动 | {spec['jitter']*100:.1f}% |
| 冻结合金比例 f_b | {fb}（键数 {spec['bonds']}） |
| 壁摩擦 mu_wall | {cfg['wall_friction']}（无摩擦侧壁, OSS 惯例） |
| 颗粒间摩擦 mu | {mu}（arctan(mu) = {math.degrees(math.atan(mu)):.1f} deg = Rowe 临界态预测） |
| 围压 sigma_3 | {cfg['confining_stress']/1e3:.0f} kPa |
| 固结步数 | {cfg['consolidation_steps']} |
| 轴向应变率/目标 | {cfg['strain_rate']} s^-1 / {cfg['strain_target']} |
| 惯性数 I | {inertial:.4f}（{regime}） |

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. 抖动三角晶格铺样{"（无冻结键，对应无黏性砂）" if not fb else f"（冻结键 fraction {fb}）"}；
2. {cfg['consolidation_steps']} 步各向同性固结（侧壁伺服 {cfg['confining_stress']/1e3:.0f} kPa）；
3. 顶压头恒应变率剪切至 {cfg['strain_target']*100:.0f}% 轴向应变；
4. 双应力定义：壁面牵引（sigma_v = -F_top/(W*t), sigma_h = 0.5(-F_l+F_r)/(H*t)）
   与 Love-Weber 内部 virial 应力，两者独立可审计；
5. 峰值偏应力处取 phi；验收算术在跑前固定并由
   `tests/test_case_dem_biaxial_friction.py` 钉扎（解析值 + 边界 + 工件自洽）。

工具链：TensorDEM（mu={mu}，其余材料常数取 RidgeMaterial 默认；本算例 runner 不落盘
kn/kt/rho，报告中如实标注为未记录而非手写猜测），{job}（TENSORDEM_THREADS=4，防 128 核超订减速）。

## 3. 计算结果（数字表，程序化注入）

| 量 | 参考 | 壁面牵引 | 内部 L-W | 误差(壁) | 误差(内) |
|---|---|---|---|---|---|
| phi_peak [deg] | {phi_ref:.1f} | {phi_wall:.2f} | {phi_int:.2f} | {err_wall:.2f}% | {err_int:.2f}% |
| sigma_1 峰值 [kPa] | -- | {peak['sigma_v']/1e3:.1f} | {sd['sigma_v']/1e3:.1f} | -- | -- |
| sigma_3 峰值 [kPa] | {cfg['confining_stress']/1e3:.0f} | {peak['sigma_h']/1e3:.1f} | {sd['sigma_h']/1e3:.1f} | {(peak['sigma_h']-cfg['confining_stress'])/cfg['confining_stress']*100:.1f}% | {(sd['sigma_h']-cfg['confining_stress'])/cfg['confining_stress']*100:.1f}% |
| 峰值偏应力 [kPa] | -- | {peak['deviator']/1e3:.1f} | {(sd['sigma_v']-sd['sigma_h'])/1e3:.1f} | -- | -- |
| 峰值轴向应变 | -- | {peak['axial_strain']:.4f} | -- | -- | -- |
| 剪胀角 [deg] | -- | {peak['dilation_angle_deg']:.2f} | -- | -- | -- |

应力定义闭合：sigma_v 差 {abs(sd['sigma_v_relative_difference'])*100:.1f}%，
sigma_h 差 {abs(sd['sigma_h_relative_difference'])*100:.1f}%（无摩擦侧壁只承法向牵引，
内部 virial 含边界层接触力，属已知小试样效应，如实报告）。

## 4. 结果图

图 1：应力路径（偏应力-轴向应变，壁面 vs 内部）；图 2：摩擦角对比（含容差带与判定）；
{"图 3：围压敏感性（c=0 压力无关性检验）。" if f3 is not None else "（无数值敏感性工件，第三图未生成——如实声明）"}

## 5. 注意点（坑 / 局限 / 适用范围）

- **生成式文档缺陷（本单修复）**：`regen_doc.py` 曾把 Love-Weber 内部应力按 Pa 直接写进
  kPa 列（sigma_1 显示 453393.6 kPa，物理上不可能的 1000 倍），并把材料行硬编码为
  「mu=0.55 (RidgeMaterial default)」「rho=917 kg/m^3 (ice)」——与算例实跑的 mu={mu} 相矛盾；
  惯性数一句话也硬编码声称 I < 0.01，而实跑 I = {inertial:.4f}。三处都会让「零手抄」的文档
  对不上它自己的工件。现已全部改为由工件/ANCHOR_CONFIG 计算，并加配置指纹门与 `--check`
  漂移门（提交的 md 必须与重新生成结果逐字一致，否则 CI 失败）。
- **工件不随 main 走**：`artifacts/` 被 PR 门禁过滤，历史 md 因此长期是「无法重建」的孤证；
  现落盘 4 KB 蒸馏记录 `docs/data/CASE-DEM-BIAXIAL-FRICTION_anchor_run.json`（丢弃 151 kB
  逐点应力路径），干净检出即可重建并校验文档。
- **惯性数**：I = {inertial:.4f}，{regime}。
- **壁面摩擦偏置**：粗糙壁（mu_wall>0）会把 sigma_1 拉低（侧壁分担竖向剪力），本例 mu_wall={cfg['wall_friction']} 消除该偏置。
- **适用范围**：预声明 sigma_3 = 100 kPa、f_b = {fb} 的试样；跨围压、跨 f_b 外推未被本算例支持。
- 更大试样（32x64）尺寸收敛未跑，如实声明未做。

## 6. 误差判定（逐项，显式回答 <3%）

| 关键量 | 基准 | 实测 | 误差 | <3%? |
|---|---|---|---|---|
| phi_peak (壁面) | {phi_ref:.1f} deg (Bolton 1986) | {phi_wall:.2f} deg | {err_wall:.2f}% | {'是' if err_wall < 3 else '否'} |
| phi_peak (内部) | {phi_ref:.1f} deg | {phi_int:.2f} deg | {err_int:.2f}% | {'是' if err_int < 3 else '否'} |
{sens_row}

**判定**：预声明验收门 = {tol*100:.0f}%（总账容差）。壁面牵引 phi 误差 {err_wall:.2f}%
{'< ' + f'{tol*100:.0f}% -> PASS' if j['passed'] else '>= ' + f'{tol*100:.0f}% -> FAIL'}；
对 3% 更严口径：壁面 {'<3% 通过' if err_wall < 3 else '不满足 3%（但满足预声明 8% 门）'}，
内部应力 {'<3% 通过' if err_int < 3 else '不满足 3%'}。
"""
    if sens is not None:
        sphi = sens["friction_angle"]["phi_peak_wall_deg"]
        dphi = sphi - phi_wall
        md += (f"\n围压敏感性复跑：phi_peak = {sphi:.2f} deg，相对 100 kPa 锚点偏移 "
               f"{dphi:+.2f} deg（{dphi/phi_wall*100:+.2f}%），"
               + ("在 c=0 压力无关性预期内（<1 deg）。\n" if abs(dphi) < 1.0 else
                  "超出 c=0 的 <1 deg 预期，压力无关性检验不通过——如实报告为未解释的"
                  "试样尺度/边界效应，不做外推。\n"))

    payload = {
        "title": (f"CASE-DEM-BIAXIAL-FRICTION 双轴剪切摩擦角：生成式文档与本算例定义"
                  f"一致化（phi={phi_wall:.2f} deg, {verdict}）"),
        "summary": (f"双轴剪切锚点 phi_peak = {phi_wall:.2f} deg "
                    f"(ref {phi_ref:.0f} deg, err {err_wall:.2f}%) -> {verdict}；"
                    f"内部应力 {phi_int:.2f} deg (err {err_int:.2f}%)。"
                    f"修复 regen_doc.py 三处自相矛盾（内部应力 Pa 混入 kPa 列、"
                    f"mu/rho 硬编码与实跑 mu={mu} 不符、惯性数阈值硬编码），"
                    "加配置指纹门 + --check 漂移门 + 4 KB 蒸馏工件使文档可重建可校验。"),
        "scenario_key": "CASE-DEM-BIAXIAL-FRICTION",
        "mission_id": args.mission,
        "tags": ["TensorDEM", "biaxial", "friction-angle", "case-pool",
                 "generated-doc", "provenance", "honest-fail"],
        "markdown": md,
        "figures": [
            {"name": "stress_path.png",
             "caption": "应力路径：偏应力-轴向应变（壁面牵引 vs Love-Weber 内部），红点=峰值",
             "png_base64": base64.b64encode(f1).decode()},
            {"name": "phi_compare.png",
             "caption": f"摩擦角对比：ref {phi_ref:.1f} vs 壁面 {phi_wall:.2f} / 内部 {phi_int:.2f} deg，绿带=±{tol*100:.0f}% 容差",
             "png_base64": base64.b64encode(f2).decode()},
        ] + ([{"name": "sensitivity_sigma3.png",
               "caption": "围压敏感性：100 vs 200 kPa 的 phi_peak（c=0 压力无关性）",
               "png_base64": base64.b64encode(f3).decode()}
           ] if f3 is not None else []),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"wrote {args.out}; artefact={os.path.relpath(art_path, REPO)} "
          f"({kind}); verdict={verdict}; sens={'yes' if sens else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
