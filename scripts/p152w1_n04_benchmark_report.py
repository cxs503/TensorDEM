"""Figures + platform report for the P152W1-N04 repose benchmark (anchors A/B).

Reads the machine-generated artifact
``artifacts/P152W1-N04/repose_benchmark.json`` and produces

* ``artifacts/P152W1-N04/figures/fig1_deposits.png``   -- final deposits + flank fit
* ``artifacts/P152W1-N04/figures/fig2_calibration.png`` -- theta(mu_rolling) bisection
* ``artifacts/P152W1-N04/figures/fig3_reference_vs_sim.png`` -- ref vs got + bands

and posts the paper-level progress report (mission 14608).  Every number in the
markdown is interpolated from the artifact -- nothing is typed by hand.

Usage: ``PYTHONPATH=src python scripts/p152w1_n04_benchmark_report.py [--no-post]``
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "P152W1-N04"
BENCH = ART / "repose_benchmark.json"
FIG = ART / "figures"

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"


def f(value, fmt=".2f", dash="n/a"):
    if value is None:
        return dash
    try:
        v = float(value)
    except (TypeError, ValueError):
        return dash
    return dash if not math.isfinite(v) else format(v, fmt)


def pct(value, fmt="+.2f", dash="n/a"):
    out = f(value, fmt, dash)
    return dash if out == dash else out + " %"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_deposits(payload: dict) -> Path | None:
    pos = payload.get("figure_positions") or {}
    entries = [(k, v) for k, v in pos.items() if v]
    if not entries:
        return None
    fig, axes = plt.subplots(1, len(entries), figsize=(5.2 * len(entries), 4.2),
                             squeeze=False)
    for ax, (name, pts) in zip(axes[0], entries):
        arr = np.asarray(pts, dtype=float)
        row = payload["materials"][name]
        ax.scatter(arr[:, 0] * 1e3, arr[:, 2] * 1e3, s=4, alpha=0.35,
                   color="#2b6cb0", label="spheres (final)")
        ax.set_title(f"{name}: theta_r = {f(row['got_deg'])} deg "
                     f"(ref {f(row['reference_deg'])} deg)", fontsize=10)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("P152W1-N04 anchor B: final deposits of the fixed-nozzle pour "
                 "(protocol v2)", fontsize=11)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fig1_deposits.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_calibration(payload: dict) -> Path:
    cal = payload["calibration"]
    hist = [h for h in cal["bisection_history"] if h["theta_r_deg"] is not None]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if hist:
        mus = [h["mu_rolling"] for h in hist]
        ths = [h["theta_r_deg"] for h in hist]
        ax.plot(mus, ths, "o-", color="#2b6cb0", label="DEM bisection")
    ax.axhline(cal["target_deg"], color="#c53030", ls="--",
               label=f"experimental target {f(cal['target_deg'])} deg")
    ax.axvline(cal["mu_rolling_star"], color="#2f855a", ls=":",
               label=f"mu_rolling* = {f(cal['mu_rolling_star'], '.4f')}")
    row = payload["materials"].get(cal["fitted_on"])
    if row and row["got_deg"] is not None:
        ax.errorbar([cal["mu_rolling_star"]], [row["got_deg"]],
                    yerr=[row["got_std_deg"] or 0.0], fmt="s", color="#2b6cb0",
                    capsize=4, label="final mean ± std")
    ax.set_xlabel("rolling friction coefficient mu_rolling [-]")
    ax.set_ylabel("repose angle theta_r [deg]")
    ax.set_title(f"Shape-parameter calibration on {cal['fitted_on']} "
                 f"({cal['bisection_steps']} bisection steps)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fig2_calibration.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_reference(payload: dict) -> Path:
    names = list(payload["materials"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(names))
    refs = [payload["materials"][n]["reference_deg"] for n in names]
    gots = [payload["materials"][n]["got_deg"] for n in names]
    errs = [payload["materials"][n]["got_std_deg"] or 0.0 for n in names]
    ax.bar(x - 0.18, refs, width=0.36, color="#a0aec0", label="literature reference")
    ax.bar(x + 0.18, [g if g is not None else 0.0 for g in gots], width=0.36,
           yerr=errs, capsize=4, color="#2b6cb0", label="DEM (protocol v2)")
    for i, n in enumerate(names):
        row = payload["materials"][n]
        ax.annotate(f"{pct(row['relative_error']*100 if row['relative_error'] is not None else None)}",
                    (i + 0.18, (row["got_deg"] or 0.0) + (row["got_std_deg"] or 0.0)),
                    ha="center", va="bottom", fontsize=8)
        lo, hi = row["reference_band_deg"]
        ax.fill_between([i - 0.42, i + 0.42], lo, hi, color="#f6ad55", alpha=0.25,
                        lw=0)
    label = " / ".join(names)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("repose angle theta_r [deg]")
    ax.set_title(f"Reference vs simulation ({label}); band = literature range")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fig3_reference_vs_sim.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# markdown (six sections; all numbers from the artifact)
# ---------------------------------------------------------------------------

def build_markdown(payload: dict, figs: dict[str, str]) -> str:
    p = payload
    plan = p["plan"]
    cal = p["calibration"]
    neg = p["negative_control"]
    mats = p["materials"]
    prov = p["provenance"]
    names = list(mats)

    lines = []
    lines.append("# P152W1-N04 锚点 A/B 收口：预注册休止角基准（协议 v2，TensorDEM）\n")
    lines.append(f"- 节点：`{p['node']}` · 预注册 `{p['prereg']['path']}` "
                 f"(sha256 `{p['prereg']['sha256'][:16]}…`)\n"
                 f"- 工件：`artifacts/P152W1-N04/repose_benchmark.json` "
                 f"(schema `{p['schema']}`) · commit `{prov['git_commit'][:12]}`\n"
                 f"- 复现命令：`python -m tensordem.friction_surface --particles "
                 f"{plan['particles']} --bisection-steps {plan['bisection_steps']} "
                 f"--jobs {plan['jobs']} --out artifacts/P152W1-N04/repose_benchmark.json`\n")

    lines.append("## 1 案例介绍\n")
    lines.append(
        "休止角（angle of repose）是颗粒材料摩擦/滚动阻力的宏观指纹，也是 DEM 微参数标定的"
        "经典标定面。本节点把 TensorDEM 的摩擦接触面（切向 `F_t = min(mu F_n, k_t delta_t) "
        "+ c_t v_t`、滚动阻力矩）放到「散体从固定高度喷嘴倾泻成堆」的实验室类比中检验：\n"
    )
    lines.append("| 参数 | 值 |\n|---|---|")
    for key in ("particles", "bisection_steps", "seeds", "source_mode", "max_time"):
        lines.append(f"| `{key}` | `{plan[key]}` |")
    mat_cells = ", ".join("`%s` (mu=%s)" % (n, f(mats[n]["mu"])) for n in names)
    lines.append("| 材料 | %s |" % mat_cells)
    lines.append("| 负控 | mu = mu_rolling = 0，判据 theta_r < %s deg |"
                 % f(neg["criterion_theta_r_deg_max"]))
    lines.append(f"| 判据 | 相对误差 ≤ {f(mats[names[0]]['relative_error_max']*100)} %，"
                 f"文献带逐项给出 |\n")
    lines.append("预注册阈值、材料微参数（文献值，非本仓拟合）与实验带全部从 "
                 "`prereg/P152W1_N04_repose_protocol_v2_prereg.json` 读取，"
                 "SHA-256 写入工件，避免「跑完再调门限」。\n")

    lines.append("## 2 怎么做的\n")
    lines.append(
        "1. **协议 v2 倾泻**：喷嘴固定在 `drop_height`，撒料节奏 = 电荷穿出喷嘴区的自由落体时间"
        "（`nozzle_clear_time()`），因此不接受「抬高喷嘴」这类被本节点 §5.3 否定过的做法；"
        "沉积体填满喷嘴板层后停料且不复开。\n"
        "2. **测量**：上包络径向分箱 + 15–90 % 坡面窗口线性拟合（`heap_surface_angle`），"
        "与 v1 一字未改，因此本轮数字与历史数字可比。\n"
        f"3. **标定**：以 `{cal['fitted_on']}` 为标定材料，对单一形状参数 `mu_rolling` 在 "
        f"[{f(cal['bracket'][0], '.2f')}, {f(cal['bracket'][1], '.2f')}] 上做 "
        f"{cal['bisection_steps']} 步二分，目标 {f(cal['target_deg'])} deg；"
        "得到的 `mu_rolling*` 原封不动用于砂的样本外预测。\n"
        f"4. **规模/算力**：{plan['particles']} 球 × {len(plan['seeds'])} 种子 × "
        f"{plan['jobs']} 进程；墙钟 {f(prov['wall_seconds'], '.1f')} s"
        f"（{prov['machine']}，Python {prov['python']}，numpy {prov['numpy']}）。\n"
        "5. **工件**：每例的 `reached_rest` / `pour_closed` / `nozzle_z_max` / "
        "`bursts_released` 全部落盘，判定可由第三方脚本重算（判据来自预注册，非代码常量）。\n")

    lines.append("## 3 计算结果\n")
    lines.append("| 材料 | 参考 theta [deg] | 文献带 | 实测 theta [deg] | std | n | 相对误差 | 带内 | 状态 |\n"
                 "|---|---|---|---|---|---|---|---|---|")
    for n in names:
        r = mats[n]
        lines.append(f"| {n} | {f(r['reference_deg'])} | "
                     f"{f(r['reference_band_deg'][0])}–{f(r['reference_band_deg'][1])} | "
                     f"{f(r['got_deg'])} | {f(r['got_std_deg'])} | {r['n_runs']} | "
                     f"{pct(r['relative_error']*100 if r['relative_error'] is not None else None)} | "
                     f"{'是' if r['within_band'] else ('否' if r['within_band'] is not None else 'n/a')} | "
                     f"**{r['status']}** |")
    lines.append(f"\n负控（mu=mu_rolling=0）：theta_r = {f(neg['theta_r_deg'])} deg"
                 f"（判据 < {f(neg['criterion_theta_r_deg_max'])} deg，"
                 f"{p['verdicts'][0]['status']}），逐次 {[f(v) for v in neg['runs_deg']]}。\n")
    lines.append(f"`mu_rolling*` = {f(cal['mu_rolling_star'], '.4f')}；"
                 f"二分历史：{[f(h['theta_r_deg']) for h in cal['bisection_history']]} deg。\n")
    lines.append(f"总体判定：**{p['overall']}**（{sum(v['status']=='PASS' for v in p['verdicts'])}/"
                 f"{len(p['verdicts'])} 判据 PASS）。\n")

    lines.append("## 4 结果图\n")
    for key, cap in figs.items():
        lines.append(f"- `{key}`：{cap}")
    lines.append("")

    lines.append("## 5 注意点\n")
    lines.append(
        "1. 文献带不是本仓实测：玻璃珠 21–25 deg、石英砂 30–34 deg 取自常规干态单分散体系，"
        "属于外部基准，不能反过来当作本仓的标定数据。\n"
        "2. 单形状参数（`mu_rolling`）代替真实颗粒形状；砂是**样本外**预测，"
        "若失败即为形状近似的证伪，不作任何事后调参。\n"
        f"3. 粒子数 {plan['particles']} 是为算力选取，未做有限尺寸收敛研究；"
        "本轮不声明尺寸收敛性。\n"
        "4. 停料由几何（喷嘴板层被沉积体填满）决定，停料后不复开——若堆体尚未稳定而停料，"
        "测到的是「停料时」的表面角，`reached_rest` 字段逐例记录该风险。\n"
        "5. 5 个 `theta(mu)` 面点只用于图示趋势，不参与任何判据。\n")

    lines.append("## 6 误差判定\n")
    lines.append("| 判据 | 基准 | 实测 | 误差 | 阈值 | 是否达标 |\n|---|---|---|---|---|---|")
    for n in names:
        r = mats[n]
        ok = {"PASS": "**是**", "FAIL": "**否**", "WITHHELD": "未裁决"}[r["status"]]
        lines.append(f"| B/{n} | {f(r['reference_deg'])} deg 文献值 | {f(r['got_deg'])} deg | "
                     f"{pct(r['relative_error']*100 if r['relative_error'] is not None else None)} | "
                     f"≤ {f(r['relative_error_max']*100)} % | {ok} |")
    hit3 = [n for n in names if mats[n]["relative_error"] is not None
            and mats[n]["relative_error"] <= 0.03]
    lines.append(f"| A/负控 | theta_r < {f(neg['criterion_theta_r_deg_max'])} deg | "
                 f"{f(neg['theta_r_deg'])} deg | — | ≤ {f(neg['criterion_theta_r_deg_max'])} deg | "
                 f"{ {'PASS':'**是**','FAIL':'**否**','WITHHELD':'未裁决'}[p['verdicts'][0]['status']] } |")
    lines.append("")
    lines.append(f"**是否 <3%？** " + (
        "是：" + "、".join(hit3) + "。" if hit3 else
        "否：本轮没有任何材料达到 3 % 以内；"
        + "、".join(f"{n} = {pct(mats[n]['relative_error']*100 if mats[n]['relative_error'] is not None else None)}"
                   for n in names) + "。"
    ) + f"（判据线本身是 10 %，3 % 是更严的自检线；站在 10 % 门限上不等于满足 3 %。）\n")
    lines.append(f"> {p['honest_note']}\n")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-post", action="store_true")
    args = ap.parse_args(argv)
    payload = json.loads(BENCH.read_text())
    figs = {}
    for fn in (fig_deposits, fig_calibration, fig_reference):
        path = fn(payload)
        if path is not None:
            figs[path.name] = "见正文表格（程序化生成，数据来自工件）"
    md = build_markdown(payload, figs)
    doc = REPO / "docs" / "P152W1-N04_repose_benchmark.md"
    doc.write_text(md + "\n")
    print("doc:", doc)

    figures = []
    for name in figs:
        path = FIG / name
        if path.exists():
            figures.append({"name": name,
                            "caption": "P152W1-N04 锚点 A/B：程序化生成图（数据来自 "
                                       "repose_benchmark.json）",
                            "png_base64": base64.b64encode(path.read_bytes()).decode()})
    report = {
        "title": "P152W1-N04 锚点 A/B 收口：预注册休止角基准（固定喷嘴协议 v2，TensorDEM）",
        "summary": (
            f"{payload['plan']['particles']} 球固定喷嘴倾泻 DEM 基准：负控 "
            f"theta={f(payload['negative_control']['theta_r_deg'])} deg "
            f"({payload['verdicts'][0]['status']})；"
            + "；".join(
                f"{n} 实测 {f(mats['got_deg'])} vs 参考 {f(mats['reference_deg'])} deg "
                f"({pct(mats['relative_error']*100 if mats['relative_error'] is not None else None)}, "
                f"{mats['status']})" for n, mats in payload["materials"].items())
            + f"；mu_rolling* = {f(payload['calibration']['mu_rolling_star'], '.4f')}；"
            f"总体 {payload['overall']}。判据与阈值全部来自预注册文件。"),
        "scenario_key": "friction_contact_surface",
        "mission_id": 14608,
        "tags": ["TensorDEM", "摩擦接触", "休止角", "预注册基准", "DEM", "P152W1-N04"],
        "markdown": md,
        "figures": figures,
    }
    Path("/tmp/report.json").write_text(json.dumps(report))
    print(f"payload: {len(md)} chars, {len(figures)} figures")
    if args.no_post:
        return 0
    login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                        "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=login,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    token = body.get("access_token") or body.get("token")
    if not token:
        raise SystemExit(f"login did not return a token: {body}")
    req = urllib.request.Request(BASE + "/tensor/reports", data=json.dumps(report).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + token},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        print("report posted:", json.loads(resp.read()))
    return 0


if __name__ == "__main__":
    sys.exit(main())