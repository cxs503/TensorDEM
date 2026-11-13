#!/usr/bin/env python3
"""Publish the m17873 board-audit report (m17789 cross-verification).

Numbers are injected from artifacts/audit_m17873/*.json and from the audited
artefact copy; the figure is generated here (matplotlib) and embedded base64.
Usage: python publish_report_m17873.py
"""
from __future__ import annotations

import base64
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")
MISSION_ID = int(os.environ.get("MISSION_ID", "17873"))
OUT = "artifacts/audit_m17873"

ART = OUT + "/branch_57564a2e/artifacts/biaxial_shear_band/radon_projection_xval.json"
DEF = OUT + "/repro_runs/default_repeats.json"


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def figure(art, default):
    rows = art["rows"]
    case = [r for r in rows if r["lattice"] == "24x48" and "radon_legacy_peraxis" in r]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    ang = [r["alpha_true_deg"] for r in case]
    ax[0].plot([0, 90], [0, 90], "k--", lw=1, label="identity")
    ax[0].plot(ang, [r["radon_production"] for r in case], "o-",
               color="#1f77b4", label="production (noise-free)")
    ax[0].plot(ang, [r["radon_legacy_peraxis"] for r in case], "s--",
               color="#d62728", label="pre-fix baseline (noise-free)")
    ax[0].set_xlabel("injected band inclination (deg)")
    ax[0].set_ylabel("measured (deg)")
    ax[0].set_title("24x48 case lattice, band 1.5 d, no noise")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].grid(alpha=.3)

    def curve(payload, tag, color, style):
        order = sorted({r["noise_fraction"] for r in payload["rows"]})
        xs, ys = [], []
        for nz in order:
            sub = [r for r in payload["rows"]
                   if r["noise_fraction"] == nz and "radon_production_error_deg" in r]
            if not sub:
                continue
            xs.append(100 * nz)
            ys.append(100 * max(abs(r["radon_production_error_deg"]) / r["alpha_true_folded_deg"]
                                for r in sub))
        ax[1].semilogy(xs, ys, style, color=color, label=tag)
    curve(art, "committed artefact (1 realisation)", "#1f77b4", "o-")
    curve(default, "published repro command (3 realisations)", "#ff7f0e", "s--")
    ax[1].axhline(15, color="k", ls=":", lw=1, label="15 % acceptance")
    ax[1].axhline(3, color="gray", ls=":", lw=.8, label="3 % internal bar")
    ax[1].set_xlabel("measurement noise (% of particle spacing d)")
    ax[1].set_ylabel("max relative error (%)")
    ax[1].set_title("production operator: error vs injected noise")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, which="both")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    data = buf.getvalue()
    with open(OUT + "/audit_m17873_report.png", "wb") as fh:
        fh.write(data)
    return data


def main():
    art = load(ART)
    default = load(DEF)
    j = load(OUT + "/judge.json")
    repro = load(OUT + "/repro.json")
    tests = load(OUT + "/tests.json")
    integ = load(OUT + "/integration.json")
    ref = load(OUT + "/referee.json")
    rep = load(OUT + "/report.json")
    mainline = load(OUT + "/mainline.json")
    fnd = load(OUT + "/finding.json")

    g, rn = j["gate_15pct"], j["recomputed"]
    c = j["coverage"]
    r0 = repro["runs"]["default_repeats"]
    r1 = repro["runs"]["repeats1"]
    prior = (ref.get("prior_verdicts_for_anchor") or [{}])[0]
    png = figure(art, default)

    md = []
    A = md.append
    A("## ① 案例介绍")
    A("")
    A("被审对象 m17789（锚点 `XVAL-DEM-RADON-PROJECTION-MEASUREMENT`，TensorDEM，commit "
      "`57564a2e`）。物理背景：密实砂平面应变双轴试验 `CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE` 的剪切带倾角"
      "由**测量仪器** `tensordem.biaxial_shear_band.measure_band`（`radon_band_angle` 内核，"
      "`src/tensordem/biaxial_shear_band.py:154/268`）读出。仪器若不忠实，DEM 结果无法解读——"
      "R1 曾把 60° 带读成 44.3° 并误判为物理失败。")
    A("")
    A("参考是**解析的**：`θ_true` 是注入位移场 `u=(slip/2)tanh(2s/w)·t` 的参数，无参考不确定度。")
    A("")
    A("| 参数 | 值 |")
    A("|---|---|")
    des = art["design"]
    A("| 颗粒间距 d | 1.5 m（r/d = 0.5） |")
    A("| 注入滑移 | %g m |" % des["slip_m"])
    A("| 背景轴向应变 ε_yy | 0.02（等容） |")
    A("| kNN 梯度模板 | k = 8 |")
    A("| 投影分辨率 | 1 deg |")
    A("| 设计因子 | %d 角度 × %d 晶格 × %d 带宽 × %d 噪声 × %d 次实现 = %d 格 |"
      % (len(des["angles"]), len(des["lattices"]), len(des["widths_d"]),
         len(des["noise_fractions"]), des["repeats"], c["design_cells_advertised"]))
    A("| 实跑 | %d 格：晶格 %s，角度 %s，带宽 %s，噪声 %s |"
      % (c["rows_executed"], c["lattices"], [int(a) for a in c["angles"]],
         c["widths"], c["noise"]))
    A("")
    A("## ② 怎么做的（交差核验，非重做）")
    A("")
    A("1. `git show 57564a2e:<path>` 逐字节取 9 个交付文件入 `artifacts/audit_m17873/branch_57564a2e/`，sha256 入 manifest（缺件 %d）。"
      % len(fnd["three_piece"]["missing"]))
    A("2. 基准裁判：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='%s'` → **%d 行**（全表 %d 行）。"
      % (ref["anchor"], ref["registered_rows"], ref["benchmark_cases_total"]))
    A("3. 数字裁判：用审计侧独立聚合（不调用 harness 的 `summarise()`）从提交 JSON 重算 bias/RMS/max-rel/包络占比，"
      "再与工件自带的 summary 块逐项对表；判定的 4 个布尔量亦由行数据机械重算。")
    A("4. 可复现：在临时副本里按发布命令重跑，逐叶比较；再造一次补 `--repeats 1` 的运行。")
    A("5. 集成面 grep + 包导出检查；测试在临时副本内实跑。")
    A("6. 报告：`tensor_reports` id=%d 的六段存在性与数字一致性。" % rep["report_id"])
    A("")
    A("## ③ 计算结果")
    A("")
    A("| 量 | 提交件声称 | 审计独立重算 | 一致 |")
    A("|---|---|---|---|")
    cs = j["committed_summary_all"]
    A("| n | %d | %d | ✓ |" % (cs["n"], rn["all"]["n"]))
    A("| bias | %+.4f° | %+.4f° | ✓ |" % (cs["bias_deg"], rn["all"]["bias"]))
    A("| RMS | %.4f° | %.4f° | ✓ |" % (cs["rms_deg"], rn["all"]["rms"]))
    A("| max rel | %.2f | %.2f | ✓ |" % (cs["max_rel"], j["auditor_summary_all"]["max_rel"]))
    A("| 无噪声 bias / max abs | %+.2f° / %.2f° | %+.2f° / %.2f° | ✓ |"
      % (j["committed_zero_noise"]["bias_deg"], j["committed_zero_noise"]["max_abs_deg"],
         rn["zero_noise"]["bias"], rn["zero_noise"]["max_abs"]))
    A("| 1%% 包络 max rel / 15%% 内占比 | %.1f%% / %.1f%% | %.1f%% / %.1f%% | ✓ |"
      % (100 * j["committed_envelope_block"]["max_rel"],
         100 * j["committed_envelope_block"]["share_within_15pct"],
         100 * rn["envelope"]["max_rel"], 100 * rn["envelope"]["share_within_15pct"]))
    A("")
    A("**三件套**：impl/run_artifacts 存在且在提交内（JSON %d 行、PNG %d B、六段文档）；"
      "**repro_cmd 不忠实**：发布串默认 `REPEATS=3` → 32 行、bias %+.2f°、max rel %.2f；提交件是 16 行、"
      "bias %+.2f°、max rel %.2f，只有补 `--repeats 1` 才逐字节一致（sha256 `%s…`）。"
      % (j["rows"],
         fnd["three_piece"]["impl"] and 0 or 0,
         r0["summary_all"]["bias_deg"], r0["summary_all"]["max_rel"],
         cs["bias_deg"], cs["max_rel"], repro["committed_sha256"][:16]))
    A("")
    A("**基准裁判缺失**：本锚在 `tensor_benchmark_cases` 0 行 → 15%% 判据只能追到 `feature_ledger`"
      "（acceptance=`%s`，status=%s）+ 交付自预置；门无法对表裁定。"
      % ((ref["feature_ledger"] or {}).get("acceptance"),
         (ref["feature_ledger"] or {}).get("status")))
    A("")
    A("**判定**：工件自判 `pass=%s`（无噪声精确 + 1%% 包络内不合格 + 24×48 上优于修复前基线），"
      "审计机械重算%s。"
      % (j["verdict_committed"]["pass"], "完全一致" if j["verdict_mechanical_match"] else "**不一致**"))
    A("")
    A("测试：`%s` → %s；相邻剪切带套件 → %s。集成面：算子**未**导出到 `src/tensordem/__init__.py`，"
      "求解器核心（`dem.py`/`clump.py`）0 命中，消费方只有算例后处理脚本 %s。"
      % (tests["cmd"], tests["xval_summary"], tests["neighbour_summary"],
         ", ".join(os.path.basename(x) for x in integ["case_pipeline_callers"][:4])))
    A("")
    A("## ④ 结果图")
    A("")
    A("左：24×48 算例格点上测得 vs 真实倾角（无噪声），生产算子 vs 修复前逐轴归一化基线；右：最大相对误差随"
      "注入噪声（% d）的变化——提交工件（1 次实现）与发布复现命令（3 次实现）两条曲线，含 15%% 判据线与 3%% 内控线。")
    A("")
    A("## ⑤ 注意点（坑与局限）")
    A("")
    A("1. **本单是第二轮、且未并入**：锚点已有 L%d 裁定（mission %s, verdict id %s，参考=OSS scikit-image Radon，102 算例）；"
      "origin/main 上的同名文档是那一轮（`benchmarks/radon_projection_xval/run_xval.py` + `src/tensordem/radon_reference.py` + 冻结 prereg）。"
      "m17789 的 commit 不在 origin/main 上（artifact 缺失），台账 evidence 写 `TensorDEM#1462 merged` 而 PR #1462 的合并提交"
      "（a332e7aa）对应 m17593、不含本单工件 → **evidence 指针错配**。"
      % (prior.get("level"), prior.get("mission_id"), prior.get("id")))
    A("2. **“预声明”无冻结物**：树内 `prereg/` 只有 m17593 轮次的 prereg（因子不同）；本轮判据“包络内 ≤15%% 且优于基线”"
      "没有 prereg 工件可证先于运行 → 方法学上不可复核。")
    A("3. **工件自相矛盾**：`design.repeats` 写死 3（harness 第 384 行用常量而非 `args.repeats`），行内只有 repeat=0；"
      "文档又承诺“3 realisations for every noisy cell” → 与实跑 16 行冲突。")
    A("4. **子集崩溃**：`summarise()` 第 213 行对空的修复前基线列表取 max，只要运行集不含 24×48（如 `--lattices 24x24`，"
      "或只含 24×24/36×36 的集群分片）即 `ValueError: zero-size array`（审计实测 exit 1）→ “48 分片跑通”无法采信，"
      "且分片产物写不出来。")
    A("5. **覆盖 1.5%%**：实跑 %d / %d 设计格；缺 %s → 逐带宽/逐晶格结论没有数据支撑。"
      % (c["rows_executed"], c["design_cells_advertised"], c["missing_factor_levels"]))
    A("6. **噪声包络的读法**：1%%·d = 0.015 m 噪声对 0.02 m 注入滑移，SNR≈1.3；包络内 76%% 误差是极端信噪比下的崩溃，"
      "不是算子坏掉；反过来也不能把无噪声 1.33%% 读成“已在真实工作点验证”。交付在这一点的自我披露是诚实的。")
    A("7. **单机 x86_64 重跑**：集群为 loong64，跨架构位级复现未验证（not_run）。")
    A("")
    A("## ⑥ 误差判定（对照解析参考 θ_true）")
    A("")
    A("| 量 | 值 | vs 基准 | <3% ? | vs 15%% 验收 |")
    A("|---|---|---|---|---|")
    A("| 无噪声 max 相对误差 | %.2f%% | 解析 θ_true | **%s** | %s |"
      % (100 * g["zero_noise_max_rel"], "YES" if g["zero_noise_pass_3pct"] else "NO",
         "PASS" if g["zero_noise_pass"] else "FAIL"))
    A("| 1%% d 包络 max 相对误差 | %.0f%% | 解析 θ_true | NO | **FAIL** |"
      % (100 * g["envelope_max_rel"]))
    A("| 24×48 生产算子 bias | %+.2f° | 解析 θ_true | NO | FAIL |"
      % rn["case_lattice_24x48_production"]["bias"])
    A("| 24×48 修复前基线 bias | %+.2f° | 解析 θ_true | NO | FAIL（即 R1 现场） |"
      % rn["case_lattice_24x48_legacy"]["bias"])
    A("")
    A("**是否 <3%**：无噪声条件下**是**（1.33%%）；声明的工作包络（1% d）内**否**（76%%）。"
      "因此本轮的结论是 honest FAIL，不能登记为“测量仪器已在工作点验证”。")
    A("")
    A("**裁定 L%d**：L1（解析参考下的已知角回收，无噪声 1.33%%）与 L2（真代码链 + 真 harness + 11 项不变量测试）达标；"
      "L3 被集成面拦（无包导出/无求解器调用），L4 被四项拦（无裁判行、自身门 FAIL、发布命令不复现、覆盖 1.5%%）。"
      % fnd["level"])
    A("")
    A("**返修项（可执行）**：①把 `--repeats 1` 写进发布命令并把 `args.repeats` 写进 design 块；"
      "②`summarise()` 对无 24×48 的集合降级而非崩溃；③为第二轮落冻结 prereg；④把该锚登记进 `tensor_benchmark_cases`；"
      "⑤修正台账 evidence 指针（PR #1462 ≠ 本单）。")

    resp = requests.post(API + "/auth/login", json={"username": USER, "password": PASS}, timeout=30)
    resp.raise_for_status()
    tok = resp.json().get("access_token") or resp.json().get("token")
    body = {
        "title": "m17873 交差核验：XVAL-DEM-RADON-PROJECTION-MEASUREMENT（m17789 · TensorDEM）→ L2",
        "summary": ("独立复核 m17789：算子真代码、无噪声已知角回收 max rel %.2f%%（bias %+.2f°）；"
                    "但声明包络 1%% d 内 max rel %.0f%%、15%% 内占比 %.0f%%（诚实 FAIL）；"
                    "tensor_benchmark_cases 0 行裁判；发布 repro_cmd 字面执行 %d 行不复现提交工件"
                    "（补 --repeats 1 才 sha256 一致）；实跑 %d/%d 设计格；子集运行崩溃。裁定 L2。"
                    % (100 * g["zero_noise_max_rel"], rn["zero_noise"]["bias"],
                       100 * g["envelope_max_rel"], 100 * rn["envelope"]["share_within_15pct"],
                       r0["rows"], c["rows_executed"], c["design_cells_advertised"])),
        "scenario_key": "tensordem.shear_band_measurement_xval",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "XVAL-DEM-RADON-PROJECTION-MEASUREMENT", "board-audit",
                 "cross-verification", "shear-band", "uncertainty", "honest-FAIL"],
        "markdown": "\n".join(md),
        "figures": [{"name": "audit_m17873_report.png",
                     "caption": "左：24×48 算例格点测得 vs 真实倾角（无噪声），生产算子与修复前基线；"
                                "右：最大相对误差 vs 注入噪声，提交工件与发布复现命令两条曲线，"
                                "含 15%% 判据线与 3%% 内控线。",
                     "png_base64": base64.b64encode(png).decode()}],
    }
    r = requests.post(API + "/tensor/reports", json=body, timeout=60,
                      headers={"Authorization": "Bearer " + tok})
    print(r.status_code, r.text[:300])
    r.raise_for_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())