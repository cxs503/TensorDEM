#!/usr/bin/env python3
"""m20066 -- corrective addendum report: platform figure payload is not persisted.

The audit report id=18267 listed "report figures not embedded" against m20004.  A follow-up
query showed the cause is platform-side: only 10 of 27715 figure rows in tensor_reports carry
`png_base64`, and the payload of *this audit's own* report was stripped as well.  This script
publishes the corrected finding with a figure, so the board record is not left with a
misattributed defect.
"""
from __future__ import annotations

import base64
import json
import pathlib
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "audit_m20004"
EV = json.loads((OUT / "evidence.json").read_text())
PF = EV["P4_report"]["platform_figure_persistence"]
ENV = dict(l.strip().split("=", 1) for l in open("/root/tp-agent/agent.env")
           if "=" in l and not l.startswith("#"))
BASE = ENV["TENSOR_API_BASE"]

figdir = OUT / "figures"
figdir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
labels = ["figure rows\nin tensor_reports", "rows carrying\npng_base64"]
vals = [PF["figure_rows_total"], PF["figure_rows_with_png_base64"]]
ax.bar(labels, vals, color=["#4c72b0", "#c44e52"])
ax.set_yscale("log")
ax.set_ylabel("count (log)")
for i, v in enumerate(vals):
    ax.annotate(f"{v}", (i, v), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9)
ax.set_title("Platform figure persistence: png_base64 kept for "
             f"{vals[1]}/{vals[0]} figure rows\n(this audit's own report 18267 was stripped too)",
             fontsize=9)
fig.tight_layout()
png = figdir / "audit_m20004_fig2_platform_figure_persistence.png"
fig.savefig(png, dpi=130)
plt.close(fig)

md = f"""# 补正 · m20004 交差审计的 F6 改写：报告图丢失的根因在平台侧，不在 m20004

## 事实

审计报告 id=18267 的 ⑤ 曾记 m20004 报告（id=18245）"图未内嵌"。追查后发现**责任方需要改写**：

| 量 | 数值 |
|---|---|
| `tensor_reports` 中 figures 行总数 | {PF['figure_rows_total']} |
| 其中真正存下 `png_base64` 的行数 | **{PF['figure_rows_with_png_base64']}** |
| 本次审计自己的报告（id=18267）落库后的 `png_base64` 长度 | {PF['this_audit_report_png_base64_len']} |
| m20004 报告（id=18245）`png_base64` 行数 | 0（3 条图仅有 name+caption） |

即 `POST /tensor/reports` 的 `figures[].png_base64` 在落库时被丢弃，只保留 `name`/`caption`。
审计者自己按纪律内嵌了 base64 的图，同样没能留存 → **这是平台管线缺陷，不是 m20004 作者缺陷**。
F6 因此在审计文档中由 MAJOR 改写为 MINOR（`docs/AUDIT_M20004/README.md`、verdict 行 813）。

## 影响与建议

* 影响：任何报告在平台上都无法渲染 png，六要素里的"④结果图"实际上只剩图注 —— 这是全生态口径问题
  （{PF['figure_rows_total']} 条图行里 {PF['figure_rows_total'] - PF['figure_rows_with_png_base64']} 条无图）。
* 建议平台：`figures[]` 落库保留 `png_base64`（或改存对象存储 + URL），并对历史报告回补。
* m20004 侧仍应有的补救（不受本补正影响）：图落到已入库的路径并写进正文，因为它的
  `artifacts/` 目录当前 0 入库（审计 F3）。

## 与本案其余结论的关系

判级不变：**L3**（功能实质成立），L4 拒绝（E_oed 偏差 15.6923% vs 自定容差 10%，裁判表 0 行，
case 工件 0 入库）。图注与数值均由 `docs/AUDIT_M20004/verify.py` 程序化产出。
"""
fig_b64 = base64.b64encode(png.read_bytes()).decode()

req = urllib.request.Request(BASE + "/auth/login", method="POST",
                             data=json.dumps({"username": ENV["TENSOR_AGENT_USER"],
                                              "password": ENV["TENSOR_AGENT_PASS"]}).encode(),
                             headers={"Content-Type": "application/json"})
tok = json.loads(urllib.request.urlopen(req, timeout=60).read())["access_token"]
body = {
    "title": "补正 · m20004 审计 F6 改写：报告图 png_base64 未落库是平台缺陷（10/27715 有图）",
    "summary": f"复查 tensor_reports：{PF['figure_rows_total']} 条图行仅 {PF['figure_rows_with_png_base64']} 条存下 "
               f"png_base64，本次审计自身报告 18267 也被剥离 → m20004 报告无图的责任在平台管线，"
               f"F6 由 MAJOR 改写为 MINOR；m20004 判级仍为 L3，L4 拒绝（15.69% vs 10% 容差）。",
    "scenario_key": "CASE-DEM-OEDOMETER-STIFF",
    "mission_id": 20066,
    "tags": ["audit-correction", "platform-defect", "tensor_reports", "figures", "m20004",
             "CASE-DEM-OEDOMETER-STIFF"],
    "markdown": md,
    "figures": [{"name": png.name,
                 "caption": "tensor_reports 图行总数 vs 真正携带 png_base64 的行数（对数轴），"
                            "说明报告图丢失是平台落库行为",
                 "png_base64": fig_b64}],
}
req = urllib.request.Request(BASE + "/tensor/reports", method="POST",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok})
resp = json.loads(urllib.request.urlopen(req, timeout=120).read() or b"{}")
print("posted addendum:", resp.get("id"), resp.get("title", "")[:60])
(OUT / "figures" / png.name).with_suffix(".post.json").write_text(json.dumps(resp, indent=2)[:2000])