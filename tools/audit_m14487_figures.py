#!/usr/bin/env python3
"""AUDIT-ASSET-12REPO / TensorDEM: figures for audit_report.md (programmatic, no hand numbers)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "audit_m14487"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)

inv = json.loads((ART / "inventory.json").read_text())
wiring = json.loads((ART / "wiring.json").read_text())

# verdict levels per PR (audit decisions, source of truth = audit_report.md §3)
LEVEL = {
    "#1151": 4, "#1157": 2, "#1162": 2, "#1168": 2, "#1170": 1, "#1171": 1,
    "#1173": 2, "#1175": 2, "#1176": 1, "#1178": 2, "#1179": 2, "#1180": 0,
    "#1181": 2, "#1182": 0, "#1188": 3, "#1191": 4, "#1195": 4, "#1196": 2,
    "#1202": 2, "#1203": 3, "#1204": 2, "#1205": 2, "#1206": 2, "#1208": 2,
    "#1209": 2,
}

pr_add = defaultdict(lambda: defaultdict(int))
for r in inv["records"]:
    tag = "#" + r["pr"].split("#")[1].split()[0]
    pr_add[tag][r["kind"]] += r["added"] or 0

# ---------------------------------------------------------------- fig1: levels
cnt = Counter(LEVEL.values())
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=130)
levels = [0, 1, 2, 3, 4, 5]
vals = [cnt.get(l, 0) for l in levels]
colors = ["#9e9e9e", "#ef6c00", "#fbc02d", "#66bb6a", "#1565c0", "#6a1b9a"]
bars = ax.bar([f"L{l}" for l in levels], vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v), ha="center", fontsize=11)
ax.set_ylabel("PR 数")
ax.set_title(
    "TensorDEM 近 25 个已并 PR 的成熟度判级分布\n"
    f"L≥4 = {sum(vals[4:])}/25 ({100*sum(vals[4:])/25:.0f}%) · L2 占 {100*vals[2]/25:.0f}% · L≤1 = {vals[0]+vals[1]}/25"
)
ax.set_ylim(0, max(vals) + 2)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig1_maturity_levels.png")
plt.close(fig)

# ------------------------------------------------------- fig2: PR composition
tags = sorted(pr_add, key=lambda t: int(t[1:]))
kinds = ["core", "test", "doc", "script", "other"]
labels = {"core": "核心算法", "test": "测试", "doc": "文档", "script": "脚本", "other": "其它(含草稿)"}
fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=130)
bottom = [0] * len(tags)
palette = {"core": "#1565c0", "test": "#66bb6a", "doc": "#fbc02d", "script": "#8d6e63", "other": "#bdbdbd"}
for k in kinds:
    v = [pr_add[t][k] for t in tags]
    ax.bar(tags, v, bottom=bottom, label=labels[k], color=palette[k])
    bottom = [a + b for a, b in zip(bottom, v)]
for i, t in enumerate(tags):
    ax.text(i, bottom[i] + 120, f"L{LEVEL[t]}", ha="center", fontsize=8)
ax.set_ylabel("新增行数")
ax.set_title("逐 PR 新增构成与判级（柱顶 = 本次审计判级）")
ax.legend(fontsize=8, ncol=5)
ax.grid(axis="y", alpha=0.3)
ax.set_xticklabels(tags, rotation=90, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig2_pr_composition.png")
plt.close(fig)

# ------------------------------------------------------------ fig3: wiring
n_all = len(wiring)
n_none = sum(1 for v in wiring.values() if not v["all"])
n_tests = sum(1 for v in wiring.values() if v["all"] and not v["wired"])
n_prod = n_all - n_none - n_tests
fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=130)
vals = [n_prod, n_tests, n_none]
names = [f"有生产调用方\n({n_prod})", f"仅测试调用\n({n_tests})", f"无任何调用方\n({n_none})"]
bars = ax.bar(names, vals, color=["#66bb6a", "#fbc02d", "#ef5350"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v} ({100*v/n_all:.0f}%)", ha="center", fontsize=10)
ax.set_ylabel("新增核心模块文件数")
ax.set_title(f"77 个新核心模块的接入状态（无生产调用方 {n_tests+n_none}/77 = {100*(n_tests+n_none)/n_all:.0f}%）")
ax.set_ylim(0, max(vals) + 6)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig3_wiring.png")
plt.close(fig)

stats = {"levels": {f"L{k}": v for k, v in sorted(cnt.items())},
         "wiring": {"all": n_all, "prod": n_prod, "tests_only": n_tests, "none": n_none},
         "total_added": {k: sum(pr_add[t][k] for t in tags) for k in kinds}}
(ART / "figures_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
print(json.dumps(stats, indent=1, ensure_ascii=False))
print("figures ->", sorted(p.name for p in FIG.glob("*.png")))