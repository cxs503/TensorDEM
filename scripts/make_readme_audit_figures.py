"""README ↔ 代码 API 巡检结果的图（全部数字从 JSON/实跑记录读，零手抄）。

产出 ``artifacts/readme_api_audit/figs/``：
* ``audit_checks_by_kind.png`` —— 巡检覆盖面（每类断言数与结果）；
* ``defects_before_after.png`` —— 三处缺陷修复前后的状态矩阵。

数据源：
* ``artifacts/readme_api_audit/report_full.json``（巡检器实跑产物）；
* ``artifacts/readme_api_audit/defects_before_after.json``（缺陷修复的实测记录，
  每条含复现命令与被观察到的输出行）。

用法::

    python scripts/make_readme_audit_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "readme_api_audit"
FIGS = ART / "figs"


def load(name: str) -> dict:
    return json.loads((ART / name).read_text(encoding="utf-8"))


def fig_by_kind(report: dict) -> Path:
    kinds = report["summary"]["by_kind"]
    labels = sorted(kinds)
    counts = [sum(kinds[k].values()) for k in labels]
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    bars = ax.bar(labels, counts, color="#2b6cb0")
    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("machine checks")
    ax.set_title(f"README/code consistency audit: {report['summary']['total']} checks, "
                 f"{report['summary']['by_status'].get('MISMATCH', 0)} mismatch")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "audit_checks_by_kind.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_before_after(record: dict) -> Path:
    defects = record["defects"]
    rows = [d["id"] for d in defects]
    cols = ["before", "after"]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for i, defect in enumerate(defects):
        for j, state in enumerate(cols):
            value = defect["states"][state]
            ok = value["status"] == "green"
            colour = "#2f855a" if ok else "#c53030"
            ax.add_patch(plt.Rectangle((j, len(rows) - i - 1), 1, 1,
                                       color=colour, alpha=0.85))
            ax.text(j + 0.5, len(rows) - i - 0.5, value["label"],
                    ha="center", va="center", color="white", fontsize=8, wrap=True)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(rows))
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["before fix", "after fix"])
    ax.set_yticks([len(rows) - i - 0.5 for i in range(len(rows))])
    ax.set_yticklabels([f"{d['id']}: {d.get('title_en', d['id'])}" for d in defects],
                       fontsize=8)
    ax.set_title("three fixed inconsistencies: measured before / after", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    out = FIGS / "defects_before_after.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    report = load("report_full.json")
    record = load("defects_before_after.json")
    made = [fig_by_kind(report), fig_before_after(record)]
    for path in made:
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())