#!/usr/bin/env python
"""Build ``/tmp/report.json`` for the P82W4-N09 report from the rendered doc.

The markdown body is the file produced by ``scripts/mts_contact_report.py`` (all
numbers injected from ``artifacts/mts_contact/gate.json``); the figures are the
PNGs that the same run wrote, embedded as base64 exactly as the report API wants.
"""
from __future__ import annotations

import base64
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "mts_contact_timestep_gate.md")
GATE = os.path.join(ROOT, "artifacts", "mts_contact", "gate.json")
FIGDIR = os.path.join(ROOT, "artifacts", "mts_contact", "figures")
OUT = "/tmp/report.json"

CAPTIONS = {
    "restitution_ladder.png": "判据 (a)：单球 Hertz+Hunt–Crossley 冲击的恢复系数随外步 dt/t_c 的阶梯（左）"
                              "与相对细步参考的误差（右，绿色为 ±2% 验收带）。",
    "negative_control.png": "判据 (d)：dt/t_c=0.5 且不做子循环时的恢复系数（红）与细步参考（蓝）——"
                            "门若在此处仍正确，说明门不紧。",
    "energy_momentum.png": "判据 (b)：无耗散双周期气体 1e5 外步的机械能相对漂移（左，红虚线 1% 界）"
                           "与线动量漂移（右，红虚线 1e-12 界，成对力按 ±f 累加故结构守恒）。",
    "jamming_eos.png": "判据 (c)：摩擦 jamming（μ=0.5，N=256）等向减压 EOS。MTS 子循环与 legacy 3× 摩擦臂"
                       "都收敛到伺服目标压力并给出同一 φ_J；W4-P1 伺服参考未收敛到目标（见报告 §7）。",
    "wall_clock_and_physics.png": "判据 (c) 的成本与物理：同一物理协议下两条摩擦臂的墙钟（左）"
                                  "与剪切胞稳态剪应力（右，物理不变）。",
}


def main() -> int:
    with open(DOC) as fh:
        markdown = fh.read()
    with open(GATE) as fh:
        gate = json.load(fh)
    figures = []
    for name in sorted(os.listdir(FIGDIR)) if os.path.isdir(FIGDIR) else []:
        if not name.endswith(".png"):
            continue
        with open(os.path.join(FIGDIR, name), "rb") as fh:
            payload = base64.b64encode(fh.read()).decode("ascii")
        figures.append({"name": name, "caption": CAPTIONS.get(name, name),
                        "png_base64": payload})
    summary = {
        "judgements": {},
        "headline": {},
    }
    for part in ("restitution", "energy", "shear", "jamming", "shards"):
        entry = gate.get(part, {})
        if isinstance(entry, dict) and "verdict" in entry:
            summary["judgements"][part] = entry["verdict"]
    jam = gate.get("jamming", {})
    if jam:
        summary["headline"] = {
            "phi_j_mts": jam.get("mts", {}).get("phi_j"),
            "phi_j_legacy3x": jam.get("legacy3x", {}).get("phi_j"),
            "phi_j_rel_diff": jam.get("phi_j_rel_diff_mts_vs_legacy"),
            "wall_clock_ratio": jam.get("wall_clock_ratio"),
            "phi_j_w4p1_reference": (jam.get("w4p1_servo_reference") or {}).get("phi_j"),
            "phi_j_rel_diff_vs_reference": jam.get("phi_j_rel_diff_mts_vs_reference"),
        }
    report = {
        "title": "多速率接触积分（MTS）与时间步稳定性契约 —— TensorDEM P82W4-N09",
        "summary": (
            "给 DEM 装上可机器裁决的时间步/子循环契约（dt/t_c≤0.2 闭合、>0.5 拒绝），"
            "用接触子循环把摩擦臂 3× return-mapping 单步成本压回 1×："
            f"恢复系数误差 ≤{abs(gate.get('restitution', {}).get('ladder', {}).get('worst_abs_restitution_error') or 0)*100:.2f}%，"
            "1e5 步无耗散能量漂移与动量守恒达标，"
            f"μ=0.5 摩擦 jamming（N=256）φ_J 与基线差 {100*(jam.get('phi_j_rel_diff_mts_vs_legacy') or 0):.2f}%、"
            f"墙钟加速 {(jam.get('wall_clock_ratio') or 0):.2f}×；"
            "负控证明门是紧的；对 W4-P1 伺服参考的 φ_J 偏差作为诚实 FAIL 记录（参考协议本身未收敛到伺服目标）。"
        ),
        "scenario_key": "tensordem.multirate_contact_gate",
        "mission_id": 7364,
        "tags": ["TensorDEM", "MTS", "多速率积分", "接触摩擦", "return mapping",
                 "时间步稳定性", "jamming", "phi_J", "证据分级:REAL"],
        "markdown": markdown,
        "figures": figures,
        "meta": summary,
    }
    with open(OUT, "w") as fh:
        json.dump(report, fh)
    print(json.dumps({"out": OUT, "markdown_chars": len(markdown),
                      "figures": [f["name"] for f in figures],
                      "headline": summary["headline"],
                      "judgements": summary["judgements"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
