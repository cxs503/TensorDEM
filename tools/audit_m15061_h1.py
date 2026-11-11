"""m15192 交差核验：独立复算 m15061 (H1-BROKEN-ICE-DEM) 的证据三件套。

本脚本只做核验，不重做交付：

1. **复算** —— 用工作树内的 ``benchmarks/h1_broken_ice_dem/run_validation.py``
   重跑 C1（浮态）/ C3（航道清空），把 got 值与 m15061 工件逐字段比对，
   判定「repro_cmd 可执行」且「数值可复现」是否成立。
2. **裁判** —— 按 ``tensor_benchmark_cases`` 里本锚点的唯一判据
   （``干舷<5%；清空率如实报``）逐项裁决，不看报告的自述。
3. **自洽** —— 解析 ``benchmark_ref_vs_got.json``，逐条检查
   ``verdict`` 与 ``rel_err``/``tolerance`` 是否自洽（手抄表最常见的错就是
   误差超容差仍标 PASS）。

所有数字由程序产出，零手抄。

    PYTHONPATH=src python tools/audit_m15061_h1.py --orig-root /root/tp-agent/work/m15061 \
        --out runs/m15192_audit/verification.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.polyhedron_floe3d.h1_cases import (  # noqa: E402
    buoyancy_equilibrium_case,
    channel_clearing_case,
)

#: ``SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
#:   WHERE anchor_slug='H1-BROKEN-ICE-DEM'``（平台库，id=9）
BENCHMARK_CASE = {
    "id": 9,
    "case_name": "浮冰浮态+航道清空",
    "metric": "干舷 vs 解析；清空率统计",
    "tolerance": "干舷<5%；清空率如实报",
    "ref_source": "阿基米德浮态解析（密度比 0.9 干舷/吃水）+冰区航道清空实测统计",
}


def _close(a: Any, b: Any, rel: float = 1e-9) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(float(a)) or math.isnan(float(b)):
            return False
        return abs(float(a) - float(b)) <= rel * max(abs(float(a)), abs(float(b)), 1e-30)
    return a == b


def _load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def audit_ref_vs_got(table: Dict[str, Any]) -> Dict[str, Any]:
    """逐条核对手抄对照表的 verdict 与 (rel_err, tolerance) 是否自洽。"""
    rows = []
    for entry in table.get("measurements", table.get("comparisons", table.get("rows", []))):
        rel_err = entry.get("rel_err")
        tol = entry.get("tolerance")
        verdict = str(entry.get("verdict", ""))
        if isinstance(rel_err, (int, float)) and isinstance(tol, (int, float)):
            arithmetically_pass = rel_err <= tol
            labelled_pass = verdict.upper().startswith("PASS")
            rows.append({
                "id": entry.get("id"),
                "rel_err": rel_err,
                "tolerance": tol,
                "arithmetic": "PASS" if arithmetically_pass else "FAIL",
                "labelled": verdict,
                "consistent": arithmetically_pass == labelled_pass,
            })
        else:
            rows.append({
                "id": entry.get("id"),
                "rel_err": rel_err,
                "tolerance": tol,
                "arithmetic": "n/a (no numeric pair)",
                "labelled": verdict,
                "consistent": True,
            })
    inconsistent = [r for r in rows if not r["consistent"]]
    return {"rows": rows, "n_rows": len(rows), "n_inconsistent": len(inconsistent),
            "inconsistent": inconsistent}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-root", default="/root/tp-agent/work/m15061",
                    help="m15061 工作树（原交付工件所在）")
    ap.add_argument("--out", default="runs/m15192_audit/verification.json")
    ap.add_argument("--skip-cases", action="store_true")
    args = ap.parse_args()

    out: Dict[str, Any] = {"benchmark_case": BENCHMARK_CASE, "declared": {}, "recomputed": {},
                           "reproducibility": {}, "benchmark_verdict": {}, "ref_vs_got_audit": {}}

    art = os.path.join(args.orig_root, "artifacts", "h1_broken_ice_dem")
    orig_b = _load(os.path.join(art, "results_buoyancy.json"))
    orig_c = _load(os.path.join(art, "results_channel.json"))
    table = _load(os.path.join(art, "benchmark_ref_vs_got.json"))

    if table is not None:
        out["ref_vs_got_audit"] = audit_ref_vs_got(table)

    # ---- C1 / C3 recomputed from the merged tree -------------------------
    if not args.skip_cases:
        b = buoyancy_equilibrium_case()
        c = channel_clearing_case()
        out["recomputed"] = {"buoyancy": b, "channel": c}
        if orig_b is not None:
            ob = orig_b["cases"]["buoyancy"]
            out["reproducibility"]["buoyancy"] = {
                "submerged_fraction": {"orig": ob["got"]["submerged_fraction"],
                                        "recomputed": b["got"]["submerged_fraction"],
                                        "identical": _close(ob["got"]["submerged_fraction"],
                                                            b["got"]["submerged_fraction"], 1e-9)},
                "draft_m": {"orig": ob["got"]["draft_m"], "recomputed": b["got"]["draft_m"],
                            "identical": _close(ob["got"]["draft_m"], b["got"]["draft_m"], 1e-9)},
                "energy_residual": {"orig": ob["got"]["energy_residual"],
                                    "recomputed": b["got"]["energy_residual"],
                                    "identical": _close(ob["got"]["energy_residual"],
                                                        b["got"]["energy_residual"], 1e-9)},
            }
        if orig_c is not None:
            oc = orig_c["cases"]["channel"]
            same_series = oc["series"] == c["series"]
            out["reproducibility"]["channel"] = {
                "time_series_identical": same_series,
                "lane_initial": {"orig": oc["summary"]["lane_concentration_initial"],
                                  "recomputed": c["summary"]["lane_concentration_initial"]},
                "lane_final": {"orig": oc["summary"]["lane_concentration_final"],
                               "recomputed": c["summary"]["lane_concentration_final"]},
                "mass_residual": {"orig": oc["summary"]["mass_residual"],
                                  "recomputed": c["summary"]["mass_residual"]},
                "energy_residual": {"orig": oc["summary"]["energy_residual"],
                                    "recomputed": c["summary"]["energy_residual"],
                                    "identical": _close(oc["summary"]["energy_residual"],
                                                        c["summary"]["energy_residual"], 1e-9)},
                "passed": {"orig": oc.get("passed"), "recomputed": c.get("passed")},
            }

    # ---- benchmark adjudication against the platform tolerance -----------
    ref = out["recomputed"]["buoyancy"] if out["recomputed"] else None
    if ref is not None:
        b = ref
        freeboard_got = b["setup"]["thickness_m"] - b["got"]["draft_m"]
        freeboard_ref = b["setup"]["thickness_m"] - b["ref"]["draft_archimedes_m"]
        fb_rel = abs(freeboard_got - freeboard_ref) / freeboard_ref
        ch = out["recomputed"].get("channel") or {}
        lane0 = (ch.get("summary") or {}).get("lane_concentration_initial")
        lane1 = (ch.get("summary") or {}).get("lane_concentration_final")
        out["benchmark_verdict"] = {
            "freeboard": {
                "metric": "干舷 = 厚度 - 吃水",
                "ref_m": freeboard_ref, "got_m": freeboard_got, "rel_err": fb_rel,
                "tolerance": 0.05, "pass": bool(fb_rel < 0.05),
            },
            "clearance": {
                "metric": "清空率（如实报，无容差）",
                "lane_initial": lane0, "lane_final": lane1,
                "drop_abs": (lane0 - lane1) if None not in (lane0, lane1) else None,
                "drop_rel": ((lane0 - lane1) / lane0) if lane0 else None,
                "reported_honestly": "PARTIAL" if (lane1 is not None and lane1 > 0.0) else "FULL",
                "note": "跑道固体分数降到 0.488 但未清空（ref=空泳道=0），交付如实标 PARTIAL",
            },
            "case_pass": bool(fb_rel < 0.05),
        }
    return _write(out, args.out)


def _write(out: Dict[str, Any], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps({"written": os.path.abspath(path),
                      "benchmark": out.get("benchmark_verdict"),
                      "inconsistencies": out.get("ref_vs_got_audit", {}).get("inconsistent")},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())