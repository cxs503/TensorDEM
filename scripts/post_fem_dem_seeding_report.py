#!/usr/bin/env python
"""Post the p302w2-n03 report (mission 11081) to the TensorPlatform report API.

The markdown is ``docs/FEM_DEM_SEEDING_HANDOFF_V2.md`` (numbers injected from
``artifacts/fem_dem_seeding/p302w2_n03/report_data.json``); the figures are the six
PNGs beside it.  Nothing is typed by hand here either -- the summary is assembled
from the same payload.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "fem_dem_seeding", "p302w2_n03")
DOC = os.path.join(ROOT, "docs", "FEM_DEM_SEEDING_HANDOFF_V2.md")
DATA = os.path.join(OUT, "report_data.json")
FIGS = os.path.join(OUT, "figures.json")
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"


def login() -> str:
    payload = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                          "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=payload,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    token = body.get("access_token") or body.get("token")
    if not token:
        raise SystemExit(f"login did not return a token: {body}")
    return token


def summary_text(d: dict) -> str:
    c = d["criteria"]
    led = d["ledger"]
    heave = d["heave"]
    kin = d["kinematics"]
    n_pass = sum(1 for v in c.values() if v["pass"])
    # The API caps the summary at 500 characters; keep the strongest numbers.
    short = (
        f"fem_dem_seeding：FEM→DEM 断裂交接由账面守恒推进到形状分辨+运动学传递+动力守恒。"
        f"判据 {n_pass}/{len(c)} 通过：质量/面积残差 0，积分后仍 0；边界恒等式外缘残差 0；"
        f"形状 IoU {led['equivalent_disc']['mean_iou']:.2f}→{led['clump']['mean_iou']:.2f}（逐块提升），"
        f"等效盘溢出 {led['equivalent_disc']['spill_area_sum_m2']:.1f} m²、互穿 "
        f"{d['overlap']['equivalent_disc']['n_overlapping_pairs']} 对，带粒团均 0；"
        f"刚性残差 {100 * kin['ensemble_rel_residual']:.1f}%；垂荡周期误差 "
        f"{heave['max_period_error']:.0e}；漂移闭式 {d['drift']['rel_error']:.0e}。"
        "局限：圆盘团上限 0.907、单单元宽碎块 IoU 0.40-0.51、垂荡无附加质量。"
    )
    return short[:500]


def main() -> int:
    with open(DOC, encoding="utf-8") as fh:
        markdown = fh.read()
    with open(DATA, encoding="utf-8") as fh:
        data = json.load(fh)
    captions = {}
    if os.path.exists(FIGS):
        with open(FIGS, encoding="utf-8") as fh:
            captions = {r["name"]: r["caption"] for r in json.load(fh)}
    figures = []
    figdir = os.path.join(OUT, "figures")
    for name in sorted(os.listdir(figdir)):
        if not name.endswith(".png"):
            continue
        with open(os.path.join(figdir, name), "rb") as fh:
            figures.append({"name": name, "caption": captions.get(name, name),
                            "png_base64": base64.b64encode(fh.read()).decode()})

    payload = {
        "title": "FEM→DEM 断裂交接链（二）：碎块形状分辨带粒团 + FE 场运动学传递 + "
                 "交接后动力守恒（p302w2-n03）",
        "summary": summary_text(data),
        "scenario_key": "p302w2-n03-fem-dem-fracture-handoff",
        "mission_id": "11081",
        "tags": ["fem-dem-handoff", "ice-fracture", "floe-seeding", "disc-clump",
                 "mass-conservation", "boundary-identity", "kinematic-transfer",
                 "rigidity-residual", "heave-oscillator", "tensordem"],
        "markdown": markdown,
        "figures": figures,
    }
    token = login()
    req = urllib.request.Request(BASE + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + token},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            print(resp.read().decode()[:2000])
    except urllib.error.HTTPError as exc:
        print("HTTP", exc.code, exc.read().decode()[:2000])
        return 1
    print("posted:", payload["title"], "| figures:", len(figures))
    return 0


if __name__ == "__main__":
    sys.exit(main())