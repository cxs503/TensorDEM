#!/usr/bin/env python3
"""Publish the m22657 Beverloo-ensemble report to the platform.

Reads the generated document + figures so the report body cannot drift from the
artefacts (zero hand copying), logs in with the agent credentials and POSTs
``/tensor/reports``.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "CASE-DEM-SILO-BEVERLOO_m22657.md"
FIGDIR = ROOT / "docs" / "figures"
SUMMARY = ROOT / "runs" / "m22657_beverloo_ensemble" / "summary.json"

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def token() -> str:
    if os.environ.get("TOKEN"):
        return os.environ["TOKEN"]
    r = requests.post(f"{BASE}/auth/login",
                      data={"username": USER, "password": PASS}, timeout=60)
    r.raise_for_status()
    body = r.json()
    return body.get("access_token") or body.get("token")


def figures() -> list[dict]:
    out = []
    caps = {
        "m22657_ensemble_errors.png":
            "单 seed 相对误差（点）+ 6-seed 系综均值与 Student-t 95 % CI（方块/误差棒）"
            "，灰带为总账 ±10 % 容差；用于判定单点 PASS 是否可外推。",
        "m22657_window_history.png":
            "逐窗口（0.05 s）流率历史，阴影为计入「稳态」的后 50 % 窗口；"
            "黑虚线为 Beverloo 文献系数预测 W_ref，红带为系综 95 % CI。",
    }
    for name, cap in caps.items():
        p = FIGDIR / name
        if p.exists():
            out.append({"name": name, "caption": cap,
                        "png_base64": base64.b64encode(p.read_bytes()).decode()})
    return out


def main() -> int:
    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    body = DOC.read_text()
    ens = summary.get("ensemble", {})
    ver = summary.get("verdicts", {})
    detail = "NOT_RUN（6 分片未回传）" if not ens else (
        f"W_ens={ens['W_ens_kg_s']:.6f} kg/s, rel={100 * ens['rel_err_ensemble']:+.2f} %, "
        f"CI95=[{100 * ens['rel_err_ci95'][0]:+.2f}, {100 * ens['rel_err_ci95'][1]:+.2f}] %, "
        f"CV={100 * ens['cv']:.2f} %, V1={ver.get('V1_ledger')}")
    payload = {
        "title": "CASE-DEM-SILO-BEVERLOO · 料仓卸料 Beverloo 律：seed 系综不确定度预算（m22657）",
        "summary": ("补上八轮战役从未测量的量：同一构型下换堆积的散布。以 6 个独立 seed 的 "
                    "Student-t 区间裁决总账 10 % 容差，并给出该算例的计数噪声下限（1/√N）。"
                    f" {detail}"),
        "scenario_key": "DEM-SILO-BEVERLOO",
        "mission_id": 22657,
        "tags": ["DEM", "Beverloo", "silo discharge", "uncertainty budget",
                 "seed ensemble", "validation debt"],
        "markdown": body,
        "figures": figures(),
    }
    tok = token()
    r = requests.post(f"{BASE}/tensor/reports", json=payload,
                      headers={"Authorization": f"Bearer {tok}"}, timeout=180)
    print(r.status_code)
    print(json.dumps(r.json() if r.headers.get("content-type", "").startswith("application/json")
                     else {"text": r.text[:500]}, ensure_ascii=False)[:800])
    return 0 if r.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())