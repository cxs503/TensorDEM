"""Build and post the IN99-N08 platform progress report (paper-grade, 6 sections).

Reads ``artifacts/in99-n08/metrics.json`` (plus the consumer assertion file) and
posts the report to ``$TENSOR_API_BASE/tensor/reports`` with the figures
embedded as base64 PNGs.  The markdown body is exactly the generated document,
so the platform copy and the repository copy cannot drift apart.

Usage::

    PYTHONPATH=src python3 benchmarks/consolidation_terzaghi/post_platform_report.py \
        [--metrics artifacts/in99-n08/metrics.json] [--tag NAME] [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

ART = REPO / "artifacts" / "in99-n08"


def build_payload(metrics_path: Path, tag: str | None) -> dict:
    from make_report import build as build_markdown

    metrics = json.loads(metrics_path.read_text())
    markdown = build_markdown(metrics_path, tag)
    suffix = f"_{tag}" if tag else ""
    figures = []
    for fig in metrics.get("figures", []):
        name = fig["file"]
        if suffix and suffix not in name:
            continue
        png = ART / "figures" / name
        if not png.exists():
            continue
        figures.append({
            "name": name,
            "caption": fig["caption"],
            "png_base64": base64.b64encode(png.read_bytes()).decode("ascii"),
        })
    gates = metrics["gates"]
    summary = (
        f"IN99-N08 一维固结/超孔压消散验证面（TensorDEM）：Terzaghi 级数锚 + 两场柱 + DEM 侧限压缩 + "
        f"r_u(T_v) 导出接口。机器裁决 {metrics['verdict']}；"
        f"C1 U(T_v) 五点最大偏差 {metrics['C1']['max_deviation_pct']:.4f}%，"
        f"C2 C_v 反演偏差 {metrics['C2']['C_v_deviation_pct']:.4f}% / s_f 偏差 {metrics['C2']['s_f_deviation_pct']:.3f}%，"
        f"C3 网格漂移 {metrics['C3']['column_refinement']['max_drift_pct']:.4f}% / DEM 域尺寸漂移 "
        f"{metrics['C3']['dem_particle_scan']['drift_pct']:.2f}%，"
        f"C4 下游断言 {'通过' if metrics['C4'].get('all_pass') else '未通过'}。"
        + ("诚实 FAIL：DEM 侧 E_oed 在 64–216 颗粒范围内未收敛。" if not gates["C3_drift_le5pct"] else "")
    )
    return {
        "title": "IN99-N08 一颗粒床一维固结与超孔压消散验证面：Terzaghi 解析锚 + DEM 反演 C_v + r_u(T_v) 接口",
        "summary": summary,
        "scenario_key": "in99-n08",
        "mission_id": "7343",
        "tags": ["TensorDEM", "consolidation", "Terzaghi", "r_u", "liquefaction",
                 "DEM", "verification", "IN99-N08"],
        "markdown": markdown,
        "figures": figures,
    }


def post(payload: dict) -> dict:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    if not base.endswith("/api/v1"):
        base = base.rstrip("/") + "/api/v1"
    login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                        "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=login,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    token = body.get("access_token") or body.get("token")
    if not token:
        raise SystemExit(f"login did not return a token: {body}")
    req = urllib.request.Request(
        base + "/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
        method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="/tmp/report.json")
    args = parser.parse_args(argv)
    metrics_path = Path(args.metrics) if args.metrics else ART / (
        f"metrics_{args.tag}.json" if args.tag else "metrics.json")
    payload = build_payload(metrics_path, args.tag)
    Path(args.out).write_text(json.dumps(payload))
    print(f"payload: {args.out} ({len(payload['markdown'])} chars, "
          f"{len(payload['figures'])} figures, {os.path.getsize(args.out)} B)")
    if args.dry_run:
        return 0
    print(json.dumps(post(payload))[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())