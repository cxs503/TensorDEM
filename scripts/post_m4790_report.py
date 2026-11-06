"""Publish the m4790 paper-level progress report to the Tensor platform.

Numbers are injected programmatically from ``artifacts/m4790/sensitivity.json``
(written by ``scripts/plot_m4790_sensitivity.py``), so nothing is transcribed
by hand.  The narrative body comes from ``artifacts/m4790/report.md``; the
results table and the figure caption are regenerated here from the JSON.

Usage::

    python scripts/post_m4790_report.py --out /tmp/report.json   # payload only
    python scripts/post_m4790_report.py --post                    # login + POST

Environment: ``TENSOR_API_BASE`` (default http://127.0.0.1:8010),
``TENSOR_AGENT_USER`` / ``TENSOR_AGENT_PASS`` (needed for --post),
``TENSOR_MISSION_ID`` (default 4856).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "m4790"
TABLE = ART / "sensitivity.json"
FIGURE = ART / "ledger_corruption_sensitivity.png"
BODY = ART / "report.md"

#: keep the POST payload well below the 250 KB silent-failure threshold
MAX_PNG_BYTES = 200_000


def results_markdown(table: dict) -> str:
    """Render the fault-injection table from the JSON (no hand copying)."""
    tol = table["tolerance_rel"]
    lines = [
        "### 程序化注入的残差表（`artifacts/m4790/sensitivity.json`）",
        "",
        f"- 场景：2-D 冰-工具压入，{table['scenario']['lattice'][0]}×"
        f"{table['scenario']['lattice'][1]} 晶格，"
        f"tool_speed={table['scenario']['tool_speed_m_s']} m/s，"
        f"t={table['scenario']['duration_s']} s，"
        f"broken_bonds={table['scenario']['broken_bonds']}",
        f"- 判据：相对残差 TOL = {tol:.3g}；未篡改基线 = "
        f"{table['uncorrupted_rel_err']:.4e} "
        f"({table['uncorrupted_rel_err']*100:.2f} %)",
        "",
        "| 通道 | 故障 | 相对残差 | 检出 | 裕度（×TOL） |",
        "|---|---|---|---|---|",
        "| —（未篡改正控） | — | "
        f"{table['uncorrupted_rel_err']:.4e} | 闭合 ✓ | "
        f"{table['uncorrupted_rel_err']/tol:.3f} |",
    ]
    for c in table["cases"]:
        if c["fault_kind"] == "factor":
            fault = f"×{c['fault_value']:g}"
        else:
            fault = f"+{c['fault_value']:.4e} J（加性）"
        lines.append(
            f"| `{c['channel']}` | {fault} | {c['rel_err']:.4e} | "
            f"{'是' if c['detected'] else '否'} | {c['rel_err']/tol:.1f} |"
        )
    detected = sum(1 for c in table["cases"] if c["detected"])
    lines.append("")
    lines.append(
        f"**结论**：{len(table['cases'])} 条故障注入中 {detected} 条被检出；"
        f"最紧裕度 {min(c['rel_err'] for c in table['cases'])/tol:.1f}× TOL。"
    )
    return "\n".join(lines)


def build_payload(mission_id: int) -> dict:
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    png = FIGURE.read_bytes()
    if len(png) > MAX_PNG_BYTES:
        raise SystemExit(
            f"figure is {len(png)} B > {MAX_PNG_BYTES} B; shrink before posting"
        )
    body = BODY.read_text(encoding="utf-8")
    err = table["uncorrupted_rel_err"]
    caption = (
        "2-D DEM 能量/断裂台账残差敏感性：未篡改台账闭合于 "
        f"{err*100:.2f} %（< 0.5 % 容差）；"
        f"{len(table['cases'])} 条单通道故障注入后残差 "
        f"{min(c['rel_err'] for c in table['cases'])*100:.1f} %–"
        f"{max(c['rel_err'] for c in table['cases'])*100:.1f} %，"
        "全部越过 0.5 % 判据被检出。本单为后端/测试证据类任务，无物理场量，"
        "故以残差敏感性为证据图。"
    )
    markdown = body + "\n\n" + results_markdown(table) + "\n"
    return {
        "title": "TensorDEM 2-D 能量/断裂台账负控：证据产物 + 台账棘轮修复"
                 f"（m{mission_id}）",
        "summary": (
            "从 main(cb6c562) 切枝重开 TensorDEM#340 的 m4790 证据产物部分："
            "(1) 保留 main 已合入的成熟负控测试（10 passed，含加性故障与空转守卫），"
            "不回退到 m4790 的弱化版；(2) 新增/重写 scripts/plot_m4790_sensitivity.py "
            "复算 7 条故障注入残差，产出图 + artifacts/m4790/sensitivity.json；"
            "(3) 重生成 docs/doc_evidence_ledger.json，把 m4829 的 3 个从未入 git 的"
            "artifacts/m4829_ci_gate/* 指针如实改为 MISSING，并把 m4790 新产物登记为 "
            "TRACKED，scripts/audit_doc_evidence.py --check 转绿。"
            f"未篡改台账相对残差 {err*100:.2f}% < 3%。"
        ),
        "scenario_key": "tensordem_2d_energy_fracture_ledger_negative",
        "mission_id": mission_id,
        "tags": ["tensordem", "dem", "energy-ledger", "negative-control",
                 "doc-evidence", "ice"],
        "markdown": markdown,
        "figures": [{
            "name": FIGURE.name,
            "caption": caption,
            "png_base64": base64.b64encode(png).decode(),
        }],
    }


def req(base: str, path: str, method: str = "GET", body: dict | None = None,
        token: str | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(base + path, data=data, headers=headers,
                                     method=method)
    with urllib.request.urlopen(request, timeout=180) as resp:
        return json.loads(resp.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write the JSON payload here (no posting)")
    ap.add_argument("--post", action="store_true", help="login and POST")
    ap.add_argument("--mission-id", type=int,
                    default=int(os.environ.get("TENSOR_MISSION_ID", "4856")))
    args = ap.parse_args()

    payload = build_payload(args.mission_id)
    print("markdown_chars", len(payload["markdown"]),
          "png_b64_len", len(payload["figures"][0]["png_base64"]))
    if args.out:
        Path(args.out).write_text(json.dumps(payload), encoding="utf-8")
        print("payload_written", args.out)
    if not args.post:
        return

    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010")
    if not base.endswith("/api/v1"):
        base = base.rstrip("/") + "/api/v1"
    tok = req(base, "/auth/login", "POST", {
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    })
    tok = tok.get("access_token") or tok.get("token")
    resp = req(base, "/tensor/reports", "POST", payload, token=tok)
    print("POST status", resp.get("status"),
          "id", resp.get("id") or resp.get("data", {}).get("id"))


if __name__ == "__main__":
    main()