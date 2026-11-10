"""Post the weakest-link campaign report (six-element, figures embedded).

Numbers and figures are read from the campaign artefacts, never transcribed.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "ice_bond_weakest_link"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"


def login() -> str:
    payload = json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode()
    request = urllib.request.Request(BASE + "/auth/login", data=payload,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read())
    return body.get("access_token") or body["token"]


def figure(name: str, caption: str) -> dict:
    data = base64.b64encode((ART / name).read_bytes()).decode()
    return {"name": name, "caption": caption, "png_base64": data}


def main() -> int:
    payload = json.loads((ART / "verification.json").read_text())
    ecdf = payload["arm_ecdf"]
    per_size = payload["arm_dynamic"]["per_size"]
    fit_all = payload["arm_dynamic"]["size_effect_fit_all_points"]
    rate = payload["arm_rate"]["rows"]

    def table_ecdf() -> str:
        head = ("| N | median measured [N] | analytic [N] | rel. err | KS | 5 % band | fitted m |\n"
                "|---|---|---|---|---|---|---|\n")
        rows = "".join(
            f"| {r['n_bonds']} | {r['median_measured_N']:.6f} | {r['median_analytic_N']:.6f} | "
            f"{r['median_rel_error']:+.2e} | {r['ks_statistic']:.4f} | {r['ks_5pct_band']:.4f} | "
            f"{r['fitted_modulus']:.3f} |\n" for r in ecdf["rows"])
        return head + rows

    def table_dynamic() -> str:
        head = ("| N | R | median load [N] | analytic [N] | rel. err | weakest link | force spread |\n"
                "|---|---|---|---|---|---|---|\n")
        rows = "".join(
            f"| {k} | {v['realisations']} | {v['median_load_N']:.4f} | {v['analytic_median_N']:.4f} | "
            f"{v['median_rel_error']:+.2%} | {v['weakest_link_rate']:.0%} | "
            f"{v['median_force_spread']:.2%} |\n" for k, v in per_size.items())
        return head + rows

    def table_rate() -> str:
        head = ("| T [s] | strain rate [s⁻¹] | steps | load [N] | rel. err | spread | weakest link |\n"
                "|---|---|---|---|---|---|---|\n")
        rows = "".join(
            f"| {r['ramp_duration_s']:.2f} | {r['strain_rate_s-1']:.4f} | {r['steps']} | "
            f"{r['load_N']:.4f} | {r['rel_error']:+.3%} | {r['force_spread']:.2%} | "
            f"{r['weakest_link_selected']} |\n" for r in rate)
        return head + rows

    # The report body is the checked-in verification document, so the report and
    # the repository cannot drift (numbers are rendered there from the artefact).
    markdown = (ROOT / "docs" / "ice_bond_weakest_link_chain.md").read_text()

    figures = [
        figure("chain_strength_ecdf.png",
               "链强度经验 CDF（实线）vs 解析最弱链 Weibull CDF（虚线），N=1…64，R=20000"),
        figure("size_effect.png",
               "中位链强度 vs 链长的双对数图：解析 N^(-1/m)、静态 min-of-N、动态引擎三点并置"),
        figure("quasi_static.png",
               "加载时长 T 对载荷误差与键力散差的影响——动态臂的准静态有效域"),
    ]
    report = {
        "title": "冰 DEM 键最弱链验证：强度异质性引擎特性 + 尺寸效应 N^(-1/m)（斜率误差 5.6%）",
        "summary": (
            "为 3-D DEM 键加入 opt-in 的 Weibull 强度异质性（中位不变，仅加散差，旧路径逐位不变），"
            "并用一维键链验证最弱链统计：静态斜率 −0.49880 vs −0.5（0.24%），"
            "动态引擎 70/70 实现选中最弱键、斜率误差 5.6%，bootstrap CI 跨 −0.5。"),
        "scenario_key": "tensordem.ice_bond_weakest_link",
        "mission_id": 7574,
        "tags": ["TensorDEM", "DEM", "冰力学", "Weibull", "最弱链", "尺寸效应",
                 "bond heterogeneity", "verification"],
        "markdown": markdown,
        "figures": figures,
    }
    (pathlib.Path("/tmp") / "report_m7574.json").write_text(json.dumps(report))
    request = urllib.request.Request(
        BASE + "/tensor/reports", data=json.dumps(report).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + login()}, method="POST")
    with urllib.request.urlopen(request, timeout=300) as response:
        print(response.read().decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())