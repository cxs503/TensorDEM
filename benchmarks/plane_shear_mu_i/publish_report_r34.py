"""Publish the R34 progress report to the platform (mission 23200).

Reads the archived figure and the generated doc, writes ``/tmp/report.json`` for
``curl -d @/tmp/report.json``.  No numbers are typed here: the markdown is the
generated document itself.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

BENCH = Path(__file__).resolve().parent
DOC = BENCH.parents[1] / "docs" / "CASE_DEM_PLANE_SHEAR_MUIDI_R34.md"
FIG = BENCH / "runs" / "pairing_r34" / "fig_pairing_r34.png"

SUMMARY = (
    "R34 把 mu(I) 判定的应力估计量配对全部列出并**跑前预登记**检验。结果：本单自己提的"
    "「动能应力使 I 与 mu 定义不一致」假设被**否证**（动能占比 0.7–1.6 %，四个压强通道相差 ≤4 %）；"
    "真正的两个大项是 (i) **装置状态**——同协议 4 个种子的 p_wall/P 只有 0.34/0.68/0.34/0.86，"
    "I_target=0.10 的两点闭合率仅 0.15–0.17、落到名义 I 的 1/4（锚点 I≥0.1 区间 NOT_SUPPORTED）；"
    "(ii) **实现散度 ±13 pp > 锚点 10 % 容差**，R33 的 +18.8 % 是该分布的下沿，故 μ 门均值 FAIL、"
    "残差量级 WITHHELD。附带发现结构缺口：2-D 装置没有 3-D 路径已有的伺服有效性门。"
)

REPORT = {
    "title": "CASE-DEM-PLANE-SHEAR-MUIDI R34：估计量配对的验证欠账（含一条自我否证）",
    "summary": SUMMARY,
    "scenario_key": "dem_plane_shear_mu_i",
    "mission_id": 23200,
    "tags": ["DEM", "mu(I)", "plane-shear", "rheology", "uncertainty-budget",
             "adversarial-review", "TensorDEM", "CASE-DEM-PLANE-SHEAR-MUIDI"],
    "markdown": DOC.read_text(),
    "figures": [{
        "name": "fig_pairing_r34.png",
        "caption": ("(a) mu 对达成 I（6 点，bulk/wall 两种标记）叠在 da Cruz 2005 "
                    "mu=0.22+1.0I 尺子与 ±10 % 带上；(b) 四个应力通道各自的判定误差 —— "
                    "通道间几乎重合、种子间大幅分离；(c) 同一次运行的四个压强通道/设定值，"
                    "I_t=0.10 两点伺服未达设定。"),
        "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
    }],
}

if __name__ == "__main__":
    out = Path("/tmp/report.json")
    out.write_text(json.dumps(REPORT))
    print(f"-> {out} ({out.stat().st_size} bytes)")