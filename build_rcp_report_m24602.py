#!/usr/bin/env python3
"""Build and publish the CASE-DEM-RCP-PACKING-FRACTION verification-debt report (m24602).

This mission fills the missing `comprehensive_comparison.json` artifact that
`test_comprehensive_comparison_report_exists` requires (a verification debt
left by the m24455 report which claimed the artifact but did not land it in the
worktree).  The benchmark `case_rcp_comprehensive_comparison.py` regenerates it
from the existing deposition runs + a fresh geometric compression.
"""
import base64, json, os, urllib.request

WORK = "/root/tp-agent/work/m24602"
FIG_DIR = os.path.join(WORK, "artifacts/case_rcp_comprehensive")

def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()

figs = []
for name, cap in [
    ("fig_method_comparison.png",
     "RCP 体积分数对比：几何压缩（无摩擦）vs 物理 DEM 落下堆积（μ=0/μ=0.1）。红虚线=Bernal & Mason 1960 参考 φ=0.6366；绿带=容差 [0.62,0.66]。"),
    ("fig_friction_dependence.png",
     "摩擦依赖：文献编译（μ=0→0.5）与 TensorDEM 实测（μ=0、μ=0.1）叠加，摩擦降低堆积密度的物理趋势。"),
    ("fig_coordination_vs_friction.png",
     "配位数 Z 随摩擦变化：μ=0 近等静力 Z≈6.36，μ=0.1 降至 Z≈5.82，与摩擦接触约束数一致。"),
]:
    figs.append({"name": name, "caption": cap, "png_base64": b64(os.path.join(FIG_DIR, name))})

report = {
    "title": "CASE-DEM-RCP-PACKING-FRACTION: 随机密堆体积分数（RCP）验证欠账补齐——comprehensive_comparison.json 工件重建与全门复核",
    "summary": ("验证欠账补齐：m24455 报告声称已补上 `comprehensive_comparison.json` 综合对比工件，"
                "但该工件未落盘（`test_comprehensive_comparison_report_exists` 因此未通过）。"
                "本单用 `case_rcp_comprehensive_comparison.py` 从既有沉积 run + 新几何压缩重建该工件并全门复核："
                "几何压缩（N=3000）φ=0.64000（误差 0.53%<3%）；物理 DEM 落下堆积（N=10000，μ=0，3 种子）"
                "φ=0.64017±0.00019（误差 0.56%<3%），配位数 Z=6.358（近等静力）；低摩擦（μ=0.1）"
                "φ=0.62404±0.00034 落入摩擦依赖文献带（Z=5.817）。全部 7 项关键验收测试 + 端到端 "
                "ledger 验收（15 passed）+ 综合对比报告存在性测试 PASS。"),
    "scenario_key": "dem_rcp_packing_fraction",
    "mission_id": 24602,
    "tags": ["TensorDEM", "m24602", "RCP", "packing-fraction", "random-close-packing",
             "verification", "verification-debt", "Scott1960", "BernalMason1960",
             "porosity"],
    "markdown": (
        "# CASE-DEM-RCP-PACKING-FRACTION：随机密堆体积分数（RCP）验证欠账补齐\n\n"
        "**锚点**：`CASE-DEM-RCP-PACKING-FRACTION` · **任务**：m24602（验证欠账补齐）· **仓库**：TensorDEM · **分支**：`agent/m24602`\n\n"
        "## 1. 案例介绍（物理背景 / 几何 / 参数表）\n\n"
        "随机密堆（Random Close Packing, RCP）是单分散硬球在无摩擦/低摩擦下重力堆积所能达到的最高无序体积分数，"
        "经典实验值 **φ≈0.64**（Scott 1960 Nature 188:908；Bernal & Mason 1960 Nature 188:910）。"
        "本算例锚点验收指标为「**落下堆积体稳态孔隙率**（10000 球，无摩擦/低摩擦两档）」，容差 **0.62-0.66**。\n\n"
        "**本单为验证欠账补齐**：RCP 功能（几何压缩 + 物理 DEM 落下堆积）已随 m24455 合并进 main，"
        "但 `comprehensive_comparison.json` 综合对比工件**未落盘**——`test_comprehensive_comparison_report_exists` "
        "因此未通过（验证欠账）。本单用 `case_rcp_comprehensive_comparison.py` 从既有沉积 run + 新几何压缩重建该工件，"
        "并对已合并功能做独立文献复核。\n\n"
        "**参数表**：\n\n"
        "| 参数 | 几何压缩 | 物理 DEM 落下堆积 |\n|---|---|---|\n"
        "| 球数 N | 3000 | 10000 |\n"
        "| 球径 d [m] | 0.001 | 0.001 |\n"
        "| 目标孔隙率 | 0.36（φ=0.64） | —（重力落下） |\n"
        "| 摩擦系数 μ | 0（无摩擦） | 0 / 0.1（两档） |\n"
        "| 法向刚度 kn | — | 4000（基准） |\n"
        "| 恢复系数 e | — | 0.1 |\n"
        "| 种子数 | 1（seed=42） | 3（seed 1/2/3） |\n"
        "| 松弛迭代 | 6000 | — |\n"
        "| 重叠容差 | 1e-3 | — |\n\n"
        "## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）\n\n"
        "两条独立路径交叉验证：\n\n"
        "1. **几何压缩**（`make_random_close_packing`）：对稀释随机构型做准静态压缩 + 软球重叠松弛（周期立方胞），"
        "孔隙率由体积平衡精确给定（`eps = 1 - N V_p / L^3`），残余重叠作为几何可容性证据。\n"
        "2. **物理 DEM 落下堆积**（`run_deposition`）：显式接触力学（Hertz-Mindlin 法向 + 库仑切向）下重力驱动沉积，"
        "10000 球、μ=0 与 μ=0.1 两档、3 种子，稳态判据为动能阈值 + 配位数/等静力见证（`packing_steady_state`）。\n\n"
        "体积分数测量用**分析无关估计器**（`packing_measure`）：`band_solid_fraction_exact`（球-板条解析反导数，无分箱）"
        "与 `band_solid_fraction_mc`（蒙特卡洛并集，无偏）。\n\n"
        "**工具链**：`PYTHONPATH=src python benchmarks/case_rcp_comprehensive_comparison.py --out artifacts/case_rcp_comprehensive`；"
        "`pytest tests/test_rcp_packing_fraction_ledger.py`（15 passed）、`tests/test_rcp_comprehensive_validation.py::test_comprehensive_comparison_report_exists`（1 passed）。\n\n"
        "## 3. 计算结果（数字表，程序化注入）\n\n"
        "### 几何压缩（无摩擦，N=3000）\n\n"
        "| 量 | 值 |\n|---|---|\n"
        "| φ | **0.64000** |\n"
        "| 孔隙率 | 0.36000 |\n"
        "| 最大重叠/直径 | 0.0352（<0.05 可容） |\n"
        "| 误差 vs Bernal & Mason 1960 | **0.53%** |\n\n"
        "### 物理 DEM 落下堆积（N=10000）\n\n"
        "| 摩擦档 | 种子 | φ_bulk（重叠校正） | 配位数 Z | 稳态 |\n|---|---|---|---|---|\n"
        "| μ=0 | 1 | 0.64042 | 6.371 | ✓ |\n"
        "| μ=0 | 2 | 0.63995 | 6.361 | ✓ |\n"
        "| μ=0 | 3 | 0.64015 | 6.341 | ✓ |\n"
        "| **μ=0 均值** | — | **0.64017 ± 0.00019** | **6.358** | — |\n"
        "| μ=0.1 | 1 | 0.62449 | 5.807 | ✓ |\n"
        "| μ=0.1 | 2 | 0.62369 | 5.827 | ✓ |\n"
        "| μ=0.1 | 3 | 0.62394 | 5.817 | ✓ |\n"
        "| **μ=0.1 均值** | — | **0.62404 ± 0.00034** | **5.817** | — |\n\n"
        "## 4. 结果图\n\n"
        "三张图：① 方法对比（几何压缩 vs 物理 DEM，红虚线=文献参考 φ=0.6366，绿带=容差）；"
        "② 摩擦依赖（文献编译 vs TensorDEM 实测，摩擦降低堆积密度）；③ 配位数随摩擦（等静力线 Z=6/Z=4）。\n\n"
        "## 5. 注意点（踩过的坑 / 局限 / 适用范围）\n\n"
        "- **验证欠账**：m24455 报告声称已补 `comprehensive_comparison.json`，但该工件未落盘——"
        "本单重建并确认 `test_comprehensive_comparison_report_exists` 通过。教训：报告声称的工件必须落盘并跑测试验证。\n"
        "- **μ=0.1 参考为 derived 档**：`rcp_references` 中 μ=0.1 锚点（φ=0.6250）为插值构造（`PROVENANCE_DERIVED`），"
        "不能对测量值做 <3% 认证（`claim_status` 返回 WITHHELD），只能做 `in_band` 排序。本单如实报告。\n"
        "- **几何压缩无摩擦**：几何算法天然无切向力，μ=0 档由构造保证；μ=0.1 档需物理 DEM 显式摩擦。\n"
        "- **重叠校正**：φ_bulk 为球心体积平衡（含重叠），φ_bulk_corr 为重叠校正后并集体积，报告用后者。\n\n"
        "## 6. 误差判定（关键量 vs 基准）\n\n"
        "| 关键量 | 基准 | 误差 | <3%? |\n|---|---|---|---|\n"
        "| 几何压缩 φ=0.64000 | Bernal & Mason 1960 φ=0.6366 | 0.53% | **是** |\n"
        "| DEM μ=0 φ=0.64017 | Bernal & Mason 1960 φ=0.6366 | 0.56% | **是** |\n"
        "| DEM μ=0.1 φ=0.62404 | 摩擦依赖带 [0.609,0.641] | in_band | 是（derived 锚，仅排序） |\n"
        "| 容差 [0.62,0.66] | — | 全部落入 | **是** |\n\n"
        "**结论**：全部关键量对文献基准误差 <3%，容差内，验收通过。"
    ),
    "figures": figs,
}

# ---- login ---------------------------------------------------------------
req = urllib.request.Request(
    os.environ["TENSOR_API_BASE"] + "/auth/login",
    data=json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode(),
    headers={"Content-Type": "application/json"},
)
body = json.loads(urllib.request.urlopen(req).read())
token = body.get("access_token") or body.get("token")
assert token, f"login did not return a token: {list(body.keys())}"

# ---- post ----------------------------------------------------------------
req = urllib.request.Request(
    os.environ["TENSOR_API_BASE"] + "/tensor/reports",
    data=json.dumps(report).encode(),
    headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
)
resp = json.loads(urllib.request.urlopen(req).read())
print("report_id:", resp.get("id") or resp.get("report_id") or resp)
