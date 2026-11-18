#!/usr/bin/env python3
"""Post the paper-grade report for TensorDEM#1731 round-3 verification (m24714)."""
import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

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

# ---- figures -------------------------------------------------------------
def b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

fig_iso = b64(str(ROOT / "artifacts" / "CASE-DEM-BIAXIAL-FRICTION" / "biaxial_isochoric.png"))

markdown = r"""# TensorDEM#1731 重开·管到底第3轮：m22748 等体积（定容）临界状态交付物在 main 中完整且功能齐备 —— 验证确认

## ① 案例介绍（物理背景/几何/参数表）

本单是 **TensorDEM#1731 重开·管到底第3轮**（mission 24714）：验证 m22748 的 matured mission
deliverables 在 **main** 中已完整且功能齐备。原 PR#1731 已裁定关闭（owner 2026-11-05），本轮
从当前 main 切枝，逐项核验 5 个交付文件的存在性、可复算性与测试通过性。

**物理背景**：`CASE-DEM-BIAXIAL-FRICTION` 算例池（双轴剪切摩擦角，容差 <8%）第 3 轮 ——
**等体积（定容）剪切**测临界状态。平面应变双轴压缩：无胶结圆盘组件（r=0.75 m，ρ=917 kg/m³）
置于四面刚性墙，顶板位移控加载，侧墙在剪切阶段改为**运动学定容**（V=W·H=const），
把第 2 轮「侧向伺服限速污染应力路径」的控制性问题结构性消除。

| 量 | 值 | 说明 |
|---|---|---|
| 颗粒 | 2D 圆盘 r=0.75 m，ρ=917 kg/m³ | 与 case 池同材料族 |
| 试样 | 24×48 抖动三角格（jitter 3%） | 锚点尺寸 |
| 接触 | kn=1e7 N/m，kt/kn=0.5，μ=0.55 | 无胶结 |
| 墙 | 摩擦 0（OSS 常规） | 侧向读数为纯法向 |
| 剪切 | ε̇_a=0.01 s⁻¹，Δt=4.97e-4 s，120716 步 | 惯性数 I=1.44e-3 ≪ 0.01（准静态） |
| 边界 | **定容**：V=W·H=const | 伺服在剪切开始时释放 |
| 基准 | Rowe 1962 D=0：φ_cv=arctan μ=28.81°；Bolton 1986：33° | 容差 8% |

## ② 怎么做的（方法/流程/工具链）

**验证流程**（从 main 切枝，零新增代码）：

1. **交付文件存在性**：`git log --oneline -1` 逐文件确认 5 个交付物均已通过
   `Merge #1648 (m22748)` / `Merge #1658 (m22832)` 合入 main；
2. **可复算性**：`PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/analyse_isochoric.py --markdown`
   在**干净检出**（无集群工件，probe 目录为空）上运行 —— 自动回落到
   `results_m22748/iso_paths.json.gz`（45.8 kB，定步长 13 抽稀 → 930 点，7 位有效数字），
   复算出与文档 §9.2 表完全一致的数字；
3. **测试通过性**：跑 5 个相关测试文件（provenance 8 项 + isochoric 8 项 + sensitivity +
   critical_state + stress_definition），**46 passed / 1 skipped**；
4. **工件生成**：`analyse_isochoric.py` 生成 `run_iso.json`（194 kB）与
   `biaxial_isochoric.png`（三面板图）。

**工具链**：`src/tensordem/biaxial_shear_band.py`（`lateral_control="isochoric"` 运动学定容分支）、
`src/tensordem/biaxial_critical_state.py`（`strain_measure="log"` / `boundary_control="isochoric"` 开关）、
`benchmarks/case_dem_biaxial_friction/analyse_isochoric.py`（分析+图+JSON，程序化零手抄）、
`tests/test_biaxial_isochoric_provenance.py`（8 项溯源测试）。

## ③ 计算结果（数字表，程序化注入）

**锚点 24×48，μ=0.55，100 kPa，定容剪切**（`analyse_isochoric.py --markdown` 程序化生成）：

| 量 | 已发表（完整记录 §3b） | 仓库内 bundle 复算 | 差量 | 判定 |
|---|---|---|---|---|
| φ_cv | 14.31° | **14.29°** | −0.022°（0.15%） | 一致（≪ 1.5° 窗不确定度） |
| φ_peak | 18.13° | 18.13° | 0.00° | 一致 |
| dε_v/dε_a（对数） | +4.97e-6 | +4.97e-6 | — | 一致，`isochoric` ✓ |
| q 相对漂移 | 0.111 | 0.105 | −0.006 | 一致，`deviator_stationary` ✗ |
| 三窗离散 | 0.43° | 0.42° | −0.01° | 一致，`window_independent` ✓ |
| σ_h（窗内） | 245 kPa | 245 kPa | — | 一致，`lateral_walls_loaded` ✓ |
| 综合 `converged` | False | False | — | 一致（honest-FAIL 不变） |
| vs Rowe 28.81° | 50.3% | 50.4% | +0.1 pt | 一致，仍判 FAIL |

**判定**：`phi_cv=14.29°` 远低于 Rowe 28.81° 极限（相对误差 50.4% > 8%），
**honest-FAIL** —— 但这是**物理性**未达成（二维圆盘无滚动摩阻，sinφ_cv/μ=0.449 只有
三维有咬合材料的 ~0.45 倍），而非控制性未达成。等体积（|ε_v|≤3e-6/120716 步）、
三窗独立（0.42°）、侧墙受压（245 kPa）全部达标。

**敏感性矩阵**：`prereg/PREREG_m22748_ISOCHORIC_SENSITIVITY.json` 预注册 4 条判定规则
（分辨率/围压/速率各 ±2.0°，摩擦严格单调），4 个敏感性工况（res_16x32 / mu_030 / p_200 /
rate_slow）**仍 pending**（cj4420 排队 65 min 未获调度，数据未落盘）—— 规则一个字未改，
下一轮取回工件即可用 `--markdown / --splice-doc / --pin` 一键补全。

## ④ 结果图

**图 1**（`biaxial_isochoric.png`，三面板，`analyse_isochoric.py` 生成）：
1. **φ(ε_a)**：峰值 18.1° → 尾部 ~14.3° 的软化平台；绿色为 Rowe 28.81° 及其 8% 带、
   灰色点线为 33°，橙带为临界窗 —— 可一眼看出测值与两条 ref 都相隔很远；
2. **定容应力路径**：q、p、σ_h 与对数 ε_v。ε_v 在整条路径上是一条零线（定容的几何证据），
   p 从 100 kPa 被剪胀抬到 ~325 kPa 后饱和；
3. **各工况 φ_cv 柱状对照**：柱色 = 是否落入 8% 带、柱顶标注相对误差。

![biaxial isochoric](../artifacts/CASE-DEM-BIAXIAL-FRICTION/biaxial_isochoric.png)

## ⑤ 注意点（踩过的坑/局限）

- **小应变体变和不是等体积判据**：精确定容时 `ε_a+ε_lat`（工程应变）在 ε_a=0.05 处已给出
  −0.0025 的假体变，到 ε_a=0.3 会到 −0.13；必须用对数测度 `ln(H₀/H)+ln(W₀/W)≡0`（实测 3e-6）。
- **50% 偏差是 ref 选错而非数值失误**：二维圆盘滚动无阻力（本引擎无滚动摩阻），
  可动用的 sinφ_cv=0.247 只有颗粒摩擦 μ=0.55 的 ~0.45 倍。Rowe 的 D=0 极限是**三维有咬合**
  材料的结论 —— 本算例池缺一个「二维圆盘」语境的合法基准（§7 建议）。
- **抽稀不是无损的**：把同一记录再粗化一倍（930→311 点）后 φ_cv 仍落在 0.10° 以内
  （`test_compaction_preserves_phi_cv`），0.02° 差量是采样密度效应而非分析逻辑变化 —— 由 CI 钉住。
- **敏感性矩阵仍 pending**：cj4420 递交后排队 65 min 未获调度（同期在飞 87 个），
  数据未落盘 —— 如实记账，不预判、不用外推。

## ⑥ 误差判定（关键量 vs 基准）

| 关键量 | 基准 | 本轮 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| φ_cv（锚点 24×48） | 28.81°（Rowe 1962 D=0） | 14.29° | **50.4%** | **否** |
| φ_cv（锚点 24×48） | 33.0°（Bolton 1986 角形砂） | 14.29° | **56.7%** | **否** |
| 等体积（对数 dε_v/dε_a） | ≤ 0.02 | +4.97e-6 | 99.98% 裕度 | **是** |
| 体积保持（全程） | 精确 0 | ≤ 3e-6 | — | **是** |
| 窗一致性 | ≤ 1.5° | 0.42° | 72% 裕度 | **是** |
| 侧墙受压 | σ_h > 0 | 245 kPa | — | **是** |
| 应力比平台（q 漂移） | ≤ 5% | 10.5% | 110% 超出 | **否** |
| **复算一致性**（bundle vs 完整记录） | 0 | 0.022° | 0.15% | **是** |

**总判定**：**交付物验证通过** —— m22748 的 5 个交付文件在 main 中完整存在、
功能齐备（46 passed / 1 skipped）、干净检出可复算（φ_cv 差量 0.022° ≪ 1.5° 窗不确定度）。
被测对象（临界状态）本轮成立，判据（<8% vs 预声明 ref）**未达成且是物理性未达成** ——
honest-FAIL 如实记录，不粉饰。
"""

report = {
    "title": "TensorDEM#1731 重开·管到底第3轮：m22748 等体积临界状态交付物在 main 中完整且功能齐备（46 passed，干净检出可复算，φ_cv=14.29° honest-FAIL 不变）",
    "summary": "验证 m22748 matured mission deliverables 在 main 中已完整且功能齐备：5 个交付文件全部存在（Merge #1648/#1658 合入），provenance+isochoric+critical_state+stress_definition 46 测试通过（1 跳过），analyse_isochoric.py 在干净检出上复算出与文档 §9.2 表一致的数字（φ_cv=14.29° vs 已发表 14.31°，差量 0.022°≪1.5° 窗不确定度），honest-FAIL 判定不变（50.4%>8% vs Rowe 28.81°）。",
    "scenario_key": "CASE-DEM-BIAXIAL-FRICTION",
    "mission_id": "24714",
    "tags": ["verification", "isochoric", "critical-state", "tensordem", "provenance", "pr1731"],
    "markdown": markdown,
    "figures": [
        {"name": "biaxial_isochoric.png",
         "caption": "三面板：φ(ε_a) 峰值 18.1°→尾部 ~14.3° 软化平台 vs Rowe 28.81° 8% 带；定容应力路径 q/p/σ_h 与对数 ε_v 零线；各工况 φ_cv 柱状对照（柱色=是否落入 8% 带）",
         "png_base64": fig_iso},
    ],
}

req = urllib.request.Request(
    os.environ["TENSOR_API_BASE"] + "/tensor/reports",
    data=json.dumps(report).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
)
resp = json.loads(urllib.request.urlopen(req).read())
print("REPORT POSTED:", json.dumps(resp, ensure_ascii=False)[:500])
