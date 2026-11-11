#!/usr/bin/env python3
"""Post the board-audit report for m14681 (GAP-DEM-CLUMP) — mission 14979.

All numbers are from the actual re-run of the C1/C2 legs (no hand transcription)
and from the committed docs tables.  Payload path is mission-unique.
"""
from __future__ import annotations
import base64, json, os, urllib.request
from pathlib import Path

# --- verified numbers from the actual re-run (C1/C2 legs) ---
C1_SPHERE = 0.0            # bit-exact vs 2/5 m r^2
C2_AR3_PROD = 0.0030975    # a/b=3 production voxel vs analytic (worst)
C2_AR3_FINE = 0.0002511    # a/b=3 fine voxel 192^3 vs analytic
# --- committed docs numbers (program-generated, from results.json) ---
C1_OVERLAP = 6.5383e-05    # overlapping pair vs 384^3 voxel (inertia)
C1_TRIPLE3 = 3.8675e-05    # three-sphere cluster 3rd order vs 384^3 voxel
C1_TRIPLE2 = 6.6275e-04    # 2nd order (reference)
C3_MAXDELTA = 2.174        # max repose delta vs sphere (deg), criterion >=3

md = f"""## 1 案例介绍
交差核验单（非重做）：m14681 = GAP-DEM-CLUMP（TensorDEM 仓）——非球形颗粒（多球刚性 clump）生成、转动惯量解析核验、休止角形状效应。本单核验三件套是否真存在并判级落表。

## 2 怎么做的
① 取证：读 m14681 的 result/报告（tensor_reports）/工作树，逐项核验三件套（run_artifacts 文件、benchmark ref vs got 数值、repro_cmd 可执行）。
② 判级（L0-L5）：级别=功能实质。
③ 落表：INSERT INTO tensor_maturity_verdicts（repo=TensorDEM, scope='m14681 GAP-DEM-CLUMP', mission_id=14979, verdict_by='board-audit'）。

## 3 计算结果（核验三件套）
| 件 | 状态 | 证据 |
|---|---|---|
| run_artifacts 文件 | **缺失** | artifacts/GAP-DEM-CLUMP/{{results.json,criteria.json,gap_dem_clump.png,repro_cmd.txt}} 在原始 commit 981a9c1e 中，但 merge 38ed2598 被门禁滤除（'5 个文件被门禁滤除'）；当前工作树无（git ls-files 空） |
| benchmark ref vs got | **存在** | docs/GAP_DEM_CLUMP_TABLES.md 程序生成；重跑 C1/C2 腿复现：C1 球=0.0（位精确）、C2 a/b=3 生产体素=0.0031、细体素=2.5e-04，与表一致 |
| repro_cmd 可执行 | **存在** | benchmarks/gap_dem_clump/run_gap_dem_clump.py 存在且可执行（C1/C2 腿已复现） |

## 4 结果图
本单为交差核验（非重做），无新场量云图。原节点图 gap_dem_clump.png（左：惯量误差 vs 解析含 1% 判据线；右：休止角 vs 长径比含文献带）在原始 commit 中，但被门禁滤除、当前工作树缺失——这是 run_artifacts 缺失的直接证据。

## 5 注意点
① 门禁滤除是平台机制：原始 commit 981a9c1e 含 12 文件（含 4 个 run_artifacts + log），merge 时 5 个被滤除（results.json/criteria.json/png/repro_cmd.txt/log）。这不是工作缺陷，但使"交差核验"无法从已提交工件确认 L4 基准证据。
② 实现本身真实：clump_inertia.py（374 行）是真实解析并集惯量核（球冠矩/lens/容斥至 3 阶），20 个测试全过（12.96s），C1/C2 腿复现精确数值。
③ C3 休止角 honest-FAIL：非单调、对球最大增量 +2.17°（判据 ≥3°），并定位机制（mu_roll 控制符号：0→+9.46°，0.6→−7.17°）与混淆项（粒子数 648→104）——这是第 2 轮靶子，非本单重做范围。

## 6 误差判定
- C1 解析核 vs 闭式解：0.0（<1e-12 判定线）→ **<3% 满足**（位精确）。
- C2 生产 clump 惯量 vs 解析：最差 0.31%（判据 1%）→ **<3% 满足**。
- C3 休止角 vs 文献：球基线 25.82° vs 文献带 22-25°（高 0.8°，+3.3% 相对带顶）；趋势判据 FAIL（非单调、Δmax +2.17° < 3°）→ **不满足**，honest-FAIL。
- 判级：**L3**（功能实质=真实解析核 L2 + 集成生产 ClumpParticle L3 + 可复现基准；但 run_artifacts 被门禁滤除、已提交基准证据不完整 → L4 不可从已提交工件确认）。
"""
payload = {
    "title": "m14681 GAP-DEM-CLUMP 交差核验：实现真实(L3)、基准可复现，但 run_artifacts 被门禁滤除",
    "summary": f"三件套核验：run_artifacts 缺失（门禁滤除 5 文件）、benchmark ref vs got 存在（重跑 C1/C2 复现 C1=0.0/C2=0.0031）、repro_cmd 可执行；判级 L3（真实解析核+集成+可复现基准，但已提交基准证据不完整）；C3 honest-FAIL 已定位机制与混淆项",
    "scenario_key": "gap_dem_clump_audit",
    "mission_id": 14979,
    "tags": ["TensorDEM", "clump", "inertia", "analytic", "audit", "maturity-verdict", "L3", "GAP-DEM-CLUMP"],
    "markdown": md,
    "figures": [],
}
OUT = Path('/tmp/report_m14979.json')
OUT.write_text(json.dumps(payload))
print('report bytes', len(json.dumps(payload)))


def post() -> None:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    data = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        tok = json.loads(resp.read())
    token = tok.get("access_token") or tok.get("token")
    if not token:
        raise SystemExit(f"login failed: {tok}")
    body = OUT.read_bytes()
    req = urllib.request.Request(base + "/tensor/reports", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {token}"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        print(resp.status, resp.read()[:300].decode())


if __name__ == "__main__":
    post()
