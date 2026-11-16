#!/usr/bin/env python3
"""Publish the m23184 audit report (six-element, figures embedded) to the platform."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
BASE = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "agent")
PASS = os.environ.get("TENSOR_AGENT_PASS", "agent")


def fig(path: Path, caption: str) -> dict:
    return {"name": path.name, "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode()}


def main() -> int:
    summary = json.loads((ROOT / "runs" / "m23184_repro" / "summary.json").read_text())
    ens = summary["ensemble_flowing"]
    recs = summary["records"]
    rows = "\n".join(
        f"| `{r['label']}` | {r['seed']} | {r['W']:.6f} | {r['W_ref']:.6f} | "
        f"{100 * r['rel']:+.2f} % | {100 * r['counting_noise_rel']:.1f} % | {r['status']} |"
        for r in recs)
    dup = summary["verdict"]["same_seed_duplicates"][0]
    markdown = f"""## 摘要

对锚点 `CASE-DEM-SILO-BEVERLOO` 做了**档案复现性审计**：把仓库里同一冻结构型
（d=3 mm、D/d=5、k_n=5e3 N/m）的 5 条卸料记录全部收集、重算参考值并逐条判活。
结果发现：**同 seed、同冻结参数的两次归档运行相差 9.2 倍**，其中一条（m23114
ladder 的 dt_nom20_s1）根本不是测量值——料床在孔口上方成拱，8 个测量窗口中 5 个
零出料。旧口径把它当成 −88.5 % 的"测量"，并会污染 Richardson 外推。本单在
`discharge_status` 里加入成拱/停流判据（4 条声明式指标），把这类记录标注为
`ARRESTED`，同时不改动 `flow_rate_kg_s` 的历史语义。

## ① 案例介绍

平底料仓重力卸料，玻璃珠：粒径 d=3 mm、密度 2500 kg/m³；孔口 D=15 mm（D/d=5）；
仓半径 24 mm；装料高度 36 mm（H/D=2.4）；摩擦系数 μ=0.5、滚动摩擦 0.05、底板
光滑（μ_wall=0）；k_n=5e3 N/m；dt=T_c/20=1.8677e-05 s；静置 0.06 s、弃置 0.12 s；
测量 8×0.05 s，W 取后 50 % 窗口均值。参考律（Beverloo 1961，C=0.58、k=1.4，
从不重拟合）：W=C ρ_b √g (D−k d)^2.5。

## ② 怎么做的

1. 只读扫描本机各 worktree 的 `runs/*` 归档（m22885 系综 4 条 + m23114 ladder 1 条），
   按冻结签名筛选，避免混入其它口径；
2. 每条用同一参考式重算 W_ref 与相对误差，误差棒取计数噪声下限 1/√N（N=稳态窗口内
   流出颗粒数）；
3. 新增 `tensordem.silo_flow.beverloo.discharge_status()`：纯后处理判活函数，4 条
   判据（全程零出料 / ≥50 % 空窗口 / 速率 < 参考 20 % / 后前半段速率比 < 20 %），
   阈值写死在模块常量并声明在 docstring；`run_discharge` 记录新增
   `discharge_status` 与 `arrested` 两个键，`flow_rate_kg_s` 语义不变；
4. 用独立进程 + 不同 `PYTHONHASHSEED` 复跑 `build_bed`，比对颗粒坐标 sha256，
   确认内核本身逐位确定（排除"编译器/哈希序不确定"这条解释路径）；
5. `analyse.py` 从归档直读生成 summary.json、审计文档与两张图（零手抄）。

## ③ 计算结果（程序化生成）

| 记录 | seed | W [kg/s] | W_ref [kg/s] | 相对误差 | 噪声地板 | 判定 |
|---|---|---|---|---|---|---|
{rows}

同 seed 重复记录：`{dup['labels'][0]}` vs `{dup['labels'][1]}` →
W ∈ [{dup['W_range'][0]:.6f}, {dup['W_range'][1]:.6f}] kg/s，跨度
{100 * dup['W_spread_over_reference']:.1f} % of W_ref，状态分别为
{dup['statuses'][0]}/{dup['statuses'][1]} → **不可复现**。

活的 4 条（同 seed 1–4）：均值 {100 * ens['mean_rel']:+.2f} %、标准差
{100 * ens['sd_rel']:.2f} %、极差 {100 * ens['min_rel']:+.2f} % … {100 * ens['max_rel']:+.2f} %。

## ④ 结果图

两张图内嵌：逐记录相对误差（含 ±10 % 带与计数噪声棒）、seed-1 的两条窗口历史
（喷流 vs 滴漏）。

## ⑤ 注意点

1. 归档来自其它 worktree，只读引用，未改动；
2) 该构型单条记录的计数噪声地板 ≈7–8 %，**任何单点 PASS/FAIL 都不可外推**；
3. 双稳态（成拱 / 流动）意味着系综均值依赖 seed 数与"是否剔除成拱记录"，必须在
   报告里同时给出成拱比例；
4. 只对 D/d=5、单分散、光滑底板成立，不外推锥斗/壁摩擦/多分散；
5. 漂移、ρ_b 口径（标称 vs 实测料面）问题沿用前几单结论，本单未改。

## ⑥ 误差判定

- 关键量 W 对 Beverloo 参考：活的 4 条 {100 * ens['min_rel']:+.2f} % … {100 * ens['max_rel']:+.2f} %，
  均值 {100 * ens['mean_rel']:+.2f} %，**点估计 <10 % 容差内、但 95 % CI 宽且有条目出带**；
- 成拱记录 {100 * (recs[-1]['rel'] if recs else 0):+.1f} % 级别，属无效测量，不计入统计；
- **是否 <3 %？否**，且该构型本质上达不到：噪声地板 7–8 %，要 3 % 需把稳态窗口内
  流出颗粒数提到 ≈1100（窗口加长或床放大 ~3 倍）。
"""
    payload = {
        "title": "CASE-DEM-SILO-BEVERLOO 档案复现性审计：同 seed 两次归档相差 9.2 倍，新增成拱判据 (m23184)",
        "summary": ("同一冻结构型下，m22885 seed-1 记录 W=0.030750 kg/s (+5.5 %) 与 m23114 "
                    "ladder seed-1 记录 W=0.003358 kg/s (-88.5 %) 相差 9.2 倍，后者 8 个窗口 "
                    "中 5 个零出料——是料床成拱，不是流率测量。新增 discharge_status() 四条"
                    "声明式判据把这类记录标为 ARRESTED，并对锚点全部归档记录逐条复核。"),
        "scenario_key": "dem.silo.beverloo",
        "mission_id": 23184,
        "tags": ["TensorDEM", "silo_flow", "Beverloo", "reproducibility", "arch", "honest-FAIL"],
        "markdown": markdown,
        "figures": [
            fig(ROOT / "docs" / "figures" / "m23184_repro_errors.png",
                "各归档记录相对 Beverloo 参考的误差；红=ARRESTED（成拱），蓝=FLOWING；"
                "误差棒为计数噪声地板 1/√N，绿带为 ±10 % 容差。"),
            fig(ROOT / "docs" / "figures" / "m23184_repro_windows.png",
                "seed 1 的两条窗口历史：正常喷流 ~40 颗粒/窗口 vs 成拱记录滴漏 0–15 颗粒/窗口。"),
        ],
    }
    tok = requests.post(f"{BASE}/auth/login",
                        json={"username": USER, "password": PASS}, timeout=30).json()
    token = tok.get("access_token") or tok.get("token")
    r = requests.post(f"{BASE}/tensor/reports", json=payload, timeout=120,
                      headers={"Authorization": f"Bearer {token}"})
    print(r.status_code, r.text[:400])
    return 0 if r.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())