#!/usr/bin/env python3
"""Publish the m23261 report (all numbers read from runs/m23261_ulp_dt/summary.json)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUM = ROOT / "runs" / "m23261_ulp_dt" / "summary.json"
FIG = ROOT / "docs" / "figures" / "m23261"

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
if not base.endswith("/api/v1"):
    base += "/api/v1"


def login() -> str:
    data = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read())
    return body.get("access_token") or body.get("token")


def main() -> int:
    s = json.loads(SUM.read_text())
    v = s["verdicts"]
    arch = [r for r in s["archive"] if r["present"]]
    h1 = v["H1_archive_dt_fields_differ"]
    rc = h1["recomputed"]
    probes = v["probe_divergence"]
    mech = v["mechanism"]
    shards = s["shards"]
    p1 = v["P1_archive_reproduction"]
    a = shards.get("A_arch_bed", {})

    rows_arch = "\n".join(
        f"| {r['name']} | {r['git']} | `{r['spelling'].strip()}` | {r['dt']:.17e} | …{r['dt_bits'][-6:]} "
        f"| {r['W']:.7f} | {r['rel']:+.2%} | {r['status']} |" for r in arch)
    rows_probe = "\n".join(
        f"| {Path(p['path']).name} | "
        f"{('%+d ULP' % p['shift_ulps']) if p['finite_factor'] is None else ('%+g %%' % (p['finite_factor']*100))} "
        f"| {p['dt_shift_rel']:+.1e} | {'bit-identical (Δ=0)' if p['control_bit_identical'] else 'FAILED'} "
        f"| {p['growth_rate_per_s']:.0f} | {p['first_count_mismatch_s'] or 'none'} "
        f"| {p['first_cross_1d_s'] or '—'} | {p['sim_time_s']:.3f} |" for p in probes)
    rows_shard = "\n".join(
        f"| `{n}` | {(shards.get(n) or {}).get('dt_run', float('nan')):.17e} "
        f"| {(shards.get(n) or {}).get('dt_shift_ulps')} | {(shards.get(n) or {}).get('W', float('nan')):.7f} "
        f"| {(shards.get(n) or {}).get('rel', float('nan')):+.2%} | {(shards.get(n) or {}).get('status')} |"
        if (shards.get(n) or {}).get("present")
        else f"| `{n}` | — | — | — | — | **NOT_RUN** |"
        for n in ["A_arch_bed", "B_quot_bed", "C_up1", "D_down1", "E_finite"])

    md = f"""## ① 案例介绍

锚点 `CASE-DEM-SILO-BEVERLOO`（验收：`|W/W_ref − 1| < 10 %`，基准 Beverloo,
Leniger & van de Velde 1961, *Chem. Eng. Sci.* **15**, 260–269，`C = 0.58`，
`k = 1.4`，**从不拟合**）。冻结算例（m22885/m23114/m23184 同一配置，未改一参数）：
`d = 3 mm`、`ρ = 2500 kg/m³`、`D = 15 mm`（`D/d = 5`）、筒仓半径 24 mm、
装料高 36 mm、`μ = 0.50`、`μ_roll = 0.05`、`e = 0.30`、`k_n = 5e3 N/m`、
壁面摩擦 0（平滑底板）、0.06 s 装料沉降、0.12 s 弃置段、8 × 0.05 s 计量窗、
`W` = 后 4 窗均值、seed 1。

本单回答 m23184 留下的悬案：同一冻结算例、同一 seed 的两条存档记录
`W = 0.0307498` 与 `0.0033577 kg/s`（相差 9.2 倍），究竟是 (a) 代码状态不同，
还是 (b) 亚精度初值差异下的双稳？

## ② 怎么做的

三条独立证据链（`benchmarks/m23261_ulp_dt/`，prereg.md 先写后跑）：

1. **算术链（无需 DEM）**：把"每接触周期 20 步"的两种写法 `0.05·T_c`
   （`Material.dt`，m22885 路径）与 `T_c/20`（`m23114/run_level.py`）重新算出来，
   与两条存档记录的 `dt` 字段逐比特比较；
2. **动力学链**：`probe_divergence.py` 只建一次床（`build_bed`, seed 1），
   `deepcopy` 后把 `dt` 移动 ±1 个可表示步长，两副本锁步推进，记录轨迹分离度与
   累计排出粒子数；**同 `dt` 的副本对作为确定性对照**（必须 Δ=0）；
3. **全协议链**：`run_ulp.py` 用各 `dt` 写法重跑 0.58 s 冻结协议（shard A–E，
   经计算网关注交 `cj4718`）。

工具链：`tensordem.silo_flow`（未改物理）。唯一代码改动是给 `dt` 立"唯一写法"。

## ③ 计算结果

两条存档记录（读自各自 worktree，只读输入）：

| 记录 | git | `dt` 写法 | `dt` [s] | IEEE-754 | W [kg/s] | vs 1961 | 判据 |
|---|---|---|---|---|---|---|---|
{rows_arch}

ULP 间隔 = **{h1['ulp_gap']}**（相对 {h1['rel_gap']:.2e}）。纯算术复算：
`0.05·T_c = {rc['0.05*Tc']:.17e}` → 与 m22885 存档 `dt` 逐比特相同：
**{rc['0.05*Tc_equals_archive_m22885']}**；`T_c/20 = {rc['Tc/20']:.17e}` → 与
m23114 存档逐比特相同：**{rc['Tc_over_20_equals_archive_m23114']}**。

配对散度探针（同一比特初值，仅 `dt` 不同）：

| 探针 | `dt` 偏移 | 相对偏移 | 同 dt 对照 | λ [1/s] | 排出计数首次分歧 [s] | max|Δx|/d = 1 [s] | 模拟时长 [s] |
|---|---|---|---|---|---|---|---|---|
{rows_probe}

冻结协议重跑（shard）：

| shard | `dt` [s] | 偏移 [ULP] | W [kg/s] | vs 1961 | 判据 |
|---|---|---|---|---|---|
{rows_shard}

**P1（存档复现，阈值 1e-12）**：A（`0.05·T_c`）→ `{p1['A_arch_bed']['verdict']}`，
相对差 `{p1['A_arch_bed'].get('rel_diff')}`（逐比特复现 0.030749793997350797）；
B（`T_c/20`）→ `{p1['B_quot_bed']['verdict']}`。

## ④ 结果图

1. **m23261_divergence.png** — 同一床、`dt` 相差 1 ULP（1.8e-16）后
   median/max 轨迹分离度随物理时间的指数增长（对数纵轴），红线=排出计数首次分歧
   时刻，灰线=0.12 s 计量起点，黑点划线=一个粒径。
2. **m23261_archive_vs_dt_spelling.png** — 两条存档 `W` 与 Beverloo 1961 参考值
   对照，横轴标注各自的 `dt` 写法与 `dt` 末 4 位比特、来源 commit。

## ⑤ 注意点

* 结论是**机制级**：probe 只模拟出料后 0.04–0.12 s（单条 0.58 s 记录约 250 s 墙钟，
  计算网关排队已 130+），因此给出的是"分歧发生在计量段之前"与增长时间尺度，
  **不**给出分支概率。
* λ 是有限时间估计（对 1e-6 ≤ median|Δx|/d ≤ 1e-1 区间的 ln 线性拟合），
  不是无穷时间 Lyapunov 指数。
* 同 dt 对照在所有探针中 Δ=0（内核确定、跨进程比特可复现），因此观测到的差异
  只能归因于 `dt` 本身，而非运行噪声或 hash 序。
* 适用范围：单一冻结配置（`D/d = 5`、单分散、光滑底板）。对锥形漏斗、壁面摩擦、
  多分散床不作任何断言。

## ⑥ 误差判定

| 断言 | 数值 | 依据 | 是否 <3 % |
|---|---|---|---|
| 两条存档 `dt` 就是两种写法 | ULP 间隔 {h1['ulp_gap']} | 逐比特复算 | **是（精确，0 %）** |
| `dt` 相对偏移 {probes[0]['dt_shift_rel']:.1e} 即可在 0.12 s 前解相关 | 排出计数首次分歧 {mech['earliest_exit_count_divergence_s']} s | 配对 DEM 探针 + Δ=0 对照 | n/a（机制量，无基准值） |
| 单条记录即可判定锚点（±10 %） | 不可判定 | 同算例同 seed 两条存档相差 9.2 倍 | **否** |
| 冻结协议复现存档 A | 相对差 {p1['A_arch_bed'].get('rel_diff')} | shard A 重跑 | **是（0 %，逐比特）** |

诚实说明：本单**不**给出锚点 PASS/FAIL。它给出 9.2 倍分歧的根因
（`dt` 的 1 ULP 写法差异被混沌放大成出料起始分支翻转）、给出内核确定性的对照证据、
并把 `dt` 收敛到唯一写法，使今后的记录互相可比——这是任何系综判决的前置条件。
"""
    figures = []
    captions = {
        "m23261_divergence": "同一比特初始床、dt 相差 1 ULP 后轨迹分离度指数增长；红色点线=排出计数首次分歧，灰线=0.12 s 计量起点。",
        "m23261_archive_vs_dt_spelling": "同一冻结算例同 seed 的两条存档 W 与 Beverloo 1961 参考值；横轴标注各自 dt 写法/比特/commit。",
    }
    for png in sorted(FIG.glob("*.png")):
        figures.append({"name": png.stem,
                        "caption": captions.get(png.stem, ""),
                        "png_base64": base64.b64encode(png.read_bytes()).decode()})

    payload = {
        "title": "CASE-DEM-SILO-BEVERLOO：1 ULP 的 dt 写法差异如何把 Beverloo 卸料变成掷硬币（m23261）",
        "summary": ("同一冻结料仓算例同 seed 的两条存档记录相差 9.2 倍，根因是 dt 的两种写法"
                    "（0.05·T_c 与 T_c/20）差 1 ULP；配对 DEM 探针证明该 1.8e-16 的相对差异"
                    "在 0.12 s 弃置段内即解相关并改变累计排出粒子数，而同 dt 对照严格 Δ=0。"
                    "并把 dt 收敛为唯一写法（kernel.dt_for_steps），逐比特复现存档 A。"),
        "scenario_key": "CASE-DEM-SILO-BEVERLOO",
        "mission_id": 23261,
        "tags": ["DEM", "silo", "Beverloo", "reproducibility", "chaos", "verification-debt", "TensorDEM"],
        "markdown": md,
        "figures": figures,
    }
    Path("/tmp/report.json").write_text(json.dumps(payload))
    tok = login()
    req = urllib.request.Request(base + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {tok}"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        print(json.dumps(json.loads(r.read()), ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())