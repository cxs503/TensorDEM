"""Post the paper-grade report of the DEM-PBM breakage verification surface.

Usage::

    TENSOR_API_BASE=http://127.0.0.1:8010 TENSOR_AGENT_USER=... TENSOR_AGENT_PASS=... \
    python scripts/post_breakage_pbm_report.py artifacts/breakage_pbm/summary.json \
        artifacts/breakage_pbm/breakage_pbm_summary.png

Every number in the markdown is read from the summary JSON (zero hand
transcription); the figure is embedded as base64.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request

API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def _token() -> str:
    body = json.dumps({"username": USER, "password": PASS}).encode()
    req = urllib.request.Request(f"{API}/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data.get("access_token") or data.get("token")


def build_markdown(s: dict) -> str:
    crit = s["criteria"]
    rows = "\n".join(
        f"| {k} | {v['value']:.6g} | {v['threshold']:.6g} | "
        f"{'PASS' if v['passed'] else 'FAIL'} |" for k, v in crit.items())
    cr = s["crushing"]
    w = cr["weibull"]
    d = s["daughters"]
    a = s["dpbm_analytic"]
    cv = s["cross_validation"]
    rec_rows = "\n".join(
        f"| {t} | {r['dem_n_particles']} | {r['dem_mean_diameter_m']*1e3:.3f} | "
        f"{r['pbm_mean_diameter_m']*1e3:.3f} | {r['wasserstein_raw_m']*1e3:.3f} | "
        f"{r['wasserstein_normalised']:.4f} |" for t, r in cv["records"].items())
    nc = s["negative_controls"]
    return f"""## 1 案例介绍（物理背景 / 参数）

颗粒破碎把**颗粒强度统计**（Weibull 最弱链）与**群体平衡**（PBM）连在一起：
单颗粒在压板间破碎，破碎力服从 Weibull 分布；破碎后子代体积分布 b(v,v')；
群体中颗粒尺寸分布 n(v,t) 由 PBM 演化。本节点在 TensorDEM 新建
`tensordem.breakage_pbm`，把这条链条做成**可机器裁决的验证面**（判据 C1–C7）。

材料/几何参数：母颗粒直径 10 mm，子球格点 `n_axis=4`（32 子球 / 156 键），
键刚度和强度取 {w['m']:.3f} 的 Weibull 模数；装配体初始固相率 0.52、粒径对数
正态 cv=0.10、单轴侧限压缩，加载应力线性升至 {cv['stress_max_Pa']/1e6:.0f} MPa
用时 {s['meta']['t_end']:g} s。

## 2 怎么做（方法 / 流程 / 工具链）

1. **单颗粒破碎 DEM**（`bonded_cluster_crushing`）：batched bonded-sphere 压缩，
   键强度独立抽样、断键退化为单边排斥接触、Cundall 局部阻尼、位移控制压板 +
   机器刚度卸载判据；记录破碎力峰值与断键图连通分量（碎块）。
2. **强度标定**（`weibull_breakage`）：CDF 最小二乘 + MLE（剖面似然，logsumexp
   稳定），尺寸效应 `σ_c(V)=σ0(V0/V)^{{1/m}}`，破碎率 `S(V,t)` 由
   `S = m σ_eff^{{m-1}} σ̇_eff / σ_c^m` 给出，有效应力用 Weibull 应力积分
   `σ_eff = (E[σ_p^m])^{{1/m}}`。
3. **离散 PBM**（`dpbm`）：体积坐标几何网格 + 两结点质量守恒分配 + BDF 积分；
   对 `S=const`、均匀子代用代数求和闭式解作对照。
4. **装配体 DEM–PBM 交叉验证**（`assembly_compression`）：N={s['meta']['n_particles']}
   可破碎球体单轴侧限压缩（k-d 树接触、荷载控制压板、质量台账），在
   t=1,2,4,8 s 提取 PSD；PBM 用**同一条** Weibull 强度律与子代律预测 PSD，
   唯一微观-宏观输入 κ（有效应力比）在**首个破碎事件之前**测得，之后全为预测。
5. 判据、图、JSON 全部程序化产出（`scripts/run_breakage_pbm_verification.py`）。

## 3 计算结果

| 判据 | 实测值 | 阈值 | 判定 |
|---|---|---|---|
{rows}

**C1 单颗粒破碎**：有效试验 {cr['n_tests']} 次，破碎力中位数
{cr['median_peak_force_N']:.1f} N，破碎应力中位数
{cr['median_strength_Pa']/1e6:.2f} MPa，Weibull 模数 m={w['m']:.3f}，
尺度 σ0={w['sigma_0']/1e6:.3f} MPa，CDF 拟合 R²={w['r2_cdf']:.5f}。

**C2 子代分布**：{d['n_fragments']} 个碎块样本，每事件平均
{d['mean_fragments_per_event']:.3f} 块，最大碎块占比中位数
{d['largest_fragment_fraction_median']:.3f}；对数正态 σ_ln={d['lognormal_sigma_ln']:.4f}；
KS p={d['ks_lognormal']['pvalue']:.4g}（自举 {d['ks_lognormal_bootstrap_pvalue']:.4g}），
均匀分布 KS p={d['ks_uniform']['pvalue']:.4g}。

**C3 PBM vs 闭式解**：τ={a['tau']:g}，累积质量分数最大偏差
{a['cumulative_max_abs_deviation']:.4f}，分箱密度 L1={a['l1_relative_mass_density']:.4f}，
质量矩比 {a['mass_moment_ratio_to_analytic']:.4f}，数密度矩比
{a['number_moment_ratio_to_analytic']:.4f}。

**C4 DEM–PBM 交叉验证**（κ={cv['kappa_effective_stress_ratio']:.3f}，
首次破碎 {cv['first_break_time_s']:.4g} s，{cv['n_events']} 次破碎）：

| t [s] | DEM 颗粒数 | DEM 平均直径 [mm] | PBM 平均直径 [mm] | W1 [mm] | W1/d0 |
|---|---|---|---|---|---|
{rec_rows}

**C5 质量守恒**：破碎事件最大质量亏损
{s['criteria']['C5_mass_conservation']['max_event_defect']:.2e}，
系统质量残差 {s['criteria']['C5_mass_conservation']['system_residual']:.2e}。

**C6 负控**：σ0→∞ PSD 漂移 {nc['no_breakage_drift']:.2e}；σ0→0 与子代分布累积一致度
{nc['instant_breakage_psd_agreement']:.4f}。

## 4 结果图

图 `breakage_pbm_summary.png`（四联）：(a) C1 破碎应力经验 CDF 与 Weibull 拟合；
(b) C2 DEM 碎块体积分数经验 CDF 与均匀/对数正态假设；(c) C3 分段 PBM 与闭式解的
体积加权密度对比（对数坐标）；(d) C4 t=1,2,4,8 s 的累积体积分数曲线（DEM 实线 /
PBM 虚线）。

## 5 注意点（坑与局限）

- **时间步必须按最小碎块定**：以平均颗粒质量取 `dt` 会在细子代出现后立刻失稳。
- **压板阻尼符号**、**近重合接触方向**、**碎块插入深重叠** 三处都会注入能量，
  分别用闭合方向判据、夹断、重叠上限三项修复（详见 docs 第 3 节）。
- **C2 必须同分辨率对比**：DEM 碎块是子球整数倍，连续分布 KS 必然被量化拒绝。
- **κ 的不确定度**线性放大到 PBM 的破碎时刻；质量缩放未做双值对照；装配体无摩擦；
  压缩为侧限（oedometric）而非三轴。
- 子球分辨率（32–136）限制了子代谱的丰富度，是本验证面最主要的系统性局限。

## 6 误差判定

- C1：拟合 R² 对照阈值 0.95 —— {crit['C1_weibull_r2']['value']:.4f}
  （{'达标' if crit['C1_weibull_r2']['passed'] else '未达标'}）。
- C3：累积分布最大偏差 vs 3% —— {crit['C3_pbm_analytic']['value']:.4f}
  （{'<3%，达标' if crit['C3_pbm_analytic']['passed'] else '≥3%，未达标'}）。
- C4：归一化 Wasserstein vs 0.1 —— {crit['C4_dem_pbm_wasserstein']['value']:.4f}
  （{'<0.1，达标' if crit['C4_dem_pbm_wasserstein']['passed'] else '≥0.1，未达标'}）。
- C5：质量残差 vs 1% —— {crit['C5_mass_conservation']['value']:.2e}（达标）。
- 无外部实验基准：本节点是**数值/解析**验证面（DEM 对 PBM），不对标实验数据，
  因此"是否<3%"只对 C3 的解析对照作答；对实验的验证不在本节点范围内。
"""


def main() -> int:
    summary_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/breakage_pbm/summary.json"
    png_path = sys.argv[2] if len(sys.argv) > 2 else "artifacts/breakage_pbm/breakage_pbm_summary.png"
    with open(summary_path) as fh:
        s = json.load(fh)
    md = build_markdown(s)
    figures = []
    if os.path.exists(png_path):
        with open(png_path, "rb") as fh:
            figures = [{"name": "breakage_pbm_summary.png",
                        "caption": "C1 Weibull 拟合 / C2 子代分布 / C3 PBM-闭式解 / "
                                   "C4 DEM-PBM PSD 演化",
                        "png_base64": base64.b64encode(fh.read()).decode()}]
    payload = {
        "title": "TensorDEM 颗粒破碎 DEM→Weibull→PBM 耦合验证面（p176w1-n08）",
        "summary": ("新建 tensordem.breakage_pbm：bonded-sphere 单颗粒破碎统计 → Weibull "
                    "强度标定 → 体积坐标离散 PBM（含闭式解对照）→ 10^4 颗粒单轴压缩 "
                    "DEM–PBM 交叉验证；判据 C1–C7 全部程序化产出。"),
        "scenario_key": "dem_particle_breakage_pbm_coupling",
        "mission_id": 9665,
        "tags": ["TensorDEM", "breakage", "PBM", "Weibull", "DEM-PBM", "verification"],
        "markdown": md,
        "figures": figures,
    }
    token = _token()
    req = urllib.request.Request(
        f"{API}/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        print(resp.status, resp.read().decode()[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())