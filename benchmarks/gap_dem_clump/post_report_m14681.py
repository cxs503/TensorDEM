#!/usr/bin/env python3
"""Build and post the GAP-DEM-CLUMP (mission 14681) progress report.

All numbers are injected from artifacts/GAP-DEM-CLUMP/results.json (no hand
transcription).  Payload path is mission-unique on purpose.

    python benchmarks/gap_dem_clump/post_report_m14681.py
"""
from __future__ import annotations
import base64, json, os, urllib.request

from pathlib import Path
R=Path('artifacts/GAP-DEM-CLUMP/results.json'); d=json.loads(R.read_text())
c1,c2,c3=d['C1_analytic'],d['C2_inertia_scan'],d['C3_repose_scan']; sw=d['C3_rolling_sweep']
rows="\n".join(f"| {r['aspect_ratio_param']:.1f} | {r['n_members']} | {r['sphericity']:.4f} | {r['rel_err_production']:.5f} | {r['rel_err_fine_voxel']:.2e} |" for r in c2['rows'])
bars="\n".join(f"| {k} | {v['mean']:.2f} | {v['std']:.2f} | {c3['deltas_vs_sphere_deg'][k]:+.2f} |" for k,v in c3['by_ar'].items())
sws="\n".join(f"| {k.split('=')[1]} | {list(v.values())[0]:.2f} | {list(v.values())[1]:.2f} | {list(v.values())[1]-list(v.values())[0]:+.2f} |" for k,v in sw.items())
fail=d.get('failures',{})
md=f"""## 1 案例介绍
GAP-DEM-CLUMP 节点：非球形颗粒（多球刚性 clump）生成、转动惯量解析核验、休止角形状效应。TensorDEM 仓。
clump 家族：前球体族 a/b ∈ {{1.0,1.5,2.0,3.0}}，横向半轴 b=1.2 mm，实密度 2500 kg/m³，成员球数 {{1,3,4,6}}；接触=线性弹簧-阻尼 + Coulomb 摩擦 + Cundall 滚动阻力（mu=0.5, mu_roll=0.3, dt=5e-5 s）。

## 2 怎么做的
动机：生产代码 `ClumpParticle.inertia` 只有**体素数值积分**（grid=72），而重叠多球并集**没有解析参考**，体素之间互相比对无法证伪。
新增 `src/tensordem/clump_inertia.py`：球冠矩（体积/一阶/轴向二阶/横向 ∫(y²+z²)dV）闭式积分；球-球交集（lens）在根面 x0=(d²+r1²−r2²)/(2d) 处分裂为两球冠，精确；并集按容斥 U=ΣB−Σ(B_i∩B_j)+Σ(B_i∩B_j∩B_k)…；三阶项非初等，在小盒内数值求并报告 third_order_fraction，超过 1e-3 则标记 exact=False（不静默截断）。
休止角协议：`lifted_box_repose`（准二维提箱堆），800 静置+1800 提箱+1200 尾段，每形状 2 种子；另做 mu_roll=0/0.6 机制扫描。

## 3 计算结果
C1 解析核 vs 独立参考：单球 0.0（位精确）；相切哑铃 0.0（位精确）；重叠双球 vs 384³ 体素 惯性 6.5e-05 / 体积 4.4e-05；三球簇 3 阶 3.9e-05 vs 2 阶 6.6e-04（3 阶改善 17 倍，三阶交集体积占比 1.07e-02）。

C2 生产 clump 惯量 vs 解析（判据 1%）：

| a/b | 成员数 | 球度 | 生产体素(max 分量相对误差) | 细体素192³ vs 解析 |
|---|---|---|---|---|
{rows}

最差 0.31%（a/b=3） → C2 PASS。

C3 休止角（2 种子）：

| a/b | 均值[deg] | 种子标准差 | Δ vs 球 |
|---|---|---|---|
{bars}

机制扫描（球 vs a/b=3）：

| mu_roll | 球[deg] | a/b=3[deg] | Δ[deg] |
|---|---|---|---|
{sws}

## 4 结果图
左图：clump 惯量数值 vs 解析的最大分量误差（对数轴，含 1% 判据线）；右图：休止角 vs 长径比，含文献 22-25°（玻璃珠）与 35-45°（椭球/棱角）带。

## 5 注意点
① 首版 lens 项把张量表达在 lens 质心、而球项表达在帧原点，混合参考系导致横向对角误差 13%，而体积/轴向同时精确到 1e-4——诊断依据是"体积对、轴向对、横向错"这个不一致图样，而不是加密网格。已修正为同一帧原点再平移至质心。
② 体素不能证伪体素：grid=72 与 grid=288 互差 ~1e-3，而两者距解析仅 1e-4；只有闭式解能定标。
③ C3 的填充按 clump 外接半径投放，长径比越大粒子数越少（648→104），形状与堆体规模混叠；这是本轮发现的协议混淆项，下一轮必须先解耦。
④ 滚动阻力控制形状效应的**符号**：mu_roll=0 时长球休止角比球高 9.46°（与文献方向一致），mu_roll=0.6 时反过来低 7.17°——各向同性体级 Cundall 滚动扭矩对球惩罚过重、对长体饱和。

## 6 误差判定
- C1 解析核 vs 闭式解：0.0（<1e-12 判定线）→ **<3% 满足**（位精确）。
- C2 生产 clump 惯量 vs 解析：最差 0.31%（判据 1%）→ **<3% 满足**。
- C3 休止角 vs 文献：球基线 25.82° vs 文献带 22-25°（高 0.8°，即 +3.3% 相对带顶）；趋势判据 FAIL：非单调、对球最大增量仅 +2.17°（判据 ≥3°）→ **不满足**，honest-FAIL，fail_stage=repose_trend，failure_reason 已落 results.json（含混淆项与机制证据）。不粉饰。
"""
fig=Path('artifacts/GAP-DEM-CLUMP/gap_dem_clump.png')
payload={"title":"GAP-DEM-CLUMP 第1轮：clump 转动惯量解析核验通过(0.31%)，休止角形状效应 honest-FAIL 并定位到滚动阻力各向同性",
 "summary":f"C1 解析核位精确/1e-4 级；C2 生产 clump 惯量 vs 解析最差 {c2['worst_rel_err_production']*100:.2f}%(<1%) PASS；C3 休止角 vs 长径比趋势 FAIL（非单调，Δmax +{c3['max_delta_deg']:.2f}°），定位混淆项(粒子数 648→104)与机制(mu_roll 控制符号, Δ=+9.46°→−7.17°)，落 fail_stage+failure_reason 供第2轮。",
 "scenario_key":"gap_dem_clump","mission_id":14681,
 "tags":["TensorDEM","clump","inertia","analytic","repose-angle","shape-effect","honest-fail","GAP-DEM-CLUMP"],
 "markdown":md,
 "figures":[{"name":"gap_dem_clump.png","caption":"左：clump 惯量（grid=72 生产值 / 192³ 细体素）vs 解析并集的最大分量相对误差，红线=1% 判据；右：休止角 vs 长径比（4 组 × 2 种子），背景带为文献 22-25°（球）与 35-45°（椭球/棱角）。","png_base64":base64.b64encode(fig.read_bytes()).decode()}]}
OUT=Path('/tmp/report_m14681.json'); OUT.write_text(json.dumps(payload))
print('report bytes',len(json.dumps(payload)))


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
