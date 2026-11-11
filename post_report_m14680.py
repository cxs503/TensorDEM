import base64
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PW = os.environ["TENSOR_AGENT_PASS"]

ROOT = os.path.dirname(os.path.abspath(__file__))
PNG = os.path.join(ROOT, "artifacts", "gap_dem_autocalib", "fig1_repose_diagnosis.png")
diag = json.load(open(os.path.join(ROOT, "artifacts", "gap_dem_autocalib", "diagnosis.json")))
s = diag["summary"]
p1 = diag["p1_friction_response"]
rep = [round(r["theta_r_deg"], 2) for r in diag["p2_replica_noise"]]
p3 = diag["p3_rest_criterion"]["rows"]


def req(path, body=None, token=None):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(BASE + path, data=data, method="POST" if body else "GET")
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(r, timeout=180) as fh:
        return json.loads(fh.read().decode())


tok = req("/auth/login", {"username": USER, "password": PW})
token = tok.get("access_token") or tok.get("token") or tok.get("data", {}).get("access_token")

table = "\n".join(f"| {r['mu']:.2f} | {r['theta_r_deg']:.2f} | {r['ke_ratio']:.3f} | {r['r2']:.2f} |"
                  for r in p1)
p3table = "\n".join(
    f"| {r['mu']:.2f} | {r['fixed_window']['theta_r_deg']:.2f} | "
    f"{r['fixed_window']['ke_ratio']:.1e} | {r['ke_stop_window']['theta_r_deg']:.2f} | "
    f"{r['ke_stop_window']['ke_ratio']:.1e} |" for r in p3)

md = f"""# GAP-DEM-AUTOCALIB round 1 — 前向模型可标定性诊断（honest FAIL + 根因量化）

**任务 m14680 / TensorDEM。判据「标定后预测角误差<10%」本轮未达标，如实 FAIL，并给出可机器复核的根因与第2轮计划。**

## ①、案例介绍
微观参数（刚度/摩擦/阻尼）→宏观响应（休止角/弹性模量）自动标定闭环。前向模型 =
`tensordem.dem_calibration.heap`（线性弹簧-阻尼法向 + Cundall–Strack 切向 + 滚动弹簧，
Velocity–Verlet，稀疏 cKDTree 接触对）；材料 = 仓库既有 `sand` 类。

| 参数 | 值 |
|---|---|
| 颗粒半径 r | 3.0 mm |
| 密度 ρ | 2650 kg/m³ |
| 滚动摩擦 μ_r | 0.10 |
| 接触刚度 k_n | 800 N/m（重力缩放软球） |
| 容器半径/装填高度 | 21 mm / 42 mm（=7 个粒径） |
| 颗粒数 | 275 |
| 变量 | μ∈{{0.20,0.35,0.52,0.70,0.85}}，e=0.38、k_n/k_s=2.2 固定 |
| 文献锚 | 干砂休止角 30–37°（Beakawi Al-Hashemi & Baghabra Al-Amoudi 2018 综述：干砂 30–35°） |

## ②、怎么做的
三项预登记的确定性实验（`scripts/m14680_autocalib_diagnosis.py`，所有数字程序化写入
`artifacts/gap_dem_autocalib/diagnosis.json`，零手抄）：
P1 摩擦响应 θ_r(μ)（可标定要求严格单调）；P2 同参数三种子复算（噪声底）；
P3 静止判据：抬升圆筒塌落，比较（a）既有固定 hold+lift+settle 时间窗与（b）新增自适应
终止 `ke_stop`（K/释放势能 < 阈值，即"堆体确已静止"）。

## ③、计算结果
**P1 非单调**：

| μ | θ_r [°] | ke_ratio | r² |
|---|---|---|---|
{table}

Spearman(μ,θ_r)=**{s['spearman_mu_theta']:.2f}**，4 个区间中 3 个非递增；模拟可达带宽
**{s['theta_simulated_band_deg'][0]:.2f}–{s['theta_simulated_band_deg'][1]:.2f}°** vs 文献
**{s['literature_theta_band_deg'][0]:.0f}–{s['literature_theta_band_deg'][1]:.0f}°** → `literature_reachable=false`。

**P2 噪声底 ≈ 信号**：同参数三种子 θ_r = {', '.join(f'{x:.2f}' for x in rep)}°，跨度
**{s['replica_noise_span_deg']:.2f}°**；μ 扫描全程仅 **{s['signal_span_deg']:.2f}°** → SNR **{s['signal_to_noise']:.2f}**。

**P3 固定窗口未静止 / 静止后不是堆**：

| μ | 固定窗口 θ_r | 固定窗口 ke/PE | ke_stop θ_r | ke_stop ke/PE |
|---|---|---|---|---|
{p3table}

静止塌落体的体积分数仅 1–3%（单层铺展），275 颗粒 @ 7 粒径柱高不足以形成稳定坡面。

## ④、结果图
左：θ_r(μ) 与文献带/复算噪声带——前向映射非单调、且整段低于文献带；右：两种终止规则下的
残余动能比与静止后 θ_r（0.8–4.5°）。图由 `scripts/m14680_autocalib_figure.py` 从工件 JSON 生成。

## ⑤、注意点（坑与局限）
- 既有 pluviation 堆在"测量时刻"残余动能占释放势能 **25–33%**，即此前所有 θ_r 都不是静止堆的角；
- 抬升圆筒协议到静止后塌成单层（φ_solid 1–3%），圆锥松弛协议完全不松弛（θ=初始 50.9°）；
- 因此本轮**没有**把优化器指向文献值：在可达带宽整体低于文献带时，任何"标定成功"都只是搜索盒的产物；
- 新增能力 `run_heap(..., ke_stop=...)` 默认 None 时轨迹与 SHA-256 摘要逐位不变（已测）。

## ⑥、误差判定（逐项，显式回答是否<3%）
| 关键量 | 基准 | 得到 | 误差 | <3%? |
|---|---|---|---|---|
| 砂休止角最佳可达 | 30–37°（文献） | 18.45° | 38–50% 偏低 | **否** |
| θ_r(μ) 单调性 | 0 次反转 | 3 次反转 | — | **否** |
| θ_r(μ) 信噪比 | >3（稳定反演下限） | 1.70 | — | **否** |
| `ke_stop` 终止语义（代码面） | 判据满足 ⇔ settled=True | 5/5 测试通过 | 无代码缺陷 | 是（仅单元层） |

**判据「标定后预测角误差<10%」本轮未达成（honest FAIL）**，失败并非优化器问题，而是前向模型
资格未过：不可达 + 非单调 + 噪声占优 + 测量时未静止。

## fail_stage / failure_reason（第2轮调研单输入）
- **fail_stage**：`forward_model_qualification`（优化器阶段之前）
- **failure_reason**：θ_r 9.92–18.45° vs 文献 30–37°，Spearman=0.50（3/4 区间非递增），
  复算噪声 5.00° vs 信号 8.53°（SNR 1.70），pluviation 测量时刻 ke/PE=0.25–0.33，
  抬升圆筒静止后塌为单层（φ=1–3%，θ=0.8–4.5°）。
- **已试路径**：5 点 μ 扫描；3 种子复算；抬升圆筒（固定窗 vs ke_stop）；圆锥松弛协议。
- **缺什么**：≥5000 颗粒的尺度收敛门（θ_r(N) 平台，单次前向 ~10² s → 走 run_cluster 分片）；
  Hertz–Mindlin + 滚动阻力、接近真实砂刚度的接触律（现 k_n=800 N/m 致 φ≈0.25）；
  第二条宏观响应"弹性模量"的前向模型（现管道只返回 φ_solid，无量纲 E 通路）。

## 本轮交付物
1. `src/tensordem/dem_calibration/heap.py`：`run_heap(..., ke_stop, settle_check_every)` 自适应静止终止 +
   结果新增 `settled/steps_used/ke_stop`（向后逐位兼容）；
2. `tests/test_dem_repose_settle_termination.py`（5 测试，15.1 s，全绿）；
3. `scripts/m14680_autocalib_diagnosis.py`（≈115 s，确定性，出 JSON 工件）、`scripts/m14680_autocalib_figure.py`；
4. `artifacts/gap_dem_autocalib/diagnosis.json` + `fig1_repose_diagnosis.png`；
5. `docs/GAP-DEM-AUTOCALIB_round1.md`（含第2轮四步计划与各自验收门）。

复现：`PYTHONPATH=src python -m pytest tests/test_dem_repose_settle_termination.py -q`；
`PYTHONPATH=src python scripts/m14680_autocalib_diagnosis.py`。
"""

fig = base64.b64encode(open(PNG, "rb").read()).decode()
body = {
    "title": "GAP-DEM-AUTOCALIB round 1: 休止角前向模型可标定性诊断（非单调+不可达+噪声占优 → honest FAIL + 根因量化）",
    "summary": "TensorDEM 休止角标定闭环前置资格检查：θ_r(μ) 非单调(Spearman 0.50)、模拟带宽 9.92–18.45° vs 文献 30–37°、复算噪声 5.00° vs 信号 8.53°(SNR 1.70)、固定窗测量时残余动能 25–33%；新增 run_heap ke_stop 自适应静止终止 + 5 项单测。",
    "scenario_key": "TensorDEM",
    "mission_id": 14680,
    "tags": ["GAP-DEM-AUTOCALIB", "TensorDEM", "DEM", "calibration", "angle-of-repose", "honest-FAIL"],
    "markdown": md,
    "figures": [{"name": "fig1_repose_diagnosis.png",
                 "caption": "左：θ_r(μ) 与文献带(30–37°)、复算噪声带；右：固定窗 vs ke_stop 终止的残余动能比与静止后 θ_r。",
                 "png_base64": fig}],
}
out = req("/tensor/reports", body, token)
print("REPORT:", json.dumps(out)[:300])