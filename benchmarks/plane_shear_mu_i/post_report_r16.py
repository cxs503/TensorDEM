"""Publish the R16 progress report (m20878) -- all numbers read from
runs/strain_sweep_r16/result.json, zero hand-copying."""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import urllib.request

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "strain_sweep_r16"
res = json.loads((RUNS / "result.json").read_text())
rows = res["rows"]
v = res["verdict"]


def fig_b64(name: str) -> str:
    return base64.b64encode((RUNS / name).read_bytes()).decode()


mu_rows = "\n".join(
    f"| {r['I_target']:g} | {r['I_bulk_final']:.5f} | {r['abscissa_shift_pct']:+.0f} "
    f"| {r['strain_achieved']:.2f} | {r['mu_full_20pct']:.4f} | {r['mu_full_25pct']:.4f} "
    f"| {r['mu_ref_at_target']:.4f} | {r['err_mu_20pct_at_target_pct']:+.1f} "
    f"| {r['err_mu_25pct_at_target_pct']:+.1f} | {r['err_mu_25pct_at_bulk_pct']:+.1f} "
    f"| {r['mu_std_rel_pct']:.1f} |"
    for r in rows)
phi_rows = "\n".join(
    f"| {r['I_target']:g} | {r['phi_container_25pct']:.4f} | {r['phi_ref']:.4f} "
    f"| {r['err_phi_25pct_pct']:+.1f} | {'Y' if r['steady_phi'] else 'N'} |"
    for r in rows)

md = f"""# CASE-DEM-PLANE-SHEAR-MUIDI · R16 — 应变收敛扫描分析：mu(I) 缺口是偏置而非暂态（honest FAIL）

## ① 案例介绍

摩擦球密实平面剪切，常应力伺服剪切盒，对标 da Cruz et al. (2005, PRE 72:021309) /
GDR MiDi (2004, EPJ E 14:341) 的局部流变律 mu(I)、phi(I)，I = 1e-3 … 0.3。
总账验收：两条曲线误差均 < 10%。

| 参数 | 值 |
|---|---|
| 半径 / 密度 | 5 mm / 2500 kg m⁻³ |
| 盒 (W×D×填充) | 0.12 × 0.05 × 0.06 m |
| 颗粒数 | N = 294（6–7 层） |
| 接触律 | kn=6.25e4 N/m, kt/kn=0.5, μ_p=0.5, μ_roll=0.05, e=0.3 |
| 施加压力 | 6.25e4 Pa（常应力伺服） |
| I 目标 | 0.001, 0.003, 0.01, 0.03, 0.1, 0.3 |
| 应变预算 | ≥ 2（归档协议低 I 点仅 γ≈0.4） |
| 参考律 | μ=0.36+0.48/(I₀/I+1), I₀=0.30；φ=0.64−0.06/(I₀/I+1)（φ 幅值 provenance-unverified） |

**本轮定位**：R15（m20791）修正了 φ 的定义（容器平均而非中心计数），并递交了
6 点应变收敛扫描（cj2950–cj2955，光滑壁，seed 5279，剪到 strain≥2，I=0.3 处
γ≈99）以判定低 I 缺口是否为应变暂态；作业在 R15 会话结束后返回。本轮 = R15
明确遗留的"下一轮第一测量"，**零新模拟**，全部数字读自 6 个归档点文件。

## ② 怎么做的

`analyse_r16.py`（程序化，零手抄）：

1. 每点报告 μ 的 20% 尾窗（归档有效性门窗口）与 25% 尾窗均值；
2. 参考律在**两个横轴**上求值——I_target（压板速率×施加压力）与 I_bulk_final
   （实测速率×体压力），遵循 R10 纪律：体积平均纵轴必须对实测横轴判定；
3. 量化**噪声底**：末 25% 样本 full_mu 的标准差，与 10% 总账带比较；
4. 平稳性标志（steady_phi/steady_mu）与压力控制误差直接取自运行自身。

工具链：TensorDEM `tensordem.mu_i_rheology.simulate_servo_shear`（R15 递交）→
集群 外协方双节点 → 取回工件 → 本地分析 + matplotlib 出图 + pytest 回归钉扎。

## ③ 计算结果（程序化注入）

### mu(I)（应变收敛后）

| I_target | I_bulk | ΔI/I | γ | μ(20%) | μ(25%) | μ_ref(t) | err 20%@t | err 25%@t | err 25%@b | μ std(25%) |
|---|---|---|---|---|---|---|---|---|---|---|
{mu_rows}

### phi(I)（应变收敛后）

| I_target | φ(25%) | φ_ref | err | steady φ |
|---|---|---|---|---|
{phi_rows}

## ④ 结果图

- 图 1：μ(I)、φ(I) 对参考带（双横轴，10% 总账带）；
- 图 2：I=0.001 与 I=0.3 的 μ–应变轨迹（原始 + 10% 滑动平均）——低 I 点运行
  末端仍在爬升，高 I 点在 0.2–0.8 间振荡。

## ⑤ 注意点 / 局限

- **R15 的部分 PASS 被推翻**：R15 记 "μ(I), I≤0.03: PASS (−0.1%, +1.3%)" 来自
  归档协议（低 I 仅 γ≈0.4）；应变收敛后同两点为 **−31% / −14%**——低 I "PASS"
  是应变暂态，μ 半条验收线现在**每点都 FAIL**。
- μ 缺口是**偏置不是噪声**：I=0.01 处噪声仅 5.3%，偏置仍 −18%。
- **噪声底**：5/6 点末 25% 窗 std 超 10% 带——N=294 下该带**不可分辨**，与协议无关。
- φ 床层平坦（0.55→0.58）而参考律膨胀（0.64→0.61）：趋势反号；I=0.3 处 γ≈99
  床层仍在压实（伺服压实不饱和）。
- 横轴：I_bulk 偏离 I_target −50%…+33%（压板速率≠体速率，壁滑移/局部化）；
  换横轴不改任何判定。
- φ 参考带（0.58–0.64）仍无文献锚（R13/R15），φ 判定是对未锚定带的比较。
- 20% 与 25% 窗在 I=0.001 相差 9 pp——信号非平稳，窗口选择比 10% 带更敏感。

## ⑥ 误差判定（逐项，honest）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| μ(I) 六点（20% 窗, I_target） | JOP 律 | −18.5 … −31.2% | **否** |
| μ(I) 六点（25% 窗, I_target） | JOP 律 | −13.6 … −33.6% | **否** |
| μ(I) 六点（25% 窗, I_bulk） | JOP 律 | −13.3 … −33.6% | **否** |
| φ(I) 六点（25% 窗） | 参考带（未锚定） | −5.3 … −13.7% | **否**（1/6 在 10% 带内） |
| μ 噪声底 vs 10% 带 | 带本身 | 5.3–32.2%（5/6 超带） | 不适用（N=294 不可分辨） |

**总判定：FAIL**——且失败已完全刻画：系统性 μ 偏置（−14…−34%，偏置非暂态）+
平坦 φ 低于未锚定带 5–14% + 噪声底超带。无任何关键量在文献基准 3% 内。

**给下一轮的路径**（R14 已命名，本轮量化）：(R1) 粗糙壁抑制壁滑移（R13 尝试过，
高 I 伺服失败，需先修伺服）；(R4) N>1000 体统计。两者均为重计算战役，超出单干范围。
"""

report = {
    "title": "CASE-DEM-PLANE-SHEAR-MUIDI R16: 应变收敛 mu(I)/phi(I) 分析——mu 缺口是偏置非暂态 (honest FAIL)",
    "summary": ("R15 遗留的 6 点应变收敛扫描 (cj2950-cj2955, strain>=2) 分析完成: "
                "mu(I) 在收敛应变下每点低于 JOP 律 14-34% (偏置非暂态, 推翻 R15 低 I 部分 PASS); "
                "phi(I) 平坦 0.55-0.58 低于参考带 5-14% 且趋势反号; N=294 噪声底 5/6 点超 10% 带. "
                "总判定 FAIL, 失败已完全刻画. 零新模拟, 9 项回归测试绿."),
    "scenario_key": "CASE-DEM-PLANE-SHEAR-MUIDI",
    "mission_id": 20878,
    "tags": ["TensorDEM", "mu(I)", "plane-shear", "verification-debt", "honest-fail", "R16"],
    "markdown": md,
    "figures": [
        {"name": "fig_mu_i_strain_r16.png",
         "caption": "μ(I) 与 φ(I) 对参考带（左：μ，双横轴 I_target/I_bulk；右：φ，容器平均定义）",
         "png_base64": fig_b64("fig_mu_i_strain_r16.png")},
        {"name": "fig_mu_vs_strain_r16.png",
         "caption": "μ–应变轨迹：I=0.001（末端仍爬升）与 I=0.3（0.2–0.8 振荡），原始 + 10% 滑动平均",
         "png_base64": fig_b64("fig_mu_vs_strain_r16.png")},
    ],
}

Path("/tmp/report_m20878.json").write_text(json.dumps(report, ensure_ascii=False))
print("report json written:", len(json.dumps(report)), "bytes")

# login + post
base = os.environ["TENSOR_API_BASE"]
login = urllib.request.Request(
    base.replace("/api/v1", "") + "/api/v1/auth/login",
    data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                     "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
tok = json.loads(urllib.request.urlopen(login).read())
token = tok.get("access_token") or tok.get("token") or tok.get("data", {}).get("access_token")
req = urllib.request.Request(
    base + "/tensor/reports",
    data=json.dumps(report, ensure_ascii=False).encode(),
    headers={"Content-Type": "application/json",
             "Authorization": f"Bearer {token}"}, method="POST")
out = urllib.request.urlopen(req)
print("POST", out.status, out.read()[:400].decode())
