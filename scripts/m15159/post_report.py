"""Post the m15159 GAP-DEM-FRICTION progress report to the platform.

Every number in the markdown is injected from ``artifacts/gap_dem_friction_m15159``
(no hand transcription), and the figure is the committed PNG, base64-inlined.
Credentials come from the environment (never from the repo).
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ART = os.path.join(REPO, "artifacts", "gap_dem_friction_m15159")
MISSION = 15159


def _login(base, user, password):
    req = urllib.request.Request(
        base + "/auth/login", method="POST",
        data=json.dumps({"username": user, "password": password}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    return d.get("access_token") or d.get("token")


def _post(base, token, path, payload):
    req = urllib.request.Request(
        base + path, method="POST", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read()[:400].decode("utf-8", "replace")


def main() -> int:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    token = _login(base, os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"])

    with open(os.path.join(ART, "benchmark.json")) as fh:
        bm = json.load(fh)
    head = bm["headline"]
    env = bm["envelope"]
    rows = bm["rows"]

    pts = []
    for name in sorted(os.listdir(ART)):
        if name.endswith(".json") and name != "benchmark.json":
            with open(os.path.join(ART, name)) as fh:
                pts.append(json.load(fh))

    level_rows = "\n".join(
        "| %s | %.2f | %.2f | %.0f | %.1f | %+.2f%% | %.1f | %.4f | %.3f | %+.3f | %d | %s | %s |"
        % (p["tag"], p["case"]["mu_p"], p["case"]["gamma"], p["level"]["sigma_target_pa"],
           p["level"]["sigma_pa"], 100 * p["level"]["sigma_rel_err"], p["level"]["tau_pa"],
           p["level"]["mu_macro"], p["level"]["mu_over_mu_p"], p["level"]["drift_rel"],
           p["level"]["total_steps"], "yes" if p["level"]["usable"] else "no",
           "yes" if p["level"]["degenerate"] else "no")
        for p in pts)

    env_txt = ("not fitted (n_levels=%d)" % env.get("n_levels", 0)
               if env.get("status") != "ok" else
               "tan(phi)=%.4f (phi=%.2f deg), c=%.2f Pa, R2=%.4f"
               % (env["tan_phi"], env["phi_deg"], env["cohesion_pa"], env["r_squared"]))

    verdict = ("PASS" if head["pass"] else "FAIL")
    md = f"""## 1. 案例介绍

直剪盒是颗粒材料**宏观摩擦系数**的测量装置（GAP-DEM-FRICTION 锚点）：粗糙底板 + 无摩擦侧壁 +
应力伺服压板拖动。本单沿用 m14721/m14840/m14960 的装置，回答 m14960 交叉审计（L1 判级）留下的
阻断项：**headline 数字取自本仓自身判据判为非稳态的窗口**。

材料/几何：单分散玻璃球 r = 5 mm、ρ = 2500 kg/m³，盒 0.080 × 0.064 × 0.080 m（{pts[0]['bed']['n_particles']} 颗，
{pts[0]['bed']['steps']} 步床面制备，体堆积率 {pts[0]['bed']['packing_fraction']:.3f}）；
接触 kn = 1e5 N/m、kt = 0.5kn、e = 0.3、dt = {pts[0]['case']['timestep_s']:.3e} s。

## 2. 怎么做的

1. **稳态门开在 headline 上**：每级都带 `stationarity_tol = 0.15`（m14840 腿）出数，剪切应变推到 γ = 1.0（m14960 只到 0.6 时 drift = +0.24，被同一判据拒），
   另跑 γ = 1.5 作为「剪得更久是否更稳」的对照。
2. **退化滑移的机器判据**（审计 §7 第 5 条）：`mu/mu_p ∈ [0.98, 1.02]` ⇒ 压板在第一层上滑动（界面滑移），
   不是体相摩擦，`degenerate=True` 拒用（`scripts/m15159/study.py:classify`）。
3. **机读工件**：每点一份紧凑 JSON（级别汇总 + 201 点降采样 τ(γ) 历史 + repro_cmd），
   审计可只凭提交文件复核门是否成立（审计 §7 第 2 条），表格与图由
   `scripts/m15159/run_gap_dem_friction.py --aggregate` 程序化生成（零手抄，审计 §7 第 3 条）。
4. 并行：6 个点本地 128 核各一进程（同 seed 6107，装置确定性）。

## 3. 计算结果（程序化注入）

| tag | μ_p | γ | σ_n 目标 [Pa] | σ_n 实测 [Pa] | 相对误差 | τ [Pa] | μ_macro | μ/μ_p | drift_rel | 步数 | 可用 | 退化 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{level_rows}

Mohr-Coulomb 包络（μ_p = 0.5 的稳态级）：{env_txt}

## 4. 结果图

见附图：左=τ(γ) 历史（含锚线 0.38σ_n），中=μ_macro vs μ_p 材料轴（带实验锚带 0.36–0.40），
右=稳态 (σ_n, τ) 点与包络拟合。

## 5. 注意点

* 锚（ref = {head['ref']:.2f}，带 {head['ref_band'][0]:.2f}–{head['ref_band'][1]:.2f}）来自文献准静态稠密球床有效摩擦
  （GDR MiDi 2004；Jop et al. 2006），**未**按被测材料标定 —— DEM 的 μ_p 是独立轴而非拟合参数，
  因此材料轴（μ_p = 0.3/0.5/0.7）本身即为「标定→独立预测」协议的标定端。
* 非周期盒：剪切位移累积会把试样推向下游墙，这是 m14960 观察到 τ 单调上升的机制候选；本单只做
  「剪得更久」的判别，未实现 x 周期（Lees–Edwards），后者列为后续。
* 单级 noise：σ_n 级间床面各自独立制备（点独立），级间差含床面噪声，故包络只作一致性检查。

## 6. 误差判定

| 量 | 基准 | 实测 | 误差 | 是否 <3% |
|---|---|---|---|---|
| σ_n 保持（{head['case']['geom']}, μ_p={head['case']['mu_p']:.2f}, 目标 {head['case']['sigma_n_pa']:.0f} Pa） | 目标值 | {pts[0]['level']['sigma_pa']:.2f} Pa | {100*pts[0]['level']['sigma_rel_err']:+.2f}% | {'**是**' if abs(pts[0]['level']['sigma_rel_err'])<0.03 else '否'} |
| μ_macro（headline） vs 文献锚 | {head['ref']:.2f} | {head['got']:.4f} | {100*head['signed_rel_error']:+.2f}% | {'**是**' if head['abs_rel_error']<0.03 else ('否（<10% 验收线：%s）' % ('满足' if head['within_tolerance'] else '不满足'))} |
| 稳态门 drift_rel（headline） | 0.15 | {head['drift_rel']:+.3f} | — | {'**是**' if abs(head['drift_rel'])<0.15 else '否'} |

**判定：{verdict}** —— headline 点 usable={head['usable']}、degenerate={head['degenerate']}、
|err| = {100*head['abs_rel_error']:.2f}% vs 验收判据 10%。
"""

    if "--md-only" in sys.argv:
        out = os.path.join(REPO, "docs", "GAP_DEM_FRICTION_M15159.md")
        with open(out, "w") as fh:
            fh.write(md)
        print("markdown ->", out)
        return 0

    fig_path = os.path.join(REPO, "docs", "figures", "gap_dem_friction_m15159.png")
    figures = []
    if os.path.exists(fig_path):
        with open(fig_path, "rb") as fh:
            figures.append({"name": "gap_dem_friction_m15159.png",
                            "caption": "tau(gamma), material axis mu_macro vs mu_p, Mohr-Coulomb envelope",
                            "png_base64": base64.b64encode(fh.read()).decode()})

    payload = {
        "title": "GAP-DEM-FRICTION 返修: 稳态门下的直剪盒宏观摩擦 + 退化滑移机器判据 (m15159)",
        "summary": ("headline mu_macro=%.4f vs 文献 0.38 (%+.2f%%), drift_rel=%+.3f (门 0.15), "
                    "usable=%s; %d 点机读工件 + benchmark ref/got + repro_cmd"
                    % (head['got'], 100 * head['signed_rel_error'], head['drift_rel'],
                       head['usable'], len(pts))),
        "scenario_key": "dem_direct_shear_friction",
        "mission_id": MISSION,
        "tags": ["TensorDEM", "GAP-DEM-FRICTION", "direct-shear", "friction", "m15159"],
        "markdown": md,
        "figures": figures,
    }
    with open("/tmp/report_m15159.json", "w") as fh:
        json.dump(payload, fh)
    status, body = _post(base, token, "/tensor/reports", payload)
    print(status, body)
    return 0 if status < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())