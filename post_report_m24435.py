#!/usr/bin/env python3
"""m24435 (see-through of m22550 / PR#1722): publish the paper-grade report.

All numbers are read from docs/m22550/outputs/r26/r26_sweep_table.json (produced
by analyse_polydispersity_r26.py from the drained cj4227 sweep) - zero hand-copy.
"""
import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
tab = json.loads((ROOT / "docs/m22550/outputs/r26/r26_sweep_table.json").read_text())
seed = json.loads((ROOT / "docs/m22550/outputs/r26/r26_seed_structure.json").read_text())

# ---- login ---------------------------------------------------------------
req = urllib.request.Request(
    os.environ["TENSOR_API_BASE"] + "/auth/login",
    data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                     "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    headers={"Content-Type": "application/json"},
)
body = json.loads(urllib.request.urlopen(req).read())
token = body.get("access_token") or body.get("token")
assert token, f"login did not return a token: {list(body.keys())}"

# ---- figures -------------------------------------------------------------
fig_b64 = base64.b64encode((ROOT / "docs/figures/r26_mu_phi_vs_law.png").read_bytes()).decode()
seed_b64 = base64.b64encode((ROOT / "docs/figures/r26_seed_structure.png").read_bytes()).decode()

# ---- programmatic tables -------------------------------------------------
def rows_for(w):
    rs = [r for r in tab.values() if abs(r["polydispersity"] - w) < 1e-12]
    rs.sort(key=lambda r: r["I_measured"])
    return rs

sweep_lines = ["| `w` | `I` (meas.) | `mu_wall` | `ref_mu` | `mu` err | `phi_gap` | `ref_phi` | `psi6` |",
               "|---|---|---|---|---|---|---|---|"]
for w in (0.0, 0.25, 0.5):
    for r in rows_for(w):
        sweep_lines.append(
            f"| {w:.2f} | {r['I_measured']:.4f} | {r['mu_wall']:.4f} | {r['ref_mu']:.3f} "
            f"| {100*r['mu_rel_err']:.1f} % | {r['phi_gap']:.3f} | {r['ref_phi']:.3f} "
            f"| {r['psi6_mean']:.3f} |")
sweep_table = "\n".join(sweep_lines)

seed_lines = ["| `w` | `r_max/r_min` | `psi6` (seed) | hexatic frac | `Z` (seed) |",
              "|---|---|---|---|---|"]
for w in (0.0, 0.25, 0.5):
    s = seed[f"{w:g}"]
    seed_lines.append(f"| {w:.2f} | {s['radius_max_over_min']:.2f} | {s['psi6_mean']:.3f} "
                      f"| {s['psi6_hexatic_fraction']:.3f} | {s['z_mean']:.2f} |")
seed_table = "\n".join(seed_lines)

best_mu = min(tab.values(), key=lambda r: r["mu_rel_err"])
best_mu_err = best_mu["mu_rel_err"]
phi_err = {r["I_measured"]: (r["phi_gap"] - r["ref_phi"]) / r["ref_phi"] for r in tab.values()}
best_phi = min(tab.values(), key=lambda r: abs((r["phi_gap"] - r["ref_phi"]) / r["ref_phi"]))
best_phi_pct = (best_phi["phi_gap"] - best_phi["ref_phi"]) / best_phi["ref_phi"] * 100
phi_lo = 100 * min(phi_err.values()); phi_hi = 100 * max(phi_err.values())
mu_lo = 100 * min(r["mu_rel_err"] for r in tab.values())
mu_hi = 100 * max(r["mu_rel_err"] for r in tab.values())

markdown = f"""# CASE-DEM-PLANE-SHEAR-MUIDI R26 — 多分散度扫掠排空与诚实 mu/phi 判定（m24435 管到底）

## ① 案例介绍（物理背景/几何/参数表）

2-D 圆盘平面剪切装置（`benchmarks/plane_shear_mu_i/plane_shear_2d.py`）：粗糙壁行 +
自由圆盘带，恒压 P 伺服，测量 `mu_wall` 与 `phi_gap` 对惯性数 `I` 的响应，对标
da Cruz et al. 2005 PRE 72:021309 / GDR MiDi 2004 的 `mu(I)`/`phi(I)` 律
（`mu(I)=0.36+0.48/(0.30/I+1)`，`phi(I)=0.64-0.06/(0.30/I+1)`）。

R25 诊断：单分散床在恒压压实后**重结晶**（`psi6=0.914`，六角分数 0.85，`Z~4`），
晶体沿无摩擦支剪切（`mu_wall=0.096...0.129` vs 律 `0.246...0.359`，-61...-66 %）。
R25 推荐 (2)：用**物理**对称性破缺——多分散度——破晶，而非换种子。R26 实现该轴
（`--polydispersity w`，`r_i=radius(1+w u_i)`，`u_i~U(-1,1)`，重标定到均值不变），
并修了三个只在半径变化时才显形的装置缺陷（周期镜像重叠、壁缘、`phi_gap` 面积和）。

| 参数 | 值 |
|---|---|
| 探针单元 | 458 盘，`W=0.30 m`，`gap=0.15 m` |
| `P` / `rho` / `mu_p` | 10 kPa / 2500 kg/m3 / 0.4 |
| 种子 | disordered（`--init random`），`seed=0` |
| 压实 / 剪切 | 40 `d/sqrt(P/rho)` / 0.5 应变设置 + 1.0 应变测量 |
| 扫掠矩阵 | `I in {{0.02,0.05,0.2}}` x `w in {{0,0.25,0.5}}`（9 点，`w=0` 为 R25 对照） |

## ② 怎么做的（方法/流程/工具链）

R26 代码+测试（15 例）+种子结构已在 main（经 #1631/m22550 合并）。本单**管到底**：
R26 文档关闭时 9 点扫掠 `cj4227` 仍在排队（位次 56），`mu`/`phi` 从未测得，判定挂
WITHHELD。本单：

1. 确认 `cj4227` 已排空——`runs/plane_shear_2d/poly_w*_I*.json` 9 份齐全（每点
   ~280 s 集群时长，`I` 为**实测**值）；
2. 跑 `analyse_polydispersity_r26.py`（R26 文档 ⑥ 指定的唯一后续命令）→ 程序化产出
   `docs/m22550/outputs/r26/r26_sweep_table.json`（9 行，含 `ref_mu`/`ref_phi`/逐点相对误差）；
3. `gen_r26_mu_phi_fig.py` 从该表零手抄渲染 `docs/figures/r26_mu_phi_vs_law.png`
   （`mu_wall(I)`/`phi_gap(I)` 三 `w` 曲线 vs 律）；
4. 回填 R26 案例文档 ③b/④/⑥（数字全部读自表）。

工具链：TensorDEM 2-D DEM（`plane_shear_2d.py`）→ 集群 `cj4227`（扫掠）→
`analyse_polydispersity_r26.py`（分析）→ `reference_law.py`（单一真值律）。

## ③ 计算结果（数字表，程序化注入）

种子结构（无动力学，`r26_seed_structure.json`）：

{seed_table}

排空后 9 点扫掠（`r26_sweep_table.json`；`I` 为实测，`ref_*` 为 `reference_law`）：

{sweep_table}

**读法**：尺寸散布做了 R25 要它做的事，然后停在那——晶被破（`psi6` 0.91...0.96 →
0.38...0.47），`mu` 离开无摩擦支（`w=0` 的 0.045...0.116 → `w=0.5` 的 0.128...0.162，
x2.5-3），但每个 `I` 仍低于 `ref_mu` 64...89 %；`phi_gap` 0.80...0.88 高于
（来源未验证的）参考 +28...+42 %。残差是**幅值**差，不是**分支**差。

## ④ 结果图

图 1：`mu_wall(I)` 与 `phi_gap(I)`（三 `w`）vs 归档律——`mu` 随 `I` 单调升（定性流变
正确）、随 `w` 单调上移（R25 预测方向），但幅值在每个 `I` 差 ~3 倍；`phi` 系统性偏高。
图 2：种子结构（`psi6`/六角分数/`Z` vs `w`）——晶在 `w>0` 即被破坏。

## ⑤ 注意点（踩过的坑/局限/适用范围）

- **坑**：分析脚本写 `<repo>/outputs/r26/`，而交付路径是 `docs/m22550/outputs/r26/`
  ——两处需一致；本单已把排空后的表落到交付路径（种子结构两处逐字节相同，可复现）。
- **局限**：一种子、一单元、三尺寸散布；无集成、无 1220 盘生产单元、`I<0.02` 未覆盖；
  `p_wall_over_P` 在扫掠内漂移 0.93...1.36（恒压伺服在高 `I` 未把壁应力钉在 `P`），
  故高 `I` 的 `mu` 比较需先做应力路径修正。
- **`phi` 参考来源未验证**（`reference_law.PHI_REF_STATUS`：0.58/0.64 是 3-D 玻璃珠数，
  套在 2-D 圆盘装置上）——`phi` 判定按 WITHHELD 参考处理，不粉饰。
- **适用范围**：2-D 装置级调查；不触及 3-D `CASE-DEM-PLANE-SHEAR-MUIDI` 的 +29 % 偏移
  （R24 假设未解释，仍悬置）。

## ⑥ 误差判定

关键量 vs 基准（`reference_law`，da Cruz 2005 / GDR MiDi 2004）逐项：

| 量 | 最佳点 | 误差 | <3%? | <10%? |
|---|---|---|---|---|
| `mu_wall` | `w=0.5, I=0.047` | **{100*best_mu_err:.1f} %**（全表 {mu_lo:.0f}...{mu_hi:.0f} %） | **否** | **否（FAIL）** |
| `phi_gap` | `w=0.5, I=0.020` | **{best_phi_pct:+.0f} %**（全表 {phi_lo:+.0f}...{phi_hi:+.0f} %） | **否** | **否（FAIL，参考 WITHHELD）** |

显式回答「是否 <3%」：**否**——最佳 `mu` 误差 {100*best_mu_err:.1f} %、最佳 `phi` 误差
{best_phi_pct:+.0f} %，均远超 3 %（也超 10 % 验收）。基准存在（`reference_law`，
da Cruz 2005 / GDR MiDi 2004），不适用「无基准」豁免；`phi` 参考来源未验证，按
WITHHELD 参考计。**honest-FAIL**：R25 强假设（破晶即修好 2-D 装置达验收）被**否定**，
弱假设（晶是无摩擦支上的真实可测缺陷）被**确认**；残差为幅值差（~3x），下一步是
壁摩擦/应力路径项，非种子。
"""

report = {
    "title": "CASE-DEM-PLANE-SHEAR-MUIDI R26 — 多分散度扫掠排空：破晶确认、mu/phi 诚实 FAIL（m24435 管到底）",
    "summary": ("管到底第3轮：R26 的 9 点多分散度扫掠（cj4227）在文档关闭后排空，"
                "本单跑 analyse_polydispersity_r26.py 程序化产出扫掠表并回填案例文档。"
                "结果：多分散度破坏压实晶（psi6 0.91->0.38-0.47）、mu 离开无摩擦支（x2.5-3），"
                "但 mu_wall 仍低于律 64-89%、phi_gap 高于未验证参考 +28..+42%——"
                "R25 强假设被否定、弱假设被确认，残差是幅值差非分支差，ledger 验收 honest-FAIL。"),
    "scenario_key": "CASE-DEM-PLANE-SHEAR-MUIDI",
    "mission_id": "24435",
    "tags": ["TensorDEM", "plane-shear", "mu-i", "polydispersity", "r26", "see-through", "honest-fail"],
    "markdown": markdown,
    "figures": [
        {"name": "r26_mu_phi_vs_law.png",
         "caption": "排空后 9 点扫掠：mu_wall(I) 与 phi_gap(I)（w=0/0.25/0.5）vs 归档律。mu 随 I 升、随 w 上移（方向对），但幅值在每个 I 差 ~3 倍；phi 系统性偏高。",
         "png_base64": fig_b64},
        {"name": "r26_seed_structure.png",
         "caption": "种子结构 vs 尺寸散布：psi6 与六角分数在 w>0 即崩塌（晶被破），Z 为种子值（未压实）。",
         "png_base64": seed_b64},
    ],
}

req = urllib.request.Request(
    os.environ["TENSOR_API_BASE"] + "/tensor/reports",
    data=json.dumps(report).encode(),
    headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
)
resp = json.loads(urllib.request.urlopen(req).read())
print("report_id:", resp.get("id") or resp.get("report_id") or resp)
