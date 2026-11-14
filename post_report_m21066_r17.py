import base64, json, os, pathlib, subprocess

ROOT = pathlib.Path("/srv/tensor/agent/work/m21066")
R17 = ROOT / "benchmarks/plane_shear_mu_i/runs/round17"

def b64(p):
    return base64.b64encode(pathlib.Path(p).read_bytes()).decode()

data = json.loads((R17 / "strain_ceiling_r17.json").read_text())
table = data["worst_case_table"]
runs = {r["label"]: r for r in data["runs"]}

rows = "\n".join(
    f"| {t['I_target']:g} | {t['gamma_dot']:.3f} | {t['min_shear_steps_floor']} | "
    f"{t['shear_steps']} | {t['worst_case_strain']:.2f} | {t['worst_case_overrun']:.2f}× |"
    for t in table)

arm_rows = "\n".join(
    f"| {r['label']} | {r['max_recover_strain']} | {r['min_shear_steps']} | "
    f"{r['shear_strain_achieved']:.4f} | {r['strain_overrun_ratio']:.2f}× | "
    f"{r['strain_budget_status']} | {r['recover_termination']} | {r['servo_valid']} |"
    for r in data["runs"])

md = f"""# CASE-DEM-PLANE-SHEAR-MUIDI Round 17 — 把「应变请求」变成上限

**锚点** `CASE-DEM-PLANE-SHEAR-MUIDI` · 任务 m21066 · 仓 TensorDEM
**ref** da Cruz et al. 2005 PRE 72:021309；GDR MiDi 2004 EPJ E 14:341
**验收** μ(I)、φ(I) 在 I = 1e-3…0.3 内误差 < 10%

## ① 案例介绍

干燥颗粒床在恒定正应力 P 下的平面剪切，是 μ(I) 局部流变律的标准装置：

`μ(I) = τ/P`，`I = γ̇ d / √(P/ρ)`，`μ = μ_s + (μ_2−μ_s)/(I_0/I + 1)`，φ 随 I 增大而下降（剪胀）。

装置 `src/tensordem/mu_i_rheology.py::ServoShearCase`：x 周期、y/z 墙、伺服恒正应力；
r = 0.005 m，d = 0.01 m，ρ = 2500 kg/m³，P = 62.5 kPa，μ_c = 0.5，μ_roll = 0.05，
e = 0.3，dt = 3.2149e-5 s，seed 5279。

第 16 轮把锚点判为 FAIL 并列出待办，其中第 2 条即本轮对象：**「应变请求」在装置里是下限
而不是上限**——恒应力恢复门可以把已剪应变放大到请求值的 49.5 倍，而运行仍报告
`servo_valid: True`。这是正确性缺陷，不是精度问题：高 I 点被静默标错，事后无论怎样
重新分析归档都无法修复，必须改装置。

## ② 怎么做的

1. **装置修复**（`mu_i_rheology.py`）：新增 `max_recover_strain`——恢复阶段的**应变**上限
   （不是步数上限，因为步数下限随 1/γ̇ 即随 I 变化）；`None` 保持历史行为**逐位可复现**。
   恢复门记录停止原因 `recover_termination`。新增 `strain_budget_status()` 作为唯一判定处，
   并把 `shear_strain_requested / strain_overrun_ratio / strain_budget_respected /
   strain_budget_status / recover_termination` 写入**每一次**运行结果——任何点都不再可能与
   其预算状态分离地被引用。
2. **驱动修复**（`phi_estimator_audit.py`，即产出第 15 轮归档的脚本）：`min_shear_steps=0`
   （旧的 `ceil(strain/(γ̇dt))+100` 把请求又变成下限，在 I=0.3 时 `+100` 步本身就是 1.93 倍）；
   恢复配额取分类容差的**一半**（`MAX_RECOVER_STRAIN_RATIO = 0.025`），另一半留给门的单步
   离散超调；采样步长按门的**最坏**步数定，修复 I=0.3 归档单步一采样（4.7 MB）的连带问题。
3. **证据**：`run_strain_ceiling_r17.py` 在**完全相同的配置**下跑三臂并存应变轨迹原文；
   `figures_strain_ceiling_r17.py` 从该 JSON 作图，零手抄。
4. **测试**：`tests/test_strain_budget_ceiling.py`，15 项断言。

## ③ 计算结果

### 3.1 缺陷由装置自身算术重现（六点标称 I，第 16 轮参数）

| I | γ̇ [1/s] | 驱动步数下限 | `shear_steps` | 最坏应变 | 最坏超剪 |
|---|---|---|---|---|---|
{rows}

六点**逐位重现**第 16 轮表 1 的「executed / strain achieved」列（124522、41574、12543、
4248、1345、515；I=0.3 处 98.93 = 49.46×），且不依赖归档工件——缺陷是算术性的。
超剪随 I **单调增长**：步数下限 ∝ 1/γ̇，而恢复配额是固定步数。

### 3.2 修复在出厂装置上的实测（I = 0.3，请求应变 0.5，同种子同几何）

| 臂 | `max_recover_strain` | `min_shear_steps` | 实际应变 | 超剪 | 判定 | 停止原因 | `servo_valid` |
|---|---|---|---|---|---|---|---|
{arm_rows}

- 第 1 行即缺陷本身：剪到请求值的 33 倍，仍报 `servo_valid: True`，且无一字提示。
- 第 2 行隔离出第二个侧面：上限约束住了**恢复**超剪，但 `min_shear_steps=200` 的**下限本身**
  仍把 0.5 的请求交付成 0.964——所以驱动改动是必需的，而标志诚实地仍判 `violated`。
- 第 3 行即第 17 轮协议：交付应变仅高出请求 **3.2%**，落在 5% 带内，并且**主动声明**
  该点不是恒应力测量（`servo_valid: False`），而不是一直剪到伺服恢复为止。这是有意的取舍：
  诚实的「此点非恒应力」优于静默的 49 倍超剪。

交付应变相对请求的误差：**3325% → 3.2%**。

## ④ 结果图

1. **fig_strain_budget_defect.png**：(a) 由装置算术重推的最坏超剪 vs I（对数坐标），每点标注
   步数下限，并画出 1.05× 验收带；第 16 轮实测的 49.5× 恰好落在 I=0.3 的解析上界上。
   (b) 三臂的累积应变 vs 剪切步：历史臂冲出画面（16.62），带上限的两臂停住。
2. **fig_ceiling_verdict.png**：各臂「交付/请求」应变与 5% 带，标注每次运行现在都会自报的
   `strain_budget_status`：33.25× / 1.93× / 1.03×。

## ⑤ 注意点

- **带上限的运行不等于好运行**：第 2 行说明只加上限仍留 1.93 倍超剪（`min_shear_steps` 也是下限）。
  标志把它抓住了——这正是标志在起作用，也是本修复必须双侧的原因。
- **上限会超调一步**：门在 `strain += γ̇dt` 之后判定，最多超出一步应变（I=0.3 时 4.82e-3，
  约 0.5 请求的 0.96%）；`MAX_RECOVER_STRAIN_RATIO = 0.025` 已为此预留。
- **`max_recover_strain=None` 是刻意的**：把上限设为默认会静默改变所有归档运行的含义。
  默认保持逐位可复现（归档点改前改后同为 `16.622514640912446`、3447 步）；由**驱动**选择启用。
- **仅转换了 `phi_estimator_audit.py`**：其余五个驱动仍只传 `min_shear_steps`，它们现在会**继承标志**
  （静默无效点无法逃逸），但**行为**仍无界，直到套用同样两行改动——这是本轮诚实的范围边界。
- **本轮不主张精度**：μ(I)/φ(I) 未向参考值靠近；φ 约 −13% 的偏差与不可引用的高 I 的 μ 原样未动，
  参考量级仍为 PROVENANCE-UNVERIFIED。
- **未做规模化重跑**：定点扫描（第 16 轮第 3 条）与预压实制备试验（第 4 条）本轮未跑。

## ⑥ 误差判定（如实）

| 量 | 判据 | 实测 | 结论 |
|---|---|---|---|
| 交付应变 vs 请求（历史路径） | ≤ 1.05× | 33.25× | **FAIL**（缺陷被忠实重现） |
| 交付应变 vs 请求（第 17 轮协议） | ≤ 1.05× | **1.03×**（+3.2%） | **PASS** |
| 最坏超剪解析 vs 第 16 轮归档 | 逐位一致 | 6/6 点一致（…、98.93、49.46×） | **PASS** |
| `None` 路径逐位保持 | 完全一致 | 两侧同为 16.622514640912446 / 3447 步 | **PASS** |
| 新测试模块 | 全绿 | 15 passed | **PASS** |
| 更广的 μ(I)/伺服回归套件 | 全绿 | **未跑**（写稿时集群作业 cj3163 仍在排队） | **WITHHELD** |
| μ(I)、φ(I) vs 参考（< 10%） | < 10% | 与第 16 轮相同（φ −12.3…−13.9%；μ 不可引用） | **FAIL**（锚点未变） |

**关键量是否 < 3%？** 本轮不提供任何物理精度量，故 3% 之问不适用于物理；锚点自身判据仍为 FAIL。
本轮唯一有外部判据的量是交付应变误差，为 **3.2% 对 5% 带**：通过本轮定义的带，
但若按 3% 的尺子则**勉强不合格**——特意这样写，而不把 3.2% 抹成 3%。

**台账结论：锚点仍 FAIL，且这是正确的。** 本轮移除了一个被点名的阻塞项（无法交付自己所声明应变的协议），
换成一台无法产生静默超剪点的装置；剩余阻塞项全部是**测量**项，原样未变。
"""

payload = {
    "title": "CASE-DEM-PLANE-SHEAR-MUIDI R17：约束应变上限，消除静默 33× 超剪",
    "summary": "平面剪切 μ(I) 算例的协议缺陷修复：应变请求原为下限而非上限，恢复门可把已剪应变放大到 49.5 倍而仍报 servo_valid。装置新增应变上限与逐次运行的预算状态标志，驱动改为纯应变预算。同配置实测超剪 33.25× → 1.03×，None 路径逐位不变。",
    "scenario_key": "CASE-DEM-PLANE-SHEAR-MUIDI",
    "mission_id": 21066,
    "tags": ["TensorDEM", "DEM", "granular-rheology", "mu(I)", "plane-shear",
             "protocol-defect", "reproducibility", "round17"],
    "markdown": md,
    "figures": [
        {"name": "fig_strain_budget_defect.png",
         "caption": "(a) 由装置算术重推的最坏超剪 vs I（标注步数下限、1.05× 带；第 16 轮实测 49.5× 落在解析上界）；(b) 三臂累积应变轨迹。",
         "png_base64": b64(R17 / "fig_strain_budget_defect.png")},
        {"name": "fig_ceiling_verdict.png",
         "caption": "各臂交付/请求应变与 5% 带：33.25× / 1.93× / 1.03×，并标注运行自报的预算状态。",
         "png_base64": b64(R17 / "fig_ceiling_verdict.png")},
    ],
}
pathlib.Path("/tmp/report_m21066.json").write_text(json.dumps(payload))
print("payload bytes:", os.path.getsize("/tmp/report_m21066.json"))