# CASE-DEM-PLANE-SHEAR-MUIDI — Round 4a: premise audit of the constant-pressure confrontation

**锚点**：`CASE-DEM-PLANE-SHEAR-MUIDI`（feature_ledger，status 仍 **open**）
**任务**：m16669（tp-merge-1 generalist 席，自找活）｜**仓库**：TensorDEM｜**分支** `agent/m16669`
**接棒对象**：R4 分析（`docs/CASE_DEM_PLANE_SHEAR_MUIDI_ANALYSIS_R4.md`，m16633）——
其建议 1「用 `phi_bed_steady` 重扫」已在 round 3 工件（`runs/sweep_baseline.json`）里存在，
R4 却只读了 round 1 的陈旧 `sweep.json`，因此把**一个控制前提问题**误判成"薄床/边界"。

## ① 案例介绍

`CASE-DEM-PLANE-SHEAR-MUIDI` 用仓内常压伺服剪切盒
（`tensordem.mu_i_rheology.simulate_servo_shear`）测 μ(I)、φ(I)，
与 da Cruz et al. (2005) PRE 72:021309 / GDR MiDi (2004) 的玻璃珠标定曲线对照，
验收判据为 **μ 与 φ 各点误差 <10%**。

参考律是**常压**律：

```
μ(I)  = μ_s + (μ_2-μ_s)/(I_0/I+1),  μ_s=0.36, μ_2=0.84, I_0=0.30
φ(I)  = φ_c + (φ_j-φ_c)/(I_0/I+1),  φ_c=0.58, φ_j=0.64   （手抄参考，PROVENANCE-UNVERIFIED）
```

装置（不改）：P=62.5 kPa 伺服，k_n=6.25e4 N/m（P/k_n=1，软接触），μ=0.5、μ_r=0.05、
e=0.3、r=5 mm、ρ=2500 kg/m³，盒 12d×5d×6d，N=294，seed=5279，γ̇ 由 I 反解。

**本单问的是一个逻辑上更靠前的题**：这套"常压"对照，压力真的被伺服保住了吗？
请求的应变真的被跑到了吗？——两个前提在 R1–R4 里**从未被判过**，而工件里
其实已经带着 `pressure_control_error` 字段，只是没人读。

## ② 怎么做的（方法 / 关键设置 / 工具链）

1. **零物理改动的最小使能**：`ServoShearCase` 增加 `servo_max_step_d: float = 0.002`
   （默认值与原来两条硬编码 `0.002*d` 完全一致，行为不变），使伺服权限可实验。
2. **前提审计工具** `benchmarks/plane_shear_mu_i/servo_control_diag.py`：
   - `--stage trace`：逐样本序列 + 采样窗平稳性统计（mean/median/CV/线性漂移/低于 0.9P_target 占比）；
   - `--stage authority`：伺服旋钮网格（`servo_relax` × `servo_max_step_d`），只动数值不动接触律；
   - `--stage ensemble`：种子集合，把**偏差 bias** 与**散布 scatter**分开记账；
   - `--stage strain`：应变阶梯（0.2→3.0），显式请求应变时 `min_shear_steps=1`，
     否则请求会被 400 步下限拦住（这本身就是本单查出的第二个缺陷）；
   - `--stage verdict`：对任意 sweep 工件出"前提门"三判（P 跟踪 / μ / φ）。
3. 图与表全部程序化从工件读：`servo_diag_figures.py` → `runs/servo_diag/figures/*.png`、
   `runs/servo_diag/tables.md`（零手抄）。
4. 回归测试 `tests/test_servo_pressure_control.py`（6 项）：采样窗统计在合成序列上的
   稳定性判定（2 项）、伺服保压 <8%（1 项，真跑 7 s）、请求应变授权性（1 项，真跑）、
   `min_shear_steps` 夹紧机制的物理无关断言（1 项）、bias/scatter 分离记账（1 项）。

## ③ 计算结果

### 表 1｜默认伺服权限（round-3 工件）—— 前提 P1 被违反

| I_meas | P 实测 [kPa] | P 误差 | μ | μ 误差 | φ_s | φ 误差 |
|---|---|---|---|---|---|---|
| 0.00101 | 61.64 | −1.38% | 0.3295 | −8.89% | 0.5356 | −7.68% |
| 0.00309 | 58.84 | −5.86% | 0.3289 | −9.86% | 0.5342 | −7.99% |
| 0.01085 | 53.12 | **−15.01%** | 0.3850 | +2.20% | 0.5287 | −9.17% |
| 0.03455 | 47.12 | **−24.61%** | 0.4074 | −0.54% | 0.5241 | −10.59% |
| 0.11790 | 44.94 | **−28.10%** | 0.4302 | −13.16% | 0.5231 | −12.37% |
| 0.32830 | 52.18 | **−16.51%** | 0.4916 | −19.52% | 0.5197 | −14.99% |

6 点里 **4 点保压失败**（目标 62.5 kPa），最差 −28.1%。即：先前各轮"高压端的 μ 差"
是在 **72% 目标压力**下测的，不是参考律所要求的常压状态。

### 表 2｜同一装置、提高伺服权限（只动数值）

| I_meas | P 误差 | P CV | P 窗内漂移 | μ | μ 误差 | φ_s | φ 误差 | 实际应变 |
|---|---|---|---|---|---|---|---|---|
| 0.00100 | +0.05% | 6.63% | −0.01% | 0.3651 | +0.97% | 0.5557 | −4.22% | 0.20 |
| 0.00306 | −3.59% | 4.70% | +4.04% | 0.3000 | −17.77% | 0.5438 | −6.34% | 0.20 |
| 0.01098 | −17.04% | 8.40% | −12.59% | 0.3267 | −13.32% | 0.5386 | −7.47% | 0.20 |
| 0.03188 | −11.47% | 6.17% | −11.21% | 0.3761 | −7.38% | 0.5346 | −8.73% | 0.20 |
| 0.10340 | −6.51% | 11.57% | −28.88% | 0.3841 | −20.49% | 0.5332 | −10.45% | **0.64** |
| 0.31080 | −6.83% | 21.03% | −23.35% | 0.5215 | −13.69% | 0.5254 | −13.94% | **1.91** |

（`servo_relax=0.5`、`servo_max_step_d=0.05 d`。）

### 表 3｜4 种子集合（两个失败点，提高权限后）

| I | n | μ 均值 | μ 标准差 | μ 偏差 | μ 散布 | φ 均值 | φ 偏差 | P 误差均值 |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 4 | 0.3546 | 0.0409 | **−25.74%** | **11.53%** | 0.5316 | −10.60% | −0.93% |
| 0.3 | 4 | 0.5343 | 0.0385 | **−9.75%** | **7.21%** | 0.5246 | −13.86% | +5.41% |

### 表 4｜应变阶梯（实际应变，非请求应变）

I=0.01：μ 误差 −13.3%(ε=0.20) / −2.9%(0.60) / −22.3%(1.50) / −11.3%(3.00)；
φ 误差 −7.5 / −5.9 / −4.3 / −5.1%。
I=0.3：μ 误差 +36.1%(ε=0.20，仅 42 步) / −30.8%(0.58) / −11.5%(1.50) / −10.3%(2.99)；
φ 误差 −12.3 / −12.7 / −13.5 / −13.9%。
（全表见 `benchmarks/plane_shear_mu_i/runs/servo_diag/tables.md`。）

## ④ 结果图

| 图 | 内容 | 路径 |
|---|---|---|
| 图 1 | 逐样本法向应力轨迹（三 I × 默认/提高权限），虚线=62.5 kPa 目标：默认权限在高 I 明显低走且漂移；提高权限后贴目标但仍有窗内漂移 | `runs/servo_diag/figures/fig1_pressure_traces.png` |
| 图 2 | μ(I)、φ(I) 实测 vs 参考律（默认 / 提高权限 / 4 种子集合含误差棒）：提高权限把保压修好，却**没有**把 μ、φ 偏差修好 | `runs/servo_diag/figures/fig2_mu_phi_curves.png` |
| 图 3 | 应变阶梯误差曲线 + ±10% 验收带：无单调收敛 | `runs/servo_diag/figures/fig3_strain_ladder.png` |

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **R4 读错了工件**：`runs/sweep.json` 是 round 1 的陈旧文件（无 `phi_bed_steady`），
   而 round 3 已产出 `runs/sweep_baseline.json`（有该字段，φ_s 随 I 下降而非恒定）。
   由此 R4「φ(I) 曲线根本没在测"扩张"」这一句只在陈旧工件上成立。
2. **`shear_strain` 是下界不是设定值**：`min_shear_steps=400` 在 I≥0.03 时反压请求值，
   配置里写 0.2 而实际跑 0.20 / 0.64 / 1.91（表 2 末列）。工件此前不记录实际应变 →
   跨 I 的"应变一致"假设不成立。已在 `run_sweep.run_point` 补 `strain_achieved` 字段。
3. **提高伺服权限会改变堆积结构**：μ 在纯数值旋钮网格上从 0.34 摆到 0.45（I=0.1），
   说明"数值设置"并非无痕，单次结果不能当作该工况的真值。
4. **采样窗不平稳**：高 I 窗内法向应力漂移 −23%…−29%（CV 21–24%），
   所以 φ_s、μ 的"稳态平均"实际上是暂态平均。本单没能把漂移压到 10% 以内
   （需要更长的物理时间或更好的伺服律，未做）。
5. φ 参考常数（0.58/0.64）在仓内**无出处锚**，仍是手抄 → φ 的误差判定带 PROVENANCE-UNVERIFIED。
6. 局限：N=294、5d 宽、6 层床，系统尺寸未收敛；本单只做前提审计，**没有**做系统尺寸/厚床研究。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 结果 | 相对误差 | <3%？ |
|---|---|---|---|---|
| 伺服保压（低 I，本单提高权限后） | 62.5 kPa | 61.64 kPa（I=0.001） | **+0.05%** | **是** |
| 伺服保压（中 I=0.011，提高权限后） | 62.5 kPa | 53.12 kPa | **−17.0%** | **否** |
| 伺服保压（高 I=0.31，提高权限后） | 62.5 kPa | 58.23 kPa | −6.8% | 否 |
| μ(I=0.001) | 0.3616 | 0.3651 | +0.97% | **是** |
| μ(I=0.1) 4 种子均值 | 0.4954 | 0.3546 ± 0.0409 | **−25.7%** | **否** |
| μ(I=0.3) 4 种子均值 | 0.6108 | 0.5343 ± 0.0385 | **−9.75%** | **否**（但已进 10% 带内） |
| φ(I=0.001) | 0.5802 | 0.5557 | −4.22% | 否 |
| φ(I=0.3) | 0.6114 | 0.5254 | −13.94% | 否 |
| 请求应变 = 实跑应变（默认扫描） | 0.2 | 0.64（I=0.1）/ 1.91（I=0.3） | +220% / +855% | **否**（缺陷） |

**显式回答**：本单**不是**一项 <3% 的物理达标结果。达标（<3%）的只有
"低 I 端伺服保压"和"μ(I=0.001)"两项；μ、φ 在 I≥0.1 仍超 10% 验收带。
本单的实质增量是**把不可信的判定变成可判定**：找出并量化了两个未被判过的前提缺陷
（保压失败 4/6 点、请求应变被夹紧 2/6 点），并用对照实验**否证**了"这些偏差来自伺服控制"
这一假设——提高伺服权限把保压修到 ≤6.8%，μ/φ 偏差却仍在（I=0.1 反而更差）。

## ⑦ 判定（对锚点）

| 检验 | 判据 | 结果 | 判定 |
|---|---|---|---|
| P1 保压前提 | 6 点 \|P/P_target−1\| ≤10% | 默认权限 2/6 通过；提高权限 4/6 通过 | **PARTIAL**（缺陷已修，中 I 仍未解） |
| P2 应变约定 | 实跑应变 = 请求应变 | 0.20/**0.64**/**1.91** 混跑 | **FAIL**（缺陷，已加 `strain_achieved` 字段） |
| P3 伺服是不是 μ 差的成因 | 提高权限后 \|μ 偏差\| 显著下降 | I=0.1：−13.2%→−20.5%（更差）；I=0.3：−19.5%→−13.7% | **否证**（非伺服主因） |
| P4 暂态是不是 μ/φ 差的成因 | 应变 0.2→3.0 单调收敛 | 非单调；φ 仅从 −7.5% 到 −5.1%（I=0.01），高 I 平台 −13% | **否证**（非暂态主因） |
| P5 判定可分辨性 | 种子散布 << 10% 验收带 | μ 散布 11.5%（I=0.1）、7.2%（I=0.3） | **FAIL**（单种子不可裁，需集合） |

**锚点仍 open**。留给下一轮的唯一主因候选：**有限尺寸 N=294 / 5d×6 层**
（R4 的中成本路径：8d×9 层、N≈2400、≥4 种子、带平稳性门），以及采样窗平稳性
（要求窗内漂移 ≤10% 才算稳态，本单给出测量方法与判据但未达标）。

## ⑧ 复现

```bash
cd <repo>
PYTHONPATH=src python3 benchmarks/plane_shear_mu_i/servo_control_diag.py \
    --stage trace --points 0.001,0.003,0.01,0.03,0.1,0.3 --servo-relax 0.5 --servo-clip 0.05 --jobs 6
PYTHONPATH=src python3 benchmarks/plane_shear_mu_i/servo_control_diag.py \
    --stage authority --points 0.1,0.3 --jobs 6
PYTHONPATH=src python3 benchmarks/plane_shear_mu_i/servo_control_diag.py \
    --stage ensemble --points 0.1,0.3 --seeds 5279,11,23,37 --servo-relax 0.5 --servo-clip 0.05 --jobs 6
PYTHONPATH=src python3 benchmarks/plane_shear_mu_i/servo_control_diag.py \
    --stage strain --points 0.01,0.1,0.3 --strains 0.2,0.6,1.5,3.0 --jobs 6
PYTHONPATH=src python3 benchmarks/plane_shear_mu_i/servo_control_diag.py \
    --stage verdict --sweep benchmarks/plane_shear_mu_i/runs/sweep_baseline.json
PYTHONPATH=src python3 benchmarks/plane_shear_mu_i/servo_diag_figures.py
PYTHONPATH=src python3 -m pytest tests/test_servo_pressure_control.py -q      # 6 passed
PYTHONPATH=src python3 -m pytest tests/test_servo_pressure_control.py \
    benchmarks/plane_shear_mu_i/test_plane_shear_mu_i.py \
    tests/test_brash_ice_rheology_mu_i_resistance.py -q                        # 49 passed, 3 skipped
```

工件：`benchmarks/plane_shear_mu_i/runs/servo_diag/`
（`trace_relax0.2_clip0.002.json`、`trace_relax0.5_clip0.05.json`、`authority_grid.json`、
`ensemble_relax0.5_clip0.05.json`、`strain_ladder.json`、`premise_gate.json`、`tables.md`、
`figures/fig1..3.png`）。环境：loongarch64 / numpy 2.x；总机时 ≈ 6 min CPU（并行）。

## ⑨ 给下一轮的线索

1. **厚床 + 集合**：`--arm thick`（8d×9 层）配 `servo_relax=0.5, servo_max_step_d=0.05`、
   ≥4 种子，把 N 从 294 提到 ~2400，看 μ/φ 偏差是否进入 10% 带（本单唯一未被否证的主因）。
2. **平稳性门**：把 `window.p_rel_drift_over_window ≤10%` 作为"稳态"判据写进判定脚本，
   漂移超限的点判 `not_steady`，不再参与 μ/φ 判定（本单已给统计量）。
3. **伺服律**：中 I（0.01–0.03）保压仍 −11…−17%，是 EMA 滞后 + 刚性墙阻尼的耦合，
   值得单独做一个"常压伺服律"小项（PI + 力前馈）。
4. φ 参考常数需补出处锚（GDR MiDi 全文），否则 φ 判定只能挂在 PROVENANCE-UNVERIFIED 上。