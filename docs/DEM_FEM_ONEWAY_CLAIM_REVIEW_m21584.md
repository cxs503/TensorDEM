# 复核锚 GAP-ICE-ONEWAY-CLAIM：one-way DEM–FEM 在册判据改名 + 换基准（TensorDEM）

**结论先行**：`examples/dem_fem_oneway` 的 5 项验收全部落地并在机检绿——①原 `impulse` 行改名为
`force_history_port_parity` 且 `results.json` 与 ledger 均标 `scope=numerical_verification_only`
（并写明它是"同一力数组 + 同一梯形规则的代数校验"，非界面守恒）；②偏转判据改为
「动力 FEM vs 连续体模态叠加 ≤1%」，实测 **0.9324%**（基准牵引写的是 0.908%，本单复核为
**锚截断敏感带 0.879–0.947%**，见 §5.1，两者的判定结论一致）；③补四档 dt 收敛链
`dt/T1 = 0.1399/0.0699/0.0350/0.0175 → 误差 0.9324/0.1268/0.0162/0.0063 %`，并把原来那个
**永真**的「相邻两档峰值比 ≥0.9」换掉；④`CouplingConfig` docstring 的 `k=200 N/m` 更正为
`platen_stiffness = 2.0e4 N/m`（并注明 `contact_stiffness=2.0e2` 在单颗粒算例中不参与接触）；
⑤ledger **内联进 `results.json`**，净检出（无 `artifacts/`）下 `validate.py` 10/10 PASS。

| 项 | 值 |
|---|---|
| 锚点 | `GAP-ICE-ONEWAY-CLAIM`（feature_ledger，in_progress → 本单交付） |
| 被测 | `examples/dem_fem_oneway`（DEM 单颗粒冲击简支梁，单向耦合） |
| 改动 | `run.py`（docstring/收敛链/门定义/ledger 内联）、`validate.py`（可单测化 + 2 条新断言） |
| 新增 | `tests/test_dem_fem_oneway_claim_review.py`（8 passed）、`scripts/make_dem_fem_oneway_dt_chain_figure_m21584.py`、本文件、图 |
| 报告 | 平台报告 id **19499**（发布脚本 `tools/publish_report_m21584.py`） |
| 复现 | `PYTHONPATH=src python examples/dem_fem_oneway/run.py`（10 s）→ `validate.py`（10/10 PASS）→ `pytest tests/test_dem_fem_oneway_claim_review.py` |

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

半径 25 mm、密度 917 kg/m³ 的单个颗粒以 1 m/s 正撞简支 Euler–Bernoulli 梁跨中（梁
50×80 mm、L=0.3 m、E=2.1e11 Pa、ρ=7850 kg/m³，`f₁=2084.74 Hz`、`T₁=0.4797 ms`）。
DEM 侧记录底板接触反力时程，单向施加为跨中节点荷载（2 节点 Hermite 单元 ×30、Newmark 平均加速度）。
**耦合交接 dt = 6.709e-5 s（`dt/T₁ = 0.1399`）**，实测 `t_c = 5.367 ms`、`F_max = 34.646 N`、
`J = −0.120034 N·s`（`t_c/T₁ = 11.35`，准静态 DAF 1.045）。

关键修正：接触由**底板弹簧** `platen_stiffness = 2.0e4 N/m` 承担（`t_c = π√(m/k) = 5.44 ms`，
m = 0.0600175 kg），`contact_stiffness = 2.0e2 N/m` 是颗粒-颗粒律，单颗粒算例中**不参与**接触——
原 docstring 把 `t_c ≈ 5.4 ms` 归因于 k=200 N/m，量级差 10 倍（正确值 54.4 ms），已更正。

## 2. 怎么做的（方法 / 工具链）

1. **判据改名 + 定级**：`impulse_*` → `force_history_port_parity_*`；`results.json` 顶层与
   `ledger.interface` 各加 `scope=numerical_verification_only` 与 note（两侧积分的是同一力数组、
   同一梯形规则 ⇒ 误差恒 0，**不是**界面守恒证据）。
2. **换基准**：偏转门从"静力 `PL³/48EI`"改为**连续体模态叠加**（`tensordem.dem_fem_gates.modal_response`，
   分段线性荷载 Duhamel 精确递推，与 FEM 零共享代码路径），阈值 1%。
3. **补收敛链**：新增 `modal_peak_on_grid()`——把同一力史线性插值到 `dt/1, dt/2, dt/4, dt/8` 网格，
   每档单独算模态锚峰值与 FEM 峰值，逐档给 `dt_over_t1 / modal_anchor_peak_m /
   deflection_error_vs_modal_relative`。
4. **删同义反复门**：原门 `peak(dt/8)/peak(dt/4) ≥ 0.9` 在两档几乎相同（比 1.000025）时**永真**；
   新门 = `error(dt/T1=0.14) ≤ 1%` 且 `error(dt/T1=0.0175) ≤ 0.1%` 且两个细档都低于粗档。
   旧比值保留但显式标 `dt_convergence_peak_ratio_is_diagnostic_only: true`，不参与判定。
5. **ledger 入库方式**：`results["ledger"] = ledger`（内联，随 `results.json` 入库），`artifacts/`
   副本仍写但用 `try/except OSError` 保护；`validate.py` 优先读内联 ledger，仅在缺失时回落 artifacts。
6. **可机检化**：`validate.py` 改为 `validate(results_path=None, ledger_path=None)`（可传参单测），
   新增 2 条断言（四档链存在且 dt/T1 对得上；interface 行必须是 port-parity 命名）。

## 3. 计算结果（程序化注入，零手抄）

`examples/dem_fem_oneway/results.json`（本单重跑）：

| 验收项 | 量 | 值 | 判定 |
|---|---|---|---|
| ① port parity | `force_history_port_parity_error_relative` | 0.0（恒等） | PASS（仅一致性） |
| ② 偏转 | FEM 4.503447e-8 m vs 模态 4.545831e-8 m | **0.9324%** | PASS（≤1%） |
| ③ dt 链 | dt/T1 = 0.1399 | 0.9324% | — |
|  | 0.0699 | 0.1268% | — |
|  | 0.0350 | 0.0162% | — |
|  | 0.0175 | 0.0063% | 门 PASS（0.9324% ≤1%，0.0063% ≤0.1%） |
| ④ 能量 | work-energy 相对误差 | 5.151e-10 | PASS（≤1e-8） |
| ⑤ 可分离 | DEM alone / FEM alone 静力误差 | ok / 2.195e-12 | PASS |

`validate.py`：**10/10 PASS**（净检出、无 `artifacts/`）；`pytest tests/test_dem_fem_oneway_claim_review.py`：**8 passed**；
`pytest tests/test_dem_fem_independent_gates.py`：**11 passed**（既有外部锚未回归）。

## 4. 结果图

`docs/figures/dem_fem_oneway_dt_chain_m21584.png`（脚本 `scripts/make_dem_fem_oneway_dt_chain_figure_m21584.py`，
数字全部从 `results.json` 直读）：
(a) FEM 峰值位移与连续体模态锚 vs `dt/T₁`（四个加密档，横轴加密方向）；
(b) 峰值位移误差 vs `dt/T₁` 的双对数图 + 二阶参考斜率 `∝(dt/T₁)²`，每点标注实测百分比。
无场量云图：本算例是单颗粒–梁的 1D/0D 集总对比（无空间场输出），故以时程/收敛曲线代替云图。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

**5.1 基准牵引里的 0.908% 与实测 0.9324% 的差**：用 `tools/probe_m21584_deflection_anchor.py`
对锚做截断敏感度扫描（只读诊断，10 s）：

| 模态阶数 | 5 | 6 | 7 | 8 | 9（ship） | 15 | 29 |
|---|---|---|---|---|---|---|---|
| 误差 [%] | 0.879 | 0.879 | 0.918 | 0.918 | **0.932** | 0.944 | 0.947 |

即 0.908% 落在 **5–29 阶的 0.879–0.947% 带内**（≈6–7 阶处），是**锚截断口径**造成的第 3 位差异
（0.024 pp），不是物理偏差；两种口径都 ≤1%，判定一致。本单不为了对齐尾数去改 `n_modes`（那会把
"复核"变成"调到漂亮数字"），而是在 results/本文件中固化 9 阶口径与上面这张敏感度表。

**5.2 粗档已经 <1%，链不是"从红到绿"**：`dt/T1=0.14` 处误差 0.93% 已满足 1% 门，因此四档链是
**误差地板证据**（0.93% → 0.13% → 0.016% → 0.006%），不是"收敛后才合格"。含义：本算例的 1% 门
不苛刻，`ω₁dt = 0.88 rad` 的时间离散误差只有 ~0.9%；对外推到短脉冲（`t_c/T₁ < 1`）算例，本文的
1% 门**不具可移植性**（此时 `dt` 必须显著小于 `T₁` 才能谈精度收敛）。

**5.3 细档误差受锚网格/插值影响**：`dt/8` 档的锚峰值在加密网格上为 4.546394e-8 m（≠ 基准网格的
4.545831e-8，差 1.2e-4 相对），故 0.0162% 与 0.0063% 之间**不是**单调的二阶收敛读数，而是"已到
地板"的读数；二阶收敛的真正读数是 `0.9324/0.1268 ≈ 7.4`（前 2 档，≥2 阶）与图 (b) 的参考斜率。

**5.4 port parity 永远是 0**：DEM 与 FEM 读同一个 `force_history` 数组并用同一梯形规则，误差
恒等于 0（`0.0`），因此它**不能**变红。本单只做"改名 + 定级 + 写清它不是守恒证据"，没有假装它是
独立门；界面守恒的外部证据在同目录 `independent_gates.py` 的 G3/G4（`(1+e)mv` 与 `mΔv`，误差 6.8e-6 / 0.15%）。

**5.5 适用范围**：单向耦合、弹性 Euler–Bernoulli 梁、简支；双向反馈（梁挠度反过来改变接触）**未建模**。
模态锚只支持 `simply_supported`。

## 6. 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 跨中峰值位移（动力 FEM vs 模态动力解） | 4.545831e-8 m | **0.9324%** | 是 |
| 跨中峰值位移（`dt/4` 档） | 锚 4.545831e-8 m | 0.0162% | 是 |
| 峰值位移（5–29 阶锚敏感带） | 锚截断 | 0.879–0.947% | 是（全带） |
| port parity（一致性，非物理） | 同一力数组 | 0.0（恒等） | —（不计入外部通过数） |
| work-energy（自比较） | 同一 FEM 对象 | 5.151e-10 | —（self_comparison） |
| 静态 FEM vs `PL³/48EI`（可分离性） | 4.3501e-8 m | 2.195e-12 | 是 |

**honest 声明**：①目标锚文里写的 `0.908%` 未被逐位复现（实测 0.9324%，差 0.024 pp），已在 §5.1
给出敏感度证据说明其来源，不粉饰；②`dt/8` 档误差大于 `dt/4` 档的"非单调"现象（若用别家锚口径）本质是
地板噪声，本单用 9 阶锚得到单调序列，但明确标注其末段已到地板；③port parity 与 work-energy 两门
仍是自比较，只是已被**定级**（scope/self_comparison），未升级为物理证据。

## 7. 复现

```bash
PYTHONPATH=src python examples/dem_fem_oneway/run.py                              # 10 s, 重写 results.json（含内联 ledger）
PYTHONPATH=src python examples/dem_fem_oneway/validate.py                         # 10/10 PASS（不需要 artifacts/）
PYTHONPATH=src python -m pytest tests/test_dem_fem_oneway_claim_review.py -q       # 8 passed, 0.3 s
PYTHONPATH=src python -m pytest tests/test_dem_fem_independent_gates.py -q         # 11 passed（回归）
PYTHONPATH=src python tools/probe_m21584_deflection_anchor.py                     # 锚截断敏感度表（只读）
PYTHONPATH=src python scripts/make_dem_fem_oneway_dt_chain_figure_m21584.py       # 重画图
```