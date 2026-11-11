# DEM-FEM 单向耦合门禁独立化：4 门中 2 门是自比较，补 8 道外部锚

**结论先行**：`examples/dem_fem_oneway` 的四道门全绿，但其中**两道不可能因物理原因变红**——
`impulse ledger` 两侧读的是同一个力数组（实测误差 `0.000e+00`），`work-energy` 的功与被测梁能量
出自同一个 FEM 对象（`5.151e-10`）；`separability` 按单向耦合构造即为真。本单新增
**8 道外部锚**（闭式理论 4 道、独立实现 3 道、跨侧运动学 1 道），把「绿」从"算得对算术"升级为
"对得上理论"：**8/8 通过，最差 |相对误差| 1.84%**，其中两道与梁动力响应直接相关。同时量化了
耦合交接的时间离散限制（`dt/T₁ = 0.14`，f₁ 偏 −4.83%，`dt/2`、`dt/4` 下 −1.35%、−0.30%，
二阶收敛），并订正了能量门的病态口径（端部残余能量是 1e-7 J 量级量相消到 1e-9 的差，
不能当门）。

| 项 | 值 |
|---|---|
| 锚点 | **NEW: DEM-FEM 耦合门独立化**（总账无对应条目；来源=议事会 session-012 求解器视角 P12 提案 D4「DEM-FEM 独立门补齐」） |
| 被测 | `examples/dem_fem_oneway`（DEM 单颗粒冲击简支梁，单向） |
| 新代码 | `src/tensordem/dem_fem_gates.py`、`examples/dem_fem_oneway/independent_gates.py`、`tests/test_dem_fem_independent_gates.py`（11 passed） |
| 产物 | `artifacts/dem_fem_coupling/independent_gates.json`、`docs/figures/dem_fem_independent_gates.png` |
| 复现 | `PYTHONPATH=src python examples/dem_fem_oneway/independent_gates.py`（约 10 s，CPU） |

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

半径为 25 mm、密度 917 kg/m³ 的单个颗粒以 1 m/s 正撞简支梁跨中；DEM 侧记录底板接触反力
时程，作为节点荷载单向施加到 Euler-Bernoulli 梁（2 节点 Hermite 单元 × 30，Newmark 平均加速度）。
梁截面 50×80 mm、L = 0.3 m、E = 2.1e11 Pa、ρ = 7850 kg/m³，故 `EI = 4.48e5 N·m²`、`ρA = 31.4 kg/m`，一阶频率 `f₁ = 2084.74 Hz`（`T₁ = 0.48 ms`）。

| 量 | 值 | 来源 |
|---|---|---|
| 颗粒质量 m | 0.0600175 kg | `BeverlooCase` 式同口径 `ρ·4/3πr³` |
| 接触弹簧（底板） k | 2.0e4 N/m | `CouplingConfig.platen_stiffness`（线性接触，无阻尼） |
| 恢复系数 e | 1.0 | `CouplingConfig.restitution` |
| DEM 步长 dt | 6.709e-5 s | `sim.dt`（接触周期/20 量级） |
| 梁 EI / ρA | 4.48e5 N·m² / 31.4 kg/m | 截面几何 + 材料 |
| 实测接触时长 t_c | 5.367 ms | 力史 overlap>0 的样本窗 |
| 实测峰值力 F_max | 34.646 N | 力史最大绝对值 |
| t_c/T₁ | **11.35** | 本单复算（shipped docstring 声明 11 ✔） |

## 2. 怎么做的（方法 / 工具链）

1. **给原例留测量通道**（最小改动）：`run_fem(..., ringdown_steps=0)` 新增可选参数，撞击结束后用
   零荷载继续积分 5T₁，返回 `ring_down_deflection`——外部锚才能从**时间响应**测梁的自振频率
   （原例在接触结束即 break，没有自由振动段）。默认值 0，原行为逐位不变。
2. **新增独立锚库** `tensordem.dem_fem_gates`：
   * `hookean_impact`：线性弹簧碰撞闭式（`t_c=π√(m/k)`、`F_max=v√(mk)`、`δ_max=v/ω`、`J=(1+e)mv`）；
   * `hertz_impact`：Hertz 闭式（`δ=(5mv²/4k)^{2/5}`、`t_c=2.943275(5m/4k)^{2/5}v^{-1/5}`、`J=2mv`）；
   * `beam_natural_frequencies` / `beam_mode_shape`：连续体模态（`ω_n=(nπ/L)²√(EI/ρA)`）；
   * `modal_response`：**连续体模态叠加 + 分段线性荷载的 Duhamel 精确递推**（无条件稳定，不调步长），
     与 `fem_euler_bernoulli` 零共享代码路径；
   * `evaluate_gates` / `summarise`：每道门强制标注独立性等级
     （`analytic` / `independent_implementation` / `cross_side` / `self_comparison`），
     **`self_comparison` 不计入外部通过数**。
3. **驱动脚本** `examples/dem_fem_oneway/independent_gates.py`：跑一次 DEM → 力史；跑 FEM（耦合 dt）
   + 自由振铃；再把力史线性插值加密到 `dt/2`、`dt/4` 重复 FEM（频率收敛研究）；同时用模态锚算同一
   荷载下的动力响应；输出 JSON + 三联图（力史 / 位移史对比 / 振铃与频谱）。

## 3. 计算结果（数字表，程序化注入，零手抄）

### 3.1 外部锚（`independent_gates.json` → `summary.gates`）

| 门 | 参考值（外部） | 独立性 | 实测 | 相对误差 | 判定 |
|---|---|---|---|---|---|
| G1 接触时长 | 5.4422 ms（`π√(m/k)`） | analytic | 5.3674 ms | −1.38% | PASS |
| G2 峰值接触力 | 34.6461 N（`v√(mk)`） | analytic | 34.6463 N | +6.9e-6 | PASS |
| G3 冲量 | −0.1200350 N·s（`(1+e)mv`） | analytic | −0.1200342 N·s | +6.8e-6 | PASS |
| G4 动量 vs 记录冲量 | 0.1198508 N·s（`m(v_f−v_i)`） | cross_side | 0.1200342 N·s | +0.15% | PASS |
| G5 梁一阶频率（dt/4） | 2084.74 Hz | analytic | 2078.42 Hz | −0.30% | PASS |
| G6 跨中峰值位移 | 4.5458e-8 m（模态） | independent impl | 4.5034e-8 m | −0.93% | PASS |
| G7 位移史 RMS | 模态解（量级 1e-9 m） | independent impl | FEM 同量级 | +0.04% | PASS |
| G8 峰值机械能 | 8.2307e-7 J（模态） | independent impl | 8.0789e-7 J | −1.84% | PASS |
| S1 shipped impulse ledger | — | **self_comparison** | — | 0.000e+00 | 仅一致性 |
| S2 shipped work-energy | — | **self_comparison** | — | −5.15e-10 | 仅一致性 |

**8/8 外部锚通过，最差 1.84%**（<3%）。

### 3.2 时间步收敛（同一力史，FEM 侧加密）

| dt 加密 | f₁ 实测 [Hz] | vs 解析 | 跨中峰值位移 [m] | 备注 |
|---|---|---|---|---|
| dt（耦合交接值） | 1983.95 | **−4.83%** | 4.5034e-8 | `dt/T₁ = 0.14`，欠解析 |
| dt/2 | 2056.50 | −1.35% | 4.5401e-8 | 误差比 3.6 |
| dt/4 | 2078.42 | −0.30% | 4.5466e-8 | 误差比 4.5 ⇒ 二阶收敛 |

峰值位移同样收敛到模态锚（dt/4 时 −0.02%），说明**梁的建模与质量/刚度阵无误，耦合 dt 处的偏差
是纯时间离散误差**（Newmark 平均加速度 + 频率 `ω₁dt = 0.88 rad`）。

### 3.3 静态锚 vs 动力锚

| 口径 | 峰值位移 [m] | DAF | 说明 |
|---|---|---|---|
| 静力解析 `P L³/48EI`（shipped 门） | 4.3501e-8 | 1.000 | t_c≫T₁ 时合法 |
| FEM（耦合 dt） | 4.5034e-8 | 1.0352 | shipped 报告的 3.52% 即此比 |
| 连续体模态动力解 | 4.5458e-8 | 1.0450 | 本单新增锚 |

读数：`t_c/T₁ = 11.35`，动力放大仅 4.5%，故 shipped 的**静力锚在此算例合法**；
但同口径下**动力锚比静力锚更严**（FEM 对静力 3.52%，对动力 0.93%）——静力锚"通过"部分来自
DAF 恰好接近 1，不宜外推到短脉冲算例。

## 4. 结果图

`docs/figures/dem_fem_independent_gates.png`：
(a) DEM 接触力时程（FEM 输入）；(b) 跨中位移：FEM（2 节点 Hermite）vs 连续体模态叠加（独立实现），
虚线标出强制段结束；(c) 自由振铃波形 + 频谱（绿虚线=解析 f₁，注明实测值与 `dt/T₁`）。
图由脚本程序化生成（数字全部从 JSON 直读，无手抄）。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **能量门必须选良态量**：强制段结束时刻的机械能是两个 1e-7 J 量级量相消到 1e-9 的差
   （残余 = 峰值 0.06%），FEM 与模态解在该时刻分别为 1.739e-9 / 7.754e-10 J（差 2.2×），
   但这**不是**能量守恒被破坏——FEM 能量在自由振动段严格守恒（1.739e-9 J 恒定），
   峰值机械能只差 1.84%。故把能量门改为**峰值机械能**，端部残值只作诊断量（JSON
   `energy_conditioning`，内含说明）。这条是本单最重要的一条方法论修正。
2. **模态截断已排除**：9 / 15 / 29 模态下总能量与峰值位移逐位相同（7.7545e-10 J / 4.5458e-8 m），
   故上述差异不是模态截断；梁的 FE 广义特征值 `f₁ = 2084.74 Hz` 与解析一致到打印精度。
3. **闭式锚与实现锚互补**：G2/G3 误差 1e-6 说明 DEM 侧接触律与记录器一致；G1 的 −1.38%
   是采样计数口径（overlap>0 样本窗少记约 2dt），属测量口径而非物理偏差。
4. **G4 的独立性是"跨通道"而非"跨物理"**：冲量与 `mΔv` 在积分器内代数相关，它验证的是
   "记录的反力＝实际施力"，故等级记为 `cross_side` 而非 `analytic`。
5. **本单未动物理、未改 shipped 门**：`run.py` 只增可选参数与输出字段（默认零影响），
   `validate.py` 与 `results.json` 语义不变；shipped 两门在报告中降级为一致性，不删不改阈值。
6. 适用范围：单向耦合、弹性 Euler-Bernoulli 梁、简支；模态锚只实现 `simply_supported`
   （clamped/cantilever 连续体模态**显式 `NotImplementedError`**，不假装支持）。

## 6. 误差判定（逐项 vs 基准）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| 接触时长 | `π√(m/k)` | 1.38% | 是 |
| 峰值接触力 | `v√(mk)` | 6.9e-6 | 是 |
| 冲量 | `(1+e)mv` | 6.8e-6 | 是 |
| 梁一阶频率（dt/4） | `(π/L)²√(EI/ρA)` | 0.30% | 是 |
| 梁一阶频率（耦合 dt） | 同上 | **4.83%** | **否**（已在 §3.2 定位为时间离散误差，二阶收敛） |
| 跨中峰值位移 | 连续体模态动力解 | 0.93% | 是 |
| 位移史 RMS | 连续体模态动力解 | 0.04% | 是 |
| 峰值机械能 | 连续体模态动力解 | 1.84% | 是 |

**honest 声明**：耦合 dt 处的 f₁ 偏差 4.83% 未达 3%，本单没有通过加密 DEM 侧采样去"修绿"它
（那会改变被验证对象），而是保留为**已知限制 + 收敛证据**：把交接加密到 dt/4 即降到 0.30%。

## 7. 复现

```bash
PYTHONPATH=src python -m pytest tests/test_dem_fem_independent_gates.py -q     # 11 passed, ~8 s
PYTHONPATH=src python examples/dem_fem_oneway/independent_gates.py             # 8/8 外部锚, ~10 s
PYTHONPATH=src python examples/dem_fem_oneway/run.py                          # shipped 例，行为不变
```