# 两向 LBM–DEM 曳力闭合（P1）标定与现状

本文件记录 `tensordem.lbmdem`（两向 LBM–DEM 耦合力闭合）的实现口径、解析锚与
2026-11 标定（战役 P55-N08 / mission 5430）的实测结论。数字全部来自
`benchmarks/verified/lbm_dem_drag_calibration/result.json` 与 `gates.json`
（由 `run.py --merge` 程序化生成，门槛由 `evaluate_gates()` 从存盘数字重算）。

## 1. 模块与证据等级

| 组件 | 文件 | 证据等级 | 说明 |
|---|---|---|---|
| D3Q19 BGK + Smagorinsky 流体 | `src/tensordem/lbmdem/lattice.py` | REAL | 动量踢力注入（每步注入动量精确等于 F）、周期 streaming、动量/质量守恒可到机器精度 |
| 动量交换浸没边界（MEM） | `src/tensordem/lbmdem/ibm.py` | REAL | Noble–Torczynski 部分饱和反弹，4³ 子采样固体分数，全固格点按吸收壁计入反作用力 |
| 直接强迫浸没边界（IBM） | `src/tensordem/lbmdem/ibm.py` | REAL | Uhlmann 面元力 + 4 点 Peskin 核（插值/铺展同核，动量交换严格反对称） |
| 解析锚 | `src/tensordem/lbmdem/drag.py` | REAL | Stokes、Schiller–Naumann、Hasimoto/Sangani–Acrivos 阵列修正、Stimson–Jeffery 串联首项 |
| 算例驱动与门槛 | `src/tensordem/lbmdem/calibration.py` | REAL | 定流量泵控固定球、自由沉降、门槛重算 |
| 标定基准 | `benchmarks/verified/lbm_dem_drag_calibration/` | REAL（部分门槛 FAIL，见 §4） | `run.py` / `figures.py` / `report.py` + `result.json` / `gates.json` |

## 2. 方法要点（为什么这么做）

* **力注入用动量踢**：$f_i \mathrel{+}= 3w_i(c_i\!\cdot\!F)$ 满足 $\sum_i w_i c_ic_i = c_s^2 I$、$\sum_i w_i c_i = 0$，
  每步注入动量恰为 $F$、质量恰为 0。因此全局恒等式
  $\Delta(mv) + \Delta P_f - \int F_{ext}\,\mathrm{d}t = 0$ 可以做到 $10^{-12}$ 量级；
  Guo  forcing 的 $F/2$ 速度修正 + $(1-\tfrac1{2\tau})$ 源项并不逐步注入精确的 $F$，
  用它做 $10^{-10}$ 级动量审计只会测到自己的记账缺陷（见 `lattice.py` 文档字符串）。
* **全固格点必须计入反作用力**：把 $\varepsilon_s=1$ 的格点“停在静止平衡态”会静默销毁
  streaming 灌入的动量，曳力随 $\tau$ 增大而系统性偏低（$\tau=12.5$ 时低约 4 倍）。
  改为“吸收壁（保持在自身密度、按壁速运动）+ 动量损失计入力”后，
  $\mathrm{d}P_f/\mathrm{d}t = gV - F$ 在相对 $10^{-12}$ 闭合。
* **定流量泵控**：体力量驱动的周期箱需要 $L^2/\nu$ 量级的扩散时间才能建立稳态剖面
  （$\nu=0.1$、$L=48$ 时约 $2\times10^4$ 步），直接用固定体力量在数千步内一直漂移。
  按平均速度反馈调节体力量把流量钉在靶值后，曳力在 $10^3$ 步内收敛；
  泵力与曳力之差即流体仍在吸收的动量，是收敛的旁证（`momentum_balance_ratio`）。
* **有限域修正**：周期球阵列的曳力增强用稀薄展开
  $F/F_\infty = 1 + 1.7601\varphi^{1/3} + \varphi$（Hasimoto 1959；Sangani–Acrivos 1982）
  解析扣除，$\varphi = N\pi d^3/6L^3$。这是把“盒子效应”与“数值误差”分开的唯一办法：
  $L/d=8\to16$ 时原始曳力比变化 7.8%，与解析预测一致，扣修正后只剩 0.36%。

## 3. 标定协议

* 固定球：$U$ 由 Re 靶值给出，泵控维持；曳力取 MEM/IBM 侧实测值（非泵力），
  同时记录泵力做交叉检验。
* 自由沉降：有效重量 $(\rho_p-\rho_f)gV$ 驱动，Stokes 终速暖启动，动力学更新颗粒速度/位置；
  记录曳力时程、终速、曳力-重力平衡残差与全局动量残差。
* 门槛：见 `gates.json`（判据文字、实测值、阈值、证据算例一一对应）。

## 4. 实测结果（P55-N08，2026-11-07）

### 4.1 通过的门槛

| 门槛 | 实测 | 阈值 | 算例 |
|---|---|---|---|
| Stokes 锚（阵列修正后 $F/3\pi\mu dU$） | $+0.98\%$ | ≤2% | `stokes_d8`（d=8, L/d=8, Re=0.0499） |
| Smagorinsky 不敏感度（Cs=0/0.5/1/2 散布） | $1.19\%$ | ≤2% | `stokes_d6`/`stokes_cs0.5`/`stokes_cs1`/`stokes_cs2` |
| 域长不敏感度（L/d=8 vs 16，修正后） | $0.36\%$ | ≤1% | `stokes_ld8`/`stokes_ld16`（d=4） |
| Schiller–Naumann（Re=5，d=8） | $+3.92\%$ | ≤5% | `reich_re5_d8` |
| 沉降曳力-重力平衡 | $0.40\%$ | ≤3% | `settling_d6` |
| 全局动量守恒残差 | $6.6\times10^{-12}$ | ≤1e-10 | `settling_d6` |

分辨率趋势（Stokes，修正后比值）：d=4 → 0.9162，d=6 → 0.9781，d=8 → 1.0098。
单调趋向 1，说明粗球（d≤6）是主要误差源，d=8 才进入 2% 门。

### 4.2 未通过的门槛（如实记录）

| 门槛 | 实测 | 阈值 | 归因 |
|---|---|---|---|
| 两种 IBM 口径一致性（Re≈0.05） | $+24.2\%$ | ≤2% | 4 点 Peskin 核支撑（4 格）与球径（6 格）同量级，单次强迫不足以复现无滑移，直接强迫曳力偏高 |
| 两种 IBM 口径一致性（Re≈20） | $+20.9\%$ | ≤2% | 同上，叠加 Re=20 的粗分辨率误差 |
| Schiller–Naumann（Re=5，d=6） | $-10.7\%$ | ≤5% | 粗球（6 格）欠分辨边界层 |
| Schiller–Naumann（Re=20，d=6 / d=8） | $-27.5\%$ / $+17.5\%$ | ≤5% | 未收敛（曳力仍漂移）+ 粗分辨；本预算内未能收敛 |
| Schiller–Naumann（Re=50） | 发散（$\tau=0.507$） | ≤5% | BGK 在 $\tau\to1/2$ 的已知稳定性限制，需 MRT/正则化碰撞 |
| 二球串联比值（$\xi=1/8$） | $+12.1\%$ | ≤5% | 周期像污染：单球阵列修正已达 13%，与串联相互作用（18.75%）同量级；DKT 级二体检验需非周期远场或 L/d≥40 |
| 沉降终速（对 SN 平衡，修正后） | $+9.2\%$ | ≤3% | 移动边界每步重建固体分数带来离散跳变；判据用 SN 而非 Stokes（Re_t≈0.8 已非 Stokes 区） |
| 曳力无负阻/尖峰伪迹 | $\min F=-0.576$，最大跳变 4.6 | >0 / ≤5% | 沉降启动瞬态的移动边界重建伪迹（暖启动仍存在） |

**结论**：Stokes 极限（Re→0）的闭合已定量达标（≤1%），并对 Cs 与域长不敏感；
中低 Re（5–20）与移动边界/二体相互作用仍未达标，属如实 FAIL，没有覆盖。
`gates.json.all_recomputed_pass = false`。

## 5. 复现

```bash
python benchmarks/verified/lbm_dem_drag_calibration/run.py --all --jobs 6 --merge
python benchmarks/verified/lbm_dem_drag_calibration/figures.py
python benchmarks/verified/lbm_dem_drag_calibration/report.py --dry-run
pytest tests/test_lbmdem_drag_calibration.py -q      # 18 tests, ~20 s
```

## 6. 局限与后续

* 只有单球/二球、周期边界、Re≤20（有效）；壁面（Brenner 型）修正、颗粒旋转/力矩、
  稠密多球极限都未实现。
* 直接强迫 IBM 需要多次强迫迭代（Uhlmann 建议 2–3 次）与 d≥10 才可能进入 2%；
  移动边界应用连续更新的固体分数或高阶核，以消除 §4.2 的负阻伪迹。
* Re>50 需要 MRT / 正则化碰撞（或 SGS 更强的稳定化）才能在 $\tau\to1/2$ 附近稳定。
* 与 DEM 侧的耦合（接触力 + 曳力 + 浮力 + 附加质量）尚未在真实颗粒群上做能量账本检验。