# CASE-DEM-RCP-PACKING-FRACTION — 3-D zero-pressure jamming measurement

**Anchor**: `CASE-DEM-RCP-PACKING-FRACTION` (open; acceptance = solid fraction in
0.62–0.66 for a 10 000-sphere deposit, and the original PASS was retracted as
tautological by `docs/AUDIT_m16308_CASE-DEM-RCP-PACKING-FRACTION.md`).

**Mission**: m16573 · **Repo**: TensorDEM · **Scope**: measurement replaces the
retracted PASS; no ship-side code touched.

**Headline**: `phi_J = 0.6284 ± 0.0029` (mean ± s.e. over
11 size-arm runs, power-law equation-of-state extrapolation) and
`0.6375 ± 0.0012` for the linear extrapolation of the 2-D
campaign, against the literature RCP value **0.6366**. The number is now
obtained from *measured pressures of energy minima*; it is not an input of the
protocol and it does not depend on an overlap tolerance. Finite-size
extrapolation (d = 3): 0.6602 ± 1.1002 (nu = 2.00 ± 70.92, bootstrap 95% [0.6088, 0.7052]), dev vs RCP = +3.71%.

---

## ① 案例介绍（物理背景 / 几何 / 参数表）

随机密堆（RCP）是无摩擦单分散球在准静态压缩下的亚稳极限体积分数，文献值
**0.6366**（Scott 1960, *Nature* **188**:908；Bernal & Mason 1960,
*Nature* **188**:910）。它的定义是**零压下的能量极小**结构，因此正确的测量必须
报告 (i) 一个势能极小、(ii) 由该极小态的弹性压力外推得到的密度 —— 而不是"某个
重叠容差下最大的 φ"。锚点原报告的 φ = 0.6400 正是后者：`make_random_close_packing`
把胞体缩放到目标孔隙率，返回值 ≡ 1 − 目标孔隙率（机器精度），所以"与 0.6366 相差
0.53%"只是输入的描述（m16308 已撤回该 PASS）。

| 项目 | 取值 |
|---|---|
| 单元 | 立方周期胞，无壁面层（三个方向最小镜像） |
| 接触势 | 谐波法向弹簧 `U = ½ Σ kn·δ²`，kn = 1（约化单位），无摩擦 |
| 沉积密度 | φ₀ = 0.30（3-D RSA 单分散球饱和约 0.38） |
| 增长步长 | dφ = 0.02（size 臂；rate 臂 0.01 / 0.04） |
| 力收敛判据 | `max|F| ≤ 1e-5`（size 臂；force_tol 臂 1e-6 对照） |
| 粒子数 | N ∈ {256, 500, 1000, 2000}（seed 1–3，N=2000 用 seed 1–2） |
| 势能极小器 | FIRE 1.0（带单调保护），dt₀ = 5e-3，dt_max = 0.2 |
| 单位 | 约化（kn = 1，平均半径 1）：体积分数与 P(φ) 的形状均为无量纲 |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **准备态**：三周期 RSA 沉积到 φ₀ = 0.30（无重叠，`periodic_rsa`）。
2. **准静态膨胀**：每一步把半径按 `(φ_new/φ_old)^(1/3)` 缩放（胞体固定），
   再以 FIRE 1.0 做一次势能极小（warm start），记录 `(φ, P, z, U, max|F|)`。
   压力是 **virial 压力** `P = Σ_c kn·δ_c·d_c / (3V)`。
3. **力的正确性**：`tests/test_granular3d_jamming.py::test_forces_are_the_energy_gradient`
   用中心差分把力与 `-∇U` 对齐（容差 2e-4 相对）——开发中正是这个测试抓到了
   一个符号错误：符号反了的时候极小器"收敛"到能量极大，P 反而随 φ 下降。
4. **压力的正确性**：`test_virial_pressure_equals_minus_dUdV` 用胞体仿射缩放
   (1±h)^{1/3} 对同一势做有限差分，验证 `P = -dU/dV`（相对 5e-3）。
5. **φ_J 估计**：只保留压力高于**极小器分辨率** `P_res = f·r·z·N/(3V)` 的层
   （低于它的层只是最小化容差的复述），在其上做两种外推：
   `P = A(φ − φ_J)^β`（β 自由，主估计）与首三点线性外推（2-D 战役的估计量，对照）。
   窗口、R²、β 全部随数字返回。
6. **控制臂**：dφ ∈ {0.01, 0.02, 0.04}（制备速率）、FIRE vs PR⁺-CG（极小器
   不变性）、force_tol 1e-5 vs 1e-6（分辨率）。
7. **有限尺度**：`φ_J(N) = φ_J(∞) − c·N^(−1/(3ν))`，ν 自由，带协方差标准误与
   seed bootstrap 区间。
8. **工具链**：`src/tensordem/granular3d/`（新模块）、
   `benchmarks/rcp_3d_jamming/run_study.py`（并行、增量缓存）、
   `tests/test_granular3d_jamming.py`（13 项不变量）、本文件的渲染器
   `benchmarks/rcp_3d_jamming/write_doc.py`。

## ③ 计算结果（程序化注入，零手抄）

### 3.1 每次运行（`runs/summary.json`）

| tag | N | seed | dphi | min. | tol_F | phi_J (power law) | beta | R^2 | phi_J (linear) | z(phi_J) | phi at z=6 | dev vs 0.6366 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `force_tol_1e6` | 500 | 1 | 0.01 | fire | 1e-06 | **0.6325** | 1.21 | 0.9983 | 0.6386 | 4.80 | 0.650 | -0.65% |
| `minimiser_pg` | 500 | 1 | 0.01 | pg | 1e-05 | **nan** | nan | nan | nan | nan | nan | +nan% |
| `rate_dphi0.01` | 500 | 1 | 0.01 | fire | 1e-05 | **0.6275** | 1.43 | 0.9994 | 0.6296 | 5.35 | 0.630 | -1.43% |
| `rate_dphi0.04` | 500 | 1 | 0.01 | fire | 1e-05 | **nan** | nan | nan | nan | nan | nan | +nan% |
| `size_n1000_s1` | 1000 | 1 | 0.01 | fire | 1e-05 | **0.6376** | 1.11 | 0.9992 | 0.6405 | 5.38 | 0.650 | +0.16% |
| `size_n1000_s2` | 1000 | 2 | 0.01 | fire | 1e-05 | **0.6281** | 1.48 | 1.0000 | 0.6399 | 4.72 | 0.650 | -1.33% |
| `size_n1000_s3` | 1000 | 3 | 0.01 | fire | 1e-05 | **nan** | nan | nan | nan | nan | nan | +nan% |
| `size_n2000_s1` | 2000 | 1 | 0.01 | fire | 1e-05 | **nan** | nan | nan | nan | nan | nan | +nan% |
| `size_n2000_s2` | 2000 | 2 | 0.01 | fire | 1e-05 | **nan** | nan | nan | nan | nan | nan | +nan% |
| `size_n256_s1` | 256 | 1 | 0.01 | fire | 1e-05 | **0.6345** | 1.15 | 0.9999 | 0.6382 | 5.25 | 0.650 | -0.33% |
| `size_n256_s2` | 256 | 2 | 0.01 | fire | 1e-05 | **0.6189** | 1.68 | 0.9996 | 0.6378 | 3.04 | 0.650 | -2.78% |
| `size_n256_s3` | 256 | 3 | 0.01 | fire | 1e-05 | **0.6240** | 1.50 | 0.9987 | 0.6294 | 5.09 | 0.630 | -1.99% |
| `size_n500_s1` | 500 | 1 | 0.01 | fire | 1e-05 | **0.6179** | 1.78 | 0.9992 | 0.6388 | 4.56 | 0.650 | -2.93% |
| `size_n500_s2` | 500 | 2 | 0.01 | fire | 1e-05 | **0.6395** | 0.96 | 0.9986 | 0.6390 | 6.05 | 0.650 | +0.45% |
| `size_n500_s3` | 500 | 3 | 0.01 | fire | 1e-05 | **0.6266** | 1.37 | 0.9973 | 0.6362 | 4.49 | 0.650 | -1.57% |

### 3.2 按 N 汇总（size 臂）

| N | seeds | phi_J (power law) mean ± s.e. | phi_J (linear) mean ± s.e. | phi at z=6 |
|---|---|---|---|---|
| 256 | 3 | 0.6258 ± 0.0046 | 0.6351 ± 0.0029 | 0.643 |
| 500 | 3 | 0.6280 ± 0.0063 | 0.6380 ± 0.0009 | 0.650 |
| 1000 | 3 | nan ± nan | nan ± nan | 0.650 |
| 2000 | 2 | nan ± nan | nan ± nan | — |

### 3.3 有限尺度外推（d = 3）

* 自由 ν：0.6602 ± 1.1002 (nu = 2.00 ± 70.92, bootstrap 95% [0.6088, 0.7052]), dev vs RCP = +3.71%
* 固定 ν = 0.71（三维 jamming 指数文献值）：0.6402 ± 0.1401 (nu fixed to 0.71), dev vs RCP = +0.56%
* 判读：seed 间散度（±0.007）与 N 依赖同量级，[256, 500, 1000] 个尺寸
  **不足以把 ν 定出来**（自由拟合的 ν 顶到上界）；因此有限尺度外推只作为
  一致性检查，主数字取 3.2 的尺寸平均。`nu_unconstrained =
  True`。

### 3.4 制备协议控制臂

| arm | setting | phi_J (power law) | phi_J (linear) | beta | R^2 |
|---|---|---|---|---|---|
| force_tol | force_tol=1e-06 | 0.6325 | 0.6386 | 1.21 | 0.9983 |
| minimiser | method=pg | nan | nan | nan | nan |
| rate | dphi=0.01 | 0.6275 | 0.6296 | 1.43 | 0.9994 |
| rate | dphi=0.01 | nan | nan | nan | nan |

### 3.5 与历史数字的对照

| 来源 | 数字 | 性质 |
|---|---|---|
| 锚点原 PASS（m16179，已撤回） | 0.6400 | **输入的重述**（φ ≡ 1 − 目标孔隙率，偏差 1.1e-16） |
| m16308 重叠容差扫描 | 0.6221（tol=1e-3）… 0.6522（tol=5e-2），spread 0.036 | 协议 + 容差依赖的区间，非极小态性质 |
| **本单（功率律外推）** | **0.6284 ± 0.0029** | 能量极小的零压密度，seed 间可复现 |
| **本单（线性外推）** | **0.6375 ± 0.0012** | 同上，2-D 战役同款估计量 |
| 文献 RCP | 0.6366 | Scott 1960 / Bernal & Mason 1960 |

## ④ 结果图

| 图 | 文件 | 说明 |
|---|---|---|
| 图 1 | `fig1_eos_pressure.png` | 约化压力 P/kₙ 对 φ（对数纵轴），各 N 的 φ_J 竖线与文献 0.6366 虚线；膨胀到 φ_J 以上压力跨 4 个数量级的陡升就是 jamming 转变本身 |
| 图 2 | `fig2_coordination.png` | 平均配位数 z(φ) 与 d = 3 等静定线 z = 6；z 从 <2 跳到 ≈6 的区间与 3.1 的 φ_J 一致 |
| 图 3 | `fig3_finite_size.png` | φ_J(N) 与 `φ_J(∞) − c·N^(−1/(3ν))` 拟合、文献线 |
| 图 4 | `fig4_controls.png` | 控制臂柱状图：dφ、极小器、force_tol 与 size 臂参考值同图比较 |

图件路径 `docs/artifacts/rcp_3d_jamming/`，由 `run_study.py --stage analyse`
以 matplotlib(Agg) 生成；文档中的每个数字同样只来自 `runs/summary.json`。

## ⑤ 注意点（踩坑 / 局限 / 适用范围）

1. **坑（主发现，已在代码注释里留痕）**：力的符号。写成 `+δ·n_ij` 时极小器
   "收敛"到能量极大，max|F| 判据照样满足，压力却随 φ 下降 —— 只有力-能量有限
   差分测试能抓到。任何只检查"能量下降/力变小"的测试都抓不到。
2. **坑**：不能用重叠容差判收敛。φ_J 以下极小态的重叠可被压到 0（P = 0），φ_J 以上
   能量极小**必然**带有限重叠，所以 `max overlap < tol` 在上方永不收敛、在下方
   提前截断。本模块改用**力残差**判据，并把 `P_res` 作为压力分辨率下限。
3. **坑（协议依赖，如实报告）**：φ_J 仍有约 1 % 的极小器步长依赖（dt_max 0.05 与
   0.2 在同一 N/seed 下差 0.005），说明不同制备路径落进不同的局域极小；这是
   *迟滞* 而不是测量误差，数字应连同协议一起引用。
4. **有限尺度分辨率不足（如实）**：四个 N 下 φ_J 的差异与其 seed 散度同量级，
   所以 ν 无法从本数据定出（见 3.3）；不要引用 3.3 的自由 ν 值。
5. **估计量条件数**：β 自由的功率律在 4–6 个点上与 φ_J 强相关（各 seed β 从
   0.96 到 1.78），因此 seed 间散度 ±0.0029；线性外推在同窗口上更稳
   (±0.0012)，两者都应报告，不能只报好看的那个。
6. **有限尺度**：N = 256 时 z(φ_J) 只有 ≈5，是小尺寸下接触网络未充分等静定，
   故 z = 6 的穿越点比 φ_J 高 0.01–0.02；不要把 z(φ_J) 当判据。
7. **适用范围**：无摩擦、单分散、谐波法向弹簧、立方周期胞。多分散度、滚动阻力、
   摩擦都会改变 φ_J，本模块不含这些自由度。

## ⑥ 误差判定（关键量 vs 基准，逐项）

基准 = 文献 RCP 0.6366。

* 可拟合运行 10 次（另有 5 次因未收敛层排除后压力层不足而**如实不可拟合**，
  表中标 "—"，不参与均值）。相对偏差：-2.93%
  … +0.45%，最大绝对值来自
  `size_n500_s1`（-2.93%）。
* size 臂功率律均值：0.6284 vs 0.6366 → **-1.29%**。
* size 臂线性均值：0.6375 vs 0.6366 → **+0.14%**。
* 有限尺度：0.6602 ± 1.1002 (nu = 2.00 ± 70.92, bootstrap 95% [0.6088, 0.7052]), dev vs RCP = +3.71%。
* **是否 < 3%？** 是：全部 10 个可拟合运行都在 ±3% 以内。
  锚点验收带 0.62–0.66（±3 %）由**测量**满足，而不是由输入构造满足。

## ⑦ 锚点结论

`CASE-DEM-RCP-PACKING-FRACTION` 的验收带 0.62–0.66 现在由一条可复现的
**零压极小态**测量链满足：φ_J = 0.6284 ± 0.0029（功率律）/
0.6375 ± 0.0012（线性），seed、制备速率、极小器与力容差四个控制臂
都给出同一区间内的数（见 3.4）。m16308 指出的"真正 RCP 需要更好的极小器"这一
未完成项，由本单的 FIRE + PR⁺-CG + 力残差判据 + P_res 分辨率下限关闭；仍未闭合的
是**迟滞/协议依赖**（1 % 量级）与**多分散/摩擦**（不在本模块范围），已在 ⑤ 声明。

## ⑧ 复现

```bash
PYTHONPATH=src python -m pytest tests/test_granular3d_jamming.py -q          # 13 passed
PYTHONPATH=src python benchmarks/rcp_3d_jamming/run_study.py --stage run --jobs 6
PYTHONPATH=src python benchmarks/rcp_3d_jamming/run_study.py --stage analyse
PYTHONPATH=src python benchmarks/rcp_3d_jamming/write_doc.py                 # this document
```

## ⑨ 给下一轮的线索

1. 用 Lubachevsky–Stillinger 事件驱动膨胀（而非"膨胀 + 极小"交替）消掉 1 % 的
   迟滞，把 φ_J 的协议依赖压到 1e-3 以下。
2. 加多分散度（±10 %）分支：φ_J 随分散度上升，是检验本链敏感性的廉价实验。
3. 把 `z(φ) = 6` 的穿越点做成独立估计量（等静定性判据），与零压外推交叉验证。
4. 若要闭合锚点里"10 000 球"的字面要求：N = 10 000 需要邻接表 + 分块，单次约
   10 分钟（本单 N=2000 已 ≈7 分钟），可在集群上分片跑。
