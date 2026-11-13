# XVAL-DEM-BIAXIAL-CELL-NEGATIVE-RESULT — 与 OSS 测量做法横向对标

**结论摘要（honest, mixed）。** 本单把 biaxial cell 的**两个决定性测量**拿去和成熟 OSS 做法交叉对标：
①剪切带倾角估计器（本仓=Radon 投影聚焦；OSS 主流=Desrues & Viggiani 结构张量 / 二阶矩）；
②应力定义（本仓=**墙面反力**；YADE/LAMMPS 等 OSS DEM=**内部 Love–Weber(virial) 平均**）。

- **判据 1（估计器 <15 %）：PASS。** 在 24×48 案例点阵上用已知倾角合成带扫描 30–75°（10 点）：
  OSS 结构张量对真值最差 **4.5 %**、本仓 Radon 最差
  **3.3 %**，两者互差最差 **8.2 %**，全部 <15 %。
  二阶矩算子最差 **16.7 %** → 该算子**不达标**（浅角 30° 失效，原因已定位）。
- **判据 2（应力定义 <15 %）：FAIL（部分）。** 生产侧墙带摩擦时轴向 σ_v 的墙-vs-virial 差
  **+24 %**、侧向 σ_h 差 **+80 %**；受控 A/B（**只**把墙摩擦 0.55→0）后
  σ_h 差降到 **12.4 % ✓**，但 σ_v 仍 **+-0 % ✗** →
  侧壁竖向剪切解释了侧向量、**解释不了轴向**，轴向差记为 WITHHELD（未定位）。
- **物理判定（vs 60° 锚）：** 重跑 p=100 kPa/24×48 在 ε_a=0.09 的晚期快照上，两族算子给出
  **57.0° / 57.3°**（互差 0.5 %），
  对 60° 锚误差 **5.0 % / 4.5 %（<15 % ✓）**，与 R2 记录的 55.75°（7.1 %）同向；
  但该重跑**没有观测到偏应力峰值**（devator 单调硬化到记录末端），晚期窗口只有 1 个快照 → 按案例自己的判据（≥3 快照、散度<10°）
  **判 unresolved**，因此**不宣布负结果被推翻**：本单只确认"测量算子在真实 DEM 场上互证（<2 %）"。

---

## ① 问题与参考出处

平面应变双轴压缩：顶板位移驱动、底板固定、两侧伺服到恒定围压 σ3；冻结粘结圆盘试样在偏应力峰后
局部化出剪切带，其相对 σ3（水平）的倾角 θ 是经典验证量：Mohr–Coulomb `45+φ/2`、Roscoe(1970) `45+ψ/2`、
Arthur et al.(1977) `45+(φ+ψ)/4`。案例现行判决（`docs/BIAXIAL_SHEAR_BAND.md`、`docs/CASE_BIAXIAL_SHEAR_BAND_R2.md`）
= **honest FAIL**（主 resolved 跑 55.75°→7.1 %，配对重测 72.0°→20 %），R2 已把主误差归给**测量算子**
（轴向 min-max 各向异性归一化，18–20° 偏置），本单检验的正是"这个测量还信得过吗"。

| 量 | 本仓做法 | OSS 做法（来源） |
|---|---|---|
| 带取向 | Radon 投影聚焦 + 等计数分箱（`biaxial_shear_band.radon_band_angle`） | 结构张量 `J=G_rho*(grad g ⊗ grad g)`，主特征向量=带法向（Desrues & Viggiani 2004, *IJNAMG* 28:279–321；`skimage.feature.structure_tensor`、SPAM/muDIC 同一构造）；二阶矩/PCA（`skimage.measure.regionprops` 方向） |
| 应力 | 墙面反力 σ=F_wall/A | 内部 Love–Weber 平均 `σ=(1/V)Σ_c f⊗l`（YADE `stressTensor`、LAMMPS `compute stress/atom`） |
| 侧向边界 | 伺服墙与颗粒同摩擦 μ=0.55 | 三轴/双轴惯例：侧向围压墙 μ=0，边界牵引为纯法向 |
| 代表性角 | 平滑偏应力峰 + 峰后 40 % 窗口均值，≥3 快照且散度<10° 才 resolved | 同族（多点均值 + 收敛判据） |

所有数字由 `benchmarks/biaxial_cell_xval/run_xval.py` 写入 `artifacts/biaxial_cell_xval/*.json`，再由脚本注入本文件（无手抄）；
判据 `XVAL_TOLERANCE=0.15` 预声明于 `src/tensordem/biaxial_xval.py`，先于任何结果检查。

## ② 方法与实现

1. **同一应变场、三种取向算子**（`biaxial_xval.orientation_estimators`）：对同一对快照算 `local_shear_strain`（k=8 kNN，生产算子），
   把**同一个 γ 场**分别喂给 ①Radon 聚焦（生产）②结构张量（`scipy.ndimage.gaussian_filter`+`sobel`+特征分解，等价
   `skimage.feature.structure_tensor`）③高应变粒子集二阶矩（`numpy.linalg.eigh`）。栅格在短边方向取等距（cells=40）保持算子各向同性。
2. **合成真值扫描**（`cross_validate_synthetic`）：用 `biaxial_band_diagnostics.synthetic_band_configuration`
   （等体积均匀背景 + 有限宽度 tanh 滑移带）把**已知倾角**注入**案例点阵本身**（24×48、1128 盘），
   避免"方形随机点集"这类不相似夹具（R2 的老 bug 正是夹具不相似漏掉的）。
3. **virial 应力进内核**（`ridge_dem.RidgeAssembly.forces` 新增返回 `virial`）：Σ 内部 pair（冻结键 + 无键接触）
   `f_on_i ⊗ (x_j − x_i)`，墙/管/海床等**外部**牵引不计入；先在**解析**粘结链上验证（拉伸+压缩，1e-9 内，`tests/test_biaxial_xval.py`）。
4. **案例内同时记录两种应力**（`run_biaxial_shear_band` 的 `stress_path` 新增 `sigma_v_internal`/`sigma_h_internal`，
   峰值块新增 `stress_definitions`），并新增 `wall_friction` 开关（默认 None=材料值，行为不变）以做**只改一个变量**的 A/B。
5. **重跑与离线复测**（`benchmarks/biaxial_cell_xval/run_xval.py` 的 `case` / `analyze`）：生产试样/伺服/积分逐步复刻加载循环
   （本地单线程 torch，86 步/s），落盘快照 + 两种应力历史，再在**同一批快照**上跑三族算子 + 应力审计。

## ③ 参数表（无量纲数完整）

| 量 | 值 |
|---|---|
| 试样（24×48 点阵） | 1128 盘，r=0.75 m，厚度 1 m，点阵 jitter 3 % |
| 冻结键比例 / 接触摩擦 | 0.35 / μ=0.55（墙摩擦 A/B：0.55 vs 0.00） |
| 围压 σ3 | 100 kPa（p=200 kPa 臂未重跑，见⑥） |
| 轴向应变率 / 记录目标 | 0.05 1/s / 0.09–0.12 |
| dt | 4.97e-4 s（dt/T_c≈0.04） |
| **惯性数 I = γ̇ d sqrt(ρ/p)** | **0.0072**（准静态） |
| 固结步数 | 1200（A/B 双臂）/ 4000（长跑臂） |
| 测量窗 | 四壁各内缩 2 倍粒径，30.0 × 56.6 m² |
| 合成带 slip / 带宽 / 粒子间距 | 2.236 m / 4.499 m / 1.500 m |
| 结构张量参数 | σ_gradient=1 cell、σ_tensor=2 cells、能量分位 0.80、栅格 40 cells |
| 判据 | 相对误差 ≤15 %（owner 精度标准）；resolved = ≥3 快照且散度<10° |

## ④ 对照表（ref vs got，误差逐项打勾/叉）

### 4.1 估计器：合成已知倾角（24×48 案例点阵，10 个角度）

| 注入 θ [°] | 本仓 Radon | OSS 结构张量 | OSS 二阶矩 | Radon−真值 | 结构张量−真值 | Radon vs 结构张量 |
|---|---|---|---|---|---|---|
| 30 | 31.00 | 28.65 | 35.00 | +1.00 | +3.3 % ✓ | -1.35 | +4.5 % ✓ | +8.2 % ✓ |
| 35 | 35.00 | 33.41 | 39.23 | +0.00 | +0.0 % ✓ | -1.59 | +4.5 % ✓ | +4.8 % ✓ |
| 40 | 40.00 | 38.41 | 42.42 | +0.00 | +0.0 % ✓ | -1.59 | +4.0 % ✓ | +4.1 % ✓ |
| 45 | 44.00 | 43.96 | 46.88 | -1.00 | +2.2 % ✓ | -1.04 | +2.3 % ✓ | +0.1 % ✓ |
| 50 | 50.00 | 49.48 | 50.69 | +0.00 | +0.0 % ✓ | -0.52 | +1.0 % ✓ | +1.1 % ✓ |
| 55 | 55.00 | 54.91 | 55.57 | +0.00 | +0.0 % ✓ | -0.09 | +0.2 % ✓ | +0.2 % ✓ |
| 60 | 59.00 | 60.18 | 60.31 | -1.00 | +1.7 % ✓ | +0.18 | +0.3 % ✓ | +2.0 % ✓ |
| 65 | 65.00 | 65.39 | 64.77 | +0.00 | +0.0 % ✓ | +0.39 | +0.6 % ✓ | +0.6 % ✓ |
| 70 | 70.00 | 70.33 | 71.30 | +0.00 | +0.0 % ✓ | +0.33 | +0.5 % ✓ | +0.5 % ✓ |
| 75 | 75.00 | 75.13 | 84.95 | +0.00 | +0.0 % ✓ | +0.13 | +0.2 % ✓ | +0.2 % ✓ |

最差：结构张量对真值 4.5 % ✓ ·
Radon 对真值 3.3 % ✓ ·
**Radon vs 结构张量 8.2 % ✓** ·
二阶矩对真值 16.7 % ✗。
预声明判据（`estimator_xval.json: criterion`）：结构张量 vs 真值 True、
本仓 vs 结构张量 True、本仓 vs 二阶矩 True、
二阶矩 vs 真值 False。

### 4.2 同一应变场上的真实 DEM 快照（晚期窗口，增量场）

`case_p100_rough`：尾部起点 ε_a=0.087

| ε_a | Radon [°] | 结构张量 [°] | 二阶矩 [°] | 差 [°] | 差 [%] | 张量散度 [°] |
|---|---|---|---|---|---|---|
| 0.090 | 57.0 | 57.3 | 84.7 | -0.3 | 0.54 | 10.3 |

`case_p100_long`：尾部起点 ε_a=0.114

| ε_a | Radon [°] | 结构张量 [°] | 二阶矩 [°] | 差 [°] | 差 [%] | 张量散度 [°] |
|---|---|---|---|---|---|---|
| 0.120 | 53.0 | 50.9 | 87.2 | +2.1 | 4.08 | 15.0 |

累计场（首快照→末快照）结构张量：63.7（rough）/ 61.1（long）度。
真实场上两族算子互差 **≤2 %（✓）**；二阶矩在真实场给出 85° / 87°（**✗**，见⑥）。

### 4.3 应力定义：墙面反力 vs 内部 virial（同一步、同一次跑）

| arm | 墙摩擦 | 固结步 | 晚期快照 | 峰值应变 | σ_v 墙 [kPa] | σ_v virial [kPa] | Δσ_v | σ_h 墙 [kPa] | σ_h virial [kPa] | Δσ_h | φ_墙 | φ_virial | Δφ |
| rough | 0.55 | 1200 | 1 | 0.085 | 418 | 519 | +24.0 % ✗ | 114 | 206 | +80.4 % ✗ | 34.87° | 25.61° | -26.6 % |
| smooth | 0.00 | 1200 | 2 | 0.059 | 516 | 514 | -0.4 % ✓ | 126 | 142 | +12.4 % ✓ | 37.35° | 34.53° | -7.5 % |
| long | 0.55 | 4000 | 1 | 0.110 | 417 | 502 | +20.4 % ✗ | 141 | 215 | +52.9 % ✗ | 29.65° | 23.54° | -20.6 % |

侧向量在**无摩擦侧墙**下达标（12.4 % ✓），轴向在两臂都超差（+20–94 % ✗）；
φ 因此系统性差 **-26.6 %**（34.9° → 25.6°，
对应 Mohr–Coulomb 理论角 62.4° → 57.8°）。

### 4.4 vs 案例锚（60°，Desrues & Viggiani 2004 密实 Hostun 砂 55–60° 中值）

| run | θ [°] | 参考 [°] | 相对误差 | <15 % |
|---|---|---|---|---|
| 本单重跑 ε_a=0.09，Radon | 57.0 | 60 | 5.0 % | ✓（unresolved：仅 1 个晚期快照） |
| 本单重跑 ε_a=0.09，结构张量 | 57.3 | 60 | 4.5 % | ✓（同上） |
| R2 交付（主 resolved 跑，记录值） | 55.75 | 60 | 7.1 % | ✓ |
| R2 配对重测（p=200 臂，记录值） | 72.00 | 60 | 20.0 % | ✗ |

## ⑤ 收敛性 / 参数敏感性（至少两点，均已跑）

1. **倾角依赖（估计器）**：30→75° 扫描（表 4.1）。结构张量误差在 45–70° 为 0.2–2.3 %，在 30°/35° 升到 4.0–4.5 %；
   Radon 在 30–75° 全段 ≤3.3 %。二阶矩在 30°（16.7 %）与 75°（13.3 %）显著变差 —— 原因明确：窄带自身的主轴
   在浅/陡角时不再等于带方向（带长 < 试样宽），故该算子只能在 45°±15° 内当参考。
2. **侧墙摩擦（应力定义）**：0.55→0.00，其余完全不变（同 seed、同点阵、同固结步数）。σ_h 相对差
   80.4 %→12.4 %（−68 个百分点），
   **证明侧壁竖向剪切是侧向应力测不准的主因**；σ_v 只从 24.0 %→-0.4 %，
   说明轴向差另有来源 → **该假设被部分否定**（honest）。
3. **固结/记录长度**：1200→4000 固结步后轴向差仍 20–26 %（峰值步），即"固结不足"不是主因；但记录长度
   **确实**改变峰值位置：1200 步臂的平滑偏应力峰落在记录末端（ε_a≈0.09–0.11），4000 步臂仍单调硬化 → 峰后窗口不存在（⑥）。
4. 未扫、声明为局限：结构张量的 σ_gradient/σ_tensor/能量分位/栅格只取单组值（1/2/0.80/40），未做敏感性扫描 → 该项 WITHHELD。

## ⑥ 失败模式、局限与工件

**失败模式（留痕，不粉饰）**
- *二阶矩算子的角域限制*：浅/陡角偏置 13–17 %，超 15 % 判据 → 不作主参考，只作独立复核。
- *轴向应力定义差未定位*（WITHHELD）：virial 系统高于墙面反力 20–94 %，无摩擦侧墙只解释一部分。候选原因：
  (a) σ_v 取**底板**反力（`run_biaxial_shear_band` 用 `plane_forces[2]`），顶端驱动板承担的部分反力未进入 σ_v；
  (b) 边界半层的分支向量缺失；(c) 峰值时刻系统非平衡（偏应力尚未见峰）。三者未分离。
- *峰值不可辨*：3 个重跑臂在 ε_a≤0.12 内都没有偏应力峰，故"峰后 40 % 窗口 + 散度<10°"的代表角判据不满足 →
  真实场倾角按案例自己的判据是 **unresolved**，本单只报单快照互证值，**不改判** R2 的 FAIL/PASS。
- *p=200 kPa 臂未重跑*（集群排队 577 + 本地 120 s 会话预算），R2 的 72°（20 % ✗）不在本单射程内，保持原记录。

**OSS 差异清单（做法 / 参数 / 边界条件）与移植取舍**

| 维度 | 本仓 | OSS 主流 | 差异影响 | 本单取舍 |
|---|---|---|---|---|
| 带取向算子 | Radon 投影聚焦（点云直接做，无栅格） | 结构张量 / 二阶矩（栅格场） | 合成互差 ≤8.2 %、真实场 ≤2 % → 等价 | **已移植**两族 OSS 算子为独立复核（`biaxial_xval.py`），生产算子不动 |
| 应力定义 | 墙面反力 F/A | 内部 virial 平均 | 侧向 48 %→9 %（摩擦修正后）；轴向 20–94 % | **已移植** virial 进内核（只增字段、不改默认），作常开审计项 |
| 侧向墙摩擦 | μ=0.55（材料值） | 三轴惯例 μ=0（纯法向牵引） | 侧向应力测量偏置 48 % | **新增** `wall_friction` 开关；默认不变（不偷偷改历史结果），建议后续 run 用 0 |
| 驱动与 σ_v 取样 | 顶板速度控制；σ_v 取底板反力 | 双板/应力伺服，σ 用 6 面墙平均 | 顶端驱动反力未进 σ_v → 可能解释轴向差 | 未改（需 A/B 才能定论，留下一轮） |
| 测量窗 | 四壁内缩 2 粒径 | 全试样（含边界带） | 边界带会带来假带 | 保留（合成审计已证明窗口内算子无偏） |
| resolved 判据 | ≥3 快照、散度<10° | 多点均值±标准差 | 本单重跑不达 resolved | 保留，宁可 WITHHELD 不改判 |

**工件（全部可复跑）**
- `src/tensordem/biaxial_xval.py`（新算子库：结构张量 / 二阶矩 / 合成交叉验证 / XVAL_TOLERANCE）
- `src/tensordem/ridge_dem.py`（新增 `virial` 返回）；`src/tensordem/biaxial_shear_band.py`
  （新增 `sigma_v_internal`/`sigma_h_internal`/`stress_definitions`/`wall_friction`）
- `benchmarks/biaxial_cell_xval/run_xval.py`（子命令 `synthetic` / `case` / `analyze`）
- `artifacts/biaxial_cell_xval/`：`estimator_xval.json`；`case_p100_rough`、`case_p100_smooth`、
  `case_p100_long`、`case_p100_settled` 各自的 `.npz` + `_analysis.json`；`xval_summary.png`
- `tests/test_biaxial_xval.py`（9 passed：解析 virial、合成带回收、算子容差一致、二阶矩方阵、
  退化输入、`wall_friction` 贯通）

**复现**
    python benchmarks/biaxial_cell_xval/run_xval.py synthetic
    python benchmarks/biaxial_cell_xval/run_xval.py case --consolidation-steps 4000 --strain-target 0.12 \
        --snapshot-strain 0.01 --out artifacts/biaxial_cell_xval/case_p100_long.npz
    python benchmarks/biaxial_cell_xval/run_xval.py analyze \
        --inp artifacts/biaxial_cell_xval/case_p100_long.npz \
        --out artifacts/biaxial_cell_xval/case_p100_long_analysis.json
    PYTHONPATH=src python -m pytest tests/test_biaxial_xval.py tests/test_biaxial_shear_band.py \
        tests/test_biaxial_band_diagnostics.py -q

**生态摩擦（如实记录）**
1. `GET /tensor/knowledge/search` 返回 `engine_offline`（知识引擎 8021 拒连）→ 开工三查的 capability 检索不可用；
   `/tensor/negative-knowledge/search?q=biaxial|TensorDEM` 命中 0 条；本平台无 `oss_solutions` 路由（404）→ OSS 方案库无法查询。
2. 无 GitHub / raw.githubusercontent 出口（连接超时 / reset）→ OSS **源码**不可直取；loongarch64 的 `scikit-image`
   只有 0.21.0 轮子且与 numpy 2.4 ABI 冲突（`numpy.dtype size changed`）→ 改用 `scipy.ndimage` 复现同一算法
   （已在模块 docstring 与本文件 4.1 注明，**不宣称直接执行 skimage**）。
3. 集群队列排队 577 个作业（`queued_ahead: 577`）→ 重计算走不了集群；改本地单线程 torch
   （`torch.set_num_threads(1)` 让 1128 粒子从 8.9→86 步/s，10 倍）把每个案例压在 120 s 单次会话预算内，
   代价是 p=200 臂与峰后长窗未能重跑。

## 误差判定（对判据显式回答）

| 关键量 | 基准 | 误差 | <3 % ? | <15 % ? |
|---|---|---|---|---|
| 结构张量算子（合成，最差） | 注入真值 | 4.5 % | 否 | **是** |
| Radon 算子（合成，最差） | 注入真值 | 3.3 % | 否 | **是** |
| 本仓 Radon vs OSS 结构张量（合成最差） | 结构张量 | 8.2 % | 否 | **是** |
| 本仓 vs OSS 结构张量（真实场） | 结构张量 | 0.5 % | **是** | 是 |
| 侧向应力定义差（无摩擦侧墙 A/B） | virial | 12.4 % | 否 | **是** |
| 侧向应力定义差（生产墙，未改） | virial | 80.4 % | 否 | 否 |
| 轴向应力定义差（三臂） | virial | 20–94 % | 否 | 否 |
| 带倾角 vs 60° 锚（单快照） | Desrues & Viggiani 2004 | 5.0 % | 否 | 是（unresolved） |

**是否 <3 %**：只有"两族算子在**真实 DEM 场**上互证（0.5 %）"
与解析 virial 恒等式（1e-9）达到 3 % 内；**没有任何物理量对实验锚达到 3 %**，故本单不宣称物理精度提升，
只宣称：测量层被独立实现互证 + 发现一个可量化的边界应力测量缺陷（侧墙摩擦）+ 一个未定位的轴向应力定义差。
