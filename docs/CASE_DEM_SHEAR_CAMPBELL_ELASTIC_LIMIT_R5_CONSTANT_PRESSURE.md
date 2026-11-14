# CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT — R5：换装置（定压伺服）之后的真实结果

> 轮次：R5 · 锚点 `CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT` · 参考：Campbell 2002, *J. Fluid Mech.* **465**:261（DOI 10.1017/s002211200200109x）
> 预注册（**先写后跑**）：`runs/constant_pressure/prereg.json`（R5a，12 点）、`runs/constant_pressure_rough/prereg.json`（R5b，9 点）
> 工件：`runs/constant_pressure/`（12 点 + `analysis.json` + `exploratory.json` + `tables.md` + `fig_constant_pressure.png`）、`runs/constant_pressure_rough/`（9 点 + `analysis.json` + `tables.md`）
> 内核：**未改动力学**——`tensordem.mu_i_rheology.simulate_servo_shear`（定压伺服直接剪切盒，仓库既有）；本轮只新增实验脚本 `benchmarks/case_dem_shear_campbell_elastic_limit/constant_pressure.py` 与 6 条单元测试
> 判定：**honest-FAIL（预注册判据全部 NOT_SUPPORTED）**，但留下三条可复用结论与一个被实测否定的假设

---

## 1. 问题设定 + 基准出处

R1–R4 全部在**定容** Lees–Edwards 盒（`phi=0.80`，2-D）里测 `tau*(kappa)`，得到单幂律 `tau* = 0.04301 kappa^0.8935`
（`n = 0.213`），并把两条欠账留在账本上：惯性支 `n=2` 从未触到、`kappa_c ~ 1e4` 从未检验。
R4 自己写下的下一轮第一线索是："**定压伺服是唯一能真正到达惯性支的路**"。

本轮先做调研（任务①），再按调研结论换装置（任务④）：

**① 开源横向（集群 `~/oss-cache` + 平台 `oss_inventory`）——负结果，如实记录。**
`oss_inventory` 共 33 行，`repo`/`methods`/`focus` 全域检索 `DEM|granul|yade|liggghts|mercurydpm|sphere` 命中 **0 行**；
tier 分布 12/9/12，`task_key` 全部形如 `campaign-osslbm-*`，即缓存语料是 **LBM/SPH 域**（`~/oss-cache` 本地仅
`Maarten-vd-Sande_lbm`/`hschwane/GraSPH`/`pylbm` 三个仓）。任务书给的
`SELECT * FROM oss_inventory WHERE feature_slug ILIKE '%LIMIT%'` 无法照抄执行：该表**没有 `feature_slug` 列**
（实际列：`repo,tier,focus,status,task_key,methods,multiphase,boundary,gpu,summary,cache_nodes,cache_commit,cache_size_mb`，
ACL 只给 `postgres`/`foundation`，需 `runuser -u postgres` 只读查询）。→ **本域无现成 OSS 实现可学结构**，
方法谱系只能走文献 + 本仓既有装置。

**② 方法谱系（external_anchor）。**
* Campbell (2002) 用**压力控制**剪切得到弹性极限图（`reference_anchors.json` 里那段开放获取转述原文：
  "one is the pressure controlled shearing and the other is the volume controlled shearing"）；
  同一转述给出 `kappa > 1e4` 时 `mu(I)` 才成立（arXiv:1712.02693）。
* 装置标准做法：定压伺服 + Lees–Edwards/SLLOD（Allen & Tildesley；da Cruz et al. 2005 PRE 72:011305 用**粗糙壁**
  抑制壁面滑移），接触律 Cundall–Strack。**这些本仓 `simulate_servo_shear` 已实现**，本轮不重写。
* 判据仍然是锚点那一条：无量纲剪切应力随刚度/Bagnold 数的**标度指数与系数** vs Campbell 2002 模拟。
  系数腿因 ref 全文闭源（`reference_anchors.json`: Unpaywall `best_oa_location=null`）继续 **WITHHELD**；
  本轮沙箱外网也被安全策略拦截（`curl` 直接被拒），无法补齐——如实声明，不凭记忆编数。

**③ 选型沉淀**：见 §7（已写入 `oss_solutions`，`feature_slug='CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT'`）。

---

## 2. 方法与实现

**装置（换，不重写）**：`mu_i_rheology.simulate_servo_shear` —— 3-D 直接剪切盒，x 周期、y/z 墙、顶板**定压伺服**
（目标法向应力 `P=6.25e5 Pa`），剪切率由目标惯性数 `I` 反解 `gdot = I sqrt(P/rho)/d`。
观测量取**内部板 virial**（`bulk_virial_stress`，`bulk_margin_z=0.25`）与壁面估计器并列：

```
kappa = k_n/(rho d^3 gdot^2)      d = 0.01 m, rho = 2500 kg/m^3
tau*  = tau/(rho d^2 gdot^2)      tau = 内部板剪应力
I     = gdot d/sqrt(P/rho)        gdot = 内部板实测剪切率（不是名义值）
phi   = 内部板固含率
```

`kappa_of`/`tau_star_of` **直接复用** `tensordem.elastic_limit_shear`（与 R1–R4 同一函数，避免二源真相），
并有单元测试钉住恒等式 `tau* = mu/I^2`（`tests/test_constant_pressure_campbell.py`，6 passed）。

**关键代码指针**
* `benchmarks/case_dem_shear_campbell_elastic_limit/constant_pressure.py`
  `point_observables()`（κ/τ*/I 与 `tau_star_from_mu` 恒等）、`overlap_proxy()`（`kappa I^2 = k_n/(dP)`）、
  `gate_point()`（5 道门禁）、`_fit_power()`（对数空间最小二乘 + σ）、`analyse()`（P1–P6 判定）、`make_figure()`。
* 复用内核：`src/tensordem/mu_i_rheology.py:916 simulate_servo_shear`、`:695 bulk_virial_stress`、`:871 shear_rate_profile`。

**两轮预注册（都先写后跑）**
* **R5a（光滑平板）**：`I = 0.01/0.02/0.04/0.10/0.20/0.40` 阶梯（`k_n=6.25e4`、`P=6.25e5`，故 `kappa I^2 = k_n/(dP)`），
  外加 2 个种子孪生、2 个 `dt/2` 细化点、2 个固定 `I` 变刚度的塌缩点；
* **R5b（粗糙平板 = 抑制滑移的文献做法 `wall_mode="rough"`，嫁接单层）**：同一阶梯取 5 点 + 2 刚度点 + 种子 + dt 点。
  门禁**未放宽**（写进 prereg 后不改）。

---

## 3. 参数表（无量纲数完整）

| 量 | 值 | 说明 |
|---|---|---|
| `P` | 6.25e5 Pa | 伺服目标法向应力（定压） |
| `k_n` | 6.25e4 / 2.5e5 / 1e6 N/m | 阶梯基准 / 刚度点 |
| `d`, `rho` | 0.01 m, 2500 kg/m³ | `kappa`、`tau*` 的归一化尺度 |
| `mu_c`, `mu_r`, `e` | 0.5, 0.05, 0.3 | Cundall–Strack 接触 |
| `k_t/k_n` | 0.5 | 切向刚度比 |
| `N` | 294（光滑）/ ~300（粗糙） | 粒子数 |
| `kappa`（R5a 可达） | **46.6 … 1.06e5**（3.36 数量级） | 实测 |
| `tau*`（R5a） | **1.66 … 2101** | 实测 |
| `I`（实测） | 0.0122 … 0.536 | 名义 0.01…0.40 |
| `phi` | 0.694 … 0.729（光滑阶梯）；0.559/0.577（刚度点） | 内部板 |
| `kappa I^2 = k_n/(dP)` | 12.0 … 15.8（理论 10） | 见 §5 结构性退化 |
| `dt` | 3.21e-5 s（`k_n=6.25e4`） | `dt/2` 细化点另设 |
| 应变/步数 | 1.1 … 45.2，≥2000 剪切步 | 低 `I` 由 `min_shear_steps` 主导 |

---

## 4. 对照表（逐项 ref vs got + 误差）

### 4.1 预注册判据逐条（**门禁原样套用，未事后修改**）

| id | 判据 | 实测 | 判定 |
|---|---|---|---|
| P1 可达 | ≥5 个过门点、≥1.5 数量级、落在 `50…2e5` | 12 点全部跑出，但**只有 1 点过门**（`I0.200`），跨度 0 数量级 | **NOT_SUPPORTED** |
| P2 指数 | `|beta_CP − beta_CV| ≤ 0.15` | 过门点 <3，无法拟合 | **NOT_SUPPORTED** |
| P3 系数 | `C_CP/C_CV` 在 2 倍内 | 同上 | **NOT_SUPPORTED** |
| P4 浓度 | `phi` 与 `kappa` 单调同号 | 未达 3 点门槛 | **NOT_SUPPORTED** |
| P5 分辨率 | 种子散度 ≤15 %、`dt/2` 变化 ≤10 % | 种子散度 **4.8 %（I0.020）/ 29.8 %（I0.200）**；`dt/2` 使 `tau*` 变 −21.9 %（I0.020）/ +6.6 %（I0.200） | **NOT_SUPPORTED** |
| P6 塌缩（诊断） | 固定 `I`、变 `k_n` 的落点应在阶梯曲线 1.5 倍内 | 比值 **0.252 / 0.053**（低 4×/19×） | **NOT_SUPPORTED** |

**门禁失败的具体原因（诊断，不是借口）**：壁面滑移 + 剖面非线性。
12 个光滑点里 `gdot_meas/gdot_imp − 1` 从 **−34.3 %**（I0.100）到 **+28.1 %**（kn=2.5e5），
内部板速度剖面线性度 `r2 = 0.731 … 0.979`；同一名义点换种子（I0.200 对 I0.200_s2）实测率 347.8 vs 403.0 /s（差 16 %），
说明**顶板光滑、粒子打滑**，名义剪切率不能当控制量。这正是 da Cruz et al. (2005) 用粗糙壁要治的病。

### 4.2 探索性（**post-hoc，标注非预注册，不作通过依据**）`runs/constant_pressure/exploratory.json`

用 R5a 全部 6 个阶梯点（忽略门禁）做对数最小二乘：

| 量 | 定压（R5a） | 定容（R1–R4） | 差 |
|---|---|---|---|
| `beta` | **0.9335 ± 0.0187** | 0.8935 | +0.0399（≪0.15，但见下） |
| `C` | **0.04711** | 0.04301 | +9.5 % |
| RMS(ln 残差) | 10.5 % | 10.3 % | — |
| `n = 2(1−beta)` | 0.133 | 0.213 | — |

**这条"符合"必须打折**：定压路径上 `kappa` 与 `I` 不独立（`kappa I^2 = k_n/(dP)` 由装置定死），
于是 `tau* = mu kappa dP/k_n`，**恒有** `beta = 1 + d ln mu/d ln kappa + d ln P/d ln kappa`。
实测 `dln mu/dln kappa = −0.0462`、`dln P/dln kappa = −0.0203` → **结构性预测 `beta = 0.93343`**，
与拟合值 `0.93345` 相差 **3×10⁻⁵**。所以 beta_CP 不是对锚点指数的独立测量，它是代数量；
把它当成"与 R1–R4 一致 = 确认"就是虚报。预注册里已把这条写成 `known_degeneracy`。

### 4.3 锚点判据（指数 + 系数 vs Campbell 2002）

| 锚点腿 | 参考 | 本轮 got | 结论 |
|---|---|---|---|
| 指数 `n`（弹性极限 `n→0`） | Campbell 摘要"stresses scale purely elastically" | 定压 0.133（结构性代数量）；定容 0.213（R1–R4） | **未判定**：两条路径都不独立，且门禁未过 |
| 惯性支 `n=2` | Bagnold | 未到达（`kappa` 最低 38.8，粗糙面装置失效） | **NOT_SUPPORTED** |
| 系数 `C` | Campbell 数值表（pp. 261–291） | 定压 0.0471 / 定容 0.0430 | **WITHHELD**（ref 闭源，两轮均不可得） |
| `kappa_c ~ 1e4` | 二手开放转述 | 未测（无有效过门数据支撑拐点） | **未判定** |

### 4.4 显式回答"是否 <3 %"

**没有任何一项关键量以 3 % 精度被证明到外部基准上。** 唯一 3×10⁻⁵ 吻合的是 §4.2 那条**代数恒等式**
（`beta = 1 + Σ` 的解析预测 vs 拟合），它是自洽性检验，不是对 Campbell 的验证；把它写成"<3 % 达标"会是错的。

---

## 5. 收敛性 / 参数敏感性（≥2 分辨率、≥2 参数点）

| 轴 | 设置 | 结果 |
|---|---|---|
| 时间步 | `dt` vs `dt/2`（I0.020） | `tau*` 841.8→1026（**+21.9 %**，实测率 26.0→25.3），未收敛 |
| 时间步 | `dt` vs `dt/2`（I0.200） | `tau*` 6.961→5.209（**−25 %**） |
| 种子 | I0.020 / I0.200 | 散度 4.8 % / **29.8 %** |
| 刚度（固定 `I=0.1`） | `k_n = 6.25e4 / 2.5e5 / 1e6` | `phi = 0.729 → 0.577 → 0.559`，`Z = 5.93 → 4.40 → 3.37`，`mu_bulk = 0.383 → 0.362 → 0.279` |
| 压力控制 | 全部点 | `|ΔP/P| ≤ 4 %`（光滑）；粗糙面失控（见下） |

**结论：时间步与种子两个轴都未收敛，这是 R5a 判 NOT_SUPPORTED 的第二个独立原因**，
与壁面滑移无关，不能只怪壁面。

**R5b（粗糙平板）实测失败模式**：`phi → 1.167`（固含率算成 >1，机理：嫁接单层把顶板抬高，
内部板体积用了 `plate_z` 而粒子被挤在一层里）、`|ΔP/P|` 超门、`Z → 9.9–10.5`、
实测率是名义值的 2.5–5 倍 → 该配置（`P=6.25e5` 比仓库已验证工况 `P=6.25e4` 高 10 倍）下**装置退化**，9/9 点全灭。
如实记为 `servo_rough_wall_degenerate_at_P6.25e5`，不当作物理结论。

---

## 6. 失败模式与工件

**已否定的假设（本轮最有价值的一条）**：`tau*(kappa)` 是**单参数律**（即"只要 `kappa` 一样，`tau*` 就一样"）。
证据：固定 `I=0.1`、只把 `k_n` 从 6.25e4 抬到 2.5e5 与 1e6（`kappa` 从 554 到 7.75e3 与 3.05e4），
在**同 `kappa`** 处与阶梯拟合比较：`tau*` 只有拟合值的 **0.252** 与 **0.053** 倍，
同时 `phi` 从 0.729 掉到 0.577/0.559（−21 %/−23 %）。
→ `tau* = f(kappa)` **不成立**，浓度（或 `I`）是共同控制量；任何把锚点压成单参数幂律的写法都过述。

**装置侧失败模式**
1. 光滑平板：顶板滑移，`gdot_meas` 偏离名义值 −34 %…+28 %，种子间 16 % → 名义剪切率不可作控制量（gate 用**实测率**建 κ 也只能算补救）。
2. 粗糙平板：本配置下固含率爆表（1.17）、压力伺服失控 → 该补救在 `P=6.25e5` 不可用。
3. 定压路径的结构性退化：`kappa I^2 = k_n/(dP)` 恒定，使 `beta` 变成代数量（§4.2）。

**复现**
```bash
cd <repo>
PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/constant_pressure.py --prereg
CP_VARIANT=rough PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/constant_pressure.py --prereg
PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/constant_pressure.py --stage grid --jobs 6   # cj2503 (n315, rc=0)
CP_VARIANT=rough PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/constant_pressure.py --stage grid --jobs 6  # cj2520 (n330, rc=0)
PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/constant_pressure.py --analyse
CP_VARIANT=rough PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/constant_pressure.py --analyse
PYTHONPATH=src:benchmarks python3 -m pytest tests/test_constant_pressure_campbell.py -q      # 6 passed
```
工件：`runs/constant_pressure/{prereg,analysis,exploratory}.json, tables.md, fig_constant_pressure.png, points/*.json`（12 点）、
`runs/constant_pressure_rough/{prereg,analysis}.json, tables.md, points/*.json`（9 点）。
环境：loongarch64 集群 n315/n330，numpy 2.x，matplotlib Agg；单点 33–99 s，整批 12 点/6 进程 3.3 min（cj2503），9 点 6.4 min（cj2520）。

---

## 7. 方法选型沉淀（`oss_solutions`，第 2 轮成果入 TK）

* **tag**：`round2-constant-pressure-servo-path+structural-degeneracy`
* **method**：本域 OSS 语料为 0 行 → 不学代码结构，改用文献装置（Campbell 的压力控制剪切）；本仓已有
  `simulate_servo_shear` 直接复用。定压路径观测 τ* 时必须同时报 `kappa I^2 = k_n/(dP)`，因为
  `tau* = mu kappa dP/k_n` 使 β 退化为代数量；判据应改用 `mu(I, phi, kappa)` 三参数面；
  门禁必须包含"实测率 vs 名义率"和"时间步收敛"，否则壁面滑移与未收敛会被误当成物理。
* **transfer**：任何"刚度/率标度"锚点：(1) 先查 `oss_inventory`（本域 0 行即结束，别在 LBM 语料里空转）；
  (2) 定容与定压两条路径不可互相当验证，需按 `kappa I^2` 报装置不变量；(3) 用**实测**率建无量纲数并显式报滑移。

## 8. 下一轮线索（按代价排序）
1. 在 `P = 6.25e4`（仓库已验证压力）重跑粗糙壁阶梯，先把 `phi≤1`、`|ΔP/P|≤5 %` 拿回来，再看 `gdot` 一致性；
2. 把 `dt` 收敛做成门槛（`dt/2` 变化 ≤10 %）后，才允许用定压点讨论指数；
3. 真正的单参数检验只能在**控制 `phi`** 的装置上做（定容 + 变 `k_n`，即 R1–R4 那条路径），
   所以"τ* 是否只依赖 κ"应改成在 `phi` 固定的切片上做二维扫描，而不是换装置。