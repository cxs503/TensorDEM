# CASE-PENETRATION — 球体贯入阻力（TensorDEM）

**算例池锚点**：CASE-PENETRATION ｜ mission 15367 ｜ 分支 `agent/m15367`
**状态**：见 §7（本文件随结果更新；未取得数值处显式标注 WITHHELD，不粉饰）

---

## ① 问题 + 参考文献出处（零手抄，逐条给出原文位置）

**问题**：刚性球体以准静态方式垂直贯入无黏性颗粒床时，深度相关阻力 `F(z)` 与文献标定的无量纲系数 `alpha` 是否一致。

**参考文献**：B. T. Brzinski III, M. Mayor, D. J. Durian,
*Depth-dependent resistance of granular media to vertical penetration*,
**Phys. Rev. Lett. 111, 168002 (2013)**；预印本 arXiv:1307.4638v2（本仓保存的取文版本）。

逐条引用（原文位置以 arXiv:1307.4638v2 的 PDF 计；抓取时间 2026-11-11）：

| 量 | 原文 | 位置 | 数值 |
|---|---|---|---|
| 局部摩擦阻力律 | `dF = alpha mu (rho g z) dA` | p. 1, Eq. (2) | — |
| 无量纲系数 | "`alpha = 35 ± 5` is a number we measure to be the same in nearly all experiments" | p. 1, Eq. (2) 下方同段 | **alpha = 35 ± 5** |
| 球体（z ≥ R）积分结果 | `F(z)/(alpha mu rho g) = pi (z − R/3) R^2` | p. 2, Eq. (3d) | — |
| 球体（z ≤ R）积分结果 | `F(z)/(alpha mu rho g) = pi (R − z/3) z^2` | p. 2, Eq. (3c) | — |
| 介质 | 玻璃球 d = 300 ± 50 µm；bulk density `rho = 1.48 g/cm³`；安息角 22° → `mu = 0.40`；装填率 `phi = 0.59` | p. 1, 第 2 栏 | rho=1480 kg/m³, mu=0.40, phi=0.59 |
| 球体尺寸 | "spherical, with radii R of 1.27 cm, 2.54 cm, and 5.05 cm" | p. 2, 第 1 栏 | R ∈ {1.27, 2.54, 5.05} cm |

**由 (3d) 得到的可对照量**（本算例的被测量）：对 z ≥ R 段做线性拟合 `F = s z + c`，
则

```
alpha_fit = s / (mu * rho_bulk * g * pi * R^2)
```

判据（动手前预定，experimental-design 技能要求）：`|alpha_fit − 35| / 35 ≤ 0.20`（任务容差 <20%）；
文献自带不确定度带为 35±5（即 [30,40]），作为更严的参考带一并报出。

**竞争假设（预先声明，避免事后挑解释）**
- H1：DEM 复现 Eq. (3d)，`alpha_fit ≈ 35`（窗口内）。
- H2（水静压 + 库仑摩擦的朴素标定）：若 `alpha → 1`，则说明 DEM 只给出 `mu rho g ∫z dA` 而没有文献强调的放大因子 alpha，H1 被否证。
- H3：粗粒化（R/d 小）、侧壁围限、或标定 mu 与介质内摩擦角不一致，使 `alpha_fit` 落在窗口外 → honest-FAIL，并给出偏差归因。

---

## ② 方法与实现（复用现有能力，不另起炉灶）

- 求解器：`src/tensordem/sphere_pack_dem.py::SpherePack`（数组化球体 DEM，Hertz/flat-punch 法向 + Mindlin 切向 + 库仑滚动阻力的已验证内核
  `src/tensordem/superquadric_kernel.py`；速度 Verlet / KDK，含半步步位置修正）。
- 初床：`src/tensordem/granular3d_core.py::hcp_positions_in_box`（HCP 点阵，`packing_gap`+抖动），
  重力沉降 → 实测装填率 `phi`（**不是**假设值，直接由粒子数与实测自由面高度算出）。
- 贯入体：额外一个 `fixed=True` 的大球（半径 R），作为刚体贯入器；其与颗粒/壁的接触力由同一内核计算，
  记录其竖向接触力合力 `F_z`（向上为正 = 阻力）。
- 加载协议（与实验的准静态载荷控制等价）：深度增量 `dz = 0.5 d`，每个增量分 `substeps` 个亚步把贯入器
  下移（避免一次性 teleport 造成超大重叠），随后在地下水背景下用强黏性阻尼（`damping_global=200 1/s`）松弛
  `relax` 步至动能可忽略，取松弛末态接触力。准静态加载路径率无关，因此末态力即对应 Eq. (3d) 的 `F(z)`。
- 一维脚本：`benchmarks/case_sphere_penetration/run_case.py`（单算例）；`run_suite.py`（工况族，一个作业跑多个工况）。
- 复现命令（repro_cmd）：
  ```
  cd benchmarks/case_sphere_penetration
  PYTHONPATH=../../src python3 run_case.py --width-grains 24 --depth-grains 10 \
      --radius-over-d 4 --mu 0.40 --dz-over-d 0.5 --relax 300 --settle 1200 \
      --out runs/case_penetration/base
  # 工况族：
  PYTHONPATH=../../src python3 run_suite.py base coarse mu030
  ```

**单位制**：SI（m, kg, s, N）。重力 g = 9.81 m/s²。

---

## ③ 参数表（无量纲数尽量补齐）

| 参数 | 符号 | base 工况 | 文献（Brzinski 2013） | 说明 |
|---|---|---|---|---|
| 颗粒直径 | d | 4.0 mm | 0.30 mm | 粗粒化（受算力约束） |
| 贯入球半径 | R | 16.0 mm（R/d = 4） | 12.7 / 25.4 / 50.5 mm | 无量纲律应与 R 无关 |
| 球/颗粒尺寸比 | R/d | 4 | ≈ 42 | **已知偏差**（粗粒化），列入敏感性 |
| 容器边长 | Lx = Ly | 96 mm = 24 d = 6 R | 190 mm 直径圆柱 ≈ 7.5 R | 侧壁围限相近 |
| 床高 | H | 40 mm = 10 d | 200 mm | 仅需浅贯入 |
| 颗粒数 | N | 见 result.json | — | HCP 沉降 |
| 颗粒密度 | rho_p | 2500 kg/m³ | 玻璃 ≈ 2500 | — |
| 体密度 | rho_bulk = phi rho_p | 实测 phi（≈0.60）→ ≈1500 kg/m³ | 1480 kg/m³ | 用实测 phi，零手抄 |
| 装填率 | phi | ≈0.60（实测，见 result.json） | 0.59 | — |
| 颗粒摩擦系数 | mu | 0.40 | 介质内摩擦 mu = tan22° = 0.40 | DEM 输入值 ≠ 内摩擦角（**混淆因素**） |
| 滚动摩擦 | mu_r | 0.05 | 未报告 | 影响内摩擦角 |
| 恢复系数 | e | 0.4 | 未报告（准静态不敏感） | — |
| 有效弹性模量 | E* | 1.0 GPa | 玻璃 ≈ 30 GPa | 准静态下不影响刚性颗粒结果 |
| 时间步 | dt | 1.44e-6 s（estimate_dt，safety 0.05） | — | — |
| 深度增量 | dz/d | 0.5 | — | 敏感性工况 0.25 |
| 松弛步数/增量 | — | 300 | — | — |
| 贯入深度 | z/R | 0 → 2.0 | 实验 z 至 ~2R | 拟合段 z/R ≥ 1.1 |

---

## ④ 对照表（ref vs got，误差逐项打勾叉）

见 `runs/case_penetration/<case>/result.json` 与 §7（结果回填）。判定列：
`✓` = 相对误差 ≤ 20%，`✗` = 超出。

## ⑤ 收敛性 / 敏感性（至少两点，预先设计）

| 工况 | 改动的量 | 检验的东西 | 预期 |
|---|---|---|---|
| `base` | — | alpha_fit vs 35 | H1 |
| `coarse` | R/d 4 → 3（同一 R，更大颗粒） | 离散化（颗粒尺寸）对 alpha 的影响 | alpha 变化应 < 20%，否则结论受粗粒化支配 |
| `mu030` | mu 0.40 → 0.30 | Eq. (2) 的 `dF ∝ mu` 线性 | `F ∝ mu` ⇒ alpha_fit 应基本不变（若变化，说明 mu 进入的是内摩擦而非直接乘子） |
| `slow`（可选） | dz/d 0.5 → 0.25，relax 300 | 加载协议/准静态性 | alpha 变化 < 20% |

另：`dt` 由 `estimate_dt` 给出；HCP 初床的稳定性在 `diag2.py` 里验证（KE 单调小、无重叠），
破坏性对照（坏初床：疏松点阵造成颗粒自由落体）见 §6 失败模式。

## ⑥ 失败模式、工件与生态摩擦

**失败模式（已实测）**
1. **坏的初床生成器 = 数值爆炸**：自己写的 FCC 填充（`fcc_bed`）实际给出 phi = 0.386（点阵常数与盒子裁剪不自洽），
   颗粒自由落体重排，`KE` 上升到 1e2–1e3 J（床自重势能仅 2e-3 J 量级），力出现 1e2 N 量级的非物理值。
   改用仓内已验证的 `hcp_positions_in_box`（phi = 0.607，KE ~ 1e-7 J，无重叠）后恢复稳定。
   教训：初床必须由仓内已验证工具生成，并**实测** phi，不能相信自写点阵的解析值。
2. **侧壁围限**：demo 尺寸下容器半宽仅 3R，Janssen 效应会压低深部阻力；正式判据要求 ≥ 5R（列为已知局限）。
3. **`mu` 的语义**：文献 `mu` 是介质内摩擦系数（tan 安息角），DEM 输入的是颗粒-颗粒摩擦系数；
   两者只有在安息角标定后才可比（本单未做安息角标定 → 记为已知缺口，`mu030` 工况给出方向性证据）。

**生态摩擦（如实记）**
- 平台知识面：`/api/v1/tensor/knowledge/search` 返回 `engine_offline`（8021 拒绝连接），`knowledge/status` 同；
  开工三查的「同域 capability 检索」因此只能查到**否定性知识台账**（可用，见下），知识库检索不可用。
- 否定性知识台账检索 `q=TensorDEM penetration`：命中均为 `inflated_delivery`（交付虚标）类条目，无「球体贯入阻力」同域的历史失败记录 → 本算例为新域。
- `oss_solutions` 域检索：平台 openapi 无对应 endpoint，未取得（WITHHELD）。
- 集群队列：递交时 `queued_ahead ≈ 300`，算例排队等待远长于计算本身 → 算例池铺开时需考虑并发配额（记 friction）。
- TensorDEM 无 numba/C++/GPU 内核：`SpherePack` 为纯 numpy，实测 N = 9350 时 ≈ 0.25 s/step（单核），
  因此 R/d ≥ 10 的合法球体贯入（N ~ 1e5–1e6）在本仓当前能力下不可达 → 本算例只能做粗粒化版本。

**工件目录**
- `benchmarks/case_sphere_penetration/run_case.py`（单算例，可一键复现）
- `benchmarks/case_sphere_penetration/run_suite.py`（工况族）
- `benchmarks/case_sphere_penetration/diag2.py`（初床稳定性诊断，失败模式 1 的证据）
- `runs/case_penetration/<case>/result.json`、`profile.npz`
---

## ⑦ 本轮结论（honest-FAIL，2026-11-12）

**判定：honest-FAIL（未取得可对照的 alpha_fit）。** 失败阶段 `fail_stage = simulation_protocol_energy_injection`。

已确证的量（可复核的真实工件）：
- 正确初床（仓内 `hcp_positions_in_box`，packing_gap=0.02）：N = 9350、实测 phi = **0.6071**，
  150 步内 KE ~ 1e-7 J、最大重叠比 0、无接触爆炸（`diag2.py` 输出）。⇒ 初床与时间步（dt = 1.438e-6 s）本身是稳定的。
- 穿透阶段：把贯入球插入床面后，系统的动能从 0 单调升到 1.5e2–3e2 J（自写诊断与实际算例均复现，
  `runs/case_penetration/mini/result.json` 的 `profile[*].ke`），而文献律给出的应是
  `F ≈ 0.1 N` 量级（R=6 mm、mu=0.40、rho_bulk≈1500 时按 Eq. (3d) 手算），实测力在 3–272 N 抖动 →
  **不是标定问题，是能量被注入、接触重叠失控**。
- 已排除的原因：坏初床（已修）、贯入器 teleport 重叠（已改为 dz=0.15d 小增量仍复现）。

**仍未排除、下一轮（两轮法接棒）首要验证的原因**
1. 贯入球与侧壁间隙仅 2 颗粒直径（demo 尺度 L 半宽 = 3R、R/d = 3），球-壁间被挤压的颗粒产生
   超大接触力，能量注入床内 → 需把箱体放大到半宽 ≥ 5R（对应 N ≈ 3e4，本仓纯 numpy 内核约 1 s/step，
   单核成本过高，需先解决性能或接受并行分片）。
2. `dt` 相对**最大重叠**的稳定性：`estimate_dt` 用 `reference_overlap=1e-3`，被挤压接触的
   `max_penetration_ratio` 达到 O(1) 时 dt 必须更小 —— 需在算例内做 dt 减半对照（1 次即可判定）。
3. 强黏性阻尼（200 1/s）与刚性固定贯入器的组合是否持续注入能量（改用速度受控的动态贯入器即可区分）。

**已递交、结果未回**：集群作业 `cj1093`（`run_suite.py base coarse mu030`，递交 22:55，递交时
`queued_ahead ≈ 300`，2.6 min 后仍 queued）。该作业的协议仍是本文件 §2 的旧版（大增量），
**其结果不可用于判据**，仅作复现记录。

**没有图的原因**：本单未产出收敛的力-深度曲线或场量数据，任何曲线图都会是失败运行的伪证据，
故不生成图（报告 §4 显式声明）。

**接力清单（下一轮最小可证单元）**
1. 算例内 dt 减半对照（同初床、同增量）→ 判定原因 2；
2. 箱体放大到半宽 5R（或在 R/d=4 下把 Lx 提到 40d）→ 判定原因 1；
3. 改用速度受控的动态贯入器（v = 0.05 m/s ≪ v_c = 0.28 m/s，位移/步 = 3.6e-5 d，永不产生大重叠）→ 判定原因 3；
   代价：行程 2R 需 ~3e5 步，必须靠内核加速（numba/C）或多核分片，否则单例 > 10 h。
4. α 的参照带：35 ± 5（文献自带）；同时报 μ_eff 标定（安息角试验）以把「内摩擦」与「DEM 颗粒摩擦」对齐。
