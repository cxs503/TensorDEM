# GAP-DEM-CLUMP · round 2 (m15028) — 静态滚动阻力 + 形状效应验证

锚点：`GAP-DEM-CLUMP`（非球形颗粒 + 形状效应验证）。本轮按 owner「两轮法第 2 轮」
先做**调研**（开源横向 + 方法谱系），再**结合实现**（本仓代码 + 机器可判据），
并把负结果按 honest-FAIL 留痕（fail_stage 见 §6）。

前轮（m14996）honest-FAIL，未留 `fail_stage`/`failure_reason`；本轮把失败点定位到
**协议几何不可行**（§3），而不是"没跑完"，因此下一轮拿到的是可执行的重启条件。

---

## 1 调研

### 1.1 开源横向（①-①）

| 检索 | 结果 |
|---|---|
| `SELECT repo, license, methods FROM oss_inventory WHERE methods/summary/focus ILIKE {clump, multi-sphere, rolling resistance}` | **0 行**（表内共 218 行 / 140 个 feature_slug） |
| 集群节点 `~/oss-cache`（n315/n330） | 只有 LBM 包（Maarten-vd-Sande_lbm、hschwane、pylbm），**无 DEM/clump 实现** |
| `oss_solutions WHERE feature_slug='GAP-DEM-CLUMP'` | 施工前 **0 行** |

→ 结论（写入 `oss_solutions`：`oss-inventory/tensorplatform` + tag
`horizontal-survey-empty`）：**没有可抄结构的成熟第三方 clump 实现**，因此
本轮的横向 leg 是"空集"这个**事实**，方法只能从文献谱系取。零手抄纪律：不引用
没读过的实现源码。

### 1.2 方法谱系（①-②，external_anchor 落 `oss_solutions`）

| 文献 | 方法 | 对本轮的作用 |
|---|---|---|
| Ai, Chen, Rotter, Ooi (2011) *Powder Technol.* 206:269–282 | 四种滚动阻力模型评估：常数力矩 / 粘性 / **速率比例正则化 `w/(w+ε)`** / 弹性-塑性滚动弹簧 + Coulomb 上限 | **判据来源**：速率比例型（本仓旧法就是这一族）**没有静态分支**，不能支撑坡面静平衡 → 用它测 repose 趋势，趋势无法归因于滚动阻力 |
| Wensrich & Katterfeld (2012) *Powder Technol.* 217:409–417 | 滚动摩擦是**形状的替代量**：球的 repose 随 μ_r 单调升（μ_r 0→0.5 对应 21°→32°） | **设计规则**：μ_r 必须在"球基线"上标定后**固定**，形状族比较才不混淆（本轮 `MU_ROLL=0.3` 全配置唯一） |
| Jiang, Yu, Harris (2005) *Comput. Geotech.* 32:340–357 | 弹性-塑性滚动弹簧 `M=-k_rθ`，`k_r=2.25 μ_r² k_n R²`，塑性上限 `μ_r R |Fn|` | **可行性算例**（§2.3）：在本基准的软接触参数下屈服角 ≈ 1.6 rad ⇒ 退化为弱弹性弹簧，故**不采纳**为默认（记录理由，不试错硬套） |
| Zhou et al. (1999) *Physica A* 269:536–553 | lifted-box 沙堆 repose 标定算例；球 22–25°，角砾/椭球 35–45° | **外部锚**：趋势方向 + 球基线带（沿用仓内 `LIT`） |

### 1.3 选型沉淀（①-③）

`oss_solutions` 共 6 行（feature_slug=`GAP-DEM-CLUMP`，见
`artifacts/GAP-DEM-CLUMP-R2/oss_solutions_insert.sql` + 已入库）：
`rolling-friction-as-shape-surrogate`(adapt)、`rolling-model-assessment-4-models`(adapt)、
`elastic-plastic-rolling-spring-…`(reference-not-adopted)、
`sandpile-repose-calibration-anchor`(fits)、`horizontal-survey-empty`(reference)、
`round2-static-coulomb-plus-com-lever`(new)。

选型结论：**把"静态分支"和"形状几何力臂"两件事分开做**，因为它们是同一失败
（趋势测不出来）的两个独立候选原因：
(a) 旧律无静态分支（Ai-2011 判据）；(b) 滚动阻力以**体级各向同性**方式施加于
clump，长轴形状提供的力臂被成员半径抹掉（前轮 §5.4 的机制诊断）。

---

## 2 结合实现（①-④）

### 2.1 `src/tensordem/contact/rolling_static.py`（新增，149 行）

* **静态分支**（`model="coulomb"`）：`|τ| = min(μ_r R |Fn|, I_eff |ω_rel|/dt)`，
  方向 `-ω_rel`。钳位项不是调参：`I_eff|ω|/dt` 正是"一步内把相对自转刹停"的角冲量，
  故 `|ω| > ω_eps = μ_r R |Fn| dt / I_eff` 时**满额 Coulomb**（速率无关/塑性），
  以下则粘住。**零新调参**。
* **形状力臂**（`lever="com"`）：`R = |x_contact − x_com|`。单球严格等于成员半径
  （逐位相同 → 球基线不变），长轴 clump 则各向异性：端部接触（~a 力臂）强烈抵抗
  端翻，侧面接触（~b 力臂）弱抵抗，绕长轴滚动仍只有 ~b 力臂。
* 旧律 `model="viscous"` **逐位复现**（第 3 条测试），前轮数字保持可复现。
* 接入点：`clump_dem.py` 的 pair 与 wall 两条滚动分支（opt-in，默认仍为旧律），
  `ContactParams` 新增 `rolling_model` / `rolling_lever`。

### 2.2 验证（机器可判，`tests/test_rolling_static.py`，**13 passed**）

| id | 判据 | 实测 |
|---|---|---|
| R1a | 单球 `com` 力臂 == `member` 力臂 | 64 方向，`atol=1e-15` 通过 |
| R1b | 长轴 clump 端/侧力臂各向异性 | 端/侧 > 2.0 倍通过 |
| R2a | `|ω|=100 ω_eps` 时满额 Coulomb | 相对误差 1e-12 内 |
| R2b | `|ω|=0.1 ω_eps` 时一步刹停 | `ω_after=0`（1e-18 内） |
| R3a | 力矩恒与 `ω_rel` 反向且不超上限 | 200 随机构型通过 |
| R3b | 旧律逐位复现 | `np.array_equal` 通过 |
| R4a | 引擎级：竖直轴自转球 | 静态 **2 步**停（0.5→0.193→0），旧律 20 步后仍 6.6e-4 rad/s |
| R4b | 坡面蠕变（`I dω/dt = M_drive − M_roll`） | 静态残差 = 一步驱动增量 `M_drive dt/I`，比旧律稳态低 **11×** |
| R4c | 能量账本 | 400 步内 `E` 不增（耗散律） |

### 2.3 弹性-塑性弹簧的可行性算术（不采纳的理由，量化）

`k_r = 2.25 μ_r² k_n R² = 2.25·0.09·150·(1.2e-3)² = 4.4e-8 N·m/rad`；
上限 `M_max = μ_r R Fn ≈ 7e-8 N·m`（Fn≈mg）⇒ 屈服角 `M_max/k_r ≈ 1.6 rad`。
本基准用 `k_n=150 N/m`（软接触，真实玻璃珠约 1e5–1e6 N/m），弹性区覆盖任何
可达滚动量 ⇒ Jiang 式弹簧在此退化为**弱弹性弹簧**而非摩擦律。这是本轮不采纳
该模型、且其"不做功"原因可算的原因（记录在 `oss_solutions`）。

---

## 3 扫描结果（真数据；`artifacts/GAP-DEM-CLUMP-R2/results.json`）

协议：质量匹配填充（二分填充高度使总固体体积 = 前轮球床 4.690e-6 m³）+
静止门（平均体速 ≤ 0.01 m/s，每 300 步探针）+ μ_r=0.3 固定 + 3 种滚动配置。

| a/b | 匹配填充高度 h [m] | n_particles | 墙高 [m] | 判定 |
|---|---|---|---|---|
| 1.0 | **0.0689** | 648 | 0.06 | h > 墙高，但床顶仅在墙上方 9 mm，流动后仍能静止 → **有效** |
| 1.5 | **0.1187** | 440 | 0.06 | **床高 = 2.0×墙高 → 侧墙无法"提起清空" → 协议失效** |
| 2.0 | **0.0938** | 324 | 0.06 | **1.56×墙高 → 协议失效** |
| 3.0 | （同量级） | 216 | 0.06 | 同上 |

有效测量（`quiesced = true`）：

| 配置 | 球基线 repose [°] | 22–25° 文献带 |
|---|---|---|
| `viscous/member`（前轮旧律，静止门） | **17.47** | 低于带下沿 |
| `coulomb/member`（静态分支） | **14.86** | 低于带下沿 |
| `coulomb/com`（静态 + 形状力臂） | **15.09** | 低于带下沿 |

失效案例（`quiesced = false`，**不得引用**为测量）：a/b=1.5 9.93/14.10/19.90°，
a/b=2.0 **87.68/88.06/14.13°**（≈90° = 床体顶着未提起的侧墙，纯几何假象）。

图：`artifacts/GAP-DEM-CLUMP-R2/gap_dem_clump_r2.png`（左：三配置 repose vs a/b
+ 文献带；右：形状效应 Δ vs 3° 判据）。有效点只有 a/b=1.0，其余按 §3 失效。

---

## 4 误差判定（关键量 vs 基准，逐项）

| 量 | 基准 | 实测 | 误差 | <3%? |
|---|---|---|---|---|
| 球力臂（单球 `com` vs `member`） | 解析相等 | 差 = 0 | 0 | **是** |
| 静态分支满额限值 | `μ_r R |Fn|` | 相对 1e-12 | 1e-12 | **是** |
| 一步刹停 | `ω_after = 0` | 0（<1e-18） | 0 | **是** |
| 旧律复现 | 前轮公式 | `array_equal` | 0 | **是** |
| **球 repose vs 22–25° 带** | 22–25° | 14.86–17.47° | **−22 % … −37 %**（相对带中心 23.5°） | **否** |
| **形状趋势（C3）** | 单调升 + Δ≥3° | **不可估计**（只有 1 个有效 a/b） | — | **无法判定** |

结论：**滚动律内核层面全部达标（数值误差 ≤1e-12）；repose 物理层面球基线低于
文献带 22–37%，形状趋势不可估计 ⇒ C3 仍是 FAIL（本轮失败点已变，见 §6）。**

---

## 5 注意点（踩坑，供下一轮）

1. **"质量匹配"在旧盒子里几何不可行**：clump 的粒度/体积随 a/b 增大，等固体体积
   意味着**更多层**，填充高度 0.069→0.119 m 而侧墙只有 0.06 m。前轮的"按外接尺寸填充"
   把形状与**粒子数**耦合（648→104），本轮的"质量匹配"则把形状与**床高**耦合。
   正解：盒高/侧墙随粒子外接尺寸缩放（保持 h ≤ 0.6·wall_height），或改为
   **等粒子数 + 等床高**并用体积分数复核。这是第 3 轮的**第一执行项**。
2. **静态分支让球 repose 更低（17.47→14.86）**，不是更高。原因是无状态实现：
   每步"刹停后再受驱动力矩、留下一步增量"的极限环使堆体在 9000 步尾巴里缓慢摊平；
   即"看起来更硬"的律反而让堆摊得更开。→ 真实静摩擦必须带接触历史
   （弹性-塑性弹簧），而后者在本软接触参数下退化（§2.3）。**这是本轮最有价值的负结果。**
3. **`quiesced` 门必须硬性前置**：87.68° 这种数如果只报均值和 std，会被当成"形状效应巨大"。
   本轮 `aggregate()` 里 `quiesced=false` 的算例被**排除且单列计数**，不参与均值。
4. 前轮 `run_gap_dem_clump.py::main()` 死代码（payload 写在 `return` 之后）已在
   m14681 修复，本轮沿用修复后的入口，不做重复工作。

---

## 6 honest-FAIL 定位（本轮）

* **fail_stage** = `repose_mass_matched_fill_infeasible`
* **failure_reason** = 质量匹配把填充高度推到 0.0938–0.1187 m（> 侧墙 0.06 m），
  侧墙无法提起成堆，a/b ≥ 1.5 的全部算例 `quiesced=false` 且角度为几何假象
  （87.7°/88.1°）；仅 a/b=1.0 三配置有效，且球基线 14.86–17.47° 低于 22–25° 文献带
  22–37%，故 C3「非球形 + 形状效应」**仍未建立**。
* **已确立**：静态/形状感知滚动律能力（13 项测试，数值误差 ≤1e-12）；
  失败点从"未标"变成"协议几何"，第 3 轮有确定的重启条件。
* **被否定的假设**：①「速率比例正则化只是小瑕疵」——否，它使旧律无静态分支（Ai-2011 同判）；
  ②「补上静态分支就能恢复形状趋势」——**否**，本参数域下静态分支反而降低球基线；
  ③「Jiang 弹性-塑性弹簧可直接借用」——**否**，软接触下屈服角 1.6 rad 退化。

---

## 7 复现

```bash
# 单算例
python benchmarks/gap_dem_clump/run_round2_scan.py --case ar=1.0,seed=0,config=coulomb/com \
  --outdir artifacts/GAP-DEM-CLUMP-R2
# 全网格（集群分片，8 片）
python benchmarks/gap_dem_clump/shard_round2.py --i 0 --n 8 --outdir artifacts/GAP-DEM-CLUMP-R2
# 聚合 + 判据 + 图（quiesced=false 自动排除并计数）
python benchmarks/gap_dem_clump/run_round2_scan.py --aggregate artifacts/GAP-DEM-CLUMP-R2
# 滚动律内核
PYTHONPATH=src python -m pytest tests/test_rolling_static.py -q   # 13 passed
```

机时：单算例 57–196 s（a/b=1.0 约 58–60 s；失效的高床算例更久）。本轮在 99 上用
6 路并行跑完 a/b ∈{1.0,1.5,2.0}×3 配置（a/b=3.0 与 seed=1 网格已递交集群 cj974，
排队未回；未回的数**不引用**）。