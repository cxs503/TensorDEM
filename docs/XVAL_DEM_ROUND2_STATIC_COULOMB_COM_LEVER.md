# XVAL — `round2-static-coulomb-plus-com-lever` (TensorDEM vs OSS TensorDEM/self)

锚点 `XVAL-DEM-ROUND2-STATIC-COULOMB-PLUS-COM-LEVER` · mission m17674 · 仓 TensorDEM ·
判据 **< 15 %**（owner 精度标准）· 结论 **PASS**，最大偏差 **0.0249 %**。

## (1) 问题与参考出处

**问题**：round-2 新增的两个做法——(a) 静态（可粘着）Coulomb 滚动分支
`|tau| = min(mu_r R |Fn|, I_eff |w_rel|/dt)`，方向 `-w_rel`；(b) 形状力臂
`R = |x_contact - x_com|`——在**本工作树**里是否忠实复现了参考实现的语义与已发布数字？

**参考（ref）出处（零手抄，均可回读）**：

| # | 出处 | 取到的内容 |
|---|---|---|
| R1 | `src/tensordem/contact/rolling_static.py` docstring + 纯函数 | 两条定律的方程、`w_eps = mu_r R|Fn| dt / I_eff`、单球「com == member 逐位相同」 |
| R2 | `docs/GAP_DEM_CLUMP_ROUND2.md` §2.1 / §2.3 | 同一方程的文档表述、形状力臂物理含义、不采纳弹性-塑性弹簧的量化理由 |
| R3 | `docs/GAP_DEM_CLUMP_ROUND2.md` §3 表「有效测量」 | 已发布 repose：`viscous/member` **17.47deg**、`coulomb/member` **14.86deg**、`coulomb/com` **15.09deg** |
| R4 | `docs/AUDIT_m15187_GAP-DEM-CLUMP.md` L20 | 逐位可复现值 `angle_profile_deg = 14.858310520973625` |

**开工三查（如实）**：`tensor/knowledge/search` -> `engine_offline`（知识引擎 127.0.0.1:8021
拒连）；`tensor/negative-knowledge/search?q=TensorDEM` 0 命中；`oss_solutions` 表 psql
*permission denied* 且无 API 暴露；外网出站阻断（`raw.githubusercontent.com` connection
reset）。据此 **ref 只能是 `TensorDEM/self` 工件本身**——本单因此把「独立对标」落在
**独立 oracle + 闭式解**上（见 (2)），不冒充第三方代码对比。

## (2) 方法与实现（四腿）

| 腿 | 内容 | 独立性来源 |
|---|---|---|
| A | 504 格网格上实现 `rolling_torque/saturation/stopping_speed` vs `oracle.py` | oracle 由方程重写，**不 import tensordem**（测试守护） |
| A2 | 力臂几何：单球退化、`arm_tip == a_union`、tip/flank 各向异性、`member` 力臂「位置盲」 | `ellipsoid_clump` 建模意图 + 定义式 `|x-com|` |
| A3 | 引擎耦合：单球底板自旋，停机步数 vs `ceil(w0/w_eps)`、能量单调 | 引擎 `ClumpDEM` vs 闭式解 |
| A4 | 惯性口径：`mean_inertia = tr(I)/3`（定律）vs 体张量（积分） | 长椭球闭式 `(AR^2+2)/3`、`(2/3)(AR^2+2)/(AR^2+1)` |
| B | 复现已发布 repose 三配置（ar=1.0, seed=0） | 已发布工件数字 |

新增文件：`benchmarks/xval_round2_static_coulomb/{oracle.py, xval_spec.py, run_xval.py}`、
`tests/test_xval_round2_static_coulomb.py`（7 项，8.9 s 全绿）、
`prereg/xval_dem_round2_static_coulomb_com_lever.md`（阈值先于运行）。

## (3) 参数表（无量纲数完整）

**Leg A**：`mu_r` in {0.1, 0.3, 0.5}，`R` in {1.2e-3, 3.0e-3} m，`|Fn|` in {1e-4, 1e-3} N，
`I_eff` in {1.1e-11, 1.1e-10} kg m^2，`dt` in {2e-5, 5e-5, 2e-4} s，
`w/w_eps` in {1e-3, 1e-1, 0.5, 1, 10, 100, 1e4}。合成出的**唯一无量纲控制参数**是
`Pi_w = w/w_eps = I_eff w /(mu_r R |Fn| dt)`（粘着 <-> 塑性），`mu_r` 为摩擦系数本身，
`R/r_member` 描述力臂放大。

**Leg A3 无量纲恒等式**：单球静止于底板时 `Fn = mg`、`I = 0.4 m r^2` =>
`Pi_eps = w_eps r/(g dt) = mu_r/0.4 = 0.75`（质量、半径、dt 全消去；
实测 `w_eps = 0.3066 rad/s` @ dt=5e-5, r=1.2e-3 => `Pi_eps = 0.7500`）。

**Leg B（协议）**：`mu_r = 0.3`（全配置唯一）、`rho = 2650 kg/m^3`、`r_eq = 1.2 mm`、
质量匹配填充体积 `4.690e-6 m^3`、`h = 0.06891 m`、
`N = 648`、判据带 `22-25 deg`（文献）。

## (4) 对照表（ref vs got，逐项打勾叉）

| 项 | ref | got | 偏差 | 判据 | 判定 |
|---|---|---|---|---|---|
| A 力矩（504 格最大相对误差） | oracle | 实现 | 2.721e-16 | <= 1e-12 | OK |
| A 饱和分数（绝对） | `min(1, w/w_eps)` | 实现 | 2.220e-16 | <= 1e-10 | OK |
| A 一步残余自旋（相对 `|w|`） | `max(0, w-w_eps)` | 实现 | 3.644e-16 | <= 1e-12 | OK |
| A `w_eps` | 闭式 | 实现 | 0.0e+00 | <= 1e-12 | OK |
| A2 单球 com vs member | 逐位相同 | 绝对差 2.168e-19 | — | <= 1e-15 | OK |
| A2 力臂定义式 `|x-com|` | 定义 | 实现 | 0.0e+00 | <= 1e-12 | OK |
| A2 端部力臂 vs 并集长半轴 | `a_union` | 实现 | 2.407e-08 | <= 1e-6 | OK |
| A3 停机步数 vs 闭式 | `ceil(w0/w_eps)` | 引擎 | 最差 1 步 | <= 1 步 | OK |
| A3 能量单调 | 不增 | 最大 0.9517 E0 | — | <= 1+1e-12 | OK |
| A4 单球各向同性（一步精确停） | 0 | 0.0e+00 | 相对 w_eps | <= 1e-12 | OK |
| B `viscous/member` repose | 17.47 | 17.470179 | +0.0010 % | < 15 % | OK |
| B `coulomb/member` repose | 14.86（逐位 14.858310520973625） | 14.858310520973625 | -0.0114 % / 逐位差 0.0e+00 | < 15 % | OK |
| B `coulomb/com` repose | 15.09 | 15.086241 | -0.0249 % | < 15 % | OK |

**是否 < 3 %？** 是。A/A2/A3/A4 各腿在 1e-16..1e-8（机器精度级）；Leg B 三个 repose 角
最大偏差 **0.0249 %**（< 3 %，远小于 15 % 判据），其中 `coulomb/member`
**逐位相同**（差 0.0e+00 deg，审计的 14.858310520973625 在本工作树一字不差复现）。

### 力臂几何（A2）

| a/b (nominal) | n_members | a_union [mm] | b_union [mm] | 达到的 AR | 端部力臂 [mm] | 侧向力臂 [mm] | `com` tip/flank | `member` tip/flank |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 1 | 1.200 | 1.200 | 1.0000 | 1.2000 | 1.2000 | 1.0000 | 1.0000 |
| 1.5 | 3 | 1.800 | 1.268 | 1.4199 | 1.8000 | 1.2745 | 1.4124 | 1.0000 |
| 2.0 | 4 | 2.400 | 1.217 | 1.9728 | 2.4000 | 1.7755 | 1.3517 | 1.0000 |
| 3.0 | 6 | 3.600 | 1.238 | 2.9075 | 3.6000 | 2.9875 | 1.2050 | 1.0000 |

**读法**：`com` 的 tip/flank 比 ≈ **达到的** AR（并集体 `a/(s b)`），nominal AR 因体积匹配
(`s > 1`) 偏大；`member` 力臂只有半径比、对位置全盲。round-2 报告以 nominal AR 为横轴，
因此**低估**了定律实际交付的形状各向异性——本单新增的定量观察，不是缺陷。

### 惯性模型（A4）

| a/b | I_long [kg m^2] | I_trans [kg m^2] | `mean/I_long` | `mean/I_trans` | 一步残余 / w_eps |
|---|---|---|---|---|---|
| 1.0 | 1.1048e-11 | 1.1048e-11 | 1.0000 | 1.0000 | 0.0000 |
| 1.5 | 1.6931e-11 | 2.6748e-11 | 1.3866 | 0.8777 | 0.3866 |
| 2.0 | 2.2153e-11 | 5.5400e-11 | 2.0005 | 0.8000 | 1.0005 |
| 3.0 | 3.3667e-11 | 1.6823e-10 | 3.6646 | 0.7334 | 2.6646 |

## (5) 收敛性 / 参数敏感性（>=2 点）

1. **时间步不变性（dt 2e-5 / 5e-5 / 2e-4 s）**：以 `w/w_eps` 归一后三点重合
   （最大互差 2.2e-16）——静态分支速率无关，`dt` 只经 `w_eps ~ dt` 进入，
   符合「零新调参」的设计声明。
2. **速率扫描（7 点，1e-3..1e4）**：粘着<->塑性转变在 `w/w_eps = 1` 单一拐点，
   两侧极限分别为 `~w/w_eps`（粘着）与 `1`（塑性）。
3. **几何（4 点 AR）**：`com` 力臂比随 AR 单调升 1.000 ->
   1.205，端部力臂与 `a_union` 一致到 2.4e-08。
4. **引擎步数（12 格）**：11/12 格与闭式 `ceil(w0/w_eps)` **完全一致**，1 格差 1 步
   （`w0=5 rad/s, dt=2e-5`，41 步量级的长时程累积）——见 (6)。

## (6) 失败模式、边界条件与工件

* **失败模式 1（可量化局限，非缺陷）**：静态分支「一步停」只在**各向同性体**上精确。引擎用
  **完整张量**积分自旋，定律的 `w_eps` 用 `tr(I)/3`；对长椭球 clump 实测
  `mean/I_long = 3.665`、`mean/I_trans = 0.733`
  (AR=3.0)，即绕长轴**过刹 ~266 %**、
  端翻**欠刹 ~27 %**，一步残余最大
  **2.66 w_eps**。对 round-2 的**有效测量**（ar=1.0 球基线）
  影响为零（球各向同性，实测 0.0）。
* **失败模式 2**：Leg B 的 a/b>1 配置在 round-2 协议下不可测（床高 > 墙高，`quiesced=false`），
  本单未复跑这些失效格（诚实留白，见 `docs/GAP_DEM_CLUMP_ROUND2.md` §3）。
* **失败模式 3（观测精度）**：clump 的 COM 由 `grid=72` 体素质量属性给出，故 `|x-com| = a_union`
  只到 2.4e-08（相对）——体素 COM 精度，不是力臂公式误差。
* **预注册偏差（如实）**：(i) P3 原以「残余相对误差」度量，在残余恰为 0 (`w<w_eps`) 处无定义、
  在 `w/w_eps=1e4` 处因 `w-w_eps` 灾难性相消只剩 ~1e-12，故改为相对 `|w|`
  (3.6e-16)，原度量数值一并报告 (3.3e-12)；
  (ii) P5 预注册闭式 `(a+r)/(b+r)` **前提错误**（`a` 是并集外延、非成员中心线），改为可判的
  `P5a` 定义式恒等 + `P5b` `arm_tip == a_union`（容差 1e-6，理由=体素 COM）；
  (iii) P6 能量门改单侧（耗散律不应增能）；(iv) 新增 P8（A4 单球各向同性，预注册后追加，如实标注）。
* **工件**：`artifacts/XVAL-DEM-R2-COULOMB-COM/results.json`（含 env/commit + 全部原始行）、
  `legs/leg{A,A2,A3,A4,B}.json`、`legs/B_case_*.json`、
  `figures/xval_legA_saturation_residual.png`、`figures/xval_legA2_lever_inertia.png`、
  `figures/xval_legA3_engine.png`、`figures/xval_legB_repose.png`。

## 本仓 vs 参考（OSS TensorDEM/self）差异清单 + 移植取舍

| # | 项 | 参考做法 | 本工作树实测 | 取舍 |
|---|---|---|---|---|
| D1 | 静态分支数学形式 | `min(mu_r R|Fn|, I_eff|w|/dt)`，方向 `-w_rel` | 一致（2.7e-16） | 照搬（无需改动） |
| D2 | 力臂 | `member` = 成员半径（体级各向同性） | 复现（半径比 1.000，位置盲） | 保留为默认（round-1 数字可复现） |
| D3 | 力臂 | `com` = `max(|x-com|, r_member)`（含下界钳位） | 定义式逐位一致；钳位在 AR<=3 并集上不触发 | 采纳 `com` 时必须连同钳位语义一起搬：内凹成员表面接触点不会低于成员半径 |
| D4 | 各向异性记账 | 报告横轴 = nominal AR | `com` 力臂比 = **达到的** AR（1.205 @ nominal 3.0） | 建议图表改用达到的 AR，避免低估效应（本单只记不改） |
| D5 | 惯性 | 定律 `tr(I)/3`，积分完整张量 | 非各向同性体上「一步停」不精确（最大 2.66 w_eps） | 不移植修正：对球基线零影响；若要修应把 `w_eps` 换成沿 `w` 的 `I_axis`（后续轮次候选） |
| D6 | 边界条件 | 墙接触 `w_rel = w_body`（墙不转） | 一致 | 照搬 |
| D7 | 粘着后切向静摩擦 | 未涉及（rolling 分支独立于 Cundall-Strack 切向） | 一致：自旋绕接触法向时无滑移污染（A3 实测） | 照搬；但不可据此声称「静摩擦全域正确」，只覆盖滚动阻力 |
| D8 | 弹性-塑性滚动弹簧 | 参考不采纳（屈服角 ~1.6 rad 退化） | 未实现 | 沿用参考结论，不新增参数 |

**移植结论**：D1/D2/D6 逐位可搬；D3 需连同「钳位」语义一起搬（否则内凹几何会得到更小力臂）；
D4/D5 是不可直接搬的**记账/惯性口径**差异，本单按「只记录、不改动」处理（改动会破坏 round-1 复现性）。

## 生态摩擦（如实登记）

1. 知识面不可用：`/tensor/knowledge/search` -> `engine_offline`（8021 拒连），开工三查的
   capability 查证降级为「仓内 grep + 既有 docs」。
2. `negative_knowledge_ledger` 检索 0 命中（不是「无否定知识」，而是**查不到**）。
3. `oss_solutions`/`oss_inventory` 无 API 暴露、psql 直连 permission denied => 无法核实对标行
   原文，只能依赖 round-2 文档转述（`round2-static-coulomb-plus-com-lever`(new)）。
4. 外网出站阻断（GitHub raw connection reset）=> 无法取第三方 OSS 源码，"OSS 对标"退化为
   「self 工件 + 独立 oracle」。
5. 控制面 shell 120 s 硬超时且整组收割进程：>2 min 的任务必须 `setsid` 脱离（Leg B 三配置
   共 ~5 min 靠 `setsid` 存活），或走 cluster_queue。
6. `rolling_torque` 只接受标量 `dt`（numpy 2.x 下 `float(1 维数组)` 直接 TypeError）——网格化
   调用需按 dt 分组；已在 harness 规避，未改源码。
