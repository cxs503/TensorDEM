# 预登记 · CASE-COLUMN-COLLAPSE 到达时间的采样步长收敛（m21945）

**锚点**：`CASE-COLUMN-COLLAPSE`（算例池·颗粒柱坍塌）
**基准**：Lube, Huppert, Sparks & Freundt (2004) *JFM* **508**:175-199,
doi:10.1017/S0022112004009036
**容差**：跑距律 / 时间律 < 15 %（锚点判据），3 % 为「好」线
**本文件在任何 DEM 运行开始前写定；跑出数据后不得修改判定规则与阈值。**

## 0. 为什么是这一件（与在飞/终态单不重复）

`docs/AUDIT_COLUMN_COLLAPSE_SHALLOW_CONVERGENCE_m17917.md` 的「给下一次自找活的建议线索」
第 2 条、以及 `docs/CASE_COLUMN_COLLAPSE_SYNTHESIS_m20978.md` §3.2 的**诚实失败项**：

> 时间律在 $a=0.5$ 上不收敛（$t_{90}$ 误差 $+39.5\%\to+12.6\%\to+19.8\%$，随分辨率非单调）。

m17534 已证明「到达定义」能把判决移动 21 个百分点；m18261 已把三种定义族
（`fraction` / `velocity` / `length`）实装进 `tensordem.granular_column.arrival`。
**但从未有人检验：这些到达时间本身对历史采样步长（`sample_stride`）是否稳定。**
既有全部归档算例以 `sample_stride = 5` 或 `20` 存储历史（`dt≈4e-4 s`，即采样间隔
2–8 ms），而定义族的移动平均窗 `DEFAULT_WINDOW_S = 10 ms` 与之同量级——
**采样步长与平滑窗是耦合的**，一个「定义诱导的判决翻转」可能只是采样伪迹。

已覆盖、本单**不重跑**的轴（避免自我重复）：长宽比 $a\in\{0.5,1,1.5,1.6,1.7,1.8,2,3\}$
（m18261/m20337/m20873/m21007）、分辨率阶梯 4–16 粒/$r_i$（m20337/m21656）、
$\phi_0$ 轴（m21723）、无序/系综（m20811/m19922/m20259）、$\mu$ 扫描与 `required_mu` 反解
（m16534）、能量闭合与 $\Delta t$ 收缩（m21164）、断点邻域（m21007）。

本单唯一新增自由度：**历史采样步长 `sample_stride`（离线抽取）**，以及它同
**到达定义族** 的交互。这是纯「验证欠账」，不新增物理。

## 1. 主假设（跑前写定，可被证伪）

- **H1（采样伪迹假设）**：$a=0.5$ 处 $t_{90}$ 随分辨率的非单调差异（$+39.5/ +12.6/ +19.8\%$）
  在密集采样（`stride=1`）下重现，即「把采样步长从 20 步一次压到 1 步一次，
  $t_{90}$ 的变化 < 3 个百分点」。
  - 判定：对每个算例，$|t_{90}^{\text{stride}=1} - t_{90}^{\text{stride}=20}| / t_{\text{ref}} < 3\%$ ⇒ H1 成立（采样伪迹被排除）。
- **H2（定义诱导假设）**：同一算例内 13 个预登记定义之间的 $t$ 散布
  $\max-\min$ 相对参考 $t_\infty$ **> 15 %**，即「时间律判决由定义选择支配，而不是由物理支配」。
- **H3（排序恢复假设）**：以 `stride=1` 数据重算后，$a=0.5$ 三个分辨率档的 $t_{90}$ 误差
  仍**非单调**（不因采样加密而恢复单调）。H3 为真 ⇒ 「浅柱时间律不收敛」是一条
  与采样无关的结论，可以写进锚点账。
- **H4（估计量不变式，纯解析、零成本）**：对单调前缘，抽稀到 `stride=k` 得到的到达时间
  $t_k$ 必满足 $t_1 \le t_k \le t_1 + k\,\Delta t$。任何违反都是实现缺陷（测试断言）。

阈值冻结：`SAMPLING_TOL = 0.03`、`DEFINITION_BAND_TOL = 0.15`（=锚点容差）、
`PRIMARY = fraction_0.9`（锚点历轮主判据）。

## 2. 算例矩阵（5 个，跑前冻结）

冻结闭合臂沿用 m18261/m20873/m21007，**只改 `sample_stride`（5 → 1）**：
$\mu=0.5,\ \mu_r=0.5,\ \mu_w=0.8,\ e=0.5,\ k_t/k_n=0.5,\ \rho=2500,\ g=9.81$,
`packing_gap=0.02`, `jitter=0`, `seed=12345`, `duration_factor=8.0`, `min_time=0.5`,
`rest_ke_tol=1e-4`, `rest_hold=200`, `contact_mode="reference"`,
$r_i = 0.48$ m 固定、$r = r_i/(2\,\mathrm{gpr})$、$k_n = 6\times10^4 (r/0.03)^2$。

| case_id | $a$ | gpr | 作用 |
|---|---|---|---|
| `AS_a0.5_g8` | 0.5 | 8 | 浅端基线档（$+39.5\%$ 那一点） |
| `AS_a0.5_g12` | 0.5 | 12 | 浅端中档（$+12.6\%$） |
| `AS_a0.5_g16` | 0.5 | 16 | 浅端细档（$+19.8\%$），H3 需要三档 |
| `AS_a1_g8` | 1.0 | 8 | 线性支参照（$t$ 误差历史为 −2.3 %，本单检验它是否被采样支配） |
| `AS_a3_g8` | 3.0 | 8 | 平方根支参照（深端） |

采样步长扫描（离线，零额外算力）：`stride ∈ {1, 2, 5, 10, 20, 40}`。
`stride=1` 为密集基准，其余由同一份密集历史抽稀得到（**同一轨迹**，无新物理）。

## 3. 判据（跑前写定）

1. **H4 不变式**（pytest）：抽稀不变式逐定义、逐 stride 断言，违反数必须为 0。
2. **H1**：`|t(i, stride=20) − t(i, stride=1)| / t_ref(fraction_0.9) < 3 %`（每个算例各自判）。
3. **H2**：每算例的 13 定义散布 `spread_rel > 15 %` ⇒ 定义支配。
4. **H3**：$a=0.5$ 三档在 `stride=1` 下的 $t_{90}$ 误差**非单调**（存在中间档低于两侧）。
5. **诚实申报**：若某 stride 因 `ArrivalError`（前缘回退超单调容差、永不达阈）无法给出定义，
   记 `nan` 并计入 `failed`，不内插、不外推。

## 4. 交付

- 模块 `src/tensordem/granular_column/arrival_sampling.py`（抽稀 / 定义栅格 / 不变式 / 散布 / 判定）
- 测试 `tests/test_column_arrival_sampling_m21945.py`（闭式解析，含反例）
- 算例 `benchmarks/case_column_collapse_arrival_sampling_m21945.py`（`--shard i/n`、`--collect`）
- 归约 `scripts/analyze_column_arrival_sampling_m21945.py`（CSV + JSON + 图）
- 报告 `docs/CASE_COLUMN_COLLAPSE_ARRIVAL_SAMPLING_m21945.md`