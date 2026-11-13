# CASE-PENETRATION R4 — 几何审计：随仓七个预设的底板间隙都不到 3 R

**锚点** CASE-PENETRATION-R4 ｜ mission 18243（重开 #1418，管到底第 2 轮）｜
状态 **PARTIAL（工具+几何审计完成；DEM 阶梯未收敛）**

> **改动边界声明**：本轮**未改动** `src/tensordem/penetration_quasistatic.py`、
> `docs/CASE_PENETRATION_QUASI_STATIC.md`、`src/tensordem/sphere_pack_dem.py`、
> `tests/test_penetration_quasistatic.py` —— 这四个文件正被开分支 `agent/m18201`
> （CASE-PENETRATION R3）修改，重复编辑只会制造冲突。本轮全部新增落在
> `scripts/penetration_deep_bed.py`、`scripts/penetration_deep_bed_report.py`、
> `tests/test_penetration_deep_bed.py`、本文件与 `artifacts/case_penetration/deep_bed/`。

## ① 案例介绍

刚体球准静态垂直贯入无黏性颗粒床，测深度相关阻力 F(z)，与 Brzinski III, Mayor &
Durian, PRL **111**, 168002 (2013) Eq.(2)/(3c)/(3d) 对照，标定摩擦放大系数 alpha
（文献 35 ± 5，p.1）。

m15404 交付（已在 main：`src/tensordem/penetration_quasistatic.py` 等 4 文件，
PR #1299 合并）报告 **alpha_fit = 349, 相对误差 898%，honest FAIL**，并把 10 倍偏差
归因于粗粒化（R/d = 2 vs 文献 ~42）。

本轮重读**几何**后提出更靠前的解释：Eq.(3c)/(3d) 的前提是**半无限床**，而随仓所有
预设的刚性底板都离最深贯入点只有 0.9–1.9 R（两三颗颗粒厚），贯入器压的是
“贯入器—颗粒层—底板”的受限压缩层，不是重力主导的半无限阻力。这不需要新的物理，
只需要把几何算清楚——所以本轮先做**几何审计**，再做**加性/乘性判别工具**。

## ② 怎么做的

1. **实测几何记账**（`scripts/penetration_deep_bed.py::protocol_audit`）：
   `floor_clearance_over_r = (z_surface − z_max)/R`，`z_surface` 取沉降后实测自由表面，
   `z_max` 取实测最大贯入深度；半无限床判据取 **3 R**。同时记录 R/d、
   `v/√(gR)`、盒半宽/R，越界即写入 `warnings`（不是静默通过）。
2. **静态审计**（`... audit`，纯几何、无仿真）：对随仓 7 个预设按名义填充高度算
   底板间隙，得到**乐观上界**（忽略沉降与贴底颗粒半径）。
3. **加性 vs 乘性判别**（`... ::line_diagnostic` / `window_ladder` /
   `additive_verdict`）：深支（z ≥ R）上 Eq.(3d) 本身是直线
   `F = alpha μ ρ g π R² (z − R/3)`，于是**斜率单独定 alpha（对常数托力免疫）**，
   截距与理论值 `−s R/3` 的差
   `tare_N = b + s R/3`
   就是深度无关的附加力（底板托力 / 台架零点 / 去皮误差）。逐窗给出
   `alpha`（模块约束一参数拟合）、`alpha_free`（斜率估计）、`tare_N`，再判定
   超额 alpha 是**加性**（斜率对、托力多）还是**乘性**（斜率本身错 → 粗粒化/μ 标定）。
   这比“alpha·S(z)+c 联合拟合”稳得多：后者两列在短窗上近共线，噪声下会给出负 alpha。
4. **外推**（`fit_clearance_limit`）：对半无限床样本拟合
   `alpha(c) = alpha_inf + k/c`，报截距与标准误，验收判据落在 alpha_inf 上。

工具链：`src/tensordem/penetration_quasistatic.py`（**只调用、未改**）、
`scripts/penetration_deep_bed.py`（审计/配置构造/阶梯/判别/`reanalyze`）、
`scripts/penetration_deep_bed_report.py`（记录 → 表 + 图，程序化注入）、
`tests/test_penetration_deep_bed.py`（10 通过 / 1 跳过）。

## ③ 计算结果

### 3.1 静态几何审计（`artifacts/case_penetration/deep_bed/preset_floor_audit.json`）

| 预设 | R/d | 床深/d | 底板间隙/R（名义上界） | 半无限床前提（≥3 R） |
|---|---|---|---|---|
| base | 3.0 | 8.0 | 1.07 | ✗ |
| rd2 | 2.0 | 7.0 | 1.90 | ✗ |
| rd4 | 4.0 | 10.0 | 0.90 | ✗ |
| slow | 3.0 | 8.0 | 1.07 | ✗ |
| rate2x | 3.0 | 8.0 | 1.07 | ✗ |
| damp2000 | 3.0 | 8.0 | 1.07 | ✗ |
| dashpot | 3.0 | 8.0 | 1.07 | ✗ |

**7/7 预设违反半无限床前提**，间隙 0.90–1.90 R。也就是说 m15404 的 alpha=349
不是在测 Eq.(3c)/(3d)，而是在测受限压缩层的等效刚度——把它归因于粗粒化之前，
必须先把这个几何误差排除。

### 3.2 判别工具的自证（合成数据，`pytest tests/test_penetration_deep_bed.py`）

| 合成输入 | 判据 | 结果 |
|---|---|---|
| 纯 Brzinski 律（alpha=35） | `alpha_free` 误差 | < 1e-9（精确） |
| 纯 Brzinski 律 | `tare_N` | < 1e-9 N（精确） |
| 纯律 + 0.5 N 常数托力 | `tare_N` | 0.500000 N，`alpha_free` 仍 = 35.0 |
| 纯律 + 0.5 N 常数托力 | 判定 | `additive`（斜率对、只多托力） |
| alpha=105（乘性错 3 倍） | 判定 | `multiplicative`，`tare_N` ≈ 0 |
| alpha(c)=35+60/c, c=1,2,3,4.5 | `alpha_inf` | 35.000000，斜率 60.000000，标准误 < 1e-6 |

（“是否 <3%”：上表合成自证全部为精确恢复；但这是**工具**的精度，不是物理算例的精度。）

### 3.3 DEM 阶梯：未收敛（honest）

阶梯命令已固化并可复现：

```
python scripts/penetration_deep_bed.py run --clearance 4.5 --rd 2 --wall 3 \
    --d 0.004 --speed 0.05 --sample-every 20 --out artifacts/case_penetration/deep_bed/clr4p5_v0p05
```

实测成本（本机，纯 numpy 内核）：N≈1.5e3 → 4.9 ms/步；N≈2.0e3 → 8–40 ms/步
（随机器负载波动 5 倍）；名义 base 预设（N≈5.3e3）单例需 ≈3e5 步，≥1 小时/例。
四~五点阶梯的实机成本 3–6 小时，本轮窗口内**未跑完**：
清理底板到 4.5 R 的两例（`clr4p5_v0p10`）与另外三例已递交集群队列
（cj1865/cj1866/cj1869/cj1873），提交后队列积压 320+ 作业，至本轮收尾仍在排队。
故本轮**不给出 alpha_inf、不声明 15% 验收通过或失败**——这需要阶梯跑完，
`... report` 会用 `fit_clearance_limit` 自动给出 alpha_inf ± 标准误与判定。

## ④ 结果图

![fig1](fig1_preset_floor_clearance.png)

图 1：随仓 7 个预设的底板间隙（名义上界）与 3 R 半无限床判据；全部低于判据，
最近的 rd2 也只有 1.9 R。数据直接来自 `preset_floor_audit.json`。

DEM 力—深曲线与 alpha–间隙曲线此刻没有可画的记录（阶梯未完成，见③.3）——
**不是省略，是没有数据**，本轮不以合成/部分数据充数。

## ⑤ 注意点 / 踩坑

- **薄床 ≠ 粗粒化**：m15404 把 10 倍偏差归因于 R/d，会诱导后续轮次去堆 R/d（成本按
  R/d 立方增长），而真正的自变量是底板间隙；先算几何，再谈物理。
- **预设名与物理不符**（main 现状，未在本轮修改）：`slow` = 0.25 m/s 是 base 的
  **5 倍**，`rate2x` = 1.0 m/s 是 **20 倍**；用这两个预设做“倍速对照”会伪造出
  “准静态已验证”的结论。已在开分支 R3 修复，本轮只上报不重复改。
- **联合拟合不可用**：`F = alpha S(z) + c` 在短窗上两列近共线，本轮早期实现给出
  过 `alpha_joint = −2.4e3` 的噪声解；改成“斜率定 alpha、截距定托力”后合成自证精确。
- **机器负载主导成本估计**：同一配置的每步耗时在 4.9 ms→63 ms 间波动 13 倍，
  任何“这算例很快/很慢”的判断必须同时报负载。
- **未触碰热点文件**：本轮刻意不改进 `penetration_quasistatic.py` 与
  `sphere_pack_dem.py`（R3 分支在改），否则两轮改动互相覆盖。

## ⑥ 误差判定

| 关键量 | 基准 | 本轮值 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 预设底板间隙（几何量） | 名义上界精确可算 | 0.90–1.90 R | 0（解析量，非拟合） | ✓（几何量） |
| 判别工具 alpha 恢复（合成） | 35 | 35.000000 | < 1e-9 | ✓ |
| 判别工具托力恢复（合成） | 0.5 N | 0.500000 N | < 1e-9 | ✓ |
| **alpha vs 文献 35（本算例）** | 35 ± 5 | **未测**（阶梯未收敛） | — | **不声明** |

结论：几何审计这一条是**硬结论**（7/7 预设违反前提，纯解析可复算）；物理算例的
alpha 结论**未完成**，本轮不做任何“已接近文献”或“已排除粗粒化”的断言。

## ⑦ 接力清单（下一轮直接可跑）

1. 等集群队列消化后跑完 4 点阶梯（clearance = 1.0/2.0/3.0/4.5 R，v=0.05，
   R/d=2，wall=4 R），`... reanalyze --dir artifacts/case_penetration/deep_bed` 后
   `... report --dir ... --doc docs/...` 自动出表、出图、出 alpha_inf 与验收判定。
2. 速度阶梯（v/√(gR) = 0.09/0.15/0.30）确认 alpha 与速率无关（本轮的对比只能固定
   v，不能证明准静态）。
3. 壁面 Janssen 对照：盒半宽 3/4/6 R 的 alpha 漂移（成本高，建议集群分片）。
4. mu_eff 标定：安息角把 DEM 颗粒摩擦与文献介质内摩擦对齐。

## 附：本轮改动文件

- `scripts/penetration_deep_bed.py`（新增，几何审计 + 阶梯 + 判别 + 外推 + reanalyze）
- `scripts/penetration_deep_bed_report.py`（新增，记录 → 文档/图，程序化注入）
- `tests/test_penetration_deep_bed.py`（新增，10 通过 / 1 跳过）
- `docs/CASE_PENETRATION_FLOOR_CLEARANCE_M18243.md`（本文件）
- `artifacts/case_penetration/deep_bed/preset_floor_audit.json`（审计记录）
- `artifacts/case_penetration/deep_bed/fig1_preset_floor_clearance.png`（图 1）