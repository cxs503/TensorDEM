# 闭曲面船体接触的深穿透缺陷（m16872 审计 + 修复）

**【锚点=NEW:总账外新缺口 —— 已合并 GAP-DEM-HULL-SIGNED（m16799）的 `closed_surface` 路径在 t ≥ contact_r 时静默丢力，与其自身文档承诺"任意穿透深度都有恢复力"矛盾】**

仓库 `TensorDEM`｜分支 `agent/m16872`｜任务 m16872（tp-merge-1 自找活）
被审对象：`src/tensordem/hull_mesh_contact.py::mesh_contact_forces`（`closed_surface` 分支）、
合并文档 `docs/hull_mesh_contact_signed_distance.md`（m16799）、
审计脚本 `scripts/hull_signed_audit.py`、测试 `tests/test_hull_mesh_contact_signed.py`。

本单产物：
`scripts/hull_signed_deep_penetration_audit.py`（深度扫描，before/after 同源）、
`scripts/probe_signed_deep_penetration.py`（最小复现）、
`scripts/plot_m16872_deep_penetration.py`（图）、
`runs/m16872/sweep_before.json` · `sweep_after.json` · `fig_deep_penetration_compare.png` ·
`fig_deep_penetration_after.png`、`tests/test_hull_mesh_contact_deep_penetration.py`（7 例）。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

船–冰耦合里，碎冰颗粒（球体代理）与船体三角网格的接触是罚函数弹簧：
`F = k·overlap·n̂`，`overlap = contact_r - d`，`contact_r = r_particle + hull_radius`。
m16799 引入 `closed_surface=True` 的**符号距离**支路，目的是修掉两个物理错：
(a) 球心越过船体表面后被"吸进"船体（力反向）；(b) 球心穿深 `t ≥ contact_r` 后力为 0，
颗粒从仿真里丢失。

其模块 docstring 与 `docs/hull_mesh_contact_signed_distance.md` §1 明确承诺：
> "`overlap = contact_r - d_s` … it keeps a restoring force **at any penetration
> depth**" / "it keeps a restoring force at any penetration depth（`closed_surface`）"

审计几何（无调参、可解析）：

| 几何 | 尺寸 | 三角面数 | 用途 |
|---|---|---|---|
| 立方体壳（watertight） | 1 m cube | 12 | **无离散误差**，任何偏差都是代码 |
| 封头圆柱船体（capped cylinder） | R=0.5 m, L=2 m | 256 | 曲面离散（弦高）影响 |
| 球颗粒 | `r_particle = 0.05 m` | — | `contact_r = 0.05 m` |
| 罚刚度 | `k = 1e6 N/m` | — | 解析期望 `k(r_c ± t)` |
| 穿透方向 | 4 个方位角 0/47/133/311° | — | 排除单点巧合 |
| 深度序列 | `t/R ∈ {-0.5,-0.2,0,0.25,0.5,0.9,1.0,1.01,1.5,2.0,3.0,4.0}` | 48 点 | 跨越 `t = r_c` |

## ② 怎么做的（方法 / 流程 / 工具链）

1. **读代码定位**：`mesh_contact_forces` 窄相循环里 `overlap = contact_r - d_min`
   之后紧接 `if overlap <= 0: continue`，而**内外判定 `ray_parity_inside` 在其后**。
   ⇒ 只要 `d_min ≥ contact_r`（即穿深 `t ≥ contact_r`），代码在执行内外判定之前就
   `continue`，符号支路的 `overlap = contact_r + d_min` 永远得不到执行。
2. **最小复现**（`probe_signed_deep_penetration.py`，立方体壳单点扫描）：
   `t=0.049 m → +99.0 kN`，`t=0.051 m → **0.0 N**`，`t=0.3 m → 0.0 N`。
3. **量化扫描**（`hull_signed_deep_penetration_audit.py`）：把**同一份扫描脚本**
   分别用 `PYTHONPATH` 指向 pre-fix 模块（`git show HEAD:` 导出的旧文件）与
   工作树模块运行，before/after 唯一差别就是这次排序修复；输出 `sweep_before.json`
   / `sweep_after.json`。另含“深处释放颗粒”积分检验（球心初始在盒内 0.2 m 处，
   `k=1e6 N/m`，`dt=2e-5 s`，3 ms）。
4. **修复（最小改动，~20 行，只动** `closed_surface` **分支）**：
   - 把内外判定提到 `overlap <= 0` 早退**之前**：`closed_surface` 下先算奇偶性，
     `inside` 时 `overlap = contact_r + d_min`，再做早退；
   - 当 `inside` 且（候选为空 或 `overlap ≤ 0`）时，粗相 `2(r + hull_radius)` 包络
     必然不含真正最近面（深处/逃逸球心），改为**全网格最近三角面回退**
     （复杂度 O(F)，与已经调用的奇偶性扫描同阶）。
   - 外侧支路算术不变（逐位一致，`test_outside_branch_unchanged` 断言
     `closed=False` 与 `closed=True` 结果**完全相同**）。
5. 回归测试 7 例（`tests/test_hull_mesh_contact_deep_penetration.py`），并回归
   已合并的 3 个接触测试文件。

## ③ 计算结果（程序化，来自 `runs/m16872/sweep_{before,after}.json`）

封头圆柱，方位角 0°（其余 3 个方位同量级）：

| 穿深 `t` [mm] | 解析力 [kN] | pre-fix [kN] | patched [kN] | patched 相对误差 |
|---|---|---|---|---|
| −25（外侧） | 37.50 | 37.50 | 37.50 | 2.20 %（弦高，见 ⑤） |
| −1（外侧） | 51.00 | 51.00 | 51.00 | 2.19 % |
| +1 | 51.00 | 51.00 | 51.00 | 2.19 % |
| +25 | 75.00 | 75.00 | 75.00 | 0.89 % |
| +45 | 95.00 | 95.00 | 95.00 | 0.56 % |
| **+50.5** | **100.50** | **0.00（丢力）** | **100.50** | 0.56 % |
| **+75** | **125.00** | **0.00** | **125.00** | 0.56 % |
| **+150** | **200.00** | **0.00** | **200.00** | 0.56 % |

汇总（48 个圆柱扫描点 + 11 个立方体点）：

| 指标 | pre-fix | patched |
|---|---|---|
| 静默丢力点数（inside 但 F = 0） | **17 / 48** | **0 / 48** |
| 深穿透段（`t ≥ r_c`, 24 点）最大相对误差 | **100.00 %** | **0.56 %** |
| 立方体（无离散误差）深穿透最大相对误差 | **100.00 %** | **3.9e-16** |
| 外侧段（`t ≤ 0`）最大相对误差 | 2.198 % | 2.198 %（未变，无回归） |
| 浅内段（`0 < t < r_c`）最大相对误差 | 0.885 % | 0.885 % |
| 深穿透段最小外向方向余弦 `cos(F, n̂out)` | **0.0000** | **0.9988** |
| 深处释放积分检验（3 ms 后球心 z） | **0.300 m（原地不动，困在盒内）** | **0.815 m（已推出盒外 z>0.5）** |

## ④ 结果图

`runs/m16872/fig_deep_penetration_compare.png`（脚本 `scripts/plot_m16872_deep_penetration.py`）：
- 左：力–深度曲线。黑实线=解析 `k(r_c±t)`；灰叉虚线=pre-fix，在 `t=r_c` 处**断崖到 0**
  （虚线竖线处）；蓝点=patched 全程贴着解析线。
- 右：对数纵轴误差图。pre-fix 深段误差恒为 100 %；patched 圆柱深段 0.56 %
  （弦高离散）、立方体 ~1e-16；红虚线=3 % 判据、灰虚线=15 % 判据。

## ⑤ 注意点（坑 / 局限 / 适用范围）

- **为什么原审计没抓到**：`scripts/hull_signed_audit.py` 的 §3 表格里确实有一行
  "force, cylinder inside 0.25–3R (max) = 100.00 %"，但被归因为"one-sided-penalty
  limitation（罚函数固有限制）"，而没有识别出这是**排序 bug**——同一张表里的
  "outside 0.25–0.5R 好"与 "inside ≥1R 差 100 %" 的分界恰好就是 `t = r_c`，
  也就是唯一能让早退先于内外判定生效的阈值。教训：**判据跨过一个代码分支的阈值时，
  先怀疑分支而不是物理**。
- **罚函数在深穿透下力无上界**（`t=0.3 m → 350 kN`）：这是罚法的固有性质，不是本次修复
  引入的；深穿透意味着数值上已经失真（罚参数标定问题），符号距离只保证**方向正确、
  不丢力**，不保证该状态物理可信。建议下游对 `t > 2 contact_r` 记诊断量并告警。
- **性能**：`closed_surface=True` 时，包络内但未接触的颗粒现在也要做一次奇偶性判定
  （O(F)/颗粒），深度越界者再走一次全网格最近面（同样 O(F)）。这是"任意深度成立"的
  必付代价；`closed_surface=False`（历史默认）完全不受影响。
- **未覆盖**：`d_min == 0`（球心恰在面上）时法向仍取三角面绕向，而绕向未被校验，
  凹面/朝向不一致网格上可能取反（本次未测）；自相交网格、嵌套壳、非闭合网格
  仍不在支持范围（`is_watertight` 只查边界边）。
- **下游数字会变**：`docs/hull_mesh_contact_signed_distance.md` §5 的 floe 动态冲击
  与 `escape_map` 是在旧排序下算的；本单已把重算脚本
  `scripts/hull_signed_downstream_recheck.py` 递交集群（`cj1368`），结果文件
  `runs/m16872/downstream_after.json`——若未及落盘，按 honest 记为 **not_run**，
  不臆测数值。

## ⑥ 误差判定（关键量 vs 基准，显式回答 ≤3 %？）

| 关键量 | 基准 | 实测 | < 3 % ? |
|---|---|---|---|
| 深穿透（`t/r_c` = 1.01…4）罚力（立方体，无离散误差） | `k(r_c + t)` 解析 | **3.9e-16** 相对误差 | **yes** |
| 深穿透罚力（封头圆柱，256 面） | `k(r_c + t)` | **0.56 %** | **yes** |
| 浅内段（`t < r_c`）罚力（圆柱） | `k(r_c − (−t))`… `k(r_c+t)` | **0.885 %** | **yes** |
| 外侧段罚力（圆柱） | `k(r_c − |t|)` | **2.20 %**（弦高，`R(1−cos(π/64))`≈0.6 mm/50 mm；pre-fix 同值） | **yes** |
| 力方向（深穿透，48 点最小外向余弦） | 径向外法向 | **0.9988**（= faceting 角 `cos(π/64)=0.99880`） | （方向量，余弦误差 0.12 %） |
| 丢力点数 | 0 | **0/48**（pre-fix 17/48） | **yes** |
| 深处释放颗粒是否被推回 | 球心 3 ms 内越过 z=0.5 m | **0.815 m**（pre-fix 0.300 m 不动） | **yes**（定性+定量） |

**< 3 % 是否达到？达到。** 关键量（深穿透罚力）对解析式：无离散误差网格上机器精度，
曲面上 0.56 %（≤3 %），方向误差即弦高角 0.12 %。不达标项没有：丢力点从 17/48 降为 0。
诚实边界：以上是**罚力量值/方向**对解析口径的判定；深穿透状态本身不具物理可信度（见 ⑤），
本单不主张"深穿透被正确模拟"，只主张"不再静默丢力、方向不再反向"。

## 复现

```bash
PYTHONPATH=src python scripts/hull_signed_deep_penetration_audit.py \
    --label patched --out runs/m16872/sweep_after.json \
    --figure runs/m16872/fig_deep_penetration_after.png
# pre-fix 对照（旧排序）
git show HEAD:src/tensordem/hull_mesh_contact.py > /tmp/old/src/tensordem/hull_mesh_contact.py
PYTHONPATH=/tmp/old/src python scripts/hull_signed_deep_penetration_audit.py \
    --label pre-fix --out runs/m16872/sweep_before.json
PYTHONPATH=src python -m pytest tests/test_hull_mesh_contact_deep_penetration.py -q
```