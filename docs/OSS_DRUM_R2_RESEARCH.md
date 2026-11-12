# CASE-ROTATING-DRUM 第 2 轮：调研 · 方法谱系 · 选型沉淀（m15649）

**锚点** `CASE-ROTATING-DRUM` · 仓库 TensorDEM · 前轮 m15478（返修轮）honest-FAIL、
再前轮 m15329 交付被判 L2（`tensor_reports` #14512：休止闸门 3/3 FAIL、流态"一致"无测量对象、
`mode=mixed` 却勾方向、S(rev) 无平台、工件未入库）。本单先把成熟方法调研清楚，再按判据推进。

---

## ① 调研 · 开源横向（`oss_inventory` + `~/oss-cache`）

**查询（可复现，零手抄）**：

```sql
SELECT repo, tier, status, left(focus,120), left(coalesce(methods,''),300)
FROM oss_inventory
WHERE focus ILIKE '%drum%' OR focus ILIKE '%rotary%' OR repo ILIKE '%drum%'
   OR coalesce(methods,'') ILIKE '%drum%' OR coalesce(summary,'') ILIKE '%drum%'
ORDER BY tier;
```

**结果：0 行**。盘点库当前 33 条全部是 LBM 系（`openLB/openLB`、`lssfau/lbmpy`、
`Pranav-Laks/…` 等，见 `SELECT repo,tier,status,focus FROM oss_inventory ORDER BY tier,repo`），
**没有任何 DEM/滚筒类实现**（LIGGGHTS / YADE / MFiX / LAMMPS granular 均未入盘）。

**集群 `~/oss-cache`**：本地镜像 `/root/oss-cache` 只有 3 个 LBM 仓
（`Maarten-vd-Sande_lbm`、`hschwane`、`pylbm`）；集群节点盘点作业 `cj1164`
（`ls ~/oss-cache; grep -ril "drum" …`）递交后长时间处于 `queued`（队列 353 单在前），
**本单未取回结果** —— 如实记为生态摩擦，不冒充"已横向调研源码"。

**结论（横向结论，非结论性空白）**：开源横向上本生态**无同域可抄结构**；本算例的
"学现成方法"只能走文献/教科书谱系（下节）+ 本仓已验证内核（`drum_segregation.kernel2d`，
p176w2-n08 验证面）。这本身是一条可复用知识：**CASE-ROTATING-DRUM 没有 OSS 参照物，
任何"对标源码"的说法都不成立**。

---

## ② 调研 · 方法谱系（文献/教科书标准做法 + 标准算例）

| # | 方法/量 | 出处 | 精确性 | 在本仓的落点 |
|---|---|---|---|---|
| M1 | 滚筒相似参数 `Fr = ω²R/g` | Haff 1983, *JFM* **132**:401–430 | 定义 | `drum_flow.regimes.froude` |
| M2 | 释放角 `θ_d = arccos(−Fr)`（6 点钟起算） | 由 M1 的径向力平衡 `N = m(ω²R + g cosθ) = 0` 直接解出（同一文献式；本仓 docstring 同式） | **精确闭式** | `drum_flow.regimes.detachment_angle_deg` |
| M3 | 离心化起点 `Fr = 1` | M2 在 θ=180° 的根：顶部法向力消失 | **精确** | `prereg/CASE_ROTATING_DRUM_R2.json` |
| M4 | 运动形态图（sliding/surging/rolling/cascading/cataracting/centrifuging） | Mellmann 2001, *Powder Technology* **118**:251–270 | 仅两端精确，中间带为量级分界 | 本轮新增 `drum_flow.regimes.MELLMANN_BANDS`，**只用于定性标签** |
| M5 | 软球 DEM 接触律：法向弹簧-阻尼 + Cundall-Strack 切向 + 滚动阻力 | Cundall & Strack 1979, *Géotechnique* **29**:47–65 | 数值方法 | `drum_segregation.kernel2d._contact_forces` |
| M6 | 双分散滚筒巴西坚果离析 + 置换零模型 | 本仓 p176w2-n08 验证面；`docs/DRUM_SEGREGATION_LESSONS.md` L1–L5 | 验证面 | `drum_segregation.metrics` + `tests/test_drum_seg_verification.py` |

**标准算例参数（本轮冻结，SI）**：R = 0.15 m，fill = 0.30，d_s = 10 mm，d_l/d_s = 2，
大颗粒面积分数 0.30（N = 209），dt = 1e-4 s，e = 0.30，μ = 0.40，μ_r = 0.10，ρ = 2500 kg/m³。
Fr ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1}，10 圈。

**第 1 轮的死法（诊断，对照 audit #14512）**：

1. **休止闸门**：`settle_rms_speed = 6.23e-2 m/s` vs 阈值 5e-3 —— 床没静止就开转。
   根因：RSA 初始床铺在 2.6× 目标面积的区域里（`bed_region_factor`），是"稀雨"，
   而旧协议的淬火轮数上限 4 轮 + 0.98/0.95 阻尼不足以在预算内压实；
   本轮把淬火改成**收敛循环**（上限 20 轮，实测 rms 与轮数都记账），并在
   `resting_ok` 里落布尔判据。
2. **流态**：旧表把 Fr=0.1 叫 "slipping"、0.5 叫 "rolling"，是节点预置分带（IN32-D01）
   而非文献分带，且报告里**没有任何测量对象**。本轮改用 M4 文献分带（定性）+ 三个
   **纯运动学观测量**（壁层速度比、上半区飞行分数、r/R 的 5 分位）驱动的
   `classify_measured`，阈值冻结在 prereg 里，先冻后跑。
3. **偏析**：旧报告在 `mode=mixed` 时仍勾"方向正确"。本轮以置换零模型的 z 为准，
   `mixed` 一律记"不显著"，不得勾方向。
4. **S(rev) 无平台**：本轮加 20 圈对照（Fr=0.1）。
5. **工件未入库**：本轮全部 JSON/图进 `artifacts/case_rotating_drum_r2/`（仓内路径）。

---

## ③ 选型沉淀（方法选型 + 迁移方案）

**选型**：

* **保留** 2-D 软球 DEM 切片内核（M5/M6）：算例判据是定性 + 20%，2-D 切片几何与
  经典巴西坚果/偏析实验同构，3-D 成本不必要。
* **新增** 三个运动学观测量的逐样本采样（壁接触弧上端角、壁层速度比、上半区飞行分数、
  r/R 5 分位、床层角速度比），使"流态/释放角"从**断言**变成**测量**。
* **新增** 文献分带表 `MELLMANN_BANDS` 与纯测量分类器 `classify_measured`
  （只用观测量，不含 Fr）。
* **定位基准**：定量锚点只用**精确闭式**（M2 释放角、M3 离心化起点 Fr=1），
  量级分带（M4）只做定性标签 —— 避免把"量级分带"当数值基准（前轮死法之一）。
* **弃用**：把 `arccos(−Fr)` 说成"从文献表格查得的目标数值"——它是闭式解，
  零手抄的含义是**脚本内计算**，报告须明写。

**迁移方案（已落地到本仓）**：

| 变更 | 文件 | 性质 |
|---|---|---|
| 逐粒子接触计数 + 运动学序列 | `src/tensordem/drum_segregation/kernel2d.py` | capability_extension |
| 休止/淬火收敛循环（上限可配，实测记账） | 同上 | defect_fix |
| `MELLMANN_BANDS` / `classify_froude_literature` / `classify_measured` | `src/tensordem/drum_flow/regimes.py` | new_capability |
| 本轮基准 + 汇总/作图 | `benchmarks/rotating_drum_froude_r2.py`、`..._report.py` | new_capability |
| 冻结判据 | `prereg/CASE_ROTATING_DRUM_R2.json` | 证据纪律 |
| 不变量测试 | `tests/test_drum_froude_r2.py` | tests |

**`oss_solutions` 落库字段**（feature_slug = `CASE-ROTATING-DRUM`）：
repo=TensorDEM；tag=`method-selection`；method=上表 M1–M6 + 运动学观测量；
conventions=精确闭式作定量锚、量级分带只做定性、阈值先冻后跑、`mixed` 不勾方向；
transfer=任何"旋转壳体+颗粒床"算例（转鼓/回转窑/球磨）可直接复用观测量与分类器；
evidence=本文件 + `artifacts/case_rotating_drum_r2/summary.json`。