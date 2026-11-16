# CASE-DEM-PLANE-SHEAR-MUIDI — R32: 薄层/厚层参考分支审计

锚点：`CASE-DEM-PLANE-SHEAR-MUIDI`（算例池·平面剪切 mu(I) 流变律标定）
验收：$\mu(I)$ 与 $\phi(I)$ 对 da Cruz 2005 / GDR MiDi 2004，容差 **10%**，$I=10^{-3}\ldots0.3$
本单：m23041（verifier 席，自找活）

---

## 1. 为什么做这件事（缺口在哪）

`reference_2d.py` 把验收基准写成 da Cruz 2005 Eqs. (13)-(14) 的**线性**二维律：

$$\mu^*(I)\simeq\mu^*_{min}+bI=0.22+1.0\,I,\qquad \nu(I)\simeq\nu_{max}-aI=0.81-0.3\,I\quad(\mu=0.4)$$

R24–R31 每轮都用 $\nu_{max}=0.81$ **无条件**。但归档原文
（`runs/provenance_archive/dacruz2005_excerpts.txt`，键 `thin_layer_packing`）明确说
0.81 是**厚层**值，薄层另有一个：

> "The parameters remain the same, except for the solid fraction which is
> smaller than for a thick layer: νmax = 0.82 instead of 0.84 for µ = 0 and
> **νmax = 0.78 instead of 0.81 for µ = 0.4**."

本轮之前的 **31 个轮次从未判定过本算例属于哪一支**。本算例 $h_{eff}=gap-d=0.15-0.01=0.14$，
$d=0.01$，即 **14 层**自由盘——既非明显"厚"也非明显"薄"，而归档摘录**没有给出分隔两支的层数**。
这就是 $\phi$ 判据上一处未被量化的**参考侧系统偏差**，属于典型的"验证欠账"。

## 2. 怎么做的

新增 `benchmarks/plane_shear_mu_i/thin_layer_branch_audit.py`（纯算术，不碰接触物理）：

* `LAYER`/`layer_count`：算例层数 $h_{eff}/d$（CLI 的 `--gap` 语义是**总间隙**，内部减 $d$）；
* `phi_ref_branch`：单支 $\nu(I)=\nu_{max}-aI$；
* `judge_point(s)`：**两支各判一次**，输出 `nu_ref`/带符号 `err`/`within`；
* `compare_branches`：给出账本口径的结论；并**显式禁止**把 $\mu$ 的判定归因到层厚分支——
  归档原文同一句说明"dilatancy and friction laws are **not affected**"，即**摩擦律与层厚无关**，
  所以分支只能动 $\phi$，**不可能**救或破 $\mu$ 的 FAIL；
* `render_tables`：markdown 表格，数字程序化插值、零手抄。

不替原文选边：原文没给分隔层数，就把两支当**包络**同时报，只声明"判定是否随分支翻转"。

## 3. 计算结果

当前可用的真实数据只有 R31 的 **smoke** 跑（200 盘 / 0.6 应变，$w=0.5$）；计例输出见
`runs/thin_layer_branch_smoke/thin_layer_branch_tables.md`：

| 文件 | $I$ | $\phi$ | $\nu_{ref}$ 厚 0.81 | err % | $\nu_{ref}$ 薄 0.78 | err % |
|---|---|---|---|---|---|---|
| `prof_r31_w0.5_I0.05_s0_smoke.json` | 0.01137 | 0.8043 | 0.8066 | **−0.29** | 0.7766 | **+3.56** |
| `prof_r31_w0.5_I0.2_s0_smoke.json`  | 0.02958 | 0.7839 | 0.8011 | **−2.16** | 0.7711 | **+1.65** |

结论（表 A3）：**两支都 PASS**，$\phi$ 最差绝对误差 厚 2.16 % / 薄 3.56 %，
`branch_sensitive = False` ⇒ **薄/厚层歧义不改变 $\phi$ 的判定**。
把 $I$ 换成名义 $I_{bulk}$ 时两支一致 FAIL（16.15 % / 21.56 %），同样不翻转 ⇒ 敏感性结论对横坐标选择也稳健。

图：`figures/thin_layer_branch_audit.png`（左：两支参考律 + 测点；右：±10% 容差带内两支误差对比）。

## 4. 注意点 / 局限（重要）

1. **这两点是 smoke 数，不是锚点证据。** smoke 用的是 200 盘、0.6 应变窗，远未到稳态；
   R31 smoke 的横坐标比 $\dot\gamma_{meas}/\dot\gamma_{nom}=0.071\text{–}0.131$、滑移长度
   $2.8h$ ⇒ 该工况近乎**塞流(slip)流动**，其 $\phi$ 不能当作验收依据。本节的结论只是
   "**分支敏感性**这一项可以判"，绝不能读成"$\phi$ 已达 10%"。
2. 归档原文未给分隔层数，故"14 层属薄还是厚"**仍未解决**；本单只把它的**影响量化**，
   并证明它不足以单独翻转现有点的判定。
3. $\mu$ 的 FAIL 与本审计**无关**（摩擦律厚度不变）；R30 的 $\mu$ 结论（$b=0.279\pm0.031$ vs 1.0）不因此改变。
4. 原文实测 $a=0.31$（$\mu=0.4$）而 Eq.(13) 用 0.3，模块两者都保留，默认取 0.3 保守。

## 5. 未完成（honest）

* **R31 全分辨率预注册轮次（6 组合，458 盘，8 应变窗）已递交集群 `cj4542`（6 核）**，
  收工时状态仍为 `queued`（队首约 74 个在飞作业），故 `outputs/r31/r31_tables.md` 仍是空表
  （`n_run=0`），R31 预注册的 P1–P4（横坐标有效性、流项大小）**没有判定**。
  该轮的判定规则已冻结在 `prereg_r31.json`，作业落地后跑
  `analyse_r31_profile.py` 即可出表，无需再改判据。
* 因此本单**不宣称**锚点验收通过，也不改动 `feature_ledger` 状态。

## 6. 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 本单误差 | <3% ? |
|---|---|---|---|
| $\phi$ 最差误差（厚层 0.81） | da Cruz Eq.13 | 2.16 % | **是**（但仅为 smoke 点，不作验收） |
| $\phi$ 最差误差（薄层 0.78） | 同上（薄层支） | 3.56 % | **否**（3.56 % > 3 %，但 < 10 % 容差） |
| 分支引起的误差漂移 $\Delta$err | 两支之差 | 1.40–3.85 百分点 | — |
| $\mu(I)$ | da Cruz Eq.14 | **未测**（R30 归档：斜率 3–4 倍偏小） | **否 / 未知** |

## 7. 测试

`pytest benchmarks/plane_shear_mu_i/test_thin_layer_branch_audit.py` → **16 passed**；
原有 `test_r30_2d_reference.py` + `test_r31_profile.py` → **27 passed, 3 skipped**（未回归）。