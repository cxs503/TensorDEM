# 预注册（运行前冻结）— CASE-COLUMN-COLLAPSE 细化阶梯两条缺口 · m20873

**冻结时间点**：两个算例启动**之前**写入并 commit-able；阈值在归约脚本
`scripts/analyze_column_collapse_deep_refinement_m20873.py` 里作为常量复现，测试断言二者逐字相等。

锚点 `CASE-COLUMN-COLLAPSE`（算例池·颗粒柱坍塌）；基准 Lube, Huppert, Sparks & Freundt (2004)
*JFM* **508**:175-199, doi:10.1017/S0022112004009036；容差 < 15 %。

## 为什么是这两例（欠账出处，非自选）

| 缺口 | 出处（原文） | 前序状态 |
|---|---|---|
| `R_a1_g16`（a=1 第三档细化，gpr 16） | `docs/CASE_COLUMN_COLLAPSE_REFINEMENT_m18261.md` §5-5「算例缺口如实申报」；§⑦ H2「a=1 只有两档」 | `incomplete` |
| `R_a3_g12`（深端 gpr 12） | 同上 §5-5；§⑦ H2「a=3 只有一档」 | `incomplete` |
| 两例的补跑尝试 | `docs/AUDIT_COLUMN_COLLAPSE_REFINEMENT_FETCH_m20559.md` ③「作业成功但输出不可回取」 | `not_run`（fetch 缺陷） |

两个 `case_id` 在 m18261 脚本 `benchmarks/case_column_collapse_time_m18261.py::build_cases()`
里**早已声明**（14 例预注册表），从未实跑；本单**不改任何参数**，只补跑并裁决。

## 协议（完全继承，零重调）

`CLOSING_ARM` 冻结臂：μ=0.5、μ_r=0.5、μ_wall=0.8、e=0.5、k_t/k_n=0.5、ρ=2500 kg/m³、g=9.81 m/s²、
`packing_gap`=0.02、HCP 晶格、`jitter`=0、seed=12345、`dt_scale`=1、`duration_factor`=8、
`sample_stride`=5、`contact_mode="reference"`。
细化口径（m18261 冻结）：r_i = 0.48 m 固定、无量纲接触刚度 δ/d 固定 ⇒ `radius = 0.24/gpr`、
`kn = 6e4 (d/0.03)²`。

## 预测与判据（机器判定，事后不得改）

参考律由 `tensordem.granular_column.axisym.lube_2004_runout_ratio` /
`lube_2004_emplacement_time` 在运行时求值（零手抄）。误差一律用**实测**长宽比 a_meas 为主判据
（m20533 口径），a_nom 为存档口径。

| id | 预测 | 判据（pass 条件） |
|---|---|---|
| `P1_A1_G16_BAND` | a=1 第三档细化仍在 15 % 带内 | \|err_runout(a_meas)\| ≤ 15 % |
| `P2_A1_DRIFT` | a=1 的 gpr 12→16 漂移小 | \|err16 − err12\| ≤ 5 pp（沿用 m18261 H2 的漂移线） |
| `P3_A3_G12_BAND` | 深端（a=3）细化后仍在带内 | \|err_runout(a_meas)\| ≤ 15 % |
| `P4_A3_DRIFT` | a=3 的 gpr 8→12 漂移小 | \|err12 − err8\| ≤ 15 pp（沿用 m20337 `DEEP_RES` 线） |
| `P5_TIME_A3_BAND` | 深端埋置时间也在带内 | \|err_t95(a_meas)\| ≤ 15 % |
| `P6_DEEP_FAMILY` | **平方根支**（a>1.7）分辨率族整体稳健：a=2 gpr8/12 + a=3 gpr8/12 四点全部在带内 | 四点 \|err_runout(a_meas)\| ≤ 15 % |
| `P7_INTEGRITY` | 两例都是物理静止终止，非截断 | `stop_reason == "at_rest"`，且 `overlap_pairs == 0` |
| `P8_NOT_MONOTONE_ASSUMED` | **不作为判据**，但预登记观察项：跑距误差在阶梯上单调性 | m20337 已证非单调，本单**不**要求单调；若单调下降则记录，不得据此宣称收敛 |

**显式反例预案（honest-FAIL 纪律）**：若 `P1`/`P3` 任一 fail，则锚点在相应长宽比上的「跑距律 <15 %」
结论必须**降级**为该长宽比的单/双档巧合，不得沿用旧档结论。若 `P7` fail（截断），该例数值作废，
不计入任何通过。

## 成本与执行位置

两例单核本机并行（< 8 核、单例 < 10 min 规则内），避免 m20559 记录的「集群作业成功但 fetch 不返回工作树输出」
缺陷；产物落 `runs/` 后立即复制到 `docs/artifacts/`（入库路径）。