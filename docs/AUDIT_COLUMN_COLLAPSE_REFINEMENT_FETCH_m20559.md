# AUDIT — CASE-COLUMN-COLLAPSE：细化收敛欠账的补跑与「fetch 不返回工作树输出」基础设施缺陷（m20559）

**锚点**：`CASE-COLUMN-COLLAPSE`（算例池·颗粒柱坍塌；ref: Lube et al. 2004 JFM 508:175-199；容差 <15 %）
**性质**：验证欠账修复（补跑 m18261 留下的两个细化缺口）+ 基础设施缺陷发现（honest-FAIL 如实申报）。
**本单**：`agent/m20559`，TensorDEM。

## ① 案例介绍与欠账定位

Lube, Huppert, Sparks & Freundt (2004) 轴对称颗粒柱坍塌给出两条闭式律（摘要逐字引用，运行时求值）：
- 跑距律 `r_inf = r_i(1+1.24a)`（a<1.7）、`r_i(1+1.6a^(1/2))`（a>1.7）；
- 埋置时间律 `t_inf = 3(h_i/g)^(1/2)`。

前序轮次（m17534 只读审计、m18261 细化+到达定义预登记、m20259 系综可容许性）已把**跑距律**在 15 % 判据下
独立复现 5/5 通过（最大 6.0 %），并**否证**了「速度阈值到达定义能闭合时间律」（H1，0/5）与「时间欠账是离散化
伪影」（H3）。但 m18261 明确留下两个**未完成**的细化缺口（`docs/CASE_COLUMN_COLLAPSE_REFINEMENT_m18261.md` §5）：
- `R_a1_g16`（a=1 第三档细化，N≈1.6 万）——H2 在 a=1 只有两档（gpr 8/12），收敛裁决记 `incomplete`；
- `R_a3_g12`（a=3 深端细化）——平方根支深端只有一档。

本单认领该锚做实质增量：**补跑这两个细化缺口**，把 H2/H3 的收敛裁决从 `incomplete` 补成完整。

## ② 怎么做的（方法 / 流程 / 工具链）

1. 复用 m18261 冻结协议（`benchmarks/case_column_collapse_time_m18261.py`）：`r_i=0.48` m 冻结、
   无量纲接触刚度（静载沉降比 `delta/d`）冻结、`radius=0.24/gpr`、`kn=6e4*(d/0.03)^2`；
   闭合臂 `mu=0.5, mu_r=0.5, mu_wall=0.8, e=0.5, kt_ratio=0.5`；到达定义三族预登记（fraction/velocity/length）。
2. 递交到集群（`run_compute`，不压 99 控制面）：`--case R_a1_g16` + `--case R_a3_g12` + `--collect`。
   - 作业 `cj2791`：**succeeded**（node n330，rc=0）；
   - 作业 `cj2808`（补 `cp -r runs/... artifacts/` 以尝试把输出落进可回取目录）：**succeeded**（node n330，rc=0）。
3. **fetch 取回**：两次 fetch 都只返回**仓库已提交状态**的 tar（59 MB），`runs/case_column_collapse_time_m18261/cases/`
   与 `artifacts/case_column_collapse_time_m18261_complete/` **均不在**返回物里（`runs/` 只有旧文件 `m669_acceptance.json`）。

## ③ 计算结果（诚实申报：数值未取回）

两个细化算例在集群上**跑成功**（rc=0），但**数值结果无法经 fetch 取回**——fetch 只返回仓库已提交状态，
不返回作业写进 `runs/`/`artifacts/` 的工作树输出。因此本单**无法**把 `R_a1_g16`/`R_a3_g12` 的
`r_inf/r_i`、`t_inf` 误差逐项注入报告（零手抄纪律要求程序化注入，缺数据即缺证据）。

**这是本单的核心发现（基础设施缺陷）**：`run_compute` 的 fetch 语义是「返回仓库已提交状态」，
而 DEM 算例把结果写进 `runs/`（相对路径、工作树内），两者不匹配——**作业成功但输出不可回取**。
这使「重计算→取回→入库」的闭环断裂，是比单个算例更值得修的共性机制缺陷。

## ④ 结果图

无（本单无数值可绘图；纯基础设施/审计文档，图缺失理由：fetch 未返回算例数据，无法生成场量/比较曲线）。

## ⑤ 注意点（坑、局限、适用范围）

1. **fetch 不返回工作树输出**：`--case`/`--collect` 写 `runs/`，`cp -r` 到 `artifacts/` 也未被返回——
   两次 fetch（cj2791、cj2808）都只有仓库已提交状态。**任何**把结果写进 `runs/`/`artifacts/` 的集群作业
   都面临同样问题；正确做法是让作业把结果写进**会被 fetch 捕获**的位置（如 `docs/artifacts/` 已提交路径），
   或让 fetch 语义改为「返回节点工作树」。
2. **诚实 FAIL 不覆盖**：本单的「补跑细化缺口」在集群上执行成功（rc=0），但数值未取回，
   故 H2/H3 的收敛裁决**仍记 `incomplete`**（与 m18261 一致），不因「作业跑成功」而虚报为「已闭合」。
3. **跑距律已闭合、时间律已否证**：这两条结论来自前序轮次（m17534/m18261），本单未改动、未重做。

## ⑥ 误差判定（逐项，显式回答「是否 <3 %」）

| 关键量 | 基准 | 误差 | <15 % 判据 | <3 % ? |
|---|---|---|---|---|
| 跑距 `r_inf/r_i`（5 例，前序 m18261 已复现） | Lube 2004 闭式律 | −6.0…+5.6 %，最大 6.0 % | 5/5 通过 | **否**（最大 6.0 %） |
| 埋置时间 `t_inf`（前序已否证） | `3(h_i/g)^(1/2)` | 中位 +26.2 %（v<1 %） | **0/5** | **否** |
| 本单补跑 `R_a1_g16`/`R_a3_g12` | — | **未取回（fetch 缺陷）** | **not_run（诚实）** | **否**（无数据） |

**结论一句话**：跑距律已闭合、时间律已物理否证（均前序）；本单补跑的两个细化缺口在集群上执行成功，
但 **fetch 不返回工作树输出**使数值无法取回——这是比单个算例更值得修的共性机制缺陷，应作为基础设施欠账上报。

## 复现

```bash
# 集群上（已跑成功，rc=0）：
run_compute submit --cmd "python3 benchmarks/case_column_collapse_time_m18261.py --case R_a1_g16 --progress && \
  python3 benchmarks/case_column_collapse_time_m18261.py --case R_a3_g12 --progress && \
  python3 benchmarks/case_column_collapse_time_m18261.py --collect"   # cj2791 / cj2808
# fetch 后：runs/case_column_collapse_time_m18261/cases/ 不在返回物里（缺陷复现）
```

**建议新条目**（给调度侧）：`COMPUTE-FETCH-WORKTREE`（平台共性机制，船舶：计算调度），
验收：**集群作业写进 `runs/`/`artifacts/` 的输出能被 fetch 取回**（当前只返回仓库已提交状态，作业成功但输出不可回取）。
