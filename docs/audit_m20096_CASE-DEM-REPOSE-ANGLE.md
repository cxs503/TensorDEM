# 交差审计 · m19975 CASE-DEM-REPOSE-ANGLE（TensorDEM 玻璃珠休止角 22°）

审计单 m20096 ｜ 被审单 m19975（succeeded，auto-approve dev_acceptance，`server_quality_audit.gate_valid=false`）
被审工作树 `/root/tp-agent/work/m19975` @ `b99c206f` ｜ 裁决 `tensor_maturity_verdicts#836`：**L2**，verdict_by=`board-audit`
报告 `tensor_reports#18328` ｜ 复现作业 `cj2495`（排队未起步，记 not_run）

## 裁决

| 级别 | 结论 | 机器可核证据 |
|---|---|---|
| L1 | ✅ | 12 pytest（审计重跑 3.86 s），但全部是簿记/单位断言，无一条精度断言 |
| L2 | ✅ | `runs/audit_m20096/audited_snapshot/run_case.py`（:557 main / :394 figures / :548 TABLES / :615-620 工件写）+ 47 run 的 `results.json` |
| L3 | ❌ | harness 0 提交（`git log --all --diff-filter=A -- benchmarks/case_dem_repose_angle` 空）、0 调用方、0 登记行、无平台接线 |
| L4 | ❌ | ①登记裁判 0 行 ②预测腿 64.61% ③标定腿是同一标量的单参数拟合且根不唯一 ④工件半成品 |
| L5 | n/a | 锚点为论文一行转录，未独立测量 |

## 关键数字（`runs/audit_m20096/findings.json`，程序化重算）

| 判据 | ref | got | Δ | 相对误差 | <3% | <15% |
|---|---|---|---|---|---|---|
| AC-1 预测腿 (μ_r=0.05) | 22.00° | 7.79° | −14.21° | **64.61%** | ✗ | ✗ |
| AC-2 标定腿 (μ_r*=0.288) | 22.00° | 21.27° | −0.73° | **3.33%** | ✗ | ✓ |
| 参考自洽 tan22° vs μ=0.40 | 0.40403 | 0.40000 | — | 1.01% | ✓ | ✓ |
| 派生 ρ_p = ρ_bulk/φ | 2508.4746 | 2508.4746 | 0 | 0.000% | ✓ | ✓ |

标定腿：留出种子 σ = 3.56° > ±3° 容差（σ/tol=1.19），|Δ|/σ=0.21 ⇒ 拟合一致性，不是精度验证。
根的唯一性：θ(μ_r) 扫描 2 处反向（θ(0.25)=15.07° < θ(0.20)=21.43°），目标 ±3° 带内至少 μ_r∈{0.20, 0.30, 0.404} 三点 ⇒ μ_r* 不可辨识。

## 关键发现

- **F1 BLOCKING** `tensor_benchmark_cases` 对 `anchor_slug='CASE-DEM-REPOSE-ANGLE'` **0 行**（`LIKE '%REPOSE%'` 也 0 行）⇒ 无登记裁判，headline “PASS” 不成立为平台级结论。
- **F2 BLOCKING** 预测腿 7.79° vs 22°（64.61%）> 15% 红线 ⇒ 该算例“可标定但不可预测”（作者自己的 H0）。
- **F3 BLOCKING** AC-2 PASS 循环：μ_r* 拟合到同一 22° 标量，σ 大于容差，根不唯一。
- **F4 MAJOR** 任务宣称的 `TABLES.md`、`fig1..3.png` 不存在；`results.json` 无 `figures` 键 ⇒ `run_case.py:620` 二次写未发生，与“cj2434 rc=0”矛盾。
- **F5 MAJOR** AC-4 以 2σ 松弛判定 “invertible”，容忍 −6.36° 反向 ⇒ 该判据在 σ≈1–4° 下不可证伪。
- **F6 MINOR** AC-3 无摩擦对照 θ=−2.63e-31°（平堆），平凡通过。
- **F7 MINOR** 报告 0 内嵌图：全库 12456 条报告仅 7 条保留 `png_base64`（10013 条有 figure 行）⇒ 平台持久化行为，非作者缺陷。
- **F8 MINOR** 被审工作树 6 项全 `??`，从未 commit ⇒ 证据以 sha256 冻结在 `runs/audit_m20096/audited_snapshot/`。

## 产物

- `runs/audit_m20096/compute_checks.py` → `findings.json`（ref/got、误差、单调性、工件齐全性，零手抄）
- `runs/audit_m20096/make_figures.py` → `figures/fig1_root_identifiability.png`、`figures/fig2_ref_vs_got.png`
- `runs/audit_m20096/audited_snapshot/`（run_case.py、测试、docs、results.json、criteria.json + `SHA256SUMS`）
- `runs/audit_m20096/verdict_m20096.{json,sql}`、`publish_report.py`

## 未跑 / 局限

- `cj2495` 全量复现（784 s 级）递交后长时间排队，**not_run（unknown）**，不得读作 repro 通过。
- 未重开 arXiv:1307.4638v2 核对 22° 原行；锚点为转录。
- 机制观察：`maturity_gate.judge()` 对 L4 的判据是“benchmark 含数字 + repro_cmd 非空”，本单同样的 evidence 若声明 L4 也会被机器确认（`judge(4, ev, 'TensorDEM', 20096) -> (4, [])`）——科学有效性只能由审计者拒绝，机器防线在此处是空的。