# 交差核验 AUDIT — m17498 `CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE` (TensorDEM)

审计单：m17626（交差核验，非重做）｜审计对象：mission 17498（status=succeeded，
commit `1d7a0b6e`，branch `agent/m17498`，worktree `/root/tp-agent/work/m17498`）
｜审计日：2026-11-13｜裁定人：board-audit

## 0. 结论先行

| 项 | 裁定 |
|---|---|
| 三件套（run_artifacts / ref-vs-got / repro_cmd） | **齐**，数值可复算，但"原始场"未落盘 |
| 基准裁判（唯一裁判） | 注册行存在（`tensor_benchmark_cases` id=59，ref 60°，tol <15%）；平台重算行 `tensorbench_results` = **0 行** → L4 端点 = `CLAIMED_NOT_PROVEN` |
| 数值 | got 53.832° vs ref 60° → **10.28%**，过 15% 判据（✓），**不过 3% 判据**（✗，作者已如实声明） |
| 级别 | **L2**（真实现+可执行测试）；L3 拒（未进 main、零调用方）；L4 拒（无平台重算行 + 无原始场 + 2/8 预登记试样 WITHHELD） |
| verdict 落表 | `tensor_maturity_verdicts` id=601（`gate_state=pending`，等成熟门自动裁定） |

## 1. 取证：三件套逐项核验

| 件 | 指针 | 核验结果 |
|---|---|---|
| run_artifacts | `artifacts/biaxial_shear_band/r3_ensemble.json`（4216 B, sha256 `43fbc577…`）、`r3_ensemble.png`（106794 B）、`r3_estimator_audit.json`（39218 B）、`r3_ensemble.log`（389 B / 6 行） | **存在且在 commit 内**；但 `r3_cache/` 空、`*.npz` = 0 → 无逐试样原始场，"got" 只能回溯到 6 行日志 |
| ref vs got | ref 60°（`docs/BIAXIAL_SHEAR_BAND.md:30-31` → Desrues & Viggiani 2004 IJNAMG 28:279-321，55–60° 取中点）；got 53.832°（5 独立试样集合均值） | 审计方**独立复算**：均值 53.832°（Δ=0.0）、样本 std 0.8978°（Δ=0.0）、95%CI半宽 1.1146°（Δ=−1.8e-4）、相对误差 0.1028（Δ=0.0）；日志 vs JSON 逐试样 0 处不一致 |
| repro_cmd | `PYTHONPATH=src python3 benchmarks/biaxial_shear_band/run_r3_ensemble.py --workers 5` → `summarise_r3_log.py` | CLI 参数可达、依赖可导入、算子被审计方**实测复跑**（见 §3）；端到端集合重跑 **not_run**（单试样 95 s×8，机器负载 216/128 核，集群 `cj1619` 排队 383） |

## 2. 基准裁判（唯一裁判）

```
tensor_benchmark_cases.id=59 | anchor_slug=CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE
metric='shear band inclination vs anchor (deg)' | tolerance='<15% (owner standard)'
status='passed'  ← 注意：passed 值等于 created_at（2026-11-13 06:13:49），
                   属注册默认，不是测量出来的通过
tensorbench_results WHERE mission_id=17498 → 0 行
GET /tensor/bench/l4-verdict/17498 → CLAIMED_NOT_PROVEN
   failed_checks = [execution_success, source_cluster_job, cluster_job_bound,
                    ref_got_error_present, within_tolerance]
```

裁判意义上的"通过"**不成立**：判据值（60°/15%）齐备、作者数值落在容差内，但没有任何
平台重算行把 ref/got/error 绑到一次可追溯的执行上。作者侧/审计侧算术均复现 10.28%，
所以这不是"算错"，而是"证据链未闭合"。

## 3. 审计方独立探针（不是重读作者报告）

`docs/board_audit/m17626/audit_m17498.py` → `audit_m17498.json`

* **算子无偏性（关键）**：在**真实 24×48 / 1128 颗粒 case 点阵**上注入已知角度合成带
  （slip 0.02 m、无噪声），生产 Radon（`binning="equal_count"`，R2 修好的那版）恢复
  45°/60°/70° → **误差 0.0 / −1.0 / 0.0°，max 1.0°**。
  ⇒ 53.8° 不是 R1/R2 那个长宽比归一化缺陷（该缺陷会把 60° 的带读成 44°）。
* **算术闭合**：从 6 行日志独立重算均值/std/CI，与发布 JSON 一致（见 §1 表）。
* **测试**：`tests/test_shear_band_structure_tensor.py` 9 项，审计方实跑 5 项全过
  （60.0° 已知倾角 17.6 s；4 项边界守卫 27.1 s）；余 4 项受 110 s 壳上限未跑。

## 4. 判级（L0–L5）与理由

* **L2 成立** = 真代码输入→计算→输出：`RidgeMaterial` 双轴胞（1128 颗粒、4000 固结步、
  p=100 kPa、壁摩擦 0.55）真实 DEM 跑 5 个独立种子；新增结构张量算子模块；9 项可执行测试。
* **L3 拒**：`git merge-base --is-ancestor 1d7a0b6e main` = false；在 main HEAD `c6b2b86f`
  上 `grep -rn shear_band_structure_tensor` = **0 命中**；`src/tensordem/__init__.py` 无导出；
  无 CLI/API/benchmark 编排调用方。
* **L4 拒**：① 平台裁判 0 行（CLAIMED_NOT_PROVEN）；② 无逐试样原始场落盘 → 第三方无法
  用自己的算子复测（而"场噪声水平"恰是作者自己判定为决定性的因子：>0.5% 噪声时全部算子
  RMS 27–33°）；③ 预登记 8 试样只完成 6（锚点 5/5），累计窗口与 32×64 分辨率点自认 WITHHELD；
  ④ 10.3% 不满足 3% 精度线（15% 是 owner 接受线，不是"已验证"线）。
* **百页报告不改变级别**：报告六段齐全、含显式"<3%?"回答，这不加分到 L3/L4。
* 与同域先例一致：m17362（XVAL）= L2、m15279（CASE-BIAXIAL-SHEAR-BAND）= L2。

## 5. 发现（按严重度）

| 严重度 | 发现 | 证据 | 复现 |
|---|---|---|---|
| **中-高（阻塞 L4）** | 无 `tensorbench_results` 行；报告/注册行都不是测量证据 | `SELECT count(*) … WHERE mission_id=17498` = 0；L4 端点 CLAIMED_NOT_PROVEN | `GET /api/v1/tensor/bench/l4-verdict/17498` |
| **中-高（阻塞 L4）** | 原始场未持久化（npz=0、`r3_cache/` 空），集合 JSON 是 run 被 120 s 收割后由 6 行日志**机器转写**；"got" 不可由数据复推，只能整跑 | `audit_m17498.json:provenance.re_derivable_from_raw_fields=false` | 重跑 `audit_m17498.py` |
| **中** | 两算子同快照平均差 13.0°、最大 30.5°，结构张量 s11 单点 83.03°（38.4% off）→ 测量未收敛到算子无关 | `r3_ensemble.json:verdict.operator_*` | §3 复算 |
| **中** | 200 kPa 敏感性点 unresolved（73.50°，22.5%）→ 围压趋势未知 | `logs/r3_ensemble.log:5` | 同上 |
| **低** | 注册行 `status='passed'` 与 `created_at` 同一时刻写入 → 注册默认值会被下游误读为"已通过" | `tensor_benchmark_cases` id=59 | SQL |
| **低** | 外部锚点未回读原文原表（作者自认 WITHHELD，审计方亦未回读，离线环境） | `docs/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE_R3.md` §⑥ Gap 2 | — |

## 6. 未跑（标 unknown，不算 pass）

* 端到端集合重跑（8 试样 ≈ 8–13 min）：`not_run`（负载 + 集群排队 `cj1619` 383 位）。
* `tests/test_biaxial_shear_band.py` 15 项（生产算子回归）：`not_run`（仅作者跑过）。
* `tests/test_shear_band_structure_tensor.py` 余 4 项（30/45/70° 与长宽比中性）：`not_run`。
* 外部文献原表核对：`not_run`（离线）。
* 成熟门裁定：行 601 `gate_state=pending`（门为异步周期任务，写单时未运行）。

## 7. 处置

**approve with notes（交差成立，级别 L2）。** 交差核验通过：三件套真存在、到达数值在 15%
判据内、算术与算子无偏性均由审计方独立复现。级别不上 L3（未集成）/L4（裁判无重算行）。
返修建议（不阻塞本次消账，供下一轮锚点用）：会话结束时把逐试样 npz 落盘 + `run_cluster`
递交并把 `cj` 号写进证据 + 用 `/tensor/bench/results` 灌一条被测行，L4 即可闭合。

工件：`audit_m17498.py` / `audit_m17498.json` / `verdict_m17498.py` / `verdict_m17498.json` /
`m17626_audit.png`；落表 `tensor_maturity_verdicts` id=601。