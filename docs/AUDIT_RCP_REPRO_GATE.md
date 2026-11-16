# AUDIT · RCP 审计账目的可复现闸门（CASE-DEM-RCP-PACKING-FRACTION）

**锚点**：`CASE-DEM-RCP-PACKING-FRACTION`（算例池·随机密堆体积分数 RCP）
**本单性质**：验证欠账补齐——把「已提交的审计产物是否由已提交的代码+数据重算得出」变成可执行的判定，
不改任何求解器方程、不改任何既有结论的判据。

## 1 欠账的准确表述

本锚点的每一次裁决（`IN_BAND`/`OUT_OF_BAND`/`UNDECIDABLE`、`CONVERGED`/`BIASED`）都是
`benchmarks/case_rcp_deposition/*_audit.py` 读 `runs/*.npz|json` 后写出的 JSON + Markdown。
账目引用的是这些产物，但仓内从来没有反向检查：**产物是不是现码现数据重算出来的**。
一个不能重算的产物是「主张」而不是「证据」：过期产物会让一个已被取代的数字继续活着，
跨工作树拷来的数字还会把别人检出里的绝对路径（进而别人的版本口径）带进本仓。

闸门覆盖 9 个审计脚本，逐叶比对 JSON（数值容差 `rtol=1e-09`, `atol=1e-12`），逐行比对 Markdown，并扫描外来绝对路径。

## 2 怎么做的

每个脚本：快照 `git status` → **用它自己的默认参数**运行 → 逐叶 diff 新产物与 `git show HEAD:<path>`
→ 逐行 diff 文档 → 扫路径泄漏 → `git checkout --` 还原。闸门对被测树是**只读**的（还原后 `git status` 为空）。

判级规则（预登记，见脚本 docstring）：

| 判级 | 含义 |
|---|---|
| `REPRODUCED` | 所有叶子在容差内，无类型/缺失变化 |
| `CENTRE_OK` | 有叶子超容差，但所有**分类裁决**（带宽判定/认证位）未变 |
| `STALE` | 至少一个分类裁决变了 |
| `FAILED` / `NO_ARTEFACT` | 脚本崩/超时，或提交产物不存在 |

## 3 结果

总体：**REVIEW**（阻断项 2 个）。

| 脚本 | 判级 | 墙钟 (s) | 叶子 | 相同 | 容差内 | 超容差 | 类型/值变 | 裁决位变 | 路径泄漏 | 随机核 |
|---|---|---|---|---|---|---|---|---|---|---|
| `aggregate.py` | `STALE` | 10.6 | 1047 | 915 | 0 | 15 | 0 | 1 | 0 | 否 |
| `debt_audit.py` | `CENTRE_OK` | 5.0 | 153 | 88 | 0 | 25 | 22 | 0 | 0 | 否 |
| `order_audit.py` | `STALE` | 7.6 | 403 | 150 | 0 | 36 | 6 | 5 | 0 | 否 |
| `steady_state_audit.py` | `CENTRE_OK` | 42.5 | 4446 | 3805 | 0 | 1 | 1 | 0 | 0 | 否 |
| `phi_definition_audit.py` | `CENTRE_OK` | 53.6 | 3668 | 2711 | 0 | 2 | 0 | 0 | 0 | 否 |
| `anchor_provenance_audit.py` | `CENTRE_OK` | 7.7 | 509 | 506 | 0 | 2 | 1 | 0 | 0 | 否 |
| `timestep_ladder_audit.py` | `REPRODUCED` | 8.1 | 316 | 316 | 0 | 0 | 0 | 0 | 0 | 否 |
| `seed_bias_audit.py` | `REPRODUCED` | 1.0 | 101 | 101 | 0 | 0 | 0 | 0 | 0 | 否 |
| `stiffness_seed_audit.py` | `CENTRE_OK` | 73.7 | 308 | 294 | 0 | 12 | 2 | 0 | 2 | 是 |

### 3.1 超容差叶子（逐条）

| 脚本 | 路径 | 提交值 | 重算值 | Δ | 相对 |
|---|---|---|---|---|---|
| `aggregate` | `/gates/id=G13_stiffness_limit_vs_ref_frictional/got` | 0.617389439912571 | 0.6183098776212702 | +9.204e-04 | +0.149% |
| `aggregate` | `/gates/id=G13_stiffness_limit_vs_ref_frictional/value` | 0.012176896139886373 | 0.010704195805967709 | -1.473e-03 | -12.094% |
| `aggregate` | `/gates/id=G14_timestep_convergence_mu0_kn64000/got` | 0.0005413970661711787 | 0.0007297640360642843 | +1.884e-04 | +34.793% |
| `aggregate` | `/gates/id=G14_timestep_convergence_mu0_kn64000/value` | 0.0005413970661711787 | 0.0007297640360642843 | +1.884e-04 | +34.793% |
| `aggregate` | `/gates/id=G6_stiffness_convergence_frictional/got` | 0.009318951823059973 | 0.008398514114360767 | -9.204e-04 | -9.877% |
| `aggregate` | `/gates/id=G6_stiffness_convergence_frictional/value` | 0.009318951823059973 | 0.008398514114360767 | -9.204e-04 | -9.877% |
| `aggregate` | `/impl_hashes/b44c19146425bc58` | 6 | 9 | +3.000e+00 | +50.000% |
| `aggregate` | `/stiffness/mu=0.1/delta_from_last` | 0.0018911687935586619 | 0.0002875007089813142 | -1.604e-03 | -84.798% |
| `aggregate` | `/stiffness/mu=0.1/limit` | 0.617389439912571 | 0.6183098776212702 | +9.204e-04 | +0.149% |
| `aggregate` | `/stiffness/mu=0.1/n_levels` | 3 | 4 | +1.000e+00 | +33.333% |
| `aggregate` | `/stiffness/mu=0.1/order` | 0.5752223894649274 | 0.8777530885312191 | +3.025e-01 | +52.594% |
| `aggregate` | `/stiffness/mu=0.1/order_used` | 0.5752223894649274 | 0.8777530885312191 | +3.025e-01 | +52.594% |
| `aggregate` | `/timestep/mu=0.0|kn=64000.0/delta` | 0.0005413970661711787 | 0.0007297640360642843 | +1.884e-04 | +34.793% |
| `aggregate` | `/timestep/mu=0.0|kn=64000.0/phi` | 0.6367992185609932 | 0.6367647448488053 | -3.447e-05 | -0.005% |
| `aggregate` | `/timestep/mu=0.0|kn=64000.0/phi_halved` | 0.636257821494822 | 0.636034980812741 | -2.228e-04 | -0.035% |
| `debt_audit` | `/after_full_set/n_prod_runs` | 20 | 25 | +5.000e+00 | +25.000% |
| `debt_audit` | `/after_full_set/n_runs` | 22 | 31 | +9.000e+00 | +40.909% |
| `debt_audit` | `/after_full_set/stiffness/0.1/delta_from_last` | 0.0022297031991581617 | 0.0002875007089813142 | -1.942e-03 | -87.106% |
| `debt_audit` | `/after_full_set/stiffness/0.1/limit` | 0.6170509055069715 | 0.6183098776212702 | +1.259e-03 | +0.204% |
| `debt_audit` | `/after_full_set/stiffness/0.1/n_levels` | 3 | 4 | +1.000e+00 | +33.333% |
| `debt_audit` | `/after_full_set/stiffness/0.1/order` | 0.5123790583140884 | 0.8777530885312191 | +3.654e-01 | +71.309% |
| `debt_audit` | `/after_full_set/stiffness/0.1/order_used` | 0.5123790583140884 | 0.8777530885312191 | +3.654e-01 | +71.309% |
| `debt_audit` | `/after_full_set/stiffness/0/delta[1]` | -0.001106093134926489 | -0.0011405668471143882 | -3.447e-05 | -3.117% |
| `debt_audit` | `/after_full_set/stiffness/0/delta_from_last` | 0.0003637649551401223 | 0.00035130390673376866 | -1.246e-05 | -3.426% |
| `debt_audit` | `/after_full_set/stiffness/0/limit` | 0.6364354536058531 | 0.6353595292218703 | -1.076e-03 | -0.169% |

### 3.2 外来绝对路径

共 2 处。它们让产物的来源不可追溯（路径指向别人的工作树），且一旦检出目录改名就失效。

| 脚本 | 叶子 | 值 |
|---|---|---|
| `stiffness_seed_audit` | `/figures[0]` | `/srv/tensor/agent/work/m22796/benchmarks/case_rcp_deposition/figures/fig_stiff_stiffness_ladder_seeds.png` |
| `stiffness_seed_audit` | `/figures[1]` | `/srv/tensor/agent/work/m22796/benchmarks/case_rcp_deposition/figures/fig_stiff_stiffness_limit_ci.png` |

### 3.3 随机核（`--n-mc`）敏感性

本次未执行（`--mc-scan` 关闭）。

### 3.4 重算修正记录（`runs/repro_regen.json`）

发现过期产物后按仓内约定**重算**（产物与文档都由脚本生成，重算即恢复一致）。重算前的差异：

| 脚本 | 重算前判级 | 超容差叶子 | 裁决位变 | 过期产物里的外来路径 |
|---|---|---|---|---|
| `aggregate.py` | `STALE` | 15 | 1 | 0 |
| `debt_audit.py` | `CENTRE_OK` | 25 | 0 | 0 |
| `order_audit.py` | `STALE` | 36 | 5 | 0 |
| `steady_state_audit.py` | `CENTRE_OK` | 1 | 0 | 0 |
| `phi_definition_audit.py` | `CENTRE_OK` | 2 | 0 | 0 |
| `anchor_provenance_audit.py` | `CENTRE_OK` | 2 | 0 | 0 |
| `timestep_ladder_audit.py` | `REPRODUCED` | 0 | 0 | 0 |
| `seed_bias_audit.py` | `REPRODUCED` | 0 | 0 | 0 |
| `stiffness_seed_audit.py` | `CENTRE_OK` | 12 | 0 | 2 |

其中**分类裁决**发生变化的位置（重算前 → 重算后）：

| 脚本 | 裁决叶子 | 旧 | 新 |
|---|---|---|---|
| `aggregate.py` | `/gates/id=G17_extrapolation_selfconsistency_mu0.1/pass` | `<absent>` | `True` |
| `order_audit.py` | `/gates/id=G13_corrected_limit_vs_ref_mu0.1/pass` | `False` | `True` |
| `order_audit.py` | `/gates/id=G13_four_level_vs_ref_mu0.1/pass` | `<absent>` | `True` |
| `order_audit.py` | `/gates/id=G13_four_level_vs_ref_mu0/pass` | `<absent>` | `True` |
| `order_audit.py` | `/gates/id=G17_extrapolation_selfconsistency_mu0.1/pass` | `<absent>` | `True` |
| `order_audit.py` | `/gates/id=G17_extrapolation_selfconsistency_mu0/pass` | `<absent>` | `True` |


## 4 注意点

1. 闸门只回答「能否重算」，不回答「结论是否物理正确」——物理裁决的证据链见 §3.1 引用的各审计文档。
2. 容差 `rtol=1e-9` 是**可复现性**容差（同码同数据同平台族），不是建模容差；平台/库版本漂移会表现为 `CENTRE_OK`。
3. `CENTRE_OK` 不等于「没问题」：它只保证分类裁决不变，端点数值可能已经与文档里的数字不一致，
   因此文档数字必须由脚本注入（本仓既有约定），不能手抄。
4. 闸门会运行被测脚本（本例单脚本墙钟见 §3 表），总时长在单机 <10 min 预算内可分片执行；
   全量在集群跑时用同一命令。

## 5 误差判定

本单的判定量是「重算值 vs 提交值」，基准就是提交值本身，故逐叶误差已在 §3.1 给出；
其中超容差叶子 7 个脚本、2 个脚本逐叶复现。
物理量对标文献的误差判定不在本单范围（见 `docs/AUDIT_m22796_RCP_STIFFNESS_SEED_RESOLVED.md` §8）。

## 6 结论与去向

* 逐叶复现：2/9 个脚本。
* 仅端点漂移、裁决不变：`debt_audit.py`、`steady_state_audit.py`、`phi_definition_audit.py`、`anchor_provenance_audit.py`、`stiffness_seed_audit.py`。
* 需处理：`aggregate.py`(STALE)、`order_audit.py`(STALE)。
* 后续每次改动审计脚本或 `runs/` 后重跑本闸门；把 `runs/repro_gate.json` 与本文档一同提交。

