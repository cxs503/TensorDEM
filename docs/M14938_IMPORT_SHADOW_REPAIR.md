# m14938 · Import-contract shadow debt：把「静默丢失的验证面」修回来

**仓** cxs503/TensorDEM · **分支** agent/m14938 · **任务** m14938（paper 席自找活）
**锚点** 【锚点=NEW:同名遮蔽致 15 符号不可导入/29 个测试无法收集（audit F3 遗留，非总账条目）】
**上游依据** `audit_report.md` F3 / §8 建议 2–3（mission 14487 审计），`tools/import_contract_scan.py`

---

## ① 案例介绍（问题背景）

TensorDEM 的 `src/tensordem/` 里同时存在 **同名模块文件与包目录**：

```
src/tensordem/ice_structure_contact.py      (779 行, m5754 接触压力统计/多面体滚动代理)
src/tensordem/ice_structure_contact/        (1095 行, m9819 冰-结构切片接触面)
```

Python 的 `FileFinder` 先判「目录 + `__init__.py`」为 regular package，再判 `X.py`。
因此包目录一旦落地，**同名模块文件即变为不可达**：它定义的每个符号都静默消失，
既不报错、也不在任何测试里变红——只是 `tests/test_ice_structure_contact.py`
在 **collection 阶段** `ImportError`，整份 29 案的验证面从未被执行过。

审计（m14487，F3）已把这类缺陷列为「模块/包同名遮蔽」，并指出：
`freeze_bond`（21 符号）、`ice_structure_contact`（15 符号）、`shape_representation`（6 个公共名）
三处同源。m14449 / m14450 曾各修一处，但**两个修复 commit 都只留在 `agent/m14449`、
`agent/m14450` 分支上，从未合入 main**（`git merge-base --is-ancestor` 均为 NO），
其中 `agent/m14450` 的 `rolling_proxy.py` 与 main 上遗留的 `ice_structure_contact.py`
**sha256 完全相同**（`9026674c…`），即「修好了但没落地」。

参数/规模表（客观量）：

| 项 | 值 |
|---|---|
| 遮蔽对（main @ `44229ca8` 实测） | 6 对：freeze_bond / ice_structure_contact / segregation / sensitivity / shape_representation / verification(=命名空间目录, 非真遮蔽) |
| 修复前 unresolved test imports | **15**（全部在 `tensordem.ice_structure_contact`） |
| 修复前不可收集测试模块 | 1（`tests/test_ice_structure_contact.py`，29 案） |
| 修复前被遮蔽但无人 import 的符号 | shape_representation 15（6 公共 + 9 私有助手）、sensitivity 1（`main`）、segregation 1（`_damping_from_restitution`） |
| 修复前全仓 collect-only | 8669 案 + **1 个 collection error**（m14450 commit message 记录） |

## ② 怎么做的（方法与工具链）

1. **重放丢失的修复**：`git mv src/tensordem/ice_structure_contact.py → ice_structure_contact/rolling_proxy.py`
   （文件字节不变，R100），在包 `__init__.py` 中显式 re-export 16 个契约符号 + 5 个测试直接引用的助手，
   旧导入路径 `tensordem.ice_structure_contact.ContactPressureField` 保持不变。
2. **同法修 shape_representation**：`git mv shape_representation.py → shape_representation/surface_deviation.py`，
   用 AST 抽取其顶层公共名并程序化生成 re-export 块（28 个名字，其中 15 个是 main 上原先不可达的）。
3. **清掉陈旧重复体**：`git rm src/tensordem/freeze_bond.py`（与 `freeze_bond/timco_johnston.py` 仅差
   一段说明性 docstring）、`git rm src/tensordem/segregation.py`（与 `segregation/legacy.py` **sha256 完全相同**
   `5ca59b5e…`）。二者都是 m14449/更早修复留下的「第二真相源」。
4. **补 sensitivity.main**：`from .resolution import main` re-export，恢复历史 CLI 导入路径。
5. **上闸门**（审计 §8 建议 3）：`tools/import_contract_scan.py` 新增 `scan_shadows()`——
   结构性检测 `X.py` 与 `X/` 并存、用 AST 求 legacy 模块公共名、逐个 `hasattr` 判定是否被包 re-export，
   并按 sha256 识别「stale duplicate」。**任一公共符号丢失即 exit 1**，成为合并门前置。

## ③ 计算结果（数字表，程序化注入）

| 指标 | 修复前 | 修复后 | 判据 |
|---|---|---|---|
| `tools/import_contract_scan.py` unresolved test imports | 15（1 模块） | **0**（0 模块） | 硬闸门 |
| shadow collisions / lost symbols | 6 对 / 17 个丢失 | **1 对 / 0 丢失** | 硬闸门 |
| `pytest --collect-only -q`（全仓） | 8669 案 + 1 error | **9257 案 + 0 error**（52.5 s） | 0 collection error |
| `tests/test_ice_structure_contact.py` | collection error | **29 passed / 29**（7.05 s） | 全绿 |
| 陈旧重复体删除 | — | freeze_bond.py 849 行、segregation.py 743 行（≈1,592 行第二真相源） | sha 证据 |
| 恢复可达的代码 | — | ice_structure_contact 779 行 + shape_representation 1100 行 | R100 rename |

图 1（`runs/m14938/fig1_shadow_debt_before_after.png`）：左=各包被遮蔽符号数 before/after；
右=全仓可收集测试数 8669→9257（+588，含原先因遮蔽不可达而无法收集的模块）。

## ④ 结果图

见 `runs/m14938/fig1_shadow_debt_before_after.png`（左：符号丢失数逐包归零；右：collect-only 计数）。
机器工件：`artifacts/import_contract/broken_imports.json`（含 `rows` 与 `shadow_collisions.rows` 全量明细）、
`runs/m14938/shadow_debt.json`。

## ⑤ 注意点（坑与局限）

- **「修好了但没合」是本次真实根因**：m14449/m14450 的 commit 只存在于 agent 分支；
  main 上同时留着它的产物副本（`timco_johnston.py`）与旧文件（`freeze_bond.py`），
  说明「push-merge / merge-master 混合流程」会把同一修复落地到一半。→ 建议：合并门禁跑本闸门，
  否则下一次同名文件还会静默吞掉能力。
- `src/tensordem/verification/` 只有 `granular/` 子包而无 `__init__.py`，属**命名空间目录**而非遮蔽，
  闸门不计入；但 `verification.py` 文件优先于命名空间目录解析，任何 `tensordem.verification.granular`
  导入都会失败 —— 现仓内无此导入，记为**潜在**风险而非已修项。
- 只做「可达性」修复，**未改任何物理/数值实现**：`rolling_proxy`/`surface_deviation` 内容字节不变（R100）。
- `tests/test_freeze_bond.py`、`tests/test_segregation.py` 等重负载套件本地 120 s 单次预算跑不完，
  已递交集群 `cj963`（stage_dir=本工作树）；截至落笔 `queued`，故这 3 个文件**记为 not_run**，不计为通过。

## ⑥ 误差判定（关键量 vs 基准）

| 关键量 | 基准（审计/主线实测） | 本次实测 | 误差 | 是否 <3% |
|---|---|---|---|---|
| 不可收集测试模块数 | 审计 m14487：1 | 0 | 绝对差 1 → 相对 **−100 %** | 是（判据=0，非误差判据） |
| unresolved test imports | m14487 目标：0 | 0 | 0 | 是 |
| 被遮蔽符号数（ice_structure_contact） | m14450 分支声称 16+5 全 re-export | 20 个全部 `hasattr` True | 0 | 是 |
| 可收集测试数 | m14450 记录 8669 | 9257（+588） | 计数非物理量，不可用相对误差 | 不适用 |
| `test_ice_structure_contact` 通过率 | 29 案 | 29/29 | 0 | 是 |

**对「是否 <3%」的显式回答**：本任务为结构/可达性缺陷修复，**无物理量基准**；
上述关键量均为计数与布尔可达性判据，取值为精确 0/全绿，不以相对误差评价。

---

## 复现命令

```bash
PYTHONPATH=src python3 tools/import_contract_scan.py --json artifacts/import_contract/broken_imports.json  # exit 0
PYTHONPATH=src python3 -m pytest --collect-only -q | tail -1        # 9257 tests collected
PYTHONPATH=src python3 -m pytest tests/test_ice_structure_contact.py -q   # 29 passed
```