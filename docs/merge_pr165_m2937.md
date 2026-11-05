# PR #165 (agent/m2937) 治愈记录：三方合并 → main

| 项 | 值 |
|---|---|
| 仓库 | TensorDEM |
| PR / 分支 | #165 / `agent/m2937` |
| 合并方向 | `agent/m2937` → `main` |
| merge-base | `8cc88f7` (Merge PR #137) |
| main tip（ours） | `e12a2ab` (Merge #203 via merge-governor)，领先 base 29 个提交 |
| 分支 tip（theirs，旧） | `ba71cad` (m2937: matured mission deliverables)，领先 base 2 个提交 |
| 合并结果 | 本地合并提交（双亲 `e12a2ab` + `ba71cad`；工作分支 `agent/m3800`，由平台推送） |
| 冲突文件 | 5（4 个内容冲突 + 2 个 add/add，其中 1 个文件同时计） |
| 机械 overlay | 未使用（全程逐文件语义并集） |

## 1. 这份 PR 在做什么

`agent/m2937` 交付的是**冰脊接地/海底管道地基承载力链的参考值校核**：把
`tensordem.ridge_geotech` 的承载力公式（Prandtl 1920 滑移线、Vesic 1973 因子、
Skempton 1951 埋深因子、Meyerhof/Hansen 深度因子）逐项对标已发表表值，并修掉
「Skempton 埋深因子被重复计入（再乘一次 Meyerhof 深度因子）」的缺陷。
main 侧在 base 之后**独立地做了同一处修复**（PR #70 之后的 m2983 审计），因此两侧
在同一个函数、同一批测试、同一段文档上各自演化 → GitHub 判 `mergeable=false`。

这正是本次冲突的性质：**不是内容争夺，而是两条平行修复线的语义归并**。

## 2. 冲突文件清单与逐文件解法

| # | 文件 | 冲突类型 | 两侧内容 | 解法（语义并集） |
|---|---|---|---|---|
| 1 | `src/tensordem/ridge_geotech.py` | content ×3 | main：`skempton_undrained_nc(depth, width, *, length_ratio)`（带状/方形的连续形状因子，值域与上限同乘 `1+0.2B/L`）；branch：`skempton_nc(z_over_b, *, footing="strip"/"square"/"circular")` | **两个 API 都保留**（两侧测试各自调用其一，删任何一个都会让另一侧测试 ImportError）。`vertical_pressure_capacity` 的未排水分支保留 main 的 `skempton_undrained_nc(..., length_ratio=length_ratio)`——branch 那行会**静默丢弃** main 的 `length_ratio` 能力；同时保留 branch 关于「双重计入」缺陷的注释与文档引用。`__all__` 两侧条目由 git 自动并集为 15 项，无重复 |
| 2 | `tests/test_ridge_geotech_capacity_verification.py` | add/add | 两侧**各自新增**同名文件校验同一件事（main 10 个 test 函数 / branch 11 个） | 全文并集为 21 个 test 函数。**关键陷阱**：两侧各有 `test_undrained_capacity_is_linear_in_shear_strength`，若直接拼接，后者会**静默遮蔽**前者（Python 同名后定义生效，pytest 不会报错，只会少跑用例）。故将 main 的改名为 `test_undrained_pressure_capacity_is_linear_in_shear_strength`。同名常量 `SKEMPTON_STRIP_TABLE`（main 是 dict、branch 是 import 的 `(z/B, Nc)` 行列表）拆为 `SKEMPTON_STRIP_TABLE_2DP` 与导入的 `SKEMPTON_STRIP_TABLE`；`_undrained()` 助手统一为兼容两种调用签名的版本 |
| 3 | `tests/test_ridge_grounding.py` | content | main：`pressure == 30e3*5.14*1.2` 且 `< ...*1.2`；branch：`pressure == 30e3*6.168` + 独立基准说明 | **两条断言都保留**（`5.14*1.2 = 6.168`，同为一条独立基准的显式与交叉两种写法），语义为严格超集 |
| 4 | `docs/ice_ridge_pipeline_grounding.md` | content | 两侧各写一段未排水承载力说明，分别引用各自的审计文档 | 合为一段：保留 strip/方形两种上限口径与两个 API 名，保留两侧引用的 `docs/ridge_capacity_skempton_audit.md` 与 `docs/ridge_geotech_capacity_verification.md` 及各自数字（20-44 % 高估、43 项参考校核、最差 0.071 %） |
| 5 | `scripts/verify_ridge_geotech_capacity.py` | add/add | 两侧**各自新增**同名可执行校核脚本，但检查集不同、产物目录不同（main → `artifacts/ridge_capacity_verification/`；branch → `artifacts/ridge_geotech_capacity/`） | main 版本保留在**规范路径**（main 侧 `docs/ridge_capacity_skempton_audit.md`、`scripts/build_ridge_capacity_audit.py` 依赖该路径与产物）；branch 版本**原样保留**为新路径 `scripts/verify_subgrade_capacity.py`，并同步更新 `docs/ridge_geotech_capacity_verification.md` 中的 3 处调用行（原文档仍写旧名）。两个脚本均已实际跑通 |

非冲突一侧的新增内容全部保留（git 自动合并）：`src/tensordem/ridge_geotech_verification.py`、
`docs/ridge_geotech_capacity_verification.md`、`artifacts/ridge_geotech_capacity/*`（JSON+2 图）、
`artifacts/ridge_grounding/*`（标定/包络/账本/主算例/时程/敏感性共 11 件）。

## 3. 验证

工具链：`pytest`（`PYTHONPATH=src`）、`scripts/audit_doc_evidence.py`（仓库自带文档证据门）、
两个校核脚本直接执行。

| 检查 | 结果 |
|---|---|
| 冲突标记残留 | 0（全树 `<<<<<<<`/`>>>>>>>` 扫描） |
| AST + 重复定义扫描 | 5 个改动文件全部可解析，函数/类无重名；`__all__` 15 项无重复 |
| 合并树全量测试 | **1241 passed, 3 skipped, 152 subtests passed**（见下"失败归因"） |
| main 基线全量测试 | 1225 passed, 3 skipped, 152 subtests passed（6 errors 均为基线抽取方式导致：`git archive` 无 `.git`，`test_doc_evidence_pointers` 需读 git 索引 → 基线该模块 0 通过；合并树该模块 14/14 通过） |
| 冲突相关测试 | `test_ridge_geotech_capacity_verification.py` + `test_ridge_grounding.py` = **56 passed** |
| 文档证据门 | `tests/test_doc_evidence_pointers.py` **14 passed**（见下"失败归因"） |
| `scripts/verify_ridge_geotech_capacity.py` | 跑通：主算例 `Nc=6.730 / q=201.9 kPa / p_ult=184.5 kN/m`，pre-fix 相对误差 `0.200` |
| `scripts/verify_subgrade_capacity.py` | 跑通：账本 43 项参考校核全 `ok`，1 项 `WITHHELD`（侧向极限压力无独立文献） |

**失败归因（honest）**：合并树首次全量跑出 3 个 FAILED，全部集中在
`tests/test_doc_evidence_pointers.py`，根因是**新脚本 `scripts/verify_subgrade_capacity.py`
当时尚未进入 git 索引**（文档引用了未跟踪文件）+ `docs/doc_evidence_ledger.json` 未重算。
`git add` 新脚本并重跑 `python3 scripts/audit_doc_evidence.py`（扫描 106 文档 / 522 指针 /
92 项 REGENERABLE / 0 orphan）后，该模块 14/14 通过。这不是模型/代码回归，而是合并必须
同时刷新仓库的文档证据账本。基线对比因此按「1241 vs 1225 通过」计，合并侧净增 +16 个用例
（并集测试带来的新增覆盖），无一侧用例丢失。

## 4. 数值证据（程序化取值，零手抄）

| 量 | 数值 | 来源 |
|---|---|---|
| `skempton_nc(1.0)`（合并后） | 6.168 | `tensordem.ridge_geotech` |
| pre-fix 的 `Nc`（z/B=1） | 7.4016（相对高估 **20.0 %**） | `_legacy_nc` 复算 |
| pre-fix 最大高估 | **44.0 %**（z/B→∞，用方形上限 9.0 而非带状 7.5） | 同左，0…6 扫描 |
| 主算例 `p_ult` | 184.55 kN/m（D=0.914 m, cover 0.5 m, s_u=30 kPa） | `vertical_capacity_per_length` |
| 参考校核最大相对误差 | 7.1e-4 = **0.071 %**（43 项，全部 < 3 %） | `verify_subgrade_capacity_model` |

## 5. 注意点 / 局限

1. **add/add 文件的静默遮蔽**：两侧新增同名测试文件时，同名 test 函数不会报错，只会
   少跑用例——必须做重名扫描（本记录第 2 行）。
2. **同名脚本不是同一件东西**：两个 `verify_ridge_geotech_capacity.py` 校验集与产物树都
   不同；随便留一个都会让另一侧文档变成断链。已按「规范路径归 main、branch 版另立名」
   处理，并同步改文档。
3. **文档证据门是合并的一部分**：新增被文档引用的脚本后必须 `git add` + 重跑
   `audit_doc_evidence.py`，否则 CI 的红不是代码问题。
4. 合并未改动任何物理模型数值：两侧对「Skempton 只计一次」的判断一致，合并后
   `vertical_pressure_capacity` 的未排水分支与 main 逐位相同（`rel=1e-12` 断言通过）。
5. 侧向极限压力 `N_p = 2 + 1.5 z/D` 仍为 `WITHHELD`（无独立文献基准），合并未改变其状态。