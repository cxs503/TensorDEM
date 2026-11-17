# REGISTRY · m24148 · TensorDEM capability 注册表（DG-ANNOTATE 范式）

生成：`scripts/registry_report_m24148.py`（三模式实跑，数字程序化注入，零手抄）· 2026-11-17 02:36 UTC
基准 commit：`8bee9d3bc4c2277747e87a9d51a61a9833878106`

## 1. 范式（TensorDG PR#807 / 双试点契约 m23186+m23215）

- 每个真实能力在 `src/tensordem/capabilities.py` 以 `@capability(...)` 注记（微服务式自注册，导入即登记）；
- `registry.py` 三模式，stdout 恒为**单 JSON 对象** `{source_commit, capabilities, summary, problems}`；
- `--static` = AST 扫描：不导入包、不依赖 torch，支持 file-path 直跑（平台宿主可能无 torch）；
- 反过度声明门（客户端镜像平台 422 门）：mature 需 verified+wired；planned 需 feature_ledger 锚；endpoint 需 wired+`cli:`/`route:`/`api:` 前缀；verified 需 ANALYTIC/XVALUE 级证据；
- 指针（module/entry/test_path）解析失败 = `problems` 诚实失败通道，不静默丢弃。

## 2. 三模式实跑（本工作树）

| 模式 | 命令 | rc | total | problems | 备注 |
|---|---|---|---|---|---|
| static | `python src/tensordem/registry.py --static` | 0 | 376 | 0 | 单 JSON，json.loads 通过 |
| import | `python src/tensordem/registry.py` | 0 | 376 | 0 | 导入即注册 |
| parity | `python src/tensordem/registry.py --parity` | 0 | static=376 import=376 | 0 | parity=True，diffs=0 |

## 3. 条目统计（static 模式）

- total = **376**（mature=35 developing=341 planned=0）
- wired=375 tested=284 verified=49 endpoints=5 problems=0
- 本 PR 增量：`dem.brash_ice_rheology`（kind=ice，maturity=developing，wired+tested，verified=false 如实——fullscale_validation 为三方对比但不在生成器机器判据内，不虚标）

## 4. 反过度声明门与指针（static 模式独立复算）

- 门违例 = 0；指针违例 = 0；合计 problems = 0

## 5. 冷验证（99 镜像语义：torch-free、无包安装、file-path 直跑）

| 环境 | 命令 | rc | total | problems |
|---|---|---|---|---|
| 本机 clean-room（`python3 -S`，无 site-packages、无 .git 裸导出） | `python3 -S src/tensordem/registry.py --static` | 0 | 376 | 0 |
| 集群节点（loong64，无 tensordem/torch，stage_dir 直传工作树） | `python3 src/tensordem/registry.py --static` | 见 `runs/cold_cluster/`（cj4893, n330） | 376 | 0 |

## 5b. 测试

- 模块测试（brash/fem_dem_fracture 四文件）：`pytest tests/test_brash_ice_rheology_mu_i_resistance.py tests/test_dem_brash_ice_fullscale.py tests/test_fem_dem_fracture_handoff.py tests/test_fem_dem_fracture_dynamic.py` → **110 passed, 1 skipped**
- 注册表自身测试：`tests/test_capability_registry.py` 全 18 例通过（16 进程内本机 + 2 CLI 子进程集群 n315 补跑，cj4910，`runs/cold_cluster/registry_cli_tests.txt`）

## 6. 修复记录

1. **brash_ice_rheology 不可见**：目录缺 `__init__.py` → 生成器（子包判据=有 `__init__.py`）跳过 → 注册表漏 1 个真实能力。修复：补 `__init__.py`（26 符号再导出），再生成 capabilities.py（375→376）。
2. **裸导出指针误报**：无 `.git` 时 `resolve_pointers` 回退 `pkg_root.parent`，src/ 布局下 284 个 test_path 全部误报 `pointer.test_missing`。修复：无 `.git` 时接受 flat 与 src/ 双布局（`alt_root`），git 检出下行为不变。

## 7. 已知局限（如实）

- `verification/` 目录（含 `granular/bulk_flow`）无 `__init__.py` 且被 `verification.py` 模块遮蔽，`tensordem.verification.granular` 不可导入；补 `__init__.py` 会翻转 `tensordem.verification` 的导入语义（模块→包），风险超出本单范围，留待专项。
- `dem.brash_ice_rheology` 的 `fullscale_validation.py` 是三方对比（DEM vs Lindqvist vs anchor），但生成器机器判据只认 `*verification*`/`*_xval`/verdict 工件，故 verified=false——不虚标。
