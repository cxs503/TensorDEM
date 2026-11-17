# SALVAGE · m24148 · m24013-ghost 逐件判定

生成：`scripts/salvage_m24148.py`（sha256 程序化比对，零手抄）· 2026-11-17 02:18 UTC
基准 commit：`8bee9d3bc4c2` · 来源：`/srv/tensor/agent/salvage/m24013-ghost.tar.gz`（协调者隔离导出）

## 1. 判定标准（冻结）

- **收**：可导入 + 有实质实现（非空壳）+ main 中不存在或 main 版更弱；
- **弃**：与 main 字节一致（无新内容）/ main 版为严格超集（tarball 为旧快照）/ 垃圾；
- 模块级判定标准（任务令）：可导入 + 有实质实现非空壳。

## 2. 模块级结论（任务点名的两个模块）

| 模块 | 可导入 | 实质实现 | 判定 |
|---|---|---|---|
| `tensordem.brash_ice_rheology` | 是（10 件实现文件，`BrashHull`/`Closure`/`mu(I)`/`JopParameters`/四准则 validation/系数台账） | 是 | **收（打包修复）**：10 件实现与 main 字节一致，唯一缺口=缺 `__init__.py` 致注册表不可见；本 PR 补 `__init__.py` 使其成为真子包并进入 capability 注册表（`dem.brash_ice_rheology`） |
| `tensordem.fem_dem_fracture` | 是（8 件，`__all__`=66 符号） | 是（plate_fem/crack_network/handoff/dynamics/study_dynamic） | **弃（tarball 3 件中 1 件 SAME、2 件为旧快照）**：crack_network.py 与 main 一致；__init__.py 与 handoff.py 的 main 版均为严格超集，回退即降级 |

## 3. 逐件判定（tarball 全部 64 个文件条目；任务令「68 件」含目录条目，目录不逐件列）

| # | 路径 | 比对 | sha256[:16] (tar / main) | 判定 | 理由 |
|---|---|---|---|---|---|
| 1 | `.githooks/pre-commit` | ABSENT | 74cea16cb21fb33d / — | 弃 | 工作树本地守卫（禁 commit main，Door-A 封印 2026-11-06）；非仓库能力，且未设 core.hooksPath 时不生效；平台分支纪律已覆盖 |
| 2 | `.tmpwork/acc_prev.json` | tmpwork | a6768b6084008838 / — | 弃 | .tmpwork 垃圾（远程模块缓存/字体缓存/临时脚本），任务令明示忽略 |
| 3 | `.tmpwork/mpl/fontlist-v390.json` | tmpwork | d9431578c1859dde / — | 弃 | .tmpwork 垃圾（远程模块缓存/字体缓存/临时脚本），任务令明示忽略 |
| 4 | `.tmpwork/post2.py` | tmpwork | ba2ec3dfa6fd9238 / — | 弃 | .tmpwork 垃圾（远程模块缓存/字体缓存/临时脚本），任务令明示忽略 |
| 5 | `.tmpwork/post_p307w1_n010.py` | tmpwork | b0d6a1211d23c8e3 / — | 弃 | .tmpwork 垃圾（远程模块缓存/字体缓存/临时脚本），任务令明示忽略 |
| 6 | `.tmpwork/tmp07cuor_k/_remote_module_non_scriptable.py` | tmpwork | 8205b16956fb2648 / — | 弃 | .tmpwork 垃圾（远程模块缓存/字体缓存/临时脚本），任务令明示忽略 |
| 7 | `.tmpwork/tmp08r8og7h/_remote_module_non_scriptable.py` | tmpwork | 8205b16956fb2648 / — | 弃 | .tmpwork 垃圾（远程模块缓存/字体缓存/临时脚本），任务令明示忽略 |
| 8 | `.tmpwork/tmphs4jse1h/_remote_module_non_scriptable.py` | tmpwork | 8205b16956fb2648 / — | 弃 | .tmpwork 垃圾（远程模块缓存/字体缓存/临时脚本），任务令明示忽略 |
| 9 | `.tmpwork/tmpizsoue3c/_remote_module_non_scriptable.py` | tmpwork | 8205b16956fb2648 / — | 弃 | .tmpwork 垃圾（远程模块缓存/字体缓存/临时脚本），任务令明示忽略 |
| 10 | `.tmpwork/tmpnwz35yxp/_remote_module_non_scriptable.py` | tmpwork | 8205b16956fb2648 / — | 弃 | .tmpwork 垃圾（远程模块缓存/字体缓存/临时脚本），任务令明示忽略 |
| 11 | `benchmarks/artifacts/keel_porosity/results.json` | SAME | 9e5681e95ad32677 / 9e5681e95ad32677 | 弃 | 与 main 字节一致，无新内容 |
| 12 | `benchmarks/freeze_bond_thermal_profile/build_report_md.py` | SAME | da46d3ee0b1c520d / da46d3ee0b1c520d | 弃 | 与 main 字节一致，无新内容 |
| 13 | `benchmarks/freeze_bond_thermal_profile/front_kinetics.csv` | SAME | ac607445f7a328d5 / ac607445f7a328d5 | 弃 | 与 main 字节一致，无新内容 |
| 14 | `benchmarks/freeze_bond_thermal_profile/profiles.csv` | SAME | f63f665965ea6452 / f63f665965ea6452 | 弃 | 与 main 字节一致，无新内容 |
| 15 | `benchmarks/freeze_bond_thermal_profile/results.json` | DIFF | 6cb497e4e5a5ab2e / 67365db5eb9ca3ce | 弃 | main 版含 L4 benchmark 对象（front_vs_stefan_quasi / front_vs_neumann_exact / energy_closure / latent_heat_budget 逐项基准对比）；tarball 版无 benchmark 块且 all_passed=true 为弱判据 |
| 16 | `benchmarks/freeze_bond_thermal_profile/run_thermal_profile_validation.py` | DIFF | 5fc7be513ca6fb05 / 5c64bea9cf0d54ee | 弃 | main 版新增 build_benchmark()（成熟度门 L4 基准表）；tarball 版为无基准块的旧版 |
| 17 | `benchmarks/ice_crack_pattern_validation.py` | SAME | c846b502d79c9f34 / c846b502d79c9f34 | 弃 | 与 main 字节一致，无新内容 |
| 18 | `benchmarks/ice_rate_friction_failure_handover.py` | SAME | d3c269865f9f0088 / d3c269865f9f0088 | 弃 | 与 main 字节一致，无新内容 |
| 19 | `benchmarks/keel_porosity/run_benchmark.py` | SAME | 763c6635659fa4ab / 763c6635659fa4ab | 弃 | 与 main 字节一致，无新内容 |
| 20 | `benchmarks/lefm_ice_crack_figures.py` | SAME | 5867b6431c5b4b32 / 5867b6431c5b4b32 | 弃 | 与 main 字节一致，无新内容 |
| 21 | `benchmarks/lefm_ice_crack_validation.py` | SAME | d8a7310e3f9c0836 / d8a7310e3f9c0836 | 弃 | 与 main 字节一致，无新内容 |
| 22 | `benchmarks/p307w1_n010_pid_mdof.py` | SAME | 37c0365f05b48de6 / 37c0365f05b48de6 | 弃 | 与 main 字节一致，无新内容 |
| 23 | `benchmarks/p307w1_n010_report.py` | SAME | 8150a3049acd7bde / 8150a3049acd7bde | 弃 | 与 main 字节一致，无新内容 |
| 24 | `docs/BRASH_ICE_CHANNEL_MU_I_RESISTANCE.md` | SAME | 7a815a4454657cfa / 7a815a4454657cfa | 弃 | 与 main 字节一致，无新内容 |
| 25 | `docs/FEM_DEM_FRACTURE_HANDOFF.md` | SAME | 946dee75202aa29c / 946dee75202aa29c | 弃 | 与 main 字节一致，无新内容 |
| 26 | `docs/LEFM_ICE_CRACK.md` | SAME | c3bd37a7f835a8a0 / c3bd37a7f835a8a0 | 弃 | 与 main 字节一致，无新内容 |
| 27 | `docs/P295W1_N03_nonlocal_rheology_nscd.md` | SAME | f2b59e1106044063 / f2b59e1106044063 | 弃 | 与 main 字节一致，无新内容 |
| 28 | `docs/P307W1_N010_pid_mdof_insertion_loss.md` | SAME | 48314a14baaa17c4 / 48314a14baaa17c4 | 弃 | 与 main 字节一致，无新内容 |
| 29 | `docs/P307W1_N06_granular_solitary_waves.md` | SAME | e4009e3bdab9cf4a / e4009e3bdab9cf4a | 弃 | 与 main 字节一致，无新内容 |
| 30 | `docs/P316_N09_DEM_RESISTANCE_3D_MAPPING.md` | SAME | 8ad09e16f5bdffe7 / 8ad09e16f5bdffe7 | 弃 | 与 main 字节一致，无新内容 |
| 31 | `docs/P316_W1_N03_freeze_front_thermal_profile.md` | SAME | 7ccf20298a0560c2 / 7ccf20298a0560c2 | 弃 | 与 main 字节一致，无新内容 |
| 32 | `docs/UQ_FRICTION_REPORT.md` | SAME | 644ce937abe5327a / 644ce937abe5327a | 弃 | 与 main 字节一致，无新内容 |
| 33 | `docs/ice_material_constitutive.md` | SAME | c46743b8ae5332b7 / c46743b8ae5332b7 | 弃 | 与 main 字节一致，无新内容 |
| 34 | `docs/ice_rate_friction_failure_handover/README.md` | SAME | 85d02b97a38a6acc / 85d02b97a38a6acc | 弃 | 与 main 字节一致，无新内容 |
| 35 | `docs/ice_rate_friction_failure_handover/energy_balance.png` | SAME | e75b04c79d129e61 / e75b04c79d129e61 | 弃 | 与 main 字节一致，无新内容 |
| 36 | `docs/ice_rate_friction_failure_handover/failure_faces_geometry.png` | SAME | 6b795a145bcf353c / 6b795a145bcf353c | 弃 | 与 main 字节一致，无新内容 |
| 37 | `docs/ice_rate_friction_failure_handover/literature_validation.png` | SAME | 691e86f3ad397a74 / 691e86f3ad397a74 | 弃 | 与 main 字节一致，无新内容 |
| 38 | `docs/ice_rate_friction_failure_handover/mu_v_curve.png` | SAME | 1a46787c03009829 / 1a46787c03009829 | 弃 | 与 main 字节一致，无新内容 |
| 39 | `docs/ice_rate_friction_failure_handover/negative_control.png` | SAME | 14adf16a3c2f7d7c / 14adf16a3c2f7d7c | 弃 | 与 main 字节一致，无新内容 |
| 40 | `post_lefm_report.py` | SAME | 5e38b368638c0a4d / 5e38b368638c0a4d | 弃 | 与 main 字节一致，无新内容 |
| 41 | `post_report_p307w1_n06.py` | SAME | f6de792ab68fba80 / f6de792ab68fba80 | 弃 | 与 main 字节一致，无新内容 |
| 42 | `scripts/generate_ice_material_results.py` | SAME | e95fc57acaef8313 / e95fc57acaef8313 | 弃 | 与 main 字节一致，无新内容 |
| 43 | `scripts/generate_uq_results.py` | SAME | 9f569a7a3bedd92d / 9f569a7a3bedd92d | 弃 | 与 main 字节一致，无新内容 |
| 44 | `scripts/make_fem_dem_fracture_doc.py` | SAME | 87ab10fa6146b666 / 87ab10fa6146b666 | 弃 | 与 main 字节一致，无新内容 |
| 45 | `scripts/p307w1_n06_granular_chain.py` | SAME | f98a1dd4f7f6c3c9 / f98a1dd4f7f6c3c9 | 弃 | 与 main 字节一致，无新内容 |
| 46 | `scripts/p316_n09_campaign.py` | SAME | 8d0da9bdadbfb388 / 8d0da9bdadbfb388 | 弃 | 与 main 字节一致，无新内容 |
| 47 | `scripts/p316_n09_figures.py` | SAME | 137cb18095016d08 / 137cb18095016d08 | 弃 | 与 main 字节一致，无新内容 |
| 48 | `scripts/p316_n09_report.py` | SAME | eb2615f43d2dc8d0 / eb2615f43d2dc8d0 | 弃 | 与 main 字节一致，无新内容 |
| 49 | `scripts/post_fem_dem_fracture_report.py` | SAME | 81af237c04c2741d / 81af237c04c2741d | 弃 | 与 main 字节一致，无新内容 |
| 50 | `scripts/run_fem_dem_fracture_handoff.py` | SAME | 74132e88a2ddbd82 / 74132e88a2ddbd82 | 弃 | 与 main 字节一致，无新内容 |
| 51 | `src/tensordem/brash_ice_rheology/anchors.py` | SAME | 9e067526c2b5cb93 / 9e067526c2b5cb93 | 弃 | 与 main 字节一致，无新内容 |
| 52 | `src/tensordem/brash_ice_rheology/calibration.py` | SAME | f2fa654de16d57b2 / f2fa654de16d57b2 | 弃 | 与 main 字节一致，无新内容 |
| 53 | `src/tensordem/brash_ice_rheology/coefficients.py` | SAME | 6312f5f933698730 / 6312f5f933698730 | 弃 | 与 main 字节一致，无新内容 |
| 54 | `src/tensordem/brash_ice_rheology/figures.py` | SAME | fc9d00cb9249a201 / fc9d00cb9249a201 | 弃 | 与 main 字节一致，无新内容 |
| 55 | `src/tensordem/brash_ice_rheology/material.py` | SAME | 3470d5dcd9ffaccb / 3470d5dcd9ffaccb | 弃 | 与 main 字节一致，无新内容 |
| 56 | `src/tensordem/brash_ice_rheology/model.py` | SAME | 0b78c728a32336a7 / 0b78c728a32336a7 | 弃 | 与 main 字节一致，无新内容 |
| 57 | `src/tensordem/brash_ice_rheology/mu_i.py` | SAME | 023a9c071dabf7db / 023a9c071dabf7db | 弃 | 与 main 字节一致，无新内容 |
| 58 | `src/tensordem/brash_ice_rheology/report.py` | SAME | 99015722ebcf645a / 99015722ebcf645a | 弃 | 与 main 字节一致，无新内容 |
| 59 | `src/tensordem/brash_ice_rheology/study.py` | SAME | ec4d0a9ff97ad57b / ec4d0a9ff97ad57b | 弃 | 与 main 字节一致，无新内容 |
| 60 | `src/tensordem/brash_ice_rheology/validation.py` | SAME | 2a62238c5569471a / 2a62238c5569471a | 弃 | 与 main 字节一致，无新内容 |
| 61 | `src/tensordem/fem_dem_failure_handover.py` | SAME | 5b6fc68547b6df8a / 5b6fc68547b6df8a | 弃 | 与 main 字节一致，无新内容 |
| 62 | `src/tensordem/fem_dem_fracture/__init__.py` | DIFF | c7ab1ea2cff552cb / 031e3072de6c9978 | 弃 | main 版为严格超集：额外导出 dynamics/study_dynamic 全部符号（wave-2 动态交接）；tarball 为更早快照，回退即降级 |
| 63 | `src/tensordem/fem_dem_fracture/crack_network.py` | SAME | 9a26dcfae7b6e775 / 9a26dcfae7b6e775 | 弃 | 与 main 字节一致，无新内容 |
| 64 | `src/tensordem/fem_dem_fracture/handoff.py` | DIFF | cdfcabcca25fd576 / caf739512cb2a7f2 | 弃 | main 版支持 schema 1+2（SCHEMAS 元组，schema 2 为严格超集）；tarball 版仅认 schema 1，回退即丢 wave-2 工件读取能力 |

## 4. 本 PR 实际入库的 salvage 产物

| 文件 | 性质 |
|---|---|
| `src/tensordem/brash_ice_rheology/__init__.py` | **新增**：ghost 模块的打包修复（docstring + 26 符号再导出），使 `dem.brash_ice_rheology` 进入注册表 |
| `src/tensordem/capabilities.py` | 再生成：375 → 376 条（+`dem.brash_ice_rheology`，grade B，wired+tested，verified=false 如实） |
| `src/tensordem/registry.py` | 修复：无 `.git` 的裸导出（冷环境）下 test_path 指针解析不再误报 284 条（src/ 布局回退） |

## 5. 验证

- `python3 -m pytest tests/test_brash_ice_rheology_mu_i_resistance.py tests/test_dem_brash_ice_fullscale.py tests/test_fem_dem_fracture_handoff.py tests/test_fem_dem_fracture_dynamic.py` → **110 passed, 1 skipped**
- `python src/tensordem/registry.py --static` → rc=0，单 JSON，total=376，problems=0
- `python src/tensordem/registry.py --parity` → parity=True（376=376，diffs=0）
- 冷环境（`python3 -S`，无 site-packages、无 .git 裸导出）直跑 file-path：rc=0，total=376，problems=0
