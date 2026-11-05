# PR #168 (agent/m2910) 三方合并治愈报告：fracture_gen 契约恢复 vs 主干断层/质量核算

本文件是任务 m3831 的交付物记录：把 `agent/m2910`（PR #168）与 `main` 做三方语义并集合并，
逐文件解冲，不丢任一侧内容，并用可复现的测试与基准脚本给出证据。

## 1. 合并几何（可复算）

| 项 | 值 |
|---|---|
| merge-base | `8cc88f7` (Merge pull request #137 from cxs503/agent/m2626) |
| main 侧 | `e12a2ab` (Merge #203 (m3317) via merge-governor)，自 base 起 92 文件 / +21912 行 |
| m2910 侧（PR #168） | `4203559`，自 base 起 7 文件 / +825 行 |
| 冲突文件 | 4 个（下表）；其余 3 个为 m2910 单侧新增/修改，自动并入 |
| 合并结果 | `be2b16d`（父：`e12a2ab` + `4203559`），分支 `agent/m3831` |

两侧“撞车”的根因：`fracture_gen`（FEM 失效面 → DEM 冰场）这套契约缺陷修复被两条线**各自独立**实现了一遍。
主干后来长成“超集”（footprint/face_extent、质量/能量核算、图示基准），m2910 侧带来的是
**描述性命名体系**（`cell_volume` / `equal_volume_spheres`）、公开 `lattice_volume()`、
`absolute_relative_error` 键、**精确平面交点**的 footprint 判据，以及独立审计脚本与说明。
机械 overlay 任何一侧都会静默丢内容，因此逐文件做语义并集。

## 2. 逐文件解法

| 文件 | 冲突 | 解法（并集，谁也没丢） |
|---|---|---|
| `src/tensordem/fracture_gen.py` | 双方都新增了 `mass_mapping` 别名表、`face_extent`、`on_plane_tolerance`、`mass_accounting()`、能量解析基准 | 以主干超集为骨架，追加 m2910 独有内容：① `MASS_MODE_TO_MAPPING`（报告名 `cell_volume` / `equal_volume_spheres`）与 `VALID_FACE_EXTENTS`；② 未显式指定拼写时 `FractureField.mass_mapping` 报**规范描述名**（m2910 契约），显式指定则回显调用者拼写（主干契约）；③ `FractureLedger.lattice_volume()`（= `field.represented_volume`，`N h³`）；④ `mass_accounting()` 同时给出 `relative_error`(有符号) / `relative_error_magnitude`(主干) / `absolute_relative_error`(m2910) 与 `continuum_volume` / `domain_volume` 别名；⑤ footprint 圆盘判定点参数化 `footprint_reference="midpoint"`（主干默认，**主干数值零漂移**）/ `"crossing"`（m2910 的精确平面交点 `t=-d_i/(d_j-d_i)`），并对非法值抛 `ValueError` |
| `benchmarks/fracture_gen_mass_footprint_audit.py` | 冻结映射拼写 `equal_volume_sphere`(主干) vs `equal_volume_spheres`(m2910) | 取主干版（超集：含 m2910 没有的 `plot_footprint_field_map()` 场量图）；拼写差异已在别名表中并存，且新增别名等价测试钉住；m2910 的删除属于“未新增”，不构成内容丢失 |
| `tests/test_fracture_gen_audit.py` | `relative_error_magnitude`(主干) vs `absolute_relative_error`(m2910) 断言 | 两条断言都保留，并加 `assertEqual(magnitude_a, magnitude_b)` 防漂移；再加一条“已提交审计报告同时含有符号值与两种幅度键”的测试 |
| `tests/test_fracture_gen_mass_footprint.py` | 冻结映射拼写；m2910 追加“默认映射即体积一致”测试 | 主干版 + m2910 的 `test_default_mapping_is_volume_consistent`（默认 `mass_mapping=="cell_volume"`、`relative_error==0`）；另加别名等价测试（`sphere↔equal_volume_spheres`、`cell↔lattice_cell`、`cell_volume↔volume_cell`）与 `footprint_reference` 双判据测试 |
| `benchmarks/fracture_gen_audit.py` | 隐性冲突：主干把 `acc["relative_error_magnitude"]` 写进 `relative_error` 列（正值），m2910 的已提交报告是**有符号**负值；合并后若照抄任一侧，另一侧发布表就不可复现 | 报告行同时写 `relative_error`(有符号，m2910) + `relative_error_magnitude` + `absolute_relative_error`(主干)，重新生成 `benchmarks/fracture_gen_audit_report.json`；绘图脚本按纵轴标签 `|m_DEM-ρV|/ρV` 取幅度并带 `abs()` 回退 |
| `benchmarks/fracture_gen_audit_report.json` | 同上（auto-merge 取了 m2910 侧，与主干脚本不一致） | 由脚本重新生成，成为两侧键的**并集**（有符号值保留，幅度键新增），与脚本输出逐字节一致 |
| `docs/fracture_gen_contract_restoration.md`、`scripts/fracture_gen_contract_report.py` | 无冲突（m2910 单侧新增，368 行脚本） | 原样保留；端到端执行通过，产出 `artifacts/fracture_gen_contract_restoration/{summary.json,tables.md,fig_*.png}` |

## 3. 计算结果（程序化产出：`artifacts/fracture_gen_contract_restoration/summary.json`）

参考晶格：r = 0.125 m、h = 2r = 0.25 m、域 (0,2)³ → n = 8 层、N = 512 粒子、1344 近邻键；
ρ = 917 kg/m³，连续体体积 8 m³ → 连续体质量 7336 kg。

| 质量映射 | DEM 质量 (kg) | 有符号相对误差 | 幅度 | 体积分数 | <3% ? |
|---|---|---|---|---|---|
| `equal_volume_spheres`（冻结 SI-2） | 3841.120618 | −0.476401 | 47.640 % | π/6 = 0.523599 | ✗ |
| `cell_volume`（体积一致） | 7336.000000 | 4.3e−16 | ≈0 % | 1.000000 | ✓ |

平面判据（解析 = 2n² = 128 / 层间 64）：

| 工况 | 测定断键 | 解析 | 相对误差 |
|---|---|---|---|
| 平面位于层间（冻结判据） | 64 | 64 | 0 % |
| 平面与粒子层重合（严格符号） | 0 | 128 | −100 % |
| 同上 + `on_plane_tolerance=1e-12` | 128 | 128 | 0 % |

footprint 圆盘判据（冻结无限平面 64 键，解析 πR²/h²，精确枚举为离散真值）：

| R (m) | 面积 (m²) | πR²/h² | 精确枚举 | footprint 测定 | 冻结平面 | vs 解析 | 过断倍数 |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.19635 | 3.142 | 4 | 4 | 64 | +27.3 % | 16.00× |
| 0.50 | 0.78540 | 12.566 | 12 | 12 | 64 | −4.5 % | 5.33× |
| 0.75 | 1.76715 | 28.274 | 32 | 32 | 64 | +13.2 % | 2.00× |
| 1.00 | 3.14159 | 50.265 | 52 | 52 | 64 | +3.5 % | 1.23× |
| 1.25 | 4.90874 | 78.540 | 64 | 64 | 64 | −18.5 % | 1.00× |

拓扑构建（向量化 vs 暴力）：

| n/轴 | 粒子 | 键 | 解析 3n²(n−1) | 与暴力一致 | 暴力 (s) | 向量化 (s) | 加速 |
|---|---|---|---|---|---|---|---|
| 4 | 64 | 144 | 144 | 是 | 0.098 | 0.0258 | 3.8× |
| 6 | 216 | 540 | 540 | 是 | 0.981 | 0.1162 | 8.4× |
| 8 | 512 | 1344 | 1344 | 是 | 58.10 | 0.6264 | 92.8× |

## 4. 误差判定（显式回答「是否 <3%」）

- 质量核算（`cell_volume`）：4.3e−16 → **是**（相对误差 <3%，机器精度级）。
- 质量核算（冻结 `equal_volume_spheres`）：47.640 % → **否**，这正是被审计的冻结契约缺陷本身（1−π/6），非实现误差。
- 平面判据层间 / 带容差层重合：0 % → **是**；严格符号下的层重合：−100 % → **否**（＝已知缺陷，已有容差修复路径）。
- footprint 断键数 vs 连续体解析 πR²/h²：R = 0.25 m 时 27.3 %、R = 1.25 m 时 −18.5 % → **否（>3%）**；
  但同一 5 个半径上「footprint 测定 − 精确枚举」恒等于 0 → 离散模型本身**精确**，误差全部来自
  “连续体 πR²/h²” 这一近似在 R ≈ h 量级的量子化偏差（±半个键长的边界效应），属**判据适用范围**问题而非算法误差。
- 拓扑：与暴力最近邻结果逐一相同（0 差异），解析键数 3n²(n−1) 全部命中 → **是**。

## 5. 测试证据

```
PYTHONPATH=src pytest tests/test_fracture_gen.py \
                  tests/test_fracture_gen_audit.py \
                  tests/test_fracture_gen_footprint_breakage.py \
                  tests/test_fracture_gen_mass_footprint.py \
                  tests/test_fracture_gen_mass_mapping.py -q
→ 16 passed / 17 passed / 17 passed / 19 passed / 16 passed  (合计 85 passed, 0 failed)
```

- 主干基线：合并前主干这 5 个文件为 82 项（含 1 项 `test_inclined_face_footprint_is_measured_in_plane` 在
  “交点判据”原型下会红）；本合并后 85 项全绿 = 基线 82 + m2910 独有 2 项 + 新增并集守护 1 项，
  **不低于 main 基线**。
- `scripts/fracture_gen_contract_report.py` 端到端跑通（生成 summary.json + 3 张图）。
- `benchmarks/fracture_gen_audit.py` 重新生成报告后与提交版逐字节一致（`git diff` 空）。

## 6. 注意点 / 局限

1. **两类 footprint 判据不可互换**：倾角 45° 的面，`midpoint` 与 `crossing` 会给出不同断键集合
   （实测 4 vs 6，两者都不含对方全集）。因此没有“选一个当默认”的合并方式——必须保留两条路径：
   主干默认 `midpoint`（数值零漂移，`docs/dem3d_fracture_footprint_breakage.md` 的图与断言不变），
   m2910 的精确交点作为 `footprint_reference="crossing"` 可用且有其几何自证测试。
2. **审计报告的有符号/幅度键**：主干脚本曾把幅度写进 `relative_error` 列，m2910 的提交报告是负值。
   合并后两者并存；绘图纵轴是幅度，因此绘图取 `relative_error_magnitude`。
3. **无法本机推送**：安全策略拦截 `git push`（`REJECTED: 命令被安全策略拦截`），
   因此 `agent/m2910` 的远端 SHA 仍为 `4203559`；本合并结果 `be2b16d` 落在 `agent/m3831`，
   由平台自动推送。若 merge_governor 只在 `agent/m2910` 上复验，需要把 `agent/m3831`
   的快进落到 `agent/m2910`（其 HEAD 已内含 `main` 全部历史，合并即为线性化结果）。
4. 未做全仓库测试（全量套件 >120 s/次，工具单次超时），只覆盖与本次合并相关的 5 个测试文件 +
   2 个基准脚本；其余仓库面未回归，属未完成部分。