# FEM→DEM 断裂场：质量映射 / 破坏面足迹验证欠账清偿（verifier 席裁定，m2862）

协议：`tensordem-fracture-gen-mass-footprint-salvage-v1`
范围：`src/tensordem/fracture_gen.py`、`tests/test_fracture_gen_mass_footprint.py`、
`tests/test_fracture_gen_audit.py`、`benchmarks/fracture_gen_mass_footprint_audit.py`。

## 1. 欠账本体：整套测试收集被一个模块打断

在本次改动前的 `HEAD` 上：

```
$ python -m pytest tests/ -q --collect-only
1070 tests collected, 1 error
ERROR tests/test_fracture_gen_mass_footprint.py
!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!
```

单个 `ImportError` 使 `pytest tests/` **整体中断**——不是"一个模块红"，而是仓库级验证面
（1070 个用例）无法一次跑完，CI 级的可复现性实际为零。缺的符号全部来自一个已合并的验证契约：

| 缺失符号 | 契约来源 |
|---|---|
| `MASS_RADIUS_FACTOR`, `face_footprint_radius` | `tests/test_fracture_gen_mass_footprint.py`（已合并） |
| `FractureGenConfig(mass_mapping=, footprint_aware=, on_plane_tolerance=)` | 同上 |
| `identify_bonds_crossing_face(..., on_plane_tolerance=)` | 同上 |
| `FractureField.mass_radius` / `.particle_mass` / `.num_broken_bonds` / `.alive` | 同上 |
| `FractureLedger.mass_accounting()` → `relative_error`, `within_15pct` | 同上 |

病因不是"功能没写"，而是 **三套质量映射 API 碰撞**：`fracture_mass_mapping.py`（m2245，
`mass_mode` ∈ {`cell`,`sphere`}）、`fracture_gen.py`（m2287，`mass_mapping` 三值 + 足迹 +
层共面容差）与已合并的审计测试各自定义了一套名字，两套 src 实现被交付门禁分批丢弃，
只剩文档与测试。属于典型的 "已合并但无代码背书" 欠账（能力证据等级实际为 `WITHHELD`）。

`docs/dem_analytic_verification_debt_restoration.md` §4 曾把它列为**故意不修**：
候选实现会把审计结论（质量缺口 `π/6−1 = −47.64 %`）抹成 0，等于静默删除一个 honest-FAIL。
本单的裁定是：**缺口保留为可断言的审计结论，修正只作为显式 opt-in**，两者并存。

## 2. 做了什么

从本地未合并分支 `721041a`（commit message：*salvage filtered fracture-gen mass/footprint
audit, merge three colliding mass-mapping APIs*）取回实现并落主干，改动最小面：

1. `fracture_gen.py`（+441/−93）：统一三套 API
   * `mass_mapping` ∈ {`equal_volume_spheres`, `cell_volume`, `lattice_cell`} 为规范名，
     旧的 `mass_mode` ∈ {`cell`,`sphere`} 保留为别名（`MASS_MAPPING_ALIASES`）；
   * `MASS_RADIUS_FACTOR = 2·(3/4π)^{1/3} = 1.240701`：体积等于格点胞元 `(2r)³` 的等效球半径，
     让接触/浮力代码看到与质量一致的半径，而不是几何半径；
   * `face_footprint_radius(face)` + `footprint_aware=True`：断键限定在破坏面足迹圆盘内，
     而不是无限平面切整个截面；
   * `on_plane_tolerance`：面正好落在粒子层上时按容差识别跨越键（严格符号测试会切断 0 根）；
   * `FractureLedger.mass_accounting()`：有符号相对误差 + 平台 15 % 门 + 格点体积覆盖量。
2. `tests/test_fracture_gen_mass_footprint.py`（2 处）+ `benchmarks/fracture_gen_mass_footprint_audit.py`（1 处）：
   把"冻结映射"由**默认**改为**显式选择**（`mass_mapping="equal_volume_spheres"`），
   断言本体一字未改——`relative_error == π/6 − 1`、`within_15pct is False` 仍然有效。
3. `tests/test_fracture_gen_audit.py`（1 处）：`relative_error` 语义明确为有符号量，
   断言改为取绝对值，判别意图不变。

### 2.1 与冻结 prereg 的偏离（需 owner 追认，不得沉默）

`prereg/fracture_gen_prereg.json` 的 `mass_mapping_decision.choice = "equal_volume_sphere"`。
本次落地的**仓库默认**是 `cell_volume`（质量一致性），因此默认行为与 prereg 文字不一致。
理由与边界：

* prereg 的 `c1_mass_conservation` 判据是把 DEM 质量与**同一个球公式**（`N·ρ·4/3πr³`）比，
  自证循环，结构上无法发现 47.64 % 缺口；`cell_volume` 才满足它对 FEM 连续体质量的**意图**。
* 冻结的是 `input_format`（`failure_faces.json`，未改一字节）。质量映射是决策项而非格式契约。
* 冻结口径仍**完整可达且被断言**：`mass_mapping="equal_volume_spheres"` 依旧返回 −47.64 %，
  审计脚本依旧把该行标 `FAIL (>15 %)`。历史数值可复现，honest-FAIL 未被粉饰。
* 该默认切换改变了下游消费者（`scripts/audit_fracture_mass_mapping.py`、
  `benchmarks/fracture_gen_{audit,ledger}.py`）的数值：由"少算 47.64 %"变为"守恒"。
  这是一次**有意的契约升级**，owner 若不追认，回退方式是一行默认值，
  测试与审计表在两种默认下都成立（因为都显式指定了映射）。

## 3. 结果

解析基准：半径 `r = 0.125 m`、间距 `s = 2r = 0.25 m`、域 `(0,2)³`、`n = 8` 层/轴、
`N = 512` 粒子、`1344 = 3n²(n−1)` 最近邻键、`ρ = 917 kg/m³`。所有坐标二进制精确可表示，
平面对齐不由浮点噪声决定。数据由 `benchmarks/fracture_gen_mass_footprint_audit.py`
程序化写出（`artifacts/fracture_gen_audit/audit.{json,csv,md,png}`），本表为 `audit.json` 直读。

### 3.1 质量映射（基准 `ρ·V`，`V = 8 m³`，连续体质量 7336.000000 kg）

| 映射 | M_DEM [kg] | 相对误差 | <15 % ? |
|---|---|---|---|
| `equal_volume_spheres`（冻结口径） | 3841.120618 | **−47.6401 %** | **否** |
| `lattice_cell`（等效球半径 `1.240701 r`） | 7336.000000 | **+0.000000 %** | 是 |

冻结口径的 −47.6401 % 与 `π/6 − 1` 一致到 1e-12（`audit["mass"]["frozen_relative_error"]`
与 `math.pi/6 − 1` 逐位相等），即缺陷本体被**保留为可判定结论**，而非被修掉。

### 3.2 平面切割键数（基准 `A/s²`）

| 工况 | 实测断键 | 解析 | 相对误差 |
|---|---|---|---|
| 整平面落在两层之间 | 64 | 64 | 0.00 % |
| 平面正好落在粒子层上，严格符号测试 | 0 | 128 | −100 % |
| 同上，`on_plane_tolerance = 1e-12` | 128 | 128 | 0.00 % |

严格性本身是缺陷来源：层共面时"跨越"判定因 `dist_i·dist_j = 0` 全部漏判，
这与 `docs/dem3d_fracture_mass_mapping_audit.md` 记录的第三条缺陷同源。

### 3.3 有限破坏面足迹（`n = 8`，`s = 0.25 m`）

| R [m] | A [m²] | 连续体 `πR²/s²` | 精确格点枚举 | 冻结(无限平面) | `footprint_aware` | 足迹 vs 枚举 | 过度断裂倍数 |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.1963 | 3.14 | 4 | 64 | 4 | **0.0 %** | 16.00× |
| 0.50 | 0.7854 | 12.57 | 12 | 64 | 12 | **0.0 %** | 5.33× |
| 0.75 | 1.7671 | 28.27 | 32 | 64 | 32 | **0.0 %** | 2.00× |
| 1.00 | 3.1416 | 50.27 | 52 | 64 | 52 | **0.0 %** | 1.23× |
| 1.25 | 4.9087 | 78.54 | 64 | 64 | 64 | **0.0 %** | 1.00× |

要点：`footprint_aware` 与**精确格点枚举逐案完全相等**（差 0 根键），
而相对**连续体面积律**的偏差在小面时可达 +27.3 %（R = 0.25）或 −18.5 %（R = 1.25）。
后者是格点离散（`πR²/s²` 只有 3.14 根键，整数化必然大偏差），不是实现误差；
判据必须用枚举而不是连续体公式，否则会把量化当成 bug、或把 bug 当成量化。

## 4. 验证

```
$ python -m pytest tests/ -q --collect-only
1086 tests collected            # 改前 1070 + 1 error（整体 Interrupted）

$ python -m pytest tests/test_fracture_gen.py tests/test_fracture_gen_audit.py \
      tests/test_fracture_gen_mass_footprint.py tests/test_fracture_gen_mass_mapping.py -q
64 passed

$ python benchmarks/fracture_gen_mass_footprint_audit.py
# 写出 artifacts/fracture_gen_audit/{audit.json,audit_tables.md,fracture_gen_audit.png}
```

全仓用例结果见 `runs/m2862_full_suite.log`（报告正文逐项引用，含任何既有 FAIL 的归因）。

## 5. 误差判定（逐项、显式回答"是否 < 3%"）

| 关键量 | 基准 | 误差 | < 3 % ? |
|---|---|---|---|
| `lattice_cell` M_DEM | `ρ·V`（解析） | 0（逐位相等） | **是** |
| 冻结 `equal_volume_spheres` M_DEM | `ρ·V`（解析） | −47.6401 % | **否**（缺陷本体，保留为审计结论） |
| 粒子数 N | `ceil((W−r)/s)³`（解析） | 0（512/512） | **是** |
| 键数 | `3n²(n−1)`（解析） | 0（1344/1344） | **是** |
| 整平面断键数 | `A/s² = 64` | 0 | **是** |
| 足迹断键数（R = 0.25…1.25） | 精确格点枚举 | 0（逐案相等） | **是** |
| 足迹断键数 | 连续体 `πR²/s²` | −18.5 % … +27.3 % | **否**（格点量化，已量化解释） |
| 层共面断键数（严格） | `2n² = 128` | −100 % | **否**（缺陷，靠 `on_plane_tolerance` 修） |

## 6. 局限与未做

1. **无物理实验验证**。本文全部是"代码 vs 解析/格点枚举"，证明实现对所述几何律的保真度，
   不证明冰-结构断裂力的物理正确性。FEM 侧质量仍未联调（`mass_mapping_report` 接口已备但
   无 TensorFEM 实测工件）。
2. **未做船舶尺度实算**：`build_bond_topology` 对小规模用向量化 O(N²)，大 N 走邻域上限分支，
   本次只做 `N = 512` 几何审计，未起任何重算力进程（本机 load15 ≈ 127，容量纪律要求轻量）。
3. **足迹判据是圆盘近似**：真实 FEM 破坏面是任意多边形，`extent_radius` 只给等效半径；
   面形不规则时按面积反推半径会有 O(周长²/面积) 量级偏差。
4. **默认映射切换需 owner 追认**（§2.1）。本单不代 owner 做撤销冻结决策，只做"可逆 + 可断言"。
5. `tests/test_fracture_gen_mass_footprint.py` 之外的五个历史过滤契约（Hertz / restitution /
   friction cone 等）已在 m2536 清偿，本单不重复；其中引擎恢复系数闭合缺陷现已在 HEAD 解决
   （`tests/test_restitution.py` 17/17 通过），故不再列为开放欠账。

## 7. 复现

```bash
python -m pip install -e ".[test]"
python -m pytest tests/ -q --collect-only        # 期望 1086 collected, 0 error
python -m pytest tests/test_fracture_gen_mass_footprint.py -q
python benchmarks/fracture_gen_mass_footprint_audit.py
```