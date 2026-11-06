# TensorDEM 测试扩面：三个零覆盖模块的正/负例测试（m4729）

日期：2026-11-06 · 分支：`agent/m4729` · 范围：`tests/`、`scripts/`、`artifacts/m4729_negative_controls/`

本页记录一次**只加测试、不改生产代码**的扩面：先把「哪些 `src/tensordem` 模块
没有任何测试引用」用可复现的方法量出来，再给其中 3 个模块各补一个测试文件，
每个文件都同时包含正例（代数恒等式、守恒、解析参考）与负例（非法输入必须
`raise`／错误参考必须被判红），并用**源码变异注入**证明新测试在缺陷出现时会变红。

## 1. 覆盖缺口是怎么量出来的

两条独立口径，都是脚本可复现的静态口径（不依赖 coverage.py）：

1. **模块级引用扫描**：把 `tests/**/*.py` 全文拼接，对每个
   `src/tensordem/**/*.py` 检查其模块名（`x.py` 的 `x` 以及点号全名
   `tensordem.x`）是否出现。只有 `tensordem.shear_convergence` 命中 0 次
   （`__init__` / `__main__` 不计）。
2. **公共符号级扫描**：对每个模块的顶层 `def` / `class`（不含下划线开头）逐个
   检查名字是否出现在测试文本里，得到「未被任何测试提及的公共符号」比例，用来
   区分「有测试文件但只覆盖了一小部分」的模块。

口径 2 的头部结果（缺口率 = 未被提及的公共符号 / 公共符号总数）：

| 模块 | 缺口率 | 未被提及的公共符号（示例） |
|---|---|---|
| `tensordem.shear_convergence` | 2/3 | `boundary_layer_fit`, `surface_lag_profile` |
| `tensordem.threefield.coupling` | 6/10 | `ice_mass_from_sphere`, `diameter_from_mass`, `dem_inventory`, `race_velocity_at`, `thermal_stage`, `follower_ship_impact` |
| `tensordem.buoyancy_verification` | 10/18 | `vertical_acceleration`, `integrate_free_float`, `analytic_displacement`, `analytic_envelope`, `stiffness_check`, … |

据此选定本次三个目标模块。注意 `tensordem.shear_static` 里另有一套同名近亲
实现（`cross_section_series` / `fit_boundary_layer`），它已有
`tests/test_dem3d_shear_cross_section_convergence.py` 覆盖；本单补的是
`shear_convergence` 这一份，二者共用 `static_direct_shear`，因此新测试里显式做了
「series 行 == 直接静力解」的交叉校验，避免两份实现各自漂移却都「有测试」。

## 2. 新增测试文件与用例构成

| 测试文件 | 目标模块 | 用例数 | 正例要点 | 负例要点 |
|---|---|---|---|---|
| `tests/test_shear_convergence_fit.py` | `shear_convergence` | 14 | 合成 `c1/n + c2/n²` 系列的系数回归（rel 1e-9）、模型 C 零偏置、几何网格上纯 `1/n` 层序（阶数=1、Richardson 极限=1 精确）、门限预测自洽、4 尺寸 series 与 `static_direct_shear` 逐行一致、`surface_lag_profile` 的 5×5×3 滞后场 | `boundary_layer_fit` 的 6 组非法序列（点数不足/非单调/负值/NaN）、`cross_section_series` 退化尺寸集、`surface_lag_profile` 非方截面与偶数 `nz` |
| `tests/test_threefield_coupling_units.py` | `threefield.coupling` | 35 | 球质量⇄等效直径往返（rel 1e-12）、尾流剖面线性插值与端点夹持、质量账本精确闭合、DEM 清单适配、破断级质量守恒（≤1e-12 相对）、热侵蚀账本闭合、跟随船两项减阻分解 | 8 组几何输入（负值/零密度/NaN/布尔）、空剖面与负半径、账本非法项、清单长度不一致/缺键、`CouplingConfig` 8 组非法字段（含 `seed=True`） |
| `tests/test_buoyancy_verification_reference.py` | `buoyancy_verification` | 12 | 弦宽几何、浸没面积分数（`f(0)=0.5`、互补性 `f(h)+f(-h)=1`、面板无关性）、Archimedes 平衡反演、线性化振子关系 `ω_d/ω_n`、吃水处力平衡与恢复力符号、`apply_gravity` 语义、解析位移/包络、RK4 有阻尼衰减 | 密度饱和夹持（不返回假根）、参考被篡改时 `geometry_check` / `stiffness_check` 必须 `passed=False`（负控） |

合计 **61** 个用例（14 + 35 + 12），全部通过：

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
  tests/test_shear_convergence_fit.py \
  tests/test_threefield_coupling_units.py \
  tests/test_buoyancy_verification_reference.py -q
# 61 passed, 1 warning in 7.56s
```

未删除、未跳过任何既有测试，未放宽任何既有断言容差（全部改动是新增文件）。

## 3. 负例自证：源码注入缺陷必须变红

`scripts/negative_controls_m4729.py` 对三个生产模块各注入一处缺陷，跑对应测试
文件（`-q -x`），要求**红**；随后用原文恢复再跑一次，要求**绿**。完整日志见
`artifacts/m4729_negative_controls/negative_injection.log`。

| 注入缺陷 | 生产文件 | 变红的测试 | 红输出摘要 |
|---|---|---|---|
| 边界层设计变量 `x = 1/n` → `1/(2n)` | `src/tensordem/shear_convergence.py` | `test_boundary_layer_fit_recovers_the_generating_coefficients` | `E assert 2.3999999999999995 == 1.2 ± 1.2e-09` |
| 球直径指数 `1/3` → `1/2` | `src/tensordem/threefield/coupling.py` | `test_sphere_mass_diameter_round_trip` | `E assert 0.1643167672515498 == 0.3 ± 1.0e-12` |
| 水线弦宽系数 `2·√(R²−h²)` → `√(R²−h²)` | `src/tensordem/buoyancy_verification.py` | `test_chord_width_geometry` | `E assert 0.5 == 1.0 ± 1.0e-06` |

三处注入均 `exit_code=1` 且出现 `FAILED` 行；恢复后三个文件分别
`14 passed` / `35 passed` / `12 passed`。此外 `buoyancy_verification` 的两个负控
用例（monkeypatch 篡改参考后要求 `passed is False`）本身就把「门必须能抓住错误」
写成了断言，不依赖测试运行者的纪律。

## 4. 质量门现状（honest）

- 新增/改动文件的静态检查干净：

```
ruff check tests/test_shear_convergence_fit.py \
           tests/test_threefield_coupling_units.py \
           tests/test_buoyancy_verification_reference.py \
           scripts/negative_controls_m4729.py \
           scripts/plot_m4729_test_coverage.py
# All checks passed!
```

- **但仓库级 `ruff check .` 在基线就不是绿的**：本次改动前（`git archive HEAD`
  导出到临时目录）为 562 处告警，改动后工作树为 561 处，差值可追溯到工作树里
  缺失的一个基线文件（`src/tensorsph/surface_tension.py`），与本次改动无关；新增
  文件的告警数为 0。历史告警以 `E702`（一行多语句分号）、`E701`、`E402` 为主，
  分布在 `scripts/` 与若干早期 `src/` 模块。要让它变绿必须做全仓风格重排，与
  「改动最小、可验证」冲突，故本单**不**顺手做，仅如实记录：仓库级
  `ruff check .` 未通过属既有欠账，本单未增加任何新告警。

## 5. 图与数值（`scripts/plot_m4729_test_coverage.py`）

`artifacts/m4729_negative_controls/figure_metrics.json` 全部由程序注入，无手抄：

- `fig1_shear_convergence.png`：`cross_section_series([5,7,9,11,13])` 的
  `d(n)=1−G_eff/C44` 与模型 A/B/C 拟合。观测阶 2.04，模型 C 的渐近偏置
  `c0 = 4.8e-3`，外推 `G_eff(∞)/C44 = 0.995`，3 % 门限预测 `n ≈ 40.8`，
  最粗网格 `d(13) = 0.102`（仍在门外，与模块文档一致）。
- `fig2_free_float_decay.png`：`integrate_free_float`（RK4，实线）与线性化参考
  （虚线）及包络（点线）。`ω_n = 3.208 rad/s`，`ζ = 0.09996`，
  `t ≥ 4 s` 段波形最大相对误差 7.0e-4（< 3 %）。

## 6. 局限与未做

- 只做**静态口径 + 变异注入**的覆盖论证，没有引入 `coverage.py` 的行覆盖数字；
  三模块的「未覆盖分支」比例因此是上界估计而非精确值。
- 未对 `src/` 做任何修改：如果这些模块本身存在缺陷，本单只会**暴露**（红）而不会
  修复；目前没有红。
- 仓库级 ruff 欠账（561 处）保留原样，见 §4。
- `shear_convergence` 的 `cross_section_series` 只跑到 `n = 13`（3 % 门外），
  更高阶尺寸的计算成本未在 CI 预算内评估。