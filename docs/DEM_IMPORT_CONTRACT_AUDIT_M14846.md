# TensorDEM 测试导入契约静态审计（m14846）

协议：`tensordem-import-contract-audit-v1`　扫描测试模块：511　静态不可解析导入：7　同名文件/包遮蔽：2

方法：对 `tests/test_*.py` 做 AST 静态解析（**不 import**），把 `from tensordem... import X` 的每个名字与目标模块（`src/tensordem/**.py` 或 `**/__init__.py`）的顶层命名空间（def/class/赋值/import 绑定）求交；缺失即「已合并的检查无法运行」。

| 测试模块 | 导入目标 | 缺失符号 | 行 | 判读 |
|---|---|---|---|---|
| test_brash_ice_rheology_mu_i_resistance.py | `tensordem.brash_ice_rheology` | <module not found> | 371 | 目标模块不存在/被同名包遮蔽 |
| test_dem_ice_contact_calibration.py | `tensordem.ice_steel_friction` | CALIBRATION_ANCHORS, VALIDATION_ANCHORS | 28 | 符号未导出 |
| test_ice_contact_strength.py | `tensordem.ice_steel_friction` | CALIBRATION_ANCHORS, VALIDATION_ANCHORS | 46 | 符号未导出 |
| test_ice_friction_thermo_physics.py | `tensordem.ice_steel_friction` | CALIBRATION_ANCHORS | 43 | 符号未导出 |
| test_ice_steel_friction.py | `tensordem.ice_steel_friction` | CALIBRATION_ANCHORS, EXTRAPOLATION_PROBES, VALIDATION_ANCHORS | 26 | 符号未导出 |
| test_ice_steel_friction_closure.py | `tensordem.ice_steel_friction` | CALIBRATION_ANCHORS, EXTRAPOLATION_PROBES, VALIDATION_ANCHORS | 25 | 符号未导出 |
| test_ice_structure_contact.py | `tensordem.ice_structure_contact` | ContactPressureField, compute_contact_pressure_field, spatial_correlation, peak_factor, polyhedron_contact_force, polyhedron_hertz_degeneracy, rolling_bias_quantification, identifiability_analysis, monotonic_scan, ks_test_seeds, run_full_verification, _polyhedron_vertices_sphere, _polyhedron_vertices_angular, _is_sphere_vertices, _estimate_angularity | 18 | 符号未导出 |

## 说明与局限

* 这是**静态**契约检查：能抓住「名字没导出/模块被同名包遮蔽/测试语法错」这一类让 pytest 在 collection 阶段就失败的缺陷；抓不住运行期才暴露的问题（数值错、依赖缺失、参数签名变化）。
* `import *` 语句被跳过（无法静态判名）。
* 该审计是**门禁**：`tests/test_import_contract_audit.py` 把「不可解析导入数」钉在已知基线（含已在他单修复的条目），新增即失败。

复现：

```bash
PYTHONPATH=src python scripts/m14846_import_contract_audit.py \
    --out artifacts/m14846_import_contract.json \
    --md docs/DEM_IMPORT_CONTRACT_AUDIT_M14846.md
```

## 静态候选 vs 运行期确认（校准）

pytest 全树 collection：**511 个测试模块、1 个 collection error**（`test_ice_structure_contact.py`）。
静态扫描标出 **7 条候选**；逐条做运行期确认的结果：

| 候选 | 运行期 | 结论 |
|---|---|---|
| `tensordem.ice_structure_contact`（15 个符号，module 作用域） | `ImportError: cannot import name 'ContactPressureField'` | **真缺陷**，与 pytest collection error 完全一致 |
| `tensordem.ice_steel_friction`（5 个测试，`CALIBRATION_ANCHORS` 等，module 作用域） | 运行时属性存在 | **静态假阳性**：锚表由数据文件/循环动态绑定，AST 顶层命名空间看不到 |
| `tensordem.brash_ice_rheology`（function 作用域延迟导入） | 包可导入，无该属性调用路径 | 延迟导入候选，不影响 collection |

判读：静态审计**能把已合并但无法运行的验证面精确挑出来**（1/1 命中），代价是会把动态导出误报为候选；
因此本工具定位为**分诊**（candidate triage），不替代 `pytest --collect-only`。门禁测试
`tests/test_m14846_import_contract_audit.py` 把「新增同名文件/包遮蔽」钉死为失败，把
`ice_structure_contact` 的检测条件化在运行期状态上（上游 m14450 修复落地后自动跳过而不是假失败）。

`tensordem.ice_structure_contact` 的真缺陷已在 m14450 分支修复（模块迁入 `rolling_proxy.py` 并重导出），
本单不重复修复，只提供可复跑的分诊工具 + 回归门禁 + 候选账。
