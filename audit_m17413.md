# Board Audit: m17413 XVAL-DEM-ELASTIC-PLASTIC-ROLLING-SPRING-KR-225-MU 交差核验

**审计人**: board-audit (m17447)
**审计日期**: 2026-11-13
**对象**: m17413（TensorDEM vs Jiang-2005 滚动弹簧交叉验证，已 succeeded）
**报告 ID**: 16421
**Verdict ID**: 559

## 审计结论

**成熟度等级**: L3（系统集成级交叉验证工具）
**处置建议**: confirmed（消账），但需补 commit 所有 untracked 文件

## 三件套核实

### ① run_artifacts ✅ 存在且完整
| 文件 | 大小 | 内容 |
|------|------|------|
| summary.json | 29KB | 完整数值结果（4 项研究） |
| acceleration_parity.csv | 6.8KB | 54 行加速度对比数据 |
| transient_dynamics.csv | 1.4KB | 10 行瞬态动力学数据 |
| static_threshold.csv | 1.7KB | 20 行静态阈值数据 |
| torque_evolution.csv | 0.8KB | 8 行扭矩演化数据 |
| figures/fig1-4.png | ~200KB | 4 张对比图 |

### ② benchmark ref vs got ✅ 数值在容差内
| 规则 | 最大相对误差 | 容差 | 判定 |
|------|-------------|------|------|
| TensorDEM | 0.011% | <15% | ✅ |
| Jiang-2005 | 0.032% | <15% | ✅ |

注：tensor_benchmark_cases 表中无此锚点正式条目（XVAL 性质）。

### ③ repro_cmd ✅ 可执行
```bash
cd /root/tp-agent/work/m17413 && PYTHONPATH=src python scripts/xval_jiang2005_rolling_spring.py
```
验证通过：脚本成功执行，输出 4 项研究结果，写入 artifacts。

## 测试验证
- `tests/test_rolling_static.py`: 13 passed ✅
- `tests/test_rolling_resistance.py`: 33 passed ✅

## 关键发现

### ⚠️ 文件未跟踪
所有产出文件均为 `??` untracked 状态：
```
?? artifacts/xval_jiang2005_rolling_spring/
?? docs/xval_jiang2005_rolling_spring.md
?? scripts/publish_xval_jiang2005.py
?? scripts/xval_jiang2005_figures.py
?? scripts/xval_jiang2005_rolling_spring.py
```
**影响**: 成果未入库，无法追溯。需补 commit。

### ⚠️ 核心代码非本任务
`src/tensordem/rolling_resistance.py` 由 m16142 实现，本任务仅添加交叉验证脚本。

### ⚠️ 门禁缺失
`server_quality_audit`: gate_valid=false, key_present=false

## 成熟度证据

| 键 | 值 |
|----|-----|
| impl_files | scripts/xval_jiang2005_rolling_spring.py (15.7KB) + 图生成 + 报告发布 |
| entry | scripts/xval_jiang2005_rolling_spring.py |
| run_artifacts | artifacts/xval_jiang2005_rolling_spring/ (summary.json + 4 CSV + 4 图) |
| tests | test_rolling_static.py (13 passed) + test_rolling_resistance.py (33 passed) |
| integration_callers | 调用 rolling_resistance 模块完整仿真链 |
| benchmark | 无正式条目（XVAL 性质），但数值对比完整 |
| repro_cmd | PYTHONPATH=src python scripts/xval_jiang2005_rolling_spring.py |
| external_anchor | Jiang-2005 Comput. Geotech. 32, 340-357 |

## Verdict 落表

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorDEM', 'm17413 XVAL-DEM-ELASTIC-PLASTIC-ROLLING-SPRING-KR-225-MU', 17447, 3, '<JSON>', 'board-audit');
```
Verdict ID: 559
