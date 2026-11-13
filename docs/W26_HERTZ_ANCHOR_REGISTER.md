# W26 · Hertz 撞击时长/最大力锚登记（TensorDEM，mission 17564）

**结论摘要**：锚 **WITHHELD**（本环境无 Johnson 1985 原文，禁凭记忆/二手转录），
算子层（量纲/代数/极限/适用域）机检全部通过，求解器与闭式在 3 % 带内一致。
**所有数字只在工件里，本文件不转录任何数值**（零手抄）：

* 工件：`papers/artifacts/w26/dem-hertz-impact-duration/anchor_register.json`
* 图：`papers/artifacts/w26/dem-hertz-impact-duration/hertz_anchor.png`
* 实现：`src/tensordem/hertz_anchor_register.py`
* 生成器：`tools/gen_w26_hertz_anchor_register.py`（`PYTHONPATH=src python tools/gen_w26_hertz_anchor_register.py`）
* 验收测试：`tests/test_hertz_anchor_register.py`
* 报告发布：`post_report_w26.py`

## 1 锚与状态

| 项 | 内容 |
|---|---|
| 锚 id | `JOHNSON-1985-CONTACT-MECHANICS-HERTZ-IMPACT`（Johnson, *Contact Mechanics*, CUP 1985） |
| 覆盖主张 | 弹性撞击的接触时长 `t_c`、峰值重叠量 `δ_max`、峰值力 `F_max` |
| 状态 | `WITHHELD`（原文不可得；`verbatim=null`，`document_sha256=null`） |
| 允许的替代举证 | 仅算子层机检 + 求解器交叉核（见工件 `criteria`） |
| 纪律 | 该 WITHHELD 判定**不得被后续轮覆盖** |

树内二手引用（**代码/文档引用，不是锚**）连同 sha256 与行号登记在工件
`anchors.primary.secondary_in_tree_citations` 与 `provenance.quote_positions`。

## 2 判据逐条

| 判据 | 工件字段 | 本轮裁决 |
|---|---|---|
| ① 逐字锚位 ≥1 或 honest WITHHELD | `criteria[C1]` | WITHHELD（如实） |
| ② 量纲闭环残差 ≤1e-12 | `criteria[C2]` / `tables.dimension_closure` | 见工件（含灵敏度对照） |
| ③ 刚度比/阻尼适用域写清 | `criteria[C3]` / `applicability_domain` | 见工件 |
| 附加：v→0 发散形式 | `criteria[C4]` / `tables.limits` | 见工件 |
| 附加：求解器 vs 闭式 | `criteria[C5]` / `tables.solver_vs_anchor` | 见工件 |
| 附加：C 的两条独立实现 | `criteria[C6]` / `tables.constants` | 见工件 |

## 3 复现

```bash
PYTHONPATH=src python tools/gen_w26_hertz_anchor_register.py   # 重算并重写工件 + 图
PYTHONPATH=src python -m pytest tests/test_hertz_anchor_register.py -q
PYTHONPATH=src python post_report_w26.py                        # 发布报告
```

生成器把 `git rev-parse HEAD`、`git status --porcelain`、依赖版本与三份源文件的
sha256 一并写进 `provenance`，工件与实现可逐位对齐。

## 4 已知局限

1. 算子层检查约束的是闭式的**形式**（指数/量纲/代数/极限），不约束文献自印数值——
   文献数值一致性在本环境**无法判定**，故留 WITHHELD，不粉饰。
2. 求解器交叉核是单颗粒-刚性底板撞击；多接触屏蔽不在本节点范围。
3. 弹性锚不含塑性屈服：峰值接触压力超过约 1.6Y 后失效，该上界本轮只作定性声明
   （无原文不给登记数字）。
4. 阻尼适用域来自一维 Hunt-Crossley 积分，`α` 的物理取值须由材料耗散标定，本轮不猜。