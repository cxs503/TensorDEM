# REGISTRY_AUDIT · m24013 · CAP-REG-DEM · TensorDEM 能力注册表

生成：`scripts/audit_capability_registry.py m24013`（数字程序化注入，零手抄）· 2026-11-16 21:37 UTC
基准 commit：`4801d5b7e16a15cce4e50dabe6cbe96e711cc502`

## 1. CLI 三模式（直调文件路径，torch 可选）

| 模式 | 命令 | rc | 结果 |
|---|---|---|---|
| static | `python src/tensordem/registry.py --static` | 0 | 375 条，problems=0 |
| import | `python src/tensordem/registry.py` | 0 | 375 条，problems=0 |
| parity | `python src/tensordem/registry.py --parity` | 0 | parity=True（375=375，diffs=0） |

stdout 契约：单 JSON 对象 `{source_commit, capabilities, summary, problems}` —— 三模式全部满足

## 2. 条目数如实性

- capabilities.py 中 `@capability` 装饰器数 = **375**
- static 扫描 total = **375** → 一致 ✓
- summary：mature=35 developing=340 planned=0；wired=374 tested=283 verified=49 endpoints=5

## 3. 反过度声明门（独立复算，非自声明）

| 门 | 违例数 |
|---|---|
| mature 需 verified+wired | 0 |
| planned 需 evidence.feature_ledger 锚 | 0 |
| endpoint 需 wired + cli:/route:/api: 前缀 | 0 |
| verified 需 ANALYTIC/XVALUE 级证据 | 0 |
| **合计** | **0** |


## 4. 指针解析（module/entry/test_path 不存在 = problem，非静默丢弃）

- 注册表自报 problems：**0**
- 独立复算指针违例：**0**

- verified 条目证据工件缺失：**0**

## 5. 契约与 parity 违例

- static 契约违例：0
- import 契约违例：0
- parity 差异：0

## 6. 裁定

**PASS**：三模式 exit 0；stdout 单 JSON 对象契约满足；条目数与装饰器数一致；四道反过度声明门 0 违例；指针 0 违例；verified 证据工件 0 缺失；static/import parity 全字段一致。
