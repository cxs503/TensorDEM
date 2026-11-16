# 交差核验 m23874 · CAP-REG-DEM · TensorDEM 能力注册表（双试点规范）

审计单 m23941（board-audit）· 基准裁判=tensor_benchmark_cases · 判级 L0-L5
被审对象：m23874（status=succeeded，PR #1709 merged，merge commit a3f08635，worktree commit 946f250a）

## 1. 验收三件套逐项核验

| 件 | 判定 | 证据 |
|---|---|---|
| run_artifacts | **缺失** | m23874 worktree 无 runs/m23874*、无 docs/m23874/、无 registry_dump.json 落盘；mission result check#5 自认「平台 sync/报告/commit 未执行（同因超时）」 |
| benchmark ref vs got | **无裁判行** | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CAP-REG-DEM'` → **0 rows**（TensorDEM 仓 6 个其他锚点共 12 行，本锚 0 行）→ L4 按定义不可达 |
| repro_cmd | **可执行** | `cd /srv/tensor/agent/work/m23874 && PYTHONPATH=src python3 -m tensordem.registry --static` → exit 0，count=374，mature 34/developing 340/planned 0，A48/B248/C78，wired 373/tested 282/verified 48，endpoint 5 个——与 mission 声明逐项一致 |

## 2. 独立复算（非自声明）

- static vs import parity：独立脚本复算 **374=374，0 字段差异**（runs/m23941_audit/parity_check.log）
- anti-overclaim 门：mature 无 verified+wired 缺项=0；endpoint 无 wired 缺项=0
- 48 条 verified=True 的 evidence 工件（verification 模块/verdict_*.json/xval 测试）逐一存在性核验：0 缺失
- 282 条 test_path 文件存在性：0 缺失
- 5 个 endpoint 均为 cli.py 真实子命令（cli.py:260-340）；spot-check `python -m tensordem binding-contract` → verify=CONFORMS probes=19/19 benchmark=11/11（runs/m23941_audit/binding_contract_cli.json）
- pytest tests/test_capability_registry.py（12 测试）：**11 passed / 1 failed**
  - FAILED `test_static_import_parity`：tests/test_capability_registry.py:129 `c["id"]` 对 Capability dataclass 下标 → TypeError（单跑）/ 套件内 import 缓存致空集 AssertionError；**该测试按提交态永红**
  - mission check#4 自认「修复后全量绿跑被 120s 硬超时打断，未拿到最终绿跑记录」——与本次独立复跑一致

## 3. 报告六段核验

tensor_reports WHERE mission_id=23874 → **0 rows**（全库按标题检索亦无）。双试点规范（m23186 ⑤「收尾必落 tensor_reports 双件」）未满足。六段齐全性=不适用（报告不存在）。

## 4. 判级：L2（独立功能实现）

- L2 成立：registry.py 293 行真码（AST 静态扫描+import 自注册+anti-overclaim 门+双模式 dump CLI），gen_dem_capabilities.py 259 行生成器（证据规则机器可核），capabilities.py 6368 行生成清单；输入(src 树)→计算→输出(JSON 契约) 经 exit-0 重放与 374=374 parity 实证；已并 main（PR #1709）。
- 非 L3：tensordem/__init__.py 不 import capabilities；平台 /tensor/capabilities?repo=TensorDEM → 0 rows（sync 未执行）；registry.dump() 无生产消费方（仅 CLI/测试）。
- 非 L4：本锚 0 基准行；无 ref vs got 数值对照。

## 5. 发现（按严重度）

- F1 medium：提交态测试套件 11/12 绿，test_static_import_parity 结构性损坏（TypeError，永红）；parity 声明本身为真（独立复算），但守护测试失效。
- F2 medium：run_artifacts 未落盘（stdout 契约未持久化）。
- F3 medium：tensor_reports 无 m23874 报告（规范双件缺一）。
- F4 low：平台 capability 表 TensorDEM 0 行，capability_sync 守护未获数据。
- F5 info：feature_ledger CAP-REG-DEM 仍 in_progress。

## 6. 裁定

level=2。evidence 指针见 tensor_maturity_verdicts（verdict_by=board-audit）。
工件：runs/m23941_audit/{registry_static_dump.json, pytest_capability_registry.log, parity_check.log, binding_contract_cli.json}
