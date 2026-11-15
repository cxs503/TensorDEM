# MATURITY EVIDENCE — m19466 / XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT (round 4)

仓库 TensorDEM · 分支 `agent/m19466` · 原单 m19453 返修轮（前轮达标复核仅 L1：缺 run_artifacts / entry）
本轮实质增量 = **把真实消费方（TensorPhysics `TensorDEMAdapter` + `Simulation`）跑在真实供应方
（`tensordem.dem.IceDEM`）上**，并把「契约声明的每个数字都被观测到」落成可复跑工件；顺带修掉
`python -m tensordem.cli` 静默成功（rc=0、零工件）的入口缺陷。

## 证据（机器可核，路径均相对仓库根）

```json
{
  "impl_files": [
    "src/tensordem/binding_contract.py",
    "scripts/xval_library_binding_contract_live.py",
    "src/tensordem/cli.py"
  ],
  "entry": [
    "scripts/xval_library_binding_contract_live.py",
    "src/tensordem/cli.py"
  ],
  "run_artifacts": [
    "runs/xval_library_binding_contract_m19466/run.json",
    "artifacts/xval_library_binding_contract_live/live_binding.json",
    "artifacts/xval_library_binding_contract_live/live_binding_table.md",
    "artifacts/xval_library_binding_contract/binding_contract_cli.json",
    "artifacts/xval_library_binding_contract/benchmark_physics_3d.json"
  ],
  "integration_callers": [
    "exchange_state",
    "assert_entry_conforms",
    "TensorDEMAdapter"
  ],
  "integration_callers_detail": {
    "src/tensordem/dem.py": "IceDEM.exchange_state() -> assert_entry_conforms(..., 'IceDEM') 导出前置门（冻结面不许静默漂移）",
    "src/tensordem/dem3d.py": "IceDEM3D.state_dict() -> assert_entry_conforms(..., 'IceDEM3D')",
    "src/tensordem/cli.py": "binding-contract 子命令 = 仓内真实调用 verify()/probe_semantics()/benchmark_physics{,_3d}()",
    "tensorphysics.adapters.tensordem.TensorDEMAdapter": "跨仓绑定方：本轮首次被放在真 IceDEM 上执行（上游 checkout）"
  },
  "benchmark": {
    "live_binding_2d": {
      "substep_count": {"ref": 12, "got": 12, "rel_error": 0.0},
      "load_time_stamp_max_dev_s": {"ref": 0.0, "got": 4.336808689942018e-19, "rel_error": 4.34e-19},
      "external_impulse": {"ref": [0.02449567132475335, -0.009798268529901339],
                           "got": [0.024495671324753346, -0.009798268529901337],
                           "rel_error": 1.77e-16},
      "impulse_ratio_got_over_ref": {"ref": 1.0, "got": [0.9999999999999999, 0.9999999999999998], "rel_error": 2.22e-16},
      "clock_n_dt_s": {"ref": 0.005443482516611855, "got": 0.005443482516611856, "rel_error": 1.59e-16},
      "h0_separation": {"threshold": 0.15, "got": 0.8461538461538461},
      "accumulating_drift_ratio_vs_H0_prediction": {"predicted": 6.5, "got": 6.500000000000001, "detected": true},
      "stale_clock_detected": true,
      "rows_passed": "14/14",
      "worst_rel_error": 2.22e-16,
      "verdict": "MET"
    },
    "supplier_side_same_commit": {
      "verify": {"status": "CONFORMS", "checks": "73/73"},
      "probes": "19/19 PASS",
      "benchmark_2d": {"rows": "11/11", "worst_rel_error": 0.045454545454616024, "verdict": "MET"},
      "benchmark_3d": {"rows": "15/15", "worst_rel_error": 0.008641567854537771, "verdict": "MET"},
      "contract_fingerprint": "sha256:048e4cf9b3a1257eca0fa5178b64102bc3c291899566e6c9b90020b3e05195cd"
    },
    "owner_tolerance": 0.15,
    "over_3pct_honest": [
      "benchmark_2d.free_flight_vs_continuum_dt_x1 = 4.55% (一阶半隐式对连续解的预期离散误差；dt 减半/四分之一→2.27%/1.14%，一阶收敛)"
    ]
  },
  "repro_cmd": "PYTHONPATH=src python3 -m tensordem.cli binding-contract --out artifacts/xval_library_binding_contract && PYTHONPATH=src python3 scripts/xval_library_binding_contract_live.py && PYTHONPATH=src python3 -m pytest tests/test_binding_contract.py tests/test_binding_contract_live_adapter.py tests/test_cli_module_entry.py -q",
  "tests": [
    "tests/test_binding_contract.py (52 passed, 2 skipped 同批运行)",
    "tests/test_binding_contract_3d_benchmark.py",
    "tests/test_binding_contract_live_adapter.py (10 passed: 14/14 arms MET, H0 separation 0.846, drift 2/2, committed artifact vs fresh run)",
    "tests/test_cli_module_entry.py (3 passed: 模块入口必须写工件; 模块入口 vs 包入口指纹一致)"
  ]
}
```

## 本轮改了什么（最小集）

| 文件 | 类型 | 说明 |
|---|---|---|
| `scripts/xval_library_binding_contract_live.py` | 新增（入口） | 真消费方 × 真供应方；14 臂；受控漂移 2 类 + 子步边界 3 点 + 不累加扫掠 n=1..16；写 JSON + 表；FAIL 则 rc≠0 |
| `src/tensordem/cli.py` | 缺陷修复 | 末尾补 `if __name__ == "__main__": main()`：此前文档复现命令 `python -m tensordem.cli binding-contract …` 导入即退出、零工件、仍返回 0 |
| `tests/test_binding_contract_live_adapter.py` | 新增测试 | 直接 import 入口脚本执行，断言 13/13 臂、指纹钉扎、H0 分离度、漂移检出、提交工件与新鲜运行的数值一致 |
| `tests/test_cli_module_entry.py` | 新增测试 | 子进程级断言模块入口真的产出工件 + 两种入口指纹一致 |
| `prereg/XVAL_DEM_LIBRARY_BINDING_CONTRACT_R4.json` | 预登记 | 阈值先于运行（stamp 1e-12 s、冲量 1e-12、H0 分离 ≥0.15、边界 3/3、漂移 2/2） |
| `prereg/XVAL_DEM_LIBRARY_BINDING_CONTRACT_R4_addendum.json` | 事后追加声明 | 不累加扫掠臂：如实标注「第一轮执行后追加」，阈值沿用未改 |
| `artifacts/xval_library_binding_contract{,_live}/` | 运行工件 | 本轮真实执行产出（含 sha256 + git sha 于 `runs/…/run.json`） |
| `docs/XVAL_DEM_LIBRARY_BINDING_CONTRACT_LIVE.md` | 文档 | 轮次报告（六要素 + ref/got 表 + 是否 <3% 逐项回答） |

## 复现记录（跨节点）

| 位置 | 命令 | 结果 |
|---|---|---|
| 本机（worktree） | `PYTHONPATH=src python3 -m pytest tests/test_binding_contract.py tests/test_binding_contract_3d_benchmark.py -q` | **52 passed, 2 skipped**（50.0 s） |
| 本机（worktree） | `pytest tests/test_binding_contract_live_adapter.py tests/test_cli_module_entry.py -q` | **13 passed**（43.5 s） |
| 集群 n315（cj2157，4 核） | 四文件全量 pytest（stage_dir = 本工作树） | succeeded（rc=0），取回工件中 `substep_count 12/12`、`load_time_stamp_max_dev_s = 4.34e-19`、`external_impulse rel 1.77e-16` 与本机逐位相同 → 结论不依赖本机 |
| 报告 | `POST /api/v1/tensor/reports` | id **17728** |

## 局限（如实）

1. **3-D 无 live binding**：上游 `TensorPhysics` 只有 2-D `TensorDEMAdapter`，3-D 面本轮仅由供应方探针 + 3-D benchmark 背书。
2. **消费方 checkout 位于仓外**：TensorDEM CI 无 `tensorphysics`，live 测试缺失时 skip；工件由本机真实执行产生，`run.json` 记录 consumer 路径与 `git_sha`，任一侧变化须重跑。
3. **不变式 vs 数值收敛**：本轮判的是「绑定语义（子步/时钟/stamp/不累加/不可重启）」，不是接触力精度；接触力精度见 2-D/3-D benchmark 行与既往 CASE-DEM-* 条目。
4. 2-D `free_flight_vs_continuum_dt_x1` 误差 4.55% > 3%（如实标注；一阶离散预期，< owner 15% 线）。