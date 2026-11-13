# m17789 交差核验：XVAL-DEM-RADON-PROJECTION-MEASUREMENT（TensorDEM）

> 交差审计（非重做）。被审交付 `57564a2eea3b` @ `agent/m17789`（未并入本地 main，远端已并：a332e7aa Merge #1462 (m17593) via merge-master main）；审计单 m17873。
> 本页与 `artifacts/audit_m17873/*.json` 全部程序化生成，无手抄数字。

**裁定：L2（board-audit）**

一句话：测量器与验证装置都是真代码，无噪声下把注入倾角复原到 1.33%（远严于 15%），但**声明的工作包络（1% d 测量噪声）内最大相对误差 76%、15% 内占比 0%**，交付自己判 `pass=False`（诚实 FAIL）；同时 `tensor_benchmark_cases` 无本锚登记行，发布的复现命令按字面执行**不复现**提交工件。故 L2，不达 L3/L4。

## 1. 对象与判据出处

- 对象：`CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE` 剪切带倾角的**测量仪器** `tensordem.biaxial_shear_band.measure_band`（`src/tensordem/biaxial_shear_band.py:268`，内核 `radon_band_angle` 同文件 :154）；修复前基线 `biaxial_band_diagnostics.legacy_radon_band_angle`（`src/tensordem/biaxial_band_diagnostics.py:258`）。
- 参考：解析——`θ_true` 是注入位移场 `u=(slip/2)tanh(2s/w)·t` 的参数，无外部锚、无参考不确定度。
- **基准裁判（唯一）**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-DEM-RADON-PROJECTION-MEASUREMENT'` → **0 行**（全表 74 行；TensorDEM 域 8 行）。**无判据行**，本次不以他物代替裁判，只如实登记这一台账缺口。
- 15% 判据的真实出处：`feature_ledger` slug=`XVAL-DEM-RADON-PROJECTION-MEASUREMENT`（acceptance=`<15%（owner 精度标准：对标误差 >15% 不可接受）`，status=open，evidence=`TensorDEM#1462 merged 11-13 · TensorDEM#1465 merged 11-13`）+ 交付页与 harness 内先于运行写死的判据。

## 2. 三件套是否真存在

| 件 | 指认 | 核验 |
|---|---|---|
| impl | `benchmarks/biaxial_shear_band/run_radon_xval_measurement.py`（新）+ `src/tensordem/biaxial_shear_band.py:{154,268}`（被验算子） | 提交内逐字节存在（sha256 入 manifest），缺件 0 |
| run_artifacts | `artifacts/biaxial_shear_band/radon_projection_xval.json`（16 行 / 255 叶）、`*.png`（105852 B）、`docs/XVAL_DEM_RADON_PROJECTION_MEASUREMENT.md` | 均 git-tracked；JSON 由审计独立聚合复核（§3） |
| benchmark ref vs got | ref=解析 θ_true；got=逐格 `radon_production`（§3 表） | 可复算；**无登记裁判行** |
| repro_cmd | 交付 checks 串：`--lattices 24x48 24x24 --angles 30 45 60 75 --widths 1.5 --noise 0.0 0.01` | 字面执行 → 32 行、与提交件不等；加 `--repeats 1` 后 sha256 逐字节一致（§4） |

## 3. 数字裁判（独立重算 vs 提交件）

| 量 | 提交件声称 | 审计独立重算 | 一致 |
|---|---|---|---|
| 全域测量数 n | 16 | 16 | ✓ |
| 全域 bias | -9.1250° | -9.1250° | ✓ |
| 全域 RMS | 25.0225° | 25.0225° | ✓ |
| 全域 max rel | 0.76 | 0.76 | ✓ |
| 无噪声 bias / max abs | +0.12° / 1.00° | +0.12° / 1.00° | ✓ |
| 1% 包络内 max rel / 15% 内占比 | 76.0% / 0.0% | 76.0% / 0.0% | ✓ |
| 24×48 生产算子 bias / max abs | -10.625° / 57.00°（原子块） | -10.625° / 57.00° | ✓ |
| 24×48 修复前基线 bias | -20.500° | -20.500° | ✓ |

**逐格核（基准裁判缺失，故用交付自己预置的 <15%% 判据 + 解析参考）**

| 场景 | ref | got（最大相对误差） | 15%% 容差 | <3%% |
|---|---|---|---|---|
| 无噪声（8 格） | 解析 θ_true | 1.33% | PASS | PASS |
| 1% d 包络（8 格） | 解析 θ_true | 76% | **FAIL** | FAIL |

逐格（24×48，带宽 1.5 d）：

| θ_true | noise | θ̂ 生产 | 误差 | θ̂ 修复前 | 基线误差 |
|---|---|---|---|---|---|
| 30° | 0 | 30° | +0° | 18° | -12° |
| 30° | 0.01 | 49° | +19° | 33° | +3° |
| 45° | 0 | 45° | +0° | 30° | -15° |
| 45° | 0.01 | 11° | -34° | 28° | -17° |
| 60° | 0 | 60° | +0° | 45° | -15° |
| 60° | 0.01 | 47° | -13° | 26° | -34° |
| 75° | 0 | 75° | +0° | 65° | -10° |
| 75° | 0.01 | 18° | -57° | 11° | -64° |

- 判定可机检：`verdict` 的四个布尔量由行数据机械重算，与提交件完全一致。
- 提交 `pass=False`（无噪声精确 + 包络内不合格 + 24×48 上优于修复前基线）—— 与审计重算一致。

## 4. 可复现性（实测，本机 x86_64；集群为 loong64，架构不同）

| 运行 | 命令 | 行数 | 与提交件 sha256 | 叶差 |
|---|---|---|---|---|
| 提交工件 | — | 16 | `5730044da0300b1b…` | — |
| 字面 repro_cmd | 无 `--repeats` | 32 | 不等 | 114/255 |
| 补 `--repeats 1` | 同上 + `--repeats 1` | 16 | **一致** | 0/255 |

- 原因：默认 `REPEATS=3`（harness 第 66 行）使每个含噪格跑 3 次实现 → 32 行（bias -5.78°，max rel 1.10）；提交件只有 16 行（含噪格仅 1 次），只有 `--repeats 1` 能逐字节复现 `sha256=5730044da0300b1b…`（255 λ 全等）。
- 但工件内 `design.repeats` 仍写 **3**（harness 第 384 行把常量 `REPEATS` 写死进 JSON，不看 `args.repeats`）→ 工件自相矛盾（设计声称 3 次实现，行内只有 repeat=0）。
- 单元格确定性现场测（新跑）：同一格跑两次 sha256 一致（角度 60.0°）。
- 交付页声称的“48-way sharded on the 702 cluster”在树内**无分片产物、无日志可查**；提交 JSON 的行序与单进程运行完全一致（§3可复现），集群分片只能计为声明（not_run）。

- **子集崩溃（新发现）**：只要运行集里没有 24×48 格（如 `--lattices 24x24`，或落到只含 24×24/36×36 的集群分片），harness 在 `summarise()` 第 213 行对空的修复前基线列表取 max → `ValueError: zero-size array`，exit=1，**分片产物都写不出来**。“48 分片跑通”的声明因此不可完全采信。
## 5. 测试与集成面

- pytest：`PYTHONPATH=src python -m pytest tests/test_radon_projection_xval.py -q` → exit 0，`11 passed, 1 warning in 68.01s (0:01:08)`；相邻剪切带套件 exit 0，`28 passed, 1 warning in 11.98s`。
- 断言是不变量而非“跑一遍”：已知角回收、折叠不变（120°≡60°）、修复前基线必须超 15%%、退化输入拒绝、分片恰好划分、汇总/判定逻辑机械性、`radon_band_angle` 链 == `measure_band` 且幂等。
- 集成面：`measure_band|radon_band_angle` 全仓 98 处命中（tests 34、docs/board_audit 13、benchmarks benchmarks/biaxial_shear_band/make_figure.py, benchmarks/biaxial_shear_band/render_radon_xval_doc.py, benchmarks/biaxial_shear_band/run_r2_case_rerun.py, benchmarks/biaxial_shear_band/run_r3_ensemble.py, benchmarks/biaxial_shear_band/run_r3_estimator_audit.py, benchmarks/biaxial_shear_band/run_radon_xval_measurement.py）；src 内部 ./src/tensordem/biaxial_band_diagnostics.py, ./src/tensordem/biaxial_band_diagnostics.py, ./src/tensordem/biaxial_band_diagnostics.py, ./src/tensordem/biaxial_band_diagnostics.py, ./src/tensordem/biaxial_band_diagnostics.py, ./src/tensordem/biaxial_band_diagnostics.py。**未**导出于 `src/tensordem/__init__.py`，**未**被求解器主循环（`dem.py`/`clump.py` 0 命中）调用 → 它是一个算例的后处理仪器，不是平台/求解器 API（L3 不达）。

## 6. 报告（tensor_reports id=16750）

- 存在：True；mission_id=17789、scenario_key=`tensordem.shear_band_measurement_xval`、10624 字符、图 有。
- 六段：① 案例介绍, ② Method, ③ Results, ④ Figures, ⑤ Notes, limitations, scope, ⑥ Error judgement (against the analytic reference)；缺失 []。数字与 §3 重算一致（报告不把失败粉饰成通过，已自宣 honest FAIL）。

## 7. 注意点（坑与局限）

1. **裁判行缺失**：`tensor_benchmark_cases` 0 行本锚 → 平台机器门无法对表裁定；15%% 只能追到 feature_ledger + 交付页。这是**台账缺口**，不是数值缺陷，但 L4 的“机器可复核基准”一环为空。
2. **包络声明与 FAIL 的读法**：1%%·d = 0.015 m 噪声对 0.02 m 注入滑移，SNR≈1.3；包络内 76%% 误差是“极端信噪比”下的实测崩溃，不能读成算子坏掉；反之也不能把无噪声 1.3%% 读成“仪器已在真实工作点验证”。交付这一点写得诚实。
3. **设计覆盖 1.5%**：实跑 16 格 / 声称 360 格×3 次实现；缺少 {'lattices': ['36x36'], 'widths': [0.5, 3.0], 'noise': [0.005, 0.02, 0.04], 'angles': [15.0, 25.0, 35.0, 55.0, 65.0, 85.0]}。“带宽就是分辨率声明”只有 1.5 d 一个点支撑。
4. **发布的命令不忠实**：checks 串字面跑出 32 行，数字不同；提交工件只能用未记录的 `--repeats 1` 逐字节复现。命令与元数据应补上（返修项）。
5. 本审计在 x86_64 本机重跑；交付声称的 loong64 集群 48 分片无产物可核，跨架构位级复现未验证（not_run）。

## 7bis. 锚点既有成熟度与本单关系

- 边界声明：**锚点层面已有 L4 裁定**（verdict id 610，mission 17632，board-audit，gate=reopened），对象是同一个 `radon_band_angle`，参考是 OSS scikit-image Radon（sha256 钉住）102 算例。本单只判 m17789 这**一轮**的交付，不推翻锚点；m17789 的 honest FAIL 不是工具不能用，而是它自己声明的噪声包络内精度不达标。
- 主线上同名锚点的内容是**另一轮**：`docs/XVAL_DEM_RADON_PROJECTION_MEASUREMENT.md` 在 origin/main 上写的是 “TensorDEM vs the OSS image-domain Radon”，配 `benchmarks/radon_projection_xval/run_xval.py` + `src/tensordem/radon_reference.py` + `prereg/XVAL_DEM_RADON_PROJECTION_MEASUREMENT.json`（预注册）。
- 提交在 origin/main 上**不**存在该 commit：本单产物是否已并：artifact=False，docs 同名但内容不同。
- 因此：m17789 是同一锚点的第二轮（已有轮次已并入 main），且本轮**无冻结 prereg**（树内 prereg 仅属于 m17593 的因子设计），“预声明判据”不可证；台账 evidence 写“TensorDEM#1462 merged”，而 PR #1462 的合并提交（a332e7aa）对应 m17593，不含本单工件。

## 8. 判级依据（L0-L5）

- L1 met: analytic known-angle reference recovered to max rel 1.33% noise-free (bias +0.12 deg, max abs 1.00 deg over 8 zero-noise cells); fold invariance, determinism and bare-chain == measure_band are asserted by tests; committed verdict reproduced mechanically from the committed rows
- L2 met: real measurement chain (kNN deformation gradient -> local max shear strain -> Radon/projection variance focus -> folded angle) driven by a real harness (displacement snapshots in -> per-cell angle JSON out); 11 passed, 1 warning in 68.01s (0:01:08); neighbouring band suites: 28 passed, 1 warning in 11.98s
- L3 blocked: measure_band is not exported by src/tensordem/__init__.py and no solver-core module calls it; consumers outside tests are the biaxial case scripts (benchmarks/biaxial_shear_band/make_figure.py, benchmarks/biaxial_shear_band/render_radon_xval_doc.py, benchmarks/biaxial_shear_band/run_r2_case_rerun.py, benchmarks/biaxial_shear_band/run_r3_ensemble.py, benchmarks/biaxial_shear_band/run_r3_estimator_audit.py, benchmarks/biaxial_shear_band/run_radon_xval_measurement.py) plus earlier audit probes — post-processing of one benchmark case, not a solver/package API
- L4 blocked (referee): tensor_benchmark_cases has 0 rows for XVAL-DEM-RADON-PROJECTION-MEASUREMENT (table total 74; TensorDEM-domain rows exist only for clump/friction/H1/Campbell and CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE) — the reference is the harness' own analytic theta_true and the <15% number traces only to feature_ledger row 'XVAL-DEM-RADON-PROJECTION-MEASUREMENT' (status open)
- L4 blocked (own gate): inside the declared 1% operating envelope max rel err 76.0% vs 15%, share within 15% = 0% over 8 cells; noise-free is 1.3%, so this is the measured SNR limit of the operator, not a coding bug — but the precision claim is unmet (committed verdict pass=False)
- L4 blocked (repro): the published command string runs to 32 rows with bias -5.78 deg / max rel 1.10, while the committed artefact has 16 rows with bias -9.12 deg / max rel 0.76; byte identity (sha256 5730044da0300b1b) is only reached after adding --repeats 1, and the artefact's design block still advertises repeats=3 -> artefact self-inconsistent
- L4 blocked (coverage): 16 of 360 advertised design cells x 3 realisations executed (1.48% of the design); missing levels {'lattices': ['36x36'], 'widths': [0.5, 3.0], 'noise': [0.005, 0.02, 0.04], 'angles': [15.0, 25.0, 35.0, 55.0, 65.0, 85.0]} — per-width / per-lattice statements in the page are not backed by the executed design
- L4 blocked (portability): the harness aborts on any subset without the 24x48 case lattice (summarise() line 213 takes max() of an empty legacy list; observed exit 1 'ValueError: zero-size array to reduction operation maximum which has no identity'), so a 48-shard reproduction cannot be trusted for shards holding only 24x24/36x36 cells
- context (not a defect of m17789): the anchor already carries an L4 verdict from mission 17632 (verdict id 610) whose reference was the OSS scikit-image Radon at sha256 pin; m17789 is an independent round on the same operator and adds a measured noise envelope, so the anchor's maturity does not hinge on this round's honest FAIL
- L4 blocked (predeclaration): the page calls its decision rule 'predeclared' but no frozen prereg artefact exists in the tree for this round (round-1 prereg belongs to m17593 and fixes different factors) — the criterion cannot be shown to predate the runs
- doc/artefact contradiction: the page promises '3 realisations for every noisy cell' while every noisy row carries repeat=0 only
- L4 also inherits the L3 integration gap

**裁定：L2。** 升 L3 需：导出到包 API 并被求解器/平台调用；升 L4 需：登记裁判行 + 包络内误差达标 + 复现命令与工件对齐。

## 9. 本次未跑（not_run，不计为 pass）

- loong64 集群 48 分片原命令跨架构重跑（无分片产物/日志）。
- 完整设计的 360 格（含 0.5 d / 3 d 带宽、 36×36 晶格、 0.5%/2%/4% 噪声，3 次实现）——本审计只复现交付的 16 格。
- 真实 DEM 算例（`run_r3_ensemble.py` 全跑）与实验/文献锚对比——本单只审仪器校验。

