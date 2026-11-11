# XVAL-DEM-CHUTE2001 — 斜面颗粒流三方横向对比（第 1 轮，honest-FAIL）

锚点：`XVAL-DEM-CHUTE2001`（与论文复现线 SILBERT2001 同案配对）
案例出处：Silbert, Ertaş, Grest, Halsey, Levine & Plimpton, *Granular flow down
an inclined plane: Bagnold scaling and rheology*, **Phys. Rev. E 64, 051302 (2001)**。

> **结论先行（honest-FAIL，不粉饰）**：本轮交付了可复现的**三方同规格横向对比基础设施**
> （我方求解器 + 一份三用的 case_spec + LIGGGHTS-PUBLIC 3.8.0 源码构建 + 中性采样口径 +
> 单元测试），但**三条判据一条也没有取得可比数值**：LIGGGHTS 工况卡在其 `pair_gran`
> 的 ghost 速度通信要求上未跑通，第二参照（MercuryDPM）不可装，我方 arrest 扫描的
> 松弛时间窗不足 8.2 t_ref 故 h_stop 未观测到停机。**因此本轮不产出任何"工具间一致"的
> 结论**；`result.metrics` 三项全部标 **WITHHELD**，`fail_stage=oss_execution_and_runs`。

## ① 案例设定与 ref 出处

斜面周期槽道（chute flow）：`x` 流向、`y` 展向、`z` 壁法向；重力倾斜 θ；
底部为**同粒径粗糙基床**（固定球层，面内间距 0.9 d，重叠 10 %，无漏孔）；
流向/展向周期，上表面自由。观测量为稳态速度剖面 `vx(z)`、Bagnold 指数 `p`
（`vx(z)=C[h^p-(h-z)^p]`，Bagnold p=3/2）与停机厚度 `h_stop(θ)`。

| 量 | 值 | 说明 |
|---|---|---|
| 粒径 d | 1.0e-3 m | 长度单位 |
| 密度 ρ | 2500 kg/m³ | m=1.309e-6 kg |
| 法向刚度 k_n | 500 N/m | k_t/k_n = 0.5 |
| 恢复系数 e | 0.88 | γ_n = 1.471e-3 N·s/m（ζ=0.0407） |
| 摩擦 μ | 0.5 | 滚动阻力 0（同 Silbert） |
| 时间步 dt | 1.0e-5 s | dt/T_contact = 0.044（22 步/接触周期） |
| 参考量 | v_ref=√(gd)=0.0990 m/s，t_ref=√(d/g)=0.0101 s | 全部结果无量纲化 |
| 稳态工况 | Lx=Ly=10 d，N=1920 → h=10.05 d，θ=24° | + 粗分辨率 Lx=Ly=7 d，N=940 → h=10.04 d |
| 停机扫描 | θ ∈ {18°, 20°} × N ∈ {300,600,900,1200}（h=1.57/3.14/4.71/6.28 d） | 11 个 case 全部生成 |

## ② 三方方法与版本

| 工具 | 版本/来源 | 状态 |
|---|---|---|
| TensorDEM（我方）| 本工作树 `src/tensordem/chute_flow/`（新模块，见下） | **跑通**（短窗验证） |
| LIGGGHTS | LIGGGHTS-PUBLIC **3.8.0**，`git commit 3d5c00f20519e6bb6eb6756f51f1ad36564e649d`，`make serial`（MPI stubs，无 MPI） | **构建成功**（lmp_serial 7,173,808 B），**工况未跑通** |
| 第二 DEM | MercuryDPM：`git ls-remote` 经 ghproxy → 仓库路径不可解析（认证提示），未安装；LAMMPS（gitee 镜像 668 MB 克隆成功）因本机 CMake 3.26.3 < 3.27 构建受阻 | **未装成** |

我方实现（关键指针）：
* `src/tensordem/chute_flow/spec.py` — case_spec（参数、粗糙基床与初始料的确定性生成、CSV+sha256 清单、采样口径写死进 manifest）
* `src/tensordem/chute_flow/engine.py` — 周期槽道速度 Verlet（半步步长求耗散力），9 像元 KDTree 候选对，最小像约定
* `src/tensordem/chute_flow/analysis.py` — **工具无关**后处理：0.5 d 分箱速度剖面、体积等效厚度、Bagnold 一维搜索拟合、arrest 判据、h_stop 括区
* `src/tensordem/granular_column/kernel.py` — 新增 `box=(Lx,Ly)` 最小像参数（默认 `None`，逐位向后兼容）
* `tools/xval_chute2001/harness.py` — 一份 case_spec → 三份输入（我方/LIGGGHTS/LAMMPS deck+data）+ 中性 dump 转换 + 汇总

### ②b 我方设计要点（为什么这样算才算"同规格"）
1. **几何同一份文件**：`base_spheres.csv` / `particles_init.csv` 由我们生成一次并 sha256 入清单，LIGGGHTS/LAMMPS 用 `read_data` 读同一坐标 —— 任何一方都没有机会"自造初始条件"。
2. **壁面同一套接触律**：基床球作为**不动粒子**进入同一 pair 列表（`group base` + `fix setforce/freeze`），没有单独的 wall 模型，避免 wall/grain 单位与法向约定错配。
3. **采样同一段代码**：三方 dump 统一成 `t,id,x,y,z,vx,vy,vz`，剖面/Bagnold/停机判定只跑 `analysis.py`。

## ③ case_spec 统一规格表

`artifacts/xval_chute2001/result.json → case_spec` 逐项落盘（含 hashes、sampling 口径、
`dt/T_contact`、`gamma_n`、`z_wall`）。11 个 case 的 `runs/xval_chute2001/<case>/`
下均有 `case.json`、`base_spheres.csv`、`particles_init.csv`、`chute.data`、`in.chute`。

## ④ 横向对照矩阵

| tool | metric | got | vs_ref_err | cross_diff | 状态 |
|---|---|---|---|---|---|
| ours | arrest state @θ=18°, h=1.09 d | flow（vx/v_ref=1.42, v_rms=5.89） | n/a | n/a | 实测（未收敛） |
| ours | arrest state @θ=20°, h=1.09 d | flow（1.59 / 5.78） | n/a | n/a | 实测（未收敛） |
| ours | arrest state @θ=20°, h=2.18 d | flow（1.22 / 4.63） | n/a | n/a | 实测（未收敛） |
| ours | 稳态 vx(z) 剖面 @θ=24°, A/B 分辨率 | **未经** | — | — | WITHHELD |
| liggghts | 全部 | 未取得 | — | — | WITHHELD |
| 第二 DEM | 全部 | 未取得 | — | — | WITHHELD |

我方另有短窗健康度数字（N=320，Lx=Ly=8 d，350 步）：最大重叠 6.2e-4 d（≪1 %，
硬球近似成立），5.7 ms/步（单核 numpy）。

## ⑤ 判据逐条勾叉

| 判据（tensor_xval_cases.metrics） | tol | 结果 |
|---|---|---|
| 稳态速度剖面 工具间差 | <15 % | ✗ **未测**（无 OSS 数值）→ WITHHELD |
| Bagnold 指数各工具 | ±20 % 内一致 | ✗ **未测** → WITHHELD |
| h_stop 两倾角点 | 端点 ±2° | ✗ **未测**（θ=18/20° 在 8.2 t_ref 内仍全流动，未观测停机）→ WITHHELD |

## ⑥ 收敛性/网格敏感性

只有我方：分辨率 A（Lx=Ly=10 d, h=10.05 d）与 B（Lx=Ly=7 d, h=10.04 d）的**输入**
已就绪（同厚度、不同周期盒/粒子数），但 A 的稳态长跑（settle 4e4 + window 2e4 步）
未在本轮时间预算内完成（本地负载 171/128，单核 ~12 ms/步 → 单 case ≈ 12 min）。
故本轮**没有**分辨率对比数字，不作为结论。

## ⑦ 失败模式与工件

1. **LIGGGHTS 工况未跑通（阻断项）**：deck 走 3.8 的 contact-model 框架
   （`pair_style gran model hooke tangential history` + `fix property/global …`），
   运行报
   `ERROR: Pair granular requires ghost atoms store velocity (../pair_gran.cpp:245)`；
   按 LAMMPS 经验补 `comm_modify vel yes` 时，本版本直接
   `ERROR: Unknown command: comm_modify vel yes (../input.cpp:259)`。
   另有 `thermo_style custom … pe` 在本构建不被接受（已绕开为 `ke`）。
   工件：`runs/xval_chute2001/arrest_t18_n300/liggghts/log.txt`（含完整报错上下文）。
2. **第二参照不可装**：MercuryDPM 经 `ghproxy.net` 解析失败；ESyS-Particle 需 boost
   （本机无 `/usr/include/boost`），未尝试完成；LAMMPS 克隆成功但 CMake 版本不足
   （LAMMPS 2025 要求 ≥3.27，本机 3.26.3），`pip install --user cmake` 亦未在预算内完成。
3. **我方 h_stop 未观测到**：`settle=8000` 步 = 8.2 t_ref，近临界 arrest 的松弛时间
   是 O(30–100) t_ref；三个薄层 case 仍是 flow（vx/v_ref 1.2–1.6），**不能据此说
   θ_c(18°)<18°**。
4. **集群不可用（生态摩擦）**：`cj995` 探针自 19:50 递交至 20:5x 仍在 `queued`，
   队列前方 214 个作业、最长排队 5 h+，两节点 load≈36 但无派发；全部构建/运行因此
   落回控制面本机（已按下述偏差声明记录）。

## ⑧ bug 判读与生态摩擦

* 我方：`chute_flow` 为新建能力，**未发现**与外部工具的数值不一致 —— 因为没有外部
  数值可比。已识别的**自家缺陷点**是预算/协议层面的：arrest 扫描的 settle 窗长
  不足以跨越近临界松弛（`locate: runs/xval_chute2001/arrest_*/run_spec.json settle=8000`），
  修复点=提到 60k–120k 步并加 KE 衰减触发的自适应停机。
* 对方/口径：LIGGGHTS-PUBLIC 3.8 的 **gran 模型框架与旧 `gran/hooke/history` 语法
  断裂**，且 serial 构建下 `pair_gran` 的 ghost 速度要求无可用的用户侧开关
  （`pair_gran.cpp:245` ↔ `input.cpp:259` 矛盾）——这是可复现的口径/可用性缺陷，
  已入 `negative_knowledge_ledger` 条目。
* 刚度口径差（须在下一轮消除）：LIGGGHTS `hooke` 模型的 k_n 由
  `E, ν, characteristicVelocity` 反推，按 `normal_model_hooke.h:269` 与
  E=5e6, ν=0.45, charVel=2√(gd) 解析得 **k_n≈459 N/m**，与我方 500 N/m 差 **−8.2 %**。
  下一轮应改用可直给 k_n 的模型（`hooke/stiffness`）或在 case_spec 显式登记该差。

## 复现命令

```bash
# 0) 我方单元测试（7 passed）
PYTHONPATH=src python3 -m pytest tests/test_chute_flow.py -q
# 1) 11 个 case 的共享规格与三方输入
PYTHONPATH=src python3 tools/xval_chute2001/harness.py build --outdir runs/xval_chute2001
# 2) 我方单 case
PYTHONPATH=src python3 tools/xval_chute2001/harness.py run --outdir runs/xval_chute2001 \
    --tool ours --cases arrest_t20_n600
# 3) LIGGGHTS（需 build/oss/liggghts/src/lmp_serial，构建见下）
python3 tools/xval_chute2001/harness.py run --outdir runs/xval_chute2001 \
    --tool liggghts --cases arrest_t20_n600
# 4) 构建 OSS（本机 loongarch64）
git clone --depth 1 https://ghproxy.net/https://github.com/CFDEMproject/LIGGGHTS-PUBLIC.git build/oss/liggghts
cd build/oss/liggghts/src/STUBS && make -j4 && cd .. && make serial -j10   # -> src/lmp_serial
```

## 边界与局限（必读）
* 本轮**没有任何工具间数值结论**；表中所有 WITHHELD 都是"没测到"，不是"一致"。
* 我方 arrest 结果取自 8.2 t_ref 的短窗，只证明"该窗内未停机"，不构成 θ_c 判据。
* LIGGGHTS 刚度口径与 Silbert 原文（线性弹簧直给 k_n）不同，即便跑通也须先消除该差。
* 全部计算落在控制面本机（集群队列 5 h 未派发），这是纪律允许外的偏差，已在
  `ecosystem_friction` 与本节显式记录。
