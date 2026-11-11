# P152W1-N04 休止角锚点 A/B：v1 协议两处缺陷的定位与修复（协议 v2）

节点：`P152W1-N04/repose-v2` · 仓：TensorDEM · 承接：`docs/P152W1-N04_friction_contact_surface.md`
（该节点锚点 A/B 以 **WITHHELD** 结账，本文件是它的根因收口）

改动面（最小）：
`src/tensordem/friction_surface/repose.py`（`ReposeCase.source_mode` / `nozzle_clear_time()` /
撒料簿记）、`src/tensordem/friction_surface/study.py`（`repose_anchor(source_mode=…)` 透传、
`--repose-particles` / `--source-mode`）、`scripts/p152w1_repose_v2_closure.py`（预算受限驱动）、
`scripts/p152w1_repose_v2_report.py`（工件→图表/表格/报告生成器）、
`tests/test_friction_surface_repose_v2.py`（7 passed）、
`prereg/P152W1_N04_repose_protocol_v2_prereg.json`（**只改协议，阈值逐字继承**）。

## 1 案例介绍（为什么吊在那儿）

P152W1-N04 §5.5 自己写了：休止角两个锚点（A 负控、B 实验带）"在节点窗口内没有拿到可裁决数据"，
判 WITHHELD。原因不是物理做不出来，而是协议与实现有缺陷——本轮把两处缺陷定量钉住：

| 缺陷 | 现象（实测） | 后果 |
|---|---|---|
| **D1 撒料越界** | 撒料批只在每步检查一次 `active < n_max`；一次"部分被拒"的批（堆体存在后是常态）就写 `pos[n_max]` → `IndexError: index 300 is out of bounds for axis 0 with size 300`（n=300 复现） | 扫掠跑到一半整个进程崩，标定扫掠永远跑不完 |
| **D2 喷嘴随堆爬升** | v1 每批把源抬到 `max(drop_height, apex + 5R)`——正是该节点 §5.3 自己列为"被否定协议"的"块状点源动态抬高"。实测 `source_mode="rising"`：喷嘴 `nozzle_z_max = 1.4403 m`、堆顶 0.6894 m、`reached_rest = False`、测出的"休止角" 50.8°（尖塔侧壁，不是安息角） | 停堆判据永不触发；测到的角度由倾倒排程决定而非摩擦 |

两处都不是"参数没调好"，是代码与协议错：D2 的行文已经被判过死刑，代码却还留着。

## 2 怎么做的（协议 v2）

1. **喷嘴固定**：`drop_height` 是常数，源高度永不随堆变化（`nozzle_z_max` 逐运行记录，测试断言它恒等于 `drop_height`）。
2. **撒料节奏 = 电荷穿出喷嘴区的自由落体时间**：`nozzle_clear_time() = (-v0 + sqrt(v0² + 2 g s)) / g`，
   `s = (2 + max(2.2, nozzle_clearance)) R`。只有喷嘴板层里"年轻于该时间"的球存在时才跳过本批（`bursts_skipped` 记账），
   因此**在飞球永不重叠**（v1 的喷泉机制），而且这个节奏是物理量，不是调出来的步数（测试用闭式值对拍 1e-12）。
3. **停料是几何的、不可重启**：喷嘴板层被**静止沉积**填满，或整批一个球都放不进去（2.2R 排斥），则 `pour_closed = True`，此后不再撒料。
4. 停堆判据、上包络径向拟合测角（`heap_surface_angle`）、材料微参数、实验带、阈值 **全部继承 v1，一字未改**。
5. 预注册：`prereg/P152W1_N04_repose_protocol_v2_prereg.json` 在任何生产计算前写下，显式声明
   "不许为把 WITHHELD 变 PASS 而放宽门限"，并预先登记了三条局限（文献带非本仓实测 / 球体+滚动阻力代替形状 / 未做有限尺寸收敛）。

## 3 计算结果

本轮**可以裁决的部分**（缺陷面）：

| 量 | v1 (`rising`) | v2 (`fixed`) | 判读 |
|---|---|---|---|
| 喷嘴最高点 | 1.4403 m（爬升 72× drop_height） | 0.0200 m ≡ `drop_height` | 缺陷 D2 消失 |
| 堆顶高度（同类算例） | 0.6894 m（尖塔） | 由摩擦与停料决定 | 缺陷 D2 消失 |
| 停堆 (`reached_rest`) | False | 由 RMS 判据给出 | 缺陷 D2 消失 |
| 撒料越界 | `IndexError`（n=300 复现） | 0 次（数组上界在批内逐球检查） | 缺陷 D1 修复 |
| 单元/回归测试 | — | `tests/test_friction_surface_repose_v2.py` **7 passed** | — |

**锚点 A/B 的休止角数值本轮未收口**：预算受限驱动
（`scripts/p152w1_repose_v2_closure.py --particles 200 --jobs 8`）与本仓 12 进程全量扫掠
（`python -m tensordem.friction_surface.study --stage repose --repose-particles 300`）在节点窗口内仍在跑，
窗口结束时 `artifacts/P152W1-N04/repose.json` 尚未落盘。按诚实记账规则：
**锚点 A/B 仍记 WITHHELD（不因"协议修好了"就算过），不预设 PASS 也不预设 FAIL。**

收口命令（工件落盘后零手抄出数）：

```bash
PYTHONPATH=src python -m tensordem.friction_surface.study --stage repose \
    --jobs 12 --repose-particles 300 --out artifacts/P152W1-N04      # 全量：5 步二分 × 3 种子
PYTHONPATH=src python scripts/p152w1_repose_v2_closure.py --particles 200 --jobs 8   # 受限：3 步二分 × 1 种子
PYTHONPATH=src python scripts/p152w1_repose_v2_report.py             # 图 + 结果表 + 平台报告
PYTHONPATH=src python -m tensordem.friction_surface.study --stage report --out artifacts/P152W1-N04
```

## 4 结果图

本轮无图，理由：图（沉积体纵剖面 + μ_r→θ_r 标定曲线 + θ_DEM vs 实验带）是
**锚点数值的呈现**，而锚点数值本轮未落盘；用缺陷面的数字画图会让人误读成"锚点已裁决"。
`scripts/p152w1_repose_v2_report.py` 已把三幅图画法写死（读 `repose.json` 的
`figure_positions` / `calibration.bisection_history` / `materials`），工件一到即出图，不需要手工改脚本。

## 5 注意点

1. **节奏与预算耦合**：v2 的撒料节奏是电荷落体时间（≈805 步/批 @ dt=4.55e-5 s），
   因此 n=300 的堆需要 ≈6 万步；这是 v1 "40 步一批"的 20 倍，也是本轮窗口不够的**真实原因**。
   这是物理保真换来的代价：v1 那种快节奏必然靠抬喷嘴躲重叠，就是 D2。
2. **停料板层高度**：`zone_lo = drop_height - max(2.2, nozzle_clearance)R`。喷嘴太低（<5R）时板层贴近地面，
   第一批沉积就会触发停料——测试 `test_pour_closes_when_the_deposit_reaches_the_nozzle` 正用这个极端几何做单元校验；
   生产算例 `drop_height = 10R` 不受影响。
3. **不继承的账**：v1 的"整体柱塌落""Casagrande 分裂剪切盒"两个被否定协议，本文件不回填、不洗掉。
4. 桥接脚本 `scripts/p152w1_repose_v2_report.py` 在本轮**未端到端跑过**（缺工件），已如实标注；
   它引用的字段全部由本轮新增的簿记键提供，字段存在性由 `tests/test_friction_surface_repose_v2.py`
   的返回字典断言间接覆盖。

## 6 误差判定

| 判据 | 基准 | 本轮实测 | 误差 | 是否达标 |
|---|---|---|---|---|
| 撒料不越界 | 无 `IndexError`，`n ≤ n_max` | 0 例外（7 项测试含 n=7 非整除用例） | — | **是** |
| 喷嘴不变 | `nozzle_z_max ≡ drop_height` | 差 0（`abs=1e-12` 断言） | 0 % | **是** |
| v1 缺陷可复现 | 喷嘴 > 2×`drop_height` | 1.4403 m vs 0.02 m（72×） | — | **是（缺陷存在）** |
| 节奏 = 落体时间 | 闭式 `(-v0+sqrt(v0²+2gs))/g` | 相对误差 < 1e-12 | ≈0 % | **是** |
| 测量层回真 | 合成 30° 锥面 | 30.19°（修复测试的锥面朝向后的结果） | 0.19° (0.6 %) | **是（<3 %）** |
| A 负控 θ(μ=μ_r=0) < 5° | — | **未裁决** | — | **WITHHELD** |
| B 玻璃珠 θ vs 23°(<10 %) | 23.0° | **未裁决** | — | **WITHHELD** |
| B 石英砂 θ vs 32°(<10 %，样本外) | 32.0° | **未裁决** | — | **WITHHELD** |

**是否 <3%？** 缺陷面三项（喷嘴不变、节奏闭式、测量层回真 0.6 %）**是**；
休止角/μ_eff = tan θ 两项**无数据，不声明**（不拿"协议修好了"当精度声明）。
