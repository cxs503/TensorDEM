"""Publish the m3001 progress report to the tensor platform."""
import base64
import io
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(BASE + path, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode() or "{}")


def figure(path, caption, width=1000, quality=78):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if image.width > width:
        height = int(image.height * width / image.width)
        image = image.resize((width, height), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return {
        "name": os.path.basename(path).replace(".png", ".jpg"),
        "caption": caption,
        "png_base64": base64.b64encode(buffer.getvalue()).decode(),
    }


MARKDOWN = r"""
# DEM3D 接触/键合力核的不变量审计：发现并修复「监测改变轨迹」缺陷

## ① 案例介绍（物理背景与参数）

本仓所有离散元数字都经过同一个函数 `IceDEM3D.forces()`。它的**本构分项**此前已逐个验证
（Hertz 律、恢复系数、Coulomb 摩擦锥、滚动阻力、能量账本），但围绕它们的**记账层**没有任何审计：
空间网格邻近搜索、键/接触互斥掩码、`index_add_` 累加、粒子标号与配对顺序无关性、以及
「探针是否无副作用」。记账类错误是研究代码最危险的一类——它产生有限、光滑、看似合理但错误的力，
单算例测试全部通过，因为每个算例只是复用了同一条错误路径。

审计对象：三维黏结颗粒冰样，简单立方 + 面对角/体对角键，48 颗粒 / 326 键，粒径 25 mm，
密度 917 kg/m³，键刚度 2.0 kN/m，接触刚度 5.0 kN/m。物理上对应破冰 DEM 的冰片试样；
核层正确性是「冰载荷可以用于工程结论」的必要前置条件。

| 参数 | 值 |
| --- | --- |
| 晶格 | nx×ny×nz = 4×4×3（48 颗粒），326 键 |
| 粒径 / 密度 | 25 mm / 917 kg/m³ |
| 键刚度 / 接触刚度 | 2000 N/m / 5000 N/m |
| 位移扰动 | 0.5 R（完整晶格）或 1.2 R（强压实 + 预断裂） |
| 初速度 | ±0.05 m/s |
| 时间步 | 3.397e-5 s（配置自动下限） |
| 随机种子 | 0,1,2,3（私有 `torch.Generator`，与全局 RNG 无关） |
| 算例数 | 5 个（完整 1 + 预断裂 4）× 4 seed = 132 个不变量样本 |

## ② 怎么做的（方法 / 流程 / 关键设置）

审计配置**强制封闭**：`drag=0`、无固定边界、无压板、破冰头停在试样上方 5 m（其分支贡献严格为 0），
因此每条不变量都是关于「颗粒-颗粒核」的陈述。每个算例实例化两次：一次完整晶格，一次
**预断裂**（先提交一次断裂步，再把得到的 `alive` 掩码回放到每一个新建实例），保证比较双方看到
同一套相互作用集合。

七条蜕变不变量（metamorphic invariants）：

| 不变量 | 断言 |
| --- | --- |
| `momentum_closure` | 封闭体系 ⇒ ΣF_i = 0，且破冰头反力严格为 0（牛顿第三定律） |
| `translation` | F(x+t) = F(x)，刚性平移不变 |
| `rotation` | F(Rx, Rv) = R·F(x,v)，R 为 QR 生成的真旋转 |
| `brute_force` | 引擎力 == 独立 O(N²) 双循环参考（跳过存活键），不共享任何空间哈希代码 |
| `determinism` | 两个独立新实例逐位一致；同一实例上的探针逐位一致 |
| `load_superposition` | F(loads) − F(0) = loads（精确可加） |
| `reciprocity` | K = ∂F/∂x 中心差分对称（弹性路径 ⇒ 力场是势场梯度） |

工具链：`tensordem.contact_invariants`（审计核）→ `scripts/audit_dem3d_contact_invariants.py`
（CLI，输出 JSON/CSV/MD/PNG）→ `tests/test_dem3d_contact_invariants.py`（13 个测试 / 38 子测试）。
运行：`python scripts/audit_dem3d_contact_invariants.py --figure`，
`python -m pytest tests/test_dem3d_contact_invariants.py -q`。

## ③ 计算结果（程序化生成，零手抄）

最坏相对残差（4 seed × 5 算例）与阈值对比：

| 不变量 | 最坏残差 | 阈值 | 判定 |
| --- | --- | --- | --- |
| momentum_closure | 6.0e-16 | 1e-12 | PASS |
| translation | 7.5e-14 | 1e-12 | PASS |
| rotation | 1.7e-15 | 1e-12 | PASS |
| brute_force | 1.6e-16 | 1e-12 | PASS |
| determinism | 0.0（逐位） | 0 | PASS |
| load_superposition | 2.4e-16 | 1e-12 | PASS |
| reciprocity | 2.4e-10 | 1e-8 | PASS |

覆盖度证据（防止「空审计」）：预断裂算例 `broken_bonds = 326/326`、`unbonded_overlaps = 40`
个真实接触对（接触力量级 1.75e2 N 线性 / 4.67e5 N Hertz），完整算例 `bonded_pairs = 326` 且
`contact_scale = 0`（构造上无接触），切向刚度尺度 1.1e4–1.1e5 N/m，旋转矩阵行列式 1−2e-16。
合计 132 个样本，违规 0 个。

## ④ 结果图

左图：全部（不变量 × 算例）最坏残差的对数条形图与阈值线（蓝=通过）；
右图：监测纯净性（monitor purity）三项偏差，修复后全部落在 float64 舍入底噪。

## ⑤ 注意点（踩坑 / 局限 / 适用范围）

* **空审计陷阱（本次实际踩到）**：第一版 brute-force 比对残差 0.0 通过——因为完整晶格中所有
  重叠对都还带着键，接触律被掩码清零，比对只测到键项。现在不变量必须报告
  `unbonded_overlaps`，测试断言断裂算例 > 10、完整算例 = 0；未来若有改动让接触不再被真正求值，
  测试会失败而不是静默通过。
* **摩擦历史本身是有状态的**：`update_history=True` 时连续两次 `forces()` 合法地累加两次滑移增量；
  因此 `determinism` 对复用实例用探针语义（`update_history=False`），对两个独立新实例用逐位相等。
* **陈旧历史键（已记录未修）**：`_tangential_state` 以颗粒对为键，键断裂或颗粒分离后从不清理，
  内存随「曾经接触过的对数」增长，重新接触的对会沿用它旧的滑移量。修改会改变摩擦物理，故本次
  只记录为已知局限。滚动阻力矩累加器仅用于诊断且从不由步进清零，属运行和；探针不再使其膨胀。
* **范围**：只覆盖颗粒-颗粒键/接触项；破冰头、压板、外载分支只通过 `load_superposition` 与
  「远距离破冰头反力严格为零」间接覆盖。brute-force 参考不复现历史相关的切向弹簧，故带摩擦
  算例只用闭合/对称类不变量审计。`reciprocity` 只对超弹性路径有意义，耗散算例跳过。
* 预断裂算例全部断键（326/326），因此「部分键存活 + 部分真实接触」的混合拓扑只是间接覆盖，
  这是下一步的自然增量。

## ⑥ 缺陷：探针改变了摩擦历史（已修复）

`IceDEM3D.diagnostics()` 为报告反力而调用 `forces()`；在摩擦路径上，`forces()` 同时积分并
**写回** Cundall–Strack 切向位移（`tang_disp += v_tangent·dt`）。于是**每次诊断调用都向摩擦历史
注入一次多余的滑移增量**——观测一个算例就改变了它。CLI 和所有绘图脚本每个输出步调一次
`diagnostics()`，即默认使用路径上就有这个问题。

量化（`linear_friction_fractured`，μ=0.4，60 步，dt=3.4e-5 s）：修复前被监测与未被监测的两个
孪生算例最终位移相差 **1.53e-4 m（0.6 % 粒径）**，探针级力的相对偏差 1.7e-4（由 `determinism`
不变量触发发现）；修复后两者**严格为 0.0**。

修复（最小、默认保序）：`forces(..., update_history=True)` 关键字；切向增量、切向状态写回、
诊断用滚动阻力矩累加均由它守卫；`diagnostics()` 传 `update_history=False`。步进语义不变，
所有既有结果逐位可复现——修复只是不让**探针**扰动状态。回归：`MonitorPurityTests` 断言
监测/未监测轨迹零偏差、探针下切向状态不变，**并且**步进路径仍然积分滑移（不能悄悄关掉摩擦）。
既有回归：`test_dem3d.py`、`test_contact_tangential.py`、`test_dem3d_pair_contact_damping.py`、
`test_dem3d_energy_audit.py`、`test_dem3d_hertz_contact.py`、`test_dem3d_external_load_balance.py`
共 75 项全部通过。

## ⑦ 误差判定（逐项，显式回答是否 <3%）

被审计的量是**浮点恒等式，不是物理预测**，平台 15 % 工程带不适用，也没有实验参考值可比：

| 关键量 | 基准 | 实测 | 与 3% 的关系 |
| --- | --- | --- | --- |
| 精确恒等式残差（闭合/平移/旋转/参考/可加） | 1e-12 相对阈值 | 最坏 7.5e-14 | 优于 3% 约 13 个数量级 |
| 切线刚度对称性残差 | 1e-8 相对阈值 | 2.4e-10 | 优于阈值约 40 倍 |
| 确定性残差 | 0（逐位） | 0.0 | 精确满足 |
| 监测纯净性轨迹偏差 | 0 | 0.0（修复前 1.53e-4 m） | 修复后精确为零 |

**是否 <3%？是，且余量约 13 个数量级**；但诚实读法是：本表证明的是**核记账正确性**，
不是冰物理精度。监测纯净性缺陷是真实的物理误差（60 步后达 0.6 % 粒径并随时间增长），现已严格为零。

## 证据指针

* 提交：`8007619`（分支 `agent/m3001`）
* 工件：`results-dem3d-contact-invariants/`（`audit_report.json`、`invariant_results.csv`、`invariant_summary.csv`、`contact_invariants.png`）
* 文档：`docs/dem3d_contact_invariants.md`
* 测试：`tests/test_dem3d_contact_invariants.py`（13 passed / 38 subtests）
"""


def main():
    token_response = post("/auth/login", {"username": USER, "password": PASSWORD})
    token = token_response.get("access_token") or token_response.get("token")
    if not token:
        raise SystemExit("login failed: " + json.dumps(token_response)[:400])
    payload = {
        "title": "DEM3D 接触核不变量审计：发现并修复「监测改变摩擦轨迹」缺陷（132 样本全绿）",
        "summary": (
            "用七条蜕变不变量 + 独立 O(N^2) 参考审计 IceDEM3D 接触/键合力核的记账层，"
            "132 个样本（5 算例×4 seed）全部通过（最坏恒等式残差 7.5e-14，对称性 2.4e-10）。"
            "审计发现真实缺陷：diagnostics() 走步进路径，每次探针向 Cundall-Strack 切向历史注入一次"
            "多余滑移增量，60 步后使被监测与未监测孪生算例位移相差 1.53e-4 m（0.6% 粒径）；"
            "以 forces(update_history=False) 修复后偏差严格为 0，步进语义与既有结果逐位不变。"
        ),
        "scenario_key": "tensordem-kernel-verification",
        "mission_id": 3001,
        "tags": ["TensorDEM", "verification", "metamorphic-testing", "contact-kernel",
                 "friction-history", "reproducibility", "m3001"],
        "markdown": MARKDOWN,
        "figures": [
            figure(
                "results-dem3d-contact-invariants/contact_invariants.png",
                "左：七条不变量 × 五个算例的最坏相对残差（对数轴）与阈值线，全部通过；"
                "右：监测纯净性三项偏差，修复后均落在 float64 舍入底噪（修复前位移偏差 1.5e-4 m）。",
            )
        ],
    }
    size = len(json.dumps(payload))
    print("payload bytes:", size)
    response = post("/tensor/reports", payload, token=token)
    print(json.dumps(response)[:600])


if __name__ == "__main__":
    main()