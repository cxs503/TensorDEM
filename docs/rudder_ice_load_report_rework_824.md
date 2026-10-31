# 报告 824 返工记录：图件缺失修复 → 重发报告 id=831（mission 995）

## 缺陷与修复对应

| # | 缺陷（返工单） | 修复 | 证据 |
|---|---|---|---|
| 1 | 报告 ④ 节仅文字描述 5 张图，无图件嵌入/图片链接 | 6 张 PNG 全部以 `png_base64` 写入新报告 `figures` 字段，并在 markdown ④ 节按图 1–图 6 顺序用 `![](../artifacts/rudder_ice_load/figures/<name>.png)` 引用 | POST 200 → 报告 id=831；`figures` 6 项；markdown 中 `![` 计数 = 6；服务端落盘 `backend/uploads/tensor_reports/831/*.png` 与工件 **byte-identical**（md5 逐个比对 6/6 一致） |
| 2 | 场量云图与比较曲线两类均未展示 | 比较曲线类 = 图 1 载荷时历、图 2 接触压力/面积演化、图 3 冰厚扫描、图 4 强度标定、图 5 设计载荷图谱；**新增**空间场量图 = 图 6 舵面接触压力场（三个压入阶段 + 每颗粒压力分布） | 图 6 由 `artifacts/rudder_ice_load/plot_pressure_field.py` 生成（数据记录 `pressure_field.json`） |
| 3 | 报告需重写并引用原报告 id=824 | 新报告标题/摘要/正文均声明"取代 id=824"，tags 追加 `rework-report-824`、`report-v2`，原 7 个标签全部保留 | tags = 9 项，含原 7 项 |
| 4 | 仅改报告文件与报告 POST，禁改业务代码 | 改动仅 `docs/`、`artifacts/`；DEM-FSM 模块从 `agent/m995`（W5-N01）只读导入，未改一行业务代码 | `git status` 仅 docs/artifacts 路径 |

## 新增图 6（舵面接触压力场）的关键读数

| 压入 [mm] | 接触颗粒 | CRUSHING | CLEARED | 面内法向合力 [kN] | 有效压力 [MPa] |
|---|---|---|---|---|---|
| 25 | 20 | 20 | 0 | 1863 | 5.18 |
| 60 | 20 | 0 | 20 | 1118 | 3.11 |
| 135 | 20 | 0 | 20 | 867 | 2.41 |

最终准稳态每颗粒接触压力 3.21 MPa = 0.60·σ_c（碎冰残余压力 f_res）；逐颗粒压力标量和 1156 kN
与面内矢量和 866.7 kN 相差 33.3 %，差值为盒型 SDF 法向在舵面边缘偏转所致 —— 规范比对必须用矢量和
（有效压力 2.41 MPa = 0.45·σ_c）。压力场沿舵面近似均匀，是宽平（flat-on-flat）压入的物理结果，
空间结构主要体现在接触面积与状态机时序（压溃 5.18 MPa → 残余 3.11 MPa），见报告 ⑤-2。

## 验证（本单实际执行）

* `pytest tests/test_rudder_ice_load.py`（agent/m995 工作树，只读复跑）：**24 passed**（56 s）。
* 场量脚本重跑参考工况：包络 866.7 kN、断键 225、剥落事件 20，与 campaign `report.json` 完全一致（同源复现）。
* `POST /tensor/reports` → HTTP 200，返回 id=831，`mission_id=995`、`scenario_key=ice_mechanics`、
  `status=published`；6 张图的 `png_base64` 全部被服务端接受（服务端校验 PNG magic，失败会回滚并返回 422）。
* 服务端落盘文件与工件 md5 6/6 一致。

## 未完成 / 局限（如实记录）

* 部署中的 :8010 实例未暴露 `GET /tensor/reports/{id}/figures/{name}`（对 824 与 831 均 404，
  带 Bearer 亦然）——该路由存在于仓库当前代码中但未部署，属平台侧部署滞后，不在本单 EDIT_WHITELIST
  范围内；因此报告页的图片展示依赖正文 markdown 图片链接与 `figures` 内嵌 base64 两条路径。
* 场量图为 20 个 0.15 m 颗粒胞元的离散采样，不是连续云图；面内梯度被粗粒化抹平（需 0.06 m 量级加密）。
* 本单未重跑整套 campaign（仅复跑参考工况），其余算术量取自 mission 995 的 `report.json`。