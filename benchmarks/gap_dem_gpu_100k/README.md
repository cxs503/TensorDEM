# GAP-DEM-GPU-100K: 10万颗粒 GPU 实测吞吐达标

## 目标

验证 TensorDEM 的 DEM3D 求解器在 GPU 上对 10 万颗粒规模的单步计算耗时 ≤ 2 秒。

## 验收判据

| 指标 | 达标线 | 测量方法 |
|------|--------|----------|
| 100K 颗粒单步耗时 | ≤ 2 s | wall-clock time (perf_counter) |
| 强扩展 2 档规模 | 10k, 50k, 100k | 同一配置不同网格 |
| CPU 基线对比 | 有 | 同硬件同配置 |

## 运行方式

```bash
# CPU 基线
python benchmarks/gap_dem_gpu_100k/run.py --device cpu

# GPU 测试（需要 CUDA 环境）
python benchmarks/gap_dem_gpu_100k/run.py --device cuda

# 全量扫描（所有可用设备）
python benchmarks/gap_dem_gpu_100k/run.py --sweep
```

## 输出工件

- `results.json` — 机器可读结果（含 acceptance 字段）
- `report.md` — 人类可读报告
- `run_manifest.json` — 运行元数据

## 物理配置

- 颗粒半径: 0.025 m
- 密度: 917 kg/m³ (冰)
- 键刚度: 100 N/m (最小化以突出接触计算瓶颈)
- 接触刚度: 5000 N/m
- 阻尼: 5.0
- 网格: 22³ (10k), 37³ (50k), 47³ (100k)

## 协议版本

`tensordem-dem3d-gpu-throughput-v1`
