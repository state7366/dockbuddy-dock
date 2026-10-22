# dockbuddy-dock

分子对接引擎，从 DockBuddy `libdock` 抽出的解耦版本。

含通用对接（Vina/smina/GNINA 引擎、盒决策、配体/受体、pose 分析、几何、质子化、多构象、重打分、自对接、校准、对照、基准）+ 对接盒决策所需的三联体/口袋几何/催化取向 + 隐式溶剂 + 报告可靠性。

## 安装

```bash
pip install dockbuddy-dock \
  --index-url https://pypi.cnb.cool/vcinex/bioinfo/-/packages/simple
```

依赖 `dockbuddy-common` + `openbabel-wheel` + `numpy`；vina 二进制通过 `VINA_EXE` 环境变量或 `DOCK_TOOLS_DIR` 指定。

## 使用

```python
from dockbuddy_dock import DockingConfig, VinaBackend, find_vina
from dockbuddy_dock.reliability import reliability_thresholds
```

## 说明

- 工具定位：`find_vina` / `find_obabel` / `find_vina_gpu`，路径可配置（`DOCK_TOOLS_DIR` / `VINA_EXE`）。
- 原始 PDB 缓存：`DOCK_RAW_PDB_CACHE` 环境变量可配置。
- 对接结果可靠性：`reliability_thresholds` / `classify_reliability_detail`（报告层复用同一口径）。
