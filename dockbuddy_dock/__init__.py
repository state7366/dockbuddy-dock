# -*- coding: utf-8 -*-
"""dockbuddy_dock —— 分子对接引擎(从 DockBuddy libdock 抽出的解耦版本)。

含通用对接(引擎/盒/配体/受体/pose/几何/质子化/多构象/重打分) + 对接盒决策所需的
三联体/口袋几何/催化取向 + 隐式溶剂 + 报告。
"""
from .config import DockResult, DockingConfig
from .backends import DockingBackend, VinaBackend, VinaGPUBackend, LeDockBackend
from .tools import find_obabel, find_vina, find_vina_gpu, fix_babel_datadir
from .reliability import reliability_thresholds, classify_reliability_detail

__version__ = "0.1.0"

__all__ = ["DockingConfig", "DockResult", "DockingBackend", "VinaBackend",
           "VinaGPUBackend", "LeDockBackend", "find_obabel", "find_vina",
           "find_vina_gpu", "fix_babel_datadir", "reliability_thresholds",
           "classify_reliability_detail", "__version__"]
