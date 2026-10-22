# -*- coding: utf-8 -*-
"""paths.py —— dockbuddy_dock 的路径常量(可配置, 与 DockBuddy 解耦)。"""
import os

# 原始 PDB 缓存(RCSB 下载结果落此, 重复运行不重复下载)
RAW_PDB_CACHE = os.environ.get(
    "DOCK_RAW_PDB_CACHE", os.path.join(os.getcwd(), "cache", "raw_pdb"))
