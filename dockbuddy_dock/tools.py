# -*- coding: utf-8 -*-
"""tools.py —— 对接工具定位(obabel/vina/vina-gpu)与 openbabel 修复(从 common 归位)。

与 DockBuddy 解耦：TOOLS_DIR 可配置(DOCK_TOOLS_DIR 环境变量 > 当前工作目录/tools)。
"""
from __future__ import annotations

import logging
import os
import sys

LOG = logging.getLogger("dockbuddy_dock")

IS_WINDOWS = (os.name == "nt")
TOOLS_DIR = os.environ.get("DOCK_TOOLS_DIR", os.path.join(os.getcwd(), "tools"))


def find_obabel() -> str:
    """定位 obabel。优先级: 环境量 > 捆绑 venv > PATH > dockbuddy_dock 兜底。"""
    env = os.environ.get("OBABEL_EXE")
    if env and os.path.exists(env):
        return env
    if IS_WINDOWS:
        cand = os.path.join(sys.prefix, "Scripts", "obabel.EXE")
    else:
        cand = os.path.join(sys.prefix, "bin", "obabel")
    if os.path.exists(cand):
        return cand
    from shutil import which
    p = which("obabel")
    if p:
        return p
    try:
        from dockbuddy_dock.obabel import _find_obabel
        return _find_obabel()
    except Exception:  # noqa: BLE001
        return "obabel"


def find_vina(explicit: str = "") -> str:
    """定位 vina。优先级: 显式路径 > 环境量 > tools/vina > PATH。"""
    if explicit and os.path.exists(explicit):
        return explicit
    env = os.environ.get("VINA_EXE")
    if env and os.path.exists(env):
        return env
    name = "vina.exe" if IS_WINDOWS else "vina"
    cand = os.path.join(TOOLS_DIR, name)
    if os.path.exists(cand):
        return cand
    from shutil import which
    p = which("vina")
    if p:
        return p
    return os.path.join(TOOLS_DIR, name)


def find_vina_gpu(explicit: str = "") -> str:
    """定位 Vina-GPU。优先级: 显式路径 > DOCK_VINA_GPU_EXE > tools 下 glob > 空串。"""
    if explicit and os.path.exists(explicit):
        return explicit
    env = os.environ.get("DOCK_VINA_GPU_EXE")
    if env and os.path.exists(env):
        return env
    import glob as _glob
    cands = []
    for pat in (os.path.join(TOOLS_DIR, "vina-gpu-2.0", "**", "Vina-GPU+.exe"),
                os.path.join(TOOLS_DIR, "vina-gpu-2.1", "**", "*.exe"),
                os.path.join(TOOLS_DIR, "vina-gpu*", "*.exe")):
        cands.extend(_glob.glob(pat, recursive=True))
    cands = [c for c in cands
             if "GUI" not in os.path.basename(c) and "uninstall" not in c.lower()]
    non_k = [c for c in cands if "_K" not in os.path.splitext(os.path.basename(c))[0]]
    if non_k:
        return non_k[0]
    return (cands[0] if cands else "")


def fix_babel_datadir() -> None:
    """修正 openbabel-wheel 的 BABEL_DATADIR 指向(Windows 打包缺陷)。幂等。"""
    try:
        import importlib.util
        spec = importlib.util.find_spec("openbabel")
        origin = getattr(spec, "origin", "") if spec else ""
        if not origin:
            return
        data = os.path.join(os.path.dirname(origin), "bin", "data")
        if not os.path.isfile(os.path.join(data, "UFF.prm")):
            return
        cur = os.environ.get("BABEL_DATADIR", "")
        if not cur or os.path.abspath(cur) != os.path.abspath(data):
            os.environ["BABEL_DATADIR"] = data
            LOG.info("已修正 BABEL_DATADIR -> %s", data)
    except Exception:  # noqa: BLE001
        pass
