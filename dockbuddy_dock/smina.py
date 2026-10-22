# -*- coding: utf-8 -*-
"""libdock.smina —— smina 对接薄适配器(2026-10-11 新增, 环节 4 对接)
===================================================================================
smina(Koes et al. 2013)是 AutoDock Vina 的 fork(支持自定义打分函数)。Linux 二进制
(conda -c bioconda smina), 本机无。

诚实边界: smina 基于 Vina 1.1.2 打分(与项目主 vina 1.2.5 不同口径), 亲和力数值
**禁止与 vina 1.2.5 结果混比**(项目既有纪律)。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

_SMINA_BIN_ENV = "SMINA_BIN"
_CANDIDATES = ("smina", "smina.exe")


class SminaError(RuntimeError):
    pass


def find_smina_bin(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        if not os.path.isfile(explicit):
            raise SminaError(f"指定的 smina 二进制不存在: {explicit}")
        return explicit
    env = os.environ.get(_SMINA_BIN_ENV, "").strip()
    if env:
        return env
    for name in _CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    return None


def smina_dock(receptor_pdbqt: str, ligand_pdbqt: str, *,
               center: Tuple[float, float, float],
               size: Tuple[float, float, float] = (22, 22, 22),
               bin_path: Optional[str] = None, exhaustiveness: int = 8,
               num_modes: int = 9, out_pdbqt: str = "", timeout: int = 600,
               seed: int = 42) -> Dict[str, Any]:
    """跑 smina 对接, 返回 {affinities: [(mode, kcal_mol)], best, out_pdbqt}。"""
    bin_path = find_smina_bin(bin_path)
    if bin_path is None:
        raise SminaError(
            "未找到 smina 二进制; 设置 SMINA_BIN 或 conda -c bioconda smina(需 Linux)")
    if not os.path.isfile(receptor_pdbqt):
        raise SminaError(f"受体 PDBQT 不存在: {receptor_pdbqt}")
    if not os.path.isfile(ligand_pdbqt):
        raise SminaError(f"配体 PDBQT 不存在: {ligand_pdbqt}")
    out = out_pdbqt or "smina_out.pdbqt"
    args = [bin_path, "--receptor", receptor_pdbqt, "--ligand", ligand_pdbqt,
            "--center_x", str(center[0]), "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
            "--exhaustiveness", str(exhaustiveness), "--num_modes", str(num_modes),
            "--seed", str(seed), "--out", out]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)

    affinities: List[Tuple[int, float]] = []
    for line in proc.stdout.splitlines():
        m = re.match(r"^\s*(\d+)\s+(-?\d+\.?\d*)\s", line)
        if m:
            affinities.append((int(m.group(1)), float(m.group(2))))
    best = min((a for _, a in affinities), default=None)
    return {"ok": bool(affinities), "affinities": affinities, "best_kcal": best,
            "out_pdbqt": out, "bin_path": bin_path,
            "stdout_tail": proc.stdout[-300:]}


__all__ = ["find_smina_bin", "smina_dock", "SminaError"]
