# -*- coding: utf-8 -*-
"""libdock.fpocket —— fpocket 口袋检测薄适配器(2026-10-11 新增, 环节 5 口袋)
===================================================================================
fpocket(Le Guilloux et al. 2009)基于 alpha spheres 检测蛋白表面口袋, 输出口袋
数量/体积/得分/可成药性。Linux 二进制(conda -c bioconda fpocket), 本机无。

诚实边界: fpocket 检测的是「所有几何口袋」(alpha spheres 聚类), 与项目 pocket_tunnel
的「Ser OG 球心 + 底物可达腔」口径不同, 两体积不可直接相减。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

_FPOCKET_BIN_ENV = "FPOCKET_BIN"
_CANDIDATES = ("fpocket", "fpocket.exe")


class FpocketError(RuntimeError):
    pass


def find_fpocket_bin(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        if not os.path.isfile(explicit):
            raise FpocketError(f"指定的 fpocket 二进制不存在: {explicit}")
        return explicit
    env = os.environ.get(_FPOCKET_BIN_ENV, "").strip()
    if env:
        return env
    for name in _CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    return None


def _parse_info(info_path: str) -> List[Dict[str, float]]:
    """解析 fpocket *_info.txt → [{score, volume, druggability, alpha_spheres}, ...]。"""
    pockets: List[Dict[str, float]] = []
    cur: Dict[str, float] = {}
    with open(info_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip().startswith("Pocket "):
                if cur:
                    pockets.append(cur)
                cur = {}
            m = re.search(r"^\s*([\w\s\.]+?)\s*:\s*([0-9.]+)", line)
            if m:
                key = m.group(1).strip().lower()
                val = float(m.group(2))
                mapping = {"score": "score", "druggability score": "druggability",
                           "volume": "volume", "number of alpha spheres": "alpha_spheres"}
                if key in mapping:
                    cur[mapping[key]] = val
    if cur:
        pockets.append(cur)
    return pockets


def fpocket_detect(receptor_pdb: str, *, bin_path: Optional[str] = None,
                   out_dir: str = "", timeout: int = 300) -> Dict[str, Any]:
    """跑 fpocket 口袋检测, 返回 {pockets, out_dir, bin_path}。"""
    bin_path = find_fpocket_bin(bin_path)
    if bin_path is None:
        raise FpocketError(
            "未找到 fpocket 二进制; 设置 FPOCKET_BIN 或 conda -c bioconda fpocket(需 Linux)")
    if not os.path.isfile(receptor_pdb):
        raise FpocketError(f"受体 PDB 不存在: {receptor_pdb}")
    workdir = out_dir or os.path.dirname(os.path.abspath(receptor_pdb))
    os.makedirs(workdir, exist_ok=True)
    base = os.path.basename(receptor_pdb)
    proc = subprocess.run([bin_path, "-f", base], cwd=workdir,
                          capture_output=True, text=True, timeout=timeout)
    info = os.path.join(workdir, f"{os.path.splitext(base)[0]}_out",
                        f"{os.path.splitext(base)[0]}_info.txt")
    pockets = _parse_info(info) if os.path.isfile(info) else []
    return {"ok": bool(pockets), "pockets": pockets, "n_pockets": len(pockets),
            "out_dir": workdir, "bin_path": bin_path,
            "stdout_tail": proc.stdout[-300:]}


__all__ = ["find_fpocket_bin", "fpocket_detect", "FpocketError"]
