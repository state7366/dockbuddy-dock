# -*- coding: utf-8 -*-
"""
libdock.caver —— CAVER 3.0 口袋通道分析适配器(2026-10-08)
================================================================================
定位: 用 CAVER 3.0(GPL-3.0, 纯 Java, CPU)识别蛋白口袋通道/隧道, 作为自研
pocket-tunnel 的**可选替代后端**(transport=local)。同一「通道分析」功能下两个
模型并存, 由 manifest 配置 `tunnel.adapter = caver | pocket_tunnel` 切换。

获取: http://www.caver.cz/fil/download/caver30/302/caver_3.0.2.zip
  (含 caver.jar + bin/ + config 模板; GPL-3.0, 无需 GPU, 可跑竞价 CPU 实例)

调用约定(CAVER 3.0 CLI, 实测 2026-10-08 校准):
  java -Xmx2g -cp caver.jar caver.ui.Launcher -home <caver_dir> \
       -conf <config.txt> -out <out_dir> -pdb <pdb_dir> [-log <log>]
  config.txt 关键项(起始点用原子序号, 非坐标):
    starting_point_atom <atom_serial>   起始点(如催化 Ser OG 的原子序号)
    probe_radius <R>  shell_radius <R>  shell_depth <R>
    clustering average_link  clustering_threshold <R>
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from ._core import LOG

_CAVER_HINT = (
    "CAVER 3.0 需要 Java 运行时 + caver_3.0.2.zip 解压后的 caver.jar。下载: "
    "http://www.caver.cz/fil/download/caver30/302/caver_3.0.2.zip (GPL-3.0)。"
)


def _find_java() -> Optional[str]:
    j = shutil.which("java")
    return j


def _find_jar(caver_dir: str) -> Optional[str]:
    """在 caver_dir 下递归找 caver.jar(不同版本解压层级不同, 如 caver_3.0/caver/)。"""
    for root, _dirs, files in os.walk(caver_dir):
        if "caver.jar" in files:
            return os.path.join(root, "caver.jar")
    return None


def _write_config(path: str, start_atom: int, probe_radius: float = 1.4,
                  shell_radius: float = 4.0, shell_depth: float = 4.0,
                  clustering_threshold: float = 3.5) -> None:
    """写 CAVER 3.0 config.txt(最小可用集, 起始点用原子序号)。"""
    lines = [
        "starting_point_atom %d" % start_atom,
        "probe_radius %.1f" % probe_radius,
        "shell_radius %.1f" % shell_radius,
        "shell_depth %.1f" % shell_depth,
        "clustering average_link",
        "clustering_threshold %.1f" % clustering_threshold,
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _parse_bottleneck(summary_dir: str) -> Dict[str, float]:
    """从 summary.txt 解析最小瓶颈半径(Å)。无隧道时返回空 dict。"""
    summary = os.path.join(summary_dir, "summary.txt")
    if not os.path.exists(summary):
        return {}
    # summary.txt 数据行: "  1     5     3  1.23  ..." —— 第 4 列 Avg_BR
    min_r = None
    for line in open(summary, encoding="utf-8", errors="ignore"):
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                r = float(parts[3])
            except ValueError:
                continue
            min_r = r if min_r is None else min(min_r, r)
    if min_r is None:  # 兜底: 匹配 "Minimum bottleneck radius" 文本(旧版)
        for line in open(summary, encoding="utf-8", errors="ignore"):
            m = re.search(r"minimum\s+bottleneck\s+radius[^0-9]*([0-9.]+)", line, re.I)
            if m:
                min_r = float(m.group(1))
                break
    return {"bottleneck_radius": min_r} if min_r is not None else {}


def caver_tunnels(pdb_dir: str, out_dir: str, start_atom: int,
                  caver_dir: str = "", probe_radius: float = 1.4,
                  java: str = "") -> Dict[str, Any]:
    """跑 CAVER 3.0, 返回 {bottleneck_radius, out_dir}(无隧道时 bottleneck_radius 缺省)。

    参数:
      pdb_dir: 含受体 PDB 的目录(CAVER 会分析其中全部 PDB)。
      out_dir: 结果输出目录。
      start_atom: 起始点**原子序号**(如催化 Ser OG 的 PDB 原子序号)。
      caver_dir: CAVER 解压目录(含 caver.jar, 缺省取环境变量 CAVER_HOME)。
      java: java 可执行文件(缺省自动 which)。
    """
    caver_dir = caver_dir or os.environ.get("CAVER_HOME", "")
    jar = _find_jar(caver_dir) if caver_dir else None
    java_exe = java or _find_java()
    if not java_exe or not jar:
        raise RuntimeError(_CAVER_HINT)
    # `-home` 须指向 caver.jar 所在目录(解压根 caver_3.0/caver/, 含 bin/ 模板等),
    # 而非用户传入的解压目录(可能只到 caver_3.0)。
    home = os.path.dirname(jar)

    os.makedirs(out_dir, exist_ok=True)
    conf = os.path.join(out_dir, "config.txt")
    _write_config(conf, start_atom, probe_radius)
    cmd = [java_exe, "-Xmx2g", "-cp", jar, "caver.ui.Launcher",
           "-home", home, "-conf", conf, "-out", out_dir, "-pdb", pdb_dir]
    LOG.info("CAVER 3.0: %s", " ".join(cmd[:6]) + " ...")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError("CAVER 3.0 失败:\n" + (proc.stderr or proc.stdout)[-2000:])
    out = _parse_bottleneck(out_dir)
    out["out_dir"] = out_dir
    return out


__all__ = ["caver_tunnels"]
