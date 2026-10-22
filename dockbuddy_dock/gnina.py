# -*- coding: utf-8 -*-
"""
libdock.gnina —— GNINA 深度学习重打分适配器(2026-10-08)
================================================================================
定位: 用 GNINA(gnina/gnina, GPL-2.0/Apache 双许可, 非商用无限制)的 CNN 打分对
Vina 已对接 pose 做**二次评估**, 区分"假结合构象", 输出 CNNscore/CNNaffinity 软指标。
与 OpenMM MM-GBSA(rescore.py)并列, 都是对接后 re-rank 证据线。

获取(预编译二进制, 免编译): github.com/gnina/gnina release 或 Docker 镜像 gnina/gnina;
自编译需 CUDA>=12 + OpenBabel3 + libmolgrid + torch。CNN 打分需 GPU(竞价 GPU 实例)。

调用约定(GNINA CLI):
  gnina -r rec.pdb -l lig.sdf --score_only --cnn_scoring rescore \
        --cnn default2018 -o scored.sdf.gz
  输出 SDF 的 property 字段: CNNscore / CNNaffinity / CNNvariance。
"""
from __future__ import annotations

import gzip
import os
import re
import shutil
import subprocess
from typing import Any, Dict, Optional

from ._core import LOG

_GNINA_HINT = (
    "GNINA 需要预编译二进制(gnina)或 Docker 镜像。获取: github.com/gnina/gnina "
    "(GPL-2.0/Apache)。CNN 打分走 CUDA, 需 GPU 实例。"
)


def _find_gnina() -> Optional[str]:
    return shutil.which("gnina")


def _parse_sdf_props(path: str) -> Dict[str, float]:
    """从 GNINA 输出 SDF(可 .gz)解析 CNNscore/CNNaffinity/CNNvariance。"""
    opener = gzip.open if path.endswith(".gz") else open
    txt = opener(path, "rt", encoding="utf-8", errors="ignore").read(200000)
    out: Dict[str, float] = {}
    for key in ("CNNscore", "CNNaffinity", "CNNvariance"):
        m = re.search(rf"^>\s*<{key}>\s*\n(\S+)", txt, re.M)
        if m:
            try:
                out[key.lower()] = float(m.group(1))
            except ValueError:
                pass
    return out


def gnina_rescore(rec_pdb: str, lig_sdf: str, out_sdf: str = "",
                  gnina_bin: str = "", cnn_model: str = "default2018",
                  cnn_scoring: str = "rescore") -> Dict[str, float]:
    """用 GNINA CNN 对单配体 pose 重打分, 返回 {cnnscore, cnnaffinity, cnnvariance}。

    参数:
      rec_pdb: 受体 PDB。
      lig_sdf: 配体 pose(SDF/MOL)。
      out_sdf: 输出 SDF 路径(缺省 <lig>_gnina.sdf.gz)。
      gnina_bin: gnina 可执行文件(缺省 which gnina / 环境变量 GNINA_BIN)。
      cnn_model: default2018 / dense / crossdock_default2018。
      cnn_scoring: rescore(缺省) / refinement / none。
    """
    gnina = gnina_bin or os.environ.get("GNINA_BIN", "") or _find_gnina()
    if not gnina:
        raise RuntimeError(_GNINA_HINT)
    out_sdf = out_sdf or (os.path.splitext(lig_sdf)[0] + "_gnina.sdf.gz")
    cmd = [gnina, "-r", rec_pdb, "-l", lig_sdf, "--score_only",
           "--cnn_scoring", cnn_scoring, "--cnn", cnn_model, "-o", out_sdf]
    LOG.info("GNINA rescore: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError("GNINA 失败:\n" + (proc.stderr or proc.stdout)[-2000:])
    out = _parse_sdf_props(out_sdf)
    out["out_sdf"] = out_sdf  # type: ignore[assignment]
    return out


__all__ = ["gnina_rescore"]
