# -*- coding: utf-8 -*-
"""reliability.py —— 对接结果可靠性阈值 + warn 二分(真问题 real vs 保守提醒 soft)。

从 DockBuddy common.py 归位，供对接引擎与报告层复用同一套判定口径。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

_DEFAULT_THRESHOLDS = {
    "contact_unreliable": 0.3,
    "contact_warn": 0.5,
    "buried_clash": 3,
    "buried_frac": 0.15,
    "energy_abs_max": 50.0,
    "heavy_max": 40,
    "top_rmsd_warn": 2.0,
    "pose_scatter_warn": 3.0,
    "nuc_offsite": 5.0,
}


def reliability_thresholds() -> Dict[str, float]:
    """返回可靠性阈值(dict); 可用 DOCK_REL_<KEY> 环境变量覆盖。"""
    out = dict(_DEFAULT_THRESHOLDS)
    for k in out:
        v = os.environ.get("DOCK_REL_" + k.upper())
        if v is not None:
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def classify_reliability_detail(s: Dict[str, Any],
                                TH: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """把一条对接结果的 warn 二分为 '真问题'(real) 与 '保守提醒'(soft)。

    返回 {warn_class, warn_reasons, real_reasons, pose_div, pose_scatter}。
    """
    TH = TH or reliability_thresholds()
    buried = bool(s.get("ligand_buried"))
    energy_reliable = s.get("energy_reliable", True)
    e = s.get("energy_kcal_mol")
    ct = s.get("pocket_contact_frac")
    box = s.get("box_assumption", "")
    n_heavy = s.get("ligand_n_heavy", 0) or 0
    has_nuc = s.get("has_nucleophile", False)
    nuc = s.get("nuc_to_ligO_A")
    top_rmsd = s.get("top_pose_rmsd_A")
    pose_set_median = s.get("pose_set_rmsd_median_A")

    real: List[str] = []
    if buried:
        real.append("buried")
    if (e is not None and abs(e) > TH["energy_abs_max"]) or (not energy_reliable):
        real.append("energy_distorted")
    if ct is not None and ct < TH["contact_unreliable"]:
        real.append("no_contact")
    elif ct is not None and ct < TH["contact_warn"]:
        real.append("weak_contact")
    if box == "protein_centroid":
        real.append("blind_box")
    if n_heavy > TH["heavy_max"]:
        real.append("overweight")
    if has_nuc and nuc is not None and nuc > TH["nuc_offsite"]:
        real.append("nuc_offsite")

    pose_div = top_rmsd is not None and top_rmsd > TH["top_rmsd_warn"]
    pose_scatter = pose_set_median is not None and pose_set_median > TH["pose_scatter_warn"]

    rl = s.get("reliability")
    if rl == "unreliable":
        warn_class = "real"
    elif rl == "warn":
        warn_class = "real" if real else ("soft" if (pose_div or pose_scatter) else "")
    else:
        warn_class = "real" if real else ""

    warn_reasons = list(real) + (["pose_div"] if pose_div else []) \
        + (["pose_scatter"] if pose_scatter else [])
    if warn_class == "real" and not warn_reasons:
        warn_reasons = [f"unreliable(旧版 summary 无 metric 字段: {rl})"]
    return {"warn_class": warn_class, "warn_reasons": warn_reasons,
            "real_reasons": real, "pose_div": pose_div, "pose_scatter": pose_scatter}
