# -*- coding: utf-8 -*-
"""
calibration.py —— 打分模型对实验 ΔΔG 的方法学校准(纯函数)
================================================================================
「分数到底有没有预测力」必须实测, 不能靠猜。本模块把「打分 vs 实验 ΔΔG」的相关性
计算做成纯函数:

  Spearman 秩相关  —— 排序一致性(稳健, 不受分数尺度影响), 主指标
  Pearson 相关    —— 线性一致性(参考)
  RMSE / MAE      —— 数值误差(需先按正负号对齐方向)

判读口径:
  |Spearman| ≥ 0.5  排序有中等以上预测力(可用作预筛排序)
  |Spearman| ≥ 0.7  排序强预测力
  |Spearman| < 0.3  排序预测力弱(该模型的分数不应单独用于排序决策)

数据来源(校准点): 需「有实验 ΔΔG 的突变体」, 如 FireProt / ProTherm / 文献。
每个点 = variant + predicted(打分工具输出) + experimental(实验 ΔΔG) + source(出处)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class CalibrationPoint:
    """一个校准点: 某突变体的预测分 vs 实验 ΔΔG。"""
    variant_id: str
    predicted: float          # 打分工具输出(如 FoldX ΔΔG)
    experimental: float       # 实验 ΔΔG(kcal/mol)
    source: str = ""          # 出处(文献/数据库, 供审计)
    model: str = ""           # 打分模型名(esm1v/foldx/...)


def _rank(vals: Sequence[float]) -> List[float]:
    """转排名(越小值 → 越小 rank), 平局取平均。"""
    order = sorted(range(len(vals)), key=lambda i: (vals[i], i))
    ranks = [0.0] * len(vals)
    i = 0
    n = len(order)
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman 秩相关。"""
    if len(a) < 2:
        return 0.0
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson 线性相关。"""
    if len(a) < 2:
        return 0.0
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(len(a)))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0:
        return 0.0
    return cov / math.sqrt(va * vb)


def calibrate(points: Sequence[CalibrationPoint]) -> Dict[str, Any]:
    """算校准指标。自动按 Pearson 符号对齐方向(预测分与实验值正相关 vs 负相关)。"""
    pts = list(points)
    if len(pts) < 3:
        return {"n": len(pts), "ok": False,
                "note": "校准点不足(需 ≥3 才有统计意义)"}
    pred = [p.predicted for p in pts]
    exp = [p.experimental for p in pts]
    rho = spearman(pred, exp)
    r = pearson(pred, exp)
    # 方向: 预测分与实验值的符号(若负相关, 说明该模型分数方向与实验相反, 需翻转)
    direction = "direct" if r >= 0 else "inverted"
    aligned_pred = pred if r >= 0 else [-x for x in pred]
    errs = [aligned_pred[i] - exp[i] for i in range(len(pts))]
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    mae = sum(abs(e) for e in errs) / len(errs)
    return {
        "n": len(pts), "ok": True,
        "spearman": round(rho, 4),
        "pearson": round(r, 4),
        "direction": direction,
        "rmse": round(rmse, 4), "mae": round(mae, 4),
        "verdict": _verdict(abs(rho)),
    }


def _verdict(abs_rho: float) -> str:
    if abs_rho >= 0.7:
        return "强预测力(可作排序决策)"
    if abs_rho >= 0.5:
        return "中等预测力(可作预筛排序)"
    if abs_rho >= 0.3:
        return "弱预测力(不建议单独用于排序)"
    return "几乎无预测力(该模型分数不可单独用于排序)"


__all__ = ["CalibrationPoint", "calibrate", "spearman", "pearson"]
