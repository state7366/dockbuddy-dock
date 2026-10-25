# -*- coding: utf-8 -*-
"""
presentation.py —— 组件结果的四层呈现(结论先行 + 自否证 + 字段分类)
================================================================================
「结果里有大量信息」的呈现不是把 JSON 打印出来, 而是让结论可追溯:
  L0 结论层  verdict + 可写/禁写(自否证)
  L1 摘要层  关键数字 mean±std
  L2 明细层  逐 pose/逐候选
  L3 原始层  全字段 JSON(单一事实源)

本模块提供两个通用抽象:
  Verdict          自否证结论(status + 可写 narratable + 禁写 forbidden + 证据 evidence)
  FieldCategory    字段分类(conclusion 结论 / evidence 证据 / proxy 代理指标 / meta 元数据)

领域特殊性: 分子对接最大的风险是「过度解读」。把"不能写什么"和"能写什么"并列,
比任何字段都更能防止站不住脚的结论。代理指标(不预测活性的打分)必须显式标注。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Verdict:
    """自否证结论: 一行可执行结论 + 禁写项 + 支撑证据。"""
    status: str = "ok"                    # ok / warn / unreliable
    narratable: Tuple[str, ...] = ()      # 可写进论文的结论
    forbidden: Tuple[str, ...] = ()       # 禁止写的结论(自否证)
    evidence: Dict[str, Any] = field(default_factory=dict)  # 支撑结论的关键数字

    def summary(self) -> str:
        """一行 verdict(终端首屏 / REPORT 开头)。"""
        parts = [f"可靠性={self.status}"]
        if self.narratable:
            parts.append("可写: " + " ｜ ".join(self.narratable))
        if self.forbidden:
            parts.append("禁写: " + " ｜ ".join(self.forbidden))
        return " ｜ ".join(parts)


# 字段分类: 结论 / 证据 / 代理指标 / 元数据
CONCLUSION = "conclusion"
EVIDENCE = "evidence"
PROXY = "proxy"
META = "meta"
FIELD_CATEGORIES = (CONCLUSION, EVIDENCE, PROXY, META)

# 常见字段分类表(单一事实源; 新字段请在此登记类别, 供呈现层过滤与"代理指标"警示)
_FIELD_CATEGORY = {
    # 结论
    "reliability": CONCLUSION, "verdict": CONCLUSION, "narratable": CONCLUSION,
    "forbidden": CONCLUSION, "warn_reasons": CONCLUSION,
    # 证据(几何/能量, 可写进论文的数字)
    "energy_kcal_mol": EVIDENCE, "nuc_to_ligO_A": EVIDENCE, "rmsd_top1_A": EVIDENCE,
    "rmsd_min_A": EVIDENCE, "pose_set_rmsd_median_A": EVIDENCE,
    "pocket_volume_A3": EVIDENCE, "delta_A3": EVIDENCE, "ca_rmsd_vs_wt_A": EVIDENCE,
    "his_asp_d_A": EVIDENCE, "bottleneck_radius_A": EVIDENCE,
    # 代理指标(打分, 不预测活性)
    "esm1v_score": PROXY, "saprot_score": PROXY, "venusrem_score": PROXY,
    "promep_score": PROXY, "evmutation_score": PROXY, "foldx_ddg": PROXY,
    "rosetta_ddg": PROXY, "cnn_score": PROXY, "cnn_affinity": PROXY,
    # 元数据(复现依据)
    "seed": META, "exhaustiveness": META, "engine": META, "box_center": META,
    "box_size": META, "schema_version": META, "run_id": META, "config": META,
}

PROXY_DISCLAIMER = "代理指标: 不预测 kcat/Km 或活性, 只作相对排序"


def field_category(name: str) -> str:
    """字段类别(未登记的缺省 meta)。"""
    return _FIELD_CATEGORY.get(name, META)


def split_fields(result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """把结果字段按类别分组(供呈现层按需过滤)。"""
    out: Dict[str, Dict[str, Any]] = {c: {} for c in FIELD_CATEGORIES}
    for k, v in result.items():
        out[field_category(k)][k] = v
    return out


def present_layered(verdict: Verdict, result: Dict[str, Any],
                    *, detail_items: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """把组件结果投影成四层 {L0, L1, L2, L3}。

    L0 = verdict 摘要; L1 = 证据字段(evidence + conclusion); L2 = 明细(逐 pose/候选);
    L3 = 全字段 JSON(单一事实源, 原样透传)。
    """
    cats = split_fields(result)
    return {
        "L0": verdict.summary(),
        "L1": {**cats[CONCLUSION], **cats[EVIDENCE]},
        "L2": detail_items if detail_items is not None else cats[PROXY],
        "L3": dict(result),
        "proxy_disclaimer": PROXY_DISCLAIMER if cats[PROXY] else "",
    }


__all__ = ["Verdict", "present_layered", "split_fields", "field_category",
           "CONCLUSION", "EVIDENCE", "PROXY", "META", "PROXY_DISCLAIMER"]
